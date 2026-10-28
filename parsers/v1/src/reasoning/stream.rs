// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The v1 reasoning stream: split each chat chunk's `content` into
//! `reasoning_content` and `content` with a [`ReasoningParser`], per choice.
//!
//! Moved from dynamo `lib/llm/src/preprocessor.rs`, where dynamo's frontend
//! runs it before the tool-call jail on the v1 parser route. The behavior is
//! unchanged. The one new seam is the stream item: the stages are generic over
//! [`ChatStreamItem`], so dynamo keeps its own item type (see
//! [`crate::chat_stream`]).
//!
//! The module also holds the guided-JSON stage that strips a leading reasoning
//! opener, and the parser-name rules that dynamo uses to choose the stages.

use std::collections::HashMap;
use std::pin::Pin;

use dynamo_protocols::types::ChatCompletionMessageContent;
use futures::{Stream, StreamExt, stream};

use crate::chat_stream::ChatStreamItem;
use crate::reasoning::{ParserResult, ReasoningParser, ReasoningParserType};
use crate::sources::{ChoiceSources, SourceChannel, SourceRun, Spans};

/// Options for [`parse_reasoning_content_from_stream`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReasoningStreamOptions {
    /// The chat template already ended the prompt with the reasoning opener
    /// (for example `<think>`), so each choice's parser starts inside the
    /// reasoning block.
    pub prompt_injected_reasoning: bool,
    /// Decide per choice, from its first non-whitespace content, whether the
    /// backend sent bare guided JSON (`[` or `{`). Such a choice is not parsed,
    /// so its content reaches the tool-call jail unchanged.
    pub bypass_bare_guided_json: bool,
    /// Hold reasoning until the choice shows real answer text, for a request
    /// that promises non-empty `content` (`force_nonempty_content`). See the
    /// note on `ReasoningState::defer_reasoning_for_nonempty_content`.
    pub defer_reasoning_for_nonempty_content: bool,
}

/// Parsers that begin streaming in reasoning mode (force_reasoning=true).
/// These swallow any leading text without an open `<think>` tag as
/// reasoning_content, so they cannot run on guided-decoding output where
/// the model emits bare JSON from token 0.
pub fn is_force_reasoning_parser(reasoning_parser: Option<&str>) -> bool {
    matches!(
        reasoning_parser,
        Some(
            "deepseek_r1"
                | "deepseek_v3"
                | "deepseek_v3_1"
                | "deepseek_v3_2"
                | "step3"
                | "kimi_k25"
                | "mistral"
                | "minimax_m2"
                | "minimax_append_think"
                | "nemotron_nano"
                | "nemotron3"
                | "nemotron_v3"
        )
    )
}

/// Force-reasoning parsers proven to receive both bare guided JSON and
/// native-reasoner-gated `reasoning</think>JSON`. These use the stream-shape
/// detector instead of the historical unconditional guided-JSON bypass.
pub fn supports_reasoning_before_guided_json(reasoning_parser: Option<&str>) -> bool {
    matches!(
        reasoning_parser,
        Some(
            "deepseek_r1"
                | "deepseek_v3"
                | "deepseek_v3_1"
                | "deepseek_v3_2"
                | "step3"
                | "kimi_k25"
                | "mistral"
                | "minimax_m2"
                | "nemotron_nano"
                | "nemotron3"
                | "nemotron_v3"
        )
    )
}

/// Reasoning openers that overlap with a valid guided-JSON prefix.
pub fn guided_json_reasoning_start_token(reasoning_parser: Option<&str>) -> Option<&'static str> {
    match reasoning_parser {
        Some("mistral") => Some("[THINK]"),
        _ => None,
    }
}

/// Parsers that, on guided output after a prompt-opened reasoning block,
/// check each choice for bare guided JSON before they parse it (see
/// [`ReasoningStreamOptions::bypass_bare_guided_json`]).
pub fn skips_guided_json_when_prompt_injected(reasoning_parser: Option<&str>) -> bool {
    matches!(
        reasoning_parser,
        Some(
            "deepseek_v4"
                | "deepseek_v41"
                | "deepseek-v4"
                | "deepseekv4"
                | "glm45"
                | "minimax_m3"
                | "minimax-m3"
        )
    )
}

/// MiniMax M3's reasoning parser recovers a reasoning block whose opener
/// is not visible (implicit tool-start recovery), so it retains
/// marker-free text until a close marker, a tool marker, or EOF decides
/// whether that text was reasoning. Its EOF decision only reaches the
/// client when the stream finalizes it.
pub fn is_minimax_m3_reasoning_parser(reasoning_parser: Option<&str>) -> bool {
    matches!(reasoning_parser, Some("minimax_m3" | "minimax-m3"))
}

/// [`skips_guided_json_when_prompt_injected`], widened for a structured
/// response (`response_format`) to Qwen3 and Kimi K3.
pub fn skips_structured_response_when_prompt_injected(reasoning_parser: Option<&str>) -> bool {
    matches!(reasoning_parser, Some("qwen3" | "kimi_k3" | "kimi-k3"))
        || skips_guided_json_when_prompt_injected(reasoning_parser)
}

/// Whether a structured response may use SGLang's reasoning gate
/// (`require_reasoning` with `json_schema`) for this reasoning parser.
pub fn structured_response_supports_sglang_reasoning_gate(reasoning_parser: Option<&str>) -> bool {
    // GPT-OSS/Harmony must skip SGLang's `require_reasoning + json_schema`
    // path for structured output until upstream fixes malformed Harmony:
    // https://github.com/sgl-project/sglang/issues/31019
    // Tool calling still uses `require_reasoning`.
    !matches!(reasoning_parser, Some("gpt_oss"))
}

// Reasoning State for reasoning parsing transformation step.
//
// The reasoning parser and the guided-JSON bypass decision are kept per
// `choice.index` so that with `n > 1` one choice's bare-JSON bypass cannot
// suppress another choice's reasoning split. This mirrors the per-choice state
// already used by the tool-call jail and the leading-`<think>` strip stage.
struct ChoiceReasoningState {
    parser: Box<dyn ReasoningParser>,
    guided_json_bypass_decision: Option<bool>,
    // Reasoning text held back while it is still ambiguous. Only used on the
    // `defer_reasoning_for_nonempty_content` path — see the buffering note on
    // `ReasoningState`.
    pending_reasoning: String,
    // Whitespace-only normal text seen while still ambiguous. Held rather than
    // emitted so the streaming path agrees with the aggregator, which counts
    // whitespace-only content as empty.
    pending_content: String,
    // Set once this choice's parser has produced normal text, which for these
    // force-reasoning parsers means it left the reasoning block. From then on
    // the ambiguity is gone and deltas pass straight through.
    left_reasoning: bool,
    // Set once this choice has been drained, so the terminal-chunk drain and the
    // end-of-stream fallback cannot both emit the same bytes. Reopened if the
    // backend keeps sending content after `finish_reason`, so those bytes still
    // get a destination.
    drained: bool,
    // Set the first time this choice's parser is finalized and NEVER reset, even
    // when `drained` is reopened. `ReasoningParser::finish_reasoning_stream` is
    // not documented to be idempotent, so calling it twice could re-emit text a
    // parser had already handed over. Draining again is fine; finalizing again
    // is not.
    parser_finished: bool,
    // While the stream records sources: where this choice's text came from.
    // `None` when it does not record, and on the stock path.
    sources: Option<Box<ChoiceStreamSources>>,
    // The stream records, but this choice's text has no source: its parser
    // cannot record, or an input chunk gave no ranges for its text. Its
    // output gets no source entry, so the caller can tell.
    unsourced: bool,
}

/// The sources of one choice's text, while its stream records.
struct ChoiceStreamSources {
    // The caller's ranges of the bytes fed to the parser, from the parser's
    // own offset `fed_base` on. Earlier bytes are never read again.
    fed: Spans,
    fed_base: usize,
    // Beside `pending_reasoning` and `pending_content`.
    pending_reasoning: Spans,
    pending_content: Spans,
    runs: Vec<SourceRun>,
}

impl ChoiceStreamSources {
    fn new() -> Self {
        Self {
            fed: Spans::recording(true),
            fed_base: 0,
            pending_reasoning: Spans::recording(true),
            pending_content: Spans::recording(true),
            runs: Vec::new(),
        }
    }

    /// The caller's ranges of what the parser returned since the last call,
    /// as `(normal, reasoning)`.
    fn take(&mut self, parser: &mut dyn ReasoningParser) -> (Spans, Spans) {
        self.runs.clear();
        parser.take_sources(&mut self.runs);
        let mut normal = Spans::recording(true);
        let mut reasoning = Spans::recording(true);
        let mut end = self.fed_base;
        for run in &self.runs {
            debug_assert!(
                run.input.start >= self.fed_base,
                "a run before the forgotten input"
            );
            let at = run.input.start - self.fed_base..run.input.end - self.fed_base;
            let spans = self.fed.slice(at);
            match run.channel {
                SourceChannel::Normal => normal.append(&spans),
                SourceChannel::Reasoning => reasoning.append(&spans),
            }
            end = end.max(run.input.end);
        }
        // The parser's runs never go back in its input.
        self.fed.drain_front(end - self.fed_base);
        self.fed_base = end;
        (normal, reasoning)
    }
}

/// The caller's ranges of what a choice's parser returned since the last
/// call, as `(normal, reasoning)`; [`Spans::OFF`] while not recording.
fn take_runs(state: &mut ChoiceReasoningState) -> (Spans, Spans) {
    match state.sources.as_mut() {
        Some(sources) => sources.take(&mut *state.parser),
        None => (Spans::OFF, Spans::OFF),
    }
}

/// Text that a drain releases, `(content, reasoning)`, with their sources.
struct Released {
    content: Option<String>,
    reasoning: Option<String>,
    content_spans: Spans,
    reasoning_spans: Spans,
}

impl Released {
    fn none() -> Self {
        Self {
            content: None,
            reasoning: None,
            content_spans: Spans::OFF,
            reasoning_spans: Spans::OFF,
        }
    }
}

/// Drain what a choice still holds on the `defer_reasoning_for_nonempty_content`
/// path, returning `(content, reasoning_content)` to add to its delta. Shared by
/// the terminal-chunk drain and the end-of-stream fallback so the two cannot
/// disagree about which channel the leftover bytes belong to.
fn drain_deferred_reasoning(state: &mut ChoiceReasoningState) -> Released {
    if state.drained {
        return Released::none();
    }
    state.drained = true;
    let pending = std::mem::take(&mut state.pending_reasoning);
    let pending_content = std::mem::take(&mut state.pending_content);
    let (mut pending_spans, mut pending_content_spans) = match state.sources.as_mut() {
        Some(sources) => (
            sources.pending_reasoning.take(),
            sources.pending_content.take(),
        ),
        None => (Spans::OFF, Spans::OFF),
    };
    // Finalize at most once per choice; see `parser_finished`.
    let result = if state.parser_finished {
        ParserResult::default()
    } else {
        state.parser_finished = true;
        state.parser.finish_reasoning_stream()
    };
    let (normal_spans, reasoning_spans) = take_runs(state);

    if state.left_reasoning {
        // An answer already streamed as content, so the non-empty-content
        // promise is met and the parser's own split stands: a later truncated
        // reasoning block stays reasoning instead of being appended to the
        // answer. `pending_content` is empty here — it was released with the
        // answer when the choice left reasoning.
        let mut content = pending_content;
        content.push_str(&result.normal_text);
        pending_content_spans.append(&normal_spans);
        Released {
            content: (!content.is_empty()).then_some(content),
            reasoning: (!result.reasoning_text.is_empty()).then_some(result.reasoning_text),
            content_spans: pending_content_spans,
            reasoning_spans,
        }
    } else {
        // No real answer ever arrived, so everything held is all the response
        // has and must land in content — including a truncated `<think>` prefix,
        // which the parser reports as reasoning or normal text depending on
        // where it stopped. Held whitespace is dropped rather than prepended when
        // there is reasoning to surface, matching the aggregator, which replaces
        // whitespace-only content with the reasoning text instead of
        // concatenating.
        let mut text = pending;
        text.push_str(&result.reasoning_text);
        text.push_str(&result.normal_text);
        pending_spans.append(&reasoning_spans);
        pending_spans.append(&normal_spans);
        if text.is_empty() {
            // ...but if whitespace is genuinely all the turn produced, it is the
            // whole response. Returning nothing here would hand back an empty
            // `content` to a request that explicitly asked for non-empty content.
            text = pending_content;
            pending_spans = pending_content_spans;
        }
        let reasoning_spans = Spans::recording(pending_spans.is_recording());
        Released {
            content: (!text.is_empty()).then_some(text),
            reasoning: None,
            content_spans: pending_spans,
            reasoning_spans,
        }
    }
}

/// Finalize a choice's parser without the deferral, returning `(content,
/// reasoning_content)` by the parser's own split: held marker-free text is
/// content, a held partial marker inside reasoning stays reasoning. Shared by
/// the terminal-chunk finalize and the end-of-stream fallback so the two
/// cannot disagree. Finalizes at most once per choice; see `parser_finished`.
fn finish_reasoning_parser(state: &mut ChoiceReasoningState) -> Released {
    if state.parser_finished {
        return Released::none();
    }
    state.parser_finished = true;
    let result = state.parser.finish_reasoning_stream();
    let (content_spans, reasoning_spans) = take_runs(state);
    Released {
        content: (!result.normal_text.is_empty()).then_some(result.normal_text),
        reasoning: (!result.reasoning_text.is_empty()).then_some(result.reasoning_text),
        content_spans,
        reasoning_spans,
    }
}

struct ReasoningState<T> {
    stream: Pin<Box<dyn Stream<Item = T> + Send>>,
    parser_name: String,
    prompt_injected_reasoning: bool,
    bypass_bare_guided_json: bool,
    choices: HashMap<u32, ChoiceReasoningState>,
    // An emitted response with choices, reused as the envelope to carry any
    // text the parsers are still buffering when the upstream stream ends. Only
    // retained when an end-of-stream flush may follow, to keep the per-token
    // clone off the common reasoning hot path: under
    // `defer_reasoning_for_nonempty_content` it is the latest such response;
    // under `finalize_parser_at_finish` alone it is cloned once, while a parsed
    // choice's parser is unfinished. Chunks with no choices (the trailing
    // usage-only chunk) are never retained — that envelope has no delta slot
    // to attach the flushed bytes to, so the flush would be dropped.
    last_response: Option<T>,
    // Nemotron force-reasoning parsers start inside the reasoning block, so
    // leading model output with no `<think>` is reported as reasoning even when
    // it is really the whole answer. Under `force_nonempty_content=true` the
    // chat template promises non-empty `content`, so emitting that text as
    // `reasoning_content` would leave `content` empty — the exact failure the
    // flag exists to prevent.
    //
    // When this is set, reasoning deltas are held in each choice's
    // `pending_reasoning` until the ambiguity resolves:
    //   - the parser produces normal text (it left the reasoning block), so the
    //     buffer really was reasoning: emit it as `reasoning_content` and stream
    //     the answer as `content` from then on; or
    //   - the stream ends first, so no answer is coming: emit the buffer (plus
    //     anything `finish_reasoning_stream` still holds, e.g. a truncated
    //     `<think>` prefix) as `content`.
    //
    // The cost is real and worth stating plainly: everything held is delivered
    // in one delta rather than token by token. For a turn that does reason, that
    // is the `reasoning_content` up to `</think>`, after which the answer streams
    // normally. For a turn with no `</think>` at all the held text IS the answer,
    // so the whole answer lands in a single delta on the terminal chunk — on
    // `main` (parser off, leading `<think>` stripped) that case streamed
    // incrementally.
    //
    // There is no cheaper split available: these parsers emit reasoning with no
    // opening `<think>` (see the `nemotron_nano` mapping in dynamo-parsers), so
    // until `</think>` arrives the same bytes are equally consistent with
    // reasoning and with a plain answer. Emitting eagerly is what leaked reasoning
    // into `content` in the first place. Scoped to the flag so no other parser
    // pays it.
    defer_reasoning_for_nonempty_content: bool,
    // Finalize each choice's parser on its terminal chunk even without the
    // deferral above, for parsers that retain undecided text until EOF
    // (MiniMax M3), and at end of stream for a choice whose parser is still
    // unfinished: it sent no `finish_reason`, or its terminal chunk carried
    // content parts. Otherwise a turn whose output never shows a marker, such
    // as grammar-constrained output under an adaptive prompt, is dropped
    // whole with the parser state. See `is_minimax_m3_reasoning_parser`.
    finalize_parser_at_finish: bool,
    // Set once the backend has sent an error annotation. The end-of-stream
    // flush must not run after that: an error is terminal, and emitting
    // recovered reasoning as ordinary `content` after it would put text on the
    // wire that the generation never successfully produced.
    //
    // This matters because the flush is the only place bytes can arrive after
    // the last upstream chunk. `/v1/chat/completions` ends its SSE on the error
    // and never sees it, but `/v1/responses` and `/v1/messages` both record the
    // error and keep consuming (`saw_error = true; continue;`), so without this
    // latch they emit the synthetic content chunk and only then report failure.
    saw_terminal_error: bool,
}

// Motivation: Each transformation on the stream should be a separate step to allow for more flexibility
// Earlier reasoning parser logic was nested under delta generation logic in choice_from_postprocessor
// Since we have tool calling parsing as separate step, it makes sense to have reasoning parser as separate step as well
/// Apply reasoning parsing to the output stream, splitting content into
/// `reasoning_content` and normal `content` based on think tags.
///
/// When `options.prompt_injected_reasoning` is `true`, the parser starts in
/// reasoning mode immediately — use this when the chat template already
/// appended the reasoning start token (e.g., `<think>`) to the prompt, so the
/// model's completion begins with thinking content without an explicit start
/// tag.
///
/// An unknown `parser_name` falls back to the basic `<think>` parser, as
/// [`ReasoningParserType::get_reasoning_parser_from_name`] does.
pub fn parse_reasoning_content_from_stream<T, S>(
    stream: S,
    parser_name: String,
    options: ReasoningStreamOptions,
) -> impl Stream<Item = T> + Send
where
    T: ChatStreamItem,
    S: Stream<Item = T> + Send + 'static,
{
    let ReasoningStreamOptions {
        prompt_injected_reasoning,
        bypass_bare_guided_json,
        defer_reasoning_for_nonempty_content,
    } = options;
    // Parsers and bypass decisions are created lazily per `choice.index`
    // inside the unfold loop, so `n > 1` choices never share state.
    let finalize_parser_at_finish = is_minimax_m3_reasoning_parser(Some(parser_name.as_str()));
    let state = ReasoningState {
        stream: Box::pin(stream),
        parser_name,
        prompt_injected_reasoning,
        bypass_bare_guided_json,
        choices: HashMap::new(),
        last_response: None,
        defer_reasoning_for_nonempty_content,
        finalize_parser_at_finish,
        saw_terminal_error: false,
    };

    stream::unfold(state, |mut state| async move {
        if let Some(mut response) = state.stream.next().await {
            // An error is terminal for the flush: latch it so the
            // end-of-stream branch below stays quiet. The chunk still takes
            // the normal path — `is_error()` keys on the annotation event,
            // not on `data`, so short-circuiting here would change how a
            // data-carrying error chunk is processed.
            if response.is_error() {
                state.saw_terminal_error = true;
            }
            // Split disjoint field borrows so the per-choice map and the
            // parser-factory inputs can be used together while the chunk is
            // rewritten in place.
            // Scoped in a block so the borrows end before `state` moves.
            let processed_response = {
                let ReasoningState {
                    parser_name,
                    prompt_injected_reasoning,
                    bypass_bare_guided_json,
                    choices,
                    defer_reasoning_for_nonempty_content,
                    finalize_parser_at_finish,
                    ..
                } = &mut state;
                let parser_name = &*parser_name;
                let prompt_injected_reasoning = *prompt_injected_reasoning;
                let bypass_bare_guided_json = *bypass_bare_guided_json;
                let defer_reasoning = *defer_reasoning_for_nonempty_content;
                let finalize_parser_at_finish = *finalize_parser_at_finish;

                // While recording, the input slot gives each choice's ranges and
                // the output slot gets the ranges of the text this stage emits.
                let mut input_sources = response.sources_mut().map(std::mem::take);
                let mut output_sources = input_sources.as_ref().map(|_| Vec::new());
                let recording = output_sources.is_some();
                if let Some(data) = response.chunk_mut() {
                    for choice in data.choices.iter_mut() {
                        let given = input_sources.as_mut().and_then(|all| {
                            let at = all.iter().position(|given| given.index == choice.index)?;
                            Some(all.remove(at))
                        });
                        let choice_state = choices.entry(choice.index).or_insert_with(|| {
                            let mut parser = Box::new(
                                ReasoningParserType::get_reasoning_parser_from_name(parser_name),
                            )
                                as Box<dyn ReasoningParser>;
                            if prompt_injected_reasoning {
                                parser.set_in_reasoning(true);
                            }
                            let sources = (recording && parser.record_sources())
                                .then(|| Box::new(ChoiceStreamSources::new()));
                            ChoiceReasoningState {
                                parser,
                                guided_json_bypass_decision: (!bypass_bare_guided_json)
                                    .then_some(false),
                                pending_reasoning: String::new(),
                                pending_content: String::new(),
                                left_reasoning: false,
                                drained: false,
                                parser_finished: false,
                                unsourced: recording && sources.is_none(),
                                sources,
                            }
                        });
                        // A choice that is not parsed keeps its text, and its
                        // ranges. A choice with no text needs no ranges.
                        let has_text = matches!(
                            &choice.delta.content,
                            Some(ChatCompletionMessageContent::Text(text)) if !text.is_empty()
                        ) || choice
                            .delta
                            .reasoning_content
                            .as_ref()
                            .is_some_and(|reasoning| !reasoning.is_empty());
                        let known = given.is_some() || !has_text;
                        let (mut content_spans, mut reasoning_spans) = match &given {
                            Some(given) => (spans_of(&given.content), spans_of(&given.reasoning)),
                            None => (Spans::recording(recording), Spans::recording(recording)),
                        };

                        // Decide once per choice, from ITS OWN first
                        // non-whitespace content, whether the backend
                        // emitted bare guided JSON (`[`/`{`) that must reach
                        // the tool jail unparsed.
                        let bypass_decision = if bypass_bare_guided_json {
                            match choice_state.guided_json_bypass_decision {
                                Some(decision) => Some(decision),
                                None => {
                                    let decision = match choice.delta.content.as_ref() {
                                        Some(ChatCompletionMessageContent::Text(text)) => {
                                            let text = text.trim_start();
                                            if text.is_empty() {
                                                None
                                            } else {
                                                Some(matches!(text.as_bytes()[0], b'[' | b'{'))
                                            }
                                        }
                                        _ => None,
                                    };
                                    if let Some(decision) = decision {
                                        choice_state.guided_json_bypass_decision = Some(decision);
                                    }
                                    decision
                                }
                            }
                        } else {
                            Some(false)
                        };

                        // Only a choice decided NOT to bypass is parsed. A
                        // bare-JSON or still-undecided (whitespace-only)
                        // choice keeps its content untouched for the jail.
                        // Reasoning parsing only applies to text content;
                        // multimodal content passes through unchanged.
                        if bypass_decision == Some(false)
                            && let Some(ChatCompletionMessageContent::Text(text)) =
                                choice.delta.content.as_ref()
                        {
                            if let Some(sources) = choice_state.sources.as_mut() {
                                match &given {
                                    Some(given) => {
                                        let fed = spans_of(&given.content);
                                        debug_assert_eq!(
                                            fed.len(),
                                            text.len(),
                                            "input ranges must tile the content"
                                        );
                                        sources.fed.append(&fed);
                                    }
                                    None if text.is_empty() => {}
                                    None => {
                                        // Bytes with no source make every later
                                        // offset of this choice unknown.
                                        choice_state.sources = None;
                                        choice_state.unsourced = true;
                                    }
                                }
                            }
                            let parser_result = choice_state
                                .parser
                                .parse_reasoning_streaming_incremental(text, &[]);
                            let (normal_spans, parsed_reasoning_spans) = take_runs(choice_state);

                            // A backend that keeps sending content after this
                            // choice's `finish_reason` is out of protocol, but
                            // it has still fed bytes to the parser. Reopen the
                            // drain so they have somewhere to go at EOF instead
                            // of being stranded in parser state.
                            choice_state.drained = false;

                            if defer_reasoning && !choice_state.left_reasoning {
                                // Still ambiguous: this text may be reasoning
                                // followed by an answer, or the answer itself
                                // reported as reasoning because the parser
                                // starts inside the reasoning block. Hold it.
                                choice_state
                                    .pending_reasoning
                                    .push_str(&parser_result.reasoning_text);
                                if let Some(sources) = choice_state.sources.as_mut() {
                                    sources.pending_reasoning.append(&parsed_reasoning_spans);
                                }
                                // Whitespace-only normal text does not settle
                                // anything: the aggregator treats such content
                                // as empty (matching vLLM's
                                // `not final_content.strip()`), so releasing
                                // here would make streaming and non-streaming
                                // disagree on the same turn. Hold it with the
                                // reasoning until real answer text arrives.
                                if parser_result.normal_text.trim().is_empty() {
                                    choice_state
                                        .pending_content
                                        .push_str(&parser_result.normal_text);
                                    if let Some(sources) = choice_state.sources.as_mut() {
                                        sources.pending_content.append(&normal_spans);
                                    }
                                    choice.delta.content = None;
                                    choice.delta.reasoning_content = None;
                                    content_spans = Spans::recording(recording);
                                    reasoning_spans = Spans::recording(recording);
                                } else {
                                    // Real answer text means the parser left
                                    // the reasoning block, so everything held
                                    // so far really was reasoning.
                                    choice_state.left_reasoning = true;
                                    choice.delta.reasoning_content = (!choice_state
                                        .pending_reasoning
                                        .is_empty())
                                    .then(|| std::mem::take(&mut choice_state.pending_reasoning));
                                    let mut text =
                                        std::mem::take(&mut choice_state.pending_content);
                                    text.push_str(&parser_result.normal_text);
                                    choice.delta.content =
                                        Some(ChatCompletionMessageContent::Text(text));
                                    match choice_state.sources.as_mut() {
                                        Some(sources) => {
                                            reasoning_spans = sources.pending_reasoning.take();
                                            content_spans = sources.pending_content.take();
                                            content_spans.append(&normal_spans);
                                        }
                                        None => {
                                            reasoning_spans = Spans::OFF;
                                            content_spans = Spans::OFF;
                                        }
                                    }
                                }
                            } else {
                                choice.delta.content = parser_result
                                    .get_some_normal_text()
                                    .map(ChatCompletionMessageContent::Text);
                                choice.delta.reasoning_content = parser_result.get_some_reasoning();
                                content_spans = normal_spans;
                                reasoning_spans = parsed_reasoning_spans;
                            }
                        }

                        // This choice is finishing, so drain what it holds
                        // onto THIS delta. Emitting it as a later chunk
                        // would put content after `finish_reason` (and
                        // after the trailing usage chunk), where a client
                        // that stops at the terminal chunk never sees it.
                        // A parts delta has no text slot to append to, so
                        // draining onto it would leave the recovered text
                        // nowhere to go. Leave the choice undrained and let
                        // the end-of-stream fallback emit it as its own
                        // chunk instead of dropping it.
                        let terminal_carries_parts = matches!(
                            choice.delta.content,
                            Some(ChatCompletionMessageContent::Parts(_))
                        );
                        // Also require that this choice was actually parsed.
                        // A guided-JSON choice that bypassed the parser never
                        // fed it anything, so finishing that parser could only
                        // contribute text the choice never generated.
                        if defer_reasoning
                            && choice.finish_reason.is_some()
                            && !terminal_carries_parts
                            && bypass_decision == Some(false)
                        {
                            let released = drain_deferred_reasoning(choice_state);
                            content_spans.append(&released.content_spans);
                            reasoning_spans.append(&released.reasoning_spans);
                            let (content, reasoning) = (released.content, released.reasoning);
                            if let Some(content) = content {
                                let merged = match choice.delta.content.take() {
                                    Some(ChatCompletionMessageContent::Text(existing)) => {
                                        existing + &content
                                    }
                                    Some(other) => {
                                        // Unreachable: guarded above.
                                        choice.delta.content = Some(other);
                                        content
                                    }
                                    None => content,
                                };
                                if choice.delta.content.is_none() {
                                    choice.delta.content =
                                        Some(ChatCompletionMessageContent::Text(merged));
                                }
                            }
                            if let Some(reasoning) = reasoning {
                                choice.delta.reasoning_content = Some(
                                    choice.delta.reasoning_content.take().unwrap_or_default()
                                        + &reasoning,
                                );
                            }
                        }

                        // Without the deferral, a parser that retains
                        // undecided text until EOF is finalized here, on
                        // the same terminal delta and under the same
                        // conditions; see `finish_reasoning_parser`.
                        if !defer_reasoning
                            && finalize_parser_at_finish
                            && choice.finish_reason.is_some()
                            && !terminal_carries_parts
                            && bypass_decision == Some(false)
                        {
                            let released = finish_reasoning_parser(choice_state);
                            content_spans.append(&released.content_spans);
                            reasoning_spans.append(&released.reasoning_spans);
                            let (content, reasoning) = (released.content, released.reasoning);
                            if let Some(content) = content {
                                let content = match choice.delta.content.take() {
                                    Some(ChatCompletionMessageContent::Text(existing)) => {
                                        existing + &content
                                    }
                                    // `None`; parts were excluded above.
                                    _ => content,
                                };
                                choice.delta.content =
                                    Some(ChatCompletionMessageContent::Text(content));
                            }
                            if let Some(reasoning) = reasoning {
                                choice.delta.reasoning_content = Some(
                                    choice.delta.reasoning_content.take().unwrap_or_default()
                                        + &reasoning,
                                );
                            }
                        }

                        if let Some(output) = output_sources.as_mut()
                            && !choice_state.unsourced
                            && known
                        {
                            output.push(ChoiceSources {
                                index: choice.index,
                                content: content_spans.into_ranges(),
                                reasoning: reasoning_spans.into_ranges(),
                            });
                        }
                    }
                }
                if let Some(output) = output_sources
                    && let Some(slot) = response.sources_mut()
                {
                    *slot = output;
                }
                response
            };

            // Retain a spare envelope only when an EOF flush may follow, so
            // the common reasoning path avoids a full per-token clone: the
            // latest chunk under the deferral, or, without it, one chunk
            // taken while a parsed choice still has a parser to finalize (a
            // bypassed choice never fed its parser, and a finished one has
            // nothing left). One is enough there: the flush rebuilds every
            // choice field and scrubs the per-chunk metadata, and the rest
            // of the envelope is the same for the whole response. Skip
            // chunks with no choices (the trailing usage-only chunk): they
            // carry no delta slot, so using one as the flush envelope would
            // drop the very bytes the flush exists to preserve.
            let retain_envelope = state.defer_reasoning_for_nonempty_content
                || (state.finalize_parser_at_finish
                    && state.last_response.is_none()
                    && state.choices.values().any(|choice_state| {
                        choice_state.guided_json_bypass_decision == Some(false)
                            && !choice_state.parser_finished
                    }));
            if retain_envelope
                && processed_response
                    .chunk()
                    .is_some_and(|data| !data.choices.is_empty())
            {
                state.last_response = Some(processed_response.clone());
            }
            Some((processed_response, state))
        } else if state.saw_terminal_error
            || !(state.defer_reasoning_for_nonempty_content || state.finalize_parser_at_finish)
        {
            // After a backend error the buffered bytes are dropped rather
            // than surfaced: the request failed, so there is no answer to
            // complete. See `saw_terminal_error`.
            None
        } else {
            // Upstream ended with no answer for the choices still holding a
            // pending buffer or an unfinished parser. Normally the terminal
            // chunk already drained or finalized them, so this is the
            // fallback for a stream that ends without any `finish_reason` —
            // an aborted or truncated generation. The synthetic chunk it
            // builds is the only case where recovered bytes arrive after the
            // last upstream chunk, which is unavoidable when there was no
            // terminal chunk to attach them to. Only the
            // force_nonempty_content path and a parser finalized at finish
            // (MiniMax M3) reach this branch; every other parser took the
            // `None` branch above and keeps its original no-flush EOF
            // behavior. Taking the envelope below rather than cloning it
            // makes this branch one-shot: once it is gone the next poll ends
            // the stream.
            // Sorted so the emitted choice order is deterministic rather
            // than following HashMap iteration order.
            let defer_reasoning = state.defer_reasoning_for_nonempty_content;
            let mut indices: Vec<u32> = state.choices.keys().copied().collect();
            indices.sort_unstable();
            #[allow(clippy::type_complexity)]
            let flushed: Vec<(
                u32,
                Option<String>,
                Option<String>,
                Option<ChoiceSources>,
            )> = indices
                .into_iter()
                .filter_map(|index| {
                    let choice_state = state.choices.get_mut(&index)?;
                    if choice_state.guided_json_bypass_decision != Some(false) {
                        return None;
                    }
                    let released = if defer_reasoning {
                        drain_deferred_reasoning(choice_state)
                    } else {
                        finish_reasoning_parser(choice_state)
                    };
                    let sources = (!choice_state.unsourced).then(|| ChoiceSources {
                        index,
                        content: released.content_spans.into_ranges(),
                        reasoning: released.reasoning_spans.into_ranges(),
                    });
                    (released.content.is_some() || released.reasoning.is_some()).then_some((
                        index,
                        released.content,
                        released.reasoning,
                        sources,
                    ))
                })
                .collect();
            if flushed.is_empty() {
                return None;
            }
            let mut response = state.last_response.take()?;
            // See `scrub_synthetic_chunk_metadata`: this chunk produced no
            // tokens, so every per-chunk field from the envelope it was
            // cloned from has to be dropped rather than reported twice.
            response.scrub_synthetic();
            if let Some(slot) = response.sources_mut() {
                *slot = flushed
                    .iter()
                    .filter_map(|(_, _, _, sources)| sources.clone())
                    .collect();
            }
            let data = response.chunk_mut()?;
            // Rebuild the choice list from the flushed indices rather than
            // reusing the envelope's own choices: with `n > 1` the last
            // content-bearing chunk carries only the choices that happened
            // to be interleaved into it, which need not be the set that has
            // buffered bytes.
            let template = data.choices.first()?.clone();
            data.choices = flushed
                .into_iter()
                .map(|(index, content, reasoning, _)| {
                    let mut choice = template.clone();
                    choice.index = index;
                    choice.delta.role = None;
                    choice.delta.tool_calls = None;
                    choice.delta.function_call = None;
                    choice.delta.refusal = None;
                    choice.finish_reason = None;
                    choice.logprobs = None;
                    choice.delta.content = content.map(ChatCompletionMessageContent::Text);
                    choice.delta.reasoning_content = reasoning;
                    choice
                })
                .collect();
            Some((response, state))
        }
    })
    .fuse()
}

/// [`Spans`] of the caller's ranges.
fn spans_of(ranges: &[std::ops::Range<usize>]) -> Spans {
    let mut spans = Spans::recording(true);
    for range in ranges {
        spans.push(range.clone());
    }
    spans
}

// Motivation: when Nemotron reasoning is disabled by request flags, the
// backend may still emit a leading <think>. Buffer the initial stream
// bytes so split chunks like "<thi" + "nk>answer" are stripped cleanly.
/// Strip a leading reasoning opener (`think_start_token`) from each choice's
/// first content, after optional whitespace.
///
/// dynamo uses it for two cases: a force-reasoning parser whose reasoning the
/// request turned off (`<think>`), and guided JSON whose reasoning opener can
/// look like JSON (`[THINK]`, see [`guided_json_reasoning_start_token`]).
/// Text that does not start with the opener passes through unchanged. A
/// choice's undecided prefix is flushed as content at the end of the stream.
pub fn strip_leading_reasoning_start_from_stream<T, S>(
    stream: S,
    think_start_token: &'static str,
) -> impl Stream<Item = T> + Send
where
    T: ChatStreamItem,
    S: Stream<Item = T> + Send + 'static,
{
    struct StripReasoningStartState<T> {
        stream: Pin<Box<dyn Stream<Item = T> + Send>>,
        think_start_token: &'static str,
        choices: HashMap<u32, StripChoiceState>,
        last_response: Option<T>,
        eof_flushed: bool,
    }

    #[derive(Default)]
    struct StripChoiceState {
        buffer: String,
        decided: bool,
        // While recording: the sources of `buffer`. `unsourced` once text
        // with no source was held.
        buffer_spans: Spans,
        unsourced: bool,
    }

    fn take_undecided_buffer(choice_state: &mut StripChoiceState) -> Option<(String, Spans)> {
        if choice_state.decided || choice_state.buffer.is_empty() {
            return None;
        }

        choice_state.decided = true;
        let spans = choice_state.buffer_spans.take();
        Some((std::mem::take(&mut choice_state.buffer), spans))
    }

    #[allow(clippy::type_complexity)]
    fn drain_undecided_buffers(
        choices: &mut HashMap<u32, StripChoiceState>,
    ) -> HashMap<u32, (String, Spans, bool)> {
        choices
            .iter_mut()
            .filter_map(|(index, choice_state)| {
                let unsourced = choice_state.unsourced;
                take_undecided_buffer(choice_state)
                    .map(|(buffer, spans)| (*index, (buffer, spans, unsourced)))
            })
            .collect()
    }

    let state = StripReasoningStartState {
        stream: Box::pin(stream),
        think_start_token,
        choices: HashMap::new(),
        last_response: None,
        eof_flushed: false,
    };

    stream::unfold(state, |mut state| async move {
        if let Some(mut response) = state.stream.next().await {
            // While recording, the input slot gives each choice's ranges and
            // the output slot gets the ranges of the text this stage emits.
            let mut input_sources = response.sources_mut().map(std::mem::take);
            let mut output_sources: Option<Vec<ChoiceSources>> =
                input_sources.as_ref().map(|_| Vec::new());
            let recording = output_sources.is_some();
            let Some(data) = response.chunk_mut() else {
                if let Some(slot) = response.sources_mut() {
                    *slot = input_sources.unwrap_or_default();
                }
                return Some((response, state));
            };

            for choice in data.choices.iter_mut() {
                let choice_state =
                    state
                        .choices
                        .entry(choice.index)
                        .or_insert_with(|| StripChoiceState {
                            buffer_spans: Spans::recording(recording),
                            ..StripChoiceState::default()
                        });
                let given = input_sources.as_mut().and_then(|all| {
                    let at = all.iter().position(|given| given.index == choice.index)?;
                    Some(all.remove(at))
                });
                let content_len = match &choice.delta.content {
                    Some(ChatCompletionMessageContent::Text(text)) => text.len(),
                    _ => 0,
                };
                let reasoning_len = choice
                    .delta
                    .reasoning_content
                    .as_ref()
                    .map_or(0, String::len);
                if given.is_none() && content_len > 0 {
                    choice_state.unsourced = true;
                }
                let content_in = given.as_ref().map_or(Spans::recording(recording), |given| {
                    spans_of(&given.content)
                });
                // This stage never touches reasoning.
                let reasoning = match &given {
                    Some(given) => Some(given.reasoning.clone()),
                    None if reasoning_len == 0 => Some(Vec::new()),
                    None => None,
                };
                let mut push_entry =
                    |choice_state: &StripChoiceState, index: u32, content: Spans| {
                        if let Some(output) = output_sources.as_mut()
                            && !choice_state.unsourced
                            && let Some(reasoning) = reasoning.clone()
                        {
                            output.push(ChoiceSources {
                                index,
                                content: content.into_ranges(),
                                reasoning,
                            });
                        }
                    };
                let text = match choice.delta.content.take() {
                    Some(ChatCompletionMessageContent::Text(text)) => text,
                    other => {
                        if let Some((buffer, spans)) = take_undecided_buffer(choice_state) {
                            choice.delta.content = Some(ChatCompletionMessageContent::Text(buffer));
                            push_entry(choice_state, choice.index, spans);
                        } else {
                            choice.delta.content = other;
                            push_entry(choice_state, choice.index, content_in);
                        }
                        continue;
                    }
                };

                let (output, output_spans) = if choice_state.decided {
                    (text, content_in)
                } else {
                    choice_state.buffer.push_str(&text);
                    choice_state.buffer_spans.append(&content_in);
                    let trimmed = choice_state.buffer.trim_start();
                    if trimmed.is_empty()
                        || (state.think_start_token.starts_with(trimmed)
                            && trimmed.len() < state.think_start_token.len())
                    {
                        choice.delta.content = None;
                        push_entry(choice_state, choice.index, Spans::recording(recording));
                        continue;
                    }

                    choice_state.decided = true;
                    let all = choice_state.buffer.len();
                    if let Some(remainder) = trimmed.strip_prefix(state.think_start_token) {
                        let spans = choice_state.buffer_spans.slice(all - remainder.len()..all);
                        (remainder.to_string(), spans)
                    } else {
                        (
                            choice_state.buffer.clone(),
                            choice_state.buffer_spans.slice(0..all),
                        )
                    }
                };

                choice_state.buffer.clear();
                choice_state.buffer_spans.clear();
                choice.delta.content = if output.is_empty() {
                    None
                } else {
                    Some(ChatCompletionMessageContent::Text(output))
                };
                push_entry(choice_state, choice.index, output_spans);
            }
            if let Some(output) = output_sources
                && let Some(slot) = response.sources_mut()
            {
                *slot = output;
            }

            if response
                .chunk()
                .is_some_and(|data| !data.choices.is_empty())
            {
                state.last_response = Some(response.clone());
            }

            Some((response, state))
        } else if state.eof_flushed {
            None
        } else {
            state.eof_flushed = true;
            let flushed = drain_undecided_buffers(&mut state.choices);
            if flushed.is_empty() {
                None
            } else {
                let mut response = state.last_response.clone()?;
                // Same envelope problem as the reasoning-stream flush: this
                // is a clone of an already-counted chunk, so it goes through
                // the shared scrub rather than repeating a partial copy of
                // it here.
                response.scrub_synthetic();
                let data = response.chunk_mut()?;
                let mut template = data.choices.first()?.clone();
                template.delta.role = None;
                template.delta.tool_calls = None;
                template.delta.function_call = None;
                template.delta.refusal = None;
                template.delta.reasoning_content = None;
                template.finish_reason = None;
                template.logprobs = None;
                let mut flushed: Vec<_> = flushed.into_iter().collect();
                flushed.sort_unstable_by_key(|(index, _)| *index);
                let sources: Vec<ChoiceSources> = flushed
                    .iter()
                    .filter(|(_, (_, spans, unsourced))| spans.is_recording() && !unsourced)
                    .map(|(index, (_, spans, _))| ChoiceSources {
                        index: *index,
                        content: spans.ranges().to_vec(),
                        reasoning: Vec::new(),
                    })
                    .collect();
                data.choices = flushed
                    .into_iter()
                    .map(|(index, (buffer, _, _))| {
                        let mut choice = template.clone();
                        choice.index = index;
                        choice.delta.content = Some(ChatCompletionMessageContent::Text(buffer));
                        choice
                    })
                    .collect();
                if data.choices.is_empty() {
                    None
                } else {
                    if let Some(slot) = response.sources_mut() {
                        *slot = sources;
                    }
                    Some((response, state))
                }
            }
        }
    })
    .fuse()
}
