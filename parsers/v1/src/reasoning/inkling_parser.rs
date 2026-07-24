// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Inkling (thinkingmachines/Inkling-NVFP4) reasoning parser.
//!
//! Reasoning is block-structured, not a `<think>` prefix:
//! `<|message_model|><|content_thinking|>REASONING<|end_message|><|message_model|><|content_text|>ANSWER<|end_message|>`.
//! `<|content_thinking|>` blocks route to `reasoning_text`, `<|content_text|>` to
//! `normal_text`, with framing stripped. A tool-call block is passed through verbatim
//! into `normal_text` (framing intact): the reasoning parser runs before the tool-call
//! parser, which extracts calls from that `normal_text`.

use crate::tool_calling::inkling::find_complete_tool_call_end;
use crate::tool_calling::inkling::tokens::{
    END_MESSAGE, END_SAMPLING, INVOKE as CONTENT_INVOKE, MESSAGE_MODEL,
};
use crate::{ParserResult, ReasoningParser};

const CONTENT_THINKING: &str = "<|content_thinking|>";
const CONTENT_TEXT: &str = "<|content_text|>";
const CONTENT_IMAGE: &str = "<|content_image|>";
const CONTENT_AUDIO: &str = "<|content_audio_input|>";

/// Every Inkling special token: for stripping framing and holding a split marker.
const ALL_MARKERS: [&str; 8] = [
    MESSAGE_MODEL,
    CONTENT_THINKING,
    CONTENT_TEXT,
    CONTENT_INVOKE,
    END_MESSAGE,
    END_SAMPLING,
    CONTENT_IMAGE,
    CONTENT_AUDIO,
];

/// Content-type markers that follow `<|message_model|>`; a partial one leaves the
/// block's routing (reasoning/content vs verbatim tool-call) undecided.
const CONTENT_MARKERS: [&str; 5] = [
    CONTENT_THINKING,
    CONTENT_TEXT,
    CONTENT_INVOKE,
    CONTENT_IMAGE,
    CONTENT_AUDIO,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// Start of the assistant turn. Inkling's generation-primer `<|message_model|>`
    /// (add_generation_prompt) is consumed by the prompt, so the model's first block
    /// arrives header-less. Re-insert the primer only once the block is confirmed
    /// structured, so marker-less plain text is never reframed as a tool block.
    Primed,
    Idle,
    InReasoning,
    InContent,
    /// Passed through verbatim (framing included) for the downstream tool parser.
    InToolBlock,
    /// Non-text placeholder block (image/audio): consumed to its `<|end_message|>`
    /// and emitted to neither channel, so no framing or payload leaks.
    InDiscard,
}

#[derive(Debug, Clone)]
pub struct InklingReasoningParser {
    buffer: String,
    state: State,
}

impl InklingReasoningParser {
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
            state: State::Primed,
        }
    }
}

impl Default for InklingReasoningParser {
    fn default() -> Self {
        Self::new()
    }
}

/// Longest suffix of `s` that is a prefix of `delim` (a marker split across chunks).
/// Markers are ASCII, so the byte slice is always on a char boundary.
fn overlap(s: &str, delim: &str) -> usize {
    let max = delim.len().min(s.len());
    (1..=max)
        .rev()
        .find(|&i| s.ends_with(&delim[..i]))
        .unwrap_or(0)
}

fn max_partial_marker_suffix(s: &str) -> usize {
    ALL_MARKERS.iter().map(|m| overlap(s, m)).max().unwrap_or(0)
}

/// True when `s` is a non-empty proper prefix of some marker: parser-owned partial
/// markup that must be dropped on flush rather than surfaced as content.
fn is_partial_leading_marker(s: &str) -> bool {
    !s.is_empty()
        && ALL_MARKERS
            .iter()
            .any(|m| m.len() > s.len() && m.starts_with(s))
}

/// At EOF, preserve only the ambiguous prefixes that are also ordinary prose.
/// A longer prefix has committed to parser-owned markup and must not leak.
fn flush_ambiguous_marker_prefix(s: &str) -> String {
    match s {
        "<" | "<|" => s.to_string(),
        _ => String::new(),
    }
}

fn find_earliest(s: &str, markers: &[&str]) -> Option<(usize, usize)> {
    markers
        .iter()
        .filter_map(|m| s.find(m).map(|i| (i, m.len())))
        .min_by_key(|&(i, _)| i)
}

/// Strip framing from outside-block text only; never from a preserved tool-call block
/// or from extracted reasoning/content.
fn strip_framing(s: &str) -> String {
    let mut out = s.to_string();
    for marker in ALL_MARKERS {
        if out.contains(marker) {
            out = out.replace(marker, "");
        }
    }
    out
}

/// True while `rem` (bytes after `<|message_model|>`) could still grow into a
/// `<|content_*|>` token, so the block's routing is undecided and must wait for input.
fn content_type_undecided(rem: &str) -> bool {
    if rem.is_empty() {
        return true;
    }
    CONTENT_MARKERS
        .iter()
        .any(|t| t.len() > rem.len() && t.starts_with(rem))
}

/// A real block header precedes the tool delimiter. A marker-looking string
/// inside the JSON arguments comes after the delimiter and must not win routing.
fn has_header_before_invoke(s: &str) -> bool {
    match s.find(MESSAGE_MODEL) {
        Some(header) => s.find(CONTENT_INVOKE).is_none_or(|invoke| header < invoke),
        None => false,
    }
}

impl InklingReasoningParser {
    /// Drive the state machine, holding a trailing partial marker (or an undecided
    /// post-`<|message_model|>` region) in `self.buffer` for the next chunk.
    fn run(&mut self, reasoning: &mut String, normal: &mut String) {
        loop {
            match self.state {
                State::Primed => {
                    // First block of the turn (generation primer consumed by the
                    // prompt). Only re-insert the primer once the block is confirmed
                    // structured; otherwise hold, so marker-less plain text stays clean.
                    if self.buffer.is_empty() {
                        break;
                    }
                    if self.buffer.starts_with(MESSAGE_MODEL) {
                        // The block already carries a real header; route via Idle.
                        self.state = State::Idle;
                    } else if CONTENT_MARKERS.iter().any(|m| self.buffer.starts_with(m)) {
                        // A header-less structured block at the head always belongs
                        // to the consumed generation primer, even if later blocks
                        // contain their own real `<|message_model|>` headers.
                        self.buffer.insert_str(0, MESSAGE_MODEL);
                        self.state = State::Idle;
                    } else if has_header_before_invoke(&self.buffer) {
                        // Prose precedes a real `<|message_model|>` header later in the
                        // buffer. Hand off to Idle, which emits the clean prefix to
                        // normal_text and strips/routes the block (no framing leak).
                        // Check this before a later INVOKE so a real header wins over
                        // reconstructing a synthetic one at the start of the prose.
                        self.state = State::Idle;
                    } else if self.buffer.contains(CONTENT_INVOKE) {
                        // Header-less tool block: `NAME<|content_invoke_tool_json|>`
                        // (name first). Re-insert the consumed `<|message_model|>` so
                        // Idle passes the tool block with a header the tool parser strips.
                        self.buffer.insert_str(0, MESSAGE_MODEL);
                        self.state = State::Idle;
                    } else {
                        // Still a prefix of a marker/header, or non-marker text (plain
                        // content, or a tool NAME whose `<|content_invoke_tool_json|>` has
                        // not arrived). Hold without emitting or injecting; finish() flushes
                        // leftover plain text clean and drops partial markup.
                        break;
                    }
                }
                State::Idle => {
                    if let Some(pos) = self.buffer.find(MESSAGE_MODEL) {
                        normal.push_str(&strip_framing(&self.buffer[..pos]));
                        let rem_start = pos + MESSAGE_MODEL.len();
                        let rem = &self.buffer[rem_start..];
                        if rem.starts_with(CONTENT_THINKING) {
                            self.buffer = self.buffer[rem_start + CONTENT_THINKING.len()..].into();
                            self.state = State::InReasoning;
                        } else if rem.starts_with(CONTENT_TEXT) {
                            self.buffer = self.buffer[rem_start + CONTENT_TEXT.len()..].into();
                            self.state = State::InContent;
                        } else if rem.starts_with(CONTENT_IMAGE) {
                            self.buffer = self.buffer[rem_start + CONTENT_IMAGE.len()..].into();
                            self.state = State::InDiscard;
                        } else if rem.starts_with(CONTENT_AUDIO) {
                            self.buffer = self.buffer[rem_start + CONTENT_AUDIO.len()..].into();
                            self.state = State::InDiscard;
                        } else if content_type_undecided(rem) {
                            // Routing undecided; hold from `<|message_model|>`.
                            self.buffer = self.buffer[pos..].into();
                            break;
                        } else {
                            // Tool-call block (`NAME<|content_invoke_tool_json|>...`):
                            // preserve verbatim with its header for the tool parser.
                            self.buffer = self.buffer[pos..].into();
                            self.state = State::InToolBlock;
                        }
                    } else {
                        let hold = max_partial_marker_suffix(&self.buffer);
                        let split = self.buffer.len() - hold;
                        normal.push_str(&strip_framing(&self.buffer[..split]));
                        self.buffer = self.buffer[split..].into();
                        break;
                    }
                }
                State::InReasoning | State::InContent => {
                    let sink = if self.state == State::InReasoning {
                        &mut *reasoning
                    } else {
                        &mut *normal
                    };
                    if let Some((idx, mlen)) =
                        find_earliest(&self.buffer, &[END_MESSAGE, END_SAMPLING])
                    {
                        sink.push_str(&self.buffer[..idx]);
                        self.buffer = self.buffer[idx + mlen..].into();
                        self.state = State::Idle;
                    } else {
                        let hold = overlap(&self.buffer, END_MESSAGE)
                            .max(overlap(&self.buffer, END_SAMPLING));
                        let split = self.buffer.len() - hold;
                        sink.push_str(&self.buffer[..split]);
                        self.buffer = self.buffer[split..].into();
                        break;
                    }
                }
                State::InToolBlock => {
                    // Hold the complete block until a JSON-aware boundary is known.
                    // A raw `find(END_MESSAGE)` is unsafe because that literal can
                    // legally occur inside a JSON string argument. The downstream
                    // tool jail buffers this same span, so holding it here adds no
                    // user-visible content latency.
                    if let Some(upto) = find_complete_tool_call_end(&self.buffer) {
                        normal.push_str(&self.buffer[..upto]);
                        self.buffer = self.buffer[upto..].into();
                        self.state = State::Idle;
                    } else {
                        break;
                    }
                }
                State::InDiscard => {
                    // Non-text placeholder payload: emit nothing to either channel.
                    if let Some(idx) = self.buffer.find(END_MESSAGE) {
                        self.buffer = self.buffer[idx + END_MESSAGE.len()..].into();
                        self.state = State::Idle;
                    } else {
                        // Drop the payload; keep only a trailing partial `<|end_message|>`
                        // so the fence is still detected across the chunk boundary.
                        let hold = overlap(&self.buffer, END_MESSAGE);
                        let split = self.buffer.len() - hold;
                        self.buffer = self.buffer[split..].into();
                        break;
                    }
                }
            }
        }
    }
}

impl ReasoningParser for InklingReasoningParser {
    fn detect_and_parse_reasoning(&mut self, text: &str, _token_ids: &[u32]) -> ParserResult {
        // Batch: parse from a clean slate and reset, so a later stream is unaffected.
        self.buffer.clear();
        self.state = State::Primed;

        let mut reasoning = String::new();
        let mut normal = String::new();
        self.buffer.push_str(text);
        self.run(&mut reasoning, &mut normal);
        let flush = self.finish_reasoning_stream();
        reasoning.push_str(&flush.reasoning_text);
        normal.push_str(&flush.normal_text);

        self.buffer.clear();
        self.state = State::Primed;

        ParserResult {
            reasoning_text: reasoning.trim().to_string(),
            normal_text: normal.trim().to_string(),
        }
    }

    fn parse_reasoning_streaming_incremental(
        &mut self,
        text: &str,
        _token_ids: &[u32],
    ) -> ParserResult {
        self.buffer.push_str(text);
        let mut reasoning = String::new();
        let mut normal = String::new();
        self.run(&mut reasoning, &mut normal);
        ParserResult {
            reasoning_text: reasoning,
            normal_text: normal,
        }
    }

    fn finish_reasoning_stream(&mut self) -> ParserResult {
        if self.buffer.is_empty() {
            // A wrapper is normally request-scoped, but leave it safe to reuse:
            // the next turn again begins after the consumed generation primer.
            self.state = State::Primed;
            return ParserResult::default();
        }
        let buffered = std::mem::take(&mut self.buffer);
        let result = match self.state {
            // First block never resolved: flush plain leftover text as content, but
            // drop a lone partial marker (parser-owned markup).
            State::Primed => {
                if is_partial_leading_marker(&buffered) {
                    ParserResult::default()
                } else {
                    ParserResult {
                        // A complete stray framing token is parser-owned markup,
                        // even when no opening header ever established a state.
                        // Strip it just as Idle does; ordinary marker-less prose
                        // passes through unchanged.
                        normal_text: strip_framing(&buffered),
                        reasoning_text: String::new(),
                    }
                }
            }
            // While a block is open, `run` already emitted all ordinary payload and
            // retained only a suffix that could complete an end marker. At EOF,
            // preserve ambiguous prose (`<` / `<|`) but drop longer committed framing.
            State::InReasoning => ParserResult {
                reasoning_text: flush_ambiguous_marker_prefix(&buffered),
                normal_text: String::new(),
            },
            State::InContent => ParserResult {
                normal_text: flush_ambiguous_marker_prefix(&buffered),
                reasoning_text: String::new(),
            },
            // Unlike reasoning/content, an unfinished tool block is buffered in
            // full for downstream EOF recovery.
            State::InToolBlock => ParserResult {
                normal_text: buffered,
                reasoning_text: String::new(),
            },
            // Truncated non-text placeholder block: only a held partial `<|end_message|>`
            // remains (payload already dropped), so emit nothing.
            State::InDiscard => ParserResult::default(),
            // Idle leftover is a held proper prefix of a framing marker. `<` / `<|` is
            // the ambiguous pre-commitment prefix shared by every marker and is commonly
            // legitimate trailing prose, so preserve it as content (it is not a complete
            // framing token, so nothing leaks). A longer prefix (e.g. `<|content_th`) is
            // committed markup from a truncated header and is dropped so it can't leak.
            State::Idle => ParserResult {
                normal_text: flush_ambiguous_marker_prefix(&buffered),
                reasoning_text: String::new(),
            },
        };
        self.state = State::Primed;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const REASONING_ANSWER: &str = "<|message_model|><|content_thinking|>The user asks a simple arithmetic question. 2+2=4.<|end_message|><|message_model|><|content_text|>2 + 2 = 4.<|end_message|><|content_model_end_sampling|>";
    const REASONING_TOOL: &str = r#"<|message_model|><|content_thinking|>I should call the weather tool for Paris.<|end_message|><|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{"location":"Paris","unit":"celsius"}}<|end_message|><|content_model_end_sampling|>"#;

    fn assert_no_framing_leak(s: &str) {
        for marker in ALL_MARKERS {
            assert!(
                !s.contains(marker),
                "framing token {marker:?} leaked into {s:?}"
            );
        }
    }

    #[test]
    fn batch_reasoning_and_answer_split() {
        let mut parser = InklingReasoningParser::new();
        let result = parser.detect_and_parse_reasoning(REASONING_ANSWER, &[]);
        assert_eq!(
            result.reasoning_text,
            "The user asks a simple arithmetic question. 2+2=4."
        );
        assert_eq!(result.normal_text, "2 + 2 = 4.");
        assert_no_framing_leak(&result.reasoning_text);
        assert_no_framing_leak(&result.normal_text);
    }

    #[test]
    fn batch_reasoning_only() {
        let mut parser = InklingReasoningParser::new();
        let result = parser.detect_and_parse_reasoning(
            "<|message_model|><|content_thinking|>thinking<|end_message|>",
            &[],
        );
        assert_eq!(result.reasoning_text, "thinking");
        assert_eq!(result.normal_text, "");
    }

    #[test]
    fn batch_content_only() {
        let mut parser = InklingReasoningParser::new();
        let result = parser.detect_and_parse_reasoning(
            "<|message_model|><|content_text|>2 + 2 = 4.<|end_message|>",
            &[],
        );
        assert_eq!(result.reasoning_text, "");
        assert_eq!(result.normal_text, "2 + 2 = 4.");
    }

    #[test]
    fn batch_preserves_tool_block_verbatim() {
        let mut parser = InklingReasoningParser::new();
        let result = parser.detect_and_parse_reasoning(REASONING_TOOL, &[]);
        assert_eq!(
            result.reasoning_text,
            "I should call the weather tool for Paris."
        );
        // The tool block is passed through verbatim (framing intact) so the
        // downstream tool parser can extract it; the trailing terminator is not.
        assert_eq!(
            result.normal_text,
            r#"<|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{"location":"Paris","unit":"celsius"}}<|end_message|>"#
        );
        assert!(!result.normal_text.contains(END_SAMPLING));
        assert_no_framing_leak(&result.reasoning_text);
    }

    #[test]
    fn batch_plain_text_no_markers() {
        let mut parser = InklingReasoningParser::new();
        let result = parser.detect_and_parse_reasoning("plain answer", &[]);
        assert_eq!(result.reasoning_text, "");
        assert_eq!(result.normal_text, "plain answer");
    }

    #[test]
    fn batch_truncated_reasoning_block() {
        let mut parser = InklingReasoningParser::new();
        let result = parser.detect_and_parse_reasoning(
            "<|message_model|><|content_thinking|>partial reasoning",
            &[],
        );
        assert_eq!(result.reasoning_text, "partial reasoning");
        assert_eq!(result.normal_text, "");
    }

    fn run_stream(chunks: &[&str]) -> (String, String) {
        let mut parser = InklingReasoningParser::new();
        let (mut reasoning, mut normal) = (String::new(), String::new());
        for chunk in chunks {
            let r = parser.parse_reasoning_streaming_incremental(chunk, &[]);
            reasoning.push_str(&r.reasoning_text);
            normal.push_str(&r.normal_text);
        }
        let f = parser.finish_reasoning_stream();
        reasoning.push_str(&f.reasoning_text);
        normal.push_str(&f.normal_text);
        (reasoning, normal)
    }

    #[test]
    fn streaming_matches_batch_for_reasoning_answer() {
        let (reasoning, normal) = run_stream(&[REASONING_ANSWER]);
        assert_eq!(
            reasoning,
            "The user asks a simple arithmetic question. 2+2=4."
        );
        assert_eq!(normal, "2 + 2 = 4.");
    }

    #[test]
    fn streaming_reasoning_split_across_chunks() {
        let (reasoning, normal) = run_stream(&[
            "<|message_model|><|content_thinking|>rea",
            "son<|end_message|><|message_model|><|content_text|>ans",
            "wer<|end_message|>",
        ]);
        assert_eq!(reasoning, "reason");
        assert_eq!(normal, "answer");
    }

    #[test]
    fn streaming_start_marker_split_across_chunks() {
        // The content-type token arrives split: routing must stay undecided until
        // it completes, so no partial marker leaks.
        let (reasoning, normal) = run_stream(&[
            "<|message_model|><|content_thin",
            "king|>reason<|end_message|>",
        ]);
        assert_eq!(reasoning, "reason");
        assert_eq!(normal, "");
        assert_no_framing_leak(&reasoning);
        assert_no_framing_leak(&normal);
    }

    #[test]
    fn streaming_end_marker_split_across_chunks() {
        let (reasoning, normal) = run_stream(&[
            "<|message_model|><|content_thinking|>reason<|end_mes",
            "sage|><|message_model|><|content_text|>answer<|end_message|>",
        ]);
        assert_eq!(reasoning, "reason");
        assert_eq!(normal, "answer");
    }

    #[test]
    fn streaming_preserves_tool_block_verbatim() {
        let (reasoning, normal) = run_stream(&[REASONING_TOOL]);
        assert_eq!(reasoning, "I should call the weather tool for Paris.");
        assert_eq!(
            normal,
            r#"<|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{"location":"Paris","unit":"celsius"}}<|end_message|>"#
        );
    }

    // ---- Header-less first block (real deployment: the generation-primer
    // <|message_model|> is consumed by add_generation_prompt, so the model's
    // first block arrives with no leading <|message_model|>). ----

    #[test]
    fn batch_headerless_reasoning_routes_to_reasoning() {
        let mut parser = InklingReasoningParser::new();
        let result =
            parser.detect_and_parse_reasoning("<|content_thinking|>reason<|end_message|>", &[]);
        assert_eq!(result.reasoning_text, "reason");
        assert_eq!(result.normal_text, "");
        assert_no_framing_leak(&result.normal_text);
    }

    #[test]
    fn batch_headerless_content_routes_to_normal() {
        let mut parser = InklingReasoningParser::new();
        let result =
            parser.detect_and_parse_reasoning("<|content_text|>answer<|end_message|>", &[]);
        assert_eq!(result.reasoning_text, "");
        assert_eq!(result.normal_text, "answer");
    }

    #[test]
    fn batch_headerless_tool_block_reconstructs_header() {
        let mut parser = InklingReasoningParser::new();
        let result = parser.detect_and_parse_reasoning(
            r#"get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{"location":"SF"}}<|end_message|>"#,
            &[],
        );
        assert_eq!(result.reasoning_text, "");
        // The consumed primer is re-inserted so the downstream tool parser strips
        // the NAME header instead of leaking it into content.
        assert_eq!(
            result.normal_text,
            r#"<|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{"location":"SF"}}<|end_message|>"#
        );
    }

    #[test]
    fn streaming_headerless_tool_block_reconstructs_header() {
        let (reasoning, normal) = run_stream(&[
            "get",
            "_weather",
            "<|content_invoke_tool_json|>",
            r#"{"name":"get_weather","args":{"location":"SF"}}"#,
            "<|end_message|>",
        ]);
        assert_eq!(reasoning, "");
        assert_eq!(
            normal,
            r#"<|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{"location":"SF"}}<|end_message|>"#
        );
    }

    #[test]
    fn streaming_headerless_reasoning_then_content() {
        // First block header-less (thinking); the second block carries a real
        // <|message_model|> (the model emits it after the first <|end_message|>).
        let (reasoning, normal) = run_stream(&[
            "<|content_thinking|>rea",
            "son<|end_message|><|message_model|><|content_text|>ans",
            "wer<|end_message|>",
        ]);
        assert_eq!(reasoning, "reason");
        assert_eq!(normal, "answer");
    }

    #[test]
    fn streaming_headerless_plain_text_stays_clean() {
        let (reasoning, normal) = run_stream(&["plain ", "answer"]);
        assert_eq!(reasoning, "");
        assert_eq!(normal, "plain answer");
    }

    #[test]
    fn streaming_headerless_content_routes_to_normal() {
        // Most common real shape: a header-less <|content_text|> first block
        // (generation primer consumed), streamed across marker boundaries.
        let (reasoning, normal) =
            run_stream(&["<|content_te", "xt|>ans", "wer<|end_mess", "age|>"]);
        assert_eq!(reasoning, "");
        assert_eq!(normal, "answer");
    }

    #[test]
    fn streaming_partial_marker_first_chunk_holds_then_reclassifies() {
        // First chunk is only a marker prefix: Primed must hold (emit nothing)
        // until it completes, then route the header-less reasoning block.
        let (reasoning, normal) = run_stream(&["<|cont", "ent_thinking|>reason<|end_message|>"]);
        assert_eq!(reasoning, "reason");
        assert_eq!(normal, "");
    }

    #[test]
    fn batch_image_block_does_not_leak_into_normal_text() {
        // Image/audio blocks are non-text placeholder data: neither framing nor payload
        // may reach the content channel. Only the following text block surfaces.
        let mut parser = InklingReasoningParser::new();
        let result = parser.detect_and_parse_reasoning(
            "<|message_model|><|content_image|>IMGDATA<|end_message|><|message_model|><|content_text|>done<|end_message|>",
            &[],
        );
        assert_eq!(result.normal_text, "done");
        assert_eq!(result.reasoning_text, "");
        assert!(!result.normal_text.contains("IMGDATA"));
        assert_no_framing_leak(&result.normal_text);
    }

    #[test]
    fn batch_audio_block_does_not_leak_into_normal_text() {
        let mut parser = InklingReasoningParser::new();
        let result = parser.detect_and_parse_reasoning(
            "<|message_model|><|content_audio_input|>AUDIOBYTES<|end_message|><|message_model|><|content_text|>hi<|end_message|>",
            &[],
        );
        assert_eq!(result.normal_text, "hi");
        assert_eq!(result.reasoning_text, "");
        assert!(!result.normal_text.contains("AUDIOBYTES"));
        assert_no_framing_leak(&result.normal_text);
    }

    #[test]
    fn streaming_image_block_split_does_not_leak() {
        let (reasoning, normal) = run_stream(&[
            "<|message_model|><|content_ima",
            "ge|>IMG<|end_mess",
            "age|><|message_model|><|content_text|>ok<|end_message|>",
        ]);
        assert_eq!(normal, "ok");
        assert_eq!(reasoning, "");
        assert!(!normal.contains("IMG"));
        assert_no_framing_leak(&normal);
    }

    #[test]
    fn batch_prose_before_first_header_is_kept_clean() {
        // Prose emitted before the first real `<|message_model|>` must surface as
        // normal_text with the framing stripped; the thinking block routes to reasoning.
        let mut parser = InklingReasoningParser::new();
        let result = parser.detect_and_parse_reasoning(
            "Hello. <|message_model|><|content_thinking|>think<|end_message|><|message_model|><|content_text|>done<|end_message|>",
            &[],
        );
        assert_eq!(result.normal_text, "Hello. done");
        assert_eq!(result.reasoning_text, "think");
        assert_no_framing_leak(&result.normal_text);
        assert_no_framing_leak(&result.reasoning_text);
    }

    #[test]
    fn streaming_prose_before_first_header_is_kept_clean() {
        let (reasoning, normal) = run_stream(&[
            "Hello. <|message_model|><|content_thinking|>th",
            "ink<|end_message|>",
        ]);
        assert_eq!(normal, "Hello. ");
        assert_eq!(reasoning, "think");
        assert_no_framing_leak(&normal);
        assert_no_framing_leak(&reasoning);
    }

    #[test]
    fn batch_prose_before_first_tool_header_routes_without_synthetic_header() {
        let block = r#"<|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{"location":"Paris"}}<|end_message|>"#;
        let mut parser = InklingReasoningParser::new();
        let result = parser.detect_and_parse_reasoning(&format!("Hello. {block}"), &[]);
        assert_eq!(result.reasoning_text, "");
        assert_eq!(result.normal_text, format!("Hello. {block}"));
    }

    #[test]
    fn streaming_trailing_partial_angle_bracket_is_preserved_as_content() {
        // A stream ending in a lone `<|` after a completed block is ambiguous
        // pre-commitment prose, not a completed marker, so it is preserved as content
        // (it can never be a full framing token, so nothing leaks). Longer committed
        // header fragments are still dropped (see the Idle finish comment).
        let (reasoning, normal) = run_stream(&["<|content_text|>ans<|end_message|>", "<|"]);
        assert_eq!(normal, "ans<|");
        assert_eq!(reasoning, "");
    }

    #[test]
    fn streaming_trailing_truncated_header_fragment_is_dropped() {
        // A committed header fragment (`<|content_th`) from a truncated block is
        // parser-owned markup and must not leak into content.
        let (reasoning, normal) =
            run_stream(&["<|content_text|>ans<|end_message|>", "<|content_th"]);
        assert_eq!(normal, "ans");
        assert_eq!(reasoning, "");
        assert_no_framing_leak(&normal);
    }

    #[test]
    fn streaming_truncated_end_marker_is_dropped_from_open_blocks() {
        let (reasoning, normal) = run_stream(&["<|content_thinking|>reason<|end_mes"]);
        assert_eq!(reasoning, "reason");
        assert_eq!(normal, "");

        let (reasoning, normal) = run_stream(&["<|content_text|>answer<|end_mes"]);
        assert_eq!(reasoning, "");
        assert_eq!(normal, "answer");
    }

    #[test]
    fn streaming_ambiguous_short_prefix_is_preserved_inside_open_block() {
        let (reasoning, normal) = run_stream(&["<|content_text|>answer<|"]);
        assert_eq!(reasoning, "");
        assert_eq!(normal, "answer<|");
    }

    #[test]
    fn streaming_parser_can_be_reused_after_finish() {
        let mut parser = InklingReasoningParser::new();
        let first = parser
            .parse_reasoning_streaming_incremental("<|content_text|>first<|end_message|>", &[]);
        assert_eq!(first.normal_text, "first");
        let finished = parser.finish_reasoning_stream();
        assert_eq!(finished.reasoning_text, "");
        assert_eq!(finished.normal_text, "");

        let second = parser.parse_reasoning_streaming_incremental(
            "<|content_thinking|>second<|end_message|>",
            &[],
        );
        assert_eq!(second.reasoning_text, "second");
        assert_eq!(second.normal_text, "");
    }

    #[test]
    fn streaming_matches_batch_at_every_single_split_boundary() {
        let cases = [
            REASONING_ANSWER,
            REASONING_TOOL,
            "<|content_thinking|>reason<|end_message|><|message_model|><|content_text|>answer<|end_message|>",
            r#"<|message_model|>echo<|content_invoke_tool_json|>{"name":"echo","args":{"text":"a<|end_message|>b"}}<|end_message|>"#,
            "<|message_model|><|content_image|>IMG<|end_message|><|message_model|><|content_text|>done<|end_message|>",
        ];

        for input in cases {
            let mut batch_parser = InklingReasoningParser::new();
            let expected = batch_parser.detect_and_parse_reasoning(input, &[]);
            for split in input
                .char_indices()
                .map(|(idx, _)| idx)
                .chain(std::iter::once(input.len()))
            {
                let (reasoning, normal) = run_stream(&[&input[..split], &input[split..]]);
                assert_eq!(
                    (reasoning, normal),
                    (
                        expected.reasoning_text.clone(),
                        expected.normal_text.clone()
                    ),
                    "batch/stream mismatch at byte {split} for {input:?}"
                );
            }
        }
    }

    #[test]
    fn tool_marker_literal_inside_json_survives_reasoning_stage() {
        let block = r#"<|message_model|>echo<|content_invoke_tool_json|>{"name":"echo","args":{"text":"a<|end_message|>b"}}<|end_message|>"#;
        let mut parser = InklingReasoningParser::new();
        let batch = parser.detect_and_parse_reasoning(block, &[]);
        assert_eq!(batch.reasoning_text, "");
        assert_eq!(batch.normal_text, block);

        let (reasoning, normal) = run_stream(&[
            r#"<|message_model|>echo<|content_invoke_tool_json|>{"name":"echo","args":{"text":"a<|end_message|>"#,
            r#"b"}}<|end_message|>"#,
        ]);
        assert_eq!(reasoning, "");
        assert_eq!(normal, block);
    }

    #[test]
    fn header_marker_literal_inside_headerless_tool_json_does_not_retarget_routing() {
        let input = r#"echo<|content_invoke_tool_json|>{"name":"echo","args":{"text":"a<|message_model|>b"}}<|end_message|>"#;
        let expected = format!("{MESSAGE_MODEL}{input}");
        let mut parser = InklingReasoningParser::new();
        let result = parser.detect_and_parse_reasoning(input, &[]);
        assert_eq!(result.reasoning_text, "");
        assert_eq!(result.normal_text, expected);
    }

    #[test]
    fn batch_empty_and_whitespace_stay_clean() {
        let mut parser = InklingReasoningParser::new();
        let empty = parser.detect_and_parse_reasoning("", &[]);
        assert_eq!(empty.reasoning_text, "");
        assert_eq!(empty.normal_text, "");
        let ws = parser.detect_and_parse_reasoning("   ", &[]);
        assert_eq!(ws.reasoning_text, "");
        assert_eq!(ws.normal_text, "");
    }

    #[test]
    fn dangling_end_marker_is_stripped_without_an_open_block() {
        let mut parser = InklingReasoningParser::new();
        let result = parser.detect_and_parse_reasoning(END_MESSAGE, &[]);
        assert_eq!(result.reasoning_text, "");
        assert_eq!(result.normal_text, "");
    }
}
