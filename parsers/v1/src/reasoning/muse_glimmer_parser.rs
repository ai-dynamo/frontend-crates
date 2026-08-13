// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Muse Glimmer recipient-routed reasoning parser.
//!
//! Muse Glimmer has no think tags. Chain-of-thought lives in `to=self`
//! channels of the message chain described in
//! [`crate::tool_calling::atem::muse_glimmer_parser`]:
//!
//! ```text
//! <|start|>assistant to=self<|message|>REASONING<|eom|>
//! <|start|>assistant to=user<|message|>ANSWER<|eot|>
//! ```
//!
//! `to=self` bodies route to `reasoning_text` (multiple blocks join with a
//! newline, matching both engines' batch parsers). `to=user` and
//! recipient-less bodies route to `normal_text` with framing stripped. A
//! tool-recipient channel is forwarded verbatim into `normal_text` — framing
//! intact, normalized to begin with `<|start|>` — so the downstream tool jail
//! can key on `<|start|>` and the `muse_glimmer` tool parser can strip the
//! framing. Pair this parser with `--dyn-tool-call-parser muse_glimmer`; on
//! its own, forwarded tool channels would reach the client wire-framed
//! (Inkling has the same pairing requirement).
//!
//! The generation prompt ends with `<|start|>assistant`, so the first message
//! of a turn arrives header-less (` to=self<|message|>`). The model has also
//! been observed to leave a channel without `<|eom|>`, writing the next
//! `to=<tool><|message|>` header directly; bodies therefore end at the next
//! header as well as at a terminator, or the trailing tool call would be
//! swallowed into reasoning.

use crate::tool_calling::atem::{
    EOM, EOT, MESSAGE, REASONING_RECIPIENT, START, USER_RECIPIENT, bare_header_pos,
    normalized_header, push_stripped, resolve_header,
};
use crate::{ParserResult, ReasoningParser};

const MARKERS: [&str; 4] = [START, MESSAGE, EOM, EOT];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// Between messages (or before the first). Prose here surfaces as
    /// content with orphan framing stripped.
    Idle,
    InReasoning,
    InContent,
    /// Forwarded verbatim (normalized framing included) for the tool parser.
    InToolChannel,
}

#[derive(Debug, Clone)]
pub struct MuseGlimmerReasoningParser {
    buffer: String,
    state: State,
    saw_reasoning_block: bool,
    /// Whether the next header may resolve WITHOUT `<|start|>` framing. True
    /// at turn start (the prompt consumed `<|start|>assistant`) and after a
    /// reasoning body cut at a bare header (missing-`<|eom|>` recovery); a
    /// bare-looking header anywhere else is quoted text, and resolving it
    /// would promote content into a live tool channel.
    allow_bare_header: bool,
    /// Last character already drained from the OPEN body, so recovery
    /// anchoring survives chunk splits (None right after the header).
    last_body_char: Option<char>,
}

impl MuseGlimmerReasoningParser {
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
            state: State::Idle,
            saw_reasoning_block: false,
            allow_bare_header: true,
            last_body_char: None,
        }
    }

    fn reset(&mut self) {
        self.buffer.clear();
        self.state = State::Idle;
        self.saw_reasoning_block = false;
        self.allow_bare_header = true;
        self.last_body_char = None;
    }
}

impl Default for MuseGlimmerReasoningParser {
    fn default() -> Self {
        Self::new()
    }
}

/// Longest suffix of `s` that is a proper prefix of `marker`.
fn overlap(s: &str, marker: &str) -> usize {
    let max = (marker.len() - 1).min(s.len());
    (1..=max)
        .rev()
        .find(|&i| marker.is_char_boundary(i) && s.ends_with(&marker[..i]))
        .unwrap_or(0)
}

fn partial_marker_suffix(s: &str) -> usize {
    MARKERS.iter().map(|m| overlap(s, m)).max().unwrap_or(0)
}

/// Length of a trailing fragment that could still grow into a bare
/// `to=<recipient><|message|>` header: `t` / `to` / `to=` + recipient
/// characters + an optional partial `<|message|>`, plus the space/tab run in
/// front that `resolve_header` would absorb into the header. Holding it back
/// keeps a recipient name from leaking into an open body and then needing to
/// be un-emitted (vLLM's `_OPEN_TAIL_HEADER_RE`, extended so streaming stays
/// byte-identical to batch at every chunk split).
fn open_header_tail(s: &str) -> usize {
    let tail = match s.rfind("to=") {
        Some(pos) if valid_header_fragment(&s[pos + 3..]) => s.len() - pos,
        _ if s.ends_with("to") => 2,
        _ if s.ends_with('t') => 1,
        _ => 0,
    };
    // Extend over the whitespace run a resolved bare header would absorb.
    let before = &s[..s.len() - tail];
    let ws = before.len() - before.trim_end_matches([' ', '\t']).len();
    tail + ws
}

/// Whether the bytes after a trailing `to=` could still complete a header:
/// recipient characters followed by nothing or a partial `<|message|>`.
/// Complete headers are handled by the boundary scan, not held.
fn valid_header_fragment(after: &str) -> bool {
    let rcpt_len = after
        .char_indices()
        .take_while(|(_, c)| !c.is_whitespace() && *c != '<')
        .map(|(i, c)| i + c.len_utf8())
        .last()
        .unwrap_or(0);
    let rest = &after[rcpt_len..];
    rest.is_empty() || (rest.len() < MESSAGE.len() && MESSAGE.starts_with(rest))
}

impl MuseGlimmerReasoningParser {
    /// Drive the state machine over `self.buffer`, appending routed text.
    fn run(&mut self, reasoning: &mut String, normal: &mut String) {
        loop {
            match self.state {
                State::Idle => {
                    if self.buffer.is_empty() {
                        break;
                    }
                    if let Some(msg_pos) = self.buffer.find(MESSAGE) {
                        let (mut header_start, recipient) = resolve_header(&self.buffer, msg_pos);
                        let framed = self.buffer[header_start..].starts_with(START);
                        let mut recipient = recipient.map(str::to_string);
                        if !framed && !self.allow_bare_header && recipient.is_some() {
                            // Quoted bare header: keep the `to=...` text as
                            // prose and treat the marker as a recipient-less
                            // content header.
                            header_start = msg_pos;
                            recipient = None;
                        }
                        push_stripped(normal, &self.buffer[..header_start]);
                        self.buffer.drain(..msg_pos + MESSAGE.len());
                        self.allow_bare_header = false;
                        self.last_body_char = None;
                        match recipient.as_deref() {
                            Some(REASONING_RECIPIENT) => {
                                if self.saw_reasoning_block {
                                    reasoning.push('\n');
                                }
                                self.saw_reasoning_block = true;
                                self.state = State::InReasoning;
                            }
                            Some(rcpt) if rcpt != USER_RECIPIENT => {
                                normal.push_str(&normalized_header(rcpt));
                                self.state = State::InToolChannel;
                            }
                            _ => self.state = State::InContent,
                        }
                        continue;
                    }
                    // No complete header yet: surface prose, hold anything
                    // that could still become framing. Idle scans start at a
                    // real boundary (turn start or a consumed terminator), so
                    // offset zero is anchored.
                    let bare_candidate = if self.allow_bare_header {
                        bare_header_pos(&self.buffer, None)
                    } else {
                        None
                    };
                    let hold_from = self
                        .buffer
                        .find(START)
                        .map(|s| bare_candidate.map_or(s, |b| s.min(b)))
                        .or(bare_candidate)
                        .unwrap_or_else(|| {
                            let mut tail = partial_marker_suffix(&self.buffer);
                            if self.allow_bare_header {
                                tail = tail.max(open_header_tail(&self.buffer));
                            }
                            self.buffer.len() - tail
                        });
                    if hold_from > 0 {
                        let emitted: String = self.buffer.drain(..hold_from).collect();
                        push_stripped(normal, &emitted);
                    }
                    break;
                }
                State::InReasoning | State::InContent | State::InToolChannel => {
                    let terminator = [EOM, EOT]
                        .iter()
                        .filter_map(|t| self.buffer.find(t).map(|p| (p, t.len())))
                        .min_by_key(|(p, _)| *p);
                    // Framed headers cut any body; bare headers cut only a
                    // reasoning body (missing-<|eom|> recovery).
                    let start_pos = self.buffer.find(START);
                    let bare_pos = if self.state == State::InReasoning {
                        bare_header_pos(&self.buffer, self.last_body_char)
                    } else {
                        None
                    };
                    let boundary = match (start_pos, bare_pos) {
                        (Some(a), Some(b)) => Some(a.min(b)),
                        (a, b) => a.or(b),
                    };

                    let cut = match (terminator, boundary) {
                        (Some((tp, _)), Some(bp)) if bp < tp => Some((bp, 0)),
                        (Some((tp, tlen)), _) => Some((tp, tlen)),
                        (None, Some(bp)) => Some((bp, 0)),
                        (None, None) => None,
                    };

                    if let Some((body_end, term_len)) = cut {
                        if term_len == 0 {
                            self.allow_bare_header =
                                bare_pos == Some(body_end) && start_pos != Some(body_end);
                        }
                        let body: String = self.buffer.drain(..body_end).collect();
                        if let Some(c) = body.chars().next_back() {
                            self.last_body_char = Some(c);
                        }
                        let term: String = self.buffer.drain(..term_len).collect();
                        match self.state {
                            State::InReasoning => reasoning.push_str(&body),
                            State::InContent => normal.push_str(&body),
                            State::InToolChannel => {
                                normal.push_str(&body);
                                if term.is_empty() {
                                    // The channel was cut by the NEXT header
                                    // (model omitted <|eom|>). Close the
                                    // forwarded message synthetically, or the
                                    // downstream parser reads whatever follows
                                    // as tool-channel payload and drops it.
                                    normal.push_str(EOM);
                                } else {
                                    normal.push_str(&term);
                                }
                            }
                            State::Idle => unreachable!(),
                        }
                        self.state = State::Idle;
                        continue;
                    }

                    // Open body: emit all but a tail that could still be a
                    // marker or, in reasoning, an incoming bare header.
                    let mut hold = partial_marker_suffix(&self.buffer);
                    if self.state == State::InReasoning {
                        hold = hold.max(open_header_tail(&self.buffer));
                    }
                    let split = self.buffer.len() - hold;
                    if split == 0 {
                        break;
                    }
                    let body: String = self.buffer.drain(..split).collect();
                    if let Some(c) = body.chars().next_back() {
                        self.last_body_char = Some(c);
                    }
                    match self.state {
                        State::InReasoning => reasoning.push_str(&body),
                        State::InContent | State::InToolChannel => normal.push_str(&body),
                        State::Idle => unreachable!(),
                    }
                    break;
                }
            }
        }
    }
}

impl ReasoningParser for MuseGlimmerReasoningParser {
    fn detect_and_parse_reasoning(&mut self, text: &str, _token_ids: &[u32]) -> ParserResult {
        // Batch: parse from a clean slate and reset, so a later stream is
        // unaffected. No trimming — engine parsers preserve body bytes.
        self.reset();
        let mut reasoning = String::new();
        let mut normal = String::new();
        self.buffer.push_str(text);
        self.run(&mut reasoning, &mut normal);
        let flush = self.finish_reasoning_stream();
        reasoning.push_str(&flush.reasoning_text);
        normal.push_str(&flush.normal_text);
        self.reset();
        ParserResult {
            reasoning_text: reasoning,
            normal_text: normal,
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
        let buffered = std::mem::take(&mut self.buffer);
        let state = self.state;
        // A wrapper is normally request-scoped, but leave it safe to reuse:
        // the next turn again begins after the consumed `<|start|>assistant`.
        self.state = State::Idle;
        self.saw_reasoning_block = false;
        self.allow_bare_header = true;
        if buffered.is_empty() {
            return ParserResult::default();
        }

        let mut result = ParserResult::default();
        match state {
            // Leftover here is either held framing or prose that could have
            // grown into framing. Complete markers are stripped; a committed
            // partial special token (`<|sta`) is parser-owned markup and
            // dropped; the ambiguous `<` / `<|` and any `to=`-shaped prose
            // stay visible (they contain no special tokens).
            State::Idle | State::InContent => {
                let text = flush_open_text(&buffered);
                match state {
                    State::Idle => push_stripped(&mut result.normal_text, &text),
                    _ => result.normal_text.push_str(&text),
                }
            }
            State::InReasoning => {
                result.reasoning_text.push_str(&flush_open_text(&buffered));
            }
            // An unfinished tool channel is forwarded in full for the tool
            // parser's end-of-stream recovery.
            State::InToolChannel => result.normal_text.push_str(&buffered),
        }
        result
    }
}

/// At end of stream, drop a committed partial special token from the tail of
/// held text but keep everything a human could have typed.
fn flush_open_text(buffered: &str) -> String {
    let tail = partial_marker_suffix(buffered);
    if tail <= 2 {
        // `<` / `<|` are ordinary prose as often as framing; keep them.
        return buffered.to_string();
    }
    buffered[..buffered.len() - tail].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    const REASONING_ANSWER: &str = " to=self<|message|>The user asks 2+2. It is 4.<|eom|><|start|>assistant to=user<|message|>2 + 2 = 4.<|eot|>";
    const REASONING_TOOL: &str = concat!(
        " to=self<|message|>Need the weather.<|eom|>",
        "<|start|>assistant to=get_weather<|message|><atem:function_calls>\n",
        "<atem:invoke name=\"get_weather\">\n",
        "<atem:parameter name=\"city\">Paris</atem:parameter>\n",
        "</atem:invoke>\n</atem:function_calls><|eom|>"
    );
    const TOOL_FORWARDED: &str = concat!(
        "<|start|>assistant to=get_weather<|message|><atem:function_calls>\n",
        "<atem:invoke name=\"get_weather\">\n",
        "<atem:parameter name=\"city\">Paris</atem:parameter>\n",
        "</atem:invoke>\n</atem:function_calls><|eom|>"
    );

    fn batch(text: &str) -> ParserResult {
        MuseGlimmerReasoningParser::new().detect_and_parse_reasoning(text, &[])
    }

    fn run_stream(chunks: &[&str]) -> (String, String) {
        let mut parser = MuseGlimmerReasoningParser::new();
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

    fn assert_no_marker_leak(s: &str) {
        for marker in MARKERS {
            assert!(!s.contains(marker), "marker {marker:?} leaked into {s:?}");
        }
    }

    #[test]
    fn batch_reasoning_then_answer() {
        let result = batch(REASONING_ANSWER);
        assert_eq!(result.reasoning_text, "The user asks 2+2. It is 4.");
        assert_eq!(result.normal_text, "2 + 2 = 4.");
        assert_no_marker_leak(&result.normal_text);
    }

    #[test]
    fn batch_reasoning_then_tool_channel_is_forwarded_framed() {
        let result = batch(REASONING_TOOL);
        assert_eq!(result.reasoning_text, "Need the weather.");
        assert_eq!(result.normal_text, TOOL_FORWARDED);
    }

    #[test]
    fn batch_multiple_reasoning_blocks_join_with_newline() {
        let result = batch(
            " to=self<|message|>first<|eom|><|start|>assistant to=self<|message|>second<|eom|><|start|>assistant to=user<|message|>done<|eot|>",
        );
        assert_eq!(result.reasoning_text, "first\nsecond");
        assert_eq!(result.normal_text, "done");
    }

    #[test]
    fn batch_plain_text_without_framing() {
        let result = batch("plain answer");
        assert_eq!(result.reasoning_text, "");
        assert_eq!(result.normal_text, "plain answer");
    }

    #[test]
    fn batch_empty_input() {
        let result = batch("");
        assert_eq!(result.reasoning_text, "");
        assert_eq!(result.normal_text, "");
    }

    #[test]
    fn batch_unterminated_reasoning_keeps_text_as_reasoning() {
        let result = batch(" to=self<|message|>cut off mid thought");
        assert_eq!(result.reasoning_text, "cut off mid thought");
        assert_eq!(result.normal_text, "");
    }

    #[test]
    fn batch_empty_reasoning_block() {
        let result =
            batch(" to=self<|message|><|eom|><|start|>assistant to=user<|message|>hi<|eot|>");
        assert_eq!(result.reasoning_text, "");
        assert_eq!(result.normal_text, "hi");
    }

    #[test]
    fn batch_bare_message_header_is_content() {
        let result = batch("<|start|>assistant<|message|>untagged content<|eot|>");
        assert_eq!(result.reasoning_text, "");
        assert_eq!(result.normal_text, "untagged content");
    }

    #[test]
    fn batch_tool_channel_missing_eom_before_framed_answer_keeps_the_answer() {
        // Model defect variant: the tool channel ends WITHOUT <|eom|> and the
        // framed answer follows directly. The forwarded channel must be closed
        // synthetically so the downstream tool parser does not absorb the
        // answer as tool-channel payload.
        let result = batch(concat!(
            " to=get_weather<|message|><atem:function_calls>\n",
            "<atem:invoke name=\"get_weather\">\n",
            "<atem:parameter name=\"city\">Paris</atem:parameter>\n",
            "</atem:invoke>\n</atem:function_calls>",
            "<|start|>assistant to=user<|message|>Here is the weather.<|eot|>",
        ));
        assert_eq!(result.reasoning_text, "");
        assert!(result.normal_text.ends_with("<|eom|>Here is the weather."));
        let (calls, content) =
            crate::tool_calling::try_tool_call_parse_muse_glimmer(&result.normal_text, None)
                .unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(content.unwrap(), "Here is the weather.");
    }

    #[test]
    fn batch_missing_eom_before_tool_header() {
        // Observed model defect: reasoning channel abandoned without <|eom|>.
        let result = batch(
            " to=self<|message|>thinking to=get_weather<|message|><atem:invoke name=\"get_weather\"></atem:invoke><|eom|>",
        );
        // The space before the recovered header stays in the body, exactly as
        // vLLM's bounded open-reasoning strip behaves.
        assert_eq!(result.reasoning_text, "thinking ");
        assert_eq!(
            result.normal_text,
            "<|start|>assistant to=get_weather<|message|><atem:invoke name=\"get_weather\"></atem:invoke><|eom|>"
        );
    }

    #[test]
    fn batch_quoted_bare_header_in_user_body_stays_content() {
        // The missing-<|eom|> recovery is reasoning-only: a bare header quoted
        // inside a `to=user` answer must not re-open a tool channel.
        let result = batch(
            "<|start|>assistant to=user<|message|>Example: to=search<|message|><atem:invoke name=\"search\"><atem:parameter name=\"q\">oops</atem:parameter></atem:invoke><|eot|>",
        );
        assert_eq!(result.reasoning_text, "");
        assert!(result.normal_text.contains("Example: to=search"));
        assert!(result.normal_text.contains("<atem:invoke name=\"search\">"));
        assert!(!result.normal_text.contains(&normalized_header("search")));
    }

    #[test]
    fn batch_tool_markup_quoted_in_reasoning_stays_reasoning() {
        let result = batch(
            " to=self<|message|>I could emit <atem:invoke name=\"f\"> but will not.<|eom|><|start|>assistant to=user<|message|>ok<|eot|>",
        );
        assert!(result.reasoning_text.contains("<atem:invoke name=\"f\">"));
        assert_eq!(result.normal_text, "ok");
    }

    #[test]
    fn batch_dangling_end_marker_is_stripped() {
        let result = batch("<|eom|>");
        assert_eq!(result.reasoning_text, "");
        assert_eq!(result.normal_text, "");
    }

    #[test]
    fn batch_prose_before_first_header_is_kept_clean() {
        let result = batch("Hello. to=self<|message|>think<|eom|>");
        assert_eq!(result.reasoning_text, "think");
        assert_eq!(result.normal_text, "Hello.");
    }

    #[test]
    fn streaming_matches_batch_for_reasoning_answer() {
        let (reasoning, normal) = run_stream(&[REASONING_ANSWER]);
        assert_eq!(reasoning, "The user asks 2+2. It is 4.");
        assert_eq!(normal, "2 + 2 = 4.");
    }

    #[test]
    fn streaming_reasoning_split_across_chunks() {
        let (reasoning, normal) = run_stream(&[
            " to=self<|message|>rea",
            "son<|eom|><|start|>assistant to=user<|message|>ans",
            "wer<|eot|>",
        ]);
        assert_eq!(reasoning, "reason");
        assert_eq!(normal, "answer");
    }

    #[test]
    fn streaming_marker_split_across_chunks() {
        let (reasoning, normal) = run_stream(&[
            " to=self<|mess",
            "age|>reason<|eo",
            "m|><|start|>assistant to=user<|message|>answer<|eot|>",
        ]);
        assert_eq!(reasoning, "reason");
        assert_eq!(normal, "answer");
        assert_no_marker_leak(&normal);
    }

    #[test]
    fn streaming_header_split_inside_recipient() {
        let (reasoning, normal) = run_stream(&[
            " to=se",
            "lf<|message|>think<|eom|><|start|>assistant to=us",
            "er<|message|>done<|eot|>",
        ]);
        assert_eq!(reasoning, "think");
        assert_eq!(normal, "done");
    }

    #[test]
    fn streaming_tool_channel_split_markers_forward_intact() {
        let chunks = [
            " to=self<|message|>go<|eom|><|start|>assistant to=get_",
            "weather<|message|><atem:function_calls>\n<atem:invoke name=\"get_weather\">\n",
            "<atem:parameter name=\"city\">Paris</atem:parameter>\n</atem:invoke>\n</atem:function_calls><|eo",
            "m|>",
        ];
        let (reasoning, normal) = run_stream(&chunks);
        assert_eq!(reasoning, "go");
        assert_eq!(normal, TOOL_FORWARDED);
    }

    #[test]
    fn streaming_matches_batch_at_every_split_boundary() {
        let cases = [
            REASONING_ANSWER,
            REASONING_TOOL,
            " to=self<|message|>a<|eom|><|start|>assistant to=self<|message|>b<|eom|>",
            " to=user<|message|>only answer<|eot|>",
            "<|message|>bare content<|eot|>",
            " to=self<|message|>thinking to=f<|message|><atem:invoke name=\"f\"></atem:invoke><|eom|>",
            "plain answer with no framing at all",
            " to=self<|message|>unterminated reasoning tail",
            " to=user<|message|>Example: to=search<|message|><atem:invoke name=\"q\"></atem:invoke><|eot|>",
            " to=f<|message|><atem:invoke name=\"f\"></atem:invoke><|start|>assistant to=user<|message|>kept<|eot|>",
            "my assistant  to=user<|message|>x<|eot|>",
            " to=self<|message|>weird potato=f<|message|><atem:invoke name=\"f\"></atem:invoke><|eom|>",
        ];
        for input in cases {
            let expected = batch(input);
            for split in input
                .char_indices()
                .map(|(i, _)| i)
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
    fn streaming_parser_reusable_after_finish() {
        let mut parser = MuseGlimmerReasoningParser::new();
        let first =
            parser.parse_reasoning_streaming_incremental(" to=user<|message|>one<|eot|>", &[]);
        assert_eq!(first.normal_text, "one");
        let _ = parser.finish_reasoning_stream();
        let second =
            parser.parse_reasoning_streaming_incremental(" to=self<|message|>two<|eom|>", &[]);
        assert_eq!(second.reasoning_text, "two");
        assert_eq!(second.normal_text, "");
    }

    #[test]
    fn streaming_trailing_partial_marker_is_dropped_at_finish() {
        let (reasoning, normal) = run_stream(&[" to=self<|message|>thought<|eo"]);
        assert_eq!(reasoning, "thought");
        assert_eq!(normal, "");
    }

    #[test]
    fn streaming_ambiguous_angle_prefix_is_kept_at_finish() {
        let (reasoning, normal) = run_stream(&[" to=user<|message|>a < b and a <| b"]);
        assert_eq!(reasoning, "");
        assert_eq!(normal, "a < b and a <| b");
    }

    #[test]
    fn crlf_reasoning_body_and_unicode_recipient_route_cleanly() {
        let result = batch(
            " to=self<|message|>line one\r\nline two<|eom|><|start|>assistant to=天気<|message|><atem:invoke name=\"天気\"></atem:invoke><|eom|>",
        );
        assert_eq!(result.reasoning_text, "line one\r\nline two");
        assert_eq!(
            result.normal_text,
            "<|start|>assistant to=天気<|message|><atem:invoke name=\"天気\"></atem:invoke><|eom|>"
        );
    }

    #[test]
    fn streaming_prose_to_tail_is_flushed_as_text() {
        // " to" could have grown into a header; at end of stream it is prose.
        let (reasoning, normal) = run_stream(&[" to=user<|message|>walk me to<|eot|>"]);
        assert_eq!(reasoning, "");
        assert_eq!(normal, "walk me to");
    }
}
