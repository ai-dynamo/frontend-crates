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

use crate::{ParserResult, ReasoningParser};

const MESSAGE_MODEL: &str = "<|message_model|>";
const CONTENT_THINKING: &str = "<|content_thinking|>";
const CONTENT_TEXT: &str = "<|content_text|>";
const CONTENT_INVOKE: &str = "<|content_invoke_tool_json|>";
const END_MESSAGE: &str = "<|end_message|>";
const END_SAMPLING: &str = "<|content_model_end_sampling|>";
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
                    } else if CONTENT_MARKERS.iter().any(|m| self.buffer.starts_with(m))
                        || self.buffer.contains(CONTENT_INVOKE)
                    {
                        // Header-less structured block: `<|content_thinking|>` /
                        // `<|content_text|>` at the head, or `NAME<|content_invoke_tool_json|>`
                        // (name first). Re-insert the consumed `<|message_model|>` so the
                        // Idle router sends reasoning to reasoning, content to content, and
                        // the tool block verbatim (with a header the tool parser can strip).
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
                        } else if content_type_undecided(rem) {
                            // Routing undecided; hold from `<|message_model|>`.
                            self.buffer = self.buffer[pos..].into();
                            break;
                        } else {
                            // Tool-call block: preserve verbatim with its header.
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
                    if let Some(idx) = self.buffer.find(END_MESSAGE) {
                        let upto = idx + END_MESSAGE.len();
                        normal.push_str(&self.buffer[..upto]);
                        self.buffer = self.buffer[upto..].into();
                        self.state = State::Idle;
                    } else {
                        let hold = overlap(&self.buffer, END_MESSAGE);
                        let split = self.buffer.len() - hold;
                        normal.push_str(&self.buffer[..split]);
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
                        normal_text: buffered,
                        reasoning_text: String::new(),
                    }
                }
            }
            // Block truncated before its `<|end_message|>`: flush what we have.
            State::InReasoning => ParserResult {
                reasoning_text: buffered,
                normal_text: String::new(),
            },
            State::InContent | State::InToolBlock => ParserResult {
                normal_text: buffered,
                reasoning_text: String::new(),
            },
            // Idle leftover is a held partial marker: parser-owned markup, so drop it.
            State::Idle => ParserResult::default(),
        };
        self.state = State::Idle;
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
    fn batch_empty_and_whitespace_stay_clean() {
        let mut parser = InklingReasoningParser::new();
        let empty = parser.detect_and_parse_reasoning("", &[]);
        assert_eq!(empty.reasoning_text, "");
        assert_eq!(empty.normal_text, "");
        let ws = parser.detect_and_parse_reasoning("   ", &[]);
        assert_eq!(ws.reasoning_text, "");
        assert_eq!(ws.normal_text, "");
    }
}
