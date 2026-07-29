// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unified streaming parser for the Qwen3 grammar.
//!
//! One state machine owns the whole assistant output:
//!
//! ```text
//! reasoning:  <think> … </think>
//! tool call:  <tool_call><function=NAME><parameter=KEY>VALUE</parameter></function></tool_call>
//! everything else: visible content
//! ```
//!
//! The scan core ([`WrappedBlockScanner`]) is the same one the tool-only
//! `Qwen3CoderToolStreamParser` runs on, configured here with the reasoning
//! channel as well. That is deliberate: block open/close, bare-`<function=>`
//! recovery, orphan-close stripping, chunk-boundary marker holdback and
//! EOF-drop behavior stay a single implementation, so the unified path cannot
//! quietly regress the tool handling that the tool-only conformance suite
//! already pins. Value typing likewise still delegates to the shared v1 batch
//! XML parser through [`Qwen3Emitter`], so an argument object streamed here is
//! byte-identical to the batch one.
//!
//! What the unified path adds is ORDER: `<think>` between two calls is a second
//! thought in its own position instead of being hoisted into the first.
//!
//! Nesting is asymmetric, because tool structure dominates. A tool call the
//! model emits INSIDE a thought is still a real call, so it is extracted and the
//! thought splits around it (burying it would drop the call and leak its markup
//! into the reasoning payload, `I3`). A reasoning marker inside a tool ARGUMENT
//! is data and survives byte-exact (`I7`), because the in-block scan never looks
//! for one.

use crate::tool_calling::qwen3_coder::{Qwen3Emitter, qwen3_scanner};
use crate::tool_calling::scan::{ReasoningSpec, WrappedBlockScanner};
use crate::tool_calling::traits::{Result, Tool};
use crate::unified::{UnifiedDelta, UnifiedParser};

const REASONING_START: &str = "<think>";
const REASONING_END: &str = "</think>";

/// Streaming unified parser for Qwen3: reasoning + content + tool calls.
pub struct Qwen3UnifiedParser {
    scanner: WrappedBlockScanner<Qwen3Emitter>,
}

impl Qwen3UnifiedParser {
    pub fn new(tools: &[Tool]) -> Self {
        Self {
            scanner: qwen3_scanner(tools).with_reasoning(ReasoningSpec {
                start: REASONING_START.to_string(),
                end: REASONING_END.to_string(),
                // Qwen3 emits its own `<think>`; the template does not pre-fill
                // one, so the stream starts in visible content (policy P5).
                forced_start: false,
            }),
        }
    }
}

impl UnifiedParser for Qwen3UnifiedParser {
    fn create(tools: &[Tool]) -> Result<Box<dyn UnifiedParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    fn preserve_special_tokens(&self) -> bool {
        true
    }

    fn push(&mut self, chunk: &str) -> Result<Vec<UnifiedDelta>> {
        self.scanner.push_ordered(chunk)
    }

    fn finish(&mut self) -> Result<Vec<UnifiedDelta>> {
        self.scanner.finish_ordered()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unified::{UnifiedEvent, assemble};

    fn weather_tools() -> Vec<Tool> {
        vec![Tool {
            name: "get_weather".to_string(),
            description: None,
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "city": { "type": "string" } }
            }),
            strict: None,
        }]
    }

    fn events(tools: &[Tool], chunks: &[&str]) -> Vec<UnifiedEvent> {
        let mut parser = Qwen3UnifiedParser::new(tools);
        let mut deltas = Vec::new();
        for chunk in chunks {
            deltas.extend(parser.push(chunk).expect("push"));
        }
        deltas.extend(parser.finish().expect("finish"));
        assemble(&deltas)
    }

    fn reasoning(text: &str) -> UnifiedEvent {
        UnifiedEvent::Reasoning { text: text.into() }
    }
    fn text(text: &str) -> UnifiedEvent {
        UnifiedEvent::Text { text: text.into() }
    }
    fn call(name: &str, arguments: serde_json::Value) -> UnifiedEvent {
        UnifiedEvent::ToolCall {
            name: name.into(),
            arguments,
        }
    }

    #[test]
    fn reasoning_after_a_call_keeps_its_position() {
        // The defect the unified parser exists to fix: under the split, both
        // thoughts merge into one span ahead of the call.
        let out = events(
            &weather_tools(),
            &[
                "<think>Look it up.</think>",
                "<tool_call><function=get_weather><parameter=city>Paris</parameter></function></tool_call>",
                "<think>Now answer.</think>It's 18C.",
            ],
        );
        assert_eq!(
            out,
            vec![
                reasoning("Look it up."),
                call("get_weather", serde_json::json!({"city": "Paris"})),
                reasoning("Now answer."),
                text("It's 18C."),
            ]
        );
    }

    #[test]
    fn content_before_reasoning_is_not_hoisted() {
        let out = events(
            &weather_tools(),
            &["Hello there. <think>let me recall</think>The capital is Paris."],
        );
        assert_eq!(
            out,
            vec![
                text("Hello there. "),
                reasoning("let me recall"),
                text("The capital is Paris."),
            ]
        );
    }

    #[test]
    fn unterminated_reasoning_is_promoted_at_finish() {
        // 4.e: not dropped, and not leaked as visible text.
        let out = events(&weather_tools(), &["<think>thinking but stream ends"]);
        assert_eq!(out, vec![reasoning("thinking but stream ends")]);
    }

    #[test]
    fn markers_split_across_chunks_never_leak() {
        // Every marker is cut in half at a chunk boundary.
        let out = events(
            &weather_tools(),
            &[
                "<thi",
                "nk>a</thin",
                "k>go: <tool",
                "_call><func",
                "tion=get_weather><parameter=city>Paris</parameter></func",
                "tion></tool",
                "_call>done",
            ],
        );
        assert_eq!(
            out,
            vec![
                reasoning("a"),
                text("go: "),
                call("get_weather", serde_json::json!({"city": "Paris"})),
                text("done"),
            ]
        );
    }

    #[test]
    fn reasoning_marker_inside_an_argument_is_data() {
        // I7: once a tool block is open, `<think>` is a value, not a control
        // token, and must survive byte-exact.
        let tools = vec![Tool {
            name: "run".to_string(),
            description: None,
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "cmd": { "type": "string" } }
            }),
            strict: None,
        }];
        let out = events(
            &tools,
            &[
                "<tool_call><function=run><parameter=cmd>echo <think>hi</think></parameter></function></tool_call>",
            ],
        );
        assert_eq!(
            out,
            vec![call(
                "run",
                serde_json::json!({"cmd": "echo <think>hi</think>"})
            )]
        );
    }

    #[test]
    fn tool_call_inside_reasoning_is_extracted_and_splits_the_thought() {
        // Tool structure dominates reasoning. A call the model emits inside a
        // thought is still a real call, so it surfaces as its own event and the
        // thought splits around it. Burying it would both drop the call and
        // leak `<tool_call>` markup into the reasoning payload (`I3`).
        let out = events(
            &weather_tools(),
            &[
                "<think>I should check. <tool_call><function=get_weather><parameter=city>Paris</parameter></function></tool_call> now answer</think>Done.",
            ],
        );
        assert_eq!(
            out,
            vec![
                reasoning("I should check. "),
                call("get_weather", serde_json::json!({"city": "Paris"})),
                reasoning(" now answer"),
                text("Done."),
            ]
        );
    }

    #[test]
    fn the_two_nestings_are_not_symmetric() {
        // Tool-inside-reasoning extracts the call (above), but
        // reasoning-inside-a-tool-argument stays argument data (`I7`) — the
        // in-block scan never looks for reasoning markers.
        let tools = vec![Tool {
            name: "log".to_string(),
            description: None,
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "note": { "type": "string" } }
            }),
            strict: None,
        }];
        let out = events(
            &tools,
            &[
                "<tool_call><function=log><parameter=note><think>reconsider</think></parameter></function></tool_call>",
            ],
        );
        assert_eq!(
            out,
            vec![call(
                "log",
                serde_json::json!({"note": "<think>reconsider</think>"})
            )]
        );
    }

    #[test]
    fn an_argument_value_may_contain_the_block_close_marker() {
        // I7: the scanner has already delimited the invoke, so typing must not
        // re-discover its bounds and cut the value at an embedded `</tool_call>`.
        let tools = vec![Tool {
            name: "run".to_string(),
            description: None,
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "cmd": { "type": "string" } }
            }),
            strict: None,
        }];
        let out = events(
            &tools,
            &[
                "<tool_call>\n<function=run>\n<parameter=cmd>\ngit log </tool_call> --oneline\n</parameter>\n</function>\n</tool_call>",
            ],
        );
        assert_eq!(
            out,
            vec![call(
                "run",
                serde_json::json!({"cmd": "git log </tool_call> --oneline"})
            )]
        );
    }

    #[test]
    fn a_duplicate_reasoning_opener_inside_a_thought_is_stripped() {
        // I3: best-effort recovery strips malformed markup rather than letting it
        // land in the payload. A second `<think>` while one is already open is a
        // duplicate opener, not content.
        let out = events(&weather_tools(), &["<think>a<think>b</think>tail"]);
        assert_eq!(out, vec![reasoning("ab"), text("tail")]);
    }

    #[test]
    fn a_stray_tool_close_inside_a_thought_is_stripped() {
        // Same rule as the orphan handler applies OUTSIDE reasoning — a stray
        // `</tool_call>` with nothing open is markup, so it must not leak into the
        // reasoning payload just because a thought happens to be open.
        let out = events(&weather_tools(), &["<think>a</tool_call>b</think>tail"]);
        assert_eq!(out, vec![reasoning("ab"), text("tail")]);
    }

    #[test]
    fn orphan_reasoning_close_is_stripped_not_leaked() {
        // I3: a `</think>` with nothing open is malformed markup.
        let out = events(&weather_tools(), &["Hello </think>world"]);
        assert_eq!(out, vec![text("Hello world")]);
    }

    #[test]
    fn truncated_tool_call_at_eof_keeps_preceding_output() {
        // P2: drop the unrecoverable partial call, no error, no leaked markup.
        let out = events(
            &weather_tools(),
            &["<think>ok</think>Checking. <tool_call><function=get_weather><parameter=city>Par"],
        );
        assert_eq!(out, vec![reasoning("ok"), text("Checking. ")]);
    }

    #[test]
    fn empty_arguments_become_an_empty_object() {
        // P3.
        let tools = vec![Tool {
            name: "ping".to_string(),
            description: None,
            parameters: serde_json::json!({"type": "object", "properties": {}}),
            strict: None,
        }];
        let out = events(
            &tools,
            &["<tool_call><function=ping></function></tool_call>"],
        );
        assert_eq!(out, vec![call("ping", serde_json::json!({}))]);
    }

    #[test]
    fn batch_and_stream_assemble_identically() {
        // I6, at the parser level: `parse_complete` routes through the same
        // push/finish lifecycle, so parity is structural.
        let input = "<think>a</think>Here you go: <tool_call><function=get_weather><parameter=city>Paris</parameter></function></tool_call><think>b</think>Done.";
        let batch = Qwen3UnifiedParser::new(&weather_tools())
            .parse_complete(input)
            .expect("parse_complete");
        assert_eq!(batch, events(&weather_tools(), &[input]));
        assert_eq!(batch.len(), 5);
    }
}
