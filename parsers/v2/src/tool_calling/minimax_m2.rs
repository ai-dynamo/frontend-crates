// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Streaming XML tool-call parser for MiniMax-M2.
//!
//! MiniMax emits tool calls as
//!   `<minimax:tool_call> <invoke name="NAME"> <parameter name="KEY">value</parameter> ... </invoke> </minimax:tool_call>`
//! plus a bare `<invoke name="..."></invoke>` back-off form when the outer
//! wrapper is absent (the v1 config sets `backoff_when_no_wrapper`).
//!
//! The streaming concern (buffering, chunk-split marker safety, normal_text
//! suppression) is owned by the shared [`scan::WrappedBlockScanner`]. The
//! per-block value typing is delegated to the v1 batch XML parser
//! `try_tool_call_parse_xml` driven by the same MiniMax config `dynamo_parsers`
//! uses for batch parsing, so a streamed call matches exactly what the batch
//! parser produces. Arguments are re-serialized in source
//! `<parameter name="...">` order because the v1 parser builds them from a
//! `HashMap` whose key order is non-deterministic; the fixtures store the
//! arguments as an exact JSON string, so order is pinned to the model-emitted
//! order (the order vLLM's Rust parser also preserves).
//!
//! M2-specific semantics (explicit spec fields, not copy drift): a bare-invoke
//! recovery CLEARS the suppression latch — when the optional outer close is
//! absent, later narration (e.g. ` Done.`) must still reach normal_text; a
//! stray close that does follow is stripped by the orphan-close handling.

use crate::tool_calling::scan::{
    BareRecoveryLatch, InvokeEmitter, InvokeLatch, WrappedBlockScanner, WrappedBlockSpec,
    reorder_arguments,
};
use crate::tool_calling::v1core::{ToolDefinition, XmlParserConfig, try_tool_call_parse_xml};

use crate::tool_calling::traits::{Tool, ToolCallDelta, ToolParseResult, ToolParser};

const BLOCK_START: &str = "<minimax:tool_call>";
const BLOCK_END: &str = "</minimax:tool_call>";
const FUNCTION_START: &str = "<invoke name=";
const FUNCTION_END: &str = "</invoke>";
const PARAMETER_START: &str = "<parameter name=";

/// MiniMax-M2 parser config, identical to `dynamo_parsers`' batch config so the
/// streamed value typing matches the v1 batch parser exactly.
fn minimax_config() -> XmlParserConfig {
    XmlParserConfig {
        tool_call_start_token: BLOCK_START.to_string(),
        tool_call_end_token: BLOCK_END.to_string(),
        function_start_token: FUNCTION_START.to_string(),
        function_end_token: FUNCTION_END.to_string(),
        parameter_start_token: PARAMETER_START.to_string(),
        parameter_end_token: "</parameter>".to_string(),
        allow_eof_recovery: false,
        strict_match: true,
        passthrough_when_no_function: false,
        backoff_when_no_wrapper: true,
    }
}

fn spec() -> WrappedBlockSpec {
    WrappedBlockSpec {
        family: "minimax_m2",
        block_starts: vec![BLOCK_START.to_string()],
        block_ends: vec![BLOCK_END.to_string()],
        invoke_start: FUNCTION_START.to_string(),
        invoke_end: FUNCTION_END.to_string(),
        orphan_markers: vec![BLOCK_END.to_string()],
        // BLOCK_END is held back too: a lone orphan close that arrives split
        // across chunks must be retained whole so the orphan-close path (which
        // strips it and never lets it leak) can match it.
        holdback_markers: vec![
            BLOCK_START.to_string(),
            FUNCTION_START.to_string(),
            BLOCK_END.to_string(),
        ],
        bare_recovery_latch: BareRecoveryLatch::Clear,
        invoke_latch: InvokeLatch::IfEmitted,
        drop_invoke_crossing_block_end: false,
        // Every wrapped family's markers are special tokens today.
        preserve_special_tokens: true,
        ..Default::default()
    }
}

/// Value-typing hook: wraps one complete `<invoke ...>...</invoke>` in the
/// block markers so the v1 parser takes its normal wrapped path, then re-orders
/// the arguments to source order.
struct M2Emitter {
    config: XmlParserConfig,
    tools: Vec<ToolDefinition>,
}

impl InvokeEmitter for M2Emitter {
    fn parse_invoke(
        &mut self,
        invoke: &str,
        tool_index: usize,
    ) -> anyhow::Result<Option<ToolCallDelta>> {
        let wrapped = format!("{BLOCK_START}{invoke}{BLOCK_END}");
        let (calls, _content) = try_tool_call_parse_xml(&wrapped, &self.config, Some(&self.tools))?;
        let Some(call) = calls.into_iter().next() else {
            return Ok(None);
        };
        let arguments =
            reorder_arguments(&call.function.arguments, &source_parameter_order(invoke));
        Ok(Some(ToolCallDelta {
            tool_index,
            name: Some(call.function.name),
            arguments,
            complete: true,
        }))
    }
}

/// Stream parser for MiniMax-M2 XML tool calls.
pub struct MiniMaxM2ToolStreamParser {
    scanner: WrappedBlockScanner<M2Emitter>,
}

impl MiniMaxM2ToolStreamParser {
    pub fn new(tools: &[Tool]) -> Self {
        Self {
            scanner: WrappedBlockScanner::new(
                spec(),
                M2Emitter {
                    config: minimax_config(),
                    tools: tools.iter().map(ToolDefinition::from).collect(),
                },
            ),
        }
    }
}

impl ToolParser for MiniMaxM2ToolStreamParser {
    fn create(tools: &[Tool]) -> anyhow::Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    fn preserve_special_tokens(&self) -> bool {
        self.scanner.preserve_special_tokens()
    }

    fn push(&mut self, chunk: &str) -> anyhow::Result<ToolParseResult> {
        self.scanner.push(chunk)
    }

    fn finish(&mut self) -> anyhow::Result<ToolParseResult> {
        self.scanner.finish()
    }
}

/// Parameter names in the order they appear in an invoke block.
fn source_parameter_order(function: &str) -> Vec<String> {
    let mut names = Vec::new();
    let mut cursor = 0;
    while let Some(rel) = function[cursor..].find(PARAMETER_START) {
        let start = cursor + rel + PARAMETER_START.len();
        let rest = &function[start..];
        let Some(after_quote) = rest.strip_prefix('"') else {
            cursor = start;
            continue;
        };
        let Some(name_end) = after_quote.find('"') else {
            break;
        };
        let name = after_quote[..name_end].trim();
        if !name.is_empty() {
            names.push(name.to_string());
        }
        cursor = start + 1 + name_end + 1;
    }
    names
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weather_tools() -> Vec<Tool> {
        vec![Tool {
            name: "get_weather".to_string(),
            description: None,
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "location": { "type": "string" } }
            }),
            strict: None,
        }]
    }

    fn parse_chunks(tools: &[Tool], chunks: &[&str]) -> ToolParseResult {
        let mut parser = MiniMaxM2ToolStreamParser::new(tools);
        let mut out = ToolParseResult::default();
        for chunk in chunks {
            out.append(parser.push(chunk).expect("push"));
        }
        out.append(parser.finish().expect("finish"));
        out
    }

    #[test]
    fn repeated_parameter_name_emits_key_once() {
        // A model that repeats `<parameter name="location">` must not produce
        // duplicate keys in the serialized arguments (the v1 object holds one
        // value per key; the reorder pass must emit it once).
        let out = parse_chunks(
            &weather_tools(),
            &["<minimax:tool_call>\n<invoke name=\"get_weather\">\
               \n<parameter name=\"location\">NYC</parameter>\
               \n<parameter name=\"location\">NYC</parameter>\
               \n</invoke>\n</minimax:tool_call>"],
        );
        let merged = out.coalesce_calls();
        assert_eq!(merged.calls.len(), 1);
        let args = merged.calls[0].arguments.clone();
        assert_eq!(
            args.matches("\"location\"").count(),
            1,
            "duplicate key in arguments: {args}"
        );
    }

    #[test]
    fn emits_complete_call_on_close() {
        let out = parse_chunks(
            &weather_tools(),
            &[
                "<minimax:tool_call>\n<invoke name=\"get_weather\">",
                "\n<parameter name=\"location\">",
                "NYC</parameter>\n</invoke>",
                "\n</minimax:tool_call>",
            ],
        );
        assert_eq!(out.normal_text, "");
        let merged = out.coalesce_calls();
        assert_eq!(merged.calls.len(), 1);
        assert_eq!(merged.calls[0].tool_index, 0);
        assert_eq!(merged.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(merged.calls[0].arguments, r#"{"location":"NYC"}"#);
    }

    #[test]
    fn preserves_prefix_text_before_block() {
        let out = parse_chunks(
            &weather_tools(),
            &[
                "I will",
                " check the weather. <minimax:tool_call>",
                "\n<invoke name=\"get_weather\">",
                "\n<parameter name=\"location\">NYC</parameter>\n</invoke>\n</minimax:tool_call>",
            ],
        );
        assert_eq!(out.normal_text, "I will check the weather. ");
        assert_eq!(out.coalesce_calls().calls.len(), 1);
    }

    #[test]
    fn emits_two_calls_in_one_block() {
        let out = parse_chunks(
            &weather_tools(),
            &[
                "<minimax:tool_call>\n<invoke name=\"get_weather\">\n<parameter name=\"location\">NYC</parameter>\n</invoke>",
                "\n<invoke name=\"get_weather\">\n<parameter name=\"location\">LA</parameter>\n</invoke>\n</minimax:tool_call>",
            ],
        );
        let merged = out.coalesce_calls();
        assert_eq!(merged.calls.len(), 2);
        assert_eq!(merged.calls[0].arguments, r#"{"location":"NYC"}"#);
        assert_eq!(merged.calls[1].arguments, r#"{"location":"LA"}"#);
    }

    #[test]
    fn preserves_trailing_text_after_block() {
        // 8.b: trailing narration after a complete block flows into normal_text.
        let out = parse_chunks(
            &weather_tools(),
            &[
                "<minimax:tool_call>\n<invoke name=\"get_weather\">\n<parameter name=\"location\">NYC</parameter>\n</invoke>\n</minimax:tool_call>",
                " Let me know if you need more.",
            ],
        );
        assert_eq!(out.normal_text, " Let me know if you need more.");
        assert_eq!(out.coalesce_calls().calls.len(), 1);
    }

    #[test]
    fn preserves_inter_call_and_trailing_text() {
        // 8.d: narration between two complete blocks flows into normal_text;
        // both calls are emitted with distinct indices.
        let out = parse_chunks(
            &weather_tools(),
            &[
                "I will check the weather. <minimax:tool_call>\n<invoke name=\"get_weather\">\n<parameter name=\"location\">NYC</parameter>\n</invoke>\n</minimax:tool_call>",
                " Then check LA weather. <minimax:tool_call>\n<invoke name=\"get_weather\">\n<parameter name=\"location\">LA</parameter>\n</invoke>\n</minimax:tool_call>",
            ],
        );
        assert_eq!(
            out.normal_text,
            "I will check the weather.  Then check LA weather. "
        );
        let merged = out.coalesce_calls();
        assert_eq!(merged.calls.len(), 2);
        assert_eq!(merged.calls[0].arguments, r#"{"location":"NYC"}"#);
        assert_eq!(merged.calls[1].arguments, r#"{"location":"LA"}"#);
    }

    #[test]
    fn suppresses_incomplete_invoke_at_eof() {
        let out = parse_chunks(
            &weather_tools(),
            &[
                "<minimax:tool_call>\n<invoke name=\"get_weather\">",
                "\n<parameter name=\"location\">NY",
            ],
        );
        assert_eq!(out.normal_text, "");
        assert!(out.calls.is_empty());
    }

    #[test]
    fn bare_invoke_preserves_trailing_text() {
        // A bare `<invoke>...</invoke>` (no outer wrapper) followed by narration:
        // the call is recovered AND the trailing ` Done.` survives in normal_text
        // (the bare-invoke path must not latch normal-text suppression).
        let out = parse_chunks(
            &weather_tools(),
            &[
                "<invoke name=\"get_weather\">\n<parameter name=\"location\">NYC</parameter>\n</invoke>",
                " Done.",
            ],
        );
        assert_eq!(out.normal_text, " Done.");
        let merged = out.coalesce_calls();
        assert_eq!(merged.calls.len(), 1);
        assert_eq!(merged.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(merged.calls[0].arguments, r#"{"location":"NYC"}"#);
    }

    #[test]
    fn bare_invoke_with_orphan_close_preserves_trailing_text() {
        // Bare invoke followed by a stray outer close then narration: the orphan
        // `</minimax:tool_call>` is stripped (never leaks) and ` Done.` survives.
        let out = parse_chunks(
            &weather_tools(),
            &[
                "<invoke name=\"get_weather\">\n<parameter name=\"location\">NYC</parameter>\n</invoke></minimax:tool_call> Done.",
            ],
        );
        assert_eq!(out.normal_text, " Done.");
        assert!(
            !out.normal_text.contains("minimax:tool_call"),
            "orphan close leaked into normal_text: {}",
            out.normal_text
        );
        let merged = out.coalesce_calls();
        assert_eq!(merged.calls.len(), 1);
        assert_eq!(merged.calls[0].arguments, r#"{"location":"NYC"}"#);
    }

    #[test]
    fn preserves_source_parameter_order() {
        let tools = vec![Tool {
            name: "file_editor".to_string(),
            description: None,
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "old_str": { "type": "string" },
                    "new_str": { "type": "string" },
                    "command": { "type": "string" }
                }
            }),
            strict: None,
        }];
        let out = parse_chunks(
            &tools,
            &[
                "<minimax:tool_call>\n<invoke name=\"file_editor\">",
                "\n<parameter name=\"path\">/app/x.go</parameter>",
                "\n<parameter name=\"old_str\">foo</parameter>",
                "\n<parameter name=\"new_str\">bar</parameter>",
                "\n<parameter name=\"command\">str_replace</parameter>",
                "\n</invoke>\n</minimax:tool_call>",
            ],
        );
        let merged = out.coalesce_calls();
        assert_eq!(merged.calls.len(), 1);
        assert_eq!(
            merged.calls[0].arguments,
            r#"{"path":"/app/x.go","old_str":"foo","new_str":"bar","command":"str_replace"}"#
        );
    }

    #[test]
    fn strips_lone_orphan_close_in_prose_whole_marker() {
        // A lone orphan `</minimax:tool_call>` in prose (no matching open, no
        // preceding recoverable invoke) must be stripped, never leaked, even when
        // it arrives as one whole marker.
        let out = parse_chunks(
            &weather_tools(),
            &["I will", " check that. ", "</minimax:tool_call>", " ok"],
        );
        assert_eq!(out.normal_text, "I will check that.  ok");
        assert!(out.calls.is_empty());
        assert!(
            !out.normal_text.contains("minimax") && !out.normal_text.contains("tool_call"),
            "orphan close leaked into normal_text: {}",
            out.normal_text
        );
    }

    #[test]
    fn strips_lone_orphan_close_in_prose_split_marker() {
        // Same lone orphan `</minimax:tool_call>`, but split across a chunk
        // boundary (`</minimax:tool` + `_call> ok`). The partial close suffix must
        // be held back whole (BLOCK_END is in the holdback list) so the
        // orphan-close path can strip it; nothing leaks into normal_text.
        let out = parse_chunks(
            &weather_tools(),
            &["I will", " check that. ", "</minimax:tool", "_call> ok", ""],
        );
        assert_eq!(out.normal_text, "I will check that.  ok");
        assert!(out.calls.is_empty());
        assert!(
            !out.normal_text.contains("minimax") && !out.normal_text.contains("tool_call"),
            "orphan close leaked into normal_text: {}",
            out.normal_text
        );
    }
}
