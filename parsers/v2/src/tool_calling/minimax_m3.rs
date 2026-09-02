// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Streaming XML-ish tool-call parser for MiniMax-M3.
//!
//! MiniMax-M3 prefixes every tag with the namespace token `]<]minimax[>[` and
//! names parameters by their TAG (not a `name=` attribute):
//!   `]<]minimax[>[<tool_call>
//!    ]<]minimax[>[<invoke name="NAME">
//!    ]<]minimax[>[<KEY>value]<]minimax[>[</KEY>
//!    ]<]minimax[>[</invoke>
//!    ]<]minimax[>[</tool_call>`
//! plus a bare `]<]minimax[>[<invoke ...>...</invoke>` back-off form when the
//! outer wrapper is absent.
//!
//! The streaming concern (buffering, chunk-split marker safety, normal_text
//! suppression) is owned by the shared [`scan::WrappedBlockScanner`]; all
//! three markers begin with the namespace token, so the shared holdback also
//! retains a split `]<]minimax[>[` run. The per-invoke value typing is
//! delegated to the v1 batch parser `try_tool_call_parse_minimax_m3` driven by
//! the same config `dynamo_parsers` uses for batch parsing, so a streamed call
//! matches exactly what the batch parser produces. Arguments are re-serialized
//! in source parameter-tag order because the v1 parser builds them from a
//! `HashMap` whose key order is non-deterministic.

use crate::tool_calling::scan::{
    BareRecoveryLatch, InvokeEmitter, InvokeLatch, WrappedBlockScanner, WrappedBlockSpec,
    reorder_arguments,
};
use crate::tool_calling::v1core::{
    MiniMaxM3ParserConfig, ToolDefinition, try_tool_call_parse_minimax_m3,
};

use crate::tool_calling::traits::{Tool, ToolCallDelta, ToolParseResult, ToolParser};

/// The namespace token emitted before every M3 tag.
const NS: &str = "]<]minimax[>[";
const BLOCK_START: &str = "]<]minimax[>[<tool_call>";
const BLOCK_END: &str = "]<]minimax[>[</tool_call>";
/// Bare `<invoke` (no trailing `>`): matches both `<invoke name="...">` and the
/// malformed nameless `<invoke>` (whose parse yields no call and is dropped).
const FUNCTION_START: &str = "]<]minimax[>[<invoke";
const FUNCTION_END: &str = "]<]minimax[>[</invoke>";

fn spec() -> WrappedBlockSpec {
    WrappedBlockSpec {
        family: "minimax_m3",
        block_starts: vec![BLOCK_START.to_string()],
        block_ends: vec![BLOCK_END.to_string()],
        invoke_start: FUNCTION_START.to_string(),
        invoke_end: FUNCTION_END.to_string(),
        orphan_markers: vec![BLOCK_END.to_string()],
        // BLOCK_END is held back too: after a bare-invoke recovery latches
        // suppression, a split closing marker must be retained whole so the
        // orphan-close path (which clears the latch) can match it.
        holdback_markers: vec![
            BLOCK_START.to_string(),
            FUNCTION_START.to_string(),
            BLOCK_END.to_string(),
        ],
        bare_recovery_latch: BareRecoveryLatch::Set,
        invoke_latch: InvokeLatch::IfEmitted,
        drop_invoke_crossing_block_end: false,
        // Every wrapped family's markers are special tokens today.
        preserve_special_tokens: true,
        ..Default::default()
    }
}

/// Value-typing hook: wraps one complete invoke run in the M3 `<tool_call>`
/// block so the v1 parser takes its normal wrapped path, then re-orders the
/// arguments to source parameter-tag order.
struct M3Emitter {
    config: MiniMaxM3ParserConfig,
    tools: Vec<ToolDefinition>,
}

impl InvokeEmitter for M3Emitter {
    fn parse_invoke(
        &mut self,
        invoke: &str,
        tool_index: usize,
    ) -> anyhow::Result<Option<ToolCallDelta>> {
        let wrapped = format!("{BLOCK_START}{invoke}{BLOCK_END}");
        let tools_opt = (!self.tools.is_empty()).then_some(self.tools.as_slice());
        let (calls, _content) = try_tool_call_parse_minimax_m3(&wrapped, &self.config, tools_opt)?;
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

/// Stream parser for MiniMax-M3 tool calls.
pub struct MiniMaxM3ToolStreamParser {
    scanner: WrappedBlockScanner<M3Emitter>,
}

impl MiniMaxM3ToolStreamParser {
    pub fn new(tools: &[Tool]) -> Self {
        Self {
            scanner: WrappedBlockScanner::new(
                spec(),
                M3Emitter {
                    // Identical to `dynamo_parsers`' batch config so the streamed
                    // value typing matches the v1 batch parser exactly.
                    config: MiniMaxM3ParserConfig::default(),
                    tools: tools.iter().map(ToolDefinition::from).collect(),
                },
            ),
        }
    }
}

impl ToolParser for MiniMaxM3ToolStreamParser {
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

/// TOP-LEVEL parameter tag names in the order they appear in an invoke run.
///
/// M3 names parameters by their tag (`]<]minimax[>[<location>value...`), and
/// values may nest further namespaced tags (arrays/objects — batch cases
/// 7.d/7.e), so the scan tracks tag depth and records only the depth-0 openers
/// inside the invoke body; nested tag names never shadow a top-level key.
fn source_parameter_order(function: &str) -> Vec<String> {
    let mut names = Vec::new();
    let mut depth: i32 = 0;
    let mut cursor = 0;
    while let Some(rel) = function[cursor..].find(NS) {
        let tag_start = cursor + rel + NS.len();
        let rest = &function[tag_start..];
        let Some(after_lt) = rest.strip_prefix('<') else {
            cursor = tag_start;
            continue;
        };
        if let Some(closer) = after_lt.strip_prefix('/') {
            // `</invoke>` closes the run; any other closer pops one level.
            if !closer.starts_with("invoke") {
                depth -= 1;
            }
            cursor = tag_start + 1;
            continue;
        }
        if after_lt.starts_with("invoke") {
            cursor = tag_start + 1;
            continue;
        }
        let Some(name_end) = after_lt.find('>') else {
            break;
        };
        let name = after_lt[..name_end].trim();
        if !name.is_empty() {
            if depth == 0 {
                names.push(name.to_string());
            }
            depth += 1;
        }
        cursor = tag_start + 1 + name_end + 1;
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
        let mut parser = MiniMaxM3ToolStreamParser::new(tools);
        let mut out = ToolParseResult::default();
        for chunk in chunks {
            out.append(parser.push(chunk).expect("push"));
        }
        out.append(parser.finish().expect("finish"));
        out
    }

    #[test]
    fn emits_complete_call_on_close() {
        let out = parse_chunks(
            &weather_tools(),
            &[
                "]<]minimax[>[<tool_call>\n]<]minimax[>[<invoke name=\"get_weather\">",
                "\n]<]minimax[>[<location>",
                "NYC]<]minimax[>[</location>\n]<]minimax[>[</invoke>",
                "\n]<]minimax[>[</tool_call>",
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
    fn preserves_prefix_and_trailing_text() {
        // 8.c: prefix before the block AND narration after the close both flow
        // into normal_text verbatim; the block markup is suppressed.
        let out = parse_chunks(
            &weather_tools(),
            &[
                "I will check the weather. ]<]minimax[>[<tool_call>\n]<]minimax[>[<invoke name=\"get_weather\">",
                "\n]<]minimax[>[<location>NYC]<]minimax[>[</location>\n]<]minimax[>[</invoke>\n]<]minimax[>[</tool_call>",
                " Let me know if you need more.",
            ],
        );
        assert_eq!(
            out.normal_text,
            "I will check the weather.  Let me know if you need more."
        );
        assert_eq!(out.coalesce_calls().calls.len(), 1);
    }

    #[test]
    fn suppresses_in_block_narration_between_invokes() {
        // 8.d: prose INSIDE the block between two invokes is part of the markup
        // block and is dropped, matching the v1 batch parser.
        let out = parse_chunks(
            &weather_tools(),
            &[
                "I will check both cities. ]<]minimax[>[<tool_call>\n]<]minimax[>[<invoke name=\"get_weather\">\n]<]minimax[>[<location>NYC]<]minimax[>[</location>\n]<]minimax[>[</invoke>",
                "\nThen check LA.\n]<]minimax[>[<invoke name=\"get_weather\">\n]<]minimax[>[<location>LA]<]minimax[>[</location>\n]<]minimax[>[</invoke>\n]<]minimax[>[</tool_call>",
            ],
        );
        assert_eq!(out.normal_text, "I will check both cities. ");
        let merged = out.coalesce_calls();
        assert_eq!(merged.calls.len(), 2);
        assert_eq!(merged.calls[0].arguments, r#"{"location":"NYC"}"#);
        assert_eq!(merged.calls[1].arguments, r#"{"location":"LA"}"#);
    }

    #[test]
    fn recovers_bare_invoke_without_wrapper() {
        // 5.g-shaped: prose + bare invoke (no `<tool_call>` opener) + orphan close.
        let out = parse_chunks(
            &weather_tools(),
            &[
                "I will check that. ]<]minimax[>[<invoke name=\"get_weather\">\n]<]minimax[>[<location>NYC]<]minimax[>[</location>\n]<]minimax[>[</invoke>",
                "\n]<]minimax[>[</tool_call>",
            ],
        );
        assert_eq!(out.normal_text, "I will check that. ");
        let merged = out.coalesce_calls();
        assert_eq!(merged.calls.len(), 1);
        assert_eq!(merged.calls[0].arguments, r#"{"location":"NYC"}"#);
    }

    #[test]
    fn holds_back_split_orphan_close_after_bare_invoke() {
        // 5.g-shaped, but the orphan `BLOCK_END` close is split across two chunk
        // boundaries while `suppress_normal_text` is latched by the bare-invoke
        // recovery. The split close must be held back whole so the orphan-close
        // path can consume it and clear the latch; otherwise its remainder is
        // unrecognizable, suppression stays on, and the trailing prose is dropped.
        let out = parse_chunks(
            &weather_tools(),
            &[
                "I will check that. ]<]minimax[>[<invoke name=\"get_weather\">\n]<]minimax[>[<location>NYC]<]minimax[>[</location>\n]<]minimax[>[</invoke>\n]<]minimax",
                "[>[</tool",
                "_call>Here is the weather.",
            ],
        );
        // The close marker never leaks, and the trailing prose survives.
        assert_eq!(out.normal_text, "I will check that. Here is the weather.");
        assert!(!out.normal_text.contains("tool_call"));
        assert!(!out.normal_text.contains("minimax"));
        let merged = out.coalesce_calls();
        assert_eq!(merged.calls.len(), 1);
        assert_eq!(merged.calls[0].arguments, r#"{"location":"NYC"}"#);
    }

    #[test]
    fn suppresses_incomplete_invoke_at_eof() {
        // 5.c: block start + invoke truncated mid-value never leaks.
        let out = parse_chunks(
            &weather_tools(),
            &[
                "]<]minimax[>[<tool_call>\n]<]minimax[>[<invoke name=\"get_weather\">",
                "\n]<]minimax[>[<location>NY",
            ],
        );
        assert_eq!(out.normal_text, "");
        assert!(out.calls.is_empty());
    }

    #[test]
    fn holds_back_split_namespace_token() {
        // A chunk boundary inside the namespace token must not leak its prefix.
        let out = parse_chunks(
            &weather_tools(),
            &[
                "text before ]<]mini",
                "max[>[<tool_call>\n]<]minimax[>[<invoke name=\"get_weather\">\n]<]minimax[>[<location>NYC]<]minimax[>[</location>\n]<]minimax[>[</invoke>\n]<]minimax[>[</tool_call>",
            ],
        );
        assert_eq!(out.normal_text, "text before ");
        assert_eq!(out.coalesce_calls().calls.len(), 1);
    }

    #[test]
    fn preserves_source_parameter_order_and_types() {
        // 7.a-shaped: multiple typed parameters keep model-emitted order.
        let tools = vec![Tool {
            name: "book_flight".to_string(),
            description: None,
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "destination": { "type": "string" },
                    "passengers": { "type": "integer" },
                    "first_class": { "type": "boolean" }
                }
            }),
            strict: None,
        }];
        let out = parse_chunks(
            &tools,
            &[
                "]<]minimax[>[<tool_call>\n]<]minimax[>[<invoke name=\"book_flight\">",
                "\n]<]minimax[>[<destination>Paris]<]minimax[>[</destination>",
                "\n]<]minimax[>[<passengers>2]<]minimax[>[</passengers>",
                "\n]<]minimax[>[<first_class>true]<]minimax[>[</first_class>",
                "\n]<]minimax[>[</invoke>\n]<]minimax[>[</tool_call>",
            ],
        );
        let merged = out.coalesce_calls();
        assert_eq!(merged.calls.len(), 1);
        assert_eq!(
            merged.calls[0].arguments,
            r#"{"destination":"Paris","passengers":2,"first_class":true}"#
        );
    }

    #[test]
    fn nested_value_tags_do_not_shadow_top_level_order() {
        // 7.d-shaped: nested array/object tags inside a value must not be taken
        // for top-level parameter names when re-ordering.
        let tools = vec![Tool {
            name: "process_data".to_string(),
            description: None,
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "items": { "type": "array", "items": { "type": "integer" } },
                    "config": { "type": "object" }
                }
            }),
            strict: None,
        }];
        let out = parse_chunks(
            &tools,
            &[
                "]<]minimax[>[<tool_call>\n]<]minimax[>[<invoke name=\"process_data\">",
                "\n]<]minimax[>[<items>]<]minimax[>[<item>1]<]minimax[>[</item>]<]minimax[>[<item>2]<]minimax[>[</item>]<]minimax[>[</items>",
                "\n]<]minimax[>[<config>]<]minimax[>[<mode>fast]<]minimax[>[</mode>]<]minimax[>[</config>",
                "\n]<]minimax[>[</invoke>\n]<]minimax[>[</tool_call>",
            ],
        );
        let merged = out.coalesce_calls();
        assert_eq!(merged.calls.len(), 1);
        let args: serde_json::Value = serde_json::from_str(&merged.calls[0].arguments).unwrap();
        assert_eq!(args["items"], serde_json::json!([1, 2]));
        assert_eq!(args["config"]["mode"], "fast");
        // Top-level order: items before config.
        assert!(
            merged.calls[0].arguments.find("\"items\"").unwrap()
                < merged.calls[0].arguments.find("\"config\"").unwrap()
        );
    }
}
