// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Streaming XML tool-call parser for Qwen3-Coder.
//!
//! Qwen3-Coder emits tool calls as
//!   `<tool_call> <function=NAME> <parameter=KEY>value</parameter> ... </function> </tool_call>`
//! plus a bare `<function=...></function>` back-off form when the outer wrapper
//! is absent (shared with nemotron_nano).
//!
//! The streaming concern (buffering, chunk-split marker safety, normal_text
//! suppression) is owned by the shared [`scan::WrappedBlockScanner`]. The
//! per-block value typing is delegated to the v1 batch XML parser
//! `try_tool_call_parse_xml`, so a streamed call matches exactly what the batch
//! parser produces (the DIS-2209 bar). Arguments are re-serialized in the
//! source parameter order because the v1 parser builds them from a `HashMap`
//! whose key order is non-deterministic; streaming fixtures store the arguments
//! as an exact JSON string, so order has to be pinned to the model-emitted
//! order (the order vLLM's Rust parser also preserves).

use crate::tool_calling::scan::{
    BareRecoveryLatch, InvokeEmitter, InvokeLatch, WrappedBlockScanner, WrappedBlockSink,
    WrappedBlockSpec, reorder_arguments,
};
use crate::tool_calling::v1core::{ToolDefinition, XmlParserConfig, try_tool_call_parse_xml};

use crate::tool_calling::traits::{Tool, ToolCallDelta, ToolParseResult, ToolParser};

const BLOCK_START: &str = "<tool_call>";
const BLOCK_END: &str = "</tool_call>";
const FUNCTION_START: &str = "<function=";
const FUNCTION_END: &str = "</function>";
const PARAMETER_START: &str = "<parameter=";

fn spec() -> WrappedBlockSpec {
    WrappedBlockSpec {
        family: "qwen3_coder",
        block_starts: vec![BLOCK_START.to_string()],
        block_ends: vec![BLOCK_END.to_string()],
        invoke_start: FUNCTION_START.to_string(),
        invoke_end: FUNCTION_END.to_string(),
        orphan_markers: vec![BLOCK_END.to_string()],
        // BLOCK_END is held back too so a split stray/orphan close (consumed
        // and dropped by the orphan-close handler once complete) never emits
        // its first half as text.
        holdback_markers: vec![
            BLOCK_START.to_string(),
            BLOCK_END.to_string(),
            FUNCTION_START.to_string(),
        ],
        bare_recovery_latch: BareRecoveryLatch::Set,
        invoke_latch: InvokeLatch::IfEmitted,
        drop_invoke_crossing_block_end: false,
    }
}

/// Value-typing hook: wraps one complete `<function=...></function>` block in
/// `<tool_call>` so the v1 parser takes its normal wrapped path, then re-orders
/// the arguments to source order.
struct Qwen3Emitter {
    config: XmlParserConfig,
    tools: Vec<ToolDefinition>,
}

impl InvokeEmitter for Qwen3Emitter {
    fn parse_invoke(
        &self,
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
        }))
    }
}

/// Stream parser for Qwen3-Coder XML tool calls.
pub struct Qwen3CoderToolStreamParser {
    scanner: WrappedBlockScanner<Qwen3Emitter>,
}

impl Qwen3CoderToolStreamParser {
    pub fn new(tools: &[Tool]) -> Self {
        Self {
            scanner: WrappedBlockScanner::new(
                spec(),
                Qwen3Emitter {
                    config: XmlParserConfig::default(),
                    tools: tools.iter().map(ToolDefinition::from).collect(),
                },
            ),
        }
    }

    pub(crate) fn push_into<S: WrappedBlockSink>(
        &mut self,
        chunk: &str,
        output: &mut S,
    ) -> anyhow::Result<()> {
        self.scanner.push_into(chunk, output)
    }

    pub(crate) fn finish_into<S: WrappedBlockSink>(
        &mut self,
        output: &mut S,
    ) -> anyhow::Result<()> {
        self.scanner.finish_into(output)
    }
}

impl ToolParser for Qwen3CoderToolStreamParser {
    fn create(tools: &[Tool]) -> anyhow::Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    fn preserve_special_tokens(&self) -> bool {
        true
    }

    fn push(&mut self, chunk: &str) -> anyhow::Result<ToolParseResult> {
        self.scanner.push(chunk)
    }

    fn finish(&mut self) -> anyhow::Result<ToolParseResult> {
        self.scanner.finish()
    }
}

/// Parameter names in the order they appear in a function block.
fn source_parameter_order(function: &str) -> Vec<String> {
    let mut names = Vec::new();
    let mut cursor = 0;
    while let Some(rel) = function[cursor..].find(PARAMETER_START) {
        let start = cursor + rel + PARAMETER_START.len();
        let Some(header_end) = function[start..].find('>') else {
            break;
        };
        let name = function[start..start + header_end]
            .trim()
            .trim_matches('"')
            .trim();
        if !name.is_empty() {
            names.push(name.to_string());
        }
        cursor = start + header_end + 1;
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
        let mut parser = Qwen3CoderToolStreamParser::new(tools);
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
                "<tool_call> <function=get_weather>",
                " <parameter=location>",
                " NYC </parameter> </function>",
                " </tool_call>",
            ],
        );
        assert_eq!(out.normal_text, "");
        assert_eq!(out.calls.len(), 1);
        assert_eq!(out.calls[0].tool_index, 0);
        assert_eq!(out.calls[0].name.as_deref(), Some("get_weather"));
        // Value is schema-typed (string) and trimmed, matching the v1 batch parser.
        assert_eq!(out.calls[0].arguments, r#"{"location":"NYC"}"#);
    }

    #[test]
    fn preserves_prefix_text_before_block() {
        let out = parse_chunks(
            &weather_tools(),
            &[
                "I will",
                " check the weather. <tool_call>",
                " <function=get_weather>",
                " <parameter=location>NYC</parameter> </function> </tool_call>",
            ],
        );
        assert_eq!(out.normal_text, "I will check the weather. ");
        assert_eq!(out.calls.len(), 1);
    }

    #[test]
    fn recovers_complete_bare_function() {
        let out = parse_chunks(
            &weather_tools(),
            &[
                "I will check that. <function=get_weather>",
                " <parameter=location>NYC</parameter>",
                " </function>",
            ],
        );
        assert_eq!(out.normal_text, "I will check that. ");
        assert_eq!(out.calls.len(), 1);
        assert_eq!(out.calls[0].arguments, r#"{"location":"NYC"}"#);
    }

    #[test]
    fn preserves_trailing_text_after_block() {
        // 8.b: trailing narration after a complete block flows into normal_text.
        let out = parse_chunks(
            &weather_tools(),
            &[
                "<tool_call> <function=get_weather> <parameter=location>NYC</parameter> </function> </tool_call>",
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
                "I will check the weather. <tool_call> <function=get_weather> <parameter=location>NYC</parameter> </function> </tool_call>",
                " Then check LA weather. <tool_call> <function=get_weather> <parameter=location>LA</parameter> </function> </tool_call>",
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
    fn suppresses_incomplete_function_at_eof() {
        let out = parse_chunks(
            &weather_tools(),
            &[
                "<tool_call> <function=get_weather>",
                " <parameter=location> NY",
            ],
        );
        assert_eq!(out.normal_text, "");
        assert!(out.calls.is_empty());
    }

    #[test]
    fn holds_back_split_orphan_close() {
        // A stray/orphan `</tool_call>` split across a chunk boundary with no tool
        // call open must NOT leak its first half ("</tool") into normal_text: the
        // partial close is held back until the next chunk completes the marker, at
        // which point the orphan-close handler drops it entirely.
        let out = parse_chunks(&weather_tools(), &["done </tool", "_call> ok"]);
        assert!(out.calls.is_empty());
        assert!(
            !out.normal_text.contains('<'),
            "markup fragment leaked into normal_text: {:?}",
            out.normal_text
        );
        assert_eq!(out.normal_text, "done  ok");
    }

    #[test]
    fn preserves_source_parameter_order() {
        // path, old_str, new_str, command is deliberately NOT alphabetical: the
        // serialized arguments must keep the model-emitted parameter order.
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
                "<tool_call> <function=file_editor>",
                " <parameter=path>/app/x.go</parameter>",
                " <parameter=old_str>foo</parameter>",
                " <parameter=new_str>bar</parameter>",
                " <parameter=command>str_replace</parameter>",
                " </function> </tool_call>",
            ],
        );
        assert_eq!(out.calls.len(), 1);
        assert_eq!(
            out.calls[0].arguments,
            r#"{"path":"/app/x.go","old_str":"foo","new_str":"bar","command":"str_replace"}"#
        );
    }
}
