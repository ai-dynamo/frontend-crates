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
//! suppression) is owned here. The per-block value typing is delegated to the v1
//! batch XML parser `try_tool_call_parse_xml`, so a streamed call matches exactly
//! what the batch parser produces (the DIS-2209 bar). Arguments are re-serialized
//! in the source parameter order because the v1 parser builds them from a
//! `HashMap` whose key order is non-deterministic; streaming fixtures store the
//! arguments as an exact JSON string, so order has to be pinned to the
//! model-emitted order (the order vLLM's Rust parser also preserves).

use std::collections::HashSet;

use dynamo_parsers::tool_calling::{ToolDefinition, XmlParserConfig, try_tool_call_parse_xml};

use crate::tool_calling::traits::{Tool, ToolCallDelta, ToolParseResult, ToolParser};

const BLOCK_START: &str = "<tool_call>";
const BLOCK_END: &str = "</tool_call>";
const FUNCTION_START: &str = "<function=";
const FUNCTION_END: &str = "</function>";
const PARAMETER_START: &str = "<parameter=";

/// Stream parser for Qwen3-Coder XML tool calls.
pub struct Qwen3CoderToolStreamParser {
    buffer: String,
    in_block: bool,
    suppress_normal_text: bool,
    next_index: usize,
    config: XmlParserConfig,
    tools: Vec<ToolDefinition>,
}

impl Qwen3CoderToolStreamParser {
    pub fn new(tools: &[Tool]) -> Self {
        Self {
            buffer: String::new(),
            in_block: false,
            suppress_normal_text: false,
            next_index: 0,
            config: XmlParserConfig::default(),
            tools: tools
                .iter()
                .map(|t| ToolDefinition {
                    name: t.name.clone(),
                    parameters: Some(t.parameters.clone()),
                    strict: t.strict,
                })
                .collect(),
        }
    }

    fn drain(&mut self, flush: bool) -> anyhow::Result<ToolParseResult> {
        let mut out = ToolParseResult::default();

        loop {
            if self.in_block {
                // Close the block once no more complete functions precede its end.
                if let Some(end) = self.buffer.find(BLOCK_END) {
                    let function_before_end = self
                        .buffer
                        .find(FUNCTION_START)
                        .is_some_and(|start| start < end);
                    if !function_before_end {
                        self.buffer.drain(..end + BLOCK_END.len());
                        self.in_block = false;
                        self.suppress_normal_text = true;
                        continue;
                    }
                }

                let Some(start) = self.buffer.find(FUNCTION_START) else {
                    if flush {
                        tracing::warn!(
                            why = "qwen3_coder_block_without_complete_function",
                            "Qwen3-Coder stream dropped incomplete block at EOF"
                        );
                        self.buffer.clear();
                        self.in_block = false;
                    }
                    break;
                };
                if start > 0 {
                    self.buffer.drain(..start);
                }
                let Some(end) = self.buffer.find(FUNCTION_END) else {
                    if flush {
                        tracing::warn!(
                            why = "qwen3_coder_incomplete_function",
                            "Qwen3-Coder stream dropped incomplete function at EOF"
                        );
                        self.buffer.clear();
                        self.in_block = false;
                    }
                    break;
                };
                let function = self.buffer[..end + FUNCTION_END.len()].to_string();
                self.buffer.drain(..end + FUNCTION_END.len());
                if let Some(delta) = self.parse_function_delta(&function)? {
                    out.calls.push(delta);
                    self.next_index += 1;
                    self.suppress_normal_text = true;
                }
                continue;
            }

            let block_start = self.buffer.find(BLOCK_START);
            let bare_function_start = self.buffer.find(FUNCTION_START);
            let next_marker = match (block_start, bare_function_start) {
                (Some(b), Some(f)) if b <= f => Some((b, Marker::Block)),
                (Some(_), Some(f)) => Some((f, Marker::BareFunction)),
                (Some(b), None) => Some((b, Marker::Block)),
                (None, Some(f)) => Some((f, Marker::BareFunction)),
                (None, None) => None,
            };

            let Some((start, marker)) = next_marker else {
                // No marker present: emit buffered text, but hold back a trailing
                // partial marker (split across this chunk boundary) unless flushing.
                let keep = if flush {
                    0
                } else {
                    marker_prefix_suffix_len(&self.buffer)
                };
                let emit_len = self.buffer.len().saturating_sub(keep);
                if emit_len > 0 {
                    if !self.suppress_normal_text {
                        out.normal_text.push_str(&self.buffer[..emit_len]);
                    }
                    self.buffer.drain(..emit_len);
                }
                break;
            };

            if start > 0 {
                if !self.suppress_normal_text {
                    out.normal_text.push_str(&self.buffer[..start]);
                }
                self.buffer.drain(..start);
            }

            match marker {
                Marker::Block => {
                    self.buffer.drain(..BLOCK_START.len());
                    self.in_block = true;
                    self.suppress_normal_text = true;
                }
                Marker::BareFunction => {
                    let Some(end) = self.buffer.find(FUNCTION_END) else {
                        if flush {
                            tracing::warn!(
                                why = "qwen3_coder_incomplete_bare_function",
                                "Qwen3-Coder stream dropped incomplete bare function at EOF"
                            );
                            self.buffer.clear();
                        }
                        break;
                    };
                    let function = self.buffer[..end + FUNCTION_END.len()].to_string();
                    self.buffer.drain(..end + FUNCTION_END.len());
                    if let Some(delta) = self.parse_function_delta(&function)? {
                        tracing::warn!(
                            why = "qwen3_coder_bare_function_recovery",
                            tool_index = delta.tool_index,
                            "Qwen3-Coder stream recovered a complete bare function"
                        );
                        out.calls.push(delta);
                        self.next_index += 1;
                        self.suppress_normal_text = true;
                    }
                }
            }
        }

        Ok(out)
    }

    /// Parse one complete `<function=...></function>` block into a delta.
    ///
    /// Wraps the function in `<tool_call>` so the v1 parser always takes its
    /// normal wrapped path, then re-orders the arguments to source order.
    fn parse_function_delta(&self, function: &str) -> anyhow::Result<Option<ToolCallDelta>> {
        let wrapped = format!("{BLOCK_START}{function}{BLOCK_END}");
        let (calls, _content) = try_tool_call_parse_xml(&wrapped, &self.config, Some(&self.tools))?;
        let Some(call) = calls.into_iter().next() else {
            return Ok(None);
        };
        let arguments = reorder_arguments(&call.function.arguments, function);
        Ok(Some(ToolCallDelta {
            tool_index: self.next_index,
            name: Some(call.function.name),
            arguments,
        }))
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
        self.buffer.push_str(chunk);
        self.drain(false)
    }

    fn finish(&mut self) -> anyhow::Result<ToolParseResult> {
        self.drain(true)
    }
}

#[derive(Clone, Copy)]
enum Marker {
    Block,
    BareFunction,
}

/// Longest non-empty proper prefix of a start marker that `text` ends with, so a
/// marker split across chunk boundaries is held back instead of leaked as text.
fn marker_prefix_suffix_len(text: &str) -> usize {
    [BLOCK_START, FUNCTION_START]
        .into_iter()
        .filter_map(|marker| {
            marker
                .char_indices()
                .map(|(idx, _)| idx)
                .filter(|idx| *idx > 0)
                .filter(|idx| *idx < marker.len())
                .rev()
                .find(|&len| text.ends_with(&marker[..len]))
        })
        .max()
        .unwrap_or(0)
}

/// Re-serialize a v1 arguments JSON object in source `<parameter=...>` order.
fn reorder_arguments(arguments: &str, function: &str) -> String {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(arguments) else {
        return arguments.to_string();
    };
    let Some(obj) = value.as_object() else {
        return arguments.to_string();
    };
    let mut parts: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for name in source_parameter_order(function) {
        if let Some(val) = obj.get(&name) {
            parts.push(format!(
                "{}:{}",
                serde_json::to_string(&name).unwrap_or_default(),
                serde_json::to_string(val).unwrap_or_default()
            ));
            seen.insert(name);
        }
    }
    // Append any keys not matched in source order (defensive; normally empty).
    for (key, val) in obj {
        if !seen.contains(key) {
            parts.push(format!(
                "{}:{}",
                serde_json::to_string(key).unwrap_or_default(),
                serde_json::to_string(val).unwrap_or_default()
            ));
        }
    }
    format!("{{{}}}", parts.join(","))
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
