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
//! suppression) is owned here. The per-invoke value typing is delegated to the
//! v1 batch parser `try_tool_call_parse_minimax_m3` driven by the same config
//! `dynamo_parsers` uses for batch parsing, so a streamed call matches exactly
//! what the batch parser produces. Arguments are re-serialized in source
//! parameter-tag order because the v1 parser builds them from a `HashMap`
//! whose key order is non-deterministic; the fixtures store the arguments as
//! an exact JSON string, so order is pinned to the model-emitted order.

use std::collections::HashSet;

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

/// Stream parser for MiniMax-M3 tool calls.
pub struct MiniMaxM3ToolStreamParser {
    buffer: String,
    in_block: bool,
    suppress_normal_text: bool,
    next_index: usize,
    config: MiniMaxM3ParserConfig,
    tools: Vec<ToolDefinition>,
}

impl MiniMaxM3ToolStreamParser {
    pub fn new(tools: &[Tool]) -> Self {
        Self {
            buffer: String::new(),
            in_block: false,
            suppress_normal_text: false,
            next_index: 0,
            // Identical to `dynamo_parsers`' batch config so the streamed value
            // typing matches the v1 batch parser exactly.
            config: MiniMaxM3ParserConfig::default(),
            tools: tools.iter().map(ToolDefinition::from).collect(),
        }
    }

    fn drain(&mut self, flush: bool) -> anyhow::Result<ToolParseResult> {
        let mut out = ToolParseResult::default();

        loop {
            if self.in_block {
                // Close the block once no more complete invokes precede its end.
                if let Some(end) = self.buffer.find(BLOCK_END) {
                    let invoke_before_end = self
                        .buffer
                        .find(FUNCTION_START)
                        .is_some_and(|start| start < end);
                    if !invoke_before_end {
                        // Complete block fully closed: drop its markup and resume
                        // keeping natural text (inter-block / trailing). Any later
                        // block re-enters `in_block` and re-suppresses its markup.
                        // Matches the v1 batch parser (cases 8.b/8.c).
                        self.buffer.drain(..end + BLOCK_END.len());
                        self.in_block = false;
                        self.suppress_normal_text = false;
                        continue;
                    }
                }

                let Some(start) = self.buffer.find(FUNCTION_START) else {
                    if flush {
                        tracing::warn!(
                            why = "minimax_m3_block_without_complete_invoke",
                            "MiniMax-M3 stream dropped incomplete block at EOF"
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
                            why = "minimax_m3_incomplete_invoke",
                            "MiniMax-M3 stream dropped incomplete invoke at EOF"
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

            // A recovered bare invoke suppresses its trailing markup; its stray
            // namespaced `</tool_call>` close (cases 5.b/5.f) ENDS that markup context.
            // Consume the orphan close and clear the latch so inter-call text —
            // e.g. the single separator space before the next block — flows
            // through verbatim, matching the v1 jail+batch output.
            // A stray/orphan close (`BLOCK_END`) before any opener is malformed
            // double-close markup. Drop it so it can NEVER leak into normal_text;
            // when suppression is off, first emit the natural text preceding it.
            // Clear the latch either way (the markup context has ended).
            if let Some(pos) = self.buffer.find(BLOCK_END) {
                let next_open = [BLOCK_START, FUNCTION_START]
                    .into_iter()
                    .filter_map(|m| self.buffer.find(m))
                    .min();
                if next_open.is_none_or(|open| pos < open) {
                    if !self.suppress_normal_text && pos > 0 {
                        out.normal_text.push_str(&self.buffer[..pos]);
                    }
                    self.buffer.drain(..pos + BLOCK_END.len());
                    self.suppress_normal_text = false;
                    continue;
                }
            }

            let block_start = self.buffer.find(BLOCK_START);
            let bare_invoke_start = self.buffer.find(FUNCTION_START);
            let next_marker = match (block_start, bare_invoke_start) {
                (Some(b), Some(f)) if b <= f => Some((b, Marker::Block)),
                (Some(_), Some(f)) => Some((f, Marker::BareInvoke)),
                (Some(b), None) => Some((b, Marker::Block)),
                (None, Some(f)) => Some((f, Marker::BareInvoke)),
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
                Marker::BareInvoke => {
                    let Some(end) = self.buffer.find(FUNCTION_END) else {
                        if flush {
                            tracing::warn!(
                                why = "minimax_m3_incomplete_bare_invoke",
                                "MiniMax-M3 stream dropped incomplete bare invoke at EOF"
                            );
                            self.buffer.clear();
                        }
                        break;
                    };
                    let function = self.buffer[..end + FUNCTION_END.len()].to_string();
                    self.buffer.drain(..end + FUNCTION_END.len());
                    if let Some(delta) = self.parse_function_delta(&function)? {
                        tracing::warn!(
                            why = "minimax_m3_bare_invoke_recovery",
                            tool_index = delta.tool_index,
                            "MiniMax-M3 stream recovered a complete bare invoke"
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

    /// Parse one complete `<invoke ...>...</invoke>` run into a delta.
    ///
    /// Wraps the invoke in the M3 `<tool_call>` block so the v1 parser always
    /// takes its normal wrapped path, then re-orders the arguments to source
    /// parameter-tag order.
    fn parse_function_delta(&self, function: &str) -> anyhow::Result<Option<ToolCallDelta>> {
        let wrapped = format!("{BLOCK_START}{function}{BLOCK_END}");
        let tools_opt = (!self.tools.is_empty()).then_some(self.tools.as_slice());
        let (calls, _content) = try_tool_call_parse_minimax_m3(&wrapped, &self.config, tools_opt)?;
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

impl ToolParser for MiniMaxM3ToolStreamParser {
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
    BareInvoke,
}

/// Longest non-empty proper prefix of a start marker that `text` ends with, so a
/// marker split across chunk boundaries is held back instead of leaked as text.
/// Both M3 start markers begin with the namespace token, so this also holds back
/// a split `]<]minimax[>[` run.
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

/// Re-serialize a v1 arguments JSON object in source parameter-tag order.
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
        if let Some(val) = obj.get(&name)
            && seen.insert(name.clone())
        {
            parts.push(format!(
                "{}:{}",
                serde_json::to_string(&name).unwrap_or_default(),
                serde_json::to_string(val).unwrap_or_default()
            ));
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
