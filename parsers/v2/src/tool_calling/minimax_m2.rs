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
//! suppression) is owned here. The per-block value typing is delegated to the v1
//! batch XML parser `try_tool_call_parse_xml` driven by the same MiniMax config
//! `dynamo_parsers` uses for batch parsing, so a streamed call matches exactly
//! what the batch parser produces. Arguments are re-serialized in source
//! `<parameter name="...">` order because the v1 parser builds them from a
//! `HashMap` whose key order is non-deterministic; the fixtures store the
//! arguments as an exact JSON string, so order is pinned to the model-emitted
//! order (the order vLLM's Rust parser also preserves).

use std::collections::HashSet;

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

/// Stream parser for MiniMax-M2 XML tool calls.
pub struct MiniMaxM2ToolStreamParser {
    buffer: String,
    in_block: bool,
    suppress_normal_text: bool,
    next_index: usize,
    config: XmlParserConfig,
    tools: Vec<ToolDefinition>,
}

impl MiniMaxM2ToolStreamParser {
    pub fn new(tools: &[Tool]) -> Self {
        Self {
            buffer: String::new(),
            in_block: false,
            suppress_normal_text: false,
            next_index: 0,
            config: minimax_config(),
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
                        // Matches the v1 batch parser (cases 8.b/8.c/8.d).
                        self.buffer.drain(..end + BLOCK_END.len());
                        self.in_block = false;
                        self.suppress_normal_text = false;
                        continue;
                    }
                }

                let Some(start) = self.buffer.find(FUNCTION_START) else {
                    if flush {
                        tracing::warn!(
                            why = "minimax_m2_block_without_complete_invoke",
                            "MiniMax-M2 stream dropped incomplete block at EOF"
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
                            why = "minimax_m2_incomplete_invoke",
                            "MiniMax-M2 stream dropped incomplete invoke at EOF"
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
            // `</minimax:tool_call>` close (cases 5.b/5.f) ENDS that markup context.
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
                                why = "minimax_m2_incomplete_bare_invoke",
                                "MiniMax-M2 stream dropped incomplete bare invoke at EOF"
                            );
                            self.buffer.clear();
                        }
                        break;
                    };
                    let function = self.buffer[..end + FUNCTION_END.len()].to_string();
                    self.buffer.drain(..end + FUNCTION_END.len());
                    if let Some(delta) = self.parse_function_delta(&function)? {
                        tracing::warn!(
                            why = "minimax_m2_bare_invoke_recovery",
                            tool_index = delta.tool_index,
                            "MiniMax-M2 stream recovered a complete bare invoke"
                        );
                        out.calls.push(delta);
                        self.next_index += 1;
                        // Do NOT latch suppression after a bare invoke: when the
                        // optional outer close (`BLOCK_END`) is absent, later
                        // narration (e.g. ` Done.`) must still reach normal_text.
                        // A stray `BLOCK_END` that DOES follow is stripped by the
                        // orphan-close handling above, so the close never leaks.
                        self.suppress_normal_text = false;
                    }
                }
            }
        }

        Ok(out)
    }

    /// Parse one complete `<invoke name="...">...</invoke>` block into a delta.
    ///
    /// Wraps the invoke in `<minimax:tool_call>` so the v1 parser always takes
    /// its normal wrapped path, then re-orders the arguments to source order.
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

impl ToolParser for MiniMaxM2ToolStreamParser {
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

/// Longest non-empty proper prefix of a marker that `text` ends with, so a
/// marker split across chunk boundaries is held back instead of leaked as text.
/// `BLOCK_END` is included: a lone orphan `</minimax:tool_call>` that arrives
/// split across chunks must be retained whole so the orphan-close path (which
/// strips it and never lets it leak) can match it — otherwise the partial suffix
/// is emitted as normal_text and the marker leaks.
fn marker_prefix_suffix_len(text: &str) -> usize {
    [BLOCK_START, FUNCTION_START, BLOCK_END]
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

/// Re-serialize a v1 arguments JSON object in source `<parameter name="...">`
/// order.
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
        // seen.insert guards a REPEATED parameter tag (same name twice in the
        // source): the v1 object holds one value per key, so emit it once.
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
