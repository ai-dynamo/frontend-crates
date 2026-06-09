// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Streaming DSML parser for DeepSeek V4 tool calls.

use serde_json::{Map, Value};

use crate::tool_calling::traits::{Tool, ToolCallDelta, ToolParseResult, ToolParser};

const BLOCK_START: &str = "<｜DSML｜tool_calls>";
const BLOCK_END: &str = "</｜DSML｜tool_calls>";
const INVOKE_START_PREFIX: &str = "<｜DSML｜invoke name=";
const INVOKE_END: &str = "</｜DSML｜invoke>";
const PARAMETER_PREFIX: &str = "<｜DSML｜parameter name=";
const PARAMETER_END: &str = "</｜DSML｜parameter>";

/// Stream parser for DeepSeek V4 DSML tool calls.
pub struct DeepSeekV4ToolStreamParser {
    buffer: String,
    in_block: bool,
    suppress_normal_text: bool,
    next_index: usize,
}

impl DeepSeekV4ToolStreamParser {
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
            in_block: false,
            suppress_normal_text: false,
            next_index: 0,
        }
    }

    fn drain(&mut self, flush: bool) -> anyhow::Result<ToolParseResult> {
        let mut out = ToolParseResult::default();

        loop {
            if self.in_block {
                if let Some(end) = self.buffer.find(BLOCK_END) {
                    let invoke_before_end = self
                        .buffer
                        .find(INVOKE_START_PREFIX)
                        .is_some_and(|start| start < end);
                    if !invoke_before_end {
                        self.buffer.drain(..end + BLOCK_END.len());
                        self.in_block = false;
                        self.suppress_normal_text = true;
                        continue;
                    }
                }

                let Some(start) = self.buffer.find(INVOKE_START_PREFIX) else {
                    if flush {
                        tracing::warn!(
                            why = "dsv4_block_without_complete_invoke",
                            "DSML stream dropped incomplete block at EOF"
                        );
                        self.buffer.clear();
                        self.in_block = false;
                    }
                    break;
                };
                if start > 0 {
                    self.buffer.drain(..start);
                }
                let Some(end) = self.buffer.find(INVOKE_END) else {
                    if flush {
                        tracing::warn!(
                            why = "dsv4_incomplete_invoke",
                            "DSML stream dropped incomplete invoke at EOF"
                        );
                        self.buffer.clear();
                        self.in_block = false;
                    }
                    break;
                };
                let invoke = self.buffer[..end + INVOKE_END.len()].to_string();
                self.buffer.drain(..end + INVOKE_END.len());
                if let Some(delta) = self.parse_invoke_delta(&invoke)? {
                    out.calls.push(delta);
                    self.next_index += 1;
                    self.suppress_normal_text = true;
                }
                continue;
            }

            let block_start = self.buffer.find(BLOCK_START);
            let bare_invoke_start = self.buffer.find(INVOKE_START_PREFIX);
            let next_marker = match (block_start, bare_invoke_start) {
                (Some(b), Some(i)) if b <= i => Some((b, Marker::Block)),
                (Some(_), Some(i)) => Some((i, Marker::BareInvoke)),
                (Some(b), None) => Some((b, Marker::Block)),
                (None, Some(i)) => Some((i, Marker::BareInvoke)),
                (None, None) => None,
            };

            let Some((start, marker)) = next_marker else {
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
                    let Some(end) = self.buffer.find(INVOKE_END) else {
                        if flush {
                            tracing::warn!(
                                why = "dsv4_incomplete_bare_invoke",
                                "DSML stream dropped incomplete bare invoke at EOF"
                            );
                            self.buffer.clear();
                        }
                        break;
                    };
                    let invoke = self.buffer[..end + INVOKE_END.len()].to_string();
                    self.buffer.drain(..end + INVOKE_END.len());
                    if let Some(delta) = self.parse_invoke_delta(&invoke)? {
                        tracing::warn!(
                            why = "dsv4_bare_invoke_recovery",
                            tool_index = delta.tool_index,
                            "DSML stream recovered a complete bare invoke"
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

    fn parse_invoke_delta(&self, invoke: &str) -> anyhow::Result<Option<ToolCallDelta>> {
        let Some(after_prefix) = invoke.strip_prefix(INVOKE_START_PREFIX) else {
            return Ok(None);
        };
        let Some(after_quote) = after_prefix.strip_prefix('"') else {
            return Ok(None);
        };
        let Some(name_end) = after_quote.find('"') else {
            return Ok(None);
        };
        let name = after_quote[..name_end].trim().to_string();
        let Some(header_end) = after_quote[name_end..].find('>') else {
            return Ok(None);
        };
        let body_start = name_end + header_end + 1;
        let Some(body_end) = after_quote[body_start..].find(INVOKE_END) else {
            return Ok(None);
        };
        let body = &after_quote[body_start..body_start + body_end];
        let arguments = serde_json::to_string(&parse_parameters(body)?)?;
        Ok(Some(ToolCallDelta {
            tool_index: self.next_index,
            name: Some(name),
            arguments,
        }))
    }
}

impl Default for DeepSeekV4ToolStreamParser {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolParser for DeepSeekV4ToolStreamParser {
    fn create(_tools: &[Tool]) -> anyhow::Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new()))
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

fn marker_prefix_suffix_len(text: &str) -> usize {
    [BLOCK_START, INVOKE_START_PREFIX]
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

fn parse_parameters(body: &str) -> anyhow::Result<Map<String, Value>> {
    let mut params = Map::new();
    let mut cursor = 0;
    while let Some(rel_start) = body[cursor..].find(PARAMETER_PREFIX) {
        let start = cursor + rel_start + PARAMETER_PREFIX.len();
        let Some(after_name_quote) = body[start..].strip_prefix('"') else {
            cursor = start;
            continue;
        };
        let Some(name_end) = after_name_quote.find('"') else {
            break;
        };
        let name = after_name_quote[..name_end].trim();
        let attrs_start = start + 1 + name_end + 1;
        let Some(header_end_rel) = body[attrs_start..].find('>') else {
            break;
        };
        let attrs = &body[attrs_start..attrs_start + header_end_rel];
        let value_start = attrs_start + header_end_rel + 1;
        let Some(value_end_rel) = body[value_start..].find(PARAMETER_END) else {
            break;
        };
        let raw_value = body[value_start..value_start + value_end_rel].trim();
        let value = if attrs.contains(r#"string="true""#) {
            Value::String(raw_value.to_string())
        } else {
            serde_json::from_str(raw_value).unwrap_or_else(|_| Value::String(raw_value.to_string()))
        };
        params.insert(name.to_string(), value);
        cursor = value_start + value_end_rel + PARAMETER_END.len();
    }
    Ok(params)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_chunks(chunks: &[&str]) -> ToolParseResult {
        let mut parser = DeepSeekV4ToolStreamParser::new();
        let mut out = ToolParseResult::default();
        for chunk in chunks {
            out.append(parser.push(chunk).expect("push"));
        }
        out.append(parser.finish().expect("finish"));
        out
    }

    #[test]
    fn emits_complete_invoke_on_close() {
        let out = parse_chunks(&[
            "<｜DSML｜tool_calls> <｜DSML｜invoke",
            " name=\"get_weather\">",
            " <｜DSML｜parameter name=\"location\" string=\"true\">",
            "NYC</｜DSML｜parameter> </｜DSML｜invoke>",
            " </｜DSML｜tool_calls>",
        ]);
        assert_eq!(out.normal_text, "");
        assert_eq!(out.calls.len(), 1);
        assert_eq!(out.calls[0].tool_index, 0);
        assert_eq!(out.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(out.calls[0].arguments, r#"{"location":"NYC"}"#);
    }

    #[test]
    fn preserves_prefix_text_before_block() {
        let out = parse_chunks(&[
            "I will",
            " check the weather. <｜DSML｜tool_calls>",
            " <｜DSML｜invoke name=\"get_weather\">",
            " <｜DSML｜parameter name=\"location\" string=\"true\">NYC</｜DSML｜parameter> </｜DSML｜invoke>",
        ]);
        assert_eq!(out.normal_text, "I will check the weather. ");
        assert_eq!(out.calls.len(), 1);
    }

    #[test]
    fn recovers_complete_bare_invoke() {
        let out = parse_chunks(&[
            "I will check that. <｜DSML｜invoke name=\"get_weather\">",
            " <｜DSML｜parameter name=\"location\" string=\"true\">NYC</｜DSML｜parameter>",
            " </｜DSML｜invoke> </｜DSML｜tool_calls>",
        ]);
        assert_eq!(out.normal_text, "I will check that. ");
        assert_eq!(out.calls.len(), 1);
        assert_eq!(out.calls[0].arguments, r#"{"location":"NYC"}"#);
    }

    #[test]
    fn suppresses_incomplete_invoke_at_eof() {
        let out = parse_chunks(&[
            "<｜DSML｜tool_calls> <｜DSML｜invoke name=\"get_weather\">",
            " <｜DSML｜parameter name=\"location\" string=\"true\">NY",
        ]);
        assert_eq!(out.normal_text, "");
        assert!(out.calls.is_empty());
    }
}
