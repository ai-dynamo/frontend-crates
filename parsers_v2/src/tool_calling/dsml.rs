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
///
/// Conforms to the v1 batch baseline: a tool call is emitted only once its
/// `</｜DSML｜invoke>` has streamed. A call truncated mid-call (header seen but no
/// closing marker by end of stream) is DROPPED rather than emitted with empty
/// arguments, matching the v1 DSML parser's "drop truncated-mid-value" rule. The
/// `name` delta is emitted immediately before the `arguments` delta (both once
/// the call closes), so consumers still coalesce the two by `tool_index` — the
/// same wire shape Harmony uses (`name`-first, then arguments fragment).
///
/// `normal_text` keeps natural-language text from COMPLETE tool-call blocks
/// verbatim — prefix before the first block, text BETWEEN blocks, and text
/// AFTER the last block — and strips only the block markup. This matches the v1
/// batch DSML parser's "remove the complete-block markup spans, keep surrounding
/// text" rule (cases 2.b/2.c/8.b/8.c/8.d).
///
/// `suppress_normal_text` is the latch that distinguishes the two contracts:
/// it is cleared when a COMPLETE block closes (so the next inter/trailing text
/// flows through) but stays LATCHED through degraded recovery (bare invoke /
/// orphan markers), preserving the v1 batch parser's drop-without-leak behavior
/// for malformed/unrecoverable input.
pub struct DeepSeekV4ToolStreamParser {
    buffer: String,
    in_block: bool,
    suppress_normal_text: bool,
    next_index: usize,
    /// Set to `Some((tool_index, name))` after the invoke header has streamed,
    /// while we buffer the body and wait for `</｜DSML｜invoke>`. Nothing is
    /// emitted until the close arrives; if it never does, the call is dropped.
    open_invoke: Option<(usize, String)>,
}

impl DeepSeekV4ToolStreamParser {
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
            in_block: false,
            suppress_normal_text: false,
            next_index: 0,
            open_invoke: None,
        }
    }

    fn drain(&mut self, flush: bool) -> anyhow::Result<ToolParseResult> {
        let mut out = ToolParseResult::default();

        loop {
            // The invoke header has streamed; buffer the body until the closing
            // marker arrives, then emit `name` + `arguments` together. If the
            // close never arrives (truncated mid-call), drop the call to match
            // the v1 batch parser instead of emitting empty arguments.
            if let Some((tool_index, name)) = self.open_invoke.clone() {
                let Some(end) = self.buffer.find(INVOKE_END) else {
                    if flush {
                        tracing::warn!(
                            why = "dsv4_incomplete_invoke",
                            tool_index,
                            "DSML stream truncated before </invoke>; dropping the call (v1 parity)"
                        );
                        self.buffer.clear();
                        self.in_block = false;
                        self.open_invoke = None;
                    }
                    break;
                };
                let body = self.buffer[..end].to_string();
                self.buffer.drain(..end + INVOKE_END.len());
                let arguments = serde_json::to_string(&parse_parameters(&body)?)?;
                // Complete call: emit the name delta, then the arguments delta
                // (coalesced by tool_index) — the call only appears once known
                // complete.
                out.calls.push(ToolCallDelta {
                    tool_index,
                    name: Some(name),
                    arguments: String::new(),
                });
                out.calls.push(ToolCallDelta {
                    tool_index,
                    name: None,
                    arguments,
                });
                self.next_index += 1;
                self.open_invoke = None;
                continue;
            }

            if self.in_block {
                if let Some(end) = self.buffer.find(BLOCK_END) {
                    let invoke_before_end = self
                        .buffer
                        .find(INVOKE_START_PREFIX)
                        .is_some_and(|start| start < end);
                    if !invoke_before_end {
                        // Complete block fully closed: drop its markup and resume
                        // keeping natural text (inter-block / trailing). Any later
                        // block re-enters `in_block` and re-suppresses its markup.
                        self.buffer.drain(..end + BLOCK_END.len());
                        self.in_block = false;
                        self.suppress_normal_text = false;
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
                if self.open_invoke_header()?.is_none() {
                    // Header not fully streamed yet; wait for more input.
                    if flush {
                        tracing::warn!(
                            why = "dsv4_incomplete_invoke",
                            "DSML stream dropped incomplete invoke header at EOF"
                        );
                        self.buffer.clear();
                        self.in_block = false;
                    }
                    break;
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
                // Outside any block / open invoke: keep natural text verbatim
                // unless suppression is latched (degraded recovery). Retain a
                // trailing partial-marker so we don't emit half a fence mid-stream.
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
                // Text before the next marker: natural text (prefix / inter-block /
                // trailing), kept verbatim unless suppression is latched.
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
                Marker::BareInvoke => match self.open_invoke_header()? {
                    Some(tool_index) => {
                        // Degraded recovery: latch suppression so the orphan
                        // markup tail around a bare invoke is dropped, not leaked
                        // (matches the v1 batch recovery contract).
                        self.suppress_normal_text = true;
                        tracing::warn!(
                            why = "dsv4_bare_invoke_recovery",
                            tool_index,
                            "DSML stream recovering a bare invoke (buffered until close)"
                        );
                    }
                    None => {
                        if flush {
                            tracing::warn!(
                                why = "dsv4_incomplete_bare_invoke",
                                "DSML stream dropped incomplete bare invoke at EOF"
                            );
                            self.buffer.clear();
                        }
                        break;
                    }
                },
            }
        }

        Ok(out)
    }

    /// Given `self.buffer` positioned at an `INVOKE_START_PREFIX`, parse the
    /// invoke header. If the header is complete (its closing `>` has streamed),
    /// consume the header bytes, buffer the `(tool_index, name)` in `open_invoke`
    /// WITHOUT emitting yet, and return `Some(tool_index)`. The name + arguments
    /// are emitted together once `</｜DSML｜invoke>` arrives, so a call that never
    /// closes is dropped (v1 parity). If the header is still partial, leave the
    /// buffer intact and return `None` so the caller can wait for more input.
    fn open_invoke_header(&mut self) -> anyhow::Result<Option<usize>> {
        let Some((name, header_len)) = parse_invoke_header(&self.buffer) else {
            return Ok(None);
        };
        self.buffer.drain(..header_len);
        let tool_index = self.next_index;
        self.open_invoke = Some((tool_index, name));
        Ok(Some(tool_index))
    }
}

/// Parse a complete invoke header `<｜DSML｜invoke name="X">` from the front of
/// `s`. Returns `(name, header_byte_len)` where `header_byte_len` covers through
/// the closing `>`. Returns `None` if the header has not fully streamed yet.
fn parse_invoke_header(s: &str) -> Option<(String, usize)> {
    let after_prefix = s.strip_prefix(INVOKE_START_PREFIX)?;
    let after_quote = after_prefix.strip_prefix('"')?;
    let name_end = after_quote.find('"')?;
    let name = after_quote[..name_end].trim().to_string();
    let rest = &after_quote[name_end + 1..];
    let gt = rest.find('>')?;
    let header_len = INVOKE_START_PREFIX.len() + 1 + name_end + 1 + gt + 1;
    Some((name, header_len))
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
    fn emits_name_then_args_on_close() {
        // Name and arguments are both emitted once `</invoke>` arrives, as two
        // deltas (name-only, then arguments-only) coalesced by tool_index.
        let out = parse_chunks(&[
            "<｜DSML｜tool_calls> <｜DSML｜invoke",
            " name=\"get_weather\">",
            " <｜DSML｜parameter name=\"location\" string=\"true\">",
            "NYC</｜DSML｜parameter> </｜DSML｜invoke>",
            " </｜DSML｜tool_calls>",
        ]);
        assert_eq!(out.normal_text, "");
        // Two deltas: name-only first, then arguments-only on close.
        assert_eq!(out.calls.len(), 2);
        assert_eq!(out.calls[0].tool_index, 0);
        assert_eq!(out.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(out.calls[0].arguments, "");
        assert_eq!(out.calls[1].tool_index, 0);
        assert_eq!(out.calls[1].name, None);
        assert_eq!(out.calls[1].arguments, r#"{"location":"NYC"}"#);

        // Coalesced wire shape matches the complete call.
        let merged = out.coalesce_calls();
        assert_eq!(merged.calls.len(), 1);
        assert_eq!(merged.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(merged.calls[0].arguments, r#"{"location":"NYC"}"#);
    }

    #[test]
    fn emits_name_before_any_arguments() {
        // The name delta must precede the arguments delta on the wire.
        let out = parse_chunks(&[
            "<｜DSML｜tool_calls> <｜DSML｜invoke name=\"get_weather\">",
            " <｜DSML｜parameter name=\"location\" string=\"true\">NYC</｜DSML｜parameter> </｜DSML｜invoke>",
        ]);
        let first_named = out.calls.iter().position(|c| c.name.is_some());
        let first_args = out.calls.iter().position(|c| !c.arguments.is_empty());
        assert_eq!(first_named, Some(0));
        assert!(first_named <= first_args, "name must stream before args");
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
        assert_eq!(out.coalesce_calls().calls.len(), 1);
    }

    #[test]
    fn recovers_complete_bare_invoke() {
        let out = parse_chunks(&[
            "I will check that. <｜DSML｜invoke name=\"get_weather\">",
            " <｜DSML｜parameter name=\"location\" string=\"true\">NYC</｜DSML｜parameter>",
            " </｜DSML｜invoke> </｜DSML｜tool_calls>",
        ]);
        assert_eq!(out.normal_text, "I will check that. ");
        let merged = out.coalesce_calls();
        assert_eq!(merged.calls.len(), 1);
        assert_eq!(merged.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(merged.calls[0].arguments, r#"{"location":"NYC"}"#);
    }

    #[test]
    fn drops_invoke_truncated_after_header() {
        // The header closed (name seen) but the parameter body never closes
        // before EOF. v1 parity: drop the call entirely rather than emit it
        // with empty arguments.
        let out = parse_chunks(&[
            "<｜DSML｜tool_calls> <｜DSML｜invoke name=\"get_weather\">",
            " <｜DSML｜parameter name=\"location\" string=\"true\">NY",
        ]);
        assert_eq!(out.normal_text, "");
        assert!(
            out.calls.is_empty(),
            "truncated-mid-value call must be dropped"
        );
    }

    #[test]
    fn drops_last_call_when_truncated_keeps_earlier_complete_call() {
        // Multi-call, last call truncated mid-arg-value (conformance 5.e): the
        // first complete call survives, the truncated second is dropped.
        let out = parse_chunks(&[
            "I'll check both. <｜DSML｜tool_calls>",
            " <｜DSML｜invoke name=\"get_weather\"> <｜DSML｜parameter name=\"location\" string=\"true\">Boston</｜DSML｜parameter> </｜DSML｜invoke>",
            " <｜DSML｜invoke name=\"get_weather\"> <｜DSML｜parameter name=\"location\" string=\"true\">New York",
        ]);
        let merged = out.coalesce_calls();
        assert_eq!(merged.calls.len(), 1, "truncated 2nd call dropped");
        assert_eq!(merged.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(merged.calls[0].arguments, r#"{"location":"Boston"}"#);
    }

    #[test]
    fn suppresses_invoke_header_truncated_mid_header() {
        // The header itself never closes, so nothing is emitted at all.
        let out = parse_chunks(&["<｜DSML｜tool_calls> <｜DSML｜invoke name=\"get_weat"]);
        assert_eq!(out.normal_text, "");
        assert!(out.calls.is_empty());
    }
}
