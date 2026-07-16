// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Streaming tool-call parser for Kimi K2.
//!
//! Kimi K2 emits tool calls as
//!   `<|tool_calls_section_begin|>`
//!     `<|tool_call_begin|>functions.NAME:IDX<|tool_call_argument_begin|>{JSON}<|tool_call_end|>`
//!     ... (one or more calls)
//!   `<|tool_calls_section_end|>`
//! The model may also emit singular section variants
//! (`<|tool_call_section_begin|>` / `<|tool_call_section_end|>`), and may drop
//! `section_end` entirely on max_tokens / EOS truncation.
//!
//! The streaming concern (buffering, chunk-split marker safety, normal_text
//! suppression) is owned here. The per-call typing (function-id parsing, JSON
//! validation, raw-string fallback for malformed args) is delegated to the v1
//! batch parser `try_tool_call_parse_kimi_k2` driven by the same
//! `KimiK2ParserConfig` `dynamo_parsers` uses for batch parsing, so a streamed
//! call matches exactly what the batch parser produces. A complete call is
//! wrapped in the section markers before delegating so the v1 parser always
//! takes its normal section path.
//!
//! The per-call arguments are already a JSON object string, so no key-order
//! reserialization is needed (unlike the XML families): the v1 parser
//! round-trips compact JSON byte-for-byte and falls back to the raw string for
//! malformed payloads, which is exactly what the fixtures expect.

use crate::tool_calling::v1core::{
    KimiK2ParserConfig, ToolDefinition, try_tool_call_parse_kimi_k2,
};

use crate::tool_calling::traits::{Tool, ToolCallDelta, ToolParseResult, ToolParser};

/// Stream parser for Kimi K2 tool calls.
pub struct KimiK2ToolStreamParser {
    buffer: String,
    in_section: bool,
    suppress_normal_text: bool,
    next_index: usize,
    config: KimiK2ParserConfig,
    tools: Vec<ToolDefinition>,
    /// All start markers (section variants + bare call start) that can open a
    /// tool region, plus the inner markers, so a marker split across a chunk
    /// boundary is held back instead of leaked as `normal_text`.
    markers: Vec<String>,
}

impl KimiK2ToolStreamParser {
    pub fn new(tools: &[Tool]) -> Self {
        let config = KimiK2ParserConfig::default();
        // Every grammar marker that must never be split-leaked as normal_text.
        let mut markers: Vec<String> = config.section_start_variants.clone();
        markers.extend(config.section_end_variants.clone());
        markers.push(config.call_start.clone());
        markers.push(config.call_end.clone());
        markers.push(config.argument_begin.clone());
        Self {
            buffer: String::new(),
            in_section: false,
            suppress_normal_text: false,
            next_index: 0,
            config,
            tools: tools.iter().map(ToolDefinition::from).collect(),
            markers,
        }
    }

    /// First occurrence of any section-start variant in `self.buffer`, returning
    /// `(position, matched_token_len)`.
    fn find_section_start(&self) -> Option<(usize, usize)> {
        self.config
            .section_start_variants
            .iter()
            .filter_map(|v| self.buffer.find(v.as_str()).map(|p| (p, v.len())))
            .min_by_key(|(p, _)| *p)
    }

    /// First occurrence of any section-end variant in `self.buffer` at/after
    /// `from`, returning `(absolute_position, matched_token_len)`.
    fn find_section_end_from(&self, from: usize) -> Option<(usize, usize)> {
        self.config
            .section_end_variants
            .iter()
            .filter_map(|v| {
                self.buffer[from..]
                    .find(v.as_str())
                    .map(|p| (from + p, v.len()))
            })
            .min_by_key(|(p, _)| *p)
    }

    /// Earliest COMPLETE orphan close/inner marker in `self.buffer` — a
    /// `call_end`, an `argument_begin`, or any `section_end` variant — returning
    /// `(position, matched_token_len)`. These only appear legitimately inside an
    /// open section; outside one they are stray grammar markup to be stripped.
    /// Mirrors the v1 batch parser's `first_orphan_kimi_marker_index` (minus
    /// `call_start`, which the bare-call recovery path already opens).
    fn find_orphan_close(&self) -> Option<(usize, usize)> {
        [
            self.config.call_end.as_str(),
            self.config.argument_begin.as_str(),
        ]
        .into_iter()
        .chain(self.config.section_end_variants.iter().map(String::as_str))
        .filter_map(|m| self.buffer.find(m).map(|p| (p, m.len())))
        .min_by_key(|(p, _)| *p)
    }

    fn drain(&mut self, flush: bool) -> anyhow::Result<ToolParseResult> {
        let mut out = ToolParseResult::default();

        loop {
            if self.in_section {
                let call_start = self.buffer.find(self.config.call_start.as_str());
                let section_end = self.find_section_end_from(0);

                // Close the section once no more complete calls precede its end.
                if let Some((end_pos, end_len)) = section_end {
                    let call_before_end = call_start.is_some_and(|s| s < end_pos);
                    if !call_before_end {
                        // Complete section fully closed: drop its markup and resume
                        // keeping natural text (inter-section / post-wrapper
                        // narration). Any later section re-enters `in_section` and
                        // re-suppresses its markup. Matches the v1 batch parser,
                        // which preserves surrounding natural text verbatim
                        // (cases 8.b/8.c/8.d).
                        self.buffer.drain(..end_pos + end_len);
                        self.in_section = false;
                        self.suppress_normal_text = false;
                        continue;
                    }
                }

                let Some(start) = call_start else {
                    // No call_start yet and no closing section_end consumed
                    // above: the section body has not produced a complete call.
                    if flush {
                        tracing::warn!(
                            why = "kimi_k2_incomplete_tool_call",
                            "Kimi K2 stream dropped incomplete section at EOF"
                        );
                        self.buffer.clear();
                        self.in_section = false;
                    }
                    break;
                };
                if start > 0 {
                    self.buffer.drain(..start);
                }
                // A complete call needs both call_start and call_end. The call
                // body ends at the first call_end; refuse to cross a section_end
                // (mismatched fences) so a call_end-less call is dropped rather
                // than swallowing the section close.
                let Some(end) = self.buffer.find(self.config.call_end.as_str()) else {
                    if flush {
                        tracing::warn!(
                            why = "kimi_k2_incomplete_tool_call",
                            "Kimi K2 stream dropped call missing call_end at EOF"
                        );
                        self.buffer.clear();
                        self.in_section = false;
                    }
                    break;
                };
                // If a section_end sits before this call_end, the call is
                // malformed (no per-call end inside the fences). Drop it.
                if let Some((se_pos, se_len)) = self.find_section_end_from(0)
                    && se_pos < end
                {
                    tracing::warn!(
                        why = "kimi_k2_incomplete_tool_call",
                        "Kimi K2 stream dropped call missing call_end before section_end"
                    );
                    self.buffer.drain(..se_pos + se_len);
                    self.in_section = false;
                    // The section is CLOSED here — narration after it is the
                    // user's text, same as the clean-close path above. Keeping
                    // the latch on dropped trailing prose after a malformed
                    // section (text loss is the worse failure).
                    self.suppress_normal_text = false;
                    continue;
                }
                let call = self.buffer[..end + self.config.call_end.len()].to_string();
                self.buffer.drain(..end + self.config.call_end.len());
                if let Some(delta) = self.parse_call_delta(&call)? {
                    out.calls.push(delta);
                    self.next_index += 1;
                }
                self.suppress_normal_text = true;
                continue;
            }

            // Not in a section. A COMPLETE orphan close/inner marker (`call_end`,
            // `argument_begin`, or a `section_end` variant) that arrives outside
            // any open section is stray double-close markup: the `next_marker`
            // lookup below only recognizes section-start variants and `call_start`,
            // so such an orphan would otherwise flow straight into `normal_text`.
            // Strip it, mirroring the v1 batch parser's
            // `first_orphan_kimi_marker_index` policy. Emit any genuine text before
            // it (unless suppressing), drain through the marker, and clear the
            // suppression latch — the markup context has ended.
            if let Some((pos, len)) = self.find_orphan_close() {
                let next_open = self
                    .find_section_start()
                    .map(|(p, _)| p)
                    .into_iter()
                    .chain(self.buffer.find(self.config.call_start.as_str()))
                    .min();
                if next_open.is_none_or(|open| pos < open) {
                    if !self.suppress_normal_text && pos > 0 {
                        out.normal_text.push_str(&self.buffer[..pos]);
                    }
                    self.buffer.drain(..pos + len);
                    self.suppress_normal_text = false;
                    continue;
                }
            }

            // Not in a section. Look for the earliest section-start variant, or a
            // bare `call_start` (recovery: a complete call without section_begin,
            // mirroring the v1 parser's `recover_bare_kimi_calls_in_span`).
            let section = self.find_section_start();
            let bare_call = self.buffer.find(self.config.call_start.as_str());
            let next_marker = match (section, bare_call) {
                (Some((s, slen)), Some(c)) if s <= c => Some((s, Marker::Section(slen))),
                (Some(_), Some(c)) => Some((c, Marker::BareCall)),
                (Some((s, slen)), None) => Some((s, Marker::Section(slen))),
                (None, Some(c)) => Some((c, Marker::BareCall)),
                (None, None) => None,
            };

            let Some((start, marker)) = next_marker else {
                // No marker present: emit buffered text, but hold back a trailing
                // partial marker (split across this chunk boundary) unless flushing.
                let keep = if flush {
                    0
                } else {
                    self.marker_prefix_suffix_len()
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
                Marker::Section(slen) => {
                    self.buffer.drain(..slen);
                    self.in_section = true;
                    self.suppress_normal_text = true;
                }
                Marker::BareCall => {
                    // A bare call (no section_begin) is recovered only when its
                    // call_end has streamed; otherwise wait for more input.
                    let Some(end) = self.buffer.find(self.config.call_end.as_str()) else {
                        if flush {
                            tracing::warn!(
                                why = "kimi_k2_incomplete_tool_call",
                                "Kimi K2 stream dropped incomplete bare call at EOF"
                            );
                            self.buffer.clear();
                        }
                        break;
                    };
                    let call = self.buffer[..end + self.config.call_end.len()].to_string();
                    self.buffer.drain(..end + self.config.call_end.len());
                    if let Some(delta) = self.parse_call_delta(&call)? {
                        tracing::warn!(
                            why = "kimi_k2_bare_call_recovery",
                            tool_index = delta.tool_index,
                            "Kimi K2 stream recovered a complete bare call"
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

    /// Parse one complete `<|tool_call_begin|>...<|tool_call_end|>` call into a
    /// delta. Wraps the call in the section markers so the v1 parser takes its
    /// normal section path, then emits `name` + JSON `arguments` as one delta.
    fn parse_call_delta(&self, call: &str) -> anyhow::Result<Option<ToolCallDelta>> {
        let wrapped = format!(
            "{}{}{}",
            self.config.section_start, call, self.config.section_end
        );
        let (calls, _content) =
            try_tool_call_parse_kimi_k2(&wrapped, &self.config, Some(&self.tools))?;
        let Some(parsed) = calls.into_iter().next() else {
            return Ok(None);
        };
        Ok(Some(ToolCallDelta {
            tool_index: self.next_index,
            name: Some(parsed.function.name),
            arguments: parsed.function.arguments,
        }))
    }

    /// Longest non-empty proper prefix of any grammar marker that `self.buffer`
    /// ends with, so a marker split across chunk boundaries is held back instead
    /// of leaked as text.
    fn marker_prefix_suffix_len(&self) -> usize {
        self.markers
            .iter()
            .filter_map(|marker| {
                marker
                    .char_indices()
                    .map(|(idx, _)| idx)
                    .filter(|idx| *idx > 0 && *idx < marker.len())
                    .rev()
                    .find(|&len| self.buffer.ends_with(&marker[..len]))
            })
            .max()
            .unwrap_or(0)
    }
}

impl ToolParser for KimiK2ToolStreamParser {
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
    /// A section-start variant; carries the matched token length.
    Section(usize),
    /// A bare `<|tool_call_begin|>` with no section wrapper (recovery path).
    BareCall,
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
        let mut parser = KimiK2ToolStreamParser::new(tools);
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
                "<|tool_calls_section_begin|><|tool_call_begin|>",
                "functions.get_weather:0<|tool_call_argument_begin|>",
                "{\"location\":\"NYC\"}<|tool_call_end|><|tool_calls_section_end|>",
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
    fn emits_two_calls_in_one_section() {
        let tools = vec![
            Tool {
                name: "get_weather".to_string(),
                description: None,
                parameters: serde_json::json!({"type": "object"}),
                strict: None,
            },
            Tool {
                name: "get_time".to_string(),
                description: None,
                parameters: serde_json::json!({"type": "object"}),
                strict: None,
            },
        ];
        let out = parse_chunks(
            &tools,
            &[
                "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"location\":\"NYC\"}<|tool_call_end|>",
                "<|tool_call_begin|>functions.get_time:1<|tool_call_argument_begin|>{\"timezone\":\"EST\"}<|tool_call_end|><|tool_calls_section_end|>",
            ],
        );
        let merged = out.coalesce_calls();
        assert_eq!(merged.calls.len(), 2);
        assert_eq!(merged.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(merged.calls[0].arguments, r#"{"location":"NYC"}"#);
        assert_eq!(merged.calls[1].name.as_deref(), Some("get_time"));
        assert_eq!(merged.calls[1].arguments, r#"{"timezone":"EST"}"#);
    }

    #[test]
    fn preserves_prefix_text_before_section() {
        let out = parse_chunks(
            &weather_tools(),
            &[
                "I will",
                " check the weather. <|tool_calls_section_begin|>",
                "<|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"location\":\"NYC\"}<|tool_call_end|><|tool_calls_section_end|>",
            ],
        );
        assert_eq!(out.normal_text, "I will check the weather. ");
        assert_eq!(out.coalesce_calls().calls.len(), 1);
    }

    #[test]
    fn preserves_post_section_narration() {
        let out = parse_chunks(
            &weather_tools(),
            &[
                "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"location\":\"NYC\"}<|tool_call_end|><|tool_calls_section_end|>",
                " Done.",
            ],
        );
        // In-section markup is suppressed; post-section narration is preserved
        // verbatim once the section closes (v1 batch parity, cases 8.b/8.c).
        assert_eq!(out.normal_text, " Done.");
        assert_eq!(out.coalesce_calls().calls.len(), 1);
    }

    #[test]
    fn preserves_inter_section_narration() {
        // Two sections separated by narration (case 8.d): the prefix and the
        // inter-section text both flow into normal_text; both calls are emitted.
        let tools = vec![
            Tool {
                name: "get_weather".to_string(),
                description: None,
                parameters: serde_json::json!({"type": "object"}),
                strict: None,
            },
            Tool {
                name: "get_time".to_string(),
                description: None,
                parameters: serde_json::json!({"type": "object"}),
                strict: None,
            },
        ];
        let out = parse_chunks(
            &tools,
            &[
                "First. <|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"location\":\"NYC\"}<|tool_call_end|><|tool_calls_section_end|>",
                " Then. <|tool_calls_section_begin|><|tool_call_begin|>functions.get_time:1<|tool_call_argument_begin|>{\"timezone\":\"EST\"}<|tool_call_end|><|tool_calls_section_end|>",
            ],
        );
        assert_eq!(out.normal_text, "First.  Then. ");
        let merged = out.coalesce_calls();
        assert_eq!(merged.calls.len(), 2);
        assert_eq!(merged.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(merged.calls[1].name.as_deref(), Some("get_time"));
    }

    #[test]
    fn holds_back_marker_split_across_every_char() {
        // Worst case: the whole input arrives one fragment at a time, splitting
        // every grammar marker. No partial marker may leak into normal_text.
        let full = "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"location\":\"NYC\"}<|tool_call_end|><|tool_calls_section_end|>";
        let chunks: Vec<&str> = full
            .as_bytes()
            .chunks(3)
            .map(|c| std::str::from_utf8(c).unwrap())
            .collect();
        let out = parse_chunks(&weather_tools(), &chunks);
        assert_eq!(out.normal_text, "");
        let merged = out.coalesce_calls();
        assert_eq!(merged.calls.len(), 1);
        assert_eq!(merged.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(merged.calls[0].arguments, r#"{"location":"NYC"}"#);
    }

    #[test]
    fn suppresses_truncated_call_at_eof() {
        // Section + call header streamed, but no call_end / section_end before
        // EOF. The truncated call is dropped and no markup leaks.
        let out = parse_chunks(
            &weather_tools(),
            &[
                "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>",
                "{\"location\":\"NY",
            ],
        );
        assert_eq!(out.normal_text, "");
        assert!(out.calls.is_empty());
    }

    #[test]
    fn strips_orphan_call_end_outside_section() {
        // A complete orphan `call_end` with no open section is stray double-close
        // markup: it must be stripped, never leaked, and the surrounding genuine
        // prose preserved (v1 `first_orphan_kimi_marker_index` parity).
        let out = parse_chunks(
            &weather_tools(),
            &["Here you go.", "<|tool_call_end|>", "All set."],
        );
        assert_eq!(out.normal_text, "Here you go.All set.");
        assert!(out.calls.is_empty());
    }

    #[test]
    fn strips_orphan_argument_begin_outside_section() {
        let out = parse_chunks(
            &weather_tools(),
            &["Here you go.", "<|tool_call_argument_begin|>", "All set."],
        );
        assert_eq!(out.normal_text, "Here you go.All set.");
        assert!(out.calls.is_empty());
    }

    #[test]
    fn strips_orphan_section_end_outside_section() {
        let out = parse_chunks(
            &weather_tools(),
            &["Here you go.", "<|tool_calls_section_end|>", "All set."],
        );
        assert_eq!(out.normal_text, "Here you go.All set.");
        assert!(out.calls.is_empty());
    }

    #[test]
    fn recovers_complete_bare_call_without_section() {
        let out = parse_chunks(
            &weather_tools(),
            &[
                "<|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>",
                "{\"location\":\"NYC\"}<|tool_call_end|>",
            ],
        );
        assert_eq!(out.normal_text, "");
        let merged = out.coalesce_calls();
        assert_eq!(merged.calls.len(), 1);
        assert_eq!(merged.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(merged.calls[0].arguments, r#"{"location":"NYC"}"#);
    }
}
