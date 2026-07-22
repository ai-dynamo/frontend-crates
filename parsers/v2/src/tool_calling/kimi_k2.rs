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
//! suppression) is owned by the shared [`scan::WrappedBlockScanner`]; the K2
//! grammar maps onto it with section variants as multi-token block markers,
//! the inner `call_end`/`argument_begin` markers as extra orphan markers, and
//! two K2-specific spec fields: the suppression latch engages after every
//! in-section call parse even when the call is malformed (`InvokeLatch::Always`),
//! and a call whose `call_end` never arrives before the section close is
//! dropped rather than swallowing the fence (`drop_invoke_crossing_block_end`).
//!
//! The per-call typing (function-id parsing, JSON validation, raw-string
//! fallback for malformed args) is delegated to the v1 batch parser
//! `try_tool_call_parse_kimi_k2` driven by the same `KimiK2ParserConfig`
//! `dynamo_parsers` uses for batch parsing, so a streamed call matches exactly
//! what the batch parser produces. A complete call is wrapped in the section
//! markers before delegating so the v1 parser always takes its normal section
//! path.
//!
//! The per-call arguments are already a JSON object string, so no key-order
//! reserialization is needed (unlike the XML families): the v1 parser
//! round-trips compact JSON byte-for-byte and falls back to the raw string for
//! malformed payloads, which is exactly what the fixtures expect.

use crate::tool_calling::scan::{
    BareRecoveryLatch, InvokeEmitter, InvokeLatch, WrappedBlockScanner, WrappedBlockSpec,
};
use crate::tool_calling::v1core::{
    KimiK2ParserConfig, ToolDefinition, try_tool_call_parse_kimi_k2,
};

use crate::tool_calling::traits::{Tool, ToolCallDelta, ToolParseResult, ToolParser};

fn spec(config: &KimiK2ParserConfig) -> WrappedBlockSpec {
    // Orphan markers: inner markers (`call_end`, `argument_begin`) and every
    // section-end variant only appear legitimately inside an open section;
    // outside one they are stray grammar markup to be stripped. Mirrors the v1
    // batch parser's `first_orphan_kimi_marker_index` (minus `call_start`,
    // which the bare-call recovery path already opens).
    let mut orphan_markers = vec![config.call_end.clone(), config.argument_begin.clone()];
    orphan_markers.extend(config.section_end_variants.clone());

    // Every grammar marker that must never be split-leaked as normal_text.
    let mut holdback_markers = config.section_start_variants.clone();
    holdback_markers.extend(config.section_end_variants.clone());
    holdback_markers.push(config.call_start.clone());
    holdback_markers.push(config.call_end.clone());
    holdback_markers.push(config.argument_begin.clone());

    WrappedBlockSpec {
        family: "kimi_k2",
        block_starts: config.section_start_variants.clone(),
        block_ends: config.section_end_variants.clone(),
        invoke_start: config.call_start.clone(),
        invoke_end: config.call_end.clone(),
        orphan_markers,
        holdback_markers,
        bare_recovery_latch: BareRecoveryLatch::Set,
        invoke_latch: InvokeLatch::Always,
        drop_invoke_crossing_block_end: true,
    }
}

/// Value-typing hook: wraps one complete
/// `<|tool_call_begin|>...<|tool_call_end|>` call in the section markers so
/// the v1 parser takes its normal section path, then emits `name` + JSON
/// `arguments` as one delta.
struct K2Emitter {
    config: KimiK2ParserConfig,
    tools: Vec<ToolDefinition>,
}

impl InvokeEmitter for K2Emitter {
    fn parse_invoke(
        &self,
        invoke: &str,
        tool_index: usize,
    ) -> anyhow::Result<Option<ToolCallDelta>> {
        let wrapped = format!(
            "{}{}{}",
            self.config.section_start, invoke, self.config.section_end
        );
        let (calls, _content) =
            try_tool_call_parse_kimi_k2(&wrapped, &self.config, Some(&self.tools))?;
        let Some(parsed) = calls.into_iter().next() else {
            return Ok(None);
        };
        Ok(Some(ToolCallDelta {
            tool_index,
            name: Some(parsed.function.name),
            arguments: parsed.function.arguments,
        }))
    }
}

/// Stream parser for Kimi K2 tool calls.
pub struct KimiK2ToolStreamParser {
    scanner: WrappedBlockScanner<K2Emitter>,
}

impl KimiK2ToolStreamParser {
    pub fn new(tools: &[Tool]) -> Self {
        let config = KimiK2ParserConfig::default();
        Self {
            scanner: WrappedBlockScanner::new(
                spec(&config),
                K2Emitter {
                    config,
                    tools: tools.iter().map(ToolDefinition::from).collect(),
                },
            ),
        }
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
        self.scanner.push(chunk)
    }

    fn finish(&mut self) -> anyhow::Result<ToolParseResult> {
        self.scanner.finish()
    }
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
