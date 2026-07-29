// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unified parsing: ONE streaming state machine per stream that owns reasoning,
//! visible content, and tool calls, and emits ONE ordered event stream.
//!
//! # Why this exists
//!
//! Dynamo serves today by chaining two independent parsers: a reasoning parser
//! strips `<think>...</think>` over the whole stream into a single assembled
//! `reasoning_text` field, and a tool parser then scans the leftover content.
//! That shape cannot represent WHERE reasoning happened. Every thought is
//! hoisted to the front and merged into one span, so
//!
//! ```text
//! <think>Look it up.</think><tool_call>…</tool_call><think>Now answer.</think>It's 18C.
//! ```
//!
//! serves as `reasoning("Look it up.Now answer.")` → call → `text("It's 18C.")`:
//! the second thought moved ahead of the call it followed and fused with the
//! first. A client rendering thoughts inline shows them in the wrong place, and
//! a client counting reasoning turns sees one where there were two.
//!
//! Ordering is not a field the split can add; it is lost at the seam between the
//! two parsers. So a unified parser owns the whole grammar and emits deltas in
//! the order the model produced them:
//!
//! ```text
//! reasoning("Look it up.") | tool_call(get_weather, …) | reasoning("Now answer.") | text("It's 18C.")
//! ```
//!
//! # Shape
//!
//! [`UnifiedDelta`] is the streaming vocabulary — what one `push` produced, in
//! order. [`UnifiedEvent`] is the assembled view: adjacent same-kind deltas
//! coalesced and per-call argument fragments joined into one typed object
//! (`I8`). [`assemble`] is the single implementation of that fold, so callers
//! and conformance harnesses never reimplement it and drift.
//!
//! Note this is a genuine unified parser, not vLLM's `CombinedParser` shape:
//! vLLM 0.25.x keeps `extract_tool_calls_streaming` and
//! `extract_reasoning_streaming` as two chained APIs behind a unified interface
//! (only gemma4 is natively unified there), which reproduces the same seam.

pub mod qwen3;

use serde::{Deserialize, Serialize};

use crate::tool_calling::traits::{Result, Tool, ToolCallDelta, ToolParseResult};

pub use qwen3::Qwen3UnifiedParser;

/// One ordered update produced while parsing assistant output.
///
/// This is the streaming vocabulary shared by the whole crate: the marker-scan
/// core emits it, tool-only parsers project it down to [`ToolParseResult`], and
/// unified parsers hand it to the caller as-is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnifiedDelta {
    /// Private chain-of-thought.
    Reasoning { text: String },
    /// User-visible content.
    Text { text: String },
    /// A tool-call update. Carries the tool-only [`ToolCallDelta`] verbatim so
    /// the two surfaces cannot drift in how a call is described.
    ToolCall(ToolCallDelta),
}

/// One assembled event: the order-sensitive unit the unified conformance
/// surface compares. Serializes to the golden-corpus schema
/// (`{kind: reasoning|text|tool_call, …}`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum UnifiedEvent {
    Reasoning {
        text: String,
    },
    Text {
        text: String,
    },
    ToolCall {
        name: String,
        #[serde(default)]
        arguments: serde_json::Value,
    },
}

/// A parser that owns reasoning + content + tool calls for one stream.
///
/// Streaming-first, like [`crate::ToolParser`]: `push` per decoded delta,
/// `finish` once at end of stream. One instance parses exactly one choice of
/// one request, which is what gives per-stream isolation (`I4`) by construction.
pub trait UnifiedParser: Send {
    /// Construct a boxed parser instance for one request stream.
    fn create(tools: &[Tool]) -> Result<Box<dyn UnifiedParser>>
    where
        Self: Sized + 'static;

    /// Return whether decoded output must preserve tokenizer special tokens.
    fn preserve_special_tokens(&self) -> bool {
        false
    }

    /// Feed one decoded text delta; returns the updates it completed, in order.
    fn push(&mut self, chunk: &str) -> Result<Vec<UnifiedDelta>>;

    /// Flush buffered partial state at end of stream.
    ///
    /// Open reasoning is promoted here rather than dropped or leaked as text,
    /// and an unrecoverable partial tool call is dropped without erroring
    /// (policy P2 — best-effort recovery).
    fn finish(&mut self) -> Result<Vec<UnifiedDelta>>;

    /// Parse complete output through the incremental lifecycle, then assemble.
    ///
    /// Routing batch through `push`/`finish` is what makes stream/batch parity
    /// (`I6`) structural instead of a property two code paths have to agree on.
    fn parse_complete(&mut self, output: &str) -> Result<Vec<UnifiedEvent>> {
        let mut deltas = self.push(output)?;
        deltas.append(&mut self.finish()?);
        Ok(assemble(&deltas))
    }
}

/// Collapse an ordered delta stream into assembled events.
///
/// Adjacent same-kind reasoning/text deltas merge (`I8`); tool-call fragments
/// are joined by `tool_index` and parsed into a typed object, holding each
/// call's position at its FIRST delta so order survives fragmentation. Empty or
/// unparseable arguments become `{}` (policy P3) rather than an error, because a
/// malformed argument payload must not take down the rest of the turn.
pub fn assemble(deltas: &[UnifiedDelta]) -> Vec<UnifiedEvent> {
    // Position of each tool_index in `out`, plus its accumulated argument text.
    let mut slots: Vec<(usize, usize, String)> = Vec::new();
    let mut out: Vec<UnifiedEvent> = Vec::new();

    for delta in deltas {
        match delta {
            UnifiedDelta::Reasoning { text } => {
                if text.is_empty() {
                    continue;
                }
                match out.last_mut() {
                    Some(UnifiedEvent::Reasoning { text: prev }) => prev.push_str(text),
                    _ => out.push(UnifiedEvent::Reasoning { text: text.clone() }),
                }
            }
            UnifiedDelta::Text { text } => {
                if text.is_empty() {
                    continue;
                }
                match out.last_mut() {
                    Some(UnifiedEvent::Text { text: prev }) => prev.push_str(text),
                    _ => out.push(UnifiedEvent::Text { text: text.clone() }),
                }
            }
            UnifiedDelta::ToolCall(call) => {
                let slot = match slots.iter_mut().find(|(idx, ..)| *idx == call.tool_index) {
                    Some(slot) => slot,
                    None => {
                        out.push(UnifiedEvent::ToolCall {
                            name: String::new(),
                            arguments: serde_json::Value::Null,
                        });
                        slots.push((call.tool_index, out.len() - 1, String::new()));
                        slots.last_mut().expect("just pushed")
                    }
                };
                slot.2.push_str(&call.arguments);
                if let Some(new_name) = &call.name
                    && let UnifiedEvent::ToolCall { name, .. } = &mut out[slot.1]
                    && name.is_empty()
                {
                    *name = new_name.clone();
                }
            }
        }
    }

    for (_, pos, raw) in &slots {
        let parsed = if raw.trim().is_empty() {
            serde_json::json!({})
        } else {
            // Best-effort (P3): a malformed payload must not take down the rest of
            // the turn. But it is NOT discarded silently — a call whose arguments
            // arrive corrupted is exactly the failure worth seeing in a log, and
            // `{}` on its own looks indistinguishable from a genuine no-arg call.
            serde_json::from_str(raw).unwrap_or_else(|e| {
                tracing::warn!(
                    why = "unified_unparseable_tool_arguments",
                    error = %e,
                    raw = %raw,
                    "tool-call arguments did not parse as JSON; emitting an empty object instead"
                );
                serde_json::json!({})
            })
        };
        if let UnifiedEvent::ToolCall { arguments, .. } = &mut out[*pos] {
            *arguments = parsed;
        }
    }

    out
}

impl ToolParseResult {
    /// Project an ordered delta stream down to the tool-only view.
    ///
    /// The tool-only contract has no reasoning channel and no text/call
    /// ordering, so reasoning folds into `normal_text` exactly where it
    /// occurred — which is what a reasoning-unaware tool parser sees anyway.
    /// This projection is the ONLY place the two surfaces are bridged, so the
    /// scan core can emit ordered deltas without changing tool-only behavior.
    pub fn from_deltas(deltas: Vec<UnifiedDelta>) -> Self {
        let mut out = Self::default();
        for delta in deltas {
            match delta {
                UnifiedDelta::Reasoning { text } | UnifiedDelta::Text { text } => {
                    out.normal_text.push_str(&text)
                }
                UnifiedDelta::ToolCall(call) => out.calls.push(call),
            }
        }
        out
    }
}

/// Every family `create_unified_parser_for_family` accepts, one entry per match
/// arm. Tests iterate this so a family registered here without conformance
/// coverage fails the suite instead of silently skipping.
pub const REGISTERED_UNIFIED_FAMILIES: &[&str] = &["qwen3", "qwen3_coder"];

/// Create the Dynamo unified parser for a conformance family.
pub fn create_unified_parser_for_family(
    family: &str,
    tools: &[Tool],
) -> Result<Box<dyn UnifiedParser>> {
    match family {
        // The conformance corpus calls this family `qwen3`; the tool-only
        // registry calls the same XML grammar `qwen3_coder`. Accept both so
        // callers do not have to know which registry they came from.
        "qwen3" | "qwen3_coder" => Qwen3UnifiedParser::create(tools),
        other => anyhow::bail!("no Dynamo unified parser for family '{other}'"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(tool_index: usize, name: Option<&str>, arguments: &str) -> UnifiedDelta {
        UnifiedDelta::ToolCall(ToolCallDelta {
            tool_index,
            name: name.map(str::to_string),
            arguments: arguments.to_string(),
        })
    }

    #[test]
    fn registered_families_all_create() {
        for family in REGISTERED_UNIFIED_FAMILIES {
            create_unified_parser_for_family(family, &[]).unwrap_or_else(|e| {
                panic!("REGISTERED_UNIFIED_FAMILIES entry '{family}' does not create: {e}")
            });
        }
    }

    #[test]
    fn assemble_coalesces_adjacent_same_kind() {
        let out = assemble(&[
            UnifiedDelta::Reasoning {
                text: "think".into(),
            },
            UnifiedDelta::Reasoning { text: "ing".into() },
            UnifiedDelta::Text { text: "he".into() },
            UnifiedDelta::Text { text: "llo".into() },
        ]);
        assert_eq!(
            out,
            vec![
                UnifiedEvent::Reasoning {
                    text: "thinking".into()
                },
                UnifiedEvent::Text {
                    text: "hello".into()
                },
            ]
        );
    }

    #[test]
    fn assemble_does_not_coalesce_across_a_call() {
        // The whole point of the surface: two thoughts separated by a call stay
        // two thoughts, in position.
        let out = assemble(&[
            UnifiedDelta::Reasoning { text: "a".into() },
            call(0, Some("f"), r#"{"x":"1"}"#),
            UnifiedDelta::Reasoning { text: "b".into() },
        ]);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0], UnifiedEvent::Reasoning { text: "a".into() });
        assert_eq!(out[2], UnifiedEvent::Reasoning { text: "b".into() });
    }

    #[test]
    fn assemble_joins_argument_fragments_at_the_first_position() {
        let out = assemble(&[
            call(0, Some("f"), r#"{"x":"#),
            UnifiedDelta::Text { text: "mid".into() },
            call(0, None, r#""1"}"#),
        ]);
        assert_eq!(
            out,
            vec![
                UnifiedEvent::ToolCall {
                    name: "f".into(),
                    arguments: serde_json::json!({"x": "1"}),
                },
                UnifiedEvent::Text { text: "mid".into() },
            ]
        );
    }

    #[test]
    fn assemble_defaults_unusable_arguments_to_empty_object() {
        // P3 / best-effort: a malformed payload must not error out the turn.
        let out = assemble(&[call(0, Some("f"), "not json")]);
        assert_eq!(
            out,
            vec![UnifiedEvent::ToolCall {
                name: "f".into(),
                arguments: serde_json::json!({}),
            }]
        );
    }

    #[test]
    fn tool_only_projection_drops_order_but_not_bytes() {
        let result = ToolParseResult::from_deltas(vec![
            UnifiedDelta::Reasoning { text: "a".into() },
            call(0, Some("f"), "{}"),
            UnifiedDelta::Text { text: "b".into() },
        ]);
        assert_eq!(result.normal_text, "ab");
        assert_eq!(result.calls.len(), 1);
    }

    #[test]
    fn unified_event_matches_the_golden_corpus_schema() {
        let yaml = "- {kind: reasoning, text: \"a\"}\n\
                    - {kind: tool_call, name: f, arguments: {x: \"1\"}}\n\
                    - {kind: text, text: \"b\"}\n";
        let parsed: Vec<UnifiedEvent> = serde_yaml::from_str(yaml).expect("golden schema");
        assert_eq!(
            parsed,
            vec![
                UnifiedEvent::Reasoning { text: "a".into() },
                UnifiedEvent::ToolCall {
                    name: "f".into(),
                    arguments: serde_json::json!({"x": "1"}),
                },
                UnifiedEvent::Text { text: "b".into() },
            ]
        );
    }
}
