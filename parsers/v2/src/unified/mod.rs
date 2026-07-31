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

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::tool_calling::scan::{InvokeEmitter, WrappedBlockScanner, push_run};
use crate::tool_calling::traits::{Result, Tool, ToolCallDelta, ToolParseResult};

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
    // Coalesce adjacent same-kind runs with the SAME helper the scan core uses, so
    // `I8` has exactly ONE implementation instead of one per type.
    let mut merged: Vec<UnifiedDelta> = Vec::new();
    for delta in deltas {
        match delta {
            UnifiedDelta::Reasoning { text } => push_run(&mut merged, Kind::Reasoning, text),
            UnifiedDelta::Text { text } => push_run(&mut merged, Kind::Text, text),
            call => merged.push(call.clone()),
        }
    }

    // Convert, joining each call's argument fragments. Keyed by `tool_index` so
    // fragments of two interleaved calls cannot merge, and carrying each call's
    // position so it stays where its FIRST delta landed.
    let mut out: Vec<UnifiedEvent> = Vec::new();
    let mut calls: BTreeMap<usize, (usize, String)> = BTreeMap::new();
    for delta in merged {
        match delta {
            UnifiedDelta::Reasoning { text } => out.push(UnifiedEvent::Reasoning { text }),
            UnifiedDelta::Text { text } => out.push(UnifiedEvent::Text { text }),
            UnifiedDelta::ToolCall(call) => {
                let (pos, raw) = calls.entry(call.tool_index).or_insert_with(|| {
                    out.push(UnifiedEvent::ToolCall {
                        name: String::new(),
                        arguments: serde_json::Value::Null,
                    });
                    (out.len() - 1, String::new())
                });
                raw.push_str(&call.arguments);
                if let Some(incoming) = call.name
                    && let UnifiedEvent::ToolCall { name, .. } = &mut out[*pos]
                    && name.is_empty()
                {
                    *name = incoming;
                }
            }
        }
    }

    for (pos, raw) in calls.into_values() {
        if let UnifiedEvent::ToolCall { arguments, .. } = &mut out[pos] {
            // Best-effort (P3): a malformed payload must not take down the turn, but
            // it is NOT discarded silently — `{}` alone is indistinguishable from a
            // genuine no-arg call, so a corrupted argument would look like a clean parse.
            *arguments = serde_json::from_str(&raw).unwrap_or_else(|e| {
                if !raw.trim().is_empty() {
                    tracing::warn!(
                        why = "unified_unparseable_tool_arguments",
                        error = %e, raw = %raw,
                        "tool-call arguments did not parse as JSON; emitting an empty object"
                    );
                }
                serde_json::json!({})
            });
        }
    }
    out
}

/// The two payload kinds that carry a text run and coalesce when adjacent (`I8`).
/// Shared with the scan core, whose `push_run` is the single implementation.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kind {
    Reasoning,
    Text,
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

/// A [`UnifiedParser`] backed by the shared marker scanner.
///
/// Any family whose grammar [`WrappedBlockScanner`] already covers becomes a
/// one-line factory in `create_unified_parser_for_family` — there is no
/// per-family struct and no per-family trait impl to write, or to forget to keep
/// in sync when the trait grows. Construction lives in the registry, which is why
/// the trait itself has no `create`.
pub(crate) struct ScannerUnified<E: InvokeEmitter> {
    pub(crate) scanner: WrappedBlockScanner<E>,
}

impl<E: InvokeEmitter + Send> UnifiedParser for ScannerUnified<E> {
    fn push(&mut self, chunk: &str) -> Result<Vec<UnifiedDelta>> {
        self.scanner.push_ordered(chunk)
    }

    fn finish(&mut self) -> Result<Vec<UnifiedDelta>> {
        self.scanner.finish_ordered()
    }
}

/// THE registry. One line per family — adding a family is adding a line here and
/// nothing else in this crate.
///
/// It used to be two things that had to agree: a `match` in the constructor and a
/// `REGISTERED_UNIFIED_FAMILIES` const the tests iterate. Adding a family meant
/// editing both, and a family added to one but not the other either failed to
/// construct or silently skipped its coverage. The macro generates both from this
/// single list, so they cannot disagree.
///
/// A family may carry aliases: the conformance corpus calls the Qwen XML grammar
/// `qwen3` while the tool-only registry calls it `qwen3_coder`, and callers should
/// not have to know which name they arrived with.
macro_rules! unified_registry {
    ($($family:literal $(| $alias:literal)* => $ctor:path),+ $(,)?) => {
        /// Every family `create_unified_parser_for_family` accepts, aliases included.
        /// Tests iterate this, so a family here without conformance coverage fails the
        /// suite instead of silently skipping.
        pub const REGISTERED_UNIFIED_FAMILIES: &[&str] = &[$($family, $($alias,)*)+];

        /// Create the Dynamo unified parser for a conformance family.
        pub fn create_unified_parser_for_family(
            family: &str,
            tools: &[Tool],
        ) -> Result<Box<dyn UnifiedParser>> {
            match family {
                $($family $(| $alias)* => Ok($ctor(tools)),)+
                other => anyhow::bail!("no Dynamo unified parser for family '{other}'"),
            }
        }
    };
}

unified_registry! {
    "qwen3" | "qwen3_coder" => qwen3::qwen3_unified,
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
