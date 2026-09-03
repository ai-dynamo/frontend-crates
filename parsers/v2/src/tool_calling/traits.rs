// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `ToolParser` trait — Dynamo's owned mirror of vLLM Rust's `ToolParser` contract.
//!
//! Mirrors vLLM Rust `rust/src/tool-parser/src/lib.rs` from v0.22.0, with one
//! Dynamo extension: parsers may accept token-id chunks as well as decoded text
//! chunks. Harmony uses that token-native path for highest fidelity.
//!
//! Keeping this trait in Dynamo's crate (rather than importing vLLM) lets a
//! separate adapter crate bridge the two sides without pulling vLLM into Dynamo's
//! dependency graph.

use std::collections::{BTreeMap, btree_map};

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Result type used by Dynamo's vLLM-shaped parser contract.
pub type Result<T> = anyhow::Result<T>;

// Mirrors vLLM Rust `Tool` verbatim so vLLM can adopt Dynamo's parser crate with
// a small crate-path change instead of an adapter rewrite.
/// One function-style tool made available to the model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Value,
    pub strict: Option<bool>,
}

// Aligned with vLLM Rust `ToolCallDelta`; `complete` is Dynamo's lifecycle extension.
/// One tool-call update emitted while parsing assistant text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCallDelta {
    /// Stable parser-local tool index for this call within one assistant turn.
    pub tool_index: usize,
    /// Function name, present on the first update for one tool call.
    pub name: Option<String>,
    /// Arguments text contributed by this update.
    pub arguments: String,
    /// Whether this delta completes a validated call.
    pub complete: bool,
}

// Mirrors vLLM Rust `ToolParseResult` verbatim.
/// Result of advancing tool parsing with one assistant-text input.
///
/// `normal_text` carries non-tool-call output interleaved with calls. Harmony
/// never produces `normal_text` (its tool calls live on a separate channel), but
/// JSON/XML families need this field to satisfy the vLLM parser contract.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ToolParseResult {
    /// Plain assistant text that is not part of any tool call.
    pub normal_text: String,
    /// Tool-call updates extracted from this input.
    pub calls: Vec<ToolCallDelta>,
}

impl ToolParseResult {
    /// Append another parser result onto this one.
    ///
    /// This does not merge multiple deltas for the same tool call. Call
    /// `coalesce_calls()` if that behavior is desired.
    pub fn append(&mut self, mut other: Self) {
        self.normal_text.push_str(&other.normal_text);
        self.calls.append(&mut other.calls);
    }

    /// Merge multiple deltas for the same tool call into one item.
    ///
    /// This is primarily used by test helpers and batch adapters that delegate
    /// through the incremental parser lifecycle.
    pub fn coalesce_calls(mut self) -> Self {
        let mut merged = BTreeMap::<usize, ToolCallDelta>::new();
        let mut order = Vec::new();

        for call in self.calls {
            match merged.entry(call.tool_index) {
                btree_map::Entry::Vacant(entry) => {
                    order.push(call.tool_index);
                    entry.insert(call);
                }
                btree_map::Entry::Occupied(mut entry) => {
                    let existing = entry.get_mut();
                    if existing.name.is_none() {
                        existing.name = call.name;
                    }
                    existing.arguments.push_str(&call.arguments);
                    existing.complete |= call.complete;
                }
            }
        }

        self.calls = order
            .into_iter()
            .filter_map(|tool_index| merged.remove(&tool_index))
            .filter(|call| call.complete)
            .collect();
        self
    }
}

// Dynamo extension: vLLM Rust is text-only, while Dynamo can route either decoded
// text chunks or token-id chunks through the same parser contract.
/// Input chunk accepted by Dynamo's owned parser contract.
///
/// vLLM's Rust trait is text-only. Dynamo keeps that path through `push`, and
/// adds this borrowed enum so a serving layer can choose an all-text or all-token
/// stream per request without allocating wrapper buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolParserInput<'a> {
    Text(&'a str),
    Tokens(&'a [u32]),
}

// Mirrors vLLM Rust `ToolParser` except for the explicitly marked token-input
// extension methods below.
/// Dynamo's owned mirror of vLLM Rust's `ToolParser` trait.
///
/// **Streaming-first** — `push` is the required text method, matching vLLM Rust.
///
/// Extension vs vLLM: `push_tokens` and `push_input` let callers feed a parser
/// all decoded text chunks or all token-id chunks. Text remains the compatibility
/// path for vLLM.
pub trait ToolParser: Send {
    /// Construct a boxed parser instance for one request stream.
    fn create(tools: &[Tool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static;

    /// Return whether decoded output must preserve tokenizer special tokens.
    fn preserve_special_tokens(&self) -> bool {
        false
    }

    /// Data-driven input preference (audit B9): token-native families (Harmony)
    /// return `true` so runners feed `ToolParserInput::Tokens` via `push_input`
    /// instead of branching on the family name. Text-format parsers keep the
    /// default; the canonical per-family value also lives in `parser_families.yaml`.
    fn prefers_tokens(&self) -> bool {
        false
    }

    /// Feed one decoded text delta into the parser.
    fn push(&mut self, chunk: &str) -> Result<ToolParseResult>;

    // Dynamo extension: token-native parsers, especially Harmony, can avoid
    // lossy text reconstruction by consuming token IDs directly.
    /// Feed one token-id chunk (Harmony-native path).
    ///
    /// Defaults to an empty result. Override for token-native families.
    fn push_tokens(&mut self, _ids: &[u32]) -> Result<ToolParseResult> {
        Ok(ToolParseResult::default())
    }

    // Dynamo extension: lets the caller choose an all-text or all-token stream
    // per request while keeping vLLM-compatible `push` available.
    /// Feed one chunk in the caller-selected stream representation.
    fn push_input(&mut self, input: ToolParserInput<'_>) -> Result<ToolParseResult> {
        match input {
            ToolParserInput::Text(chunk) => self.push(chunk),
            ToolParserInput::Tokens(ids) => self.push_tokens(ids),
        }
    }

    /// Flush any buffered partial state at end of stream.
    fn finish(&mut self) -> Result<ToolParseResult> {
        Ok(ToolParseResult::default())
    }

    /// Parse complete tool calls from final output.
    ///
    /// The default implementation reuses the incremental parser lifecycle by
    /// feeding the full output through `push()` and then calling `finish()`.
    fn parse_complete(&mut self, output: &str) -> Result<ToolParseResult> {
        let mut result = self.push(output)?;
        result.append(self.finish()?);
        Ok(result.coalesce_calls())
    }
}
