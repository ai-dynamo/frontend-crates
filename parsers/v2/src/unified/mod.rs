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

use crate::tool_calling::scan::{
    InvokeEmitter, ReasoningSpec, WrappedBlockScanner, marker_prefix_suffix_len, push_run,
};
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

/// Assistant-channel state established by the rendered generation prompt.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum UnifiedParserPrefill {
    /// Generated output includes any channel-opening marker itself.
    #[default]
    None,
    /// The prompt opened reasoning, so generated output begins inside it.
    Reasoning,
    /// The prompt opened the visible response channel.
    Response,
}

/// Tool-call wire format selected for one request.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum UnifiedToolOutputMode<'a> {
    /// Model-native tool-call markup.
    #[default]
    Native,
    /// Guided decoding emits bare JSON instead of model-native markup.
    ///
    /// A named choice contains only that tool's arguments. A required choice
    /// contains one call object or an array of call objects.
    GuidedJson { named_tool: Option<&'a str> },
}

/// A parser that owns reasoning + content + tool calls for one stream.
///
/// Streaming-first, like [`crate::ToolParser`]: `push` per decoded delta,
/// `finish` once at end of stream. One instance parses exactly one choice of
/// one request, which is what gives per-stream isolation (`I4`) by construction.
pub trait UnifiedParser: Send {
    /// Initialize request-scoped prompt state.
    fn initialize(&mut self, prefill: UnifiedParserPrefill) -> Result<()> {
        if prefill != UnifiedParserPrefill::None {
            anyhow::bail!("this unified parser does not support prompt-prefilled channels");
        }
        Ok(())
    }

    /// Initialize prompt state and the backend's selected tool wire format.
    fn initialize_with_output_mode(
        &mut self,
        prefill: UnifiedParserPrefill,
        tool_output_mode: UnifiedToolOutputMode<'_>,
    ) -> Result<()> {
        if tool_output_mode != UnifiedToolOutputMode::Native {
            anyhow::bail!("this unified parser does not support guided tool output");
        }
        self.initialize(prefill)
    }

    /// Feed one decoded text delta; returns the updates it completed, in order.
    fn push(&mut self, chunk: &str) -> Result<Vec<UnifiedDelta>>;

    /// Flush buffered partial state at end of stream.
    ///
    /// Open reasoning is promoted here rather than dropped or leaked as text,
    /// and an unrecoverable partial tool call is dropped without erroring
    /// (policy P2 — best-effort recovery).
    fn finish(&mut self) -> Result<Vec<UnifiedDelta>>;

    /// Return the parser to a FRESH-STREAM state and hand back any unconsumed text.
    ///
    /// This is not a mid-turn continuation hook. Everything restarts, including the
    /// tool index, so the returned text must be re-parsed as a NEW stream and any
    /// calls already dispatched belong to the abandoned one — feeding the remainder
    /// back into the same turn would re-number from index 0 and collide with them.
    /// That follows from `I4`: one parser instance owns exactly one stream, so a
    /// retry is a new stream and gets a reset (or a new) parser, not a splice.
    fn reset(&mut self) -> String {
        String::new()
    }

    /// Parse complete output through the incremental lifecycle, then assemble.
    ///
    /// Routing batch through `push`/`finish` is what makes stream/batch parity
    /// (`I6`) structural instead of a property two code paths have to agree on.
    fn parse_complete(&mut self, output: &str) -> Result<Vec<UnifiedEvent>> {
        let mut deltas = self.push(output)?;
        deltas.append(&mut self.finish()?);
        Ok(assemble(&deltas))
    }

    // --- vLLM-shaped surface -------------------------------------------------
    // vLLM's Rust `UnifiedParser` (rust/src/parser/src/unified/mod.rs) is the
    // upstream this mirrors, the same way `ToolParser` mirrors its tool trait.
    // Its required method appends into a caller-owned buffer instead of
    // returning a fresh Vec, so a serving loop can accumulate one turn without
    // a per-delta allocation. Both spellings are kept: `push` returns, which is
    // what the conformance corpus asserts against, and `parse_into` appends,
    // which is what an upstream-shaped caller wants. Neither is a second
    // implementation — `parse_into` is defined in terms of `push`.

    /// Feed one decoded text delta, appending committed updates into `output`.
    ///
    /// vLLM's error contract, which this keeps: on `Err`, whatever was already
    /// appended stays committed and the parser's uncommitted buffer is intact,
    /// so the caller can recover it with [`UnifiedParser::reset`].
    fn parse_into(&mut self, delta: &str, output: &mut UnifiedParserOutput) -> Result<()> {
        output.events.extend(self.push(delta)?);
        Ok(())
    }

    /// Flush buffered state at end of stream into a caller-owned buffer.
    fn finish_into(&mut self) -> Result<UnifiedParserOutput> {
        Ok(UnifiedParserOutput {
            events: self.finish()?,
        })
    }

    /// Whether decoded output must keep tokenizer special tokens.
    ///
    /// Mirrors the tool trait's method of the same name: a family whose markers
    /// ARE special tokens cannot be parsed from text that dropped them.
    fn preserve_special_tokens(&self) -> bool {
        false
    }

    /// The model-emitted id for a tool call, when the grammar carries one.
    ///
    /// Qwen3's XML grammar does not, so the default is correct for it; families
    /// whose envelope names the call (kimi's `functions.NAME:INDEX`) override.
    fn tool_call_id(&self, _tool_index: usize) -> Option<&str> {
        None
    }

    /// Derive prompt state by INSPECTING the rendered prompt, then initialize.
    ///
    /// vLLM's reasoning module argues the rendered prompt is a more faithful
    /// signal than a per-family convention, because the same family can be
    /// rendered with or without an open channel depending on the template. This
    /// crate keeps [`UnifiedParserPrefill`] as the declared form of that fact,
    /// so a caller that has already resolved it can still pass it directly.
    ///
    /// The default detects nothing and starts in `None`; a family whose prompt
    /// can end mid-channel overrides this.
    fn initialize_from_prompt(&mut self, _prompt_text: &str) -> Result<()> {
        self.initialize(UnifiedParserPrefill::None)
    }
}

/// Ordered updates committed by one parser advance.
///
/// Mirrors vLLM's `UnifiedParserOutput`: a vector, not a bundle of parallel
/// channel fields. That is the whole point — a bundle cannot say whether text
/// came before or after a call, which is the ordering this surface exists to
/// pin (vLLM reached the same conclusion in their PR #46584).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct UnifiedParserOutput {
    /// Updates in the order the model produced them.
    pub events: Vec<UnifiedDelta>,
}

impl UnifiedParserOutput {
    /// Append another advance's updates, preserving order.
    pub fn append(&mut self, other: &mut Self) {
        self.events.append(&mut other.events);
    }

    /// Collapse into assembled events (see [`assemble`]).
    pub fn assembled(&self) -> Vec<UnifiedEvent> {
        assemble(&self.events)
    }

    /// Verbatim argument bytes per `tool_index` (see [`tool_arguments_raw`]).
    pub fn tool_arguments_raw(&self) -> BTreeMap<usize, String> {
        tool_arguments_raw(&self.events)
    }
}

/// Each call's argument bytes exactly as the model produced them, keyed by
/// `tool_index`.
///
/// [`assemble`] parses arguments into a [`serde_json::Value`] because the
/// conformance corpus compares them semantically — key order and whitespace are
/// not defects there. A SERVING path has the opposite requirement: the OpenAI
/// wire format carries `arguments` as a string, and re-serializing a `Value`
/// rewrites the model's bytes (`{"city": "Tokyo"}` becomes `{"city":"Tokyo"}`).
///
/// A streaming caller never hits this because it forwards
/// [`ToolCallDelta::arguments`] verbatim. A non-streaming caller assembling the
/// same turn would, which would make the two disagree on identical input and
/// break argument fidelity (`I7`) on the batch path alone. This returns the same
/// joined bytes `assemble` folds, so a caller can have the parsed view and the
/// verbatim view without reimplementing the join and drifting from it (`I6`).
pub fn tool_arguments_raw(deltas: &[UnifiedDelta]) -> BTreeMap<usize, String> {
    let mut raw: BTreeMap<usize, String> = BTreeMap::new();
    for delta in deltas {
        if let UnifiedDelta::ToolCall(call) = delta {
            raw.entry(call.tool_index)
                .or_default()
                .push_str(&call.arguments);
        }
    }
    raw
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
                        error = %e,
                        argument_bytes = raw.len(),
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
    /// Which channel the PROMPT already opened. Held here as well as on the
    /// scanner because `reset` has to restore it, and because the guided path
    /// below never reaches the scanner at all.
    prefill: UnifiedParserPrefill,
    /// Set once the backend selects guided decoding for this request; `None`
    /// on the native path, which is every request that did not ask for it.
    /// Boxed so a native stream carries one null pointer, not six idle fields.
    guided: Option<Box<GuidedState>>,
    /// Whether decoded text must keep tokenizer special tokens, carried from the
    /// FAMILY rather than defaulted. The tool-only parser for the same grammar
    /// answers this too, and the two must agree: a host that honours a `false`
    /// here would hand the parser text with its own call markers already
    /// stripped, so the calls silently vanish on the unified path only.
    preserve_special_tokens: bool,
    started: bool,
    finished: bool,
}

impl<E: InvokeEmitter> ScannerUnified<E> {
    /// `preserve_special_tokens` is REQUIRED rather than defaulted: it must match
    /// what the tool-only parser for the same grammar answers, and a family that
    /// silently inherited `false` would have its calls stripped before they ever
    /// reached the parser. Making it a parameter forces each family to state it.
    pub(crate) fn new(scanner: WrappedBlockScanner<E>, preserve_special_tokens: bool) -> Self {
        Self {
            scanner,
            prefill: UnifiedParserPrefill::None,
            guided: None,
            preserve_special_tokens,
            started: false,
            finished: false,
        }
    }

    fn initialize_request(
        &mut self,
        prefill: UnifiedParserPrefill,
        tool_output_mode: UnifiedToolOutputMode<'_>,
    ) -> Result<()> {
        if self.started || self.finished {
            anyhow::bail!("cannot initialize a unified parser after parsing has started");
        }
        self.prefill = prefill;
        self.scanner.reset();
        // `Response` means the prompt already opened visible content, so this
        // stream has no reasoning channel at all; `Reasoning` means it opened a
        // thought the model will close without ever emitting the opener.
        self.scanner.set_reasoning_mode(
            prefill != UnifiedParserPrefill::Response,
            prefill == UnifiedParserPrefill::Reasoning,
        );
        self.guided = match tool_output_mode {
            UnifiedToolOutputMode::Native => None,
            UnifiedToolOutputMode::GuidedJson { named_tool } => {
                let Some(reasoning) = self.scanner.reasoning_spec() else {
                    anyhow::bail!("guided tool output needs a reasoning-aware scanner");
                };
                Some(Box::new(GuidedState::new(
                    reasoning,
                    self.scanner.control_markers().to_vec(),
                    self.scanner.invoke_end().to_string(),
                    named_tool.map(str::to_string),
                    prefill,
                )))
            }
        };
        self.finished = false;
        Ok(())
    }
}

impl<E: InvokeEmitter + Send> UnifiedParser for ScannerUnified<E> {
    fn preserve_special_tokens(&self) -> bool {
        self.preserve_special_tokens
    }

    fn initialize(&mut self, prefill: UnifiedParserPrefill) -> Result<()> {
        self.initialize_request(prefill, UnifiedToolOutputMode::Native)
    }

    fn initialize_with_output_mode(
        &mut self,
        prefill: UnifiedParserPrefill,
        tool_output_mode: UnifiedToolOutputMode<'_>,
    ) -> Result<()> {
        self.initialize_request(prefill, tool_output_mode)
    }

    fn push(&mut self, chunk: &str) -> Result<Vec<UnifiedDelta>> {
        if self.finished {
            anyhow::bail!("cannot push to a finished unified parser");
        }
        self.started = true;
        match self.guided.as_mut() {
            None => self.scanner.push_ordered(chunk),
            Some(guided) => Ok(guided.push(chunk)),
        }
    }

    fn finish(&mut self) -> Result<Vec<UnifiedDelta>> {
        if self.finished {
            anyhow::bail!("cannot finish a unified parser twice");
        }
        self.started = true;
        self.finished = true;
        match self.guided.as_mut() {
            None => self.scanner.finish_ordered(),
            Some(guided) => guided.finish(),
        }
    }

    fn reset(&mut self) -> String {
        let mut recovered = String::new();
        if let Some(guided) = self.guided.as_mut() {
            recovered.push_str(&guided.reset(self.prefill));
        }
        recovered.push_str(&self.scanner.reset());
        self.scanner.set_reasoning_mode(
            self.prefill != UnifiedParserPrefill::Response,
            self.prefill == UnifiedParserPrefill::Reasoning,
        );
        self.started = false;
        self.finished = false;
        recovered
    }
}

/// Where a guided-decoding stream currently is, relative to the reasoning span.
///
/// Guided decoding constrains only the TOOL output to JSON; the model still
/// opens and closes its reasoning channel with native markers, so those have to
/// be stripped before the remainder can be parsed as JSON.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum GuidedMode {
    #[default]
    OutsideReasoning,
    Reasoning,
    /// Visible output has started; every later byte is JSON payload. Marker-like
    /// text inside an argument value must stay literal from here on (`I7`).
    VisibleOnly,
}

/// One guided tool call as the backend emits it. `parameters` and `arguments`
/// are accepted interchangeably because backends disagree on the key.
#[derive(Debug, serde::Deserialize)]
struct GuidedToolCall {
    name: String,
    #[serde(default, deserialize_with = "deserialize_present_raw")]
    parameters: Option<Box<serde_json::value::RawValue>>,
    #[serde(default, deserialize_with = "deserialize_present_raw")]
    arguments: Option<Box<serde_json::value::RawValue>>,
}

/// Preserve a present JSON value as raw bytes, including `null`.
///
/// Serde's normal `Option<T>` field handling maps both a missing field and an
/// explicit `null` to `None`. Guided calls need the distinction: missing means
/// a parameterless call, while present `null` is a malformed argument value.
fn deserialize_present_raw<'de, D>(
    deserializer: D,
) -> std::result::Result<Option<Box<serde_json::value::RawValue>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Box::<serde_json::value::RawValue>::deserialize(deserializer).map(Some)
}

/// Request-scoped state for a guided-decoding stream.
///
/// Grammar-independent: the only thing it needs from the family is the pair of
/// reasoning markers, taken from the scanner's own [`ReasoningSpec`]. Guided
/// decoding is a BACKEND feature — any family can be served with it — so this
/// lives on the shared unified parser rather than in one family's module.
struct GuidedState {
    reasoning: ReasoningSpec,
    /// EVERY control marker of the family's tool grammar, from the scanner's own
    /// declaration. One set, used for both lookup and chunk-boundary holdback, in
    /// both the inside-a-thought and outside-a-thought scopes. Assembling it
    /// per-site from openers and orphans is how those four uses drifted apart.
    control_markers: Vec<String>,
    /// Paired with a stripped prefix-form invoke opener; see `invoke_end()`.
    invoke_end: String,
    named_tool: Option<String>,
    /// Response prefill disables reasoning markers, but tool control markers
    /// still need scanning until the JSON value actually starts.
    reasoning_enabled: bool,
    mode: GuidedMode,
    /// Some backends re-emit the reasoning opener even though the prompt already
    /// opened the channel. Consume exactly one such echo instead of leaking it.
    accept_redundant_reasoning_start: bool,
    input: String,
    json: String,
}

/// Earliest control marker in `haystack`, as `(pos, consume_len)`.
///
/// A PREFIX-form marker (`<function=`, which anchors `<function=NAME>`) consumes
/// through its terminating `>`; stripping only the declared prefix left `NAME>`
/// behind to poison the payload. `None` for a prefix form whose `>` has not
/// streamed yet, so the caller holds the bytes back instead of splitting it.
fn control_marker_at(
    haystack: &str,
    markers: &[String],
    invoke_end: &str,
    flush: bool,
) -> Option<(usize, usize)> {
    markers
        .iter()
        .filter_map(|m| {
            let at = haystack.find(m.as_str())?;
            if m.ends_with('=') {
                match haystack[at..].find('>') {
                    // An invoke opener owns its terminator: stripping `<function=NAME>`
                    // and leaving `</function>` behind put that fragment in the shown
                    // thinking. Consume the pair when the tail is present; a BARE
                    // terminator elsewhere stays text, as it is natively.
                    Some(rel) => Some((
                        at,
                        haystack[at..]
                            .find(invoke_end)
                            .map_or(rel + 1, |e| e + invoke_end.len()),
                    )),
                    None if flush => Some((at, haystack.len() - at)),
                    None => None,
                }
            } else {
                Some((at, m.len()))
            }
        })
        .min_by_key(|(at, _)| *at)
}

/// Trailing bytes the guided drain must retain across a chunk boundary.
///
/// Two reasons to hold back, and missing either one loses the payload:
/// a marker SPLIT across the boundary (`<tool_ca` | `ll>`), and a COMPLETE
/// prefix-form marker still waiting for its terminator (`<function=` | `NAME>`).
/// The second is not a partial marker, so the prefix scan does not see it, and it
/// was flushed into the payload buffer where it broke the parse.
fn guided_holdback_len(
    input: &str,
    reasoning_markers: &[&str],
    control: &[String],
    flush: bool,
) -> usize {
    if flush {
        return 0;
    }
    let split = marker_prefix_suffix_len(
        input,
        reasoning_markers
            .iter()
            .copied()
            .chain(control.iter().map(String::as_str)),
    );
    let pending_prefix_form = control
        .iter()
        .filter(|m| m.ends_with('='))
        .filter_map(|m| input.rfind(m.as_str()))
        .filter(|at| !input[*at..].contains('>'))
        .map(|at| input.len() - at)
        .max()
        .unwrap_or(0);
    split.max(pending_prefix_form)
}

/// Whether a buffered guided run has opened a JSON value. Anything before the
/// first `{`/`[` is prose, not payload.
fn json_payload_started(buf: &str) -> bool {
    matches!(buf.trim_start().as_bytes().first(), Some(b'{') | Some(b'['))
}

fn json_payload_kind(payload: &str) -> &'static str {
    match serde_json::from_str::<serde_json::Value>(payload) {
        Ok(serde_json::Value::Object(_)) => "object",
        Ok(serde_json::Value::Array(_)) => "array",
        Ok(serde_json::Value::String(_)) => "string",
        Ok(serde_json::Value::Number(_)) => "number",
        Ok(serde_json::Value::Bool(_)) => "boolean",
        Ok(serde_json::Value::Null) => "null",
        Err(_) => "invalid_json",
    }
}

impl GuidedState {
    fn new(
        reasoning: ReasoningSpec,
        control_markers: Vec<String>,
        invoke_end: String,
        named_tool: Option<String>,
        prefill: UnifiedParserPrefill,
    ) -> Self {
        Self {
            reasoning,
            control_markers,
            invoke_end,
            named_tool,
            reasoning_enabled: prefill != UnifiedParserPrefill::Response,
            mode: Self::mode_for(prefill),
            accept_redundant_reasoning_start: prefill == UnifiedParserPrefill::Reasoning,
            input: String::new(),
            json: String::new(),
        }
    }

    fn mode_for(prefill: UnifiedParserPrefill) -> GuidedMode {
        match prefill {
            UnifiedParserPrefill::None => GuidedMode::OutsideReasoning,
            UnifiedParserPrefill::Reasoning => GuidedMode::Reasoning,
            UnifiedParserPrefill::Response => GuidedMode::OutsideReasoning,
        }
    }

    fn push(&mut self, chunk: &str) -> Vec<UnifiedDelta> {
        if self.mode == GuidedMode::VisibleOnly {
            self.json.push_str(chunk);
            return Vec::new();
        }
        self.input.push_str(chunk);
        self.drain(false)
    }

    fn finish(&mut self) -> Result<Vec<UnifiedDelta>> {
        let mut output = self.drain(true);
        output.extend(self.finish_json()?);
        Ok(output)
    }

    fn reset(&mut self, prefill: UnifiedParserPrefill) -> String {
        let mut recovered = std::mem::take(&mut self.json);
        recovered.push_str(&std::mem::take(&mut self.input));
        // Buffers alone are not the state. Leaving `mode` at VisibleOnly would make
        // the NEXT stream treat its reasoning as JSON payload and surface it as text,
        // so put the channel back where `new` would have started it.
        self.mode = Self::mode_for(prefill);
        self.reasoning_enabled = prefill != UnifiedParserPrefill::Response;
        self.accept_redundant_reasoning_start = prefill == UnifiedParserPrefill::Reasoning;
        recovered
    }

    /// Strip the reasoning markers wrapping the JSON payload. Once visible
    /// output starts every later byte is JSON data, so native-looking strings
    /// inside argument values stay literal.
    fn drain(&mut self, flush: bool) -> Vec<UnifiedDelta> {
        let (start, end) = (self.reasoning.start, self.reasoning.end);
        let mut output = Vec::new();

        loop {
            match self.mode {
                GuidedMode::VisibleOnly => {
                    self.json.push_str(&self.input);
                    self.input.clear();
                    break;
                }
                GuidedMode::OutsideReasoning => {
                    // PAYLOAD FIRST. Once the run has opened a JSON value we are inside
                    // the payload, and a `<think>` from there on is ARGUMENT DATA, not a
                    // channel marker (`I7`). Searching for the opener before this check
                    // meant a whole-input push found the `<think>` embedded in an
                    // argument string, split the payload into text/reasoning/text and
                    // dropped the call — while the same bytes arriving in small chunks
                    // latched here first and parsed correctly. Same input, two answers,
                    // decided by chunking (`I6`).
                    if json_payload_started(&self.json)
                        || (self.json.trim().is_empty() && json_payload_started(&self.input))
                    {
                        self.mode = GuidedMode::VisibleOnly;
                        continue;
                    }

                    // A reasoning opener ANYWHERE ahead means the thought has not
                    // started yet and whatever precedes it is ordinary visible text —
                    // not the beginning of the JSON payload. Requiring a
                    // whitespace-only prefix here meant a turn that said anything
                    // before it began thinking (`content_then_reason`, the shape
                    // `UNIFIED.11.f`/`11.g` pin natively) fell through to the payload
                    // buffer, latched VisibleOnly, and then surfaced the markers AND
                    // the model's private thinking to the user as the answer, with the
                    // call never dispatched.
                    // Whichever marker comes FIRST wins — position is the ONLY
                    // precedence rule. Deciding by which branch was written first is
                    // what let an orphan closer ahead of a real thought ride out as
                    // text, and an opener beside a stripped closer survive into the
                    // payload. One set from the scanner covers both lookup and the
                    // holdback below, so the two cannot drift apart again.
                    let open_at = self
                        .reasoning_enabled
                        .then(|| self.input.find(start))
                        .flatten();
                    let stray_close = control_marker_at(
                        &self.input,
                        &self.control_markers,
                        &self.invoke_end,
                        flush,
                    )
                    .into_iter()
                    .chain(
                        self.reasoning_enabled
                            .then(|| self.input.find(end).map(|at| (at, end.len())))
                            .flatten(),
                    )
                    .min_by_key(|(at, _)| *at);
                    let close_at = stray_close.map(|(at, _)| at);
                    let close_len = stray_close.map(|(_, l)| l).unwrap_or(end.len());
                    let closer_first = matches!((open_at, close_at), (Some(o), Some(c)) if c < o)
                        || (open_at.is_none() && close_at.is_some());

                    if !closer_first && let Some(at) = open_at {
                        // Whatever was buffered as "payload so far", plus this prefix,
                        // was visible text after all — a thought is opening behind it.
                        let mut pending = std::mem::take(&mut self.json);
                        pending.push_str(&self.input[..at]);
                        if pending.trim().is_empty() {
                            self.json = pending;
                        } else {
                            push_run(&mut output, Kind::Text, &pending);
                        }
                        self.input.drain(..at + start.len());
                        self.mode = GuidedMode::Reasoning;
                        self.accept_redundant_reasoning_start = false;
                        continue;
                    }

                    // An orphan closer with no opener before it is malformed markup,
                    // stripped wherever it appears — the same rule the native scanner's
                    // orphan handler applies. Decide on the COMBINATION of what is
                    // already buffered and this prefix, as the opener branch does:
                    // judging the current prefix alone left prose buffered by an
                    // EARLIER chunk glued to the JSON that followed, losing the call.
                    if let Some(at) = close_at {
                        let mut pending = std::mem::take(&mut self.json);
                        pending.push_str(&self.input[..at]);
                        if pending.trim().is_empty() {
                            self.json = pending;
                        } else {
                            push_run(&mut output, Kind::Text, &pending);
                        }
                        self.input.drain(..at + close_len);
                        continue;
                    }

                    let keep = if flush {
                        0
                    } else {
                        // Same set as the lookup above, plus a complete prefix-form
                        // marker awaiting its `>`. This was `[start, end]` only, so a
                        // control marker split across a boundary went into the payload
                        // and was lost exactly like a whole one.
                        let reasoning_markers = if self.reasoning_enabled {
                            &[start, end][..]
                        } else {
                            &[]
                        };
                        guided_holdback_len(
                            &self.input,
                            reasoning_markers,
                            &self.control_markers,
                            flush,
                        )
                    };
                    let visible_len = self.input.len().saturating_sub(keep);
                    if visible_len > 0 {
                        self.json.push_str(&self.input[..visible_len]);
                        self.input.drain(..visible_len);
                        // Latch onto the payload only once it actually LOOKS like
                        // one. Guided decoding constrains the call to bare JSON, so a
                        // run that has not opened a value is prose, and a thought may
                        // still follow it in a later chunk. Latching on any
                        // non-whitespace byte is what let prose arriving in its own
                        // chunk swallow the thought that came after it.
                        if json_payload_started(&self.json) {
                            self.mode = GuidedMode::VisibleOnly;
                            continue;
                        }
                    }
                    if flush && !self.input.is_empty() {
                        self.json.push_str(&self.input);
                        self.input.clear();
                    }
                    break;
                }
                GuidedMode::Reasoning => {
                    if self.accept_redundant_reasoning_start {
                        let non_whitespace = self.input.trim_start();
                        let leading = self.input.len() - non_whitespace.len();
                        if non_whitespace.starts_with(start) {
                            push_run(&mut output, Kind::Reasoning, &self.input[..leading]);
                            self.input.drain(..leading + start.len());
                            self.accept_redundant_reasoning_start = false;
                            continue;
                        }
                        if !flush && start.starts_with(non_whitespace) {
                            push_run(&mut output, Kind::Reasoning, &self.input[..leading]);
                            self.input.drain(..leading);
                            break;
                        }
                        self.accept_redundant_reasoning_start = false;
                    }

                    // The closer ends the span; anything else in the stray set is
                    // malformed markup to strip, exactly as the native scanner does
                    // (`stray_in_reasoning`). Guided decoding constrains the TOOL
                    // payload, not the reasoning channel, so the model can still
                    // emit a duplicate opener or a stray tool close inside a thought
                    // — and being inside a thought must not turn markup into content
                    // (`I3`). Taking whichever lands first keeps the two request
                    // modes byte-identical on the same reasoning bytes.
                    // Three ways an open thought can end, in the same precedence the
                    // native scanner uses: its own closer; a TOOL OPENER, which
                    // terminates the span without being consumed because tool
                    // structure dominates reasoning; or a stray, which is stripped
                    // and leaves the span open.
                    let close = self.input.find(end).map(|at| (at, end.len(), true));
                    // Under guided decoding the reasoning channel is UNCONSTRAINED, so
                    // the model can legitimately narrate `<tool_call>` while thinking —
                    // and the real call arrives later as JSON, not as markup. So a tool
                    // opener here is STRAY markup to strip (span stays open), not
                    // structure that ends the turn. Terminating on it discarded the
                    // guided payload that followed and returned an empty response.
                    // The native scanner does treat it as structural, but it can: it
                    // opens a block and recovers the call from the markup itself.
                    let stray = control_marker_at(
                        &self.input,
                        &self.control_markers,
                        &self.invoke_end,
                        flush,
                    )
                    .into_iter()
                    .chain(self.input.find(start).map(|at| (at, start.len())))
                    .min_by_key(|(at, _)| *at)
                    .map(|(at, len)| (at, len, false));
                    if let Some((at, consume, closes)) = [close, stray]
                        .into_iter()
                        .flatten()
                        .min_by_key(|(at, _, _)| *at)
                    {
                        push_run(&mut output, Kind::Reasoning, &self.input[..at]);
                        self.input.drain(..at + consume);
                        if closes {
                            // Back to OutsideReasoning, NOT straight to VisibleOnly. The
                            // old latch was justified by keeping marker-like bytes inside
                            // a started payload literal, but the payload-first check at
                            // the top of that scope now owns that. Latching here instead
                            // meant markup AFTER a thought — `<think>x</think><tool_call>{…}`
                            // — was never examined, so the opener rode into the payload
                            // and the call was lost. This also handles several thoughts.
                            self.mode = GuidedMode::OutsideReasoning;
                        } else {
                            tracing::debug!(
                                why = "guided_stray_marker_in_reasoning",
                                "stream stripped malformed markup inside a reasoning span"
                            );
                        }
                        continue;
                    }

                    let keep = if flush {
                        0
                    } else {
                        guided_holdback_len(
                            &self.input,
                            &[start, end],
                            &self.control_markers,
                            flush,
                        )
                    };
                    let reasoning_len = self.input.len().saturating_sub(keep);
                    if reasoning_len > 0 {
                        push_run(&mut output, Kind::Reasoning, &self.input[..reasoning_len]);
                        self.input.drain(..reasoning_len);
                    }
                    break;
                }
            }
        }
        output
    }

    /// Parse the accumulated payload. Anything that does not parse as the
    /// expected call shape is surfaced as visible text rather than dropped
    /// (policy P2 — best-effort recovery, never silent loss).
    fn finish_json(&mut self) -> Result<Vec<UnifiedDelta>> {
        let payload = self.json.trim();
        if payload.is_empty() {
            self.json.clear();
            return Ok(Vec::new());
        }

        let raw_payload = self.json.clone();
        let calls = match &self.named_tool {
            // A named choice constrains output to that tool's ARGUMENTS alone,
            // so the payload is the argument object and the name is known.
            // Arguments are an OBJECT. A bare string / number / null / array is
            // syntactically valid JSON but is not an argument set, and dispatching it
            // would hand the tool a shape it cannot bind — surface it as text instead.
            Some(name) => serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(
                payload,
            )
            .ok()
            .map(|obj| {
                // The payload IS this tool's argument object — forward it verbatim.
                //
                // An earlier revision unwrapped a `{"name", "arguments"}` shape here to
                // tolerate a backend that emits the whole call envelope despite
                // `tool_choice` already naming the tool. That heuristic is unsound: the
                // shape is not exclusive to envelopes, and a tool like
                // `register_handler({"name": …, "parameters": …})` produces it. It broke
                // BOTH ways — a non-matching inner name voided a legitimate forced call
                // entirely, and a matching one forwarded only the inner value as the
                // argument set. Guided decoding is schema-constrained by the backend, so
                // the payload is trusted; a wrapping backend is out of spec and gets a
                // warning rather than a guess.
                if obj.contains_key("name")
                    && (obj.contains_key("arguments") || obj.contains_key("parameters"))
                {
                    tracing::warn!(
                        why = "guided_named_payload_looks_like_an_envelope",
                        named_tool = %name,
                        "named-choice payload carries `name` plus `arguments`/`parameters`; forwarding it verbatim as the argument set"
                    );
                }
                vec![(name.clone(), raw_payload)]
            }),
            None => parse_required_guided_calls(payload),
        };

        let Some(calls) = calls.filter(|calls| !calls.is_empty()) else {
            // Best-effort (P2): guided decoding promised a tool call and did not
            // deliver one, so the payload goes out as visible text rather than being
            // dropped. That recovery is NOT silent — to a caller the result is
            // indistinguishable from a model that simply chose to answer in prose,
            // so without this the backend's guided-decoding failure never surfaces.
            tracing::warn!(
                why = "unified_guided_json_not_a_tool_call",
                choice = if self.named_tool.is_some() {
                    "named"
                } else {
                    "required"
                },
                named_tool = self.named_tool.as_deref().unwrap_or("-"),
                payload_bytes = payload.len(),
                payload_kind = json_payload_kind(payload),
                "guided output did not parse as a tool call; emitting it as text"
            );
            return Ok(vec![UnifiedDelta::Text {
                text: std::mem::take(&mut self.json),
            }]);
        };

        self.json.clear();
        Ok(calls
            .into_iter()
            .enumerate()
            .map(|(tool_index, (name, arguments))| {
                UnifiedDelta::ToolCall(ToolCallDelta {
                    tool_index,
                    name: Some(name),
                    arguments,
                })
            })
            .collect())
    }
}

/// A required (un-named) choice emits one call object or an array of them.
fn parse_required_guided_calls(payload: &str) -> Option<Vec<(String, String)>> {
    fn convert(call: GuidedToolCall) -> Option<(String, String)> {
        // No argument key means NO ARGUMENTS, not a malformed call. `UNIFIED.6.a`
        // already fixes that semantic on the native path — same tool, no parameter
        // block, golden `arguments: {}` — so voiding it here made guided disagree with
        // native on an identical shape and made a parameterless tool uncallable. What
        // makes an element invalid is a missing `name` (required on GuidedToolCall);
        // that still voids the whole array, per `31.c` / `51.b`.
        let arguments = match (call.parameters, call.arguments) {
            // PRESENT but not an object is a malformed call, the same judgement the
            // NAMED path makes on its whole payload: arguments that are a string or
            // a number cannot bind to the tool's parameters, so dispatching would
            // hand the tool a shape it cannot use. Absent is different — that means
            // no arguments, and stays valid (see the note above).
            (Some(raw), None) | (None, Some(raw)) => {
                serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(raw.get())
                    .ok()?;
                raw.get().to_string()
            }
            (None, None) => "{}".to_string(),
            // The aliases are alternatives, not two independently meaningful
            // argument sets. Choosing one silently can dispatch different bytes from
            // what the backend intended, so reject the ambiguous call fail-closed.
            (Some(_), Some(_)) => return None,
        };
        Some((call.name, arguments))
    }

    if let Ok(raw_calls) = serde_json::from_str::<Vec<Box<serde_json::value::RawValue>>>(payload) {
        return raw_calls
            .into_iter()
            .map(|raw| {
                serde_json::from_str::<GuidedToolCall>(raw.get())
                    .ok()
                    .and_then(convert)
            })
            .collect();
    }

    serde_json::from_str::<GuidedToolCall>(payload)
        .ok()
        .and_then(convert)
        .map(|call| vec![call])
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
            let parser = match family {
                $($family $(| $alias)* => $ctor(tools),)+
                other => anyhow::bail!("no Dynamo unified parser for family '{other}'"),
            };
            let canonical = match family {
                $($family $(| $alias)* => $family,)+
                _ => unreachable!("matched above"),
            };

            // Parser construction happens per request, so keep the selection signal
            // at debug level. Operators can enable the target when diagnosing routing
            // without adding one production info line for every generation.
            tracing::debug!(
                target: "dynamo_parsers_v2",
                family = canonical,
                requested = family,
                "v2 UNIFIED parser active"
            );

            // Optional stderr instrumentation, same contract as the tool-only
            // registry: a host WITHOUT a tracing subscriber can still confirm it.
            if crate::tool_calling::debug::debug_enabled() {
                return Ok(DebugUnifiedParser::wrap(canonical, parser));
            }
            Ok(parser)
        }
    };
}

unified_registry! {
    "qwen3" | "qwen3_coder" => qwen3::qwen3_unified,
}

/// Stderr instrumentation for the unified path under `DYNAMO_PARSERS_DEBUG`.
///
/// The tool-only registry has wrapped its parsers since audit B9 so a host can
/// confirm a Dynamo parser was selected and is parsing. The unified registry did
/// not, so setting the flag and seeing nothing was indistinguishable from the
/// parser never being reached — the exact question the flag is turned on to
/// answer. This mirrors `DebugToolParser`: announce at construction, report the
/// resolved request mode at initialize, and report each batch of updates.
struct DebugUnifiedParser {
    family: &'static str,
    inner: Box<dyn UnifiedParser>,
}

impl DebugUnifiedParser {
    fn wrap(family: &'static str, inner: Box<dyn UnifiedParser>) -> Box<dyn UnifiedParser> {
        crate::tool_calling::debug::emit(format_args!("UNIFIED family={family} created"));
        Box::new(Self { family, inner })
    }

    fn log(&self, method: &str, deltas: &[UnifiedDelta]) {
        if deltas.is_empty() {
            return;
        }
        let calls = deltas
            .iter()
            .filter_map(|d| match d {
                UnifiedDelta::ToolCall(c) => Some(c.name.as_deref().unwrap_or("…")),
                _ => None,
            })
            .collect::<Vec<_>>();
        crate::tool_calling::debug::emit(format_args!(
            "UNIFIED family={} {} emitted {} delta(s) calls={:?}",
            self.family,
            method,
            deltas.len(),
            calls
        ));
    }
}

impl UnifiedParser for DebugUnifiedParser {
    fn initialize_with_output_mode(
        &mut self,
        prefill: UnifiedParserPrefill,
        tool_output_mode: UnifiedToolOutputMode<'_>,
    ) -> Result<()> {
        let mode = match tool_output_mode {
            UnifiedToolOutputMode::Native => "native".to_string(),
            UnifiedToolOutputMode::GuidedJson {
                named_tool: Some(n),
            } => {
                format!("guided_json(named={n})")
            }
            UnifiedToolOutputMode::GuidedJson { named_tool: None } => {
                "guided_json(required)".to_string()
            }
        };
        crate::tool_calling::debug::emit(format_args!(
            "UNIFIED family={} initialize prefill={prefill:?} tool_output_mode={mode}",
            self.family
        ));
        self.inner
            .initialize_with_output_mode(prefill, tool_output_mode)
    }

    fn initialize(&mut self, prefill: UnifiedParserPrefill) -> Result<()> {
        self.initialize_with_output_mode(prefill, UnifiedToolOutputMode::Native)
    }

    fn push(&mut self, chunk: &str) -> Result<Vec<UnifiedDelta>> {
        let out = self.inner.push(chunk)?;
        self.log("push", &out);
        Ok(out)
    }

    fn finish(&mut self) -> Result<Vec<UnifiedDelta>> {
        let out = self.inner.finish()?;
        self.log("finish", &out);
        Ok(out)
    }

    fn reset(&mut self) -> String {
        self.inner.reset()
    }

    // Everything below forwards to the wrapped parser. These have trait defaults,
    // so NOT forwarding them would silently swap a family's override for the
    // default the moment debug logging is switched on — the decode would differ
    // between a debug run and the run it is meant to explain. `DebugToolParser`
    // forwards its equivalents for the same reason. No family overrides these
    // today, which is exactly why the gap has to close before one does.

    fn preserve_special_tokens(&self) -> bool {
        self.inner.preserve_special_tokens()
    }

    fn tool_call_id(&self, tool_index: usize) -> Option<&str> {
        self.inner.tool_call_id(tool_index)
    }

    fn initialize_from_prompt(&mut self, prompt_text: &str) -> Result<()> {
        crate::tool_calling::debug::emit(format_args!(
            "UNIFIED family={} initialize_from_prompt prompt_len={}",
            self.family,
            prompt_text.len()
        ));
        self.inner.initialize_from_prompt(prompt_text)
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

    /// The returning and appending spellings are one implementation, so they
    /// cannot disagree — but only a test says so. Without this, `parse_into`
    /// could be re-implemented later and silently drift from `push`, which is
    /// the divergent-copy failure this crate keeps hitting.
    #[test]
    fn parse_into_and_push_agree_chunk_for_chunk() {
        let chunks = [
            "<think>weigh it</think>ok ",
            "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n",
            "</function>\n</tool_call>done",
        ];

        let mut pushed = Vec::new();
        let mut a = create_unified_parser_for_family("qwen3", &[]).unwrap();
        for c in chunks {
            pushed.extend(a.push(c).unwrap());
        }
        pushed.extend(a.finish().unwrap());

        let mut appended = UnifiedParserOutput::default();
        let mut b = create_unified_parser_for_family("qwen3", &[]).unwrap();
        for c in chunks {
            b.parse_into(c, &mut appended).unwrap();
        }
        appended.append(&mut b.finish_into().unwrap());

        assert_eq!(
            pushed, appended.events,
            "parse_into must commit exactly what push returns, in the same order"
        );
        assert_eq!(assemble(&pushed), appended.assembled());
        assert!(
            pushed
                .iter()
                .any(|d| matches!(d, UnifiedDelta::ToolCall(_))),
            "fixture should produce a call, otherwise this asserts nothing"
        );
    }

    /// The batch path must be able to emit the model's argument bytes, not a
    /// re-serialization of them. Without this, a non-streaming turn rewrites
    /// `{"city": "Tokyo"}` to `{"city":"Tokyo"}` while the streaming path (which
    /// forwards `ToolCallDelta.arguments`) does not — the two disagree on
    /// identical input, which is an `I6`/`I7` break on the batch path alone.
    #[test]
    fn raw_arguments_survive_assembly_verbatim() {
        let spaced = r#"{"city": "Tokyo",  "unit": "c"}"#;
        let deltas = vec![
            UnifiedDelta::Text { text: "ok".into() },
            call(0, Some("get_weather"), &spaced[..14]),
            call(0, None, &spaced[14..]),
        ];

        let raw = tool_arguments_raw(&deltas);
        assert_eq!(
            raw.get(&0).map(String::as_str),
            Some(spaced),
            "fragments must rejoin byte-for-byte"
        );

        // The assembled view still parses, so semantic consumers are unchanged.
        let events = assemble(&deltas);
        let UnifiedEvent::ToolCall { arguments, .. } = &events[1] else {
            panic!("expected a tool call at position 1, got {events:?}")
        };
        assert_eq!(arguments["city"], "Tokyo");
        // …and the re-serialization is genuinely lossy, which is why raw is needed.
        assert_ne!(serde_json::to_string(arguments).unwrap(), spaced);
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

#[cfg(test)]
mod debug_marker_tests {
    use crate::tool_calling::debug::{DEBUG_ENV, is_truthy};

    /// The values `DYNAMO_PARSERS_DEBUG` must accept.
    ///
    /// Born from a real miss: the flag was set, the unified parser demonstrably
    /// ran, and no marker appeared — so an operator (and the author) concluded
    /// the parser was never reached. It cost hours.
    ///
    /// This asserts the PURE predicate, not `debug_enabled()`. That wrapper
    /// caches in a `OnceLock`, so a test that sets the env and calls it only
    /// passes when it happens to run before anything else touches the lock —
    /// which is exactly how the first version of this test passed locally and
    /// failed in CI. A global latch is not testable in-process; the parsing
    /// rule it depends on is.
    #[test]
    fn debug_env_accepts_the_documented_truthy_values() {
        assert_eq!(DEBUG_ENV, "DYNAMO_PARSERS_DEBUG");
        for v in ["1", "true", "TRUE", "on", "yes", "Yes"] {
            assert!(is_truthy(v), "{v:?} should enable debug output");
        }
        for v in ["0", "false", "off", "no", "", "maybe"] {
            assert!(!is_truthy(v), "{v:?} must NOT enable debug output");
        }
    }
}
