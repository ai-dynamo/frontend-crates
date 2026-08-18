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
//! [`UnifiedParserEvent`] is the streaming vocabulary — what one parser advance
//! produced, in order. [`UnifiedEvent`] is the assembled view: adjacent same-kind deltas
//! coalesced and per-call argument fragments joined into one typed object
//! (`I8`). [`assemble`] is the single implementation of that fold, so callers
//! and conformance harnesses never reimplement it and drift.
//!
//! Note the distinction between a unified INTERFACE and a unified PARSER. An
//! interface that internally calls a reasoning parser and then hands its leftover
//! content to a tool parser still has the seam described above — it has only moved
//! it behind one method. This is the latter: one state machine sees every byte and
//! decides its channel, so there is no handoff for ordering to be lost across.
//!
//! The public contract here is deliberately ALIGNED WITH PEER TRAITS — the Rust
//! streaming-parser contracts serving engines already expose — so this crate can be
//! adopted without a translation layer. Where it diverges from the peer shape, the
//! divergence is stated at the item.

pub mod qwen3;

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::tool_calling::scan::{
    InvokeEmitter, InvokeScan, ReasoningSpec, WrappedBlockScanner, marker_prefix_suffix_len,
    push_run, reasoning_opener_len,
};
use crate::tool_calling::traits::{Result, Tool, ToolCallDelta, ToolParseResult};

/// One ordered update produced while parsing assistant output.
///
/// This is the streaming vocabulary shared by the whole crate: the marker-scan
/// core emits it, tool-only parsers project it down to [`ToolParseResult`], and
/// unified parsers hand it to the caller as-is.
///
/// Name, variant order and payload shapes are aligned with the peer trait's event
/// type, so the two translate variant-for-variant under a compiler rather than by
/// a reader's judgement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnifiedParserEvent {
    /// Normal assistant-visible text.
    Text(String),
    /// Reasoning text hidden from the normal content stream.
    Reasoning(String),
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
pub enum UnifiedParserStartingState {
    /// Generated output includes any channel-opening marker itself.
    #[default]
    None,
    /// The prompt opened reasoning, so generated output begins inside it.
    Reasoning,
    /// The prompt opened the visible response channel.
    Response,
}

/// Tool-call wire format selected for one request.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum UnifiedToolOutputMode {
    /// Model-native tool-call markup.
    #[default]
    Native,
    /// Guided decoding emits bare JSON instead of model-native markup.
    ///
    /// A named choice contains only that tool's arguments. A required choice
    /// contains one call object or an array of call objects.
    GuidedJson { named_tool: Option<String> },
}

/// Caller policy when guided decoding violates the promised tool-call shape.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum InvalidGuidedPayloadPolicy {
    /// Return a typed error so the serving layer can reject or retry the output.
    #[default]
    Reject,
    /// Preserve the corpus' best-effort behavior by surfacing the bytes as text.
    RecoverAsText,
}

/// Fully resolved request-scoped parser configuration.
///
/// Prompt inspection and backend request resolution happen before this value is
/// built. Passing one owned object prevents starting state, wire format, and
/// malformed-payload policy from being initialized through paths that can drift.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct UnifiedParserInit {
    pub prompt_token_ids: Vec<u32>,
    pub starting_state: UnifiedParserStartingState,
    pub tool_output_mode: UnifiedToolOutputMode,
    pub invalid_guided_payload: InvalidGuidedPayloadPolicy,
}

impl UnifiedParserInit {
    /// Neutral native-mode initialization used by peer-shaped callers.
    pub fn native(prompt_token_ids: &[u32]) -> Self {
        Self {
            prompt_token_ids: prompt_token_ids.to_vec(),
            ..Self::default()
        }
    }
}

/// Coarse payload classification carried by [`InvalidGuidedPayload`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InvalidGuidedPayloadKind {
    Missing,
    InvalidJson,
    WrongShape,
}

/// Typed guided-decoding contract failure.
///
/// Raw model bytes are deliberately excluded. Callers can downcast the error and
/// recover uncommitted bytes through [`UnifiedParser::reset`] without leaking them
/// into logs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidGuidedPayload {
    pub kind: InvalidGuidedPayloadKind,
    pub choice: &'static str,
}

impl std::fmt::Display for InvalidGuidedPayload {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "guided decoding emitted {:?} instead of the promised {} tool-call payload",
            self.kind, self.choice
        )
    }
}

impl std::error::Error for InvalidGuidedPayload {}

/// A parser that owns reasoning + content + tool calls for one stream.
///
/// Streaming-first, like [`crate::ToolParser`]: [`Self::parse_into`] per decoded
/// delta, [`Self::finish`] once at end of stream. One instance parses exactly one
/// choice of one request, which is what gives per-stream isolation (`I4`) by
/// construction.
///
/// # This is a SYNTAX-only surface
///
/// A parser reports what the model's bytes SAY, never whether saying it was
/// allowed. An emitted [`UnifiedParserEvent::ToolCall`] therefore carries no
/// guarantee that the tool was offered, that `tool_choice` permits it, that the
/// arguments satisfy the tool's schema, or that parallel calls are allowed. A
/// model can name a tool that was never offered, and this will emit it.
///
/// The serving layer MUST validate tool availability, authorization, argument
/// schema and call policy before executing anything derived from these events.
/// Splitting it this way is deliberate — a parser that silently dropped
/// unrecognized calls would hide model misbehaviour that the caller needs to see —
/// but it means "the parser produced a ToolCall" is not "this call is safe to run".
pub trait UnifiedParser: Send {
    /// Initialize parser state from prompt token IDs before output deltas arrive.
    ///
    /// Aligned with the peer trait, so a caller written against it reaches the same
    /// method with the same argument here.
    ///
    /// This is the peer-trait compatibility adapter. Request-aware callers resolve
    /// every request fact once and call [`UnifiedParser::initialize_request`].
    fn initialize(&mut self, prompt_token_ids: &[u32]) -> Result<()> {
        self.initialize_request(UnifiedParserInit::native(prompt_token_ids))
    }

    /// Apply the one fully resolved request configuration before parsing starts.
    fn initialize_request(&mut self, init: UnifiedParserInit) -> Result<()> {
        if init.starting_state != UnifiedParserStartingState::None {
            anyhow::bail!("this unified parser does not support prompt-prefilled channels");
        }
        if init.tool_output_mode != UnifiedToolOutputMode::Native {
            anyhow::bail!("this unified parser does not support guided tool output");
        }
        Ok(())
    }

    /// Feed one decoded text delta, appending committed events into `output`.
    ///
    /// THE required method, matching the peer traits. It is the only method that
    /// advances the parser; [`UnifiedParserExt::push`] and
    /// [`UnifiedParserExt::parse_complete`] are non-overridable conveniences defined
    /// in terms of it, so every family has one advance implementation.
    ///
    /// Error contract, aligned with the peer traits: on `Err`, whatever was already
    /// appended to `output` stays committed and the parser's uncommitted buffer is
    /// intact, so the caller can recover it with [`UnifiedParser::reset`].
    ///
    /// This guarantee is specific to `parse_into`, because the caller owns `output` and
    /// can still read it after an error. [`UnifiedParserExt::push`] owns its buffer and
    /// returns `Result<Vec<_>>`, which has nowhere to carry partial output — a parser
    /// that may commit events and THEN fail must be driven through `parse_into`.
    fn parse_into(&mut self, delta: &str, output: &mut UnifiedParserOutput) -> Result<()>;

    /// Flush buffered partial state at end of stream.
    ///
    /// Open reasoning is promoted here rather than dropped or leaked as text,
    /// and an unrecoverable partial tool call is dropped without erroring
    /// (policy P2 — best-effort recovery).
    ///
    /// The peer traits give this a default that returns nothing. It is REQUIRED here:
    /// the signature a caller sees is identical, but a family that forgets to flush
    /// would silently drop the tail of every stream, and that is not a failure worth
    /// inheriting for symmetry's sake.
    fn finish(&mut self) -> Result<UnifiedParserOutput>;

    /// Return the parser to a FRESH-STREAM state and hand back any unconsumed text.
    ///
    /// This is not a mid-turn continuation hook. Everything restarts, including the
    /// tool index, so the returned text must be re-parsed as a NEW stream and any
    /// calls already emitted belong to the abandoned one — feeding the remainder
    /// back into the same turn would re-number from index 0 and collide with them.
    /// That follows from `I4`: one parser instance owns exactly one stream, so a
    /// retry is a new stream and gets a reset (or a new) parser, not a splice.
    fn reset(&mut self) -> String {
        String::new()
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
}

/// Allocation conveniences over the required [`UnifiedParser`] lifecycle.
///
/// These methods live in a blanket extension trait so parser implementations cannot
/// override them and create a second advance path. Import this trait to call them.
pub trait UnifiedParserExt: UnifiedParser {
    /// Feed one decoded text delta; returns the events it committed, in order.
    ///
    /// This allocates a fresh output per advance, which is why a serving loop prefers
    /// [`UnifiedParser::parse_into`]. On `Err`, committed events in that local output
    /// cannot be recovered through `Result<Vec<_>>`; use `parse_into` when partial
    /// committed output must survive an error.
    fn push(&mut self, chunk: &str) -> Result<Vec<UnifiedParserEvent>> {
        let mut out = UnifiedParserOutput::default();
        self.parse_into(chunk, &mut out)?;
        Ok(out.events)
    }

    /// Parse complete output through `parse_into` + `finish`, then assemble.
    ///
    /// The fixed lifecycle makes stream/batch parity (`I6`) structural instead of a
    /// property two independently overridable paths have to agree on.
    fn parse_complete(&mut self, text: &str) -> Result<Vec<UnifiedEvent>> {
        let mut out = UnifiedParserOutput::default();
        self.parse_into(text, &mut out)?;
        out.append(&mut self.finish()?);
        Ok(assemble(&out.events))
    }
}

impl<T: UnifiedParser + ?Sized> UnifiedParserExt for T {}

/// Ordered updates committed by one parser advance.
///
/// Aligned with the peer traits' output type: a vector, not a bundle of parallel
/// channel fields. That is the whole point — a bundle cannot say whether text came
/// before or after a call, which is the ordering this surface exists to pin.
///
/// # The buffer is CUMULATIVE, and appending COALESCES
///
/// One buffer may be driven through many advances. `push_text` and `push_reasoning`
/// merge into a trailing event of the same kind, so two advances carrying `"hel"`
/// then `"lo"` produce ONE `Text("hello")`, not two events.
///
/// Two consequences a caller must know, because they were previously unstated and the
/// shipped implementations answered them differently:
///
/// - **Do not index a "what did this advance produce" window.** A watermark loop —
///   record `len()`, advance, read `events[n..]` — can legally observe NOTHING, because
///   the new bytes may have merged into `events[n - 1]`. Use [`UnifiedParserExt::push`],
///   which returns exactly one advance's events, when that is the question.
/// - **Append through the helpers, not `events.extend`.** `extend` bypasses the merge
///   and yields a different event vector for identical bytes. [`Self::append`] is NOT an
///   exception: it routes every event through these same helpers, so joining two
///   independently-built buffers gives the same result as accumulating straight through.
///
/// # This is a CONVENTION, not an enforced invariant
///
/// `events` is public — matching the peer type, which is the point of this surface — so
/// nothing stops a caller writing `UnifiedParserOutput { events: vec![Text("hel"), Text("lo")] }`
/// or `out.events.extend(..)` and holding a value that breaks the merge rule. Every
/// route this crate owns (the push helpers, [`Self::append`], `FromIterator`, and the
/// scanner's sink) does apply it; direct field access does not, and cannot be made to
/// without diverging from the peer shape. So: build through the helpers. A value that
/// did not come through them may carry adjacent same-kind events.
///
/// [`assemble`] performs the same fold, so a caller that coalesces here and one that
/// folds afterwards agree on the assembled result either way.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct UnifiedParserOutput {
    /// Updates in the order the model produced them.
    pub events: Vec<UnifiedParserEvent>,
}

impl crate::tool_calling::scan::EventSink for UnifiedParserOutput {
    fn push_text(&mut self, text: &str) {
        UnifiedParserOutput::push_text(self, text);
    }

    fn push_reasoning(&mut self, text: &str) {
        UnifiedParserOutput::push_reasoning(self, text);
    }

    fn push_call(&mut self, call: ToolCallDelta) {
        UnifiedParserOutput::push_call(self, call);
    }
}

impl UnifiedParserOutput {
    fn push_event(&mut self, event: UnifiedParserEvent) {
        match event {
            UnifiedParserEvent::Text(text) => self.push_text(text),
            UnifiedParserEvent::Reasoning(text) => self.push_reasoning(text),
            UnifiedParserEvent::ToolCall(call) => self.push_call(call),
        }
    }

    /// Append another advance's updates, preserving order.
    ///
    /// Coalesces ACROSS the seam, matching the peer helper: routing every event through
    /// `push_text`/`push_reasoning`/`push_call` means two buffers joined here produce the
    /// same events as one buffer accumulated straight through. A plain `Vec::append` left
    /// `Text("hello") + Text(" world")` as two events where the peer yields one, so the
    /// same bytes described a different event stream depending on how the caller batched
    /// them.
    pub fn append(&mut self, other: &mut Self) {
        for event in std::mem::take(&mut other.events) {
            match event {
                UnifiedParserEvent::Text(t) => self.push_text(t),
                UnifiedParserEvent::Reasoning(t) => self.push_reasoning(t),
                UnifiedParserEvent::ToolCall(c) => self.push_call(c),
            }
        }
    }

    // --- Accumulation helpers, aligned with the peer traits in name and semantics
    // These COALESCE: appending text onto a trailing text event extends it rather
    // than adding a second one. `assemble` performs the same fold, so a caller
    // that accumulates through these and one that folds afterwards agree. Nothing
    // in this crate's parse path routes through them today — the corpus asserts on
    // the raw per-advance events — so adopting them cannot move the golden feed.

    /// Append one visible text event if `delta` is non-empty.
    pub fn push_text(&mut self, delta: impl AsRef<str> + Into<String>) {
        if delta.as_ref().is_empty() {
            return;
        }
        if let Some(UnifiedParserEvent::Text(last)) = self.events.last_mut() {
            last.push_str(delta.as_ref());
            return;
        }
        self.events.push(UnifiedParserEvent::Text(delta.into()));
    }

    /// Append one reasoning text event if `delta` is non-empty.
    pub fn push_reasoning(&mut self, delta: impl AsRef<str> + Into<String>) {
        if delta.as_ref().is_empty() {
            return;
        }
        if let Some(UnifiedParserEvent::Reasoning(last)) = self.events.last_mut() {
            last.push_str(delta.as_ref());
            return;
        }
        self.events
            .push(UnifiedParserEvent::Reasoning(delta.into()));
    }

    /// Append one tool-call event.
    pub fn push_call(&mut self, call: ToolCallDelta) {
        self.events.push(UnifiedParserEvent::ToolCall(call));
    }

    /// Whether this advance committed nothing.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Number of committed events.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Borrowing iterator over the committed events, in order.
    pub fn iter(&self) -> std::slice::Iter<'_, UnifiedParserEvent> {
        self.events.iter()
    }
}

// Additive ergonomics: the type carries a single `events` field, so these cannot
// change what is emitted — they only spare every caller
// an explicit `.events` when consuming an advance.
impl IntoIterator for UnifiedParserOutput {
    type Item = UnifiedParserEvent;
    type IntoIter = std::vec::IntoIter<UnifiedParserEvent>;
    fn into_iter(self) -> Self::IntoIter {
        self.events.into_iter()
    }
}

impl<'a> IntoIterator for &'a UnifiedParserOutput {
    type Item = &'a UnifiedParserEvent;
    type IntoIter = std::slice::Iter<'a, UnifiedParserEvent>;
    fn into_iter(self) -> Self::IntoIter {
        self.events.iter()
    }
}

impl FromIterator<UnifiedParserEvent> for UnifiedParserOutput {
    fn from_iter<T: IntoIterator<Item = UnifiedParserEvent>>(iter: T) -> Self {
        // Through the helpers, like every other way of building this type. A plain
        // `collect()` bypassed the merge, so `collect()`ing two adjacent `Text` events
        // produced a different event stream than pushing the same bytes — the same
        // defect `append` had, one constructor over.
        let mut out = Self::default();
        for event in iter {
            match event {
                UnifiedParserEvent::Text(t) => out.push_text(t),
                UnifiedParserEvent::Reasoning(t) => out.push_reasoning(t),
                UnifiedParserEvent::ToolCall(c) => out.push_call(c),
            }
        }
        out
    }
}

impl UnifiedParserOutput {
    /// Collapse into assembled events (see [`assemble`]).
    ///
    /// Additive: the peer traits have no assembled form, so their arguments stay
    /// string fragments end to end.
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
pub fn tool_arguments_raw(deltas: &[UnifiedParserEvent]) -> BTreeMap<usize, String> {
    let mut raw: BTreeMap<usize, String> = BTreeMap::new();
    for delta in deltas {
        if let UnifiedParserEvent::ToolCall(call) = delta {
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
pub fn assemble(deltas: &[UnifiedParserEvent]) -> Vec<UnifiedEvent> {
    // Coalesce adjacent same-kind runs with the SAME helper the scan core uses, so
    // `I8` has exactly ONE implementation instead of one per type.
    let mut merged: Vec<UnifiedParserEvent> = Vec::new();
    for delta in deltas {
        match delta {
            UnifiedParserEvent::Reasoning(text) => push_run(&mut merged, Kind::Reasoning, text),
            UnifiedParserEvent::Text(text) => push_run(&mut merged, Kind::Text, text),
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
            UnifiedParserEvent::Reasoning(text) => out.push(UnifiedEvent::Reasoning { text }),
            UnifiedParserEvent::Text(text) => out.push(UnifiedEvent::Text { text }),
            UnifiedParserEvent::ToolCall(call) => {
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
    pub fn from_deltas(deltas: Vec<UnifiedParserEvent>) -> Self {
        let mut out = Self::default();
        for delta in deltas {
            match delta {
                UnifiedParserEvent::Reasoning(text) | UnifiedParserEvent::Text(text) => {
                    out.normal_text.push_str(&text)
                }
                UnifiedParserEvent::ToolCall(call) => out.calls.push(call),
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
    starting_state: UnifiedParserStartingState,
    /// Set once the backend selects guided decoding for this request; `None`
    /// on the native path, which is every request that did not ask for it.
    /// Boxed so a native stream carries one null pointer, not six idle fields.
    guided: Option<Box<GuidedState>>,
    started: bool,
    finished: bool,
}

impl<E: InvokeEmitter> ScannerUnified<E> {
    pub(crate) fn new(scanner: WrappedBlockScanner<E>) -> Self {
        Self {
            scanner,
            starting_state: UnifiedParserStartingState::None,
            guided: None,
            started: false,
            finished: false,
        }
    }

    fn apply_init(&mut self, init: UnifiedParserInit) -> Result<()> {
        if self.started || self.finished {
            anyhow::bail!("cannot initialize a unified parser after parsing has started");
        }
        let UnifiedParserInit {
            prompt_token_ids: _,
            starting_state,
            tool_output_mode,
            invalid_guided_payload,
        } = init;
        // An EXPLICIT `Reasoning` demand that this family cannot represent must fail
        // here, before anything is mutated. `set_reasoning_mode` folds "cannot
        // represent" into "off", which is right for the neutral `None` default but
        // wrong for a caller that stated the prompt already opened a thought: the
        // model then closes a channel that was never open, and its private reasoning
        // is emitted as VISIBLE TEXT. Rejecting is the only honest answer, and it
        // belongs here rather than in `set_reasoning_mode`, whose generic
        // `enabled=true` cannot tell an explicit demand from the default.
        if starting_state == UnifiedParserStartingState::Reasoning
            && self.scanner.reasoning_spec().is_none()
        {
            anyhow::bail!(
                "starting_state=Reasoning was requested, but this family has no reasoning \
                 channel to continue; its private reasoning would be emitted as visible text"
            );
        }
        // EVERY prerequisite is resolved BEFORE any mutation. Guided mode used to be
        // validated after `starting_state`, the scanner reset and the reasoning mode
        // had already been applied, so a rejected initialization left the parser
        // half-configured, and a caller that caught the error and retried in a
        // supported mode was building on mutated state. A failed initialize must be a
        // no-op.
        let guided_reasoning = match tool_output_mode {
            UnifiedToolOutputMode::Native => None,
            UnifiedToolOutputMode::GuidedJson { .. } => match self.scanner.reasoning_spec() {
                Some(reasoning) => Some(reasoning),
                None => anyhow::bail!("guided tool output needs a reasoning-aware scanner"),
            },
        };

        self.starting_state = starting_state;
        self.scanner.reset();
        // `Response` means the prompt already opened visible content, so this
        // stream has no reasoning channel at all; `Reasoning` means it opened a
        // thought the model will close without ever emitting the opener.
        self.scanner.set_reasoning_mode(
            starting_state != UnifiedParserStartingState::Response,
            match starting_state {
                UnifiedParserStartingState::None => None,
                UnifiedParserStartingState::Reasoning => Some(true),
                UnifiedParserStartingState::Response => Some(false),
            },
        );
        self.guided = match (tool_output_mode, guided_reasoning) {
            (UnifiedToolOutputMode::Native, _) => None,
            (UnifiedToolOutputMode::GuidedJson { named_tool }, Some(reasoning)) => {
                Some(Box::new(GuidedState::new(
                    reasoning,
                    GuidedGrammar {
                        control_markers: self.scanner.control_markers().to_vec(),
                        invoke_start: self.scanner.invoke_start().to_string(),
                        invoke_end: self.scanner.invoke_end().to_string(),
                        invoke_scan: self.scanner.invoke_scan(),
                    },
                    named_tool,
                    starting_state,
                    invalid_guided_payload,
                )))
            }
            (UnifiedToolOutputMode::GuidedJson { .. }, None) => {
                unreachable!("guided mode without a reasoning spec bails before any mutation")
            }
        };
        self.finished = false;
        Ok(())
    }
}

impl<E: InvokeEmitter + Send> UnifiedParser for ScannerUnified<E> {
    /// Whether decoding must preserve special tokens, delegated to the shared grammar.
    ///
    /// NOT the trait default, and NOT a field on this adapter. Inheriting `false` here
    /// while the tool-only adapter over this same scanner returned `true` meant two
    /// surfaces reported contradictory decoding requirements for identical markup.
    fn preserve_special_tokens(&self) -> bool {
        self.scanner.preserve_special_tokens()
    }

    fn initialize_request(&mut self, init: UnifiedParserInit) -> Result<()> {
        self.apply_init(init)
    }

    fn parse_into(&mut self, delta: &str, output: &mut UnifiedParserOutput) -> Result<()> {
        if self.finished {
            anyhow::bail!("cannot push to a finished unified parser");
        }
        self.started = true;
        match self.guided.as_mut() {
            // Native: straight into the caller's output. An event written here is
            // COMMITTED, so a later error in the same advance cannot retract it.
            None => self.scanner.push_ordered_into(delta, output),
            // Guided: the guided parser owns its own vector, so route what it returns
            // through the accumulation helpers. NOT `output.events.extend(..)`, which
            // bypasses the same-kind merge and makes identical bytes describe a
            // different event stream depending on which mode produced them.
            Some(guided) => guided.push_into(delta, output),
        }
    }

    fn finish(&mut self) -> Result<UnifiedParserOutput> {
        if self.finished {
            anyhow::bail!("cannot finish a unified parser twice");
        }
        self.started = true;
        self.finished = true;
        let mut out = UnifiedParserOutput::default();
        match self.guided.as_mut() {
            None => self.scanner.finish_ordered_into(&mut out)?,
            Some(guided) => {
                for event in guided.finish()? {
                    match event {
                        UnifiedParserEvent::Text(t) => out.push_text(t),
                        UnifiedParserEvent::Reasoning(t) => out.push_reasoning(t),
                        UnifiedParserEvent::ToolCall(c) => out.push_call(c),
                    }
                }
            }
        }
        Ok(out)
    }

    fn reset(&mut self) -> String {
        let mut recovered = String::new();
        if let Some(guided) = self.guided.as_mut() {
            recovered.push_str(&guided.reset(self.starting_state));
        }
        recovered.push_str(&self.scanner.reset());
        self.scanner.set_reasoning_mode(
            self.starting_state != UnifiedParserStartingState::Response,
            match self.starting_state {
                UnifiedParserStartingState::None => None,
                UnifiedParserStartingState::Reasoning => Some(true),
                UnifiedParserStartingState::Response => Some(false),
            },
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
struct GuidedGrammar {
    control_markers: Vec<String>,
    invoke_start: String,
    invoke_end: String,
    invoke_scan: Option<InvokeScan>,
}

struct GuidedState {
    /// Any control markup was stripped this turn. Used only to tell a turn that
    /// produced nothing because it was ALL markup from a model that genuinely said
    /// nothing — the first is worth a log line, the second is not.
    stripped_markup: bool,
    reasoning: ReasoningSpec,
    /// EVERY control marker of the family's tool grammar, from the scanner's own
    /// declaration. One set, used for both lookup and chunk-boundary holdback, in
    /// both the inside-a-thought and outside-a-thought scopes. Assembling it
    /// per-site from openers and orphans is how those four uses drifted apart.
    grammar: GuidedGrammar,
    named_tool: Option<String>,
    invalid_payload: InvalidGuidedPayloadPolicy,
    /// Response starting_state disables reasoning markers, but tool control markers
    /// still need scanning until the JSON value actually starts.
    reasoning_enabled: bool,
    mode: GuidedMode,
    /// Some backends re-emit the reasoning opener even though the prompt already
    /// opened the channel. Consume exactly one such echo instead of leaking it.
    accept_redundant_reasoning_start: bool,
    /// The one guided JSON payload has been emitted. Later bytes are ordinary
    /// visible/reasoning output and control markup, never a second payload.
    payload_emitted: bool,
    /// Visible post-payload text has started. Structural whitespace immediately
    /// after JSON may be discarded, but whitespace after visible text is content.
    post_payload_text_started: bool,
    input: String,
    json: String,
}

/// Earliest control marker in `haystack`, as `(pos, consume_len)`.
///
/// A PREFIX-form marker (`<function=`, which anchors `<function=NAME>`) consumes
/// through its terminating `>`; stripping only the declared prefix left `NAME>`
/// behind to poison the payload. `None` for a prefix form whose `>` has not
/// streamed yet, so the caller holds the bytes back instead of splitting it.
/// `limit` bounds the terminator search: an invoke's terminator has to belong to
/// THAT invoke. Searching the whole buffer let a `</function>` occurring far later
/// — inside a guided argument string, past the end of the thought — be claimed as
/// the terminator, swallowing the reasoning closer and the entire payload with it,
/// so the call was silently dropped and payload fragments were shown as thinking.
/// Where a prefix-form marker's header must end.
///
/// ONE rule, used by both the consume path (`control_marker_at`) and the holdback
/// (`guided_holdback_len`). Two predicates drifted twice: first the `>` scan was
/// unbounded, then it was bounded only by the payload start — which let a stray
/// `<function=` BORROW the `>` from a later `<think>`, so
/// `<function=<think>secret</think>[…]` consumed through the thought opener and
/// emitted the model's private reasoning as visible text (`I3`).
///
/// The boundary is the earliest of: the payload start, and any competing control or
/// reasoning marker. A header that has no `>` before that point is not a complete
/// header — it is literal text the model happened to write.
fn prefix_header_end(haystack: &str, at: usize, limit: Option<usize>, competing: &[&str]) -> usize {
    let mut bound = limit
        .filter(|&end| end > at)
        .unwrap_or(haystack.len())
        .min(haystack.len());
    for m in competing.iter().copied() {
        if let Some(rel) = haystack[at..bound]
            .match_indices(m)
            .map(|(rel, _)| rel)
            .find(|&rel| rel > 0)
        {
            bound = at + rel;
        }
    }
    bound
}

fn control_marker_at(
    haystack: &str,
    markers: &[String],
    invoke_end: &str,
    limit: Option<usize>,
    competing: &[&str],
    flush: bool,
) -> Option<(usize, usize)> {
    let competitors: Vec<&str> = competing
        .iter()
        .copied()
        .chain(markers.iter().map(String::as_str))
        .collect();
    markers
        .iter()
        .filter_map(|m| {
            let at = haystack.find(m.as_str())?;
            control_marker_len_at(haystack, at, m, invoke_end, limit, &competitors, flush)
                .map(|len| (at, len))
        })
        .min_by_key(|(at, _)| *at)
}

/// Length of `marker` when it owns syntax at the exact byte `at`.
/// All guided syntax consumers use this owner so prefix-form completeness and
/// its bounded terminator rule cannot differ between leading, reasoning, and
/// post-payload paths.
fn control_marker_len_at(
    haystack: &str,
    at: usize,
    marker: &str,
    invoke_end: &str,
    limit: Option<usize>,
    competing: &[&str],
    flush: bool,
) -> Option<usize> {
    if !haystack[at..].starts_with(marker) {
        return None;
    }
    if marker.ends_with('=') {
        // BOTH searches stop at `limit`. Bounding only the `</function>`
        // search let the `>` scan run into the payload: for
        // `<function=[{"city": "a>b"}]` it consumed through the `>` INSIDE
        // an argument string and emitted the tail `b"}}]` as text, losing
        // the call; with no `>` anywhere the flush arm consumed the whole
        // buffer and the turn produced nothing at all.
        let bound = prefix_header_end(haystack, at, limit, competing);
        match haystack[at..bound].find('>') {
            // An invoke opener owns its terminator: stripping `<function=NAME>`
            // and leaving `</function>` behind put that fragment in the shown
            // thinking. Consume the pair when the tail is present; a BARE
            // terminator elsewhere stays text, as it is natively.
            Some(rel) => match haystack[at..bound].find(invoke_end) {
                Some(end) => Some(end + invoke_end.len()),
                // A complete prefix header is not necessarily a complete native
                // invoke. Keep it with the streamed body until the terminator
                // arrives; consuming only the header made a split immediately
                // after `>` leak the parameter markup as visible text. A payload
                // or competing marker is a known boundary and EOF cannot gain a
                // terminator, so those cases retain the existing header-only
                // recovery instead of swallowing the guided payload.
                None if !flush && bound == haystack.len() => None,
                None => Some(rel + 1),
            },
            // No `>` before the boundary: this is NOT a header, it is the
            // literal marker text. Strip it alone so the payload behind it
            // still parses (taking everything to EOF here swallowed the call),
            // and strip it NOW when the boundary is already known — a competing
            // marker or the payload is present, so more input cannot put a `>`
            // in front of it. Waiting emitted `<function=` to the user as text
            // and left it inside the thought (`I3`).
            None if flush || bound < haystack.len() => Some(marker.len()),
            None => None,
        }
    } else {
        Some(marker.len())
    }
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
    invoke_end: &str,
    start_label: Option<(&str, &str)>,
    invoke_control: Option<(&str, InvokeScan)>,
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
    // A prefix-form marker counts as COMPLETE only when its `>` arrives before the
    // payload does — the same rule `control_marker_at` uses, and it has to be the
    // same or the two disagree about the identical bytes. It used to accept any `>`
    // ANYWHERE after the marker, which a `>` inside an argument string satisfies:
    // `<function=[{"city": "a > b"}]` was then neither consumed (no `>` before the
    // payload) nor held back (a `>` exists somewhere), so the marker flushed into
    // the payload buffer, the JSON failed to parse, and the user got the call as
    // raw text with `<function=` still attached.
    let payload_at = input.find(['{', '[']);
    let competitors: Vec<&str> = reasoning_markers
        .iter()
        .copied()
        .chain(control.iter().map(String::as_str))
        .collect();
    let pending_prefix_form = control
        .iter()
        .filter(|m| m.ends_with('='))
        .filter_map(|m| input.rfind(m.as_str()).map(|at| (at, m.as_str())))
        .filter(|(at, marker)| {
            // SAME owner as `control_marker_at`: retain both an incomplete header
            // and a complete header whose native invoke terminator has not arrived.
            control_marker_len_at(
                input,
                *at,
                marker,
                invoke_end,
                payload_at,
                &competitors,
                false,
            )
            .is_none()
        })
        .map(|(at, _)| input.len() - at)
        .max()
        .unwrap_or(0);
    let pending_label = start_label
        .and_then(|(start, label)| {
            let at = input.rfind(start)?;
            let rest = &input[at + start.len()..];
            (rest.len() < label.len() && label.starts_with(rest)).then_some(input.len() - at)
        })
        .unwrap_or(0);
    let pending_invoke = invoke_control
        .map(|(start, scan)| {
            let partial = (scan.holdback)(input);
            let body = input
                .match_indices(start)
                .filter_map(|(at, _)| {
                    let suffix = &input[at..];
                    ((scan.opens)(input, at) && (scan.end)(suffix, false).is_none())
                        .then_some(input.len() - at)
                })
                .max()
                .unwrap_or(0);
            partial.max(body)
        })
        .unwrap_or(0);
    split
        .max(pending_prefix_form)
        .max(pending_label)
        .max(pending_invoke)
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

/// First control-syntax byte outside a JSON string after a payload has opened.
/// Marker-looking text inside a string remains payload data, including when the
/// JSON is malformed or truncated. Syntax outside a string is returned to the
/// normal channel scanner instead of being trimmed by a tail-only implementation.
fn guided_payload_syntax_boundary(
    input: &str,
    reasoning: ReasoningSpec,
    control_markers: &[String],
    invoke_end: &str,
) -> Option<usize> {
    let start = input.len() - input.trim_start().len();
    if !matches!(input.as_bytes().get(start), Some(b'{') | Some(b'[')) {
        return None;
    }

    let mut in_string = false;
    let mut escaped = false;
    let competitors: Vec<&str> = [reasoning.start, reasoning.end]
        .into_iter()
        .chain(control_markers.iter().map(String::as_str))
        .collect();
    for (relative, ch) in input[start..].char_indices() {
        let at = start + relative;
        if in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }
        if ch == '"' {
            in_string = true;
            continue;
        }
        if at == start {
            continue;
        }
        if input[at..].starts_with(reasoning.start)
            || input[at..].starts_with(reasoning.end)
            || input[at..].starts_with(invoke_end)
            || control_markers.iter().any(|marker| {
                control_marker_len_at(input, at, marker, invoke_end, None, &competitors, true)
                    .is_some()
            })
        {
            return Some(at);
        }
    }
    None
}

impl GuidedState {
    fn new(
        reasoning: ReasoningSpec,
        grammar: GuidedGrammar,
        named_tool: Option<String>,
        starting_state: UnifiedParserStartingState,
        invalid_payload: InvalidGuidedPayloadPolicy,
    ) -> Self {
        Self {
            stripped_markup: false,
            reasoning,
            grammar,
            named_tool,
            invalid_payload,
            reasoning_enabled: starting_state != UnifiedParserStartingState::Response,
            mode: Self::mode_for(starting_state),
            accept_redundant_reasoning_start: starting_state
                == UnifiedParserStartingState::Reasoning,
            payload_emitted: false,
            post_payload_text_started: false,
            input: String::new(),
            json: String::new(),
        }
    }

    fn mode_for(starting_state: UnifiedParserStartingState) -> GuidedMode {
        match starting_state {
            UnifiedParserStartingState::None => GuidedMode::OutsideReasoning,
            UnifiedParserStartingState::Reasoning => GuidedMode::Reasoning,
            UnifiedParserStartingState::Response => GuidedMode::OutsideReasoning,
        }
    }

    fn push_into(&mut self, chunk: &str, output: &mut UnifiedParserOutput) -> Result<()> {
        if self.mode == GuidedMode::VisibleOnly {
            self.json.push_str(chunk);
        } else {
            self.input.push_str(chunk);
        }
        for event in self.drain(false) {
            output.push_event(event);
        }
        for event in self.emit_completed_json()? {
            output.push_event(event);
        }
        if self.payload_emitted {
            for event in self.drain(false) {
                output.push_event(event);
            }
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<Vec<UnifiedParserEvent>> {
        let mut output = self.drain(true);
        output.extend(self.emit_completed_json()?);
        if self.payload_emitted {
            output.extend(self.drain(true));
        } else {
            if let Some(boundary) = guided_payload_syntax_boundary(
                &self.json,
                self.reasoning,
                &self.grammar.control_markers,
                &self.grammar.invoke_end,
            ) {
                let tail = self.json.split_off(boundary);
                let payload_end = self.json.trim_end().len();
                self.json.truncate(payload_end);
                self.input.push_str(&tail);
            }
            output.extend(self.finish_json()?);
            self.payload_emitted = true;
            self.mode = GuidedMode::OutsideReasoning;
            output.extend(self.drain(true));
        }
        Ok(output)
    }

    fn reset(&mut self, starting_state: UnifiedParserStartingState) -> String {
        let mut recovered = std::mem::take(&mut self.json);
        recovered.push_str(&std::mem::take(&mut self.input));
        // Buffers alone are not the state. Leaving `mode` at VisibleOnly would make
        // the NEXT stream treat its reasoning as JSON payload and surface it as text,
        // so put the channel back where `new` would have started it.
        self.mode = Self::mode_for(starting_state);
        self.reasoning_enabled = starting_state != UnifiedParserStartingState::Response;
        self.accept_redundant_reasoning_start =
            starting_state == UnifiedParserStartingState::Reasoning;
        self.stripped_markup = false;
        self.payload_emitted = false;
        self.post_payload_text_started = false;
        recovered
    }

    /// Emit a complete JSON value as soon as the closing byte arrives. The
    /// stream decoder gives us the exact end of the first value, so bytes after
    /// it go back through the channel/control scanner rather than being mistaken
    /// for JSON or stripped by a separate tail-only marker implementation.
    fn emit_completed_json(&mut self) -> Result<Vec<UnifiedParserEvent>> {
        if self.payload_emitted || !json_payload_started(&self.json) {
            return Ok(Vec::new());
        }

        let leading = self.json.len() - self.json.trim_start().len();
        let mut values = serde_json::Deserializer::from_str(&self.json[leading..])
            .into_iter::<serde_json::Value>();
        let Some(Ok(_)) = values.next() else {
            return Ok(Vec::new());
        };
        let value_end = leading + values.byte_offset();
        drop(values);
        let whitespace_end = value_end
            + self.json[value_end..]
                .find(|ch: char| !ch.is_whitespace())
                .unwrap_or(self.json.len() - value_end);
        let syntax_at = guided_payload_syntax_boundary(
            &self.json,
            self.reasoning,
            &self.grammar.control_markers,
            &self.grammar.invoke_end,
        );
        let (payload_end, tail_start) = if syntax_at == Some(whitespace_end) {
            // Whitespace separating the value from control syntax belongs to
            // neither output; the old tail recovery trimmed it with the marker.
            (value_end, whitespace_end)
        } else {
            (value_end, value_end)
        };
        let uncommitted_suffix = self.json[payload_end..].to_string();
        let tail = self.json[tail_start..].to_string();
        self.json.truncate(payload_end);
        let mut output = match self.finish_json() {
            Ok(output) => output,
            Err(error) => {
                self.json.push_str(&uncommitted_suffix);
                return Err(error);
            }
        };
        self.payload_emitted = true;
        self.mode = GuidedMode::OutsideReasoning;
        self.input.push_str(&tail);
        Ok(std::mem::take(&mut output))
    }

    fn opener_len_at(&self, at: usize, flush: bool) -> Option<usize> {
        reasoning_opener_len(
            self.reasoning.start,
            self.reasoning.start_label,
            &self.input[at + self.reasoning.start.len()..],
            flush,
        )
    }

    fn start_label(&self) -> Option<(&str, &str)> {
        self.reasoning
            .start_label
            .map(|label| (self.reasoning.start, label))
    }

    fn invoke_control(&self) -> Option<(&str, InvokeScan)> {
        self.grammar
            .invoke_scan
            .map(|scan| (self.grammar.invoke_start.as_str(), scan))
    }

    fn control_marker_at(
        &self,
        haystack: &str,
        limit: Option<usize>,
        competing: &[&str],
        flush: bool,
    ) -> Option<(usize, usize)> {
        let regular = control_marker_at(
            haystack,
            &self.grammar.control_markers,
            &self.grammar.invoke_end,
            limit,
            competing,
            flush,
        );
        let Some(scan) = self.grammar.invoke_scan else {
            return regular;
        };

        let mut cursor = 0;
        while let Some(relative) = haystack[cursor..].find(&self.grammar.invoke_start) {
            let at = cursor + relative;
            let suffix = &haystack[at..];
            if (scan.opens)(haystack, at) {
                if let Some(len) = (scan.end)(suffix, flush) {
                    return regular
                        .filter(|(regular_at, _)| *regular_at < at)
                        .or(Some((at, len)));
                }
                return regular.filter(|(regular_at, _)| *regular_at < at);
            }
            if !flush && (scan.holdback)(suffix) == suffix.len() {
                return regular.filter(|(regular_at, _)| *regular_at < at);
            }
            cursor = at + self.grammar.invoke_start.len();
        }
        regular
    }

    /// Strip the reasoning markers wrapping the JSON payload. Once visible
    /// output starts every later byte is JSON data, so native-looking strings
    /// inside argument values stay literal.
    fn drain(&mut self, flush: bool) -> Vec<UnifiedParserEvent> {
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
                    if !self.payload_emitted
                        && (json_payload_started(&self.json)
                            || (self.json.trim().is_empty() && json_payload_started(&self.input)))
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
                    // call never emitted.
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
                    let stray_close = self
                        .control_marker_at(
                            &self.input,
                            // Nor past the start of the payload itself.
                            self.input.find(['{', '[']),
                            // A thought marker ahead also ends the header: a stray
                            // `<function=` must not borrow the `>` from `<think>` and
                            // swallow the thought — that put private reasoning in the
                            // user's answer.
                            &[start, end],
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

                    if !closer_first
                        && let Some(at) = open_at
                        && let Some(open_len) = self.opener_len_at(at, flush)
                    {
                        // Whatever was buffered as "payload so far", plus this prefix,
                        // was visible text after all — a thought is opening behind it.
                        let mut pending = std::mem::take(&mut self.json);
                        pending.push_str(&self.input[..at]);
                        if pending.trim().is_empty() {
                            if !self.payload_emitted {
                                self.json = pending;
                            }
                        } else {
                            push_run(&mut output, Kind::Text, &pending);
                            if self.payload_emitted {
                                self.post_payload_text_started = true;
                            }
                        }
                        self.input.drain(..at + open_len);
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
                            if !self.payload_emitted {
                                self.json = pending;
                            }
                        } else {
                            push_run(&mut output, Kind::Text, &pending);
                            if self.payload_emitted {
                                self.post_payload_text_started = true;
                            }
                        }
                        self.stripped_markup = true;
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
                            &self.grammar.control_markers,
                            &self.grammar.invoke_end,
                            self.start_label(),
                            self.invoke_control(),
                            flush,
                        )
                    };
                    let visible_len = self.input.len().saturating_sub(keep);
                    if visible_len > 0 {
                        if self.payload_emitted
                            && !self.post_payload_text_started
                            && self.input[..visible_len].trim().is_empty()
                        {
                            if !flush {
                                break;
                            }
                            if self.named_tool.is_some() {
                                output.push(UnifiedParserEvent::ToolCall(ToolCallDelta {
                                    tool_index: 0,
                                    name: None,
                                    arguments: self.input[..visible_len].to_string(),
                                }));
                            }
                        } else if self.payload_emitted {
                            push_run(&mut output, Kind::Text, &self.input[..visible_len]);
                            self.post_payload_text_started = true;
                        } else {
                            self.json.push_str(&self.input[..visible_len]);
                        }
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
                        if self.payload_emitted {
                            push_run(&mut output, Kind::Text, &self.input);
                            self.post_payload_text_started = true;
                        } else {
                            self.json.push_str(&self.input);
                        }
                        self.input.clear();
                    }
                    break;
                }
                GuidedMode::Reasoning => {
                    if self.accept_redundant_reasoning_start {
                        let non_whitespace = self.input.trim_start();
                        let leading = self.input.len() - non_whitespace.len();
                        if non_whitespace.starts_with(start)
                            && let Some(open_len) = self.opener_len_at(leading, flush)
                        {
                            push_run(&mut output, Kind::Reasoning, &self.input[..leading]);
                            self.input.drain(..leading + open_len);
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
                    // (the native path's `stray_in_reasoning`, which this deliberately
                    // does NOT share — see its doc). Guided decoding constrains the TOOL
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
                    let stray = self
                        .control_marker_at(
                            &self.input,
                            // A narrated invoke lives INSIDE this thought, so its
                            // terminator cannot be past the span's closer.
                            self.input.find(end),
                            &[end],
                            flush,
                        )
                        .into_iter()
                        .chain(
                            self.input
                                .find(start)
                                .and_then(|at| Some((at, self.opener_len_at(at, flush)?))),
                        )
                        .min_by_key(|(at, _)| *at)
                        .map(|(at, len)| (at, len, false));
                    if let Some((at, consume, closes)) = [close, stray]
                        .into_iter()
                        .flatten()
                        .min_by_key(|(at, _, _)| *at)
                    {
                        push_run(&mut output, Kind::Reasoning, &self.input[..at]);
                        self.stripped_markup = true;
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
                            &self.grammar.control_markers,
                            &self.grammar.invoke_end,
                            self.start_label(),
                            self.invoke_control(),
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
    fn finish_json(&mut self) -> Result<Vec<UnifiedParserEvent>> {
        let payload = self.json.trim();
        if payload.is_empty() {
            if self.invalid_payload == InvalidGuidedPayloadPolicy::Reject {
                return Err(InvalidGuidedPayload {
                    kind: InvalidGuidedPayloadKind::Missing,
                    choice: if self.named_tool.is_some() {
                        "named"
                    } else {
                        "required"
                    },
                }
                .into());
            }
            // The turn produced ONLY control markup — everything was stripped and
            // there is nothing left to parse. Emitting no events is right (markup is
            // not an answer), but doing it silently is not: the caller cannot tell
            // this from a model that legitimately said nothing, and the usual cause
            // is a backend configured for guided decoding against a model still
            // emitting its native call grammar. P2 is best-effort recovery, NOT
            // silent loss, and the sibling not-a-tool-call path is instrumented for
            // exactly this reason — so this one is too.
            if self.stripped_markup {
                tracing::warn!(
                    why = "unified_guided_output_was_only_markup",
                    choice = if self.named_tool.is_some() {
                        "named"
                    } else {
                        "required"
                    },
                    named_tool = self.named_tool.as_deref().unwrap_or("-"),
                    stripped_markup = true,
                    "guided output contained no JSON payload, only control markup; \
                     emitting nothing (is the backend's guided decoding actually on?)"
                );
            }
            self.json.clear();
            return Ok(Vec::new());
        }

        let raw_payload = self.json.clone();
        let calls = match &self.named_tool {
            // A named choice constrains output to that tool's ARGUMENTS alone,
            // so the payload is the argument object and the name is known.
            // Arguments are an OBJECT. A bare string / number / null / array is
            // syntactically valid JSON but is not an argument set, and EMITTING it
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
                vec![(name.clone(), raw_payload.clone())]
            }),
            None => parse_required_guided_calls(payload),
        };

        let Some(calls) = calls.filter(|calls| !calls.is_empty()) else {
            if self.invalid_payload == InvalidGuidedPayloadPolicy::Reject {
                return Err(InvalidGuidedPayload {
                    kind: if serde_json::from_str::<serde_json::Value>(payload).is_ok() {
                        InvalidGuidedPayloadKind::WrongShape
                    } else {
                        InvalidGuidedPayloadKind::InvalidJson
                    },
                    choice: if self.named_tool.is_some() {
                        "named"
                    } else {
                        "required"
                    },
                }
                .into());
            }
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
            // The SAME bytes the parse was given, not the raw buffer. Trailing
            // control markup was stripped above precisely because it is markup, and
            // handing it back here would put `</tool_call>` in the user's visible
            // answer — a marker leak (`I3`) on the one path that exists to recover
            // gracefully. `raw_payload` is already the tail-trimmed value, and it
            // stays byte-identical to the buffer when nothing was stripped (`I7`).
            self.json.clear();
            return Ok(vec![UnifiedParserEvent::Text(raw_payload)]);
        };

        self.json.clear();
        Ok(calls
            .into_iter()
            .enumerate()
            .map(|(tool_index, (name, arguments))| {
                UnifiedParserEvent::ToolCall(ToolCallDelta {
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
            // a number cannot bind to the tool's parameters, so emitting it would
            // hand the tool a shape it cannot use. Absent is different — that means
            // no arguments, and stays valid (see the note above).
            (Some(raw), None) | (None, Some(raw)) => {
                serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(raw.get())
                    .ok()?;
                raw.get().to_string()
            }
            (None, None) => "{}".to_string(),
            // The aliases are alternatives, not two independently meaningful
            // argument sets. Choosing one silently can emit different bytes from
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

/// How a vendor supplies a parser: given the request's tools, build one parser for
/// one stream.
///
/// A plain `fn` pointer, not a boxed closure, so registering is `const`-friendly and
/// a factory cannot capture per-request state by accident — the per-stream state
/// belongs in the parser the factory returns (`I4`).
pub type UnifiedParserFactory = fn(&[Tool]) -> Result<Box<dyn UnifiedParser>>;

/// Vendor-supplied families, consulted BEFORE the built-in table.
///
/// Checking this first is what makes "implement your own version of a family we
/// already ship" work: registering `qwen3` shadows the built-in `qwen3` for the
/// whole process, and unregistering restores it. An add-only registry would force a
/// vendor who disagrees with one of our families to fork the crate.
static VENDOR_PARSERS: std::sync::LazyLock<
    std::sync::RwLock<std::collections::HashMap<String, UnifiedParserFactory>>,
> = std::sync::LazyLock::new(|| std::sync::RwLock::new(std::collections::HashMap::new()));

/// Register `factory` for `family`, returning whatever it displaced.
///
/// Returns `Some(previous)` if this replaced an earlier VENDOR registration, and
/// `None` otherwise — including when it shadows a built-in, since the built-in is
/// still there and returns as soon as this registration is removed. Callers that
/// care whether they are shadowing should ask
/// [`builtin_unified_families`] first.
///
/// # Startup-only
///
/// Register during startup, BEFORE serving. Every access is guarded by one `RwLock`,
/// and a create linearizes at the moment it reads the table — so the outcome is always
/// SOME well-defined selection, never undefined behaviour or a torn read. What is not
/// guaranteed is ORDERING against an overlapping mutation: the lookup copies the factory
/// and releases the lock before calling it, so a create that read the table first can
/// finish building after a concurrent `unregister` returns. Registering at startup
/// avoids having to reason about that window at all. A parser already constructed keeps
/// what it was built with, so a request in progress never changes implementation
/// mid-stream.
pub fn register_unified_parser(
    family: &str,
    factory: UnifiedParserFactory,
) -> Option<UnifiedParserFactory> {
    // Register under the CANONICAL name. A built-in family can be reached by more
    // than one routing name (`qwen3` and `qwen3_coder` are one grammar), and keying
    // on the caller's spelling shadowed only the spelling they happened to use:
    // `register_unified_parser("qwen3", ..)` left `qwen3_coder` on the built-in, so
    // the same family silently ran two different parsers depending on how the
    // request was routed. Canonicalizing on both sides is what makes "replace a
    // family this crate ships" true for every name that family answers to.
    let key = canonical_unified_family(family).unwrap_or(family);
    let previous = VENDOR_PARSERS
        .write()
        .unwrap_or_else(|e| e.into_inner())
        .insert(key.to_string(), factory);
    tracing::info!(
        target: "dynamo_parsers_v2",
        family = key,
        requested = family,
        shadows_builtin = canonical_unified_family(family).is_some(),
        replaced_vendor = previous.is_some(),
        "unified parser registered"
    );
    previous
}

/// Remove a vendor registration, returning it. A shadowed built-in becomes
/// reachable again.
///
/// Accepts any alias of the family, matching [`register_unified_parser`], and
/// inherits its STARTUP-ONLY guidance: a create that read the table before this call
/// can still finish building the parser it removes, so the returned factory may be used
/// once more after this returns.
pub fn unregister_unified_parser(family: &str) -> Option<UnifiedParserFactory> {
    let key = canonical_unified_family(family).unwrap_or(family);
    VENDOR_PARSERS
        .write()
        .unwrap_or_else(|e| e.into_inner())
        .remove(key)
}

/// Families currently registered by a vendor, sorted.
pub fn vendor_unified_families() -> Vec<String> {
    let mut v: Vec<String> = VENDOR_PARSERS
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .keys()
        .cloned()
        .collect();
    v.sort();
    v
}

/// Look up a vendor factory without constructing anything.
///
/// Canonicalizes first, so every alias of a built-in family resolves to the same
/// vendor registration.
fn vendor_factory(family: &str) -> Option<UnifiedParserFactory> {
    let key = canonical_unified_family(family).unwrap_or(family);
    VENDOR_PARSERS
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .get(key)
        .copied()
}

/// THE built-in registry. One line per family — adding a family is adding a line
/// here and nothing else in this crate.
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

        /// Every family built INTO this crate, aliases included.
        ///
        /// Deliberately excludes vendor registrations: the conformance suite
        /// iterates this, and a vendor parser has no corpus here to be measured
        /// against. Ask [`vendor_unified_families`] for those.
        pub fn builtin_unified_families() -> &'static [&'static str] {
            REGISTERED_UNIFIED_FAMILIES
        }

        /// The canonical name of a built-in family, given any of its aliases.
        ///
        /// `None` for a name this crate does not ship, which is how a vendor family
        /// keeps its own spelling. Generated from the same list as the constructor,
        /// so an alias cannot exist for dispatch but be invisible to the vendor
        /// registry — that split is exactly what made `register_unified_parser`
        /// shadow one routing name and not its sibling.
        pub fn canonical_unified_family(family: &str) -> Option<&'static str> {
            match family {
                $($family $(| $alias)* => Some($family),)+
                _ => None,
            }
        }

        /// Create the unified parser for a family.
        ///
        /// A vendor registration wins over the built-in of the same name — see
        /// [`register_unified_parser`]. Both branches return the parser directly:
        /// there is no unified debug wrapper (the only `DebugToolParser` wraps the
        /// separate tool-only trait). Selection is observable through the
        /// `tracing::debug!` event emitted here, and it reports vendor and built-in
        /// on the same terms.
        pub fn create_unified_parser_for_family(
            family: &str,
            tools: &[Tool],
        ) -> Result<Box<dyn UnifiedParser>> {
            if let Some(factory) = vendor_factory(family) {
                let key = canonical_unified_family(family).unwrap_or(family);
                let parser = factory(tools)?;
                tracing::debug!(
                    target: "dynamo_parsers_v2",
                    family = key,
                    requested = family,
                    source = "vendor",
                    "v2 UNIFIED parser active"
                );
                if crate::tool_calling::debug::debug_enabled() {
                    // Owned, NOT leaked. Construction is per REQUEST, so `Box::leak`
                    // here grew the process without bound for as long as debug mode
                    // stayed on -- the diagnostic aid became the fault. `DebugToolParser`
                    // already stores an owned `String`; this now matches it.
                    return Ok(DebugUnifiedParser::wrap(key, parser));
                }
                return Ok(parser);
            }

            let parser = match family {
                $($family $(| $alias)* => $ctor(tools),)+
                other => anyhow::bail!(
                    "no unified parser for family '{other}'. Built-in: {:?}. \
                     Vendor-registered: {:?}. To supply your own, call \
                     dynamo_parsers_v2::register_unified_parser(\"{other}\", your_factory) \
                     before serving.",
                    REGISTERED_UNIFIED_FAMILIES,
                    vendor_unified_families(),
                ),
            };
            // Same helper the vendor branch above uses. A second generated match cost an
            // arm per family plus an `unreachable!` that existed only because the compiler
            // cannot see that the `bail!` above already returned.
            let canonical = canonical_unified_family(family).unwrap_or(family);

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
    family: String,
    inner: Box<dyn UnifiedParser>,
}

impl DebugUnifiedParser {
    fn wrap(family: impl Into<String>, inner: Box<dyn UnifiedParser>) -> Box<dyn UnifiedParser> {
        let family = family.into();
        crate::tool_calling::debug::emit(format_args!("UNIFIED family={family} created"));
        Box::new(Self { family, inner })
    }

    fn log(&self, method: &str, deltas: &[UnifiedParserEvent]) {
        if deltas.is_empty() {
            return;
        }
        let calls = deltas
            .iter()
            .filter_map(|d| match d {
                UnifiedParserEvent::ToolCall(c) => Some(c.name.as_deref().unwrap_or("…")),
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
    fn initialize_request(&mut self, init: UnifiedParserInit) -> Result<()> {
        let mode = match &init.tool_output_mode {
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
            "UNIFIED family={} initialize prompt_token_ids_len={} starting_state={:?} tool_output_mode={} invalid_guided_payload={:?}",
            self.family,
            init.prompt_token_ids.len(),
            init.starting_state,
            mode,
            init.invalid_guided_payload,
        ));
        self.inner.initialize_request(init)
    }

    /// Logs only what THIS advance committed, so the debug trace still reads one
    /// line per advance even though the caller's buffer may already hold earlier
    /// events.
    fn parse_into(&mut self, delta: &str, output: &mut UnifiedParserOutput) -> Result<()> {
        let mut mine = UnifiedParserOutput::default();
        let r = self.inner.parse_into(delta, &mut mine);
        self.log("push", &mine.events);
        output.append(&mut mine);
        r
    }

    fn finish(&mut self) -> Result<UnifiedParserOutput> {
        let out = self.inner.finish()?;
        self.log("finish", &out.events);
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
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(tool_index: usize, name: Option<&str>, arguments: &str) -> UnifiedParserEvent {
        UnifiedParserEvent::ToolCall(ToolCallDelta {
            tool_index,
            name: name.map(str::to_string),
            arguments: arguments.to_string(),
        })
    }

    #[test]
    fn guided_payload_boundary_accepts_multibyte_trailing_text() {
        let reasoning = ReasoningSpec {
            start: "<think>",
            end: "</think>",
            forced_start: false,
            start_label: None,
            preserve_special_tokens: false,
        };

        assert_eq!(
            guided_payload_syntax_boundary(
                "[{\"name\":\"f\",\"arguments\":{}}]é",
                reasoning,
                &[],
                "</function>",
            ),
            None
        );
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
    fn guided_reset_restores_all_request_scoped_flags() {
        let mut guided = GuidedState::new(
            ReasoningSpec {
                start: "<think>",
                end: "</think>",
                forced_start: false,
                start_label: None,
                preserve_special_tokens: false,
            },
            GuidedGrammar {
                control_markers: vec!["<tool_call>".into(), "</tool_call>".into()],
                invoke_start: "<function=".into(),
                invoke_end: "</function>".into(),
                invoke_scan: None,
            },
            None,
            UnifiedParserStartingState::None,
            InvalidGuidedPayloadPolicy::RecoverAsText,
        );
        guided
            .push_into("</tool_call>", &mut UnifiedParserOutput::default())
            .unwrap();
        assert!(guided.stripped_markup, "fixture did not mutate the flag");
        guided.payload_emitted = true;

        guided.reset(UnifiedParserStartingState::None);

        assert!(!guided.stripped_markup);
        assert!(!guided.payload_emitted);
        assert_eq!(guided.mode, GuidedMode::OutsideReasoning);
        assert!(guided.input.is_empty());
        assert!(guided.json.is_empty());
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
        pushed.extend(a.finish().unwrap().events);

        let mut appended = UnifiedParserOutput::default();
        let mut b = create_unified_parser_for_family("qwen3", &[]).unwrap();
        for c in chunks {
            b.parse_into(c, &mut appended).unwrap();
        }
        appended.append(&mut b.finish().unwrap());

        assert_eq!(
            pushed, appended.events,
            "parse_into must commit exactly what push returns, in the same order"
        );
        assert_eq!(assemble(&pushed), appended.assembled());
        assert!(
            pushed
                .iter()
                .any(|d| matches!(d, UnifiedParserEvent::ToolCall(_))),
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
            UnifiedParserEvent::Text("ok".into()),
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
            UnifiedParserEvent::Reasoning("think".into()),
            UnifiedParserEvent::Reasoning("ing".into()),
            UnifiedParserEvent::Text("he".into()),
            UnifiedParserEvent::Text("llo".into()),
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
            UnifiedParserEvent::Reasoning("a".into()),
            call(0, Some("f"), r#"{"x":"1"}"#),
            UnifiedParserEvent::Reasoning("b".into()),
        ]);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0], UnifiedEvent::Reasoning { text: "a".into() });
        assert_eq!(out[2], UnifiedEvent::Reasoning { text: "b".into() });
    }

    #[test]
    fn assemble_joins_argument_fragments_at_the_first_position() {
        let out = assemble(&[
            call(0, Some("f"), r#"{"x":"#),
            UnifiedParserEvent::Text("mid".into()),
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
            UnifiedParserEvent::Reasoning("a".into()),
            call(0, Some("f"), "{}"),
            UnifiedParserEvent::Text("b".into()),
        ]);
        assert_eq!(result.normal_text, "ab");
        assert_eq!(result.calls.len(), 1);
    }

    #[test]
    fn debug_wrapper_preserves_capabilities_and_guided_deltas() {
        let mut plain = qwen3::qwen3_unified(&[]);
        let mut debug = DebugUnifiedParser::wrap("qwen3", qwen3::qwen3_unified(&[]));

        assert_eq!(
            plain.preserve_special_tokens(),
            debug.preserve_special_tokens()
        );
        assert_eq!(plain.tool_call_id(0), debug.tool_call_id(0));

        for parser in [&mut plain, &mut debug] {
            parser
                .initialize_request(UnifiedParserInit {
                    tool_output_mode: UnifiedToolOutputMode::GuidedJson { named_tool: None },
                    invalid_guided_payload: InvalidGuidedPayloadPolicy::RecoverAsText,
                    ..UnifiedParserInit::default()
                })
                .unwrap();
        }
        for chunk in [
            "<think>checking</think>",
            r#"[{"name":"get_weather","arguments":{"city":"Paris"}}]"#,
        ] {
            assert_eq!(plain.push(chunk).unwrap(), debug.push(chunk).unwrap());
        }
        assert_eq!(
            plain.finish().unwrap().events,
            debug.finish().unwrap().events
        );
        assert_eq!(plain.reset(), debug.reset());
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
mod append_seam_tests {
    use super::*;

    /// The peer's regression: joining two buffers must yield the same events as
    /// accumulating straight through, so the same bytes cannot describe a different
    /// event stream depending on how the caller batched them.
    /// The peer's exact seam regression: a multi-buffer join must produce the same
    /// events as one buffer accumulated straight through, across two appends and both
    /// text and reasoning runs.
    #[test]
    fn append_coalesces_adjacent_same_kind_events_across_the_seam() {
        let mut acc = UnifiedParserOutput::default();
        acc.push_text("hello");

        let mut second = UnifiedParserOutput::default();
        second.push_text(" world");
        second.push_reasoning("think");
        acc.append(&mut second);

        let mut third = UnifiedParserOutput::default();
        third.push_reasoning("ing");
        third.push_text("!");
        acc.append(&mut third);

        assert_eq!(
            acc.events,
            vec![
                UnifiedParserEvent::Text("hello world".to_string()),
                UnifiedParserEvent::Reasoning("thinking".to_string()),
                UnifiedParserEvent::Text("!".to_string()),
            ],
            "same-kind runs must merge across every seam, different kinds must not"
        );
        assert!(
            second.events.is_empty(),
            "append must consume the source events"
        );
        assert!(
            third.events.is_empty(),
            "append must consume the source events"
        );
    }

    /// Different kinds must NOT merge, and calls never merge with anything.
    #[test]
    fn append_keeps_distinct_kinds_separate() {
        let mut a = UnifiedParserOutput::default();
        a.push_text("visible");
        let mut b = UnifiedParserOutput::default();
        b.push_reasoning("thought");
        a.append(&mut b);

        assert_eq!(
            a.events.len(),
            2,
            "text and reasoning must stay distinct: {:?}",
            a.events
        );
    }
}

/// The recovery contract through the PUBLIC surface a caller actually uses:
/// `UnifiedParser::parse_into` with a caller-owned `UnifiedParserOutput`.
///
/// The scanner-level tests in `scan::recovery_tests` prove drain-after-success. They do
/// NOT prove this: they drive a `Vec` sink directly, so reverting `parse_into` to collect
/// into a local vector and copy at the end would leave them green while the caller's
/// committed events were silently dropped by `?`. These tests are that missing control.
#[cfg(test)]
mod parse_into_recovery_tests {
    use super::*;
    use crate::tool_calling::scan::test_support::{FailOnBoom, failing_scanner};

    fn parser() -> ScannerUnified<FailOnBoom> {
        ScannerUnified::new(failing_scanner())
    }

    fn call(index: usize) -> UnifiedParserEvent {
        UnifiedParserEvent::ToolCall(ToolCallDelta {
            tool_index: index,
            name: Some("ok".to_string()),
            arguments: "{}".to_string(),
        })
    }

    /// A failure on the FIRST wrapped invoke: text committed before it survives.
    #[test]
    fn first_wrapped_failure_keeps_committed_text_and_recovers_the_invoke() {
        let mut p = parser();
        let mut out = UnifiedParserOutput::default();
        let r = p.parse_into(
            "prefix<tool_call><function=boom></function></tool_call>suffix",
            &mut out,
        );
        assert!(r.is_err(), "the injected emitter must surface its error");
        assert_eq!(
            out.events,
            vec![UnifiedParserEvent::Text("prefix".to_string())],
            "text committed before the failure belongs to the caller"
        );
        assert_eq!(
            p.reset(),
            "<tool_call><function=boom></function></tool_call>suffix"
        );
    }

    /// A failure on a LATER wrapped invoke: the call that already succeeded survives too.
    /// This is the case that a local-vector implementation loses entirely.
    #[test]
    fn later_wrapped_failure_keeps_the_call_that_already_succeeded() {
        let mut p = parser();
        let mut out = UnifiedParserOutput::default();
        let r = p.parse_into(
            "prefix<tool_call><function=ok></function><function=boom></function></tool_call>suffix",
            &mut out,
        );
        assert!(r.is_err(), "the injected emitter must surface its error");
        assert_eq!(
            out.events,
            vec![UnifiedParserEvent::Text("prefix".to_string()), call(0)],
            "an event already committed cannot be retracted by a later error"
        );
        assert_eq!(p.reset(), "<function=boom></function></tool_call>suffix");
    }

    #[test]
    fn first_bare_failure_keeps_committed_text_and_recovers_the_invoke() {
        let mut p = parser();
        let mut out = UnifiedParserOutput::default();
        let r = p.parse_into("prefix<function=boom></function>suffix", &mut out);
        assert!(r.is_err(), "the injected emitter must surface its error");
        assert_eq!(
            out.events,
            vec![UnifiedParserEvent::Text("prefix".to_string())],
            "text committed before the failure belongs to the caller"
        );
        assert_eq!(p.reset(), "<function=boom></function>suffix");
    }

    #[test]
    fn later_bare_failure_keeps_the_call_that_already_succeeded() {
        let mut p = parser();
        let mut out = UnifiedParserOutput::default();
        let r = p.parse_into(
            "prefix<function=ok></function><function=boom></function>suffix",
            &mut out,
        );
        assert!(r.is_err(), "the injected emitter must surface its error");
        assert_eq!(
            out.events,
            vec![UnifiedParserEvent::Text("prefix".to_string()), call(0)],
            "an event already committed cannot be retracted by a later error"
        );
        assert_eq!(p.reset(), "<function=boom></function>suffix");
    }
}

/// Every way of BUILDING this type must agree, not just the push helpers.
///
/// `append` was fixed to coalesce and `FromIterator` was not, so `collect()` still
/// produced a different event stream than pushing the same bytes. These pin every
/// constructor to the one merge rule so the next one cannot drift alone.
#[cfg(test)]
mod construction_parity_tests {
    use super::*;

    fn adjacent() -> Vec<UnifiedParserEvent> {
        vec![
            UnifiedParserEvent::Text("hel".to_string()),
            UnifiedParserEvent::Text("lo".to_string()),
            UnifiedParserEvent::Reasoning("thin".to_string()),
            UnifiedParserEvent::Reasoning("king".to_string()),
        ]
    }

    fn pushed() -> UnifiedParserOutput {
        let mut out = UnifiedParserOutput::default();
        for e in adjacent() {
            match e {
                UnifiedParserEvent::Text(t) => out.push_text(t),
                UnifiedParserEvent::Reasoning(t) => out.push_reasoning(t),
                UnifiedParserEvent::ToolCall(c) => out.push_call(c),
            }
        }
        out
    }

    #[test]
    fn collect_agrees_with_pushing_the_same_events() {
        let collected: UnifiedParserOutput = adjacent().into_iter().collect();
        assert_eq!(
            collected,
            pushed(),
            "`collect()` must apply the same merge rule as the push helpers"
        );
        assert_eq!(
            collected.events,
            vec![
                UnifiedParserEvent::Text("hello".to_string()),
                UnifiedParserEvent::Reasoning("thinking".to_string()),
            ],
            "adjacent same-kind events must merge when collected"
        );
    }

    #[test]
    fn append_agrees_with_pushing_the_same_events() {
        let mut joined = UnifiedParserOutput::default();
        let mut src: UnifiedParserOutput = adjacent().into_iter().collect();
        joined.append(&mut src);
        assert_eq!(
            joined,
            pushed(),
            "`append` must apply the same merge rule as the push helpers"
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

/// Initialization must REJECT what it cannot represent, and a rejection must leave
/// the parser untouched.
///
/// These live here, not in an integration test, because they need a scanner built
/// WITHOUT `.with_reasoning(..)` — a shape no registered family has today, so it is
/// unreachable through the public registry. An earlier version of this test used
/// the built-in `qwen3`, which always installs a reasoning spec: it asserted that
/// `Reasoning` SUCCEEDS and so could never have caught the defect it was named for.
#[cfg(test)]
mod initialize_preflight_tests {
    use super::*;

    fn init(
        starting_state: UnifiedParserStartingState,
        tool_output_mode: UnifiedToolOutputMode,
    ) -> UnifiedParserInit {
        UnifiedParserInit {
            starting_state,
            tool_output_mode,
            invalid_guided_payload: InvalidGuidedPayloadPolicy::RecoverAsText,
            ..UnifiedParserInit::default()
        }
    }

    /// The REAL qwen3 scanner, built WITHOUT `.with_reasoning(..)`.
    ///
    /// Same plumbing every family uses; the single difference is the missing
    /// reasoning channel, which is precisely the condition under test.
    fn reasoningless()
    -> ScannerUnified<impl crate::tool_calling::scan::InvokeEmitter + Send + 'static> {
        ScannerUnified::new(crate::tool_calling::qwen3_coder::qwen3_scanner(&[]))
    }

    #[test]
    fn explicit_reasoning_is_rejected_when_the_family_has_no_reasoning_channel() {
        let mut p = reasoningless();
        let err = p
            .initialize_request(init(
                UnifiedParserStartingState::Reasoning,
                UnifiedToolOutputMode::Native,
            ))
            .expect_err("Reasoning must be rejected: there is no channel to continue");
        assert!(err.to_string().contains("no reasoning channel"), "{err}");
    }

    #[test]
    fn neutral_and_response_starts_remain_accepted_without_a_reasoning_channel() {
        for state in [
            UnifiedParserStartingState::None,
            UnifiedParserStartingState::Response,
        ] {
            let mut p = reasoningless();
            assert!(
                p.initialize_request(init(state, UnifiedToolOutputMode::Native))
                    .is_ok(),
                "{state:?} must stay accepted; only an EXPLICIT Reasoning demand fails"
            );
        }
    }

    /// A rejected initialization must not have mutated anything. Otherwise a caller
    /// that catches the error and retries in a supported mode builds on half-applied
    /// state.
    #[test]
    fn a_rejected_initialization_leaves_the_parser_reusable() {
        let mut p = reasoningless();

        assert!(
            p.initialize_request(init(
                UnifiedParserStartingState::Reasoning,
                UnifiedToolOutputMode::Native
            ))
            .is_err()
        );
        assert_eq!(
            p.starting_state,
            UnifiedParserStartingState::None,
            "starting_state must be untouched by a rejected initialize"
        );

        // Guided is unsupported here too, and used to mutate before discovering that.
        assert!(
            p.initialize_request(init(
                UnifiedParserStartingState::None,
                UnifiedToolOutputMode::GuidedJson { named_tool: None }
            ))
            .is_err()
        );
        assert!(
            p.guided.is_none(),
            "a rejected guided initialize must not install guided state"
        );

        // ...and the parser still works through a supported mode afterwards.
        p.initialize_request(init(
            UnifiedParserStartingState::None,
            UnifiedToolOutputMode::Native,
        ))
        .expect("a supported mode must still initialize after two rejections");
        let out = p.push("plain text").unwrap();
        assert_eq!(out, vec![UnifiedParserEvent::Text("plain text".into())]);
    }
}
