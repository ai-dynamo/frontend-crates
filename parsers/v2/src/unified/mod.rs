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
///
/// Name, variant order and payload shapes are aligned with the peer streaming-parser
/// traits, so the two translate variant-for-variant under a compiler rather than by a
/// reader's judgement.
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

/// A parser that owns reasoning + content + tool calls for one stream.
///
/// Streaming-first, like [`crate::ToolParser`]: [`Self::parse_into`] per decoded
/// delta, [`Self::finish`] once at end of stream. One instance parses exactly one
/// choice of one request, which is what gives per-stream isolation (`I4`) by
/// construction.
pub trait UnifiedParser: Send {
    /// Initialize parser state from prompt token IDs before output deltas arrive.
    ///
    /// This is the peer traits' `initialize` signature, so a caller written against
    /// them reaches the same method with the same argument here. The default detects
    /// nothing, matching the peer default; a family whose prompt can end mid-channel
    /// overrides it and reads the tokens.
    fn initialize(&mut self, _prompt_token_ids: &[u32]) -> Result<()> {
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
    /// Open reasoning is promoted here rather than dropped or leaked as text, and an
    /// unrecoverable partial tool call is dropped without erroring (policy P2 —
    /// best-effort recovery).
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
    /// calls already emitted belong to the abandoned one — feeding the remainder back
    /// into the same turn would re-number from index 0 and collide with them.
    fn reset(&mut self) -> String {
        String::new()
    }

    /// Whether decoded output must keep tokenizer special tokens.
    ///
    /// A family whose markers ARE special tokens cannot be parsed from text that
    /// dropped them.
    fn preserve_special_tokens(&self) -> bool {
        false
    }

    /// The model-emitted id for a tool call, when the grammar carries one.
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

    // --- Accumulation helpers, aligned with the peer traits in name and semantics.
    // These COALESCE: appending text onto a trailing text event extends it rather than
    // adding a second one. `assemble` performs the same fold, so a caller that
    // accumulates through these and one that folds afterwards agree.

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

    /// Collapse into assembled events (see [`assemble`]).
    pub fn assembled(&self) -> Vec<UnifiedEvent> {
        assemble(&self.events)
    }
}

// Additive ergonomics: the type carries a single `events` field, so these cannot
// change what is emitted — they only spare every caller an explicit `.events`.
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
}

impl<E: InvokeEmitter + Send> UnifiedParser for ScannerUnified<E> {
    /// Whether decoding must preserve special tokens, delegated to the shared grammar.
    ///
    /// NOT the trait default. Inheriting `false` here while the tool-only adapter over
    /// this same scanner returned `true` meant two surfaces reported contradictory
    /// decoding requirements for identical markup. The value now lives on the grammar and
    /// both surfaces read it, so they cannot disagree.
    fn preserve_special_tokens(&self) -> bool {
        self.scanner.preserve_special_tokens()
    }

    /// The trait default returns an empty string and leaves state untouched, which
    /// would tell a caller nothing was buffered while the scanner still held a partial
    /// marker, `in_block`, and a used `next_index`. A caller following the documented
    /// recovery path after a `parse_into` error would then resume on stale state and
    /// mis-number tool indices. The only shipped family must honour the contract.
    fn reset(&mut self) -> String {
        self.scanner.reset_stream()
    }

    fn parse_into(&mut self, delta: &str, output: &mut UnifiedParserOutput) -> Result<()> {
        // Straight into the caller's output, with no vector in between. Two reasons, and
        // the first is a correctness one: an event written here is COMMITTED, so a later
        // error in the same advance cannot retract it. Collecting into a local vector and
        // copying at the end meant `?` dropped everything already emitted. It also drops
        // the second allocation per advance. Coalescing is unchanged: the sink routes
        // through these same `push_*` helpers.
        self.scanner.push_ordered_into(delta, output)
    }

    fn finish(&mut self) -> Result<UnifiedParserOutput> {
        // Direct into the output, matching `parse_into`: no intermediate vector, and the
        // events reach the caller through the same coalescing helpers.
        let mut out = UnifiedParserOutput::default();
        self.scanner.finish_ordered_into(&mut out)?;
        Ok(out)
    }
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

            Ok(parser)
        }
    };
}

unified_registry! {
    "qwen3" | "qwen3_coder" => qwen3::qwen3_unified,
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
        ScannerUnified {
            scanner: failing_scanner(),
        }
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
        assert_eq!(p.reset(), "<function=boom></function></tool_call>suffix");
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
