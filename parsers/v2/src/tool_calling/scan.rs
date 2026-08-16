// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared streaming-scan core for the marker-delimited tool-call families.
//!
//! Before this module, every family reimplemented the same streaming concerns
//! by copy-paste-and-edit: the longest-prefix marker holdback existed 7 times,
//! the source-order argument reserialization 5 times, and the buffer-and-scan
//! `drain(flush)` loop 7 times (the MiniMax M2 vs M3 copies were ~90%
//! line-identical). Divergent copies are how chunk-boundary bugs get fixed in
//! one family and stay broken in the others; this module is the single parent.
//!
//! Three layers, adopted per family as far as its grammar allows:
//!
//! * [`marker_prefix_suffix_len`] — partial-marker holdback, used by ALL
//!   scan families (the bespoke ones compose it with extra components).
//! * [`reorder_arguments`] — source-order argument reserialization for the
//!   families that delegate typing to a v1core batch parser (which builds
//!   arguments from a `HashMap` with nondeterministic key order).
//! * [`WrappedBlockScanner`] — the whole drain loop for the four families
//!   whose grammar is `BLOCK_START (INVOKE .. INVOKE_END)* BLOCK_END` with a
//!   bare-invoke back-off: qwen3_coder, minimax_m2, minimax_m3, kimi_k2.
//!   dsml (incremental invoke-header state), glm47 (identifier-anchored bare
//!   recovery), and gemma4 (brace/string-aware end scanning) keep bespoke
//!   drains and share the primitives.
//!
//! Behavior differences between the wrapped families are explicit
//! [`WrappedBlockSpec`] fields, not silent copy drift — e.g. MiniMax M2
//! clears the normal-text suppression latch after a bare-invoke recovery
//! while M3/qwen3/kimi keep it latched ([`BareRecoveryLatch`]).
//!
//! # Ordered output
//!
//! The drain loop emits [`UnifiedParserEvent`]s — text, reasoning and calls in the
//! order the model produced them. Tool-only parsers project that down to
//! [`ToolParseResult`] via `push`/`finish` and see no behavior change (that
//! projection is lossy about order, which is all the tool-only contract ever
//! promised). Unified parsers take the ordered list through `push_ordered` /
//! `finish_ordered`. One scan, two views — so reasoning-aware and
//! reasoning-unaware families cannot drift on marker handling, chunk-boundary
//! holdback, or recovery.
//!
//! Reasoning is opt-in per scanner via [`WrappedBlockScanner::with_reasoning`].
//! The two nestings are deliberately ASYMMETRIC, because tool structure
//! dominates: a tool call emitted INSIDE a thought is a real call, so it is
//! extracted and the thought splits around it; but a reasoning marker inside a
//! tool argument is argument data (`I7`), because the in-block scan never looks
//! for one. Burying a nested call would drop it AND leak its markup into the
//! reasoning payload (`I3`).

use std::collections::HashSet;

use crate::tool_calling::traits::{ToolCallDelta, ToolParseResult};
use crate::unified::{Kind, UnifiedParserEvent};

/// Longest non-empty proper prefix of any `marker` that `text` ends with, so a
/// marker split across chunk boundaries is held back instead of leaked as
/// normal_text. Closing markers belong in the list too: a split stray/orphan
/// close must be retained whole so the orphan-close path (which strips it and
/// never lets it leak) can match it — otherwise the partial suffix is emitted
/// (or, under a suppression latch, silently discarded) and the remainder is
/// unrecognizable.
pub(crate) fn marker_prefix_suffix_len<'a, I>(text: &str, markers: I) -> usize
where
    I: IntoIterator<Item = &'a str>,
{
    markers
        .into_iter()
        .filter_map(|marker| {
            marker
                .char_indices()
                .map(|(idx, _)| idx)
                .filter(|idx| *idx > 0 && *idx < marker.len())
                .rev()
                .find(|&len| text.ends_with(&marker[..len]))
        })
        .max()
        .unwrap_or(0)
}

/// Re-serialize a v1core arguments JSON object in the model-emitted source
/// order (`source_names`, extracted from the raw invoke body by the family).
/// A repeated source name is emitted once (the v1 object holds one value per
/// key); keys absent from the source order are appended in object order
/// (defensive; normally empty). Non-object payloads pass through untouched.
pub(crate) fn reorder_arguments(arguments: &str, source_names: &[String]) -> String {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(arguments) else {
        return arguments.to_string();
    };
    let Some(obj) = value.as_object() else {
        return arguments.to_string();
    };
    let mut parts: Vec<String> = Vec::new();
    let mut seen: HashSet<&str> = HashSet::new();
    for name in source_names {
        if let Some(val) = obj.get(name)
            && seen.insert(name.as_str())
        {
            parts.push(format!(
                "{}:{}",
                serde_json::to_string(name).unwrap_or_default(),
                serde_json::to_string(val).unwrap_or_default()
            ));
        }
    }
    for (key, val) in obj {
        if !seen.contains(key.as_str()) {
            parts.push(format!(
                "{}:{}",
                serde_json::to_string(key).unwrap_or_default(),
                serde_json::to_string(val).unwrap_or_default()
            ));
        }
    }
    format!("{{{}}}", parts.join(","))
}

/// What happens to the normal-text suppression latch after a bare-invoke
/// recovery emits a call.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum BareRecoveryLatch {
    /// Clear the latch: when the optional outer close is absent, later
    /// narration must still reach normal_text (MiniMax M2 semantics; a stray
    /// close that DOES follow is stripped by the orphan-close handling).
    Clear,
    /// Keep the latch set: trailing markup around the recovered invoke is
    /// dropped until an orphan close ends the markup context
    /// (MiniMax M3 / Qwen3-Coder / Kimi K2 semantics).
    Set,
}

/// When the in-block invoke latch engages after parsing a complete invoke.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum InvokeLatch {
    /// Only when the invoke produced a call delta (XML families).
    IfEmitted,
    /// Always — even when the invoke parsed to nothing (Kimi K2).
    Always,
}

/// Declarative description of one wrapped-invoke grammar.
pub(crate) struct WrappedBlockSpec {
    /// Family name used in tracing `why` diagnostics.
    pub family: &'static str,
    /// Block openers (Kimi has singular/plural section variants).
    pub block_starts: Vec<String>,
    /// Block closers, matched earliest-first.
    pub block_ends: Vec<String>,
    /// Invoke opener (prefix form is fine — it only anchors scanning).
    pub invoke_start: String,
    /// Invoke closer; an invoke is parsed only once this has streamed.
    pub invoke_end: String,
    /// Markers that are stray markup when seen OUTSIDE a block before any
    /// opener; stripped so they never leak (always includes `block_ends`).
    pub orphan_markers: Vec<String>,
    /// Markers whose split prefixes are held back at chunk boundaries.
    pub holdback_markers: Vec<String>,
    pub bare_recovery_latch: BareRecoveryLatch,
    pub invoke_latch: InvokeLatch,
    /// Refuse to let an invoke body cross a block close: a `block_end` before
    /// the `invoke_end` means the call is malformed — drop it and close the
    /// block (Kimi K2's mismatched-fences rule).
    pub drop_invoke_crossing_block_end: bool,
    /// Whether a decoder must keep tokenizer special tokens so this grammar's
    /// markers survive to the parser.
    ///
    /// Lives on the GRAMMAR, not on an adapter, because it is a property of the
    /// markers being scanned. Two adapters over one scanner previously answered
    /// differently for identical markup — the tool-only parser said `true` while the
    /// unified one inherited the trait default `false`.
    pub preserve_special_tokens: bool,
}

/// The reasoning channel a unified scanner also owns.
///
/// Only meaningful outside tool blocks — see the module docs. `Copy` over
/// `&'static str`: every family's markers are compile-time constants, so
/// `drain_reasoning` copies two pointers per push instead of allocating two
/// `String`s on the hot path.
#[derive(Clone, Copy)]
pub(crate) struct ReasoningSpec {
    /// Opener, e.g. `<think>`.
    pub start: &'static str,
    /// Closer, e.g. `</think>`.
    pub end: &'static str,
    /// Stream begins INSIDE reasoning with no opener, because the chat template
    /// pre-filled it (policy P5). Qwen3 is not one of these; DeepSeek-R1-style
    /// forced-reasoning templates are.
    pub forced_start: bool,
    /// Whether the reasoning markers require special-token preservation.
    pub preserve_special_tokens: bool,
}

/// Per-family hook: parse one complete invoke (opener..closer inclusive) into
/// a call delta. `None` means the invoke was malformed and is dropped.
pub(crate) trait InvokeEmitter {
    fn parse_invoke(
        &self,
        invoke: &str,
        tool_index: usize,
    ) -> anyhow::Result<Option<ToolCallDelta>>;
}

/// First occurrence of any of `markers` in `text`: `(position, marker_len)`.
fn find_first(text: &str, markers: &[String]) -> Option<(usize, usize)> {
    markers
        .iter()
        .filter_map(|m| text.find(m.as_str()).map(|p| (p, m.len())))
        .min_by_key(|(p, _)| *p)
}

/// Earliest `(position, payload)` among the candidates.
///
/// Both scan scopes — inside a thought and outside a block — answer the same
/// question: of everything that could interrupt the current run, which comes
/// first? One helper keeps that precedence rule in one place, and ties resolve to
/// the FIRST candidate listed, so each call site states its own priority order
/// simply by the order it writes them.
fn earliest<T>(candidates: impl IntoIterator<Item = Option<(usize, T)>>) -> Option<(usize, T)> {
    candidates.into_iter().flatten().min_by_key(|(pos, _)| *pos)
}

#[derive(Clone, Copy)]
enum Marker {
    /// A block opener; carries the matched token length.
    Block(usize),
    /// A bare invoke opener with no block wrapper (recovery path).
    BareInvoke,
    /// A reasoning opener; carries the matched token length.
    ReasoningStart(usize),
}

/// What the earliest marker inside an OPEN reasoning span means.
enum InReasoning {
    /// The span's own closer — reasoning ends here.
    Close,
    /// A tool opener — suspend the thought, resume after the call closes.
    ToolOpen,
    /// Malformed markup to strip, carrying its length.
    Stray(usize),
}

/// Append a text run, merging into a trailing run of the SAME kind so one drain
/// never emits a string of adjacent same-kind fragments (`I8`). The or-pattern is
/// the whole point: text and reasoning coalesce by identical rules, so there is
/// one implementation to get right instead of two that can drift.
/// Where the drain loop puts events as it commits them.
///
/// The loop writes through this rather than into a local `Vec` it returns at the end.
/// That is what makes the error contract honest: an event handed to the sink belongs to
/// the CALLER, so a later `Err` in the same advance cannot take it back. A local vector
/// is dropped by `?` on the way out, which silently un-commits everything already
/// emitted in that advance.
///
/// Method names match the peer accumulation helpers, so an implementation reads the
/// same on both sides.
pub(crate) trait EventSink {
    /// Append visible text, merging into a trailing text event.
    fn push_text(&mut self, text: &str);
    /// Append reasoning text, merging into a trailing reasoning event.
    fn push_reasoning(&mut self, text: &str);
    /// Append one tool call. Calls never merge.
    fn push_call(&mut self, call: ToolCallDelta);
}

/// The batch/tool-only path still collects into a vector.
impl EventSink for Vec<UnifiedParserEvent> {
    fn push_text(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        if let Some(UnifiedParserEvent::Text(prev)) = self.last_mut() {
            prev.push_str(text);
            return;
        }
        self.push(UnifiedParserEvent::Text(text.to_string()));
    }

    fn push_reasoning(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        if let Some(UnifiedParserEvent::Reasoning(prev)) = self.last_mut() {
            prev.push_str(text);
            return;
        }
        self.push(UnifiedParserEvent::Reasoning(text.to_string()));
    }

    fn push_call(&mut self, call: ToolCallDelta) {
        self.push(UnifiedParserEvent::ToolCall(call));
    }
}

pub(crate) fn push_run<S: EventSink + ?Sized>(out: &mut S, kind: Kind, text: &str) {
    match kind {
        Kind::Text => out.push_text(text),
        Kind::Reasoning => out.push_reasoning(text),
    }
}

/// The shared buffer-and-scan drain loop for wrapped-invoke grammars.
///
/// Streaming contract (identical to the loops it replaces): natural text
/// around COMPLETE blocks is preserved verbatim (prefix / inter-block /
/// trailing); markup of complete blocks is stripped; a bare invoke without
/// its wrapper is recovered once complete; stray orphan closes are stripped
/// and never leak; a partial marker at a chunk boundary is held back; at
/// flush (EOF) incomplete blocks/invokes are dropped with a `why=` warning
/// rather than leaked.
pub(crate) struct WrappedBlockScanner<E: InvokeEmitter> {
    spec: WrappedBlockSpec,
    emitter: E,
    reasoning: Option<ReasoningSpec>,
    buffer: String,
    in_block: bool,
    in_reasoning: bool,
    /// A tool call opened INSIDE a reasoning span; reasoning resumes when the
    /// call closes. Without this the tail of the thought would surface as
    /// visible text instead of reasoning.
    resume_reasoning: bool,
    suppress_normal_text: bool,
    next_index: usize,
}

impl<E: InvokeEmitter> WrappedBlockScanner<E> {
    pub(crate) fn new(spec: WrappedBlockSpec, emitter: E) -> Self {
        Self {
            spec,
            emitter,
            reasoning: None,
            buffer: String::new(),
            in_block: false,
            in_reasoning: false,
            resume_reasoning: false,
            suppress_normal_text: false,
            next_index: 0,
        }
    }

    /// Also own the reasoning channel, making this scanner unified.
    ///
    /// The reasoning markers are registered for chunk-boundary holdback here
    /// rather than by the caller, so a family cannot add reasoning and forget
    /// the holdback (which would leak a split `<thi` + `nk>` as visible text).
    /// The closer additionally joins `orphan_markers`: a stray `</think>` with
    /// nothing open is malformed markup and is stripped, never leaked (`I3`).
    pub(crate) fn with_reasoning(mut self, reasoning: ReasoningSpec) -> Self {
        self.spec.holdback_markers.push(reasoning.start.to_string());
        self.spec.holdback_markers.push(reasoning.end.to_string());
        self.spec.orphan_markers.push(reasoning.end.to_string());
        self.in_reasoning = reasoning.forced_start;
        self.reasoning = Some(reasoning);
        self
    }

    pub(crate) fn push(&mut self, chunk: &str) -> anyhow::Result<ToolParseResult> {
        Ok(ToolParseResult::from_deltas(self.push_ordered(chunk)?))
    }

    pub(crate) fn finish(&mut self) -> anyhow::Result<ToolParseResult> {
        Ok(ToolParseResult::from_deltas(self.finish_ordered()?))
    }

    /// Whether decoding must preserve special tokens for THIS scanner's grammar.
    ///
    /// The OR over every component whose markers the scan consumes — the peer's
    /// `CombinedParser` rule, expressed over one scanner rather than two wrapped
    /// parser objects. A component that needs preservation cannot have its
    /// requirement dropped by the composition.
    pub(crate) fn preserve_special_tokens(&self) -> bool {
        self.spec.preserve_special_tokens
            || self
                .reasoning
                .as_ref()
                .is_some_and(|r| r.preserve_special_tokens)
    }

    pub(crate) fn push_ordered(&mut self, chunk: &str) -> anyhow::Result<Vec<UnifiedParserEvent>> {
        let mut out = Vec::new();
        self.push_ordered_into(chunk, &mut out)?;
        Ok(out)
    }

    pub(crate) fn finish_ordered(&mut self) -> anyhow::Result<Vec<UnifiedParserEvent>> {
        let mut out = Vec::new();
        self.finish_ordered_into(&mut out)?;
        Ok(out)
    }

    /// Drain one advance directly into the caller's sink.
    ///
    /// Preferred over [`Self::push_ordered`] on any fallible path: events reach the
    /// caller as they are committed, so an `Err` later in the same advance leaves the
    /// earlier ones in place instead of dropping them with the returned vector.
    pub(crate) fn push_ordered_into<S: EventSink + ?Sized>(
        &mut self,
        chunk: &str,
        out: &mut S,
    ) -> anyhow::Result<()> {
        self.buffer.push_str(chunk);
        self.drain(false, out)
    }

    pub(crate) fn finish_ordered_into<S: EventSink + ?Sized>(
        &mut self,
        out: &mut S,
    ) -> anyhow::Result<()> {
        self.drain(true, out)
    }

    /// Position and length of the reasoning opener in the buffer, if configured.
    fn find_reasoning_start(&self) -> Option<(usize, usize)> {
        let reasoning = self.reasoning.as_ref()?;
        let pos = self.buffer.find(reasoning.start)?;
        Some((pos, reasoning.start.len()))
    }

    /// Consume buffered reasoning up to its closer, a nested tool opener, or as
    /// far as is safe.
    ///
    /// Returns `true` once the reasoning span yielded (closer seen, tool opener
    /// reached, or promoted at EOF) and the caller should keep draining; `false`
    /// means more input is needed.
    fn drain_reasoning<S: EventSink + ?Sized>(
        &mut self,
        out: &mut S,
        flush: bool,
    ) -> anyhow::Result<bool> {
        let Some(reasoning) = self.reasoning else {
            anyhow::bail!("reasoning state active without a reasoning spec");
        };
        let (start, end) = (reasoning.start, reasoning.end);

        // Everything that can interrupt an open thought, as (position, meaning).
        // Collecting them into ONE list and taking the earliest keeps the
        // precedence rule in a single place — and makes it visible in a debugger
        // as a list of candidates rather than three separately-compared Options.
        // Everything that can interrupt an open thought, in PRECEDENCE order:
        //  - the span's own closer;
        //  - a tool opener — tool structure dominates reasoning, so a call emitted
        //    INSIDE a thought is extracted and the thought splits around it. Burying
        //    it would drop the call AND leak its markup (`I3`). The reverse nesting
        //    is NOT symmetric: a reasoning marker inside a tool argument stays
        //    argument data (`I7`), because the in-block branch never scans for it;
        //  - malformed markup (duplicate opener / stray tool close) to strip, the
        //    same rule the orphan handler applies outside reasoning, so being inside
        //    a thought cannot turn markup into content (`I3`). The closer is excluded
        //    from this set because it legitimately ends the span.
        let tool_open = find_first(&self.buffer, &self.spec.block_starts)
            .map(|(pos, _)| pos)
            .into_iter()
            .chain(self.buffer.find(self.spec.invoke_start.as_str()))
            .min();
        let stray = std::iter::once(start)
            .chain(
                self.spec
                    .orphan_markers
                    .iter()
                    .map(String::as_str)
                    .filter(|m| *m != end),
            )
            .filter_map(|m| self.buffer.find(m).map(|pos| (pos, m.len())))
            .min_by_key(|(pos, _)| *pos);

        if let Some((at, what)) = earliest([
            self.buffer.find(end).map(|pos| (pos, InReasoning::Close)),
            tool_open.map(|pos| (pos, InReasoning::ToolOpen)),
            stray.map(|(pos, len)| (pos, InReasoning::Stray(len))),
        ]) {
            // One transition table: how many marker bytes to consume, and the state
            // the span is left in. ToolOpen consumes nothing — the block handler
            // still needs to see its own opener.
            let (consume, in_reasoning, resume) = match what {
                InReasoning::Close => (end.len(), false, false),
                InReasoning::ToolOpen => (0, false, true),
                InReasoning::Stray(len) => (len, true, false),
            };
            push_run(out, Kind::Reasoning, &self.buffer[..at]);
            self.buffer.drain(..at + consume);
            self.in_reasoning = in_reasoning;
            self.resume_reasoning = resume;
            if matches!(what, InReasoning::Stray(_)) {
                tracing::debug!(
                    why = %format!("{}_stray_marker_in_reasoning", self.spec.family),
                    "stream stripped malformed markup inside a reasoning span"
                );
            }
            return Ok(true);
        }

        // Nothing complete yet: stream what has arrived, holding back ANY marker
        // this scanner reacts to that is split across the chunk boundary.
        let keep = if flush {
            0
        } else {
            marker_prefix_suffix_len(
                &self.buffer,
                self.spec.holdback_markers.iter().map(String::as_str),
            )
        };
        let emit_len = self.buffer.len().saturating_sub(keep);
        if emit_len > 0 {
            push_run(out, Kind::Reasoning, &self.buffer[..emit_len]);
            self.buffer.drain(..emit_len);
        }
        if flush {
            // 4.e: the stream ended mid-thought. The open reasoning is promoted
            // as reasoning — not dropped, and not leaked as visible text.
            self.in_reasoning = false;
            self.resume_reasoning = false;
            tracing::debug!(
                why = %format!("{}_reasoning_open_at_eof", self.spec.family),
                "stream promoted unterminated reasoning at EOF"
            );
        }
        Ok(false)
    }

    /// Return to a FRESH-STREAM state and hand back whatever was still buffered.
    ///
    /// Every field that carries stream position resets, `next_index` included: the
    /// returned text is a NEW stream, so a call parsed out of it must not reuse an
    /// index already dispatched from the abandoned one.
    pub(crate) fn reset_stream(&mut self) -> String {
        let carried = std::mem::take(&mut self.buffer);
        self.in_block = false;
        self.in_reasoning = self.reasoning.as_ref().is_some_and(|r| r.forced_start);
        self.resume_reasoning = false;
        self.suppress_normal_text = false;
        self.next_index = 0;
        carried
    }

    fn drain<S: EventSink + ?Sized>(&mut self, flush: bool, out: &mut S) -> anyhow::Result<()> {
        loop {
            // Reasoning yields to tool structure: while a thought is open, its
            // closer OR a nested tool opener ends the current reasoning run (see
            // `drain_reasoning`), so a call emitted inside a thought is extracted
            // and the thought resumes after it.
            if self.in_reasoning {
                if self.drain_reasoning(out, flush)? {
                    continue;
                }
                break;
            }

            if self.in_block {
                let invoke_start = self.buffer.find(self.spec.invoke_start.as_str());

                // Close the block once no more complete invokes precede its end.
                if let Some((end_pos, end_len)) = find_first(&self.buffer, &self.spec.block_ends) {
                    let invoke_before_end = invoke_start.is_some_and(|start| start < end_pos);
                    if !invoke_before_end {
                        // Complete block fully closed: drop its markup and resume
                        // keeping natural text (inter-block / trailing). Any later
                        // block re-enters `in_block` and re-suppresses its markup.
                        // Matches the v1 batch parsers (cases 8.b/8.c/8.d).
                        self.buffer.drain(..end_pos + end_len);
                        self.in_block = false;
                        self.suppress_normal_text = false;
                        self.in_reasoning = std::mem::take(&mut self.resume_reasoning);
                        continue;
                    }
                }

                let Some(start) = invoke_start else {
                    if flush {
                        tracing::warn!(
                            why = %format!("{}_block_without_complete_invoke", self.spec.family),
                            "stream dropped incomplete block at EOF"
                        );
                        self.buffer.clear();
                        self.in_block = false;
                    }
                    break;
                };
                if start > 0 {
                    self.buffer.drain(..start);
                }
                let Some(end) = self.buffer.find(self.spec.invoke_end.as_str()) else {
                    if flush {
                        tracing::warn!(
                            why = %format!("{}_incomplete_invoke", self.spec.family),
                            "stream dropped incomplete invoke at EOF"
                        );
                        self.buffer.clear();
                        self.in_block = false;
                    }
                    break;
                };
                // Mismatched fences: a block close inside the invoke body means
                // the invoke never closed. Drop it and close the block; narration
                // after the close is the user's text again.
                if self.spec.drop_invoke_crossing_block_end
                    && let Some((be_pos, be_len)) = find_first(&self.buffer, &self.spec.block_ends)
                    && be_pos < end
                {
                    tracing::warn!(
                        why = %format!("{}_incomplete_invoke", self.spec.family),
                        "stream dropped invoke missing its close before the block end"
                    );
                    self.buffer.drain(..be_pos + be_len);
                    self.in_block = false;
                    self.suppress_normal_text = false;
                    continue;
                }
                let invoke_len = end + self.spec.invoke_end.len();
                let invoke = self.buffer[..invoke_len].to_string();
                // Emit BEFORE consuming. `parse_invoke` is fallible, and draining first
                // meant a failing emitter destroyed the invoke bytes: they were gone from
                // the buffer, so `reset_stream` could no longer hand them back and the
                // documented recovery contract was false for the only shipped family.
                let emitted = self.emitter.parse_invoke(&invoke, self.next_index)?;
                self.buffer.drain(..invoke_len);
                if let Some(delta) = emitted {
                    out.push_call(delta);
                    self.next_index += 1;
                    if self.spec.invoke_latch == InvokeLatch::IfEmitted {
                        self.suppress_normal_text = true;
                    }
                }
                if self.spec.invoke_latch == InvokeLatch::Always {
                    self.suppress_normal_text = true;
                }
                continue;
            }

            // Not in a block. A COMPLETE orphan marker (stray close / inner
            // marker) before any opener is malformed markup: strip it so it can
            // NEVER leak into normal_text; when suppression is off, first emit
            // the natural text preceding it. Clear the latch either way (the
            // markup context has ended).
            if let Some((pos, len)) = find_first(&self.buffer, &self.spec.orphan_markers) {
                let next_open = find_first(&self.buffer, &self.spec.block_starts)
                    .map(|(p, _)| p)
                    .into_iter()
                    .chain(self.buffer.find(self.spec.invoke_start.as_str()))
                    // A reasoning opener counts as an opener here, so the
                    // matching closer in `<think>a</think>` is not mistaken for
                    // an orphan and stripped.
                    .chain(self.find_reasoning_start().map(|(p, _)| p))
                    .min();
                if next_open.is_none_or(|open| pos < open) {
                    if !self.suppress_normal_text && pos > 0 {
                        push_run(out, Kind::Text, &self.buffer[..pos]);
                    }
                    self.buffer.drain(..pos + len);
                    self.suppress_normal_text = false;
                    continue;
                }
            }

            // Block wins a tie with a bare invoke, preserving the pre-unified
            // tie-break.
            let next_marker = earliest([
                find_first(&self.buffer, &self.spec.block_starts)
                    .map(|(pos, len)| (pos, Marker::Block(len))),
                self.buffer
                    .find(self.spec.invoke_start.as_str())
                    .map(|pos| (pos, Marker::BareInvoke)),
                self.find_reasoning_start()
                    .map(|(pos, len)| (pos, Marker::ReasoningStart(len))),
            ]);

            let Some((start, marker)) = next_marker else {
                // No marker present: emit buffered text, but hold back a trailing
                // partial marker (split across this chunk boundary) unless flushing.
                let keep = if flush {
                    0
                } else {
                    marker_prefix_suffix_len(
                        &self.buffer,
                        self.spec.holdback_markers.iter().map(String::as_str),
                    )
                };
                let emit_len = self.buffer.len().saturating_sub(keep);
                if emit_len > 0 {
                    if !self.suppress_normal_text {
                        push_run(out, Kind::Text, &self.buffer[..emit_len]);
                    }
                    self.buffer.drain(..emit_len);
                }
                break;
            };

            if start > 0 {
                if !self.suppress_normal_text {
                    push_run(out, Kind::Text, &self.buffer[..start]);
                }
                self.buffer.drain(..start);
            }

            match marker {
                Marker::Block(blen) => {
                    self.buffer.drain(..blen);
                    self.in_block = true;
                    self.suppress_normal_text = true;
                }
                Marker::ReasoningStart(rlen) => {
                    self.buffer.drain(..rlen);
                    self.in_reasoning = true;
                    // An explicit reasoning opener is an unambiguous return to
                    // real content, so it ends any markup-suppression context
                    // left over from a bare-invoke recovery. Without this a
                    // thought following a recovered call would be dropped.
                    self.suppress_normal_text = false;
                }
                Marker::BareInvoke => {
                    // A bare invoke (no wrapper) is recovered only once its close
                    // has streamed; otherwise wait for more input.
                    let Some(end) = self.buffer.find(self.spec.invoke_end.as_str()) else {
                        if flush {
                            tracing::warn!(
                                why = %format!("{}_incomplete_bare_invoke", self.spec.family),
                                "stream dropped incomplete bare invoke at EOF"
                            );
                            self.buffer.clear();
                        }
                        break;
                    };
                    let invoke_len = end + self.spec.invoke_end.len();
                    let invoke = self.buffer[..invoke_len].to_string();
                    // Emit before consuming — same recovery contract as the wrapped site.
                    let emitted = self.emitter.parse_invoke(&invoke, self.next_index)?;
                    self.buffer.drain(..invoke_len);
                    if let Some(delta) = emitted {
                        tracing::warn!(
                            why = %format!("{}_bare_invoke_recovery", self.spec.family),
                            tool_index = delta.tool_index,
                            "stream recovered a complete bare invoke"
                        );
                        out.push_call(delta);
                        self.next_index += 1;
                        self.suppress_normal_text =
                            self.spec.bare_recovery_latch == BareRecoveryLatch::Set;
                    }
                    // A bare invoke nested in a thought has no block close to
                    // resume on, so resume here.
                    if self.resume_reasoning {
                        self.suppress_normal_text = false;
                    }
                    self.in_reasoning = std::mem::take(&mut self.resume_reasoning);
                }
            }
        }

        Ok(())
    }
}

/// Test-only: an emitter that fails partway through a stream, so the recovery contract
/// can be exercised from every surface that drives a scanner.
#[cfg(test)]
pub(crate) mod test_support {
    use super::*;

    /// Succeeds until it sees `boom`, then fails — the shape of any real emitter whose
    /// arguments fail to type-check partway through a stream.
    pub(crate) struct FailOnBoom;

    impl InvokeEmitter for FailOnBoom {
        fn parse_invoke(
            &self,
            invoke: &str,
            tool_index: usize,
        ) -> anyhow::Result<Option<ToolCallDelta>> {
            if invoke.contains("boom") {
                anyhow::bail!("injected emitter failure");
            }
            Ok(Some(ToolCallDelta {
                tool_index,
                name: Some("ok".to_string()),
                arguments: "{}".to_string(),
            }))
        }
    }

    pub(crate) fn failing_scanner() -> WrappedBlockScanner<FailOnBoom> {
        WrappedBlockScanner::new(
            WrappedBlockSpec {
                family: "test",
                block_starts: vec!["<tool_call>".into()],
                block_ends: vec!["</tool_call>".into()],
                invoke_start: "<function=".into(),
                invoke_end: "</function>".into(),
                orphan_markers: vec!["</tool_call>".into()],
                holdback_markers: vec!["<tool_call>".into(), "</tool_call>".into()],
                bare_recovery_latch: BareRecoveryLatch::Set,
                invoke_latch: InvokeLatch::IfEmitted,
                drop_invoke_crossing_block_end: false,
                preserve_special_tokens: true,
            },
            FailOnBoom,
        )
    }
}

#[cfg(test)]
mod recovery_tests {
    use super::test_support::*;
    use super::*;

    /// Scanner-level contract: on an emitter error the failed invoke stays in the buffer,
    /// so `reset_stream` can hand it back. Pre-fix this drained before the fallible call.
    #[test]
    fn wrapped_emitter_error_leaves_invoke_recoverable() {
        let mut s = failing_scanner();
        let mut out: Vec<UnifiedParserEvent> = Vec::new();
        let r = s.push_ordered_into(
            "prefix<tool_call><function=ok></function><function=boom></function></tool_call>suffix",
            &mut out,
        );
        assert!(r.is_err(), "the injected emitter must surface its error");
        assert_eq!(
            s.reset_stream(),
            "<function=boom></function></tool_call>suffix",
            "the failed invoke and everything after it must remain recoverable"
        );
    }

    #[test]
    fn bare_emitter_error_leaves_invoke_recoverable() {
        let mut s = failing_scanner();
        let mut out: Vec<UnifiedParserEvent> = Vec::new();
        let r = s.push_ordered_into("prefix<function=boom></function>suffix", &mut out);
        assert!(r.is_err(), "the injected emitter must surface its error");
        assert_eq!(
            s.reset_stream(),
            "<function=boom></function>suffix",
            "the failed bare invoke and everything after it must remain recoverable"
        );
    }
}
