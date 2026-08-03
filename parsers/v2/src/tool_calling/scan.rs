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
//! The drain loop emits [`UnifiedDelta`]s — text, reasoning and calls in the
//! order the model produced them. Tool-only parsers project that down to
//! [`ToolParseResult`] via `push`/`finish` and see no behavior change (that
//! projection is lossy about order, which is all the tool-only contract ever
//! promised). Unified parsers take the ordered list through `push_ordered` /
//! `finish_ordered`. One scan, two views — so reasoning-aware and
//! reasoning-unaware families cannot drift on marker handling, chunk-boundary
//! holdback, or recovery.
//!
//! Reasoning is opt-in per scanner via [`WrappedBlockScanner::with_reasoning`].
//! It is scoped OUTSIDE tool blocks only: once a block or bare invoke is open,
//! a reasoning marker inside it is argument data, not a control token (`I7`).

use std::collections::HashSet;

use crate::tool_calling::traits::{ToolCallDelta, ToolParseResult};
use crate::unified::{Kind, UnifiedDelta};

/// Longest non-empty proper prefix of any `marker` that `text` ends with, so a
/// marker split across chunk boundaries is held back instead of leaked as
/// normal_text. Closing markers belong in the list too: a split stray/orphan
/// close must be retained whole so the orphan-close path (which strips it and
/// never lets it leak) can match it — otherwise the partial suffix is emitted
/// (or, under a suppression latch, silently discarded) and the remainder is
/// unrecognizable.
/// The earliest malformed marker inside an OPEN reasoning span, as `(pos, len)`.
///
/// The candidates are the span's own OPENER (a second one is a duplicate, not
/// content) plus every orphan marker except the closer, which legitimately ends
/// the span. Being inside a thought must not turn markup into content (`I3`).
///
/// Shared by the native scanner and the guided-JSON drain deliberately: these
/// were two implementations, and the guided one only looked for the closer, so
/// identical bytes produced `reasoning("a<think>b")` under guided decoding and
/// `reasoning("ab")` natively — the request mode silently changed the payload.
pub(crate) fn stray_in_reasoning(
    haystack: &str,
    start: &str,
    end: &str,
    orphan_markers: &[String],
) -> Option<(usize, usize)> {
    std::iter::once(start)
        .chain(
            orphan_markers
                .iter()
                .map(String::as_str)
                .filter(|m| *m != end),
        )
        .filter_map(|m| haystack.find(m).map(|pos| (pos, m.len())))
        .min_by_key(|(pos, _)| *pos)
}

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
pub(crate) fn push_run(out: &mut Vec<UnifiedDelta>, kind: Kind, text: &str) {
    if text.is_empty() {
        return;
    }
    match (out.last_mut(), kind) {
        (Some(UnifiedDelta::Text { text: prev }), Kind::Text)
        | (Some(UnifiedDelta::Reasoning { text: prev }), Kind::Reasoning) => prev.push_str(text),
        _ => out.push(match kind {
            Kind::Text => UnifiedDelta::Text {
                text: text.to_string(),
            },
            Kind::Reasoning => UnifiedDelta::Reasoning {
                text: text.to_string(),
            },
        }),
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
    reasoning_enabled: bool,
    buffer: String,
    /// Raw block bytes consumed before any call delta commits them.
    uncommitted_block: String,
    in_block: bool,
    in_reasoning: bool,
    accept_redundant_reasoning_start: bool,
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
            reasoning_enabled: false,
            buffer: String::new(),
            uncommitted_block: String::new(),
            in_block: false,
            in_reasoning: false,
            accept_redundant_reasoning_start: false,
            resume_reasoning: false,
            suppress_normal_text: false,
            next_index: 0,
        }
    }

    /// Also own the reasoning channel, making this scanner unified.
    ///
    /// Reasoning marker holdback and orphan-close handling are enabled here
    /// rather than by the caller, so a family cannot add reasoning and forget
    /// either behavior. They remain separate from the tool marker lists so
    /// [`Self::set_reasoning_mode`] can make the markers literal for response
    /// prefill without rebuilding the scanner.
    pub(crate) fn with_reasoning(mut self, reasoning: ReasoningSpec) -> Self {
        self.reasoning_enabled = true;
        self.in_reasoning = reasoning.forced_start;
        self.accept_redundant_reasoning_start = reasoning.forced_start;
        self.reasoning = Some(reasoning);
        self
    }

    /// The reasoning markers this scanner was configured with.
    ///
    /// `Copy`, so the guided-decoding path can hold them without borrowing the
    /// scanner — it needs the markers but never the scan state.
    pub(crate) fn reasoning_spec(&self) -> Option<ReasoningSpec> {
        self.reasoning
    }

    /// Every control marker of this family's tool grammar, as ONE set.
    ///
    /// This is `holdback_markers`, which the family already declares for exactly
    /// this purpose — the markers the scanner reacts to, so a split one is never
    /// leaked. The guided path needs the same set for BOTH lookup and boundary
    /// holdback, and assembling it there from `orphan_markers` plus openers, at
    /// several sites, is how the two drifted apart repeatedly: a closer stripped
    /// while the opener beside it leaked, an opener recognised whole but lost when
    /// split. One owner, one set, both uses.
    pub(crate) fn control_markers(&self) -> &[String] {
        &self.spec.holdback_markers
    }

    /// The invoke terminator, for pairing with a stripped invoke opener.
    ///
    /// NOT part of [`Self::control_markers`]: a BARE `</function>` with no invoke
    /// open is ordinary text to the native scanner, and measured identical on both
    /// paths, so stripping it unconditionally would create a divergence rather than
    /// remove one. It is consumed only as the tail of an invoke already stripped.
    pub(crate) fn invoke_end(&self) -> &str {
        &self.spec.invoke_end
    }

    /// Select whether this stream interprets reasoning markers, without
    /// rebuilding the scanner or cloning its tool schemas.
    pub(crate) fn set_reasoning_mode(&mut self, enabled: bool, forced_start: bool) {
        debug_assert!(!enabled || self.reasoning.is_some());
        self.reasoning_enabled = enabled;
        if let Some(reasoning) = self.reasoning.as_mut() {
            reasoning.forced_start = forced_start;
        }
        self.in_reasoning = enabled && forced_start;
        self.accept_redundant_reasoning_start = self.in_reasoning;
    }

    pub(crate) fn push(&mut self, chunk: &str) -> anyhow::Result<ToolParseResult> {
        Ok(ToolParseResult::from_deltas(self.push_ordered(chunk)?))
    }

    pub(crate) fn finish(&mut self) -> anyhow::Result<ToolParseResult> {
        Ok(ToolParseResult::from_deltas(self.finish_ordered()?))
    }

    pub(crate) fn push_ordered(&mut self, chunk: &str) -> anyhow::Result<Vec<UnifiedDelta>> {
        self.buffer.push_str(chunk);
        // Emitter errors are possible only once a complete invoke closer is
        // buffered. Snapshot those comparatively rare pushes so reset() can
        // recover every byte that produced no returned deltas, without cloning
        // the buffer on the ordinary token hot path.
        let checkpoint = self
            .buffer
            .contains(self.spec.invoke_end.as_str())
            .then(|| {
                (
                    self.buffer.clone(),
                    self.uncommitted_block.clone(),
                    self.in_block,
                    self.in_reasoning,
                    self.accept_redundant_reasoning_start,
                    // Same flag `reset` was missing: without it a rollback leaves
                    // "resume reasoning after the call" armed, and the next visible
                    // answer is emitted as reasoning.
                    self.resume_reasoning,
                    self.suppress_normal_text,
                    self.next_index,
                )
            });
        match self.drain(false) {
            Ok(deltas) => Ok(deltas),
            Err(error) => {
                if let Some((
                    buffer,
                    uncommitted_block,
                    in_block,
                    in_reasoning,
                    accept_redundant_reasoning_start,
                    resume_reasoning,
                    suppress_normal_text,
                    next_index,
                )) = checkpoint
                {
                    self.buffer = buffer;
                    self.uncommitted_block = uncommitted_block;
                    self.in_block = in_block;
                    self.in_reasoning = in_reasoning;
                    self.accept_redundant_reasoning_start = accept_redundant_reasoning_start;
                    self.resume_reasoning = resume_reasoning;
                    self.suppress_normal_text = suppress_normal_text;
                    self.next_index = next_index;
                }
                Err(error)
            }
        }
    }

    pub(crate) fn finish_ordered(&mut self) -> anyhow::Result<Vec<UnifiedDelta>> {
        self.drain(true)
    }

    /// Clear one stream's scan state and return bytes not yet emitted.
    pub(crate) fn reset(&mut self) -> String {
        let mut pending = std::mem::take(&mut self.uncommitted_block);
        pending.push_str(&std::mem::take(&mut self.buffer));
        self.in_block = false;
        self.in_reasoning = self
            .reasoning
            .as_ref()
            .is_some_and(|reasoning| self.reasoning_enabled && reasoning.forced_start);
        self.accept_redundant_reasoning_start = self.in_reasoning;
        // Without this a reset mid-thought leaves "resume reasoning after the call"
        // armed, and the NEXT stream's first post-call answer is emitted as reasoning.
        self.resume_reasoning = false;
        self.suppress_normal_text = false;
        self.next_index = 0;
        pending
    }

    /// Position and length of the reasoning opener in the buffer, if configured.
    fn find_reasoning_start(&self) -> Option<(usize, usize)> {
        if !self.reasoning_enabled {
            return None;
        }
        let reasoning = self.reasoning.as_ref()?;
        let pos = self.buffer.find(reasoning.start)?;
        Some((pos, reasoning.start.len()))
    }

    /// Position and length of the earliest malformed close outside a block.
    fn find_orphan_marker(&self) -> Option<(usize, usize)> {
        let regular = find_first(&self.buffer, &self.spec.orphan_markers);
        let reasoning_close = self
            .reasoning
            .as_ref()
            .filter(|_| self.reasoning_enabled)
            .and_then(|reasoning| {
                self.buffer
                    .find(reasoning.end)
                    .map(|pos| (pos, reasoning.end.len()))
            });
        regular
            .into_iter()
            .chain(reasoning_close)
            .min_by_key(|(pos, _)| *pos)
    }

    /// Longest marker prefix held at the end of the current chunk.
    fn holdback_len(&self) -> usize {
        let regular = marker_prefix_suffix_len(
            &self.buffer,
            self.spec.holdback_markers.iter().map(String::as_str),
        );
        let reasoning = self
            .reasoning
            .as_ref()
            .filter(|_| self.reasoning_enabled)
            .map(|reasoning| {
                marker_prefix_suffix_len(&self.buffer, [reasoning.start, reasoning.end])
            })
            .unwrap_or_default();
        regular.max(reasoning)
    }

    /// Consume buffered reasoning up to its closer, or as far as is safe.
    ///
    /// Returns `true` once the reasoning span yielded (closer seen, tool opener
    /// reached, or promoted at EOF) and the caller should keep draining; `false`
    /// means more input is needed.
    fn drain_reasoning(
        &mut self,
        out: &mut Vec<UnifiedDelta>,
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
        let stray = stray_in_reasoning(&self.buffer, start, end, &self.spec.orphan_markers);

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
        let keep = if flush { 0 } else { self.holdback_len() };
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

    fn drain(&mut self, flush: bool) -> anyhow::Result<Vec<UnifiedDelta>> {
        let mut out: Vec<UnifiedDelta> = Vec::new();

        loop {
            // Reasoning is the innermost scope: while it is open, ONLY its
            // closer ends it, so a `<tool_call>` inside a thought stays part of
            // the thought instead of being executed.
            if self.in_reasoning {
                if self.drain_reasoning(&mut out, flush)? {
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
                        self.uncommitted_block.clear();
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
                        self.uncommitted_block.clear();
                        self.in_block = false;
                    }
                    break;
                };
                if start > 0 {
                    self.uncommitted_block.push_str(&self.buffer[..start]);
                    self.buffer.drain(..start);
                }
                let Some(end) = self.buffer.find(self.spec.invoke_end.as_str()) else {
                    if flush {
                        tracing::warn!(
                            why = %format!("{}_incomplete_invoke", self.spec.family),
                            "stream dropped incomplete invoke at EOF"
                        );
                        self.buffer.clear();
                        self.uncommitted_block.clear();
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
                    self.uncommitted_block.clear();
                    self.in_block = false;
                    self.suppress_normal_text = false;
                    continue;
                }
                let invoke = self.buffer[..end + self.spec.invoke_end.len()].to_string();
                let emitted = self.emitter.parse_invoke(&invoke, self.next_index)?;
                self.buffer.drain(..end + self.spec.invoke_end.len());
                if let Some(delta) = emitted {
                    out.push(UnifiedDelta::ToolCall(delta));
                    self.next_index += 1;
                    self.uncommitted_block.clear();
                    if self.spec.invoke_latch == InvokeLatch::IfEmitted {
                        self.suppress_normal_text = true;
                    }
                } else {
                    self.uncommitted_block.push_str(&invoke);
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
            if let Some((pos, len)) = self.find_orphan_marker() {
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
                        push_run(&mut out, Kind::Text, &self.buffer[..pos]);
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
                let keep = if flush { 0 } else { self.holdback_len() };
                let emit_len = self.buffer.len().saturating_sub(keep);
                if emit_len > 0 {
                    if !self.suppress_normal_text {
                        push_run(&mut out, Kind::Text, &self.buffer[..emit_len]);
                    }
                    self.buffer.drain(..emit_len);
                }
                break;
            };

            if start > 0 {
                if !self.suppress_normal_text {
                    push_run(&mut out, Kind::Text, &self.buffer[..start]);
                }
                self.buffer.drain(..start);
            }

            match marker {
                Marker::Block(blen) => {
                    self.uncommitted_block.push_str(&self.buffer[..blen]);
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
                    let invoke = self.buffer[..end + self.spec.invoke_end.len()].to_string();
                    let emitted = self.emitter.parse_invoke(&invoke, self.next_index)?;
                    self.buffer.drain(..end + self.spec.invoke_end.len());
                    if let Some(delta) = emitted {
                        tracing::warn!(
                            why = %format!("{}_bare_invoke_recovery", self.spec.family),
                            tool_index = delta.tool_index,
                            "stream recovered a complete bare invoke"
                        );
                        out.push(UnifiedDelta::ToolCall(delta));
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

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FailingEmitter;

    impl InvokeEmitter for FailingEmitter {
        fn parse_invoke(
            &self,
            _invoke: &str,
            _tool_index: usize,
        ) -> anyhow::Result<Option<ToolCallDelta>> {
            anyhow::bail!("synthetic emitter failure")
        }
    }

    fn failing_scanner() -> WrappedBlockScanner<FailingEmitter> {
        WrappedBlockScanner::new(
            WrappedBlockSpec {
                family: "test",
                block_starts: vec!["<tool>".to_string()],
                block_ends: vec!["</tool>".to_string()],
                invoke_start: "<call>".to_string(),
                invoke_end: "</call>".to_string(),
                orphan_markers: vec!["</tool>".to_string()],
                holdback_markers: vec!["<tool>".to_string(), "<call>".to_string()],
                bare_recovery_latch: BareRecoveryLatch::Clear,
                invoke_latch: InvokeLatch::IfEmitted,
                drop_invoke_crossing_block_end: false,
            },
            FailingEmitter,
        )
    }

    #[test]
    fn reset_recovers_the_complete_push_after_an_emitter_error() {
        let mut scanner = failing_scanner();
        let input = "prefix <tool><call>bad</call></tool>suffix";

        assert!(scanner.push_ordered(input).is_err());
        assert_eq!(scanner.reset(), input);
    }

    #[test]
    fn reset_recovers_a_block_opener_consumed_by_an_earlier_push() {
        let mut scanner = failing_scanner();
        assert_eq!(
            scanner.push_ordered("prefix <tool>  <call>bad").unwrap(),
            vec![UnifiedDelta::Text {
                text: "prefix ".to_string()
            }]
        );

        assert!(scanner.push_ordered("</call></tool>suffix").is_err());
        assert_eq!(scanner.reset(), "<tool>  <call>bad</call></tool>suffix");
    }
}
