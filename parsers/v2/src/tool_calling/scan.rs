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
//! The two nestings are deliberately ASYMMETRIC, because tool structure
//! dominates: a tool call emitted INSIDE a thought is a real call, so it is
//! extracted and the thought splits around it; but a reasoning marker inside a
//! tool argument is argument data (`I7`), because the in-block scan never looks
//! for one. Burying a nested call would drop it AND leak its markup into the
//! reasoning payload (`I3`).

use std::collections::HashSet;

use crate::tool_calling::traits::{ToolCallDelta, ToolParseResult};
use crate::unified::UnifiedDelta;

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
}

/// The reasoning channel a unified scanner also owns.
///
/// Only meaningful outside tool blocks — see the module docs.
pub(crate) struct ReasoningSpec {
    /// Opener, e.g. `<think>`.
    pub start: String,
    /// Closer, e.g. `</think>`.
    pub end: String,
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

#[derive(Clone, Copy)]
enum Marker {
    /// A block opener; carries the matched token length.
    Block(usize),
    /// A bare invoke opener with no block wrapper (recovery path).
    BareInvoke,
    /// A reasoning opener; carries the matched token length.
    ReasoningStart(usize),
}

/// Append `text` as visible content, merging with a trailing text delta so one
/// drain does not emit a run of adjacent same-kind fragments.
fn push_text(out: &mut Vec<UnifiedDelta>, text: &str) {
    if text.is_empty() {
        return;
    }
    match out.last_mut() {
        Some(UnifiedDelta::Text { text: prev }) => prev.push_str(text),
        _ => out.push(UnifiedDelta::Text {
            text: text.to_string(),
        }),
    }
}

/// Append `text` as reasoning, merging with a trailing reasoning delta.
fn push_reasoning(out: &mut Vec<UnifiedDelta>, text: &str) {
    if text.is_empty() {
        return;
    }
    match out.last_mut() {
        Some(UnifiedDelta::Reasoning { text: prev }) => prev.push_str(text),
        _ => out.push(UnifiedDelta::Reasoning {
            text: text.to_string(),
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
        self.spec.holdback_markers.push(reasoning.start.clone());
        self.spec.holdback_markers.push(reasoning.end.clone());
        self.spec.orphan_markers.push(reasoning.end.clone());
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

    pub(crate) fn push_ordered(&mut self, chunk: &str) -> anyhow::Result<Vec<UnifiedDelta>> {
        self.buffer.push_str(chunk);
        self.drain(false)
    }

    pub(crate) fn finish_ordered(&mut self) -> anyhow::Result<Vec<UnifiedDelta>> {
        self.drain(true)
    }

    /// Position and length of the reasoning opener in the buffer, if configured.
    fn find_reasoning_start(&self) -> Option<(usize, usize)> {
        let reasoning = self.reasoning.as_ref()?;
        let pos = self.buffer.find(reasoning.start.as_str())?;
        Some((pos, reasoning.start.len()))
    }

    /// Consume buffered reasoning up to its closer, a nested tool opener, or as
    /// far as is safe.
    ///
    /// Returns `true` once the reasoning span yielded (closer seen, tool opener
    /// reached, or promoted at EOF) and the caller should keep draining; `false`
    /// means more input is needed.
    fn drain_reasoning(&mut self, out: &mut Vec<UnifiedDelta>, flush: bool) -> bool {
        let end = self
            .reasoning
            .as_ref()
            .expect("in_reasoning implies a reasoning spec")
            .end
            .clone();
        let end_pos = self.buffer.find(end.as_str());

        // Tool structure dominates reasoning: a tool call the model emits INSIDE
        // a thought is still a real call, so it is extracted and the thought
        // splits around it. Treating reasoning as the innermost scope instead
        // would bury the call and leak its markup into the reasoning payload,
        // violating `I3`. (The reverse nesting is NOT symmetric — a reasoning
        // marker inside a tool argument stays argument data, `I7`, because the
        // in-block branch never scans for it.)
        let tool_pos = find_first(&self.buffer, &self.spec.block_starts)
            .map(|(p, _)| p)
            .into_iter()
            .chain(self.buffer.find(self.spec.invoke_start.as_str()))
            .min();

        if let Some(tool) = tool_pos
            && end_pos.is_none_or(|e| tool < e)
        {
            // Emit the thought up to the call and suspend reasoning; the block
            // handler resumes it once the call closes.
            push_reasoning(out, &self.buffer[..tool]);
            self.buffer.drain(..tool);
            self.in_reasoning = false;
            self.resume_reasoning = true;
            return true;
        }

        if let Some(pos) = end_pos {
            push_reasoning(out, &self.buffer[..pos]);
            self.buffer.drain(..pos + end.len());
            self.in_reasoning = false;
            self.resume_reasoning = false;
            return true;
        }

        // No closer yet: stream what has arrived, holding back a closer OR a
        // nested tool opener split across this chunk boundary.
        let keep = if flush {
            0
        } else {
            marker_prefix_suffix_len(
                &self.buffer,
                std::iter::once(end.as_str())
                    .chain(self.spec.block_starts.iter().map(String::as_str))
                    .chain(std::iter::once(self.spec.invoke_start.as_str())),
            )
        };
        let emit_len = self.buffer.len().saturating_sub(keep);
        if emit_len > 0 {
            push_reasoning(out, &self.buffer[..emit_len]);
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
        false
    }

    /// Re-enter reasoning after a tool call that was nested inside a thought.
    fn resume_reasoning_after_tool(&mut self) {
        if self.resume_reasoning {
            self.resume_reasoning = false;
            self.in_reasoning = true;
        }
    }

    fn drain(&mut self, flush: bool) -> anyhow::Result<Vec<UnifiedDelta>> {
        let mut out: Vec<UnifiedDelta> = Vec::new();

        loop {
            // Reasoning is the innermost scope: while it is open, ONLY its
            // closer ends it, so a `<tool_call>` inside a thought stays part of
            // the thought instead of being executed.
            if self.in_reasoning {
                if self.drain_reasoning(&mut out, flush) {
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
                        self.resume_reasoning_after_tool();
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
                let invoke = self.buffer[..end + self.spec.invoke_end.len()].to_string();
                self.buffer.drain(..end + self.spec.invoke_end.len());
                if let Some(delta) = self.emitter.parse_invoke(&invoke, self.next_index)? {
                    out.push(UnifiedDelta::ToolCall(delta));
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
                        push_text(&mut out, &self.buffer[..pos]);
                    }
                    self.buffer.drain(..pos + len);
                    self.suppress_normal_text = false;
                    continue;
                }
            }

            // Earliest marker wins. Ties resolve in push order (block over bare
            // invoke), preserving the pre-unified tie-break.
            let mut candidates: Vec<(usize, Marker)> = Vec::new();
            if let Some((pos, len)) = find_first(&self.buffer, &self.spec.block_starts) {
                candidates.push((pos, Marker::Block(len)));
            }
            if let Some(pos) = self.buffer.find(self.spec.invoke_start.as_str()) {
                candidates.push((pos, Marker::BareInvoke));
            }
            if let Some((pos, len)) = self.find_reasoning_start() {
                candidates.push((pos, Marker::ReasoningStart(len)));
            }
            let next_marker = candidates.into_iter().min_by_key(|(pos, _)| *pos);

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
                        push_text(&mut out, &self.buffer[..emit_len]);
                    }
                    self.buffer.drain(..emit_len);
                }
                break;
            };

            if start > 0 {
                if !self.suppress_normal_text {
                    push_text(&mut out, &self.buffer[..start]);
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
                    let invoke = self.buffer[..end + self.spec.invoke_end.len()].to_string();
                    self.buffer.drain(..end + self.spec.invoke_end.len());
                    if let Some(delta) = self.emitter.parse_invoke(&invoke, self.next_index)? {
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
                        self.resume_reasoning_after_tool();
                    }
                }
            }
        }

        Ok(out)
    }
}
