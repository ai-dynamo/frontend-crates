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

use std::collections::HashSet;

use crate::tool_calling::traits::{ToolCallDelta, ToolParseResult};

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
    buffer: String,
    in_block: bool,
    suppress_normal_text: bool,
    next_index: usize,
}

impl<E: InvokeEmitter> WrappedBlockScanner<E> {
    pub(crate) fn new(spec: WrappedBlockSpec, emitter: E) -> Self {
        Self {
            spec,
            emitter,
            buffer: String::new(),
            in_block: false,
            suppress_normal_text: false,
            next_index: 0,
        }
    }

    pub(crate) fn push(&mut self, chunk: &str) -> anyhow::Result<ToolParseResult> {
        self.buffer.push_str(chunk);
        self.drain(false)
    }

    pub(crate) fn finish(&mut self) -> anyhow::Result<ToolParseResult> {
        self.drain(true)
    }

    fn drain(&mut self, flush: bool) -> anyhow::Result<ToolParseResult> {
        let mut out = ToolParseResult::default();

        loop {
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
                    out.calls.push(delta);
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
                    .min();
                if next_open.is_none_or(|open| pos < open) {
                    if !self.suppress_normal_text && pos > 0 {
                        out.normal_text.push_str(&self.buffer[..pos]);
                    }
                    self.buffer.drain(..pos + len);
                    self.suppress_normal_text = false;
                    continue;
                }
            }

            let block = find_first(&self.buffer, &self.spec.block_starts);
            let bare = self.buffer.find(self.spec.invoke_start.as_str());
            let next_marker = match (block, bare) {
                (Some((b, blen)), Some(f)) if b <= f => Some((b, Marker::Block(blen))),
                (Some(_), Some(f)) => Some((f, Marker::BareInvoke)),
                (Some((b, blen)), None) => Some((b, Marker::Block(blen))),
                (None, Some(f)) => Some((f, Marker::BareInvoke)),
                (None, None) => None,
            };

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
                        out.normal_text.push_str(&self.buffer[..emit_len]);
                    }
                    self.buffer.drain(..emit_len);
                }
                break;
            };

            if start > 0 {
                if !self.suppress_normal_text {
                    out.normal_text.push_str(&self.buffer[..start]);
                }
                self.buffer.drain(..start);
            }

            match marker {
                Marker::Block(blen) => {
                    self.buffer.drain(..blen);
                    self.in_block = true;
                    self.suppress_normal_text = true;
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
                        out.calls.push(delta);
                        self.next_index += 1;
                        self.suppress_normal_text =
                            self.spec.bare_recovery_latch == BareRecoveryLatch::Set;
                    }
                }
            }
        }

        Ok(out)
    }
}
