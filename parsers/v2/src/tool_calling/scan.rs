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
//! It is scoped OUTSIDE tool blocks only: once a block or bare invoke is open,
//! a reasoning marker inside it is argument data, not a control token (`I7`).

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
/// The earliest malformed marker inside an OPEN reasoning span, as `(pos, len)`.
///
/// The candidates are the span's own OPENER (a second one is a duplicate, not
/// content) plus every orphan marker except the closer, which legitimately ends
/// the span. Being inside a thought must not turn markup into content (`I3`).
///
/// NATIVE PATH ONLY. An earlier revision of this comment claimed the guided drain
/// shared it; that stopped being true when the guided path moved to its own
/// vocabulary, and the two sets are not the same:
///
/// * native here — the span's opener plus `orphan_markers` (`</tool_call>`),
///   because tool OPENERS are structural on this path and are handled separately
///   as `InReasoning::ToolOpen`;
/// * guided — `holdback_markers` plus the opener, because guided decoding delivers
///   the call as JSON, so tool markup inside a thought is narration to strip, not
///   structure to enter.
///
/// The divergence is deliberate, but it is a divergence, so the parity it used to
/// guarantee by construction is now only guaranteed by
/// `guided_and_native_agree_on_the_same_reasoning_bytes` in `unified/qwen3.rs`.
/// That test is what stops the two drifting; this helper no longer can.
///
/// A second copy of this rule is how the two modes drift apart (`I6`).
pub(crate) fn reasoning_opener_len(
    start: &str,
    label: Option<&str>,
    after_start: &str,
    flush: bool,
) -> Option<usize> {
    let len = start.len();
    let Some(label) = label else {
        return Some(len);
    };
    if after_start.starts_with(label) {
        return Some(len + label.len());
    }
    if !flush && after_start.len() < label.len() && label.starts_with(after_start) {
        return None;
    }
    Some(len)
}

pub(crate) fn stray_in_reasoning(
    haystack: &str,
    opener: Option<(usize, usize)>,
    end: &str,
    orphan_markers: &[String],
) -> Option<(usize, usize)> {
    opener
        .into_iter()
        .chain(
            orphan_markers
                .iter()
                .map(String::as_str)
                .filter(|m| *m != end)
                .filter_map(|m| haystack.find(m).map(|pos| (pos, m.len()))),
        )
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

/// Byte offset just past the first complete top-level JSON value (object or
/// array) in `text`, skipping leading whitespace before it — or `None` if the
/// value is absent, not object/array-shaped, or still open (unterminated
/// string, unbalanced braces).
///
/// String contents are tracked with escape awareness, so a marker-looking
/// byte sequence inside a quoted value is never mistaken for structure (`I7`).
/// A family whose invoke body is `ARGUMENT_MARKER {json} CLOSE_MARKER` uses
/// this to find where the json ends BEFORE searching for `CLOSE_MARKER`, so a
/// copy of that marker embedded in a string argument cannot be mistaken for
/// the real one (`WrappedBlockSpec::invoke_scan`).
pub(crate) fn json_value_end(text: &str) -> Option<usize> {
    let mut chars = text.char_indices();
    let (_, first) = chars.find(|(_, c)| !c.is_whitespace())?;
    if first != '{' && first != '[' {
        return None;
    }
    // A stack of the closer each opener actually requires -- not a numeric
    // depth counter. A numeric counter treats `}` and `]` as interchangeable,
    // so a mismatched-type input like `{]` or `[}` reads as balanced at
    // depth 0 instead of being rejected. That's not just cosmetic: this
    // function's whole purpose (see the doc comment above) is to find the
    // structural end of the value so a literal closer search past it can't
    // be fooled by a copy of that closer embedded in an still-open string
    // (`I7`) -- a false "balanced" reading here can hand back a boundary
    // BEFORE a string that legitimately belongs to the same value, reopening
    // exactly that corruption.
    let mut stack = vec![if first == '{' { '}' } else { ']' }];
    let mut in_string = false;
    let mut escape = false;
    for (idx, c) in chars {
        if in_string {
            if escape {
                escape = false;
            } else if c == '\\' {
                escape = true;
            } else if c == '"' {
                in_string = false;
            }
            continue;
        }
        match c {
            '"' => in_string = true,
            '{' => stack.push('}'),
            '[' => stack.push(']'),
            '}' | ']' => {
                // A closer that doesn't match the innermost opener's type
                // can never be part of a well-formed JSON value from here --
                // there is no future input that fixes a type mismatch, so
                // this reading can never become balanced (unlike a plain
                // depth shortfall, which just needs more input).
                if stack.pop() != Some(c) {
                    return None;
                }
                if stack.is_empty() {
                    return Some(idx + c.len_utf8());
                }
            }
            _ => {}
        }
    }
    None
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
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum BareRecoveryLatch {
    /// Clear the latch: when the optional outer close is absent, later
    /// narration must still reach normal_text (MiniMax M2 semantics; a stray
    /// close that DOES follow is stripped by the orphan-close handling).
    Clear,
    #[default]
    /// Keep the latch set: trailing markup around the recovered invoke is
    /// dropped until an orphan close ends the markup context
    /// (MiniMax M3 / Qwen3-Coder / Kimi K2 semantics).
    Set,
}

/// When the in-block invoke latch engages after parsing a complete invoke.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum InvokeLatch {
    #[default]
    /// Only when the invoke produced a call delta (XML families).
    IfEmitted,
    /// Always — even when the invoke parsed to nothing (Kimi K2).
    Always,
}

/// Grammar-aware overrides for locating invokes when marker-only scanning is
/// insufficient.
///
/// Gemma 4 needs all three answers because its string-delimited values may
/// contain both `call:` and `<tool_call|>` as data. Keeping them on one hook
/// prevents opener, end, and holdback ownership from drifting apart.
#[derive(Clone, Copy)]
pub(crate) struct InvokeScan {
    /// End of the invoke that begins at byte zero, just past its closer. `flush`
    /// allows a family to recover a balanced body whose closer never arrived.
    /// `tool_index` is the invoke's position in this stream (0 for the first
    /// call) -- some best-effort recoveries are only justified for the first
    /// call in a response (see `kimi_invoke_end`'s doc comment).
    pub end: fn(text: &str, flush: bool, tool_index: usize) -> Option<usize>,
    /// Whether the `invoke_start` occurrence at `at` opens a real invoke.
    pub opens: fn(text: &str, at: usize) -> bool,
    /// Additional trailing bytes to retain while an opener is still arriving.
    pub holdback: fn(text: &str) -> usize,
}

/// Declarative description of one wrapped-invoke grammar.
#[derive(Default)]
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
    /// Grammar-aware invoke location; `None` uses the marker-only rules.
    pub invoke_scan: Option<InvokeScan>,
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
#[derive(Clone, Copy, Default)]
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
    /// Role label the tokenizer writes immediately after `start`, e.g. gemma4's
    /// `thought\n` in `<|channel>thought\n`. Structural, and stripped from the
    /// thought rather than emitted as reasoning text. `None` for families whose
    /// opener carries no label.
    pub start_label: Option<&'static str>,
}

/// Per-family hook: parse one complete invoke (opener..closer inclusive) into
/// a call delta. `None` means the invoke was malformed and is dropped.
///
/// `&mut self` (not `&self`): a family whose envelope names the call natively
/// (Kimi's `functions.NAME:IDX`) needs somewhere to remember that id per
/// `tool_index` for [`Self::tool_call_id`] to hand back later, and a plain
/// owned field is the ordinary way to do that — no `RefCell` needed, since
/// parsing and the later lookup never run at the same time.
pub(crate) trait InvokeEmitter {
    fn parse_invoke(
        &mut self,
        invoke: &str,
        tool_index: usize,
    ) -> anyhow::Result<Option<ToolCallDelta>>;

    /// The model-emitted id for a call already parsed at `tool_index`, when
    /// this family's grammar carries one. Mirrors
    /// [`crate::unified::UnifiedParser::tool_call_id`] one layer down, so a
    /// family overrides it in exactly one place regardless of whether it is
    /// reached through the tool-only or unified adapter.
    fn tool_call_id(&self, _tool_index: usize) -> Option<&str> {
        None
    }

    /// Clear any per-stream state before [`WrappedBlockScanner::reset`] hands
    /// the scanner back for a NEW stream at `tool_index` 0. A no-op default
    /// is correct for every stateless emitter; an emitter that tracks ids per
    /// `tool_index` (Kimi) must override this or a new stream's index 0
    /// would read back the abandoned stream's id.
    fn reset(&mut self) {}
}

/// First occurrence of any of `markers` in `text`: `(position, marker_len)`.
fn find_first(text: &str, markers: &[String]) -> Option<(usize, usize)> {
    markers
        .iter()
        .filter_map(|m| text.find(m.as_str()).map(|p| (p, m.len())))
        .min_by_key(|(p, _)| *p)
}

/// Same as [`find_first`], but a marker-looking byte sequence inside a
/// double-quoted JSON string is data, not structure, and is skipped (`I7`,
/// same escape-aware tracking as [`json_value_end`]). A raw `find_first`
/// over a buffer that still contains an invoke's own JSON argument body can
/// match a copy of a block-end marker the model happened to echo inside a
/// string argument (e.g. a shell command echoing the exact closing tag) and
/// mistake it for the real block boundary -- silently dropping a
/// well-formed invoke and corrupting the `tool_index` of every call after
/// it. Callers checking "did the block end before this invoke's own
/// resolved close" need this variant; callers scanning plain narration
/// (never inside an open JSON string) can keep using `find_first`.
pub(crate) fn find_first_outside_strings(text: &str, markers: &[String]) -> Option<(usize, usize)> {
    let mut in_string = false;
    let mut escape = false;
    for (idx, c) in text.char_indices() {
        if in_string {
            if escape {
                escape = false;
            } else if c == '\\' {
                escape = true;
            } else if c == '"' {
                in_string = false;
            }
            continue;
        }
        if c == '"' {
            in_string = true;
            continue;
        }
        if let Some((_, len)) = markers
            .iter()
            .find(|m| text[idx..].starts_with(m.as_str()))
            .map(|m| (idx, m.len()))
        {
            return Some((idx, len));
        }
    }
    None
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
    reasoning_enabled: bool,
    /// Effective request-scoped start, separate from the family declaration.
    reasoning_forced_start: bool,
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
            reasoning_forced_start: false,
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
    /// starting_state without rebuilding the scanner.
    pub(crate) fn with_reasoning(mut self, reasoning: ReasoningSpec) -> Self {
        self.reasoning_enabled = true;
        self.in_reasoning = reasoning.forced_start;
        self.reasoning_forced_start = reasoning.forced_start;
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

    pub(crate) fn invoke_start(&self) -> &str {
        &self.spec.invoke_start
    }

    pub(crate) fn invoke_scan(&self) -> Option<InvokeScan> {
        self.spec.invoke_scan
    }

    /// The model-emitted id for a call at `tool_index`, delegated to the
    /// family's emitter. See [`InvokeEmitter::tool_call_id`].
    pub(crate) fn tool_call_id(&self, tool_index: usize) -> Option<&str> {
        self.emitter.tool_call_id(tool_index)
    }

    /// Select whether this stream interprets reasoning markers, without
    /// rebuilding the scanner or cloning its tool schemas.
    pub(crate) fn set_reasoning_mode(&mut self, enabled: bool, forced_start: Option<bool>) {
        // A family with NO `ReasoningSpec` has no reasoning channel to turn on, so
        // "enabled" for it is simply false — not a caller error. This used to be a
        // `debug_assert!`, but `initialize_request` enables the channel for every
        // starting state except `Response`, so the first family registered without
        // reasoning would have aborted on EVERY request in debug and test builds.
        // `UNIFIED_PORTING.md` documents that registration as supported (it only
        // rules out `GuidedJson`), so the assert forbade a shape the porting guide
        // invites. The check could not tell "this family has no reasoning" from
        // "this scanner was built wrong" anyway; it only ever caught the former.
        let enabled = enabled && self.reasoning.is_some();
        self.reasoning_enabled = enabled;
        self.reasoning_forced_start = forced_start
            .or_else(|| self.reasoning.map(|reasoning| reasoning.forced_start))
            .unwrap_or(false);
        self.in_reasoning = enabled && self.reasoning_forced_start;
        self.accept_redundant_reasoning_start = self.in_reasoning;
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

    /// Clear one stream's scan state and return bytes not yet emitted.
    pub(crate) fn reset(&mut self) -> String {
        let mut pending = std::mem::take(&mut self.uncommitted_block);
        pending.push_str(&std::mem::take(&mut self.buffer));
        self.in_block = false;
        self.in_reasoning = self.reasoning_enabled && self.reasoning_forced_start;
        self.accept_redundant_reasoning_start = self.in_reasoning;
        // Without this a reset mid-thought leaves "resume reasoning after the call"
        // armed, and the NEXT stream's first post-call answer is emitted as reasoning.
        self.resume_reasoning = false;
        self.suppress_normal_text = false;
        self.next_index = 0;
        self.emitter.reset();
        pending
    }

    /// Whether one invoke closer also closes the surrounding block.
    fn invoke_closes_block(&self) -> bool {
        self.spec.block_ends.contains(&self.spec.invoke_end)
    }

    /// Find the next real invoke opener, applying the family hook when present.
    fn find_invoke_start(&self, text: &str) -> Option<usize> {
        let Some(scan) = self.spec.invoke_scan else {
            return text.find(self.spec.invoke_start.as_str());
        };
        let mut cursor = 0;
        while let Some(relative) = text[cursor..].find(self.spec.invoke_start.as_str()) {
            let at = cursor + relative;
            if (scan.opens)(text, at) {
                return Some(at);
            }
            cursor = at + self.spec.invoke_start.len();
        }
        None
    }

    /// Offset just past the closer of the invoke beginning at byte zero.
    fn invoke_end_at(&self, text: &str, flush: bool) -> Option<usize> {
        match self.spec.invoke_scan {
            Some(scan) => (scan.end)(text, flush, self.next_index),
            None => text
                .find(self.spec.invoke_end.as_str())
                .map(|position| position + self.spec.invoke_end.len()),
        }
    }

    /// Bytes a reasoning opener occupies at `at`, structural label included.
    fn opener_len_at(&self, at: usize, flush: bool) -> Option<usize> {
        let reasoning = self.reasoning.as_ref()?;
        reasoning_opener_len(
            reasoning.start,
            reasoning.start_label,
            &self.buffer[at + reasoning.start.len()..],
            flush,
        )
    }

    /// Position and length of the reasoning opener in the buffer, if configured.
    fn find_reasoning_start(&self, flush: bool) -> Option<(usize, usize)> {
        if !self.reasoning_enabled {
            return None;
        }
        let reasoning = self.reasoning.as_ref()?;
        let pos = self.buffer.find(reasoning.start)?;
        Some((pos, self.opener_len_at(pos, flush)?))
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
        let invoke = self
            .spec
            .invoke_scan
            .map(|scan| (scan.holdback)(&self.buffer))
            .unwrap_or_default();
        regular
            .max(reasoning)
            .max(self.pending_label_len())
            .max(invoke)
    }

    /// Retain a complete reasoning opener while its optional role label is only
    /// partially buffered.
    fn pending_label_len(&self) -> usize {
        let Some(reasoning) = self.reasoning.as_ref().filter(|_| self.reasoning_enabled) else {
            return 0;
        };
        let Some(label) = reasoning.start_label else {
            return 0;
        };
        let Some(at) = self.buffer.rfind(reasoning.start) else {
            return 0;
        };
        let rest = &self.buffer[at + reasoning.start.len()..];
        if rest.len() < label.len() && label.starts_with(rest) {
            self.buffer.len() - at
        } else {
            0
        }
    }

    /// Consume buffered reasoning up to its closer, or as far as is safe.
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
            .chain(self.find_invoke_start(&self.buffer))
            .min();
        let duplicate_start = self
            .buffer
            .find(start)
            .and_then(|pos| Some((pos, self.opener_len_at(pos, flush)?)));
        let stray = stray_in_reasoning(
            &self.buffer,
            duplicate_start,
            end,
            &self.spec.orphan_markers,
        );

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

    fn drain<S: EventSink + ?Sized>(&mut self, flush: bool, out: &mut S) -> anyhow::Result<()> {
        loop {
            // While a thought is open, `drain_reasoning` owns precedence: its
            // closer ends the span, a tool opener suspends it so the call can be
            // extracted, and a stray marker is stripped. The reverse nesting is
            // asymmetric: the in-block branch treats reasoning markers as arguments.
            if self.in_reasoning {
                if self.drain_reasoning(out, flush)? {
                    continue;
                }
                break;
            }

            if self.in_block {
                let invoke_start = self.find_invoke_start(&self.buffer);

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
                let Some(end) = self.invoke_end_at(&self.buffer, flush) else {
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
                //
                // `find_first_outside_strings`, not `find_first`: the search
                // runs over `self.buffer` from the invoke's start, which can
                // legitimately contain a block-end-looking byte sequence
                // inside the invoke's own JSON string argument (a shell
                // command echoing the closing tag). Safety only needs to
                // hold for any match this call actually returns, which the
                // `be_pos < end` filter below always bounds to inside the
                // invoke (the scan is strictly left-to-right, so `in_string`
                // at any returned `be_pos < end` depends only on bytes
                // already known to belong to this invoke) -- not for the
                // full buffer being scanned. A raw search there mistakes an
                // embedded copy for the real boundary and drops a
                // well-formed invoke -- also corrupting every later
                // `tool_index`, since `next_index` never advances for a
                // dropped call.
                if self.spec.drop_invoke_crossing_block_end
                    && let Some((be_pos, be_len)) =
                        find_first_outside_strings(&self.buffer, &self.spec.block_ends)
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
                let invoke = self.buffer[..end].to_string();
                // Emit BEFORE consuming. `parse_invoke` is fallible, and draining first
                // meant a failing emitter destroyed the invoke bytes: they were gone from
                // the buffer, so `reset` could no longer hand them back and the
                // documented recovery contract was false for the only shipped family.
                let emitted = self.emitter.parse_invoke(&invoke, self.next_index)?;
                self.buffer.drain(..end);
                if let Some(delta) = emitted {
                    out.push_call(delta);
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
                if self.invoke_closes_block() {
                    self.uncommitted_block.clear();
                    self.in_block = false;
                    self.suppress_normal_text = false;
                    self.in_reasoning = std::mem::take(&mut self.resume_reasoning);
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
                    .chain(self.find_invoke_start(&self.buffer))
                    // A reasoning opener counts as an opener here, so the
                    // matching closer in `<think>a</think>` is not mistaken for
                    // an orphan and stripped.
                    .chain(self.find_reasoning_start(flush).map(|(p, _)| p))
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
                self.find_invoke_start(&self.buffer)
                    .map(|pos| (pos, Marker::BareInvoke)),
                self.find_reasoning_start(flush)
                    .map(|(pos, len)| (pos, Marker::ReasoningStart(len))),
            ]);

            let Some((start, marker)) = next_marker else {
                // No marker present: emit buffered text, but hold back a trailing
                // partial marker (split across this chunk boundary) unless flushing.
                let keep = if flush { 0 } else { self.holdback_len() };
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
                    let Some(end) = self.invoke_end_at(&self.buffer, flush) else {
                        if flush {
                            tracing::warn!(
                                why = %format!("{}_incomplete_bare_invoke", self.spec.family),
                                "stream dropped incomplete bare invoke at EOF"
                            );
                            self.buffer.clear();
                        }
                        break;
                    };
                    let invoke = self.buffer[..end].to_string();
                    // Emit before consuming — same recovery contract as the wrapped site.
                    let emitted = self.emitter.parse_invoke(&invoke, self.next_index)?;
                    self.buffer.drain(..end);
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
            &mut self,
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
                invoke_scan: None,
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
    /// so `reset` can hand it back. Pre-fix this drained before the fallible call.
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
            s.reset(),
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
            s.reset(),
            "<function=boom></function>suffix",
            "the failed bare invoke and everything after it must remain recoverable"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::failing_scanner;
    use super::*;

    #[test]
    fn json_value_end_rejects_mismatched_bracket_types() {
        // A numeric depth counter treats `}`/`]` as interchangeable and
        // would read these as balanced at the position of the mismatched
        // closer -- e.g. `{]` closing at index 1. Neither can ever become
        // valid JSON no matter what follows, so both must be `None`, not a
        // false-positive boundary.
        assert_eq!(json_value_end("{]"), None);
        assert_eq!(json_value_end("[}"), None);
        assert_eq!(json_value_end(r#"{"a":[1,2}}"#), None);
        assert_eq!(json_value_end("[1,2}"), None);
        // Negative control: the same shapes with correctly matched closers
        // still resolve normally, proving the fix only rejects the type
        // mismatch, not depth-nesting itself.
        assert_eq!(json_value_end("{}"), Some(2));
        assert_eq!(json_value_end(r#"{"a":[1,2]}"#), Some(11));
    }

    #[test]
    fn find_first_outside_strings_skips_a_marker_inside_a_quoted_string() {
        let markers = vec!["CLOSE".to_string()];
        // A marker-looking sequence entirely inside a string is data, not
        // structure -- must be skipped in favor of the real one afterward.
        assert_eq!(
            find_first_outside_strings(r#"{"a":"has CLOSE inside"}CLOSE"#, &markers),
            Some((24, 5)),
        );
    }

    #[test]
    fn find_first_outside_strings_handles_an_escaped_quote_next_to_the_marker() {
        let markers = vec!["CLOSE".to_string()];
        // `\"` immediately before the embedded marker must not be misread as
        // the string's real closing quote (which would wrongly un-toggle
        // `in_string` one byte early and expose the embedded marker).
        assert_eq!(
            find_first_outside_strings(r#"{"a":"ends with \" then CLOSE inside"}CLOSE"#, &markers),
            Some((38, 5)),
        );
    }

    #[test]
    fn find_first_outside_strings_returns_the_earliest_of_multiple_markers() {
        let markers = vec!["BB".to_string(), "A".to_string()];
        // Earliest BYTE POSITION wins regardless of which marker string it
        // belongs to or the order markers are listed in.
        assert_eq!(
            find_first_outside_strings("xxAxxBBxx", &markers),
            Some((2, 1))
        );
    }

    #[test]
    fn find_first_outside_strings_on_empty_or_no_match_is_none() {
        let markers = vec!["CLOSE".to_string()];
        assert_eq!(find_first_outside_strings("", &markers), None);
        assert_eq!(find_first_outside_strings("no marker here", &markers), None);
        assert_eq!(
            find_first_outside_strings(r#""CLOSE only inside a string""#, &markers),
            None
        );
    }

    #[test]
    fn reset_recovers_the_complete_push_after_an_emitter_error() {
        let mut scanner = failing_scanner();
        let input = "prefix <tool_call><function=boom></function></tool_call>suffix";

        assert!(scanner.push_ordered(input).is_err());
        assert_eq!(
            scanner.reset(),
            "<tool_call><function=boom></function></tool_call>suffix"
        );
    }

    #[test]
    fn reset_recovers_a_block_opener_consumed_by_an_earlier_push() {
        let mut scanner = failing_scanner();
        assert_eq!(
            scanner
                .push_ordered("prefix <tool_call>  <function=boom")
                .unwrap(),
            vec![UnifiedParserEvent::Text("prefix ".to_string())]
        );

        assert!(
            scanner
                .push_ordered("</function></tool_call>suffix")
                .is_err()
        );
        assert_eq!(
            scanner.reset(),
            "<tool_call>  <function=boom</function></tool_call>suffix"
        );
    }

    /// A family with no `ReasoningSpec` must survive request setup.
    ///
    /// `ScannerUnified::initialize_request` enables the reasoning channel for every
    /// starting state except `Response`, so while this was a `debug_assert!` the
    /// first family registered without reasoning would have aborted on EVERY request
    /// in debug and test builds — a shape `UNIFIED_PORTING.md` explicitly documents
    /// as supported. Unreachable with today's registry (qwen3 and gemma4 both have
    /// reasoning), which is exactly why it needs a test rather than a reader.
    #[test]
    fn enabling_reasoning_on_a_family_without_it_is_a_no_op_not_a_panic() {
        let mut scanner = failing_scanner();
        assert!(
            scanner.reasoning.is_none(),
            "fixture must have no ReasoningSpec"
        );

        scanner.set_reasoning_mode(true, Some(false));
        assert!(!scanner.reasoning_enabled, "no channel to enable");
        assert!(!scanner.in_reasoning);

        // The prefilled-reasoning starting state must not latch either.
        scanner.set_reasoning_mode(true, Some(true));
        assert!(!scanner.reasoning_enabled);
        assert!(
            !scanner.in_reasoning,
            "cannot start inside a channel that does not exist"
        );
    }

    #[test]
    fn request_reasoning_start_does_not_overwrite_the_family_default() {
        let mut scanner = failing_scanner().with_reasoning(ReasoningSpec {
            start: "<think>",
            end: "</think>",
            forced_start: true,
            start_label: None,
            preserve_special_tokens: false,
        });

        scanner.set_reasoning_mode(true, Some(false));
        assert!(
            !scanner.in_reasoning,
            "request override must apply immediately"
        );
        assert!(
            scanner.reasoning.unwrap().forced_start,
            "request setup must not mutate the family declaration"
        );

        scanner.set_reasoning_mode(true, None);
        assert!(
            scanner.in_reasoning,
            "an unspecified request state must recover the family default"
        );
        scanner.reset();
        assert!(
            scanner.in_reasoning,
            "reset must preserve the effective default"
        );
    }
}
