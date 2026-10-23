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
//! * [`WrappedBlockScanner`] — the whole drain loop for the five families
//!   whose grammar is `BLOCK_START (INVOKE .. INVOKE_END)* BLOCK_END` with a
//!   bare-invoke back-off: qwen3_coder, minimax_m2, minimax_m3, kimi_k2, gemma4.
//!   Gemma uses [`InvokeScan`] to supply grammar-aware opener, end, and holdback
//!   decisions where static markers cannot distinguish structure from data.
//!   dsml (incremental invoke-header state), glm47 (identifier-anchored bare
//!   recovery) keep bespoke drains and share the primitives.
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
use crate::unified::{
    GuidedChannel, GuidedChannelState, GuidedTurnScope, Kind, MalformedToolCall,
    MalformedToolCallPolicy, ToolCallPolicy, UnifiedParserEvent,
};

/// Declarative channel envelope for grammars whose messages are framed as
/// `START ROLE [recipient_key NAME] CHANNEL name [constrain TOKEN] MESSAGE body TERM`
/// rather than as wrapped blocks (Beam). The scanner owns buffering, quote/control
/// protection, holdback, call indices and lifecycle; the family supplies only this
/// table, an [`InvokeBoundary::channel_body`] walk and an
/// [`InvokeEmitter::parse_channel_call`].
///
/// A header may be full (`START ROLE ..`), a continuation (`CHANNEL ..`), or begin
/// after the role the prompt already consumed (` to=NAME CHANNEL ..`).
pub(crate) struct ChannelEnvelope {
    pub start: &'static str,
    pub channel: &'static str,
    pub message: &'static str,
    pub role: &'static str,
    pub recipient_key: &'static str,
    pub constrain: &'static str,
    /// Tokens the grammar allows after `constrain`; any other token makes the header
    /// invalid (the model's own template only emits `xml`).
    pub constrain_tokens: &'static [&'static str],
    /// Longest valid recipient prefix of the text (0 when none).
    pub recipient_len: fn(&str) -> usize,
    pub channels: &'static [EnvelopeChannel],
    /// The only channel a tool header may name; any other is a malformed header.
    pub tool_channel: &'static str,
    /// A channel header longer than this without `message` is not a header.
    pub max_header: usize,
    /// Whether bytes outside every channel body (between messages, before the first
    /// header) are visible text. `false` for a grammar whose assistant turn is channel
    /// framing only (no implicit region): such bytes are framing debris and are dropped,
    /// never shown (goal 3/4), exactly as the builder's template parser treats them.
    pub prose_between_messages: bool,
    /// Whether a quotation in prose or in a message body keeps the control markers
    /// inside it literal (`"<|return|>"` stays text until the quote closes). `false`
    /// for a grammar whose markers are special tokens: a terminator ends its body
    /// wherever it stands, so an unbalanced quote can never hold a body open, swallow
    /// the next message, or stall streaming until the stream ends.
    pub quote_aware: bool,
    /// Control tokens that are debris BETWEEN messages: a terminator with no open
    /// channel is stripped there, while inside a body only that channel's own
    /// `terminators` are structural and the rest stay literal.
    pub orphan_controls: &'static [&'static str],
    /// Terminators that end the whole TURN for the guided reader (the answer's
    /// `return`): once one is seen, no guided payload can follow. A visible-channel
    /// terminator left out of this list only closes its message, so a tool-call
    /// handoff (`call`) standing alone before the payload is debris, not the end.
    pub turn_end: &'static [&'static str],
}

pub(crate) struct EnvelopeChannel {
    pub name: &'static str,
    pub reasoning: bool,
    /// Markers closing this channel's body. A trailing `start` is lookahead for the
    /// next header and is never consumed, including when it is the whole marker.
    pub terminators: &'static [&'static str],
    /// Whether a body that opens with `{` or `[` is read as a JSON answer: a terminator
    /// inside one of its open strings is data, not the end of the body. The reference
    /// serving layer keeps closers inside content literal under constrained decoding
    /// (`response_format` JSON answers), and the request does not tell the parser when
    /// that is, so the body's own shape decides.
    pub json_body: bool,
}

/// String-context tracker for a body read as a JSON answer. It survives the body's
/// streamed emissions, so a marker inside an open string is still data when the bytes
/// before it have already gone out.
#[derive(Clone, Copy, Default)]
pub(crate) struct JsonBodyState {
    /// `Some(true)` once the body's first non-blank byte is `{` or `[`, `Some(false)`
    /// once it is anything else, `None` while the body is still blank.
    shape: Option<bool>,
    in_string: bool,
    escaped: bool,
}

impl JsonBodyState {
    pub(crate) fn reset(&mut self) {
        *self = Self::default();
    }

    /// Advance over body bytes that were emitted (or are about to be).
    pub(crate) fn advance(&mut self, text: &str) {
        for c in text.chars() {
            if self.shape.is_none() {
                if matches!(c, ' ' | '\t' | '\n' | '\r') {
                    continue;
                }
                self.shape = Some(matches!(c, '{' | '['));
            }
            if self.shape != Some(true) {
                return;
            }
            if self.escaped {
                self.escaped = false;
            } else if self.in_string && c == '\\' {
                self.escaped = true;
            } else if c == '"' {
                self.in_string = !self.in_string;
            }
        }
    }

    /// Whether byte `at` of `buffer` (the unemitted rest of the body) falls inside an
    /// open JSON string of a JSON-shaped body.
    pub(crate) fn inside_string_at(mut self, buffer: &str, at: usize) -> bool {
        self.advance(&buffer[..at]);
        self.shape == Some(true) && self.in_string
    }
}

/// Verdict of a family's walk over one tool-channel body.
pub(crate) enum BodyScan {
    Wait,
    /// `body_end` is where the body stops; `consume` includes its terminator.
    Complete {
        body_end: usize,
        consume: usize,
    },
    /// The call can never complete; `discard` bytes are the bad region.
    Malformed {
        discard: usize,
        reason: &'static str,
    },
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum EnvMode {
    Idle,
    Channel(usize),
    Call,
}

enum HeaderLead {
    /// `Some(recipient)` for a tool header, `None` for a message header.
    Lead(Option<String>),
    NotHeader,
}

fn envelope_parse_prefix(env: &ChannelEnvelope, prefix: &str) -> HeaderLead {
    let t = prefix.strip_prefix(env.role).unwrap_or(prefix);
    let t = t.trim_start_matches(' ');
    if t.is_empty() {
        return HeaderLead::Lead(None);
    }
    // The recipient is whatever `recipient_key NAME` ends the lead; bytes glued in front
    // of it (`assistant\tto=NAME`, `assistantto=NAME`) are header debris, never prose:
    // the header is identified by its tail, as the builder's template regex does.
    let mut after_invalid = 0;
    for (k, _) in t.match_indices(env.recipient_key) {
        let start = k + env.recipient_key.len();
        if start < after_invalid {
            continue;
        }
        let r = &t[start..];
        let len = (env.recipient_len)(r);
        if !r.is_empty() && len == r.len() {
            return HeaderLead::Lead(Some(r.to_string()));
        }
        // Earlier suffixes share this invalid byte; never scan them again.
        after_invalid = start + len + 1;
    }
    HeaderLead::NotHeader
}

/// Whether `text` could still grow into a valid header lead.
fn envelope_prefix_possible(env: &ChannelEnvelope, text: &str) -> bool {
    let t = match text.strip_prefix(env.role) {
        Some(t) => t,
        None if env.role.starts_with(text) => return true,
        None => text,
    };
    let t = t.trim_start_matches(' ');
    if t.is_empty() || env.recipient_key.starts_with(t) {
        return true;
    }
    t.strip_prefix(env.recipient_key)
        .is_some_and(|r| (env.recipient_len)(r) == r.len())
}

/// The span of a complete tool header plus body starting at `at`, which prose
/// quote tracking must skip. Anything that is not a legal tool header is not owned.
///
/// The body is walked by the SAME boundary the drain installs for that recipient:
/// `boundary_for` is the emitter's per-recipient hook
/// ([`InvokeEmitter::channel_boundary`]), falling back to the static factory when it
/// declines. Walking a quoted call with a different framing than the real call would be
/// walked with (a factory boundary with no schema knowledge) read a JSON slot's embedded
/// closer as the end of the value, judged the complete call malformed and let the prose
/// lexer run through its markup — whose attribute quotes then closed the prose quote.
fn envelope_tool_span(
    env: &ChannelEnvelope,
    factory: Option<InvokeBoundaryFactory>,
    policy: ToolCallPolicy,
    boundary_for: &dyn Fn(&str) -> Option<Box<dyn InvokeBoundary>>,
    text: &str,
    at: usize,
    flush: bool,
) -> ProseGrammarSpan {
    let suffix = &text[at..];
    // A chunk ending inside `start` may still grow into a tool header: the prose
    // cursor must wait here, not walk into the header bytes and miss the header once
    // it completes (whose attribute quotes would then close the prose quote).
    let partial_start = !suffix.is_empty() && suffix.len() < env.start.len();
    if !flush && partial_start && env.start.starts_with(suffix) {
        return ProseGrammarSpan::Pending;
    }
    if !suffix.starts_with(env.start) {
        return ProseGrammarSpan::NotOwned;
    }
    let rest = &suffix[env.start.len()..];
    let Some(ci) = rest.find(env.channel) else {
        let lead = &rest[..rest.len() - marker_prefix_suffix_len(rest, [env.channel])];
        return if !flush && rest.len() <= env.max_header && envelope_prefix_possible(env, lead) {
            ProseGrammarSpan::Pending
        } else {
            ProseGrammarSpan::NotOwned
        };
    };
    let HeaderLead::Lead(Some(recipient)) = envelope_parse_prefix(env, &rest[..ci]) else {
        return ProseGrammarSpan::NotOwned;
    };
    let after = ci + env.channel.len();
    let Some(mi) = rest[after..].find(env.message) else {
        return if !flush && rest.len() - after <= env.max_header {
            ProseGrammarSpan::Pending
        } else {
            ProseGrammarSpan::NotOwned
        };
    };
    let legal = envelope_parse_channel(env, &rest[after..after + mi])
        .is_some_and(|index| env.channels[index].name == env.tool_channel);
    if !legal {
        return ProseGrammarSpan::NotOwned;
    }
    // The recipient's own walk first (schema-framed slots), the factory otherwise.
    let Some(mut boundary) =
        boundary_for(&recipient).or_else(|| factory.map(InvokeBoundaryFactory::create))
    else {
        return ProseGrammarSpan::NotOwned;
    };
    let header = env.start.len() + after + mi + env.message.len();
    boundary.configure(policy);
    // Only a COMPLETE call is grammar: a quoted header that never closes its call
    // stays prose, so its closing quote still counts.
    match boundary.channel_body(&suffix[header..], flush) {
        BodyScan::Wait if !flush => ProseGrammarSpan::Pending,
        BodyScan::Complete { consume, .. } => ProseGrammarSpan::Owned(header + consume),
        BodyScan::Wait | BodyScan::Malformed { .. } => ProseGrammarSpan::NotOwned,
    }
}

/// Length of the longest prefix of `rest` (bytes after `start`) that could still grow
/// into a header lead, on a char boundary.
fn envelope_lead_len(env: &ChannelEnvelope, rest: &str) -> usize {
    let mut best = 0;
    for (idx, c) in rest.char_indices() {
        let end = idx + c.len_utf8();
        if envelope_prefix_possible(env, &rest[..end]) {
            best = end;
        } else {
            break;
        }
    }
    best
}

/// One header resolved by the shared envelope grammar, for the guided reader.
struct EnvelopeHeader {
    start: usize,
    /// Bytes through `message`.
    len: usize,
    recipient: Option<String>,
    /// Index into `env.channels`, `None` for an unknown channel name.
    channel: Option<usize>,
    framed: bool,
}

/// How much of an UNFRAMED header the guided reader may honour, from the turn scope.
///
/// The native drain's header slot is open at turn start and between messages, and a
/// message BODY is cut only by its own terminators and `start`. The guided scope maps
/// onto that: an open thought is a body, so nothing unframed resolves inside it (a
/// bare `to=` there is narration, exactly as it is natively); a routed turn is between
/// messages, where a `channel` continuation is still a header but the words in front
/// of it are prose; only the unrouted turn may still carry the prompt-consumed lead.
#[derive(Clone, Copy, PartialEq, Eq)]
enum EnvelopeBare {
    /// `[role][ ][recipient_key NAME] CHANNEL …`: the prompt consumed the framing.
    Lead,
    /// `CHANNEL …` only; a lead in front of it is prose.
    Marker,
    /// Nothing unframed is a header; only `start` interrupts the body.
    None,
}

fn envelope_bare(state: GuidedChannelState) -> EnvelopeBare {
    match state.scope {
        GuidedTurnScope::Unrouted => EnvelopeBare::Lead,
        // A response prefill has routed the turn the same way visible content does:
        // a bare lead is prose, and only the channel marker itself opens a header.
        GuidedTurnScope::Routed | GuidedTurnScope::ResponsePrefill => EnvelopeBare::Marker,
        GuidedTurnScope::InReasoning => EnvelopeBare::None,
    }
}

/// The bare lead directly before a `channel` marker: `[role][ ]recipient_key NAME`, or
/// the role alone, at a word boundary. `(start, recipient)`; `None` when the bytes
/// before the marker are not a lead (so the header starts at the marker).
///
/// The role-only form is the template's own `assistant<|channel|>…` continuation
/// after a prompt that consumed `<|start|>`; without it the role word leaked as text
/// in front of the thought. Separator spaces are absorbed only together with a lead
/// word, never alone: `Sure. <|channel|>` keeps its trailing space as prose.
fn envelope_bare_lead(env: &ChannelEnvelope, before: &str) -> Option<(usize, Option<String>)> {
    let (mut p, recipient) = match before.find(env.recipient_key).and_then(|k| {
        match envelope_parse_prefix(env, &before[k..]) {
            HeaderLead::Lead(Some(r)) => Some((k, r)),
            _ => None,
        }
    }) {
        Some((k, r)) => (k, Some(r)),
        _ => (before.len(), None),
    };
    while p > 0 && before.as_bytes()[p - 1] == b' ' {
        p -= 1;
    }
    if before[..p].ends_with(env.role) {
        p -= env.role.len();
    } else if recipient.is_none() {
        return None;
    }
    let boundary = p == 0 || before[..p].ends_with(|c: char| c.is_whitespace() || c == '>');
    boundary.then_some((p, recipient))
}

/// The first complete header at or after `floor`, resolved by the SAME lead and
/// channel rules the native drain applies. Nothing before `floor` is consulted: the
/// bytes there belong to a header or native body already resolved, so neither a
/// `channel` marker nor a lead may be borrowed from them. An unframed header is
/// honoured as far as the turn scope allows ([`EnvelopeBare`]); otherwise its words
/// are prose and only the fixed markers are structural.
fn envelope_header_after(
    env: &ChannelEnvelope,
    haystack: &str,
    floor: usize,
    bare: EnvelopeBare,
) -> Option<EnvelopeHeader> {
    let hay = &haystack[floor..];
    let mut search = 0;
    while let Some(rel) = hay[search..].find(env.message) {
        let msg = search + rel;
        search = msg + env.message.len();
        let Some(ci) = hay[..msg].rfind(env.channel) else {
            continue;
        };
        let after = ci + env.channel.len();
        if msg - after > env.max_header {
            continue;
        }
        let channel = envelope_parse_channel(env, &hay[after..msg]);
        let before = &hay[..ci];
        let framed = before.rfind(env.start).filter(|&si| {
            matches!(
                envelope_parse_prefix(env, &before[si + env.start.len()..]),
                HeaderLead::Lead(_)
            )
        });
        let (start, recipient) = if let Some(si) = framed {
            match envelope_parse_prefix(env, &before[si + env.start.len()..]) {
                HeaderLead::Lead(r) => (si, r),
                HeaderLead::NotHeader => unreachable!("filtered above"),
            }
        } else {
            match bare {
                EnvelopeBare::None => continue,
                EnvelopeBare::Marker => (ci, None),
                EnvelopeBare::Lead => envelope_bare_lead(env, before).unwrap_or((ci, None)),
            }
        };
        return Some(EnvelopeHeader {
            start: floor + start,
            len: search - start,
            recipient,
            channel,
            framed: framed.is_some(),
        });
    }
    None
}

/// Who owns which bytes of a guided haystack, resolved LEFT TO RIGHT.
///
/// Header-shaped bytes inside a native parameter value are data, not headers: a model
/// may legitimately pass a whole framed message as a string argument. Resolving every
/// header-shaped run independently read that value as a thought (or a content switch)
/// and released the outer call's header in front of it as text. So ownership is
/// settled from the earliest tool header onwards, with the family's own
/// [`InvokeBoundary::channel_body`] — the same walk the native drain uses — deciding
/// where its body ends: a complete call owns exactly its span; a body still arriving
/// owns everything behind its header to the end of the haystack; a body that can
/// never complete (or is cut off at EOF) owns only its header; that header alone is
/// stripped and the body bytes are read again as guided narration. This guided
/// recovery does NOT reuse the native drain's malformed discard span.
///
/// Every guided hook of the envelope consults ONE of these over the same bytes, so
/// open, close, turn end, transition, routing, stray and holdback cannot disagree
/// about what is a header and what is argument data.
struct EnvelopeOwnership {
    /// Headers outside every native body, in order.
    headers: Vec<EnvelopeHeader>,
    /// Complete native calls, `(start, end)`, in order.
    owned: Vec<(usize, usize)>,
    /// A tool header whose native body is still arriving owns from here to the end.
    pending_from: Option<usize>,
}

impl EnvelopeOwnership {
    /// Whether the byte at `at` belongs to no native body, complete or arriving.
    fn free(&self, at: usize) -> bool {
        self.pending_from.is_none_or(|from| at < from)
            && !self
                .owned
                .iter()
                .any(|&(start, end)| start <= at && at < end)
    }

    /// The end of the last complete native body, below which no lead may reach.
    fn floor(&self) -> usize {
        self.owned.last().map_or(0, |&(_, end)| end)
    }

    /// Bytes a stray tool header at `h` owns: its whole native call when complete,
    /// its header alone when the body is malformed or absent, `None` while the body
    /// is still arriving (what follows decides whether it is narration or a call).
    fn stray_len(&self, h: &EnvelopeHeader) -> Option<usize> {
        if self.pending_from == Some(h.start) {
            return None;
        }
        Some(
            self.owned
                .iter()
                .find(|&&(start, _)| start == h.start)
                .map_or(h.len, |&(start, end)| end - start),
        )
    }
}

fn envelope_resolve(
    env: &ChannelEnvelope,
    factory: Option<InvokeBoundaryFactory>,
    haystack: &str,
    flush: bool,
    bare: EnvelopeBare,
) -> EnvelopeOwnership {
    let mut resolved = EnvelopeOwnership {
        headers: Vec::new(),
        owned: Vec::new(),
        pending_from: None,
    };
    let mut floor = 0;
    while let Some(h) = envelope_header_after(env, haystack, floor, bare) {
        let body_at = h.start + h.len;
        floor = body_at;
        let body = &haystack[body_at..];
        // A tool header wraps a payload the backend delivers as JSON; only native
        // parameter markup after a LEGAL tool header has a body to walk.
        let legal = h.recipient.is_some()
            && h.channel
                .is_some_and(|i| env.channels[i].name == env.tool_channel);
        let payload = body.trim_start().starts_with(['{', '[']);
        if legal && !payload {
            let verdict = match factory {
                Some(factory) => factory.create().channel_body(body, flush),
                None if body.trim().is_empty() => BodyScan::Wait,
                None => BodyScan::Malformed {
                    discard: 0,
                    reason: "no body walker",
                },
            };
            match verdict {
                BodyScan::Complete { consume, .. } => {
                    resolved.owned.push((h.start, body_at + consume));
                    floor = body_at + consume;
                }
                BodyScan::Wait if !flush => {
                    resolved.pending_from = Some(h.start);
                    resolved.headers.push(h);
                    break;
                }
                BodyScan::Wait | BodyScan::Malformed { .. } => {}
            }
        }
        resolved.headers.push(h);
    }
    resolved
}

/// An UNFRAMED header still arriving at the tail: `channel` without `message` within
/// `max_header`, outside every native body, with whatever bare lead the scope lets it
/// carry (the SAME rule [`envelope_header_after`] resolves it by, so the bytes held
/// or discarded are exactly the bytes the header would own). `(start, channel_at)`.
fn envelope_unframed_tail(
    env: &ChannelEnvelope,
    haystack: &str,
    resolved: &EnvelopeOwnership,
    bare: EnvelopeBare,
) -> Option<(usize, usize)> {
    if bare == EnvelopeBare::None {
        return None;
    }
    let ci = haystack.rfind(env.channel)?;
    if !resolved.free(ci)
        || haystack[ci..].contains(env.message)
        || haystack.len() - ci - env.channel.len() > env.max_header
    {
        return None;
    }
    let floor = resolved.floor();
    let start = match bare {
        EnvelopeBare::Lead => {
            envelope_bare_lead(env, &haystack[floor..ci]).map_or(ci, |(p, _)| floor + p)
        }
        _ => ci,
    };
    Some((start, ci))
}

fn envelope_header_is_reasoning(env: &ChannelEnvelope, h: &EnvelopeHeader) -> bool {
    h.recipient.is_none() && h.channel.is_some_and(|i| env.channels[i].reasoning)
}

fn envelope_header_is_content(env: &ChannelEnvelope, h: &EnvelopeHeader) -> bool {
    h.recipient.is_none() && h.channel.is_some_and(|i| !env.channels[i].reasoning)
}

/// Guided reader: earliest header that OPENS a thought.
pub(crate) fn envelope_guided_open(
    env: &ChannelEnvelope,
    factory: Option<InvokeBoundaryFactory>,
    haystack: &str,
    flush: bool,
    state: GuidedChannelState,
) -> Option<(usize, usize)> {
    envelope_resolve(env, factory, haystack, flush, envelope_bare(state))
        .headers
        .into_iter()
        .find(|h| envelope_header_is_reasoning(env, h))
        .map(|h| (h.start, h.len))
}

/// Guided reader: earliest header that routes to VISIBLE content.
pub(crate) fn envelope_guided_transition(
    env: &ChannelEnvelope,
    factory: Option<InvokeBoundaryFactory>,
    haystack: &str,
    flush: bool,
    state: GuidedChannelState,
) -> Option<(usize, usize)> {
    envelope_resolve(env, factory, haystack, flush, envelope_bare(state))
        .headers
        .into_iter()
        .find(|h| envelope_header_is_content(env, h))
        .map(|h| (h.start, h.len))
}

/// Guided reader: earliest header that names a channel at all.
pub(crate) fn envelope_guided_routing(
    env: &ChannelEnvelope,
    factory: Option<InvokeBoundaryFactory>,
    haystack: &str,
    flush: bool,
    state: GuidedChannelState,
) -> Option<(usize, usize)> {
    envelope_resolve(env, factory, haystack, flush, envelope_bare(state))
        .headers
        .into_iter()
        .find(|h| h.recipient.is_some() || h.framed || h.channel.is_some())
        .map(|h| (h.start, h.len))
}

/// Earliest occurrence of any of `markers` outside every native body.
fn envelope_free_marker<'a>(
    haystack: &str,
    resolved: &EnvelopeOwnership,
    markers: impl Iterator<Item = &'a str>,
    json: Option<JsonBodyState>,
) -> Option<(usize, usize)> {
    markers
        .filter_map(|m| {
            haystack
                .match_indices(m)
                .map(|(at, _)| at)
                .find(|&at| {
                    resolved.free(at)
                        && json.is_none_or(|state| !state.inside_string_at(haystack, at))
                })
                .map(|at| (at, m.len()))
        })
        .min_by_key(|(at, _)| *at)
}

/// Guided reader: earliest reasoning CLOSER — the reasoning channels' terminators —
/// outside every native tool body (inside one it is parameter data).
pub(crate) fn envelope_guided_close(
    env: &ChannelEnvelope,
    factory: Option<InvokeBoundaryFactory>,
    haystack: &str,
    flush: bool,
    state: GuidedChannelState,
) -> Option<(usize, usize)> {
    let resolved = envelope_resolve(env, factory, haystack, flush, envelope_bare(state));
    envelope_free_marker(
        haystack,
        &resolved,
        env.channels
            .iter()
            .filter(|c| c.reasoning)
            .flat_map(|c| c.terminators.iter().copied()),
        state.content_routed.then_some(state.content_json),
    )
    .map(|(at, len)| (at, envelope_close_len(env, &haystack[at..at + len])))
}

/// A closer can include the next header's start marker as lookahead.
fn envelope_close_len(env: &ChannelEnvelope, marker: &str) -> usize {
    marker.strip_suffix(env.start).unwrap_or(marker).len()
}

/// Guided reader: earliest TURN-END marker ([`ChannelEnvelope::turn_end`]) outside
/// every native tool body.
pub(crate) fn envelope_guided_turn_end(
    env: &ChannelEnvelope,
    factory: Option<InvokeBoundaryFactory>,
    haystack: &str,
    flush: bool,
    state: GuidedChannelState,
) -> Option<(usize, usize)> {
    let resolved = envelope_resolve(env, factory, haystack, flush, envelope_bare(state));
    envelope_free_marker(
        haystack,
        &resolved,
        env.turn_end.iter().copied(),
        state.content_routed.then_some(state.content_json),
    )
}

/// What an unfinished header lead at a `start` marker turns out to be.
enum LeadVerdict {
    /// Still growing: hold every byte from the marker.
    Pending,
    /// Can never become a header: strip this many bytes from the marker.
    Strip(usize),
}

/// Judge the `start` marker at `at` that has NOT resolved into a complete header.
///
/// Mirrors the native idle drain: a lead that is still possible waits; a lead that
/// stops at a payload, a control marker or EOF is framing debris and is stripped
/// through its recipient key (the recipient itself stays text, as a prefix-form invoke
/// opener leaves its NAME); anything else was never a header and only the marker goes.
fn envelope_lead_verdict(
    env: &ChannelEnvelope,
    haystack: &str,
    at: usize,
    flush: bool,
    competitors: &[&str],
) -> LeadVerdict {
    let full = &haystack[at + env.start.len()..];
    // A later `start` marker ends this header for certain, whatever it was.
    let next_start = full.find(env.start);
    let rest = &full[..next_start.unwrap_or(full.len())];
    let settled = flush || next_start.is_some();
    if let Some(ci) = rest.find(env.channel) {
        if matches!(
            envelope_parse_prefix(env, &rest[..ci]),
            HeaderLead::NotHeader
        ) {
            return LeadVerdict::Strip(env.start.len());
        }
        let after = ci + env.channel.len();
        if !settled && rest.len() - after <= env.max_header {
            return LeadVerdict::Pending;
        }
        // A header that never reached `message`: strip through the channel marker and
        // whatever channel bytes follow, up to the next competing marker.
        let tail = &rest[after..];
        let bound = competitors
            .iter()
            .filter_map(|m| tail.find(m))
            .min()
            .unwrap_or(tail.len());
        return LeadVerdict::Strip(env.start.len() + after + bound);
    }
    // A trailing partial of ANY fixed marker — the `channel` marker that would continue
    // the header, or a control that would end it (`to=weather<|e` before `nd|>`) — is
    // not yet judged: the whole lead stays held until that marker resolves, otherwise
    // the lead would be stripped before the control that decides it has arrived.
    let hold = if settled {
        0
    } else {
        marker_prefix_suffix_len(rest, competitors.iter().copied())
    };
    let lead_len = envelope_lead_len(env, &rest[..rest.len() - hold]);
    if lead_len + hold == rest.len() && !settled {
        return LeadVerdict::Pending;
    }
    // Once the lead cannot grow into a header, preserve a guided JSON payload
    // even though its opening bracket was also a possible recipient character.
    let lead_len = rest[..lead_len].find(['{', '[']).unwrap_or(lead_len);
    let lead = &rest[..lead_len];
    let next = &rest[lead_len..];
    let decided = next.is_empty()
        || next.starts_with(['{', '['])
        || competitors.iter().any(|m| next.starts_with(m));
    if !decided {
        return LeadVerdict::Strip(env.start.len());
    }
    // Framing through the recipient key goes; a recipient NAME stays text, as the
    // name after a prefix-form invoke opener does for the marker-pair grammars.
    let through = lead
        .find(env.recipient_key)
        .map_or(lead_len, |k| k + env.recipient_key.len());
    LeadVerdict::Strip(env.start.len() + through)
}

/// Guided reader: earliest framing run that is neither a thought nor a channel switch.
///
/// A tool header wraps a payload the backend delivers as JSON, so it is stripped WHOLE;
/// when native parameter markup follows it instead, the family's own body walk decides
/// where that call ends and the whole native call is stripped, exactly as the prose
/// path does through [`envelope_tool_span`]. A header to an unknown channel and a
/// `start` marker whose header can no longer complete are debris too.
pub(crate) fn envelope_guided_stray(
    env: &ChannelEnvelope,
    factory: Option<InvokeBoundaryFactory>,
    haystack: &str,
    flush: bool,
    state: GuidedChannelState,
) -> Option<(usize, usize)> {
    let bare = envelope_bare(state);
    let resolved = envelope_resolve(env, factory, haystack, flush, bare);
    let stray_header = resolved
        .headers
        .iter()
        .find(|h| !envelope_header_is_reasoning(env, h) && !envelope_header_is_content(env, h))
        .map(|h| (h.start, resolved.stray_len(h)));
    let competitors = envelope_competitors(env);
    let stray_start = haystack
        .match_indices(env.start)
        .filter(|(at, _)| resolved.free(*at))
        .filter(|(at, _)| !resolved.headers.iter().any(|h| h.framed && h.start == *at))
        .find_map(
            |(at, _)| match envelope_lead_verdict(env, haystack, at, flush, &competitors) {
                LeadVerdict::Strip(len) => Some((at, Some(len))),
                LeadVerdict::Pending => None,
            },
        );
    // At EOF an UNFRAMED header that never reached `message` is debris too, exactly
    // like its framed twin above: the native idle drain emits nothing for it. While
    // the stream is open `envelope_guided_holdback` retains the same bytes instead.
    // Only a run the scope already recognises as a header-in-progress qualifies —
    // the `channel` marker with its lead — so a role word or `to=` in prose, and
    // anything inside an open body or a native parameter, stays untouched.
    let stray_unframed = flush
        .then(|| envelope_unframed_tail(env, haystack, &resolved, bare))
        .flatten()
        .map(|(start, ci)| {
            let tail = &haystack[ci + env.channel.len()..];
            let bound = competitors
                .iter()
                .filter_map(|m| tail.find(m))
                .min()
                .unwrap_or(tail.len());
            (start, Some(ci + env.channel.len() + bound - start))
        });
    [stray_header, stray_start, stray_unframed]
        .into_iter()
        .flatten()
        .min_by_key(|(at, _)| *at)
        .and_then(|(at, len)| len.map(|len| (at, len)))
}

/// Every fixed marker of the envelope: the terminator competition for a header that
/// never completes, and the split-marker set.
fn envelope_competitors(env: &ChannelEnvelope) -> Vec<&'static str> {
    std::iter::once(env.start)
        .chain([env.channel, env.message])
        .chain(
            env.channels
                .iter()
                .flat_map(|c| c.terminators.iter().copied()),
        )
        .chain(env.orphan_controls.iter().copied())
        .collect()
}

/// Guided reader: trailing bytes a header still arriving occupies.
///
/// OWNERSHIP FIRST: a tool header whose native body is still arriving owns every
/// byte from its first one, whatever header-shaped bytes that body contains. Only
/// outside every native body is a later `start` marker, an unframed `channel`
/// marker or a bare lead judged on its own.
pub(crate) fn envelope_guided_holdback(
    env: &ChannelEnvelope,
    factory: Option<InvokeBoundaryFactory>,
    haystack: &str,
    state: GuidedChannelState,
) -> usize {
    let bare = envelope_bare(state);
    let resolved = envelope_resolve(env, factory, haystack, false, bare);
    // A complete tool header whose native body is still arriving, or that nothing has
    // followed yet: what comes next decides whether it is narration or a call.
    if let Some(at) = resolved.pending_from {
        return haystack.len() - at;
    }
    let competitors = envelope_competitors(env);
    // A framed header still arriving.
    if let Some(at) = haystack
        .rmatch_indices(env.start)
        .map(|(at, _)| at)
        .find(|&at| resolved.free(at))
        && !resolved.headers.iter().any(|h| h.framed && h.start == at)
        && matches!(
            envelope_lead_verdict(env, haystack, at, false, &competitors),
            LeadVerdict::Pending
        )
    {
        return haystack.len() - at;
    }
    // An unframed header still arriving: `channel` without `message`, with whatever
    // bare lead the scope lets it carry.
    if let Some((start, _)) = envelope_unframed_tail(env, haystack, &resolved, bare) {
        return haystack.len() - start;
    }
    if bare != EnvelopeBare::Lead {
        return 0;
    }
    // A bare lead that has not reached its `channel` marker yet — possibly with the
    // first bytes of that marker already behind it (`assistant<|chan`), which are not
    // part of the lead and are judged separately as a split marker.
    let lead_end = haystack.len() - marker_prefix_suffix_len(haystack, [env.channel]);
    let floor = lead_end
        .saturating_sub(env.max_header)
        .max(resolved.floor());
    haystack[..lead_end]
        .char_indices()
        .map(|(p, _)| p)
        .filter(|&p| p >= floor)
        .filter(|&p| p == 0 || haystack[..p].ends_with(|c: char| c.is_whitespace() || c == '>'))
        .find(|&p| {
            let t = haystack[p..lead_end].trim_start_matches(' ');
            !t.is_empty() && envelope_prefix_possible(env, t)
        })
        .map_or(0, |p| haystack.len() - p)
}

/// Channel index for `name`, optionally followed by ` constrain TOKEN`.
/// Offset of the first control marker (`<|start|>`, `<|channel|>` or an orphan
/// control) inside a header's label region when it comes BEFORE any `<|message|>`:
/// the header is broken there. `None` when the label is intact so far.
fn envelope_label_cut(env: &ChannelEnvelope, label_region: &str) -> Option<usize> {
    let message_at = label_region.find(env.message).unwrap_or(usize::MAX);
    [env.start, env.channel]
        .into_iter()
        .chain(env.orphan_controls.iter().copied())
        // `<|message|>` ends the label and `<|constrain|>` is part of a legal label.
        .filter(|m| *m != env.message && *m != env.constrain)
        .filter_map(|m| label_region.find(m))
        .min()
        .filter(|cut| *cut < message_at)
}

fn envelope_parse_channel(env: &ChannelEnvelope, text: &str) -> Option<usize> {
    let (name, rest) = match text.split_once(' ') {
        Some((name, rest)) => (name, Some(rest)),
        None => (text, None),
    };
    if let Some(rest) = rest {
        let token = rest.strip_prefix(env.constrain)?;
        if token.is_empty() || !token.chars().all(|c| c.is_ascii_alphanumeric()) {
            return None;
        }
        // Spec-bound: the template emits exactly `<|constrain|>xml`; a different token is not
        // a tool header (goal 1 — follow the model's own grammar, as the reference does).
        if !env.constrain_tokens.is_empty() && !env.constrain_tokens.contains(&token) {
            return None;
        }
    }
    env.channels.iter().position(|c| c.name == name)
}

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

/// Resumable JSON quote ownership for boundary scans and whitespace projection.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct JsonStringState {
    inside: bool,
    escaped: bool,
}

impl JsonStringState {
    /// True when this character belongs to a quoted string, including its quotes.
    pub(crate) fn advance(&mut self, character: char) -> bool {
        if self.inside {
            if self.escaped {
                self.escaped = false;
            } else if character == '\\' {
                self.escaped = true;
            } else if character == '"' {
                self.inside = false;
            }
            true
        } else if character == '"' {
            self.inside = true;
            true
        } else {
            false
        }
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct JsonValueBoundary {
    quoted: JsonStringState,
    closers: Vec<char>,
    started: bool,
    root_string: bool,
    complete: bool,
    invalid: bool,
}

impl JsonValueBoundary {
    pub(crate) fn inside_string(&self) -> bool {
        self.quoted.inside
    }

    /// True only on the character completing the first root value. JSON syntax
    /// validation remains with serde; matching bracket types cannot be repaired
    /// by a later suffix, and quoted delimiters never settle a structural boundary.
    pub(crate) fn advance(&mut self, character: char) -> bool {
        if self.complete || self.invalid {
            self.quoted.advance(character);
            return false;
        }
        if !self.started {
            if character.is_whitespace() {
                return false;
            }
            self.started = true;
            match character {
                '{' => self.closers.push('}'),
                '[' => self.closers.push(']'),
                '"' => {
                    self.root_string = true;
                    self.quoted.advance(character);
                }
                _ => self.invalid = true,
            }
            return false;
        }
        if self.quoted.advance(character) {
            if self.root_string && !self.quoted.inside {
                self.complete = true;
                return true;
            }
            return false;
        }
        match character {
            '{' => self.closers.push('}'),
            '[' => self.closers.push(']'),
            '}' | ']' => {
                if self.closers.pop() != Some(character) {
                    self.invalid = true;
                } else if self.closers.is_empty() {
                    self.complete = true;
                    return true;
                }
            }
            _ => {}
        }
        false
    }
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
    json_container_end(text.char_indices())
}

fn json_container_end(mut chars: impl Iterator<Item = (usize, char)>) -> Option<usize> {
    let (_, first) = chars.find(|(_, ch)| !ch.is_whitespace())?;
    if !matches!(first, '{' | '[') {
        return None;
    }
    let mut boundary = JsonValueBoundary::default();
    boundary.advance(first);
    for (at, character) in chars {
        if boundary.advance(character) {
            return Some(at + character.len_utf8());
        }
        if boundary.invalid {
            return None;
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
    // Only reorder keys. Parsing values through Value would round large/fractional
    // JSON numbers that the coercer deliberately preserved as raw text.
    let Ok(obj) = serde_json::from_str::<
        std::collections::BTreeMap<String, Box<serde_json::value::RawValue>>,
    >(arguments) else {
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
    for (key, val) in &obj {
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
    /// Find a later complete wrapped invoke after the current one proved
    /// malformed. Families with ambiguous string syntax own this scan so a
    /// block marker inside argument data cannot become a resynchronization
    /// point. The hook must inspect `text` in linear time.
    pub resync: Option<fn(text: &str, tool_index: usize) -> Option<usize>>,
}

/// Immutable family recipe for a request-local invoke boundary.
#[derive(Clone, Copy)]
pub(crate) enum InvokeBoundaryFactory {
    Stateless(InvokeScan),
    Custom(fn() -> Box<dyn InvokeBoundary>),
    /// Native XML boundaries with a separate guided-prefix grammar.
    NativeOnly(fn() -> Box<dyn InvokeBoundary>),
}

#[derive(Clone, Copy)]
pub(crate) struct GuidedInvokePrefixContext {
    pub outside_reasoning: bool,
    pub payload_is_empty: bool,
    pub followed_by_competing_marker: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GuidedInvokePrefix {
    NoMatch,
    Pending,
    Match(usize),
    Strip(usize),
}

impl InvokeBoundaryFactory {
    pub(crate) const fn stateless(stateless: InvokeScan) -> Self {
        Self::Stateless(stateless)
    }

    pub(crate) fn create(self) -> Box<dyn InvokeBoundary> {
        match self {
            Self::Stateless(scan) => Box::new(StatelessInvokeBoundary { scan }),
            Self::Custom(create) | Self::NativeOnly(create) => create(),
        }
    }

    pub(crate) fn for_guided(self) -> Option<Self> {
        match self {
            Self::NativeOnly(_) => None,
            other => Some(other),
        }
    }

    pub(crate) const fn custom(create: fn() -> Box<dyn InvokeBoundary>) -> Self {
        Self::Custom(create)
    }
}

/// Family-owned request-local invoke-boundary contract.
///
pub(crate) trait InvokeBoundary: Send {
    /// Restore request/channel context after a retained candidate is rebased.
    fn set_guided_context(&mut self, _context: GuidedInvokePrefixContext) {}

    /// Whether the outer block is also the invoke consumed by the emitter.
    fn block_is_invoke(&self) -> bool {
        false
    }

    /// Locate an opener-less invoke owned by this family.
    fn bare_invoke_start(&self, _text: &str) -> Option<usize> {
        None
    }

    /// Hold back an undecidable suffix of an opener-less invoke.
    fn bare_invoke_holdback(&self, _text: &str) -> usize {
        0
    }

    /// Whether a syntactically located opener-less invoke is safe to recover.
    fn accepts_bare_invoke(&self, _invoke: &str) -> bool {
        true
    }

    /// Whether a bare invoke may finish from a family-owned saved closer at EOF.
    fn bare_invoke_uses_eof_boundary(&self) -> bool {
        false
    }

    fn owns_guided_prefix(&self) -> bool {
        false
    }

    /// Locate a lexical native body whose nested JSON is data in guided mode.
    fn guided_native_body_start(&self, _text: &str, _at: usize) -> Option<usize> {
        None
    }

    /// Find a family-owned guided invoke opener, including accepted spelling variants.
    fn guided_invoke_at(&self, _text: &str) -> Option<(usize, usize)> {
        None
    }

    /// Whether a control marker is one of the family-owned guided invoke openers.
    fn is_guided_invoke_marker(&self, _marker: &str) -> bool {
        false
    }

    /// Classify a family-specific guided wrapper using only newly appended bytes.
    /// `None` leaves classification to the shared stateless policy.
    fn guided_prefix_append(
        &mut self,
        _candidate: &str,
        _append: &str,
        _context: GuidedInvokePrefixContext,
    ) -> Option<GuidedInvokePrefix> {
        None
    }

    /// Advance the bound candidate with newly appended bytes only.
    fn end_append(
        &mut self,
        candidate: &str,
        append: &str,
        flush: bool,
        tool_index: usize,
    ) -> Option<usize>;
    fn opens(&self, text: &str, at: usize) -> bool;
    fn accepts_buffer_start(&self, _previous: Option<char>) -> bool {
        true
    }
    fn discard_prefix(&mut self, _bytes: usize) {}
    fn discard_unified_prefix(&mut self, _bytes: usize) {}
    /// A recovery opener can accept malformed prose; quoting shields only grammar-owned headers.
    fn owns_prose_invoke(&self, text: &str, at: usize) -> bool {
        self.opens(text, at)
    }
    fn holdback(&self, text: &str) -> usize;
    /// The Unified consumer owns a separate input buffer from the wrapped scanner.
    /// Keep consumer-specific incremental scans apart when a boundary is shared.
    fn holdback_for_unified_input(&self, text: &str) -> usize {
        self.holdback(text)
    }
    fn resync(&mut self, text: &str, flush: bool, tool_index: usize) -> Option<usize>;
    fn reset(&mut self) {}

    /// Apply the request-scoped tool-call policy.
    fn configure(&mut self, _policy: ToolCallPolicy) {}

    /// Walk one tool-channel body (bytes after the header). Channel envelopes only.
    /// `body` only ever grows between calls until a verdict is returned.
    fn channel_body(&mut self, _body: &str, _flush: bool) -> BodyScan {
        BodyScan::Wait
    }

    fn channel_body_at_eos(&mut self, body: &str) -> BodyScan {
        self.channel_body(body, true)
    }
}

struct StatelessInvokeBoundary {
    scan: InvokeScan,
}

impl InvokeBoundary for StatelessInvokeBoundary {
    fn end_append(
        &mut self,
        candidate: &str,
        _append: &str,
        flush: bool,
        tool_index: usize,
    ) -> Option<usize> {
        (self.scan.end)(candidate, flush, tool_index)
    }

    fn opens(&self, text: &str, at: usize) -> bool {
        (self.scan.opens)(text, at)
    }

    fn holdback(&self, text: &str) -> usize {
        (self.scan.holdback)(text)
    }

    fn resync(&mut self, text: &str, _flush: bool, tool_index: usize) -> Option<usize> {
        self.scan.resync.and_then(|resync| resync(text, tool_index))
    }
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
    /// Inner invoke opener. For grammars where the block itself is the invoke,
    /// this is set to the block opener and the scanner derives the block-is-invoke shape.
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
    /// Optional family-owned boundary capability. `None` preserves the
    /// marker-only path with no boundary allocation.
    pub invoke_boundary_factory: Option<InvokeBoundaryFactory>,
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
    /// Emit an append-safe update while an invoke is still open. Families that
    /// cannot prove a fragment will survive their final typing leave this as a
    /// no-op and continue to emit only at the invoke close.
    fn parse_partial_invoke(
        &mut self,
        _invoke: &str,
        _tool_index: usize,
    ) -> anyhow::Result<Option<ToolCallDelta>> {
        Ok(None)
    }

    fn parse_invoke(
        &mut self,
        invoke: &str,
        tool_index: usize,
    ) -> anyhow::Result<Option<ToolCallDelta>>;

    /// Whether an incomplete structured value may be held until a later invoke
    /// either recovers the stream or EOF confirms that the malformed call is final.
    fn defer_parse_error(&self, _invoke: &str, _error: &anyhow::Error) -> bool {
        false
    }

    /// Emit the wire-level updates for one complete invoke. Families normally
    /// have one completed update; DSML preserves its historical name-first
    /// projection at the legacy ToolParser boundary.
    fn parse_invoke_deltas(
        &mut self,
        invoke: &str,
        tool_index: usize,
    ) -> anyhow::Result<Option<Vec<ToolCallDelta>>> {
        Ok(self
            .parse_invoke(invoke, tool_index)?
            .map(|delta| vec![delta]))
    }

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

    /// Apply the request-scoped tool-call policy.
    fn configure(&mut self, _policy: ToolCallPolicy) {}

    /// A body walker for the channel-envelope call addressed to `recipient`, when the
    /// family frames that recipient's body differently from the next one (per-tool
    /// schema knowledge the static [`InvokeBoundaryFactory`] cannot carry). The scanner
    /// installs it at the completed tool header, applies the request policy and resets
    /// its cursor; `None` (the default) keeps the factory boundary.
    fn channel_boundary(&self, _recipient: &str) -> Option<Box<dyn InvokeBoundary>> {
        None
    }

    /// Decode one complete channel-envelope call body for `recipient`. An `Err` is a
    /// malformed body; the scanner applies the request's [`MalformedToolCallPolicy`].
    fn parse_channel_call(
        &mut self,
        _recipient: &str,
        _body: &str,
        _tool_index: usize,
    ) -> anyhow::Result<Option<ToolCallDelta>> {
        Ok(None)
    }

    /// Emit an append-safe update while a channel-envelope call body is still open: a
    /// prefix of the text [`Self::parse_channel_call`] will produce for the finished
    /// body. `body` only grows between calls. The default streams nothing before the
    /// terminator.
    fn parse_partial_channel_call(
        &mut self,
        _recipient: &str,
        _body: &str,
        _tool_index: usize,
    ) -> anyhow::Result<Option<ToolCallDelta>> {
        Ok(None)
    }

    /// The open channel-envelope call is over without [`Self::parse_channel_call`]
    /// completing it (dropped or rejected): forget any partial-emission state. `true`
    /// when fragments of it had already been emitted — they cannot be retracted, so the
    /// scanner spends their index rather than let the next call append to them.
    fn channel_call_done(&mut self) -> bool {
        false
    }
}

#[derive(Clone, Default)]
struct ProseQuoteState {
    quote: Option<char>,
    escaped: bool,
    previous: Option<char>,
    pending_apostrophe: bool,
}

impl ProseQuoteState {
    fn resolve_apostrophe(&mut self, next: Option<char>) {
        if self.pending_apostrophe {
            if !next.is_some_and(char::is_alphanumeric) {
                self.quote = None;
            }
            self.pending_apostrophe = false;
        }
    }

    fn advance(&mut self, character: char, next: Option<char>) {
        self.resolve_apostrophe(Some(character));
        if self.escaped {
            self.escaped = false;
        } else if character == '\\' {
            self.escaped = true;
        } else if self.quote == Some(character)
            && !(character == '\''
                && self.previous.is_some_and(char::is_alphanumeric)
                && next.is_some_and(char::is_alphanumeric))
        {
            if character == '\''
                && next.is_none()
                && self.previous.is_some_and(char::is_alphanumeric)
            {
                self.pending_apostrophe = true;
            } else {
                self.quote = None;
            }
        } else if self.quote.is_none()
            && matches!(character, '"' | '\'' | '`')
            && !(character == '\'' && self.previous.is_some_and(char::is_alphanumeric))
        {
            self.quote = Some(character);
        }
        self.previous = Some(character);
    }

    fn consume(&mut self, text: &str) {
        let mut characters = text.chars().peekable();
        while let Some(character) = characters.next() {
            self.advance(character, characters.peek().copied());
        }
    }
}

#[derive(Clone)]
struct PendingProseControl {
    at: usize,
    scanned: usize,
    state: ProseQuoteState,
    grammar_pending: bool,
}

#[derive(Clone, Default)]
pub(crate) struct ProseControlState {
    state: ProseQuoteState,
    pending: Option<PendingProseControl>,
    inactive_invoke_header: bool,
    #[cfg(test)]
    examined_bytes: usize,
}

pub(crate) enum ProseGrammarSpan {
    NotOwned,
    Owned(usize),
    Pending,
}

pub(crate) enum ProseControl {
    Ordinary,
    Literal(usize),
    Pending(usize),
    Unmatched(usize),
}

pub(crate) fn prose_invoke_span(
    text: &str,
    at: usize,
    flush: bool,
    invoke_start: &str,
    invoke_end: &str,
    factory: Option<InvokeBoundaryFactory>,
) -> ProseGrammarSpan {
    let suffix = &text[at..];
    if !flush && invoke_start.starts_with(suffix) {
        return ProseGrammarSpan::Pending;
    }
    if !suffix.starts_with(invoke_start) {
        return ProseGrammarSpan::NotOwned;
    }
    if !flush {
        return ProseGrammarSpan::Pending;
    }
    if let Some(factory) = factory {
        let mut boundary = factory.create();
        if !boundary.owns_prose_invoke(text, at) {
            return ProseGrammarSpan::NotOwned;
        }
        boundary
            .end_append(suffix, suffix, true, 0)
            .map(ProseGrammarSpan::Owned)
            .unwrap_or(ProseGrammarSpan::Pending)
    } else {
        suffix
            .find(invoke_end)
            .map(|end| ProseGrammarSpan::Owned(end + invoke_end.len()))
            .unwrap_or(ProseGrammarSpan::Pending)
    }
}

impl ProseControlState {
    pub(crate) fn has_active_quote(&self) -> bool {
        self.state.quote.is_some() || self.state.pending_apostrophe || self.pending.is_some()
    }

    pub(crate) fn consume(&mut self, text: &str) {
        self.state.consume(text);
        self.pending = None;
    }

    /// A rejected native header remains grammar debris until the next control;
    /// its attribute quotation cannot become a prose quote after chunking.
    pub(crate) fn consume_recovery(&mut self, text: &str, invoke_start: &str) {
        self.inactive_invoke_header |= text.contains(invoke_start);
        if self.inactive_invoke_header {
            self.state = ProseQuoteState::default();
            self.pending = None;
        } else {
            self.consume(text);
        }
    }

    /// A stripped control token cannot open a prose quote with its own `"`.
    pub(crate) fn consume_control(&mut self, consumed: &str, invoke_start: &str) {
        self.consume_recovery(consumed, invoke_start);
        self.state = ProseQuoteState::default();
        self.pending = None;
    }

    pub(crate) fn clear(&mut self) {
        *self = Self::default();
    }

    /// Only prose owns quotation. Invoke headers and argument bodies must never
    /// enter this lexer. A candidate waits for its closing quote or EOF because
    /// an unmatched quote cannot turn a genuine control token into literal text.
    pub(crate) fn classify<'a>(
        &mut self,
        text: &str,
        markers: impl IntoIterator<Item = &'a str>,
        flush: bool,
        mut grammar: impl FnMut(&str, usize, bool) -> ProseGrammarSpan,
    ) -> ProseControl {
        if self.inactive_invoke_header {
            return ProseControl::Ordinary;
        }
        let (at, scanned, mut state) = if let Some(pending) = self.pending.take() {
            if pending.grammar_pending && !flush {
                let at = pending.at;
                self.pending = Some(pending);
                return ProseControl::Pending(at);
            }
            (pending.at, pending.scanned, pending.state)
        } else {
            let Some(at) = markers
                .into_iter()
                .filter_map(|marker| text.find(marker))
                .min()
            else {
                return ProseControl::Ordinary;
            };
            let mut state = self.state.clone();
            state.resolve_apostrophe(text.chars().next());
            let mut cursor = 0;
            while cursor < at {
                match grammar(text, cursor, flush) {
                    ProseGrammarSpan::Owned(length) => {
                        cursor = (cursor + length).min(at);
                        continue;
                    }
                    ProseGrammarSpan::Pending => break,
                    ProseGrammarSpan::NotOwned => {}
                }
                let character = text[cursor..].chars().next().expect("prose cursor");
                state.advance(
                    character,
                    text[cursor + character.len_utf8()..].chars().next(),
                );
                cursor += character.len_utf8();
            }
            if state.quote.is_none() {
                return ProseControl::Ordinary;
            }
            (at, at, state)
        };
        let mut cursor = scanned;
        while cursor < text.len() {
            match grammar(text, cursor, flush) {
                ProseGrammarSpan::Owned(length) => {
                    cursor += length;
                    continue;
                }
                ProseGrammarSpan::Pending => {
                    if !flush {
                        self.pending = Some(PendingProseControl {
                            at,
                            scanned: cursor,
                            state,
                            grammar_pending: true,
                        });
                        return ProseControl::Pending(at);
                    }
                }
                ProseGrammarSpan::NotOwned => {}
            }
            let character = text[cursor..].chars().next().expect("prose cursor");
            if !flush && character == '\'' && cursor + character.len_utf8() == text.len() {
                self.pending = Some(PendingProseControl {
                    at,
                    scanned: cursor,
                    state,
                    grammar_pending: false,
                });
                return ProseControl::Pending(at);
            }
            #[cfg(test)]
            {
                self.examined_bytes += character.len_utf8();
            }
            cursor += character.len_utf8();
            state.advance(character, text[cursor..].chars().next());
            if flush && cursor == text.len() {
                state.resolve_apostrophe(None);
            }
            if state.quote.is_none() {
                return ProseControl::Literal(cursor);
            }
        }
        if flush {
            ProseControl::Unmatched(at)
        } else {
            self.pending = Some(PendingProseControl {
                at,
                scanned: text.len(),
                state,
                grammar_pending: false,
            });
            ProseControl::Pending(at)
        }
    }

    pub(crate) fn punctuation_holdback(&self, text: &str) -> usize {
        usize::from(text.ends_with('\''))
    }
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
pub(crate) fn find_first_outside_strings<'a, I>(text: &str, markers: I) -> Option<(usize, usize)>
where
    I: Clone + IntoIterator<Item = &'a str>,
{
    let mut quoted = JsonStringState::default();
    for (idx, c) in text.char_indices() {
        if quoted.advance(c) {
            continue;
        }
        if let Some((_, len)) = markers
            .clone()
            .into_iter()
            .find(|m| text[idx..].starts_with(m))
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

struct ToolSeparator {
    separator: &'static str,
    prefixed_openers: Vec<String>,
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
    buffer_previous: Option<char>,
    prose: ProseControlState,
    tool_separator: Option<ToolSeparator>,
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
    pending_parse_error: Option<DeferredParseError>,
    invoke_boundary: Option<Box<dyn InvokeBoundary>>,
    invoke_boundary_len: usize,
    envelope: Option<&'static ChannelEnvelope>,
    /// The guided reader's view of the envelope: the same header rules as fn hooks.
    guided_channel: Option<GuidedChannel>,
    env_mode: EnvMode,
    /// Mode a fresh stream starts in, chosen by the request's starting state.
    env_initial: EnvMode,
    env_recipient: String,
    /// JSON string context of the open message body (see [`EnvelopeChannel::json_body`]).
    env_json: JsonBodyState,
    /// A header may begin at the buffer start (stream start or after a message end),
    /// so a prompt-consumed `role NAME` lead must be held, not emitted.
    header_slot: bool,
    policy: ToolCallPolicy,
    finish_at_eos: bool,
}

struct DeferredParseError {
    error: anyhow::Error,
    tool_index: usize,
    bytes: usize,
}

impl<E: InvokeEmitter> WrappedBlockScanner<E> {
    pub(crate) fn new(spec: WrappedBlockSpec, emitter: E) -> Self {
        let invoke_boundary = spec
            .invoke_boundary_factory
            .map(InvokeBoundaryFactory::create);
        Self {
            spec,
            emitter,
            reasoning: None,
            reasoning_enabled: false,
            reasoning_forced_start: false,
            buffer: String::new(),
            buffer_previous: None,
            prose: ProseControlState::default(),
            tool_separator: None,
            uncommitted_block: String::new(),
            in_block: false,
            in_reasoning: false,
            accept_redundant_reasoning_start: false,
            resume_reasoning: false,
            suppress_normal_text: false,
            next_index: 0,
            pending_parse_error: None,
            invoke_boundary,
            invoke_boundary_len: 0,
            envelope: None,
            guided_channel: None,
            env_mode: EnvMode::Idle,
            env_initial: EnvMode::Idle,
            env_recipient: String::new(),
            env_json: JsonBodyState::default(),
            header_slot: true,
            policy: ToolCallPolicy::default(),
            finish_at_eos: false,
        }
    }

    /// Frame this scanner's messages with a [`ChannelEnvelope`] instead of wrapped
    /// blocks. Tool bodies are walked by the family's invoke boundary.
    pub(crate) fn with_envelope(mut self, envelope: &'static ChannelEnvelope) -> Self {
        self.envelope = Some(envelope);
        self
    }

    pub(crate) fn has_envelope(&self) -> bool {
        self.envelope.is_some()
    }

    /// Hand the guided reader the envelope's header hooks. Built by the family from
    /// the shared `envelope_guided_*` resolvers over its own [`ChannelEnvelope`], so
    /// native and guided ownership follow one grammar table.
    pub(crate) fn with_guided_channel(mut self, channel: GuidedChannel) -> Self {
        self.guided_channel = Some(channel);
        self
    }

    pub(crate) fn guided_channel(&self) -> Option<GuidedChannel> {
        self.guided_channel
    }

    /// Control markers for the guided reader. An envelope's header markers (`start`,
    /// `channel`, `message`) belong to its header resolver — a header is held,
    /// stripped or opened WHOLE, from its first byte — so none of them is also a
    /// standalone control there; only the envelope's terminators and orphan controls
    /// are. Treating `channel` as a control consumed it out of a header still
    /// arriving and released the lead in front of it as text.
    pub(crate) fn guided_control_markers(&self) -> Vec<String> {
        self.spec
            .holdback_markers
            .iter()
            .filter(|m| {
                self.envelope
                    .is_none_or(|env| ![env.start, env.channel, env.message].contains(&m.as_str()))
            })
            .cloned()
            .collect()
    }

    /// Request-scoped malformed-call and argument-framing policy. Survives `reset`.
    pub(crate) fn set_tool_policy(&mut self, policy: ToolCallPolicy) {
        self.policy = policy;
        self.emitter.configure(policy);
        if let Some(boundary) = self.invoke_boundary.as_mut() {
            boundary.configure(policy);
        }
    }

    /// Consume one template separator before native tool openers, retaining it
    /// across chunk boundaries until the opener is distinguished from text.
    pub(crate) fn with_tool_separator(
        mut self,
        separator: &'static str,
        bare_invoke_prefixes: &[&str],
    ) -> Self {
        let prefixed_openers = self
            .spec
            .block_starts
            .iter()
            .map(String::as_str)
            .chain(bare_invoke_prefixes.iter().copied())
            .map(|opener| format!("{separator}{opener}"))
            .collect();
        self.tool_separator = Some(ToolSeparator {
            separator,
            prefixed_openers,
        });
        self
    }

    pub(crate) fn allow_implicit_tool_terminator(&mut self) {
        self.finish_at_eos = true;
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

    /// The invoke terminator, for pairing with a stripped invoke opener.
    ///
    /// NOT part of [`Self::guided_control_markers`]: a BARE `</function>` with no invoke
    /// open is ordinary text to the native scanner, and measured identical on both
    /// paths, so stripping it unconditionally would create a divergence rather than
    /// remove one. It is consumed only as the tail of an invoke already stripped.
    pub(crate) fn invoke_end(&self) -> &str {
        &self.spec.invoke_end
    }

    pub(crate) fn invoke_start(&self) -> &str {
        &self.spec.invoke_start
    }

    pub(crate) fn invoke_boundary_factory(&self) -> Option<InvokeBoundaryFactory> {
        self.spec.invoke_boundary_factory
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
        if let Some(env) = self.envelope {
            // The prompt already opened a channel: a thought (`Some(true)`) or the
            // visible reply (`Some(false)`). `None` starts between messages.
            self.env_initial = match forced_start {
                Some(reasoning) => env
                    .channels
                    .iter()
                    .position(|c| c.reasoning == reasoning)
                    .map_or(EnvMode::Idle, EnvMode::Channel),
                None => EnvMode::Idle,
            };
            self.env_mode = self.env_initial;
            self.header_slot = true;
            self.env_recipient.clear();
        }
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
        self.drain(true, out)?;
        if let Some(deferred) = self.pending_parse_error.take() {
            return Err(deferred.error);
        }
        Ok(())
    }

    /// Clear one stream's scan state and return bytes not yet emitted.
    pub(crate) fn reset(&mut self) -> String {
        self.finish_at_eos = false;
        let mut pending = std::mem::take(&mut self.uncommitted_block);
        pending.push_str(&std::mem::take(&mut self.buffer));
        self.buffer_previous = None;
        self.in_block = false;
        self.prose.clear();
        self.in_reasoning = self.reasoning_enabled && self.reasoning_forced_start;
        self.accept_redundant_reasoning_start = self.in_reasoning;
        // Without this a reset mid-thought leaves "resume reasoning after the call"
        // armed, and the NEXT stream's first post-call answer is emitted as reasoning.
        self.resume_reasoning = false;
        self.suppress_normal_text = false;
        self.next_index = 0;
        self.pending_parse_error = None;
        self.emitter.reset();
        self.reset_invoke_boundary();
        self.env_mode = self.env_initial;
        self.header_slot = true;
        self.env_recipient.clear();
        // The JSON-body string state belongs to the stream too: left over from a
        // reply that ended inside a string, it made the next reply's terminator
        // read as string data (or a quoted marker read as the terminator).
        self.env_json.reset();
        pending
    }

    fn consume_buffer_prefix(&mut self, bytes: usize) {
        if bytes == 0 {
            return;
        }
        if self.pending_parse_error.is_some() && self.in_block {
            self.uncommitted_block.push_str(&self.buffer[..bytes]);
        }
        self.buffer_previous = self.buffer[..bytes].chars().next_back();
        self.buffer.drain(..bytes);
        if let Some(boundary) = self.invoke_boundary.as_mut() {
            boundary.discard_prefix(bytes);
        }
    }

    fn clear_uncommitted_block(&mut self) {
        if self.pending_parse_error.is_none() {
            self.uncommitted_block.clear();
        }
    }

    /// Whether one invoke closer also closes the surrounding block.
    fn invoke_closes_block(&self) -> bool {
        self.spec.block_ends.contains(&self.spec.invoke_end)
    }

    fn parse_complete_invoke(
        &mut self,
        invoke: &str,
        tool_index: usize,
    ) -> anyhow::Result<Option<Vec<ToolCallDelta>>> {
        match self.emitter.parse_invoke_deltas(invoke, tool_index) {
            Ok(Some(deltas)) if !deltas.is_empty() => {
                if let Some(deferred) = self.pending_parse_error.take() {
                    tracing::warn!(
                        why = %format!("{}_incomplete_invoke_recovered", self.spec.family),
                        tool_index = deferred.tool_index,
                        dropped_bytes = deferred.bytes,
                        "Dropped an incomplete invoke after a later call parsed successfully"
                    );
                }
                Ok(Some(deltas))
            }
            Err(error) if self.emitter.defer_parse_error(invoke, &error) => {
                let deferred = self
                    .pending_parse_error
                    .get_or_insert_with(|| DeferredParseError {
                        error,
                        tool_index,
                        bytes: 0,
                    });
                deferred.bytes += invoke.len();
                Ok(None)
            }
            result => result,
        }
    }

    fn reset_invoke_boundary(&mut self) {
        if let Some(boundary) = self.invoke_boundary.as_mut() {
            boundary.reset();
        }
        self.invoke_boundary_len = 0;
    }

    fn block_is_invoke(&self) -> bool {
        self.invoke_boundary
            .as_ref()
            .is_some_and(|boundary| boundary.block_is_invoke())
    }

    /// Find the next real invoke opener, applying the family hook when present.
    fn find_invoke_start(&self, text: &str) -> Option<usize> {
        if self.block_is_invoke() && self.in_block {
            return Some(0);
        }
        if self.block_is_invoke() {
            let boundary = self
                .invoke_boundary
                .as_ref()
                .expect("block-is-invoke requires a family boundary");
            return boundary
                .bare_invoke_start(text)
                .filter(|&start| boundary.accepts_bare_invoke(&text[start..]));
        }
        let invoke_start = &self.spec.invoke_start;
        let Some(boundary) = self.invoke_boundary.as_ref() else {
            return text.find(invoke_start.as_str());
        };
        let mut cursor = 0;
        while let Some(relative) = text[cursor..].find(invoke_start.as_str()) {
            let at = cursor + relative;
            if (at > 0 || boundary.accepts_buffer_start(self.buffer_previous))
                && boundary.opens(text, at)
            {
                return Some(at);
            }
            cursor = at + invoke_start.len();
        }
        None
    }

    fn active_invoke_start(&self) -> Option<usize> {
        if self.block_is_invoke() {
            Some(0)
        } else {
            self.find_invoke_start(&self.buffer)
        }
    }

    /// Offset just past the closer of the invoke beginning at byte zero.
    fn invoke_end_at(&mut self, flush: bool) -> Option<usize> {
        match self.invoke_boundary.as_mut() {
            Some(boundary) => {
                let append = &self.buffer[self.invoke_boundary_len..];
                self.invoke_boundary_len = self.buffer.len();
                boundary.end_append(&self.buffer, append, flush, self.next_index)
            }
            None => self
                .buffer
                .find(self.spec.invoke_end.as_str())
                .map(|position| position + self.spec.invoke_end.len()),
        }
    }

    /// Find a later complete block that can safely restart scanning after the
    /// current invoke proved malformed. The new block must begin with a real,
    /// complete invoke; a marker-looking string inside the malformed payload
    /// cannot satisfy both grammar checks.
    fn resync_block_start(&mut self, flush: bool) -> Option<usize> {
        self.invoke_boundary
            .as_mut()?
            .resync(&self.buffer, flush, self.next_index)
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
            .invoke_boundary
            .as_ref()
            .map(|boundary| {
                let keep = boundary.holdback(&self.buffer);
                if keep == self.buffer.len() && !boundary.accepts_buffer_start(self.buffer_previous)
                {
                    0
                } else {
                    keep
                }
            })
            .unwrap_or_default();
        let bare = self
            .invoke_boundary
            .as_ref()
            .map(|boundary| boundary.bare_invoke_holdback(&self.buffer))
            .unwrap_or_default();
        let separator = self
            .tool_separator
            .as_ref()
            .filter(|_| !self.in_reasoning && !self.suppress_normal_text)
            .map(|framing| {
                marker_prefix_suffix_len(
                    &self.buffer,
                    framing.prefixed_openers.iter().map(String::as_str),
                )
            })
            .unwrap_or_default();
        regular
            .max(self.prose.punctuation_holdback(&self.buffer))
            .max(reasoning)
            .max(self.pending_label_len())
            .max(invoke)
            .max(bare)
            .max(separator)
    }

    fn tool_separator_len_at(&self, at: usize) -> usize {
        let Some(framing) = self
            .tool_separator
            .as_ref()
            .filter(|_| !self.in_reasoning && !self.suppress_normal_text)
        else {
            return 0;
        };
        if !self.buffer[..at].ends_with(framing.separator) {
            return 0;
        }
        let tail = &self.buffer[at..];
        if self.spec.block_starts.iter().any(|m| tail.starts_with(m))
            || self.find_invoke_start(tail) == Some(0)
        {
            framing.separator.len()
        } else {
            0
        }
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
            self.prose
                .consume_recovery(&self.buffer[..at], &self.spec.invoke_start);
            self.prose.clear();
            self.consume_buffer_prefix(at + consume);
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
            self.prose
                .consume_recovery(&self.buffer[..emit_len], &self.spec.invoke_start);
            self.consume_buffer_prefix(emit_len);
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

    /// Drain a channel-envelope grammar. Same buffer, prose-quote protection,
    /// holdback, call indices and reset contract as the wrapped-block drain.
    fn drain_envelope<S: EventSink + ?Sized>(
        &mut self,
        env: &'static ChannelEnvelope,
        flush: bool,
        out: &mut S,
    ) -> anyhow::Result<()> {
        loop {
            if self.env_mode == EnvMode::Call {
                if self.envelope_call(flush, out)? {
                    continue;
                }
                return Ok(());
            }
            let channel = match self.env_mode {
                EnvMode::Channel(i) => Some(&env.channels[i]),
                _ => None,
            };
            // `Response` means the prompt already opened visible content, so this
            // stream has no reasoning channel: a framed thought's body is visible text
            // (its framing is still control, never leaked).
            let kind = match channel {
                Some(c) if c.reasoning && self.reasoning_enabled => Kind::Reasoning,
                _ => Kind::Text,
            };
            let markers: Vec<&'static str> = match channel {
                Some(c) => c.terminators.iter().copied().chain([env.start]).collect(),
                None => [env.start, env.channel]
                    .into_iter()
                    .chain(env.orphan_controls.iter().copied())
                    .collect(),
            };
            // A tool header and its body belong to the grammar, not to prose: their
            // attribute quotes can never close or open a prose quotation. The body is
            // walked with the recipient's own boundary — the one `envelope_idle` would
            // install for the real call — so prose ownership and the call agree about
            // where every parameter value ends.
            let factory = self.spec.invoke_boundary_factory;
            let policy = self.policy;
            let emitter = &self.emitter;
            let boundary_for = |recipient: &str| emitter.channel_boundary(recipient);
            let control = if env.quote_aware {
                self.prose.classify(
                    &self.buffer,
                    markers.iter().copied(),
                    flush,
                    |text, at, flush| {
                        envelope_tool_span(env, factory, policy, &boundary_for, text, at, flush)
                    },
                )
            } else {
                ProseControl::Ordinary
            };
            let (length, waiting, unmatched) = match control {
                ProseControl::Literal(length) => (length, false, false),
                ProseControl::Pending(at) => (at, true, false),
                ProseControl::Unmatched(at) => (at, false, true),
                ProseControl::Ordinary => (0, false, false),
            };
            if length > 0 {
                let prose: String = self.buffer.drain(..length).collect();
                self.prose.consume(&prose);
                push_run(out, kind, &prose);
                if channel.is_none() {
                    self.header_slot = false;
                }
            }
            if waiting {
                return Ok(());
            }
            if unmatched {
                self.prose.clear();
            } else if length > 0 {
                continue;
            }
            let progressed = match channel {
                Some(c) => self.envelope_body(env, c, kind, &markers, flush, out),
                None => self.envelope_idle(env, flush, out)?,
            };
            if !progressed {
                return Ok(());
            }
        }
    }

    /// One step inside a message body. `true` means progress was made.
    fn envelope_body<S: EventSink + ?Sized>(
        &mut self,
        env: &'static ChannelEnvelope,
        channel: &EnvelopeChannel,
        kind: Kind,
        markers: &[&'static str],
        flush: bool,
        out: &mut S,
    ) -> bool {
        // The first marker that is not data: inside a JSON answer's open string a
        // marker is literal (`{"value":"<|return|>"}<|return|>` ends at the second).
        let mut candidates: Vec<(usize, &'static str)> = markers
            .iter()
            .flat_map(|m| self.buffer.match_indices(m).map(move |(at, _)| (at, *m)))
            .collect();
        candidates.sort_unstable();
        let found = candidates.into_iter().find(|(at, _)| {
            !(channel.json_body && self.env_json.inside_string_at(&self.buffer, *at))
        });
        if let Some((at, marker)) = found {
            push_run(out, kind, &self.buffer[..at]);
            self.env_json.reset();
            let consumed = at + envelope_close_len(env, marker);
            self.buffer.drain(..consumed);
            self.prose.clear();
            self.env_mode = EnvMode::Idle;
            self.header_slot = true;
            return true;
        }
        let keep = if flush {
            0
        } else {
            marker_prefix_suffix_len(&self.buffer, markers.iter().copied())
                .max(self.prose.punctuation_holdback(&self.buffer))
        };
        let emit = self.buffer.len() - keep;
        if emit > 0 {
            push_run(out, kind, &self.buffer[..emit]);
            self.env_json.advance(&self.buffer[..emit]);
            self.prose.consume(&self.buffer[..emit]);
            self.buffer.drain(..emit);
        }
        false
    }

    /// One step between messages: text, or a header. `Ok(true)` means progress.
    fn envelope_idle<S: EventSink + ?Sized>(
        &mut self,
        env: &'static ChannelEnvelope,
        flush: bool,
        out: &mut S,
    ) -> anyhow::Result<bool> {
        if self.buffer.is_empty() {
            return Ok(false);
        }
        let hold = marker_prefix_suffix_len(
            &self.buffer,
            [env.start, env.channel]
                .into_iter()
                .chain(env.orphan_controls.iter().copied()),
        );
        let s = self.buffer.find(env.start);
        let c = self.buffer.find(env.channel);
        // A terminator with no channel open is debris between messages (`I3`): strip
        // it, keeping the text before it. Inside a body the channel's own terminator
        // list decides, so a `final` body keeps a literal call marker.
        if let Some((p, len)) = env
            .orphan_controls
            .iter()
            .filter_map(|m| self.buffer.find(m).map(|p| (p, m.len())))
            .min_by_key(|(p, _)| *p)
            && s.is_none_or(|s| p < s)
            && c.is_none_or(|c| p < c)
        {
            if p > 0 {
                self.envelope_text(p, out);
            }
            tracing::debug!(
                why = %format!("{}_orphan_control", self.spec.family),
                "stream stripped a control token outside any channel"
            );
            self.buffer.drain(..len);
            self.prose.clear();
            return Ok(true);
        }
        // `(position of the channel marker, header lead)`.
        let (channel_pos, recipient) = match (s, c) {
            (None, None) => {
                let lead = &self.buffer[..self.buffer.len() - hold];
                if self.header_slot
                    && !flush
                    && (lead.len() <= env.max_header || envelope_prefix_possible(env, lead))
                {
                    return Ok(false);
                }
                let n = self.buffer.len() - if flush { 0 } else { hold };
                if n == 0 {
                    return Ok(false);
                }
                self.envelope_text(n, out);
                return Ok(true);
            }
            (Some(s), c) if c.is_none_or(|c| s < c) => {
                if s > 0 {
                    self.envelope_text(s, out);
                    return Ok(true);
                }
                let rest = &self.buffer[env.start.len()..];
                // A control marker inside the lead (`<|start|>ysis<|message|>…<|start|>
                // assistant<|channel|>`) ends the garbled region there: the bytes up to it
                // are debris, the marker itself is resolved on the next step.
                let lead_cut = env
                    .orphan_controls
                    .iter()
                    .chain(std::iter::once(&env.start))
                    .filter_map(|m| rest.find(m))
                    .min();
                match rest.find(env.channel) {
                    found if lead_cut.is_some_and(|q| found.is_none_or(|ci| q < ci)) => {
                        let q = lead_cut.expect("checked above");
                        tracing::debug!(
                            why = %format!("{}_garbled_lead", self.spec.family),
                            "stream dropped a start marker with a broken lead"
                        );
                        self.buffer.drain(..env.start.len() + q);
                        self.prose.clear();
                        return Ok(true);
                    }
                    // Within a header's length budget the lead is held back until
                    // CHANNEL arrives (or not): a garbled header such as
                    // `<|start|>assistant\tto=NAME<|channel|>` must be judged whole, in
                    // the chunked stream exactly as in the batch one. A lead that can
                    // still grow into a legal header waits regardless of length.
                    None if !flush
                        && (rest.len() <= env.max_header
                            || envelope_prefix_possible(
                                env,
                                &rest[..rest.len() - marker_prefix_suffix_len(rest, [env.channel])],
                            )) =>
                    {
                        return Ok(false);
                    }
                    None => {
                        // A recognizable header that stops before or inside CHANNEL can
                        // never complete. A tool lead is an unfinished call, not prose:
                        // strict rejects it with every byte (START included) still
                        // buffered for `reset`; a role-only lead is framing debris and is
                        // dropped whole. Ordinary text after START keeps only START out.
                        if flush {
                            let lead =
                                &rest[..rest.len() - marker_prefix_suffix_len(rest, [env.channel])];
                            match envelope_parse_prefix(env, lead) {
                                HeaderLead::Lead(Some(_)) => {
                                    return self.envelope_bad_header(true, self.buffer.len());
                                }
                                HeaderLead::Lead(None) => {
                                    return self.envelope_bad_header(false, self.buffer.len());
                                }
                                HeaderLead::NotHeader => {}
                            }
                        }
                        self.buffer.drain(..env.start.len());
                        return Ok(true);
                    }
                    Some(ci) => match envelope_parse_prefix(env, &rest[..ci]) {
                        HeaderLead::NotHeader if ci <= env.max_header => {
                            // `<|start|>` + a lead that is not `role[ to=NAME]` + `<|channel|>`:
                            // a garbled header. Its bytes are framing, not prose (goal 3/4), so
                            // the whole header is skipped; a tool-framed one (`to=` in the lead
                            // or `<|constrain|>` in the label) also takes its body up to the
                            // tool channel's terminator under the malformed-call policy.
                            return self.envelope_garbled_header(env, ci, flush);
                        }
                        HeaderLead::NotHeader => {
                            // Longer than any header: START is debris in front of prose.
                            self.buffer.drain(..env.start.len());
                            return Ok(true);
                        }
                        HeaderLead::Lead(r) => (env.start.len() + ci, r),
                    },
                }
            }
            (_, Some(c)) => match envelope_parse_prefix(env, &self.buffer[..c]) {
                HeaderLead::Lead(r) => (c, r),
                HeaderLead::NotHeader => {
                    self.envelope_text(c, out);
                    return Ok(true);
                }
            },
            _ => unreachable!("one marker is present"),
        };
        let after = channel_pos + env.channel.len();
        // A control marker inside the label (`<|channel|>analysis<|end|>`) means the
        // header never had a `<|message|>`: the label is debris up to that marker, which
        // the next step then handles as an orphan or a new header.
        if let Some(cut) = envelope_label_cut(env, &self.buffer[after..]) {
            return self.envelope_bad_header(recipient.is_some(), after + cut);
        }
        let Some(mi) = self.buffer[after..].find(env.message) else {
            if !flush && self.buffer.len() - after <= env.max_header {
                return Ok(false);
            }
            let skip = if flush { self.buffer.len() } else { after };
            return self.envelope_bad_header(recipient.is_some(), skip);
        };
        let label = &self.buffer[after..after + mi];
        match envelope_parse_channel(env, label) {
            // A tool-framed header (`<|constrain|>` in the label) on a message channel is
            // an illegal tool header: strict rejects it, drop skips it with its body.
            Some(index)
                if env.channels[index].name != env.tool_channel
                    && label.contains(env.constrain) =>
            {
                self.envelope_bad_tool_header(env, after + mi + env.message.len(), flush)
            }
            // `<|channel|>analysis <|constrain|>xml<|message|>` with no recipient: a tool
            // header missing its addressee, not a thought. Its body is tool markup and is
            // skipped with it; it never becomes reasoning text.
            Some(_) if recipient.is_none() && label.contains(env.constrain) => {
                self.envelope_bad_tool_header(env, after + mi + env.message.len(), flush)
            }
            Some(index) => {
                let consume = after + mi + env.message.len();
                // A bare recipient on a message channel (`to=x<|channel|>final<|message|>`)
                // is header debris: the channel opens as written, nothing is dropped.
                let recipient = if env.channels[index].name == env.tool_channel {
                    recipient
                } else {
                    None
                };
                match recipient {
                    Some(name) => {
                        self.uncommitted_block = self.buffer[..consume].to_string();
                        // A family may frame THIS recipient's body from its schema; the
                        // factory boundary stays in place otherwise.
                        if let Some(mut boundary) = self.emitter.channel_boundary(&name) {
                            boundary.configure(self.policy);
                            self.invoke_boundary = Some(boundary);
                        }
                        self.env_recipient = name;
                        self.env_mode = EnvMode::Call;
                        self.reset_invoke_boundary();
                    }
                    None => {
                        self.env_json.reset();
                        self.env_mode = EnvMode::Channel(index);
                    }
                }
                self.buffer.drain(..consume);
                self.prose.clear();
                Ok(true)
            }
            None if recipient.is_some() => {
                self.envelope_bad_tool_header(env, after + mi + env.message.len(), flush)
            }
            // An unknown label (`<|channel|>reasoning  <|constrain|>xml<|message|>`) is
            // still a complete header: skip it whole instead of leaking the label.
            None if self.buffer[after..after + mi].contains(env.constrain) => {
                self.envelope_bad_tool_header(env, after + mi + env.message.len(), flush)
            }
            // An unknown channel (`<|channel|>commentary<|message|>…<|end|>`) is a whole
            // message the grammar does not define: its body is not visible text either,
            // so header and body go together, through the nearest terminator.
            None => self.envelope_skip_message(env, after + mi + env.message.len(), flush),
        }
    }

    /// `<|start|>` followed by a lead that is not a header, then `<|channel|>` at
    /// `ci` (relative to the byte after START). Skips the header as framing debris;
    /// when it is tool-framed, also skips the body through the tool channel's
    /// terminator (waiting for it while the stream is open). `Ok(true)` on progress.
    fn envelope_garbled_header(
        &mut self,
        env: &'static ChannelEnvelope,
        ci: usize,
        flush: bool,
    ) -> anyhow::Result<bool> {
        let start_len = env.start.len();
        let lead = &self.buffer[start_len..start_len + ci];
        let after = start_len + ci + env.channel.len();
        let tool_lead = lead.contains(env.recipient_key);
        let Some(mi) = self.buffer[after..].find(env.message) else {
            if !flush && self.buffer.len() - after <= env.max_header {
                return Ok(false);
            }
            let skip = if flush { self.buffer.len() } else { after };
            return self.envelope_bad_header(tool_lead, skip);
        };
        let label = &self.buffer[after..after + mi];
        let tool = tool_lead || label.contains(env.constrain);
        let header_end = after + mi + env.message.len();
        if !tool {
            return self.envelope_skip_message(env, header_end, flush);
        }
        self.envelope_bad_tool_header(env, header_end, flush)
    }

    /// A complete but undefined message header whose `<|message|>` ends at `header_end`:
    /// the body is skipped with it, through the nearest terminator any channel knows
    /// (waiting for one while the stream is open; everything goes at flush).
    fn envelope_skip_message(
        &mut self,
        env: &'static ChannelEnvelope,
        header_end: usize,
        flush: bool,
    ) -> anyhow::Result<bool> {
        let body_end = env
            .channels
            .iter()
            .flat_map(|c| c.terminators.iter())
            .filter_map(|m| {
                self.buffer[header_end..]
                    .find(m)
                    .map(|p| header_end + p + m.len())
            })
            .min();
        match body_end {
            Some(end) => self.envelope_bad_header(false, end),
            None if flush => self.envelope_bad_header(false, self.buffer.len()),
            None => Ok(false),
        }
    }

    /// A malformed TOOL header whose `<|message|>` ends at `header_end`: the body that
    /// follows is tool markup too, so it is skipped through the tool channel's terminator
    /// (waiting for it while the stream is open) under the malformed-call policy.
    fn envelope_bad_tool_header(
        &mut self,
        env: &'static ChannelEnvelope,
        header_end: usize,
        flush: bool,
    ) -> anyhow::Result<bool> {
        let body_end = env
            .channels
            .iter()
            .find(|c| c.name == env.tool_channel)
            .map(|c| c.terminators)
            .unwrap_or(&[])
            .iter()
            .filter_map(|m| {
                self.buffer[header_end..]
                    .find(m)
                    .map(|p| header_end + p + m.len())
            })
            .min();
        match body_end {
            Some(end) => self.envelope_bad_header(true, end),
            None if flush => self.envelope_bad_header(true, self.buffer.len()),
            None => Ok(false),
        }
    }

    fn envelope_text<S: EventSink + ?Sized>(&mut self, n: usize, out: &mut S) {
        // `Response` (no reasoning channel: the prompt already opened the visible reply)
        // keeps every byte visible, as the whole stream IS the reply.
        let prose =
            self.envelope.is_none_or(|env| env.prose_between_messages) || !self.reasoning_enabled;
        if prose {
            out.push_text(&self.buffer[..n]);
            self.prose.consume(&self.buffer[..n]);
        } else {
            tracing::debug!(
                why = %format!("{}_outside_channel", self.spec.family),
                bytes = n,
                "stream dropped bytes outside every channel"
            );
        }
        self.buffer.drain(..n);
        self.header_slot = false;
    }

    /// A header that can never complete. A tool header follows the request's
    /// malformed-call policy; a message header is control debris and is skipped.
    fn envelope_bad_header(&mut self, tool: bool, skip: usize) -> anyhow::Result<bool> {
        if tool && self.policy.malformed == MalformedToolCallPolicy::Reject {
            return Err(anyhow::Error::new(MalformedToolCall {
                reason: "malformed tool-call header",
            }));
        }
        tracing::warn!(
            why = %format!("{}_malformed_header", self.spec.family),
            "stream skipped a malformed channel header"
        );
        self.buffer.drain(..skip);
        Ok(true)
    }

    /// One step inside a tool call body. `Ok(true)` means progress was made.
    fn envelope_call<S: EventSink + ?Sized>(
        &mut self,
        flush: bool,
        out: &mut S,
    ) -> anyhow::Result<bool> {
        let verdict = match self.invoke_boundary.as_mut() {
            Some(boundary) if flush && self.finish_at_eos => {
                boundary.channel_body_at_eos(&self.buffer)
            }
            Some(boundary) => boundary.channel_body(&self.buffer, flush),
            None => anyhow::bail!("channel envelope needs an invoke boundary"),
        };
        let (body_end, consume) = match verdict {
            BodyScan::Wait => {
                // Still open: a family that can prove a prefix of its final arguments
                // streams it now instead of holding everything for the terminator.
                if !flush
                    && let Some(delta) = self.emitter.parse_partial_channel_call(
                        &self.env_recipient,
                        &self.buffer,
                        self.next_index,
                    )?
                {
                    out.push_call(delta);
                }
                return Ok(false);
            }
            BodyScan::Malformed { discard, reason } => {
                if self.policy.malformed == MalformedToolCallPolicy::Reject {
                    // Leave every byte in place so `reset` returns the whole call.
                    return Err(anyhow::Error::new(MalformedToolCall { reason }));
                }
                tracing::warn!(
                    why = %format!("{}_malformed_call", self.spec.family),
                    reason,
                    "stream dropped a malformed tool call"
                );
                self.buffer.drain(..discard);
                self.envelope_call_done();
                return Ok(true);
            }
            BodyScan::Complete { body_end, consume } => (body_end, consume),
        };
        let recipient = self.env_recipient.clone();
        // Emit BEFORE consuming, like the wrapped drain: a failure keeps the bytes.
        match self
            .emitter
            .parse_channel_call(&recipient, &self.buffer[..body_end], self.next_index)
        {
            Ok(Some(delta)) => {
                out.push_call(delta);
                self.next_index += 1;
            }
            Ok(None) => {}
            Err(error) => {
                if self.policy.malformed == MalformedToolCallPolicy::Reject {
                    return Err(error);
                }
                tracing::warn!(
                    why = %format!("{}_malformed_call", self.spec.family),
                    "stream dropped a tool call with a malformed body"
                );
            }
        }
        self.buffer.drain(..consume);
        self.envelope_call_done();
        Ok(true)
    }

    fn envelope_call_done(&mut self) {
        if self.emitter.channel_call_done() {
            self.next_index += 1;
        }
        self.reset_invoke_boundary();
        self.uncommitted_block.clear();
        self.env_recipient.clear();
        self.env_mode = EnvMode::Idle;
        self.header_slot = true;
        self.prose.clear();
    }

    fn drain<S: EventSink + ?Sized>(&mut self, flush: bool, out: &mut S) -> anyhow::Result<()> {
        if let Some(env) = self.envelope {
            return self.drain_envelope(env, flush, out);
        }
        loop {
            if !self.in_block {
                let control = self.prose.classify(
                    &self.buffer,
                    self.spec
                        .holdback_markers
                        .iter()
                        .map(String::as_str)
                        .chain(self.reasoning.iter().flat_map(|r| [r.start, r.end])),
                    flush,
                    |text, at, flush| {
                        prose_invoke_span(
                            text,
                            at,
                            flush,
                            &self.spec.invoke_start,
                            &self.spec.invoke_end,
                            self.spec.invoke_boundary_factory,
                        )
                    },
                );
                let (length, waiting, unmatched) = match control {
                    ProseControl::Literal(length) => (length, false, false),
                    ProseControl::Pending(at) => (at - self.tool_separator_len_at(at), true, false),
                    ProseControl::Unmatched(at) => {
                        (at - self.tool_separator_len_at(at), false, true)
                    }
                    ProseControl::Ordinary => (0, false, false),
                };
                if length > 0 {
                    let prose = self.buffer[..length].to_owned();
                    self.consume_buffer_prefix(length);
                    self.prose.consume(&prose);
                    if self.in_reasoning || !self.suppress_normal_text {
                        push_run(
                            out,
                            if self.in_reasoning {
                                Kind::Reasoning
                            } else {
                                Kind::Text
                            },
                            &prose,
                        );
                    }
                }
                if waiting {
                    break;
                }
                if unmatched {
                    self.prose.clear();
                } else if length > 0 {
                    continue;
                }
            }

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
                let invoke_start = self.active_invoke_start();

                // Close the block once no more complete invokes precede its end.
                if !self.block_is_invoke()
                    && let Some((end_pos, end_len)) =
                        find_first(&self.buffer, &self.spec.block_ends)
                {
                    let invoke_before_end = invoke_start.is_some_and(|start| start < end_pos);
                    if !invoke_before_end {
                        // Complete block fully closed: drop its markup and resume
                        // keeping natural text (inter-block / trailing). Any later
                        // block re-enters `in_block` and re-suppresses its markup.
                        // Matches the v1 batch parsers (cases 8.b/8.c/8.d).
                        self.consume_buffer_prefix(end_pos + end_len);
                        self.reset_invoke_boundary();
                        self.clear_uncommitted_block();
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
                        self.consume_buffer_prefix(self.buffer.len());
                        self.clear_uncommitted_block();
                        self.in_block = false;
                    }
                    break;
                };
                if start > 0 {
                    if self.pending_parse_error.is_none() {
                        self.uncommitted_block.push_str(&self.buffer[..start]);
                    }
                    self.consume_buffer_prefix(start);
                    self.reset_invoke_boundary();
                }
                let Some(end) = self.invoke_end_at(flush) else {
                    if !flush
                        && let Some(delta) = self
                            .emitter
                            .parse_partial_invoke(&self.buffer, self.next_index)?
                    {
                        out.push_call(delta);
                        continue;
                    }
                    if let Some(next_block) = self.resync_block_start(flush) {
                        tracing::warn!(
                            why = %format!("{}_resynchronized_after_incomplete_invoke", self.spec.family),
                            skipped_bytes = next_block,
                            "stream skipped a malformed invoke and resumed at the next complete block"
                        );
                        self.consume_buffer_prefix(next_block);
                        self.reset_invoke_boundary();
                        self.clear_uncommitted_block();
                        self.in_block = false;
                        self.suppress_normal_text = false;
                        self.in_reasoning = std::mem::take(&mut self.resume_reasoning);
                        continue;
                    }
                    if flush {
                        // Reviewer-caught regression: `invoke_end_at`
                        // returning `None` at flush does NOT always mean
                        // "genuinely incomplete, nothing left to salvage".
                        // Kimi's own "mismatched fences" guard (in
                        // `kimi_invoke_end`) also returns `None` when a
                        // REAL block-end marker follows a call that never
                        // got its own closer -- deliberately dropping just
                        // that call while the section boundary itself is
                        // still genuine. Unconditionally clearing the whole
                        // buffer here discarded that boundary too, along
                        // with any visible text genuinely following it --
                        // batch mode's regex-based section extraction
                        // preserves that trailing narration; streaming
                        // didn't. Same string-aware search as the
                        // `drop_invoke_crossing_block_end` check above
                        // (safe for the identical reason): if a real
                        // block-end marker is present, drop through it and
                        // resume draining the suffix as ordinary text
                        // instead of discarding everything after it.
                        if self.spec.drop_invoke_crossing_block_end
                            && let Some((be_pos, be_len)) = find_first_outside_strings(
                                &self.buffer,
                                self.spec.block_ends.iter().map(String::as_str),
                            )
                        {
                            tracing::warn!(
                                why = %format!("{}_incomplete_invoke", self.spec.family),
                                "stream dropped invoke with no evidence it ever closed before the block end"
                            );
                            self.consume_buffer_prefix(be_pos + be_len);
                            self.reset_invoke_boundary();
                            self.clear_uncommitted_block();
                            self.in_block = false;
                            self.suppress_normal_text = false;
                            self.in_reasoning = std::mem::take(&mut self.resume_reasoning);
                            continue;
                        }
                        tracing::warn!(
                            why = %format!("{}_incomplete_invoke", self.spec.family),
                            "stream dropped incomplete invoke at EOF"
                        );
                        self.consume_buffer_prefix(self.buffer.len());
                        self.clear_uncommitted_block();
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
                    && let Some((be_pos, be_len)) = find_first_outside_strings(
                        &self.buffer,
                        self.spec.block_ends.iter().map(String::as_str),
                    )
                    && be_pos < end
                {
                    tracing::warn!(
                        why = %format!("{}_incomplete_invoke", self.spec.family),
                        "stream dropped invoke missing its close before the block end"
                    );
                    self.consume_buffer_prefix(be_pos + be_len);
                    self.clear_uncommitted_block();
                    self.in_block = false;
                    self.suppress_normal_text = false;
                    continue;
                }
                let invoke = self.buffer[..end].to_string();
                // Emit BEFORE consuming. `parse_invoke` is fallible, and draining first
                // meant a failing emitter destroyed the invoke bytes: they were gone from
                // the buffer, so `reset` could no longer hand them back and the
                // documented recovery contract was false for the only shipped family.
                let emitted = self.parse_complete_invoke(&invoke, self.next_index)?;
                self.consume_buffer_prefix(end);
                self.reset_invoke_boundary();
                if let Some(deltas) = emitted {
                    let emitted_call = !deltas.is_empty();
                    for delta in deltas {
                        out.push_call(delta);
                    }
                    self.next_index += 1;
                    self.clear_uncommitted_block();
                    if emitted_call && self.spec.invoke_latch == InvokeLatch::IfEmitted {
                        self.suppress_normal_text = true;
                    }
                } else {
                    if self.pending_parse_error.is_none() {
                        self.uncommitted_block.push_str(&invoke);
                    }
                }
                if self.spec.invoke_latch == InvokeLatch::Always {
                    self.suppress_normal_text = true;
                }
                if self.invoke_closes_block() {
                    self.clear_uncommitted_block();
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
                    self.prose
                        .consume_recovery(&self.buffer[..pos], &self.spec.invoke_start);
                    self.prose.clear();
                    self.consume_buffer_prefix(pos + len);
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
                    self.prose
                        .consume_recovery(&self.buffer[..emit_len], &self.spec.invoke_start);
                    self.consume_buffer_prefix(emit_len);
                }
                break;
            };

            if start > 0 {
                let emit_len = start - self.tool_separator_len_at(start);
                if !self.suppress_normal_text {
                    push_run(out, Kind::Text, &self.buffer[..emit_len]);
                }
                self.uncommitted_block
                    .push_str(&self.buffer[emit_len..start]);
                self.prose
                    .consume_recovery(&self.buffer[..start], &self.spec.invoke_start);
                self.consume_buffer_prefix(start);
            }

            self.prose.clear();
            match marker {
                Marker::Block(blen) => {
                    self.uncommitted_block.push_str(&self.buffer[..blen]);
                    self.consume_buffer_prefix(blen);
                    self.reset_invoke_boundary();
                    self.in_block = true;
                    self.suppress_normal_text = true;
                }
                Marker::ReasoningStart(rlen) => {
                    self.consume_buffer_prefix(rlen);
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
                    // A bare invoke already lacks its opener, so it may recover
                    // only after a real closer arrives. Passing `flush` here used
                    // to combine missing-start and missing-end recovery and turn
                    // narrated syntax into a dispatched call.
                    let use_eof_boundary = flush
                        && self
                            .invoke_boundary
                            .as_ref()
                            .is_some_and(|boundary| boundary.bare_invoke_uses_eof_boundary());
                    let Some(end) = self.invoke_end_at(use_eof_boundary) else {
                        if !flush
                            && let Some(delta) = self
                                .emitter
                                .parse_partial_invoke(&self.buffer, self.next_index)?
                        {
                            out.push_call(delta);
                            continue;
                        }
                        if flush {
                            tracing::warn!(
                                why = %format!("{}_incomplete_bare_invoke", self.spec.family),
                                "stream dropped incomplete bare invoke at EOF"
                            );
                            self.consume_buffer_prefix(self.buffer.len());
                            self.clear_uncommitted_block();
                        }
                        break;
                    };
                    let invoke = self.buffer[..end].to_string();
                    // Emit before consuming — same recovery contract as the wrapped site.
                    let emitted = self.parse_complete_invoke(&invoke, self.next_index)?;
                    self.consume_buffer_prefix(end);
                    self.clear_uncommitted_block();
                    self.reset_invoke_boundary();
                    if let Some(deltas) = emitted {
                        tracing::warn!(
                            why = %format!("{}_bare_invoke_recovery", self.spec.family),
                            tool_index = self.next_index,
                            "stream recovered a complete bare invoke"
                        );
                        for delta in deltas {
                            out.push_call(delta);
                        }
                        self.next_index += 1;
                        self.clear_uncommitted_block();
                        self.suppress_normal_text =
                            self.spec.bare_recovery_latch == BareRecoveryLatch::Set;
                    } else if self.pending_parse_error.is_some() {
                        self.uncommitted_block.push_str(&invoke);
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
                complete: true,
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
                invoke_boundary_factory: None,
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
    fn speculative_prose_quote_tail_advances_once_and_clear_releases_state() {
        let mut prose = ProseControlState::default();
        prose.consume("Quoted \"");
        let mut input = String::from("<control>");
        for _ in 0..4096 {
            input.push('x');
            assert!(matches!(
                prose.classify(&input, ["<control>"], false, |_, _, _| {
                    ProseGrammarSpan::NotOwned
                }),
                ProseControl::Pending(0)
            ));
        }
        input.push('"');
        assert!(
            matches!(prose.classify(&input, ["<control>"], false, |_, _, _| ProseGrammarSpan::NotOwned), ProseControl::Literal(length) if length == input.len())
        );
        assert_eq!(prose.examined_bytes, input.len());
        prose.clear();
        assert!(matches!(
            prose.classify("<control>", ["<control>"], true, |_, _, _| {
                ProseGrammarSpan::NotOwned
            }),
            ProseControl::Ordinary
        ));
    }

    #[test]
    fn invoke_boundary_factory_constructs_and_dispatches_stateless_callbacks() {
        fn end(text: &str, flush: bool, tool_index: usize) -> Option<usize> {
            (flush && tool_index == 2).then_some(text.len())
        }
        fn opens(text: &str, at: usize) -> bool {
            text[at..].starts_with("open")
        }
        fn holdback(text: &str) -> usize {
            text.len().min(3)
        }
        fn resync(text: &str, tool_index: usize) -> Option<usize> {
            (tool_index == 2).then(|| text.find("resume")).flatten()
        }

        let factory = InvokeBoundaryFactory::stateless(InvokeScan {
            end,
            opens,
            holdback,
            resync: Some(resync),
        });
        let mut boundary = factory.create();

        assert_eq!(boundary.end_append("call", "call", false, 2), None);
        assert_eq!(boundary.end_append("call", "", true, 2), Some(4));
        assert!(boundary.opens("open-call", 0));
        assert_eq!(boundary.holdback("abcdef"), 3);
        assert_eq!(boundary.resync("skip resume", false, 2), Some(5));
    }

    #[test]
    fn marker_only_scanner_has_no_invoke_boundary_factory() {
        assert!(failing_scanner().invoke_boundary_factory().is_none());
    }

    /// An emitter without per-recipient framing never replaces the factory boundary.
    #[test]
    fn emitter_channel_boundary_defaults_to_none() {
        let scanner = failing_scanner();
        assert!(scanner.emitter.channel_boundary("any").is_none());
        assert!(scanner.invoke_boundary.is_none());
    }

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
        let markers = ["CLOSE"];
        // A marker-looking sequence entirely inside a string is data, not
        // structure -- must be skipped in favor of the real one afterward.
        assert_eq!(
            find_first_outside_strings(r#"{"a":"has CLOSE inside"}CLOSE"#, markers,),
            Some((24, 5)),
        );
    }

    #[test]
    fn find_first_outside_strings_handles_an_escaped_quote_next_to_the_marker() {
        let markers = ["CLOSE"];
        // `\"` immediately before the embedded marker must not be misread as
        // the string's real closing quote (which would wrongly un-toggle
        // `in_string` one byte early and expose the embedded marker).
        assert_eq!(
            find_first_outside_strings(r#"{"a":"ends with \" then CLOSE inside"}CLOSE"#, markers,),
            Some((38, 5)),
        );
    }

    #[test]
    fn find_first_outside_strings_returns_the_earliest_of_multiple_markers() {
        let markers = ["BB", "A"];
        // Earliest BYTE POSITION wins regardless of which marker string it
        // belongs to or the order markers are listed in.
        assert_eq!(
            find_first_outside_strings("xxAxxBBxx", markers),
            Some((2, 1))
        );
    }

    #[test]
    fn find_first_outside_strings_on_empty_or_no_match_is_none() {
        let markers = ["CLOSE"];
        assert_eq!(find_first_outside_strings("", markers), None);
        assert_eq!(find_first_outside_strings("no marker here", markers), None);
        assert_eq!(
            find_first_outside_strings(r#""CLOSE only inside a string""#, markers,),
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

    #[test]
    fn resumable_json_boundary_settles_once_after_many_sibling_fields() {
        for count in [1, 100, 800] {
            let fields = (0..count)
                .map(|i| format!("\"k{i}\":\"é[]{{}}\\\\\\\"\""))
                .collect::<Vec<_>>()
                .join(",");
            let value = format!(" \t{{\"marker\":\"<|call|>\",{fields}}}");
            assert!(serde_json::from_str::<serde_json::Value>(&value).is_ok());
            let mut boundary = super::JsonValueBoundary::default();
            let completions = value
                .char_indices()
                .filter_map(|(at, ch)| boundary.advance(ch).then_some(at + ch.len_utf8()))
                .collect::<Vec<_>>();
            assert_eq!(completions, vec![value.len()]);
            assert_eq!(super::json_value_end(&value), Some(value.len()));
            assert!(
                " trailing {\"more\":[]}"
                    .chars()
                    .all(|ch| !boundary.advance(ch))
            );
        }
    }

    #[test]
    fn mismatched_json_boundary_never_scans_a_growing_suffix() {
        for suffix in [0, 100, 8192] {
            for prefix in ["{]", "[}"] {
                let input = format!("{prefix}{}", "é{}[]".repeat(suffix));
                let seen = std::cell::Cell::new(0usize);
                let result = super::json_container_end(
                    input.char_indices().inspect(|_| seen.set(seen.get() + 1)),
                );
                assert_eq!(result, None);
                assert_eq!(seen.get(), 2, "suffix={suffix}");
            }
        }
    }
}
