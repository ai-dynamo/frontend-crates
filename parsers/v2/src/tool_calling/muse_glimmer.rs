// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Muse Glimmer channel-routed streaming parser (ATEM tool calls + reasoning).
//!
//! Grammar (the `response_template` in the model's `tokenizer_config.json`):
//!
//! ```text
//! <|start|>assistant to=self<|message|>...reasoning...<|eom|>
//! <|start|>assistant to=get_weather<|message|>
//! <atem:function_calls><atem:invoke name="get_weather">
//! <atem:parameter name="city">Paris</atem:parameter>
//! </atem:invoke></atem:function_calls><|eom|>
//! <|start|>assistant to=user<|message|>...final answer...<|eot|>
//! ```
//!
//! `self` routes to reasoning, `user`/no-recipient to content, anything else opens a
//! tool channel of ATEM XML. `<|eom|>` continues the turn, `<|eot|>` ends it. The prompt
//! ends with `<|start|>assistant`, so a turn's first message arrives header-less.
//!
//! Own scanner, not `WrappedBlockScanner`: the channel is decided by the dynamic header
//! content, not a fixed marker pair. One [`MuseChannelScanner`] drives both surfaces
//! ([`MuseGlimmerToolStreamParser`] projects its deltas; `crate::unified::muse_glimmer`
//! takes them in order). Safety: invokes are extracted ONLY from a tool-recipient body,
//! so ATEM quoted in a reasoning or content body never parses as a call.

use std::sync::OnceLock;

use regex::Regex;

use crate::tool_calling::scan::{
    InvokeEmitter, marker_prefix_suffix_len, push_run, reorder_arguments,
};
use crate::tool_calling::traits::{Tool, ToolCallDelta, ToolParseResult, ToolParser};
use crate::tool_calling::v1core::ToolDefinition;
use crate::unified::{GuidedChannelState, Kind, UnifiedParserEvent, UnifiedParserStartingState};

const START: &str = "<|start|>";
const MESSAGE: &str = "<|message|>";
const EOM: &str = "<|eom|>";
const EOT: &str = "<|eot|>";
const INVOKE_OPEN_PREFIX: &str = "<atem:invoke";
const INVOKE_CLOSE: &str = "</atem:invoke>";
const REASONING_RECIPIENT: &str = "self";
const USER_RECIPIENT: &str = "user";

/// Structural markers held back when split across a chunk boundary.
const MARKERS: [&str; 4] = [START, MESSAGE, EOM, EOT];

/// The ATEM tool BLOCK pair, which wraps a tool channel's invokes.
const BLOCK_OPEN: &str = "<atem:function_calls>";
const BLOCK_CLOSE: &str = "</atem:function_calls>";

// The ATEM block pair is deliberately NOT stripped from visible runs. Outside a
// tool channel this family treats ATEM as ORDINARY TEXT — `unframed_atem_does_not_parse`
// and `quoted_atem_in_content_is_not_a_call` pin that, and it is the safety rule that
// keeps a model quoting ATEM in prose from being read as a call. Only the guided
// reader strips it, because guided decoding constrains the payload to bare JSON and
// leaves any native markup around it stray by construction.

/// One complete ATEM invoke open tag, capturing the tool name. `[^>]*?` ends the tag at
/// the first `>`, so only a `>` inside a non-`name` attribute value (which the template
/// never writes) misparses.
fn invoke_open_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r#"<atem:invoke\b[^>]*?\bname="(?P<name>[^"]+)"[^>]*?>"#).unwrap())
}

/// One complete ATEM parameter element, capturing key + raw value, under the same `>`
/// bound as `invoke_open_re`.
fn parameter_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r#"(?s)<atem:parameter\b[^>]*?\bname="(?P<key>[^"]+)"[^>]*?>(?P<value>.*?)</atem:parameter>"#,
        )
        .unwrap()
    })
}

/// The channel the scanner is currently inside.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// Between messages (or before the first). Prose here surfaces as content
    /// with orphan framing stripped.
    Idle,
    InReasoning,
    InContent,
    InToolChannel,
}

/// A recipient is a run of non-whitespace, non-`<` characters (SGLang's
/// `to=([^\s<]+)`; broader than vLLM's `[A-Za-z0-9_.\-]+` so namespaced and
/// unicode tool names survive).
fn is_recipient_char(c: char) -> bool {
    !c.is_whitespace() && c != '<'
}

/// Length of the leading run of recipient characters in `s`.
fn recipient_run_len(s: &str) -> usize {
    s.char_indices()
        .take_while(|(_, c)| is_recipient_char(*c))
        .map(|(i, c)| i + c.len_utf8())
        .last()
        .unwrap_or(0)
}

/// Resolve the header that ends at the `<|message|>` found at `msg_pos`.
///
/// Returns `(header_start, recipient)`. The recipient must immediately abut
/// `<|message|>` (vLLM's anchoring); an optional `<|start|>assistant` prefix and
/// the non-newline whitespace between prompt framing and `to=` are absorbed into
/// the header.
fn resolve_header(text: &str, msg_pos: usize) -> (usize, Option<&str>) {
    let before = &text[..msg_pos];

    // Maximal recipient-charactered run ending at `<|message|>`.
    let run_start = before
        .char_indices()
        .rev()
        .take_while(|(_, c)| is_recipient_char(*c))
        .last()
        .map(|(i, _)| i)
        .unwrap_or(msg_pos);
    let run = &before[run_start..];

    let (mut header_start, recipient) = match run.find("to=") {
        Some(rel) => {
            let recipient = &run[rel + 3..];
            if recipient.is_empty() {
                (msg_pos, None)
            } else {
                (run_start + rel, Some(recipient))
            }
        }
        None => (msg_pos, None),
    };

    // Absorb `<|start|>assistant` (with the whitespace the template renders
    // between its parts) into the header so it never leaks as body text. The role
    // word only counts when `<|start|>` really precedes it; prose that happens to
    // end in "assistant" stays prose and takes the bare-header path below.
    let ws_start = |s: &str| s.len() - s.trim_end_matches(|c: char| c.is_whitespace()).len();
    let pre = &text[..header_start];
    let pre_trimmed = &pre[..pre.len() - ws_start(pre)];
    let framed_start = match pre_trimmed.strip_suffix("assistant") {
        Some(stripped) => {
            let stripped_trimmed = &stripped[..stripped.len() - ws_start(stripped)];
            stripped_trimmed
                .ends_with(START)
                .then(|| stripped_trimmed.len() - START.len())
        }
        // `<|start|><|message|>` / `<|start|>to=...` without the role word.
        None => pre_trimmed
            .ends_with(START)
            .then(|| pre_trimmed.len() - START.len()),
    };
    if let Some(start) = framed_start {
        header_start = start;
    } else if recipient.is_some() {
        // A bare `to=` header (the prompt consumed `<|start|>assistant`): absorb
        // the template's separating space so it does not leak.
        let gap = &text[..header_start];
        let gap_ws = gap.len() - gap.trim_end_matches([' ', '\t']).len();
        header_start -= gap_ws;
    }

    (header_start, recipient)
}

/// Position of the first `to=<recipient><|message|>` run inside `body`, anchored to a
/// body start or whitespace. Bounds the missing-`<|eom|>` recovery heuristic so an
/// unanchored `pota`|`to=...` split cannot promote prose into a tool channel. Resolution
/// at real message boundaries stays unanchored, like both engines.
fn bare_header_pos(body: &str, prev: Option<char>) -> Option<usize> {
    let mut search = 0;
    while let Some(rel) = body[search..].find("to=") {
        let at = search + rel;
        let anchored = if at == 0 {
            // `prev` is the character right before `body`: None at a real body
            // start (which is an anchor), else the last drained byte of the same
            // body, so a mid-word `to=` split by a chunk boundary stays
            // unanchored.
            prev.is_none_or(|c| c.is_whitespace())
        } else {
            body[..at]
                .chars()
                .next_back()
                .is_some_and(|c| c.is_whitespace())
        };
        let after = &body[at + 3..];
        let rcpt_len = recipient_run_len(after);
        if anchored && rcpt_len > 0 && after[rcpt_len..].starts_with(MESSAGE) {
            return Some(at);
        }
        // Resume past the whole recipient run: every `to=` inside it reads the same
        // tail and fails the same check, and none of its characters can anchor. Skipping
        // the run (not three bytes) keeps this linear on a `to=to=to=` body.
        search = at + 3 + rcpt_len;
    }
    None
}

/// Length of a trailing fragment that could still grow into a bare header: a `t`/`to`/`to=`
/// prefix, a recipient run, and a partial `<|message|>`, plus the leading space run that
/// `resolve_header` absorbs. Held back so a recipient never leaks into an open body.
fn open_header_tail(s: &str) -> usize {
    let tail = match first_header_start(s) {
        Some(pos) => s.len() - pos,
        _ if s.ends_with("to") => 2,
        _ if s.ends_with('t') => 1,
        _ => 0,
    };
    // Extend over the whitespace run a resolved bare header would absorb.
    let before = &s[..s.len() - tail];
    let ws = before.len() - before.trim_end_matches([' ', '\t']).len();
    tail + ws
}

/// Start of the EARLIEST `to=` the open tail could resolve from. `resolve_header` reads
/// the FIRST `to=` abutting `<|message|>`, so scanning from the last one holds too little:
/// `to=to=<|message|>` would release the leading `to=` as prose and drop a call.
fn first_header_start(s: &str) -> Option<usize> {
    let mut at = 0;
    while let Some(rel) = s[at..].find("to=") {
        let pos = at + rel;
        let after = &s[pos + 3..];
        if valid_header_fragment(after) {
            return Some(pos);
        }
        // Same linear skip `bare_header_pos` makes: every `to=` inside this
        // recipient run reads the same tail, so it fails the same check.
        at = pos + 3 + recipient_run_len(after);
    }
    None
}

/// Whether the bytes after a trailing `to=` could still complete a header:
/// recipient characters followed by nothing or a partial `<|message|>`. Complete
/// headers are handled by the boundary scan, not held.
fn valid_header_fragment(after: &str) -> bool {
    let rest = &after[recipient_run_len(after)..];
    rest.is_empty() || (rest.len() < MESSAGE.len() && MESSAGE.starts_with(rest))
}

/// The markers the guided reader may strip ON THEIR OWN, without reading anything
/// around them.
///
/// `<|start|>` and `<|message|>` are deliberately ABSENT even though they are this
/// grammar's most common markup. They are header CONSTITUENTS, not standalone
/// markers: stripping `<|start|>` the moment it arrives consumes the very byte
/// [`resolve_header`] looks back for, so the role word and the recipient behind it
/// are released as visible text and the thought then opens at the wrong offset. A
/// header is consumed whole by [`guided_reasoning_open`] or [`guided_stray_header`]
/// instead. `<|eom|>` and `<|eot|>` stay here because they really are standalone —
/// nothing is read around them to decide what they mean.
///
/// The ATEM entries matter for the guided cases where native markup BRACKETS a
/// constrained payload: a `<atem:function_calls>` that is not stripped enters the
/// JSON buffer, breaks the parse and costs the call.
pub(crate) const GUIDED_CONTROL_MARKERS: [&str; 5] = [
    EOM,
    EOT,
    BLOCK_OPEN,
    BLOCK_CLOSE,
    // The PREFIX FORM of the invoke opener: it introduces the tool name, which runs
    // to the tag's `>`. Declared through the name's opening quote rather than as the
    // bare `<atem:invoke`, so an opener that is never terminated is stripped whole
    // instead of leaving ` name="` behind to poison the payload.
    INVOKE_OPEN_HEADER,
];

// `</atem:invoke>` is deliberately NOT in that set, exactly as qwen3 omits its own
// `</function>`. The invoke CLOSER is supplied separately as the grammar's
// `invoke_end`, and the opener consumes through it to take the whole invoke —
// parameter elements and all — in one strip. Listing it here made it bound the
// opener's own terminator search, so the pair was cut at the header and the
// `<atem:parameter …>` body between them was emitted to the user as text.

/// The invoke opener through the tool name's opening quote.
const INVOKE_OPEN_HEADER: &str = "<atem:invoke name=\"";

/// The framing markers that compete with tool syntax for a terminator, and whose
/// split prefixes are held back.
///
/// Only the FIXED bytes of a header can compete; the recipient inside one is data
/// whose length the grammar does not bound.
pub(crate) const GUIDED_COMPETITORS: [&str; 4] = MARKERS;

/// The markers that CLOSE a message, and so close a thought.
pub(crate) const GUIDED_CLOSE_MARKERS: [&str; 2] = [EOM, EOT];

/// Earliest header in `haystack` whose recipient satisfies `want`.
///
/// Routes through [`resolve_header`], the SAME owner the native scan uses, so the
/// guided path and the native path cannot disagree about where a header begins or
/// how much of it is framing. A fixed opener string would be wrong in both
/// directions: it would miss the bare `to=self<|message|>` form this family accepts
/// when the prompt consumed `<|start|>assistant`, and it would read a `to=user`
/// content header as a thought.
///
/// Returns `(header_start, bytes through `<|message|>`)`. Whole headers only — an
/// incomplete one is retained by [`guided_header_holdback`] instead, so a header is
/// never half-consumed and never released as text.
/// Resolve the header ending at `msg_pos`, applying the QUOTED-BARE-HEADER demotion.
///
/// The one place that rule lives, consulted by the native scan and the guided reader
/// alike. An unframed header carrying a recipient, seen after the turn has already
/// been routed, is text the model QUOTED: the `to=…` words are prose and only
/// `<|message|>` is structural, so it demotes to a recipient-less content header
/// starting at the marker. The native scan applied this inline while the guided hooks
/// did not, so `I mean to=self<|message|>literal` inside a `to=user` answer opened a
/// real thought under guided decoding and stayed visible text natively — the same
/// bytes read two ways by request mode (`I3`).
pub(crate) fn resolve_header_latched(
    text: &str,
    msg_pos: usize,
    allow_bare_header: bool,
) -> (usize, Option<&str>) {
    let (header_start, recipient) = resolve_header(text, msg_pos);
    // Whether `<|start|>` really opens this header. It is also what separates a
    // recipient-less HEADER from a bare `<|message|>` written mid-body.
    let framed = text[header_start..].starts_with(START);
    if !framed && !allow_bare_header && recipient.is_some() {
        (msg_pos, None)
    } else {
        (header_start, recipient)
    }
}

fn guided_header(
    haystack: &str,
    state: GuidedChannelState,
    want: fn(recipient: Option<&str>, framed: bool) -> bool,
) -> Option<(usize, usize)> {
    let mut search = 0;
    while let Some(relative) = haystack[search..].find(MESSAGE) {
        let msg_pos = search + relative;
        let (header_start, recipient) =
            resolve_header_latched(haystack, msg_pos, state.scope.allows_bare_header());
        let framed = haystack[header_start..].starts_with(START);
        if want(recipient, framed) {
            return Some((header_start, msg_pos + MESSAGE.len() - header_start));
        }
        search = msg_pos + MESSAGE.len();
    }
    None
}

/// Earliest REASONING header: the one that OPENS a thought.
///
/// `flush` is unused because a header is recognised only once its `<|message|>` has
/// arrived; at end of stream a header that never completed is not a header.
pub(crate) fn guided_reasoning_open(
    haystack: &str,
    _flush: bool,
    state: GuidedChannelState,
) -> Option<(usize, usize)> {
    guided_header(haystack, state, |recipient, _framed| {
        recipient == Some(REASONING_RECIPIENT)
    })
}

/// Whether a header routes to the VISIBLE content channel.
///
/// `to=user` always does. A recipient-LESS header does only when `<|start|>` frames
/// it: an unframed bare `<|message|>` is not a header at all, it is a stray marker
/// the model emitted inside whatever channel is already open. Reading every
/// recipient-less `<|message|>` as a content header ENDED an open thought on that
/// marker, so `…<|message|>still thinking` split one thought into a thought plus a
/// visible answer, while the native scan strips the marker and keeps thinking.
fn is_content_header(recipient: Option<&str>, framed: bool) -> bool {
    match recipient {
        Some(USER_RECIPIENT) => true,
        None => framed,
        _ => false,
    }
}

/// Earliest header that routes to VISIBLE CONTENT, which ENDS an open thought.
///
/// Separate from [`guided_stray_header`] because guided decoding constrains only the
/// TOOL channel. A `to=user` header is a real channel switch whatever the request
/// mode, and folding it into strippable markup made the model's visible answer come
/// out as its private thinking. A `to=<tool>` header under guided decoding cannot be
/// a real tool channel — the call arrives as JSON — so that one really is narration.
pub(crate) fn guided_content_header(
    haystack: &str,
    _flush: bool,
    state: GuidedChannelState,
) -> Option<(usize, usize)> {
    guided_header(haystack, state, is_content_header)
}

/// Earliest header that actually ROUTES the turn — one that names a channel.
///
/// An orphan recipient-less `<|message|>` is control markup, not a routing decision:
/// stripping it says nothing about where the turn is going. Counting it as a header
/// spent the turn's routing scope, and the next REAL bare header was then demoted and
/// its recipient words leaked into the visible answer.
///
/// "Names a channel" is `to=<recipient>`, or framing that stands in for one — the same
/// two things `resolve_header` treats as structural.
pub(crate) fn guided_routing_header(
    haystack: &str,
    _flush: bool,
    state: GuidedChannelState,
) -> Option<(usize, usize)> {
    guided_header(haystack, state, |recipient, framed| {
        recipient.is_some() || framed
    })
}

/// Earliest header that is neither a thought nor a channel switch: markup to strip.
///
/// Under guided decoding a `to=NAME` tool header wraps a payload that does not need
/// it. It has to be consumed WHOLE. Listing `<|start|>` and `<|message|>` as ordinary
/// control markers is not enough — that strips the two markers and releases the role
/// word and the recipient between them as visible text, so the user reads
/// `assistant to=weather` as the model's answer.
pub(crate) fn guided_stray_header(
    haystack: &str,
    flush: bool,
    state: GuidedChannelState,
) -> Option<(usize, usize)> {
    if let Some(found) = guided_header(haystack, state, |recipient, framed| {
        recipient != Some(REASONING_RECIPIENT) && !is_content_header(recipient, framed)
    }) {
        return Some(found);
    }
    // At end of stream a `<|start|>` whose `<|message|>` never arrived can no longer
    // become a header. It is committed parser-owned markup and is dropped rather than
    // shown to the user, which is the same judgement `flush_open_text` makes for the
    // native path; anything after it is ordinary prose and stays visible.
    if flush {
        return haystack.find(START).map(|at| (at, START.len()));
    }
    None
}

/// Earliest message terminator in `haystack`: `(offset, bytes to consume)`.
///
/// Both terminators close a thought. `<|eot|>` ends the whole turn while `<|eom|>`
/// continues it, but the guided reader's question is only "is this span over", and
/// answering it differently for the two would leave a thought open past the end of
/// the turn and emit the payload that follows as reasoning.
pub(crate) fn guided_reasoning_close(haystack: &str) -> Option<(usize, usize)> {
    GUIDED_CLOSE_MARKERS
        .iter()
        .filter_map(|marker| haystack.find(marker).map(|at| (at, marker.len())))
        .min_by_key(|(at, _)| *at)
}

/// The TERMINAL closer: `<|eot|>` ends the turn, where `<|eom|>` only ends a message.
///
/// The two are not interchangeable. After `<|eom|>` a later message may route the turn
/// again, so a guided payload behind it is still a payload. After `<|eot|>` there is no
/// later message, so those same bytes are trailing text and must never dispatch a call.
pub(crate) fn guided_turn_end(haystack: &str) -> Option<(usize, usize)> {
    haystack.find(EOT).map(|at| (at, EOT.len()))
}

/// Trailing bytes the guided reader must retain so a header split across a chunk
/// boundary is never flushed as text or into the payload buffer.
///
/// Two ways a header can be mid-arrival, and holding only the second is what made
/// the same bytes parse differently whole than one character at a time:
///
/// 1. A COMPLETE `<|start|>` whose `<|message|>` has not arrived yet. Everything
///    after it — the role word, the spacing, the `to=` and the recipient — is header
///    bytes. Releasing them emitted `assistant` as visible text, and by the time
///    `<|message|>` arrived [`resolve_header`] could no longer see the `<|start|>`
///    it needed to look back at, so the thought opened at the wrong offset.
/// 2. A trailing `to=`-shaped fragment with no framing, which is the bare-header
///    form. [`open_header_tail`] is the same helper the native scan holds bodies
///    with, so a recipient cannot leak on one path and be held on the other.
///
/// A PARTIAL `<|start|>` (`<|sta`) needs no case here: it is a split marker, and the
/// shared holdback already retains every declared marker's prefix.
/// Remove this grammar's framing from a run the guided reader is about to show.
///
/// The SAME `stripped()` the native path applies, so a marker cannot leak on one
/// request mode and be stripped on the other. It matters where a header only
/// partly resolves: `<|start|>wrong-role to=self<|message|>` opens a real thought,
/// but `<|start|>wrong-role` is not part of the header and would otherwise reach
/// the user with the control marker still attached.
pub(crate) fn guided_strip_text(text: &str) -> String {
    stripped(text)
}

pub(crate) fn guided_header_holdback(haystack: &str, state: GuidedChannelState) -> usize {
    if let Some(at) = haystack.rfind(START)
        && !haystack[at..].contains(MESSAGE)
    {
        return haystack.len() - at;
    }
    // A COMPLETE header with nothing after it yet is also unfinished business: what
    // follows decides whether a tool-routed header is narration or the recovery point
    // for a thought whose terminator never arrived. Releasing it before that byte
    // arrives made a per-character stream answer differently from a whole-input push.
    if let Some((at, len)) = guided_stray_header(haystack, true, state)
        && haystack[at + len..].trim().is_empty()
    {
        return haystack.len() - at;
    }
    // Once the turn is ROUTED, a trailing bare-looking `to=` fragment can no longer
    // become a header — it is prose, and holding it back delays bytes whose meaning is
    // already unambiguous. Framed and partial-marker holdback above still applies,
    // because those really can still complete.
    if !state.scope.allows_bare_header() {
        return 0;
    }
    open_header_tail(haystack)
}

/// Visible text with orphan framing markers removed so they never reach the
/// client (`I3`).
fn stripped(text: &str) -> String {
    // Removing one marker SPLICES its neighbours together, and the splice can spell
    // another marker (`<|st` + `<|eom|>` + `art|>` leaves `<|start|>`). Append one char
    // at a time and retract whatever marker the append completes, so a spliced one goes
    // the moment it forms. Every marker ends `>`, so only that char can complete one and
    // a single retract per char is enough.
    let mut cleaned = String::with_capacity(text.len());
    for ch in text.chars() {
        cleaned.push(ch);
        if ch != '>' {
            continue;
        }
        if let Some(marker) = MARKERS.iter().find(|m| cleaned.ends_with(**m)) {
            cleaned.truncate(cleaned.len() - marker.len());
        }
    }
    cleaned
}

/// At end of stream, drop a committed partial special token from the tail of held
/// text but keep everything a human could have typed.
fn flush_open_text(buffered: &str) -> String {
    let tail = marker_prefix_suffix_len(buffered, MARKERS);
    if tail <= 2 {
        // `<` / `<|` are ordinary prose as often as framing; keep them.
        return buffered.to_string();
    }
    buffered[..buffered.len() - tail].to_string()
}

/// The ONE direct construction of a tool-call delta in this module, so the
/// ordered vocabulary is named in a single place.
fn emit_call(out: &mut Vec<UnifiedParserEvent>, delta: ToolCallDelta) {
    out.push(UnifiedParserEvent::ToolCall(delta));
}

/// Value-typing hook: types one complete `<atem:invoke ...></atem:invoke>` block
/// the scanner has already delimited, and re-orders the arguments to source
/// order.
pub(crate) struct MuseInvokeEmitter {
    tools: Vec<ToolDefinition>,
}

impl InvokeEmitter for MuseInvokeEmitter {
    fn parse_invoke(
        &mut self,
        invoke: &str,
        tool_index: usize,
    ) -> anyhow::Result<Option<ToolCallDelta>> {
        let Some(open) = invoke_open_re().captures(invoke) else {
            return Ok(None);
        };
        let whole = open.get(0).expect("regex match has group 0");
        let name_raw = open.name("name").expect("regex requires name").as_str();
        let body = invoke[whole.end()..]
            .strip_suffix(INVOKE_CLOSE)
            .unwrap_or_default();

        let mut arguments = serde_json::Map::new();
        let mut source_names = Vec::new();
        for param in parameter_re().captures_iter(body) {
            let key = param.name("key").expect("regex requires key").as_str();
            let raw = param.name("value").expect("regex requires value").as_str();
            arguments.insert(key.to_string(), decode_value(raw));
            source_names.push(key.to_string());
        }

        // `serde_json::Map` serializes alphabetically when `preserve_order` is off, so
        // restore the model-emitted order with the same helper the XML families use.
        // The call stays even though every build of THIS workspace has the feature on:
        // `openai-harmony` unifies it in, and whether that happens is a property of the
        // dependent's graph, not of this crate.
        let arguments = reorder_arguments(&serde_json::to_string(&arguments)?, &source_names);
        Ok(Some(ToolCallDelta {
            tool_index,
            name: Some(normalize_name(name_raw, &self.tools)),
            arguments,
            complete: true,
        }))
    }
}

/// JSON-decode a parameter value when possible, else keep the raw string — the
/// spec's `value_parser: json` with `allow_non_json: true`. The raw fallback is
/// byte-preserving (NOT trimmed): both engines keep surrounding whitespace in
/// non-JSON string values.
fn decode_value(raw: &str) -> serde_json::Value {
    serde_json::from_str(raw).unwrap_or_else(|_| serde_json::Value::String(raw.to_string()))
}

/// Collapse the chat template's doubled namespace (`get_weather.get_weather` ->
/// `get_weather`) when the collapsed name is a registered tool. Both engines do
/// this. Leaf-only matching (SGLang's extra step) is deliberately NOT done: an
/// emitted `weather.get` would silently dispatch a registered `calendar.get`.
/// Unknown names pass through unchanged with a warning (vLLM's policy).
fn normalize_name(emitted: &str, tools: &[ToolDefinition]) -> String {
    let registered: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
    if registered.is_empty() || registered.contains(&emitted) {
        return emitted.to_string();
    }
    if let Some((head, tail)) = emitted.split_once('.')
        && head == tail
        && registered.contains(&head)
    {
        return head.to_string();
    }
    tracing::warn!(
        emitted_name = emitted,
        "Muse Glimmer: emitted tool name does not match any registered tool; passing through unchanged"
    );
    emitted.to_string()
}

/// The channel-routing scan core: one state machine over the whole turn, emitting
/// reasoning, content and tool calls in the order the model produced them.
pub(crate) struct MuseChannelScanner {
    emitter: MuseInvokeEmitter,
    buffer: String,
    state: State,
    /// Whether the next header may resolve WITHOUT `<|start|>` framing. True at
    /// turn start (the prompt consumed `<|start|>assistant`) and after a reasoning
    /// body cut at a bare header (missing-`<|eom|>` recovery); a bare-looking
    /// header anywhere else is quoted text, and resolving it would promote content
    /// into a live tool channel.
    allow_bare_header: bool,
    /// Last character already drained from the OPEN body, so recovery anchoring
    /// survives chunk splits (None right after the header).
    last_body_char: Option<char>,
    /// A reasoning body closed and nothing else has been emitted since, so the
    /// NEXT reasoning body is adjacent to it and joins with a newline.
    pending_reasoning_join: bool,
    /// An adjacent reasoning body opened and OWES that newline. Held until the body
    /// produces bytes, so an EMPTY thought contributes nothing: emitting at the header
    /// made it add visible whitespace, against the empty-block contract.
    reasoning_join_armed: bool,
    /// The reasoning body currently open has emitted at least once. Tracked across the
    /// WHOLE body, not just its closing chunk: a body that streamed incrementally has
    /// nothing left at the terminator, and reading only that last emit made adjacency
    /// depend on where the chunk boundaries fell.
    reasoning_body_emitted: bool,
    /// Any byte has been fed. `initialize_request` is a BEFORE-parsing hook, so it
    /// must reject a late call rather than silently reinterpret a live stream.
    started: bool,
    next_index: usize,
}

/// Build the scan core for the Muse Glimmer grammar.
///
/// The single construction site: the tool-only parser and the unified factory
/// both call it, so the two surfaces cannot drift on routing, recovery or value
/// typing.
pub(crate) fn muse_scanner(tools: &[Tool]) -> MuseChannelScanner {
    MuseChannelScanner {
        emitter: MuseInvokeEmitter {
            tools: tools.iter().map(ToolDefinition::from).collect(),
        },
        buffer: String::new(),
        state: State::Idle,
        allow_bare_header: true,
        last_body_char: None,
        pending_reasoning_join: false,
        reasoning_join_armed: false,
        reasoning_body_emitted: false,
        started: false,
        next_index: 0,
    }
}

impl MuseChannelScanner {
    pub(crate) fn push_ordered(&mut self, chunk: &str) -> anyhow::Result<Vec<UnifiedParserEvent>> {
        let mut out = Vec::new();
        self.push_ordered_into(chunk, &mut out)?;
        Ok(out)
    }

    /// Scan `chunk`, appending committed events to a carry the caller keeps on `Err`.
    ///
    /// `push_ordered` cannot offer that: its `Result<Vec<_>>` drops whatever the drain
    /// had already committed before a failing invoke. `UnifiedParser::parse_into`
    /// promises those events survive, so it runs on this instead.
    pub(crate) fn push_ordered_into(
        &mut self,
        chunk: &str,
        out: &mut Vec<UnifiedParserEvent>,
    ) -> anyhow::Result<()> {
        self.started = true;
        self.buffer.push_str(chunk);
        self.drain(out)
    }

    /// Apply the channel state the prompt left this stream in, before any byte is
    /// parsed.
    ///
    /// `starting_state` is cheap for this grammar: the prompt consumed a channel header,
    /// so the stream simply opens in that channel instead of `Idle`. It is the whole
    /// reason `State` is an enum rather than a bool.
    ///
    /// Guided tool output used to be REJECTED here, on the ground that `GuidedState` was
    /// built around a `ReasoningSpec` — an open/close marker PAIR — and muse has none:
    /// reasoning opens with a dynamic `to=self<|message|>` header it shares with the
    /// content and tool channels. Having no marker pair is a statement about how the
    /// opener is SPELLED, not about whether a reasoning channel exists. Muse has one; it
    /// is routed by recipient. The guided reader now asks [`crate::unified::GuidedReasoning`]
    /// where a thought starts instead of assuming a fixed string, this family answers
    /// with [`guided_reasoning_open`], and there is nothing left to reject.
    pub(crate) fn apply_starting_state(&mut self, starting_state: UnifiedParserStartingState) {
        self.state = match starting_state {
            UnifiedParserStartingState::None => State::Idle,
            UnifiedParserStartingState::Reasoning => State::InReasoning,
            UnifiedParserStartingState::Response => State::InContent,
        };
        // A header the prompt already consumed cannot arrive again, so the turn-start
        // latch only stays armed when this stream really does begin at `Idle`.
        self.allow_bare_header = starting_state != UnifiedParserStartingState::Response;
        self.last_body_char = None;
    }

    pub(crate) fn finish_ordered(&mut self) -> anyhow::Result<Vec<UnifiedParserEvent>> {
        let mut out = Vec::new();
        self.flush(&mut out);
        Ok(out)
    }

    // Mirrors `WrappedBlockScanner::push`/`finish` so the tool-only projection
    // lives in ONE place per scanner type.
    pub(crate) fn push(&mut self, chunk: &str) -> anyhow::Result<ToolParseResult> {
        Ok(ToolParseResult::from_deltas(self.push_ordered(chunk)?))
    }

    pub(crate) fn finish(&mut self) -> anyhow::Result<ToolParseResult> {
        Ok(ToolParseResult::from_deltas(self.finish_ordered()?))
    }

    /// Emit a reasoning run: the ONE route into the reasoning channel.
    ///
    /// Strips orphan framing exactly as `emit_text` does. `I3` covers reasoning no less
    /// than text, but only the text route ran `stripped()`, so a complete orphan marker
    /// inside a thought reached the client verbatim on all three reasoning routes
    /// (closed body, open body, finish).
    ///
    /// The owed adjacency newline is paid HERE rather than at the header, so a thought
    /// that turns out empty contributes nothing at all.
    ///
    /// NOTE: this puts Dynamo AHEAD of both engines rather than level with them. vLLM's
    /// `_CHANNEL_HEADER_RE` needs a recipient, so a bare `<|message|>` never ends a body
    /// and its parser returns the marker intact; SGLang's `MuseGlimmerDetector` appends
    /// the body unstripped and pushes the separator on the header. Adopting the stricter
    /// reading is deliberate — the conformance suite will score these two cases as
    /// divergences until the engines follow.
    fn emit_reasoning(&mut self, out: &mut Vec<UnifiedParserEvent>, text: &str) -> bool {
        let text = stripped(text);
        if text.is_empty() {
            return false;
        }
        if std::mem::take(&mut self.reasoning_join_armed) {
            push_run(out, Kind::Reasoning, "\n");
        }
        push_run(out, Kind::Reasoning, &text);
        self.reasoning_body_emitted = true;
        true
    }

    /// Emit visible content, clearing the reasoning-join latch: a thought separated
    /// from the next by content is no longer adjacent to it. The strip runs on every
    /// route into text, content bodies included, so a quoted `to=x<|message|>` never
    /// leaks its marker.
    fn emit_text(&mut self, out: &mut Vec<UnifiedParserEvent>, text: &str) {
        let text = stripped(text);
        if text.is_empty() {
            return;
        }
        push_run(out, Kind::Text, &text);
        self.pending_reasoning_join = false;
        self.reasoning_join_armed = false;
    }

    /// Byte offset where the OPEN body ends, ignoring the bare-header recovery
    /// (which applies to reasoning only): the earliest terminator or framed
    /// header, else the buffer end.
    fn tool_body_limit(&self) -> usize {
        [EOM, EOT, START]
            .iter()
            .filter_map(|m| self.buffer.find(m))
            .min()
            .unwrap_or(self.buffer.len())
    }

    /// Bounds of the first COMPLETE `<atem:invoke ...>...</atem:invoke>` block
    /// inside `buffer[..limit]`. An invoke without its close is not complete: both
    /// engines require the literal close, so it waits for more input.
    fn next_invoke(&self, limit: usize) -> Option<(usize, usize)> {
        let body = &self.buffer[..limit];
        let open = invoke_open_re().captures(body)?;
        let whole = open.get(0).expect("regex match has group 0");
        let close = body[whole.end()..].find(INVOKE_CLOSE)?;
        Some((whole.start(), whole.end() + close + INVOKE_CLOSE.len()))
    }

    /// Drive the state machine over the buffer, appending routed deltas.
    ///
    /// Every state change `continue`s rather than breaking, so one push of a
    /// multi-channel delta emits every complete call before the terminal chunk.
    fn drain(&mut self, out: &mut Vec<UnifiedParserEvent>) -> anyhow::Result<()> {
        loop {
            if self.state == State::Idle {
                if self.buffer.is_empty() {
                    return Ok(());
                }
                if self.resolve_next_header(out) {
                    continue;
                }
                self.drain_idle_prose(out);
                return Ok(());
            }

            // A complete invoke inside the OPEN tool body emits as soon as its
            // close has streamed, before the channel terminator arrives.
            if self.state == State::InToolChannel
                && let Some((start, end)) = self.next_invoke(self.tool_body_limit())
            {
                let invoke = self.buffer[start..end].to_string();
                self.buffer.drain(..end);
                if let Some(delta) = self.emitter.parse_invoke(&invoke, self.next_index)? {
                    emit_call(out, delta);
                    self.next_index += 1;
                    // A call ends adjacency exactly as content does, so an owed
                    // separator from a preceding empty thought is void.
                    self.pending_reasoning_join = false;
                    self.reasoning_join_armed = false;
                }
                continue;
            }

            let terminator = [EOM, EOT]
                .iter()
                .filter_map(|t| self.buffer.find(t).map(|p| (p, t.len())))
                .min_by_key(|(p, _)| *p);
            // Framed headers cut any body; bare headers cut only a reasoning body
            // (missing-`<|eom|>` recovery).
            let start_pos = self.buffer.find(START);
            let bare_pos = if self.state == State::InReasoning {
                bare_header_pos(&self.buffer, self.last_body_char)
            } else {
                None
            };
            let boundary = match (start_pos, bare_pos) {
                (Some(a), Some(b)) => Some(a.min(b)),
                (a, b) => a.or(b),
            };
            let cut = match (terminator, boundary) {
                (Some((tp, _)), Some(bp)) if bp < tp => Some((bp, 0)),
                (Some((tp, tlen)), _) => Some((tp, tlen)),
                (None, Some(bp)) => Some((bp, 0)),
                (None, None) => None,
            };

            if let Some((body_end, term_len)) = cut {
                if term_len == 0 {
                    self.allow_bare_header =
                        bare_pos == Some(body_end) && start_pos != Some(body_end);
                }
                let body: String = self.buffer.drain(..body_end).collect();
                if let Some(c) = body.chars().next_back() {
                    self.last_body_char = Some(c);
                }
                self.buffer.drain(..term_len);
                match self.state {
                    State::InReasoning => {
                        // Only a thought that PRODUCED something can be adjacent to the
                        // next one. A redundant opener cuts a zero-length body, and
                        // arming on that prefixed the real thought with a newline.
                        self.emit_reasoning(out, &body);
                        self.pending_reasoning_join =
                            std::mem::take(&mut self.reasoning_body_emitted);
                    }
                    State::InContent => self.emit_text(out, &body),
                    // Residual markup around the invokes already emitted above is
                    // never text; dropping it is what keeps ATEM out of content.
                    State::InToolChannel => {}
                    State::Idle => unreachable!("Idle is handled above"),
                }
                self.state = State::Idle;
                continue;
            }

            // Open body: emit all but a tail that could still be a marker or, in
            // reasoning, an incoming bare header. A tool body emits nothing, so it
            // keeps everything buffered for the invoke scan.
            if self.state == State::InToolChannel {
                return Ok(());
            }
            let mut hold = marker_prefix_suffix_len(&self.buffer, MARKERS);
            if self.state == State::InReasoning {
                hold = hold.max(open_header_tail(&self.buffer));
            }
            let split = self.buffer.len() - hold;
            if split == 0 {
                return Ok(());
            }
            let body: String = self.buffer.drain(..split).collect();
            if let Some(c) = body.chars().next_back() {
                self.last_body_char = Some(c);
            }
            match self.state {
                State::InReasoning => {
                    self.emit_reasoning(out, &body);
                }
                State::InContent => self.emit_text(out, &body),
                State::Idle | State::InToolChannel => unreachable!("handled above"),
            }
            return Ok(());
        }
    }

    /// Resolve a complete header in the buffer and enter its channel. Returns
    /// whether the state machine advanced.
    fn resolve_next_header(&mut self, out: &mut Vec<UnifiedParserEvent>) -> bool {
        let Some(msg_pos) = self.buffer.find(MESSAGE) else {
            return false;
        };
        let (header_start, recipient) =
            resolve_header_latched(&self.buffer, msg_pos, self.allow_bare_header);
        let recipient = recipient.map(str::to_string);

        let prefix = self.buffer[..header_start].to_string();
        self.buffer.drain(..msg_pos + MESSAGE.len());
        self.emit_text(out, &prefix);
        self.allow_bare_header = false;
        self.last_body_char = None;
        match recipient.as_deref() {
            Some(REASONING_RECIPIENT) => {
                // Adjacent thoughts join with a newline, matching v1 and both
                // engines' single `reasoning_text` field; a thought after a call or
                // answer starts clean instead (the latch is cleared on emit).
                // `|=`, not `=`: an owed separator SURVIVES an intervening empty
                // thought. Overwriting it made the empty block eat the newline
                // between the two real thoughts around it.
                self.reasoning_join_armed |= std::mem::take(&mut self.pending_reasoning_join);
                self.state = State::InReasoning;
            }
            Some(rcpt) if rcpt != USER_RECIPIENT => self.state = State::InToolChannel,
            _ => self.state = State::InContent,
        }
        true
    }

    /// Surface prose between messages, holding back anything that could still
    /// become framing. Idle scans start at a real boundary (turn start or a
    /// consumed terminator), so offset zero is anchored.
    fn drain_idle_prose(&mut self, out: &mut Vec<UnifiedParserEvent>) {
        let bare_candidate = if self.allow_bare_header {
            bare_header_pos(&self.buffer, None)
        } else {
            None
        };
        let hold_from = self
            .buffer
            .find(START)
            .map(|s| bare_candidate.map_or(s, |b| s.min(b)))
            .or(bare_candidate)
            .unwrap_or_else(|| {
                let mut tail = marker_prefix_suffix_len(&self.buffer, MARKERS);
                if self.allow_bare_header {
                    tail = tail.max(open_header_tail(&self.buffer));
                }
                self.buffer.len() - tail
            });
        if hold_from > 0 {
            let emitted: String = self.buffer.drain(..hold_from).collect();
            self.emit_text(out, &emitted);
        }
    }

    /// Return every field carrying stream position to its fresh-stream value, handing
    /// back the buffer and the channel it was open in.
    ///
    /// The single place those fields are cleared, so ending a stream (`flush`) and
    /// abandoning one (`reset_stream`) cannot drift on what "fresh" means. A stale
    /// `allow_bare_header` would drop the next stream's header-less first message and
    /// leak its opening call as content; a stale `next_index` would file that stream's
    /// first call under an index the abandoned one already dispatched.
    fn take_stream_state(&mut self) -> (String, State) {
        self.allow_bare_header = true;
        self.last_body_char = None;
        self.pending_reasoning_join = false;
        self.reasoning_join_armed = false;
        self.reasoning_body_emitted = false;
        self.next_index = 0;
        (
            std::mem::take(&mut self.buffer),
            std::mem::replace(&mut self.state, State::Idle),
        )
    }

    /// Return to a FRESH-STREAM state and hand back whatever was still buffered.
    ///
    /// The carry is NOT emitted, unlike `flush`: it belongs to a stream the caller
    /// abandoned and must re-parse as a NEW one.
    pub(crate) fn reset_stream(&mut self) -> String {
        self.take_stream_state().0
    }

    /// End of stream: promote what is provably complete, drop parser-owned
    /// markup, and never leak an unfinished tool call.
    fn flush(&mut self, out: &mut Vec<UnifiedParserEvent>) {
        // Taken before the empty-buffer return so a drainless finish still resets.
        let (buffered, state) = self.take_stream_state();
        if buffered.is_empty() {
            return;
        }
        match state {
            // Leftover here is either held framing or prose that could have grown
            // into framing. Complete markers are stripped; a committed partial
            // special token (`<|sta`) is parser-owned markup and dropped; the
            // ambiguous `<` / `<|` and any `to=`-shaped prose stay visible.
            State::Idle | State::InContent => {
                let text = flush_open_text(&buffered);
                self.emit_text(out, &text);
            }
            State::InReasoning => {
                self.emit_reasoning(out, &flush_open_text(&buffered));
            }
            // Complete invokes already emitted during `push`; a truncated one is
            // dropped, markup and all, because the spec requires the literal close.
            State::InToolChannel => {
                if buffered.contains(INVOKE_OPEN_PREFIX) {
                    tracing::warn!(
                        why = "muse_glimmer_truncated_invoke",
                        buffered_bytes = buffered.len(),
                        "dropping ATEM invoke without a closing </atem:invoke> (truncated tool call?)"
                    );
                }
            }
        }
    }
}

/// Stream parser for Muse Glimmer ATEM tool calls.
pub struct MuseGlimmerToolStreamParser {
    scanner: MuseChannelScanner,
}

impl MuseGlimmerToolStreamParser {
    pub fn new(tools: &[Tool]) -> Self {
        Self {
            scanner: muse_scanner(tools),
        }
    }
}

impl ToolParser for MuseGlimmerToolStreamParser {
    fn create(tools: &[Tool]) -> anyhow::Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    fn preserve_special_tokens(&self) -> bool {
        true
    }

    fn push(&mut self, chunk: &str) -> anyhow::Result<ToolParseResult> {
        self.scanner.push(chunk)
    }

    fn finish(&mut self) -> anyhow::Result<ToolParseResult> {
        self.scanner.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tools(names: &[&str]) -> Vec<Tool> {
        names
            .iter()
            .map(|n| Tool {
                name: n.to_string(),
                description: None,
                parameters: serde_json::json!({"type": "object", "properties": {}}),
                strict: None,
            })
            .collect()
    }

    #[test]
    fn a_recipient_containing_to_routes_the_same_streamed_as_whole() {
        // `resolve_header` takes the FIRST `to=` in the run abutting
        // `<|message|>`, so the streaming holdback has to start there too. Held
        // from the LAST one instead, ` to=to=<|message|>` released the leading
        // `to=` as prose and then resolved the shorter recipient, which routes
        // the channel to a different lane than the whole-text parse — the call
        // vanished and its raw ATEM markup went to the client as content.
        //
        // Only BARE headers reach this: a framed one is held from `<|start|>`.
        let defs = tools(&["f"]);
        for text in [
            " to=to=<|message|><atem:invoke name=\"f\"><atem:parameter name=\"x\">1</atem:parameter></atem:invoke><|eom|>",
            " to=userto=self<|message|><atem:invoke name=\"f\"><atem:parameter name=\"x\">1</atem:parameter></atem:invoke><|eom|>",
            " to=auto=f<|message|>body<|eom|>",
        ] {
            let whole = parse_with(&defs, text);
            let mut parser = MuseGlimmerToolStreamParser::new(&defs);
            let mut streamed = ToolParseResult::default();
            for ch in text.chars() {
                streamed.append(parser.push(&ch.to_string()).expect("push"));
            }
            streamed.append(parser.finish().expect("finish"));
            let streamed = streamed.coalesce_calls();
            assert_eq!(whole.normal_text, streamed.normal_text, "text for {text:?}");
            assert_eq!(
                whole.calls.len(),
                streamed.calls.len(),
                "call count for {text:?}"
            );
        }
    }

    fn parse_with(tools: &[Tool], text: &str) -> ToolParseResult {
        let mut parser = MuseGlimmerToolStreamParser::new(tools);
        let mut out = parser.push(text).expect("push");
        out.append(parser.finish().expect("finish"));
        out.coalesce_calls()
    }

    #[test]
    fn a_reused_tool_parser_restarts_indices_at_zero() {
        // The unified factory has its own reuse pins; this covers the OTHER surface
        // built from the SAME scanner. `flush` resets `next_index`, so a second turn
        // on one reused `MuseGlimmerToolStreamParser` numbers its calls from 0 like a
        // fresh parser does. Left counting, the reused turn hands the client index 1,
        // which any assembler keying calls by `tool_index` folds into the prior turn.
        let defs = tools(&["f", "g"]);
        let turn = |parser: &mut MuseGlimmerToolStreamParser, text: &str| {
            let mut out = parser.push(text).expect("push");
            out.append(parser.finish().expect("finish"));
            out
        };
        let first = " to=f<|message|><atem:invoke name=\"f\"></atem:invoke><|eom|>";
        let second = " to=g<|message|><atem:invoke name=\"g\"></atem:invoke><|eom|>";
        let mut reused = MuseGlimmerToolStreamParser::new(&defs);
        turn(&mut reused, first);
        let got = turn(&mut reused, second);
        let want = turn(&mut MuseGlimmerToolStreamParser::new(&defs), second);
        assert_eq!(
            got.calls, want.calls,
            "reused parser diverged on the second turn"
        );
        assert_eq!(
            got.calls[0].tool_index, 0,
            "the second turn's call is not index 0"
        );
    }

    fn parse(text: &str) -> ToolParseResult {
        parse_with(&[], text)
    }

    fn args(result: &ToolParseResult, idx: usize) -> serde_json::Value {
        serde_json::from_str(&result.calls[idx].arguments).expect("arguments are JSON")
    }

    fn name(result: &ToolParseResult, idx: usize) -> &str {
        result.calls[idx].name.as_deref().expect("call has a name")
    }

    const SINGLE_CALL: &str = concat!(
        " to=get_weather<|message|><atem:function_calls>\n",
        "<atem:invoke name=\"get_weather\">\n",
        "<atem:parameter name=\"city\">Paris</atem:parameter>\n",
        "</atem:invoke>\n",
        "</atem:function_calls><|eom|>"
    );

    #[test]
    fn single_call_headerless_first_message() {
        // Invariant 2: the turn's first header is legitimately bare, because the
        // generation prompt consumed `<|start|>assistant`.
        let out = parse(SINGLE_CALL);
        assert_eq!(out.calls.len(), 1);
        assert_eq!(name(&out, 0), "get_weather");
        assert_eq!(args(&out, 0), serde_json::json!({"city": "Paris"}));
        assert_eq!(out.normal_text, "");
    }

    #[test]
    fn parallel_calls_in_separate_eom_chained_messages() {
        // Invariant 7: both channels of one delta emit in the same push.
        let out = parse(concat!(
            " to=get_weather<|message|><atem:function_calls>\n",
            "<atem:invoke name=\"get_weather\">\n",
            "<atem:parameter name=\"city\">Paris</atem:parameter>\n",
            "</atem:invoke>\n</atem:function_calls><|eom|>",
            "<|start|>assistant to=get_time<|message|><atem:function_calls>\n",
            "<atem:invoke name=\"get_time\">\n",
            "<atem:parameter name=\"timezone\">CET</atem:parameter>\n",
            "</atem:invoke>\n</atem:function_calls><|eom|>"
        ));
        assert_eq!(out.calls.len(), 2);
        assert_eq!(name(&out, 0), "get_weather");
        assert_eq!(out.calls[0].tool_index, 0);
        assert_eq!(name(&out, 1), "get_time");
        assert_eq!(out.calls[1].tool_index, 1);
        assert_eq!(args(&out, 1), serde_json::json!({"timezone": "CET"}));
        assert_eq!(out.normal_text, "");
    }

    #[test]
    fn multiple_invokes_in_one_message() {
        let out = parse(concat!(
            " to=tools<|message|><atem:function_calls>\n",
            "<atem:invoke name=\"a\">\n",
            "<atem:parameter name=\"x\">1</atem:parameter>\n",
            "</atem:invoke>\n",
            "<atem:invoke name=\"b\">\n",
            "<atem:parameter name=\"y\">2</atem:parameter>\n",
            "</atem:invoke>\n",
            "</atem:function_calls><|eom|>"
        ));
        assert_eq!(out.calls.len(), 2);
        assert_eq!(name(&out, 0), "a");
        assert_eq!(args(&out, 0), serde_json::json!({"x": 1}));
        assert_eq!(name(&out, 1), "b");
        assert_eq!(args(&out, 1), serde_json::json!({"y": 2}));
        assert_eq!(out.normal_text, "");
    }

    #[test]
    fn value_types_json_and_raw_fallback() {
        // Invariant 6: the spec's `value_parser: json` with `allow_non_json: true`.
        let out = parse(concat!(
            " to=f<|message|><atem:invoke name=\"f\">",
            "<atem:parameter name=\"n\">3</atem:parameter>",
            "<atem:parameter name=\"flag\">true</atem:parameter>",
            "<atem:parameter name=\"none\">null</atem:parameter>",
            "<atem:parameter name=\"obj\">{\"k\": [1, 2]}</atem:parameter>",
            "<atem:parameter name=\"plain\">just words</atem:parameter>",
            "<atem:parameter name=\"jsonish\">{not json}</atem:parameter>",
            "</atem:invoke><|eom|>"
        ));
        assert_eq!(
            args(&out, 0),
            serde_json::json!({
                "n": 3,
                "flag": true,
                "none": null,
                "obj": {"k": [1, 2]},
                "plain": "just words",
                "jsonish": "{not json}",
            })
        );
    }

    #[test]
    fn an_attribute_value_holding_a_close_bracket_matches_no_invoke() {
        // The `[^>]*?` bound in `invoke_open_re` ends the tag at the FIRST `>`, so
        // a `>` inside an attribute value other than `name` closes the tag early
        // and the invoke matches nothing. Pinned as the CONTRACT, not as a wish:
        // the response template never writes such a value, and v1 reads it the
        // same way, so aligning the two on a recovery here would be inventing a
        // shape the model cannot emit.
        let out = parse(concat!(
            " to=f<|message|><atem:function_calls>\n",
            "<atem:invoke id=\"a>b\" name=\"f\">\n",
            "<atem:parameter name=\"x\">1</atem:parameter>\n",
            "</atem:invoke>\n</atem:function_calls><|eom|>"
        ));
        assert!(out.calls.is_empty());
        // The tool channel still absorbs its body, so nothing leaks as prose.
        assert_eq!(out.normal_text, "");
    }

    #[test]
    fn the_same_attribute_after_name_matches_and_folds_its_tail_into_the_body() {
        // The pin above reads as "a `>` in a non-`name` invoke attribute drops the
        // call". It does not: POSITION decides. Behind `name` the pattern has already
        // captured what it needs, so the tag MATCHES and the early `>` only moves
        // where the argument body starts. The tail lands at the head of that body.
        //
        // Usually that is harmless — the parameter scan skips text it does not match:
        let out = parse(concat!(
            " to=f<|message|><atem:invoke name=\"f\" id=\"a>b\">",
            "<atem:parameter name=\"x\">1</atem:parameter></atem:invoke><|eom|>"
        ));
        assert_eq!(name(&out, 0), "f");
        assert_eq!(args(&out, 0), serde_json::json!({"x": 1}));

        // It stops being harmless once the value holds markup. A close tag inside it
        // ends the call before its real parameters, which ships an EMPTY argument set
        // under a live tool name:
        let out = parse(concat!(
            " to=f<|message|><atem:invoke name=\"f\" id=\"a></atem:invoke>b\">",
            "<atem:parameter name=\"x\">1</atem:parameter></atem:invoke><|eom|>"
        ));
        assert_eq!(args(&out, 0), serde_json::json!({}));

        // And a parameter tag inside it becomes a REAL argument — the same fail-open
        // shape `parameter_re` has, reached through the invoke tag. v1 agrees on all
        // three, so the bound stays ONE documented shape across the two crates:
        let out = parse(concat!(
            " to=f<|message|><atem:invoke name=\"f\" id=\"a>",
            "<atem:parameter name=\"y\">9</atem:parameter>b\">",
            "<atem:parameter name=\"x\">1</atem:parameter></atem:invoke><|eom|>"
        ));
        assert_eq!(args(&out, 0), serde_json::json!({"y": 9, "x": 1}));
    }

    #[test]
    fn a_close_bracket_inside_the_name_value_itself_is_read_as_the_name() {
        // The THIRD shape of the same `[^>]*?` bound, and the one the two pins around
        // it can be read as denying. The bound is on the attributes AROUND `name`; the
        // name value is read by `[^"]+`, which spans a `>` happily. So neither the
        // fail-closed nor the fail-open path fires — the tag matches and the `>` lands
        // in the identifier.
        let out = parse(concat!(
            " to=f<|message|><atem:invoke name=\"a>b\">",
            "<atem:parameter name=\"x\">1</atem:parameter>",
            "</atem:invoke><|eom|>"
        ));
        assert_eq!(name(&out, 0), "a>b");
        assert_eq!(args(&out, 0), serde_json::json!({"x": 1}));
        // Same on the parameter side: the KEY carries the `>` through.
        let out = parse(concat!(
            " to=f<|message|><atem:invoke name=\"f\">",
            "<atem:parameter name=\"a>b\">1</atem:parameter>",
            "</atem:invoke><|eom|>"
        ));
        assert_eq!(name(&out, 0), "f");
        assert_eq!(args(&out, 0), serde_json::json!({"a>b": 1}));
    }

    #[test]
    fn an_attribute_value_holding_a_close_bracket_mis_reads_a_parameter() {
        // The SAME `[^>]*?` bound, the OTHER failure. `invoke_open_re` fails closed
        // (no call at all, above); `parameter_re` fails OPEN — the tag still matches
        // and the argument is silently wrong, which is why the two are pinned apart.
        // Before `name`, the early `>` swallows the key and the parameter is gone:
        let out = parse(concat!(
            " to=f<|message|><atem:invoke name=\"f\">",
            "<atem:parameter id=\"a>b\" name=\"x\">1</atem:parameter>",
            "</atem:invoke><|eom|>"
        ));
        assert_eq!(args(&out, 0), serde_json::json!({}));
        // After `name`, the tag remainder folds into the VALUE:
        let out = parse(concat!(
            " to=f<|message|><atem:invoke name=\"f\">",
            "<atem:parameter name=\"x\" id=\"a>b\">1</atem:parameter>",
            "</atem:invoke><|eom|>"
        ));
        assert_eq!(args(&out, 0), serde_json::json!({"x": "b\">1"}));
    }

    #[test]
    fn quoted_json_string_stays_string_and_raw_keeps_whitespace() {
        // Invariant 6 is byte-preserving: the raw fallback is not trimmed.
        let out = parse(concat!(
            " to=f<|message|><atem:invoke name=\"f\">",
            "<atem:parameter name=\"quoted\">\"true\"</atem:parameter>",
            "<atem:parameter name=\"padded\"> spaced out </atem:parameter>",
            "</atem:invoke><|eom|>"
        ));
        assert_eq!(
            args(&out, 0),
            serde_json::json!({"quoted": "true", "padded": " spaced out "})
        );
    }

    #[test]
    fn reasoning_channel_never_parses_as_call() {
        // Invariant 5. The tool-only projection folds the reasoning body into
        // normal_text (it has no reasoning channel); what matters here is that the
        // quoted markup never becomes a live call.
        let out = parse(concat!(
            " to=self<|message|>I could call <atem:invoke name=\"fake\">",
            "<atem:parameter name=\"x\">1</atem:parameter></atem:invoke> here.<|eom|>",
            "<|start|>assistant to=user<|message|>No tool needed.<|eot|>"
        ));
        assert!(out.calls.is_empty());
        // Pin the FOLD, not just its tail. `ToolParseResult::from_deltas` concatenates a
        // reasoning body where it occurred, so the thought abuts the answer with no
        // separator. v1's `try_tool_call_parse_muse_glimmer` skips a `to=self` body
        // instead, and returns "No tool needed." alone for these bytes.
        assert_eq!(
            out.normal_text,
            concat!(
                "I could call <atem:invoke name=\"fake\">",
                "<atem:parameter name=\"x\">1</atem:parameter></atem:invoke> here.",
                "No tool needed."
            )
        );

        // Left as is, not adopted as a contract. Which side is right needs an engine
        // reference this repo cannot capture yet — the vLLM parser is unmerged
        // (PR #51655) and `parser_families.yaml` carries `vllm_rust: null` for the
        // family — the same block that leaves the I3 gap open in
        // `crate::unified::muse_glimmer`. It costs nothing on Muse's intended Dynamo
        // path, the UNIFIED parser, which keeps the two channels apart; reaching it
        // needs the tool-only surface wired on RAW output with no muse reasoning parser
        // in front, which the module doc above already calls only half-supported.
    }

    #[test]
    fn atem_in_user_channel_stays_text() {
        // Invariant 5.
        let out = parse(concat!(
            " to=user<|message|>Example: <atem:invoke name=\"demo\">",
            "<atem:parameter name=\"x\">1</atem:parameter></atem:invoke><|eot|>"
        ));
        assert!(out.calls.is_empty());
        assert!(out.normal_text.contains("<atem:invoke name=\"demo\">"));
    }

    #[test]
    fn unframed_atem_does_not_parse() {
        // Invariant 5: no scan-everything fallback, so unframed ATEM is prose.
        let text = concat!(
            "<atem:function_calls><atem:invoke name=\"f\">",
            "<atem:parameter name=\"x\">1</atem:parameter></atem:invoke></atem:function_calls>"
        );
        let out = parse(text);
        assert!(out.calls.is_empty());
        assert_eq!(out.normal_text, text);
    }

    #[test]
    fn plain_text_passthrough() {
        let out = parse("The capital of France is Paris.");
        assert!(out.calls.is_empty());
        assert_eq!(out.normal_text, "The capital of France is Paris.");
    }

    #[test]
    fn empty_and_whitespace_input() {
        let out = parse("");
        assert!(out.calls.is_empty());
        assert_eq!(out.normal_text, "");
        let out = parse("   ");
        assert!(out.calls.is_empty());
        assert_eq!(out.normal_text, "   ");
    }

    #[test]
    fn framed_header_inside_user_body_is_a_real_channel_switch() {
        // Invariant 3: `<|start|>` is a reserved special token, so a framed header
        // mid-answer means the model really switched channels.
        let out = parse(concat!(
            " to=user<|message|>Look: <|start|>assistant to=get_weather<|message|>",
            "<atem:invoke name=\"get_weather\"><atem:parameter name=\"city\">Nice</atem:parameter></atem:invoke><|eom|>"
        ));
        assert_eq!(out.calls.len(), 1);
        assert_eq!(name(&out, 0), "get_weather");
        assert_eq!(out.normal_text, "Look: ");
    }

    #[test]
    fn quoted_bare_header_in_user_body_is_not_promoted_to_a_call() {
        // Invariants 2 and 5: the missing-`<|eom|>` recovery is reasoning-only, so
        // quoted markup in an answer can never become a live call.
        let out = parse(concat!(
            " to=user<|message|>Example: to=search<|message|>",
            "<atem:invoke name=\"search\"><atem:parameter name=\"q\">oops</atem:parameter></atem:invoke><|eot|>"
        ));
        assert!(out.calls.is_empty());
        assert!(out.normal_text.contains("Example: to=search"));
        assert!(out.normal_text.contains("<atem:invoke name=\"search\">"));
    }

    #[test]
    fn bare_header_in_prose_after_a_closed_channel_is_not_a_call() {
        // Invariant 2 at the OTHER latch site: once any header has resolved, the
        // turn-start licence is spent, so `to=` prose sitting between messages
        // stays prose. Only a bare-cut reasoning body re-arms it.
        let out = parse(concat!(
            " to=user<|message|>x<|eot|>",
            "Example: to=search<|message|><atem:invoke name=\"search\"></atem:invoke><|eot|>"
        ));
        assert!(out.calls.is_empty());
        assert_eq!(
            out.normal_text,
            "xExample: to=search<atem:invoke name=\"search\"></atem:invoke>"
        );
    }

    #[test]
    fn concatenated_prose_around_to_is_not_a_recovery_boundary() {
        // Invariant 2: `potato=` must not cut the reasoning body mid-word.
        let out = parse(concat!(
            " to=self<|message|>weird potato=get_weather<|message|>",
            "<atem:invoke name=\"get_weather\"></atem:invoke><|eom|>",
        ));
        assert!(out.calls.is_empty());
    }

    #[test]
    fn missing_eom_before_tool_header_recovers_the_call() {
        // Invariant 1: observed model defect — the reasoning channel ends without
        // `<|eom|>` and the tool header follows directly.
        let out = parse(concat!(
            " to=self<|message|>thinking to=get_weather<|message|>",
            "<atem:invoke name=\"get_weather\"></atem:invoke><|eom|>"
        ));
        assert_eq!(out.calls.len(), 1);
        assert_eq!(name(&out, 0), "get_weather");
        assert_eq!(args(&out, 0), serde_json::json!({}));
    }

    #[test]
    fn truncated_invoke_is_dropped() {
        // P2: no call, and no leaked markup.
        let out = parse(concat!(
            " to=get_weather<|message|><atem:function_calls>\n",
            "<atem:invoke name=\"get_weather\">\n",
            "<atem:parameter name=\"city\">Par"
        ));
        assert!(out.calls.is_empty());
        assert_eq!(out.normal_text, "");
    }

    #[test]
    fn unterminated_message_with_complete_invoke_recovers() {
        let out = parse(concat!(
            " to=get_weather<|message|><atem:function_calls>\n",
            "<atem:invoke name=\"get_weather\">\n",
            "<atem:parameter name=\"city\">Paris</atem:parameter>\n",
            "</atem:invoke>\n</atem:function_calls>"
        ));
        assert_eq!(out.calls.len(), 1);
        assert_eq!(name(&out, 0), "get_weather");
    }

    #[test]
    fn orphan_terminator_is_stripped_from_prose() {
        // Invariant 3.
        let out = parse("some prose<|eom|> more");
        assert!(out.calls.is_empty());
        assert_eq!(out.normal_text, "some prose more");
    }

    #[test]
    fn empty_arguments_object() {
        // P3.
        let out = parse(" to=ping<|message|><atem:invoke name=\"ping\"></atem:invoke><|eom|>");
        assert_eq!(out.calls[0].arguments, "{}");
    }

    #[test]
    fn multiline_and_unicode_values() {
        let out = parse(concat!(
            " to=f<|message|><atem:invoke name=\"f\">",
            "<atem:parameter name=\"text\">line one\nline two — καλημέρα 你好</atem:parameter>",
            "</atem:invoke><|eom|>"
        ));
        assert_eq!(
            args(&out, 0),
            serde_json::json!({"text": "line one\nline two — καλημέρα 你好"})
        );
    }

    #[test]
    fn namespaced_names_pass_through_and_doubled_collapses() {
        let out = parse_with(
            &tools(&["get_weather", "web.search"]),
            concat!(
                " to=get_weather.get_weather<|message|>",
                "<atem:invoke name=\"get_weather.get_weather\"></atem:invoke><|eom|>",
                "<|start|>assistant to=web.search<|message|>",
                "<atem:invoke name=\"web.search\"></atem:invoke><|eom|>"
            ),
        );
        assert_eq!(name(&out, 0), "get_weather");
        assert_eq!(name(&out, 1), "web.search");
    }

    #[test]
    fn unknown_name_passes_through() {
        // No leaf matching: `weather.get` must not dispatch `calendar.get`.
        let out = parse_with(
            &tools(&["calendar.get"]),
            " to=weather.get<|message|><atem:invoke name=\"weather.get\"></atem:invoke><|eom|>",
        );
        assert_eq!(name(&out, 0), "weather.get");
    }

    #[test]
    fn recipient_and_invoke_name_disagreement_uses_invoke_name() {
        let out =
            parse(" to=get_weather<|message|><atem:invoke name=\"get_time\"></atem:invoke><|eom|>");
        assert_eq!(name(&out, 0), "get_time");
    }

    #[test]
    fn eot_terminates_a_tool_channel_too() {
        let out = parse(" to=f<|message|><atem:invoke name=\"f\"></atem:invoke><|eot|>");
        assert_eq!(out.calls.len(), 1);
    }

    #[test]
    fn earliest_terminator_wins_when_both_present() {
        // The user body ends at `<|eot|>`; the trailing `<|eom|>` is an orphan.
        let out = parse(" to=user<|message|>done<|eot|><|eom|>");
        assert!(out.calls.is_empty());
        assert_eq!(out.normal_text, "done");
    }

    #[test]
    fn duplicate_parameter_keys_last_wins() {
        let out = parse(concat!(
            " to=f<|message|><atem:invoke name=\"f\">",
            "<atem:parameter name=\"x\">1</atem:parameter>",
            "<atem:parameter name=\"x\">2</atem:parameter>",
            "</atem:invoke><|eom|>"
        ));
        assert_eq!(args(&out, 0), serde_json::json!({"x": 2}));
    }

    #[test]
    fn source_parameter_order_survives_reserialization() {
        // End to end, the argument order is the model's and not alphabetical.
        let out = parse(concat!(
            " to=f<|message|><atem:invoke name=\"f\">",
            "<atem:parameter name=\"path\">/app/x.go</atem:parameter>",
            "<atem:parameter name=\"command\">str_replace</atem:parameter>",
            "</atem:invoke><|eom|>"
        ));
        assert_eq!(
            out.calls[0].arguments,
            r#"{"path":"/app/x.go","command":"str_replace"}"#
        );

        // That half cannot fail here. `openai-harmony` unifies
        // `serde_json/preserve_order` into every build of this workspace, so the map
        // already keeps insertion order and the assertion above stays green with the
        // `reorder_arguments` call deleted outright. Only a consumer whose graph lacks
        // that feature gets the alphabetical map the call exists for, and no test in
        // this workspace can reach that build, so the call site is uncovered by
        // construction. Cover the HELPER instead, which had no direct test at all
        // although six families route their arguments through it.
        assert_eq!(
            reorder_arguments(
                r#"{"command":"str_replace","path":"/app/x.go"}"#,
                &["path".to_string(), "command".to_string()],
            ),
            r#"{"path":"/app/x.go","command":"str_replace"}"#
        );
    }

    #[test]
    fn invoke_markup_inside_value_truncates_like_engines() {
        // The regex reads the value to the FIRST close marker, exactly like both
        // engine parsers; markup-as-data truncates rather than escapes.
        let out = parse(concat!(
            " to=f<|message|><atem:invoke name=\"f\">",
            "<atem:parameter name=\"x\">a</atem:parameter>b</atem:parameter>",
            "</atem:invoke><|eom|>"
        ));
        assert_eq!(args(&out, 0), serde_json::json!({"x": "a"}));
    }

    #[test]
    fn crlf_and_unicode_survive_in_values_and_recipients() {
        let out = parse(concat!(
            " to=天気.lookup<|message|><atem:invoke name=\"天気.lookup\">",
            "<atem:parameter name=\"città\">Rome\r\nItaly</atem:parameter>",
            "</atem:invoke><|eom|>"
        ));
        assert_eq!(out.calls.len(), 1);
        assert_eq!(name(&out, 0), "天気.lookup");
        assert_eq!(args(&out, 0), serde_json::json!({"città": "Rome\r\nItaly"}));
        assert_eq!(out.normal_text, "");
    }

    #[test]
    fn prose_ending_in_assistant_before_a_bare_header_stays_prose() {
        // The role word absorbs into the header only behind a real `<|start|>`.
        let out = parse("my assistant  to=user<|message|>x<|eot|>");
        assert!(out.calls.is_empty());
        assert_eq!(out.normal_text, "my assistantx");
    }

    #[test]
    fn resolve_header_finds_framed_bare_and_recipient_less_headers() {
        let text = "<|start|>assistant to=self<|message|>a<|eom|>";
        let msg = text.find(MESSAGE).unwrap();
        assert_eq!(resolve_header(text, msg), (0, Some("self")));

        let text = " to=user<|message|>hi<|eot|>";
        let msg = text.find(MESSAGE).unwrap();
        assert_eq!(resolve_header(text, msg), (0, Some("user")));

        let text = "<|message|>hi<|eot|>";
        assert_eq!(resolve_header(text, 0), (0, None));
    }
}

#[cfg(test)]
mod strip_splice_tests {
    use super::*;

    #[test]
    fn a_marker_the_strip_splices_together_is_stripped_too() {
        // Removing one marker JOINS its neighbours, and the join can spell a
        // second marker: `<|st` + `<|eom|>` + `art|>` leaves a live `<|start|>`,
        // the token the streaming jail keys on. A strip that made one whole-text
        // pass per marker left that one behind whenever its own pass had already
        // run, so the strip retracts a marker the moment an append completes it.
        //
        // The pass ORDER was the other half. This crate listed [START, MESSAGE,
        // EOM, EOT] and parsers-v1's `push_stripped` listed [EOM, EOT, START,
        // MESSAGE], so the two leaked on DIFFERENT inputs and returned different
        // text for the same bytes. Retracting on append tries the markers in no
        // order at all, which is what keeps the two sides equal.
        //
        // Known limit: a marker spliced across the seam between two SEPARATE
        // `emit_text` runs. The strip is per-run by construction, and
        // `flush_open_text` releases a trailing `<` / `<|` as prose instead of
        // holding it, so closing that seam would stall every `<` a model writes.
        // Nor is it only a concatenation artifact: where a run ENDS can differ
        // between batch and stream, so those inputs are the exception to I5 as
        // well. `crate::unified::muse_glimmer` pins the shapes and the reasoning.
        for outer in MARKERS {
            for cut in 1..outer.len() {
                for inner in MARKERS {
                    let core = format!("{}{inner}{}", &outer[..cut], &outer[cut..]);
                    assert_eq!(
                        stripped(&format!("keep {core} keep")),
                        "keep  keep",
                        "for {core:?}"
                    );
                }
            }
        }
    }
}
