// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unified parser for the Beam channel grammar (NOT GPT-OSS Harmony).
//!
//! ```text
//! reasoning:  <|start|>assistant<|channel|>analysis<|message|> … <|end|>
//! content:    <|start|>assistant<|channel|>final<|message|> … <|return|>
//! tool call:  <|start|>assistant to=NAME<|channel|>analysis <|constrain|>xml<|message|>
//!             <rfl:parameter name="KEY">VALUE</rfl:parameter>… <|call|>   (or <|end|>)
//! ```
//!
//! There is no Beam scanner here. The shared [`WrappedBlockScanner`] owns buffering,
//! quote/control protection, holdback, call indices and reset through its channel
//! envelope mode; this file is only the grammar table ([`ENVELOPE`]), the tool-body walk
//! ([`BeamBoundary`]) and the argument emitter ([`BeamEmitter`]).
//!
//! Analysis closes only on `<|end|>`; the final channel closes on `<|end|>` or
//! `<|return|>`; `<|call|>` inside either is ordinary text. A tool call completes only
//! once its terminator (`<|call|>` / `<|end|>`) arrives, but its arguments stream ahead
//! of it ([`OpenCall`]): a raw-text string parameter as its bytes arrive, any other
//! parameter whole at its closer, every fragment an exact prefix of the completed JSON.
//!
//! Under guided decoding the same table drives the header hooks ([`GUIDED_CHANNEL`]):
//! a header owns its bytes from its first one (`<|start|>`, or the bare lead the prompt
//! left open) until it resolves, so no marker inside it is consumed early; a bare lead
//! is honoured only in the turn's open header slot, never inside a body; and inside a
//! body only that channel's own terminators are structural.
//!
//! Malformed calls follow the request's [`ToolCallPolicy`]
//! ([`UnifiedParser::set_tool_call_policy`]): `Drop` (default, the corpus contract)
//! discards only the bad region, `Reject` fails with a typed error and leaves every
//! uncommitted byte, header included, for [`UnifiedParser::reset`].
//!
//! Syntax decoding is separate from request tool policy: an unknown tool name still
//! yields a call whose arguments keep their text, and the caller decides whether to
//! honour it.
//!
//! Argument framing ([`ArgumentFraming`]) is request-scoped too. The native serving
//! grammar mixes slots: a parameter of a tool the grammar holds (every tool whose
//! schema it can type, `strict` or not) whose schema admits only non-string types is
//! grammar-constrained JSON, every other slot is raw text. Under
//! [`ArgumentFraming::SchemaConstrained`] the emitter precomputes that key set per
//! held tool and hands the scanner, at each tool header AND at each tool header the
//! prose-quote lexer must skip, the same [`BeamBoundary`] walk carrying the recipient's
//! set ([`InvokeEmitter::channel_boundary`]), so the streaming walk, prose ownership and
//! the final decoder agree about an embedded `</rfl:parameter>`. The key set follows
//! the serving grammar's own union type rule, not the conjunctive typing rule (see
//! `JsonKeys`), because it recovers how the bytes were framed, not what they mean.

use crate::tool_calling::beam_body::{
    JsonKeys, PARAM_FOLLOW, ValueCursor, held_json_keys, param_value_span_with,
    recipient_admits_empty_arguments, recipient_len, recipient_properties, typed_arguments,
    typed_value, unframe, value_framing, verbatim_string,
};
use crate::tool_calling::beam_py::{PyValue, dump_str, dump_str_fragment};
use crate::tool_calling::scan::{
    BodyScan, ChannelEnvelope, EnvelopeChannel, InvokeBoundary, InvokeBoundaryFactory,
    InvokeEmitter, ReasoningSpec, WrappedBlockScanner, WrappedBlockSpec, envelope_guided_close,
    envelope_guided_holdback, envelope_guided_open, envelope_guided_routing, envelope_guided_stray,
    envelope_guided_transition, envelope_guided_turn_end, marker_prefix_suffix_len,
};
use crate::tool_calling::traits::{Tool, ToolCallDelta};
use crate::unified::{
    ArgumentFraming, GuidedChannel, GuidedChannelState, GuidedRouted, MalformedToolCallPolicy,
    ScannerUnified, ToolCallPolicy, UnifiedParser,
};
use serde_json::{Map, Value};
use std::collections::{HashMap, HashSet};

const START: &str = "<|start|>";
const CHANNEL: &str = "<|channel|>";
const MESSAGE: &str = "<|message|>";
const END: &str = "<|end|>";
const RETURN: &str = "<|return|>";
const CALL: &str = "<|call|>";
const CONSTRAIN: &str = "<|constrain|>";
const PARAM_OPEN: &str = "<rfl:parameter name=\"";
const PARAM_CLOSE: &str = "</rfl:parameter>";

static CHANNELS: [EnvelopeChannel; 2] = [
    EnvelopeChannel {
        name: "analysis",
        reasoning: true,
        terminators: &[END],
        json_body: false,
    },
    EnvelopeChannel {
        name: "final",
        reasoning: false,
        // The serving grammar closes the final answer with `<|return|>`, `<|end|>` or
        // `<|call|>` (all three are the model's stop tokens), so a `<|call|>` ends the
        // answer here too rather than reaching the user as text.
        terminators: &[END, RETURN, CALL],
        // A `response_format` JSON answer may quote a marker inside a string.
        json_body: true,
    },
];

static ENVELOPE: ChannelEnvelope = ChannelEnvelope {
    start: START,
    channel: CHANNEL,
    message: MESSAGE,
    role: "assistant",
    recipient_key: "to=",
    constrain: CONSTRAIN,
    constrain_tokens: &["xml"],
    recipient_len,
    channels: &CHANNELS,
    tool_channel: "analysis",
    max_header: 128,
    // The assistant turn is channel framing only: there is no implicit region, so
    // bytes between messages are debris, never text (the builder's template has no
    // implicit field either).
    prose_between_messages: false,
    // Markers are special tokens: a terminator ends its body wherever it stands. A
    // quotation never keeps one literal (an unbalanced `"` in a thought must not
    // swallow the final answer, nor hold the stream back until it ends).
    quote_aware: false,
    // Terminators AND the inner header markers: a `<|constrain|>` or `<|message|>` with no
    // header around it is framing debris, never visible text (goal 3/4).
    orphan_controls: &[CALL, END, RETURN, CONSTRAIN, MESSAGE],
    // Only `<|return|>` ends the turn for the guided reader: `<|call|>` closes a tool
    // call (or the final answer) but a guided payload may still follow it.
    turn_end: &[RETURN],
};

/// Every fixed marker of the grammar: terminator competition and split-marker holdback
/// for the guided reader. The recipient inside a header is data and cannot compete.
static GUIDED_COMPETITORS: [&str; 6] = [START, CHANNEL, MESSAGE, END, RETURN, CALL];

/// Only `<|end|>` closes an analysis body (`<|call|>` and `<|return|>` are literal there).
static GUIDED_CLOSE_MARKERS: [&str; 1] = [END];

const BOUNDARY: InvokeBoundaryFactory =
    InvokeBoundaryFactory::NativeOnly(|| Box::new(BeamBoundary::default()));

/// The guided reader's view of [`ENVELOPE`]: every hook is the shared envelope
/// resolver over the same table the native drain uses, so a header is owned the same
/// way whether the call arrives as native markup or as guided JSON. Each hook also
/// receives [`BOUNDARY`], so a native tool body is resolved left to right by the SAME
/// walk the native drain uses and header-shaped bytes inside a parameter value are
/// argument data for every candidate kind alike.
static GUIDED_CHANNEL: GuidedChannel = GuidedChannel {
    find_open: |h, flush, state| envelope_guided_open(&ENVELOPE, Some(BOUNDARY), h, flush, state),
    find_close: |h, flush, state| envelope_guided_close(&ENVELOPE, Some(BOUNDARY), h, flush, state),
    find_turn_end: |h, flush, state| {
        envelope_guided_turn_end(&ENVELOPE, Some(BOUNDARY), h, flush, state)
    },
    find_transition: |h, flush, state| {
        envelope_guided_transition(&ENVELOPE, Some(BOUNDARY), h, flush, state)
    },
    find_stray: |h, flush, state| envelope_guided_stray(&ENVELOPE, Some(BOUNDARY), h, flush, state),
    find_routing: |h, flush, state| {
        envelope_guided_routing(&ENVELOPE, Some(BOUNDARY), h, flush, state)
    },
    holdback: |h, state: GuidedChannelState| {
        envelope_guided_holdback(&ENVELOPE, Some(BOUNDARY), h, state)
    },
    // Beam's markers are special tokens the native path strips positionally; prose
    // that quotes them under a Response prefill keeps them literal.
    strip_text: str::to_string,
    competitors: &GUIDED_COMPETITORS,
    close_markers: &GUIDED_CLOSE_MARKERS,
    response_literal_markers: &GUIDED_COMPETITORS,
    // A header owns its bytes from `<|start|>` (or its bare lead) until it resolves,
    // and inside an analysis body only `<|end|>` is structural.
    envelope_owned: true,
    header_pending: None,
};

/// Walks one tool-channel body: complete `<rfl:parameter>` regions, then a terminator.
///
/// Under [`ArgumentFraming::SchemaConstrained`] the recipient's [`JsonKeys`] decide per
/// parameter whether the value owns JSON strings or ends at the first closer.
/// [`BeamEmitter::channel_boundary`] builds the populated one for each call, on the
/// native drain and on the prose-quote lexer alike; only the guided reader's ownership
/// (a native call inside a guided haystack) uses the factory-built boundary, whose empty
/// set is the raw-text rule for every key.
#[derive(Default)]
struct BeamBoundary {
    framing: ArgumentFraming,
    /// `Reject` reads the body strictly; `Drop` reads it as the template's tag search.
    malformed: MalformedToolCallPolicy,
    /// JSON-owned keys of the current recipient; metadata, survives `reset`.
    json_keys: JsonKeys,
    /// Offset up to which parameter regions are known complete.
    scanned: usize,
    /// Complete parameters read in this body.
    complete: usize,
    /// The walk over the parameter open at `open_from`, resumed across pushes.
    open_from: usize,
    open_value: ValueCursor,
    /// The recipient's schema admits an empty argument body (declared, nothing
    /// required), so the grammar lets the stream end right after the header.
    empty_body_ok: bool,
}

fn malformed(discard: usize, reason: &'static str) -> BodyScan {
    BodyScan::Malformed { discard, reason }
}

impl InvokeBoundary for BeamBoundary {
    fn configure(&mut self, policy: ToolCallPolicy) {
        self.framing = policy.argument_framing;
        self.malformed = policy.malformed;
    }

    fn reset(&mut self) {
        self.scanned = 0;
        self.complete = 0;
        self.open_from = 0;
        self.open_value = ValueCursor::default();
    }

    fn channel_body(&mut self, body: &str, flush: bool) -> BodyScan {
        let lenient = self.malformed != MalformedToolCallPolicy::Reject;
        loop {
            let rest = &body[self.scanned..];
            let t = rest.trim_start_matches([' ', '\t', '\n', '\r']);
            let at = self.scanned + (rest.len() - t.len());
            self.scanned = at;
            if t.is_empty() {
                // The serving grammar lets EOS end a call right after its last
                // parameter, without the closer: a body of complete parameters is a
                // call at the end of the stream. A header alone still is not.
                return match (flush, self.complete) {
                    (false, _) => BodyScan::Wait,
                    (true, 0) if !self.empty_body_ok => {
                        malformed(body.len(), "missing call terminator")
                    }
                    (true, _) => BodyScan::Complete {
                        body_end: at,
                        consume: at,
                    },
                };
            }
            if let Some(term) = [CALL, END].iter().find(|m| t.starts_with(**m)) {
                return BodyScan::Complete {
                    body_end: at,
                    consume: at + term.len(),
                };
            }
            // A framed next header never closes THIS call: no synthesized terminator.
            if t.starts_with(START) {
                return malformed(at, "missing call terminator");
            }
            if let Some(after) = t.strip_prefix(PARAM_OPEN) {
                // The opener is `name="KEY">` with a quote-free, non-empty key (the
                // template's tag pattern). Judged only once its bytes are all there.
                let key_end = match after.find('"') {
                    None if !flush => return BodyScan::Wait,
                    None => None,
                    Some(k) if after.len() == k + 1 && !flush => return BodyScan::Wait,
                    Some(k) if k > 0 && after[k..].starts_with("\">") => Some(k),
                    Some(_) => None,
                };
                let Some(key_end) = key_end else {
                    if lenient {
                        // A malformed opener is junk: the search resumes right after it.
                        self.scanned = at + PARAM_OPEN.len();
                        continue;
                    }
                    return self.unfinished(body, at, flush, "malformed parameter opener");
                };
                let value_from = at + PARAM_OPEN.len() + key_end + 2;
                let framing = value_framing(self.framing, &self.json_keys, &after[..key_end]);
                if self.open_from != value_from {
                    self.open_from = value_from;
                    self.open_value = ValueCursor::default();
                }
                match param_value_span_with(&body[value_from..], framing, &mut self.open_value) {
                    Some((_, next)) => {
                        self.scanned = value_from + next;
                        self.complete += 1;
                        continue;
                    }
                    None if !flush => return BodyScan::Wait,
                    // A parameter that never closes makes the call malformed in every
                    // reading: the builder's parser contract rejects a partial call
                    // (`malformed_parameter`, `truncated_parameter`) rather than serving
                    // the parameters around it. The region ends at the call's own
                    // terminator when one is already here, so what follows still parses.
                    None => return self.unclosed_parameter(body, at, flush),
                }
            }
            if !flush
                && [CALL, END, START, PARAM_OPEN]
                    .iter()
                    .any(|m| m.starts_with(t))
            {
                return BodyScan::Wait;
            }
            if lenient {
                match self.skip_junk(body, at, flush) {
                    Some(scan) => return scan,
                    None => continue,
                }
            }
            // Garbage: discard through the next terminator, or up to the next header.
            let term = [CALL, END]
                .iter()
                .filter_map(|m| t.find(m).map(|p| (p, p + m.len())))
                .min();
            let next = t.find(START);
            return match (term, next) {
                (Some((p, e)), s) if s.is_none_or(|s| p < s) => {
                    let _ = p;
                    malformed(at + e, "unexpected content in tool body")
                }
                (_, Some(s)) => malformed(at + s, "unexpected content in tool body"),
                _ if flush => malformed(body.len(), "unexpected content in tool body"),
                _ => BodyScan::Wait,
            };
        }
    }

    fn end_append(&mut self, _: &str, _: &str, _: bool, _: usize) -> Option<usize> {
        None
    }

    fn opens(&self, _: &str, _: usize) -> bool {
        false
    }

    fn holdback(&self, _: &str) -> usize {
        0
    }

    fn resync(&mut self, _: &str, _: bool, _: usize) -> Option<usize> {
        None
    }
}

impl BeamBoundary {
    /// Lenient reading of bytes at `at` that are not a parameter: the template's body
    /// grammar is a tag search, so they are skipped. The call still ends at its own
    /// terminator (junk before it stays in the body for the same lenient typed read),
    /// a framed next header still leaves it unterminated, and the stream end with
    /// neither is still a missing terminator. `None` means "resume at the next opener".
    fn skip_junk(&mut self, body: &str, at: usize, flush: bool) -> Option<BodyScan> {
        let t = &body[at..];
        let term = [CALL, END]
            .iter()
            .filter_map(|m| t.find(m).map(|p| (p, p + m.len())))
            .min();
        let next_header = t.find(START);
        // `t` never starts with the opener here, so a hit is strictly ahead.
        let next_param = t.find(PARAM_OPEN);
        let first = [term.map(|(p, _)| p), next_header, next_param]
            .into_iter()
            .flatten()
            .min();
        match first {
            Some(p) if next_header == Some(p) => Some(malformed(at + p, "missing call terminator")),
            Some(p) if term.is_some_and(|(tp, _)| tp == p) => {
                let (_, e) = term.expect("matched above");
                Some(BodyScan::Complete {
                    body_end: at + p,
                    consume: at + e,
                })
            }
            Some(p) => {
                self.scanned = at + p;
                None
            }
            None if flush => Some(malformed(body.len(), "missing call terminator")),
            None => Some(BodyScan::Wait),
        }
    }

    /// A parameter whose close never comes: the call is malformed. It is discarded
    /// through its own terminator when one is already in the buffer, up to the next
    /// framed header when that comes first, and whole at the end of the stream.
    fn unclosed_parameter(&self, body: &str, at: usize, flush: bool) -> BodyScan {
        let t = &body[at..];
        let term = [CALL, END]
            .iter()
            .filter_map(|m| t.find(m).map(|p| (p, p + m.len())))
            .min();
        let next_header = t.find(START);
        match (term, next_header) {
            (Some((p, e)), s) if s.is_none_or(|s| p < s) => malformed(at + e, "unclosed parameter"),
            (_, Some(s)) => malformed(at + s, "unclosed parameter"),
            _ if flush => malformed(body.len(), "unclosed parameter"),
            _ => BodyScan::Wait,
        }
    }

    /// A parameter that has not closed: wait, or at EOF discard it (up to the next
    /// header when one follows, so a later valid call survives).
    fn unfinished(&self, body: &str, at: usize, flush: bool, reason: &'static str) -> BodyScan {
        if !flush {
            return BodyScan::Wait;
        }
        let end = body[at..].find(START).map_or(body.len(), |p| at + p);
        malformed(end, reason)
    }
}

/// A tool-channel call whose body is still open: the JSON text handed out so far is a
/// prefix of what [`BeamEmitter::parse_channel_call`] produces for the finished body.
///
/// The walk is the lenient body reading, one parameter at a time. A parameter that
/// [`typed_value`] serves verbatim as a string ([`verbatim_string`]) under raw-text
/// framing streams as its bytes arrive (the framing newlines and a split
/// `</rfl:parameter>` are held back); any other parameter is typed and emitted whole at
/// its closer, since its JSON needs the complete text. Anything the typed reading would
/// skip or re-read (junk, a malformed opener, a duplicate key) stops streaming, and the
/// terminator reconciles: normally the completed JSON extends what was streamed and only
/// the remainder follows; when a later duplicate changed a value that is already out, the
/// changed members are appended again, so the concatenation is still one JSON object
/// whose last-wins reading is the typed call (the template's own duplicate rule).
///
/// A call that fails after fragments have streamed cannot be taken back: it stays
/// incomplete at its index (the assembler discards it) and the next call takes a new
/// index. That is the price of streaming; the builder's reference streams nothing and
/// so never shows a rejected call at all.
struct OpenCall {
    tool_index: usize,
    /// The recipient's typed `properties`, resolved once per call.
    properties: Option<Map<String, Value>>,
    json_keys: JsonKeys,
    /// JSON text already handed to the sink.
    emitted: String,
    /// Completed members as `(key, JSON value text)`, for reconciliation.
    members: Vec<(String, String)>,
    /// Body offset up to which parameters have been read.
    cursor: usize,
    keys: HashSet<String>,
    value: Option<OpenValue>,
    /// Streaming stopped; the terminator reconciles.
    blocked: bool,
}

/// The parameter whose value is open.
struct OpenValue {
    key: String,
    /// Body offset where the raw value begins (right after `">`).
    start: usize,
    /// Value bytes before this offset have been streamed (incremental values only).
    streamed: usize,
    /// Streamed as JSON string text; otherwise typed whole at the closer.
    incremental: bool,
    framing: ArgumentFraming,
    at_start: bool,
    pending_newline: bool,
    /// The JSON string text streamed so far (incremental values only), opening quote
    /// included.
    json_text: String,
    /// The walk over this value, resumed across pushes.
    cursor: ValueCursor,
}

impl OpenValue {
    /// Stream raw value bytes as JSON string text: the framing newline at the start is
    /// dropped, and the last newline is held until a later byte proves it is data.
    fn append(&mut self, raw: &str, out: &mut String) {
        let raw = if self.at_start {
            self.at_start = false;
            raw.strip_prefix('\n').unwrap_or(raw)
        } else {
            raw
        };
        if raw.is_empty() {
            return;
        }
        if self.pending_newline {
            dump_str_fragment("\n", out);
        }
        self.pending_newline = raw.ends_with('\n');
        let text = if self.pending_newline {
            &raw[..raw.len() - 1]
        } else {
            raw
        };
        let from = out.len();
        dump_str_fragment(text, out);
        self.json_text.push_str(&out[from..]);
    }
}

/// `{` before the first member, `, ` before every other one: the `json.dumps` separators.
fn push_separator(emitted: &str, pending: &mut String) {
    if emitted.is_empty() && pending.is_empty() {
        pending.push('{');
    } else {
        pending.push_str(", ");
    }
}

/// Types one complete body into the call's JSON arguments, streaming what it can of
/// them while the body is still open ([`OpenCall`]).
pub(crate) struct BeamEmitter {
    tools: Vec<Tool>,
    framing: ArgumentFraming,
    malformed: MalformedToolCallPolicy,
    /// JSON-owned keys per held tool name (as the serving grammar frames them), computed
    /// once per request so no chunk re-reads a schema.
    held_json_keys: HashMap<String, JsonKeys>,
    /// The call whose body is open, while any of its arguments have streamed.
    open: Option<OpenCall>,
}

impl BeamEmitter {
    fn new(tools: &[Tool]) -> Self {
        Self {
            tools: tools.to_vec(),
            framing: ArgumentFraming::default(),
            malformed: MalformedToolCallPolicy::default(),
            held_json_keys: held_json_keys(tools),
            open: None,
        }
    }

    /// The JSON-owned keys of `recipient`: empty for an unknown or unheld tool, never
    /// inherited from a previous recipient.
    fn json_keys(&self, recipient: &str) -> JsonKeys {
        self.held_json_keys
            .get(recipient)
            .cloned()
            .unwrap_or_default()
    }
}

impl InvokeEmitter for BeamEmitter {
    fn configure(&mut self, policy: ToolCallPolicy) {
        self.framing = policy.argument_framing;
        self.malformed = policy.malformed;
    }

    fn parse_invoke(
        &mut self,
        _invoke: &str,
        _tool_index: usize,
    ) -> anyhow::Result<Option<ToolCallDelta>> {
        Ok(None)
    }

    /// The SAME [`BeamBoundary`] walk the factory builds, carrying this recipient's
    /// [`JsonKeys`] so the streaming walk, the prose-quote lexer's ownership of a quoted
    /// call and [`Self::parse_channel_call`] agree about every embedded
    /// `</rfl:parameter>`.
    fn channel_boundary(&self, recipient: &str) -> Option<Box<dyn InvokeBoundary>> {
        Some(Box::new(BeamBoundary {
            framing: self.framing,
            malformed: self.malformed,
            json_keys: self.json_keys(recipient),
            scanned: 0,
            complete: 0,
            open_from: 0,
            open_value: ValueCursor::default(),
            empty_body_ok: recipient_admits_empty_arguments(&self.tools, recipient) == Some(true),
        }))
    }

    fn parse_channel_call(
        &mut self,
        recipient: &str,
        body: &str,
        tool_index: usize,
    ) -> anyhow::Result<Option<ToolCallDelta>> {
        // Decode first: on an error the open state stays, so `channel_call_done` can
        // report that fragments of this call are already out.
        let typed = typed_arguments(
            body,
            recipient,
            &self.tools,
            self.framing,
            &self.json_keys(recipient),
            self.malformed != MalformedToolCallPolicy::Reject,
        )?;
        let arguments = PyValue::Dict(typed.clone()).dumps();
        let Some(open) = self
            .open
            .take()
            .filter(|open| open.tool_index == tool_index && !open.emitted.is_empty())
        else {
            return Ok(Some(ToolCallDelta {
                tool_index,
                name: Some(recipient.to_string()),
                arguments,
                complete: true,
            }));
        };
        // The first fragment carried the name; the close carries only the remainder.
        if let Some(rest) = arguments.strip_prefix(open.emitted.as_str()) {
            return Ok(Some(ToolCallDelta {
                tool_index,
                name: None,
                arguments: rest.to_string(),
                complete: true,
            }));
        }
        // A later duplicate changed a value that is already out. Close what is open,
        // then append every member whose text is not already out: the concatenation is
        // one JSON object whose last-wins reading is the typed call.
        let mut members = open.members;
        let mut rest = String::new();
        if let Some(value) = open.value.filter(|value| value.incremental) {
            rest.push('"');
            members.push((value.key, format!("{}\"", value.json_text)));
        }
        for (key, value) in typed.iter() {
            let text = value.dumps();
            if members.iter().any(|(k, t)| k == key && *t == text) {
                continue;
            }
            rest.push_str(", ");
            dump_str(key, &mut rest);
            rest.push_str(": ");
            rest.push_str(&text);
        }
        rest.push('}');
        tracing::warn!(
            why = "beam_streamed_call_reconciled_after_duplicate_parameter",
            tool_index,
            "a later duplicate changed a streamed Beam argument; the member is appended again"
        );
        Ok(Some(ToolCallDelta {
            tool_index,
            name: None,
            arguments: rest,
            complete: true,
        }))
    }

    fn parse_partial_channel_call(
        &mut self,
        recipient: &str,
        body: &str,
        tool_index: usize,
    ) -> anyhow::Result<Option<ToolCallDelta>> {
        if self
            .open
            .as_ref()
            .is_none_or(|open| open.tool_index != tool_index)
        {
            // Honouring an unknown tool is the caller's policy: never stream one ahead
            // of that decision.
            let known = self.tools.iter().any(|tool| tool.name == recipient);
            self.open = Some(OpenCall {
                tool_index,
                properties: recipient_properties(&self.tools, recipient),
                json_keys: self.json_keys(recipient),
                emitted: String::new(),
                members: Vec::new(),
                cursor: 0,
                keys: HashSet::new(),
                value: None,
                blocked: !known,
            });
        }
        let framing = self.framing;
        let open = self.open.as_mut().expect("installed above");
        if open.blocked {
            return Ok(None);
        }
        let mut arguments = String::new();
        loop {
            if let Some(value) = open.value.as_mut() {
                let after_key = &body[value.start..];
                if value.incremental {
                    // Raw text: the value ends at its closer or where the next parameter
                    // opens (the serving grammar's implicit boundary), whichever is first.
                    let span = param_value_span_with(
                        after_key,
                        ArgumentFraming::Lenient,
                        &mut value.cursor,
                    );
                    let safe_end = match span {
                        Some((end, _)) => value.start + end,
                        None => {
                            body.len()
                                - marker_prefix_suffix_len(after_key, [PARAM_CLOSE, PARAM_FOLLOW])
                        }
                    };
                    if safe_end > value.streamed {
                        value.append(&body[value.streamed..safe_end], &mut arguments);
                        value.streamed = safe_end;
                    }
                    let Some((_, next)) = span else {
                        break;
                    };
                    arguments.push('"');
                    value.json_text.push('"');
                    open.cursor = value.start + next;
                    let value = open.value.take().expect("open value");
                    open.members.push((value.key, value.json_text));
                    continue;
                }
                let Some((end, next)) =
                    param_value_span_with(after_key, value.framing, &mut value.cursor)
                else {
                    break;
                };
                let schema = open.properties.as_ref().and_then(|p| p.get(&value.key));
                let typed = typed_value(schema, unframe(&after_key[..end]));
                let text = typed.dumps();
                push_separator(&open.emitted, &mut arguments);
                dump_str(&value.key, &mut arguments);
                arguments.push_str(": ");
                arguments.push_str(&text);
                open.cursor = value.start + next;
                let value = open.value.take().expect("open value");
                open.members.push((value.key, text));
                continue;
            }
            let rest = &body[open.cursor..];
            let t = rest.trim_start_matches([' ', '\t', '\n', '\r']);
            let at = open.cursor + (rest.len() - t.len());
            if t.is_empty() || [CALL, END].iter().any(|m| t.starts_with(*m)) {
                break;
            }
            let Some(after) = t.strip_prefix(PARAM_OPEN) else {
                if [CALL, END, START, PARAM_OPEN]
                    .iter()
                    .any(|m| m.starts_with(t))
                {
                    break;
                }
                open.blocked = true;
                break;
            };
            let key_end = match after.find('"') {
                None => break,
                Some(k) if after.len() == k + 1 => break,
                Some(k) if k > 0 && after[k..].starts_with("\">") => k,
                Some(_) => {
                    open.blocked = true;
                    break;
                }
            };
            let key = &after[..key_end];
            if !open.keys.insert(key.to_string()) {
                open.blocked = true;
                break;
            }
            let key_framing = value_framing(framing, &open.json_keys, key);
            let schema = open.properties.as_ref().and_then(|p| p.get(key));
            let incremental = key_framing == ArgumentFraming::Lenient && verbatim_string(schema);
            if incremental {
                push_separator(&open.emitted, &mut arguments);
                dump_str(key, &mut arguments);
                arguments.push_str(": \"");
            }
            let start = at + PARAM_OPEN.len() + key_end + 2;
            open.cursor = start;
            open.value = Some(OpenValue {
                key: key.to_string(),
                start,
                streamed: start,
                incremental,
                framing: key_framing,
                at_start: true,
                pending_newline: false,
                json_text: if incremental {
                    "\"".to_string()
                } else {
                    String::new()
                },
                cursor: ValueCursor::default(),
            });
        }
        if arguments.is_empty() {
            return Ok(None);
        }
        let first = open.emitted.is_empty();
        open.emitted.push_str(&arguments);
        Ok(Some(ToolCallDelta {
            tool_index,
            name: first.then(|| recipient.to_string()),
            arguments,
            complete: false,
        }))
    }

    fn channel_call_done(&mut self) -> bool {
        self.open
            .take()
            .is_some_and(|open| !open.emitted.is_empty())
    }

    fn reset(&mut self) {
        self.open = None;
    }
}

/// The one Beam state machine. The unified parser and the legacy tool-only projection
/// both wrap THIS scanner; there is no second Beam parser.
pub(crate) fn beam_scanner(tools: &[Tool]) -> WrappedBlockScanner<BeamEmitter> {
    let spec = WrappedBlockSpec {
        family: "beam",
        invoke_start: "<|start|>assistant to=".to_string(),
        invoke_end: CALL.to_string(),
        holdback_markers: [START, CHANNEL, MESSAGE, END, RETURN, CALL]
            .iter()
            .map(|m| m.to_string())
            .collect(),
        invoke_boundary_factory: Some(BOUNDARY),
        preserve_special_tokens: true,
        ..Default::default()
    };
    WrappedBlockScanner::new(spec, BeamEmitter::new(tools))
        .with_envelope(&ENVELOPE)
        .with_guided_channel(GUIDED_CHANNEL)
        .with_reasoning(ReasoningSpec {
            start: "<|channel|>analysis<|message|>",
            end: END,
            preserve_special_tokens: true,
            ..Default::default()
        })
}

/// Build the Beam unified parser. The malformed-call policy is request-scoped:
/// [`UnifiedParser::set_tool_call_policy`] before the first push.
pub(crate) fn beam_unified(tools: &[Tool]) -> Box<dyn UnifiedParser> {
    Box::new(GuidedRouted::new(ScannerUnified::new(beam_scanner(tools))))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unified::{
        MalformedToolCall, UnifiedEvent, UnifiedParserEvent, UnifiedParserExt, UnifiedParserInit,
        UnifiedParserStartingState, assemble,
    };
    use serde_json::json;

    fn tools() -> Vec<Tool> {
        let t = |name: &str, props: serde_json::Value| Tool {
            name: name.into(),
            description: None,
            parameters: json!({"type": "object", "properties": props}),
            strict: None,
        };
        vec![
            t("weather", json!({"location": {"type": "string"}})),
            t(
                "fn",
                json!({"b": {"type": "boolean"}, "n": {"type": "number"}}),
            ),
        ]
    }

    fn strict() -> ToolCallPolicy {
        ToolCallPolicy {
            malformed: MalformedToolCallPolicy::Reject,
            ..Default::default()
        }
    }

    fn run_with(
        tools: &[Tool],
        chunks: &[&str],
        policy: ToolCallPolicy,
    ) -> anyhow::Result<Vec<UnifiedEvent>> {
        let mut parser = beam_unified(tools);
        parser.set_tool_call_policy(policy)?;
        let mut deltas = Vec::new();
        for chunk in chunks {
            deltas.extend(parser.push(chunk)?);
        }
        deltas.extend(parser.finish()?);
        Ok(assemble(&deltas))
    }

    /// Whole, every UTF-8 split in two, and one char at a time.
    fn splits_with(
        tools: &[Tool],
        input: &str,
        policy: ToolCallPolicy,
    ) -> Vec<anyhow::Result<Vec<UnifiedEvent>>> {
        let mut all = vec![run_with(tools, &[input], policy)];
        for at in (0..=input.len()).filter(|&at| input.is_char_boundary(at)) {
            all.push(run_with(tools, &[&input[..at], &input[at..]], policy));
        }
        let chars: Vec<String> = input.chars().map(String::from).collect();
        let refs: Vec<&str> = chars.iter().map(String::as_str).collect();
        all.push(run_with(tools, &refs, policy));
        all
    }

    fn splits(input: &str, policy: ToolCallPolicy) -> Vec<anyhow::Result<Vec<UnifiedEvent>>> {
        splits_with(&tools(), input, policy)
    }

    fn expect_with(tools: &[Tool], input: &str, policy: ToolCallPolicy, want: Vec<UnifiedEvent>) {
        for got in splits_with(tools, input, policy) {
            assert_eq!(got.unwrap(), want, "{input}");
        }
    }

    fn expect(input: &str, want: Vec<UnifiedEvent>) {
        expect_with(&tools(), input, ToolCallPolicy::default(), want);
    }

    fn reasoning(t: &str) -> UnifiedEvent {
        UnifiedEvent::Reasoning { text: t.into() }
    }
    fn text(t: &str) -> UnifiedEvent {
        UnifiedEvent::Text { text: t.into() }
    }
    fn call(name: &str, arguments: serde_json::Value) -> UnifiedEvent {
        UnifiedEvent::ToolCall {
            name: name.into(),
            arguments,
        }
    }

    const HDR: &str =
        "<|start|>assistant to=weather<|channel|>analysis <|constrain|>xml<|message|>";
    const LONDON: &str = "<rfl:parameter name=\"location\">\nLondon\n</rfl:parameter>";

    /// Raw deltas (no assembly): `chunks`, then `finish`.
    fn deltas_with(tools: &[Tool], chunks: &[&str]) -> Vec<UnifiedParserEvent> {
        let mut parser = beam_unified(tools);
        let mut out = Vec::new();
        for chunk in chunks {
            out.extend(parser.push(chunk).unwrap());
        }
        out.extend(parser.finish().unwrap());
        out
    }

    fn call_deltas(events: &[UnifiedParserEvent]) -> Vec<&ToolCallDelta> {
        events
            .iter()
            .filter_map(|event| match event {
                UnifiedParserEvent::ToolCall(delta) => Some(delta),
                _ => None,
            })
            .collect()
    }

    fn argument_text(events: &[UnifiedParserEvent]) -> String {
        call_deltas(events)
            .iter()
            .map(|delta| delta.arguments.as_str())
            .collect()
    }

    /// Every split assembles to `want` AND its argument fragments concatenate to the
    /// whole-input text byte for byte: what streams is a prefix of the completed JSON.
    fn expect_streamed(tools: &[Tool], input: &str, want: Vec<UnifiedEvent>) {
        let whole = deltas_with(tools, &[input]);
        assert_eq!(assemble(&whole), want, "{input}");
        let whole_text = argument_text(&whole);
        for at in (0..=input.len()).filter(|&at| input.is_char_boundary(at)) {
            let split = deltas_with(tools, &[&input[..at], &input[at..]]);
            assert_eq!(argument_text(&split), whole_text, "split {at}: {input}");
            assert_eq!(assemble(&split), want, "split {at}: {input}");
        }
        let chars: Vec<String> = input.chars().map(String::from).collect();
        let refs: Vec<&str> = chars.iter().map(String::as_str).collect();
        let by_char = deltas_with(tools, &refs);
        assert_eq!(argument_text(&by_char), whole_text, "by char: {input}");
        assert_eq!(assemble(&by_char), want, "by char: {input}");
    }

    /// Tools the serving grammar admits: two string parameters, and names outside
    /// `[a-zA-Z_][a-zA-Z0-9_.-]*`.
    fn grammar_tools() -> Vec<Tool> {
        let t = |name: &str, props: serde_json::Value, required: &[&str]| Tool {
            name: name.into(),
            description: None,
            parameters: json!({"type": "object", "properties": props, "required": required}),
            strict: None,
        };
        vec![
            t(
                "t",
                json!({"a": {"type": "string"}, "b": {"type": "string"}}),
                &["a", "b"],
            ),
            t(
                "n",
                json!({"a": {"type": "string"}, "b": {"type": "integer"}}),
                &["a", "b"],
            ),
            t("2fa_check", json!({}), &[]),
            t("-tool", json!({}), &[]),
        ]
    }

    fn header(name: &str) -> String {
        format!("<|start|>assistant to={name}<|channel|>analysis <|constrain|>xml<|message|>")
    }

    #[test]
    fn a_parameter_may_end_where_the_next_one_opens() {
        // The serving grammar lets every parameter but the last end by opening the next
        // one without its closer; the parser reads the same boundary, raw and typed.
        let input = format!(
            "{}<rfl:parameter name=\"a\">\nA\n<rfl:parameter name=\"b\">\nB\n</rfl:parameter><|call|>",
            header("t")
        );
        expect_streamed(
            &grammar_tools(),
            &input,
            vec![call("t", json!({"a": "A", "b": "B"}))],
        );
        let input = format!(
            "{}<rfl:parameter name=\"a\">\nfirst\nline two\n<rfl:parameter name=\"b\">\n7\n</rfl:parameter><|call|>",
            header("n")
        );
        expect_streamed(
            &grammar_tools(),
            &input,
            vec![call("n", json!({"a": "first\nline two", "b": 7}))],
        );
    }

    #[test]
    fn grammar_admitted_tool_names_are_recipients() {
        for name in ["2fa_check", "-tool"] {
            let input = format!("{}<|call|>", header(name));
            expect_streamed(&grammar_tools(), &input, vec![call(name, json!({}))]);
        }
    }

    /// `noop` admits an empty body (strict, no properties); `maybe` has one optional
    /// property; `must` requires one.
    fn empty_body_tools() -> Vec<Tool> {
        let t = |name: &str, parameters: serde_json::Value| Tool {
            name: name.into(),
            description: None,
            parameters,
            strict: Some(true),
        };
        vec![
            t(
                "noop",
                json!({"type": "object", "properties": {}, "additionalProperties": false}),
            ),
            t(
                "maybe",
                json!({"type": "object", "properties": {"a": {"type": "string"}}, "additionalProperties": false}),
            ),
            t(
                "must",
                json!({"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"], "additionalProperties": false}),
            ),
        ]
    }

    #[test]
    fn a_header_alone_is_a_call_when_the_schema_admits_an_empty_body() {
        // The serving grammar makes the whole parameter body optional for such a tool
        // and lets EOS stand for the closer: the header is the call, under both
        // malformed-call policies and the schema-constrained framing.
        for malformed in [
            MalformedToolCallPolicy::Drop,
            MalformedToolCallPolicy::Reject,
        ] {
            let policy = ToolCallPolicy {
                malformed,
                argument_framing: ArgumentFraming::SchemaConstrained,
            };
            for name in ["noop", "maybe"] {
                for input in [header(name), format!("{}<|call|>", header(name))] {
                    for got in splits_with(&empty_body_tools(), &input, policy) {
                        assert_eq!(
                            got.unwrap(),
                            vec![call(name, json!({}))],
                            "{malformed:?} {input:?}"
                        );
                    }
                }
            }
            // A required argument keeps the header-only stream a truncation, and an open
            // parameter body stays one: no empty call is invented.
            for input in [
                header("must"),
                format!("{}<rfl:parameter name=\"a\">\nA", header("must")),
                format!("{}<rfl:parameter name=\"a\">\nA", header("maybe")),
            ] {
                for got in splits_with(&empty_body_tools(), &input, policy) {
                    match got {
                        Ok(events) => assert!(
                            events
                                .iter()
                                .all(|e| !matches!(e, UnifiedEvent::ToolCall { .. })),
                            "{malformed:?} {input:?}: {events:?}"
                        ),
                        Err(error) => assert_eq!(
                            malformed,
                            MalformedToolCallPolicy::Reject,
                            "{input:?}: {error}"
                        ),
                    }
                }
            }
        }
    }

    #[test]
    fn a_delimited_call_completes_at_the_end_of_the_stream() {
        // The serving grammar lets EOS end a call right after its last closer.
        let input = format!(
            "{}<rfl:parameter name=\"a\">\nA\n</rfl:parameter><rfl:parameter name=\"b\">\nB\n</rfl:parameter>",
            header("t")
        );
        expect_streamed(
            &grammar_tools(),
            &input,
            vec![call("t", json!({"a": "A", "b": "B"}))],
        );
        // A header alone, or a parameter still open, is not a call.
        for input in [
            header("t"),
            format!("{}<rfl:parameter name=\"a\">\nA", header("t")),
            format!(
                "{}<rfl:parameter name=\"a\">\nA\n</rfl:parameter><rfl:parameter name=\"b",
                header("t")
            ),
        ] {
            let events = deltas_with(&grammar_tools(), &[&input]);
            assert!(
                call_deltas(&events).iter().all(|delta| !delta.complete),
                "{input:?}: {events:?}"
            );
            assert!(
                assemble(&events)
                    .iter()
                    .all(|event| !matches!(event, UnifiedEvent::ToolCall { .. })),
                "{input:?}: {events:?}"
            );
        }
    }

    #[test]
    fn string_arguments_stream_ahead_of_the_call_terminator() {
        let value = "London, with \"quotes\", a \\ backslash and\ntwo lines of substantial text";
        let input =
            format!("{HDR}<rfl:parameter name=\"location\">\n{value}\n</rfl:parameter><|call|>");
        let want = vec![call("weather", json!({"location": value}))];
        expect_streamed(&tools(), &input, want.clone());

        // Char by char up to the parameter closer: the name and the value are already
        // out, nothing is complete, and the held framing newline has not leaked.
        let before_close = input.find(PARAM_CLOSE).unwrap();
        let mut parser = beam_unified(&tools());
        let mut early = Vec::new();
        for c in input[..before_close].chars() {
            early.extend(parser.push(&c.to_string()).unwrap());
        }
        let early_calls = call_deltas(&early);
        assert_eq!(
            early_calls.first().and_then(|d| d.name.as_deref()),
            Some("weather")
        );
        assert!(early_calls.iter().all(|d| !d.complete), "{early:?}");
        let early_text = argument_text(&early);
        assert!(
            early_text.starts_with("{\"location\": \"London, with \\\"quotes\\\""),
            "{early_text}"
        );
        assert!(early_text.ends_with("substantial text"), "{early_text}");
        // The closer and terminator complete the call with the remainder only.
        let mut all = early;
        all.extend(parser.push(&input[before_close..]).unwrap());
        all.extend(parser.finish().unwrap());
        let calls = call_deltas(&all);
        assert_eq!(calls.iter().filter(|d| d.name.is_some()).count(), 1);
        let last = calls.last().unwrap();
        assert!(last.complete && last.name.is_none(), "{last:?}");
        assert_eq!(last.arguments, "\"}");
        assert_eq!(assemble(&all), want);
    }

    #[test]
    fn typed_parameters_are_emitted_whole_at_their_closer() {
        let header = "<|start|>assistant to=fn<|channel|>analysis <|constrain|>xml<|message|>";
        let b = "<rfl:parameter name=\"b\">\ntrue\n</rfl:parameter>";
        let n = "<rfl:parameter name=\"n\">\n7\n</rfl:parameter>";
        let input = format!("{header}{b}{n}<|call|>");
        let want = vec![call("fn", json!({"b": true, "n": 7}))];
        expect_streamed(&tools(), &input, want.clone());

        let mut parser = beam_unified(&tools());
        let mut events = parser.push(&format!("{header}{b}")).unwrap();
        assert_eq!(argument_text(&events), "{\"b\": true");
        events.extend(parser.push(n).unwrap());
        assert_eq!(argument_text(&events), "{\"b\": true, \"n\": 7");
        events.extend(parser.push("<|call|>").unwrap());
        events.extend(parser.finish().unwrap());
        assert_eq!(argument_text(&events), "{\"b\": true, \"n\": 7}");
        assert_eq!(assemble(&events), want);

        let integer = declare("limit", json!({"count": {"type": "integer"}}), Some(false));
        expect_streamed(
            &[integer],
            &echo_call("limit", &[("count", "3.0")]),
            vec![call("limit", json!({"count": 3}))],
        );
    }

    #[test]
    fn a_duplicate_key_after_streaming_is_appended_again_so_the_last_one_wins() {
        let input = format!(
            "{HDR}{LONDON}<rfl:parameter name=\"location\">\nParis\n</rfl:parameter><|call|>"
        );
        let want = vec![call("weather", json!({"location": "Paris"}))];
        // Whole: the last duplicate wins, as in the typed reading.
        assert_eq!(deltas_with(&tools(), &[&input]).len(), 1);
        assert_eq!(
            argument_text(&deltas_with(&tools(), &[&input])),
            "{\"location\": \"Paris\"}"
        );
        // Streamed after the first value is out: it cannot be retracted, so the member
        // is appended again and every last-wins JSON reader still gets the typed call.
        let at = input.find("London").unwrap() + "London".len();
        let events = deltas_with(&tools(), &[&input[..at], &input[at..]]);
        assert_eq!(
            argument_text(&events),
            "{\"location\": \"London\", \"location\": \"Paris\"}"
        );
        assert_eq!(assemble(&events), want);
        // The same holds when the duplicate arrives while the first value is still open.
        let at = input.find("Lon").unwrap() + "Lon".len();
        let events = deltas_with(&tools(), &[&input[..at], &input[at..]]);
        assert_eq!(assemble(&events), want, "{events:?}");
        for got in splits(&input, ToolCallPolicy::default()) {
            assert_eq!(got.unwrap(), want);
        }
    }

    #[test]
    fn a_call_dropped_after_streaming_spends_its_index() {
        // The first call is left unterminated by the next framed header and is dropped;
        // what streamed of it stays incomplete at index 0 and the real call takes 1.
        let input = format!("{HDR}{LONDON}{HDR}{LONDON}<|call|>");
        let want = vec![call("weather", json!({"location": "London"}))];
        let at = input.find("Lon").unwrap() + "Lon".len();
        let events = deltas_with(&tools(), &[&input[..at], &input[at..]]);
        let calls = call_deltas(&events);
        assert!(
            calls.iter().any(|d| d.tool_index == 0 && !d.complete),
            "{events:?}"
        );
        assert!(
            calls.iter().all(|d| d.tool_index != 0 || !d.complete),
            "{events:?}"
        );
        assert!(
            calls.iter().any(|d| d.tool_index == 1 && d.complete),
            "{events:?}"
        );
        assert_eq!(assemble(&events), want);
        // Whole: nothing streamed, so the surviving call is index 0.
        let whole = deltas_with(&tools(), &[&input]);
        assert!(call_deltas(&whole).iter().all(|d| d.tool_index == 0));
        assert_eq!(assemble(&whole), want);
    }

    #[test]
    fn junk_between_parameters_stops_streaming_but_the_call_completes_identically() {
        let input = format!(
            "{HDR}{LONDON} junk <rfl:parameter name=bad>x</rfl:parameter>\
             <rfl:parameter name=\"extra\">\nmore\n</rfl:parameter><|call|>"
        );
        expect_streamed(
            &tools(),
            &input,
            vec![call(
                "weather",
                json!({"location": "London", "extra": "more"}),
            )],
        );
    }

    #[test]
    fn an_unknown_tool_never_streams_ahead_of_its_terminator() {
        let input = "<|start|>assistant to=mystery<|channel|>analysis <|constrain|>xml<|message|><rfl:parameter name=\"q\">\nwho\n</rfl:parameter><|call|>";
        let at = input.find("<|call|>").unwrap();
        let mut parser = beam_unified(&tools());
        let early = parser.push(&input[..at]).unwrap();
        assert!(call_deltas(&early).is_empty(), "{early:?}");
        let mut all = early;
        all.extend(parser.push(&input[at..]).unwrap());
        all.extend(parser.finish().unwrap());
        assert_eq!(assemble(&all), vec![call("mystery", json!({"q": "who"}))]);
    }

    #[test]
    fn thinking_and_final() {
        expect(
            "<|start|>assistant<|channel|>analysis<|message|>Thinking...<|end|><|start|>assistant<|channel|>final<|message|>Hi there!<|return|>",
            vec![reasoning("Thinking..."), text("Hi there!")],
        );
    }

    #[test]
    fn continuation_header_and_empty_analysis() {
        expect(
            "<|channel|>analysis<|message|>Check.<|end|><|start|>assistant<|channel|>final<|message|>391<|return|>",
            vec![reasoning("Check."), text("391")],
        );
        expect(
            "<|start|>assistant<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>Hello.<|return|>",
            vec![text("Hello.")],
        );
    }

    #[test]
    fn tool_call_with_trimmed_lf_and_prompt_consumed_prefix() {
        let want = vec![
            reasoning("Look it up."),
            call("weather", json!({"location": "London"})),
        ];
        expect(
            &format!(
                "<|start|>assistant<|channel|>analysis<|message|>Look it up.<|end|>{HDR}{LONDON}<|call|>"
            ),
            want.clone(),
        );
        expect(
            &format!(
                "<|channel|>analysis<|message|>Look it up.<|end|><|start|>assistant to=weather<|channel|>analysis <|constrain|>xml<|message|>{LONDON}<|call|>"
            ),
            want,
        );
        expect(
            &format!(" to=weather<|channel|>analysis <|constrain|>xml<|message|>{LONDON}<|call|>"),
            vec![call("weather", json!({"location": "London"}))],
        );
    }

    #[test]
    fn parallel_calls_in_order_with_interleaved_reasoning() {
        let input = format!(
            "{HDR}<rfl:parameter name=\"location\">a</rfl:parameter><|end|><|start|>assistant<|channel|>analysis<|message|>mid<|end|>{HDR}<rfl:parameter name=\"location\">b</rfl:parameter><|call|>"
        );
        expect(
            &input,
            vec![
                call("weather", json!({"location": "a"})),
                reasoning("mid"),
                call("weather", json!({"location": "b"})),
            ],
        );
    }

    #[test]
    fn framing_literals_in_arguments_are_data() {
        let value =
            " <|call|><|end|><|start|>assistant<|channel|>final<|message|> &amp; </rfl:paramet ";
        let input =
            format!("{HDR}<rfl:parameter name=\"location\">{value}</rfl:parameter><|call|>");
        expect(&input, vec![call("weather", json!({"location": value}))]);
    }

    #[test]
    fn empty_call_and_plain_text() {
        expect(&format!("{HDR}<|call|>"), vec![call("weather", json!({}))]);
        expect(
            "<|start|>assistant<|channel|>final<|message|>You can call get_weather.<|return|>",
            vec![text("You can call get_weather.")],
        );
    }

    #[test]
    fn schema_coercion_end_to_end() {
        let input = format!(
            "{}<rfl:parameter name=\"b\">\ntrue\n</rfl:parameter>\n<rfl:parameter name=\"n\">1.5</rfl:parameter><|call|>",
            HDR.replace("weather", "fn")
        );
        expect(&input, vec![call("fn", json!({"b": true, "n": 1.5}))]);
    }

    #[test]
    fn channel_specific_terminators() {
        expect(
            "<|channel|>analysis<|message|>a<|call|>b<|return|>c<|end|>",
            vec![reasoning("a<|call|>b<|return|>c")],
        );
        // The final answer closes at any of the grammar's three closers; what follows
        // the closed message is debris between messages.
        expect(
            "<|channel|>final<|message|>b<|call|>c<|return|>",
            vec![text("b")],
        );
    }

    /// The serving grammar admits `<|call|>` as the final answer's closer (it is a stop
    /// token like `<|return|>`), so the parser reads it as one: a `<|call|>` delivered
    /// after the answer ends it instead of reaching the user as text.
    #[test]
    fn a_call_marker_closes_the_final_answer_as_the_grammar_admits() {
        expect(
            "<|channel|>analysis<|message|>think<|end|><|start|>assistant<|channel|>final<|message|>answer<|call|>",
            vec![reasoning("think"), text("answer")],
        );
        expect(
            "<|channel|>final<|message|>answer<|return|>",
            vec![text("answer")],
        );
    }

    /// `reset` clears the JSON-body string state with the rest of the stream: a reply
    /// that ended inside a string must not make the next reply's terminator string
    /// data, nor a quoted marker in the next reply its terminator.
    #[test]
    fn reset_clears_the_json_body_state_of_a_prefilled_response() {
        for (previous, next, want) in [
            (
                "plain",
                "{\"new\":\"<|return|>\"}<|return|>",
                "{\"new\":\"<|return|>\"}",
            ),
            ("{\"old\":\"", "fresh<|return|>", "fresh"),
        ] {
            for at in (0..=next.len()).filter(|&at| next.is_char_boundary(at)) {
                let mut parser = beam_unified(&tools());
                parser
                    .initialize_request(UnifiedParserInit {
                        starting_state: UnifiedParserStartingState::Response,
                        ..UnifiedParserInit::default()
                    })
                    .expect("init");
                let _ = parser.push(previous).expect("push");
                let _ = parser.reset();
                let mut deltas = parser.push(&next[..at]).expect("push");
                deltas.extend(parser.push(&next[at..]).expect("push"));
                deltas.extend(parser.finish().expect("finish"));
                assert_eq!(
                    assemble(&deltas),
                    vec![text(want)],
                    "{previous:?} then {next:?} split {at}"
                );
            }
        }
    }

    /// The markers are special tokens, so a quotation never keeps one literal: the
    /// first `<|return|>` ends the reply even inside `"…"`, and an unbalanced quote in
    /// a thought cannot swallow the final answer (the serving reference reads both the
    /// same way). The reply after the odd quote also streams as it arrives instead of
    /// being held back for a closing quote that may never come.
    #[test]
    fn quotes_never_keep_a_control_literal_nor_hold_a_body_open() {
        expect(
            "<|channel|>final<|message|>Explain \"<|return|>\" literally.<|return|>",
            vec![text("Explain \"")],
        );
        expect(
            "<|channel|>final<|message|>say \"hi<|return|>",
            vec![text("say \"hi")],
        );
        let input = "<|channel|>analysis<|message|>He said \"stop<|end|><|start|>assistant<|channel|>final<|message|>She said \"go\".<|return|>";
        expect(
            input,
            vec![reasoning("He said \"stop"), text("She said \"go\".")],
        );
        let mut parser = beam_unified(&tools());
        parser
            .set_tool_call_policy(ToolCallPolicy::default())
            .unwrap();
        let mut streamed = Vec::new();
        for character in input.chars() {
            streamed.extend(parser.push(&character.to_string()).unwrap());
        }
        let finished: Vec<_> = parser.finish().unwrap().into_iter().collect();
        let at_finish = assemble(&finished);
        assert!(
            assemble(&streamed).contains(&text("She said \"go\".")),
            "the reply must stream before the end of the stream"
        );
        assert!(
            !at_finish
                .iter()
                .any(|e| matches!(e, UnifiedEvent::Text { text } if text.contains("She said"))),
            "nothing of the reply may be held back to finish: {at_finish:?}"
        );
    }

    #[test]
    fn bytes_outside_every_channel_are_dropped_not_text() {
        // Before the first header, between messages, after the last one, at EOF.
        expect("tail <|sta", vec![]);
        expect(
            "Sure.<|channel|>analysis<|message|>t<|end|>junk<|start|>assistant<|channel|>final<|message|>ok<|return|>trailer",
            vec![reasoning("t"), text("ok")],
        );
    }

    #[test]
    fn prompt_starting_state_reasoning_and_response_prefill() {
        for (state, input, want) in [
            (
                UnifiedParserStartingState::Reasoning,
                "thought<|end|><|start|>assistant<|channel|>final<|message|>hi<|return|>",
                vec![reasoning("thought"), text("hi")],
            ),
            (
                UnifiedParserStartingState::Response,
                "hi<|return|>",
                vec![text("hi")],
            ),
        ] {
            for got in {
                let mut all = Vec::new();
                for at in (0..=input.len()).filter(|&at| input.is_char_boundary(at)) {
                    let mut parser = beam_unified(&tools());
                    parser
                        .initialize_request(UnifiedParserInit {
                            starting_state: state,
                            ..UnifiedParserInit::default()
                        })
                        .expect("init");
                    let mut deltas = parser.push(&input[..at]).expect("push");
                    deltas.extend(parser.push(&input[at..]).expect("push"));
                    deltas.extend(parser.finish().expect("finish"));
                    all.push(assemble(&deltas));
                }
                all
            } {
                assert_eq!(got, want);
            }
        }
    }

    #[test]
    fn strict_rejects_unterminated_header_only_and_malformed() {
        // A delimited call that the stream ends after is a call (the serving grammar
        // lets EOS stand for the closer), under the strict policy too.
        for got in splits(&format!("{HDR}{LONDON}"), strict()) {
            assert_eq!(
                got.unwrap(),
                vec![call("weather", json!({"location": "London"}))]
            );
        }
        // `weather` requires nothing, so its header alone is the call `{}` as well.
        for got in splits(HDR, strict()) {
            assert_eq!(got.unwrap(), vec![call("weather", json!({}))]);
        }
        for tail in [
            format!("{HDR}<rfl:parameter name=\"location\">\nLondon\n"),
            format!("{HDR}<rfl:parameter name=\"location\">\nLondon\n</wrong><|call|>"),
            format!("{HDR}{LONDON}<|start|>assistant<|channel|>final<|message|>x<|return|>"),
        ] {
            for got in splits(&tail, strict()) {
                let error = got.expect_err("strict policy must reject");
                assert!(
                    error.downcast_ref::<MalformedToolCall>().is_some()
                        || error
                            .downcast_ref::<crate::tool_calling::beam_body::BeamBodyError>()
                            .is_some(),
                    "{error}"
                );
            }
            // `Drop` never dispatches a partial call either: an unterminated call and a
            // call with an unclosed parameter are both gone (the builder's parser
            // contract: `truncated_parameter`, `malformed_parameter`).
            for got in splits(&tail, ToolCallPolicy::default()) {
                assert!(
                    !got.unwrap()
                        .iter()
                        .any(|e| matches!(e, UnifiedEvent::ToolCall { .. })),
                    "{tail}"
                );
            }
        }
    }

    /// Under `Drop` the body is read as the template's tag search (goal 1): junk between
    /// parameters (including non-ASCII whitespace) and a malformed opener are skipped,
    /// every complete parameter counts, duplicates keep the last value, and a terminated
    /// call with no complete parameter is an empty call. A parameter that never closes
    /// still makes the call malformed (the builder's contract rejects partial calls),
    /// and a framed next header still leaves the call unterminated; in both cases the
    /// reply after the bad call is unaffected.
    #[test]
    fn drop_reads_the_body_as_the_template_tag_search() {
        expect(
            &format!(
                "{HDR}<rfl:parameter name=\"location\">\nParis\n</rfl:parameter>garbage<rfl:parameter name=\"q\"\">\nx\n</rfl:parameter>\u{2003}<rfl:parameter name=\"location\">\nLondon\n</rfl:parameter><|call|>"
            ),
            vec![call("weather", json!({"location": "London"}))],
        );
        expect(
            &format!("{HDR}garbage<rfl:parameter name=\"q\"\">\nx\n</rfl:parameter><|call|>"),
            vec![call("weather", json!({}))],
        );
        let reply = "<|start|>assistant<|channel|>final<|message|>ok<|return|>";
        expect(
            &format!(
                "{HDR}<rfl:parameter name=\"location\">\nParis\n</rfl:parameter><rfl:parameter name=\"unit\">\nopen<|call|>{reply}"
            ),
            vec![text("ok")],
        );
        expect(
            &format!("{HDR}<rfl:parameter name=\"location\">\nParis\n</rfl:parameter>junk{reply}"),
            vec![text("ok")],
        );
    }

    #[test]
    fn strict_rejects_unfinished_and_illegal_tool_headers() {
        for input in [
            "<|start|>assistant to=fn".to_string(),
            "<|start|>assistant to=fn<|channel|>final <|constrain|>xml<|message|><rfl:parameter name=\"b\">true</rfl:parameter><|call|>".to_string(),
        ] {
            for got in splits(&input, strict()) {
                assert!(got.is_err(), "{input}");
            }
            for got in splits(&input, ToolCallPolicy::default()) {
                assert!(
                    !got.unwrap()
                        .iter()
                        .any(|e| matches!(e, UnifiedEvent::ToolCall { .. })),
                    "{input}"
                );
            }
        }
        let mut parser = beam_unified(&tools());
        parser.set_tool_call_policy(strict()).unwrap();
        parser.push("<|start|>assistant to=fn").unwrap();
        assert!(parser.finish().is_err());
        assert_eq!(parser.reset(), "<|start|>assistant to=fn");
    }

    #[test]
    fn drop_discards_only_the_bad_region_independent_of_chunking() {
        let input = format!("{HDR}{LONDON}{HDR}{LONDON}<|call|>");
        expect(&input, vec![call("weather", json!({"location": "London"}))]);
    }

    #[test]
    fn reject_leaves_committed_output_and_lossless_reset() {
        let mut parser = beam_unified(&tools());
        parser.set_tool_call_policy(strict()).unwrap();
        let mut out = crate::unified::UnifiedParserOutput::default();
        let input = format!(
            "<|start|>assistant<|channel|>analysis<|message|>think<|end|>{HDR}<rfl:parameter name=\"location\">pending"
        );
        parser.parse_into(&input, &mut out).unwrap();
        assert!(parser.finish().is_err());
        assert_eq!(out.assembled(), vec![reasoning("think")]);
        assert_eq!(
            parser.reset(),
            format!("{HDR}<rfl:parameter name=\"location\">pending")
        );
    }

    #[test]
    fn strict_rejects_header_truncated_inside_channel_marker_and_reset_keeps_start() {
        // A complete header for `fn` (nothing required) is the call `{}` at EOS.
        let complete = "<|start|>assistant to=fn<|channel|>analysis <|constrain|>xml<|message|>";
        for policy in [strict(), ToolCallPolicy::default()] {
            for got in splits(complete, policy) {
                assert_eq!(got.unwrap(), vec![call("fn", json!({}))]);
            }
        }
        for input in [
            "<|start|>assistant to=fn<|chan",
            "<|start|>assistant to=fn<|channel|>",
            "<|start|>assistant to=fn<|channel|>analysis <|constr",
        ] {
            for got in splits(input, strict()) {
                let error = got.expect_err(input);
                assert!(
                    error.downcast_ref::<MalformedToolCall>().is_some(),
                    "{error}"
                );
            }
            for got in splits(input, ToolCallPolicy::default()) {
                assert_eq!(got.unwrap(), vec![], "{input}");
            }
            let mut parser = beam_unified(&tools());
            parser.set_tool_call_policy(strict()).unwrap();
            let _ = parser.push(input);
            assert!(parser.finish().is_err(), "{input}");
            assert_eq!(parser.reset(), input, "reset must return every header byte");
        }
        // Ordinary text and ambiguous `to=` prose are never a tool error.
        for input in [
            "<|start|>hello to= you",
            "walk me to=",
            "<|start|>assistant<|chan",
        ] {
            for got in splits(input, strict()) {
                got.unwrap_or_else(|e| panic!("{input}: {e}"));
            }
        }
        expect("<|start|>hello to= you", vec![]);
        expect("<|start|>assistant<|chan", vec![]);
    }

    #[test]
    fn idle_call_marker_is_orphan_and_closes_the_final_answer() {
        expect("<|call|>", vec![]);
        expect("before<|call|>after", vec![]);
        expect(
            "<|channel|>final<|message|>say <|call|> now<|return|><|call|>",
            vec![text("say ")],
        );
        expect(
            &format!("{HDR}{LONDON}<|call|><|call|><|end|>"),
            vec![call("weather", json!({"location": "London"}))],
        );
    }

    #[test]
    fn response_prefill_framed_thought_is_visible_text_with_framing_consumed() {
        let input = format!(
            "<|start|>assistant<|channel|>analysis<|message|>literal<|end|> then a call{HDR}<rfl:parameter name=\"location\">\nParis\n</rfl:parameter><|call|>"
        );
        for at in (0..=input.len()).filter(|&at| input.is_char_boundary(at)) {
            let mut parser = beam_unified(&tools());
            parser
                .initialize_request(UnifiedParserInit {
                    starting_state: UnifiedParserStartingState::Response,
                    ..UnifiedParserInit::default()
                })
                .unwrap();
            let mut deltas = parser.push(&input[..at]).unwrap();
            deltas.extend(parser.push(&input[at..]).unwrap());
            deltas.extend(parser.finish().unwrap());
            assert_eq!(
                assemble(&deltas),
                vec![
                    text("literal then a call"),
                    call("weather", json!({"location": "Paris"}))
                ],
                "split {at}"
            );
        }
    }

    fn constrained() -> ToolCallPolicy {
        ToolCallPolicy {
            argument_framing: ArgumentFraming::ConstrainedJson,
            ..Default::default()
        }
    }

    #[test]
    fn constrained_json_waits_for_an_open_json_value_at_every_split() {
        let t = Tool {
            name: "fn".into(),
            description: None,
            parameters: json!({"type": "object", "properties": {"o": {"type": "object"}, "s": {"type": "string"}}}),
            strict: None,
        };
        let hdr = "<|start|>assistant to=fn<|channel|>analysis <|constrain|>xml<|message|>";
        let object = r#"{"text":"</rfl:parameter><|call|>payload \" \\ ünï"}"#;
        let string = r#""a</rfl:parameter><|call|> \"q\" ü""#;
        let array = r#"["</rfl:parameter>", "<|call|>"]"#;
        for (key, value, want) in [
            (
                "o",
                object,
                json!({"o": {"text": "</rfl:parameter><|call|>payload \" \\ ünï"}}),
            ),
            ("s", string, json!({"s": string})),
            ("o", array, json!({"o": array})),
            ("s", "London", json!({"s": "London"})),
        ] {
            let input = format!(
                "{hdr}<rfl:parameter name=\"{key}\">{value}</rfl:parameter><|call|><|start|>assistant<|channel|>final<|message|>tail<|return|>"
            );
            let wanted = vec![call("fn", want), text("tail")];
            let run = |chunks: &[&str]| {
                let mut parser = beam_unified(std::slice::from_ref(&t));
                parser.set_tool_call_policy(constrained()).unwrap();
                let mut deltas = Vec::new();
                for chunk in chunks {
                    deltas.extend(parser.push(chunk).unwrap());
                }
                deltas.extend(parser.finish().unwrap());
                assemble(&deltas)
            };
            assert_eq!(run(&[&input]), wanted, "{input}");
            for at in (0..=input.len()).filter(|&at| input.is_char_boundary(at)) {
                assert_eq!(
                    run(&[&input[..at], &input[at..]]),
                    wanted,
                    "split {at}: {input}"
                );
            }
            let chars: Vec<String> = input.chars().map(String::from).collect();
            let refs: Vec<&str> = chars.iter().map(String::as_str).collect();
            assert_eq!(run(&refs), wanted, "{input}");
        }
        // An open JSON value at EOF is an unclosed parameter, never a premature call.
        let open = format!("{hdr}<rfl:parameter name=\"o\">{{\"text\":\"</rfl:parameter><|call|>");
        let mut parser = beam_unified(std::slice::from_ref(&t));
        parser.set_tool_call_policy(constrained()).unwrap();
        let mut deltas = parser.push(&open).unwrap();
        deltas.extend(parser.finish().unwrap());
        assert_eq!(assemble(&deltas), vec![]);
        // Lenient keeps the reference rule: the first closer ends the value.
        let mut parser = beam_unified(std::slice::from_ref(&t));
        let mut deltas = parser.push(&open).unwrap();
        deltas.extend(parser.finish().unwrap());
        assert_eq!(
            assemble(&deltas),
            vec![call("fn", json!({"o": "{\"text\":\""}))]
        );
    }

    use crate::unified::InvalidGuidedPayloadPolicy;

    const POLICIES: [InvalidGuidedPayloadPolicy; 3] = [
        InvalidGuidedPayloadPolicy::RecoverAsText,
        InvalidGuidedPayloadPolicy::Reject,
        InvalidGuidedPayloadPolicy::StreamBestEffort,
    ];

    fn guided_with(
        state: UnifiedParserStartingState,
        named: Option<&str>,
        policy: InvalidGuidedPayloadPolicy,
        chunks: &[&str],
    ) -> Vec<UnifiedEvent> {
        let mut parser = beam_unified(&tools());
        parser
            .initialize_request(UnifiedParserInit {
                starting_state: state,
                tool_output_mode: crate::unified::UnifiedToolOutputMode::GuidedJson {
                    named_tool: named.map(str::to_string),
                },
                invalid_guided_payload: policy,
                ..UnifiedParserInit::default()
            })
            .unwrap();
        let mut deltas = Vec::new();
        for chunk in chunks {
            deltas.extend(parser.push(chunk).expect("push"));
        }
        deltas.extend(parser.finish().expect("finish"));
        assemble(&deltas)
    }

    fn guided(
        state: UnifiedParserStartingState,
        named: Option<&str>,
        chunks: &[&str],
    ) -> Vec<UnifiedEvent> {
        guided_with(
            state,
            named,
            InvalidGuidedPayloadPolicy::RecoverAsText,
            chunks,
        )
    }

    fn guided_every_split_with(
        state: UnifiedParserStartingState,
        named: Option<&str>,
        policy: InvalidGuidedPayloadPolicy,
        input: &str,
        want: Vec<UnifiedEvent>,
    ) {
        assert_eq!(
            guided_with(state, named, policy, &[input]),
            want,
            "whole ({policy:?}): {input}"
        );
        for at in (1..input.len()).filter(|&at| input.is_char_boundary(at)) {
            assert_eq!(
                guided_with(state, named, policy, &[&input[..at], &input[at..]]),
                want,
                "split {at} ({policy:?}): {input}"
            );
        }
        let chars: Vec<String> = input.chars().map(String::from).collect();
        let refs: Vec<&str> = chars.iter().map(String::as_str).collect();
        assert_eq!(
            guided_with(state, named, policy, &refs),
            want,
            "per char ({policy:?}): {input}"
        );
    }

    fn guided_every_split(
        state: UnifiedParserStartingState,
        named: Option<&str>,
        input: &str,
        want: Vec<UnifiedEvent>,
    ) {
        guided_every_split_with(
            state,
            named,
            InvalidGuidedPayloadPolicy::RecoverAsText,
            input,
            want,
        );
    }

    const PAYLOAD: &str = r#"[{"name": "weather", "arguments": {"location": "a > b"}}]"#;
    const NAMED: &str = r#"{"location": "a > b"}"#;
    const TOOL_HDR: &str =
        "<|start|>assistant to=NAMEX<|channel|>analysis <|constrain|>xml<|message|>";

    fn paris() -> UnifiedEvent {
        call("weather", json!({"location": "a > b"}))
    }

    #[test]
    fn guided_headers_around_the_payload_follow_the_envelope_grammar() {
        use UnifiedParserStartingState::None as N;
        let want = vec![paris()];
        for input in [
            PAYLOAD.to_string(),
            format!("{TOOL_HDR}{PAYLOAD}<|call|>"),
            format!("{TOOL_HDR}{PAYLOAD}"),
            format!("{PAYLOAD}<|call|>"),
            format!("<|start|>assistant to={PAYLOAD}"),
            format!("<|end|>{PAYLOAD}"),
            format!("<|call|>{PAYLOAD}"),
        ] {
            guided_every_split(N, None, &input, want.clone());
        }
        guided_every_split(
            N,
            Some("weather"),
            &format!("{TOOL_HDR}{NAMED}<|call|>"),
            want.clone(),
        );
        // A whole native call under guided decoding is one control-only turn.
        guided_every_split(N, None, &format!("{HDR}{LONDON}<|call|>"), vec![]);
        // Reasoning around the payload, and a narrated header inside the thought.
        guided_every_split(
            N,
            None,
            &format!("<|start|>assistant<|channel|>analysis<|message|>checking<|end|>{PAYLOAD}"),
            vec![reasoning("checking"), paris()],
        );
        guided_every_split(
            N,
            None,
            &format!("Sure. <|channel|>analysis<|message|>checking<|end|>{PAYLOAD}"),
            vec![text("Sure. "), reasoning("checking"), paris()],
        );
        guided_every_split(
            N,
            None,
            &format!(
                "<|start|>assistant<|channel|>analysis<|message|>I'll use {TOOL_HDR} next<|end|>{PAYLOAD}"
            ),
            vec![reasoning("I'll use "), paris()],
        );
        guided_every_split(
            N,
            None,
            &format!(
                "<|start|>assistant<|channel|>analysis<|message|>I'll call <|start|>assistant to=weather<|end|>{PAYLOAD}"
            ),
            vec![reasoning("I'll call weather"), paris()],
        );
        guided_every_split(
            N,
            None,
            &format!(
                "<|start|>assistant to=<|start|>assistant<|channel|>analysis<|message|>secret<|end|>{PAYLOAD}"
            ),
            vec![reasoning("secret"), paris()],
        );
        // An unterminated thought ends at the wrapper that leads into the payload.
        guided_every_split(
            N,
            None,
            &format!(
                "<|start|>assistant<|channel|>analysis<|message|>thinking{TOOL_HDR}{PAYLOAD}<|call|>"
            ),
            vec![reasoning("thinking"), paris()],
        );
        guided_every_split(
            UnifiedParserStartingState::Reasoning,
            None,
            &format!("thinking {TOOL_HDR}{PAYLOAD}<|call|>"),
            vec![reasoning("thinking "), paris()],
        );
        // A control marker inside a started payload is argument data.
        guided_every_split(
            N,
            None,
            r#"[{"name": "weather", "arguments": {"location": "<|end|>"}}]"#,
            vec![call("weather", json!({"location": "<|end|>"}))],
        );
    }

    /// A narrated header lead inside a thought that a CONTROL ends (`to=weather<|end|>`)
    /// is judged only once that control has arrived whole. Split inside the marker
    /// (`<|e` / `nd|>`) the lead must stay held, not be stripped through its `start`
    /// marker alone and leak `assistant to=` as reasoning text. Inside analysis
    /// `<|call|>` and `<|return|>` are text, so they decide the lead the same way.
    #[test]
    fn guided_narrated_lead_before_a_split_control_stays_held() {
        use UnifiedParserStartingState::None as N;
        guided_every_split(
            N,
            None,
            &format!(
                "<|start|>assistant<|channel|>analysis<|message|>I'll call <|start|>assistant to=weather<|end|>{PAYLOAD}"
            ),
            vec![reasoning("I'll call weather"), paris()],
        );
        for control in [CALL, RETURN] {
            guided_every_split(
                N,
                None,
                &format!(
                    "<|start|>assistant<|channel|>analysis<|message|>I'll call <|start|>assistant to=weather{control} now<|end|>{PAYLOAD}"
                ),
                vec![
                    reasoning(&format!("I'll call weather{control} now")),
                    paris(),
                ],
            );
        }
    }

    /// Headers arriving one special token at a time, as the live corpus captures
    /// them. Every byte from a header's first one is owned by that header until it
    /// resolves: a `<|channel|>` chunk inside it is not a stray control, and the lead in
    /// front of it never leaks as text.
    #[test]
    fn guided_headers_split_at_token_boundaries_are_owned_whole() {
        use UnifiedParserStartingState::None as N;
        let thought: [&str; 7] = [
            "<|start|>assistant",
            "<|channel|>",
            "analysis",
            "<|message|>",
            "checking",
            "<|end|>",
            PAYLOAD,
        ];
        assert_eq!(
            guided(N, None, &thought),
            vec![reasoning("checking"), paris()]
        );
        let tool: [&str; 6] = [
            "<|start|>assistant to=NAMEX",
            "<|channel|>",
            "analysis <|constrain|>xml",
            "<|message|>",
            PAYLOAD,
            "<|call|>",
        ];
        assert_eq!(guided(N, None, &tool), vec![paris()]);
        // Inside an open thought the same tokens are owned the same way.
        let narrated: [&str; 8] = [
            "<|start|>assistant<|channel|>analysis<|message|>I'll use ",
            "<|start|>assistant to=NAMEX",
            "<|channel|>",
            "analysis <|constrain|>xml",
            "<|message|>",
            " next",
            "<|end|>",
            PAYLOAD,
        ];
        assert_eq!(
            guided(N, None, &narrated),
            vec![reasoning("I'll use "), paris()]
        );
    }

    /// The template's own continuation after a prompt that consumed `<|start|>`:
    /// the role word is the header's lead, not the user's text, under every policy
    /// and tool choice.
    #[test]
    fn guided_unframed_role_lead_opens_the_thought() {
        use UnifiedParserStartingState::None as N;
        let want = vec![reasoning("checking 🧪"), paris()];
        for policy in POLICIES {
            guided_every_split_with(
                N,
                None,
                policy,
                &format!("assistant<|channel|>analysis<|message|>checking 🧪<|end|>{PAYLOAD}"),
                want.clone(),
            );
            guided_every_split_with(
                N,
                Some("weather"),
                policy,
                &format!("assistant<|channel|>analysis<|message|>checking 🧪<|end|>{NAMED}"),
                want.clone(),
            );
        }
        // A role word that is NOT a lead keeps its prose and spacing.
        guided_every_split(
            N,
            None,
            &format!("I am an assistant. <|channel|>analysis<|message|>checking<|end|>{PAYLOAD}"),
            vec![text("I am an assistant. "), reasoning("checking"), paris()],
        );
    }

    /// Body terminators follow the channel grammar in every request mode: analysis
    /// ends only at `<|end|>` (`<|call|>` and `<|return|>` are text there), and nothing
    /// unframed is a header inside an open body. Under guided decoding the answer body
    /// ends only at the turn end (`<|return|>`); the native drain also closes it at
    /// `<|call|>`, the grammar's third closer.
    #[test]
    fn guided_body_terminators_follow_the_channel_grammar() {
        use UnifiedParserStartingState::{None as N, Reasoning as R};
        guided_every_split(
            N,
            None,
            &format!(
                "<|start|>assistant<|channel|>analysis<|message|>a<|call|>b<|return|>c<|end|>{PAYLOAD}"
            ),
            vec![reasoning("a<|call|>b<|return|>c"), paris()],
        );
        guided_every_split(
            R,
            None,
            &format!("a<|call|>b<|return|>c<|end|>{PAYLOAD}"),
            vec![reasoning("a<|call|>b<|return|>c"), paris()],
        );
        guided_every_split(
            N,
            None,
            "<|start|>assistant<|channel|>analysis<|message|>t<|end|><|start|>assistant<|channel|>final<|message|>b<|call|>c<|return|>",
            vec![reasoning("t"), text("b<|call|>c")],
        );
        guided_every_split(
            N,
            None,
            &format!("{PAYLOAD}<|start|>assistant<|channel|>final<|message|>done<|return|>"),
            vec![paris(), text("done")],
        );
        // A bare `to=` header inside an open thought is narration the native body
        // keeps literal; only a framed `<|start|>` interrupts a body.
        let narrated = "use to=weather<|channel|>analysis <|constrain|>xml<|message|>later";
        guided_every_split(
            N,
            None,
            &format!("<|start|>assistant<|channel|>analysis<|message|>{narrated}<|end|>{PAYLOAD}"),
            vec![reasoning(narrated), paris()],
        );
        guided_every_split(
            R,
            None,
            &format!("{narrated}<|end|>{PAYLOAD}"),
            vec![reasoning(narrated), paris()],
        );
    }

    /// Under a RESPONSE prefill the framing tokens are literal prose, but a tool-routed
    /// message is still a native call: it is stripped whole (header, native body and
    /// terminator) before the guided payload, at every chunking, and an argument value
    /// shaped like the payload stays inside it. The prose before the header and after the
    /// terminator's structural whitespace is the answer.
    #[test]
    fn guided_response_prefill_strips_a_native_wrapper_whole() {
        let tools = vec![
            Tool {
                name: "get_weather".into(),
                description: None,
                parameters: json!({"type": "object", "properties": {"city": {"type": "string"}}}),
                strict: None,
            },
            Tool {
                name: "f".into(),
                description: None,
                parameters: json!({"type": "object", "properties": {"x": {"type": "string"}}}),
                strict: None,
            },
        ];
        let run = |chunks: &[&str]| -> Vec<UnifiedEvent> {
            let mut parser = beam_unified(&tools);
            parser
                .initialize_request(UnifiedParserInit {
                    starting_state: UnifiedParserStartingState::Response,
                    tool_output_mode: crate::unified::UnifiedToolOutputMode::GuidedJson {
                        named_tool: None,
                    },
                    ..UnifiedParserInit::default()
                })
                .unwrap();
            let mut deltas = Vec::new();
            for chunk in chunks {
                deltas.extend(parser.push(chunk).expect("push"));
            }
            deltas.extend(parser.finish().expect("finish"));
            assemble(&deltas)
        };
        let payload = r#"[{"name": "get_weather", "arguments": {"city": "Paris"}}]"#;
        let paris = call("get_weather", json!({"city": "Paris"}));
        let header = "<|start|>assistant to=f<|channel|>analysis <|constrain|>xml<|message|>";
        let quoted = "I mean <|start|>assistant<|channel|>analysis<|message|>self literal<|end|>";
        for body in [
            "{\"city\": \"Rome\"}",
            "[{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Rome\"}}]",
        ] {
            let wrapper =
                format!("{header}<rfl:parameter name=\"x\">\n{body}\n</rfl:parameter><|call|>");
            let input = format!("{quoted}{wrapper}{payload}");
            let want = vec![text(quoted), paris.clone()];
            assert_eq!(run(&[&input]), want, "whole: {input}");
            for at in (1..input.len()).filter(|&at| input.is_char_boundary(at)) {
                assert_eq!(
                    run(&[&input[..at], &input[at..]]),
                    want,
                    "split {at}: {input}"
                );
            }
            let input = format!("{{}}{quoted}{wrapper}{payload}");
            assert_eq!(
                run(&[&input]),
                vec![text(&format!("{{}}{quoted}")), paris.clone()]
            );
            let input = format!("{wrapper} \n\t ok then {payload}");
            let want = vec![text("ok then "), paris.clone()];
            assert_eq!(run(&[&input]), want, "whole: {input}");
            for at in (1..input.len()).filter(|&at| input.is_char_boundary(at)) {
                assert_eq!(
                    run(&[&input[..at], &input[at..]]),
                    want,
                    "split {at}: {input}"
                );
            }
        }
    }

    #[test]
    fn guided_response_prefill_keeps_quoted_framing_literal() {
        use UnifiedParserStartingState::Response as R;
        let quoted = "I mean <|start|>assistant<|channel|>analysis<|message|>self literal<|end|>";
        guided_every_split(
            R,
            None,
            &format!("{quoted}{PAYLOAD}"),
            vec![text(quoted), paris()],
        );
        guided_every_split(
            R,
            None,
            &format!("{PAYLOAD}{quoted}"),
            vec![paris(), text(quoted)],
        );
        let braces =
            "The literal \"<|start|>assistant to=weather<|channel|> { example }\" stays visible. ";
        guided_every_split(
            R,
            None,
            &format!("{braces}{PAYLOAD}"),
            vec![text(braces), paris()],
        );
        guided_every_split(
            R,
            Some("weather"),
            &format!("{braces}{NAMED}"),
            vec![text(braces), paris()],
        );
        // Prefilled reasoning quoting its closer inside a contraction-bearing quotation.
        let thought = "The literal 'doesn't <|end|> marker' stays quoted.";
        guided_every_split(
            UnifiedParserStartingState::Reasoning,
            None,
            &format!("{thought}<|end|>{PAYLOAD}"),
            vec![reasoning(thought), paris()],
        );
    }

    /// An OUTER native call whose string parameter quotes a WHOLE framed message —
    /// header, body and terminator — followed by the guided payload. The outer call is
    /// stripped whole (the native-markup-only contract above) and only the payload
    /// dispatches: nothing inside the parameter value is a thought, a content switch
    /// or a terminator, at any chunking. Resolving every header-shaped run on its own
    /// used to release the outer header as text and the quoted body as reasoning (or
    /// route the turn to content and lose the call).
    #[test]
    fn guided_outer_native_parameter_owns_framed_markup_inside_its_value() {
        use UnifiedParserStartingState::None as N;
        let payload = r#"[{"name": "fn", "arguments": {"x": "Paris"}}]"#;
        let want = vec![call("fn", json!({"x": "Paris"}))];
        for value in [
            "<|start|>assistant<|channel|>analysis<|message|>inside<|end|>",
            "<|start|>assistant<|channel|>final<|message|>inside<|return|>",
        ] {
            let input = format!(
                "<|start|>assistant to=fn<|channel|>analysis <|constrain|>xml<|message|><rfl:parameter name=\"x\">{value}</rfl:parameter><|call|>{payload}"
            );
            for policy in POLICIES {
                guided_every_split_with(N, None, policy, &input, want.clone());
            }
            // The live corpus delivers the header one special token at a time.
            let chunks: Vec<&str> = [
                "<|start|>assistant to=fn",
                "<|channel|>",
                "analysis <|constrain|>xml",
                "<|message|>",
                "<rfl:parameter name=\"x\">",
                value,
                "</rfl:parameter>",
                "<|call|>",
                payload,
            ]
            .to_vec();
            assert_eq!(guided(N, None, &chunks), want, "{input}");
        }
    }

    /// A header the model never finished at EOF is framing debris, not the answer. The
    /// native drain emits nothing for `assistant<|channel|>analysis`; under guided
    /// decoding the same bytes were released whole as text once holdback no longer
    /// applied. Only a run the scope already recognises as a header-in-progress is
    /// discarded: a role word that is plain prose stays visible.
    #[test]
    fn guided_unfinished_unframed_header_at_eof_is_discarded() {
        use UnifiedParserStartingState::None as N;
        for input in [
            "assistant<|channel|>analysis",
            "assistant to=fn<|channel|>analysis",
        ] {
            expect(input, vec![]);
            guided_every_split(N, None, input, vec![]);
        }
        guided_every_split(
            N,
            None,
            "I am an assistant",
            vec![text("I am an assistant")],
        );
    }

    #[test]
    fn constrained_json_protects_closer_inside_json_string() {
        let t = Tool {
            name: "fn".into(),
            description: None,
            parameters: json!({"type": "object", "properties": {"o": {"type": "object"}}}),
            strict: None,
        };
        let value = r#"{"s": "a</rfl:parameter>b"}"#;
        let input = format!(
            "<|start|>assistant to=fn<|channel|>analysis <|constrain|>xml<|message|><rfl:parameter name=\"o\">{value}</rfl:parameter><|call|>"
        );
        let mut parser = beam_unified(&[t]);
        parser
            .set_tool_call_policy(ToolCallPolicy {
                argument_framing: ArgumentFraming::ConstrainedJson,
                ..Default::default()
            })
            .unwrap();
        let mut deltas = parser.push(&input).unwrap();
        deltas.extend(parser.finish().unwrap());
        assert_eq!(
            assemble(&deltas),
            vec![call("fn", json!({"o": {"s": "a</rfl:parameter>b"}}))]
        );
    }

    // ---- SchemaConstrained: the native serving grammar's mixed raw/JSON slots ----

    fn schema_constrained() -> ToolCallPolicy {
        ToolCallPolicy {
            argument_framing: ArgumentFraming::SchemaConstrained,
            ..Default::default()
        }
    }

    fn declare(name: &str, props: serde_json::Value, strict: Option<bool>) -> Tool {
        let required: Vec<String> = props
            .as_object()
            .map(|p| p.keys().cloned().collect())
            .unwrap_or_default();
        Tool {
            name: name.into(),
            description: None,
            parameters: json!({
                "type": "object",
                "properties": props,
                "required": required,
                "additionalProperties": false,
            }),
            strict,
        }
    }

    /// The `echo` tool: one `value` parameter with `schema`.
    fn echo(schema: serde_json::Value, strict: Option<bool>) -> Tool {
        declare("echo", json!({"value": schema}), strict)
    }

    fn tool_hdr(name: &str) -> String {
        format!("<|start|>assistant to={name}<|channel|>analysis <|constrain|>xml<|message|>")
    }

    /// The corpus wire form: boundary LFs around the value, as the template renders.
    fn echo_call(name: &str, params: &[(&str, &str)]) -> String {
        let body: String = params
            .iter()
            .map(|(key, value)| {
                format!("<rfl:parameter name=\"{key}\">\n{value}\n</rfl:parameter>")
            })
            .collect();
        format!("{}{body}<|call|>", tool_hdr(name))
    }

    const OBJECT_WITH_CLOSER: &str = r#"{"text":"a</rfl:parameter>b"}"#;

    /// The call a raw-text `value` slot yields for [`OBJECT_WITH_CLOSER`]: the embedded
    /// closer ends the value and the leftover `b"}` plus stray closer is junk the
    /// template's tag search skips (the serving reference reads it the same way).
    fn truncated_text_call(name: &str, key: &str) -> Vec<UnifiedEvent> {
        vec![call(name, json!({key: "{\"text\":\"a"}))]
    }

    fn no_calls(results: Vec<anyhow::Result<Vec<UnifiedEvent>>>, input: &str) {
        for got in results {
            assert!(
                !got.unwrap()
                    .iter()
                    .any(|e| matches!(e, UnifiedEvent::ToolCall { .. })),
                "{input}"
            );
        }
    }

    /// The seven differential cases against the original serving parser, at every
    /// chunking: raw string slots keep quotes, unmatched quotes and bare braces
    /// verbatim; the object slot owns the closer inside its JSON string.
    #[test]
    fn schema_constrained_corpus_matches_serving_at_every_split() {
        let string = json!({"type": "string"});
        let object = json!({"type": "object"});
        let corpus = [
            (&string, "London", json!("London")),
            (&string, "\"unfinished", json!("\"unfinished")),
            (&string, "{\"x\":\"unfinished", json!("{\"x\":\"unfinished")),
            (&string, "{", json!("{")),
            (
                &object,
                OBJECT_WITH_CLOSER,
                json!({"text": "a</rfl:parameter>b"}),
            ),
            (&string, "\"Paris\"", json!("\"Paris\"")),
            (&string, "not json", json!("not json")),
        ];
        for (schema, value, want) in corpus {
            let tools = [echo(schema.clone(), Some(true))];
            expect_with(
                &tools,
                &echo_call("echo", &[("value", value)]),
                schema_constrained(),
                vec![call("echo", json!({"value": want}))],
            );
        }
        // Why the global modes are wrong for this grammar, unchanged: `ConstrainedJson`
        // waits forever on a raw string that opens with a quote or brace, and `Lenient`
        // closes the typed object at the embedded closer (the leftover is skipped junk).
        for value in ["\"unfinished", "{\"x\":\"unfinished", "{"] {
            let tools = [echo(string.clone(), Some(true))];
            let input = echo_call("echo", &[("value", value)]);
            no_calls(splits_with(&tools, &input, constrained()), &input);
        }
        let tools = [echo(object.clone(), Some(true))];
        let input = echo_call("echo", &[("value", OBJECT_WITH_CLOSER)]);
        expect_with(
            &tools,
            &input,
            ToolCallPolicy::default(),
            truncated_text_call("echo", "value"),
        );
        // Under `ConstrainedJson` the object case still decodes (that mode is unchanged).
        expect_with(
            &tools,
            &input,
            constrained(),
            vec![call(
                "echo",
                json!({"value": {"text": "a</rfl:parameter>b"}}),
            )],
        );
    }

    #[test]
    fn schema_constrained_mixes_raw_and_json_slots_in_either_order() {
        let tools = [declare(
            "mixed",
            json!({
                "s": {"type": "string"},
                "o": {"type": "object"},
                "n": {"type": "number"},
            }),
            Some(true),
        )];
        let raw = "{\"q\": \"x";
        let want = vec![call(
            "mixed",
            json!({"s": raw, "o": {"text": "a</rfl:parameter>b"}, "n": 1.5}),
        )];
        for order in [
            [("s", raw), ("o", OBJECT_WITH_CLOSER), ("n", "1.5")],
            [("n", "1.5"), ("o", OBJECT_WITH_CLOSER), ("s", raw)],
        ] {
            expect_with(
                &tools,
                &echo_call("mixed", &order),
                schema_constrained(),
                want.clone(),
            );
        }
    }

    /// Key ownership is per recipient: the strict tool's `value` owns JSON, the
    /// unknown tools' `value` slots are raw text (so their embedded closer ends the
    /// value and the leftover is skipped junk), a held tool's are JSON whatever its
    /// `strict` flag, and a later call does not lose, nor an earlier one leak, its key
    /// set.
    #[test]
    fn schema_constrained_key_ownership_does_not_leak_between_recipients() {
        let object = json!({"type": "object"});
        let tools = [
            echo(object.clone(), Some(true)),
            declare("plain", json!({"value": object.clone()}), Some(false)),
        ];
        let input = [
            echo_call("plain", &[("value", OBJECT_WITH_CLOSER)]),
            echo_call("echo", &[("value", OBJECT_WITH_CLOSER)]),
            echo_call("plain", &[("value", r#"{"t": 1}"#)]),
            echo_call("mystery", &[("value", OBJECT_WITH_CLOSER)]),
            echo_call("echo", &[("value", OBJECT_WITH_CLOSER)]),
            echo_call("plain", &[("value", OBJECT_WITH_CLOSER)]),
        ]
        .concat();
        let object_call = call("echo", json!({"value": {"text": "a</rfl:parameter>b"}}));
        let plain_object = || call("plain", json!({"value": {"text": "a</rfl:parameter>b"}}));
        expect_with(
            &tools,
            &input,
            schema_constrained(),
            vec![
                plain_object(),
                object_call.clone(),
                call("plain", json!({"value": {"t": 1}})),
                truncated_text_call("mystery", "value").remove(0),
                object_call,
                plain_object(),
            ],
        );
    }

    /// Every tool the serving grammar holds frames its JSON slots, `strict` or not;
    /// unknown tools, unknown keys and unions admitting `string` are raw text. A local
    /// `$ref` is normalized before the decision.
    #[test]
    fn schema_constrained_frames_every_held_tool_known_keys_and_normalized_types() {
        let object = json!({"type": "object"});
        let input = echo_call("echo", &[("value", OBJECT_WITH_CLOSER)]);
        let object_call = vec![call(
            "echo",
            json!({"value": {"text": "a</rfl:parameter>b"}}),
        )];
        expect_with(
            &[echo(object.clone(), Some(true))],
            &input,
            schema_constrained(),
            object_call.clone(),
        );
        let raw_text = truncated_text_call("echo", "value");
        for strict in [Some(false), None] {
            let tools = [echo(object.clone(), strict)];
            expect_with(&tools, &input, schema_constrained(), object_call.clone());
        }
        // Unknown tool, unknown key, union containing string.
        expect_with(&[], &input, schema_constrained(), raw_text.clone());
        let tools = [echo(object.clone(), Some(true))];
        let other = echo_call("echo", &[("other", OBJECT_WITH_CLOSER)]);
        expect_with(
            &tools,
            &other,
            schema_constrained(),
            truncated_text_call("echo", "other"),
        );
        let union = json!({"anyOf": [{"type": "string"}, {"type": "object"}]});
        let tools = [echo(union, Some(true))];
        expect_with(&tools, &input, schema_constrained(), raw_text.clone());
        // A normalized local ref to an object schema owns JSON.
        let tools = [Tool {
            name: "echo".into(),
            description: None,
            parameters: json!({
                "type": "object",
                "properties": {"value": {"$ref": "#/$defs/Payload"}},
                "required": ["value"],
                "additionalProperties": false,
                "$defs": {"Payload": {"type": "object"}},
            }),
            strict: Some(true),
        }];
        expect_with(&tools, &input, schema_constrained(), object_call);
    }

    /// Framing and typing both follow the last held declaration of a name, whatever
    /// its `strict` flag.
    #[test]
    fn schema_constrained_duplicate_tool_names_follow_the_last_declaration() {
        let tools = [
            echo(json!({"type": "object"}), Some(true)),
            echo(json!({"type": "string"}), Some(false)),
        ];
        let input = echo_call("echo", &[("value", OBJECT_WITH_CLOSER)]);
        // The last declaration types `value` as a string, so the grammar framed it as
        // raw text: the embedded closer ends the value.
        expect_with(
            &tools,
            &input,
            schema_constrained(),
            truncated_text_call("echo", "value"),
        );
        // Reversed: the last declaration is the object, framed and typed as JSON.
        let tools = [
            echo(json!({"type": "string"}), Some(false)),
            echo(json!({"type": "object"}), Some(true)),
        ];
        expect_with(
            &tools,
            &input,
            schema_constrained(),
            vec![call(
                "echo",
                json!({"value": {"text": "a</rfl:parameter>b"}}),
            )],
        );
    }

    /// An open JSON object in a JSON slot is an unclosed parameter: never closed for
    /// the model, dropped or rejected by the request policy. JSON that can never
    /// balance keeps the first-closer rule and types as text.
    #[test]
    fn schema_constrained_never_synthesizes_json_closure() {
        let tools = [echo(json!({"type": "object"}), Some(true))];
        let open = format!(
            "{}<rfl:parameter name=\"value\">{{\"text\":\"</rfl:parameter><|call|>",
            tool_hdr("echo")
        );
        for got in splits_with(&tools, &open, schema_constrained()) {
            assert_eq!(got.unwrap(), vec![], "{open}");
        }
        let reject = ToolCallPolicy {
            malformed: MalformedToolCallPolicy::Reject,
            argument_framing: ArgumentFraming::SchemaConstrained,
        };
        for got in splits_with(&tools, &open, reject) {
            let error = got.expect_err("an open JSON value must be rejected");
            assert!(
                error.downcast_ref::<MalformedToolCall>().is_some(),
                "{error}"
            );
        }
        expect_with(
            &tools,
            &echo_call("echo", &[("value", "{]")]),
            schema_constrained(),
            vec![call("echo", json!({"value": "{]"}))],
        );
    }

    /// A complete strict call directly after an UNMATCHED prose quote is walked by the
    /// prose lexer with the recipient's own framing, so its JSON slot's embedded closer
    /// does not make it "malformed" and the `name="` quote of its markup never closes
    /// the prose quote: the quote stays prose, the call is emitted, and what follows
    /// is routed normally. `ConstrainedJson` and `SchemaConstrained` agree.
    #[test]
    fn prose_quote_before_a_strict_object_call_keeps_the_recipients_framing() {
        let tools = [echo(json!({"type": "object"}), Some(true))];
        let object_call = call("echo", json!({"value": {"text": "a</rfl:parameter>b"}}));
        let call_text = echo_call("echo", &[("value", OBJECT_WITH_CLOSER)]);
        let tail = "<|start|>assistant<|channel|>final<|message|>done<|return|>";
        for policy in [schema_constrained(), constrained()] {
            // Inside a thought, the call immediately after the quote.
            expect_with(
                &tools,
                &format!("<|channel|>analysis<|message|>I will call \"{call_text}{tail}"),
                policy,
                vec![
                    reasoning("I will call \""),
                    object_call.clone(),
                    text("done"),
                ],
            );
            // Inside visible content.
            expect_with(
                &tools,
                &format!("<|channel|>final<|message|>I will call \"{call_text}"),
                policy,
                vec![text("I will call \""), object_call.clone()],
            );
            // The quote left unmatched in a thought that closes before the call.
            expect_with(
                &tools,
                &format!("<|channel|>analysis<|message|>say \"hi<|end|>{call_text}"),
                policy,
                vec![reasoning("say \"hi"), object_call.clone()],
            );
            // Two JSON slots and a raw slot in the quoted call.
            let mixed_tools = [declare(
                "mixed",
                json!({"s": {"type": "string"}, "o": {"type": "object"}, "n": {"type": "number"}}),
                Some(true),
            )];
            let mixed = echo_call(
                "mixed",
                &[("o", OBJECT_WITH_CLOSER), ("s", "say \"x"), ("n", "2")],
            );
            expect_with(
                &mixed_tools,
                &format!("<|channel|>analysis<|message|>I'd say \"{mixed}"),
                policy,
                vec![
                    reasoning("I'd say \""),
                    call(
                        "mixed",
                        json!({"o": {"text": "a</rfl:parameter>b"}, "s": "say \"x", "n": 2}),
                    ),
                ],
            );
        }
        // A quotation is no protection (`quote_aware: false`): a call quoted inside a
        // thought is still a call, as the serving reference reads it, and the next real
        // call is emitted too. The quote's tail (`" yet.`) is debris between messages.
        expect_with(
            &tools,
            &format!("<|channel|>analysis<|message|>Not \"{call_text}\" yet.<|end|>{call_text}"),
            schema_constrained(),
            vec![
                reasoning("Not \""),
                object_call.clone(),
                object_call.clone(),
            ],
        );
        // A raw-slot call (the default rule) was never affected; the per-recipient
        // framing reproduces it.
        let string_tools = [echo(json!({"type": "string"}), Some(true))];
        let raw = echo_call("echo", &[("value", "{\"x\":\"unfinished")]);
        expect_with(
            &string_tools,
            &format!("<|channel|>analysis<|message|>I will call \"{raw}"),
            schema_constrained(),
            vec![
                reasoning("I will call \""),
                call("echo", json!({"value": "{\"x\":\"unfinished"})),
            ],
        );
    }

    /// Framing follows the grammar's union type list (`JsonKeys`), so a strict slot whose
    /// sibling `type` conflicts with a `string` union branch is raw text on the wire and
    /// an unfinished `{` or quote in it is accepted at every split, while typing still
    /// follows the conjunctive rule.
    #[test]
    fn schema_constrained_frames_union_slots_by_the_grammar_rule() {
        let raw_values = ["{", "\"unfinished", "{\"x\":\"unfinished"];
        for schema in [
            json!({"type": "object", "anyOf": [{"type": "object"}, {"type": "string"}]}),
            json!({"type": "object", "oneOf": [{"type": "object"}, {"type": "string"}]}),
            json!({"type": "integer", "anyOf": [{"type": "string"}]}),
            json!({"$ref": "#/$defs/U"}),
        ] {
            let mut tool = echo(schema, Some(true));
            tool.parameters["$defs"] = json!({
                "U": {"type": "object", "anyOf": [{"type": "string"}, {"type": "object"}]}
            });
            for raw in raw_values {
                expect_with(
                    std::slice::from_ref(&tool),
                    &echo_call("echo", &[("value", raw)]),
                    schema_constrained(),
                    vec![call("echo", json!({"value": raw}))],
                );
            }
            // The grammar emitted the slot raw, so an embedded closer ends it.
            let input = echo_call("echo", &[("value", OBJECT_WITH_CLOSER)]);
            expect_with(
                std::slice::from_ref(&tool),
                &input,
                schema_constrained(),
                truncated_text_call("echo", "value"),
            );
        }
        // An untyped union branch contributes no type: the slot stays JSON, and so does
        // a nullable object (and a nullable `$ref` alias to one).
        for schema in [
            json!({"type": "object", "anyOf": [{"description": "x"}, {"type": "object"}]}),
            json!({"type": "object", "nullable": true}),
            json!({"$ref": "#/$defs/Obj", "nullable": true}),
        ] {
            let mut tool = echo(schema, Some(true));
            tool.parameters["$defs"] = json!({"Obj": {"type": "object"}});
            expect_with(
                std::slice::from_ref(&tool),
                &echo_call("echo", &[("value", OBJECT_WITH_CLOSER)]),
                schema_constrained(),
                vec![call(
                    "echo",
                    json!({"value": {"text": "a</rfl:parameter>b"}}),
                )],
            );
        }
    }

    /// The emitter hook hands the scanner the SAME `BeamBoundary` walk with the
    /// recipient's key set; `reset` clears the cursor but keeps that metadata; the
    /// factory boundary and the global modes have no key set.
    #[test]
    fn channel_boundary_carries_recipient_json_keys() {
        let mut emitter = BeamEmitter::new(&[
            echo(json!({"type": "object"}), Some(true)),
            declare("plain", json!({"value": {"type": "object"}}), Some(false)),
        ]);
        emitter.configure(schema_constrained());
        let body =
            format!("<rfl:parameter name=\"value\">{OBJECT_WITH_CLOSER}</rfl:parameter><|call|>");
        let complete = |boundary: &mut Box<dyn InvokeBoundary>| {
            matches!(
                boundary.channel_body(&body, true),
                BodyScan::Complete { .. }
            )
        };
        let mut strict = emitter
            .channel_boundary("echo")
            .expect("beam always frames");
        assert!(complete(&mut strict));
        strict.reset();
        assert!(complete(&mut strict), "reset keeps the key set");
        // A raw-text boundary also completes the call: the embedded closer ends the
        // value and the leftover is junk the tag search skips. The two boundaries
        // differ in the value they type, not in where the call ends.
        for recipient in ["plain", "mystery"] {
            let mut raw = emitter.channel_boundary(recipient).unwrap();
            assert!(complete(&mut raw), "{recipient}");
        }
        // Likewise for a boundary without the key set (global `Lenient` mode, or the
        // factory-built one): the call completes, with the value read as raw text.
        emitter.configure(ToolCallPolicy::default());
        let mut lenient = emitter.channel_boundary("echo").unwrap();
        assert!(complete(&mut lenient));
        let mut factory = BOUNDARY.create();
        factory.configure(schema_constrained());
        assert!(complete(&mut factory));
    }

    // ---- framing-debris rules (reference-golden differential, classes D/G) ----

    /// `<|channel|>analysis<|end|>`: a channel header that never reaches `<|message|>`
    /// is framing debris; its label must not surface as text (goal 3/4).
    #[test]
    fn channel_label_without_message_is_framing_not_text() {
        expect(
            "<|channel|>analysis<|end|><|channel|>final<|message|>answer<|return|>",
            vec![text("answer")],
        );
    }

    /// The grammar spells the lead `assistant to=NAME`; a lead with debris glued to the
    /// recipient (`assistant\tto=NAME`, `assistantto=NAME`) is still identified by its
    /// `to=NAME` tail, as the builder's template regex does. The debris is header bytes,
    /// never text.
    #[test]
    fn lead_debris_in_front_of_the_recipient_is_header_not_text() {
        for lead in [
            "assistant\tto=weather",
            "assistantto=weather",
            "assistant\nto=weather",
        ] {
            expect(
                &format!(
                    "<|channel|>analysis<|message|>think<|end|><|start|>{lead}<|channel|>analysis <|constrain|>xml<|message|><rfl:parameter name=\"location\">\nParis\n</rfl:parameter><|call|>"
                ),
                vec![
                    reasoning("think"),
                    call("weather", json!({"location": "Paris"})),
                ],
            );
        }
    }

    /// `<|start|>` + a lead that names no recipient and is not the role + a channel header
    /// is a garbled message: header AND body are skipped, through the body's terminator.
    #[test]
    fn garbled_start_header_is_skipped_whole_with_its_body() {
        expect(
            "<|channel|>analysis<|message|>think<|end|><|start|>assistxnt<|channel|>final<|message|>hidden<|return|><|start|>assistant<|channel|>final<|message|>ok<|return|>",
            vec![reasoning("think"), text("ok")],
        );
    }

    /// A channel the grammar does not define is a whole message that is not visible text:
    /// header and body are skipped together, through the nearest terminator.
    #[test]
    fn unknown_channel_body_is_skipped_with_its_header() {
        expect(
            "<|channel|>commentary<|message|>hidden<|end|><|start|>assistant<|channel|>final<|message|>ok<|return|>",
            vec![text("ok")],
        );
    }

    /// A recipient on a message channel is header debris: the channel opens as written.
    #[test]
    fn recipient_on_a_message_channel_opens_the_channel() {
        expect(
            "<|start|>assistant to=weather<|channel|>final<|message|>Paris is sunny.<|return|>",
            vec![text("Paris is sunny.")],
        );
    }

    /// A `final` body that opens like JSON (a `response_format` answer) keeps a marker
    /// inside an open JSON string as data; the body still ends at the marker after the
    /// JSON. Prose that merely quotes a marker is not JSON and ends at the marker.
    #[test]
    fn json_answer_keeps_a_marker_inside_its_string_but_prose_does_not() {
        expect(
            "<|channel|>final<|message|>{\"value\":\"<|return|>\"}<|return|>",
            vec![text("{\"value\":\"<|return|>\"}")],
        );
        expect(
            "<|channel|>final<|message|>\n[{\"k\": \"a <|end|> b\", \"n\": 1}]<|return|>",
            vec![text("\n[{\"k\": \"a <|end|> b\", \"n\": 1}]")],
        );
        expect(
            "<|channel|>final<|message|>Note: \"<|return|>\" ends it.<|return|>",
            vec![text("Note: \"")],
        );
        // Not a `final` body: a thought that opens like JSON still ends at `<|end|>`.
        expect(
            "<|channel|>analysis<|message|>{\"a\":\"<|end|>\"}<|end|><|start|>assistant<|channel|>final<|message|>ok<|return|>",
            vec![reasoning("{\"a\":\""), text("ok")],
        );
    }

    /// A stray `<|constrain|>` or `<|message|>` between messages is stripped like the
    /// other orphan controls, and never opens anything: the next header still parses.
    #[test]
    fn stray_constrain_and_message_between_messages_are_stripped() {
        expect(
            "<|channel|>analysis<|message|>think<|end|><|constrain|><|message|><|start|>assistant<|channel|>final<|message|>ok<|return|>",
            vec![reasoning("think"), text("ok")],
        );
    }

    /// A start marker whose lead is broken by another control (`<|start|>ysis<|message|>…
    /// <|start|>assistant<|channel|>final…`) is debris only up to that control: the real
    /// header after it still opens its channel.
    #[test]
    fn broken_start_lead_does_not_swallow_the_next_header() {
        expect(
            "<|channel|>anal<|start|>ysis<|message|>Paris><|end|><|start|>assistant<|channel|>final<|message|> to=xml<|return|>",
            vec![text(" to=xml")],
        );
    }

    /// The template emits exactly `<|constrain|>xml`; any other token is not a tool header
    /// (goal 1), so the call is dropped whole with its body instead of being parsed.
    #[test]
    fn constrain_token_other_than_xml_is_not_a_tool_header() {
        expect(
            "<|start|>assistant to=weather<|channel|>analysis <|constrain|>json<|message|><rfl:parameter name=\"location\">\nParis\n</rfl:parameter><|end|>",
            vec![],
        );
    }
}
