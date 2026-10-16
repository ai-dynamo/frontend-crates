// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Beam reasoning parser (NOT GPT-OSS Harmony).
//!
//! ```text
//! reasoning:  <|start|>assistant<|channel|>analysis<|message|> … <|end|>
//! content:    <|start|>assistant<|channel|>final<|message|> … <|return|>
//! tool call:  <|start|>assistant to=NAME<|channel|>analysis <|constrain|>xml<|message|>
//!             <rfl:parameter name="KEY">VALUE</rfl:parameter>… <|call|>   (or <|end|>)
//! ```
//!
//! V1 splits reasoning from tools: this parser strips the analysis/final framing and
//! returns reasoning text and visible text, and it passes every tool envelope (header,
//! body and terminator) through VERBATIM in `normal_text` so a downstream tool parser
//! sees the exact bytes. It cannot emit calls itself (`ParserResult` has no such
//! field), so a call is never dropped here: anything it cannot classify, including a
//! truncated or malformed tool body, is forwarded as normal text for the tool parser to
//! accept or reject. Headers may be full, continuation (`<|channel|>…`) or begin with the
//! `assistant` prefix the prompt already consumed (` to=NAME<|channel|>…`); the last two
//! forms are forwarded exactly as received.
//!
//! The `ReasoningParser` API is infallible, so this parser can never report an error;
//! malformed tool bodies surface only when the tool parser runs over the forwarded text.
//!
//! Inside a tool body only `<rfl:parameter …>…</rfl:parameter>` regions are walked and
//! everything inside one is data, so framing strings in an argument are not boundaries.
//!
//! Bytes outside every channel (before the first header, between messages) are framing
//! debris and are dropped, as the builder's template parser drops them; a message on a
//! channel the grammar does not define is skipped whole. A quotation never keeps a
//! control marker literal: the markers are special tokens, and an unbalanced quote in a
//! thought must not swallow the reply nor hold the stream back. These are the same rules
//! the `dynamo-parsers-v2` `beam` family applies, so both routes read a stream alike.

use crate::{ParserResult, ReasoningParser};

const START: &str = "<|start|>";
const CHANNEL: &str = "<|channel|>";
const MESSAGE: &str = "<|message|>";
const CONSTRAIN: &str = "<|constrain|>";
const END: &str = "<|end|>";
const RETURN: &str = "<|return|>";
const CALL: &str = "<|call|>";
const PARAM_OPEN: &str = "<rfl:parameter name=\"";
const PARAM_CLOSE: &str = "</rfl:parameter>";
/// Only `<|end|>` closes a thought; `<|start|>` ends one the model forgot to close.
const ANALYSIS_CUTS: [&str; 2] = [END, START];
/// A reply closes on `<|return|>` or `<|end|>`; `<|start|>` ends one left open.
const CONTENT_CUTS: [&str; 3] = [END, RETURN, START];
/// Terminators of every channel, for skipping a message the grammar does not define.
const ANY_TERMINATOR: [&str; 3] = [END, RETURN, CALL];
/// Terminators of a tool body.
const TOOL_TERMINATORS: [&str; 2] = [CALL, END];
/// Control markers that end a garbled `<|start|>` lead early.
const LEAD_CUTS: [&str; 6] = [START, MESSAGE, CONSTRAIN, END, RETURN, CALL];
/// Control markers that are debris between messages.
const ORPHAN_CONTROLS: [&str; 5] = [END, RETURN, CALL, CONSTRAIN, MESSAGE];
const MAX_CHANNEL_HEADER: usize = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Idle,
    Reasoning,
    Content,
    /// Offset up to which parameter regions are complete.
    Tool(usize),
}

/// What a header's lead says about the message.
enum Lead {
    /// `assistant` (or nothing): a message header.
    Role,
    /// `… to=NAME`: a tool header addressed to `NAME`. Debris glued in front of the
    /// recipient (`assistant\tto=NAME`, `assistantto=NAME`) is header bytes, not text.
    Recipient(String),
}

/// The channel named by a header label.
enum Channel {
    Analysis,
    Final,
    /// A label the grammar does not define (`commentary`, `reasoning`, …).
    Unknown,
}

enum Scan {
    Wait,
    /// Bytes outside every channel: framing debris, dropped (visible only under an
    /// explicit response prefill, where the whole stream IS the reply).
    Outside(usize),
    Skip(usize),
    /// A header and its body that the grammar cannot place: skipped together, through
    /// the nearest terminator in `terminators` (waiting for it while the stream is open).
    SkipMessage {
        header_end: usize,
        terminators: &'static [&'static str],
    },
    Header {
        consume: usize,
        kind: HeaderKind,
    },
}

enum HeaderKind {
    Reasoning,
    Content,
    /// A tool header: `lead_end` is where the raw lead stops (the `<|channel|>` marker),
    /// `name` the recipient. The lead is forwarded in canonical form.
    Tool {
        name: String,
        lead_end: usize,
    },
}

#[derive(Debug)]
pub struct BeamReasoningParser {
    buf: String,
    mode: Mode,
    /// JSON string context of an open reply body.
    json: JsonBodyState,
    header_slot: bool,
    prefill: Mode,
    reasoning: String,
    normal: String,
}

fn recipient_len(text: &str) -> usize {
    let mut end = 0;
    for (i, c) in text.char_indices() {
        let ok = if i == 0 {
            c.is_ascii_alphabetic() || c == '_'
        } else {
            c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-')
        };
        if !ok {
            break;
        }
        end = i + c.len_utf8();
    }
    end
}

fn trailing_partial(buf: &str, markers: &[&str]) -> usize {
    let max = markers.iter().map(|m| m.len()).max().unwrap_or(1) - 1;
    (1..=max.min(buf.len()))
        .rev()
        .find(|&k| {
            buf.is_char_boundary(buf.len() - k)
                && markers
                    .iter()
                    .any(|m| m.len() > k && m.starts_with(&buf[buf.len() - k..]))
        })
        .unwrap_or(0)
}

/// The lead before a `<|channel|>` marker. The recipient is whatever `to=NAME` ends the
/// lead (the builder's template identifies a tool header by that tail); `None` when the
/// lead names no recipient and is not the bare role.
fn parse_prefix(prefix: &str) -> Option<Lead> {
    let t = prefix.strip_prefix("assistant").unwrap_or(prefix);
    let t = t.trim_start_matches(' ');
    if t.is_empty() {
        return Some(Lead::Role);
    }
    let k = t.rfind("to=")?;
    let r = &t[k + 3..];
    (!r.is_empty() && recipient_len(r) == r.len()).then(|| Lead::Recipient(r.to_string()))
}

fn header_prefix_possible(text: &str) -> bool {
    let t = match text.strip_prefix("assistant") {
        Some(t) => t,
        None if "assistant".starts_with(text) => return true,
        None => text,
    };
    let t = t.trim_start_matches(' ');
    if t.is_empty() || "to=".starts_with(t) {
        return true;
    }
    t.strip_prefix("to=")
        .is_some_and(|r| recipient_len(r) == r.len())
}

/// The channel a label names, and whether the label carries `<|constrain|>`. Only the
/// template's own ` <|constrain|>xml` is a well-formed constraint; any other constrain
/// token makes the header malformed (`None`).
fn parse_channel(text: &str) -> Option<(Channel, bool)> {
    let (name, rest) = match text.split_once(' ') {
        Some((name, rest)) => (name, Some(rest)),
        None => (text, None),
    };
    let constrained = match rest {
        None => false,
        Some(rest) => {
            if rest.strip_prefix(CONSTRAIN) != Some("xml") {
                return None;
            }
            true
        }
    };
    let channel = match name {
        "analysis" => Channel::Analysis,
        "final" => Channel::Final,
        _ => Channel::Unknown,
    };
    Some((channel, constrained))
}

/// String-context tracker for a reply read as a JSON answer; it survives the body's
/// streamed emissions so a marker inside an open string stays data.
#[derive(Debug, Clone, Copy, Default)]
struct JsonBodyState {
    /// `Some(true)` once the first non-blank byte is `{` or `[`, `Some(false)` once it
    /// is anything else, `None` while the body is still blank.
    shape: Option<bool>,
    in_string: bool,
    escaped: bool,
}

impl JsonBodyState {
    fn advance(&mut self, text: &str) {
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

    fn inside_string_at(mut self, buffer: &str, at: usize) -> bool {
        self.advance(&buffer[..at]);
        self.shape == Some(true) && self.in_string
    }
}

/// The end of the nearest terminator in `terminators` at or after `from`.
fn next_terminator(buf: &str, from: usize, terminators: &[&str]) -> Option<usize> {
    terminators
        .iter()
        .filter_map(|m| buf[from..].find(m).map(|p| from + p + m.len()))
        .min()
}

impl Default for BeamReasoningParser {
    fn default() -> Self {
        Self::new()
    }
}

impl BeamReasoningParser {
    pub fn new() -> Self {
        Self {
            buf: String::new(),
            mode: Mode::Idle,
            json: JsonBodyState::default(),
            header_slot: true,
            prefill: Mode::Idle,
            reasoning: String::new(),
            normal: String::new(),
        }
    }

    fn restart(&mut self) {
        self.buf.clear();
        self.json = JsonBodyState::default();
        self.reasoning.clear();
        self.normal.clear();
        self.header_slot = true;
        self.mode = self.prefill;
    }

    fn scan_idle(&self, flush: bool) -> Scan {
        let buf = self.buf.as_str();
        if buf.is_empty() {
            return Scan::Wait;
        }
        let s = buf.find(START);
        let c = buf.find(CHANNEL);
        // A control marker before any header is debris between messages: the bytes
        // before it go first, then the marker itself, so a lead after it is judged
        // on its own (`<|end|><|constrain|>assistant to=fs.read<|channel|>…`).
        if let Some((p, len)) = ORPHAN_CONTROLS
            .iter()
            .filter_map(|m| buf.find(m).map(|p| (p, m.len())))
            .min_by_key(|(p, _)| *p)
            && s.is_none_or(|s| p < s)
            && c.is_none_or(|c| p < c)
        {
            return if p > 0 {
                Scan::Outside(p)
            } else {
                Scan::Skip(len)
            };
        }
        let (channel_pos, lead) = match (s, c) {
            (None, None) => {
                let hold = trailing_partial(buf, &LEAD_CUTS);
                let lead = &buf[..buf.len() - hold];
                // At the stream start the lead is judged whole within the header budget.
                if self.header_slot
                    && !flush
                    && (lead.len() <= MAX_CHANNEL_HEADER || header_prefix_possible(lead))
                {
                    return Scan::Wait;
                }
                let keep = if flush { 0 } else { hold };
                return match buf.len() - keep {
                    0 => Scan::Wait,
                    n => Scan::Outside(n),
                };
            }
            (Some(s), c) if c.is_none_or(|c| s < c) => {
                if s > 0 {
                    return Scan::Outside(s);
                }
                let rest = &buf[START.len()..];
                // A control inside the lead ends the garbled region there: the bytes up
                // to it are debris, the marker itself is resolved on the next step.
                let lead_cut = LEAD_CUTS.iter().filter_map(|m| rest.find(m)).min();
                match rest.find(CHANNEL) {
                    found if lead_cut.is_some_and(|q| found.is_none_or(|ci| q < ci)) => {
                        return Scan::Skip(START.len() + lead_cut.expect("checked"));
                    }
                    None if !flush
                        && (rest.len() <= MAX_CHANNEL_HEADER
                            || header_prefix_possible(
                                &rest[..rest.len() - trailing_partial(rest, &[CHANNEL])],
                            )) =>
                    {
                        return Scan::Wait;
                    }
                    None => return Scan::Skip(buf.len()),
                    Some(ci) => match parse_prefix(&rest[..ci]) {
                        Some(lead) => (START.len() + ci, lead),
                        // A garbled header is skipped whole with its body; a `to=` in
                        // the lead or `<|constrain|>` in the label makes it tool-framed.
                        None => {
                            let after = START.len() + ci + CHANNEL.len();
                            return self.skip_header(after, flush, |label| {
                                rest[..ci].contains("to=") || label.contains(CONSTRAIN)
                            });
                        }
                    },
                }
            }
            (_, Some(c)) => match parse_prefix(&buf[..c]) {
                Some(lead) => (c, lead),
                None => return Scan::Outside(c),
            },
            _ => unreachable!("one marker is present"),
        };
        let after = channel_pos + CHANNEL.len();
        // A control marker inside the label (`<|channel|>analysis<|end|>`): the header
        // never had a `<|message|>`; the label is debris up to that marker.
        let label_region = &buf[after..];
        if let Some(cut) = [START, END, RETURN, CALL]
            .iter()
            .filter_map(|m| label_region.find(m))
            .min()
            && label_region[..cut].find(MESSAGE).is_none()
        {
            return Scan::Skip(after + cut);
        }
        let Some(mi) = buf[after..].find(MESSAGE) else {
            if flush {
                return Scan::Skip(buf.len());
            }
            if buf.len() - after > MAX_CHANNEL_HEADER {
                return Scan::Skip(after);
            }
            return Scan::Wait;
        };
        let header_end = after + mi + MESSAGE.len();
        let label = &buf[after..after + mi];
        let tool_framed = matches!(lead, Lead::Recipient(_)) || label.contains(CONSTRAIN);
        match parse_channel(label) {
            // A recipient on the tool channel: a call. Its lead is forwarded canonically.
            Some((Channel::Analysis, _)) if let Lead::Recipient(name) = lead => Scan::Header {
                consume: header_end,
                kind: HeaderKind::Tool {
                    name,
                    lead_end: channel_pos,
                },
            },
            // A constraint without a recipient, or on a message channel: a tool header
            // missing its addressee or on the wrong channel. Skipped with its body.
            Some((_, true)) => Scan::SkipMessage {
                header_end,
                terminators: &TOOL_TERMINATORS,
            },
            // A bare recipient on `final` is header debris: the reply opens as written.
            Some((Channel::Final, false)) => Scan::Header {
                consume: header_end,
                kind: HeaderKind::Content,
            },
            Some((Channel::Analysis, false)) => Scan::Header {
                consume: header_end,
                kind: HeaderKind::Reasoning,
            },
            // Unknown channel, or a malformed constraint: the whole message goes.
            Some((Channel::Unknown, _)) | None => Scan::SkipMessage {
                header_end,
                terminators: if tool_framed {
                    &TOOL_TERMINATORS
                } else {
                    &ANY_TERMINATOR
                },
            },
        }
    }

    /// A garbled header whose `<|channel|>` ends at `after`: skip it and its body.
    fn skip_header(&self, after: usize, flush: bool, tool_framed: impl Fn(&str) -> bool) -> Scan {
        let buf = self.buf.as_str();
        let Some(mi) = buf[after..].find(MESSAGE) else {
            if flush {
                return Scan::Skip(buf.len());
            }
            if buf.len() - after > MAX_CHANNEL_HEADER {
                return Scan::Skip(after);
            }
            return Scan::Wait;
        };
        let label = &buf[after..after + mi];
        Scan::SkipMessage {
            header_end: after + mi + MESSAGE.len(),
            terminators: if tool_framed(label) {
                &TOOL_TERMINATORS
            } else {
                &ANY_TERMINATOR
            },
        }
    }

    fn step(&mut self, flush: bool) -> bool {
        match self.mode {
            Mode::Idle => match self.scan_idle(flush) {
                Scan::Wait => false,
                Scan::Outside(n) => {
                    // The whole stream is the reply under an explicit response prefill;
                    // stray control markers in it are still debris.
                    if self.prefill == Mode::Content {
                        let mut text = self.buf[..n].to_string();
                        for marker in ORPHAN_CONTROLS {
                            text = text.replace(marker, "");
                        }
                        self.normal.push_str(&text);
                    }
                    self.buf.drain(..n);
                    // Debris sits between messages, so a header may start right after it
                    // and its lead is held for the channel marker; visible prefill prose
                    // is not a boundary.
                    self.header_slot = self.prefill != Mode::Content;
                    true
                }
                Scan::Skip(n) => {
                    self.buf.drain(..n);
                    self.header_slot = true;
                    true
                }
                Scan::SkipMessage {
                    header_end,
                    terminators,
                } => match next_terminator(&self.buf, header_end, terminators) {
                    Some(end) => {
                        self.buf.drain(..end);
                        true
                    }
                    None if flush => {
                        self.buf.clear();
                        true
                    }
                    None => false,
                },
                Scan::Header { consume, kind } => {
                    self.mode = match kind {
                        HeaderKind::Tool { name, lead_end } => {
                            // The envelope is forwarded for the tool parser: the lead in
                            // its canonical spelling, label and body verbatim.
                            self.normal.push_str("<|start|>assistant to=");
                            self.normal.push_str(&name);
                            self.normal.push_str(&self.buf[lead_end..consume]);
                            Mode::Tool(0)
                        }
                        HeaderKind::Reasoning => Mode::Reasoning,
                        HeaderKind::Content => {
                            self.json = JsonBodyState::default();
                            Mode::Content
                        }
                    };
                    self.buf.drain(..consume);
                    true
                }
            },
            Mode::Reasoning | Mode::Content => {
                let cuts: &[&str] = if self.mode == Mode::Reasoning {
                    &ANALYSIS_CUTS
                } else {
                    &CONTENT_CUTS
                };
                // The markers are special tokens: a terminator ends its body wherever it
                // stands, inside a quotation or not. The one exception is a `final` body
                // that opens like JSON (a `response_format` answer): a marker inside one
                // of its open strings is data, as the reference keeps it under constrained
                // decoding.
                let json_answer = self.mode == Mode::Content;
                let mut candidates: Vec<(usize, &str)> = cuts
                    .iter()
                    .flat_map(|m| self.buf.match_indices(m).map(move |(at, _)| (at, *m)))
                    .collect();
                candidates.sort_unstable();
                let found = candidates
                    .into_iter()
                    .find(|(at, _)| !(json_answer && self.json.inside_string_at(&self.buf, *at)));
                let (emit, cut) = match found {
                    Some((at, marker)) => (at, Some(marker)),
                    None => {
                        // At EOF a partial marker is literal text, never dropped.
                        let hold = if flush {
                            0
                        } else {
                            trailing_partial(&self.buf, cuts)
                        };
                        (self.buf.len() - hold, None)
                    }
                };
                let mut consumed = emit;
                if cut.is_none() && flush {
                    consumed = self.buf.len();
                }
                self.emit_body(consumed);
                if let Some(marker) = cut {
                    if marker != START {
                        self.buf.drain(..marker.len());
                    }
                    self.mode = Mode::Idle;
                    self.header_slot = true;
                    self.json = JsonBodyState::default();
                }
                consumed > 0 || cut.is_some()
            }
            Mode::Tool(mut scan) => {
                // The body is forwarded verbatim through its terminator. It is read as
                // the template's tag search: complete `<rfl:parameter>` regions are
                // stepped over (a marker inside a value is data), anything else is
                // skipped up to the next region or terminator. A framed next header
                // leaves the call unterminated and ends the forward there; the tool
                // parser decides what the body means.
                let end = loop {
                    let rest = &self.buf[scan..];
                    let t = rest.trim_start_matches([' ', '\t', '\n', '\r']);
                    let at = scan + (rest.len() - t.len());
                    scan = at;
                    if t.is_empty() {
                        break flush.then_some(self.buf.len());
                    }
                    if let Some(term) = TOOL_TERMINATORS.iter().find(|m| t.starts_with(**m)) {
                        break Some(at + term.len());
                    }
                    if t.starts_with(START) {
                        break Some(at);
                    }
                    if let Some(rest) = t.strip_prefix(PARAM_OPEN) {
                        match t.find(PARAM_CLOSE) {
                            Some(i) => {
                                scan = at + i + PARAM_CLOSE.len();
                                continue;
                            }
                            // No close yet: wait for it while the stream is open. At EOF
                            // the forward ends at the terminator or framed header after
                            // the opener (the tool parser rejects the unclosed region),
                            // or takes the truncated body whole.
                            None if !flush => break None,
                            None => {
                                let next = TOOL_TERMINATORS
                                    .iter()
                                    .filter_map(|m| rest.find(m).map(|p| (p, p + m.len())))
                                    .min();
                                let header = rest.find(START);
                                break match (next, header) {
                                    (Some((p, e)), h) if h.is_none_or(|h| p < h) => {
                                        Some(at + PARAM_OPEN.len() + e)
                                    }
                                    (_, Some(h)) => Some(at + PARAM_OPEN.len() + h),
                                    _ => flush.then_some(self.buf.len()),
                                };
                            }
                        }
                    }
                    if !flush
                        && [CALL, END, START, PARAM_OPEN]
                            .iter()
                            .any(|m| m.starts_with(t))
                    {
                        break None;
                    }
                    // Junk: resume at the next region, terminator or framed header.
                    let next_param = t.find(PARAM_OPEN);
                    let next_term = TOOL_TERMINATORS
                        .iter()
                        .filter_map(|m| t.find(m).map(|p| (p, p + m.len())))
                        .min();
                    let next_header = t.find(START);
                    let first = [next_param, next_term.map(|(p, _)| p), next_header]
                        .into_iter()
                        .flatten()
                        .min();
                    match first {
                        Some(p) if next_header == Some(p) => break Some(at + p),
                        Some(p) if next_term.is_some_and(|(tp, _)| tp == p) => {
                            break Some(at + next_term.expect("matched").1);
                        }
                        Some(p) => {
                            scan = at + p;
                            continue;
                        }
                        None => break flush.then_some(self.buf.len()),
                    }
                };
                match end {
                    Some(end) => {
                        self.normal.push_str(&self.buf[..end]);
                        self.buf.drain(..end);
                        self.mode = Mode::Idle;
                        self.header_slot = true;
                        true
                    }
                    None => {
                        self.mode = Mode::Tool(scan);
                        false
                    }
                }
            }
        }
    }

    /// Emit `buf[..end]` to the current channel.
    fn emit_body(&mut self, end: usize) {
        let target = if self.mode == Mode::Reasoning {
            &mut self.reasoning
        } else {
            &mut self.normal
        };
        target.push_str(&self.buf[..end]);
        if self.mode == Mode::Content {
            self.json.advance(&self.buf[..end]);
        }
        self.buf.drain(..end);
    }

    fn drain(&mut self, flush: bool) -> ParserResult {
        while self.step(flush) {}
        ParserResult {
            normal_text: std::mem::take(&mut self.normal),
            reasoning_text: std::mem::take(&mut self.reasoning),
        }
    }
}

impl ReasoningParser for BeamReasoningParser {
    fn detect_and_parse_reasoning(&mut self, text: &str, _token_ids: &[u32]) -> ParserResult {
        self.restart();
        self.buf.push_str(text);
        let result = self.drain(true);
        self.restart();
        result
    }

    fn parse_reasoning_streaming_incremental(
        &mut self,
        text: &str,
        _token_ids: &[u32],
    ) -> ParserResult {
        self.buf.push_str(text);
        self.drain(false)
    }

    fn finish_reasoning_stream(&mut self) -> ParserResult {
        let result = self.drain(true);
        self.buf.clear();
        self.mode = self.prefill;
        self.header_slot = true;
        result
    }

    /// The prompt already opened the visible reply (explicit response prefill): bytes are
    /// final-channel content, so `<|return|>` / `<|end|>` close and are stripped.
    fn set_in_content(&mut self, in_content: bool) {
        self.prefill = if in_content {
            Mode::Content
        } else {
            Mode::Idle
        };
        if self.buf.is_empty() {
            self.mode = self.prefill;
        }
    }

    fn set_in_reasoning(&mut self, in_reasoning: bool) {
        self.prefill = if in_reasoning {
            Mode::Reasoning
        } else {
            Mode::Idle
        };
        // Only before any bytes are held: a mid-stream flip would reroute held text.
        if self.buf.is_empty() {
            self.mode = self.prefill;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOOL: &str = "<|start|>assistant to=weather<|channel|>analysis <|constrain|>xml<|message|><rfl:parameter name=\"location\">\nLondon <|call|> \n</rfl:parameter><|call|>";

    fn stream(chunks: &[&str]) -> (String, String) {
        let mut parser = BeamReasoningParser::new();
        let (mut reasoning, mut normal) = (String::new(), String::new());
        for chunk in chunks {
            let r = parser.parse_reasoning_streaming_incremental(chunk, &[]);
            reasoning += &r.reasoning_text;
            normal += &r.normal_text;
        }
        let r = parser.finish_reasoning_stream();
        (reasoning + &r.reasoning_text, normal + &r.normal_text)
    }

    #[test]
    fn beam_splits_reasoning_and_final() {
        let input = "<|start|>assistant<|channel|>analysis<|message|>Thinking...<|end|><|start|>assistant<|channel|>final<|message|>Hi there!<|return|>";
        let mut parser = BeamReasoningParser::new();
        let r = parser.detect_and_parse_reasoning(input, &[]);
        assert_eq!(
            (r.reasoning_text.as_str(), r.normal_text.as_str()),
            ("Thinking...", "Hi there!")
        );
        for at in (0..=input.len()).filter(|&at| input.is_char_boundary(at)) {
            assert_eq!(
                stream(&[&input[..at], &input[at..]]),
                ("Thinking...".to_string(), "Hi there!".to_string()),
                "split {at}"
            );
        }
    }

    #[test]
    fn beam_tool_envelope_is_verbatim_even_with_framing_literals() {
        let input =
            format!("<|start|>assistant<|channel|>analysis<|message|>Look it up.<|end|>{TOOL}");
        for at in (0..=input.len()).filter(|&at| input.is_char_boundary(at)) {
            assert_eq!(
                stream(&[&input[..at], &input[at..]]),
                ("Look it up.".to_string(), TOOL.to_string()),
                "split {at}"
            );
        }
    }

    /// A continuation or prefix-consumed tool header (` to=a<|channel|>…`), and a lead
    /// with debris glued to the recipient, are forwarded with the lead in its canonical
    /// spelling; label and body stay verbatim.
    #[test]
    fn beam_tool_leads_are_forwarded_canonically() {
        for lead in [
            " to=a",
            "assistant to=a",
            "assistant\tto=a",
            "<|start|>assistantto=a",
        ] {
            let (reasoning, normal) = stream(&[&format!(
                "<|channel|>analysis<|message|>t<|end|>{lead}<|channel|>analysis <|constrain|>xml<|message|><|call|>"
            )]);
            assert_eq!(reasoning, "t", "{lead}");
            assert_eq!(
                normal,
                "<|start|>assistant to=a<|channel|>analysis <|constrain|>xml<|message|><|call|>",
                "{lead}"
            );
        }
    }

    #[test]
    fn beam_truncated_tool_body_is_forwarded_not_dropped() {
        let input = "<|start|>assistant to=weather<|channel|>analysis <|constrain|>xml<|message|><rfl:parameter name=\"location\">\nLondon\n";
        assert_eq!(stream(&[input]), (String::new(), input.to_string()));
    }

    #[test]
    fn beam_channel_specific_terminators() {
        let (reasoning, normal) = stream(&["<|channel|>analysis<|message|>a<|call|>b<|end|>"]);
        assert_eq!((reasoning.as_str(), normal.as_str()), ("a<|call|>b", ""));
        let (reasoning, normal) = stream(&["<|channel|>final<|message|>b<|call|>c<|return|>"]);
        assert_eq!((reasoning.as_str(), normal.as_str()), ("", "b<|call|>c"));
    }

    /// Inside a body a partial marker at EOF is literal text; outside every channel the
    /// bytes are framing debris and are dropped (before the first header, between
    /// messages, after the last one).
    #[test]
    fn beam_partial_marker_is_literal_in_a_body_and_outside_bytes_are_dropped() {
        let input = "<|channel|>final<|message|>tail <|ret";
        for at in (0..=input.len()).filter(|&at| input.is_char_boundary(at)) {
            let (_, normal) = stream(&[&input[..at], &input[at..]]);
            assert_eq!(normal, "tail <|ret");
        }
        for input in [
            "tail <|sta",
            "Sure.<|channel|>analysis<|message|>t<|end|>junk<|start|>assistant<|channel|>final<|message|>ok<|return|>trailer",
        ] {
            for at in (0..=input.len()).filter(|&at| input.is_char_boundary(at)) {
                let (reasoning, normal) = stream(&[&input[..at], &input[at..]]);
                let want = if input.contains("<|channel|>") {
                    ("t", "ok")
                } else {
                    ("", "")
                };
                assert_eq!(
                    (reasoning.as_str(), normal.as_str()),
                    want,
                    "{input} at {at}"
                );
            }
        }
    }

    /// A tool header that never completes, a message on a channel the grammar does not
    /// define, and a constraint without a recipient are skipped with their bodies: the
    /// reply after them is unaffected and nothing of them reaches either output.
    #[test]
    fn beam_bad_headers_are_skipped_with_their_bodies() {
        let reply = "<|start|>assistant<|channel|>final<|message|>ok<|return|>";
        for bad in [
            "to=weather<|channel|>analysis",
            "<|start|>assistant to=weather<|channel|>bogus<|message|>hidden<|end|>",
            "<|channel|>commentary<|message|>hidden<|end|>",
            "<|channel|>analysis <|constrain|>xml<|message|><rfl:parameter name=\"k\">\n1\n</rfl:parameter><|call|>",
            "<|start|>assistant to=weather<|channel|>analysis <|constrain|>json<|message|><|end|>",
        ] {
            let input = format!("{bad}{reply}");
            let want = if bad.ends_with("analysis") {
                // Still a plausible header prefix when the reply's `<|start|>` arrives:
                // the lead is cut there and the reply parses.
                ("", "ok")
            } else {
                ("", "ok")
            };
            for at in (0..=input.len()).filter(|&at| input.is_char_boundary(at)) {
                let (reasoning, normal) = stream(&[&input[..at], &input[at..]]);
                assert_eq!((reasoning.as_str(), normal.as_str()), want, "{bad} at {at}");
            }
        }
    }

    /// A `final` body that opens like JSON keeps a marker inside an open string; prose
    /// that quotes a marker, and a thought, still end at the marker.
    #[test]
    fn beam_json_answer_keeps_a_marker_inside_its_string() {
        for (input, reasoning, normal) in [
            (
                "<|channel|>final<|message|>{\"value\":\"<|return|>\"}<|return|>",
                "",
                "{\"value\":\"<|return|>\"}",
            ),
            (
                "<|channel|>final<|message|>Note: \"<|return|>\" ends it.<|return|>",
                "",
                "Note: \"",
            ),
            (
                "<|channel|>analysis<|message|>{\"a\":\"<|end|>\"}<|end|><|start|>assistant<|channel|>final<|message|>ok<|return|>",
                "{\"a\":\"",
                "ok",
            ),
        ] {
            for (r, n) in split_runs(input) {
                assert_eq!((r.as_str(), n.as_str()), (reasoning, normal), "{input}");
            }
        }
    }

    #[test]
    fn beam_response_prefill_strips_return() {
        let mut parser = BeamReasoningParser::new();
        parser.set_in_content(true);
        let r = parser.parse_reasoning_streaming_incremental("hi<|return|>", &[]);
        assert_eq!(
            (r.reasoning_text.as_str(), r.normal_text.as_str()),
            ("", "hi")
        );
    }

    fn split_runs(input: &str) -> Vec<(String, String)> {
        let mut all = Vec::new();
        for at in (0..=input.len()).filter(|&at| input.is_char_boundary(at)) {
            let mut parser = BeamReasoningParser::new();
            let mut r = parser.parse_reasoning_streaming_incremental(&input[..at], &[]);
            let b = parser.parse_reasoning_streaming_incremental(&input[at..], &[]);
            let c = parser.finish_reasoning_stream();
            for part in [b, c] {
                r.reasoning_text.push_str(&part.reasoning_text);
                r.normal_text.push_str(&part.normal_text);
            }
            all.push((r.reasoning_text, r.normal_text));
        }
        all
    }

    /// The markers are special tokens: a quotation never keeps one literal, so the first
    /// terminator ends its body, and an unbalanced quote in a thought cannot swallow the
    /// reply (what follows a closed reply is debris).
    #[test]
    fn beam_quotes_never_keep_a_control_literal() {
        for (input, reasoning, normal) in [
            (
                "<|channel|>final<|message|>Explain \"<|end|>\" literally.<|return|>",
                "",
                "Explain \"",
            ),
            (
                "<|channel|>final<|message|>Explain \"<|return|>\" literally.<|return|>",
                "",
                "Explain \"",
            ),
            (
                "<|channel|>analysis<|message|>say \"<|end|>\" now<|end|>",
                "say \"",
                "",
            ),
            (
                "<|channel|>final<|message|>say \"hi<|return|><|start|>assistant<|channel|>final<|message|>ok<|return|>",
                "",
                "say \"hiok",
            ),
            (
                "<|channel|>analysis<|message|>He said \"stop<|end|><|start|>assistant<|channel|>final<|message|>She said \"go\".<|return|>",
                "He said \"stop",
                "She said \"go\".",
            ),
            (
                "<|channel|>final<|message|>don't stop<|return|>",
                "",
                "don't stop",
            ),
        ] {
            for (r, n) in split_runs(input) {
                assert_eq!((r.as_str(), n.as_str()), (reasoning, normal), "{input}");
            }
        }
    }

    /// Under an explicit response prefill the whole stream is the reply: the first
    /// `<|return|>` closes and is stripped, the bytes after it stay visible.
    #[test]
    fn beam_response_prefill_is_reachable_through_the_registered_wrapper() {
        use crate::reasoning::ReasoningParserType;
        let mut parser = ReasoningParserType::Beam.get_reasoning_parser();
        parser.set_in_content(true);
        let mut r =
            parser.parse_reasoning_streaming_incremental("hi \"<|return|>\" x<|return|>", &[]);
        let f = parser.finish_reasoning_stream();
        r.normal_text.push_str(&f.normal_text);
        r.reasoning_text.push_str(&f.reasoning_text);
        assert_eq!(
            (r.reasoning_text.as_str(), r.normal_text.as_str()),
            ("", "hi \"\" x")
        );
    }

    #[test]
    fn beam_prefilled_reasoning() {
        let mut parser = BeamReasoningParser::new();
        parser.set_in_reasoning(true);
        let r = parser.parse_reasoning_streaming_incremental(
            "thought<|end|><|start|>assistant<|channel|>final<|message|>hi<|return|>",
            &[],
        );
        assert_eq!(
            (r.reasoning_text.as_str(), r.normal_text.as_str()),
            ("thought", "hi")
        );
    }
}
