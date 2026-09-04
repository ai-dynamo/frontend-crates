// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Kimi K3 XTML parsing.
//!
//! K3 has nested reasoning, response, tools, call, and argument channels, so it
//! owns one family state machine rather than forcing that grammar through the
//! wrapped-block scanner. The legacy tool parser projects this parser's events;
//! it does not maintain a second buffer or parser.

use std::collections::HashMap;

use serde_json::Value;

use crate::tool_calling::scan::{
    GuidedInvokePrefix, GuidedInvokePrefixContext, InvokeBoundary, InvokeBoundaryFactory,
    find_first_outside_strings, json_value_end, marker_prefix_suffix_len,
};
use crate::tool_calling::traits::{Tool, ToolCallDelta};
use crate::unified::{
    GuidedChannel, GuidedChannelState, GuidedGrammar, GuidedReasoning, GuidedRouted, NativeUnified,
    UnifiedParser, UnifiedParserOutput, UnifiedParserStartingState,
};

const OPEN: &str = "<|open|>";
const CLOSE: &str = "<|close|>";
const END_OF_MSG: &str = "<|end_of_msg|>";

const THINK_OPEN: Marker = Marker::pair("<|open|>think<|sep|>", "<|open|> think <|sep|>");
const THINK_CLOSE: Marker = Marker::pair("<|close|>think<|sep|>", "<|close|> think <|sep|>");
const THINK_CLOSE_HEAD: Marker = Marker::pair("<|close|>think", "<|close|> think");
const RESPONSE_OPEN: Marker = Marker::pair("<|open|>response<|sep|>", "<|open|> response <|sep|>");
const RESPONSE_CLOSE: Marker =
    Marker::pair("<|close|>response<|sep|>", "<|close|> response <|sep|>");
const TOOLS_OPEN: Marker = Marker::pair("<|open|>tools<|sep|>", "<|open|> tools <|sep|>");
const TOOLS_CLOSE: Marker = Marker::pair("<|close|>tools<|sep|>", "<|close|> tools <|sep|>");
const MESSAGE_OPEN: Marker = Marker::pair("<|open|>message", "<|open|> message");
const ASSISTANT_MESSAGE_OPEN: Marker = Marker::pair(
    "<|open|>message role=\"assistant\"<|sep|>",
    "<|open|> message role=\"assistant\" <|sep|>",
);
const MESSAGE_CLOSE: Marker = Marker::pair("<|close|>message<|sep|>", "<|close|> message <|sep|>");
const CALL_OPEN: Marker = Marker::pair("<|open|>call", "<|open|> call");
const CALL_CLOSE: Marker = Marker::pair("<|close|>call<|sep|>", "<|close|> call <|sep|>");
const ARG_OPEN: Marker = Marker::pair("<|open|>argument", "<|open|> argument");
const ARG_CLOSE: Marker = Marker::pair("<|close|>argument<|sep|>", "<|close|> argument <|sep|>");
const JSON_OPEN: Marker = Marker::pair("<|open|>json", "<|open|> json");
const JSON_CLOSE: Marker = Marker::pair("<|close|>json<|sep|>", "<|close|> json <|sep|>");

const ALL_MARKERS: &[Marker] = &[
    THINK_OPEN,
    THINK_CLOSE,
    RESPONSE_OPEN,
    RESPONSE_CLOSE,
    TOOLS_OPEN,
    TOOLS_CLOSE,
    ASSISTANT_MESSAGE_OPEN,
    MESSAGE_CLOSE,
    CALL_OPEN,
    CALL_CLOSE,
    ARG_OPEN,
    ARG_CLOSE,
    JSON_OPEN,
    JSON_CLOSE,
    Marker::single(END_OF_MSG),
];

const IDLE_MARKERS: &[Marker] = ALL_MARKERS;
const SEP_MARKER: Marker = Marker::pair("<|sep|>", " <|sep|>");
const REASONING_MARKERS: &[Marker] = &[
    THINK_CLOSE_HEAD,
    THINK_OPEN,
    THINK_CLOSE,
    RESPONSE_OPEN,
    RESPONSE_CLOSE,
    TOOLS_OPEN,
    TOOLS_CLOSE,
    CALL_OPEN,
    CALL_CLOSE,
    ARG_OPEN,
    ARG_CLOSE,
    JSON_OPEN,
    JSON_CLOSE,
    MESSAGE_CLOSE,
    Marker::single(END_OF_MSG),
];
const RESPONSE_MARKERS: &[Marker] = &[
    RESPONSE_CLOSE,
    TOOLS_OPEN,
    CALL_OPEN,
    MESSAGE_CLOSE,
    Marker::single(END_OF_MSG),
];
const TOOLS_MARKERS: &[Marker] = &[
    CALL_OPEN,
    TOOLS_CLOSE,
    MESSAGE_CLOSE,
    Marker::single(END_OF_MSG),
];
#[derive(Clone, Copy)]
struct Marker {
    canonical: &'static str,
    spaced: Option<&'static str>,
}

impl Marker {
    const fn single(canonical: &'static str) -> Self {
        Self {
            canonical,
            spaced: None,
        }
    }

    const fn pair(canonical: &'static str, spaced: &'static str) -> Self {
        Self {
            canonical,
            spaced: Some(spaced),
        }
    }

    fn variants(self) -> impl Iterator<Item = &'static str> {
        std::iter::once(self.canonical).chain(self.spaced)
    }

    fn match_at(self, text: &str) -> Option<(usize, usize)> {
        self.variants().find_map(|variant| {
            text.starts_with(variant)
                .then_some((0, variant.len()))
                .or_else(|| {
                    variant
                        .strip_prefix(' ')
                        .filter(|variant| text.starts_with(variant))
                        .map(|variant| (0, variant.len()))
                })
        })
    }

    fn prefix_len(self, text: &str) -> Option<usize> {
        self.match_at(text).map(|(_, len)| len)
    }

    fn find(self, text: &str) -> Option<(usize, usize)> {
        self.variants()
            .filter_map(|variant| text.find(variant).map(|at| (at, variant.len())))
            .min_by_key(|(at, _)| *at)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum Mode {
    #[default]
    Idle,
    Reasoning,
    Response,
    Tools,
    Call,
    Done,
}

#[derive(Debug)]
struct ActiveCall {
    name: String,
    id: String,
    return_mode: Mode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CallBoundary {
    Complete { body_end: usize, consumed: usize },
    Recover { body_end: usize },
    Resync { at: usize },
    Pending,
    Malformed,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CallBoundaryContext {
    Native,
    Guided,
}

/// Request context for the shared K3 call-boundary owner.
///
/// Guided mode may strip an incomplete native wrapper immediately before its JSON
/// payload. Native mode must instead wait for later native structure, because an
/// object- or array-shaped suffix can still be typed-string data.
struct KimiK3CallBoundary {
    context: CallBoundaryContext,
    return_channel: Mode,
    scanned: usize,
    header_len: Option<usize>,
    body_kind: CallBodyKind,
    root_call_open: Option<usize>,
    pending_call_open: Option<usize>,
    complete_call_open: Option<usize>,
    call_closes: Vec<TokenHit>,
    arg_opens: Vec<TokenHit>,
    arg_closes: Vec<TokenHit>,
    json_closes: Vec<TokenHit>,
    outer_closes: Vec<TokenHit>,
    completed_arguments: Option<String>,
    guided_prefix: GuidedBareCallPrefix,
    #[cfg(test)]
    scanned_bytes: usize,
    #[cfg(test)]
    parsed_body_bytes: usize,
    #[cfg(test)]
    body_parse_count: usize,
}

impl KimiK3CallBoundary {
    fn new(context: CallBoundaryContext) -> Self {
        Self {
            context,
            return_channel: Mode::Idle,
            scanned: 0,
            header_len: None,
            body_kind: CallBodyKind::Unknown,
            root_call_open: None,
            pending_call_open: None,
            complete_call_open: None,
            call_closes: Vec::new(),
            arg_opens: Vec::new(),
            arg_closes: Vec::new(),
            json_closes: Vec::new(),
            outer_closes: Vec::new(),
            completed_arguments: None,
            guided_prefix: GuidedBareCallPrefix::default(),
            #[cfg(test)]
            scanned_bytes: 0,
            #[cfg(test)]
            parsed_body_bytes: 0,
            #[cfg(test)]
            body_parse_count: 0,
        }
    }

    fn begin(&mut self, header_len: usize, return_channel: Mode) {
        self.reset();
        self.header_len = Some(header_len);
        self.return_channel = return_channel;
    }

    fn advance(&mut self, text: &str, flush: bool) -> CallBoundary {
        self.scan_appended(text, flush);
        let Some(header_len) = self.header_len else {
            return if flush {
                CallBoundary::Malformed
            } else {
                CallBoundary::Pending
            };
        };

        if let Some(next_call) = self.complete_call_open.filter(|at| *at > header_len) {
            if let Some(call_close) = self
                .call_closes
                .iter()
                .copied()
                .filter(|close| close.end() <= next_call)
                .find(|close| {
                    let rest = text[close.end()..next_call].trim_start();
                    TOOLS_CLOSE.prefix_len(rest).is_some() || rest.is_empty()
                })
            {
                let arguments = self.parse_body(&text[header_len..call_close.at]);
                if arguments.is_none() {
                    return CallBoundary::Resync { at: next_call };
                }
                if text[call_close.end()..next_call].trim().is_empty() {
                    self.completed_arguments = arguments;
                    return CallBoundary::Complete {
                        body_end: call_close.at,
                        consumed: call_close.end(),
                    };
                }
            }
            if let Some(body_end) = self.recovery_body_end(text, header_len, next_call) {
                return self.recover_at(text, header_len, body_end);
            }
            if let Some(call_close) = self
                .call_closes
                .iter()
                .copied()
                .find(|close| close.end() <= next_call)
                && let Some(arguments) = self.parse_body(&text[header_len..call_close.at])
            {
                self.completed_arguments = Some(arguments);
                return CallBoundary::Complete {
                    body_end: call_close.at,
                    consumed: call_close.end(),
                };
            }
            if self.body_kind == CallBodyKind::Malformed {
                return CallBoundary::Resync { at: next_call };
            }
        }

        if let Some((call_close, arguments)) = self.structural_call_close(text, header_len, flush) {
            self.completed_arguments = Some(arguments);
            return CallBoundary::Complete {
                body_end: call_close.at,
                consumed: call_close.end(),
            };
        }

        if !flush {
            return CallBoundary::Pending;
        }
        let recovery_limit = self
            .outer_closes
            .iter()
            .copied()
            .find(|outer| {
                !self.outer_boundary_is_typed_data(
                    text,
                    header_len,
                    TokenHit {
                        at: outer.end(),
                        len: 0,
                    },
                )
            })
            .map_or(text.len(), |outer| outer.at);
        if let Some(body_end) = self.recovery_body_end(text, header_len, recovery_limit) {
            return self.recover_at(text, header_len, body_end);
        }
        if self.context == CallBoundaryContext::Guided
            && let Some(body_end) = self.guided_recovery_body_end(text, header_len)
        {
            return self.recover_at(text, header_len, body_end);
        }
        CallBoundary::Malformed
    }

    fn structural_call_close(
        &mut self,
        text: &str,
        header_len: usize,
        flush: bool,
    ) -> Option<(TokenHit, String)> {
        let call_closes = self.call_closes.clone();
        for call_close in call_closes.into_iter().rev() {
            let boundary = self
                .outer_closes
                .iter()
                .copied()
                .find(|outer| outer.at >= call_close.end())
                .is_some()
                || self.context == CallBoundaryContext::Guided
                    && guided_payload_starts(&text[call_close.end()..]);
            if !boundary && (!flush || !text[call_close.end()..].trim().is_empty()) {
                continue;
            }
            let typed_data = self.outer_boundary_is_typed_data(text, header_len, call_close);
            let arguments = (!typed_data)
                .then(|| self.parse_body(&text[header_len..call_close.at]))
                .flatten();
            if let Some(arguments) = arguments {
                return Some((call_close, arguments));
            }
        }
        None
    }

    fn outer_boundary_is_typed_data(
        &self,
        text: &str,
        header_len: usize,
        call_close: TokenHit,
    ) -> bool {
        let Some((outer, arg_close, real_arg_open)) = self
            .outer_closes
            .iter()
            .copied()
            .filter(|outer| outer.at >= header_len && outer.at < call_close.at)
            .find_map(|outer| {
                let arg_close = self
                    .arg_closes
                    .iter()
                    .copied()
                    .rfind(|close| close.end() <= outer.at)?;
                let real_arg_open = self
                    .arg_opens
                    .iter()
                    .copied()
                    .rfind(|open| open.at < arg_close.at)?;
                let arg_type = parse_tag_header(&text[real_arg_open.at..], ARG_OPEN)
                    .and_then(|(attrs, _)| attr_value(&attrs, "type").map(str::to_string));
                let string_argument = arg_type.is_none_or(|arg_type| arg_type == "string");
                let later_arg_close = self
                    .arg_closes
                    .iter()
                    .copied()
                    .find(|close| close.at > outer.at)?;
                let later_arg_open = self
                    .arg_opens
                    .iter()
                    .copied()
                    .rfind(|open| open.at < later_arg_close.at)?;
                (string_argument && later_arg_open.at == real_arg_open.at).then_some((
                    outer,
                    arg_close,
                    real_arg_open,
                ))
            })
        else {
            return false;
        };
        if outer.end() < call_close.at {
            return self.arg_closes.iter().any(|close| close.at > outer.at)
                && self
                    .arg_opens
                    .iter()
                    .copied()
                    .rfind(|open| open.at < call_close.at)
                    .is_some_and(|open| open.at == real_arg_open.at);
        }
        let value_start = parse_tag_header(&text[real_arg_open.at..], ARG_OPEN)
            .map(|(_, len)| real_arg_open.at + len);
        value_start.is_some_and(|value_start| value_start <= arg_close.at)
    }

    fn recover_at(&mut self, text: &str, header_len: usize, body_end: usize) -> CallBoundary {
        match self.parse_body(&text[header_len..body_end]) {
            Some(arguments) => {
                self.completed_arguments = Some(arguments);
                CallBoundary::Recover { body_end }
            }
            None => CallBoundary::Malformed,
        }
    }

    fn parse_body(&mut self, body: &str) -> Option<String> {
        #[cfg(test)]
        {
            self.body_parse_count += 1;
            self.parsed_body_bytes += body.len();
        }
        parse_call_body(body)
    }

    fn recovery_body_end(&self, text: &str, header_len: usize, limit: usize) -> Option<usize> {
        if !self.call_closes.is_empty() {
            return None;
        }
        let candidate = match self.body_kind {
            CallBodyKind::Empty => Some(header_len),
            CallBodyKind::Arguments => self
                .arg_closes
                .iter()
                .copied()
                .rev()
                .find(|close| {
                    close.end() <= limit
                        && text[close.end()..limit].trim().is_empty()
                        && !self.outer_boundary_is_typed_data(
                            text,
                            header_len,
                            TokenHit { at: limit, len: 0 },
                        )
                })
                .map(TokenHit::end),
            CallBodyKind::RawJson => self
                .json_closes
                .iter()
                .copied()
                .rev()
                .find(|close| close.end() <= limit && text[close.end()..limit].trim().is_empty())
                .map(TokenHit::end),
            CallBodyKind::Unknown | CallBodyKind::Malformed => None,
        }?;
        (candidate <= limit && text[candidate..limit].trim().is_empty()).then_some(candidate)
    }

    fn guided_recovery_body_end(&self, text: &str, header_len: usize) -> Option<usize> {
        let candidates: Box<dyn Iterator<Item = usize> + '_> = match self.body_kind {
            CallBodyKind::Empty => Box::new(std::iter::once(header_len)),
            CallBodyKind::Arguments => {
                Box::new(self.arg_closes.iter().copied().rev().map(TokenHit::end))
            }
            CallBodyKind::RawJson => {
                Box::new(self.json_closes.iter().copied().rev().map(TokenHit::end))
            }
            CallBodyKind::Unknown | CallBodyKind::Malformed => Box::new(std::iter::empty()),
        };
        candidates.into_iter().find(|body_end| {
            let prefix = text[header_len..*body_end].trim();
            if self.context == CallBoundaryContext::Guided
                && !prefix.is_empty()
                && !prefix.contains("<|open|>")
                && !prefix.contains("<|close|>")
            {
                return false;
            }
            let rest = text[*body_end..].trim_start();
            let Some(end) = json_value_end(rest) else {
                return false;
            };
            let payload = &rest[..end];
            serde_json::from_str::<Value>(payload).is_ok()
                && rest[end..].trim().is_empty()
                && (self.context != CallBoundaryContext::Guided || guided_payload_starts(rest))
        })
    }

    fn take_arguments(&mut self) -> Option<String> {
        self.completed_arguments.take()
    }

    fn scan_appended(&mut self, text: &str, flush: bool) {
        if text.len() < self.scanned {
            self.reset_progress();
        }
        let ends_at_boundary = SCANNER_MARKERS
            .iter()
            .any(|(_, marker)| marker.variants().any(|variant| text.ends_with(variant)));
        let mut scan_limit = text
            .len()
            .saturating_sub((!flush && !ends_at_boundary) as usize * SCANNER_HOLDBACK);
        while !text.is_char_boundary(scan_limit) {
            scan_limit -= 1;
        }
        while self.scanned < scan_limit {
            let suffix = &text[self.scanned..scan_limit];
            if !flush && scanner_marker_is_partial(suffix) {
                break;
            }
            if let Some((kind, len)) = scanner_marker_at(suffix) {
                let hit = TokenHit {
                    at: self.scanned,
                    len,
                };
                self.note_token(text, kind, hit);
                self.scanned += len;
                #[cfg(test)]
                {
                    self.scanned_bytes += len;
                }
                continue;
            }

            let character = suffix.chars().next().expect("non-empty scanner suffix");
            if self
                .header_len
                .is_some_and(|header_len| self.scanned >= header_len)
                && self.body_kind == CallBodyKind::Unknown
                && !character.is_whitespace()
            {
                self.body_kind = CallBodyKind::Malformed;
            }
            self.scanned += character.len_utf8();
            #[cfg(test)]
            {
                self.scanned_bytes += character.len_utf8();
            }
        }
    }

    fn reset_progress(&mut self) {
        let context = self.context;
        let return_channel = self.return_channel;
        let header_len = self.header_len;
        *self = Self::new(context);
        self.return_channel = return_channel;
        self.header_len = header_len;
    }

    fn note_token(&mut self, text: &str, kind: ScannerToken, hit: TokenHit) {
        if kind == ScannerToken::CallOpen {
            if self.root_call_open.is_none() {
                self.root_call_open = Some(hit.at);
            } else {
                self.pending_call_open = Some(hit.at);
            }
        }
        if kind == ScannerToken::Sep {
            if self.header_len.is_none()
                && let Some(at) = self.root_call_open
                && let Some((_, len)) = parse_call_header(&text[at..])
                && at + len == hit.end()
            {
                self.header_len = Some(hit.end());
            }
            if let Some(at) = self.pending_call_open
                && let Some((_, len)) = parse_call_header(&text[at..])
                && at + len == hit.end()
            {
                self.complete_call_open = Some(at);
                self.pending_call_open = None;
            }
        }

        let Some(header_len) = self.header_len else {
            return;
        };
        if hit.at >= header_len && self.body_kind == CallBodyKind::Unknown {
            self.body_kind = match kind {
                ScannerToken::ArgumentOpen => CallBodyKind::Arguments,
                ScannerToken::JsonOpen => CallBodyKind::RawJson,
                ScannerToken::CallClose => CallBodyKind::Empty,
                ScannerToken::Sep => return,
                _ => CallBodyKind::Malformed,
            };
        }
        if hit.at < header_len {
            return;
        }
        match kind {
            ScannerToken::CallClose => self.call_closes.push(hit),
            ScannerToken::ArgumentOpen => self.arg_opens.push(hit),
            ScannerToken::ArgumentClose => self.arg_closes.push(hit),
            ScannerToken::JsonClose => self.json_closes.push(hit),
            ScannerToken::ToolsClose | ScannerToken::MessageClose | ScannerToken::EndOfMessage => {
                self.outer_closes.push(hit)
            }
            ScannerToken::ThinkClose if self.return_channel == Mode::Reasoning => {
                self.outer_closes.push(hit)
            }
            ScannerToken::ResponseClose if self.return_channel == Mode::Response => {
                self.outer_closes.push(hit)
            }
            ScannerToken::CallOpen
            | ScannerToken::JsonOpen
            | ScannerToken::Sep
            | ScannerToken::ThinkClose
            | ScannerToken::ResponseClose => {}
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum CallBodyKind {
    #[default]
    Unknown,
    Empty,
    Arguments,
    RawJson,
    Malformed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TokenHit {
    at: usize,
    len: usize,
}

impl TokenHit {
    fn end(self) -> usize {
        self.at + self.len
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScannerToken {
    CallOpen,
    CallClose,
    ArgumentOpen,
    ArgumentClose,
    JsonOpen,
    JsonClose,
    ToolsClose,
    MessageClose,
    ThinkClose,
    ResponseClose,
    EndOfMessage,
    Sep,
}

const SCANNER_MARKERS: &[(ScannerToken, Marker)] = &[
    (ScannerToken::CallOpen, CALL_OPEN),
    (ScannerToken::CallClose, CALL_CLOSE),
    (ScannerToken::ArgumentOpen, ARG_OPEN),
    (ScannerToken::ArgumentClose, ARG_CLOSE),
    (ScannerToken::JsonOpen, JSON_OPEN),
    (ScannerToken::JsonClose, JSON_CLOSE),
    (ScannerToken::ToolsClose, TOOLS_CLOSE),
    (ScannerToken::MessageClose, MESSAGE_CLOSE),
    (ScannerToken::ThinkClose, THINK_CLOSE),
    (ScannerToken::ResponseClose, RESPONSE_CLOSE),
    (ScannerToken::EndOfMessage, Marker::single(END_OF_MSG)),
    (ScannerToken::Sep, SEP_MARKER),
];

const SCANNER_HOLDBACK: usize = 32;

fn scanner_marker_at(text: &str) -> Option<(ScannerToken, usize)> {
    SCANNER_MARKERS
        .iter()
        .flat_map(|(kind, marker)| marker.variants().map(|variant| (*kind, variant)))
        .filter_map(|(kind, marker)| text.starts_with(marker).then_some((kind, marker.len())))
        .max_by_key(|(_, len)| *len)
}

fn scanner_marker_is_partial(text: &str) -> bool {
    SCANNER_MARKERS
        .iter()
        .flat_map(|(_, marker)| marker.variants())
        .any(|marker| marker.len() > text.len() && marker.starts_with(text))
}

fn guided_payload_starts(text: &str) -> bool {
    let text = text.trim_start();
    json_value_end(text).is_some_and(|end| serde_json::from_str::<Value>(&text[..end]).is_ok())
}

fn kimi_k3_call_boundary() -> Box<dyn InvokeBoundary> {
    Box::new(KimiK3CallBoundary::new(CallBoundaryContext::Guided))
}

impl InvokeBoundary for KimiK3CallBoundary {
    fn owns_guided_prefix(&self) -> bool {
        true
    }

    fn guided_prefix_append(
        &mut self,
        candidate: &str,
        _append: &str,
        _context: GuidedInvokePrefixContext,
    ) -> Option<GuidedInvokePrefix> {
        Some(match self.guided_prefix.advance(candidate) {
            BareCallPrefix::NoMatch => GuidedInvokePrefix::NoMatch,
            BareCallPrefix::Pending => GuidedInvokePrefix::Pending,
            BareCallPrefix::Complete(len) => GuidedInvokePrefix::Strip(len),
        })
    }

    fn end_append(
        &mut self,
        candidate: &str,
        _append: &str,
        flush: bool,
        _tool_index: usize,
    ) -> Option<usize> {
        match self.guided_prefix.advance(candidate) {
            BareCallPrefix::Complete(len) => return Some(len),
            BareCallPrefix::Pending => return None,
            BareCallPrefix::NoMatch => {}
        }
        match self.advance(candidate, flush) {
            CallBoundary::Complete { consumed, .. } => Some(consumed),
            CallBoundary::Recover { body_end } => Some(body_end),
            CallBoundary::Resync { .. } => None,
            CallBoundary::Pending | CallBoundary::Malformed => None,
        }
    }

    fn opens(&self, text: &str, at: usize) -> bool {
        CALL_OPEN.prefix_len(&text[at..]).is_some()
    }

    fn holdback(&self, text: &str) -> usize {
        if let Some(at) = CALL_OPEN
            .variants()
            .filter_map(|marker| text.rfind(marker))
            .max()
            && parse_call_header(&text[at..]).is_none()
        {
            return text.len() - at;
        }
        marker_prefix_suffix_len(
            text,
            [CALL_CLOSE, ARG_OPEN, ARG_CLOSE, JSON_OPEN, JSON_CLOSE]
                .into_iter()
                .flat_map(Marker::variants),
        )
    }

    fn resync(&mut self, text: &str, _flush: bool, _tool_index: usize) -> Option<usize> {
        match self.advance(text, false) {
            CallBoundary::Resync { at } => Some(at),
            CallBoundary::Complete { .. }
            | CallBoundary::Recover { .. }
            | CallBoundary::Pending
            | CallBoundary::Malformed => None,
        }
    }

    fn reset(&mut self) {
        let context = self.context;
        *self = Self::new(context);
    }
}

/// K3-owned native state. [`GuidedRouted`] adds the shared guided-JSON mode.
pub(crate) struct KimiK3Native {
    buffer: String,
    mode: Mode,
    active_call: Option<ActiveCall>,
    call_boundary: KimiK3CallBoundary,
    tools_open: String,
    tools_return_mode: Option<Mode>,
    next_tool_index: usize,
    call_ids: Vec<String>,
}

impl KimiK3Native {
    fn new() -> Self {
        Self {
            buffer: String::new(),
            mode: Mode::Idle,
            active_call: None,
            call_boundary: KimiK3CallBoundary::new(CallBoundaryContext::Native),
            tools_open: String::new(),
            tools_return_mode: None,
            next_tool_index: 0,
            call_ids: Vec::new(),
        }
    }

    fn drain(&mut self, flush: bool, output: &mut UnifiedParserOutput) {
        loop {
            let progressed = match self.mode {
                Mode::Idle => self.drain_idle(flush, output),
                Mode::Reasoning => self.drain_reasoning(flush, output),
                Mode::Response => self.drain_response(flush, output),
                Mode::Tools => self.drain_tools(flush, output),
                Mode::Call => self.drain_call(flush, output),
                Mode::Done => {
                    self.buffer.clear();
                    false
                }
            };
            if !progressed {
                break;
            }
        }
    }

    fn drain_idle(&mut self, flush: bool, output: &mut UnifiedParserOutput) -> bool {
        if self.consume_assistant_message_open() {
            return true;
        }
        if self.consume(THINK_OPEN) {
            self.mode = Mode::Reasoning;
            return true;
        }
        if self.consume(RESPONSE_OPEN) {
            self.mode = Mode::Response;
            return true;
        }
        if self.consume_tools_open(Mode::Idle) {
            self.mode = Mode::Tools;
            return true;
        }
        if self.consume_call_open(Mode::Tools, output) {
            self.mode = Mode::Call;
            return true;
        }
        if self.consume_message_end() {
            self.mode = Mode::Done;
            return true;
        }
        if let Some(len) = self.consume_any_at_start(&[
            THINK_CLOSE,
            RESPONSE_CLOSE,
            TOOLS_CLOSE,
            CALL_CLOSE,
            ARG_OPEN,
            ARG_CLOSE,
            JSON_OPEN,
            JSON_CLOSE,
        ]) {
            tracing::warn!(
                why = "kimi_k3_orphan_marker",
                skipped_bytes = len,
                "stripping orphan Kimi K3 XTML marker"
            );
            return true;
        }
        self.emit_safe(flush, IDLE_MARKERS, output, |output, text| {
            output.push_text(text)
        })
    }

    fn drain_reasoning(&mut self, flush: bool, output: &mut UnifiedParserOutput) -> bool {
        if self.consume_reasoning_close(flush) {
            self.mode = Mode::Idle;
            return true;
        }
        if self.consume(RESPONSE_OPEN) {
            tracing::warn!(
                why = "kimi_k3_elided_think_close",
                "recovering Kimi K3 response channel without a think close"
            );
            self.mode = Mode::Response;
            return true;
        }
        if self.consume_tools_open(Mode::Reasoning) {
            tracing::warn!(
                why = "kimi_k3_elided_think_close",
                "recovering Kimi K3 tools channel without a think close"
            );
            self.mode = Mode::Tools;
            return true;
        }
        if self.consume_call_open(Mode::Reasoning, output) {
            tracing::warn!(
                why = "kimi_k3_elided_think_close",
                "recovering Kimi K3 call without a think close"
            );
            self.mode = Mode::Call;
            return true;
        }
        if self.consume_message_end() {
            self.mode = Mode::Done;
            return true;
        }
        if self.consume_reasoning_structure(flush) {
            return true;
        }
        if let Some(len) = self.consume_any_at_start(&[
            THINK_OPEN,
            RESPONSE_CLOSE,
            TOOLS_CLOSE,
            CALL_CLOSE,
            ARG_CLOSE,
            JSON_CLOSE,
        ]) {
            tracing::warn!(
                why = "kimi_k3_orphan_marker_in_reasoning",
                skipped_bytes = len,
                "quarantining orphan Kimi K3 structure inside reasoning"
            );
            return true;
        }
        self.emit_safe(flush, REASONING_MARKERS, output, |output, text| {
            output.push_reasoning(text)
        })
    }

    fn drain_response(&mut self, flush: bool, output: &mut UnifiedParserOutput) -> bool {
        if self.consume(RESPONSE_CLOSE) {
            self.mode = Mode::Idle;
            return true;
        }
        if self.consume_tools_open(Mode::Idle) {
            self.mode = Mode::Tools;
            return true;
        }
        if self.consume_call_open(Mode::Response, output) {
            self.mode = Mode::Call;
            return true;
        }
        if self.consume_message_end() {
            self.mode = Mode::Done;
            return true;
        }
        self.emit_safe(flush, RESPONSE_MARKERS, output, |output, text| {
            output.push_text(text)
        })
    }

    fn drain_tools(&mut self, flush: bool, output: &mut UnifiedParserOutput) -> bool {
        if self.consume_call_open(Mode::Tools, output) {
            self.mode = Mode::Call;
            return true;
        }
        if self.consume(TOOLS_CLOSE) {
            self.tools_open.clear();
            self.mode = self.tools_return_mode.take().unwrap_or(Mode::Idle);
            return true;
        }
        if self.consume_message_end() {
            self.mode = Mode::Done;
            return true;
        }
        self.drop_safe(flush, TOOLS_MARKERS)
    }

    fn drain_call(&mut self, flush: bool, output: &mut UnifiedParserOutput) -> bool {
        let boundary = self.call_boundary.advance(&self.buffer, flush);
        match boundary {
            CallBoundary::Complete { body_end, consumed } => {
                self.complete_call(body_end, consumed, output);
                true
            }
            CallBoundary::Recover { body_end } => {
                tracing::warn!(
                    why = "kimi_k3_recovered_missing_call_close",
                    recovered_bytes = body_end,
                    "recovering delimiter-terminated Kimi K3 call before an outer boundary"
                );
                self.complete_call(body_end, body_end, output);
                true
            }
            CallBoundary::Resync { at } => {
                tracing::warn!(
                    why = "kimi_k3_resynchronized_after_incomplete_call",
                    skipped_bytes = at,
                    "dropping malformed Kimi K3 call and resuming at the next call"
                );
                self.buffer.drain(..at);
                self.active_call = None;
                self.call_boundary.reset();
                self.mode = Mode::Tools;
                true
            }
            CallBoundary::Pending if !flush => false,
            CallBoundary::Pending | CallBoundary::Malformed => {
                tracing::warn!(
                    why = "kimi_k3_incomplete_call",
                    buffered_bytes = self.buffer.len(),
                    "dropping incomplete Kimi K3 call at EOF"
                );
                self.buffer.clear();
                self.active_call = None;
                self.call_boundary.reset();
                self.mode = Mode::Tools;
                true
            }
        }
    }

    fn complete_call(
        &mut self,
        _body_end: usize,
        consumed: usize,
        output: &mut UnifiedParserOutput,
    ) {
        let arguments = self.call_boundary.take_arguments();
        self.buffer.drain(..consumed);
        self.call_boundary.reset();
        self.finish_call(arguments, output);
    }

    fn finish_call(&mut self, arguments: Option<String>, output: &mut UnifiedParserOutput) {
        let Some(call) = self.active_call.take() else {
            self.mode = Mode::Tools;
            return;
        };
        self.mode = call.return_mode;
        if call.name.is_empty() {
            tracing::warn!(
                why = "kimi_k3_missing_tool_name",
                "dropping Kimi K3 call without a tool name"
            );
            return;
        }
        let Some(arguments) = arguments else {
            tracing::warn!(
                why = "kimi_k3_malformed_call_body",
                "dropping malformed Kimi K3 call"
            );
            return;
        };

        let tool_index = self.next_tool_index;
        self.next_tool_index += 1;
        self.call_ids.push(call.id);
        output.push_call(ToolCallDelta {
            tool_index,
            name: Some(call.name),
            arguments,
            complete: true,
        });
    }

    fn consume_call_open(&mut self, return_mode: Mode, _output: &mut UnifiedParserOutput) -> bool {
        let Some(open_len) = CALL_OPEN.prefix_len(&self.buffer) else {
            return false;
        };
        let Some((attrs, header_len)) = parse_call_header(&self.buffer) else {
            if SEP_MARKER.find(&self.buffer[open_len..]).is_none() {
                return false;
            }
            let (sep_at, sep_len) = SEP_MARKER
                .find(&self.buffer[open_len..])
                .expect("separator existence checked");
            let malformed_len = open_len + sep_at + sep_len;
            tracing::warn!(
                why = "kimi_k3_malformed_call_header",
                skipped_bytes = malformed_len,
                "dropping malformed Kimi K3 call header"
            );
            self.buffer.drain(..malformed_len);
            return true;
        };
        let name = attr_value(&attrs, "tool").unwrap_or_default().to_string();
        let index = attr_value(&attrs, "index")
            .filter(|index| !index.is_empty())
            .map(str::to_string);
        let id = tool_call_id(&name, index.as_deref());
        self.active_call = Some(ActiveCall {
            name,
            id,
            return_mode,
        });
        self.call_boundary.begin(header_len, self.mode);
        true
    }

    fn consume_tools_open(&mut self, return_mode: Mode) -> bool {
        let Some(len) = TOOLS_OPEN.prefix_len(&self.buffer) else {
            return false;
        };
        self.tools_open = self.buffer[..len].to_string();
        self.buffer.drain(..len);
        self.tools_return_mode = Some(return_mode);
        true
    }

    fn consume_assistant_message_open(&mut self) -> bool {
        let Some((attrs, header_len)) = parse_tag_header(&self.buffer, MESSAGE_OPEN) else {
            return false;
        };
        if attr_value(&attrs, "role") != Some("assistant") {
            return false;
        }
        self.buffer.drain(..header_len);
        true
    }

    fn consume_reasoning_structure(&mut self, flush: bool) -> bool {
        let Some((open, close)) = [(ARG_OPEN, ARG_CLOSE), (JSON_OPEN, JSON_CLOSE)]
            .into_iter()
            .find(|(open, _)| open.prefix_len(&self.buffer).is_some())
        else {
            return false;
        };
        let open_len = open.prefix_len(&self.buffer).expect("matched opener");
        let Some((sep_at, sep_len)) = SEP_MARKER.find(&self.buffer[open_len..]) else {
            if flush {
                self.buffer.clear();
                return true;
            }
            return false;
        };
        let value_start = open_len + sep_at + sep_len;
        let close_hit = if close.canonical == JSON_CLOSE.canonical {
            find_first_outside_strings(
                &self.buffer[value_start..],
                [close.canonical, close.spaced.expect("paired marker")],
            )
        } else {
            close.find(&self.buffer[value_start..])
        };
        let Some((close_at, close_len)) = close_hit else {
            if flush {
                self.buffer.clear();
                return true;
            }
            return false;
        };
        let consumed = value_start + close_at + close_len;
        self.buffer.drain(..consumed);
        tracing::warn!(
            why = "kimi_k3_structure_in_reasoning",
            skipped_bytes = consumed,
            "quarantining Kimi K3 argument/json structure inside reasoning"
        );
        true
    }

    fn consume_reasoning_close(&mut self, flush: bool) -> bool {
        let Some(head_len) = THINK_CLOSE_HEAD.prefix_len(&self.buffer) else {
            return false;
        };
        if let Some(sep_len) = SEP_MARKER.prefix_len(&self.buffer[head_len..]) {
            self.buffer.drain(..head_len + sep_len);
            return true;
        }
        let tail = &self.buffer[head_len..];
        if tail.is_empty() && flush
            || [OPEN, CLOSE, END_OF_MSG]
                .iter()
                .any(|marker| tail.starts_with(marker))
        {
            self.buffer.drain(..head_len);
            return true;
        }
        false
    }

    fn consume_message_end(&mut self) -> bool {
        self.consume(MESSAGE_CLOSE) || self.consume(Marker::single(END_OF_MSG))
    }

    fn consume(&mut self, marker: Marker) -> bool {
        let Some(len) = marker.prefix_len(&self.buffer) else {
            return false;
        };
        self.buffer.drain(..len);
        true
    }

    fn consume_any_at_start(&mut self, markers: &[Marker]) -> Option<usize> {
        let len = markers
            .iter()
            .filter_map(|marker| marker.prefix_len(&self.buffer))
            .min()?;
        self.buffer.drain(..len);
        Some(len)
    }

    fn emit_safe(
        &mut self,
        flush: bool,
        markers: &[Marker],
        output: &mut UnifiedParserOutput,
        emit: impl FnOnce(&mut UnifiedParserOutput, String),
    ) -> bool {
        let len = safe_len(&self.buffer, markers, flush);
        if len == 0 {
            return false;
        }
        let text = self.buffer.drain(..len).collect();
        emit(output, text);
        true
    }

    fn drop_safe(&mut self, flush: bool, markers: &[Marker]) -> bool {
        let len = safe_len(&self.buffer, markers, flush);
        if len == 0 {
            return false;
        }
        self.buffer.drain(..len);
        true
    }

    fn reset_state(&mut self) {
        self.mode = Mode::Idle;
        self.active_call = None;
        self.call_boundary.reset();
        self.tools_open.clear();
        self.tools_return_mode = None;
        self.next_tool_index = 0;
        self.call_ids.clear();
    }
}

impl NativeUnified for KimiK3Native {
    fn preserve_special_tokens(&self) -> bool {
        true
    }

    fn tool_call_id(&self, tool_index: usize) -> Option<&str> {
        self.call_ids.get(tool_index).map(String::as_str)
    }

    fn guided_reasoning(&self) -> Option<GuidedReasoning> {
        Some(GuidedReasoning::Channel(GuidedChannel {
            find_open: guided_find_open,
            find_close: guided_find_close,
            find_turn_end: guided_find_turn_end,
            find_transition: guided_find_transition,
            find_stray: guided_find_stray,
            find_routing: guided_find_routing,
            holdback: guided_holdback,
            strip_text: guided_strip_text,
            competitors: GUIDED_COMPETITORS,
            close_markers: GUIDED_CLOSE_MARKERS,
            response_literal_markers: RESPONSE_LITERAL_MARKERS,
        }))
    }

    fn guided_grammar(&self) -> GuidedGrammar {
        GuidedGrammar {
            control_markers: ALL_MARKERS
                .iter()
                .flat_map(|marker| marker.variants())
                .map(str::to_string)
                .collect(),
            invoke_start: CALL_OPEN.canonical.to_string(),
            invoke_end: CALL_CLOSE.canonical.to_string(),
            invoke_boundary_factory: Some(InvokeBoundaryFactory::custom(kimi_k3_call_boundary)),
            guided_prefix_policy: None,
        }
    }

    fn apply_native_init(&mut self, starting_state: UnifiedParserStartingState) {
        self.buffer.clear();
        self.reset_state();
        self.restore_native_state(starting_state);
    }

    fn restore_native_state(&mut self, starting_state: UnifiedParserStartingState) {
        self.mode = match starting_state {
            UnifiedParserStartingState::None => Mode::Idle,
            UnifiedParserStartingState::Reasoning => Mode::Reasoning,
            UnifiedParserStartingState::Response => Mode::Response,
        };
    }

    fn push_native(&mut self, delta: &str, output: &mut UnifiedParserOutput) -> anyhow::Result<()> {
        self.buffer.push_str(delta);
        self.drain(false, output);
        Ok(())
    }

    fn finish_native(&mut self, output: &mut UnifiedParserOutput) -> anyhow::Result<()> {
        self.drain(true, output);
        match self.mode {
            Mode::Idle | Mode::Response => output.push_text(std::mem::take(&mut self.buffer)),
            Mode::Reasoning => output.push_reasoning(std::mem::take(&mut self.buffer)),
            Mode::Tools | Mode::Call | Mode::Done => self.buffer.clear(),
        }
        self.mode = Mode::Idle;
        self.active_call = None;
        Ok(())
    }

    fn reset_native(&mut self) -> String {
        let mut buffered = std::mem::take(&mut self.tools_open);
        buffered.push_str(&std::mem::take(&mut self.buffer));
        self.reset_state();
        buffered
    }
}

/// Build a native and guided Kimi K3 parser for one request stream.
pub(crate) fn kimi_k3_unified(_tools: &[Tool]) -> Box<dyn UnifiedParser> {
    Box::new(GuidedRouted::new(KimiK3Native::new()))
}

const GUIDED_COMPETITORS: &[&str] = &[
    "<|open|>think<|sep|>",
    "<|open|> think <|sep|>",
    "<|close|>think<|sep|>",
    "<|close|> think <|sep|>",
    "<|open|>response<|sep|>",
    "<|open|> response <|sep|>",
    "<|close|>message<|sep|>",
    "<|close|> message <|sep|>",
    END_OF_MSG,
];
const GUIDED_CLOSE_MARKERS: &[&str] = &["<|close|>think<|sep|>", "<|close|> think <|sep|>"];
const RESPONSE_LITERAL_MARKERS: &[&str] = &[
    "<|open|>think<|sep|>",
    "<|open|> think <|sep|>",
    "<|close|>think<|sep|>",
    "<|close|> think <|sep|>",
];

fn guided_find_open(
    haystack: &str,
    _flush: bool,
    _state: GuidedChannelState,
) -> Option<(usize, usize)> {
    THINK_OPEN.find(haystack)
}

fn guided_find_close(haystack: &str) -> Option<(usize, usize)> {
    THINK_CLOSE.find(haystack).or_else(|| {
        THINK_CLOSE_HEAD
            .variants()
            .filter_map(|head| {
                haystack.match_indices(head).find_map(|(at, _)| {
                    let tail = &haystack[at + head.len()..];
                    [OPEN, CLOSE, END_OF_MSG]
                        .iter()
                        .any(|marker| tail.starts_with(marker))
                        .then_some((at, head.len()))
                        .or_else(|| {
                            tail.trim_start()
                                .as_bytes()
                                .first()
                                .filter(|byte| matches!(byte, b'{' | b'['))
                                .map(|_| (at, head.len()))
                        })
                })
            })
            .min_by_key(|(at, _)| *at)
    })
}

fn guided_find_turn_end(haystack: &str) -> Option<(usize, usize)> {
    Marker::single(END_OF_MSG).find(haystack)
}

fn guided_find_transition(
    haystack: &str,
    _flush: bool,
    _state: GuidedChannelState,
) -> Option<(usize, usize)> {
    RESPONSE_OPEN.find(haystack)
}

fn guided_find_stray(
    _haystack: &str,
    _flush: bool,
    _state: GuidedChannelState,
) -> Option<(usize, usize)> {
    None
}

fn guided_find_routing(
    haystack: &str,
    flush: bool,
    state: GuidedChannelState,
) -> Option<(usize, usize)> {
    guided_find_transition(haystack, flush, state)
}

fn guided_holdback(haystack: &str, _state: GuidedChannelState) -> usize {
    haystack.len()
        - safe_len(
            haystack,
            &[THINK_OPEN, THINK_CLOSE, THINK_CLOSE_HEAD, RESPONSE_OPEN],
            false,
        )
}

fn guided_strip_text(text: &str) -> String {
    text.to_string()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BareCallPrefix {
    NoMatch,
    Pending,
    Complete(usize),
}

#[derive(Default)]
struct GuidedBareCallPrefix {
    scanned: usize,
    stage: GuidedBareCallPrefixStage,
    consumed: usize,
    pending_marker: Option<usize>,
    result: Option<BareCallPrefix>,
}

#[derive(Default)]
enum GuidedBareCallPrefixStage {
    #[default]
    Open,
    Whitespace,
    Tool {
        matched: usize,
    },
    Quote,
    Header {
        first: bool,
    },
}

impl GuidedBareCallPrefix {
    fn advance(&mut self, text: &str) -> BareCallPrefix {
        if let Some(result) = self.result {
            return result;
        }
        if matches!(self.stage, GuidedBareCallPrefixStage::Open) {
            let Some(open_len) = CALL_OPEN
                .variants()
                .find_map(|marker| text.starts_with(marker).then_some(marker.len()))
            else {
                return if CALL_OPEN.variants().any(|marker| marker.starts_with(text)) {
                    BareCallPrefix::Pending
                } else {
                    self.result = Some(BareCallPrefix::NoMatch);
                    BareCallPrefix::NoMatch
                };
            };
            self.scanned = open_len;
            self.stage = GuidedBareCallPrefixStage::Whitespace;
            count_guided_prefix_bytes(open_len);
        }
        while self.scanned < text.len() {
            match &mut self.stage {
                GuidedBareCallPrefixStage::Open => unreachable!(),
                GuidedBareCallPrefixStage::Whitespace => {
                    let ch = text[self.scanned..].chars().next().expect("header suffix");
                    if ch.is_whitespace() {
                        self.scanned += ch.len_utf8();
                        count_guided_prefix_bytes(ch.len_utf8());
                    } else {
                        self.stage = GuidedBareCallPrefixStage::Tool { matched: 0 };
                    }
                }
                GuidedBareCallPrefixStage::Tool { matched } => {
                    let expected = b"tool=";
                    let byte = text.as_bytes()[self.scanned];
                    count_guided_prefix_bytes(1);
                    if byte != expected[*matched] {
                        self.result = Some(BareCallPrefix::NoMatch);
                        break;
                    }
                    self.scanned += 1;
                    *matched += 1;
                    if *matched == expected.len() {
                        self.stage = GuidedBareCallPrefixStage::Quote;
                    }
                }
                GuidedBareCallPrefixStage::Quote => {
                    count_guided_prefix_bytes(1);
                    if text.as_bytes()[self.scanned] != b'"' {
                        self.result = Some(BareCallPrefix::NoMatch);
                        break;
                    }
                    self.scanned += 1;
                    self.consumed = self.scanned;
                    self.stage = GuidedBareCallPrefixStage::Header { first: true };
                }
                GuidedBareCallPrefixStage::Header { first } => {
                    if let Some(at) = self.pending_marker {
                        let rest = &text[at..];
                        if ALL_MARKERS.iter().any(|marker| {
                            marker.variants().any(|variant| rest.starts_with(variant))
                        }) {
                            self.result = Some(BareCallPrefix::Complete(self.consumed));
                            break;
                        }
                        if ALL_MARKERS.iter().any(|marker| {
                            marker.variants().any(|variant| variant.starts_with(rest))
                        }) {
                            break;
                        }
                        self.pending_marker = None;
                    }
                    let ch = text[self.scanned..].chars().next().expect("header body");
                    count_guided_prefix_bytes(ch.len_utf8());
                    if *first && matches!(ch, '{' | '[') {
                        self.result = Some(BareCallPrefix::Complete(self.consumed));
                        break;
                    }
                    *first = false;
                    if ch == '"' {
                        self.result = Some(BareCallPrefix::NoMatch);
                        break;
                    }
                    if ch == '<' {
                        self.pending_marker = Some(self.scanned);
                    }
                    self.scanned += ch.len_utf8();
                }
            }
        }
        self.result.unwrap_or(BareCallPrefix::Pending)
    }
}

#[cfg(test)]
std::thread_local! {
    static GUIDED_PREFIX_EXAMINED_BYTES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn count_guided_prefix_bytes(bytes: usize) {
    #[cfg(test)]
    GUIDED_PREFIX_EXAMINED_BYTES.with(|examined| examined.set(examined.get() + bytes));
    #[cfg(not(test))]
    let _ = bytes;
}

fn safe_len(text: &str, markers: &[Marker], flush: bool) -> usize {
    if let Some(position) = markers
        .iter()
        .filter_map(|marker| marker.find(text).map(|(at, _)| at))
        .min()
    {
        return position;
    }
    if flush {
        return text.len();
    }
    let holdback = markers
        .iter()
        .flat_map(|marker| marker.variants())
        .filter_map(|marker| {
            marker
                .char_indices()
                .map(|(at, _)| at)
                .filter(|at| *at > 0)
                .rev()
                .find(|at| text.ends_with(&marker[..*at]))
        })
        .max()
        .unwrap_or_default();
    text.len() - holdback
}

fn parse_tag_header(text: &str, open: Marker) -> Option<(Vec<(String, String)>, usize)> {
    let open_len = open.prefix_len(text)?;
    let (sep_at, sep_len) = SEP_MARKER.find(&text[open_len..])?;
    let header_end = open_len + sep_at;
    Some((
        parse_attrs(&text[open_len..header_end])?,
        header_end + sep_len,
    ))
}

fn parse_call_header(text: &str) -> Option<(Vec<(String, String)>, usize)> {
    parse_tag_header(text, CALL_OPEN)
}

fn parse_call_body(body: &str) -> Option<String> {
    let trimmed = body.trim();
    if trimmed.is_empty() {
        return Some("{}".to_string());
    }

    if let Some(open_len) = JSON_OPEN.prefix_len(trimmed) {
        let (sep_at, sep_len) = SEP_MARKER.find(&trimmed[open_len..])?;
        let value_start = open_len + sep_at + sep_len;
        let (close_at, close_len) = find_first_outside_strings(
            &trimmed[value_start..],
            [
                JSON_CLOSE.canonical,
                JSON_CLOSE.spaced.expect("paired marker"),
            ],
        )?;
        let value_end = value_start + close_at;
        if !trimmed[value_end + close_len..].trim().is_empty() {
            return None;
        }
        let raw = &trimmed[value_start..value_end];
        let Value::Object(_) = serde_json::from_str::<Value>(raw).ok()? else {
            return None;
        };
        return Some(compact_json(raw));
    }

    let mut fields = Vec::<(String, String)>::new();
    let mut field_positions = HashMap::<String, usize>::new();
    let mut cursor = 0;
    while cursor < trimmed.len() {
        cursor += trimmed[cursor..].len() - trimmed[cursor..].trim_start().len();
        if cursor == trimmed.len() {
            break;
        }
        let open_len = ARG_OPEN.prefix_len(&trimmed[cursor..])?;
        let (sep_at, sep_len) = SEP_MARKER.find(&trimmed[cursor + open_len..])?;
        let header_end = cursor + open_len + sep_at;
        let attrs = parse_attrs(&trimmed[cursor + open_len..header_end])?;
        let value_start = header_end + sep_len;
        let key = attr_value(&attrs, "key").unwrap_or_default().to_string();
        let arg_type = attr_value(&attrs, "type").unwrap_or("string");
        let (close_at, close_len) = structural_argument_close(&trimmed[value_start..])?;
        let value_end = value_start + close_at;
        let field_end = value_end + close_len;
        let value = encode_argument_value(arg_type, &trimmed[value_start..value_end]);
        count_argument_field_lookup();
        if let Some(position) = field_positions.get(&key).copied() {
            fields[position].1 = value;
        } else {
            field_positions.insert(key.clone(), fields.len());
            fields.push((key, value));
        }
        cursor = field_end;
    }

    let mut output = String::from("{");
    for (position, (key, value)) in fields.iter().enumerate() {
        if position > 0 {
            output.push(',');
        }
        output.push_str(&serde_json::to_string(key).ok()?);
        output.push(':');
        output.push_str(value);
    }
    output.push('}');
    Some(output)
}

fn structural_argument_close(value_and_rest: &str) -> Option<(usize, usize)> {
    #[derive(Clone, Copy)]
    enum NextArgument {
        None,
        AfterClose(TokenHit),
        Header(TokenHit, HeaderPhase),
    }

    #[derive(Clone, Copy)]
    enum HeaderPhase {
        Whitespace,
        Key,
        Quote,
        Value,
    }

    let mut next_argument = NextArgument::None;
    let mut cursor = 0;
    while cursor < value_and_rest.len() {
        let suffix = &value_and_rest[cursor..];
        if let Some(close_len) = argument_close_len(suffix) {
            count_argument_scan(close_len);
            next_argument = NextArgument::AfterClose(TokenHit {
                at: cursor,
                len: close_len,
            });
            cursor += close_len;
            continue;
        }

        match next_argument {
            NextArgument::AfterClose(close) => {
                let ch = suffix.chars().next()?;
                if ch.is_whitespace() {
                    count_argument_scan(ch.len_utf8());
                    cursor += ch.len_utf8();
                    continue;
                }
                if let Some(open_len) = ARG_OPEN.prefix_len(suffix) {
                    count_argument_scan(open_len);
                    cursor += open_len;
                    next_argument = NextArgument::Header(close, HeaderPhase::Whitespace);
                    continue;
                }
                next_argument = NextArgument::None;
            }
            NextArgument::Header(close, phase) => {
                if let Some(sep_len) = SEP_MARKER.prefix_len(suffix) {
                    count_argument_scan(sep_len);
                    if matches!(phase, HeaderPhase::Whitespace) {
                        return Some((close.at, close.len));
                    }
                    next_argument = NextArgument::None;
                    cursor += sep_len;
                    continue;
                }

                let ch = suffix.chars().next()?;
                let next_phase = match phase {
                    HeaderPhase::Whitespace if ch.is_whitespace() => HeaderPhase::Whitespace,
                    HeaderPhase::Whitespace if ch.is_alphanumeric() || ch == '_' => {
                        HeaderPhase::Key
                    }
                    HeaderPhase::Key if ch.is_alphanumeric() || ch == '_' => HeaderPhase::Key,
                    HeaderPhase::Key if ch == '=' => HeaderPhase::Quote,
                    HeaderPhase::Quote if ch == '"' => HeaderPhase::Value,
                    HeaderPhase::Value if ch == '"' => HeaderPhase::Whitespace,
                    HeaderPhase::Value => HeaderPhase::Value,
                    HeaderPhase::Whitespace | HeaderPhase::Key | HeaderPhase::Quote => {
                        next_argument = NextArgument::None;
                        continue;
                    }
                };
                count_argument_scan(ch.len_utf8());
                cursor += ch.len_utf8();
                next_argument = NextArgument::Header(close, next_phase);
                continue;
            }
            NextArgument::None => {}
        }

        let ch = suffix.chars().next()?;
        count_argument_scan(ch.len_utf8());
        cursor += ch.len_utf8();
    }
    match next_argument {
        NextArgument::AfterClose(close) => Some((close.at, close.len)),
        NextArgument::None | NextArgument::Header(_, _) => None,
    }
}

fn argument_close_len(text: &str) -> Option<usize> {
    ARG_CLOSE
        .variants()
        .find(|marker| marker_matches(text, marker))
        .map(str::len)
}

fn marker_matches(text: &str, marker: &str) -> bool {
    if text.len() < marker.len() {
        return false;
    }
    text.as_bytes()
        .iter()
        .zip(marker.as_bytes())
        .take(marker.len())
        .all(|(actual, expected)| {
            count_argument_comparison();
            actual == expected
        })
}

#[cfg(test)]
std::thread_local! {
    static ARGUMENT_SCAN_BYTES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static ARGUMENT_MARKER_COMPARISONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static ARGUMENT_FIELD_LOOKUPS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn count_argument_scan(bytes: usize) {
    #[cfg(test)]
    ARGUMENT_SCAN_BYTES.with(|scanned| scanned.set(scanned.get() + bytes));
    #[cfg(not(test))]
    let _ = bytes;
}

fn count_argument_comparison() {
    #[cfg(test)]
    ARGUMENT_MARKER_COMPARISONS.with(|comparisons| comparisons.set(comparisons.get() + 1));
}

fn count_argument_field_lookup() {
    #[cfg(test)]
    ARGUMENT_FIELD_LOOKUPS.with(|lookups| lookups.set(lookups.get() + 1));
}

#[cfg(test)]
fn reset_argument_work() {
    ARGUMENT_SCAN_BYTES.with(|scanned| scanned.set(0));
    ARGUMENT_MARKER_COMPARISONS.with(|comparisons| comparisons.set(0));
    ARGUMENT_FIELD_LOOKUPS.with(|lookups| lookups.set(0));
}

#[cfg(test)]
fn argument_work() -> (usize, usize, usize) {
    (
        ARGUMENT_SCAN_BYTES.with(std::cell::Cell::get),
        ARGUMENT_MARKER_COMPARISONS.with(std::cell::Cell::get),
        ARGUMENT_FIELD_LOOKUPS.with(std::cell::Cell::get),
    )
}

fn encode_argument_value(arg_type: &str, raw: &str) -> String {
    if arg_type == "string" {
        return serde_json::to_string(raw).expect("serializing a Rust string cannot fail");
    }
    if serde_json::from_str::<Value>(raw).is_ok() {
        compact_json(raw)
    } else {
        serde_json::to_string(raw).expect("serializing a Rust string cannot fail")
    }
}

fn compact_json(raw: &str) -> String {
    let mut output = String::with_capacity(raw.len());
    let mut in_string = false;
    let mut escaped = false;
    for character in raw.chars() {
        if in_string {
            output.push(character);
            if escaped {
                escaped = false;
            } else if character == '\\' {
                escaped = true;
            } else if character == '"' {
                in_string = false;
            }
        } else if character == '"' {
            in_string = true;
            output.push(character);
        } else if !character.is_whitespace() {
            output.push(character);
        }
    }
    output
}

fn parse_attrs(input: &str) -> Option<Vec<(String, String)>> {
    let mut attrs = Vec::new();
    let mut cursor = 0;
    while cursor < input.len() {
        cursor += input[cursor..].len() - input[cursor..].trim_start().len();
        if cursor == input.len() {
            break;
        }
        let rest = &input[cursor..];
        let key_len = rest
            .char_indices()
            .take_while(|(_, character)| character.is_alphanumeric() || *character == '_')
            .map(|(at, character)| at + character.len_utf8())
            .last()?;
        let key = &rest[..key_len];
        let value = rest[key_len..].strip_prefix("=\"")?;
        let end = value.find('"')?;
        attrs.push((key.to_string(), unescape_attr(&value[..end])));
        cursor += key_len + 2 + end + 1;
    }
    Some(attrs)
}

fn unescape_attr(value: &str) -> String {
    value.replace("&quot;", "\"").replace("&amp;", "&")
}

fn attr_value<'a>(attrs: &'a [(String, String)], key: &str) -> Option<&'a str> {
    attrs
        .iter()
        .find(|(name, _)| name == key)
        .map(|(_, value)| value.as_str())
}

fn tool_call_id(name: &str, index: Option<&str>) -> String {
    match index.filter(|index| !index.is_empty()) {
        None => name.to_string(),
        Some(index) => index.parse::<i64>().map_or_else(
            |_| format!("{name}:{index}"),
            |index| format!("{name}:{}", index - 1),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool_calling::traits::ToolParser;
    use crate::unified::{
        InvalidGuidedPayloadPolicy, UnifiedEvent, UnifiedParserExt, UnifiedParserInit,
        UnifiedToolOutputMode, assemble,
    };

    const SEP: &str = "<|sep|>";

    fn arg(key: &str, arg_type: &str, value: &str) -> String {
        format!("{OPEN}argument key=\"{key}\" type=\"{arg_type}\"{SEP}{value}{CLOSE}argument{SEP}")
    }

    fn call(name: &str, index: &str, body: &str) -> String {
        format!("{OPEN}call tool=\"{name}\" index=\"{index}\"{SEP}{body}{CLOSE}call{SEP}")
    }

    fn run(input: &str, state: UnifiedParserStartingState) -> (Vec<UnifiedEvent>, Vec<String>) {
        let mut parser = kimi_k3_unified(&[]);
        parser
            .initialize_request(UnifiedParserInit {
                starting_state: state,
                ..UnifiedParserInit::default()
            })
            .unwrap();
        let mut events = parser.push(input).unwrap();
        events.extend(parser.finish().unwrap().events);
        let ids = (0..4)
            .filter_map(|index| parser.tool_call_id(index).map(str::to_string))
            .collect();
        (assemble(&events), ids)
    }

    fn assert_all_splits(input: &str, expected: &[UnifiedEvent]) {
        for split in (0..=input.len()).filter(|split| input.is_char_boundary(*split)) {
            let mut parser = kimi_k3_unified(&[]);
            let mut events = parser.push(&input[..split]).unwrap();
            events.extend(parser.push(&input[split..]).unwrap());
            events.extend(parser.finish().unwrap().events);
            assert_eq!(assemble(&events), expected, "split at byte {split}");
        }
    }

    fn native_events_from_state(
        input: &str,
        chunks: &[usize],
        starting_state: UnifiedParserStartingState,
    ) -> Vec<crate::UnifiedParserEvent> {
        let mut parser = kimi_k3_unified(&[]);
        parser
            .initialize_request(UnifiedParserInit {
                starting_state,
                ..UnifiedParserInit::default()
            })
            .unwrap();
        let mut events = Vec::new();
        let mut start = 0;
        for end in chunks.iter().copied().chain(std::iter::once(input.len())) {
            events.extend(parser.push(&input[start..end]).unwrap());
            start = end;
        }
        events.extend(parser.finish().unwrap().events);
        events
    }

    fn assert_all_utf8_fragmentations(input: &str, expected: &[UnifiedEvent]) {
        assert_all_utf8_fragmentations_from_state(
            input,
            UnifiedParserStartingState::None,
            expected,
        );
    }

    fn assert_all_utf8_fragmentations_from_state(
        input: &str,
        starting_state: UnifiedParserStartingState,
        expected: &[UnifiedEvent],
    ) {
        let boundaries: Vec<usize> = (1..input.len())
            .filter(|at| input.is_char_boundary(*at))
            .collect();
        assert_eq!(
            assemble(&native_events_from_state(input, &[], starting_state)),
            expected
        );
        for split in &boundaries {
            assert_eq!(
                assemble(&native_events_from_state(input, &[*split], starting_state)),
                expected,
                "split at byte {split}"
            );
        }
        assert_eq!(
            assemble(&native_events_from_state(
                input,
                &boundaries,
                starting_state,
            )),
            expected,
            "one Unicode scalar per push"
        );
    }

    fn assert_every_utf8_split_cross_product(input: &str, expected: &[UnifiedEvent]) {
        let boundaries: Vec<usize> = (0..=input.len())
            .filter(|at| input.is_char_boundary(*at))
            .collect();
        for split in &boundaries {
            assert_eq!(
                assemble(&native_events_from_state(
                    input,
                    &[*split],
                    UnifiedParserStartingState::None,
                )),
                expected,
                "split at byte {split}"
            );
        }
        assert_eq!(
            assemble(&native_events_from_state(
                input,
                &boundaries,
                UnifiedParserStartingState::None,
            )),
            expected,
            "one Unicode scalar per push"
        );
    }

    fn guided_events(
        input: &str,
        chunks: &[usize],
        starting_state: UnifiedParserStartingState,
    ) -> Vec<UnifiedEvent> {
        let mut parser = kimi_k3_unified(&[]);
        parser
            .initialize_request(UnifiedParserInit {
                starting_state,
                tool_output_mode: UnifiedToolOutputMode::GuidedJson { named_tool: None },
                invalid_guided_payload: InvalidGuidedPayloadPolicy::RecoverAsText,
                ..UnifiedParserInit::default()
            })
            .unwrap();
        let mut events = Vec::new();
        let mut start = 0;
        for end in chunks.iter().copied().chain(std::iter::once(input.len())) {
            events.extend(parser.push(&input[start..end]).unwrap());
            start = end;
        }
        events.extend(parser.finish().unwrap().events);
        assemble(&events)
    }

    fn assert_guided_all_utf8_fragmentations(
        input: &str,
        starting_state: UnifiedParserStartingState,
        expected: &[UnifiedEvent],
    ) {
        let boundaries: Vec<usize> = (1..input.len())
            .filter(|at| input.is_char_boundary(*at))
            .collect();
        assert_eq!(guided_events(input, &[], starting_state), expected);
        for split in &boundaries {
            assert_eq!(
                guided_events(input, &[*split], starting_state),
                expected,
                "split at byte {split}"
            );
        }
        assert_eq!(
            guided_events(input, &boundaries, starting_state),
            expected,
            "one Unicode scalar per push"
        );
    }

    #[test]
    fn ordered_channels_typed_arguments_ids_and_all_splits() {
        let body = [
            arg("city", "string", "Zürich"),
            arg("days", "number", "1.0"),
            arg("rain", "boolean", "true"),
        ]
        .concat();
        let input = format!(
            "{OPEN}think{SEP}plan{CLOSE}think{SEP}{OPEN}response{SEP}checking{CLOSE}response{SEP}{OPEN}tools{SEP}{}{OPEN}call tool=\"second\" index=\"raw\"{SEP}{CLOSE}call{SEP}{CLOSE}tools{SEP}{CLOSE}message{SEP}{END_OF_MSG}",
            call("weather", "1", &body)
        );
        let expected = vec![
            UnifiedEvent::Reasoning {
                text: "plan".into(),
            },
            UnifiedEvent::Text {
                text: "checking".into(),
            },
            UnifiedEvent::ToolCall {
                name: "weather".into(),
                arguments: serde_json::json!({"city":"Zürich","days":1.0,"rain":true}),
            },
            UnifiedEvent::ToolCall {
                name: "second".into(),
                arguments: serde_json::json!({}),
            },
        ];
        assert_all_splits(&input, &expected);
        let (_, ids) = run(&input, UnifiedParserStartingState::None);
        assert_eq!(ids, ["weather:0", "second:raw"]);
    }

    #[test]
    fn every_closed_channel_returns_to_idle_for_later_channels() {
        let first = call("f", "1", &arg("x", "number", "1"));
        let second = call("g", "2", &arg("y", "string", "Zürich"));
        let input = format!(
            "{}before{}{}{}{}{}mid{}{}{}{}{}after{}{}final{}",
            RESPONSE_OPEN.canonical,
            RESPONSE_CLOSE.canonical,
            TOOLS_OPEN.canonical,
            first,
            TOOLS_CLOSE.canonical,
            THINK_OPEN.canonical,
            THINK_CLOSE.canonical,
            TOOLS_OPEN.canonical,
            second,
            TOOLS_CLOSE.canonical,
            RESPONSE_OPEN.canonical,
            RESPONSE_CLOSE.canonical,
            THINK_OPEN.canonical,
            THINK_CLOSE.canonical,
        );
        assert_all_utf8_fragmentations(
            &input,
            &[
                UnifiedEvent::Text {
                    text: "before".into(),
                },
                UnifiedEvent::ToolCall {
                    name: "f".into(),
                    arguments: serde_json::json!({"x":1}),
                },
                UnifiedEvent::Reasoning { text: "mid".into() },
                UnifiedEvent::ToolCall {
                    name: "g".into(),
                    arguments: serde_json::json!({"y":"Zürich"}),
                },
                UnifiedEvent::Text {
                    text: "after".into(),
                },
                UnifiedEvent::Reasoning {
                    text: "final".into(),
                },
            ],
        );
    }

    #[test]
    fn spaced_markers_and_prefilled_channels() {
        let input = concat!(
            "private<|close|> think <|sep|>",
            "<|open|> response <|sep|>visible",
            "<|close|> response <|sep|>",
            "<|close|> message <|sep|>"
        );
        assert_eq!(
            run(input, UnifiedParserStartingState::Reasoning).0,
            vec![
                UnifiedEvent::Reasoning {
                    text: "private".into()
                },
                UnifiedEvent::Text {
                    text: "visible".into()
                }
            ]
        );
        assert_eq!(
            run(
                "visible<|close|>response<|sep|>",
                UnifiedParserStartingState::Response
            )
            .0,
            vec![UnifiedEvent::Text {
                text: "visible".into()
            }]
        );
    }

    #[test]
    fn elided_think_close_hands_off_to_response_or_tools() {
        let response = format!("private{}visible", RESPONSE_OPEN.canonical);
        assert_eq!(
            run(&response, UnifiedParserStartingState::Reasoning).0,
            vec![
                UnifiedEvent::Reasoning {
                    text: "private".into()
                },
                UnifiedEvent::Text {
                    text: "visible".into()
                }
            ]
        );
        let tool = format!(
            "private{}{}{}",
            TOOLS_OPEN.canonical,
            call("calc", "x", &arg("n", "number", "4")),
            TOOLS_CLOSE.canonical
        );
        assert_eq!(
            run(&tool, UnifiedParserStartingState::Reasoning).0,
            vec![
                UnifiedEvent::Reasoning {
                    text: "private".into()
                },
                UnifiedEvent::ToolCall {
                    name: "calc".into(),
                    arguments: serde_json::json!({"n":4})
                }
            ]
        );
    }

    #[test]
    fn elided_think_close_head_hands_off_to_structure_or_eof() {
        let response = format!(
            "private{}{}visible",
            THINK_CLOSE_HEAD.canonical, RESPONSE_OPEN.canonical
        );
        assert_eq!(
            run(&response, UnifiedParserStartingState::Reasoning).0,
            vec![
                UnifiedEvent::Reasoning {
                    text: "private".into()
                },
                UnifiedEvent::Text {
                    text: "visible".into()
                }
            ]
        );
        assert_eq!(
            run(
                &format!("private{}", THINK_CLOSE_HEAD.canonical),
                UnifiedParserStartingState::Reasoning
            )
            .0,
            vec![UnifiedEvent::Reasoning {
                text: "private".into()
            }]
        );
    }

    #[test]
    fn raw_json_and_marker_like_argument_data_survive() {
        let raw = r#"{"command":"literal <|close|>call<|sep|>, <|close|>json<|sep|>, and <|open|>call data"}"#;
        let body = format!("{OPEN}json type=\"object\"{SEP}{raw}{CLOSE}json{SEP}");
        let input = format!(
            "{}{}{}",
            TOOLS_OPEN.canonical,
            call("run", "2", &body),
            TOOLS_CLOSE.canonical
        );
        let expected = vec![UnifiedEvent::ToolCall {
            name: "run".into(),
            arguments: serde_json::from_str(raw).unwrap(),
        }];
        assert_all_splits(&input, &expected);
    }

    #[test]
    fn raw_json_must_be_a_valid_object_before_call_commit() {
        for raw in ["{", r#"{"x":}"#, r#""scalar""#, "[]", "null"] {
            let body = format!("{OPEN}json type=\"object\"{SEP}{raw}{CLOSE}json{SEP}");
            let invalid = call("bad", "1", &body);
            let valid = call("good", "2", &arg("x", "number", "7"));
            let input = format!(
                "{}{}{}{}",
                TOOLS_OPEN.canonical, invalid, valid, TOOLS_CLOSE.canonical
            );

            for split in (0..=input.len()).filter(|at| input.is_char_boundary(*at)) {
                let mut parser = kimi_k3_unified(&[]);
                let mut events = parser.push(&input[..split]).unwrap();
                events.extend(parser.push(&input[split..]).unwrap());
                events.extend(parser.finish().unwrap().events);
                assert_eq!(
                    events,
                    vec![crate::UnifiedParserEvent::ToolCall(ToolCallDelta {
                        tool_index: 0,
                        name: Some("good".into()),
                        arguments: r#"{"x":7}"#.into(),
                        complete: true,
                    })],
                    "raw JSON {raw:?}, split at byte {split}"
                );
                assert_eq!(parser.tool_call_id(0), Some("good:1"));
                assert_eq!(parser.tool_call_id(1), None);
            }
        }
    }

    #[test]
    fn typed_string_preserves_argument_close_marker_data() {
        let value = "before<|close|>argument<|sep|>after";
        let input = format!(
            "{}{}{}",
            TOOLS_OPEN.canonical,
            call("echo", "1", &arg("value", "string", value)),
            TOOLS_CLOSE.canonical
        );
        assert_all_splits(
            &input,
            &[UnifiedEvent::ToolCall {
                name: "echo".into(),
                arguments: serde_json::json!({"value": value}),
            }],
        );
    }

    #[test]
    fn typed_string_preserves_argument_close_before_brace_or_bracket() {
        for marker in ARG_CLOSE.variants() {
            for suffix in [
                "{literal}after",
                "[literal]after",
                "{\"city\":\"Zürich\"}",
                "[\"Zürich\"]",
                "{\"city\":\"Zürich\"}after",
                "[\"Zürich\"]after",
            ] {
                let value = format!("before{marker}{suffix}");
                let input = format!(
                    "{}{}{}",
                    TOOLS_OPEN.canonical,
                    call("echo", "1", &arg("value", "string", &value)),
                    TOOLS_CLOSE.canonical
                );
                assert_all_utf8_fragmentations(
                    &input,
                    &[UnifiedEvent::ToolCall {
                        name: "echo".into(),
                        arguments: serde_json::json!({"value": value}),
                    }],
                );
            }
        }
    }

    #[test]
    fn typed_string_owns_every_marker_like_suffix_until_real_close() {
        let suffixes = [
            TOOLS_CLOSE.canonical,
            MESSAGE_CLOSE.canonical,
            END_OF_MSG,
            THINK_CLOSE.canonical,
            RESPONSE_CLOSE.canonical,
            ARG_OPEN.canonical,
            CALL_CLOSE.canonical,
            "{literal}[bytes]",
            "arbitrary Zürich text",
        ];
        for arg_close in ARG_CLOSE.variants() {
            for suffix in suffixes {
                let value = format!("before{arg_close}{suffix} after");
                let body = [
                    arg("first", "number", "1"),
                    arg("value", "string", &value),
                    arg("last", "boolean", "true"),
                ]
                .concat();
                let input = format!(
                    "{}{}{}",
                    TOOLS_OPEN.canonical,
                    call("echo", "1", &body),
                    TOOLS_CLOSE.canonical
                );
                assert_every_utf8_split_cross_product(
                    &input,
                    &[UnifiedEvent::ToolCall {
                        name: "echo".into(),
                        arguments: serde_json::json!({
                            "first": 1,
                            "value": value,
                            "last": true,
                        }),
                    }],
                );
            }
        }
    }

    #[test]
    fn call_boundary_scan_work_is_linear_for_one_character_chunks() {
        fn work(size: usize) -> (usize, usize, usize) {
            let value = "x".repeat(size);
            let input = call("echo", "1", &arg("value", "string", &value));
            let (_, header_len) = parse_call_header(&input).expect("complete call header");
            let mut boundary = KimiK3CallBoundary::new(CallBoundaryContext::Native);
            boundary.begin(header_len, Mode::Tools);
            for end in (1..=input.len()).filter(|at| input.is_char_boundary(*at)) {
                let flush = end == input.len();
                let _ = boundary.advance(&input[..end], flush);
            }
            (
                boundary.scanned_bytes,
                boundary.parsed_body_bytes,
                boundary.body_parse_count,
            )
        }

        let small = work(4_096);
        let large = work(8_192);
        assert!(small.0 <= 4_096 + 160, "small scan work: {small:?}");
        assert!(large.0 <= 8_192 + 160, "large scan work: {large:?}");
        assert!(
            large.0 <= small.0 * 2,
            "scan work did not double: {small:?} -> {large:?}"
        );
        assert_eq!(small.2, 1, "small body parses: {small:?}");
        assert_eq!(large.2, 1, "large body parses: {large:?}");
        assert!(
            large.1 <= small.1 * 2,
            "parse work did not double: {small:?} -> {large:?}"
        );
    }

    #[test]
    fn typed_argument_parse_work_is_linear_for_size_doubling() {
        fn work(repetitions: usize) -> (usize, usize, usize) {
            reset_argument_work();
            let false_close = format!("{}literal", ARG_CLOSE.canonical);
            let body = arg("value", "string", &false_close.repeat(repetitions));
            parse_call_body(&body).expect("typed argument body");
            argument_work()
        }

        let small = work(128);
        let large = work(256);
        println!("K3 typed size work: 128={small:?}, 256={large:?}");
        assert!(small.0 > 0 && small.1 > 0, "small work: {small:?}");
        assert!(
            large.0 <= small.0 * 2 + 64,
            "scan bytes grew faster than linearly: {small:?} -> {large:?}"
        );
        assert!(
            large.1 <= small.1 * 2 + 64,
            "marker comparisons grew faster than linearly: {small:?} -> {large:?}"
        );
        assert_eq!(small.2, 1);
        assert_eq!(large.2, 1);
    }

    #[test]
    fn typed_argument_parse_work_is_linear_in_field_count() {
        fn work(field_count: usize) -> (usize, usize, usize) {
            reset_argument_work();
            let body = (0..field_count)
                .map(|index| arg(&format!("field_{index}"), "number", &index.to_string()))
                .collect::<String>();
            let parsed = parse_call_body(&body).expect("many typed arguments");
            let value: Value = serde_json::from_str(&parsed).expect("arguments JSON");
            assert_eq!(
                value.as_object().expect("arguments object").len(),
                field_count
            );
            argument_work()
        }

        let small = work(128);
        let large = work(256);
        println!("K3 typed field work: 128={small:?}, 256={large:?}");
        assert_eq!(small.2, 128, "one map lookup per small field: {small:?}");
        assert_eq!(large.2, 256, "one map lookup per large field: {large:?}");
        assert!(
            large.0 <= small.0 * 2 + 1_024,
            "scan bytes grew faster than bytes plus fields: {small:?} -> {large:?}"
        );
        assert!(
            large.1 <= small.1 * 2 + 1_024,
            "marker comparisons grew faster than bytes plus fields: {small:?} -> {large:?}"
        );
    }

    #[test]
    fn duplicate_typed_arguments_keep_first_position_and_last_value() {
        let body = [
            arg("first", "number", "1"),
            arg("duplicate", "string", "old"),
            arg("last", "boolean", "true"),
            arg("duplicate", "string", "new"),
        ]
        .concat();
        assert_eq!(
            parse_call_body(&body).as_deref(),
            Some(r#"{"first":1,"duplicate":"new","last":true}"#)
        );
    }

    #[test]
    fn malformed_close_candidate_inside_string_preserves_multi_argument_split_parity() {
        let malformed_next = format!("{} broken{SEP}", ARG_OPEN.canonical);
        let value = format!(
            "before{}{}after",
            ARG_CLOSE.spaced.expect("paired marker"),
            malformed_next
        );
        let body = [
            arg("first", "string", &value),
            arg(
                "second",
                "object",
                r#"{"marker":"<|close|>argument<|sep|>"}"#,
            ),
            arg("third", "number", "3"),
        ]
        .concat();
        let input = format!(
            "{}{}{}",
            TOOLS_OPEN.canonical,
            call("echo", "1", &body),
            TOOLS_CLOSE.canonical
        );
        assert_every_utf8_split_cross_product(
            &input,
            &[UnifiedEvent::ToolCall {
                name: "echo".into(),
                arguments: serde_json::json!({
                    "first": value,
                    "second": {"marker":"<|close|>argument<|sep|>"},
                    "third": 3,
                }),
            }],
        );
    }

    #[test]
    fn argument_close_before_guided_payload_remains_structural() {
        let payload = r#"[{"name":"weather","arguments":{"city":"Zürich"}}]"#;
        let wrapper = format!(
            "{}{OPEN}call tool=\"ignored\" index=\"1\"{SEP}{OPEN}argument key=\"quoted\" type=\"string\"{SEP}literal{}",
            TOOLS_OPEN.canonical, ARG_CLOSE.canonical
        );
        let input = format!("{wrapper}{payload}");
        assert_guided_all_utf8_fragmentations(
            &input,
            UnifiedParserStartingState::None,
            &[UnifiedEvent::ToolCall {
                name: "weather".into(),
                arguments: serde_json::json!({"city":"Zürich"}),
            }],
        );
    }

    #[test]
    fn mixed_argument_close_spellings_keep_source_order() {
        let first = format!(
            "{OPEN}argument key=\"first\" type=\"string\"{SEP}one{}",
            ARG_CLOSE.spaced.expect("paired marker")
        );
        let body = format!("{first}{}", arg("second", "string", "two"));
        let input = format!(
            "{}{}{}",
            TOOLS_OPEN.canonical,
            call("echo", "1", &body),
            TOOLS_CLOSE.canonical
        );
        assert_all_utf8_fragmentations(
            &input,
            &[UnifiedEvent::ToolCall {
                name: "echo".into(),
                arguments: serde_json::json!({"first":"one","second":"two"}),
            }],
        );
    }

    #[test]
    fn non_string_json_keeps_argument_close_inside_quoted_data() {
        for (arg_type, value, expected) in [
            (
                "object",
                r#"{"x":"before<|close|>argument<|sep|>{literal}after"}"#,
                serde_json::json!({"x":"before<|close|>argument<|sep|>{literal}after"}),
            ),
            (
                "array",
                r#"["before<|close|>argument<|sep|>[literal]after"]"#,
                serde_json::json!(["before<|close|>argument<|sep|>[literal]after"]),
            ),
        ] {
            let input = format!(
                "{}{}{}",
                TOOLS_OPEN.canonical,
                call("echo", "1", &arg("value", arg_type, value)),
                TOOLS_CLOSE.canonical
            );
            assert_all_utf8_fragmentations(
                &input,
                &[UnifiedEvent::ToolCall {
                    name: "echo".into(),
                    arguments: serde_json::json!({"value": expected}),
                }],
            );
        }
    }

    #[test]
    fn typed_string_ending_in_argument_close_marker_is_not_truncated() {
        let value = "ends with <|close|>argument<|sep|>";
        let input = format!(
            "{}{}{}",
            TOOLS_OPEN.canonical,
            call("echo", "1", &arg("value", "string", value)),
            TOOLS_CLOSE.canonical
        );
        assert_all_splits(
            &input,
            &[UnifiedEvent::ToolCall {
                name: "echo".into(),
                arguments: serde_json::json!({"value": value}),
            }],
        );
    }

    #[test]
    fn typed_string_preserves_argument_and_call_close_marker_data() {
        for value in [
            "before<|close|>argument<|sep|><|close|>call<|sep|>after",
            "before<|close|> argument <|sep|><|close|> call <|sep|>after",
        ] {
            let input = format!(
                "{}{}{}",
                TOOLS_OPEN.canonical,
                call("echo", "1", &arg("value", "string", value)),
                TOOLS_CLOSE.canonical
            );
            assert_all_utf8_fragmentations(
                &input,
                &[UnifiedEvent::ToolCall {
                    name: "echo".into(),
                    arguments: serde_json::json!({"value": value}),
                }],
            );
        }
    }

    #[test]
    fn typed_string_call_open_is_data_not_a_resync_boundary() {
        let value = format!(
            "before{}after",
            call("quoted", "8", &arg("nested", "string", "literal"))
        );
        let input = format!(
            "{}{}{}",
            TOOLS_OPEN.canonical,
            call("echo", "1", &arg("value", "string", value.as_str())),
            TOOLS_CLOSE.canonical
        );
        assert_all_utf8_fragmentations(
            &input,
            &[UnifiedEvent::ToolCall {
                name: "echo".into(),
                arguments: serde_json::json!({"value": value}),
            }],
        );
    }

    #[test]
    fn call_inside_reasoning_restores_reasoning_and_quarantines_reserved_structure() {
        let input = format!(
            "{}before{}after{}",
            THINK_OPEN.canonical,
            call("echo", "1", &arg("value", "string", "ok")),
            THINK_CLOSE.canonical
        );
        assert_all_utf8_fragmentations(
            &input,
            &[
                UnifiedEvent::Reasoning {
                    text: "before".into(),
                },
                UnifiedEvent::ToolCall {
                    name: "echo".into(),
                    arguments: serde_json::json!({"value":"ok"}),
                },
                UnifiedEvent::Reasoning {
                    text: "after".into(),
                },
            ],
        );

        let quarantined = format!(
            "{}a{}b{}c{} type=\"object\"{}{{\"x\":1}}{}d{}e{}f{}g{}h{}",
            THINK_OPEN.canonical,
            arg("hidden", "string", "secret"),
            ARG_CLOSE.canonical,
            JSON_OPEN.canonical,
            SEP,
            JSON_CLOSE.canonical,
            JSON_CLOSE.canonical,
            RESPONSE_CLOSE.canonical,
            CALL_CLOSE.canonical,
            TOOLS_CLOSE.canonical,
            THINK_CLOSE.canonical,
        );
        assert_all_utf8_fragmentations(
            &quarantined,
            &[UnifiedEvent::Reasoning {
                text: "abcdefgh".into(),
            }],
        );
    }

    #[test]
    fn missing_call_close_recovers_at_each_outer_boundary() {
        for boundary in [TOOLS_CLOSE.canonical, MESSAGE_CLOSE.canonical, END_OF_MSG] {
            let input = format!(
                "{}{OPEN}call tool=\"calc\" index=\"1\"{SEP}{}{boundary}",
                TOOLS_OPEN.canonical,
                arg("n", "number", "4")
            );
            assert_all_utf8_fragmentations(
                &input,
                &[UnifiedEvent::ToolCall {
                    name: "calc".into(),
                    arguments: serde_json::json!({"n":4}),
                }],
            );
        }
    }

    #[test]
    fn missing_call_close_recovers_at_active_return_channel_close() {
        for (close, state, leading) in [
            (
                THINK_CLOSE,
                UnifiedParserStartingState::Reasoning,
                UnifiedEvent::Reasoning {
                    text: "before Zürich".into(),
                },
            ),
            (
                RESPONSE_CLOSE,
                UnifiedParserStartingState::Response,
                UnifiedEvent::Text {
                    text: "before Zürich".into(),
                },
            ),
        ] {
            for close in close.variants() {
                let input = format!(
                    "before Zürich{OPEN}call tool=\"calc\" index=\"1\"{SEP}{}{close}after",
                    arg("n", "number", "4")
                );
                assert_all_utf8_fragmentations_from_state(
                    &input,
                    state,
                    &[
                        leading.clone(),
                        UnifiedEvent::ToolCall {
                            name: "calc".into(),
                            arguments: serde_json::json!({"n":4}),
                        },
                        UnifiedEvent::Text {
                            text: "after".into(),
                        },
                    ],
                );
            }
        }
    }

    #[test]
    fn reset_returns_the_full_uncommitted_native_envelope() {
        let input = format!(
            "{}{}{}",
            TOOLS_OPEN.canonical,
            call("calc", "1", &arg("n", "string", "Paris")),
            TOOLS_CLOSE.canonical
        );
        let call_close = input.find(CALL_CLOSE.canonical).expect("call close");
        for split in (1..=call_close).filter(|at| input.is_char_boundary(*at)) {
            let mut parser = kimi_k3_unified(&[]);
            parser.push(&input[..split]).unwrap();
            let recovered = parser.reset();
            let mut reparsed = kimi_k3_unified(&[]);
            let mut events = reparsed.push(&recovered).unwrap();
            events.extend(reparsed.push(&input[split..]).unwrap());
            events.extend(reparsed.finish().unwrap().events);
            assert_eq!(
                assemble(&events),
                vec![UnifiedEvent::ToolCall {
                    name: "calc".into(),
                    arguments: serde_json::json!({"n":"Paris"}),
                }],
                "reset at byte {split}"
            );
        }
    }

    #[test]
    fn native_stream_commits_name_arguments_index_and_id_together() {
        let header = format!(
            "{}{OPEN}call tool=\"echo\" index=\"1\"{SEP}",
            TOOLS_OPEN.canonical
        );
        let mut parser = kimi_k3_unified(&[]);
        assert!(parser.push(&header).unwrap().is_empty());
        assert_eq!(parser.tool_call_id(0), None);
        let body = arg("value", "string", "Zürich");
        assert!(parser.push(&body).unwrap().is_empty());
        assert!(parser.push(CALL_CLOSE.canonical).unwrap().is_empty());
        assert_eq!(
            parser.push(TOOLS_CLOSE.canonical).unwrap(),
            vec![crate::UnifiedParserEvent::ToolCall(ToolCallDelta {
                tool_index: 0,
                name: Some("echo".into()),
                arguments: r#"{"value":"Zürich"}"#.into(),
                complete: true,
            })]
        );
        assert_eq!(parser.tool_call_id(0), Some("echo:0"));
    }

    #[test]
    fn assistant_message_header_and_renderer_envelope_are_stripped_at_every_split() {
        let visible = "visible Zürich";
        for header in [
            r#"<|open|>message role="assistant"<|sep|>"#,
            r#"<|open|> message role="assistant" <|sep|>"#,
        ] {
            for channel in [
                format!(
                    "{}{}{}",
                    RESPONSE_OPEN.canonical, visible, RESPONSE_CLOSE.canonical
                ),
                visible.to_string(),
            ] {
                let input = format!("{header}{channel}{}{}", MESSAGE_CLOSE.canonical, END_OF_MSG);
                assert_all_utf8_fragmentations(
                    &input,
                    &[UnifiedEvent::Text {
                        text: visible.into(),
                    }],
                );
            }
        }
    }

    #[test]
    fn legacy_and_unified_are_exact_native_projections_at_every_split() {
        let input = format!(
            "{}private{}{}{}after{}",
            THINK_OPEN.canonical,
            call("echo", "1", &arg("value", "string", "Zürich")),
            ARG_CLOSE.canonical,
            "more",
            THINK_CLOSE.canonical
        );
        for split in (0..=input.len()).filter(|at| input.is_char_boundary(*at)) {
            let mut unified = kimi_k3_unified(&[]);
            let mut unified_events = unified.push(&input[..split]).unwrap();
            unified_events.extend(unified.push(&input[split..]).unwrap());
            unified_events.extend(unified.finish().unwrap().events);

            let mut legacy = crate::KimiK3ToolStreamParser::new(&[]);
            let mut legacy_result = legacy.push(&input[..split]).unwrap();
            legacy_result.append(legacy.push(&input[split..]).unwrap());
            legacy_result.append(legacy.finish().unwrap());
            assert_eq!(
                legacy_result,
                crate::ToolParseResult::from_deltas(unified_events),
                "split at byte {split}"
            );
        }
    }

    #[test]
    fn guided_native_xtml_wrapper_is_stripped_as_one_boundary() {
        let payload = r#"[{"name":"weather","arguments":{"city":"Zürich"}}]"#;
        let wrappers = [
            format!(
                "{}{OPEN}call tool=\"ignored\" index=\"1\"{SEP}{}{}{}",
                TOOLS_OPEN.canonical,
                arg("quoted", "string", "literal"),
                CALL_CLOSE.canonical,
                TOOLS_CLOSE.canonical
            ),
            format!(
                "{}{OPEN}call tool=\"ignored\" index=\"1\"{SEP}{}",
                TOOLS_OPEN.canonical,
                arg("quoted", "string", "literal")
            ),
        ];
        for wrapper in wrappers {
            let input = format!("{wrapper}{payload}");
            for split in (0..=input.len()).filter(|at| input.is_char_boundary(*at)) {
                let mut parser = kimi_k3_unified(&[]);
                parser
                    .initialize_request(UnifiedParserInit {
                        tool_output_mode: UnifiedToolOutputMode::GuidedJson { named_tool: None },
                        invalid_guided_payload: InvalidGuidedPayloadPolicy::RecoverAsText,
                        ..UnifiedParserInit::default()
                    })
                    .unwrap();
                let mut events = parser.push(&input[..split]).unwrap();
                events.extend(parser.push(&input[split..]).unwrap());
                events.extend(parser.finish().unwrap().events);
                assert_eq!(
                    assemble(&events),
                    vec![UnifiedEvent::ToolCall {
                        name: "weather".into(),
                        arguments: serde_json::json!({"city":"Zürich"}),
                    }],
                    "wrapper {wrapper:?}, split at byte {split}"
                );
            }
        }
    }

    #[test]
    fn guided_recovery_does_not_dispatch_json_after_prose() {
        let input = format!(
            "{}{OPEN}call tool=\"weather\" index=\"1\"{SEP}{}{close}{{\"x\":1}}",
            TOOLS_OPEN.canonical,
            arg("city", "string", "Paris"),
            close = CALL_CLOSE.canonical,
        );
        assert_guided_all_utf8_fragmentations(
            &input,
            UnifiedParserStartingState::None,
            &[UnifiedEvent::Text {
                text: "{\"x\":1}".into(),
            }],
        );
    }

    #[test]
    fn guided_required_recovery_rejects_non_call_json() {
        for payload in [r#"{"x":1}"#, "[]", "null", "1"] {
            let mut parser = kimi_k3_unified(&[]);
            parser
                .initialize_request(UnifiedParserInit {
                    tool_output_mode: UnifiedToolOutputMode::GuidedJson { named_tool: None },
                    invalid_guided_payload: InvalidGuidedPayloadPolicy::RecoverAsText,
                    ..UnifiedParserInit::default()
                })
                .unwrap();
            let mut events = parser.push(payload).unwrap();
            events.extend(parser.finish().unwrap().events);
            assert_eq!(
                assemble(&events),
                vec![UnifiedEvent::Text {
                    text: payload.into()
                }],
                "payload {payload:?}"
            );
        }
    }

    #[test]
    fn eof_recovers_delimiter_terminated_call_but_drops_partial_value() {
        let recovered = format!(
            "{}{OPEN}call tool=\"calc\" index=\"1\"{SEP}{}",
            TOOLS_OPEN.canonical,
            arg("n", "number", "4")
        );
        assert_eq!(
            run(&recovered, UnifiedParserStartingState::None).0,
            vec![UnifiedEvent::ToolCall {
                name: "calc".into(),
                arguments: serde_json::json!({"n":4})
            }]
        );
        let partial = format!(
            "{}{OPEN}call tool=\"calc\" index=\"1\"{SEP}{OPEN}argument key=\"n\" type=\"string\"{SEP}Par",
            TOOLS_OPEN.canonical
        );
        assert!(run(&partial, UnifiedParserStartingState::None).0.is_empty());
    }

    #[test]
    fn malformed_first_call_resynchronizes_to_a_later_complete_call() {
        let input = format!(
            "{}{OPEN}call tool=\"bad\" index=\"1\"{SEP}not-an-argument{}{}",
            TOOLS_OPEN.canonical,
            call("good", "2", &arg("x", "number", "7")),
            TOOLS_CLOSE.canonical
        );
        assert_all_splits(
            &input,
            &[UnifiedEvent::ToolCall {
                name: "good".into(),
                arguments: serde_json::json!({"x":7}),
            }],
        );
    }

    #[test]
    fn malformed_then_valid_call_has_no_ghost_name_index_or_id() {
        let malformed = format!("{OPEN}call tool=\"bad\" index=\"1\"{SEP}not-an-argument");
        let valid = call("good", "2", &arg("x", "number", "7"));
        let input = format!(
            "{}{}{}{}",
            TOOLS_OPEN.canonical, malformed, valid, TOOLS_CLOSE.canonical
        );

        for split in (0..=input.len()).filter(|at| input.is_char_boundary(*at)) {
            let mut parser = kimi_k3_unified(&[]);
            let mut events = parser.push(&input[..split]).unwrap();
            events.extend(parser.push(&input[split..]).unwrap());
            events.extend(parser.finish().unwrap().events);
            assert_eq!(
                events,
                vec![crate::UnifiedParserEvent::ToolCall(ToolCallDelta {
                    tool_index: 0,
                    name: Some("good".into()),
                    arguments: r#"{"x":7}"#.into(),
                    complete: true,
                })],
                "split at byte {split}"
            );
            assert_eq!(parser.tool_call_id(0), Some("good:1"));
            assert_eq!(parser.tool_call_id(1), None);
        }
    }

    #[test]
    fn complete_call_survives_a_truncated_later_call() {
        let input = format!(
            "{}{}{OPEN}call tool=\"later\" index=\"2\"{SEP}{OPEN}argument key=\"x\" type=\"string\"{SEP}unfinished",
            TOOLS_OPEN.canonical,
            call("first", "1", &arg("x", "number", "1"))
        );
        assert_all_splits(
            &input,
            &[UnifiedEvent::ToolCall {
                name: "first".into(),
                arguments: serde_json::json!({"x":1}),
            }],
        );
    }

    #[test]
    fn guided_required_and_named_modes_use_the_shared_router() {
        for (named_tool, input, expected_name) in [
            (
                None,
                r#"[{"name":"weather","arguments":{"city":"Paris"}}]"#,
                "weather",
            ),
            (
                Some("weather".to_string()),
                r#"{"city":"Paris"}"#,
                "weather",
            ),
        ] {
            let mut parser = kimi_k3_unified(&[]);
            parser
                .initialize_request(UnifiedParserInit {
                    starting_state: UnifiedParserStartingState::None,
                    tool_output_mode: UnifiedToolOutputMode::GuidedJson { named_tool },
                    invalid_guided_payload: InvalidGuidedPayloadPolicy::RecoverAsText,
                    ..UnifiedParserInit::default()
                })
                .unwrap();
            let mut events = parser.push(input).unwrap();
            events.extend(parser.finish().unwrap().events);
            assert_eq!(
                assemble(&events),
                vec![UnifiedEvent::ToolCall {
                    name: expected_name.into(),
                    arguments: serde_json::json!({"city":"Paris"})
                }]
            );
        }
    }

    #[test]
    fn guided_spaced_reasoning_markers_are_split_invariant() {
        let payload = r#"[{"name":"weather","arguments":{"city":"Paris"}}]"#;
        let input = format!("<|open|> think <|sep|>private<|close|> think <|sep|>{payload}");
        let expected = vec![
            UnifiedEvent::Reasoning {
                text: "private".into(),
            },
            UnifiedEvent::ToolCall {
                name: "weather".into(),
                arguments: serde_json::json!({"city":"Paris"}),
            },
        ];
        for split in (0..=input.len()).filter(|split| input.is_char_boundary(*split)) {
            let mut parser = kimi_k3_unified(&[]);
            parser
                .initialize_request(UnifiedParserInit {
                    tool_output_mode: UnifiedToolOutputMode::GuidedJson { named_tool: None },
                    invalid_guided_payload: InvalidGuidedPayloadPolicy::RecoverAsText,
                    ..UnifiedParserInit::default()
                })
                .unwrap();
            let mut events = parser.push(&input[..split]).unwrap();
            events.extend(parser.push(&input[split..]).unwrap());
            events.extend(parser.finish().unwrap().events);
            assert_eq!(assemble(&events), expected, "split at byte {split}");
        }
    }

    #[test]
    fn guided_elided_reasoning_close_is_split_invariant() {
        let payload = r#"[{"name":"weather","arguments":{"city":"Paris"}}]"#;
        let input = format!("<|open|>think<|sep|>private<|close|>think{payload}");
        let expected = vec![
            UnifiedEvent::Reasoning {
                text: "private".into(),
            },
            UnifiedEvent::ToolCall {
                name: "weather".into(),
                arguments: serde_json::json!({"city":"Paris"}),
            },
        ];
        for split in (0..=input.len()).filter(|split| input.is_char_boundary(*split)) {
            let mut parser = kimi_k3_unified(&[]);
            parser
                .initialize_request(UnifiedParserInit {
                    tool_output_mode: UnifiedToolOutputMode::GuidedJson { named_tool: None },
                    invalid_guided_payload: InvalidGuidedPayloadPolicy::RecoverAsText,
                    ..UnifiedParserInit::default()
                })
                .unwrap();
            let mut events = parser.push(&input[..split]).unwrap();
            events.extend(parser.push(&input[split..]).unwrap());
            events.extend(parser.finish().unwrap().events);
            assert_eq!(assemble(&events), expected, "split at byte {split}");
        }
    }

    #[test]
    fn guided_bare_call_prefixes_are_owned_before_any_split_can_release_them() {
        let payload = r#"[{"name":"weather","arguments":{"city":"a > b"}}]"#;
        let malformed = r#"[{"name":"weather","arguments":{"city":"#;
        for (input, expected) in [
            (
                format!("{OPEN}call tool=\"{payload}"),
                vec![UnifiedEvent::ToolCall {
                    name: "weather".into(),
                    arguments: serde_json::json!({"city":"a > b"}),
                }],
            ),
            (
                format!("{OPEN}call tool=\"{malformed}"),
                vec![UnifiedEvent::Text {
                    text: malformed.into(),
                }],
            ),
        ] {
            assert_guided_all_utf8_fragmentations(
                &input,
                UnifiedParserStartingState::None,
                &expected,
            );
        }

        let narrated = format!(
            "{}I'll call {OPEN}call tool=\"weather{}{payload}",
            THINK_OPEN.canonical, THINK_CLOSE.canonical
        );
        assert_guided_all_utf8_fragmentations(
            &narrated,
            UnifiedParserStartingState::None,
            &[
                UnifiedEvent::Reasoning {
                    text: "I'll call weather".into(),
                },
                UnifiedEvent::ToolCall {
                    name: "weather".into(),
                    arguments: serde_json::json!({"city":"a > b"}),
                },
            ],
        );

        let before_reasoning = format!(
            "{OPEN}call tool=\"{}secret{}{payload}",
            THINK_OPEN.canonical, THINK_CLOSE.canonical
        );
        assert_guided_all_utf8_fragmentations(
            &before_reasoning,
            UnifiedParserStartingState::None,
            &[
                UnifiedEvent::Reasoning {
                    text: "secret".into(),
                },
                UnifiedEvent::ToolCall {
                    name: "weather".into(),
                    arguments: serde_json::json!({"city":"a > b"}),
                },
            ],
        );
    }

    #[test]
    fn guided_incomplete_and_malformed_header_work_scales_linearly() {
        fn work(size: usize, terminated: bool) -> usize {
            GUIDED_PREFIX_EXAMINED_BYTES.with(|examined| examined.set(0));
            let suffix = if terminated { "\"" } else { "" };
            let input = format!("{OPEN}call tool=\"{}{suffix}", "x".repeat(size));
            let mut parser = kimi_k3_unified(&[]);
            parser
                .initialize_request(UnifiedParserInit {
                    tool_output_mode: UnifiedToolOutputMode::GuidedJson { named_tool: None },
                    invalid_guided_payload: InvalidGuidedPayloadPolicy::RecoverAsText,
                    ..UnifiedParserInit::default()
                })
                .expect("initialize");
            for ch in input.chars() {
                parser.push(&ch.to_string()).expect("push character");
            }
            parser.finish().expect("finish");
            GUIDED_PREFIX_EXAMINED_BYTES.with(std::cell::Cell::get)
        }

        for terminated in [false, true] {
            let small = work(4_096, terminated);
            let large = work(8_192, terminated);
            println!("K3 guided header terminated={terminated}: {small} -> {large}");
            assert!(
                large <= small * 2 + 256,
                "guided header scan work grew faster than linearly: {small} -> {large}"
            );
        }
    }

    #[test]
    fn guided_response_prefill_keeps_quoted_channel_markup_literal() {
        let payload = r#"[{"name":"weather","arguments":{"city":"Paris"}}]"#;
        let quote = format!(
            "I mean {}self literal{}",
            THINK_OPEN.canonical, THINK_CLOSE.canonical
        );
        for input in [format!("{quote}{payload}"), format!("{payload}{quote}")] {
            let expected = if input.starts_with('[') {
                vec![
                    UnifiedEvent::ToolCall {
                        name: "weather".into(),
                        arguments: serde_json::json!({"city":"Paris"}),
                    },
                    UnifiedEvent::Text {
                        text: quote.clone(),
                    },
                ]
            } else {
                vec![
                    UnifiedEvent::Text {
                        text: quote.clone(),
                    },
                    UnifiedEvent::ToolCall {
                        name: "weather".into(),
                        arguments: serde_json::json!({"city":"Paris"}),
                    },
                ]
            };
            assert_guided_all_utf8_fragmentations(
                &input,
                UnifiedParserStartingState::Response,
                &expected,
            );
        }
    }

    #[test]
    fn reset_restarts_channels_indices_and_lifecycle() {
        let first = format!(
            "{}{}{}",
            TOOLS_OPEN.canonical,
            call("first", "1", ""),
            TOOLS_CLOSE.canonical
        );
        let second = format!(
            "{}{}{}",
            TOOLS_OPEN.canonical,
            call("second", "1", ""),
            TOOLS_CLOSE.canonical
        );
        let mut parser = kimi_k3_unified(&[]);
        assert_eq!(parser.push(&first).unwrap().len(), 1);
        assert!(parser.finish().unwrap().events.is_empty());
        assert!(parser.push("later").is_err());
        assert_eq!(parser.reset(), "");
        let events = parser.push(&second).unwrap();
        let crate::UnifiedParserEvent::ToolCall(call) = &events[0] else {
            panic!("expected a tool call")
        };
        assert_eq!(call.tool_index, 0);
        assert_eq!(parser.tool_call_id(0), Some("second:0"));
    }
}
