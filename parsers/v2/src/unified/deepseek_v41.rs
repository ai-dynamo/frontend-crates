// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Context;
use serde_json::{Map, Value};

use crate::tool_calling::scan::{
    BareRecoveryLatch, InvokeBoundary, InvokeBoundaryFactory, InvokeEmitter, ReasoningSpec,
    WrappedBlockScanner, WrappedBlockSpec,
};
use crate::tool_calling::traits::{Tool, ToolCallDelta};
use crate::unified::{
    GuidedInvokePrefix, GuidedInvokePrefixContext, GuidedRouted, JsonPrefixState, ScannerUnified,
    UnifiedParser,
};

pub(crate) const BLOCK_START: &str = "<｜DSML｜ calls>";
pub(crate) const BLOCK_END: &str = "</｜DSML｜ calls>";
pub(crate) const INVOKE_START: &str = "<｜DSML｜ invoke name=\"";
pub(crate) const INVOKE_END: &str = "</｜DSML｜ invoke>";
pub(crate) const PARAMETER_START: &str = "<｜DSML｜ parameter name=\"";
pub(crate) const PARAMETER_END: &str = "</｜DSML｜ parameter>";

pub(crate) fn deepseek_v41_unified(_tools: &[Tool]) -> Box<dyn UnifiedParser> {
    let spec = WrappedBlockSpec {
        family: "deepseek_v41",
        block_starts: vec![BLOCK_START.into()],
        block_ends: vec![BLOCK_END.into()],
        invoke_start: INVOKE_START.into(),
        invoke_end: INVOKE_END.into(),
        orphan_markers: vec![BLOCK_END.into()],
        holdback_markers: vec![BLOCK_START.into(), BLOCK_END.into(), INVOKE_START.into()],
        bare_recovery_latch: BareRecoveryLatch::Set,
        invoke_boundary_factory: Some(InvokeBoundaryFactory::custom(invocation_boundary)),
        preserve_special_tokens: true,
        ..Default::default()
    };
    let scanner = WrappedBlockScanner::new(spec, DeepSeekV41).with_reasoning(ReasoningSpec {
        start: "<think>",
        end: "</think>",
        preserve_special_tokens: true,
        ..Default::default()
    });
    Box::new(GuidedRouted::new(ScannerUnified::new(scanner)))
}

fn parameter_header(text: &str) -> Option<(&str, bool, &str)> {
    let (name, rest) = text.strip_prefix(PARAMETER_START)?.split_once('"')?;
    let (string, value) = rest.strip_prefix(" string=\"")?.split_once("\">")?;
    let string = match string {
        "true" => true,
        "false" => false,
        _ => return None,
    };
    Some((name, string, value))
}

#[cfg(test)]
std::thread_local! {
    static BOUNDARY_EXAMINED_BYTES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn count_boundary_bytes(bytes: usize) {
    #[cfg(test)]
    BOUNDARY_EXAMINED_BYTES.with(|examined| examined.set(examined.get() + bytes));
    #[cfg(not(test))]
    let _ = bytes;
}

fn find_from(text: &str, start: usize, marker: &str) -> Option<usize> {
    let suffix = &text[start..];
    count_boundary_bytes(suffix.len());
    suffix.find(marker).map(|at| start + at)
}

fn find_payload_from(text: &str, start: usize) -> Option<usize> {
    let suffix = &text[start..];
    count_boundary_bytes(suffix.len());
    suffix.find(['{', '[']).map(|at| start + at)
}

fn next_scan_start(text: &str, marker_len: usize) -> usize {
    let mut start = text.len().saturating_sub(marker_len.saturating_sub(1));
    while !text.is_char_boundary(start) {
        start -= 1;
    }
    start
}

/// Close selection for `string="true"` parameter values.
///
/// The grammar has no escape, so a value that quotes DSML markup can contain a
/// literal `</｜DSML｜ parameter>`. Closes are chosen so that the whole body
/// parses: well-formed parameters with distinct names and valid
/// `string="false"` JSON, each value ending at the earliest close that still
/// lets the rest parse. When the first-close rule parses a body, this selects
/// the same closes.
///
/// A close followed by a malformed parameter header is never skipped: that
/// header is a model error, not quoted data, so the call stays rejected as
/// under the first-close rule instead of losing the parameter silently.
struct ValueCloses<'a> {
    body: &'a str,
    /// Offset just past each `PARAMETER_END` in `body`, ascending.
    ends: Vec<usize>,
    /// For each close, whether the text after it parses as a run of
    /// parameters when their names are not compared.
    parses: Vec<bool>,
    /// For each index into `ends`, the first index at or after it whose close
    /// is followed by text that parses or by a malformed parameter header.
    /// One extra trailing `None`.
    next: Vec<Option<usize>>,
    /// `(value start, index into ends)` for each parameter of the selected
    /// parse of the whole body, ascending; empty when the body has no parse.
    selected: Vec<(usize, usize)>,
}

impl<'a> ValueCloses<'a> {
    fn new(body: &'a str) -> Self {
        let ends: Vec<usize> = body
            .match_indices(PARAMETER_END)
            .map(|(at, _)| at + PARAMETER_END.len())
            .collect();
        let mut closes = Self {
            body,
            parses: vec![false; ends.len()],
            next: vec![None; ends.len() + 1],
            selected: Vec::new(),
            ends,
        };
        // A remainder only depends on closes after it, so fill from the end.
        for index in (0..closes.ends.len()).rev() {
            let at = closes.ends[index];
            closes.parses[index] = closes.parses_from(at);
            closes.next[index] = if closes.parses[index] || closes.malformed_header_at(at) {
                Some(index)
            } else {
                closes.next[index + 1]
            };
        }
        closes.selected = closes.select().unwrap_or_default();
        closes
    }

    /// Whether `body[at..]` starts, after whitespace, with a parameter opener
    /// whose header does not parse.
    fn malformed_header_at(&self, at: usize) -> bool {
        let rest = self.body[at..].trim_start();
        rest.starts_with(PARAMETER_START) && parameter_header(rest).is_none()
    }

    /// Index into `ends` of the first close at or after `value_start`.
    fn first_close(&self, value_start: usize) -> usize {
        self.ends
            .partition_point(|&end| end < value_start + PARAMETER_END.len())
    }

    /// The parameter opening `body[at..]` after whitespace: its name, whether
    /// it is a string, where its value starts, and the earliest close its
    /// value can end at.
    fn parameter_at(&self, at: usize) -> Option<(&'a str, bool, usize, usize)> {
        let (name, string, value) = parameter_header(self.body[at..].trim_start())?;
        let value_start = self.body.len() - value.len();
        let first = self.first_close(value_start);
        let close = if string { self.next[first]? } else { first };
        if !*self.parses.get(close)? {
            return None;
        }
        if !string {
            let value_end = self.ends[close] - PARAMETER_END.len();
            serde_json::from_str::<Value>(&self.body[value_start..value_end]).ok()?;
        }
        Some((name, string, value_start, close))
    }

    /// Whether `body[at..]` parses as a run of parameters with valid
    /// `string="false"` values, without comparing their names.
    fn parses_from(&self, at: usize) -> bool {
        self.body[at..].trim_start().is_empty() || self.parameter_at(at).is_some()
    }

    /// The parse of the whole body to use, as `(value start, close)` pairs.
    ///
    /// A parse gives every parameter a distinct name. Parses are visited with
    /// each string value, in order, ending at its earliest possible close, so
    /// the first one visited is the first-close parse whenever that exists,
    /// and it is then returned as is. Otherwise the parse whose values quote
    /// the most balanced markup wins (see `unbalanced_markers`), the earliest
    /// visited on a tie: a quoted example is normally complete, so its closes
    /// pair with its own openers rather than with the call's.
    ///
    /// Only a repeated name or a rejected parse makes the search back up, so
    /// an ordinary body takes one step per parameter. The step and parse
    /// limits keep crafted input linear; past them the best parse found so
    /// far stands.
    fn select(&self) -> Option<Vec<(usize, usize)>> {
        let mut steps = 64 + 4 * self.ends.len();
        let mut parses_left = MAX_CLOSE_CANDIDATES;
        let mut best: Option<(usize, Vec<(usize, usize)>)> = None;
        // Parameters taken so far: name, whether it is a string, value start,
        // chosen close.
        let mut taken: Vec<(&'a str, bool, usize, usize)> = Vec::new();
        let mut at = 0;
        'search: while steps > 0 {
            steps -= 1;
            let mut advanced = false;
            if self.body[at..].trim_start().is_empty() {
                let parse: Vec<(usize, usize)> = taken
                    .iter()
                    .map(|&(_, _, value_start, close)| (value_start, close))
                    .collect();
                let first_closes = parse
                    .iter()
                    .all(|&(value_start, close)| close == self.first_close(value_start));
                let cost = if first_closes {
                    Some(0)
                } else {
                    parse
                        .iter()
                        .map(|&(value_start, close)| self.unbalanced_markers(value_start, close))
                        .sum()
                };
                if let Some(cost) = cost
                    && best.as_ref().is_none_or(|(least, _)| cost < *least)
                {
                    best = Some((cost, parse));
                }
                parses_left -= 1;
                if cost == Some(0) || parses_left == 0 {
                    break;
                }
            } else if let Some(parameter) = self
                .parameter_at(at)
                .filter(|&(name, ..)| taken.iter().all(|&(seen, ..)| seen != name))
            {
                at = self.ends[parameter.3];
                taken.push(parameter);
                advanced = true;
            }
            if advanced {
                continue;
            }
            // Move the latest string value that has one to its next close.
            while let Some((name, string, value_start, close)) = taken.pop() {
                let later = self.next[close + 1].filter(|&later| self.parses[later]);
                if let Some(later) = later.filter(|_| string) {
                    at = self.ends[later];
                    taken.push((name, string, value_start, later));
                    continue 'search;
                }
            }
            break;
        }
        best.map(|(_, parse)| parse)
    }

    /// Number of parameter openers and closes in the value from `value_start`
    /// to `close` that do not pair up inside it.
    ///
    /// `None` when the value holds both a close with no opener of its own and,
    /// after it, an opener it never closes. Such a value runs across the end
    /// of one real parameter and the start of another, as with a repeated
    /// parameter; it is not quoted data.
    fn unbalanced_markers(&self, value_start: usize, close: usize) -> Option<usize> {
        let value = &self.body[value_start..self.ends[close] - PARAMETER_END.len()];
        let mut opens = value
            .match_indices(PARAMETER_START)
            .map(|(at, _)| at)
            .peekable();
        let (mut open, mut unpaired) = (0usize, 0usize);
        for (end, _) in value.match_indices(PARAMETER_END) {
            while opens.next_if(|&at| at < end).is_some() {
                open += 1;
            }
            match open.checked_sub(1) {
                Some(left) => open = left,
                None => unpaired += 1,
            }
        }
        // Every opener still open came after the last unpaired close.
        let open = open + opens.count();
        (open == 0 || unpaired == 0).then_some(open + unpaired)
    }

    /// Whether the whole body parses.
    fn is_complete(&self) -> bool {
        self.body.trim().is_empty() || !self.selected.is_empty()
    }

    /// Start of the close ending the value at `value_start`: the close the
    /// selected parse gives it, else its first close, so a body without a
    /// parse is rejected exactly as under the first-close rule.
    fn value_end(&self, value_start: usize) -> Option<usize> {
        let chosen = match self
            .selected
            .binary_search_by_key(&value_start, |&(start, _)| start)
        {
            Ok(found) => self.selected[found].1,
            Err(_) => self.first_close(value_start),
        };
        self.ends.get(chosen).map(|end| end - PARAMETER_END.len())
    }
}

/// Name and body of a complete invocation, without validating the body.
fn invocation_parts(invoke: &str) -> Option<(&str, &str)> {
    let (name, body) = invoke.strip_prefix(INVOKE_START)?.split_once("\">")?;
    Some((name, body.strip_suffix(INVOKE_END)?))
}

/// Whether `invoke` is one structurally well-formed invocation.
fn invocation_is_well_formed(invoke: &str) -> bool {
    let Some((name, body)) = invocation_parts(invoke) else {
        return false;
    };
    if name.is_empty() {
        return false;
    }
    if body.trim_start().starts_with('{') {
        return serde_json::from_str::<Map<String, Value>>(body).is_ok();
    }
    ValueCloses::new(body).is_complete()
}

/// Invocation closes tried per decision when resolving a quoted close. Each
/// try is a linear pass, so the cap keeps resolution linear in the stream
/// even when the tail is crafted to hold many closes; a real quoted example
/// needs one or two.
const MAX_CLOSE_CANDIDATES: usize = 16;

/// Offset just past the calls block when `text[at..]` holds zero or more
/// well-formed invocations followed by its close.
fn block_end_after(text: &str, mut at: usize) -> Option<usize> {
    loop {
        at = text.len() - text[at..].trim_start().len();
        let rest = &text[at..];
        if rest.starts_with(BLOCK_END) {
            return Some(at + BLOCK_END.len());
        }
        if !rest.starts_with(INVOKE_START) {
            return None;
        }
        at += rest
            .match_indices(INVOKE_END)
            .map(|(close, _)| close + INVOKE_END.len())
            .take(MAX_CLOSE_CANDIDATES)
            .find(|&end| invocation_is_well_formed(&rest[..end]))?;
    }
}

/// Whether no DSML closer is stranded in `text[at..]`, i.e. every closer sits
/// inside a later calls block. One linear pass.
fn trailing_is_coherent(text: &str, mut at: usize) -> bool {
    loop {
        let rest = &text[at..];
        let Some(close) = [PARAMETER_END, INVOKE_END, BLOCK_END]
            .into_iter()
            .filter_map(|marker| rest.find(marker))
            .min()
        else {
            return true;
        };
        let Some(open) = rest.find(BLOCK_START).filter(|&open| open < close) else {
            return false;
        };
        let after_open = at + open + BLOCK_START.len();
        let Some(end) = text[after_open..].find(BLOCK_END) else {
            return true;
        };
        at = after_open + end + BLOCK_END.len();
    }
}

/// Invocation end to commit for a call whose string value quoted a parameter
/// opener, given the whole remaining stream.
///
/// `first` is the end the first-close rule found. It stands unless it strands
/// DSML closers after its calls block (the signature of a literal close taken
/// as structure) and a later invocation close yields a well-formed invocation
/// and calls block with nothing stranded after it.
fn coherent_invocation_end(text: &str, first: usize) -> usize {
    let coherent = |end: usize| {
        invocation_is_well_formed(&text[..end])
            && block_end_after(text, end).is_some_and(|block| trailing_is_coherent(text, block))
    };
    if coherent(first) {
        return first;
    }
    text.match_indices(INVOKE_END)
        .map(|(at, _)| at + INVOKE_END.len())
        .filter(|&end| end > first)
        .take(MAX_CLOSE_CANDIDATES)
        .find(|&end| coherent(end))
        .unwrap_or(first)
}

#[derive(Default)]
enum InvocationPosition {
    #[default]
    Header,
    Body,
    JsonValue,
    BetweenParameters,
    ParameterHeader {
        start: usize,
        scan_from: usize,
    },
    ParameterValue {
        start: usize,
        value_start: usize,
        string: bool,
        scan_from: usize,
    },
    InvalidParameter {
        start: usize,
    },
}

#[derive(Default)]
struct DeepSeekV41InvocationBoundary {
    position: InvocationPosition,
    scan_from: usize,
    /// A string value, read up to its first close, contains a parameter
    /// opener, so that close may be quoted data rather than structure.
    quoted_parameter_open: bool,
    /// First-close end held back until EOF while `quoted_parameter_open`.
    deferred_end: Option<usize>,
    json: JsonPrefixState,
    guided_prefix_scan_from: usize,
    guided_prefix_payload_at: Option<usize>,
    guided_prefix_header_end: Option<usize>,
}

impl DeepSeekV41InvocationBoundary {
    fn malformed_end(text: &str, start: usize, flush: bool) -> Option<usize> {
        flush
            .then(|| find_from(text, start, INVOKE_END))
            .flatten()
            .map(|at| at + INVOKE_END.len())
    }
}

impl InvokeBoundary for DeepSeekV41InvocationBoundary {
    fn owns_guided_prefix(&self) -> bool {
        true
    }

    fn guided_prefix_append(
        &mut self,
        candidate: &str,
        append: &str,
        context: GuidedInvokePrefixContext,
    ) -> Option<GuidedInvokePrefix> {
        let header = candidate.strip_prefix(INVOKE_START)?;
        if !context.outside_reasoning || context.followed_by_competing_marker {
            return Some(GuidedInvokePrefix::Strip(INVOKE_START.len()));
        }

        let append_start = candidate.len() - append.len();
        let scan_from = self
            .guided_prefix_scan_from
            .max(append_start.saturating_sub(INVOKE_START.len()));
        if self.guided_prefix_payload_at.is_none() {
            self.guided_prefix_payload_at = find_payload_from(header, scan_from);
        }
        if self.guided_prefix_header_end.is_none() {
            self.guided_prefix_header_end = find_from(header, scan_from, ">");
        }
        self.guided_prefix_scan_from = header.len();

        if self.guided_prefix_header_end.is_some_and(|header_end| {
            self.guided_prefix_payload_at
                .is_none_or(|payload| header_end < payload)
        }) {
            return Some(GuidedInvokePrefix::NoMatch);
        }
        if let Some(payload_at) = self.guided_prefix_payload_at {
            // A bare DSML header has no closing quote or `>` before guided JSON.
            // Stop at the payload opener: the first quote in a JSON key is payload
            // data, not the header terminator.
            return Some(if context.payload_is_empty {
                GuidedInvokePrefix::Match(INVOKE_START.len() + payload_at)
            } else {
                GuidedInvokePrefix::Strip(INVOKE_START.len() + payload_at)
            });
        }
        Some(GuidedInvokePrefix::Pending)
    }

    fn end_append(
        &mut self,
        candidate: &str,
        _append: &str,
        flush: bool,
        _tool_index: usize,
    ) -> Option<usize> {
        // As with the INVOKE_END/BLOCK_END lookahead in `tool_calling::dsml`,
        // a close inside a value is only a candidate. A call that quoted a
        // parameter opener is held to EOF, where the whole stream decides which
        // close is structural; every other call commits at its first close,
        // exactly as before.
        let first = match self.deferred_end {
            Some(end) => end,
            None => self.first_close_end(candidate, flush)?,
        };
        if !self.quoted_parameter_open {
            return Some(first);
        }
        if !flush {
            self.deferred_end = Some(first);
            return None;
        }
        Some(coherent_invocation_end(candidate, first))
    }

    fn opens(&self, _text: &str, _at: usize) -> bool {
        true
    }

    fn holdback(&self, _text: &str) -> usize {
        0
    }

    fn resync(&mut self, _text: &str, _flush: bool, _tool_index: usize) -> Option<usize> {
        None
    }

    fn reset(&mut self) {
        *self = Self::default();
    }
}

impl DeepSeekV41InvocationBoundary {
    /// Invocation end under the first-close rule, scanning appended bytes only.
    fn first_close_end(&mut self, candidate: &str, flush: bool) -> Option<usize> {
        loop {
            match self.position {
                InvocationPosition::Header => {
                    let Some(end) = find_from(candidate, self.scan_from, "\">") else {
                        self.scan_from = next_scan_start(candidate, 2);
                        return None;
                    };
                    self.scan_from = end + 2;
                    self.position = InvocationPosition::Body;
                }
                InvocationPosition::Body => {
                    let tail = &candidate[self.scan_from..];
                    let trimmed = tail.trim_start();
                    count_boundary_bytes(
                        tail.len() - trimmed.len() + usize::from(!trimmed.is_empty()),
                    );
                    self.scan_from = candidate.len() - trimmed.len();
                    let first = trimmed.chars().next()?;
                    if first == '{' {
                        self.json = JsonPrefixState::new(first);
                        self.scan_from += 1;
                        self.position = InvocationPosition::JsonValue;
                    } else {
                        self.position = InvocationPosition::BetweenParameters;
                    }
                }
                InvocationPosition::JsonValue => {
                    for ch in candidate[self.scan_from..].chars() {
                        count_boundary_bytes(ch.len_utf8());
                        self.scan_from += ch.len_utf8();
                        self.json.consume(ch);
                        if self.json.invalid || self.json.complete {
                            break;
                        }
                    }
                    if self.json.invalid {
                        self.position = InvocationPosition::InvalidParameter { start: 0 };
                    } else if self.json.complete {
                        self.position = InvocationPosition::BetweenParameters;
                    } else {
                        return None;
                    }
                }
                InvocationPosition::BetweenParameters => {
                    let close = find_from(candidate, self.scan_from, INVOKE_END);
                    let parameter = find_from(candidate, self.scan_from, PARAMETER_START);
                    match (close, parameter) {
                        (Some(close), Some(parameter)) if parameter <= close => {
                            self.position = InvocationPosition::ParameterHeader {
                                start: parameter,
                                scan_from: parameter + PARAMETER_START.len(),
                            };
                        }
                        (Some(close), _) => return Some(close + INVOKE_END.len()),
                        (None, Some(parameter)) => {
                            self.position = InvocationPosition::ParameterHeader {
                                start: parameter,
                                scan_from: parameter + PARAMETER_START.len(),
                            };
                        }
                        (None, None) => {
                            self.scan_from = next_scan_start(
                                candidate,
                                INVOKE_END.len().max(PARAMETER_START.len()),
                            );
                            return None;
                        }
                    }
                }
                InvocationPosition::ParameterHeader { start, scan_from } => {
                    let Some(header_end) = find_from(candidate, scan_from, "\">") else {
                        if flush {
                            return Self::malformed_end(candidate, start, true);
                        }
                        self.position = InvocationPosition::ParameterHeader {
                            start,
                            scan_from: next_scan_start(candidate, 2),
                        };
                        return None;
                    };
                    let Some((_, string, value)) = parameter_header(&candidate[start..]) else {
                        self.position = InvocationPosition::InvalidParameter { start };
                        continue;
                    };
                    let value_start = candidate.len() - value.len();
                    debug_assert!(value_start >= header_end + 2);
                    self.position = InvocationPosition::ParameterValue {
                        start,
                        value_start,
                        string,
                        scan_from: value_start,
                    };
                }
                InvocationPosition::ParameterValue {
                    start,
                    value_start,
                    string,
                    scan_from,
                } => {
                    let Some(value_end) = find_from(candidate, scan_from, PARAMETER_END) else {
                        if flush {
                            return Self::malformed_end(candidate, start, true);
                        }
                        self.position = InvocationPosition::ParameterValue {
                            start,
                            value_start,
                            string,
                            scan_from: next_scan_start(candidate, PARAMETER_END.len()),
                        };
                        return None;
                    };
                    if string {
                        let value = &candidate[value_start..value_end];
                        count_boundary_bytes(value.len());
                        self.quoted_parameter_open |= value.contains(PARAMETER_START);
                    }
                    self.scan_from = value_end + PARAMETER_END.len();
                    self.position = InvocationPosition::BetweenParameters;
                }
                InvocationPosition::InvalidParameter { start } => {
                    return Self::malformed_end(candidate, start, flush);
                }
            }
        }
    }
}

pub(crate) fn invocation_boundary() -> Box<dyn InvokeBoundary> {
    Box::new(DeepSeekV41InvocationBoundary::default())
}

pub(crate) struct DeepSeekV41;

impl InvokeEmitter for DeepSeekV41 {
    fn parse_invoke(
        &mut self,
        invoke: &str,
        tool_index: usize,
    ) -> anyhow::Result<Option<ToolCallDelta>> {
        let (name, body) = invoke
            .strip_prefix(INVOKE_START)
            .and_then(|text| text.split_once("\">"))
            .context("invalid DeepSeek V4.1 invocation header")?;
        anyhow::ensure!(!name.is_empty(), "empty DeepSeek V4.1 tool name");
        let mut body = body
            .strip_suffix(INVOKE_END)
            .context("incomplete DeepSeek V4.1 invocation")?;
        let mut arguments = if body.trim_start().starts_with('{') {
            let arguments = serde_json::from_str::<Map<String, Value>>(body)
                .context("invalid DeepSeek V4.1 JSON arguments")?;
            body = "";
            arguments
        } else {
            Map::new()
        };
        let closes = ValueCloses::new(body);
        while !body.trim().is_empty() {
            let (name, string, value) = parameter_header(body.trim_start())
                .context("invalid DeepSeek V4.1 parameter header")?;
            let value_start = closes.body.len() - value.len();
            let value_end = closes
                .value_end(value_start)
                .context("incomplete DeepSeek V4.1 parameter")?;
            let raw = &closes.body[value_start..value_end];
            let remainder = &closes.body[value_end + PARAMETER_END.len()..];
            let value = if string {
                Value::String(raw.to_string())
            } else {
                serde_json::from_str(raw)?
            };
            anyhow::ensure!(
                arguments.insert(name.to_string(), value).is_none(),
                "duplicate DeepSeek V4.1 parameter {name:?}"
            );
            body = remainder;
        }
        Ok(Some(ToolCallDelta {
            tool_index,
            name: Some(name.to_string()),
            arguments: serde_json::to_string(&arguments)?,
            complete: true,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unified::{
        InvalidGuidedPayloadPolicy, UnifiedEvent, UnifiedParserExt, UnifiedParserInit,
        UnifiedParserOutput, UnifiedParserStartingState, UnifiedToolOutputMode,
        create_unified_parser_for_family,
    };

    fn parse_chunks(input: &str, split: usize, init: UnifiedParserInit) -> UnifiedParserOutput {
        let mut parser = create_unified_parser_for_family("deepseek_v41", &[]).unwrap();
        assert!(parser.preserve_special_tokens());
        parser.initialize_request(init).unwrap();
        let mut output = UnifiedParserOutput::default();
        parser.parse_into(&input[..split], &mut output).unwrap();
        parser.parse_into(&input[split..], &mut output).unwrap();
        output.append(&mut parser.finish().unwrap());
        output
    }

    fn assert_every_split_with_init(
        input: &str,
        init: UnifiedParserInit,
        expected: Vec<UnifiedEvent>,
    ) {
        for split in (0..=input.len()).filter(|&i| input.is_char_boundary(i)) {
            assert_eq!(
                parse_chunks(input, split, init.clone()).assembled(),
                expected,
                "split {split}"
            );
        }
        let mut parser = deepseek_v41_unified(&[]);
        parser.initialize_request(init).unwrap();
        let mut output = UnifiedParserOutput::default();
        for ch in input.chars() {
            parser
                .parse_into(ch.encode_utf8(&mut [0; 4]), &mut output)
                .unwrap();
        }
        output.append(&mut parser.finish().unwrap());
        assert_eq!(output.assembled(), expected, "one character at a time");
    }

    fn assert_every_split(
        input: &str,
        state: UnifiedParserStartingState,
        expected: Vec<UnifiedEvent>,
    ) {
        assert_every_split_with_init(
            input,
            UnifiedParserInit {
                starting_state: state,
                ..Default::default()
            },
            expected,
        );
    }

    #[test]
    fn deepseek_v41_registration() {
        assert_every_split(
            "hello 世界",
            UnifiedParserStartingState::None,
            vec![UnifiedEvent::Text {
                text: "hello 世界".into(),
            }],
        );
    }

    #[test]
    fn reasoning_transition_and_multiple_calls() {
        let input = concat!(
            "Check the tools.\n</think>\n\n<｜DSML｜ calls>\n",
            "<｜DSML｜ invoke name=\"weather\">\n",
            "<｜DSML｜ parameter name=\"city\" string=\"true\">東京 &amp; \"Paris\"\\\n</｜DSML｜ parameter>\n",
            "<｜DSML｜ parameter name=\"days\" string=\"false\">3</｜DSML｜ parameter>\n",
            "<｜DSML｜ parameter name=\"options\" string=\"false\">{\"x\":[true,null]}</｜DSML｜ parameter>\n",
            "</｜DSML｜ invoke>\n<｜DSML｜ invoke name=\"done\">\n</｜DSML｜ invoke>\n",
            "</｜DSML｜ calls>",
        );
        assert_every_split(
            input,
            UnifiedParserStartingState::Reasoning,
            vec![
                UnifiedEvent::Reasoning {
                    text: "Check the tools.\n".into(),
                },
                UnifiedEvent::Text {
                    text: "\n\n".into(),
                },
                UnifiedEvent::ToolCall {
                    name: "weather".into(),
                    arguments: serde_json::json!({
                        "city": "東京 &amp; \"Paris\"\\\n", "days": 3, "options": {"x": [true, null]}
                    }),
                },
                UnifiedEvent::ToolCall {
                    name: "done".into(),
                    arguments: serde_json::json!({}),
                },
            ],
        );
    }

    #[test]
    fn json_invocation_bodies_preserve_values_and_literal_markers() {
        let arguments = serde_json::json!({
            "value": " café 🐈 </｜DSML｜ invoke> </｜DSML｜ calls> \\\"\n",
            "nested": {"values": [true, null, 42, -1.25e3]},
        });
        let input = format!(
            "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"inspect\">\n{arguments}\n</｜DSML｜ invoke>\n<｜DSML｜ invoke name=\"done\">{{}}</｜DSML｜ invoke>\n</｜DSML｜ calls>"
        );
        assert_every_split(
            &input,
            UnifiedParserStartingState::None,
            vec![
                UnifiedEvent::ToolCall {
                    name: "inspect".into(),
                    arguments,
                },
                UnifiedEvent::ToolCall {
                    name: "done".into(),
                    arguments: serde_json::json!({}),
                },
            ],
        );
    }

    #[test]
    fn tool_markup_inside_string_is_data() {
        let input = "<｜DSML｜ calls><｜DSML｜ invoke name=\"run\"><｜DSML｜ parameter name=\"text\" string=\"true\"><think>quoted</think> <｜DSML｜ calls></｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>";
        assert_every_split(
            input,
            UnifiedParserStartingState::None,
            vec![UnifiedEvent::ToolCall {
                name: "run".into(),
                arguments: serde_json::json!({"text": "<think>quoted</think> <｜DSML｜ calls>"}),
            }],
        );
    }

    #[test]
    fn closing_markers_and_whitespace_inside_strings_are_data() {
        let value = " X</｜DSML｜ calls>Y</｜DSML｜ invoke>Z\n ";
        let input = format!(
            "<｜DSML｜ calls><｜DSML｜ invoke name=\"run\"><｜DSML｜ parameter name=\"user name\" string=\"true\">{value}</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>"
        );
        assert_every_split(
            &input,
            UnifiedParserStartingState::None,
            vec![UnifiedEvent::ToolCall {
                name: "run".into(),
                arguments: serde_json::json!({"user name":value}),
            }],
        );
    }

    #[test]
    fn incomplete_arguments_do_not_emit_calls() {
        for input in [
            "<｜DSML｜ calls><｜DSML｜ invoke name=\"run\"><｜DSML｜ parameter name=\"text\" string=\"true\">unfinished",
            "<｜DSML｜ calls><｜DSML｜ invoke name=\"run\">{\"text\":\"unfinished </｜DSML｜ invoke>",
            "<｜DSML｜ calls><｜DSML｜ invoke name=\"run\">{\"text\":\"complete JSON, missing close\"}",
        ] {
            assert_every_split(input, UnifiedParserStartingState::None, vec![]);
        }
    }

    #[test]
    fn partial_plain_markers_and_open_reasoning_survive_eof() {
        for input in ["hello <", "hello <｜DS", "ordinary text\n", "你好"] {
            assert_every_split(
                input,
                UnifiedParserStartingState::None,
                vec![UnifiedEvent::Text { text: input.into() }],
            );
            assert_every_split(
                input,
                UnifiedParserStartingState::Reasoning,
                vec![UnifiedEvent::Reasoning { text: input.into() }],
            );
        }
    }

    #[test]
    fn calls_stream_at_each_invocation_close() {
        let mut parser = deepseek_v41_unified(&[]);
        assert!(parser.push("<｜DSML｜ calls><｜DSML｜ invoke name=\"run\"><｜DSML｜ parameter name=\"text\" string=\"true\">hello").unwrap().is_empty());
        let events = parser
            .push("</｜DSML｜ parameter></｜DSML｜ invoke>")
            .unwrap();
        let output: UnifiedParserOutput = events.into_iter().collect();
        assert_eq!(
            output.assembled(),
            vec![UnifiedEvent::ToolCall {
                name: "run".into(),
                arguments: serde_json::json!({"text":"hello"}),
            }]
        );
        assert!(parser.push("</｜DSML｜ calls>").unwrap().is_empty());
    }

    #[test]
    fn invocation_requires_its_complete_closing_tag() {
        for suffix in ["", " invoke", " banana>"] {
            let input = format!("<｜DSML｜ calls><｜DSML｜ invoke name=\"run\"></｜DSML｜{suffix}");
            let mut parser = deepseek_v41_unified(&[]);
            assert!(parser.push(&input).unwrap().is_empty());
            assert!(parser.finish().unwrap().events.is_empty());
        }
    }

    #[test]
    fn invocation_boundary_scans_large_streamed_parameters_linearly() {
        let mut parser = deepseek_v41_unified(&[]);
        parser
            .push("<｜DSML｜ calls><｜DSML｜ invoke name=\"run\"><｜DSML｜ parameter name=\"text\" string=\"true\">")
            .unwrap();
        BOUNDARY_EXAMINED_BYTES.with(|examined| examined.set(0));

        let value = "x".repeat(16 * 1024);
        for byte in value.as_bytes() {
            parser
                .push(std::str::from_utf8(std::slice::from_ref(byte)).unwrap())
                .unwrap();
        }
        let events = parser
            .push("</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>")
            .unwrap();
        let examined = BOUNDARY_EXAMINED_BYTES.with(std::cell::Cell::get);

        let output: UnifiedParserOutput = events.into_iter().collect();
        assert_eq!(
            output.assembled(),
            vec![UnifiedEvent::ToolCall {
                name: "run".into(),
                arguments: serde_json::json!({"text": value}),
            }]
        );
        assert!(
            examined < value.len() * PARAMETER_END.len() * 2,
            "boundary examined {examined} bytes for a {}-byte value",
            value.len()
        );
    }

    #[test]
    fn json_invocation_boundary_scans_incrementally() {
        let mut boundary = DeepSeekV41InvocationBoundary::default();
        let mut candidate = format!("{INVOKE_START}run\">{{\"text\":\"");
        assert_eq!(boundary.end_append(&candidate, &candidate, false, 0), None);
        BOUNDARY_EXAMINED_BYTES.with(|examined| examined.set(0));
        for _ in 0..16 * 1024 {
            candidate.push('x');
            assert_eq!(boundary.end_append(&candidate, "x", false, 0), None);
        }
        let tail = format!("\"}}{INVOKE_END}");
        candidate.push_str(&tail);
        assert_eq!(
            boundary.end_append(&candidate, &tail, false, 0),
            Some(candidate.len())
        );
        let examined = BOUNDARY_EXAMINED_BYTES.with(std::cell::Cell::get);
        assert!(examined < candidate.len() * 2, "examined {examined} bytes");
    }

    #[test]
    fn guided_bare_header_scans_streamed_name_linearly() {
        let mut boundary = DeepSeekV41InvocationBoundary::default();
        let context = GuidedInvokePrefixContext {
            outside_reasoning: true,
            payload_is_empty: true,
            followed_by_competing_marker: false,
        };
        let mut candidate = INVOKE_START.to_string();
        assert_eq!(
            boundary.guided_prefix_append(&candidate, INVOKE_START, context),
            Some(GuidedInvokePrefix::Pending)
        );
        BOUNDARY_EXAMINED_BYTES.with(|examined| examined.set(0));

        let name = "x".repeat(16 * 1024);
        for byte in name.bytes() {
            let append = std::str::from_utf8(std::slice::from_ref(&byte)).unwrap();
            candidate.push_str(append);
            assert_eq!(
                boundary.guided_prefix_append(&candidate, append, context),
                Some(GuidedInvokePrefix::Pending)
            );
        }
        let examined = BOUNDARY_EXAMINED_BYTES.with(std::cell::Cell::get);
        assert!(
            examined < name.len() * 4,
            "guided prefix examined {examined} bytes for a {}-byte name",
            name.len()
        );
    }

    #[test]
    fn malformed_invocation_is_an_error_without_tool_deltas() {
        for input in [
            "<｜DSML｜ calls><｜DSML｜ invoke name=\"run\"><｜DSML｜ parameter name=\"x\" string=\"true\">first</｜DSML｜ parameter><｜DSML｜ parameter name=\"x\" string=\"true\">second</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>",
            "<｜DSML｜ calls><｜DSML｜ invoke name=\"run\"><｜DSML｜ parameter name=\"value\" string=\"false\">invalid</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>",
        ] {
            for split in (0..=input.len()).filter(|&i| input.is_char_boundary(i)) {
                let mut parser = deepseek_v41_unified(&[]);
                let mut output = UnifiedParserOutput::default();
                let result = parser
                    .parse_into(&input[..split], &mut output)
                    .and_then(|()| parser.parse_into(&input[split..], &mut output));
                assert!(result.is_err(), "split {split}");
                assert!(output.events.is_empty(), "split {split}");
            }
        }
    }

    #[test]
    fn closed_malformed_parameter_is_an_error_at_eof() {
        for input in [
            "<｜DSML｜ calls><｜DSML｜ invoke name=\"run\">{\"value\":é}</｜DSML｜ invoke></｜DSML｜ calls>",
            "<｜DSML｜ calls><｜DSML｜ invoke name=\"run\">{} trailing junk</｜DSML｜ invoke></｜DSML｜ calls>",
            "<｜DSML｜ calls><｜DSML｜ invoke name=\"run\"><｜DSML｜ parameter name=\"value\" string=\"maybe\">1</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>",
            "<｜DSML｜ calls><｜DSML｜ invoke name=\"run\"><｜DSML｜ parameter name=\"value\" string=\"false\">1</｜DSML｜ invoke></｜DSML｜ calls>",
        ] {
            for split in (0..=input.len()).filter(|&i| input.is_char_boundary(i)) {
                let mut parser = deepseek_v41_unified(&[]);
                let mut output = UnifiedParserOutput::default();
                let result = parser
                    .parse_into(&input[..split], &mut output)
                    .and_then(|()| parser.parse_into(&input[split..], &mut output))
                    .and_then(|()| parser.finish().map(|_| ()));
                assert!(result.is_err(), "split {split}");
                assert!(output.events.is_empty(), "split {split}");
            }
        }
    }

    #[test]
    fn guided_output_uses_shared_decoder() {
        let mut parser = deepseek_v41_unified(&[]);
        parser
            .initialize_request(UnifiedParserInit {
                starting_state: UnifiedParserStartingState::Reasoning,
                tool_output_mode: UnifiedToolOutputMode::GuidedJson {
                    named_tool: Some("weather".into()),
                },
                ..Default::default()
            })
            .unwrap();
        assert_eq!(
            parser
                .parse_complete("checking weather</think>{\"city\":\"Paris\"}")
                .unwrap(),
            vec![
                UnifiedEvent::Reasoning {
                    text: "checking weather".into()
                },
                UnifiedEvent::ToolCall {
                    name: "weather".into(),
                    arguments: serde_json::json!({"city":"Paris"})
                },
            ]
        );
    }

    #[test]
    fn guided_bare_headers_do_not_consume_json_or_reasoning() {
        let payload = r#"[{"name":"get_weather","arguments":{"city":"Paris"}}]"#;
        let invalid_payloads = [
            (
                r#"[{"name":"get_weather","arguments":{"city": "#,
                r#"[{"name":"get_weather","arguments":{"city": "#,
            ),
            (r#"{"unexpected":"shape"}"#, r#"{"unexpected":"shape"}"#),
            (
                r#"[{"name":"get_weather","arguments":{"city":"Paris"}},{"arguments":{}}]"#,
                r#"[{"name":"get_weather","arguments":{"city":"Paris"}},{"arguments":{}}]"#,
            ),
        ];
        let init = UnifiedParserInit {
            tool_output_mode: UnifiedToolOutputMode::GuidedJson { named_tool: None },
            invalid_guided_payload: InvalidGuidedPayloadPolicy::RecoverAsText,
            ..Default::default()
        };
        for (input, expected) in invalid_payloads {
            assert_every_split_with_init(
                &format!("{INVOKE_START}{input}"),
                init.clone(),
                vec![UnifiedEvent::Text {
                    text: expected.into(),
                }],
            );
        }
        assert_every_split_with_init(&format!("{INVOKE_START}>{payload}"), init.clone(), vec![]);
        assert_every_split_with_init(
            &format!("{INVOKE_START}<think>secret</think>{payload}"),
            init.clone(),
            vec![
                UnifiedEvent::Reasoning {
                    text: "secret".into(),
                },
                UnifiedEvent::ToolCall {
                    name: "get_weather".into(),
                    arguments: serde_json::json!({"city":"Paris"}),
                },
            ],
        );
        assert_every_split_with_init(
            &format!("<think>I'll use {INVOKE_START} next</think>{payload}"),
            init.clone(),
            vec![
                UnifiedEvent::Reasoning {
                    text: "I'll use  next".into(),
                },
                UnifiedEvent::ToolCall {
                    name: "get_weather".into(),
                    arguments: serde_json::json!({"city":"Paris"}),
                },
            ],
        );
        assert_every_split_with_init(
            &format!("<think>I'll call {INVOKE_START}get_weather</think>{payload}"),
            init,
            vec![
                UnifiedEvent::Reasoning {
                    text: "I'll call get_weather".into(),
                },
                UnifiedEvent::ToolCall {
                    name: "get_weather".into(),
                    arguments: serde_json::json!({"city":"Paris"}),
                },
            ],
        );
    }

    fn call(name: &str, arguments: serde_json::Value) -> UnifiedEvent {
        UnifiedEvent::ToolCall {
            name: name.into(),
            arguments,
        }
    }

    fn text(text: &str) -> UnifiedEvent {
        UnifiedEvent::Text { text: text.into() }
    }

    /// One `write` invocation whose `content` value is `value`, followed by an
    /// `i` parameter.
    fn write_invoke(value: &str) -> String {
        format!(
            "<｜DSML｜ invoke name=\"write\">\n<｜DSML｜ parameter name=\"path\" string=\"true\">docs/format.md</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"content\" string=\"true\">{value}</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"i\" string=\"true\">doc</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n"
        )
    }

    /// [`write_invoke`] alone inside a complete calls block.
    fn write_call(value: &str) -> String {
        format!("<｜DSML｜ calls>\n{}</｜DSML｜ calls>", write_invoke(value))
    }

    fn write_arguments(value: &str) -> serde_json::Value {
        serde_json::json!({"path": "docs/format.md", "content": value, "i": "doc"})
    }

    #[test]
    fn literal_dsml_block_in_string_value_is_data() {
        // A file that documents the call syntax quotes a complete, well-formed
        // calls block. Its first `</｜DSML｜ parameter>` is data: taking it as
        // the close strands the rest of the block, the real close, and `i` in
        // the trailing text.
        let value = concat!(
            "# Format\n\n```\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"read\">\n",
            "<｜DSML｜ parameter name=\"path\" string=\"true\">app/main.py</｜DSML｜ parameter>\n",
            "</｜DSML｜ invoke>\n</｜DSML｜ calls>\n```\n\n<!-- END-OF-FILE-7f3a -->\n",
        );
        assert_every_split(
            &format!("Writing it.\n\n{}", write_call(value)),
            UnifiedParserStartingState::None,
            vec![
                text("Writing it.\n\n"),
                call("write", write_arguments(value)),
            ],
        );
    }

    #[test]
    fn mixed_dialect_literal_block_in_string_value_is_data() {
        // The shape a model actually emitted: the quoted example mixes the V4
        // openers with V4.1 parameter markup, and a second fenced example
        // quotes a bare block close.
        let value = concat!(
            "# DeepSeek Tool Call Format\n\nBelow is an example of a valid tool call:\n\n```\n",
            "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"read\">\n",
            "<｜DSML｜ parameter name=\"path\">app/main.py</｜DSML｜ parameter>\n",
            "</｜DSML｜ invoke>\n</｜DSML｜ calls>\n```\n\n",
            "Below is a broken call that has only the closing tags:\n\n```\n</invoke>\n</｜DSML｜ calls>\n```\n\n",
            "<!-- END-OF-FILE-7f3a -->\n",
        );
        assert_every_split(
            &format!("Writing it.\n\n{}", write_call(value)),
            UnifiedParserStartingState::None,
            vec![
                text("Writing it.\n\n"),
                call("write", write_arguments(value)),
            ],
        );
    }

    #[test]
    fn literal_dsml_block_before_a_later_call_is_data() {
        let value = concat!(
            "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"read\">\n",
            "<｜DSML｜ parameter name=\"path\" string=\"true\">a.py</｜DSML｜ parameter>\n",
            "</｜DSML｜ invoke>\n</｜DSML｜ calls>\n",
        );
        let input = format!(
            "<｜DSML｜ calls>\n{}<｜DSML｜ invoke name=\"done\">\n</｜DSML｜ invoke>\n</｜DSML｜ calls>",
            write_invoke(value)
        );
        assert_every_split(
            &input,
            UnifiedParserStartingState::None,
            vec![
                call("write", write_arguments(value)),
                call("done", serde_json::json!({})),
            ],
        );
    }

    #[test]
    fn lone_literal_parameter_close_in_string_value_is_data() {
        // Only one reading leaves well-formed markup after the close: the one
        // that keeps the quoted close inside the value.
        for value in [
            "the close tag is </｜DSML｜ parameter> here",
            "ends with a close tag </｜DSML｜ parameter>",
        ] {
            assert_every_split(
                &write_call(value),
                UnifiedParserStartingState::None,
                vec![call("write", write_arguments(value))],
            );
        }
    }

    #[test]
    fn guard_lone_literal_parameter_open_keeps_first_close() {
        // Guard: a quoted opener with no quoted close must not make the parser
        // look for a second close. Naive depth counting fails here.
        let value = "fragment <｜DSML｜ parameter name=\"x\" string=\"true\">tail";
        assert_every_split(
            &write_call(value),
            UnifiedParserStartingState::None,
            vec![call("write", write_arguments(value))],
        );
    }

    #[test]
    fn guard_plain_parameters_are_unchanged() {
        assert_every_split(
            &write_call("plain text\nwith lines"),
            UnifiedParserStartingState::None,
            vec![call("write", write_arguments("plain text\nwith lines"))],
        );
    }

    #[test]
    fn guard_stranded_markers_without_coherent_alternative_keep_first_close() {
        // Guard: markup after the block is stranded, but no other close yields
        // a well-formed call, so the first close stands.
        let value = "fragment <｜DSML｜ parameter name=\"x\" string=\"true\">tail";
        let after = " Done; a value ends with </｜DSML｜ parameter> in this syntax.";
        assert_every_split(
            &format!("{}{after}", write_call(value)),
            UnifiedParserStartingState::None,
            vec![call("write", write_arguments(value)), text(after)],
        );
    }

    #[test]
    fn guard_literal_close_without_coherent_alternative_matches_first_close() {
        // Guard (known residual): a quoted close followed by markup that closes
        // the call is indistinguishable from a real close.
        let input = "<｜DSML｜ calls><｜DSML｜ invoke name=\"run\"><｜DSML｜ parameter name=\"text\" string=\"true\">a</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>";
        assert_every_split(
            input,
            UnifiedParserStartingState::None,
            vec![call("run", serde_json::json!({"text": "a"}))],
        );
        // With nothing coherent after the quoted close, the call stays an error.
        let input = "<｜DSML｜ calls><｜DSML｜ invoke name=\"run\"><｜DSML｜ parameter name=\"text\" string=\"true\">a</｜DSML｜ parameter> b</｜DSML｜ invoke></｜DSML｜ calls>";
        for split in (0..=input.len()).filter(|&i| input.is_char_boundary(i)) {
            let mut parser = deepseek_v41_unified(&[]);
            let mut output = UnifiedParserOutput::default();
            let result = parser
                .parse_into(&input[..split], &mut output)
                .and_then(|()| parser.parse_into(&input[split..], &mut output))
                .and_then(|()| parser.finish().map(|_| ()));
            assert!(result.is_err(), "split {split}");
            assert!(output.events.is_empty(), "split {split}");
        }
    }

    #[test]
    fn malformed_parameter_header_after_a_close_stays_an_error() {
        // A model emitted `</｜DSML｜ parameter>` followed by a real parameter
        // header without its `string` attribute. The first-close rule rejects
        // the call; reading the close as quoted data instead would fold the
        // malformed header and its value into the previous string and drop
        // that parameter without any error.
        let close = "</｜DSML｜ parameter>";
        for header in [
            "<｜DSML｜ parameter name=\"content\">",
            "<｜DSML｜ parameter name=\"content\" string=\"maybe\">",
        ] {
            for before in [
                "write docs file",
                "a quoted close </｜DSML｜ parameter> then",
            ] {
                let invoke = format!(
                    "<｜DSML｜ invoke name=\"write\">\n<｜DSML｜ parameter name=\"i\" string=\"true\">{before}{close}\n{header}body{close}\n</｜DSML｜ invoke>"
                );
                assert!(DeepSeekV41.parse_invoke(&invoke, 0).is_err(), "{invoke:?}");
                let input = format!("<｜DSML｜ calls>\n{invoke}\n</｜DSML｜ calls>");
                for split in (0..=input.len()).filter(|&i| input.is_char_boundary(i)) {
                    let mut parser = deepseek_v41_unified(&[]);
                    let mut output = UnifiedParserOutput::default();
                    let result = parser
                        .parse_into(&input[..split], &mut output)
                        .and_then(|()| parser.parse_into(&input[split..], &mut output))
                        .and_then(|()| parser.finish().map(|_| ()));
                    assert!(result.is_err(), "{input:?} split {split}");
                    assert!(output.events.is_empty(), "{input:?} split {split}");
                }
            }
        }
    }

    #[test]
    fn quoted_parameter_named_like_a_real_one_is_data() {
        // The quoted example repeats the call's own `language` parameter, and
        // its `string="false"` twin quotes invalid JSON. Reading the quoted
        // close as structure yields a duplicate or invalid parameter, so only
        // the reading that keeps the example inside `code` parses.
        for (quoted, real) in [
            (
                "<｜DSML｜ parameter name=\"language\" string=\"true\">python</｜DSML｜ parameter>",
                "<｜DSML｜ parameter name=\"language\" string=\"true\">rust</｜DSML｜ parameter>",
            ),
            (
                "<｜DSML｜ parameter name=\"count\" string=\"false\">bad</｜DSML｜ parameter>",
                "<｜DSML｜ parameter name=\"language\" string=\"true\">rust</｜DSML｜ parameter>",
            ),
        ] {
            let code = format!("x</｜DSML｜ parameter>\n{quoted}\nmore");
            let input = format!(
                "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"exec\">\n<｜DSML｜ parameter name=\"code\" string=\"true\">{code}</｜DSML｜ parameter>\n{real}\n</｜DSML｜ invoke>\n</｜DSML｜ calls>"
            );
            assert_every_split(
                &input,
                UnifiedParserStartingState::None,
                vec![call(
                    "exec",
                    serde_json::json!({"code": code, "language": "rust"}),
                )],
            );
        }
    }

    #[test]
    fn quoted_call_reusing_the_parameter_names_is_data() {
        // The quoted example is a whole call to the same tool, so every one of
        // its parameters repeats a real name. Whichever real parameter holds
        // the example, the only reading with distinct names keeps the example
        // inside that value.
        let quoted = "Doc:\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"exec\">\n<｜DSML｜ parameter name=\"language\" string=\"true\">python</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"code\" string=\"true\">x</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>\nend";
        let parameter = |name: &str, value: &str| {
            format!(
                "<｜DSML｜ parameter name=\"{name}\" string=\"true\">{value}</｜DSML｜ parameter>\n"
            )
        };
        for (language, code) in [("rust", quoted), (quoted, "y")] {
            for order in [["language", "code"], ["code", "language"]] {
                let value = |name: &str| if name == "language" { language } else { code };
                let input = format!(
                    "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"exec\">\n{}{}</｜DSML｜ invoke>\n</｜DSML｜ calls>",
                    parameter(order[0], value(order[0])),
                    parameter(order[1], value(order[1])),
                );
                assert_every_split(
                    &input,
                    UnifiedParserStartingState::None,
                    vec![call(
                        "exec",
                        serde_json::json!({"language": language, "code": code}),
                    )],
                );
            }
        }
    }

    #[test]
    fn repeated_parameter_is_not_read_as_quoted_data() {
        // A real parameter emitted twice is a model error. No reading may hide
        // it by folding the parameters between the two into one string value.
        let parameter = |name: &str, value: &str| {
            format!(
                "<｜DSML｜ parameter name=\"{name}\" string=\"true\">{value}</｜DSML｜ parameter>\n"
            )
        };
        for body in [
            [
                ("language", "python"),
                ("language", "python"),
                ("code", "x"),
            ],
            [
                ("language", "python"),
                ("code", "x"),
                ("language", "python"),
            ],
            [("code", "x"), ("language", "python"), ("language", "rust")],
        ] {
            let body: String = body
                .iter()
                .map(|(name, value)| parameter(name, value))
                .collect();
            let input = format!(
                "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"exec\">\n{body}</｜DSML｜ invoke>\n</｜DSML｜ calls>"
            );
            for split in (0..=input.len()).filter(|&i| input.is_char_boundary(i)) {
                let mut parser = deepseek_v41_unified(&[]);
                let mut output = UnifiedParserOutput::default();
                let result = parser
                    .parse_into(&input[..split], &mut output)
                    .and_then(|()| parser.parse_into(&input[split..], &mut output));
                let error = result.expect_err("repeated parameter").to_string();
                assert!(error.contains("duplicate"), "split {split}: {error}");
            }
        }
    }

    #[test]
    fn invocation_parse_matches_stream_for_quoted_markup() {
        // The emitter alone, given only the invocation, selects the same closes
        // as the streamed boundary did for the whole stream.
        for value in [
            "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"read\">\n<｜DSML｜ parameter name=\"path\" string=\"true\">a.py</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>\n",
            "<｜DSML｜tool_calls>\n<｜DSML｜ parameter name=\"path\">a.py</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n",
            "the close tag is </｜DSML｜ parameter> here",
            "ends with a close tag </｜DSML｜ parameter>",
            "fragment <｜DSML｜ parameter name=\"x\" string=\"true\">tail",
            "plain text\nwith lines",
        ] {
            let delta = DeepSeekV41
                .parse_invoke(write_invoke(value).trim_end(), 0)
                .unwrap()
                .unwrap();
            assert_eq!(
                serde_json::from_str::<Value>(&delta.arguments).unwrap(),
                write_arguments(value),
                "{value:?}"
            );
            assert_every_split(
                &write_call(value),
                UnifiedParserStartingState::None,
                vec![call("write", write_arguments(value))],
            );
        }
    }

    #[test]
    fn call_quoting_a_parameter_opener_is_committed_at_eof() {
        // Only the whole stream shows whether a close after a quoted opener was
        // structure, so that call is held until `finish`. A call without one
        // still streams at its close (`calls_stream_at_each_invocation_close`).
        let value = "fragment <｜DSML｜ parameter name=\"x\" string=\"true\">tail";
        let mut parser = deepseek_v41_unified(&[]);
        assert!(parser.push(&write_call(value)).unwrap().is_empty());
        let output: UnifiedParserOutput = parser.finish().unwrap().events.into_iter().collect();
        assert_eq!(
            output.assembled(),
            vec![call("write", write_arguments(value))]
        );
    }

    #[test]
    fn quoted_close_resolution_falls_back_after_bounded_work() {
        // Thousands of later invocation closes, none of which yields a
        // well-formed call: resolution tries a bounded number and keeps the
        // first close, so the output matches the first-close rule.
        let value = "<｜DSML｜ parameter name=\"x\" string=\"true\">q";
        let tail = INVOKE_END.repeat(4096);
        let input = format!(
            "<｜DSML｜ calls><｜DSML｜ invoke name=\"run\"><｜DSML｜ parameter name=\"text\" string=\"true\">{value}</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>{tail}"
        );
        let mut parser = deepseek_v41_unified(&[]);
        assert_eq!(
            parser.parse_complete(&input).unwrap(),
            vec![call("run", serde_json::json!({"text": value})), text(&tail)]
        );
    }

    #[test]
    fn reset_restarts_tool_indices() {
        let input =
            "<｜DSML｜ calls><｜DSML｜ invoke name=\"done\"></｜DSML｜ invoke></｜DSML｜ calls>";
        let mut parser = deepseek_v41_unified(&[]);
        let first = parser.push(input).unwrap();
        parser.finish().unwrap();
        assert!(parser.reset().is_empty());
        assert_eq!(parser.push(input).unwrap(), first);
    }
}
