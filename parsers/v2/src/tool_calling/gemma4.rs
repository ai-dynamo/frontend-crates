// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Streaming tool-call parser for Gemma 4.
//!
//! Gemma 4 emits tool calls with a custom non-JSON, non-XML grammar:
//!   `<|tool_call>call:NAME{key:<|"|>value<|"|>, key2:42, key3:[...]}<tool_call|>`
//! with bare unquoted keys, `<|"|>`-delimited strings, nested objects/arrays, and
//! MULTIPLE calls concatenated with NO separator. START = `<|tool_call>`,
//! END = `<tool_call|>` (asymmetric), and the block IS the invoke — the same
//! `<tool_call|>` closes both.
//!
//! The streaming concerns (buffering, chunk-split marker safety, normal_text
//! suppression, orphan-close stripping, EOF-truncation drop) live in the shared
//! [`WrappedBlockScanner`], not here. Gemma 4 supplies only what its markers
//! cannot express, through [`GemmaInvokeDriver`]: `<tool_call|>` and `call:` both
//! occur legitimately INSIDE a `<|"|>`-delimited string value. The model-owned
//! driver advances one cursor across chunks; the shared coordinator owns no
//! Gemma markers or states. One scanner, one boundary definition, shared with
//! the batch parser and with the unified parser
//! ([`crate::unified::gemma4`]).
//!
//! Per-invoke name + value typing is delegated to the v1 batch parser
//! `parse_one_tool_call_gemma4`, so a streamed call matches exactly what the
//! batch parser produces. A call is emitted only once its complete block has
//! streamed; at EOF a body that balanced without its close marker is still
//! recovered (v1 parity, case `5.b`), while one truncated mid-value is DROPPED
//! rather than emitted with empty arguments.
//!
//! Arguments are re-serialized in source key order because the v1 parser builds
//! them from a `serde_json::Map` (a `BTreeMap` without the `preserve_order`
//! feature), which sorts keys alphabetically; the fixtures store arguments as an
//! exact JSON string in the model-emitted order (the order vLLM's Rust parser also
//! preserves), so order has to be pinned to source order.

use crate::tool_calling::scan::{
    BareRecoveryLatch, GuidedInvokeContext, GuidedPrefix, GuidedPrefixScan, InvokeBoundary,
    InvokeDriver, InvokeEmitter, InvokeLatch, InvokeScan, InvokeStart, WrappedBlockScanner,
    WrappedBlockSpec, reorder_arguments,
};
use crate::tool_calling::v1core::ToolDefinition;
use crate::tool_calling::v1core::gemma4::{is_call_prefix_boundary, parse_one_tool_call_gemma4};

use crate::tool_calling::traits::{Tool, ToolCallDelta, ToolParseResult, ToolParser};

pub(crate) const TOOL_CALL_START: &str = "<|tool_call>";
pub(crate) const TOOL_CALL_END: &str = "<tool_call|>";
pub(crate) const REASONING_START: &str = "<|channel>";
pub(crate) const REASONING_END: &str = "<channel|>";
pub(crate) const REASONING_START_LABEL: &str = "thought\n";
const CALL_PREFIX: &str = "call:";
const STRING_DELIM: &str = "<|\"|>";

/// Build the Gemma 4 scanner. ONE construction site, shared by the tool-only
/// parser below and by the unified parser, so the grammar cannot drift between
/// the two surfaces.
pub(crate) fn gemma4_scanner(
    tools: &[Tool],
) -> WrappedBlockScanner<Gemma4InvokeEmitter, GemmaInvokeDriver> {
    WrappedBlockScanner::with_driver(
        WrappedBlockSpec {
            family: "gemma4",
            block_starts: vec![TOOL_CALL_START.to_string()],
            // Same token as `invoke_end`: closing the call closes the block.
            block_ends: vec![TOOL_CALL_END.to_string()],
            invoke_start: CALL_PREFIX.to_string(),
            invoke_end: TOOL_CALL_END.to_string(),
            orphan_markers: vec![TOOL_CALL_END.to_string()],
            // Only unconditional structural markers belong here because the
            // guided path strips this set as markup. The ambiguous `call:`
            // prefix is owned entirely by `GemmaInvokeDriver`.
            // The string delimiter is unconditionally structural and must not
            // be released when split across a chunk boundary.
            holdback_markers: vec![
                TOOL_CALL_START.to_string(),
                TOOL_CALL_END.to_string(),
                STRING_DELIM.to_string(),
            ],
            // No outer wrapper survives a recovered bare call, so later narration
            // is the user's text again.
            bare_recovery_latch: BareRecoveryLatch::Clear,
            invoke_latch: InvokeLatch::IfEmitted,
            drop_invoke_crossing_block_end: false,
            invoke_scan: None,
            preserve_special_tokens: true,
        },
        Gemma4InvokeEmitter {
            tools: tools.iter().map(ToolDefinition::from).collect(),
        },
        GemmaInvokeDriver::default(),
    )
}

/// Incremental ownership of Gemma's ambiguous `call:NAME{...}` envelope.
/// The shared coordinator sees only [`InvokeDriver`]; all marker, string, and
/// recovery policy remains beside the grammar that defines it.
#[derive(Default)]
pub(crate) struct GemmaInvokeDriver {
    base: Option<usize>,
    cursor: usize,
    name_started: bool,
    depth: usize,
    in_string: bool,
    body_end: Option<usize>,
    invalid: bool,
    saw_recovery_candidate: bool,
    recovery: GemmaRecovery,
    resync_target: Option<usize>,
    #[cfg(test)]
    visited_bytes: usize,
}

#[derive(Default)]
struct GemmaRecovery {
    cursor: usize,
    in_string: bool,
    candidate: Option<RecoveryCandidate>,
}

#[derive(Clone, Copy)]
struct RecoveryCandidate {
    start: usize,
    phase: RecoveryPhase,
}

#[derive(Clone, Copy)]
enum RecoveryPhase {
    Prefix(usize),
    Name(bool),
    Body(usize),
    Closer,
}

impl GemmaInvokeDriver {
    fn begin_at(&mut self, at: usize) {
        if self.base != Some(at) {
            *self = Self {
                base: Some(at),
                cursor: at + CALL_PREFIX.len(),
                ..Self::default()
            };
        }
    }

    fn visit(&mut self, bytes: usize) {
        #[cfg(test)]
        {
            self.visited_bytes += bytes;
        }
        #[cfg(not(test))]
        let _ = bytes;
    }

    fn advance(&mut self, text: &str, flush: bool) -> InvokeBoundary {
        if self.invalid {
            return self.resynchronize(text);
        }

        while self.body_end.is_none() && self.cursor < text.len() {
            let rest = &text[self.cursor..];
            let Some(ch) = rest.chars().next() else {
                break;
            };
            if self.depth == 0 {
                if ch == '{' && self.name_started {
                    self.depth = 1;
                } else if ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.') {
                    self.name_started = true;
                } else {
                    self.invalid = true;
                    return self.resynchronize(text);
                }
                self.visit(ch.len_utf8());
                self.cursor += ch.len_utf8();
                continue;
            }

            if !self.in_string
                && (rest.starts_with(TOOL_CALL_START) || TOOL_CALL_START.starts_with(rest))
            {
                if rest.len() < TOOL_CALL_START.len() {
                    return InvokeBoundary::Pending;
                }
                self.saw_recovery_candidate = true;
            }

            if STRING_DELIM.starts_with(rest) && rest.len() < STRING_DELIM.len() {
                return InvokeBoundary::Pending;
            }
            if rest.starts_with(STRING_DELIM) {
                self.in_string = !self.in_string;
                self.visit(STRING_DELIM.len());
                self.cursor += STRING_DELIM.len();
                continue;
            }
            if !self.in_string {
                match ch {
                    '{' => self.depth += 1,
                    '}' => {
                        self.depth -= 1;
                        if self.depth == 0 {
                            self.body_end = Some(self.cursor);
                            self.cursor += ch.len_utf8();
                            break;
                        }
                    }
                    _ => {}
                }
            }
            self.visit(ch.len_utf8());
            self.cursor += ch.len_utf8();
        }

        let Some(body_end) = self.body_end else {
            return if self.saw_recovery_candidate {
                self.resynchronize(text)
            } else {
                InvokeBoundary::Pending
            };
        };
        let after = &text[body_end + 1..];
        if after.starts_with(TOOL_CALL_END) {
            let mut end = body_end + 1 + TOOL_CALL_END.len();
            while text[end..].starts_with(TOOL_CALL_END) {
                end += TOOL_CALL_END.len();
            }
            return InvokeBoundary::Complete(end - self.base.unwrap_or_default());
        }
        if after.is_empty() {
            return if flush {
                InvokeBoundary::Complete(body_end + 1 - self.base.unwrap_or_default())
            } else {
                InvokeBoundary::Pending
            };
        }
        if after.len() < TOOL_CALL_END.len() && TOOL_CALL_END.starts_with(after) {
            return InvokeBoundary::Pending;
        }
        while self.cursor < text.len() {
            let ch = text[self.cursor..]
                .chars()
                .next()
                .expect("cursor is before end");
            if !ch.is_whitespace() {
                self.invalid = true;
                return self.resynchronize(text);
            }
            self.visit(ch.len_utf8());
            self.cursor += ch.len_utf8();
        }
        if flush {
            return InvokeBoundary::Complete(body_end + 1 - self.base.unwrap_or_default());
        }
        InvokeBoundary::Pending
    }

    fn resynchronize(&mut self, text: &str) -> InvokeBoundary {
        if let Some(next) = self.resync_target {
            return InvokeBoundary::Resynchronize(next - self.base.unwrap_or_default());
        }

        let base = self.base.unwrap_or_default();
        if self.recovery.cursor <= base {
            // Skip the malformed invoke's own `call:` start. A structural
            // recovery target must be a later wrapped call.
            self.recovery.cursor = base + 1;
        }

        while self.recovery.cursor < text.len() {
            let rest = &text[self.recovery.cursor..];

            if STRING_DELIM.starts_with(rest) && rest.len() < STRING_DELIM.len() {
                return InvokeBoundary::Pending;
            }
            if rest.starts_with(STRING_DELIM) {
                self.recovery.in_string = !self.recovery.in_string;
                self.visit(STRING_DELIM.len());
                self.recovery.cursor += STRING_DELIM.len();
                continue;
            }

            if self.recovery.in_string {
                let ch = rest.chars().next().expect("cursor is before end");
                self.visit(ch.len_utf8());
                self.recovery.cursor += ch.len_utf8();
                continue;
            }

            if TOOL_CALL_START.starts_with(rest) && rest.len() < TOOL_CALL_START.len() {
                return InvokeBoundary::Pending;
            }
            if rest.starts_with(TOOL_CALL_START) {
                self.recovery.candidate = Some(RecoveryCandidate {
                    start: self.recovery.cursor,
                    phase: RecoveryPhase::Prefix(0),
                });
                self.visit(TOOL_CALL_START.len());
                self.recovery.cursor += TOOL_CALL_START.len();
                continue;
            }

            let phase = self.recovery.candidate.map(|candidate| candidate.phase);
            match phase {
                Some(RecoveryPhase::Prefix(matched)) => {
                    let byte = text.as_bytes()[self.recovery.cursor];
                    if byte == CALL_PREFIX.as_bytes()[matched] {
                        self.visit(1);
                        self.recovery.cursor += 1;
                        let next = matched + 1;
                        self.recovery
                            .candidate
                            .as_mut()
                            .expect("candidate exists")
                            .phase = if next == CALL_PREFIX.len() {
                            RecoveryPhase::Name(false)
                        } else {
                            RecoveryPhase::Prefix(next)
                        };
                    } else {
                        self.recovery.candidate = None;
                    }
                    continue;
                }
                Some(RecoveryPhase::Name(started)) => {
                    let ch = rest.chars().next().expect("cursor is before end");
                    if ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.') {
                        self.visit(ch.len_utf8());
                        self.recovery.cursor += ch.len_utf8();
                        self.recovery
                            .candidate
                            .as_mut()
                            .expect("candidate exists")
                            .phase = RecoveryPhase::Name(true);
                    } else if ch == '{' && started {
                        self.visit(1);
                        self.recovery.cursor += 1;
                        self.recovery
                            .candidate
                            .as_mut()
                            .expect("candidate exists")
                            .phase = RecoveryPhase::Body(1);
                    } else {
                        self.recovery.candidate = None;
                    }
                    continue;
                }
                Some(RecoveryPhase::Body(depth)) => {
                    let ch = rest.chars().next().expect("cursor is before end");
                    let next_depth = match ch {
                        '{' => depth + 1,
                        '}' => depth - 1,
                        _ => depth,
                    };
                    self.visit(ch.len_utf8());
                    self.recovery.cursor += ch.len_utf8();
                    self.recovery
                        .candidate
                        .as_mut()
                        .expect("candidate exists")
                        .phase = if next_depth == 0 {
                        RecoveryPhase::Closer
                    } else {
                        RecoveryPhase::Body(next_depth)
                    };
                    continue;
                }
                Some(RecoveryPhase::Closer) => {
                    if TOOL_CALL_END.starts_with(rest) && rest.len() < TOOL_CALL_END.len() {
                        return InvokeBoundary::Pending;
                    }
                    if rest.starts_with(TOOL_CALL_END) {
                        let next = self.recovery.candidate.expect("candidate exists").start;
                        self.resync_target = Some(next);
                        return InvokeBoundary::Resynchronize(next - base);
                    }
                    self.recovery.candidate = None;
                    continue;
                }
                None => {}
            }

            let ch = rest.chars().next().expect("cursor is before end");
            self.visit(ch.len_utf8());
            self.recovery.cursor += ch.len_utf8();
        }
        InvokeBoundary::Pending
    }
}

impl InvokeDriver for GemmaInvokeDriver {
    fn guided_prefix(&self) -> Option<GuidedPrefixScan> {
        Some(GuidedPrefixScan {
            classify: guided_call_prefix,
            max_pending_len: CALL_PREFIX.len() + REASONING_START.len() - 1,
        })
    }

    fn start(
        &mut self,
        _scan: Option<InvokeScan>,
        text: &str,
        at: usize,
        flush: bool,
    ) -> InvokeStart {
        if !is_call_prefix_boundary(text, at) {
            return InvokeStart::NoMatch;
        }
        self.begin_at(at);
        let boundary = self.advance(text, flush);
        if self.invalid && self.body_end.is_none() && self.depth == 0 {
            return InvokeStart::NoMatch;
        }
        match boundary {
            InvokeBoundary::Complete(_) => InvokeStart::Match,
            InvokeBoundary::Pending if self.depth > 0 || (flush && self.body_end.is_some()) => {
                InvokeStart::Match
            }
            InvokeBoundary::Pending if flush => InvokeStart::NoMatch,
            InvokeBoundary::Pending => InvokeStart::Pending,
            InvokeBoundary::Resynchronize(_) => InvokeStart::Match,
        }
    }

    fn boundary(
        &mut self,
        _scan: Option<InvokeScan>,
        text: &str,
        _invoke_end: &str,
        at: usize,
        flush: bool,
        _tool_index: usize,
    ) -> InvokeBoundary {
        self.begin_at(at);
        self.advance(text, flush)
    }

    fn holdback(&self, _scan: Option<InvokeScan>, text: &str) -> usize {
        partial_bare_opener_suffix_len(text)
    }

    fn pending_start(&self) -> Option<usize> {
        if self.depth == 0 && !self.invalid {
            self.base
        } else {
            None
        }
    }

    fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Classify Gemma's guided `call:` prefix at payload and reasoning boundaries.
/// A split prefix remains pending; ordinary prose is never reclassified as
/// control markup.
fn guided_call_prefix(
    text: &str,
    at: usize,
    context: GuidedInvokeContext,
    flush: bool,
) -> GuidedPrefix {
    if !is_call_prefix_boundary(text, at) {
        return GuidedPrefix::NoMatch;
    }
    let suffix = &text[at..];
    if suffix.len() < CALL_PREFIX.len() && CALL_PREFIX.starts_with(suffix) {
        return GuidedPrefix::Pending;
    }
    let Some(after_prefix) = suffix.strip_prefix(CALL_PREFIX) else {
        return GuidedPrefix::NoMatch;
    };
    match context {
        GuidedInvokeContext::PayloadBoundary if !text[..at].trim().is_empty() => {
            GuidedPrefix::NoMatch
        }
        GuidedInvokeContext::PayloadBoundary => match after_prefix.as_bytes().first() {
            None if flush => GuidedPrefix::NoMatch,
            None => GuidedPrefix::Pending,
            Some(b'{') | Some(b'[') => GuidedPrefix::Match,
            Some(_) if after_prefix.starts_with(REASONING_START) => GuidedPrefix::Match,
            Some(_) if REASONING_START.starts_with(after_prefix) => GuidedPrefix::Pending,
            Some(_) => GuidedPrefix::NoMatch,
        },
        GuidedInvokeContext::Reasoning => {
            let name_len = after_prefix
                .char_indices()
                .take_while(|(_, ch)| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.'))
                .map(|(at, ch)| at + ch.len_utf8())
                .last()
                .unwrap_or_default();
            if name_len == 0 {
                GuidedPrefix::NoMatch
            } else {
                match after_prefix[name_len..].chars().next() {
                    Some('{') => GuidedPrefix::NoMatch,
                    Some(_) => GuidedPrefix::Match,
                    None if flush => GuidedPrefix::Match,
                    None => GuidedPrefix::Pending,
                }
            }
        }
    }
}

/// Stream parser for Gemma 4 tool calls.
pub struct Gemma4ToolStreamParser {
    scanner: WrappedBlockScanner<Gemma4InvokeEmitter, GemmaInvokeDriver>,
}

impl Gemma4ToolStreamParser {
    pub fn new(tools: &[Tool]) -> Self {
        Self {
            scanner: gemma4_scanner(tools),
        }
    }
}

impl ToolParser for Gemma4ToolStreamParser {
    fn create(tools: &[Tool]) -> anyhow::Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    fn preserve_special_tokens(&self) -> bool {
        self.scanner.preserve_special_tokens()
    }

    fn push(&mut self, chunk: &str) -> anyhow::Result<ToolParseResult> {
        self.scanner.push(chunk)
    }

    fn finish(&mut self) -> anyhow::Result<ToolParseResult> {
        self.scanner.finish()
    }
}

/// Per-invoke typing for Gemma 4: hand the complete `call:NAME{…}<tool_call|>`
/// block to the v1 batch parser, then restore source key order.
pub(crate) struct Gemma4InvokeEmitter {
    tools: Vec<ToolDefinition>,
}

impl InvokeEmitter for Gemma4InvokeEmitter {
    fn parse_invoke(
        &mut self,
        invoke: &str,
        tool_index: usize,
    ) -> anyhow::Result<Option<ToolCallDelta>> {
        // The scanner already delimited this invoke, so type it directly rather
        // than re-scanning: re-discovering bounds is what truncates a value at an
        // embedded `<tool_call|>` (`I7`), and the span scanner also refuses a call
        // missing BOTH its opener and closer — the exact shape of case `5.b`.
        let call = parse_one_tool_call_gemma4(invoke, Some(&self.tools))?;
        // A complete-but-malformed block (no `call:NAME{...}` body, e.g. the
        // `<|tool_call>nonsense<tool_call|>` recovery case) yields no call;
        // drop it without leaking markup, matching the v1 no-leak contract.
        let Some(call) = call else {
            tracing::warn!(
                why = "gemma4_block_without_call",
                "Gemma 4 stream dropped a complete block that produced no call"
            );
            return Ok(None);
        };
        Ok(Some(ToolCallDelta {
            tool_index,
            name: Some(call.function.name),
            arguments: reorder_arguments(&call.function.arguments, &source_key_order(invoke)),
        }))
    }
}

/// A partial `call:` prefix at a word boundary. A complete prefix and growing
/// function name are retained by [`GemmaInvokeDriver::pending_start`] without a
/// suffix-wide search.
fn partial_bare_opener_suffix_len(text: &str) -> usize {
    for len in (1..CALL_PREFIX.len()).rev() {
        if text.ends_with(&CALL_PREFIX[..len]) && is_call_prefix_boundary(text, text.len() - len) {
            return len;
        }
    }
    0
}

/// Top-level argument key names in the order they appear in a Gemma 4 call body
/// `call:NAME{ key:value, key2:value2, ... }`. Walks the body once, tracking
/// brace/bracket depth and `<|"|>` string state so only depth-1 keys (the ones
/// immediately after the opening `{`) are collected; nested-object/array keys are
/// skipped. A key is the run of `[\w\-.]` characters that precedes a `:` at the
/// top level.
fn source_key_order(block: &str) -> Vec<String> {
    // Locate the call body: after `call:NAME` to the matching outer `{`.
    let Some(prefix_at) = block.find(CALL_PREFIX) else {
        return Vec::new();
    };
    let after_prefix = &block[prefix_at + CALL_PREFIX.len()..];
    let Some(open_rel) = after_prefix.find('{') else {
        return Vec::new();
    };
    let body = &after_prefix[open_rel + 1..];

    let mut names = Vec::new();
    let bytes = body.as_bytes();
    let mut i = 0usize;
    let mut depth = 0usize; // nesting depth INSIDE the outer object (0 = top level)
    let mut in_string = false;
    let mut expect_key = true; // at the start, and right after a top-level `,`

    while i < bytes.len() {
        // `<|"|>` toggles string state; skip its bytes wholesale so structural
        // chars inside a string value are ignored. `<|"|>` is ASCII and `i` is
        // always on a char boundary here (the in-string/fallback arms advance by
        // full char width), so the slice is safe even with multibyte values.
        if body[i..].starts_with(STRING_DELIM) {
            in_string = !in_string;
            i += STRING_DELIM.len();
            continue;
        }
        if in_string {
            // Advance by a full UTF-8 char so a multibyte value char (e.g. `ō`)
            // doesn't land `i` inside a code point.
            i += utf8_char_len(bytes[i]);
            continue;
        }

        let b = bytes[i];
        match b {
            b'{' | b'[' => {
                depth += 1;
                expect_key = false;
                i += 1;
            }
            b'}' | b']' => {
                depth = depth.saturating_sub(1);
                i += 1;
            }
            b',' if depth == 0 => {
                expect_key = true;
                i += 1;
            }
            _ if depth == 0 && expect_key && is_key_byte(b) => {
                let start = i;
                while i < bytes.len() && is_key_byte(bytes[i]) {
                    i += 1;
                }
                let name = body[start..i].to_string();
                // Only treat it as a key if a `:` follows (after optional spaces).
                let mut j = i;
                while j < bytes.len() && bytes[j].is_ascii_whitespace() {
                    j += 1;
                }
                if j < bytes.len() && bytes[j] == b':' {
                    names.push(name);
                }
                expect_key = false;
            }
            _ => {
                // Advance by a full UTF-8 char so a non-ASCII byte never lands
                // `i` mid-code-point (keeps the `<|"|>` slice check boundary-safe).
                i += utf8_char_len(b);
            }
        }
    }
    names
}

fn is_key_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b == b'-' || b == b'.'
}

/// Byte width of a UTF-8 code point from its leading byte.
fn utf8_char_len(lead: u8) -> usize {
    match lead {
        0x00..=0x7F => 1,
        0xC0..=0xDF => 2,
        0xE0..=0xEF => 3,
        _ => 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weather_tools() -> Vec<Tool> {
        vec![Tool {
            name: "get_weather".to_string(),
            description: None,
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "location": { "type": "string" } }
            }),
            strict: None,
        }]
    }

    fn parse_chunks(tools: &[Tool], chunks: &[&str]) -> ToolParseResult {
        let mut parser = Gemma4ToolStreamParser::new(tools);
        let mut out = ToolParseResult::default();
        for chunk in chunks {
            out.append(parser.push(chunk).expect("push"));
        }
        out.append(parser.finish().expect("finish"));
        out
    }

    #[test]
    fn incremental_driver_visits_long_pending_regions_once() {
        let mut driver = GemmaInvokeDriver::default();
        let mut pending = CALL_PREFIX.to_string();
        for chunk in "n".repeat(32 * 1024).as_bytes().chunks(4) {
            pending.push_str(std::str::from_utf8(chunk).expect("ASCII name"));
            assert_eq!(driver.start(None, &pending, 0, false), InvokeStart::Pending);
        }
        assert!(
            driver.visited_bytes <= pending.len(),
            "visited {} bytes for {} buffered bytes",
            driver.visited_bytes,
            pending.len()
        );

        driver.reset();
        let value = TOOL_CALL_END.repeat(2048);
        let complete =
            format!("call:get_weather{{city:{STRING_DELIM}{value}{STRING_DELIM}}}{TOOL_CALL_END}");
        let mut streamed = String::new();
        for chunk in complete.as_bytes().chunks(4) {
            streamed.push_str(std::str::from_utf8(chunk).expect("ASCII call"));
            let status = driver.start(None, &streamed, 0, false);
            if status == InvokeStart::Match {
                let _ = driver.boundary(None, &streamed, TOOL_CALL_END, 0, false, 0);
            }
        }
        assert!(
            driver.visited_bytes <= complete.len(),
            "visited {} bytes for {} buffered bytes",
            driver.visited_bytes,
            complete.len()
        );

        driver.reset();
        let mut malformed = "call:f{}junk".to_string();
        for chunk in TOOL_CALL_END.repeat(2048).as_bytes().chunks(4) {
            malformed.push_str(std::str::from_utf8(chunk).expect("ASCII closer"));
            let _ = driver.start(None, &malformed, 0, false);
            assert_eq!(
                driver.boundary(None, &malformed, TOOL_CALL_END, 0, false, 0),
                InvokeBoundary::Pending
            );
        }
        assert!(
            driver.visited_bytes <= malformed.len() * 2,
            "visited {} bytes for {} malformed buffered bytes",
            driver.visited_bytes,
            malformed.len()
        );
    }

    #[test]
    fn repeated_key_emits_key_once() {
        // A repeated top-level key must not produce duplicate keys in the arguments.
        let out = parse_chunks(
            &weather_tools(),
            &[
                "<|tool_call>call:get_weather{location:<|\"|>NYC<|\"|>,location:<|\"|>NYC<|\"|>}<tool_call|>",
            ],
        );
        let merged = out.coalesce_calls();
        assert_eq!(merged.calls.len(), 1);
        let args = merged.calls[0].arguments.clone();
        assert_eq!(
            args.matches("\"location\"").count(),
            1,
            "duplicate key in arguments: {args}"
        );
    }

    #[test]
    fn emits_complete_call_on_close() {
        let out = parse_chunks(
            &weather_tools(),
            &[
                "<|tool_call>call:get_weather{location:<|\"|>",
                "NYC<|\"|>",
                "}<tool_call|>",
            ],
        );
        assert_eq!(out.normal_text, "");
        let merged = out.coalesce_calls();
        assert_eq!(merged.calls.len(), 1);
        assert_eq!(merged.calls[0].tool_index, 0);
        assert_eq!(merged.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(merged.calls[0].arguments, r#"{"location":"NYC"}"#);
    }

    #[test]
    fn emits_multiple_concatenated_calls() {
        // Two back-to-back calls with NO separator, split mid-marker.
        let out = parse_chunks(
            &weather_tools(),
            &[
                "<|tool_call>call:get_weather{location:<|\"|>NYC<|\"|>",
                "}<tool_call|><|tool_call>call:get_weather{location:<|\"|>",
                "LA<|\"|>}<tool_call|>",
            ],
        );
        assert_eq!(out.normal_text, "");
        let merged = out.coalesce_calls();
        assert_eq!(merged.calls.len(), 2);
        assert_eq!(merged.calls[0].tool_index, 0);
        assert_eq!(merged.calls[0].arguments, r#"{"location":"NYC"}"#);
        assert_eq!(merged.calls[1].tool_index, 1);
        assert_eq!(merged.calls[1].arguments, r#"{"location":"LA"}"#);
    }

    #[test]
    fn preserves_prefix_text_before_block() {
        let out = parse_chunks(
            &weather_tools(),
            &[
                "I will",
                " check the weather. <|tool_call>",
                "call:get_weather{location:<|\"|>NYC<|\"|>}<tool_call|>",
            ],
        );
        assert_eq!(out.normal_text, "I will check the weather. ");
        assert_eq!(out.coalesce_calls().calls.len(), 1);
    }

    #[test]
    fn preserves_narration_after_call() {
        // Text after a complete call is still user-visible normal_text.
        let out = parse_chunks(
            &weather_tools(),
            &[
                "<|tool_call>call:get_weather{location:<|\"|>NYC<|\"|>",
                "}<tool_call|> Let me",
                " know if you need more.",
            ],
        );
        assert_eq!(out.normal_text, " Let me know if you need more.");
        assert_eq!(out.coalesce_calls().calls.len(), 1);
    }

    #[test]
    fn holds_back_partial_start_marker_across_boundaries() {
        // The full `<|tool_call>` and string delimiter `<|"|>` and end marker
        // `<tool_call|>` are all split across many tiny chunks; nothing must leak
        // into normal_text and the assembled call must be correct.
        let out = parse_chunks(
            &weather_tools(),
            &[
                "<|",
                "too",
                "l_cal",
                "l>call:get_",
                "weather{location:<|",
                "\"|",
                ">NY",
                "C<|\"|",
                ">}<tool_cal",
                "l|>",
            ],
        );
        assert_eq!(out.normal_text, "");
        let merged = out.coalesce_calls();
        assert_eq!(merged.calls.len(), 1);
        assert_eq!(merged.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(merged.calls[0].arguments, r#"{"location":"NYC"}"#);
    }

    #[test]
    fn recovers_complete_body_missing_end_marker() {
        // Body complete but no `<tool_call|>` end marker before EOF. The v1 batch
        // parser recovers this (batch case 5.a), and the streamv2 conformance tab
        // grades stream-vs-own-batch, so the stream parser must recover it too —
        // not drop it. (Contrast `drops_call_truncated_mid_value` below, where the
        // body itself is incomplete and v1 yields no call.)
        let out = parse_chunks(
            &weather_tools(),
            &[
                "<|tool_call>call:get_weather{location:<|\"|>",
                "NYC<|\"|>",
                "}",
            ],
        );
        assert_eq!(out.normal_text, "");
        let merged = out.coalesce_calls();
        assert_eq!(merged.calls.len(), 1, "complete body must be recovered");
        assert_eq!(merged.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(merged.calls[0].arguments, r#"{"location":"NYC"}"#);
    }

    #[test]
    fn complete_body_missing_end_marker_is_recovered_at_every_split() {
        // 5.b at every chunk boundary: the body balances but the close marker
        // never streams, so `finish` must recover it regardless of split point.
        let input = "<|tool_call>call:get_weather{location:<|\"|>NYC<|\"|>}";
        for split in input
            .char_indices()
            .map(|(index, _)| index)
            .chain(std::iter::once(input.len()))
        {
            let out = parse_chunks(&weather_tools(), &[&input[..split], &input[split..]]);
            assert_eq!(out.normal_text, "", "split={split}");
            let merged = out.coalesce_calls();
            assert_eq!(merged.calls.len(), 1, "split={split}");
            assert_eq!(
                merged.calls[0].name.as_deref(),
                Some("get_weather"),
                "split={split}"
            );
            assert_eq!(
                merged.calls[0].arguments, r#"{"location":"NYC"}"#,
                "split={split}"
            );
        }
    }

    #[test]
    fn mismatched_close_after_narration_is_rejected_at_every_split() {
        let input = "<|tool_call>call:get_weather{location:<|\"|>NYC<|\"|>} trailing </tool_call>";
        for split in input
            .char_indices()
            .map(|(index, _)| index)
            .chain(std::iter::once(input.len()))
        {
            let out = parse_chunks(&weather_tools(), &[&input[..split], &input[split..]]);
            assert_eq!(out.normal_text, "", "split={split}");
            assert!(out.calls.is_empty(), "split={split}");
        }
    }

    #[test]
    fn narrated_call_missing_both_wrappers_is_not_dispatched_at_every_split() {
        let input = "for example, call:get_weather{location:<|\"|>NYC<|\"|>}";
        for split in input
            .char_indices()
            .map(|(index, _)| index)
            .chain(std::iter::once(input.len()))
        {
            let out = parse_chunks(&weather_tools(), &[&input[..split], &input[split..]]);
            assert_eq!(out.normal_text, "for example, ", "split={split}");
            assert!(out.calls.is_empty(), "split={split}");
        }
    }

    #[test]
    fn drops_call_truncated_mid_value() {
        let out = parse_chunks(
            &weather_tools(),
            &["<|tool_call>call:get_weather{location:<|\"|>", "N"],
        );
        assert_eq!(out.normal_text, "");
        assert!(out.calls.is_empty());
    }

    #[test]
    fn truncated_mid_value_call_is_dropped_at_every_split() {
        // EOF-mid-value truncation is unrecoverable at every chunk boundary: the
        // string value never closes, so the whole call is dropped with no leak,
        // regardless of where the stream happens to be cut.
        let input = "<|tool_call>call:get_weather{location:<|\"|>N";
        for split in input
            .char_indices()
            .map(|(index, _)| index)
            .chain(std::iter::once(input.len()))
        {
            let out = parse_chunks(&weather_tools(), &[&input[..split], &input[split..]]);
            assert_eq!(out.normal_text, "", "split={split}");
            assert!(out.calls.is_empty(), "split={split}");
        }
    }

    #[test]
    fn keeps_complete_call_drops_truncated_tail() {
        // First call complete, second truncated mid-value: keep the first.
        let out = parse_chunks(
            &weather_tools(),
            &[
                "<|tool_call>call:get_weather{location:<|\"|>Boston<|\"|>}<tool_call|>",
                "<|tool_call>call:get_weather{location:<|\"|>New York",
            ],
        );
        let merged = out.coalesce_calls();
        assert_eq!(merged.calls.len(), 1, "truncated 2nd call dropped");
        assert_eq!(merged.calls[0].arguments, r#"{"location":"Boston"}"#);
    }

    #[test]
    fn preserves_source_key_order() {
        // destination, passengers, first_class is NOT alphabetical; the v1 parser
        // sorts keys (BTreeMap), so the parser must restore source order.
        let tools = vec![Tool {
            name: "book_flight".to_string(),
            description: None,
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "destination": { "type": "string" },
                    "passengers": { "type": "integer" },
                    "first_class": { "type": "boolean" }
                }
            }),
            strict: None,
        }];
        let out = parse_chunks(
            &tools,
            &[
                "<|tool_call>call:book_flight{destination:<|\"|>Paris<|\"|>",
                ",passengers:2,first_class:true}<tool_call|>",
            ],
        );
        let merged = out.coalesce_calls();
        assert_eq!(merged.calls.len(), 1);
        assert_eq!(
            merged.calls[0].arguments,
            r#"{"destination":"Paris","passengers":2,"first_class":true}"#
        );
    }

    #[test]
    fn handles_unicode_value_split_across_chunks() {
        // Multibyte chars (`ō`) inside a `<|"|>` string value must not break the
        // source-order key scan (regression: byte-index slicing inside a string).
        let out = parse_chunks(
            &weather_tools(),
            &[
                "<|tool_call>call:get_weather{location:<|\"|>",
                "Tōkyō",
                " central<|\"|>}<tool_call|>",
            ],
        );
        assert_eq!(out.normal_text, "");
        let merged = out.coalesce_calls();
        assert_eq!(merged.calls.len(), 1);
        assert_eq!(merged.calls[0].arguments, r#"{"location":"Tōkyō central"}"#);
    }

    #[test]
    fn no_tool_call_is_plain_text() {
        let out = parse_chunks(
            &weather_tools(),
            &["Hello, how", " can", " I help you", " today?"],
        );
        assert_eq!(out.normal_text, "Hello, how can I help you today?");
        assert!(out.calls.is_empty());
    }

    #[test]
    fn embedded_end_marker_inside_string_does_not_close_early() {
        // A literal `<tool_call|>` inside a `<|"|>` string value must not close
        // the block; the real close comes after the string ends.
        let tools = vec![Tool {
            name: "run_query".to_string(),
            description: None,
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "sql": { "type": "string" } }
            }),
            strict: None,
        }];
        let out = parse_chunks(
            &tools,
            &[
                "<|tool_call>call:run_query{sql:<|\"|>literal",
                " <|tool_call marker> call:get_time{}",
                " stays text<|\"|>",
                "}<tool_call|>",
            ],
        );
        assert_eq!(out.normal_text, "");
        let merged = out.coalesce_calls();
        assert_eq!(merged.calls.len(), 1);
        assert_eq!(
            merged.calls[0].arguments,
            r#"{"sql":"literal <|tool_call marker> call:get_time{} stays text"}"#
        );
    }

    #[test]
    fn recovers_bare_call_without_opener() {
        // streamv2.5.b: `call:NAME{...}<tool_call|>` with NO `<|tool_call>` opener.
        // The v1 parser recovers it (missing-start); the streaming parser must too,
        // so the call body is a recovered tool call, NOT leaked as normal_text.
        let out = parse_chunks(
            &weather_tools(),
            &["call:get_weather{location:<|\"|>NYC<|\"|>", "}<tool_call|>"],
        );
        assert_eq!(out.normal_text, "", "bare call body must not leak as text");
        let merged = out.coalesce_calls();
        assert_eq!(merged.calls.len(), 1);
        assert_eq!(merged.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(merged.calls[0].arguments, r#"{"location":"NYC"}"#);
    }

    #[test]
    fn recovers_bare_call_without_opener_at_every_split() {
        // streamv2.5.b at every chunk boundary: a bare `call:NAME{...}<tool_call|>`
        // with no `<|tool_call>` opener must recover as a call, never leak as text,
        // no matter where the stream happens to be cut.
        let input = "call:get_weather{location:<|\"|>NYC<|\"|>}<tool_call|>";
        for split in input
            .char_indices()
            .map(|(index, _)| index)
            .chain(std::iter::once(input.len()))
        {
            let out = parse_chunks(&weather_tools(), &[&input[..split], &input[split..]]);
            assert_eq!(out.normal_text, "", "split={split}");
            let merged = out.coalesce_calls();
            assert_eq!(merged.calls.len(), 1, "split={split}");
            assert_eq!(
                merged.calls[0].name.as_deref(),
                Some("get_weather"),
                "split={split}"
            );
            assert_eq!(
                merged.calls[0].arguments, r#"{"location":"NYC"}"#,
                "split={split}"
            );
        }
    }

    #[test]
    fn recovers_bare_call_keeps_prefix_prose() {
        // streamv2.5.g: genuine prose precedes a bare `call:` (no opener). The prose
        // stays normal_text; only the `call:...` body is recovered as a call.
        let out = parse_chunks(
            &weather_tools(),
            &[
                "I will",
                " check",
                " that. call:get_weather{location:<|\"|>NYC<|\"|>",
                "}<tool_call|>",
            ],
        );
        assert_eq!(out.normal_text, "I will check that. ");
        let merged = out.coalesce_calls();
        assert_eq!(merged.calls.len(), 1);
        assert_eq!(merged.calls[0].arguments, r#"{"location":"NYC"}"#);
    }

    #[test]
    fn recovers_bare_call_then_wrapped_call() {
        // streamv2.5.f: a bare valid call followed by a complete wrapped call. Both
        // are emitted, the bare one recovered, with no leak.
        let out = parse_chunks(
            &weather_tools(),
            &[
                "call:get_weather{location:<|\"|>NYC<|\"|>",
                "}<tool_call|>",
                "<|tool_call>call:get_weather{location:<|\"|>Boston<|\"|>",
                "}<tool_call|>",
            ],
        );
        assert_eq!(out.normal_text, "");
        let merged = out.coalesce_calls();
        assert_eq!(merged.calls.len(), 2);
        assert_eq!(merged.calls[0].arguments, r#"{"location":"NYC"}"#);
        assert_eq!(merged.calls[1].arguments, r#"{"location":"Boston"}"#);
    }

    #[test]
    fn bare_call_word_inside_prose_is_not_recovered() {
        // A `call:` that is just prose ("I will call: you") has no `{...}` body and
        // must NOT be treated as a tool call — it flows through as normal_text.
        let out = parse_chunks(
            &weather_tools(),
            &["I will call: you tomorrow", " about the trip."],
        );
        assert_eq!(out.normal_text, "I will call: you tomorrow about the trip.");
        assert!(out.calls.is_empty());
    }

    #[test]
    fn bare_call_truncated_at_eof_is_dropped() {
        // A bare `call:` body that never closes before EOF is dropped (truncation
        // parity), not recovered and not leaked.
        let out = parse_chunks(&weather_tools(), &["call:get_weather{location:<|\"|>NY"]);
        assert_eq!(out.normal_text, "");
        assert!(out.calls.is_empty());
    }
}
