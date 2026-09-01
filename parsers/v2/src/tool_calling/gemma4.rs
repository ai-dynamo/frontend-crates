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
//! cannot express, through [`InvokeScan`]: `<tool_call|>` and `call:` both occur
//! legitimately INSIDE a `<|"|>`-delimited string value, so where an invoke ends
//! and whether a `call:` opens one are answered by the v1 balanced,
//! string-aware scan rather than by `find`. One scanner, one boundary
//! definition, shared with the batch parser and with the unified parser
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
    BareRecoveryLatch, InvokeBoundary, InvokeBoundaryFactory, InvokeEmitter, InvokeLatch,
    WrappedBlockScanner, WrappedBlockSpec, reorder_arguments,
};
use crate::tool_calling::v1core::ToolDefinition;
use crate::tool_calling::v1core::gemma4::{
    has_bare_call_body_start_gemma4, is_call_prefix_boundary, parse_one_tool_call_gemma4,
};

use crate::tool_calling::traits::{Tool, ToolCallDelta, ToolParseResult, ToolParser};

pub(crate) const TOOL_CALL_START: &str = "<|tool_call>";
pub(crate) const TOOL_CALL_END: &str = "<tool_call|>";
const CALL_PREFIX: &str = "call:";
const STRING_DELIM: &str = "<|\"|>";

#[cfg(any(test, feature = "test-utils"))]
std::thread_local! {
    static BOUNDARY_EXAMINED_BYTES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(any(test, feature = "test-utils"))]
#[doc(hidden)]
pub fn reset_boundary_examined_bytes() {
    BOUNDARY_EXAMINED_BYTES.with(|examined| examined.set(0));
}

#[cfg(any(test, feature = "test-utils"))]
#[doc(hidden)]
pub fn boundary_examined_bytes() -> usize {
    BOUNDARY_EXAMINED_BYTES.with(std::cell::Cell::get)
}

fn count_boundary_bytes(bytes: usize) {
    #[cfg(any(test, feature = "test-utils"))]
    BOUNDARY_EXAMINED_BYTES.with(|examined| examined.set(examined.get() + bytes));
    #[cfg(not(any(test, feature = "test-utils")))]
    let _ = bytes;
}

#[derive(Default)]
struct Gemma4InvokeProgress {
    cursor: usize,
    name_started: bool,
    depth: usize,
    in_string: bool,
    body_end: Option<usize>,
    invalid: bool,
}

impl Gemma4InvokeProgress {
    fn reset(&mut self) {
        *self = Self::default();
    }

    fn end(&mut self, text: &str, flush: bool) -> Option<usize> {
        if self.invalid || !text.starts_with(CALL_PREFIX) {
            self.invalid = true;
            return None;
        }
        if self.body_end.is_none() {
            self.cursor = self.cursor.max(CALL_PREFIX.len());
            while self.cursor < text.len() {
                let rest = &text[self.cursor..];
                let ch = rest.chars().next()?;
                count_boundary_bytes(ch.len_utf8());
                if self.depth == 0 {
                    if ch == '{' && self.name_started {
                        self.depth = 1;
                    } else if ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.') {
                        self.name_started = true;
                    } else {
                        self.invalid = true;
                        return None;
                    }
                    self.cursor += ch.len_utf8();
                    continue;
                }
                if STRING_DELIM.starts_with(rest) && rest.len() < STRING_DELIM.len() {
                    return None;
                }
                if rest.starts_with(STRING_DELIM) {
                    self.in_string = !self.in_string;
                    self.cursor += STRING_DELIM.len();
                    continue;
                }
                if !self.in_string {
                    match ch {
                        '{' => self.depth += 1,
                        '}' => {
                            let Some(depth) = self.depth.checked_sub(1) else {
                                self.invalid = true;
                                return None;
                            };
                            self.depth = depth;
                            if self.depth == 0 {
                                self.body_end = Some(self.cursor);
                                self.cursor += ch.len_utf8();
                                break;
                            }
                        }
                        _ => {}
                    }
                }
                self.cursor += ch.len_utf8();
            }
        }
        let body_end = self.body_end?;
        let after = &text[body_end + 1..];
        if after.starts_with(TOOL_CALL_END) {
            let mut end = body_end + 1 + TOOL_CALL_END.len();
            while text[end..].starts_with(TOOL_CALL_END) {
                end += TOOL_CALL_END.len();
            }
            return Some(end);
        }
        if flush && after.trim().is_empty() {
            return Some(body_end + 1);
        }
        None
    }
}

#[derive(Default)]
struct Gemma4InvokeBoundary {
    progress: Gemma4InvokeProgress,
    candidate: String,
    resync_cursor: usize,
    resync_in_string: bool,
    resync_candidate: Option<(usize, usize)>,
}

impl InvokeBoundary for Gemma4InvokeBoundary {
    fn end_append(
        &mut self,
        _candidate: &str,
        append: &str,
        flush: bool,
        _tool_index: usize,
    ) -> Option<usize> {
        self.candidate.push_str(append);
        self.progress.end(&self.candidate, flush)
    }

    fn opens(&self, text: &str, at: usize) -> bool {
        opens_bare_call(text, at)
    }

    fn holdback(&self, text: &str) -> usize {
        partial_bare_opener_suffix_len(text)
    }

    fn resync(&mut self, input: &str, flush: bool, _tool_index: usize) -> Option<usize> {
        if self.resync_cursor > input.len() {
            self.resync_cursor = 0;
            self.resync_in_string = false;
            self.resync_candidate = None;
        }
        while self.resync_cursor < input.len() {
            let cursor = self.resync_cursor;
            let rest = &input[cursor..];
            if STRING_DELIM.starts_with(rest) && rest.len() < STRING_DELIM.len() {
                return None;
            }
            if TOOL_CALL_START.starts_with(rest) && rest.len() < TOOL_CALL_START.len() {
                return None;
            }
            if rest.starts_with(STRING_DELIM) {
                self.resync_in_string = !self.resync_in_string;
                self.resync_cursor += STRING_DELIM.len();
                count_boundary_bytes(STRING_DELIM.len());
                continue;
            }
            if !self.resync_in_string && cursor > 0 && rest.starts_with(TOOL_CALL_START) {
                let after_marker = &rest[TOOL_CALL_START.len()..];
                if CALL_PREFIX.starts_with(after_marker) {
                    return None;
                }
                if let Some(after_prefix) = after_marker.strip_prefix(CALL_PREFIX)
                    && after_prefix.find('{').is_none()
                    && after_prefix
                        .chars()
                        .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.'))
                {
                    return None;
                }
                if let Some(after_prefix) = after_marker.strip_prefix(CALL_PREFIX)
                    && let Some(name_len) = after_prefix
                        .char_indices()
                        .find_map(|(at, ch)| (ch == '{' && at > 0).then_some(at))
                    && after_prefix[..name_len]
                        .chars()
                        .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.'))
                {
                    self.resync_candidate = Some((cursor, 1));
                    let consumed = TOOL_CALL_START.len() + CALL_PREFIX.len() + name_len + 1;
                    self.resync_cursor += consumed;
                    count_boundary_bytes(consumed);
                    continue;
                }
            }
            if let Some((start, depth)) = self.resync_candidate {
                if depth == 0 {
                    if rest.starts_with(TOOL_CALL_END) {
                        return Some(start);
                    }
                    if TOOL_CALL_END.starts_with(rest) {
                        return None;
                    }
                    if !rest.chars().next()?.is_whitespace() {
                        self.resync_candidate = None;
                    }
                } else if !self.resync_in_string {
                    self.resync_candidate = match rest.chars().next()? {
                        '{' => Some((start, depth + 1)),
                        '}' => Some((start, depth - 1)),
                        _ => self.resync_candidate,
                    };
                }
            }
            let consumed = rest.chars().next()?.len_utf8();
            self.resync_cursor += consumed;
            count_boundary_bytes(consumed);
        }
        flush
            .then(|| {
                self.resync_candidate
                    .and_then(|(start, depth)| (depth == 0).then_some(start))
            })
            .flatten()
    }

    fn reset(&mut self) {
        self.progress.reset();
        self.candidate.clear();
        self.resync_cursor = 0;
        self.resync_in_string = false;
        self.resync_candidate = None;
    }
}

fn gemma4_invoke_boundary() -> Box<dyn InvokeBoundary> {
    Box::new(Gemma4InvokeBoundary::default())
}

/// Build the Gemma 4 scanner. ONE construction site, shared by the tool-only
/// parser below and by the unified parser, so the grammar cannot drift between
/// the two surfaces.
pub(crate) fn gemma4_scanner(tools: &[Tool]) -> WrappedBlockScanner<Gemma4InvokeEmitter> {
    WrappedBlockScanner::new(
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
            // prefix has grammar-aware holdback through `Gemma4InvokeBoundary`.
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
            invoke_boundary_factory: Some(InvokeBoundaryFactory::custom(gemma4_invoke_boundary)),
            preserve_special_tokens: true,
        },
        Gemma4InvokeEmitter {
            tools: tools.iter().map(ToolDefinition::from).collect(),
        },
    )
}

/// Whether the `call:` at `at` opens a real call rather than being the English
/// word: it must sit on an identifier boundary AND look like a call start to the
/// candidate-local grammar probe (`call:NAME{`).
///
/// Without the second test, "I will call: you tomorrow" would be buffered as an
/// invoke that never closes and then dropped at EOF — losing ordinary prose.
fn opens_bare_call(text: &str, at: usize) -> bool {
    is_call_prefix_boundary(text, at) && has_bare_call_body_start_gemma4(&text[at..])
}

pub(crate) fn is_gemma_call_prefix_boundary(text: &str, at: usize) -> bool {
    is_call_prefix_boundary(text, at)
}

/// Stream parser for Gemma 4 tool calls.
pub struct Gemma4ToolStreamParser {
    scanner: WrappedBlockScanner<Gemma4InvokeEmitter>,
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

#[cfg(test)]
mod boundary_tests {
    use super::*;

    #[test]
    fn gemma_exposes_its_request_local_boundary() {
        let scanner = gemma4_scanner(&[]);
        let factory = scanner
            .invoke_boundary_factory()
            .expect("Gemma needs a grammar-aware boundary factory");
        let boundary = factory.create();

        assert!(boundary.opens("call:get_weather{}", 0));
        assert_eq!(boundary.holdback("ordinary text"), 0);
    }

    #[test]
    fn resync_advances_only_over_new_bytes_and_recovers_a_balanced_eof_candidate() {
        let mut boundary = Gemma4InvokeBoundary::default();
        let mut input = "<|tool_call>call:broken{note:<|\"|>unterminated".to_string();

        assert_eq!(boundary.resync(&input, false, 0), None);
        let examined_after_first_push = boundary.resync_cursor;
        assert_eq!(examined_after_first_push, input.len());

        input.push_str("<|\"|>}<|tool_call>call:get_weather{city:<|\"|>NYC<|\"|>}");
        assert_eq!(boundary.resync(&input, false, 0), None);
        assert_eq!(boundary.resync_cursor, input.len());
        assert!(boundary.resync_cursor > examined_after_first_push);

        // The later candidate is balanced but truncated before the wrapper close.
        // At EOF it is the safe forward recovery point, without revisiting bytes
        // from either preceding push.
        assert_eq!(
            boundary.resync(&input, true, 0),
            Some(input.rfind(TOOL_CALL_START).unwrap())
        );
    }

    #[test]
    fn resync_examines_each_byte_once_across_sequential_chunk_sizes() {
        let stream = concat!(
            "<|tool_call>call:broken{note:<|\"|>unterminated<|\"|>}",
            "<|tool_call>call:get_weather{city:<|\"|>NYC<|\"|>}<|tool_call|>"
        );

        for chunk_size in [1, 4, 16] {
            let mut boundary = Gemma4InvokeBoundary::default();
            let mut input = String::new();
            let mut examined = 0;
            for chunk in stream.as_bytes().chunks(chunk_size) {
                input.push_str(std::str::from_utf8(chunk).expect("ASCII test input"));
                let before = boundary.resync_cursor;
                let _ = boundary.resync(&input, false, 0);
                examined += boundary.resync_cursor - before;
            }
            assert_eq!(examined, stream.len(), "chunk_size={chunk_size}");
            let _ = boundary.resync(&input, true, 0);

            boundary.reset();
            assert_eq!(
                boundary.resync("<|tool_call>call:get_weather{}<|tool_call|>", true, 0),
                None
            );
        }
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

/// Trailing run that may still grow into a recoverable bare call but that the
/// v1 probes cannot detect yet: a partial `call:` prefix at a word boundary
/// (`c`, `ca`, ..., `call:`), or a complete boundary `call:` followed only by
/// identifier characters (the function name, awaiting its `{`). Once the `{`
/// arrives, the scanner's candidate-local opener probe recognizes it directly.
/// Held-back bytes are flushed on the next chunk (or at EOF), so the
/// concatenated output is unchanged — only its chunk boundaries shift.
fn partial_bare_opener_suffix_len(text: &str) -> usize {
    for len in (1..=CALL_PREFIX.len()).rev() {
        if text.ends_with(&CALL_PREFIX[..len]) && is_call_prefix_boundary(text, text.len() - len) {
            return len;
        }
    }
    if let Some(idx) = text.rfind(CALL_PREFIX)
        && is_call_prefix_boundary(text, idx)
        && text[idx + CALL_PREFIX.len()..]
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.'))
    {
        return text.len() - idx;
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
    fn later_balanced_call_with_trailing_whitespace_is_recovered_at_eof() {
        let input = concat!(
            "<|tool_call>call:broken{note:<|\"|>unterminated",
            "<|\"|>}<|tool_call>call:get_weather{location:<|\"|>NYC<|\"|>}   \n\t"
        );
        for chunk_size in [1, 4, 16] {
            let chunks = input
                .as_bytes()
                .chunks(chunk_size)
                .map(|chunk| std::str::from_utf8(chunk).expect("ASCII test input"))
                .collect::<Vec<_>>();
            let out = parse_chunks(&weather_tools(), &chunks);
            assert_eq!(out.normal_text, "   \n\t", "chunk_size={chunk_size}");
            let merged = out.coalesce_calls();
            assert_eq!(merged.calls.len(), 1, "chunk_size={chunk_size}");
            assert_eq!(merged.calls[0].name.as_deref(), Some("get_weather"));
            assert_eq!(merged.calls[0].arguments, r#"{"location":"NYC"}"#);
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
