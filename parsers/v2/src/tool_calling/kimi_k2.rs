// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Streaming tool-call parser for Kimi K2.
//!
//! Kimi K2 emits tool calls as
//!   `<|tool_calls_section_begin|>`
//!     `<|tool_call_begin|>functions.NAME:IDX<|tool_call_argument_begin|>{JSON}<|tool_call_end|>`
//!     ... (one or more calls)
//!   `<|tool_calls_section_end|>`
//! The model may also emit singular section variants
//! (`<|tool_call_section_begin|>` / `<|tool_call_section_end|>`), and may drop
//! `section_end` entirely on max_tokens / EOS truncation.
//!
//! The streaming concern (buffering, chunk-split marker safety, normal_text
//! suppression) is owned by the shared [`scan::WrappedBlockScanner`]; the K2
//! grammar maps onto it with section variants as multi-token block markers,
//! the inner `call_end`/`argument_begin` markers as extra orphan markers, and
//! two K2-specific spec fields: the suppression latch engages after every
//! in-section call parse even when the call is malformed (`InvokeLatch::Always`),
//! and a call whose `call_end` never arrives before the section close is
//! dropped rather than swallowing the fence (`drop_invoke_crossing_block_end`).
//!
//! The per-call typing (function-id parsing, JSON validation, raw-string
//! fallback for malformed args) is delegated to the v1 batch parser
//! `try_tool_call_parse_kimi_k2` driven by the same `KimiK2ParserConfig`
//! `dynamo_parsers` uses for batch parsing. A complete call is wrapped in the section
//! markers before delegating so the v1 parser always takes its normal section
//! path.
//!
//! Native streaming preserves valid JSON source order, duplicate keys, and internal
//! whitespace even when the entire call arrives in one push. KimiProgress finalizes
//! both buffered and provisional calls without rewriting previously released bytes;
//! malformed calls without provisional output retain the batch parser's fallback.

use crate::tool_calling::scan::{
    BareRecoveryLatch, InvokeBoundary, InvokeBoundaryFactory, InvokeEmitter, InvokeLatch,
    WrappedBlockScanner, WrappedBlockSpec, json_value_end,
};
use crate::tool_calling::v1core::{
    KimiK2ParserConfig, ToolDefinition, try_tool_call_parse_kimi_k2,
};

use crate::tool_calling::json_prefix::JsonPrefixState;
use crate::tool_calling::kimi_progress::KimiProgress;
use crate::tool_calling::traits::{Tool, ToolCallDelta, ToolParseResult, ToolParser};

// Native markers mirror the default batch grammar used by both Kimi entry points.
const CALL_START: &str = "<|tool_call_begin|>";
const CALL_END: &str = "<|tool_call_end|>";
const ARGUMENT_BEGIN: &str = "<|tool_call_argument_begin|>";

// Mirrors `KimiK2ParserConfig::default().section_end_variants` for the same
// reason as the consts above -- `K2Boundary` needs to recognize a real
// section close to distinguish it from genuine EOS truncation (see its use
// below).
const SECTION_END_PLURAL: &str = "<|tool_calls_section_end|>";
const SECTION_END_SINGULAR: &str = "<|tool_call_section_end|>";

const FUNCTIONS_PREFIX: &str = "functions.";

/// Bytes valid in a `NAME:IDX` identifier, per `get_id_regex`'s `[\w.\-]+`.
fn ident_char(c: char) -> bool {
    c.is_alphanumeric() || matches!(c, '.' | '_' | '-')
}

/// Result of scanning for a `NAME:IDX` id at the start of a buffer.
enum NativeId {
    /// A complete id, with its length. At least one digit follows `:` --
    /// more digits streaming in later would only extend it, and every
    /// length already satisfies `\d+`, so there is no ambiguity to wait out.
    Complete(usize),
    /// What's buffered so far could still grow into a complete id (more
    /// name bytes, the `:` itself, or its digits) with more input.
    Pending,
    /// Terminates in a way that rules out `NAME:IDX` ever matching here.
    None,
}

/// Scan `text` for a complete `NAME:IDX` id (mirrors `get_id_regex`'s
/// `[\w.\-]+:\d+`), distinguishing "never going to match" from "not
/// determinable yet from what's buffered" -- the latter must wait for more
/// input rather than being treated as a bare, unindexed name. The
/// batch-path regex this mirrors is permissive by design (an unindexed name
/// is still valid prose, not a malformed id), which is exactly why this
/// cannot default to `None` just because a terminator hasn't streamed yet.
fn native_id_len(text: &str, flush: bool) -> NativeId {
    NativeIdCursor::default().advance(text, flush)
}

#[derive(Default)]
struct NativeIdCursor {
    cursor: usize,
    colon: Option<usize>,
    end: Option<usize>,
    invalid: bool,
}

impl NativeIdCursor {
    fn advance(&mut self, text: &str, flush: bool) -> NativeId {
        if self.invalid {
            return NativeId::None;
        }
        if let Some(end) = self.end {
            return NativeId::Complete(end);
        }
        for ch in text[self.cursor..].chars() {
            #[cfg(test)]
            crate::tool_calling::kimi_progress::count_work(1, ch.len_utf8());
            match self.colon {
                None if ident_char(ch) => {}
                None if ch == ':' && self.cursor > 0 => self.colon = Some(self.cursor),
                Some(_) if ch.is_ascii_digit() => {}
                Some(colon) if self.cursor > colon + 1 => {
                    self.end = Some(self.cursor);
                    return NativeId::Complete(self.cursor);
                }
                _ => {
                    self.invalid = true;
                    return NativeId::None;
                }
            }
            self.cursor += ch.len_utf8();
        }
        if self.colon.is_some_and(|colon| self.cursor > colon + 1) {
            NativeId::Complete(self.cursor)
        } else if flush {
            NativeId::None
        } else {
            NativeId::Pending
        }
    }
}

#[derive(Default)]
struct K2HeaderCursor {
    searched: usize,
    args_at: Option<usize>,
    native_id: NativeIdCursor,
    whitespace_at: usize,
    stopped_at: Option<(usize, bool)>,
    identity_rejected: bool,
}

impl K2HeaderCursor {
    fn arguments(&mut self, text: &str) -> Option<usize> {
        if self.args_at.is_some() || self.stopped_at.is_some() {
            return self.args_at;
        }
        self.searched = self.searched.max(CALL_START.len());
        while self.searched < text.len() {
            let tail = &text[self.searched..];
            let ch = tail.chars().next().expect("nonempty header");
            #[cfg(test)]
            crate::tool_calling::kimi_progress::count_work(1, ch.len_utf8());
            if ch == '<' {
                let markers = [ARGUMENT_BEGIN, CALL_START, CALL_END];
                if let Some(marker) = markers.iter().find(|marker| tail.starts_with(**marker)) {
                    if *marker == ARGUMENT_BEGIN {
                        self.args_at = Some(self.searched + marker.len());
                    } else {
                        // A closer or sibling opener prevents a later argument
                        // marker from borrowing this invocation's identity.
                        self.stopped_at = Some((self.searched, *marker == CALL_END));
                    }
                    break;
                }
                if markers.iter().any(|marker| marker.starts_with(tail)) {
                    break;
                }
            }
            self.searched += ch.len_utf8();
        }
        self.args_at
    }

    fn end_without_arguments(&mut self, text: &str, flush: bool) -> Option<usize> {
        if flush && let Some((at, true)) = self.stopped_at {
            return Some(at + CALL_END.len());
        }
        let end = match self.native_id.advance(&text[CALL_START.len()..], flush) {
            NativeId::Pending => return None,
            NativeId::Complete(len) => CALL_START.len() + len,
            NativeId::None if text[CALL_START.len()..].starts_with(FUNCTIONS_PREFIX) => {
                CALL_START.len() + FUNCTIONS_PREFIX.len()
            }
            NativeId::None => CALL_START.len(),
        };
        self.whitespace_at = self.whitespace_at.max(end);
        let tail = &text[self.whitespace_at..];
        #[cfg(test)]
        crate::tool_calling::kimi_progress::count_work(1, tail.len() - tail.trim_start().len());
        self.whitespace_at += tail.len() - tail.trim_start().len();
        let remainder = &text[self.whitespace_at..];
        if !flush
            && (remainder.starts_with(CALL_END)
                || [ARGUMENT_BEGIN, CALL_END]
                    .iter()
                    .any(|marker| marker.starts_with(remainder)))
        {
            None
        } else {
            Some(end)
        }
    }
}

/// Locate the end of one Kimi invoke (`call_start .. call_end`) by finding
/// where its JSON argument body actually closes first, rather than searching
/// the raw buffer for `call_end` from byte zero.
///
/// Two things follow from that ordering:
/// - A `call_end`-looking byte sequence embedded INSIDE the JSON string
///   argument is data, not the real closer (`UNIFIED.7-2`,
///   `arg_marker_in_string`) — the naive whole-buffer search matched the
///   embedded copy first and truncated the argument there.
/// - At true EOF (`flush`), a body whose JSON is syntactically complete but
///   whose `call_end` never streamed (max_tokens / EOS) is recoverable
///   (`UNIFIED.5-2`, `tool_no_close`, the same best-effort-recovery contract
///   as policy P2) instead of being dropped as if it were genuinely
///   truncated. `K2Emitter` synthesizes the missing closer before typing it.
///   BUT only for `tool_index == 0`: the captured batch contract
///   (`TOOLCALLING.batch.5.d`/`TOOLCALLING.streamv1.5.d`, independently
///   pinned against the real `parse_section_block` regex and its own
///   streaming golden capture) drops a later call that never gets its own
///   `call_end`, while still recovering an incomplete FIRST call at EOF
///   (`TOOLCALLING.batch.5.a`/`UNIFIED.tool_no_close`). This mirrors that
///   observed asymmetry; it is not a claim about model reliability beyond
///   what these two fixtures establish. `tool_index` is the same monotonic
///   per-stream counter `WrappedBlockScanner` already tracks for
///   `tool_call_id`.
#[cfg(test)]
fn kimi_invoke_end(text: &str, flush: bool, tool_index: usize) -> Option<usize> {
    K2Boundary::default().end_append(text, text, flush, tool_index)
}

/// Kimi's `call_start` marker is unambiguous wherever it appears; every
/// occurrence opens a real invoke (same effective behavior as the
/// marker-only path this hook replaces).
fn kimi_invoke_opens(_text: &str, _at: usize) -> bool {
    true
}

/// No additional holdback beyond the generic marker holdback: `call_end` and
/// `argument_begin` are already in `holdback_markers`, which already retains
/// a partial marker split across a chunk boundary.
fn kimi_invoke_holdback(_text: &str) -> usize {
    0
}

#[derive(Default)]
struct K2Boundary {
    header: K2HeaderCursor,
    args_at: Option<usize>,
    cursor: usize,
    json: JsonPrefixState,
    json_started: bool,
    json_end: Option<usize>,
    invoke_end: Option<usize>,
    recovery: K2RecoveryBoundary,
    next_call: Option<usize>,
    call_end: Option<usize>,
}

#[derive(Default)]
struct K2RecoveryBoundary {
    cursor: Option<usize>,
    quoted: crate::tool_calling::scan::JsonStringState,
    next_call: Option<usize>,
    raw_end: Option<usize>,
    structural_end: Option<usize>,
    raw_section: Option<usize>,
    structural_section: Option<usize>,
}

impl K2RecoveryBoundary {
    fn end(&mut self, text: &str, args_at: usize, flush: bool) -> Option<usize> {
        let mut cursor = self.cursor.unwrap_or(args_at);
        while cursor < text.len() {
            let tail = &text[cursor..];
            let ch = tail.chars().next().expect("nonempty tail");
            let markers = [
                CALL_START,
                CALL_END,
                SECTION_END_PLURAL,
                SECTION_END_SINGULAR,
            ];
            if !flush
                && ch == '<'
                && markers
                    .iter()
                    .any(|marker| tail.len() < marker.len() && marker.starts_with(tail))
            {
                break;
            }
            #[cfg(test)]
            crate::tool_calling::kimi_progress::count_work(1, ch.len_utf8());
            let quoted = self.quoted.advance(ch);
            if ch == '<' {
                if tail.starts_with(CALL_START) {
                    self.next_call.get_or_insert(cursor);
                } else if tail.starts_with(CALL_END) {
                    self.raw_end.get_or_insert(cursor);
                    if !quoted {
                        self.structural_end.get_or_insert(cursor);
                    }
                } else if tail.starts_with(SECTION_END_PLURAL)
                    || tail.starts_with(SECTION_END_SINGULAR)
                {
                    self.raw_section.get_or_insert(cursor);
                    if !quoted {
                        self.structural_section.get_or_insert(cursor);
                    }
                }
            }
            cursor += ch.len_utf8();
        }
        self.cursor = Some(cursor);
        let structural = self
            .structural_end
            .filter(|end| self.next_call.is_none_or(|next| *end < next));
        let raw = self
            .raw_end
            .filter(|end| flush || self.next_call.is_some_and(|next| *end < next));
        let (end, section) = match structural {
            Some(end) => (end, self.structural_section),
            None => (raw?, self.raw_section),
        };
        if self.next_call.is_some_and(|next| next <= end)
            || section.is_some_and(|section| section < end)
        {
            return None;
        }
        Some(end + CALL_END.len())
    }
}

fn kimi_boundary() -> Box<dyn InvokeBoundary> {
    Box::<K2Boundary>::default()
}

impl InvokeBoundary for K2Boundary {
    fn end_append(
        &mut self,
        text: &str,
        _append: &str,
        flush: bool,
        index: usize,
    ) -> Option<usize> {
        if let Some(end) = self.invoke_end {
            return Some(end);
        }
        let args_at = match self.args_at {
            Some(at) => at,
            None => {
                let Some(at) = self.header.arguments(text) else {
                    return self.header.end_without_arguments(text, flush);
                };
                self.args_at = Some(at);
                self.cursor = at;
                at
            }
        };
        if self.json_end.is_none() && !self.json.invalid {
            for (offset, ch) in text[self.cursor..].char_indices() {
                #[cfg(test)]
                crate::tool_calling::kimi_progress::count_work(1, ch.len_utf8());
                let end = self.cursor + offset + ch.len_utf8();
                if !self.json_started {
                    if ch.is_whitespace() {
                        continue;
                    }
                    self.json = JsonPrefixState::new(ch);
                    self.json_started = true;
                } else {
                    self.json.consume(ch);
                }
                if self.json.complete {
                    if serde_json::from_str::<serde_json::Value>(&text[args_at..end]).is_ok() {
                        self.json_end = Some(end);
                    } else {
                        self.json.invalid = true;
                    }
                    break;
                }
                if self.json.invalid {
                    break;
                }
            }
            self.cursor = self.json_end.unwrap_or(text.len());
        }
        if let Some(json_end) = self.json_end {
            let tail = &text[self.cursor..];
            #[cfg(test)]
            crate::tool_calling::kimi_progress::count_work(1, tail.len() * 2);
            if self.call_end.is_none() {
                self.call_end = tail.find(CALL_END).map(|at| self.cursor + at);
            }
            if self.next_call.is_none() {
                self.next_call = tail.find(CALL_START).map(|at| self.cursor + at);
            }
            self.cursor = text.len()
                - crate::tool_calling::scan::marker_prefix_suffix_len(tail, [CALL_END, CALL_START]);
            if let Some(end) = self
                .call_end
                .filter(|end| self.next_call.is_none_or(|next| *end < next))
            {
                self.invoke_end = Some(end + CALL_END.len());
            } else if flush && index == 0 {
                let remainder = text[json_end..].trim_start();
                if ![SECTION_END_PLURAL, SECTION_END_SINGULAR]
                    .iter()
                    .any(|marker| remainder.starts_with(marker))
                {
                    self.invoke_end = Some(json_end);
                }
            }
            return self.invoke_end;
        }
        if self.json.invalid || flush {
            self.invoke_end = self.recovery.end(text, args_at, flush);
            return self.invoke_end;
        }
        None
    }
    fn opens(&self, text: &str, at: usize) -> bool {
        kimi_invoke_opens(text, at)
    }
    fn holdback(&self, text: &str) -> usize {
        kimi_invoke_holdback(text)
    }
    fn resync(&mut self, _text: &str, _flush: bool, _index: usize) -> Option<usize> {
        None
    }
    fn reset(&mut self) {
        *self = Self::default();
    }
}

fn spec(config: &KimiK2ParserConfig) -> WrappedBlockSpec {
    // Orphan markers: inner markers (`call_end`, `argument_begin`) and every
    // section-end variant only appear legitimately inside an open section;
    // outside one they are stray grammar markup to be stripped. Mirrors the v1
    // batch parser's `first_orphan_kimi_marker_index` (minus `call_start`,
    // which the bare-call recovery path already opens).
    let mut orphan_markers = vec![config.call_end.clone(), config.argument_begin.clone()];
    orphan_markers.extend(config.section_end_variants.clone());

    // Every grammar marker that must never be split-leaked as normal_text.
    let mut holdback_markers = config.section_start_variants.clone();
    holdback_markers.extend(config.section_end_variants.clone());
    holdback_markers.push(config.call_start.clone());
    holdback_markers.push(config.call_end.clone());
    holdback_markers.push(config.argument_begin.clone());

    WrappedBlockSpec {
        family: "kimi_k2",
        block_starts: config.section_start_variants.clone(),
        block_ends: config.section_end_variants.clone(),
        invoke_start: config.call_start.clone(),
        invoke_end: config.call_end.clone(),
        orphan_markers,
        holdback_markers,
        bare_recovery_latch: BareRecoveryLatch::Set,
        invoke_latch: InvokeLatch::Always,
        drop_invoke_crossing_block_end: true,
        // Every wrapped family's markers are special tokens today.
        preserve_special_tokens: true,
        invoke_boundary_factory: Some(InvokeBoundaryFactory::custom(kimi_boundary)),
    }
}

#[cfg(test)]
mod boundary_tests {
    use super::*;

    #[test]
    fn kimi_exposes_its_request_local_boundary_as_family_metadata() {
        let scanner = kimi_k2_scanner(&[]);
        let factory = scanner
            .invoke_boundary_factory()
            .expect("Kimi needs grammar-aware boundary callbacks");
        let boundary = factory.create();

        assert!(boundary.opens(CALL_START, 0));
        assert_eq!(boundary.holdback("ordinary text"), 0);
    }
}

/// Completed invokes use the v1 section path for identity and malformed typing.
/// KimiProgress preserves native JSON bytes and emits only the unreleased suffix.
pub(crate) struct K2Emitter {
    header: K2HeaderCursor,
    config: KimiK2ParserConfig,
    tools: Vec<ToolDefinition>,
    /// Native `functions.NAME:IDX` id per `tool_index`, for
    /// [`InvokeEmitter::tool_call_id`]. The v1core parser already extracts
    /// this id (`ToolCallResponse::id`) to resolve the function name; Kimi's
    /// envelope is the only wrapped grammar that NAMES the call this way, so
    /// this is the one family that needs to remember it past `parse_invoke`.
    native_ids: Vec<Option<String>>,
    partial: Option<(usize, KimiProgress, String)>,
}

impl InvokeEmitter for K2Emitter {
    fn parse_partial_invoke(
        &mut self,
        invoke: &str,
        tool_index: usize,
    ) -> anyhow::Result<Option<ToolCallDelta>> {
        if self.header.identity_rejected {
            return Ok(None);
        }
        if self.partial.is_none() {
            let Some(args_at) = self.header.arguments(invoke) else {
                return Ok(None);
            };
            let at = args_at - ARGUMENT_BEGIN.len();
            let header = invoke[CALL_START.len()..at].trim();
            if !matches!(native_id_len(header, true), NativeId::Complete(len) if len == header.len())
            {
                self.header.identity_rejected = true;
                return Ok(None);
            }
            let name = header
                .rsplit_once(':')
                .expect("validated native ID")
                .0
                .strip_prefix(FUNCTIONS_PREFIX)
                .filter(|name| !name.is_empty())
                .unwrap_or(header.rsplit_once(':').unwrap().0)
                .to_string();
            self.partial = Some((
                at + ARGUMENT_BEGIN.len(),
                KimiProgress::new(tool_index, name),
                header.to_string(),
            ));
        }
        let (at, progress, id) = self.partial.as_mut().expect("initialized above");
        if !progress.published() {
            *at += invoke[*at..].len() - invoke[*at..].trim_start().len();
        }
        let mut markers = vec![
            self.config.call_end.as_str(),
            self.config.call_start.as_str(),
        ];
        markers.extend(self.config.section_end_variants.iter().map(String::as_str));
        let delta = progress.advance_json_with_markers(&invoke[*at..], &markers, true);
        if delta.is_some() {
            self.native_ids.resize(tool_index + 1, None);
            self.native_ids[tool_index] = Some(id.clone());
        }
        Ok(delta)
    }

    fn abandon_invoke(&mut self) {
        self.header = K2HeaderCursor::default();
        self.partial = None;
    }

    fn parse_invoke(
        &mut self,
        invoke: &str,
        tool_index: usize,
    ) -> anyhow::Result<Option<ToolCallDelta>> {
        self.header = K2HeaderCursor::default();
        // The boundary may hand back a call recovered at EOF whose JSON
        // body is complete but whose `call_end` never streamed (`UNIFIED.5-2`).
        // Normalize it here: the regex-based v1 parser requires the literal
        // closer to delimit the arguments capture, so synthesize it rather
        // than re-feeding the raw, still-unclosed bytes.
        let synthesized;
        let invoke = if invoke.ends_with(self.config.call_end.as_str()) {
            invoke
        } else {
            synthesized = format!("{invoke}{}", self.config.call_end);
            synthesized.as_str()
        };
        let wrapped = format!(
            "{}{}{}",
            self.config.section_start, invoke, self.config.section_end
        );
        let (calls, _content) =
            try_tool_call_parse_kimi_k2(&wrapped, &self.config, Some(&self.tools))?;
        let Some(parsed) = calls.into_iter().next() else {
            return Ok(None);
        };
        // Published provisional calls reserve their index even when abandoned,
        // so later identities must retain any gaps in the scanner's index sequence.
        if self.native_ids.len() <= tool_index {
            self.native_ids.resize(tool_index + 1, None);
        }
        self.native_ids[tool_index] = Some(parsed.id);
        let partial = self.partial.take();
        let at = partial.as_ref().map(|(at, _, _)| *at).or_else(|| {
            invoke
                .find(ARGUMENT_BEGIN)
                .map(|at| at + ARGUMENT_BEGIN.len())
        });
        if let Some(at) = at {
            let raw = invoke[at..]
                .strip_suffix(self.config.call_end.as_str())
                .unwrap_or(&invoke[at..])
                .trim_start();
            let valid_end = json_value_end(raw)
                .filter(|end| serde_json::from_str::<serde_json::Value>(&raw[..*end]).is_ok());
            if valid_end.is_some()
                || partial
                    .as_ref()
                    .is_some_and(|(_, progress, _)| progress.published())
            {
                let mut progress = partial
                    .map(|(_, progress, _)| progress)
                    .unwrap_or_else(|| KimiProgress::new(tool_index, parsed.function.name.clone()));
                return Ok(progress.finish_json(raw[..valid_end.unwrap_or(raw.len())].trim_end()));
            }
        }
        Ok(Some(ToolCallDelta {
            tool_index,
            name: Some(parsed.function.name),
            arguments: parsed.function.arguments,
            complete: true,
        }))
    }

    fn tool_call_id(&self, tool_index: usize) -> Option<&str> {
        self.native_ids.get(tool_index)?.as_deref()
    }

    fn reset(&mut self) {
        self.header = K2HeaderCursor::default();
        self.native_ids.clear();
        self.partial = None;
    }
}

/// Stream parser for Kimi K2 tool calls.
pub struct KimiK2ToolStreamParser {
    scanner: WrappedBlockScanner<K2Emitter>,
}

/// Build the Kimi K2 marker scanner for one stream.
///
/// Extracted so the tool-only parser and the unified adapter share ONE scanner
/// construction. Two constructions would be two grammars that drift.
pub(crate) fn kimi_k2_scanner(tools: &[Tool]) -> WrappedBlockScanner<K2Emitter> {
    let config = KimiK2ParserConfig::default();
    WrappedBlockScanner::new(
        spec(&config),
        K2Emitter {
            header: K2HeaderCursor::default(),
            config,
            tools: tools.iter().map(ToolDefinition::from).collect(),
            native_ids: Vec::new(),
            partial: None,
        },
    )
}

impl KimiK2ToolStreamParser {
    pub fn new(tools: &[Tool]) -> Self {
        Self {
            scanner: kimi_k2_scanner(tools),
        }
    }
}

impl ToolParser for KimiK2ToolStreamParser {
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

    fn tool_call_id(&self, tool_index: usize) -> Option<&str> {
        self.scanner.tool_call_id(tool_index)
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
        let mut parser = KimiK2ToolStreamParser::new(tools);
        let mut out = ToolParseResult::default();
        for chunk in chunks {
            out.append(parser.push(chunk).expect("push"));
        }
        out.append(parser.finish().expect("finish"));
        out.coalesce_calls()
    }

    fn assert_native_header_schedules(header: &str, expected_name: Option<&str>) {
        let input = format!(
            "<|tool_calls_section_begin|>{CALL_START}{header}{ARGUMENT_BEGIN}{{}}{CALL_END}{SECTION_END_PLURAL}"
        );
        let (batch, _) =
            try_tool_call_parse_kimi_k2(&input, &KimiK2ParserConfig::default(), None).unwrap();
        assert_eq!(
            batch.len(),
            usize::from(expected_name.is_some()),
            "{header:?}"
        );
        if let Some(name) = expected_name {
            assert_eq!(batch[0].function.name, name, "{header:?}");
            assert_eq!(batch[0].function.arguments, "{}");
        }
        let whole = parse_chunks(&[], &[&input]);
        let mut schedules: Vec<Vec<&str>> = input
            .char_indices()
            .map(|(at, _)| vec![&input[..at], &input[at..]])
            .collect();
        schedules.push(vec![&input]);
        schedules.push(
            input
                .char_indices()
                .map(|(at, ch)| &input[at..at + ch.len_utf8()])
                .collect(),
        );
        for chunks in schedules {
            let mut parser = KimiK2ToolStreamParser::new(&[]);
            for _ in 0..2 {
                let mut output = ToolParseResult::default();
                for chunk in &chunks {
                    output.append(parser.push(chunk).unwrap());
                }
                output = output.coalesce_calls();
                // These inputs have a real closing marker: completion must not wait for EOF.
                assert_eq!(output.calls.len(), batch.len(), "{header:?}, {chunks:?}");
                for (index, (call, expected)) in output.calls.iter().zip(&batch).enumerate() {
                    assert_eq!(call.name.as_deref(), Some(expected.function.name.as_str()));
                    assert_eq!(call.arguments, expected.function.arguments);
                    assert!(call.complete);
                    assert_eq!(call.tool_index, index);
                    assert_eq!(parser.tool_call_id(index), Some(expected.id.as_str()));
                }
                output.append(parser.finish().unwrap());
                assert_eq!(output, whole, "{header:?}, {chunks:?}");
                parser.scanner.reset();
                assert_eq!(parser.tool_call_id(0), None);
            }
        }
    }

    #[test]
    fn native_header_empty_suffix_keeps_full_identifier_at_every_split() {
        assert_native_header_schedules("functions.:17", Some("functions."));
    }

    #[test]
    fn native_header_optional_prefix_and_index_match_batch_at_every_split() {
        for prefix in ["", FUNCTIONS_PREFIX] {
            for name in ["f", "_", "-", ".", "f_g", "f-g", "f.g", "é"] {
                for index in ["0", "17", "0017"] {
                    for whitespace in ["", " ", "\n\t"] {
                        assert_native_header_schedules(
                            &format!("{prefix}{name}:{index}{whitespace}"),
                            Some(name),
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn native_header_malformed_colon_and_index_remain_rejected() {
        for header in [
            "",
            ":17",
            "functions.",
            "functions.:",
            "functions.::17",
            "functions.f",
            "functions.f:",
            "functions.f:x",
            "functions.f:-1",
            "functions.f:17x",
            "functions.f: 17",
            "functions.f::17",
            "f:",
            "f:x",
            "f:-1",
            "f:17x",
            "f: 17",
            "f::17",
        ] {
            assert_native_header_schedules(header, None);
        }
    }

    #[test]
    fn hardcoded_markers_mirror_the_config_default() {
        // The native boundary and batch typing must agree on structural markers.
        let config = KimiK2ParserConfig::default();
        assert_eq!(config.call_start, CALL_START);
        assert_eq!(config.call_end, CALL_END);
        assert_eq!(config.argument_begin, ARGUMENT_BEGIN);
        assert_eq!(
            config.section_end_variants,
            vec![
                SECTION_END_PLURAL.to_string(),
                SECTION_END_SINGULAR.to_string()
            ],
        );
    }

    #[test]
    fn native_tool_call_id_is_delegated_from_the_scanner() {
        let input = concat!(
            "<|tool_calls_section_begin|>",
            "<|tool_call_begin|>functions.get_weather:7",
            "<|tool_call_argument_begin|>{\"location\":\"NYC\"}",
            "<|tool_call_end|><|tool_calls_section_end|>"
        );
        let mut parser = KimiK2ToolStreamParser::new(&weather_tools());
        let result = parser.parse_complete(input).expect("parse complete");
        assert_eq!(result.calls.len(), 1);
        assert_eq!(result.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(parser.tool_call_id(0), Some("functions.get_weather:7"));
        assert_eq!(parser.tool_call_id(1), None);
    }

    #[test]
    fn native_tool_call_id_state_resets_for_reuse() {
        let input = concat!(
            "<|tool_calls_section_begin|>",
            "<|tool_call_begin|>functions.get_weather:7",
            "<|tool_call_argument_begin|>{\"location\":\"NYC\"}",
            "<|tool_call_end|><|tool_calls_section_end|>"
        );
        let mut parser = KimiK2ToolStreamParser::new(&weather_tools());
        parser.parse_complete(input).expect("first parse");
        assert_eq!(parser.tool_call_id(0), Some("functions.get_weather:7"));
        parser.scanner.reset();
        assert_eq!(parser.tool_call_id(0), None);
        parser.parse_complete(input).expect("second parse");
        assert_eq!(parser.tool_call_id(0), Some("functions.get_weather:7"));
    }

    #[test]
    fn section_end_marker_embedded_in_a_string_argument_is_data_not_a_boundary() {
        // A well-formed call whose own JSON string argument happens to
        // contain the literal bytes of the section-end marker (e.g. echoing
        // a shell command) must NOT be mistaken for the real block boundary.
        // The shared `drop_invoke_crossing_block_end` safety net used a raw,
        // non-string-aware search that matched this embedded copy and
        // dropped the whole call, leaking its JSON tail as garbage text and
        // corrupting the `tool_index` of the following call (`next_index`
        // never advanced for the dropped one).
        let out = parse_chunks(
            &weather_tools(),
            &[
                "<|tool_calls_section_begin|><|tool_call_begin|>functions.run:0<|tool_call_argument_begin|>",
                "{\"cmd\":\"echo <|tool_calls_section_end|>\"}<|tool_call_end|>",
                "<|tool_call_begin|>functions.get_weather:1<|tool_call_argument_begin|>{\"location\":\"NYC\"}<|tool_call_end|><|tool_calls_section_end|>",
            ],
        );
        assert_eq!(out.normal_text, "");
        let merged = out.coalesce_calls();
        assert_eq!(merged.calls.len(), 2);
        assert_eq!(merged.calls[0].tool_index, 0);
        assert_eq!(merged.calls[0].name.as_deref(), Some("run"));
        assert_eq!(
            merged.calls[0].arguments,
            r#"{"cmd":"echo <|tool_calls_section_end|>"}"#
        );
        assert_eq!(merged.calls[1].tool_index, 1);
        assert_eq!(merged.calls[1].name.as_deref(), Some("get_weather"));
        assert_eq!(merged.calls[1].arguments, r#"{"location":"NYC"}"#);
    }

    #[test]
    fn emits_complete_call_on_close() {
        let out = parse_chunks(
            &weather_tools(),
            &[
                "<|tool_calls_section_begin|><|tool_call_begin|>",
                "functions.get_weather:0<|tool_call_argument_begin|>",
                "{\"location\":\"NYC\"}<|tool_call_end|><|tool_calls_section_end|>",
            ],
        );
        assert_eq!(out.normal_text, "");
        let merged = out.coalesce_calls();
        assert_eq!(merged.calls.len(), 1);
        assert_eq!(merged.calls[0].tool_index, 0);
        assert_eq!(merged.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(merged.calls[0].arguments, r#"{"location":"NYC"}"#);
    }

    /// Direct unit test on the boundary finder itself, bypassing the
    /// scanner/typing pipeline entirely -- the pipeline-level symptom of this
    /// bug is easy to mask by accident (the downstream regex's own
    /// `captures_iter` happens to skip a malformed prefix and still find the
    /// valid second call on its own, and content between two invokes in one
    /// section is suppressed by unrelated, correct `InvokeLatch::Always`
    /// behavior regardless of this fix). The boundary VALUE is the actual
    /// contract: a bare `NAME:IDX<|tool_call_end|>` invoke with no
    /// `argument_begin` at all must bound to itself, not reach across a
    /// second invoke's `call_start` to grab that invoke's `argument_begin`.
    #[test]
    fn invoke_end_does_not_reach_across_a_bare_close_to_a_later_argument_begin() {
        let text = "<|tool_call_begin|>functions.run:0<|tool_call_end|><|tool_call_begin|>functions.get_weather:1<|tool_call_argument_begin|>{\"city\": \"Paris\"}<|tool_call_end|>";
        // The correct boundary is the bare invoke's OWN call_end, not just
        // its id -- that call_end genuinely belongs to it, and stopping
        // there (rather than reaching into the second invoke) is what makes
        // `find(CALL_END)` in the fallback above correctly resolve this case
        // on its own, without ever needing the native-id path below it.
        let bare_invoke_with_own_close = "<|tool_call_begin|>functions.run:0<|tool_call_end|>";
        assert_eq!(
            kimi_invoke_end(text, true, 0),
            Some(bare_invoke_with_own_close.len()),
            "must bound to the bare invoke's own close, not span through the second invoke's call_end"
        );
    }

    /// Sibling of the test above: a bare invoke that has NEITHER its own
    /// `argument_begin` NOR its own `call_end` -- its id text runs straight
    /// into a second invoke's `call_start`. The `argument_begin` bound only
    /// checked for an intervening `call_end`; this shape has none, so it was
    /// unguarded and the second invoke's `argument_begin`/JSON/`call_end`
    /// still got attributed to the first, merging both spans -- masked from
    /// producing a visibly wrong result only by the downstream regex's
    /// forgiving `captures_iter` and the JSON-argument branch's own
    /// `CALL_START` bound, not by this check being correct.
    #[test]
    fn invoke_end_does_not_reach_across_a_bare_open_with_no_close_at_all() {
        let text = "<|tool_call_begin|>functions.run:0<|tool_call_begin|>functions.get_weather:1<|tool_call_argument_begin|>{\"city\": \"Paris\"}<|tool_call_end|>";
        let bare_invoke_id_only = "<|tool_call_begin|>functions.run:0";
        assert_eq!(
            kimi_invoke_end(text, true, 0),
            Some(bare_invoke_id_only.len()),
            "must bound to the bare invoke's id alone (it has no close of its own), \
             not span through the second invoke's call_end"
        );
    }

    #[test]
    fn emits_two_calls_in_one_section() {
        let tools = vec![
            Tool {
                name: "get_weather".to_string(),
                description: None,
                parameters: serde_json::json!({"type": "object"}),
                strict: None,
            },
            Tool {
                name: "get_time".to_string(),
                description: None,
                parameters: serde_json::json!({"type": "object"}),
                strict: None,
            },
        ];
        let out = parse_chunks(
            &tools,
            &[
                "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"location\":\"NYC\"}<|tool_call_end|>",
                "<|tool_call_begin|>functions.get_time:1<|tool_call_argument_begin|>{\"timezone\":\"EST\"}<|tool_call_end|><|tool_calls_section_end|>",
            ],
        );
        let merged = out.coalesce_calls();
        assert_eq!(merged.calls.len(), 2);
        assert_eq!(merged.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(merged.calls[0].arguments, r#"{"location":"NYC"}"#);
        assert_eq!(merged.calls[1].name.as_deref(), Some("get_time"));
        assert_eq!(merged.calls[1].arguments, r#"{"timezone":"EST"}"#);
    }

    #[test]
    fn preserves_prefix_text_before_section() {
        let out = parse_chunks(
            &weather_tools(),
            &[
                "I will",
                " check the weather. <|tool_calls_section_begin|>",
                "<|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"location\":\"NYC\"}<|tool_call_end|><|tool_calls_section_end|>",
            ],
        );
        assert_eq!(out.normal_text, "I will check the weather. ");
        assert_eq!(out.coalesce_calls().calls.len(), 1);
    }

    #[test]
    fn preserves_post_section_narration() {
        let out = parse_chunks(
            &weather_tools(),
            &[
                "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"location\":\"NYC\"}<|tool_call_end|><|tool_calls_section_end|>",
                " Done.",
            ],
        );
        // In-section markup is suppressed; post-section narration is preserved
        // verbatim once the section closes (v1 batch parity, cases 8.b/8.c).
        assert_eq!(out.normal_text, " Done.");
        assert_eq!(out.coalesce_calls().calls.len(), 1);
    }

    #[test]
    fn preserves_inter_section_narration() {
        // Two sections separated by narration (case 8.d): the prefix and the
        // inter-section text both flow into normal_text; both calls are emitted.
        let tools = vec![
            Tool {
                name: "get_weather".to_string(),
                description: None,
                parameters: serde_json::json!({"type": "object"}),
                strict: None,
            },
            Tool {
                name: "get_time".to_string(),
                description: None,
                parameters: serde_json::json!({"type": "object"}),
                strict: None,
            },
        ];
        let out = parse_chunks(
            &tools,
            &[
                "First. <|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"location\":\"NYC\"}<|tool_call_end|><|tool_calls_section_end|>",
                " Then. <|tool_calls_section_begin|><|tool_call_begin|>functions.get_time:1<|tool_call_argument_begin|>{\"timezone\":\"EST\"}<|tool_call_end|><|tool_calls_section_end|>",
            ],
        );
        assert_eq!(out.normal_text, "First.  Then. ");
        let merged = out.coalesce_calls();
        assert_eq!(merged.calls.len(), 2);
        assert_eq!(merged.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(merged.calls[1].name.as_deref(), Some("get_time"));
    }

    #[test]
    fn holds_back_marker_split_across_every_char() {
        // Worst case: the whole input arrives one fragment at a time, splitting
        // every grammar marker. No partial marker may leak into normal_text.
        let full = "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"location\":\"NYC\"}<|tool_call_end|><|tool_calls_section_end|>";
        let chunks: Vec<&str> = full
            .as_bytes()
            .chunks(3)
            .map(|c| std::str::from_utf8(c).unwrap())
            .collect();
        let out = parse_chunks(&weather_tools(), &chunks);
        assert_eq!(out.normal_text, "");
        let merged = out.coalesce_calls();
        assert_eq!(merged.calls.len(), 1);
        assert_eq!(merged.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(merged.calls[0].arguments, r#"{"location":"NYC"}"#);
    }

    #[test]
    fn suppresses_truncated_call_at_eof() {
        // Section + call header streamed, but no call_end / section_end before
        // EOF. The truncated call is dropped and no markup leaks.
        let out = parse_chunks(
            &weather_tools(),
            &[
                "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>",
                "{\"location\":\"NY",
            ],
        );
        assert_eq!(out.normal_text, "");
        assert!(out.calls.is_empty());
    }

    #[test]
    fn mismatched_fences_drop_the_call_instead_of_recovering_it() {
        // `TOOLCALLING.batch.4.d` (sourced from vLLM's own kimi_k2 parser
        // tests): the JSON body is syntactically complete, but the model
        // closes the whole tool_calls section (`section_end`) without ever
        // giving this call its own `call_end`. This is NOT the same as
        // running out of tokens mid-call (`suppresses_truncated_call_at_eof`)
        // or completing right at EOF with nothing else following
        // (`UNIFIED.tool_no_close`, `TOOLCALLING.streamv1.5.a`) -- a real
        // section-end marker DOES follow, so the model had more to say and
        // chose not to close this call. Batch mode's regex requires a
        // literal `call_end` unconditionally and drops it; streaming must
        // match, or `conformance_toolcalling_batch_via_stream` diverges.
        let out = parse_chunks(
            &weather_tools(),
            &[
                "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>",
                "{\"location\":\"NYC\"}<|tool_calls_section_end|>",
            ],
        );
        assert_eq!(out.normal_text, "");
        assert!(out.calls.is_empty());
    }

    /// Reviewer-caught regression, sibling of the test above: dropping a
    /// call with mismatched fences is correct, but the ABOVE test ends
    /// right after `section_end` and so cannot detect a different bug --
    /// `WrappedBlockScanner::drain` used to treat `invoke_end_at`
    /// returning `None` at flush as ALWAYS "genuinely incomplete", clearing
    /// the entire remaining buffer including any real visible text that
    /// followed the section-end marker. Batch mode's regex-based
    /// extraction correctly preserves that trailing text; streaming
    /// dropped it too. The correct result is an empty call list AND the
    /// visible suffix preserved as `normal_text`.
    #[test]
    fn mismatched_fences_preserve_text_after_section_end() {
        let out = parse_chunks(
            &weather_tools(),
            &[
                "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{}<|tool_calls_section_end|>Visible answer",
            ],
        );
        assert!(out.calls.is_empty());
        assert_eq!(out.normal_text, "Visible answer");
    }

    /// Reviewer-caught regression: a bracket-balanced but JSON-GRAMMAR-INVALID
    /// body (`{not-json}` -- braces match, but it's not legal JSON: no
    /// quoted key, no colon, no value) with NO `call_end` anywhere in the
    /// buffer used to still recover a call at EOF, because `json_value_end`
    /// only proves bracket/quote nesting is balanced, not that the bytes
    /// parse as JSON. `parse_section_block`'s regex (batch mode) requires a
    /// literal `call_end` to match anything at all and so drops this input
    /// outright -- reproduced directly: streaming shipped a call with
    /// `arguments: "{not-json}"` while batch mode produced zero calls.
    #[test]
    fn eof_recovery_requires_actually_valid_json_not_just_balanced_brackets() {
        let text = "<|tool_call_begin|>functions.run:0<|tool_call_argument_begin|>{not-json}";
        assert_eq!(
            kimi_invoke_end(text, true, 0),
            None,
            "a bracket-balanced but grammatically invalid body with no call_end evidence \
             at all must not be recovered -- batch mode's regex can never match it either"
        );
    }

    /// Sibling positive control: the SAME shape but with genuinely valid
    /// JSON still recovers normally -- this fix must not regress the
    /// existing `UNIFIED.5-2`/`tool_no_close` best-effort recovery contract.
    #[test]
    fn eof_recovery_still_recovers_genuinely_valid_json_with_no_call_end() {
        let text =
            "<|tool_call_begin|>functions.run:0<|tool_call_argument_begin|>{\"city\": \"Paris\"}";
        assert_eq!(
            kimi_invoke_end(text, true, 0),
            Some(text.len()),
            "valid JSON with no call_end at true EOF must still be recovered"
        );
    }

    /// Reviewer-caught regression: a malformed (odd quote count) argument
    /// followed by a real section-end marker, with a literal `call_end`
    /// only appearing AFTER that section-end, used to still recover a call
    /// whose raw-string argument swallowed the section-end marker as text
    /// -- the same "mismatched fences" shape the well-formed-JSON path
    /// already guards against (see the `flush` block above this function's
    /// EOF-recovery gate), just reached through the malformed/raw-string
    /// fallback instead, which was missing the equivalent guard.
    #[test]
    fn malformed_argument_recovery_also_respects_an_intervening_section_end() {
        let text = "<|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"location\": \"unterminated<|tool_calls_section_end|><|tool_call_end|>";
        assert_eq!(
            kimi_invoke_end(text, true, 0),
            None,
            "a real section-end marker occurring before the only available call_end \
             means this call never got its own closer -- must drop, not swallow the \
             section-end into a malformed raw-string argument"
        );
    }

    /// Sibling positive control: the SAME malformed-argument raw-string
    /// fallback still recovers normally when no section-end marker
    /// intervenes -- this fix must not regress the raw-string recovery
    /// contract `malformed_non_json_arguments_still_ship_the_call_instead_of_vanishing`
    /// already covers end-to-end.
    #[test]
    fn malformed_argument_recovery_still_works_without_an_intervening_section_end() {
        let text = "<|tool_call_begin|>functions.run:0<|tool_call_argument_begin|>bad\"arg<|tool_call_end|>";
        assert_eq!(
            kimi_invoke_end(text, true, 0),
            Some(text.len()),
            "a malformed raw-string argument with its own real call_end and no \
             intervening section-end must still be recovered"
        );
    }

    #[test]
    fn closed_malformed_arguments_emit_on_the_closing_chunk() {
        let mut parser = KimiK2ToolStreamParser::new(&weather_tools());
        assert!(
            parser
                .push("<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>")
                .unwrap()
                .calls
                .is_empty()
        );

        let emitted = parser
            .push("{\"location\":\"NYC\"<|tool_call_end|><|tool_calls_section_end|>")
            .unwrap();
        assert_eq!(emitted.calls.len(), 1);
        assert_eq!(emitted.calls[0].arguments, r#"{"location":"NYC""#);
        assert!(parser.finish().unwrap().calls.is_empty());
    }

    #[test]
    fn closed_malformed_arguments_emit_before_finish_at_every_valid_utf8_split() {
        let input = "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"location\":\"München\"<|tool_call_end|><|tool_calls_section_end|>";
        for split in (0..=input.len()).filter(|&index| input.is_char_boundary(index)) {
            let mut parser = KimiK2ToolStreamParser::new(&weather_tools());
            let mut emitted = parser.push(&input[..split]).unwrap();
            emitted.append(parser.push(&input[split..]).unwrap());
            emitted = emitted.coalesce_calls();
            assert_eq!(
                emitted.calls.len(),
                1,
                "split at byte {split} must emit once the closer is available"
            );
            assert_eq!(emitted.calls[0].arguments, r#"{"location":"München""#);
            assert!(
                parser.finish().unwrap().calls.is_empty(),
                "split at byte {split} must not defer the call until finish"
            );
        }
    }

    fn assert_closed_malformed_quoted_marker_emits_on_the_closing_chunk(marker: &str) {
        let arguments = format!(r#"{{"location" "München {marker} literal"}}"#);
        let mut parser = KimiK2ToolStreamParser::new(&weather_tools());
        assert!(
            parser
                .push("<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>")
                .unwrap()
                .calls
                .is_empty()
        );

        let emitted = parser
            .push(&format!("{arguments}{CALL_END}{SECTION_END_PLURAL}"))
            .unwrap();
        assert_eq!(emitted.calls.len(), 1);
        assert_eq!(emitted.calls[0].arguments, arguments);
        assert!(parser.finish().unwrap().calls.is_empty());
    }

    #[test]
    fn malformed_quoted_call_end_is_data_and_emits_on_the_closing_chunk() {
        assert_closed_malformed_quoted_marker_emits_on_the_closing_chunk(CALL_END);
    }

    #[test]
    fn malformed_quoted_section_end_is_data_and_emits_on_the_closing_chunk() {
        assert_closed_malformed_quoted_marker_emits_on_the_closing_chunk(SECTION_END_PLURAL);
    }

    #[test]
    fn closed_malformed_quoted_markers_emit_before_finish_at_every_valid_utf8_split() {
        let header = "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>";
        for marker in [CALL_END, SECTION_END_PLURAL] {
            let arguments = format!(r#"{{"location" "München {marker} literal"}}"#);
            let input = format!("{header}{arguments}{CALL_END}{SECTION_END_PLURAL}");
            for split in (0..=input.len()).filter(|&index| input.is_char_boundary(index)) {
                let mut parser = KimiK2ToolStreamParser::new(&weather_tools());
                let mut emitted = parser.push(&input[..split]).unwrap();
                emitted.append(parser.push(&input[split..]).unwrap());
                emitted = emitted.coalesce_calls();
                assert_eq!(
                    emitted.calls.len(),
                    1,
                    "marker {marker:?}, split at byte {split} must emit once the closer is available"
                );
                assert_eq!(emitted.calls[0].arguments, arguments);
                assert!(
                    parser.finish().unwrap().calls.is_empty(),
                    "marker {marker:?}, split at byte {split} must not defer the call until finish"
                );
            }
        }
    }

    #[test]
    fn mismatched_fences_singular_section_variant_also_drops_the_call() {
        // Same shape as above, through the singular `<|tool_call_section_end|>`
        // variant -- the guard checks both `section_end_variants`, not just
        // the plural default.
        let out = parse_chunks(
            &weather_tools(),
            &[
                "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"location\":\"NYC\"}<|tool_call_section_end|>",
            ],
        );
        assert_eq!(out.normal_text, "");
        assert!(out.calls.is_empty());
    }

    #[test]
    fn multi_call_mismatched_fences_keeps_the_closed_call_drops_the_open_one() {
        // `TOOLCALLING.batch.5.d`-adjacent shape: a properly closed first
        // call followed by a second call whose JSON is complete but whose
        // `call_end` is missing, with a real section_end right after (unlike
        // `.5.d`, which has no section_end at all and is a genuine-truncation
        // case handled by `known-divergences.yaml`, not this guard). The
        // first call must still ship; only the malformed second is dropped.
        let out = parse_chunks(
            &weather_tools(),
            &[
                "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"location\":\"Boston\"}<|tool_call_end|>",
                "<|tool_call_begin|>functions.get_weather:1<|tool_call_argument_begin|>{\"location\":\"New York\"}<|tool_calls_section_end|>",
            ],
        );
        assert_eq!(out.normal_text, "");
        let merged = out.coalesce_calls();
        assert_eq!(merged.calls.len(), 1);
        assert_eq!(merged.calls[0].arguments, r#"{"location":"Boston"}"#);
    }

    #[test]
    fn strips_orphan_call_end_outside_section() {
        // A complete orphan `call_end` with no open section is stray double-close
        // markup: it must be stripped, never leaked, and the surrounding genuine
        // prose preserved (v1 `first_orphan_kimi_marker_index` parity).
        let out = parse_chunks(
            &weather_tools(),
            &["Here you go.", "<|tool_call_end|>", "All set."],
        );
        assert_eq!(out.normal_text, "Here you go.All set.");
        assert!(out.calls.is_empty());
    }

    #[test]
    fn strips_orphan_argument_begin_outside_section() {
        let out = parse_chunks(
            &weather_tools(),
            &["Here you go.", "<|tool_call_argument_begin|>", "All set."],
        );
        assert_eq!(out.normal_text, "Here you go.All set.");
        assert!(out.calls.is_empty());
    }

    #[test]
    fn strips_orphan_section_end_outside_section() {
        let out = parse_chunks(
            &weather_tools(),
            &["Here you go.", "<|tool_calls_section_end|>", "All set."],
        );
        assert_eq!(out.normal_text, "Here you go.All set.");
        assert!(out.calls.is_empty());
    }

    #[test]
    fn recovers_complete_bare_call_without_section() {
        let out = parse_chunks(
            &weather_tools(),
            &[
                "<|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>",
                "{\"location\":\"NYC\"}<|tool_call_end|>",
            ],
        );
        assert_eq!(out.normal_text, "");
        let merged = out.coalesce_calls();
        assert_eq!(merged.calls.len(), 1);
        assert_eq!(merged.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(merged.calls[0].arguments, r#"{"location":"NYC"}"#);
    }
}
