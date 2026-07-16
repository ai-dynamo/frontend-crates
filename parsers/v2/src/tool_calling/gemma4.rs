// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Streaming tool-call parser for Gemma 4.
//!
//! Gemma 4 emits tool calls with a custom non-JSON, non-XML grammar:
//!   `<|tool_call>call:NAME{key:<|"|>value<|"|>, key2:42, key3:[...]}<tool_call|>`
//! with bare unquoted keys, `<|"|>`-delimited strings, nested objects/arrays, and
//! MULTIPLE calls concatenated with NO separator. START = `<|tool_call>`,
//! END = `<tool_call|>` (asymmetric).
//!
//! The streaming concern (buffering, chunk-split marker safety, normal_text
//! suppression, EOF-truncation drop) is owned here. Per-block name + value typing
//! is delegated to the v1 batch parser `try_tool_call_parse_gemma4`, so a streamed
//! call matches exactly what the batch parser produces. A call is emitted only
//! once its complete `<|tool_call> ... <tool_call|>` block has streamed; an
//! incomplete trailing block at EOF is DROPPED (v1 parity), not emitted with empty
//! arguments.
//!
//! Block-completion detection (where the `<tool_call|>` end marker actually closes
//! a call, ignoring a bare `<tool_call|>` literal embedded inside a `<|"|>` string
//! value) reuses the v1 helper `find_tool_call_end_position_gemma4` rather than
//! re-implementing the balanced-brace + string-delimiter scan, so the streaming
//! and batch paths share one block boundary definition.
//!
//! Arguments are re-serialized in source key order because the v1 parser builds
//! them from a `serde_json::Map` (a `BTreeMap` without the `preserve_order`
//! feature), which sorts keys alphabetically; the fixtures store arguments as an
//! exact JSON string in the model-emitted order (the order vLLM's Rust parser also
//! preserves), so order has to be pinned to source order.

use std::collections::HashSet;

use crate::tool_calling::v1core::ToolDefinition;
use crate::tool_calling::v1core::gemma4::{
    detect_tool_call_start_gemma4, find_tool_call_end_position_gemma4, try_tool_call_parse_gemma4,
};

use crate::tool_calling::traits::{Tool, ToolCallDelta, ToolParseResult, ToolParser};

const TOOL_CALL_START: &str = "<|tool_call>";
const CALL_PREFIX: &str = "call:";
const STRING_DELIM: &str = "<|\"|>";

/// Which kind of opener started the call span currently being buffered.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Block {
    /// A wrapped call opened by a literal `<|tool_call>` marker (still in the
    /// buffer as the block's leading marker).
    Wrapped,
    /// A bare call opened by a boundary `call:` token with NO `<|tool_call>`
    /// opener (the "orphan close" / "missing start" recovery shape). The v1 batch
    /// parser recovers these; the streaming parser does too, so the call body is
    /// recovered as a tool call instead of leaking as normal_text. Genuine prose
    /// before the `call:` token has already been emitted as normal_text before
    /// the block was entered.
    Bare,
}

/// Stream parser for Gemma 4 tool calls.
pub struct Gemma4ToolStreamParser {
    buffer: String,
    /// `Some(kind)` once a call opener (`<|tool_call>` or a recoverable bare
    /// `call:`) has been entered and we are buffering the call body until its
    /// complete `<tool_call|>` end marker arrives. While set, intra-call bytes
    /// never leak to `normal_text`; suppression is scoped to this in-block window
    /// (narration after a call is preserved, matching the engines).
    block: Option<Block>,
    /// Stable parser-local tool index, incremented per emitted call.
    next_index: usize,
    tools: Vec<ToolDefinition>,
}

impl Gemma4ToolStreamParser {
    pub fn new(tools: &[Tool]) -> Self {
        Self {
            buffer: String::new(),
            block: None,
            next_index: 0,
            tools: tools.iter().map(ToolDefinition::from).collect(),
        }
    }

    fn drain(&mut self, flush: bool) -> anyhow::Result<ToolParseResult> {
        let mut out = ToolParseResult::default();

        loop {
            if let Some(kind) = self.block {
                // Inside a call span. For a wrapped block the leading `<|tool_call>`
                // is still in the buffer; for a bare block the buffer starts at the
                // `call:` token. Find the end of the FIRST complete `}<tool_call|>`
                // block (the v1 helper respects brace balance + `<|"|>` strings, so
                // an embedded `<tool_call|>` literal inside a string value does not
                // close the block early). The helper returns the position after the
                // LAST complete block, so isolating the buffer to a single leading
                // block (next opener onward held back) keeps us emitting one call
                // per iteration with the correct per-call source order.
                let next_opener = self.next_opener_after_current(kind);
                let scan = match next_opener {
                    Some(s) => &self.buffer[..s],
                    None => &self.buffer[..],
                };
                match find_tool_call_end_position_gemma4(scan) {
                    Some(end) => {
                        let block = self.buffer[..end].to_string();
                        self.buffer.drain(..end);
                        self.block = None;
                        self.emit_block(&block, kind, &mut out)?;
                        continue;
                    }
                    None => {
                        // No complete block in the leading segment. If a later
                        // opener exists, the leading block is malformed/incomplete
                        // (e.g. missing end before the next call); drop it and
                        // continue scanning from the next opener.
                        if let Some(s) = next_opener {
                            tracing::warn!(
                                why = "gemma4_incomplete_tool_call",
                                next_index = self.next_index,
                                "Gemma 4 stream dropped an incomplete leading block before the next call"
                            );
                            self.buffer.drain(..s);
                            self.block = None; // re-enter on the next opener below
                        } else if flush {
                            // EOF with no `<tool_call|>` end marker. The streamv2
                            // conformance tab grades dynamo's stream output against
                            // dynamo's own v1 BATCH output, and the v1 batch parser
                            // recovers a call whose body is complete even when the end
                            // marker never streamed (batch 5.a/5.d). Match it:
                            // delegate the recover-or-drop decision to the v1 parser
                            // via emit_block. A complete body is recovered; a
                            // genuinely incomplete body (truncated mid-value, 5.c)
                            // yields no call and is dropped (emit_block logs it). The
                            // buffered markup never leaks to normal_text either way.
                            let block = std::mem::take(&mut self.buffer);
                            self.block = None;
                            self.emit_block(&block, kind, &mut out)?;
                            break;
                        } else {
                            break;
                        }
                        // Fall through to the not-in-block opener scan.
                    }
                }
            }

            // Not in a block: find the next opener — either a literal `<|tool_call>`
            // marker, or a bare boundary `call:` token that is a recoverable call
            // (orphan-close / missing-start shape). Whichever comes first wins.
            let wrapped_pos = self.buffer.find(TOOL_CALL_START);
            let bare_pos = self.first_bare_call_opener();
            let opener = match (wrapped_pos, bare_pos) {
                (Some(w), Some(b)) if w <= b => Some((w, Block::Wrapped)),
                (Some(_), Some(b)) => Some((b, Block::Bare)),
                (Some(w), None) => Some((w, Block::Wrapped)),
                (None, Some(b)) => Some((b, Block::Bare)),
                (None, None) => None,
            };

            match opener {
                Some((start, kind)) => {
                    // Text before the opener is user-visible normal_text. For the
                    // bare case this is the genuine prose prefix (e.g. 5.g's
                    // "I will check that. "), which must stay normal_text.
                    if start > 0 {
                        out.normal_text.push_str(&self.buffer[..start]);
                    }
                    // Keep the opener bytes in the buffer; the in-block branch needs
                    // them as the block's leading content for the v1 parser and for
                    // next-opener detection.
                    self.buffer.drain(..start);
                    self.block = Some(kind);
                }
                None => {
                    // No complete opener present. Hold back a trailing partial
                    // opener (a `<|tool_call>` start marker OR a `call:` token that
                    // may still grow into a recoverable bare call) split across this
                    // chunk boundary. At flush, an in-progress bare call that never
                    // completed is DROPPED (truncation parity) rather than leaked as
                    // normal_text; its preceding prose is still emitted.
                    let pending_bare = pending_bare_call_suffix_len(&self.buffer);
                    let keep = if flush {
                        pending_bare
                    } else {
                        self.opener_holdback_len()
                    };
                    let emit_len = self.buffer.len().saturating_sub(keep);
                    if emit_len > 0 {
                        out.normal_text.push_str(&self.buffer[..emit_len]);
                        self.buffer.drain(..emit_len);
                    }
                    if flush && pending_bare > 0 {
                        tracing::warn!(
                            why = "gemma4_incomplete_tool_call",
                            next_index = self.next_index,
                            "Gemma 4 stream truncated an incomplete bare call at EOF; dropping it (v1 parity)"
                        );
                        self.buffer.clear();
                    }
                    break;
                }
            }
        }

        Ok(out)
    }

    /// Position of the next opener strictly after the current leading block, used
    /// to bound the end-scan to a single block. For a wrapped block, the current
    /// block starts at the leading `<|tool_call>`; for a bare block it starts at
    /// the leading `call:`. The next opener is the earliest of a later
    /// `<|tool_call>` marker or a later recoverable bare `call:`.
    fn next_opener_after_current(&self, kind: Block) -> Option<usize> {
        let skip = match kind {
            Block::Wrapped => TOOL_CALL_START.len(),
            Block::Bare => CALL_PREFIX.len(),
        };
        if skip > self.buffer.len() {
            return None;
        }
        let tail = &self.buffer[skip..];
        let wrapped = tail.find(TOOL_CALL_START).map(|rel| rel + skip);
        // A bare `call:` at the very start of a WRAPPED block's tail is the
        // block's OWN body (`<|tool_call>call:...`), not a next call. Without
        // this skip, every complete wrapped block truncated its own end-scan
        // and detoured through drop + bare-recovery (correct output, but two
        // misleading warnings per call and a dead Wrapped emit path).
        let mut bare = first_bare_call_opener_in(tail);
        if kind == Block::Wrapped && bare == Some(0) {
            bare = first_bare_call_opener_in(&tail[CALL_PREFIX.len()..])
                .map(|rel| rel + CALL_PREFIX.len());
        }
        let bare = bare.map(|rel| rel + skip);
        match (wrapped, bare) {
            (Some(w), Some(b)) => Some(w.min(b)),
            (Some(w), None) => Some(w),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        }
    }

    /// First boundary `call:` in the buffer that begins a recoverable bare call
    /// (its complete `}<tool_call|>` close has streamed). A boundary `call:` whose
    /// body has not yet completed is NOT returned here (it is held back by
    /// `opener_holdback_len` until it completes or EOF drops it), so an in-progress
    /// body is never leaked as normal_text.
    fn first_bare_call_opener(&self) -> Option<usize> {
        first_bare_call_opener_in(&self.buffer)
    }

    /// Bytes to hold back from the tail when no complete opener is present: the
    /// longest of a trailing partial `<|tool_call>` prefix, or a trailing boundary
    /// `call:`-prefixed run that `detect_tool_call_start_gemma4` thinks may still
    /// grow into a tool call (so a bare call whose end marker has not yet arrived
    /// is buffered, not leaked).
    fn opener_holdback_len(&self) -> usize {
        let partial_start = start_marker_suffix_len(&self.buffer);
        let partial_bare = pending_bare_call_suffix_len(&self.buffer);
        partial_start.max(partial_bare)
    }

    /// Parse one complete call block into a delta, delegating name + value typing
    /// to the v1 batch parser. For a wrapped block the input includes its leading
    /// `<|tool_call>` start marker; for a bare block it starts at `call:` (the v1
    /// parser's missing-start recovery handles the absent opener). Both end with
    /// the closing `<tool_call|>`.
    fn emit_block(
        &mut self,
        block: &str,
        kind: Block,
        out: &mut ToolParseResult,
    ) -> anyhow::Result<()> {
        let (calls, _content) = try_tool_call_parse_gemma4(block, Some(&self.tools))?;
        // A complete-but-malformed block (no `call:NAME{...}` body, e.g. the
        // `<|tool_call>nonsense<tool_call|>` recovery case) yields zero calls;
        // drop it without leaking markup, matching the v1 no-leak contract.
        let Some(call) = calls.into_iter().next() else {
            tracing::warn!(
                why = "gemma4_block_without_call",
                "Gemma 4 stream dropped a complete block that produced no call"
            );
            return Ok(());
        };
        if kind == Block::Bare {
            tracing::warn!(
                why = "gemma4_bare_call_recovery",
                tool_index = self.next_index,
                "Gemma 4 stream recovered a bare call (no <|tool_call> opener) instead of leaking it as normal_text"
            );
        }
        let arguments = reorder_arguments(&call.function.arguments, block);
        out.calls.push(ToolCallDelta {
            tool_index: self.next_index,
            name: Some(call.function.name),
            arguments,
        });
        self.next_index += 1;
        Ok(())
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
        true
    }

    fn push(&mut self, chunk: &str) -> anyhow::Result<ToolParseResult> {
        self.buffer.push_str(chunk);
        self.drain(false)
    }

    fn finish(&mut self) -> anyhow::Result<ToolParseResult> {
        self.drain(true)
    }
}

/// Longest non-empty proper prefix of the `<|tool_call>` start marker that `text`
/// ends with, so a marker split across chunk boundaries is held back instead of
/// leaked as text. The `<|"|>` string delimiter and `<tool_call|>` end marker also
/// start with `<`, but only `<|tool_call>` opens a call from the not-in-block
/// state, so holding back its prefixes is sufficient.
fn start_marker_suffix_len(text: &str) -> usize {
    TOOL_CALL_START
        .char_indices()
        .map(|(idx, _)| idx)
        .filter(|idx| *idx > 0 && *idx < TOOL_CALL_START.len())
        .rev()
        .find(|&len| text.ends_with(&TOOL_CALL_START[..len]))
        .unwrap_or(0)
}

/// True if `s` begins (at `idx`) on a `call:` word boundary — the preceding char
/// is not part of an identifier, so `"call:"` at the start of a word counts but
/// the `call:` inside `"recall:..."` does not. Mirrors the v1 parser's
/// `is_call_prefix_boundary`.
fn is_call_boundary(s: &str, idx: usize) -> bool {
    idx == 0
        || s[..idx]
            .chars()
            .next_back()
            .is_none_or(|c| !(c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.')))
}

/// Position of the first boundary `call:` token in `text` that begins a COMPLETE
/// recoverable bare call (its `}<tool_call|>` close has streamed). Reuses the v1
/// helper `find_tool_call_end_position_gemma4` to confirm the candidate completes
/// (and to reject a `call:` that is just prose, e.g. "I will call: you"). An
/// in-progress bare call (no end marker yet) returns `None` here; it is held back
/// by `pending_bare_call_suffix_len` until it completes or EOF drops it.
fn first_bare_call_opener_in(text: &str) -> Option<usize> {
    let mut cursor = 0usize;
    while let Some(rel) = text[cursor..].find(CALL_PREFIX) {
        let idx = cursor + rel;
        if is_call_boundary(text, idx) && find_tool_call_end_position_gemma4(&text[idx..]).is_some()
        {
            return Some(idx);
        }
        cursor = idx + CALL_PREFIX.len();
    }
    None
}

/// Bytes to hold back from the tail of `text` for a boundary `call:` run that may
/// still grow into a recoverable bare call. Finds the last boundary `call:` whose
/// suffix `detect_tool_call_start_gemma4` still considers a tool-call start (a
/// complete or in-progress `call:NAME{...}` shape) but whose end marker has not
/// yet arrived (`find_tool_call_end_position_gemma4` is `None`), and returns the
/// number of trailing bytes from that `call:` onward so they are buffered, not
/// leaked. Returns 0 when no such pending bare call is present.
fn pending_bare_call_suffix_len(text: &str) -> usize {
    let mut cursor = 0usize;
    let mut best = 0usize;
    while let Some(rel) = text[cursor..].find(CALL_PREFIX) {
        let idx = cursor + rel;
        if is_call_boundary(text, idx) {
            let suffix = &text[idx..];
            // Hold back only an in-progress bare call (looks like a tool-call start
            // but has not completed yet); a completed one is handled as an opener.
            if find_tool_call_end_position_gemma4(suffix).is_none()
                && detect_tool_call_start_gemma4(suffix)
            {
                best = best.max(suffix.len());
            }
        }
        cursor = idx + CALL_PREFIX.len();
    }
    best
}

/// Re-serialize a v1 arguments JSON object in source key order. The v1 parser
/// emits alphabetically-sorted keys (`serde_json::Map` is a `BTreeMap` here); the
/// fixtures want the model-emitted order, so we read the top-level key order
/// directly from the call body and rebuild the JSON in that order.
fn reorder_arguments(arguments: &str, block: &str) -> String {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(arguments) else {
        return arguments.to_string();
    };
    let Some(obj) = value.as_object() else {
        return arguments.to_string();
    };
    let mut parts: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for name in source_key_order(block) {
        if let Some(val) = obj.get(&name) {
            parts.push(format!(
                "{}:{}",
                serde_json::to_string(&name).unwrap_or_default(),
                serde_json::to_string(val).unwrap_or_default()
            ));
            seen.insert(name);
        }
    }
    // Append any keys not matched in source order (defensive; normally empty).
    for (key, val) in obj {
        if !seen.contains(key) {
            parts.push(format!(
                "{}:{}",
                serde_json::to_string(key).unwrap_or_default(),
                serde_json::to_string(val).unwrap_or_default()
            ));
        }
    }
    format!("{{{}}}", parts.join(","))
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
    fn drops_call_truncated_mid_value() {
        let out = parse_chunks(
            &weather_tools(),
            &["<|tool_call>call:get_weather{location:<|\"|>", "N"],
        );
        assert_eq!(out.normal_text, "");
        assert!(out.calls.is_empty());
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
