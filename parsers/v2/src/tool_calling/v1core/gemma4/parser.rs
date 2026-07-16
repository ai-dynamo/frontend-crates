// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Reference implementation:
// https://github.com/vllm-project/vllm/blob/main/vllm/tool_parsers/gemma4_tool_parser.py
//
// Gemma 4 tool-call grammar (custom, non-JSON):
//
//     <|tool_call>call:func_name{key:<|"|>value<|"|>,num:42}<tool_call|>
//
// `<|"|>`-delimited strings, bare unquoted keys, nested objects/arrays,
// multiple calls concatenated without separators.

use std::sync::OnceLock;

use regex::Regex;
use serde_json::{Map, Value};
use uuid::Uuid;

use super::super::ToolDefinition;
use super::super::response::{CalledFunction, ToolCallResponse, ToolCallType};

pub(crate) const TOOL_CALL_START: &str = "<|tool_call>";
pub(crate) const TOOL_CALL_END: &str = "<tool_call|>";
pub(crate) const STRING_DELIM: &str = "<|\"|>";
pub(crate) const CALL_PREFIX: &str = "call:";

static TOOL_CALL_REGEX: OnceLock<Regex> = OnceLock::new();

/// Captures the function-name + raw-args body of a single complete tool call.
/// `(?s)` enables dot-all so nested arg bodies that span newlines parse correctly.
fn tool_call_regex() -> &'static Regex {
    TOOL_CALL_REGEX.get_or_init(|| {
        let pattern = format!(
            r"(?s){}{}(?P<name>[\w\-\.]+)\{{(?P<args>.*?)\}}{}",
            regex::escape(TOOL_CALL_START),
            regex::escape(CALL_PREFIX),
            regex::escape(TOOL_CALL_END),
        );
        Regex::new(&pattern).expect("Failed to compile gemma4 tool call regex")
    })
}

fn parse_gemma_call_parts(
    name: &str,
    args_raw: &str,
    tools: Option<&[ToolDefinition]>,
) -> anyhow::Result<ToolCallResponse> {
    let name = name.to_string();
    if let Some(tools) = tools
        && !tools.iter().any(|t| t.name == name)
    {
        tracing::warn!(
            "Tool '{}' is not defined in the tools list (Gemma 4 parser).",
            name
        );
    }

    let args_value = match parse_args_object(args_raw) {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(
                "Failed to parse Gemma 4 args for '{}': {}. Falling back to empty object.",
                name,
                e
            );
            Value::Object(Map::new())
        }
    };
    let arguments = serde_json::to_string(&args_value)?;

    Ok(ToolCallResponse {
        id: format!("call-{}", Uuid::new_v4()),
        tp: ToolCallType::Function,
        function: CalledFunction { name, arguments },
    })
}

fn find_balanced_args_end(input: &str, open_brace: usize) -> Option<usize> {
    debug_assert_eq!(input.as_bytes().get(open_brace), Some(&b'{'));
    let mut cursor = open_brace;
    let mut depth = 0usize;
    let mut in_string = false;

    while cursor < input.len() {
        let rest = &input[cursor..];
        if rest.starts_with(STRING_DELIM) {
            in_string = !in_string;
            cursor += STRING_DELIM.len();
            continue;
        }

        let ch = rest.chars().next()?;
        if !in_string {
            match ch {
                '{' => depth += 1,
                '}' => {
                    depth = depth.checked_sub(1)?;
                    if depth == 0 {
                        return Some(cursor);
                    }
                }
                _ => {}
            }
        }
        cursor += ch.len_utf8();
    }

    None
}

fn parse_recoverable_call_at(
    input: &str,
    allow_missing_start: bool,
    allow_missing_end: bool,
) -> Option<(&str, &str, usize)> {
    let after_start_offset = if let Some(rest) = input.strip_prefix(TOOL_CALL_START) {
        input.len() - rest.len()
    } else if allow_missing_start && input.starts_with(CALL_PREFIX) {
        0
    } else {
        return None;
    };

    let after_start = &input[after_start_offset..];
    let after_prefix = after_start.strip_prefix(CALL_PREFIX)?;
    let name_len = after_prefix.find('{').filter(|idx| *idx > 0)?;
    let name = &after_prefix[..name_len];
    if !name
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.'))
    {
        return None;
    }

    let open_brace = after_start_offset + CALL_PREFIX.len() + name_len;
    let close_brace = find_balanced_args_end(input, open_brace)?;
    let args_start = open_brace + 1;
    let args_raw = &input[args_start..close_brace];
    let after_args = &input[close_brace + 1..];

    if after_args.starts_with(TOOL_CALL_END) {
        return Some((name, args_raw, close_brace + 1 + TOOL_CALL_END.len()));
    }

    if allow_missing_end && after_args.trim().is_empty() {
        return Some((name, args_raw, close_brace + 1));
    }

    None
}

fn is_call_prefix_boundary(input: &str, idx: usize) -> bool {
    idx == 0
        || input[..idx]
            .chars()
            .next_back()
            .is_none_or(|ch| !(ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.')))
}

fn find_call_prefix_at_boundary(input: &str, from: usize) -> Option<usize> {
    let mut cursor = from;
    while cursor < input.len() {
        let rel = input[cursor..].find(CALL_PREFIX)?;
        let idx = cursor + rel;
        if is_call_prefix_boundary(input, idx) {
            return Some(idx);
        }
        cursor = idx + CALL_PREFIX.len();
    }
    None
}

/// Detect whether `chunk` contains the start of a Gemma 4 tool call, including
/// partial-prefix matches at the chunk boundary so streaming pipelines can hold
/// off emitting bytes that may belong to a tool-call marker.
pub fn detect_tool_call_start_gemma4(chunk: &str) -> bool {
    if chunk.contains(TOOL_CALL_START) {
        return true;
    }

    let mut cursor = 0usize;
    while let Some(idx) = find_call_prefix_at_boundary(chunk, cursor) {
        let candidate = &chunk[idx..];
        if parse_recoverable_call_at(candidate, true, true).is_some()
            || has_bare_call_body_start(candidate)
        {
            return true;
        }
        cursor = idx + CALL_PREFIX.len();
    }

    for i in 1..TOOL_CALL_START.len() {
        if TOOL_CALL_START.is_char_boundary(i) && chunk.ends_with(&TOOL_CALL_START[..i]) {
            return true;
        }
    }

    false
}

fn has_bare_call_body_start(input: &str) -> bool {
    let Some(after_prefix) = input.strip_prefix(CALL_PREFIX) else {
        return false;
    };
    let Some(open_brace) = after_prefix.find('{') else {
        return false;
    };
    if open_brace == 0 {
        return false;
    }
    after_prefix[..open_brace]
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.'))
}

/// Returns the position immediately after the last *complete* tool-call match
/// in `chunk`, or `None` if no complete call has arrived yet (caller should
/// keep accumulating). The regex requires `}<tool_call|>` adjacency, so a bare
/// `<tool_call|>` literal embedded inside a `<|"|>` string value does not
/// false-trigger a "section complete" signal here — matches upstream.
pub fn find_tool_call_end_position_gemma4(chunk: &str) -> Option<usize> {
    let mut cursor = 0usize;
    let mut last_end = None;

    while cursor < chunk.len() {
        let next_start = chunk[cursor..]
            .find(TOOL_CALL_START)
            .map(|rel| (cursor + rel, false, TOOL_CALL_START.len()));
        let next_bare =
            find_call_prefix_at_boundary(chunk, cursor).map(|idx| (idx, true, CALL_PREFIX.len()));
        let Some((rel_start, allow_missing_start, marker_len)) = [next_start, next_bare]
            .into_iter()
            .flatten()
            .min_by_key(|(idx, _, _)| *idx)
        else {
            break;
        };

        if let Some((_, _, consumed)) =
            parse_recoverable_call_at(&chunk[rel_start..], allow_missing_start, false)
        {
            let mut end = rel_start + consumed;
            while chunk[end..].starts_with(TOOL_CALL_END) {
                end += TOOL_CALL_END.len();
            }
            last_end = Some(end);
            cursor = end;
        } else {
            cursor = rel_start + marker_len;
        }
    }

    last_end
}

fn push_recovered_call(
    calls: &mut Vec<ToolCallResponse>,
    first_tool_start: &mut Option<usize>,
    removed_spans: &mut Vec<(usize, usize)>,
    absolute_start: usize,
    recovered: (&str, &str, usize),
    tools: Option<&[ToolDefinition]>,
    reason: &'static str,
) -> anyhow::Result<()> {
    if first_tool_start.is_none_or(|idx| absolute_start < idx) {
        *first_tool_start = Some(absolute_start);
    }
    // Record the byte span of the recovered call so the normal_text assembler
    // can strip exactly this markup and keep the surrounding natural text.
    removed_spans.push((absolute_start, absolute_start + recovered.2));
    tracing::warn!(
        why = reason,
        recovered_calls = 1,
        recovered_bytes = recovered.2,
        "gemma4 recovery: recovered complete call body from damaged wrapper"
    );
    calls.push(parse_gemma_call_parts(recovered.0, recovered.1, tools)?);
    Ok(())
}

fn recover_calls_in_span(
    span: &str,
    span_offset: usize,
    allow_missing_end: bool,
    tools: Option<&[ToolDefinition]>,
    calls: &mut Vec<ToolCallResponse>,
    first_tool_start: &mut Option<usize>,
    removed_spans: &mut Vec<(usize, usize)>,
) -> anyhow::Result<()> {
    let mut cursor = 0usize;

    while cursor < span.len() {
        let next_start = span[cursor..]
            .find(TOOL_CALL_START)
            .map(|rel| (cursor + rel, false, TOOL_CALL_START.len()));
        let next_bare =
            find_call_prefix_at_boundary(span, cursor).map(|idx| (idx, true, CALL_PREFIX.len()));
        let Some((rel_start, allow_missing_start, marker_len)) = [next_start, next_bare]
            .into_iter()
            .flatten()
            .min_by_key(|(idx, _, _)| *idx)
        else {
            break;
        };

        let parsed = parse_recoverable_call_at(
            &span[rel_start..],
            allow_missing_start,
            allow_missing_end && !allow_missing_start,
        );

        if let Some(recovered) = parsed {
            let reason = if allow_missing_start {
                "missing_start_recovery"
            } else {
                "missing_end_recovery"
            };
            push_recovered_call(
                calls,
                first_tool_start,
                removed_spans,
                span_offset + rel_start,
                recovered,
                tools,
                reason,
            )?;
            cursor = rel_start + recovered.2;
        } else {
            cursor = rel_start + marker_len;
        }
    }

    Ok(())
}

/// Parse a Gemma 4 model response into structured tool calls + leftover text.
///
/// Returns `(parsed_tool_calls, normal_text_content)`. `normal_text` is the
/// model text with each complete tool-call block (start marker `<|tool_call>`
/// through end marker `<tool_call|>`) removed, keeping ALL other text:
/// the prefix before the first call, the text BETWEEN calls, and the text
/// AFTER the last call. Interior whitespace is preserved as-is (only the
/// outermost ends are trimmed, matching the rest of the parser family); only
/// tool-call markup is stripped, natural text is never dropped, and markup
/// never leaks.
///
/// This is a deliberate divergence from upstream vLLM
/// (`vllm/tool_parsers/gemma4_tool_parser.py::extract_tool_calls`, which keeps
/// only `model_output[:content_end].strip()`, i.e. the prefix before the first
/// marker) and from SGLang (which also drops trailing narration). Dynamo
/// preserves the surrounding narration so a model that sandwiches a tool call
/// between sentences doesn't silently lose the user-visible text. Cases:
/// TOOLCALLING.batch.{2.c, 8.b, 8.c, 8.d}.
///
/// Malformed / unrecoverable / truncated tool-call markup keeps its no-leak
/// drop behavior: a gap that still contains gemma4 markup tokens after the
/// complete calls are removed is suppressed rather than echoed verbatim.
///
/// The `}<tool_call|>` adjacency requirement in the regex means embedded
/// `<tool_call|>` literals inside string-typed args (e.g. a `description`
/// field documenting the tool-call format) don't truncate the match
/// prematurely.
pub fn try_tool_call_parse_gemma4(
    message: &str,
    tools: Option<&[ToolDefinition]>,
) -> anyhow::Result<(Vec<ToolCallResponse>, Option<String>)> {
    let regex = tool_call_regex();
    let mut calls = Vec::new();
    let mut first_tool_start = None;
    // Byte spans of every complete (parsed or recovered) tool-call block, used
    // to subtract markup from the assembled normal_text below.
    let mut removed_spans: Vec<(usize, usize)> = Vec::new();
    let mut cursor = 0usize;

    for caps in regex.captures_iter(message) {
        if let Some(m) = caps.get(0) {
            recover_calls_in_span(
                &message[cursor..m.start()],
                cursor,
                false,
                tools,
                &mut calls,
                &mut first_tool_start,
                &mut removed_spans,
            )?;
            first_tool_start.get_or_insert(m.start());
            removed_spans.push((m.start(), m.end()));
            cursor = m.end();
        }
        let name = caps.name("name").map(|m| m.as_str()).unwrap_or_default();
        if name.is_empty() {
            continue;
        }
        let args_raw = caps.name("args").map(|m| m.as_str()).unwrap_or("");

        calls.push(parse_gemma_call_parts(name, args_raw, tools)?);
    }

    recover_calls_in_span(
        &message[cursor..],
        cursor,
        true,
        tools,
        &mut calls,
        &mut first_tool_start,
        &mut removed_spans,
    )?;

    // `normal_text` assembly:
    //   - Success path (≥1 call extracted): concatenate the text outside every
    //     complete tool-call block — prefix, inter-call narration, and trailing
    //     narration — preserving whitespace verbatim. A gap that still contains
    //     gemma4 markup tokens (`<|tool_call>`, `<tool_call|>`, `<|"|>`) is a
    //     malformed / truncated / unrecoverable remnant; it is dropped so
    //     tool-call markup never leaks into normal_text.
    //   - Recovery path (zero calls extracted): if the message contains ANY
    //     gemma4 markup token, return empty — Dynamo intentionally diverges
    //     from vLLM's exception-fallback (which echoes raw bytes) so tool-call
    //     markup never leaks on malformed / truncated / orphan-close / no-body
    //     inputs. Cases flagged by the parity table's `↯`:
    //     TOOLCALLING.batch.{4.a, 4.b, 4.c, 4.d, 5.a, 5.b, 5.c, 6.c}.
    //   - Plain-text path (zero calls AND no markup): return the message as-is.
    let has_markup = message.contains(TOOL_CALL_START)
        || message.contains(TOOL_CALL_END)
        || message.contains(STRING_DELIM);
    let normal_text = if calls.is_empty() {
        if has_markup {
            // Recovery: malformed/truncated/orphan-close/no-body shapes.
            // Suppress the whole message so tool-call markup doesn't leak.
            let preview: String = message.chars().take(120).collect();
            tracing::warn!(
                why = "no_calls_with_markup",
                stripped_bytes = message.len(),
                has_start = message.contains(TOOL_CALL_START),
                has_end = message.contains(TOOL_CALL_END),
                has_string_delim = message.contains(STRING_DELIM),
                "gemma4 strip (recovery): zero calls extracted but gemma4 markup present (<|tool_call>, <tool_call|>, <|\"|>); suppressing entire message to prevent leak into normal_text. preview={:?}",
                preview
            );
            String::new()
        } else {
            // No markup at all → plain text passes through unchanged. No strip.
            message.trim().to_string()
        }
    } else {
        // Success: keep every gap between/around the complete tool-call blocks.
        removed_spans.sort_unstable();
        let mut kept = String::new();
        let mut prev_end = 0usize;
        for &(start, end) in &removed_spans {
            // Spans are non-overlapping and sorted; guard against any pathological
            // overlap so we never slice on a stale offset.
            if start >= prev_end {
                push_natural_text_gap(&mut kept, &message[prev_end..start]);
            }
            prev_end = prev_end.max(end);
        }
        push_natural_text_gap(&mut kept, &message[prev_end..]);
        // Trim only the outermost ends (interior whitespace stays verbatim) to
        // match the dynamo normal_text convention used across this family.
        kept.trim().to_string()
    };

    Ok((calls, Some(normal_text)))
}

/// Append a single inter-call / surrounding gap to the accumulated normal_text.
/// A gap that still contains gemma4 tool-call markup is a malformed / truncated
/// remnant (e.g. an incomplete trailing call that never closed) — drop it so
/// markup never leaks. Natural text is appended verbatim, whitespace included.
fn push_natural_text_gap(kept: &mut String, gap: &str) {
    if gap.is_empty() {
        return;
    }
    if gap.contains(TOOL_CALL_START) || gap.contains(TOOL_CALL_END) || gap.contains(STRING_DELIM) {
        let preview: String = gap.chars().take(120).collect();
        tracing::warn!(
            why = "dropped_markup_gap",
            stripped_bytes = gap.len(),
            "gemma4 strip (success): dropped a gap that still contained gemma4 markup (malformed/truncated remnant) to prevent leak into normal_text. preview={:?}",
            preview
        );
        return;
    }
    kept.push_str(gap);
}

// ---------------------------------------------------------------------------
// Recursive-descent parser for the Gemma 4 argument grammar
// ---------------------------------------------------------------------------
//
// Grammar (informal):
//
//   args     = (entry ("," entry)*)?
//   entry    = key ":" value
//   key      = bare-identifier (no quoting in Gemma 4 emit)
//   value    = string | number | bool | null | object | array
//   string   = "<|\"|>" .* "<|\"|>"
//   number   = -? [0-9]+ ( "." [0-9]+ )?
//   bool     = "true" | "false"
//   null     = "null" | "none" | "nil"
//   object   = "{" args "}"
//   array    = "[" (value ("," value)*)? "]"
//
// We parse straight into `serde_json::Value` so the rest of the pipeline sees
// the same shape every other parser produces.

struct Cursor<'a> {
    src: &'a str,
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(src: &'a str) -> Self {
        Self { src, pos: 0 }
    }

    fn rest(&self) -> &'a str {
        &self.src[self.pos..]
    }

    fn eof(&self) -> bool {
        self.pos >= self.src.len()
    }

    fn skip_whitespace(&mut self) {
        let bytes = self.src.as_bytes();
        while self.pos < bytes.len() && bytes[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }

    fn peek_byte(&self) -> Option<u8> {
        self.src.as_bytes().get(self.pos).copied()
    }

    fn consume_byte(&mut self, b: u8) -> bool {
        if self.peek_byte() == Some(b) {
            self.pos += 1;
            true
        } else {
            false
        }
    }
}

pub(crate) fn parse_args_object(input: &str) -> anyhow::Result<Value> {
    let mut cur = Cursor::new(input);
    cur.skip_whitespace();
    let val = parse_object_body(&mut cur)?;
    cur.skip_whitespace();
    if !cur.eof() {
        anyhow::bail!(
            "trailing characters after Gemma 4 args object at offset {}: {:?}",
            cur.pos,
            cur.rest()
        );
    }
    Ok(val)
}

fn parse_object_body(cur: &mut Cursor) -> anyhow::Result<Value> {
    let mut map = Map::new();
    cur.skip_whitespace();
    if cur.eof() || cur.peek_byte() == Some(b'}') {
        return Ok(Value::Object(map));
    }
    loop {
        cur.skip_whitespace();
        let key = parse_key(cur)?;
        cur.skip_whitespace();
        if !cur.consume_byte(b':') {
            anyhow::bail!("expected ':' after key '{}' at offset {}", key, cur.pos);
        }
        cur.skip_whitespace();
        // `key:` with no value emits `{"key": ""}` (matches upstream).
        let value = match cur.peek_byte() {
            None | Some(b',') | Some(b'}') => Value::String(String::new()),
            _ => parse_value(cur)?,
        };
        map.insert(key, value);
        cur.skip_whitespace();
        if !cur.consume_byte(b',') {
            break;
        }
    }
    Ok(Value::Object(map))
}

fn parse_key(cur: &mut Cursor) -> anyhow::Result<String> {
    let bytes = cur.src.as_bytes();
    let start = cur.pos;
    while cur.pos < bytes.len() {
        let b = bytes[cur.pos];
        if b.is_ascii_alphanumeric() || b == b'_' || b == b'-' || b == b'.' {
            cur.pos += 1;
        } else {
            break;
        }
    }
    if cur.pos == start {
        anyhow::bail!("expected bare key at offset {}", start);
    }
    Ok(cur.src[start..cur.pos].to_string())
}

/// Consume `keyword` ASCII-case-insensitively, only when the next byte is not
/// a word character (so `nullable` doesn't match `null` + leftover `able`).
fn try_consume_keyword(cur: &mut Cursor, keyword: &str) -> bool {
    let bytes = cur.src.as_bytes();
    let kw = keyword.as_bytes();
    let end = cur.pos + kw.len();
    if end > bytes.len() {
        return false;
    }
    if !bytes[cur.pos..end].eq_ignore_ascii_case(kw) {
        return false;
    }
    if let Some(&next) = bytes.get(end)
        && (next.is_ascii_alphanumeric() || next == b'_')
    {
        return false;
    }
    cur.pos = end;
    true
}

fn parse_value(cur: &mut Cursor) -> anyhow::Result<Value> {
    cur.skip_whitespace();

    // Delimited string `<|"|>...<|"|>`. If the closing delimiter is missing
    // (model truncation), take everything after the opener as the value.
    if cur.rest().starts_with(STRING_DELIM) {
        cur.pos += STRING_DELIM.len();
        let body_start = cur.pos;
        match cur.src[body_start..].find(STRING_DELIM) {
            Some(end_rel) => {
                let body_end = body_start + end_rel;
                let s = cur.src[body_start..body_end].to_string();
                cur.pos = body_end + STRING_DELIM.len();
                return Ok(Value::String(s));
            }
            None => {
                let s = cur.src[body_start..].to_string();
                cur.pos = cur.src.len();
                return Ok(Value::String(s));
            }
        }
    }

    // Object
    if cur.consume_byte(b'{') {
        let v = parse_object_body(cur)?;
        cur.skip_whitespace();
        if !cur.consume_byte(b'}') {
            anyhow::bail!("expected '}}' to close object at offset {}", cur.pos);
        }
        return Ok(v);
    }

    // Array
    if cur.consume_byte(b'[') {
        return parse_array(cur);
    }

    // Booleans + null aliases (case-insensitive).
    if try_consume_keyword(cur, "true") {
        return Ok(Value::Bool(true));
    }
    if try_consume_keyword(cur, "false") {
        return Ok(Value::Bool(false));
    }
    if try_consume_keyword(cur, "null")
        || try_consume_keyword(cur, "none")
        || try_consume_keyword(cur, "nil")
    {
        return Ok(Value::Null);
    }

    // Number
    parse_number(cur)
}

fn parse_array(cur: &mut Cursor) -> anyhow::Result<Value> {
    let mut items = Vec::new();
    cur.skip_whitespace();
    if cur.consume_byte(b']') {
        return Ok(Value::Array(items));
    }
    loop {
        cur.skip_whitespace();
        items.push(parse_value(cur)?);
        cur.skip_whitespace();
        if cur.consume_byte(b']') {
            return Ok(Value::Array(items));
        }
        if !cur.consume_byte(b',') {
            anyhow::bail!("expected ',' or ']' in array at offset {}", cur.pos);
        }
    }
}

fn parse_number(cur: &mut Cursor) -> anyhow::Result<Value> {
    let start = cur.pos;
    let bytes = cur.src.as_bytes();
    if cur.peek_byte() == Some(b'-') {
        cur.pos += 1;
    }
    let int_start = cur.pos;
    while cur.pos < bytes.len() && bytes[cur.pos].is_ascii_digit() {
        cur.pos += 1;
    }
    if cur.pos == int_start {
        anyhow::bail!(
            "expected value at offset {} but got: {:?}",
            start,
            &cur.src[start..]
        );
    }
    let mut is_float = false;
    if cur.peek_byte() == Some(b'.') {
        is_float = true;
        cur.pos += 1;
        while cur.pos < bytes.len() && bytes[cur.pos].is_ascii_digit() {
            cur.pos += 1;
        }
    }
    let lex = &cur.src[start..cur.pos];
    if is_float {
        let f: f64 = lex.parse()?;
        Ok(serde_json::json!(f))
    } else {
        let i: i64 = lex.parse()?;
        Ok(serde_json::json!(i))
    }
}
