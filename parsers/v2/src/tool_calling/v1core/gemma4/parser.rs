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

use serde_json::{Map, Value};
use uuid::Uuid;

use super::super::ToolDefinition;
use super::super::response::{CalledFunction, ToolCallResponse, ToolCallType};

pub(crate) const TOOL_CALL_START: &str = "<|tool_call>";
pub(crate) const TOOL_CALL_END: &str = "<tool_call|>";
pub(crate) const STRING_DELIM: &str = "<|\"|>";
pub(crate) const CALL_PREFIX: &str = "call:";
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

pub fn is_call_prefix_boundary(input: &str, idx: usize) -> bool {
    idx == 0
        || input[..idx]
            .chars()
            .next_back()
            .is_none_or(|ch| !(ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.')))
}

/// Type one call whose boundaries have already been resolved by the caller.
pub fn parse_one_tool_call_gemma4(
    invoke: &str,
    tools: Option<&[ToolDefinition]>,
) -> anyhow::Result<Option<ToolCallResponse>> {
    let allow_missing_start = !invoke.starts_with(TOOL_CALL_START);
    let Some((name, args_raw, _)) = parse_recoverable_call_at(invoke, allow_missing_start, true)
    else {
        return Ok(None);
    };
    parse_gemma_call_parts(name, args_raw, tools).map(Some)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_body_missing_its_close_marker_is_still_typed() {
        let body = "call:get_weather{city:<|\"|>Paris<|\"|>}";
        assert_eq!(
            parse_one_tool_call_gemma4(body, None)
                .unwrap()
                .expect("recovered")
                .function
                .arguments,
            r#"{"city":"Paris"}"#
        );
    }

    #[test]
    fn narration_does_not_hide_a_later_mismatched_close() {
        let input = "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>} trailing </tool_call>";
        assert!(parse_one_tool_call_gemma4(input, None).unwrap().is_none());
    }

    #[test]
    fn a_call_missing_both_wrapper_boundaries_is_typed_after_scanning() {
        let input = "call:get_weather{city:<|\"|>Paris<|\"|>}";
        assert!(parse_one_tool_call_gemma4(input, None).unwrap().is_some());
    }

    #[test]
    fn prose_containing_the_word_call_is_not_a_leading_call() {
        assert!(
            parse_one_tool_call_gemma4("call: you tomorrow", None)
                .unwrap()
                .is_none()
        );
    }
}
