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

use std::collections::HashMap;

use serde::ser::{Serialize, SerializeMap, Serializer};
use serde_json::Value;
use serde_json::value::RawValue;
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
            ArgValue::Object(Vec::new())
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
    if !name.chars().all(is_call_name_char) {
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

pub(crate) fn is_call_name_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.')
}

pub fn is_call_prefix_boundary(input: &str, idx: usize) -> bool {
    idx == 0
        || input[..idx]
            .chars()
            .next_back()
            .is_none_or(|ch| !is_call_name_char(ch))
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

/// Whether a candidate has complete Gemma call boundaries, independent of whether
/// its argument body parses. The emitter owns malformed-argument fallback behavior.
pub(crate) fn has_recoverable_tool_call_boundaries_gemma4(invoke: &str) -> bool {
    parse_recoverable_call_at(invoke, true, true).is_some()
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
//   value    = string | number | bool | null | object | array | bare
//   string   = "<|\"|>" .* "<|\"|>"
//   number   = -? [0-9]+ ( "." [0-9]* )? ( [eE] [+-]? [0-9]+ )?
//   bool     = "true" | "false"
//   null     = "null" | "none" | "nil"
//   object   = "{" args "}"
//   array    = "[" (value ("," value)*)? "]"
//   bare     = any other token before "," "}" "]", kept as a string
//
// We parse into `ArgValue`, which serializes to the same JSON shape every other
// parser produces while keeping each number's digits exact.

/// A parsed argument value. `serde_json::Value` would round numbers through
/// `f64`, so every number other than an `i64` keeps its source text instead.
enum ArgValue {
    /// A string, boolean, null, `i64`, or an `f64` for a spelling JSON forbids.
    Scalar(Value),
    RawNumber(Box<RawValue>),
    Array(Vec<ArgValue>),
    /// Entries in source order; a repeated key keeps its first position and last value.
    Object(Vec<(String, ArgValue)>),
}

impl From<Value> for ArgValue {
    fn from(value: Value) -> Self {
        Self::Scalar(value)
    }
}

impl Serialize for ArgValue {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Scalar(value) => value.serialize(serializer),
            Self::RawNumber(raw) => raw.serialize(serializer),
            Self::Array(items) => items.serialize(serializer),
            Self::Object(entries) => {
                let mut map = serializer.serialize_map(Some(entries.len()))?;
                for (key, value) in entries {
                    map.serialize_entry(key, value)?;
                }
                map.end()
            }
        }
    }
}

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

fn parse_args_object(input: &str) -> anyhow::Result<ArgValue> {
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

fn parse_object_body(cur: &mut Cursor) -> anyhow::Result<ArgValue> {
    let mut entries: Vec<(String, ArgValue)> = Vec::new();
    let mut indices: HashMap<&str, usize> = HashMap::new();
    cur.skip_whitespace();
    if cur.eof() || cur.peek_byte() == Some(b'}') {
        return Ok(ArgValue::Object(entries));
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
            None | Some(b',') | Some(b'}') => Value::String(String::new()).into(),
            _ => parse_value(cur)?,
        };
        match indices.get(key) {
            Some(&index) => entries[index].1 = value,
            None => {
                indices.insert(key, entries.len());
                entries.push((key.to_string(), value));
            }
        }
        cur.skip_whitespace();
        if !cur.consume_byte(b',') {
            break;
        }
    }
    Ok(ArgValue::Object(entries))
}

fn parse_key<'a>(cur: &mut Cursor<'a>) -> anyhow::Result<&'a str> {
    let bytes = cur.src.as_bytes();
    let start = cur.pos;
    while cur.pos < bytes.len() {
        let b = bytes[cur.pos];
        if is_call_name_char(char::from(b)) {
            cur.pos += 1;
        } else {
            break;
        }
    }
    if cur.pos == start {
        anyhow::bail!("expected bare key at offset {}", start);
    }
    Ok(&cur.src[start..cur.pos])
}

fn parse_value(cur: &mut Cursor) -> anyhow::Result<ArgValue> {
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
                return Ok(Value::String(s).into());
            }
            None => {
                let s = cur.src[body_start..].to_string();
                cur.pos = cur.src.len();
                return Ok(Value::String(s).into());
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

    parse_bare_value(cur)
}

fn parse_array(cur: &mut Cursor) -> anyhow::Result<ArgValue> {
    let mut items = Vec::new();
    cur.skip_whitespace();
    if cur.consume_byte(b']') {
        return Ok(ArgValue::Array(items));
    }
    loop {
        cur.skip_whitespace();
        items.push(parse_value(cur)?);
        cur.skip_whitespace();
        if cur.consume_byte(b']') {
            return Ok(ArgValue::Array(items));
        }
        if !cur.consume_byte(b',') {
            anyhow::bail!("expected ',' or ']' in array at offset {}", cur.pos);
        }
    }
}

/// A bare value runs to the next `,`, `}`, or `]`. Booleans and null aliases
/// (case-insensitive) and numbers are typed; any other token is kept as a string,
/// as upstream vLLM's original parser did, so one unexpected value cannot discard
/// the other arguments.
fn parse_bare_value(cur: &mut Cursor) -> anyhow::Result<ArgValue> {
    let rest = cur.rest();
    let len = rest.find([',', '}', ']']).unwrap_or(rest.len());
    let token = rest[..len].trim_ascii();
    if token.is_empty() {
        anyhow::bail!("expected value at offset {}", cur.pos);
    }
    cur.pos += len;
    let is = |keyword: &str| token.eq_ignore_ascii_case(keyword);
    Ok(if is("true") {
        Value::Bool(true).into()
    } else if is("false") {
        Value::Bool(false).into()
    } else if is("null") || is("none") || is("nil") {
        Value::Null.into()
    } else {
        parse_number(token).unwrap_or_else(|| Value::String(token.to_string()).into())
    })
}

/// Types a JSON number token; leading zeros and a bare trailing `.` stay accepted.
/// An `i64` keeps its `Value` form and any other number its exact source text,
/// with `f64` only for spellings JSON forbids. `serde_json` cannot read back a
/// number beyond `f64` range, so such a token is not typed.
fn parse_number(token: &str) -> Option<ArgValue> {
    let (int, rest) = split_digits(token.strip_prefix('-').unwrap_or(token));
    if int.is_empty() {
        return None;
    }
    let (fraction, rest) = match rest.strip_prefix('.') {
        Some(rest) => (true, split_digits(rest).1),
        None => (false, rest),
    };
    let (exponent, rest) = match rest.strip_prefix(['e', 'E']) {
        Some(rest) => {
            let (digits, rest) = split_digits(rest.strip_prefix(['+', '-']).unwrap_or(rest));
            if digits.is_empty() {
                return None;
            }
            (true, rest)
        }
        None => (false, rest),
    };
    if !rest.is_empty() {
        return None;
    }
    if !fraction
        && !exponent
        && let Ok(value) = token.parse::<i64>()
    {
        return Some(Value::from(value).into());
    }
    let value = token
        .parse::<f64>()
        .ok()
        .filter(|value| value.is_finite())?;
    Some(match RawValue::from_string(token.to_string()) {
        Ok(raw) => ArgValue::RawNumber(raw),
        Err(_) => Value::from(value).into(),
    })
}

fn split_digits(s: &str) -> (&str, &str) {
    s.split_at(s.bytes().take_while(u8::is_ascii_digit).count())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_body_missing_its_close_marker_is_still_typeable() {
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
    fn a_mismatched_xml_close_is_not_typeable_as_a_call() {
        let input = "call:get_weather{city:<|\"|>Paris<|\"|>}</tool_call>";
        assert!(parse_one_tool_call_gemma4(input, None).unwrap().is_none());
    }

    #[test]
    fn narration_does_not_hide_a_later_mismatched_close() {
        let input = "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>} trailing </tool_call>";
        assert!(parse_one_tool_call_gemma4(input, None).unwrap().is_none());
    }

    #[test]
    fn bare_complete_call_remains_typeable() {
        let input = "call:get_weather{city:<|\"|>Paris<|\"|>}";
        assert!(parse_one_tool_call_gemma4(input, None).unwrap().is_some());
    }

    #[test]
    fn json_numbers_and_bare_words_keep_their_siblings() {
        // Python float repr (what the Gemma 4 chat template prints), an integer
        // beyond i64, and a bare word used to discard every argument of the call.
        for (args, expected) in [
            (
                r#"huge:1e+16,lr:1e-07,p:0.10000000000000000001,unit:<|"|>c<|"|>"#,
                r#"{"huge":1e+16,"lr":1e-07,"p":0.10000000000000000001,"unit":"c"}"#,
            ),
            (
                r#"big:1180591620717411303424,unit:<|"|>c<|"|>"#,
                r#"{"big":1180591620717411303424,"unit":"c"}"#,
            ),
            (
                r#"mode:fast,unit:<|"|>c<|"|>"#,
                r#"{"mode":"fast","unit":"c"}"#,
            ),
        ] {
            let input = format!("<|tool_call>call:f{{{args}}}<tool_call|>");
            let call = parse_one_tool_call_gemma4(&input, None)
                .unwrap()
                .expect("call");
            assert_eq!(call.function.arguments, expected, "{args}");
        }
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
