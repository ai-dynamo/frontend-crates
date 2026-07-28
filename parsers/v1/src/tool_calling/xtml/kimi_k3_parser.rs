// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Kimi K3 XTML tool-call parser.
//!
//! The authoritative grammar is Moonshot's `encoding_k3.py` at model revision
//! `a527b42cb673d79569f3ffe0b3a0a655df98a739`:
//! <https://huggingface.co/moonshotai/Kimi-K3/blob/a527b42cb673d79569f3ffe0b3a0a655df98a739/encoding_k3.py>
//!
//! ```text
//! <|open|>response<|sep|>answer<|close|>response<|sep|>
//! <|open|>tools<|sep|>
//!   <|open|>call tool="get_weather" index="1"<|sep|>
//!     <|open|>argument key="city" type="string"<|sep|>Paris<|close|>argument<|sep|>
//!   <|close|>call<|sep|>
//! <|close|>tools<|sep|>
//! <|close|>message<|sep|><|end_of_msg|>
//! ```

use std::borrow::Cow;

use serde_json::Value;

use super::super::ToolDefinition;
use super::super::config::KimiK3ParserConfig;
use super::super::response::{CalledFunction, ToolCallResponse, ToolCallType};

pub(crate) const SEP: &str = "<|sep|>";

pub(crate) const RESPONSE_OPEN: &str = "<|open|>response<|sep|>";
pub(crate) const RESPONSE_CLOSE: &str = "<|close|>response<|sep|>";
pub(crate) const TOOLS_OPEN: &str = "<|open|>tools<|sep|>";
pub(crate) const TOOLS_CLOSE: &str = "<|close|>tools<|sep|>";
pub(crate) const CALL_OPEN_PREFIX: &str = "<|open|>call";
pub(crate) const CALL_CLOSE: &str = "<|close|>call<|sep|>";
pub(crate) const ARGUMENT_OPEN_PREFIX: &str = "<|open|>argument";
pub(crate) const ARGUMENT_CLOSE: &str = "<|close|>argument<|sep|>";
pub(crate) const JSON_OPEN_PREFIX: &str = "<|open|>json";
pub(crate) const JSON_CLOSE: &str = "<|close|>json<|sep|>";
pub(crate) const MESSAGE_OPEN_PREFIX: &str = "<|open|>message";
pub(crate) const MESSAGE_CLOSE: &str = "<|close|>message<|sep|>";
pub(crate) const THINK_CLOSE: &str = "<|close|>think<|sep|>";
pub(crate) const END_OF_MSG: &str = "<|end_of_msg|>";

/// Every reserved Kimi K3 structural boundary that can safely terminate an
/// implicit reasoning channel. Keeping this list K3-specific lets the
/// reasoning-to-XTML handoff recover at an inner channel boundary without
/// broad stripping of arbitrary `<|...|>` user text.
pub(crate) const JAIL_BOUNDARIES: [&str; 14] = [
    RESPONSE_OPEN,
    RESPONSE_CLOSE,
    TOOLS_OPEN,
    CALL_OPEN_PREFIX,
    ARGUMENT_OPEN_PREFIX,
    JSON_OPEN_PREFIX,
    MESSAGE_OPEN_PREFIX,
    ARGUMENT_CLOSE,
    JSON_CLOSE,
    CALL_CLOSE,
    TOOLS_CLOSE,
    MESSAGE_CLOSE,
    END_OF_MSG,
    // Defense in depth: the reasoning parser normally consumes this token.
    // If an upstream split ever leaves an orphan exact close in normal text,
    // jail it rather than exposing parser-owned markup.
    THINK_CLOSE,
];

/// Common detokenizer spelling when spacing between adjacent special tokens is
/// enabled. vLLM normally disables that option for K3, but accepting this
/// exact wire variant keeps the parser safe if another engine leaves it on.
///
/// These entries correspond to the response/tools/call/argument/json/message
/// boundaries in [`JAIL_BOUNDARIES`]. `END_OF_MSG` is a single token, while
/// the reasoning close is deliberately accepted only in its exact canonical
/// form for narrow defense in depth.
pub(crate) const SPACED_JAIL_BOUNDARIES: [&str; 12] = [
    "<|open|> response <|sep|>",
    "<|close|> response <|sep|>",
    "<|open|> tools <|sep|>",
    "<|open|> call",
    "<|open|> argument",
    "<|open|> json",
    "<|open|> message",
    "<|close|> argument <|sep|>",
    "<|close|> json <|sep|>",
    "<|close|> call <|sep|>",
    "<|close|> tools <|sep|>",
    "<|close|> message <|sep|>",
];

const SPACED_MARKER_ALIASES: [(&str, &str); 12] = [
    (SPACED_JAIL_BOUNDARIES[0], RESPONSE_OPEN),
    (SPACED_JAIL_BOUNDARIES[1], RESPONSE_CLOSE),
    (SPACED_JAIL_BOUNDARIES[2], TOOLS_OPEN),
    (SPACED_JAIL_BOUNDARIES[3], CALL_OPEN_PREFIX),
    (SPACED_JAIL_BOUNDARIES[4], ARGUMENT_OPEN_PREFIX),
    (SPACED_JAIL_BOUNDARIES[5], JSON_OPEN_PREFIX),
    (SPACED_JAIL_BOUNDARIES[6], MESSAGE_OPEN_PREFIX),
    (SPACED_JAIL_BOUNDARIES[7], ARGUMENT_CLOSE),
    (SPACED_JAIL_BOUNDARIES[8], JSON_CLOSE),
    (SPACED_JAIL_BOUNDARIES[9], CALL_CLOSE),
    (SPACED_JAIL_BOUNDARIES[10], TOOLS_CLOSE),
    (SPACED_JAIL_BOUNDARIES[11], MESSAGE_CLOSE),
];

/// Detect complete or partial Kimi K3 structural markers at a chunk boundary.
pub fn detect_tool_call_start_kimi_k3(chunk: &str, config: &KimiK3ParserConfig) -> bool {
    config.start_tokens().iter().any(|marker| {
        chunk.contains(marker)
            || (1..marker.len())
                .any(|len| marker.is_char_boundary(len) && chunk.ends_with(&marker[..len]))
    })
}

/// Return the end of the first complete jailed XTML span.
///
/// Response/message markers are consumed independently so the v1 jail can
/// continue streaming response body text. A tools section remains jailed until
/// its outer close marker, while a recoverable bare call ends at `call` close.
pub fn find_tool_call_end_position_kimi_k3(
    chunk: &str,
    _config: &KimiK3ParserConfig,
) -> Option<usize> {
    let (start, wire_marker, marker) = first_wire_marker(chunk)?;
    let after_start = &chunk[start..];

    // An interval-batched backend delta can contain a complete nested XTML
    // message. Prefer the outermost visible terminator so all framing is
    // parsed and suppressed as one span rather than released as trailing text.
    if let Some(end) = find_wire_marker_end(after_start, END_OF_MSG) {
        return Some(start + end);
    }
    if let Some(end) = find_wire_marker_end(after_start, MESSAGE_CLOSE) {
        return Some(start + end);
    }

    let relative_end = match marker {
        RESPONSE_OPEN | RESPONSE_CLOSE | ARGUMENT_CLOSE | JSON_CLOSE | TOOLS_CLOSE | CALL_CLOSE
        | MESSAGE_CLOSE | THINK_CLOSE | END_OF_MSG => Some(wire_marker.len()),
        TOOLS_OPEN => find_wire_marker_end(after_start, TOOLS_CLOSE),
        CALL_OPEN_PREFIX => find_wire_marker_end(after_start, CALL_CLOSE),
        ARGUMENT_OPEN_PREFIX => find_wire_marker_end(after_start, ARGUMENT_CLOSE),
        JSON_OPEN_PREFIX => find_wire_marker_end(after_start, JSON_CLOSE),
        MESSAGE_OPEN_PREFIX => find_wire_marker_end(after_start, MESSAGE_CLOSE),
        _ => None,
    }?;

    Some(start + relative_end)
}

/// Parse a complete Kimi K3 output (or one span accumulated by the v1 jail).
pub fn try_tool_call_parse_kimi_k3(
    message: &str,
    config: &KimiK3ParserConfig,
    _tools: Option<&[ToolDefinition]>,
) -> anyhow::Result<(Vec<ToolCallResponse>, Option<String>)> {
    let normalized = normalize_spaced_markers(message);
    let sanitized = strip_orphan_think_close(normalized.as_ref());
    let message = sanitized.as_ref();
    let normal_text = extract_response_text(message);
    let calls = extract_calls(message, config);
    Ok((calls, Some(normal_text)))
}

fn strip_orphan_think_close(message: &str) -> Cow<'_, str> {
    if !message.contains(THINK_CLOSE) {
        return Cow::Borrowed(message);
    }
    Cow::Owned(message.replace(THINK_CLOSE, ""))
}

fn normalize_spaced_markers(message: &str) -> Cow<'_, str> {
    // Keep the canonical production path allocation-free and avoid probing
    // every alias unless the engine-added spacing signature is present.
    if !message.contains("<|open|> ") && !message.contains("<|close|> ") {
        return Cow::Borrowed(message);
    }

    let mut normalized = message.to_string();
    for (spaced, canonical) in SPACED_MARKER_ALIASES {
        normalized = normalized.replace(spaced, canonical);
    }
    Cow::Owned(normalized)
}

fn first_marker(chunk: &str) -> Option<(usize, &'static str)> {
    JAIL_BOUNDARIES
        .into_iter()
        .filter_map(|marker| chunk.find(marker).map(|position| (position, marker)))
        .min_by_key(|(position, _)| *position)
}

fn first_wire_marker(chunk: &str) -> Option<(usize, &'static str, &'static str)> {
    JAIL_BOUNDARIES
        .into_iter()
        .map(|marker| (marker, marker))
        .chain(SPACED_MARKER_ALIASES)
        .filter_map(|(wire, canonical)| {
            chunk.find(wire).map(|position| (position, wire, canonical))
        })
        .min_by_key(|(position, _, _)| *position)
}

/// Split a reasoning delta at the first reserved K3 structural boundary.
///
/// This is a narrow recovery hook for the reasoning-to-XTML handoff. If an
/// upstream reasoning parser has already placed a complete response/tools
/// channel in `reasoning_content`, the K3 jail can move only that protocol
/// suffix back to `content` and parse it normally. Arbitrary `<|...|>` text is
/// preserved because only the explicit K3 boundary tables are considered.
pub(crate) fn split_reasoning_handoff(message: &str) -> Option<(&str, &str)> {
    // Reasoning prose overwhelmingly contains no K3 control-token prefix.
    // Keep that hot path to one cheap scan; the full reserved-marker table is
    // consulted only for the exceptional handoff-recovery case.
    if !message.contains("<|") {
        return None;
    }
    let (position, _, _) = first_wire_marker(message)?;
    Some(message.split_at(position))
}

fn find_wire_marker_end(text: &str, canonical: &str) -> Option<usize> {
    std::iter::once(canonical)
        .chain(
            SPACED_MARKER_ALIASES
                .iter()
                .filter_map(|(wire, target)| (*target == canonical).then_some(*wire)),
        )
        .filter_map(|wire| text.find(wire).map(|position| position + wire.len()))
        .min()
}

fn earliest_position(text: &str, markers: &[&str]) -> Option<usize> {
    markers.iter().filter_map(|marker| text.find(marker)).min()
}

/// Extract only the model's response channel, or pass plain unwrapped text
/// through. Text after the response closes is XTML epilogue, not user content.
fn extract_response_text(message: &str) -> String {
    let logical_end =
        earliest_position(message, &[MESSAGE_CLOSE, END_OF_MSG]).unwrap_or(message.len());
    let logical = &message[..logical_end];

    if let Some(open) = logical.find(RESPONSE_OPEN) {
        let body_start = open + RESPONSE_OPEN.len();
        let body = &logical[body_start..];
        let body_end = earliest_position(body, &[RESPONSE_CLOSE, TOOLS_OPEN]).unwrap_or(body.len());
        return body[..body_end].to_string();
    }

    // The generation prompt consumes response-open when thinking is disabled.
    if let Some(close) = logical.find(RESPONSE_CLOSE) {
        return strip_leading_message_open(&logical[..close]).to_string();
    }

    if let Some(tools) = earliest_position(logical, &[TOOLS_OPEN, CALL_OPEN_PREFIX]) {
        return strip_leading_message_open(&logical[..tools]).to_string();
    }

    if first_marker(logical).is_some()
        || logical.contains(TOOLS_CLOSE)
        || logical.contains(CALL_CLOSE)
    {
        return String::new();
    }

    strip_leading_message_open(logical).to_string()
}

fn strip_leading_message_open(text: &str) -> &str {
    let Some(rest) = text.strip_prefix(MESSAGE_OPEN_PREFIX) else {
        return text;
    };
    let Some(end) = rest.find(SEP) else {
        return text;
    };
    &rest[end + SEP.len()..]
}

fn extract_calls(message: &str, config: &KimiK3ParserConfig) -> Vec<ToolCallResponse> {
    let logical_end =
        earliest_position(message, &[MESSAGE_CLOSE, END_OF_MSG]).unwrap_or(message.len());
    let logical = &message[..logical_end];
    let mut calls = Vec::new();

    if let Some(tools_start) = logical.find(TOOLS_OPEN) {
        let body_start = tools_start + TOOLS_OPEN.len();
        if let Some(close) = logical[body_start..].find(TOOLS_CLOSE) {
            parse_calls_region(&logical[body_start..body_start + close], false, &mut calls);
        } else if config.allow_eof_recovery {
            parse_calls_region(&logical[body_start..], true, &mut calls);
        }
        return calls;
    }

    // A complete bare call is delimiter-terminated and therefore recoverable
    // even if the outer tools wrapper is absent.
    parse_calls_region(logical, config.allow_eof_recovery, &mut calls);
    calls
}

fn parse_calls_region(
    region: &str,
    allow_missing_call_close: bool,
    calls: &mut Vec<ToolCallResponse>,
) {
    let mut cursor = 0usize;
    while let Some(relative) = region[cursor..].find(CALL_OPEN_PREFIX) {
        let start = cursor + relative;
        match parse_call_at(&region[start..], allow_missing_call_close) {
            Some((call, consumed)) => {
                if let Some(call) = call {
                    calls.push(call);
                }
                cursor = start + consumed.max(CALL_OPEN_PREFIX.len());
            }
            None => {
                tracing::warn!(
                    why = "kimi_k3_incomplete_call",
                    buffered_bytes = region.len().saturating_sub(start),
                    "dropping incomplete or malformed Kimi K3 call"
                );
                cursor = start + CALL_OPEN_PREFIX.len();
            }
        }
    }
}

/// Returns `(decoded call, consumed bytes)`. A syntactically complete call with
/// no tool name is consumed but intentionally produces no API call.
fn parse_call_at(
    input: &str,
    allow_missing_call_close: bool,
) -> Option<(Option<ToolCallResponse>, usize)> {
    let after_prefix = input.strip_prefix(CALL_OPEN_PREFIX)?;
    let tag_end_relative = after_prefix.find(SEP)?;
    let attrs_raw = &after_prefix[..tag_end_relative];
    let attrs = parse_attrs(attrs_raw)?;
    let body_start = CALL_OPEN_PREFIX.len() + tag_end_relative + SEP.len();

    let (body_end, consumed, missing_close) =
        if let Some(close) = input[body_start..].find(CALL_CLOSE) {
            let body_end = body_start + close;
            (body_end, body_end + CALL_CLOSE.len(), false)
        } else if allow_missing_call_close {
            let outer_end = earliest_position(
                &input[body_start..],
                &[TOOLS_CLOSE, MESSAGE_CLOSE, END_OF_MSG],
            )
            .map(|position| body_start + position)
            .unwrap_or(input.len());
            (outer_end, outer_end, true)
        } else {
            return None;
        };

    let body = &input[body_start..body_end];
    if missing_close {
        let trimmed = body.trim_end();
        let delimiter_terminated =
            trimmed.ends_with(ARGUMENT_CLOSE) || trimmed.ends_with(JSON_CLOSE);
        if !delimiter_terminated {
            return None;
        }
        tracing::warn!(
            why = "kimi_k3_recovered_missing_call_close",
            body_bytes = body.len(),
            "recovering delimiter-terminated Kimi K3 call without outer call close"
        );
    }

    let arguments = parse_call_body(body)?;
    let name = attr_value(&attrs, "tool").unwrap_or_default();
    if name.is_empty() {
        tracing::warn!(
            why = "kimi_k3_missing_tool_name",
            call_bytes = consumed,
            "dropping Kimi K3 call without a tool name"
        );
        return Some((None, consumed));
    }

    let id = tool_call_id(name, attr_value(&attrs, "index"));
    Some((
        Some(ToolCallResponse {
            id,
            tp: ToolCallType::Function,
            function: CalledFunction {
                name: name.to_string(),
                arguments,
            },
        }),
        consumed,
    ))
}

fn parse_call_body(body: &str) -> Option<String> {
    let trimmed = body.trim();
    if trimmed.is_empty() {
        return Some("{}".to_string());
    }

    if let Some(after_open) = trimmed.strip_prefix(JSON_OPEN_PREFIX) {
        let tag_end = after_open.find(SEP)?;
        let value_start = JSON_OPEN_PREFIX.len() + tag_end + SEP.len();
        let close = trimmed[value_start..].find(JSON_CLOSE)?;
        let value_end = value_start + close;
        if !trimmed[value_end + JSON_CLOSE.len()..].trim().is_empty() {
            return None;
        }
        // Raw-json form is intentionally byte-preserving and unvalidated.
        return Some(trimmed[value_start..value_end].to_string());
    }

    let mut ordered: Vec<(String, String)> = Vec::new();
    let mut cursor = 0usize;
    while cursor < trimmed.len() {
        let rest = &trimmed[cursor..];
        let whitespace = rest.len() - rest.trim_start().len();
        cursor += whitespace;
        if cursor == trimmed.len() {
            break;
        }

        let rest = &trimmed[cursor..];
        let after_open = rest.strip_prefix(ARGUMENT_OPEN_PREFIX)?;
        let tag_end = after_open.find(SEP)?;
        let attrs = parse_attrs(&after_open[..tag_end])?;
        let value_start = cursor + ARGUMENT_OPEN_PREFIX.len() + tag_end + SEP.len();
        let close = trimmed[value_start..].find(ARGUMENT_CLOSE)?;
        let value_end = value_start + close;
        let raw = &trimmed[value_start..value_end];

        let key = attr_value(&attrs, "key").unwrap_or_default().to_string();
        let arg_type = attr_value(&attrs, "type").unwrap_or("string");
        let encoded_value = encode_argument_value(arg_type, raw);
        if let Some((_, existing)) = ordered.iter_mut().find(|(existing, _)| *existing == key) {
            *existing = encoded_value;
        } else {
            ordered.push((key, encoded_value));
        }
        cursor = value_end + ARGUMENT_CLOSE.len();
    }

    let mut output = String::from("{");
    for (position, (key, value)) in ordered.iter().enumerate() {
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

fn encode_argument_value(arg_type: &str, raw: &str) -> String {
    if arg_type == "string" {
        return serde_json::to_string(raw).expect("serializing a Rust string cannot fail");
    }

    if serde_json::from_str::<Value>(raw).is_ok() {
        compact_json_lexically(raw)
    } else {
        serde_json::to_string(raw).expect("serializing a Rust string cannot fail")
    }
}

/// Remove insignificant JSON whitespace without reparsing into `Value`, which
/// would reorder object keys and normalize number spelling in this workspace.
fn compact_json_lexically(raw: &str) -> String {
    let mut output = String::with_capacity(raw.len());
    let mut in_string = false;
    let mut escaped = false;
    for ch in raw.chars() {
        if in_string {
            output.push(ch);
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
        } else if ch == '"' {
            in_string = true;
            output.push(ch);
        } else if !ch.is_whitespace() {
            output.push(ch);
        }
    }
    output
}

fn parse_attrs(input: &str) -> Option<Vec<(String, String)>> {
    let mut attrs = Vec::new();
    let mut cursor = 0usize;
    while cursor < input.len() {
        let rest = &input[cursor..];
        let trimmed = rest.trim_start();
        cursor += rest.len() - trimmed.len();
        if cursor == input.len() {
            break;
        }

        let rest = &input[cursor..];
        let key_len = rest
            .char_indices()
            .take_while(|(_, ch)| ch.is_alphanumeric() || *ch == '_')
            .map(|(position, ch)| position + ch.len_utf8())
            .last()?;
        let key = &rest[..key_len];
        let value_input = rest[key_len..].strip_prefix("=\"")?;
        let value_end = value_input.find('"')?;
        let value = unescape_attr(&value_input[..value_end]);
        attrs.push((key.to_string(), value));
        cursor += key_len + 2 + value_end + 1;
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
        Some(raw) => match raw.parse::<i64>() {
            Ok(one_based) => format!("{name}:{}", one_based - 1),
            Err(_) => format!("{name}:{raw}"),
        },
    }
}

#[cfg(test)]
mod tests {
    use serde_json::Value;

    use super::*;

    fn arg(key: &str, arg_type: Option<&str>, value: &str) -> String {
        let type_attr = arg_type
            .map(|arg_type| format!(" type=\"{arg_type}\""))
            .unwrap_or_default();
        format!("<|open|>argument key=\"{key}\"{type_attr}<|sep|>{value}{ARGUMENT_CLOSE}")
    }

    fn call(attrs: &str, body: &str) -> String {
        format!("{CALL_OPEN_PREFIX} {attrs}{SEP}{body}{CALL_CLOSE}")
    }

    fn tools(body: &str) -> String {
        format!("{TOOLS_OPEN}{body}{TOOLS_CLOSE}")
    }

    fn parse(input: &str) -> (Vec<ToolCallResponse>, String) {
        let (calls, normal) =
            try_tool_call_parse_kimi_k3(input, &KimiK3ParserConfig::default(), None).unwrap();
        (calls, normal.unwrap())
    }

    #[test]
    fn parses_response_and_all_typed_arguments() {
        let body = [
            arg("city", Some("string"), "Paris"),
            arg("days", Some("number"), "1.0"),
            arg("rain", Some("boolean"), "true"),
            arg("none", Some("null"), "null"),
            arg("filters", Some("object"), r#"{"b": 1, "a": 2}"#),
            arg("hours", Some("array"), "[8, 20]"),
        ]
        .concat();
        let input = format!(
            "{RESPONSE_OPEN}I'll check.{RESPONSE_CLOSE}{}{MESSAGE_CLOSE}{END_OF_MSG}",
            tools(&call("tool=\"get_weather\" index=\"1\"", &body))
        );

        let (calls, normal) = parse(&input);

        assert_eq!(normal, "I'll check.");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "get_weather:0");
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(
            calls[0].function.arguments,
            r#"{"city":"Paris","days":1.0,"rain":true,"none":null,"filters":{"b":1,"a":2},"hours":[8,20]}"#
        );
    }

    #[test]
    fn raw_json_arguments_are_byte_preserving() {
        let raw = r#"{"b": 1,  "a": [2 , 3]}"#;
        let json = format!("{JSON_OPEN_PREFIX} type=\"object\"{SEP}{raw}{JSON_CLOSE}");
        let (calls, _) = parse(&tools(&call("tool=\"run\" index=\"2\"", &json)));

        assert_eq!(calls[0].id, "run:1");
        assert_eq!(calls[0].function.arguments, raw);
    }

    #[test]
    fn prompt_consumed_response_open_preserves_content_and_strips_markers() {
        let input = format!("answer{RESPONSE_CLOSE}{MESSAGE_CLOSE}{END_OF_MSG}");
        let (calls, normal) = parse(&input);

        assert!(calls.is_empty());
        assert_eq!(normal, "answer");

        for marker in [RESPONSE_OPEN, RESPONSE_CLOSE, MESSAGE_CLOSE, END_OF_MSG] {
            let (_, marker_normal) = parse(marker);
            assert_eq!(marker_normal, "", "marker {marker}");
        }
    }

    #[test]
    fn orphan_exact_think_close_is_stripped_without_dropping_content() {
        let input = format!("{THINK_CLOSE}answer");
        let (calls, normal) = parse(&input);

        assert!(calls.is_empty());
        assert_eq!(normal, "answer");
    }

    #[test]
    fn multiple_calls_ids_and_missing_name() {
        let input = tools(
            &[
                call("tool=\"first\" index=\"3\"", ""),
                call("index=\"4\"", &arg("x", Some("number"), "1")),
                call("tool=\"second\" index=\"raw\"", ""),
                call("tool=\"third\"", ""),
            ]
            .concat(),
        );

        let (calls, _) = parse(&input);

        assert_eq!(calls.len(), 3);
        assert_eq!(calls[0].id, "first:2");
        assert_eq!(calls[1].id, "second:raw");
        assert_eq!(calls[2].id, "third");
    }

    #[test]
    fn attributes_unescape_in_inverse_order() {
        let input = tools(&call("tool=\"a&amp;quot;b&amp;c\" index=\"1\"", ""));
        let (calls, _) = parse(&input);

        // `&quot;` is replaced before `&amp;`, so an encoded literal
        // `&quot;` remains text rather than becoming a quote recursively.
        assert_eq!(calls[0].function.name, "a&quot;b&c");

        let input = tools(&call("tool=\"a&quot;b&amp;c\" index=\"1\"", ""));
        let (calls, _) = parse(&input);
        assert_eq!(calls[0].function.name, "a\"b&c");
    }

    #[test]
    fn missing_type_defaults_to_string_and_malformed_typed_value_falls_back() {
        let body = [
            arg("plain", None, "raw text"),
            arg("bad", Some("number"), "not-a-number"),
        ]
        .concat();
        let (calls, _) = parse(&tools(&call("tool=\"f\" index=\"1\"", &body)));
        let args: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();

        assert_eq!(args["plain"], "raw text");
        assert_eq!(args["bad"], "not-a-number");
    }

    #[test]
    fn complete_calls_survive_a_truncated_later_call() {
        let first = call("tool=\"first\" index=\"1\"", "");
        let second = format!(
            "{CALL_OPEN_PREFIX} tool=\"second\" index=\"2\"{SEP}<|open|>argument key=\"x\" type=\"string\"{SEP}ambiguous"
        );
        let input = format!("{TOOLS_OPEN}{first}{second}{TOOLS_CLOSE}");

        let (calls, normal) = parse(&input);

        assert_eq!(normal, "");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "first");
    }

    #[test]
    fn finalize_recovers_only_delimiter_terminated_calls() {
        let complete_body = arg("x", Some("number"), "1");
        let complete_without_outer_closes =
            format!("{TOOLS_OPEN}{CALL_OPEN_PREFIX} tool=\"calc\" index=\"1\"{SEP}{complete_body}");
        let incomplete_value = format!(
            "{TOOLS_OPEN}{CALL_OPEN_PREFIX} tool=\"calc\" index=\"1\"{SEP}<|open|>argument key=\"x\" type=\"string\"{SEP}Par"
        );

        let (before_finalize, _) = parse(&complete_without_outer_closes);
        assert!(before_finalize.is_empty());

        let recovery = KimiK3ParserConfig {
            allow_eof_recovery: true,
        };
        let (recovered, _) =
            try_tool_call_parse_kimi_k3(&complete_without_outer_closes, &recovery, None).unwrap();
        let (dropped, _) = try_tool_call_parse_kimi_k3(&incomplete_value, &recovery, None).unwrap();

        assert_eq!(recovered.len(), 1);
        assert_eq!(recovered[0].function.arguments, r#"{"x":1}"#);
        assert!(dropped.is_empty());
    }

    #[test]
    fn unknown_tool_is_still_emitted() {
        let defined = [ToolDefinition {
            name: "known".to_string(),
            parameters: None,
            strict: None,
        }];
        let input = tools(&call("tool=\"unknown\" index=\"1\"", ""));
        let (calls, _) =
            try_tool_call_parse_kimi_k3(&input, &KimiK3ParserConfig::default(), Some(&defined))
                .unwrap();

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "unknown");
    }

    #[test]
    fn detects_partial_markers_and_finds_contextual_end() {
        let config = KimiK3ParserConfig::default();
        assert!(detect_tool_call_start_kimi_k3("answer<|ope", &config));
        assert!(!detect_tool_call_start_kimi_k3("ordinary answer", &config));

        let response = format!("{RESPONSE_OPEN}answer{RESPONSE_CLOSE}");
        assert_eq!(
            find_tool_call_end_position_kimi_k3(&response, &config),
            Some(RESPONSE_OPEN.len())
        );

        let tools = tools(&call("tool=\"f\" index=\"1\"", ""));
        assert_eq!(
            find_tool_call_end_position_kimi_k3(&tools, &config),
            Some(tools.len())
        );
        assert_eq!(
            find_tool_call_end_position_kimi_k3(TOOLS_OPEN, &config),
            None
        );
    }
}
