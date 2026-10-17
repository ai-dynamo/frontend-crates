// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Beam tool-call parser (NOT GPT-OSS Harmony).
//!
//! ```text
//! <|start|>assistant to=NAME<|channel|>analysis <|constrain|>xml<|message|>
//!   <rfl:parameter name="KEY">VALUE</rfl:parameter>… <|call|>      (or <|end|>)
//! ```
//!
//! Pair with the `beam` reasoning parser, which strips the analysis/final framing and
//! forwards every tool envelope verbatim. The header may be full, may start after the
//! `assistant` role the prompt already consumed (` to=NAME<|channel|>…`), and several
//! calls may follow one another.
//!
//! This is the fallible half of the pair: a call whose terminator is missing or whose
//! parameter never closes is an `Err`, never a partial call (the builder's parser
//! contract). The body itself is read as the template's tag search: junk between
//! complete `<rfl:parameter>` regions and malformed openers are skipped, so a call with
//! stray bytes between its parameters still decodes. Syntax is separate from request
//! tool policy: a tool absent from `tools` still decodes, with its values kept as text.
//!
//! The argument typing (`schema_types` / `coerce`) is derived from the model builder's frozen
//! reference (attribution in the derived file's SPDX header) via [`py`], an
//! independent copy of the helpers shipped in `dynamo-parsers-v2`.

mod py;
mod typing;

use py::{PyDict, PyValue};
use uuid::Uuid;

/// How the end of a parameter value is found.
///
/// `Lenient` is the reference regex: the first `</rfl:parameter>` ends the value.
/// `ConstrainedJson` is for requests whose output was constrained to JSON: a value that
/// is a JSON object/array owns its strings, so a closer inside one is data. Mirrors
/// `ArgumentFraming` in `dynamo-parsers-v2`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BeamArgumentFraming {
    #[default]
    Lenient,
    ConstrainedJson,
}

/// Request-scoped Beam parser configuration (the grammar itself is fixed).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub struct BeamParserConfig {
    #[serde(default)]
    pub argument_framing: BeamArgumentFraming,
}

/// Byte offset just past the first complete top-level JSON object/array in `text`
/// (leading whitespace skipped), or `None` when absent or still open.
fn json_value_end(text: &str) -> Option<usize> {
    let mut stack: Vec<char> = Vec::new();
    let (mut in_string, mut escaped) = (false, false);
    for (idx, c) in text.char_indices() {
        if stack.is_empty() {
            match c {
                '{' => stack.push('}'),
                '[' => stack.push(']'),
                c if c.is_whitespace() => {}
                _ => return None,
            }
            continue;
        }
        if in_string {
            match (escaped, c) {
                (true, _) => escaped = false,
                (false, '\\') => escaped = true,
                (false, '"') => in_string = false,
                _ => {}
            }
            continue;
        }
        match c {
            '"' => in_string = true,
            '{' => stack.push('}'),
            '[' => stack.push(']'),
            '}' | ']' => {
                if stack.pop() != Some(c) {
                    return None;
                }
                if stack.is_empty() {
                    return Some(idx + c.len_utf8());
                }
            }
            _ => {}
        }
    }
    None
}

/// Offset of the `</rfl:parameter>` that ends a value starting at `after_key`.
fn param_value_end(after_key: &str, framing: BeamArgumentFraming) -> Option<usize> {
    if framing == BeamArgumentFraming::ConstrainedJson
        && let Some(json_end) = json_value_end(after_key)
        && let Some(close) = after_key[json_end..].find(PARAM_CLOSE)
    {
        return Some(json_end + close);
    }
    after_key.find(PARAM_CLOSE)
}

/// What follows `<rfl:parameter name="`.
enum Opener {
    /// A non-empty name closed by `">`; the opener's remaining length is `key_len + 2`.
    Valid {
        key_len: usize,
    },
    Incomplete,
    Malformed,
}

/// A name must be non-empty and quoted with no further attributes.
fn parse_opener(after: &str) -> Opener {
    let Some(quote) = after.find('"') else {
        return Opener::Incomplete;
    };
    let rest = &after[quote..];
    if quote == 0 {
        return Opener::Malformed;
    }
    if rest == "\"" {
        return Opener::Incomplete;
    }
    if rest.starts_with("\">") {
        Opener::Valid { key_len: quote }
    } else {
        Opener::Malformed
    }
}

use super::ToolDefinition;
use super::response::{CalledFunction, ToolCallResponse, ToolCallType};

pub const TOOL_CALL_START: &str = "<|start|>assistant to=";
pub const BARE_TOOL_CALL_START: &str = " to=";
pub const TOOL_CALL_END: &str = "<|call|>";
const START: &str = "<|start|>";
const CHANNEL: &str = "<|channel|>";
const MESSAGE: &str = "<|message|>";
const END: &str = "<|end|>";
const PARAM_OPEN: &str = "<rfl:parameter name=\"";
const PARAM_CLOSE: &str = "</rfl:parameter>";
const MAX_CHANNEL_HEADER: usize = 128;

/// Length of the longest valid recipient prefix of `text` (`[a-zA-Z_][a-zA-Z0-9_.-]*`).
fn recipient_len(text: &str) -> usize {
    let mut end = 0;
    for (i, c) in text.char_indices() {
        let ok = if i == 0 {
            c.is_ascii_alphabetic() || c == '_'
        } else {
            c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-')
        };
        if !ok {
            break;
        }
        end = i + c.len_utf8();
    }
    end
}

/// A tool header found in `text`.
struct Header {
    /// Where the header (`<|start|>assistant to=` or ` to=`) begins.
    start: usize,
    /// Just past `<|message|>`.
    end: usize,
    name: String,
}

/// `analysis`, optionally followed by the template's own ` <|constrain|>xml`.
fn is_tool_channel(text: &str) -> bool {
    let (name, rest) = match text.split_once(' ') {
        Some((name, rest)) => (name, Some(rest)),
        None => (text, None),
    };
    if name != "analysis" {
        return false;
    }
    rest.is_none_or(|rest| rest == "<|constrain|>xml")
}

/// Where the header whose `to=` sits at `at` begins, when what precedes `to=` is
/// `<|start|>assistant `, `assistant `, or the text start / whitespace the prompt left
/// after its consumed role. The flag says whether the full `<|start|>assistant` lead is
/// present.
fn header_lead(text: &str, at: usize) -> Option<(usize, bool)> {
    let pre = &text[..at];
    let trimmed = pre.trim_end_matches(' ');
    if let Some(role_head) = trimmed.strip_suffix("assistant") {
        return Some(match role_head.strip_suffix(START) {
            Some(head) => (head.len(), true),
            None => (role_head.len(), false),
        });
    }
    if trimmed.is_empty() {
        Some((0, false))
    } else if trimmed.len() != pre.len() {
        Some((trimmed.len(), false))
    } else {
        None
    }
}

/// The complete tool header whose `to=` sits at `at`.
fn header_at(text: &str, at: usize) -> Option<Header> {
    let name_len = recipient_len(&text[at + 3..]);
    if name_len == 0 {
        return None;
    }
    let name_end = at + 3 + name_len;
    let after = text[name_end..].strip_prefix(CHANNEL)?;
    let (start, _) = header_lead(text, at)?;
    let limit = (after.len()).min(MAX_CHANNEL_HEADER + MESSAGE.len());
    let window = &after[..floor_boundary(after, limit)];
    let mi = window.find(MESSAGE)?;
    if !is_tool_channel(&window[..mi]) {
        return None;
    }
    Some(Header {
        start,
        end: name_end + CHANNEL.len() + mi + MESSAGE.len(),
        name: text[at + 3..name_end].to_string(),
    })
}

/// The first complete tool header at or after `from`.
fn find_header(text: &str, from: usize) -> Option<Header> {
    let mut cursor = from;
    while let Some(rel) = text[cursor..].find("to=") {
        let at = cursor + rel;
        cursor = at + 3;
        if let Some(header) = header_at(text, at) {
            return Some(header);
        }
    }
    None
}

/// Whether `text` holds a header that is recognizably a tool call but can never
/// complete: `<|start|>assistant to=NAME` that stops (or is cut) before its channel,
/// or `to=NAME<|channel|>` whose channel/message is not the tool channel. Ordinary prose
/// that merely mentions `to=` is not one.
fn has_broken_header(text: &str) -> bool {
    let mut cursor = 0;
    while let Some(rel) = text[cursor..].find("to=") {
        let at = cursor + rel;
        cursor = at + 3;
        let name_len = recipient_len(&text[at + 3..]);
        if name_len == 0 || header_at(text, at).is_some() {
            continue;
        }
        let Some((_, framed)) = header_lead(text, at) else {
            continue;
        };
        let tail = &text[at + 3 + name_len..];
        let after_name_is_header_part =
            tail.starts_with(CHANNEL) || (framed && (tail.is_empty() || CHANNEL.starts_with(tail)));
        if after_name_is_header_part {
            return true;
        }
    }
    false
}

fn floor_boundary(text: &str, mut at: usize) -> usize {
    while !text.is_char_boundary(at) {
        at -= 1;
    }
    at
}

/// How one tool body ends.
enum Walk {
    /// `body_end` is where the parameters stop; `consume` includes the terminator.
    Complete {
        body_end: usize,
        consume: usize,
    },
    /// Valid so far, terminator not yet seen.
    Incomplete,
    Malformed(&'static str),
}

/// Walk the body as the template's tag search, then require `<|call|>` / `<|end|>`:
/// complete `<rfl:parameter>` regions are stepped over, junk between them and a
/// malformed opener are skipped. A parameter that never closes is malformed (the
/// builder's contract rejects a partial call), and a framed next header never closes
/// the call.
fn walk_body(body: &str, framing: BeamArgumentFraming, lenient: bool) -> Walk {
    let mut at = 0;
    loop {
        let rest = &body[at..];
        let t = rest.trim_start_matches([' ', '\t', '\n', '\r']);
        at += rest.len() - t.len();
        if t.is_empty() {
            return Walk::Incomplete;
        }
        if let Some(term) = [TOOL_CALL_END, END].iter().find(|m| t.starts_with(**m)) {
            return Walk::Complete {
                body_end: at,
                consume: at + term.len(),
            };
        }
        if t.starts_with(START) {
            return Walk::Malformed("missing call terminator");
        }
        if let Some(after) = t.strip_prefix(PARAM_OPEN) {
            let key_end = match parse_opener(after) {
                Opener::Valid { key_len } => key_len,
                Opener::Incomplete => return Walk::Incomplete,
                Opener::Malformed if lenient => {
                    // Junk: the search resumes right after the opener.
                    at += PARAM_OPEN.len();
                    continue;
                }
                Opener::Malformed => return Walk::Malformed("malformed parameter opener"),
            };
            let value = &after[key_end + 2..];
            match param_value_end(value, framing) {
                Some(end) => {
                    at += PARAM_OPEN.len() + key_end + 2 + end + PARAM_CLOSE.len();
                    continue;
                }
                // No close yet: the region may still close (a marker inside a value is
                // data), so the call is incomplete; on a finished text that is an error.
                None => return Walk::Incomplete,
            }
        }
        if [TOOL_CALL_END, END, START, PARAM_OPEN]
            .iter()
            .any(|m| m.starts_with(t))
        {
            return Walk::Incomplete;
        }
        if !lenient {
            return Walk::Malformed("unexpected content in tool body");
        }
        // Junk between parameters: resume at the next region, terminator or header.
        let next = [PARAM_OPEN, TOOL_CALL_END, END, START]
            .iter()
            .filter_map(|m| t.find(m))
            .min();
        match next {
            Some(p) => at += p,
            None => return Walk::Incomplete,
        }
    }
}

/// Ordered `(key, raw value)` pairs of a complete body; at most one LF is stripped at
/// each edge of a value, the last duplicate wins and keeps its first position.
fn parse_parameters(body: &str, framing: BeamArgumentFraming) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    let mut at = 0;
    while let Some(rel) = body[at..].find(PARAM_OPEN) {
        let after_open = &body[at + rel + PARAM_OPEN.len()..];
        let Opener::Valid { key_len: key_end } = parse_opener(after_open) else {
            // A malformed opener is junk the tag search skips.
            at += rel + PARAM_OPEN.len();
            continue;
        };
        let key = &after_open[..key_end];
        let after_key = &after_open[key_end + 2..];
        let Some(value_end) = param_value_end(after_key, framing) else {
            break;
        };
        let mut value = &after_key[..value_end];
        value = value.strip_prefix('\n').unwrap_or(value);
        value = value.strip_suffix('\n').unwrap_or(value);
        match out.iter_mut().find(|(k, _)| k == key) {
            Some(slot) => slot.1 = value.to_string(),
            None => out.push((key.to_string(), value.to_string())),
        }
        at += rel + PARAM_OPEN.len() + key_end + 2 + value_end + PARAM_CLOSE.len();
    }
    out
}

/// The JSON-schema type names a parameter schema declares, in order.
fn schema_types(schema: &serde_json::Value) -> Vec<String> {
    let serde_json::Value::Object(map) = schema else {
        return Vec::new();
    };
    let mut types: Vec<String> = Vec::new();
    match map.get("type") {
        Some(serde_json::Value::String(t)) => types.push(t.clone()),
        Some(serde_json::Value::Array(list)) => {
            types.extend(list.iter().filter_map(|v| v.as_str()).map(str::to_string))
        }
        _ => {}
    }
    for union in ["anyOf", "oneOf"] {
        if let Some(serde_json::Value::Array(choices)) = map.get(union) {
            for choice in choices {
                types.extend(schema_types(choice));
            }
        }
    }
    let nullable = map
        .get("nullable")
        .is_some_and(|v| PyValue::from_json(v).is_truthy());
    if nullable && !types.iter().any(|t| t == "null") {
        types.push("null".to_string());
    }
    types
}

fn json_type_names(value: &PyValue) -> &'static [&'static str] {
    match value {
        PyValue::Bool(_) => &["boolean"],
        PyValue::Int(_) => &["integer", "number"],
        PyValue::Float(_) => &["number"],
        PyValue::None => &["null"],
        PyValue::Dict(_) => &["object"],
        PyValue::List(_) => &["array"],
        PyValue::Str(_) => &["string"],
    }
}

/// `raw` typed by the first of `types` it can be read as, else `raw` unchanged.
fn coerce(raw: &str, types: &[String]) -> PyValue {
    let mut filtered: Vec<&str> = types.iter().map(String::as_str).collect();
    let has_string = filtered.contains(&"string");
    if has_string && filtered.iter().any(|t| *t != "string") {
        let Some(decoded) = py::json_loads(raw) else {
            return PyValue::Str(raw.to_string());
        };
        if raw != py::strip(raw) {
            return PyValue::Str(raw.to_string());
        }
        let names = json_type_names(&decoded);
        filtered.retain(|t| names.contains(t));
    }
    for type_name in filtered {
        match type_name {
            "integer" => {
                if let Some(i) = py::parse_int(raw) {
                    return PyValue::Int(i);
                }
            }
            "number" => {
                if let Some(i) = py::parse_int(raw) {
                    return PyValue::Int(i);
                }
                let Some(number) = py::parse_float(raw) else {
                    continue;
                };
                if !number.is_finite() {
                    continue;
                }
                return if number.fract() == 0.0 && !raw.contains('.') {
                    PyValue::Int(py::int_from_float(number))
                } else {
                    PyValue::Float(number)
                };
            }
            "boolean" => match py::strip(raw).to_lowercase().as_str() {
                "true" | "1" => return PyValue::Bool(true),
                "false" | "0" => return PyValue::Bool(false),
                _ => {}
            },
            "null" => {
                if matches!(py::strip(raw), "null" | "None") {
                    return PyValue::None;
                }
            }
            "object" | "array" => {
                if let Some(decoded) = py::json_loads(raw) {
                    let wanted = matches!(
                        (&decoded, type_name),
                        (PyValue::Dict(_), "object") | (PyValue::List(_), "array")
                    );
                    if wanted {
                        return decoded;
                    }
                }
            }
            _ => {}
        }
    }
    PyValue::Str(raw.to_string())
}

/// The call's `function.arguments` JSON text. The LAST definition of a tool name wins,
/// as in the reference's name -> schema map.
fn arguments_json(
    body: &str,
    name: &str,
    tools: Option<&[ToolDefinition]>,
    framing: BeamArgumentFraming,
) -> String {
    let prepared = tools
        .and_then(|tools| tools.iter().rev().find(|tool| tool.name == name))
        .and_then(|tool| tool.parameters.as_ref())
        .map(typing::typed_parameters);
    let properties = prepared
        .as_ref()
        .and_then(|parameters| parameters.get("properties"))
        .and_then(|p| p.as_object());
    let mut args = PyDict::new();
    for (key, raw) in parse_parameters(body, framing) {
        let schema = properties.and_then(|p| p.get(&key));
        let types = schema.map(schema_types).unwrap_or_default();
        let value = if schema.is_some_and(|schema| typing::string_literal_admits(schema, &raw)) {
            PyValue::Str(raw)
        } else {
            coerce(&raw, &types)
        };
        args.insert(key, value);
    }
    PyValue::Dict(args).dumps()
}

/// [`try_tool_call_parse_beam_with`] with the default (`Lenient`) argument framing.
pub fn try_tool_call_parse_beam(
    message: &str,
    tools: Option<&[ToolDefinition]>,
) -> anyhow::Result<(Vec<ToolCallResponse>, Option<String>)> {
    try_tool_call_parse_beam_with(message, tools, BeamArgumentFraming::Lenient)
}

/// Parse every Beam tool call in `message` (the text the reasoning parser forwarded).
///
/// Returns the calls and the visible text around them: before, between and after, in
/// order, with no markers and no duplicated bytes. Any call that cannot complete, and any
/// recognizable tool header that can never complete, is an `Err`.
pub fn try_tool_call_parse_beam_with(
    message: &str,
    tools: Option<&[ToolDefinition]>,
    framing: BeamArgumentFraming,
) -> anyhow::Result<(Vec<ToolCallResponse>, Option<String>)> {
    let mut calls = Vec::new();
    let mut normal = String::new();
    let mut cursor = 0;
    let mut header = find_header(message, 0);
    while let Some(h) = header {
        normal.push_str(&message[cursor..h.start.max(cursor)]);
        let body = &message[h.end..];
        match walk_body(body, framing, true) {
            Walk::Complete { body_end, consume } => {
                calls.push(ToolCallResponse {
                    id: format!("call-{}", Uuid::new_v4()),
                    tp: ToolCallType::Function,
                    function: CalledFunction {
                        arguments: arguments_json(&body[..body_end], &h.name, tools, framing),
                        name: h.name,
                    },
                });
                cursor = h.end + consume;
                header = find_header(message, cursor);
            }
            Walk::Incomplete => anyhow::bail!("beam tool call `{}` is truncated", h.name),
            Walk::Malformed(why) => anyhow::bail!("beam tool call `{}`: {why}", h.name),
        }
    }
    if has_broken_header(&message[cursor..]) {
        anyhow::bail!("beam tool-call header is incomplete or not on the tool channel");
    }
    normal.push_str(&message[cursor..]);
    Ok((calls, Some(normal)))
}

/// Whether `chunk` holds, or may be the beginning of, a Beam tool header.
pub fn detect_tool_call_start_beam(chunk: &str) -> bool {
    if chunk.contains(TOOL_CALL_START)
        || find_header(chunk, 0).is_some()
        || has_broken_header(chunk)
    {
        return true;
    }
    [TOOL_CALL_START, BARE_TOOL_CALL_START]
        .iter()
        .any(|marker| {
            (1..marker.len()).any(|i| marker.is_char_boundary(i) && chunk.ends_with(&marker[..i]))
        })
}

/// [`find_tool_call_end_position_beam_with`] with the default argument framing.
pub fn find_tool_call_end_position_beam(chunk: &str) -> Option<usize> {
    find_tool_call_end_position_beam_with(chunk, BeamArgumentFraming::Lenient)
}

/// Offset just past the last COMPLETE (terminated) call in `chunk`, or `None`. Only
/// complete `<rfl:parameter>` regions are skipped, so a `<|call|>` inside a value never
/// ends the call.
pub fn find_tool_call_end_position_beam_with(
    chunk: &str,
    framing: BeamArgumentFraming,
) -> Option<usize> {
    match scan_completion(chunk, framing) {
        BeamCompletion::Complete(end) => Some(end),
        _ => None,
    }
}

/// Where a jailed Beam buffer stands.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BeamCompletion {
    /// No complete call yet; more bytes may still finish one.
    Incomplete,
    /// Every call through this offset is complete.
    Complete(usize),
    /// The first call can never complete.
    Invalid(String),
}

/// [`scan_completion_with`] with the lenient (tag-search) body reading.
pub fn scan_completion(chunk: &str, framing: BeamArgumentFraming) -> BeamCompletion {
    scan_completion_with(chunk, framing, true)
}

/// Walk `chunk` call by call. A malformed call after at least one complete call ends
/// the completed span there (the rest is handled as its own jail); a malformed first
/// call is [`BeamCompletion::Invalid`]. With `lenient` the body is read as the
/// template's tag search (junk and malformed openers skipped); without it, as the
/// strict request policy wants, they make the call invalid.
pub fn scan_completion_with(
    chunk: &str,
    framing: BeamArgumentFraming,
    lenient: bool,
) -> BeamCompletion {
    let mut end = None;
    let mut header = find_header(chunk, 0);
    while let Some(h) = header {
        match walk_body(&chunk[h.end..], framing, lenient) {
            Walk::Complete { consume, .. } => {
                let done = h.end + consume;
                end = Some(done);
                header = find_header(chunk, done);
            }
            Walk::Incomplete => break,
            Walk::Malformed(why) => {
                if end.is_none() {
                    return BeamCompletion::Invalid(format!("beam tool call `{}`: {why}", h.name));
                }
                break;
            }
        }
    }
    match end {
        Some(end) => BeamCompletion::Complete(end),
        None => BeamCompletion::Incomplete,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const HDR: &str =
        "<|start|>assistant to=weather<|channel|>analysis <|constrain|>xml<|message|>";
    const LONDON: &str = "<rfl:parameter name=\"location\">\nLondon\n</rfl:parameter>";

    fn tool(name: &str, props: serde_json::Value) -> ToolDefinition {
        ToolDefinition {
            name: name.into(),
            parameters: Some(json!({"type": "object", "properties": props})),
            strict: None,
        }
    }

    fn args(call: &ToolCallResponse) -> serde_json::Value {
        serde_json::from_str(&call.function.arguments).unwrap()
    }

    #[test]
    fn parses_full_bare_and_parallel_headers() {
        for text in [
            format!("{HDR}{LONDON}<|call|>"),
            format!(" to=weather<|channel|>analysis <|constrain|>xml<|message|>{LONDON}<|call|>"),
            format!("to=weather<|channel|>analysis<|message|>{LONDON}<|end|>"),
        ] {
            let (calls, _) = try_tool_call_parse_beam(&text, None).unwrap();
            assert_eq!(calls.len(), 1, "{text}");
            assert_eq!(args(&calls[0]), json!({"location": "London"}));
        }
        let two = format!("{HDR}{LONDON}<|end|>{HDR}{LONDON}<|call|>");
        let (calls, normal) = try_tool_call_parse_beam(&two, None).unwrap();
        assert_eq!((calls.len(), normal.as_deref()), (2, Some("")));
    }

    #[test]
    fn rejects_truncated_malformed_and_unterminated_calls() {
        for text in [
            format!("{HDR}<rfl:parameter name=\"location\">\nLondon\n"),
            format!("{HDR}<rfl:parameter name=\"location\">\nLondon\n</wrong><|call|>"),
            format!("{HDR}{LONDON}"),
            HDR.to_string(),
            format!("{HDR}{LONDON}<|start|>assistant<|channel|>final<|message|>x"),
        ] {
            assert!(try_tool_call_parse_beam(&text, None).is_err(), "{text}");
        }
    }

    #[test]
    fn plain_text_and_prefix_are_not_calls() {
        let (calls, normal) = try_tool_call_parse_beam("go to=home now", None).unwrap();
        assert!(calls.is_empty());
        assert_eq!(normal.as_deref(), Some("go to=home now"));
        let (calls, normal) = try_tool_call_parse_beam(&format!("hi {HDR}<|call|>"), None).unwrap();
        assert_eq!((calls.len(), normal.as_deref()), (1, Some("hi ")));
        assert_eq!(calls[0].function.arguments, "{}");
    }

    #[test]
    fn typing_last_definition_duplicates_and_entities() {
        let tools = [
            tool("weather", json!({"n": {"type": "string"}})),
            tool(
                "weather",
                json!({"n": {"type": "integer"}, "b": {"type": "boolean"}}),
            ),
        ];
        let text = format!(
            "{HDR}<rfl:parameter name=\"n\">1</rfl:parameter><rfl:parameter name=\"n\">\n7\n</rfl:parameter><rfl:parameter name=\"b\">1</rfl:parameter><rfl:parameter name=\"s\">&amp; <|call|></rfl:parameter><|call|>"
        );
        let (calls, _) = try_tool_call_parse_beam(&text, Some(&tools)).unwrap();
        assert_eq!(
            calls[0].function.arguments,
            r#"{"n": 7, "b": true, "s": "&amp; <|call|>"}"#
        );
    }

    #[test]
    fn visible_text_before_between_and_after_calls_is_kept_once() {
        let text = format!("BEFORE {HDR}{LONDON}<|call|> MID {HDR}{LONDON}<|end|> AFTER");
        let (calls, normal) = try_tool_call_parse_beam(&text, None).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(normal.as_deref(), Some("BEFORE  MID  AFTER"));
    }

    #[test]
    fn inner_terminators_in_a_value_are_data() {
        let text =
            format!("{HDR}<rfl:parameter name=\"x\">a<|call|>b<|end|></rfl:parameter><|call|>");
        let (calls, _) = try_tool_call_parse_beam(&text, None).unwrap();
        assert_eq!(args(&calls[0]), json!({"x": "a<|call|>b<|end|>"}));
        assert_eq!(
            scan_completion(
                &format!("{HDR}<rfl:parameter name=\"x\">a<|call|>"),
                BeamArgumentFraming::Lenient
            ),
            BeamCompletion::Incomplete
        );
        let done = format!("{text}AFTER");
        assert_eq!(
            find_tool_call_end_position_beam(&done),
            Some(done.len() - "AFTER".len())
        );
    }

    /// A malformed opener (empty name, extra attribute) is junk the template's tag
    /// search skips: the call still completes, without that parameter.
    #[test]
    fn malformed_openers_are_skipped_junk() {
        for opener in [
            "<rfl:parameter name=\"\">",
            "<rfl:parameter name=\"a\" extra=\"b\">",
        ] {
            let text = format!("{HDR}{opener}x</rfl:parameter>{LONDON}<|call|>");
            let (calls, _) = try_tool_call_parse_beam(&text, None).unwrap();
            assert_eq!(args(&calls[0]), json!({"location": "London"}), "{opener}");
            assert_eq!(
                scan_completion(&text, BeamArgumentFraming::Lenient),
                BeamCompletion::Complete(text.len())
            );
        }
        // Junk between parameters (including non-ASCII whitespace) is skipped too.
        let text = format!("{HDR}{LONDON}garbage\u{2003}{LONDON}<|call|>");
        let (calls, _) = try_tool_call_parse_beam(&text, None).unwrap();
        assert_eq!(args(&calls[0]), json!({"location": "London"}));
    }

    #[test]
    fn recognizable_unfinished_headers_are_errors_but_prose_is_not() {
        for text in [
            "to=weather<|channel|>analysis",
            " to=weather<|channel|>final <|constrain|>xml<|message|>",
            "<|start|>assistant to=weather",
        ] {
            assert!(try_tool_call_parse_beam(text, None).is_err(), "{text}");
            assert!(detect_tool_call_start_beam(text), "{text}");
        }
        let (calls, normal) =
            try_tool_call_parse_beam("set to=home, then go to=work", None).unwrap();
        assert!(calls.is_empty());
        assert_eq!(normal.as_deref(), Some("set to=home, then go to=work"));
    }

    #[test]
    fn constrained_json_owns_strings_but_lenient_stops_at_first_closer() {
        let value = r#"{"text": "</rfl:parameter>"}"#;
        let text = format!("{HDR}<rfl:parameter name=\"o\">{value}</rfl:parameter><|call|>");
        let tools = [tool("weather", json!({"o": {"type": "object"}}))];
        let (calls, _) = try_tool_call_parse_beam_with(
            &text,
            Some(&tools),
            BeamArgumentFraming::ConstrainedJson,
        )
        .unwrap();
        assert_eq!(args(&calls[0]), json!({"o": {"text": "</rfl:parameter>"}}));
        // Lenient: the embedded closer ends the value and the leftover is skipped junk.
        let (calls, _) = try_tool_call_parse_beam(&text, Some(&tools)).unwrap();
        assert_eq!(args(&calls[0]), json!({"o": "{\"text\": \""}));
    }

    #[test]
    fn serving_typing_resolves_refs_and_single_allof() {
        let tools = [ToolDefinition {
            name: "weather".into(),
            parameters: Some(json!({
                "type": "object",
                "properties": {
                    "x": {"$ref": "#/$defs/N"},
                    "y": {"allOf": [{"type": "integer"}]},
                    "z": {"$ref": "#/$defs/C"},
                    "e": {"enum": [1, 2]},
                    "u": {"type": ["string", "null"]},
                },
                "$defs": {"N": {"type": "integer"}, "C": {"$ref": "#/$defs/C"}},
            })),
            strict: None,
        }];
        let body = ["x", "y", "z", "e", "u"]
            .iter()
            .map(|k| format!("<rfl:parameter name=\"{k}\">7</rfl:parameter>"))
            .collect::<String>();
        let (calls, _) =
            try_tool_call_parse_beam(&format!("{HDR}{body}<|call|>"), Some(&tools)).unwrap();
        assert_eq!(
            calls[0].function.arguments,
            r#"{"x": 7, "y": 7, "z": "7", "e": 7, "u": "7"}"#
        );
    }

    #[test]
    fn detect_and_end_position() {
        assert!(detect_tool_call_start_beam(HDR));
        assert!(detect_tool_call_start_beam("text <|start|>assis"));
        assert!(!detect_tool_call_start_beam("plain text"));
        let full = format!("{HDR}{LONDON}<|call|>tail");
        assert_eq!(
            find_tool_call_end_position_beam(&full),
            Some(full.len() - "tail".len())
        );
        assert_eq!(
            find_tool_call_end_position_beam(&format!("{HDR}{LONDON}")),
            None
        );
    }
}
