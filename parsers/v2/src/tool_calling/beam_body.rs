// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pure Beam grammar helpers: recipient names, the strict `<rfl:parameter>` body
//! decoder and schema-driven argument typing.
//!
//! No scanning, buffering or streaming lives here; a scanner hands over one complete
//! tool-channel body. The schema coercion (`schema_types` / `coerce`) is derived from the
//! model builder's frozen reference (upstream copyright kept in the derived file's SPDX header) and
//! relies on [`super::beam_py`] for Python-compatible primitives.
//!
//! Body policy, checked against the reference tag regex
//! `<rfl:parameter name="(?P<key>[^"]+)">\n?(?P<value>.*?)\n?</rfl:parameter>`:
//!
//! * at most ONE LF is removed at each edge of a value; no CRLF normalisation, no
//!   trimming, no entity unescaping (`&amp;` stays literal);
//! * unlike the reference regex, which silently skips what it cannot match, every
//!   parameter region must be well formed and closed, and only ASCII whitespace may sit
//!   between parameters, so a malformed or truncated body is an error and never `{}`;
//! * the last duplicate key wins and keeps its first position.

use crate::tool_calling::beam_py::{self, PyDict, PyValue};
use crate::tool_calling::scan::JsonStringState;
use crate::tool_calling::traits::Tool;
use crate::unified::ArgumentFraming;
use serde_json::{Map, Value};
use std::collections::{BTreeSet, HashMap, HashSet};

const PARAM_OPEN: &str = "<rfl:parameter name=\"";
const PARAM_CLOSE: &str = "</rfl:parameter>";
/// The serving grammar lets every parameter but the last end by opening the next one
/// (`\n<rfl:parameter name="`) without writing its closer; the newline is the value's
/// framing newline and the opener belongs to the next parameter.
pub(crate) const PARAM_FOLLOW: &str = "\n<rfl:parameter name=\"";

/// Why a Beam body or header could not be decoded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BeamBodyError {
    /// Non-whitespace bytes where a `<rfl:parameter` opener was required.
    UnexpectedContent { offset: usize },
    /// A parameter opener whose name quote or `">` is missing, or whose name is empty.
    MalformedOpener { offset: usize },
    /// A parameter that never reached `</rfl:parameter>` (truncated or mismatched close).
    UnclosedParameter { offset: usize },
}

impl std::fmt::Display for BeamBodyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnexpectedContent { offset } => {
                write!(f, "unexpected content in Beam tool body at byte {offset}")
            }
            Self::MalformedOpener { offset } => {
                write!(f, "malformed rfl:parameter opener at byte {offset}")
            }
            Self::UnclosedParameter { offset } => {
                write!(f, "unclosed rfl:parameter at byte {offset}")
            }
        }
    }
}

impl std::error::Error for BeamBodyError {}

/// A recipient (tool name) in a call header: `[A-Za-z0-9_.-]+`, the one name contract
/// the structural-tag builder holds declared tools to (`name_encodable`), so a call the
/// grammar admits for a declared tool (`2fa_check`, `-tool`) is never dropped here.
pub fn is_valid_recipient(name: &str) -> bool {
    !name.is_empty() && recipient_len(name) == name.len()
}

/// Whether `c` may appear anywhere in a recipient.
pub(crate) fn is_recipient_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-')
}

/// Length of the longest valid recipient prefix of `text` (0 when none).
pub fn recipient_len(text: &str) -> usize {
    text.find(|c: char| !is_recipient_char(c))
        .unwrap_or(text.len())
}

/// How far a JSON value that has STARTED at byte zero of `text` has got.
enum JsonSpan {
    /// Byte offset just past the value.
    Complete(usize),
    /// Still open: more bytes can complete it.
    Open,
    /// Can never become a JSON value from here (mismatched closer).
    Invalid,
}

/// Where the walk over one parameter value has got, so a value that arrives in many
/// pushes is scanned once rather than from its first byte on every push (a 64 KiB
/// string argument in 8-byte chunks otherwise costs seconds of CPU: the walk doubled
/// its cost with every doubling of the value). The caller keeps one cursor per open
/// parameter and hands it the same value prefix, extended, on each push.
#[derive(Clone, Debug, Default)]
pub(crate) struct ValueCursor {
    /// Bytes of the (lead-trimmed) JSON value already walked.
    json_scanned: usize,
    json_started: bool,
    json_quoted: JsonStringState,
    json_stack: Vec<char>,
    /// `Some(Ok(end))` once the value completed, `Some(Err(()))` once it can never.
    json_done: Option<Result<usize, ()>>,
    /// Bytes of the remainder after the JSON value already searched for a closer.
    closer_searched: usize,
}

impl ValueCursor {
    /// Continue the JSON walk over `text`, which extends the text of the previous call.
    fn json_span(&mut self, text: &str) -> JsonSpan {
        if let Some(done) = self.json_done {
            return match done {
                Ok(end) => JsonSpan::Complete(end),
                Err(()) => JsonSpan::Invalid,
            };
        }
        if text.len() < self.json_scanned {
            // Not an extension of what was walked: start over.
            *self = Self::default();
        }
        let base = self.json_scanned;
        let mut chars = text[base..].char_indices().map(|(i, c)| (base + i, c));
        if !self.json_started {
            let Some((_, first)) = chars.next() else {
                return JsonSpan::Open;
            };
            match first {
                '"' => {
                    self.json_quoted.advance('"');
                }
                '{' => self.json_stack.push('}'),
                '[' => self.json_stack.push(']'),
                _ => {
                    self.json_done = Some(Err(()));
                    return JsonSpan::Invalid;
                }
            }
            self.json_started = true;
            self.json_scanned = first.len_utf8();
        }
        for (idx, c) in chars {
            self.json_scanned = idx + c.len_utf8();
            if self.json_quoted.advance(c) {
                if self.json_stack.is_empty() && c == '"' {
                    // The closing quote of a bare string value.
                    let mut probe = self.json_quoted;
                    if !probe.advance('x') {
                        self.json_done = Some(Ok(self.json_scanned));
                        return JsonSpan::Complete(self.json_scanned);
                    }
                }
                continue;
            }
            match c {
                '{' => self.json_stack.push('}'),
                '[' => self.json_stack.push(']'),
                '}' | ']' => {
                    if self.json_stack.pop() != Some(c) {
                        self.json_done = Some(Err(()));
                        return JsonSpan::Invalid;
                    }
                    if self.json_stack.is_empty() {
                        self.json_done = Some(Ok(self.json_scanned));
                        return JsonSpan::Complete(self.json_scanned);
                    }
                }
                _ => {}
            }
        }
        JsonSpan::Open
    }

    /// Earliest `needle` in `rest`, searching only the bytes no earlier call has seen
    /// (less one needle length, for a needle split across pushes).
    fn find_closer(&self, rest: &str, needle: &str) -> Option<usize> {
        let from = self.closer_searched.saturating_sub(needle.len() - 1);
        let from = (from..=rest.len()).find(|&i| rest.is_char_boundary(i))?;
        rest[from..].find(needle).map(|c| from + c)
    }
}

/// The parameter keys of ONE held tool (one the serving grammar types, see
/// `structural_tag::builders::beam::beam_holds_tool`) whose values the grammar emits as
/// JSON: the grammar's own type list for the normalized schema is non-empty and contains
/// no `string`. Every other key of that tool is raw text.
///
/// Built once per tool from [`typed_parameters`] + [`framing_types`], so neither the
/// streaming boundary nor the final body decoder re-reads a schema per chunk.
///
/// WIRE FRAMING, NOT TYPING. The serving grammar that produced the bytes decided raw
/// text versus JSON with the frozen reference's UNION type list ([`framing_types`]);
/// ordinary argument typing uses the corrected conjunctive [`schema_types`]. Where the
/// two disagree the input grammar is the one that already emitted the value, so framing
/// must follow it or a raw slot is walked as JSON (an unfinished `{` never closes) and
/// a JSON slot is cut at an embedded closer.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct JsonKeys(BTreeSet<String>);

impl JsonKeys {
    /// The JSON-owned keys of a held tool's `parameters` schema.
    pub(crate) fn of_held_tool(parameters: &Value) -> Self {
        let prepared = typed_parameters(parameters);
        let keys = prepared
            .get("properties")
            .and_then(Value::as_object)
            .into_iter()
            .flatten()
            .filter(|(_, schema)| {
                let types = framing_types(schema);
                !types.is_empty() && !types.iter().any(|t| t == "string")
            })
            .map(|(key, _)| key.clone())
            .collect();
        Self(keys)
    }
}

/// The type list the native serving grammar framed a parameter slot by: the frozen
/// model builder's reference `schema_types`, which UNIONS a schema's own `type` (string or
/// array of strings) with every `anyOf`/`oneOf` branch recursively, then appends `null`
/// when `nullable` is Python-truthy. A branch without a type contributes nothing.
///
/// Deliberately NOT [`schema_types`]: that function intersects sibling keywords, which is
/// the right reading for typing an argument but not for recovering how the grammar
/// framed it. Under `{"type": "object", "anyOf": [{"type": "string"}, {"type":
/// "object"}]}` the grammar's list contains `string`, so the slot was emitted as raw
/// text and may legitimately hold an unfinished `{`; the conjunctive reading would
/// claim it as JSON and wait forever. Private to the framing decision; the typing and
/// coercion contract is unchanged.
fn framing_types(schema: &Value) -> Vec<String> {
    let Value::Object(map) = schema else {
        return Vec::new();
    };
    let mut types: Vec<String> = match map.get("type") {
        Some(Value::String(t)) => vec![t.clone()],
        Some(Value::Array(list)) => list
            .iter()
            .filter_map(|v| v.as_str())
            .map(str::to_string)
            .collect(),
        _ => Vec::new(),
    };
    for union in ["anyOf", "oneOf"] {
        if let Some(Value::Array(choices)) = map.get(union) {
            for choice in choices {
                for t in framing_types(choice) {
                    if !types.contains(&t) {
                        types.push(t);
                    }
                }
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

impl JsonKeys {
    pub(crate) fn contains(&self, key: &str) -> bool {
        self.0.contains(key)
    }

    #[cfg(test)]
    fn keys(&self) -> Vec<&str> {
        self.0.iter().map(String::as_str).collect()
    }
}

/// `name -> JsonKeys` over the tools the serving grammar holds (every tool whose schema
/// it can type, `strict` or not — the grammar holds every tool it can), inserted in order
/// so the LAST held declaration of a name wins. A tool the grammar cannot hold takes any
/// parameters as raw text and is absent here.
pub(crate) fn held_json_keys(tools: &[Tool]) -> HashMap<String, JsonKeys> {
    let mut out = HashMap::new();
    for tool in tools
        .iter()
        .filter(|t| crate::structural_tag::builders::beam::beam_holds_tool(t))
    {
        out.insert(tool.name.clone(), JsonKeys::of_held_tool(&tool.parameters));
    }
    out
}

/// The ONE per-key framing decision shared by the streaming boundary and the body
/// decoder: how the value of `key` ends. `Lenient` and `ConstrainedJson` are
/// request-global; `SchemaConstrained` resolves per key through the recipient's
/// [`JsonKeys`], and a key outside that set (or an empty set: unknown or unheld tool) is
/// raw text.
pub(crate) fn value_framing(
    framing: ArgumentFraming,
    json_keys: &JsonKeys,
    key: &str,
) -> ArgumentFraming {
    match framing {
        ArgumentFraming::SchemaConstrained if json_keys.contains(key) => {
            ArgumentFraming::ConstrainedJson
        }
        ArgumentFraming::SchemaConstrained => ArgumentFraming::Lenient,
        global => global,
    }
}

/// Where the value of a parameter ends: the offset of its `</rfl:parameter>`, or
/// `None` while the value is still open.
///
/// `Lenient` is the reference regex (first closer). `ConstrainedJson` applies only
/// where the request constrained the output to JSON: a value that opens as a JSON
/// string, object or array owns its strings, so a closer inside one is data, and an
/// OPEN value waits rather than falling back to the first literal closer — otherwise a
/// chunk ending inside `{"text":"</rfl:parameter><|call|>…"}` completed the call early.
/// A value that opens with JSON punctuation but can never balance, and any value that
/// does not open as JSON, keeps the reference rule.
///
/// `SchemaConstrained` is resolved per key by [`value_framing`] BEFORE this call; passed
/// here directly (no key metadata) it is the raw-text rule.
#[cfg(test)]
pub(crate) fn param_value_end(after_key: &str, framing: ArgumentFraming) -> Option<usize> {
    param_value_span(after_key, framing).map(|(end, _)| end)
}

/// Where a parameter region ends: `(value_end, next)` with the raw value (framing
/// newlines included) in `after_key[..value_end]` and the body resuming at `next`, or
/// `None` while the value is still open.
///
/// A value ends at its `</rfl:parameter>`, or, as the serving grammar admits for every
/// parameter but the last, where the next parameter opens (`\n<rfl:parameter name="`):
/// the newline is the value's framing newline and the opener is read again as the next
/// parameter. Whichever comes first ends the value; under `ConstrainedJson` both are
/// looked for only after the JSON value the parameter opened with.
pub(crate) fn param_value_span(
    after_key: &str,
    framing: ArgumentFraming,
) -> Option<(usize, usize)> {
    param_value_span_with(after_key, framing, &mut ValueCursor::default())
}

/// [`param_value_span`] resuming from `cursor`, which must have seen a prefix of
/// `after_key` (the same parameter value, extended by the bytes that arrived since).
pub(crate) fn param_value_span_with(
    after_key: &str,
    framing: ArgumentFraming,
    cursor: &mut ValueCursor,
) -> Option<(usize, usize)> {
    let mut from = 0;
    if framing == ArgumentFraming::ConstrainedJson {
        let lead = after_key.len() - after_key.trim_start_matches(['\n', ' ', '\t', '\r']).len();
        if after_key[lead..].starts_with(['"', '{', '[']) {
            match cursor.json_span(&after_key[lead..]) {
                JsonSpan::Complete(json_end) => from = lead + json_end,
                JsonSpan::Open => return None,
                JsonSpan::Invalid => {}
            }
        }
    }
    let rest = &after_key[from..];
    let close = cursor
        .find_closer(rest, PARAM_CLOSE)
        .map(|c| (from + c, from + c + PARAM_CLOSE.len()));
    let follow = cursor
        .find_closer(rest, PARAM_FOLLOW)
        .map(|f| (from + f + 1, from + f + 1));
    let span = match (close, follow) {
        (Some(close), Some(follow)) if follow.0 <= close.0 => Some(follow),
        (close, follow) => close.or(follow),
    };
    if span.is_none() {
        cursor.closer_searched = rest.len();
    }
    span
}

/// The reference template's `\n?VALUE\n?`: exactly one framing newline on each side.
pub(crate) fn unframe(value: &str) -> &str {
    let value = value.strip_prefix('\n').unwrap_or(value);
    value.strip_suffix('\n').unwrap_or(value)
}

/// Decode one complete tool-channel body into ordered `(key, raw value)` pairs under
/// one request-global framing.
pub fn parse_parameters(
    body: &str,
    framing: ArgumentFraming,
) -> Result<Vec<(String, String)>, BeamBodyError> {
    parse_parameters_by(body, |_| framing, false)
}

/// How a body that is not a clean sequence of parameters is read.
///
/// * strict (`lenient == false`): the first byte that is not whitespace, a complete
///   parameter or the end is an error — the shape the `Reject` policy wants to surface.
/// * lenient (`lenient == true`): the template's own body grammar, a tag search. Bytes
///   between complete parameters are skipped and a malformed opener is skipped; every
///   complete parameter still counts (last duplicate wins). A parameter that never
///   closes is an error in both readings (no partial call is ever served).
fn parse_parameters_by(
    body: &str,
    framing_for: impl Fn(&str) -> ArgumentFraming,
    lenient: bool,
) -> Result<Vec<(String, String)>, BeamBodyError> {
    let mut out: Vec<(String, String)> = Vec::new();
    let mut at = 0;
    loop {
        let rest = &body[at..];
        let trimmed = rest.trim_start_matches([' ', '\t', '\n', '\r']);
        at += rest.len() - trimmed.len();
        if trimmed.is_empty() {
            return Ok(out);
        }
        let Some(after_open) = trimmed.strip_prefix(PARAM_OPEN) else {
            if lenient {
                // Junk: resume at the next opener, or finish with what was collected.
                match trimmed.find(PARAM_OPEN) {
                    Some(next) => {
                        at += next;
                        continue;
                    }
                    None => return Ok(out),
                }
            }
            // A bare prefix of the opener is a truncation, anything else is garbage.
            return Err(if PARAM_OPEN.starts_with(trimmed) {
                BeamBodyError::UnclosedParameter { offset: at }
            } else {
                BeamBodyError::UnexpectedContent { offset: at }
            });
        };
        let opener = match after_open.find('"') {
            Some(key_end) if key_end > 0 => after_open[key_end..]
                .strip_prefix("\">")
                .map(|after_key| (key_end, after_key)),
            _ => None,
        };
        let Some((key_end, after_key)) = opener else {
            if lenient {
                // A malformed opener is junk; the search resumes right after it.
                at += PARAM_OPEN.len();
                continue;
            }
            return Err(BeamBodyError::MalformedOpener { offset: at });
        };
        let key = &after_open[..key_end];
        let framing = framing_for(key);
        let Some((value_end, next)) = param_value_span(after_key, framing) else {
            // A parameter that never closes is malformed in every reading: the builder's
            // contract rejects a partial call rather than serving the parameters around it.
            return Err(BeamBodyError::UnclosedParameter { offset: at });
        };
        let value = unframe(&after_key[..value_end]);
        match out.iter_mut().find(|(k, _)| k == key) {
            Some(slot) => slot.1 = value.to_string(),
            None => out.push((key.to_string(), value.to_string())),
        }
        at += PARAM_OPEN.len() + key_end + 2 + next;
    }
}

const JSON_TYPES: [&str; 7] = [
    "string", "number", "integer", "boolean", "array", "object", "null",
];
const ANNOTATIONS: [&str; 8] = [
    "$comment",
    "default",
    "deprecated",
    "description",
    "examples",
    "readOnly",
    "title",
    "writeOnly",
];
const K_OBJECT_KEYWORDS: [&str; 9] = [
    "additionalProperties",
    "dependentRequired",
    "dependentSchemas",
    "maxProperties",
    "minProperties",
    "patternProperties",
    "properties",
    "propertyNames",
    "required",
];
const K_ARRAY_KEYWORDS: [&str; 8] = [
    "contains",
    "items",
    "maxContains",
    "maxItems",
    "minContains",
    "minItems",
    "prefixItems",
    "uniqueItems",
];
const K_STRING_KEYWORDS: [&str; 4] = ["format", "maxLength", "minLength", "pattern"];
const K_NUMBER_KEYWORDS: [&str; 5] = [
    "exclusiveMaximum",
    "exclusiveMinimum",
    "maximum",
    "minimum",
    "multipleOf",
];

/// `_resolve_local_ref`: a `#/...` JSON pointer into `root`, if it names a schema.
/// `#/…` JSON Pointer into `root`. URI percent-decoding precedes JSON Pointer
/// unescaping (`%20`, UTF-8 escapes, then `~1` / `~0`); an invalid `%` escape or a
/// pointer that leaves the document resolves nothing.
fn resolve_local_ref<'a>(reference: &str, root: &'a Value) -> Option<&'a Value> {
    let fragment = reference.strip_prefix('#')?;
    let bytes = fragment.as_bytes();
    let invalid_escape = bytes.iter().enumerate().any(|(i, b)| {
        *b == b'%'
            && !bytes
                .get(i + 1..i + 3)
                .is_some_and(|digits| digits.iter().all(u8::is_ascii_hexdigit))
    });
    if invalid_escape {
        return None;
    }
    let pointer = percent_encoding::percent_decode_str(fragment)
        .decode_utf8()
        .ok()?;
    let path = pointer.strip_prefix('/')?;
    let mut value = root;
    for part in path.split('/') {
        let key = part.replace("~1", "/").replace("~0", "~");
        value = value.as_object()?.get(&key)?;
    }
    matches!(value, Value::Bool(_) | Value::Object(_)).then_some(value)
}

/// Whether a schema's `const` / `enum` admit `raw` itself as a string value. Such text
/// IS that string: a nullable alias of `{"type": "string", "const": "null"}` reads the
/// bare text `null` as the schema-valid string `"null"`, not as JSON null.
fn string_literal_admits(schema: &Value, raw: &str) -> bool {
    let Value::Object(map) = schema else {
        return false;
    };
    let text = beam_py::strip(raw);
    if matches!(map.get("const"), Some(Value::String(s)) if s == text) {
        return true;
    }
    matches!(map.get("enum"), Some(Value::Array(values)) if values.iter().any(|v| v.as_str() == Some(text)))
}

fn json_type(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(n) if n.is_i64() || n.is_u64() => "integer",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn ordered(types: &HashSet<&str>) -> Vec<&'static str> {
    JSON_TYPES
        .iter()
        .copied()
        .filter(|t| types.contains(t))
        .collect()
}

/// The JSON types a schema admits, following local refs and combinators.
fn admitted_types(
    schema: &Value,
    root: &Value,
    seen: Option<&HashSet<String>>,
) -> Vec<&'static str> {
    let map = match schema {
        Value::Bool(false) => return Vec::new(),
        Value::Object(map) => map,
        _ => return JSON_TYPES.to_vec(),
    };
    let mut seen_refs: HashSet<String> = seen.cloned().unwrap_or_default();
    if let Some(Value::String(reference)) = map.get("$ref")
        && !seen_refs.contains(reference)
        && let Some(target) = resolve_local_ref(reference, root)
    {
        seen_refs.insert(reference.clone());
        return admitted_types(target, root, Some(&seen_refs));
    }
    match map.get("type") {
        Some(Value::String(t)) => {
            return if JSON_TYPES.contains(&t.as_str()) {
                JSON_TYPES.iter().copied().filter(|j| j == t).collect()
            } else {
                JSON_TYPES.to_vec()
            };
        }
        Some(Value::Array(list)) => {
            return JSON_TYPES
                .iter()
                .copied()
                .filter(|j| list.iter().any(|t| t.as_str() == Some(j)))
                .collect();
        }
        _ => {}
    }
    for keyword in ["anyOf", "oneOf"] {
        if let Some(Value::Array(options)) = map.get(keyword) {
            let mut types = HashSet::new();
            for option in options {
                if matches!(option, Value::Bool(_) | Value::Object(_)) {
                    types.extend(admitted_types(option, root, Some(&seen_refs)));
                }
            }
            return ordered(&types);
        }
    }
    if let Some(Value::Array(options)) = map.get("allOf") {
        let all: HashSet<&str> = JSON_TYPES.iter().copied().collect();
        let constrained: Vec<HashSet<&str>> = options
            .iter()
            .filter(|o| matches!(o, Value::Bool(_) | Value::Object(_)))
            .map(|o| {
                admitted_types(o, root, Some(&seen_refs))
                    .into_iter()
                    .collect::<HashSet<&str>>()
            })
            .filter(|types| *types != all)
            .collect();
        if let Some(first) = constrained.first() {
            let widen = |types: &HashSet<&'static str>| {
                let mut out = types.clone();
                if types.contains("number") {
                    out.insert("integer");
                }
                out
            };
            let mut result = widen(first);
            for types in &constrained[1..] {
                let widened = widen(types);
                result = result.intersection(&widened).copied().collect();
            }
            if result.contains("number") && result.contains("integer") {
                result.remove("integer");
            }
            return ordered(&result);
        }
    }
    if let Some(value) = map.get("const") {
        return vec![json_type(value)];
    }
    if let Some(Value::Array(values)) = map.get("enum") {
        let types: HashSet<&str> = values.iter().map(json_type).collect();
        return ordered(&types);
    }
    let has = |keywords: &[&str]| keywords.iter().any(|k| map.contains_key(*k));
    if has(&K_OBJECT_KEYWORDS) {
        return vec!["object"];
    }
    if has(&K_ARRAY_KEYWORDS) {
        return vec!["array"];
    }
    if has(&K_STRING_KEYWORDS) {
        return vec!["string"];
    }
    if has(&K_NUMBER_KEYWORDS) {
        return vec!["number"];
    }
    JSON_TYPES.to_vec()
}

fn typed_schema(schema: &Value, root: &Value, seen: &BTreeSet<String>) -> Value {
    let Value::Object(map) = schema else {
        return schema.clone();
    };
    let aliased = ["$ref", "allOf"]
        .iter()
        .filter(|k| map.contains_key(**k))
        .count();
    let rest: Map<String, Value> = map
        .iter()
        .filter(|(k, _)| k.as_str() != "$ref" && k.as_str() != "allOf")
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    let mut seen = seen.clone();
    let mut target: Option<Value> = None;
    if aliased == 1
        && rest
            .keys()
            .all(|k| ANNOTATIONS.contains(&k.as_str()) || k == "nullable")
    {
        match (map.get("$ref"), map.get("allOf")) {
            (Some(Value::String(reference)), _) if !seen.contains(reference) => {
                target = resolve_local_ref(reference, root).cloned();
                seen.insert(reference.clone());
            }
            (_, Some(Value::Array(branches))) if branches.len() == 1 => {
                target = Some(branches[0].clone());
            }
            _ => {}
        }
    }
    if let Some(Value::Object(mut merged)) = target {
        for (k, v) in rest {
            merged.insert(k, v);
        }
        return typed_schema(&Value::Object(merged), root, &seen);
    }
    let mut schema = map.clone();
    for keyword in ["anyOf", "oneOf"] {
        if let Some(Value::Array(items)) = schema.get(keyword) {
            let items: Vec<Value> = items
                .iter()
                .map(|item| typed_schema(item, root, &seen))
                .collect();
            schema.insert(keyword.to_string(), Value::Array(items));
        }
    }
    if !["type", "anyOf", "oneOf"]
        .iter()
        .any(|k| schema.contains_key(*k))
    {
        let types = admitted_types(&Value::Object(schema.clone()), root, None);
        if !types.is_empty() && types.len() < JSON_TYPES.len() {
            let value = if types.len() == 1 {
                serde_json::json!(types[0])
            } else {
                serde_json::json!(types)
            };
            schema.insert("type".to_string(), value);
        }
    }
    Value::Object(schema)
}

/// Serving-parity schema preparation (`typed_parameters`): resolve annotated aliases
/// (a local `$ref` or a single-branch `allOf`) and infer the types argument decoding
/// needs, for each property. Beam only; other families keep their own typing.
pub fn typed_parameters(parameters: &Value) -> Value {
    let Some(Value::Object(properties)) = parameters.get("properties") else {
        return parameters.clone();
    };
    let typed: Map<String, Value> = properties
        .iter()
        .map(|(k, schema)| {
            (
                k.clone(),
                typed_schema(schema, parameters, &BTreeSet::new()),
            )
        })
        .collect();
    let mut out = parameters.as_object().cloned().unwrap_or_default();
    out.insert("properties".to_string(), Value::Object(typed));
    Value::Object(out)
}

/// The JSON-schema type names a parameter schema admits, in order.
///
/// Sibling keywords are CONJUNCTIVE, as JSON Schema defines them: a schema's own
/// `type` and its `anyOf`/`oneOf` branches must both hold, so the result is their
/// intersection. A branch without a type admits every type and so removes nothing.
///
/// Deliberate correction relative to the model builder's frozen reference, which UNIONS
/// the sibling `type` with the branch types: under that rule `{"type": "string",
/// "anyOf": [{"type": "string"}, {"type": "null"}]}` read the text `null` as JSON null
/// although the schema excludes it. Every schema the reference's own fixtures exercise
/// has no such sibling conflict, so their results are unchanged: a bare
/// `["string", "null"]` still yields JSON null, a plain `string` keeps the text `"null"`,
/// and `nullable: true` still adds `null`.
///
/// Typing only. How the native grammar FRAMED a slot (raw text or JSON) is decided by
/// the reference's union rule in [`framing_types`], because the grammar that emitted the
/// bytes used that rule; see [`JsonKeys`].
pub fn schema_types(schema: &serde_json::Value) -> Vec<String> {
    let serde_json::Value::Object(map) = schema else {
        return Vec::new();
    };
    let declared: Option<Vec<String>> = match map.get("type") {
        Some(serde_json::Value::String(t)) => Some(vec![t.clone()]),
        Some(serde_json::Value::Array(list)) => Some(
            list.iter()
                .filter_map(|v| v.as_str())
                .map(str::to_string)
                .collect(),
        ),
        _ => None,
    };
    let mut branches: Option<Vec<String>> = None;
    for union in ["anyOf", "oneOf"] {
        if let Some(serde_json::Value::Array(choices)) = map.get(union) {
            let mut union_types: Vec<String> = Vec::new();
            for choice in choices {
                let choice_types = schema_types(choice);
                if choice_types.is_empty() {
                    // An untyped branch admits every type.
                    union_types = JSON_TYPES.iter().map(|t| t.to_string()).collect();
                    break;
                }
                for t in choice_types {
                    if !union_types.contains(&t) {
                        union_types.push(t);
                    }
                }
            }
            branches = Some(match branches {
                None => union_types,
                Some(previous) => previous
                    .into_iter()
                    .filter(|t| union_types.contains(t))
                    .collect(),
            });
        }
    }
    let mut types = match (declared, branches) {
        (None, None) => Vec::new(),
        (Some(d), None) => d,
        (None, Some(b)) => b,
        (Some(d), Some(b)) => {
            let both: Vec<String> = d.iter().filter(|t| b.contains(t)).cloned().collect();
            // A contradictory schema admits nothing; keep the explicit declaration.
            if both.is_empty() { d } else { both }
        }
    };
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

/// Beam can emit an integer with a zero decimal fraction. Keep its digits exact;
/// this deliberately extends the frozen reference's Python `int()` conversion.
fn integral_decimal(raw: &str) -> Option<beam_py::PyInt> {
    let (whole, fraction) = beam_py::strip(raw).split_once('.')?;
    let digits = whole.strip_prefix(['-', '+']).unwrap_or(whole);
    if digits.is_empty()
        || !digits.bytes().all(|b| b.is_ascii_digit())
        || !fraction.bytes().all(|b| b == b'0')
    {
        return None;
    }
    beam_py::parse_int(whole)
}

/// `raw` typed by the first of `types` it can be read as, else `raw` unchanged.
pub(crate) fn coerce(raw: &str, types: &[String]) -> PyValue {
    let mut filtered: Vec<&str> = types.iter().map(String::as_str).collect();
    let has_string = filtered.contains(&"string");
    if has_string && filtered.iter().any(|t| *t != "string") {
        let Some(decoded) = beam_py::json_loads(raw) else {
            return PyValue::Str(raw.to_string());
        };
        if raw != beam_py::strip(raw) {
            return PyValue::Str(raw.to_string());
        }
        let names = json_type_names(&decoded);
        filtered.retain(|t| names.contains(t));
    }
    for type_name in filtered {
        match type_name {
            "integer" => {
                if let Some(i) = beam_py::parse_int(raw).or_else(|| integral_decimal(raw)) {
                    return PyValue::Int(i);
                }
            }
            "number" => {
                if let Some(i) = beam_py::parse_int(raw) {
                    return PyValue::Int(i);
                }
                let Some(number) = beam_py::parse_float(raw) else {
                    continue;
                };
                if !number.is_finite() {
                    continue;
                }
                return if number.fract() == 0.0 && !raw.contains('.') {
                    PyValue::Int(beam_py::int_from_float(number))
                } else {
                    PyValue::Float(number)
                };
            }
            "boolean" => match beam_py::strip(raw).to_lowercase().as_str() {
                "true" | "1" => return PyValue::Bool(true),
                "false" | "0" => return PyValue::Bool(false),
                _ => {}
            },
            "null" => {
                if matches!(beam_py::strip(raw), "null" | "None") {
                    return PyValue::None;
                }
            }
            "object" | "array" => {
                if let Some(decoded) = beam_py::json_loads(raw) {
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

/// Decode a complete body into the typed argument object for tool `name`.
///
/// An unknown tool, or a parameter the schema does not declare, has no declared types
/// and so keeps its text verbatim. That is a typing fallback only: whether an unknown
/// tool may be called at all is a request policy the caller applies separately.
///
/// `json_keys` is the recipient's [`JsonKeys`] (consulted only under
/// [`ArgumentFraming::SchemaConstrained`]); the streaming boundary that framed this
/// body must have used the same set so both agree where every value ends.
pub(crate) fn typed_arguments(
    body: &str,
    name: &str,
    tools: &[Tool],
    framing: ArgumentFraming,
    json_keys: &JsonKeys,
    lenient: bool,
) -> Result<PyDict, BeamBodyError> {
    let properties = recipient_properties(tools, name);
    let mut args = PyDict::new();
    for (key, raw) in
        parse_parameters_by(body, |key| value_framing(framing, json_keys, key), lenient)?
    {
        let value = typed_value(properties.as_ref().and_then(|p| p.get(&key)), &raw);
        args.insert(key, value);
    }
    Ok(args)
}

/// Whether tool `name` declares a schema that admits an empty argument body: an object
/// with no `required` properties (an empty `properties` map included). `None` for an
/// unknown tool. The serving grammar makes the whole parameter body optional for such a
/// tool, so a header the stream ends after is a complete call `{}`.
pub(crate) fn recipient_admits_empty_arguments(tools: &[Tool], name: &str) -> Option<bool> {
    let tool = tools.iter().rev().find(|tool| tool.name == name)?;
    let schema = typed_parameters(&tool.parameters);
    let required = match schema.get("required") {
        Some(Value::Array(required)) => !required.is_empty(),
        _ => false,
    };
    Some(!required)
}

/// The typed `properties` of tool `name`. The reference builds its name -> schema map
/// in order, so the LAST definition wins; `None` for an unknown tool or a schema
/// without an object `properties` map (every parameter then keeps its text).
pub(crate) fn recipient_properties(tools: &[Tool], name: &str) -> Option<Map<String, Value>> {
    let tool = tools.iter().rev().find(|tool| tool.name == name)?;
    match typed_parameters(&tool.parameters).get("properties") {
        Some(Value::Object(properties)) => Some(properties.clone()),
        _ => None,
    }
}

/// The typed value of one parameter: a schema `const`/`enum` naming the text keeps it a
/// string, otherwise the declared types coerce it ([`coerce`]). The streaming emitter and
/// the completed-body reading both go through here, so they can never disagree.
pub(crate) fn typed_value(schema: Option<&Value>, raw: &str) -> PyValue {
    if schema.is_some_and(|schema| string_literal_admits(schema, raw)) {
        return PyValue::Str(raw.to_string());
    }
    let types = schema.map(schema_types).unwrap_or_default();
    coerce(raw, &types)
}

/// Whether [`typed_value`] serves this parameter's text verbatim as a JSON string: no
/// declared type, or only `string`. Such a value can stream before it is complete.
pub(crate) fn verbatim_string(schema: Option<&Value>) -> bool {
    schema
        .map(schema_types)
        .unwrap_or_default()
        .iter()
        .all(|t| t == "string")
}

/// [`typed_arguments`] serialised as the `function.arguments` JSON text, with exact
/// big-integer digits (no `f64` round trip). Under `SchemaConstrained` the recipient's
/// [`JsonKeys`] are derived here from `tools`; the streaming emitter precomputes them
/// instead and uses [`arguments_json_with`].
pub fn arguments_json(
    body: &str,
    name: &str,
    tools: &[Tool],
    framing: ArgumentFraming,
) -> Result<String, BeamBodyError> {
    let json_keys = if framing == ArgumentFraming::SchemaConstrained {
        held_json_keys(tools).remove(name).unwrap_or_default()
    } else {
        JsonKeys::default()
    };
    arguments_json_with(body, name, tools, framing, &json_keys, false)
}

/// [`arguments_json`] with the recipient's precomputed [`JsonKeys`] and the body
/// reading of [`parse_parameters_by`] (`lenient` for the `Drop` policy).
pub(crate) fn arguments_json_with(
    body: &str,
    name: &str,
    tools: &[Tool],
    framing: ArgumentFraming,
    json_keys: &JsonKeys,
    lenient: bool,
) -> Result<String, BeamBodyError> {
    Ok(PyValue::Dict(typed_arguments(
        body, name, tools, framing, json_keys, lenient,
    )?)
    .dumps())
}

#[cfg(test)]
mod tests {
    use super::recipient_len;
    use super::{
        ArgumentFraming, BeamBodyError, PARAM_CLOSE, PARAM_OPEN, PyDict, PyValue, Tool,
        ValueCursor, is_valid_recipient, param_value_span, param_value_span_with, unframe,
    };
    use serde_json::json;

    fn parse_parameters(body: &str) -> Result<Vec<(String, String)>, BeamBodyError> {
        super::parse_parameters(body, ArgumentFraming::Lenient)
    }

    fn typed_arguments(body: &str, name: &str, tools: &[Tool]) -> Result<PyDict, BeamBodyError> {
        super::typed_arguments(
            body,
            name,
            tools,
            ArgumentFraming::Lenient,
            &super::JsonKeys::default(),
            false,
        )
    }

    fn lenient_parameters(body: &str) -> Vec<(String, String)> {
        super::parse_parameters_by(body, |_| ArgumentFraming::Lenient, true).unwrap()
    }

    /// `$ref` targets are URI percent-decoded before JSON Pointer unescaping, and a
    /// nullable alias of a string `const`/`enum` keeps the bare text `null` as a string.
    #[test]
    fn reference_targets_decode_and_literal_null_stays_a_string() {
        let tools = [tool(
            "capture",
            json!({
                "properties": {
                    "space": {"$ref": "#/$defs/postal%20code"},
                    "utf8_plus": {"$ref": "#/$defs/caf%c3%a9+"},
                    "pointer": {"$ref": "#/$defs/a%7E1b%7E0c"},
                    "bad_escape": {"$ref": "#/$defs/x%zz"},
                    "const_text": {"$ref": "#/$defs/ConstAlias", "nullable": true},
                    "enum_text": {"$ref": "#/$defs/EnumAlias", "nullable": true},
                    "plain_null": {"type": "string", "nullable": true},
                },
                "$defs": {
                    "postal code": {"type": "integer"}, "café+": {"type": "integer"},
                    "a/b~c": {"type": "integer"}, "x%zz": {"type": "integer"},
                    "ConstAlias": {"$ref": "#/$defs/ConstText"}, "EnumAlias": {"$ref": "#/$defs/EnumText"},
                    "ConstText": {"type": "string", "const": "null"},
                    "EnumText": {"type": "string", "enum": ["null"]},
                },
            }),
        )];
        let body: String = ["space", "utf8_plus", "pointer", "bad_escape"]
            .iter()
            .map(|k| param(k, "42"))
            .chain(
                ["const_text", "enum_text", "plain_null"]
                    .iter()
                    .map(|k| param(k, "null")),
            )
            .collect();
        assert_eq!(
            arguments_json(&body, "capture", &tools).unwrap(),
            r#"{"space": 42, "utf8_plus": 42, "pointer": 42, "bad_escape": "42", "const_text": "null", "enum_text": "null", "plain_null": null}"#
        );
    }

    /// The lenient reading is the template's tag search: junk between parameters and a
    /// malformed opener are skipped, duplicates keep the last value, and the strict
    /// reading still rejects each of those bodies. An unclosed parameter is an error in
    /// both readings.
    #[test]
    fn lenient_reading_is_the_template_tag_search() {
        let body = "<rfl:parameter name=\"a\">\n1\n</rfl:parameter>garbage<rfl:parameter name=\"q\"\">\nx\n</rfl:parameter>\u{2003}<rfl:parameter name=\"a\">\n2\n</rfl:parameter>";
        assert_eq!(
            lenient_parameters(body),
            vec![("a".to_string(), "2".to_string())]
        );
        assert!(parse_parameters(body).is_err());
        let unclosed = format!("{body}<rfl:parameter name=\"b\">\nopen");
        assert!(super::parse_parameters_by(&unclosed, |_| ArgumentFraming::Lenient, true).is_err());
        assert_eq!(
            lenient_parameters("nothing here"),
            Vec::<(String, String)>::new()
        );
        assert_eq!(
            lenient_parameters("<rfl:parameter name=\"\">\n</rfl:parameter>"),
            vec![]
        );
    }

    fn arguments_json(body: &str, name: &str, tools: &[Tool]) -> Result<String, BeamBodyError> {
        super::arguments_json(body, name, tools, ArgumentFraming::Lenient)
    }

    #[test]
    fn last_duplicate_tool_definition_wins() {
        let tools = [
            tool("t", json!({"properties": {"n": {"type": "string"}}})),
            tool("t", json!({"properties": {"n": {"type": "integer"}}})),
        ];
        let body = param("n", "7");
        assert_eq!(arguments_json(&body, "t", &tools).unwrap(), r#"{"n": 7}"#);
    }

    /// The held map covers every tool the serving grammar can type, `strict` or not (the
    /// last held declaration of a name wins), and a key is JSON only when its normalized
    /// types are known and exclude `string`.
    #[test]
    fn held_json_keys_follow_schema_types_for_every_held_tool() {
        use super::{ArgumentFraming as F, JsonKeys, held_json_keys, value_framing};
        let declare = |name: &str, parameters: serde_json::Value, strict: Option<bool>| Tool {
            name: name.into(),
            description: None,
            parameters,
            strict,
        };
        let tools = [
            declare(
                "echo",
                json!({"type": "object", "properties": {
                    "s": {"type": "string"},
                    "o": {"type": "object"},
                    "n": {"type": ["number", "null"]},
                    "u": {"anyOf": [{"type": "string"}, {"type": "object"}]},
                    "r": {"$ref": "#/$defs/Obj"},
                    "rs": {"$ref": "#/$defs/Str"},
                    "w": {"description": "untyped"},
                    "ns": {"type": "string", "nullable": true},
                    "e": {"enum": [1, 2]},
                }, "$defs": {"Obj": {"type": "object"}, "Str": {"type": "string"}}}),
                Some(true),
            ),
            declare(
                "loose",
                json!({"properties": {"o": {"type": "object"}}}),
                Some(false),
            ),
            declare(
                "none",
                json!({"properties": {"o": {"type": "object"}}}),
                None,
            ),
            declare(
                "dup",
                json!({"properties": {"a": {"type": "object"}}}),
                Some(true),
            ),
            declare(
                "dup",
                json!({"properties": {"b": {"type": "object"}}}),
                Some(true),
            ),
            declare(
                "dup",
                json!({"properties": {"c": {"type": "object"}}}),
                Some(false),
            ),
        ];
        let map = held_json_keys(&tools);
        assert_eq!(map.len(), 4);
        for name in ["loose", "none"] {
            assert!(
                map[name].contains("o"),
                "{name} is held regardless of strict"
            );
        }
        let echo = &map["echo"];
        for key in ["o", "n", "r", "e"] {
            assert!(echo.contains(key), "{key}");
            assert_eq!(
                value_framing(F::SchemaConstrained, echo, key),
                F::ConstrainedJson
            );
        }
        for raw in ["s", "u", "rs", "w", "ns", "missing"] {
            assert!(!echo.contains(raw), "{raw}");
            assert_eq!(value_framing(F::SchemaConstrained, echo, raw), F::Lenient);
        }
        // The last held `dup` declaration wins, strict or not.
        assert!(map["dup"].contains("c"));
        assert!(!map["dup"].contains("a"));
        assert!(!map["dup"].contains("b"));
        // Global modes ignore the key set; an empty set is raw text everywhere.
        for mode in [F::Lenient, F::ConstrainedJson] {
            assert_eq!(value_framing(mode, echo, "o"), mode);
            assert_eq!(value_framing(mode, &JsonKeys::default(), "o"), mode);
        }
        assert_eq!(
            value_framing(F::SchemaConstrained, &JsonKeys::default(), "o"),
            F::Lenient
        );
        // `parse_parameters` under the global `SchemaConstrained` has no key set: raw.
        let body = param("o", r#"{"s": "a</rfl:parameter>b"}"#);
        assert!(super::parse_parameters(&body, F::SchemaConstrained).is_err());
        // `arguments_json` derives the recipient's set itself and decodes the object.
        assert_eq!(
            super::arguments_json(&body, "echo", &tools, F::SchemaConstrained).unwrap(),
            r#"{"o": {"s": "a</rfl:parameter>b"}}"#
        );
        // Raw string slots keep literal quotes, unmatched quotes and bare braces.
        for value in [
            "\"unfinished",
            "{\"x\":\"unfinished",
            "{",
            "\"Paris\"",
            "not json",
        ] {
            let body = param("s", value);
            assert_eq!(
                super::arguments_json(&body, "echo", &tools, F::SchemaConstrained).unwrap(),
                format!("{{\"s\": {}}}", serde_json::Value::String(value.into())),
                "{value}"
            );
        }
    }

    /// Framing follows the grammar's UNION type list, typing the conjunctive one. Where
    /// they differ the slot is raw text on the wire, so an unfinished `{` or quote is
    /// accepted verbatim, while the value is still typed by `schema_types`.
    #[test]
    fn json_keys_follow_the_grammar_union_rule_not_the_typing_intersection() {
        use super::{ArgumentFraming as F, JsonKeys, framing_types, schema_types};
        let sibling_union =
            json!({"type": "object", "anyOf": [{"type": "string"}, {"type": "object"}]});
        let sibling_oneof =
            json!({"type": "object", "oneOf": [{"type": "object"}, {"type": "string"}]});
        let conflict = json!({"type": "integer", "anyOf": [{"type": "string"}]});
        let untyped_branch =
            json!({"type": "object", "anyOf": [{"description": "x"}, {"type": "object"}]});
        let only_untyped = json!({"anyOf": [{"description": "x"}]});
        let nested = json!({"anyOf": [{"type": "object"}, {"oneOf": [{"type": "null"}, {"type": "string"}]}]});
        let nullable_object = json!({"type": "object", "nullable": true});
        let falsy_nullable = json!({"type": "object", "nullable": 0});
        let nullable_string = json!({"type": "string", "nullable": true});
        // The union lists, in reference order.
        let list = |s: &serde_json::Value| framing_types(s);
        assert_eq!(list(&sibling_union), ["object", "string"]);
        assert_eq!(list(&conflict), ["integer", "string"]);
        assert_eq!(list(&untyped_branch), ["object"]);
        assert!(list(&only_untyped).is_empty());
        assert_eq!(list(&nested), ["object", "null", "string"]);
        assert_eq!(list(&nullable_object), ["object", "null"]);
        assert_eq!(list(&falsy_nullable), ["object"]);
        assert_eq!(list(&nullable_string), ["string", "null"]);
        // Typing intersects; framing does not.
        assert_eq!(schema_types(&sibling_union), ["object"]);
        assert_eq!(schema_types(&conflict), ["integer"]);

        let parameters = json!({
            "type": "object",
            "properties": {
                "su": sibling_union, "so": sibling_oneof, "c": conflict,
                "ub": untyped_branch, "ou": only_untyped, "ne": nested,
                "no": nullable_object, "fo": falsy_nullable, "ns": nullable_string,
                "r": {"$ref": "#/$defs/Union"},
                "ro": {"$ref": "#/$defs/Obj", "nullable": true},
                "a": {"allOf": [{"type": "integer", "oneOf": [{"type": "string"}]}]},
            },
            "$defs": {
                "Union": {"type": "object", "anyOf": [{"type": "string"}, {"type": "object"}]},
                "Obj": {"type": "object"},
            },
        });
        let keys = JsonKeys::of_held_tool(&parameters);
        assert_eq!(keys.keys(), ["fo", "no", "ro", "ub"]);
        // End to end: the raw slots accept unfinished JSON text and keep it, typed by
        // the conjunctive rule (`su` types as object, so `{` stays text). `ou` has no
        // type the parser can read, so the grammar would leave a tool declaring it
        // unheld (all raw); drop it to keep this tool held.
        let mut held = parameters.clone();
        held["properties"].as_object_mut().unwrap().remove("ou");
        let tools = [Tool {
            name: "t".into(),
            description: None,
            parameters: held,
            strict: Some(true),
        }];
        for (key, raw) in [
            ("su", "{"),
            ("so", "\"unfinished"),
            ("c", "{\"x\":\"u"),
            ("r", "{"),
        ] {
            let body = param(key, raw);
            assert_eq!(
                super::arguments_json(&body, "t", &tools, F::SchemaConstrained).unwrap(),
                format!("{{\"{key}\": {}}}", serde_json::Value::String(raw.into())),
                "{key}"
            );
        }
        // JSON slots still own an embedded closer.
        for key in ["no", "ro", "ub"] {
            let body = param(key, r#"{"s": "a</rfl:parameter>b"}"#);
            assert_eq!(
                super::arguments_json(&body, "t", &tools, F::SchemaConstrained).unwrap(),
                format!("{{\"{key}\": {{\"s\": \"a</rfl:parameter>b\"}}}}"),
                "{key}"
            );
        }
    }

    #[test]
    fn constrained_json_owns_strings_but_lenient_stops_at_first_closer() {
        let value = r#"{"s": "a</rfl:parameter>b"}"#;
        let body = param("o", value);
        let strict = super::parse_parameters(&body, ArgumentFraming::ConstrainedJson).unwrap();
        assert_eq!(strict, vec![("o".to_string(), value.to_string())]);
        assert!(super::parse_parameters(&body, ArgumentFraming::Lenient).is_err());
    }

    fn tool(name: &str, parameters: serde_json::Value) -> Tool {
        Tool {
            name: name.into(),
            description: None,
            parameters,
            strict: None,
        }
    }

    fn param(key: &str, value: &str) -> String {
        format!("<rfl:parameter name=\"{key}\">{value}</rfl:parameter>")
    }

    #[test]
    fn serving_typing_resolves_aliases_and_survives_cycles() {
        let tools = [tool(
            "t",
            json!({"type": "object",
                "properties": {
                    "x": {"$ref": "#/$defs/N"},
                    "y": {"allOf": [{"type": "integer"}]},
                    "z": {"$ref": "#/$defs/C"},
                    "e": {"enum": [1, 2]},
                    "n": {"minimum": 0},
                    "u": {"type": ["string", "null"]},
                    "w": {"description": "d"},
                },
                "$defs": {"N": {"type": "integer"}, "C": {"$ref": "#/$defs/C"}}}),
        )];
        let body = [
            param("x", "7"),
            param("y", "8"),
            param("z", "5"),
            param("e", "1"),
            param("n", "2.5"),
            param("u", "null"),
            param("w", "9"),
        ]
        .concat();
        // `u` is `["string", "null"]`: the text `null` IS the admitted JSON null (the
        // frozen reference and this implementation agree).
        assert_eq!(
            arguments_json(&body, "t", &tools).unwrap(),
            r#"{"x": 7, "y": 8, "z": "5", "e": 1, "n": 2.5, "u": null, "w": "9"}"#
        );
    }

    #[test]
    fn sibling_type_constraints_intersect_with_union_branches() {
        let tools = [tool(
            "t",
            json!({"type": "object", "properties": {
                // type string AND a nullable anyOf: both apply, null is excluded.
                "a": {"type": "string", "anyOf": [{"type": "string"}, {"type": "null"}]},
                // nullable type array AND a non-nullable oneOf: the intersection is string.
                "b": {"type": ["string", "null"], "oneOf": [{"type": "string"}, {"type": "integer"}]},
                // an untyped branch does not remove the string requirement.
                "c": {"type": "string", "anyOf": [{"minLength": 1}, {"type": "null"}]},
                // no sibling type: the union alone decides, so null is admitted.
                "d": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "e": {"anyOf": [{"const": "null"}, {"type": "integer"}]},
                "f": {"type": "string", "enum": ["null", null]},
                "g": {"type": "string", "nullable": true},
                "h": {"type": "integer", "anyOf": [{"type": "string"}]},
            }}),
        )];
        let body = ["a", "b", "c", "d", "e", "f", "g"]
            .iter()
            .map(|k| param(k, "null"))
            .collect::<String>()
            + &param("h", "12345678901234567890123")
            + &param("i", " 7 ");
        assert_eq!(
            arguments_json(&body, "t", &tools).unwrap(),
            r#"{"a": "null", "b": "null", "c": "null", "d": null, "e": "null", "f": "null", "g": null, "h": 12345678901234567890123, "i": " 7 "}"#
        );
    }

    #[test]
    fn constrained_json_waits_on_open_values_and_keeps_raw_text() {
        use super::param_value_end;
        let constrained = ArgumentFraming::ConstrainedJson;
        let open = [
            r#"{"text":"</rfl:parameter><|call|>pay"#,
            r#"{"text":"a\"</rfl:parameter>"#,
            r#"{"text":"a\\"#,
            r#"["</rfl:parameter>", "ü"#,
            r#""</rfl:parameter>"#,
            r#""abc\"</rfl:parameter>"#,
        ];
        for value in open {
            assert_eq!(param_value_end(value, constrained), None, "{value}");
            assert_eq!(
                param_value_end(value, ArgumentFraming::Lenient),
                value.find("</rfl:parameter>"),
                "{value}"
            );
        }
        for (value, end) in [
            (r#"{"t":"</rfl:parameter>"}</rfl:parameter>x"#, 24),
            (r#""</rfl:parameter>"</rfl:parameter>"#, 18),
            (r#"[1, "ünï"]</rfl:parameter>"#, 12),
            ("London</rfl:parameter>", 6),
            ("\n{\"a\": 1}\n</rfl:parameter>", 10),
        ] {
            assert_eq!(param_value_end(value, constrained), Some(end), "{value}");
        }
        // A value that opens as JSON but can never balance keeps the reference rule.
        assert_eq!(param_value_end("{]</rfl:parameter>", constrained), Some(2));
    }

    #[test]
    fn recipient_grammar() {
        // The builder's name contract: digit-, hyphen- and dot-leading names included.
        for ok in [
            "weather",
            "_x",
            "a.b-c9",
            "functions.get_weather",
            "2fa_check",
            "-tool",
            ".a",
        ] {
            assert!(is_valid_recipient(ok), "{ok}");
        }
        for bad in ["", "a b", "a\tb", "a<b", "a<|channel|>", "é", "[{\"name\""] {
            assert!(!is_valid_recipient(bad), "{bad}");
        }
        assert_eq!(recipient_len("get_weather<|channel|>"), 11);
        assert_eq!(recipient_len("2fa_check<|channel|>"), 9);
        assert_eq!(recipient_len(" x"), 0);
    }

    #[test]
    fn a_value_may_end_where_the_next_parameter_opens() {
        use super::param_value_span;
        let lenient = ArgumentFraming::Lenient;
        // Implicit boundary: the value keeps its framing newline, the body resumes at
        // the opener.
        let body = "\nA\n<rfl:parameter name=\"b\">\nB\n</rfl:parameter>";
        assert_eq!(param_value_span(body, lenient), Some((3, 3)));
        assert_eq!(unframe(&body[..3]), "A");
        assert!(body[3..].starts_with(PARAM_OPEN));
        // An explicit closer that comes first still wins and is consumed.
        let closed = "\nA\n</rfl:parameter>\n<rfl:parameter name=\"b\">";
        assert_eq!(
            param_value_span(closed, lenient),
            Some((3, 3 + PARAM_CLOSE.len()))
        );
        // Both decoders read the same two parameters either way.
        let tools = vec![Tool {
            name: "t".into(),
            description: None,
            parameters: json!({"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "string"}}}),
            strict: None,
        }];
        for body in [
            "<rfl:parameter name=\"a\">\nA\n<rfl:parameter name=\"b\">\nB\n</rfl:parameter>",
            "<rfl:parameter name=\"a\">\nA\n</rfl:parameter><rfl:parameter name=\"b\">\nB\n</rfl:parameter>",
        ] {
            assert_eq!(
                arguments_json(body, "t", &tools).unwrap(),
                r#"{"a": "A", "b": "B"}"#,
                "{body}"
            );
        }
    }

    #[test]
    fn single_lf_stripped_at_each_edge_only() {
        let body = param("a", "\n\n x \r\n\n");
        assert_eq!(
            parse_parameters(&body).unwrap(),
            vec![("a".into(), "\n x \r\n".into())]
        );
        let body = param("a", "\nLondon\n");
        assert_eq!(parse_parameters(&body).unwrap()[0].1, "London");
    }

    #[test]
    fn malformed_and_truncated_bodies_are_errors_not_empty() {
        assert!(matches!(
            parse_parameters("<rfl:parameter name=\"location\">\nLondon\n"),
            Err(BeamBodyError::UnclosedParameter { .. })
        ));
        assert!(matches!(
            parse_parameters("<rfl:parameter name=\"location\">\nLondon\n</wrong>"),
            Err(BeamBodyError::UnclosedParameter { .. })
        ));
        assert!(matches!(
            parse_parameters("junk"),
            Err(BeamBodyError::UnexpectedContent { .. })
        ));
        assert!(matches!(
            parse_parameters(&format!("{}x", param("a", "1"))),
            Err(BeamBodyError::UnexpectedContent { .. })
        ));
        assert_eq!(parse_parameters("").unwrap(), vec![]);
    }

    #[test]
    fn duplicates_last_wins_entities_literal_and_framing_is_data() {
        let body = format!(
            "{}\n{}",
            param("a", "1"),
            param("a", "&amp; <|call|><|end|><|start|>")
        );
        let args = typed_arguments(&body, "t", &[]).unwrap();
        assert_eq!(
            PyValue::Dict(args).dumps(),
            r#"{"a": "&amp; <|call|><|end|><|start|>"}"#
        );
    }

    #[test]
    fn integral_decimals_are_exact_for_integer_parameters() {
        let integer = json!({"type": "integer"});
        for (raw, expected) in [
            ("3.0", "3"),
            (" -12.00\n", "-12"),
            ("7.", "7"),
            ("-0.00", "0"),
            ("123456789012345678901.0", "123456789012345678901"),
            ("3.5", r#""3.5""#),
            ("7e0", r#""7e0""#),
            (".0", r#"".0""#),
            ("1_0.0", r#""1_0.0""#),
            ("3.0000000000000001", r#""3.0000000000000001""#),
        ] {
            assert_eq!(
                super::typed_value(Some(&integer), raw).dumps(),
                expected,
                "{raw}"
            );
        }
        for (schema, expected) in [
            (json!({"type": "string"}), r#""3.0""#),
            (json!({"type": ["string", "integer"]}), r#""3.0""#),
            (json!({"type": "number"}), "3.0"),
        ] {
            assert_eq!(super::typed_value(Some(&schema), "3.0").dumps(), expected);
        }
    }

    #[test]
    fn coercion_follows_reference() {
        let tools = [tool(
            "fn",
            json!({"type": "object", "properties": {
                "s": {"type": "string"},
                "u": {"type": ["string", "integer"]},
                "b": {"type": "boolean"},
                "n": {"type": "number"},
                "o": {"type": "object"},
            }}),
        )];
        let body = [
            param("s", "null"),
            param("u", "007"),
            param("b", "1"),
            param("n", "123456789012345678901"),
            param("o", "[1]"),
            param("extra", "true"),
        ]
        .concat();
        assert_eq!(
            arguments_json(&body, "fn", &tools).unwrap(),
            r#"{"s": "null", "u": "007", "b": true, "n": 123456789012345678901, "o": "[1]", "extra": "true"}"#
        );
        let body = param("u", "3 ") + &param("v", "3");
        assert_eq!(
            arguments_json(&body, "fn", &tools).unwrap(),
            r#"{"u": "3 ", "v": "3"}"#
        );
    }

    /// A value delivered in many pushes is walked once: the cursor resumes at the byte
    /// the previous push ended on, and reads the same span as a fresh walk of the whole.
    #[test]
    fn a_value_cursor_resumes_where_the_previous_push_ended() {
        let json = format!("{{\"s\": \"{}\"}}", "x".repeat(50_000));
        let raw = "y".repeat(50_000);
        for (value, framing) in [
            (json.clone(), ArgumentFraming::ConstrainedJson),
            (raw.clone(), ArgumentFraming::Lenient),
            (json.clone(), ArgumentFraming::Lenient),
        ] {
            let after_key = format!("\n{value}\n</rfl:parameter><|call|>");
            let mut cursor = ValueCursor::default();
            let mut json_walked = 0;
            let mut searched = 0;
            for end in (1..after_key.len()).step_by(7).chain([after_key.len()]) {
                let got = param_value_span_with(&after_key[..end], framing, &mut cursor);
                assert_eq!(
                    got,
                    param_value_span(&after_key[..end], framing),
                    "{framing:?} at {end}"
                );
                assert!(cursor.json_scanned >= json_walked && cursor.closer_searched >= searched);
                json_walked = cursor.json_scanned;
                searched = cursor.closer_searched;
            }
            if framing == ArgumentFraming::ConstrainedJson {
                // The JSON value was walked exactly once; only the short remainder
                // after it was searched for the closer.
                assert_eq!(json_walked, json.len(), "{framing:?}");
            } else {
                // Raw text: the closer search advanced over the value once.
                assert!(
                    searched + PARAM_CLOSE.len() >= value.len(),
                    "the closer search never restarted"
                );
            }
        }
    }
}
