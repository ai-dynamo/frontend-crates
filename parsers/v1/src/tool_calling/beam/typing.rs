// SPDX-FileCopyrightText: Copyright (c) 2026 Reflection AI. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Serving-parity schema preparation for Beam argument typing (`typed_parameters`).
//!
//! Resolves annotated aliases (a local `$ref` or a single-branch `allOf`) and infers the
//! JSON types argument decoding needs, so `{"$ref": "#/$defs/N"}` with `N: integer`
//! types its value like the reference does. Derived from the model builder's frozen
//! reference; an independent copy of the one in `dynamo-parsers-v2`. Beam only: other
//! families keep their own typing.

use serde_json::{Map, Value, json};
use std::collections::{BTreeSet, HashSet};

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
/// unescaping; an invalid `%` escape or a pointer that leaves the document resolves
/// nothing (same rule as `dynamo-parsers-v2`).
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

/// Whether a schema's `const` / `enum` admit `raw` itself as a string value (a nullable
/// alias of `{"type": "string", "const": "null"}` keeps the text `null` as a string).
pub(super) fn string_literal_admits(schema: &Value, raw: &str) -> bool {
    let Value::Object(map) = schema else {
        return false;
    };
    let text = super::py::strip(raw);
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
                json!(types[0])
            } else {
                json!(types)
            };
            schema.insert("type".to_string(), value);
        }
    }
    Value::Object(schema)
}

/// Resolve annotated aliases and infer the types argument decoding needs, for each
/// property of `parameters`.
pub(super) fn typed_parameters(parameters: &Value) -> Value {
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
