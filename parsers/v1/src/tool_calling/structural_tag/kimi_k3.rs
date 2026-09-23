// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Kimi K3 XTML structural-tag generation.
//!
//! K3 does not emit the generic JSON shape used by legacy forced-tool guided
//! decoding. Tool calls live in a native XTML `tools` channel with one or more
//! nested `call` and typed `argument` elements. This builder mirrors that wire
//! format so named and required tool choices can be constrained without
//! changing what the K3 parser expects.

use std::collections::HashSet;

use serde_json::{Map, Value};

use super::builder::{ToolCallFormatBuildContext, resolve_tools_to_include};
use super::format::{
    AnyTextFormat, ConstStringFormat, Format, JsonSchemaFormat, JsonSchemaStyle, OptionalFormat,
    OrFormat, RegexFormat, SequenceFormat, StarFormat, StructuralTag, TagFormat,
    TagsWithSeparatorFormat,
};
use crate::tool_calling::ToolDefinition;

const OPEN: &str = "<|open|>";
const CLOSE: &str = "<|close|>";
const SEP: &str = "<|sep|>";
const RESPONSE_OPEN: &str = "<|open|>response<|sep|>";
const RESPONSE_CLOSE: &str = "<|close|>response<|sep|>";
const THINK_OPEN: &str = "<|open|>think<|sep|>";
const THINK_CLOSE: &str = "<|close|>think<|sep|>";
const TOOLS_OPEN: &str = "<|open|>tools<|sep|>";
const TOOLS_CLOSE: &str = "<|close|>tools<|sep|>";
const CALL_OPEN: &str = "<|open|>call";
const CALL_CLOSE: &str = "<|close|>call<|sep|>";
const ARGUMENT_CLOSE: &str = "<|close|>argument<|sep|>";
const MESSAGE_CLOSE: &str = "<|close|>message<|sep|>";

const STRING_ATOM: &str = r"(?:[^<]|<[^|])";
const CALL_INDEX_PATTERN: &str = "[1-9][0-9]*";

fn escape_attr(value: &str) -> String {
    value.replace('&', "&amp;").replace('"', "&quot;")
}

fn optional(content: Format) -> Format {
    Format::Optional(OptionalFormat {
        content: Box::new(content),
    })
}

fn star(content: Format) -> Format {
    Format::Star(StarFormat {
        content: Box::new(content),
    })
}

fn one_of(elements: Vec<Format>) -> Format {
    if elements.len() == 1 {
        elements.into_iter().next().expect("one element")
    } else {
        Format::Or(OrFormat { elements })
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum JsonType {
    String,
    Number,
    Integer,
    Boolean,
    Array,
    Object,
    Null,
}

const JSON_TYPES: [JsonType; 7] = [
    JsonType::String,
    JsonType::Number,
    JsonType::Integer,
    JsonType::Boolean,
    JsonType::Array,
    JsonType::Object,
    JsonType::Null,
];

fn is_integral_number(number: &serde_json::Number) -> bool {
    number.is_i64()
        || number.is_u64()
        || number
            .as_f64()
            .is_some_and(|value| value.is_finite() && value.fract() == 0.0)
}

impl JsonType {
    fn from_name(value: &str) -> Option<Self> {
        match value {
            "string" => Some(Self::String),
            "number" => Some(Self::Number),
            "integer" => Some(Self::Integer),
            "boolean" => Some(Self::Boolean),
            "array" => Some(Self::Array),
            "object" => Some(Self::Object),
            "null" => Some(Self::Null),
            _ => None,
        }
    }

    fn for_value(value: &Value) -> Self {
        match value {
            Value::Null => Self::Null,
            Value::Bool(_) => Self::Boolean,
            Value::Number(number) if is_integral_number(number) => Self::Integer,
            Value::Number(_) => Self::Number,
            Value::String(_) => Self::String,
            Value::Array(_) => Self::Array,
            Value::Object(_) => Self::Object,
        }
    }

    fn xtml_name(self) -> &'static str {
        match self {
            Self::String => "string",
            Self::Number | Self::Integer => "number",
            Self::Boolean => "boolean",
            Self::Array => "array",
            Self::Object => "object",
            Self::Null => "null",
        }
    }

    fn json_name(self) -> &'static str {
        match self {
            Self::String => "string",
            Self::Number => "number",
            Self::Integer => "integer",
            Self::Boolean => "boolean",
            Self::Array => "array",
            Self::Object => "object",
            Self::Null => "null",
        }
    }

    fn accepts(self, value: &Value) -> bool {
        match self {
            Self::String => value.is_string(),
            Self::Number => value.is_number(),
            Self::Integer => value.as_number().is_some_and(is_integral_number),
            Self::Boolean => value.is_boolean(),
            Self::Array => value.is_array(),
            Self::Object => value.is_object(),
            Self::Null => value.is_null(),
        }
    }
}

fn resolve_local_ref<'a>(reference: &str, root: &'a Value) -> Option<&'a Value> {
    if reference == "#" {
        return matches!(root, Value::Bool(_) | Value::Object(_)).then_some(root);
    }
    let mut value = root;
    for raw_part in reference.strip_prefix("#/")?.split('/') {
        let part = raw_part.replace("~1", "/").replace("~0", "~");
        value = value.get(&part)?;
    }
    matches!(value, Value::Bool(_) | Value::Object(_)).then_some(value)
}

fn schema_types(schema: &Value, root: &Value, seen_refs: &HashSet<String>) -> Vec<JsonType> {
    if let Some(allowed) = schema.as_bool() {
        return if allowed {
            JSON_TYPES.to_vec()
        } else {
            Vec::new()
        };
    }
    let Some(schema) = schema.as_object() else {
        return JSON_TYPES.to_vec();
    };

    if let Some(reference) = schema.get("$ref").and_then(Value::as_str)
        && !seen_refs.contains(reference)
        && let Some(target) = resolve_local_ref(reference, root)
    {
        let mut nested_seen = seen_refs.clone();
        nested_seen.insert(reference.to_string());
        return schema_types(target, root, &nested_seen);
    }

    if let Some(schema_type) = schema.get("type") {
        if let Some(schema_type) = schema_type.as_str() {
            return JsonType::from_name(schema_type)
                .map(|value| vec![value])
                .unwrap_or_else(|| JSON_TYPES.to_vec());
        }
        if let Some(schema_types) = schema_type.as_array() {
            return JSON_TYPES
                .into_iter()
                .filter(|candidate| {
                    schema_types.iter().any(|value| {
                        value
                            .as_str()
                            .and_then(JsonType::from_name)
                            .is_some_and(|value| value == *candidate)
                    })
                })
                .collect();
        }
    }

    for keyword in ["anyOf", "oneOf"] {
        if let Some(options) = schema.get(keyword).and_then(Value::as_array) {
            let option_types: HashSet<_> = options
                .iter()
                .filter(|option| matches!(option, Value::Bool(_) | Value::Object(_)))
                .flat_map(|option| schema_types(option, root, seen_refs))
                .collect();
            return JSON_TYPES
                .into_iter()
                .filter(|candidate| option_types.contains(candidate))
                .collect();
        }
    }

    if let Some(options) = schema.get("allOf").and_then(Value::as_array) {
        let all_types: HashSet<_> = JSON_TYPES.into_iter().collect();
        let mut constrained = options
            .iter()
            .filter(|option| matches!(option, Value::Bool(_) | Value::Object(_)))
            .map(|option| {
                schema_types(option, root, seen_refs)
                    .into_iter()
                    .collect::<HashSet<_>>()
            })
            .filter(|types| types != &all_types);
        if let Some(mut result) = constrained.next() {
            if result.contains(&JsonType::Number) {
                result.insert(JsonType::Integer);
            }
            for mut types in constrained {
                if types.contains(&JsonType::Number) {
                    types.insert(JsonType::Integer);
                }
                result.retain(|value| types.contains(value));
            }
            if result.contains(&JsonType::Number) {
                result.remove(&JsonType::Integer);
            }
            return JSON_TYPES
                .into_iter()
                .filter(|candidate| result.contains(candidate))
                .collect();
        }
    }

    if let Some(value) = schema.get("const") {
        return vec![JsonType::for_value(value)];
    }
    if let Some(values) = schema.get("enum").and_then(Value::as_array) {
        let enum_types: HashSet<_> = values.iter().map(JsonType::for_value).collect();
        return JSON_TYPES
            .into_iter()
            .filter(|candidate| enum_types.contains(candidate))
            .collect();
    }

    const OBJECT_KEYWORDS: [&str; 9] = [
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
    const ARRAY_KEYWORDS: [&str; 8] = [
        "contains",
        "items",
        "maxContains",
        "maxItems",
        "minContains",
        "minItems",
        "prefixItems",
        "uniqueItems",
    ];
    const STRING_KEYWORDS: [&str; 4] = ["format", "maxLength", "minLength", "pattern"];
    const NUMBER_KEYWORDS: [&str; 5] = [
        "exclusiveMaximum",
        "exclusiveMinimum",
        "maximum",
        "minimum",
        "multipleOf",
    ];
    if OBJECT_KEYWORDS.iter().any(|key| schema.contains_key(*key)) {
        vec![JsonType::Object]
    } else if ARRAY_KEYWORDS.iter().any(|key| schema.contains_key(*key)) {
        vec![JsonType::Array]
    } else if STRING_KEYWORDS.iter().any(|key| schema.contains_key(*key)) {
        vec![JsonType::String]
    } else if NUMBER_KEYWORDS.iter().any(|key| schema.contains_key(*key)) {
        vec![JsonType::Number]
    } else {
        JSON_TYPES.to_vec()
    }
}

fn xtml_types(schema: &Value, root: &Value) -> Vec<&'static str> {
    let mut seen = HashSet::new();
    schema_types(schema, root, &HashSet::new())
        .into_iter()
        .map(JsonType::xtml_name)
        .filter(|xtml_type| seen.insert(*xtml_type))
        .collect()
}

fn string_length_regex(schema: &Map<String, Value>) -> Option<String> {
    let min_len = match schema.get("minLength") {
        Some(value) => value.as_u64()?,
        None => 0,
    };
    let max_len = match schema.get("maxLength") {
        Some(value) => Some(value.as_u64()?),
        None => None,
    };
    if min_len > 4096 || max_len.is_some_and(|max_len| max_len > 4096 || min_len > max_len) {
        return None;
    }
    match max_len {
        Some(max_len) => Some(format!("{STRING_ATOM}{{{min_len},{max_len}}}")),
        None if schema.contains_key("minLength") => Some(format!("{STRING_ATOM}{{{min_len},}}")),
        None => None,
    }
}

fn has_unescaped_trailing_dollar(pattern: &str) -> bool {
    let Some(prefix) = pattern.strip_suffix('$') else {
        return false;
    };
    prefix
        .as_bytes()
        .iter()
        .rev()
        .take_while(|byte| **byte == b'\\')
        .count()
        % 2
        == 0
}

fn merge_compatible_all_of(schema: &Map<String, Value>) -> Option<Map<String, Value>> {
    let options = schema.get("allOf")?.as_array()?;
    let mut merged = schema.clone();
    merged.remove("allOf");

    for option in options {
        let option = match option {
            Value::Bool(true) => continue,
            Value::Object(option) => {
                merge_compatible_all_of(option).unwrap_or_else(|| option.clone())
            }
            _ => return None,
        };
        for (key, value) in option {
            if merged.get(&key).is_some_and(|existing| existing != &value) {
                return None;
            }
            merged.insert(key, value);
        }
    }
    Some(merged)
}

fn string_content_format(schema: &Value) -> Format {
    let root = schema;
    let Some(schema) = schema.as_object() else {
        return Format::AnyText(AnyTextFormat {
            excludes: vec![CLOSE.to_string()],
        });
    };
    // XTML string arguments contain raw text rather than a JSON string, so they
    // cannot use JsonSchemaFormat. Flatten compatible allOf branches before
    // translating their string constraints to raw-text formats.
    let merged_schema = merge_compatible_all_of(schema);
    let schema = merged_schema.as_ref().unwrap_or(schema);

    let enum_values = schema
        .get("enum")
        .and_then(Value::as_array)
        .cloned()
        .or_else(|| {
            schema
                .get("const")
                .and_then(Value::as_str)
                .map(|value| vec![Value::String(value.to_string())])
        });
    if let Some(values) = enum_values {
        let pattern = schema.get("pattern").and_then(Value::as_str);
        let compiled_pattern = pattern.map(regex::Regex::new);
        if !values.is_empty()
            && values.len() <= 256
            && values.iter().all(|value| value.as_str().is_some())
            && compiled_pattern.as_ref().is_none_or(Result::is_ok)
        {
            let min_len = schema.get("minLength").and_then(Value::as_u64);
            let max_len = schema.get("maxLength").and_then(Value::as_u64);
            let const_value = schema.get("const").and_then(Value::as_str);
            let values = values
                .iter()
                .filter_map(Value::as_str)
                .filter(|value| !value.contains("<|"))
                .filter(|value| {
                    let len = value.chars().count() as u64;
                    min_len.is_none_or(|min| len >= min)
                        && max_len.is_none_or(|max| len <= max)
                        && const_value.is_none_or(|constant| *value == constant)
                        && compiled_pattern.as_ref().is_none_or(|pattern| {
                            pattern.as_ref().is_ok_and(|re| re.is_match(value))
                        })
                })
                .collect::<Vec<_>>();
            return one_of(
                values
                    .iter()
                    .map(|value| {
                        Format::ConstString(ConstStringFormat {
                            value: (*value).to_string(),
                        })
                    })
                    .collect(),
            );
        }
    }

    if let Some(pattern) = schema.get("pattern").and_then(Value::as_str) {
        // XGrammar 0.2.3 regexes do not support lookahead, so a general
        // intersection of `pattern` and length bounds cannot be expressed.
        // Match XGrammar's JSON-schema policy: `pattern` takes precedence when
        // both are present instead of silently discarding the pattern.
        let anchored_start = pattern.starts_with('^');
        let anchored_end = has_unescaped_trailing_dollar(pattern);
        let pattern = pattern.strip_prefix('^').unwrap_or(pattern);
        let pattern = if anchored_end {
            pattern
                .strip_suffix('$')
                .expect("anchored pattern ends with a dollar")
        } else {
            pattern
        };
        let prefix = if anchored_start {
            String::new()
        } else {
            format!("{STRING_ATOM}*")
        };
        let suffix = if anchored_end {
            String::new()
        } else {
            format!("{STRING_ATOM}*")
        };
        return Format::Regex(RegexFormat {
            pattern: format!("{prefix}(?:{pattern}){suffix}"),
        });
    }

    if let Some(pattern) = string_length_regex(schema) {
        return Format::Regex(RegexFormat { pattern });
    }

    for keyword in ["anyOf", "oneOf"] {
        let Some(options) = schema.get(keyword).and_then(Value::as_array) else {
            continue;
        };
        let formats = options
            .iter()
            .filter(|option| {
                schema_types(option, root, &HashSet::new()).contains(&JsonType::String)
            })
            .map(string_content_format)
            .collect::<Vec<_>>();
        if !formats.is_empty() {
            return one_of(formats);
        }
    }

    Format::AnyText(AnyTextFormat {
        excludes: vec![CLOSE.to_string()],
    })
}

fn resolved_root_parameters(root: &Value) -> Option<Value> {
    let resolved = resolve_local_refs(root, root, &HashSet::new())?;
    let object = resolved.as_object()?;
    if object.contains_key("allOf") {
        merge_compatible_all_of(object).map(Value::Object)
    } else {
        Some(resolved)
    }
}

fn permissive_argument_tag() -> TagFormat {
    TagFormat {
        begin: format!("{OPEN}argument "),
        content: Box::new(Format::Sequence(SequenceFormat {
            elements: vec![
                Format::Regex(RegexFormat {
                    pattern: format!(r"[^<]*{}", SEP.replace('|', r"\|")),
                }),
                Format::AnyText(AnyTextFormat {
                    excludes: vec![CLOSE.to_string()],
                }),
            ],
        })),
        end: ARGUMENT_CLOSE.to_string(),
    }
}

fn resolve_schema_map(value: &Value, root: &Value, seen_refs: &HashSet<String>) -> Option<Value> {
    let Some(values) = value.as_object() else {
        return Some(value.clone());
    };
    values
        .iter()
        .map(|(key, value)| {
            resolve_local_refs(value, root, seen_refs).map(|value| (key.clone(), value))
        })
        .collect::<Option<Map<_, _>>>()
        .map(Value::Object)
}

// JSON Schema documents mix subschemas with arbitrary JSON data. These JSON Schema
// 2020-12 keywords, plus draft-07 compatibility forms, identify the positions where
// local `$ref` values have schema semantics and are therefore safe to resolve.

// Each property of these keyword values is a subschema.
const SCHEMA_MAP_KEYWORDS: &[&str] = &[
    "$defs",
    "definitions",
    "dependentSchemas",
    "dependencies",
    "patternProperties",
    "properties",
];

// The keyword value itself is a subschema.
const SCHEMA_VALUE_KEYWORDS: &[&str] = &[
    "additionalItems",
    "additionalProperties",
    "contains",
    "contentSchema",
    "else",
    "if",
    "items",
    "not",
    "propertyNames",
    "then",
    "unevaluatedItems",
    "unevaluatedProperties",
];

// Each array element in these keyword values is a subschema.
const SCHEMA_ARRAY_KEYWORDS: &[&str] = &["allOf", "anyOf", "oneOf", "prefixItems"];

fn local_ref_with_prefix(reference: &str, prefix: &str) -> Option<String> {
    if reference == "#" {
        Some(prefix.to_string())
    } else {
        reference
            .strip_prefix("#/")
            .map(|suffix| format!("{prefix}/{suffix}"))
    }
}

fn rewrite_local_schema_refs(schema: &mut Value, prefix: &str) {
    let Some(object) = schema.as_object_mut() else {
        return;
    };

    if let Some(reference) = object.get_mut("$ref")
        && let Some(rewritten) = reference
            .as_str()
            .and_then(|reference| local_ref_with_prefix(reference, prefix))
    {
        *reference = Value::String(rewritten);
    }
    for key in SCHEMA_MAP_KEYWORDS {
        if let Some(values) = object.get_mut(*key).and_then(Value::as_object_mut) {
            for value in values.values_mut() {
                rewrite_local_schema_refs(value, prefix);
            }
        }
    }
    for key in SCHEMA_VALUE_KEYWORDS {
        if let Some(value) = object.get_mut(*key) {
            rewrite_local_schema_refs(value, prefix);
        }
    }
    for key in SCHEMA_ARRAY_KEYWORDS {
        if let Some(values) = object.get_mut(*key).and_then(Value::as_array_mut) {
            for value in values {
                rewrite_local_schema_refs(value, prefix);
            }
        }
    }
}

fn has_unresolved_local_refs(schema: &Value, root: &Value) -> bool {
    let Some(object) = schema.as_object() else {
        return false;
    };

    if object
        .get("$ref")
        .and_then(Value::as_str)
        .is_some_and(|reference| {
            (reference == "#" || reference.starts_with("#/"))
                && resolve_local_ref(reference, root).is_none()
        })
    {
        return true;
    }
    SCHEMA_MAP_KEYWORDS.iter().any(|key| {
        object
            .get(*key)
            .and_then(Value::as_object)
            .is_some_and(|values| {
                values
                    .values()
                    .any(|value| has_unresolved_local_refs(value, root))
            })
    }) || SCHEMA_VALUE_KEYWORDS.iter().any(|key| {
        object
            .get(*key)
            .is_some_and(|value| has_unresolved_local_refs(value, root))
    }) || SCHEMA_ARRAY_KEYWORDS.iter().any(|key| {
        object
            .get(*key)
            .and_then(Value::as_array)
            .is_some_and(|values| {
                values
                    .iter()
                    .any(|value| has_unresolved_local_refs(value, root))
            })
    })
}

fn preserve_recursive_root_refs(mut schema: Value, root: &Value) -> Value {
    if !has_unresolved_local_refs(&schema, &schema) {
        return schema;
    }

    let Some(schema_object) = schema.as_object_mut() else {
        return schema;
    };
    let definitions = schema_object
        .entry("$defs")
        .or_insert_with(|| Value::Object(Map::new()));
    let Some(definitions) = definitions.as_object_mut() else {
        return schema;
    };
    let mut definition_name = "__dynamo_root".to_string();
    let mut suffix = 2;
    while definitions.contains_key(&definition_name) {
        definition_name = format!("__dynamo_root_{suffix}");
        suffix += 1;
    }
    let escaped_name = definition_name.replace('~', "~0").replace('/', "~1");
    let prefix = format!("#/$defs/{escaped_name}");

    // A cycle can leave a reference to a location outside the extracted
    // argument schema (for example, `#/properties/node`). Preserve the original
    // root under a private definition and rebase only schema-position refs to it.
    rewrite_local_schema_refs(&mut schema, &prefix);
    let mut embedded_root = root.clone();
    rewrite_local_schema_refs(&mut embedded_root, &prefix);
    schema
        .as_object_mut()
        .and_then(|schema| schema.get_mut("$defs"))
        .and_then(Value::as_object_mut)
        .expect("definitions were initialized above")
        .insert(definition_name, embedded_root);
    schema
}

fn resolve_schema_keyword(
    key: &str,
    value: &Value,
    root: &Value,
    seen_refs: &HashSet<String>,
) -> Option<Value> {
    if SCHEMA_MAP_KEYWORDS.contains(&key) {
        return resolve_schema_map(value, root, seen_refs);
    }
    if SCHEMA_VALUE_KEYWORDS.contains(&key) {
        return resolve_local_refs(value, root, seen_refs);
    }
    if SCHEMA_ARRAY_KEYWORDS.contains(&key) {
        let Some(values) = value.as_array() else {
            return Some(value.clone());
        };
        return values
            .iter()
            .map(|value| resolve_local_refs(value, root, seen_refs))
            .collect::<Option<Vec<_>>>()
            .map(Value::Array);
    }

    // Unknown keywords and values under `const`, `enum`, `default`, and `examples`
    // may contain arbitrary JSON data, where a `$ref` object is not a schema reference.
    Some(value.clone())
}

fn resolve_local_refs(schema: &Value, root: &Value, seen_refs: &HashSet<String>) -> Option<Value> {
    match schema {
        Value::Array(values) => values
            .iter()
            .map(|value| resolve_local_refs(value, root, seen_refs))
            .collect::<Option<Vec<_>>>()
            .map(Value::Array),
        Value::Object(object) => {
            if let Some(reference) = object.get("$ref").and_then(Value::as_str) {
                if seen_refs.contains(reference) {
                    return Some(schema.clone());
                }
                let target = resolve_local_ref(reference, root)?;
                let mut nested_seen = seen_refs.clone();
                nested_seen.insert(reference.to_string());
                let resolved = resolve_local_refs(target, root, &nested_seen)?;
                let siblings = object
                    .iter()
                    .filter(|(key, _)| key.as_str() != "$ref")
                    .map(|(key, value)| {
                        resolve_schema_keyword(key, value, root, seen_refs)
                            .map(|value| (key.clone(), value))
                    })
                    .collect::<Option<Map<_, _>>>()?;
                if siblings.is_empty() {
                    return Some(resolved);
                }
                return Some(serde_json::json!({
                    "allOf": [resolved, Value::Object(siblings)]
                }));
            }

            object
                .iter()
                .map(|(key, value)| {
                    resolve_schema_keyword(key, value, root, seen_refs)
                        .map(|value| (key.clone(), value))
                })
                .collect::<Option<Map<_, _>>>()
                .map(Value::Object)
        }
        _ => Some(schema.clone()),
    }
}

fn intersect_json_types(candidates: &[JsonType], requested: &[JsonType]) -> Vec<JsonType> {
    let candidate_has_number = candidates.contains(&JsonType::Number);
    let requested_has_number = requested.contains(&JsonType::Number);
    let accepts_integer = (candidate_has_number || candidates.contains(&JsonType::Integer))
        && (requested_has_number || requested.contains(&JsonType::Integer));
    JSON_TYPES
        .into_iter()
        .filter(|json_type| match json_type {
            JsonType::Number => candidate_has_number && requested_has_number,
            JsonType::Integer => accepts_integer && !(candidate_has_number && requested_has_number),
            _ => candidates.contains(json_type) && requested.contains(json_type),
        })
        .collect()
}

fn json_type_constraint(json_types: &[JsonType]) -> Value {
    match json_types {
        [json_type] => Value::String(json_type.json_name().to_string()),
        _ => Value::Array(
            json_types
                .iter()
                .map(|json_type| Value::String(json_type.json_name().to_string()))
                .collect(),
        ),
    }
}

fn narrow_resolved_schema(
    schema: Value,
    root: &Value,
    requested_types: &[JsonType],
) -> Option<Map<String, Value>> {
    let candidates = schema_types(&schema, root, &HashSet::new());
    let json_types = intersect_json_types(&candidates, requested_types);
    if json_types.is_empty() {
        return None;
    }

    let mut narrowed = match schema {
        Value::Bool(true) => Map::new(),
        Value::Object(object) => object,
        _ => return None,
    };
    for keyword in ["anyOf", "oneOf"] {
        let Some(options) = narrowed.get(keyword).and_then(Value::as_array) else {
            continue;
        };
        let options = options
            .iter()
            .filter_map(|option| {
                narrow_resolved_schema(option.clone(), root, &json_types).map(Value::Object)
            })
            .collect::<Vec<_>>();
        if options.is_empty() {
            return None;
        }
        narrowed.insert(keyword.to_string(), Value::Array(options));
    }
    if let Some(values) = narrowed.get_mut("enum").and_then(Value::as_array_mut) {
        values.retain(|value| json_types.iter().any(|json_type| json_type.accepts(value)));
        if values.is_empty() {
            return None;
        }
    }
    if narrowed
        .get("const")
        .is_some_and(|value| !json_types.iter().any(|json_type| json_type.accepts(value)))
    {
        return None;
    }
    narrowed.insert("type".to_string(), json_type_constraint(&json_types));
    Some(narrowed)
}

fn schema_for_xtml_type(schema: &Value, root: &Value, xtml_type: &str) -> Option<Value> {
    let resolved = resolve_local_refs(schema, root, &HashSet::new())?;
    let mut json_types = schema_types(&resolved, &resolved, &HashSet::new())
        .into_iter()
        .filter(|json_type| json_type.xtml_name() == xtml_type)
        .collect::<Vec<_>>();
    if json_types.contains(&JsonType::Number) {
        json_types.retain(|json_type| *json_type != JsonType::Integer);
    }
    let mut narrowed = narrow_resolved_schema(resolved, root, &json_types)?;
    if let Some(root) = root.as_object() {
        for key in ["$defs", "definitions"] {
            if let Some(value) = root.get(key) {
                narrowed
                    .entry(key.to_string())
                    .or_insert_with(|| value.clone());
            }
        }
    }
    Some(preserve_recursive_root_refs(Value::Object(narrowed), root))
}

fn argument_format(key: &str, schema: &Value, root: &Value) -> Option<Format> {
    let alternatives = xtml_types(schema, root)
        .into_iter()
        .filter_map(|xtml_type| {
            let content = if xtml_type == "string" {
                string_content_format(&schema_for_xtml_type(schema, root, xtml_type)?)
            } else {
                Format::JsonSchema(JsonSchemaFormat {
                    json_schema: schema_for_xtml_type(schema, root, xtml_type)?,
                    style: JsonSchemaStyle::Json,
                })
            };
            Some(Format::Tag(TagFormat {
                begin: format!(
                    "{OPEN}argument key=\"{}\" type=\"{xtml_type}\"{SEP}",
                    escape_attr(key)
                ),
                content: Box::new(content),
                end: ARGUMENT_CLOSE.to_string(),
            }))
        })
        .collect::<Vec<_>>();

    (!alternatives.is_empty()).then(|| one_of(alternatives))
}

fn optional_arguments(
    properties: &Map<String, Value>,
    required_keys: &[&str],
    root: &Value,
) -> Option<Format> {
    let arguments = properties
        .iter()
        .filter(|(key, _)| !required_keys.contains(&key.as_str()))
        .filter_map(|(key, schema)| {
            matches!(schema, Value::Bool(_) | Value::Object(_))
                .then(|| argument_format(key, schema, root))
                .flatten()
        })
        .collect::<Vec<_>>();

    (!arguments.is_empty()).then(|| star(one_of(arguments)))
}

fn canonical_required_arguments(
    mut arguments: Vec<Format>,
    optional_arguments: Option<Format>,
) -> Format {
    // Guided decoding only needs one schema-valid order. Keep the schema's
    // required-array order, then allow only complete, declared optional
    // argument tags. Arbitrary text would be accepted by the grammar but
    // rejected by the K3 parser, while a permissive argument tag could repeat
    // a required key and overwrite its constrained value.
    if let Some(optional_arguments) = optional_arguments {
        arguments.push(optional_arguments);
    }
    Format::Sequence(SequenceFormat {
        elements: arguments,
    })
}

fn arguments_block(tool: &ToolDefinition, strict_schema: bool) -> Format {
    if !super::builder::uses_declared_tool_schema(tool, strict_schema) {
        return star(Format::Tag(permissive_argument_tag()));
    }

    let Some(root) = tool.parameters.as_ref() else {
        return star(Format::Tag(permissive_argument_tag()));
    };
    let resolved_root = root
        .as_object()
        .filter(|object| object.contains_key("$ref") || object.contains_key("allOf"))
        .and_then(|_| resolved_root_parameters(root));
    let Some(parameters) = resolved_root.as_ref().unwrap_or(root).as_object() else {
        return star(Format::Tag(permissive_argument_tag()));
    };
    let Some(properties) = parameters.get("properties").and_then(Value::as_object) else {
        return star(Format::Tag(permissive_argument_tag()));
    };
    let required = match parameters.get("required") {
        None => Vec::new(),
        Some(Value::Array(required)) if required.iter().all(|value| value.as_str().is_some()) => {
            required
                .iter()
                .filter_map(Value::as_str)
                .collect::<Vec<_>>()
        }
        Some(_) => return star(Format::Tag(permissive_argument_tag())),
    };
    let optional_arguments = optional_arguments(properties, &required, root);

    if required.is_empty() {
        return optional_arguments.unwrap_or_else(|| {
            Format::ConstString(ConstStringFormat {
                value: String::new(),
            })
        });
    }

    let mut arguments = Vec::with_capacity(required.len());
    for key in &required {
        let Some(schema) = properties.get(*key) else {
            return star(Format::Tag(permissive_argument_tag()));
        };
        if !matches!(schema, Value::Bool(_) | Value::Object(_)) {
            return star(Format::Tag(permissive_argument_tag()));
        }
        let Some(argument) = argument_format(key, schema, root) else {
            return star(Format::Tag(permissive_argument_tag()));
        };
        arguments.push(argument);
    }
    canonical_required_arguments(arguments, optional_arguments)
}

fn build_auto_structural_tag(
    tools: Vec<&ToolDefinition>,
    ctx: &ToolCallFormatBuildContext<'_>,
) -> StructuralTag {
    let strict_schema = ctx.strict_schema();
    let call_tags: Vec<_> = tools
        .into_iter()
        .map(|tool| call_tag(tool, strict_schema))
        .collect();
    let parallel_tool_calls = !ctx.stop_after_first();
    let calls = if parallel_tool_calls {
        Format::TagsWithSeparator(TagsWithSeparatorFormat {
            tags: call_tags,
            separator: String::new(),
            at_least_one: true,
            stop_after_first: false,
        })
    } else {
        one_of(call_tags.into_iter().map(Format::Tag).collect())
    };
    let tools_tag = TagFormat {
        begin: TOOLS_OPEN.to_string(),
        content: Box::new(calls),
        end: TOOLS_CLOSE.to_string(),
    };
    // Express the optional tools suffix with existing public format nodes.
    // Adding `excludes` to the public `TriggeredTagsFormat` struct would break
    // downstream struct literals in the published parser crate.
    let suffix = Format::Sequence(SequenceFormat {
        elements: vec![
            Format::AnyText(AnyTextFormat {
                excludes: vec![
                    TOOLS_OPEN.to_string(),
                    THINK_OPEN.to_string(),
                    THINK_CLOSE.to_string(),
                    CALL_OPEN.to_string(),
                ],
            }),
            optional(Format::Tag(tools_tag)),
        ],
    });
    let format = if ctx.starts_in_reasoning {
        Format::Sequence(SequenceFormat {
            elements: vec![
                Format::Tag(TagFormat {
                    begin: String::new(),
                    content: Box::new(Format::AnyText(AnyTextFormat { excludes: vec![] })),
                    end: THINK_CLOSE.to_string(),
                }),
                suffix,
            ],
        })
    } else {
        suffix
    };
    StructuralTag { format }
}

fn call_tag(tool: &ToolDefinition, strict_schema: bool) -> TagFormat {
    let begin = format!("{OPEN}call tool=\"{}\" index=\"", escape_attr(&tool.name));
    TagFormat {
        begin,
        content: Box::new(Format::Sequence(SequenceFormat {
            elements: vec![
                Format::Regex(RegexFormat {
                    pattern: CALL_INDEX_PATTERN.to_string(),
                }),
                Format::ConstString(ConstStringFormat {
                    value: format!("\"{SEP}"),
                }),
                arguments_block(tool, strict_schema),
            ],
        })),
        end: CALL_CLOSE.to_string(),
    }
}

/// Build the format-style xgrammar tag for K3's response + tools channels.
pub(crate) fn build_kimi_k3(
    ctx: &ToolCallFormatBuildContext<'_>,
) -> anyhow::Result<Option<StructuralTag>> {
    let (tools, at_least_one) = resolve_tools_to_include(ctx)?;
    if tools.is_empty() {
        return Ok(None);
    }
    if matches!(ctx.tool_choice, crate::tool_calling::ToolChoice::Auto) {
        return Ok(Some(build_auto_structural_tag(tools, ctx)));
    }

    // Moonshot's named-tool contract returns the selected call with no
    // assistant content. Keeping the response channel itself is required by
    // K3's XTML wire format, but leaving its body as `any_text` lets the model
    // put a second, generic `<tool_call>...</tool_call>` representation there
    // before emitting the structurally constrained XTML call. Restrict only
    // named choice; auto/required may legitimately include response text.
    let response_content = if matches!(ctx.tool_choice, crate::tool_calling::ToolChoice::Named(_)) {
        Format::ConstString(ConstStringFormat {
            value: String::new(),
        })
    } else {
        Format::AnyText(AnyTextFormat { excludes: vec![] })
    };
    let response = vec![
        optional(Format::ConstString(ConstStringFormat {
            value: RESPONSE_OPEN.to_string(),
        })),
        Format::Tag(TagFormat {
            begin: String::new(),
            content: Box::new(response_content),
            end: RESPONSE_CLOSE.to_string(),
        }),
    ];
    let calls = Format::TagsWithSeparator(TagsWithSeparatorFormat {
        tags: tools
            .into_iter()
            .map(|tool| call_tag(tool, ctx.strict_schema()))
            .collect(),
        separator: String::new(),
        at_least_one: true,
        stop_after_first: ctx.stop_after_first(),
    });
    let tools_channel = Format::Tag(TagFormat {
        begin: TOOLS_OPEN.to_string(),
        content: Box::new(calls),
        end: TOOLS_CLOSE.to_string(),
    });

    let tools_part = if at_least_one {
        tools_channel
    } else {
        optional(tools_channel)
    };
    let mut elements = response;
    elements.push(tools_part);
    elements.push(optional(Format::ConstString(ConstStringFormat {
        value: MESSAGE_CLOSE.to_string(),
    })));

    Ok(Some(StructuralTag {
        format: Format::Sequence(SequenceFormat { elements }),
    }))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::tool_calling::structural_tag::builder::StructuralTagSchemaMode;
    use crate::tool_calling::{ToolChoice, ToolDefinition};

    fn tools() -> Vec<ToolDefinition> {
        vec![
            ToolDefinition {
                name: "get_weather".to_string(),
                parameters: Some(json!({
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "days": {"type": "integer"}
                    },
                    "required": ["city"]
                })),
                strict: None,
            },
            ToolDefinition {
                name: "run_command".to_string(),
                parameters: Some(json!({
                    "type": "object",
                    "properties": {"command": {"type": "string"}}
                })),
                strict: None,
            },
        ]
    }

    fn context<'a>(
        choice: &'a ToolChoice,
        tools: &'a [ToolDefinition],
    ) -> ToolCallFormatBuildContext<'a> {
        ToolCallFormatBuildContext {
            tool_choice: choice,
            tools,
            parallel_tool_calls: None,
            schema_mode: StructuralTagSchemaMode::Auto,
            starts_in_reasoning: false,
        }
    }

    #[test]
    fn named_choice_requires_only_selected_xtml_call() {
        let tools = tools();
        let choice = ToolChoice::Named("get_weather".to_string());
        let value =
            serde_json::to_value(build_kimi_k3(&context(&choice, &tools)).unwrap().unwrap())
                .unwrap();

        assert_eq!(value["type"], "structural_tag");
        assert_eq!(value["format"]["type"], "sequence");
        let tools_tag = &value["format"]["elements"][2];
        assert_eq!(tools_tag["begin"], TOOLS_OPEN);
        let calls = tools_tag["content"]["tags"].as_array().unwrap();
        assert_eq!(calls.len(), 1);
        assert!(
            calls[0]["begin"]
                .as_str()
                .unwrap()
                .contains("tool=\"get_weather\"")
        );
        assert!(
            !value.to_string().contains("tool=\\\"run_command\\\""),
            "a named choice must exclude every other tool"
        );
    }

    #[test]
    fn named_choice_requires_an_empty_response_body() {
        let tools = tools();
        let choice = ToolChoice::Named("get_weather".to_string());
        let value =
            serde_json::to_value(build_kimi_k3(&context(&choice, &tools)).unwrap().unwrap())
                .unwrap();

        let response_body = &value["format"]["elements"][1]["content"];
        assert_eq!(response_body["type"], "const_string");
        assert_eq!(response_body["value"], "");
    }

    #[test]
    fn required_choice_keeps_the_existing_response_body() {
        let tools = tools();
        let choice = ToolChoice::Required;
        let value =
            serde_json::to_value(build_kimi_k3(&context(&choice, &tools)).unwrap().unwrap())
                .unwrap();
        let response_body = &value["format"]["elements"][1]["content"];

        assert_eq!(response_body["type"], "any_text");
        assert_eq!(response_body["excludes"], json!([]));
    }

    #[test]
    fn named_choice_is_mandatory_and_auto_uses_an_optional_tools_suffix() {
        let tools = tools();
        let named = ToolChoice::Named("get_weather".to_string());
        let named_value =
            serde_json::to_value(build_kimi_k3(&context(&named, &tools)).unwrap().unwrap())
                .unwrap();
        assert_eq!(named_value["format"]["elements"][2]["type"], "tag");

        let auto = ToolChoice::Auto;
        let auto_value =
            serde_json::to_value(build_kimi_k3(&context(&auto, &tools)).unwrap().unwrap()).unwrap();
        assert_eq!(auto_value["format"]["type"], "sequence");
        assert_eq!(auto_value["format"]["elements"][0]["type"], "any_text");
        assert_eq!(
            auto_value["format"]["elements"][0]["excludes"],
            json!([TOOLS_OPEN, THINK_OPEN, THINK_CLOSE, CALL_OPEN])
        );
        assert_eq!(auto_value["format"]["elements"][1]["type"], "optional");
        assert_eq!(
            auto_value["format"]["elements"][1]["content"]["begin"],
            TOOLS_OPEN
        );
    }

    #[test]
    fn auto_requires_declared_arguments_then_allows_optional_content() {
        let tools = tools();
        let choice = ToolChoice::Auto;
        let value =
            serde_json::to_value(build_kimi_k3(&context(&choice, &tools)).unwrap().unwrap())
                .unwrap();
        let calls = &value["format"]["elements"][1]["content"]["content"];
        assert_eq!(calls["type"], "tags_with_separator");
        let call = &calls["tags"][0];
        assert_eq!(call["begin"], "<|open|>call tool=\"get_weather\" index=\"");
        assert_eq!(call["content"]["elements"][0]["pattern"], "[1-9][0-9]*");

        let arguments = &call["content"]["elements"][2];
        assert_eq!(arguments["type"], "sequence");
        assert_eq!(
            arguments["elements"][0]["begin"],
            "<|open|>argument key=\"city\" type=\"string\"<|sep|>"
        );
        assert_eq!(arguments["elements"][0]["content"]["type"], "any_text");
        assert_eq!(
            arguments["elements"][0]["content"]["excludes"],
            json!([CLOSE])
        );
        assert_eq!(arguments["elements"][1]["type"], "star");
        assert_eq!(
            arguments["elements"][1]["content"]["begin"],
            "<|open|>argument key=\"days\" type=\"number\"<|sep|>"
        );
    }

    #[test]
    fn auto_required_arguments_use_canonical_schema_order() {
        let tools = vec![ToolDefinition {
            name: "get_weather".to_string(),
            parameters: Some(json!({
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"},
                    "units": {"type": "string"}
                },
                "required": ["city", "days"]
            })),
            strict: None,
        }];
        let choice = ToolChoice::Auto;
        let value =
            serde_json::to_value(build_kimi_k3(&context(&choice, &tools)).unwrap().unwrap())
                .unwrap();
        let arguments = &value["format"]["elements"][1]["content"]["content"]["tags"][0]["content"]
            ["elements"][2];
        let elements = arguments["elements"].as_array().unwrap();

        assert_eq!(arguments["type"], "sequence");
        assert_eq!(elements.len(), 3);
        assert_eq!(
            elements[0]["begin"],
            "<|open|>argument key=\"city\" type=\"string\"<|sep|>"
        );
        assert_eq!(
            elements[1]["begin"],
            "<|open|>argument key=\"days\" type=\"number\"<|sep|>"
        );
        assert_eq!(elements[1]["content"]["type"], "json_schema");
        assert_eq!(elements[1]["content"]["json_schema"]["type"], "integer");
        assert_eq!(elements[2]["type"], "star");
        assert_eq!(
            elements[2]["content"]["begin"],
            "<|open|>argument key=\"units\" type=\"string\"<|sep|>"
        );
    }

    #[test]
    fn auto_required_non_string_arguments_use_their_json_schemas() {
        let tools = vec![ToolDefinition {
            name: "typed_tool".to_string(),
            parameters: Some(json!({
                "type": "object",
                "properties": {
                    "count": {"type": "integer", "minimum": 1},
                    "enabled": {"type": "boolean"},
                    "items": {"type": "array", "items": {"type": "string"}},
                    "metadata": {
                        "type": "object",
                        "properties": {"source": {"type": "string"}},
                        "required": ["source"]
                    }
                },
                "required": ["count", "enabled", "items", "metadata"]
            })),
            strict: None,
        }];
        let choice = ToolChoice::Auto;
        let value =
            serde_json::to_value(build_kimi_k3(&context(&choice, &tools)).unwrap().unwrap())
                .unwrap();
        let arguments = &value["format"]["elements"][1]["content"]["content"]["tags"][0]["content"]
            ["elements"][2];
        let elements = &arguments["elements"];

        for index in 0..4 {
            assert_eq!(elements[index]["content"]["type"], "json_schema");
        }
        assert_eq!(elements[0]["content"]["json_schema"]["minimum"], 1);
        assert_eq!(
            elements[2]["content"]["json_schema"]["items"]["type"],
            "string"
        );
        assert_eq!(
            elements[3]["content"]["json_schema"]["required"],
            json!(["source"])
        );
    }

    #[test]
    fn auto_required_ref_keeps_root_definitions_in_typed_content() {
        let tools = vec![ToolDefinition {
            name: "lookup".to_string(),
            parameters: Some(json!({
                "type": "object",
                "$defs": {
                    "identifier": {"type": "integer", "minimum": 1}
                },
                "properties": {
                    "id": {"$ref": "#/$defs/identifier"}
                },
                "required": ["id"]
            })),
            strict: None,
        }];
        let choice = ToolChoice::Auto;
        let value =
            serde_json::to_value(build_kimi_k3(&context(&choice, &tools)).unwrap().unwrap())
                .unwrap();
        let argument = &value["format"]["elements"][1]["content"]["content"]["tags"][0]["content"]
            ["elements"][2]["elements"][0];

        assert_eq!(argument["content"]["type"], "json_schema");
        assert_eq!(argument["content"]["json_schema"]["type"], "integer");
        assert_eq!(argument["content"]["json_schema"]["minimum"], 1);
        assert!(argument["content"]["json_schema"].get("$ref").is_none());
    }

    #[test]
    fn auto_required_ref_into_properties_is_resolved_from_the_original_root() {
        let tools = vec![ToolDefinition {
            name: "lookup".to_string(),
            parameters: Some(json!({
                "type": "object",
                "properties": {
                    "other": {"type": "integer", "minimum": 1},
                    "value": {"$ref": "#/properties/other"}
                },
                "required": ["value"]
            })),
            strict: None,
        }];
        let choice = ToolChoice::Auto;
        let value =
            serde_json::to_value(build_kimi_k3(&context(&choice, &tools)).unwrap().unwrap())
                .unwrap();
        let argument = &value["format"]["elements"][1]["content"]["content"]["tags"][0]["content"]
            ["elements"][2]["elements"][0];

        assert_eq!(argument["content"]["json_schema"]["type"], "integer");
        assert_eq!(argument["content"]["json_schema"]["minimum"], 1);
        assert!(argument["content"]["json_schema"].get("$ref").is_none());
    }

    #[test]
    fn string_content_enforces_min_length_without_a_maximum() {
        let value = serde_json::to_value(string_content_format(&json!({
            "type": "string",
            "minLength": 2
        })))
        .unwrap();

        assert_eq!(value["type"], "regex");
        assert_eq!(value["pattern"], format!("{STRING_ATOM}{{2,}}"));
    }

    #[test]
    fn string_content_uses_xgrammar_pattern_precedence_over_length_bounds() {
        let value = serde_json::to_value(string_content_format(&json!({
            "type": "string",
            "pattern": "^item-[0-9]+$",
            "minLength": 6,
            "maxLength": 12
        })))
        .unwrap();

        assert_eq!(value["type"], "regex");
        assert_eq!(value["pattern"], "(?:item-[0-9]+)");
    }

    #[test]
    fn string_content_preserves_an_escaped_trailing_dollar() {
        assert!(!has_unescaped_trailing_dollar(r"^price\$"));
        assert!(has_unescaped_trailing_dollar(r"^path\\$"));

        let value = serde_json::to_value(string_content_format(&json!({
            "type": "string",
            "pattern": r"^price\$"
        })))
        .unwrap();

        assert_eq!(value["type"], "regex");
        assert_eq!(value["pattern"], format!(r"(?:price\$){STRING_ATOM}*"));
    }

    #[test]
    fn auto_string_arguments_preserve_schema_constraints_and_allow_empty_values() {
        let tools = vec![ToolDefinition {
            name: "strings".to_string(),
            parameters: Some(json!({
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["fast", "safe"]},
                    "bounded": {"type": "string", "minLength": 0, "maxLength": 8},
                    "prefixed": {"type": "string", "pattern": "^item-[0-9]+$"},
                    "empty_ok": {"type": "string", "minLength": 0, "maxLength": 0}
                },
                "required": ["mode", "bounded", "prefixed", "empty_ok"]
            })),
            strict: None,
        }];
        let choice = ToolChoice::Auto;
        let value =
            serde_json::to_value(build_kimi_k3(&context(&choice, &tools)).unwrap().unwrap())
                .unwrap();
        let arguments = &value["format"]["elements"][1]["content"]["content"]["tags"][0]["content"]
            ["elements"][2]["elements"];

        assert_eq!(arguments[0]["content"]["type"], "or");
        assert_eq!(arguments[0]["content"]["elements"][0]["value"], "fast");
        assert_eq!(
            arguments[1]["content"]["pattern"],
            format!("{STRING_ATOM}{{0,8}}")
        );
        assert_eq!(arguments[2]["content"]["pattern"], "(?:item-[0-9]+)");
        assert_eq!(
            arguments[3]["content"]["pattern"],
            format!("{STRING_ATOM}{{0,0}}")
        );
    }

    #[test]
    fn auto_numeric_union_uses_one_number_tag_with_the_declared_schema() {
        let tools = vec![ToolDefinition {
            name: "measure".to_string(),
            parameters: Some(json!({
                "type": "object",
                "properties": {
                    "value": {"anyOf": [{"type": "integer"}, {"type": "number"}]}
                },
                "required": ["value"]
            })),
            strict: None,
        }];
        let choice = ToolChoice::Auto;
        let value =
            serde_json::to_value(build_kimi_k3(&context(&choice, &tools)).unwrap().unwrap())
                .unwrap();
        let argument = &value["format"]["elements"][1]["content"]["content"]["tags"][0]["content"]
            ["elements"][2]["elements"][0];

        assert_eq!(
            argument["begin"],
            "<|open|>argument key=\"value\" type=\"number\"<|sep|>"
        );
        assert_eq!(argument["content"]["type"], "json_schema");
        assert_eq!(
            argument["content"]["json_schema"]["anyOf"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
    }

    #[test]
    fn auto_optional_union_keeps_each_representable_xtml_type() {
        let tools = vec![ToolDefinition {
            name: "lookup".to_string(),
            parameters: Some(json!({
                "type": "object",
                "properties": {
                    "value": {"type": ["string", "null"]}
                }
            })),
            strict: None,
        }];
        let choice = ToolChoice::Auto;
        let value =
            serde_json::to_value(build_kimi_k3(&context(&choice, &tools)).unwrap().unwrap())
                .unwrap();
        let alternatives = &value["format"]["elements"][1]["content"]["content"]["tags"][0]["content"]
            ["elements"][2]["content"]["elements"];

        assert_eq!(alternatives.as_array().unwrap().len(), 2);
        assert!(
            alternatives[0]["begin"]
                .as_str()
                .unwrap()
                .contains("type=\"string\"")
        );
        assert!(
            alternatives[1]["begin"]
                .as_str()
                .unwrap()
                .contains("type=\"null\"")
        );
    }

    #[test]
    fn auto_recursive_optional_object_keeps_the_argument_and_definitions() {
        let tools = vec![ToolDefinition {
            name: "walk".to_string(),
            parameters: Some(json!({
                "type": "object",
                "properties": {
                    "child": {"$ref": "#/$defs/node"}
                },
                "$defs": {
                    "node": {
                        "type": "object",
                        "properties": {
                            "next": {"$ref": "#/$defs/node"}
                        }
                    }
                }
            })),
            strict: None,
        }];
        let choice = ToolChoice::Auto;
        let value =
            serde_json::to_value(build_kimi_k3(&context(&choice, &tools)).unwrap().unwrap())
                .unwrap();
        let arguments = &value["format"]["elements"][1]["content"]["content"]["tags"][0]["content"]
            ["elements"][2];

        assert_eq!(arguments["type"], "star");
        assert_eq!(
            arguments["content"]["begin"],
            "<|open|>argument key=\"child\" type=\"object\"<|sep|>"
        );
        let schema = &arguments["content"]["content"]["json_schema"];
        assert_eq!(schema["type"], "object");
        assert_eq!(schema["properties"]["next"]["$ref"], "#/$defs/node");
        assert_eq!(schema["$defs"]["node"]["type"], "object");
    }

    #[test]
    fn recursive_property_reference_remains_resolvable_after_extraction() {
        let root = json!({
            "type": "object",
            "properties": {
                "node": {
                    "type": "object",
                    "properties": {
                        "next": {"$ref": "#/properties/node"},
                        "metadata": {
                            "type": "object",
                            "const": {"$ref": "literal"}
                        }
                    }
                }
            }
        });
        let schema = schema_for_xtml_type(&root["properties"]["node"], &root, "object").unwrap();
        let reference = schema["properties"]["next"]["properties"]["next"]["$ref"]
            .as_str()
            .unwrap();

        assert!(resolve_local_ref(reference, &schema).is_some());
        assert!(!has_unresolved_local_refs(&schema, &schema));
        assert!(reference.starts_with("#/$defs/__dynamo_root"));
        assert_eq!(
            schema["properties"]["metadata"]["const"],
            json!({"$ref": "literal"})
        );
    }

    #[test]
    fn root_reference_resolves_to_the_original_schema() {
        let root = json!({"type": "object"});

        assert_eq!(resolve_local_ref("#", &root), Some(&root));
    }

    #[test]
    fn auto_union_narrowing_preserves_value_constraints() {
        let tools = vec![ToolDefinition {
            name: "choose".to_string(),
            parameters: Some(json!({
                "type": "object",
                "properties": {
                    "value": {
                        "type": ["integer", "null"],
                        "enum": [1, null],
                        "minimum": 1
                    }
                },
                "required": ["value"]
            })),
            strict: None,
        }];
        let choice = ToolChoice::Auto;
        let value =
            serde_json::to_value(build_kimi_k3(&context(&choice, &tools)).unwrap().unwrap())
                .unwrap();
        let alternatives = &value["format"]["elements"][1]["content"]["content"]["tags"][0]["content"]
            ["elements"][2]["elements"][0]["elements"];
        let number = alternatives
            .as_array()
            .unwrap()
            .iter()
            .find(|alternative| {
                alternative["begin"]
                    .as_str()
                    .is_some_and(|begin| begin.contains("type=\"number\""))
            })
            .expect("number alternative");
        let schema = &number["content"]["json_schema"];

        assert_eq!(schema["type"], "integer");
        assert_eq!(schema["enum"], json!([1]));
        assert_eq!(schema["minimum"], 1);
        let null = alternatives
            .as_array()
            .unwrap()
            .iter()
            .find(|alternative| {
                alternative["begin"]
                    .as_str()
                    .is_some_and(|begin| begin.contains("type=\"null\""))
            })
            .expect("null alternative");
        assert_eq!(null["content"]["json_schema"]["enum"], json!([null]));
    }

    #[test]
    fn integer_union_keeps_integral_float_enum_values() {
        let schema = json!({
            "type": ["integer", "null"],
            "enum": [1.0, null]
        });
        let narrowed = schema_for_xtml_type(&schema, &schema, "number").unwrap();

        assert_eq!(narrowed["type"], "integer");
        assert_eq!(narrowed["enum"], json!([1.0]));
    }

    #[test]
    fn union_composition_removes_other_xtml_types() {
        let schema = json!({
            "anyOf": [
                {"type": "integer", "enum": [1]},
                {"type": "null"}
            ]
        });
        let narrowed = schema_for_xtml_type(&schema, &schema, "number").unwrap();
        let options = narrowed["anyOf"].as_array().unwrap();

        assert_eq!(narrowed["type"], "integer");
        assert_eq!(options.len(), 1);
        assert_eq!(options[0]["type"], "integer");
        assert_eq!(options[0]["enum"], json!([1]));
    }

    #[test]
    fn auto_string_type_union_preserves_applicable_enum_values() {
        let tools = vec![ToolDefinition {
            name: "choose".to_string(),
            parameters: Some(json!({
                "type": "object",
                "properties": {
                    "value": {
                        "type": ["string", "null"],
                        "enum": ["safe", null]
                    }
                },
                "required": ["value"]
            })),
            strict: None,
        }];
        let choice = ToolChoice::Auto;
        let value =
            serde_json::to_value(build_kimi_k3(&context(&choice, &tools)).unwrap().unwrap())
                .unwrap();
        let alternatives = &value["format"]["elements"][1]["content"]["content"]["tags"][0]["content"]
            ["elements"][2]["elements"][0]["elements"];
        let string = alternatives
            .as_array()
            .unwrap()
            .iter()
            .find(|alternative| {
                alternative["begin"]
                    .as_str()
                    .is_some_and(|begin| begin.contains("type=\"string\""))
            })
            .expect("string alternative");

        assert_eq!(string["content"]["type"], "const_string");
        assert_eq!(string["content"]["value"], "safe");
    }

    #[test]
    fn auto_string_references_and_unions_preserve_enum_constraints() {
        let tools = vec![ToolDefinition {
            name: "choose".to_string(),
            parameters: Some(json!({
                "type": "object",
                "properties": {
                    "direct": {"$ref": "#/$defs/safe"},
                    "union": {
                        "anyOf": [
                            {"type": "string", "enum": ["safe"]},
                            {"type": "null"}
                        ]
                    }
                },
                "required": ["direct", "union"],
                "$defs": {
                    "safe": {"type": "string", "enum": ["safe"]}
                }
            })),
            strict: None,
        }];
        let choice = ToolChoice::Auto;
        let value =
            serde_json::to_value(build_kimi_k3(&context(&choice, &tools)).unwrap().unwrap())
                .unwrap();
        let arguments = &value["format"]["elements"][1]["content"]["content"]["tags"][0]["content"]
            ["elements"][2]["elements"];

        assert_eq!(arguments[0]["content"]["type"], "const_string");
        assert_eq!(arguments[0]["content"]["value"], "safe");
        let union_alternatives = arguments[1]["elements"].as_array().unwrap();
        let string = union_alternatives
            .iter()
            .find(|alternative| {
                alternative["begin"]
                    .as_str()
                    .is_some_and(|begin| begin.contains("type=\"string\""))
            })
            .expect("string alternative");
        assert_eq!(string["content"]["type"], "const_string");
        assert_eq!(string["content"]["value"], "safe");
    }

    #[test]
    fn string_all_of_preserves_compatible_enum_constraints() {
        let schema = json!({
            "allOf": [
                {"type": "string"},
                {"enum": ["safe"]}
            ]
        });
        let narrowed = schema_for_xtml_type(&schema, &schema, "string").unwrap();
        let format = serde_json::to_value(string_content_format(&narrowed)).unwrap();

        assert_eq!(format["type"], "const_string");
        assert_eq!(format["value"], "safe");
    }

    #[test]
    fn string_enum_intersects_length_and_pattern_constraints() {
        let schema = json!({
            "allOf": [
                {"type": "string", "enum": ["x", "code-42", "other"]},
                {"minLength": 3, "pattern": "^code-[0-9]+$"}
            ]
        });
        let narrowed = schema_for_xtml_type(&schema, &schema, "string").unwrap();
        let format = serde_json::to_value(string_content_format(&narrowed)).unwrap();

        assert_eq!(format["type"], "const_string");
        assert_eq!(format["value"], "code-42");

        let tools = vec![ToolDefinition {
            name: "select".to_string(),
            parameters: Some(json!({
                "type": "object",
                "properties": {"value": schema},
                "required": ["value"]
            })),
            strict: None,
        }];
        let choice = ToolChoice::Auto;
        let value =
            serde_json::to_value(build_kimi_k3(&context(&choice, &tools)).unwrap().unwrap())
                .unwrap();
        let argument = &value["format"]["elements"][1]["content"]["content"]["tags"][0]["content"]
            ["elements"][2]["elements"][0];
        assert_eq!(argument["content"]["type"], "const_string");
        assert_eq!(argument["content"]["value"], "code-42");
    }

    #[test]
    fn root_referenced_parameters_keep_required_arguments() {
        let tools = vec![ToolDefinition {
            name: "weather".to_string(),
            parameters: Some(json!({
                "$ref": "#/$defs/Args",
                "$defs": {
                    "Args": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                        "additionalProperties": false
                    }
                }
            })),
            strict: None,
        }];
        let choice = ToolChoice::Auto;
        let value =
            serde_json::to_value(build_kimi_k3(&context(&choice, &tools)).unwrap().unwrap())
                .unwrap();
        let arguments = &value["format"]["elements"][1]["content"]["content"]["tags"][0]["content"]
            ["elements"][2];

        assert_eq!(arguments["type"], "sequence");
        assert_eq!(
            arguments["elements"][0]["begin"],
            "<|open|>argument key=\"city\" type=\"string\"<|sep|>"
        );
    }

    #[test]
    fn auto_object_const_treats_ref_as_literal_data() {
        let tools = vec![ToolDefinition {
            name: "literal".to_string(),
            parameters: Some(json!({
                "type": "object",
                "properties": {
                    "value": {
                        "type": "object",
                        "const": {"$ref": "literal"}
                    }
                },
                "required": ["value"]
            })),
            strict: None,
        }];
        let choice = ToolChoice::Auto;
        let value =
            serde_json::to_value(build_kimi_k3(&context(&choice, &tools)).unwrap().unwrap())
                .unwrap();
        let argument = &value["format"]["elements"][1]["content"]["content"]["tags"][0]["content"]
            ["elements"][2]["elements"][0];

        assert_eq!(
            argument["begin"],
            "<|open|>argument key=\"value\" type=\"object\"<|sep|>"
        );
        assert_eq!(
            argument["content"]["json_schema"]["const"],
            json!({"$ref": "literal"})
        );
    }

    #[test]
    fn auto_without_required_properties_allows_an_empty_argument_body() {
        let tools = vec![ToolDefinition {
            name: "run_command".to_string(),
            parameters: Some(json!({
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer"}
                }
            })),
            strict: None,
        }];
        let choice = ToolChoice::Auto;
        let value =
            serde_json::to_value(build_kimi_k3(&context(&choice, &tools)).unwrap().unwrap())
                .unwrap();
        let arguments = &value["format"]["elements"][1]["content"]["content"]["tags"][0]["content"]
            ["elements"][2];

        assert_eq!(arguments["type"], "star");
        assert_eq!(arguments["content"]["type"], "or");
        assert_eq!(
            arguments["content"]["elements"].as_array().unwrap().len(),
            2
        );
        let alternatives = arguments["content"]["elements"].as_array().unwrap();
        assert_eq!(alternatives[0]["content"]["type"], "any_text");
        assert_eq!(alternatives[0]["content"]["excludes"], json!([CLOSE]));
        assert_eq!(alternatives[1]["content"]["type"], "json_schema");
        assert_eq!(alternatives[1]["content"]["json_schema"]["type"], "integer");
    }

    #[test]
    fn auto_explicit_non_strict_tool_uses_permissive_arguments() {
        let tools = vec![ToolDefinition {
            name: "get_weather".to_string(),
            parameters: Some(json!({
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            })),
            strict: Some(false),
        }];
        let choice = ToolChoice::Auto;
        let value =
            serde_json::to_value(build_kimi_k3(&context(&choice, &tools)).unwrap().unwrap())
                .unwrap();
        let arguments = &value["format"]["elements"][1]["content"]["content"]["tags"][0]["content"]
            ["elements"][2];

        assert_eq!(arguments["type"], "star");
        assert_eq!(arguments["content"]["begin"], "<|open|>argument ");
        assert!(!arguments.to_string().contains("key=\\\"city\\\""));

        let strict_ctx = ToolCallFormatBuildContext {
            tool_choice: &choice,
            tools: &tools,
            parallel_tool_calls: None,
            schema_mode: StructuralTagSchemaMode::Strict,
            starts_in_reasoning: false,
        };
        let strict_value =
            serde_json::to_value(build_kimi_k3(&strict_ctx).unwrap().unwrap()).unwrap();
        let strict_arguments = &strict_value["format"]["elements"][1]["content"]["content"]["tags"]
            [0]["content"]["elements"][2];

        assert_eq!(strict_arguments["type"], "sequence");
        assert!(strict_arguments.to_string().contains("key=\\\"city\\\""));
    }

    #[test]
    fn auto_thinking_closes_reasoning_before_the_triggered_tools_suffix() {
        let tools = tools();
        let choice = ToolChoice::Auto;
        let ctx = ToolCallFormatBuildContext {
            tool_choice: &choice,
            tools: &tools,
            parallel_tool_calls: None,
            schema_mode: StructuralTagSchemaMode::Auto,
            starts_in_reasoning: true,
        };
        let value = serde_json::to_value(build_kimi_k3(&ctx).unwrap().unwrap()).unwrap();

        assert_eq!(value["format"]["type"], "sequence");
        assert_eq!(value["format"]["elements"][0]["type"], "tag");
        assert_eq!(value["format"]["elements"][0]["end"], THINK_CLOSE);
        assert_eq!(value["format"]["elements"][1]["type"], "sequence");
    }

    #[test]
    fn named_choice_enforces_required_and_optional_argument_schemas() {
        let tools = tools();
        let choice = ToolChoice::Named("get_weather".to_string());
        let value =
            serde_json::to_value(build_kimi_k3(&context(&choice, &tools)).unwrap().unwrap())
                .unwrap();
        let call = &value["format"]["elements"][2]["content"]["tags"][0];
        let arguments = &call["content"]["elements"][2];

        assert_eq!(call["content"]["elements"][0]["pattern"], "[1-9][0-9]*");
        assert_eq!(arguments["type"], "sequence");
        assert_eq!(
            arguments["elements"][0]["begin"],
            "<|open|>argument key=\"city\" type=\"string\"<|sep|>"
        );
        assert_eq!(arguments["elements"][1]["type"], "star");
        assert_eq!(
            arguments["elements"][1]["content"]["begin"],
            "<|open|>argument key=\"days\" type=\"number\"<|sep|>"
        );
    }

    #[test]
    fn explicit_non_strict_tool_uses_vllm_permissive_argument_shape() {
        let tools = vec![ToolDefinition {
            name: "get_weather".to_string(),
            parameters: Some(json!({
                "type": "object",
                "properties": {"city": {"type": "string"}}
            })),
            strict: Some(false),
        }];
        let choice = ToolChoice::Named("get_weather".to_string());
        let value =
            serde_json::to_value(build_kimi_k3(&context(&choice, &tools)).unwrap().unwrap())
                .unwrap();
        let call = &value["format"]["elements"][2]["content"]["tags"][0];
        let arguments = &call["content"]["elements"][2];

        assert_eq!(arguments["type"], "star");
        assert_eq!(arguments["content"]["begin"], "<|open|>argument ");
        assert!(
            !arguments.to_string().contains("key=\\\"city\\\""),
            "vLLM treats strict=false as a permissive argument schema"
        );
    }

    #[test]
    fn parallel_false_stops_after_the_first_k3_call() {
        let tools = tools();
        let choice = ToolChoice::Required;
        let ctx = ToolCallFormatBuildContext {
            tool_choice: &choice,
            tools: &tools,
            parallel_tool_calls: Some(false),
            schema_mode: StructuralTagSchemaMode::Auto,
            starts_in_reasoning: false,
        };
        let value = serde_json::to_value(build_kimi_k3(&ctx).unwrap().unwrap()).unwrap();

        assert_eq!(
            value["format"]["elements"][2]["content"]["stop_after_first"],
            true
        );
    }
}
