// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use percent_encoding::percent_decode_str;
use serde_json::{Map, Number, Value};
use uuid::Uuid;

use super::super::ToolDefinition;
use super::super::config::MiniMaxM3ParserConfig;
use super::parsed_value::{coerce_integer_literal, raw_number_literal};
use super::response::{CalledFunction, ToolCallResponse, ToolCallType};

// Main entry point: strips normal prefix text and turns M3 tool markup into tool-call responses.
pub fn try_tool_call_parse_minimax_m3(
    message: &str,
    config: &MiniMaxM3ParserConfig,
    tools: Option<&[ToolDefinition]>,
) -> anyhow::Result<(Vec<ToolCallResponse>, Option<String>)> {
    // `normal_text` is the model text with each complete tool-call block removed
    // (from `]<]minimax[>[<tool_call>` through `]<]minimax[>[</tool_call>`),
    // keeping the surrounding text verbatim: the prefix before the first block,
    // text BETWEEN blocks, and text AFTER the last block. Text INSIDE a block
    // that is not a complete invoke (narration between invokes, junk like batch
    // case 4.a) is part of the markup block and stays dropped, like the other
    // families. Malformed / unterminated blocks keep drop-without-leak: their
    // markup never reaches normal_text.
    let tool_call_start = tool_call_start(config);
    let tool_call_end = tool_call_end(config);
    let mut calls: Vec<ToolCallResponse> = Vec::new();
    let mut normal_parts: Vec<String> = Vec::new();
    let mut cursor = 0;

    while cursor <= message.len() {
        let Some(start_rel) = message[cursor..].find(tool_call_start.as_str()) else {
            // No more blocks: this gap is the prefix (no block at all), the text
            // after the last `</tool_call>`, or both. A bare invoke run in the
            // gap (missing `<tool_call>` opener — cases 5.b/5.g) is recovered;
            // otherwise a stray orphan marker is dropped without leaking, keeping
            // the text before it.
            push_gap(
                &message[cursor..],
                config,
                tools,
                &mut normal_parts,
                &mut calls,
            )?;
            break;
        };
        let abs_start = cursor + start_rel;
        push_gap(
            &message[cursor..abs_start],
            config,
            tools,
            &mut normal_parts,
            &mut calls,
        )?;

        let block_start = abs_start + tool_call_start.len();
        match message[block_start..].find(tool_call_end.as_str()) {
            Some(end_rel) => {
                calls.extend(parse_invokes(
                    &message[block_start..block_start + end_rel],
                    config,
                    tools,
                )?);
                cursor = block_start + end_rel + tool_call_end.len();
            }
            None => {
                // Unterminated block: parse it only under EOF recovery; either
                // way its markup tail never leaks into normal_text.
                if config.allow_eof_recovery {
                    calls.extend(parse_invokes(&message[block_start..], config, tools)?);
                }
                break;
            }
        }
    }

    let normal_text = normal_parts.join("");
    let normal_text = if calls.is_empty() {
        normal_text.trim().to_string()
    } else {
        normal_text
    };
    Ok((calls, Some(normal_text)))
}

/// Fold one between-block gap into `normal_parts` / `calls`: recover a bare
/// invoke run (keeping its prose prefix), else drop from a stray orphan marker
/// onward (drop-without-leak), else keep the gap text verbatim.
fn push_gap(
    gap: &str,
    config: &MiniMaxM3ParserConfig,
    tools: Option<&[ToolDefinition]>,
    normal_parts: &mut Vec<String>,
    calls: &mut Vec<ToolCallResponse>,
) -> anyhow::Result<()> {
    if gap.is_empty() {
        return Ok(());
    }
    if let Some((prefix, recovered)) = recover_orphan_invokes_in_span(gap, config, tools)? {
        normal_parts.push(prefix);
        calls.extend(recovered);
    } else if let Some(marker_idx) = first_orphan_minimax_m3_marker_index(gap, config) {
        normal_parts.push(gap[..marker_idx].trim_end().to_string());
    } else {
        normal_parts.push(gap.to_string());
    }
    Ok(())
}

// Builds the configured outer tool-call start marker.
fn tool_call_start(config: &MiniMaxM3ParserConfig) -> String {
    format!("{}<{}>", config.namespace_token, config.tool_call_tag)
}

// Builds the configured outer tool-call end marker.
fn tool_call_end(config: &MiniMaxM3ParserConfig) -> String {
    format!("{}</{}>", config.namespace_token, config.tool_call_tag)
}

// Builds the marker that introduces an individual function invocation before attributes.
fn invoke_start(config: &MiniMaxM3ParserConfig) -> String {
    format!("{}<invoke", config.namespace_token)
}

// Builds the marker that closes an individual function invocation.
fn invoke_end(config: &MiniMaxM3ParserConfig) -> String {
    format!("{}</invoke>", config.namespace_token)
}

// Builds the shared namespace prefix that starts any M3 XML-ish tag.
fn parameter_start(config: &MiniMaxM3ParserConfig) -> String {
    format!("{}<", config.namespace_token)
}

fn first_orphan_minimax_m3_marker_index(
    text: &str,
    config: &MiniMaxM3ParserConfig,
) -> Option<usize> {
    [
        tool_call_end(config),
        invoke_start(config),
        invoke_end(config),
        parameter_start(config),
    ]
    .iter()
    .filter_map(|marker| text.find(marker.as_str()))
    .min()
}

fn recover_orphan_invokes_in_span(
    span: &str,
    config: &MiniMaxM3ParserConfig,
    tools: Option<&[ToolDefinition]>,
) -> anyhow::Result<Option<(String, Vec<ToolCallResponse>)>> {
    let Some(marker_idx) = first_orphan_minimax_m3_marker_index(span, config) else {
        return Ok(None);
    };

    let marker_tail = &span[marker_idx..];
    if !marker_tail.starts_with(invoke_start(config).as_str()) {
        return Ok(None);
    }
    if !marker_tail.contains(tool_call_end(config).as_str()) && !config.allow_eof_recovery {
        return Ok(None);
    }

    let calls = parse_invokes(marker_tail, config, tools)?;
    if calls.is_empty() {
        return Ok(None);
    }

    Ok(Some((span[..marker_idx].trim_end().to_string(), calls)))
}

// Extracts one or more `<invoke name="...">` blocks from the outer tool-call block.
fn parse_invokes(
    block: &str,
    config: &MiniMaxM3ParserConfig,
    tools: Option<&[ToolDefinition]>,
) -> anyhow::Result<Vec<ToolCallResponse>> {
    let invoke_start = invoke_start(config);
    let invoke_end = invoke_end(config);
    let mut calls = Vec::new();
    let mut cursor = 0;

    while let Some(start_rel) = block[cursor..].find(invoke_start.as_str()) {
        let tag_attrs_start = cursor + start_rel + invoke_start.len();
        let Some(tag_end_rel) = block[tag_attrs_start..].find('>') else {
            break;
        };
        let tag_attrs = &block[tag_attrs_start..tag_attrs_start + tag_end_rel];
        let function_name = parse_invoke_name(tag_attrs);
        let body_start = tag_attrs_start + tag_end_rel + 1;
        let Some(body_end_rel) = block[body_start..].find(invoke_end.as_str()) else {
            break;
        };
        let body_end = body_start + body_end_rel;
        let function_body = &block[body_start..body_end];

        if let Some(function_name) = function_name
            && !function_name.is_empty()
        {
            let arguments = parse_parameters(&function_name, function_body, config, tools)?;
            calls.push(ToolCallResponse {
                id: format!("call-{}", Uuid::new_v4()),
                tp: ToolCallType::Function,
                function: CalledFunction {
                    name: function_name,
                    arguments: serde_json::to_string(&Value::Object(arguments))?,
                },
            });
        }

        cursor = body_end + invoke_end.len();
    }

    Ok(calls)
}

// Reads the `name` attribute from `<invoke ...>` using vLLM-compatible quoting variants.
fn parse_invoke_name(tag_attrs: &str) -> Option<String> {
    let attrs = tag_attrs.trim_start();
    let after_name = attrs.strip_prefix("name")?.trim_start();
    let value = after_name.strip_prefix('=')?.trim_start();

    if let Some(value) = value.strip_prefix('"') {
        return value.find('"').map(|end| value[..end].trim().to_string());
    }
    if let Some(value) = value.strip_prefix('\'') {
        return value.find('\'').map(|end| value[..end].trim().to_string());
    }

    let end = value.find(char::is_whitespace).unwrap_or(value.len());
    if end == 0 {
        None
    } else {
        Some(value[..end].trim().to_string())
    }
}

// Extracts MiniMax M3 parameter tags, where each parameter name is the tag name itself.
fn parse_parameters(
    function_name: &str,
    body: &str,
    config: &MiniMaxM3ParserConfig,
    tools: Option<&[ToolDefinition]>,
) -> anyhow::Result<Map<String, Value>> {
    let parameter_start = parameter_start(config);
    let root_schema = get_arguments_config(function_name, tools).unwrap_or(&Value::Null);
    let param_config = root_schema
        .get("properties")
        .filter(|value| value.is_object())
        .unwrap_or(root_schema);
    let mut parameters = Map::new();
    let mut cursor = 0;

    while let Some(start_rel) = body[cursor..].find(parameter_start.as_str()) {
        let start = cursor + start_rel + parameter_start.len();
        if body[start..].starts_with('/') {
            cursor = start + 1;
            continue;
        }

        let Some(name_end_rel) = body[start..].find('>') else {
            break;
        };
        let parameter_name = &body[start..start + name_end_rel];
        if parameter_name.is_empty() || parameter_name.contains(char::is_whitespace) {
            cursor = start + name_end_rel + 1;
            continue;
        }

        let value_start = start + name_end_rel + 1;
        let parameter_end = format!("{}</{}>", config.namespace_token, parameter_name);
        let Some(value_end_rel) = body[value_start..].find(parameter_end.as_str()) else {
            break;
        };
        let value_end = value_start + value_end_rel;
        let raw_value = &body[value_start..value_end];
        let schema = param_config.get(parameter_name);
        let value = parse_parameter_value(raw_value, schema, root_schema, config);
        insert_parameter(&mut parameters, parameter_name.to_string(), value);

        cursor = value_end + parameter_end.len();
    }

    Ok(parameters)
}

// Preserves duplicate XML tags by collecting repeated values into arrays.
fn insert_parameter(parameters: &mut Map<String, Value>, key: String, value: Value) {
    if let Some(existing) = parameters.remove(&key) {
        let merged = match existing {
            Value::Array(mut values) => {
                values.push(value);
                Value::Array(values)
            }
            existing => Value::Array(vec![existing, value]),
        };
        parameters.insert(key, merged);
    } else {
        parameters.insert(key, value);
    }
}

// Chooses scalar conversion or nested XML parsing based on whether the value contains M3 tags.
fn parse_parameter_value(
    raw: &str,
    schema: Option<&Value>,
    root_schema: &Value,
    config: &MiniMaxM3ParserConfig,
) -> Value {
    if raw.contains(parameter_start(config).as_str()) {
        parse_nested_minimax_xml(raw, schema, root_schema, config)
    } else {
        convert_scalar_value(raw, schema, root_schema)
    }
}

// Parses nested parameter bodies such as arrays of `<item>` objects into JSON values.
fn parse_nested_minimax_xml(
    raw: &str,
    schema: Option<&Value>,
    root_schema: &Value,
    config: &MiniMaxM3ParserConfig,
) -> Value {
    let chunks: Vec<&str> = raw.split(config.namespace_token.as_str()).collect();
    let leading_text = chunks.first().copied().unwrap_or_default();
    let root_value = if SchemaWalker::new(root_schema).has_type(schema, "array")
        && chunks
            .get(1)
            .is_some_and(|chunk| chunk.starts_with("<item>"))
    {
        Some(StackValue::Array(Vec::new()))
    } else {
        Some(StackValue::Object(Map::new()))
    };
    let mut stack = vec![StackItem {
        tag: None,
        value: root_value,
        // Whitespace-only leading text is pretty-print formatting, not a value:
        // treat it as empty so it never becomes a spurious `$text` / array item.
        texts: if leading_text.trim().is_empty() {
            Vec::new()
        } else {
            vec![leading_text.to_string()]
        },
        schema,
        root_schema,
    }];

    for (chunk_index, chunk) in chunks.iter().enumerate().skip(1) {
        if chunk.starts_with("</") {
            let (tag, trailing_text) = split_end_tag_chunk(chunk);
            while stack.len() > 1 {
                let item = stack.pop().expect("stack has child item");
                let matched = item.tag.as_deref() == Some(tag.as_str());
                stack
                    .last_mut()
                    .expect("stack has parent item")
                    .append(item);
                if matched {
                    break;
                }
            }
            // Skip whitespace-only formatting between tags; otherwise it would
            // append a spurious array element (or `$text`) to the parent node.
            if !trailing_text.trim().is_empty() {
                stack
                    .last_mut()
                    .expect("stack has current item")
                    .append_text(trailing_text);
            }
        } else if chunk.starts_with('<') {
            let (tag, trailing_text) = split_start_tag_chunk(chunk);
            if tag.is_empty() {
                continue;
            }
            let child_schema = stack
                .last()
                .expect("stack has current item")
                .schema_for_child(tag.as_str());
            let child_value = if SchemaWalker::new(root_schema).has_type(child_schema, "array")
                && chunks
                    .get(chunk_index + 1)
                    .is_some_and(|next| next.starts_with("<item>"))
            {
                Some(StackValue::Array(Vec::new()))
            } else {
                None
            };
            stack.push(StackItem {
                tag: Some(tag),
                value: child_value,
                texts: if trailing_text.trim().is_empty() {
                    Vec::new()
                } else {
                    vec![trailing_text.to_string()]
                },
                schema: child_schema,
                root_schema,
            });
        } else if !chunk.trim().is_empty() {
            stack
                .last_mut()
                .expect("stack has current item")
                .append_text(chunk);
        }
    }

    while stack.len() > 1 {
        let item = stack.pop().expect("stack has child item");
        stack
            .last_mut()
            .expect("stack has parent item")
            .append(item);
    }

    stack.pop().expect("root item exists").into_value()
}

// Splits a start-tag chunk into its tag name and any text after `>`.
fn split_start_tag_chunk(chunk: &str) -> (String, &str) {
    let Some(gt) = chunk.find('>') else {
        return (chunk.trim_start_matches('<').to_string(), "");
    };
    (chunk[1..gt].to_string(), &chunk[gt + 1..])
}

// Splits an end-tag chunk into its tag name and any text after `>`.
fn split_end_tag_chunk(chunk: &str) -> (String, &str) {
    let Some(gt) = chunk.find('>') else {
        return (chunk.trim_start_matches("</").to_string(), "");
    };
    (chunk[2..gt].to_string(), &chunk[gt + 1..])
}

#[derive(Debug)]
enum StackValue {
    Object(Map<String, Value>),
    Array(Vec<Value>),
}

#[derive(Debug)]
struct StackItem<'a> {
    tag: Option<String>,
    value: Option<StackValue>,
    texts: Vec<String>,
    schema: Option<&'a Value>,
    root_schema: &'a Value,
}

impl<'a> StackItem<'a> {
    // Converts a stack node into the JSON value it represents.
    fn into_value(self) -> Value {
        match self.value {
            None => {
                convert_scalar_value(self.texts.join("").as_str(), self.schema, self.root_schema)
            }
            Some(StackValue::Object(mut map)) => {
                if !self.texts.is_empty() {
                    let mut text_key = "$text".to_string();
                    while map.contains_key(&text_key) {
                        text_key = format!("${text_key}");
                    }
                    map.insert(text_key, Value::String(self.texts.join("")));
                }
                Value::Object(map)
            }
            Some(StackValue::Array(values)) => Value::Array(values),
        }
    }

    // Attaches a completed child node to the current object, array, or implicit object.
    fn append(&mut self, item: StackItem) {
        let key = item.tag.clone().unwrap_or_default();
        let value = item.into_value();
        match self.value.as_mut() {
            None => {
                let mut map = Map::new();
                map.insert(key, value);
                self.value = Some(StackValue::Object(map));
            }
            Some(StackValue::Object(map)) => insert_parameter(map, key, value),
            Some(StackValue::Array(values)) => values.push(value),
        }
    }

    // Adds text to the current node, coercing array items through item schema when available.
    fn append_text(&mut self, text: &str) {
        if let Some(StackValue::Array(values)) = self.value.as_mut() {
            let item_schema = SchemaWalker::new(self.root_schema).array_item(self.schema);
            values.push(convert_scalar_value(text, item_schema, self.root_schema));
        } else {
            self.texts.push(text.to_string());
        }
    }

    // Finds the schema that should be used for a nested child tag.
    fn schema_for_child(&self, tag: &str) -> Option<&'a Value> {
        let mut schemas = SchemaWalker::new(self.root_schema);
        if tag == "item"
            && let Some(item_schema) = schemas.array_item(self.schema)
        {
            return Some(item_schema);
        }

        self.schema
            .and_then(|schema| schemas.object_child(schema, tag))
    }
}

// Looks up the selected tool's parameter schema so parsed strings can be type-coerced.
fn get_arguments_config<'a>(
    func_name: &str,
    tools: Option<&'a [ToolDefinition]>,
) -> Option<&'a Value> {
    let tools = tools?;
    if let Some(tool) = tools.iter().find(|tool| tool.name == func_name) {
        return tool.parameters.as_ref();
    }
    tracing::warn!("Tool '{}' is not defined in the tools list.", func_name);
    None
}

// Converts a scalar XML text value into the schema-expected JSON type when possible.
fn convert_scalar_value(raw: &str, schema: Option<&Value>, root_schema: &Value) -> Value {
    let mut schemas = SchemaWalker::new(root_schema);
    let value = html_unescape(raw);
    let trimmed = value.trim();

    // Without a schema we cannot know the intended type, so preserve the literal
    // text (including the string "null") instead of inventing a JSON null.
    let Some(schema) = schema else {
        return Value::String(value);
    };

    // Only collapse the literal "null" into JSON null when the schema actually
    // permits null. A `string`-typed parameter keeps the literal value "null".
    if trimmed.eq_ignore_ascii_case("null") && schemas.permits_null(schema) {
        return Value::Null;
    }

    if schemas.has_type(Some(schema), "string") || schemas.has_type(Some(schema), "enum") {
        return Value::String(value);
    }
    if schemas.has_type(Some(schema), "integer") {
        return coerce_integer_literal(trimmed)
            .and_then(|parsed| serde_json::to_value(parsed).ok())
            .unwrap_or(Value::String(value));
    }
    if schemas.has_type(Some(schema), "number") {
        if let Some(parsed) = coerce_integer_literal(trimmed)
            && let Ok(json) = serde_json::to_value(parsed)
        {
            return json;
        }
        if let Ok(number) = trimmed.parse::<f64>()
            && let Some(number) = Number::from_f64(number)
        {
            return Value::Number(number);
        }
        if let Some(parsed) = raw_number_literal(trimmed)
            && let Ok(json) = serde_json::to_value(parsed)
        {
            return json;
        }
        return Value::String(value);
    }
    if schemas.has_type(Some(schema), "boolean") {
        return match trimmed.to_ascii_lowercase().as_str() {
            "true" => Value::Bool(true),
            "1" => Value::Bool(true),
            "false" => Value::Bool(false),
            "0" => Value::Bool(false),
            _ => Value::String(value),
        };
    }
    if schemas.has_type(Some(schema), "object") {
        if trimmed.is_empty() {
            return Value::Object(Map::new());
        }
        if let Ok(json) = serde_json::from_str::<Value>(trimmed) {
            return json;
        }
    }
    if schemas.has_type(Some(schema), "array") {
        if trimmed.is_empty() {
            return Value::Array(Vec::new());
        }
        if let Ok(json) = serde_json::from_str::<Value>(trimmed) {
            return json;
        }
    }

    Value::String(value)
}

// Bound graph traversal as well as recursion: shared references can expand
// exponentially even when there are no cycles. Exhaustion preserves untyped output.
const MAX_SCHEMA_WORK: usize = 1024;
const MAX_SCHEMA_DEPTH: usize = 64;

// Each lookup tracks its own schema path: recursive schemas may be revisited
// after consuming another XML child, but reference/composition cycles cannot loop.
struct SchemaWalker<'a> {
    root: &'a Value,
    path: Vec<&'a Value>,
    remaining_work: usize,
    exhausted: bool,
}

impl<'a> SchemaWalker<'a> {
    fn new(root: &'a Value) -> Self {
        Self {
            root,
            path: Vec::new(),
            remaining_work: MAX_SCHEMA_WORK,
            exhausted: false,
        }
    }

    fn spend_work(&mut self) -> bool {
        if self.exhausted || self.remaining_work == 0 {
            self.exhausted = true;
            return false;
        }
        self.remaining_work -= 1;
        true
    }

    // Leave unknown references and cycles untouched; do not discard sibling constraints.
    fn resolve_ref(&mut self, schema: &'a Value) -> Option<&'a Value> {
        let mut current = schema;
        let mut visited = Vec::new();
        while let Some(reference) = current.get("$ref").and_then(Value::as_str) {
            if !self.spend_work() {
                return None;
            }
            if current.as_object().is_some_and(|object| {
                object.keys().any(|key| {
                    !matches!(
                        key.as_str(),
                        "$ref" | "title" | "description" | "default" | "examples" | "$comment"
                    )
                })
            }) || visited.contains(&reference)
            {
                return Some(schema);
            }
            let Some(fragment) = reference.strip_prefix('#') else {
                return Some(schema);
            };
            // percent_decode_str preserves invalid escapes, so reject them before lookup.
            if fragment.as_bytes().iter().enumerate().any(|(index, byte)| {
                *byte == b'%'
                    && !fragment
                        .as_bytes()
                        .get(index + 1..index + 3)
                        .is_some_and(|digits| digits.iter().all(u8::is_ascii_hexdigit))
            }) {
                return Some(schema);
            }
            let Ok(pointer) = percent_decode_str(fragment).decode_utf8() else {
                return Some(schema);
            };
            let Some(target) = self.root.pointer(&pointer) else {
                return Some(schema);
            };
            visited.push(reference);
            current = target;
        }
        Some(current)
    }

    fn with_schema<T: Copy>(
        &mut self,
        schema: &'a Value,
        fallback: T,
        query: impl FnOnce(&mut Self, &'a Value) -> T,
    ) -> T {
        if !self.spend_work() || self.path.len() >= MAX_SCHEMA_DEPTH {
            self.exhausted = true;
            return fallback;
        }
        let Some(schema) = self.resolve_ref(schema) else {
            return fallback;
        };
        if self.path.iter().any(|seen| std::ptr::eq(*seen, schema)) {
            return fallback;
        }
        self.path.push(schema);
        let result = query(self, schema);
        self.path.pop();
        if self.exhausted { fallback } else { result }
    }

    // Unknown constraints remain possible, preserving object-union ambiguity.
    fn may_describe_object(&mut self, schema: &'a Value) -> bool {
        self.with_schema(schema, true, |walker, schema| {
            if schema == &Value::Bool(false) {
                return false;
            }
            if let Some(ty) = schema.get("type") {
                let object = ty.as_str() == Some("object")
                    || ty
                        .as_array()
                        .is_some_and(|types| types.iter().any(|ty| ty == "object"));
                if !object {
                    return false;
                }
            }
            if schema.get("const").is_some_and(|value| !value.is_object())
                || schema
                    .get("enum")
                    .and_then(Value::as_array)
                    .is_some_and(|values| !values.iter().any(Value::is_object))
            {
                return false;
            }
            for keyword in ["allOf", "anyOf", "oneOf"] {
                if let Some(branches) = schema.get(keyword).and_then(Value::as_array) {
                    let possible = if keyword == "allOf" {
                        branches
                            .iter()
                            .all(|branch| walker.may_describe_object(branch))
                    } else {
                        branches
                            .iter()
                            .any(|branch| walker.may_describe_object(branch))
                    };
                    if !possible {
                        return false;
                    }
                }
            }
            true
        })
    }

    // Nested XML identifies an object, but does not choose among object variants.
    fn object_child(&mut self, schema: &'a Value, tag: &str) -> Option<&'a Value> {
        self.with_schema(schema, None, |walker, schema| {
            if let Some(child) = schema.get("properties").and_then(|props| props.get(tag)) {
                return Some(child);
            }
            if let Some(additional) = schema
                .get("additionalProperties")
                .filter(|value| value.is_object())
            {
                return Some(additional);
            }
            let branches = match (schema.get("anyOf"), schema.get("oneOf")) {
                (Some(branches), None) | (None, Some(branches)) => branches.as_array()?,
                _ => return None,
            };
            let mut objects = branches
                .iter()
                .filter(|branch| walker.may_describe_object(branch));
            let object = objects.next()?;
            if objects.next().is_some() {
                return None;
            }
            walker.object_child(object, tag)
        })
    }

    fn permits_null(&mut self, schema: &'a Value) -> bool {
        let Some(schema) = self.resolve_ref(schema) else {
            return false;
        };
        let permitted = self.has_type(Some(schema), "null")
            || schema.get("nullable").and_then(Value::as_bool) == Some(true)
            || (schema.get("type").is_none()
                && schema.get("anyOf").is_none()
                && schema.get("oneOf").is_none());
        permitted && !self.exhausted
    }

    fn has_type(&mut self, schema: Option<&'a Value>, expected: &str) -> bool {
        let Some(schema) = schema else {
            return false;
        };
        self.with_schema(schema, false, |walker, schema| {
            if let Some(ty) = schema.get("type")
                && (ty.as_str() == Some(expected)
                    || ty
                        .as_array()
                        .is_some_and(|types| types.iter().any(|ty| ty.as_str() == Some(expected))))
            {
                return true;
            }
            for key in ["anyOf", "oneOf"] {
                if let Some(options) = schema.get(key).and_then(Value::as_array)
                    && options
                        .iter()
                        .any(|option| walker.has_type(Some(option), expected))
                {
                    return true;
                }
            }
            false
        })
    }

    fn array_item(&mut self, schema: Option<&'a Value>) -> Option<&'a Value> {
        self.with_schema(schema?, None, |walker, schema| {
            if let Some(items) = schema.get("items") {
                return Some(items);
            }
            for key in ["anyOf", "oneOf"] {
                if let Some(options) = schema.get(key).and_then(Value::as_array) {
                    for option in options {
                        if let Some(items) = walker.array_item(Some(option)) {
                            return Some(items);
                        }
                    }
                }
            }
            None
        })
    }
}

// Decodes common XML/HTML entities so tool arguments receive the intended literal text.
fn html_unescape(s: &str) -> String {
    s.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
        .replace("&quot;", "\"")
        .replace("&#x27;", "'")
        .replace("&#39;", "'")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // Namespace token emitted before every M3 tag; keeps the test inputs readable.
    const TOK: &str = "]<]minimax[>[";

    // Finding 1: pretty-printed nested arguments must not turn inter-tag
    // formatting whitespace into a spurious `$text` property or array element.
    #[test]
    fn pretty_printed_nested_object_has_no_spurious_whitespace_text() {
        let config = MiniMaxM3ParserConfig::default();
        // Leading whitespace before the first tag and newlines/indent between the
        // sibling tags, exactly as a model would emit when pretty-printing.
        let raw = format!("\n  {TOK}<a>1{TOK}</a>\n  {TOK}<b>2{TOK}</b>\n");
        let parsed = parse_nested_minimax_xml(&raw, None, &Value::Null, &config);
        assert_eq!(parsed, json!({ "a": "1", "b": "2" }));
        // No `$text` (or `$$text`) key should have been synthesized from whitespace.
        let obj = parsed.as_object().expect("object value");
        assert!(
            obj.keys().all(|k| !k.contains("$text")),
            "unexpected whitespace $text key in {parsed}"
        );
    }

    #[test]
    fn pretty_printed_nested_array_has_no_spurious_whitespace_item() {
        let config = MiniMaxM3ParserConfig::default();
        let schema = json!({ "type": "array", "items": { "type": "string" } });
        // Whitespace between `</item>` and the next `<item>` previously became an
        // extra array element via the end-tag trailing-text append.
        let raw = format!("\n  {TOK}<item>a{TOK}</item>\n  {TOK}<item>b{TOK}</item>\n");
        let parsed = parse_nested_minimax_xml(&raw, Some(&schema), &schema, &config);
        assert_eq!(parsed, json!(["a", "b"]));
    }

    // Finding 2: honor the schema before coercing the literal string "null".
    #[test]
    fn string_typed_null_stays_a_string() {
        let schema = json!({ "type": "string" });
        assert_eq!(
            convert_scalar_value("null", Some(&schema), &schema),
            json!("null"),
            "a string-typed parameter must keep the literal value \"null\""
        );
    }

    #[test]
    fn nullable_typed_null_becomes_json_null() {
        // Explicit null type, a `["string", "null"]` union, and `nullable: true`
        // all permit null and should coerce.
        for schema in [
            json!({ "type": "null" }),
            json!({ "type": ["string", "null"] }),
            json!({ "type": "string", "nullable": true }),
        ] {
            assert_eq!(
                convert_scalar_value("null", Some(&schema), &schema),
                Value::Null,
                "nullable schema {schema} should coerce \"null\" to JSON null"
            );
        }
    }

    #[test]
    fn schemaless_null_stays_a_string() {
        // With no schema the intended type is unknown, so the literal is preserved.
        assert_eq!(
            convert_scalar_value("null", None, &Value::Null),
            json!("null")
        );
    }

    #[test]
    fn nested_union_child_values_preserve_scalar_and_object_shapes() {
        let tok = "]<]minimax[>[";
        let config = MiniMaxM3ParserConfig::default();
        for union in ["anyOf", "oneOf"] {
            let schema = serde_json::json!({union: [
                {"type": "object", "properties": {
                    "mode": {"anyOf": [{"type": "string"}, {"type": "object"}]},
                    "after": {"type": ["object", "null"]},
                    "config": {"type": "object", "properties": {"enabled": {"type": "boolean"}}}
                }},
                {"type": "null"}
            ]});
            let raw = format!(
                "{tok}<mode>one{tok}</mode>{tok}<after>null{tok}</after>\
                 {tok}<config>{tok}<enabled>true{tok}</enabled>{tok}</config>"
            );
            assert_eq!(
                parse_nested_minimax_xml(&raw, Some(&schema), &schema, &config),
                serde_json::json!({"mode": "one", "after": null, "config": {"enabled": true}}),
                "{union}"
            );
        }
    }

    #[test]
    fn nested_union_object_types_and_ambiguity() {
        let tok = "]<]minimax[>[";
        let config = MiniMaxM3ParserConfig::default();
        // MOD2-167: nullable pagination and the object branch of the stress schema.
        for union in ["anyOf", "oneOf"] {
            let schema = serde_json::json!({union: [
                {"type":"object","properties":{"page":{"type":"integer","minimum":1},"per_page":{"type":"integer","minimum":1,"maximum":100}}},
                {"type":"null"}
            ]});
            let raw = format!("{tok}<page>2{tok}</page>{tok}<per_page>25{tok}</per_page>");
            assert_eq!(
                parse_nested_minimax_xml(&raw, Some(&schema), &schema, &config),
                serde_json::json!({"page":2,"per_page":25})
            );
            let object = serde_json::json!({"type":"object","properties":{"enabled":{"type":"boolean"},"mode":{"type":"string","enum":["one","two","three","four"]}},"required":["enabled"]});
            let schema = serde_json::json!({union:[{"type":"string"},{"type":"array","items":{"type":"string"},"maxItems":20},object]});
            let raw = format!("{tok}<enabled>true{tok}</enabled>{tok}<mode>one{tok}</mode>");
            assert_eq!(
                parse_nested_minimax_xml(&raw, Some(&schema), &schema, &config),
                serde_json::json!({"enabled":true,"mode":"one"})
            );
            // Literal-only and composed non-object branches cannot make the object ambiguous.
            for alternative in [
                serde_json::json!({"enum": [null]}),
                serde_json::json!({"const": null}),
                serde_json::json!({"enum": [null, "text", 2, []]}),
                serde_json::json!({"type": ["object", "null"], "const": null}),
                serde_json::json!({"allOf": [{"enum": [null]}, {}]}),
                serde_json::json!({"anyOf": [{"const": null}, {"type": "string"}]}),
                serde_json::json!(false),
            ] {
                let schema = serde_json::json!({union: [
                    {"type":"object","properties":{"page":{"type":"integer"}}}, alternative
                ]});
                let raw = format!("{tok}<page>2{tok}</page>");
                assert_eq!(
                    parse_nested_minimax_xml(&raw, Some(&schema), &schema, &config),
                    serde_json::json!({"page":2}),
                    "{union}: {alternative}"
                );
            }
            for alternative in [
                serde_json::json!({"type":"object"}),
                serde_json::json!({"enum":[null, {"value":"2"}]}),
                serde_json::json!({"const":{"value":"2"}}),
                serde_json::json!(true),
                serde_json::json!({}),
                serde_json::json!({"type":"object","properties":{"value":{"type":"string"}}}),
            ] {
                let schema = serde_json::json!({union:[{"type":"object","properties":{"value":{"type":"integer"}}}, alternative]});
                let raw = format!("{tok}<value>2{tok}</value>");
                assert_eq!(
                    parse_nested_minimax_xml(&raw, Some(&schema), &schema, &config),
                    serde_json::json!({"value":"2"})
                );
            }
        }
    }

    #[test]
    fn local_ref_object_argument_accepts_json_text() {
        let parameters = serde_json::json!({
            "$defs": {"Payload": {"type": "object"}},
            "properties": {"data": {"$ref": "#/$defs/Payload"}}
        });
        let tools = vec![ToolDefinition {
            name: "capture".into(),
            parameters: Some(parameters),
        }];
        let raw = "]<]minimax[>[<data>{\"input\":\"Alex\"}]<]minimax[>[</data>";
        let actual = parse_parameters(
            "capture",
            raw,
            &MiniMaxM3ParserConfig::default(),
            Some(&tools),
        )
        .unwrap();
        assert_eq!(
            Value::Object(actual),
            serde_json::json!({"data":{"input":"Alex"}})
        );
    }

    #[test]
    fn parameter_refs_handle_chains_escaped_names_and_unknown_targets() {
        let root = serde_json::json!({"$defs":{
            "alias":{"$ref":"#/$defs/a~1b~0c"},
            "a/b~c":{"type":"object"},
            "cycle":{"$ref":"#/$defs/cycle"},
            "cycle_a":{"$ref":"#/$defs/cycle_b"},
            "cycle_b":{"$ref":"#/$defs/cycle_a"},
            "broken":{"$ref":"#/$defs/missing"},
            "bad%":{"type":"integer"},
            "bad%0":{"type":"integer"},
            "bad%GG":{"type":"integer"}
        }});
        let chain = serde_json::json!({"$ref":"#/$defs/alias","description":"value"});
        assert_eq!(
            SchemaWalker::new(&root).resolve_ref(&chain).unwrap(),
            &serde_json::json!({"type":"object"})
        );
        for schema in [
            serde_json::json!({"$ref":"#/$defs/missing"}),
            serde_json::json!({"$ref":"https://example.test/schema"}),
            serde_json::json!({"$ref":"#/$defs/cycle"}),
            serde_json::json!({"$ref":"#/$defs/cycle_a"}),
            serde_json::json!({"$ref":"#/$defs/broken"}),
            serde_json::json!({"$ref":"#/$defs/bad%"}),
            serde_json::json!({"$ref":"#/$defs/bad%0"}),
            serde_json::json!({"$ref":"#/$defs/bad%GG"}),
            serde_json::json!({"$ref":"#/$defs/%FF"}),
            serde_json::json!({"$ref":"#/$defs/alias","type":"string"}),
        ] {
            assert_eq!(
                SchemaWalker::new(&root).resolve_ref(&schema).unwrap(),
                &schema
            );
        }
    }
    #[test]
    fn nested_parameter_refs_preserve_types() {
        let parameters = serde_json::json!({
            "$defs": {
                "Count": {"type": "integer"},
                "Enabled": {"type": "boolean"},
                "Counts": {"type": "array", "items": {"$ref": "#/$defs/Count"}},
                "Options": {"type": "object", "properties": {
                    "count": {"$ref": "#/$defs/Count"},
                    "enabled": {"$ref": "#/$defs/Enabled"},
                    "counts": {"$ref": "#/$defs/Counts"}
                }}
            },
            "properties": {"options": {"$ref": "#/$defs/Options"}}
        });
        let tools = vec![ToolDefinition {
            name: "capture".into(),
            parameters: Some(parameters),
        }];
        let tok = "]<]minimax[>[";
        let raw = format!(
            "{tok}<options>{tok}<count>2{tok}</count>{tok}<enabled>true{tok}</enabled>\
             {tok}<counts>{tok}<item>3{tok}</item>{tok}</counts>{tok}</options>"
        );
        let actual = parse_parameters(
            "capture",
            &raw,
            &MiniMaxM3ParserConfig::default(),
            Some(&tools),
        )
        .unwrap();
        assert_eq!(
            Value::Object(actual),
            serde_json::json!({
                "options": {"count": 2, "enabled": true, "counts": [3]}
            })
        );
    }

    #[test]
    fn parameter_refs_decode_uri_fragments_once() {
        let root = serde_json::json!({"$defs": {
            "postal code": {"type": "integer"},
            "a/b~c": {"type": "boolean"},
            "café": {"type": "string"},
            "percent%20name": {"type": "array"}
        }});
        for (reference, expected) in [
            ("#/$defs/postal%20code", "integer"),
            ("#/$defs/a%7E1b%7e0c", "boolean"),
            ("#/$defs/caf%C3%A9", "string"),
            ("#/$defs/percent%2520name", "array"),
        ] {
            let schema = serde_json::json!({"$ref": reference});
            assert_eq!(
                SchemaWalker::new(&root).resolve_ref(&schema).unwrap(),
                &serde_json::json!({"type": expected}),
                "{reference}"
            );
        }
    }
    #[test]
    fn parameter_refs_follow_array_additional_property_and_union_schemas() {
        let definitions = serde_json::json!({
            "Count": {"type": "integer"},
            "Numbers": {"type": "array", "items": {"$ref": "#/$defs/Count"}},
            "Page": {"type": "object", "properties": {"count": {"$ref": "#/$defs/Count"}}},
            "Nothing": {"const": null}
        });
        let tok = "]<]minimax[>[";
        let count = format!("{tok}<count>2{tok}</count>");
        for (schema, body, expected) in [
            (
                serde_json::json!({"type": "object", "additionalProperties": {"$ref": "#/$defs/Count"}}),
                count.clone(),
                serde_json::json!({"count": 2}),
            ),
            (
                serde_json::json!({"type": "array", "items": {"$ref": "#/$defs/Page"}}),
                format!("{tok}<item>{count}{tok}</item>"),
                serde_json::json!([{"count": 2}]),
            ),
            (
                serde_json::json!({"anyOf": [{"$ref": "#/$defs/Page"}, {"$ref": "#/$defs/Nothing"}]}),
                count.clone(),
                serde_json::json!({"count": 2}),
            ),
            (
                serde_json::json!({"oneOf": [{"$ref": "#/$defs/Page"}, {"$ref": "#/$defs/Nothing"}]}),
                count.clone(),
                serde_json::json!({"count": 2}),
            ),
            (
                serde_json::json!({"anyOf": [{"$ref": "#/$defs/Page"}, {"type": "object"}]}),
                count,
                serde_json::json!({"count": "2"}),
            ),
            (
                serde_json::json!({"anyOf": [{"$ref": "#/$defs/Numbers"}, {"type": "null"}]}),
                format!("{tok}<item>2{tok}</item>"),
                serde_json::json!([2]),
            ),
            (
                serde_json::json!({"oneOf": [{"$ref": "#/$defs/Count"}, {"type": "null"}]}),
                "2".into(),
                serde_json::json!(2),
            ),
            (
                serde_json::json!({"$ref": "#/$defs/Count", "type": "string"}),
                "2".into(),
                serde_json::json!("2"),
            ),
        ] {
            let tools = vec![ToolDefinition {
                name: "capture".into(),
                parameters: Some(
                    serde_json::json!({"$defs": definitions, "properties": {"value": schema}}),
                ),
            }];
            let raw = format!("{tok}<value>{body}{tok}</value>");
            let actual = parse_parameters(
                "capture",
                &raw,
                &MiniMaxM3ParserConfig::default(),
                Some(&tools),
            )
            .unwrap();
            assert_eq!(actual["value"], expected, "{schema}");
        }
    }

    #[test]
    fn parameter_refs_allow_finite_recursive_values_and_stop_composition_cycles() {
        let tok = "]<]minimax[>[";
        let parameters = serde_json::json!({
            "$defs": {
                "Node": {"type": "object", "properties": {
                    "count": {"type": "integer"},
                    "child": {"anyOf": [{"$ref": "#/$defs/Node"}, {"type": "null"}]}
                }},
                "Loop": {"anyOf": [{"$ref": "#/$defs/Loop"}]}
            },
            "properties": {"tree": {"$ref": "#/$defs/Node"}, "loop": {"$ref": "#/$defs/Loop"}}
        });
        let tools = vec![ToolDefinition {
            name: "capture".into(),
            parameters: Some(parameters),
        }];
        let raw = format!(
            "{tok}<tree>{tok}<count>1{tok}</count>{tok}<child>{tok}<count>2{tok}</count>\
             {tok}<child>{tok}<count>3{tok}</count>{tok}</child>{tok}</child>{tok}</tree>\
             {tok}<loop>{tok}<count>4{tok}</count>{tok}</loop>"
        );
        let actual = parse_parameters(
            "capture",
            &raw,
            &MiniMaxM3ParserConfig::default(),
            Some(&tools),
        )
        .unwrap();
        assert_eq!(
            Value::Object(actual),
            serde_json::json!({
                "tree": {"count": 1, "child": {"count": 2, "child": {"count": 3}}},
                "loop": {"count": "4"}
            })
        );
        // A composed cycle may also occur while coercing a scalar, without any XML descent.
        for keyword in ["allOf", "anyOf", "oneOf", "not"] {
            let recursive = serde_json::json!({"$ref": "#/$defs/Loop"});
            let cycle = if keyword == "not" {
                recursive
            } else {
                serde_json::json!([recursive])
            };
            let root = serde_json::json!({"$defs": {"Loop": {keyword: cycle}}});
            let schema = serde_json::json!({"$ref": "#/$defs/Loop"});
            assert_eq!(
                convert_scalar_value("2", Some(&schema), &root),
                serde_json::json!("2")
            );
            let _ = convert_scalar_value("null", Some(&schema), &root);
        }
    }
    #[test]
    fn parameter_ref_expansion_is_bounded_and_preserves_uncertain_types() {
        let mut definitions = Map::new();
        definitions.insert("n0".into(), serde_json::json!({"type": "integer"}));
        for index in 1..=20 {
            let reference = serde_json::json!({"$ref": format!("#/$defs/n{}", index - 1)});
            definitions.insert(
                format!("n{index}"),
                serde_json::json!({"anyOf": [reference, reference]}),
            );
        }
        let root = serde_json::json!({"$defs": definitions});
        let schema = serde_json::json!({"$ref": "#/$defs/n20"});
        let mut walker = SchemaWalker::new(&root);
        assert!(!walker.has_type(Some(&schema), "string"));
        assert!(walker.exhausted);
        assert_eq!(walker.remaining_work, 0);
        // An inconclusive string lookup must not permit a later integer coercion.
        assert!(!walker.has_type(Some(&schema), "integer"));
        assert!(walker.may_describe_object(&schema));
        assert!(walker.object_child(&schema, "count").is_none());
        assert_eq!(
            convert_scalar_value("2", Some(&schema), &root),
            serde_json::json!("2")
        );
        assert_eq!(
            convert_scalar_value("null", Some(&schema), &root),
            serde_json::json!("null")
        );
    }

    #[test]
    fn parameter_ref_depth_and_chain_work_are_bounded() {
        for (composed, length) in [(true, MAX_SCHEMA_DEPTH + 1), (false, MAX_SCHEMA_WORK + 1)] {
            let mut definitions = Map::new();
            definitions.insert("n0".into(), serde_json::json!({"type": "integer"}));
            for index in 1..=length {
                let reference = serde_json::json!({"$ref": format!("#/$defs/n{}", index - 1)});
                let next = if composed {
                    serde_json::json!({"anyOf": [reference]})
                } else {
                    reference
                };
                definitions.insert(format!("n{index}"), next);
            }
            let root = serde_json::json!({"$defs": definitions});
            let schema = serde_json::json!({"$ref": format!("#/$defs/n{length}")});
            let mut walker = SchemaWalker::new(&root);
            assert!(!walker.has_type(Some(&schema), "integer"));
            assert!(walker.exhausted, "composed: {composed}");
            assert!(walker.path.is_empty());
            assert_eq!(
                convert_scalar_value("2", Some(&schema), &root),
                serde_json::json!("2")
            );
        }
    }
}
