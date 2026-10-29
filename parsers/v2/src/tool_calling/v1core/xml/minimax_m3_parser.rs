// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde_json::Value;
use uuid::Uuid;

use super::super::ToolDefinition;
use super::super::config::MiniMaxM3ParserConfig;
use super::response::{CalledFunction, ToolCallResponse, ToolCallType};
#[cfg(test)]
use crate::tool_calling::arguments::{MAX_SCHEMA_DEPTH, MAX_SCHEMA_WORK};
use crate::tool_calling::arguments::{
    ParsedObject, ParsedValue, SchemaFacts, coerce_integral_number, normalize_decimal_spelling,
    parsed_json_literal, raw_number_literal,
};

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
                    arguments: serde_json::to_string(&ParsedValue::Object(arguments))?,
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
) -> anyhow::Result<ParsedObject> {
    let parameter_start = parameter_start(config);
    let root_schema = get_arguments_config(function_name, tools).unwrap_or(&Value::Null);
    let param_config = root_schema
        .get("properties")
        .filter(|value| value.is_object())
        .unwrap_or(root_schema);
    let mut parameters = ParsedObject::new();
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
fn insert_parameter(parameters: &mut ParsedObject, key: String, value: ParsedValue) {
    if let Some(existing) = parameters.remove(&key) {
        let mut values = match existing {
            ParsedValue::Array(values) => values,
            ParsedValue::Json(Value::Array(values)) => {
                values.into_iter().map(ParsedValue::from).collect()
            }
            ParsedValue::RawJson(raw) if raw.get().trim_start().starts_with('[') => {
                serde_json::from_str::<Vec<Box<serde_json::value::RawValue>>>(raw.get())
                    .expect("validated raw array")
                    .into_iter()
                    .map(ParsedValue::RawJson)
                    .collect()
            }
            existing => vec![existing],
        };
        values.push(value);
        parameters.insert(key, ParsedValue::Array(values));
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
) -> ParsedValue {
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
) -> ParsedValue {
    let chunks: Vec<&str> = raw.split(config.namespace_token.as_str()).collect();
    let leading_text = chunks.first().copied().unwrap_or_default();
    let root_value = if schema
        .is_some_and(|schema| SchemaFacts::new(root_schema).may_describe_array(schema))
        && chunks
            .get(1)
            .is_some_and(|chunk| chunk.starts_with("<item>"))
    {
        Some(StackValue::Array(Vec::new()))
    } else {
        Some(StackValue::Object(ParsedObject::new()))
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
        schema: schema.into_iter().collect(),
        root_schema,
        array_parent_schemas: None,
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
            let child_schemas = stack
                .last()
                .expect("stack has current item")
                .schema_for_child(tag.as_str());
            let child_value = if SchemaFacts::new(root_schema)
                .may_describe_array_constraints(&child_schemas)
                && chunks
                    .get(chunk_index + 1)
                    .is_some_and(|next| next.starts_with("<item>"))
            {
                Some(StackValue::Array(Vec::new()))
            } else {
                None
            };
            // Whitespace before a child tag is formatting; before `</tag>` (or at the end of a
            // truncated value) it is the leaf value.
            let closes_leaf = chunks
                .get(chunk_index + 1)
                .is_none_or(|next| next.starts_with("</"));
            let array_parent_schemas = if tag == "item" {
                Some(stack.last().expect("stack has current item").schema.clone())
            } else {
                None
            };
            stack.push(StackItem {
                tag: Some(tag),
                value: child_value,
                texts: if trailing_text.is_empty()
                    || (trailing_text.trim().is_empty() && !closes_leaf)
                {
                    Vec::new()
                } else {
                    vec![trailing_text.to_string()]
                },
                schema: child_schemas,
                root_schema,
                array_parent_schemas,
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
    Object(ParsedObject),
    Array(Vec<ParsedValue>),
}

#[derive(Debug)]
struct StackItem<'a> {
    tag: Option<String>,
    value: Option<StackValue>,
    texts: Vec<String>,
    schema: Vec<&'a Value>,
    root_schema: &'a Value,
    array_parent_schemas: Option<Vec<&'a Value>>,
}

impl<'a> StackItem<'a> {
    // Converts a stack node into the JSON value it represents.
    fn into_value(self) -> ParsedValue {
        match self.value {
            None => {
                let text = self.texts.join("");
                if let Some(parent_schemas) = &self.array_parent_schemas {
                    convert_array_item_scalar(&text, parent_schemas, &self.schema, self.root_schema)
                } else {
                    convert_scalar_value_with_schemas(&text, &self.schema, self.root_schema)
                }
            }
            Some(StackValue::Object(mut map)) => {
                if !self.texts.is_empty() {
                    let mut text_key = "$text".to_string();
                    while map.contains_key(&text_key) {
                        text_key = format!("${text_key}");
                    }
                    map.insert(
                        text_key,
                        ParsedValue::from(Value::String(self.texts.join(""))),
                    );
                }
                ParsedValue::Object(map)
            }
            Some(StackValue::Array(values)) => ParsedValue::Array(values),
        }
    }

    // Attaches a completed child node to the current object, array, or implicit object.
    fn append(&mut self, item: StackItem) {
        let key = item.tag.clone().unwrap_or_default();
        let value = item.into_value();
        match self.value.as_mut() {
            None => {
                let mut map = ParsedObject::new();
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
            let item_schemas = self
                .schema
                .iter()
                .flat_map(|schema| SchemaFacts::new(self.root_schema).array_items(schema))
                .collect::<Vec<_>>();
            values.push(convert_array_item_scalar(
                text,
                &self.schema,
                &item_schemas,
                self.root_schema,
            ));
        } else {
            self.texts.push(text.to_string());
        }
    }

    // Finds the schema that should be used for a nested child tag.
    fn schema_for_child(&self, tag: &str) -> Vec<&'a Value> {
        let mut schemas = SchemaFacts::new(self.root_schema);
        if tag == "item" {
            let items = self
                .schema
                .iter()
                .flat_map(|schema| schemas.array_items(schema))
                .collect::<Vec<_>>();
            if !items.is_empty() {
                return items;
            }
        }

        self.schema
            .iter()
            .flat_map(|schema| schemas.object_children(schema, tag))
            .collect()
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
fn convert_scalar_value(raw: &str, schema: Option<&Value>, root_schema: &Value) -> ParsedValue {
    let schemas = schema.into_iter().collect::<Vec<_>>();
    convert_scalar_value_with_schemas(raw, &schemas, root_schema)
}

fn convert_array_item_scalar(
    raw: &str,
    array_schemas: &[&Value],
    item_schemas: &[&Value],
    root_schema: &Value,
) -> ParsedValue {
    let value = html_unescape(raw);
    let trimmed = value.trim();
    let mut facts = SchemaFacts::new(root_schema);

    if trimmed.eq_ignore_ascii_case("null") {
        return if facts.array_item_permits_null(array_schemas) {
            ParsedValue::from(Value::Null)
        } else {
            ParsedValue::from(Value::String(value))
        };
    }

    // An array's anyOf/oneOf item schemas are alternatives, unlike the
    // intersection used for sibling constraints on one selected property.
    if facts.has_array_item_type(array_schemas, "string") {
        return ParsedValue::from(Value::String(value));
    }
    if facts.has_array_item_type(array_schemas, "integer")
        && let Some(integer) = coerce_integral_number(trimmed)
    {
        return integer;
    }
    if facts.has_array_item_type(array_schemas, "number")
        && let Some(number) = coerce_integral_number(trimmed)
            .or_else(|| raw_number_literal(trimmed))
            .or_else(|| normalize_decimal_spelling(trimmed))
    {
        return number;
    }
    if facts.has_array_item_type(array_schemas, "boolean") {
        return match trimmed.to_ascii_lowercase().as_str() {
            "true" | "1" => ParsedValue::from(Value::Bool(true)),
            "false" | "0" => ParsedValue::from(Value::Bool(false)),
            _ => ParsedValue::from(Value::String(value)),
        };
    }

    convert_scalar_value_with_schemas(raw, item_schemas, root_schema)
}

// Null must be decided before string/number preference, for both properties and
// array items. Unknown or absent schema constraints preserve the original text.
fn convert_null_scalar<'a>(
    value: &str,
    schemas: &[&'a Value],
    facts: &mut SchemaFacts<'a>,
) -> Option<ParsedValue> {
    if !value.trim().eq_ignore_ascii_case("null") {
        return None;
    }
    let permits_null = !schemas.is_empty()
        && schemas
            .iter()
            .all(|schema| facts.permits_null(schema) == Some(true));
    Some(if permits_null {
        ParsedValue::from(Value::Null)
    } else {
        ParsedValue::from(Value::String(value.to_string()))
    })
}

fn convert_scalar_value_with_schemas(
    raw: &str,
    schema: &[&Value],
    root_schema: &Value,
) -> ParsedValue {
    let mut schemas = SchemaFacts::new(root_schema);
    let value = html_unescape(raw);
    let trimmed = value.trim();

    // Without a schema we cannot know the intended type, so preserve the literal
    // text (including the string "null") instead of inventing a JSON null.
    if schema.is_empty() {
        return ParsedValue::from(Value::String(value));
    }

    if let Some(value) = convert_null_scalar(&value, schema, &mut schemas) {
        return value;
    }

    if schemas.has_type_constraints(schema, "string")
        || schemas.has_type_constraints(schema, "enum")
    {
        return ParsedValue::from(Value::String(value));
    }
    if schemas.has_type_constraints(schema, "integer")
        && let Some(integer) = coerce_integral_number(trimmed)
    {
        return integer;
    }
    if schemas.has_type_constraints(schema, "number") {
        return coerce_integral_number(trimmed)
            .or_else(|| raw_number_literal(trimmed))
            .or_else(|| normalize_decimal_spelling(trimmed))
            .unwrap_or_else(|| ParsedValue::from(Value::String(value)));
    }
    if schemas.has_type_constraints(schema, "boolean") {
        return match trimmed.to_ascii_lowercase().as_str() {
            "true" => ParsedValue::from(Value::Bool(true)),
            "1" => ParsedValue::from(Value::Bool(true)),
            "false" => ParsedValue::from(Value::Bool(false)),
            "0" => ParsedValue::from(Value::Bool(false)),
            _ => ParsedValue::from(Value::String(value)),
        };
    }
    if schemas.has_type_constraints(schema, "object") {
        if trimmed.is_empty() {
            return ParsedValue::Object(ParsedObject::new());
        }
        if let Some(json) = parsed_json_literal(trimmed) {
            return json;
        }
    }
    if schemas.has_type_constraints(schema, "array") {
        if trimmed.is_empty() {
            return ParsedValue::Array(Vec::new());
        }
        if let Some(json) = parsed_json_literal(trimmed) {
            return json;
        }
    }

    ParsedValue::from(Value::String(value))
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

    #[test]
    fn scalar_hints_preserve_explicit_type_and_alternative_policy() {
        for (schema, raw, expected) in [
            (
                json!({"anyOf": [{"const": "42"}, {"type": "integer"}]}),
                "42",
                "42",
            ),
            (
                json!({"anyOf": [{"const": "{\"x\":1}"}, {"type": "object"}]}),
                "{\"x\":1}",
                "{\"x\":1}",
            ),
            (
                json!({"allOf": [{"type": "integer"}, {"anyOf": [{"const": "auto"}, {"type": "integer"}]}]}),
                "42",
                "\"42\"",
            ),
            (
                json!({"allOf": [{"type": "integer"}, {"oneOf": [{"const": "auto"}, {"type": "integer"}]}]}),
                "42",
                "\"42\"",
            ),
            (
                json!({"type": "number", "anyOf": [{"type": "integer"}]}),
                "2.5",
                "2.5",
            ),
            (
                json!({"type": "number", "anyOf": [{"type": "string"}]}),
                "42",
                "\"42\"",
            ),
            (
                json!({"type": "string", "anyOf": [{"type": "number"}]}),
                "42",
                "\"42\"",
            ),
        ] {
            assert_eq!(
                convert_scalar_value(raw, Some(&schema), &schema)
                    .json_text()
                    .unwrap(),
                expected,
                "schema={schema}"
            );
        }
    }

    #[test]
    fn array_item_union_coerces_a_scalar_when_one_branch_allows_it() {
        let config = MiniMaxM3ParserConfig::default();
        let input = format!(
            "{}<item>9007199254740992.5{}</item>",
            config.namespace_token, config.namespace_token
        );
        for keyword in ["anyOf", "oneOf"] {
            for branches in [
                json!([{"type": "number"}, {"type": "object"}]),
                json!([{"type": "object"}, {"type": "number"}]),
            ] {
                let mut item_alternatives = serde_json::Map::new();
                item_alternatives.insert(keyword.to_string(), branches.clone());
                let item_schema = json!({
                    "type": "array",
                    "items": item_alternatives
                });
                let root_schema = json!({"properties": {"values": item_schema}});
                let value = parse_nested_minimax_xml(
                    &input,
                    root_schema
                        .get("properties")
                        .and_then(|props| props.get("values")),
                    &root_schema,
                    &config,
                );

                assert_eq!(
                    serde_json::to_string(&value).unwrap(),
                    "[9007199254740992.5]",
                    "keyword: {keyword}, branches: {branches}"
                );
            }
        }

        let string_or_number = json!({
            "type": "array",
            "items": {"anyOf": [{"type": "number"}, {"type": "string"}]}
        });
        let string_root = json!({"properties": {"values": string_or_number}});
        let string_value = parse_nested_minimax_xml(
            &input,
            string_root
                .get("properties")
                .and_then(|props| props.get("values")),
            &string_root,
            &config,
        );
        assert_eq!(
            serde_json::to_string(&string_value).unwrap(),
            "[\"9007199254740992.5\"]"
        );

        let contradictory_items = json!({
            "type": "array",
            "items": {"allOf": [{"type": "number"}, {"type": "object"}]}
        });
        let contradictory_root = json!({"properties": {"values": contradictory_items}});
        let contradictory_value = parse_nested_minimax_xml(
            &input,
            contradictory_root
                .get("properties")
                .and_then(|props| props.get("values")),
            &contradictory_root,
            &config,
        );
        assert_eq!(
            serde_json::to_string(&contradictory_value).unwrap(),
            "[\"9007199254740992.5\"]"
        );
    }

    #[test]
    fn array_schema_alternatives_preserve_null_and_boolean_items() {
        let config = MiniMaxM3ParserConfig::default();
        let token = &config.namespace_token;
        for keyword in ["anyOf", "oneOf"] {
            for branches in [
                json!([
                    {"type": "array", "items": {"type": ["number", "null"]}},
                    {"type": "array", "items": {"type": "number"}}
                ]),
                json!([
                    {"type": "array", "items": {"type": "number"}},
                    {"type": "array", "items": {"type": ["number", "null"]}}
                ]),
            ] {
                let mut schema = serde_json::Map::new();
                schema.insert(keyword.to_string(), branches.clone());
                let values = Value::Object(schema);
                let root = json!({"properties": {"values": values}});
                let values_schema = root["properties"]["values"].clone();
                let input = format!("{token}<item>null{token}</item>");
                assert_eq!(
                    parse_nested_minimax_xml(&input, Some(&values_schema), &root, &config),
                    json!([null]),
                    "{keyword}: {branches}"
                );
            }

            for branches in [
                json!([
                    {"type": "array", "items": {"type": "boolean"}},
                    {"type": "array", "items": {"type": "number"}}
                ]),
                json!([
                    {"type": "array", "items": {"type": "number"}},
                    {"type": "array", "items": {"type": "boolean"}}
                ]),
            ] {
                let mut schema = serde_json::Map::new();
                schema.insert(keyword.to_string(), branches.clone());
                let values = Value::Object(schema);
                let root = json!({"properties": {"values": values}});
                let values_schema = root["properties"]["values"].clone();
                let input = format!("{token}<item>true{token}</item>");
                assert_eq!(
                    parse_nested_minimax_xml(&input, Some(&values_schema), &root, &config),
                    json!([true]),
                    "{keyword}: {branches}"
                );
            }
        }
    }

    #[test]
    fn array_item_union_uses_items_when_array_type_is_implicit() {
        let config = MiniMaxM3ParserConfig::default();
        let input = format!(
            "{}<item>{}<item>9007199254740992.5{}</item>{}</item>",
            config.namespace_token,
            config.namespace_token,
            config.namespace_token,
            config.namespace_token
        );
        let schema = json!({
            "type": "array",
            "items": {"anyOf": [
                {"items": {"type": "number"}},
                {"type": "object"}
            ]}
        });
        let root_schema = json!({"properties": {"values": schema}});
        let value = parse_nested_minimax_xml(
            &input,
            root_schema
                .get("properties")
                .and_then(|props| props.get("values")),
            &root_schema,
            &config,
        );

        assert_eq!(
            serde_json::to_string(&value).unwrap(),
            "[[9007199254740992.5]]"
        );
    }

    #[test]
    fn null_coercion_intersects_schema_constraints() {
        for (schema, expected) in [
            (
                json!({"anyOf": [{"type": "string"}, {"const": null}]}),
                json!(null),
            ),
            (
                json!({"type": "string", "anyOf": [{"type": "string"}, {"type": "null"}]}),
                json!("null"),
            ),
            (
                json!({"type": "string", "oneOf": [{"type": "string"}, {"type": "null"}]}),
                json!("null"),
            ),
            (
                json!({"type": "string", "anyOf": [{}, {"type": "integer"}]}),
                json!("null"),
            ),
            (json!({"enum": ["null"]}), json!("null")),
            (json!({"const": "null"}), json!("null")),
            (
                json!({"allOf": [{"type": ["string", "null"]}, {"type": "string"}]}),
                json!("null"),
            ),
            (json!({"oneOf": [{}, {"type": "null"}]}), json!("null")),
            (json!({"not": {"type": "null"}}), json!("null")),
            (json!(false), json!("null")),
        ] {
            assert_eq!(
                convert_scalar_value("null", Some(&schema), &schema),
                expected,
                "{schema}"
            );
        }

        let root = json!({
            "$defs": {"value": {"anyOf": [{"type": "string"}, {"const": null}]}}
        });
        let reference = json!({"$ref": "#/$defs/value"});
        assert_eq!(
            convert_scalar_value("null", Some(&reference), &root),
            json!(null)
        );
    }

    #[test]
    fn unresolved_null_constraints_remain_unknown() {
        for unknown in [
            json!({"$ref": "#/$defs/missing"}),
            json!({"$ref": "https://example.com/schema"}),
            json!({"$ref": "#"}),
            json!({"not": {"$ref": "#"}}),
        ] {
            for (schema, expected) in [
                (unknown.clone(), json!("null")),
                (
                    json!({"anyOf": [{"type": "string"}, unknown]}),
                    json!("null"),
                ),
                (json!({"allOf": [true, unknown]}), json!("null")),
                (json!({"oneOf": [true, unknown]}), json!("null")),
                (json!({"not": unknown}), json!("null")),
                (json!({"type": "object", "not": unknown}), json!("null")),
            ] {
                assert_eq!(
                    convert_scalar_value("null", Some(&schema), &schema),
                    expected,
                    "{schema}"
                );
            }
        }
        let unknown = json!({"$ref": "#/$defs/missing"});
        for (schema, expected) in [
            (json!({"anyOf": [unknown, true]}), Some(true)),
            (json!({"allOf": [unknown, false]}), Some(false)),
            (json!({"oneOf": [true, unknown, true]}), Some(false)),
            (json!({"not": {"allOf": [unknown, false]}}), Some(true)),
        ] {
            assert_eq!(
                SchemaFacts::new(&schema).permits_null(&schema),
                expected,
                "{schema}"
            );
        }
    }

    #[test]
    fn exhausted_null_constraints_remain_unknown() {
        for keyword in ["allOf", "anyOf", "oneOf", "not"] {
            let mut schema = json!({"type": "null"});
            for _ in 0..=MAX_SCHEMA_DEPTH {
                schema = if keyword == "not" {
                    json!({"not": schema})
                } else {
                    json!({keyword: [schema]})
                };
            }
            let mut walker = SchemaFacts::new(&schema);
            assert_eq!(walker.permits_null(&schema), None, "{keyword}");
            assert!(walker.exhausted);
            assert_eq!(
                convert_scalar_value("null", Some(&schema), &schema),
                json!("null")
            );
        }
        let schema = json!({"not": {"anyOf": vec![json!(false); MAX_SCHEMA_WORK + 1]}});
        let mut walker = SchemaFacts::new(&schema);
        assert_eq!(walker.permits_null(&schema), None);
        assert!(walker.exhausted);
        assert_eq!(
            convert_scalar_value("null", Some(&schema), &schema),
            json!("null")
        );
    }

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

    #[test]
    fn pretty_printed_nested_string_leaves_keep_whitespace() {
        let config = MiniMaxM3ParserConfig::default();
        let schema = json!({"type": "object", "properties": {
            "name": {"type": "string"},
            "sep": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}}
        }});
        // Whitespace-only leaves are values: the template emits `<sep>  </sep>` verbatim.
        let raw = format!(
            "\n{TOK}<name>  src  {TOK}</name>\n{TOK}<sep>  {TOK}</sep>\n{TOK}<tags>\n\
             {TOK}<item> a b {TOK}</item>\n{TOK}<item>\tc\n{TOK}</item>\n\
             {TOK}<item>\n{TOK}</item>\n{TOK}</tags>\n"
        );
        assert_eq!(
            parse_nested_minimax_xml(&raw, Some(&schema), &schema, &config),
            json!({"name": "  src  ", "sep": "  ", "tags": [" a b ", "\tc\n", "\n"]})
        );
        // A truncated value that ends after a start tag keeps that whitespace as the leaf.
        let raw = format!("{TOK}<sep>  ");
        assert_eq!(
            parse_nested_minimax_xml(&raw, Some(&schema), &schema, &config),
            json!({"sep": "  "})
        );
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
            ParsedValue::Object(actual).to_value(),
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
            SchemaFacts::new(&root).resolve_ref(&chain).unwrap(),
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
                SchemaFacts::new(&root).resolve_ref(&schema).unwrap(),
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
            ParsedValue::Object(actual).to_value(),
            serde_json::json!({
                "options": {"count": 2, "enabled": true, "counts": [3]}
            })
        );
    }

    #[test]
    fn nested_ref_siblings_keep_referenced_numeric_properties_exact() {
        let parameters = serde_json::json!({
            "$defs": {
                "Base": {
                    "type": "object",
                    "properties": {"amount": {"type": "number"}}
                }
            },
            "properties": {
                "data": {
                    "$ref": "#/$defs/Base",
                    "properties": {"label": {"type": "string"}}
                }
            }
        });
        let tools = vec![ToolDefinition {
            name: "capture".into(),
            parameters: Some(parameters),
        }];
        let tok = "]<]minimax[>[";
        let raw = format!(
            "{tok}<data>{tok}<amount>9007199254740992.5{tok}</amount>{tok}<label>x{tok}</label>{tok}</data>"
        );
        let actual = parse_parameters(
            "capture",
            &raw,
            &MiniMaxM3ParserConfig::default(),
            Some(&tools),
        )
        .unwrap();

        assert_eq!(
            ParsedValue::Object(actual).json_text().unwrap(),
            r#"{"data":{"amount":9007199254740992.5,"label":"x"}}"#
        );

        let conflicting_sibling = serde_json::json!({
            "$defs": {
                "Base": {
                    "type": "object",
                    "properties": {"amount": {"type": "number"}}
                }
            },
            "properties": {
                "data": {
                    "$ref": "#/$defs/Base",
                    "properties": {"amount": {"type": "string"}}
                }
            }
        });
        let tools = vec![ToolDefinition {
            name: "capture".into(),
            parameters: Some(conflicting_sibling),
        }];
        let raw = format!("{tok}<data>{tok}<amount>2.5{tok}</amount>{tok}</data>");
        let actual = parse_parameters(
            "capture",
            &raw,
            &MiniMaxM3ParserConfig::default(),
            Some(&tools),
        )
        .unwrap();
        assert_eq!(
            ParsedValue::Object(actual).json_text().unwrap(),
            r#"{"data":{"amount":"2.5"}}"#
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
                SchemaFacts::new(&root).resolve_ref(&schema).unwrap(),
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
            assert_eq!(
                ParsedValue::Object(actual).as_object().unwrap()["value"],
                expected,
                "{schema}"
            );
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
            ParsedValue::Object(actual).to_value(),
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
        let mut definitions = serde_json::Map::new();
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
        let mut walker = SchemaFacts::new(&root);
        assert!(!walker.has_type(Some(&schema), "string"));
        assert!(walker.exhausted);
        assert_eq!(walker.remaining_work, 0);
        // An inconclusive string lookup must not permit a later integer coercion.
        assert!(!walker.has_type(Some(&schema), "integer"));
        assert!(walker.may_describe_object(&schema));
        assert!(walker.object_children(&schema, "count").is_empty());
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
            let mut definitions = serde_json::Map::new();
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
            let mut walker = SchemaFacts::new(&root);
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
