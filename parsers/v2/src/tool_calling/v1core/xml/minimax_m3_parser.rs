// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{borrow::Cow, collections::HashMap};

use serde_json::Value;

use super::super::ToolDefinition;
use super::super::config::MiniMaxM3ParserConfig;
use crate::tool_calling::arguments::{
    ParsedObject, ParsedValue, SchemaFacts, coerce_integral_number, normalize_decimal_spelling,
    parsed_json_literal, raw_number_literal, resolve_local_schema_ref,
};

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

/// Type a scanner-delimited invocation without interpreting argument text as
/// outer framing. Return JSON in source order for append-only streaming deltas.
pub(crate) fn parse_minimax_m3_invoke(
    invoke: &str,
    config: &MiniMaxM3ParserConfig,
    tools: &[ToolDefinition],
) -> anyhow::Result<Option<(String, String)>> {
    let Some(header) = invoke.strip_prefix(&invoke_start(config)) else {
        return Ok(None);
    };
    let Some(end) = header.find('>') else {
        return Ok(None);
    };
    let Some(name) = parse_invoke_name(&header[..end]).filter(|name| !name.is_empty()) else {
        return Ok(None);
    };
    let Some(body) = header[end + 1..].strip_suffix(&invoke_end(config)) else {
        return Ok(None);
    };
    let mut parameters = parse_parameters(&name, body, config, Some(tools))?;
    let mut arguments = String::from("{");
    for (parameter, _) in parameter_values(
        body,
        config,
        get_arguments_config(&name, Some(tools)).unwrap_or(&Value::Null),
    ) {
        if let Some(value) = parameters.remove(parameter) {
            if arguments.len() > 1 {
                arguments.push(',');
            }
            arguments.push_str(&serde_json::to_string(parameter)?);
            arguments.push(':');
            arguments.push_str(&serde_json::to_string(&value)?);
        }
    }
    arguments.push('}');
    Ok(Some((name, arguments)))
}

// Reads the `name` attribute from `<invoke ...>` using vLLM-compatible quoting variants.
pub(crate) fn parse_invoke_name(tag_attrs: &str) -> Option<String> {
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

// Iterate complete top-level parameter bodies once, respecting same-name nesting.
fn parameter_values<'a>(
    body: &'a str,
    config: &MiniMaxM3ParserConfig,
    root_schema: &Value,
) -> Vec<(&'a str, &'a str)> {
    let parameter_start = parameter_start(config);
    let close_tags = match_close_tags(body, &config.namespace_token, root_schema);
    let mut values = Vec::new();
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
        if parameter_name.is_empty() {
            cursor = start + name_end_rel + 1;
            continue;
        }

        let value_start = start + name_end_rel + 1;
        let parameter_end = format!("{}</{}>", config.namespace_token, parameter_name);
        let Some(value_end) = close_tags.get(&value_start).copied().or_else(|| {
            body[value_start..]
                .find(parameter_end.as_str())
                .map(|rel| value_start + rel)
        }) else {
            break;
        };
        values.push((parameter_name, &body[value_start..value_end]));
        cursor = value_end + parameter_end.len();
    }
    values
}

// Extracts MiniMax M3 parameter tags, where each parameter name is the tag name itself.
fn parse_parameters(
    function_name: &str,
    body: &str,
    config: &MiniMaxM3ParserConfig,
    tools: Option<&[ToolDefinition]>,
) -> anyhow::Result<ParsedObject> {
    let root_schema = get_arguments_config(function_name, tools).unwrap_or(&Value::Null);
    let mut parameters = ParsedObject::new();

    for (parameter_name, raw_value) in parameter_values(body, config, root_schema) {
        let schema = parameter_schema(root_schema, parameter_name);
        let value = parse_parameter_value(raw_value, schema.as_deref(), root_schema, config);
        insert_parameter(&mut parameters, parameter_name.to_string(), value);
    }

    Ok(parameters)
}

fn parameter_schema<'a>(root: &'a Value, name: &str) -> Option<Cow<'a, Value>> {
    let (required, alternatives) = SchemaWalker::new(root).parameter_schemas(name);
    if required.len() > 1 {
        return Some(Cow::Owned(serde_json::json!({"allOf": required})));
    }
    match (required.first(), alternatives.as_slice()) {
        (Some(schema), _) => Some(Cow::Borrowed(*schema)),
        (None, []) => root
            .get("properties")
            .filter(|value| value.is_object())
            .unwrap_or(root)
            .get(name)
            .map(Cow::Borrowed),
        (None, [schema]) => Some(Cow::Borrowed(*schema)),
        (None, _) => Some(Cow::Owned(serde_json::json!({"anyOf": alternatives}))),
    }
}

/// Native XML does not encode whether a scalar-looking value was a string.
/// Use the same bounded schema interpretation as the parser when deciding if
/// guided generation needs JSON to preserve that distinction.
pub(crate) fn native_types_require_json(root: &Value, schema: &Value) -> bool {
    let mut walker = SchemaWalker::new(root);
    let string = walker.has_type(Some(schema), "string");
    let mixed = string
        && (walker.permits_null(schema) == Some(true)
            || ["integer", "number", "boolean", "object", "array"]
                .iter()
                .any(|kind| walker.has_type(Some(schema), kind)));
    mixed || walker.exhausted
}

/// Nested XML identifies an object or an array, but not which union alternative
/// it belongs to. `object_child` types a child only when exactly one alternative
/// can describe an object, and `array_item` reads the first alternative's
/// `items`. Report unions the native grammar would render as nested tags but
/// the parser could not type, so the builder keeps them on the JSON path.
pub(crate) fn native_union_children_ambiguous(root: &Value, schema: &Value) -> bool {
    let mut walker = SchemaWalker::new(root);
    // Empty objects and arrays have identical native tags, including when the
    // alternatives are expressed as a type list or reached through a reference.
    if walker.has_type(Some(schema), "object") && walker.has_type(Some(schema), "array") {
        return true;
    }
    let Some(object) = schema.as_object() else {
        return false;
    };
    for keyword in ["anyOf", "oneOf"] {
        let Some(branches) = object.get(keyword).and_then(Value::as_array) else {
            continue;
        };
        let objects = branches
            .iter()
            .filter(|branch| walker.may_describe_object(branch))
            .count();
        let arrays = branches
            .iter()
            .filter(|branch| walker.array_item(Some(branch)).is_some())
            .count();
        if objects > 1 || arrays > 1 {
            return true;
        }
        // Keep implicit object/array shapes on the same conservative path.
        if objects >= 1 && arrays >= 1 {
            return true;
        }
    }
    walker.exhausted
}

#[derive(Clone, Copy)]
pub(crate) enum M3ParameterKind {
    LiteralString { streamable: bool },
    Typed,
    // Repeated array tags and unknown-schema tags can aggregate earlier values.
    Buffered,
}

/// Use the value converter's schema lookup to classify raw tag ownership and
/// whether a completed parameter is stable enough for append-only output.
pub(crate) fn parameter_kind(
    name: &str,
    parameter: &str,
    tools: &[ToolDefinition],
) -> M3ParameterKind {
    parameter_path_kind(name, &[parameter.to_string()], tools)
}

pub(crate) fn parameter_path_kind(
    name: &str,
    path: &[String],
    tools: &[ToolDefinition],
) -> M3ParameterKind {
    let Some(root) = get_arguments_config(name, Some(tools)) else {
        return M3ParameterKind::Buffered;
    };
    path_kind(root, path)
}

fn path_kind(root: &Value, path: &[String]) -> M3ParameterKind {
    if path.len() > MAX_SCHEMA_DEPTH {
        return M3ParameterKind::Buffered;
    }
    let Some(first) = path.first() else {
        return M3ParameterKind::Buffered;
    };
    let Some(mut schema) = parameter_schema(root, first) else {
        return M3ParameterKind::Buffered;
    };
    for tag in &path[1..] {
        let mut walker = SchemaWalker::new(root);
        let child = if tag == "item" {
            walker.array_item(Some(&schema))
        } else {
            None
        }
        .or_else(|| walker.object_child(&schema, tag));
        let Some(child) = child else {
            return M3ParameterKind::Buffered;
        };
        schema = Cow::Owned(child.clone());
    }
    let schema = schema.as_ref();
    let mut walker = SchemaWalker::new(root);
    let string = walker.has_type(Some(schema), "string");
    let object = walker.has_type(Some(schema), "object");
    let array = walker.has_type(Some(schema), "array");
    let scalar = walker.has_type(Some(schema), "integer")
        || walker.has_type(Some(schema), "number")
        || walker.has_type(Some(schema), "boolean");
    if string && !object && !array && !walker.exhausted {
        return M3ParameterKind::LiteralString {
            streamable: !scalar && walker.permits_null(schema) == Some(false),
        };
    }
    if !array && (object || scalar) && !walker.exhausted {
        M3ParameterKind::Typed
    } else {
        M3ParameterKind::Buffered
    }
}

// Maps each tag's value start to its matching same-name close tag. Nested objects render
// recursively, so a child may reuse its parent's tag name; a per-name stack pairs them in
// one pass. Unbalanced openers have no entry, and callers fall back to the first close tag.
fn match_close_tags(
    body: &str,
    namespace_token: &str,
    root_schema: &Value,
) -> HashMap<usize, usize> {
    let mut open: HashMap<&str, Vec<usize>> = HashMap::new();
    let mut matched = HashMap::new();
    let mut path = Vec::new();
    let mut cursor = 0;
    while let Some(tag_rel) = body[cursor..].find(namespace_token) {
        let tag_start = cursor + tag_rel;
        cursor = tag_start + namespace_token.len();
        let Some(tag) = body[cursor..].strip_prefix('<') else {
            continue;
        };
        let Some(name_end) = tag.find('>') else {
            break;
        };
        match tag[..name_end].strip_prefix('/') {
            Some(name) => {
                if let Some(at) = path.iter().rposition(|part| part == name) {
                    path.truncate(at);
                }
                if let Some(value_start) = open.get_mut(name).and_then(Vec::pop) {
                    matched.insert(value_start, tag_start);
                }
            }
            None => {
                let name = &tag[..name_end];
                let value_start = cursor + name_end + 2;
                path.push(name.to_string());
                if matches!(
                    path_kind(root_schema, &path),
                    M3ParameterKind::LiteralString { .. }
                ) {
                    let close = format!("{namespace_token}</{name}>");
                    if let Some(at) = body[value_start..].find(&close) {
                        let at = value_start + at;
                        matched.insert(value_start, at);
                        cursor = at + close.len();
                        path.pop();
                        continue;
                    }
                }
                open.entry(name).or_default().push(value_start);
            }
        }
    }
    matched
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
    let mut walker = SchemaWalker::new(root_schema);
    let literal_string = walker.has_type(schema, "string")
        && !walker.has_type(schema, "object")
        && !walker.has_type(schema, "array");
    if !literal_string && raw.contains(parameter_start(config).as_str()) {
        parse_nested_minimax_xml(raw, schema, root_schema, config)
    } else {
        convert_scalar_value(raw, schema, root_schema)
    }
}

fn is_literal_string(schema: Option<&Value>, root_schema: &Value) -> bool {
    let mut walker = SchemaWalker::new(root_schema);
    walker.has_type(schema, "string")
        && !walker.has_type(schema, "object")
        && !walker.has_type(schema, "array")
        && !walker.exhausted
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
        // Strings in this format are unescaped, including namespace/tag-looking
        // bytes. Only their own closing tag changes the nested parse stack.
        if let Some(item) = stack.last_mut()
            && let Some(tag) = &item.tag
            && item
                .schema
                .iter()
                .any(|schema| is_literal_string(Some(schema), root_schema))
            && !chunk.starts_with(&format!("</{tag}>"))
        {
            item.append_text(&config.namespace_token);
            item.append_text(chunk);
            continue;
        }
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
                    || (trailing_text.trim().is_empty()
                        && !closes_leaf
                        && !child_schemas
                            .iter()
                            .any(|schema| is_literal_string(Some(schema), root_schema)))
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
// Values are literal text: the chat template renders `{{ val }}` unescaped.
fn convert_scalar_value(raw: &str, schema: Option<&Value>, root_schema: &Value) -> ParsedValue {
    let mut schemas = SchemaWalker::new(root_schema);
    let value = raw.to_string();
    let trimmed = value.trim();

    // Without a schema we cannot know the intended type, so preserve the literal
    // text (including the string "null") instead of inventing a JSON null.
    let Some(schema) = schema else {
        return ParsedValue::from(Value::String(value));
    };

    // Only collapse the literal "null" into JSON null when the schema actually
    // permits null. A `string`-typed parameter keeps the literal value "null".
    if trimmed.eq_ignore_ascii_case("null") {
        return if schemas.permits_null(schema) == Some(true) {
            ParsedValue::from(Value::Null)
        } else {
            ParsedValue::from(Value::String(value))
        };
    }

    if (schemas.has_type(Some(schema), "string")
        && !schemas.rejects_string_literal(schema, raw)
        && (schemas.type_constraint(schema, "string", false) == Some(true)
            || !["integer", "number", "boolean", "object", "array"]
                .iter()
                .any(|kind| schemas.type_constraint(schema, kind, false) == Some(true))))
        || SchemaFacts::new(root_schema).has_type_constraints(&[schema], "enum")
    {
        return ParsedValue::from(Value::String(value));
    }
    if schemas.has_type(Some(schema), "integer")
        && let Some(parsed) = coerce_integral_number(trimmed)
    {
        return parsed;
    }
    if schemas.has_type(Some(schema), "number")
        && let Some(number) = coerce_integral_number(trimmed)
            .or_else(|| raw_number_literal(trimmed))
            .or_else(|| normalize_decimal_spelling(trimmed))
    {
        return number;
    }
    if schemas.has_type(Some(schema), "boolean") {
        match trimmed.to_ascii_lowercase().as_str() {
            "true" | "1" => return ParsedValue::from(Value::Bool(true)),
            "false" | "0" => return ParsedValue::from(Value::Bool(false)),
            _ => {}
        }
    }
    if schemas.has_type(Some(schema), "object") {
        if trimmed.is_empty() {
            return ParsedValue::Object(ParsedObject::new());
        }
        if let Some(json) = parsed_json_literal(trimmed) {
            return json;
        }
    }
    if schemas.has_type(Some(schema), "array") {
        if trimmed.is_empty() {
            return ParsedValue::Array(Vec::new());
        }
        if let Some(json) = parsed_json_literal(trimmed) {
            return json;
        }
    }

    ParsedValue::from(Value::String(value))
}

fn convert_array_item_scalar(
    raw: &str,
    array_schemas: &[&Value],
    item_schemas: &[&Value],
    root_schema: &Value,
) -> ParsedValue {
    let value = raw.to_string();
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

fn convert_scalar_value_with_schemas(
    raw: &str,
    schemas: &[&Value],
    root_schema: &Value,
) -> ParsedValue {
    match schemas {
        [] => convert_scalar_value(raw, None, root_schema),
        [schema] => convert_scalar_value(raw, Some(schema), root_schema),
        _ => convert_scalar_value(
            raw,
            Some(&serde_json::json!({"allOf": schemas})),
            root_schema,
        ),
    }
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
                        "$ref"
                            | "$defs"
                            | "definitions"
                            | "title"
                            | "description"
                            | "default"
                            | "examples"
                            | "$comment"
                    )
                })
            }) || visited.contains(&reference)
            {
                return Some(schema);
            }
            let Some(target) = resolve_local_schema_ref(reference, self.root) else {
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

    // Tool parameters may be declared at the root, under allOf, or under anyOf/oneOf
    // variants. Root and allOf declarations always apply, so they win over variant
    // declarations; a parameter declared only by several variants is typed by their union.
    fn parameter_schemas(&mut self, tag: &str) -> (Vec<&'a Value>, Vec<&'a Value>) {
        let mut required = Vec::new();
        let mut alternatives = Vec::new();
        self.collect_parameter_schemas(self.root, tag, false, &mut required, &mut alternatives);
        if self.exhausted {
            (Vec::new(), Vec::new())
        } else {
            (required, alternatives)
        }
    }

    fn collect_parameter_schemas(
        &mut self,
        schema: &'a Value,
        tag: &str,
        in_variant: bool,
        required: &mut Vec<&'a Value>,
        alternatives: &mut Vec<&'a Value>,
    ) {
        self.with_schema(schema, (), |walker, schema| {
            if let Some(child) = schema.get("properties").and_then(|props| props.get(tag)) {
                let declared = if in_variant {
                    &mut *alternatives
                } else {
                    &mut *required
                };
                if !declared.contains(&child) {
                    declared.push(child);
                }
            }
            for keyword in ["allOf", "anyOf", "oneOf"] {
                if let Some(branches) = schema.get(keyword).and_then(Value::as_array) {
                    let in_variant = in_variant || keyword != "allOf";
                    for branch in branches {
                        walker.collect_parameter_schemas(
                            branch,
                            tag,
                            in_variant,
                            required,
                            alternatives,
                        );
                    }
                }
            }
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

    fn permits_null(&mut self, schema: &'a Value) -> Option<bool> {
        self.with_schema(schema, None, |walker, schema| {
            if let Some(allowed) = schema.as_bool() {
                return Some(allowed);
            }
            if let Some(ty) = schema.get("type") {
                let nullable = schema.get("nullable").and_then(Value::as_bool) == Some(true);
                if !nullable
                    && ty.as_str() != Some("null")
                    && !ty
                        .as_array()
                        .is_some_and(|types| types.iter().any(|ty| ty == "null"))
                {
                    return Some(false);
                }
            }
            if schema.get("const").is_some_and(|value| !value.is_null())
                || schema
                    .get("enum")
                    .and_then(Value::as_array)
                    .is_some_and(|values| !values.iter().any(Value::is_null))
            {
                return Some(false);
            }
            let mut result = match schema.get("$ref") {
                Some(reference) => reference
                    .as_str()
                    .and_then(|reference| resolve_local_schema_ref(reference, walker.root))
                    .and_then(|target| walker.permits_null(target)),
                None => Some(true),
            };
            if result == Some(false) {
                return Some(false);
            }
            for keyword in ["allOf", "anyOf", "oneOf"] {
                if let Some(branches) = schema.get(keyword).and_then(Value::as_array) {
                    let mut matches = 0;
                    let mut rejects = 0;
                    let mut unknown = false;
                    for branch in branches {
                        match walker.permits_null(branch) {
                            Some(true) => matches += 1,
                            Some(false) => rejects += 1,
                            None => unknown = true,
                        }
                    }
                    let permitted = match keyword {
                        "allOf" if rejects > 0 => Some(false),
                        "anyOf" if matches > 0 => Some(true),
                        "oneOf" if matches > 1 => Some(false),
                        _ if unknown => None,
                        "allOf" => Some(true),
                        "anyOf" => Some(false),
                        _ => Some(matches == 1),
                    };
                    if permitted == Some(false) {
                        return Some(false);
                    }
                    if permitted.is_none() {
                        result = None;
                    }
                }
            }
            if let Some(branch) = schema.get("not") {
                match walker.permits_null(branch) {
                    Some(true) => return Some(false),
                    Some(false) => {}
                    None => result = None,
                }
            }
            result
        })
    }

    fn has_type(&mut self, schema: Option<&'a Value>, expected: &str) -> bool {
        schema.is_some_and(|schema| self.type_constraint(schema, expected, true) == Some(true))
    }

    // A string literal branch such as `const: "auto"` must not hide a numeric
    // alternative when the input is "42". Only reject known type/literal
    // contradictions here; this coercion helper is not a schema validator.
    fn rejects_string_literal(&mut self, schema: &'a Value, raw: &str) -> bool {
        self.with_schema(schema, false, |walker, schema| {
            if schema == &Value::Bool(false) {
                return true;
            }
            if schema.get("type").is_some_and(|ty| {
                ty.as_str() != Some("string")
                    && !ty
                        .as_array()
                        .is_some_and(|types| types.iter().any(|ty| ty == "string"))
            }) || schema
                .get("const")
                .is_some_and(|value| value.as_str() != Some(raw))
                || schema
                    .get("enum")
                    .and_then(Value::as_array)
                    .is_some_and(|values| !values.iter().any(|value| value.as_str() == Some(raw)))
            {
                return true;
            }
            if let Some(reference) = schema.get("$ref").and_then(Value::as_str)
                && let Some(target) = resolve_local_schema_ref(reference, walker.root)
                && walker.rejects_string_literal(target, raw)
            {
                return true;
            }
            for keyword in ["allOf", "anyOf", "oneOf"] {
                if let Some(branches) = schema.get(keyword).and_then(Value::as_array) {
                    let rejected = if keyword == "allOf" {
                        branches
                            .iter()
                            .any(|branch| walker.rejects_string_literal(branch, raw))
                    } else {
                        branches
                            .iter()
                            .all(|branch| walker.rejects_string_literal(branch, raw))
                    };
                    if rejected {
                        return true;
                    }
                }
            }
            false
        })
    }

    // None means no known type constraint; false means this type is excluded.
    // Sibling keywords and allOf intersect, while anyOf/oneOf combine alternatives.
    fn type_constraint(
        &mut self,
        schema: &'a Value,
        expected: &str,
        infer_literals: bool,
    ) -> Option<bool> {
        self.with_schema(schema, None, |walker, schema| {
            if schema == &Value::Bool(false) {
                return Some(false);
            }
            let matches =
                |name: &str| name == expected || (name == "number" && expected == "integer");
            let mut constraint = schema.get("type").and_then(|ty| {
                ty.as_str().map(matches).or_else(|| {
                    ty.as_array()
                        .map(|types| types.iter().filter_map(Value::as_str).any(matches))
                })
            });
            // A reference and its sibling constraints apply together. Keep the
            // sibling node intact and intersect the target's facts with it.
            if let Some(reference) = schema.get("$ref").and_then(Value::as_str)
                && let Some(target) = resolve_local_schema_ref(reference, walker.root)
                && let Some(permits) = walker.type_constraint(target, expected, infer_literals)
            {
                constraint = Some(constraint.unwrap_or(true) && permits);
            }
            // A plain negated type can exclude strings without declaring a
            // positive type. More complex negations do not supply a safe hint.
            if expected != "enum"
                && let Some(negated) = schema.get("not").and_then(Value::as_object)
                && negated.keys().all(|key| {
                    matches!(
                        key.as_str(),
                        "type" | "title" | "description" | "default" | "examples" | "$comment"
                    )
                })
                && let Some(ty) = negated.get("type")
                && let Some(excluded) = ty.as_str().map(matches).or_else(|| {
                    ty.as_array()
                        .map(|types| types.iter().filter_map(Value::as_str).any(matches))
                })
            {
                constraint = Some(constraint.unwrap_or(true) && !excluded);
            }
            // Literal constraints can establish a type even without `type`.
            // Intersect them with explicit types so an enum cannot widen a
            // sibling constraint. Ambiguous unions still preserve strings.
            let literal_matches = |value: &Value| match expected {
                "string" => value.is_string(),
                "boolean" => value.is_boolean(),
                "null" => value.is_null(),
                "object" => value.is_object(),
                "array" => value.is_array(),
                "number" => value.is_number(),
                "integer" => {
                    value.is_i64()
                        || value.is_u64()
                        || value.as_f64().is_some_and(|number| number.fract() == 0.0)
                }
                _ => false,
            };
            if infer_literals && let Some(value) = schema.get("const") {
                constraint = Some(constraint.unwrap_or(true) && literal_matches(value));
            }
            if infer_literals && let Some(values) = schema.get("enum").and_then(Value::as_array) {
                constraint = Some(constraint.unwrap_or(true) && values.iter().any(literal_matches));
            }
            for key in ["anyOf", "oneOf"] {
                if let Some(options) = schema.get(key).and_then(Value::as_array) {
                    // Branches without known type constraints supply no hint.
                    if let Some(permits) = options
                        .iter()
                        .filter_map(|option| {
                            walker.type_constraint(option, expected, infer_literals)
                        })
                        .reduce(|left, right| left || right)
                    {
                        constraint = Some(constraint.unwrap_or(true) && permits);
                    }
                }
            }
            if let Some(branches) = schema.get("allOf").and_then(Value::as_array) {
                for branch in branches {
                    if let Some(permits) = walker.type_constraint(branch, expected, infer_literals)
                    {
                        constraint = Some(constraint.unwrap_or(true) && permits);
                    }
                }
            }
            constraint
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool_calling::arguments::{MAX_SCHEMA_DEPTH, MAX_SCHEMA_WORK};
    use serde_json::json;

    #[test]
    fn scalar_hints_preserve_explicit_and_composed_types() {
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
                "42",
            ),
            (
                json!({"allOf": [{"type": "integer"}, {"oneOf": [{"const": "auto"}, {"type": "integer"}]}]}),
                "42",
                "42",
            ),
            (
                json!({"type": "number", "anyOf": [{"type": "integer"}]}),
                "2.5",
                "\"2.5\"",
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
    fn parse_with_parameters(parameters: Value, raw: &str) -> Value {
        let tools = vec![ToolDefinition {
            name: "f".into(),
            parameters: Some(parameters),
        }];
        let parsed =
            parse_parameters("f", raw, &MiniMaxM3ParserConfig::default(), Some(&tools)).unwrap();
        ParsedValue::Object(parsed).to_value()
    }

    #[test]
    fn same_name_nested_parameters_end_at_matching_close_tag() {
        let multi_match_schema = json!({"type": "object", "properties": {
            "query": {"type": "string"},
            "fields": {"type": "array", "items": {"type": "string"}}
        }});
        let parameters = json!({"type": "object", "properties": {
            "query": {"type": "object", "properties": {
                "multi_match": multi_match_schema,
                "nested": {"type": "object", "properties": {
                    "path": {"type": "string"},
                    "query": {"type": "object", "properties": {"multi_match": multi_match_schema}}
                }}
            }},
            "size": {"type": "integer"}
        }});
        let multi_match = format!(
            "{TOK}<multi_match>{TOK}<query>connection timeout{TOK}</query>{TOK}<fields>\
             {TOK}<item>message{TOK}</item>{TOK}<item>error.reason{TOK}</item>{TOK}</fields>\
             {TOK}</multi_match>"
        );
        let size = format!("{TOK}<size>20{TOK}</size>");
        let matched = json!({"multi_match": {
            "query": "connection timeout", "fields": ["message", "error.reason"]
        }});
        for (body, expected) in [
            (
                format!("{TOK}<query>{multi_match}{TOK}</query>{size}"),
                json!({"query": matched, "size": 20}),
            ),
            (
                format!(
                    "{TOK}<query>{TOK}<nested>{TOK}<path>logs{TOK}</path>\
                     {TOK}<query>{multi_match}{TOK}</query>{TOK}</nested>{TOK}</query>{size}"
                ),
                json!({"query": {"nested": {"path": "logs", "query": matched}}, "size": 20}),
            ),
            // A missing outer close falls back to the first close tag, so `fields` is promoted.
            (
                format!("{TOK}<query>{multi_match}{size}"),
                json!({
                    "query": {"multi_match": {"query": "connection timeout"}},
                    "fields": {"item": ["message", "error.reason"]},
                    "size": 20
                }),
            ),
        ] {
            let tools = vec![ToolDefinition {
                name: "search_logs".into(),
                parameters: Some(parameters.clone()),
            }];
            let wire = format!("{TOK}<invoke name=\"search_logs\">{body}{TOK}</invoke>");
            let (_, arguments) =
                parse_minimax_m3_invoke(&wire, &MiniMaxM3ParserConfig::default(), &tools)
                    .unwrap()
                    .unwrap();
            let args: Value = serde_json::from_str(&arguments).unwrap();
            assert_eq!(args, expected);
        }
    }

    #[test]
    fn literal_entities_remain_argument_data() {
        let tools = vec![ToolDefinition {
            name: "inspect".into(),
            parameters: Some(json!({
                "properties": {"text": {"type": "string"}, "payload": {"type": "object"}}
            })),
        }];
        let raw = format!(
            "{TOK}<text>a &amp; b &lt;div&gt; &amp;quot;{TOK}</text>\
             {TOK}<payload>{{\"text\":\"&quot; &amp;\"}}{TOK}</payload>\
             {TOK}<note>&lt;b&gt; &#39;x&#39;{TOK}</note>"
        );
        let actual = parse_parameters(
            "inspect",
            &raw,
            &MiniMaxM3ParserConfig::default(),
            Some(&tools),
        )
        .unwrap();
        assert_eq!(
            ParsedValue::Object(actual).to_value(),
            json!({
                "text": "a &amp; b &lt;div&gt; &amp;quot;",
                "payload": {"text": "&quot; &amp;"},
                "note": "&lt;b&gt; &#39;x&#39;"
            })
        );
    }

    #[test]
    fn top_level_composed_parameters_keep_declared_types() {
        let tok = "]<]minimax[>[";
        let raw = format!(
            "{tok}<number>42{tok}</number>{tok}<ratio>0.5{tok}</ratio>\
             {tok}<flag>true{tok}</flag>{tok}<ids>[1, 2]{tok}</ids>\
             {tok}<name>42{tok}</name>{tok}<extra>7{tok}</extra>"
        );
        for union in ["oneOf", "anyOf", "allOf"] {
            let parameters = serde_json::json!({"type": "object", union: [
                {"properties": {
                    "number": {"type": "integer"},
                    "ratio": {"type": "number"},
                    "flag": {"type": "boolean"},
                    "ids": {"type": "array", "items": {"type": "integer"}}
                }, "required": ["number"]},
                {"properties": {"name": {"type": "string"}}, "required": ["name"]}
            ]});
            assert_eq!(
                parse_with_parameters(parameters, &raw),
                serde_json::json!({
                    "number": 42, "ratio": 0.5, "flag": true, "ids": [1, 2],
                    "name": "42", "extra": "7"
                }),
                "{union}"
            );
        }
    }

    #[test]
    fn top_level_one_of_branches_type_scalars_and_item_arrays() {
        let tok = "]<]minimax[>[";
        let parameters = serde_json::json!({"type": "object", "oneOf": [
            {"properties": {"number": {"type": "number"}}, "additionalProperties": false},
            {"properties": {"stringList": {"type": "array", "items": {"type": "string"}}},
             "additionalProperties": false},
            {"properties": {"numberList": {"type": "array", "items": {"type": "number"}}},
             "additionalProperties": false}
        ]});
        for (raw, expected) in [
            (
                format!("{tok}<number>42{tok}</number>"),
                serde_json::json!({"number": 42}),
            ),
            (
                format!(
                    "{tok}<stringList>{tok}<item>12{tok}</item>{tok}<item>34{tok}</item>{tok}</stringList>"
                ),
                serde_json::json!({"stringList": ["12", "34"]}),
            ),
            (
                format!(
                    "{tok}<numberList>{tok}<item>12{tok}</item>{tok}<item>34{tok}</item>{tok}</numberList>"
                ),
                serde_json::json!({"numberList": [12, 34]}),
            ),
        ] {
            assert_eq!(
                parse_with_parameters(parameters.clone(), &raw),
                expected,
                "{raw}"
            );
        }
    }

    #[test]
    fn top_level_composed_parameters_follow_nesting_and_refs() {
        let tok = "]<]minimax[>[";
        let parameters = serde_json::json!({
            "$defs": {"Count": {"properties": {"count": {"type": "integer"}}}},
            "properties": {"label": {"type": "string"}},
            "allOf": [{"anyOf": [
                {"$ref": "#/$defs/Count"},
                {"properties": {"limits": {"type": "object", "properties": {
                    "max": {"type": "integer"}
                }}}}
            ]}]
        });
        let raw = format!(
            "{tok}<label>7{tok}</label>{tok}<count>3{tok}</count>\
             {tok}<limits>{tok}<max>10{tok}</max>{tok}</limits>"
        );
        assert_eq!(
            parse_with_parameters(parameters, &raw),
            serde_json::json!({"label": "7", "count": 3, "limits": {"max": 10}})
        );
    }

    #[test]
    fn top_level_parameters_declared_by_several_variants_use_their_union() {
        let tok = "]<]minimax[>[";
        for (declarations, raw, expected) in [
            (
                [
                    serde_json::json!({"type": "integer"}),
                    serde_json::json!({"type": "boolean"}),
                ],
                "2",
                serde_json::json!(2),
            ),
            (
                [
                    serde_json::json!({"type": "integer"}),
                    serde_json::json!({"type": "string"}),
                ],
                "2",
                serde_json::json!("2"),
            ),
            (
                [
                    serde_json::json!({"type": "integer"}),
                    serde_json::json!({"type": "null"}),
                ],
                "null",
                serde_json::json!(null),
            ),
            (
                [
                    serde_json::json!({"type": "integer"}),
                    serde_json::json!({"type": "string"}),
                ],
                "null",
                serde_json::json!("null"),
            ),
        ] {
            let [first, second] = declarations;
            let parameters = serde_json::json!({"oneOf": [
                {"properties": {"value": first}},
                {"properties": {"value": second}}
            ]});
            let raw_value = format!("{tok}<value>{raw}{tok}</value>");
            assert_eq!(
                parse_with_parameters(parameters.clone(), &raw_value),
                serde_json::json!({"value": expected}),
                "{parameters}: {raw}"
            );
        }
    }

    #[test]
    fn root_and_all_of_parameters_take_precedence_over_variants() {
        let tok = "]<]minimax[>[";
        let variants = serde_json::json!([
            {"properties": {"value": {"type": "string"}}},
            {"properties": {"other": {"type": "string"}}}
        ]);
        let declaration = serde_json::json!({"properties": {"value": {"type": "integer"}}});
        let raw = format!("{tok}<value>2{tok}</value>");
        for parameters in [
            serde_json::json!({
                "properties": {"value": {"type": "integer"}},
                "oneOf": variants
            }),
            serde_json::json!({"allOf": [declaration], "anyOf": variants}),
        ] {
            assert_eq!(
                parse_with_parameters(parameters.clone(), &raw),
                serde_json::json!({"value": 2}),
                "{parameters}"
            );
        }
    }

    #[test]
    fn plain_parameter_maps_without_properties_stay_supported() {
        let tok = "]<]minimax[>[";
        let parameters = serde_json::json!({"count": {"type": "integer"}});
        let raw = format!("{tok}<count>3{tok}</count>{tok}<other>4{tok}</other>");
        assert_eq!(
            parse_with_parameters(parameters, &raw),
            serde_json::json!({"count": 3, "other": "4"})
        );
    }
}
