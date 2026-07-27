// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Structural-tag grammar for Kimi K3 XTML tool calls.

use serde_json::{Map, Value, json};

use super::{
    ARG_CLOSE, CALL_CLOSE, END_OF_MSG, JSON_CLOSE, JSON_OPEN, MESSAGE_CLOSE, OPEN, RESPONSE_CLOSE,
    RESPONSE_OPEN, SEP, THINK_CLOSE, TOOLS_CLOSE, TOOLS_OPEN,
};
use crate::Tool;
use crate::unified::{
    UnifiedStructuralTagBuilder, UnifiedToolCallFormatContext, UnifiedToolChoice,
};

pub static KIMI_K3_STRUCTURAL_TAG_BUILDER: KimiK3StructuralTagBuilder = KimiK3StructuralTagBuilder;

const XTML_TYPES: &[&str] = &["string", "number", "boolean", "null", "object", "array"];

/// Builds xgrammar-compatible Kimi K3 XTML constraints.
#[derive(Debug, Clone, Copy, Default)]
pub struct KimiK3StructuralTagBuilder;

impl UnifiedStructuralTagBuilder for KimiK3StructuralTagBuilder {
    fn build_tool_call_format(
        &self,
        ctx: &UnifiedToolCallFormatContext<'_>,
    ) -> anyhow::Result<Option<Value>> {
        let (tools, calls_required) = resolve_tools(ctx)?;
        let mut elements = response_prefix(ctx.starts_in_reasoning);

        if !tools.is_empty() {
            let tools_channel = tools_channel(
                &tools,
                ctx.strict_schema,
                ctx.parallel_tool_calls == Some(false),
            );
            elements.push(if calls_required {
                tools_channel
            } else {
                optional(tools_channel)
            });
        }
        elements.push(optional(const_string(MESSAGE_CLOSE)));

        Ok(Some(json!({
            "type": "structural_tag",
            "format": sequence(elements),
        })))
    }
}

fn resolve_tools<'a>(
    ctx: &'a UnifiedToolCallFormatContext<'a>,
) -> anyhow::Result<(Vec<&'a Tool>, bool)> {
    match ctx.tool_choice {
        UnifiedToolChoice::None => Ok((Vec::new(), false)),
        UnifiedToolChoice::Auto => Ok((ctx.tools.iter().collect(), false)),
        UnifiedToolChoice::Required => {
            anyhow::ensure!(
                !ctx.tools.is_empty(),
                "tool_choice is required but tools is empty"
            );
            Ok((ctx.tools.iter().collect(), true))
        }
        UnifiedToolChoice::Named(name) => {
            let tool = ctx
                .tools
                .iter()
                .find(|tool| tool.name == name)
                .ok_or_else(|| {
                    anyhow::anyhow!("tool named {name:?} in tool_choice is not present in tools")
                })?;
            Ok((vec![tool], true))
        }
    }
}

fn response_prefix(reasoning: bool) -> Vec<Value> {
    let mut elements = Vec::new();
    if reasoning {
        elements.push(tag(
            "",
            any_text_excluding(&[THINK_CLOSE, END_OF_MSG]),
            THINK_CLOSE,
        ));
        elements.push(const_string(RESPONSE_OPEN));
    } else {
        elements.push(optional(const_string(RESPONSE_OPEN)));
    }
    elements.push(tag(
        "",
        any_text_excluding(&[RESPONSE_CLOSE, TOOLS_OPEN, MESSAGE_CLOSE, END_OF_MSG]),
        RESPONSE_CLOSE,
    ));
    elements
}

fn tools_channel(tools: &[&Tool], strict_schema: bool, stop_after_first: bool) -> Value {
    let calls: Vec<Value> = tools
        .iter()
        .map(|tool| call_tag(tool, strict_schema))
        .collect();
    tag(
        TOOLS_OPEN,
        json!({
            "type": "tags_with_separator",
            "tags": calls,
            "separator": "",
            "at_least_one": true,
            "stop_after_first": stop_after_first,
        }),
        TOOLS_CLOSE,
    )
}

fn call_tag(tool: &Tool, strict_schema: bool) -> Value {
    let parameters = if strict_schema || tool.strict.unwrap_or(false) {
        tool.parameters.clone()
    } else {
        Value::Bool(true)
    };
    let body = format_or(vec![
        typed_arguments(&parameters),
        raw_json_arguments(&parameters),
    ]);
    tag(
        format!(
            "{OPEN}call tool=\"{}\" index=\"",
            escape_attr_value(&tool.name)
        ),
        sequence(vec![
            regex("[1-9][0-9]*"),
            const_string(format!("\"{SEP}")),
            body,
        ]),
        CALL_CLOSE,
    )
}

fn typed_arguments(parameters: &Value) -> Value {
    let Some(schema) = parameters.as_object() else {
        return if parameters == &Value::Bool(false) {
            const_string("")
        } else {
            star(permissive_argument())
        };
    };
    let Some(properties) = schema.get("properties").and_then(Value::as_object) else {
        return star(permissive_argument());
    };
    if properties.is_empty() {
        return star(permissive_argument());
    }

    let root_defs = root_definitions(schema);
    let arguments = properties
        .iter()
        .flat_map(|(key, schema)| argument_tags(key, schema, &root_defs))
        .collect::<Vec<_>>();
    let arguments = format_or(arguments);
    if schema
        .get("required")
        .and_then(Value::as_array)
        .is_some_and(|required| !required.is_empty())
    {
        plus(arguments)
    } else {
        star(arguments)
    }
}

fn argument_tags(key: &str, schema: &Value, root_defs: &Map<String, Value>) -> Vec<Value> {
    schema_types(schema)
        .into_iter()
        .map(|xtml_type| {
            let content = if xtml_type == "string" {
                string_argument_content(schema)
            } else {
                json_schema(attach_root_definitions(
                    &narrow_schema_type(schema, xtml_type),
                    root_defs,
                ))
            };
            tag(
                format!(
                    "{OPEN}argument key=\"{}\" type=\"{xtml_type}\"{SEP}",
                    escape_attr_value(key)
                ),
                content,
                ARG_CLOSE,
            )
        })
        .collect()
}

fn schema_types(schema: &Value) -> Vec<&'static str> {
    let Some(schema) = schema.as_object() else {
        return XTML_TYPES.to_vec();
    };
    let mut types = Vec::new();
    match schema.get("type") {
        Some(Value::String(value)) => push_schema_type(&mut types, value),
        Some(Value::Array(values)) => {
            for value in values.iter().filter_map(Value::as_str) {
                push_schema_type(&mut types, value);
            }
        }
        _ => {}
    }
    if types.is_empty()
        && let Some(value) = schema.get("const")
    {
        push_value_type(&mut types, value);
    }
    if types.is_empty()
        && let Some(values) = schema.get("enum").and_then(Value::as_array)
    {
        for value in values {
            push_value_type(&mut types, value);
        }
    }
    if types.is_empty() {
        XTML_TYPES.to_vec()
    } else {
        types
    }
}

fn push_schema_type(types: &mut Vec<&'static str>, json_type: &str) {
    let xtml_type = match json_type {
        "string" => Some("string"),
        "integer" | "number" => Some("number"),
        "boolean" => Some("boolean"),
        "null" => Some("null"),
        "object" => Some("object"),
        "array" => Some("array"),
        _ => None,
    };
    if let Some(xtml_type) = xtml_type
        && !types.contains(&xtml_type)
    {
        types.push(xtml_type);
    }
}

fn push_value_type(types: &mut Vec<&'static str>, value: &Value) {
    let xtml_type = match value {
        Value::String(_) => "string",
        Value::Number(_) => "number",
        Value::Bool(_) => "boolean",
        Value::Null => "null",
        Value::Object(_) => "object",
        Value::Array(_) => "array",
    };
    if !types.contains(&xtml_type) {
        types.push(xtml_type);
    }
}

fn narrow_schema_type(schema: &Value, xtml_type: &str) -> Value {
    let Some(mut schema) = schema.as_object().cloned() else {
        return schema.clone();
    };
    let json_type = if xtml_type == "number" && explicitly_integer_only(&schema) {
        "integer"
    } else {
        xtml_type
    };
    schema.insert("type".to_string(), Value::String(json_type.to_string()));
    Value::Object(schema)
}

fn explicitly_integer_only(schema: &Map<String, Value>) -> bool {
    match schema.get("type") {
        Some(Value::String(value)) => value == "integer",
        Some(Value::Array(values)) => {
            let values = values.iter().filter_map(Value::as_str).collect::<Vec<_>>();
            values.contains(&"integer") && !values.contains(&"number")
        }
        _ => false,
    }
}

fn string_argument_content(schema: &Value) -> Value {
    let Some(schema) = schema.as_object() else {
        return any_text_excluding(&[ARG_CLOSE, CALL_CLOSE]);
    };
    let values = schema
        .get("enum")
        .and_then(Value::as_array)
        .cloned()
        .or_else(|| schema.get("const").cloned().map(|value| vec![value]));
    let Some(values) = values else {
        return any_text_excluding(&[ARG_CLOSE, CALL_CLOSE]);
    };
    if values.is_empty()
        || values.len() > 256
        || values
            .iter()
            .any(|value| value.as_str().is_none_or(|value| value.contains("<|")))
    {
        return any_text_excluding(&[ARG_CLOSE, CALL_CLOSE]);
    }
    format_or(
        values
            .iter()
            .filter_map(Value::as_str)
            .map(const_string)
            .collect(),
    )
}

fn raw_json_arguments(parameters: &Value) -> Value {
    tag(
        format!("{JSON_OPEN} type=\"object\"{SEP}"),
        json_schema(parameters.clone()),
        JSON_CLOSE,
    )
}

fn permissive_argument() -> Value {
    let key = regex(r#"(?:[^<\"&]|&(?:amp|quot);|<[^|])*"#);
    let alternatives = XTML_TYPES
        .iter()
        .map(|xtml_type| {
            sequence(vec![
                key.clone(),
                const_string(format!("\" type=\"{xtml_type}\"{SEP}")),
                if *xtml_type == "string" {
                    any_text_excluding(&[ARG_CLOSE, CALL_CLOSE])
                } else {
                    json_schema(Value::Bool(true))
                },
            ])
        })
        .collect();
    tag(
        format!("{OPEN}argument key=\""),
        format_or(alternatives),
        ARG_CLOSE,
    )
}

fn root_definitions(schema: &Map<String, Value>) -> Map<String, Value> {
    ["$defs", "definitions"]
        .into_iter()
        .filter_map(|key| {
            schema
                .get(key)
                .map(|value| (key.to_string(), value.clone()))
        })
        .collect()
}

fn attach_root_definitions(schema: &Value, root_defs: &Map<String, Value>) -> Value {
    let Some(mut schema) = schema.as_object().cloned() else {
        return schema.clone();
    };
    for (key, value) in root_defs {
        schema.entry(key.clone()).or_insert_with(|| value.clone());
    }
    Value::Object(schema)
}

fn escape_attr_value(value: &str) -> String {
    value.replace('&', "&amp;").replace('"', "&quot;")
}

fn tag(begin: impl Into<String>, content: Value, end: impl Into<String>) -> Value {
    json!({
        "type": "tag",
        "begin": begin.into(),
        "content": content,
        "end": end.into(),
    })
}

fn sequence(elements: Vec<Value>) -> Value {
    json!({ "type": "sequence", "elements": elements })
}

fn optional(content: Value) -> Value {
    json!({ "type": "optional", "content": content })
}

fn star(content: Value) -> Value {
    json!({ "type": "star", "content": content })
}

fn plus(content: Value) -> Value {
    json!({ "type": "plus", "content": content })
}

fn const_string(value: impl Into<String>) -> Value {
    json!({ "type": "const_string", "value": value.into() })
}

fn regex(pattern: impl Into<String>) -> Value {
    json!({ "type": "regex", "pattern": pattern.into() })
}

fn any_text_excluding(excludes: &[&str]) -> Value {
    json!({ "type": "any_text", "excludes": excludes })
}

fn json_schema(schema: Value) -> Value {
    json!({
        "type": "json_schema",
        "json_schema": schema,
        "style": "json",
        "any_order": false,
        "max_whitespace_cnt": Value::Null,
    })
}

fn format_or(elements: Vec<Value>) -> Value {
    match elements.as_slice() {
        [element] => element.clone(),
        _ => json!({ "type": "or", "elements": elements }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool(name: &str, parameters: Value) -> Tool {
        Tool {
            name: name.to_string(),
            description: None,
            parameters,
            strict: Some(true),
        }
    }

    #[test]
    fn required_format_contains_native_xtml_channels() {
        let tools = [tool(
            "get_weather",
            json!({
                "type": "object",
                "properties": {
                    "city": { "type": "string" },
                    "days": { "type": "integer" }
                },
                "required": ["city"]
            }),
        )];
        let tag = KimiK3StructuralTagBuilder
            .build_tool_call_format(&UnifiedToolCallFormatContext {
                tool_choice: UnifiedToolChoice::Required,
                tools: &tools,
                parallel_tool_calls: Some(false),
                strict_schema: true,
                starts_in_reasoning: true,
            })
            .unwrap()
            .unwrap();
        let encoded = serde_json::to_string(&tag).unwrap();

        assert!(encoded.contains("close|>think"));
        assert!(encoded.contains("open|>response"));
        assert!(encoded.contains("open|>tools"));
        assert!(encoded.contains("tool=\\\"get_weather"));
        assert!(encoded.contains("type=\\\"string"));
        assert!(encoded.contains("\"stop_after_first\":true"));
    }

    #[test]
    fn named_choice_filters_tools() {
        let tools = [
            tool("search", json!({ "type": "object" })),
            tool("lookup", json!({ "type": "object" })),
        ];
        let tag = KimiK3StructuralTagBuilder
            .build_tool_call_format(&UnifiedToolCallFormatContext {
                tool_choice: UnifiedToolChoice::Named("lookup"),
                tools: &tools,
                parallel_tool_calls: None,
                strict_schema: true,
                starts_in_reasoning: false,
            })
            .unwrap()
            .unwrap();
        let encoded = serde_json::to_string(&tag).unwrap();

        assert!(encoded.contains("lookup"));
        assert!(!encoded.contains("search"));
    }

    #[test]
    fn auto_without_tools_keeps_response_grammar() {
        let tag = KimiK3StructuralTagBuilder
            .build_tool_call_format(&UnifiedToolCallFormatContext {
                tool_choice: UnifiedToolChoice::Auto,
                tools: &[],
                parallel_tool_calls: None,
                strict_schema: false,
                starts_in_reasoning: false,
            })
            .unwrap()
            .unwrap();
        let encoded = serde_json::to_string(&tag).unwrap();

        assert!(encoded.contains("response"));
        assert!(!encoded.contains(r#""begin":"<|open|>tools"#));
    }
}
