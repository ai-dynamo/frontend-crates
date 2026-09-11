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
    TagsWithSeparatorFormat, TriggeredTagsFormat,
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
            Value::Number(number) if number.is_i64() || number.is_u64() => Self::Integer,
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
}

fn resolve_local_ref<'a>(reference: &str, root: &'a Value) -> Option<&'a Value> {
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

fn single_xtml_type(schema: &Value, root: &Value) -> Option<&'static str> {
    let types = schema_types(schema, root, &HashSet::new());
    (types.len() == 1).then(|| types[0].xtml_name())
}

fn bounded_string_regex(schema: &Map<String, Value>) -> Option<String> {
    let max_len = schema.get("maxLength")?.as_u64()?;
    if max_len > 4096 {
        return None;
    }
    let min_len = schema
        .get("minLength")
        .and_then(Value::as_u64)
        .filter(|min| *min <= max_len)
        .unwrap_or(0);
    Some(format!("{STRING_ATOM}{{{min_len},{max_len}}}"))
}

fn argument_tag(
    key: &str,
    schema: &Value,
    root_defs: Option<&Map<String, Value>>,
) -> Option<TagFormat> {
    let schema_object = schema.as_object()?;
    let (json_type, xtml_type) = match schema_object.get("type").and_then(Value::as_str)? {
        "string" => ("string", "string"),
        "integer" => ("integer", "number"),
        "number" => ("number", "number"),
        "boolean" => ("boolean", "boolean"),
        "null" => ("null", "null"),
        "object" => ("object", "object"),
        "array" => ("array", "array"),
        _ => return None,
    };
    let begin = format!(
        "{OPEN}argument key=\"{}\" type=\"{xtml_type}\"{SEP}",
        escape_attr(key)
    );

    let content = if json_type == "string" {
        let enum_values = schema_object
            .get("enum")
            .and_then(Value::as_array)
            .cloned()
            .or_else(|| {
                schema_object
                    .get("const")
                    .and_then(Value::as_str)
                    .map(|value| vec![Value::String(value.to_string())])
            });
        if let Some(values) = enum_values.filter(|values| {
            !values.is_empty()
                && values.len() <= 256
                && values
                    .iter()
                    .all(|value| value.as_str().is_some_and(|string| !string.contains("<|")))
        }) {
            one_of(
                values
                    .iter()
                    .filter_map(Value::as_str)
                    .map(|value| {
                        Format::ConstString(ConstStringFormat {
                            value: value.to_string(),
                        })
                    })
                    .collect(),
            )
        } else if let Some(pattern) = bounded_string_regex(schema_object) {
            Format::Regex(RegexFormat { pattern })
        } else {
            Format::AnyText(AnyTextFormat {
                excludes: vec![CLOSE.to_string()],
            })
        }
    } else {
        let mut embedded = schema_object.clone();
        if let Some(root_defs) = root_defs {
            for (key, value) in root_defs {
                embedded.entry(key.clone()).or_insert_with(|| value.clone());
            }
        }
        Format::JsonSchema(JsonSchemaFormat {
            json_schema: Value::Object(embedded),
            style: JsonSchemaStyle::Json,
        })
    };

    Some(TagFormat {
        begin,
        content: Box::new(content),
        end: ARGUMENT_CLOSE.to_string(),
    })
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

fn root_definitions(parameters: &Map<String, Value>) -> Map<String, Value> {
    ["$defs", "definitions"]
        .into_iter()
        .filter_map(|key| {
            parameters
                .get(key)
                .and_then(Value::as_object)
                .map(|value| (key.to_string(), Value::Object(value.clone())))
        })
        .collect()
}

fn arguments_block(parameters: Option<&Value>) -> Format {
    let Some(parameters) = parameters.and_then(Value::as_object) else {
        return star(Format::Tag(permissive_argument_tag()));
    };
    let Some(properties) = parameters.get("properties").and_then(Value::as_object) else {
        return star(Format::Tag(permissive_argument_tag()));
    };
    if properties.is_empty() {
        return star(Format::Tag(permissive_argument_tag()));
    }

    let root_defs = root_definitions(parameters);
    let tags = properties
        .iter()
        .map(|(key, schema)| {
            Format::Tag(
                argument_tag(key, schema, Some(&root_defs)).unwrap_or_else(permissive_argument_tag),
            )
        })
        .collect();
    star(one_of(tags))
}

fn auto_argument_format(
    key: &str,
    schema: &Value,
    root: &Value,
    root_defs: &Map<String, Value>,
    require_nonempty_string: bool,
) -> Option<Format> {
    let xtml_type = single_xtml_type(schema, root)?;
    let content = if xtml_type == "string" {
        // K3 string values are raw rather than JSON quoted. Required string
        // arguments need at least one atom; optional strings may be empty when
        // their schema permits it.
        Format::Regex(RegexFormat {
            pattern: format!(
                "{STRING_ATOM}{}",
                if require_nonempty_string { "+" } else { "*" }
            ),
        })
    } else {
        // Non-string K3 argument bodies are JSON. Preserve the complete
        // property schema instead of using the string-only regex, and carry
        // root definitions so local references remain resolvable.
        let mut embedded = schema.as_object()?.clone();
        for (definition_key, definitions) in root_defs {
            embedded
                .entry(definition_key.clone())
                .or_insert_with(|| definitions.clone());
        }
        Format::JsonSchema(JsonSchemaFormat {
            json_schema: Value::Object(embedded),
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
}

fn optional_auto_arguments(
    properties: &Map<String, Value>,
    required_keys: &[&str],
    root: &Value,
    root_defs: &Map<String, Value>,
) -> Option<Format> {
    let arguments = properties
        .iter()
        .filter(|(key, _)| !required_keys.contains(&key.as_str()))
        .filter_map(|(key, schema)| {
            matches!(schema, Value::Bool(_) | Value::Object(_))
                .then(|| auto_argument_format(key, schema, root, root_defs, false))
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

fn auto_arguments_block(tool: &ToolDefinition, strict_schema: bool) -> Format {
    if !super::builder::uses_declared_tool_schema(tool, strict_schema) {
        return arguments_block(None);
    }

    let Some(parameters) = tool.parameters.as_ref().and_then(Value::as_object) else {
        return arguments_block(None);
    };
    let root = Value::Object(parameters.clone());
    let Some(properties) = parameters.get("properties").and_then(Value::as_object) else {
        return arguments_block(None);
    };
    let root_defs = root_definitions(parameters);
    let required = match parameters.get("required") {
        None => Vec::new(),
        Some(Value::Array(required)) if required.iter().all(|value| value.as_str().is_some()) => {
            required
                .iter()
                .filter_map(Value::as_str)
                .collect::<Vec<_>>()
        }
        Some(_) => return arguments_block(None),
    };
    let optional_arguments = optional_auto_arguments(properties, &required, &root, &root_defs);

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
            return arguments_block(None);
        };
        if !matches!(schema, Value::Bool(_) | Value::Object(_)) {
            return arguments_block(None);
        }
        let Some(argument) = auto_argument_format(key, schema, &root, &root_defs, true) else {
            return arguments_block(None);
        };
        arguments.push(argument);
    }
    canonical_required_arguments(arguments, optional_arguments)
}

fn auto_call_tag(tool: &ToolDefinition, strict_schema: bool) -> TagFormat {
    TagFormat {
        begin: format!("{OPEN}call tool=\"{}\" index=\"", escape_attr(&tool.name)),
        content: Box::new(Format::Sequence(SequenceFormat {
            elements: vec![
                Format::Regex(RegexFormat {
                    pattern: "[1-9][0-9]*".to_string(),
                }),
                Format::ConstString(ConstStringFormat {
                    value: format!("\"{SEP}"),
                }),
                auto_arguments_block(tool, strict_schema),
            ],
        })),
        end: CALL_CLOSE.to_string(),
    }
}

fn build_auto_structural_tag(
    tools: Vec<&ToolDefinition>,
    ctx: &ToolCallFormatBuildContext<'_>,
) -> StructuralTag {
    let strict_schema = ctx.strict_schema();
    let call_tags: Vec<_> = tools
        .into_iter()
        .map(|tool| auto_call_tag(tool, strict_schema))
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
    let suffix = Format::TriggeredTags(TriggeredTagsFormat {
        triggers: vec![TOOLS_OPEN.to_string()],
        tags: vec![tools_tag],
        at_least_one: false,
        stop_after_first: !parallel_tool_calls,
        excludes: vec![
            THINK_OPEN.to_string(),
            THINK_CLOSE.to_string(),
            CALL_OPEN.to_string(),
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
    // Match vLLM's K3 behavior: use the declared schema unless the caller
    // explicitly sets strict=false. Global strict mode overrides that opt-out.
    let parameters = if super::builder::uses_declared_tool_schema(tool, strict_schema) {
        tool.parameters.as_ref()
    } else {
        None
    };
    let begin = format!("{OPEN}call tool=\"{}\" index=\"", escape_attr(&tool.name));
    TagFormat {
        begin,
        content: Box::new(Format::Sequence(SequenceFormat {
            elements: vec![
                Format::Regex(RegexFormat {
                    pattern: "[0-9]+".to_string(),
                }),
                Format::ConstString(ConstStringFormat {
                    value: format!("\"{SEP}"),
                }),
                arguments_block(parameters),
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
    fn named_choice_is_mandatory_and_auto_uses_an_optional_trigger() {
        let tools = tools();
        let named = ToolChoice::Named("get_weather".to_string());
        let named_value =
            serde_json::to_value(build_kimi_k3(&context(&named, &tools)).unwrap().unwrap())
                .unwrap();
        assert_eq!(named_value["format"]["elements"][2]["type"], "tag");

        let auto = ToolChoice::Auto;
        let auto_value =
            serde_json::to_value(build_kimi_k3(&context(&auto, &tools)).unwrap().unwrap()).unwrap();
        assert_eq!(auto_value["format"]["type"], "triggered_tags");
        assert_eq!(auto_value["format"]["at_least_one"], false);
        assert_eq!(auto_value["format"]["triggers"], json!([TOOLS_OPEN]));
        assert_eq!(
            auto_value["format"]["excludes"],
            json!([THINK_OPEN, THINK_CLOSE, CALL_OPEN])
        );
    }

    #[test]
    fn auto_requires_nonempty_required_arguments_then_allows_optional_content() {
        let tools = tools();
        let choice = ToolChoice::Auto;
        let value =
            serde_json::to_value(build_kimi_k3(&context(&choice, &tools)).unwrap().unwrap())
                .unwrap();
        let calls = &value["format"]["tags"][0]["content"];
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
        assert_eq!(
            arguments["elements"][0]["content"]["pattern"],
            format!("{STRING_ATOM}+")
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
        let arguments = &value["format"]["tags"][0]["content"]["tags"][0]["content"]["elements"][2];
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
        let arguments = &value["format"]["tags"][0]["content"]["tags"][0]["content"]["elements"][2];
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
        let argument = &value["format"]["tags"][0]["content"]["tags"][0]["content"]["elements"][2]
            ["elements"][0];

        assert_eq!(argument["content"]["type"], "json_schema");
        assert_eq!(
            argument["content"]["json_schema"]["$ref"],
            "#/$defs/identifier"
        );
        assert_eq!(
            argument["content"]["json_schema"]["$defs"]["identifier"]["type"],
            "integer"
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
        let arguments = &value["format"]["tags"][0]["content"]["tags"][0]["content"]["elements"][2];

        assert_eq!(arguments["type"], "star");
        assert_eq!(arguments["content"]["type"], "or");
        assert_eq!(
            arguments["content"]["elements"].as_array().unwrap().len(),
            2
        );
        let alternatives = arguments["content"]["elements"].as_array().unwrap();
        assert_eq!(
            alternatives[0]["content"]["pattern"],
            format!("{STRING_ATOM}*")
        );
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
        let arguments = &value["format"]["tags"][0]["content"]["tags"][0]["content"]["elements"][2];

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
        let strict_arguments =
            &strict_value["format"]["tags"][0]["content"]["tags"][0]["content"]["elements"][2];

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
        assert_eq!(value["format"]["elements"][1]["type"], "triggered_tags");
    }

    #[test]
    fn named_choice_keeps_declared_argument_schema() {
        let tools = tools();
        let choice = ToolChoice::Named("get_weather".to_string());
        let value =
            serde_json::to_value(build_kimi_k3(&context(&choice, &tools)).unwrap().unwrap())
                .unwrap();
        let call = &value["format"]["elements"][2]["content"]["tags"][0];
        let argument_alternatives = &call["content"]["elements"][2]["content"]["elements"];
        assert!(
            argument_alternatives
                .to_string()
                .contains("key=\\\"city\\\"")
        );
        assert!(
            argument_alternatives
                .to_string()
                .contains("key=\\\"days\\\"")
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
