// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Kimi K3 XTML structural-tag generation.
//!
//! K3 does not emit the generic JSON shape used by legacy forced-tool guided
//! decoding. Tool calls live in a native XTML `tools` channel with one or more
//! nested `call` and typed `argument` elements. This builder mirrors that wire
//! format so named and required tool choices can be constrained without
//! changing what the K3 parser expects.

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
const TOOLS_OPEN: &str = "<|open|>tools<|sep|>";
const TOOLS_CLOSE: &str = "<|close|>tools<|sep|>";
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

    let root_defs: Map<String, Value> = ["$defs", "definitions"]
        .into_iter()
        .filter_map(|key| {
            parameters
                .get(key)
                .and_then(Value::as_object)
                .map(|value| (key.to_string(), Value::Object(value.clone())))
        })
        .collect();
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

fn call_tag(tool: &ToolDefinition, strict_schema: bool) -> TagFormat {
    // Match vLLM's K3 behavior: use the declared schema unless the caller
    // explicitly sets strict=false. Global strict mode overrides that opt-out.
    let parameters = if strict_schema || tool.strict != Some(false) {
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
    fn non_named_choices_keep_the_existing_response_body() {
        let tools = tools();

        for choice in [ToolChoice::Auto, ToolChoice::Required] {
            let value =
                serde_json::to_value(build_kimi_k3(&context(&choice, &tools)).unwrap().unwrap())
                    .unwrap();
            let response_body = &value["format"]["elements"][1]["content"];

            assert_eq!(
                response_body["type"], "any_text",
                "{choice:?} must retain response text"
            );
            assert_eq!(
                response_body["excludes"],
                json!([]),
                "{choice:?} must retain the existing unrestricted response body"
            );
        }
    }

    #[test]
    fn named_choice_is_mandatory_and_auto_is_optional() {
        let tools = tools();
        let named = ToolChoice::Named("get_weather".to_string());
        let named_value =
            serde_json::to_value(build_kimi_k3(&context(&named, &tools)).unwrap().unwrap())
                .unwrap();
        assert_eq!(named_value["format"]["elements"][2]["type"], "tag");

        let auto = ToolChoice::Auto;
        let auto_value =
            serde_json::to_value(build_kimi_k3(&context(&auto, &tools)).unwrap().unwrap()).unwrap();
        assert_eq!(auto_value["format"]["elements"][2]["type"], "optional");
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
