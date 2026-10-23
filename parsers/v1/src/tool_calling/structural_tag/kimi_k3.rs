// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::builder::{
    ToolCallFormatBuildContext, resolve_tools_to_include, uses_declared_tool_schema,
};
use super::format::*;
use crate::tool_calling::ToolChoice;
#[cfg(test)]
use dynamo_structural_tag::kimi_k3::*;

pub(crate) fn build_kimi_k3(
    ctx: &ToolCallFormatBuildContext<'_>,
) -> anyhow::Result<Option<StructuralTag>> {
    let (tools, at_least_one) = resolve_tools_to_include(ctx)?;
    if tools.is_empty() {
        return Ok(None);
    }
    let relaxed = serde_json::Value::Bool(true);
    let selected: Vec<_> = tools
        .iter()
        .map(|tool| dynamo_structural_tag::Tool {
            name: &tool.name,
            parameters: if uses_declared_tool_schema(tool, ctx.strict_schema()) {
                tool.parameters.as_ref().unwrap_or(&relaxed)
            } else {
                &relaxed
            },
        })
        .collect();
    if matches!(ctx.tool_choice, ToolChoice::Auto) {
        return dynamo_structural_tag::kimi_k3::build_auto(&selected, ctx.stop_after_first())
            .map(Some);
    }
    let response = if matches!(ctx.tool_choice, ToolChoice::Named(_)) {
        Format::ConstString(ConstStringFormat {
            value: String::new(),
        })
    } else {
        Format::AnyText(AnyTextFormat {
            excludes: vec![
                dynamo_structural_tag::kimi_k3::OPEN.to_string(),
                dynamo_structural_tag::kimi_k3::CLOSE.to_string(),
            ],
        })
    };
    Ok(Some(dynamo_structural_tag::kimi_k3::build(
        &selected,
        at_least_one,
        ctx.stop_after_first(),
        response,
        false,
    )?))
}

#[cfg(test)]
mod tests {
    // Exceed the shared resolver depth limit without exposing its internals.
    const REF_CHAIN_LENGTH: usize = 32;
    use serde_json::{Map, json};

    use super::*;
    use crate::tool_calling::structural_tag::builder::{
        StructuralTagBuilder, StructuralTagSchemaMode,
    };
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
        let tools_tag = &value["format"]["elements"][3];
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

        let response_body = &value["format"]["elements"][1];
        assert_eq!(response_body["type"], "const_string");
        assert_eq!(response_body["value"], "");
    }

    #[test]
    fn required_choice_response_text_reserves_xtml_controls() {
        let tools = tools();
        let choice = ToolChoice::Required;
        let value =
            serde_json::to_value(build_kimi_k3(&context(&choice, &tools)).unwrap().unwrap())
                .unwrap();
        let response_body = &value["format"]["elements"][1];
        assert_eq!(response_body["type"], "any_text");
        assert_eq!(response_body["excludes"], json!([OPEN, CLOSE]));
    }

    #[test]
    fn named_choice_is_mandatory_and_auto_uses_an_optional_tools_suffix() {
        let tools = tools();
        let named = ToolChoice::Named("get_weather".to_string());
        let named_value =
            serde_json::to_value(build_kimi_k3(&context(&named, &tools)).unwrap().unwrap())
                .unwrap();
        assert_eq!(named_value["format"]["elements"][3]["type"], "tag");

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
    fn impossible_required_argument_errors_for_every_tool_choice() {
        let mut tools = vec![ToolDefinition {
            name: "test".to_string(),
            parameters: Some(json!({
                "type": "object",
                "properties": {"value": {"type": "string", "enum": ["a"], "minLength": 2}},
                "required": ["value"]
            })),
            strict: None,
        }];
        for choice in [
            ToolChoice::Auto,
            ToolChoice::Required,
            ToolChoice::Named("test".into()),
        ] {
            let error = StructuralTagBuilder::KimiK3
                .build_tool_call_format(&context(&choice, &tools))
                .unwrap_err();
            let message = format!("{error:#}");
            assert!(
                message.contains("test") && message.contains("value"),
                "{message}"
            );
        }
        tools[0].strict = Some(false);
        assert!(
            StructuralTagBuilder::KimiK3
                .build_tool_call_format(&context(&ToolChoice::Auto, &tools))
                .unwrap()
                .is_some()
        );
    }

    #[test]
    fn unsupported_optional_intersection_returns_an_error() {
        let tools = vec![ToolDefinition {
            name: "test".to_string(),
            parameters: Some(json!({
                "type": "object",
                "properties": {"value": {"allOf": [
                    {"type": "string", "pattern": "^s"},
                    {"pattern": "e$"}
                ]}}
            })),
            strict: None,
        }];
        assert!(
            StructuralTagBuilder::KimiK3
                .build_tool_call_format(&context(&ToolChoice::Auto, &tools))
                .is_err()
        );
    }

    #[test]
    fn impossible_optional_string_enum_does_not_emit_empty_or() {
        let root = json!({
            "type": "object",
            "properties": {"value": {"type": "string", "enum": ["a"], "minLength": 2}}
        });
        let tools = vec![ToolDefinition {
            name: "optional".to_string(),
            parameters: Some(root),
            strict: None,
        }];
        let choice = ToolChoice::Auto;
        let tag = serde_json::to_value(build_kimi_k3(&context(&choice, &tools)).unwrap().unwrap())
            .unwrap();
        assert!(!tag.to_string().contains("\"elements\":[]"));
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
        assert_eq!(arguments[1]["content"]["pattern"], r"(?:[^<]|<[^|]){0,8}");
        assert_eq!(arguments[2]["content"]["pattern"], "(?:item-[0-9]+)");
        assert_eq!(arguments[3]["content"]["pattern"], r"(?:[^<]|<[^|]){0,0}");
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
        assert_eq!(
            schema["properties"]["next"]["$ref"],
            "#/$defs/__dynamo_root/$defs/node"
        );
        assert!(
            schema
                .pointer(
                    schema["properties"]["next"]["$ref"]
                        .as_str()
                        .unwrap()
                        .strip_prefix('#')
                        .unwrap()
                )
                .is_some()
        );
        assert_eq!(schema["$defs"]["node"]["type"], "object");
    }

    #[test]
    fn unresolved_string_references_return_errors() {
        let mut definitions = Map::new();
        for index in 0..REF_CHAIN_LENGTH {
            definitions.insert(
                format!("step_{index}"),
                json!({"$ref": format!("#/$defs/step_{}", index + 1)}),
            );
        }
        definitions.insert(
            format!("step_{REF_CHAIN_LENGTH}"),
            json!({"type": "string", "enum": ["safe"]}),
        );
        let schemas = [
            json!({"type": "string", "allOf": [{"$ref": "#/$defs/step_0"}]}),
            json!({
                "type": "string",
                "$id": "urn:example:scoped",
                "$defs": {"allowed": {"type": "string", "enum": ["safe"]}},
                "allOf": [{"$ref": "#/$defs/allowed"}]
            }),
        ];
        for schema in schemas {
            for required in [json!([]), json!(["value"])] {
                let tools = vec![ToolDefinition {
                    name: "test".into(),
                    parameters: Some(json!({
                        "type": "object",
                        "$defs": definitions,
                        "properties": {"value": schema},
                        "required": required
                    })),
                    strict: None,
                }];
                let error = StructuralTagBuilder::KimiK3
                    .build_tool_call_format(&context(&ToolChoice::Auto, &tools))
                    .unwrap_err();
                let message = format!("{error:#}");
                assert!(
                    message.contains("value") && message.contains("string"),
                    "{message}"
                );
            }
        }
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
    fn string_enum_intersects_length_and_pattern_constraints() {
        let schema = json!({
            "allOf": [
                {"type": "string", "enum": ["x", "code-42", "other"]},
                {"minLength": 3, "pattern": "^code-[0-9]+$"}
            ]
        });
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
    fn auto_thinking_has_one_reasoning_prefix() {
        let tools = tools();
        let choice = ToolChoice::Auto;
        let ctx = ToolCallFormatBuildContext {
            tool_choice: &choice,
            tools: &tools,
            parallel_tool_calls: None,
            schema_mode: StructuralTagSchemaMode::Auto,
            starts_in_reasoning: true,
        };
        let value = StructuralTagBuilder::KimiK3
            .build_tool_call_format(&ctx)
            .unwrap()
            .unwrap();

        assert_eq!(value["format"]["type"], "sequence");
        assert_eq!(value["format"]["elements"][0]["type"], "tag");
        assert_eq!(value["format"]["elements"][0]["end"], THINK_CLOSE);
        assert_eq!(value["format"]["elements"][1]["type"], "sequence");
        assert_eq!(
            value["format"]["elements"][1]["elements"][0]["type"],
            "any_text"
        );
    }

    #[test]
    fn named_choice_enforces_required_and_optional_argument_schemas() {
        let tools = tools();
        let choice = ToolChoice::Named("get_weather".to_string());
        let value =
            serde_json::to_value(build_kimi_k3(&context(&choice, &tools)).unwrap().unwrap())
                .unwrap();
        let call = &value["format"]["elements"][3]["content"]["tags"][0];
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
        let call = &value["format"]["elements"][3]["content"]["tags"][0];
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
            value["format"]["elements"][3]["content"]["stop_after_first"],
            true
        );
    }
}
