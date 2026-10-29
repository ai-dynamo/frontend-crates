// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! dynamo-llm's tests of the moved code, at ai-dynamo/dynamo `37f2f0a712`: all of
//! `lib/llm/src/protocols/openai/tools.rs`, and the tests in
//! `lib/llm/src/preprocessor/tool_choice.rs` that call `guided_tool_constraint` or
//! `installed_json_constraint`. The tests there that build an `OpenAIPreprocessor` and
//! call `apply_tool_choice_guided_decoding` stay in dynamo-llm.

use super::*;
use dynamo_protocols::types::{ChatCompletionToolChoiceOption, ChatCompletionToolType};

fn sample_tools() -> Vec<ChatCompletionTool> {
    vec![
        ChatCompletionTool {
            r#type: ChatCompletionToolType::Function,
            function: FunctionObject {
                name: "add_numbers".to_string(),
                description: Some("Add two integers".to_string()),
                parameters: Some(json!({
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"},
                    },
                    "required": ["a", "b"],
                })),
                strict: None,
                defer_loading: None,
            },
            custom: None,
            defer_loading: None,
        },
        ChatCompletionTool {
            r#type: ChatCompletionToolType::Function,
            function: FunctionObject {
                name: "get_weather".to_string(),
                description: Some("Get weather".to_string()),
                parameters: Some(json!({
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location", "unit"],
                })),
                strict: None,
                defer_loading: None,
            },
            custom: None,
            defer_loading: None,
        },
    ]
}

fn zero_arg_tool(parameters: Value) -> Vec<ChatCompletionTool> {
    vec![ChatCompletionTool {
        r#type: ChatCompletionToolType::Function,
        function: FunctionObject {
            name: "get_server_time".to_string(),
            description: Some("Get the current server time.".to_string()),
            parameters: Some(parameters),
            strict: None,
            defer_loading: None,
        },
        custom: None,
        defer_loading: None,
    }]
}

fn named_choice(name: &str) -> ChatCompletionToolChoiceOption {
    ChatCompletionToolChoiceOption::Named(dynamo_protocols::types::ChatCompletionNamedToolChoice {
        r#type: ChatCompletionToolType::Function,
        function: dynamo_protocols::types::FunctionName {
            name: name.to_string(),
        },
    })
}

/// GH-13789: a named choice on a tool whose schema admits only `{}` needs an exact regex.
/// JSON-schema whitespace can otherwise run to `max_tokens` before the closing brace.
#[test]
fn named_choice_on_closed_zero_arg_tool_uses_exact_regex() {
    let tools = zero_arg_tool(json!({
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": false,
    }));
    let guidance = get_tool_choice_guidance_from_tools(
        Some(&named_choice("get_server_time")),
        Some(&tools),
        None,
    )
    .expect("guidance");
    assert_eq!(
        guidance,
        Some(ToolChoiceGuidance::Regex(r"\{\}".to_string())),
        "a schema admitting only the empty object needs an exact constraint"
    );
}

/// The escape hatch is deliberately narrow. Anything that admits more than `{}` keeps
/// its constraint, because the grammar then always has a required token to emit.
#[test]
fn named_choice_keeps_constraint_for_schemas_admitting_more_than_empty() {
    let open = json!({"type": "object", "properties": {}});
    let bare = json!({"type": "object"});
    let with_property = json!({
        "type": "object",
        "properties": {"note": {"type": "string"}},
        "required": [],
        "additionalProperties": false,
    });
    let unknown_keyword = json!({
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": false,
        "patternProperties": {},
    });

    for parameters in [open, bare, with_property, unknown_keyword] {
        let tools = zero_arg_tool(parameters.clone());
        let schema =
            get_json_schema_from_tools(Some(&named_choice("get_server_time")), Some(&tools), None)
                .expect("schema");
        assert_eq!(
            schema,
            Some(parameters.clone()),
            "constraint should survive for {parameters}"
        );
    }
}

#[test]
fn named_choice_uses_regex_for_zero_property_bounds_and_annotations() {
    for (key, value) in [
        ("minProperties", json!(0)),
        ("maxProperties", json!(1)),
        ("examples", json!([{}])),
    ] {
        let tools = zero_arg_tool(json!({
            "type": "object",
            "properties": {},
            "additionalProperties": false,
            key: value,
        }));
        assert_eq!(
            get_tool_choice_guidance_from_tools(
                Some(&named_choice("get_server_time")),
                Some(&tools),
                None,
            )
            .expect("guidance"),
            Some(ToolChoiceGuidance::Regex(r"\{\}".to_string()))
        );
    }
}

/// `tool_choice=required` wraps every tool with a required `name` key, so it can never
/// stall on whitespace and must keep its constraint even for a zero-argument tool.
#[test]
fn required_choice_on_closed_zero_arg_tool_keeps_constraint() {
    let tools = zero_arg_tool(json!({
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": false,
    }));
    let schema = get_json_schema_from_tools(
        Some(&ChatCompletionToolChoiceOption::Required),
        Some(&tools),
        None,
    )
    .expect("schema")
    .expect("required always installs a constraint");

    let item = &schema["items"]["anyOf"][0];
    assert_eq!(item["properties"]["name"]["enum"][0], "get_server_time");
    assert_eq!(item["required"], json!(["name", "parameters"]));
}

#[test]
fn named_choice_returns_parameters() {
    let tools = sample_tools();
    let tool_choice = ChatCompletionToolChoiceOption::Named(
        dynamo_protocols::types::ChatCompletionNamedToolChoice {
            r#type: ChatCompletionToolType::Function,
            function: dynamo_protocols::types::FunctionName {
                name: "get_weather".to_string(),
            },
        },
    );
    let schema =
        get_json_schema_from_tools(Some(&tool_choice), Some(&tools), None).expect("schema");

    assert_eq!(
        schema.unwrap(),
        json!({
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location", "unit"],
        })
    );
}

#[test]
fn required_choice_builds_any_of_schema() {
    let tools = sample_tools();
    let schema = get_json_schema_from_tools(
        Some(&ChatCompletionToolChoiceOption::Required),
        Some(&tools),
        None,
    )
    .expect("schema");

    let schema = schema.expect("required schema");
    assert_eq!(schema["type"], "array");
    assert_eq!(schema["minItems"], 1);
    assert!(schema["items"]["anyOf"].is_array());
    assert!(schema.get("$defs").is_none());

    let any_of = schema["items"]["anyOf"].as_array().unwrap();
    assert_eq!(any_of.len(), 2);
    assert_eq!(
        any_of[0]["properties"]["name"],
        json!({"type": "string", "enum": ["add_numbers"]})
    );
}

#[test]
fn missing_tool_errors() {
    let tools = sample_tools();
    let tool_choice = ChatCompletionToolChoiceOption::Named(
        dynamo_protocols::types::ChatCompletionNamedToolChoice {
            r#type: ChatCompletionToolType::Function,
            function: dynamo_protocols::types::FunctionName {
                name: "unknown".to_string(),
            },
        },
    );
    let err = get_json_schema_from_tools(Some(&tool_choice), Some(&tools), None).unwrap_err();
    assert_eq!(err, ToolChoiceError::ToolNotFound("unknown".to_string()));
}

#[test]
fn required_choice_preserves_local_references_without_defs() {
    for (parameters, reference_path, target) in [
        (
            json!({
                "type": "object",
                "properties": {"value": {"$ref": "#/definitions/value"}},
                "definitions": {"value": {"type": "string"}}
            }),
            "/properties/value/$ref",
            json!({"type": "string"}),
        ),
        (
            json!({
                "type": "object",
                "properties": {
                    "value": {"type": "string"},
                    "alias": {"$ref": "#/properties/value"}
                }
            }),
            "/properties/alias/$ref",
            json!({"type": "string"}),
        ),
        (
            json!({"type": "object", "properties": {"child": {"$ref": "#"}}}),
            "/properties/child/$ref",
            json!({
                "type": "object",
                "properties": {"child": {"$ref": "#/$defs/tool_0"}}
            }),
        ),
    ] {
        let mut tools = sample_tools();
        tools.truncate(1);
        tools[0].function.parameters = Some(parameters);
        let schema = build_required_schema(&tools, None).unwrap();
        let parameters = schema.pointer("/$defs/tool_0").expect("parameter document");
        let reference = parameters
            .pointer(reference_path)
            .unwrap()
            .as_str()
            .unwrap();
        assert_eq!(schema.pointer(&reference[1..]), Some(&target));
    }
}

#[test]
fn required_choice_isolates_conflicting_definitions() {
    let mut tools = sample_tools();
    for (tool, kind) in tools.iter_mut().zip(["string", "number"]) {
        tool.function.parameters = Some(json!({
            "type": "object",
            "properties": {"value": {"$ref": "#/$defs/shared"}},
            "$defs": {"shared": {"type": kind}}
        }));
    }
    let schema = build_required_schema(&tools, None).expect("independent schemas");
    for (index, kind) in ["string", "number"].into_iter().enumerate() {
        let root = format!("#/$defs/tool_{index}");
        assert_eq!(
            schema["items"]["anyOf"][index]["properties"]["parameters"]["$ref"],
            root
        );
        let parameters = schema.pointer(&root[1..]).expect("tool schema");
        assert_eq!(
            parameters["properties"]["value"]["$ref"],
            format!("{root}/$defs/shared")
        );
        assert_eq!(parameters["$defs"]["shared"]["type"], kind);
    }
    assert_eq!(
        tools[0].function.parameters.as_ref().unwrap()["properties"]["value"]["$ref"],
        "#/$defs/shared"
    );
}

#[test]
fn required_choice_rebases_only_document_local_schema_references() {
    let mut tools = sample_tools();
    tools.truncate(1);
    tools[0].function.parameters = Some(json!({
        "type": "object",
        "properties": {
            "value": {"$ref": "#/$defs/a~1b~0c"},
            "child": {"$ref": "#"},
            "encoded": {"$ref": "#%2F$defs%2Fa~1b~0c"},
            "remote": {"$ref": "urn:example:remote"},
            "relative": {"$ref": "a%2Fremote"},
            "scoped": {"$id": "urn:example:scoped", "$ref": "#/$defs/local", "$defs": {"local": {"type": "string"}}},
            "literal": {"const": {"$ref": "#/$defs/a~1b~0c"}}
        },
        "$defs": {"a/b~c": {"anyOf": [{"type": "string"}, {"$ref": "#/$defs/a~1b~0c"}]}},
        "default": {"$ref": "#"}
    }));
    let schema = build_required_schema(&tools, None).unwrap();
    let parameters = &schema["$defs"]["tool_0"];
    assert_eq!(
        parameters["properties"]["value"]["$ref"],
        "#/$defs/tool_0/$defs/a~1b~0c"
    );
    assert_eq!(parameters["properties"]["child"]["$ref"], "#/$defs/tool_0");
    assert_eq!(
        parameters["properties"]["encoded"]["$ref"],
        "#/$defs/tool_0%2F$defs%2Fa~1b~0c"
    );
    assert_eq!(
        parameters["$defs"]["a/b~c"]["anyOf"][1]["$ref"],
        "#/$defs/tool_0/$defs/a~1b~0c"
    );
    assert_eq!(
        parameters["properties"]["remote"]["$ref"],
        "urn:example:remote"
    );
    assert_eq!(parameters["properties"]["relative"]["$ref"], "a%2Fremote");
    assert_eq!(parameters["properties"]["scoped"]["$ref"], "#/$defs/local");
    assert_eq!(
        parameters["properties"]["literal"]["const"]["$ref"],
        "#/$defs/a~1b~0c"
    );
    assert_eq!(parameters["default"]["$ref"], "#");
}

#[test]
fn required_schema_is_unbounded_by_default() {
    let tools = sample_tools();
    let schema = build_required_schema(&tools, None).expect("schema");
    assert_eq!(schema["minItems"], json!(1));
    assert!(
        schema.get("maxItems").is_none(),
        "parallel calls stay unbounded unless the request disables them"
    );
    let schema = build_required_schema(&tools, Some(true)).expect("schema");
    assert!(schema.get("maxItems").is_none());
}

/// `parallel_tool_calls: false` must constrain GENERATION, not just be filtered
/// downstream - otherwise the extra calls are still produced, still cost tokens,
/// and are still observable upstream of the HTTP filter.
#[test]
fn required_schema_caps_at_one_when_parallel_calls_are_disabled() {
    let tools = sample_tools();
    let schema = build_required_schema(&tools, Some(false)).expect("schema");
    assert_eq!(schema["minItems"], json!(1));
    assert_eq!(
        schema["maxItems"],
        json!(1),
        "parallel_tool_calls=false must cap the array at one element"
    );
}

// The tests below are from `lib/llm/src/preprocessor/tool_choice.rs`. Dynamo's
// request carries its `guided_*` fields and its tool validation; here neither
// applies, so each call passes `false` and a validator that accepts every tool.

fn request(extra: Value) -> CreateChatCompletionRequest {
    let mut value = json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "test"}]
    });
    value
        .as_object_mut()
        .expect("base request is an object")
        .extend(extra.as_object().expect("extra is an object").clone());
    serde_json::from_value(value).expect("request must deserialize")
}

fn tools() -> Value {
    json!([{
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"]
            }
        }
    }])
}

/// dynamo-llm's `guided_tool_constraint(request, tool_call_parser, reasoning_parser,
/// uses_structural_tag)`, with no `guided_*` fields and no tool validation.
fn constraint(
    request: &CreateChatCompletionRequest,
    tool_call_parser: Option<&str>,
    reasoning_parser: Option<&str>,
    uses_structural_tag: bool,
) -> Result<GuidedToolConstraint, InvalidRequest> {
    guided_tool_constraint(
        request,
        false,
        tool_call_parser,
        reasoning_parser,
        uses_structural_tag,
        |_| Ok(()),
    )
}

#[test]
fn only_structural_tag_reports_a_structural_tag() {
    assert!(GuidedToolConstraint::StructuralTag.uses_structural_tag());
    assert!(!GuidedToolConstraint::None.uses_structural_tag());
    assert!(!GuidedToolConstraint::GuidedJsonRequired.uses_structural_tag());
    assert!(
        !GuidedToolConstraint::GuidedJsonNamed {
            tool_name: "get_weather".to_string(),
        }
        .uses_structural_tag()
    );
}

#[test]
fn required_reports_the_array_shape() {
    assert_eq!(
        installed_json_constraint(&ChatCompletionToolChoiceOption::Required),
        GuidedToolConstraint::GuidedJsonRequired
    );
}

#[test]
fn named_carries_the_tool_name_from_the_request() {
    assert_eq!(
        installed_json_constraint(&named_choice("get_weather")),
        GuidedToolConstraint::GuidedJsonNamed {
            tool_name: "get_weather".to_string(),
        }
    );
}

#[test]
fn named_closed_zero_arg_tool_keeps_the_named_parser_constraint() {
    let request = request(json!({
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_server_time",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false
                }
            }
        }],
        "tool_choice": {
            "type": "function",
            "function": {"name": "get_server_time"}
        }
    }));

    assert_eq!(
        constraint(&request, None, None, false).expect("constraint is valid"),
        GuidedToolConstraint::GuidedJsonNamed {
            tool_name: "get_server_time".to_string(),
        }
    );
}

#[test]
fn unforced_choices_install_no_tool_constraint() {
    assert_eq!(
        installed_json_constraint(&ChatCompletionToolChoiceOption::Auto),
        GuidedToolConstraint::None
    );
    assert_eq!(
        installed_json_constraint(&ChatCompletionToolChoiceOption::None),
        GuidedToolConstraint::None
    );
}

#[test]
fn native_format_forced_choices_install_no_compatibility_json_constraint() {
    let requests = [
        request(json!({
            "tools": tools(),
            "tool_choice": "required"
        })),
        request(json!({
            "tools": tools(),
            "tool_choice": {
                "type": "function",
                "function": {"name": "get_weather"}
            }
        })),
    ];

    for request in &requests {
        for (tool_call_parser, reasoning_parser) in [
            (Some("kimi_k2"), None),
            (None, Some("kimi_k2")),
            (Some("kimi_k3"), None),
            (None, Some("kimi-k3")),
        ] {
            assert_eq!(
                constraint(request, tool_call_parser, reasoning_parser, false)
                    .expect("native-format choice is valid"),
                GuidedToolConstraint::None,
            );
        }
    }
}

#[test]
fn kimi_k3_required_without_tools_is_rejected_before_the_xtml_return() {
    for extra in [
        json!({"tool_choice": "required"}),
        json!({"tool_choice": "required", "tools": []}),
    ] {
        let request = request(extra);
        let result = constraint(&request, Some("kimi_k3"), None, false);
        assert!(
            result.is_err(),
            "Kimi K3 required must reject both missing and empty tools"
        );
    }
}

#[test]
fn kimi_k3_forced_choices_accept_dynamic_system_tools() {
    for tool_choice in [
        json!("required"),
        json!({"type": "function", "function": {"name": "lookup"}}),
    ] {
        let request = request(json!({
            "messages": [
                {
                    "role": "system",
                    "content": "",
                    "tools": [{"name": "lookup", "parameters": {"type": "object"}}]
                },
                {"role": "user", "content": "go"}
            ],
            "tool_choice": tool_choice
        }));
        assert_eq!(
            constraint(&request, Some("kimi_k3"), None, false)
                .expect("dynamic forced choice must validate"),
            GuidedToolConstraint::None,
            "K3 forced choices use native XTML rather than guided JSON"
        );
    }
}
