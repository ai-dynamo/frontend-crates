// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_parsers_v2::structural_tag::{
    ReasoningBoundary, StructuralTagContext, StructuralTagOptions, StructuralTagSchemaMode,
    StructuralTagToolChoice,
};
use dynamo_parsers_v2::{
    Tool, UnifiedEvent, UnifiedParserExt, create_unified_parser_for_family,
    structural_tag_builder_for_family,
};
use serde_json::{Value, json};

fn tools() -> Vec<Tool> {
    vec![
        Tool {
            name: "weather.lookup".into(),
            description: None,
            strict: Some(true),
            parameters: json!({"type":"object", "properties":{
            "city":{"type":"string", "enum":["Paris", "London"]},
            "count":{"type":"integer", "minimum":1, "maximum":3},
            "options":{"type":"object", "properties":{"units":{"type":"string", "enum":["C", "F"]}}, "required":["units"], "additionalProperties":false}
        }, "required":["city", "count"], "additionalProperties":false}),
        },
        Tool {
            name: "ping".into(),
            description: None,
            strict: Some(true),
            parameters: json!({"type":"object", "properties":{}, "additionalProperties":false}),
        },
    ]
}

fn call(city: &str, count: &str, end: &str) -> String {
    format!(
        " to=weather.lookup<|message|><atem:function_calls>\n<atem:invoke name=\"weather.lookup\">\n<atem:parameter name=\"city\">{city}</atem:parameter>\n<atem:parameter name=\"count\">{count}</atem:parameter>\n</atem:invoke>\n</atem:function_calls>{end}"
    )
}

fn grammar(
    choice: StructuralTagToolChoice<'_>,
    parallel: Option<bool>,
    schema: Option<&Value>,
    tools: &[Tool],
) -> Value {
    structural_tag_builder_for_family("muse_spark_open")
        .unwrap()
        .build(&StructuralTagContext {
            tool_choice: choice,
            tools,
            parallel_tool_calls: parallel,
            schema_mode: StructuralTagSchemaMode::Auto,
            structured_output_schema: schema,
            starts_in_reasoning: false,
        })
        .unwrap()
        .unwrap()
}

fn capture(name: &str, grammar: &Value, accept: Vec<String>, reject: Vec<String>) {
    // The same Rust-built grammar is exercised by each pinned backend's real
    // XGrammar compiler, without reimplementing the builder in Python.
    if let Some(directory) = std::env::var_os("MUSE_STRUCTURAL_TAG_CAPTURE_DIR") {
        let path = std::path::PathBuf::from(directory);
        std::fs::create_dir_all(&path).unwrap();
        std::fs::write(
            path.join(format!("{name}.json")),
            serde_json::to_vec_pretty(&json!({"grammar":grammar,"accept":accept,"reject":reject}))
                .unwrap(),
        )
        .unwrap();
    }
}

#[test]
fn required_constrains_native_names_types_and_call_count() {
    let tools = tools();
    for (name, parallel) in [
        ("required_single", Some(false)),
        ("required_parallel", Some(true)),
        ("required_default", None),
    ] {
        let tag = grammar(StructuralTagToolChoice::Required, parallel, None, &tools);
        let wire = tag.to_string();
        assert!(wire.contains("atem:invoke"));
        assert!(wire.contains("json_schema")); // nested/scalar values retain JSON Schema validation
        assert!(wire.contains("weather.lookup"));
        let first = call("Paris", "2", "<|eom|>");
        let second = format!(
            "{first}<|start|>assistant{}",
            call("London", "1", "<|eot|>")
        );
        let progress = format!(
            " to=self<|message|>Check both cities.<|eom|><|start|>assistant to=commentary<|message|>Checking.<|eom|><|start|>assistant{first}"
        );
        let mut accept = vec![first.clone(), progress, call("London", "3", "<|eot|>")];
        let mut reject = vec![
            String::new(),
            " to=user<|message|>No tool.<|eot|>".into(),
            format!(
                " to=commentary<|message|>One.<|eom|><|start|>assistant to=commentary<|message|>Two.<|eom|><|start|>assistant{first}"
            ),
            first.replace("weather.lookup", "unregistered"),
            first.replacen("name=\"weather.lookup\"", "name=\"ping\"", 1),
            call("Rome", "2", "<|eom|>"),
            call("Paris", "0", "<|eom|>"),
            call("Paris", "\"2\"", "<|eom|>"),
            first.replace("<atem:parameter name=\"count\">2</atem:parameter>", ""),
            first.replace(
                "</atem:invoke>",
                "<atem:parameter name=\"city\">London</atem:parameter></atem:invoke>",
            ),
        ];
        if parallel == Some(false) {
            reject.push(second)
        } else {
            accept.push(second)
        }
        capture(name, &tag, accept, reject);
        let parsed = create_unified_parser_for_family("muse_spark_open", &tools)
            .unwrap()
            .parse_complete(&format!("<|start|>assistant{first}"))
            .unwrap();
        assert_eq!(
            parsed,
            vec![UnifiedEvent::ToolCall {
                name: "weather.lookup".into(),
                arguments: json!({"city":"Paris", "count":2})
            }]
        );
    }
}

#[test]
fn named_allows_only_the_selected_tool_even_with_parallel_enabled() {
    let tag = grammar(
        StructuralTagToolChoice::Named("weather.lookup"),
        Some(true),
        None,
        &tools(),
    );
    assert!(!tag.to_string().contains("to=ping"));
    let first = call("Paris", "1", "<|eot|>");
    capture("named", &tag, vec![first.clone()], vec![format!("{first}<|start|>assistant{first}"),
        " to=ping<|message|><atem:function_calls><atem:invoke name=\"ping\"></atem:invoke></atem:function_calls><|eot|>".into()]);
}

#[test]
fn auto_composes_structured_final_answers_inside_the_native_user_envelope() {
    let schema = json!({"type":"object", "properties":{"ok":{"const":true}}, "required":["ok"], "additionalProperties":false});
    let tag = grammar(
        StructuralTagToolChoice::Auto,
        Some(false),
        Some(&schema),
        &tools(),
    );
    assert!(tag.to_string().contains("to=user"));
    capture(
        "structured_final",
        &tag,
        vec![
            call("Paris", "1", "<|eom|>"),
            " to=user<|message|>{\"ok\":true}<|eot|>".into(),
        ],
        vec![
            "{\"ok\":true}".into(),
            " to=user<|message|>{\"ok\":false}<|eot|>".into(),
        ],
    );
}

#[test]
fn parameterless_tools_and_non_strict_arguments_remain_usable() {
    let mut tools = tools();
    let empty = grammar(StructuralTagToolChoice::Named("ping"), None, None, &tools);
    capture("empty", &empty, vec![" to=ping<|message|><atem:function_calls><atem:invoke name=\"ping\"></atem:invoke></atem:function_calls><|eot|>".into()], vec![]);
    tools[0].strict = Some(false);
    let tag = grammar(
        StructuralTagToolChoice::Named("weather.lookup"),
        None,
        None,
        &tools,
    );
    capture(
        "non_strict",
        &tag,
        vec![call("Rome", "anything", "<|eot|>")],
        vec![],
    );
    tools[1].parameters = Value::Null;
    let tag = grammar(StructuralTagToolChoice::Named("ping"), None, None, &tools);
    assert!(!tag.to_string().contains("atem:parameter"));
}

#[test]
fn unsupported_raw_string_constraints_are_errors_instead_of_weaker_grammars() {
    let mut tools = tools();
    tools[0].parameters["properties"]["city"] =
        json!({"type":"string", "pattern":"^A", "minLength":4, "contentMediaType":"text/plain"});
    let error = structural_tag_builder_for_family("muse_spark_open")
        .unwrap()
        .build(&StructuralTagContext {
            tool_choice: StructuralTagToolChoice::Required,
            tools: &tools,
            parallel_tool_calls: None,
            schema_mode: StructuralTagSchemaMode::Strict,
            structured_output_schema: None,
            starts_in_reasoning: false,
        })
        .unwrap_err();
    assert!(error.to_string().contains("unsupported"));
}

#[test]
fn external_reasoning_cannot_guarantee_a_forced_muse_tool_call() {
    let tools = tools();
    let result = structural_tag_builder_for_family("muse_spark_open")
        .unwrap()
        .build_with_options(
            &StructuralTagContext {
                tool_choice: StructuralTagToolChoice::Required,
                tools: &tools,
                parallel_tool_calls: Some(false),
                schema_mode: StructuralTagSchemaMode::Auto,
                structured_output_schema: None,
                starts_in_reasoning: false,
            },
            &StructuralTagOptions {
                reasoning_boundary: ReasoningBoundary::External,
                ..Default::default()
            },
        );
    assert!(
        result.is_err(),
        "a gate activated only after a tool channel opens cannot enforce required"
    );
}

#[test]
fn nested_json_keeps_definitions_and_nullable_strings_use_raw_atem_values() {
    let mut tools = tools();
    tools[0].parameters = json!({"type":"object", "properties":{
        "city":{"type":["string","null"]},
        "options":{"type":"object", "properties":{"n":{"$ref":"#/$defs/count"}}, "required":["n"], "additionalProperties":false}
    }, "required":["city","options"], "additionalProperties":false,
    "$defs":{"count":{"type":"integer", "minimum":2}}});
    let tag = grammar(
        StructuralTagToolChoice::Named("weather.lookup"),
        None,
        None,
        &tools,
    );
    let output = |city, n| {
        format!(
            " to=weather.lookup<|message|><atem:function_calls><atem:invoke name=\"weather.lookup\"><atem:parameter name=\"city\">{city}</atem:parameter><atem:parameter name=\"options\">{{\"n\":{n}}}</atem:parameter></atem:invoke></atem:function_calls><|eot|>"
        )
    };
    capture(
        "nested_refs",
        &tag,
        vec![output("\"Paris\"", "2"), output("null", "3")],
        vec![output("\"Paris\"", "0")],
    );
}

#[test]
fn freeform_custom_tool_input_is_unquoted_and_preserved_verbatim() {
    let tools = [Tool {
        name: "apply_patch".into(),
        description: None,
        strict: Some(true),
        parameters: json!({"type":"object", "properties":{"input":{"type":"string", "contentMediaType":"text/plain"}}, "required":["input"]}),
    }];
    let tag = grammar(
        StructuralTagToolChoice::Named("apply_patch"),
        Some(false),
        None,
        &tools,
    );
    let payload = "*** Begin Patch\n+{\"n\":2}\n*** End Patch";
    let output = format!(
        " to=apply_patch<|message|><atem:function_calls><atem:invoke name=\"apply_patch\"><atem:parameter name=\"input\">{payload}</atem:parameter></atem:invoke></atem:function_calls><|eot|>"
    );
    let events = create_unified_parser_for_family("muse_spark_open", &tools)
        .unwrap()
        .parse_complete(&format!("<|start|>assistant{output}"))
        .unwrap();
    assert_eq!(
        events,
        vec![UnifiedEvent::ToolCall {
            name: "apply_patch".into(),
            arguments: json!({"input":payload})
        }]
    );
    capture("freeform", &tag, vec![output], vec![]);
}

#[test]
fn tool_names_matching_reserved_recipients_still_generate_tool_calls() {
    for name in ["self", "user", "commentary", "developer"] {
        let tools = [Tool {
            name: name.into(),
            description: None,
            strict: Some(true),
            parameters: json!({"type":"object", "properties":{}, "additionalProperties":false}),
        }];
        let tag = grammar(StructuralTagToolChoice::Named(name), None, None, &tools);
        assert!(
            tag.to_string().contains("to=tool<|message|>"),
            "reserved recipient {name} needs a tool-channel envelope"
        );
        let output = format!(
            " to=tool<|message|><atem:function_calls><atem:invoke name=\"{name}\"></atem:invoke></atem:function_calls><|eot|>"
        );
        let events = create_unified_parser_for_family("muse_spark_open", &tools)
            .unwrap()
            .parse_complete(&format!("<|start|>assistant{output}"))
            .unwrap();
        assert_eq!(
            events,
            vec![UnifiedEvent::ToolCall {
                name: name.into(),
                arguments: json!({})
            }]
        );
        capture(&format!("reserved_{name}"), &tag, vec![output], vec![]);
    }
}

#[test]
fn none_uses_a_native_final_message_and_disallows_tool_channels() {
    let tag = grammar(StructuralTagToolChoice::None, None, None, &tools());
    assert!(tag.to_string().contains("to=user<|message|>"));
    capture(
        "none",
        &tag,
        vec![" to=user<|message|>Hello.<|eot|>".into()],
        vec![call("Paris", "2", "<|eot|>")],
    );
}

#[test]
fn raw_string_constraints_that_cannot_exclude_xml_delimiters_are_rejected() {
    for schema in [
        json!({"type":"string","pattern":".*"}),
        json!({"type":"string","minLength":1}),
    ] {
        let mut tools = tools();
        tools[0].parameters["properties"]["city"] = schema;
        tools[0].parameters["properties"]["city"]["contentMediaType"] = json!("text/plain");
        let result = structural_tag_builder_for_family("muse_spark_open")
            .unwrap()
            .build(&StructuralTagContext {
                tool_choice: StructuralTagToolChoice::Named("weather.lookup"),
                tools: &tools,
                parallel_tool_calls: None,
                schema_mode: StructuralTagSchemaMode::Strict,
                structured_output_schema: None,
                starts_in_reasoning: false,
            });
        assert!(
            result.is_err(),
            "a raw-text regex must not admit parameter-closing delimiters as value data"
        );
    }
}

#[test]
fn ordinary_string_arguments_remain_strings_when_they_look_like_json() {
    let tools = [Tool {
        name: "record".into(),
        description: None,
        strict: Some(true),
        parameters: json!({"type":"object", "properties":{"value":{"type":"string"}}, "required":["value"], "additionalProperties":false}),
    }];
    let tag = grammar(StructuralTagToolChoice::Named("record"), None, None, &tools);
    assert!(
        tag.to_string().contains("any_text"),
        "ordinary strings use native unquoted ATEM syntax"
    );
    let mut accept = Vec::new();
    let reject = Vec::new();
    for value in ["42", "true", "null", "\"quoted\"", "Paris", "é🙂"] {
        let output = |body: &str| {
            format!(
                " to=record<|message|><atem:function_calls><atem:invoke name=\"record\"><atem:parameter name=\"value\">{body}</atem:parameter></atem:invoke></atem:function_calls><|eot|>"
            )
        };
        let encoded = output(&serde_json::to_string(value).unwrap());
        let events = create_unified_parser_for_family("muse_spark_open", &tools)
            .unwrap()
            .parse_complete(&format!("<|start|>assistant{encoded}"))
            .unwrap();
        assert_eq!(
            events,
            vec![UnifiedEvent::ToolCall {
                name: "record".into(),
                arguments: json!({"value":value})
            }]
        );
        accept.push(encoded);
        if !matches!(serde_json::from_str::<Value>(value), Ok(Value::String(_))) {
            accept.push(output(value));
        }
    }
    capture("string_types", &tag, accept, reject);
}

#[test]
fn empty_and_implicit_object_parameter_schemas_match_the_chat_template() {
    let mut tools = tools();
    for schema in [
        Value::Null,
        json!({}),
        json!({"type":"object"}),
        json!({"properties":{}}),
    ] {
        tools[1].parameters = schema;
        let tag = grammar(StructuralTagToolChoice::Named("ping"), None, None, &tools);
        assert!(
            !tag.to_string().contains("atem:parameter"),
            "empty tool schemas must describe zero arguments"
        );
    }
    tools[0].parameters.as_object_mut().unwrap().remove("type");
    let tag = grammar(
        StructuralTagToolChoice::Named("weather.lookup"),
        None,
        None,
        &tools,
    );
    assert!(tag.to_string().contains("atem:parameter"));
    capture(
        "implicit_object",
        &tag,
        vec![call("Paris", "2", "<|eot|>")],
        vec![],
    );
}

#[test]
fn non_object_argument_schemas_are_errors_without_panics() {
    for schema in [
        json!(false),
        json!([]),
        json!("invalid"),
        json!({"type":"array"}),
    ] {
        let tools = [Tool {
            name: "f".into(),
            description: None,
            strict: Some(true),
            parameters: schema,
        }];
        let result = structural_tag_builder_for_family("muse_spark_open")
            .unwrap()
            .build(&StructuralTagContext {
                tool_choice: StructuralTagToolChoice::Required,
                tools: &tools,
                parallel_tool_calls: None,
                schema_mode: StructuralTagSchemaMode::Strict,
                structured_output_schema: None,
                starts_in_reasoning: false,
            });
        assert!(result.is_err());
    }
}

#[test]
fn native_unquoted_declared_strings_keep_their_schema_type() {
    let tools = [Tool {
        name: "record".into(),
        description: None,
        strict: Some(true),
        parameters: json!({"type":"object","properties":{"value":{"type":"string"}},"required":["value"],"additionalProperties":false}),
    }];
    for value in ["42", "true", "null", "Boston"] {
        let input = format!(
            "<|start|>assistant to=record<|message|><atem:function_calls><atem:invoke name=\"record\"><atem:parameter name=\"value\">{value}</atem:parameter></atem:invoke></atem:function_calls><|eot|>"
        );
        let parsed = create_unified_parser_for_family("muse_spark_open", &tools)
            .unwrap()
            .parse_complete(&input)
            .unwrap();
        assert_eq!(
            parsed,
            vec![UnifiedEvent::ToolCall {
                name: "record".into(),
                arguments: json!({"value":value})
            }],
            "native string {value}"
        );
    }
}

#[test]
fn checkpoint_eot_is_engine_stop_not_literal_grammar_text() {
    for choice in [
        StructuralTagToolChoice::Required,
        StructuralTagToolChoice::Named("weather.lookup"),
    ] {
        let tag = grammar(choice, None, None, &tools());
        assert!(
            !tag.to_string().contains("<|eot|>"),
            "the checkpoint EOS token must remain available once the native payload is complete"
        );
    }
}

#[test]
fn any_order_cannot_weaken_required_parameter_validation() {
    let tools = tools();
    let result = structural_tag_builder_for_family("muse_spark_open")
        .unwrap()
        .build_with_options(
            &StructuralTagContext {
                tool_choice: StructuralTagToolChoice::Required,
                tools: &tools,
                parallel_tool_calls: None,
                schema_mode: StructuralTagSchemaMode::Strict,
                structured_output_schema: None,
                starts_in_reasoning: false,
            },
            &StructuralTagOptions {
                tool_arguments_any_order: true,
                ..Default::default()
            },
        );
    assert!(
        result.is_err(),
        "permitting repeated alternatives would allow the same optional field in place of a required field"
    );
}
