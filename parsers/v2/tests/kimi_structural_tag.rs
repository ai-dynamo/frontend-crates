// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_parsers_v2::structural_tag::{
    ReasoningBoundary, StructuralTagContext, StructuralTagOptions, StructuralTagSchemaMode,
    StructuralTagToolChoice,
};
use dynamo_parsers_v2::{Tool, create_tool_parser_for_family, structural_tag_builder_for_family};
use serde_json::{Value, json};

fn tools() -> Vec<Tool> {
    vec![
        Tool {
            name: "weather".into(),
            description: None,
            parameters: json!({
                "type": "object", "properties": {"days": {"type": "integer"}}, "required": ["days"], "additionalProperties": false
            }),
            strict: Some(true),
        },
        Tool {
            name: "other".into(),
            description: None,
            parameters: json!({"type":"object"}),
            strict: None,
        },
    ]
}
fn context(tools: &[Tool]) -> StructuralTagContext<'_> {
    StructuralTagContext {
        tools,
        tool_choice: StructuralTagToolChoice::Auto,
        parallel_tool_calls: None,
        schema_mode: StructuralTagSchemaMode::Auto,
        structured_output_schema: None,
        starts_in_reasoning: false,
    }
}
fn nodes<'a>(value: &'a Value, kind: &str) -> Vec<&'a Value> {
    let mut found = Vec::new();
    if value["type"] == kind {
        found.push(value);
    }
    match value {
        Value::Object(map) => {
            for v in map.values() {
                found.extend(nodes(v, kind));
            }
        }
        Value::Array(array) => {
            for v in array {
                found.extend(nodes(v, kind));
            }
        }
        _ => (),
    }
    found
}

#[test]
fn kimi_request_policy_matrix() {
    let mut tools = tools();
    for family in ["kimi_k2", "kimi_k3"] {
        let builder = structural_tag_builder_for_family(family).unwrap();
        assert!(builder.build(&context(&[])).unwrap().is_none());
        for choice in [
            StructuralTagToolChoice::Required,
            StructuralTagToolChoice::Named("missing"),
        ] {
            let mut ctx = context(&[]);
            ctx.tool_choice = choice;
            assert!(builder.build(&ctx).is_err());
        }
        for choice in [
            StructuralTagToolChoice::Auto,
            StructuralTagToolChoice::Required,
            StructuralTagToolChoice::Named("weather"),
        ] {
            for parallel in [None, Some(false), Some(true)] {
                let mut ctx = context(&tools);
                ctx.tool_choice = choice;
                ctx.parallel_tool_calls = parallel;
                let value = builder.build(&ctx).unwrap().unwrap();
                let sequences = nodes(&value, "tags_with_separator");
                assert_eq!(sequences.len(), 1);
                assert_eq!(
                    sequences[0]["stop_after_first"],
                    parallel == Some(false) || matches!(choice, StructuralTagToolChoice::Named(_))
                );
                assert_eq!(
                    sequences[0]["tags"].as_array().unwrap().len(),
                    if matches!(choice, StructuralTagToolChoice::Named(_)) {
                        1
                    } else {
                        2
                    }
                );
                assert_eq!(sequences[0]["at_least_one"], true);
            }
        }
        // Both generations retain their own defaults. V2 omitted strict is relaxed.
        for strict in [None, Some(false), Some(true)] {
            tools[0].strict = strict;
            for mode in [
                StructuralTagSchemaMode::Auto,
                StructuralTagSchemaMode::Strict,
            ] {
                let mut ctx = context(&tools);
                ctx.tool_choice = StructuralTagToolChoice::Named("weather");
                ctx.schema_mode = mode;
                let value = builder.build(&ctx).unwrap().unwrap();
                let schemas = nodes(&value, "json_schema");
                let enforced = mode == StructuralTagSchemaMode::Strict || strict == Some(true);
                if family == "kimi_k2" {
                    assert_eq!(
                        schemas[0]["json_schema"],
                        if enforced {
                            tools[0].parameters.clone()
                        } else {
                            json!(true)
                        }
                    );
                } else {
                    assert_eq!(!schemas.is_empty(), enforced);
                    if enforced {
                        assert_eq!(schemas[0]["json_schema"]["type"], "integer");
                    }
                }
            }
        }
    }
}

#[test]
fn kimi_reasoning_exclusions_and_argument_order() {
    let tools = tools();
    for (family, close) in [
        ("kimi_k2", "</think>"),
        ("kimi_k3", "<|close|>think<|sep|>"),
    ] {
        let builder = structural_tag_builder_for_family(family).unwrap();
        for choice in [
            StructuralTagToolChoice::Auto,
            StructuralTagToolChoice::Required,
            StructuralTagToolChoice::Named("weather"),
        ] {
            let mut ctx = context(&tools);
            ctx.tool_choice = choice;
            ctx.starts_in_reasoning = true;
            let value = builder.build(&ctx).unwrap().unwrap();
            assert_eq!(value["format"]["elements"][0]["end"], close);
            let external = builder
                .build_with_options(
                    &ctx,
                    &StructuralTagOptions {
                        reasoning_boundary: ReasoningBoundary::External,
                        ..Default::default()
                    },
                )
                .unwrap()
                .unwrap();
            assert_eq!(&value["format"]["elements"][1], &external["format"]);
            let relaxed = builder
                .build_with_options(
                    &ctx,
                    &StructuralTagOptions {
                        exclude_special_tokens: Some(false),
                        tool_arguments_any_order: true,
                        ..Default::default()
                    },
                )
                .unwrap()
                .unwrap();
            assert_eq!(
                relaxed["format"]["elements"][0]["content"]["excludes"],
                json!([])
            );
            for node in nodes(&relaxed, "json_schema") {
                assert_eq!(node["any_order"], true);
            }
        }
        let mut ctx = context(&tools);
        ctx.tool_choice = StructuralTagToolChoice::None;
        let value = builder.build(&ctx).unwrap().unwrap();
        assert!(
            !value["format"]["content"]["excludes"]
                .as_array()
                .unwrap()
                .is_empty()
        );
    }
}

#[test]
fn kimi_structured_response_keeps_schema_and_channel() {
    let tools = tools();
    let schema =
        json!({"type":"object", "properties":{"answer":{"type":"integer"}}, "required":["answer"]});
    for family in ["kimi_k2", "kimi_k3"] {
        let builder = structural_tag_builder_for_family(family).unwrap();
        let mut ctx = context(&tools);
        ctx.structured_output_schema = Some(&schema);
        let value = builder
            .build_with_options(
                &ctx,
                &StructuralTagOptions {
                    tool_arguments_any_order: true,
                    ..Default::default()
                },
            )
            .unwrap()
            .unwrap();
        let alternatives = &value["format"]["elements"];
        let response = &alternatives[1];
        let schemas = nodes(response, "json_schema");
        assert_eq!(schemas.len(), 1);
        assert_eq!(schemas[0]["json_schema"], schema);
        assert!(schemas[0].get("any_order").is_none());
        if family == "kimi_k3" {
            assert_eq!(response["elements"][1]["end"], "<|close|>response<|sep|>");
        }
    }
}

fn call(family: &str, tool: &str, value: &str) -> String {
    if family == "kimi_k2" {
        format!(
            "<|tool_calls_section_begin|><|tool_call_begin|>functions.{tool}:0<|tool_call_argument_begin|>{{\"days\":{value}}}<|tool_call_end|><|tool_calls_section_end|>"
        )
    } else {
        format!(
            "<|close|>response<|sep|><|open|>tools<|sep|><|open|>call tool=\"{tool}\" index=\"0\"<|sep|><|open|>argument key=\"days\" type=\"number\"<|sep|>{value}<|close|>argument<|sep|><|close|>call<|sep|><|close|>tools<|sep|>"
        )
    }
}

/// Set KIMI_GRAMMAR_CASES to export these same cases for the XGrammar guard.
#[test]
fn native_outputs_parse_and_export_grammar_cases() {
    let tools = tools();
    let mut cases = Vec::new();
    for family in ["kimi_k2", "kimi_k3"] {
        let builder = structural_tag_builder_for_family(family).unwrap();
        let valid = call(family, "weather", "2");
        let mut parser = create_tool_parser_for_family(family, &tools).unwrap();
        let parsed = parser.parse_complete(&valid).unwrap().coalesce_calls();
        assert_eq!(parsed.calls.len(), 1, "{family}: {parsed:?}");
        assert_eq!(parsed.calls[0].name.as_deref(), Some("weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&parsed.calls[0].arguments).unwrap(),
            json!({"days":2})
        );
        for choice in [
            StructuralTagToolChoice::Auto,
            StructuralTagToolChoice::Required,
            StructuralTagToolChoice::Named("weather"),
        ] {
            let mut ctx = context(&tools);
            ctx.tool_choice = choice;
            ctx.parallel_tool_calls = Some(false);
            let tag = builder.build(&ctx).unwrap().unwrap();
            let mut accepted = vec![valid.clone()];
            let mut rejected = vec![
                call(family, "missing", "2"),
                call(family, "weather", "false"),
            ];
            let (call_end, section_end, call_begin) = if family == "kimi_k2" {
                (
                    "<|tool_call_end|>",
                    "<|tool_calls_section_end|>",
                    "<|tool_call_begin|>",
                )
            } else {
                (
                    "<|close|>call<|sep|>",
                    "<|close|>tools<|sep|>",
                    "<|open|>call ",
                )
            };
            let first = valid.find(call_begin).unwrap();
            let last = valid.find(call_end).unwrap() + call_end.len();
            let second_call = &valid[first..last];
            let parallel = valid.replace(section_end, &format!("{second_call}{section_end}"));
            rejected.push(parallel.clone());
            if matches!(choice, StructuralTagToolChoice::Named(_)) {
                rejected.push(call(family, "other", "2"));
            }
            let text = if family == "kimi_k2" {
                "hello"
            } else {
                "hello<|close|>response<|sep|>"
            };
            if choice == StructuralTagToolChoice::Auto {
                accepted.push(text.into());
            } else {
                rejected.push(text.into());
            }
            cases.push(json!({"family":family,"tag":tag,"accepted":accepted,"rejected":rejected}));
            ctx.parallel_tool_calls = Some(true);
            cases.push(json!({"family":family,"tag":builder.build(&ctx).unwrap().unwrap(),
                "accepted": if matches!(choice, StructuralTagToolChoice::Named(_)) { vec![valid.clone()] } else { vec![parallel.clone()] },
                "rejected": if matches!(choice, StructuralTagToolChoice::Named(_)) { vec![parallel] } else { vec![] }}));
            ctx.starts_in_reasoning = true;
            let close = if family == "kimi_k2" {
                "</think>"
            } else {
                "<|close|>think<|sep|>"
            };
            cases.push(json!({"family":family,"tag":builder.build(&ctx).unwrap().unwrap(),
                "accepted":[format!("thinking{close}{valid}")], "rejected":[format!("{valid}{close}")]}));
        }
        let schema = json!({"type":"object","properties":{"answer":{"type":"integer"}},"required":["answer"],"additionalProperties":false});
        let mut ctx = context(&tools);
        ctx.structured_output_schema = Some(&schema);
        let response = |body: &str| {
            if family == "kimi_k2" {
                body.to_owned()
            } else {
                format!("{body}<|close|>response<|sep|>")
            }
        };
        cases.push(json!({"family":family,"tag":builder.build(&ctx).unwrap().unwrap(),"accepted":[valid,response("{\"answer\":2}")],"rejected":[response("{\"answer\":false}")]}));
    }
    if let Some(path) = std::env::var_os("KIMI_GRAMMAR_CASES") {
        std::fs::write(path, serde_json::to_vec_pretty(&cases).unwrap()).unwrap();
    }
}
