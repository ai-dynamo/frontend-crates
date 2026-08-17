// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde_json::Value;

use crate::Tool;

use super::builders::{DEEPSEEK_DSML, GLM47, QWEN3_CODER};
use super::test_support::{build, build_context, tools};
use super::{
    ReasoningBoundary, StructuralTagContext, StructuralTagOptions, StructuralTagSchemaMode,
    StructuralTagToolChoice,
};

fn build_with_structured_output(
    builder: &super::StructuralTagBuilder,
    tool_choice: StructuralTagToolChoice<'_>,
    tools: &[Tool],
    parallel_tool_calls: Option<bool>,
    starts_in_reasoning: bool,
    schema: &Value,
) -> Value {
    build_context(
        builder,
        &StructuralTagContext {
            tool_choice,
            tools,
            parallel_tool_calls,
            schema_mode: StructuralTagSchemaMode::Auto,
            structured_output_schema: Some(schema),
            starts_in_reasoning,
        },
    )
}

fn call_sequence(format: &Value) -> &Value {
    match format["type"].as_str() {
        Some("tags_with_separator") => format,
        Some("tag") if format["content"]["type"] == "tags_with_separator" => &format["content"],
        other => panic!("expected a tool-calls-only sequence, got {other:?}"),
    }
}

#[test]
fn common_builder_closes_prompt_opened_reasoning_before_tool_output() {
    let tools = tools();
    for (family, builder, tool_call_exclude, required_prefix) in [
        ("qwen3_coder", &QWEN3_CODER, "<tool_call>", Some("\n\n")),
        ("deepseek_v4", &DEEPSEEK_DSML, "<｜DSML｜", Some("\n\n")),
        ("glm47", &GLM47, "<tool_call>", None),
    ] {
        for tool_choice in [
            StructuralTagToolChoice::Required,
            StructuralTagToolChoice::Named("get_weather"),
        ] {
            let actual = build(
                builder,
                tool_choice,
                &tools,
                None,
                StructuralTagSchemaMode::Auto,
                true,
            );

            assert_eq!(actual["format"]["type"], "sequence", "{family}");
            assert_eq!(actual["format"]["elements"][0]["type"], "tag", "{family}");
            assert_eq!(
                actual["format"]["elements"][0]["content"]["type"], "any_text",
                "{family}"
            );
            assert!(
                actual["format"]["elements"][0]["content"]["excludes"]
                    .as_array()
                    .is_some_and(|excludes| excludes
                        .iter()
                        .any(|exclude| exclude == tool_call_exclude)),
                "{family} reasoning must exclude {tool_call_exclude}"
            );
            assert_eq!(
                actual["format"]["elements"][0]["end"], "</think>",
                "{family}"
            );
            let output = &actual["format"]["elements"][1];
            match (tool_choice, required_prefix) {
                (StructuralTagToolChoice::Required, Some(prefix)) => {
                    assert_eq!(output["type"], "sequence", "{family}");
                    assert_eq!(output["elements"][0]["value"], prefix, "{family}");
                    assert_eq!(output["elements"][1]["type"], "triggered_tags", "{family}");
                }
                (StructuralTagToolChoice::Required, None) => {
                    assert_eq!(output["type"], "triggered_tags", "{family}");
                }
                (StructuralTagToolChoice::Named(_), _) => {
                    assert_eq!(output["type"], "tag", "{family}");
                }
                _ => unreachable!(),
            }
        }
    }
}

#[test]
fn auto_closes_prompt_opened_reasoning_before_triggered_dispatch() {
    let tools = tools();
    for (family, builder, tool_call_exclude) in [
        ("qwen3_coder", &QWEN3_CODER, "<tool_call>"),
        ("deepseek_v4", &DEEPSEEK_DSML, "<｜DSML｜"),
        ("glm47", &GLM47, "<tool_call>"),
    ] {
        let actual = build(
            builder,
            StructuralTagToolChoice::Auto,
            &tools,
            None,
            StructuralTagSchemaMode::Auto,
            true,
        );

        assert_eq!(actual["format"]["type"], "sequence", "{family}");
        assert_eq!(actual["format"]["elements"][0]["end"], "</think>");
        assert!(
            actual["format"]["elements"][0]["content"]["excludes"]
                .as_array()
                .is_some_and(|excludes| excludes
                    .iter()
                    .any(|exclude| exclude == tool_call_exclude)),
            "{family} reasoning must exclude {tool_call_exclude}"
        );
        assert_eq!(
            actual["format"]["elements"][1]["type"], "triggered_tags",
            "{family}"
        );
    }
}

#[test]
fn auto_with_structured_output_offers_tool_calls_only_or_schema_constrained_response() {
    let tools = tools();
    let schema = serde_json::json!({
        "type": "object",
        "properties": {"response": {"type": "string"}},
        "required": ["response"]
    });

    for (family, builder) in [
        ("qwen3_coder", &QWEN3_CODER),
        ("deepseek_v4", &DEEPSEEK_DSML),
        ("glm47", &GLM47),
    ] {
        let actual = build_with_structured_output(
            builder,
            StructuralTagToolChoice::Auto,
            &tools,
            None,
            false,
            &schema,
        );

        assert_eq!(actual["format"]["type"], "or", "{family}");
        let calls = &actual["format"]["elements"][0];
        let sequence = call_sequence(calls);
        assert_eq!(sequence["at_least_one"], true, "{family}");
        assert_eq!(sequence["stop_after_first"], false, "{family}");
        assert_eq!(
            actual["format"]["elements"][1],
            serde_json::json!({
                "type": "json_schema",
                "json_schema": schema,
                "style": "json"
            }),
            "{family}"
        );
    }
}

#[test]
fn auto_with_structured_output_honors_single_call_limit() {
    let tools = tools();
    let schema = serde_json::json!({"type": "object"});

    for (family, builder) in [
        ("qwen3_coder", &QWEN3_CODER),
        ("deepseek_v4", &DEEPSEEK_DSML),
        ("glm47", &GLM47),
    ] {
        let actual = build_with_structured_output(
            builder,
            StructuralTagToolChoice::Auto,
            &tools,
            Some(false),
            false,
            &schema,
        );
        let calls = &actual["format"]["elements"][0];

        assert_eq!(call_sequence(calls)["stop_after_first"], true, "{family}");
    }
}

#[test]
fn auto_with_structured_output_closes_prompt_opened_reasoning_before_choice() {
    let tools = tools();
    let schema = serde_json::json!({"type": "object"});

    for (family, builder) in [
        ("qwen3_coder", &QWEN3_CODER),
        ("deepseek_v4", &DEEPSEEK_DSML),
        ("glm47", &GLM47),
    ] {
        let actual = build_with_structured_output(
            builder,
            StructuralTagToolChoice::Auto,
            &tools,
            None,
            true,
            &schema,
        );

        assert_eq!(actual["format"]["type"], "sequence", "{family}");
        assert_eq!(actual["format"]["elements"][0]["end"], "</think>");
        assert_eq!(actual["format"]["elements"][1]["type"], "or");
    }
}

#[test]
fn external_reasoning_boundary_builds_only_the_post_reasoning_suffix() {
    let tools = tools();
    for (family, builder) in [
        ("qwen3_coder", &QWEN3_CODER),
        ("deepseek_v4", &DEEPSEEK_DSML),
        ("glm47", &GLM47),
    ] {
        let actual = builder
            .build_with_options(
                &StructuralTagContext {
                    tool_choice: StructuralTagToolChoice::Auto,
                    tools: &tools,
                    parallel_tool_calls: None,
                    schema_mode: StructuralTagSchemaMode::Auto,
                    structured_output_schema: None,
                    starts_in_reasoning: true,
                },
                &StructuralTagOptions {
                    reasoning_boundary: ReasoningBoundary::External,
                    ..Default::default()
                },
            )
            .unwrap_or_else(|error| panic!("{family} should build: {error}"))
            .unwrap_or_else(|| panic!("{family} should need a structural tag"));

        assert_eq!(actual["format"]["type"], "triggered_tags", "{family}");
    }
}

#[test]
fn any_order_override_applies_only_to_tool_arguments() {
    let tools = tools();
    let schema = serde_json::json!({
        "type": "object",
        "properties": {"response": {"type": "string"}},
        "required": ["response"]
    });
    let actual = QWEN3_CODER
        .build_with_options(
            &StructuralTagContext {
                tool_choice: StructuralTagToolChoice::Auto,
                tools: &tools,
                parallel_tool_calls: None,
                schema_mode: StructuralTagSchemaMode::Strict,
                structured_output_schema: Some(&schema),
                starts_in_reasoning: false,
            },
            &StructuralTagOptions {
                tool_arguments_any_order: true,
                ..Default::default()
            },
        )
        .expect("structural tag should build")
        .expect("request should need a structural tag");

    let elements = actual["format"]["elements"]
        .as_array()
        .expect("auto with structured output should build an alternation");
    let tool_tags = call_sequence(&elements[0])["tags"]
        .as_array()
        .expect("Qwen tool branch should contain tool tags");
    assert!(
        tool_tags
            .iter()
            .all(|tag| tag["content"]["any_order"] == true)
    );
    assert!(elements[1].get("any_order").is_none());
}

#[test]
fn required_and_named_choices_do_not_offer_final_response_branch() {
    let tools = tools();
    let schema = serde_json::json!({"type": "object"});

    for (family, builder) in [
        ("qwen3_coder", &QWEN3_CODER),
        ("deepseek_v4", &DEEPSEEK_DSML),
        ("glm47", &GLM47),
    ] {
        for tool_choice in [
            StructuralTagToolChoice::Required,
            StructuralTagToolChoice::Named("get_weather"),
        ] {
            let actual =
                build_with_structured_output(builder, tool_choice, &tools, None, false, &schema);

            assert_ne!(actual["format"]["type"], "or", "{family}");
        }
    }
}

#[test]
fn auto_with_empty_tools_needs_no_structural_tag() {
    for (family, builder) in [
        ("qwen3_coder", &QWEN3_CODER),
        ("deepseek_v4", &DEEPSEEK_DSML),
        ("glm47", &GLM47),
    ] {
        let result = builder
            .build(&StructuralTagContext {
                tool_choice: StructuralTagToolChoice::Auto,
                tools: &[],
                parallel_tool_calls: None,
                schema_mode: StructuralTagSchemaMode::Auto,
                structured_output_schema: None,
                starts_in_reasoning: false,
            })
            .unwrap_or_else(|error| panic!("{family} should build: {error}"));

        assert!(result.is_none(), "{family}");
    }
}

#[test]
fn strict_schema_mode_enforces_non_strict_tool_schema() {
    let tools = tools();
    let actual = build(
        &QWEN3_CODER,
        StructuralTagToolChoice::Named("add_numbers"),
        &tools,
        None,
        StructuralTagSchemaMode::Strict,
        false,
    );

    assert_eq!(
        actual["format"]["content"]["json_schema"],
        tools[0].parameters
    );
}

#[test]
fn null_parameters_remain_unconstrained_even_in_strict_mode() {
    let tools = [Tool {
        name: "no_schema".to_string(),
        description: None,
        parameters: Value::Null,
        strict: Some(true),
    }];
    let actual = build(
        &QWEN3_CODER,
        StructuralTagToolChoice::Named("no_schema"),
        &tools,
        None,
        StructuralTagSchemaMode::Strict,
        false,
    );

    assert_eq!(actual["format"]["content"]["json_schema"], true);
}

#[test]
fn invalid_required_and_named_requests_fail() {
    let required = StructuralTagContext {
        tool_choice: StructuralTagToolChoice::Required,
        tools: &[],
        parallel_tool_calls: None,
        schema_mode: StructuralTagSchemaMode::Auto,
        structured_output_schema: None,
        starts_in_reasoning: false,
    };
    assert!(QWEN3_CODER.build(&required).is_err());

    let tools = tools();
    let named = StructuralTagContext {
        tool_choice: StructuralTagToolChoice::Named("missing"),
        tools: &tools,
        parallel_tool_calls: None,
        schema_mode: StructuralTagSchemaMode::Auto,
        structured_output_schema: None,
        starts_in_reasoning: false,
    };
    assert!(QWEN3_CODER.build(&named).is_err());
}
