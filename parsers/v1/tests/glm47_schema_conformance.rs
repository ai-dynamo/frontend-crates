// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[path = "../../tests/glm47_schema_matrix.rs"]
mod matrix;

use dynamo_parsers::tool_calling::{
    StructuralTagSchemaMode, ToolCallConfig, ToolCallFormatBuildContext, ToolChoice,
    ToolDefinition, try_tool_call_parse_aggregate,
};
use serde_json::{Value, json};

fn tool(case: &matrix::Case) -> ToolDefinition {
    ToolDefinition {
        name: "probe".into(),
        parameters: Some(case.parameters()),
        strict: Some(true),
    }
}

fn build(tools: &[ToolDefinition], mode: StructuralTagSchemaMode) -> anyhow::Result<Value> {
    ToolCallConfig::glm47()
        .structural_tag_builder
        .unwrap()
        .build_tool_call_format(&ToolCallFormatBuildContext {
            tools,
            tool_choice: &ToolChoice::Named("probe".into()),
            parallel_tool_calls: Some(false),
            schema_mode: mode,
            starts_in_reasoning: false,
        })?
        .ok_or_else(|| anyhow::anyhow!("GLM schema matrix must build a tool format"))
}

// Preserve these parser-positive inputs while asserting that v1 never emits a
// grammar which silently discards their constraining $ref siblings.
fn expects_builder_error(case: &matrix::Case) -> bool {
    matches!(
        case.name.as_str(),
        "ref_sibling_integer" | "unresolved_ref_with_type" | "cyclic_ref_with_type"
    )
}

async fn arguments(case: &matrix::Case) -> Value {
    let (calls, _) =
        try_tool_call_parse_aggregate(&case.wire(), Some("glm47"), Some(&[tool(case)]))
            .await
            .unwrap();
    assert_eq!(calls.len(), 1, "{}", case.name);
    serde_json::from_str(&calls[0].function.arguments).unwrap()
}

#[tokio::test]
async fn authored_schema_matrix_preserves_types_and_literals() {
    let mut failures = Vec::new();
    let mut results = Vec::new();
    for case in matrix::cases() {
        let actual = arguments(&case).await;
        let built = build(&[tool(&case)], StructuralTagSchemaMode::Auto);
        let (tag, builder_error) = if expects_builder_error(&case) {
            let error = format!("{:#}", built.unwrap_err());
            assert!(
                error.contains("$ref sibling `type`"),
                "{}: {error}",
                case.name
            );
            (None, Some(error))
        } else {
            (Some(built.unwrap()), None)
        };
        results.push(
            json!({"name":case.name, "schema":case.parameters(), "wire":case.wire(),
            "expected":case.expected(), "actual":actual,
            "tag":tag, "builder_error":builder_error}),
        );
        if actual != case.expected() {
            failures.push(format!(
                "{}: expected {}, got {}",
                case.name,
                case.expected(),
                actual
            ));
        }
    }
    matrix::write_report("v1", &results);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[tokio::test]
#[ignore = "requires python3 with xgrammar and jsonschema"]
async fn authored_schemas_accept_valid_and_reject_invalid_wires() {
    let mut probes = Vec::new();
    for case in matrix::cases().into_iter().filter(|case| case.grammar) {
        let actual = arguments(&case).await;
        assert_eq!(actual, case.expected(), "{}", case.name);
        let built = build(&[tool(&case)], StructuralTagSchemaMode::Auto);
        if expects_builder_error(&case) {
            let error = format!("{:#}", built.unwrap_err());
            assert!(
                error.contains("$ref sibling `type`"),
                "{}: {error}",
                case.name
            );
        } else {
            probes.push(case.probe(built.unwrap(), actual));
        }
    }
    let case = matrix::cases()
        .into_iter()
        .find(|case| case.name == "const_plain_bare")
        .unwrap();
    for strict in [None, Some(false), Some(true)] {
        for mode in [
            StructuralTagSchemaMode::Auto,
            StructuralTagSchemaMode::Strict,
        ] {
            let mut tool = tool(&case);
            tool.strict = strict;
            let enforce = mode == StructuralTagSchemaMode::Strict || strict != Some(false);
            let mut probe = case.probe(build(&[tool], mode).unwrap(), case.expected());
            probe["name"] = json!(format!("strict_{strict:?}_{mode:?}"));
            if !enforce {
                probe["accept"] = probe["reject"].take();
                probe["reject"] = json!([]);
            }
            probes.push(probe);
        }
    }
    matrix::check_grammar(&probes);
}

#[test]
fn schema_modes_preserve_declared_schema_unless_explicitly_relaxed() {
    let schema = json!({"type":"object", "properties":{"value":{"const":"auto"}}, "required":["value"], "additionalProperties":false});
    for strict in [None, Some(false), Some(true)] {
        for mode in [
            StructuralTagSchemaMode::Auto,
            StructuralTagSchemaMode::Strict,
        ] {
            let tools = [ToolDefinition {
                name: "probe".into(),
                parameters: Some(schema.clone()),
                strict,
            }];
            let enforce = mode == StructuralTagSchemaMode::Strict || strict != Some(false);
            assert_eq!(
                build(&tools, mode).unwrap()["format"]["content"]["json_schema"],
                if enforce { schema.clone() } else { json!(true) }
            );
        }
    }
}
