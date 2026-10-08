// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_parsers::tool_calling::{
    StructuralTagSchemaMode, ToolCallConfig, ToolCallFormatBuildContext, ToolChoice,
    ToolDefinition, try_tool_call_parse_aggregate,
};
use serde_json::{Value, json};

fn build(
    tools: &[ToolDefinition],
    choice: &ToolChoice,
    schema_mode: StructuralTagSchemaMode,
    starts_in_reasoning: bool,
    parallel_tool_calls: Option<bool>,
) -> Value {
    ToolCallConfig::glm47()
        .structural_tag_builder
        .unwrap()
        .build_tool_call_format(&ToolCallFormatBuildContext {
            tool_choice: choice,
            tools,
            parallel_tool_calls,
            schema_mode,
            starts_in_reasoning,
        })
        .unwrap()
        .unwrap()
}

#[test]
fn glm47_builder_schema_policy_and_reasoning_modes() {
    let parameters = json!({"type":"object", "properties":{"seconds":{"type":"integer","maximum":600}},
        "required":["seconds"], "additionalProperties":false});
    for strict in [None, Some(false), Some(true)] {
        let tools = ["probe", "other"].map(|name| ToolDefinition {
            name: name.into(),
            parameters: Some(parameters.clone()),
            strict,
        });
        for mode in [
            StructuralTagSchemaMode::Auto,
            StructuralTagSchemaMode::Strict,
        ] {
            for choice in [
                ToolChoice::Auto,
                ToolChoice::Required,
                ToolChoice::Named("probe".into()),
            ] {
                for reasoning in [false, true] {
                    for parallel in [None, Some(false), Some(true)] {
                        let built = build(&tools, &choice, mode, reasoning, parallel);
                        let mut format = &built["format"];
                        if reasoning {
                            assert_eq!(format["type"], "sequence");
                            assert_eq!(format["elements"].as_array().unwrap().len(), 2);
                            let prefix = &format["elements"][0];
                            assert_eq!(prefix["type"], "tag");
                            assert_eq!(prefix["begin"], "");
                            assert_eq!(prefix["end"], "</think>");
                            assert_eq!(prefix["content"]["type"], "any_text");
                            for marker in [
                                "<think>",
                                "</think>",
                                "<tool_call>",
                                "</tool_call>",
                                "<arg_key>",
                                "</arg_key>",
                                "<arg_value>",
                                "</arg_value>",
                            ] {
                                assert!(
                                    prefix["content"]["excludes"]
                                        .as_array()
                                        .unwrap()
                                        .contains(&json!(marker))
                                );
                            }
                            format = &format["elements"][1];
                        }
                        let tags: Vec<&Value> = if matches!(choice, ToolChoice::Named(_)) {
                            assert_eq!(format["type"], "tag");
                            assert_eq!(format["begin"], "<tool_call>probe");
                            vec![format]
                        } else {
                            assert_eq!(format["type"], "triggered_tags");
                            assert_eq!(format["triggers"], json!(["<tool_call>"]));
                            assert_eq!(format["at_least_one"], choice == ToolChoice::Required);
                            assert_eq!(format["stop_after_first"], parallel == Some(false));
                            assert_eq!(format["tags"].as_array().unwrap().len(), 2);
                            format["tags"].as_array().unwrap().iter().collect()
                        };
                        let expected_schema =
                            if mode == StructuralTagSchemaMode::Strict || strict != Some(false) {
                                parameters.clone()
                            } else {
                                json!(true)
                            };
                        for tag in tags {
                            assert_eq!(tag["end"], "</tool_call>");
                            assert_eq!(tag["content"]["style"], "glm_xml");
                            assert_eq!(tag["content"]["json_schema"], expected_schema);
                            assert!(tag["content"].get("any_order").is_none());
                        }
                    }
                }
            }
        }
    }
}

fn constant_cases() -> Vec<(ToolDefinition, String, Value)> {
    let mut cases = Vec::new();
    for literal in [
        "\"hello\"",
        "{\"x\":1}",
        "[1,2]",
        "null",
        "true",
        "42",
        "  spaced  ",
        "&lt;",
    ] {
        let constant = json!({"const": literal});
        for property in [
            constant.clone(),
            json!({"type":"string", "const":literal}),
            json!({"$ref":"#/$defs/Literal"}),
            json!({"allOf":[constant.clone()]}),
            json!({"anyOf":[constant.clone(), {"type":"null"}]}),
        ] {
            // A bare null in a nullable union intentionally selects JSON null.
            if literal == "null" && property.get("anyOf").is_some() {
                continue;
            }
            let tool = ToolDefinition {
                name: "probe".into(),
                parameters: Some(json!({"type":"object", "$defs":{"Literal":constant},
                    "properties":{"value":property}, "required":["value"], "additionalProperties":false})),
                strict: Some(true),
            };
            let wire = format!(
                "<tool_call>probe<arg_key>value</arg_key><arg_value>{literal}</arg_value></tool_call>"
            );
            cases.push((tool, wire, json!({"value":literal})));
        }
    }
    cases
}

#[tokio::test]
async fn glm47_string_constants_preserve_literal_arguments() {
    for (tool, wire, expected) in constant_cases() {
        let (calls, _) =
            try_tool_call_parse_aggregate(&wire, Some("glm47"), Some(std::slice::from_ref(&tool)))
                .await
                .unwrap();
        assert_eq!(calls.len(), 1);
        let actual: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(actual, expected, "schema: {:?}", tool.parameters);
    }
}

// XGrammar is a Python dependency, so keep this explicit rather than requiring
// it for the Rust-only suite. Run with either supported version, for example:
// uv run --no-project --with xgrammar==0.2.4 cargo test -p dynamo-parsers \
//   --test glm47_structural_tag glm47_xgrammar_roundtrip -- --ignored
#[tokio::test]
#[ignore = "requires python3 with xgrammar (validated with 0.2.4 and 0.2.7)"]
async fn glm47_xgrammar_roundtrip() {
    use std::io::Write;
    use std::process::{Command, Stdio};

    let mut probes = Vec::new();
    for (tool, wire, expected) in constant_cases() {
        let tag = build(
            std::slice::from_ref(&tool),
            &ToolChoice::Named("probe".into()),
            StructuralTagSchemaMode::Auto,
            false,
            None,
        );
        let (calls, _) =
            try_tool_call_parse_aggregate(&wire, Some("glm47"), Some(std::slice::from_ref(&tool)))
                .await
                .unwrap();
        let actual: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(actual, expected, "schema: {:?}", tool.parameters);
        probes.push(json!({"tag":tag, "wire":wire}));
    }
    let script = r#"
import json, sys
import xgrammar as xg
compiler = xg.GrammarCompiler(xg.TokenizerInfo([bytes([i]) for i in range(256)]))
for probe in json.load(sys.stdin):
    compiled = compiler.compile_grammar(xg.Grammar.from_structural_tag(probe['tag']))
    matcher = xg.GrammarMatcher(compiled)
    assert matcher.accept_string(probe['wire']) and matcher.is_completed(), probe
"#;
    let mut child = Command::new("python3")
        .args(["-c", script])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(&serde_json::to_vec(&probes).unwrap())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}
