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

#[test]
fn glm47_rejects_ignored_ref_siblings_unless_explicitly_relaxed() {
    for (target, property, keyword) in [
        (
            json!({"type":"integer", "maximum":1000}),
            json!({"$ref":"#/$defs/Target", "maximum":600}),
            "maximum",
        ),
        (
            json!({"type":"integer", "maximum":1000}),
            json!({"$ref":"#/$defs/Target", "maximum":2000}),
            "maximum",
        ),
        (
            json!({"type":"integer"}),
            json!({"$ref":"#/$defs/Target", "maximum":600}),
            "maximum",
        ),
        (
            json!({"anyOf":[{"const":"auto"},{"type":"integer"}]}),
            json!({"$ref":"#/$defs/Target", "type":"integer"}),
            "type",
        ),
    ] {
        for strict in [None, Some(false), Some(true)] {
            for mode in [
                StructuralTagSchemaMode::Auto,
                StructuralTagSchemaMode::Strict,
            ] {
                let tools = [ToolDefinition {
                    name: "probe".into(),
                    parameters: Some(json!({"type":"object", "$defs":{"Target":target},
                        "properties":{"seconds":property}, "required":["seconds"]})),
                    strict,
                }];
                for choice in [
                    ToolChoice::Auto,
                    ToolChoice::Required,
                    ToolChoice::Named("probe".into()),
                ] {
                    let built = ToolCallConfig::glm47()
                        .structural_tag_builder
                        .unwrap()
                        .build_tool_call_format(&ToolCallFormatBuildContext {
                            tools: &tools,
                            tool_choice: &choice,
                            parallel_tool_calls: None,
                            schema_mode: mode,
                            starts_in_reasoning: false,
                        });
                    if mode == StructuralTagSchemaMode::Auto && strict == Some(false) {
                        let built = built.unwrap().unwrap();
                        let format = &built["format"];
                        let tag = if matches!(choice, ToolChoice::Named(_)) {
                            format
                        } else {
                            &format["tags"][0]
                        };
                        assert_eq!(tag["content"]["json_schema"], json!(true));
                    } else {
                        let error = format!("{:#}", built.unwrap_err());
                        assert!(
                            error.contains("probe")
                                && error.contains("seconds")
                                && error.contains(keyword),
                            "{error}"
                        );
                    }
                }
            }
        }
    }
}

fn bounded_schema_cases() -> Vec<(ToolDefinition, String, Value, Vec<String>)> {
    let mut cases = Vec::new();
    for (schema, valid, invalid) in [
        (
            json!({"type":"object", "properties":{
            "seconds":{"type":"integer", "maximum":600},
            "opts":{"type":"object", "default":{"$ref":"literal"}}},
            "required":["seconds"], "additionalProperties":false}),
            600,
            900,
        ),
        (
            json!({"type":"object", "$defs":{"Timeout":{"type":"integer", "maximum":1000}},
            "properties":{"seconds":{"$ref":"#/$defs/Timeout", "maximum":1000}},
            "required":["seconds"], "additionalProperties":false}),
            1000,
            1500,
        ),
        (
            json!({"type":"object", "$defs":{
            "Timeout":{"type":"integer", "maximum":600},
            "Alias":{"$ref":"#/$defs/Timeout", "description":"alias"}},
            "properties":{"seconds":{"$ref":"#/$defs/Alias", "description":"property"}},
            "required":["seconds"], "additionalProperties":false}),
            600,
            900,
        ),
    ] {
        let tool = ToolDefinition {
            name: "probe".into(),
            parameters: Some(schema),
            strict: Some(true),
        };
        let wire = format!(
            "<tool_call>probe<arg_key>seconds</arg_key><arg_value>{valid}</arg_value></tool_call>"
        );
        let reject = vec![
            "<tool_call>probe</tool_call>".into(),
            format!(
                "<tool_call>probe<arg_key>seconds</arg_key><arg_value>{invalid}</arg_value></tool_call>"
            ),
        ];
        cases.push((tool, wire, json!({"seconds":valid}), reject));
    }
    cases
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
    for (tool, wire, expected, reject) in bounded_schema_cases() {
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
        assert_eq!(calls.len(), 1);
        let actual: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(actual, expected);
        probes.push(json!({"tag":tag, "wire":wire, "reject":reject}));
    }
    let script = r#"
import json, sys
import xgrammar as xg
compiler = xg.GrammarCompiler(xg.TokenizerInfo([bytes([i]) for i in range(256)]))
for probe in json.load(sys.stdin):
    compiled = compiler.compile_grammar(xg.Grammar.from_structural_tag(probe['tag']))
    matcher = xg.GrammarMatcher(compiled)
    assert matcher.accept_string(probe['wire']) and matcher.is_completed(), probe
    for invalid in probe.get('reject', []):
        matcher = xg.GrammarMatcher(compiled)
        assert not (matcher.accept_string(invalid) and matcher.is_completed()), (probe, invalid)
"#;
    let mut child = Command::new(std::env::var("GLM_SCHEMA_PYTHON").unwrap_or("python3".into()))
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
