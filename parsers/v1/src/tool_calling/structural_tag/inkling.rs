// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Inkling's `<|message_model|>` primer and redundant name header precede tool JSON.
//! Trigger on `<|content_invoke_tool_json|>`; constrain its authoritative `name` and `args`.

use std::sync::LazyLock;

use super::builder::{ToolCallFormatBuildContext, resolve_tool_schema, resolve_tools_to_include};
use super::format::{
    ConstStringFormat, Format, JsonSchemaFormat, JsonSchemaStyle, OrFormat, RegexFormat,
    SequenceFormat, StructuralTag, TagFormat, TriggeredTagsFormat,
};

const INVOKE: &str = "<|content_invoke_tool_json|>";
const WS: &str = r"[ \t\r\n]*";
pub(crate) static BAN_TOKENS: LazyLock<Vec<String>> = LazyLock::new(|| vec![INVOKE.to_string()]);

pub(crate) fn build_inkling(
    ctx: &ToolCallFormatBuildContext<'_>,
) -> anyhow::Result<Option<StructuralTag>> {
    let (tools, at_least_one) = resolve_tools_to_include(ctx)?;
    if tools.is_empty() {
        return Ok(None);
    }
    let elements = tools
        .iter()
        .map(|tool| {
            Format::Sequence(SequenceFormat {
                elements: vec![
                    Format::Regex(RegexFormat {
                        pattern: format!(r#"{WS}\{{{WS}"name"{WS}:{WS}"#),
                    }),
                    Format::ConstString(ConstStringFormat {
                        value: serde_json::Value::String(tool.name.clone()).to_string(),
                    }),
                    Format::Regex(RegexFormat {
                        pattern: format!(r#"{WS},{WS}"args"{WS}:{WS}"#),
                    }),
                    // Keep argument schemas standalone so local references retain their root.
                    Format::JsonSchema(JsonSchemaFormat {
                        json_schema: resolve_tool_schema(tool, ctx.strict_schema()),
                        style: JsonSchemaStyle::Json,
                    }),
                    Format::Regex(RegexFormat {
                        pattern: format!(r"{WS}\}}{WS}"),
                    }),
                ],
            })
        })
        .collect();
    Ok(Some(StructuralTag {
        format: Format::TriggeredTags(TriggeredTagsFormat {
            triggers: vec![INVOKE.to_string()],
            tags: vec![TagFormat {
                begin: INVOKE.to_string(),
                content: Box::new(Format::Or(OrFormat { elements })),
                end: "<|end_message|>".to_string(),
            }],
            at_least_one,
            stop_after_first: ctx.stop_after_first(),
        }),
    }))
}

#[cfg(test)]
mod tests {
    use crate::tool_calling::{
        StructuralTagSchemaMode, ToolCallConfig, ToolCallFormatBuildContext, ToolChoice,
        ToolDefinition,
    };
    use serde_json::json;

    fn tools() -> Vec<ToolDefinition> {
        vec![
            ToolDefinition {
                name: "add_numbers".into(),
                parameters: Some(json!({"type": "object", "properties": {
                    "a": {"type": "integer"}, "b": {"type": "integer"}
                }, "required": ["a", "b"]})),
                strict: None,
            },
            ToolDefinition {
                name: "get_weather".into(),
                parameters: Some(json!({
                    "type": "object", "properties": {"location": {"$ref": "#/$defs/city"}},
                    "required": ["location"], "$defs": {"city": {"type": "string"}}
                })),
                strict: None,
            },
        ]
    }

    #[test]
    fn inkling_config_builds_tool_call() {
        let tools = tools();
        let config = ToolCallConfig::inkling();
        let builder = config
            .structural_tag_builder
            .as_ref()
            .expect("inkling should provide a structural tag builder");
        let tool_choice = ToolChoice::Named("get_weather".to_string());
        let c = ToolCallFormatBuildContext {
            tool_choice: &tool_choice,
            tools: &tools,
            parallel_tool_calls: None,
            schema_mode: StructuralTagSchemaMode::Auto,
            starts_in_reasoning: false,
        };
        let parsed = builder
            .build_tool_call_format(&c)
            .expect("build should not error")
            .expect("build should return Some");

        let fmt = &parsed["format"];
        assert_eq!(fmt["type"], "triggered_tags");
        assert_eq!(fmt["triggers"][0], "<|content_invoke_tool_json|>");
        assert_eq!(fmt["at_least_one"], true);

        // Named tool_choice constrains to exactly the one requested tool.
        let tags = fmt["tags"].as_array().unwrap();
        assert_eq!(tags.len(), 1);
        let tag = &tags[0];
        assert_eq!(tag["type"], "tag");
        assert_eq!(tag["begin"], "<|content_invoke_tool_json|>");
        assert_eq!(tag["end"], "<|end_message|>");
        let alternatives = tag["content"]["elements"].as_array().unwrap();
        assert_eq!(alternatives.len(), 1);
        assert_eq!(alternatives[0]["elements"][1]["type"], "const_string");
        assert_eq!(alternatives[0]["elements"][1]["value"], r#""get_weather""#);
        assert_eq!(
            alternatives[0]["elements"][3]["json_schema"],
            tools[1].parameters.clone().unwrap()
        );
        assert_eq!(builder.ban_tokens(), &["<|content_invoke_tool_json|>"]);
    }

    #[test]
    #[ignore = "requires lean CPU XGrammar; uses the shared v2 verifier"]
    fn inkling_json_whitespace_xgrammar() {
        use serde_json::Value;
        use std::{
            io::Write,
            process::{Command, Stdio},
        };

        let mut requests = Vec::new();
        let mut expected = Vec::new();
        for name in ["get_weather", "get-weather"] {
            let tools = [
                ToolDefinition {
                    name: name.into(),
                    parameters: Some(json!({
                        "type": "object", "properties": {"location": {"$ref": "#/$defs/city"}},
                        "required": ["location"], "$defs": {"city": {"type": "string"}}
                    })),
                    strict: None,
                },
                ToolDefinition {
                    name: "other".into(),
                    parameters: Some(
                        json!({"type": "object", "properties": {"count": {"type": "integer"}}, "required": ["count"]}),
                    ),
                    strict: None,
                },
            ];
            let builder = ToolCallConfig::inkling().structural_tag_builder.unwrap();
            for choice in [
                ToolChoice::Auto,
                ToolChoice::Required,
                ToolChoice::Named(name.into()),
            ] {
                for reasoning in [false, true] {
                    for parallel in [false, true] {
                        let grammar = builder
                            .build_tool_call_format(&ToolCallFormatBuildContext {
                                tool_choice: &choice,
                                tools: &tools,
                                parallel_tool_calls: Some(parallel),
                                schema_mode: StructuralTagSchemaMode::Auto,
                                starts_in_reasoning: reasoning,
                            })
                            .unwrap()
                            .unwrap();
                        let prefix = if reasoning && choice != ToolChoice::Auto {
                            "thinking<|end_message|>"
                        } else {
                            ""
                        };
                        let compact =
                            format!(r#"{{"name":"{name}","args":{{"location":"Paris"}}}}"#);
                        for (body, accepted) in [
                            (compact.clone(), true),
                            (
                                format!(r#"{{"name": "{name}", "args": {{"location": "Paris"}}}}"#),
                                true,
                            ),
                            (
                                format!(
                                    " \t\r\n{{\n\t\"name\" : \"{name}\",\n\"args\" : {{\"location\":\"Paris\"}}\n}} \n"
                                ),
                                true,
                            ),
                            (
                                format!(r#"{{"name":"{name}","args":{{"location":7}}}}"#),
                                false,
                            ),
                            (format!(r#"{{"name":"{name}","args":{{}}}}"#), false),
                            (
                                r#"{"name":"unknown","args":{"location":"Paris"}}"#.into(),
                                false,
                            ),
                            (
                                format!(r#"{{"name":"{name}" "args":{{"location":"Paris"}}}}"#),
                                false,
                            ),
                            (
                                format!(
                                    r#"{{"name":"{name}","arguments":{{"location":"Paris"}}}}"#
                                ),
                                false,
                            ),
                            (
                                r#"{"name":"other","args":{"count":2}}"#.into(),
                                choice != ToolChoice::Named(name.into()),
                            ),
                            (
                                format!(
                                    "{compact}<|end_message|><|content_invoke_tool_json|>{compact}"
                                ),
                                parallel,
                            ),
                        ] {
                            let id =
                                format!("inkling/{name}/{choice:?}/{reasoning}/{parallel}/{body}");
                            requests.push(json!({"id": id, "grammar": grammar, "input": format!("{prefix}<|content_invoke_tool_json|>{body}<|end_message|>")}));
                            expected.push((id, accepted));
                        }
                    }
                }
            }
        }
        let mut child = Command::new(
            std::env::var_os("CONFORMANCE_GRAMMAR_PYTHON").unwrap_or_else(|| "python3".into()),
        )
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../v2/tests/check_structural_tag_xgrammar.py"
        ))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("start XGrammar verifier");
        let mut stdin = child.stdin.take().unwrap();
        let writer = std::thread::spawn(move || {
            stdin
                .write_all(&serde_json::to_vec(&requests).unwrap())
                .unwrap();
        });
        let output = child.wait_with_output().unwrap();
        writer.join().unwrap();
        assert!(output.status.success(), "XGrammar verifier failed");
        let results: Vec<Value> = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(results.len(), expected.len(), "verifier skipped cases");
        for ((id, accepted), result) in expected.iter().zip(results) {
            assert_eq!(result["id"], *id);
            assert_eq!(
                result["accepted"] == true && result["complete"] == true,
                *accepted,
                "{result}"
            );
        }
        eprintln!("{} Inkling grammar cases passed", expected.len());
    }
}
