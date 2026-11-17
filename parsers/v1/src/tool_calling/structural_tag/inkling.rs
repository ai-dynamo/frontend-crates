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
}
