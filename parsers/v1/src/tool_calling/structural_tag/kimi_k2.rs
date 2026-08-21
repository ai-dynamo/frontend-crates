// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Kimi K2 native structural-tag generation.
//!
//! K2 emits tool calls inside a special-token section rather than as the raw
//! JSON array used by the legacy forced-tool path. The numeric suffix belongs
//! to the model-generated call ID and must remain dynamic for parallel calls.
//!
//! This builder supports K2-Instruct and K2.5/K2.6 reasoning prompts. The
//! K2.5/K2.6 chat template injects `<think>` into the generation prompt, and
//! the model later emits `</think>`. It does not support original K2-Thinking,
//! whose chat template leaves the opening `<think>` for the model to generate.

use serde_json::{Value, json};

use super::builder::{
    ToolCallFormatBuildContext, kimi_uses_declared_tool_schema, resolve_tools_to_include,
};
use super::format::{
    ConstStringFormat, Format, JsonSchemaFormat, JsonSchemaStyle, RegexFormat, SequenceFormat,
    StructuralTag, TagFormat, TagsWithSeparatorFormat, TriggeredTagsFormat,
};
use crate::tool_calling::{ToolChoice, ToolDefinition};

const TOOL_CALL_BEGIN_PREFIX: &str = "<|tool_call_begin|>functions.";
const TOOL_CALL_ARGUMENT_BEGIN: &str = "<|tool_call_argument_begin|>";
const TOOL_CALL_END: &str = "<|tool_call_end|>";
const TOOL_CALLS_SECTION_BEGIN: &str = "<|tool_calls_section_begin|>";
const TOOL_CALLS_SECTION_END: &str = "<|tool_calls_section_end|>";

fn tool_schema(tool: &ToolDefinition, strict_schema: bool) -> Value {
    // Match vLLM/xgrammar: use the declared parameters unless the request
    // explicitly opts out with strict=false. Global strict mode overrides the
    // opt-out. Xgrammar uses `true` for unconstrained but valid JSON.
    if kimi_uses_declared_tool_schema(tool, strict_schema) {
        tool.parameters.clone().unwrap_or_else(|| json!(true))
    } else {
        json!(true)
    }
}

fn call_tag(tool: &ToolDefinition, strict_schema: bool) -> TagFormat {
    TagFormat {
        begin: format!("{TOOL_CALL_BEGIN_PREFIX}{}:", tool.name),
        content: Box::new(Format::Sequence(SequenceFormat {
            elements: vec![
                Format::Regex(RegexFormat {
                    pattern: r"\d+".to_string(),
                }),
                Format::ConstString(ConstStringFormat {
                    value: TOOL_CALL_ARGUMENT_BEGIN.to_string(),
                }),
                Format::JsonSchema(JsonSchemaFormat {
                    json_schema: tool_schema(tool, strict_schema),
                    style: JsonSchemaStyle::Json,
                }),
            ],
        })),
        end: TOOL_CALL_END.to_string(),
    }
}

/// Build Kimi K2's native tool-call section.
pub(crate) fn build_kimi_k2(
    ctx: &ToolCallFormatBuildContext<'_>,
) -> anyhow::Result<Option<StructuralTag>> {
    let (tools, outer_at_least_one) = resolve_tools_to_include(ctx)?;
    if tools.is_empty() {
        return Ok(None);
    }

    let calls: Vec<TagFormat> = tools
        .into_iter()
        .map(|tool| call_tag(tool, ctx.strict_schema()))
        .collect();

    let calls_format = if matches!(ctx.tool_choice, ToolChoice::Named(_)) {
        Format::Tag(
            calls
                .into_iter()
                .next()
                .ok_or_else(|| anyhow::anyhow!("named tool choice resolved no tool"))?,
        )
    } else {
        Format::TagsWithSeparator(TagsWithSeparatorFormat {
            tags: calls,
            separator: String::new(),
            at_least_one: true,
            stop_after_first: ctx.stop_after_first(),
        })
    };

    let format = if matches!(ctx.tool_choice, ToolChoice::Auto) {
        Format::TriggeredTags(TriggeredTagsFormat {
            triggers: vec![TOOL_CALLS_SECTION_BEGIN.to_string()],
            tags: vec![TagFormat {
                begin: TOOL_CALLS_SECTION_BEGIN.to_string(),
                content: Box::new(calls_format),
                end: TOOL_CALLS_SECTION_END.to_string(),
            }],
            at_least_one: outer_at_least_one,
            stop_after_first: ctx.stop_after_first(),
            excludes: vec![],
        })
    } else {
        Format::Sequence(SequenceFormat {
            elements: vec![
                Format::ConstString(ConstStringFormat {
                    value: TOOL_CALLS_SECTION_BEGIN.to_string(),
                }),
                calls_format,
                Format::ConstString(ConstStringFormat {
                    value: TOOL_CALLS_SECTION_END.to_string(),
                }),
            ],
        })
    };

    Ok(Some(StructuralTag { format }))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::tool_calling::structural_tag::StructuralTagSchemaMode;

    fn tools() -> Vec<ToolDefinition> {
        vec![
            ToolDefinition {
                name: "get_weather".to_string(),
                parameters: Some(json!({
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"]
                })),
                strict: None,
            },
            ToolDefinition {
                name: "get_time".to_string(),
                parameters: Some(json!({
                    "type": "object",
                    "properties": {"timezone": {"type": "string"}}
                })),
                strict: Some(false),
            },
        ]
    }

    fn context<'a>(
        tool_choice: &'a ToolChoice,
        tools: &'a [ToolDefinition],
        parallel_tool_calls: Option<bool>,
        starts_in_reasoning: bool,
        schema_mode: StructuralTagSchemaMode,
    ) -> ToolCallFormatBuildContext<'a> {
        ToolCallFormatBuildContext {
            tool_choice,
            tools,
            parallel_tool_calls,
            schema_mode,
            starts_in_reasoning,
        }
    }

    #[test]
    fn required_uses_native_section_dynamic_ids_and_vllm_schema_semantics() {
        let tools = tools();
        let ctx = context(
            &ToolChoice::Required,
            &tools,
            None,
            false,
            StructuralTagSchemaMode::Auto,
        );
        let value = serde_json::to_value(build_kimi_k2(&ctx).unwrap().unwrap()).unwrap();
        let elements = value["format"]["elements"].as_array().unwrap();

        assert_eq!(elements[0]["value"], TOOL_CALLS_SECTION_BEGIN);
        assert_eq!(elements[1]["type"], "tags_with_separator");
        assert_eq!(elements[1]["at_least_one"], true);
        assert_eq!(elements[1]["tags"].as_array().unwrap().len(), 2);
        assert_eq!(elements[2]["value"], TOOL_CALLS_SECTION_END);

        let weather = &elements[1]["tags"][0];
        assert_eq!(
            weather["begin"],
            "<|tool_call_begin|>functions.get_weather:"
        );
        assert_eq!(weather["content"]["elements"][0]["pattern"], r"\d+");
        assert_eq!(
            weather["content"]["elements"][1]["value"],
            TOOL_CALL_ARGUMENT_BEGIN
        );
        assert_eq!(
            weather["content"]["elements"][2]["json_schema"],
            tools[0].parameters.clone().unwrap()
        );
        assert_eq!(weather["end"], TOOL_CALL_END);

        // Unlike the omitted strict flag above, explicit strict=false opts out.
        assert_eq!(
            elements[1]["tags"][1]["content"]["elements"][2]["json_schema"],
            true
        );
    }

    #[test]
    fn named_choice_includes_only_the_selected_tool() {
        let tools = tools();
        let choice = ToolChoice::Named("get_time".to_string());
        let ctx = context(&choice, &tools, None, false, StructuralTagSchemaMode::Auto);
        let value = serde_json::to_value(build_kimi_k2(&ctx).unwrap().unwrap()).unwrap();
        let call = &value["format"]["elements"][1];

        assert_eq!(call["type"], "tag");
        assert_eq!(call["begin"], "<|tool_call_begin|>functions.get_time:");
        assert_eq!(call["content"]["elements"][2]["json_schema"], true);
    }

    #[test]
    fn auto_is_triggered_but_inner_section_requires_a_call() {
        let tools = tools();
        let ctx = context(
            &ToolChoice::Auto,
            &tools,
            None,
            false,
            StructuralTagSchemaMode::Auto,
        );
        let value = serde_json::to_value(build_kimi_k2(&ctx).unwrap().unwrap()).unwrap();

        assert_eq!(value["format"]["type"], "triggered_tags");
        assert_eq!(value["format"]["at_least_one"], false);
        assert_eq!(value["format"]["triggers"][0], TOOL_CALLS_SECTION_BEGIN);
        assert_eq!(value["format"]["tags"][0]["content"]["at_least_one"], true);
    }

    #[test]
    fn required_honors_single_call_and_reasoning_prefix() {
        let tools = tools();
        let ctx = context(
            &ToolChoice::Required,
            &tools,
            Some(false),
            true,
            StructuralTagSchemaMode::Strict,
        );
        let builder = crate::tool_calling::StructuralTagBuilder::KimiK2;
        let value = builder.build_tool_call_format(&ctx).unwrap().unwrap();

        assert_eq!(value["format"]["type"], "sequence");
        assert_eq!(value["format"]["elements"][0]["end"], "</think>");
        let native = &value["format"]["elements"][1];
        assert_eq!(
            native["elements"][1]["stop_after_first"], true,
            "parallel_tool_calls=false must stop after the first native call"
        );
        assert_eq!(
            native["elements"][1]["tags"][1]["content"]["elements"][2]["json_schema"],
            tools[1].parameters.clone().unwrap(),
            "global strict mode overrides explicit strict=false"
        );
    }
}
