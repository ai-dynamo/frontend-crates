// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::builder::{
    ToolCallFormatBuildContext, resolve_tools_to_include, uses_declared_tool_schema,
};
use super::format::*;
use crate::tool_calling::ToolChoice;
#[cfg(test)]
use crate::tool_calling::ToolDefinition;
#[cfg(test)]
use dynamo_structural_tag::kimi_k2::*;

pub(crate) fn build_kimi_k2(
    ctx: &ToolCallFormatBuildContext<'_>,
) -> anyhow::Result<Option<StructuralTag>> {
    let (tools, _) = resolve_tools_to_include(ctx)?;
    if tools.is_empty() {
        return Ok(None);
    }
    let relaxed = serde_json::Value::Bool(true);
    let selected: Vec<_> = tools
        .iter()
        .map(|tool| dynamo_structural_tag::Tool {
            name: &tool.name,
            parameters: if uses_declared_tool_schema(tool, ctx.strict_schema()) {
                tool.parameters.as_ref().unwrap_or(&relaxed)
            } else {
                &relaxed
            },
        })
        .collect();
    Ok(Some(dynamo_structural_tag::kimi_k2::build(
        &selected,
        matches!(ctx.tool_choice, ToolChoice::Auto),
        matches!(ctx.tool_choice, ToolChoice::Named(_)),
        ctx.stop_after_first(),
    )))
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
