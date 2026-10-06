// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::builder::{
    ToolCallFormatBuildContext, kimi_uses_declared_tool_schema, resolve_tools_to_include,
};
use super::format::*;
use crate::tool_calling::ToolChoice;
#[cfg(test)]
use dynamo_structural_tag::kimi_k3::*;

pub(crate) fn build_kimi_k3(
    ctx: &ToolCallFormatBuildContext<'_>,
) -> anyhow::Result<Option<StructuralTag>> {
    let (tools, at_least_one) = resolve_tools_to_include(ctx)?;
    if tools.is_empty() {
        return Ok(None);
    }
    let relaxed = serde_json::Value::Bool(true);
    let selected: Vec<_> = tools
        .iter()
        .map(|tool| dynamo_structural_tag::Tool {
            name: &tool.name,
            parameters: if kimi_uses_declared_tool_schema(tool, ctx.strict_schema()) {
                tool.parameters.as_ref().unwrap_or(&relaxed)
            } else {
                &relaxed
            },
        })
        .collect();
    let response = if matches!(ctx.tool_choice, ToolChoice::Named(_)) {
        Format::ConstString(ConstStringFormat {
            value: String::new(),
        })
    } else {
        Format::AnyText(AnyTextFormat { excludes: vec![] })
    };
    Ok(Some(dynamo_structural_tag::kimi_k3::build(
        &selected,
        at_least_one,
        ctx.stop_after_first(),
        response,
    )))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::tool_calling::structural_tag::builder::StructuralTagSchemaMode;
    use crate::tool_calling::{ToolChoice, ToolDefinition};

    fn tools() -> Vec<ToolDefinition> {
        vec![
            ToolDefinition {
                name: "get_weather".to_string(),
                parameters: Some(json!({
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "days": {"type": "integer"}
                    },
                    "required": ["city"]
                })),
                strict: None,
            },
            ToolDefinition {
                name: "run_command".to_string(),
                parameters: Some(json!({
                    "type": "object",
                    "properties": {"command": {"type": "string"}}
                })),
                strict: None,
            },
        ]
    }

    fn context<'a>(
        choice: &'a ToolChoice,
        tools: &'a [ToolDefinition],
    ) -> ToolCallFormatBuildContext<'a> {
        ToolCallFormatBuildContext {
            tool_choice: choice,
            tools,
            parallel_tool_calls: None,
            schema_mode: StructuralTagSchemaMode::Auto,
            starts_in_reasoning: false,
        }
    }

    #[test]
    fn named_choice_requires_only_selected_xtml_call() {
        let tools = tools();
        let choice = ToolChoice::Named("get_weather".to_string());
        let value =
            serde_json::to_value(build_kimi_k3(&context(&choice, &tools)).unwrap().unwrap())
                .unwrap();

        assert_eq!(value["type"], "structural_tag");
        assert_eq!(value["format"]["type"], "sequence");
        let tools_tag = &value["format"]["elements"][2];
        assert_eq!(tools_tag["begin"], TOOLS_OPEN);
        let calls = tools_tag["content"]["tags"].as_array().unwrap();
        assert_eq!(calls.len(), 1);
        assert!(
            calls[0]["begin"]
                .as_str()
                .unwrap()
                .contains("tool=\"get_weather\"")
        );
        assert!(
            !value.to_string().contains("tool=\\\"run_command\\\""),
            "a named choice must exclude every other tool"
        );
    }

    #[test]
    fn named_choice_requires_an_empty_response_body() {
        let tools = tools();
        let choice = ToolChoice::Named("get_weather".to_string());
        let value =
            serde_json::to_value(build_kimi_k3(&context(&choice, &tools)).unwrap().unwrap())
                .unwrap();

        let response_body = &value["format"]["elements"][1]["content"];
        assert_eq!(response_body["type"], "const_string");
        assert_eq!(response_body["value"], "");
    }

    #[test]
    fn non_named_choices_keep_the_existing_response_body() {
        let tools = tools();

        for choice in [ToolChoice::Auto, ToolChoice::Required] {
            let value =
                serde_json::to_value(build_kimi_k3(&context(&choice, &tools)).unwrap().unwrap())
                    .unwrap();
            let response_body = &value["format"]["elements"][1]["content"];

            assert_eq!(
                response_body["type"], "any_text",
                "{choice:?} must retain response text"
            );
            assert_eq!(
                response_body["excludes"],
                json!([]),
                "{choice:?} must retain the existing unrestricted response body"
            );
        }
    }

    #[test]
    fn named_choice_is_mandatory_and_auto_is_optional() {
        let tools = tools();
        let named = ToolChoice::Named("get_weather".to_string());
        let named_value =
            serde_json::to_value(build_kimi_k3(&context(&named, &tools)).unwrap().unwrap())
                .unwrap();
        assert_eq!(named_value["format"]["elements"][2]["type"], "tag");

        let auto = ToolChoice::Auto;
        let auto_value =
            serde_json::to_value(build_kimi_k3(&context(&auto, &tools)).unwrap().unwrap()).unwrap();
        assert_eq!(auto_value["format"]["elements"][2]["type"], "optional");
    }

    #[test]
    fn named_choice_keeps_declared_argument_schema() {
        let tools = tools();
        let choice = ToolChoice::Named("get_weather".to_string());
        let value =
            serde_json::to_value(build_kimi_k3(&context(&choice, &tools)).unwrap().unwrap())
                .unwrap();
        let call = &value["format"]["elements"][2]["content"]["tags"][0];
        let argument_alternatives = &call["content"]["elements"][2]["content"]["elements"];
        assert!(
            argument_alternatives
                .to_string()
                .contains("key=\\\"city\\\"")
        );
        assert!(
            argument_alternatives
                .to_string()
                .contains("key=\\\"days\\\"")
        );
    }

    #[test]
    fn explicit_non_strict_tool_uses_vllm_permissive_argument_shape() {
        let tools = vec![ToolDefinition {
            name: "get_weather".to_string(),
            parameters: Some(json!({
                "type": "object",
                "properties": {"city": {"type": "string"}}
            })),
            strict: Some(false),
        }];
        let choice = ToolChoice::Named("get_weather".to_string());
        let value =
            serde_json::to_value(build_kimi_k3(&context(&choice, &tools)).unwrap().unwrap())
                .unwrap();
        let call = &value["format"]["elements"][2]["content"]["tags"][0];
        let arguments = &call["content"]["elements"][2];

        assert_eq!(arguments["type"], "star");
        assert_eq!(arguments["content"]["begin"], "<|open|>argument ");
        assert!(
            !arguments.to_string().contains("key=\\\"city\\\""),
            "vLLM treats strict=false as a permissive argument schema"
        );
    }

    #[test]
    fn parallel_false_stops_after_the_first_k3_call() {
        let tools = tools();
        let choice = ToolChoice::Required;
        let ctx = ToolCallFormatBuildContext {
            tool_choice: &choice,
            tools: &tools,
            parallel_tool_calls: Some(false),
            schema_mode: StructuralTagSchemaMode::Auto,
            starts_in_reasoning: false,
        };
        let value = serde_json::to_value(build_kimi_k3(&ctx).unwrap().unwrap()).unwrap();

        assert_eq!(
            value["format"]["elements"][2]["content"]["stop_after_first"],
            true
        );
    }
}
