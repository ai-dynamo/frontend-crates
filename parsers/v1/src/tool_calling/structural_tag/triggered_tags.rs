// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! [`StructuralTagBuilder::TriggeredTags`] implementation.
//!
//! Builds xgrammar `triggered_tags` tool-call formats.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::TOOL_NAME_PLACEHOLDER;
use super::builder::{ToolCallFormatBuildContext, resolve_tool_schema, resolve_tools_to_include};
use super::format::{
    ConstStringFormat, Format, JsonSchemaFormat, JsonSchemaStyle, StructuralTag, TagFormat,
    TriggeredTagsFormat,
};

/// Configuration for `triggered_tags` formats where each tool is a tag.
///
/// The model can emit free text until a trigger is encountered, then guided
/// decoding constrains the output to the matching tool call structure.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TriggeredTagsConfig {
    /// Begin tag template; [`TOOL_NAME_PLACEHOLDER`] is replaced per tool.
    pub begin_template: String,

    /// End tag appended after each tool call's content.
    pub end_template: String,

    /// Trigger patterns that signal the start of a tool call in the output.
    pub triggers: Vec<String>,

    /// Content style passed to xgrammar.
    #[serde(default)]
    pub content_style: JsonSchemaStyle,

    /// Tokens to ban when `tool_choice="none"`.
    #[serde(default)]
    pub tool_call_ban_tokens: Vec<String>,

    /// Closing tag for prompt-injected reasoning, if this parser supports it.
    #[serde(default)]
    pub reasoning_end: Option<String>,
}

/// Recognize a simple schema whose only valid argument value is `{}`.
/// Its XML body can be an empty literal, avoiding XGrammar 0.1.32's compilation
/// error and the unbounded whitespace branch observed with 0.2.7.
fn is_closed_empty_object(schema: &Value) -> bool {
    let Some(schema) = schema.as_object() else {
        return false;
    };
    schema.get("type").and_then(Value::as_str) == Some("object")
        && schema
            .get("properties")
            .and_then(Value::as_object)
            .is_some_and(|p| p.is_empty())
        && schema.get("additionalProperties") == Some(&Value::Bool(false))
        && schema
            .get("required")
            .is_none_or(|r| r.as_array().is_some_and(|r| r.is_empty()))
        // Other keywords may change validity or reference scope; keep their schema path.
        && schema.keys().all(|key| {
            matches!(
                key.as_str(),
                "type" | "properties" | "required" | "additionalProperties"
            )
        })
}

/// Build a `triggered_tags` structural tag from config and request context.
pub(crate) fn build_triggered_tags(
    config: &TriggeredTagsConfig,
    ctx: &ToolCallFormatBuildContext<'_>,
) -> anyhow::Result<Option<StructuralTag>> {
    let (tools_to_include, at_least_one) = resolve_tools_to_include(ctx)?;

    if tools_to_include.is_empty() {
        return Ok(None);
    }

    let strict_schema = ctx.strict_schema();

    let tags: Vec<TagFormat> = tools_to_include
        .iter()
        .map(|tool| {
            // function name validated by validate_tools() in the request handler
            let begin = config
                .begin_template
                .replace(TOOL_NAME_PLACEHOLDER, &tool.name);

            let json_schema = resolve_tool_schema(tool, strict_schema);
            let content = if config.content_style == JsonSchemaStyle::QwenXml
                && is_closed_empty_object(&json_schema)
            {
                // No XML parameters represent {}. Avoid old XGrammar's empty-object
                // compilation error and newer versions' unbounded whitespace body.
                Format::ConstString(ConstStringFormat {
                    value: String::new(),
                })
            } else {
                Format::JsonSchema(JsonSchemaFormat {
                    json_schema,
                    style: config.content_style,
                })
            };

            TagFormat {
                begin,
                content: Box::new(content),
                end: config.end_template.clone(),
            }
        })
        .collect();

    Ok(Some(StructuralTag {
        format: Format::TriggeredTags(TriggeredTagsFormat {
            triggers: config.triggers.clone(),
            tags,
            at_least_one,
            stop_after_first: ctx.stop_after_first(),
        }),
    }))
}
