// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde_json::Value;

use super::templated_tool_calls::TemplatedToolCallFormat;
use super::{StructuralTagBuilder, ToolCallGrammar};
use crate::structural_tag::policy::ResolvedToolCallingPolicy;
use crate::structural_tag::wire::{
    Format, JsonSchemaFormat, JsonSchemaStyle, TagFormat, TriggeredTagsFormat,
};

pub(crate) static HARMONY: StructuralTagBuilder = StructuralTagBuilder::new(&HarmonyGrammar);

struct HarmonyGrammar;

impl HarmonyGrammar {
    fn final_tag(schema: &Value) -> TagFormat {
        TagFormat {
            begin: "<|channel|>final<|message|>".to_string(),
            content: Box::new(Format::JsonSchema(JsonSchemaFormat {
                json_schema: schema.clone(),
                style: JsonSchemaStyle::Json,
                any_order: false,
            })),
            // The engine consumes the turn stop after the JSON grammar completes.
            end: String::new(),
        }
    }

    fn structured_output(
        &self,
        schema: &Value,
        policy: Option<&ResolvedToolCallingPolicy<'_>>,
        tool_arguments_any_order: bool,
    ) -> Format {
        // vLLM considers Harmony reasoning already finished. The final-header
        // trigger must own activation; a bare JSON schema constrains analysis.
        let mut triggers = vec!["<|channel|>final".to_string()];
        let mut tags = vec![Self::final_tag(schema)];
        if let Some(policy) = policy {
            triggers.push(FORMAT.tool_call_trigger.to_string());
            tags.extend(FORMAT.tool_tags(policy, tool_arguments_any_order));
        }
        Format::TriggeredTags(TriggeredTagsFormat {
            triggers,
            tags,
            excludes: if policy.is_none() {
                FORMAT
                    .tool_call_excludes
                    .iter()
                    .map(|s| (*s).to_string())
                    .collect()
            } else {
                Vec::new()
            },
            // Requiring a tag also forbids text before it, including analysis.
            at_least_one: false,
            stop_after_first: policy.is_none_or(|policy| policy.stop_after_first()),
        })
    }
}

impl ToolCallGrammar for HarmonyGrammar {
    fn optional_reasoning(&self) -> bool {
        true
    }

    fn reasoning_begin(&self) -> Option<&'static str> {
        FORMAT.reasoning_begin()
    }
    fn reasoning_end(&self) -> Option<&'static str> {
        FORMAT.reasoning_end()
    }
    fn reasoning_suffix(&self) -> &'static str {
        FORMAT.reasoning_suffix()
    }
    fn tool_call_excludes(&self) -> &'static [&'static str] {
        FORMAT.tool_call_excludes()
    }
    fn default_exclude_special_tokens(&self) -> bool {
        FORMAT.default_exclude_special_tokens()
    }

    fn build_structured_output(&self, schema: &Value) -> Format {
        // With tools disabled, require the final payload after optional reasoning.
        Format::Tag(Self::final_tag(schema))
    }

    fn build_auto_with_structured_output(
        &self,
        policy: &ResolvedToolCallingPolicy<'_>,
        schema: &Value,
        tool_arguments_any_order: bool,
    ) -> anyhow::Result<Format> {
        Ok(self.structured_output(schema, Some(policy), tool_arguments_any_order))
    }

    fn build_triggered_calls(
        &self,
        policy: &ResolvedToolCallingPolicy<'_>,
        exclude_special_tokens: bool,
        tool_arguments_any_order: bool,
    ) -> anyhow::Result<Format> {
        FORMAT.build_triggered_calls(policy, exclude_special_tokens, tool_arguments_any_order)
    }

    fn build_tool_calls_only(
        &self,
        policy: &ResolvedToolCallingPolicy<'_>,
        tool_arguments_any_order: bool,
    ) -> anyhow::Result<Format> {
        FORMAT.build_tool_calls_only(policy, tool_arguments_any_order)
    }
}

// The generation prompt already supplies `<|start|>assistant`. Subsequent
// directed messages explicitly reopen the assistant role after the handoff.
static FORMAT: TemplatedToolCallFormat = TemplatedToolCallFormat {
    tool_call_begin_prefix: "<|channel|>commentary to=functions.",
    tool_call_begin_suffix: "<|constrain|>json<|message|>",
    // XGrammar rejects <|call|> as a grammar terminal while it is also a stop token.
    // https://github.com/mlc-ai/xgrammar/issues/701
    // tool_call_end: "<|call|>",
    tool_call_end: "",
    // Enter before the excluded `to=functions.` completes, preserving ordinary commentary.
    tool_call_trigger: "<|channel|>commentary to=functions",
    first_tool_call_prefix: "",
    tool_call_separator: "<|start|>assistant",
    arguments_style: JsonSchemaStyle::Json,
    reasoning_begin: Some("<|channel|>analysis<|message|>"),
    reasoning_end: Some("<|end|>"),
    reasoning_suffix: "<|start|>assistant",
    free_text_excludes: &["to=functions."],
    tool_call_excludes: &["to=functions."],
    exclude_special_tokens: false,
};
