// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{ToolCallGrammar, triggered_calls_format};
use crate::Tool;
use crate::structural_tag::policy::{
    ResolvedToolCallingPolicy, ToolCallingMode, resolve_tool_schema,
};
use crate::structural_tag::wire::{
    ConstStringFormat, Format, JsonSchemaFormat, JsonSchemaStyle, SequenceFormat, TagFormat,
    TagsWithSeparatorFormat,
};

/// Structural-tag grammar for flat, repeated tool-call formats.
///
/// For every resolved tool it creates:
///
/// ```text
/// Tag {
///     begin: tool_call_begin_prefix + tool_name + tool_call_begin_suffix,
///     content: JsonSchema(arguments, arguments_style),
///     end: tool_call_end,
/// }
/// ```
///
/// The resulting structural-tag format depends on tool choice as follows
/// (`tool_tags` is the set of tags above):
///
/// ```text
/// Auto:
///     TriggeredTags {
///         triggers: [tool_call_trigger],
///         tags: tool_tags,
///         at_least_one: false,
///     }
///
/// Required:
///     Sequence {
///         elements: [
///             ConstString(first_tool_call_prefix),
///             TriggeredTags {
///                 triggers: [tool_call_trigger],
///                 tags: tool_tags,
///                 at_least_one: true,
///             },
///         ],
///     }
///
/// Named:
///     Tag {
///         begin: first_tool_call_prefix
///             + tool_call_begin_prefix + selected_tool_name
///             + tool_call_begin_suffix,
///         content: JsonSchema(selected_tool_arguments, arguments_style),
///         end: tool_call_end,
///     }
/// ```
///
/// When `first_tool_call_prefix` is empty, `Required` returns `TriggeredTags`
/// directly instead of wrapping it in a `Sequence`.
///
/// If generation starts inside reasoning, the common builder first closes the
/// reasoning tag and then emits `reasoning_suffix` before the format above.
/// `free_text_excludes` reserves model control strings in `TriggeredTags`
/// without preventing the configured trigger itself.
///
/// `tool_call_trigger` must be a prefix of `tool_call_begin_prefix`. Optional
/// whitespace before a call belongs in `first_tool_call_prefix`, so it is
/// omitted from the `Auto` trigger but emitted before the first `Required`
/// call and in tool-calls-only formats. Tool-calls-only formats are used for
/// named choice and the tool-call branch combined with structured output.
///
/// Use this format when calls form a flat sequence, the tool name is inserted
/// verbatim at one fixed position, and the arguments are the only constrained
/// content inside each call. Use a dedicated [`ToolCallGrammar`] for more
/// complex formats that cannot be expressed by this template.
pub(super) struct TemplatedToolCallFormat {
    /// Fixed bytes immediately before the tool name in every call.
    pub tool_call_begin_prefix: &'static str,
    /// Fixed bytes immediately after the tool name and before its arguments.
    pub tool_call_begin_suffix: &'static str,
    /// Fixed bytes closing every call after its arguments.
    pub tool_call_end: &'static str,
    /// Tool-independent prefix that activates a tag in `TriggeredTags`.
    pub tool_call_trigger: &'static str,
    /// Bytes emitted once before the first call when tool output must begin immediately.
    pub first_tool_call_prefix: &'static str,
    /// Bytes emitted between adjacent calls in tool-calls-only formats.
    pub tool_call_separator: &'static str,
    /// XGrammar JSON Schema encoding used for tool arguments.
    pub arguments_style: JsonSchemaStyle,
    /// Model reasoning marker excluded from free text when configured.
    pub reasoning_begin: Option<&'static str>,
    /// Model reasoning close, also used to close an active reasoning block.
    pub reasoning_end: Option<&'static str>,
    /// Bytes emitted after a structural-tag-owned reasoning close.
    pub reasoning_suffix: &'static str,
    /// Reserved strings excluded from free text outside tool calls.
    pub free_text_excludes: &'static [&'static str],
    /// Strings excluded for `tool_choice = none` and, when enabled, reasoning.
    pub tool_call_excludes: &'static [&'static str],
    /// Default for excluding reasoning and tool-call special strings.
    pub exclude_special_tokens: bool,
}

impl TemplatedToolCallFormat {
    fn tool_tag(
        &self,
        tool: &Tool,
        policy: &ResolvedToolCallingPolicy<'_>,
        tool_arguments_any_order: bool,
        leading_prefix: &str,
    ) -> TagFormat {
        let mut begin = String::with_capacity(
            leading_prefix.len()
                + self.tool_call_begin_prefix.len()
                + tool.name.len()
                + self.tool_call_begin_suffix.len(),
        );
        begin.push_str(leading_prefix);
        begin.push_str(self.tool_call_begin_prefix);
        begin.push_str(&tool.name);
        begin.push_str(self.tool_call_begin_suffix);

        TagFormat {
            begin,
            content: Box::new(Format::JsonSchema(JsonSchemaFormat {
                json_schema: resolve_tool_schema(tool, policy.schema_mode),
                style: self.arguments_style,
                any_order: tool_arguments_any_order,
            })),
            end: self.tool_call_end.to_string(),
        }
    }

    fn tool_tags(
        &self,
        policy: &ResolvedToolCallingPolicy<'_>,
        tool_arguments_any_order: bool,
    ) -> Vec<TagFormat> {
        policy
            .tools
            .iter()
            .map(|tool| self.tool_tag(tool, policy, tool_arguments_any_order, ""))
            .collect()
    }
}

impl ToolCallGrammar for TemplatedToolCallFormat {
    fn reasoning_begin(&self) -> Option<&'static str> {
        self.reasoning_begin
    }

    fn reasoning_end(&self) -> Option<&'static str> {
        self.reasoning_end
    }

    fn reasoning_suffix(&self) -> &'static str {
        self.reasoning_suffix
    }

    fn tool_call_excludes(&self) -> &'static [&'static str] {
        self.tool_call_excludes
    }

    fn default_exclude_special_tokens(&self) -> bool {
        self.exclude_special_tokens
    }

    fn build_triggered_calls(
        &self,
        policy: &ResolvedToolCallingPolicy<'_>,
        exclude_special_tokens: bool,
        tool_arguments_any_order: bool,
    ) -> anyhow::Result<Format> {
        anyhow::ensure!(
            matches!(
                policy.mode,
                ToolCallingMode::Auto | ToolCallingMode::Required
            ),
            "auto or required policy expected"
        );
        anyhow::ensure!(
            self.tool_call_begin_prefix
                .starts_with(self.tool_call_trigger),
            "tool call trigger must be a prefix of the tool call begin prefix"
        );

        let format = triggered_calls_format(
            self.tool_call_trigger,
            self.tool_tags(policy, tool_arguments_any_order),
            self.reasoning_begin,
            self.reasoning_end,
            self.free_text_excludes,
            exclude_special_tokens,
            policy,
        );
        if policy.mode != ToolCallingMode::Required || self.first_tool_call_prefix.is_empty() {
            return Ok(format);
        }

        Ok(Format::Sequence(SequenceFormat {
            elements: vec![
                Format::ConstString(ConstStringFormat {
                    value: self.first_tool_call_prefix.to_string(),
                }),
                format,
            ],
        }))
    }

    fn build_tool_calls_only(
        &self,
        policy: &ResolvedToolCallingPolicy<'_>,
        tool_arguments_any_order: bool,
    ) -> anyhow::Result<Format> {
        if policy.mode == ToolCallingMode::Named {
            anyhow::ensure!(
                policy.tools.len() == 1,
                "named policy must contain exactly one tool"
            );
            return Ok(Format::Tag(self.tool_tag(
                policy.tools[0],
                policy,
                tool_arguments_any_order,
                self.first_tool_call_prefix,
            )));
        }

        let calls = Format::TagsWithSeparator(TagsWithSeparatorFormat {
            tags: self.tool_tags(policy, tool_arguments_any_order),
            separator: self.tool_call_separator.to_string(),
            at_least_one: true,
            stop_after_first: policy.stop_after_first(),
        });
        if self.first_tool_call_prefix.is_empty() {
            return Ok(calls);
        }

        Ok(Format::Tag(TagFormat {
            begin: self.first_tool_call_prefix.to_string(),
            content: Box::new(calls),
            end: String::new(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::super::StructuralTagBuilder;
    use super::super::qwen3_coder::QWEN3_CODER;
    use super::TemplatedToolCallFormat;
    use crate::structural_tag::test_support::{build, tools};
    use crate::structural_tag::wire::JsonSchemaStyle;
    use crate::structural_tag::{
        StructuralTagContext, StructuralTagOptions, StructuralTagSchemaMode,
        StructuralTagToolChoice,
    };

    static NO_BAN_FORMAT: TemplatedToolCallFormat = TemplatedToolCallFormat {
        tool_call_begin_prefix: "<function=",
        tool_call_begin_suffix: ">",
        tool_call_end: "</function>",
        tool_call_trigger: "<function=",
        first_tool_call_prefix: "",
        tool_call_separator: "",
        arguments_style: JsonSchemaStyle::QwenXml,
        reasoning_begin: None,
        reasoning_end: None,
        reasoning_suffix: "",
        free_text_excludes: &[],
        tool_call_excludes: &[],
        exclude_special_tokens: true,
    };

    static NO_BAN_BUILDER: StructuralTagBuilder = StructuralTagBuilder::new(&NO_BAN_FORMAT);

    #[test]
    fn required_starts_with_a_triggered_tool_call() {
        let tools = tools();
        let actual = build(
            &QWEN3_CODER,
            StructuralTagToolChoice::Required,
            &tools,
            None,
            StructuralTagSchemaMode::Auto,
            false,
        );

        let calls = &actual["format"];
        assert_eq!(calls["type"], "triggered_tags");
        assert_eq!(calls["at_least_one"], true);
        assert_eq!(calls["stop_after_first"], false);
        assert_eq!(calls["triggers"], json!(["<tool_call>\n<function="]));
        assert_eq!(
            calls["tags"][0]["begin"],
            "<tool_call>\n<function=add_numbers>\n"
        );
    }

    #[test]
    fn named_tool_choice_allows_exactly_one_call() {
        let tools = tools();
        let actual = build(
            &QWEN3_CODER,
            StructuralTagToolChoice::Named("get_weather"),
            &tools,
            Some(true),
            StructuralTagSchemaMode::Auto,
            false,
        );

        assert_eq!(actual["format"]["type"], "tag");
        assert_eq!(
            actual["format"]["begin"],
            "<tool_call>\n<function=get_weather>\n"
        );
        assert!(actual["format"].get("triggers").is_none());
    }

    #[test]
    fn required_respects_parallel_tool_calls_false() {
        let tools = tools();
        let actual = build(
            &QWEN3_CODER,
            StructuralTagToolChoice::Required,
            &tools,
            Some(false),
            StructuralTagSchemaMode::Auto,
            false,
        );

        assert_eq!(actual["format"]["stop_after_first"], true);
    }

    #[test]
    fn auto_builds_triggered_tool_dispatch() {
        let tools = tools();
        let actual = build(
            &QWEN3_CODER,
            StructuralTagToolChoice::Auto,
            &tools,
            None,
            StructuralTagSchemaMode::Auto,
            false,
        );

        assert_eq!(
            actual,
            json!({
                "type": "structural_tag",
                "format": {
                    "type": "triggered_tags",
                    "triggers": ["<tool_call>\n<function="],
                    "tags": [
                        {
                            "type": "tag",
                            "begin": "<tool_call>\n<function=add_numbers>\n",
                            "content": {
                                "type": "json_schema",
                                "json_schema": true,
                                "style": "qwen_xml"
                            },
                            "end": "\n</function>\n</tool_call>"
                        },
                        {
                            "type": "tag",
                            "begin": "<tool_call>\n<function=get_weather>\n",
                            "content": {
                                "type": "json_schema",
                                "json_schema": tools[1].parameters,
                                "style": "qwen_xml"
                            },
                            "end": "\n</function>\n</tool_call>"
                        }
                    ],
                    "excludes": ["<think>", "</think>"],
                    "at_least_one": false,
                    "stop_after_first": false
                }
            })
        );
    }

    #[test]
    fn tool_call_excludes_do_not_depend_on_the_tools_list() {
        let actual = build(
            &QWEN3_CODER,
            StructuralTagToolChoice::None,
            &[],
            None,
            StructuralTagSchemaMode::Auto,
            false,
        );

        assert_eq!(
            actual,
            json!({
                "type": "structural_tag",
                "format": {
                    "type": "tag",
                    "begin": "",
                    "content": {
                        "type": "any_text",
                        "excludes": ["<tool_call>", "<function="]
                    },
                    "end": ""
                }
            })
        );
    }

    #[test]
    fn empty_tool_call_excludes_build_no_constraint() {
        let result = NO_BAN_BUILDER
            .build(&StructuralTagContext {
                tool_choice: StructuralTagToolChoice::None,
                tools: &[],
                parallel_tool_calls: None,
                schema_mode: StructuralTagSchemaMode::Auto,
                structured_output_schema: None,
                starts_in_reasoning: false,
            })
            .expect("tool-call ban should build");

        assert!(result.is_none());
    }

    #[test]
    fn config_can_disable_special_token_exclusion() {
        let tools = tools();
        let actual = QWEN3_CODER
            .build_with_options(
                &StructuralTagContext {
                    tool_choice: StructuralTagToolChoice::Required,
                    tools: &tools,
                    parallel_tool_calls: None,
                    schema_mode: StructuralTagSchemaMode::Auto,
                    structured_output_schema: None,
                    starts_in_reasoning: true,
                },
                &StructuralTagOptions {
                    exclude_special_tokens: Some(false),
                    ..Default::default()
                },
            )
            .expect("structural tag should build")
            .expect("required tool choice should need a structural tag");

        assert_eq!(
            actual["format"]["elements"][0]["content"],
            json!({
                "type": "any_text",
                "excludes": []
            })
        );
        assert!(actual["format"]["elements"][2].get("excludes").is_none());
    }
}
