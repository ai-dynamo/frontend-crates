// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{StructuralTagBuilder, ToolCallGrammar, triggered_calls_format};
use crate::structural_tag::policy::{
    ResolvedToolCallingPolicy, ToolCallingMode, resolve_tool_schema,
};
use crate::structural_tag::wire::{
    ConstStringFormat, Format, JsonSchemaFormat, JsonSchemaStyle, SequenceFormat, TagFormat,
    TagsWithSeparatorFormat,
};

pub(crate) static DEEPSEEK_DSML: StructuralTagBuilder =
    StructuralTagBuilder::new(&DEEPSEEK_DSML_FORMAT);

static DEEPSEEK_DSML_FORMAT: DeepSeekDsmlFormat = DeepSeekDsmlFormat;

struct DeepSeekDsmlFormat;

const TRIGGER: &str = "<｜DSML｜tool_calls>";
const TOOL_CALLS_PREFIX: &str = "\n\n";
const TOOL_CALLS_BEGIN: &str = "<｜DSML｜tool_calls>\n";
const TOOL_CALLS_END: &str = "</｜DSML｜tool_calls>";
const INVOKE_BEGIN_PREFIX: &str = "<｜DSML｜invoke name=\"";
const INVOKE_BEGIN_SUFFIX: &str = "\">\n";
const INVOKE_END: &str = "</｜DSML｜invoke>\n";
const INVOKE_SEPARATOR: &str = "";
const REASONING_BEGIN: &str = "<think>";
const REASONING_END: &str = "</think>";

impl DeepSeekDsmlFormat {
    fn tool_call_block(
        &self,
        policy: &ResolvedToolCallingPolicy<'_>,
        tool_arguments_any_order: bool,
    ) -> TagFormat {
        let invoke_tags = policy
            .tools
            .iter()
            .map(|tool| TagFormat {
                begin: [INVOKE_BEGIN_PREFIX, &tool.name, INVOKE_BEGIN_SUFFIX].concat(),
                content: Box::new(Format::JsonSchema(JsonSchemaFormat {
                    json_schema: resolve_tool_schema(tool, policy.schema_mode),
                    style: JsonSchemaStyle::DeepseekXml,
                    any_order: tool_arguments_any_order,
                })),
                end: INVOKE_END.to_string(),
            })
            .collect();

        TagFormat {
            begin: TOOL_CALLS_BEGIN.to_string(),
            content: Box::new(Format::TagsWithSeparator(TagsWithSeparatorFormat {
                tags: invoke_tags,
                separator: INVOKE_SEPARATOR.to_string(),
                at_least_one: true,
                stop_after_first: policy.stop_after_first(),
            })),
            end: TOOL_CALLS_END.to_string(),
        }
    }
}

impl ToolCallGrammar for DeepSeekDsmlFormat {
    fn reasoning_begin(&self) -> Option<&'static str> {
        Some(REASONING_BEGIN)
    }

    fn reasoning_end(&self) -> Option<&'static str> {
        Some(REASONING_END)
    }

    fn tool_call_excludes(&self) -> &'static [&'static str] {
        &["<｜DSML｜"]
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

        let format = triggered_calls_format(
            TRIGGER,
            vec![self.tool_call_block(policy, tool_arguments_any_order)],
            self.reasoning_begin(),
            self.reasoning_end(),
            exclude_special_tokens,
            policy,
        );
        if policy.mode != ToolCallingMode::Required {
            return Ok(format);
        }

        Ok(Format::Sequence(SequenceFormat {
            elements: vec![
                Format::ConstString(ConstStringFormat {
                    value: TOOL_CALLS_PREFIX.to_string(),
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
        let mut block = self.tool_call_block(policy, tool_arguments_any_order);
        block.begin.insert_str(0, TOOL_CALLS_PREFIX);
        Ok(Format::Tag(block))
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::DEEPSEEK_DSML;
    use crate::structural_tag::test_support::{build, tools};
    use crate::structural_tag::{StructuralTagSchemaMode, StructuralTagToolChoice};

    #[test]
    fn named_builds_one_dsml_invoke() {
        let tools = tools();
        let actual = build(
            &DEEPSEEK_DSML,
            StructuralTagToolChoice::Named("get_weather"),
            &tools,
            Some(true),
            StructuralTagSchemaMode::Auto,
            false,
        );

        assert_eq!(
            actual,
            json!({
                "type": "structural_tag",
                "format": {
                    "type": "tag",
                    "begin": "\n\n<｜DSML｜tool_calls>\n",
                    "content": {
                        "type": "tags_with_separator",
                        "tags": [{
                            "type": "tag",
                            "begin": "<｜DSML｜invoke name=\"get_weather\">\n",
                            "content": {
                                "type": "json_schema",
                                "json_schema": tools[1].parameters,
                                "style": "deepseek_xml"
                            },
                            "end": "</｜DSML｜invoke>\n"
                        }],
                        "separator": "",
                        "at_least_one": true,
                        "stop_after_first": true
                    },
                    "end": "</｜DSML｜tool_calls>"
                }
            })
        );
    }

    #[test]
    fn required_starts_with_a_triggered_dsml_block() {
        let tools = tools();
        let actual = build(
            &DEEPSEEK_DSML,
            StructuralTagToolChoice::Required,
            &tools,
            Some(true),
            StructuralTagSchemaMode::Auto,
            false,
        );

        assert_eq!(actual["format"]["type"], "sequence");
        assert_eq!(actual["format"]["elements"][0]["type"], "const_string");
        assert_eq!(actual["format"]["elements"][0]["value"], "\n\n");
        let calls = &actual["format"]["elements"][1];
        assert_eq!(calls["type"], "triggered_tags");
        assert_eq!(calls["triggers"], json!(["<｜DSML｜tool_calls>"]));
        assert_eq!(calls["at_least_one"], true);
        assert_eq!(calls["stop_after_first"], false);
        assert_eq!(
            calls["tags"][0]["content"]["tags"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
        assert_eq!(calls["tags"][0]["content"]["at_least_one"], true);
    }

    #[test]
    fn auto_builds_triggered_tool_dispatch() {
        let tools = tools();
        let actual = build(
            &DEEPSEEK_DSML,
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
                    "triggers": ["<｜DSML｜tool_calls>"],
                    "tags": [{
                        "type": "tag",
                        "begin": "<｜DSML｜tool_calls>\n",
                        "content": {
                            "type": "tags_with_separator",
                            "tags": [
                                {
                                    "type": "tag",
                                    "begin": "<｜DSML｜invoke name=\"add_numbers\">\n",
                                    "content": {
                                        "type": "json_schema",
                                        "json_schema": true,
                                        "style": "deepseek_xml"
                                    },
                                    "end": "</｜DSML｜invoke>\n"
                                },
                                {
                                    "type": "tag",
                                    "begin": "<｜DSML｜invoke name=\"get_weather\">\n",
                                    "content": {
                                        "type": "json_schema",
                                        "json_schema": tools[1].parameters,
                                        "style": "deepseek_xml"
                                    },
                                    "end": "</｜DSML｜invoke>\n"
                                }
                            ],
                            "separator": "",
                            "at_least_one": true,
                            "stop_after_first": false
                        },
                        "end": "</｜DSML｜tool_calls>"
                    }],
                    "excludes": [
                        "<think>",
                        "</think>"
                    ],
                    "at_least_one": false,
                    "stop_after_first": false
                }
            })
        );
    }

    #[test]
    fn none_excludes_dsml_markup() {
        let tools = tools();
        let actual = build(
            &DEEPSEEK_DSML,
            StructuralTagToolChoice::None,
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
                    "type": "tag",
                    "begin": "",
                    "content": {
                        "type": "any_text",
                        "excludes": ["<｜DSML｜"]
                    },
                    "end": ""
                }
            })
        );
    }
}
