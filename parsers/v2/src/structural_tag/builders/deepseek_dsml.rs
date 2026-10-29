// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::tool_calling::dsml;
use crate::unified::deepseek_v41;

use super::{StructuralTagBuilder, ToolCallGrammar, triggered_calls_format};
use crate::structural_tag::policy::{
    ResolvedToolCallingPolicy, ToolCallingMode, resolve_tool_schema,
};
use crate::structural_tag::wire::{
    ConstStringFormat, Format, JsonSchemaFormat, JsonSchemaStyle, SequenceFormat, TagFormat,
    TagsWithSeparatorFormat,
};

pub(crate) static DEEPSEEK_DSML: StructuralTagBuilder =
    StructuralTagBuilder::new(&DeepSeekDsmlFormat {
        dialect: DsmlDialect::V4,
    });
pub(crate) static DEEPSEEK_DSML_V41: StructuralTagBuilder =
    StructuralTagBuilder::new(&DeepSeekDsmlFormat {
        dialect: DsmlDialect::V41,
    });

#[derive(Clone, Copy)]
enum DsmlDialect {
    V4,
    V41,
}

struct DeepSeekDsmlFormat {
    dialect: DsmlDialect,
}

const BARE_INVOKE_BEGIN: &str = dsml::INVOKE_START_PREFIX
    .split_at(dsml::INVOKE_START_PREFIX.len() - 1)
    .0;
const V41_BARE_INVOKE_BEGIN: &str = deepseek_v41::INVOKE_START
    .split_at(deepseek_v41::INVOKE_START.len() - 1)
    .0;
const REASONING_BEGIN: &str = "<think>";
const REASONING_END: &str = "</think>";

impl DeepSeekDsmlFormat {
    fn block_start(&self) -> &'static str {
        match self.dialect {
            DsmlDialect::V4 => dsml::BLOCK_START,
            DsmlDialect::V41 => deepseek_v41::BLOCK_START,
        }
    }

    fn block_end(&self) -> &'static str {
        match self.dialect {
            DsmlDialect::V4 => dsml::BLOCK_END,
            DsmlDialect::V41 => deepseek_v41::BLOCK_END,
        }
    }

    fn invoke_start(&self) -> &'static str {
        match self.dialect {
            DsmlDialect::V4 => dsml::INVOKE_START_PREFIX,
            DsmlDialect::V41 => deepseek_v41::INVOKE_START,
        }
    }

    fn invoke_end(&self) -> &'static str {
        match self.dialect {
            DsmlDialect::V4 => dsml::INVOKE_END,
            DsmlDialect::V41 => deepseek_v41::INVOKE_END,
        }
    }

    fn argument_style(&self) -> JsonSchemaStyle {
        match self.dialect {
            DsmlDialect::V4 => JsonSchemaStyle::DeepseekXml,
            DsmlDialect::V41 => JsonSchemaStyle::DeepseekV41Xml,
        }
    }

    fn tool_call_block(
        &self,
        policy: &ResolvedToolCallingPolicy<'_>,
        tool_arguments_any_order: bool,
    ) -> TagFormat {
        let invoke_tags = policy
            .tools
            .iter()
            .map(|tool| TagFormat {
                begin: [self.invoke_start(), &tool.name, "\">\n"].concat(),
                content: Box::new(Format::JsonSchema(JsonSchemaFormat {
                    json_schema: resolve_tool_schema(tool, policy.schema_mode),
                    style: self.argument_style(),
                    any_order: tool_arguments_any_order,
                })),
                end: [self.invoke_end(), "\n"].concat(),
            })
            .collect();

        TagFormat {
            begin: [self.block_start(), "\n"].concat(),
            content: Box::new(Format::TagsWithSeparator(TagsWithSeparatorFormat {
                tags: invoke_tags,
                separator: String::new(),
                at_least_one: true,
                stop_after_first: policy.stop_after_first(),
            })),
            end: self.block_end().to_string(),
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
        match self.dialect {
            DsmlDialect::V4 => &[dsml::BLOCK_START, BARE_INVOKE_BEGIN],
            DsmlDialect::V41 => &[deepseek_v41::BLOCK_START, V41_BARE_INVOKE_BEGIN],
        }
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
            self.block_start(),
            vec![self.tool_call_block(policy, tool_arguments_any_order)],
            self.reasoning_begin(),
            self.reasoning_end(),
            &self.tool_call_excludes()[1..],
            exclude_special_tokens,
            policy,
        );
        if policy.mode != ToolCallingMode::Required {
            return Ok(format);
        }

        Ok(Format::Sequence(SequenceFormat {
            elements: vec![
                Format::ConstString(ConstStringFormat {
                    value: dsml::BLOCK_SEPARATOR.to_string(),
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
        block.begin.insert_str(0, dsml::BLOCK_SEPARATOR);
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
                        "</think>",
                        "<｜DSML｜invoke name="
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
                        "excludes": [
                            "<｜DSML｜tool_calls>",
                            "<｜DSML｜invoke name="
                        ]
                    },
                    "end": ""
                }
            })
        );
    }
}
