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

pub(crate) static DEEPSEEK_V41: StructuralTagBuilder =
    StructuralTagBuilder::new(&DEEPSEEK_V41_FORMAT);

static DEEPSEEK_DSML_FORMAT: DeepSeekDsmlFormat = DeepSeekDsmlFormat {
    trigger: TRIGGER,
    block_begin: TOOL_CALLS_BEGIN,
    block_end: TOOL_CALLS_END,
    invoke_begin_prefix: INVOKE_BEGIN_PREFIX,
    invoke_end: INVOKE_END,
    excludes: &[TRIGGER, BARE_INVOKE_BEGIN],
    style: JsonSchemaStyle::DeepseekXml,
};
static DEEPSEEK_V41_FORMAT: DeepSeekDsmlFormat = DeepSeekDsmlFormat {
    trigger: "<｜DSML｜ calls>",
    block_begin: "<｜DSML｜ calls>\n",
    block_end: "</｜DSML｜ calls>",
    invoke_begin_prefix: "<｜DSML｜ invoke name=\"",
    invoke_end: "</｜DSML｜ invoke>\n",
    excludes: &["<｜DSML｜ calls>", "<｜DSML｜ invoke name="],
    style: JsonSchemaStyle::DeepseekV41Xml,
};

struct DeepSeekDsmlFormat {
    trigger: &'static str,
    block_begin: &'static str,
    block_end: &'static str,
    invoke_begin_prefix: &'static str,
    invoke_end: &'static str,
    excludes: &'static [&'static str],
    style: JsonSchemaStyle,
}

const TRIGGER: &str = "<｜DSML｜tool_calls>";
const TOOL_CALLS_PREFIX: &str = "\n\n";
const TOOL_CALLS_BEGIN: &str = "<｜DSML｜tool_calls>\n";
const TOOL_CALLS_END: &str = "</｜DSML｜tool_calls>";
const BARE_INVOKE_BEGIN: &str = "<｜DSML｜invoke name=";
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
                begin: [self.invoke_begin_prefix, &tool.name, INVOKE_BEGIN_SUFFIX].concat(),
                content: Box::new(Format::JsonSchema(JsonSchemaFormat {
                    json_schema: resolve_tool_schema(tool, policy.schema_mode),
                    style: self.style,
                    any_order: tool_arguments_any_order,
                })),
                end: self.invoke_end.to_string(),
            })
            .collect();

        TagFormat {
            begin: self.block_begin.to_string(),
            content: Box::new(Format::TagsWithSeparator(TagsWithSeparatorFormat {
                tags: invoke_tags,
                separator: INVOKE_SEPARATOR.to_string(),
                at_least_one: true,
                stop_after_first: policy.stop_after_first(),
            })),
            end: self.block_end.to_string(),
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
        self.excludes
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
            self.trigger,
            vec![self.tool_call_block(policy, tool_arguments_any_order)],
            self.reasoning_begin(),
            self.reasoning_end(),
            &self.excludes[1..],
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

    use super::{DEEPSEEK_DSML, DEEPSEEK_V41};
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
    #[test]
    fn v41_is_a_distinct_registered_unified_grammar() {
        let builder = crate::structural_tag_builder_for_family("deepseek_v41").unwrap();
        assert!(std::ptr::eq(builder, &DEEPSEEK_V41));
        assert!(!std::ptr::eq(builder, &DEEPSEEK_DSML));
    }

    #[test]
    fn v41_auto_preserves_its_schema_and_native_argument_style() {
        let tools = tools();
        let tag = build(
            &DEEPSEEK_V41,
            StructuralTagToolChoice::Auto,
            &tools,
            Some(false),
            StructuralTagSchemaMode::Strict,
            false,
        );
        let outer = &tag["format"];
        assert_eq!(outer["type"], "triggered_tags");
        assert_eq!(outer["triggers"], json!(["<｜DSML｜ calls>"]));
        assert_eq!(outer["at_least_one"], false);
        let block = &outer["tags"][0];
        assert_eq!(block["begin"], "<｜DSML｜ calls>\n");
        assert_eq!(block["end"], "</｜DSML｜ calls>");
        let invokes = block["content"]["tags"].as_array().unwrap();
        assert_eq!(invokes.len(), 2);
        for (invoke, tool) in invokes.iter().zip(&tools) {
            assert_eq!(
                invoke["begin"],
                format!("<｜DSML｜ invoke name=\"{}\">\n", tool.name)
            );
            assert_eq!(invoke["end"], "</｜DSML｜ invoke>\n");
            assert_eq!(invoke["content"]["style"], "deepseek_v4_1_xml");
            assert_eq!(invoke["content"]["json_schema"], tool.parameters);
        }
        assert_eq!(block["content"]["stop_after_first"], true);
        let encoded = serde_json::to_string(&tag).unwrap();
        assert!(!encoded.contains("<｜DSML｜tool_calls>"));
        assert!(!encoded.contains("<｜DSML｜invoke"));
    }

    #[test]
    fn v41_required_and_named_keep_call_cardinality() {
        let tools = tools();
        let required = build(
            &DEEPSEEK_V41,
            StructuralTagToolChoice::Required,
            &tools,
            Some(false),
            StructuralTagSchemaMode::Strict,
            false,
        );
        assert_eq!(required["format"]["elements"][0]["value"], "\n\n");
        let calls = &required["format"]["elements"][1];
        assert_eq!(calls["triggers"], json!(["<｜DSML｜ calls>"]));
        assert_eq!(calls["at_least_one"], true);
        assert_eq!(calls["tags"][0]["content"]["stop_after_first"], true);
        let named = build(
            &DEEPSEEK_V41,
            StructuralTagToolChoice::Named("get_weather"),
            &tools,
            None,
            StructuralTagSchemaMode::Strict,
            false,
        );
        let block = &named["format"];
        assert_eq!(block["begin"], "\n\n<｜DSML｜ calls>\n");
        let invokes = block["content"]["tags"].as_array().unwrap();
        assert_eq!(invokes.len(), 1);
        assert_eq!(
            invokes[0]["begin"],
            "<｜DSML｜ invoke name=\"get_weather\">\n"
        );
        assert_eq!(invokes[0]["content"]["style"], "deepseek_v4_1_xml");
    }

    #[test]
    fn v41_none_and_prompt_opened_reasoning_use_v41_markers() {
        let tools = tools();
        let none = build(
            &DEEPSEEK_V41,
            StructuralTagToolChoice::None,
            &tools,
            None,
            StructuralTagSchemaMode::Strict,
            false,
        );
        assert_eq!(
            none["format"]["content"]["excludes"],
            json!(["<｜DSML｜ calls>", "<｜DSML｜ invoke name="])
        );
        let reasoning = build(
            &DEEPSEEK_V41,
            StructuralTagToolChoice::Required,
            &tools,
            Some(false),
            StructuralTagSchemaMode::Strict,
            true,
        );
        let encoded = serde_json::to_string(&reasoning).unwrap();
        assert!(encoded.contains("</think>"));
        assert!(encoded.contains("<｜DSML｜ calls>"));
    }
}
