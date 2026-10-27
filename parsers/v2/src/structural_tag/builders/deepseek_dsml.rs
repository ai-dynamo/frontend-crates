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
use crate::unified::deepseek_v41;

pub(crate) static DEEPSEEK_DSML: StructuralTagBuilder =
    StructuralTagBuilder::new(&DEEPSEEK_DSML_FORMAT);
pub(crate) static DEEPSEEK_V41: StructuralTagBuilder =
    StructuralTagBuilder::new(&DEEPSEEK_V41_FORMAT);

static DEEPSEEK_DSML_FORMAT: DeepSeekDsmlFormat = DeepSeekDsmlFormat {
    tool_call_excludes: &["<｜DSML｜tool_calls>", "<｜DSML｜invoke name="],
    block_end: "</｜DSML｜tool_calls>",
    invoke_begin: "<｜DSML｜invoke name=\"",
    invoke_end: "</｜DSML｜invoke>",
    schema_style: JsonSchemaStyle::DeepseekXml,
};

static DEEPSEEK_V41_FORMAT: DeepSeekDsmlFormat = DeepSeekDsmlFormat {
    tool_call_excludes: &[deepseek_v41::BLOCK_START, "<｜DSML｜ invoke name="],
    block_end: deepseek_v41::BLOCK_END,
    invoke_begin: deepseek_v41::INVOKE_START,
    invoke_end: deepseek_v41::INVOKE_END,
    schema_style: JsonSchemaStyle::DeepseekV41Xml,
};

struct DeepSeekDsmlFormat {
    tool_call_excludes: &'static [&'static str; 2],
    block_end: &'static str,
    invoke_begin: &'static str,
    invoke_end: &'static str,
    schema_style: JsonSchemaStyle,
}

const TOOL_CALLS_PREFIX: &str = "\n\n";
const INVOKE_BEGIN_SUFFIX: &str = "\">\n";
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
                begin: [self.invoke_begin, &tool.name, INVOKE_BEGIN_SUFFIX].concat(),
                content: Box::new(Format::JsonSchema(JsonSchemaFormat {
                    json_schema: resolve_tool_schema(tool, policy.schema_mode),
                    style: self.schema_style,
                    any_order: tool_arguments_any_order,
                })),
                end: format!("{}\n", self.invoke_end),
            })
            .collect();

        TagFormat {
            begin: format!("{}\n", self.tool_call_excludes[0]),
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
        self.tool_call_excludes
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
            self.tool_call_excludes[0],
            vec![self.tool_call_block(policy, tool_arguments_any_order)],
            self.reasoning_begin(),
            self.reasoning_end(),
            &self.tool_call_excludes[1..],
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
    fn v41_uses_spaced_framing_and_parameter_schema_style() {
        let tools = tools();
        for choice in [
            StructuralTagToolChoice::Auto,
            StructuralTagToolChoice::Required,
            StructuralTagToolChoice::Named("get_weather"),
        ] {
            let actual = build(
                &DEEPSEEK_V41,
                choice,
                &tools,
                Some(false),
                StructuralTagSchemaMode::Strict,
                false,
            );
            let format = &actual["format"];
            let block = match choice {
                StructuralTagToolChoice::Auto => {
                    assert_eq!(format["triggers"], json!(["<｜DSML｜ calls>"]));
                    assert_eq!(format["at_least_one"], false);
                    &format["tags"][0]
                }
                StructuralTagToolChoice::Required => {
                    assert_eq!(format["elements"][0]["value"], "\n\n");
                    let dispatch = &format["elements"][1];
                    assert_eq!(dispatch["at_least_one"], true);
                    &dispatch["tags"][0]
                }
                StructuralTagToolChoice::Named(_) => format,
                StructuralTagToolChoice::None => unreachable!(),
            };
            assert!(
                block["begin"]
                    .as_str()
                    .unwrap()
                    .ends_with("<｜DSML｜ calls>\n")
            );
            assert_eq!(block["end"], "</｜DSML｜ calls>");
            assert_eq!(block["content"]["stop_after_first"], true);
            let calls = block["content"]["tags"].as_array().unwrap();
            assert_eq!(
                calls.len(),
                if matches!(choice, StructuralTagToolChoice::Named(_)) {
                    1
                } else {
                    2
                }
            );
            for call in calls {
                assert!(
                    call["begin"]
                        .as_str()
                        .unwrap()
                        .starts_with("<｜DSML｜ invoke name=\"")
                );
                assert_eq!(call["end"], "</｜DSML｜ invoke>\n");
                assert_eq!(call["content"]["style"], "deepseek_v4_1_xml");
                assert!(call["content"]["json_schema"].is_object());
            }
        }
    }

    #[test]
    fn v41_named_envelope_round_trips_through_the_unified_parser() {
        use crate::{UnifiedEvent, UnifiedParserExt, assemble, create_unified_parser_for_family};

        let tools = tools();
        let tag = build(
            &DEEPSEEK_V41,
            StructuralTagToolChoice::Named("get_weather"),
            &tools,
            None,
            StructuralTagSchemaMode::Auto,
            false,
        );
        let block = &tag["format"];
        let call = &block["content"]["tags"][0];
        let input = format!(
            "{}{}<｜DSML｜ parameter name=\"city\" string=\"true\">東京🙂</｜DSML｜ parameter>\n{}{}",
            block["begin"].as_str().unwrap(),
            call["begin"].as_str().unwrap(),
            call["end"].as_str().unwrap(),
            block["end"].as_str().unwrap(),
        );
        let expected = vec![UnifiedEvent::ToolCall {
            name: "get_weather".into(),
            arguments: json!({"city": "東京🙂"}),
        }];
        for split in input.char_indices().map(|(at, _)| at).chain([input.len()]) {
            let mut parser = create_unified_parser_for_family("deepseek_v41", &tools).unwrap();
            let mut events = parser.push(&input[..split]).unwrap();
            events.extend(parser.push(&input[split..]).unwrap());
            events.extend(parser.finish().unwrap().events);
            assert_eq!(assemble(&events), expected, "split={split}");
        }
    }

    #[test]
    fn v41_none_bans_spaced_tool_markers() {
        let actual = build(
            &DEEPSEEK_V41,
            StructuralTagToolChoice::None,
            &tools(),
            None,
            StructuralTagSchemaMode::Auto,
            false,
        );
        assert_eq!(
            actual["format"]["content"]["excludes"],
            json!(["<｜DSML｜ calls>", "<｜DSML｜ invoke name="])
        );
    }

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
