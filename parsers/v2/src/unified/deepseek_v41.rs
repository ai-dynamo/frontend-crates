// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Context;
use serde_json::{Map, Value};

use crate::tool_calling::scan::{
    BareRecoveryLatch, InvokeBoundaryFactory, InvokeEmitter, InvokeScan, ReasoningSpec,
    WrappedBlockScanner, WrappedBlockSpec,
};
use crate::tool_calling::traits::{Tool, ToolCallDelta};
use crate::unified::{GuidedRouted, ScannerUnified, UnifiedParser};

const BLOCK_START: &str = "<｜DSML｜ calls>";
const BLOCK_END: &str = "</｜DSML｜ calls>";
const INVOKE_START: &str = "<｜DSML｜ invoke name=\"";
const INVOKE_END: &str = "</｜DSML｜ invoke>";
const PARAMETER_START: &str = "<｜DSML｜ parameter name=\"";
const PARAMETER_END: &str = "</｜DSML｜ parameter>";

pub(crate) fn deepseek_v41_unified(_tools: &[Tool]) -> Box<dyn UnifiedParser> {
    let spec = WrappedBlockSpec {
        family: "deepseek_v41",
        block_starts: vec![BLOCK_START.into()],
        block_ends: vec![BLOCK_END.into()],
        invoke_start: INVOKE_START.into(),
        invoke_end: INVOKE_END.into(),
        orphan_markers: vec![BLOCK_END.into()],
        holdback_markers: vec![BLOCK_START.into(), BLOCK_END.into(), INVOKE_START.into()],
        bare_recovery_latch: BareRecoveryLatch::Set,
        invoke_boundary_factory: Some(InvokeBoundaryFactory::stateless(InvokeScan {
            end: invocation_end,
            opens: |_, _| true,
            holdback: |_| 0,
            resync: None,
        })),
        preserve_special_tokens: true,
        ..Default::default()
    };
    let scanner = WrappedBlockScanner::new(spec, DeepSeekV41).with_reasoning(ReasoningSpec {
        start: "<think>",
        end: "</think>",
        preserve_special_tokens: true,
        ..Default::default()
    });
    Box::new(GuidedRouted::new(ScannerUnified::new(scanner)))
}

fn parameter_header(text: &str) -> Option<(&str, bool, &str)> {
    let (name, rest) = text.strip_prefix(PARAMETER_START)?.split_once('"')?;
    let (string, value) = rest.strip_prefix(" string=\"")?.split_once("\">")?;
    let string = match string {
        "true" => true,
        "false" => false,
        _ => return None,
    };
    Some((name, string, value))
}

fn invocation_end(text: &str, _flush: bool, _tool_index: usize) -> Option<usize> {
    let mut cursor = 0;
    loop {
        let remaining = &text[cursor..];
        let close = remaining.find(INVOKE_END)?;
        let parameter = remaining.find(PARAMETER_START);
        if parameter.is_none_or(|parameter| close < parameter) {
            return Some(cursor + close + INVOKE_END.len());
        }
        let parameter = &remaining[parameter?..];
        let (_, _, value) = parameter_header(parameter)?;
        let value_end = value.find(PARAMETER_END)?;
        cursor = text.len() - value.len() + value_end + PARAMETER_END.len();
    }
}

struct DeepSeekV41;

impl InvokeEmitter for DeepSeekV41 {
    fn parse_invoke(
        &mut self,
        invoke: &str,
        tool_index: usize,
    ) -> anyhow::Result<Option<ToolCallDelta>> {
        let (name, body) = invoke
            .strip_prefix(INVOKE_START)
            .and_then(|text| text.split_once("\">"))
            .context("invalid DeepSeek V4.1 invocation header")?;
        anyhow::ensure!(!name.is_empty(), "empty DeepSeek V4.1 tool name");
        let mut body = body
            .strip_suffix(INVOKE_END)
            .context("incomplete DeepSeek V4.1 invocation")?;
        let mut arguments = Map::new();
        while !body.trim().is_empty() {
            let (name, string, value) = parameter_header(body.trim_start())
                .context("invalid DeepSeek V4.1 parameter header")?;
            let (raw, remainder) = value
                .split_once(PARAMETER_END)
                .context("incomplete DeepSeek V4.1 parameter")?;
            let value = if string {
                Value::String(raw.to_string())
            } else {
                serde_json::from_str(raw)?
            };
            anyhow::ensure!(
                arguments.insert(name.to_string(), value).is_none(),
                "duplicate DeepSeek V4.1 parameter {name:?}"
            );
            body = remainder;
        }
        Ok(Some(ToolCallDelta {
            tool_index,
            name: Some(name.to_string()),
            arguments: serde_json::to_string(&arguments)?,
            complete: true,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unified::{
        UnifiedEvent, UnifiedParserExt, UnifiedParserInit, UnifiedParserOutput,
        UnifiedParserStartingState, UnifiedToolOutputMode, create_unified_parser_for_family,
    };

    fn parse_chunks(
        input: &str,
        split: usize,
        state: UnifiedParserStartingState,
    ) -> UnifiedParserOutput {
        let mut parser = create_unified_parser_for_family("deepseek_v41", &[]).unwrap();
        assert!(parser.preserve_special_tokens());
        parser
            .initialize_request(UnifiedParserInit {
                starting_state: state,
                ..Default::default()
            })
            .unwrap();
        let mut output = UnifiedParserOutput::default();
        parser.parse_into(&input[..split], &mut output).unwrap();
        parser.parse_into(&input[split..], &mut output).unwrap();
        output.append(&mut parser.finish().unwrap());
        output
    }

    fn assert_every_split(
        input: &str,
        state: UnifiedParserStartingState,
        expected: Vec<UnifiedEvent>,
    ) {
        for split in (0..=input.len()).filter(|&i| input.is_char_boundary(i)) {
            assert_eq!(
                parse_chunks(input, split, state).assembled(),
                expected,
                "split {split}"
            );
        }
        let mut parser = deepseek_v41_unified(&[]);
        parser
            .initialize_request(UnifiedParserInit {
                starting_state: state,
                ..Default::default()
            })
            .unwrap();
        let mut output = UnifiedParserOutput::default();
        for ch in input.chars() {
            parser
                .parse_into(ch.encode_utf8(&mut [0; 4]), &mut output)
                .unwrap();
        }
        output.append(&mut parser.finish().unwrap());
        assert_eq!(output.assembled(), expected, "one character at a time");
    }

    #[test]
    fn deepseek_v41_registration() {
        assert_every_split(
            "hello 世界",
            UnifiedParserStartingState::None,
            vec![UnifiedEvent::Text {
                text: "hello 世界".into(),
            }],
        );
    }

    #[test]
    fn reasoning_transition_and_multiple_calls() {
        let input = concat!(
            "Check the tools.\n</think>\n\n<｜DSML｜ calls>\n",
            "<｜DSML｜ invoke name=\"weather\">\n",
            "<｜DSML｜ parameter name=\"city\" string=\"true\">東京 &amp; \"Paris\"\\\n</｜DSML｜ parameter>\n",
            "<｜DSML｜ parameter name=\"days\" string=\"false\">3</｜DSML｜ parameter>\n",
            "<｜DSML｜ parameter name=\"options\" string=\"false\">{\"x\":[true,null]}</｜DSML｜ parameter>\n",
            "</｜DSML｜ invoke>\n<｜DSML｜ invoke name=\"done\">\n</｜DSML｜ invoke>\n",
            "</｜DSML｜ calls>",
        );
        assert_every_split(
            input,
            UnifiedParserStartingState::Reasoning,
            vec![
                UnifiedEvent::Reasoning {
                    text: "Check the tools.\n".into(),
                },
                UnifiedEvent::Text {
                    text: "\n\n".into(),
                },
                UnifiedEvent::ToolCall {
                    name: "weather".into(),
                    arguments: serde_json::json!({
                        "city": "東京 &amp; \"Paris\"\\\n", "days": 3, "options": {"x": [true, null]}
                    }),
                },
                UnifiedEvent::ToolCall {
                    name: "done".into(),
                    arguments: serde_json::json!({}),
                },
            ],
        );
    }

    #[test]
    fn tool_markup_inside_string_is_data() {
        let input = "<｜DSML｜ calls><｜DSML｜ invoke name=\"run\"><｜DSML｜ parameter name=\"text\" string=\"true\"><think>quoted</think> <｜DSML｜ calls></｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>";
        assert_every_split(
            input,
            UnifiedParserStartingState::None,
            vec![UnifiedEvent::ToolCall {
                name: "run".into(),
                arguments: serde_json::json!({"text": "<think>quoted</think> <｜DSML｜ calls>"}),
            }],
        );
    }

    #[test]
    fn closing_markers_and_whitespace_inside_strings_are_data() {
        let value = " X</｜DSML｜ calls>Y</｜DSML｜ invoke>Z\n ";
        let input = format!(
            "<｜DSML｜ calls><｜DSML｜ invoke name=\"run\"><｜DSML｜ parameter name=\"user name\" string=\"true\">{value}</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>"
        );
        assert_every_split(
            &input,
            UnifiedParserStartingState::None,
            vec![UnifiedEvent::ToolCall {
                name: "run".into(),
                arguments: serde_json::json!({"user name":value}),
            }],
        );
    }

    #[test]
    fn incomplete_arguments_do_not_emit_calls() {
        let input = "<｜DSML｜ calls><｜DSML｜ invoke name=\"run\"><｜DSML｜ parameter name=\"text\" string=\"true\">unfinished";
        assert_every_split(input, UnifiedParserStartingState::None, vec![]);
    }

    #[test]
    fn partial_plain_markers_and_open_reasoning_survive_eof() {
        for input in ["hello <", "hello <｜DS", "ordinary text\n", "你好"] {
            assert_every_split(
                input,
                UnifiedParserStartingState::None,
                vec![UnifiedEvent::Text { text: input.into() }],
            );
            assert_every_split(
                input,
                UnifiedParserStartingState::Reasoning,
                vec![UnifiedEvent::Reasoning { text: input.into() }],
            );
        }
    }

    #[test]
    fn calls_stream_at_each_invocation_close() {
        let mut parser = deepseek_v41_unified(&[]);
        assert!(parser.push("<｜DSML｜ calls><｜DSML｜ invoke name=\"run\"><｜DSML｜ parameter name=\"text\" string=\"true\">hello").unwrap().is_empty());
        let events = parser
            .push("</｜DSML｜ parameter></｜DSML｜ invoke>")
            .unwrap();
        let output: UnifiedParserOutput = events.into_iter().collect();
        assert_eq!(
            output.assembled(),
            vec![UnifiedEvent::ToolCall {
                name: "run".into(),
                arguments: serde_json::json!({"text":"hello"}),
            }]
        );
        assert!(parser.push("</｜DSML｜ calls>").unwrap().is_empty());
    }

    #[test]
    fn invocation_requires_its_complete_closing_tag() {
        for suffix in ["", " ", " inv", " invoke", " banana>"] {
            let input = format!("<｜DSML｜ calls><｜DSML｜ invoke name=\"run\"></｜DSML｜{suffix}");
            let mut parser = deepseek_v41_unified(&[]);
            assert!(parser.push(&input).unwrap().is_empty());
            assert!(parser.finish().unwrap().events.is_empty());
        }
    }

    #[test]
    fn malformed_invocation_is_an_error_without_tool_deltas() {
        for input in [
            "<｜DSML｜ calls><｜DSML｜ invoke name=\"run\"><｜DSML｜ parameter name=\"x\" string=\"true\">first</｜DSML｜ parameter><｜DSML｜ parameter name=\"x\" string=\"true\">second</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>",
            "<｜DSML｜ calls><｜DSML｜ invoke name=\"run\"><｜DSML｜ parameter name=\"value\" string=\"false\">invalid</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>",
        ] {
            for split in (0..=input.len()).filter(|&i| input.is_char_boundary(i)) {
                let mut parser = deepseek_v41_unified(&[]);
                let mut output = UnifiedParserOutput::default();
                let result = parser
                    .parse_into(&input[..split], &mut output)
                    .and_then(|()| parser.parse_into(&input[split..], &mut output));
                assert!(result.is_err(), "split {split}");
                assert!(output.events.is_empty(), "split {split}");
            }
        }
    }

    #[test]
    fn guided_output_uses_shared_decoder() {
        let mut parser = deepseek_v41_unified(&[]);
        parser
            .initialize_request(UnifiedParserInit {
                starting_state: UnifiedParserStartingState::Reasoning,
                tool_output_mode: UnifiedToolOutputMode::GuidedJson {
                    named_tool: Some("weather".into()),
                },
                ..Default::default()
            })
            .unwrap();
        assert_eq!(
            parser
                .parse_complete("check</think>{\"city\":\"Paris\"}")
                .unwrap(),
            vec![
                UnifiedEvent::Reasoning {
                    text: "check".into()
                },
                UnifiedEvent::ToolCall {
                    name: "weather".into(),
                    arguments: serde_json::json!({"city":"Paris"})
                },
            ]
        );
    }

    #[test]
    fn reset_restarts_tool_indices() {
        let input =
            "<｜DSML｜ calls><｜DSML｜ invoke name=\"done\"></｜DSML｜ invoke></｜DSML｜ calls>";
        let mut parser = deepseek_v41_unified(&[]);
        let first = parser.push(input).unwrap();
        parser.finish().unwrap();
        assert!(parser.reset().is_empty());
        assert_eq!(parser.push(input).unwrap(), first);
    }
}
