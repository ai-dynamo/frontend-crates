// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unified GLM-4.7/5.x parser wiring.

use crate::tool_calling::glm47::glm47_scanner;
use crate::tool_calling::scan::ReasoningSpec;
use crate::tool_calling::traits::Tool;
use crate::unified::{GuidedRouted, ScannerUnified, UnifiedParser};

/// Keep native and guided GLM parsing on the scanner shared with the legacy projection.
pub(crate) fn glm47_unified(tools: &[Tool]) -> Box<dyn UnifiedParser> {
    Box::new(GuidedRouted::new(ScannerUnified::new(
        glm47_scanner(tools).with_reasoning(ReasoningSpec {
            start: "<think>",
            end: "</think>",
            forced_start: false,
            preserve_special_tokens: false,
            ..Default::default()
        }),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unified::{
        InvalidGuidedPayloadPolicy, UnifiedEvent, UnifiedParserExt, UnifiedParserInit,
        UnifiedParserStartingState, UnifiedToolOutputMode, assemble,
        create_unified_parser_for_family,
    };

    fn tools() -> Vec<Tool> {
        vec![
            Tool {
                name: "get_weather".into(),
                description: None,
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": { "city": { "type": "string" } }
                }),
                strict: None,
            },
            Tool {
                name: "get_time".into(),
                description: None,
                parameters: serde_json::json!({"type": "object", "properties": {}}),
                strict: None,
            },
            Tool {
                name: "run".into(),
                description: None,
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": { "cmd": { "type": "string" } }
                }),
                strict: None,
            },
        ]
    }

    fn parse(tools: &[Tool], input: &str, split: Option<usize>) -> Vec<UnifiedEvent> {
        let mut parser = create_unified_parser_for_family("glm47", tools).expect("registry");
        let mut deltas = Vec::new();
        match split {
            Some(at) => {
                deltas.extend(parser.push(&input[..at]).expect("prefix"));
                deltas.extend(parser.push(&input[at..]).expect("suffix"));
            }
            None => deltas.extend(parser.push(input).expect("whole input")),
        }
        deltas.extend(parser.finish().expect("finish").events);
        assemble(&deltas)
    }

    fn parse_guided(input: &str, split: Option<usize>) -> Vec<UnifiedEvent> {
        let mut parser = create_unified_parser_for_family("glm47", &tools()).expect("registry");
        parser
            .initialize_request(UnifiedParserInit {
                tool_output_mode: UnifiedToolOutputMode::GuidedJson { named_tool: None },
                invalid_guided_payload: InvalidGuidedPayloadPolicy::RecoverAsText,
                ..UnifiedParserInit::default()
            })
            .expect("initialize guided parser");
        let mut deltas = Vec::new();
        match split {
            Some(at) => {
                deltas.extend(parser.push(&input[..at]).expect("prefix"));
                deltas.extend(parser.push(&input[at..]).expect("suffix"));
            }
            None => deltas.extend(parser.push(input).expect("whole input")),
        }
        deltas.extend(parser.finish().expect("finish").events);
        assemble(&deltas)
    }

    fn call() -> UnifiedEvent {
        UnifiedEvent::ToolCall {
            name: "get_weather".into(),
            arguments: serde_json::json!({"city": "Paris"}),
        }
    }

    #[test]
    fn native_whole_input_preserves_reasoning_call_reasoning_order() {
        let input = "<think>look</think><tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call><think>answer</think>Done";
        assert_eq!(
            parse(&tools(), input, None),
            vec![
                UnifiedEvent::Reasoning {
                    text: "look".into()
                },
                call(),
                UnifiedEvent::Reasoning {
                    text: "answer".into()
                },
                UnifiedEvent::Text {
                    text: "Done".into()
                },
            ]
        );
    }

    #[test]
    fn native_output_is_identical_at_every_valid_split() {
        let input = "before <tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call> after";
        let want = parse(&tools(), input, None);
        for split in input.char_indices().map(|(at, _)| at).chain([input.len()]) {
            assert_eq!(
                parse(&tools(), input, Some(split)),
                want,
                "split at {split}"
            );
        }
    }

    #[test]
    fn registered_parser_recovers_known_no_argument_bare_call_at_every_valid_split() {
        let input = "get_time</tool_call>";
        let want = vec![UnifiedEvent::ToolCall {
            name: "get_time".into(),
            arguments: serde_json::json!({}),
        }];
        assert_eq!(parse(&tools(), input, None), want);
        for split in input.char_indices().map(|(at, _)| at).chain([input.len()]) {
            assert_eq!(
                parse(&tools(), input, Some(split)),
                want,
                "split at {split}"
            );
        }
    }

    #[test]
    fn registered_parser_recovers_bare_call_with_arguments_at_every_valid_split() {
        let input = "run<arg_key>cmd</arg_key><arg_value>git status</arg_value></tool_call>";
        let want = vec![UnifiedEvent::ToolCall {
            name: "run".into(),
            arguments: serde_json::json!({"cmd": "git status"}),
        }];
        assert_eq!(parse(&tools(), input, None), want);
        for split in input.char_indices().map(|(at, _)| at).chain([input.len()]) {
            assert_eq!(
                parse(&tools(), input, Some(split)),
                want,
                "split at {split}"
            );
        }
    }

    #[test]
    fn registered_parser_does_not_recover_prose_before_orphan_close_at_any_split() {
        for (input, want) in [
            (
                "get_time.</tool_call>",
                vec![UnifiedEvent::Text {
                    text: "get_time.".into(),
                }],
            ),
            (
                "Please wait café</tool_call>",
                vec![UnifiedEvent::Text {
                    text: "Please wait café".into(),
                }],
            ),
            (
                "unknown_tool</tool_call>",
                vec![UnifiedEvent::ToolCall {
                    name: "unknown_tool".into(),
                    arguments: serde_json::json!({}),
                }],
            ),
            (
                "Please wait</tool_call><tool_call>get_time</tool_call>",
                vec![
                    UnifiedEvent::Text {
                        text: "Please wait".into(),
                    },
                    UnifiedEvent::ToolCall {
                        name: "get_time".into(),
                        arguments: serde_json::json!({}),
                    },
                ],
            ),
        ] {
            assert_eq!(parse(&tools(), input, None), want);
            for split in input.char_indices().map(|(at, _)| at).chain([input.len()]) {
                assert_eq!(
                    parse(&tools(), input, Some(split)),
                    want,
                    "split at {split} for {input:?}"
                );
            }
        }
    }

    #[test]
    fn registered_parser_preserves_embedded_close_and_following_call_at_every_split() {
        let input = "<tool_call>run<arg_key>cmd</arg_key><arg_value>git log </tool_call> --oneline</arg_value></tool_call> café <tool_call>get_time</tool_call>";
        let want = vec![
            UnifiedEvent::ToolCall {
                name: "run".into(),
                arguments: serde_json::json!({"cmd": "git log </tool_call> --oneline"}),
            },
            UnifiedEvent::Text {
                text: " café ".into(),
            },
            UnifiedEvent::ToolCall {
                name: "get_time".into(),
                arguments: serde_json::json!({}),
            },
        ];
        assert_eq!(parse(&tools(), input, None), want);
        for split in input.char_indices().map(|(at, _)| at).chain([input.len()]) {
            assert_eq!(
                parse(&tools(), input, Some(split)),
                want,
                "split at {split}"
            );
        }
    }

    #[test]
    fn guided_native_markup_only_emits_nothing_at_every_valid_split() {
        let input =
            "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>";
        assert!(parse_guided(input, None).is_empty());
        for split in input.char_indices().map(|(at, _)| at).chain([input.len()]) {
            assert!(
                parse_guided(input, Some(split)).is_empty(),
                "split at {split}"
            );
        }
    }

    #[test]
    fn malformed_eof_recovers_at_the_only_possible_outer_boundary() {
        let input = "<tool_call>run<arg_key>cmd</arg_key><arg_value>git log </tool_call> --oneline";
        let want = vec![
            UnifiedEvent::ToolCall {
                name: "run".into(),
                arguments: serde_json::json!({"cmd": "git log "}),
            },
            UnifiedEvent::Text {
                text: " --oneline".into(),
            },
        ];
        assert_eq!(parse(&tools(), input, None), want);
        for split in input.char_indices().map(|(at, _)| at).chain([input.len()]) {
            assert_eq!(
                parse(&tools(), input, Some(split)),
                want,
                "split at {split}"
            );
        }
    }

    #[test]
    fn paired_tool_markers_inside_an_argument_remain_data_at_every_split() {
        let input = "<tool_call>run<arg_key>cmd</arg_key><arg_value>before </tool_call><tool_call> after</arg_value></tool_call><tool_call>get_time</tool_call>";
        let want = vec![
            UnifiedEvent::ToolCall {
                name: "run".into(),
                arguments: serde_json::json!({
                    "cmd": "before </tool_call><tool_call> after"
                }),
            },
            UnifiedEvent::ToolCall {
                name: "get_time".into(),
                arguments: serde_json::json!({}),
            },
        ];
        assert_eq!(parse(&tools(), input, None), want);
        for split in input.char_indices().map(|(at, _)| at).chain([input.len()]) {
            assert_eq!(
                parse(&tools(), input, Some(split)),
                want,
                "split at {split}"
            );
        }
    }

    #[test]
    fn unclosed_argument_recovers_before_a_following_call_at_every_split() {
        let input = "<tool_call>run<arg_key>cmd</arg_key><arg_value>first</tool_call><tool_call>get_time</tool_call>";
        let want = vec![
            UnifiedEvent::ToolCall {
                name: "run".into(),
                arguments: serde_json::json!({"cmd": "first"}),
            },
            UnifiedEvent::ToolCall {
                name: "get_time".into(),
                arguments: serde_json::json!({}),
            },
        ];
        assert_eq!(parse(&tools(), input, None), want);
        for split in input.char_indices().map(|(at, _)| at).chain([input.len()]) {
            assert_eq!(
                parse(&tools(), input, Some(split)),
                want,
                "split at {split}"
            );
        }
    }

    #[test]
    fn terminal_outer_close_recovers_a_missing_argument_value_close_at_every_split() {
        let input = "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</tool_call>";
        let want = vec![UnifiedEvent::ToolCall {
            name: "get_weather".into(),
            arguments: serde_json::json!({"city": "Paris"}),
        }];
        assert_eq!(parse(&tools(), input, None), want);
        for split in input.char_indices().map(|(at, _)| at).chain([input.len()]) {
            assert_eq!(
                parse(&tools(), input, Some(split)),
                want,
                "split at {split}"
            );
        }
    }

    #[test]
    fn malformed_block_does_not_cost_a_following_call_at_every_split() {
        let input = "<tool_call></tool_call><tool_call>get_time</tool_call>";
        let want = vec![UnifiedEvent::ToolCall {
            name: "get_time".into(),
            arguments: serde_json::json!({}),
        }];
        assert_eq!(parse(&tools(), input, None), want);
        for split in input.char_indices().map(|(at, _)| at).chain([input.len()]) {
            assert_eq!(
                parse(&tools(), input, Some(split)),
                want,
                "split at {split}"
            );
        }
    }

    #[test]
    fn tool_call_inside_reasoning_splits_the_reasoning_channel() {
        let input = "<think>before <tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call> after</think>done";
        assert_eq!(
            parse(&tools(), input, None),
            vec![
                UnifiedEvent::Reasoning {
                    text: "before ".into()
                },
                call(),
                UnifiedEvent::Reasoning {
                    text: " after".into()
                },
                UnifiedEvent::Text {
                    text: "done".into()
                },
            ]
        );
    }

    #[test]
    fn malformed_and_eof_tool_calls_do_not_leak_markup() {
        assert_eq!(
            parse(
                &tools(),
                "visible <tool_call>get_weather<arg_key>city",
                None
            ),
            vec![UnifiedEvent::Text {
                text: "visible ".into()
            }]
        );
        assert_eq!(
            parse(&tools(), "visible! </tool_call> text", None),
            vec![UnifiedEvent::Text {
                text: "visible!  text".into()
            }]
        );
    }

    #[test]
    fn registry_constructs_glm47_with_reasoning_start_state() {
        let mut parser = create_unified_parser_for_family("glm47", &tools()).expect("registry");
        parser
            .initialize_request(crate::unified::UnifiedParserInit {
                starting_state: UnifiedParserStartingState::None,
                ..Default::default()
            })
            .expect("initialize");
        assert!(parser.preserve_special_tokens());
    }
}
