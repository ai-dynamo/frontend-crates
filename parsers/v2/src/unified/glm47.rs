// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unified GLM-4.7/5.x parser wiring.

use crate::tool_calling::glm47::glm47_scanner;
use crate::tool_calling::scan::ReasoningSpec;
use crate::tool_calling::traits::Tool;
use crate::unified::{GuidedRouted, ScannerUnified, UnifiedParser};

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
    fn guided_native_envelope_after_visible_prose_is_split_invariant() {
        let input = "hello <tool_call>run<arg_key>cmd</arg_key><arg_value>{\"name\":\"get_time\",\"arguments\":{}}</arg_value></tool_call>";
        let want = vec![UnifiedEvent::Text {
            text: "hello ".into(),
        }];
        assert_eq!(parse_guided(input, None), want);
        for split in input.char_indices().map(|(at, _)| at).chain([input.len()]) {
            assert_eq!(parse_guided(input, Some(split)), want, "split at {split}");
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
