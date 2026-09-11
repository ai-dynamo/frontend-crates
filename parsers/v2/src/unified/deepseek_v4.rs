// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DeepSeek V4 UnifiedParser wiring.

use crate::tool_calling::dsml::deepseek_v4_scanner;
use crate::tool_calling::scan::ReasoningSpec;
use crate::tool_calling::traits::Tool;
use crate::unified::{GuidedRouted, ScannerUnified, UnifiedParser};

/// Build DeepSeek V4's single DSML and reasoning parser.
pub(crate) fn deepseek_v4_unified(tools: &[Tool]) -> Box<dyn UnifiedParser> {
    Box::new(GuidedRouted::new(ScannerUnified::new(
        deepseek_v4_scanner(tools).with_reasoning(ReasoningSpec {
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
        UnifiedToolOutputMode, assemble,
    };

    fn at_every_split(input: &str, init: UnifiedParserInit) -> Vec<Vec<UnifiedEvent>> {
        (0..=input.len())
            .filter(|&at| input.is_char_boundary(at))
            .map(|at| {
                let mut parser = deepseek_v4_unified(&[]);
                parser.initialize_request(init.clone()).expect("initialize");
                let mut events = parser.push(&input[..at]).expect("push prefix");
                events.extend(parser.push(&input[at..]).expect("push suffix"));
                events.extend(parser.finish().expect("finish").events);
                assemble(&events)
            })
            .collect()
    }

    fn parse_chars(input: &str) -> Vec<UnifiedEvent> {
        let mut parser = deepseek_v4_unified(&[]);
        parser
            .initialize_request(UnifiedParserInit::default())
            .expect("initialize");
        let mut events = Vec::new();
        for ch in input.chars() {
            events.extend(parser.push(&ch.to_string()).expect("push character"));
        }
        events.extend(parser.finish().expect("finish").events);
        assemble(&events)
    }

    fn guided_header_work(name_len: usize, terminated: bool) -> usize {
        crate::tool_calling::dsml::reset_boundary_examined_bytes();
        let suffix = if terminated { ">" } else { "" };
        let input = format!(
            "{}{}{}",
            crate::tool_calling::dsml::INVOKE_START_PREFIX,
            "x".repeat(name_len),
            suffix
        );
        let mut parser = deepseek_v4_unified(&[]);
        parser
            .initialize_request(UnifiedParserInit {
                tool_output_mode: UnifiedToolOutputMode::GuidedJson { named_tool: None },
                invalid_guided_payload: InvalidGuidedPayloadPolicy::RecoverAsText,
                ..UnifiedParserInit::default()
            })
            .expect("initialize");
        for ch in input.chars() {
            parser.push(&ch.to_string()).expect("push character");
        }
        parser.finish().expect("finish");
        crate::tool_calling::dsml::boundary_examined_bytes()
    }

    #[test]
    fn guided_incomplete_and_malformed_header_work_scales_linearly() {
        for terminated in [false, true] {
            let small = guided_header_work(4_096, terminated);
            let large = guided_header_work(8_192, terminated);
            println!("DSv4 guided header terminated={terminated}: {small} -> {large}");
            assert!(
                large <= small * 2 + 256,
                "guided header scan work grew faster than linearly: {small} -> {large}"
            );
        }
    }

    fn assert_native_every_split(input: &str, want: &[UnifiedEvent]) {
        for (at, got) in at_every_split(input, UnifiedParserInit::default())
            .into_iter()
            .enumerate()
        {
            assert_eq!(got, want, "split at byte {at}");
        }
        assert_eq!(parse_chars(input), want, "one-character chunks");
    }

    #[test]
    fn native_json_body_without_invoke_close_is_dropped_across_every_split() {
        let input =
            "<｜DSML｜tool_calls><｜DSML｜invoke name=\"get_weather\">{\"city\": \"Paris\"}";
        let want = Vec::new();
        assert_native_every_split(input, &want);
    }

    #[test]
    fn missing_invoke_close_before_block_close_drops_call_and_preserves_trailing_text() {
        let input = "<｜DSML｜tool_calls><｜DSML｜invoke name=\"get_weather\"><｜DSML｜parameter name=\"city\" string=\"true\">Paris</｜DSML｜parameter></｜DSML｜tool_calls>Still visible.";
        let want = vec![UnifiedEvent::Text {
            text: "Still visible.".into(),
        }];
        assert_native_every_split(input, &want);
    }

    #[test]
    fn malformed_invoke_cannot_borrow_closers_from_trailing_text() {
        let input = "<｜DSML｜tool_calls><｜DSML｜invoke name=\"get_weather\"><｜DSML｜parameter name=\"city\" string=\"true\">Paris</｜DSML｜parameter></｜DSML｜tool_calls>Visible </｜DSML｜parameter></｜DSML｜invoke> text.";
        let want = vec![UnifiedEvent::Text {
            text: "Visible </｜DSML｜parameter> text.".into(),
        }];
        assert_native_every_split(input, &want);
    }

    #[test]
    fn unclosed_parameter_before_block_close_drops_call_and_preserves_trailing_text() {
        let want = vec![UnifiedEvent::Text {
            text: "Still visible.".into(),
        }];
        let input = "<｜DSML｜tool_calls><｜DSML｜invoke name=\"get_weather\"><｜DSML｜parameter name=\"city\" string=\"true\">Par</｜DSML｜tool_calls>Still visible.";
        assert_native_every_split(input, &want);
    }

    #[test]
    fn missing_invoke_close_at_eof_drops_complete_and_partial_parameters() {
        for input in [
            "<｜DSML｜tool_calls><｜DSML｜invoke name=\"get_weather\"><｜DSML｜parameter name=\"city\" string=\"true\">Paris</｜DSML｜parameter>",
            "<｜DSML｜tool_calls><｜DSML｜invoke name=\"get_weather\"><｜DSML｜parameter name=\"city\" string=\"true\">Par",
        ] {
            assert_native_every_split(input, &[]);
        }
    }

    #[test]
    fn missing_outer_close_after_valid_invoke_still_dispatches_at_every_split() {
        let input = "<｜DSML｜tool_calls><｜DSML｜invoke name=\"get_weather\"><｜DSML｜parameter name=\"city\" string=\"true\">Paris</｜DSML｜parameter></｜DSML｜invoke>";
        let want = vec![UnifiedEvent::ToolCall {
            name: "get_weather".into(),
            arguments: serde_json::json!({"city": "Paris"}),
        }];
        assert_native_every_split(input, &want);
    }

    #[test]
    fn invoke_and_block_markers_inside_parameter_values_are_data() {
        let input = "<｜DSML｜tool_calls><｜DSML｜invoke name=\"run\"><｜DSML｜parameter name=\"command\" string=\"true\">echo </｜DSML｜invoke> and </｜DSML｜tool_calls></｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>";
        let want = vec![UnifiedEvent::ToolCall {
            name: "run".into(),
            arguments: serde_json::json!({
                "command": "echo </｜DSML｜invoke> and </｜DSML｜tool_calls>"
            }),
        }];
        assert_native_every_split(input, &want);
    }

    #[test]
    fn reset_after_partial_invoke_restores_a_fresh_stream() {
        let partial = "<｜DSML｜tool_calls><｜DSML｜invoke name=\"abandoned\"><｜DSML｜parameter name=\"x\" string=\"true\">partial";
        let mut parser = deepseek_v4_unified(&[]);
        parser
            .initialize_request(UnifiedParserInit::default())
            .expect("initialize");
        assert!(parser.push(partial).expect("push partial").is_empty());
        assert_eq!(parser.reset(), partial);

        let valid = "<｜DSML｜tool_calls><｜DSML｜invoke name=\"fresh\"><｜DSML｜parameter name=\"x\" string=\"true\">complete</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>";
        let mut events = parser.push(valid).expect("push after reset");
        events.extend(parser.finish().expect("finish after reset").events);
        assert_eq!(
            assemble(&events),
            vec![UnifiedEvent::ToolCall {
                name: "fresh".into(),
                arguments: serde_json::json!({"x": "complete"}),
            }]
        );
    }

    #[test]
    fn narrated_incomplete_dsml_header_stays_split_invariant_in_guided_reasoning() {
        let input = "<think>I'll call <｜DSML｜invoke name=\"get_weather</think>[{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}]";
        let want = vec![
            UnifiedEvent::Reasoning {
                text: "I'll call get_weather".into(),
            },
            UnifiedEvent::ToolCall {
                name: "get_weather".into(),
                arguments: serde_json::json!({"city": "Paris"}),
            },
        ];
        let init = UnifiedParserInit {
            tool_output_mode: UnifiedToolOutputMode::GuidedJson { named_tool: None },
            invalid_guided_payload: InvalidGuidedPayloadPolicy::RecoverAsText,
            ..UnifiedParserInit::default()
        };
        let mut parser = deepseek_v4_unified(&[]);
        parser.initialize_request(init.clone()).expect("initialize");
        let mut whole = parser.push(input).expect("push");
        whole.extend(parser.finish().expect("finish").events);
        assert_eq!(assemble(&whole), want);
        for (at, got) in at_every_split(input, init).into_iter().enumerate() {
            assert_eq!(got, want, "split at byte {at}");
        }
    }
}
