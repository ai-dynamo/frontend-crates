// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unified Muse Spark parser.
//!
//! Muse Spark inherits Muse Glimmer's recipient-routed ATEM grammar and adds the
//! reserved, user-visible `commentary` recipient. The shared scanner owns the full
//! ordered stream; this factory selects the Muse Spark recipient policy.

use crate::tool_calling::muse_glimmer::muse_spark_scanner;
use crate::tool_calling::traits::Tool;
use crate::unified::{GuidedRouted, UnifiedParser};

/// Build the Muse Spark unified parser for one stream.
pub(crate) fn muse_spark_unified(tools: &[Tool]) -> Box<dyn UnifiedParser> {
    Box::new(GuidedRouted::new(muse_spark_scanner(tools)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unified::{
        InvalidGuidedPayloadPolicy, UnifiedEvent, UnifiedParserEvent, UnifiedParserExt,
        UnifiedParserInit, UnifiedParserStartingState, UnifiedToolOutputMode, assemble,
    };

    fn tools() -> Vec<Tool> {
        ["get_weather", "commentary"]
            .iter()
            .map(|name| Tool {
                name: (*name).to_string(),
                description: None,
                parameters: serde_json::json!({"type": "object", "properties": {}}),
                strict: None,
            })
            .collect()
    }

    fn parse_chunks(chunks: &[String]) -> Vec<UnifiedEvent> {
        let mut parser = muse_spark_unified(&tools());
        let mut deltas = Vec::new();
        for chunk in chunks {
            deltas.extend(parser.push(chunk).expect("push"));
        }
        deltas.extend(parser.finish().expect("finish"));
        assemble(&deltas)
    }

    fn parse(input: &str) -> Vec<UnifiedEvent> {
        muse_spark_unified(&tools())
            .parse_complete(input)
            .expect("parse complete")
    }

    fn parse_guided_chunks(chunks: &[String]) -> Vec<UnifiedEvent> {
        let mut parser = muse_spark_unified(&tools());
        parser
            .initialize_request(UnifiedParserInit {
                prompt_token_ids: Vec::new(),
                starting_state: UnifiedParserStartingState::Reasoning,
                tool_output_mode: UnifiedToolOutputMode::GuidedJson { named_tool: None },
                invalid_guided_payload: InvalidGuidedPayloadPolicy::RecoverAsText,
            })
            .expect("guided init");

        let mut deltas = Vec::new();
        for chunk in chunks {
            deltas.extend(parser.push(chunk).expect("guided push"));
        }
        deltas.extend(parser.finish().expect("guided finish"));
        assemble(&deltas)
    }

    fn assert_every_chunking(input: &str, expected: &[UnifiedEvent]) {
        assert_eq!(parse(input), expected, "whole input");

        let per_character: Vec<String> = input.chars().map(|ch| ch.to_string()).collect();
        assert_eq!(
            parse_chunks(&per_character),
            expected,
            "one character at a time"
        );

        for split in 1..input.len() {
            if !input.is_char_boundary(split) {
                continue;
            }
            assert_eq!(
                parse_chunks(&[input[..split].to_string(), input[split..].to_string()]),
                expected,
                "split at byte {split}"
            );
        }
    }

    fn text(value: &str) -> UnifiedEvent {
        UnifiedEvent::Text {
            text: value.to_string(),
        }
    }

    fn reasoning(value: &str) -> UnifiedEvent {
        UnifiedEvent::Reasoning {
            text: value.to_string(),
        }
    }

    fn call(name: &str, arguments: serde_json::Value) -> UnifiedEvent {
        UnifiedEvent::ToolCall {
            name: name.to_string(),
            arguments,
        }
    }

    #[test]
    fn responses_phases_are_chunk_invariant_in_native_and_guided_output() {
        let input = concat!(
            "<|start|>assistant to=commentary<|message|>Checking.<|eom|>",
            "<|start|>assistant to=commentary<|message|>Still checking.<|eom|>",
            "<|start|>assistant to=user<|message|>Done.<|eot|>"
        );
        for guided in [false, true] {
            let tool = if guided {
                r#"[{"name":"get_weather","arguments":{"city":"Paris"}}]"#
            } else {
                r#"<|start|>assistant to=get_weather<|message|><atem:function_calls><atem:invoke name="get_weather"><atem:parameter name="city">Paris</atem:parameter></atem:invoke></atem:function_calls><|eom|>"#
            };
            let input = input.replace(
                "<|start|>assistant to=user",
                &format!("{tool}<|start|>assistant to=user"),
            );
            let input = input.as_str();
            let run = |chunks: Vec<&str>| {
                let mut parser = muse_spark_unified(&tools());
                if guided {
                    parser
                        .initialize_request(UnifiedParserInit {
                            tool_output_mode: UnifiedToolOutputMode::GuidedJson {
                                named_tool: None,
                            },
                            ..Default::default()
                        })
                        .unwrap();
                }
                parser.preserve_message_phases(true);
                let mut deltas = Vec::new();
                for chunk in chunks {
                    deltas.extend(parser.push(chunk).unwrap());
                }
                deltas.extend(parser.finish().unwrap());
                assemble(&deltas)
            };
            let expected = run(vec![input]);
            let value = serde_json::to_value(&expected).unwrap();
            assert_eq!(value[0]["phase"], "commentary", "guided={guided}: {value}");
            assert_eq!(value[3]["phase"], "commentary", "guided={guided}: {value}");
            assert_eq!(
                value[7]["phase"], "final_answer",
                "guided={guided}: {value}"
            );
            for split in 1..input.len() {
                assert_eq!(
                    run(vec![&input[..split], &input[split..]]),
                    expected,
                    "guided={guided} split={split}"
                );
            }
            let chars: Vec<_> = input
                .char_indices()
                .map(|(i, c)| &input[i..i + c.len_utf8()])
                .collect();
            assert_eq!(run(chars), expected, "guided={guided} character chunks");
        }
    }

    #[test]
    fn phase_markers_inside_arguments_are_data_and_eof_does_not_claim_completion() {
        let mut parser = muse_spark_unified(&tools());
        parser.preserve_message_phases(true);
        let input = concat!(
            "<|start|>assistant to=get_weather<|message|><atem:function_calls>",
            "<atem:invoke name=\"get_weather\"><atem:parameter name=\"note\">to=commentary&lt;|message|&gt;literal</atem:parameter></atem:invoke></atem:function_calls><|eom|>",
            "<|start|>assistant to=user<|message|>Unfinished"
        );
        let output = parser.parse_complete(input).unwrap();
        assert!(matches!(output[0], UnifiedEvent::ToolCall { .. }));
        assert!(matches!(
            output[1],
            UnifiedEvent::MessageStart {
                phase: crate::unified::MessagePhase::FinalAnswer
            }
        ));
        assert!(matches!(output.last(),Some(UnifiedEvent::Text {text}) if text=="Unfinished"));
    }

    #[test]
    fn responses_phases_keep_adjacent_commentary_and_final_separate() {
        let mut parser = muse_spark_unified(&tools());
        parser.preserve_message_phases(true);
        let events = parser
            .parse_complete(concat!(
                "<|start|>assistant to=commentary<|message|>Checking.<|eom|>",
                "<|start|>assistant to=user<|message|>Done.<|eot|>"
            ))
            .unwrap();
        let value = serde_json::to_value(events).unwrap();
        assert_eq!(value[0]["kind"], "message_start");
        assert_eq!(value[0]["phase"], "commentary");
        assert_eq!(value[2]["kind"], "message_end");
        assert_eq!(value[3]["phase"], "final_answer");
    }

    #[test]
    fn commentary_is_visible_even_when_a_tool_has_that_name() {
        let input = "<|start|>assistant to=commentary<|message|>Checking first.<|eom|>";
        assert_eq!(parse(input), vec![text("Checking first.")]);

        let mut muse = crate::unified::muse_glimmer::muse_glimmer_unified(&tools());
        assert!(
            muse.parse_complete(input).expect("Muse parse").is_empty(),
            "Muse Glimmer must retain its existing non-user-recipient tool-channel behavior"
        );
    }

    #[test]
    fn preserves_reasoning_commentary_tool_commentary_final_order() {
        let input = concat!(
            "<|start|>assistant to=self<|message|>Need data.<|eom|>",
            "<|start|>assistant to=commentary<|message|>Checking weather.<|eom|>",
            "<|start|>assistant to=get_weather<|message|><atem:function_calls>",
            "<atem:invoke name=\"get_weather\"><atem:parameter name=\"city\">Paris</atem:parameter>",
            "</atem:invoke></atem:function_calls><|eom|>",
            "<|start|>assistant to=commentary<|message|>I found it.<|eom|>",
            "<|start|>assistant to=user<|message|>It is sunny.<|eot|>"
        );
        assert_eq!(
            parse(input),
            vec![
                reasoning("Need data."),
                text("Checking weather."),
                call("get_weather", serde_json::json!({"city": "Paris"})),
                text("I found it.It is sunny."),
            ]
        );
    }

    #[test]
    fn commentary_before_multiple_calls_and_marker_text_stays_visible() {
        let input = concat!(
            "<|start|>assistant to=commentary<|message|>I may quote ",
            "<atem:function_calls> here.<|eom|>",
            "<|start|>assistant to=get_weather<|message|><atem:function_calls>",
            "<atem:invoke name=\"get_weather\"><atem:parameter name=\"city\">Paris</atem:parameter></atem:invoke>",
            "<atem:invoke name=\"get_weather\"><atem:parameter name=\"city\">Rome</atem:parameter></atem:invoke>",
            "</atem:function_calls><|eom|>"
        );
        assert_eq!(
            parse(input),
            vec![
                text("I may quote <atem:function_calls> here."),
                call("get_weather", serde_json::json!({"city": "Paris"})),
                call("get_weather", serde_json::json!({"city": "Rome"})),
            ]
        );
    }

    #[test]
    fn commentary_markers_inside_tool_arguments_are_data() {
        let input = concat!(
            "<|start|>assistant to=get_weather<|message|><atem:function_calls>",
            "<atem:invoke name=\"get_weather\"><atem:parameter name=\"note\">",
            "to=commentary<|message|>literal",
            "</atem:parameter></atem:invoke></atem:function_calls><|eom|>"
        );
        assert_eq!(
            parse(input),
            vec![call(
                "get_weather",
                serde_json::json!({"note": "to=commentary<|message|>literal"})
            )]
        );
    }

    #[test]
    fn every_character_chunking_matches_batch() {
        let input = concat!(
            "<|start|>assistant to=self<|message|>think<|eom|>",
            "<|start|>assistant to=commentary<|message|>working<|eom|>",
            "<|start|>assistant to=user<|message|>done<|eot|>"
        );
        let chunks: Vec<String> = input.chars().map(|ch| ch.to_string()).collect();
        assert_eq!(parse_chunks(&chunks), parse(input));
    }

    #[test]
    fn guided_commentary_transitions_to_text_not_tool_framing() {
        let mut parser = muse_spark_unified(&tools());
        parser
            .initialize_request(UnifiedParserInit {
                prompt_token_ids: Vec::new(),
                starting_state: UnifiedParserStartingState::Reasoning,
                tool_output_mode: UnifiedToolOutputMode::GuidedJson { named_tool: None },
                invalid_guided_payload: Default::default(),
            })
            .expect("guided init");

        let input = concat!(
            "thinking<|eom|>",
            "<|start|>assistant to=commentary<|message|>Still working.<|eom|>",
            "[{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}]"
        );
        let mut deltas = parser.push(input).expect("push");
        deltas.extend(parser.finish().expect("finish"));
        assert_eq!(
            assemble(&deltas),
            vec![
                reasoning("thinking"),
                text("Still working."),
                call("get_weather", serde_json::json!({"city": "Paris"})),
            ]
        );
    }

    #[test]
    fn reset_returns_partial_header_and_restarts_policy() {
        let mut parser = muse_spark_unified(&tools());
        parser.push("<|start|>assistant to=comm").expect("push");
        assert_eq!(parser.reset(), "<|start|>assistant to=comm");
        assert_eq!(
            parser
                .push("<|start|>assistant to=commentary<|message|>again<|eom|>")
                .expect("push"),
            vec![UnifiedParserEvent::Text("again".to_string())]
        );
        assert!(parser.finish().expect("finish").is_empty());
    }

    #[test]
    fn reasoning_drops_narrated_invoke_closer_at_every_chunk_boundary() {
        let input = concat!(
            "<|start|>assistant to=self<|message|>",
            "I would close the example with </atem:invoke> before calling the tool.",
            "<|eom|>",
            "<|start|>assistant to=get_weather<|message|><atem:function_calls>",
            "<atem:invoke name=\"get_weather\"><atem:parameter name=\"city\">Paris</atem:parameter>",
            "</atem:invoke></atem:function_calls><|eom|>"
        );
        let expected = vec![
            reasoning("I would close the example with  before calling the tool."),
            call("get_weather", serde_json::json!({"city": "Paris"})),
        ];

        assert_every_chunking(input, &expected);
    }

    #[test]
    fn reasoning_preserves_narrated_self_recipient_at_every_chunk_boundary() {
        let input = concat!(
            "<|start|>assistant to=self<|message|>",
            "The previous assistant to=self used private reasoning.",
            "<|eom|><|start|>assistant to=user<|message|>Paris<|eot|>"
        );
        let expected = vec![
            reasoning("The previous assistant to=self used private reasoning."),
            text("Paris"),
        ];

        assert_every_chunking(input, &expected);
    }

    #[test]
    fn reasoning_drops_narrated_parameter_tags_at_every_chunk_boundary() {
        let input = concat!(
            "<|start|>assistant to=self<|message|>",
            "The example uses <atem:parameter name=\"tier\">2</atem:parameter> as its argument.",
            "<|eom|>",
            "<|start|>assistant to=get_weather<|message|><atem:function_calls>",
            "<atem:invoke name=\"get_weather\"><atem:parameter name=\"city\">Paris</atem:parameter>",
            "</atem:invoke></atem:function_calls><|eom|>"
        );
        let expected = vec![
            reasoning("The example uses 2 as its argument."),
            call("get_weather", serde_json::json!({"city": "Paris"})),
        ];

        assert_every_chunking(input, &expected);
    }

    #[test]
    fn guided_reasoning_drops_narrated_atem_before_required_call_at_every_chunk_boundary() {
        let input = concat!(
            "The assistant to=self would invoke the function in this format:\n",
            "<atem:invoke name=\"sum_values\">\n",
            "<atem:parameter name=\"values\">[1, 2, 3]</atem:parameter>\n",
            "</atem:invoke>\n",
            "<|eom|>",
            r#"[{"name":"sum_values","parameters":{"values":[1,2,3]}}]"#,
        );
        let expected = vec![
            reasoning("The assistant to=self would invoke the function in this format:\n\n"),
            call("sum_values", serde_json::json!({"values": [1, 2, 3]})),
        ];

        assert_eq!(
            parse_guided_chunks(&[
                concat!(
                    "The assistant to=self would invoke the function in this format:\n",
                    "<atem:invoke name=\"sum_values\">\n",
                    "<atem:parameter name=\"values\">[",
                )
                .to_string(),
                "1, 2, 3".to_string(),
                "]</atem:parameter>\n".to_string(),
                "</atem:invoke>\n".to_string(),
                "<|eom|>".to_string(),
                r#"[{"name":"sum_values","parameters":{"values":[1,2,3]}}]"#.to_string(),
            ]),
            expected,
            "production token chunks"
        );

        assert_eq!(
            parse_guided_chunks(&[input.to_string()]),
            expected,
            "whole input"
        );

        let per_character: Vec<String> = input.chars().map(|ch| ch.to_string()).collect();
        assert_eq!(
            parse_guided_chunks(&per_character),
            expected,
            "one character at a time"
        );

        for split in 1..input.len() {
            if !input.is_char_boundary(split) {
                continue;
            }
            assert_eq!(
                parse_guided_chunks(&[input[..split].to_string(), input[split..].to_string()]),
                expected,
                "split at byte {split}"
            );
        }
    }

    #[test]
    fn guided_reasoning_filter_does_not_touch_marker_shaped_tool_arguments() {
        let literal = concat!(
            "literal to=self <atem:invoke name=\"demo\">",
            "<atem:parameter name=\"x\">1</atem:parameter></atem:invoke>"
        );
        let payload = serde_json::json!([{
            "name": "get_weather",
            "parameters": {"note": literal},
        }]);
        let input = format!("thinking<|eom|>{payload}");

        assert_eq!(
            parse_guided_chunks(&[input]),
            vec![
                reasoning("thinking"),
                call("get_weather", serde_json::json!({"note": literal})),
            ]
        );
    }

    #[test]
    fn reserved_reasoning_markers_remain_data_outside_reasoning() {
        let content = concat!(
            "Visible examples: to=self, </atem:invoke>, and ",
            "<atem:parameter name=\"tier\">2</atem:parameter>."
        );
        let input = format!("<|start|>assistant to=user<|message|>{content}<|eot|>");
        assert_eq!(parse(&input), vec![text(content)]);

        let argument = "literal to=self";
        let tool_input = format!(
            "<|start|>assistant to=get_weather<|message|><atem:function_calls>\
             <atem:invoke name=\"get_weather\"><atem:parameter name=\"note\">\
             {argument}</atem:parameter></atem:invoke></atem:function_calls><|eom|>"
        );
        assert_eq!(
            parse(&tool_input),
            vec![call("get_weather", serde_json::json!({"note": argument}))]
        );
    }

    #[test]
    fn incomplete_reserved_prefix_is_preserved_at_finish_and_reset() {
        let input = concat!(
            "<|start|>assistant to=self<|message|>",
            "An incomplete example: </atem:inv"
        );
        assert_eq!(
            parse(input),
            vec![reasoning("An incomplete example: </atem:inv")]
        );

        let mut parser = muse_spark_unified(&tools());
        assert_eq!(
            parser.push(input).expect("push"),
            vec![UnifiedParserEvent::Reasoning(
                "An incomplete example: ".to_string()
            )]
        );
        assert_eq!(parser.reset(), "</atem:inv");
    }

    #[test]
    fn truncated_framed_answer_header_is_dropped_at_finish() {
        let out = parse_chunks(&[
            " to=self<|message|>Look it up.<|eom|>".to_string(),
            "<|start|>assistant to=user".to_string(),
        ]);

        assert_eq!(out, vec![reasoning("Look it up.")]);
    }

    #[test]
    fn guided_truncated_framed_answer_header_is_dropped_at_finish() {
        let mut parser = muse_spark_unified(&tools());
        parser
            .initialize_request(UnifiedParserInit {
                prompt_token_ids: Vec::new(),
                starting_state: UnifiedParserStartingState::None,
                tool_output_mode: UnifiedToolOutputMode::GuidedJson { named_tool: None },
                invalid_guided_payload: InvalidGuidedPayloadPolicy::RecoverAsText,
            })
            .expect("guided init");

        let mut deltas = parser
            .push(" to=self<|message|>Look it up.<|eom|>")
            .expect("reasoning push");
        deltas.extend(
            parser
                .push("<|start|>assistant to=user")
                .expect("truncated header push"),
        );
        deltas.extend(parser.finish().expect("finish"));

        assert_eq!(assemble(&deltas), vec![reasoning("Look it up.")]);
    }
}
