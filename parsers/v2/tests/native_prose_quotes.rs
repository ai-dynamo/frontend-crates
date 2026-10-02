// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_parsers_v2::{
    InvalidGuidedPayloadPolicy, Tool, UnifiedEvent, UnifiedParserExt, UnifiedParserInit,
    UnifiedParserStartingState, UnifiedToolOutputMode, assemble, create_unified_parser_for_family,
};

const FAMILIES: [(&str, &str, &str); 8] = [
    ("deepseek_v4", "<｜DSML｜tool_calls>", "</think>"),
    ("deepseek_v41", "<｜DSML｜ calls>", "</think>"),
    ("gemma4", "<|tool_call>", "<channel|>"),
    ("glm47", "<tool_call>", "</think>"),
    ("kimi_k2", "<|tool_calls_section_begin|>", "</think>"),
    ("kimi_k3", "<|open|>call", "<|close|>think<|sep|>"),
    (
        "muse_glimmer",
        "<|start|>assistant to=run<|message|>",
        "<|eom|>",
    ),
    ("qwen3", "<tool_call>", "</think>"),
];

fn tools() -> Vec<Tool> {
    vec![Tool {
        name: "run".into(),
        description: None,
        parameters: serde_json::json!({"type":"object","properties":{"cmd":{"type":"string"}}}),
        strict: None,
    }]
}

fn native_call(family: &str, value: &str) -> String {
    match family {
        "deepseek_v4" => format!(
            "<｜DSML｜tool_calls><｜DSML｜invoke name=\"run\"><｜DSML｜parameter name=\"cmd\" string=\"true\">{value}</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>"
        ),
        "deepseek_v41" => format!(
            "<｜DSML｜ calls><｜DSML｜ invoke name=\"run\"><｜DSML｜ parameter name=\"cmd\" string=\"true\">{value}</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>"
        ),
        "gemma4" => format!("<|tool_call>call:run{{cmd:<|\"|>{value}<|\"|>}}<tool_call|>"),
        "glm47" => format!(
            "<tool_call>run<arg_key>cmd</arg_key><arg_value>{value}</arg_value></tool_call>"
        ),
        "qwen3" => format!(
            "<tool_call><function=run><parameter=cmd>{value}</parameter></function></tool_call>"
        ),
        "kimi_k2" => format!(
            "<|tool_calls_section_begin|><|tool_call_begin|>functions.run:0<|tool_call_argument_begin|>{}<|tool_call_end|><|tool_calls_section_end|>",
            serde_json::json!({"cmd":value})
        ),
        "kimi_k3" => format!(
            "<|open|>tools<|sep|><|open|>call tool=\"run\" index=\"1\"<|sep|><|open|>argument key=\"cmd\" type=\"string\"<|sep|>{value}<|close|>argument<|sep|><|close|>call<|sep|><|close|>tools<|sep|>"
        ),
        "muse_glimmer" => format!(
            "<|start|>assistant to=run<|message|><atem:invoke name=\"run\"><atem:parameter name=\"cmd\">{value}</atem:parameter></atem:invoke><|eom|>"
        ),
        _ => unreachable!(),
    }
}

fn parse(
    family: &str,
    input: &str,
    state: UnifiedParserStartingState,
    at: usize,
) -> Vec<UnifiedEvent> {
    let mut parser = create_unified_parser_for_family(family, &tools()).unwrap();
    parser
        .initialize_request(UnifiedParserInit {
            starting_state: state,
            ..UnifiedParserInit::default()
        })
        .unwrap();
    let mut events = parser.push(&input[..at]).unwrap();
    events.extend(parser.push(&input[at..]).unwrap());
    events.extend(parser.finish().unwrap().events);
    assemble(&events)
}

fn every_split(
    family: &str,
    input: &str,
    state: UnifiedParserStartingState,
    expected: &[UnifiedEvent],
) {
    for at in (0..=input.len()).filter(|at| input.is_char_boundary(*at)) {
        assert_eq!(
            parse(family, input, state, at),
            expected,
            "{family} at {at}: {input:?}"
        );
    }
    let mut parser = create_unified_parser_for_family(family, &tools()).unwrap();
    parser
        .initialize_request(UnifiedParserInit {
            starting_state: state,
            ..UnifiedParserInit::default()
        })
        .unwrap();
    let mut events = Vec::new();
    for character in input.chars() {
        events.extend(parser.push(&character.to_string()).unwrap());
    }
    events.extend(parser.finish().unwrap().events);
    assert_eq!(assemble(&events), expected, "{family} character chunks");
}

#[test]
fn quoted_native_controls_remain_prose_in_both_channels_at_every_split() {
    for (family, marker, _) in FAMILIES {
        for quote in ['"', '\'', '`'] {
            let input =
                format!("The literal {quote}{marker}{quote} marker is part of the explanation.");
            for (state, expected) in [
                (
                    UnifiedParserStartingState::Response,
                    UnifiedEvent::Text {
                        text: input.clone(),
                    },
                ),
                (
                    UnifiedParserStartingState::Reasoning,
                    UnifiedEvent::Reasoning {
                        text: input.clone(),
                    },
                ),
            ] {
                every_split(family, &input, state, &[expected]);
            }
        }
    }
}

#[test]
fn literal_controls_do_not_hide_following_calls_or_argument_bytes() {
    for (family, marker, _) in FAMILIES {
        let prefix = format!("The literal \"{marker}\" marker; James' request isn't 'twas wrong. ");
        let value = format!("echo \"{marker}\" \\path `quoted` value");
        let input = format!("{prefix}{}", native_call(family, &value));
        every_split(
            family,
            &input,
            UnifiedParserStartingState::Response,
            &[
                UnifiedEvent::Text { text: prefix },
                UnifiedEvent::ToolCall {
                    name: "run".into(),
                    arguments: serde_json::json!({"cmd":value}),
                },
            ],
        );
    }
}

#[test]
fn unmatched_quote_cannot_hide_the_real_reasoning_closer() {
    for (family, _, closer) in FAMILIES {
        for quote in ['"', '\'', '`'] {
            let reasoning = format!("He said {quote}maybe");
            let input = format!("{reasoning}{closer}Answer");
            every_split(
                family,
                &input,
                UnifiedParserStartingState::Reasoning,
                &[
                    UnifiedEvent::Reasoning { text: reasoning },
                    UnifiedEvent::Text {
                        text: "Answer".into(),
                    },
                ],
            );
        }
    }
}

#[test]
fn definite_prose_progresses_and_reset_returns_the_ambiguous_control() {
    for (family, marker, _) in FAMILIES {
        let mut parser = create_unified_parser_for_family(family, &tools()).unwrap();
        parser
            .initialize_request(UnifiedParserInit {
                starting_state: UnifiedParserStartingState::Response,
                ..UnifiedParserInit::default()
            })
            .unwrap();
        let prefix = "Definite prose before a quote \"";
        let emitted = parser.push(prefix).unwrap();
        assert_eq!(
            assemble(&emitted),
            vec![UnifiedEvent::Text {
                text: prefix.into()
            }],
            "{family} progress"
        );
        assert!(
            parser.push(marker).unwrap().is_empty(),
            "{family} ambiguous control waits"
        );
        assert_eq!(parser.reset(), marker, "{family} reset ownership");
        let call = native_call(family, "Paris");
        let mut got = parser.push(&call).unwrap();
        got.extend(parser.finish().unwrap().events);
        assert_eq!(
            assemble(&got),
            vec![UnifiedEvent::ToolCall {
                name: "run".into(),
                arguments: serde_json::json!({"cmd":"Paris"})
            }],
            "{family} reset quote state"
        );
    }
}

#[test]
fn unmatched_quote_cannot_hide_a_real_call_with_quoted_attributes() {
    for (family, _, _) in FAMILIES {
        let prefix = "He said \"maybe";
        let input = format!("{prefix}{}", native_call(family, "Paris"));
        every_split(
            family,
            &input,
            UnifiedParserStartingState::Response,
            &[
                UnifiedEvent::Text {
                    text: prefix.into(),
                },
                UnifiedEvent::ToolCall {
                    name: "run".into(),
                    arguments: serde_json::json!({"cmd":"Paris"}),
                },
            ],
        );
    }
}

#[test]
fn inactive_muse_invoke_attributes_keep_the_same_marker_recovery_at_every_split() {
    let input = r#"<atem:invoke name="<|eom|>[{"name":"f","arguments":{"x":"ok"}}]"#;
    let expected = parse(
        "muse_glimmer",
        input,
        UnifiedParserStartingState::Response,
        0,
    );
    every_split(
        "muse_glimmer",
        input,
        UnifiedParserStartingState::Response,
        &expected,
    );
}

#[test]
fn single_quoted_words_own_their_embedded_control() {
    for (family, marker, _) in FAMILIES {
        let input = format!("The literal 'example {marker} marker' is part of the explanation.");
        every_split(
            family,
            &input,
            UnifiedParserStartingState::Response,
            &[UnifiedEvent::Text {
                text: input.clone(),
            }],
        );
    }
}

#[test]
fn contractions_do_not_close_single_quoted_controls() {
    for (family, marker, closer) in FAMILIES {
        for control in [marker, closer] {
            for body in ["doesn't", "isn't", "doesn't mean it isn't", "caf\u{e9}'s"] {
                let input = format!("The literal '{body} {control} marker' stays visible.");
                for (state, expected) in [
                    (
                        UnifiedParserStartingState::Response,
                        UnifiedEvent::Text {
                            text: input.clone(),
                        },
                    ),
                    (
                        UnifiedParserStartingState::Reasoning,
                        UnifiedEvent::Reasoning {
                            text: input.clone(),
                        },
                    ),
                ] {
                    every_split(family, &input, state, &[expected]);
                }
            }
        }
    }
}

#[test]
fn guided_quoted_reasoning_controls_preserve_the_channel_at_every_split() {
    for (family, marker, closer) in FAMILIES {
        for quote in ['"', '\'', '`'] {
            for control in [marker, closer] {
                let reasoning =
                    format!("The literal {quote}doesn't {control} marker{quote} stays quoted.");
                for named_tool in [None, Some("run".to_string())] {
                    let payload = if named_tool.is_some() {
                        serde_json::json!({"cmd":"ok"})
                    } else {
                        serde_json::json!({"name":"run","arguments":{"cmd":"ok"}})
                    };
                    let input = format!("{reasoning}{closer}{payload}");
                    let expected = vec![
                        UnifiedEvent::Reasoning {
                            text: reasoning.clone(),
                        },
                        UnifiedEvent::ToolCall {
                            name: "run".to_string(),
                            arguments: serde_json::json!({"cmd":"ok"}),
                        },
                    ];
                    for at in (0..=input.len()).filter(|at| input.is_char_boundary(*at)) {
                        let mut parser =
                            create_unified_parser_for_family(family, &tools()).unwrap();
                        parser
                            .initialize_request(UnifiedParserInit {
                                starting_state: UnifiedParserStartingState::Reasoning,
                                tool_output_mode: UnifiedToolOutputMode::GuidedJson {
                                    named_tool: named_tool.clone(),
                                },
                                invalid_guided_payload:
                                    InvalidGuidedPayloadPolicy::StreamBestEffort,
                                ..UnifiedParserInit::default()
                            })
                            .unwrap();
                        let mut events = parser.push(&input[..at]).unwrap();
                        events.extend(parser.push(&input[at..]).unwrap());
                        events.extend(parser.finish().unwrap().events);
                        assert_eq!(
                            assemble(&events),
                            expected,
                            "{family} {quote} {named_tool:?} at {at}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn muse_parameter_quotes_preserve_literals_and_unmatched_quotes_keep_channel_recovery() {
    for marker in ["<|eom|>", "<|eot|>", "<|start|>"] {
        let value = format!("a\"{marker}\"b");
        let input = native_call("muse_glimmer", &value);
        every_split(
            "muse_glimmer",
            &input,
            UnifiedParserStartingState::Response,
            &[UnifiedEvent::ToolCall {
                name: "run".into(),
                arguments: serde_json::json!({"cmd":value}),
            }],
        );
    }
    let input = "<|start|>assistant to=run<|message|><atem:invoke name=\"run\"><atem:parameter name=\"cmd\">He said \"maybe<|eom|><|start|>assistant to=user<|message|>Answer<|eom|>";
    every_split(
        "muse_glimmer",
        input,
        UnifiedParserStartingState::Response,
        &[UnifiedEvent::Text {
            text: "Answer".into(),
        }],
    );
}

#[test]
fn rejected_gemma_call_header_does_not_turn_attribute_quotes_into_prose() {
    let input = r#"call:f"<tool_call|>[{"name":"f","arguments":{"x":"ok"}}]"#;
    let expected = parse("gemma4", input, UnifiedParserStartingState::Response, 0);
    every_split(
        "gemma4",
        input,
        UnifiedParserStartingState::Response,
        &expected,
    );
}

#[test]
fn guided_visible_quote_owns_braces_before_the_payload() {
    for (family, marker, _) in FAMILIES {
        for quote in ['"', '\'', '`'] {
            for value in ["{ example }", "[ example ]", "é🙂 \\\"escaped\\\" value"] {
                let prose = format!("The literal {quote}{marker} {value}{quote} stays visible. ");
                for named_tool in [None, Some("run".to_string())] {
                    let payload = if named_tool.is_some() {
                        serde_json::json!({"cmd":"ok"})
                    } else {
                        serde_json::json!({"name":"run","arguments":{"cmd":"ok"}})
                    };
                    let input = format!("{prose}{payload}");
                    for at in (0..=input.len()).filter(|at| input.is_char_boundary(*at)) {
                        let mut parser =
                            create_unified_parser_for_family(family, &tools()).unwrap();
                        parser
                            .initialize_request(UnifiedParserInit {
                                starting_state: UnifiedParserStartingState::Response,
                                tool_output_mode: UnifiedToolOutputMode::GuidedJson {
                                    named_tool: named_tool.clone(),
                                },
                                invalid_guided_payload:
                                    InvalidGuidedPayloadPolicy::StreamBestEffort,
                                ..Default::default()
                            })
                            .unwrap();
                        let mut events = parser.push(&input[..at]).unwrap();
                        events.extend(parser.push(&input[at..]).unwrap());
                        events.extend(parser.finish().unwrap().events);
                        assert_eq!(
                            assemble(&events),
                            vec![
                                UnifiedEvent::Text {
                                    text: prose.clone()
                                },
                                UnifiedEvent::ToolCall {
                                    name: "run".into(),
                                    arguments: serde_json::json!({"cmd":"ok"})
                                }
                            ],
                            "{family} {quote} {value} {named_tool:?} at {at}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn guided_rejected_header_quotes_do_not_shield_parameter_markup() {
    let tool = Tool {
        name: "f".into(),
        description: None,
        parameters: serde_json::json!({"type":"object","properties":{"x":{"type":"string"}}}),
        strict: None,
    };
    let input = "<｜DSML｜invoke name=\"f\"<｜DSML｜parameter name=\"x\" string=\"true\">x[{\"name\":\"f\",\"arguments\":{\"x\":\"ok\"}}]";
    for at in (0..=input.len()).filter(|at| input.is_char_boundary(*at)) {
        let mut parser =
            create_unified_parser_for_family("deepseek_v4", std::slice::from_ref(&tool)).unwrap();
        parser
            .initialize_request(UnifiedParserInit {
                starting_state: UnifiedParserStartingState::Response,
                tool_output_mode: UnifiedToolOutputMode::GuidedJson { named_tool: None },
                invalid_guided_payload: InvalidGuidedPayloadPolicy::RecoverAsText,
                ..Default::default()
            })
            .unwrap();
        let mut events = parser.push(&input[..at]).unwrap();
        events.extend(parser.push(&input[at..]).unwrap());
        events.extend(parser.finish().unwrap().events);
        assert_eq!(
            assemble(&events),
            vec![
                UnifiedEvent::Text {
                    text: "f\"x".into()
                },
                UnifiedEvent::ToolCall {
                    name: "f".into(),
                    arguments: serde_json::json!({"x":"ok"})
                }
            ],
            "split {at}"
        );
    }
}

#[test]
fn reject_guided_preserves_quoted_complete_native_calls() {
    // The corpus Init schema cannot select Reject, so exercise its transactional policy here.
    for (family, _, _) in FAMILIES {
        let prose = format!(
            "The literal `{}` stays visible. ",
            native_call(family, "quoted")
        );
        for named_tool in [None, Some("run".to_string())] {
            let payload = if named_tool.is_some() {
                serde_json::json!({"cmd":"ok"})
            } else {
                serde_json::json!({"name":"run","arguments":{"cmd":"ok"}})
            };
            let input = format!("{prose}{payload}");
            for at in (0..=input.len()).filter(|at| input.is_char_boundary(*at)) {
                let mut parser = create_unified_parser_for_family(family, &tools()).unwrap();
                parser
                    .initialize_request(UnifiedParserInit {
                        starting_state: UnifiedParserStartingState::Response,
                        tool_output_mode: UnifiedToolOutputMode::GuidedJson {
                            named_tool: named_tool.clone(),
                        },
                        invalid_guided_payload: InvalidGuidedPayloadPolicy::Reject,
                        ..Default::default()
                    })
                    .unwrap();
                let mut events = parser.push(&input[..at]).unwrap();
                events.extend(parser.push(&input[at..]).unwrap());
                events.extend(parser.finish().unwrap().events);
                assert_eq!(
                    assemble(&events),
                    vec![
                        UnifiedEvent::Text {
                            text: prose.clone()
                        },
                        UnifiedEvent::ToolCall {
                            name: "run".into(),
                            arguments: serde_json::json!({"cmd":"ok"})
                        }
                    ],
                    "{family} {named_tool:?} split {at}"
                );
            }
        }
    }
}

#[test]
fn malformed_response_candidate_has_chunk_independent_provisional_fate() {
    // Corpus Init cannot select StreamBestEffort or observe provisional call fate.
    for tail in [
        r#"{"cmd":"x{"cmd":"ok"}"#,
        r#"{"cmd":"x<|tool_call_end|>{"cmd":"ok"}"#,
    ] {
        let input = format!("<|tool_call_begin|>{tail}");
        let run = |split: usize| {
            let mut parser = create_unified_parser_for_family("kimi_k2", &tools()).unwrap();
            parser
                .initialize_request(UnifiedParserInit {
                    starting_state: UnifiedParserStartingState::Response,
                    tool_output_mode: UnifiedToolOutputMode::GuidedJson {
                        named_tool: Some("run".into()),
                    },
                    invalid_guided_payload: InvalidGuidedPayloadPolicy::StreamBestEffort,
                    ..Default::default()
                })
                .unwrap();
            let mut events = parser.push(&input[..split]).unwrap();
            events.extend(parser.push(&input[split..]).unwrap());
            events.extend(parser.finish().unwrap().events);
            assemble(&events)
        };
        let whole = run(0);
        for split in (0..=input.len()).filter(|at| input.is_char_boundary(*at)) {
            assert_eq!(run(split), whole, "split {split}");
        }
    }
}

#[test]
fn response_resynchronization_respects_earlier_provisional_ownership() {
    // Corpus Init cannot select StreamBestEffort or assert provisional call fate.
    for (family, _, _) in FAMILIES {
        for named in [false, true] {
            for invalid in [" BAD ", "] ", "\\q ", ", invalid ", "\n "] {
                let input = if named {
                    format!(r#"{{"cmd":"x"{invalid}{{"cmd":"ok"}}"#)
                } else {
                    format!(
                        r#"[{{"name":"run","arguments":{{"cmd":"x"{invalid}[{{"name":"run","arguments":{{"cmd":"ok"}}}}]"#
                    )
                };
                let run = |split: usize| {
                    let mut parser = create_unified_parser_for_family(family, &tools()).unwrap();
                    parser
                        .initialize_request(UnifiedParserInit {
                            starting_state: UnifiedParserStartingState::Response,
                            tool_output_mode: UnifiedToolOutputMode::GuidedJson {
                                named_tool: named.then(|| "run".into()),
                            },
                            invalid_guided_payload: InvalidGuidedPayloadPolicy::StreamBestEffort,
                            ..Default::default()
                        })
                        .unwrap();
                    let mut events = parser.push(&input[..split]).unwrap();
                    events.extend(parser.push(&input[split..]).unwrap());
                    events.extend(parser.finish().unwrap().events);
                    assemble(&events)
                };
                let whole = run(0);
                for split in (0..=input.len()).filter(|at| input.is_char_boundary(*at)) {
                    assert_eq!(
                        run(split),
                        whole,
                        "{family} named={named} invalid={invalid:?} split={split}"
                    );
                }
            }
        }
    }
}
