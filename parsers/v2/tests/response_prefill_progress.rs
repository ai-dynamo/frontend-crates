// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_parsers_v2::{
    InvalidGuidedPayloadPolicy, Qwen3CoderToolStreamParser, Tool, ToolParser, UnifiedEvent,
    UnifiedParserEvent, UnifiedParserExt, UnifiedParserInit, UnifiedParserStartingState,
    UnifiedToolOutputMode, assemble, create_unified_parser_for_family,
};

fn tools() -> Vec<Tool> {
    vec![Tool {
        name: "write_file".into(),
        description: None,
        parameters: serde_json::json!({
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"]
        }),
        strict: None,
    }]
}

fn mode(named: bool) -> UnifiedToolOutputMode {
    UnifiedToolOutputMode::GuidedJson {
        named_tool: named.then(|| "write_file".into()),
    }
}

fn parts(named: bool) -> (&'static str, &'static str) {
    if named {
        (r#"{"path":"x","content":"prefix <think>"#, r#" suffix"}"#)
    } else {
        (
            r#"[{"name":"write_file","arguments":{"path":"x","content":"prefix <think>"#,
            r#" suffix"}}]"#,
        )
    }
}

#[test]
fn response_prefill_native_body_cannot_become_the_guided_payload() {
    for (family, header, trailer, literal_prefix) in [
        (
            "gpt_oss",
            "<|start|>assistant<|channel|>commentary to=functions.ignored <|constrain|>json<|message|>",
            "<|call|>",
            "I mean <|start|>assistant<|channel|>analysis<|message|>self literal<|end|>",
        ),
        (
            "gpt_oss",
            "<|start|>assistant to=functions.ignored<|channel|>commentary<|message|>",
            "<|call|>",
            "I mean <|start|>assistant<|channel|>analysis<|message|>self literal<|end|>",
        ),
        (
            "kimi_k3",
            "<|open|>tools<|sep|><|open|>call tool=\"ignored\" index=\"1\"<|sep|>",
            "<|close|>call<|sep|><|close|>tools<|sep|>",
            "I mean <|open|>think<|sep|>self literal<|close|>think<|sep|>",
        ),
    ] {
        for array in [false, true] {
            let value = if array {
                serde_json::json!([{"name":"get_weather","arguments":{"city":"Rome"}}])
            } else {
                serde_json::json!({"city":"Rome"})
            };
            let body = if family == "gpt_oss" {
                serde_json::json!({"x":value}).to_string()
            } else {
                format!(
                    "<|open|>argument key=\"x\" type=\"{}\"<|sep|>{value}<|close|>argument<|sep|>",
                    if array { "array" } else { "object" }
                )
            };
            let quoted_prefix = format!("\"{literal_prefix}\" then prose ");
            let non_call_prefix = format!("{{}}{literal_prefix}");
            for prefix in ["", literal_prefix, &quoted_prefix, &non_call_prefix] {
                for named in [false, true] {
                    if named && prefix == non_call_prefix {
                        continue;
                    }
                    let payload = if named {
                        r#"{"city":"Paris"}"#
                    } else {
                        r#"[{"name":"get_weather","arguments":{"city":"Paris"}}]"#
                    };
                    let input = format!("{prefix}{header}{body}{trailer}{payload}");
                    let mut expected = Vec::new();
                    if !prefix.is_empty() {
                        expected.push(UnifiedEvent::Text {
                            text: prefix.into(),
                        });
                    }
                    expected.push(UnifiedEvent::ToolCall {
                        name: "get_weather".into(),
                        arguments: serde_json::json!({"city":"Paris"}),
                    });
                    for policy in [
                        InvalidGuidedPayloadPolicy::Reject,
                        InvalidGuidedPayloadPolicy::RecoverAsText,
                        InvalidGuidedPayloadPolicy::StreamBestEffort,
                    ] {
                        let mut parser = create_unified_parser_for_family(family, &[]).unwrap();
                        let init = UnifiedParserInit {
                            starting_state: UnifiedParserStartingState::Response,
                            tool_output_mode: UnifiedToolOutputMode::GuidedJson {
                                named_tool: named.then(|| "get_weather".into()),
                            },
                            invalid_guided_payload: policy,
                            ..Default::default()
                        };
                        if policy == InvalidGuidedPayloadPolicy::Reject {
                            let incomplete = format!("{prefix}{header}{body}{trailer}");
                            for split in incomplete
                                .char_indices()
                                .map(|(at, _)| at)
                                .chain([incomplete.len()])
                            {
                                parser.reset();
                                parser.initialize_request(init.clone()).unwrap();
                                assert!(parser.push(&incomplete[..split]).unwrap().is_empty());
                                assert!(parser.push(&incomplete[split..]).unwrap().is_empty());
                                assert!(parser.finish().is_err());
                            }
                        }
                        for split in input.char_indices().map(|(at, _)| at).chain([input.len()]) {
                            parser.reset();
                            parser.initialize_request(init.clone()).unwrap();
                            let mut events = parser.push(&input[..split]).unwrap();
                            events.extend(parser.push(&input[split..]).unwrap());
                            events.extend(parser.finish().unwrap().events);
                            assert_eq!(
                                assemble(&events),
                                expected,
                                "{family} {policy:?} named={named} array={array} prefix={} split={split}",
                                !prefix.is_empty()
                            );
                        }
                        parser.reset();
                        parser.initialize_request(init.clone()).unwrap();
                        let mut events = Vec::new();
                        for character in input.chars() {
                            events.extend(parser.push(&character.to_string()).unwrap());
                        }
                        events.extend(parser.finish().unwrap().events);
                        assert_eq!(
                            assemble(&events),
                            expected,
                            "{family} {policy:?} named={named} array={array} character chunks"
                        );
                        if family == "gpt_oss" {
                            let ids =
                                dynamo_parsers_v2::tool_calling::harmony::encode_harmony(&input)
                                    .unwrap();
                            for split in 0..=ids.len() {
                                parser.reset();
                                parser.initialize_request(init.clone()).unwrap();
                                let mut events = parser.push_tokens(&ids[..split]).unwrap();
                                events.extend(parser.push_tokens(&ids[split..]).unwrap());
                                events.extend(parser.finish().unwrap().events);
                                assert_eq!(
                                    assemble(&events),
                                    expected,
                                    "{family} {policy:?} named={named} array={array} token split={split}"
                                );
                            }
                            parser.reset();
                            parser.initialize_request(init).unwrap();
                            let mut events = Vec::new();
                            for id in ids.chunks(1) {
                                events.extend(parser.push_tokens(id).unwrap());
                            }
                            events.extend(parser.finish().unwrap().events);
                            assert_eq!(
                                assemble(&events),
                                expected,
                                "{family} {policy:?} named={named} array={array} single tokens"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn response_prefill_boundary_owned_native_body_stays_outside_guided_payload() {
    // Corpus initialization cannot choose all three public invalid-payload policies.
    for (family, input) in [
        (
            "deepseek_v4",
            "I mean <think>self literal</think><｜DSML｜tool_calls><｜DSML｜invoke name=\"f\"><｜DSML｜parameter name=\"x\" string=\"false\">{\"city\": \"Rome\"}</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>[{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}]",
        ),
        (
            "deepseek_v4",
            "I mean <think>self literal</think><｜DSML｜tool_calls><｜DSML｜invoke name=\"f\"><｜DSML｜parameter name=\"x\" string=\"false\">[{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Rome\"}}]</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>[{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}]",
        ),
        (
            "deepseek_v41",
            "I mean <think>self literal</think><｜DSML｜ calls><｜DSML｜ invoke name=\"f\"><｜DSML｜ parameter name=\"x\" string=\"false\">{\"city\": \"Rome\"}</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>[{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}]",
        ),
        (
            "deepseek_v41",
            "I mean <think>self literal</think><｜DSML｜ calls><｜DSML｜ invoke name=\"f\"><｜DSML｜ parameter name=\"x\" string=\"false\">[{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Rome\"}}]</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>[{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}]",
        ),
        (
            "glm47",
            "I mean <think>self literal</think><tool_call>f<arg_key>x</arg_key><arg_value>{\"city\": \"Rome\"}</arg_value></tool_call>[{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}]",
        ),
        (
            "glm47",
            "I mean <think>self literal</think><tool_call>f<arg_key>x</arg_key><arg_value>[{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Rome\"}}]</arg_value></tool_call>[{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}]",
        ),
    ] {
        for non_call in ["", "{}"] {
            let input = format!("{non_call}{input}");
            let expected = vec![
                UnifiedEvent::Text {
                    text: format!("{non_call}I mean <think>self literal</think>"),
                },
                UnifiedEvent::ToolCall {
                    name: "get_weather".into(),
                    arguments: serde_json::json!({"city":"Paris"}),
                },
            ];
            for policy in [
                InvalidGuidedPayloadPolicy::Reject,
                InvalidGuidedPayloadPolicy::RecoverAsText,
                InvalidGuidedPayloadPolicy::StreamBestEffort,
            ] {
                let mut parser = create_unified_parser_for_family(family, &[]).unwrap();
                let init = UnifiedParserInit {
                    starting_state: UnifiedParserStartingState::Response,
                    tool_output_mode: UnifiedToolOutputMode::GuidedJson { named_tool: None },
                    invalid_guided_payload: policy,
                    ..Default::default()
                };
                for split in input.char_indices().map(|(at, _)| at).chain([input.len()]) {
                    parser.reset();
                    parser.initialize_request(init.clone()).unwrap();
                    let mut events = parser.push(&input[..split]).unwrap();
                    events.extend(parser.push(&input[split..]).unwrap());
                    events.extend(parser.finish().unwrap().events);
                    assert_eq!(
                        assemble(&events),
                        expected,
                        "{family} {policy:?} split={split}"
                    );
                }
                parser.reset();
                parser.initialize_request(init).unwrap();
                let mut events = Vec::new();
                for character in input.chars() {
                    events.extend(parser.push(&character.to_string()).unwrap());
                }
                events.extend(parser.finish().unwrap().events);
                assert_eq!(
                    assemble(&events),
                    expected,
                    "{family} {policy:?} character chunks"
                );
            }
        }
    }
}

#[test]
fn response_prefill_native_string_json_is_not_a_guided_call() {
    // Corpus initialization cannot choose all three public invalid-payload policies.
    let native = r#"<|tool_call>call:f{x:<|"|>[{"name":"get_weather","arguments":{"city":"Rome"}}]<|"|>}<tool_call|>"#;
    let payload = r#"[{"name":"get_weather","arguments":{"city":"Paris"}}]"#;
    for prefix in ["", "{}"] {
        let input = format!("{prefix}{native}{payload}");
        let mut expected = Vec::new();
        if !prefix.is_empty() {
            expected.push(UnifiedEvent::Text {
                text: prefix.into(),
            });
        }
        expected.push(UnifiedEvent::ToolCall {
            name: "get_weather".into(),
            arguments: serde_json::json!({"city":"Paris"}),
        });
        for policy in [
            InvalidGuidedPayloadPolicy::Reject,
            InvalidGuidedPayloadPolicy::RecoverAsText,
            InvalidGuidedPayloadPolicy::StreamBestEffort,
        ] {
            let mut parser = create_unified_parser_for_family("gemma4", &[]).unwrap();
            let init = UnifiedParserInit {
                starting_state: UnifiedParserStartingState::Response,
                tool_output_mode: UnifiedToolOutputMode::GuidedJson { named_tool: None },
                invalid_guided_payload: policy,
                ..Default::default()
            };
            for split in 0..=input.len() {
                parser.reset();
                parser.initialize_request(init.clone()).unwrap();
                let mut events = parser.push(&input[..split]).unwrap();
                events.extend(parser.push(&input[split..]).unwrap());
                assert_eq!(
                    assemble(&events),
                    expected,
                    "{policy:?} split={split} before finish"
                );
                events.extend(parser.finish().unwrap().events);
                assert_eq!(assemble(&events), expected, "{policy:?} split={split}");
            }
        }
    }
}

#[test]
fn response_prefill_pending_native_array_keeps_its_tail_owned() {
    // Corpus initialization cannot choose all three public invalid-payload policies.
    for prefix in ["", "{}"] {
        let input = format!(
            "{prefix}{}",
            "I mean <|channel>thought\nself literal<channel|><|tool_call>call:f{x:[{name:<|\"|>get_weather<|\"|>,arguments:{city:<|\"|>Rome<|\"|>}}]}<tool_call|>[{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}]"
        );
        let expected = vec![
            UnifiedEvent::Text {
                text: format!(
                    "{prefix}{}",
                    "I mean <|channel>thought\nself literal<channel|>"
                ),
            },
            UnifiedEvent::ToolCall {
                name: "get_weather".into(),
                arguments: serde_json::json!({"city":"Paris"}),
            },
        ];
        for policy in [
            InvalidGuidedPayloadPolicy::Reject,
            InvalidGuidedPayloadPolicy::RecoverAsText,
            InvalidGuidedPayloadPolicy::StreamBestEffort,
        ] {
            let mut parser = create_unified_parser_for_family("gemma4", &[]).unwrap();
            let init = UnifiedParserInit {
                starting_state: UnifiedParserStartingState::Response,
                tool_output_mode: UnifiedToolOutputMode::GuidedJson { named_tool: None },
                invalid_guided_payload: policy,
                ..Default::default()
            };
            for split in 0..=input.len() {
                parser.reset();
                parser.initialize_request(init.clone()).unwrap();
                let mut events = parser.push(&input[..split]).unwrap();
                events.extend(parser.push(&input[split..]).unwrap());
                assert_eq!(
                    assemble(&events),
                    expected,
                    "{policy:?} split={split} before finish"
                );
                events.extend(parser.finish().unwrap().events);
                assert_eq!(assemble(&events), expected, "{policy:?} split={split}");
            }
        }
    }
}

#[test]
fn response_prefill_streams_long_argument_after_literal_marker() {
    // Corpus Init cannot select StreamBestEffort or assert pre-close deltas, so
    // the live request policy and its progress deadline need a Rust regression.
    for family in dynamo_parsers_v2::REGISTERED_UNIFIED_FAMILIES {
        for named in [false, true] {
            let (head, tail) = parts(named);
            for split in 0..=head.len() {
                let mut parser = create_unified_parser_for_family(family, &tools()).unwrap();
                parser
                    .initialize_request(UnifiedParserInit {
                        starting_state: UnifiedParserStartingState::Response,
                        tool_output_mode: mode(named),
                        invalid_guided_payload: InvalidGuidedPayloadPolicy::StreamBestEffort,
                        ..Default::default()
                    })
                    .unwrap();
                let mut events = parser.push(&head[..split]).unwrap();
                events.extend(parser.push(&head[split..]).unwrap());
                let body = "a".repeat(8192);
                for fragment in body.as_bytes().chunks(128) {
                    let deltas = parser.push(std::str::from_utf8(fragment).unwrap()).unwrap();
                    assert!(
                        deltas.iter().any(|event| matches!(event, UnifiedParserEvent::ToolCall(call) if call.arguments.contains("aaaa"))),
                        "{family} named={named} split={split}: argument progress stopped before the string closed"
                    );
                    events.extend(deltas);
                }
                events.extend(parser.push(tail).unwrap());
                events.extend(parser.finish().unwrap().events);
                assert_eq!(
                    assemble(&events),
                    vec![UnifiedEvent::ToolCall {
                        name: "write_file".into(),
                        arguments: serde_json::json!({"path":"x", "content":format!("prefix <think>{body} suffix")})
                    }],
                    "{family} named={named} split={split}"
                );
                assert_eq!(events.iter().filter(|event| matches!(event, UnifiedParserEvent::ToolCall(call) if call.complete)).count(), 1);
            }
        }
    }
}

#[test]
fn response_prefill_buffering_contracts_keep_arguments_transactional() {
    for family in dynamo_parsers_v2::REGISTERED_UNIFIED_FAMILIES {
        for named in [false, true] {
            for policy in [
                InvalidGuidedPayloadPolicy::RecoverAsText,
                InvalidGuidedPayloadPolicy::Reject,
            ] {
                let mut parser = create_unified_parser_for_family(family, &tools()).unwrap();
                parser
                    .initialize_request(UnifiedParserInit {
                        starting_state: UnifiedParserStartingState::Response,
                        tool_output_mode: mode(named),
                        invalid_guided_payload: policy,
                        ..Default::default()
                    })
                    .unwrap();
                let (head, tail) = parts(named);
                let mut early = parser.push(head).unwrap();
                early.extend(parser.push("aaaa").unwrap());
                assert!(
                    !early
                        .iter()
                        .any(|event| matches!(event, UnifiedParserEvent::ToolCall(_))),
                    "{family} named={named} {policy:?}"
                );
                let mut events = early;
                events.extend(parser.push(tail).unwrap());
                events.extend(parser.finish().unwrap().events);
                assert!(assemble(&events).iter().any(|event| matches!(event, UnifiedEvent::ToolCall { name, .. } if name == "write_file")));
            }
        }
    }
}

#[test]
fn response_prefill_rejects_or_recovers_malformed_payload_without_dispatch() {
    for family in dynamo_parsers_v2::REGISTERED_UNIFIED_FAMILIES {
        for named in [false, true] {
            let raw = if named {
                r#"{"path":"x","content":"unterminated"#
            } else {
                r#"[{"name":"write_file","arguments":{"path":"x","content":"unterminated"#
            };
            for policy in [
                InvalidGuidedPayloadPolicy::RecoverAsText,
                InvalidGuidedPayloadPolicy::Reject,
            ] {
                let mut parser = create_unified_parser_for_family(family, &tools()).unwrap();
                parser
                    .initialize_request(UnifiedParserInit {
                        starting_state: UnifiedParserStartingState::Response,
                        tool_output_mode: mode(named),
                        invalid_guided_payload: policy,
                        ..Default::default()
                    })
                    .unwrap();
                let mut events = Vec::new();
                for character in raw.chars() {
                    events.extend(parser.push(&character.to_string()).unwrap());
                }
                assert!(events.is_empty(), "{family} {named} {policy:?}");
                if policy == InvalidGuidedPayloadPolicy::Reject {
                    let error = parser.finish().unwrap_err();
                    assert!(error.is::<dynamo_parsers_v2::InvalidGuidedPayload>());
                } else {
                    events.extend(parser.finish().unwrap().events);
                    assert_eq!(
                        assemble(&events),
                        vec![UnifiedEvent::Text { text: raw.into() }]
                    );
                }
            }
        }
    }
}

#[test]
fn response_prefill_skips_non_call_json_and_preserves_no_tool_prose() {
    for family in dynamo_parsers_v2::REGISTERED_UNIFIED_FAMILIES {
        for offered_tools in [tools(), Vec::new()] {
            for prose in [
                r#"{"message":"ordinary JSON prose"}"#,
                "The list is [].",
                "See [docs] before ",
            ] {
                let mut parser = create_unified_parser_for_family(family, &offered_tools).unwrap();
                parser
                    .initialize_request(UnifiedParserInit {
                        starting_state: UnifiedParserStartingState::Response,
                        tool_output_mode: mode(false),
                        invalid_guided_payload: InvalidGuidedPayloadPolicy::StreamBestEffort,
                        ..Default::default()
                    })
                    .unwrap();
                let mut events = Vec::new();
                for character in prose.chars() {
                    events.extend(parser.push(&character.to_string()).unwrap());
                }
                events.extend(parser.finish().unwrap().events);
                assert_eq!(
                    assemble(&events),
                    vec![UnifiedEvent::Text { text: prose.into() }],
                    "{family} no-tools={} {prose:?}",
                    offered_tools.is_empty()
                );
            }
        }
        let prefix = r#"{"message":"ordinary JSON prose"} [] "#;
        let (head, tail) = parts(false);
        let mut parser = create_unified_parser_for_family(family, &tools()).unwrap();
        parser
            .initialize_request(UnifiedParserInit {
                starting_state: UnifiedParserStartingState::Response,
                tool_output_mode: mode(false),
                invalid_guided_payload: InvalidGuidedPayloadPolicy::StreamBestEffort,
                ..Default::default()
            })
            .unwrap();
        let mut events = Vec::new();
        for character in prefix.chars() {
            events.extend(parser.push(&character.to_string()).unwrap());
        }
        events.extend(parser.push(head).unwrap());
        let middle = parser.push("aaaa").unwrap();
        assert!(middle.iter().any(|event| matches!(event, UnifiedParserEvent::ToolCall(call) if call.arguments.contains("aaaa"))), "{family}");
        events.extend(middle);
        events.extend(parser.push(tail).unwrap());
        events.extend(parser.finish().unwrap().events);
        assert_eq!(
            assemble(&events),
            vec![
                UnifiedEvent::Text {
                    text: prefix.into()
                },
                UnifiedEvent::ToolCall {
                    name: "write_file".into(),
                    arguments: serde_json::json!({"path":"x","content":"prefix <think>aaaa suffix"})
                }
            ],
            "{family}"
        );
    }
}

#[test]
fn all_starting_channels_keep_long_field_progress_and_argument_order() {
    for family in dynamo_parsers_v2::REGISTERED_UNIFIED_FAMILIES {
        let closer = match dynamo_parsers_v2::canonical_unified_family(family).unwrap() {
            "gemma4" => "<channel|>",
            "gpt_oss" => "<|end|>",
            "kimi_k3" => "<|close|>think<|sep|>",
            "muse_glimmer" => "<|eom|>",
            _ => "</think>",
        };
        for named in [false, true] {
            for state in [
                UnifiedParserStartingState::None,
                UnifiedParserStartingState::Response,
                UnifiedParserStartingState::Reasoning,
            ] {
                let starts_in_reasoning = state == UnifiedParserStartingState::Reasoning
                    || (state == UnifiedParserStartingState::None
                        && dynamo_parsers_v2::canonical_unified_family(family)
                            == Some("nemotron_v3"));
                for (args_head, args_tail) in ["", " ", "x"].into_iter().flat_map(|preceding| {
                    let path = serde_json::to_string(preceding).unwrap();
                    [
                        (
                            r#"{"content":"prefix <think>"#.to_string(),
                            format!(r#" suffix","path":{path},"extra":""}}"#),
                        ),
                        (
                            format!(r#"{{"path":{path},"content":"prefix <think>"#),
                            r#" suffix","extra":""}"#.to_string(),
                        ),
                        (
                            format!(r#"{{"path":{path},"extra":"","content":"prefix <think>"#),
                            r#" suffix"}"#.to_string(),
                        ),
                    ]
                }) {
                    let mut parser = create_unified_parser_for_family(family, &tools()).unwrap();
                    parser
                        .initialize_request(UnifiedParserInit {
                            starting_state: state,
                            tool_output_mode: mode(named),
                            invalid_guided_payload: InvalidGuidedPayloadPolicy::StreamBestEffort,
                            ..Default::default()
                        })
                        .unwrap();
                    let mut head = if starts_in_reasoning {
                        format!("plan{closer}")
                    } else {
                        String::new()
                    };
                    if !named {
                        head.push_str(r#"[{"name":"write_file","arguments":"#);
                    }
                    head.push_str(&args_head);
                    let mut events = parser.push(&head).unwrap();
                    let body = "a".repeat(8192);
                    for fragment in body.as_bytes().chunks(128) {
                        let middle = parser.push(std::str::from_utf8(fragment).unwrap()).unwrap();
                        assert!(middle.iter().any(|event| matches!(event, UnifiedParserEvent::ToolCall(call) if call.arguments.contains("aaaa"))), "{family} {named} {state:?} head={args_head}");
                        events.extend(middle);
                    }
                    let mut tail = args_tail.to_string();
                    if !named {
                        tail.push_str("}]");
                    }
                    events.extend(parser.push(&tail).unwrap());
                    events.extend(parser.finish().unwrap().events);
                    let mut expected = Vec::new();
                    if starts_in_reasoning {
                        expected.push(UnifiedEvent::Reasoning {
                            text: "plan".into(),
                        });
                    }
                    expected.push(UnifiedEvent::ToolCall {
                        name: "write_file".into(),
                        arguments: serde_json::from_str(&format!("{args_head}{body}{args_tail}"))
                            .unwrap(),
                    });
                    assert_eq!(assemble(&events), expected, "{family} {named} {state:?}");
                    assert_eq!(events.iter().filter(|event| matches!(event, UnifiedParserEvent::ToolCall(call) if call.complete)).count(), 1);
                }
            }
        }
    }
}

#[test]
fn visible_prose_preserves_quoted_native_markers_without_changing_routing() {
    for starting_state in [
        UnifiedParserStartingState::None,
        UnifiedParserStartingState::Response,
    ] {
        for named in [false, true] {
            for quote in ['"', '\'', '`'] {
                let prefix = format!("The literal {quote}<tool_call>{quote} marker is quoted. ");
                let arguments = r#"{"path":"x","content":"ok"}"#;
                let payload = if named {
                    arguments.to_string()
                } else {
                    format!(r#"[{{"name":"write_file","arguments":{arguments}}}]"#)
                };
                let input = format!("{prefix}{payload}");
                let expected = if starting_state == UnifiedParserStartingState::Response {
                    vec![
                        UnifiedEvent::Text {
                            text: prefix.clone(),
                        },
                        UnifiedEvent::ToolCall {
                            name: "write_file".into(),
                            arguments: serde_json::json!({"path":"x", "content":"ok"}),
                        },
                    ]
                } else {
                    vec![UnifiedEvent::Text {
                        text: input.clone(),
                    }]
                };
                for split in 0..=input.len() {
                    let mut parser = create_unified_parser_for_family("qwen3", &tools()).unwrap();
                    parser
                        .initialize_request(UnifiedParserInit {
                            starting_state,
                            tool_output_mode: mode(named),
                            invalid_guided_payload: InvalidGuidedPayloadPolicy::StreamBestEffort,
                            ..Default::default()
                        })
                        .unwrap();
                    let mut events = parser.push(&input[..split]).unwrap();
                    events.extend(parser.push(&input[split..]).unwrap());
                    events.extend(parser.finish().unwrap().events);
                    assert_eq!(
                        assemble(&events),
                        expected,
                        "{starting_state:?} named={named} quote={quote} split={split}"
                    );
                }
            }
        }
    }
}

#[test]
fn reject_quoted_response_prefix_waits_for_payload_validation() {
    for family in dynamo_parsers_v2::REGISTERED_UNIFIED_FAMILIES {
        for named in [false, true] {
            for quote in ['"', '\'', '`'] {
                let prefix = format!("The literal {quote}<tool_call>{quote} marker. ");
                let (head, tail) = parts(named);
                let invalid = format!("{prefix}{head}");
                for split in 0..=invalid.len() {
                    let mut parser = create_unified_parser_for_family(family, &tools()).unwrap();
                    parser
                        .initialize_request(UnifiedParserInit {
                            starting_state: UnifiedParserStartingState::Response,
                            tool_output_mode: mode(named),
                            invalid_guided_payload: InvalidGuidedPayloadPolicy::Reject,
                            ..Default::default()
                        })
                        .unwrap();
                    assert!(
                        parser.push(&invalid[..split]).unwrap().is_empty(),
                        "{family} named={named} quote={quote} split={split}"
                    );
                    assert!(
                        parser.push(&invalid[split..]).unwrap().is_empty(),
                        "{family} named={named} quote={quote} split={split}"
                    );
                    assert!(parser.finish().is_err());
                }
                let mut parser = create_unified_parser_for_family(family, &tools()).unwrap();
                parser
                    .initialize_request(UnifiedParserInit {
                        starting_state: UnifiedParserStartingState::Response,
                        tool_output_mode: mode(named),
                        invalid_guided_payload: InvalidGuidedPayloadPolicy::Reject,
                        ..Default::default()
                    })
                    .unwrap();
                assert!(parser.push(&prefix).unwrap().is_empty());
                assert_eq!(parser.reset(), prefix, "{family} buffered prose recovery");
                parser
                    .initialize_request(UnifiedParserInit {
                        starting_state: UnifiedParserStartingState::Response,
                        tool_output_mode: mode(named),
                        invalid_guided_payload: InvalidGuidedPayloadPolicy::Reject,
                        ..Default::default()
                    })
                    .unwrap();
                for character in invalid.chars() {
                    assert!(parser.push(&character.to_string()).unwrap().is_empty());
                }
                let mut events = parser.push(tail).unwrap();
                events.extend(parser.finish().unwrap().events);
                assert_eq!(
                    assemble(&events),
                    vec![
                        UnifiedEvent::Text { text: prefix },
                        UnifiedEvent::ToolCall {
                            name: "write_file".into(),
                            arguments: serde_json::json!({"path":"x", "content":"prefix <think> suffix"})
                        }
                    ],
                    "{family} named={named} quote={quote}"
                );
            }
        }
    }
}

#[test]
fn native_response_qwen_streams_long_quoted_second_parameter_at_every_split() {
    for head in [
        r#"<tool_call><function=write_file><parameter=path>x</parameter><parameter=content>"prefix <think>"#,
        r#"<tool_call><function=write_file><parameter=path>x</parameter><parameter=content>"prefix é \"quoted\" <think>"#,
    ] {
        let body = "a".repeat(8192);
        let tail = r#" suffix"</parameter></function></tool_call>"#;
        let input = format!("{head}{body}{tail}");
        // A completed native invocation goes directly through the shared batch
        // XML value typer; compare it with incremental unified argument output.
        let mut batch = Qwen3CoderToolStreamParser::new(&tools());
        let mut batch_result = batch.push(&input).unwrap();
        batch_result.append(batch.finish().unwrap());
        let batch_result = batch_result.coalesce_calls();
        assert_eq!(batch_result.calls.len(), 1);
        let expected = vec![UnifiedEvent::ToolCall {
            name: "write_file".into(),
            arguments: serde_json::from_str(&batch_result.calls[0].arguments).unwrap(),
        }];
        for split in (0..=head.len()).filter(|at| head.is_char_boundary(*at)) {
            for fragment_size in [8192, 128] {
                let mut parser = create_unified_parser_for_family("qwen3", &tools()).unwrap();
                parser
                    .initialize_request(UnifiedParserInit {
                        starting_state: UnifiedParserStartingState::Response,
                        tool_output_mode: UnifiedToolOutputMode::Native,
                        ..Default::default()
                    })
                    .unwrap();
                let mut events = parser.push(&head[..split]).unwrap();
                events.extend(parser.push(&head[split..]).unwrap());
                for fragment in body.as_bytes().chunks(fragment_size) {
                    let middle = parser.push(std::str::from_utf8(fragment).unwrap()).unwrap();
                    assert!(middle.iter().any(|event| matches!(event, UnifiedParserEvent::ToolCall(call) if call.arguments.contains("aaaa"))), "native Response split={split} fragment_size={fragment_size}");
                    events.extend(middle);
                }
                events.extend(parser.push(tail).unwrap());
                events.extend(parser.finish().unwrap().events);
                assert_eq!(
                    assemble(&events),
                    expected,
                    "native Response split={split} fragment_size={fragment_size}"
                );
            }
        }
    }
}
