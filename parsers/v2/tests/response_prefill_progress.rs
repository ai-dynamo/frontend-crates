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
                    let mut head = if state == UnifiedParserStartingState::Reasoning {
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
                    if state == UnifiedParserStartingState::Reasoning {
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
