// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Gemma 4 guided-mode behavior through the public unified-parser factory.

use dynamo_parsers_v2::{
    InvalidGuidedPayloadPolicy, Tool, UnifiedEvent, UnifiedParserExt, UnifiedParserInit,
    UnifiedParserStartingState, UnifiedToolOutputMode, assemble,
};

fn weather_tools() -> Vec<Tool> {
    vec![Tool {
        name: "get_weather".into(),
        description: None,
        parameters: serde_json::json!({
            "type": "object",
            "properties": { "city": { "type": "string" } }
        }),
        strict: None,
    }]
}

fn assert_guided_at_every_split(
    input: &str,
    mode: UnifiedToolOutputMode,
    expected: &[UnifiedEvent],
) {
    assert_initialized_at_every_split(
        input,
        UnifiedParserInit {
            starting_state: UnifiedParserStartingState::None,
            tool_output_mode: mode,
            invalid_guided_payload: InvalidGuidedPayloadPolicy::RecoverAsText,
            ..UnifiedParserInit::default()
        },
        expected,
    );
}

fn assert_initialized_at_every_split(
    input: &str,
    init: UnifiedParserInit,
    expected: &[UnifiedEvent],
) {
    let tools = weather_tools();
    let mut split_points = vec![None];
    split_points.extend(
        (1..input.len())
            .filter(|&split| input.is_char_boundary(split))
            .map(Some),
    );
    let mut chunkings: Vec<Vec<&str>> = split_points
        .into_iter()
        .map(|split| split.map_or_else(|| vec![input], |at| vec![&input[..at], &input[at..]]))
        .collect();
    chunkings.push(
        input
            .char_indices()
            .map(|(at, ch)| &input[at..at + ch.len_utf8()])
            .collect(),
    );
    for chunks in chunkings {
        let mut parser = dynamo_parsers_v2::create_unified_parser_for_family("gemma4", &tools)
            .expect("built-in Gemma 4 parser");
        parser
            .initialize_request(init.clone())
            .expect("guided request");

        let mut deltas = Vec::new();
        for chunk in &chunks {
            deltas.extend(parser.push(chunk).expect("push"));
        }
        deltas.extend(parser.finish().expect("finish").events);
        assert_eq!(assemble(&deltas), expected, "chunks {chunks:?}");
    }
}

// These family-specific malformed headers preserve the authored corpus and
// exercise control-token quotation through the public initialization boundary.
#[test]
fn rejected_native_header_cannot_quote_a_stripped_control_token() {
    let input = "call:é{value:<|\"|>Café<|\"|>}<tool_call|>";
    for state in [
        UnifiedParserStartingState::None,
        UnifiedParserStartingState::Reasoning,
    ] {
        for named_tool in [None, Some("echo".to_owned())] {
            let text = "call:é{value:Café}".to_owned();
            let expected = if state == UnifiedParserStartingState::Reasoning {
                UnifiedEvent::Reasoning { text }
            } else {
                UnifiedEvent::Text { text }
            };
            assert_initialized_at_every_split(
                input,
                UnifiedParserInit {
                    starting_state: state,
                    tool_output_mode: UnifiedToolOutputMode::GuidedJson { named_tool },
                    invalid_guided_payload: InvalidGuidedPayloadPolicy::RecoverAsText,
                    ..Default::default()
                },
                &[expected],
            );
        }
    }
}

#[test]
fn guided_bare_call_admission_keeps_the_raw_identifier_predecessor() {
    for preceding in ["abc", "0", "_", "-", "."] {
        let input = format!("{preceding}call:echo{{value:<|\"|>Café<|\"|>}}<tool_call|>");
        for state in [
            UnifiedParserStartingState::None,
            UnifiedParserStartingState::Reasoning,
            UnifiedParserStartingState::Response,
        ] {
            for named_tool in [None, Some("echo".to_owned())] {
                let text = format!("{preceding}call:echo{{value:Café}}");
                let expected = if state == UnifiedParserStartingState::Reasoning {
                    UnifiedEvent::Reasoning { text }
                } else {
                    UnifiedEvent::Text { text }
                };
                assert_initialized_at_every_split(
                    &input,
                    UnifiedParserInit {
                        starting_state: state,
                        tool_output_mode: UnifiedToolOutputMode::GuidedJson { named_tool },
                        invalid_guided_payload: InvalidGuidedPayloadPolicy::RecoverAsText,
                        ..Default::default()
                    },
                    &[expected],
                );
            }
        }
    }
}

#[test]
fn embedded_prefix_cannot_retain_a_stale_header_across_utf8_json() {
    let payload = "[{\"name\":\"echo\",\"arguments\":{\"value\":\"Café\"}}]";
    for preceding in ["abc", "0", "_", "-", "."] {
        let input = format!("{preceding}call:echo{payload}");
        for state in [
            UnifiedParserStartingState::None,
            UnifiedParserStartingState::Reasoning,
        ] {
            for named_tool in [None, Some("echo".to_owned())] {
                let expected = if state == UnifiedParserStartingState::Reasoning {
                    UnifiedEvent::Reasoning {
                        text: input.clone(),
                    }
                } else {
                    UnifiedEvent::Text {
                        text: input.clone(),
                    }
                };
                assert_initialized_at_every_split(
                    &input,
                    UnifiedParserInit {
                        starting_state: state,
                        tool_output_mode: UnifiedToolOutputMode::GuidedJson { named_tool },
                        invalid_guided_payload: InvalidGuidedPayloadPolicy::RecoverAsText,
                        ..Default::default()
                    },
                    &[expected],
                );
            }
        }
    }
}

#[test]
fn response_wrapper_supplies_the_predecessor_for_its_native_header() {
    for preceding in ["h", "hello ", "é ", "abc "] {
        let input = format!("{preceding}<|tool_call>call:echo{{\"value\":\"Café\"}}");
        assert_initialized_at_every_split(
            &input,
            UnifiedParserInit {
                starting_state: UnifiedParserStartingState::Response,
                tool_output_mode: UnifiedToolOutputMode::GuidedJson {
                    named_tool: Some("echo".into()),
                },
                invalid_guided_payload: InvalidGuidedPayloadPolicy::RecoverAsText,
                ..Default::default()
            },
            &[
                UnifiedEvent::Text {
                    text: preceding.into(),
                },
                UnifiedEvent::ToolCall {
                    name: "echo".into(),
                    arguments: serde_json::json!({"value": "Café"}),
                },
            ],
        );
    }
}

#[test]
fn leading_response_whitespace_does_not_become_prefix_narration() {
    for before in [" ", "\n", "\t", " \n ", " \n <|tool_call>"] {
        for named_tool in [None, Some("echo".to_owned())] {
            let payload = if named_tool.is_some() {
                "{\"value\":\"Café\"}"
            } else {
                "[{\"name\":\"echo\",\"arguments\":{\"value\":\"Café\"}}]"
            };
            // Object payloads follow `call:` directly; a name before `[` is the
            // accepted guided envelope spelling, distinct from native `{` bodies.
            let prefix = if named_tool.is_some() {
                "call:"
            } else {
                "call:echo"
            };
            let input = format!("{before}{prefix}{payload}");
            assert_initialized_at_every_split(
                &input,
                UnifiedParserInit {
                    starting_state: UnifiedParserStartingState::Response,
                    tool_output_mode: UnifiedToolOutputMode::GuidedJson { named_tool },
                    invalid_guided_payload: InvalidGuidedPayloadPolicy::RecoverAsText,
                    ..Default::default()
                },
                &[UnifiedEvent::ToolCall {
                    name: "echo".into(),
                    arguments: serde_json::json!({"value": "Café"}),
                }],
            );
        }
    }
}

#[test]
fn response_prefix_keeps_emitted_prose_and_accepts_an_empty_wrapper() {
    let payload = "[{\"name\":\"echo\",\"arguments\":{\"value\":\"Café\"}}]";
    for before in [
        "",
        " ",
        "\t",
        "\n",
        " \n ",
        "hello ",
        "é ",
        "<|tool_call>",
        " \n <|tool_call>",
    ] {
        for named_tool in [None, Some("echo".to_owned())] {
            let input = format!("{before}call:echo{payload}");
            let narration = !before.trim().is_empty() && before.trim() != "<|tool_call>";
            let expected = if named_tool.is_some() {
                vec![UnifiedEvent::Text {
                    text: format!(
                        "{}{}{}",
                        if narration { before } else { "" },
                        if narration { "call:echo" } else { "" },
                        payload
                    ),
                }]
            } else {
                let mut events = Vec::new();
                if narration {
                    events.push(UnifiedEvent::Text {
                        text: format!("{before}call:echo"),
                    });
                }
                events.push(UnifiedEvent::ToolCall {
                    name: "echo".to_owned(),
                    arguments: serde_json::json!({"value": "Café"}),
                });
                events
            };
            assert_initialized_at_every_split(
                &input,
                UnifiedParserInit {
                    starting_state: UnifiedParserStartingState::Response,
                    tool_output_mode: UnifiedToolOutputMode::GuidedJson { named_tool },
                    invalid_guided_payload: InvalidGuidedPayloadPolicy::RecoverAsText,
                    ..Default::default()
                },
                &expected,
            );
        }
    }
}

fn guided_init(named_tool: Option<&str>, policy: InvalidGuidedPayloadPolicy) -> UnifiedParserInit {
    UnifiedParserInit {
        starting_state: UnifiedParserStartingState::None,
        tool_output_mode: UnifiedToolOutputMode::GuidedJson {
            named_tool: named_tool.map(str::to_owned),
        },
        invalid_guided_payload: policy,
        ..UnifiedParserInit::default()
    }
}

#[test]
fn reset_restarts_guided_prefix_and_response_prose_context() {
    for named_tool in [None, Some("echo")] {
        let init = UnifiedParserInit {
            starting_state: UnifiedParserStartingState::Response,
            ..guided_init(named_tool, InvalidGuidedPayloadPolicy::RecoverAsText)
        };
        let payload = if named_tool.is_some() {
            "{\"value\":\"Café\"}"
        } else {
            "[{\"name\":\"echo\",\"arguments\":{\"value\":\"Café\"}}]"
        };
        for finish_first in [false, true] {
            for pending in [
                "hello call:abcdefghijklmnop",
                "call:é{value:<|\"|>unfinished",
            ] {
                let mut parser = dynamo_parsers_v2::create_unified_parser_for_family("gemma4", &[])
                    .expect("parser");
                parser.initialize_request(init.clone()).expect("initialize");
                parser.push(pending).expect("partial push");
                if finish_first {
                    parser.finish().expect("previous finish");
                }
                parser.reset();
                parser.initialize_request(init.clone()).expect("reuse");
                let mut fresh = dynamo_parsers_v2::create_unified_parser_for_family("gemma4", &[])
                    .expect("fresh parser");
                fresh
                    .initialize_request(init.clone())
                    .expect("fresh initialize");
                let mut actual = Vec::new();
                let mut expected = Vec::new();
                for chunk in ["call:", "echo", payload] {
                    actual.extend(parser.push(chunk).expect("reuse push"));
                    expected.extend(fresh.push(chunk).expect("fresh push"));
                }
                actual.extend(parser.finish().expect("reuse finish").events);
                expected.extend(fresh.finish().expect("fresh finish").events);
                // Raw events retain tool indexes that `assemble` would hide.
                assert_eq!(actual, expected);
                assert_eq!(
                    assemble(&actual),
                    vec![UnifiedEvent::ToolCall {
                        name: "echo".into(),
                        arguments: serde_json::json!({"value": "Café"}),
                    }]
                );
            }
        }
    }
}

/// Gemma's lexical `call:` prefix is only structural when the grammar-aware
/// scanner accepts a call body. Guided mode must not strip the same bytes from
/// ordinary visible prose, even when the prefix is split across pushes. This
/// stays a public-factory regression instead of joining the unified corpus:
/// the corpus generator cannot add one family-specific guided input without
/// creating irrelevant rows for every guided family.
#[test]
fn preserves_ordinary_call_prefix_at_every_split() {
    let cases = [
        (
            "I will call: you tomorrow<|channel>thought\nchecking<channel|>[{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}]",
            "I will call: you tomorrow",
            "checking",
        ),
        (
            "prefix call:{\"x\":1}<|channel>thought\nwhy<channel|>[{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}]",
            "prefix call:{\"x\":1}",
            "why",
        ),
    ];
    for (input, visible, reasoning) in cases {
        let expected = vec![
            UnifiedEvent::Text {
                text: visible.into(),
            },
            UnifiedEvent::Reasoning {
                text: reasoning.into(),
            },
            UnifiedEvent::ToolCall {
                name: "get_weather".into(),
                arguments: serde_json::json!({"city": "Paris"}),
            },
        ];
        assert_guided_at_every_split(
            input,
            UnifiedToolOutputMode::GuidedJson { named_tool: None },
            &expected,
        );
    }
}

#[test]
fn consumes_valid_call_prefix_at_every_split() {
    let expected = [UnifiedEvent::ToolCall {
        name: "get_weather".into(),
        arguments: serde_json::json!({"city": "Paris"}),
    }];
    let cases = [
        (
            "call:[{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}]",
            UnifiedToolOutputMode::GuidedJson { named_tool: None },
        ),
        (
            "call:{\"city\":\"Paris\"}",
            UnifiedToolOutputMode::GuidedJson {
                named_tool: Some("get_weather".into()),
            },
        ),
    ];
    for (input, mode) in cases {
        assert_guided_at_every_split(input, mode, &expected);
    }
}

#[test]
fn strips_malformed_call_prefixes_without_losing_reasoning_at_every_split() {
    let cases = [
        (
            "call:<|channel>thought\nsecret<channel|>[{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}]",
            "secret",
        ),
        (
            "<|channel>thought\nI'll call call:get_weather<channel|>[{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}]",
            "I'll call get_weather",
        ),
    ];
    for (input, reasoning) in cases {
        let expected = vec![
            UnifiedEvent::Reasoning {
                text: reasoning.into(),
            },
            UnifiedEvent::ToolCall {
                name: "get_weather".into(),
                arguments: serde_json::json!({"city": "Paris"}),
            },
        ];
        assert_guided_at_every_split(
            input,
            UnifiedToolOutputMode::GuidedJson { named_tool: None },
            &expected,
        );
    }
}

/// A public factory parser can abandon an incomplete native-looking candidate,
/// then serve a fresh guided request without carrying scanner or cursor state
/// across requests.
#[test]
fn reset_after_partial_native_candidate_reuses_the_public_factory_for_guided_json() {
    let tools = weather_tools();
    let mut parser = dynamo_parsers_v2::create_unified_parser_for_family("gemma4", &tools)
        .expect("built-in Gemma 4 parser");
    parser
        .initialize_request(guided_init(None, InvalidGuidedPayloadPolicy::RecoverAsText))
        .expect("first guided request");

    assert!(
        parser
            .push("<|tool_call>call:get_weather{city:<|\"|>Par")
            .expect("partial native-looking candidate")
            .is_empty()
    );
    assert_eq!(
        parser.reset(),
        "call:get_weather{city:<|\"|>Par",
        "the recognized native wrapper is control markup, while its incomplete body is recovered"
    );

    parser
        .initialize_request(guided_init(
            Some("get_weather"),
            InvalidGuidedPayloadPolicy::RecoverAsText,
        ))
        .expect("fresh named guided request");
    let mut events = parser.push(r#"{"city":"Paris"}"#).expect("guided JSON");
    events.extend(parser.finish().expect("finish").events);
    assert_eq!(
        assemble(&events),
        vec![UnifiedEvent::ToolCall {
            name: "get_weather".into(),
            arguments: serde_json::json!({"city": "Paris"}),
        }]
    );
}

/// A rejected request setup must not poison a parser that is reset before a
/// valid guided request. This exercises the public lifecycle rather than the
/// generic router directly.
#[test]
fn rejected_guided_initialization_then_reset_allows_valid_initialization() {
    let tools = weather_tools();
    let mut parser = dynamo_parsers_v2::create_unified_parser_for_family("gemma4", &tools)
        .expect("built-in Gemma 4 parser");
    parser
        .initialize_request(UnifiedParserInit::default())
        .expect("native request");
    parser.push("visible").expect("start native request");

    assert!(
        parser
            .initialize_request(guided_init(None, InvalidGuidedPayloadPolicy::RecoverAsText))
            .is_err()
    );
    assert_eq!(parser.reset(), "");

    parser
        .initialize_request(guided_init(None, InvalidGuidedPayloadPolicy::RecoverAsText))
        .expect("valid guided request after rejected setup");
    let mut events = parser
        .push(r#"[{"name":"get_weather","arguments":{"city":"Paris"}}]"#)
        .expect("guided JSON");
    events.extend(parser.finish().expect("finish").events);
    assert_eq!(
        assemble(&events),
        vec![UnifiedEvent::ToolCall {
            name: "get_weather".into(),
            arguments: serde_json::json!({"city": "Paris"}),
        }]
    );
}

/// Native-looking markup is suppressed immediately. Once a real reasoning
/// channel closes, it must be emitted on that push, and guided JSON must commit
/// before finish under the explicitly streaming policy.
#[test]
fn malformed_native_envelope_then_reasoning_and_guided_json_emit_at_their_crossings() {
    let tools = weather_tools();
    let mut parser = dynamo_parsers_v2::create_unified_parser_for_family("gemma4", &tools)
        .expect("built-in Gemma 4 parser");
    parser
        .initialize_request(guided_init(
            None,
            InvalidGuidedPayloadPolicy::StreamBestEffort,
        ))
        .expect("streaming guided request");

    assert!(
        parser
            .push("<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>")
            .expect("native-looking envelope")
            .is_empty()
    );
    assert_eq!(
        parser
            .push("<|channel>thought\nchecking<channel|>")
            .expect("completed reasoning"),
        vec![dynamo_parsers_v2::UnifiedParserEvent::Reasoning(
            "checking".into()
        )],
        "reasoning must not wait for guided JSON or finish"
    );
    let json_events = parser
        .push(r#"[{"name":"get_weather","arguments":{"city":"Paris"}}]"#)
        .expect("completed guided JSON");
    assert!(
        json_events.iter().any(|event| matches!(
            event,
            dynamo_parsers_v2::UnifiedParserEvent::ToolCall(call)
                if call.name.as_deref() == Some("get_weather")
        )),
        "guided JSON must commit when its call crosses completion, before finish"
    );
    assert!(parser.finish().expect("finish").events.is_empty());
}

/// Kimi does not install Gemma's optional `call:` policy. Its existing guided
/// JSON input must still dispatch normally, proving the generic router treats an
/// absent policy as the former no-op behavior.
#[test]
fn kimi_guided_dispatch_is_unchanged_without_a_prefix_policy() {
    let tools = weather_tools();
    let mut parser = dynamo_parsers_v2::create_unified_parser_for_family("kimi_k2", &tools)
        .expect("built-in Kimi K2 parser");
    parser
        .initialize_request(UnifiedParserInit {
            starting_state: UnifiedParserStartingState::None,
            tool_output_mode: UnifiedToolOutputMode::GuidedJson { named_tool: None },
            invalid_guided_payload: InvalidGuidedPayloadPolicy::RecoverAsText,
            ..UnifiedParserInit::default()
        })
        .expect("guided request");

    let input = r#"[{"name":"get_weather","arguments":{"city":"Paris"}}]"#;
    let mut events = parser.push(input).expect("push");
    events.extend(parser.finish().expect("finish").events);
    assert_eq!(
        assemble(&events),
        vec![UnifiedEvent::ToolCall {
            name: "get_weather".into(),
            arguments: serde_json::json!({"city": "Paris"}),
        }]
    );
}
