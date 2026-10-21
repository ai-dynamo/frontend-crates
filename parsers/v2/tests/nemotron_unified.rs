// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_parsers_v2::{
    Tool, UnifiedEvent, UnifiedParserExt, assemble, create_unified_parser_for_family,
};

#[test]
fn ultra_implicit_reasoning_and_xml_call() {
    let tools = [Tool {
        name: "get_weather".into(),
        description: None,
        parameters: serde_json::json!({"type":"object","properties":{"city":{"type":"string"}}}),
        strict: None,
    }];
    let input = "Check Montréal.</think><tool_call><function=get_weather><parameter=city>Montréal</parameter></function></tool_call>";
    let mut parser = create_unified_parser_for_family("nemotron_v3", &tools).unwrap();
    parser.initialize(&[]).unwrap();
    let mut events = parser.push(input).unwrap();
    events.extend(parser.finish().unwrap().events);
    assert_eq!(
        assemble(&events),
        vec![
            UnifiedEvent::Reasoning {
                text: "Check Montréal.".into()
            },
            UnifiedEvent::ToolCall {
                name: "get_weather".into(),
                arguments: serde_json::json!({"city":"Montréal"})
            },
        ]
    );
}

use dynamo_parsers_v2::{
    InvalidGuidedPayloadPolicy, UnifiedParser, UnifiedParserInit, UnifiedParserStartingState,
    UnifiedToolOutputMode,
};

fn parser() -> Box<dyn UnifiedParser> {
    create_unified_parser_for_family("nemotron_v3", &[]).unwrap()
}

fn reason(text: &str) -> UnifiedEvent {
    UnifiedEvent::Reasoning { text: text.into() }
}

fn text(value: &str) -> UnifiedEvent {
    UnifiedEvent::Text { text: value.into() }
}

fn call(value: &str) -> UnifiedEvent {
    UnifiedEvent::ToolCall {
        name: "log".into(),
        arguments: serde_json::json!({"note": value}),
    }
}

fn native_call(value: &str) -> String {
    format!("<tool_call><function=log><parameter=note>{value}</parameter></function></tool_call>")
}

fn run(parser: &mut dyn UnifiedParser, chunks: &[&str]) -> Vec<UnifiedEvent> {
    let mut events = Vec::new();
    for chunk in chunks {
        events.extend(parser.push(chunk).unwrap());
    }
    events.extend(parser.finish().unwrap().events);
    assemble(&events)
}

#[test]
fn native_defaults_initialization_reset_and_every_unicode_split() {
    let tool = native_call("<think>雪</think> </tool_call> &amp;");
    let cases = [
        (
            format!("réfléchir</think>回答{tool}<think>suite</think>fin"),
            vec![
                reason("réfléchir"),
                text("回答"),
                call("<think>雪</think> </tool_call> &amp;"),
                reason("suite"),
                text("fin"),
            ],
        ),
        (
            format!("<think>réfléchir{tool}suite</think>fin"),
            vec![
                reason("réfléchir"),
                call("<think>雪</think> </tool_call> &amp;"),
                reason("suite"),
                text("fin"),
            ],
        ),
        (
            format!("réfléchir{tool}suite"),
            vec![
                reason("réfléchir"),
                call("<think>雪</think> </tool_call> &amp;"),
                reason("suite"),
            ],
        ),
        ("<think>雪".into(), vec![reason("雪")]),
        (
            "雪<tool_call><function=log><parameter=note>partial".into(),
            vec![reason("雪")],
        ),
        (
            format!("</think>{tool}{tool}"),
            vec![
                call("<think>雪</think> </tool_call> &amp;"),
                call("<think>雪</think> </tool_call> &amp;"),
            ],
        ),
    ];
    for (input, want) in cases {
        for state in [
            None,
            Some(UnifiedParserStartingState::None),
            Some(UnifiedParserStartingState::Reasoning),
        ] {
            for at in input.char_indices().map(|(at, _)| at).chain([input.len()]) {
                let mut p = parser();
                if let Some(starting_state) = state {
                    p.initialize_request(UnifiedParserInit {
                        starting_state,
                        ..Default::default()
                    })
                    .unwrap();
                }
                let chunks = [&input[..at], &input[at..]];
                assert_eq!(
                    run(p.as_mut(), &chunks),
                    want,
                    "state={state:?} split={at} input={input}"
                );
                assert!(p.finish().is_err());
                assert!(p.push("late").is_err());
                p.reset();
                assert_eq!(
                    run(p.as_mut(), &chunks),
                    want,
                    "reuse state={state:?} split={at}"
                );
            }
        }
        let chunks: Vec<_> = input
            .char_indices()
            .map(|(at, ch)| &input[at..at + ch.len_utf8()])
            .collect();
        assert_eq!(run(parser().as_mut(), &chunks), want);
    }
}

#[test]
fn response_disables_thinking_and_reinitialization_restores_family_default() {
    let input = format!("<think>literal 雪</think>{}", native_call("data"));
    let mut p = parser();
    p.initialize_request(UnifiedParserInit {
        starting_state: UnifiedParserStartingState::Response,
        ..Default::default()
    })
    .unwrap();
    let want = vec![text("<think>literal 雪</think>"), call("data")];
    assert_eq!(run(p.as_mut(), &[&input]), want);
    p.reset();
    assert_eq!(run(p.as_mut(), &[&input]), want);
    p.reset();
    p.initialize_request(UnifiedParserInit::default()).unwrap();
    assert_eq!(
        run(p.as_mut(), &["雪</think>answer"]),
        vec![reason("雪"), text("answer")]
    );
    assert!(p.initialize_request(UnifiedParserInit::default()).is_err());
}

#[test]
fn guided_defaults_named_required_and_reset_are_split_invariant() {
    for named in [false, true] {
        let payload = if named {
            r#"{"note":"<think>雪</think>"}"#
        } else {
            r#"[{"name":"log","arguments":{"note":"<think>雪</think>"}}]"#
        };
        for prefix in ["réfléchir</think>", "<think>réfléchir</think>"] {
            let input = format!("{prefix}{payload}");
            for at in input.char_indices().map(|(at, _)| at).chain([input.len()]) {
                for policy in [
                    InvalidGuidedPayloadPolicy::Reject,
                    InvalidGuidedPayloadPolicy::RecoverAsText,
                    InvalidGuidedPayloadPolicy::StreamBestEffort,
                ] {
                    let mut p = parser();
                    p.initialize_request(UnifiedParserInit {
                        tool_output_mode: UnifiedToolOutputMode::GuidedJson {
                            named_tool: named.then(|| "log".into()),
                        },
                        invalid_guided_payload: policy,
                        ..Default::default()
                    })
                    .unwrap();
                    let chunks = [&input[..at], &input[at..]];
                    let want = vec![reason("réfléchir"), call("<think>雪</think>")];
                    assert_eq!(
                        run(p.as_mut(), &chunks),
                        want,
                        "named={named} split={at} policy={policy:?}"
                    );
                    p.reset();
                    assert_eq!(run(p.as_mut(), &chunks), want);
                }
            }
        }
    }
}

#[test]
fn reasoning_and_calls_make_progress_before_finish() {
    let mut p = parser();
    assert_eq!(
        assemble(&p.push("réfléchir").unwrap()),
        vec![reason("réfléchir")]
    );
    assert_eq!(
        assemble(&p.push(&native_call("雪")).unwrap()),
        vec![call("雪")]
    );
    assert_eq!(
        assemble(&p.push("suite</think>answer").unwrap()),
        vec![reason("suite"), text("answer")]
    );
    assert!(p.finish().unwrap().events.is_empty());
    assert_eq!(
        dynamo_parsers_v2::canonical_unified_family("nemotron3"),
        Some("nemotron_v3")
    );
    let mut alias = create_unified_parser_for_family("nemotron3", &[]).unwrap();
    assert_eq!(
        run(alias.as_mut(), &["雪</think>answer"]),
        vec![reason("雪"), text("answer")]
    );
}

#[test]
fn guided_explicit_states_and_mode_switches_preserve_request_ownership() {
    for named in [false, true] {
        for state in [
            UnifiedParserStartingState::None,
            UnifiedParserStartingState::Reasoning,
            UnifiedParserStartingState::Response,
        ] {
            let payload = if named {
                r#"{"note":"<think>雪</think>"}"#
            } else {
                r#"[{"name":"log","arguments":{"note":"<think>雪</think>"}}]"#
            };
            let prefix = if state == UnifiedParserStartingState::Response {
                ""
            } else {
                "Montréal</think>"
            };
            let input = format!("{prefix}{payload}");
            let mut want = if prefix.is_empty() {
                vec![]
            } else {
                vec![reason("Montréal")]
            };
            want.push(call("<think>雪</think>"));
            for at in input.char_indices().map(|(at, _)| at).chain([input.len()]) {
                let mut p = parser();
                p.initialize_request(UnifiedParserInit {
                    starting_state: state,
                    tool_output_mode: UnifiedToolOutputMode::GuidedJson {
                        named_tool: named.then(|| "log".into()),
                    },
                    ..Default::default()
                })
                .unwrap();
                let chunks = [&input[..at], &input[at..]];
                assert_eq!(run(p.as_mut(), &chunks), want);
                p.reset();
                assert_eq!(run(p.as_mut(), &chunks), want);
                p.reset();
                p.initialize_request(UnifiedParserInit::default()).unwrap();
                assert_eq!(
                    run(p.as_mut(), &["Montréal</think>雪"]),
                    vec![reason("Montréal"), text("雪")]
                );
            }
        }
    }
}

#[test]
fn default_reasoning_and_response_counterfactual_use_identical_input() {
    for (input, default, response) in [
        (
            "Check Montréal.</think>雪".to_string(),
            vec![reason("Check Montréal."), text("雪")],
            vec![text("Check Montréal.</think>雪")],
        ),
        (
            format!("Check.{}Then.", native_call("<think>雪</think>")),
            vec![reason("Check."), call("<think>雪</think>"), reason("Then.")],
            vec![text("Check."), call("<think>雪</think>"), text("Then.")],
        ),
    ] {
        for at in input.char_indices().map(|(at, _)| at).chain([input.len()]) {
            let chunks = [&input[..at], &input[at..]];
            assert_eq!(run(parser().as_mut(), &chunks), default);
            let mut p = parser();
            p.initialize_request(UnifiedParserInit {
                starting_state: UnifiedParserStartingState::Response,
                ..Default::default()
            })
            .unwrap();
            assert_eq!(run(p.as_mut(), &chunks), response);
        }
    }
}
