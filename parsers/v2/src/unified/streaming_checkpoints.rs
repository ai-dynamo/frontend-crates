// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

struct Phase {
    input: String,
    events: Vec<UnifiedEvent>,
    ids: Vec<Option<String>>,
    eager_text: bool,
}

fn text(value: &str) -> UnifiedEvent {
    UnifiedEvent::Text { text: value.into() }
}

fn reasoning(value: &str) -> UnifiedEvent {
    UnifiedEvent::Reasoning { text: value.into() }
}

fn call(value: &str) -> UnifiedEvent {
    UnifiedEvent::ToolCall {
        name: "echo".into(),
        arguments: serde_json::json!({"value": value}),
    }
}

fn phase(input: impl Into<String>, events: Vec<UnifiedEvent>, ids: &[&str]) -> Phase {
    Phase {
        input: input.into(),
        events,
        ids: ids.iter().map(|id| Some((*id).into())).collect(),
        eager_text: false,
    }
}

fn safe_text(input: &str, events: Vec<UnifiedEvent>) -> Phase {
    Phase {
        eager_text: true,
        ..phase(input, events, &[])
    }
}

fn echo_tools() -> [Tool; 1] {
    [Tool {
        name: "echo".into(),
        description: None,
        strict: None,
        parameters: serde_json::json!({"type":"object","properties":{"value":{"type":"string"}}}),
    }]
}

// The conformance table compares final assembly. These checkpoints instead reject
// output that arrives before ownership is settled or only after a later push/EOF.
fn assert_checkpoints(
    family: &str,
    init: UnifiedParserInit,
    tools: &[Tool],
    phases: &[Phase],
    eof: &[UnifiedEvent],
    eof_ids: &[&str],
) {
    let mut schedules = vec![None, Some((usize::MAX, 0))];
    for (index, phase) in phases.iter().enumerate() {
        schedules.extend(
            (0..=phase.input.len())
                .filter(|at| phase.input.is_char_boundary(*at))
                .map(|at| Some((index, at))),
        );
    }
    for schedule in schedules {
        let mut parser = create_unified_parser_for_family(family, tools).unwrap();
        for reuse in 0..2 {
            parser.initialize_request(init.clone()).unwrap();
            let mut events = Vec::new();
            for (index, phase) in phases.iter().enumerate() {
                let input = phase.input.as_str();
                let chunks = match schedule {
                    Some((usize::MAX, _)) => input
                        .char_indices()
                        .map(|(at, ch)| &input[at..at + ch.len_utf8()])
                        .collect(),
                    Some((target, at)) if target == index => vec![&input[..at], &input[at..]],
                    _ => vec![input],
                };
                let mut consumed = 0;
                for chunk in chunks {
                    events.extend(parser.push(chunk).unwrap());
                    consumed += chunk.len();
                    let got = assemble(&events);
                    if phase.eager_text {
                        let mut expected = phase.events.clone();
                        let last = expected.last_mut().unwrap();
                        let (UnifiedEvent::Text { text } | UnifiedEvent::Reasoning { text }) = last
                        else {
                            panic!("safe-text phase needs text/reasoning")
                        };
                        text.truncate(text.len() - input.len() + consumed);
                        if text.is_empty() {
                            expected.pop();
                        }
                        assert_eq!(
                            got, expected,
                            "safe content delayed: {family} phase={index} {schedule:?}"
                        );
                    }
                    assert!(
                        got.len() <= phase.events.len(),
                        "premature events: {family} phase={index} {schedule:?}: {got:?}"
                    );
                    for (got, want) in got.iter().zip(&phase.events) {
                        match (got, want) {
                            (
                                UnifiedEvent::Text { text: got },
                                UnifiedEvent::Text { text: want },
                            )
                            | (
                                UnifiedEvent::Reasoning { text: got },
                                UnifiedEvent::Reasoning { text: want },
                            ) => assert!(
                                want.starts_with(got),
                                "{family} phase={index}: {got:?} vs {want:?}"
                            ),
                            _ => assert_eq!(
                                got, want,
                                "premature/corrupt call: {family} phase={index} {schedule:?}"
                            ),
                        }
                    }
                    // Do not let assembly hide incomplete, duplicate, or ghost deltas.
                    let calls: Vec<_> = events
                        .iter()
                        .filter_map(|event| match event {
                            UnifiedParserEvent::ToolCall(delta) => Some(delta),
                            _ => None,
                        })
                        .collect();
                    assert!(
                        calls.len() <= phase.ids.len(),
                        "premature call delta: {family} phase={index} {schedule:?}"
                    );
                    for (call_index, delta) in calls.iter().enumerate() {
                        assert!(delta.complete);
                        assert_eq!(delta.tool_index, call_index);
                        assert_eq!(
                            parser.tool_call_id(call_index),
                            phase.ids[call_index].as_deref()
                        );
                    }
                    for absent in calls.len()
                        ..=phases
                            .iter()
                            .map(|phase| phase.ids.len())
                            .max()
                            .unwrap_or(0)
                            + eof_ids.len()
                    {
                        assert_eq!(
                            parser.tool_call_id(absent),
                            None,
                            "ghost ID: {family} phase={index}"
                        );
                    }
                }
                assert_eq!(
                    assemble(&events),
                    phase.events,
                    "delayed emission: {family} phase={index} {schedule:?} reuse={reuse}"
                );
                for (index, id) in phase.ids.iter().enumerate() {
                    assert_eq!(parser.tool_call_id(index), id.as_deref());
                }
            }
            let finished = parser.finish().unwrap().events;
            assert_eq!(
                assemble(&finished),
                eof,
                "unexpected EOF output: {family} {schedule:?}"
            );
            events.extend(finished);
            let final_ids: Vec<_> = phases
                .last()
                .unwrap()
                .ids
                .iter()
                .map(|id| id.as_deref())
                .chain(eof_ids.iter().copied().map(Some))
                .collect();
            let calls: Vec<_> = events
                .iter()
                .filter_map(|event| match event {
                    UnifiedParserEvent::ToolCall(delta) => Some(delta),
                    _ => None,
                })
                .collect();
            assert_eq!(
                calls.len(),
                final_ids.len(),
                "duplicate/unfinished EOF call"
            );
            for (index, (delta, id)) in calls.iter().zip(&final_ids).enumerate() {
                assert_eq!(delta.tool_index, index);
                assert!(delta.complete);
                assert_eq!(parser.tool_call_id(index), *id);
            }
            assert_eq!(parser.tool_call_id(final_ids.len()), None);
            let mut expected = phases.last().unwrap().events.clone();
            expected.extend_from_slice(eof);
            assert_eq!(assemble(&events), expected);
            assert!(parser.finish().is_err());
            assert_eq!(parser.reset(), "");
            for index in 0..=final_ids.len() {
                assert_eq!(parser.tool_call_id(index), None);
            }
            // Reset while syntax is pending as well as after a finished request.
            parser.initialize_request(init.clone()).unwrap();
            assert!(parser.push("<").unwrap().is_empty());
            parser.reset();
        }
    }
}

#[test]
fn kimi_text_reasoning_checkpoints() {
    for (family, open, close) in [
        ("kimi_k2", "<think>", "</think>"),
        ("kimi_k3", "<|open|>think<|sep|>", "<|close|>think<|sep|>"),
    ] {
        let phases = [
            phase(format!("{open}ré"), vec![reasoning("ré")], &[]),
            safe_text("界", vec![reasoning("ré界")]),
            phase("<", vec![reasoning("ré界")], &[]),
            phase("x", vec![reasoning("ré界<x")], &[]),
            phase(&close[..close.len() - 1], vec![reasoning("ré界<x")], &[]),
            phase(">", vec![reasoning("ré界<x")], &[]),
            safe_text("visible é", vec![reasoning("ré界<x"), text("visible é")]),
            phase("<", vec![reasoning("ré界<x"), text("visible é")], &[]),
            phase("x", vec![reasoning("ré界<x"), text("visible é<x")], &[]),
        ];
        assert_checkpoints(
            family,
            UnifiedParserInit::default(),
            &echo_tools(),
            &phases,
            &[],
            &[],
        );
    }
}

fn k2_call(index: usize) -> String {
    format!(
        "<|tool_call_begin|>functions.echo:{index}<|tool_call_argument_begin|>{{\"value\":\"é\"}}<|tool_call_end|>"
    )
}

fn k3_call(index: usize, value: &str) -> String {
    k3_named_call("echo", index, &k3_argument("value", "string", value))
}

#[test]
fn kimi_k2_recovery_and_repeated_call_checkpoints() {
    let phases = [
        phase(
            "<|tool_calls_section_begin|><|tool_call_begin|>call-5ec3039f-cc26-4d27-bd09-deedbe2b0c7b<|tool_call_argument_begin|>{\"value\":\"bad\"}<|tool_call_end|><|tool_calls_section_end|>",
            vec![],
            &[],
        ),
        phase(k2_call(7), vec![call("é")], &["functions.echo:7"]),
        phase(
            k2_call(9),
            vec![call("é"); 2],
            &["functions.echo:7", "functions.echo:9"],
        ),
        phase(
            k2_call(11),
            vec![call("é"); 3],
            &["functions.echo:7", "functions.echo:9", "functions.echo:11"],
        ),
    ];
    assert_checkpoints(
        "kimi_k2",
        UnifiedParserInit::default(),
        &echo_tools(),
        &phases,
        &[],
        &[],
    );
}

#[test]
fn kimi_k3_repeated_call_checkpoints() {
    let second = k3_call(9, "é");
    let third = k3_call(11, "é");
    let header_end = second.find("<|sep|>").unwrap() + "<|sep|>".len();
    let third_header_end = third.find("<|sep|>").unwrap() + "<|sep|>".len();
    let phases = [
        phase(
            format!("<|open|>tools<|sep|>{}", k3_call(7, "é")),
            vec![],
            &[],
        ),
        phase(&second[..header_end], vec![call("é")], &["echo:6"]),
        phase(&second[header_end..], vec![call("é")], &["echo:6"]),
        phase(
            &third[..third_header_end],
            vec![call("é"); 2],
            &["echo:6", "echo:8"],
        ),
        phase(
            &third[third_header_end..],
            vec![call("é"); 2],
            &["echo:6", "echo:8"],
        ),
        phase(
            "<|close|>tools<|sep|>",
            vec![call("é"); 3],
            &["echo:6", "echo:8", "echo:10"],
        ),
    ];
    assert_checkpoints(
        "kimi_k3",
        UnifiedParserInit::default(),
        &echo_tools(),
        &phases,
        &[],
        &[],
    );
}

#[test]
fn kimi_k3_embedded_call_checkpoints() {
    // Typed strings have no escape for argument/call closers. Even a complete
    // call-shaped passage can still be data; emitting it cannot be retracted.
    // Following outer structure, not the apparent closer alone, settles ownership.
    let literal = k3_call(8, "quoted").repeat(2);
    let start = "<|open|>tools<|sep|><|open|>call tool=\"echo\" index=\"1\"<|sep|><|open|>argument key=\"value\" type=\"string\"<|sep|>before";
    let value = format!("before{literal}after");
    let phases = [
        phase(start, vec![], &[]),
        phase(literal, vec![], &[]),
        phase(
            "after<|close|>argument<|sep|><|close|>call<|sep|>",
            vec![],
            &[],
        ),
        phase(k3_call(2, "é"), vec![call(&value)], &["echo:0"]),
        phase(
            "<|close|>tools<|sep|>",
            vec![call(&value), call("é")],
            &["echo:0", "echo:1"],
        ),
    ];
    assert_checkpoints(
        "kimi_k3",
        UnifiedParserInit::default(),
        &echo_tools(),
        &phases,
        &[],
        &[],
    );
}

#[test]
fn kimi_guided_tool_checkpoints() {
    for family in ["kimi_k2", "kimi_k3"] {
        for named in [false, true] {
            let init = UnifiedParserInit {
                tool_output_mode: UnifiedToolOutputMode::GuidedJson {
                    named_tool: named.then(|| "echo".into()),
                },
                ..Default::default()
            };
            let payload = if named {
                "{\"value\":\"é\"}"
            } else {
                "[{\"name\":\"echo\",\"arguments\":{\"value\":\"é\"}}]"
            };
            let mut checkpoint = phase(payload, vec![call("é")], &[]);
            checkpoint.ids.push(None);
            assert_checkpoints(family, init, &echo_tools(), &[checkpoint], &[], &[]);
        }
    }
}

#[test]
fn kimi_k3_malformed_then_valid_checkpoints() {
    let phases = [
        phase(
            "<|open|>tools<|sep|><|open|>call tool=\"bad\" index=\"1\"<|sep|>not-an-argument",
            vec![],
            &[],
        ),
        phase(k3_call(2, "é"), vec![], &[]),
        phase("<|close|>tools<|sep|>", vec![call("é")], &["echo:1"]),
    ];
    assert_checkpoints(
        "kimi_k3",
        UnifiedParserInit::default(),
        &echo_tools(),
        &phases,
        &[],
        &[],
    );
}

#[test]
fn kimi_k3_malformed_json_calls_resync_only_at_eof() {
    let malformed_object = r#"{"value":"unfinished\"#;
    let malformed_array = r#"["unfinished é"#;
    let raw_json =
        format!("<|open|>json type=\"object\"<|sep|>{malformed_object}<|close|>json<|sep|>");
    let mut malformed_bodies = vec![(raw_json, None)];
    for kind in ["object", "array"] {
        let malformed_value = if kind == "object" {
            malformed_object
        } else {
            malformed_array
        };
        for position in 0..3 {
            let mut fields = Vec::new();
            let mut arguments = serde_json::Map::new();
            for index in 0..position {
                let key = format!("before{index}");
                let value = format!("pré{index}");
                fields.push(k3_argument(&key, "string", &value));
                arguments.insert(key, serde_json::Value::String(value));
            }
            fields.push(k3_argument("payload", kind, malformed_value));
            arguments.insert(
                "payload".into(),
                serde_json::Value::String(malformed_value.into()),
            );
            for index in position..2 {
                let key = format!("after{index}");
                let value = format!("Café{index}");
                fields.push(k3_argument(&key, "string", &value));
                arguments.insert(key, serde_json::Value::String(value));
            }
            malformed_bodies.push((fields.concat(), Some(serde_json::Value::Object(arguments))));
        }
    }
    for (body, fallback) in malformed_bodies {
        for count in 1..=2 {
            let malformed = k3_named_call("bad", 1, &body);
            let valid: Vec<_> = ["é", "Café"]
                .into_iter()
                .take(count)
                .enumerate()
                .map(|(offset, value)| {
                    k3_named_call("echo", offset + 2, &k3_argument("value", "string", value))
                })
                .collect();
            let phases = [
                phase(format!("<|open|>tools<|sep|>{malformed}"), vec![], &[]),
                phase(
                    format!("{}<|close|>tools<|sep|>", valid.concat()),
                    vec![],
                    &[],
                ),
            ];
            let mut eof = Vec::new();
            let mut eof_ids = Vec::new();
            if let Some(fallback) = &fallback {
                eof.push(UnifiedEvent::ToolCall {
                    name: "bad".into(),
                    arguments: fallback.clone(),
                });
                eof_ids.push("bad:0");
            }
            for value in ["é", "Café"].into_iter().take(count) {
                eof.push(call(value));
                eof_ids.push(if value == "é" { "echo:1" } else { "echo:2" });
            }
            assert_checkpoints(
                "kimi_k3",
                UnifiedParserInit::default(),
                &echo_tools(),
                &phases,
                &eof,
                &eof_ids,
            );
        }
    }
}

#[test]
fn kimi_k3_unfinished_json_string_markers_alone_never_dispatch() {
    for suffix in [
        "<|close|>call<|sep|>",
        "<|open|>call tool=\"ghost\" index=\"8\"<|sep|>incomplete",
    ] {
        let malformed = format!(
            "<|open|>tools<|sep|><|open|>call tool=\"bad\" index=\"1\"<|sep|><|open|>json type=\"object\"<|sep|>{{\"value\":\"unfinished{suffix}"
        );
        assert_checkpoints(
            "kimi_k3",
            UnifiedParserInit::default(),
            &echo_tools(),
            &[phase(malformed, vec![], &[])],
            &[],
            &[],
        );
    }
}

#[test]
fn kimi_k3_unfinished_json_string_keeps_complete_literal_calls_inert() {
    let malformed = format!(
        "<|open|>tools<|sep|>{}<|open|>json type=\"object\"<|sep|>{{\"value\":\"unfinished{}{}<|close|>json<|sep|><|close|>call<|sep|>",
        "<|open|>call tool=\"bad\" index=\"1\"<|sep|>",
        k3_call(8, "literal"),
        k3_call(9, "literal2"),
    );
    let valid = k3_call(2, "real");
    let phases = [
        phase(malformed, vec![], &[]),
        phase(format!("{valid}<|close|>tools<|sep|>"), vec![], &[]),
    ];
    assert_checkpoints(
        "kimi_k3",
        UnifiedParserInit::default(),
        &echo_tools(),
        &phases,
        &[call("real")],
        &["echo:1"],
    );
}

#[test]
fn kimi_k3_malformed_json_recovers_calls_after_the_wrapper_boundary() {
    let malformed = "<|open|>tools<|sep|><|open|>call tool=\"bad\" index=\"1\"<|sep|><|open|>json type=\"object\"<|sep|>{\"value\":\"unfinished<|close|>json<|sep|><|close|>call<|sep|>";
    let later_calls = [
        k3_call(8, "literal"),
        k3_call(9, "literal2"),
        k3_call(2, "real"),
    ]
    .concat();
    let phases = [
        phase(malformed, vec![], &[]),
        phase(format!("{later_calls}<|close|>tools<|sep|>"), vec![], &[]),
    ];
    assert_checkpoints(
        "kimi_k3",
        UnifiedParserInit::default(),
        &echo_tools(),
        &phases,
        &[call("literal"), call("literal2"), call("real")],
        &["echo:7", "echo:8", "echo:1"],
    );
}

#[test]
fn kimi_k3_unmatched_embedded_header_waits_for_eof() {
    // An unmatched quoted header can borrow the outer closers. Only EOF permits
    // best-effort recovery; the ordinary balanced case above must progress sooner.
    let value = "before<|open|>call tool=\"quoted\" index=\"8\"<|sep|><|open|>argument key=\"value\" type=\"string\"<|sep|>unfinished";
    let phases = [
        phase(
            format!("<|open|>tools<|sep|>{}", k3_call(1, value)),
            vec![],
            &[],
        ),
        phase(
            format!("{}<|close|>tools<|sep|>", k3_call(2, "é")),
            vec![],
            &[],
        ),
    ];
    assert_checkpoints(
        "kimi_k3",
        UnifiedParserInit::default(),
        &echo_tools(),
        &phases,
        &[call(value), call("é")],
        &["echo:0", "echo:1"],
    );
}

#[test]
fn kimi_k3_open_response_checkpoints() {
    for prefilled in [false, true] {
        let init = UnifiedParserInit {
            starting_state: if prefilled {
                UnifiedParserStartingState::Response
            } else {
                UnifiedParserStartingState::None
            },
            ..Default::default()
        };
        let phases = [
            phase(
                if prefilled {
                    ""
                } else {
                    "<|open|>response<|sep|>"
                },
                vec![],
                &[],
            ),
            safe_text("visible é", vec![text("visible é")]),
            safe_text("界", vec![text("visible é界")]),
            phase("<", vec![text("visible é界")], &[]),
            phase("x", vec![text("visible é界<x")], &[]),
            phase("<|close|>response<|sep|", vec![text("visible é界<x")], &[]),
            phase(">", vec![text("visible é界<x")], &[]),
        ];
        assert_checkpoints("kimi_k3", init, &echo_tools(), &phases, &[], &[]);
    }
}

fn k3_argument(key: &str, kind: &str, value: &str) -> String {
    format!("<|open|>argument key=\"{key}\" type=\"{kind}\"<|sep|>{value}<|close|>argument<|sep|>")
}

fn k3_named_call(name: &str, index: usize, body: &str) -> String {
    format!("<|open|>call tool=\"{name}\" index=\"{index}\"<|sep|>{body}<|close|>call<|sep|>")
}

fn weather_call() -> UnifiedEvent {
    UnifiedEvent::ToolCall {
        name: "get_weather".into(),
        arguments: serde_json::json!({"city":"Paris"}),
    }
}

fn review_tools() -> Vec<Tool> {
    [
        ("search", serde_json::json!({"q":{"type":"string"}})),
        ("get_weather", serde_json::json!({"city":{"type":"string"}})),
        ("write_file", serde_json::json!({"path":{"type":"string"},"data":{"type":"object"},"content":{"type":"string"}})),
    ].into_iter().map(|(name, properties)| Tool {
        name: name.into(), description: None, strict: None,
        parameters: serde_json::json!({"type":"object","properties":properties}),
    }).collect()
}

#[test]
fn kimi_k3_prose_between_calls_resolves_at_channel_fence() {
    let first = k3_named_call("search", 1, &k3_argument("q", "string", "Paris"));
    let second = k3_named_call("get_weather", 2, &k3_argument("city", "string", "Paris"));
    let header_end = second.find("<|sep|>").unwrap() + "<|sep|>".len();
    let first_event = UnifiedEvent::ToolCall {
        name: "search".into(),
        arguments: serde_json::json!({"q":"Paris"}),
    };
    for (channel, starting_state) in [
        ("tools", UnifiedParserStartingState::None),
        ("response", UnifiedParserStartingState::None),
        ("response", UnifiedParserStartingState::Response),
        ("think", UnifiedParserStartingState::None),
        ("think", UnifiedParserStartingState::Reasoning),
    ] {
        let mut expected = vec![first_event.clone()];
        if channel != "tools" {
            expected.push(if channel == "think" {
                reasoning("between é")
            } else {
                text("between é")
            });
        }
        expected.push(weather_call());
        let start = if starting_state == UnifiedParserStartingState::None {
            format!("<|open|>{channel}<|sep|>{first}")
        } else {
            first.clone()
        };
        let phases = [
            phase(start, vec![], &[]),
            phase("between é", vec![], &[]),
            phase(&second[..header_end], vec![], &[]),
            phase(&second[header_end..], vec![], &[]),
            phase(
                format!("<|close|>{channel}<|sep|>"),
                expected.clone(),
                &["search:0", "get_weather:1"],
            ),
        ];
        assert_checkpoints(
            "kimi_k3",
            UnifiedParserInit {
                starting_state,
                tool_output_mode: UnifiedToolOutputMode::Native,
                ..Default::default()
            },
            &review_tools(),
            &phases,
            &[],
            &[],
        );
    }
}

#[test]
fn kimi_k3_call_close_survives_response_or_reasoning_prose() {
    for (starting_state, channel_close, prose) in [
        (
            UnifiedParserStartingState::Response,
            "<|close|>response<|sep|>",
            text("ordinary response prose"),
        ),
        (
            UnifiedParserStartingState::Reasoning,
            "<|close|>think<|sep|>",
            reasoning("ordinary reasoning prose"),
        ),
    ] {
        let phases = [
            phase(
                k3_named_call("echo", 1, &k3_argument("value", "string", "é")),
                vec![],
                &[],
            ),
            phase(
                match prose.clone() {
                    UnifiedEvent::Text { text } | UnifiedEvent::Reasoning { text } => text,
                    _ => unreachable!(),
                },
                vec![],
                &[],
            ),
            phase(channel_close, vec![call("é"), prose], &["echo:0"]),
        ];
        assert_checkpoints(
            "kimi_k3",
            UnifiedParserInit {
                starting_state,
                tool_output_mode: UnifiedToolOutputMode::Native,
                ..Default::default()
            },
            &echo_tools(),
            &phases,
            &[],
            &[],
        );
    }
}

#[test]
fn kimi_k3_response_string_keeps_complete_literal_call_passages() {
    let literal = k3_named_call("quoted", 8, &k3_argument("value", "string", "literal"));
    let value = format!("before{literal}after");
    let phases = [
        phase(
            k3_named_call("echo", 1, &k3_argument("value", "string", &value)),
            vec![],
            &[],
        ),
        phase("<|close|>response<|sep|>", vec![call(&value)], &["echo:0"]),
    ];
    assert_checkpoints(
        "kimi_k3",
        UnifiedParserInit {
            starting_state: UnifiedParserStartingState::Response,
            tool_output_mode: UnifiedToolOutputMode::Native,
            ..Default::default()
        },
        &echo_tools(),
        &phases,
        &[],
        &[],
    );
}

#[test]
fn kimi_k3_json_string_keeps_complete_literal_call_passages() {
    let literal = k3_named_call("quoted", 8, &k3_argument("value", "string", "literal"));
    let value = format!("before{literal}after");
    let json = serde_json::json!({"value": value}).to_string();
    let body = format!("<|open|>json type=\"object\"<|sep|>{json}<|close|>json<|sep|>");
    let phases = [
        phase(
            format!("<|open|>tools<|sep|>{}", k3_named_call("echo", 1, &body)),
            vec![],
            &[],
        ),
        phase("<|close|>tools<|sep|>", vec![call(&value)], &["echo:0"]),
    ];
    assert_checkpoints(
        "kimi_k3",
        UnifiedParserInit::default(),
        &echo_tools(),
        &phases,
        &[],
        &[],
    );
}

#[test]
fn kimi_k3_json_markers_and_nested_calls_keep_payload_ownership() {
    let marker = "<|close|>call<|sep|><|open|>call<|sep|>";
    let pair = "<|close|>argument<|sep|><|close|>call<|sep|>";
    let second = k3_named_call("get_weather", 2, &k3_argument("city", "string", "Paris"));
    for data in [
        serde_json::json!({"text":marker}),
        serde_json::json!({"text":format!("before{pair}after")}),
        serde_json::json!({"text":format!("quote \\\" é {marker}")}),
    ] {
        let raw = serde_json::to_string(&data).unwrap();
        for body in [
            k3_argument("data", "object", &raw),
            format!(
                "{}{}",
                k3_argument("path", "string", "a.json"),
                k3_argument("data", "object", &raw)
            ),
        ] {
            let first = k3_named_call("write_file", 1, &body);
            let arguments = if body.contains("key=\"path\"") {
                serde_json::json!({"path":"a.json","data":data})
            } else {
                serde_json::json!({"data":data})
            };
            let expected = vec![
                UnifiedEvent::ToolCall {
                    name: "write_file".into(),
                    arguments,
                },
                weather_call(),
            ];
            let input = format!("<|open|>tools<|sep|>{first}{second}<|close|>tools<|sep|>");
            assert_checkpoints(
                "kimi_k3",
                UnifiedParserInit {
                    tool_output_mode: UnifiedToolOutputMode::Native,
                    ..Default::default()
                },
                &review_tools(),
                &[phase(
                    &input,
                    expected.clone(),
                    &["write_file:0", "get_weather:1"],
                )],
                &[],
                &[],
            );
        }
        let nested = k3_named_call("write_file", 8, &k3_argument("data", "object", &raw));
        for count in 1..=2 {
            let value = format!("before{}after", nested.repeat(count));
            let input = format!(
                "<|open|>tools<|sep|>{}{}<|close|>tools<|sep|>",
                k3_call(1, &value),
                k3_call(2, "é")
            );
            assert_checkpoints(
                "kimi_k3",
                UnifiedParserInit {
                    tool_output_mode: UnifiedToolOutputMode::Native,
                    ..Default::default()
                },
                &echo_tools(),
                &[phase(
                    &input,
                    vec![call(&value), call("é")],
                    &["echo:0", "echo:1"],
                )],
                &[],
                &[],
            );
        }
    }
}

#[test]
fn kimi_k3_json_open_in_string_does_not_own_closing_markers() {
    for suffix in ["\"unfinished", "{\"x\":\"unfinished", "[\"unfinished"] {
        let value = format!("before<|open|>json<|sep|>{suffix}");
        let phases = [
            phase(
                format!("<|open|>tools<|sep|>{}", k3_call(1, &value)),
                vec![],
                &[],
            ),
            phase(
                format!("{}<|close|>tools<|sep|>", k3_call(2, "é")),
                vec![call(&value), call("é")],
                &["echo:0", "echo:1"],
            ),
        ];
        assert_checkpoints(
            "kimi_k3",
            UnifiedParserInit {
                tool_output_mode: UnifiedToolOutputMode::Native,
                ..Default::default()
            },
            &echo_tools(),
            &phases,
            &[],
            &[],
        );
    }
}

#[test]
fn kimi_k3_missing_call_close_recovers_before_eof_without_borrowing_next_call() {
    let first = format!(
        "<|open|>call tool=\"search\" index=\"1\"<|sep|>{}",
        k3_argument("q", "string", "Paris")
    );
    let second = k3_named_call("get_weather", 2, &k3_argument("city", "string", "Paris"));
    let phases = [
        phase(format!("<|open|>tools<|sep|>{first}"), vec![], &[]),
        phase(second, vec![], &[]),
        phase(
            "<|close|>tools<|sep|>",
            vec![
                UnifiedEvent::ToolCall {
                    name: "search".into(),
                    arguments: serde_json::json!({"q":"Paris"}),
                },
                weather_call(),
            ],
            &["search:0", "get_weather:1"],
        ),
    ];
    assert_checkpoints(
        "kimi_k3",
        UnifiedParserInit {
            tool_output_mode: UnifiedToolOutputMode::Native,
            ..Default::default()
        },
        &review_tools(),
        &phases,
        &[],
        &[],
    );
}

#[test]
fn kimi_k3_balanced_literal_calls_before_narration_remain_argument_data() {
    let literal = k3_named_call("quoted", 8, &k3_argument("value", "string", "Zurich"));
    for count in 1..=2 {
        for prefix in ["", "<|close|>argument<|sep|><|close|>call<|sep|>prose"] {
            let value = format!("{prefix}{}", literal.repeat(count));
            let phases = [
                phase(
                    format!("<|open|>tools<|sep|>{}", k3_call(1, &value)),
                    vec![],
                    &[],
                ),
                phase(" narration ", vec![], &[]),
                phase(
                    format!("{}<|close|>tools<|sep|>", k3_call(2, "é")),
                    vec![call(&value), call("é")],
                    &["echo:0", "echo:1"],
                ),
            ];
            assert_checkpoints(
                "kimi_k3",
                UnifiedParserInit::default(),
                &echo_tools(),
                &phases,
                &[],
                &[],
            );
        }
    }
}

#[test]
fn kimi_k3_guided_native_wrappers_release_json_before_eof() {
    let object = serde_json::json!({"text":"<|close|>call<|sep|><|open|>call<|sep|>"}).to_string();
    let wrappers = [
        format!("{} narration {}", k3_call(1, "Paris"), k3_call(2, "é")),
        k3_named_call("write_file", 1, &k3_argument("data", "object", &object)),
        k3_call(1, "before<|open|>json<|sep|>\"unfinished"),
    ];
    for (index, wrapper) in wrappers.into_iter().enumerate() {
        for named in [false, true] {
            let payload = if named {
                "{\"value\":\"é\"}"
            } else {
                "[{\"name\":\"echo\",\"arguments\":{\"value\":\"é\"}}]"
            };
            let mut output = phase(payload, vec![call("é")], &[]);
            output.ids.push(None);
            let prefix = if index == 0 {
                vec![text(" narration ")]
            } else {
                vec![]
            };
            output.events.splice(0..0, prefix.clone());
            let phases = [
                phase(
                    format!("<|open|>tools<|sep|>{wrapper}<|close|>tools<|sep|>"),
                    prefix,
                    &[],
                ),
                output,
            ];
            assert_checkpoints(
                "kimi_k3",
                UnifiedParserInit {
                    tool_output_mode: UnifiedToolOutputMode::GuidedJson {
                        named_tool: named.then(|| "echo".into()),
                    },
                    ..Default::default()
                },
                &echo_tools(),
                &phases,
                &[],
                &[],
            );
        }
    }
}

#[test]
fn kimi_k3_guided_malformed_raw_json_recovers_later_payload_at_eof() {
    let malformed = k3_named_call(
        "bad",
        1,
        "<|open|>json type=\"object\"<|sep|>{\"value\":\"unfinished<|close|>json<|sep|>",
    );
    for named in [false, true] {
        let (payload, expected) = if named {
            ("{\"value\":\"é\"}".to_string(), vec![call("é")])
        } else {
            (
                "[{\"name\":\"echo\",\"arguments\":{\"value\":\"é\"}},{\"name\":\"echo\",\"arguments\":{\"value\":\"Café\"}}]".to_string(),
                vec![call("é"), call("Café")],
            )
        };
        let input = format!("<|open|>tools<|sep|>{malformed}<|close|>tools<|sep|>{payload}");
        let splits: Vec<_> = input
            .char_indices()
            .map(|(at, _)| at)
            .chain([input.len()])
            .collect();
        for split in splits {
            let mut parser = create_unified_parser_for_family("kimi_k3", &[]).unwrap();
            parser
                .initialize_request(UnifiedParserInit {
                    tool_output_mode: UnifiedToolOutputMode::GuidedJson {
                        named_tool: named.then(|| "echo".into()),
                    },
                    invalid_guided_payload: InvalidGuidedPayloadPolicy::RecoverAsText,
                    ..Default::default()
                })
                .unwrap();
            let mut events = parser.push(&input[..split]).unwrap();
            assert!(events.is_empty(), "premature guided dispatch before EOF");
            events.extend(parser.push(&input[split..]).unwrap());
            assert!(events.is_empty(), "premature guided dispatch before EOF");
            assert_eq!(assemble(&parser.finish().unwrap().events), expected);
            assert_eq!(parser.tool_call_id(0), None);
            assert_eq!(parser.tool_call_id(1), None);
            assert!(parser.finish().is_err());
            assert!(parser.reset().is_empty());
        }

        let mut parser = create_unified_parser_for_family("kimi_k3", &[]).unwrap();
        parser
            .initialize_request(UnifiedParserInit {
                tool_output_mode: UnifiedToolOutputMode::GuidedJson {
                    named_tool: named.then(|| "echo".into()),
                },
                invalid_guided_payload: InvalidGuidedPayloadPolicy::RecoverAsText,
                ..Default::default()
            })
            .unwrap();
        for (at, character) in input.char_indices() {
            assert!(
                parser
                    .push(&input[at..at + character.len_utf8()])
                    .unwrap()
                    .is_empty()
            );
        }
        assert_eq!(assemble(&parser.finish().unwrap().events), expected);
        assert_eq!(parser.tool_call_id(0), None);
        assert_eq!(parser.tool_call_id(1), None);
        assert!(parser.finish().is_err());
        assert!(parser.reset().is_empty());
    }
}

#[test]
fn kimi_k3_malformed_json_in_literal_call_recovers_at_eof() {
    // A quoted native passage may contain invalid JSON. Its unmatched quote
    // cannot seize the outer argument; EOF permits recovery without dispatching it.
    for body in [
        "<|open|>json<|sep|>{\"x\":\"unfinished<|close|>json<|sep|>".to_string(),
        k3_argument("data", "object", "{\"x\":\"unfinished"),
        k3_argument("data", "array", "[\"unfinished"),
    ] {
        let value = k3_named_call("quoted", 8, &body);
        let input = format!(
            "<|open|>tools<|sep|>{}{}<|close|>tools<|sep|>",
            k3_call(1, &value),
            k3_call(2, "é")
        );
        assert_checkpoints(
            "kimi_k3",
            UnifiedParserInit::default(),
            &echo_tools(),
            &[phase(input, vec![], &[])],
            &[call(&value), call("é")],
            &["echo:0", "echo:1"],
        );
    }
}
