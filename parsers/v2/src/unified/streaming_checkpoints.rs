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

// The conformance table compares final assembly. These checkpoints instead reject
// output that arrives before ownership is settled or only after a later push/EOF.
fn assert_checkpoints(
    family: &str,
    init: UnifiedParserInit,
    phases: &[Phase],
    eof: &[UnifiedEvent],
    eof_ids: &[&str],
) {
    let tools = [Tool {
        name: "echo".into(),
        description: None,
        strict: None,
        parameters: serde_json::json!({"type":"object","properties":{"value":{"type":"string"}}}),
    }];
    let mut schedules = vec![None, Some((usize::MAX, 0))];
    for (index, phase) in phases.iter().enumerate() {
        schedules.extend(
            (0..=phase.input.len())
                .filter(|at| phase.input.is_char_boundary(*at))
                .map(|at| Some((index, at))),
        );
    }
    for schedule in schedules {
        let mut parser = create_unified_parser_for_family(family, &tools).unwrap();
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
        assert_checkpoints(family, UnifiedParserInit::default(), &phases, &[], &[]);
    }
}

fn k2_call(index: usize) -> String {
    format!(
        "<|tool_call_begin|>functions.echo:{index}<|tool_call_argument_begin|>{{\"value\":\"é\"}}<|tool_call_end|>"
    )
}

fn k3_call(index: usize, value: &str) -> String {
    format!(
        "<|open|>call tool=\"echo\" index=\"{index}\"<|sep|><|open|>argument key=\"value\" type=\"string\"<|sep|>{value}<|close|>argument<|sep|><|close|>call<|sep|>"
    )
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
    assert_checkpoints("kimi_k2", UnifiedParserInit::default(), &phases, &[], &[]);
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
    assert_checkpoints("kimi_k3", UnifiedParserInit::default(), &phases, &[], &[]);
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
    assert_checkpoints("kimi_k3", UnifiedParserInit::default(), &phases, &[], &[]);
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
            assert_checkpoints(family, init, &[checkpoint], &[], &[]);
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
    assert_checkpoints("kimi_k3", UnifiedParserInit::default(), &phases, &[], &[]);
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
        assert_checkpoints("kimi_k3", init, &phases, &[], &[]);
    }
}
