// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[path = "support/quote_policy.rs"]
mod quote_policy;

use dynamo_parsers_v2::{
    InvalidGuidedPayloadPolicy, UnifiedEvent, UnifiedParserEvent, UnifiedParserExt,
    UnifiedParserInit, UnifiedParserOutput, UnifiedParserStartingState, UnifiedToolOutputMode,
    create_unified_parser_for_family,
};

fn call(name: &str, arguments: serde_json::Value) -> UnifiedEvent {
    UnifiedEvent::ToolCall {
        name: name.into(),
        arguments,
    }
}

fn parse(chunks: &[&str], init: UnifiedParserInit) -> UnifiedParserOutput {
    let mut parser = create_unified_parser_for_family("hermes", &[]).unwrap();
    parser.initialize_request(init).unwrap();
    let mut output = UnifiedParserOutput::default();
    for chunk in chunks {
        parser.parse_into(chunk, &mut output).unwrap();
    }
    output.append(&mut parser.finish().unwrap());
    output
}

fn every_split(input: &str, init: UnifiedParserInit, expected: &[UnifiedEvent]) {
    for at in (0..=input.len()).filter(|&at| input.is_char_boundary(at)) {
        assert_eq!(
            parse(&[&input[..at], &input[at..]], init.clone()).assembled(),
            expected,
            "split {at}: {input}"
        );
    }
    let chunks: Vec<_> = input
        .char_indices()
        .map(|(at, ch)| &input[at..at + ch.len_utf8()])
        .collect();
    assert_eq!(
        parse(&chunks, init).assembled(),
        expected,
        "character cadence"
    );
}

#[test]
fn native_boundaries_and_recovery() {
    let weather = r#"<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>"#;
    for (input, expected) in [
        (weather.into(), vec![call("get_weather", serde_json::json!({"city":"Paris"}))]),
        (format!("He said \"maybe{weather}"),vec![UnifiedEvent::Text{text:"He said \"maybe".into()},call("get_weather",serde_json::json!({"city":"Paris"}))]),
        (format!("before{weather}after{weather}"), vec![UnifiedEvent::Text{text:"before".into()},call("get_weather",serde_json::json!({"city":"Paris"})),UnifiedEvent::Text{text:"after".into()},call("get_weather",serde_json::json!({"city":"Paris"}))]),
        (format!("<think>check{weather}again</think>done"), vec![UnifiedEvent::Reasoning{text:"check".into()},call("get_weather",serde_json::json!({"city":"Paris"})),UnifiedEvent::Reasoning{text:"again".into()},UnifiedEvent::Text{text:"done".into()}]),
        (r#"<tool_call>{"name":"f","arguments":{"x":"雪</tool_call><tool_call><think>é\\\""}}</tool_call>"#.into(),vec![call("f",serde_json::json!({"x":"雪</tool_call><tool_call><think>é\\\""}))]),
        (r#"<tool_call>{"name":"f"}</tool_call>"#.into(),vec![call("f",serde_json::json!({}))]),
        (r#"<tool_call>{"name":"f","parameters":{"x":null}}"#.into(),vec![call("f",serde_json::json!({"x":null}))]),
        (r#"[1,2,{"name":"f","arguments":{}}]"#.into(),vec![UnifiedEvent::Text{text:r#"[1,2,{"name":"f","arguments":{}}]"#.into()}]),
        (format!("<tool_call>{{\"name\":\"bad\",\"arguments\":{{\"x\": }} }}</tool_call>{weather}"),vec![call("get_weather",serde_json::json!({"city":"Paris"}))]),
        (format!("<tool_call>{{\"name\":\"bad\",\"arguments\":{{{weather}"),vec![call("get_weather",serde_json::json!({"city":"Paris"}))]),
        ("before</tool_call>after".into(),vec![UnifiedEvent::Text{text:"beforeafter".into()}]),
        ("<tool_call>{\"name\":\"f\",\"arguments\":{\"x\":\"unfinished".into(),vec![]),
    ] {
        every_split(&input, UnifiedParserInit::default(), &expected);
    }
}

#[test]
fn native_quoted_markers_do_not_hide_a_following_call() {
    if !quote_policy::quote_aware_process("native_quoted_markers_do_not_hide_a_following_call") {
        return;
    }
    let weather = r#"<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>"#;
    for prose in [
        "The literal \"<tool_call>\" marker is part of the explanation.".to_string(),
        "The literal \"<tool_call> \n\t\" marker is part of the explanation.".to_string(),
        format!("The literal `{weather}` stays visible."),
        "The literal '<tool_call>{ example }</tool_call>' stays visible.".to_string(),
    ] {
        every_split(
            &format!("{prose}{weather}"),
            UnifiedParserInit {
                starting_state: UnifiedParserStartingState::Response,
                ..Default::default()
            },
            &[
                UnifiedEvent::Text { text: prose },
                call("get_weather", serde_json::json!({"city":"Paris"})),
            ],
        );
    }
    every_split("<tool_call>", UnifiedParserInit::default(), &[]);
}

fn assert_native_recovery(chunks: &[&str], expected: &[UnifiedEvent]) {
    // Hermes has no legacy ToolParser registry entry; Unified owns its native
    // envelope and these recovery/argument-ownership regressions.
    let output = parse(
        chunks,
        UnifiedParserInit {
            starting_state: UnifiedParserStartingState::Response,
            ..UnifiedParserInit::default()
        },
    );
    assert_eq!(output.assembled(), expected, "{chunks:?}");
    let completions = output
        .events
        .iter()
        .filter(|event| matches!(event, UnifiedParserEvent::ToolCall(delta) if delta.complete))
        .count();
    let expected_completions = expected
        .iter()
        .filter(|event| matches!(event, UnifiedEvent::ToolCall { .. }))
        .count();
    assert_eq!(completions, expected_completions, "{chunks:?}");
    assert!(output.events.iter().all(|event| {
        !matches!(event, UnifiedParserEvent::ToolCall(delta) if delta.name.as_deref() == Some("fake"))
    }));
}

fn every_native_recovery_delivery(input: &str, expected: &[UnifiedEvent]) {
    assert_native_recovery(&[input], expected);
    for at in (0..=input.len()).filter(|at| input.is_char_boundary(*at)) {
        assert_native_recovery(&[&input[..at], &input[at..]], expected);
    }
    let characters: Vec<_> = input
        .char_indices()
        .map(|(at, ch)| &input[at..at + ch.len_utf8()])
        .collect();
    assert_native_recovery(&characters, expected);
}

#[test]
fn native_malformed_quote_roots_preserve_the_next_call_and_visible_suffix() {
    let weather = r#"{"name":"get_weather","arguments":{"city":"Paris"}}"#;
    let chunks = [
        "The literal \"",
        "<tool_call>",
        "\" marker is part of the explanation.",
        "<tool_call>",
        weather,
        "</tool_call>",
    ];
    let expected = [
        UnifiedEvent::Text {
            text: "The literal \"".into(),
        },
        call("get_weather", serde_json::json!({"city":"Paris"})),
    ];
    assert_native_recovery(&chunks, &expected);
    every_native_recovery_delivery(&chunks.concat(), &expected);

    let quoted =
        serde_json::to_string(r#"<tool_call>{"name":"lookup","arguments":{"q":"x"}}</tool_call>"#)
            .unwrap();
    every_native_recovery_delivery(
        &format!("{quoted} trailing"),
        &[UnifiedEvent::Text {
            text: "\"\" trailing".into(),
        }],
    );
}

#[test]
fn native_valid_argument_strings_own_complete_call_markers() {
    let value = r#"雪 <tool_call>{"name":"fake","arguments":{}}</tool_call> suffix"#;
    let body = serde_json::json!({"name":"lookup","arguments":{"q":value}});
    every_native_recovery_delivery(
        &format!("<tool_call>{body}</tool_call> tail"),
        &[
            call("lookup", serde_json::json!({"q":value})),
            UnifiedEvent::Text {
                text: " tail".into(),
            },
        ],
    );
}

#[test]
fn native_invalid_owned_strings_keep_markers_as_data_until_their_lexical_close() {
    let next = r#"<tool_call>{"name":"ok","arguments":{"x":"雪"}}</tool_call>"#;
    for escape in [r"\q", r"\u12Z4", r"\u12"] {
        let input = format!(
            r#"before <tool_call>{{"name":"bad","arguments":{{"q":"雪{escape} <tool_call>{{\"name\":\"fake\",\"arguments\":{{}}}}</tool_call> suffix"}}}}</tool_call> after {next} done"#
        );
        every_native_recovery_delivery(
            &input,
            &[
                UnifiedEvent::Text {
                    text: "before  after ".into(),
                },
                call("ok", serde_json::json!({"x":"雪"})),
                UnifiedEvent::Text {
                    text: " done".into(),
                },
            ],
        );
    }

    every_native_recovery_delivery(
        &format!(
            r#"<tool_call>{{"name":"lookup","arguments":{{"q":"</tool_call>"}} oops</tool_call> tail{next}"#
        ),
        &[
            UnifiedEvent::Text {
                text: " tail".into(),
            },
            call("ok", serde_json::json!({"x":"雪"})),
        ],
    );
}

#[test]
fn native_malformed_root_recovery_progresses_before_finish_and_resets() {
    let mut parser = create_unified_parser_for_family("hermes", &[]).unwrap();
    let init = UnifiedParserInit {
        starting_state: UnifiedParserStartingState::Response,
        ..UnifiedParserInit::default()
    };
    parser.initialize_request(init.clone()).unwrap();
    let mut output = UnifiedParserOutput::default();
    for chunk in [
        "The literal \"",
        "<tool_call>",
        "\" marker is part of the explanation.",
        "<tool_call>",
        r#"{"name":"get_weather","arguments":{"city":"Paris"}}"#,
    ] {
        parser.parse_into(chunk, &mut output).unwrap();
    }
    assert!(
        output
            .events
            .iter()
            .all(|event| !matches!(event, UnifiedParserEvent::ToolCall(delta) if delta.complete))
    );
    parser.parse_into("</tool_call>", &mut output).unwrap();
    assert_eq!(
        output.assembled(),
        [
            UnifiedEvent::Text {
                text: "The literal \"".into(),
            },
            call("get_weather", serde_json::json!({"city":"Paris"})),
        ]
    );

    parser.reset();
    parser.initialize_request(init.clone()).unwrap();
    let mut output = UnifiedParserOutput::default();
    for chunk in [
        "\"",
        "<tool_call>",
        r#"{\"name\":\"lookup\",\"arguments\":{\"q\":\"x\"}}"#,
        "</tool_call>",
        "\" trailing",
    ] {
        parser.parse_into(chunk, &mut output).unwrap();
    }
    assert_eq!(
        output.assembled(),
        [UnifiedEvent::Text {
            text: "\"\" trailing".into(),
        }]
    );

    for abandoned in [
        "<tool_call>\"scalar",
        r#"<tool_call>{\"name\":\"bad\""#,
        r#"<tool_call>{"name":"bad","arguments":{"q":"\q <tool_call>{\"name\":\"fake\"}"#,
    ] {
        parser.reset();
        parser.initialize_request(init.clone()).unwrap();
        parser
            .parse_into(abandoned, &mut UnifiedParserOutput::default())
            .unwrap();
        parser.reset();
        parser.initialize_request(init.clone()).unwrap();
        assert_eq!(
            parser
                .parse_complete(r#"<tool_call>{"name":"ok","arguments":{}}</tool_call> tail"#)
                .unwrap(),
            [
                call("ok", serde_json::json!({})),
                UnifiedEvent::Text {
                    text: " tail".into(),
                },
            ]
        );
    }
}

#[test]
fn modes_and_starting_states() {
    for starting_state in [
        UnifiedParserStartingState::None,
        UnifiedParserStartingState::Reasoning,
        UnifiedParserStartingState::Response,
    ] {
        for named in [false, true] {
            let payload = if named {
                r#"{"city":"Paris"}"#
            } else {
                r#"[{"name":"get_weather","arguments":{"city":"Paris"}}]"#
            };
            let (input, mut expected) = if starting_state == UnifiedParserStartingState::Reasoning {
                (
                    format!("check</think>{payload}"),
                    vec![UnifiedEvent::Reasoning {
                        text: "check".into(),
                    }],
                )
            } else {
                (payload.into(), vec![])
            };
            expected.push(call("get_weather", serde_json::json!({"city":"Paris"})));
            every_split(
                &input,
                UnifiedParserInit {
                    starting_state,
                    tool_output_mode: UnifiedToolOutputMode::GuidedJson {
                        named_tool: named.then(|| "get_weather".into()),
                    },
                    ..Default::default()
                },
                &expected,
            );
        }
    }
    let literal = "<think>literal</think>";
    every_split(
        literal,
        UnifiedParserInit {
            starting_state: UnifiedParserStartingState::Response,
            ..Default::default()
        },
        &[UnifiedEvent::Text {
            text: literal.into(),
        }],
    );
}

#[test]
fn guided_wrapped_payload_inside_reasoning() {
    for payload in [
        r#"[{"name":"get_weather","arguments":{"city":"Paris"}}]"#,
        r#"{"name":"get_weather","arguments":{"city":"Paris"}}"#,
    ] {
        every_split(
            &format!("thinking <tool_call>{payload}</tool_call>"),
            UnifiedParserInit {
                starting_state: UnifiedParserStartingState::Reasoning,
                tool_output_mode: UnifiedToolOutputMode::GuidedJson { named_tool: None },
                ..Default::default()
            },
            &[
                UnifiedEvent::Reasoning {
                    text: "thinking ".into(),
                },
                call("get_weather", serde_json::json!({"city":"Paris"})),
            ],
        );
    }
}

#[test]
fn guided_quoted_wrapper_and_braces_are_visible_prose() {
    if !quote_policy::quote_aware_process("guided_quoted_wrapper_and_braces_are_visible_prose") {
        return;
    }
    for named in [false, true] {
        let prose = "The literal \"<tool_call> { example }\" stays visible. ";
        let payload = if named {
            r#"{"city":"Paris"}"#
        } else {
            r#"{"name":"get_weather","arguments":{"city":"Paris"}}"#
        };
        every_split(
            &format!("{prose}{payload}"),
            UnifiedParserInit {
                starting_state: UnifiedParserStartingState::Response,
                tool_output_mode: UnifiedToolOutputMode::GuidedJson {
                    named_tool: named.then(|| "get_weather".into()),
                },
                ..Default::default()
            },
            &[
                UnifiedEvent::Text { text: prose.into() },
                call("get_weather", serde_json::json!({"city":"Paris"})),
            ],
        );
    }
}

#[test]
fn native_progress_preserves_raw_arguments_and_completion() {
    for position in 0..3 {
        let mut fields = vec![r#""empty":"""#.to_string(), r#""short":" ""#.to_string()];
        fields.insert(
            position,
            format!(r#""long":"{}""#, "雪<&amp;>\\\"".repeat(300)),
        );
        let arguments = format!("{{{}}}", fields.join(","));
        let body = format!(r#"{{"name":"write_file","arguments":{arguments}}}"#);
        let mut parser = create_unified_parser_for_family("hermes", &[]).unwrap();
        let mut output = UnifiedParserOutput::default();
        parser.parse_into("<tool_call>", &mut output).unwrap();
        let mut progress = 0;
        for (at, ch) in body.char_indices() {
            let before = output.tool_arguments_raw().get(&0).map_or(0, String::len);
            parser
                .parse_into(&body[at..at + ch.len_utf8()], &mut output)
                .unwrap();
            progress +=
                usize::from(output.tool_arguments_raw().get(&0).map_or(0, String::len) > before);
        }
        assert!(
            progress > 100,
            "long field must keep streaming before outer close"
        );
        assert!(
            output.events.iter().all(
                |event| !matches!(event,UnifiedParserEvent::ToolCall(delta) if delta.complete)
            )
        );
        parser.parse_into("</tool_call>", &mut output).unwrap();
        output.append(&mut parser.finish().unwrap());
        assert_eq!(output.tool_arguments_raw()[&0], arguments);
        assert!(
            output
                .events
                .iter()
                .any(|event| matches!(event,UnifiedParserEvent::ToolCall(delta) if delta.complete))
        );
    }
}

#[test]
fn reset_reuse_and_invalid_guided_payload() {
    let mut parser = create_unified_parser_for_family("hermes", &[]).unwrap();
    assert!(parser.preserve_special_tokens());
    for abandoned in [
        "<tool_call>{\"name\":\"bad\",\"arguments\":{",
        "<think>unfinished",
        "<tool_",
    ] {
        parser
            .parse_into(abandoned, &mut UnifiedParserOutput::default())
            .unwrap();
        parser.reset();
        parser
            .initialize_request(UnifiedParserInit::default())
            .unwrap();
        assert_eq!(
            parser.parse_complete("visible").unwrap(),
            vec![UnifiedEvent::Text {
                text: "visible".into()
            }]
        );
        parser.reset();
    }
    parser
        .initialize_request(UnifiedParserInit {
            tool_output_mode: UnifiedToolOutputMode::GuidedJson {
                named_tool: Some("f".into()),
            },
            invalid_guided_payload: InvalidGuidedPayloadPolicy::Reject,
            ..Default::default()
        })
        .unwrap();
    let mut output = UnifiedParserOutput::default();
    let rejected = parser.parse_into("[1,2]", &mut output).is_err() || parser.finish().is_err();
    assert!(rejected);
    parser.reset();
    parser
        .initialize_request(UnifiedParserInit::default())
        .unwrap();
    assert_eq!(
        parser
            .parse_complete(r#"<tool_call>{"name":"f"}</tool_call>"#)
            .unwrap(),
        vec![call("f", serde_json::json!({}))]
    );
}

#[test]
fn invalid_native_shapes_never_complete_and_later_calls_survive() {
    for body in [
        r#"{"name":"bad","arguments":null}"#,
        r#"{"name":"bad","arguments":42}"#,
        r#"{"name":"bad","arguments":[]}"#,
        r#"{"name":"bad","arguments":"text"}"#,
        r#"{"arguments":{}}"#,
        r#"{"name":"bad","name":"changed","arguments":{}}"#,
        r#"{"name":"bad","arguments":{"x":"é"},"parameters":{}}"#,
        r#"{"name":"bad","arguments":{"x":"é"},"arguments":{}}"#,
    ] {
        let input = format!(
            r#"<tool_call>{body}</tool_call><tool_call>{{"name":"ok","arguments":{{"x":"雪"}}}}</tool_call>"#
        );
        for at in (0..=input.len()).filter(|&at| input.is_char_boundary(at)) {
            let output = parse(&[&input[..at], &input[at..]], UnifiedParserInit::default());
            assert_eq!(
                output.assembled(),
                vec![call("ok", serde_json::json!({"x":"雪"}))],
                "{body} split {at}"
            );
            let completions: Vec<_> = output
                .events
                .iter()
                .filter_map(|event| match event {
                    UnifiedParserEvent::ToolCall(delta) if delta.complete => Some(delta.tool_index),
                    _ => None,
                })
                .collect();
            assert_eq!(
                completions.len(),
                1,
                "invalid provisional output must not gain completion"
            );
        }
    }
    for name in [
        "a",
        "not_offered",
        "my-tool.v2",
        "server.tools.weather",
        "雪",
    ] {
        let body = serde_json::json!({"name":name,"arguments":{"name":"literal","arguments":{"nested":true}}});
        every_split(
            &format!("<tool_call>{body}</tool_call>"),
            UnifiedParserInit::default(),
            &[call(
                name,
                serde_json::json!({"name":"literal","arguments":{"nested":true}}),
            )],
        );
    }
}
