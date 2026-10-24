// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_parsers_v2::{
    InvalidGuidedPayloadPolicy, Tool, UnifiedEvent, UnifiedParser, UnifiedParserEvent,
    UnifiedParserExt, UnifiedParserInit, UnifiedParserStartingState, UnifiedToolOutputMode,
    assemble, create_unified_parser_for_family,
};
use serde_json::{Value, json};

fn tools(schema: Value) -> Vec<Tool> {
    vec![Tool {
        name: "record".into(),
        description: None,
        parameters: schema,
        strict: None,
    }]
}

fn parser(schema: Value) -> Box<dyn UnifiedParser> {
    create_unified_parser_for_family("minimax_m3", &tools(schema)).unwrap()
}

fn call(body: &str) -> String {
    format!(
        "]<]minimax[>[<tool_call>]<]minimax[>[<invoke name=\"record\">{body}]<]minimax[>[</invoke>]<]minimax[>[</tool_call>"
    )
}

fn expected(arguments: Value) -> UnifiedEvent {
    UnifiedEvent::ToolCall {
        name: "record".into(),
        arguments,
    }
}

fn assert_splits(input: &str, schema: Value, init: UnifiedParserInit, want: &[UnifiedEvent]) {
    for split in input.char_indices().map(|(at, _)| at).chain([input.len()]) {
        let mut p = parser(schema.clone());
        p.initialize_request(init.clone()).unwrap();
        let mut output = p.push(&input[..split]).unwrap();
        output.extend(p.push(&input[split..]).unwrap());
        output.extend(p.finish().unwrap().events);
        assert_eq!(assemble(&output), want, "split={split}, input={input:?}");
    }
    let mut p = parser(schema);
    p.initialize_request(init).unwrap();
    let mut output = Vec::new();
    for ch in input.chars() {
        output.extend(p.push(ch.encode_utf8(&mut [0; 4])).unwrap());
    }
    output.extend(p.finish().unwrap().events);
    assert_eq!(assemble(&output), want, "one character per chunk");
}

#[test]
fn guided_response_prefill_strips_native_body_before_json_payload() {
    let schema = json!({"type":"object","properties":{"text":{"type":"string"}}});
    let payload = r#"[{"name":"record","arguments":{"text":"guided"}}]"#;
    for native in [
        call("]<]minimax[>[<text>native]<]minimax[>[</text>"),
        call("]<]minimax[>[<text>[1,2]]<]minimax[>[</text>"),
        call("]<]minimax[>[<text>{\"a\":1}]<]minimax[>[</text>"),
    ] {
        assert_splits(
            &format!("{native}{payload}"),
            schema.clone(),
            UnifiedParserInit {
                starting_state: UnifiedParserStartingState::Response,
                tool_output_mode: UnifiedToolOutputMode::GuidedJson { named_tool: None },
                ..Default::default()
            },
            &[expected(json!({"text":"guided"}))],
        );
    }
}

#[test]
fn guided_native_fallback_streams_parallel_calls_and_resets() {
    let schema = json!({"type":"object","properties":{"text":{"type":"string"}}});
    let init = UnifiedParserInit {
        starting_state: UnifiedParserStartingState::Reasoning,
        tool_output_mode: UnifiedToolOutputMode::GuidedJson { named_tool: None },
        ..Default::default()
    };
    let first = call("]<]minimax[>[<text>one]<]minimax[>[</text>");
    let second = call("]<]minimax[>[<text>two]<]minimax[>[</text>");
    // Two invocations within one outer block, as emitted by M3 parallel calls.
    let input = format!(
        "Look</mm:think>{}\n{}",
        first.trim_end_matches("]<]minimax[>[</tool_call>"),
        second.trim_start_matches("]<]minimax[>[<tool_call>")
    );
    assert_splits(
        &input,
        schema.clone(),
        init.clone(),
        &[
            UnifiedEvent::Reasoning {
                text: "Look".into(),
            },
            expected(json!({"text":"one"})),
            expected(json!({"text":"two"})),
        ],
    );
    let mut p = parser(schema);
    p.initialize_request(init).unwrap();
    let cut = input.find("one").unwrap() + 3;
    let mut output = p.push(&input[..cut]).unwrap();
    assert!(output.iter().any(
        |e| matches!(e, UnifiedParserEvent::ToolCall(c) if c.name.as_deref() == Some("record"))
    ));
    assert!(output.iter().any(|e| matches!(e, UnifiedParserEvent::ToolCall(c) if c.arguments.contains("one") && !c.complete)));
    output.extend(p.push(&input[cut..]).unwrap());
    output.extend(p.finish().unwrap().events);
    let indices: Vec<_> = output
        .iter()
        .filter_map(|e| match e {
            UnifiedParserEvent::ToolCall(c) if c.complete => Some(c.tool_index),
            _ => None,
        })
        .collect();
    assert_eq!(indices, [0, 1]);
    p.reset();
    let mut output = p
        .push("Again</mm:think>{\"name\":\"record\",\"arguments\":{\"text\":\"json\"}}")
        .unwrap();
    output.extend(p.finish().unwrap().events);
    assert_eq!(
        assemble(&output),
        [
            UnifiedEvent::Reasoning {
                text: "Again".into()
            },
            expected(json!({"text":"json"}))
        ]
    );
}

#[test]
fn guided_native_fallback_enforces_named_tool_and_clears_rejection_on_reset() {
    let schema = json!({"type":"object","properties":{"text":{"type":"string"}}});
    let mut catalog = tools(schema);
    let mut other = catalog[0].clone();
    other.name = "other".into();
    catalog.push(other);
    let mut p = create_unified_parser_for_family("minimax_m3", &catalog).unwrap();
    p.initialize_request(UnifiedParserInit {
        tool_output_mode: UnifiedToolOutputMode::GuidedJson {
            named_tool: Some("record".into()),
        },
        ..Default::default()
    })
    .unwrap();
    let input = call("]<]minimax[>[<text>secret]<]minimax[>[</text>")
        .replace("name=\"record\"", "name=\"other\"");
    let mut output = Vec::new();
    for ch in input.chars() {
        output.extend(p.push(ch.encode_utf8(&mut [0; 4])).unwrap());
    }
    output.extend(p.finish().unwrap().events);
    assert!(
        output.is_empty(),
        "wrong tool must not emit a name or argument fragments: {output:?}"
    );
    p.reset();
    let mut output = p
        .push(&call("]<]minimax[>[<text>ok]<]minimax[>[</text>"))
        .unwrap();
    output.extend(p.finish().unwrap().events);
    assert_eq!(assemble(&output), [expected(json!({"text":"ok"}))]);
}

#[test]
fn guided_json_native_header_in_arguments_is_literal() {
    let schema = json!({"type":"object","properties":{"text":{"type":"string"}}});
    let text = call("]<]minimax[>[<text>literal]<]minimax[>[</text>");
    let arguments = serde_json::to_string(&json!({"text":text})).unwrap();
    for payload in [
        format!("{{\"name\":\"record\",\"arguments\":{arguments}}}"),
        format!("{{\"arguments\":{arguments},\"name\":\"record\"}}"),
    ] {
        assert_splits(
            &payload,
            schema.clone(),
            UnifiedParserInit {
                tool_output_mode: UnifiedToolOutputMode::GuidedJson { named_tool: None },
                ..Default::default()
            },
            &[expected(json!({"text":text}))],
        );
    }
}

#[test]
fn guided_named_native_fallback_accepts_only_one_matching_call() {
    let schema = json!({"type":"object","properties":{"text":{"type":"string"}}});
    let wrong = call("]<]minimax[>[<text>wrong]<]minimax[>[</text>")
        .replace("name=\"record\"", "name=\"other\"");
    let input = format!(
        "{wrong}{}{}",
        call("]<]minimax[>[<text>ok]<]minimax[>[</text>"),
        call("]<]minimax[>[<text>extra]<]minimax[>[</text>")
    );
    let init = UnifiedParserInit {
        tool_output_mode: UnifiedToolOutputMode::GuidedJson {
            named_tool: Some("record".into()),
        },
        ..Default::default()
    };
    assert_splits(
        &input,
        schema.clone(),
        init.clone(),
        &[expected(json!({"text":"ok"}))],
    );
    let mut p = parser(schema);
    p.initialize_request(init).unwrap();
    let mut output = p.push(&input).unwrap();
    output.extend(p.finish().unwrap().events);
    assert!(
        output.iter().all(
            |event| !matches!(event, UnifiedParserEvent::ToolCall(call) if call.tool_index != 0)
        )
    );
}

#[test]
fn guided_native_recovery_does_not_reuse_a_rejected_header() {
    let schema = json!({"type":"object","properties":{"text":{"type":"string"}}});
    let input = format!(
        "]<]minimax[>[<invoke name=\"<mm:think>Check.</mm:think>{}",
        call("]<]minimax[>[<text>ok]<]minimax[>[</text>")
    );
    assert_splits(
        &input,
        schema,
        UnifiedParserInit {
            tool_output_mode: UnifiedToolOutputMode::GuidedJson { named_tool: None },
            invalid_guided_payload: InvalidGuidedPayloadPolicy::RecoverAsText,
            ..Default::default()
        },
        &[
            UnifiedEvent::Reasoning {
                text: "Check.".into(),
            },
            expected(json!({"text":"ok"})),
        ],
    );
}

#[test]
fn ordered_channels_and_missing_reasoning_close() {
    let schema = json!({"type":"object","properties":{"text":{"type":"string"}}});
    let tool = call("]<]minimax[>[<text>Montréal 🐈]<]minimax[>[</text>");
    for reason in ["<mm:think>Look</mm:think>", "<mm:think>Look"] {
        assert_splits(
            &format!("{reason}{tool}Done."),
            schema.clone(),
            UnifiedParserInit::default(),
            &[
                UnifiedEvent::Reasoning {
                    text: "Look".into(),
                },
                expected(json!({"text":"Montréal 🐈"})),
                if reason.ends_with("</mm:think>") {
                    UnifiedEvent::Text {
                        text: "Done.".into(),
                    }
                } else {
                    // A tool suspends an open reasoning channel. Only the
                    // reasoning closer changes subsequent prose to text.
                    UnifiedEvent::Reasoning {
                        text: "Done.".into(),
                    }
                },
            ],
        );
    }
}

#[test]
fn prefills_and_parallel_indices() {
    let schema = json!({"type":"object","properties":{"count":{"type":"integer"}}});
    let tool = call("]<]minimax[>[<count>2]<]minimax[>[</count>");
    assert_splits(
        &format!("Look</mm:think>{tool}{tool}"),
        schema.clone(),
        UnifiedParserInit {
            starting_state: UnifiedParserStartingState::Reasoning,
            ..Default::default()
        },
        &[
            UnifiedEvent::Reasoning {
                text: "Look".into(),
            },
            expected(json!({"count":2})),
            expected(json!({"count":2})),
        ],
    );
    let mut p = parser(schema);
    let mut output = p.push(&format!("{tool}{tool}")).unwrap();
    output.extend(p.finish().unwrap().events);
    let completed: Vec<_> = output
        .iter()
        .filter_map(|event| match event {
            UnifiedParserEvent::ToolCall(call) if call.complete => Some(call.tool_index),
            _ => None,
        })
        .collect();
    assert_eq!(completed, [0, 1]);
}

#[test]
fn nested_same_name_and_schema_unions() {
    let schema = json!({"oneOf":[{"type":"object","properties":{
        "payload":{"type":"object","properties":{
            "payload":{"type":"object","properties":{"count":{"type":"integer"}}},
            "enabled":{"type":"boolean"}
        }}
    }}]});
    let body = concat!(
        "]<]minimax[>[<payload>]<]minimax[>[<payload>]<]minimax[>[<count>2",
        "]<]minimax[>[</count>]<]minimax[>[</payload>]<]minimax[>[<enabled>true",
        "]<]minimax[>[</enabled>]<]minimax[>[</payload>"
    );
    assert_splits(
        &call(body),
        schema,
        UnifiedParserInit::default(),
        &[expected(
            json!({"payload":{"payload":{"count":2},"enabled":true}}),
        )],
    );
}

#[test]
fn root_composed_parameters_preserve_types_at_every_split() {
    for keyword in ["allOf", "anyOf", "oneOf"] {
        let schema = json!({keyword:[{"type":"object","properties":{
            "count":{"type":"integer"},"text":{"type":"string"}
        }}]});
        assert_splits(
            &call(concat!(
                "]<]minimax[>[<count>2]<]minimax[>[</count>",
                "]<]minimax[>[<text>002]<]minimax[>[</text>"
            )),
            schema,
            UnifiedParserInit::default(),
            &[expected(json!({"count":2,"text":"002"}))],
        );
    }
}

#[test]
fn string_entities_whitespace_and_channel_markers_are_literal() {
    let text = "  &amp; &#39; %20 <mm:think>literal</mm:think> \"雪\"\n ";
    let schema = json!({"type":"object","properties":{"text":{"type":"string"}}});
    assert_splits(
        &call(&format!("]<]minimax[>[<text>{text}]<]minimax[>[</text>")),
        schema,
        UnifiedParserInit::default(),
        &[expected(json!({"text":text}))],
    );
}

#[test]
fn native_control_markers_inside_strings_do_not_dispatch() {
    let text = "before ]<]minimax[>[</invoke> ]<]minimax[>[</tool_call> ]<]minimax[>[<invoke name=\"other\"> after";
    let schema = json!({"type":"object","properties":{
        "text":{"type":"string"}, "after":{"type":"string"}
    }});
    assert_splits(
        &call(&format!(
            "]<]minimax[>[<text>{text}]<]minimax[>[</text>]<]minimax[>[<after>ok]<]minimax[>[</after>"
        )),
        schema,
        UnifiedParserInit::default(),
        &[expected(json!({"text":text,"after":"ok"}))],
    );
}

#[test]
fn consecutive_string_arguments_stream_independently() {
    let schema = json!({"type":"object","properties":{
        "first":{"type":"string"}, "second":{"type":"string"}
    }});
    let mut p = parser(schema);
    let mut output = p
        .push(
            "]<]minimax[>[<tool_call>]<]minimax[>[<invoke name=\"record\">]<]minimax[>[<first>one",
        )
        .unwrap();
    let next = p
        .push("]<]minimax[>[</first>]<]minimax[>[<second>two")
        .unwrap();
    assert!(next.iter().any(|event| matches!(event, UnifiedParserEvent::ToolCall(call) if call.arguments.contains("two") && !call.complete)));
    output.extend(next);
    output.extend(
        p.push("]<]minimax[>[</second>]<]minimax[>[</invoke>]<]minimax[>[</tool_call>")
            .unwrap(),
    );
    output.extend(p.finish().unwrap().events);
    assert_eq!(
        assemble(&output),
        [expected(json!({"first":"one","second":"two"}))]
    );
}

#[test]
fn arrays_references_and_nullable_values() {
    let schema = json!({"type":"object","$defs":{"Count":{"type":"integer"}},"properties":{
        "items":{"type":"array","items":{"$ref":"#/$defs/Count"}},
        "value":{"anyOf":[{"type":"string"},{"type":"null"}]},
        "text":{"type":"string"}
    }});
    let body = concat!(
        "]<]minimax[>[<items>]<]minimax[>[<item>1]<]minimax[>[</item>]<]minimax[>[<item>2]<]minimax[>[</item>]<]minimax[>[</items>",
        "]<]minimax[>[<value>null]<]minimax[>[</value>]<]minimax[>[<text>null]<]minimax[>[</text>"
    );
    assert_splits(
        &call(body),
        schema,
        UnifiedParserInit::default(),
        &[expected(json!({"items":[1,2],"value":null,"text":"null"}))],
    );
}

#[test]
fn native_arguments_make_progress_before_invoke_close() {
    let schema = json!({"type":"object","properties":{"text":{"type":"string"}}});
    let mut p = parser(schema);
    let mut output = p
        .push(
            "]<]minimax[>[<tool_call>]<]minimax[>[<invoke name=\"record\">]<]minimax[>[<text>Hello",
        )
        .unwrap();
    let first: String = output
        .iter()
        .filter_map(|event| match event {
            UnifiedParserEvent::ToolCall(call) => Some(call.arguments.as_str()),
            _ => None,
        })
        .collect();
    assert!(
        first.contains("Hello"),
        "string argument was buffered: {output:?}"
    );
    let next = p.push(" world").unwrap();
    assert!(next.iter().any(|event| matches!(event, UnifiedParserEvent::ToolCall(call) if call.arguments.contains("world") && !call.complete)));
    output.extend(next);
    output.extend(
        p.push("]<]minimax[>[</text>]<]minimax[>[</invoke>]<]minimax[>[</tool_call>")
            .unwrap(),
    );
    output.extend(p.finish().unwrap().events);
    assert_eq!(assemble(&output), [expected(json!({"text":"Hello world"}))]);
}

#[test]
fn guided_named_and_required_after_reasoning() {
    let schema = json!({"type":"object","properties":{"text":{"type":"string"}}});
    for (named_tool, body) in [
        (Some("record".to_string()), r#"{"text":"雪"}"#),
        (None, r#"[{"name":"record","arguments":{"text":"雪"}}]"#),
    ] {
        let init = UnifiedParserInit {
            tool_output_mode: UnifiedToolOutputMode::GuidedJson { named_tool },
            invalid_guided_payload: InvalidGuidedPayloadPolicy::StreamBestEffort,
            ..Default::default()
        };
        assert_splits(
            &format!("<mm:think>Look</mm:think>{body}"),
            schema.clone(),
            init,
            &[
                UnifiedEvent::Reasoning {
                    text: "Look".into(),
                },
                expected(json!({"text":"雪"})),
            ],
        );
    }
}

#[test]
fn guided_fallback_recovers_live_native_call_after_reasoning() {
    let schema = json!({"type":"object","properties":{"text":{"type":"string"}},"additionalProperties":true});
    let native = call("]<]minimax[>[<text>937 * 18 + 42]<]minimax[>[</text>");
    for named_tool in [None, Some("record".to_string())] {
        for policy in [
            InvalidGuidedPayloadPolicy::Reject,
            InvalidGuidedPayloadPolicy::RecoverAsText,
            InvalidGuidedPayloadPolicy::StreamBestEffort,
        ] {
            // A prompt that already opened visible content follows the shared
            // suppression contract instead; the conformance corpus pins that state.
            for (state, prefix) in [
                (
                    UnifiedParserStartingState::Reasoning,
                    "Use the calculator.</mm:think>",
                ),
                (
                    UnifiedParserStartingState::None,
                    "<mm:think>Use the calculator.</mm:think>",
                ),
            ] {
                let want = vec![
                    UnifiedEvent::Reasoning {
                        text: "Use the calculator.".into(),
                    },
                    expected(json!({"text":"937 * 18 + 42"})),
                ];
                assert_splits(
                    &format!("{prefix}{native}"),
                    schema.clone(),
                    UnifiedParserInit {
                        starting_state: state,
                        tool_output_mode: UnifiedToolOutputMode::GuidedJson {
                            named_tool: named_tool.clone(),
                        },
                        invalid_guided_payload: policy,
                        ..Default::default()
                    },
                    &want,
                );
            }
        }
    }
}

#[test]
fn closed_typed_argument_does_not_delay_a_later_string() {
    let mut p = parser(json!({"type":"object","properties":{
        "count":{"type":"integer"}, "text":{"type":"string"}
    }}));
    let mut output = p.push("]<]minimax[>[<tool_call>]<]minimax[>[<invoke name=\"record\">]<]minimax[>[<count>2]<]minimax[>[</count>").unwrap();
    assert!(output.iter().any(|event| matches!(event, UnifiedParserEvent::ToolCall(call) if call.arguments.contains("2") && !call.complete)));
    let next = p.push("]<]minimax[>[<text>Hello").unwrap();
    assert!(next.iter().any(|event| matches!(event, UnifiedParserEvent::ToolCall(call) if call.arguments.contains("Hello") && !call.complete)));
    output.extend(next);
    output.extend(
        p.push("]<]minimax[>[</text>]<]minimax[>[</invoke>]<]minimax[>[</tool_call>")
            .unwrap(),
    );
    output.extend(p.finish().unwrap().events);
    assert_eq!(
        assemble(&output),
        [expected(json!({"count":2,"text":"Hello"}))]
    );
}

#[test]
fn truncated_markup_is_hidden_and_reset_clears_partial_call() {
    let schema = json!({"type":"object","properties":{"text":{"type":"string"}}});
    // An unrecognized marker prefix is ordinary text at EOF; a recognized
    // tool opener owns (and suppresses) an unfinished call.
    assert_splits(
        "Before ]<]mini",
        schema.clone(),
        UnifiedParserInit::default(),
        &[UnifiedEvent::Text {
            text: "Before ]<]mini".into(),
        }],
    );
    for tail in [
        "]<]minimax[>[<tool_call>",
        "]<]minimax[>[<invoke name=\"record\">]<]minimax[>[<text>unfinished",
    ] {
        assert_splits(
            &format!("Before {tail}"),
            schema.clone(),
            UnifiedParserInit::default(),
            &[UnifiedEvent::Text {
                text: "Before ".into(),
            }],
        );
    }
    let mut p = parser(schema);
    p.push("]<]minimax[>[<tool_call>]<]minimax[>[<invoke name=\"record\">]<]minimax[>[<text>old")
        .unwrap();
    p.reset();
    let mut output = p
        .push(&call("]<]minimax[>[<text>new]<]minimax[>[</text>"))
        .unwrap();
    output.extend(p.finish().unwrap().events);
    assert_eq!(assemble(&output), [expected(json!({"text":"new"}))]);
}

#[test]
fn guided_native_headers_and_bodies_preserve_channel_ownership() {
    let schema = json!({"type":"object","properties":{"text":{"type":"string"}}});
    let payload = r#"[{"name":"record","arguments":{"text":"a > b"}}]"#;
    let init = UnifiedParserInit {
        tool_output_mode: UnifiedToolOutputMode::GuidedJson { named_tool: None },
        invalid_guided_payload: InvalidGuidedPayloadPolicy::StreamBestEffort,
        ..Default::default()
    };
    for prefix in [
        "]<]minimax[>[<invoke name=\"",
        "]<]minimax[>[<invoke name=\"record\"",
    ] {
        assert_splits(
            &format!("{prefix}{payload}"),
            schema.clone(),
            init.clone(),
            &[expected(json!({"text":"a > b"}))],
        );
    }
    assert_splits(
        &format!("]<]minimax[>[<invoke name=\"<mm:think>secret</mm:think>{payload}"),
        schema.clone(),
        init.clone(),
        &[
            UnifiedEvent::Reasoning {
                text: "secret".into(),
            },
            expected(json!({"text":"a > b"})),
        ],
    );
    assert_splits(
        &format!("<mm:think>I'll call ]<]minimax[>[<invoke name=\"record</mm:think>{payload}"),
        schema.clone(),
        init.clone(),
        &[
            UnifiedEvent::Reasoning {
                text: "I'll call record".into(),
            },
            expected(json!({"text":"a > b"})),
        ],
    );
    let native = call("]<]minimax[>[<text><mm:think>quoted</mm:think>]<]minimax[>[</text>");
    assert_splits(
        &native,
        schema.clone(),
        init.clone(),
        &[expected(json!({"text":"<mm:think>quoted</mm:think>"}))],
    );
    assert_splits(
        &format!("<mm:think>before {native} after</mm:think>{payload}"),
        schema,
        init,
        &[
            UnifiedEvent::Reasoning {
                text: "before  after".into(),
            },
            expected(json!({"text":"a > b"})),
        ],
    );
}

#[test]
fn scalar_types_intersect_sibling_and_allof_constraints() {
    for value_schema in [
        json!({"type":"integer","anyOf":[{"type":"integer"},{"type":"string"}]}),
        json!({"type":"integer","oneOf":[{"type":"integer"},{"type":"string"}]}),
        json!({"allOf":[{"type":["string","integer"]},{"type":"number"}]}),
    ] {
        let schema = json!({"type":"object","properties":{"count":value_schema}});
        assert_splits(
            &call("]<]minimax[>[<count>42]<]minimax[>[</count>"),
            schema,
            UnifiedParserInit::default(),
            &[expected(json!({"count":42}))],
        );
    }
    for (broad, narrow, raw, value) in [
        (
            json!({"type":["string","null"]}),
            json!({"type":"string"}),
            "null",
            json!("null"),
        ),
        (
            json!({"type":["string","integer"]}),
            json!({"type":"integer"}),
            "42",
            json!(42),
        ),
    ] {
        for (first, second) in [(&broad, &narrow), (&narrow, &broad)] {
            for schema in [
                json!({"type":"object","allOf":[
                    {"properties":{"value":first}}, {"properties":{"value":second}}]}),
                json!({"type":"object","properties":{"value":first},
                    "allOf":[{"properties":{"value":second}}]}),
            ] {
                assert_splits(
                    &call(&format!("]<]minimax[>[<value>{raw}]<]minimax[>[</value>")),
                    schema,
                    UnifiedParserInit::default(),
                    &[expected(json!({"value":value}))],
                );
            }
        }
    }
}

#[test]
fn parameter_names_with_invoke_prefix_are_not_headers() {
    for key in ["invoke", "invoked"] {
        let schema = json!({"type":"object","properties":{key:{"type":"string"}}});
        let value = if key == "invoke" {
            "]<]minimax[>[</tool_call>literal"
        } else {
            "]<]minimax[>[</invoke>literal"
        };
        assert_splits(
            &call(&format!("]<]minimax[>[<{key}>{value}]<]minimax[>[</{key}>")),
            schema,
            UnifiedParserInit::default(),
            &[expected(json!({key:value}))],
        );
    }
}

#[test]
fn literal_union_alternatives_do_not_hide_typed_parameters() {
    for (kind, raw, value) in [
        ("integer", "42", json!(42)),
        ("number", "3.5", json!(3.5)),
        ("boolean", "true", json!(true)),
        ("object", "{\"x\":1}", json!({"x":1})),
        ("array", "[1,2]", json!([1, 2])),
    ] {
        for union in ["anyOf", "oneOf"] {
            for literal in ["auto", raw] {
                let schema = json!({"type":"object","properties":{"value":{union:[{"const":literal},{"type":kind}]}}});
                assert_splits(
                    &call(&format!("]<]minimax[>[<value>{raw}]<]minimax[>[</value>")),
                    schema,
                    UnifiedParserInit::default(),
                    &[expected(json!({"value":value}))],
                );
            }
        }
    }
}

#[test]
fn repeated_array_parameters_preserve_legacy_values_at_every_split() {
    let schema = json!({"type":"object","properties":{"values":{"type":"array","items":{"type":"integer"}}}});
    assert_splits(
        &call(
            "]<]minimax[>[<values>1]<]minimax[>[</values>]<]minimax[>[<values>2]<]minimax[>[</values>",
        ),
        schema,
        UnifiedParserInit::default(),
        &[expected(json!({"values":[1,2]}))],
    );
}

#[test]
fn incomplete_namespace_like_text_does_not_delay_a_closed_string_call() {
    let schema = json!({"type":"object","properties":{"text":{"type":"string"}}});
    for value in ["]<]minimax[>[<unfinished", "]<]minimax[>[<text>literal"] {
        let input = call(&format!("]<]minimax[>[<text>{value}]<]minimax[>[</text>"));
        let mut p = parser(schema.clone());
        let output = p.push(&input).unwrap();
        assert_eq!(
            assemble(&output),
            [expected(json!({"text":value}))],
            "closed call waited for EOF: {value:?}"
        );
        assert_splits(
            &input,
            schema.clone(),
            UnifiedParserInit::default(),
            &[expected(json!({"text":value}))],
        );
    }
}

#[test]
fn nested_string_markers_are_literal_and_do_not_delay_completion() {
    for text in [
        "]<]minimax[>[</payload>",
        "]<]minimax[>[<payload>",
        "]<]minimax[>[<unfinished",
        "]<]minimax[>[</invoke>",
    ] {
        let schema = json!({"type":"object","properties":{"payload":{"type":"object","properties":{"text":{"type":"string"}}}}});
        let wire = call(&format!(
            "]<]minimax[>[<payload>]<]minimax[>[<text>{text}]<]minimax[>[</text>]<]minimax[>[</payload>"
        ));
        let mut p = parser(schema.clone());
        let output = p.push(&wire).unwrap();
        assert_eq!(
            assemble(&output),
            [expected(json!({"payload":{"text":text}}))],
            "whole completed call: {text}"
        );
        assert_splits(
            &wire,
            schema,
            UnifiedParserInit::default(),
            &[expected(json!({"payload":{"text":text}}))],
        );
    }
}

#[test]
fn fixed_parameter_names_preserve_spaces_and_unicode() {
    let arguments =
        json!({"a b":"space", "é:name":"Unicode", " name ":"padded", "text<part":"angle"});
    let properties: serde_json::Map<String, Value> = arguments
        .as_object()
        .unwrap()
        .keys()
        .map(|key| (key.clone(), json!({"type":"string"})))
        .collect();
    let body = arguments
        .as_object()
        .unwrap()
        .iter()
        .map(|(key, value)| {
            format!(
                "]<]minimax[>[<{key}>{}]<]minimax[>[</{key}>",
                value.as_str().unwrap()
            )
        })
        .collect::<String>();
    assert_splits(
        &call(&body),
        json!({"type":"object","properties":properties}),
        UnifiedParserInit::default(),
        &[expected(arguments)],
    );
}

#[test]
fn literal_schema_values_and_scalar_unions_preserve_json_types() {
    // Reduced from the pinned OpenRouter realistic schema-fuzz cases. The
    // native value is unambiguous when a schema excludes its string spelling.
    for (schema, raw, value) in [
        (
            json!({"enum":[106254871930264_u64,108837402915583_u64]}),
            "106254871930264",
            json!(106254871930264_u64),
        ),
        (json!({"const":0.2}), "0.2", json!(0.2)),
        (json!({"const":false}), "false", json!(false)),
        (
            json!({"anyOf":[{"const":false},{"type":"null"}]}),
            "false",
            json!(false),
        ),
        (
            json!({"oneOf":[{"type":"integer"},{"type":"boolean"}]}),
            "false",
            json!(false),
        ),
        (
            json!({"anyOf":[{"type":"number"},{"type":"boolean"}]}),
            "true",
            json!(true),
        ),
        (
            json!({"anyOf":[{"type":"integer"},{"type":"object"}]}),
            "{\"x\":1}",
            json!({"x":1}),
        ),
        (json!({"type":["number","array"]}), "[1,2]", json!([1, 2])),
        (json!({"not":{"type":"string"}}), "-2500", json!(-2500)),
    ] {
        assert_splits(
            &call(&format!("]<]minimax[>[<value>{raw}]<]minimax[>[</value>")),
            json!({"type":"object","properties":{"value":schema}}),
            UnifiedParserInit::default(),
            &[expected(json!({"value":value}))],
        );
    }
}

#[test]
fn root_reference_with_definition_containers_resolves_parameters() {
    for defs in ["$defs", "definitions"] {
        let mut schema = json!({"$ref":format!("#/{defs}/Parameters")});
        schema[defs] = json!({"Parameters":{
            "type":"object","properties":{
                "enabled":{"type":"boolean"},"text":{"type":"string"}
            }
        }});
        assert_splits(
            &call(concat!(
                "]<]minimax[>[<enabled>false]<]minimax[>[</enabled>",
                "]<]minimax[>[<text>002]<]minimax[>[</text>"
            )),
            schema,
            UnifiedParserInit::default(),
            &[expected(json!({"enabled":false,"text":"002"}))],
        );
    }
}
