// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_parsers_v2::{
    ToolParseResult, UnifiedEvent, UnifiedParserExt, UnifiedParserInit, UnifiedParserOutput,
    UnifiedParserStartingState, assemble, create_tool_parser_for_family,
    create_unified_parser_for_family,
};

const FAMILIES: [&str; 2] = ["deepseek_v4", "deepseek_v41"];

fn native_call(family: &str, wrapped: bool, value: &str) -> String {
    let (gap, block) = match family {
        "deepseek_v4" => ("", "tool_calls"),
        "deepseek_v41" => (" ", " calls"),
        _ => unreachable!(),
    };
    let invoke = format!(
        "<｜DSML｜{gap}invoke name=\"run\"><｜DSML｜{gap}parameter name=\"value\" string=\"true\">{value}</｜DSML｜{gap}parameter></｜DSML｜{gap}invoke>"
    );
    if wrapped {
        format!("<｜DSML｜{block}>{invoke}</｜DSML｜{block}>")
    } else {
        invoke
    }
}

fn chunkings(input: &str) -> Vec<Vec<&str>> {
    let boundaries: Vec<_> = input
        .char_indices()
        .map(|(at, _)| at)
        .chain([input.len()])
        .collect();
    let mut chunks: Vec<_> = boundaries
        .iter()
        .map(|&at| vec![&input[..at], "", &input[at..]])
        .collect();
    chunks.push(boundaries.windows(2).map(|w| &input[w[0]..w[1]]).collect());
    chunks
}

fn text_and_call(text: &str, value: &str) -> Vec<UnifiedEvent> {
    let mut events = Vec::new();
    if !text.is_empty() {
        events.push(UnifiedEvent::Text { text: text.into() });
    }
    events.push(UnifiedEvent::ToolCall {
        name: "run".into(),
        arguments: serde_json::json!({"value": value}),
    });
    events
}

fn assert_chunks(
    family: &str,
    chunks: &[&str],
    state: UnifiedParserStartingState,
    expected: &[UnifiedEvent],
) {
    let mut parser = create_unified_parser_for_family(family, &[]).unwrap();
    parser
        .initialize_request(UnifiedParserInit {
            starting_state: state,
            ..Default::default()
        })
        .unwrap();
    let mut events = Vec::new();
    for chunk in chunks {
        events.extend(parser.push(chunk).unwrap());
    }
    events.extend(parser.finish().unwrap().events);
    assert_eq!(assemble(&events), expected, "{family}: {chunks:?}");

    if state == UnifiedParserStartingState::Response {
        // The legacy V4 selector accepts both dialects and projects the same scanner.
        let mut legacy = create_tool_parser_for_family("deepseek_v4", &[]).unwrap();
        let mut output = ToolParseResult::default();
        for chunk in chunks {
            output.append(legacy.push(chunk).unwrap());
        }
        output.append(legacy.finish().unwrap());
        let output = output.coalesce_calls();
        let text: String = expected
            .iter()
            .filter_map(|event| match event {
                UnifiedEvent::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(output.normal_text, text, "legacy {family}: {chunks:?}");
        let calls: Vec<_> = expected
            .iter()
            .filter_map(|event| match event {
                UnifiedEvent::ToolCall { name, arguments } => Some((name, arguments)),
                _ => None,
            })
            .collect();
        assert_eq!(
            output.calls.len(),
            calls.len(),
            "legacy {family}: {chunks:?}"
        );
        for (index, (actual, (name, arguments))) in output.calls.iter().zip(calls).enumerate() {
            assert_eq!(actual.tool_index, index);
            assert_eq!(actual.name.as_ref(), Some(name));
            assert_eq!(
                serde_json::from_str::<serde_json::Value>(&actual.arguments).unwrap(),
                *arguments
            );
            assert!(actual.complete);
        }
    }
}

fn assert_every_split(
    family: &str,
    input: &str,
    state: UnifiedParserStartingState,
    expected: &[UnifiedEvent],
) {
    for chunks in chunkings(input) {
        assert_chunks(family, &chunks, state, expected);
    }
}

#[test]
fn wrapped_and_bare_calls_consume_exactly_one_separator_at_every_split() {
    for family in FAMILIES {
        for wrapped in [true, false] {
            for (prefix, content) in [
                ("", ""),
                ("\n", "\n"),
                ("\n\n", ""),
                ("\n\n\n", "\n"),
                ("\n\n\n\n", "\n\n"),
                (" \t\n\n", " \t"),
                ("é🙂.\n\n", "é🙂."),
                ("Check. \t\n\n", "Check. \t"),
                ("Check.\r\n\r\n", "Check.\r\n\r\n"),
            ] {
                let input = format!("{prefix}{}", native_call(family, wrapped, "Paris"));
                assert_every_split(
                    family,
                    &input,
                    UnifiedParserStartingState::Response,
                    &text_and_call(content, "Paris"),
                );
            }
        }
    }
}

#[test]
fn punctuation_and_partial_dsml_anchor_chunks_do_not_emit_framing() {
    for family in FAMILIES {
        for wrapped in [true, false] {
            let call = native_call(family, wrapped, "Paris");
            let tail = call.strip_prefix("<｜DSML｜").unwrap();
            assert_chunks(
                family,
                &["I will", " check", ".\n\n", "<", "｜DSML｜", tail],
                UnifiedParserStartingState::Response,
                &text_and_call("I will check.", "Paris"),
            );
        }
    }
}

#[test]
fn unmatched_separators_and_partial_openers_are_text_at_eof() {
    for family in FAMILIES {
        for input in [
            "First.\n\nSecond.",
            "First.\n",
            "First.\n\n",
            "\n\n",
            "First.\n\n<｜DSML｜tool_",
            "First.\n\n<｜DSML｜unknown>",
        ] {
            for state in [
                UnifiedParserStartingState::None,
                UnifiedParserStartingState::Response,
            ] {
                assert_every_split(
                    family,
                    input,
                    state,
                    &[UnifiedEvent::Text { text: input.into() }],
                );
            }
        }
    }
}

#[test]
fn quoted_literal_and_unmatched_quote_resolve_separator_ownership() {
    for family in FAMILIES {
        for wrapped in [true, false] {
            let call = native_call(family, wrapped, "Paris");
            for quote in ['"', '\'', '`'] {
                let literal = format!("Quoted {quote}\n\n{call}{quote} remains text.");
                assert_every_split(
                    family,
                    &literal,
                    UnifiedParserStartingState::Response,
                    &[UnifiedEvent::Text {
                        text: literal.clone(),
                    }],
                );
                let prefix = format!("Unmatched {quote}quotation");
                assert_every_split(
                    family,
                    &format!("{prefix}\n\n{call}"),
                    UnifiedParserStartingState::Response,
                    &text_and_call(&prefix, "Paris"),
                );
            }
        }
    }
}

#[test]
fn framing_leaves_reasoning_and_argument_whitespace_intact() {
    for family in FAMILIES {
        let value = " \t\n\n<｜DSML｜tool_calls> café\n\n";
        let call = native_call(family, true, value);
        let mut expected = vec![UnifiedEvent::Reasoning {
            text: "reason\n\n".into(),
        }];
        expected.extend(text_and_call("Check.", value));
        assert_every_split(
            family,
            &format!("<think>reason\n\n</think>Check.\n\n{call}"),
            UnifiedParserStartingState::None,
            &expected,
        );
        assert_every_split(
            family,
            &format!("<think>reason\n\n{call}tail</think>"),
            UnifiedParserStartingState::None,
            &[
                UnifiedEvent::Reasoning {
                    text: "reason\n\n".into(),
                },
                UnifiedEvent::ToolCall {
                    name: "run".into(),
                    arguments: serde_json::json!({"value":value}),
                },
                UnifiedEvent::Reasoning {
                    text: "tail".into(),
                },
            ],
        );
        assert_every_split(
            family,
            "Check.\n\n<think>reason</think>",
            UnifiedParserStartingState::None,
            &[
                UnifiedEvent::Text {
                    text: "Check.\n\n".into(),
                },
                UnifiedEvent::Reasoning {
                    text: "reason".into(),
                },
            ],
        );
    }
}

#[test]
fn each_wrapped_block_consumes_its_own_separator() {
    for family in FAMILIES {
        let first = native_call(family, true, "Paris");
        let second = native_call(family, true, "Rome");
        let mut expected = text_and_call("First.", "Paris");
        expected.extend(text_and_call("Second.\n\n", "Rome"));
        assert_every_split(
            family,
            &format!("First.\n\n{first}Second.\n\n\n\n{second}"),
            UnifiedParserStartingState::Response,
            &expected,
        );
    }
}

#[test]
fn incomplete_calls_drop_framing_but_keep_content_at_eof() {
    for family in FAMILIES {
        for wrapped in [true, false] {
            let call = native_call(family, wrapped, "Paris");
            let partial = &call[..call.find("Paris").unwrap()];
            assert_every_split(
                family,
                &format!("Check.\n\n\n\n{partial}"),
                UnifiedParserStartingState::Response,
                &[UnifiedEvent::Text {
                    text: "Check.\n\n".into(),
                }],
            );
        }
    }
}

#[test]
fn emitter_errors_leave_separator_and_call_recoverable() {
    for wrapped in [true, false] {
        let call = native_call("deepseek_v41", wrapped, "invalid")
            .replace("string=\"true\"", "string=\"false\"");
        let input = format!("Check.\n\n{call}");
        for split in [0, input.find("invalid").unwrap(), input.len()] {
            let mut parser = create_unified_parser_for_family("deepseek_v41", &[]).unwrap();
            parser
                .initialize_request(UnifiedParserInit {
                    starting_state: UnifiedParserStartingState::Response,
                    ..Default::default()
                })
                .unwrap();
            let mut output = UnifiedParserOutput::default();
            let result = parser
                .parse_into(&input[..split], &mut output)
                .and_then(|()| parser.parse_into(&input[split..], &mut output));
            assert!(result.is_err());
            assert_eq!(
                assemble(&output.events),
                vec![UnifiedEvent::Text {
                    text: "Check.".into(),
                }]
            );
            assert_eq!(parser.reset(), format!("\n\n{call}"));
        }
    }
}

#[test]
fn reset_returns_uncommitted_separator_and_reuses_the_parser() {
    for family in FAMILIES {
        for wrapped in [true, false] {
            let call = native_call(family, wrapped, "Paris");
            let end = call.find("Paris").unwrap();
            let pending = format!("\n\n{}", &call[..end]);
            let mut parser = create_unified_parser_for_family(family, &[]).unwrap();
            parser
                .initialize_request(UnifiedParserInit {
                    starting_state: UnifiedParserStartingState::Response,
                    ..Default::default()
                })
                .unwrap();
            let emitted = parser.push(&format!("Check.{pending}")).unwrap();
            assert_eq!(
                assemble(&emitted),
                vec![UnifiedEvent::Text {
                    text: "Check.".into()
                }]
            );
            assert_eq!(parser.reset(), pending);
            let mut events = parser.push(&format!("Again.\n\n{call}")).unwrap();
            events.extend(parser.finish().unwrap().events);
            assert_eq!(assemble(&events), text_and_call("Again.", "Paris"));
            assert!(parser.reset().is_empty());
        }
    }
}

#[test]
fn families_without_separator_framing_preserve_newlines_before_calls() {
    let input = "Check.\n\n<tool_call><function=run><parameter=value>Paris</parameter></function></tool_call>";
    let mut parser = create_unified_parser_for_family("qwen3", &[]).unwrap();
    parser
        .initialize_request(UnifiedParserInit {
            starting_state: UnifiedParserStartingState::Response,
            ..Default::default()
        })
        .unwrap();
    let mut events = parser.push(input).unwrap();
    events.extend(parser.finish().unwrap().events);
    assert_eq!(assemble(&events), text_and_call("Check.\n\n", "Paris"));
}
