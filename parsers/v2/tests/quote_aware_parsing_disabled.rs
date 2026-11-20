// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{ffi::OsStr, process::Command};

use dynamo_parsers_v2::{
    Tool, ToolParseResult, UnifiedEvent, UnifiedParser, UnifiedParserExt, UnifiedParserInit,
    UnifiedParserStartingState, assemble, create_tool_parser_for_family,
    create_unified_parser_for_family,
};

const SETTING: &str = "DYNAMO_PARSERS_DISABLE_QUOTE_AWARE_PARSING";
const CHILD_MODE: &str = "DYNAMO_PARSERS_QUOTE_POLICY_TEST_MODE";
const FAMILIES: [(&str, &str); 4] = [
    ("qwen3", "</think>"),
    ("deepseek_v41", "</think>"),
    ("gemma4", "<channel|>"),
    ("kimi_k3", "<|close|>think<|sep|>"),
];

fn native_call(family: &str, value: &str) -> String {
    match family {
        "qwen3" => format!(
            "<tool_call><function=run><parameter=cmd>{value}</parameter></function></tool_call>"
        ),
        "deepseek_v41" => format!(
            "<｜DSML｜ calls><｜DSML｜ invoke name=\"run\"><｜DSML｜ parameter name=\"cmd\" string=\"true\">{value}</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>"
        ),
        "gemma4" => format!("<|tool_call>call:run{{cmd:<|\"|>{value}<|\"|>}}<tool_call|>"),
        "kimi_k3" => format!(
            "<|open|>tools<|sep|><|open|>call tool=\"run\" index=\"1\"<|sep|><|open|>argument key=\"cmd\" type=\"string\"<|sep|>{value}<|close|>argument<|sep|><|close|>call<|sep|><|close|>tools<|sep|>"
        ),
        "minimax_m2" => format!(
            "<minimax:tool_call><invoke name=\"run\"><parameter name=\"cmd\">{value}</parameter></invoke></minimax:tool_call>"
        ),
        _ => unreachable!(),
    }
}

fn tools() -> [Tool; 1] {
    [Tool {
        name: "run".into(),
        description: None,
        parameters: serde_json::json!({"type":"object","properties":{"cmd":{"type":"string"}}}),
        strict: None,
    }]
}

fn parser(family: &str, starting_state: UnifiedParserStartingState) -> Box<dyn UnifiedParser> {
    let mut parser = create_unified_parser_for_family(family, &tools()).unwrap();
    parser
        .initialize_request(UnifiedParserInit {
            starting_state,
            ..UnifiedParserInit::default()
        })
        .unwrap();
    parser
}

fn parse<'a>(
    family: &str,
    chunks: impl IntoIterator<Item = &'a str>,
    starting_state: UnifiedParserStartingState,
) -> Vec<UnifiedEvent> {
    let mut parser = parser(family, starting_state);
    let mut events = Vec::new();
    for chunk in chunks {
        events.extend(parser.push(chunk).unwrap());
    }
    events.extend(parser.finish().unwrap().events);
    assemble(&events)
}

fn every_delivery(
    family: &str,
    input: &str,
    starting_state: UnifiedParserStartingState,
    expected: &[UnifiedEvent],
) {
    every_chunks(input, |chunks| {
        assert_eq!(
            parse(family, chunks.iter().copied(), starting_state),
            expected,
            "{family} chunks={chunks:?}"
        );
    });
}

fn every_chunks(input: &str, mut check: impl FnMut(&[&str])) {
    check(&[input]);
    for split in (0..=input.len()).filter(|at| input.is_char_boundary(*at)) {
        check(&[&input[..split], &input[split..]]);
    }
    check(
        &input
            .char_indices()
            .map(|(at, character)| &input[at..at + character.len_utf8()])
            .collect::<Vec<_>>(),
    );
}

fn check_policy(is_quote_aware: bool) {
    for (family, closer) in FAMILIES {
        {
            let reasoning = "`load_config`'s check is fine, but `open('/tmp/x')` is not.\n";
            let text = "\n\nI've found the bug in `app.py`.";
            every_delivery(
                family,
                &format!("{reasoning}{closer}{text}"),
                UnifiedParserStartingState::Reasoning,
                &[
                    UnifiedEvent::Reasoning {
                        text: reasoning.into(),
                    },
                    UnifiedEvent::Text { text: text.into() },
                ],
            );
        }

        for (reasoning, text) in [
            ("analysis \"", "Answer\""),
            (
                r"re.split(r'(\s)', text). Let's final.",
                "The code prints: ['café', ' ', 'b']",
            ),
        ] {
            let input = format!("{reasoning}{closer}{text}");
            let expected = if is_quote_aware {
                vec![UnifiedEvent::Reasoning {
                    text: input.clone(),
                }]
            } else {
                vec![
                    UnifiedEvent::Reasoning {
                        text: reasoning.into(),
                    },
                    UnifiedEvent::Text { text: text.into() },
                ]
            };
            every_delivery(
                family,
                &input,
                UnifiedParserStartingState::Reasoning,
                &expected,
            );
        }

        for quote in ['"', '\'', '`'] {
            let prefix = format!("The café marker {quote}");
            let suffix = format!("{quote} is literal.");
            let input = format!("{prefix}{closer}{suffix}");
            let expected = if is_quote_aware {
                vec![UnifiedEvent::Reasoning {
                    text: input.clone(),
                }]
            } else {
                vec![
                    UnifiedEvent::Reasoning { text: prefix },
                    UnifiedEvent::Text { text: suffix },
                ]
            };
            every_delivery(
                family,
                &input,
                UnifiedParserStartingState::Reasoning,
                &expected,
            );

            check_progress_and_reset(family, closer, quote, is_quote_aware);

            let prefix = format!("Example: {quote}");
            let suffix = format!("{quote}.");
            let input = format!("{prefix}{}{suffix}", native_call(family, "echo hi"));
            let expected = if is_quote_aware {
                vec![UnifiedEvent::Text {
                    text: input.clone(),
                }]
            } else {
                vec![
                    UnifiedEvent::Text {
                        text: prefix.clone(),
                    },
                    UnifiedEvent::ToolCall {
                        name: "run".into(),
                        arguments: serde_json::json!({"cmd":"echo hi"}),
                    },
                    UnifiedEvent::Text { text: suffix },
                ]
            };
            every_delivery(
                family,
                &input,
                UnifiedParserStartingState::Response,
                &expected,
            );

            let partial = format!("{prefix}{}", native_call(family, "echo hi"));
            every_chunks(&partial, |chunks| {
                let mut parser = parser(family, UnifiedParserStartingState::Response);
                let mut events = Vec::new();
                for chunk in chunks {
                    events.extend(parser.push(chunk).unwrap());
                }
                let events = assemble(&events);
                if is_quote_aware {
                    assert!(
                        !events
                            .iter()
                            .any(|event| matches!(event, UnifiedEvent::ToolCall { .. })),
                        "{family} compatibility call awaits quote resolution"
                    );
                } else {
                    assert_eq!(
                        events,
                        expected[..2],
                        "{family} complete call before closing quote or EOF chunks={chunks:?}"
                    );
                }
            });
        }

        let value = format!("echo \"{closer}\" `quoted`");
        every_delivery(
            family,
            &native_call(family, &value),
            UnifiedParserStartingState::Response,
            &[UnifiedEvent::ToolCall {
                name: "run".into(),
                arguments: serde_json::json!({"cmd":value}),
            }],
        );
    }
    check_tool_only_policy(is_quote_aware);
}

fn check_progress_and_reset(family: &str, closer: &str, quote: char, is_quote_aware: bool) {
    let prefix = format!("analysis {quote}");
    let partial = format!("{prefix}{closer}Answer");
    // The closing quote arrives later. The default policy must publish the answer
    // at this pause, rather than wait for that quote or end-of-stream.
    for split in (0..=partial.len()).filter(|at| partial.is_char_boundary(*at)) {
        let mut parser = parser(family, UnifiedParserStartingState::Reasoning);
        let mut events = parser.push(&partial[..split]).unwrap();
        events.extend(parser.push(&partial[split..]).unwrap());
        if !is_quote_aware {
            assert_eq!(
                assemble(&events),
                [
                    UnifiedEvent::Reasoning {
                        text: prefix.clone()
                    },
                    UnifiedEvent::Text {
                        text: "Answer".into()
                    },
                ],
                "{family} progress before closing quote split={split}"
            );
            assert!(parser.reset().is_empty(), "{family} committed boundary");
        } else {
            assert!(
                !assemble(&events)
                    .iter()
                    .any(|event| matches!(event, UnifiedEvent::Text { .. })),
                "{family} compatibility quote remains unresolved"
            );
            assert_eq!(
                parser.reset(),
                format!("{closer}Answer"),
                "{family} compatibility reset returns uncommitted control"
            );
        }
        parser
            .initialize_request(UnifiedParserInit {
                starting_state: UnifiedParserStartingState::Reasoning,
                ..Default::default()
            })
            .unwrap();
        let next = format!("fresh{closer}Visible");
        let mut events = parser.push(&next).unwrap();
        events.extend(parser.finish().unwrap().events);
        assert_eq!(
            assemble(&events),
            [
                UnifiedEvent::Reasoning {
                    text: "fresh".into()
                },
                UnifiedEvent::Text {
                    text: "Visible".into()
                },
            ],
            "{family} reset and reinitialize"
        );
    }

    let mut parser = parser(family, UnifiedParserStartingState::Reasoning);
    let mut events = Vec::new();
    for character in partial.chars() {
        events.extend(parser.push(&character.to_string()).unwrap());
    }
    if !is_quote_aware {
        assert_eq!(
            assemble(&events),
            [
                UnifiedEvent::Reasoning {
                    text: prefix.clone()
                },
                UnifiedEvent::Text {
                    text: "Answer".into()
                },
            ],
            "{family} character-chunk progress before EOF"
        );
    }
    events.extend(parser.finish().unwrap().events);
    assert_eq!(
        assemble(&events),
        [
            UnifiedEvent::Reasoning { text: prefix },
            UnifiedEvent::Text {
                text: "Answer".into()
            },
        ],
        "{family} unfinished quote at EOF"
    );
}

fn check_tool_only_policy(is_quote_aware: bool) {
    for (entry_point, family) in [
        ("qwen3_coder", "qwen3"),
        ("gemma4", "gemma4"),
        ("kimi_k3", "kimi_k3"),
        ("minimax_m2", "minimax_m2"),
    ] {
        for quote in ['"', '\'', '`'] {
            let prefix = format!("Example: {quote}");
            let suffix = format!("{quote}.");
            let input = format!("{prefix}{}{suffix}", native_call(family, "echo hi"));
            every_chunks(&input, |chunks| {
                let mut parser = create_tool_parser_for_family(entry_point, &tools()).unwrap();
                let mut result = ToolParseResult::default();
                for chunk in chunks {
                    result.append(parser.push(chunk).unwrap());
                }
                result.append(parser.finish().unwrap());
                let result = result.coalesce_calls();
                if is_quote_aware {
                    assert_eq!(result.normal_text, input, "{entry_point}");
                    assert!(result.calls.is_empty(), "{entry_point}");
                } else {
                    assert_eq!(
                        result.normal_text,
                        format!("{prefix}{suffix}"),
                        "{entry_point}"
                    );
                    assert_eq!(result.calls.len(), 1, "{entry_point}");
                    assert_eq!(result.calls[0].name.as_deref(), Some("run"));
                    assert_eq!(result.calls[0].arguments, r#"{"cmd":"echo hi"}"#);
                }
            });
        }

        let value = "echo \"</think>\" `quoted`";
        every_chunks(&native_call(family, value), |chunks| {
            let mut parser = create_tool_parser_for_family(entry_point, &tools()).unwrap();
            let mut result = ToolParseResult::default();
            for chunk in chunks {
                result.append(parser.push(chunk).unwrap());
            }
            result.append(parser.finish().unwrap());
            let result = result.coalesce_calls();
            assert!(result.normal_text.is_empty(), "{entry_point}");
            assert_eq!(result.calls.len(), 1, "{entry_point}");
            assert_eq!(result.calls[0].name.as_deref(), Some("run"));
            assert_eq!(
                serde_json::from_str::<serde_json::Value>(&result.calls[0].arguments).unwrap(),
                serde_json::json!({"cmd": value}),
                "{entry_point} argument markers remain data"
            );
        });
    }
}

fn check_configuration(value: Option<&OsStr>, mode: &str) {
    let mut command = Command::new(std::env::current_exe().unwrap());
    command
        .args(["--exact", "quote_policy_in_fresh_processes", "--nocapture"])
        .env(CHILD_MODE, mode)
        .env_remove(SETTING);
    if let Some(value) = value {
        command.env(SETTING, value);
    }
    let output = command.output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success() && stdout.lines().any(|line| line == "running 1 test"),
        "{SETTING}={value:?}\nstdout:\n{stdout}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn quote_policy_in_fresh_processes() {
    if let Ok(mode) = std::env::var(CHILD_MODE) {
        check_policy(match mode.as_str() {
            "quoted" => true,
            "controls" => false,
            _ => panic!("unexpected child mode: {mode}"),
        });
        return;
    }

    // The setting is cached globally; configure each process before any parser runs.
    for (value, mode) in [
        (None, "controls"),
        (Some("0"), "quoted"),
        (Some("false"), "quoted"),
        (Some("off"), "quoted"),
        (Some("no"), "quoted"),
        (Some(""), "quoted"),
        (Some("unexpected"), "quoted"),
        (Some(" true "), "quoted"),
        (Some("1"), "controls"),
        (Some("true"), "controls"),
        (Some("TRUE"), "controls"),
        (Some("on"), "controls"),
        (Some("YeS"), "controls"),
    ] {
        check_configuration(value.map(OsStr::new), mode);
    }
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        check_configuration(Some(OsStr::from_bytes(b"\xff")), "quoted");
    }
}
