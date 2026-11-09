// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::process::Command;

use dynamo_parsers_v2::{
    Tool, UnifiedEvent, UnifiedParserExt, UnifiedParserInit, UnifiedParserStartingState, assemble,
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
        _ => unreachable!(),
    }
}

fn parse<'a>(
    family: &str,
    chunks: impl IntoIterator<Item = &'a str>,
    starting_state: UnifiedParserStartingState,
) -> Vec<UnifiedEvent> {
    let tools = [Tool {
        name: "run".into(),
        description: None,
        parameters: serde_json::json!({"type":"object","properties":{"cmd":{"type":"string"}}}),
        strict: None,
    }];
    let mut parser = create_unified_parser_for_family(family, &tools).unwrap();
    parser
        .initialize_request(UnifiedParserInit {
            starting_state,
            ..UnifiedParserInit::default()
        })
        .unwrap();
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
    assert_eq!(parse(family, [input], starting_state), expected, "{family}");
    for split in (0..=input.len()).filter(|at| input.is_char_boundary(*at)) {
        assert_eq!(
            parse(family, [&input[..split], &input[split..]], starting_state),
            expected,
            "{family} split={split}: {input:?}"
        );
    }
    assert_eq!(
        parse(
            family,
            input
                .char_indices()
                .map(|(at, character)| &input[at..at + character.len_utf8()]),
            starting_state,
        ),
        expected,
        "{family} character chunks: {input:?}"
    );
}

fn check_policy(is_quote_aware: bool) {
    for (family, closer) in FAMILIES {
        if !is_quote_aware {
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

            let prefix = format!("Example: {quote}");
            let suffix = format!("{quote}.");
            let input = format!("{prefix}{}{suffix}", native_call(family, "echo hi"));
            let expected = if is_quote_aware {
                vec![UnifiedEvent::Text {
                    text: input.clone(),
                }]
            } else {
                vec![
                    UnifiedEvent::Text { text: prefix },
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
        (None, "quoted"),
        (Some("0"), "quoted"),
        (Some("1"), "controls"),
        (Some("true"), "controls"),
    ] {
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args(["--exact", "quote_policy_in_fresh_processes", "--nocapture"])
            .env(CHILD_MODE, mode)
            .env_remove(SETTING);
        if let Some(value) = value {
            command.env(SETTING, value);
        }
        let output = command.output().unwrap();
        assert!(
            output.status.success(),
            "{SETTING}={value:?}\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
