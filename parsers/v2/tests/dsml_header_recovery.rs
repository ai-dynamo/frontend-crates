// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_parsers_v2::{
    Tool, UnifiedEvent, UnifiedParserExt, UnifiedParserInit, UnifiedParserStartingState, assemble,
    create_unified_parser_for_family,
};
use serde_json::json;

const SAMPLE: &str = "<｜DSML｜ parameter  name=\"Bash\">\n\
<｜DSML｜ parameter name=\"command\" string=\"true\">pwd && ls -a</｜DSML｜ parameter>\n\
<｜DSML｜ parameter name=\"description\" string=\"true\">Show cwd listing</｜DSML｜ parameter>\n\
<｜DSML｜ parameter name=\"command_ine\">x</｜DSML｜ parameter>\n\
<｜DSML｜ parameter name=\"dangerouslyDisableSandbox\" string=\"false\">false</｜DSML｜ parameter>";

fn tools() -> Vec<Tool> {
    vec![Tool {
        name: "Bash".into(),
        description: None,
        parameters: json!({"type":"object","properties":{"command":{"type":"string"}}}),
        strict: None,
    }]
}

fn call() -> UnifiedEvent {
    UnifiedEvent::ToolCall {
        name: "Bash".into(),
        arguments: json!({
            "command":"pwd && ls -a", "description":"Show cwd listing",
            "command_ine":"x", "dangerouslyDisableSandbox":false
        }),
    }
}

fn check(input: &str, tools: &[Tool], expected: &[UnifiedEvent]) {
    let boundaries: Vec<_> = input
        .char_indices()
        .map(|(at, _)| at)
        .chain([input.len()])
        .collect();
    let mut splits: Vec<Vec<&str>> = boundaries
        .iter()
        .map(|&at| vec![&input[..at], &input[at..]])
        .collect();
    splits.push(boundaries.windows(2).map(|w| &input[w[0]..w[1]]).collect());
    for chunks in splits {
        for state in [
            UnifiedParserStartingState::None,
            UnifiedParserStartingState::Response,
        ] {
            let mut parser = create_unified_parser_for_family("deepseek_v41", tools).unwrap();
            parser
                .initialize_request(UnifiedParserInit {
                    starting_state: state,
                    ..Default::default()
                })
                .unwrap();
            let mut events = Vec::new();
            for chunk in &chunks {
                events.extend(parser.push(chunk).unwrap());
            }
            events.extend(parser.finish().unwrap().events);
            assert_eq!(
                assemble(&events),
                expected,
                "state {state:?}, chunks {chunks:?}"
            );
        }
    }
}

#[test]
fn declared_tool_header_recovers_with_or_without_wrappers_and_closer() {
    for wrapped in [false, true] {
        for closer in ["", "</｜DSML｜ invoke>"] {
            let input = if wrapped {
                // An unclosed calls wrapper is permitted at EOF.
                format!("<｜DSML｜ calls>\n{SAMPLE}{closer}")
            } else {
                format!("{SAMPLE}{closer}")
            };
            check(&input, &tools(), &[call()]);
            check(&format!("\n\n{input}"), &tools(), &[call()]);
            if wrapped {
                check(&format!("{input}</｜DSML｜ calls>"), &tools(), &[call()]);
            }
        }
    }
}

#[test]
fn recovered_headers_preserve_later_calls_and_surrounding_text() {
    let next = "<｜DSML｜ invoke name=\"Bash\"><｜DSML｜ parameter name=\"command\" string=\"true\">next</｜DSML｜ parameter></｜DSML｜ invoke>";
    let expected = [
        call(),
        UnifiedEvent::ToolCall {
            name: "Bash".into(),
            arguments: json!({"command":"next"}),
        },
    ];
    for closer in ["", "</｜DSML｜ invoke>"] {
        check(&format!("{SAMPLE}{closer}{next}"), &tools(), &expected);
    }
    check(
        &format!("before\n{SAMPLE}</｜DSML｜ invoke>"),
        &tools(),
        &[
            UnifiedEvent::Text {
                text: "before\n".into(),
            },
            call(),
        ],
    );
}

#[test]
fn tool_names_are_exact_and_regular_parameters_and_quoted_examples_are_preserved() {
    check(
        SAMPLE,
        &[],
        &[UnifiedEvent::Text {
            text: SAMPLE.into(),
        }],
    );
    let unknown = SAMPLE.replacen("name=\"Bash\"", "name=\"bash\"", 1);
    check(
        &unknown,
        &tools(),
        &[UnifiedEvent::Text {
            text: unknown.clone(),
        }],
    );
    let parameter = "<｜DSML｜ parameter name=\"Bash\" string=\"true\">value</｜DSML｜ parameter>";
    check(
        parameter,
        &tools(),
        &[UnifiedEvent::Text {
            text: parameter.into(),
        }],
    );
    for input in [
        format!("```xml\n{SAMPLE}\n```"),
        format!("`{SAMPLE}`"),
        format!("\"{SAMPLE}\""),
    ] {
        check(
            &input,
            &tools(),
            &[UnifiedEvent::Text {
                text: input.clone(),
            }],
        );
    }
    let nested = "<｜DSML｜ parameter  name=\"Bash\">";
    let input = format!(
        "<｜DSML｜ invoke name=\"Bash\"><｜DSML｜ parameter name=\"command\" string=\"true\">{nested}</｜DSML｜ parameter></｜DSML｜ invoke>"
    );
    check(
        &input,
        &tools(),
        &[UnifiedEvent::ToolCall {
            name: "Bash".into(),
            arguments: json!({"command":nested}),
        }],
    );
    let input = "<｜DSML｜ invoke name=\"Bash\"><｜DSML｜ parameter name=\"Bash\">x</｜DSML｜ parameter></｜DSML｜ invoke>";
    check(
        input,
        &tools(),
        &[UnifiedEvent::ToolCall {
            name: "Bash".into(),
            arguments: json!({"Bash":"x"}),
        }],
    );
    let input = "<｜DSML｜ invoke name=\"undeclared\">{}</｜DSML｜ invoke>";
    check(
        input,
        &tools(),
        &[UnifiedEvent::ToolCall {
            name: "undeclared".into(),
            arguments: json!({}),
        }],
    );
}

#[test]
fn eof_recovery_waits_for_complete_parameters_and_reset_keeps_tool_definitions() {
    let mut parser = create_unified_parser_for_family("deepseek_v41", &tools()).unwrap();
    assert!(parser.push(SAMPLE).unwrap().is_empty());
    assert_eq!(parser.finish().unwrap().assembled(), vec![call()]);
    parser.reset();
    let partial = SAMPLE.strip_suffix("</｜DSML｜ parameter>").unwrap();
    assert!(parser.push(partial).unwrap().is_empty());
    assert!(parser.finish().unwrap().assembled().is_empty());
    parser.reset();
    assert!(parser.push(SAMPLE).unwrap().is_empty());
    assert_eq!(parser.finish().unwrap().assembled(), vec![call()]);
    for body in [
        "",
        "plain text",
        "<｜DSML｜ parameter name=\"command\" string=\"maybe\">x</｜DSML｜ parameter>",
    ] {
        let input = format!("<｜DSML｜ parameter name=\"Bash\">{body}</｜DSML｜ invoke>");
        check(&input, &tools(), &[]);
    }
}
