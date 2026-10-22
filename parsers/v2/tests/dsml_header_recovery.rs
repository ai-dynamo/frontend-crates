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
fn declared_tool_header_requires_an_explicit_invoke_closer() {
    for wrapped in [false, true] {
        for closer in ["", "</｜DSML｜ invoke>"] {
            let input = if wrapped {
                // An unclosed calls wrapper is permitted at EOF.
                format!("<｜DSML｜ calls>\n{SAMPLE}{closer}")
            } else {
                format!("{SAMPLE}{closer}")
            };
            let expected = if closer.is_empty() {
                vec![]
            } else {
                vec![call()]
            };
            check(&input, &tools(), &expected);
            check(&format!("\n\n{input}"), &tools(), &expected);
            if wrapped {
                check(&format!("{input}</｜DSML｜ calls>"), &tools(), &expected);
            }
        }
    }
}

#[test]
fn recovered_headers_preserve_later_calls_and_surrounding_text() {
    let next = "<｜DSML｜ invoke name=\"Bash\"><｜DSML｜ parameter name=\"command\" string=\"true\">next</｜DSML｜ parameter></｜DSML｜ invoke>";
    let next_call = UnifiedEvent::ToolCall {
        name: "Bash".into(),
        arguments: json!({"command":"next"}),
    };
    for closer in ["", "</｜DSML｜ invoke>"] {
        let expected = if closer.is_empty() {
            vec![next_call.clone()]
        } else {
            vec![call(), next_call.clone()]
        };
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
    for closer in ["", "</｜DSML｜ invoke>"] {
        for input in [
            format!("```xml\n{SAMPLE}{closer}\n```"),
            format!("`{SAMPLE}{closer}`"),
            format!("\"{SAMPLE}{closer}\""),
        ] {
            check(
                &input,
                &tools(),
                &[UnifiedEvent::Text {
                    text: input.clone(),
                }],
            );
        }
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
fn recovery_waits_for_explicit_closure_and_reset_keeps_tool_definitions() {
    let mut parser = create_unified_parser_for_family("deepseek_v41", &tools()).unwrap();
    assert!(parser.push(SAMPLE).unwrap().is_empty());
    assert!(parser.finish().unwrap().assembled().is_empty());
    parser.reset();
    let partial = SAMPLE.strip_suffix("</｜DSML｜ parameter>").unwrap();
    assert!(parser.push(partial).unwrap().is_empty());
    assert!(parser.finish().unwrap().assembled().is_empty());
    parser.reset();
    assert!(parser.push(SAMPLE).unwrap().is_empty());
    assert_eq!(
        assemble(&parser.push("</｜DSML｜ invoke>").unwrap()),
        vec![call()]
    );
    assert!(parser.finish().unwrap().events.is_empty());
    for body in [
        "",
        "plain text",
        "<｜DSML｜ parameter name=\"command\" string=\"maybe\">x</｜DSML｜ parameter>",
    ] {
        let input = format!("<｜DSML｜ parameter name=\"Bash\">{body}</｜DSML｜ invoke>");
        check(&input, &tools(), &[]);
    }
}

#[test]
fn same_named_real_parameter_does_not_become_an_invocation() {
    let value = "<｜DSML｜ parameter name=\"command\">payload</｜DSML｜ parameter>";
    let real_parameter = format!("<｜DSML｜ parameter name=\"Bash\">{value}</｜DSML｜ parameter>");
    check(&real_parameter, &tools(), &[]);
    // A surviving invoke closer still cannot turn the outer parameter into a call.
    check(
        &format!("{real_parameter}</｜DSML｜ invoke>"),
        &tools(),
        &[],
    );
    let next = "<｜DSML｜ invoke name=\"Bash\">{\"command\":\"next\"}</｜DSML｜ invoke>";
    check(
        &format!("{real_parameter}</｜DSML｜ invoke>{next}"),
        &tools(),
        &[UnifiedEvent::ToolCall {
            name: "Bash".into(),
            arguments: json!({"command":"next"}),
        }],
    );
    for extra in [
        "</｜DSML｜ parameter>",
        "</｜DSML｜ parameter></｜DSML｜ parameter>",
    ] {
        check(&format!("{SAMPLE}{extra}</｜DSML｜ invoke>"), &tools(), &[]);
    }
    let mut declared = tools();
    declared.push(Tool {
        name: "SomeTool".into(),
        description: None,
        strict: None,
        parameters: json!({"type":"object","properties":{"Bash":{"type":"string"}}}),
    });
    let input = "<｜DSML｜ invoke name=\"SomeTool\"><｜DSML｜ parameter name=\"Bash\">value</｜DSML｜ parameter></｜DSML｜ invoke>";
    check(
        input,
        &declared,
        &[UnifiedEvent::ToolCall {
            name: "SomeTool".into(),
            arguments: json!({"Bash":"value"}),
        }],
    );
}
