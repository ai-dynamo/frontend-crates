// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_parsers_v2::{
    Tool, ToolParseResult, UnifiedEvent, UnifiedParserExt, UnifiedParserInit,
    UnifiedParserStartingState, UnifiedToolOutputMode, assemble, create_tool_parser_for_family,
    create_unified_parser_for_family,
};

const FAMILIES: [(&str, &str); 4] = [
    ("qwen3", "</think>"),
    ("deepseek_v41", "</think>"),
    ("gemma4", "<channel|>"),
    ("kimi_k3", "<|close|>think<|sep|>"),
];

fn tools() -> Vec<Tool> {
    vec![Tool {
        name: "run".into(),
        description: None,
        parameters: serde_json::json!({"type":"object","properties":{"cmd":{"type":"string"}}}),
        strict: None,
    }]
}

fn native_call(family: &str, value: &str) -> String {
    match family {
        "qwen3" | "qwen3_coder" => format!(
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

fn every_delivery(input: &str, mut check: impl FnMut(&[&str])) {
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

fn init(state: UnifiedParserStartingState, mode: UnifiedToolOutputMode) -> UnifiedParserInit {
    UnifiedParserInit {
        starting_state: state,
        tool_output_mode: mode,
        ..Default::default()
    }
}

fn expect(family: &str, input: &str, init: UnifiedParserInit, expected: &[UnifiedEvent]) {
    every_delivery(input, |chunks| {
        let mut parser = create_unified_parser_for_family(family, &tools()).unwrap();
        parser.initialize_request(init.clone()).unwrap();
        let mut events = Vec::new();
        for chunk in chunks {
            events.extend(parser.push(chunk).unwrap());
        }
        events.extend(parser.finish().unwrap().events);
        assert_eq!(assemble(&events), expected, "{family} chunks={chunks:?}");
    });
}

fn reasoning(text: impl Into<String>) -> UnifiedEvent {
    UnifiedEvent::Reasoning { text: text.into() }
}

fn text(value: impl Into<String>) -> UnifiedEvent {
    UnifiedEvent::Text { text: value.into() }
}

fn call(value: &str) -> UnifiedEvent {
    UnifiedEvent::ToolCall {
        name: "run".into(),
        arguments: serde_json::json!({"cmd":value}),
    }
}

fn check_reasoning_closer(quote: char) {
    for (family, closer) in FAMILIES {
        for suffix in ['s', 'S'] {
            let thought = format!(
                "{quote}load_config{quote}'{suffix} check is fine, but `open('/tmp/x')` is not.\n"
            );
            let answer = "\n\nI've found the bug in `app.py`.";
            expect(
                family,
                &format!("{thought}{closer}{answer}"),
                init(
                    UnifiedParserStartingState::Reasoning,
                    UnifiedToolOutputMode::Native,
                ),
                &[reasoning(thought), text(answer)],
            );
        }
    }
}

#[test]
fn possessive_after_code_does_not_hide_the_reasoning_closer() {
    check_reasoning_closer('`');
}

#[test]
fn possessive_after_double_quoted_name_does_not_hide_the_reasoning_closer() {
    check_reasoning_closer('"');
}

#[test]
fn possessives_do_not_hide_native_or_guided_calls() {
    for (family, closer) in FAMILIES {
        for quote in ['`', '"'] {
            let prefix = format!("{quote}café{quote}'s check uses `open('/tmp/x')`. ");
            let suffix = " See `app.py`.";
            expect(
                family,
                &format!("{prefix}{}{suffix}", native_call(family, "ok")),
                init(
                    UnifiedParserStartingState::Response,
                    UnifiedToolOutputMode::Native,
                ),
                &[text(prefix.clone()), call("ok"), text(suffix)],
            );
            for named in [false, true] {
                let payload = if named {
                    r#"{"cmd":"ok"}"#
                } else {
                    r#"{"name":"run","arguments":{"cmd":"ok"}}"#
                };
                expect(
                    family,
                    &format!("{prefix}{closer}{payload}"),
                    init(
                        UnifiedParserStartingState::Reasoning,
                        UnifiedToolOutputMode::GuidedJson {
                            named_tool: named.then(|| "run".into()),
                        },
                    ),
                    &[reasoning(prefix.clone()), call("ok")],
                );
            }
        }
    }
}

#[test]
fn suffix_boundary_resolves_at_the_next_control() {
    for (family, closer) in FAMILIES {
        for quote in ['`', '"'] {
            for suffix in ['s', 'S'] {
                for punctuation in ["", ".", ","] {
                    let prefix = format!("{quote}café{quote}'{suffix}{punctuation}");
                    expect(
                        family,
                        &format!("{prefix}{closer}Answer"),
                        init(
                            UnifiedParserStartingState::Reasoning,
                            UnifiedToolOutputMode::Native,
                        ),
                        &[reasoning(prefix.clone()), text("Answer")],
                    );
                    expect(
                        family,
                        &format!("{prefix}{}", native_call(family, "ok")),
                        init(
                            UnifiedParserStartingState::Response,
                            UnifiedToolOutputMode::Native,
                        ),
                        &[text(prefix.clone()), call("ok")],
                    );
                    expect(
                        family,
                        &format!(
                            "{prefix}{closer}{{\"name\":\"run\",\"arguments\":{{\"cmd\":\"ok\"}}}}"
                        ),
                        init(
                            UnifiedParserStartingState::Reasoning,
                            UnifiedToolOutputMode::GuidedJson { named_tool: None },
                        ),
                        &[reasoning(prefix), call("ok")],
                    );
                }
            }
        }
    }
}

#[test]
fn contractions_and_genuine_quotes_after_possessives_keep_their_rules() {
    for (family, closer) in FAMILIES {
        for quote in ['`', '"'] {
            for tail in [
                " don't mistake a contraction.".to_string(),
                format!(" \"literal {closer}\" stays quoted."),
                format!(" 'literal {closer}' stays quoted."),
            ] {
                let thought = format!("{quote}name{quote}'s{tail}");
                expect(
                    family,
                    &format!("{thought}{closer}Answer"),
                    init(
                        UnifiedParserStartingState::Reasoning,
                        UnifiedToolOutputMode::Native,
                    ),
                    &[reasoning(thought), text("Answer")],
                );
            }
        }
    }
}

#[test]
fn non_possessive_and_separated_single_quotes_still_protect_controls() {
    for (family, closer) in FAMILIES {
        for quote in ['`', '"'] {
            for word in ["some", "s_name", "s2", "sβ"] {
                let thought = format!("{quote}name{quote}'{word} {closer}'");
                expect(
                    family,
                    &format!("{thought}{closer}Answer"),
                    init(
                        UnifiedParserStartingState::Reasoning,
                        UnifiedToolOutputMode::Native,
                    ),
                    &[reasoning(thought), text("Answer")],
                );
            }
            for separator in [" ", "\n"] {
                let thought = format!("{quote}name{quote}{separator}'s {closer}'");
                expect(
                    family,
                    &format!("{thought}{closer}Answer"),
                    init(
                        UnifiedParserStartingState::Reasoning,
                        UnifiedToolOutputMode::Native,
                    ),
                    &[reasoning(thought), text("Answer")],
                );
            }
            let visible = format!(
                "{quote}name{quote} 'doesn't mean {} is a call'",
                native_call(family, "ok")
            );
            expect(
                family,
                &visible,
                init(
                    UnifiedParserStartingState::Response,
                    UnifiedToolOutputMode::Native,
                ),
                &[text(visible.clone())],
            );
        }
    }
}

#[test]
fn escaped_quote_endings_do_not_establish_possessive_context() {
    for (family, closer) in FAMILIES {
        for quote in ['`', '"'] {
            let thought = format!("name\\{quote}'s {closer}'");
            expect(
                family,
                &format!("{thought}{closer}Answer"),
                init(
                    UnifiedParserStartingState::Reasoning,
                    UnifiedToolOutputMode::Native,
                ),
                &[reasoning(thought), text("Answer")],
            );
        }
    }
}

#[test]
fn suffix_eof_and_reset_preserve_bytes_and_release_context() {
    for (family, closer) in FAMILIES {
        for quote in ['`', '"'] {
            for suffix in ["", "'", "'s", "'S"] {
                let input = format!("{quote}café{quote}{suffix}");
                let request = init(
                    UnifiedParserStartingState::Reasoning,
                    UnifiedToolOutputMode::Native,
                );
                expect(family, &input, request.clone(), &[reasoning(input.clone())]);
                every_delivery(&input, |chunks| {
                    let mut parser = create_unified_parser_for_family(family, &tools()).unwrap();
                    parser.initialize_request(request.clone()).unwrap();
                    for chunk in chunks {
                        parser.push(chunk).unwrap();
                    }
                    let _ = parser.reset();
                    parser.initialize_request(request.clone()).unwrap();
                    let next = format!("'s {closer}'");
                    let mut events = Vec::new();
                    for character in next.chars() {
                        events.extend(parser.push(&character.to_string()).unwrap());
                    }
                    events.extend(parser.finish().unwrap().events);
                    assert_eq!(
                        assemble(&events),
                        [reasoning(next)],
                        "{family} reset after {input:?}"
                    );
                });
            }
        }
    }
}

#[test]
fn reasoning_boundary_clears_closed_span_context() {
    for (family, closer) in FAMILIES {
        for quote in ['`', '"'] {
            let thought = format!("{quote}name{quote}");
            let visible = format!("'s {}'", native_call(family, "ok"));
            expect(
                family,
                &format!("{thought}{closer}{visible}"),
                init(
                    UnifiedParserStartingState::Reasoning,
                    UnifiedToolOutputMode::Native,
                ),
                &[reasoning(thought), text(visible)],
            );
        }
    }
}

#[test]
fn quoted_names_and_markers_inside_tool_arguments_remain_data() {
    for (family, closer) in FAMILIES {
        let value = format!("echo \"café\"'s {closer} `quoted`");
        expect(
            family,
            &native_call(family, &value),
            init(
                UnifiedParserStartingState::Response,
                UnifiedToolOutputMode::Native,
            ),
            &[call(&value)],
        );
    }
}

#[test]
fn shared_tool_only_scanners_extract_calls_after_possessives() {
    for family in ["qwen3_coder", "minimax_m2"] {
        for quote in ['`', '"'] {
            let prefix = format!("{quote}café{quote}'s check uses `open('/tmp/x')`. ");
            let suffix = " See `app.py`.";
            let input = format!("{prefix}{}{suffix}", native_call(family, "ok"));
            every_delivery(&input, |chunks| {
                let mut parser = create_tool_parser_for_family(family, &tools()).unwrap();
                let mut result = ToolParseResult::default();
                for chunk in chunks {
                    result.append(parser.push(chunk).unwrap());
                }
                result.append(parser.finish().unwrap());
                let result = result.coalesce_calls();
                assert_eq!(
                    result.normal_text,
                    format!("{prefix}{suffix}"),
                    "{family} chunks={chunks:?}"
                );
                assert_eq!(result.calls.len(), 1, "{family} chunks={chunks:?}");
                assert_eq!(result.calls[0].name.as_deref(), Some("run"));
                assert_eq!(result.calls[0].arguments, r#"{"cmd":"ok"}"#);
            });
        }
    }
}
