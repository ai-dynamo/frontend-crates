// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Whitespace-only text in a tool-only turn (issue #406).
//!
//! Nemotron frames each Qwen3-Coder call with a newline on both sides. A Nemotron V3
//! turn that only calls tools must not return that framing as text; a turn with any
//! visible text keeps every whitespace byte. Every other family is unchanged: the
//! rule is scoped to the template where dropping the framing was measured to keep
//! prefix-cache reuse.

use dynamo_parsers_v2::{
    Tool, UnifiedEvent, UnifiedParserExt, UnifiedParserInit, UnifiedParserOutput,
    UnifiedParserStartingState, assemble, create_unified_parser_for_family,
};

/// The Nemotron V3 family and its alias.
const FAMILIES: [&str; 2] = ["nemotron_v3", "nemotron3"];

/// Families that share Nemotron's call grammar but not its template. Their output
/// is unchanged by this rule.
const UNCHANGED_FAMILIES: [&str; 2] = ["qwen3", "qwen3_coder"];

/// Recorded Nemotron 3.5 Lightning call for "Press digit 1 using the tool."
const DIGIT_CALL: &str = "<tool_call>\n<function=enter_digit>\n<parameter=digit>\n1\n</parameter>\n</function>\n</tool_call>";

/// The recorded tool-only turn, in the chunks the engine delivered.
const RECORDED_CHUNKS: [&str; 4] = [
    "\n",
    "<tool_call>\n<function=enter_digit>\n<parameter=digit>\n",
    "1\n</parameter>\n</function>\n</tool_call>",
    "\n",
];

fn tools() -> Vec<Tool> {
    vec![Tool {
        name: "enter_digit".into(),
        description: None,
        parameters: serde_json::json!({
            "type": "object",
            "properties": {"digit": {"type": "string"}},
            "required": ["digit"]
        }),
        strict: None,
    }]
}

fn digit_call(digit: &str) -> String {
    DIGIT_CALL.replace("\n1\n", &format!("\n{digit}\n"))
}

fn call(digit: &str) -> UnifiedEvent {
    UnifiedEvent::ToolCall {
        name: "enter_digit".into(),
        arguments: serde_json::json!({"digit": digit}),
    }
}

fn text(value: &str) -> UnifiedEvent {
    UnifiedEvent::Text { text: value.into() }
}

fn reason(value: &str) -> UnifiedEvent {
    UnifiedEvent::Reasoning { text: value.into() }
}

/// The whole input, every two-way split (with an empty middle chunk), and one
/// chunk per character.
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

fn run(family: &str, state: UnifiedParserStartingState, chunks: &[&str]) -> Vec<UnifiedEvent> {
    let mut parser = create_unified_parser_for_family(family, &tools()).unwrap();
    parser
        .initialize_request(UnifiedParserInit {
            starting_state: state,
            ..Default::default()
        })
        .unwrap();
    let mut output = UnifiedParserOutput::default();
    for chunk in chunks {
        parser.parse_into(chunk, &mut output).unwrap();
    }
    output.append(&mut parser.finish().unwrap());
    assemble(&output.events)
}

fn assert_every_split(
    family: &str,
    state: UnifiedParserStartingState,
    input: &str,
    expected: &[UnifiedEvent],
) {
    for chunks in chunkings(input) {
        assert_eq!(
            run(family, state, &chunks),
            expected,
            "{family}: {chunks:?}"
        );
    }
}

const RESPONSE: UnifiedParserStartingState = UnifiedParserStartingState::Response;

#[test]
fn recorded_tool_only_turn_emits_no_text() {
    for family in FAMILIES {
        assert_eq!(
            run(family, RESPONSE, &RECORDED_CHUNKS),
            [call("1")],
            "{family}"
        );
        assert_eq!(
            run(family, RESPONSE, &[&RECORDED_CHUNKS.concat()]),
            [call("1")],
            "{family}"
        );
    }
}

#[test]
fn tool_only_turn_emits_no_text_at_every_split() {
    let turn = format!("\n{DIGIT_CALL}\n");
    for family in FAMILIES {
        assert_every_split(family, RESPONSE, &turn, &[call("1")]);
    }
}

#[test]
fn whitespace_between_and_after_calls_is_dropped() {
    let turn = format!("\n{}\n\n{}\n", digit_call("1"), digit_call("2"));
    for family in FAMILIES {
        assert_every_split(family, RESPONSE, &turn, &[call("1"), call("2")]);
    }
    for family in FAMILIES {
        let mixed = format!(" \t\n{DIGIT_CALL}\r\n ");
        assert_eq!(run(family, RESPONSE, &[&mixed]), [call("1")], "{family}");
    }
}

#[test]
fn prose_before_the_call_keeps_every_whitespace_byte() {
    let turn = format!("Pressing it now.\n\n{DIGIT_CALL}\n");
    for family in FAMILIES {
        assert_every_split(
            family,
            RESPONSE,
            &turn,
            &[text("Pressing it now.\n\n"), call("1"), text("\n")],
        );
    }
    let leading = format!("\n\nPressing it now.\n{DIGIT_CALL}");
    for family in FAMILIES {
        assert_every_split(
            family,
            RESPONSE,
            &leading,
            &[text("\n\nPressing it now.\n"), call("1")],
        );
    }
}

/// Held whitespace returns in front of the visible text that decided it, so the
/// text channel is byte-identical; only its position relative to an earlier call
/// moves.
#[test]
fn prose_after_the_call_keeps_every_whitespace_byte() {
    let turn = format!("\n{DIGIT_CALL}\nDone.");
    for family in FAMILIES {
        assert_every_split(family, RESPONSE, &turn, &[call("1"), text("\n\nDone.")]);
    }
}

#[test]
fn whitespace_only_output_without_a_call_is_kept() {
    for family in FAMILIES {
        for input in ["\n\n", " \t\n", "\n"] {
            assert_every_split(family, RESPONSE, input, &[text(input)]);
        }
    }
}

#[test]
fn reasoning_then_a_tool_only_turn_keeps_reasoning_and_drops_framing() {
    let turn = format!("Press it.</think>\n\n{DIGIT_CALL}\n");
    for family in FAMILIES {
        assert_every_split(
            family,
            UnifiedParserStartingState::Reasoning,
            &turn,
            &[reason("Press it."), call("1")],
        );
    }
}

#[test]
fn hold_is_bounded_and_releases_past_the_cap() {
    const CAP: usize = 4096;
    for family in FAMILIES {
        let at_cap = format!("{}{DIGIT_CALL}", "\n".repeat(CAP));
        assert_eq!(run(family, RESPONSE, &[&at_cap]), [call("1")], "{family}");

        let past_cap = "\n".repeat(CAP + 1);
        let turn = format!("{past_cap}{DIGIT_CALL}");
        assert_eq!(
            run(family, RESPONSE, &[&turn]),
            [text(&past_cap), call("1")],
            "{family}"
        );
        // Delivered a byte at a time, holding stops at the byte that crosses the cap
        // and stays off for the rest of the stream.
        let bytes: Vec<String> = past_cap.chars().map(String::from).collect();
        let mut chunks: Vec<&str> = bytes.iter().map(String::as_str).collect();
        chunks.push(DIGIT_CALL);
        chunks.push("\n");
        assert_eq!(
            run(family, RESPONSE, &chunks),
            [text(&past_cap), call("1"), text("\n")],
            "{family}"
        );
    }
}

/// DSML parsers own their call framing (#405): whitespace that is not the
/// encoder's separator stays content. The rule does not touch them.
#[test]
fn dsml_families_keep_their_whitespace_contract() {
    for (family, call) in [
        (
            "deepseek_v4",
            "<｜DSML｜tool_calls><｜DSML｜invoke name=\"run\"><｜DSML｜parameter name=\"value\" string=\"true\">Paris</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>",
        ),
        (
            "deepseek_v41",
            "<｜DSML｜ calls><｜DSML｜ invoke name=\"run\"><｜DSML｜ parameter name=\"value\" string=\"true\">Paris</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>",
        ),
    ] {
        let expected_call = UnifiedEvent::ToolCall {
            name: "run".into(),
            arguments: serde_json::json!({"value": "Paris"}),
        };
        let mut parser = create_unified_parser_for_family(family, &[]).unwrap();
        parser
            .initialize_request(UnifiedParserInit {
                starting_state: RESPONSE,
                ..Default::default()
            })
            .unwrap();
        let input = format!("\n\n\n\n{call}");
        let events = parser.parse_complete(&input).unwrap();
        assert_eq!(events, [text("\n\n"), expected_call], "{family}");
    }
}

/// GPT-OSS ingests token ids; its framing text is unchanged on both input paths.
#[test]
fn gpt_oss_output_is_unchanged() {
    let native_call = "<|start|>assistant<|channel|>commentary to=functions.enter_digit <|constrain|>json<|message|>{\"digit\": \"1\"}<|call|>";
    let blank = "<|start|>assistant<|channel|>final<|message|>\n<|return|>";
    let turn = format!("{blank}{native_call}{blank}");
    let ids = dynamo_parsers_v2::unified::harmony::encode_harmony(&turn).unwrap();

    let mut parser = create_unified_parser_for_family("gpt_oss", &tools()).unwrap();
    parser.initialize(&[]).unwrap();
    let mut output = UnifiedParserOutput::default();
    parser.parse_token_ids_into(&ids, &mut output).unwrap();
    output.append(&mut parser.finish().unwrap());
    assert_eq!(
        assemble(&output.events),
        [text("\n"), call("1"), text("\n")]
    );

    let mut parser = create_unified_parser_for_family("gpt_oss", &tools()).unwrap();
    parser.initialize(&[]).unwrap();
    assert_eq!(
        parser.parse_complete(&turn).unwrap(),
        [text("\n"), call("1"), text("\n")]
    );
}

/// Qwen3 and Qwen3-Coder parse the same grammar but render content verbatim, so
/// the framing stays text, as before #406.
#[test]
fn qwen3_families_keep_framing_whitespace() {
    let turn = format!("\n{DIGIT_CALL}\n");
    let between = format!("\n{}\n\n{}\n", digit_call("1"), digit_call("2"));
    for family in UNCHANGED_FAMILIES {
        assert_eq!(
            run(family, RESPONSE, &RECORDED_CHUNKS),
            [text("\n"), call("1"), text("\n")],
            "{family}"
        );
        assert_every_split(
            family,
            RESPONSE,
            &turn,
            &[text("\n"), call("1"), text("\n")],
        );
        assert_every_split(
            family,
            RESPONSE,
            &between,
            &[text("\n"), call("1"), text("\n\n"), call("2"), text("\n")],
        );
        assert_every_split(
            family,
            UnifiedParserStartingState::Reasoning,
            &format!("Press it.</think>\n\n{DIGIT_CALL}\n"),
            &[reason("Press it."), text("\n\n"), call("1"), text("\n")],
        );
    }
}

/// Review finding: reset returns the wrapper's held whitespace with the inner
/// parser's unconsumed text, so recovery sees every byte that was not emitted.
#[test]
fn reset_returns_held_whitespace_before_the_inner_remainder() {
    for family in FAMILIES {
        let mut parser = create_unified_parser_for_family(family, &tools()).unwrap();
        parser
            .initialize_request(UnifiedParserInit {
                starting_state: RESPONSE,
                ..Default::default()
            })
            .unwrap();
        let mut output = UnifiedParserOutput::default();
        parser.parse_into("\n", &mut output).unwrap();
        parser.parse_into("<tool_c", &mut output).unwrap();
        assert!(output.is_empty(), "{family}");
        assert_eq!(parser.reset(), "\n<tool_c", "{family}");

        // The flags restart too: a fresh stream holds and settles again.
        parser
            .initialize_request(UnifiedParserInit {
                starting_state: RESPONSE,
                ..Default::default()
            })
            .unwrap();
        let mut output = UnifiedParserOutput::default();
        for chunk in RECORDED_CHUNKS {
            parser.parse_into(chunk, &mut output).unwrap();
        }
        output.append(&mut parser.finish().unwrap());
        assert_eq!(assemble(&output.events), [call("1")], "{family}");
    }
}

/// Review finding: only a completed call makes the turn a tool turn. A call that
/// never completes (truncated, or invalidated by a duplicate parameter after its
/// first value streamed) is dropped by `assemble`, so the whitespace around it is kept.
#[test]
fn whitespace_is_kept_when_no_call_completes() {
    let truncated = "<tool_call><function=enter_digit><parameter=digit>1";
    let duplicate = "<tool_call>\n<function=enter_digit>\n<parameter=digit>\n1\n</parameter>\n<parameter=digit>\n2\n</parameter>\n</function>\n</tool_call>";
    for family in FAMILIES {
        assert_eq!(
            run(family, RESPONSE, &["\n", truncated]),
            [text("\n")],
            "{family}: truncated"
        );
        let mut invalidated = 0;
        for split in (1..duplicate.len()).filter(|&at| duplicate.is_char_boundary(at)) {
            let chunks = ["\n", &duplicate[..split], &duplicate[split..], "\n"];
            let events = run(family, RESPONSE, &chunks);
            if events
                .iter()
                .any(|event| matches!(event, UnifiedEvent::ToolCall { .. }))
            {
                assert_eq!(events, [call("2")], "{family}: {chunks:?}");
            } else {
                invalidated += 1;
                assert_eq!(events, [text("\n\n")], "{family}: {chunks:?}");
            }
        }
        assert!(
            invalidated > 0,
            "{family}: no split invalidated the duplicate"
        );
    }
}
