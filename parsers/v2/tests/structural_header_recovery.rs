// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_parsers_v2::{
    Tool, ToolParseResult, UnifiedEvent, UnifiedParserExt, UnifiedParserInit,
    UnifiedParserStartingState, assemble, create_tool_parser_for_family,
    create_unified_parser_for_family,
};

const FAMILIES: [&str; 3] = ["glm47", "minimax_m3", "kimi_k3"];
const NS: &str = "]<]minimax[>[";

fn tools() -> [Tool; 1] {
    [Tool {
        name: "run".into(),
        description: None,
        parameters: serde_json::json!({
            "type": "object", "properties": {"cmd": {"type": "string"}}
        }),
        strict: None,
    }]
}

fn call(family: &str, name: &str, value: &str) -> String {
    match family {
        "glm47" => format!(
            "<tool_call>{name}<arg_key>cmd</arg_key><arg_value>{value}</arg_value></tool_call>"
        ),
        "minimax_m3" => format!(
            "{NS}<tool_call>{NS}<invoke name=\"{name}\">{NS}<cmd>{value}{NS}</cmd>{NS}</invoke>{NS}</tool_call>"
        ),
        "kimi_k3" => format!(
            "<|open|>tools<|sep|><|open|>call tool=\"{name}\" index=\"1\"<|sep|><|open|>argument key=\"cmd\" type=\"string\"<|sep|>{value}<|close|>argument<|sep|><|close|>call<|sep|><|close|>tools<|sep|>"
        ),
        _ => unreachable!(),
    }
}

fn opener(family: &str) -> &'static str {
    match family {
        "glm47" => "<tool_call>",
        "minimax_m3" => "]<]minimax[>[<tool_call>",
        "kimi_k3" => "<|open|>call",
        _ => unreachable!(),
    }
}

fn expected_call(name: &str, value: &str) -> UnifiedEvent {
    UnifiedEvent::ToolCall {
        name: name.into(),
        arguments: serde_json::json!({"cmd": value}),
    }
}

fn assert_legacy(result: ToolParseResult, expected: &[UnifiedEvent]) {
    let result = result.coalesce_calls();
    let text: String = expected
        .iter()
        .filter_map(|event| match event {
            UnifiedEvent::Text { text } | UnifiedEvent::Reasoning { text } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    let calls: Vec<_> = expected
        .iter()
        .filter_map(|event| match event {
            UnifiedEvent::ToolCall { name, arguments } => Some((name, arguments)),
            _ => None,
        })
        .collect();
    assert_eq!(result.normal_text, text);
    assert_eq!(result.calls.len(), calls.len());
    for (actual, (name, arguments)) in result.calls.iter().zip(calls) {
        assert!(actual.complete);
        assert_eq!(actual.name.as_ref(), Some(name));
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&actual.arguments).unwrap(),
            *arguments
        );
    }
}

fn assert_delivery(family: &str, chunks: &[&str], expected: &[UnifiedEvent]) {
    let mut parser = create_unified_parser_for_family(family, &tools()).unwrap();
    let init = UnifiedParserInit {
        starting_state: UnifiedParserStartingState::Response,
        ..Default::default()
    };
    parser.initialize_request(init.clone()).unwrap();
    let mut events = Vec::new();
    let mut legacy = create_tool_parser_for_family(family, &tools()).unwrap();
    let mut tool_result = ToolParseResult::default();
    for chunk in chunks {
        events.extend(parser.push(chunk).unwrap());
        tool_result.append(legacy.push(chunk).unwrap());
    }
    assert_eq!(
        assemble(&events),
        expected,
        "{family} before EOF: {chunks:?}"
    );
    assert_legacy(tool_result.clone(), expected);
    events.extend(parser.finish().unwrap().events);
    tool_result.append(legacy.finish().unwrap());
    assert_eq!(assemble(&events), expected, "{family}: {chunks:?}");
    assert_legacy(tool_result, expected);

    parser.reset();
    parser.initialize_request(init.clone()).unwrap();
    parser.push(opener(family)).unwrap();
    parser.reset();
    parser.initialize_request(init).unwrap();
    let fresh = call(family, "run", "fresh");
    let mut events = parser.push(&fresh).unwrap();
    events.extend(parser.finish().unwrap().events);
    assert_eq!(
        assemble(&events),
        [expected_call("run", "fresh")],
        "{family} reset after an unfinished header"
    );
}

fn every_delivery(family: &str, input: &str, expected: &[UnifiedEvent]) {
    assert_delivery(family, &[input], expected);
    for split in (0..=input.len()).filter(|at| input.is_char_boundary(*at)) {
        assert_delivery(family, &[&input[..split], &input[split..]], expected);
    }
    let characters: Vec<_> = input
        .char_indices()
        .map(|(at, ch)| &input[at..at + ch.len_utf8()])
        .collect();
    assert_delivery(family, &characters, expected);
}

#[test]
fn malformed_quoted_header_preserves_the_following_unknown_call() {
    for family in FAMILIES {
        let prefix = "The café literal \"";
        let input = format!(
            "{prefix}{}\" marker is prose.{}",
            opener(family),
            call(family, "外部.lookup-v2", "Paris")
        );
        every_delivery(
            family,
            &input,
            &[
                UnifiedEvent::Text {
                    text: prefix.into(),
                },
                expected_call("外部.lookup-v2", "Paris"),
            ],
        );
    }
}

#[test]
fn real_arguments_keep_native_opener_bytes() {
    for family in FAMILIES {
        let value = format!("café {} remains argument data", opener(family));
        every_delivery(
            family,
            &call(family, "run", &value),
            &[expected_call("run", &value)],
        );
    }
}

#[test]
fn escaped_minimax_name_is_dropped_with_its_owned_wrapper() {
    let wire = call("minimax_m3", "lookup", "x");
    let input = format!("{} trailing", serde_json::to_string(&wire).unwrap());
    every_delivery(
        "minimax_m3",
        &input,
        &[UnifiedEvent::Text {
            text: "\"\" trailing".into(),
        }],
    );
    let unquoted = wire.replace("name=\"lookup\"", "name=lookup");
    every_delivery("minimax_m3", &unquoted, &[expected_call("lookup", "x")]);
}

#[test]
fn kimi_malformed_header_preserves_spaced_wrappers_and_response_channels() {
    let wire = call("kimi_k3", "外部.lookup-v2", "Paris");
    let spaced = wire
        .replace("<|open|>tools<|sep|>", "<|open|> tools <|sep|>")
        .replace("<|close|>tools<|sep|>", "<|close|> tools <|sep|>");
    every_delivery(
        "kimi_k3",
        &format!("before<|open|>call malformed{spaced}"),
        &[
            UnifiedEvent::Text {
                text: "before".into(),
            },
            expected_call("外部.lookup-v2", "Paris"),
        ],
    );
    every_delivery(
        "kimi_k3",
        "<|open|>tools<|sep|><|open|>call malformed<|open|>response<|sep|>Visible café<|close|>response<|sep|><|close|>tools<|sep|>",
        &[UnifiedEvent::Text {
            text: "Visible café".into(),
        }],
    );
}
