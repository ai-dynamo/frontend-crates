// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_parsers_v2::{
    ToolParseResult, UnifiedEvent, UnifiedParserExt, UnifiedParserInit, assemble,
    create_tool_parser_for_family, create_unified_parser_for_family,
};

const HEADER: &str =
    "<|tool_calls_section_begin|><|tool_call_begin|>functions.lookup:0<|tool_call_argument_begin|>";
const CALL_END: &str = "<|tool_call_end|>";
const SECTION_END: &str = "<|tool_calls_section_end|>";

fn assert_call(result: ToolParseResult, arguments: &str, text: &str) {
    let result = result.coalesce_calls();
    assert_eq!(result.calls.len(), 1, "complete call must precede EOF");
    let call = &result.calls[0];
    assert_eq!(call.tool_index, 0);
    assert_eq!(call.name.as_deref(), Some("lookup"));
    assert_eq!(call.arguments, arguments);
    assert!(call.complete);
    assert_eq!(result.normal_text, text);
}

fn assert_delivery(chunks: &[&str], expected: &[UnifiedEvent], arguments: &str, text: &str) {
    let mut unified = create_unified_parser_for_family("kimi_k2", &[]).unwrap();
    for _ in 0..2 {
        unified
            .initialize_request(UnifiedParserInit::default())
            .unwrap();
        let mut legacy = create_tool_parser_for_family("kimi_k2", &[]).unwrap();
        let mut events = Vec::new();
        let mut output = ToolParseResult::default();
        for chunk in chunks {
            events.extend(unified.push(chunk).unwrap());
            output.append(legacy.push(chunk).unwrap());
        }
        assert_call(
            ToolParseResult::from_deltas(events.clone()),
            arguments,
            text,
        );
        assert_call(output.clone(), arguments, text);
        assert_eq!(assemble(&events), expected, "before EOF: {chunks:?}");
        events.extend(unified.finish().unwrap().events);
        output.append(legacy.finish().unwrap());
        assert_eq!(assemble(&events), expected, "after EOF: {chunks:?}");
        assert_call(output, arguments, text);
        assert_eq!(unified.tool_call_id(0), Some("functions.lookup:0"));
        assert_eq!(legacy.tool_call_id(0), Some("functions.lookup:0"));
        unified.reset();
        assert_eq!(unified.tool_call_id(0), None);
        unified.push(&format!("{HEADER}{{\\")).unwrap();
        unified.reset();
    }
}

fn every_delivery(input: &str, expected: &[UnifiedEvent], arguments: &str, text: &str) {
    assert_delivery(&[input], expected, arguments, text);
    for split in (0..=input.len()).filter(|at| input.is_char_boundary(*at)) {
        assert_delivery(
            &[&input[..split], "", &input[split..]],
            expected,
            arguments,
            text,
        );
    }
    let characters: Vec<_> = input
        .char_indices()
        .map(|(at, ch)| &input[at..at + ch.len_utf8()])
        .collect();
    assert_delivery(&characters, expected, arguments, text);
}

#[test]
fn escaped_quoted_native_call_commits_before_eof() {
    let arguments = r#"{\"q\": \"x\"}"#;
    let input = format!("\"{HEADER}{arguments}{CALL_END}{SECTION_END}\" trailing");
    every_delivery(
        &input,
        &[
            UnifiedEvent::Text { text: "\"".into() },
            UnifiedEvent::ToolCall {
                name: "lookup".into(),
                arguments: serde_json::json!({}),
            },
            UnifiedEvent::Text {
                text: "\" trailing".into(),
            },
        ],
        arguments,
        "\"\" trailing",
    );
}

#[test]
fn recovery_preserves_root_strings_and_scalars() {
    for (arguments, value) in [
        (
            "\"München <|tool_call_end|> <|tool_calls_section_end|> literal\"",
            serde_json::json!("München <|tool_call_end|> <|tool_calls_section_end|> literal"),
        ),
        ("true", serde_json::json!(true)),
        ("42", serde_json::json!(42)),
        ("null", serde_json::Value::Null),
    ] {
        every_delivery(
            &format!("{HEADER}{arguments}{CALL_END}{SECTION_END}"),
            &[UnifiedEvent::ToolCall {
                name: "lookup".into(),
                arguments: value,
            }],
            arguments,
            "",
        );
    }
}

#[test]
fn invalid_escape_inside_owned_string_preserves_literal_markers() {
    let arguments = r#"{"q":"München \q <|tool_call_end|> <|tool_calls_section_end|> literal"}"#;
    every_delivery(
        &format!("{HEADER}{arguments}{CALL_END}{SECTION_END}"),
        &[UnifiedEvent::ToolCall {
            name: "lookup".into(),
            arguments: serde_json::json!({}),
        }],
        arguments,
        "",
    );
}
