// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::traits::{Tool, ToolCallDelta, ToolParseResult};
use crate::{
    UnifiedParserEvent, UnifiedParserExt, create_tool_parser_for_family,
    create_unified_parser_for_family,
};

pub(super) fn deliveries(input: &str) -> Vec<Vec<&str>> {
    let mut schedules = vec![
        vec![input],
        input
            .char_indices()
            .map(|(i, c)| &input[i..i + c.len_utf8()])
            .collect(),
    ];
    schedules.extend(
        (0..=input.len())
            .filter(|&i| input.is_char_boundary(i))
            .map(|i| vec![&input[..i], &input[i..]]),
    );
    schedules
}

fn assert_numeric_case(schema: &serde_json::Value, raw: &str, expected: &str) {
    let tools = vec![Tool {
        name: "get_weather".into(),
        description: None,
        strict: None,
        parameters: serde_json::json!({"type":"object", "properties":{"location":schema}}),
    }];
    let expected = ToolParseResult {
        normal_text: String::new(),
        calls: vec![ToolCallDelta {
            tool_index: 0,
            name: Some("get_weather".into()),
            arguments: format!(r#"{{"location":{expected}}}"#),
            complete: true,
        }],
    };
    for family in ["qwen3_coder", "minimax_m2"] {
        let input = if family == "qwen3_coder" {
            format!(
                "<tool_call><function=get_weather><parameter=location>{raw}</parameter></function></tool_call>"
            )
        } else {
            format!(
                "<minimax:tool_call><invoke name=\"get_weather\"><parameter name=\"location\">{raw}</parameter></invoke></minimax:tool_call>"
            )
        };
        for chunks in deliveries(&input) {
            let mut parser = create_tool_parser_for_family(family, &tools).expect("tool parser");
            let mut output = ToolParseResult::default();
            for chunk in &chunks {
                output.append(parser.push(chunk).expect("push"));
            }
            output.append(parser.finish().expect("finish"));
            assert_eq!(
                output.coalesce_calls(),
                expected,
                "{family}: {schema}, {raw}, {chunks:?}"
            );
            assert_eq!(parser.tool_call_id(0), None);
            if family == "qwen3_coder" {
                for alias in ["qwen3", "qwen3_coder"] {
                    let mut parser =
                        create_unified_parser_for_family(alias, &tools).expect("Unified parser");
                    let mut events = Vec::new();
                    for chunk in &chunks {
                        events.extend(parser.push(chunk).expect("push"));
                    }
                    events.extend(parser.finish().expect("finish").events);
                    let mut output = ToolParseResult::default();
                    for event in events {
                        match event {
                            UnifiedParserEvent::ToolCall(call) => output.calls.push(call),
                            UnifiedParserEvent::Text(text) => output.normal_text.push_str(&text),
                            UnifiedParserEvent::Reasoning(text) => {
                                panic!("unexpected reasoning: {text}")
                            }
                        }
                    }
                    // Keep raw argument bytes: assembling into Value would round large fractions.
                    assert_eq!(
                        output.coalesce_calls(),
                        expected,
                        "{alias}: {schema}, {raw}, {chunks:?}"
                    );
                    assert_eq!(parser.tool_call_id(0), None);
                }
            }
        }
    }
}

#[test]
fn integral_decimal_arguments_with_literal_constraints() {
    for keyword in ["const", "enum"] {
        for literal in [serde_json::json!(42), serde_json::json!(42.0)] {
            for ty in [
                serde_json::json!(["number", "null"]),
                serde_json::json!(["integer", "null"]),
                serde_json::json!(["number", "string"]),
            ] {
                let mut schema = serde_json::json!({"type":ty});
                schema[keyword] = if keyword == "const" {
                    literal.clone()
                } else {
                    serde_json::json!([literal])
                };
                for raw in ["42", "42.0", "4.2e1"] {
                    assert_numeric_case(&schema, raw, "42");
                }
            }
        }
    }
}

#[test]
fn integral_decimal_coercion_preserves_precision() {
    for (raw, expected) in [
        ("9007199254740993.0", "9007199254740993"),
        ("9.007199254740993e15", "9007199254740993"),
        ("-4.2E+1", "-42"),
        ("4200e-2", "42"),
        ("0.0e-400", "0"),
        ("42.0000000000000001", "\"42.0000000000000001\""),
        ("1e-400", "\"1e-400\""),
        ("42.5", "\"42.5\""),
    ] {
        assert_numeric_case(
            &serde_json::json!({"type":["integer","string"]}),
            raw,
            expected,
        );
    }
}
#[test]
fn large_fractional_arguments_remain_exact_json_numbers() {
    for raw in [
        "9007199254740992.5",
        "9007199254740993.1",
        "9007199254740993.25",
        "9.0071992547409925e15",
        "0.10000000000000000001",
        "-9007199254740992.5",
    ] {
        for schema in [
            serde_json::json!({"type":"number"}),
            serde_json::json!({"type":["number","null"],"const":serde_json::from_str::<serde_json::Value>(raw).unwrap()}),
            serde_json::json!({"type":["number","null"],"enum":[serde_json::from_str::<serde_json::Value>(raw).unwrap()]}),
        ] {
            assert_numeric_case(&schema, raw, raw);
        }
    }
}
