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

fn tool_result(events: Vec<UnifiedParserEvent>) -> ToolParseResult {
    let mut output = ToolParseResult::default();
    for event in events {
        match event {
            UnifiedParserEvent::ToolCall(call) => output.calls.push(call),
            UnifiedParserEvent::Text(text) => output.normal_text.push_str(&text),
            UnifiedParserEvent::Reasoning(text) => panic!("unexpected reasoning: {text}"),
            UnifiedParserEvent::MessageStart(_) | UnifiedParserEvent::MessageEnd => {}
        }
    }
    output.coalesce_calls()
}

fn assert_native_raw_case(family: &str, input: &str, arguments: &str) {
    let tools = [Tool {
        name: "echo".into(),
        description: None,
        strict: None,
        parameters: serde_json::json!({"type":"object","properties":{
            "n":{"type":["number","null"]}, "payload":{"type":"object"}
        }}),
    }];
    let expected = ToolParseResult {
        normal_text: String::new(),
        calls: vec![ToolCallDelta {
            tool_index: 0,
            name: Some("echo".into()),
            arguments: arguments.into(),
            complete: true,
        }],
    };
    assert_native_result(family, input, &tools, expected);
}

fn assert_native_result(family: &str, input: &str, tools: &[Tool], expected: ToolParseResult) {
    for chunks in deliveries(input) {
        let mut parser = create_unified_parser_for_family(family, tools).unwrap();
        let mut events = Vec::new();
        for chunk in &chunks {
            events.extend(parser.push(chunk).unwrap());
        }
        events.extend(parser.finish().unwrap().events);
        assert_eq!(tool_result(events), expected, "{family}: {chunks:?}");
        let mut parser = create_tool_parser_for_family(family, tools).unwrap();
        let mut output = ToolParseResult::default();
        for chunk in &chunks {
            output.append(parser.push(chunk).unwrap());
        }
        output.append(parser.finish().unwrap());
        assert_eq!(
            output.coalesce_calls(),
            expected,
            "legacy {family}: {chunks:?}"
        );
    }
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
                            UnifiedParserEvent::MessageStart(_)
                            | UnifiedParserEvent::MessageEnd => {}
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
        let literal: serde_json::Value = serde_json::from_str(raw).expect("numeric schema value");
        for schema in [
            serde_json::json!({"type":"number"}),
            serde_json::json!({"type":["number","null"], "const":literal.clone()}),
            serde_json::json!({"type":["number","null"], "enum":[literal]}),
        ] {
            assert_numeric_case(&schema, raw, raw);
        }
    }
}

#[cfg(not(feature = "arbitrary-precision"))]
#[test]
fn default_json_numbers_serialize_as_native_scalars() {
    // YAML uses an independent Serde serializer, like Pythonizer and MessagePack
    // consumers; serde_json's private arbitrary-precision struct must not escape.
    for (raw, expected) in [
        ("512", serde_yaml::to_value(512_u64).unwrap()),
        ("-512", serde_yaml::to_value(-512_i64).unwrap()),
        ("0.5", serde_yaml::to_value(0.5_f64).unwrap()),
    ] {
        let value: serde_json::Value = serde_json::from_str(raw).unwrap();
        assert_eq!(serde_yaml::to_value(value).unwrap(), expected, "{raw}");
    }
}

#[test]
fn guided_numeric_arguments_remain_exact_at_every_split() {
    let arguments = r#"{"large":18446744073709551617,"fraction":9007199254740992.5,"nested":[1e-400,{"overflow":1e9999}]}"#;
    for family in ["qwen3", "deepseek_v4", "deepseek_v41", "kimi_k2"] {
        for policy in [
            crate::InvalidGuidedPayloadPolicy::RecoverAsText,
            crate::InvalidGuidedPayloadPolicy::Reject,
            crate::InvalidGuidedPayloadPolicy::StreamBestEffort,
        ] {
            for payload in [
                arguments.to_string(),
                format!(r#"[{{"name":"echo","arguments":{arguments}}}]"#),
                format!(r#"[{{"name":"echo","parameters":{arguments}}}]"#),
            ] {
                let named = payload.starts_with('{');
                for chunks in deliveries(&payload) {
                    let mut parser = create_unified_parser_for_family(family, &[]).unwrap();
                    parser
                        .initialize_request(crate::UnifiedParserInit {
                            tool_output_mode: crate::UnifiedToolOutputMode::GuidedJson {
                                named_tool: named.then(|| "echo".into()),
                            },
                            invalid_guided_payload: policy,
                            ..Default::default()
                        })
                        .unwrap();
                    let mut events = Vec::new();
                    for chunk in chunks {
                        events.extend(parser.push(chunk).unwrap());
                    }
                    assert!(
                        events.iter().any(|event| matches!(event,
                        UnifiedParserEvent::ToolCall(delta) if delta.complete)),
                        "{family}: {policy:?}: completed payload must commit before EOF"
                    );
                    let tail = parser.push("after").unwrap();
                    assert_eq!(
                        tool_result(tail.clone()).normal_text,
                        "after",
                        "{family}: {policy:?}: trailing text must route before EOF"
                    );
                    events.extend(tail);
                    events.extend(parser.finish().unwrap().events);
                    assert_eq!(
                        tool_result(events),
                        ToolParseResult {
                            normal_text: "after".into(),
                            calls: vec![ToolCallDelta {
                                tool_index: 0,
                                name: Some("echo".into()),
                                arguments: arguments.into(),
                                complete: true,
                            }],
                        },
                        "{family}: {policy:?}, {payload}"
                    );
                }
            }
        }
    }
}

#[test]
fn kimi_k3_json_numeric_payload_remains_exact_at_every_split() {
    let arguments = r#"{"n":1e9999,"payload":{"fraction":9007199254740992.5,"underflow":1e-400}}"#;
    let body = format!("<|open|>json<|sep|>{arguments}<|close|>json<|sep|>");
    let input = format!(
        "<|open|>tools<|sep|><|open|>call tool=\"echo\" index=\"0\"<|sep|>{body}<|close|>call<|sep|><|close|>tools<|sep|>"
    );
    assert_native_raw_case("kimi_k3", &input, arguments);
}

#[test]
fn kimi_k3_typed_numeric_payload_remains_exact_at_every_split() {
    let input = "<|open|>tools<|sep|><|open|>call tool=\"echo\" index=\"0\"<|sep|><|open|>argument key=\"n\" type=\"number\"<|sep|>1e9999<|close|>argument<|sep|><|close|>call<|sep|><|close|>tools<|sep|>";
    assert_native_raw_case("kimi_k3", input, r#"{"n":1e9999}"#);
}

#[test]
fn kimi_k2_numeric_payload_owns_embedded_closer_at_every_split() {
    let arguments = r#"{"n":1e9999,"payload":{"text":"<|tool_call_end|>"}}"#;
    let input = format!(
        "<|tool_calls_section_begin|><|tool_call_begin|>functions.echo:0<|tool_call_argument_begin|>{arguments}<|tool_call_end|><|tool_calls_section_end|>"
    );
    assert_native_raw_case("kimi_k2", &input, arguments);
}

#[test]
fn muse_numeric_payload_owns_embedded_closer_at_every_split() {
    let payload = r#"{"text":"</atem:parameter></atem:invoke>","n":1e9999}"#;
    let input = format!(
        "<|start|>assistant to=echo<|message|><atem:function_calls><atem:invoke name=\"echo\"><atem:parameter name=\"payload\">{payload}</atem:parameter></atem:invoke></atem:function_calls><|eom|>"
    );
    assert_native_raw_case(
        "muse_glimmer",
        &input,
        &format!(r#"{{"payload":{payload}}}"#),
    );
    let malformed = r#"{"n":1eX}"#;
    let input = input.replace(payload, malformed);
    let arguments = serde_json::to_string(&serde_json::json!({"payload":malformed})).unwrap();
    assert_native_raw_case("muse_glimmer", &input, &arguments);
}

#[test]
fn glm_numeric_union_preserves_nonintegral_overflow_at_every_split() {
    // A non-integral number beyond f64 avoids the integer hint's separate path.
    let raw = format!("1.{}1e309", "0".repeat(399));
    let input =
        format!("<tool_call>echo<arg_key>n</arg_key><arg_value>{raw}</arg_value></tool_call>");
    assert_native_raw_case("glm47", &input, &format!(r#"{{"n":{raw}}}"#));
}

#[test]
fn muse_malformed_json_recovers_before_two_valid_calls_at_every_split() {
    let values = [
        ("bad", r#"{"x":"unfinished"#),
        ("echo", "é"),
        ("echo", "Café"),
    ];
    let input = values.iter().map(|(name, value)| {
        format!("<|start|>assistant to={name}<|message|><atem:function_calls>\n<atem:invoke name=\"{name}\">\n<atem:parameter name=\"value\">{value}</atem:parameter>\n</atem:invoke>\n</atem:function_calls><|eom|>")
    }).collect::<String>();
    let expected = ToolParseResult {
        normal_text: String::new(),
        calls: values
            .iter()
            .enumerate()
            .map(|(tool_index, (name, value))| ToolCallDelta {
                tool_index,
                name: Some((*name).into()),
                arguments: serde_json::to_string(&serde_json::json!({"value":value})).unwrap(),
                complete: true,
            })
            .collect(),
    };
    for family in ["muse_glimmer", "muse_spark"] {
        assert_native_result(family, &input, &[], expected.clone());
    }
}

// Exercise both Cargo feature configurations: raw tokens stay identical, but
// only the default build projects out-of-range numbers to strings.
#[test]
fn unified_numeric_assembly_preserves_siblings_at_every_split() {
    let tools = [Tool {
        name: "echo".into(),
        description: None,
        strict: None,
        parameters: serde_json::json!({"type":"object","properties":{
            "city":{"type":"string"}, "value":{"type":"number"}
        }}),
    }];
    for raw in [
        "1.5", "1e308", "-1e308", "1e400", "-1E+400", "1e9999", "1e-400",
    ] {
        let projected = serde_json::from_str::<serde_json::Value>(raw)
            .unwrap_or_else(|_| serde_json::Value::String(raw.into()));
        let expected = vec![crate::UnifiedEvent::ToolCall {
            name: "echo".into(),
            arguments: serde_json::json!({"city":"Paris", "value":projected}),
        }];
        for family in ["deepseek_v4", "glm47"] {
            let input = if family == "deepseek_v4" {
                format!(
                    "<｜DSML｜tool_calls><｜DSML｜invoke name=\"echo\"><｜DSML｜parameter name=\"city\" string=\"true\">Paris</｜DSML｜parameter><｜DSML｜parameter name=\"value\" string=\"false\">{raw}</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>"
                )
            } else {
                format!(
                    "<tool_call>echo<arg_key>city</arg_key><arg_value>Paris</arg_value><arg_key>value</arg_key><arg_value>{raw}</arg_value></tool_call>"
                )
            };
            let arguments = format!(r#"{{"city":"Paris","value":{raw}}}"#);
            let mut batch = create_unified_parser_for_family(family, &tools).unwrap();
            assert_eq!(
                batch.parse_complete(&input).unwrap(),
                expected,
                "{family}: {raw}"
            );
            for chunks in deliveries(&input) {
                let mut parser = create_unified_parser_for_family(family, &tools).unwrap();
                let mut output = crate::UnifiedParserOutput::default();
                for chunk in chunks {
                    parser.parse_into(chunk, &mut output).unwrap();
                }
                assert_eq!(
                    output.tool_arguments_raw().get(&0),
                    Some(&arguments),
                    "{family}: {raw}"
                );
                output.append(&mut parser.finish().unwrap());
                assert_eq!(output.tool_arguments_raw().get(&0), Some(&arguments));
                assert_eq!(output.assembled(), expected, "{family}: {raw}");
            }
        }
    }
}
