// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_parsers_v2::tool_calling::kimi_k2::KimiK2ToolStreamParser;
use dynamo_parsers_v2::tool_calling::kimi_k3::KimiK3ToolStreamParser;
use dynamo_parsers_v2::tool_calling::traits::{ToolCallDelta, ToolParser};

const FORMS: [(&str, &str, &str, &str); 3] = [
    (
        "<|tool_calls_section_begin|><|tool_call_begin|>functions.write_file:0<|tool_call_argument_begin|>{\"count\":7,\"content\":\"",
        "\"}",
        "<|tool_call_end|>",
        "<|tool_calls_section_end|>",
    ),
    (
        "<|open|>tools<|sep|><|open|>call tool=\"write_file\" index=\"1\"<|sep|><|open|>argument key=\"count\" type=\"number\"<|sep|>7<|close|>argument<|sep|><|open|>argument key=\"content\" type=\"string\"<|sep|>",
        "<|close|>argument<|sep|>",
        "<|close|>call<|sep|>",
        "<|close|>tools<|sep|>",
    ),
    (
        "<|open|>tools<|sep|><|open|>call tool=\"write_file\" index=\"1\"<|sep|><|open|>json type=\"object\"<|sep|>{\"count\":7,\"content\":\"",
        "\"}<|close|>json<|sep|>",
        "<|close|>call<|sep|>",
        "<|close|>tools<|sep|>",
    ),
];

fn deliver(parser: &mut dyn ToolParser, input: &str, size: usize, calls: &mut Vec<ToolCallDelta>) {
    for chunk in input.as_bytes().chunks(size) {
        let result = parser.push(std::str::from_utf8(chunk).unwrap()).unwrap();
        assert!(
            result.normal_text.is_empty(),
            "markup leaked: {:?}",
            result.normal_text
        );
        calls.extend(result.calls);
    }
}

#[test]
fn rejected_typed_header_resumes_literal_string_progress() {
    let (prefix, value_close, call_close, section_close) = FORMS[1];
    let content = format!(
        "head<|close|>argument<|sep|> <|open|>argument gibberish<|sep|>{}",
        "q".repeat(3072)
    );
    for size in [1, 7, 1024] {
        let mut parser = KimiK3ToolStreamParser::new(&[]);
        let mut calls = Vec::new();
        deliver(&mut parser, &format!("{prefix}{content}"), size, &mut calls);
        let received: String = calls.iter().map(|delta| delta.arguments.as_str()).collect();
        assert_eq!(received, format!("{{\"count\":7,\"content\":\"{content}"));
        assert!(calls.iter().all(|delta| !delta.complete));
        deliver(
            &mut parser,
            &format!("{value_close}{call_close}{section_close}"),
            size,
            &mut calls,
        );
        calls.extend(parser.finish().unwrap().calls);
        let arguments: String = calls.iter().map(|delta| delta.arguments.as_str()).collect();
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&arguments).unwrap(),
            serde_json::json!({"count":7,"content":content})
        );
        assert_eq!(calls.iter().filter(|delta| delta.name.is_some()).count(), 1);
        assert_eq!(calls.iter().filter(|delta| delta.complete).count(), 1);
    }
}

#[test]
fn unterminated_string_recovery_preserves_buffered_fallback_and_siblings() {
    use dynamo_parsers_v2::unified::{
        UnifiedParserEvent, UnifiedParserExt, create_unified_parser_for_family,
    };

    for (wrapped, placement, payload) in [false, true].into_iter().flat_map(|wrapped| {
        ["none", "before", "after"]
            .into_iter()
            .flat_map(move |placement| {
                ["abc", "abc ", "abc  "]
                    .into_iter()
                    .map(move |payload| (wrapped, placement, payload))
            })
    }) {
        let sibling = placement != "none";
        let good = if sibling {
            "<|tool_call_begin|>functions.good:1<|tool_call_argument_begin|>{\"x\":7}<|tool_call_end|>"
        } else {
            ""
        };
        let prefix = if wrapped {
            "<|tool_calls_section_begin|>"
        } else {
            ""
        };
        let before = if placement == "before" { good } else { "" };
        let after = if placement == "after" { good } else { "" };
        let input = format!(
            "{prefix}{before}<|tool_call_begin|>functions.write_file:0<|tool_call_argument_begin|>{{\"x\":\"{payload}<|tool_call_end|>{after}<|tool_calls_section_end|>"
        );
        for size in [1, 7, input.len()] {
            let mut tool = KimiK2ToolStreamParser::new(&[]);
            let mut unified = create_unified_parser_for_family("kimi_k2", &[]).unwrap();
            let mut output = Default::default();
            let mut unified_deltas = Vec::new();
            for chunk in input.as_bytes().chunks(size) {
                let chunk = std::str::from_utf8(chunk).unwrap();
                let result = tool.push(chunk).unwrap();
                assert!(result.normal_text.is_empty());
                output = {
                    let mut output: dynamo_parsers_v2::tool_calling::traits::ToolParseResult =
                        output;
                    output.append(result);
                    output
                };
                unified_deltas.extend(unified.push(chunk).unwrap().into_iter().filter_map(
                    |event| match event {
                        UnifiedParserEvent::ToolCall(delta) => Some(delta),
                        _ => None,
                    },
                ));
            }
            let index = usize::from(placement == "before");
            let id = tool.tool_call_id(index).unwrap().to_string();
            output.append(tool.finish().unwrap());
            let final_unified = unified.finish().unwrap();
            unified_deltas.extend(final_unified.events.into_iter().filter_map(
                |event| match event {
                    UnifiedParserEvent::ToolCall(delta) => Some(delta),
                    _ => None,
                },
            ));
            assert_eq!(output.calls, unified_deltas);
            assert_eq!(tool.tool_call_id(index), Some(id.as_str()));
            // Bare recovery needs a settled closer before EOF; a later sibling
            // can settle an otherwise ambiguous closer inside an open string.
            let recovered = wrapped || placement == "after";
            assert_eq!(
                output
                    .calls
                    .iter()
                    .filter(|delta| delta.tool_index == index)
                    .filter(|delta| delta.complete)
                    .count(),
                usize::from(recovered)
            );
            let completed = output.coalesce_calls().calls;
            assert_eq!(
                completed.len(),
                usize::from(recovered) + usize::from(sibling)
            );
            if recovered {
                let bad = completed
                    .iter()
                    .find(|call| call.tool_index == index)
                    .unwrap();
                assert_eq!(bad.name.as_deref(), Some("write_file"));
                assert_eq!(bad.arguments, "{\"x\":\"abc");
            }
            if sibling {
                let good_index = usize::from(placement == "after");
                let good = completed
                    .iter()
                    .find(|call| call.tool_index == good_index)
                    .unwrap();
                assert_eq!(good.name.as_deref(), Some("good"));
                assert_eq!(good.arguments, "{\"x\":7}");
            }
        }
    }
}

#[test]
fn malformed_whitespace_recovery_preserves_every_released_byte() {
    use dynamo_parsers_v2::unified::{
        UnifiedParserEvent, UnifiedParserExt, create_unified_parser_for_family,
    };

    for payload in [
        "{ ",
        "{\"x\": ",
        "{\"x\":7, ",
        "{\"x\": \t\n",
        "{\"x\": \u{2003}",
    ] {
        let input = format!(
            "<|tool_calls_section_begin|><|tool_call_begin|>functions.write_file:0<|tool_call_argument_begin|>{payload}<|tool_call_end|><|tool_calls_section_end|>"
        );
        let mut tool = KimiK2ToolStreamParser::new(&[]);
        let mut unified = create_unified_parser_for_family("kimi_k2", &[]).unwrap();
        let mut tool_deltas = Vec::new();
        let mut unified_deltas = Vec::new();
        for ch in input.chars() {
            tool_deltas.extend(tool.push(&ch.to_string()).unwrap().calls);
            unified_deltas.extend(
                unified
                    .push(&ch.to_string())
                    .unwrap()
                    .into_iter()
                    .filter_map(|event| match event {
                        UnifiedParserEvent::ToolCall(delta) => Some(delta),
                        _ => None,
                    }),
            );
        }
        tool_deltas.extend(tool.finish().unwrap().calls);
        unified_deltas.extend(
            unified
                .finish()
                .unwrap()
                .events
                .into_iter()
                .filter_map(|event| match event {
                    UnifiedParserEvent::ToolCall(delta) => Some(delta),
                    _ => None,
                }),
        );
        let arguments = tool_deltas
            .iter()
            .map(|delta| delta.arguments.as_str())
            .collect::<String>();
        let mut whole = KimiK2ToolStreamParser::new(&[]);
        let mut whole_result = whole.push(&input).unwrap();
        whole_result.append(whole.finish().unwrap());
        let whole_calls = whole_result.coalesce_calls().calls;
        assert_eq!(arguments, whole_calls[0].arguments, "payload={payload:?}");
        assert_eq!(tool_deltas, unified_deltas);
    }
}

#[test]
fn native_argument_bytes_are_independent_of_provisional_timing() {
    let typed_body = "<|open|>argument key=\"x\" type=\"number\"<|sep|>1<|close|>argument<|sep|><|open|>argument key=\"x\" type=\"number\"<|sep|>2<|close|>argument<|sep|>";
    let mut cases = Vec::new();
    for (form, (prefix, value_close, call_close, section_close)) in FORMS.iter().enumerate() {
        if form == 1 {
            cases.push((
                form,
                format!(
                    "{}{typed_body}{call_close}{section_close}",
                    prefix.split_once("<|open|>argument").unwrap().0
                ),
                "{\"x\":1,\"x\":2}",
            ));
        } else {
            for raw in ["{ }", "{ \"x\" : 1, \"x\" : 2 }"] {
                cases.push((
                    form,
                    format!(
                        "{}{raw}{}{call_close}{section_close}",
                        prefix.split_once('{').unwrap().0,
                        if form == 0 { "" } else { &value_close[2..] }
                    ),
                    raw,
                ));
            }
        }
    }
    for (form, wrapped, expected) in cases {
        for bare in [false, true] {
            let input = if bare {
                wrapped
                    .strip_prefix(if form == 0 {
                        "<|tool_calls_section_begin|>"
                    } else {
                        "<|open|>tools<|sep|>"
                    })
                    .unwrap()
                    .strip_suffix(FORMS[form].3)
                    .unwrap()
                    .to_string()
            } else {
                wrapped.clone()
            };
            for split in (0..=input.len()).filter(|at| input.is_char_boundary(*at)) {
                let mut parser: Box<dyn ToolParser> = if form == 0 {
                    Box::new(KimiK2ToolStreamParser::new(&[]))
                } else {
                    Box::new(KimiK3ToolStreamParser::new(&[]))
                };
                let mut output = parser.push(&input[..split]).unwrap();
                output.append(parser.push(&input[split..]).unwrap());
                output.append(parser.finish().unwrap());
                assert!(output.normal_text.is_empty());
                assert_eq!(
                    output
                        .calls
                        .iter()
                        .filter(|delta| delta.name.is_some())
                        .count(),
                    1
                );
                assert_eq!(
                    output.calls.iter().filter(|delta| delta.complete).count(),
                    1
                );
                assert_eq!(
                    output
                        .calls
                        .iter()
                        .map(|delta| delta.arguments.as_str())
                        .collect::<String>(),
                    expected,
                    "form={form} bare={bare} split={split}"
                );
            }
        }
    }
}

#[test]
fn native_long_strings_progress_in_every_field_position() {
    for (form, (prefix, value_close, call_close, section_close)) in FORMS.iter().enumerate() {
        for position in 0..3 {
            for earlier in ["", " \t\n", "short"] {
                let content = "q".repeat(3072);
                let mut fields = vec![("lead", earlier, "string"), ("count", "7", "number")];
                fields.insert(position, ("content", &content, "string"));
                let encode = |key: &str, value: &str, kind: &str| {
                    format!(
                        "{}:{}",
                        serde_json::to_string(key).unwrap(),
                        if kind == "string" {
                            serde_json::to_string(value).unwrap()
                        } else {
                            value.to_string()
                        }
                    )
                };
                let mut paused = if form == 1 {
                    prefix.split_once("<|open|>argument").unwrap().0.to_string()
                } else {
                    format!("{}{{", prefix.split_once('{').unwrap().0)
                };
                let mut expected_prefix = "{".to_string();
                for (index, (key, value, kind)) in fields.iter().enumerate().take(position + 1) {
                    if index > 0 {
                        expected_prefix.push(',');
                        if form != 1 {
                            paused.push(',');
                        }
                    }
                    let encoded = encode(key, value, kind);
                    if form == 1 {
                        paused.push_str(&format!(
                            "<|open|>argument key=\"{key}\" type=\"{kind}\"<|sep|>{value}"
                        ));
                        if index < position {
                            paused.push_str(value_close);
                        }
                    } else if index < position {
                        paused.push_str(&encoded);
                    } else {
                        paused.push_str(&encoded[..encoded.len() - 1]);
                    }
                    if index < position {
                        expected_prefix.push_str(&encoded);
                    } else {
                        expected_prefix.push_str(&encoded[..encoded.len() - 1]);
                    }
                }
                let mut tail = if form == 1 {
                    value_close.to_string()
                } else {
                    "\"".to_string()
                };
                for (key, value, kind) in fields.iter().skip(position + 1) {
                    if form == 1 {
                        tail.push_str(&format!("<|open|>argument key=\"{key}\" type=\"{kind}\"<|sep|>{value}{value_close}"));
                    } else {
                        tail.push(',');
                        tail.push_str(&encode(key, value, kind));
                    }
                }
                if form != 1 {
                    tail.push_str(&value_close[1..]);
                }
                tail.push_str(call_close);
                tail.push_str(section_close);
                for size in [1, 7, 1024] {
                    let mut parser: Box<dyn ToolParser> = if form == 0 {
                        Box::new(KimiK2ToolStreamParser::new(&[]))
                    } else {
                        Box::new(KimiK3ToolStreamParser::new(&[]))
                    };
                    let mut calls = Vec::new();
                    deliver(parser.as_mut(), &paused, size, &mut calls);
                    assert_eq!(
                        calls
                            .iter()
                            .map(|delta| delta.arguments.as_str())
                            .collect::<String>(),
                        expected_prefix,
                        "form={form} position={position} earlier={earlier:?} size={size}"
                    );
                    assert_eq!(calls.iter().filter(|delta| delta.name.is_some()).count(), 1);
                    assert!(
                        calls
                            .iter()
                            .all(|delta| delta.tool_index == 0 && !delta.complete)
                    );
                    let id = parser.tool_call_id(0).unwrap().to_string();
                    deliver(parser.as_mut(), &tail, size, &mut calls);
                    calls.extend(parser.finish().unwrap().calls);
                    assert_eq!(calls.iter().filter(|delta| delta.complete).count(), 1);
                    assert_eq!(parser.tool_call_id(0), Some(id.as_str()));
                    assert_eq!(
                        serde_json::from_str::<serde_json::Value>(
                            &calls
                                .iter()
                                .map(|delta| delta.arguments.as_str())
                                .collect::<String>()
                        )
                        .unwrap(),
                        serde_json::json!({"lead":earlier,"count":7,"content":content})
                    );
                }
            }
        }
    }
}

#[test]
fn native_count_before_open_content_progresses_at_each_pause() {
    for (form, (prefix, value_close, call_close, section_close)) in FORMS.iter().enumerate() {
        for size in [7, 1024] {
            let mut parser: Box<dyn ToolParser> = if form == 0 {
                Box::new(KimiK2ToolStreamParser::new(&[]))
            } else {
                Box::new(KimiK3ToolStreamParser::new(&[]))
            };
            let mut calls = Vec::new();
            deliver(
                parser.as_mut(),
                &format!("{prefix}{}", "q".repeat(3072)),
                size,
                &mut calls,
            );
            let expected = format!("{{\"count\":7,\"content\":\"{}", "q".repeat(3072));
            assert_eq!(
                calls
                    .iter()
                    .map(|c| c.arguments.as_str())
                    .collect::<String>(),
                expected,
                "form={form} chunk={size}"
            );
            assert_eq!(calls.iter().filter(|c| c.name.is_some()).count(), 1);
            assert_eq!(
                calls.iter().find_map(|c| c.name.as_deref()),
                Some("write_file")
            );
            assert!(calls.iter().all(|c| c.tool_index == 0 && !c.complete));
            let id = parser
                .tool_call_id(0)
                .expect("native identity before close")
                .to_string();
            deliver(parser.as_mut(), &"q".repeat(1024), size, &mut calls);
            assert_eq!(
                calls
                    .iter()
                    .map(|c| c.arguments.as_str())
                    .collect::<String>(),
                format!("{expected}{}", "q".repeat(1024))
            );
            assert!(calls.iter().all(|c| !c.complete));
            deliver(parser.as_mut(), value_close, size, &mut calls);
            assert!(calls.iter().all(|c| !c.complete));
            deliver(parser.as_mut(), call_close, size, &mut calls);
            assert_eq!(
                calls.iter().filter(|c| c.complete).count(),
                usize::from(form == 0)
            );
            deliver(parser.as_mut(), section_close, size, &mut calls);
            calls.extend(parser.finish().unwrap().calls);
            assert_eq!(calls.iter().filter(|c| c.complete).count(), 1);
            assert_eq!(calls.iter().filter(|c| c.name.is_some()).count(), 1);
            assert_eq!(parser.tool_call_id(0), Some(id.as_str()));
            let arguments = calls
                .iter()
                .map(|c| c.arguments.as_str())
                .collect::<String>();
            assert_eq!(
                serde_json::from_str::<serde_json::Value>(&arguments).unwrap(),
                serde_json::json!({"count": 7, "content": "q".repeat(4096)})
            );
        }
    }
}

#[test]
fn compact_native_inputs_match_unified_at_every_utf8_split() {
    use dynamo_parsers_v2::tool_calling::traits::ToolParseResult;
    use dynamo_parsers_v2::unified::{UnifiedParserExt, create_unified_parser_for_family};

    let cases = [
        (
            "kimi_k2",
            "<|tool_calls_section_begin|><|tool_call_begin|>functions.write_file:0<|tool_call_argument_begin|> \n {\"count\":7, \"content\":\"é\\n\\\"<|tool_call_end|>\"} trailing<|tool_call_end|><|tool_calls_section_end|>",
            serde_json::json!({"count":7,"content":"é\n\"<|tool_call_end|>"}),
        ),
        (
            "kimi_k3",
            "<|open|>tools<|sep|><|open|>call tool=\"write_file\" index=\"1\"<|sep|><|open|>json type=\"object\"<|sep|> \n {\"count\":7, \"content\":\"é\\n\\\"<|close|>json<|sep|>\"}<|close|>json<|sep|><|close|>call<|sep|><|close|>tools<|sep|>",
            serde_json::json!({"count":7,"content":"é\n\"<|close|>json<|sep|>"}),
        ),
        (
            "kimi_k3",
            "<|open|>tools<|sep|><|open|>call tool=\"write_file\" index=\"1\"<|sep|><|open|>argument key=\"count\" type=\"number\"<|sep|>7<|close|>argument<|sep|><|open|>argument key=\"content\" type=\"string\"<|sep|>é\n\\\"<|close|>argument<|sep|> literal<|close|>argument<|sep|><|open|>argument key=\"count\" type=\"number\"<|sep|>8<|close|>argument<|sep|><|close|>call<|sep|><|close|>tools<|sep|>",
            serde_json::json!({"count":8,"content":"é\n\\\"<|close|>argument<|sep|> literal"}),
        ),
    ];
    for (family, input, expected) in cases {
        for split in (0..=input.len()).filter(|at| input.is_char_boundary(*at)) {
            let mut legacy: Box<dyn ToolParser> = if family == "kimi_k2" {
                Box::new(KimiK2ToolStreamParser::new(&[]))
            } else {
                Box::new(KimiK3ToolStreamParser::new(&[]))
            };
            let mut unified = create_unified_parser_for_family(family, &[]).unwrap();
            let mut legacy_output = ToolParseResult::default();
            let mut unified_output = ToolParseResult::default();
            for chunk in [&input[..split], &input[split..]] {
                let left = legacy.push(chunk).unwrap();
                let right = ToolParseResult::from_deltas(unified.push(chunk).unwrap());
                assert_eq!(left, right, "{family} split={split}");
                legacy_output.append(left);
                unified_output.append(right);
                assert_eq!(legacy.tool_call_id(0), unified.tool_call_id(0));
            }
            legacy_output.append(legacy.finish().unwrap());
            unified_output.append(ToolParseResult::from_deltas(
                unified.finish().unwrap().events,
            ));
            assert_eq!(legacy_output, unified_output);
            assert_eq!(
                legacy_output
                    .calls
                    .iter()
                    .filter(|delta| delta.name.is_some())
                    .count(),
                1
            );
            assert_eq!(
                legacy_output
                    .calls
                    .iter()
                    .filter(|delta| delta.complete)
                    .count(),
                1
            );
            let final_output = legacy_output.coalesce_calls();
            assert!(final_output.normal_text.is_empty());
            assert_eq!(final_output.calls.len(), 1);
            assert_eq!(
                serde_json::from_str::<serde_json::Value>(&final_output.calls[0].arguments)
                    .unwrap(),
                expected,
                "{family} split={split}"
            );
        }
    }
}

#[test]
fn abandoned_native_calls_keep_identity_and_do_not_reuse_their_index() {
    use dynamo_parsers_v2::tool_calling::traits::ToolParseResult;
    for family in ["kimi_k2", "kimi_k3"] {
        let mut parser: Box<dyn ToolParser> = if family == "kimi_k2" {
            Box::new(KimiK2ToolStreamParser::new(&[]))
        } else {
            Box::new(KimiK3ToolStreamParser::new(&[]))
        };
        let (prefix, _, _, _) = FORMS[if family == "kimi_k2" { 0 } else { 2 }];
        let mut output = parser.push(&format!("{prefix}qq")).unwrap();
        assert!(
            output
                .calls
                .iter()
                .all(|delta| delta.tool_index == 0 && !delta.complete)
        );
        let id = parser.tool_call_id(0).unwrap().to_string();
        let rest = if family == "kimi_k2" {
            "\"}<|tool_calls_section_end|><|tool_calls_section_begin|><|tool_call_begin|>functions.good:1<|tool_call_argument_begin|>{\"x\":7}<|tool_call_end|><|tool_calls_section_end|>"
        } else {
            "\"<|close|>json<|sep|><|close|>call<|sep|><|close|>tools<|sep|><|open|>tools<|sep|><|open|>call tool=\"good\" index=\"2\"<|sep|><|open|>json type=\"object\"<|sep|>{\"x\":7}<|close|>json<|sep|><|close|>call<|sep|><|close|>tools<|sep|>"
        };
        output.append(parser.push(rest).unwrap());
        output.append(parser.finish().unwrap());
        assert_eq!(parser.tool_call_id(0), Some(id.as_str()));
        assert!(
            output
                .calls
                .iter()
                .filter(|delta| delta.tool_index == 0)
                .all(|delta| !delta.complete)
        );
        let final_output: ToolParseResult = output.coalesce_calls();
        assert!(final_output.normal_text.is_empty());
        assert_eq!(final_output.calls.len(), 1);
        assert_eq!(final_output.calls[0].tool_index, 1);
        assert_eq!(final_output.calls[0].name.as_deref(), Some("good"));
        assert_eq!(final_output.calls[0].arguments, "{\"x\":7}");
    }
}
