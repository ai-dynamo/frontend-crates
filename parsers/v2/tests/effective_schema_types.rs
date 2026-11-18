// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_parsers_v2::{
    Tool, UnifiedEvent, UnifiedParserExt, UnifiedParserInit, assemble,
    create_unified_parser_for_family,
};
use serde_json::{Value, json};

fn wire(family: &str, fields: &[(&str, &str)]) -> String {
    let body: String = fields
        .iter()
        .map(|(key, value)| match family {
            "beam" => format!("<rfl:parameter name=\"{key}\">\n{value}\n</rfl:parameter>"),
            "glm47" => format!("<arg_key>{key}</arg_key><arg_value>{value}</arg_value>"),
            "minimax_m3" => format!("]<]minimax[>[<{key}>{value}]<]minimax[>[</{key}>"),
            _ => unreachable!(),
        })
        .collect();
    match family {
        "beam" => format!(
            "<|start|>assistant to=capture_payload<|channel|>analysis <|constrain|>xml<|message|>{body}<|call|>"
        ),
        "glm47" => format!("<tool_call>capture_payload{body}</tool_call>"),
        "minimax_m3" => format!(
            "]<]minimax[>[<tool_call>]<]minimax[>[<invoke name=\"capture_payload\">{body}]<]minimax[>[</invoke>]<]minimax[>[</tool_call>"
        ),
        _ => unreachable!(),
    }
}

fn check(family: &str, schema: Value, fields: &[(&str, &str)], expected: Value) {
    let tools = [Tool {
        name: "capture_payload".into(),
        description: None,
        strict: None,
        parameters: schema,
    }];
    let input = wire(family, fields);
    let mut parser = create_unified_parser_for_family(family, &tools).unwrap();
    for split in input.char_indices().map(|(at, _)| at).chain([input.len()]) {
        parser.reset();
        parser
            .initialize_request(UnifiedParserInit::default())
            .unwrap();
        let mut events = parser.push(&input[..split]).unwrap();
        events.extend(parser.push(&input[split..]).unwrap());
        assert_eq!(
            assemble(&events),
            [UnifiedEvent::ToolCall {
                name: "capture_payload".into(),
                arguments: expected.clone()
            }],
            "{family} progress before finish split={split}"
        );
        events.extend(parser.finish().unwrap().events);
        assert_eq!(
            assemble(&events),
            [UnifiedEvent::ToolCall {
                name: "capture_payload".into(),
                arguments: expected.clone()
            }],
            "{family} split={split}"
        );
        if family != "beam" {
            let mut tool_parser =
                dynamo_parsers_v2::create_tool_parser_for_family(family, &tools).unwrap();
            let mut result = dynamo_parsers_v2::ToolParseResult::default();
            result.append(tool_parser.push(&input[..split]).unwrap());
            result.append(tool_parser.push(&input[split..]).unwrap());
            result.append(tool_parser.finish().unwrap());
            let result = result.coalesce_calls();
            assert_eq!(result.calls.len(), 1, "{family} legacy split={split}");
            assert_eq!(
                serde_json::from_str::<Value>(&result.calls[0].arguments).unwrap(),
                expected,
                "{family} legacy split={split}"
            );
        }
    }
    parser.reset();
    parser
        .initialize_request(UnifiedParserInit::default())
        .unwrap();
    let mut events = Vec::new();
    for ch in input.chars() {
        events.extend(parser.push(ch.encode_utf8(&mut [0; 4])).unwrap());
    }
    events.extend(parser.finish().unwrap().events);
    assert_eq!(
        assemble(&events),
        [UnifiedEvent::ToolCall {
            name: "capture_payload".into(),
            arguments: expected
        }],
        "{family} character schedule"
    );
}

fn conditional(family: &str) {
    check(
        family,
        json!({"type":"object","properties":{
            "amount":{"if":{"minimum":0},"then":{"type":"number","minimum":0},"else":{"type":"number"}},
            "retries":{"if":{"type":"integer"},"then":{"minimum":0},"else":{"type":"number"}}
        }}),
        &[("amount", "21.5"), ("retries", "37")],
        json!({"amount":21.5,"retries":37}),
    );
}
fn negated(family: &str) {
    check(
        family,
        json!({"type":"object","properties":{
            "count":{"not":{"type":"string"}},"text":{"not":{"type":"integer"}}
        }}),
        &[("count", "-2500"), ("text", "42")],
        json!({"count":-2500,"text":"42"}),
    );
}
#[test]
fn beam_conditional() {
    conditional("beam");
}
#[test]
fn glm_conditional() {
    conditional("glm47");
}
#[test]
fn minimax_conditional() {
    conditional("minimax_m3");
}
#[test]
fn beam_negated() {
    negated("beam");
}
#[test]
fn glm_negated() {
    negated("glm47");
}

#[test]
fn shared_constraints_keep_impossible_unknown_and_fractional_cases_distinct() {
    for family in ["beam", "glm47", "minimax_m3"] {
        for (schema, raw, expected) in [
            (
                json!({"type":"number","not":{"type":"integer"}}),
                "42",
                json!("42"),
            ),
            (
                json!({"type":"integer","allOf":[{"type":"string"}]}),
                "42",
                json!("42"),
            ),
            (
                json!({"if":{"type":"string"},"then":false,"else":{"type":"number"}}),
                "42",
                json!(42),
            ),
            (
                json!({"if":{"minimum":0},"then":{"allOf":[{"type":"string"},{"type":"number"}]},"else":{"type":"number"}}),
                "42",
                json!(42),
            ),
            (json!({"anyOf":[false,{"type":"integer"}]}), "42", json!(42)),
            (
                json!({"if":{"type":"integer"},"then":{"minimum":1}}),
                "42",
                json!("42"),
            ),
            (
                json!({"if":{"minimum":0},"then":{"type":"number"}}),
                "42",
                json!("42"),
            ),
            (
                json!({"allOf":[{"type":"number"},{"not":{"type":"integer"}}]}),
                "2.5",
                json!(2.5),
            ),
            (json!({"not":{"type":"integer"}}), "42", json!("42")),
            (json!({"not":{"const":0}}), "42", json!("42")),
            (json!({"not":{"$ref":"#/$defs/Text"}}), "42", json!(42)),
            (
                json!({"$ref":"#/$defs/Numeric","type":"integer"}),
                "42",
                json!(42),
            ),
            (json!({"$ref":"#/$defs/Cycle"}), "42", json!("42")),
            (json!({"not":{"$ref":"#/$defs/Missing"}}), "42", json!("42")),
            (
                json!({"if":{"minimum":0},"then":{"type":"number"},"else":{"type":"number"}}),
                "null",
                json!("null"),
            ),
            (
                json!({"if":{"type":"null"},"then":{"type":"null"},"else":{"type":"number"}}),
                "null",
                json!(null),
            ),
        ] {
            check(
                family,
                json!({"type":"object","$defs":{
                "Text":{"type":"string"},"Numeric":{"type":["integer","number"]},"Cycle":{"$ref":"#/$defs/Cycle"}
            },"properties":{"value":schema}}),
                &[("value", raw)],
                json!({"value":expected}),
            );
        }
    }
}

#[test]
fn exact_fractional_tokens_survive_conditional_typing() {
    for family in ["beam", "glm47", "minimax_m3"] {
        for raw in ["9007199254740992.5", "1e-400", "1.0000000000000001"] {
            let tools = [Tool {
                name: "capture_payload".into(),
                description: None,
                strict: None,
                parameters: json!({"type":"object","properties":{"value":{
                    "if":{"minimum":0},"then":{"type":"number"},"else":{"type":"number"}
                }}}),
            }];
            let input = wire(family, &[("value", raw)]);
            let mut parser = create_unified_parser_for_family(family, &tools).unwrap();
            for split in input.char_indices().map(|(at, _)| at).chain([input.len()]) {
                parser.reset();
                parser
                    .initialize_request(UnifiedParserInit::default())
                    .unwrap();
                let mut output = dynamo_parsers_v2::UnifiedParserOutput::default();
                parser.parse_into(&input[..split], &mut output).unwrap();
                parser.parse_into(&input[split..], &mut output).unwrap();
                output.append(&mut parser.finish().unwrap());
                let arguments = output.tool_arguments_raw();
                let expected = if family == "beam" {
                    format!("{{\"value\": {raw}}}")
                } else {
                    format!("{{\"value\":{raw}}}")
                };
                assert_eq!(arguments[&0], expected, "{family} split={split} raw={raw}");
            }
        }
    }
}

#[test]
fn beam_retains_native_keyword_inference() {
    check(
        "beam",
        json!({"type":"object","properties":{"value":{"minimum":0}}}),
        &[("value", "42")],
        json!({"value":42}),
    );
    check(
        "beam",
        json!({"type":"object","properties":{"value":{"properties":{"x":{"type":"integer"}}}}}),
        &[("value", r#"{"x":42}"#)],
        json!({"value":{"x":42}}),
    );
}

#[test]
fn glm_numeric_aliases_keep_native_conversion_and_effective_exclusions() {
    for kind in ["number", "float", "double"] {
        check(
            "glm47",
            json!({"type":"object","properties":{"value":{"type":kind}}}),
            &[("value", "2.5")],
            json!({"value":2.5}),
        );
        check(
            "glm47",
            json!({"type":"object","properties":{"value":{"type":kind,"not":{"type":"integer"}}}}),
            &[("value", "42")],
            json!({"value":"42"}),
        );
    }
}

#[test]
fn reference_depth_budget_preserves_unknown_text() {
    for family in ["beam", "glm47", "minimax_m3"] {
        for depth in [16, 17, 24] {
            let mut defs = serde_json::Map::new();
            for i in 0..depth {
                defs.insert(
                    format!("N{i}"),
                    if i + 1 == depth {
                        json!({"type":"integer"})
                    } else {
                        json!({"$ref":format!("#/$defs/N{}",i+1)})
                    },
                );
            }
            check(
                family,
                json!({"type":"object","$defs":defs,"properties":{"value":{"$ref":"#/$defs/N0"}}}),
                &[("value", "42")],
                json!({"value":if depth<=16 {json!(42)} else {json!("42")}}),
            );
        }
    }
}

#[test]
fn utf8_payloads_keep_native_string_preference() {
    for family in ["beam", "glm47", "minimax_m3"] {
        check(
            family,
            json!({"type":"object","properties":{"value":{"type":"string"}}}),
            &[("value", "München")],
            json!({"value":"München"}),
        );
    }
}

#[test]
fn glm_passive_literals_keep_numeric_alternatives_without_hiding_unknown_refs() {
    check(
        "glm47",
        json!({"type":"object","properties":{"value":{"anyOf":[{"enum":["auto"]},{"type":"integer"}]}}}),
        &[("value", "42")],
        json!({"value":42}),
    );
    for schema in [
        json!({"anyOf":[{"$ref":"#/$defs/Missing"},{"type":"integer"}]}),
        json!({"anyOf":[{"$ref":"#/$defs/Cycle"},{"type":"integer"}]}),
        json!({"anyOf":[{"$dynamicRef":"#/$defs/Missing"},{"type":"integer"}]}),
        json!({"anyOf":[{"$recursiveRef":"#/$defs/Missing"},{"type":"integer"}]}),
    ] {
        check(
            "glm47",
            json!({"type":"object","$defs":{"Cycle":{"$ref":"#/$defs/Cycle"}},"properties":{"value":schema}}),
            &[("value", "42")],
            json!({"value":"42"}),
        );
    }
}

#[test]
fn beam_preserves_declared_coercion_preference() {
    for (schema, expected) in [
        (json!({"type":["boolean","number"]}), json!(true)),
        (json!({"type":["number","boolean"]}), json!(1)),
        (json!({"type":["integer","boolean","number"]}), json!(1)),
        (json!({"type":["boolean","integer","number"]}), json!(true)),
        (json!({"type":["number","boolean","integer"]}), json!(1)),
        (
            json!({"anyOf":[{"type":"boolean"},{"type":"number"}]}),
            json!(true),
        ),
        (json!({"$ref":"#/$defs/Choice"}), json!(true)),
    ] {
        check(
            "beam",
            json!({"type":"object","$defs":{"Choice":{"type":["boolean","number"]}},"properties":{"value":schema}}),
            &[("value", "1")],
            json!({"value":expected}),
        );
    }
}

#[test]
fn glm_enum_hints_preserve_native_string_ambiguity() {
    for (schema, raw, expected) in [
        (json!({"enum":[1,2,4]}), "2", json!(2)),
        (json!({"enum":[true,false]}), "true", json!(true)),
        (json!({"enum":["42"]}), "42", json!("42")),
        (json!({"enum":[1,"auto"]}), "1", json!("1")),
        (
            json!({"anyOf":[{"type":"string","enum":["auto"]},{"type":"integer"}]}),
            "42",
            json!("42"),
        ),
        (json!({"type":"number","enum":[1,"auto"]}), "1", json!(1)),
        (
            json!({"anyOf":[{"enum":["auto"]},{"type":"integer"}]}),
            "42",
            json!(42),
        ),
    ] {
        for schema in [schema.clone(), json!({"if":true,"then":schema})] {
            check(
                "glm47",
                json!({"type":"object","properties":{"value":schema}}),
                &[("value", raw)],
                json!({"value":expected}),
            );
        }
    }
}

#[test]
fn beam_negation_infers_null_and_object_values() {
    for (raw, expected) in [("null", json!(null)), (r#"{"k":"é"}"#, json!({"k":"é"}))] {
        check(
            "beam",
            json!({"type":"object","properties":{"value":{"not":{"type":"string"}}}}),
            &[("value", raw)],
            json!({"value":expected}),
        );
    }
}

#[test]
fn minimax_declared_types_override_only_contradicting_literals() {
    for (schema, raw, expected) in [
        (json!({"type":"integer","enum":["auto"]}), "5", json!(5)),
        (
            json!({"type":"integer","enum":["auto"]}),
            "auto",
            json!("auto"),
        ),
        (json!({"type":"number","const":"auto"}), "21.5", json!(21.5)),
        (
            json!({"type":"boolean","const":"auto"}),
            "true",
            json!(true),
        ),
        (json!({"type":"string","enum":[1]}), "42", json!("42")),
        (
            json!({"type":["integer","boolean"],"enum":[true]}),
            "1",
            json!(true),
        ),
    ] {
        for parameter in [
            schema.clone(),
            json!({"$ref":"#/$defs/Choice"}),
            json!({"allOf":[schema.clone()]}),
            json!({"if":true,"then":schema.clone()}),
        ] {
            check(
                "minimax_m3",
                json!({"type":"object","$defs":{"Choice":schema},"properties":{"value":parameter}}),
                &[("value", raw)],
                json!({"value":expected}),
            );
        }
    }
}

#[test]
fn beam_declared_types_override_misplaced_enums() {
    check(
        "beam",
        json!({"type":"object","properties":{
            "metrics":{"type":"array","items":{"type":"string"},"enum":["temperature","humidity"]},
            "count":{"type":"integer","enum":["auto"]}
        }}),
        &[("metrics", r#"["temperature","humidity"]"#), ("count", "5")],
        json!({"metrics":["temperature","humidity"],"count":5}),
    );
}
