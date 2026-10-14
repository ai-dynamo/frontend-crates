// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_parsers_v2::{
    MiniMaxM2ToolStreamParser, Qwen3CoderToolStreamParser, Tool, ToolParseResult, ToolParser,
};
use serde_json::{Value, json};

fn assert_calls(schema: Value, fields: &[(&str, &str)], expected: Value) {
    let tools = vec![Tool {
        name: "f".into(),
        description: None,
        strict: None,
        parameters: schema,
    }];
    for qwen in [true, false] {
        let body: String = fields
            .iter()
            .map(|(name, value)| {
                if qwen {
                    format!("<parameter={name}>{value}</parameter>")
                } else {
                    format!("<parameter name=\"{name}\">{value}</parameter>")
                }
            })
            .collect();
        let input = if qwen {
            format!("<tool_call><function=f>{body}</function></tool_call>")
        } else {
            format!("<minimax:tool_call><invoke name=\"f\">{body}</invoke></minimax:tool_call>")
        };
        let schedules = std::iter::once(vec![input.as_str()])
            .chain(std::iter::once(
                input
                    .as_bytes()
                    .chunks(1)
                    .map(|b| std::str::from_utf8(b).unwrap())
                    .collect(),
            ))
            .chain(
                input
                    .char_indices()
                    .map(|(at, _)| vec![&input[..at], &input[at..]]),
            );
        for chunks in schedules {
            let mut parser: Box<dyn ToolParser> = if qwen {
                Box::new(Qwen3CoderToolStreamParser::new(&tools))
            } else {
                Box::new(MiniMaxM2ToolStreamParser::new(&tools))
            };
            let mut output = ToolParseResult::default();
            for chunk in chunks {
                output.append(parser.push(chunk).unwrap());
            }
            output.append(parser.finish().unwrap());
            let output = output.coalesce_calls();
            assert_eq!(output.calls.len(), 1, "qwen={qwen}: {output:?}");
            assert!(output.calls[0].complete);
            assert_eq!(
                serde_json::from_str::<Value>(&output.calls[0].arguments).unwrap(),
                expected,
                "qwen={qwen}"
            );
        }
    }
}

#[test]
fn unused_reference_graph_cannot_change_other_parameter_types() {
    let mut defs = serde_json::Map::new();
    for i in 0..7 {
        defs.insert(format!("N{i}"), if i == 6 { json!({"type":"string"}) } else {
            json!({"allOf": (0..4).map(|_| json!({"$ref":format!("#/$defs/N{}", i + 1)})).collect::<Vec<_>>()})
        });
    }
    for unused in [None, Some("a"), Some("zz")] {
        let mut properties = serde_json::Map::new();
        if unused == Some("a") {
            properties.insert("a".into(), json!({"$ref":"#/$defs/N0"}));
        }
        for (name, ty) in [
            ("count", "integer"),
            ("flag", "boolean"),
            ("text", "string"),
        ] {
            properties.insert(name.into(), json!({"type":ty}));
        }
        if unused == Some("zz") {
            properties.insert("zz".into(), json!({"$ref":"#/$defs/N0"}));
        }
        let schema = json!({"type":"object","$defs":defs,"properties":properties});
        for fields in [
            vec![("count", "42"), ("flag", "yes"), ("text", "null")],
            vec![("text", "null"), ("flag", "false"), ("count", "42")],
        ] {
            assert_calls(
                schema.clone(),
                &fields,
                json!({"count":42,"flag":false,"text":"null"}),
            );
        }
    }
}

#[test]
fn nullable_aliases_preserve_literal_constraints() {
    for literal in [
        json!({"type":"string","const":"null"}),
        json!({"type":"string","enum":["null"]}),
    ] {
        for target in [
            literal.clone(),
            json!({"allOf":[literal.clone()]}),
            json!({"anyOf":[literal.clone()]}),
            json!({"oneOf":[literal.clone()]}),
        ] {
            for chained in [false, true] {
                assert_calls(
                    json!({"type":"object","$defs":{"A":{"$ref":"#/$defs/B"},"B":target},"properties":{"z":{"$ref":if chained {"#/$defs/A"} else {"#/$defs/B"},"nullable":true}}}),
                    &[("z", "null")],
                    json!({"z":"null"}),
                );
            }
        }
    }
    assert_calls(
        json!({"type":"object","$defs":{"A":{"$ref":"#/$defs/B"},"B":{"type":"string"}},"properties":{"z":{"$ref":"#/$defs/A","nullable":true}}}),
        &[("z", "null")],
        json!({"z":null}),
    );
}

#[test]
fn unsupported_reference_scope_preserves_local_constraints() {
    for scope in ["$id", "$dynamicRef", "$recursiveRef"] {
        for use_ref in [false, true] {
            let mut property = json!({"type":"integer"});
            property[scope] = json!("https://example.com/scope");
            if use_ref {
                property["$ref"] = json!("#/$defs/Text");
            }
            assert_calls(
                json!({"type":"object","$defs":{"Text":{"type":"string"}},"properties":{"z":property}}),
                &[("z", "42")],
                json!({"z":42}),
            );
        }
    }
}

#[test]
fn reachable_large_self_reference_retains_complete_calls() {
    let mut properties = serde_json::Map::new();
    for i in 0..10 {
        properties.insert(format!("p{i}"), json!({"$ref":"#/$defs/X"}));
    }
    assert_calls(
        json!({
            "type":"object",
            "$defs":{"X":{"type":"string","description":"x".repeat(100_000),"$ref":"#/$defs/X"}},
            "properties":properties,
        }),
        &[("p0", "42")],
        json!({"p0":"42"}),
    );
}
