// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_parsers_v2::structural_tag::{
    ReasoningBoundary, StructuralTagContext, StructuralTagOptions, StructuralTagSchemaMode,
    StructuralTagToolChoice,
};
use dynamo_parsers_v2::{Tool, structural_tag_builder_for_family};
use serde_json::json;

const NS: &str = "]<]minimax[>[";

fn tools() -> Vec<Tool> {
    vec![Tool {
        name: "record".into(),
        description: None,
        strict: Some(true),
        parameters: json!({"type":"object", "properties":{"count":{"type":"integer"}},
            "required":["count"],"additionalProperties":false}),
    }]
}

fn invoke(name: &str, value: &str) -> String {
    format!("{NS}<invoke name=\"{name}\">{NS}<count>{value}{NS}</count>{NS}</invoke>")
}

fn block(calls: &str) -> String {
    format!("{NS}<tool_call>\n{calls}\n{NS}</tool_call>")
}

#[test]
fn native_eligibility_checks_constraints_and_referenced_schemas() {
    let builder = structural_tag_builder_for_family("minimax_m3").unwrap();
    for (schema, supported) in [
        (json!({"allOf":[{"type":"integer"},{"minimum":1}]}), false),
        (json!({"type":"integer","allOf":[{"minimum":1}]}), false),
        (json!({"type":"integer","not":{"const":1}}), false),
        (json!({"$ref":"#/$defs/missing"}), false),
        (
            json!({"$ref":"#/$defs/count","$defs":{"count":{"type":"integer"}}}),
            true,
        ),
        (
            json!({"$ref":"#/$defs/count","$defs":{"count":{"allOf":[{"type":"integer"},{"minimum":1}]}}}),
            false,
        ),
        (
            json!({"$ref":"#/$defs/text","$defs":{"text":{"type":"string"}}}),
            false,
        ),
        (
            json!({"$ref":"#/$defs/a~1b","$defs":{"a/b":{"type":"integer"}}}),
            false,
        ),
        (
            json!({"$ref":"#/$defs/%63ount","$defs":{"count":{"type":"integer"}}}),
            false,
        ),
        (json!({"$ref":"#"}), false),
        (
            json!({"type":"integer","$defs":{"unused":{"type":"string"}}}),
            true,
        ),
    ] {
        let mut tools = tools();
        // References resolve against the complete tool schema, not the property.
        tools[0].parameters["properties"]["count"] = schema.clone();
        if let Some(definitions) = schema.get("$defs") {
            tools[0].parameters["$defs"] = definitions.clone();
        }
        let result = builder.build(&StructuralTagContext {
            tool_choice: StructuralTagToolChoice::Required,
            tools: &tools,
            parallel_tool_calls: None,
            schema_mode: StructuralTagSchemaMode::Auto,
            structured_output_schema: None,
            starts_in_reasoning: false,
            allow_unprefilled_reasoning: false,
        });
        assert_eq!(result.is_ok(), supported, "{schema}: {result:?}");
    }
}

#[test]
fn nested_child_types_the_parser_cannot_recover_select_json_fallback() {
    // XGrammar renders these as nested tags, but the parser types a child only
    // through `properties`, object-valued `additionalProperties`/`items`, or a
    // union with exactly one object or array alternative; anything else comes
    // back as a string and violates the schema the grammar enforced.
    let builder = structural_tag_builder_for_family("minimax_m3").unwrap();
    let object =
        |name: &str, kind: &str| json!({"type":"object","properties":{name:{"type":kind}}});
    for (schema, declined) in [
        (json!({"const":{"n":1}}), Some("nested const literals")),
        (
            json!({"enum":[{"n":1},{"n":2}]}),
            Some("nested enum literals"),
        ),
        (json!({"const":[1,true]}), Some("nested const literals")),
        (
            json!({"type":"array","prefixItems":[{"type":"integer"},{"type":"boolean"}],"items":false,"minItems":2}),
            Some("positional array items"),
        ),
        (
            json!({"type":"array","items":[{"type":"integer"},{"type":"boolean"}]}),
            Some("positional array items"),
        ),
        (
            json!({"anyOf":[object("n", "integer"), object("m", "boolean")]}),
            Some("several object or array alternatives"),
        ),
        (
            json!({"oneOf":[object("n", "integer"), object("m", "boolean")]}),
            Some("several object or array alternatives"),
        ),
        (
            json!({"anyOf":[{"type":"array","items":{"type":"integer"}},{"type":"array","items":{"type":"boolean"}}]}),
            Some("several object or array alternatives"),
        ),
        (
            json!({"type":"integer","if":{"minimum":1},"then":{"maximum":5}}),
            Some("if constraints"),
        ),
        (
            json!({"type":"object","properties":{"n":{"type":"integer"}},"dependentSchemas":{"n":{"required":["n"]}}}),
            Some("dependentSchemas constraints"),
        ),
        (
            json!({"anyOf":[object("n", "integer"), {"type":"null"}]}),
            None,
        ),
        (json!({"type":"array","items":{"type":"integer"}}), None),
        (json!({"const":5}), None),
        (json!({"enum":[1,2]}), None),
        (object("n", "integer"), None),
    ] {
        let mut tools = tools();
        tools[0].parameters["properties"]["count"] = schema.clone();
        let result = builder.build(&StructuralTagContext {
            tool_choice: StructuralTagToolChoice::Required,
            tools: &tools,
            parallel_tool_calls: None,
            schema_mode: StructuralTagSchemaMode::Auto,
            structured_output_schema: None,
            starts_in_reasoning: false,
            allow_unprefilled_reasoning: false,
        });
        match declined {
            Some(reason) => {
                let error = result.expect_err(&format!("{schema} must select fallback"));
                assert!(error.to_string().contains(reason), "{schema}: {error}");
            }
            None => assert!(result.is_ok(), "{schema}: {result:?}"),
        }
    }
}

#[test]
fn annotation_only_schemas_select_json_fallback() {
    // XGrammar's schema parser dispatches on a fixed keyword set; a node with
    // none of them is unconstrained, which the M3 XML converter refuses after
    // the tag is installed. The builder must decline exactly those nodes and
    // keep admitting untyped nodes that do carry a shape keyword.
    let builder = structural_tag_builder_for_family("minimax_m3").unwrap();
    for (schema, supported) in [
        (json!({"description":"any value"}), false),
        (json!({"title":"count"}), false),
        (json!({"default":1}), false),
        (json!({"minimum":1}), false),
        (json!({"minimum":1,"maximum":5}), false),
        (json!({"anyOf":[{"description":"x"}]}), false),
        (
            json!({"anyOf":[{"type":"integer"},{"description":"x"}]}),
            false,
        ),
        (
            json!({"oneOf":[{"type":"integer"},{"description":"x"}]}),
            false,
        ),
        (json!({"type":"array","items":{"description":"x"}}), false),
        (
            json!({"type":"object","properties":{"n":{"minimum":1}},"required":["n"],"additionalProperties":false}),
            false,
        ),
        (
            json!({"$ref":"#/$defs/count","$defs":{"count":{"description":"x"}}}),
            false,
        ),
        (
            json!({"$ref":"#/$defs/count","$defs":{"count":{"minimum":1}}}),
            false,
        ),
        (
            json!({"properties":{"n":{"type":"integer"}},"required":["n"],"additionalProperties":false}),
            true,
        ),
        (json!({"items":{"type":"integer"}}), true),
        (json!({"enum":[1,2]}), true),
        (json!({"const":1}), true),
        (json!({"description":"x","type":"integer"}), true),
        (json!({"type":"integer","minimum":1}), true),
        (
            json!({"type":"integer","$defs":{"unused":{"description":"x"}}}),
            true,
        ),
    ] {
        let mut tools = tools();
        tools[0].parameters["properties"]["count"] = schema.clone();
        if let Some(definitions) = schema.get("$defs") {
            tools[0].parameters["$defs"] = definitions.clone();
        }
        let result = builder.build(&StructuralTagContext {
            tool_choice: StructuralTagToolChoice::Required,
            tools: &tools,
            parallel_tool_calls: None,
            schema_mode: StructuralTagSchemaMode::Auto,
            structured_output_schema: None,
            starts_in_reasoning: false,
            allow_unprefilled_reasoning: false,
        });
        assert_eq!(result.is_ok(), supported, "{schema}: {result:?}");
        if let Err(error) = result {
            assert!(
                error.to_string().contains("constrained tool schema"),
                "{schema}: {error}"
            );
        }
    }
}

#[test]
fn reasoning_gate_override_requires_explicit_family_support() {
    for (family, expected) in [
        ("minimax_m3", true),
        ("qwen3_coder", false),
        ("deepseek_v4", false),
    ] {
        let builder = structural_tag_builder_for_family(family).unwrap();
        assert_eq!(
            builder.supports_reasoning_gate_override(),
            expected,
            "{family}"
        );
    }
}

#[test]
fn optional_reasoning_preserves_prefill_and_external_boundary() {
    let builder = structural_tag_builder_for_family("minimax_m3").unwrap();
    let tools = tools();
    let mut context = StructuralTagContext {
        tool_choice: StructuralTagToolChoice::Auto,
        tools: &tools,
        parallel_tool_calls: Some(true),
        schema_mode: StructuralTagSchemaMode::Auto,
        structured_output_schema: None,
        starts_in_reasoning: true,
        allow_unprefilled_reasoning: false,
    };
    let prefilled = builder.build(&context).unwrap();
    context.allow_unprefilled_reasoning = true;
    assert_eq!(builder.build(&context).unwrap(), prefilled);

    context.starts_in_reasoning = false;
    let external = StructuralTagOptions {
        reasoning_boundary: ReasoningBoundary::External,
        ..Default::default()
    };
    let optional = builder.build_with_options(&context, &external).unwrap();
    context.allow_unprefilled_reasoning = false;
    assert_eq!(
        builder.build_with_options(&context, &external).unwrap(),
        optional
    );
}

/// Export the actual Rust builder output, along with independent accepted and
/// rejected wire examples, for check_minimax_m3_xgrammar.py to execute.
#[test]
fn structural_tag_policy_matrix() {
    let builder = structural_tag_builder_for_family("minimax_m3").unwrap();
    let tools = tools();
    let one = block(&invoke("record", "2"));
    let two = block(&format!(
        "{}\n{}",
        invoke("record", "2"),
        invoke("record", "3")
    ));
    let mut cases = Vec::new();
    for (label, choice) in [
        ("auto", StructuralTagToolChoice::Auto),
        ("required", StructuralTagToolChoice::Required),
        ("named", StructuralTagToolChoice::Named("record")),
        ("none", StructuralTagToolChoice::None),
    ] {
        for parallel in [true, false] {
            for (reasoning, optional) in [(true, false), (false, false), (false, true)] {
                let tag = builder
                    .build(&StructuralTagContext {
                        tool_choice: choice,
                        tools: &tools,
                        parallel_tool_calls: Some(parallel),
                        schema_mode: StructuralTagSchemaMode::Auto,
                        structured_output_schema: None,
                        starts_in_reasoning: reasoning,
                        allow_unprefilled_reasoning: optional,
                    })
                    .unwrap()
                    .unwrap();
                let prefix = if reasoning && label != "none" {
                    "Look.</mm:think>"
                } else if optional && label != "none" {
                    "<mm:think>Look.</mm:think>"
                } else {
                    ""
                };
                if optional && label != "none" {
                    assert_eq!(tag["format"]["elements"][0]["type"], "or");
                    assert_eq!(
                        tag["format"]["elements"][0]["elements"][1]["elements"][0]["begin"],
                        "<mm:think>"
                    );
                }
                let mut accepted = Vec::new();
                let mut rejected = Vec::new();
                if label == "none" {
                    accepted.push("Just text.".to_string());
                    rejected.push(one.clone());
                } else {
                    accepted.push(format!("{prefix}{one}"));
                    if optional {
                        accepted.push(one.clone());
                        rejected.push(block(&invoke("record", "wrong")));
                    }
                    rejected.push(format!("{prefix}{}", block(&invoke("other", "2"))));
                    rejected.push(format!("{prefix}{}", block(&invoke("record", "wrong"))));
                    rejected.push(format!("{prefix}{}", block("")));
                    // These were all accepted by the unbounded grammar. Check
                    // each structural position independently of the Rust wire.
                    for (before, after) in [
                        ("\n".repeat(1024), String::new()),
                        (String::new(), " ".repeat(1024)),
                    ] {
                        rejected.push(format!("{prefix}{}", block(&format!("{NS}<invoke name=\"record\">{before}{NS}<count>2{NS}</count>{after}{NS}</invoke>"))));
                    }
                    if parallel && label != "named" {
                        accepted.push(format!("{prefix}{two}"));
                    } else {
                        rejected.push(format!("{prefix}{two}"));
                        rejected.push(format!("{prefix}{one}{one}"));
                    }
                    if label == "auto" {
                        accepted.push(format!("{prefix}Just text."));
                        accepted.push(format!("{prefix}Let me check.{one}"));
                        if optional {
                            accepted.push("Just text.".to_string());
                        }
                    } else {
                        rejected.push(format!("{prefix}Just text."));
                    }
                }
                let mode = if optional {
                    "adaptive".to_string()
                } else {
                    reasoning.to_string()
                };
                cases.push(
                    json!({"case":format!("{label}/parallel={parallel}/reasoning={mode}"),
                    "tag":tag,"accepted":accepted,"rejected":rejected}),
                );
            }
        }
    }
    assert_eq!(cases.len(), 24);
    // Formatting bounds must not trim or cap whitespace within string values,
    // including nested objects and arrays.
    let literal = format!("  &amp; &#39; literal\n{} ", " ".repeat(1024));
    let complex_tools = vec![Tool {
        name: "record".into(),
        description: None,
        strict: Some(true),
        parameters: json!({"type":"object","properties":{
            "text":{"type":"string","enum":[literal]},
            "payload":{"type":"object","properties":{"count":{"type":"integer"}},"required":["count"],"additionalProperties":false},
            "values":{"type":"array","items":{"type":"integer"}}
        },"required":["text","payload","values"],"additionalProperties":false}),
    }];
    let tag = builder
        .build(&StructuralTagContext {
            tool_choice: StructuralTagToolChoice::Required,
            tools: &complex_tools,
            parallel_tool_calls: Some(false),
            schema_mode: StructuralTagSchemaMode::Auto,
            structured_output_schema: None,
            starts_in_reasoning: false,
            allow_unprefilled_reasoning: false,
        })
        .unwrap()
        .unwrap();
    let body = format!(
        "{NS}<text>{literal}{NS}</text>{NS}<payload>{NS}<count>2{NS}</count>{NS}</payload>{NS}<values>{NS}<item>1{NS}</item>{NS}<item>2{NS}</item>{NS}</values>"
    );
    let wire = block(&format!("{NS}<invoke name=\"record\">{body}{NS}</invoke>"));
    cases.push(json!({"case":"bounded-whitespace-preserves-literals-and-nesting", "tag":tag,
        "accepted":[wire],"rejected":[
            wire.replace(&format!("{NS}</text>{NS}<payload>"), &format!("{NS}</text>{}{NS}<payload>", "\n".repeat(1024)))
        ]}));
    // Eligibility must agree with the actual compiler, including definitions
    // reached through references and declarations that are never used.
    for referenced in [false, true] {
        let mut tools = tools.clone();
        tools[0].parameters["$defs"] = json!({
            "count":{"type":"integer"},
            "unused":{"type":"string"}
        });
        if referenced {
            tools[0].parameters["properties"]["count"] = json!({"$ref":"#/$defs/count"});
        }
        let tag = builder
            .build(&StructuralTagContext {
                tool_choice: StructuralTagToolChoice::Required,
                tools: &tools,
                parallel_tool_calls: Some(false),
                schema_mode: StructuralTagSchemaMode::Auto,
                structured_output_schema: None,
                starts_in_reasoning: false,
                allow_unprefilled_reasoning: false,
            })
            .unwrap()
            .unwrap();
        cases.push(
            json!({"case":format!("definitions/referenced={referenced}"), "tag":tag,
            "accepted":[one], "rejected":[block(&invoke("record", "wrong"))]}),
        );
    }
    // Eligibility must agree with the compiler on the constrained-node rule as
    // well: every schema the walk declines as unconstrained must fail to compile,
    // and untyped nodes that carry a shape keyword must compile and accept their
    // native wire. The last row is a nested union the parser can type.
    for (label, schema, accepted) in [
        ("annotation-only", json!({"description":"any value"}), None),
        ("bare-bound", json!({"minimum":1}), None),
        (
            "union-with-annotation-branch",
            json!({"anyOf":[{"type":"integer"},{"description":"x"}]}),
            None,
        ),
        (
            "untyped-properties",
            json!({"properties":{"a":{"type":"integer"}},"required":["a"],"additionalProperties":false}),
            Some(format!("{NS}<a>1{NS}</a>")),
        ),
        (
            "untyped-items",
            json!({"items":{"type":"integer"}}),
            Some(format!("{NS}<item>1{NS}</item>")),
        ),
        ("untyped-const", json!({"const":1}), Some("1".to_string())),
        (
            "nullable-object-union",
            json!({"anyOf":[{"type":"object","properties":{"n":{"type":"integer"}}},{"type":"null"}]}),
            Some(format!("{NS}<n>1{NS}</n>")),
        ),
    ] {
        let mut tools = tools.clone();
        tools[0].parameters["properties"]["count"] = schema;
        let result = builder.build(&StructuralTagContext {
            tool_choice: StructuralTagToolChoice::Required,
            tools: &tools,
            parallel_tool_calls: Some(false),
            schema_mode: StructuralTagSchemaMode::Auto,
            structured_output_schema: None,
            starts_in_reasoning: false,
            allow_unprefilled_reasoning: false,
        });
        match accepted {
            Some(value) => cases.push(json!({
                "case": format!("eligibility/{label}"),
                "tag": result.unwrap().unwrap(),
                "accepted": [block(&invoke("record", &value))],
                "rejected": [block(&invoke("record", "wrong"))],
            })),
            None => {
                assert!(result.is_err(), "eligibility/{label} must select fallback");
                cases.push(json!({
                    "case": format!("eligibility/{label}"),
                    "declined": true,
                    "json_schema": tools[0].parameters,
                }));
            }
        }
    }
    if let Some(path) = std::env::var_os("MINIMAX_M3_XGRAMMAR_CASES") {
        std::fs::write(path, serde_json::to_vec_pretty(&cases).unwrap()).unwrap();
    }
}

#[test]
fn unconstrained_native_schema_requests_explicit_fallback() {
    let builder = structural_tag_builder_for_family("minimax_m3").unwrap();
    let mut tools = tools();
    for strict in [None, Some(false)] {
        tools[0].strict = strict;
        let error = builder
            .build(&StructuralTagContext {
                tool_choice: StructuralTagToolChoice::Required,
                tools: &tools,
                parallel_tool_calls: None,
                schema_mode: StructuralTagSchemaMode::Auto,
                structured_output_schema: None,
                starts_in_reasoning: false,
                allow_unprefilled_reasoning: false,
            })
            .unwrap_err();
        assert!(error.to_string().contains("constrained tool schema"));
    }
}

#[test]
fn unsupported_native_schema_features_preserve_json_fallback() {
    let builder = structural_tag_builder_for_family("minimax_m3").unwrap();
    for schema in [
        json!({"type":"object", "additionalProperties":true}),
        json!({"type":"object", "properties":{"text":{"type":"string","minLength":1}}}),
        json!({"type":"object", "properties":{"text":{"type":"string"}}}),
        json!({"type":"object", "properties":{"values":{"type":"array","items":{"type":"string"}}}}),
        json!({"type":"object", "properties":{"payload":{"type":"object","properties":{"text":{"type":"string"}}}}}),
        json!({"type":"object", "properties":{"text":{"$ref":"#/$defs/text"}}, "$defs":{"text":{"type":"string"}}}),
        json!({"type":"object", "properties":{"text":{"type":"string","pattern":"[a-z]+"}}}),
        json!({"type":"object", "properties":{"text":{"type":"string","format":"date"}}}),
        json!({"type":"object", "properties":{"value":true}}),
        json!({"type":"object", "properties":{"value":{"const":"]<]minimax[>[literal"}}}),
        json!({"type":"object", "properties":{"value":{"const":"Montréal"}}}),
        json!({"type":"object", "properties":{"value":{"type":"string","enum":["雪"]}}}),
        json!({"type":"object", "properties":{"雪":{"type":"integer"}}}),
    ] {
        let mut tools = tools();
        tools[0].parameters = schema.clone();
        let result = builder.build(&StructuralTagContext {
            tool_choice: StructuralTagToolChoice::Required,
            tools: &tools,
            parallel_tool_calls: None,
            schema_mode: StructuralTagSchemaMode::Auto,
            structured_output_schema: None,
            starts_in_reasoning: false,
            allow_unprefilled_reasoning: false,
        });
        assert!(
            result.is_err(),
            "unsupported schema must select fallback: {schema}"
        );
        assert_eq!(tools[0].parameters, schema);
    }
}

#[test]
fn ambiguous_string_union_types_select_json_without_changing_the_schema() {
    // M3 XML uses the same bytes for, e.g., the string "42" and the integer 42.
    // Guided JSON preserves the requested type when both are schema-valid.
    let builder = structural_tag_builder_for_family("minimax_m3").unwrap();
    let mut schemas = vec![json!({"enum":["42",42]})];
    for other in ["integer", "number", "boolean", "object", "array", "null"] {
        schemas.push(json!({"type":["string",other]}));
        for keyword in ["anyOf", "oneOf"] {
            schemas.push(json!({keyword:[{"type":"string"},{"type":other}]}));
        }
    }
    for schema in schemas {
        for referenced in [false, true] {
            let mut tools = tools();
            tools[0].parameters = if referenced {
                json!({"type":"object", "properties":{"value":{"$ref":"#/$defs/value"}},
                    "$defs":{"value":schema},"required":["value"],"additionalProperties":false})
            } else {
                json!({"type":"object", "properties":{"value":schema},
                    "required":["value"],"additionalProperties":false})
            };
            let before = tools[0].parameters.clone();
            let result = builder.build(&StructuralTagContext {
                tool_choice: StructuralTagToolChoice::Required,
                tools: &tools,
                parallel_tool_calls: None,
                schema_mode: StructuralTagSchemaMode::Auto,
                structured_output_schema: None,
                starts_in_reasoning: false,
                allow_unprefilled_reasoning: false,
            });
            assert!(
                result.is_err(),
                "ambiguous native schema must select JSON: {before}"
            );
            assert_eq!(tools[0].parameters, before);
        }
    }
}
