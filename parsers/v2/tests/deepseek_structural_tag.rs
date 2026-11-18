// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_parsers_v2::structural_tag::{
    StructuralTagContext, StructuralTagOptions, StructuralTagSchemaMode, StructuralTagToolChoice,
};
use dynamo_parsers_v2::{
    Tool, UnifiedEvent, UnifiedParserExt, UnifiedParserInit, UnifiedParserStartingState, assemble,
    create_tool_parser_for_family, create_unified_parser_for_family,
    structural_tag_builder_for_family,
};
use serde_json::json;

fn parameter(spaced: bool, name: &str, string: bool, value: &str) -> String {
    let gap = if spaced { " " } else { "" };
    format!(
        "<｜DSML｜{gap}parameter name=\"{name}\" string=\"{string}\">{value}</｜DSML｜{gap}parameter>\n"
    )
}

fn invoke(spaced: bool, name: &str, arguments: &str) -> String {
    let gap = if spaced { " " } else { "" };
    format!("<｜DSML｜{gap}invoke name=\"{name}\">\n{arguments}</｜DSML｜{gap}invoke>\n")
}

fn block(spaced: bool, calls: &str) -> String {
    let tag = if spaced { " calls" } else { "tool_calls" };
    format!("\n\n<｜DSML｜{tag}>\n{calls}</｜DSML｜{tag}>")
}

fn assert_parser(family: &str, tools: &[Tool], sample: &str, expected: &[UnifiedEvent]) {
    assert_parser_in_state(
        family,
        tools,
        sample,
        expected,
        UnifiedParserStartingState::None,
    );
}

fn assert_parser_in_state(
    family: &str,
    tools: &[Tool],
    sample: &str,
    expected: &[UnifiedEvent],
    state: UnifiedParserStartingState,
) {
    let mut parser = create_unified_parser_for_family(family, tools).unwrap();
    parser
        .initialize_request(UnifiedParserInit {
            starting_state: state,
            ..Default::default()
        })
        .unwrap();
    assert_eq!(parser.parse_complete(sample).unwrap(), expected, "{sample}");
    for split in sample
        .char_indices()
        .map(|(at, _)| at)
        .chain([sample.len()])
    {
        let mut parser = create_unified_parser_for_family(family, tools).unwrap();
        parser
            .initialize_request(UnifiedParserInit {
                starting_state: state,
                ..Default::default()
            })
            .unwrap();
        let mut events = parser.push(&sample[..split]).unwrap();
        events.extend(parser.push(&sample[split..]).unwrap());
        events.extend(parser.finish().unwrap());
        assert_eq!(assemble(&events), expected, "{family} split {split}");
    }
}

#[test]
fn deepseek_grammar_acceptance_corpus() {
    let tools = [
        Tool {
            name: "weather".into(),
            description: None,
            strict: Some(true),
            parameters: json!({"type":"object", "properties": {
                "city":{"type":"string"}, "days":{"type":"integer"},
                "options":{"type":"object", "properties":{"units":{"type":"string"}},
                    "required":["units"], "additionalProperties":false}
            }, "required":["city","days","options"], "additionalProperties":false}),
        },
        Tool {
            name: "empty".into(),
            description: None,
            strict: Some(true),
            parameters: json!({"type":"object", "properties":{}, "additionalProperties":false}),
        },
    ];
    let expected = UnifiedEvent::ToolCall {
        name: "weather".into(),
        arguments: json!({"city":"Paris", "days":3, "options":{"units":"C"}}),
    };
    let empty = UnifiedEvent::ToolCall {
        name: "empty".into(),
        arguments: json!({}),
    };
    let mut cases = Vec::new();
    for (family, spaced) in [("deepseek_v4", false), ("deepseek_v41", true)] {
        let builder = structural_tag_builder_for_family(family).expect("DSML grammar lookup");
        if spaced {
            assert!(create_tool_parser_for_family(family, &tools).is_err());
        }
        for choice in [
            StructuralTagToolChoice::Auto,
            StructuralTagToolChoice::Required,
            StructuralTagToolChoice::Named("weather"),
        ] {
            for parallel in [None, Some(false), Some(true)] {
                for any_order in [false, true] {
                    let ctx = StructuralTagContext {
                        tools: &tools,
                        tool_choice: choice,
                        parallel_tool_calls: parallel,
                        schema_mode: StructuralTagSchemaMode::Auto,
                        structured_output_schema: None,
                        starts_in_reasoning: false,
                    };
                    let tag = builder
                        .build_with_options(
                            &ctx,
                            &StructuralTagOptions {
                                tool_arguments_any_order: any_order,
                                ..Default::default()
                            },
                        )
                        .unwrap()
                        .unwrap();
                    let args = format!(
                        "{}{}{}",
                        parameter(spaced, "city", true, "Paris"),
                        parameter(spaced, "days", false, "3"),
                        parameter(spaced, "options", false, r#"{"units":"C"}"#)
                    );
                    let call = invoke(spaced, "weather", &args);
                    let mut accepted = vec![block(spaced, &call)];
                    let mut rejected = vec![
                        block(spaced, &invoke(spaced, "unknown", &args)),
                        block(
                            spaced,
                            &invoke(spaced, "weather", &parameter(spaced, "city", true, "Paris")),
                        ),
                        block(
                            spaced,
                            &invoke(spaced, "weather", &args.replace(">3<", ">false<")),
                        ),
                        block(
                            spaced,
                            &invoke(
                                spaced,
                                "weather",
                                &args.replace(r#"{"units":"C"}"#, r#"{"units":3}"#),
                            ),
                        ),
                    ];
                    if !matches!(choice, StructuralTagToolChoice::Auto) {
                        rejected.push(block(
                            !spaced,
                            &invoke(
                                !spaced,
                                "weather",
                                &format!(
                                    "{}{}{}",
                                    parameter(!spaced, "city", true, "Paris"),
                                    parameter(!spaced, "days", false, "3"),
                                    parameter(!spaced, "options", false, r#"{"units":"C"}"#)
                                ),
                            ),
                        ));
                    }
                    let multiple = block(spaced, &format!("{call}{call}"));
                    if parallel == Some(false)
                        || matches!(choice, StructuralTagToolChoice::Named(_))
                    {
                        rejected.push(multiple);
                    } else {
                        accepted.push(multiple);
                    }
                    let reordered = block(
                        spaced,
                        &invoke(
                            spaced,
                            "weather",
                            &format!(
                                "{}{}{}",
                                parameter(spaced, "days", false, "3"),
                                parameter(spaced, "city", true, "Paris"),
                                parameter(spaced, "options", false, r#"{"units":"C"}"#)
                            ),
                        ),
                    );
                    if any_order {
                        accepted.push(reordered);
                    } else {
                        rejected.push(reordered);
                    }
                    let empty_sample = block(spaced, &invoke(spaced, "empty", ""));
                    if matches!(choice, StructuralTagToolChoice::Named(_)) {
                        rejected.push(empty_sample);
                    } else {
                        accepted.push(empty_sample);
                    }
                    for sample in &accepted {
                        let events = if sample.contains("name=\"empty\"") {
                            vec![empty.clone()]
                        } else if sample.matches("name=\"weather\"").count() == 2 {
                            vec![expected.clone(), expected.clone()]
                        } else {
                            vec![expected.clone()]
                        };
                        assert_parser(family, &tools, sample, &events);
                    }
                    cases.push(
                        json!({"family":family,"tag":tag,"accepted":accepted,"rejected":rejected}),
                    );
                }
            }
        }
    }
    for (family, spaced) in [("deepseek_v4", false), ("deepseek_v41", true)] {
        let builder = structural_tag_builder_for_family(family).unwrap();
        let ctx = StructuralTagContext {
            tools: &tools,
            tool_choice: StructuralTagToolChoice::None,
            parallel_tool_calls: None,
            schema_mode: StructuralTagSchemaMode::Auto,
            structured_output_schema: None,
            starts_in_reasoning: false,
        };
        let sample = block(spaced, &invoke(spaced, "empty", ""));
        assert_parser(
            family,
            &tools,
            "Plain response",
            &[UnifiedEvent::Text {
                text: "Plain response".into(),
            }],
        );
        cases.push(
            json!({"family":family,"tag":builder.build(&ctx).unwrap().unwrap(),
            "accepted":["Plain response"],"rejected":[sample]}),
        );
        for choice in [
            StructuralTagToolChoice::Required,
            StructuralTagToolChoice::Named("empty"),
        ] {
            let ctx = StructuralTagContext {
                tool_choice: choice,
                starts_in_reasoning: true,
                ..ctx
            };
            let accepted = format!("Check first.</think>{sample}");
            let rejected = format!(
                "Check first.</think>{}",
                block(spaced, &invoke(spaced, "unknown", ""))
            );
            let events = [
                UnifiedEvent::Reasoning {
                    text: "Check first.".into(),
                },
                empty.clone(),
            ];
            assert_parser_in_state(
                family,
                &tools,
                &accepted,
                &events,
                UnifiedParserStartingState::Reasoning,
            );
            cases.push(
                json!({"family":family,"tag":builder.build(&ctx).unwrap().unwrap(),
                "accepted":[accepted],"rejected":[rejected]}),
            );
        }
        for strict in [None, Some(false), Some(true)] {
            let tool = [Tool {
                strict,
                ..tools[0].clone()
            }];
            for mode in [
                StructuralTagSchemaMode::Auto,
                StructuralTagSchemaMode::Strict,
            ] {
                let ctx = StructuralTagContext {
                    tools: &tool,
                    tool_choice: StructuralTagToolChoice::Named("weather"),
                    schema_mode: mode,
                    ..ctx
                };
                let tag = builder.build(&ctx).unwrap().unwrap();
                let constrained = mode == StructuralTagSchemaMode::Strict || strict == Some(true);
                assert_eq!(
                    tag["format"]["content"]["tags"][0]["content"]["json_schema"],
                    if constrained {
                        tools[0].parameters.clone()
                    } else {
                        json!(true)
                    }
                );
                let good = block(
                    spaced,
                    &invoke(
                        spaced,
                        "weather",
                        &format!(
                            "{}{}{}",
                            parameter(spaced, "city", true, "Paris"),
                            parameter(spaced, "days", false, "3"),
                            parameter(spaced, "options", false, r#"{"units":"C"}"#)
                        ),
                    ),
                );
                let missing = block(
                    spaced,
                    &invoke(spaced, "weather", &parameter(spaced, "city", true, "Paris")),
                );
                assert_parser(family, &tool, &good, std::slice::from_ref(&expected));
                if !constrained {
                    assert_parser(
                        family,
                        &tool,
                        &missing,
                        &[UnifiedEvent::ToolCall {
                            name: "weather".into(),
                            arguments: json!({"city":"Paris"}),
                        }],
                    );
                }
                cases.push(if constrained {
                    json!({"family":family,"tag":tag,"accepted":[good],"rejected":[missing]})
                } else {
                    json!({"family":family,"tag":tag,"accepted":[good,missing],
                        "rejected":[block(spaced,&invoke(spaced,"unknown",""))]})
                });
            }
        }
    }
    if let Some(path) = std::env::var_os("DSML_GRAMMAR_CASES") {
        std::fs::write(path, serde_json::to_vec_pretty(&cases).unwrap()).unwrap();
    }
}
