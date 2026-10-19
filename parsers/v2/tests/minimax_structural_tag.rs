// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Run the executable grammar check with:
//! MINIMAX_GRAMMAR_CASES=/tmp/minimax.json cargo test -p dynamo-parsers-v2 --test minimax_structural_tag
//! uv run --no-project --with xgrammar==0.2.8 python parsers/v2/tests/check_minimax_xgrammar.py /tmp/minimax.json

use dynamo_parsers_v2::structural_tag::{
    StructuralTagContext, StructuralTagOptions, StructuralTagSchemaMode, StructuralTagToolChoice,
};
use dynamo_parsers_v2::{Tool, create_tool_parser_for_family, structural_tag_builder_for_family};
use serde_json::{Value, json};

fn context(tools: &[Tool]) -> StructuralTagContext<'_> {
    StructuralTagContext {
        tools,
        tool_choice: StructuralTagToolChoice::Auto,
        parallel_tool_calls: None,
        schema_mode: StructuralTagSchemaMode::Auto,
        structured_output_schema: None,
        starts_in_reasoning: false,
    }
}

fn block(family: &str, name: &str, value: &str, count: usize) -> String {
    assert_eq!(family, "minimax_m2");
    let (open, close, invoke, end) = (
        "<minimax:tool_call>",
        "</minimax:tool_call>",
        "<invoke name=\"",
        "</invoke>",
    );
    let args = format!("<parameter name=\"days\">{value}</parameter>");
    let call = format!("{invoke}{name}\">\n{args}\n{end}");
    format!("{open}\n{}\n{close}", vec![call; count].join("\n"))
}

#[test]
fn minimax_grammars_and_native_parsers_agree() {
    let tools = [Tool {
        name: "weather".into(),
        description: None,
        strict: Some(true),
        parameters: json!({"type":"object","properties":{"days":{"type":"integer"}},"required":["days"],"additionalProperties":false}),
    }];
    let mut cases = Vec::new();
    for family in ["minimax_m2"] {
        let builder = structural_tag_builder_for_family(family).expect("missing MiniMax builder");
        for choice in [
            StructuralTagToolChoice::Auto,
            StructuralTagToolChoice::Required,
            StructuralTagToolChoice::Named("weather"),
        ] {
            for parallel in [None, Some(false), Some(true)] {
                let mut ctx = context(&tools);
                ctx.tool_choice = choice;
                ctx.parallel_tool_calls = parallel;
                let single = block(family, "weather", "2", 1);
                let multiple = block(family, "weather", "2", 2);
                let separate_blocks = format!("{single}\n{single}");
                let mut accepted = vec![single.clone()];
                let mut rejected = vec![
                    block(family, "missing", "2", 1),
                    block(family, "weather", "false", 1),
                    block(family, "weather", "2", 0),
                ];
                if parallel != Some(false) && !matches!(choice, StructuralTagToolChoice::Named(_)) {
                    accepted.extend([multiple, separate_blocks]);
                } else {
                    rejected.extend([multiple, separate_blocks]);
                }
                if choice == StructuralTagToolChoice::Auto {
                    accepted.push("hello".into());
                } else {
                    rejected.push("hello".into());
                }
                for sample in &accepted {
                    // Every byte boundary must preserve the same parsed arguments and visible text.
                    let expected_count = sample.matches("<invoke name=").count();
                    for split in 0..=sample.len() {
                        let mut parser = create_tool_parser_for_family(family, &tools).unwrap();
                        let mut result = parser.push(&sample[..split]).unwrap();
                        result.append(parser.push(&sample[split..]).unwrap());
                        result.append(parser.finish().unwrap());
                        let result = result.coalesce_calls();
                        assert_eq!(result.calls.len(), expected_count, "{family} at {split}");
                        assert_eq!(
                            result.normal_text,
                            if expected_count == 0 {
                                "hello"
                            } else if sample.contains("</minimax:tool_call>\n<minimax:tool_call>") {
                                "\n"
                            } else {
                                ""
                            }
                        );
                        for call in result.calls {
                            assert_eq!(call.name.as_deref(), Some("weather"));
                            assert_eq!(
                                serde_json::from_str::<Value>(&call.arguments).unwrap(),
                                json!({"days":2})
                            );
                        }
                    }
                }
                cases.push(json!({"family":family,"tag":builder.build(&ctx).unwrap().unwrap(),"accepted":accepted,"rejected":rejected}));
            }
        }
        let empty_tools = [Tool {
            name: "ping".into(),
            description: None,
            strict: Some(true),
            parameters: json!({"type":"object","properties":{},"additionalProperties":false}),
        }];
        let empty_call =
            "<minimax:tool_call>\n<invoke name=\"ping\">\n\n</invoke>\n</minimax:tool_call>";
        let mut empty_ctx = context(&empty_tools);
        empty_ctx.tool_choice = StructuralTagToolChoice::Named("ping");
        cases.push(json!({"family":family,"tag":builder.build(&empty_ctx).unwrap().unwrap(),
            "accepted":[empty_call], "rejected":[empty_call.replace("\n\n", "\n<parameter name=\"extra\">1</parameter>\n")]}));
        let empty_parsed = create_tool_parser_for_family(family, &empty_tools)
            .unwrap()
            .parse_complete(empty_call)
            .unwrap()
            .coalesce_calls();
        assert_eq!(empty_parsed.calls.len(), 1);
        assert_eq!(
            serde_json::from_str::<Value>(&empty_parsed.calls[0].arguments).unwrap(),
            json!({})
        );
        let mut ctx = context(&tools);
        ctx.tool_choice = StructuralTagToolChoice::Required;
        ctx.starts_in_reasoning = true;
        let valid = block(family, "weather", "2", 1);
        cases.push(
            json!({"family":family,"tag":builder.build(&ctx).unwrap().unwrap(),
            "accepted":[format!("thinking</think>{valid}")], "rejected":[valid]}),
        );
        ctx.starts_in_reasoning = false;
        ctx.tool_choice = StructuralTagToolChoice::None;
        cases.push(
            json!({"family":family,"tag":builder.build(&ctx).unwrap().unwrap(),
            "accepted":["hello"],"rejected":[block(family,"weather","2",1)]}),
        );
        // The new native styles must honor the shared schema policy as well.
        for strict in [None, Some(false), Some(true)] {
            let mut policy_tools = tools.clone();
            policy_tools[0].strict = strict;
            for mode in [
                StructuralTagSchemaMode::Auto,
                StructuralTagSchemaMode::Strict,
            ] {
                let mut policy_ctx = context(&policy_tools);
                policy_ctx.schema_mode = mode;
                let enforced = strict == Some(true) || mode == StructuralTagSchemaMode::Strict;
                let wrong_type = block(family, "weather", "false", 1);
                cases.push(
                    json!({"family":family,"tag":builder.build(&policy_ctx).unwrap().unwrap(),
                    "accepted": if enforced { vec![] } else { vec![wrong_type.clone()] },
                    "rejected": if enforced { vec![wrong_type] } else { vec![] }}),
                );
            }
        }
        let response_schema = json!({"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"],"additionalProperties":false});
        let mut structured = context(&tools);
        structured.structured_output_schema = Some(&response_schema);
        cases.push(
            json!({"family":family,"tag":builder.build(&structured).unwrap().unwrap(),
            "accepted":[block(family,"weather","2",1), "{\"ok\":true}"],
            "rejected":["{\"ok\":2}", "hello"]}),
        );
        assert!(builder.build(&context(&[])).unwrap().is_none());
        ctx.tool_choice = StructuralTagToolChoice::Named("missing");
        assert!(builder.build(&ctx).is_err());
        // Disabling optional exclusions must not bypass the constrained tools branch.
        ctx.tool_choice = StructuralTagToolChoice::Auto;
        cases.push(json!({"family":family,"tag":builder.build_with_options(&ctx,&StructuralTagOptions {
            exclude_special_tokens:Some(false), ..Default::default()
        }).unwrap().unwrap(),"accepted":[block(family,"weather","2",1)],"rejected":[block(family,"missing","2",1)]}));
    }
    if let Some(path) = std::env::var_os("MINIMAX_GRAMMAR_CASES") {
        std::fs::write(path, serde_json::to_vec_pretty(&cases).unwrap()).unwrap();
    }
}
