// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[path = "../../tests/glm47_schema_matrix.rs"]
mod matrix;

use dynamo_parsers_v2::structural_tag::{
    StructuralTagContext, StructuralTagSchemaMode, StructuralTagToolChoice,
};
use dynamo_parsers_v2::unified::{
    UnifiedEvent, UnifiedParserExt, assemble, create_unified_parser_for_family,
};
use dynamo_parsers_v2::{
    Glm47ToolStreamParser, Tool, ToolParseResult, ToolParser, structural_tag_builder_for_family,
};
use serde_json::{Value, json};

fn tool(case: &matrix::Case) -> Tool {
    Tool {
        name: "probe".into(),
        parameters: case.parameters(),
        description: None,
        strict: Some(true),
    }
}

fn build(tools: &[Tool], mode: StructuralTagSchemaMode) -> Value {
    structural_tag_builder_for_family("glm47")
        .unwrap()
        .build(&StructuralTagContext {
            tools,
            tool_choice: StructuralTagToolChoice::Named("probe"),
            parallel_tool_calls: Some(false),
            schema_mode: mode,
            structured_output_schema: None,
            starts_in_reasoning: false,
        })
        .unwrap()
        .unwrap()
}

fn schedules(input: &str) -> Vec<Vec<&str>> {
    let boundaries: Vec<usize> = input
        .char_indices()
        .map(|(at, _)| at)
        .chain([input.len()])
        .collect();
    let mut schedules = vec![
        vec![input],
        boundaries
            .windows(2)
            .map(|pair| &input[pair[0]..pair[1]])
            .collect(),
    ];
    for marker in [
        "<tool_call>",
        "<arg_key>",
        "</arg_key>",
        "<arg_value>",
        "</arg_value>",
        "</tool_call>",
    ] {
        let at = input.find(marker).unwrap() + marker.len() / 2;
        schedules.push(vec![&input[..at], &input[at..]]);
    }
    let start = input.find("<arg_value>").unwrap() + "<arg_value>".len();
    let end = input.find("</arg_value>").unwrap();
    if let Some(at) = boundaries.into_iter().find(|at| *at >= (start + end) / 2) {
        schedules.push(vec![&input[..at], &input[at..]]);
    }
    schedules
}

fn legacy_arguments(case: &matrix::Case, chunks: &[&str]) -> Value {
    let mut parser = Glm47ToolStreamParser::new(&[tool(case)]);
    let mut result = ToolParseResult::default();
    for chunk in chunks {
        result.append(parser.push(chunk).unwrap());
    }
    result.append(parser.finish().unwrap());
    let result = result.coalesce_calls();
    assert_eq!(result.calls.len(), 1, "{}", case.name);
    serde_json::from_str(&result.calls[0].arguments).unwrap()
}

fn unified_arguments(case: &matrix::Case, chunks: &[&str]) -> Value {
    let mut parser = create_unified_parser_for_family("glm47", &[tool(case)]).unwrap();
    let mut events = Vec::new();
    for chunk in chunks {
        events.extend(parser.push(chunk).unwrap());
    }
    events.extend(parser.finish().unwrap().events);
    let events = assemble(&events);
    assert_eq!(events.len(), 1, "{}", case.name);
    match &events[0] {
        UnifiedEvent::ToolCall { name, arguments } => {
            assert_eq!(name, "probe");
            arguments.clone()
        }
        event => panic!("{}: unexpected event {event:?}", case.name),
    }
}

#[test]
fn authored_schema_matrix_matches_legacy_and_unified_at_chunk_boundaries() {
    let mut failures = Vec::new();
    let mut results = Vec::new();
    for case in matrix::cases() {
        let input = case.wire();
        for (index, chunks) in schedules(&input).into_iter().enumerate() {
            for (mode, actual) in [
                ("legacy", legacy_arguments(&case, &chunks)),
                ("unified", unified_arguments(&case, &chunks)),
            ] {
                if actual != case.expected() {
                    failures.push(format!(
                        "{} {mode} schedule{index}: expected {}, got {}",
                        case.name,
                        case.expected(),
                        actual
                    ));
                }
            }
        }
        let mut parser = Glm47ToolStreamParser::new(&[tool(&case)]);
        let result = parser.parse_complete(&input).unwrap();
        let actual = serde_json::from_str::<Value>(&result.calls[0].arguments).unwrap();
        results.push(json!({"name":case.name,"schema":case.parameters(),"wire":case.wire(),
            "expected":case.expected(),"actual":actual,"unified":unified_arguments(&case, &[&input]),
            "tag":build(&[tool(&case)], StructuralTagSchemaMode::Auto)}));
        if actual != case.expected() {
            failures.push(format!(
                "{} batch-on-stream: expected {}, got {}",
                case.name,
                case.expected(),
                actual
            ));
        }
    }
    matrix::write_report("v2", &results);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
#[ignore = "requires python3 with xgrammar and jsonschema"]
fn authored_schemas_accept_valid_and_reject_invalid_wires() {
    let mut probes = Vec::new();
    for case in matrix::cases().into_iter().filter(|case| case.grammar) {
        let input = case.wire();
        let actual = legacy_arguments(&case, &[&input]);
        assert_eq!(actual, case.expected(), "{}", case.name);
        assert_eq!(unified_arguments(&case, &[&input]), actual, "{}", case.name);
        probes.push(case.probe(build(&[tool(&case)], StructuralTagSchemaMode::Auto), actual));
    }
    let case = matrix::cases()
        .into_iter()
        .find(|case| case.name == "const_plain_bare")
        .unwrap();
    for strict in [None, Some(false), Some(true)] {
        for mode in [
            StructuralTagSchemaMode::Auto,
            StructuralTagSchemaMode::Strict,
        ] {
            let mut tool = tool(&case);
            tool.strict = strict;
            let enforce = mode == StructuralTagSchemaMode::Strict || strict == Some(true);
            let mut probe = case.probe(build(&[tool], mode), case.expected());
            probe["name"] = json!(format!("strict_{strict:?}_{mode:?}"));
            if !enforce {
                probe["accept"] = probe["reject"].take();
                probe["reject"] = json!([]);
            }
            probes.push(probe);
        }
    }
    matrix::check_grammar(&probes);
}

#[test]
fn schema_modes_pin_existing_v2_policy() {
    let schema = json!({"type":"object", "properties":{"value":{"const":"auto"}}, "required":["value"], "additionalProperties":false});
    for strict in [None, Some(false), Some(true)] {
        for mode in [
            StructuralTagSchemaMode::Auto,
            StructuralTagSchemaMode::Strict,
        ] {
            let tools = [Tool {
                name: "probe".into(),
                parameters: schema.clone(),
                description: None,
                strict,
            }];
            let enforce = mode == StructuralTagSchemaMode::Strict || strict == Some(true);
            assert_eq!(
                build(&tools, mode)["format"]["content"]["json_schema"],
                if enforce { schema.clone() } else { json!(true) }
            );
        }
    }
}
