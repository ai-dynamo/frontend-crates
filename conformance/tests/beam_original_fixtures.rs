// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The original 13 Beam parser fixtures (11 positive, 2 strict rejections), unchanged in
//! `fixtures-beam/original_parser_fixtures.yaml`, driven through the PUBLIC V2 route:
//! the `beam` unified parser, whole and split at every char boundary, under both the
//! default (`Drop`) and strict (`Reject`) tool-call policies.

use dynamo_parsers_v2::{
    MalformedToolCallPolicy, Tool, ToolCallPolicy, UnifiedEvent, UnifiedParserExt,
    create_unified_parser_for_family,
};
use serde_json::Value;

mod common;

const FIXTURES: &str = include_str!("../fixtures-beam/original_parser_fixtures.yaml");

struct Case {
    id: String,
    text: String,
    /// `(name, parameters)` in request order.
    tools: Vec<(String, Value)>,
    expected: Option<Expected>,
}

struct Expected {
    reasoning: String,
    content: String,
    calls: Vec<(String, Value)>,
}

fn load() -> Vec<Case> {
    let doc: serde_yaml::Value = serde_yaml::from_str(FIXTURES).expect("fixture yaml");
    let json: Value = serde_json::to_value(&doc).expect("yaml to json");
    let section = |key: &str, positive: bool| -> Vec<Case> {
        json[key]
            .as_array()
            .unwrap_or_else(|| panic!("missing {key}"))
            .iter()
            .map(|case| {
                let tools = case["input"]["tools"]
                    .as_array()
                    .expect("tools")
                    .iter()
                    .map(|t| {
                        (
                            t["function"]["name"].as_str().unwrap().to_string(),
                            t["function"]["parameters"].clone(),
                        )
                    })
                    .collect();
                let expected = positive.then(|| {
                    let e = &case["expected"];
                    Expected {
                        reasoning: e["reasoning_content"].as_str().unwrap_or("").to_string(),
                        content: e["content"].as_str().unwrap_or("").to_string(),
                        calls: e["tool_calls"]
                            .as_array()
                            .expect("tool_calls")
                            .iter()
                            .map(|c| {
                                (
                                    c["name"].as_str().unwrap().to_string(),
                                    c["arguments"].clone(),
                                )
                            })
                            .collect(),
                    }
                });
                Case {
                    id: case["case_id"].as_str().unwrap().to_string(),
                    text: case["input"]["text"].as_str().unwrap().to_string(),
                    tools,
                    expected,
                }
            })
            .collect()
    };
    let mut all = section("message_cases", true);
    all.extend(section("rejection_cases", false));
    all
}

fn v2_tools(case: &Case) -> Vec<Tool> {
    case.tools
        .iter()
        .map(|(name, parameters)| Tool {
            name: name.clone(),
            description: None,
            parameters: parameters.clone(),
            strict: None,
        })
        .collect()
}

fn splits(text: &str) -> Vec<Vec<&str>> {
    let mut all = vec![vec![text]];
    for at in (0..=text.len()).filter(|&at| text.is_char_boundary(at)) {
        all.push(vec![&text[..at], &text[at..]]);
    }
    all
}

fn run_v2(case: &Case, chunks: &[&str], policy: ToolCallPolicy) -> anyhow::Result<Expected> {
    let mut parser = create_unified_parser_for_family("beam", &v2_tools(case))?;
    parser.set_tool_call_policy(policy)?;
    let mut deltas = Vec::new();
    for chunk in chunks {
        deltas.extend(parser.push(chunk)?);
    }
    deltas.extend(parser.finish()?);
    let mut out = Expected {
        reasoning: String::new(),
        content: String::new(),
        calls: Vec::new(),
    };
    for event in dynamo_parsers_v2::assemble(&deltas) {
        match event {
            UnifiedEvent::MessageStart { .. } | UnifiedEvent::MessageEnd => {}
            UnifiedEvent::Reasoning { text } => out.reasoning += &text,
            UnifiedEvent::Text { text } => out.content += &text,
            UnifiedEvent::ToolCall { name, arguments } => out.calls.push((name, arguments)),
        }
    }
    Ok(out)
}

fn strict() -> ToolCallPolicy {
    ToolCallPolicy {
        malformed: MalformedToolCallPolicy::Reject,
        ..Default::default()
    }
}

#[test]
fn fixtures_load_with_the_original_counts() {
    let cases = load();
    assert_eq!(cases.iter().filter(|c| c.expected.is_some()).count(), 11);
    assert_eq!(cases.iter().filter(|c| c.expected.is_none()).count(), 2);
}

#[test]
fn v2_unified_matches_every_positive_fixture_at_every_split() {
    for case in load().iter().filter(|c| c.expected.is_some()) {
        let want = case.expected.as_ref().unwrap();
        for policy in [ToolCallPolicy::default(), strict()] {
            for chunks in splits(&case.text) {
                let got = run_v2(case, &chunks, policy)
                    .unwrap_or_else(|e| panic!("{} {chunks:?}: {e}", case.id));
                assert_eq!(got.reasoning, want.reasoning, "{} reasoning", case.id);
                assert_eq!(got.content, want.content, "{} content", case.id);
                assert!(
                    common::json_values_equal(&got.calls, &want.calls),
                    "{} calls: got {:?}, want {:?}",
                    case.id,
                    got.calls,
                    want.calls
                );
            }
        }
    }
}

#[test]
fn v2_unified_strict_rejects_and_default_drops_the_rejection_fixtures() {
    for case in load().iter().filter(|c| c.expected.is_none()) {
        for chunks in splits(&case.text) {
            assert!(
                run_v2(case, &chunks, strict()).is_err(),
                "{} {chunks:?} must be rejected",
                case.id
            );
            let got = run_v2(case, &chunks, ToolCallPolicy::default()).unwrap();
            assert!(got.calls.is_empty(), "{} must not dispatch", case.id);
        }
    }
}
