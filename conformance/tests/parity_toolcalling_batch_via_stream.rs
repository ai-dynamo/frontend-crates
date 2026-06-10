// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stream parser on BATCH samples: feed each batch fixture's full
//! `model_text` to the streaming parser and assert the assembled tool calls match
//! the BATCH parser's `expected.dynamo`. This is the streaming-vs-batch
//! consistency check — the stream parser, given the complete output, must land on
//! the same calls as the batch parser.

use std::collections::BTreeMap;
use std::path::Path;

mod common;
use common::{collect_yaml, fixture_name};

use dynamo_parsers_v2::{
    HarmonyToolStreamParser, ToolCallDelta, ToolParseResult, assemble_tool_calls,
    create_tool_parser_for_family,
};
use serde::Deserialize;
use serde_json::Value;

#[derive(Deserialize)]
struct Fixture {
    family: String,
    mode: String,
    #[serde(default)]
    cases: BTreeMap<String, Case>,
}

#[derive(Deserialize)]
struct Case {
    #[serde(default)]
    model_text: Option<String>,
    #[serde(default)]
    expected: Option<Expected>,
}

#[derive(Deserialize)]
struct Expected {
    dynamo: EngineExpected,
}

#[derive(Deserialize)]
struct EngineExpected {
    #[serde(default)]
    calls: Vec<ExpCall>,
    #[serde(default)]
    normal_text: String,
}

#[derive(Deserialize)]
struct ExpCall {
    name: String,
    #[serde(default)]
    arguments: Value,
}

#[test]
fn toolcalling_batch_via_stream_parity() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/toolcalling/fixtures");
    let mut files = Vec::new();
    collect_yaml(Path::new(root), &mut files);
    files.sort();

    // Batch samples where the streaming parser deliberately differs from the
    // strict batch parser. Removing an entry asserts that stream and batch now
    // agree on that sample.
    let known_divergences: std::collections::BTreeSet<&str> = [
        "deepseek_v4:TOOLCALLING.batch.5.a",
        "deepseek_v4:TOOLCALLING.batch.5.d",
        "deepseek_v4:TOOLCALLING.batch.5.e",
        "deepseek_v4:TOOLCALLING.batch.5.g",
    ]
    .into_iter()
    .collect();

    let mut total = 0usize;
    let mut consistent = 0usize;
    let mut diverged = 0usize;
    let mut failures: Vec<String> = Vec::new();
    let mut unexpected_match: Vec<String> = Vec::new();

    for path in &files {
        let yaml = std::fs::read_to_string(path).unwrap();
        let fx: Fixture = match serde_yaml::from_str(&yaml) {
            Ok(f) => f,
            Err(e) => {
                failures.push(format!("{}: YAML parse error: {e}", path.display()));
                continue;
            }
        };
        if !(fx.family == "harmony" || fx.family == "deepseek_v4") || fx.mode != "batch" {
            continue;
        }
        eprintln!("fixture {}", fixture_name(path));

        for (cid, case) in &fx.cases {
            let (Some(text), Some(expected)) = (case.model_text.as_ref(), case.expected.as_ref())
            else {
                continue; // placeholder case
            };
            total += 1;

            let got = parse_stream_result(&fx.family, text).unwrap();
            let want = EngineResult {
                calls: expected
                    .dynamo
                    .calls
                    .iter()
                    .map(|c| (c.name.clone(), c.arguments.clone()))
                    .collect(),
                normal_text: expected.dynamo.normal_text.clone(),
            };

            let known_id = format!("{}:{cid}", fx.family);
            let known = known_divergences.contains(known_id.as_str());
            if got == want {
                consistent += 1;
                if known {
                    // It now agrees — the allowlist entry is stale.
                    unexpected_match.push(known_id);
                }
            } else {
                diverged += 1;
                if !known {
                    failures.push(format!(
                        "{} {cid}:\n        stream got {got:?}\n        batch want {want:?}",
                        fx.family
                    ));
                }
            }
        }
    }

    eprintln!(
        "Dynamo stream-on-batch: {consistent}/{total} consistent, {diverged} diverged \
         ({} are known/documented)",
        diverged - failures.len(),
    );
    for f in &failures {
        eprintln!("UNEXPECTED DIVERGENCE {f}");
    }
    for c in &unexpected_match {
        eprintln!("STALE ALLOWLIST (now agrees, drop it): {c}");
    }
    assert!(
        failures.is_empty(),
        "{} batch samples newly diverged between stream and batch (not in the \
         known-divergence allowlist)",
        failures.len()
    );
    assert!(
        unexpected_match.is_empty(),
        "{} allowlist entries now agree — remove them",
        unexpected_match.len()
    );
}

#[derive(Debug, PartialEq, Eq)]
struct EngineResult {
    calls: Vec<(String, Value)>,
    normal_text: String,
}

fn parse_stream_result(
    family: &str,
    text: &str,
) -> Result<EngineResult, Box<dyn std::error::Error>> {
    if family == "harmony" {
        let mut parser = HarmonyToolStreamParser::new()?;
        let mut result = parser.parse_tool_call_streaming_text(text);
        let finish = parser.finish_tool_call_stream();
        result.normal_text.push_str(&finish.normal_text);
        result.tool_call_chunks.extend(finish.tool_call_chunks);
        return Ok(EngineResult {
            calls: assemble_tool_calls(&result.tool_call_chunks)
                .into_iter()
                .map(|(n, a)| {
                    let v = serde_json::from_str(&a).unwrap_or(Value::String(a));
                    (n, v)
                })
                .collect(),
            normal_text: result.normal_text,
        });
    }

    let mut parser = create_tool_parser_for_family(family, &[])?;
    let mut result = parser.push(text)?;
    result.append(parser.finish()?);
    Ok(EngineResult {
        normal_text: result.normal_text.clone(),
        calls: assemble_trait_calls(result),
    })
}

fn assemble_trait_calls(result: ToolParseResult) -> Vec<(String, Value)> {
    let mut names = BTreeMap::<usize, String>::new();
    let mut args = BTreeMap::<usize, String>::new();
    for ToolCallDelta {
        tool_index,
        name,
        arguments,
    } in result.calls
    {
        if let Some(name) = name {
            names.entry(tool_index).or_default().push_str(&name);
        }
        args.entry(tool_index).or_default().push_str(&arguments);
    }
    names
        .into_iter()
        .map(|(idx, name)| {
            let raw = args.remove(&idx).unwrap_or_default();
            let value = serde_json::from_str(&raw).unwrap_or(Value::String(raw));
            (name, value)
        })
        .collect()
}
