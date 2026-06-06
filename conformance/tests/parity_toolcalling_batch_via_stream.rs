// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stream parser on BATCH samples (harmony): feed each batch fixture's full
//! `model_text` to the streaming parser and assert the assembled tool calls match
//! the BATCH parser's `expected.dynamo`. This is the streaming-vs-batch
//! consistency check — the stream parser, given the complete output, must land on
//! the same calls as the batch parser.
//!
//! Harmony only (the first family with Dynamo parser v2). Reads the
//! dynamo-synced batch corpus directly (no overlay needed — model_text is input,
//! expected.dynamo is the batch reference).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use dynamo_parsers_v2::{HarmonyToolStreamParser, assemble_tool_calls};
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
}

#[derive(Deserialize)]
struct ExpCall {
    name: String,
    #[serde(default)]
    arguments: Value,
}

fn collect_yaml(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in rd.flatten() {
        let p = entry.path();
        if p.is_dir() {
            collect_yaml(&p, out);
        } else if p.extension().is_some_and(|x| x == "yaml") {
            out.push(p);
        }
    }
}

fn fixture_name(path: &Path) -> String {
    path.strip_prefix(env!("CARGO_MANIFEST_DIR"))
        .unwrap_or(path)
        .display()
        .to_string()
}

#[test]
fn toolcalling_batch_via_stream_parity() {
    // Harmony batch fixtures live in the dynamo-synced corpus.
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/toolcalling/fixtures/harmony");
    let mut files = Vec::new();
    collect_yaml(Path::new(root), &mut files);
    files.sort();

    // Batch samples where the streaming parser legitimately differs from the
    // batch parser. The batch parser (detect_and_parse_tool_call_with_recovery)
    // repairs truncated/malformed output via EOF recovery; the streaming parser does
    // not. batch.5.* and 8.* are the truncation/malformed/recovery cases:
    //   - batch parser recovers a call where the stream emits nothing, OR
    //   - the stream emits optimistically where the batch parser rejects.
    // Listed here so the test stays green while documenting the gap. Removing an
    // entry asserts that stream and batch now agree on that sample.
    let known_divergences: std::collections::BTreeSet<&str> = [
        "TOOLCALLING.batch.2.c",
        "TOOLCALLING.batch.5.a",
        "TOOLCALLING.batch.5.c",
        "TOOLCALLING.batch.5.d",
        "TOOLCALLING.batch.5.e",
        "TOOLCALLING.batch.8.a",
        "TOOLCALLING.batch.8.c",
        "TOOLCALLING.batch.8.d",
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
        if fx.family != "harmony" || fx.mode != "batch" {
            continue;
        }
        eprintln!("fixture {}", fixture_name(path));

        for (cid, case) in &fx.cases {
            let (Some(text), Some(expected)) = (case.model_text.as_ref(), case.expected.as_ref())
            else {
                continue; // placeholder case
            };
            total += 1;

            // Feed the full batch text to the stream parser (text path), finish,
            // assemble.
            let mut parser = HarmonyToolStreamParser::new().unwrap();
            let mut all = parser.parse_tool_call_streaming_text(text).tool_call_chunks;
            all.extend(parser.finish_tool_call_stream().tool_call_chunks);

            let got: Vec<(String, Value)> = assemble_tool_calls(&all)
                .into_iter()
                .map(|(n, a)| {
                    let v = serde_json::from_str(&a).unwrap_or(Value::String(a));
                    (n, v)
                })
                .collect();
            let want: Vec<(String, Value)> = expected
                .dynamo
                .calls
                .iter()
                .map(|c| (c.name.clone(), c.arguments.clone()))
                .collect();

            let known = known_divergences.contains(cid.as_str());
            if got == want {
                consistent += 1;
                if known {
                    // It now agrees — the allowlist entry is stale.
                    unexpected_match.push(cid.clone());
                }
            } else {
                diverged += 1;
                if !known {
                    failures.push(format!(
                        "{cid}:\n        stream got {got:?}\n        batch want {want:?}"
                    ));
                }
            }
        }
    }

    eprintln!(
        "harmony stream-on-batch: {consistent}/{total} consistent, {diverged} diverged \
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
