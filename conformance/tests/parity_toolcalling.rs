// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tool-calling parity: run every vendored `batch` fixture through
//! `dynamo-parsers` and assert the result matches `expected.dynamo_v1`.
//!
//! The fixture `family` field IS the parser name — passed straight to
//! `detect_and_parse_tool_call_with_recovery`, exactly like dynamo's own
//! `parse_tool_calls_batch` PyO3 binding (see tests/parity/toolcalling/dynamo.py).
//!
//! Scope: batch mode only. Streaming parity lives with the per-parser streaming
//! work (the jail is in lib/llm, not in this crate). Stream-mode fixtures and the
//! cross-family placeholder cases (no model_text / no expected) are skipped.

use std::collections::BTreeMap;

mod common;
use common::{collect_yaml, fixture_name};

use dynamo_parsers::tool_calling::ToolDefinition;
use dynamo_parsers::tool_calling::parsers::detect_and_parse_tool_call_with_recovery;
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
    tools: Option<Vec<RawTool>>,
    #[serde(default)]
    expected: Option<Expected>,
}
// ToolDefinition does not derive Deserialize, so deserialize into this and build it.
#[derive(Deserialize)]
struct RawTool {
    name: String,
    #[serde(default)]
    parameters: Option<Value>,
    #[serde(default)]
    strict: Option<bool>,
}
#[derive(Deserialize)]
struct Expected {
    dynamo_v1: EngineExpected,
}
#[derive(Deserialize)]
struct EngineExpected {
    #[serde(default)]
    calls: Vec<ExpCall>,
    #[serde(default)]
    normal_text: Option<String>,
}
#[derive(Deserialize)]
struct ExpCall {
    name: String,
    #[serde(default)]
    arguments: Value,
}

/// Mirror of dynamo's `decode_arguments`: a tool call's `arguments` is a JSON
/// string; parse it to a value, or keep it raw (malformed-body cases expect the
/// raw string back).
fn decode_args(s: &str) -> Value {
    serde_json::from_str::<Value>(s).unwrap_or_else(|_| Value::String(s.to_string()))
}

/// `normal_text` parity: None / empty / whitespace are equivalent (matches
/// common.py's normalization), so compare trimmed with None treated as "".
fn norm_text(s: Option<&str>) -> String {
    s.unwrap_or("").trim().to_string()
}

#[tokio::test(flavor = "multi_thread")]
async fn toolcalling_batch_parity() {
    // Versioned corpus (inputs/ + <impl>-<version>/): shared model_text/tools live in
    // inputs/, Dynamo v1's expected in the (single) dynamo_v1-<version>/ dir. Read inputs
    // and fold Dynamo's expected back in; the version dirs' peer-only files are not
    // top-level fixtures here.
    let batch_root = common::ensure_fixtures().join("toolcalling/fixtures-batch-v1");
    let inputs_root = batch_root.join("inputs");
    let dyn_dir = std::fs::read_dir(batch_root)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| {
            p.is_dir()
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("dynamo_v1-"))
        })
        .expect("no dynamo_v1-<version> dir under fixtures-batch-v1");
    let mut files = Vec::new();
    collect_yaml(&inputs_root, &mut files);
    files.sort();
    assert!(
        !files.is_empty(),
        "no fixtures found under {}",
        inputs_root.display()
    );

    let mut total = 0usize;
    let mut failures: Vec<String> = Vec::new();

    for path in &files {
        let yaml = std::fs::read_to_string(path).unwrap();
        let mut fx: Fixture = match serde_yaml::from_str(&yaml) {
            Ok(f) => f,
            Err(e) => {
                failures.push(format!("{}: YAML parse error: {e}", path.display()));
                continue;
            }
        };
        if fx.mode != "batch" {
            continue;
        }
        // Fold Dynamo v1's `expected.dynamo_v1` (from dynamo_v1-<version>/<family>/<name>)
        // into the shared inputs cases.
        let rel = path.strip_prefix(&inputs_root).unwrap();
        let dyn_fx = std::fs::read_to_string(dyn_dir.join(rel))
            .ok()
            .and_then(|t| serde_yaml::from_str::<Fixture>(&t).ok());
        if let Some(dfx) = dyn_fx {
            for (cid, dcase) in dfx.cases {
                if let (Some(c), Some(exp)) = (fx.cases.get_mut(&cid), dcase.expected) {
                    c.expected = Some(exp);
                }
            }
        }
        eprintln!("fixture {}", fixture_name(path));
        for (cid, case) in &fx.cases {
            let (Some(text), Some(expected)) = (case.model_text.as_ref(), case.expected.as_ref())
            else {
                continue; // placeholder / non-batch case
            };
            total += 1;

            let tools: Vec<ToolDefinition> = case
                .tools
                .as_ref()
                .map(|ts| {
                    ts.iter()
                        .map(|t| ToolDefinition {
                            name: t.name.clone(),
                            parameters: t.parameters.clone(),
                            strict: t.strict,
                        })
                        .collect()
                })
                .unwrap_or_default();
            let tools_opt = (!tools.is_empty()).then_some(tools.as_slice());

            let (got_calls, got_normal) =
                match detect_and_parse_tool_call_with_recovery(text, Some(&fx.family), tools_opt)
                    .await
                {
                    Ok(v) => v,
                    Err(e) => {
                        failures.push(format!("{} [{cid}]: parser error: {e}", fx.family));
                        continue;
                    }
                };

            let got: Vec<(String, Value)> = got_calls
                .iter()
                .map(|c| (c.function.name.clone(), decode_args(&c.function.arguments)))
                .collect();
            let want: Vec<(String, Value)> = expected
                .dynamo_v1
                .calls
                .iter()
                .map(|c| (c.name.clone(), c.arguments.clone()))
                .collect();

            let calls_ok = got == want;
            let normal_ok = norm_text(got_normal.as_deref())
                == norm_text(expected.dynamo_v1.normal_text.as_deref());

            if !calls_ok || !normal_ok {
                failures.push(format!(
                    "{family} [{cid}]\n        got  calls={got:?}\n        got  normal={got_n:?}\n        want calls={want:?}\n        want normal={want_n:?}",
                    family = fx.family,
                    got_n = norm_text(got_normal.as_deref()),
                    want_n = norm_text(expected.dynamo_v1.normal_text.as_deref()),
                ));
            }
        }
    }

    eprintln!(
        "toolcalling batch parity: {}/{} cases passed",
        total.saturating_sub(failures.len()),
        total
    );
    if !failures.is_empty() {
        for f in &failures {
            eprintln!("FAIL {f}");
        }
        panic!(
            "{} of {} batch cases diverged from expected.dynamo_v1",
            failures.len(),
            total
        );
    }
}
