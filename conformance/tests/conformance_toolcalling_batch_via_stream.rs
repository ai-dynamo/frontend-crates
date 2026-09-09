// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stream parser on BATCH samples: feed each batch fixture's full
//! `model_text` to the streaming parser and assert the assembled tool calls match
//! the BATCH parser's `expected.dynamo_v1`. This is the streaming-vs-batch
//! consistency check — the stream parser, given the complete output, must land on
//! the same calls as the batch parser.

use std::collections::BTreeMap;

mod common;
use common::{collect_yaml, fixture_name};

use dynamo_parsers_v2::{
    HarmonyToolStreamParser, Tool, ToolCallDelta, ToolParseResult, assemble_tool_calls,
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
    // The schema-dependent parsers (glm47, kimi_k2, qwen3_coder, minimax_m2, …)
    // need the tool schema to coerce argument types the way the v1 batch parser
    // did; the batch fixture carries it per case.
    #[serde(default)]
    tools: Vec<Tool>,
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
    // Versioned corpus (inputs/ + <impl>-<version>/): read the shared inputs and fold
    // Dynamo v1's `expected.dynamo_v1` from the dynamo_v1-<version>/ dirs back in,
    // ASCENDING — old version dirs are capture history, the latest wins per case.
    let batch_root = common::ensure_fixtures().join("toolcalling/fixtures-batch-v1");
    let inputs_root = batch_root.join("inputs");
    let dyn_dirs = common::version_dirs_ascending(&batch_root, "dynamo_v1-");
    assert!(
        !dyn_dirs.is_empty(),
        "no dynamo_v1-<version> dir under fixtures-batch-v1"
    );
    let mut files = Vec::new();
    collect_yaml(&inputs_root, &mut files);
    files.sort();

    // Batch samples where the v2 STREAMING parser deliberately differs from the
    // v1 BATCH parser. This compares BOTH calls and normal_text; the HTML
    // batch-on-stream tab compares calls only, so the `normal_text`-only entries
    // below still render green there. Removing an entry asserts stream and batch
    // now agree.
    //
    // v1 and v2 are INDEPENDENT parsers with NO shared code (v2 owns its
    // extraction in `parsers/v2/.../v1core`; v1 is unchanged from its release and
    // slated for deletion). They differ BY DESIGN on how much text around a tool
    // call survives:
    //   * v2 (streaming) preserves the model's text AROUND tool calls VERBATIM —
    //     the prose BEFORE the first call, BETWEEN consecutive calls, and AFTER the
    //     last one, plus bare whitespace-only and un-framed bare-prose answers.
    //   * v1 (batch), unchanged, drops most of that surrounding/inter-call text and
    //     trims boundary whitespace.
    // So the divergences below are almost all `normal_text`-only, and are the
    // expected v1-vs-v2 difference — NOT a regression. The dominant families of
    // entry:
    //   *:2.b/2.c/2.d (multi-call): v2 keeps the inter-call / trailing prose
    //        ("Both:  Done."), v1 keeps only the leading fragment ("Both:").
    //   *:8.b/8.c/8.d (call then trailing prose): v2 keeps "... Let me know if you
    //        need more.", v1 returns "".
    //   *:5.f/5.g (bare call recovery): v2 recovers the bare invoke and keeps the
    //        separator/prefix space; v1's strict batch path drops or trims it.
    //   *:9.b (whitespace-only input): v2 passes the bare whitespace through; v1
    //        returns "".
    //   harmony 3 (un-framed whole answer, whole-answer-drop class): v2 passes the answer
    //        through; v1 returns "". Text loss is the worse failure.
    //   The streaming peers (vLLM/SGLang) stream surrounding text the same way v2
    //   does, and the HTML batch-on-stream tab compares calls only, so all of these
    //   render green there.
    // The allowlist lives in conformance/toolcalling/known-divergences.yaml (the
    // same file the HTML renderer reads): every `stream_vs_batch:` entry is one
    // allowed `family:case` divergence, and its note is what the batch-on-stream
    // tab shows on the cell. An entry with an empty note fails here, so a new
    // divergence can only be allowed WITH its documentation.
    let kd_path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("toolcalling/known-divergences.yaml");
    let kd: BTreeMap<String, BTreeMap<String, BTreeMap<String, String>>> =
        serde_yaml::from_str(&std::fs::read_to_string(&kd_path).unwrap()).unwrap();
    let known_divergences: std::collections::BTreeSet<String> = kd
        .iter()
        .flat_map(|(fam, cases)| {
            cases.iter().filter_map(move |(cid, keys)| {
                let note = keys.get("stream_vs_batch")?;
                assert!(
                    !note.trim().is_empty(),
                    "{fam}:{cid}: empty stream_vs_batch note in known-divergences.yaml"
                );
                Some(format!("{fam}:{cid}"))
            })
        })
        .collect();
    assert!(
        !known_divergences.is_empty(),
        "no stream_vs_batch entries parsed from {}",
        kd_path.display()
    );

    let mut total = 0usize;
    let mut consistent = 0usize;
    let mut diverged = 0usize;
    let mut failures: Vec<String> = Vec::new();
    let mut unexpected_match: Vec<String> = Vec::new();

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
        // Data-driven coverage (reuse the family registry, no hardcoded list):
        // harmony runs the token/text Harmony path; every other family is
        // exercised iff `create_tool_parser_for_family` can build a v2 parser for
        // it. Registering a new family there auto-adds it to this stream-on-batch
        // consistency check.
        if fx.family != "harmony" && create_tool_parser_for_family(&fx.family, &[]).is_err() {
            continue;
        }
        let rel = path.strip_prefix(&inputs_root).unwrap();
        for dyn_dir in &dyn_dirs {
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
        }
        eprintln!("fixture {}", fixture_name(path));

        for (cid, case) in &fx.cases {
            let (Some(text), Some(expected)) = (case.model_text.as_ref(), case.expected.as_ref())
            else {
                continue; // placeholder case
            };
            total += 1;

            let got = parse_stream_result(&fx.family, text, &case.tools).unwrap();
            let want = EngineResult {
                calls: expected
                    .dynamo_v1
                    .calls
                    .iter()
                    .map(|c| (c.name.clone(), c.arguments.clone()))
                    .collect(),
                normal_text: expected.dynamo_v1.normal_text.clone(),
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
    tools: &[Tool],
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

    let mut parser = create_tool_parser_for_family(family, tools)?;
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
    let mut complete = BTreeMap::<usize, bool>::new();
    for ToolCallDelta {
        tool_index,
        name,
        arguments,
        complete: call_complete,
    } in result.calls
    {
        *complete.entry(tool_index).or_default() |= call_complete;
        if let Some(name) = name {
            names.entry(tool_index).or_insert(name);
        }
        args.entry(tool_index).or_default().push_str(&arguments);
    }
    names
        .into_iter()
        .filter_map(|(idx, name)| {
            let raw = args.remove(&idx).unwrap_or_default();
            if complete.get(&idx) != Some(&true) {
                return None;
            }
            let value = serde_json::from_str(&raw).unwrap_or(Value::String(raw));
            Some((name, value))
        })
        .collect()
}
