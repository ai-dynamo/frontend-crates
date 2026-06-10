// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Streaming tool-calling parity for Dynamo parser v2.
//!
//! Fixtures live in `conformance/toolcalling/fixtures-stream-v2/` (the frontend-crates overlay).
//! Each chunk records, under `expected.<impl>`, the tool-call deltas that impl
//! emits at that chunk boundary. This test drives the DYNAMO parser (the only
//! impl with a Rust streaming parser) and asserts:
//!
//! 1. **Per-chunk emit, token path** — feeding `delta_token_ids` per chunk
//!    produces exactly `expected.dynamo` for that chunk.
//! 2. **Per-chunk emit, text path** — feeding `delta_text` per chunk produces
//!    the same per-chunk emit (exercises `parse_tool_call_streaming_text`).
//! 3. **Assembled** — concatenating the per-chunk deltas yields the expected
//!    final calls.
//!
//! Cases with `unavailable.dynamo` (e.g. character-split fixtures a token parser
//! can't consume per-chunk) are skipped for Dynamo. vLLM/SGLang per-chunk data in
//! the fixtures is captured from the engines in their containers, not re-run here.

use std::collections::BTreeMap;
use std::path::Path;

mod common;
use common::{collect_yaml, fixture_name};

use dynamo_parsers::tool_calling::ToolCallResponseChunk;
use dynamo_parsers_v2::{
    HarmonyToolStreamParser, Tool, ToolCallDelta, ToolParseResult, create_tool_parser_for_family,
};
use serde::Deserialize;
use serde_json::Value;

// ── Fixture schema ────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct Fixture {
    family: String,
    #[serde(default)]
    mode: Option<String>,
    #[serde(default)]
    cases: BTreeMap<String, Case>,
}

#[derive(Deserialize)]
struct Case {
    #[serde(default)]
    tools: Vec<Tool>,
    #[serde(default)]
    chunks: Vec<Chunk>,
    /// Impls that can't run this case at all (e.g. vllm harmony stub, or dynamo
    /// on character-split fixtures). Keyed by impl name → reason.
    #[serde(default)]
    unavailable: BTreeMap<String, String>,
}

#[derive(Deserialize)]
struct Chunk {
    #[serde(default)]
    delta_text: String,
    #[serde(default)]
    delta_token_ids: Vec<u32>,
    #[serde(default)]
    finish_reason: Option<String>,
    /// Per-impl tool-call deltas emitted at this chunk.
    #[serde(default)]
    expected: BTreeMap<String, Vec<FixtureDelta>>,
    #[serde(default)]
    normal_text: BTreeMap<String, String>,
}

/// One expected delta. `id: true` in YAML means an id was emitted; absent fields
/// (name/arguments) mean that field was not present on the delta.
#[derive(Deserialize, Debug)]
struct FixtureDelta {
    index: u32,
    #[serde(default)]
    id: bool,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Compare emitted deltas for one chunk against the fixture's expected list.
struct ChunkDiff<'a> {
    label: &'a str,
    cid: &'a str,
    chunk_idx: usize,
    emitted: &'a [EmittedDelta],
    expected: &'a [FixtureDelta],
    emitted_normal_text: &'a str,
    expected_normal_text: &'a str,
}

fn diff_chunk(input: ChunkDiff<'_>, failures: &mut Vec<String>) {
    if input.emitted_normal_text != input.expected_normal_text {
        failures.push(format!(
            "{} {} chunk[{}]: normal_text {:?} != {:?}",
            input.label,
            input.cid,
            input.chunk_idx,
            input.emitted_normal_text,
            input.expected_normal_text,
        ));
    }
    if input.emitted.len() != input.expected.len() {
        failures.push(format!(
            "{} {} chunk[{}]: emitted {} deltas, want {}",
            input.label,
            input.cid,
            input.chunk_idx,
            input.emitted.len(),
            input.expected.len()
        ));
        return;
    }
    for (i, (got, want)) in input.emitted.iter().zip(input.expected.iter()).enumerate() {
        let mut errs: Vec<String> = Vec::new();
        if got.index != want.index as usize {
            errs.push(format!("index {} != {}", got.index, want.index));
        }
        if got.id != want.id {
            errs.push(format!("has_id {} != {}", got.id, want.id));
        }
        if got.name.as_deref() != want.name.as_deref() {
            errs.push(format!("name {:?} != {:?}", got.name, want.name));
        }
        if got.arguments.as_deref() != want.arguments.as_deref() {
            errs.push(format!(
                "arguments {:?} != {:?}",
                got.arguments, want.arguments
            ));
        }
        if !errs.is_empty() {
            failures.push(format!(
                "{} {} chunk[{}] delta[{i}]: {}",
                input.label,
                input.cid,
                input.chunk_idx,
                errs.join("; ")
            ));
        }
    }
}

#[derive(Debug)]
struct EmittedDelta {
    index: usize,
    id: bool,
    name: Option<String>,
    arguments: Option<String>,
}

// The v2 stream overlay is fully canonical (`dynamo_rust`); the legacy `dynamo`
// fallback was dropped as part of the v2 key migration. The v1 batch corpus
// (read by `parity_toolcalling_batch_via_stream.rs`) stays legacy and is untouched.
fn dynamo_expected(expected: &BTreeMap<String, Vec<FixtureDelta>>) -> Option<&Vec<FixtureDelta>> {
    expected.get("dynamo_rust")
}

fn dynamo_normal_text(normal_text: &BTreeMap<String, String>) -> &str {
    normal_text
        .get("dynamo_rust")
        .map(String::as_str)
        .unwrap_or("")
}

fn dynamo_unavailable(unavailable: &BTreeMap<String, String>) -> bool {
    unavailable.contains_key("dynamo_rust")
}

/// Derive the assembled calls from the fixture's per-chunk expected dynamo deltas.
fn expected_assembled(case: &Case) -> EngineResult {
    let mut names: BTreeMap<usize, String> = BTreeMap::new();
    let mut args: BTreeMap<usize, String> = BTreeMap::new();
    let mut normal_text = String::new();
    for chunk in &case.chunks {
        normal_text.push_str(dynamo_normal_text(&chunk.normal_text));
        for d in dynamo_expected(&chunk.expected).into_iter().flatten() {
            if let Some(n) = &d.name {
                names.entry(d.index as usize).or_default().push_str(n);
            }
            if let Some(a) = &d.arguments {
                args.entry(d.index as usize).or_default().push_str(a);
            }
        }
    }
    let calls = names
        .into_iter()
        .map(|(idx, name)| {
            let raw = args.get(&idx).cloned().unwrap_or_default();
            let v = serde_json::from_str(&raw).unwrap_or(Value::String(raw));
            (name, v)
        })
        .collect();
    EngineResult { calls, normal_text }
}

#[derive(Debug, PartialEq, Eq)]
struct EngineResult {
    calls: Vec<(String, Value)>,
    normal_text: String,
}

fn assemble_emitted(chunks: &[EmittedDelta], normal_text: String) -> EngineResult {
    let mut names: BTreeMap<usize, String> = BTreeMap::new();
    let mut args: BTreeMap<usize, String> = BTreeMap::new();
    for chunk in chunks {
        if let Some(name) = &chunk.name {
            names.entry(chunk.index).or_default().push_str(name);
        }
        if let Some(arguments) = &chunk.arguments {
            args.entry(chunk.index).or_default().push_str(arguments);
        }
    }
    let calls = names
        .into_iter()
        .map(|(idx, name)| {
            let raw = args.remove(&idx).unwrap_or_default();
            let v = serde_json::from_str(&raw).unwrap_or(Value::String(raw));
            (name, v)
        })
        .collect();
    EngineResult { calls, normal_text }
}

fn emitted_from_chunk(chunk: ToolCallResponseChunk) -> EmittedDelta {
    EmittedDelta {
        index: chunk.index as usize,
        id: chunk.id.is_some(),
        name: chunk.function.as_ref().and_then(|f| f.name.clone()),
        arguments: chunk.function.as_ref().and_then(|f| f.arguments.clone()),
    }
}

fn emitted_from_result(result: ToolParseResult) -> Vec<EmittedDelta> {
    result
        .calls
        .into_iter()
        .map(
            |ToolCallDelta {
                 tool_index,
                 name,
                 arguments,
             }| EmittedDelta {
                index: tool_index,
                id: false,
                name,
                arguments: Some(arguments),
            },
        )
        .collect()
}

// ── Test ──────────────────────────────────────────────────────────────────────

#[test]
fn toolcalling_stream_parity() {
    let root = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/toolcalling/fixtures-stream-v2"
    );
    let mut files = Vec::new();
    collect_yaml(Path::new(root), &mut files);
    files.sort();
    assert!(!files.is_empty(), "no fixtures found under {root}");

    let mut total = 0usize;
    let mut skipped = 0usize;
    let mut failures: Vec<String> = Vec::new();

    for path in &files {
        let yaml = std::fs::read_to_string(path).unwrap();
        let fx: Fixture = match serde_yaml::from_str(&yaml) {
            Ok(f) => f,
            Err(e) => {
                failures.push(format!("{}: YAML parse error: {e}", path.display()));
                continue;
            }
        };
        if !(fx.family == "harmony" || fx.family == "harmony_text" || fx.family == "deepseek_v4")
            || !matches!(fx.mode.as_deref(), Some("stream" | "streamv2"))
        {
            continue;
        }
        eprintln!("fixture {}", fixture_name(path));

        // `harmony` drives the token-id path; `harmony_text` drives the text path.
        // Both must match their own per-chunk `expected.dynamo` data.
        let is_text = fx.family == "harmony_text";

        for (cid, case) in &fx.cases {
            if dynamo_unavailable(&case.unavailable) {
                skipped += 1;
                continue;
            }
            total += 1;

            let mut all: Vec<EmittedDelta> = Vec::new();
            let mut all_normal_text = String::new();
            let mut finished = false;

            if fx.family == "harmony" || fx.family == "harmony_text" {
                let mut parser = HarmonyToolStreamParser::new().unwrap();
                for (ci, chunk) in case.chunks.iter().enumerate() {
                    let res = if is_text {
                        parser.parse_tool_call_streaming_text(&chunk.delta_text)
                    } else {
                        parser.parse_tool_call_streaming_incremental(&chunk.delta_token_ids)
                    };
                    let mut normal_text = res.normal_text;
                    let mut emitted: Vec<EmittedDelta> = res
                        .tool_call_chunks
                        .into_iter()
                        .map(emitted_from_chunk)
                        .collect();
                    if chunk.finish_reason.is_some() {
                        let finish = parser.finish_tool_call_stream();
                        normal_text.push_str(&finish.normal_text);
                        emitted.extend(finish.tool_call_chunks.into_iter().map(emitted_from_chunk));
                        finished = true;
                    }
                    let want = dynamo_expected(&chunk.expected)
                        .map(|v| v.as_slice())
                        .unwrap_or(&[]);
                    let want_normal_text = dynamo_normal_text(&chunk.normal_text);
                    let label = if is_text { "text" } else { "token" };
                    diff_chunk(
                        ChunkDiff {
                            label,
                            cid,
                            chunk_idx: ci,
                            emitted: &emitted,
                            expected: want,
                            emitted_normal_text: &normal_text,
                            expected_normal_text: want_normal_text,
                        },
                        &mut failures,
                    );
                    all_normal_text.push_str(&normal_text);
                    all.extend(emitted);
                }
                if !finished {
                    let finish = parser.finish_tool_call_stream();
                    all_normal_text.push_str(&finish.normal_text);
                    all.extend(finish.tool_call_chunks.into_iter().map(emitted_from_chunk));
                }
            } else {
                let mut parser = create_tool_parser_for_family(&fx.family, &case.tools).unwrap();
                for (ci, chunk) in case.chunks.iter().enumerate() {
                    let mut result = parser.push(&chunk.delta_text).unwrap();
                    if chunk.finish_reason.is_some() {
                        result.append(parser.finish().unwrap());
                        finished = true;
                    }
                    let normal_text = result.normal_text.clone();
                    let emitted = emitted_from_result(result);
                    let want = dynamo_expected(&chunk.expected)
                        .map(|v| v.as_slice())
                        .unwrap_or(&[]);
                    let want_normal_text = dynamo_normal_text(&chunk.normal_text);
                    diff_chunk(
                        ChunkDiff {
                            label: &fx.family,
                            cid,
                            chunk_idx: ci,
                            emitted: &emitted,
                            expected: want,
                            emitted_normal_text: &normal_text,
                            expected_normal_text: want_normal_text,
                        },
                        &mut failures,
                    );
                    all_normal_text.push_str(&normal_text);
                    all.extend(emitted);
                }
                if !finished {
                    let finish = parser.finish().unwrap();
                    all_normal_text.push_str(&finish.normal_text);
                    all.extend(emitted_from_result(finish));
                }
            }

            // Both paths must assemble to the same expected calls.
            let got = assemble_emitted(&all, all_normal_text);
            let want = expected_assembled(case);
            if got != want {
                let label = if is_text { "text" } else { "token" };
                failures.push(format!(
                    "{cid} [{label}] assembled:\n        got  {got:?}\n        want {want:?}"
                ));
            }
        }
    }

    eprintln!(
        "Dynamo streaming parity: {}/{} cases passed ({skipped} local-parser-unavailable)",
        total.saturating_sub(failures.len()),
        total,
    );
    assert!(total > 0, "no Dynamo streamv2 cases were exercised");
    if !failures.is_empty() {
        for f in &failures {
            eprintln!("FAIL {f}");
        }
        panic!("{} of {} cases diverged", failures.len(), total);
    }
}
