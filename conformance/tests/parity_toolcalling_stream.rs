// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Streaming tool-calling parity for the Dynamo harmony parser.
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
use std::path::{Path, PathBuf};

use dynamo_parsers::tool_calling::ToolCallResponseChunk;
use dynamo_parsers_v2::{HarmonyToolStreamParser, assemble_tool_calls};
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

/// Compare emitted deltas for one chunk against the fixture's expected list.
fn diff_chunk(
    label: &str,
    cid: &str,
    chunk_idx: usize,
    emitted: &[ToolCallResponseChunk],
    expected: &[FixtureDelta],
    failures: &mut Vec<String>,
) {
    if emitted.len() != expected.len() {
        failures.push(format!(
            "{label} {cid} chunk[{chunk_idx}]: emitted {} deltas, want {}",
            emitted.len(),
            expected.len()
        ));
        return;
    }
    for (i, (got, want)) in emitted.iter().zip(expected.iter()).enumerate() {
        let mut errs: Vec<String> = Vec::new();
        if got.index != want.index {
            errs.push(format!("index {} != {}", got.index, want.index));
        }
        if got.id.is_some() != want.id {
            errs.push(format!("has_id {} != {}", got.id.is_some(), want.id));
        }
        let gname = got.function.as_ref().and_then(|f| f.name.as_deref());
        if gname != want.name.as_deref() {
            errs.push(format!("name {gname:?} != {:?}", want.name));
        }
        let gargs = got.function.as_ref().and_then(|f| f.arguments.as_deref());
        if gargs != want.arguments.as_deref() {
            errs.push(format!("arguments {gargs:?} != {:?}", want.arguments));
        }
        if !errs.is_empty() {
            failures.push(format!(
                "{label} {cid} chunk[{chunk_idx}] delta[{i}]: {}",
                errs.join("; ")
            ));
        }
    }
}

/// Derive the assembled calls from the fixture's per-chunk expected dynamo deltas.
fn expected_assembled(case: &Case) -> Vec<(String, Value)> {
    let mut names: BTreeMap<u32, String> = BTreeMap::new();
    let mut args: BTreeMap<u32, String> = BTreeMap::new();
    for chunk in &case.chunks {
        for d in chunk.expected.get("dynamo").into_iter().flatten() {
            if let Some(n) = &d.name {
                names.entry(d.index).or_default().push_str(n);
            }
            if let Some(a) = &d.arguments {
                args.entry(d.index).or_default().push_str(a);
            }
        }
    }
    names
        .into_iter()
        .map(|(idx, name)| {
            let raw = args.get(&idx).cloned().unwrap_or_default();
            let v = serde_json::from_str(&raw).unwrap_or(Value::String(raw));
            (name, v)
        })
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
        // Both harmony families have a frontend-crate v2 streaming parser: `harmony`
        // (token-id path) and `harmony_text` (text path). All other families are
        // local-parser-unavailable (TODO) and carry no expected.dynamo to assert.
        if !(fx.family == "harmony" || fx.family == "harmony_text")
            || fx.mode.as_deref() != Some("stream")
        {
            continue;
        }
        eprintln!("fixture {}", fixture_name(path));

        // `harmony` drives the token-id path; `harmony_text` drives the text path.
        // Both must match their own per-chunk `expected.dynamo` data.
        let is_text = fx.family == "harmony_text";

        for (cid, case) in &fx.cases {
            if case.unavailable.contains_key("dynamo") {
                skipped += 1;
                continue;
            }
            total += 1;

            let mut parser = HarmonyToolStreamParser::new().unwrap();
            let mut all: Vec<ToolCallResponseChunk> = Vec::new();
            let mut finished = false;

            for (ci, chunk) in case.chunks.iter().enumerate() {
                let res = if is_text {
                    parser.parse_tool_call_streaming_text(&chunk.delta_text)
                } else {
                    parser.parse_tool_call_streaming_incremental(&chunk.delta_token_ids)
                };
                let mut emitted = res.tool_call_chunks;
                if chunk.finish_reason.is_some() {
                    emitted.extend(parser.finish_tool_call_stream().tool_call_chunks);
                    finished = true;
                }
                let want = chunk
                    .expected
                    .get("dynamo")
                    .map(|v| v.as_slice())
                    .unwrap_or(&[]);
                let label = if is_text { "text" } else { "token" };
                diff_chunk(label, cid, ci, &emitted, want, &mut failures);
                all.extend(emitted);
            }
            if !finished {
                all.extend(parser.finish_tool_call_stream().tool_call_chunks);
            }

            // Both paths must assemble to the same expected calls.
            let got: Vec<(String, Value)> = assemble_tool_calls(&all)
                .into_iter()
                .map(|(n, a)| {
                    let v = serde_json::from_str(&a).unwrap_or(Value::String(a));
                    (n, v)
                })
                .collect();
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
        "harmony streaming parity: {}/{} cases passed ({skipped} local-parser-unavailable)",
        total.saturating_sub(failures.len()),
        total,
    );
    if !failures.is_empty() {
        for f in &failures {
            eprintln!("FAIL {f}");
        }
        panic!("{} of {} cases diverged", failures.len(), total);
    }
}
