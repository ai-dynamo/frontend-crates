// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Streaming tool-calling parity for Dynamo parser v2.
//!
//! Fixtures live in `conformance/toolcalling/fixtures-stream-v1/` (the frontend-crates overlay).
//! Each chunk records, under `expected.<impl>`, the tool-call deltas that impl
//! emits at that chunk boundary. This test drives the DYNAMO parser (the only
//! impl with a Rust streaming parser) and asserts:
//!
//! 1. **Per-chunk emit, token path** — feeding `delta_token_ids` per chunk
//!    produces exactly `expected.dynamo_v1` for that chunk.
//! 2. **Per-chunk emit, text path** — feeding `delta_text` per chunk produces
//!    the same per-chunk emit (exercises `parse_tool_call_streaming_text`).
//! 3. **Assembled** — concatenating the per-chunk deltas yields the expected
//!    final calls.
//!
//! Cases with `unavailable.dynamo_v2` (e.g. character-split fixtures a token parser
//! can't consume per-chunk) are skipped for Dynamo. vLLM/SGLang per-chunk data in
//! the fixtures is captured from the engines in their containers, not re-run here.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

mod common;
#[path = "common/migration_stream_snapshot.rs"]
mod migration_stream_snapshot;
use common::{collect_yaml, fixture_name};

use dynamo_parsers_v2::{
    HarmonyToolStreamParser, Tool, ToolCallDelta, ToolCallResponseChunk, ToolParseResult,
    create_tool_parser_for_family,
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
    complete: Option<bool>,
}

// The corpus is versioned (inputs/ + <impl>-<version>/): the shared per-chunk
// delta_text lives in inputs/, Dynamo's per-chunk expected in dynamo_v2-<ver>/.
// These structs load the Dynamo version dir so we can fold `expected.dynamo_v2`
// (and `unavailable.dynamo_v2`) back into the inputs Fixture before parsing.
#[derive(Deserialize)]
struct DynFixture {
    #[serde(default)]
    cases: BTreeMap<String, DynCase>,
}

#[derive(Deserialize)]
struct DynCase {
    #[serde(default)]
    unavailable: Option<String>,
    #[serde(default)]
    chunks: Vec<DynChunk>,
}

#[derive(Deserialize)]
struct DynChunk {
    #[serde(default)]
    expected: Vec<FixtureDelta>,
    #[serde(default)]
    normal_text: Option<String>,
}

/// Fold Dynamo's expected (from dynamo_v2-<ver>/<family>/<name>) into an inputs
/// Fixture, keyed under "dynamo_v2" per chunk + case, so the rest of the test —
/// written for the old bundled layout — is unchanged.
fn merge_dynamo(fx: &mut Fixture, dyn_dir: &Path, rel: &Path) -> BTreeSet<String> {
    let complete_snapshot = dyn_dir
        .components()
        .any(|part| part.as_os_str() == ".reader-views");
    if complete_snapshot {
        for case in fx.cases.values_mut() {
            case.unavailable.remove("dynamo_v2");
            for chunk in &mut case.chunks {
                chunk.expected.remove("dynamo_v2");
                chunk.normal_text.remove("dynamo_v2");
            }
        }
    }
    let dfp = dyn_dir.join(rel);
    // A missing overlay is benign; any other I/O error must surface, not vanish.
    let text = match std::fs::read_to_string(&dfp) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return BTreeSet::new(),
        Err(e) => panic!("{}: dynamo overlay read error: {e}", dfp.display()),
    };
    let dyn_fx: DynFixture = serde_yaml::from_str(&text)
        .unwrap_or_else(|e| panic!("{}: dynamo overlay parse error: {e}", dfp.display()));
    let observed = dyn_fx.cases.keys().cloned().collect();
    for (cid, dcase) in dyn_fx.cases {
        let Some(case) = fx.cases.get_mut(&cid) else {
            continue;
        };
        if let Some(reason) = &dcase.unavailable {
            case.unavailable
                .insert("dynamo_v2".to_string(), reason.clone());
            if !complete_snapshot {
                continue;
            }
        } else {
            case.unavailable.remove("dynamo_v2");
        }
        // Complete snapshots retain the prior chunks even when a later legacy
        // overlay marked the case unavailable without supplying new events.
        for (i, dchunk) in dcase.chunks.into_iter().enumerate() {
            if let Some(chunk) = case.chunks.get_mut(i) {
                chunk
                    .expected
                    .insert("dynamo_v2".to_string(), dchunk.expected);
                // Latest-wins applies to `normal_text` too: an ABSENT value in the
                // newer capture means "this chunk emits no visible text", which must
                // clear whatever an older capture recorded. Only inserting on Some()
                // made the field write-only — a chunk that legitimately stopped
                // emitting text kept the stale older string, so the test compared the
                // newest deltas against a previous capture's normal_text.
                match dchunk.normal_text {
                    Some(nt) => {
                        chunk.normal_text.insert("dynamo_v2".to_string(), nt);
                    }
                    None => {
                        chunk.normal_text.remove("dynamo_v2");
                    }
                }
            }
        }
    }
    observed
}

fn stream_dynamo_dirs(sv1: &Path) -> Vec<std::path::PathBuf> {
    common::historical_capture_dirs(sv1, "stream", "dynamo_v2")
}

#[test]
fn empty_complete_snapshot_clears_input_observations() {
    let root = std::env::temp_dir().join(format!("dynamo-empty-snapshot-{}", std::process::id()));
    let capture = root.join(".reader-views/rust/dynamo_v2-1.0.0");
    std::fs::create_dir_all(capture.join("qwen3")).unwrap();
    let mut fixture: Fixture = serde_yaml::from_str("family: qwen3\ncases:\n  A:\n    unavailable: {dynamo_v2: old}\n    chunks:\n      - delta_text: x\n        expected: {dynamo_v2: [{index: 0, name: old}]}\n        normal_text: {dynamo_v2: old}\n").unwrap();
    merge_dynamo(&mut fixture, &capture, Path::new("qwen3/probe.yaml"));
    assert!(fixture.cases["A"].unavailable.is_empty());
    assert!(fixture.cases["A"].chunks[0].expected.is_empty());
    assert!(fixture.cases["A"].chunks[0].normal_text.is_empty());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn recorded_empty_stream_is_distinct_from_absent_measurement() {
    let root = std::env::temp_dir().join(format!("dynamo-observed-empty-{}", std::process::id()));
    let capture = root.join(".reader-views/rust/dynamo_v2-1.0.0");
    std::fs::create_dir_all(capture.join("qwen3")).unwrap();
    std::fs::write(
        capture.join("qwen3/probe.yaml"),
        "cases:\n  measured:\n    chunks: []\n",
    )
    .unwrap();
    let mut fixture: Fixture = serde_yaml::from_str(
        "family: qwen3\ncases:\n  measured: {chunks: []}\n  absent: {chunks: []}\n",
    )
    .unwrap();
    let observed = merge_dynamo(&mut fixture, &capture, Path::new("qwen3/probe.yaml"));
    assert_eq!(observed, BTreeSet::from(["measured".to_string()]));
    assert!(fixture.cases["measured"].chunks.is_empty());
    assert!(fixture.cases["absent"].chunks.is_empty());
    assert!(merge_dynamo(&mut fixture, &capture, Path::new("new_family/probe.yaml")).is_empty());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn complete_unavailable_snapshot_retains_its_resolved_chunks() {
    let root = std::env::temp_dir().join(format!(
        "dynamo-unavailable-snapshot-{}",
        std::process::id()
    ));
    let capture = root.join(".reader-views/rust/dynamo_v2-1.0.0");
    std::fs::create_dir_all(capture.join("qwen3")).unwrap();
    std::fs::write(capture.join("qwen3/probe.yaml"),
        "cases:\n  A:\n    unavailable: skipped\n    chunks:\n      - expected:\n          - index: 0\n            name: retained\n").unwrap();
    let mut fixture: Fixture =
        serde_yaml::from_str("family: qwen3\ncases:\n  A:\n    chunks:\n      - delta_text: x\n")
            .unwrap();
    merge_dynamo(&mut fixture, &capture, Path::new("qwen3/probe.yaml"));
    assert_eq!(fixture.cases["A"].unavailable["dynamo_v2"], "skipped");
    assert_eq!(
        fixture.cases["A"].chunks[0].expected["dynamo_v2"][0]
            .name
            .as_deref(),
        Some("retained")
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn stream_dynamo_dirs_follow_policy_and_exclude_qualified_history() {
    let root = std::env::temp_dir().join(format!(
        "dynamo-stream-dirs-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(root.join("dynamo_v2-0.3.1")).unwrap();
    std::fs::create_dir_all(root.join("dynamo_v2-0.3.4+historical")).unwrap();
    std::fs::create_dir_all(root.join("dynamo_v2-0.6.1")).unwrap();

    let names: Vec<_> = stream_dynamo_dirs(&root)
        .into_iter()
        .map(|path| path.file_name().unwrap().to_owned())
        .collect();
    assert_eq!(
        names,
        ["dynamo_v2-0.3.1", "dynamo_v2-0.6.1"].map(std::ffi::OsString::from)
    );

    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn prerelease_compaction_preserves_latest_stream_capture() {
    fn expected_name(fixture: &Fixture) -> &str {
        fixture.cases["A"].chunks[0].expected["dynamo_v2"][0]
            .name
            .as_deref()
            .unwrap()
    }

    let root = std::env::temp_dir().join(format!(
        "dynamo-stream-prerelease-fold-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let relative = Path::new("qwen3/one.yaml");
    let captures = [
        ("0.1.0-rc1", "prerelease"),
        ("0.1.0", "release"),
        ("0.6.1", "release"),
    ];
    for (version, name) in captures {
        let capture = root.join(format!("dynamo_v2-{version}/qwen3"));
        std::fs::create_dir_all(&capture).unwrap();
        std::fs::write(
            capture.join("one.yaml"),
            format!(
                "cases:\n  A:\n    chunks:\n      - expected:\n          - index: 0\n            name: {name}\n            arguments: \"{{}}\"\n"
            ),
        )
        .unwrap();
    }

    let base_fixture = || {
        serde_yaml::from_str::<Fixture>(
            "family: qwen3\ncases:\n  A:\n    chunks:\n      - delta_text: x\n",
        )
        .unwrap()
    };
    let dirs = common::version_dirs_ascending(&root, "dynamo_v2-");
    let mut uncompacted = base_fixture();
    for directory in &dirs {
        merge_dynamo(&mut uncompacted, directory, relative);
    }

    let mut compacted = base_fixture();
    for directory in dirs.iter().filter(|directory| {
        directory.file_name().and_then(|name| name.to_str()) != Some("dynamo_v2-0.6.1")
    }) {
        merge_dynamo(&mut compacted, directory, relative);
    }

    assert_eq!(expected_name(&uncompacted), "release");
    assert_eq!(expected_name(&compacted), "release");
    std::fs::remove_dir_all(root).unwrap();
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
        if let Some(complete) = want.complete
            && got.complete != complete
        {
            errs.push(format!("complete {} != {}", got.complete, complete));
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
    complete: bool,
}

// The v2 stream overlay is fully canonical (`dynamo_v2`); the legacy `dynamo`
// fallback was dropped as part of the v2 key migration. The v1 batch corpus
// (read by `conformance_toolcalling_batch_via_stream.rs`) stays legacy and is untouched.
fn dynamo_expected(expected: &BTreeMap<String, Vec<FixtureDelta>>) -> Option<&Vec<FixtureDelta>> {
    expected.get("dynamo_v2")
}

fn dynamo_normal_text(normal_text: &BTreeMap<String, String>) -> &str {
    normal_text
        .get("dynamo_v2")
        .map(String::as_str)
        .unwrap_or("")
}

fn dynamo_unavailable(unavailable: &BTreeMap<String, String>) -> bool {
    unavailable.contains_key("dynamo_v2")
}

/// Derive the assembled calls from the fixture's per-chunk expected dynamo deltas.
fn expected_assembled(case: &Case) -> EngineResult {
    let mut names: BTreeMap<usize, String> = BTreeMap::new();
    let mut args: BTreeMap<usize, String> = BTreeMap::new();
    let mut complete: BTreeMap<usize, bool> = BTreeMap::new();
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
            // Older immutable captures predate explicit completion metadata. Their
            // emitted deltas were all assembled calls, so retain that behavior.
            *complete.entry(d.index as usize).or_default() |= d.complete.unwrap_or(true);
        }
    }
    let calls = names
        .into_iter()
        .filter_map(|(idx, name)| {
            if complete.get(&idx) != Some(&true) {
                return None;
            }
            let raw = args.get(&idx).cloned().unwrap_or_default();
            let v = serde_json::from_str(&raw).unwrap_or(Value::String(raw));
            Some((name, v))
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
    let mut complete: BTreeMap<usize, bool> = BTreeMap::new();
    for chunk in chunks {
        if let Some(name) = &chunk.name {
            names.entry(chunk.index).or_default().push_str(name);
        }
        if let Some(arguments) = &chunk.arguments {
            args.entry(chunk.index).or_default().push_str(arguments);
        }
        *complete.entry(chunk.index).or_default() |= chunk.complete;
    }
    let calls = names
        .into_iter()
        .filter_map(|(idx, name)| {
            if complete.get(&idx) != Some(&true) {
                return None;
            }
            let raw = args.remove(&idx).unwrap_or_default();
            let v = serde_json::from_str(&raw).unwrap_or(Value::String(raw));
            Some((name, v))
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
        complete: true,
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
                 complete,
                 ..
             }| EmittedDelta {
                index: tool_index,
                id: false,
                name,
                arguments: Some(arguments),
                complete,
            },
        )
        .collect()
}

// ── Test ──────────────────────────────────────────────────────────────────────

#[test]
fn toolcalling_stream_parity() {
    // Versioned corpus: shared chunks in inputs/, Dynamo's expected in the
    // dynamo_v2-<version>/ dirs. This test drives the Dynamo parser *v2*; with the
    // impl-key split every dynamo_v2-* dir belongs to it (the v1 jail reference has
    // its own dynamo_v1-* namespace, tested elsewhere). Old version dirs are capture
    // history; each family selects its latest complete Rust-view checkpoint.
    let sv1 = common::rust_stream_fixture_root(&common::ensure_fixtures());
    let inputs_root = sv1.join("inputs");
    let dyn_dirs = stream_dynamo_dirs(&sv1);
    if dyn_dirs.is_empty() {
        eprintln!("historical parity skipped: no eligible captured measurements");
        return;
    }
    let mut files = Vec::new();
    collect_yaml(&inputs_root, &mut files);
    files.sort();
    assert!(
        !files.is_empty(),
        "no fixtures found under {}",
        inputs_root.display()
    );

    let mut total = 0usize;
    let mut skipped = 0usize;
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
        let rel = path.strip_prefix(&inputs_root).unwrap();
        let observed = common::latest_family_capture(&dyn_dirs, rel)
            .map(|dyn_dir| merge_dynamo(&mut fx, dyn_dir, rel))
            .unwrap_or_default();
        if !matches!(fx.mode.as_deref(), Some("stream" | "streamv1")) {
            continue;
        }
        // Data-driven coverage (reuse the family registry, no hardcoded list):
        // harmony/harmony_text run the token-native path below; every other
        // family is exercised iff `create_tool_parser_for_family` can build a v2
        // parser for it. Registering a new family there auto-adds it here.
        let is_harmony = fx.family == "harmony" || fx.family == "harmony_text";
        if !is_harmony && create_tool_parser_for_family(&fx.family, &[]).is_err() {
            continue;
        }
        eprintln!("fixture {}", fixture_name(path));

        // `harmony` drives the token-id path; `harmony_text` drives the text path.
        // Both must match their own per-chunk `expected.dynamo_v1` data.
        let is_text = fx.family == "harmony_text";

        for (cid, case) in &fx.cases {
            if !observed.contains(cid) || dynamo_unavailable(&case.unavailable) {
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
    assert!(total > 0, "no Dynamo streamv1 cases were exercised");
    if !failures.is_empty() {
        for f in &failures {
            eprintln!("FAIL {f}");
        }
        panic!("{} of {} cases diverged", failures.len(), total);
    }
}
