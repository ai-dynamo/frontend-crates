// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! U0a — schema round-trip for the unified (reasoning + content + tool calls)
//! conformance surface.
//!
//! Proves the event schema in `conformance/utils/lib/parsers/UNIFIED_CASES.md`
//! is machine-consumable and that every authored golden file
//! (`conformance/unified/golden_spec/<family>.yaml`, rendered on demand by
//! gen_unified_golden.py into the gitignored build tree) parses and round-trips
//! through it. This is the U0 spike gate — it does NOT run any parser;
//! capture/parity land in later phases (see `DOIT.p1.e4.unifiedparsers-capture.md`).

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

mod common;

/// One golden file = the authored oracle cases for a single grammar family.
#[derive(Deserialize, Serialize, PartialEq, Debug)]
struct GoldenFile {
    version: u32,
    family: String,
    cases: BTreeMap<String, GoldenCase>,
}

#[derive(Deserialize, Serialize, PartialEq, Debug)]
#[serde(deny_unknown_fields)]
struct GoldenCase {
    description: String,
    /// Policy decisions (P1/P2/...) this case's correctness depends on.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    policy: Vec<String>,
    #[serde(default)]
    init: common::Init,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    finish_reason: Option<String>,
    /// Raw streamed model text.
    input: String,
    /// Spec-derived correct event list — the oracle.
    golden: Vec<UnifiedEvent>,
    /// Provisional documentation of expected per-engine verdicts (not asserted in U0).
    expect: BTreeMap<String, ExpectEntry>,
}

/// One ordered unified event. `kind` tags the variant.
#[derive(Deserialize, Serialize, PartialEq, Debug)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum UnifiedEvent {
    Reasoning {
        text: String,
    },
    Text {
        text: String,
    },
    ToolCall {
        name: String,
        #[serde(default)]
        arguments: serde_json::Value,
    },
}

#[derive(Deserialize, Serialize, PartialEq, Debug)]
struct ExpectEntry {
    verdict: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    class: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    note: Option<String>,
}

fn golden_dir() -> PathBuf {
    common::ensure_unified_golden()
}

fn load_golden_files() -> Vec<(String, GoldenFile)> {
    let dir = golden_dir();
    let mut out = Vec::new();
    for entry in std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("read {}: {e}", dir.display())) {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) != Some("yaml") {
            continue;
        }
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let parsed: GoldenFile =
            serde_yaml::from_str(&text).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()));
        out.push((path.display().to_string(), parsed));
    }
    out
}

#[test]
fn every_golden_file_round_trips_through_the_schema() {
    let files = load_golden_files();
    assert!(
        !files.is_empty(),
        "no golden files found in {}",
        golden_dir().display()
    );

    for (path, parsed) in &files {
        // parse -> serialize -> parse must be stable (the schema is total for the corpus).
        let reserialized = serde_yaml::to_string(parsed).unwrap();
        let reparsed: GoldenFile =
            serde_yaml::from_str(&reserialized).unwrap_or_else(|e| panic!("re-parse {path}: {e}"));
        assert_eq!(parsed, &reparsed, "round-trip changed {path}");
    }
}

#[test]
fn every_case_is_well_formed() {
    for (path, file) in load_golden_files() {
        assert!(!file.cases.is_empty(), "{path}: no cases");
        for (id, case) in &file.cases {
            assert!(
                id.starts_with("UNIFIED."),
                "{path}: case id `{id}` missing UNIFIED. prefix"
            );
            assert!(
                id.ends_with(&format!(".{}", file.family)),
                "{path}: case id `{id}` does not end with family `.{}`",
                file.family
            );
            assert!(!case.input.is_empty(), "{path}: `{id}` has empty input");
            // An EMPTY golden is legitimate: a turn made only of control markup must
            // emit nothing at all, and forbidding it is why no row ever pinned that.
            // A golden forgotten by mistake is still caught — `unified_parity` compares
            // every case against the live parser, so an empty golden passes only when
            // the parser genuinely produces no events. Require the description to SAY
            // so, else a reader cannot tell a deliberate no-op from a missing one.
            if case.golden.is_empty() {
                assert!(
                    case.description
                        .to_ascii_lowercase()
                        .contains("emits nothing")
                        || case
                            .description
                            .to_ascii_lowercase()
                            .contains("nothing is emitted")
                        || case
                            .description
                            .to_ascii_lowercase()
                            .contains("emits no events"),
                    "{path}: `{id}` has an empty golden but its description does not say \
                     the turn emits nothing"
                );
            }
            assert!(
                !case.description.is_empty(),
                "{path}: `{id}` has empty description"
            );
            for engine in ["vllm", "dynamo"] {
                let e = case
                    .expect
                    .get(engine)
                    .unwrap_or_else(|| panic!("{path}: `{id}` missing expect.{engine}"));
                match e.verdict.as_str() {
                    "match" => {}
                    "diverge" => assert!(
                        e.class.is_some(),
                        "{path}: `{id}` expect.{engine} diverge without a class"
                    ),
                    other => panic!("{path}: `{id}` expect.{engine} unknown verdict `{other}`"),
                }
            }
        }
    }
}

#[test]
fn seed_corpus_covers_every_verdict_category() {
    let files = load_golden_files();

    let mut families = BTreeSet::new();
    let mut classes = BTreeSet::new();
    let mut dynamo_diverges = false;
    let mut vllm_diverges = false;
    let mut total_cases = 0usize;

    for (_, file) in &files {
        families.insert(file.family.clone());
        for case in file.cases.values() {
            total_cases += 1;
            if let Some(e) = case.expect.get("dynamo")
                && e.verdict == "diverge"
            {
                dynamo_diverges = true;
                classes.extend(e.class.clone());
            }
            if let Some(e) = case.expect.get("vllm")
                && e.verdict == "diverge"
            {
                vllm_diverges = true;
                classes.extend(e.class.clone());
            }
        }
    }

    // Baselines + both red categories must exist, or the surface is measuring a strawman.
    assert!(
        total_cases >= 10,
        "seed corpus too small: {total_cases} cases"
    );
    for fam in ["gemma4", "qwen3", "kimi_k2", "muse_glimmer"] {
        assert!(
            families.contains(fam),
            "missing family `{fam}` in seed corpus"
        );
    }
    assert!(
        dynamo_diverges,
        "no Dynamo-red case — the unification gap is unproven"
    );
    assert!(
        vllm_diverges,
        "no vLLM-red case — the spec must catch vLLM too, not treat it as truth"
    );
    for expected in ["ORDER", "MERGE", "ERROR", "ARG_MISMATCH", "LEAK"] {
        assert!(
            classes.contains(expected),
            "seed corpus does not exercise divergence class `{expected}`"
        );
    }
}
