// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Chunk-split sweep for the v2 streaming tool-call parsers.
//!
//! `conformance_toolcalling_stream.rs` pins per-chunk behavior to captured fixture
//! expectations at ONE chunking (the one recorded at capture time). This test
//! checks the complementary property: **chunk-size invariance**. For every
//! stream case, the assembled output (tool calls + `normal_text`) must be
//! identical no matter where the input is split. A marker split across a chunk
//! boundary that leaks into `normal_text` or corrupts `arguments` shows up as
//! a divergence from the single-shot parse — the bug class behind vLLM's
//! `</｜DSML｜parameter` arguments leak and vllm#48846's whitespace loss.
//!
//! Reference = a fresh parser's `parse_complete(full_text)` (single shot).
//! Sweeps per case:
//!   1. char-by-char and fixed chunk sizes (1, 2, 3, 5, 7, 11, 13 chars) —
//!      size 1 makes EVERY split point a chunk boundary;
//!   2. every 2-part split (long prefix, then the rest) — the "stable stream,
//!      then one awkward split" pattern. Cases longer than `MAX_TWO_PART`
//!      chars are strided down to `MAX_TWO_PART` boundaries and the subsample
//!      is reported, never silent.
//!
//! The reference itself is pinned to captured expectations by the parity test,
//! so equality here means every chunking matches the pinned behavior too.
//!
//! Coverage is enforced: every family in `REGISTERED_FAMILIES` must sweep at
//! least one case (`harmony` sweeps the token path via `push_tokens`), so a
//! newly registered family without stream fixtures fails this suite instead of
//! silently skipping.
//!
//! Known chunking-dependent cases are allowlisted in
//! `conformance/toolcalling/known-chunking-divergences.yaml` — strict both
//! ways (undocumented divergence fails; a documented case that stops
//! diverging fails as stale), same policy as `known-divergences.yaml`.

use std::collections::{BTreeMap, BTreeSet};

mod common;
use common::{collect_yaml, fixture_name};

use dynamo_parsers_v2::{
    REGISTERED_FAMILIES, Tool, ToolParseResult, ToolParser, create_tool_parser_for_family,
};
use serde::Deserialize;
use serde_json::Value;

/// Cap on 2-part split boundaries per case; longer cases are strided.
const MAX_TWO_PART: usize = 512;
/// Fixed chunk sizes (in chars for text, in ids for the token path).
const CHUNK_SIZES: &[usize] = &[1, 2, 3, 5, 7, 11, 13];

// ── Fixture schema (inputs only — expectations are not read here) ────────────

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
}

#[derive(Deserialize)]
struct Chunk {
    #[serde(default)]
    delta_text: String,
    #[serde(default)]
    delta_token_ids: Vec<u32>,
}

// ── Assembled-output model ────────────────────────────────────────────────────

/// Chunking-independent view of a full parse: per-index (name, parsed args)
/// plus the concatenated `normal_text`. Delta *granularity* may differ across
/// chunkings (name+args in one delta vs split); the assembly must not.
#[derive(Debug, PartialEq, Eq)]
struct Assembled {
    calls: Vec<(String, Value)>,
    normal_text: String,
}

fn assemble(results: &[ToolParseResult]) -> Assembled {
    let mut names: BTreeMap<usize, String> = BTreeMap::new();
    let mut args: BTreeMap<usize, String> = BTreeMap::new();
    let mut complete: BTreeMap<usize, bool> = BTreeMap::new();
    let mut normal_text = String::new();
    for r in results {
        normal_text.push_str(&r.normal_text);
        for d in &r.calls {
            if let Some(n) = &d.name {
                names.entry(d.tool_index).or_default().push_str(n);
            }
            args.entry(d.tool_index).or_default().push_str(&d.arguments);
            *complete.entry(d.tool_index).or_default() |= d.complete;
        }
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
    Assembled { calls, normal_text }
}

#[test]
fn assemble_omits_incomplete_tool_deltas() {
    let result = ToolParseResult {
        normal_text: String::new(),
        calls: vec![dynamo_parsers_v2::ToolCallDelta {
            tool_index: 0,
            name: Some("get_weather".into()),
            arguments: r#"{"city":"Par"#.into(),
            complete: false,
        }],
    };
    assert!(assemble(&[result]).calls.is_empty());
}

fn new_parser(family: &str, tools: &[Tool]) -> Box<dyn ToolParser> {
    create_tool_parser_for_family(family, tools)
        .unwrap_or_else(|e| panic!("create parser for '{family}': {e}"))
}

/// Push text chunks + finish, returning every intermediate result.
fn run_text(family: &str, tools: &[Tool], chunks: &[&str]) -> Vec<ToolParseResult> {
    let mut parser = new_parser(family, tools);
    let mut out = Vec::with_capacity(chunks.len() + 1);
    for c in chunks {
        out.push(parser.push(c).unwrap_or_else(|e| panic!("push: {e}")));
    }
    out.push(parser.finish().unwrap_or_else(|e| panic!("finish: {e}")));
    out
}

/// Push token-id chunks + finish (harmony token path).
fn run_tokens(family: &str, tools: &[Tool], chunks: &[&[u32]]) -> Vec<ToolParseResult> {
    let mut parser = new_parser(family, tools);
    let mut out = Vec::with_capacity(chunks.len() + 1);
    for c in chunks {
        out.push(
            parser
                .push_tokens(c)
                .unwrap_or_else(|e| panic!("push_tokens: {e}")),
        );
    }
    out.push(parser.finish().unwrap_or_else(|e| panic!("finish: {e}")));
    out
}

/// Boundary indices for 2-part splits, strided down to `MAX_TWO_PART`.
/// Returns (boundaries, was_subsampled). Boundaries are element counts (chars
/// or ids), 1..len exclusive of the trivial empty splits.
fn two_part_boundaries(len: usize) -> (Vec<usize>, bool) {
    if len <= 1 {
        return (Vec::new(), false);
    }
    let n = len - 1;
    if n <= MAX_TWO_PART {
        return ((1..len).collect(), false);
    }
    let stride = n.div_ceil(MAX_TWO_PART);
    ((1..len).step_by(stride).collect(), true)
}

/// Split `text` into chunks of `k` chars (respecting UTF-8 boundaries).
fn char_chunks(text: &str, k: usize) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut count = 0;
    for (i, _) in text.char_indices() {
        if count == k {
            out.push(&text[start..i]);
            start = i;
            count = 0;
        }
        count += 1;
    }
    if start < text.len() {
        out.push(&text[start..]);
    }
    out
}

struct SweepStats {
    cases: usize,
    chunkings: usize,
    subsampled_cases: usize,
}

/// Load the strict chunking-divergence allowlist: family -> case id -> note.
fn load_allowlist() -> BTreeMap<String, BTreeMap<String, String>> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("toolcalling/known-chunking-divergences.yaml");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let allow: BTreeMap<String, BTreeMap<String, String>> =
        serde_yaml::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    for (fam, cases) in &allow {
        for (cid, note) in cases {
            assert!(
                !note.trim().is_empty(),
                "{fam}:{cid}: empty note in known-chunking-divergences.yaml"
            );
        }
    }
    allow
}

// ── Test ──────────────────────────────────────────────────────────────────────

#[test]
fn toolcalling_stream_split_sweep() {
    let sv2 = common::ensure_fixtures().join("toolcalling/fixtures-stream-v2");
    let inputs_root = sv2.join("inputs");
    let mut files = Vec::new();
    collect_yaml(&inputs_root, &mut files);
    files.sort();
    assert!(
        !files.is_empty(),
        "no fixtures found under {}",
        inputs_root.display()
    );

    let allowlist = load_allowlist();
    let mut stats: BTreeMap<String, SweepStats> = BTreeMap::new();
    let mut failures: Vec<String> = Vec::new();
    // (family, case id) pairs that diverged at any chunking this run, and all
    // pairs swept this run (for allowlist stale detection).
    let mut diverged: BTreeSet<(String, String)> = BTreeSet::new();
    let mut swept: BTreeSet<(String, String)> = BTreeSet::new();

    for path in &files {
        let yaml = std::fs::read_to_string(path).unwrap();
        let fx: Fixture = match serde_yaml::from_str(&yaml) {
            Ok(f) => f,
            Err(e) => panic!("{}: YAML parse error: {e}", path.display()),
        };
        if !matches!(fx.mode.as_deref(), Some("stream" | "streamv2")) {
            continue;
        }
        if !REGISTERED_FAMILIES.contains(&fx.family.as_str()) {
            continue;
        }
        // `harmony` is the token path; every other family (incl. harmony_text)
        // sweeps text. Mirrors the preferred_input split in parser_families.yaml.
        let token_path = fx.family == "harmony";
        eprintln!("sweeping {}", fixture_name(path));

        for (cid, case) in &fx.cases {
            let stat = stats.entry(fx.family.clone()).or_insert(SweepStats {
                cases: 0,
                chunkings: 0,
                subsampled_cases: 0,
            });
            let label = format!("{} {cid}", fx.family);
            let allowlisted = allowlist
                .get(&fx.family)
                .is_some_and(|c| c.contains_key(cid.as_str()));

            if token_path {
                let full: Vec<u32> = case
                    .chunks
                    .iter()
                    .flat_map(|c| c.delta_token_ids.iter().copied())
                    .collect();
                if full.is_empty() {
                    continue;
                }
                stat.cases += 1;
                swept.insert((fx.family.clone(), cid.clone()));
                let reference = assemble(&run_tokens(&fx.family, &case.tools, &[&full]));

                let mut check = |chunks: &[&[u32]], desc: &str| {
                    let got = assemble(&run_tokens(&fx.family, &case.tools, chunks));
                    stat.chunkings += 1;
                    if got != reference {
                        diverged.insert((fx.family.clone(), cid.clone()));
                        if !allowlisted {
                            failures.push(format!(
                                "{label} [{desc}]:\n        got  {got:?}\n        want {reference:?}"
                            ));
                        }
                    }
                };
                for &k in CHUNK_SIZES {
                    if k < full.len() {
                        let chunks: Vec<&[u32]> = full.chunks(k).collect();
                        check(&chunks, &format!("ids k={k}"));
                    }
                }
                let (boundaries, subsampled) = two_part_boundaries(full.len());
                if subsampled {
                    stat.subsampled_cases += 1;
                }
                for b in boundaries {
                    check(&[&full[..b], &full[b..]], &format!("ids split@{b}"));
                }
            } else {
                let full: String = case.chunks.iter().map(|c| c.delta_text.as_str()).collect();
                if full.is_empty() {
                    continue;
                }
                stat.cases += 1;
                swept.insert((fx.family.clone(), cid.clone()));
                let reference = assemble(&run_text(&fx.family, &case.tools, &[&full]));

                let mut check = |chunks: &[&str], desc: &str| {
                    let got = assemble(&run_text(&fx.family, &case.tools, chunks));
                    stat.chunkings += 1;
                    if got != reference {
                        diverged.insert((fx.family.clone(), cid.clone()));
                        if !allowlisted {
                            failures.push(format!(
                                "{label} [{desc}]:\n        got  {got:?}\n        want {reference:?}"
                            ));
                        }
                    }
                };
                let char_count = full.chars().count();
                for &k in CHUNK_SIZES {
                    if k < char_count {
                        check(&char_chunks(&full, k), &format!("k={k}"));
                    }
                }
                let char_starts: Vec<usize> = full.char_indices().map(|(i, _)| i).collect();
                let (boundaries, subsampled) = two_part_boundaries(char_starts.len());
                if subsampled {
                    stat.subsampled_cases += 1;
                }
                for b in boundaries {
                    let byte = char_starts[b];
                    check(&[&full[..byte], &full[byte..]], &format!("split@{b}"));
                }
            }
        }
    }

    let mut total_cases = 0usize;
    let mut total_chunkings = 0usize;
    for (family, s) in &stats {
        total_cases += s.cases;
        total_chunkings += s.chunkings;
        let note = if s.subsampled_cases > 0 {
            format!(
                " ({} case(s) 2-part-strided to {MAX_TWO_PART})",
                s.subsampled_cases
            )
        } else {
            String::new()
        };
        eprintln!(
            "  {family}: {} cases, {} chunkings{note}",
            s.cases, s.chunkings
        );
    }
    eprintln!("split sweep: {total_cases} cases, {total_chunkings} chunkings total");

    // Coverage enforcement: a registered family with zero swept cases is a
    // hole in the suite, not a skip. (`harmony`/`harmony_text` are one parser
    // driven two ways; both must appear in the corpus.)
    for family in REGISTERED_FAMILIES {
        let n = stats.get(*family).map(|s| s.cases).unwrap_or(0);
        assert!(
            n > 0,
            "family '{family}' is registered in REGISTERED_FAMILIES but swept 0 stream cases — \
             add stream-v2 fixtures (or a trace) for it"
        );
    }

    // Allowlist reconciliation — strict both ways.
    let mut known = 0usize;
    for (fam, cases) in &allowlist {
        for cid in cases.keys() {
            let key = (fam.clone(), cid.clone());
            if !swept.contains(&key) {
                failures.push(format!(
                    "{fam}:{cid}: allowlisted in known-chunking-divergences.yaml but no such \
                     case was swept — stale or misspelled entry"
                ));
            } else if diverged.contains(&key) {
                known += 1;
                eprintln!("KNOWN {fam} {cid}: chunking-divergent (allowlisted)");
            } else {
                failures.push(format!(
                    "{fam}:{cid}: allowlisted as chunking-divergent but now matches the \
                     single-shot parse at every chunking — STALE, delete the entry"
                ));
            }
        }
    }
    if known > 0 {
        eprintln!("{known} known chunking-divergent case(s) allowlisted");
    }

    if !failures.is_empty() {
        for f in &failures {
            eprintln!("FAIL {f}");
        }
        panic!(
            "{} chunking(s) diverged from the single-shot parse across {} case(s)",
            failures.len(),
            total_cases
        );
    }
}
