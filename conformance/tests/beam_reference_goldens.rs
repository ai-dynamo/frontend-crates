// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Serving-level differential test of the `beam` V2 unified parser against the model
//! builder's frozen reference corpus (`fixtures-beam/reference-goldens.jsonl`, see the
//! NOTICE next to it).
//!
//! Each case records the generated text as chunks, the request's tools, and what the
//! reference serving layer produced: `reasoning`, `content`, typed `calls`, and the regions
//! it judged `unusable`. The parser is replayed twice per case — on the recorded chunking and
//! on the whole text — and must reproduce the reference, except for the cases listed with a
//! reason in `fixtures-beam/reference-known-divergences.yaml`. That list is checked in both
//! directions: an unlisted mismatch fails, and a listed case that no longer mismatches fails.
//!
//! The recorded prompt prefix selects the parser's starting state: the plain anchor
//! (`<|start|>assistant`, an empty prefix, or a history ending in the anchor) starts
//! between messages; a prefix that opened the reply starts in `Response`, one that
//! opened a thought in `Reasoning`. Prefixes Dynamo's prompt never produces (no anchor,
//! a marker or word cut in half, a prefill inside a tool body) are counted, not replayed.

use std::collections::{BTreeMap, BTreeSet};

use dynamo_parsers_v2::{
    Tool, ToolCallPolicy, UnifiedEvent, UnifiedParserExt, UnifiedParserInit,
    UnifiedParserStartingState, create_unified_parser_for_family,
};
use serde_json::Value;

const GOLDENS: &str = include_str!("../fixtures-beam/reference-goldens.jsonl");
const KNOWN: &str = include_str!("../fixtures-beam/reference-known-divergences.yaml");
const PLAIN_PREFIX: &str = "<|start|>assistant";

#[derive(Debug, Clone, PartialEq)]
struct Served {
    reasoning: String,
    content: String,
    /// `(name, arguments)` in emission order; arguments compared as JSON values.
    calls: Vec<(String, Value)>,
}

struct Case {
    id: String,
    prefix: String,
    chunks: Vec<String>,
    tools: Vec<Tool>,
    serving: Served,
    serving_batch: Served,
    unusable: usize,
}

fn served(v: &Value) -> Served {
    Served {
        reasoning: v["reasoning"].as_str().unwrap_or("").to_string(),
        content: v["content"].as_str().unwrap_or("").to_string(),
        calls: v["calls"]
            .as_array()
            .map(|calls| {
                calls
                    .iter()
                    .map(|c| {
                        let name = c["name"].as_str().unwrap_or("").to_string();
                        let args = match &c["arguments"] {
                            Value::String(s) => {
                                serde_json::from_str(s).unwrap_or(Value::String(s.clone()))
                            }
                            other => other.clone(),
                        };
                        (name, args)
                    })
                    .collect()
            })
            .unwrap_or_default(),
    }
}

fn load() -> Vec<Case> {
    GOLDENS
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            let d: Value = serde_json::from_str(line).expect("golden jsonl line");
            let tools = d["tools"]
                .as_array()
                .map(|ts| {
                    ts.iter()
                        .map(|t| {
                            let f = &t["function"];
                            Tool {
                                name: f["name"].as_str().unwrap_or("").to_string(),
                                description: None,
                                parameters: f.get("parameters").cloned().unwrap_or(Value::Null),
                                strict: f.get("strict").and_then(Value::as_bool),
                            }
                        })
                        .collect()
                })
                .unwrap_or_default();
            let batch = d
                .get("serving_batch")
                .filter(|v| !v.is_null())
                .unwrap_or(&d["serving"]);
            Case {
                id: d["id"].as_str().unwrap().to_string(),
                prefix: d["prefix"].as_str().unwrap_or("").to_string(),
                chunks: d["chunks"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|c| c.as_str().unwrap().to_string())
                    .collect(),
                tools,
                serving: served(&d["serving"]),
                serving_batch: served(batch),
                unusable: d["serving"]["unusable"].as_array().map_or(0, Vec::len),
            }
        })
        .collect()
}

/// The parser starting state a recorded prompt prefix stands for; `None` when the
/// prefix is a shape Dynamo's prompt never produces.
fn starting_state(prefix: &str) -> Option<UnifiedParserStartingState> {
    if prefix.is_empty() || prefix.ends_with(PLAIN_PREFIX) {
        return Some(UnifiedParserStartingState::None);
    }
    let tail = &prefix[prefix.rfind("<|start|>")?..];
    if tail.starts_with("<|start|>assistant<|channel|>final<|message|>") {
        return Some(UnifiedParserStartingState::Response);
    }
    if tail.starts_with("<|start|>assistant<|channel|>analysis<|message|>") {
        return Some(UnifiedParserStartingState::Reasoning);
    }
    None
}

fn run_v2(
    case: &Case,
    chunks: &[&str],
    state: UnifiedParserStartingState,
) -> anyhow::Result<Served> {
    let mut parser = create_unified_parser_for_family("beam", &case.tools)?;
    if state != UnifiedParserStartingState::None {
        parser.initialize_request(UnifiedParserInit {
            starting_state: state,
            ..UnifiedParserInit::default()
        })?;
    }
    parser.set_tool_call_policy(ToolCallPolicy::default())?;
    let mut deltas = Vec::new();
    for chunk in chunks {
        deltas.extend(parser.push(chunk)?);
    }
    deltas.extend(parser.finish()?);
    let mut out = Served {
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

fn served_json(s: &Served) -> Value {
    serde_json::json!({
        "reasoning": s.reasoning,
        "content": s.content,
        "calls": s.calls.iter().map(|(n, a)| serde_json::json!({"name": n, "arguments": a})).collect::<Vec<_>>(),
    })
}

fn describe(expected: &Served, got: &Served) -> String {
    let mut parts = Vec::new();
    if expected.reasoning != got.reasoning {
        parts.push(format!(
            "reasoning: expected {:?} got {:?}",
            clip(&expected.reasoning),
            clip(&got.reasoning)
        ));
    }
    if expected.content != got.content {
        parts.push(format!(
            "content: expected {:?} got {:?}",
            clip(&expected.content),
            clip(&got.content)
        ));
    }
    if expected.calls != got.calls {
        parts.push(format!(
            "calls: expected {} got {}",
            clip(&format!("{:?}", expected.calls)),
            clip(&format!("{:?}", got.calls))
        ));
    }
    parts.join(" | ")
}

fn clip(s: &str) -> String {
    const MAX: usize = 160;
    if s.chars().count() <= MAX {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(MAX).collect::<String>())
    }
}

fn known_divergences() -> BTreeMap<String, String> {
    let doc: Value = serde_yaml::from_str::<serde_yaml::Value>(KNOWN)
        .map(|y| serde_json::to_value(y).unwrap())
        .expect("known divergences yaml");
    doc["known_divergences"]
        .as_array()
        .map(|entries| {
            entries
                .iter()
                .map(|e| {
                    let id = e["id"].as_str().expect("id").to_string();
                    let reason = e["reason"].as_str().expect("reason").to_string();
                    assert!(!reason.trim().is_empty(), "divergence {id} needs a reason");
                    (id, reason)
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Which cases a replay covers.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Subset {
    /// The plain anchor: `<|start|>assistant`.
    Plain,
    /// A prefix the parser can start from (`starting_state` is `Some`) that is not plain.
    Prefilled,
}

fn subset_of(case: &Case) -> Option<Subset> {
    if case.prefix == PLAIN_PREFIX {
        return Some(Subset::Plain);
    }
    starting_state(&case.prefix).map(|_| Subset::Prefilled)
}

/// One route's replay outcome for a case: the notes of what differed, if anything.
fn judge(
    case: &Case,
    chunked: &anyhow::Result<Served>,
    batch: &anyhow::Result<Served>,
) -> Option<String> {
    let mut notes = Vec::new();
    match chunked {
        Ok(got) if *got == case.serving => {}
        Ok(got) => notes.push(format!("chunked: {}", describe(&case.serving, got))),
        Err(e) => notes.push(format!("chunked: parser error {e}")),
    }
    match batch {
        Ok(got) if *got == case.serving_batch => {}
        Ok(got) => notes.push(format!("batch: {}", describe(&case.serving_batch, got))),
        Err(e) => notes.push(format!("batch: parser error {e}")),
    }
    if notes.is_empty() {
        return None;
    }
    let unusable = if case.unusable > 0 {
        format!(" [reference marked {} region(s) unusable]", case.unusable)
    } else {
        String::new()
    };
    Some(format!("{}{}", notes.join(" || "), unusable))
}

/// Triage aid: `BEAM_DUMP_MISMATCHES=<path>` appends every mismatching case in full
/// (input, reference, replay) as one JSON object per line, tagged with the route.
fn dump(
    route: &str,
    case: &Case,
    chunked: &anyhow::Result<Served>,
    batch: &anyhow::Result<Served>,
    known: &BTreeMap<String, String>,
) {
    let Ok(path) = std::env::var("BEAM_DUMP_MISMATCHES") else {
        return;
    };
    use std::io::Write;
    let got = |r: &anyhow::Result<Served>| match r {
        Ok(s) => served_json(s),
        Err(e) => Value::String(format!("error: {e}")),
    };
    let row = serde_json::json!({
        "route": route,
        "id": case.id,
        "prefix": case.prefix,
        "input": case.chunks.concat(),
        "chunks": case.chunks,
        "tools": case.tools.iter().map(|t| &t.name).collect::<Vec<_>>(),
        "unusable": case.unusable,
        "expected_chunked": served_json(&case.serving),
        "got_chunked": got(chunked),
        "expected_batch": served_json(&case.serving_batch),
        "got_batch": got(batch),
        "known": known.get(&case.id),
    });
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .expect("dump file");
    writeln!(f, "{row}").expect("dump write");
}

/// The divergence list is exact for the replayed set: an unlisted mismatch fails, and a
/// listed case in the set that no longer mismatches fails.
fn check(
    label: &str,
    replayed: &BTreeSet<String>,
    mismatches: &BTreeMap<String, String>,
    known: &BTreeMap<String, String>,
) {
    assert!(!replayed.is_empty(), "{label}: nothing replayed");
    let mismatched: BTreeSet<&String> = mismatches.keys().collect();
    let listed: BTreeSet<&String> = known.keys().filter(|id| replayed.contains(*id)).collect();
    let unlisted: Vec<_> = mismatched.difference(&listed).collect();
    let stale: Vec<_> = listed.difference(&mismatched).collect();
    let mut report = String::new();
    if !unlisted.is_empty() {
        report += &format!(
            "\n{label}: {} unlisted mismatch(es) out of {} replayed cases:\n",
            unlisted.len(),
            replayed.len()
        );
        for id in &unlisted {
            report += &format!("  - {id}: {}\n", mismatches[id.as_str()]);
        }
    }
    if !stale.is_empty() {
        report += &format!(
            "\n{label}: {} listed divergence(s) no longer mismatch (remove them):\n",
            stale.len()
        );
        for id in &stale {
            report += &format!("  - {id}: {}\n", known[id.as_str()]);
        }
    }
    assert!(report.is_empty(), "{report}");
}

fn replay_v2(subset: Subset) {
    let cases = load();
    let known = known_divergences();
    let mut mismatches = BTreeMap::new();
    let mut replayed = BTreeSet::new();
    for case in cases.iter().filter(|c| subset_of(c) == Some(subset)) {
        let state = starting_state(&case.prefix).expect("replayable");
        replayed.insert(case.id.clone());
        let chunk_refs: Vec<&str> = case.chunks.iter().map(String::as_str).collect();
        let whole = case.chunks.concat();
        let chunked = run_v2(case, &chunk_refs, state);
        let batch = run_v2(case, &[whole.as_str()], state);
        if let Some(note) = judge(case, &chunked, &batch) {
            mismatches.insert(case.id.clone(), note);
            dump("v2", case, &chunked, &batch, &known);
        }
    }
    check(&format!("v2/{subset:?}"), &replayed, &mismatches, &known);
}

impl std::fmt::Debug for Subset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Subset::Plain => "plain",
            Subset::Prefilled => "prefilled",
        })
    }
}

#[test]
fn corpus_loads_with_the_recorded_shape() {
    let cases = load();
    assert_eq!(cases.len(), 288, "288 beam-template cases were extracted");
    let plain = cases.iter().filter(|c| c.prefix == PLAIN_PREFIX).count();
    assert_eq!(plain, 217, "217 cases use the plain assistant start anchor");
    let prefilled = cases
        .iter()
        .filter(|c| subset_of(c) == Some(Subset::Prefilled))
        .count();
    let not_applicable = cases.iter().filter(|c| subset_of(c).is_none()).count();
    // The remaining 71: 31 start from a state Dynamo can set (empty prefix or a history
    // ending in the anchor: 22; reply prefill: 4; thought prefill: 5), 40 are shapes
    // Dynamo's prompt never produces (no anchor, a marker or word cut in half, a
    // prefill inside a tool body).
    assert_eq!((prefilled, not_applicable), (31, 40));
    assert!(
        cases
            .iter()
            .all(|c| !c.chunks.is_empty() || c.serving.content.is_empty())
    );
}

#[test]
fn plain_prefix_cases_match_the_reference_except_known_divergences() {
    replay_v2(Subset::Plain);
}

#[test]
fn prefilled_prefix_cases_match_the_reference_except_known_divergences() {
    replay_v2(Subset::Prefilled);
}
