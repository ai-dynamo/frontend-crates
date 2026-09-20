// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compute the Dynamo v2 column LIVE, render the unified conformance tab, and
//! write the capture the packaged shard is built from.
//!
//! Columns: GOLDEN (authored oracle) | vLLM (captured) | Dynamo v2.
//!
//! The Dynamo column is a PER-FAMILY MIXTURE, the same shape as vLLM Rust's: the
//! native UnifiedParser where one exists, and the v1-reasoning + v2-tool split
//! everywhere else. The split is the "before" state — it parses ALL reasoning
//! first, so reasoning interleaved with tool calls loses its position. Every
//! current corpus family has a native UnifiedParser; a new split-only family
//! reintroduces the fallback.
//!
//! Output: `conformance/unified/unified_results.yaml`. The exploder and packager turn that
//! feed into the committed `dynamo_v2-<ver>` shard, which is what the
//! CONFORMANCE_v2.html tab actually reads — the tab never runs these parsers. The
//! `committed_dynamo_capture_matches_the_live_parsers` test below fails if that
//! shard drifts from the parsers.

use std::collections::BTreeMap;
use std::path::PathBuf;

use dynamo_parsers::{ReasoningParser, ReasoningParserType};
use dynamo_parsers_v2::{
    UnifiedParserExt, assemble, create_tool_parser_for_family, create_unified_parser_for_family,
};
use serde::Deserialize;
use serde_json::{Value, json};

mod common;

use common::{Init, unified_tools as tools};

#[derive(Deserialize)]
struct GoldenFile {
    family: String,
    cases: BTreeMap<String, GoldenCase>,
}

#[derive(Deserialize)]
struct GoldenCase {
    description: String,
    #[serde(default)]
    policy: Vec<String>,
    #[serde(default)]
    init: Init,
    #[serde(default)]
    finish_reason: Option<String>,
    input: String,
    golden: Vec<Ev>,
    expect: BTreeMap<String, Expect>,
}

#[derive(Deserialize, Clone)]
struct Expect {
    verdict: String,
    #[serde(default)]
    class: Option<String>,
    #[serde(default)]
    note: Option<String>,
}

/// A unified event. PartialEq drives the golden comparison.
#[derive(Deserialize, serde::Serialize, Clone, PartialEq, Debug)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Ev {
    Reasoning {
        text: String,
    },
    Text {
        text: String,
    },
    ToolCall {
        name: String,
        #[serde(default)]
        arguments: Value,
    },
}

impl Ev {
    fn render(&self) -> String {
        match self {
            Ev::Reasoning { text } => format!("reasoning({text:?})"),
            Ev::Text { text } => format!("text({text:?})"),
            Ev::ToolCall { name, arguments } => format!("tool_call({name}, {arguments})"),
        }
    }
}

impl common::UnifiedEventView for Ev {
    fn reasoning_text(&self) -> Option<&str> {
        match self {
            Ev::Reasoning { text } => Some(text),
            _ => None,
        }
    }

    fn visible_text(&self) -> Option<&str> {
        match self {
            Ev::Text { text } => Some(text),
            _ => None,
        }
    }

    fn tool_call(&self) -> Option<(&str, &Value)> {
        match self {
            Ev::ToolCall { name, arguments } => Some((name, arguments)),
            _ => None,
        }
    }
}

/// Map a corpus family to (v1 reasoning parser, v2 tool parser) for the SPLIT path.
/// Declared once in `parser_families.yaml` under `unified:`; see `common::unified_family`.
fn parsers_for(family: &str) -> (String, String) {
    let f = common::unified_family(family);
    (f.reasoning_parser, f.tool_parser)
}

/// Fold one tool-parser result into the event list, preserving text/call order
/// and coalescing per-`tool_index` deltas into one call.
fn feed(
    res: dynamo_parsers_v2::ToolParseResult,
    out: &mut Vec<Ev>,
    slots: &mut BTreeMap<usize, usize>,
    raw_args: &mut BTreeMap<usize, String>,
) {
    if !res.normal_text.is_empty() {
        if let Some(Ev::Text { text }) = out.last_mut() {
            text.push_str(&res.normal_text);
        } else {
            out.push(Ev::Text {
                text: res.normal_text,
            });
        }
    }
    for d in res.calls {
        let pos = *slots.entry(d.tool_index).or_insert_with(|| {
            out.push(Ev::ToolCall {
                name: d.name.clone().unwrap_or_default(),
                arguments: Value::Null,
            });
            out.len() - 1
        });
        if let Some(n) = &d.name
            && let Ev::ToolCall { name, .. } = &mut out[pos]
            && name.is_empty()
        {
            *name = n.clone();
        }
        raw_args
            .entry(d.tool_index)
            .or_default()
            .push_str(&d.arguments);
    }
}

impl From<dynamo_parsers_v2::UnifiedEvent> for Ev {
    fn from(e: dynamo_parsers_v2::UnifiedEvent) -> Self {
        match e {
            dynamo_parsers_v2::UnifiedEvent::Reasoning { text } => Ev::Reasoning { text },
            dynamo_parsers_v2::UnifiedEvent::Text { text } => Ev::Text { text },
            dynamo_parsers_v2::UnifiedEvent::ToolCall { name, arguments } => {
                Ev::ToolCall { name, arguments }
            }
        }
    }
}

fn unified_delta_json(d: &dynamo_parsers_v2::UnifiedParserEvent) -> Value {
    match d {
        dynamo_parsers_v2::UnifiedParserEvent::Reasoning(text) => {
            json!({"kind": "reasoning", "text": text})
        }
        dynamo_parsers_v2::UnifiedParserEvent::Text(text) => json!({"kind": "text", "text": text}),
        dynamo_parsers_v2::UnifiedParserEvent::ToolCall(c) => {
            json!({"kind": "tool_call", "name": c.name, "arguments": c.arguments, "complete": c.complete})
        }
    }
}

/// Compute the Dynamo v2 unified event list for one input.
///
/// A family with a native unified parser is parsed by one state machine per
/// stream owning reasoning + content + tool calls. Every current corpus family
/// takes this path; the split fallback remains for other inputs.
///
/// Both paths are driven from the SAME chunking as `dynamo_chunks`, so the
/// assembled row and the per-chunk rows in the popup describe one run.
fn dynamo_events(family: &str, input: &str, init: &Init) -> Vec<Ev> {
    if let Ok(mut parser) = create_unified_parser_for_family(family, &tools()) {
        init.apply(&mut parser, family);

        let mut deltas = Vec::new();
        for chunk in chunk_input(input) {
            deltas.extend(
                parser
                    .push(&chunk)
                    .unwrap_or_else(|e| panic!("unified push `{family}`: {e}")),
            );
        }
        deltas.extend(
            parser
                .finish()
                .unwrap_or_else(|e| panic!("unified finish `{family}`: {e}")),
        );
        return assemble(&deltas).into_iter().map(Ev::from).collect();
    }

    let (reasoning_name, tool_family) = parsers_for(family);

    let mut rp = ReasoningParserType::get_reasoning_parser_from_name(&reasoning_name);
    let split = rp.detect_and_parse_reasoning(input, &[]);

    let mut out = Vec::new();
    if !split.reasoning_text.is_empty() {
        out.push(Ev::Reasoning {
            text: split.reasoning_text.clone(),
        });
    }

    let mut tp = create_tool_parser_for_family(&tool_family, &tools())
        .unwrap_or_else(|e| panic!("create tool parser for `{tool_family}`: {e}"));
    let mut slots: BTreeMap<usize, usize> = BTreeMap::new();
    let mut raw_args: BTreeMap<usize, String> = BTreeMap::new();
    let mut buf = [0u8; 4];
    for ch in split.normal_text.chars() {
        let s = ch.encode_utf8(&mut buf);
        let r = tp
            .push(s)
            .unwrap_or_else(|e| panic!("push `{tool_family}`: {e}"));
        feed(r, &mut out, &mut slots, &mut raw_args);
    }
    let r = tp
        .finish()
        .unwrap_or_else(|e| panic!("finish `{tool_family}`: {e}"));
    feed(r, &mut out, &mut slots, &mut raw_args);

    for (ti, pos) in &slots {
        let raw = raw_args.get(ti).map(String::as_str).unwrap_or("");
        let val = if raw.trim().is_empty() {
            json!({})
        } else {
            serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_string()))
        };
        if let Ev::ToolCall { arguments, .. } = &mut out[*pos] {
            *arguments = val;
        }
    }
    out
}

/// Tokenize an input into streaming chunks: each control marker (`<...>`, incl.
/// `<|...|>` / `<|"|>`) is its own chunk, and each run of text between markers is
/// a chunk. Generic across the gemma4 / qwen3 / kimi grammars.
fn chunk_input(input: &str) -> Vec<String> {
    let bytes = input.as_bytes();
    let mut chunks = Vec::new();
    let mut i = 0;
    let mut text_start = 0;
    while i < bytes.len() {
        if bytes[i] == b'<' {
            // flush any pending text run
            if text_start < i {
                chunks.push(input[text_start..i].to_string());
            }
            // consume through the matching '>'
            let mut j = i + 1;
            while j < bytes.len() && bytes[j] != b'>' {
                j += 1;
            }
            let end = (j + 1).min(bytes.len());
            chunks.push(input[i..end].to_string());
            i = end;
            text_start = i;
        } else {
            i += 1;
        }
    }
    if text_start < bytes.len() {
        chunks.push(input[text_start..].to_string());
    }
    chunks
}

/// One streaming chunk: the delta text fed, and the RAW per-chunk deltas Dynamo
/// emitted (reasoning/text/tool_call fragments, not coalesced) as JSON.
struct ChunkRow {
    delta_text: String,
    deltas: Vec<Value>,
}

fn tool_deltas(res: &dynamo_parsers_v2::ToolParseResult, out: &mut Vec<Value>) {
    if !res.normal_text.is_empty() {
        out.push(json!({"kind": "text", "text": res.normal_text}));
    }
    for c in &res.calls {
        out.push(json!({"kind": "tool_call", "name": c.name, "arguments": c.arguments}));
    }
}

/// Stream `input` through Dynamo's split pipeline CHUNK BY CHUNK, recording the
/// real per-chunk emitted deltas (v1 reasoning streaming incremental -> v2 tool
/// streaming push on the leftover content).
fn dynamo_chunks(family: &str, input: &str, init: &Init) -> Vec<ChunkRow> {
    // Unified families: ONE parser, so a chunk's deltas are simply what it emitted.
    if let Ok(mut parser) = create_unified_parser_for_family(family, &tools()) {
        init.apply(&mut parser, family);

        let mut rows = Vec::new();
        for chunk in chunk_input(input) {
            let deltas = parser.push(&chunk).expect("native capture push failed");
            rows.push(ChunkRow {
                delta_text: chunk,
                deltas: deltas.iter().map(unified_delta_json).collect(),
            });
        }
        let tail: Vec<Value> = parser
            .finish()
            .expect("native capture finish failed")
            .iter()
            .map(unified_delta_json)
            .collect();
        rows.push(ChunkRow {
            delta_text: "‹finish›".to_string(),
            deltas: tail,
        });
        return rows;
    }

    let (reasoning_name, tool_family) = parsers_for(family);
    let mut rp = ReasoningParserType::get_reasoning_parser_from_name(&reasoning_name);
    let mut tp = create_tool_parser_for_family(&tool_family, &tools())
        .unwrap_or_else(|e| panic!("create tool parser for `{tool_family}`: {e}"));

    let mut rows = Vec::new();
    for chunk in chunk_input(input) {
        let mut deltas: Vec<Value> = Vec::new();
        let rr = rp.parse_reasoning_streaming_incremental(&chunk, &[]);
        if !rr.reasoning_text.is_empty() {
            deltas.push(json!({"kind": "reasoning", "text": rr.reasoning_text}));
        }
        if !rr.normal_text.is_empty() {
            let tr = tp.push(&rr.normal_text).expect("split capture push failed");
            tool_deltas(&tr, &mut deltas);
        }
        rows.push(ChunkRow {
            delta_text: chunk,
            deltas,
        });
    }
    // Flush: reasoning tail -> tool -> finish.
    let mut tail: Vec<Value> = Vec::new();
    let rf = rp.finish_reasoning_stream();
    if !rf.reasoning_text.is_empty() {
        tail.push(json!({"kind": "reasoning", "text": rf.reasoning_text}));
    }
    if !rf.normal_text.is_empty() {
        let tr = tp
            .push(&rf.normal_text)
            .expect("split capture tail push failed");
        tool_deltas(&tr, &mut tail);
    }
    tool_deltas(
        &tp.finish().expect("split capture finish failed"),
        &mut tail,
    );
    rows.push(ChunkRow {
        delta_text: "‹finish›".to_string(),
        deltas: tail,
    });
    rows
}

#[test]
fn finish_is_part_of_the_stream_schedule_even_when_it_emits_nothing() {
    let rows = dynamo_chunks("qwen3", "plain response", &Init::default());
    let finish = rows.last().expect("finish row");
    assert_eq!(finish.delta_text, "‹finish›");
    assert!(
        finish.deltas.is_empty(),
        "fixture must exercise an empty finish"
    );
}

fn todo_for(class: &str) -> &'static str {
    match class {
        "MERGE" | "ORDER" => {
            "TODO: adopt the UnifiedParser. The split parses ALL reasoning first, so reasoning that occurs between or after tool calls is merged up front and loses its position. One state machine per stream (owning reasoning+content+tools) fixes this by construction."
        }
        "LOSS" => {
            "TODO: content/reasoning dropped by the split. UnifiedParser must preserve every segment in order."
        }
        _ => "TODO: unify reasoning + tool parsing into one ordered event stream.",
    }
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn events_html(evs: &[Ev]) -> String {
    if evs.is_empty() {
        return "<i>(no events)</i>".to_string();
    }
    evs.iter()
        .map(|e| format!("<div>{}</div>", esc(&e.render())))
        .collect()
}

struct Cell {
    verdict: String, // "MATCH" or a divergence class
    tip: String,
}

fn cell(
    engine: &str,
    input: &str,
    golden: &[Ev],
    got_html: &str,
    verdict: &str,
    class: &str,
    extra: &str,
) -> Cell {
    let v = if verdict == "match" || class == "MATCH" {
        "MATCH"
    } else {
        class
    };
    let tip = format!(
        "<b>{engine}</b> — <b>{v}</b><hr><b>input</b><pre>{}</pre><b>golden</b>{}<b>{engine}</b>{}{}",
        esc(input),
        events_html(golden),
        got_html,
        extra,
    );
    Cell {
        verdict: v.to_string(),
        tip,
    }
}

#[test]
fn render_unified_conformance_html() {
    let capture_provenance = common::dynamo_capture_provenance(None);
    // The vLLM column is LIVE, not an expectation. `capture_vllm_rust_unified.py`
    // records the `vllm-parser` crate against this same corpus; reading it here is
    // what makes the column evidence instead of a claim.
    let vllm_live: BTreeMap<(String, String), Vec<Ev>> = {
        let froot = common::ensure_fixtures().join("unified");
        let mut m = BTreeMap::new();
        // Shards key by TAXONOMY id (`UNIFIED.30-1`); the golden keys by SCENARIO
        // (`UNIFIED.guided_json_named_tool.qwen3`). The inputs shard carries both, so
        // it is the bridge — without it every cell reads NO-DATA while the capture
        // sits right there, which is how this first went wrong.
        let mut by_tax: BTreeMap<(String, String), String> = BTreeMap::new();
        for entry in glob_yaml(&froot.join("inputs")) {
            if let Ok(doc) = serde_yaml::from_str::<InputDoc>(
                &std::fs::read_to_string(&entry).unwrap_or_default(),
            ) {
                for (cid, c) in doc.cases {
                    by_tax.insert((doc.family.clone(), cid), c.scenario);
                }
            }
        }
        if let Some(d) = common::version_dirs_ascending(&froot, "vllm_rust-").pop() {
            for entry in glob_yaml(&d) {
                if let Ok(doc) = serde_yaml::from_str::<CaptureDoc>(
                    &std::fs::read_to_string(&entry).unwrap_or_default(),
                ) {
                    for (cid, c) in doc.cases {
                        if let Some(sc) = by_tax.get(&(doc.family.clone(), cid)) {
                            m.insert(
                                (doc.family.clone(), format!("UNIFIED.{sc}.{}", doc.family)),
                                c.assembled,
                            );
                        }
                    }
                }
            }
        }
        m
    };
    let dir = common::ensure_unified_golden();
    let mut files: Vec<GoldenFile> = Vec::new();
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) != Some("yaml") {
            continue;
        }
        let text = std::fs::read_to_string(&path).unwrap();
        files.push(
            serde_yaml::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display())),
        );
    }
    files.sort_by(|a, b| a.family.cmp(&b.family));

    let mut rows = String::new();
    let mut dynamo_red = 0usize;
    let mut glm47_red = 0usize;
    let mut dynamo_expectation_errors = Vec::new();
    let mut vllm_red = 0usize;
    let mut total = 0usize;
    // Machine-readable feed for the CONFORMANCE_v2.html generator (Python reads this).
    let mut json_cases: Vec<Value> = Vec::new();

    for file in &files {
        rows.push_str(&format!(
            "<tr class=fam><td colspan=4>{} &nbsp; <span class=sub>reasoning=`{}` · tool=`{}`</span></td></tr>",
            esc(&file.family),
            parsers_for(&file.family).0,
            parsers_for(&file.family).1,
        ));
        for (id, case) in &file.cases {
            total += 1;

            // Dynamo: live.
            let got = dynamo_events(&file.family, &case.input, &case.init);
            let dclass = common::classify_unified_events(&file.family, &case.golden, &got);
            let current_expected = case.expect.get("dynamo_current");
            if let Err(reason) = common::validate_current_dynamo_expectation(
                current_expected.map(|expected| expected.verdict.as_str()),
                current_expected.and_then(|expected| expected.class.as_deref()),
                current_expected.and_then(|expected| expected.note.as_deref()),
                dclass,
            ) {
                dynamo_expectation_errors.push(format!("{id}: {reason}"));
            }
            eprintln!(
                "{id:44} dynamo={dclass:6} :: {}",
                got.iter().map(Ev::render).collect::<Vec<_>>().join("  |  ")
            );
            let chunk_feed: Vec<Value> = dynamo_chunks(&file.family, &case.input, &case.init)
                .into_iter()
                .map(|r| json!({"delta_text": r.delta_text, "dynamo": r.deltas}))
                .collect();

            let scenario = id
                .strip_prefix("UNIFIED.")
                .and_then(|s| s.strip_suffix(&format!(".{}", file.family)))
                .unwrap_or(id.as_str());
            let vx = case.expect.get("vllm");
            json_cases.push(serde_json::json!({
                "id": id,
                "family": file.family,
                "scenario": scenario,
                "description": case.description,
                "policy": case.policy,
                "init": case.init.applied(),
                "finish_reason": case.finish_reason.clone().unwrap_or_else(|| "stop".to_string()),
                "input": case.input,
                "tools": common::unified_tool_schemas(),
                "golden": case.golden,
                "dynamo": got,
                "dynamo_verdict": dclass,
                "vllm_verdict": vx.map(|e| if e.verdict == "match" { "MATCH".to_string() } else { e.class.clone().unwrap_or_else(|| "DIVERGE".into()) }),
                "vllm_note": vx.and_then(|e| e.note.clone()),
                "policy_tags": case.policy,
                "chunks": chunk_feed,
            }));
            let policy = if case.policy.is_empty() {
                String::new()
            } else {
                format!("<div class=pol>policy: {}</div>", case.policy.join(", "))
            };
            let dtodo = if dclass == "MATCH" {
                String::new()
            } else {
                dynamo_red += 1;
                if file.family == "glm47" {
                    glm47_red += 1;
                }
                format!("<hr><div class=todo>{}</div>", esc(todo_for(dclass)))
            };
            let dnote = current_expected
                .and_then(|expected| expected.note.as_ref())
                .map(|note| format!("<hr><div class=note>{}</div>", esc(note)))
                .unwrap_or_default();
            let dcell = cell(
                "Dynamo today (native unified, LIVE)",
                &case.input,
                &case.golden,
                &events_html(&got),
                if dclass == "MATCH" {
                    "match"
                } else {
                    "diverge"
                },
                dclass,
                &format!("{policy}{dnote}{dtodo}"),
            );

            // vLLM: expected (from golden expect.vllm).
            let vx = case.expect.get("vllm").cloned().unwrap_or(Expect {
                verdict: "match".into(),
                class: None,
                note: None,
            });
            let vclass = if vx.verdict == "match" {
                "MATCH".to_string()
            } else {
                vx.class.clone().unwrap_or_else(|| "DIVERGE".into())
            };
            if vclass != "MATCH" {
                vllm_red += 1;
            }
            let vnote = vx
                .note
                .map(|n| format!("<hr><div class=note>{}</div>", esc(&n)))
                .unwrap_or_default();
            let vlive = vllm_live.get(&(file.family.clone(), id.clone()));
            let (vgot_html, vverdict, vclass_final) = match vlive {
                Some(ev) => {
                    let matches = ev == &case.golden;
                    (
                        events_html(ev),
                        if matches { "match" } else { "diverge" },
                        if matches {
                            "MATCH".to_string()
                        } else {
                            "DIVERGE".to_string()
                        },
                    )
                }
                // No capture for this case: say so. Do NOT fall back to the authored
                // expectation dressed up as a result — that is how this column spent
                // its life claiming MATCH with nothing behind it.
                None => (
                    "<i>(no vLLM capture for this case)</i>".to_string(),
                    "diverge",
                    "NO-DATA".to_string(),
                ),
            };
            let vcell = cell(
                "vLLM Rust 0.25.1 (LIVE)",
                &case.input,
                &case.golden,
                &vgot_html,
                vverdict,
                &vclass_final,
                &vnote,
            );

            let gtip = format!(
                "<b>GOLDEN</b> (authored oracle){}<hr><b>input</b><pre>{}</pre>{}",
                policy,
                esc(&case.input),
                events_html(&case.golden),
            );

            rows.push_str(&format!(
                "<tr><td class=case>{}<div class=desc>{}</div></td>\
                 <td class='c gold'>golden<div class=tip>{}</div></td>\
                 <td class='c {}'>{}<div class=tip>{}</div></td>\
                 <td class='c {}'>{}<div class=tip>{}</div></td></tr>",
                esc(id),
                esc(&case.description),
                gtip,
                css(&vcell.verdict),
                label(&vcell.verdict),
                vcell.tip,
                css(&dcell.verdict),
                label(&dcell.verdict),
                dcell.tip,
            ));
        }
    }

    // Machine-readable feed consumed by generate_conformance_table.py's Unified tab.
    // YAML so it reads like the rest of the conformance fixture corpus. conformance/unified/
    // is the gitignored build tree — create it (a fresh checkout won't have it; the
    // committed data is the self-contained family and capture YAML under conformance/fixtures-unified-v2/).
    let yaml_out = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("unified/unified_results.yaml");
    assert_eq!(
        common::dynamo_capture_provenance(Some(capture_provenance["label"].as_str().unwrap())),
        capture_provenance,
        "capture source changed during the run"
    );
    std::fs::create_dir_all(yaml_out.parent().unwrap()).unwrap();
    let feed = serde_json::json!({
        "schema": "unified-results/v1",
        "capture_provenance": capture_provenance,
        "note": "GOLDEN = authored oracle. dynamo = LIVE native UnifiedParser output. vllm = documented expectation when a live capture is unavailable.",
        "cases": json_cases,
    });
    std::fs::write(&yaml_out, serde_yaml::to_string(&feed).unwrap()).unwrap();
    eprintln!(
        "wrote {} ({total} cases, dynamo_red={dynamo_red}, vllm_red={vllm_red})",
        yaml_out.display()
    );

    // Sanity: the harness computes REAL failures, not a strawman. vLLM's
    // documented-expectation column supplies that signal (`vllm_red` is large
    // and asserted elsewhere via the rendered legend), so this only needs to
    // pin Dynamo's own invariant.
    assert!(total >= 14, "expected the seed corpus");
    assert!(
        dynamo_expectation_errors.is_empty(),
        "Dynamo results disagree with their documented expectations:\n{}",
        dynamo_expectation_errors.join("\n")
    );
    assert_eq!(glm47_red, 0, "GLM must have zero red rows");
}

#[test]
fn documented_current_dynamo_divergence_requires_exact_class_and_note() {
    assert!(
        common::validate_current_dynamo_expectation(
            Some("diverge"),
            Some("LOSS"),
            Some("drops call"),
            "LOSS"
        )
        .is_ok()
    );
    assert!(
        common::validate_current_dynamo_expectation(
            Some("diverge"),
            Some("LOSS"),
            Some("drops call"),
            "LEAK"
        )
        .is_err()
    );
    assert!(
        common::validate_current_dynamo_expectation(
            Some("diverge"),
            Some("LOSS"),
            Some(""),
            "LOSS"
        )
        .is_err()
    );
    assert!(
        common::validate_current_dynamo_expectation(
            Some("diverge"),
            Some("LOSS"),
            Some("drops call"),
            "MATCH"
        )
        .is_err()
    );
    assert!(common::validate_current_dynamo_expectation(None, None, None, "MATCH").is_ok());
    assert!(common::validate_current_dynamo_expectation(None, None, None, "LOSS").is_err());
}

/// One committed `dynamo_v2-<ver>/<family>/<key>.yaml` capture.
#[derive(Deserialize)]
struct CaptureDoc {
    family: String,
    cases: BTreeMap<String, CaptureCase>,
}

#[derive(Deserialize)]
struct CaptureCase {
    #[serde(default)]
    assembled: Vec<Ev>,
    #[serde(default)]
    chunks: Vec<CaptureChunk>,
}

#[derive(Deserialize)]
struct CaptureChunk {
    #[serde(default)]
    expected: Vec<Value>,
}

/// The scenario slug for each committed case key, read from the `inputs/` shard
/// (the numbered `UNIFIED.<group>-<sub>` key lives only in the Python taxonomy).
#[derive(Deserialize)]
struct InputDoc {
    family: String,
    cases: BTreeMap<String, InputCase>,
}

#[derive(Deserialize)]
struct InputCase {
    #[serde(default)]
    scenario: String,
    #[serde(default)]
    input: String,
    /// The guard must re-run each case under the SAME configuration the shard was
    /// captured with; re-running everything under the default would report false
    /// drift for every prefilled / guided-JSON case.
    #[serde(default)]
    init: Init,
}

/// GUARD: the COMMITTED Dynamo capture must equal what the parsers produce NOW.
///
/// The Unified tab is rendered by Python from the committed shard — it never runs
/// the Rust parsers. So changing a parser without re-capturing leaves the page
/// showing the OLD behavior while every Rust test still passes, and the two
/// disagree silently. (That is exactly what happened when the unified parser
/// landed: `unified_parity` was 33/33 green while the page still drew the split.)
///
/// This closes that gap: touch a parser, and the capture must be regenerated.
#[test]
fn committed_dynamo_capture_matches_the_live_parsers() {
    let root = common::ensure_fixtures().join("unified");
    validate_committed_dynamo_capture(&root);
}

fn validate_committed_dynamo_capture(root: &std::path::Path) {
    if !root.join("inputs").is_dir() {
        panic!(
            "no committed unified fixtures under {} — extract them first",
            root.display()
        );
    }
    // Select the parser identity before resolving its effective per-case owners.
    // A release's sparse patches and a source's complete snapshot differ here.
    let capture_dir = common::version_dirs_ascending_with_current(
        root,
        "dynamo_v2-",
        common::UNIFIED_DYNAMO_V2_CURRENT_CAPTURE,
    )
    .pop()
    .expect("no committed dynamo_v2-<ver> capture dir");

    validate_selected_dynamo_capture(root, &capture_dir);
}

fn validate_selected_dynamo_capture(root: &std::path::Path, capture_dir: &std::path::Path) {
    let input_dirs = shared_overlay_dirs(root, "inputs");
    let validation = common::capture_stimulus_command()
        .arg("--validate-current")
        .arg(capture_dir)
        .args(["--format", "json"])
        .arg("--inputs")
        .args(&input_dirs)
        .output()
        .expect("validate current capture stimulus");
    assert!(
        validation.status.success(),
        "current capture stimulus validation failed: {}",
        String::from_utf8_lossy(&validation.stderr)
    );
    let captures: Vec<CaptureDoc> =
        serde_json::from_slice(&validation.stdout).expect("validated effective capture records");

    // key -> (family, scenario, input, init), from the base inputs shard plus
    // PR-qualified sparse overlays. New cases must carry their input metadata in
    // the same overlay as the capture, rather than making the released shard mutable.
    let mut meta: BTreeMap<(String, String), (String, String, Init)> = BTreeMap::new();
    for input_dir in input_dirs {
        for entry in glob_yaml(&input_dir) {
            let doc: InputDoc = serde_yaml::from_str(&std::fs::read_to_string(&entry).unwrap())
                .unwrap_or_else(|e| panic!("{}: {e}", entry.display()));
            for (key, case) in doc.cases {
                // A sparse overlay can rename a case while preserving its scenario.
                // The newer key replaces the released key for capture validation.
                meta.retain(|(family, _), (scenario, _, _)| {
                    family != &doc.family || scenario != &case.scenario
                });
                meta.insert(
                    (doc.family.clone(), key),
                    (case.scenario, case.input, case.init),
                );
            }
        }
    }

    let mut stale: Vec<String> = Vec::new();
    let mut checked = 0usize;
    let capture_keys: std::collections::BTreeSet<(String, String)> = captures
        .iter()
        .flat_map(|doc| {
            doc.cases
                .keys()
                .map(|key| (doc.family.clone(), key.clone()))
        })
        .collect();
    for key in meta.keys() {
        if !capture_keys.contains(key) {
            stale.push(format!(
                "{} [{}] has no current Dynamo capture",
                key.0, key.1
            ));
        }
    }
    for doc in captures {
        for (key, committed) in doc.cases {
            let Some((scenario, input, init)) = meta.get(&(doc.family.clone(), key.clone())) else {
                stale.push(format!(
                    "{} [{key}] has no input metadata in inputs or its PR overlays",
                    doc.family
                ));
                continue;
            };
            checked += 1;
            let id = format!("UNIFIED.{scenario}.{}", doc.family);

            let live_assembled = dynamo_events(&doc.family, input, init);
            if live_assembled != committed.assembled {
                stale.push(format!(
                    "{id} [{key}] assembled\n    committed: {}\n         live: {}",
                    committed
                        .assembled
                        .iter()
                        .map(Ev::render)
                        .collect::<Vec<_>>()
                        .join("  |  "),
                    live_assembled
                        .iter()
                        .map(Ev::render)
                        .collect::<Vec<_>>()
                        .join("  |  "),
                ));
                continue;
            }
            // The page assembles the Dynamo column from these per-chunk deltas, so
            // they have to be current too — not just the assembled list.
            let live_chunks: Vec<Vec<Value>> = dynamo_chunks(&doc.family, input, init)
                .into_iter()
                .map(|r| r.deltas)
                .collect();
            let committed_chunks: Vec<Vec<Value>> =
                committed.chunks.into_iter().map(|c| c.expected).collect();
            if live_chunks != committed_chunks {
                stale.push(format!("{id} [{key}] per-chunk deltas differ"));
            }
        }
    }

    assert!(checked > 0, "no committed capture cases were compared");
    assert!(
        stale.is_empty(),
        "{} of {checked} committed Dynamo capture cases are STALE — the HTML tab will \
         show the old parser behavior. Regenerate:\n  \
         cargo test -p dynamo-conformance-fixtures-v2 --test unified_render\n  \
         python3 conformance/utils/src/explode_unified_fixtures.py\n  \
         python3 conformance/utils/src/package_fixtures.py\n\n{}",
        stale.len(),
        stale.join("\n\n"),
    );
}

#[test]
fn release_overlay_records_reach_live_guard() {
    let root = std::env::temp_dir().join(format!("dynamo-release-overlay-{}", std::process::id()));
    let base = root.join("dynamo_v2-0.6.0");
    let patch = root.join("dynamo_v2-0.6.0.patch10");
    let write = |directory: &std::path::Path, key: &str, doc: &Value| {
        std::fs::create_dir_all(directory.join("gemma4")).unwrap();
        std::fs::write(
            directory.join(format!("gemma4/{key}.yaml")),
            serde_json::to_vec(doc).unwrap(),
        )
        .unwrap();
    };
    let mut captures = BTreeMap::new();
    for key in ["old", "retained", "added"] {
        let chunks = dynamo_chunks("gemma4", key, &Init::default());
        let input_chunks: Vec<Value> = chunks
            .iter()
            .map(|row| json!({"delta_text":row.delta_text}))
            .collect();
        let stimulus = json!({"input":key,
            "init":{"starting_state":"None","tool_output_mode":"Native","named_tool":null},
            "finish_reason":"stop","tools":common::unified_tool_schemas(),"chunks":input_chunks});
        let mut input = stimulus.clone();
        input["scenario"] = json!(key);
        write(
            &root.join("inputs"),
            key,
            &json!({"family":"gemma4","cases":{key:input}}),
        );
        let output_chunks: Vec<Value> = chunks
            .into_iter()
            .map(|row| json!({"expected":row.deltas}))
            .collect();
        captures.insert(key, json!({"family":"gemma4","cases":{key:{
            "capture_input":stimulus,"assembled":[{"kind":"text","text":key}],"chunks":output_chunks}}}));
    }
    let invalid = json!({"family":"gemma4","cases":{"old":{"error":"obsolete"}}});
    write(&base, "old", &invalid);
    write(&base, "retained", &captures["retained"]);
    write(&patch, "old", &captures["old"]);
    write(&patch, "added", &captures["added"]);
    validate_selected_dynamo_capture(&root, &base);

    // The same invalid base record becomes authoritative if its patch is absent.
    std::fs::remove_file(patch.join("gemma4/old.yaml")).unwrap();
    assert!(std::panic::catch_unwind(|| validate_selected_dynamo_capture(&root, &base)).is_err());
    write(&patch, "old", &captures["old"]);
    let mut wrong = captures["added"].clone();
    wrong["cases"]["added"]["assembled"] = json!([{"kind":"text","text":"wrong"}]);
    write(&patch, "added", &wrong);
    let failure = std::panic::catch_unwind(|| validate_selected_dynamo_capture(&root, &base))
        .expect_err("the live comparison must read the patch-only record");
    assert!(
        failure
            .downcast_ref::<String>()
            .unwrap()
            .contains("assembled")
    );
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn current_source_snapshot_reaches_live_guard_and_binds_tools() {
    if std::env::var_os("DYNAMO_SOURCE_SNAPSHOT_TEST_CHILD").is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "current_source_snapshot_reaches_live_guard_and_binds_tools",
                "--nocapture",
            ])
            .env("DYNAMO_SOURCE_SNAPSHOT_TEST_CHILD", "1")
            .env("CONFORMANCE_DYNAMO_V2_LABEL", "current")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }
    let root = std::env::temp_dir().join(format!("dynamo-source-snapshot-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let provenance = common::dynamo_capture_provenance(Some("current"));
    assert!(provenance["label"].as_str().unwrap().contains("+source."));
    let base = root.join(format!(
        "dynamo_v2-{}",
        provenance["label"].as_str().unwrap()
    ));
    let patch = root.join(format!(
        "{}.patch1",
        base.file_name().unwrap().to_str().unwrap()
    ));
    let make = |key: &str, text: &str, assembled: Value| {
        let init = Init::default();
        let chunks = dynamo_chunks("gemma4", text, &init);
        let input_chunks: Vec<Value> = chunks
            .iter()
            .map(|row| json!({"delta_text": row.delta_text}))
            .collect();
        let stimulus = json!({"input": text, "init": {"starting_state":"None", "tool_output_mode":"Native", "named_tool":null},
            "finish_reason":"stop", "tools":common::unified_tool_schemas(), "chunks":input_chunks});
        let input = json!({"family":"gemma4", "cases":{key: {
            "scenario":key, "input":text, "init":stimulus["init"], "tools":stimulus["tools"],
            "finish_reason":"stop", "chunks":stimulus["chunks"]}}});
        let output_chunks: Vec<Value> = chunks
            .into_iter()
            .map(|row| json!({"expected":row.deltas}))
            .collect();
        let capture = json!({"family":"gemma4", "capture_provenance":provenance, "cases":{key:{
            "capture_input":stimulus, "assembled":assembled, "chunks":output_chunks}}});
        (input, capture)
    };
    let write = |directory: &std::path::Path, key: &str, doc: &Value| {
        std::fs::create_dir_all(directory.join("gemma4")).unwrap();
        std::fs::write(
            directory.join(format!("gemma4/{key}.yaml")),
            serde_json::to_vec(doc).unwrap(),
        )
        .unwrap();
    };
    let (old_input, old_capture) = make(
        "old",
        "old text",
        json!([{"kind":"text","text":"old text"}]),
    );
    write(&root.join("inputs"), "old", &old_input);
    write(&base, "old", &old_capture);
    validate_committed_dynamo_capture(&root);
    let (new_input, new_capture) = make(
        "added",
        "<|tool_call>call:f{x:<|\"|>1<|\"|>}<tool_call|>",
        json!([{"kind":"tool_call","name":"f","arguments":{"x":"1"}}]),
    );
    write(&root.join("inputs"), "added", &new_input);
    write(&patch, "old", &old_capture);
    write(&patch, "added", &new_capture);
    std::fs::write(
        patch.join("capture-snapshot.json"),
        r#"{"schema_version":1,"records":["gemma4/old.yaml","gemma4/added.yaml"]}"#,
    )
    .unwrap();
    validate_committed_dynamo_capture(&root);
    std::fs::remove_file(root.join("inputs/gemma4/old.yaml")).unwrap();
    std::fs::remove_file(patch.join("gemma4/old.yaml")).unwrap();
    std::fs::write(
        patch.join("capture-snapshot.json"),
        r#"{"schema_version":1,"records":["gemma4/added.yaml"]}"#,
    )
    .unwrap();
    validate_committed_dynamo_capture(&root);
    let mut wrong = new_capture.clone();
    let terminal = wrong["cases"]["added"]["chunks"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .flat_map(|chunk| chunk["expected"].as_array_mut().unwrap())
        .find(|delta| delta["complete"] == true)
        .expect("the captured call must contain a completion delta");
    terminal["complete"] = json!(false);
    write(&patch, "added", &wrong);
    let failure = std::panic::catch_unwind(|| validate_committed_dynamo_capture(&root))
        .expect_err("changing only completion must fail the per-chunk comparison");
    assert!(
        failure
            .downcast_ref::<String>()
            .unwrap()
            .contains("per-chunk deltas differ")
    );
    let mut wrong = new_capture.clone();
    wrong["cases"]["added"]["capture_input"]["tools"] = json!([]);
    write(&patch, "added", &wrong);
    assert!(std::panic::catch_unwind(|| validate_committed_dynamo_capture(&root)).is_err());
    write(&patch, "added", &new_capture);
    let mut wrong_input = new_input;
    wrong_input["cases"]["added"]["tools"] = json!([]);
    write(&root.join("inputs"), "added", &wrong_input);
    assert!(std::panic::catch_unwind(|| validate_committed_dynamo_capture(&root)).is_err());
    std::fs::remove_dir_all(&root).unwrap();
}

/// Return the released shared shard followed by its PR-qualified sparse overlays.
/// The overlays are ordered by PR number and then patch number so a later patch is
/// authoritative if a branch intentionally extends its own metadata.
fn shared_overlay_dirs(root: &std::path::Path, base: &str) -> Vec<PathBuf> {
    let mut overlays: Vec<(u64, u64, PathBuf)> = std::fs::read_dir(root)
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .filter_map(|path| {
            let name = path.file_name()?.to_str()?;
            let rest = name.strip_prefix(&format!("{base}+pr"))?;
            let (pr, patch) = rest.split_once(".patch")?;
            Some((pr.parse().ok()?, patch.parse().ok()?, path))
        })
        .collect();
    overlays.sort_by_key(|(pr, patch, _)| (*pr, *patch));

    let mut dirs = vec![root.join(base)];
    dirs.extend(overlays.into_iter().map(|(_, _, path)| path));
    dirs
}

#[test]
fn release_qualified_capture_order_preserves_current_and_released_owners() {
    let prefix = "dynamo_v2-";
    let older_release = common::version_capture_sort_key("dynamo_v2-0.3.2", prefix).unwrap();
    let current_release = common::version_capture_sort_key("dynamo_v2-0.4.0", prefix).unwrap();
    assert!(older_release < current_release);
    assert!(common::version_capture_sort_key("dynamo_v2-0.3.3.patch1", prefix).is_none());
}

#[test]
fn shared_input_overlays_are_folded_after_the_released_shard() {
    let root = std::env::temp_dir().join(format!(
        "unified-render-overlay-order-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(root.join("inputs")).unwrap();
    std::fs::create_dir_all(root.join("inputs+pr166.patch10")).unwrap();
    std::fs::create_dir_all(root.join("inputs+pr166.patch2")).unwrap();
    std::fs::create_dir_all(root.join("inputs+pr167.patch1")).unwrap();

    let names: Vec<String> = shared_overlay_dirs(&root, "inputs")
        .into_iter()
        .map(|path| path.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    assert_eq!(
        names,
        [
            "inputs",
            "inputs+pr166.patch2",
            "inputs+pr166.patch10",
            "inputs+pr167.patch1",
        ]
    );
    std::fs::remove_dir_all(root).unwrap();
}

fn glob_yaml(dir: &std::path::Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for fam in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let p = fam.path();
        if !p.is_dir() {
            continue;
        }
        for f in std::fs::read_dir(&p).into_iter().flatten().flatten() {
            let fp = f.path();
            if fp.extension().and_then(|e| e.to_str()) == Some("yaml") {
                out.push(fp);
            }
        }
    }
    out.sort();
    out
}

fn label(v: &str) -> String {
    if v == "MATCH" {
        "✓".to_string()
    } else {
        format!("✗ {v}")
    }
}
fn css(v: &str) -> &'static str {
    if v == "MATCH" { "MATCH" } else { "RED" }
}
