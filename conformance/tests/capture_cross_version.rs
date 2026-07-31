// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Run an OLDER build's unified parser against the CURRENT corpus.
//!
//! Every committed `dynamo_v2-<ver>/` shard holds only the cases that existed when it
//! was taken, so a case added later shows "no data" on the older column and the table
//! cannot say what the old parser WOULD have done with it. That is the interesting
//! question for a behavior change: not "did the old build pass the old tests" but
//! "what does the old build do with the new ones".
//!
//! Usage — check this file out into a worktree at the OLD commit, then run it there
//! pointed at the CURRENT corpus:
//!
//! ```text
//! git worktree add --detach /tmp/old <old-sha>
//! cp conformance/tests/capture_cross_version.rs /tmp/old/conformance/tests/
//! cd /tmp/old && \
//!   XVER_INPUTS=<repo>/conformance/unified/inputs \
//!   XVER_OUT=<repo>/conformance/unified/dynamo_v2-<label> \
//!   XVER_LABEL=<label> \
//!   cargo test -p dynamo-conformance-fixtures-v2 --test capture_cross_version -- --nocapture
//! ```
//!
//! Pick `<label>` as `<version>+<tag>` (e.g. `0.1.24+pre163`). The table sorts a `+tag`
//! capture BEFORE the plain release it qualifies, so the released build stays the
//! reference and the tagged one is the historical column.
//!
//! It deliberately uses only `push`/`finish` — the smallest surface every build of the
//! trait has had — so it compiles against old trees whose parser has no `initialize` or
//! output-mode API. A build lacking those APIs runs every case in its only mode, and
//! that IS the finding for a request-mode case.
//!
//! No-op unless `XVER_INPUTS` is set, so it costs nothing in a normal test run.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use dynamo_parsers::{ReasoningParser, ReasoningParserType};
use dynamo_parsers_v2::{
    Tool, UnifiedEvent, assemble, create_tool_parser_for_family, create_unified_parser_for_family,
};
use serde_json::{Value, json};

/// Corpus family -> (v1 reasoning parser, v2 tool parser) for the SPLIT path.
/// Mirrors `unified_render::parsers_for`.
fn parsers_for(family: &str) -> Option<(&'static str, &'static str)> {
    match family {
        "gemma4" => Some(("gemma4", "gemma4")),
        "qwen3" => Some(("qwen3", "qwen3_coder")),
        "kimi_k2" => Some(("kimi_k25", "kimi_k2")),
        // Unknown to this build: reported, not guessed.
        _ => None,
    }
}

fn tool_deltas(res: &dynamo_parsers_v2::ToolParseResult, out: &mut Vec<Value>) {
    if !res.normal_text.is_empty() {
        out.push(json!({"kind": "text", "text": res.normal_text}));
    }
    for c in &res.calls {
        out.push(json!({"kind": "tool_call", "name": c.name, "arguments": c.arguments}));
    }
}

/// The SPLIT path: v1 reasoning over the whole stream, then the v2 tool parser on the
/// leftover. This is what Dynamo still serves for families with no native unified
/// parser, and what `unified_render::dynamo_chunks` records for them — so a
/// cross-version capture has to reproduce it or those rows come out empty and the
/// table reads "this build produced nothing" for gemma4/kimi_k2.
fn split_path_chunks(family: &str, input: &str) -> Option<Vec<Vec<Value>>> {
    let (reasoning_name, tool_family) = parsers_for(family)?;
    let mut rp = ReasoningParserType::get_reasoning_parser_from_name(reasoning_name);
    let mut tp = create_tool_parser_for_family(tool_family, &tools()).ok()?;

    let mut rows = Vec::new();
    for chunk in chunk_input(input) {
        let mut deltas: Vec<Value> = Vec::new();
        let rr = rp.parse_reasoning_streaming_incremental(&chunk, &[]);
        if !rr.reasoning_text.is_empty() {
            deltas.push(json!({"kind": "reasoning", "text": rr.reasoning_text}));
        }
        if !rr.normal_text.is_empty() {
            let tr = tp.push(&rr.normal_text).unwrap_or_default();
            tool_deltas(&tr, &mut deltas);
        }
        rows.push(deltas);
    }
    // Flush in the same order the live harness uses: reasoning tail -> tool -> finish.
    let mut tail: Vec<Value> = Vec::new();
    let rf = rp.finish_reasoning_stream();
    if !rf.reasoning_text.is_empty() {
        tail.push(json!({"kind": "reasoning", "text": rf.reasoning_text}));
    }
    if !rf.normal_text.is_empty() {
        let tr = tp.push(&rf.normal_text).unwrap_or_default();
        tool_deltas(&tr, &mut tail);
    }
    tool_deltas(&tp.finish().unwrap_or_default(), &mut tail);
    if !tail.is_empty() {
        rows.push(tail);
    }
    Some(rows)
}

/// Marker-aligned chunking, byte-for-byte `unified_render::chunk_input`. The per-chunk
/// rows are compared across columns, so a different split would surface as a parser
/// difference that is really a harness difference.
fn chunk_input(input: &str) -> Vec<String> {
    let bytes = input.as_bytes();
    let mut chunks = Vec::new();
    let mut i = 0;
    let mut text_start = 0;
    while i < bytes.len() {
        if bytes[i] == b'<' {
            if text_start < i {
                chunks.push(input[text_start..i].to_string());
            }
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

/// Identical to `unified_render::tools()`. The tool list is a parser INPUT — it is baked
/// into the emitter at construction — so a different list here would read as a version
/// difference.
fn tools() -> Vec<Tool> {
    let mk = |name: &str, key: &str| Tool {
        name: name.to_string(),
        description: None,
        parameters: serde_json::json!({"type":"object","properties":{key:{"type":"string"}}}),
        strict: None,
    };
    vec![
        mk("get_weather", "city"),
        mk("f", "x"),
        mk("g", "y"),
        mk("run", "cmd"),
    ]
}

fn ev_to_yaml(ev: &UnifiedEvent) -> serde_yaml::Value {
    serde_yaml::to_value(ev).expect("event serializes")
}

/// Fold per-chunk deltas into assembled events, mirroring the page's `_assemble_stream`:
/// consecutive same-kind text/reasoning runs concatenate, and a tool_call delta carrying
/// a name opens a call while later nameless fragments append to its argument string.
fn fold_chunks(rows: &[Vec<Value>]) -> Vec<serde_yaml::Value> {
    let mut out: Vec<Value> = Vec::new();
    let mut raw: Vec<String> = Vec::new(); // argument text per open call, by position in `out`
    for row in rows {
        for d in row {
            let kind = d.get("kind").and_then(|k| k.as_str()).unwrap_or("");
            let text = d.get("text").and_then(|t| t.as_str()).unwrap_or("");
            match kind {
                "reasoning" | "text" => {
                    let same = out
                        .last()
                        .and_then(|l| l.get("kind"))
                        .and_then(|k| k.as_str())
                        == Some(kind);
                    if same {
                        let last = out.last_mut().unwrap();
                        let joined = format!("{}{}", last["text"].as_str().unwrap_or(""), text);
                        last["text"] = Value::String(joined);
                    } else {
                        out.push(json!({"kind": kind, "text": text}));
                        raw.push(String::new());
                    }
                }
                "tool_call" => {
                    let name = d.get("name").and_then(|n| n.as_str()).unwrap_or("");
                    let args = d.get("arguments").and_then(|a| a.as_str()).unwrap_or("");
                    let open_last = out
                        .last()
                        .and_then(|l| l.get("kind"))
                        .and_then(|k| k.as_str())
                        == Some("tool_call")
                        && name.is_empty();
                    if !open_last {
                        out.push(json!({"kind": "tool_call", "name": name, "arguments": ""}));
                        raw.push(String::new());
                    }
                    if let Some(r) = raw.last_mut() {
                        r.push_str(args);
                    }
                }
                _ => {}
            }
        }
    }
    // Argument text is a JSON fragment stream; parse once at the end, and keep it as a
    // string when it never became valid JSON rather than dropping what the parser said.
    for (i, ev) in out.iter_mut().enumerate() {
        if ev.get("kind").and_then(|k| k.as_str()) == Some("tool_call") {
            let r = raw.get(i).map(String::as_str).unwrap_or("");
            ev["arguments"] = if r.trim().is_empty() {
                json!({})
            } else {
                serde_json::from_str(r).unwrap_or_else(|_| Value::String(r.to_string()))
            };
        }
    }
    out.into_iter()
        .map(|v| serde_yaml::to_value(v).expect("event"))
        .collect()
}

/// Per-chunk rows record RAW deltas, not assembled events — `arguments` stays the
/// literal fragment the parser emitted. Mirrors `unified_render::unified_delta_json`.
/// (Assembling per chunk instead produces a mapping and makes every case look changed.)
fn delta_to_yaml(d: &dynamo_parsers_v2::UnifiedDelta) -> serde_yaml::Value {
    let v = match d {
        dynamo_parsers_v2::UnifiedDelta::Reasoning { text } => {
            serde_json::json!({"kind": "reasoning", "text": text})
        }
        dynamo_parsers_v2::UnifiedDelta::Text { text } => {
            serde_json::json!({"kind": "text", "text": text})
        }
        dynamo_parsers_v2::UnifiedDelta::ToolCall(c) => {
            serde_json::json!({"kind": "tool_call", "name": c.name, "arguments": c.arguments})
        }
    };
    serde_yaml::to_value(v).expect("delta serializes")
}

#[test]
fn capture_this_build_against_the_current_corpus() {
    let Ok(inputs_root) = std::env::var("XVER_INPUTS") else {
        return; // not a cross-version run
    };
    let out_root = PathBuf::from(std::env::var("XVER_OUT").expect("XVER_OUT"));
    let label = std::env::var("XVER_LABEL").expect("XVER_LABEL");

    let mut families: Vec<PathBuf> = std::fs::read_dir(Path::new(&inputs_root))
        .expect("inputs dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
        .collect();
    families.sort();

    let mut total = 0usize;
    let mut skipped_families = Vec::new();
    for fam_dir in families {
        let family = fam_dir.file_name().unwrap().to_string_lossy().to_string();
        // Native unified parser if this build has one, else the SPLIT path — the same
        // per-family mixture the live harness records. A family with neither is
        // reported, not written as an empty dir: "no parser here" and "parser emitted
        // nothing" must not look alike on the page.
        let native = create_unified_parser_for_family(&family, &tools()).is_ok();
        if !native && parsers_for(&family).is_none() {
            skipped_families.push(family);
            continue;
        }
        let mut cases: BTreeMap<String, serde_yaml::Value> = BTreeMap::new();
        let mut files: Vec<PathBuf> = std::fs::read_dir(&fam_dir)
            .expect("family dir")
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "yaml"))
            .collect();
        files.sort();
        for fp in files {
            let doc: serde_yaml::Value =
                serde_yaml::from_str(&std::fs::read_to_string(&fp).expect("read")).expect("yaml");
            let Some(case_map) = doc.get("cases").and_then(|c| c.as_mapping()) else {
                continue;
            };
            for (cid, cdoc) in case_map {
                let cid = cid.as_str().unwrap_or_default().to_string();
                let input = cdoc.get("input").and_then(|v| v.as_str()).unwrap_or("");

                let (per_chunk, assembled): (Vec<Vec<serde_yaml::Value>>, Vec<serde_yaml::Value>) =
                    if native {
                        let mut parser = create_unified_parser_for_family(&family, &tools())
                            .expect("registered");
                        let mut deltas = Vec::new();
                        let mut rows = Vec::new();
                        for ch in chunk_input(input) {
                            // A push error is this build's honest answer for that chunk;
                            // record no deltas rather than aborting the whole capture.
                            let d = parser.push(&ch).unwrap_or_default();
                            rows.push(d.iter().map(delta_to_yaml).collect::<Vec<_>>());
                            deltas.extend(d);
                        }
                        let tail = parser.finish().unwrap_or_default();
                        if !tail.is_empty() {
                            // Its own row, as `dynamo_chunks` records it.
                            rows.push(tail.iter().map(delta_to_yaml).collect::<Vec<_>>());
                            deltas.extend(tail);
                        }
                        (rows, assemble(&deltas).iter().map(ev_to_yaml).collect())
                    } else {
                        let rows = split_path_chunks(&family, input).expect("split path");
                        // The page assembles the Dynamo column from the per-chunk deltas
                        // (`_assemble_stream`), not from this field, so folding the same
                        // rows keeps the two consistent by construction.
                        let assembled = fold_chunks(&rows);
                        let rows = rows
                            .into_iter()
                            .map(|r| {
                                r.into_iter()
                                    .map(|v| serde_yaml::to_value(v).expect("delta"))
                                    .collect::<Vec<_>>()
                            })
                            .collect();
                        (rows, assembled)
                    };

                let chunk_rows: Vec<serde_yaml::Value> = per_chunk
                    .into_iter()
                    .map(|expected| {
                        let mut m = serde_yaml::Mapping::new();
                        m.insert("expected".into(), serde_yaml::Value::Sequence(expected));
                        serde_yaml::Value::Mapping(m)
                    })
                    .collect();
                let mut cm = serde_yaml::Mapping::new();
                cm.insert("assembled".into(), serde_yaml::Value::Sequence(assembled));
                cm.insert("chunks".into(), serde_yaml::Value::Sequence(chunk_rows));
                cases.insert(cid, serde_yaml::Value::Mapping(cm));
            }
        }
        let fam_out = out_root.join(&family);
        std::fs::create_dir_all(&fam_out).expect("mkdir");
        for (cid, case) in &cases {
            let mut cw = serde_yaml::Mapping::new();
            cw.insert("dynamo_v2".into(), label.clone().into());
            let mut one = serde_yaml::Mapping::new();
            one.insert(cid.clone().into(), case.clone());
            let mut doc = serde_yaml::Mapping::new();
            doc.insert("family".into(), family.clone().into());
            doc.insert("mode".into(), "unified".into());
            doc.insert("captured_with".into(), serde_yaml::Value::Mapping(cw));
            doc.insert("cases".into(), serde_yaml::Value::Mapping(one));
            std::fs::write(
                fam_out.join(format!("{cid}.yaml")),
                serde_yaml::to_string(&serde_yaml::Value::Mapping(doc)).expect("emit"),
            )
            .expect("write");
            total += 1;
        }
        println!("[xver] {family}: {} cases", cases.len());
    }
    println!("[xver] wrote {total} case files to {}", out_root.display());
    if !skipped_families.is_empty() {
        println!("[xver] no unified parser in this build for: {skipped_families:?}");
    }
    assert!(total > 0, "captured nothing — check XVER_INPUTS layout");
}
