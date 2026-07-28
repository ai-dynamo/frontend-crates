#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture the vLLM 0.25.x RUST unified parser for the Unified conformance tab.

vLLM's native Rust `UnifiedParser` (crate `vllm-parser`, module `unified`) is NOT
exposed to Python (the PyO3 bindings only bind tool parsers), so — like
capture_vllm_rust.py — this builds a small temporary Rust binary that depends on the
`vllm-parser` + `vllm-tokenizer` crates from a checked-out vLLM source tree and feeds
the cases through the right unified parser per family:

  * gemma4        -> Gemma4UnifiedParser   (NATIVE unified: one ordered pass)
  * everything else -> CombinedParser(reasoning, tool)  (mock-unified / split)

Both emit ordered `UnifiedParserEvent { Text | Reasoning | ToolCall }`, which is exactly
the golden event schema. Output JSON: {"vllm_rust_version", "results": {id: {assembled,
chunks, parser}}}. Runs on the HOST (needs cargo + the vLLM rust source).

Usage:
  python3 capture_vllm_rust_unified.py --vllm-rust-source /path/to/vllm-0.25.1/rust \
      --job job.json --out conformance/unified/vllm_rust_capture.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import yaml
import sys
import tempfile
from pathlib import Path

# family -> (unified-variant, reasoning-parser, tool-parser). gemma4 is the only native
# unified parser in 0.25.x; the rest compose reasoning + tool via CombinedParser.
FAMILY_PARSERS = {
    "gemma4": ("unified", None, None),
    "qwen3": ("combined", "Qwen3ReasoningParser", "Qwen3CoderToolParser"),
    "kimi_k2": ("combined", "KimiReasoningParser", "KimiK2ToolParser"),
}

RUST_MAIN = r'''
use std::collections::BTreeMap;
use std::io::Read;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use vllm_parser::reasoning::{Qwen3ReasoningParser, ReasoningParser};
use vllm_parser::tool::{KimiK2ToolParser, Qwen3CoderToolParser, Tool, ToolParser};
use vllm_parser::unified::{
    CombinedParser, Gemma4UnifiedParser, UnifiedParser, UnifiedParserEvent, UnifiedParserOutput,
};
use vllm_tokenizer::test_utils::TestTokenizer;
use vllm_tokenizer::DynTokenizer;

#[derive(Deserialize)]
struct Job {
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    id: String,
    family: String,
    input: String,
    #[serde(default)]
    chunks: Vec<String>,
}

#[derive(Serialize)]
struct CaseOut {
    assembled: Vec<Value>,
    chunks: Vec<Vec<Value>>,
    parser: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

fn tools() -> Vec<Tool> {
    let mk = |name: &str, key: &str| Tool {
        name: name.to_string(),
        description: None,
        parameters: json!({"type":"object","properties":{key:{"type":"string"}}}),
        strict: None,
    };
    vec![
        mk("get_weather", "city"),
        mk("f", "x"),
        mk("g", "y"),
        mk("run", "cmd"),
    ]
}

fn make_parser(family: &str) -> (Box<dyn UnifiedParser>, String) {
    match family {
        "gemma4" => {
            let tok: DynTokenizer = Arc::new(
                TestTokenizer::new()
                    .with_special_token("<|channel>", 256)
                    .with_special_token("<channel|>", 257),
            );
            (
                Gemma4UnifiedParser::create(&tools(), tok).expect("gemma4 unified create"),
                "vLLM Rust (UnifiedParser)".to_string(),
            )
        }
        "qwen3" => {
            let tok: DynTokenizer = Arc::new(
                TestTokenizer::new()
                    .with_regular_token("<think>", 256)
                    .with_regular_token("</think>", 257),
            );
            let reasoning = Qwen3ReasoningParser::create(tok).expect("qwen3 reasoning");
            let tool = Qwen3CoderToolParser::create(&tools()).expect("qwen3 tool");
            (
                Box::new(CombinedParser::new(Some(reasoning), Some(tool))),
                "vLLM Rust (CombinedParser)".to_string(),
            )
        }
        "kimi_k2" => {
            // Golden `kimi_k2` is Kimi K2.5, whose reasoning delimiter is `<think>`/`</think>`
            // (the legacy `KimiReasoningParser` uses Unicode `◁think▷` for OLD Kimi and would
            // read K2.5 `<think>` as content). vLLM Python's kimi_k2 + Dynamo's kimi_k25 both
            // use `<think>`, so compose the generic `<think>` reasoning parser with the kimi
            // tool-section parser.
            let tok: DynTokenizer = Arc::new(
                TestTokenizer::new()
                    .with_special_token("<think>", 256)
                    .with_special_token("</think>", 257),
            );
            let reasoning = Qwen3ReasoningParser::create(tok).expect("kimi (K2.5) <think> reasoning");
            let tool = KimiK2ToolParser::create(&tools()).expect("kimi tool");
            (
                Box::new(CombinedParser::new(Some(reasoning), Some(tool))),
                "vLLM Rust (CombinedParser)".to_string(),
            )
        }
        other => panic!("no vLLM Rust unified mapping for family `{other}`"),
    }
}

/// Coalesce an ordered event list into JSON [reasoning|text|tool_call], merging
/// per-`tool_index` ToolCall deltas into one call (mirrors the Dynamo feed()).
fn events_to_json(events: &[UnifiedParserEvent]) -> Vec<Value> {
    let mut out: Vec<Value> = Vec::new();
    let mut slots: BTreeMap<usize, usize> = BTreeMap::new();
    let mut names: BTreeMap<usize, String> = BTreeMap::new();
    let mut raw_args: BTreeMap<usize, String> = BTreeMap::new();
    for ev in events {
        match ev {
            UnifiedParserEvent::Reasoning(t) => out.push(json!({"kind":"reasoning","text":t})),
            UnifiedParserEvent::Text(t) => out.push(json!({"kind":"text","text":t})),
            UnifiedParserEvent::ToolCall(d) => {
                slots.entry(d.tool_index).or_insert_with(|| {
                    out.push(json!({"kind":"tool_call","name":"","arguments":{}}));
                    out.len() - 1
                });
                if let Some(n) = &d.name {
                    names.entry(d.tool_index).or_default().push_str(n);
                }
                raw_args.entry(d.tool_index).or_default().push_str(&d.arguments);
            }
        }
    }
    for (ti, pos) in &slots {
        let name = names.get(ti).cloned().unwrap_or_default();
        let raw = raw_args.get(ti).cloned().unwrap_or_default();
        let args: Value = if raw.trim().is_empty() {
            json!({})
        } else {
            serde_json::from_str(&raw).unwrap_or(Value::String(raw))
        };
        out[*pos] = json!({"kind":"tool_call","name":name,"arguments":args});
    }
    out
}

/// Raw per-chunk deltas (not coalesced), matching the Dynamo chunk feed shape.
fn deltas_to_json(events: &[UnifiedParserEvent]) -> Vec<Value> {
    events
        .iter()
        .map(|ev| match ev {
            UnifiedParserEvent::Reasoning(t) => json!({"kind":"reasoning","text":t}),
            UnifiedParserEvent::Text(t) => json!({"kind":"text","text":t}),
            UnifiedParserEvent::ToolCall(d) => {
                json!({"kind":"tool_call","name":d.name,"arguments":d.arguments})
            }
        })
        .collect()
}

fn main() {
    let mut buf = String::new();
    std::io::stdin().read_to_string(&mut buf).unwrap();
    let job: Job = serde_json::from_str(&buf).unwrap();
    let mut results: BTreeMap<String, CaseOut> = BTreeMap::new();

    for case in &job.cases {
        let (mut p, parser) = make_parser(&case.family);
        let mut error: Option<String> = None;

        // Batch: whole input -> assembled events.
        let mut out = UnifiedParserOutput::default();
        if let Err(e) = p.parse_into(&case.input, &mut out) {
            error = Some(format!("UnifiedParserError::{e:?}"));
        }
        match p.finish() {
            Ok(fin) => out.events.extend(fin.events),
            Err(e) => {
                error.get_or_insert_with(|| format!("UnifiedParserError::{e:?}"));
            }
        };
        let assembled = events_to_json(&out.events);

        // Streaming: fresh parser, per-chunk deltas.
        let (mut ps, _) = make_parser(&case.family);
        let mut chunk_rows: Vec<Vec<Value>> = Vec::new();
        for (i, ch) in case.chunks.iter().enumerate() {
            let mut co = UnifiedParserOutput::default();
            let _ = ps.parse_into(ch, &mut co);
            if i == case.chunks.len() - 1 {
                if let Ok(fin) = ps.finish() {
                    co.events.extend(fin.events);
                }
            }
            chunk_rows.push(deltas_to_json(&co.events));
        }

        results.insert(
            case.id.clone(),
            CaseOut { assembled, chunks: chunk_rows, parser, error },
        );
    }

    let feed = json!({"vllm_rust_version": "0.25.1", "results": results});
    println!("{}", serde_json::to_string(&feed).unwrap());
}
'''


def build_and_run(vllm_rust_source: Path, job_json: str) -> str:
    """Create a temp crate depending on the vLLM rust parser/tokenizer crates, build,
    and run it with the job on stdin. Returns the captured stdout JSON."""
    parser_crate = vllm_rust_source / "src/parser"
    tok_crate = vllm_rust_source / "src/tokenizer"
    for p in (parser_crate / "Cargo.toml", tok_crate / "Cargo.toml"):
        if not p.exists():
            sys.exit(f"vLLM rust crate not found: {p}")

    with tempfile.TemporaryDirectory(prefix="vllm-rust-uni-") as td:
        crate = Path(td)
        (crate / "src").mkdir()
        (crate / "Cargo.toml").write_text(f'''[package]
name = "vllm-rust-unified-capture"
version = "0.0.0"
edition = "2024"

[dependencies]
vllm-parser = {{ path = "{parser_crate}" }}
vllm-tokenizer = {{ path = "{tok_crate}", features = ["test-utils"] }}
serde = {{ version = "1", features = ["derive"] }}
serde_json = "1"

[workspace]
''')
        (crate / "src/main.rs").write_text(RUST_MAIN)
        build = subprocess.run(
            ["cargo", "build", "--release", "--quiet"],
            cwd=crate, capture_output=True, text=True)
        if build.returncode != 0:
            sys.exit(f"cargo build failed:\n{build.stderr}")
        run = subprocess.run(
            [str(crate / "target/release/vllm-rust-unified-capture")],
            input=job_json, capture_output=True, text=True)
        if run.returncode != 0:
            sys.exit(f"capture run failed:\n{run.stderr}")
        return run.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm-rust-source", required=True, type=Path)
    ap.add_argument("--job", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    out = build_and_run(args.vllm_rust_source, args.job.read_text())
    data = json.loads(out)
    args.out.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False,
                                  allow_unicode=True, width=4096))
    print(f"wrote {args.out} "
          f"(vllm_rust_version={data.get('vllm_rust_version')}, "
          f"results={len(data['results'])})")


if __name__ == "__main__":
    main()
