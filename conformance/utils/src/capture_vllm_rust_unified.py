#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture vLLM Rust parser output for the Unified conformance tab.

vLLM's native Rust `UnifiedParser` (crate `vllm-parser`, module `unified`) is NOT
exposed to Python (the PyO3 bindings only bind tool parsers), so — like
capture_vllm_rust.py — this builds a small temporary Rust binary that depends on the
`vllm-parser` + `vllm-tokenizer` crates from a checked-out vLLM source tree and feeds
the cases through the right unified parser per family:

  * gemma4 -> Gemma4UnifiedParser (native unified: one ordered pass)
  * qwen3/kimi_k2 -> CombinedParser(reasoning, tool)

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
import re
import subprocess
import yaml
import sys
import tempfile
from pathlib import Path

from capture_stimulus import capture_input, capture_peer_results, unavailable_result, validate_family_scoped_cases
from dynamo_version import git_subprocess_env
from unified_tools import SCHEMA_PATH

# Family parser wiring for the released 0.25.1 capture.
FAMILY_PARSERS = {
    "gemma4": ("unified", None, None),
    "qwen3": ("combined", "Qwen3ReasoningParser", "Qwen3CoderToolParser"),
    "kimi_k2": ("combined", "KimiReasoningParser", "KimiK2ToolParser"),
}


def _unsupported_request(case: dict) -> tuple[str, str] | None:
    """Classify request state the Rust parser API cannot receive."""
    init = case.get("init") or {}
    if init.get("tool_output_mode", "Native") == "GuidedJson":
        return (
            "vllm_rust_guided_json_unsupported",
            "vLLM Rust UnifiedParser accepts native model text only; it has no GuidedJson/tool-choice request API.",
        )
    if init.get("named_tool") is not None:
        return (
            "vllm_rust_named_tool_unsupported",
            "vLLM Rust UnifiedParser has no named-tool request API.",
        )
    starting_state = init.get("starting_state", "None")
    if starting_state == "Response":
        return (
            "vllm_rust_starting_state_unsupported",
            "vLLM Rust UnifiedParser has no request-prefilled Response starting-state API.",
        )
    if starting_state not in {"None", "Reasoning"}:
        return (
            "vllm_rust_starting_state_unsupported",
            f"vLLM Rust UnifiedParser capture does not recognize starting_state={starting_state!r}.",
        )
    return None

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
    #[serde(default)]
    terminal_step: bool,
    #[serde(default)]
    init: Init,
}

#[derive(Default, Deserialize)]
struct Init {
    #[serde(default)]
    starting_state: String,
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
    serde_json::from_str(include_str!("unified_tools.json"))
        .expect("Unified corpus tool schemas")
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
            let tok: DynTokenizer = Arc::new(
                TestTokenizer::new()
                    .with_special_token("<think>", 256)
                    .with_special_token("</think>", 257),
            );
            let reasoning = Qwen3ReasoningParser::create(tok).expect("kimi reasoning");
            let tool = KimiK2ToolParser::create(&tools()).expect("kimi tool");
            (
                Box::new(CombinedParser::new(Some(reasoning), Some(tool))),
                "vLLM Rust (CombinedParser)".to_string(),
            )
        }
        other => panic!("no vLLM Rust unified mapping for family `{other}`"),
    }
}

fn prompt_token_ids(case: &Case) -> Vec<u32> {
    if case.init.starting_state != "Reasoning" {
        return Vec::new();
    }
    match case.family.as_str() {
        // These IDs match the TestTokenizer profiles in make_parser(). The
        // parser only needs the open-reasoning marker to detect the prompt's
        // initial state; generated text still supplies the closing marker.
        "gemma4" => vec![256],
        "qwen3" | "kimi_k2" => vec![256],
        other => panic!("no prompt marker profile for family `{other}`"),
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

        let prompt_ids = prompt_token_ids(case);
        if !prompt_ids.is_empty() {
            if let Err(e) = p.initialize(&prompt_ids) {
                error = Some(format!("UnifiedParserError::{e:?}"));
            }
        }

        // Batch: whole input -> assembled events.
        let mut out = UnifiedParserOutput::default();
        if error.is_none() {
            if let Err(e) = p.parse_into(&case.input, &mut out) {
            error = Some(format!("UnifiedParserError::{e:?}"));
            }
        }
        if error.is_none() {
            match p.finish() {
                Ok(fin) => out.events.extend(fin.events),
                Err(e) => {
                    error.get_or_insert_with(|| format!("UnifiedParserError::{e:?}"));
                }
            }
        }
        let assembled = events_to_json(&out.events);

        // Streaming: fresh parser, per-chunk deltas.
        let (mut ps, _) = make_parser(&case.family);
        if !prompt_ids.is_empty() {
            if let Err(e) = ps.initialize(&prompt_ids) {
                error.get_or_insert_with(|| format!("UnifiedParserError::{e:?}"));
            }
        }
        let mut chunk_rows: Vec<Vec<Value>> = Vec::new();
        for (i, ch) in case.chunks.iter().enumerate() {
            let mut co = UnifiedParserOutput::default();
            if let Err(e) = ps.parse_into(ch, &mut co) {
                error.get_or_insert_with(|| format!("UnifiedParserError::{e:?}"));
            }
            if !case.terminal_step && i == case.chunks.len() - 1 {
                match ps.finish() {
                    Ok(fin) => co.events.extend(fin.events),
                    Err(e) => {
                        error.get_or_insert_with(|| format!("UnifiedParserError::{e:?}"));
                    }
                }
            }
            chunk_rows.push(deltas_to_json(&co.events));
        }
        if case.terminal_step {
            match ps.finish() {
                Ok(fin) => chunk_rows.push(deltas_to_json(&fin.events)),
                Err(e) => {
                    error.get_or_insert_with(|| format!("UnifiedParserError::{e:?}"));
                    chunk_rows.push(Vec::new());
                }
            }
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


def _vllm_rust_version(vllm_rust_source: Path, parser_crate: Path) -> str:
    """Read the version from the checked-out vLLM Rust workspace, never a literal."""
    root = vllm_rust_source.parent.resolve()
    git_env = git_subprocess_env()
    top = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=root, env=git_env, capture_output=True, text=True,
    )
    if top.returncode != 0 or Path(top.stdout.strip()).resolve() != root:
        raise ValueError(f"vLLM Rust capture requires its own Git checkout: {root}")
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", str(vllm_rust_source.resolve())],
        cwd=root, env=git_env, capture_output=True, text=True,
    )
    if status.returncode != 0 or status.stdout.strip():
        raise ValueError(f"vLLM Rust capture requires clean Rust source: {vllm_rust_source}")
    tag = subprocess.run(
        ["git", "describe", "--tags", "--exact-match", "HEAD"],
        cwd=root, env=git_env, capture_output=True, text=True,
    )
    if tag.returncode == 0 and re.fullmatch(r"v?\d+\.\d+\.\d+", tag.stdout.strip()):
        return tag.stdout.strip().removeprefix("v")
    # The parser crate has its own version, and a checkout's directory can be
    # renamed. Neither establishes which vLLM release produced the observation.
    raise ValueError(
        f"vLLM Rust capture requires an exact semantic release tag at HEAD: {vllm_rust_source}"
    )


def build_and_run(vllm_rust_source: Path, job_json: str) -> str:
    """Create a temp crate depending on the vLLM rust parser/tokenizer crates, build,
    and run it with the job on stdin. Returns the captured stdout JSON."""
    parser_crate = vllm_rust_source / "src/parser"
    tok_crate = vllm_rust_source / "src/tokenizer"
    for p in (parser_crate / "Cargo.toml", tok_crate / "Cargo.toml"):
        if not p.exists():
            sys.exit(f"vLLM rust crate not found: {p}")
    version = _vllm_rust_version(vllm_rust_source, parser_crate)

    with tempfile.TemporaryDirectory(prefix="vllm-rust-uni-") as td:
        crate = Path(td)
        (crate / "src").mkdir()
        (crate / "src/unified_tools.json").write_bytes(SCHEMA_PATH.read_bytes())
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
        (crate / "src/main.rs").write_text(
            RUST_MAIN.replace('"vllm_rust_version": "0.25.1"',
                              f'"vllm_rust_version": {json.dumps(version)}')
        )
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


def capture_job(vllm_rust_source, job):
    validate_family_scoped_cases(job.get("cases", []))
    feed = {}
    schema_bytes = SCHEMA_PATH.read_bytes()
    version = _vllm_rust_version(vllm_rust_source, vllm_rust_source / "src/parser")

    unsupported = {}
    for case in job.get("cases", []):
        if case["family"] in FAMILY_PARSERS:
            continue
        request = capture_input({**case, "chunks": [{"delta_text": chunk} for chunk in case.get("chunks", [])]})
        detail = f"No vLLM Rust Unified parser is registered for {case['family']} at {version}."
        unsupported[case["id"]] = unavailable_result(
            "vllm_rust_parser_not_registered",
            detail,
            capture=request,
            observation=True,
        )

    if unsupported and all(case["family"] not in FAMILY_PARSERS for case in job.get("cases", [])):
        return {
            "vllm_rust_version": version,
            "results": unsupported,
        }

    def capture(cases):
        feed.update(json.loads(build_and_run(vllm_rust_source, json.dumps({"cases": cases}))))
        return feed["results"]

    results = capture_peer_results(
        job.get("cases", []),
        FAMILY_PARSERS,
        capture,
        tools=json.loads(schema_bytes),
        supports_finish=True,
        supports_init=True,
        unsupported_reason=_unsupported_request,
    )
    if SCHEMA_PATH.read_bytes() != schema_bytes:
        raise ValueError("tool schema changed during peer capture")
    if not feed:
        feed["vllm_rust_version"] = _vllm_rust_version(vllm_rust_source, vllm_rust_source / "src/parser")
    return {**feed, "results": {**unsupported, **results}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm-rust-source", required=True, type=Path)
    ap.add_argument("--job", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    data = capture_job(args.vllm_rust_source, json.loads(args.job.read_text()))
    args.out.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False,
                                  allow_unicode=True, width=4096))
    print(f"wrote {args.out} "
          f"(vllm_rust_version={data.get('vllm_rust_version')}, "
          f"results={len(data['results'])})")


if __name__ == "__main__":
    main()
