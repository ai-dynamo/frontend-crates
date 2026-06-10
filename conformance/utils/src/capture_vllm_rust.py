#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture fixture output through the vLLM Rust tool-parser crate.

This runs on the host. It builds a small temporary Rust binary that depends on
`vllm-tool-parser` by path from a checked-out vLLM source tree, then feeds the
fixture cases to the requested parser.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import capture_driver as cd  # noqa: E402

RUST_MAIN = r'''
use std::collections::BTreeMap;
use std::fs;
use std::io::{self, Read};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use vllm_tool_parser::{
    DeepSeekV31ToolParser, DeepSeekV32ToolParser, DeepSeekV3ToolParser, DeepSeekV4ToolParser,
    Gemma4ToolParser, Glm47MoeToolParser, HermesToolParser, KimiK2ToolParser,
    Llama3JsonToolParser, MinimaxM2ToolParser, MistralToolParser, Qwen3CoderToolParser,
    Qwen3XmlToolParser, Tool, ToolCallDelta, ToolParseResult, ToolParser,
};

#[derive(Debug, Deserialize)]
struct Input {
    mode: String,
    parser: String,
    cases: BTreeMap<String, Case>,
}

#[derive(Debug, Deserialize)]
struct Case {
    #[serde(default)]
    tools: Vec<InputTool>,
    #[serde(default)]
    chunks: Vec<Chunk>,
    model_text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct InputTool {
    name: String,
    description: Option<String>,
    #[serde(default = "empty_object")]
    parameters: Value,
    strict: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct Chunk {
    #[serde(default)]
    delta_text: String,
    finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
struct OutputChunk {
    deltas: Vec<OutputDelta>,
    normal_text: String,
}

#[derive(Debug, Serialize)]
struct OutputDelta {
    index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    arguments: Option<String>,
}

fn empty_object() -> Value {
    json!({})
}

fn make_tools(tools: Vec<InputTool>) -> Vec<Tool> {
    tools
        .into_iter()
        .map(|tool| Tool {
            name: tool.name,
            description: tool.description,
            parameters: tool.parameters,
            strict: tool.strict,
        })
        .collect()
}

fn create_parser(parser: &str, tools: &[Tool]) -> anyhow::Result<Box<dyn ToolParser>> {
    match parser {
        "deepseek_v3" => Ok(DeepSeekV3ToolParser::create(tools)?),
        "deepseek_v31" | "deepseek_v3_1" => Ok(DeepSeekV31ToolParser::create(tools)?),
        "deepseek_v32" | "deepseek_v3_2" => Ok(DeepSeekV32ToolParser::create(tools)?),
        "deepseek_v4" => Ok(DeepSeekV4ToolParser::create(tools)?),
        "gemma4" => Ok(Gemma4ToolParser::create(tools)?),
        "glm47" | "glm47_moe" => Ok(Glm47MoeToolParser::create(tools)?),
        "hermes" => Ok(HermesToolParser::create(tools)?),
        "kimi_k2" => Ok(KimiK2ToolParser::create(tools)?),
        "llama3_json" => Ok(Llama3JsonToolParser::create(tools)?),
        "minimax_m2" => Ok(MinimaxM2ToolParser::create(tools)?),
        "mistral" => Ok(MistralToolParser::create(tools)?),
        "qwen3_coder" => Ok(Qwen3CoderToolParser::create(tools)?),
        "qwen3_xml" => Ok(Qwen3XmlToolParser::create(tools)?),
        _ => anyhow::bail!("unsupported vLLM Rust parser: {parser}"),
    }
}

fn output_delta(delta: ToolCallDelta) -> OutputDelta {
    OutputDelta {
        index: delta.tool_index,
        name: delta.name,
        arguments: if delta.arguments.is_empty() {
            None
        } else {
            Some(delta.arguments)
        },
    }
}

fn output_chunk(result: ToolParseResult) -> OutputChunk {
    OutputChunk {
        deltas: result.calls.into_iter().map(output_delta).collect(),
        normal_text: result.normal_text,
    }
}

fn append_result(out: &mut ToolParseResult, mut next: ToolParseResult) {
    out.normal_text.push_str(&next.normal_text);
    out.calls.append(&mut next.calls);
}

fn assembled_call_map(result: ToolParseResult) -> Vec<Value> {
    let mut order = Vec::<usize>::new();
    let mut names = BTreeMap::<usize, String>::new();
    let mut args = BTreeMap::<usize, String>::new();
    for call in result.calls {
        if !order.contains(&call.tool_index) {
            order.push(call.tool_index);
        }
        if let Some(name) = call.name {
            names.entry(call.tool_index).or_default().push_str(&name);
        }
        args.entry(call.tool_index).or_default().push_str(&call.arguments);
    }
    order
        .into_iter()
        .map(|idx| {
            let raw_args = args.remove(&idx).unwrap_or_default();
            let parsed_args = if raw_args.is_empty() {
                json!({})
            } else {
                serde_json::from_str(&raw_args).unwrap_or_else(|_| json!(raw_args))
            };
            json!({
                "name": names.remove(&idx).unwrap_or_default(),
                "arguments": parsed_args,
            })
        })
        .collect()
}

fn run_stream(input: Input) -> anyhow::Result<Value> {
    let mut out = serde_json::Map::new();
    for (case_id, case) in input.cases {
        let result = (|| -> anyhow::Result<Value> {
            let tools = make_tools(case.tools);
            let mut parser = create_parser(&input.parser, &tools)?;
            let mut chunks = Vec::new();
            for chunk in case.chunks {
                let mut result = parser.push(&chunk.delta_text)?;
                if chunk.finish_reason.is_some() {
                    append_result(&mut result, parser.finish()?);
                }
                chunks.push(output_chunk(result));
            }
            Ok(serde_json::to_value(chunks)?)
        })();
        match result {
            Ok(value) => {
                out.insert(case_id, value);
            }
            Err(error) => {
                out.insert(case_id, json!({"error": format!("{error:#}")}));
            }
        }
    }
    Ok(Value::Object(out))
}

fn run_batch_on_stream(input: Input) -> anyhow::Result<Value> {
    let mut out = serde_json::Map::new();
    for (case_id, case) in input.cases {
        let result = (|| -> anyhow::Result<Option<Value>> {
            let tools = make_tools(case.tools);
            let mut parser = create_parser(&input.parser, &tools)?;
            let mut result = ToolParseResult::default();
            if let Some(model_text) = case.model_text {
                append_result(&mut result, parser.push(&model_text)?);
                append_result(&mut result, parser.finish()?);
                let normal_text = std::mem::take(&mut result.normal_text);
                let calls = assembled_call_map(result);
                return Ok(Some(json!({
                    "calls": calls,
                    "normal_text": normal_text,
                })));
            }
            Ok(None)
        })();
        match result {
            Ok(Some(value)) => {
                out.insert(case_id, value);
            }
            Ok(None) => {}
            Err(error) => {
                out.insert(case_id, json!({"error": format!("{error:#}")}));
            }
        }
    }
    Ok(Value::Object(out))
}

fn main() -> anyhow::Result<()> {
    let arg = std::env::args().nth(1);
    let mut text = String::new();
    if let Some(path) = arg {
        text = fs::read_to_string(path)?;
    } else {
        io::stdin().read_to_string(&mut text)?;
    }
    let input: Input = serde_json::from_str(&text)?;
    let out = match input.mode.as_str() {
        "stream" => run_stream(input)?,
        "batch-on-stream" => run_batch_on_stream(input)?,
        other => anyhow::bail!("unsupported mode: {other}"),
    };
    println!("{}", serde_json::to_string(&out)?);
    Ok(())
}
'''


def _parser_input(mode: str, parser: str, cases: dict) -> dict:
    return {"mode": mode, "parser": parser, "cases": cases}


def _write_probe_project(project_dir: Path, crate_path: Path) -> None:
    (project_dir / "src").mkdir(parents=True, exist_ok=True)
    cargo_toml = f"""
[package]
name = "vllm-rust-parser-probe"
version = "0.1.0"
edition = "2024"

[dependencies]
anyhow = "1"
serde = {{ version = "1", features = ["derive"] }}
serde_json = "1"
vllm-tool-parser = {{ path = {json.dumps(str(crate_path))} }}
"""
    (project_dir / "Cargo.toml").write_text(textwrap.dedent(cargo_toml).lstrip(), encoding="utf-8")
    (project_dir / "src/main.rs").write_text(RUST_MAIN, encoding="utf-8")


def _run_probe(source: str, payload: dict, work: str | None) -> dict:
    root = Path(source).expanduser().resolve()
    crate_path = root / "rust/src/tool-parser"
    if not crate_path.joinpath("Cargo.toml").exists():
        raise SystemExit(f"vLLM Rust source path {root} does not contain rust/src/tool-parser/Cargo.toml")
    work_root = Path(work) if work else Path(tempfile.mkdtemp(prefix="vllm_rust_probe_"))
    project_dir = work_root / "probe"
    input_path = work_root / "vllm_rust_probe_input.json"
    _write_probe_project(project_dir, crate_path)
    input_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    env = os.environ.copy()
    env.setdefault("CARGO_TARGET_DIR", "/tmp/vllm-rust-parser-probe-target")
    proc = subprocess.run(
        [
            "cargo",
            "run",
            "--offline",
            "--quiet",
            "--manifest-path",
            str(project_dir / "Cargo.toml"),
            "--",
            str(input_path),
        ],
        capture_output=True,
        env=env,
        text=True,
    )
    if proc.returncode:
        raise RuntimeError(proc.stderr[-4000:])
    return json.loads(proc.stdout)


def _load_cases(fixture: str) -> dict:
    doc = yaml.safe_load(open(fixture))
    return doc.get("cases", {})


def _run_batch(source: str, mode: str, jobs: list[dict], work: str | None) -> dict:
    fixtures = {job["fixture"]: {"cases": {}} for job in jobs}
    by_parser: dict[str, dict[str, object]] = {}
    for job_index, job in enumerate(jobs):
        parser = job["parser"]
        parser_group = by_parser.setdefault(parser, {"cases": {}, "mapping": {}})
        cases = _load_cases(job["fixture"])
        for case_id, case in cases.items():
            synthetic_id = f"{job_index}::{case_id}"
            parser_group["cases"][synthetic_id] = case
            parser_group["mapping"][synthetic_id] = (job["fixture"], case_id)

    for parser, group in by_parser.items():
        try:
            payload = _parser_input(mode, parser, group["cases"])
            captured = _run_probe(source, payload, work)
        except (RuntimeError, ValueError, KeyError) as exc:
            for fixture, _case_id in group["mapping"].values():
                fixtures[fixture] = {"error": str(exc)}
            continue
        for synthetic_id, result in captured.items():
            fixture, case_id = group["mapping"][synthetic_id]
            if "error" in fixtures[fixture]:
                continue
            fixtures[fixture]["cases"][case_id] = result
    return fixtures


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", required=True, choices=("stream", "batch-on-stream"))
    ap.add_argument("--vllm-rust-source", help="vLLM source checkout root; defaults to VLLM_RUST_SOURCE")
    ap.add_argument("--fixture")
    ap.add_argument("--parser")
    ap.add_argument("--batch", help="JSON list of {fixture, parser} jobs")
    ap.add_argument("--work", help="work dir for generated probe files")
    args = ap.parse_args()

    source = args.vllm_rust_source or os.environ.get("VLLM_RUST_SOURCE")
    if not source:
        ap.error("--vllm-rust-source or VLLM_RUST_SOURCE is required")
    version = cd._vllm_rust_source_version(source)
    if args.batch:
        fixtures = _run_batch(source, args.mode, json.loads(args.batch), args.work)
        print(json.dumps({"version": version, "fixtures": fixtures}, ensure_ascii=False))
        return
    if not args.fixture or not args.parser:
        ap.error("--fixture and --parser are required without --batch")
    payload = _parser_input(args.mode, args.parser, _load_cases(args.fixture))
    cases = _run_probe(source, payload, args.work)
    print(json.dumps({"version": version, "cases": cases}, ensure_ascii=False))


if __name__ == "__main__":
    main()
