#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture fixture output through the vLLM Rust REASONING parsers.

Sibling of capture_vllm_rust.py (tool calling). Same shape: build a temporary Rust
binary that depends on the vLLM parser crate by path from a checked-out source tree,
feed it the fixture cases, read back JSON. Crate-layout resolution is shared —
capture_driver.vllm_rust_crate_dir() picks `vllm-parser` (>= 0.25, parsers under
`vllm_parser::reasoning`) or `vllm-reasoning-parser` (< 0.25, at rust/src/reasoning-parser).

Unlike the tool parsers, `ReasoningParser::create` takes a `DynTokenizer`. That is NOT a
model dependency: `DynTokenizer` is `Arc<dyn Tokenizer>` and the delimited parsers only
call `token_to_id("<think>")` / `token_to_id("</think>")` to resolve their two delimiter
ids. So the probe supplies a synthetic tokenizer over the shared delimiter table
(capture_driver.vllm_rust_reasoning_tokens) and the capture stays hermetic — no model,
no tokenizer.json, no GPU. This is the same trick capture_reasoning.py already uses for
the Python side.

Output per case is the ASSEMBLED {reasoning_text, normal_text}, matching the reasoning
fixture schema's `expected.<impl>` block; a case whose parser raises records {error}.
"""

from __future__ import annotations

import argparse
import json
import os
import string
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

RUST_MAIN_TEMPLATE = string.Template(r'''
use std::fs;
use std::io::{self, Read};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use $reasoning_root::{
    CohereCmdReasoningParser, DeepSeekR1ReasoningParser, DeepSeekV3ReasoningParser,
    DeepSeekV4ReasoningParser, KimiK2ReasoningParser, KimiReasoningParser,
    MiniMaxM2ReasoningParser, MiniMaxM3ReasoningParser, NemotronV3ReasoningParser,
    Qwen3ReasoningParser, ReasoningParser, SeedOssReasoningParser,
};
use vllm_tokenizer::DynTokenizer;
$tokenizer_impl

#[derive(Debug, Deserialize)]
struct Input {
    mode: String,
    parser: String,
    cases: std::collections::BTreeMap<String, Case>,
}

#[derive(Debug, Deserialize)]
struct Case {
    #[serde(default)]
    chunks: Vec<String>,
    model_text: Option<String>,
}

#[derive(Debug, Serialize)]
struct Block {
    reasoning_text: String,
    normal_text: String,
}

fn create_parser(parser: &str, tok: DynTokenizer) -> anyhow::Result<Box<dyn ReasoningParser>> {
    match parser {
        "cohere_cmd" => Ok(CohereCmdReasoningParser::create(tok)?),
        "deepseek_r1" => Ok(DeepSeekR1ReasoningParser::create(tok)?),
        "deepseek_v3" => Ok(DeepSeekV3ReasoningParser::create(tok)?),
        "deepseek_v4" => Ok(DeepSeekV4ReasoningParser::create(tok)?),
        "kimi" => Ok(KimiReasoningParser::create(tok)?),
        "kimi_k2" => Ok(KimiK2ReasoningParser::create(tok)?),
        "minimax_m2" => Ok(MiniMaxM2ReasoningParser::create(tok)?),
        "minimax_m3" => Ok(MiniMaxM3ReasoningParser::create(tok)?),
        "nemotron_v3" => Ok(NemotronV3ReasoningParser::create(tok)?),
        "qwen3" => Ok(Qwen3ReasoningParser::create(tok)?),
        "seed_oss" => Ok(SeedOssReasoningParser::create(tok)?),
        _ => anyhow::bail!("unsupported vLLM Rust reasoning parser: {parser}"),
    }
}

/// Assemble the per-delta reasoning/content pieces into one block, matching the
/// reasoning fixture schema (`expected.<impl>` = {reasoning_text, normal_text}).
fn run_case(parser_name: &str, tok: DynTokenizer, deltas: &[String]) -> anyhow::Result<Block> {
    let mut parser = create_parser(parser_name, tok)?;
    let mut reasoning_text = String::new();
    let mut normal_text = String::new();
    for delta in deltas {
        let d = parser.push(delta)?;
        if let Some(r) = d.reasoning {
            reasoning_text.push_str(&r);
        }
        if let Some(c) = d.content {
            normal_text.push_str(&c);
        }
    }
    let d = parser.finish()?;
    if let Some(r) = d.reasoning {
        reasoning_text.push_str(&r);
    }
    if let Some(c) = d.content {
        normal_text.push_str(&c);
    }
    Ok(Block { reasoning_text, normal_text })
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
    let mut out = serde_json::Map::new();
    for (case_id, case) in input.cases {
        // batch = the whole text as ONE delta; stream = the recorded chunk splits.
        let deltas: Vec<String> = if input.mode == "batch" {
            match case.model_text {
                Some(t) => vec![t],
                None => continue,
            }
        } else {
            case.chunks
        };
        let tok = make_tokenizer();
        match run_case(&input.parser, tok, &deltas) {
            Ok(b) => {
                out.insert(case_id, serde_json::to_value(b)?);
            }
            Err(e) => {
                out.insert(case_id, json!({"error": format!("{e:#}")}));
            }
        }
    }
    println!("{}", serde_json::to_string(&Value::Object(out))?);
    Ok(())
}
''')

# >= 0.25: vllm-tokenizer ships TestTokenizer behind the `test-utils` feature.
_TOKENIZER_TEST_UTILS = r'''
use vllm_tokenizer::test_utils::TestTokenizer;

fn make_tokenizer() -> DynTokenizer {
    let mut t = TestTokenizer::new();
$token_registrations
    Arc::new(t)
}
'''

# < 0.25: no test-utils feature, so implement Tokenizer here. Only `token_to_id` has to
# be real — the delimited parsers resolve their two delimiters through it and never
# encode/decode during capture.
_TOKENIZER_INLINE = r'''
use vllm_tokenizer::{Result as TokResult, Tokenizer};

struct ProbeTokenizer {
    tokens: Vec<(&'static str, u32)>,
}

impl Tokenizer for ProbeTokenizer {
    fn encode(&self, _text: &str, _add_special_tokens: bool) -> TokResult<Vec<u32>> {
        Ok(Vec::new())
    }
    fn decode(&self, _token_ids: &[u32], _skip_special_tokens: bool) -> TokResult<String> {
        Ok(String::new())
    }
    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.tokens.iter().find(|(t, _)| *t == token).map(|(_, id)| *id)
    }
}

fn make_tokenizer() -> DynTokenizer {
    Arc::new(ProbeTokenizer { tokens: vec![
$token_pairs
    ] })
}
'''


def _tokenizer_impl(layout: dict) -> str:
    """The probe's tokenizer, seeded from the delimiter table shared with the Python
    reasoning capture."""
    tokens = cd.vllm_rust_reasoning_tokens()
    if layout["tokenizer_test_utils"]:
        regs = "\n".join(
            f'    t = t.with_special_token({json.dumps(tok, ensure_ascii=False)}, {tid});' for tok, tid in tokens
        )
        return string.Template(_TOKENIZER_TEST_UTILS).substitute(token_registrations=regs)
    pairs = "\n".join(f'        ({json.dumps(tok, ensure_ascii=False)}, {tid}),' for tok, tid in tokens)
    return string.Template(_TOKENIZER_INLINE).substitute(token_pairs=pairs)


def _rust_main(layout: dict) -> str:
    return RUST_MAIN_TEMPLATE.substitute(
        reasoning_root=layout["reasoning_root"],
        tokenizer_impl=_tokenizer_impl(layout),
    )


def _write_probe_project(project_dir: Path, root: Path, layout: dict) -> None:
    (project_dir / "src").mkdir(parents=True, exist_ok=True)
    tok_feat = ', features = ["test-utils"]' if layout["tokenizer_test_utils"] else ""
    reasoning_path = root / layout["reasoning_rel"]
    cargo_toml = f"""
[package]
name = "vllm-rust-reasoning-probe"
version = "0.1.0"
edition = "2024"

[dependencies]
anyhow = "1"
serde = {{ version = "1", features = ["derive"] }}
serde_json = "1"
{layout["reasoning_crate"]} = {{ path = {json.dumps(str(reasoning_path))} }}
vllm-tokenizer = {{ path = {json.dumps(str(root / "rust/src/tokenizer"))}{tok_feat} }}
"""
    (project_dir / "Cargo.toml").write_text(textwrap.dedent(cargo_toml).lstrip(), encoding="utf-8")
    (project_dir / "src/main.rs").write_text(_rust_main(layout), encoding="utf-8")


def run_probe(source: str, payload: dict, work: str | None = None) -> dict:
    """Run one {mode, parser, cases} payload through the vLLM Rust reasoning parsers."""
    _, layout = cd.vllm_rust_crate_dir(source)
    root = Path(source).expanduser().resolve()
    work_root = Path(work) if work else Path(tempfile.mkdtemp(prefix="vllm_rust_reasoning_"))
    project_dir = work_root / "probe"
    input_path = work_root / "input.json"
    _write_probe_project(project_dir, root, layout)
    input_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    env = os.environ.copy()
    # Keyed by layout so the two crate dialects don't thrash one shared target dir.
    env.setdefault("CARGO_TARGET_DIR", f"/tmp/vllm-rust-reasoning-probe-{layout['dialect']}")
    proc = subprocess.run(
        ["cargo", "run", "--offline", "--quiet", "--manifest-path",
         str(project_dir / "Cargo.toml"), "--", str(input_path)],
        capture_output=True, env=env, text=True,
    )
    if proc.returncode:
        raise RuntimeError(proc.stderr[-4000:])
    return json.loads(proc.stdout)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", required=True, choices=("batch", "stream"))
    ap.add_argument("--vllm-rust-source", help="vLLM source checkout root; defaults to VLLM_RUST_SOURCE")
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--parser", required=True)
    ap.add_argument("--work")
    args = ap.parse_args()

    source = args.vllm_rust_source or os.environ.get("VLLM_RUST_SOURCE")
    if not source:
        ap.error("--vllm-rust-source or VLLM_RUST_SOURCE is required")
    doc = yaml.safe_load(open(args.fixture)) or {}
    payload = {"mode": args.mode, "parser": args.parser, "cases": doc.get("cases", {})}
    cases = run_probe(source, payload, args.work)
    print(json.dumps({"version": cd._vllm_rust_source_version(source), "cases": cases},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
