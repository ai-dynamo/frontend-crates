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
from dataclasses import dataclass, replace
import hashlib
import json
import os
import re
import subprocess
import yaml
import sys
import tempfile
from pathlib import Path

from capture_stimulus import capture_input, capture_peer_results, unavailable_result, validate_family_scoped_cases
from dynamo_version import git_subprocess_env
from impls import unified_starts_in_reasoning
from unified_tools import SCHEMA_PATH

@dataclass(frozen=True)
class ParserProfile:
    unified: str | None = None
    reasoning: str | None = None
    reasoning_fallbacks: tuple[str, ...] = ()
    tool: str | None = None
    markers: tuple[str, ...] = ("<think>", "</think>")
    special: bool = False
    reasoning_suffix: str = ""

    @property
    def reasoning_prompt(self):
        return self.markers[0] + self.reasoning_suffix

    @property
    def response_prompt(self):
        # CombinedParser supports persistent Response by omitting its optional
        # reasoner. K3 instead exposes the mode through its native prompt header.
        if self.unified == "KimiK3UnifiedParser":
            return self.markers[0] + "response" + self.markers[2]
        if self.unified is None and self.tool is not None:
            return ""
        return None


# One owner generates both the public factory and its prompt-tokenizer context.
FAMILY_PARSERS = {
    "gemma4": ParserProfile(unified="Gemma4UnifiedParser", markers=("<|channel>", "<channel|>"), special=True, reasoning_suffix="thought\n"),
    "kimi_k3": ParserProfile(unified="KimiK3UnifiedParser", markers=("<|open|>", "<|close|>", "<|sep|>", "<|end_of_msg|>"), special=True, reasoning_suffix="think<|sep|>"),
    "deepseek_v4": ParserProfile(reasoning="DeepSeekV4ReasoningParser", tool="DeepSeekV4ToolParser"),
    "deepseek_v41": ParserProfile(reasoning="DeepSeekV41ReasoningParser", tool="DeepSeekV41ToolParser"),
    "glm47": ParserProfile(reasoning="Glm47ReasoningParser", reasoning_fallbacks=("Glm45ReasoningParser",), tool="Glm47MoeToolParser"),
    "qwen3": ParserProfile(reasoning="Qwen3ReasoningParser", tool="Qwen3CoderToolParser"),
    "nemotron_v3": ParserProfile(reasoning="NemotronV3ReasoningParser", tool="Qwen3CoderToolParser", reasoning_suffix="\n"),
    "kimi_k2": ParserProfile(reasoning="KimiK2ReasoningParser", tool="KimiK2ToolParser"),
    "minimax_m3": ParserProfile(reasoning="MiniMaxM3ReasoningParser", tool="MinimaxM3ToolParser", markers=("<mm:think>", "</mm:think>")),
}


def _unsupported_request(case: dict) -> tuple[str, str] | None:
    """Classify request state the Rust parser API cannot receive."""
    init = case.get("init") or {}
    if init.get("structural_tag") is not None:
        return (
            "peer_request_unsupported",
            "Parser-only vLLM Rust capture does not execute structural-tag generation constraints.",
        )
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
    profile = FAMILY_PARSERS.get(case.get("family"))
    if starting_state == "Response" and (profile is None or profile.response_prompt is None):
        return (
            "vllm_rust_starting_state_unsupported",
            "This public vLLM Rust parser cannot keep reasoning markers literal after Response prefill.",
        )
    if starting_state not in {"None", "Reasoning", "Response"}:
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
use vllm_parser::reasoning::ReasoningParser;
use vllm_parser::tool::{Tool, ToolParser};
use vllm_parser::unified::{
    CombinedParser, UnifiedParser, UnifiedParserEvent, UnifiedParserOutput,
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
    tools: Vec<Tool>,
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

fn make_parser(family: &str, tools: &[Tool], starting_state: &str) -> (Box<dyn UnifiedParser>, String, DynTokenizer) {
    match family {
        __FACTORY_ARMS__
        other => panic!("no public vLLM Rust unified mapping for `{other}`"),
    }
}

fn prompt_token_ids(case: &Case, tokenizer: &DynTokenizer) -> Vec<u32> {
    let prompt = match case.family.as_str() {
        __PROMPT_ARMS__
        other => panic!("no prompt marker profile for `{other}`"),
    };
    tokenizer.encode(prompt, false).expect("encode parser prompt context")
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
            // Keep the producer's raw JSON argument tokens. Parsing them into
            // an ordinary serde_json::Value can round decimals in the adapter.
            Value::String(raw)
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
        let (mut p, parser, tokenizer) = make_parser(&case.family, &case.tools, &case.init.starting_state);
        let mut error: Option<String> = None;

        let prompt_ids = prompt_token_ids(case, &tokenizer);
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
        let (mut ps, _, _) = make_parser(&case.family, &case.tools, &case.init.starting_state);
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


def public_profiles(vllm_rust_source: Path) -> dict[str, ParserProfile]:
    api_root = vllm_rust_source / "src/parser/src"
    exports = {}
    for module in ("unified", "reasoning", "tool"):
        api = (api_root / module / "mod.rs").read_text()
        public = " ".join(re.findall(r"\bpub\s+use\b[^;]+;|\bpub\s+(?:type|struct)\s+\w+", api, re.S))
        exports[module] = set(re.findall(r"\b[A-Za-z_]\w*\b", public))
    profiles = {}
    for family, profile in FAMILY_PARSERS.items():
        if profile.reasoning is not None:
            reasoning = next((name for name in (profile.reasoning, *profile.reasoning_fallbacks)
                              if name in exports["reasoning"]), None)
            if reasoning is None:
                continue
            profile = replace(profile, reasoning=reasoning)
        if all(name is None or name in exports[module] for module, name in (
            ("unified", profile.unified), ("tool", profile.tool),
        )):
            profiles[family] = profile
    return profiles


def rust_main(version: str, unified_api: str, profiles: dict[str, ParserProfile] | None = None) -> str:
    profiles = FAMILY_PARSERS if profiles is None else profiles
    factories, prompts = [], []
    for family, profile in profiles.items():
        tokenizer = "TestTokenizer::new()" + "".join(
            f'.with_{"special" if profile.special else "regular"}_token({json.dumps(marker)}, {256 + index})'
            for index, marker in enumerate(profile.markers)
        )
        if profile.unified is not None:
            factory = f'vllm_parser::unified::{profile.unified}::create(tools, tokenizer.clone()).expect("public unified create")'
            label = "vLLM Rust (UnifiedParser)"
        else:
            factory = (f'Box::new(CombinedParser::new(if starting_state == "Response" {{ None }} else {{ Some(vllm_parser::reasoning::{profile.reasoning}::create(tokenizer.clone()).expect("public reasoning create")) }}, '
                       f'Some(vllm_parser::tool::{profile.tool}::create(tools).expect("public tool create"))))')
            label = "vLLM Rust (CombinedParser)"
        factories.append(f'{json.dumps(family)} => {{ let tokenizer: DynTokenizer = Arc::new({tokenizer}); '
                         f'let parser: Box<dyn UnifiedParser> = {factory}; (parser, {json.dumps(label)}.to_string(), tokenizer) }},')
        response = (f'"Response" => {json.dumps(profile.response_prompt)},' if profile.response_prompt is not None else "")
        default_prompt = profile.reasoning_prompt if unified_starts_in_reasoning(family) else ""
        prompts.append(f'{json.dumps(family)} => match case.init.starting_state.as_str() {{ '
                       f'"Reasoning" => {json.dumps(profile.reasoning_prompt)}, {response} '
                       f'"None" | "" => {json.dumps(default_prompt)}, other => panic!("unsupported prompt state {{other}}"), }},')
    source = RUST_MAIN.replace('"vllm_rust_version": "0.25.1"',
                               f'"vllm_rust_version": {json.dumps(version)}')
    source = source.replace("__FACTORY_ARMS__", "\n".join(factories)).replace("__PROMPT_ARMS__", "\n".join(prompts))
    # Newer releases carry token attribution with decoded reasoning text. The
    # authored corpus supplies text, so use the released unattributed constructor.
    if re.search(r"fn\s+parse_into\s*\([^)]*\bdelta:\s*DecodedText\b", unified_api):
        source = source.replace("use vllm_tokenizer::DynTokenizer;", "use vllm_tokenizer::{DecodedText, DynTokenizer};")
        source = source.replace('"kind":"reasoning","text":t', '"kind":"reasoning","text":t.text')
        source = source.replace("p.parse_into(&case.input, &mut out)", "p.parse_into(DecodedText::unattributed(case.input.clone()), &mut out)")
        source = source.replace("ps.parse_into(ch, &mut co)", "ps.parse_into(DecodedText::unattributed(ch.clone()), &mut co)")
    return source


def build_and_run(vllm_rust_source: Path, job_json: str) -> str:
    """Create a temp crate depending on the vLLM rust parser/tokenizer crates, build,
    and run it with the job on stdin. Returns the captured stdout JSON."""
    parser_crate = vllm_rust_source / "src/parser"
    tok_crate = vllm_rust_source / "src/tokenizer"
    for p in (parser_crate / "Cargo.toml", tok_crate / "Cargo.toml"):
        if not p.exists():
            sys.exit(f"vLLM rust crate not found: {p}")
    version = _vllm_rust_version(vllm_rust_source, parser_crate)
    unified_api = (parser_crate / "src/unified/mod.rs").read_text()

    with tempfile.TemporaryDirectory(prefix="vllm-rust-uni-") as td:
        crate = Path(td)
        # Cargo's build lock ends before execution; each harness must own its
        # binary even when concurrent jobs share the dependency target directory.
        package_name = f"vllm-rust-unified-capture-{crate.name}"
        (crate / "src").mkdir()
        (crate / "Cargo.toml").write_text(f'''[package]
name = "{package_name}"
version = "0.0.0"
edition = "2024"

[dependencies]
vllm-parser = {{ path = "{parser_crate}" }}
vllm-tokenizer = {{ path = "{tok_crate}", features = ["test-utils"] }}
serde = {{ version = "1", features = ["derive"] }}
serde_json = "1"

[workspace]
''')
        (crate / "src/main.rs").write_text(rust_main(version, unified_api, public_profiles(vllm_rust_source)))
        subprocess.run(["cargo", "fmt"], cwd=crate, check=True, capture_output=True, text=True)
        workspace_lock = vllm_rust_source / "Cargo.lock"
        if workspace_lock.is_file():
            (crate / "Cargo.lock").write_bytes(workspace_lock.read_bytes())
        build = subprocess.run(
            ["cargo", "build", "--quiet"],
            cwd=crate, capture_output=True, text=True)
        if build.returncode != 0:
            sys.exit(f"cargo build failed:\n{build.stderr}")
        target = Path(os.environ.get("CARGO_TARGET_DIR", str(crate / "target")))
        if not target.is_absolute():
            target = (crate / target).resolve()
        run = subprocess.run(
            [str(target / "debug" / package_name)],
            input=job_json, capture_output=True, text=True)
        if run.returncode != 0:
            sys.exit(f"capture run failed:\n{run.stderr}")
        evidence_root = os.environ.get("DYN_CAPTURE_EVIDENCE")
        if evidence_root is not None:
            evidence = Path(evidence_root) / hashlib.sha256(job_json.encode()).hexdigest()
            evidence.mkdir(parents=True, exist_ok=False)
            for name in ("Cargo.toml", "Cargo.lock", "src/main.rs"):
                (evidence / Path(name).name).write_bytes((crate / name).read_bytes())
            (evidence / "job.json").write_text(job_json)
            executable = target / "debug" / package_name
            (evidence / "execution.json").write_text(json.dumps({
                "binary_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
                "stdout_sha256": hashlib.sha256(run.stdout.encode()).hexdigest(),
                "exit_code": run.returncode,
            }, sort_keys=True))
        return run.stdout


def capture_job(vllm_rust_source, job):
    validate_family_scoped_cases(job.get("cases", []))
    feed = {}
    schema_bytes = SCHEMA_PATH.read_bytes()
    version = _vllm_rust_version(vllm_rust_source, vllm_rust_source / "src/parser")
    profiles = public_profiles(vllm_rust_source)

    unsupported = {}
    for case in job.get("cases", []):
        if case["family"] in profiles:
            continue
        request = capture_input({**case, "chunks": [{"delta_text": chunk} for chunk in case.get("chunks", [])]})
        detail = f"No vLLM Rust Unified parser is registered for {case['family']} at {version}."
        unsupported[case["id"]] = unavailable_result(
            "vllm_rust_parser_not_registered",
            detail,
            capture=request,
            observation=True,
        )

    if unsupported and all(case["family"] not in profiles for case in job.get("cases", [])):
        return {
            "vllm_rust_version": version,
            "results": unsupported,
        }

    def capture(cases):
        feed.update(json.loads(build_and_run(vllm_rust_source, json.dumps({"cases": cases}))))
        return feed["results"]

    results = capture_peer_results(
        job.get("cases", []),
        profiles,
        capture,
        tools=json.loads(schema_bytes),
        supports_finish=True,
        supports_init=True,
        supports_tools=True,
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
