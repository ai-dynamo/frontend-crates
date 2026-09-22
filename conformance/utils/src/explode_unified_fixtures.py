# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explode the monolithic unified capture feeds into the per-case / per-family /
per-version fixture layout that every other conformance tab uses, so unified
fixtures package as versioned LFS shards exactly like toolcalling/reasoning.

Source (loose build tree, conformance/unified/):
  unified_results.yaml   Rust harness feed: per case input + golden + dynamo + chunks
  vllm_capture.yaml       vLLM Python assembled + per-chunk
  vllm_rust_capture.yaml   vLLM Rust
  sglang_capture.yaml      SGLang

Output (conformance/unified/, one YAML per case per family per version-dir):
  inputs/<family>/UNIFIED.<scenario>.yaml         {description, policy, chunks:[{delta_text}]}
  golden/<family>/UNIFIED.<scenario>.yaml         {captured_with:{golden}, assembled}
  dynamo_v2-<ver>/<family>/UNIFIED.<scenario>.yaml {captured_with, capture_origin, assembled, chunks:[{expected}]}
  vllm_python-<ver>/<family>/...
  vllm_rust-<ver>/<family>/...
  sglang_python-<ver>/<family>/...

Same schema shape as toolcalling/fixtures-stream-v2 (family/mode/captured_with/cases).
Run:  python3 conformance/utils/src/explode_unified_fixtures.py
"""
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dynamo_version import validate_capture_provenance  # noqa: E402
from capture_stimulus import capture_input  # noqa: E402
from unified_taxonomy import numbered_id  # noqa: E402

CONF = Path(__file__).resolve().parents[2]   # <repo>/conformance
REPO = Path(__file__).resolve().parents[3]   # <repo>
BUILD = CONF / "unified"
MODE = "unified"


def _dump(doc, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=4096)
    )


def _case_key(case_id):
    # "UNIFIED.arg_marker_in_string.gemma4" -> numbered id "UNIFIED.7-2", family, slug
    fam = case_id.rsplit(".", 1)[1]
    scenario = case_id[len("UNIFIED."):].rsplit(".", 1)[0]
    return numbered_id(scenario), fam, scenario


def _peer_cell(result):
    """Store a peer failure instead of its partial output."""
    if result.get("unavailable"):
        record = {"unavailable": result["unavailable"]}
    elif result.get("error"):
        record = {"error": result["error"]}
    else:
        record = {
            "assembled": result.get("assembled") or [],
            "chunks": [{"expected": events or []} for events in (result.get("chunks") or [])],
        }
    if "capture_input" in result:
        record["capture_input"] = result["capture_input"]
    return record


def _clear_generated_dirs():
    """Remove every generated view before materializing canonical YAML captures."""
    for sub in ("inputs", "golden"):
        shutil.rmtree(BUILD / sub, ignore_errors=True)
    for d in BUILD.glob("*-*"):
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)


def _canonical_dynamo_capture_version(provenance):
    """Use the semantic crate version as the only Dynamo capture filename."""
    return provenance["crate_version"]


def main():
    feed = yaml.safe_load((BUILD / "unified_results.yaml").read_text())
    provenance = validate_capture_provenance(REPO, feed.get("capture_provenance"))
    caps = {}
    for impl, fname in (
        ("vllm_python", "vllm_capture.yaml"),
        ("vllm_rust", "vllm_rust_capture.yaml"),
        ("sglang_python", "sglang_capture.yaml"),
    ):
        fp = BUILD / fname
        caps[impl] = yaml.safe_load(fp.read_text()) if fp.exists() else {"results": {}}

    ver = {
        "vllm_python": caps["vllm_python"].get("vllm_version") or "0.25.x",
        "vllm_rust": caps["vllm_rust"].get("vllm_rust_version") or "0.25.x",
        "sglang_python": caps["sglang_python"].get("sglang_version") or "0.5.x",
        "dynamo_v2": _canonical_dynamo_capture_version(provenance),
    }

    # A version dir is written once; accumulate cases into per-(dir, family) docs.
    docs = {}  # (dirname, family) -> {family, mode, [model_label|captured_with], cases:{}}

    def slot(dirname, family, captured_with=None, model_label=None):
        k = (dirname, family)
        if k not in docs:
            d = {"family": family, "mode": MODE}
            if model_label is not None:
                d["model_label"] = model_label
            if captured_with is not None:
                d["captured_with"] = captured_with
                if "dynamo_v2" in captured_with:
                    d["capture_origin"] = {
                        key: provenance[key]
                        for key in ("crate_version", "source_sha256", "git_commit")
                    }
            d["cases"] = {}
            docs[k] = d
        return docs[k]["cases"]

    for c in feed["cases"]:
        cid = c["id"]
        key, fam, scenario = _case_key(cid)
        chunks = c.get("chunks") or []

        slot("inputs", fam, model_label=fam)[key] = {
            "scenario": scenario,
            "description": c.get("description", ""),
            "policy": c.get("policy") or [],
            "init": c.get("init") or {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None},
            "finish_reason": c.get("finish_reason") or "stop",
            "input": c.get("input", ""),
            "tools": c.get("tools"),
            "chunks": [
                {"delta_text": ch.get("delta_text", "")} for ch in chunks
            ],
        }

        # golden/<family>/<key>.yaml — the authored oracle (assembled events)
        slot("golden", fam, captured_with={"golden": "v1"})[key] = {
            "assembled": c.get("golden") or [],
        }

        # dynamo_v2-<ver>/<family>/<key>.yaml — LIVE dynamo (assembled + per-chunk)
        ddir = f"dynamo_v2-{ver['dynamo_v2']}"
        slot(ddir, fam, captured_with={"dynamo_v2": provenance["crate_version"]})[key] = {
            "capture_input": capture_input(c),
            "assembled": c.get("dynamo") or [],
            "chunks": [{"expected": ch.get("dynamo") or []} for ch in chunks],
        }

        # peer engine overlays
        for impl in ("vllm_python", "vllm_rust", "sglang_python"):
            res = (caps[impl].get("results") or {}).get(cid)
            if res is None:
                continue
            vdir = f"{impl}-{ver[impl]}"
            entry = slot(vdir, fam, captured_with={impl: ver[impl]})
            entry[key] = _peer_cell(res)

    # Rebuild generated views from canonical captures.
    _clear_generated_dirs()

    n = 0
    for (dirname, family), doc in docs.items():
        for key in list(doc["cases"]):
            one = dict(doc)
            one["cases"] = {key: doc["cases"][key]}
            _dump(one, BUILD / dirname / family / f"{key}.yaml")
            n += 1
    print(f"wrote {n} case files across {len({d for d, _ in docs})} version dirs")


if __name__ == "__main__":
    main()
