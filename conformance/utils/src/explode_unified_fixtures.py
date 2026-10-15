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

Same schema shape as toolcalling/fixtures-stream-v1 (family/mode/captured_with/cases).
Run:  python3 conformance/utils/src/explode_unified_fixtures.py
"""
import argparse
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dynamo_version import validate_capture_provenance  # noqa: E402
from capture_stimulus import capture_input  # noqa: E402
from unified_taxonomy import numbered_id  # noqa: E402
import unified_history  # noqa: E402

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
    # Internal scenario keys stay stable while the taxonomy owns published labels.
    fam = case_id.rsplit(".", 1)[1]
    scenario = case_id[len("UNIFIED."):].rsplit(".", 1)[0]
    return numbered_id(scenario), fam, scenario


def _normalize_peer_value(
    result, expected_chunk_count=None, *, require_assembled=False
):
    assembled = result.get("assembled")
    chunks = result.get("chunks")
    if not isinstance(assembled, list):
        if require_assembled:
            raise ValueError("raw peer result assembled must be a list")
        assembled = []
    if not isinstance(chunks, list):
        raise ValueError("raw peer result chunks must be a list")
    if expected_chunk_count is not None and len(chunks) != expected_chunk_count:
        raise ValueError(
            "raw peer result chunk count differs from authored request: "
            f"expected={expected_chunk_count} actual={len(chunks)}"
        )
    for index, events in enumerate(chunks):
        if not isinstance(events, list):
            raise ValueError(f"raw peer result chunk {index} must be a list")
        for event in events:
            if not isinstance(event, dict):
                raise ValueError(f"raw peer result chunk {index} contains a non-mapping event")
            if "expected" in event:
                raise ValueError(
                    "raw peer result contains canonical expected wrapper; "
                    "double-wrapping is not allowed"
                )
    return {
        "assembled": assembled,
        "chunks": [{"expected": events} for events in chunks],
    }


def _peer_cell(result, expected_chunk_count=None, *, require_assembled=False):
    """Keep parser failures separate from partial streaming evidence."""
    if result.get("unavailable"):
        record = {"unavailable": result["unavailable"]}
    elif result.get("error"):
        record = {"error": result["error"]}
    else:
        record = _normalize_peer_value(
            result,
            expected_chunk_count,
            require_assembled=require_assembled,
        )
    if "partial_chunks" in result:
        if not result.get("error"):
            raise ValueError("partial chunk evidence requires a parser error")
        record["partial_chunks"] = _normalize_peer_value({"chunks": result["partial_chunks"]})["chunks"]
    if "error_traceback" in result:
        if not result.get("error") or not isinstance(result["error_traceback"], str):
            raise ValueError("error traceback evidence requires a parser error and string traceback")
        record["error_traceback"] = result["error_traceback"]
    record.update({key: result[key] for key in ("capture_input", "capture_stimulus", "capture_observation") if key in result})
    return record


def _history_peer_cell(store, implementation, capture_id, family, scenario, request):
    """Reuse an immutable peer observation omitted from the scratch capture feed."""
    history = store.histories.get((family, implementation))
    if history is None or capture_id not in history.captures:
        return None
    matching = [
        case_id
        for case_id, case in history.family.cases.items()
        if case["scenario"] == scenario
    ]
    if len(matching) != 1:
        return None
    case_id = matching[0]
    change = history.resolve(capture_id).get(case_id)
    if change is None:
        return None
    stimulus = change["stimulus"]
    if "ref" in stimulus:
        captured_request = history.family.cases[case_id]["request"]
    elif "inline" in stimulus:
        captured_request = stimulus["inline"]
    else:
        return None
    if captured_request != request:
        return None
    record, document = unified_history.materialized_record(
        history.family.cases[case_id], change
    )
    record.update(document.get("record_metadata", {}))
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


def _split_only_cases(family, document):
    """Build input/GOLDEN rows for a family without a native Unified parser."""
    if document.get("family") != family or not isinstance(document.get("cases"), dict):
        raise ValueError(f"{family}: expected a family-scoped golden spec")
    rows = {}
    for case_id, case in document["cases"].items():
        key, actual_family, scenario = _case_key(case_id)
        if actual_family != family:
            raise ValueError(f"{case_id}: golden case belongs to {actual_family}, not {family}")
        if not all(field in case for field in ("input", "init", "finish_reason", "golden")):
            raise ValueError(f"{case_id}: golden case is missing authored request fields")
        raw_input = case["input"]
        if not isinstance(raw_input, str):
            raise ValueError(f"{case_id}: input must be text")
        request = {
            "input": raw_input,
            "init": case["init"],
            "finish_reason": case["finish_reason"],
            # The request schema stores an empty offered-tool set as [], matching
            # the legacy batch API's required slice without inventing a schema.
            "tools": case.get("tools") or [],
            "chunks": [{"delta_text": raw_input}],
        }
        rows[key] = {
            "scenario": scenario,
            "description": case.get("description", ""),
            "policy": case.get("policy") or [],
            "request": request,
            "golden": {"assembled": case["golden"]},
        }
    return rows


def _input_case(case, request):
    return {
        "scenario": case["scenario"],
        "description": case.get("description", ""),
        "policy": case.get("policy") or [],
        **{key: request[key] for key in ("init", "finish_reason", "input", "tools", "chunks")},
    }


def write_split_only_family(family):
    """Materialize only authored inputs and GOLDEN for a non-native family."""
    registry = yaml.safe_load((CONF / "utils/src/parser_families.yaml").read_text())["unified"]
    entry = registry.get(family)
    if entry is None or entry.get("native") is not False:
        raise ValueError(f"{family}: split-only generation requires native: false")
    spec_path = BUILD / "golden_spec" / entry["golden_spec"]
    document = yaml.safe_load(spec_path.read_text())
    rows = _split_only_cases(family, document)
    input_header = {"family": family, "mode": MODE, "model_label": family}
    golden_header = {"family": family, "mode": MODE, "captured_with": {"golden": "v1"}}
    for key, row in rows.items():
        _dump({**input_header, "cases": {key: _input_case(row, row["request"])}}, BUILD / "inputs" / family / f"{key}.yaml")
        _dump({**golden_header, "cases": {key: row["golden"]}}, BUILD / "golden" / family / f"{key}.yaml")
    print(f"wrote {len(rows)} input and GOLDEN cases for split-only family {family}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-only-family", help="materialize authored input and GOLDEN only for a native: false family")
    args = parser.parse_args()
    if args.split_only_family:
        write_split_only_family(args.split_only_family)
        return
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
    active_cases = {}

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
        request = capture_input(c)
        active_cases[(fam, scenario)] = (key, request)

        slot("inputs", fam, model_label=fam)[key] = _input_case(
            {**c, "scenario": scenario},
            {**request, "chunks": [{"delta_text": ch.get("delta_text", "")} for ch in chunks]},
        )

        # golden/<family>/<key>.yaml — the authored oracle (assembled events)
        slot("golden", fam, captured_with={"golden": "v1"})[key] = {
            "assembled": c.get("golden") or [],
        }

        # dynamo_v2-<ver>/<family>/<key>.yaml — LIVE dynamo (assembled + per-chunk)
        ddir = f"dynamo_v2-{ver['dynamo_v2']}"
        slot(ddir, fam, captured_with={"dynamo_v2": provenance["crate_version"]})[key] = {
            "capture_input": request,
            "assembled": c.get("dynamo") or [],
            "chunks": [{"expected": ch.get("dynamo") or []} for ch in chunks],
        }

        if c.get("dynamo_error"):
            slot(ddir, fam)[key] = {"capture_input": capture_input(c), "error": c["dynamo_error"]}

        # peer engine overlays
        for impl in ("vllm_python", "vllm_rust", "sglang_python"):
            res = (caps[impl].get("results") or {}).get(cid)
            if res is None:
                continue
            vdir = f"{impl}-{ver[impl]}"
            entry = slot(vdir, fam, captured_with={impl: ver[impl]})
            entry[key] = _peer_cell(
                res,
                expected_chunk_count=len(chunks),
                require_assembled=impl in {"vllm_python", "vllm_rust"},
            )

    # The scratch peer feeds are ignored build products and can predate a restored
    # case. Preserve the exact checked-in observation when its scenario and request
    # still match; a changed request deliberately falls through to package validation.
    store = unified_history.load_store(CONF / "fixtures-unified-v2")
    for (fam, scenario), (key, request) in active_cases.items():
        for impl in ("vllm_python", "vllm_rust", "sglang_python"):
            capture_id = f"{impl}-{ver[impl]}"
            existing = docs.get((capture_id, fam), {}).get("cases", {})
            if key in existing:
                continue
            record = _history_peer_cell(
                store, impl, capture_id, fam, scenario, request
            )
            if record is not None:
                slot(capture_id, fam, captured_with={impl: ver[impl]})[key] = record

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
