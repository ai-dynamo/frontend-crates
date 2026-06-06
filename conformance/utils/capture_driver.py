#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-side orchestrator for the conformance captures. Copies capture.py into the
engine containers, runs one batched capture per engine (one import per engine),
then assembles fixtures. Runs on the HOST (docker exec), not inside a container.

Modes (`--mode`):
  stream           Per-chunk vLLM + SGLang streaming for every non-harmony family;
                   builds conformance/toolcalling/fixtures-stream-v2/<family>/TOOLCALLING.stream.*.yaml
                   (Dynamo parser v2 marked unavailable/TODO). Calls build_stream_fixtures.py.
  batch-on-stream  Each family's batch text through each engine's streaming parser;
                   writes conformance/toolcalling/fixtures-batch-on-stream-v2/<family>/TOOLCALLING.batch*.yaml
                   overlay (optionally with the Dynamo harmony recorder JSON).
  merge            Merge the three per-engine flat stream-on-batch captures
                   (--dynamo/--vllm/--sglang JSON) into the nested
                   harmony_batch_stream.json the older harmony flow consumes.

Recipes:
  conformance/utils/capture_all_families.sh                # mode stream (wrapper)
  python3 conformance/utils/capture_driver.py --mode batch-on-stream \
      --root <repo> --work /tmp/bos [--dynamo-harmony-json /tmp/dynamo_bs.json]
  cargo run -p dynamo-parsers-v2 --bin record_batch_via_stream > /tmp/dynamo_bs.json
  python3 conformance/utils/capture_driver.py --mode merge \
      --dynamo /tmp/dynamo_bs.json --vllm /tmp/vllm_bs.json --sglang /tmp/sglang_bs.json \
      -o conformance/utils/harmony_batch_stream.json
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import yaml

# family -> parser/detector name per engine. None = no parser for this engine
# (the family's tool format isn't supported there) -> marked unavailable.
VLLM = {
    "deepseek_v3": "deepseek_v3", "deepseek_v3_1": "deepseek_v31",
    "deepseek_v3_2": "deepseek_v32", "deepseek_v4": "deepseek_v4",
    "gemma4": "gemma4", "glm47": "glm47", "hermes": "hermes", "jamba": "jamba",
    "kimi_k2": "kimi_k2", "llama3_json": "llama3_json",
    "minimax_m2": "minimax_m2", "mistral": "mistral",
    "nemotron_deci": "hermes", "nemotron_nano": "hermes",
    "phi4": "phi4_mini_json", "pythonic": "pythonic",
    "qwen25": "hermes", "qwen3_coder": "qwen3_coder",
}
SGLANG = {
    "deepseek_v3": "deepseekv3", "deepseek_v3_1": "deepseekv31",
    "deepseek_v3_2": "deepseekv32", "deepseek_v4": "deepseekv4",
    "gemma4": "gemma4", "glm47": "glm47", "hermes": "hermes", "jamba": None,
    "kimi_k2": "kimi_k2", "llama3_json": "llama3",
    "minimax_m2": "minimax-m2", "mistral": "mistral",
    "nemotron_deci": None, "nemotron_nano": None,
    "phi4": None, "pythonic": "pythonic",
    "qwen25": "qwen25", "qwen3_coder": "qwen3_coder",
}


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def _copy_worker(containers):
    """Copy capture.py into each container once."""
    here = os.path.dirname(os.path.abspath(__file__))
    for c in containers:
        run(["docker", "cp", os.path.join(here, "capture.py"), f"{c}:/tmp/capture.py"])


def _container_capture(container, impl, mode, jobs, work):
    """One batched capture for ALL families in a single container exec (one engine
    import total). `jobs`: [{src, container_path, parser}]. Returns (version,
    {src: entry}). `entry` is {cases: {...}} on success or {error: ...}."""
    for j in jobs:
        run(["docker", "cp", j["src"], f"{container}:{j['container_path']}"])
    batch = json.dumps(
        [{"fixture": j["container_path"], "parser": j["parser"]} for j in jobs]
    )
    # Pass the batch JSON via a file in the container to avoid shell-quoting limits.
    batch_path = f"/tmp/batch_{mode}_{impl}.json"
    bf = os.path.join(work, f"batch_{mode}_{impl}.json")
    open(bf, "w").write(batch)
    run(["docker", "cp", bf, f"{container}:{batch_path}"])
    proc = subprocess.run(
        ["docker", "exec", container, "bash", "-lc",
         f'python3 /tmp/capture.py --mode {mode} --impl {impl} --batch "$(cat {batch_path})"'],
        capture_output=True, text=True)
    out = "\n".join(l for l in proc.stdout.splitlines() if l.strip().startswith("{"))
    if not out:
        raise RuntimeError(f"{container} {mode} capture failed: {proc.stderr[-1000:]}")
    data = json.loads(out)
    by_src = {j["src"]: data["fixtures"].get(j["container_path"], {}) for j in jobs}
    return data["version"], by_src


def _cpath(fp, mode):
    family = os.path.basename(os.path.dirname(fp))
    tag = "bos" if mode == "batch-on-stream" else "cap"
    return f"/tmp/{tag}_{family}_{os.path.basename(fp)}"


# --------------------------------------------------------------------------- #
# mode=stream (was capture_all_families_driver.py)
# --------------------------------------------------------------------------- #
def _impl_args(impl, family, parser, entry, version, work, tag, src):
    """Build build_stream_fixtures.py args for one impl: pass captured data, or an
    accurate `unavailable` reason (no parser registered vs. capture error)."""
    engine = "vLLM" if impl == "vllm" else "SGLang"
    if parser is None:
        return ["--unavailable",
                f"{impl}=No {engine} parser for family '{family}'."]
    if "cases" in entry:
        f = os.path.join(work, f"{tag}_{impl}.json")
        json.dump(entry["cases"], open(f, "w"))
        return [f"--{impl}", f, "--captured", f"{impl}={version}"]
    # Parser exists but capture errored (typically: requires the model tokenizer's
    # special tool tokens, which a stub tokenizer can't supply).
    err = (entry.get("error") or "capture failed").splitlines()[-1][:160]
    return ["--unavailable",
            f"{impl}={engine} '{parser}' parser not captured with a stub tokenizer: {err}"]


def _run_stream(args):
    here = os.path.dirname(os.path.abspath(__file__))
    conf = os.path.join(args.root, "conformance/toolcalling/fixtures")
    _copy_worker((args.vllm_container, args.sglang_container))

    families = sorted(VLLM.keys())
    vllm_jobs, sglang_jobs = [], []
    family_fixtures = {}
    for family in families:
        fixtures = sorted(glob.glob(f"{conf}/{family}/TOOLCALLING.stream.*.yaml"))
        family_fixtures[family] = fixtures
        for fp in fixtures:
            if VLLM[family]:
                vllm_jobs.append({"src": fp, "container_path": _cpath(fp, "stream"), "parser": VLLM[family]})
            if SGLANG[family]:
                sglang_jobs.append({"src": fp, "container_path": _cpath(fp, "stream"), "parser": SGLANG[family]})

    print(f"capturing vllm ({len(vllm_jobs)} fixtures, 1 import)...", file=sys.stderr)
    vllm_ver, vllm_caps = _container_capture(args.vllm_container, "vllm", "stream", vllm_jobs, args.work)
    print(f"capturing sglang ({len(sglang_jobs)} fixtures, 1 import)...", file=sys.stderr)
    sglang_ver, sglang_caps = _container_capture(args.sglang_container, "sglang", "stream", sglang_jobs, args.work)

    for family in families:
        fixtures = family_fixtures[family]
        if not fixtures:
            continue
        for fp in fixtures:
            base = os.path.basename(fp)
            outdir = os.path.join(args.root, "conformance", "toolcalling", "fixtures-stream-v2", family)
            os.makedirs(outdir, exist_ok=True)
            outfp = os.path.join(outdir, base)

            cmd = ["python3", os.path.join(here, "build_stream_fixtures.py"),
                   "--source", fp, "--out", outfp,
                   "--unavailable", f"dynamo={args.dynamo_todo}"]
            cmd += _impl_args(
                "vllm", family, VLLM[family], vllm_caps.get(fp, {}), vllm_ver,
                args.work, f"{family}_{base}", fp)
            cmd += _impl_args(
                "sglang", family, SGLANG[family], sglang_caps.get(fp, {}), sglang_ver,
                args.work, f"{family}_{base}", fp)
            run(cmd)
            print(f"  built {family}/{base}", file=sys.stderr)


# --------------------------------------------------------------------------- #
# mode=batch-on-stream (was capture_batch_on_stream_all.py)
# --------------------------------------------------------------------------- #
def _parser_for(impl, family):
    if family == "harmony":
        return "harmony"
    return VLLM.get(family) if impl == "vllm" else SGLANG.get(family)


def _block_for(impl, family, parser, entry):
    engine = "vLLM" if impl == "vllm" else "SGLang"
    if parser is None:
        return {"unavailable": f"No {engine} parser for family '{family}'."}
    if "cases" in entry:
        return entry["cases"]
    return {}


def _load_harmony_dynamo(path):
    if not path:
        return {}
    with open(path) as f:
        return json.load(f)


def _write_overlay(src, outfp, vllm_entry, sglang_entry, versions, dynamo_harmony):
    doc = yaml.safe_load(open(src))
    family = doc["family"]
    out = {
        "family": family,
        "mode": "batch-on-stream",
        "captured_with": {
            "vllm": versions["vllm"],
            "sglang": versions["sglang"],
        },
        "cases": {},
    }
    if family == "harmony":
        out["captured_with"]["dynamo"] = "Dynamo parser v2"

    vllm_parser = _parser_for("vllm", family)
    sglang_parser = _parser_for("sglang", family)
    vllm_cases = _block_for("vllm", family, vllm_parser, vllm_entry)
    sglang_cases = _block_for("sglang", family, sglang_parser, sglang_entry)

    for cid, case in (doc.get("cases") or {}).items():
        row = {}
        if family == "harmony" and cid in dynamo_harmony:
            row["dynamo"] = dynamo_harmony[cid]
        for impl, parser, cases in (
            ("vllm", vllm_parser, vllm_cases),
            ("sglang", sglang_parser, sglang_cases),
        ):
            if parser is None:
                row[impl] = {
                    "unavailable": f"No {'vLLM' if impl == 'vllm' else 'SGLang'} parser for family '{family}'."
                }
            elif cid in cases:
                row[impl] = cases[cid]
            elif "model_text" not in case:
                row[impl] = {"unavailable": "No batch model_text for this case."}
            else:
                row[impl] = {"unavailable": "Capture did not return this case."}
        out["cases"][cid] = row

    os.makedirs(os.path.dirname(outfp), exist_ok=True)
    with open(outfp, "w") as f:
        yaml.safe_dump(out, f, allow_unicode=True, sort_keys=False)


def _run_batch_on_stream(args):
    _copy_worker((args.vllm_container, args.sglang_container))
    fixture_root = os.path.join(args.root, "conformance/toolcalling/fixtures")
    sources = sorted(glob.glob(f"{fixture_root}/*/TOOLCALLING.batch*.yaml"))
    jobs = {"vllm": [], "sglang": []}
    for src in sources:
        family = os.path.basename(os.path.dirname(src))
        cpath = _cpath(src, "batch-on-stream")
        for impl in ("vllm", "sglang"):
            parser = _parser_for(impl, family)
            if parser:
                jobs[impl].append({"src": src, "container_path": cpath, "parser": parser})

    print(f"capturing vllm ({len(jobs['vllm'])} batch fixtures)...", file=sys.stderr)
    vllm_ver, vllm_caps = _container_capture(
        args.vllm_container, "vllm", "batch-on-stream", jobs["vllm"], args.work)
    print(f"capturing sglang ({len(jobs['sglang'])} batch fixtures)...", file=sys.stderr)
    sglang_ver, sglang_caps = _container_capture(
        args.sglang_container, "sglang", "batch-on-stream", jobs["sglang"], args.work)

    dynamo_harmony = _load_harmony_dynamo(args.dynamo_harmony_json)
    versions = {"vllm": vllm_ver, "sglang": sglang_ver}
    out_root = os.path.join(args.root, "conformance/toolcalling/fixtures-batch-on-stream-v2")
    for src in sources:
        family = os.path.basename(os.path.dirname(src))
        outfp = os.path.join(out_root, family, os.path.basename(src))
        _write_overlay(
            src, outfp, vllm_caps.get(src, {}), sglang_caps.get(src, {}),
            versions, dynamo_harmony)
        print(f"  wrote {family}/{os.path.basename(src)}", file=sys.stderr)


# --------------------------------------------------------------------------- #
# mode=merge (was merge_batch_stream.py)
# --------------------------------------------------------------------------- #
def _run_merge(args):
    layers = {
        "dynamo": json.load(open(args.dynamo)),
        "vllm": json.load(open(args.vllm)),
        "sglang": json.load(open(args.sglang)),
    }
    cids = sorted({cid for layer in layers.values() for cid in layer})
    nested = {
        cid: {
            engine: {"calls": layer.get(cid, {}).get("calls", [])}
            for engine, layer in layers.items()
        }
        for cid in cids
    }
    json.dump(nested, open(args.output, "w"), ensure_ascii=False, indent=2)
    print(f"wrote {args.output}: {len(nested)} cases × {len(layers)} engines")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", required=True, choices=("stream", "batch-on-stream", "merge"))
    # stream / batch-on-stream
    ap.add_argument("--root")
    ap.add_argument("--work")
    ap.add_argument("--vllm-container", default="vllm-localdev")
    ap.add_argument("--sglang-container", default="sglang-localdev")
    ap.add_argument("--dynamo-todo", help="stream: Dynamo unavailable/TODO reason")
    ap.add_argument("--dynamo-harmony-json", help="batch-on-stream: recorder JSON")
    # merge
    ap.add_argument("--dynamo")
    ap.add_argument("--vllm")
    ap.add_argument("--sglang")
    ap.add_argument("-o", "--output")
    args = ap.parse_args()

    # Per-mode required args (argparse can't express "required only for some modes").
    if args.mode == "merge":
        missing = [n for n in ("dynamo", "vllm", "sglang", "output") if not getattr(args, n)]
        if missing:
            ap.error("--mode merge requires --dynamo --vllm --sglang -o/--output")
        _run_merge(args)
        return
    if not args.root or not args.work:
        ap.error(f"--mode {args.mode} requires --root and --work")
    if args.mode == "stream" and not args.dynamo_todo:
        ap.error("--mode stream requires --dynamo-todo")

    os.makedirs(args.work, exist_ok=True)
    if args.mode == "stream":
        _run_stream(args)
    else:
        _run_batch_on_stream(args)


if __name__ == "__main__":
    main()
