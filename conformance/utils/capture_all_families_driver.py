#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Driver for capture_all_families.sh. For each non-harmony family: run vLLM +
SGLang batch capture in their containers, then build the new-format fixture with
build_stream_fixtures.py (dynamo unavailable=TODO, captured engine versions)."""
import argparse
import glob
import json
import os
import subprocess
import sys

# family -> parser/detector name per engine. None = no parser for this engine
# (the family's tool format isn't supported there) -> marked unavailable.
VLLM = {
    "deepseek_v3": "deepseek_v3", "deepseek_v3_1": "deepseek_v31",
    "deepseek_v3_2": "deepseek_v32", "deepseek_v4": "deepseek_v4",
    "gemma4": "gemma4", "glm47": "glm47", "jamba": "jamba",
    "kimi_k2": "kimi_k2", "llama3_json": "llama3_json",
    "minimax_m2": "minimax_m2", "mistral": "mistral",
    "nemotron_deci": "hermes", "nemotron_nano": "hermes",
    "phi4": "phi4_mini_json", "pythonic": "pythonic",
    "qwen25": "hermes", "qwen3_coder": "qwen3_coder",
}
SGLANG = {
    "deepseek_v3": "deepseekv3", "deepseek_v3_1": "deepseekv31",
    "deepseek_v3_2": "deepseekv32", "deepseek_v4": "deepseekv4",
    "gemma4": "gemma4", "glm47": "glm47", "jamba": None,
    "kimi_k2": "kimi_k2", "llama3_json": "llama3",
    "minimax_m2": "minimax-m2", "mistral": "mistral",
    "nemotron_deci": None, "nemotron_nano": None,
    "phi4": None, "pythonic": "pythonic",
    "qwen25": "qwen25", "qwen3_coder": "qwen3_coder",
}


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def container_capture_all(container, impl, jobs, work):
    """One batch capture for ALL families in a single container exec (one engine
    import total). `jobs`: [{src, container_path, parser}]. Returns (version,
    {src: entry})."""
    # Copy every fixture into the container, then run a single batched capture.
    for j in jobs:
        run(["docker", "cp", j["src"], f"{container}:{j['container_path']}"])
    batch = json.dumps([{"fixture": j["container_path"], "parser": j["parser"]} for j in jobs])
    # Pass the batch JSON via a file in the container to avoid shell-quoting limits.
    batch_path = f"/tmp/batch_{impl}.json"
    bf = os.path.join(work, f"batch_{impl}.json")
    open(bf, "w").write(batch)
    run(["docker", "cp", bf, f"{container}:{batch_path}"])
    proc = subprocess.run(
        ["docker", "exec", container, "bash", "-lc",
         f"python3 /tmp/capture_stream.py --impl {impl} --batch \"$(cat {batch_path})\""],
        capture_output=True, text=True)
    out = "\n".join(l for l in proc.stdout.splitlines() if l.strip().startswith("{"))
    if not out:
        raise RuntimeError(f"{container} capture failed: {proc.stderr[-800:]}")
    data = json.loads(out)
    by_src = {j["src"]: data["fixtures"].get(j["container_path"], {}) for j in jobs}
    return data["version"], by_src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--vllm-container", required=True)
    ap.add_argument("--sglang-container", required=True)
    ap.add_argument("--dynamo-todo", required=True)
    args = ap.parse_args()

    ph = os.path.join(args.root, "conformance/utils")
    conf = os.path.join(args.root, "conformance/toolcalling/fixtures")

    # copy capture_stream.py into both containers once
    for c in (args.vllm_container, args.sglang_container):
        run(["docker", "cp", os.path.join(ph, "capture_stream.py"), f"{c}:/tmp/capture_stream.py"])

    families = sorted(VLLM.keys())

    # Build one job list per engine across ALL families, so each engine is
    # imported exactly once (the import is the expensive part).
    def cpath(fp):
        return f"/tmp/cap_{os.path.basename(os.path.dirname(fp))}_{os.path.basename(fp)}"

    vllm_jobs, sglang_jobs = [], []
    family_fixtures = {}
    for family in families:
        fixtures = sorted(glob.glob(f"{conf}/{family}/TOOLCALLING.stream.*.yaml"))
        family_fixtures[family] = fixtures
        for fp in fixtures:
            if VLLM[family]:
                vllm_jobs.append({"src": fp, "container_path": cpath(fp), "parser": VLLM[family]})
            if SGLANG[family]:
                sglang_jobs.append({"src": fp, "container_path": cpath(fp), "parser": SGLANG[family]})

    print(f"capturing vllm ({len(vllm_jobs)} fixtures, 1 import)...", file=sys.stderr)
    vllm_ver, vllm_caps = container_capture_all(args.vllm_container, "vllm", vllm_jobs, args.work)
    print(f"capturing sglang ({len(sglang_jobs)} fixtures, 1 import)...", file=sys.stderr)
    sglang_ver, sglang_caps = container_capture_all(args.sglang_container, "sglang", sglang_jobs, args.work)

    for family in families:
        fixtures = family_fixtures[family]
        if not fixtures:
            continue
        for fp in fixtures:
            base = os.path.basename(fp)
            outdir = os.path.join(args.root, "conformance", "toolcalling", "fixtures-stream-v2", family)
            os.makedirs(outdir, exist_ok=True)
            outfp = os.path.join(outdir, base)

            cmd = ["python3", os.path.join(ph, "build_stream_fixtures.py"),
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


if __name__ == "__main__":
    main()
