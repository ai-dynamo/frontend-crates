#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate batch-on-stream overlay fixtures for every tool-calling family."""
import argparse
import glob
import json
import os
import subprocess
import sys

import yaml

from capture_all_families_driver import SGLANG, VLLM


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def _container_path(fp):
    family = os.path.basename(os.path.dirname(fp))
    return f"/tmp/bos_{family}_{os.path.basename(fp)}"


def _capture_all(container, impl, jobs, work):
    for job in jobs:
        run(["docker", "cp", job["src"], f"{container}:{job['container_path']}"])
    batch_path = f"/tmp/batch_on_stream_{impl}.json"
    host_batch = os.path.join(work, f"batch_on_stream_{impl}.json")
    open(host_batch, "w").write(
        json.dumps(
            [
                {"fixture": job["container_path"], "parser": job["parser"]}
                for job in jobs
            ]
        )
    )
    run(["docker", "cp", host_batch, f"{container}:{batch_path}"])
    proc = subprocess.run(
        [
            "docker",
            "exec",
            container,
            "bash",
            "-lc",
            f"python3 /tmp/capture_batch_on_stream.py --impl {impl} --batch \"$(cat {batch_path})\"",
        ],
        capture_output=True,
        text=True,
    )
    out = "\n".join(line for line in proc.stdout.splitlines() if line.startswith("{"))
    if not out:
        raise RuntimeError(f"{impl} capture failed: {proc.stderr[-1000:]}")
    data = json.loads(out)
    return data["version"], {
        job["src"]: data["fixtures"].get(job["container_path"], {}) for job in jobs
    }


def _load_harmony_dynamo(path):
    if not path:
        return {}
    with open(path) as f:
        return json.load(f)


def _block_for(impl, family, parser, entry):
    engine = "vLLM" if impl == "vllm" else "SGLang"
    if parser is None:
        return {"unavailable": f"No {engine} parser for family '{family}'."}
    if "cases" in entry:
        return entry["cases"]
    return {}


def _parser_for(impl, family):
    if family == "harmony":
        return "harmony"
    return VLLM.get(family) if impl == "vllm" else SGLANG.get(family)


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
    vllm_cases = _block_for(
        "vllm", family, vllm_parser, vllm_entry
    )
    sglang_cases = _block_for(
        "sglang",
        family,
        sglang_parser,
        sglang_entry,
    )

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--vllm-container", default="vllm-localdev")
    ap.add_argument("--sglang-container", default="sglang-localdev")
    ap.add_argument("--dynamo-harmony-json")
    args = ap.parse_args()
    os.makedirs(args.work, exist_ok=True)

    ph = os.path.join(args.root, "conformance/utils")
    for container in (args.vllm_container, args.sglang_container):
        for name in (
            "capture_stream.py",
            "capture_harmony_batch_stream.py",
            "capture_batch_on_stream.py",
        ):
            run(["docker", "cp", os.path.join(ph, name), f"{container}:/tmp/{name}"])

    fixture_root = os.path.join(args.root, "conformance/toolcalling/fixtures")
    sources = sorted(glob.glob(f"{fixture_root}/*/TOOLCALLING.batch*.yaml"))
    jobs = {"vllm": [], "sglang": []}
    for src in sources:
        family = os.path.basename(os.path.dirname(src))
        cpath = _container_path(src)
        parser = _parser_for("vllm", family)
        if parser:
            jobs["vllm"].append(
                {"src": src, "container_path": cpath, "parser": parser}
            )
        parser = _parser_for("sglang", family)
        if parser:
            jobs["sglang"].append(
                {"src": src, "container_path": cpath, "parser": parser}
            )

    print(f"capturing vllm ({len(jobs['vllm'])} batch fixtures)...", file=sys.stderr)
    vllm_ver, vllm_caps = _capture_all(
        args.vllm_container, "vllm", jobs["vllm"], args.work
    )
    print(f"capturing sglang ({len(jobs['sglang'])} batch fixtures)...", file=sys.stderr)
    sglang_ver, sglang_caps = _capture_all(
        args.sglang_container, "sglang", jobs["sglang"], args.work
    )

    dynamo_harmony = _load_harmony_dynamo(args.dynamo_harmony_json)
    versions = {"vllm": vllm_ver, "sglang": sglang_ver}
    out_root = os.path.join(
        args.root, "conformance/toolcalling/fixtures-batch-on-stream-v2"
    )
    for src in sources:
        family = os.path.basename(os.path.dirname(src))
        outfp = os.path.join(out_root, family, os.path.basename(src))
        _write_overlay(
            src,
            outfp,
            vllm_caps.get(src, {}),
            sglang_caps.get(src, {}),
            versions,
            dynamo_harmony,
        )
        print(f"  wrote {family}/{os.path.basename(src)}", file=sys.stderr)


if __name__ == "__main__":
    main()
