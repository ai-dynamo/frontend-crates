#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate a Python engine's tool-call parser (vLLM or SGLang) against the
vendored conformance fixtures' ``expected.<impl>`` blocks.

The engines are Python and cannot run inside the Rust conformance crate, so this
crosses the language barrier two ways:

  --pip            import the adapter + engine in THIS interpreter (engine must be
                   importable here).
  --container N    ship a minimal bundle into docker container N and run the
                   adapter there (no local engine needed). Mirrors the docker-exec
                   capture pattern from dynamo PR #10296: the worker writes results
                   to a file because engine import spews to stdout.

Each case is compared with ``tests.parity.common.canonical``, the same contract
the dynamo M2 harness and the Rust conformance crate use. Reports the live engine
version and warns when it differs from the version dynamo pinned (the fixtures'
``expected.<impl>`` columns were captured against that pin — a mismatch makes
diffs version drift, not parser bugs). Exits non-zero on any real mismatch.

Run via ``conformance/utils/check.sh vllm|sglang``; it builds the staged fixtures dir
and passes --fixtures. Direct use:
    validate.py --impl sglang --container sglang-localdev --fixtures <dir>
    validate.py --impl vllm   --pip                       --fixtures <dir>
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

from tests.parity.common import ParseResult, canonical

PH = Path(__file__).resolve().parent
PKG = PH / "tests" / "parity"
STUB = PH / "pyproject.stub.toml"
from impls import FIXTURE_IMPL_ALIASES  # noqa: E402  (identity table; see impls.py)

# Minimal worker shipped into the engine container. Imports the adapter once
# (heavy), then maps stdin JSONL requests to result JSONL written to --out.
_WORKER_SRC = r'''
import argparse, importlib, json, sys
ap = argparse.ArgumentParser()
ap.add_argument("--impl", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
mod = importlib.import_module(f"tests.parity.toolcalling.{a.impl}")
with open(a.out, "w", encoding="utf-8") as f:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r["mode"] == "stream":
            pr = mod.parse_tool_calls_stream(r["family"], r.get("chunks"), r.get("tools"))
        else:
            pr = mod.parse_tool_calls_batch(r["family"], r.get("model_text"), r.get("tools"))
        d = pr.to_dict()
        d["key"] = r["key"]
        f.write(json.dumps(d) + "\n")
'''


def pinned_version(impl: str) -> str | None:
    """The vllm/sglang version dynamo pinned, from the synced stub."""
    if not STUB.exists():
        return None
    m = re.search(rf'"{impl}(?:\[[^\]]*\])?==([0-9][^"]*)"', STUB.read_text())
    return m.group(1) if m else None


def collect_cases(fixtures_dir: Path, impl: str) -> list[dict]:
    """Runnable cases for ``impl``: those with an expected.<impl> block that isn't
    'unavailable' and that carry an input (model_text for batch / chunks for
    stream). Each carries its expected spec for host-side comparison."""

    fixture_impl = FIXTURE_IMPL_ALIASES.get(impl, impl)
    out: list[dict] = []
    for fp in sorted(fixtures_dir.glob("*/*.yaml")):
        doc = yaml.safe_load(fp.read_text())
        family, mode = doc.get("family"), doc.get("mode")
        for cid, case in (doc.get("cases") or {}).items():
            if not isinstance(case, dict) or "expected" not in case:
                continue
            spec = (case.get("expected") or {}).get(fixture_impl)
            if spec is None and fixture_impl != impl:
                spec = (case.get("expected") or {}).get(impl)
            if not isinstance(spec, dict) or "unavailable" in spec:
                continue
            payload = (
                {"chunks": case.get("chunks")}
                if mode == "stream"
                else {"model_text": case.get("model_text")}
            )
            if payload.get("chunks") is None and payload.get("model_text") is None:
                continue
            out.append(
                {
                    "key": f"{family}/{cid}",
                    "family": family,
                    "mode": mode,
                    "tools": case.get("tools"),
                    "spec": spec,
                    **payload,
                }
            )
    return out


def _request(c: dict) -> dict:
    r = {"key": c["key"], "family": c["family"], "mode": c["mode"], "tools": c["tools"]}
    if c["mode"] == "stream":
        r["chunks"] = c["chunks"]
    else:
        r["model_text"] = c["model_text"]
    return r


def run_pip(impl: str, cases: list[dict]) -> dict[str, dict]:
    """Run the adapter in-process. The adapters return ParseResult(error=...) for
    bad input rather than raising, so no broad except is needed here."""
    mod = importlib.import_module(f"tests.parity.toolcalling.{impl}")
    results: dict[str, dict] = {}
    for c in cases:
        if c["mode"] == "stream":
            pr = mod.parse_tool_calls_stream(c["family"], c["chunks"], c["tools"])
        else:
            pr = mod.parse_tool_calls_batch(c["family"], c["model_text"], c["tools"])
        results[c["key"]] = pr.to_dict()
    return results


def run_container(impl: str, container: str, cases: list[dict]) -> dict[str, dict]:
    """Ship the adapter + a worker into ``container`` and run all cases there."""
    dest = "/tmp/parity_validate"
    bundle = {
        "tests/__init__.py": PH / "tests" / "__init__.py",
        "tests/parity/__init__.py": PKG / "__init__.py",
        "tests/parity/common.py": PKG / "common.py",
        "tests/parity/toolcalling/__init__.py": PKG / "toolcalling" / "__init__.py",
        f"tests/parity/toolcalling/{impl}.py": PKG / "toolcalling" / f"{impl}.py",
    }
    subprocess.run(
        ["docker", "exec", container, "bash", "-lc",
         f"rm -rf {dest} && mkdir -p {dest}/tests/parity/toolcalling"],
        check=True,
    )
    for rel, src in bundle.items():
        subprocess.run(["docker", "cp", str(src), f"{container}:{dest}/{rel}"], check=True)
    # Write the worker via stdin to avoid a host temp file.
    subprocess.run(
        ["docker", "exec", "-i", container, "bash", "-lc", f"cat > {dest}/worker.py"],
        input=_WORKER_SRC, text=True, check=True,
    )
    payload = "\n".join(json.dumps(_request(c)) for c in cases)
    out_path = f"{dest}/out.jsonl"
    proc = subprocess.run(
        ["docker", "exec", "-i", container, "env", f"PYTHONPATH={dest}",
         "python3", f"{dest}/worker.py", "--impl", impl, "--out", out_path],
        input=payload, capture_output=True, text=True,
    )
    cat = subprocess.run(
        ["docker", "exec", container, "cat", out_path],
        capture_output=True, text=True,
    )
    results: dict[str, dict] = {}
    for line in cat.stdout.splitlines():
        line = line.strip()
        if line:
            d = json.loads(line)
            results[d.pop("key")] = d
    if not results and proc.returncode != 0:
        sys.exit(f"worker failed in {container}:\n{proc.stderr[-2000:]}")
    return results


def engine_version(impl: str, container: str | None) -> str | None:
    pkg = "vllm" if impl == "vllm" else "sglang"
    code = f"import {pkg},sys; sys.stdout.write(getattr({pkg},'__version__',''))"
    try:
        if container:
            r = subprocess.run(["docker", "exec", container, "python3", "-c", code],
                               capture_output=True, text=True)
        else:
            r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        v = r.stdout.strip()
        return v or None
    except FileNotFoundError:
        return None


def compare(impl: str, spec: dict, got: dict | None) -> tuple[str, str]:
    if got is None:
        return ("fail", "no result returned by engine")
    got_err = got.get("error")
    if "error" in spec:
        if got_err and spec["error"] in got_err:
            return ("pass", "")
        return ("fail", f"expected error {spec['error']!r}, got {got_err!r}")
    if got_err:
        return ("fail", f"engine crashed: {got_err}")
    exp = ParseResult(calls=spec.get("calls", []), normal_text=spec.get("normal_text"))
    got_pr = ParseResult(calls=got.get("calls", []), normal_text=got.get("normal_text"))
    if canonical(got_pr.to_dict()) == canonical(exp.to_dict()):
        return ("pass", "")
    return ("fail",
            f"\n    expected: {canonical(exp.to_dict())}"
            f"\n    got:      {canonical(got_pr.to_dict())}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--impl", required=True, choices=("vllm", "sglang"))
    ap.add_argument("--fixtures", required=True, type=Path,
                    help="toolcalling fixtures dir staged by check.sh")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--container", help="run the engine in this docker container")
    mode.add_argument("--pip", action="store_true", help="run the engine in-process")
    args = ap.parse_args()

    cases = collect_cases(args.fixtures, args.impl)
    if not cases:
        print(f"no runnable {args.impl} cases under {args.fixtures}")
        return 0

    live = engine_version(args.impl, args.container)
    pin = pinned_version(args.impl)
    print(f"=== {args.impl} conformance ({'container ' + args.container if args.container else 'pip in-process'}) ===")
    print(f"engine version: {live or 'unknown'}   pinned (fixtures captured against): {pin or 'unknown'}")
    if live and pin and live != pin:
        print(f"WARNING: engine {live} != pin {pin} — diffs below may be version drift, not parser bugs.")

    got = (run_container(args.impl, args.container, cases) if args.container
           else run_pip(args.impl, cases))

    npass = nfail = 0
    fails: list[str] = []
    for c in cases:
        status, detail = compare(args.impl, c["spec"], got.get(c["key"]))
        if status == "pass":
            npass += 1
        else:
            nfail += 1
            fails.append(f"FAIL {c['key']} [{c['mode']}]{detail}")
    for f in fails:
        print(f)
    print(f"\n{args.impl} conformance: {npass}/{npass + nfail} cases passed"
          + (f", {nfail} failed" if nfail else ""))
    return 1 if nfail else 0


if __name__ == "__main__":
    raise SystemExit(main())
