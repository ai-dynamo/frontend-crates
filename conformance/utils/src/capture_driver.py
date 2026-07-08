#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-side orchestrator for the conformance captures. Copies capture.py into the
engine containers, runs one batched capture per engine (one import per engine),
then assembles fixtures. Runs on the HOST (docker exec), not inside a container.

Modes (`--mode`):
  stream           Per-chunk vLLM Python + vLLM Rust + SGLang Python streaming for configured families;
                   captures into local fixture trees, then publish to HuggingFace via package_and_publish.py
                   (Dynamo parser v2 marked unavailable/TODO). Calls build_stream_fixtures.py.
  batch-on-stream  Each family's batch text through each engine's streaming parser;
                   captures into local fixture trees, then publish to HuggingFace via package_and_publish.py
                   (optionally with Dynamo Rust recorder JSON).
  merge            Merge the three per-engine flat stream-on-batch captures
                   (--dynamo-rust/--vllm-python/--sglang JSON) into the nested
                   harmony_batch_stream.json the older flow consumes.

Recipes:
  Prefer conformance/utils/capture.sh for normal fixture refreshes.
  This module is the lower-level orchestrator used by that wrapper.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from impls import PARSER_NOT_CAPTURED  # noqa: E402  (shared failure-marker contract, B11)

# family -> parser/detector name per engine, loaded from parser_families.yaml (B2 —
# single source of truth). None = no parser for this engine -> marked unavailable.
# Peer-capture families are those with a vLLM Python parser (the 18 text-format
# families); Harmony is captured via its own token flow and is excluded here.
_FAMILIES = yaml.safe_load(
    (Path(__file__).resolve().parent / "parser_families.yaml").read_text()
)["families"]
_PEER_FAMILIES = [f for f, s in _FAMILIES.items() if s.get("vllm_python")]
VLLM = {f: _FAMILIES[f]["vllm_python"] for f in _PEER_FAMILIES}
VLLM_RUST = {f: _FAMILIES[f]["vllm_rust"] for f in _PEER_FAMILIES if _FAMILIES[f].get("vllm_rust")}
SGLANG = {f: _FAMILIES[f].get("sglang_python") for f in _PEER_FAMILIES}
VLLM_RUST_UNAVAILABLE = (
    "vLLM Rust capture not implemented yet; source checkout is available for the Rust probe."
)


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def _vllm_rust_source_arg(args):
    return args.vllm_rust_source or os.environ.get("VLLM_RUST_SOURCE")


def _vllm_rust_source_version(source):
    if not source:
        return None
    root = Path(source).expanduser().resolve()
    crate = root / "rust/src/tool-parser/Cargo.toml"
    if not crate.exists():
        raise SystemExit(
            f"vLLM Rust source path {root} does not contain rust/src/tool-parser/Cargo.toml"
        )
    try:
        sha = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError:
        sha = "unknown"
    try:
        tag = subprocess.run(
            ["git", "-C", str(root), "describe", "--tags", "--exact-match"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError:
        tag = "untagged"
    return f"{tag} {sha}"


def _vllm_rust_unavailable(source_version):
    if source_version:
        return f"{VLLM_RUST_UNAVAILABLE} Source: {source_version}."
    return "vLLM Rust source not available; set VLLM_RUST_SOURCE or pass --vllm-rust-source."


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


def _vllm_rust_capture(source, mode, jobs, work):
    if not source:
        return None, {}
    here = os.path.dirname(os.path.abspath(__file__))
    batch = json.dumps([{"fixture": j["src"], "parser": j["parser"]} for j in jobs])
    proc = subprocess.run(
        [
            "python3",
            os.path.join(here, "capture_vllm_rust.py"),
            "--mode",
            mode,
            "--vllm-rust-source",
            source,
            "--batch",
            batch,
            "--work",
            work,
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode:
        raise RuntimeError(proc.stderr[-1000:] or proc.stdout[-1000:])
    data = json.loads(proc.stdout)
    return data["version"], {j["src"]: data["fixtures"].get(j["src"], {}) for j in jobs}


def _cpath(fp, mode):
    family = os.path.basename(os.path.dirname(fp))
    tag = "bos" if mode == "batch-on-stream" else "cap"
    return f"/tmp/{tag}_{family}_{os.path.basename(fp)}"


# --------------------------------------------------------------------------- #
# mode=stream
# --------------------------------------------------------------------------- #
def _impl_args(impl, family, parser, entry, version, work, tag, src):
    """Build build_stream_fixtures.py args for one impl: pass captured data, or an
    accurate `unavailable` reason (no parser registered vs. capture error)."""
    engine = {
        "vllm_rust": "vLLM Rust",
        "vllm_python": "vLLM Python",
        "sglang_python": "SGLang Python",
    }[impl]
    if parser is None:
        return ["--unavailable",
                f"{impl}=No {engine} parser for family '{family}'."]
    if "cases" in entry:
        f = os.path.join(work, f"{tag}_{impl}.json")
        json.dump(entry["cases"], open(f, "w"))
        flag = "--sglang" if impl == "sglang_python" else f"--{impl.replace('_', '-')}"
        return [flag, f, "--captured", f"{impl}={version}"]
    # Parser exists but capture errored (typically: requires the model tokenizer's
    # special tool tokens, which a stub tokenizer can't supply).
    err = (entry.get("error") or "capture failed").splitlines()[-1][:160]
    return ["--unavailable",
            f"{impl}={engine} '{parser}' {PARSER_NOT_CAPTURED} with a stub tokenizer: {err}"]


def _select_families(families, args):
    """B4: narrow an all-family list to a single `--family` for a tight capture
    loop. Errors if the family is unknown so a typo fails loudly."""
    fam = getattr(args, "family", None)
    if not fam:
        return families
    if fam not in families:
        raise SystemExit(f"--family {fam!r} not in capture set: {', '.join(families)}")
    return [fam]


def _select_fixtures(fixtures, args):
    """B4: narrow a family's fixtures to a single `--fixture <path>` when set."""
    target = getattr(args, "fixture", None)
    if not target:
        return fixtures
    target = os.path.abspath(target)
    return [fp for fp in fixtures if os.path.abspath(fp) == target]


def _run_stream(args):
    here = os.path.dirname(os.path.abspath(__file__))
    # A3: stream-capture SEEDS are the HF-downloaded v1 corpus's shared stream inputs
    # (`fixtures-batch-v1/inputs/<family>/TOOLCALLING.stream.*.yaml`) —
    # the chunking derives from the same model_text. Captured per-chunk output is
    # written locally, then published to HuggingFace via package_and_publish.py.
    # To add a new family's stream case, add its TOOLCALLING.stream.*.yaml seed
    # under the HF fixtures-batch-v1/inputs/ tree and re-publish.
    conf = os.path.join(args.root, "conformance/toolcalling/fixtures-batch-v1/inputs")
    _copy_worker((args.vllm_container, args.sglang_container))
    vllm_rust_source_version = _vllm_rust_source_version(_vllm_rust_source_arg(args))
    vllm_rust_source = _vllm_rust_source_arg(args)

    families = _select_families(sorted(VLLM.keys()), args)
    vllm_jobs, vllm_rust_jobs, sglang_jobs = [], [], []
    family_fixtures = {}
    for family in families:
        fixtures = sorted(glob.glob(f"{conf}/{family}/TOOLCALLING.stream.*.yaml"))
        fixtures = _select_fixtures(fixtures, args)
        family_fixtures[family] = fixtures
        for fp in fixtures:
            if VLLM[family]:
                vllm_jobs.append({"src": fp, "container_path": _cpath(fp, "stream"), "parser": VLLM[family]})
            if VLLM_RUST.get(family):
                vllm_rust_jobs.append({"src": fp, "parser": VLLM_RUST[family]})
            if SGLANG[family]:
                sglang_jobs.append({"src": fp, "container_path": _cpath(fp, "stream"), "parser": SGLANG[family]})

    print(f"capturing vllm ({len(vllm_jobs)} fixtures, 1 import)...", file=sys.stderr)
    vllm_ver, vllm_caps = _container_capture(args.vllm_container, "vllm", "stream", vllm_jobs, args.work)
    print(f"capturing vllm rust ({len(vllm_rust_jobs)} fixtures)...", file=sys.stderr)
    vllm_rust_ver, vllm_rust_caps = _vllm_rust_capture(
        vllm_rust_source, "stream", vllm_rust_jobs, args.work)
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
                   "--unavailable", f"dynamo_rust={args.dynamo_todo}"]
            if vllm_rust_source:
                cmd += _impl_args(
                    "vllm_rust", family, VLLM_RUST.get(family),
                    vllm_rust_caps.get(fp, {}), vllm_rust_ver or vllm_rust_source_version,
                    args.work, f"{family}_{base}", fp)
            else:
                cmd += ["--unavailable", f"vllm_rust={_vllm_rust_unavailable(vllm_rust_source_version)}"]
            cmd += _impl_args(
                "vllm_python", family, VLLM[family], vllm_caps.get(fp, {}), vllm_ver,
                args.work, f"{family}_{base}", fp)
            cmd += _impl_args(
                "sglang_python", family, SGLANG[family], sglang_caps.get(fp, {}), sglang_ver,
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
    engine = {
        "vllm_rust": "vLLM Rust",
        "vllm_python": "vLLM Python",
        "sglang_python": "SGLang Python",
    }[impl]
    if parser is None:
        return {"unavailable": f"No {engine} parser for family '{family}'."}
    if "cases" in entry:
        return entry["cases"]
    return {}


def _load_dynamo_rust(path):
    if not path:
        return {}
    with open(path) as f:
        return json.load(f)


def _dynamo_cases_for_family(data, family):
    if not data:
        return {}
    if family in data and isinstance(data[family], dict):
        return data[family]
    return data


def _write_overlay(src, outfp, vllm_entry, vllm_rust_entry, sglang_entry, versions, dynamo_rust):
    doc = yaml.safe_load(open(src))
    family = doc["family"]
    dynamo_cases = _dynamo_cases_for_family(dynamo_rust, family)
    out = {
        "family": family,
        "mode": "batch-on-stream",
        "captured_with": {
            "vllm_python": versions["vllm_python"],
            "sglang_python": versions["sglang_python"],
        },
        "cases": {},
    }
    if versions.get("vllm_rust"):
        out["captured_with"]["vllm_rust"] = versions["vllm_rust"]
    if dynamo_cases:
        out["captured_with"]["dynamo_rust"] = "Dynamo parser v2"

    vllm_parser = _parser_for("vllm", family)
    vllm_rust_parser = VLLM_RUST.get(family)
    sglang_parser = _parser_for("sglang", family)
    vllm_cases = _block_for("vllm_python", family, vllm_parser, vllm_entry)
    vllm_rust_cases = _block_for("vllm_rust", family, vllm_rust_parser, vllm_rust_entry)
    sglang_cases = _block_for("sglang_python", family, sglang_parser, sglang_entry)

    for cid, case in (doc.get("cases") or {}).items():
        row = {}
        if cid in dynamo_cases:
            row["dynamo_rust"] = dynamo_cases[cid]
        if not versions.get("vllm_rust"):
            row["vllm_rust"] = {
                "unavailable": _vllm_rust_unavailable(None)
            }
        elif vllm_rust_parser is None:
            row["vllm_rust"] = {
                "unavailable": f"No vLLM Rust parser for family '{family}'."
            }
        elif cid in vllm_rust_cases:
            row["vllm_rust"] = vllm_rust_cases[cid]
        elif "model_text" not in case:
            row["vllm_rust"] = {"unavailable": "No batch model_text for this case."}
        else:
            row["vllm_rust"] = {"unavailable": "Capture did not return this case."}
        for impl, parser, cases in (
            ("vllm_python", vllm_parser, vllm_cases),
            ("sglang_python", sglang_parser, sglang_cases),
        ):
            if parser is None:
                row[impl] = {
                    "unavailable": f"No {'vLLM Python' if impl == 'vllm_python' else 'SGLang'} parser for family '{family}'."
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
    vllm_rust_source_version = _vllm_rust_source_version(_vllm_rust_source_arg(args))
    vllm_rust_source = _vllm_rust_source_arg(args)
    fixture_root = os.path.join(args.root, "conformance/toolcalling/fixtures-batch-v1")
    sources = sorted(glob.glob(f"{fixture_root}/*/TOOLCALLING.batch*.yaml"))
    if getattr(args, "family", None):
        sources = [s for s in sources if os.path.basename(os.path.dirname(s)) == args.family]
    sources = _select_fixtures(sources, args)
    jobs = {"vllm": [], "vllm_rust": [], "sglang": []}
    for src in sources:
        family = os.path.basename(os.path.dirname(src))
        cpath = _cpath(src, "batch-on-stream")
        for impl in ("vllm", "sglang"):
            parser = _parser_for(impl, family)
            if parser:
                jobs[impl].append({"src": src, "container_path": cpath, "parser": parser})
        parser = VLLM_RUST.get(family)
        if parser:
            jobs["vllm_rust"].append({"src": src, "parser": parser})

    print(f"capturing vllm ({len(jobs['vllm'])} batch fixtures)...", file=sys.stderr)
    vllm_ver, vllm_caps = _container_capture(
        args.vllm_container, "vllm", "batch-on-stream", jobs["vllm"], args.work)
    print(f"capturing vllm rust ({len(jobs['vllm_rust'])} batch fixtures)...", file=sys.stderr)
    vllm_rust_ver, vllm_rust_caps = _vllm_rust_capture(
        vllm_rust_source, "batch-on-stream", jobs["vllm_rust"], args.work)
    print(f"capturing sglang ({len(jobs['sglang'])} batch fixtures)...", file=sys.stderr)
    sglang_ver, sglang_caps = _container_capture(
        args.sglang_container, "sglang", "batch-on-stream", jobs["sglang"], args.work)

    dynamo_rust = _load_dynamo_rust(args.dynamo_rust_json)
    versions = {"vllm_python": vllm_ver, "sglang_python": sglang_ver}
    if vllm_rust_ver or vllm_rust_source_version:
        versions["vllm_rust"] = vllm_rust_ver or vllm_rust_source_version
    out_root = os.path.join(args.root, "conformance/toolcalling/fixtures-batch-on-stream-v2")
    for src in sources:
        family = os.path.basename(os.path.dirname(src))
        outfp = os.path.join(out_root, family, os.path.basename(src))
        _write_overlay(
            src, outfp, vllm_caps.get(src, {}), vllm_rust_caps.get(src, {}),
            sglang_caps.get(src, {}),
            versions, dynamo_rust)
        print(f"  wrote {family}/{os.path.basename(src)}", file=sys.stderr)


# --------------------------------------------------------------------------- #
# mode=merge (was merge_batch_stream.py)
# --------------------------------------------------------------------------- #
def _run_merge(args):
    layers = {
        "dynamo_rust": json.load(open(args.dynamo_rust)),
        "vllm_python": json.load(open(args.vllm_python)),
        "sglang_python": json.load(open(args.sglang)),
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
    ap.add_argument("--family", help="B4: capture only this family (default: all)")
    ap.add_argument("--fixture", help="B4: capture only this fixture path (default: all)")
    ap.add_argument("--vllm-container", default="vllm-localdev")
    ap.add_argument("--sglang-container", default="sglang-localdev")
    ap.add_argument("--vllm-rust-source", help="vLLM source checkout root; defaults to VLLM_RUST_SOURCE")
    ap.add_argument("--dynamo-todo", help="stream: Dynamo unavailable/TODO reason")
    ap.add_argument("--dynamo-rust-json", help="batch-on-stream: Dynamo Rust recorder JSON")
    ap.add_argument("--dynamo-harmony-json", dest="dynamo_rust_json", help=argparse.SUPPRESS)
    # merge
    ap.add_argument("--dynamo", dest="dynamo_rust")
    ap.add_argument("--dynamo-rust")
    ap.add_argument("--vllm", dest="vllm_python")
    ap.add_argument("--vllm-python")
    ap.add_argument("--sglang")
    ap.add_argument("-o", "--output")
    args = ap.parse_args()

    # Per-mode required args (argparse can't express "required only for some modes").
    if args.mode == "merge":
        missing = [n for n in ("dynamo_rust", "vllm_python", "sglang", "output") if not getattr(args, n)]
        if missing:
            ap.error("--mode merge requires --dynamo-rust --vllm-python --sglang -o/--output")
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
