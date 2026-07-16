#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Refresh the DYNAMO capture dirs of the local fixture staging tree after an
intentional parser behavior change, without touching any peer (vLLM/SGLang)
capture. Peers need engine containers to re-capture; the Dynamo captures come
from this repo's own crates, so they can (and must) be refreshed whenever the
parser output changes — the parity tests compare the live parsers against them.

Modes (any subset; default all):
  batch            fixtures-batch-v1/dynamo_v1-<v1 crate ver>/    (expected.dynamo_v1,
                   via the record_dynamo_batch bin — the v1 batch parser).
  stream           fixtures-stream-v2/dynamo_v2-<v2 crate ver>/   (per-chunk
                   expected, via record_dynamo_stream — the v2 stream parser).
                   Both write the CURRENT crate version's dir and leave every
                   other version dir in place: old captures are history the
                   chart compares as candidates; readers resolve ascending.
  batch-on-stream  fixtures-batch-on-stream-v2/<family>/*.yaml    (the dynamo_v2
                   case blocks + captured_with stamp, via record_batch_via_stream —
                   the v2 stream parser fed each batch sample as one chunk).

The tree is the working copy that package_and_publish.py packages
(conformance/toolcalling/...). If a fixture tree is missing there, it is first
copied from the fixture cache (CONFORMANCE_FIXTURES_ROOT or
~/.cache/dynamo/conformance-fixtures) so peer data carries over unchanged.

Usage:
  python3 refresh_dynamo_captures.py                # all three modes
  python3 refresh_dynamo_captures.py batch stream   # a subset
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent  # conformance/utils/src -> repo root
TREE = ROOT / "conformance" / "toolcalling"
CARGO = os.environ.get("CARGO", "cargo").split()

SPDX = [
    "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.",
    "# SPDX-License-Identifier: Apache-2.0",
]

_FAMILIES = yaml.safe_load((HERE / "parser_families.yaml").read_text())["families"]
# Families the Dynamo v2 stream parser implements (the registry's single source
# of truth) — these get stream + batch-on-stream captures.
V2_FAMILIES = sorted(f for f, s in _FAMILIES.items() if s.get("dynamo_v2"))


def crate_version(cargo_toml: Path) -> str:
    m = re.search(r'^version\s*=\s*"([^"]+)"', cargo_toml.read_text(), re.MULTILINE)
    if not m:
        raise SystemExit(f"no version in {cargo_toml}")
    return m.group(1)


def cache_root() -> Path:
    env = os.environ.get("CONFORMANCE_FIXTURES_ROOT")
    if env:
        return Path(env)
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "dynamo" / "conformance-fixtures"


def ensure_tree(name: str) -> Path:
    """Return TREE/<name>, copying it from the fixture cache on first use."""
    dst = TREE / name
    if not (dst / "inputs").is_dir() and not any(dst.glob("*/*.yaml")):
        src = cache_root() / "toolcalling" / name
        if not src.is_dir():
            raise SystemExit(f"{src} not cached — run extract_fixtures.py first")
        print(f"[refresh] copying {src} -> {dst}")
        shutil.copytree(src, dst, dirs_exist_ok=True)
    return dst


def run_bin(crate: str, bin_name: str, args: list[str]) -> str:
    cmd = [*CARGO, "run", "-q", "-p", crate, "--bin", bin_name, "--", *args]
    return subprocess.run(
        cmd, cwd=ROOT, check=True, stdout=subprocess.PIPE, text=True
    ).stdout


def dump_yaml(data: dict, header: list[str] | None = None) -> str:
    body = yaml.safe_dump(
        data, sort_keys=False, allow_unicode=True, width=100000, default_flow_style=False
    )
    return "\n".join((header or []) + ["", body]) if header else body


def run_json_bin(crate: str, bin_name: str, payload: dict) -> dict:
    """Run a JSON-file-in / JSON-stdout-out recorder bin."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump(payload, tf)
        tmp = tf.name
    try:
        return json.loads(run_bin(crate, bin_name, [tmp]))
    finally:
        os.unlink(tmp)


def _sorted_value(v):
    """Recursively sort mapping keys so decoded call arguments are deterministic
    across captures (the v1 parser serializes arguments from a hash map)."""
    if isinstance(v, dict):
        return {k: _sorted_value(v[k]) for k in sorted(v)}
    if isinstance(v, list):
        return [_sorted_value(x) for x in v]
    return v


def _assemble_jail_chunks(chunks: list) -> dict:
    """Assemble record_dynamo_jail_stream per-chunk output into the legacy stream
    fixture's {calls, normal_text} block (concatenate per tool index, decode args)."""
    names: dict[int, str] = {}
    args: dict[int, str] = {}
    normal_text = ""
    for ch in chunks:
        normal_text += ch.get("normal_text") or ""
        for d in ch.get("deltas") or []:
            idx = d["index"]
            if d.get("name"):
                names[idx] = names.get(idx, "") + d["name"]
            if d.get("arguments"):
                args[idx] = args.get(idx, "") + d["arguments"]
    calls = []
    for idx in sorted(names):
        raw = args.get(idx, "")
        try:
            arguments = _sorted_value(json.loads(raw))
        except (json.JSONDecodeError, ValueError):
            arguments = raw
        calls.append({"name": names[idx], "arguments": arguments})
    return {"calls": calls, "normal_text": normal_text}


def refresh_batch(v1_ver: str) -> None:
    tree = ensure_tree("fixtures-batch-v1")
    inputs = tree / "inputs"
    # The current-version dir is (re)written in place; OLDER version dirs are
    # capture history and MUST stay — the chart compares them as candidates and
    # readers resolve versions ascending (never delete an existing version).
    out_root = tree / f"dynamo_v1-{v1_ver}"
    if out_root.is_dir():
        shutil.rmtree(out_root)
    header = SPDX + [
        f"# Full anchor for dynamo@{v1_ver}. expected.dynamo_v1 only."
    ]
    n_batch = n_stream = 0
    for fam_dir in sorted(p for p in inputs.iterdir() if p.is_dir()):
        family = fam_dir.name
        for fp in sorted(fam_dir.glob("TOOLCALLING.*.yaml")):
            src = yaml.safe_load(fp.read_text())
            mode = src.get("mode")
            cases_out = {}
            if mode == "batch":
                cases_in = {
                    cid: {"model_text": c["model_text"], "tools": c.get("tools") or []}
                    for cid, c in (src.get("cases") or {}).items()
                    if isinstance(c, dict) and c.get("model_text") is not None
                }
                if not cases_in:
                    continue
                rec = run_json_bin(
                    "dynamo-parsers",
                    "record_dynamo_batch",
                    {"family": family, "cases": cases_in},
                )
                cases_out = {
                    cid: {"expected": {"dynamo_v1": rec[cid]}} for cid in cases_in if cid in rec
                }
                n_batch += len(cases_out)
            elif mode == "stream":
                # Legacy v1 jail expected (assembled): feed the per-chunk delta_text
                # through JailedStream + the v1 batch parser, like the v1 page did.
                # Tool schemas ride along so the batch parse coerces argument types.
                cases_in = {
                    cid: {
                        "chunks": [ch.get("delta_text", "") for ch in (c.get("chunks") or [])],
                        "tools": c.get("tools") or [],
                    }
                    for cid, c in (src.get("cases") or {}).items()
                    if isinstance(c, dict) and c.get("chunks")
                }
                if not cases_in:
                    continue
                rec = run_json_bin(
                    "dynamo-parsers",
                    "record_dynamo_jail_stream",
                    {"family": family, "cases": cases_in},
                )
                cases_out = {
                    cid: {"expected": {"dynamo_v1": _assemble_jail_chunks(rec[cid])}}
                    for cid in cases_in
                    if cid in rec
                }
                n_stream += len(cases_out)
            if not cases_out:
                continue
            doc = {"family": family, "mode": mode, "cases": cases_out}
            dst = out_root / family / fp.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(dump_yaml(doc, header))
        print(f"[batch] {family}: recorded")
    print(f"[batch] wrote {out_root.name} ({n_batch} batch + {n_stream} jail-stream cases)")


def refresh_stream(v2_ver: str) -> None:
    tree = ensure_tree("fixtures-stream-v2")
    inputs = tree / "inputs"
    # The current-version dir is (re)written in place; OLDER version dirs
    # (earlier 0.x v2 captures, the v1-major jail references) are capture
    # history and MUST stay — never delete an existing version.
    out_root = tree / f"dynamo_v2-{v2_ver}"
    if out_root.is_dir():
        shutil.rmtree(out_root)
    n_cases = 0
    for family in V2_FAMILIES:
        fam_dir = inputs / family
        if not fam_dir.is_dir():
            print(f"[stream] {family}: no inputs, skipped")
            continue
        for fp in sorted(fam_dir.glob("TOOLCALLING.stream*.yaml")):
            src = yaml.safe_load(fp.read_text())
            extra = ["--text"] if family == "harmony_text" else []
            rec = json.loads(
                run_bin("dynamo-parsers-v2", "record_dynamo_stream", [str(fp), *extra])
            )
            cases_out = {}
            for cid, chunks in rec.items():
                out_chunks = []
                for ch in chunks:
                    entry = {"expected": ch.get("deltas") or []}
                    if ch.get("normal_text"):
                        entry["normal_text"] = ch["normal_text"]
                    out_chunks.append(entry)
                cases_out[cid] = {"chunks": out_chunks}
            n_cases += len(cases_out)
            doc = {
                "family": family,
                "mode": src.get("mode", "streamv2"),
                "captured_with": {"dynamo_v2": v2_ver},
                "cases": cases_out,
            }
            dst = out_root / family / fp.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(dump_yaml(doc, SPDX))
        print(f"[stream] {family}: recorded")
    print(f"[stream] wrote {out_root.name} ({n_cases} cases)")


def refresh_batch_on_stream(v2_ver: str) -> None:
    tree = ensure_tree("fixtures-batch-on-stream-v2")
    batch_inputs = ensure_tree("fixtures-batch-v1") / "inputs"
    for family in V2_FAMILIES:
        fam_dir = tree / family
        if not (batch_inputs / family).is_dir():
            print(f"[batch-on-stream] {family}: no batch inputs, skipped")
            continue
        rec = json.loads(
            run_bin(
                "dynamo-parsers-v2",
                "record_batch_via_stream",
                ["--family", family, "--root", str(batch_inputs)],
            )
        )
        if not fam_dir.is_dir():
            # First capture for this family (e.g. a newly registered v2 parser):
            # create the dir from the batch inputs' file layout, Dynamo-only.
            # Peer keys stay absent — the peers were never captured on
            # batch-on-stream for this family (needs engine containers) and the
            # renderer shows an absent impl as unavailable, not as clean output.
            for src in sorted((batch_inputs / family).glob("TOOLCALLING.batch*.yaml")):
                src_doc = yaml.safe_load(src.read_text())
                if src_doc.get("mode") != "batch":
                    continue
                cases = {
                    cid: {"dynamo_v2": rec[cid]}
                    for cid in (src_doc.get("cases") or {})
                    if cid in rec
                }
                if not cases:
                    continue
                doc = {
                    "family": family,
                    "mode": "batch-on-stream",
                    "captured_with": {"dynamo_v2": v2_ver},
                    "cases": cases,
                }
                fam_dir.mkdir(parents=True, exist_ok=True)
                (fam_dir / src.name).write_text(dump_yaml(doc))
            print(f"[batch-on-stream] {family}: created dir, recorded {len(rec)} cases")
            continue
        for fp in sorted(fam_dir.glob("TOOLCALLING.batch*.yaml")):
            doc = yaml.safe_load(fp.read_text())
            changed = False
            for cid, case in (doc.get("cases") or {}).items():
                if cid in rec and isinstance(case, dict):
                    # dynamo_v2 leads each case block, like the previously
                    # captured families.
                    case.pop("dynamo_v2", None)
                    new_case = {"dynamo_v2": rec[cid], **case}
                    doc["cases"][cid] = new_case
                    changed = True
            if changed:
                doc.setdefault("captured_with", {})["dynamo_v2"] = v2_ver
                fp.write_text(dump_yaml(doc))
        print(f"[batch-on-stream] {family}: folded {len(rec)} cases")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "modes", nargs="*", choices=[[], "batch", "stream", "batch-on-stream"],
        help="subset of captures to refresh (default: all)",
    )
    args = ap.parse_args()
    modes = args.modes or ["batch", "stream", "batch-on-stream"]

    v1_ver = crate_version(ROOT / "parsers" / "v1" / "Cargo.toml")
    v2_ver = crate_version(ROOT / "parsers" / "v2" / "Cargo.toml")
    print(f"[refresh] dynamo-parsers {v1_ver}, dynamo-parsers-v2 {v2_ver}")

    if "batch" in modes:
        refresh_batch(v1_ver)
    if "stream" in modes:
        refresh_stream(v2_ver)
    if "batch-on-stream" in modes:
        refresh_batch_on_stream(v2_ver)
    return 0


if __name__ == "__main__":
    sys.exit(main())
