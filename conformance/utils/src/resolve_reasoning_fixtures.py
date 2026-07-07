#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve the versioned reasoning fixture snapshots into a flat fixtures tree for staging.

Layout under <root>/conformance/reasoning/fixtures-v1/:
  inputs/                  anchor fixtures (model_text + expected.dynamo for all families);
                           the LOWEST peer version's expected outputs live here too, so a
                           fresh checkout with no overlay dirs still renders the full table.
  vllm-<version>/          changed-only expected.vllm overrides for that vLLM version.
  sglang-<version>/        changed-only expected.sglang overrides for that SGLang version.

Resolution: copy inputs/ verbatim, then for each requested peer impl apply its version dirs
in ascending order up to and including the selected version, patching expected.<impl>.
Default select = latest available per impl. Readers (reasoning/table.py) consume the flat
output unchanged.
"""
import argparse, re, sys
from pathlib import Path
import yaml


def load(p):
    return yaml.safe_load(Path(p).read_text())


def version_key(ver: str):
    """Order versions like 0.5.12.post1 < 0.5.14 < 0.24.0 < 3.0.0."""
    m = re.match(r"(\d+(?:\.\d+)*)(?:[.-]?post(\d+))?", ver)
    release = tuple(int(x) for x in m.group(1).split(".")) if m else ()
    post = int(m.group(2)) if m and m.group(2) else 0
    return (release, post)


def split_impl_ver(dirname: str):
    """'vllm-0.23.0' -> ('vllm', '0.23.0'); impl names never contain '-'."""
    impl, _, ver = dirname.partition("-")
    return impl, ver


def resolve(fixtures_root, out, select, verbose=False):
    """Stage inputs/ + selected peer-version overlays into a flat tree at `out`.

    `select` is a list of "<impl>-<version>" targets (e.g. ['vllm-0.24.0', 'sglang-0.5.14']).
    Each impl's overlays are applied in ascending version order up to and including the
    selected version, patching expected.<impl> in each case."""
    root = Path(fixtures_root)
    out = Path(out)

    # 1. Copy inputs/ verbatim (the anchor — full expected.dynamo + lowest peer outputs).
    inputs = root / "inputs"
    for fp in inputs.glob("*/*.yaml"):
        dst = out / fp.parent.name / fp.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(fp.read_text())

    # 2. For each selected peer impl, apply its version overlay dirs in ascending order.
    for sel in select:
        impl, target = split_impl_ver(sel)
        target_k = version_key(target)
        vdirs = sorted(
            (
                (version_key(split_impl_ver(d.name)[1]), d)
                for d in root.glob(f"{impl}-*")
                if d.is_dir()
            ),
            key=lambda t: t[0],
        )
        applied = [(k, d) for k, d in vdirs if k <= target_k]
        if not applied:
            print(
                f"resolve_reasoning_fixtures: no version dirs for {impl} <= {target}, skipping",
                file=sys.stderr,
            )
            continue
        for _, vdir in applied:
            for ofp in vdir.glob("*/*.yaml"):
                tgt = out / ofp.parent.name / ofp.name
                if not tgt.exists():
                    continue
                base_doc = load(tgt)
                ov = load(ofp)
                for cid, oc in (ov.get("cases") or {}).items():
                    bc = (base_doc.get("cases") or {}).get(cid)
                    if bc is None or "expected" not in oc:
                        continue
                    bc.setdefault("expected", {})
                    for k, val in oc["expected"].items():
                        bc["expected"][k] = val
                tgt.write_text(
                    yaml.safe_dump(base_doc, sort_keys=False, allow_unicode=True, width=4096)
                )

    if verbose:
        print(
            f"resolve_reasoning_fixtures: staged {len(list(out.glob('*/*.yaml')))} files"
            f" (select: {select or 'none'})",
            file=sys.stderr,
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixtures-root", required=True, help="Path to fixtures-v1/")
    ap.add_argument("--out", required=True, help="Destination flat fixtures dir")
    ap.add_argument("--select", nargs="*", default=[], help="<impl>-<version> targets")
    a = ap.parse_args()
    resolve(a.fixtures_root, a.out, a.select, verbose=True)
