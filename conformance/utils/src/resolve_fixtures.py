#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve the versioned fixture snapshots into a flat fixtures tree for staging.

Layout under <root>/conformance/toolcalling/fixtures-batch-v1/:
  inputs/                  shared case inputs (version-independent); no expected blocks
  <impl>-<version>/        per-impl expected. The LOWEST version present for an impl
                           is its full anchor (every case); higher versions are
                           changed-only overlays (diff against the anchor).

There is no "baseline" dir: the anchor is simply whichever version is lowest, so a
new lowest version can be added at any time without renaming anything.

Resolution: copy base inputs, then for each requested "<impl>-<version>" apply that
impl's version dirs in ascending version order up to and including the requested one,
merging each case's expected.<impl> block. Default select = latest per impl.
Readers/renderers consume the flat output unchanged.
"""
import argparse, re, sys
from pathlib import Path
import yaml
# PERF: route safe_load through libyaml's CSafeLoader (identical result, ~15x faster).
if hasattr(yaml, "CSafeLoader"):
    yaml.safe_load = lambda _s, _loader=yaml.CSafeLoader: yaml.load(_s, Loader=_loader)

def load(p): return yaml.safe_load(Path(p).read_text())

def version_key(ver: str):
    """Order versions like 0.5.12.post1 < 0.5.14 < 0.24.0 < 3.0.0."""
    m = re.match(r"(\d+(?:\.\d+)*)(?:[.-]?post(\d+))?", ver)
    release = tuple(int(x) for x in m.group(1).split(".")) if m else ()
    post = int(m.group(2)) if m and m.group(2) else 0
    return (release, post)

def split_sel(sel: str):
    """'vllm-0.24.0' -> ('vllm', '0.24.0'); impl names never contain '-'."""
    impl, _, ver = sel.partition("-")
    return impl, ver

def resolve(fixtures_root, out, select, verbose=False):
    """Stage inputs/ + the selected per-impl versions into a flat tree at `out`.

    `select` is a list of "<impl>-<version>" targets. Importable for callers that
    need multiple version snapshots (e.g. the parity table's version radios)."""
    root = Path(fixtures_root); out = Path(out)

    # 1) copy shared inputs verbatim (preserve text so comments/anchors survive)
    inputs = root / "inputs"
    for fp in inputs.glob("*/*.yaml"):
        dst = out / fp.parent.name / fp.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(fp.read_text())

    # 2) for each selected impl, apply its version dirs ascending up to the target,
    #    merging expected.<impl>. Lowest applied dir is the full anchor.
    for sel in select:
        impl, target = split_sel(sel)
        target_k = version_key(target)
        vdirs = sorted(
            ((version_key(split_sel(d.name)[1]), d) for d in root.glob(f"{impl}-*") if d.is_dir()),
            key=lambda t: t[0],
        )
        applied = [(k, d) for k, d in vdirs if k <= target_k]
        if not applied:
            print(f"resolve_fixtures: no version dirs for {impl} <= {target}, skipping", file=sys.stderr)
            continue
        for _, vdir in applied:
            for ofp in vdir.glob("*/*.yaml"):
                tgt = out / ofp.parent.name / ofp.name
                if not tgt.exists():
                    continue
                base_doc = load(tgt); ov = load(ofp)
                for cid, oc in (ov.get("cases") or {}).items():
                    bc = (base_doc.get("cases") or {}).get(cid)
                    if bc is None or "expected" not in oc:
                        continue
                    bc.setdefault("expected", {})
                    for k, val in oc["expected"].items():
                        bc["expected"][k] = val
                tgt.write_text(yaml.safe_dump(base_doc, sort_keys=False, allow_unicode=True, width=4096))

    if verbose:
        print(f"resolve_fixtures: staged {len(list(out.glob('*/*.yaml')))} files (select: {select or 'none'})", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--select", nargs="*", default=[],
                    help="target '<impl>-<version>' per impl, e.g. dynamo-3.0.0 vllm-0.24.0 sglang-0.5.14")
    a = ap.parse_args()
    resolve(a.fixtures_root, a.out, a.select, verbose=True)

if __name__ == "__main__":
    main()
