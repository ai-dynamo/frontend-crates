#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve the versioned TC stream-v2 fixtures into a flat tree for a selected
version set, mirroring resolve_fixtures.py for the batch corpus.

Layout under <root>/conformance/toolcalling/fixtures-stream-v2/ (batch convention —
no unversioned "baseline"; the anchor is whichever version is lowest, per impl):
  inputs/<family>/TOOLCALLING.streamv2.*.yaml        shared per-chunk delta_text
                                                     (+ finish_reason/tools) — NO expected
  <impl>-<version>/<family>/TOOLCALLING.streamv2.*.yaml
                                                     per-impl per-chunk `expected`
                                                     (+ `normal_text`); lowest version
                                                     is the full anchor, higher versions
                                                     are changed-only overlays.

Resolution mirrors resolve_fixtures.py: copy `inputs/` as the base tree, then for each
impl merge its version dirs ascending up to the target, folding that impl's per-chunk
`expected`/`normal_text` back into the shared chunks and stamping
`captured_with[impl] = target`. Every impl present is included at its LOWEST version by
default; `--select <impl>-<version>` bumps a specific impl to that version. So a
single-version impl (vllm_rust, dynamo_rust) needs no explicit select.
Readers (load_all_cases("streamv2")) consume the flat output unchanged.
"""
import argparse
import re
import sys
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


def split_sel(sel: str):
    """'vllm_python-0.24.0' -> ('vllm_python', '0.24.0'). Impl keys may contain '_'
    but the version token starts after the first '-'."""
    impl, _, ver = sel.partition("-")
    return impl, ver


def _impl_version_dirs(root: Path) -> dict[str, list[tuple]]:
    """{impl: [(version_key, version, dir), ...] ascending} discovered from the
    <impl>-<version>/ dirs (no hardcoded anchor)."""
    out: dict[str, list[tuple]] = {}
    for d in root.iterdir():
        if not d.is_dir() or d.name == "inputs" or "-" not in d.name:
            continue
        impl, ver = split_sel(d.name)
        out.setdefault(impl, []).append((version_key(ver), ver, d))
    for impl in out:
        out[impl].sort(key=lambda t: t[0])
    return out


def _merge_impl(base_doc, vdoc, impl):
    """Fold one impl's per-chunk expected/normal_text (from a version dir doc) into
    the shared base doc's chunks. Case-level `unavailable` is copied to the impl."""
    bcases = base_doc.setdefault("cases", {})
    for cid, vc in (vdoc.get("cases") or {}).items():
        bc = bcases.get(cid)
        if bc is None:
            continue
        # For a case this version's doc lists, replace the impl's prior state entirely
        # (clear any lower-version chunks/unavailable before applying this version's)
        # rather than merging field-by-field: Dynamo v1 3.0.0 and v2 0.1.11 are different
        # parsers, not a refinement. Cases NOT listed here keep the lower version (the
        # normal changed-only-overlay behavior for peers). The two dynamo dirs cover
        # identical case sets, so v1 cleanly supersedes v2 with no stale-v2 leakage.
        if "unavailable" in vc:
            bc.setdefault("unavailable", {})[impl] = vc["unavailable"]
            for ch in bc.get("chunks") or []:
                if isinstance(ch, dict):
                    (ch.get("expected") or {}).pop(impl, None)
                    if isinstance(ch.get("normal_text"), dict):
                        ch["normal_text"].pop(impl, None)
            continue
        if isinstance(bc.get("unavailable"), dict):
            bc["unavailable"].pop(impl, None)
        bchunks = bc.get("chunks") or []
        # Clear the impl from EVERY base chunk before applying this version's chunks.
        # A version doc may carry FEWER chunks than the base (the v1 jail records 2
        # chunks against a 6-chunk input while the v2 anchor emits in chunk 3); the
        # per-index overwrite below never reaches the tail chunks, so without this
        # clear the lower version's deltas survive there and assembly concatenates
        # both versions (the `get_weatherget_weather` doubling).
        for ch in bchunks:
            if isinstance(ch, dict):
                (ch.get("expected") or {}).pop(impl, None)
                if isinstance(ch.get("normal_text"), dict):
                    ch["normal_text"].pop(impl, None)
        for i, ve in enumerate(vc.get("chunks") or []):
            if i >= len(bchunks) or not isinstance(bchunks[i], dict):
                continue
            bchunks[i].setdefault("expected", {})[impl] = ve.get("expected") or []
            nt = ve.get("normal_text")
            if nt:
                bchunks[i].setdefault("normal_text", {})[impl] = nt
            elif isinstance(bchunks[i].get("normal_text"), dict):
                bchunks[i]["normal_text"].pop(impl, None)


def resolve(sv2_root, out, select, verbose=False):
    root = Path(sv2_root)
    out = Path(out)

    # 1) copy the shared inputs tree verbatim (the base every impl folds into).
    inputs = root / "inputs"
    for fp in inputs.glob("*/*.yaml"):
        dst = out / fp.parent.name / fp.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(fp.read_text())

    # 2) per-impl target: default = that impl's lowest version; --select bumps it.
    dirs = _impl_version_dirs(root)
    targets = {impl: vers[0][1] for impl, vers in dirs.items()}  # lowest by default
    for sel in select:
        impl, ver = split_sel(sel)
        if impl in dirs:
            targets[impl] = ver

    # 3) fold each impl's version dirs ascending up to its target into the base tree.
    for impl, target in targets.items():
        tk = version_key(target)
        applied = [(k, d) for k, _v, d in dirs[impl] if k <= tk]
        for _k, vdir in applied:
            for vfp in vdir.glob("*/*.yaml"):
                tgt = out / vfp.parent.name / vfp.name
                if not tgt.exists():
                    continue
                base_doc = load(tgt)
                _merge_impl(base_doc, load(vfp), impl)
                base_doc.setdefault("captured_with", {})[impl] = target
                tgt.write_text(
                    yaml.safe_dump(base_doc, sort_keys=False, allow_unicode=True, width=4096)
                )

    if verbose:
        n = len(list(out.glob("*/TOOLCALLING.streamv2.*.yaml")))
        print(f"resolve_stream_fixtures: staged {n} files (select: {select or 'defaults'})",
              file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures-root", required=True,
                    help="the fixtures-stream-v2 dir (inputs/ + <impl>-<version>/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--select", nargs="*", default=[],
                    help="bump an impl to a version, e.g. vllm_python-0.24.0 "
                         "sglang_python-0.5.14 (others default to their lowest)")
    a = ap.parse_args()
    resolve(a.fixtures_root, a.out, a.select, verbose=True)


if __name__ == "__main__":
    main()
