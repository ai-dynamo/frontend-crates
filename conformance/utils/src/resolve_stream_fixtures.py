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
single-version impl (vllm_rust, dynamo_v2) needs no explicit select.
Readers (load_all_cases("streamv2")) consume the flat output unchanged.
"""
import argparse
import copy
import sys
from pathlib import Path

import yaml
import yaml_fast  # noqa: F401 — routes safe_load/safe_dump through libyaml
# Re-exported: test_render_invariants.py and the generator import version_key/load
# from this module by name.
from fixture_corpus import load, load_corpus, split_sel, version_key  # noqa: F401


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
        # `unavailable` (no such parser) and `exception` (parser ran and threw) are both
        # case-level, per-impl states that supersede any lower-version chunk data. They
        # are mutually exclusive for a given impl, so applying one clears the other.
        if "unavailable" in vc or "exception" in vc:
            state = "unavailable" if "unavailable" in vc else "exception"
            other = "exception" if state == "unavailable" else "unavailable"
            bc.setdefault(state, {})[impl] = vc[state]
            if isinstance(bc.get(other), dict):
                bc[other].pop(impl, None)
            for ch in bc.get("chunks") or []:
                if isinstance(ch, dict):
                    (ch.get("expected") or {}).pop(impl, None)
                    if isinstance(ch.get("normal_text"), dict):
                        ch["normal_text"].pop(impl, None)
            continue
        if isinstance(bc.get("unavailable"), dict):
            bc["unavailable"].pop(impl, None)
        if isinstance(bc.get("exception"), dict):
            bc["exception"].pop(impl, None)
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


def resolve_docs(sv2_root, select, corpus=None):
    """Resolve one version selection entirely in memory.

    Returns `(docs, folded)`: `docs` is {(family, filename): doc} for the whole staged
    tree, `folded` is the subset of those keys an impl overlay actually merged into.
    The fold used to run through the filesystem — load the staged file, merge, dump it
    back, once per impl per version layer — which meant every output file was parsed
    and re-emitted several times per run. Keeping the accumulator in memory does the
    identical merge with one parse of each source file and at most one dump.
    """
    root = Path(sv2_root)
    if corpus is None:
        corpus = load_corpus(root)

    # 1) the shared inputs tree is the base every impl folds into.
    docs = {key[1:]: copy.deepcopy(doc) for key, doc in corpus.items() if key[0] == "inputs"}

    # 2) per-impl target: default = that impl's lowest version; --select bumps it.
    dirs = _impl_version_dirs(root)
    targets = {impl: vers[0][1] for impl, vers in dirs.items()}  # lowest by default
    for sel in select:
        impl, ver = split_sel(sel)
        if impl in dirs:
            targets[impl] = ver

    # 3) fold each impl's version dirs ascending up to its target into the base tree.
    folded: set[tuple[str, str]] = set()
    for impl, target in targets.items():
        tk = version_key(target)
        for k, _v, vdir in dirs[impl]:
            if k > tk:
                continue
            for key, vdoc in corpus.items():
                if key[0] != vdir.name:
                    continue
                doc = docs.get(key[1:])
                if doc is None:
                    continue
                # deepcopy: _merge_impl grafts the overlay's own lists/dicts into the
                # base doc by reference, and with a shared corpus that would alias the
                # same objects into every resolved selection.
                _merge_impl(doc, copy.deepcopy(vdoc), impl)
                doc.setdefault("captured_with", {})[impl] = target
                folded.add(key[1:])
    return docs, folded


def resolve(sv2_root, out, select, verbose=False):
    root = Path(sv2_root)
    out = Path(out)
    docs, folded = resolve_docs(root, select)

    for (family, name), doc in docs.items():
        dst = out / family / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if (family, name) in folded:
            dst.write_text(
                yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=4096)
            )
        else:
            # No overlay touched this one, so it stays the verbatim inputs/ text —
            # same as when the fold ran on disk and simply never rewrote it.
            dst.write_text((root / "inputs" / family / name).read_text())

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
