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
import argparse, copy, sys
from pathlib import Path
import yaml
import yaml_fast  # noqa: F401 — routes safe_load/safe_dump through libyaml
# Re-exported: callers import load/version_key/split_sel from this module by name.
from fixture_corpus import load, load_corpus, split_sel, version_key  # noqa: F401

def resolve_docs(fixtures_root, select, corpus=None):
    """Resolve one version selection entirely in memory.

    Returns `(docs, folded)`: `docs` is {(family, filename): doc} for the whole staged
    tree, `folded` is the subset of those keys an overlay actually merged into. The
    fold used to run through the filesystem — load the staged file, merge, dump it
    back, once per impl per version layer — so each output file was parsed and
    re-emitted several times per run. Keeping the accumulator in memory does the
    identical merge with one parse of each source file and at most one dump.

    Pass `corpus` (from fixture_corpus.load_corpus) to resolve several selections out
    of one parse of the source tree."""
    root = Path(fixtures_root)
    if corpus is None:
        corpus = load_corpus(root)

    # 1) shared inputs are the base every impl folds into.
    docs = {key[1:]: copy.deepcopy(doc) for key, doc in corpus.items() if key[0] == "inputs"}

    # 2) for each selected impl, apply its version dirs ascending up to the target,
    #    merging expected.<impl>. Lowest applied dir is the full anchor.
    folded: set[tuple[str, str]] = set()
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
            for key, src_ov in corpus.items():
                if key[0] != vdir.name:
                    continue
                base_doc = docs.get(key[1:])
                if base_doc is None:
                    continue
                # deepcopy the WHOLE overlay doc in one call: expected blocks are
                # grafted into the base by reference, so a shared corpus would
                # otherwise alias one object into every resolved selection. Copying
                # the doc (not each value) keeps the object sharing the source file
                # encodes as YAML anchors, so the re-emitted file is unchanged.
                ov = copy.deepcopy(src_ov)
                for cid, oc in (ov.get("cases") or {}).items():
                    bc = (base_doc.get("cases") or {}).get(cid)
                    if bc is None or "expected" not in oc:
                        continue
                    bc.setdefault("expected", {})
                    for k, val in oc["expected"].items():
                        bc["expected"][k] = val
                folded.add(key[1:])

    return docs, folded

def resolve(fixtures_root, out, select, verbose=False):
    """Stage inputs/ + the selected per-impl versions into a flat tree at `out`.

    `select` is a list of "<impl>-<version>" targets. Importable for callers that
    need multiple version snapshots (e.g. the parity table's version radios)."""
    root = Path(fixtures_root); out = Path(out)
    docs, folded = resolve_docs(root, select)

    for (family, name), doc in docs.items():
        dst = out / family / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if (family, name) in folded:
            dst.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=4096))
        else:
            # No overlay touched this one, so it keeps the verbatim inputs/ text
            # (comments/anchors survive) — same as when the fold ran on disk and
            # simply never rewrote it.
            dst.write_text((root / "inputs" / family / name).read_text())

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
