#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Re-capture the TC stream-v2 fixtures against a NEWER engine version and write
changed-only overlays, so the stream tab can compare peer versions (0.23.0 vs
0.24.0, 0.5.12.post1 vs 0.5.14) the way the batch tab already does.

The HF-downloaded anchor fixtures in `fixtures-stream-v2/inputs/<family>/TOOLCALLING.streamv2.*.yaml`
are the ANCHOR (the older version: vLLM 0.23.0 / SGLang 0.5.12.post1). This tool
feeds each anchor fixture's chunks back through the parser IN THE CURRENT container
(one engine import per container, via capture_driver's plumbing), diffs each chunk's
`expected.<impl>` / `normal_text.<impl>` against the anchor, and writes a
changed-only overlay per family:

  fixtures-stream-v2/overlays/<impl>-<version>/<family>/TOOLCALLING.streamv2.*.yaml

Only cases with at least one differing chunk are written; only differing chunks
are recorded. Cases that error in-container are logged and carried forward (no
overlay), never fabricated. `resolve_stream_fixtures.py` reconstructs a flat tree
for a selected version set from anchor + overlays.

Usage:
  python3 capture_streamv2_versions.py                       # all peer families, both engines
  python3 capture_streamv2_versions.py --family hermes       # one family
  python3 capture_streamv2_versions.py --impl vllm_python     # one engine only
"""
import argparse
import glob
import os
import sys
import tempfile

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import capture_driver as cd  # noqa: E402  (parser maps + container-capture plumbing)

# Engine impl -> (fixture expected key, parser-family-map, container arg).
_ENGINES = {
    "vllm_python": {"map": cd.VLLM, "impl_arg": "vllm", "container": "vllm-localdev"},
    "sglang_python": {"map": cd.SGLANG, "impl_arg": "sglang", "container": "sglang-localdev"},
}


def _norm_deltas(deltas):
    """Canonical form for one chunk's delta list, for equality comparison. YAML and
    JSON both load id/name/arguments the same way, so plain dict/list equality
    works once both sides are lists of dicts."""
    return [dict(d) for d in (deltas or []) if isinstance(d, dict)]


def _anchor_chunk_impl(chunk, impl):
    """(deltas, normal_text) the anchor recorded for `impl` at this chunk."""
    exp = chunk.get("expected") or {}
    nt = chunk.get("normal_text") or {}
    return _norm_deltas(exp.get(impl)), (nt.get(impl) or "")


def _build_overlay(anchor_doc, captured_cases, impl):
    """Return {cid: {chunk_idx: {expected, normal_text}}} for cases where the newly
    captured per-chunk output differs from the anchor, plus a list of skipped
    (errored) cids. Only differing chunks are recorded."""
    overlay_cases = {}
    errored = []
    for cid, case in (anchor_doc.get("cases") or {}).items():
        # An anchor case may mark this impl unavailable (no parser / stub-tokenizer
        # failure). Those are not re-captured here; carry the anchor forward.
        if impl in (case.get("unavailable") or {}):
            continue
        cap = captured_cases.get(cid)
        if cap is None:
            continue
        anchor_chunks = case.get("chunks") or []
        if any(isinstance(c, dict) and c.get("error") for c in cap):
            errored.append(cid)
            continue
        changed = {}
        for idx, anchor_chunk in enumerate(anchor_chunks):
            if not isinstance(anchor_chunk, dict) or idx >= len(cap):
                continue
            a_deltas, a_nt = _anchor_chunk_impl(anchor_chunk, impl)
            c_deltas = _norm_deltas(cap[idx].get("deltas"))
            c_nt = cap[idx].get("normal_text") or ""
            if c_deltas != a_deltas or c_nt != a_nt:
                changed[idx] = {"expected": c_deltas, "normal_text": c_nt}
        if changed:
            overlay_cases[cid] = changed
    return overlay_cases, errored


def _overlay_dir(out_root, impl, version, family):
    """overlays/<impl>-<version>/<family>/ — one overlay file per anchor stream file
    so the flat basenames line up on resolve."""
    outdir = os.path.join(out_root, "overlays", f"{impl}-{version}", family)
    os.makedirs(outdir, exist_ok=True)
    return outdir


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.dirname(HERE))),
                    help="repo root (default: three levels above this script: src/../../..)")
    ap.add_argument("--family", help="capture only this family (default: all peer families)")
    ap.add_argument("--impl", choices=tuple(_ENGINES), help="capture only this engine")
    ap.add_argument("--vllm-container", default="vllm-localdev")
    ap.add_argument("--sglang-container", default="sglang-localdev")
    ap.add_argument("--work", help="work dir (default: a fresh temp dir)")
    args = ap.parse_args()

    sv2_root = os.path.join(args.root, "conformance/toolcalling/fixtures-stream-v2")
    work = args.work or tempfile.mkdtemp(prefix="streamv2_ver_")
    os.makedirs(work, exist_ok=True)
    _ENGINES["vllm_python"]["container"] = args.vllm_container
    _ENGINES["sglang_python"]["container"] = args.sglang_container

    families = [args.family] if args.family else sorted(cd.VLLM.keys())
    impls = [args.impl] if args.impl else list(_ENGINES)

    # Collect the anchor fixture files per family (each carries chunks + tools that
    # feed capture.py --mode stream directly).
    anchor_files = {}
    for family in families:
        fs = sorted(glob.glob(os.path.join(sv2_root, family, "TOOLCALLING.streamv2.*.yaml")))
        if fs:
            anchor_files[family] = fs

    for impl in impls:
        spec = _ENGINES[impl]
        fam_map = spec["map"]
        jobs, job_family = [], {}
        for family, files in anchor_files.items():
            parser = fam_map.get(family)
            if not parser:
                continue
            for fp in files:
                jobs.append({"src": fp, "container_path": cd._cpath(fp, "stream"), "parser": parser})
                job_family[fp] = family
        if not jobs:
            print(f"[{impl}] no fixtures with a parser; skipping", file=sys.stderr)
            continue
        cd._copy_worker((spec["container"],))
        print(f"[{impl}] capturing {len(jobs)} fixtures in {spec['container']} "
              f"(1 import)...", file=sys.stderr)
        version, caps = cd._container_capture(
            spec["container"], spec["impl_arg"], "stream", jobs, work)
        print(f"[{impl}] engine version {version}", file=sys.stderr)

        n_files_changed = n_cases_changed = n_errored = 0
        families_touched = set()
        for fp, family in job_family.items():
            entry = caps.get(fp, {})
            if "cases" not in entry:
                # whole-fixture capture failure (e.g. import/parser construction)
                print(f"  [{impl}] {family}/{os.path.basename(fp)}: "
                      f"capture error, carried forward ({entry.get('error', '?')[:120]})",
                      file=sys.stderr)
                n_errored += 1
                continue
            anchor_doc = yaml.safe_load(open(fp))
            overlay_cases, errored = _build_overlay(anchor_doc, entry["cases"], impl)
            n_errored += len(errored)
            if errored:
                print(f"  [{impl}] {family}/{os.path.basename(fp)}: errored cases "
                      f"carried forward: {', '.join(errored)}", file=sys.stderr)
            if not overlay_cases:
                continue
            outdir = _overlay_dir(sv2_root, impl, version, family)
            out = {
                "family": family,
                "mode": "streamv2-overlay",
                "overlay_impl": impl,
                "overlay_version": version,
                # {cid: {chunk_index: {expected: [...], normal_text: '...'}}}
                "cases": overlay_cases,
            }
            outfp = os.path.join(outdir, os.path.basename(fp))
            with open(outfp, "w") as f:
                yaml.safe_dump(out, f, allow_unicode=True, sort_keys=False, width=4096)
            n_files_changed += 1
            n_cases_changed += len(overlay_cases)
            families_touched.add(family)
            print(f"  [{impl}] wrote overlay {family}/{os.path.basename(fp)} "
                  f"({len(overlay_cases)} changed case(s))", file=sys.stderr)

        print(f"[{impl}] version {version}: {n_cases_changed} changed case(s) across "
              f"{n_files_changed} file(s) in {len(families_touched)} family(ies); "
              f"{n_errored} errored/carried-forward", file=sys.stderr)


if __name__ == "__main__":
    main()
