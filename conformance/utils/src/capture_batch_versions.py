#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Re-capture the TC batch (v1) fixtures against a NEWER engine version and write
changed-only overlays, so the batch tab can compare peer versions (0.23.0 vs
0.24.0 vs 0.25.1) the way the stream tab already does via
capture_streamv2_versions.py.

The shared inputs (`fixtures-batch-v1/inputs/<family>/TOOLCALLING.batch*.yaml`)
carry each case's `model_text` + `tools` but no `expected` block. The ANCHOR is
the LOWEST vllm_python version dir folded onto those inputs (resolve_fixtures);
its `expected.vllm_python` is the baseline. This tool feeds each input case's
complete `model_text` through the vLLM batch parser IN THE CURRENT container —
the non-streaming `extract_tool_calls(model_output, request)` path, via the
canonical `tests/parity/toolcalling/vllm.py` adapter (the same one validate.py
and the fixture authors use) — diffs the `{calls, normal_text}` against the
anchor, and writes a changed-only overlay per family:

  fixtures-batch-v1/vllm_python-<version>/<family>/TOOLCALLING.batch*.yaml

Only cases whose captured output differs from the anchor are written; only those
cases. Cases the parser errors on in-container are logged and carried forward (no
overlay), never fabricated. resolve_fixtures.py folds these top-level version dirs
in ascending order, so the overlay is additive and append-only.

The version string is read from the container's `vllm.__version__` at capture
time, never hardcoded.

Usage:
  python3 capture_batch_versions.py                     # all batch families
  python3 capture_batch_versions.py --family qwen3_coder
  python3 capture_batch_versions.py --vllm-container vllm-localdev
"""
import argparse
import glob
import os
import shutil
import sys
import tempfile

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
# capture_batch_versions.py lives in conformance/utils/src/, so repo root is 3 up
# and the parity test package (tests.parity.*) lives under conformance/utils/.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
for p in (HERE, os.path.join(ROOT, "conformance", "utils")):
    if p not in sys.path:
        sys.path.insert(0, p)

import resolve_fixtures  # noqa: E402  (anchor resolution: version_key + resolve)
import validate  # noqa: E402  (run_container ships the parity adapter + worker)
from tests.parity.common import canonical  # noqa: E402  (whitespace-tolerant diff)


class _QuotedStr(str):
    """Version string forced to single-quoted YAML, to match the 0.24.0 overlays'
    `captured_with: {vllm_python: '0.24.0'}` shape."""


def _represent_quoted(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", str(data), style="'")


yaml.SafeDumper.add_representer(_QuotedStr, _represent_quoted)


def _lowest_vllm_dir(fixtures_root):
    """The lowest vllm_python-<version> dir = the full anchor (every batch case)."""
    vdirs = [
        d
        for d in glob.glob(os.path.join(fixtures_root, "vllm_python-*"))
        if os.path.isdir(d)
    ]
    if not vdirs:
        raise SystemExit(f"no vllm_python-* anchor under {fixtures_root}")
    return min(vdirs, key=lambda d: resolve_fixtures.version_key(os.path.basename(d).split("-", 1)[1]))


def _anchor_expected(fixtures_root, work):
    """Fold the lowest vllm_python version onto inputs and return
    {(family, basename): {cid: {calls, normal_text}}} baseline blocks."""
    lowest = os.path.basename(_lowest_vllm_dir(fixtures_root)).split("-", 1)[1]
    staged = os.path.join(work, "baseline")
    resolve_fixtures.resolve(fixtures_root, staged, [f"vllm_python-{lowest}"])
    baseline = {}
    for fp in sorted(glob.glob(os.path.join(staged, "*", "*.yaml"))):
        family = os.path.basename(os.path.dirname(fp))
        base = os.path.basename(fp)
        doc = yaml.safe_load(open(fp))
        for cid, case in (doc.get("cases") or {}).items():
            if not isinstance(case, dict):
                continue
            exp = (case.get("expected") or {}).get("vllm_python")
            if isinstance(exp, dict) and "unavailable" not in exp:
                baseline[(family, base, cid)] = {
                    "calls": exp.get("calls", []),
                    "normal_text": exp.get("normal_text"),
                }
    return lowest, baseline


def _collect_batch_cases(fixtures_root, families):
    """Every input batch case with a model_text, as a run_container job list plus
    a (family, basename)->[cid] index for overlay assembly."""
    inputs = os.path.join(fixtures_root, "inputs")
    cases, by_file = [], {}
    fam_dirs = families or sorted(
        d for d in os.listdir(inputs) if os.path.isdir(os.path.join(inputs, d))
    )
    for family in fam_dirs:
        for fp in sorted(glob.glob(os.path.join(inputs, family, "TOOLCALLING.batch*.yaml"))):
            base = os.path.basename(fp)
            doc = yaml.safe_load(open(fp))
            fam = doc["family"]
            for cid, case in (doc.get("cases") or {}).items():
                if not isinstance(case, dict) or "model_text" not in case:
                    continue
                key = f"{fam}/{base}/{cid}"
                cases.append({
                    "key": key,
                    "family": fam,
                    "mode": "batch",
                    "tools": case.get("tools"),
                    "model_text": case.get("model_text"),
                })
                by_file.setdefault((fam, base), []).append((cid, key))
    return cases, by_file


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", default=ROOT, help="repo root (default: src/../../..)")
    ap.add_argument("--family", action="append", help="capture only this family (repeatable)")
    ap.add_argument("--vllm-container", default="vllm-localdev")
    ap.add_argument("--work", help="work dir (default: a fresh temp dir)")
    args = ap.parse_args()

    fixtures_root = os.path.join(args.root, "conformance/toolcalling/fixtures-batch-v1")
    work = args.work or tempfile.mkdtemp(prefix="batchv1_ver_")
    os.makedirs(work, exist_ok=True)

    version = validate.engine_version("vllm", args.vllm_container)
    if not version:
        raise SystemExit(f"could not read vllm version from {args.vllm_container}")
    print(f"[vllm_python] engine version {version}", file=sys.stderr)

    lowest, baseline = _anchor_expected(fixtures_root, work)
    print(f"[vllm_python] anchor = vllm_python-{lowest} ({len(baseline)} baseline cases)",
          file=sys.stderr)

    cases, by_file = _collect_batch_cases(fixtures_root, args.family)
    if not cases:
        print("[vllm_python] no batch input cases; nothing to do", file=sys.stderr)
        return
    print(f"[vllm_python] capturing {len(cases)} batch cases in {args.vllm_container} "
          f"(1 import)...", file=sys.stderr)
    got = validate.run_container("vllm", args.vllm_container, cases)

    n_files = n_cases = n_errored = 0
    families_touched = set()
    for (family, base), cids in sorted(by_file.items()):
        changed = {}
        for cid, key in cids:
            cap = got.get(key)
            if cap is None:
                continue
            if cap.get("error"):
                n_errored += 1
                print(f"  [vllm_python] {family}/{base} {cid}: parser error, carried "
                      f"forward ({cap['error'][:120]})", file=sys.stderr)
                continue
            captured = {"calls": cap.get("calls", []), "normal_text": cap.get("normal_text")}
            base_exp = baseline.get((family, base, cid))
            if base_exp is not None and canonical(captured) == canonical(base_exp):
                continue
            # vLLM emits None for "no narration"; the corpus (anchor + 0.24.0
            # overlays) renders that as '' — normalize so the stored shape matches
            # (the diff decision above already treats ''/None equal via canonical).
            changed[cid] = {"expected": {"vllm_python": {
                "calls": cap.get("calls", []),
                "normal_text": cap.get("normal_text") or "",
            }}}
        if not changed:
            continue
        outdir = os.path.join(fixtures_root, f"vllm_python-{version}", family)
        os.makedirs(outdir, exist_ok=True)
        out = {
            "family": family,
            "mode": "batch",
            "captured_with": {"vllm_python": _QuotedStr(version)},
            "cases": changed,
        }
        outfp = os.path.join(outdir, base)
        with open(outfp, "w") as f:
            f.write("# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n")
            f.write("# SPDX-License-Identifier: Apache-2.0\n")
            f.write("# Version overlay (changed-only): cases where this impl@version diverges from baseline.\n")
            yaml.safe_dump(out, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
        n_files += 1
        n_cases += len(changed)
        families_touched.add(family)
        print(f"  [vllm_python] wrote {family}/{base} ({len(changed)} changed case(s))",
              file=sys.stderr)

    print(f"[vllm_python] version {version}: {n_cases} changed case(s) across {n_files} "
          f"file(s) in {len(families_touched)} family(ies); {n_errored} errored/carried-forward",
          file=sys.stderr)


if __name__ == "__main__":
    main()
