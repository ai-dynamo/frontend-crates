#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Re-capture the REASONING fixtures against a NEWER engine version and write
changed-only overlays, so the reasoning tab can compare peer versions (0.23.0 vs
0.24.0 vs 0.25.1) the way the TC batch/stream tabs already do.

The shared inputs (`reasoning/fixtures-v1/inputs/<family>/REASONING.{batch,stream}.yaml`)
carry each case's `model_text` / `chunks` AND an `expected.vllm_python` block
captured against the LOWEST version (`captured_with.vllm_python`, currently
0.23.0). That baseline block is the ANCHOR. This tool feeds each input case
through the vLLM reasoning parser IN THE CURRENT container (reusing
capture_reasoning.py's worker, which emits {reasoning_text, normal_text} per
case), diffs against the anchor, and writes a changed-only overlay per family:

  reasoning/fixtures-v1/vllm_python-<version>/<family>/REASONING.{batch,stream}.yaml

Only cases whose captured output differs from the anchor are written; only those
cases. Empty-string and None are treated as the same "absent text" for the diff
(capture_reasoning._blocks_match), matching the parity harness. Cases the parser
errors on in-container are logged and carried forward (no overlay), never
fabricated. resolve_fixtures.py folds these top-level version dirs in ascending
order, so the overlay is additive and append-only.

The version string is read from the container's `vllm.__version__` at capture
time, never hardcoded.

Usage:
  python3 capture_reasoning_versions.py                  # all reasoning families
  python3 capture_reasoning_versions.py --family qwen3
  python3 capture_reasoning_versions.py --vllm-container vllm-localdev
"""
import argparse
import glob
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
# capture_reasoning_versions.py lives in conformance/utils/src/, repo root is 3 up;
# the parity test package (tests.parity.*) lives under conformance/utils/.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
for p in (HERE, os.path.join(ROOT, "conformance", "utils")):
    if p not in sys.path:
        sys.path.insert(0, p)

import capture_reasoning as cr  # noqa: E402  (worker + _container_run + _blocks_match)
from tests.parity.common import _FAMILY_TO_VLLM_REASONING  # noqa: E402


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", default=ROOT, help="repo root (default: src/../../..)")
    ap.add_argument("--family", action="append", help="capture only this family (repeatable)")
    ap.add_argument("--vllm-container", default="vllm-localdev")
    args = ap.parse_args()

    fixtures_root = os.path.join(args.root, "conformance/reasoning/fixtures-v1")
    inputs = os.path.join(fixtures_root, "inputs")

    fam_dirs = args.family or sorted(
        d for d in os.listdir(inputs) if os.path.isdir(os.path.join(inputs, d))
    )

    n_files = n_cases = n_errored = 0
    version = None
    families_touched = set()
    no_parser = []
    for family in fam_dirs:
        parser = _FAMILY_TO_VLLM_REASONING.get(family)
        if parser is None:
            no_parser.append(family)
            continue
        for mode in ("batch", "stream"):
            fixture = os.path.join(inputs, family, f"REASONING.{mode}.yaml")
            if not os.path.exists(fixture):
                continue
            try:
                captured = cr._container_run(args.vllm_container, "vllm", fixture, parser)
            except Exception as e:  # noqa: BLE001 - whole-fixture failure, carry forward
                n_errored += 1
                print(f"  [vllm_python] {family}/REASONING.{mode}: capture error, carried "
                      f"forward ({str(e)[:120]})", file=sys.stderr)
                continue
            version = captured["version"]
            doc = yaml.safe_load(open(fixture))
            changed = {}
            for cid, case in (doc.get("cases") or {}).items():
                if not isinstance(case, dict) or "expected" not in case:
                    continue
                base = (case.get("expected") or {}).get("vllm_python")
                if not isinstance(base, dict) or "unavailable" in base:
                    continue
                if "model_text" not in case and "chunks" not in case:
                    continue
                cap = captured["cases"].get(cid)
                if cap is None:
                    continue
                if "error" in cap:
                    n_errored += 1
                    print(f"  [vllm_python] {family}/REASONING.{mode} {cid}: parser error, "
                          f"carried forward ({cap['error'][:120]})", file=sys.stderr)
                    continue
                if cr._blocks_match(cap, base):
                    continue
                # Render "absent text" as '' (None -> '') to match the anchor +
                # 0.24.0 overlay shape; _blocks_match above already treats ''/None
                # equal for the diff decision.
                changed[cid] = {"expected": {"vllm_python": {
                    "reasoning_text": cap.get("reasoning_text") or "",
                    "normal_text": cap.get("normal_text") or "",
                }}}
            if not changed:
                continue
            outdir = os.path.join(fixtures_root, f"vllm_python-{version}", family)
            os.makedirs(outdir, exist_ok=True)
            out = {"family": family, "mode": mode, "cases": changed}
            outfp = os.path.join(outdir, f"REASONING.{mode}.yaml")
            with open(outfp, "w") as f:
                f.write("# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n")
                f.write("# SPDX-License-Identifier: Apache-2.0\n")
                f.write(f"# Changed-only vllm {version} reasoning overlay (vs the inputs/ anchor).\n")
                yaml.safe_dump(out, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
            n_files += 1
            n_cases += len(changed)
            families_touched.add(family)
            print(f"  [vllm_python] wrote {family}/REASONING.{mode} "
                  f"({len(changed)} changed case(s))", file=sys.stderr)

    if no_parser:
        print(f"[vllm_python] no vLLM reasoning parser (skipped): {', '.join(no_parser)}",
              file=sys.stderr)
    print(f"[vllm_python] version {version}: {n_cases} changed case(s) across {n_files} "
          f"file(s) in {len(families_touched)} family(ies); {n_errored} errored/carried-forward",
          file=sys.stderr)


if __name__ == "__main__":
    main()
