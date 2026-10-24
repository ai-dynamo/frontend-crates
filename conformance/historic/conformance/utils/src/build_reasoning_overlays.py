#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture reasoning peer output at the CURRENT engines and write changed-only version
overlays, so the reasoning tabs can compare peer versions (old vs new) the way the
tool-calling tabs already do.

The `inputs/` tree is the ANCHOR — the older capture (vLLM 0.23.0 / SGLang 0.5.12.post1,
stamped in each fixture's `captured_with`). This tool runs each fixture's cases through
the current engine container (vllm-localdev / sglang-localdev), diffs each case's
`expected.<impl>` against the anchor, and writes a per-family overlay holding ONLY the
cases whose output changed:

  reasoning/fixtures-v1/<impl>-<version>/<family>/REASONING.<mode>.yaml
      cases: {cid: {expected: {<impl>: {reasoning_text, normal_text}}}}

resolve_reasoning_fixtures.py folds an overlay in when its version is selected and stamps
`captured_with.<impl> = <version>`. A case that reproduces unchanged gets no overlay entry
(the anchor already covers it); a family with zero changed cases gets no overlay file, and
an impl with no changed cases anywhere gets no version dir at all — i.e. "if the new
capture is still the same, there is no new version".

Cases the current container cannot run (e.g. the vLLM Mistral-tokenizer requirement) are
logged and skipped: their anchor output is carried forward, never overwritten with an
error.

Usage (from repo root, with the fixtures populated under conformance/reasoning/fixtures-v1/):
  python3 conformance/utils/src/build_reasoning_overlays.py \
      [--vllm-container vllm-localdev] [--sglang-container sglang-localdev] [--family F]
"""
import argparse
import glob
import os
import sys

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from capture_reasoning import (  # noqa: E402
    _CAPTURED_KEY,
    _blocks_match,
    _container_run,
    _load_family_maps,
)

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
_FIXROOT = os.path.join(_ROOT, "conformance", "reasoning", "fixtures-v1")
_INPUTS = os.path.join(_FIXROOT, "inputs")


_HEADER = (
    "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n"
    "# SPDX-License-Identifier: Apache-2.0\n"
)


def _write_overlay(impl: str, version: str, family: str, mode: str, changed: dict) -> str:
    """Write reasoning/fixtures-v1/<impl>-<version>/<family>/REASONING.<mode>.yaml holding
    only the changed cases' expected.<impl>. Returns the path."""
    d = os.path.join(_FIXROOT, f"{impl}-{version}", family)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"REASONING.{mode}.yaml")
    key = _CAPTURED_KEY[impl]
    doc = {
        "family": family,
        "mode": mode,
        "cases": {
            cid: {
                "expected": {
                    key: {
                        "reasoning_text": block.get("reasoning_text") or "",
                        "normal_text": block.get("normal_text") or "",
                    }
                }
            }
            for cid, block in changed.items()
        },
    }
    with open(path, "w") as f:
        f.write(_HEADER)
        f.write(f"# Changed-only {impl} {version} reasoning overlay (vs the inputs/ anchor).\n")
        f.write(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=4096))
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vllm-container", default="vllm-localdev")
    ap.add_argument("--sglang-container", default="sglang-localdev")
    ap.add_argument("--family")
    args = ap.parse_args(argv)

    vllm_map, sglang_map = _load_family_maps()
    fixtures = sorted(
        os.path.join(_INPUTS, d, f)
        for d in os.listdir(_INPUTS)
        for f in ("REASONING.batch.yaml", "REASONING.stream.yaml")
        if os.path.exists(os.path.join(_INPUTS, d, f))
    )
    # per impl: does ANY family have a changed case? (else drop the whole version dir)
    any_changed = {"vllm": False, "sglang": False}
    for fixture in fixtures:
        family = os.path.basename(os.path.dirname(fixture))
        if args.family and family != args.family:
            continue
        mode = "stream" if fixture.endswith("stream.yaml") else "batch"
        doc = yaml.safe_load(open(fixture)) or {}
        cases = doc.get("cases", {})
        for impl, fam_map, container in (
            ("vllm", vllm_map, args.vllm_container),
            ("sglang", sglang_map, args.sglang_container),
        ):
            parser = fam_map.get(family)
            if parser is None:
                continue
            try:
                captured = _container_run(container, impl, fixture, parser)
            except Exception as e:  # noqa: BLE001
                print(f"  {family:22s} {mode:6s} {impl:6s}: CAPTURE ERROR {e}")
                continue
            version = captured["version"]
            changed = {}
            errors = 0
            for cid, case in cases.items():
                if not isinstance(case, dict) or "expected" not in case:
                    continue
                expected = case["expected"].get(impl)
                if not isinstance(expected, dict) or "unavailable" in expected:
                    continue
                if "model_text" not in case and "chunks" not in case:
                    continue
                got = captured["cases"].get(cid)
                if not isinstance(got, dict) or "error" in got:
                    errors += 1
                    continue
                if not _blocks_match(got, expected):
                    changed[cid] = got
            if changed:
                path = _write_overlay(impl, version, family, mode, changed)
                any_changed[impl] = True
                print(f"  {family:22s} {mode:6s} {impl:6s} v{version}: {len(changed)} changed"
                      f"{f', {errors} uncapturable' if errors else ''} -> {os.path.relpath(path, _ROOT)}")
            else:
                print(f"  {family:22s} {mode:6s} {impl:6s} v{version}: unchanged"
                      f"{f' ({errors} uncapturable)' if errors else ''}")

    for impl, changed in any_changed.items():
        if not changed:
            print(f"{impl}: no changed cases anywhere — no new version dir (new == old)")


if __name__ == "__main__":
    main()
