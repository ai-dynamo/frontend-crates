#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Schema validation for the v2 overlay fixtures (audit D3).

The frontend-crate-owned v2 overlays (`fixtures-stream-v2/`,
`fixtures-batch-on-stream-v2/`) must use canonical impl keys only — `dynamo_v2`,
`vllm_python`, `sglang_python`, `vllm_rust` — never the legacy `dynamo`/`vllm`/`sglang`
spellings. The renderer accepts both via aliases (the legacy v1 corpus still
uses legacy keys), so legacy keys in a v2 overlay are silent drift; this validator
fails loudly on them. Also checks that a `captured_with` block exists once any peer
has captured output.

  python3 conformance/utils/validate_fixtures.py            # validate v2 overlays
  python3 conformance/utils/validate_fixtures.py --mode v2  # (same; explicit)
"""
import argparse
import glob
import os
import sys

import yaml

from impls import LEGACY_IMPL_ALIASES

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
V2_OVERLAY_GLOBS = (
    "conformance/toolcalling/fixtures-stream-v2/*/*.yaml",
    "conformance/toolcalling/fixtures-batch-on-stream-v2/*/*.yaml",
)
_LEGACY_KEYS = set(LEGACY_IMPL_ALIASES)  # {"dynamo", "vllm", "sglang"}
# Peer impls that, when present with captured output, require a captured_with stamp.
_CAPTURED_PEERS = ("vllm_python", "sglang_python", "vllm_rust")


def _legacy_key_paths(node, path=""):
    """Every dotted path at which a legacy impl key appears as a mapping key."""
    found = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key in _LEGACY_KEYS:
                found.append(f"{path}.{key}" if path else str(key))
            found.extend(_legacy_key_paths(value, f"{path}.{key}" if path else str(key)))
    elif isinstance(node, list):
        for i, value in enumerate(node):
            found.extend(_legacy_key_paths(value, f"{path}[{i}]"))
    return found


def _has_captured_peer_output(doc) -> bool:
    """True if any peer impl appears anywhere as a key (so a captured_with is owed)."""
    text = yaml.safe_dump(doc)
    return any(f"{peer}:" in text for peer in _CAPTURED_PEERS)


def validate_file(path: str) -> list[str]:
    doc = yaml.safe_load(open(path, encoding="utf-8")) or {}
    errors = [f"{path}: legacy impl key at {p}" for p in _legacy_key_paths(doc)]
    if _has_captured_peer_output(doc) and not doc.get("captured_with"):
        errors.append(f"{path}: peer output present but no captured_with block")
    return errors


def validate_overlays(root: str = REPO) -> list[str]:
    errors = []
    for pattern in V2_OVERLAY_GLOBS:
        for path in sorted(glob.glob(os.path.join(root, pattern))):
            errors.extend(validate_file(path))
    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", default="v2", choices=("v2",))
    ap.add_argument("--root", default=REPO)
    args = ap.parse_args()
    errors = validate_overlays(args.root)
    for e in errors:
        print(e, file=sys.stderr)
    print(f"validate_fixtures: {len(errors)} problem(s)", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
