#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Assert that report generation preserves evidence and creates no per-case YAML."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def snapshot(root: Path, *, include_history: bool = False) -> dict:
    conformance = root / "conformance"
    evidence = {}
    directories = [path for path in conformance.glob("fixtures*") if path.is_dir()]
    directories.extend(conformance / name for name in ("captures", "inputs") if (conformance / name).is_dir())
    if include_history:
        directories.append(conformance / "historic")
    for directory in sorted(directories):
        if directory.is_symlink():
            raise ValueError(f"evidence must not contain links: {directory}")
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"evidence must not contain links: {path}")
            if path.is_file():
                with path.open("rb") as source:
                    evidence[path.relative_to(root).as_posix()] = hashlib.file_digest(source, "sha256").hexdigest()
    yaml_paths = sorted(path.relative_to(root).as_posix() for directory in directories for path in directory.rglob("*.yaml") if path.is_file())
    return {"evidence": evidence, "yaml_paths": yaml_paths}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("snapshot", "check"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--include-history", action="store_true")
    args = parser.parse_args()
    current = snapshot(args.root.resolve(), include_history=args.include_history)
    if args.operation == "snapshot":
        args.inventory.write_text(json.dumps(current, sort_keys=True) + "\n")
        return 0
    expected = json.loads(args.inventory.read_bytes())
    if expected != current:
        old = expected["evidence"]
        new = current["evidence"]
        changed = sorted(path for path in old.keys() | new.keys() if old.get(path) != new.get(path))
        added_yaml = sorted(set(current["yaml_paths"]) - set(expected["yaml_paths"]))
        removed_yaml = sorted(set(expected["yaml_paths"]) - set(current["yaml_paths"]))
        raise ValueError(f"report generation changed evidence={changed}, created YAML={added_yaml}, removed YAML={removed_yaml}")
    print(f"permanent evidence unchanged: {len(current['evidence'])} files; no per-case YAML created")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
