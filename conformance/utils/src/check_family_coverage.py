#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check parser marker registration, leak detection, and popup coloring."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml
import markers as markers_mod

HERE = Path(__file__).resolve().parent
REGISTRY_PATH = HERE / "parser_families.yaml"

class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def fail(self, msg: str) -> None:
        self.failures.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)


def check_markers(report: Report, registry: dict, family: str) -> None:
    """Marker-registration lint: the family's grammar tokens must be declared in
    parser_families.yaml `markers:`, leak-detectable, and popup-colorizable."""
    declared = (registry.get("markers") or {}).get(family)
    if declared is None:
        report.fail(
            f"{family} markers: no `markers:` entry in parser_families.yaml — declare the"
            " family's grammar tokens (pairs/singletons/leak), or an explicit empty {} for"
            " markup-less grammars; without it the leak detector and popup colorizer are blind"
        )
        return
    pairs = declared.get("pairs") or []
    singletons = declared.get("singletons") or []
    leak = declared.get("leak") or []
    for token in [t for pair in pairs for t in pair] + list(singletons) + list(leak):
        if not markers_mod._TOOL_CALL_MARKUP_RE.search(token):
            report.fail(
                f"{family} markers: declared token {token!r} is NOT matched by the leak"
                " detector (markers._TOOL_CALL_MARKUP_RE) — a leak of this token would"
                " render as a clean green cell"
            )
    environment = {key: os.environ[key] for key in ("PATH", "LANG", "LC_ALL") if key in os.environ}
    result = subprocess.run(
        ["node", str(HERE.parents[1] / "viewer" / "qualification.js"), "markers"],
        input=json.dumps({"family": family, "markers": registry.get("markers") or {}}),
        text=True,
        capture_output=True,
        check=True,
        env=environment,
    )
    for reason in json.loads(result.stdout):
        report.fail(f"{family} markers: {reason} in the popup colorizer — register/classify it in src/assets/colorize.js")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", action="append", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    args = parser.parse_args()
    if not args.family and not args.all:
        parser.error("pass --family <family> or --all")
    registry = yaml.safe_load(args.registry.read_text())
    registered = set(registry["families"]) | {
        row.get("registry", family) for family, row in registry["unified"].items()
    }
    families = sorted(registered if args.all else set(args.family))
    report = Report()
    for family in families:
        if family not in registered:
            report.fail(f"{family}: family is not registered")
        else:
            check_markers(report, registry, family)
    for failure in report.failures:
        print(f"FAIL  {failure}")
    print(f"marker lint: {len(report.failures)} failure(s) across {len(families)} families")
    return int(bool(report.failures))


if __name__ == "__main__":
    sys.exit(main())
