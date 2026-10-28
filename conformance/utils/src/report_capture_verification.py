#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Publish executed conformance comparisons independently of test-runner output."""

import argparse
import json
import os
from pathlib import Path


SUITES = {"unified-golden": False, "render-golden": False, "adapter-golden": False, "recorded-capture": True}


def annotation(value):
    return value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def report(directory, summary):
    rows = []
    failures = []
    for suite, blocking in SUITES.items():
        path = directory / f"{suite}.json"
        if not path.is_file():
            failures.append(f"Missing executed comparison: {suite}")
            continue
        document = json.loads(path.read_bytes())
        if (document["schema"] != 1 or document["suite"] != suite or document["blocking"] is not blocking
                or type(document["checked"]) is not int or document["checked"] <= 0
                or not isinstance(document["differences"], list)
                or any(not isinstance(item, str) for item in document["differences"])):
            raise ValueError(f"Invalid comparison receipt: {suite}")
        for variable, key in (("GITHUB_SHA", "revision"), ("CONFORMANCE_CANDIDATE_TREE", "candidate_tree")):
            if os.environ.get(variable) and document.get(key) != os.environ[variable]:
                raise ValueError(f"Stale comparison receipt: {suite}")
        differences = document["differences"]
        rows.append(f"| {suite} | {document['checked']} | {len(differences)} | {'error' if blocking else 'warning'} |")
        if differences:
            message = f"{suite}: {len(differences)} differences across {document['checked']} comparisons; see capture-verification artifact"
            print(f"::{'error' if blocking else 'warning'}::{annotation(message)}")
            if blocking:
                failures.append(message)
    if summary:
        with summary.open("a") as handle:
            handle.write("\nCapture verification\n\n| Comparison | Cases/executions checked | Differences | Severity |\n|---|---:|---:|---|\n")
            handle.write("\n".join(rows) + "\n")
            for failure in failures:
                handle.write(f"\n{failure}\n")
    for failure in failures:
        print(f"::error::{annotation(failure)}")
    return int(bool(failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    return report(args.directory, args.summary)


if __name__ == "__main__":
    raise SystemExit(main())
