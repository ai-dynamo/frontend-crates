# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fail-closed CI profiles and deterministic, disjoint capture assignments."""
import argparse
import json
import os
from pathlib import Path

from capture_versions import capture_release_sort_key


# Only these presentation owners can use reduced historical coverage. Unknown
# paths, deletions and changes to this classifier require exhaustive proof.
PRESENTATION = frozenset({
    "index.html", "conformance/viewer/viewer.css",
    "conformance/utils/tests/test_browser_smoke.py", "conformance/utils/tests/test_browser_viewer.py",
})


def classify(paths):
    paths = list(paths)
    return "latest-two" if paths and all(path in PRESENTATION for path in paths) else "exhaustive"


def classify_changes(data):
    fields = data.rstrip("\0").split("\0")
    if len(fields) % 2 or any(status != "M" for status in fields[::2]):
        return "exhaustive"
    return classify(fields[1::2])


def select_resources(resources, *, profile=None, shard=None, shards=None):
    profile = profile or os.environ.get("CONFORMANCE_COVERAGE", "exhaustive")
    shard = int(os.environ.get("CONFORMANCE_PARITY_SHARD", "0")) if shard is None else shard
    shards = int(os.environ.get("CONFORMANCE_PARITY_SHARDS", "1")) if shards is None else shards
    if profile not in {"latest-two", "exhaustive"} or not 0 <= shard < shards:
        raise ValueError("invalid conformance coverage profile or shard")
    ordered = sorted(resources, key=lambda row: row["path"])
    if profile == "latest-two":
        # Native bundles include every family at each implementation/version.
        # Source snapshots additionally need independent family selection.
        groups = {}
        for row in ordered:
            key = (row["implementation"], row.get("mode", "unified"), row.get("family"))
            groups.setdefault(key, set()).add(row["version"])
        newest = {key: set(sorted(versions, key=capture_release_sort_key)[-2:]) for key, versions in groups.items()}
        ordered = [row for row in ordered if row["version"] in newest[
            (row["implementation"], row.get("mode", "unified"), row.get("family"))]]
    return [row for index, row in enumerate(ordered) if index % shards == shard]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, help="NUL-separated git diff paths; print coverage profile")
    parser.add_argument("--changes", type=Path, help="NUL-separated git diff --name-status; structural changes require exhaustive coverage")
    parser.add_argument("--catalog", type=Path, help="write selected resolved resource paths as JSON")
    args = parser.parse_args()
    if args.changes is not None:
        print(classify_changes(args.changes.read_text()))
    elif args.paths is not None:
        print(classify(args.paths.read_text().rstrip("\0").split("\0")))
    elif args.catalog is not None:
        catalog = json.loads(args.catalog.read_bytes())
        rows = [row for row in catalog["sources"]["unified"] if row["kind"] == "resolved-capture"]
        print(json.dumps([row["path"] for row in select_resources(rows)]))
    else:
        parser.error("--changes, --paths or --catalog is required")


if __name__ == "__main__":
    main()
