# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reject missing or accidentally duplicated tests across CI groups."""
import argparse
from collections import Counter
import json
from pathlib import Path


QUALIFICATION_SHARDS = 8

QUALIFICATION = {
    "test_bundled_history_json_snapshots_match_original_yaml",
    "test_shared_javascript_matrix_contracts",
    "test_full_corpus_json_snapshots_match_python_and_javascript_resolution",
    "test_reference_backfills_require_request_bound_version_receipts",
    "test_structural_request_is_bound_in_every_retained_dynamo_column",
}


def read_assignments(directory):
    paths = sorted(directory.glob("*.json"))
    if not paths:
        raise ValueError(f"missing test assignment manifests: {directory}")
    return [json.loads(path.read_bytes()) for path in paths]


def verify(baseline, assignments):
    expected = {node for document in baseline for node in document["selected"]}
    observed = Counter(node for document in assignments for node in document["selected"])
    missing = sorted(expected - observed.keys())
    extra = sorted(observed.keys() - expected)
    wrong_counts = {node: count for node, count in observed.items()
                    if count != (QUALIFICATION_SHARDS if node.rsplit("::", 1)[-1] in QUALIFICATION else 1)}
    if not expected or missing or extra or wrong_counts:
        raise ValueError(f"CI selection mismatch: missing={missing}, extra={extra}, counts={wrong_counts}")
    return {"unique_tests": len(expected), "test_assignments": sum(observed.values()),
            "qualification_shards": QUALIFICATION_SHARDS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--selections", type=Path)
    parser.add_argument("--shard-matrix", action="store_true", help="emit the complete qualification matrix for CI")
    args = parser.parse_args()
    if args.shard_matrix:
        if args.baseline is not None or args.selections is not None:
            parser.error("--shard-matrix cannot be combined with test assignments")
        print(json.dumps(list(range(QUALIFICATION_SHARDS))))
        return
    if args.baseline is None or args.selections is None:
        parser.error("--baseline and --selections are required for accounting")
    print(json.dumps(verify(read_assignments(args.baseline), read_assignments(args.selections))))


if __name__ == "__main__":
    main()
