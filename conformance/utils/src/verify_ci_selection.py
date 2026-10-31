# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reject missing or accidentally duplicated tests across CI groups."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

import yaml

from verify_publication_transport import publication_metadata, runtime_catalog


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


def verify_jobs(results, jobs, gate="conformance-table"):
    def dependencies(job):
        value = job.get("needs", [])
        return {value} if isinstance(value, str) else set(value)

    if any(dependencies(job) - jobs.keys() for job in jobs.values()):
        raise ValueError("publication workflow contains an unknown job dependency")
    # Downstream deployment jobs cannot qualify the artifact they consume.
    downstream = {gate}
    while True:
        expanded = downstream | {name for name, job in jobs.items() if dependencies(job) & downstream}
        if expanded == downstream:
            break
        downstream = expanded
    required = set(jobs) - downstream
    if not required or dependencies(jobs[gate]) != required or set(results) != required:
        raise ValueError("publication gate must depend on every validation job")
    failed = {name: result.get("result") for name, result in results.items() if result.get("result") != "success"}
    if failed:
        raise ValueError(f"publication validation did not succeed: {failed}")
    return {"validation_jobs": sorted(required)}


def verify(baseline, assignments, *, require_history=False):
    if require_history:
        for document in [*baseline, *assignments]:
            if document.get("include_history") is not True or document.get("coverage") != "exhaustive":
                raise ValueError("publication gate requires exhaustive history test assignments")
        if any(document.get("scope") is not None or document.get("group") is not None
               or document.get("selected") != document.get("eligible") for document in baseline):
            raise ValueError("publication baseline must collect every eligible test without filtering")
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


def verify_publication(site, revision):
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("publication gate requires a full commit revision")
    metadata = publication_metadata(site)
    catalog = runtime_catalog(site)
    if metadata.get("revision") != revision:
        raise ValueError("publication source revision differs from the checked commit")
    if catalog.get("capture_generation", {}).get("mode") != "history":
        raise ValueError("publication gate requires a history publication")
    return {"source_revision": revision, "publication_mode": "history"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--selections", type=Path)
    parser.add_argument("--shard-matrix", action="store_true", help="emit the complete qualification matrix for CI")
    parser.add_argument("--job-results", type=Path, help="validate GitHub needs results against the workflow graph")
    parser.add_argument("--workflow", type=Path, default=Path(__file__).resolve().parents[3] / ".github/workflows/ci.yml")
    parser.add_argument("--require-history", action="store_true")
    parser.add_argument("--publication", type=Path)
    parser.add_argument("--revision")
    args = parser.parse_args()
    if args.shard_matrix:
        if any((args.baseline, args.selections, args.job_results, args.require_history, args.publication, args.revision)):
            parser.error("--shard-matrix cannot be combined with test assignments")
        print(json.dumps(list(range(QUALIFICATION_SHARDS))))
        return
    if args.job_results is not None:
        if any((args.baseline, args.selections, args.require_history, args.publication, args.revision)):
            parser.error("--job-results cannot be combined with publication accounting")
        print(json.dumps(verify_jobs(json.loads(args.job_results.read_bytes()), yaml.safe_load(args.workflow.read_text())["jobs"])))
        return
    if args.baseline is None or args.selections is None:
        parser.error("--baseline and --selections are required for accounting")
    if bool(args.publication) != bool(args.revision) or args.publication and not args.require_history:
        parser.error("--publication and --revision require each other and --require-history")
    result = verify(read_assignments(args.baseline), read_assignments(args.selections), require_history=args.require_history)
    if args.publication:
        result.update(verify_publication(args.publication, args.revision))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
