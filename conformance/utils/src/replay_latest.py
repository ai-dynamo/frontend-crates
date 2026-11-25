#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replay complete Latest observations without consulting retained history."""

import argparse
from collections import Counter
import json
from pathlib import Path

import yaml

import capture_dynamo_unified_batch as batch
import capture_sources
from capture_stimulus import (
    execution_outcome, select_peer_observation, validate_capture_input,
    validate_execution,
)
from capture_unified_job import build_job
from impls import parser_families_path
import normalized_yaml
from numeric_cases import canonical_arguments, canonical_events
import unified_history
from viewer_codec import typed_equal


ROOT = Path(__file__).resolve().parents[3]


def latest_records(root, family, implementation, *, catalog=None):
    directory = root / capture_sources.STORE / "families" / family
    authored = yaml.safe_load((directory / "inputs_and_golden.yaml").read_bytes())
    expanded = normalized_yaml.expand_family(authored)
    owner = unified_history._validate_family(directory / "inputs_and_golden.yaml", expanded)
    cases = expanded["cases"]
    catalog = capture_sources.current(root) if catalog is None else catalog
    rows = [row for row in catalog["captures"]
            if row["surface"] == "unified" and row["family"] == family and row["implementation"] == implementation]
    if len(rows) != 1:
        raise ValueError("Latest has no unique recorded selection")
    raw = yaml.safe_load(capture_sources.checked_bytes(root, rows[0]))
    selection = raw[capture_sources.SELECTION]
    if selection["family"] != family or selection["implementation"] != implementation:
        raise ValueError("Latest selection identity mismatch")
    snapshot = normalized_yaml.expand_snapshot(raw, authored)
    unified_history._validate_complete_snapshot(
        directory / f"{implementation}.yaml", snapshot, owner, implementation, selection["version"],
    )
    records = {}
    for key, item in snapshot["records"].items():
        if key not in cases or cases[key].get("lifecycle", "active") == "retired":
            continue
        record, metadata = unified_history.materialized_record(cases[key], item["value"])
        record.update(metadata.get("record_metadata", {}))
        records[key] = select_peer_observation(
            record, cases[key]["request"], implementation=implementation,
            version=selection["version"],
        )
    return selection["version"], records


def compare_results(job, document, recorded, implementation, version, job_bytes):
    """Both batch and stream compare requests, outcomes and their real fields."""
    validate_execution(document, job, job_bytes, implementation, version)
    cases = {case["case_id"]: case for case in job["cases"]}
    if not cases or set(cases) != set(recorded):
        raise ValueError("Latest recorded case set differs from authored requests")
    mode = "batch" if implementation in {"dynamo_v1", "dynamo_v1_jail"} else "stream"
    counts = Counter()
    for key, case in cases.items():
        expected = recorded[key]
        actual = document["results"][case["id"]]
        validate_capture_input(case["request"])
        if not typed_equal(expected.get("capture_input"), case["request"]):
            raise ValueError(f"{case['id']}: recorded request differs")
        outcome = execution_outcome(actual, mode)
        if outcome[0] == "source_build_blocker":
            raise ValueError(f"{case['id']}: producer did not execute")
        if outcome[0] == "unsupported_capability" and outcome[1] not in {
            "unsupported_family", "unsupported_capture_mode", "unsupported_request_mode",
        }:
            raise ValueError(f"{case['id']}: unverified unsupported outcome")
        if outcome != execution_outcome(expected, mode):
            raise ValueError(f"{case['id']}: recorded support/error status differs")
        for field in ("assembled", "batch_final", "chunks", "partial_chunks", "error", "unavailable", "capture_observation"):
            before, after = expected.get(field), actual.get(field)
            if before is not None and after is not None:
                if field == "assembled":
                    before, after = canonical_events(before), canonical_events(after)
                elif field == "batch_final":
                    before, after = (
                        dict(final, calls=[dict(call, arguments=canonical_arguments(call["arguments"]))
                                           for call in final["calls"]])
                        for final in (before, after)
                    )
            if (field in actual) != (field in expected) or not typed_equal(before, after):
                raise ValueError(f"{case['id']}: recorded {field} differs")
        counts[outcome[0]] += 1
    return dict(counts)


def replay(root, work):
    store = root / capture_sources.STORE / "families"
    families = sorted(path.parent.name for path in store.glob("*/inputs_and_golden.yaml"))
    registered = set(yaml.safe_load(parser_families_path().read_bytes())["unified"])
    if set(families) != registered:
        raise ValueError("Latest authored families differ from the registered corpus")
    work.mkdir(parents=True, exist_ok=False)
    catalog = capture_sources.current(root)
    jobs, recorded, versions = [], {}, set()
    for family in families:
        job = build_job(store, family, mode="batch")
        path = work / "jobs" / f"{family}.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(job, ensure_ascii=False) + "\n")
        jobs.append(path)
        for implementation in ("dynamo_v1", "dynamo_v1_jail"):
            version, records = latest_records(root, family, implementation, catalog=catalog)
            versions.add(version)
            recorded[family, implementation] = records
    if len(versions) != 1:
        raise ValueError("Latest v1 modes must select one published release")
    version = versions.pop()
    inventory = yaml.safe_load((root / "conformance/dynamo-capture-inventory.yaml").read_bytes())
    target = [row for row in inventory["targets"]
              if row["implementation"] == "dynamo_v1" and row["version"] == version]
    if len(target) != 1 or target[0]["source_kind"] != "crate":
        raise ValueError("Latest v1 has no uniquely pinned published source")
    summary = {}
    for implementation in ("dynamo_v1", "dynamo_v1_jail"):
        jail = implementation.endswith("_jail")
        outputs = batch.capture_jobs(
            version, jobs, work / implementation, work / "producers" / implementation,
            work / "target", jail_buffer=jail,
            lockfile=work / "producers/dynamo_v1" / jobs[0].stem / "Cargo.lock" if jail else None,
        )
        for job_path, output in zip(jobs, outputs, strict=True):
            job_bytes = job_path.read_bytes()
            job = json.loads(job_bytes)
            document = json.loads(output.read_bytes())
            if document["provenance"]["crate_checksum"] != target[0]["checksum"]:
                raise ValueError("Latest v1 producer checksum differs from inventory")
            summary[f"{job['family']}/{implementation}"] = compare_results(
                job, document, recorded[job["family"], implementation], implementation, version, job_bytes,
            )
    (work / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(replay(args.root.resolve(), args.work.resolve()), sort_keys=True))


if __name__ == "__main__":
    main()
