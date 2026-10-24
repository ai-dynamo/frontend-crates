#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-time conversion of compact Unified history into complete v2 snapshots."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import execution_manifests
import unified_archives
import unified_history
import yaml


def _expanded_evidence(root: Path, path: Path) -> dict:
    manifest = unified_history.load_yaml(path)
    family = manifest["family"]
    implementation = manifest["implementation"]
    version = manifest["version"]
    receipt_documents = {
        reference: unified_history.load_yaml(
            root / execution_manifests.source_reference(reference, "receipts")
        )
        for reference in manifest.get("receipts", {})
    }
    expanded = execution_manifests.expand(
        manifest,
        family=family,
        implementation=implementation,
        version=version,
        receipt_documents=receipt_documents,
    )
    receipts = [
        {"members": members, "body": receipt_documents[reference]}
        for reference, members in sorted(manifest.get("receipts", {}).items())
    ]
    return {
        "kind": "execution_manifest",
        "schema_version": 1,
        "family": family,
        "implementation": implementation,
        "version": version,
        "checkpoint": manifest["checkpoint"],
        "cases": manifest["cases"],
        "defaults": manifest["defaults"],
        "exceptions": manifest["exceptions"],
        "metadata_changes": expanded["metadata_changes"],
        "receipt_membership": receipts,
        **({"execution_proof": expanded["execution_proof"]}
           if "execution_proof" in expanded else {}),
    }


def convert(source: Path, destination: Path, spool: Path) -> dict:
    source = source.resolve()
    destination = destination.resolve()
    spool = spool.resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("source and destination must be separate trees")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"destination must be empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="unified-snapshot-v2-", dir=spool))
    (staging / "families").mkdir()
    archive_references = set()
    captures = {}
    capture_count = 0
    record_count = 0

    try:
        with unified_archives.expanded(source, temporary_parent=spool) as expanded_root:
            family_root = expanded_root / "families"
            family_directories = sorted(path for path in family_root.iterdir() if path.is_dir())
            for family_directory in family_directories:
                family_path = family_directory / unified_history.FAMILY_DOCUMENT_NAME
                family = unified_history._validate_family(
                    family_path, unified_history.load_yaml(family_path)
                )
                family_output = staging / "families" / family.name
                family_output.mkdir()
                shutil.copyfile(family_path, family_output / "inputs_and_golden.yaml")

                def write_snapshot(history, capture_id, records):
                    nonlocal capture_count, record_count
                    capture = history.captures[capture_id]
                    capture_document = {
                        "schema_version": 2,
                        "provenance": capture["provenance"],
                        "document": capture["document"],
                        "records": {},
                    }
                    for case_id, value in records.items():
                        source_member = history.snapshot_members[capture_id][case_id]
                        capture_document["records"][case_id] = {"value": value, "member": source_member}
                        if case_id in history.snapshot_suppressed_inheritance.get(capture_id, set()):
                            capture_document["records"][case_id]["suppress_inherited_from"] = True
                    if "execution_proof" in capture:
                        capture_document["execution_proof"] = capture["execution_proof"]
                    member = f"families/{family.name}/{capture_id}.yaml"
                    identity = (family.name, history.implementation, capture["runtime_version"])
                    if identity in captures:
                        raise ValueError(f"duplicate version capture: {family.name}/{capture_id}")
                    member_path = staging / member
                    member_path.parent.mkdir(parents=True, exist_ok=True)
                    member_path.write_text(unified_history.dump_yaml(capture_document, shared_metadata=True), encoding="utf-8", newline="\n")
                    captures[identity] = member_path
                    capture_count += 1
                    record_count += len(records)
                    if capture_count % 250 == 0:
                        print(f"converted snapshots={capture_count} records={record_count}", flush=True)

                _, _, _, resources = unified_history._load_family(
                    (expanded_root, family), snapshot_callback=write_snapshot
                )
                archive_references.update(
                    value["execution"] for value in resources.values() if value.get("execution")
                )

            referenced_paths = {
                (expanded_root / execution_manifests.source_reference(reference, "executions")).resolve()
                for reference in archive_references
            }
            evidence_count = 0
            for path in sorted((expanded_root / "executions").glob("*.yaml")):
                if path.resolve() in referenced_paths:
                    continue
                evidence = _expanded_evidence(expanded_root, path)
                family = evidence["family"]
                implementation = evidence["implementation"]
                version = evidence["version"]
                target = captures.get((family, implementation, version))
                if target is None:
                    raise ValueError(f"unreferenced execution has no matching capture: {path.name}")
                snapshot_path = target
                snapshot = yaml.load(snapshot_path.read_bytes(), Loader=unified_history.StrictLoader)
                snapshot.setdefault("historical_evidence", []).append(evidence)
                snapshot_path.write_text(unified_history.dump_yaml(snapshot, shared_metadata=True), encoding="utf-8", newline="\n")
                evidence_count += 1

        unified_archives.pack(staging, unified_history.CAPTURE_DIRECTORY_RE,
                              unified_history._capture_release_sort_key)
        archive_count = len(unified_archives.paths(staging))
        result = {
            "captures": capture_count,
            "records": record_count,
            "historical_evidence": evidence_count,
            "archives": archive_count,
            "authored_documents": sum(1 for _ in (staging / "families").glob("*/inputs_and_golden.yaml")),
            "compressed_bytes": sum(path.stat().st_size for path in unified_archives.paths(staging)),
        }
        if destination.exists():
            destination.rmdir()
        staging.rename(destination)
        return result
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--spool", type=Path, required=True)
    args = parser.parse_args()
    print(convert(args.source, args.destination, args.spool))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
