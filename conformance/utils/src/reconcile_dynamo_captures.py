#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Discover, qualify, capture and transactionally import missing Dynamo releases."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import yaml

import capture_coverage
import capture_dynamo_versions
import capture_sources
import import_peer_unified
import unified_archives
import unified_history
import v1_capture_retention
from viewer_codec import typed_equal

ROOT = Path(__file__).resolve().parents[3]
IMPLEMENTATIONS = {"dynamo_v1": ("dynamo_v1", "dynamo_v1_jail"), "dynamo_v2": ("dynamo_v2",)}


def qualified_capture(history, target, implementation, job_bytes, adapter):
    if history is None:
        return False
    identity = f"{implementation}-{target['version']}"
    if identity not in history.captures:
        return False
    checkpoint = history.captures[identity]
    records = history.resolve(identity)
    unified_history._validate_historical_evidence(checkpoint, history.family, implementation, target["version"],
                                                  str(history.capture_path(identity)), records=records)
    job = json.loads(job_bytes)
    expected = {"implementation": implementation, "version": target["version"],
                "job_sha256": hashlib.sha256(job_bytes).hexdigest(), "adapter_sha256": adapter}
    if target["source_kind"] == "crate":
        expected["crate_checksum"] = target["checksum"]
    if implementation == "dynamo_v2":
        expected["git_commit"] = target["git_commit"]

    def matches(receipt):
        identity_matches = (isinstance(receipt, dict) and all(receipt.get(key) == value for key, value in expected.items())
                and all(isinstance(receipt.get(key), str) and len(receipt[key]) == 64
                        for key in ("source_sha256", "results_sha256", "cargo_lock_sha256"))
                and receipt.get("status") != "source_unavailable")
        if not identity_matches:
            return False
        if implementation == "dynamo_v2":
            try:
                capture_dynamo_versions.validate_source_identity(Path(), receipt, target, implementation, job["family"])
            except ValueError:
                return False
        return True

    proofs = [checkpoint.get("execution_proof", {}), *[evidence.get("execution_proof", {})
              for evidence in checkpoint.get("historical_evidence", [])]]
    for case in job["cases"]:
        key, request = case["case_id"], case["request"]
        if any(matches(proof.get("receipt")) and proof.get("requests", {}).get(key) == unified_history._request_digest(request)
               for proof in proofs):
            continue
        record = records.get(key)
        if record is None:
            return False
        metadata = record["document"].get("record_metadata", {})
        supplements = [metadata.get("verified_recapture"), *metadata.get("verified_recaptures", [])]
        if any(isinstance(value, dict) and matches(value["provenance"])
               and typed_equal(value["result"]["capture_input"], request) for value in supplements):
            continue
        stimulus = record["stimulus"]
        if stimulus.get("semantic_sha256") != unified_history._request_digest(request):
            return False
        if not any(matches(value) for name, value in metadata.items()
                   if isinstance(name, str) and (name == "capture_provenance" or name.endswith("_capture_provenance"))):
            return False
    return bool(job["cases"])


def pending_targets(store, targets, jobs, adapters):
    return [target for target in targets
            if not all(qualified_capture(store.histories.get((family, implementation)), target, implementation,
                                         data, adapters[implementation])
                       for implementation in IMPLEMENTATIONS[target["implementation"]]
                       for (mode, family), data in jobs.items()
                       if mode == ("stream" if implementation == "dynamo_v2" else "batch"))]


def _physical_sources(store):
    return sorted([*store.glob("families/*/*.yaml"), *store.glob("executions/*.yaml"),
                   *store.glob("receipts/*.yaml"), *unified_archives.paths(store)])


def publish_history_store(root, candidate_store, expected_inventory, *, inventory_update=None):
    """Publish history and current selections together while preserving unrelated files."""
    live = root / capture_sources.STORE
    parent = live.parent
    with unified_history._store_mutation_lock(live):
        if not typed_equal(unified_history._store_inventory_unlocked(live), expected_inventory):
            raise ValueError("capture store changed during reconciliation")
        with tempfile.TemporaryDirectory(prefix=".capture-reconcile-", dir=parent) as temporary:
            staged = Path(temporary) / live.name
            shutil.copytree(live, staged)
            for path in _physical_sources(live):
                relative = path.relative_to(live)
                if (relative.parts[0] == "families" and path.name != unified_history.FAMILY_DOCUMENT_NAME
                        and "-" not in path.stem and not (candidate_store / relative).exists()):
                    continue
                (staged / relative).unlink()
            for path in _physical_sources(candidate_store):
                destination = staged / path.relative_to(candidate_store)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, destination)
            replacements = [(staged, live)]
            if inventory_update is not None:
                inventory_path, before, data = inventory_update
                if inventory_path.read_bytes() != before:
                    raise ValueError("release inventory changed during reconciliation")
                inventory_candidate = Path(temporary) / inventory_path.name
                inventory_candidate.write_bytes(data)
                replacements.append((inventory_candidate, inventory_path))
            unified_history.publish_paths_transactionally(replacements, backup_parent=parent)


def reconcile(root: Path, work: Path, *, workers=4, versions=None, discover=True, inventory=None):
    if workers < 1:
        raise ValueError("workers must be positive")
    inventory = inventory or root / "conformance/dynamo-capture-inventory.yaml"
    before = inventory.read_bytes()
    retained = yaml.safe_load(before)
    targets = [target for target in retained["targets"] if target["implementation"] in IMPLEMENTATIONS]
    work.mkdir(parents=True, exist_ok=False)
    if discover:
        targets = capture_coverage.discover_dynamo_releases(targets, work / "registry")
    selected = v1_capture_retention.selected_targets(targets, versions)
    inventory_document = {**retained, "policy": {"corpus": "UNIFIED",
        "v1": v1_capture_retention.POLICY,
        "v2": "retain every captured stable release and retained source labels; deduplicate unchanged observations"},
        "targets": targets}
    data = yaml.safe_dump(inventory_document, sort_keys=False, width=4096).encode()
    (work / "discovered-inventory.yaml").write_bytes(data)
    store_root = root / capture_sources.STORE
    expected_inventory = unified_history.store_inventory(store_root)
    jobs = capture_dynamo_versions.frozen_family_jobs(store_root / "families")
    if not jobs:
        raise ValueError("empty registered family corpus")
    capture_dynamo_versions.freeze_adapters(work)
    adapters = json.loads((work / "adapter-manifest.json").read_text())["implementations"]
    with capture_sources.history_view(root, root) as staged_root:
        staged_store = staged_root / capture_sources.STORE
        store = unified_history._load_store_unlocked(staged_store)
        if unified_history.preserve_current_snapshots(root, store):
            unified_history._commit_store_documents(staged_store,
                unified_history._store_documents(store, compact_executions=True), expected_store=store)
        pending = pending_targets(store, selected, jobs, adapters)
        summary = {"targets": len(selected), "pending": len(pending), "qualified": len(selected) - len(pending)}
        (work / "reconciliation.json").write_text(json.dumps(summary, indent=2) + "\n")
        if pending:
            run_inventory = work / "pending-inventory.yaml"
            run_inventory.write_text(yaml.safe_dump({**inventory_document, "targets": pending}, sort_keys=False))
            run = work / "run"
            result = capture_dynamo_versions.run_inventory(run_inventory, run, workers, store_root=store_root / "families")
            (run / "run.json").write_text(json.dumps(result, indent=2) + "\n")
            coverage = capture_dynamo_versions.coverage_inventory(run, result)
            (run / "coverage.json").write_text(json.dumps(coverage, ensure_ascii=False) + "\n")
            blockers = [row for row in coverage["combinations"] if row["status"] in {"source_build_blocker", "missing_evidence"}]
            if blockers:
                failures = sorted({(row["implementation"], row["version"], row["family"], row["status"]) for row in blockers})
                raise RuntimeError(f"capture reconciliation blocked; no evidence published: {failures}")
            if jobs != capture_dynamo_versions.frozen_family_jobs(store_root / "families"):
                raise ValueError("authored requests changed during reconciliation")
            groups = {}
            for target in pending:
                for implementation in IMPLEMENTATIONS[target["implementation"]]:
                    mode = "stream" if implementation == "dynamo_v2" else "batch"
                    for (job_mode, family), expected in jobs.items():
                        if job_mode != mode:
                            continue
                        job = run / "jobs" / mode / (family + ".json")
                        if job.read_bytes() != expected:
                            raise ValueError("captured job differs from reconciliation request")
                        capture = run / "captures" / f"{implementation}-{target['version']}" / (family + ".json")
                        provenance = json.loads(capture.read_text())["provenance"]
                        groups.setdefault(implementation, []).append((job, capture, provenance))
            import_peer_unified.import_capture_groups(staged_store, groups, compact_executions=True)
            store = unified_history._load_store_unlocked(staged_store)
            if pending_targets(store, pending, jobs, adapters):
                raise ValueError("imported release lacks qualified execution evidence")
        completed_v1 = {target["version"] for target in selected if target["implementation"] == "dynamo_v1"}
        if completed_v1:
            v1_capture_retention.retain(staged_store, latest=next(iter(completed_v1)))
            store = unified_history._load_store_unlocked(staged_store)
            unified_history.refresh_current_snapshots(staged_root, store, versions=completed_v1)
        publish_history_store(root, staged_store, expected_inventory, inventory_update=(inventory, before, data))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--versions", nargs="+")
    parser.add_argument("--no-discover", action="store_true", help="execute only the already frozen inventory")
    args = parser.parse_args()
    print(json.dumps(reconcile(args.root.resolve(), args.work.resolve(), workers=args.workers, versions=args.versions,
                               discover=not args.no_discover, inventory=args.inventory), sort_keys=True))


if __name__ == "__main__":
    main()
