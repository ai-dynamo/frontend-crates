#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preview or publish YAML capture migration, policy edits, and safe removal."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import capture_policy
import fixture_disposition
import legacy_history
import package_fixtures
import unified_history


STORE_NAMES = ("fixtures-unified-v2", "fixtures-v1")


def resolved_inventory(conformance: Path) -> dict:
    return {
        **{"unified/" + key: value for key, value in unified_history.resolved_inventory(
            conformance / STORE_NAMES[0]).items()},
        **legacy_history.resolved_inventory(conformance / STORE_NAMES[1]),
    }


def _file_inventory(root: Path) -> dict[str, str]:
    paths = [root / "fixtures-manifest.json", root / fixture_disposition.CAPTURE_POLICY_PATH]
    for name in STORE_NAMES:
        paths.extend((root / name).rglob("*.yaml"))
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths if path.is_file()}


def _materialize(conformance: Path, target: Path) -> None:
    legacy_history.materialize_store(conformance / STORE_NAMES[1], target)
    unified_history.materialize_store(conformance / STORE_NAMES[0], target / "unified")


def _certificates(policy: dict) -> dict:
    return {(corpus, identity): rule["equivalence_sha256"]
            for corpus, rules in policy["corpora"].items()
            for identity, rule in rules.get("selectors", {}).items()
            if "equivalence_sha256" in rule}


def _retained_evidence(value: dict, identity: str, *, selection_changes: bool) -> dict:
    if not (selection_changes and identity.startswith("batch_on_stream/") and identity.endswith("/inputs")):
        return value
    # Selecting another snapshot changes these headers, never the saved requests.
    record = value["record"]
    documents = {name: {key: item for key, item in document.items()
                        if key not in {"capture_selection", "capture_order", "uncaptured"}}
                 for name, document in record["documents"].items()}
    return {**value, "record": {**record, "documents": documents}}


def _pin_manifest(conformance: Path) -> None:
    path = conformance / "fixtures-manifest.json"
    manifest = json.loads(path.read_text())
    replacements = {}
    for store, module, format_name in (
        (STORE_NAMES[0], unified_history, "unified-history"),
        (STORE_NAMES[1], legacy_history, legacy_history.HISTORY_FORMAT),
    ):
        digest, size = module.store_digest(conformance / store)
        name = (fixture_disposition.UNIFIED_HISTORY_PATH if format_name == "unified-history"
                else legacy_history.HISTORY_PATH)
        replacements[name] = {"path": name, "format": format_name, "sha256": digest, "size": size}
    policy_pin = fixture_disposition.capture_policy_pin(conformance / fixture_disposition.CAPTURE_POLICY_PATH)
    replacements[policy_pin["path"]] = policy_pin
    prior = {entry["path"]: entry for entry in manifest["shards"]}
    manifest["shards"] = sorted((prior | replacements).values(), key=lambda item: item["path"])
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def maintain(
    repo_root: Path, operation: str, *, apply: bool = False, corpus: str | None = None,
    capture: str | None = None, family: str | None = None, policy_path: Path | None = None,
    replacement: str | None = None, unavailable: str | None = None,
) -> dict:
    conformance = repo_root.resolve() / "conformance"
    if operation not in {"migrate", "remove", "policy"}:
        raise ValueError(f"unknown maintenance operation: {operation}")
    if operation == "remove" and (corpus is None or capture is None):
        raise ValueError("removal requires --corpus and --capture")
    if operation == "policy" and policy_path is None:
        raise ValueError("policy changes require --policy")
    if replacement is not None and unavailable is not None:
        raise ValueError("choose a replacement or an unavailable reason")
    with unified_history._store_mutation_lock(conformance / STORE_NAMES[0]):
        before_files = _file_inventory(conformance)
        with tempfile.TemporaryDirectory(prefix=".capture-maintenance-", dir=conformance) as temporary:
            candidate = Path(temporary)
            for name in STORE_NAMES:
                shutil.copytree(conformance / name, candidate / name)
            # Public readers acquire their own locks. Read the private copy while
            # the live parent lock prevents a concurrent publisher changing it.
            before = resolved_inventory(candidate)
            if operation == "remove":
                for name in STORE_NAMES:
                    shutil.copytree(candidate / name, candidate / "original" / name)
                unified_history.migrate_store(candidate / "original" / STORE_NAMES[0])
                legacy_history.migrate_store(candidate / "original" / STORE_NAMES[1])
            shutil.copyfile(conformance / "fixtures-manifest.json", candidate / "fixtures-manifest.json")
            shutil.copyfile(policy_path or conformance / fixture_disposition.CAPTURE_POLICY_PATH,
                            candidate / fixture_disposition.CAPTURE_POLICY_PATH)
            policy = capture_policy.load_policy(candidate / fixture_disposition.CAPTURE_POLICY_PATH)
            if policy_path is not None:
                original_policy = capture_policy.load_policy(conformance / fixture_disposition.CAPTURE_POLICY_PATH)
                if _certificates(policy) != _certificates(original_policy):
                    raise ValueError("policy imports cannot change equivalence certificates; use verified removal")
            unified_history.migrate_store(candidate / STORE_NAMES[0])
            legacy_history.migrate_store(candidate / STORE_NAMES[1])
            removed = set()
            if operation == "remove":
                removed = {key for key in before if key.split("/")[0] == corpus
                           and key.split("/")[-1] == capture
                           and (family is None or key.split("/")[1] == family)}
                if not removed:
                    raise ValueError(f"capture is not recorded: {corpus}/{family or '*'}/{capture}")
                if corpus == "unified":
                    unified_history.remove_capture(candidate / STORE_NAMES[0], capture, family=family)
                else:
                    legacy_history.remove_capture(candidate / STORE_NAMES[1], capture, corpus=corpus,
                                                  family=family, replacement=replacement,
                                                  unavailable=unavailable)
            after = resolved_inventory(candidate)
            expected = {key: value for key, value in before.items() if key not in removed}
            selection_changes = operation == "remove" and corpus == "batch_on_stream" and bool(replacement or unavailable)
            normalized_expected = {key: _retained_evidence(value, key, selection_changes=selection_changes)
                                   for key, value in expected.items()}
            normalized_after = {key: _retained_evidence(value, key, selection_changes=selection_changes)
                                for key, value in after.items()}
            if normalized_after != normalized_expected:
                changed = sorted(key for key in expected.keys() | after.keys()
                                 if normalized_expected.get(key) != normalized_after.get(key))
                raise ValueError(f"maintenance changed retained capture evidence: {changed[:10]}")
            with tempfile.TemporaryDirectory(prefix="capture-maintenance-readers-") as loose:
                loose_root = Path(loose)
                if operation == "remove":
                    _materialize(candidate / "original", loose_root / "before")
                    if any(key.startswith(f"{corpus}/") and key.endswith(f"/{capture}") for key in after):
                        capture_policy.validate_partial_removal_policy(
                            policy, corpus, capture, family, loose_root / "before")
                    else:
                        policy = capture_policy.retire_capture(
                            policy, corpus, capture, replacement=replacement,
                            unavailable=unavailable, before_root=loose_root / "before")
                        (candidate / fixture_disposition.CAPTURE_POLICY_PATH).write_text(unified_history.dump_yaml(policy))
                _materialize(candidate, loose_root / "after")
                capture_policy.validate_materialized_policy(policy, loose_root / "after")
            _pin_manifest(candidate)
            package_fixtures._validate_candidate_package(
                json.loads((candidate / "fixtures-manifest.json").read_text()),
                conformance / "fixtures", candidate / STORE_NAMES[0], candidate / STORE_NAMES[1])
            after_files = _file_inventory(candidate)
            changes = [{"path": key, "action": "remove" if key not in after_files else
                        "add" if key not in before_files else "update"}
                       for key in sorted(before_files.keys() | after_files.keys())
                       if before_files.get(key) != after_files.get(key)]
            result = {"operation": operation, "applied": apply, "removed": sorted(removed),
                      "retained_checkpoints": sum(not key.endswith("/inputs") for key in after), "changes": changes}
            if apply and changes:
                targets = [*STORE_NAMES, fixture_disposition.CAPTURE_POLICY_PATH, "fixtures-manifest.json"]
                unified_history.publish_paths_transactionally(
                    [(candidate / name, conformance / name) for name in targets], backup_parent=conformance)
            return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("migrate", "remove", "policy"))
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--corpus", choices=("unified", *legacy_history.CORPORA))
    parser.add_argument("--capture")
    parser.add_argument("--family")
    parser.add_argument("--policy", type=Path)
    disposition = parser.add_mutually_exclusive_group()
    disposition.add_argument("--replacement")
    disposition.add_argument("--unavailable")
    parser.add_argument("--apply", action="store_true", help="publish the validated candidate (default: preview)")
    args = parser.parse_args()
    print(json.dumps(maintain(args.repo_root, args.operation, apply=args.apply, corpus=args.corpus,
                             capture=args.capture, family=args.family, policy_path=args.policy,
                             replacement=args.replacement, unavailable=args.unavailable), indent=2))


if __name__ == "__main__":
    main()
