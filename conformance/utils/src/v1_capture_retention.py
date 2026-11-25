# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Archive major v1 checkpoints with parent sources and retain the current v1 source."""

from collections import defaultdict
from pathlib import Path
import shutil
import tempfile

from packaging.version import Version

import capture_sources
import execution_manifests
import unified_archives
import unified_history
from viewer_codec import typed_equal


IMPLEMENTATIONS = {"dynamo_v1", "dynamo_v1_jail"}
POLICY = "archive the latest captured v1 release for each 3.x through 10.x major with parent sources; use only the independent versionless current capture for the latest v1 release"


def selected_targets(targets, versions=None):
    eligible = [target for target in targets if target["source_kind"] == "crate"
                and not target.get("yanked", False) and not Version(target["version"]).is_prerelease]
    modern = [target for target in eligible if target["implementation"] == "dynamo_v1"
              and Version(target["version"]) >= Version("11.0.0")]
    newest = max((Version(target["version"]) for target in modern), default=None)
    selected = [target for target in (targets if versions else eligible)
                if target["implementation"] != "dynamo_v1"
                or target in modern and Version(target["version"]) == newest]
    if versions:
        selected = [target for target in selected if target["version"] in versions]
        if set(versions) - {target["version"] for target in selected}:
            raise ValueError("requested release is absent or superseded by the latest-v1 retention policy")
    return selected


def retain(root: Path, *, latest: str | None = None) -> dict:
    root = root.resolve()
    with unified_history._store_mutation_lock(root):
        inventory = unified_archives.inventory(root)
        store = unified_history._load_store_unlocked(root)
        histories = [history for history in store.histories.values() if history.implementation in IMPLEMENTATIONS]
        if not histories:
            return {"archives": 0, "retained": 0, "retired": 0}
        modern = [Version(capture["runtime_version"]) for history in histories for capture in history.captures.values()
                  if Version(capture["runtime_version"]) >= Version("11.0.0")
                  and not Version(capture["runtime_version"]).is_prerelease]
        newest = Version(latest) if latest is not None else max(modern, default=None)
        if latest is not None and (newest < Version("11.0.0") or newest not in modern):
            raise ValueError("selected latest v1 release is absent from the captured store")
        historical = [Version(capture["runtime_version"]) for history in histories for capture in history.captures.values()
                      if Version("3.0.0") <= Version(capture["runtime_version"]) < Version("11.0.0")
                      and not Version(capture["runtime_version"]).is_prerelease]
        checkpoints = {major: max(version for version in historical if version.major == major)
                       for major in {version.major for version in historical}}
        sources = {row["path"]: data for row, data in unified_archives.read_sources(root, inventory)}
        documents = {name: unified_history._parse_yaml_raw(Path(name), data) for name, data in sources.items()
                     if name.startswith(("executions/", "receipts/"))}
        removed = {name for name, document in documents.items() if document.get("implementation") in IMPLEMENTATIONS}
        archives = defaultdict(dict)
        snapshots = {}
        retained = 0
        retired = 0
        for name, document in documents.items():
            if document.get("implementation") in IMPLEMENTATIONS:
                version = Version(document["version"])
                if Version("3.0.0") <= version < Version("11.0.0"):
                    checkpoint = checkpoints[version.major]
                    archives[f"dynamo-{checkpoint}.tar.gz"][name] = sources[name]
        for history in histories:
            for identity, records in history.iter_resolved_snapshots(include_members=True):
                path = history.capture_path(identity).relative_to(root).as_posix()
                removed.add(path)
                version = Version(history.captures[identity]["runtime_version"])
                if version >= Version("11.0.0") and version != newest:
                    retired += 1
                    continue
                if version < Version("3.0.0"):
                    raise ValueError(f"v1 retention has no archive policy for {version}")
                snapshot = history.complete_snapshot(identity, records=records,
                    source_member=history._member(identity) if version >= Version("11.0.0") else None)
                snapshots[path] = snapshot
                retained += 1
                if version >= Version("11.0.0"):
                    continue
                group = archives[f"dynamo-{checkpoints[version.major]}.tar.gz"]
                group[path] = sources[path]
                capture = unified_history._parse_yaml_raw(Path(path), sources[path])
                if "execution" in capture:
                    execution = execution_manifests.source_reference(capture["execution"], "executions")
                    group[execution] = sources[execution]
                    for reference in documents[execution].get("receipts", {}):
                        receipt = execution_manifests.source_reference(reference, "receipts")
                        group[receipt] = sources[receipt]
        # A shared receipt has one physical owner even when releases reference it.
        assigned = set()
        for group in archives.values():
            for name in list(group):
                if name in assigned:
                    del group[name]
                assigned.add(name)
        with tempfile.TemporaryDirectory(prefix="v1-retention-", dir=root.parent) as temporary:
            staged = Path(temporary) / root.name
            shutil.copytree(root, staged)
            affected = {row["archive"]["path"] for row in inventory if row["path"] in removed and "archive" in row}
            for archive in affected:
                remaining = {name: data for name, data in unified_archives.members(root / archive) if name not in removed}
                target = staged / archive
                if remaining:
                    target.write_bytes(unified_archives.capture_archive_bytes(remaining))
                else:
                    target.unlink()
            for name in removed:
                (staged / name).unlink(missing_ok=True)
            for filename, group in archives.items():
                target = staged / unified_archives.DIRECTORY / filename
                if target.exists():
                    existing = dict(unified_archives.members(target))
                    if set(existing) & set(group):
                        raise ValueError("duplicate archived v1 source")
                    group = {**existing, **group}
                target.parent.mkdir(exist_ok=True)
                target.write_bytes(unified_archives.capture_archive_bytes(group))
            for name, snapshot in snapshots.items():
                if Version(Path(name).stem.rsplit("-", 1)[1]) >= Version("11.0.0"):
                    (staged / name).write_text(unified_history.dump_yaml(snapshot))
            original = {row["path"]: (row["sha256"], row["size"]) for row in inventory}
            installed = {row["path"]: (row["sha256"], row["size"]) for row in unified_archives.inventory(staged)}
            if any(installed.get(name) != original[name] for name in assigned):
                raise ValueError("v1 archiving changed original logical source bytes")
            candidate = unified_history._load_store_unlocked(staged)
            for history in store.histories.values():
                expected_ids = {Path(name).stem for name in snapshots if Path(name).parent.name == history.family.name
                                and Path(name).stem.startswith(history.implementation + "-")}
                if history.implementation not in IMPLEMENTATIONS:
                    expected_ids = set(history.captures)
                actual = candidate.histories.get((history.family.name, history.implementation))
                if actual is None:
                    if expected_ids:
                        raise ValueError("v1 retention removed a nonempty capture history")
                    continue
                if set(actual.captures) != expected_ids:
                    raise ValueError("v1 retention changed the selected checkpoint membership")
                original_snapshots = history.iter_resolved_snapshots(expected_ids, include_members=True)
                actual_snapshots = actual.iter_resolved_snapshots(expected_ids, include_members=True)
                for (identity, records), (actual_identity, actual_records) in zip(original_snapshots, actual_snapshots, strict=True):
                    if identity != actual_identity:
                        raise ValueError("v1 retention changed checkpoint ordering")
                    modern_capture = history.implementation in IMPLEMENTATIONS and Version(
                        history.captures[identity]["runtime_version"]) >= Version("11.0.0")
                    member = history._member(identity) if modern_capture else None
                    if not typed_equal(history.complete_snapshot(identity, records=records, source_member=member),
                                       actual.complete_snapshot(identity, records=actual_records, source_member=member)):
                        raise ValueError(f"v1 retention changed resolved evidence: {history.family.name}/{identity}")
            unified_history.publish_paths_transactionally([(staged, root)], backup_parent=root.parent)
        return {"archives": len(archives), "retained": retained, "retired": retired,
                "latest": str(newest) if newest is not None else None}


def remove_current_duplicates(checkout: Path) -> list[Path]:
    """Remove backing copies only after their full evidence matches current YAML."""
    root = checkout / capture_sources.STORE
    with unified_history._store_mutation_lock(root):
        capture_sources.current(checkout)
        duplicates = []
        for current in root.glob("families/*/*.yaml"):
            if current.stem not in IMPLEMENTATIONS:
                continue
            data = current.read_bytes()
            snapshot = unified_history._parse_yaml(current, data)
            source = snapshot["capture_selection"]["source"]
            target = checkout / capture_sources.relative(source)
            expected = current.with_name(f"{current.stem}-{snapshot['capture_selection']['version']}.yaml")
            if target != expected:
                raise ValueError(f"current v1 source differs from selected release: {current}")
            if not target.exists():
                continue
            authored_path = current.with_name("inputs_and_golden.yaml")
            authored = unified_history._parse_yaml(authored_path, authored_path.read_bytes())
            original = target.read_bytes()
            complete = capture_sources.independent_snapshot(
                unified_history._parse_yaml(target, original), authored, source_member=source)
            if not typed_equal({key: value for key, value in snapshot.items() if key != "capture_selection"}, complete):
                raise ValueError(f"current v1 evidence differs from backing source: {current}")
            duplicates.append((current, data, target, original))
        for current, data, target, original in duplicates:
            if current.read_bytes() != data or target.read_bytes() != original:
                raise ValueError("v1 sources changed while removing duplicate captures")
        for _current, _data, target, _original in duplicates:
            target.unlink()
        return [target for _current, _data, target, _original in duplicates]
