# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import tarfile
import zlib

import pytest

import unified_archives
import unified_history
import capture_sources
import migrate_v1_retention
import import_peer_unified
import v1_capture_retention as retention
from test_import_peer_unified import _receipted_jobs
from test_unified_history import _change, _write_capture, _write_family


def test_reconciliation_selects_only_newest_stable_v1_but_all_v2():
    targets = [{"implementation": implementation, "version": version, "source_kind": "crate", **extra}
               for implementation, version, extra in (
                   ("dynamo_v1", "10.0.0", {}), ("dynamo_v1", "11.0.9", {}),
                   ("dynamo_v1", "11.0.13", {}), ("dynamo_v1", "11.0.14", {"yanked": True}),
                   ("dynamo_v1", "12.0.0-rc.1", {}),
                   ("dynamo_v2", "0.8.0", {}), ("dynamo_v2", "0.9.0", {}))]
    assert [(row["implementation"], row["version"]) for row in retention.selected_targets(targets)] == [
        ("dynamo_v1", "11.0.13"), ("dynamo_v2", "0.8.0"), ("dynamo_v2", "0.9.0")]
    assert retention.selected_targets(targets, ["11.0.13"]) == [targets[2]]
    with pytest.raises(ValueError, match="superseded"):
        retention.selected_targets(targets, ["11.0.9"])


def test_explicit_v2_selection_preserves_git_and_yanked_targets():
    targets = [{"implementation": "dynamo_v2", "version": "0.8.0", "source_kind": "git"},
               {"implementation": "dynamo_v2", "version": "0.9.0", "source_kind": "crate", "yanked": True}]
    assert retention.selected_targets(targets) == []
    assert retention.selected_targets(targets, ["0.8.0", "0.9.0"]) == targets


def test_retention_keeps_the_latest_modern_source_for_current_capture(tmp_path):
    root = tmp_path / "store"
    _write_family(root)
    _write_capture(root, "3.0.0", {"text_only": _change("historical")}, implementation="dynamo_v1")
    for version in ("11.0.13", "11.0.14"):
        _write_capture(root, version, {"text_only": _change(version)}, implementation="dynamo_v1")
    targets = [{"implementation": "dynamo_v1", "version": version, "source_kind": "crate", "yanked": yanked}
               for version, yanked in (("11.0.13", False), ("11.0.14", True))]
    latest = retention.selected_targets(targets)[0]["version"]
    assert retention.retain(root, latest=latest)["latest"] == latest
    assert set(unified_history.load_store(root).histories["gemma4", "dynamo_v1"].captures) == {
        "dynamo_v1-3.0.0", "dynamo_v1-11.0.13"}


def test_corrupt_archive_cannot_publish_a_partial_retention_candidate(tmp_path, monkeypatch):
    root = tmp_path / "store"
    _write_family(root)
    _write_capture(root, "3.0.0", {"text_only": _change("old")}, implementation="dynamo_v1")
    before = unified_history.store_inventory(root)
    monkeypatch.setattr(unified_archives, "capture_archive_bytes", lambda documents: b"corrupt gzip")
    with pytest.raises(zlib.error):
        retention.retain(root)
    assert unified_history.store_inventory(root) == before


def test_migration_preserves_recovery_archive_and_current_files_when_publishing(tmp_path):
    checkout = tmp_path / "checkout"
    root = checkout / capture_sources.STORE
    _write_family(root)
    _write_capture(root, "3.0.0", {"text_only": _change("original")}, implementation="dynamo_v1")
    _write_capture(root, "11.0.0", {"text_only": _change("old modern")}, implementation="dynamo_v1")
    _write_capture(root, "11.0.13", {"text_only": _change("latest")}, implementation="dynamo_v1")
    current = root / "families/gemma4/dynamo_v1.yaml"
    current.write_bytes(b"independent current capture\n")
    original = {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*.yaml")}
    work = tmp_path / "migration"
    result = migrate_v1_retention.migrate(checkout, work, publish=True)
    assert result["latest"] == "11.0.13"
    assert current.read_bytes() == original["families/gemma4/dynamo_v1.yaml"]
    with tarfile.open(work / "original-store.tar.gz") as archive:
        for name, data in original.items():
            assert archive.extractfile(capture_sources.STORE + "/" + name).read() == data
    assert not (root / "families/gemma4/dynamo_v1-11.0.0.yaml").exists()
    assert (root / "families/gemma4/dynamo_v1-11.0.13.yaml").is_file()
    assert (work / "proof.json").is_file()


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("archive", [False, True])
def test_v1_mutation_preserves_unrelated_history_bytes_and_archive_layout(tmp_path, compact, archive):
    root = tmp_path / "store"
    _write_family(root)
    _write_capture(root, "11.0.13", {"text_only": _change("before")}, implementation="dynamo_v1")
    for version in ("0.5.0", "0.5.2", "0.5.3"):
        _write_capture(root, version, {"text_only": _change(version)})
    if compact:
        unified_history.compact_execution_store(root)
    if archive:
        unified_history.archive_versions(root)
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()
              and not path.name.startswith(("dynamo_v1-", "gemma4--dynamo_v1-"))}
    original = unified_history.load_store(root).histories["gemma4", "dynamo_v2"]
    expected = {identity: original.complete_snapshot(identity) for identity in original.captures}

    def mutate(store):
        store.histories["gemma4", "dynamo_v1"].captures["dynamo_v1-11.0.13"]["changes"] = {"text_only": _change("after")}
        return {("gemma4", "dynamo_v1")}

    unified_history._mutate_store(root, mutate, compact_executions=True, preserve_untouched_histories=True)
    for name, data in before.items():
        assert (root / name).read_bytes() == data
    actual = unified_history.load_store(root).histories["gemma4", "dynamo_v2"]
    assert set(actual.captures) == set(expected)
    for identity, snapshot in expected.items():
        assert actual.complete_snapshot(identity) == snapshot


@pytest.mark.parametrize("action", ["v1_append", "unrelated_v2"])
def test_real_capture_import_and_rerun_preserve_archived_sources_and_unreferenced_receipts(tmp_path, action):
    root = tmp_path / "store"
    _write_family(root)
    groups = {implementation: _receipted_jobs(root, tmp_path / implementation, implementation, "batch",
                                              versions=("3.0.0", "10.0.0", "11.0.13"))
              for implementation in ("dynamo_v1", "dynamo_v1_jail")}
    import_peer_unified.import_capture_groups(root, groups, compact_executions=True)
    old = root / "families/gemma4/dynamo_v1-3.0.0.yaml"
    old.write_bytes(b"# Retained original measurement source.\n" + old.read_bytes())
    orphan = root / "receipts/old-measurement.yaml"
    orphan.write_text(unified_history.dump_yaml(groups["dynamo_v1"][0][2]))
    retention.retain(root)
    archive = root / "versions/dynamo-3.0.0.tar.gz"
    members = dict(unified_archives.members(archive))
    assert "receipts/old-measurement.yaml" in members
    assert any(name.startswith("executions/") for name in members)
    assert len([name for name in members if name.startswith("receipts/")]) >= 3
    original = {path.name: path.read_bytes() for path in unified_archives.paths(root)}
    if action == "v1_append":
        groups = {implementation: _receipted_jobs(root, tmp_path / (implementation + "-new"), implementation,
                                                  "batch", versions=("11.0.14",))
                  for implementation in ("dynamo_v1", "dynamo_v1_jail")}
    else:
        groups = {"dynamo_v2": _receipted_jobs(root, tmp_path / "v2", "dynamo_v2", "stream", versions=("0.9.0",))}
    import_peer_unified.import_capture_groups(root, groups, compact_executions=True)
    if action == "v1_append":
        retention.retain(root, latest="11.0.14")
    import_peer_unified.import_capture_groups(root, groups, compact_executions=True)
    retention.retain(root)
    assert original == {path.name: path.read_bytes() for path in unified_archives.paths(root)}


@pytest.mark.parametrize("compact", [False, True])
def test_retention_archives_major_checkpoints_with_parent_sources_and_keeps_current(tmp_path, compact):
    root = tmp_path / "store"
    _write_family(root)
    for implementation in ("dynamo_v1", "dynamo_v1_jail"):
        previous = None
        for version in ("3.0.0", "5.0.1", "5.1.2", "5.1.3", "9.2.13", "10.0.7", "10.2.2", "11.0.0", "11.0.13"):
            path = _write_capture(root, version, {"text_only": _change(version)} if version != "11.0.13" else {},
                                  implementation=implementation)
            if previous is not None:
                document = unified_history.load_yaml(path)
                document["parent"] = f"{implementation}-{previous}"
                path.write_text(unified_history.dump_yaml(document))
            previous = version
    _write_capture(root, "0.8.0", {"text_only": _change("peer remains")})
    if compact:
        unified_history.compact_execution_store(root)
    before = unified_history.load_store(root)
    inventory = unified_archives.inventory(root)
    old = {row["path"]: data for row, data in unified_archives.read_sources(root, inventory)
           if Path(row["path"]).name in {"dynamo_v1-3.0.0.yaml", "dynamo_v1_jail-3.0.0.yaml"}}
    expected = {identity: before.histories["gemma4", "dynamo_v1"].complete_snapshot(identity,
                    source_member=before.histories["gemma4", "dynamo_v1"]._member(identity)
                    if identity.endswith("11.0.13") else None)
                for identity in ("dynamo_v1-3.0.0", "dynamo_v1-5.0.1", "dynamo_v1-5.1.2", "dynamo_v1-5.1.3",
                                 "dynamo_v1-9.2.13", "dynamo_v1-10.0.7", "dynamo_v1-10.2.2", "dynamo_v1-11.0.13")}
    peer = (root / "families/gemma4/dynamo_v2-0.8.0.yaml").read_bytes()
    result = retention.retain(root)
    assert result == {"archives": 4, "retained": 16, "retired": 2, "latest": "11.0.13"}
    assert dict(unified_archives.members(root / "versions/dynamo-3.0.0.tar.gz")).items() >= old.items()
    after = unified_history.load_store(root)
    history = after.histories["gemma4", "dynamo_v1"]
    assert set(history.captures) == set(expected)
    for identity, snapshot in expected.items():
        assert history.complete_snapshot(identity) == snapshot
    assert (root / "families/gemma4/dynamo_v2-0.8.0.yaml").read_bytes() == peer
    assert not (root / "families/gemma4/dynamo_v1-11.0.0.yaml").exists()
    assert unified_history.load_yaml(root / "families/gemma4/dynamo_v1-11.0.13.yaml")["schema_version"] == 2
    for version in ("3.0.0", "5.1.3", "9.2.13", "10.2.2"):
        assert (root / f"versions/dynamo-{version}.tar.gz").is_file()
    for version in ("5.0.1", "5.1.2", "5.1.3"):
        assert f"families/gemma4/dynamo_v1-{version}.yaml" in dict(
            unified_archives.members(root / "versions/dynamo-5.1.3.tar.gz"))
    physical = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    assert retention.retain(root)["retired"] == 0
    assert physical == {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
