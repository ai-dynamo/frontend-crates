# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json

import pytest
import unified_archives
import unified_history
from test_unified_history import _store, _write_capture, _change
from test_execution_manifests import compact_store
from test_unified_taxonomy_covers_corpus import retained_unified_snapshots


@pytest.mark.parametrize("archived", [False, True])
@pytest.mark.parametrize("complete", [False, True])
def test_selected_original_matches_full_store_and_parses_each_source_once(tmp_path, monkeypatch, archived, complete):
    root = _store(tmp_path / "store")
    _write_capture(root, "0.5.3", {"text_only": _change("unselected")})
    if complete:
        loaded = unified_history.load_store(root)
        for history in loaded.histories.values():
            for capture_id in history.captures:
                path = history.capture_paths[capture_id]
                path.write_text(unified_history.dump_yaml(history.complete_snapshot(capture_id)))
    if archived:
        unified_history.archive_versions(root)
    before = unified_archives.inventory(root)
    expected = unified_history._load_store_unlocked(root)
    reads = []
    parse = unified_history._parse_yaml

    def counted(path, text):
        reads.append(path)
        return parse(path, text)

    monkeypatch.setattr(unified_history, "_parse_yaml", counted)
    selected = unified_history._load_store_unlocked(root, {("gemma4", "dynamo_v2"): {"0.5.0", "0.5.2", "0.5.1"}})
    assert len(reads) == len(set(reads))
    assert not any(path.stem == "dynamo_v2-0.5.3" for path in reads)
    latest_only = unified_history._load_store_unlocked(root, {("gemma4", "dynamo_v2"): {"0.5.2"}})
    assert latest_only.histories["gemma4", "dynamo_v2"].resolve("dynamo_v2-0.5.2") == expected.histories["gemma4", "dynamo_v2"].resolve("dynamo_v2-0.5.2")
    for version in ("0.5.0", "0.5.1", "0.5.2"):
        wanted = unified_history.original_capture([(0, "", expected)], "gemma4", "dynamo_v2", version)
        actual = unified_history.original_capture([(0, "", selected)], "gemma4", "dynamo_v2", version)
        identity = f"dynamo_v2-{version}"
        assert actual[2].resolve(actual[3], view_capture_id=identity) == wanted[2].resolve(wanted[3], view_capture_id=identity)
        assert actual[2].complete_snapshot(actual[3]) == wanted[2].complete_snapshot(wanted[3])
    assert unified_archives.inventory(root) == before


@pytest.mark.parametrize("fault", ["missing-parent", "bad-provenance", "malformed", "corrupt-archive"])
def test_selected_original_keeps_admission_and_validation(tmp_path, fault):
    root = _store(tmp_path / "store")
    path = root / "families/gemma4/dynamo_v2-0.5.2.yaml"
    if fault == "missing-parent":
        document = unified_history.load_yaml(path)
        document["parent"] = "dynamo_v2-0.5.0"
        path.write_text(unified_history.dump_yaml(document))
        (path.parent / "dynamo_v2-0.5.0.yaml").unlink()
    elif fault == "malformed":
        path.write_bytes(b"[not a mapping]")
    elif fault == "bad-provenance":
        document = unified_history.load_yaml(path)
        document["provenance"]["status"] = "invalid"
        path.write_text(unified_history.dump_yaml(document))
    else:
        _write_capture(root, "0.5.3", {"text_only": _change()})
        unified_history.archive_versions(root)
        archive = next((root / "versions").iterdir())
        archive.write_bytes(archive.read_bytes()[:-8])
    with pytest.raises((ValueError, FileNotFoundError)):
        unified_history._load_store_unlocked(root, {("gemma4", "dynamo_v2"): {"0.5.2"}})


def test_selected_original_deduplicates_execution_and_receipt_reads(tmp_path, monkeypatch):
    root, *_ = compact_store(tmp_path)
    expected = unified_history._load_store_unlocked(root)
    reads = []
    parse = unified_history._parse_yaml

    def counted(path, text):
        reads.append(path)
        return parse(path, text)

    monkeypatch.setattr(unified_history, "_parse_yaml", counted)
    selected = unified_history._load_store_unlocked(root, {("gemma4", "dynamo_v2"): {"0.1.0", "0.2.0"}})
    unified_history._validate_store_parity(expected, selected)
    assert len(reads) == len(set(reads))
    assert any(path.parent.name == "receipts" for path in reads)


def test_original_stores_validate_all_retained_yaml(historical_unified_sources):
    root, _ = historical_unified_sources
    owners = unified_history.original_stores(root.parent.parent)
    assert owners
    assert sum(len(store.histories) for _, _, store in owners) > 0


def test_full_validation_rejects_unselected_malformed_yaml(tmp_path):
    root = _store(tmp_path / "store")
    path = root / "families/gemma4/dynamo_v2-0.9.0.yaml"
    path.write_bytes(b"[malformed unselected capture]")
    selected = unified_history._load_store_unlocked(root, {("gemma4", "dynamo_v2"): {"0.5.2"}})
    assert selected.histories
    with pytest.raises(ValueError):
        unified_history._load_store_unlocked(root)


def test_selected_original_preserves_dataset_precedence(tmp_path):
    checkout = tmp_path / "checkout"
    descriptors = []
    for prefix, text in [("first", "first"), ("second", "second"), ("", "current")]:
        root = _store(checkout / prefix / "conformance/fixtures-unified-v2")
        _write_capture(root, "0.5.0", {"text_only": _change(text)})
        if prefix:
            descriptors.append({"id": prefix, "root": prefix})
    (checkout / "conformance/history-datasets.json").write_text(json.dumps(descriptors))
    requested = {("gemma4", "dynamo_v2"): {"0.5.0", "0.5.1", "0.5.2"}}
    full = unified_history.original_stores(checkout)
    selected = unified_history.original_stores(checkout, requested)
    for version in requested["gemma4", "dynamo_v2"]:
        for owners in (selected, selected[:-1]):
            expected = unified_history.original_capture(full if len(owners) == 3 else full[:-1], "gemma4", "dynamo_v2", version)
            actual = unified_history.original_capture(owners, "gemma4", "dynamo_v2", version)
            assert actual[0] == expected[0]
            assert actual[2].resolve(actual[3]) == expected[2].resolve(expected[3])


def test_retained_snapshots_resolve_sparse_history_once(tmp_path, monkeypatch):
    root = _store(tmp_path / "conformance/fixtures-unified-v2")
    for patch in range(3, 10):
        _write_capture(root, f"0.5.{patch}", {})
    store = unified_history._load_store_unlocked(root)
    history = store.histories["gemma4", "dynamo_v2"]
    resources = [{"kind": "history", "family": "gemma4", "implementation": "dynamo_v2",
                  "version": capture["runtime_version"], "path": history.capture_paths[capture_id].as_posix()}
                 for capture_id, capture in history.captures.items()]
    monkeypatch.setattr(unified_history, "original_stores", lambda *_: [(0, "", store)])
    monkeypatch.delenv("CONFORMANCE_PARITY_SHARD", raising=False)
    monkeypatch.delenv("CONFORMANCE_PARITY_SHARDS", raising=False)
    calls = []
    apply = unified_history.History._apply_capture

    def counted(self, state, capture_id):
        calls.append(capture_id)
        return apply(self, state, capture_id)

    monkeypatch.setattr(unified_history.History, "_apply_capture", counted)
    snapshots = list(retained_unified_snapshots(store, resources=resources))
    assert len(snapshots) == len(resources)
    assert calls == history.ordered_capture_ids()
