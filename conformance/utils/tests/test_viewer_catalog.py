from __future__ import annotations

import json
import io
import tarfile
import threading
import hashlib
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "conformance/utils/src"))
import viewer_catalog
import package_fixtures
import unified_history
import unified_archives
import capture_sources

import pytest

from conformance.utils.tests.fixture_test_support import run_node_contract, fixture_catalog
from test_capture_sources import native_publisher
from test_unified_history import _store, _change


def test_v1_family_major_sources_include_archive_parents_but_exclude_modern_history():
    anchor = {"implementation": "dynamo_v1", "version": "5.1.3",
              "archive": {"path": "versions/dynamo-5.1.3.tar.gz"}}
    parent = {"implementation": "dynamo_v1", "version": "5.1.2",
              "archive": {"path": "versions/dynamo-5.1.3.tar.gz"}}
    modern = {"implementation": "dynamo_v1_jail", "version": "11.0.13", "kind": "history"}
    current = {"implementation": "dynamo_v1_jail", "version": "11.0.13", "kind": "current"}
    assert viewer_catalog._selectable_v1_history(anchor)
    assert viewer_catalog._selectable_v1_history(parent)
    assert not viewer_catalog._selectable_v1_history(modern)
    assert viewer_catalog._selectable_v1_history(current)




def _package():
    package_fixtures.package_snapshot("new", "new America/Los_Angeles", {}, {}, dry_run=False, prune=False)


def test_history_keeps_inputs_for_families_absent_from_latest(tmp_path, monkeypatch):
    declaration = tmp_path / "conformance/history-datasets.json"
    declaration.parent.mkdir()
    declaration.write_text(json.dumps([{"id": "retained", "root": "history"}]))

    def authored(family, digest):
        return {"path": f"conformance/fixtures-unified-v2/families/{family}/inputs_and_golden.yaml",
                "kind": "inputs-and-golden", "family": family, "implementation": None, "sha256": digest}

    current = authored("gemma4", "current")
    retained = [authored("gemma4", "historical"), authored("hermes", "historical")]
    historical_inputs = {"gemma4": {"active_cases": ["old"]}, "hermes": {"active_cases": ["tool_only"]}}
    snapshot = {"version": unified_history._capture_release_sort_key("1.0.0"),
                "runtime_version": "1.0.0", "fragment": "historical", "capture_fragment": "a" * 64,
                "capture": {"evidence": "large retained evidence" * 1000},
                "record_count": 1, "active_cases_complete": True, "code_values": [None],
                "has_results": True, "native": False, "route_recorded": True}
    monkeypatch.setattr(viewer_catalog, "_native_catalog_document_index",
                        lambda *args, **kwargs: ([], {}, {"snapshots": {("hermes", "dynamo_v1"): [snapshot]},
                                                          "families": historical_inputs}))
    monkeypatch.setattr(viewer_catalog, "_unified_records", lambda *args, **kwargs: deepcopy(retained))
    unified = [current]
    current_inputs = {"active_cases": ["current"]}
    metadata = {"snapshots": {}, "families": {"gemma4": current_inputs}}
    datasets = viewer_catalog._history_dataset_sources(tmp_path, unified, metadata, {})
    assert {resource["family"] for resource in unified} == {"gemma4", "hermes"}
    assert next(resource for resource in unified if resource["family"] == "gemma4") == current
    historical = next(resource for resource in unified if resource["family"] == "hermes")
    assert historical["dataset_root"] == "history"
    assert historical["dataset"] == "retained"
    assert datasets[0]["resources"][0]["sha256"] == "historical"
    checkpoint = datasets[0]["checkpoints"][0]
    assert checkpoint["capture_fragment"] == "a" * 64
    assert "capture" not in checkpoint
    assert metadata["families"] == {"gemma4": current_inputs, "hermes": historical_inputs["hermes"]}
    candidate = {"mode": "unified", "implementation": "dynamo_v1", "version": "1.0.0"}
    viewer_catalog._index_snapshot_families([candidate], metadata)
    assert "hermes" in candidate["capture_families"]
    assert candidate["has_captured_results"]


def test_native_history_catalog_keeps_large_checkpoint_evidence_in_authenticated_fragments(tmp_path, monkeypatch, native_publisher):
    root = tmp_path / "source"
    store_root = _store(root / capture_sources.STORE)
    store = unified_history.load_store(store_root)
    history = store.histories["gemma4", "dynamo_v2"]
    snapshot = history.complete_snapshot("dynamo_v2-0.5.0")
    snapshot["records"]["text_only"]["value"]["observation"] = _change("retained text " * 20000)["observation"]
    selected = {"surface": "unified", "mode": "unified", "family": "gemma4", "implementation": "dynamo_v2",
                "version": "0.5.0", "source": f"{capture_sources.STORE}/families/gemma4/dynamo_v2-0.5.0.yaml"}
    current = store_root / "families/gemma4/dynamo_v2.yaml"
    current.write_bytes(capture_sources.selected_bytes(unified_history.dump_yaml(snapshot).encode(), selected, [],
        authored=store.families["gemma4"].document))
    snapshot = unified_history.load_yaml(current)
    unified_history.preserve_current_snapshots(root, store, implementations=("dynamo_v2",))
    unified_history._commit_store_documents(store_root, unified_history._store_documents(store), expected_store=store)
    catalog = fixture_catalog(root, {"metadata": {"unified_capture_policy": {"dynamo_v2": {"mode": "stream"}}}, "sources": {}})
    cache = tmp_path / "cache"
    monkeypatch.setenv("CONFORMANCE_PUBLISHER", str(native_publisher))
    monkeypatch.setenv("CONFORMANCE_CONVERSION_CACHE", str(cache))
    _inventory, _documents, indexed = viewer_catalog._native_catalog_document_index(store_root,
        conversion_context={"dataset": "retained", "metadata": catalog["metadata"]})
    rows = indexed["snapshots"]["gemma4", "dynamo_v2"]
    assert len(capture_sources.encoded(rows)) < 4096
    assert all("capture" not in row for row in rows)
    first = next(row for row in rows if row["runtime_version"] == "0.5.0")
    key = first["capture_fragment"]
    entry = json.loads((cache / f"{key}.json").read_bytes())
    encoded = (json.dumps(entry["payload"], ensure_ascii=False, separators=(",", ":")) + "\n").encode()
    assert entry["identity"] == entry["sha256"] == key == hashlib.sha256(encoded).hexdigest()
    evidence = unified_history.decode_typed(entry["payload"]["historical_evidence"])
    assert evidence[0]["source_snapshot"]["semantic_sha256"] == unified_history._current_snapshot_digest(snapshot)
