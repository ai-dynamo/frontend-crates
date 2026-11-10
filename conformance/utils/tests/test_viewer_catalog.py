from __future__ import annotations

import json
import io
import tarfile
import threading
import hashlib
import subprocess
import sys
import yaml
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "conformance/utils/src"))
import viewer_catalog
import package_fixtures
import unified_history
import unified_archives

import pytest

from conformance.utils.tests.fixture_test_support import run_node_contract




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
                "runtime_version": "1.0.0", "fragment": "historical", "capture": {},
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
    assert metadata["families"] == {"gemma4": current_inputs, "hermes": historical_inputs["hermes"]}
    candidate = {"mode": "unified", "implementation": "dynamo_v1", "version": "1.0.0"}
    viewer_catalog._index_snapshot_families([candidate], metadata)
    assert "hermes" in candidate["capture_families"]
    assert candidate["has_captured_results"]


def test_history_dataset_indexes_complete_snapshot_fragments(tmp_path, monkeypatch):
    declaration = tmp_path / "conformance/history-datasets.json"
    declaration.parent.mkdir(parents=True)
    declaration.write_text(json.dumps([{"id": "retained", "root": "history"}]))
    cache = tmp_path / "history-cache"
    cache.mkdir()
    (tmp_path / "conformance/history-reader-cache.json").write_text(json.dumps({"directory": str(cache)}))
    store = tmp_path / "history/conformance/fixtures-unified-v2"
    family = store / "families/hermes"
    family.mkdir(parents=True)
    snapshot_path = family / "dynamo_v1-1.0.0.yaml"
    capture = {
        "schema_version": 2,
        "provenance": {"status": "captured"},
        "document": {"mode": "unified"},
        "records": {"tool_only": {"member": "conformance/fixtures-unified-v2/families/hermes/dynamo_v1-1.0.0.yaml",
                                    "value": {"case_key": "UNIFIED.1", "document": {}, "stimulus": {"ref": "current"},
                                              "observation": {"value": {"assembled": []}}}}},
    }
    snapshot_path.write_text(yaml.safe_dump(capture, sort_keys=False))
    snapshot = {"version": unified_history._capture_release_sort_key("1.0.0"),
                "runtime_version": "1.0.0", "record_count": 1, "active_cases_complete": True,
                "code_values": [None], "has_results": True, "native": False, "route_recorded": True}
    monkeypatch.setattr(viewer_catalog, "_native_catalog_document_index",
                        lambda *args, **kwargs: ([], {}, {"snapshots": {("hermes", "dynamo_v1"): [snapshot]},
                                                          "families": {"hermes": {"active_cases": ["tool_only"]}}}))
    monkeypatch.setattr(viewer_catalog, "_unified_records", lambda *args, **kwargs: [])
    datasets = viewer_catalog._history_dataset_sources(tmp_path, [], {"snapshots": {}, "families": {}}, {})
    checkpoint = datasets[0]["checkpoints"][0]
    fragment_path = cache / f"{checkpoint['fragment']}.json"
    fragment = json.loads(fragment_path.read_bytes())
    assert checkpoint["capture"] == viewer_catalog.encode_typed({"provenance": capture["provenance"],
                                                                   "document": capture["document"]})
    assert fragment["identity"] == checkpoint["fragment"]
    encoded_payload = json.dumps(fragment["payload"], ensure_ascii=False, sort_keys=True,
                                 separators=(",", ":")).encode() + b"\n"
    assert fragment["sha256"] == hashlib.sha256(encoded_payload).hexdigest()
