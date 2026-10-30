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
    monkeypatch.setattr(viewer_catalog, "_native_catalog_document_index",
                        lambda *args, **kwargs: ([], {}, {"snapshots": {}}))
    monkeypatch.setattr(viewer_catalog, "_unified_records", lambda *args, **kwargs: deepcopy(retained))
    unified = [current]
    datasets = viewer_catalog._history_dataset_sources(tmp_path, unified, {"snapshots": {}}, {})
    assert {resource["family"] for resource in unified} == {"gemma4", "hermes"}
    assert next(resource for resource in unified if resource["family"] == "gemma4") == current
    historical = next(resource for resource in unified if resource["family"] == "hermes")
    assert historical["dataset_root"] == "history"
    assert historical["dataset"] == "retained"
    assert datasets[0]["resources"][0]["sha256"] == "historical"
