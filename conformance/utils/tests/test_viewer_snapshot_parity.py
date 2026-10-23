# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "conformance/utils/src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
import ci_coverage
import unified_history
import migrate_unified_snapshots
import viewer_catalog
from native_publication import native_snapshots, proof_catalog
from verify_publication_transport import read_index
from conformance.utils.tests.test_execution_manifests import compact_store


@pytest.mark.parametrize("fault", [None, "missing-execution", "missing-receipt", "corrupt-execution", "corrupt-receipt", "corrupt-history", "corrupt-snapshot", "omitted-version", "omitted-family", "extra-version", "duplicate-selection", "missing-input-family", "missing-all-inputs", "extra-input-family"])
def test_compact_snapshot_parity_uses_validated_capture_reader(tmp_path, fault):
    _snapshot_parity_case(tmp_path, fault)


def test_complete_snapshot_parity_reads_source_records(tmp_path):
    _snapshot_parity_case(tmp_path, None, complete=True)


def _snapshot_parity_case(tmp_path, fault, *, complete=False):
    membership_faults = {"omitted-version", "omitted-family", "extra-version", "duplicate-selection"}
    input_faults = {"missing-input-family", "missing-all-inputs", "extra-input-family"}
    original, _first, _second, _before = compact_store(tmp_path / "seed")
    root = tmp_path / "root"
    store = root / "conformance/fixtures-unified-v2"
    shutil.copytree(original, store)
    if fault is None or fault in membership_faults or fault in input_faults:
        loaded = unified_history.load_store(store)
        for path, document in unified_history._store_documents(loaded, compact_executions=False).items():
            target = store / "families/qwen3" / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if path.name == unified_history.FAMILY_DOCUMENT_NAME:
                document["input_document"]["family"] = "qwen3"
                document["golden_document"]["family"] = "qwen3"
            target.write_text(unified_history.dump_yaml(document))
        unified_history.compact_execution_store(store)
    sparse_resources = viewer_catalog._unified_records(root)
    sparse = tmp_path / "sparse"
    store.rename(sparse)
    migrate_unified_snapshots.convert(sparse, store, tmp_path)
    site = tmp_path / "site"
    catalog = {
        "revision": "a" * 64,
        "metadata": {"unified_capture_policy": {"dynamo_v2": {"mode": "stream"}}},
        "sources": {
            "unified": viewer_catalog._unified_records(root),
            "candidates": [{"mode": "unified", "implementation": "dynamo_v2", "version": version}
                           for version in ("0.1.0", "0.2.0")],
        },
    }
    original_candidates = [dict(item) for item in catalog["sources"]["candidates"]]
    if fault == "omitted-family":
        catalog["sources"]["candidates"][0]["capture_families"] = ["gemma4"]
    elif fault == "extra-version":
        catalog["sources"]["candidates"].append({"mode": "unified", "implementation": "dynamo_v2", "version": "0.3.0"})
    published = native_snapshots(root, site, catalog)
    if not complete:
        # Sparse decoding belongs to this independent historical oracle, never
        # the production publisher. Compare both readers with the same packs.
        store.rename(tmp_path / "complete-source")
        sparse.rename(store)
        published["proof_original_unified"] = sparse_resources
    published["sources"]["candidates"] = original_candidates
    if fault == "omitted-version":
        published["sources"]["unified"] = [item for item in published["sources"]["unified"]
                                           if item.get("version") != "0.2.0"]
    elif fault == "duplicate-selection":
        published["sources"]["unified"].append(next(item for item in published["sources"]["unified"] if item["kind"] == "resolved-capture"))
    elif fault == "missing-input-family":
        published["sources"]["unified"] = [item for item in published["sources"]["unified"]
                                           if item["kind"] != "inputs-and-golden" or "qwen3" not in item.get("families", [item.get("family")])]
    elif fault == "missing-all-inputs":
        published["sources"]["unified"] = [item for item in published["sources"]["unified"] if item["kind"] != "inputs-and-golden"]
    elif fault == "extra-input-family":
        resource = next(item for item in published["sources"]["unified"] if item["kind"] == "inputs-and-golden" and item.get("family") == "gemma4")
        published["sources"]["unified"].append({**resource, "family": "unexpected_family"})
    elif fault and fault not in membership_faults:
        action, kind = fault.split("-")
        resources = published["sources"]["unified"] if kind == "snapshot" else published["proof_original_unified"]
        resource = next(item for item in resources if item["kind"] == ("resolved-capture" if kind == "snapshot" else kind))
        if action == "missing":
            resources.remove(resource)
        else:
            if kind == "snapshot":
                recipe = read_index(site, resource["recipe_index"])[resource["path"]]
                path = site / recipe["chunks"][0]["pack"]["path"]
            else:
                path = root / resource["path"]
            data = path.read_bytes()
            path.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    catalog_path = tmp_path / "published-catalog.json"
    catalog_path.write_text(json.dumps(published))
    result = subprocess.run(
        ["node", "conformance/viewer/snapshot_parity.js", str(root), str(site), str(catalog_path)],
        cwd=ROOT, text=True, capture_output=True, check=False, timeout=30,
    )
    if fault:
        assert result.returncode != 0
        expected = "published authored family membership" if fault in input_faults else "published capture selection" if fault in membership_faults else "not indexed" if fault.startswith("missing-") else "SHA-256 mismatch"
        assert expected in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        assert "2 authored family comparisons, 4 resolved captures, 4 complete records" in result.stdout
        assert "4 checkpoint visits, 2 capture bundles decoded, 1 retained capture bundle" in result.stdout


def test_full_corpus_json_snapshots_match_python_and_javascript_resolution(tmp_path, published_viewer):
    site = published_viewer
    published = proof_catalog(ROOT, site)
    catalog_path = tmp_path / "published-catalog.json"
    catalog_path.write_text(json.dumps(published))
    selection = tmp_path / "selection.json"
    resources = [row for row in published["sources"]["unified"] if row["kind"] == "resolved-capture"]
    selected = ci_coverage.select_resources(resources)
    selection.write_text(json.dumps([row["path"] for row in selected]))
    result = subprocess.run(
        ["node", "conformance/viewer/snapshot_parity.js", str(ROOT), str(site), str(catalog_path), str(selection)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=60 * 60,
    )
    assert "full corpus parity passed:" in result.stdout
