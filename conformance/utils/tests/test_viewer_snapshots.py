# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "conformance/utils/src"))
import unified_history
import viewer_snapshots


def test_snapshot_generation_resolves_history_without_repeating_source_bytes(tmp_path, monkeypatch):
    source = tmp_path / "source"
    directory = source / "conformance/fixtures-unified-v2/families/sample"
    family = unified_history.Family("sample", {"input_document": {}, "golden_document": {}, "cases": {}}, directory / "inputs_and_golden.yaml")
    capture = {"runtime_version": "1.0.0", "provenance": {"status": "legacy", "captured_with": {"sample": "1.0.0"}}, "document": {},
               "changes": {"one": {"case_key": "UNIFIED.1", "stimulus": {"ref": "current"}, "observation": {"value": {"assembled": []}}}},
               "metadata_changes": {"one": {"verified_recapture": {"receipt": "first-checkpoint"}}}, "document_overrides": {}}
    checkpoint = "sample-1.0.0"
    history = unified_history.History(family, "sample", {checkpoint: capture}, directory, {checkpoint: directory / (checkpoint + ".yaml")})
    store = unified_history.Store(source / "conformance/fixtures-unified-v2", {"sample": family}, {("sample", "sample"): history})
    monkeypatch.setattr(unified_history, "load_store", lambda root: store)
    catalog = {"schema": 1, "revision": "a" * 64, "sources": {"unified": [{"path": "original.yaml", "sha256": "b" * 64, "size": 5000}],
               "candidates": [{"implementation": "sample", "version": version, "mode": "unified"} for version in ("1.0.0", "1.1.0")], "archives": []},
               "metadata": {"unified_capture_policy": {"sample": {"mode": "stream"}}}}
    site = tmp_path / "site"
    published = viewer_snapshots.publish_snapshots(source, site, catalog)
    assert published["revision"] != catalog["revision"]
    assert published["source_revision"] == catalog["revision"]
    assert published["sources"]["original_unified"] == catalog["sources"]["unified"]
    assert len(published["sources"]["unified"]) == 3
    for resource in published["sources"]["unified"]:
        data = (site / resource["path"]).read_bytes()
        assert len(data) == resource["size"]
        assert hashlib.sha256(data).hexdigest() == resource["sha256"]
        assert b"sourceBytes" not in gzip.decompress(data)
    descriptors = site / "descriptors.json"
    descriptors.write_text(json.dumps(published["sources"]["unified"]))
    script = """
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { parseSnapshot } from './conformance/viewer/snapshots.js';
const [site, descriptors] = process.argv.slice(1);
const resources = JSON.parse(readFileSync(descriptors));
const resolved = await Promise.all(resources.filter(item => item.kind === 'resolved-capture').map(item => parseSnapshot(readFileSync(site + '/' + item.path), item)));
assert.equal(resolved[0].one.value.document.record_metadata.verified_recapture.receipt, 'first-checkpoint');
assert.equal(Object.hasOwn(resolved[1].one.value.document.record_metadata, 'verified_recapture'), false);
assert.equal(resolved[0].one.member, 'conformance/fixtures-unified-v2/families/sample/sample-1.0.0.yaml');
assert.equal(Object.hasOwn(resolved[0].one, 'sourceBytes'), false);
"""
    subprocess.run(["node", "--input-type=module", "-e", script, str(site), str(descriptors)], cwd=ROOT, capture_output=True, check=True)
    assert catalog["sources"]["unified"][0]["path"] == "original.yaml"


def test_snapshot_integrity_negative_controls():
    subprocess.run(["node", "conformance/viewer/snapshots.test.js"], cwd=ROOT, capture_output=True, check=True)
