# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import extract_fixtures  # noqa: E402
import capture_stimulus  # noqa: E402
import fixture_disposition  # noqa: E402
import package_fixtures  # noqa: E402
import stream_capture_archive  # noqa: E402
import unified_history  # noqa: E402
import capture_sources
from fixture_test_support import configure_extractor_root

_TEST_STREAM_CAPTURE_VERSION = "99.0.0"
_TEST_STREAM_CAPTURE_ROOT = f"dynamo_v2-{_TEST_STREAM_CAPTURE_VERSION}"


def test_canonicalize_unified_inputs_rejects_duplicate_scenario_owners():
    records = {
        ("gemma4", "UNIFIED.3-1"): {"scenario": "text_only"},
        ("gemma4", "UNIFIED.9-9"): {"scenario": "text_only"},
    }

    with pytest.raises(ValueError, match="duplicate Unified scenario ownership"):
        fixture_disposition.canonicalize_unified_inputs(records)


def test_canonicalize_unified_inputs_allows_one_historical_alias_rename():
    scenario = "gemma4_guided_json_visible_call_prose_before_reasoning"
    records = {
        ("gemma4", "UNIFIED.31-29"): {"scenario": scenario, "input": "same"},
        ("gemma4", "UNIFIED.g4-1"): {"scenario": scenario, "input": "same"},
    }

    canonical, aliases = fixture_disposition.canonicalize_unified_inputs(records)

    ident = (
        "gemma4",
        fixture_disposition.canonical_unified_case_key(
            "gemma4", "UNIFIED.g4-1", scenario
        ),
    )
    assert canonical == {ident: records[("gemma4", "UNIFIED.g4-1")]}
    assert aliases[("gemma4", "UNIFIED.31-29")] == ident
    assert aliases[("gemma4", "UNIFIED.g4-1")] == ident


def test_stream_input_corrections_compare_yaml_scalar_types():
    old_docs = {
        "glm47/TOOLCALLING.streamv1.7.yaml": {
            "family": "glm47",
            "mode": "streamv1",
            "cases": {"TOOLCALLING.streamv1.7.g": {"enum": [True]}},
        }
    }
    corrected_docs = {
        "glm47/TOOLCALLING.streamv1.7.yaml": {
            "family": "glm47",
            "mode": "streamv1",
            "cases": {"TOOLCALLING.streamv1.7.g": {"enum": [1]}},
        }
    }

    assert stream_capture_archive._approved_case_additions(old_docs, corrected_docs, {}, {}) is None
    assert stream_capture_archive._approved_case_additions(
        old_docs,
        corrected_docs,
        {},
        {},
        {("glm47", "TOOLCALLING.streamv1.7.g")},
    ) == (set(), {("glm47", "TOOLCALLING.streamv1.7.g")})


@pytest.mark.parametrize("records", [["missing.yaml"], [], ["a.yaml", "a.yaml"], [1]])
def test_complete_snapshot_rejects_corrupt_member_index(records):
    with pytest.raises(ValueError, match="capture snapshot"):
        fixture_disposition.capture_snapshot_members(
            json.dumps({"schema_version": 1, "records": records}).encode(), ["a.yaml"],
        )


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    conf = tmp_path / "conformance"
    store = conf / "fixtures"
    false_path = "unified/dynamo_v2-0.6.0.tar.gz"
    path = store / false_path
    path.parent.mkdir(parents=True)
    path.write_bytes(b"original mislabeled evidence")
    inactive = {"path": false_path, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size": path.stat().st_size, "disposition": "quarantined", "reason": "wrong producer"}
    manifest = {"snapshot": "test", "shards": [], "inactive_shards": [inactive]}
    manifest_path = conf / "fixtures-manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(package_fixtures, "ROOT", tmp_path)
    monkeypatch.setattr(package_fixtures, "FIXTURES_DIR", store)
    monkeypatch.setattr(extract_fixtures, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(extract_fixtures, "FIXTURES_DIR", store)
    monkeypatch.setattr(extract_fixtures, "get_cache_root", lambda: tmp_path / "cache")
    return conf, store, manifest, manifest_path



def test_partial_package_rejects_drift_in_unselected_shards(evidence):
    _conf, store, manifest, manifest_path = evidence
    shard_path = "toolcalling/fixtures-stream-v1/inputs.tar.gz"
    shard_file = store / shard_path
    shard_file.parent.mkdir(parents=True)
    shard_file.write_bytes(b"current stream inputs")
    manifest["shards"] = [{
        "path": shard_path,
        "sha256": "0" * 64,
        "size": len(b"previous stream inputs"),
    }]
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="unselected fixture shard differs from manifest"):
        package_fixtures.merge_shards(
            [],
            prune=False,
            fixtures_dir=store,
            manifest_path=manifest_path,
            reject_unbuilt_drift=True,
        )


def test_partial_package_rejects_missing_unselected_shards(evidence):
    _conf, store, manifest, manifest_path = evidence
    shard_path = "toolcalling/fixtures-stream-v1/inputs.tar.gz"
    manifest["shards"] = [{"path": shard_path, "sha256": "0" * 64, "size": 1}]
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(FileNotFoundError, match="unselected fixture shard is missing"):
        package_fixtures.merge_shards(
            [],
            prune=False,
            fixtures_dir=store,
            manifest_path=manifest_path,
            reject_unbuilt_drift=True,
        )


def test_partial_package_preserves_unselected_shard_pin(evidence):
    _conf, store, manifest, manifest_path = evidence
    shard_path = "toolcalling/fixtures-stream-v1/inputs.tar.gz"
    shard_file = store / shard_path
    shard_file.parent.mkdir(parents=True)
    shard_file.write_bytes(b"stream inputs")
    digest = hashlib.sha256(shard_file.read_bytes()).hexdigest()
    shard = {"path": shard_path, "sha256": digest, "size": shard_file.stat().st_size}
    manifest["shards"] = [shard]
    manifest_path.write_text(json.dumps(manifest))

    assert package_fixtures.merge_shards(
        [],
        prune=False,
        fixtures_dir=store,
        manifest_path=manifest_path,
        reject_unbuilt_drift=True,
    ) == [shard]


def test_stream_only_package_rejects_unselected_drift_wiring(tmp_path, monkeypatch):
    conformance = tmp_path / "conformance"
    fixtures = conformance / "fixtures"
    history = conformance / "fixtures-unified-v2"
    fixtures.mkdir(parents=True)
    history.mkdir()
    (conformance / "fixtures-manifest.json").write_text(
        json.dumps({"snapshot": "old", "shards": [], "inactive_shards": []}) + "\n"
    )
    monkeypatch.setattr(package_fixtures, "ROOT", tmp_path)
    monkeypatch.setattr(package_fixtures, "FIXTURES_DIR", fixtures)
    monkeypatch.setattr(package_fixtures, "UNIFIED_HISTORY_DIR", history)
    staged_trees = []
    merge_options = []
    monkeypatch.setattr(
        package_fixtures,
        "stage_fixtures",
        lambda _source, _destination, *, trees=None: staged_trees.extend(trees or []),
    )
    monkeypatch.setattr(package_fixtures, "build_shards", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(package_fixtures, "sync_store", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        package_fixtures,
        "merge_shards",
        lambda _shards, _prune, **kwargs: merge_options.append(kwargs) or [],
    )
    monkeypatch.setattr(package_fixtures, "_validate_candidate_package", lambda *_args, **_kwargs: None)

    package_fixtures.package_snapshot(
        "new", "new America/Los_Angeles", {}, {}, dry_run=True, prune=False, stream_only=True
    )

    assert staged_trees == ["toolcalling/fixtures-stream-v1"]
    assert merge_options[0]["reject_unbuilt_drift"] is True
