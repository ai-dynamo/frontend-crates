# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import capture_policy
import extract_fixtures
import fixture_disposition
import legacy_history
import maintain_captures
import unified_history
from test_unified_history import _store


@pytest.fixture
def repository(tmp_path):
    repo = tmp_path / "repo"
    conformance = repo / "conformance"
    _store(conformance / "fixtures-unified-v2")
    (conformance / "fixtures").mkdir()
    family = conformance / "fixtures-v1/batch/families/gemma4"
    family.mkdir(parents=True)
    records = {
        "inputs": {"one.yaml": {"family": "gemma4", "cases": {"A": {"model_text": "request"}}}},
        "vllm_python-1.0.0": {"one.yaml": {"family": "gemma4", "cases": {"A": {"expected": {"vllm_python": {"normal_text": "old"}}}}}},
        "vllm_python-1.1.0": {"one.yaml": {"family": "gemma4", "cases": {"A": {"expected": {"vllm_python": {"normal_text": "new"}}}}}},
        "vllm_python-1.2.0": {"one.yaml": {"family": "gemma4", "cases": {}}},
    }
    for capture, documents in records.items():
        path = family / ("inputs_and_golden.yaml" if capture == "inputs" else f"{capture}.yaml")
        path.write_text(yaml.safe_dump({"corpus": "batch", "family": "gemma4",
                                        "capture": capture, "documents": documents}))
    (conformance / "capture-policy.yaml").write_text(yaml.safe_dump({
        "schema_version": 1, "corpora": {corpus: {} for corpus in capture_policy.CORPORA}}))
    (conformance / "fixtures-manifest.json").write_text(json.dumps({"snapshot": "test", "shards": []}))
    maintain_captures._pin_manifest(conformance)
    return repo


def test_preview_preserves_bytes_and_apply_is_stable(repository):
    root = repository / "conformance"
    before = maintain_captures._file_inventory(root)
    evidence = maintain_captures.resolved_inventory(root)
    preview = maintain_captures.maintain(repository, "migrate")
    assert preview["changes"] and not preview["applied"]
    assert maintain_captures._file_inventory(root) == before
    applied = maintain_captures.maintain(repository, "migrate", apply=True)
    assert applied["changes"] == preview["changes"]
    assert maintain_captures.resolved_inventory(root) == evidence
    assert maintain_captures.maintain(repository, "migrate")["changes"] == []


def test_removal_retains_evidence_and_pins_tombstone(repository):
    root = repository / "conformance"
    before = maintain_captures.resolved_inventory(root)
    maintain_captures.maintain(repository, "remove", corpus="batch", capture="vllm_python-1.1.0", apply=True)
    after = maintain_captures.resolved_inventory(root)
    assert after == {key: value for key, value in before.items() if key != "batch/gemma4/vllm_python-1.1.0"}
    policy_path = root / "capture-policy.yaml"
    assert capture_policy.load_policy(policy_path)["corpora"]["batch"]["selectors"]["vllm_python-1.1.0"]["unavailable"]
    manifest = json.loads((root / "fixtures-manifest.json").read_text())
    assert fixture_disposition.capture_policy_pin(policy_path) in manifest["shards"]


def test_selected_removal_requires_explicit_disposition(repository):
    root = repository / "conformance"
    path = root / "capture-policy.yaml"
    policy = capture_policy.load_policy(path)
    policy["corpora"]["batch"]["references"] = {"vllm_python": {"selection": "vllm_python-1.1.0"}}
    path.write_text(yaml.safe_dump(policy))
    before = maintain_captures._file_inventory(root)
    with pytest.raises(ValueError, match="requires replacement or unavailable"):
        maintain_captures.maintain(repository, "remove", corpus="batch", capture="vllm_python-1.1.0", apply=True)
    assert maintain_captures._file_inventory(root) == before
    maintain_captures.maintain(repository, "remove", corpus="batch", capture="vllm_python-1.1.0",
                               unavailable="Measurement retired", apply=True)
    assert capture_policy.load_policy(path)["corpora"]["batch"]["references"]["vllm_python"] == {
        "selection": "unavailable", "reason": "Measurement retired"}


def test_equivalent_removal_survives_next_validation(repository):
    maintain_captures.maintain(repository, "remove", corpus="batch", capture="vllm_python-1.1.0",
                               replacement="vllm_python-1.2.0", apply=True)
    assert maintain_captures.maintain(repository, "migrate")["changes"] == []


@pytest.mark.parametrize("partial", [False, True])
def test_latest_selected_removal_requires_disposition(repository, tmp_path, partial):
    root = repository / "conformance"
    path = root / "capture-policy.yaml"
    policy = capture_policy.load_policy(path)
    policy["corpora"]["batch"]["references"] = {"vllm_python": {"selection": "latest"}}
    path.write_text(yaml.safe_dump(policy))
    if partial:
        source = root / "fixtures-v1/batch/families/gemma4"
        sibling = source.parent / "qwen3"
        sibling.mkdir()
        for original in source.glob("*.yaml"):
            record = yaml.safe_load(original.read_text())
            record["family"] = "qwen3"
            for document in record["documents"].values():
                document["family"] = "qwen3"
            (sibling / original.name).write_text(yaml.safe_dump(record))
    before = maintain_captures._file_inventory(root)
    kwargs = {"corpus": "batch", "capture": "vllm_python-1.2.0",
              "family": "gemma4" if partial else None, "apply": True}
    with pytest.raises(ValueError, match="requires"):
        maintain_captures.maintain(repository, "remove", **kwargs)
    assert maintain_captures._file_inventory(root) == before
    if partial:
        policy["corpora"]["batch"]["references"]["vllm_python"] = {
            "selection": "unavailable", "reason": "Retired reference"}
        candidate = tmp_path / "candidate-policy.yaml"
        candidate.write_text(yaml.safe_dump(policy))
        maintain_captures.maintain(repository, "remove", policy_path=candidate, **kwargs)
        assert (root / "fixtures-v1/batch/families/qwen3/vllm_python-1.2.0.yaml").exists()
    else:
        maintain_captures.maintain(repository, "remove", unavailable="Retired reference", **kwargs)
    assert capture_policy.load_policy(path)["corpora"]["batch"]["references"]["vllm_python"]["selection"] == "unavailable"


@pytest.mark.parametrize("failure", [OSError, KeyboardInterrupt])
def test_interrupted_publication_restores_stores_policy_and_manifest(repository, monkeypatch, failure):
    root = repository / "conformance"
    before = maintain_captures._file_inventory(root)
    replace = unified_history.os.replace
    fired = False

    def fail_policy_install(source, destination):
        nonlocal fired
        if Path(destination) == root / "capture-policy.yaml" and not fired:
            fired = True
            raise failure("interrupted policy publication")
        return replace(source, destination)

    monkeypatch.setattr(unified_history.os, "replace", fail_policy_install)
    with pytest.raises(failure, match="interrupted policy publication"):
        maintain_captures.maintain(repository, "migrate", apply=True)
    assert fired
    assert maintain_captures._file_inventory(root) == before
    assert not (root / unified_history.TRANSACTION_JOURNAL).exists()
    assert maintain_captures.resolved_inventory(root)


def test_policy_pin_participates_in_cache_identity(repository, monkeypatch, tmp_path):
    root = repository / "conformance"
    path = root / "capture-policy.yaml"
    pin = fixture_disposition.capture_policy_pin(path)
    original = extract_fixtures.fixtures_identity([pin])
    monkeypatch.setattr(extract_fixtures, "MANIFEST_PATH", root / "fixtures-manifest.json")
    assert extract_fixtures.shard_file(pin) == path
    extract_fixtures.materialize_shard(pin, path, tmp_path / "snapshot")
    assert (tmp_path / "snapshot/capture-policy.yaml").read_bytes() == path.read_bytes()
    path.write_text(path.read_text() + "# Policy revision\n")
    assert extract_fixtures.fixtures_identity([fixture_disposition.capture_policy_pin(path)]) != original
    with pytest.raises(ValueError, match="differs from the manifest pin"):
        extract_fixtures.shard_file(pin)
