# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import capture_cli
import capture_driver
import legacy_history


def _capture(tmp_path, monkeypatch, dynamo_json=None, version=None):
    source = tmp_path / "conformance/toolcalling/fixtures-batch-v1/qwen3/TOOLCALLING.batch.1.yaml"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(yaml.safe_dump({"family": "qwen3", "mode": "batch", "cases": {"A": {"model_text": "hello"}}}))
    monkeypatch.delenv("VLLM_RUST_SOURCE", raising=False)
    monkeypatch.setattr(capture_driver, "_copy_worker", lambda _containers: None)
    monkeypatch.setattr(capture_driver, "_container_capture", lambda _container, impl, *_args: (
        "0.26.0" if impl == "vllm" else "0.5.14", {str(source): {"cases": {"A": {"calls": [], "normal_text": "hello"}}}},
    ))
    args = SimpleNamespace(root=str(tmp_path), work=str(tmp_path / "work"), family=None, fixture=None,
                           vllm_rust_source=None, vllm_container="unused", sglang_container="unused",
                           dynamo_v2_json=dynamo_json, dynamo_v2_version=version)
    capture_driver._run_batch_on_stream(args)
    return tmp_path / "conformance" / legacy_history.CORPORA["batch_on_stream"] / "qwen3/TOOLCALLING.batch.1.yaml"


def test_optional_rust_absence_round_trips_without_inventing_capture(tmp_path, monkeypatch):
    output = _capture(tmp_path, monkeypatch)
    original = yaml.safe_load(output.read_bytes())
    store = tmp_path / "store"
    legacy_history.update_from_loose(store, tmp_path / "conformance")
    assert not list(store.glob("batch_on_stream/families/qwen3/vllm_rust-*.yaml"))
    materialized = tmp_path / "materialized"
    legacy_history.materialize_store(store, materialized)
    restored = yaml.safe_load((materialized / output.relative_to(tmp_path / "conformance")).read_bytes())
    assert restored == original
    assert list(restored["cases"]["A"]) == list(original["cases"]["A"])
    assert list(restored["captured_with"]) == list(original["captured_with"])
    assert legacy_history.update_from_loose(store, materialized) == []


@pytest.mark.parametrize("block", [{"calls": []}, {"error": "failed"}, {"unavailable": "absent", "calls": []}, {"unavailable": ""}])
def test_unlabeled_observation_still_fails_closed(tmp_path, monkeypatch, block):
    output = _capture(tmp_path, monkeypatch)
    document = yaml.safe_load(output.read_bytes())
    document["cases"]["A"]["vllm_rust"] = block
    output.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match="result has no capture label"):
        legacy_history.update_from_loose(tmp_path / "store", tmp_path / "conformance")
    assert not list((tmp_path / "store").rglob("*.yaml"))


def test_uncaptured_header_cannot_smuggle_observations(tmp_path, monkeypatch):
    _capture(tmp_path, monkeypatch)
    store = tmp_path / "store"
    legacy_history.update_from_loose(store, tmp_path / "conformance")
    header = store / "batch_on_stream/families/qwen3/inputs_and_golden.yaml"
    record = yaml.safe_load(header.read_bytes())
    record["documents"]["TOOLCALLING.batch.1.yaml"]["uncaptured"]["A"]["vllm_rust"] = {"calls": []}
    header.write_text(yaml.safe_dump(record))
    with pytest.raises(ValueError, match="result has no capture label"):
        legacy_history.materialize_store(store, tmp_path / "out")
    assert not list((tmp_path / "out").rglob("*.yaml"))


@pytest.mark.parametrize("nested", [False, True])
def test_recorded_dynamo_versions_append_without_rewriting_history(tmp_path, monkeypatch, nested):
    capture = tmp_path / "dynamo.json"
    capture.write_text(json.dumps({"A": {"calls": [], "normal_text": "historical"}}))
    output = _capture(tmp_path, monkeypatch, capture, "0.6.0")
    document = yaml.safe_load(output.read_bytes())
    document["captured_with"]["dynamo_v2"] = "Dynamo parser v2"
    output.write_text(yaml.safe_dump(document))
    store = tmp_path / "store"
    legacy_history.update_from_loose(store, tmp_path / "conformance")
    historical = store / "batch_on_stream/families/qwen3/dynamo_v2-unversioned.yaml"
    historical_bytes = historical.read_bytes()
    for version, text in [("0.7.0", "first"), ("0.8.0", "second")]:
        cases = {"A": {"calls": [], "normal_text": text}}
        capture.write_text(json.dumps({"captured_with": {"dynamo_v2": version}, "cases": {"qwen3": cases} if nested else cases}))
        output = _capture(tmp_path, monkeypatch, capture)
        legacy_history.update_from_loose(store, tmp_path / "conformance")
        assert yaml.safe_load(output.read_bytes())["captured_with"]["dynamo_v2"] == version
        assert (historical.parent / f"dynamo_v2-{version}.yaml").is_file()
        assert historical.read_bytes() == historical_bytes
    materialized = tmp_path / "materialized"
    legacy_history.materialize_store(store, materialized)
    final = yaml.safe_load((materialized / output.relative_to(tmp_path / "conformance")).read_bytes())
    assert final["captured_with"]["dynamo_v2"] == "0.8.0"
    assert final["cases"]["A"]["dynamo_v2"]["normal_text"] == "second"
    assert legacy_history.update_from_loose(store, materialized) == []


def test_imported_dynamo_json_needs_known_producer(tmp_path):
    path = tmp_path / "dynamo.json"
    cases = {"A": {"calls": []}}
    path.write_text(json.dumps(cases))
    with pytest.raises(ValueError, match="explicit --dynamo-v2-version"):
        capture_driver._load_dynamo_v2(path)
    assert capture_driver._load_dynamo_v2(path, "0.7.0") == (cases, "0.7.0")
    path.write_text(json.dumps({"captured_with": {"dynamo_v2": "0.7.0"}, "cases": cases}))
    with pytest.raises(ValueError, match="differs from recorder provenance"):
        capture_driver._load_dynamo_v2(path, "0.8.0")
    with pytest.raises(ValueError, match="requires --dynamo-v2-json"):
        capture_driver._load_dynamo_v2(None, "0.8.0")


@pytest.mark.parametrize("command", ["dynamo-batch-on-stream", "batch-on-stream"])
def test_capture_cli_requests_recorder_provenance(tmp_path, capsys, command):
    output = str(tmp_path / "dynamo.json")
    option = "--output" if command == "dynamo-batch-on-stream" else "--capture-dynamo-v2-json"
    capture_cli.main([command, option, output, "--dry-run"])
    assert "--bin record_batch_via_stream -- --with-provenance" in capsys.readouterr().out


def test_capture_cli_forwards_explicit_legacy_version(tmp_path, capsys):
    capture_cli.main(["batch-on-stream", "--dynamo-v2-json", str(tmp_path / "old.json"),
                      "--dynamo-v2-version", "0.6.0", "--dry-run"])
    assert "--dynamo-v2-version 0.6.0" in capsys.readouterr().out


@pytest.mark.parametrize("with_provenance", [False, True])
def test_merge_accepts_both_recorder_formats(tmp_path, with_provenance):
    dynamo = tmp_path / "dynamo.json"
    cases = {"A": {"calls": [{"name": "get_weather", "arguments": {"city": "Paris"}}]}}
    dynamo.write_text(json.dumps({"captured_with": {"dynamo_v2": "0.7.0"}, "cases": cases} if with_provenance else cases))
    peer = tmp_path / "peer.json"
    peer.write_text("{}")
    output = tmp_path / "merged.json"
    capture_driver._run_merge(SimpleNamespace(dynamo_v2=dynamo, dynamo_v2_version=None,
                                              vllm_python=peer, sglang=peer, output=output))
    assert json.loads(output.read_text()) == {"A": {"dynamo_v2": cases["A"],
                                                     "vllm_python": {"calls": []}, "sglang_python": {"calls": []}}}
