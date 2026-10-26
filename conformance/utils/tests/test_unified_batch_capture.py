# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import unified_history
import capture_unified_batch
from capture_stimulus import batch_final_events, select_peer_observation, unavailable_result
from capture_unified_batch import capture_cases
from capture_unified_job import build_job
from import_peer_unified import digest, import_capture
from conformance.utils.tests.test_unified_history import _change, _request, _write_capture, _write_family


def test_batch_projection_retains_ordered_calls_but_cannot_claim_interleaving():
    events = [{"kind": "reasoning", "text": "r"}, {"kind": "text", "text": "t"},
              {"kind": "tool_call", "name": "one", "arguments": {}},
              {"kind": "tool_call", "name": "two", "arguments": "{malformed"}]
    final = {"reasoning_text": "r", "normal_text": "t", "calls": [
        {"name": "one", "arguments": {}}, {"name": "two", "arguments": "{malformed"}]}
    actual = batch_final_events(final)
    assert actual == events
    reversed_final = {**final, "calls": list(reversed(final["calls"]))}
    assert batch_final_events(reversed_final) != actual


def test_capture_accounts_for_every_request_without_default_substitution():
    cases = [{"id": str(index), "family": "family", "request": _request(text)} for index, text in enumerate(("plain", "bad", "guided", "finish"))]
    cases[2]["request"]["init"]["tool_output_mode"] = "Json"
    cases[3]["request"]["finish_reason"] = "unsupported"
    executed = []

    def execute(names, request):
        executed.append(copy.deepcopy(request))
        if request["input"] == "bad":
            raise ValueError("malformed arguments")
        return {"reasoning_text": "", "normal_text": request["input"], "calls": []}

    before = copy.deepcopy(cases)
    results = capture_cases({"cases": cases}, {"family": ("reasoning", "tools")}, lambda names: True, execute)
    assert results.keys() == {case["id"] for case in cases}
    assert len(executed) == 2
    assert "batch_final" in results["0"]
    assert results["1"]["capture_observation"]["error"]["code"] == "parser_exception"
    for key in ("2", "3"):
        assert results[key]["capture_observation"]["unavailable"]["code"] == "unsupported_request_mode"
    assert cases == before
    for case in cases:
        assert results[case["id"]]["capture_input"] == case["request"]
        assert "chunks" not in results[case["id"]]
    absent = capture_cases({"cases": cases}, {}, lambda names: False, execute)
    assert all(result["capture_observation"]["unavailable"]["code"] == "unsupported_family" for result in absent.values())
    with pytest.raises(ValueError, match="duplicate"):
        capture_cases({"cases": [cases[0], cases[0]]}, {}, lambda names: False, execute)


def test_batch_history_is_separate_sparse_bound_and_immutable(tmp_path):
    store = tmp_path / "store"
    _write_family(store)
    _write_capture(store, "0.26.0", {"text_only": _change("unified")}, implementation="vllm_python")
    old = (store / "families/gemma4/vllm_python-0.26.0.yaml").read_bytes()
    job = tmp_path / "job.json"
    job.write_text(json.dumps(build_job(store / "families", "gemma4", mode="batch")))
    capture = tmp_path / "capture.json"
    final = {"reasoning_text": "", "normal_text": "batch \U0001f600", "calls": []}
    result = {"capture_input": _request(), "assembled": batch_final_events(final), "batch_final": final}
    provenance = {"implementation": "vllm_python_batch", "version": "0.26.0", "job_sha256": digest(job), "source_sha256": "a" * 64, "adapter_sha256": "b" * 64}
    for version in ("0.26.0", "0.27.0"):
        capture.write_text(json.dumps({"version": version, "provenance": {**provenance, "version": version}, "results": {"UNIFIED.text_only.gemma4": result}}))
        import_capture(store, job, capture, {**provenance, "version": version}, implementation="vllm_python_batch")
    assert (store / "families/gemma4/vllm_python-0.26.0.yaml").read_bytes() == old
    history = unified_history.load_store(store).histories[("gemma4", "vllm_python_batch")]
    assert history.captures["vllm_python_batch-0.27.0"]["changes"] == {}
    state = history.resolve("vllm_python_batch-0.27.0")["text_only"]
    assert state["document"]["inherited_from"] == "0.26.0"
    assert state["document"]["capture_origin"]["source_sha256"] == "a" * 64
    assert state["observation"]["value"] == {"assembled": result["assembled"]}
    assert import_capture(store, job, capture, {**provenance, "version": "0.27.0"}, implementation="vllm_python_batch") == []
    changed = copy.deepcopy(result)
    changed["capture_input"]["init"]["starting_state"] = "Reasoning"
    capture.write_text(json.dumps({"version": "0.27.0", "provenance": {**provenance, "version": "0.27.0"}, "results": {"UNIFIED.text_only.gemma4": changed}}))
    with pytest.raises(ValueError, match="request differs"):
        import_capture(store, job, capture, {**provenance, "version": "0.27.0"}, implementation="vllm_python_batch")
    changed = {**result, "assembled": []}
    capture.write_text(json.dumps({"version": "0.27.0", "provenance": {**provenance, "version": "0.27.0"}, "results": {"UNIFIED.text_only.gemma4": changed}}))
    with pytest.raises(ValueError, match="executed batch differs from final fields"):
        import_capture(store, job, capture, {**provenance, "version": "0.27.0"}, implementation="vllm_python_batch")


def test_jail_history_preserves_the_same_version_direct_batch_capture(tmp_path):
    store = tmp_path / "store"
    _write_family(store)
    _write_capture(store, "0.26.0", {"text_only": _change("unified")}, implementation="vllm_python")
    job = tmp_path / "job.json"
    job.write_text(json.dumps(build_job(store / "families", "gemma4", mode="batch")))
    capture = tmp_path / "capture.json"
    for implementation, text in (("dynamo_v1", "direct batch"), ("dynamo_v1_jail", "jail output")):
        final = {"reasoning_text": "", "normal_text": text, "calls": []}
        result = {"capture_input": _request(), "batch_final": final, "assembled": batch_final_events(final)}
        provenance = {"implementation": implementation, "version": "10.0.1", "job_sha256": digest(job),
                      "source_sha256": "a" * 64, "adapter_sha256": "b" * 64}
        provenance["results_sha256"] = hashlib.sha256(json.dumps({"UNIFIED.text_only.gemma4": result}, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        capture.write_text(json.dumps({"version": "10.0.1", "provenance": provenance,
                                       "results": {"UNIFIED.text_only.gemma4": result}}))
        import_capture(store, job, capture, provenance, implementation=implementation)
        if implementation == "dynamo_v1":
            direct_bytes = (store / "families/gemma4/dynamo_v1-10.0.1.yaml").read_bytes()
    assert (store / "families/gemma4/dynamo_v1-10.0.1.yaml").read_bytes() == direct_bytes
    histories = unified_history.load_store(store).histories
    assert histories[("gemma4", "dynamo_v1")].resolve("dynamo_v1-10.0.1")["text_only"]["observation"] != histories[("gemma4", "dynamo_v1_jail")].resolve("dynamo_v1_jail-10.0.1")["text_only"]["observation"]


def test_batch_import_rejects_a_different_embedded_producer(tmp_path):
    store = tmp_path / "store"
    _write_family(store)
    job = tmp_path / "job.json"
    job.write_text(json.dumps(build_job(store / "families", "gemma4", mode="batch")))
    provenance = {"implementation": "vllm_python_batch", "version": "0.26.0", "job_sha256": digest(job), "source_sha256": "a" * 64, "adapter_sha256": "b" * 64}
    final = {"reasoning_text": "", "normal_text": "hello", "calls": []}
    result = {"capture_input": _request(), "assembled": batch_final_events(final), "batch_final": final}
    for field in ("implementation", "source_sha256", "adapter_sha256", "job_sha256", "version"):
        capture = tmp_path / "capture.json"
        capture.write_text(json.dumps({"version": "0.26.0", "provenance": {**provenance, field: "c" * 64}, "results": {"UNIFIED.text_only.gemma4": result}}))
        with pytest.raises(ValueError, match="producer receipt"):
            import_capture(store, job, capture, provenance, implementation="vllm_python_batch")


def test_batch_adapter_correction_preserves_original_and_binds_new_producer(tmp_path):
    store = tmp_path / "store"
    _write_family(store)
    _write_capture(store, "0.26.0", {"text_only": _change("unified")}, implementation="vllm_python")
    job = tmp_path / "job.json"
    job.write_text(json.dumps(build_job(store / "families", "gemma4", mode="batch")))
    provenance = {"implementation": "dynamo_v1", "version": "10.0.1", "job_sha256": digest(job), "source_sha256": "a" * 64, "adapter_sha256": "b" * 64}
    capture = tmp_path / "capture.json"
    request = _request()
    unavailable = unavailable_result("unsupported_request_mode", "old adapter", capture=request, observation=True)
    provenance["results_sha256"] = hashlib.sha256(json.dumps({"UNIFIED.text_only.gemma4": unavailable}, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    capture.write_text(json.dumps({"version": "10.0.1", "provenance": provenance, "results": {"UNIFIED.text_only.gemma4": unavailable}}))
    import_capture(store, job, capture, provenance, implementation="dynamo_v1")
    before = unified_history.load_store(store).histories[("gemma4", "dynamo_v1")].resolve("dynamo_v1-10.0.1")["text_only"]
    new_provenance = {**provenance, "adapter_sha256": "c" * 64}
    final = {"reasoning_text": "", "normal_text": "hello", "calls": []}
    result = {"capture_input": request, "batch_final": final, "assembled": batch_final_events(final)}
    new_provenance["results_sha256"] = hashlib.sha256(json.dumps({"UNIFIED.text_only.gemma4": result}, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    capture.write_text(json.dumps({"version": "10.0.1", "provenance": new_provenance, "results": {"UNIFIED.text_only.gemma4": result}}))
    import_capture(store, job, capture, new_provenance, implementation="dynamo_v1")
    after = unified_history.load_store(store).histories[("gemma4", "dynamo_v1")].resolve("dynamo_v1-10.0.1")["text_only"]
    assert after["observation"] == before["observation"]
    assert after["stimulus"] == before["stimulus"]
    assert after["document"]["capture_origin"] == before["document"]["capture_origin"]
    record, metadata = unified_history.materialized_record(unified_history.load_store(store).families["gemma4"].cases["text_only"], after)
    record.update(metadata["record_metadata"])
    selected = select_peer_observation(record, request, implementation="dynamo_v1", version="10.0.1")
    assert selected["batch_final"] == final
    assert selected["producer_origin"]["adapter_sha256"] == "c" * 64
    assert import_capture(store, job, capture, new_provenance, implementation="dynamo_v1") == []
    provenance["results_sha256"] = hashlib.sha256(json.dumps({"UNIFIED.text_only.gemma4": unavailable}, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    capture.write_text(json.dumps({"version": "10.0.1", "provenance": provenance, "results": {"UNIFIED.text_only.gemma4": unavailable}}))
    assert import_capture(store, job, capture, provenance, implementation="dynamo_v1") == []
    restored = unified_history.load_store(store).histories[("gemma4", "dynamo_v1")].resolve("dynamo_v1-10.0.1")["text_only"]
    assert restored == after


def test_peer_batch_adapter_identity_binds_registry_bytes(tmp_path, monkeypatch):
    package = tmp_path / "vllm"
    source = package / "parser"
    source.mkdir(parents=True)
    (source / "reader.py").write_text("producer source")
    config = tmp_path / "parser_families.yaml"
    config.write_text("first registry")
    monkeypatch.setattr(capture_unified_batch.importlib, "import_module", lambda _name: SimpleNamespace(__file__=str(package / "__init__.py"), __version__="1.0.0"))
    monkeypatch.setattr(capture_unified_batch, "parser_families_path", lambda: config)
    before = capture_unified_batch.source_identity("vllm")
    config.write_text("changed registry")
    after = capture_unified_batch.source_identity("vllm")
    assert before["source_sha256"] == after["source_sha256"]
    assert before["adapter_sha256"] != after["adapter_sha256"]
    assert before["adapter_files"][config.name] != after["adapter_files"][config.name]
