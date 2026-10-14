# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backfills preserve historical outputs and each peer's final-message contract."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import unified_history
from capture_stimulus import select_peer_observation
from capture_unified_job import build_job
from import_peer_unified import digest, import_capture, main
from conformance.utils.tests.test_unified_history import _add_backfill_case, _change, _request, _write_capture, _write_family


@pytest.mark.parametrize("implementation", ["vllm_python", "vllm_rust", "sglang_python"])
@pytest.mark.parametrize("later_changes", [False, True])
@pytest.mark.parametrize("override_provenance", [False, True])
def test_older_import_preserves_later_snapshot_defaults_and_overrides(
    tmp_path, implementation, later_changes, override_provenance,
):
    # Provenance reset/inheritance is storage behavior, outside the parser corpus schema.
    store = tmp_path / "store"
    _write_family(store)
    _write_capture(store, "0.26.0", {"text_only": _change("one")}, implementation=implementation)
    overrides = {"text_only": {"capture_provenance": {
        "captured_with": {implementation: "0.26.0"},
        "capture_origin": None, "inherited_from": None,
    }}} if override_provenance else {}
    _write_capture(
        store, "0.27.0", {"text_only": _change("two")} if later_changes else {},
        implementation=implementation, document_overrides=overrides,
    )
    history = unified_history.load_store(store).histories[("gemma4", implementation)]
    snapshots = {key: history.resolve(key) for key in history.ordered_capture_ids()}
    job = tmp_path / "job.json"
    job.write_text(json.dumps(build_job(store / "families", "gemma4")))
    capture = tmp_path / "capture.yaml"
    version_key = "sglang_version" if implementation == "sglang_python" else (
        "vllm_version" if implementation == "vllm_python" else "vllm_rust_version"
    )
    capture.write_text(unified_history.dump_yaml({version_key: "0.25.1", "results": {
        "UNIFIED.text_only.gemma4": {
            "capture_input": _request(), "assembled": [{"kind": "text", "text": "zero"}],
            "chunks": [[{"kind": "text", "text": "zero"}]],
        },
    }}))
    provenance = {"implementation": implementation, "version": "0.25.1",
                  "job_sha256": digest(job), "source_sha256": "a" * 64,
                  "adapter_sha256": "b" * 64}
    import_capture(store, job, capture, provenance, implementation=implementation)
    history = unified_history.load_store(store).histories[("gemma4", implementation)]
    assert {key: history.resolve(key) for key in snapshots} == snapshots
    assert history.resolve(f"{implementation}-0.25.1")["text_only"]["document"]["captured_with"] == {
        implementation: "0.25.1",
    }
    assert import_capture(
        store, job, capture, provenance, implementation=implementation, backfill_only=True,
    ) == []


@pytest.fixture(params=["vllm_python", "vllm_rust"])
def peer_capture(request, tmp_path):
    implementation = request.param
    store = tmp_path / "store"
    _write_family(store)
    _write_capture(store, "0.25.1", {}, implementation=implementation)
    job = tmp_path / "job.json"
    job.write_text(json.dumps(build_job(store / "families", "gemma4")))
    capture = tmp_path / "capture.yaml"
    version_key = "vllm_version" if implementation == "vllm_python" else "vllm_rust_version"
    capture.write_text(unified_history.dump_yaml({version_key: "0.25.1", "results": {
        "UNIFIED.text_only.gemma4": {
            "capture_input": _request(),
            "assembled": [{"kind": "text", "text": "batch"}],
            "chunks": [[{"kind": "text", "text": "stream"}]],
        },
    }}))
    provenance = {"implementation": implementation, "version": "0.25.1",
                  "job_sha256": digest(job), "source_sha256": "a" * 64,
                  "adapter_sha256": "b" * 64}
    return store, job, capture, provenance, implementation


def test_missing_result_backfill_keeps_python_batch_and_rust_stream_assembly(peer_capture):
    store, job, capture, provenance, implementation = peer_capture
    _write_capture(store, "0.26.0", {}, implementation=implementation)
    import_capture(store, job, capture, provenance, implementation=implementation, backfill_only=True)
    history = unified_history.load_store(store).histories[("gemma4", implementation)]
    result = history.resolve(f"{implementation}-0.25.1")["text_only"]
    record, _document = unified_history.materialized_record(history.family.cases["text_only"], result)
    expected = "batch" if implementation == "vllm_python" else "stream"
    assert record["assembled"] == [{"kind": "text", "text": expected}]
    assert record["capture_input"] == _request()
    assert history.resolve(f"{implementation}-0.26.0") == {}


def test_backfill_retains_existing_bound_result_and_later_version(peer_capture):
    store, job, capture, provenance, implementation = peer_capture
    _write_capture(store, "0.25.1", {"text_only": _change("historic")}, implementation=implementation)
    _write_capture(store, "0.26.0", {}, implementation=implementation)
    history = unified_history.load_store(store).histories[("gemma4", implementation)]
    snapshots = {key: history.resolve(key) for key in history.ordered_capture_ids()}
    assert import_capture(store, job, capture, provenance, implementation=implementation, backfill_only=True) == []
    history = unified_history.load_store(store).histories[("gemma4", implementation)]
    assert {key: history.resolve(key) for key in history.ordered_capture_ids()} == snapshots


def test_existing_older_backfill_preserves_later_provenance(peer_capture):
    store, job, capture, provenance, implementation = peer_capture
    _add_backfill_case(store)
    overrides = {"text_only": {"capture_provenance": {
        "captured_with": {implementation: "0.24.0"},
        "capture_origin": None, "inherited_from": None,
    }}}
    _write_capture(store, "0.25.1", {"text_only": _change("old")},
                   implementation=implementation, document_overrides=overrides)
    _write_capture(store, "0.26.0", {"text_only": _change("later")},
                   implementation=implementation, document_overrides=overrides)
    history = unified_history.load_store(store).histories[("gemma4", implementation)]
    later = history.resolve(f"{implementation}-0.26.0")
    job.write_text(json.dumps(build_job(store / "families", "gemma4")))
    payload = unified_history.load_yaml(capture)
    payload["results"]["UNIFIED.new_case.gemma4"] = {
        "capture_input": _request("new"), "assembled": [{"kind": "text", "text": "new"}],
        "chunks": [[{"kind": "text", "text": "new"}]],
    }
    capture.write_text(unified_history.dump_yaml(payload))
    provenance["job_sha256"] = digest(job)
    import_capture(store, job, capture, provenance, implementation=implementation, backfill_only=True)
    history = unified_history.load_store(store).histories[("gemma4", implementation)]
    assert "new_case" in history.resolve(f"{implementation}-0.25.1")
    assert history.resolve(f"{implementation}-0.26.0") == later


@pytest.mark.parametrize("fresh_unavailable", [False, True])
def test_new_execution_preserves_unavailable_adapter_evidence(peer_capture, fresh_unavailable):
    store, job, capture, provenance, implementation = peer_capture
    change = _change("unused")
    change["observation"] = {"unavailable": {"code": "adapter_parser_not_registered", "detail": "old adapter lacked this parser"}}
    _write_capture(store, "0.25.1", {"text_only": change}, implementation=implementation)
    _write_capture(store, "0.26.0", {}, implementation=implementation)
    history = unified_history.load_store(store).histories[("gemma4", implementation)]
    before = history.resolve(f"{implementation}-0.25.1")["text_only"]
    later = history.resolve(f"{implementation}-0.26.0")
    if fresh_unavailable:
        version_key = "vllm_version" if implementation == "vllm_python" else "vllm_rust_version"
        capture.write_text(unified_history.dump_yaml({version_key: "0.25.1", "results": {
            "UNIFIED.text_only.gemma4": {
                "capture_input": _request(), "unavailable": "GuidedJson request mode is unsupported",
            },
        }}))
    import_capture(store, job, capture, provenance, implementation=implementation, backfill_only=True)
    history = unified_history.load_store(store).histories[("gemma4", implementation)]
    after = history.resolve(f"{implementation}-0.25.1")["text_only"]
    assert after["stimulus"] == before["stimulus"]
    assert after["observation"] == before["observation"]
    assert history.resolve(f"{implementation}-0.26.0") == later
    record, document = unified_history.materialized_record(history.family.cases["text_only"], after)
    record.update(document["record_metadata"])
    selected = select_peer_observation(record, _request(), implementation=implementation, version="0.25.1")
    if fresh_unavailable:
        assert selected["unavailable"] == "GuidedJson request mode is unsupported"
        assert "was not executed" in selected["historical_capture_note"]
    else:
        assert "unavailable" not in selected
    assert "historical_capture_note" in selected
    with pytest.raises(ValueError, match="implementation"):
        select_peer_observation(record, _request(), implementation="vllm_python" if implementation == "vllm_rust" else "vllm_rust", version="0.25.1")


def test_parser_error_retains_ordered_partial_chunks(peer_capture):
    store, job, capture, provenance, implementation = peer_capture
    payload = {
        "vllm_version" if implementation == "vllm_python" else "vllm_rust_version": "0.25.1",
        "results": {"UNIFIED.text_only.gemma4": {
            "capture_input": _request(), "error": "ParserError: malformed call",
            "chunks": [[{"kind": "text", "text": "before error"}]],
        }},
    }
    capture.write_text(unified_history.dump_yaml(payload))
    import_capture(store, job, capture, provenance, implementation=implementation)
    history = unified_history.load_store(store).histories[("gemma4", implementation)]
    record, document = unified_history.materialized_record(history.family.cases["text_only"], history.resolve(f"{implementation}-0.25.1")["text_only"])
    record.update(document["record_metadata"])
    assert record["error"] == "ParserError: malformed call"
    assert record["partial_chunks"] == [{"expected": [{"kind": "text", "text": "before error"}]}]


def test_backfill_keeps_each_releases_unavailable_explanation(peer_capture):
    store, job, capture, provenance, implementation = peer_capture
    code = "vllm_rust_parser_not_registered"
    later = _change("unused")
    later["observation"] = {"unavailable": {"code": code, "detail": "No parser at 0.26.0."}}
    _write_capture(store, "0.26.0", {"text_only": later}, implementation=implementation)
    version_key = "vllm_version" if implementation == "vllm_python" else "vllm_rust_version"
    capture.write_text(unified_history.dump_yaml({version_key: "0.25.1", "results": {
        "UNIFIED.text_only.gemma4": {
            "capture_input": _request(), "unavailable": "No parser at 0.25.1.",
            "capture_observation": {"unavailable": {"code": code, "detail": "No parser at 0.25.1."}},
        },
    }}))
    import_capture(store, job, capture, provenance, implementation=implementation, backfill_only=True)
    history = unified_history.load_store(store).histories[("gemma4", implementation)]
    assert history.resolve(f"{implementation}-0.25.1")["text_only"]["observation"]["unavailable"]["detail"] == "No parser at 0.25.1."
    assert history.resolve(f"{implementation}-0.26.0")["text_only"]["observation"]["unavailable"]["detail"] == "No parser at 0.26.0."


@pytest.mark.parametrize("nested_source", [False, True])
@pytest.mark.parametrize("structured_job", [False, True])
@pytest.mark.parametrize("corrupt", [None, "adapter", "job"])
def test_cli_verifies_each_frozen_bundle_format(peer_capture, tmp_path, monkeypatch, nested_source, structured_job, corrupt):
    store, job, capture, provenance, implementation = peer_capture
    bundle = tmp_path / "bundle"
    source = bundle / "utils/src" if nested_source else bundle
    source.mkdir(parents=True)
    adapter = Path(__file__).resolve().parents[1] / "src/capture_stimulus.py"
    (source / adapter.name).write_bytes(adapter.read_bytes())
    expected_job = digest(job)
    if corrupt == "job":
        expected_job = "c" * 64
    manifest = bundle / "manifest.json"
    manifest.write_text(json.dumps({"files": {adapter.name: digest(adapter)},
        "jobs": {"gemma4": {"sha256": expected_job} if structured_job else expected_job}}))
    provenance["adapter_sha256"] = digest(manifest)
    provenance_path = tmp_path / "provenance.json"
    provenance_path.write_text(json.dumps(provenance))
    if corrupt == "adapter":
        (source / adapter.name).write_bytes(b"changed adapter")
    before = {path.relative_to(store): path.read_bytes() for path in store.rglob("*.yaml")}
    monkeypatch.setattr(sys, "argv", ["import_peer_unified.py", "--store", str(store),
        "--job", str(job), "--capture", str(capture), "--provenance", str(provenance_path),
        "--bundle", str(bundle), "--implementation", implementation, "--backfill-only"])
    if corrupt:
        with pytest.raises(ValueError, match="adapter differs" if corrupt == "adapter" else "job differs"):
            main()
        assert {path.relative_to(store): path.read_bytes() for path in store.rglob("*.yaml")} == before
    else:
        main()
        history = unified_history.load_store(store).histories[("gemma4", implementation)]
        assert "text_only" in history.resolve(f"{implementation}-0.25.1")
