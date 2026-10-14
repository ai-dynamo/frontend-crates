# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verified recaptures retain legacy evidence and cannot rewrite bound results."""

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import unified_history
from capture_stimulus import select_peer_observation
from capture_unified_job import build_job
from import_sglang_unified import digest, import_capture, import_captures
from conformance.utils.tests.test_unified_history import _change, _request, _write_capture, _write_family


def _setup(tmp_path, *, bound=False):
    store = tmp_path / "store"
    _write_family(store)
    change = _change("old output")
    if not bound:
        change["stimulus"] = {"unavailable": {"code": "original_request_not_retained"}}
    _write_capture(store, "0.5.14", {"text_only": change}, implementation="sglang_python")
    _write_capture(store, "0.5.16", {}, implementation="sglang_python")
    job = tmp_path / "job.json"
    job.write_text(json.dumps(build_job(store / "families", "gemma4")))
    capture = tmp_path / "capture.yaml"
    capture.write_text(unified_history.dump_yaml({"sglang_version": "0.5.14", "results": {
        "UNIFIED.text_only.gemma4": {"capture_input": _request(),
                                      "chunks": [[{"kind": "text", "text": "hello"}]]},
    }}))
    provenance = {"version": "0.5.14", "job_sha256": digest(job),
                  "source_sha256": "a" * 64, "adapter_sha256": "b" * 64}
    return store, job, capture, provenance


def test_recapture_preserves_original_and_later_observations(tmp_path):
    store, job, capture, provenance = _setup(tmp_path)
    history = unified_history.load_store(store).histories[("gemma4", "sglang_python")]
    before = history.resolve("sglang_python-0.5.14")
    later = history.resolve("sglang_python-0.5.16")
    import_capture(store, job, capture, provenance)
    history = unified_history.load_store(store).histories[("gemma4", "sglang_python")]
    after = history.resolve("sglang_python-0.5.14")["text_only"]
    assert after["observation"] == before["text_only"]["observation"]
    assert after["stimulus"] == before["text_only"]["stimulus"]
    assert history.resolve("sglang_python-0.5.16") == later
    recapture = after["document"]["record_metadata"]["verified_recapture"]
    assert recapture["result"]["assembled"] == [{"kind": "text", "text": "hello"}]
    record, document = unified_history.materialized_record(history.family.cases["text_only"], after)
    record.update(document["record_metadata"])
    selected = select_peer_observation(record, _request(), implementation="sglang_python", version="0.5.14")
    assert selected["assembled"] == recapture["result"]["assembled"]
    assert "historical_capture_note" in selected
    assert import_capture(store, job, capture, provenance) == []


def test_bound_observation_cannot_be_overwritten(tmp_path):
    store, job, capture, provenance = _setup(tmp_path, bound=True)
    with pytest.raises(ValueError, match="bound SGLang observation is immutable"):
        import_capture(store, job, capture, provenance)


def test_job_must_match_final_candidate(tmp_path):
    store, job, capture, provenance = _setup(tmp_path)
    payload = json.loads(job.read_text())
    payload["cases"][0]["input"] = "different"
    job.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="final candidate"):
        import_capture(store, job, capture, provenance)


@pytest.mark.parametrize("field", ["version", "source_sha256", "job_sha256", "resolution", "input"])
def test_recapture_selection_fails_closed(tmp_path, field):
    store, job, capture, provenance = _setup(tmp_path)
    import_capture(store, job, capture, provenance)
    history = unified_history.load_store(store).histories[("gemma4", "sglang_python")]
    record, document = unified_history.materialized_record(history.family.cases["text_only"], history.resolve("sglang_python-0.5.14")["text_only"])
    record.update(copy.deepcopy(document["record_metadata"]))
    supplement = record["verified_recapture"]
    if field == "resolution":
        supplement[field] = "pending"
    elif field == "input":
        supplement["result"]["capture_input"][field] = "different"
    else:
        supplement["provenance"][field] = "invalid"
    with pytest.raises(ValueError):
        select_peer_observation(record, _request(), implementation="sglang_python", version="0.5.14")


def test_batch_import_is_atomic_when_later_capture_is_invalid(tmp_path):
    store, job, capture, provenance = _setup(tmp_path)
    before = {path.relative_to(store): path.read_bytes() for path in store.rglob("*.yaml")}
    bad_capture = tmp_path / "bad.yaml"
    bad_capture.write_text(unified_history.dump_yaml({"sglang_version": "0.5.14", "results": {}}))
    with pytest.raises(ValueError, match="exact family job"):
        import_captures(store, [(job, capture, provenance), (job, bad_capture, provenance)])
    assert {path.relative_to(store): path.read_bytes() for path in store.rglob("*.yaml")} == before
