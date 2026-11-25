# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from capture_stimulus import batch_final_events
import replay_latest
import capture_sources
import normalized_yaml
import unified_history
import yaml


def _execution(implementation="dynamo_v1"):
    request = {"input": "authored", "init": {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None},
               "finish_reason": "stop", "tools": [], "chunks": [{"delta_text": "authored"}]}
    case = {"id": "UNIFIED.text_only.jamba", "case_id": "text_only", "request": request}
    job = {"family": "jamba", "cases": [case]}
    raw = json.dumps(job).encode()
    final = {"normal_text": "authored", "reasoning_text": "", "calls": []}
    record = {"batch_final": final, "assembled": batch_final_events(final), "capture_input": request}
    results = {case["id"]: record}
    document = {"version": "11.0.13", "results": results, "provenance": {
        "implementation": implementation, "version": "11.0.13", "source_sha256": "a" * 64,
        "adapter_sha256": "b" * 64, "job_sha256": hashlib.sha256(raw).hexdigest(),
        "results_sha256": hashlib.sha256(json.dumps(results, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
    }}
    return job, document, {"text_only": deepcopy(record)}, raw


def _compare(job, document, recorded, raw, implementation="dynamo_v1"):
    return replay_latest.compare_results(job, document, recorded, implementation, "11.0.13", raw)


def test_batch_and_jail_replay_accept_their_own_final_fields():
    for implementation in ("dynamo_v1", "dynamo_v1_jail"):
        job, document, recorded, raw = _execution(implementation)
        assert _compare(job, document, recorded, raw, implementation) == {"executed": 1}


@pytest.mark.parametrize("arguments,accepted", [
    ('{ "b": [2], "a": 9007199254740993 }', True),
    ('{"a":9007199254740992,"b":[2]}', False),
    ('{"a":"9007199254740993","b":[2]}', False),
    ('{"a":9007199254740993,"b":[3]}', False),
])
def test_replay_compares_json_arguments_without_rewriting_raw_fields(arguments, accepted):
    job, document, recorded, raw = _execution()
    final = recorded["text_only"]["batch_final"]
    final["calls"] = [{"name": "f", "arguments": '{"a":9007199254740993,"b":[2]}'}]
    recorded["text_only"]["assembled"] = batch_final_events(final)
    actual = deepcopy(recorded["text_only"])
    actual["batch_final"]["calls"][0]["arguments"] = arguments
    actual["assembled"] = batch_final_events(actual["batch_final"])
    document["results"]["UNIFIED.text_only.jamba"] = actual
    document["provenance"]["results_sha256"] = hashlib.sha256(
        json.dumps(document["results"], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    originals = deepcopy((document, recorded))
    if accepted:
        assert _compare(job, document, recorded, raw) == {"executed": 1}
    else:
        with pytest.raises(ValueError, match="differs"):
            _compare(job, document, recorded, raw)
    assert (document, recorded) == originals


def test_replay_preserves_whitespace_inside_json_strings():
    from_before = '{"text":"a b"}'
    from_after = '{"text":"a  b"}'
    assert replay_latest.canonical_arguments(from_before) != replay_latest.canonical_arguments(from_after)


@pytest.mark.parametrize("mutation", ["missing", "altered", "request", "chunks", "swapped_mode", "no_receipt"])
def test_replay_rejects_missing_altered_unbound_and_swapped_results(mutation):
    job, document, recorded, raw = _execution()
    if mutation == "missing":
        recorded.clear()
    elif mutation == "altered":
        recorded["text_only"]["batch_final"]["normal_text"] = "wrong"
    elif mutation == "request":
        recorded["text_only"]["capture_input"]["input"] = "wrong"
    elif mutation == "chunks":
        recorded["text_only"]["chunks"] = []
    elif mutation == "swapped_mode":
        document["provenance"]["implementation"] = "dynamo_v1_jail"
    else:
        del document["provenance"]["results_sha256"]
    with pytest.raises(ValueError):
        _compare(job, document, recorded, raw)


@pytest.mark.parametrize("field,code", [("error", "parser_exception"), ("unavailable", "unsupported_request_mode")])
def test_replay_compares_parser_errors_and_verified_unsupported_outcomes(field, code):
    job, document, recorded, raw = _execution()
    record = {field: "measured reason", "capture_observation": {field: {"code": code, "detail": "measured reason"}},
              "capture_input": job["cases"][0]["request"]}
    document["results"]["UNIFIED.text_only.jamba"] = record
    document["provenance"]["results_sha256"] = hashlib.sha256(
        json.dumps(document["results"], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    recorded["text_only"] = deepcopy(record)
    assert sum(_compare(job, document, recorded, raw).values()) == 1
    recorded["text_only"][field] = "different reason"
    with pytest.raises(ValueError, match="differs"):
        _compare(job, document, recorded, raw)


@pytest.mark.parametrize("mutation", ["payload", "expected_mode"])
def test_latest_reader_rejects_tampered_expected_identity(tmp_path, mutation):
    root = Path(__file__).resolve().parents[3]
    relative = Path(capture_sources.STORE) / "families/jamba"
    directory = tmp_path / relative
    directory.mkdir(parents=True)
    shutil.copyfile(root / relative / "inputs_and_golden.yaml", directory / "inputs_and_golden.yaml")
    authored = yaml.safe_load((directory / "inputs_and_golden.yaml").read_bytes())
    snapshot = normalized_yaml.expand_snapshot(
        yaml.safe_load((root / relative / "dynamo_v1.yaml").read_bytes()), authored,
    )
    selection = snapshot.pop("capture_selection")
    shutil.copyfile(root / relative / "dynamo_v1.yaml", directory / "dynamo_v1.yaml")
    version, records = replay_latest.latest_records(tmp_path, "jamba", "dynamo_v1")
    assert version == "11.0.13" and records
    if mutation == "expected_mode":
        record = next(item["value"]["document"]["record_metadata"] for item in snapshot["records"].values()
                      if "verified_recapture" in item["value"]["document"]["record_metadata"])
        record["verified_recapture"]["provenance"]["implementation"] = "dynamo_v1_jail"
    body = unified_history.dump_yaml(snapshot).encode()
    selection["payload_sha256"] = "0" * 64 if mutation == "payload" else hashlib.sha256(body).hexdigest()
    (directory / "dynamo_v1.yaml").write_bytes(
        b"capture_selection: " + json.dumps(selection).encode() + b"\n" + body,
    )
    with pytest.raises(ValueError):
        replay_latest.latest_records(tmp_path, "jamba", "dynamo_v1")


def test_latest_v1_replay_matches_committed_yaml(tmp_path):
    root = Path(__file__).resolve().parents[3]
    result = replay_latest.replay(root, tmp_path / "latest-v1-replay")
    assert len(result) == 62
    assert all(sum(counts.values()) > 0 for counts in result.values())
