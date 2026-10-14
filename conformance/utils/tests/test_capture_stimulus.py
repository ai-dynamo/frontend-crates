# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import capture_stimulus
from unified_tools import unified_tools


def _input(text):
    return {"input": text, "init": {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None},
            "finish_reason": "stop", "tools": [], "chunks": [{"delta_text": text}, {"delta_text": "‹finish›"}]}


def _write(base, directory, record, key="UNIFIED.7-2"):
    path = base / directory / f"deepseek_v41/{key}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"family": "deepseek_v41", "cases": {key: record}}))
    return path


@pytest.mark.parametrize("engine", ["dynamo_v2-0.5.3", "vllm_python-0.25.1", "vllm_rust-0.25.1", "sglang_python-0.5.16"])
def test_old_fx_capture_is_not_scored_against_new_run_cmd_input(tmp_path, monkeypatch, engine):
    original = _input("old captured request")
    current = _input("new authored request") | {"scenario": "arg_marker_in_string"}
    event = {"kind": "tool_call", "name": "f", "arguments": {"x": "original argument"}}
    capture = {"assembled": [event], "chunks": [{"expected": [event]}], "capture_input": capture_stimulus.capture_input(original)}
    path = _write(tmp_path, engine, capture)
    before = path.read_bytes()
    reason = capture_stimulus.comparison_failure(capture, current, before, path.name, {})
    assert "stimulus mismatch" in reason
    assert "input" in reason and "chunks" in reason
    assert path.read_bytes() == before
    assert capture_stimulus.comparison_failure(capture, original, before, path.name, {}) is None


@pytest.mark.parametrize("dimension", ["input", "init", "chunks", "finish_reason", "tools"])
def test_each_stimulus_dimension_is_bound(dimension):
    original = _input("same")
    current = copy.deepcopy(original)
    replacements = {"input": "different", "init": {"starting_state": "Reasoning"},
                    "chunks": [{"delta_text": "sa"}, {"delta_text": "me"}], "finish_reason": "length",
                    "tools": [{"name": "f", "parameters": {"type": "object"}}]}
    current[dimension] = replacements[dimension]
    reason = capture_stimulus.comparison_failure({"capture_input": original}, current, b"", "case.yaml", {})
    assert f"({dimension})" in reason


def test_missing_stimulus_is_not_a_parser_failure():
    reason = capture_stimulus.comparison_failure({"assembled": []}, _input("input"), b"", "case.yaml", {})
    assert "stimulus unavailable" in reason and "cannot be compared" in reason


def test_old_binding_without_tools_is_unverified_not_retroactively_current():
    original = _input("same")
    del original["tools"]
    reason = capture_stimulus.comparison_failure({"capture_input": original}, _input("same"), b"", "case", {})
    assert "original tool schema was not retained" in reason


def test_sidecar_binds_exact_capture_bytes_without_rewriting_them(tmp_path):
    original = _input("original")
    raw = b"immutable capture bytes"
    bindings = {"family/case.yaml": {"capture_sha256": hashlib.sha256(raw).hexdigest(), "capture_input": original}}
    (tmp_path / "capture-inputs.json").write_text(json.dumps({"schema_version": 1, "records": bindings}))
    loaded = capture_stimulus.read_bindings(tmp_path)
    assert capture_stimulus.comparison_failure({}, original, raw, "family/case.yaml", loaded) is None
    with pytest.raises(ValueError, match="capture bytes"):
        capture_stimulus.comparison_failure({}, original, raw + b"changed", "family/case.yaml", loaded)


def test_description_changes_do_not_invalidate_identical_stimulus():
    original = _input("same")
    current = original | {"description": "reworded", "policy": ["explanation"]}
    assert capture_stimulus.comparison_failure({"capture_input": original}, current, b"", "case.yaml", {}) is None


@pytest.mark.parametrize("dimension", ["input", "init", "chunks", "finish_reason", "tools"])
def test_current_capture_guard_rejects_each_stale_dimension(tmp_path, dimension):
    current = _input("same") | {"tools": unified_tools()}
    _write(tmp_path, "inputs", current)
    record = {"capture_input": capture_stimulus.capture_input(current), "assembled": []}
    _write(tmp_path, "capture", record)
    assert capture_stimulus.validate_current_capture(tmp_path / "capture", [tmp_path / "inputs"]) == 1
    replacements = {"input": "different", "init": {}, "chunks": [], "finish_reason": "length", "tools": []}
    record["capture_input"][dimension] = replacements[dimension]
    _write(tmp_path, "capture", record)
    with pytest.raises(ValueError, match=f"stimulus mismatch.*{dimension}"):
        capture_stimulus.validate_current_capture(tmp_path / "capture", [tmp_path / "inputs"])


@pytest.mark.parametrize("schema_type", ["string", ["string", "null"]])
def test_current_capture_guard_accepts_the_executed_per_case_schema(tmp_path, schema_type):
    current = _input("null") | {"tools": [{"name": "get_weather", "parameters": {
        "type": "object", "properties": {"city": {"type": schema_type}},
    }}]}
    _write(tmp_path, "inputs", current)
    _write(tmp_path, "capture", {"capture_input": capture_stimulus.capture_input(current), "assembled": []})
    assert capture_stimulus.validate_current_capture(tmp_path / "capture", [tmp_path / "inputs"]) == 1


def test_current_capture_guard_rejects_missing_executable_schema(tmp_path):
    current = _input("same")
    del current["tools"]
    _write(tmp_path, "inputs", current)
    _write(tmp_path, "capture", {"capture_input": capture_stimulus.capture_input(current), "assembled": []})
    with pytest.raises(ValueError, match="executable tool schema"):
        capture_stimulus.validate_current_capture(tmp_path / "capture", [tmp_path / "inputs"])


def test_current_version_validates_exact_version_records(tmp_path):
    current = _input("same") | {"tools": unified_tools()}
    record = {"capture_input": capture_stimulus.capture_input(current),
              "assembled": [{"kind": "text", "text": "same"}], "chunks": []}
    base = "dynamo_v2-0.6.0"
    for key in ("UNIFIED.31-1", "retained", "added"):
        _write(tmp_path, "inputs", current, key)
        _write(tmp_path, base, record, key)
    assert capture_stimulus.validate_current_capture(tmp_path / base, [tmp_path / "inputs"]) == 3
    docs = capture_stimulus.validated_current_capture_docs(tmp_path / base, [tmp_path / "inputs"])
    assert len(docs) == 1
    assert set(docs[0]["cases"]) == {"UNIFIED.31-1", "retained", "added"}
    assert docs[0]["cases"]["UNIFIED.31-1"] == record
    assert docs[0]["cases"]["added"] == record
    result = subprocess.run(
        [sys.executable, str(SRC / "capture_stimulus.py"), "--validate-current", str(tmp_path / base),
         "--inputs", str(tmp_path / "inputs"), "--format", "json"],
        check=True, capture_output=True, text=True,
    )
    assert json.loads(result.stdout) == docs


def test_dynamo_v2_current_capture_skips_only_v1_split_scenarios(tmp_path):
    split_id = "UNIFIED.v1_split-parallel_tool_calls"
    split = _input("split") | {"scenario": "v1_split_parallel_tool_calls"}
    current = _input("current")
    _write(tmp_path, "inputs", split, split_id)
    _write(tmp_path, "inputs", current, "UNIFIED.7-2")
    current_record = {"capture_input": capture_stimulus.capture_input(current), "assembled": []}
    _write(tmp_path, "dynamo_v2-0.7.22", current_record, "UNIFIED.7-2")

    selected = tmp_path / "dynamo_v2-0.7.22"
    docs = capture_stimulus.validated_current_capture_docs(selected, [tmp_path / "inputs"])
    assert capture_stimulus.validate_current_capture(selected, [tmp_path / "inputs"]) == 1
    assert [case for doc in docs for case in doc["cases"]] == ["UNIFIED.7-2"]

    split_record = {"capture_input": capture_stimulus.capture_input(split), "assembled": []}
    _write(tmp_path, "dynamo_v2-0.7.22", split_record, split_id)
    with pytest.raises(ValueError, match="current capture/input sets differ"):
        capture_stimulus.validate_current_capture(selected, [tmp_path / "inputs"])

    with pytest.raises(ValueError, match="current capture/input sets differ"):
        capture_stimulus.validate_current_capture(tmp_path / "dynamo_v1-10.0.1", [tmp_path / "inputs"])


@pytest.mark.parametrize("invalid", [{"error": "failed"}, {"unavailable": "missing"},
                                     {"capture_input": _input("other")}])
def test_current_release_rejects_surviving_invalid_records(tmp_path, invalid):
    current = _input("same") | {"tools": unified_tools()}
    record = {"capture_input": capture_stimulus.capture_input(current), "assembled": []} | invalid
    base = "dynamo_v2-0.6.0"
    _write(tmp_path, "inputs", current)
    _write(tmp_path, base, record)
    with pytest.raises(ValueError, match="did not succeed|stimulus mismatch"):
        capture_stimulus.validate_current_capture(tmp_path / base, [tmp_path / "inputs"])


@pytest.mark.parametrize("selected_patch", [False, True])
@pytest.mark.parametrize("version", ["0.6.0", "0.7.0-rc.1"])
def test_current_capture_does_not_fold_removed_source_snapshot_patches(tmp_path, selected_patch, version):
    current = _input("same") | {"tools": unified_tools()}
    record = {"capture_input": capture_stimulus.capture_input(current), "assembled": []}
    base = f"dynamo_v2-{version}+source." + "a" * 64
    _write(tmp_path, "inputs", current)
    _write(tmp_path, base, {"error": "old source snapshot"}, "obsolete")
    _write(tmp_path, base + ".patch2", {"error": "old source snapshot"}, "obsolete")
    _write(tmp_path, base + ".patch10", record)
    for suffix, key in ((".patch2", "obsolete"), (".patch10", "UNIFIED.7-2")):
        (tmp_path / (base + suffix) / "capture-snapshot.json").write_text(json.dumps({
            "schema_version": 1, "records": [f"deepseek_v41/{key}.yaml"],
        }))
    selected = tmp_path / (base + ".patch10" if selected_patch else base)
    if not selected_patch:
        with pytest.raises(ValueError, match="current capture/input sets differ"):
            capture_stimulus.validate_current_capture(selected, [tmp_path / "inputs"])
        return
    assert capture_stimulus.validate_current_capture(selected, [tmp_path / "inputs"]) == 1
    assert capture_stimulus.validated_current_capture_docs(selected, [tmp_path / "inputs"]) == [
        {"family": "deepseek_v41", "cases": {"UNIFIED.7-2": record}},
    ]


def test_recorded_error_for_resolved_divergence_is_rejected(tmp_path):
    current = _input("same") | {"scenario": "malformed_json_then_two_valid_calls"}
    error = "native push: EOF while parsing a string at line 1 column 16"
    record = {"capture_input": capture_stimulus.capture_input(current), "error": error}
    _write(tmp_path, "inputs", current, "UNIFIED.5-4")
    _write(tmp_path, "capture", record, "UNIFIED.5-4")
    with pytest.raises(ValueError, match="unexpected error"):
        capture_stimulus.validate_current_capture(tmp_path / "capture", [tmp_path / "inputs"])
