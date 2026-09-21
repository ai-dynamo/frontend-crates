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
import generate_conformance_table as table
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
    old_text = '<｜DSML｜ calls><｜DSML｜ invoke name="f"><｜DSML｜ parameter name="x" string="true"> <think>quoted</think> <｜DSML｜ calls> </｜DSML｜ calls> </｜DSML｜ invoke> &amp; "x"\\\n </｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>'
    new_text = '<｜DSML｜ calls><｜DSML｜ invoke name="run"><｜DSML｜ parameter name="cmd" string="true">git log </｜DSML｜ invoke> --oneline</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>'
    original = _input(old_text)
    current = _input(new_text) | {"scenario": "arg_marker_in_string"}
    _write(tmp_path, "inputs", current)
    event = {"kind": "tool_call", "name": "f", "arguments": {"x": "original argument"}}
    _write(tmp_path, "golden", {"assembled": [event]})
    capture = {"assembled": [event], "chunks": [{"expected": [event]}], "capture_input": capture_stimulus.capture_input(original)}
    path = _write(tmp_path, engine, capture)
    before = path.read_bytes()
    monkeypatch.setattr(table, "_unified_dynamo_label", lambda captures: "0.5.3")
    cases, _caps, _versions = table._load_unified_fixtures(tmp_path)
    if engine.startswith("dynamo_v2"):
        record = cases[0]["dynamo_by_ver"]["0.5.3"]
    else:
        impl, version = engine.split("-", 1)
        record = cases[0]["peer_by_ver"][impl][version]
    assert "stimulus mismatch" in record["unavailable"]
    assert "input" in record["unavailable"] and "chunks" in record["unavailable"]
    assert path.read_bytes() == before
    monkeypatch.setattr(table, "_unified_base", lambda root: tmp_path)
    monkeypatch.setattr(table.gen_unified_golden, "CLEAN", [case for case in table.gen_unified_golden.CLEAN if case[0] == "arg_marker_in_string"])
    monkeypatch.setattr(table.gen_unified_golden, "EDGE", [case for case in table.gen_unified_golden.EDGE if case[0] == "arg_marker_in_string"])
    model = table._unified_tab_model(tmp_path, {})
    row = next(row for row in model["rows"] if row["family"] == "deepseek_v41")
    cell = row["cells"]["arg_marker_in_string"]
    candidate = {"dynamo_v2": "dynamo", "vllm_python": "vllm", "vllm_rust": "vllm_rust", "sglang_python": "sglang"}[engine.split("-", 1)[0]]
    popups = {item["key"]: item["block"] for item in cell["tooltip"]["candidates"]}
    if candidate in popups:
        popup = popups[candidate]
        assert "stimulus mismatch" in popup["unavailable"]
        assert "verdict" not in popup and "events" not in popup
        assert all(chunk["expected"][candidate] == [] for chunk in cell["tooltip"]["input"]["chunks"])
    else:
        assert engine.startswith("sglang_python-")
    _write(tmp_path, "inputs", original | {"scenario": "arg_marker_in_string"})
    cases, _caps, _versions = table._load_unified_fixtures(tmp_path)
    record = (cases[0]["dynamo_by_ver"]["0.5.3"] if engine.startswith("dynamo_v2")
              else cases[0]["peer_by_ver"][impl][version])
    assert "unavailable" not in record
    assert table._unified_classify("deepseek_v41", [event], record["assembled"]) == "MATCH"


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


def test_current_capture_guard_rejects_tools_when_input_and_capture_agree_on_stale_schema(tmp_path):
    current = _input("same")
    _write(tmp_path, "inputs", current)
    _write(tmp_path, "capture", {"capture_input": capture_stimulus.capture_input(current), "assembled": []})
    with pytest.raises(ValueError, match="executable shared schema"):
        capture_stimulus.validate_current_capture(tmp_path / "capture", [tmp_path / "inputs"])


@pytest.mark.parametrize("version", ["0.6.0", "0.7.0-rc.1"])
def test_source_snapshot_selection_is_numeric_and_never_promotes_release_patch(tmp_path, version):
    source = tmp_path / (f"dynamo_v2-{version}+source." + "a" * 64)
    source.mkdir()
    for suffix in (".patch10", ".patch2", ".patch1"):
        path = source.with_name(source.name + suffix)
        path.mkdir()
        (path / "capture-snapshot.json").write_text('{"schema_version":1,"records":[]}')
    assert capture_stimulus.current_source_snapshot(source).name.endswith(".patch10")
    release = tmp_path / f"dynamo_v2-{version}"
    release.mkdir()
    release.with_name(release.name + ".patch1").mkdir()
    assert capture_stimulus.current_source_snapshot(release) == release


def test_current_release_overlays_replace_invalid_records_and_add_cases(tmp_path):
    current = _input("same") | {"tools": unified_tools()}
    record = {"capture_input": capture_stimulus.capture_input(current),
              "assembled": [{"kind": "text", "text": "same"}], "chunks": []}
    base = "dynamo_v2-0.6.0"
    for key in ("UNIFIED.31-1", "retained", "added"):
        _write(tmp_path, "inputs", current, key)
    # The renamed base record is both unbound and unsuccessful. Only its effective
    # replacement may be validated; the other base case still needs its own sidecar.
    _write(tmp_path, base, {"error": "obsolete output"}, "UNIFIED.31-1")
    retained = _write(tmp_path, base, {"assembled": record["assembled"], "chunks": []}, "retained")
    (tmp_path / base / "capture-inputs.json").write_text(json.dumps({
        "schema_version": 1, "records": {"deepseek_v41/retained.yaml": {
            "capture_sha256": hashlib.sha256(retained.read_bytes()).hexdigest(),
            "capture_input": record["capture_input"],
        }},
    }))
    _write(tmp_path, base + ".patch2", {"error": "also obsolete"}, "UNIFIED.31-1")
    _write(tmp_path, base + ".patch10", record, "UNIFIED.31-1")
    _write(tmp_path, base + ".patch10", record, "added")
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
    # A bad binding on the surviving base case must still fail.
    retained.write_bytes(retained.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="capture bytes"):
        capture_stimulus.validate_current_capture(tmp_path / base, [tmp_path / "inputs"])


@pytest.mark.parametrize("invalid", [{"error": "failed"}, {"unavailable": "missing"},
                                     {"capture_input": _input("other")}])
def test_current_release_rejects_surviving_invalid_records(tmp_path, invalid):
    current = _input("same") | {"tools": unified_tools()}
    record = {"capture_input": capture_stimulus.capture_input(current), "assembled": []}
    base = "dynamo_v2-0.6.0"
    _write(tmp_path, "inputs", current)
    _write(tmp_path, "inputs", current, "added")
    _write(tmp_path, base, record | invalid)
    _write(tmp_path, base + ".patch1", record, "added")
    with pytest.raises(ValueError, match="did not succeed|stimulus mismatch"):
        capture_stimulus.validate_current_capture(tmp_path / base, [tmp_path / "inputs"])


@pytest.mark.parametrize("selected_patch", [False, True])
@pytest.mark.parametrize("version", ["0.6.0", "0.7.0-rc.1"])
def test_current_source_uses_only_complete_snapshot_records(tmp_path, selected_patch, version):
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
    assert capture_stimulus.validate_current_capture(selected, [tmp_path / "inputs"]) == 1
    assert capture_stimulus.validated_current_capture_docs(selected, [tmp_path / "inputs"]) == [
        {"family": "deepseek_v41", "cases": {"UNIFIED.7-2": record}},
    ]
