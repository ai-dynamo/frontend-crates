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
import capture_stimulus
from unified_tools import unified_tools
from capture_stimulus import execution_outcome, validate_adapter_bundle, validate_events, validate_execution
from capture_stimulus import batch_final_events
from unified_history import load_yaml
from capture_coverage import combine_coverage
from capture_versions import UNIFIED_VERSION_RE, provenance_release_version
import unified_history


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _input(text):
    return {"input": text, "init": {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None},
            "finish_reason": "stop", "tools": [], "chunks": [{"delta_text": text}, {"delta_text": "‹finish›"}]}


def _write(base, directory, record, key="UNIFIED.7-2"):
    path = base / directory / f"deepseek_v41/{key}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"family": "deepseek_v41", "cases": {key: record}}))
    return path


def test_current_capture_family_selection_filters_inputs_and_captures(tmp_path):
    current = _input("current")
    record = {"capture_input": capture_stimulus.capture_input(current), "assembled": []}
    _write(tmp_path, "inputs", current)
    _write(tmp_path, "capture", record)
    other = tmp_path / "capture/other/UNIFIED.7-2.yaml"
    other.parent.mkdir(parents=True)
    other.write_text(yaml.safe_dump({"family": "other", "cases": {"UNIFIED.7-2": record}}))
    docs = capture_stimulus.validated_current_capture_docs(tmp_path / "capture", [tmp_path / "inputs"], {"deepseek_v41"})
    assert docs == [{"family": "deepseek_v41", "cases": {"UNIFIED.7-2": record}}]
    with pytest.raises(ValueError, match="current capture/input sets differ"):
        capture_stimulus.validated_current_capture_docs(tmp_path / "capture", [tmp_path / "inputs"])


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
    changed = reason.partition("mismatch (")[2].partition(")")[0].split(", ")
    assert dimension in changed


def test_synthetic_finish_step_does_not_change_authored_chunk_schedule():
    original = _input("same")
    current = copy.deepcopy(original)
    current["chunks"] = current["chunks"][:-1]
    assert capture_stimulus.comparison_failure(
        {"capture_input": original}, current, b"", "case.yaml", {}
    ) is None


def test_synthetic_finish_step_does_not_hide_real_chunk_boundary_changes():
    original = _input("same")
    current = copy.deepcopy(original)
    current["chunks"] = [{"delta_text": "sa"}, {"delta_text": "me"}]
    reason = capture_stimulus.comparison_failure(
        {"capture_input": original}, current, b"", "case.yaml", {}
    )
    assert "(chunks)" in reason


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


@pytest.mark.parametrize("corruption", ["input", "init", "chunks", "finish_reason", "tools", "version", "implementation", "unavailable"])
def test_current_capture_guard_uses_the_bound_recapture_without_rewriting_history(tmp_path, corruption):
    current = _input("same") | {"finish_reason": "tool_calls"}
    selected = tmp_path / "dynamo_v2-0.7.24"
    result = {"capture_input": capture_stimulus.capture_input(current), "assembled": [], "chunks": []}
    recapture = {"resolution": "fresh_execution", "result": result, "provenance": {
        "implementation": "dynamo_v2", "version": "0.7.24", "source_sha256": "a" * 64,
        "adapter_sha256": "b" * 64, "job_sha256": "c" * 64,
    }}
    record = {"capture_input": capture_stimulus.capture_input(_input("historical")),
              "unavailable": "old adapter rejected terminal tool_calls", "verified_recapture": recapture}
    _write(tmp_path, "inputs", current)
    path = _write(tmp_path, selected.name, record)
    original = path.read_bytes()
    docs = capture_stimulus.validated_current_capture_docs(selected, [tmp_path / "inputs"])
    assert docs[0]["cases"]["UNIFIED.7-2"] == capture_stimulus.select_peer_observation(
        record, current, implementation="dynamo_v2", version="0.7.24")
    assert path.read_bytes() == original
    if corruption in {"version", "implementation"}:
        recapture["provenance"][corruption] = "wrong"
        reason = f"{corruption} differs"
    elif corruption == "unavailable":
        del result["chunks"]
        result["unavailable"] = "still unsupported"
        reason = "did not succeed"
    else:
        result["capture_input"][corruption] = None
        reason = "request differs"
    _write(tmp_path, selected.name, record)
    with pytest.raises(ValueError, match=reason):
        capture_stimulus.validated_current_capture_docs(selected, [tmp_path / "inputs"])


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


@pytest.mark.parametrize("expected,actual", [(True, 1), (1, True), (0.0, -0.0), (-0.0, 0.0)])
def test_rehashed_capture_cannot_change_request_value_types(expected, actual):
    job = {"cases": [{"id": "case", "request": {"tools": [{"value": expected}]}}]}
    job_bytes = json.dumps(job).encode()
    results = {"case": {"capture_input": {"tools": [{"value": actual}]}, "chunks": []}}
    document = {"vllm_version": "1.0.0", "results": results, "provenance": {
        "implementation": "vllm_python", "version": "1.0.0",
        "source_sha256": "a" * 64, "adapter_sha256": "b" * 64,
        "job_sha256": hashlib.sha256(job_bytes).hexdigest(),
        "results_sha256": hashlib.sha256(json.dumps(results, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
    }}
    with pytest.raises(ValueError, match="request mismatch"):
        validate_execution(document, job, job_bytes, "vllm_python", "1.0.0")


@pytest.mark.parametrize("corruption", [None, "empty", "traversal", "symlink", "bytes", "support"])
def test_adapter_bundle_validates_ordered_bytes_and_support(tmp_path, corruption):
    (tmp_path / "probe").write_bytes(b"probe")
    (tmp_path / "helper").write_bytes(b"helper")
    manifest = {"files": [{"path": "probe", "sha256": hashlib.sha256(b"probe").hexdigest()}],
                "support_files": [{"path": "helper", "sha256": hashlib.sha256(b"helper").hexdigest()}]}
    if corruption == "empty":
        manifest["files"] = []
    elif corruption == "traversal":
        manifest["files"][0]["path"] = "../probe"
    elif corruption == "symlink":
        (tmp_path / "probe").unlink()
        (tmp_path / "probe").symlink_to(tmp_path / "helper")
    elif corruption == "bytes":
        (tmp_path / "probe").write_bytes(b"altered")
    elif corruption == "support":
        (tmp_path / "helper").write_bytes(b"altered")
    if corruption:
        with pytest.raises(ValueError, match="adapter"):
            validate_adapter_bundle(tmp_path, manifest)
    else:
        assert validate_adapter_bundle(tmp_path, manifest) == hashlib.sha256(b"probe").hexdigest()


def test_committed_muse_spark_stream_phase_events_are_executed():
    capture = Path(__file__).resolve().parents[2] / "fixtures-unified-v2/families/muse_spark/dynamo_v2.yaml"
    document = load_yaml(capture)
    phase_records = [row["value"]["observation"]["value"] for row in document["records"].values()
                     if "value" in row["value"]["observation"]
                     and any(event["kind"] == "message_start" for event in row["value"]["observation"]["value"].get("assembled", []))]
    assert phase_records
    assert {event["phase"] for record in phase_records for event in record["assembled"] if event["kind"] == "message_start"} == {"commentary", "final_answer"}
    for record in phase_records:
        assert execution_outcome(record, "stream") == ("executed", None)


@pytest.mark.parametrize("phase", ["commentary", "final_answer"])
@pytest.mark.parametrize("chunk_shape", ["list", "expected"])
@pytest.mark.parametrize("failed", [False, True])
def test_phase_events_validate_in_success_and_partial_streams(phase, chunk_shape, failed):
    events = [{"kind": "message_start", "phase": phase}, {"kind": "text", "text": "answer"}, {"kind": "message_end"}]
    chunks = [events if chunk_shape == "list" else {"expected": events}]
    record = {"assembled": events, "chunks": chunks}
    if failed:
        record.update(error="interrupted", partial_chunks=chunks)
    assert execution_outcome(record, "stream") == (("parser_error", None) if failed else ("executed", None))


@pytest.mark.parametrize("event", [
    {"kind": "message_start"},
    *({"kind": "message_start", "phase": phase} for phase in (None, True, 1, [], {}, "analysis", "", "Commentary")),
])
def test_message_start_rejects_missing_or_invalid_phase(event):
    with pytest.raises(ValueError, match="valid phase"):
        validate_events([event])


def test_phase_support_preserves_batch_final_projection_and_unknown_kind_rejection():
    final = {"reasoning_text": "reason", "normal_text": "answer", "calls": []}
    events = batch_final_events(final)
    assert execution_outcome({"batch_final": final, "assembled": events}, "batch") == ("executed", None)
    with pytest.raises(ValueError, match="differs from final fields"):
        execution_outcome({"batch_final": final, "assembled": [{"kind": "message_start", "phase": "final_answer"}, *events]}, "batch")
    with pytest.raises(ValueError, match="invalid kind"):
        validate_events([{"kind": "unknown"}])


@pytest.mark.parametrize("corruption", [None, "duplicate", "missing", "version", "case", "status"])
def test_global_accounting_rejects_incomplete_or_misidentified_members(tmp_path, corruption):
    inventory = tmp_path / "inventory.yaml"
    inventory.write_text('targets:\n- implementation: dynamo_v1\n  version: "1.0.0"\n')
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    (jobs / "family.json").write_text(json.dumps({"family": "family", "cases": [
        {"id": "UNIFIED.case.family", "case_id": "case", "scenario": "case"}]}))
    rows = [{"implementation": implementation, "version": "1.0.0", "family": "family",
             "scenario": "case", "case_id": "case", "status": status}
            for implementation, status in [("dynamo_v1", "executed"), ("dynamo_v1_jail", "source_build_blocker")]]
    if corruption == "duplicate":
        rows.append(rows[0])
    elif corruption == "missing":
        rows.pop()
    elif corruption == "version":
        rows[0]["version"] = "2.0.0"
    elif corruption == "case":
        rows[0]["case_id"] = "altered"
    elif corruption == "status":
        rows[0]["status"] = "captured"
    source = tmp_path / "coverage.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "combined.jsonl"
    if corruption:
        output.write_text("original generation")
        with pytest.raises(ValueError):
            combine_coverage(inventory, jobs, [source], output)
        assert output.read_text() == "original generation"
    else:
        summary = combine_coverage(inventory, jobs, [source], output)
        assert summary["combinations"] == 2
        assert summary["counts"] == {"executed": 1, "source_build_blocker": 1}
        assert summary["source_build_blockers"][0]["implementation"] == "dynamo_v1_jail"
        assert all(json.loads(line)["case_id"] == "UNIFIED.case.family" for line in output.read_text().splitlines())


def test_frozen_source_identity_compares_as_release_without_becoming_capture_version():
    qualified = '0.7.25+source.' + 'a' * 64
    assert provenance_release_version(qualified) == '0.7.25'
    assert not UNIFIED_VERSION_RE.fullmatch(qualified)


@pytest.mark.parametrize('version', ['0.7.25+source.a', '0.7.25+other.abc', '0.7.25+source./bad'])
def test_malformed_frozen_source_identity_is_rejected(version):
    with pytest.raises(ValueError):
        provenance_release_version(version)


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_mixed_legacy_and_adapter_batch_observations_preserve_their_contracts(tmp_path):
    family = unified_history.Family("sample", {"cases": {}}, tmp_path / "inputs_and_golden.yaml")
    final = {"reasoning_text": "", "normal_text": "hello", "calls": []}
    records = {
        "legacy": {"case_key": "UNIFIED.legacy", "stimulus": {"ref": "current"}, "observation": {"value": {"chunks": [[{"kind": "text", "text": "legacy"}]]}}},
        "batch": {"case_key": "UNIFIED.batch", "stimulus": {"ref": "current"}, "observation": {"value": {"assembled": capture_stimulus.batch_final_events(final)}}},
    }
    checkpoint = "dynamo_v1-10.0.1"
    capture = {"runtime_version": "10.0.1", "provenance": {"status": "captured", "origin": {"crate_version": "10.0.1", "source_sha256": "a" * 64, "adapter_sha256": "b" * 64, "job_sha256": "c" * 64}},
               "document": {}, "changes": records, "metadata_changes": {"batch": {"batch_final": final}},
               "document_overrides": {"legacy": {"capture_provenance": {"captured_with": {"dynamo_v1": "10.0.1"}, "capture_origin": None, "inherited_from": None}}}}
    history = unified_history.History(family, "dynamo_v1", {checkpoint: capture}, tmp_path, {checkpoint: tmp_path / (checkpoint + ".yaml")})
    resolved = history.resolve(checkpoint)
    assert resolved["legacy"]["observation"] == records["legacy"]["observation"]
    assert "capture_origin" not in resolved["legacy"]["document"]
    assert capture_stimulus.unified_record_contract("dynamo_v1", resolved["legacy"]["document"]) == "legacy-stream"
    assert capture_stimulus.unified_record_contract("dynamo_v1", resolved["batch"]["document"]) == "batch"
    frozen = copy.deepcopy(capture)
    for change in ("missing_final", "wrong_projection", "chunks"):
        history.captures[checkpoint] = copy.deepcopy(frozen)
        bad = history.captures[checkpoint]
        if change == "missing_final":
            bad["metadata_changes"] = {}
        elif change == "wrong_projection":
            bad["changes"]["batch"]["observation"]["value"]["assembled"] = []
        else:
            bad["changes"]["batch"]["observation"]["value"]["chunks"] = []
        with pytest.raises(ValueError, match="batch"):
            history.resolve(checkpoint)


def test_adapter_batch_contract_rejects_a_partial_producer_identity():
    with pytest.raises(ValueError, match="adapter and job"):
        capture_stimulus.unified_record_contract("dynamo_v1", {"capture_origin": {"adapter_sha256": "a" * 64}})
