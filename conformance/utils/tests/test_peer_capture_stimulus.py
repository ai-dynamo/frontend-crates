# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import io
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import yaml

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import capture_stimulus
import capture_peer_versions
import explode_unified_fixtures as explode
import gen_unified_golden as generator
import generate_conformance_table as table
import package_fixtures
from unified_tools import unified_tools


@pytest.fixture(params=["vllm_python", "vllm_rust", "sglang_python"])
def producer(request, monkeypatch, tmp_path):
    engine = request.param
    calls = []

    class Box:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class Parser:
        def __init__(self, *args, **kwargs):
            pass

        def parse(self, text, *args):
            calls.append(("batch", text))
            return None, text, []

        def parse_delta(self, text, *args, finished=False):
            calls.append(("delta", text, finished))
            return SimpleNamespace(content=text, reasoning_content=None, tool_calls=[])

        def parse_stream_chunk(self, text):
            calls.append(("stream", text))
            return "", text

    class ToolParser(Parser):
        def parse_stream_chunk(self, text):
            return text, []

    class Manager:
        def get_parser(self, **kwargs):
            return Parser

    modules = {
        "vllm": {"__version__": "0.25.1"},
        "vllm.entrypoints.openai.chat_completion.protocol": {"ChatCompletionRequest": Box},
        "vllm.parser.parser_manager": {"ParserManager": Manager},
        "sglang": {"__version__": "0.5.16"},
        "sglang.srt.entrypoints.openai.protocol": {"Function": Box, "Tool": Box},
        "sglang.srt.function_call.function_call_parser": {"FunctionCallParser": ToolParser},
        "sglang.srt.parser.reasoning_parser": {"ReasoningParser": Parser},
    }
    for name, attrs in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    name = {"vllm_python": "capture_vllm_unified", "vllm_rust": "capture_vllm_rust_unified", "sglang_python": "capture_sglang_unified"}[engine]
    monkeypatch.delitem(sys.modules, name, raising=False)
    module = importlib.import_module(name)

    def rust_run(_source, job_json):
        results = {}
        for case in json.loads(job_json)["cases"]:
            calls.append(("rust_job", case))
            results[case["id"]] = {"assembled": [{"kind": "text", "text": case["input"]}],
                                   "chunks": [[{"kind": "text", "text": chunk}] for chunk in case["chunks"]] + ([[]] if case.get("terminal_step") else [])}
        return json.dumps({"vllm_rust_version": "0.25.1", "results": results})

    if engine == "vllm_rust":
        monkeypatch.setattr(module, "build_and_run", rust_run)
        monkeypatch.setattr(module, "_vllm_rust_version", lambda *_args: "0.25.1")

    def run(case):
        if engine == "vllm_rust":
            return module.capture_job(tmp_path, {"cases": [case]})["results"][case["id"]]
        output = io.StringIO()
        monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({"cases": [case]})))
        monkeypatch.setattr(sys, "stdout", output)
        module.main()
        return yaml.safe_load(output.getvalue())["results"][case["id"]]

    yield engine, run, calls
    sys.modules.pop(name, None)


def _case(text="hi", **extra):
    return {"id": "UNIFIED.text_only.gemma4", "family": "gemma4", "input": text, "chunks": [text], **extra}


def test_fresh_peer_capture_binds_actual_input_and_remains_comparable(producer, tmp_path, monkeypatch):
    engine, run, calls = producer
    case = _case()
    result = run(case)
    current = {"scenario": "text_only", "input": "hi", "tools": unified_tools(), "chunks": [{"delta_text": "hi"}]}
    record = explode._peer_cell(result)
    assert capture_stimulus.comparison_failure(record, current, b"", "case", {}) is None
    assert calls
    events = [{"kind": "text", "text": "hi"}]
    version = "0.5.16" if engine == "sglang_python" else "0.25.1"
    for directory, value in [("inputs", current), ("golden", {"assembled": events}), (f"{engine}-{version}", record)]:
        path = tmp_path / directory / "gemma4/UNIFIED.3-1.yaml"
        path.parent.mkdir(parents=True)
        path.write_text(yaml.safe_dump({"family": "gemma4", "cases": {"UNIFIED.3-1": value}}))
    monkeypatch.setattr(table, "_unified_base", lambda _root: tmp_path)
    monkeypatch.setattr(table, "_unified_dynamo_label", lambda: "missing-source")
    monkeypatch.setattr(generator, "CLEAN", [row for row in generator.CLEAN if row[0] == "text_only"])
    monkeypatch.setattr(generator, "EDGE", [])
    model = table._unified_tab_model(tmp_path, {})
    cell = model["rows"][0]["cells"]["text_only"]
    key = {"vllm_python": "vllm", "vllm_rust": "vllm_rust", "sglang_python": "sglang"}[engine]
    blocks = {candidate["key"]: candidate["block"] for candidate in cell["tooltip"]["candidates"]}
    if key in blocks:
        assert blocks[key]["verdict"] == "MATCH"
        assert blocks[key]["events"] == events
    else:
        assert engine == "sglang_python"
        cases, caps, _versions = table._load_unified_fixtures(tmp_path)
        assert "unavailable" not in caps[engine][cases[0]["id"]]
        assert caps[engine][cases[0]["id"]]["chunks"] == [events]


def test_stream_recapture_can_force_an_unchanged_result_into_the_patch():
    case_id = "TOOLCALLING.streamv1.7-6"
    anchor = {
        "cases": {
            case_id: {"chunks": [{"expected": {"vllm_python": [{"name": "get_weather"}]}}]}
        }
    }
    captured = {case_id: [{"deltas": [{"name": "get_weather"}]}]}

    assert capture_peer_versions._changed_stream_cases(anchor, captured, "vllm_python") == ({}, [])
    assert capture_peer_versions._changed_stream_cases(
        anchor, captured, "vllm_python", {case_id}
    )[0] == {case_id: {"chunks": [{"expected": [{"name": "get_weather"}]}]}}


def test_stream_case_replacement_requires_explicit_family(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "capture_peer_versions.py",
            "--corpus",
            "stream",
            "--replace-stream-case",
            "TOOLCALLING.streamv1.7.h",
        ],
    )

    with pytest.raises(SystemExit) as error:
        capture_peer_versions.main()

    assert error.value.code == 2
    assert "--replace-stream-case requires --family" in capsys.readouterr().err


@pytest.mark.parametrize("implementation", ["vllm_python", "vllm_rust"])
def test_requested_stream_correction_packages_only_approved_cases(tmp_path, monkeypatch, implementation):
    # A correction is a packaging transaction, which parser input/output fixtures cannot express.
    corrected = "TOOLCALLING.streamv1.7.h"
    unrelated = "TOOLCALLING.streamv1.7.a"
    root = tmp_path / "repo"
    stream_root = root / capture_peer_versions.STREAM_ROOT_REL
    relative_root = Path("toolcalling/fixtures-stream-v1")
    filename = "TOOLCALLING.streamv1.7.yaml"
    store, blobs = tmp_path / "store", tmp_path / "blobs"

    def write_document(label, cases):
        document = {"family": "glm47", "mode": "streamv1", "cases": cases}
        if label != "inputs":
            document["captured_with"] = {implementation: label.split("-", 1)[1]}
        path = stream_root / label / "glm47" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(document))

    def archive(destination, label):
        path = destination / relative_root / f"{label}.tar.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        digest, size = package_fixtures._tar_dir(stream_root / label, str(relative_root / label), path)
        return {"path": str(path.relative_to(destination)), "sha256": digest, "size": size}

    old_inputs = {cid: {"chunks": [{"delta_text": "old"}]} for cid in (corrected, unrelated)}
    write_document("inputs", old_inputs)
    inputs_shard = archive(store, "inputs")
    old_result = {"chunks": [{"expected": [{"index": 0, "name": "old"}]}]}
    new_result = {"chunks": [{"expected": [{"index": 0, "name": "new"}]}]}
    write_document(f"{implementation}-0.23.0", {corrected: old_result, unrelated: old_result})
    write_document(f"{implementation}-0.26.0", {corrected: old_result, unrelated: new_result})
    base_shard = archive(store, f"{implementation}-0.26.0")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"shards": [inputs_shard, base_shard]}))
    write_document("inputs", {**old_inputs, corrected: {"chunks": [{"delta_text": "corrected"}]}})
    captures = {
        corrected: [{"deltas": [{"index": 0, "name": "corrected"}]}],
        unrelated: [{"deltas": [{"index": 0, "name": "new"}]}],
    }
    monkeypatch.setattr(capture_peer_versions, "_STAGED_VERSION_ROOTS", {})
    monkeypatch.setattr(capture_peer_versions.cd, "_copy_worker", lambda *_args: None)
    monkeypatch.setattr(capture_peer_versions.cd, "_container_capture",
                        lambda container, short, mode, jobs, work: ("0.26.0", {job["src"]: {"cases": captures} for job in jobs}))
    monkeypatch.setattr(capture_peer_versions.cd, "_vllm_rust_capture",
                        lambda source, mode, jobs, work: ("0.26.0", {job["src"]: {"cases": captures} for job in jobs}))
    engine = SimpleNamespace(name=implementation, short="vllm", tc_map={"glm47": "parser"},
                             source_based=implementation == "vllm_rust", container=lambda args: "unused")
    args = SimpleNamespace(root=str(root), work=str(tmp_path / "work"), family="glm47",
                           vllm_rust_source=str(tmp_path / "vllm"), replace_stream_case=[corrected])
    capture_peer_versions._run_stream(engine, args)
    capture_peer_versions._publish_staged()
    label = f"{implementation}-0.26.0.patch1"
    patch = yaml.safe_load((stream_root / label / "glm47" / filename).read_text())
    assert set(patch["cases"]) == {corrected}
    shards = [archive(blobs, "inputs"), archive(blobs, label)]
    package_fixtures.sync_store(blobs, shards, False, False, fixtures_dir=store,
                               manifest_path=manifest, approved_stream_replacements={corrected})
    assert (store / relative_root / f"{label}.tar.gz").is_file()
    output = tmp_path / "resolved"
    capture_peer_versions.resolve_stream_fixtures.resolve(stream_root, output, [f"{implementation}-0.26.0"])
    resolved = yaml.safe_load((output / "glm47" / filename).read_text())["cases"]
    assert resolved[corrected]["chunks"][0]["expected"][implementation][0]["name"] == "corrected"
    assert resolved[unrelated]["chunks"][0]["expected"][implementation][0]["name"] == "new"


def test_same_version_stream_recapture_publishes_an_append_only_patch(tmp_path, monkeypatch):
    root = tmp_path / "fixtures-stream-v1"
    original = root / "vllm_python-0.26.0"
    original.mkdir(parents=True)
    monkeypatch.setattr(capture_peer_versions, "_STAGED_VERSION_ROOTS", {})

    first = Path(capture_peer_versions._version_outdir(
        str(root), "vllm_python", "0.26.0", "glm47", append_patch=True
    ))
    assert next(iter(capture_peer_versions._STAGED_VERSION_ROOTS)).endswith(
        "/vllm_python-0.26.0.patch1"
    )
    (first / "capture.yaml").write_text("patch one")
    capture_peer_versions._publish_staged()
    assert (root / "vllm_python-0.26.0.patch1" / "glm47" / "capture.yaml").read_text() == "patch one"

    second = Path(capture_peer_versions._version_outdir(
        str(root), "vllm_python", "0.26.0", "glm47", append_patch=True
    ))
    assert next(iter(capture_peer_versions._STAGED_VERSION_ROOTS)).endswith(
        "/vllm_python-0.26.0.patch2"
    )
    (second / "capture.yaml").write_text("patch two")
    capture_peer_versions._publish_staged()
    assert original.is_dir()
    assert (root / "vllm_python-0.26.0.patch2" / "glm47" / "capture.yaml").read_text() == "patch two"


def test_rust_new_version_compares_against_latest_patch_before_writing_overlay(tmp_path, monkeypatch):
    case_id = "TOOLCALLING.streamv1.7-6"
    root = tmp_path / "repo"
    stream_root = root / "conformance/toolcalling/fixtures-stream-v1"
    base_name = "TOOLCALLING.streamv1.7.yaml"
    input_file = stream_root / "inputs/glm47" / base_name
    input_file.parent.mkdir(parents=True)
    input_file.write_text(yaml.safe_dump({
        "family": "glm47",
        "mode": "streamv1",
        "cases": {case_id: {"chunks": [{"delta_text": "payload"}]}},
    }))

    def write_capture(directory, name):
        path = stream_root / directory / "glm47" / base_name
        path.parent.mkdir(parents=True)
        path.write_text(yaml.safe_dump({
            "family": "glm47",
            "mode": "streamv1",
            "captured_with": {"vllm_rust": "0.25.0"},
            "cases": {case_id: {"chunks": [{"expected": [{"index": 0, "name": name}]}]}},
        }))

    write_capture("vllm_rust-0.25.0", "old")
    write_capture("vllm_rust-0.25.0.patch1", "patched")
    input_path = str(input_file)
    captured = {input_path: {"cases": {case_id: [{"deltas": [{"index": 0, "name": "old"}]}]}}}
    monkeypatch.setattr(capture_peer_versions.cd, "_vllm_rust_capture", lambda *_args: ("0.27.0", captured))
    monkeypatch.setattr(capture_peer_versions, "_STAGED_VERSION_ROOTS", {})
    engine = SimpleNamespace(tc_map={"glm47": "parser"})
    args = SimpleNamespace(
        vllm_rust_source=str(tmp_path / "vllm"),
        root=str(root),
        work=str(tmp_path / "work"),
        family="glm47",
        replace_stream_case=[],
    )

    capture_peer_versions._run_stream_rust(engine, args)
    capture_peer_versions._publish_staged()

    release_capture = stream_root / "vllm_rust-0.27.0" / "glm47" / base_name
    assert release_capture.is_file()
    output = tmp_path / "resolved"
    capture_peer_versions.resolve_stream_fixtures.resolve(
        stream_root, output, ["vllm_rust-0.27.0"]
    )
    resolved = yaml.safe_load((output / "glm47" / base_name).read_text())
    assert resolved["cases"][case_id]["chunks"][0]["expected"]["vllm_rust"] == [
        {"index": 0, "name": "old"}
    ]


@pytest.mark.parametrize(
    ("new_deltas", "expected_deltas", "writes_overlay"),
    [
        ([], [], True),
        ([{"index": 0, "name": "old"}], [{"index": 0, "name": "old"}], False),
    ],
)
def test_rust_new_version_records_result_changes_and_skips_unchanged(
    tmp_path, monkeypatch, new_deltas, expected_deltas, writes_overlay
):
    case_id = "TOOLCALLING.streamv1.7-6"
    root = tmp_path / "repo"
    stream_root = root / "conformance/toolcalling/fixtures-stream-v1"
    base_name = "TOOLCALLING.streamv1.7.yaml"
    input_file = stream_root / "inputs/glm47" / base_name
    input_file.parent.mkdir(parents=True)
    input_file.write_text(yaml.safe_dump({
        "family": "glm47",
        "mode": "streamv1",
        "cases": {case_id: {"chunks": [{"delta_text": "payload"}]}},
    }))
    anchor_file = stream_root / "vllm_rust-0.25.0/glm47" / base_name
    anchor_file.parent.mkdir(parents=True)
    anchor_file.write_text(yaml.safe_dump({
        "family": "glm47",
        "mode": "streamv1",
        "captured_with": {"vllm_rust": "0.25.0"},
        "cases": {case_id: {"chunks": [{"expected": [{"index": 0, "name": "old"}]}]}},
    }))
    input_path = str(input_file)
    captured = {input_path: {"cases": {case_id: [{"deltas": new_deltas}]}}}
    monkeypatch.setattr(capture_peer_versions.cd, "_vllm_rust_capture", lambda *_args: ("0.27.0", captured))
    monkeypatch.setattr(capture_peer_versions, "_STAGED_VERSION_ROOTS", {})
    engine = SimpleNamespace(tc_map={"glm47": "parser"})
    args = SimpleNamespace(
        vllm_rust_source=str(tmp_path / "vllm"),
        root=str(root),
        work=str(tmp_path / "work"),
        family="glm47",
        replace_stream_case=[],
    )

    capture_peer_versions._run_stream_rust(engine, args)
    capture_peer_versions._publish_staged()

    release_capture = stream_root / "vllm_rust-0.27.0" / "glm47" / base_name
    assert release_capture.is_file() is writes_overlay
    output = tmp_path / "resolved"
    capture_peer_versions.resolve_stream_fixtures.resolve(
        stream_root, output, ["vllm_rust-0.27.0"]
    )
    resolved = yaml.safe_load((output / "glm47" / base_name).read_text())
    assert resolved["cases"][case_id]["chunks"][0]["expected"]["vllm_rust"] == expected_deltas


def test_rust_anchor_comparison_projects_only_rust_results():
    anchor = {
        "unavailable": {"vllm_python": "other parser unavailable"},
        "chunks": [{
            "expected": {
                "vllm_rust": [{"index": 0, "name": "old"}],
                "vllm_python": [{"index": 0, "name": "peer"}],
            },
            "normal_text": {"vllm_rust": "before", "vllm_python": "peer text"},
        }],
    }
    same_capture = [{"deltas": [{"index": 0, "name": "old"}], "normal_text": "before"}]
    assert capture_peer_versions._rust_anchor_case_form(anchor) == (
        capture_peer_versions._rust_captured_case_form(same_capture)
    )
    assert capture_peer_versions._rust_anchor_case_form(anchor) != (
        capture_peer_versions._rust_captured_case_form([{"deltas": []}])
    )

    exception_anchor = {
        "exception": {"vllm_rust": "rust failure", "vllm_python": "peer failure"},
    }
    assert capture_peer_versions._rust_anchor_case_form(exception_anchor) == (
        "exception", "rust failure"
    )
    assert capture_peer_versions._rust_anchor_case_form(exception_anchor) != (
        "exception", "changed rust failure"
    )

    unavailable_anchor = {"unavailable": {"vllm_rust": "rust parser unavailable"}}
    assert capture_peer_versions._rust_anchor_case_form(unavailable_anchor) == ("unavail",)


@pytest.mark.parametrize("init", [{"starting_state": "Reasoning"}, {"starting_state": "Response"},
                                 {"tool_output_mode": "GuidedJson"}, {"named_tool": "f"}])
def test_unsupported_requested_init_does_not_run_or_get_stamped_as_applied(producer, init):
    _engine, run, calls = producer
    result = run(_case(init=init))
    assert "unsupported request: init" in result["unavailable"]
    assert result["capture_input"]["init"] == capture_stimulus.capture_input({})["init"]
    assert not calls


def test_real_authored_native_case_executes_its_explicit_finish_step(producer):
    engine, run, calls = producer
    authored = generator.build_cases("gemma4")["UNIFIED.text_only.gemma4"]
    case = _case(authored["input"], init=authored["init"], finish_reason=authored["finish_reason"],
                 chunks=[authored["input"], "‹finish›"])
    result = run(case)
    if engine == "sglang_python":
        assert "finish operation" in result["unavailable"]
        assert not calls
    else:
        assert "unavailable" not in result
        assert result["capture_input"]["chunks"] == [{"delta_text": authored["input"]}, {"delta_text": "‹finish›"}]
        assert len(result["chunks"]) == 2
        if engine == "vllm_python":
            assert ("delta", authored["input"], False) in calls
            assert ("delta", "", True) in calls
        else:
            assert calls[0][1]["chunks"] == [authored["input"]]
            assert calls[0][1]["terminal_step"] is True


@pytest.mark.parametrize("returned", [{}, {"unexpected": {}}])
def test_peer_result_cardinality_is_fail_closed(returned):
    with pytest.raises(ValueError, match="executed request"):
        capture_stimulus.capture_peer_results([_case()], {"gemma4"}, lambda _cases: returned, tools=[])


def test_literal_finish_marker_is_not_an_unexecuted_terminal_operation(producer):
    engine, run, calls = producer
    result = run(_case("hi‹finish›", chunks=["hi", "‹finish›"]))
    assert "unavailable" not in result
    assert result["capture_input"]["input"] == "hi‹finish›"
    if engine == "vllm_rust":
        assert calls[0][1]["terminal_step"] is False
        assert calls[0][1]["chunks"] == ["hi", "‹finish›"]
    else:
        assert any(call[1] == "‹finish›" for call in calls)


def test_peer_rejects_unapplied_tool_schema(producer):
    _engine, run, calls = producer
    result = run(_case(tools=[]))
    assert "tools" in result["unavailable"]
    assert result["capture_input"]["tools"] == unified_tools()
    assert not calls
