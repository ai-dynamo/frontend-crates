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
import explode_unified_fixtures as explode
import gen_unified_golden as generator
import generate_conformance_table as table
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
    monkeypatch.setattr(table, "_unified_dynamo_label", lambda _captures: "missing-source")
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
