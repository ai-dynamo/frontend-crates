# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The unified SGLang harness must run on BOTH SGLang shapes.

`tool_call_parser_active` tells the reasoning detector that a tool parser will consume
its normal text, which is what keeps a channel-framed family (muse) from having its
`to=user` channel unwrapped and its later tool channel dropped. But the kwarg arrived
WITH muse (PR #34262): released SGLang — including the 0.5.14 that captured the committed
shard — has no such parameter and raises `TypeError` on it.

`main()` catches per case, so passing it unconditionally does not crash: it silently turns
EVERY case of EVERY family into an empty result with an error string nobody reads. These
tests pin the signature probe that prevents that, in both directions.
"""
from __future__ import annotations

import importlib
import io
import json
import sys
from pathlib import Path
from unittest import mock

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gen_unified_golden import UNIFIED_FAMILIES

CONSTRUCTED: list[dict] = []


class _Released:
    DetectorMap = {"deepseek-v4": object, "gemma4": object, "qwen3": object, "glm45": object, "kimi_k2": object, "muse": object}
    """`ReasoningParser.__init__` as released SGLang declares it (no muse)."""

    def __init__(self, model_type=None, stream_reasoning=True, force_reasoning=None, request=None):
        CONSTRUCTED.append({"model_type": model_type, "active": None})

    def parse_stream_chunk(self, chunk):
        return "", chunk


class _WithMuse(_Released):
    """...and as PR #34262 declares it."""

    def __init__(self, model_type=None, stream_reasoning=True, force_reasoning=None,
                 request=None, tool_call_parser_active=False):
        CONSTRUCTED.append({"model_type": model_type, "active": tool_call_parser_active})


class _FunctionCallParser:
    ToolCallParserEnum = {"deepseekv4": object, "gemma4": object, "qwen3_coder": object, "glm47": object, "kimi_k2": object, "muse": object}

    def __init__(self, tools=None, tool_call_parser=None):
        pass

    def parse_stream_chunk(self, text):
        return text, []


def _stub(name, **attrs):
    mod = type(sys)(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


def _load(reasoning_parser):
    """Import the harness against a stubbed SGLang carrying `reasoning_parser`."""
    # `Tool`/`Function` build the request-specific schemas passed to the detector.
    box = type("box", (), {"__init__": lambda self, **kw: self.__dict__.update(kw)})
    mods = {
        "sglang": _stub("sglang", __version__="stub"),
        "sglang.srt": _stub("sglang.srt"),
        "sglang.srt.entrypoints": _stub("sglang.srt.entrypoints"),
        "sglang.srt.entrypoints.openai": _stub("sglang.srt.entrypoints.openai"),
        "sglang.srt.entrypoints.openai.protocol": _stub(
            "sglang.srt.entrypoints.openai.protocol", Function=box, Tool=box),
        "sglang.srt.function_call": _stub("sglang.srt.function_call"),
        "sglang.srt.function_call.function_call_parser": _stub(
            "sglang.srt.function_call.function_call_parser",
            FunctionCallParser=_FunctionCallParser),
        "sglang.srt.parser": _stub("sglang.srt.parser"),
        "sglang.srt.parser.reasoning_parser": _stub(
            "sglang.srt.parser.reasoning_parser", ReasoningParser=reasoning_parser),
    }
    CONSTRUCTED.clear()
    with mock.patch.dict(sys.modules, mods):
        sys.modules.pop("capture_sglang_unified", None)
        module = importlib.import_module("capture_sglang_unified")
    sys.modules.pop("capture_sglang_unified", None)
    return module


@pytest.mark.parametrize(
    "reasoning_parser, expected",
    [(_Released, {}), (_WithMuse, {"tool_call_parser_active": True})],
    ids=["released-sglang", "muse-sglang"],
)
def test_the_flag_is_passed_only_when_sglang_accepts_it(reasoning_parser, expected):
    module = _load(reasoning_parser)
    assert module.sglang_reasoning_kwargs(reasoning_parser, streaming=True) == {"stream_reasoning": True, "force_reasoning": False, **expected}


@pytest.mark.parametrize(
    "reasoning_parser, expected",
    [(_Released, None), (_WithMuse, True)],
    ids=["released-sglang", "muse-sglang"],
)
def test_a_capture_runs_on_both_sglang_shapes(reasoning_parser, expected):
    # The regression this guards is silent, so assert the capture really produced its
    # deltas rather than only that the constructor did not raise.
    module = _load(reasoning_parser)
    assert module._stream_chunks("muse_glimmer", ["hi"]) == [[{"kind": "text", "text": "hi"}]]
    assert [c["active"] for c in CONSTRUCTED] == [expected]


def test_glm_uses_the_registered_reasoning_detector_name():
    assert _load(_Released).FAMILY_PARSERS["glm47"] == ("glm45", "glm47")


def test_all_authored_families_have_explicit_detector_pairs():
    module = _load(_Released)
    # GPT-OSS uses Harmony framing; this split-detector adapter has no registered pair.
    assert "gpt_oss" not in module.FAMILY_PARSERS
    assert set(module.FAMILY_PARSERS) | {"gpt_oss"} == UNIFIED_FAMILIES
    assert module.FAMILY_PARSERS["nemotron_v3"] == ("nemotron_3", "qwen3_coder")
    with mock.patch.object(module, "version", return_value="0.5.16"):
        assert module._registration_unavailable({"family": "deepseek_v4"}) is None
        code, detail = module._registration_unavailable({"family": "deepseek_v41"})
    assert code == "peer_detector_unavailable"
    assert "deepseek-v41" in detail and "deepseekv41" in detail


def test_stream_end_flushes_reasoning_before_tool_detector():
    class FinishingReasoning(_Released):
        def parse_stream_end(self):
            return "reason", "tail"
    module = _load(FinishingReasoning)
    assert module._stream_chunks("qwen3", ["head"]) == [[
        {"kind": "reasoning", "text": "reason"},
        {"kind": "text", "text": "headtail"},
    ]]


def test_serving_argument_suffix_is_preserved_as_raw_fragment():
    module = _load(_Released)
    detector = type("Detector", (), {
        "prev_tool_call_arr": [{"arguments": '{"a": 1}'}],
        "streamed_args_for_tool": ['{"a": '],
    })()
    fp = type("Parser", (), {"detector": detector})()
    with mock.patch.object(module, "version", return_value="0.5.21"):
        assert module._remaining_tool_args(fp) == [{"kind": "tool_call", "name": None, "arguments": "1}"}]


def test_errors_are_concrete_and_other_cases_continue():
    class Broken(_Released):
        def parse_stream_chunk(self, chunk):
            if chunk == "bad":
                raise ValueError("concrete failure")
            return "", chunk
    module = _load(Broken)
    with mock.patch.object(module, "version", return_value="0.5.21"):
        results = module._capture_cases([
            {"family": "qwen3", "id": "one", "chunks": ["bad"]},
            {"family": "qwen3", "id": "two", "chunks": ["good"]},
        ])
    assert results["one"]["error"] == "ValueError: concrete failure"
    assert results["two"]["chunks"] == [[{"kind": "text", "text": "good"}]]


def test_main_uses_each_authored_tool_schema():
    module = _load(_Released)
    cases = [
        {"family": "qwen3", "id": "one", "tools": [{"name": "first"}]},
        {"family": "qwen3", "id": "two", "tools": [{"name": "second"}]},
    ]
    calls = []
    def capture(cases, families, callback, **kwargs):
        calls.append(kwargs["tools"])
        return {cases[0]["id"]: {"chunks": []}}
    with mock.patch.object(module, "capture_peer_results", side_effect=capture), mock.patch.object(module, "version", return_value="0.5.14"), mock.patch.object(sys, "stdin", io.StringIO(json.dumps({"cases": cases}))), mock.patch.object(sys, "stdout", io.StringIO()):
        module.main()
    assert calls == [[{"name": "first"}], [{"name": "second"}]]


def test_main_rejects_repeating_ids_across_families_before_capture():
    module = _load(_Released)
    job = {"cases": [{"id": "repeat", "family": "qwen3"}, {"id": "repeat", "family": "gemma4"}]}
    with mock.patch.object(sys, "stdin", io.StringIO(json.dumps(job))), pytest.raises(ValueError, match="exactly one family"):
        module.main()


def test_native_tool_end_flush_is_a_raw_stream_delta():
    class Item:
        def model_dump(self):
            return {"name": "run", "parameters": '{"value":1}'}
    class FinishingTool(_FunctionCallParser):
        def parse_stream_end(self):
            return "tail", [Item()]
    module = _load(_Released)
    module.FunctionCallParser = FinishingTool
    assert module._stream_chunks("qwen3", ["head"]) == [[
        {"kind": "text", "text": "headtail"},
        {"kind": "tool_call", "name": "run", "arguments": '{"value":1}'},
    ]]


def test_abort_termination_is_explicitly_unsupported():
    module = _load(_Released)
    code, detail = module._registration_unavailable({"family": "qwen3", "finish_reason": "abort"})
    assert code == "peer_termination_unsupported"
    assert "stop/length" in detail


def test_terminal_boundary_is_never_fed_as_text_to_old_detector():
    class Tracking(_Released):
        def parse_stream_chunk(self, chunk):
            assert chunk != ""
            return "", chunk
    module = _load(Tracking)
    assert module._stream_chunks("qwen3", ["head"], terminal_step=True) == [[{"kind": "text", "text": "head"}], []]


@pytest.mark.parametrize("release", ["0.5.14", "0.5.16", "0.5.21"])
@pytest.mark.parametrize("expected", ["abcdef", {"a": 1, "b": 2}])
@pytest.mark.parametrize("actual", ["abc", "cde", "absent", ""])
def test_remaining_arguments_match_each_release_serving_semantics(release, expected, actual):
    module = _load(_Released)
    detector = type("Detector", (), {"prev_tool_call_arr": [{"arguments": expected}], "streamed_args_for_tool": [actual]})()
    fp = type("Parser", (), {"detector": detector})()
    encoded = json.dumps(expected, ensure_ascii=False) if release == "0.5.14" or not isinstance(expected, str) else expected
    suffix = encoded.replace(actual, "", 1) if release == "0.5.14" and actual in encoded else encoded[len(actual):] if release != "0.5.14" and encoded.startswith(actual) else ""
    with mock.patch.object(module, "version", return_value=release):
        observed = module._remaining_tool_args(fp)
    assert observed == ([{"kind": "tool_call", "name": None, "arguments": suffix}] if suffix else [])


def test_parser_error_retains_prior_ordered_rows_and_traceback():
    class BrokenLater(_Released):
        def parse_stream_chunk(self, chunk):
            if chunk == "bad":
                raise ValueError("second chunk failure")
            return "", chunk
    module = _load(BrokenLater)
    with mock.patch.object(module, "version", return_value="0.5.21"):
        result = module._capture_cases([{"family": "qwen3", "id": "one", "chunks": ["first", "bad"]}])["one"]
    assert result["partial_chunks"] == [[{"kind": "text", "text": "first"}]]
    assert result["error"] == "ValueError: second chunk failure"
    assert "ValueError: second chunk failure" in result["error_traceback"]
    assert "chunks" not in result
