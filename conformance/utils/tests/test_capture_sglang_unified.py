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
import sys
from pathlib import Path
from unittest import mock

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

CONSTRUCTED: list[dict] = []


class _Released:
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
    # `Tool`/`Function` are only used to build the module-level TOOLS list.
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
    assert _load(reasoning_parser)._ACTIVE_KWARG == expected


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
