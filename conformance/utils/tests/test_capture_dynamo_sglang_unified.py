# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Dynamo SGLang harness must replay each authored chunk exactly once.

Each chunk becomes one synthetic token, and the published post-processor decodes
it. A wrong token-to-chunk mapping or finish placement silently shifts text
between rows or drops the final flush, so these tests pin both against a
processor that decodes incrementally like the published one.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest import mock

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class _Processor:
    """Decodes like ``SglangStreamingPostProcessor``: context plus pending, minus context."""

    def __init__(self, *, tokenizer, tool_call_parser, reasoning_parser, sglang_tools, tool_call_parser_name):
        self.tokenizer, self.context, self.pending, self.finishes = tokenizer, [], [], []

    def process_output(self, engine_response):
        finish = engine_response["finish_reason"]
        self.finishes.append(finish)
        self.pending += engine_response["token_ids"]
        context = self.tokenizer.decode(self.context, skip_special_tokens=False)
        text = self.tokenizer.decode(self.context + self.pending, skip_special_tokens=False)[len(context):]
        if not text and finish is None:
            return None
        self.context, self.pending = self.pending, []
        return {"delta": {"content": text}, "finish_reason": finish}


PROCESSORS: list[_Processor] = []


def _stub(name, **attrs):
    module = type(sys)(name)
    module.__dict__.update(attrs)
    return module


def _load():
    def make(**kwargs):
        PROCESSORS.append(_Processor(**kwargs))
        return PROCESSORS[-1]

    prepost = _stub("dynamo.frontend.sglang_prepost", convert_tools=lambda tools: tools,
                    create_parsers=lambda request, **kwargs: (None, None), SglangStreamingPostProcessor=make)
    detectors = {"ToolCallParserEnum": {}, "DetectorMap": {}}
    modules = {
        "dynamo": _stub("dynamo"), "dynamo.frontend": _stub("dynamo.frontend"),
        "dynamo.frontend.sglang_prepost": prepost,
        "sglang": _stub("sglang"), "sglang.srt": _stub("sglang.srt"),
        "sglang.srt.function_call": _stub("sglang.srt.function_call"),
        "sglang.srt.function_call.function_call_parser": _stub(
            "sglang.srt.function_call.function_call_parser", FunctionCallParser=type("F", (), detectors)),
        "sglang.srt.parser": _stub("sglang.srt.parser"),
        "sglang.srt.parser.reasoning_parser": _stub(
            "sglang.srt.parser.reasoning_parser", ReasoningParser=type("R", (), detectors)),
    }
    with mock.patch.dict(sys.modules, modules):
        sys.modules.pop("capture_dynamo_sglang_unified", None)
        module = importlib.import_module("capture_dynamo_sglang_unified")
    sys.modules.pop("capture_dynamo_sglang_unified", None)
    return module


def _texts(rows):
    return [[delta["text"] for delta in row] for row in rows]


def test_each_chunk_is_decoded_in_its_own_row_with_finish_on_the_last():
    PROCESSORS.clear()
    rows = _load()._stream_chunks("qwen3", ["<think>a", "", "</think>b"], tools=[])
    assert _texts(rows) == [["<think>a"], [], ["</think>b"]]
    assert PROCESSORS[-1].finishes == [None, None, "stop"]


def test_terminal_step_flushes_with_an_empty_finishing_row():
    PROCESSORS.clear()
    rows = _load()._stream_chunks("qwen3", ["a", "b"], terminal_step=True, finish_reason="length", tools=[])
    assert _texts(rows) == [["a"], ["b"], []]
    assert PROCESSORS[-1].finishes == [None, None, "length"]
