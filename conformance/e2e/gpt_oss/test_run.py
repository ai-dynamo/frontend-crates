# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from pathlib import Path

import httpx
import pytest

spec = importlib.util.spec_from_file_location("gpt_oss_replay", Path(__file__).with_name("run.py"))
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def sse(delta=None, finish=None):
    return "data: " + json.dumps({"choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}]}) + "\n\n"


def test_replay_assembles_interleaved_tool_indices_and_validates_argument_types():
    data = sse({"reasoning_content": "Use both tools."})
    for index in [0, 1]:
        data += sse({"tool_calls": [{"index": index, "id": f"call_{index}", "type": "function", "function": {"name": "echo", "arguments": '{"count":'}}]})
    for index in [1, 0]:
        data += sse({"tool_calls": [{"index": index, "function": {"arguments": str(index + 1) + '}'}}]})
    data += sse(finish="tool_calls") + "data: [DONE]\n\n"
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=data))
    with httpx.Client(transport=transport) as client:
        trace = {}
        message, fragments, finish = replay.complete(client, "http://test", {"stream": True}, trace)
    tools = [{"function": {"name": "echo", "parameters": {"type": "object", "properties": {"count": {"type": "integer"}}, "required": ["count"]}}}]
    expected = {"calls": [{"name": "echo", "arguments": {"count": n}} for n in [1, 2]]}
    replay.validate(message, expected, tools, "enabled", fragments, True, True)
    assert finish == "tool_calls"
    assert len(trace["events"]) == 6
    message["tool_calls"][0]["function"]["arguments"] = '{"count":"1"}'
    with pytest.raises(Exception, match="not of type 'integer'"):
        replay.validate(message, expected, tools, "enabled", fragments, True, True)


def test_replay_rejects_truncated_transport_even_with_a_finish_reason():
    data = sse({"content": "answer"}) + sse(finish="stop")
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=data))
    with httpx.Client(transport=transport) as client, pytest.raises(AssertionError, match="without \\[DONE\\]"):
        replay.complete(client, "http://test", {"stream": True}, {})


def test_replay_records_http_failure_without_inventing_a_completion():
    transport = httpx.MockTransport(lambda request: httpx.Response(400, json={"error": "invalid named tool"}))
    with httpx.Client(transport=transport) as client:
        trace = {}
        message, fragments, finish = replay.complete(client, "http://test", {"stream": True}, trace)
    assert trace["status"] == 400
    assert "invalid named tool" in trace["error_body"]
    assert not message["tool_calls"] and not fragments and finish is None
