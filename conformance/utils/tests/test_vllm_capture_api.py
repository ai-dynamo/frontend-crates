# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from types import SimpleNamespace

import pytest

import capture_peer_unified_versions as peers
from peer_modules import AdapterContractError, parser_error_detail, peer_call
import vllm_stream_compat as compat


@pytest.fixture
def published_api(tmp_path, monkeypatch):
    source = tmp_path / "entrypoints/openai/chat_completion/serving.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"if should_check and tool_parser and auto_tools_called:\n    if isinstance(args, str):\n        expected_call = args\nself._create_remaining_args_delta(message, remaining, index)\n")
    protocol = SimpleNamespace(DeltaMessage=lambda **kwargs: SimpleNamespace(**kwargs),
                               DeltaToolCall=lambda **kwargs: SimpleNamespace(**kwargs),
                               DeltaFunctionCall=lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(compat, "resolve_peer_module", lambda *_args: protocol)
    monkeypatch.setattr(compat.importlib, "import_module", lambda _name: SimpleNamespace(__file__=str(tmp_path / "__init__.py")))
    calls = []

    class Native:
        def __init__(self, tokenizer, tools=None, **kwargs):
            calls.append(("initialize", tokenizer, tools, kwargs))
            self.tool_parser = SimpleNamespace(prev_tool_call_arr=[{"arguments": '{"a": 1}'}],
                                               streamed_args_for_tool=['{"a": 1'])

        def extract_reasoning(self, text, request):
            calls.append(("extract_reasoning", text, request))
            return "reason", text

        def extract_tool_calls(self, text, request):
            calls.append(("extract_tools", text, request))
            return SimpleNamespace(content=text, tool_calls=[])

        def delta(self, text, ids, request, prompt_ids):
            calls.append(("delta", text, ids, request, prompt_ids))
            return SimpleNamespace(tool_calls=[SimpleNamespace(index=0, id="call-1", type="function",
                                   function=SimpleNamespace(name="lookup", arguments="1"))])

    return Native, calls


@pytest.mark.parametrize("native_finished", [False, True])
def test_published_delta_signature_and_external_eof_ownership(published_api, native_finished):
    native, calls = published_api

    def old_delta(self, text, ids, request, prompt_ids=None):
        return self.delta(text, ids, request, prompt_ids)

    def new_delta(self, text, ids, request, prompt_ids=None, *, finished):
        calls.append(("native_finished", finished))
        return self.delta(text, ids, request, prompt_ids)

    native.parse_delta = new_delta if native_finished else old_delta
    tokenizer, request, tools = object(), SimpleNamespace(tools=[{}], tool_choice="auto"), [{"schema": "specific"}]
    parser = compat.ModernServingParser(native, tokenizer, tools=tools, chat_template_kwargs={"thinking": False})
    assert calls[0] == ("initialize", tokenizer, tools, {"chat_template_kwargs": {"thinking": False}})
    result = parser.parse_delta("delta", [7], request, [9], finished=True).tool_calls[0]
    assert result.function.arguments == ("1" if native_finished else "1}")
    assert (result.id, result.type, result.function.name) == ("call-1", "function", "lookup")
    assert ("native_finished", True) in calls if native_finished else not any(call[0] == "native_finished" for call in calls)
    if not native_finished:
        assert parser.parse("body", request) == ("reason", "body", [])
        request.tool_choice = "none"
        calls.clear()
        assert parser.parse("plain", request) == ("reason", "plain", [])
        assert not any(call[0] == "extract_tools" for call in calls)


@pytest.mark.parametrize("accepts_ids", [False, True])
def test_complete_parse_uses_only_real_context_parameters_and_actual_ids(published_api, accepts_ids):
    native, calls = published_api
    native.parse_delta = lambda self, text, ids, request, prompt_ids=None, *, finished: None

    def old_parse(self, text, request, enable_auto_tools=False):
        calls.append(("parse", text, request, enable_auto_tools))
        return None, text, None

    def new_parse(self, text, request, enable_auto_tools=False, model_output_token_ids=()):
        calls.append(("parse_ids", text, request, enable_auto_tools, model_output_token_ids))
        return None, text, None

    native.parse = new_parse if accepts_ids else old_parse
    tokenizer = SimpleNamespace(generated_ids=lambda previous, text: [11, 12])
    parser = compat.ModernServingParser(native, tokenizer, tools=[])
    request = object()
    assert parser.parse("text", request) == (None, "text", None)
    assert calls[-1] == (("parse_ids", "text", request, True, [11, 12]) if accepts_ids else ("parse", "text", request, True))


def test_adapter_binding_failures_block_but_native_runtime_errors_remain_observations():
    with pytest.raises(AdapterContractError):
        peer_call(lambda text: text, "value", finished=True)
    with pytest.raises(AdapterContractError):
        parser_error_detail(AdapterContractError("return shape sentinel"))

    def native(text):
        raise TypeError("upstream parser sentinel")

    with pytest.raises(TypeError, match="upstream parser sentinel") as failure:
        peer_call(native, "value")
    assert parser_error_detail(failure.value) == "TypeError: upstream parser sentinel"


def test_stream_only_supervisor_does_not_run_or_report_batch(tmp_path, monkeypatch):
    target = tmp_path / "target.json"
    target.write_bytes(json.dumps({"implementation": "vllm", "version": "0.23.0"}).encode())
    commands = []
    monkeypatch.setattr(peers.subprocess, "run", lambda command, **kwargs: commands.append(command) or SimpleNamespace(returncode=0))
    peers.worker_modes(target, tmp_path / "jobs", tmp_path / "captures", ["stream"])
    assert len(commands) == 1 and commands[0][-2:] == ["--mode", "stream"]
    status = json.loads((tmp_path / "worker-status.json").read_bytes())
    assert set(status) == {"stream"}
    assert not (tmp_path / "batch.log").exists()
