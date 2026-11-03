# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from types import ModuleType, SimpleNamespace
import pytest
import peer_modules
import capture_unified_batch
import json
from types import SimpleNamespace
import capture_peer_unified_versions as peers
from peer_modules import AdapterContractError, parser_error_detail, peer_call
import peer_modules as compat


@pytest.mark.parametrize("engine,capability,paths", [
    (engine, capability, paths) for engine, roles in peer_modules.MODULE_PATHS.items()
    for capability, paths in roles.items()
])
def test_published_layout_selects_each_actual_module_without_import_fallback(tmp_path, monkeypatch, engine, capability, paths):
    package = ModuleType(engine)
    package.__file__ = str(tmp_path / "__init__.py")
    sentinel = SimpleNamespace()
    imported = []

    def load(name):
        imported.append(name)
        return package if name == engine else sentinel

    monkeypatch.setattr(peer_modules.importlib, "import_module", load)
    for relative in reversed(paths):
        path = tmp_path.joinpath(*relative.split(".")).with_suffix(".py")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"# published API\n")
        assert peer_modules.published_module_paths(tmp_path, engine)[capability] == relative
        assert peer_modules.resolve_peer_module(engine, capability) is sentinel
        assert imported[-1] == engine + "." + relative


def test_present_modern_module_dependency_error_never_selects_old_layout(tmp_path, monkeypatch):
    package = ModuleType("sglang")
    package.__file__ = str(tmp_path / "__init__.py")
    for relative in peer_modules.MODULE_PATHS["sglang"]["reasoning"]:
        path = tmp_path.joinpath(*relative.split(".")).with_suffix(".py")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"# real module\n")
    imported = []

    def load(name):
        imported.append(name)
        if name == "sglang":
            return package
        raise ModuleNotFoundError("upstream_dependency_sentinel")

    monkeypatch.setattr(peer_modules.importlib, "import_module", load)
    with pytest.raises(ModuleNotFoundError, match="upstream_dependency_sentinel"):
        peer_modules.resolve_peer_module("sglang", "reasoning")
    assert imported == ["sglang", "sglang.srt.parser.reasoning_parser"]


def test_old_reasoning_api_preserves_native_defaults_and_rejects_unrepresentable_init():
    class PublishedOldReasoning:
        def __init__(self, model_type=None, stream_reasoning=True):
            pass

    assert peer_modules.sglang_reasoning_kwargs(PublishedOldReasoning, streaming=True) == {"stream_reasoning": True}
    with pytest.raises(ValueError, match="cannot initialize forced reasoning"):
        peer_modules.sglang_reasoning_kwargs(PublishedOldReasoning, streaming=False, force_reasoning=True)


@pytest.fixture(params=["content", "renamed"])
def legacy_parser(monkeypatch, request):
    calls = []

    class Reasoning:
        def __init__(self, tokenizer):
            pass

        def extract_reasoning_content_streaming(self, previous, current, delta, previous_ids, current_ids, delta_ids):
            calls.append(("reasoning", previous, current, delta))
            return SimpleNamespace(content="held" if 99 in delta_ids else None, reasoning_content=delta, tool_calls=None)

        def is_reasoning_end(self, ids):
            return 99 in ids

        def extract_content_ids(self, ids):
            return ids[ids.index(99) + 1:]

        def extract_reasoning_content(self, text, request):
            return "reason", text

    class Tools:
        def __init__(self, tokenizer):
            self.prev_tool_call_arr = []
            self.streamed_args_for_tool = []

        def extract_tool_calls_streaming(self, **kwargs):
            calls.append(("tools", kwargs))
            return SimpleNamespace(content=kwargs["delta_text"], tool_calls=None)

        def extract_tool_calls(self, text, request):
            return SimpleNamespace(content=text, tool_calls=[])

    if request.param == "renamed":
        Reasoning.extract_reasoning_streaming = Reasoning.extract_reasoning_content_streaming
        Reasoning.extract_reasoning = Reasoning.extract_reasoning_content
        del Reasoning.extract_reasoning_content_streaming
        del Reasoning.extract_reasoning_content

    protocol = SimpleNamespace(DeltaMessage=lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(peer_modules, "resolve_peer_module", lambda *_args: protocol)
    parser = type("PublishedParser", (peer_modules.LegacyServingParser,), {
        "reasoning_class": Reasoning, "tool_class": Tools})
    return parser, calls


def test_legacy_serving_preserves_transition_delay_and_previous_input(legacy_parser):
    parser_class, calls = legacy_parser
    parser = parser_class(object(), tools=[{}])
    parser.parse_delta("reason", [1], object())
    boundary = parser.parse_delta("end", [99, 2], object())
    assert boundary.content is None
    assert not [call for call in calls if call[0] == "tools"]
    assert parser.parse_delta("next", [3], object()).content == "heldnext"
    assert calls[-1] == ("tools", {"previous_text": "", "current_text": "heldnext", "delta_text": "heldnext",
                                  "previous_token_ids": [], "current_token_ids": [2, 3], "delta_token_ids": [2, 3], "request": calls[-1][1]["request"]})
    parser.parse_delta("tail", [4], object())
    assert calls[-1][1]["previous_text"] == "heldnext"
    assert calls[-1][1]["delta_text"] == "tail"


def test_legacy_serving_final_reasoning_boundary_does_not_invent_same_chunk_tool_output(legacy_parser):
    parser_class, calls = legacy_parser
    parser = parser_class(object(), tools=[{}])
    assert parser.parse_delta("end", [99, 2], object(), finished=True).content is None
    assert not [call for call in calls if call[0] == "tools"]
    no_tools = parser_class(object(), tools=[])
    assert no_tools.parse_delta("end", [99, 2], object(), finished=True).content == "held"
    assert no_tools.parse("input", object(), True, None) == ("reason", "input", [])


def test_legacy_serving_prompt_end_routes_next_delta_to_tools(legacy_parser):
    parser_class, calls = legacy_parser
    parser = parser_class(object(), tools=[{}])
    parser.parse_delta("first", [2], object(), prompt_ids=[99])
    parser.parse_delta("second", [3], object())
    assert calls[-1][0] == "tools"
    assert calls[-1][1]["current_token_ids"] == [2, 3]


def test_later_legacy_serving_routes_boundary_tail_in_the_same_delta(legacy_parser):
    parser_class, calls = legacy_parser
    parser = parser_class(object(), tools=[{}])
    parser.immediate_transition = True
    result = parser.parse_delta("end", [99, 2], object(), finished=True)
    assert result.content == "held"
    assert calls[-1][0] == "tools"
    assert calls[-1][1]["delta_token_ids"] == [2]
    assert calls[-1][1]["previous_text"] == ""


def test_later_legacy_serving_prompt_end_bypasses_reasoning_and_preserves_text(legacy_parser):
    parser_class, calls = legacy_parser
    parser = parser_class(object(), tools=[{}])
    parser.immediate_transition = True
    assert parser.parse_delta("first", [2], object(), prompt_ids=[99]).content == "first"
    assert [call[0] for call in calls] == ["tools"]


def test_early_legacy_serving_does_not_invent_prompt_initialization(legacy_parser):
    parser_class, calls = legacy_parser
    parser = parser_class(object(), tools=[{}])
    parser.prompt_end_transition = False
    parser.parse_delta("first", [2], object(), prompt_ids=[99])
    assert [call[0] for call in calls] == ["reasoning"]
    assert not parser.reasoning_end


@pytest.mark.parametrize("immediate", [False, True])
def test_published_serving_source_selects_actual_transition(tmp_path, immediate):
    path = tmp_path / "entrypoints/openai/serving_chat.py"
    path.parent.mkdir(parents=True)
    tail = "\n    if reasoning_end_arr[i]:\n        tools()" if immediate else "\n    else:\n        tools()"
    path.write_bytes(("if tool_choice_auto and self.reasoning_parser:\n    if not reasoning_end_arr[i]:\n        reason()" + tail + "\n").encode())
    profile = peer_modules.published_serving_profile(tmp_path)
    assert profile["immediate_transition"] is immediate
    assert profile["prompt_end_transition"] is False
    assert profile["preserve_final_identity"] is False


def test_unknown_published_serving_branch_is_not_guessed(tmp_path):
    path = tmp_path / "entrypoints/openai/serving_chat.py"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"unexpected_serving_api()\n")
    with pytest.raises(ValueError, match="not recognized"):
        peer_modules.published_serving_profile(tmp_path)


@pytest.mark.parametrize("finished", [False, True])
def test_legacy_serving_flushes_only_the_actual_final_argument_suffix(legacy_parser, finished):
    parser_class, _calls = legacy_parser
    parser = parser_class(object(), tools=[{}])
    parser.parse_delta("end", [99, 2], object())
    parser.tool.prev_tool_call_arr = [{"arguments": {"city": "東京"}}]
    parser.tool.streamed_args_for_tool = ['{"city": "東']
    message = SimpleNamespace(tool_calls=[SimpleNamespace(function=SimpleNamespace(arguments="東"))])
    parser.tool.extract_tool_calls_streaming = lambda **_kwargs: message
    parser.protocol = SimpleNamespace(DeltaMessage=lambda **kwargs: SimpleNamespace(**kwargs),
                                      DeltaToolCall=lambda **kwargs: SimpleNamespace(**kwargs),
                                      DeltaFunctionCall=lambda **kwargs: SimpleNamespace(**kwargs))
    actual = parser.parse_delta("next", [3], object(), finished=finished)
    assert actual.tool_calls[0].function.arguments == ('東京"}' if finished else "東")
    if finished:
        assert actual.tool_calls[0].index == 0


@pytest.mark.parametrize("preserve,matching", [(False, True), (True, True), (True, False)])
def test_final_only_tool_identity_follows_published_flush(legacy_parser, preserve, matching):
    parser_class, _calls = legacy_parser
    parser = parser_class(object(), tools=[{}])
    parser.reasoning_end = True
    parser.preserve_final_identity = preserve
    parser.tool.prev_tool_call_arr = [{"arguments": {}}]
    parser.tool.streamed_args_for_tool = ["{}"]
    original = SimpleNamespace(index=0 if matching else 1, id="call-1", type="function",
                               function=SimpleNamespace(name="lookup", arguments="{}"))
    parser.tool.extract_tool_calls_streaming = lambda **_kwargs: SimpleNamespace(tool_calls=[original])
    parser.protocol = SimpleNamespace(DeltaMessage=lambda **kwargs: SimpleNamespace(**kwargs),
                                      DeltaToolCall=lambda **kwargs: SimpleNamespace(**kwargs),
                                      DeltaFunctionCall=lambda **kwargs: SimpleNamespace(**kwargs))
    result = parser.parse_delta("call", [3], object(), finished=True).tool_calls[0]
    assert result.function.arguments == "{}"
    if preserve:
        assert (result.id, result.type, result.function.name) == (("call-1", "function", "lookup") if matching else (None, None, None))
    else:
        assert vars(result.function) == {"arguments": "{}"}


def test_published_final_flush_identity_is_selected_from_its_call(tmp_path):
    path = tmp_path / "entrypoints/openai/serving_chat.py"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"if tool_choice_auto and self.reasoning_parser:\n    if not reasoning_end_arr[i]:\n        reason()\n    else:\n        tools()\nself._create_remaining_args_delta(delta_message, remaining_call, index)\n")
    assert peer_modules.published_serving_profile(tmp_path)["preserve_final_identity"]


@pytest.mark.parametrize("error", [ImportError("dependency sentinel"), ModuleNotFoundError("lazy dependency sentinel")])
def test_dependency_failure_is_not_a_parser_observation(error):
    with pytest.raises(ImportError, match="dependency sentinel"):
        peer_modules.parser_error_detail(error)
    assert peer_modules.parser_error_detail(ValueError("bad input")) == "ValueError: bad input"


def test_published_manager_wrapped_dependency_failure_is_not_unsupported():
    try:
        try:
            raise ModuleNotFoundError("lazy dependency sentinel")
        except ImportError as dependency:
            raise TypeError("parser has not been registered") from dependency
    except TypeError as registration:
        with pytest.raises(ModuleNotFoundError, match="lazy dependency sentinel"):
            peer_modules.parser_error_detail(registration)


def test_legacy_serving_reinitialization_drops_every_request_state(legacy_parser):
    parser_class, _calls = legacy_parser
    parser = parser_class(object(), tools=[{}])
    parser.parse_delta("end", [99, 2], object())
    parser.parse_delta("next", [3], object())
    reasoning, tool = parser.reasoning, parser.tool
    parser.initialize_streaming()
    assert parser.reasoning is not reasoning and parser.tool is not tool
    assert (parser.previous_text, parser.previous_ids, parser.reasoning_end, parser.added_content) == ("", [], False, False)


def test_source_inventory_includes_flat_historical_modules_without_engine_import(tmp_path, monkeypatch):
    package = tmp_path / "sglang"
    (package / "srt").mkdir(parents=True)
    (package / "__init__.py").write_bytes(b'__version__="0.4.9"\n')
    (package / "srt/reasoning_parser.py").write_bytes(b'class ReasoningParser: pass\n')

    def forbidden_import(_name):
        raise AssertionError("source inspection must not import optional engine dependencies")

    monkeypatch.setattr(capture_unified_batch.importlib, "import_module", forbidden_import)
    identity = capture_unified_batch.source_identity_root("sglang", package, "0.4.9")
    assert set(identity["package_files"]) == {"__init__.py", "srt/reasoning_parser.py"}
    assert identity["version"] == "0.4.9"


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
