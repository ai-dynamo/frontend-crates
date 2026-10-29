# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve parser APIs from the installed published package, preserving import errors."""

import importlib
import inspect
import sys
from pathlib import Path
import ast
import json


MODULE_PATHS = {
    "sglang": {
        "protocol": ("srt.entrypoints.openai.protocol", "srt.openai_api.protocol"),
        "reasoning": ("srt.parser.reasoning_parser", "srt.reasoning_parser"),
        "tools": ("srt.function_call.function_call_parser", "srt.function_call_parser"),
    },
    "vllm": {
        "protocol": (
            "entrypoints.openai.chat_completion.protocol",
            "entrypoints.openai.protocol",
        ),
        "tools": ("tool_parsers", "entrypoints.openai.tool_parsers"),
        "reasoning": ("reasoning",),
        "combined": ("parser.parser_manager",),
    },
}


class AdapterContractError(RuntimeError):
    """The capture adapter cannot represent the published call contract."""


def peer_call(function, /, *args, **kwargs):
    try:
        inspect.signature(function).bind(*args, **kwargs)
    except TypeError as error:
        raise AdapterContractError(f"Invalid published API call {function}: {error}") from error
    return function(*args, **kwargs)


def parser_error_detail(error):
    cause = error
    visited = set()
    while cause is not None and id(cause) not in visited:
        visited.add(id(cause))
        if isinstance(cause, (ImportError, AdapterContractError)):
            raise cause
        cause = cause.__cause__ or (None if cause.__suppress_context__ else cause.__context__)
    return f"{type(error).__name__}: {error}"


def published_module_paths(root, engine):
    """Describe recognized APIs without importing their optional dependencies."""
    result = {}
    for capability, candidates in MODULE_PATHS[engine].items():
        result[capability] = None
        for relative in candidates:
            path = root.joinpath(*relative.split("."))
            if path.with_suffix(".py").is_file() or (path / "__init__.py").is_file():
                result[capability] = relative
                break
    return result


def resolve_peer_module(engine, capability):
    package = importlib.import_module(engine)
    root = Path(package.__file__).resolve().parent if package.__dict__.get("__file__") else None
    if root is None:
        for relative in MODULE_PATHS[engine][capability]:
            name = engine + "." + relative
            if name in sys.modules:
                return importlib.import_module(name)
    else:
        relative = published_module_paths(root, engine)[capability]
        if relative is not None:
            # A present module's missing dependency is a source/build blocker;
            # switching layouts on ImportError would hide that failure.
            return importlib.import_module(engine + "." + relative)
    raise ModuleNotFoundError(f"Published {engine} package has no {capability} parser API")


def sglang_reasoning_kwargs(parser, *, streaming, force_reasoning=False):
    parameters = inspect.signature(parser.__init__).parameters
    kwargs = {"stream_reasoning": streaming}
    if "force_reasoning" in parameters:
        kwargs["force_reasoning"] = force_reasoning
    elif force_reasoning:
        raise ValueError("Published SGLang reasoning API cannot initialize forced reasoning")
    if "tool_call_parser_active" in parameters:
        # The next detector needs channel headers that reasoning-only serving strips.
        kwargs["tool_call_parser_active"] = True
    return kwargs


def sglang_supports_reasoning_init():
    parser = resolve_peer_module("sglang", "reasoning").ReasoningParser
    return "force_reasoning" in inspect.signature(parser.__init__).parameters


def vllm_stream_is_legacy():
    package = importlib.import_module("vllm")
    if not package.__dict__.get("__file__"):
        return "vllm.parser.parser_manager" not in sys.modules
    if published_module_paths(Path(package.__file__).resolve().parent, "vllm")["combined"] is None:
        return True
    return not hasattr(importlib.import_module("vllm.parser.abstract_parser").Parser, "parse_delta")


def vllm_parser_factory():
    if not vllm_stream_is_legacy():
        return resolve_peer_module("vllm", "combined").ParserManager().get_parser
    return parser_factory


class _TextToolVocab(dict):
    def get(self, key, default=None):
        return 1


class VllmTextTokenizer:
    """Synthetic token IDs are only used to construct text-mode tool parsers."""

    def get_vocab(self):
        return _TextToolVocab()


def published_serving_profile(root, require_transition=True):
    for relative in ("entrypoints/openai/chat_completion/serving.py", "entrypoints/openai/serving_chat.py"):
        path = root / relative
        if not path.is_file():
            continue
        tree = ast.parse(path.read_bytes())
        preserves_identity = any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                                 and node.func.attr == "_create_remaining_args_delta" for node in ast.walk(tree))
        profile = {"preserve_final_identity": preserves_identity,
                   "preserve_final_strings": any(isinstance(node, ast.Assign) and any(
                       isinstance(target, ast.Name) and target.id == "expected_call" for target in node.targets)
                       and isinstance(node.value, ast.Name) and node.value.id == "args" for node in ast.walk(tree)),
                   "require_detected_tool": any(isinstance(node, ast.If) and "auto_tools_called" in ast.unparse(node.test)
                       and any(isinstance(child, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "expected_call"
                           for target in child.targets) for child in ast.walk(node)) for node in ast.walk(tree))}
        for branch in ast.walk(tree):
            if isinstance(branch, ast.If) and ast.unparse(branch.test) in (
                    "tool_choice_auto and self.reasoning_parser", "tool_choice_auto and self.enable_reasoning",
                    "tool_choice_auto and reasoning_parser"):
                prompt_end = any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                                 and node.func.attr == "is_reasoning_end"
                                 and any(ast.unparse(arg) == "res.prompt_token_ids" for arg in node.args)
                                 for node in ast.walk(branch))
                prompt_end = prompt_end or any(isinstance(node, ast.Name) and node.id == "prompt_is_reasoning_end_arr"
                                              for node in ast.walk(branch))
                for transition in branch.body:
                    if isinstance(transition, ast.If) and ast.unparse(transition.test) == "reasoning_end_arr[i]":
                        return {**profile, "immediate_transition": True, "prompt_end_transition": prompt_end}
                    if isinstance(transition, ast.If) and ast.unparse(transition.test) == "not reasoning_end_arr[i]" and transition.orelse:
                        return {**profile, "immediate_transition": False, "prompt_end_transition": prompt_end}
        if not require_transition:
            return profile
        raise ValueError("Published legacy vLLM serving transition is not recognized")
    raise ValueError("Published legacy vLLM serving source is missing")


def finish_serving_delta(message, tool, protocol, profile):
    if not (tool and message and message.tool_calls and message.tool_calls[0].function):
        return message
    if profile["require_detected_tool"] and not tool.prev_tool_call_arr:
        return message
    arguments = message.tool_calls[0].function.arguments
    if arguments is None:
        return message
    index = max(0, len(tool.prev_tool_call_arr) - 1)
    expected = tool.prev_tool_call_arr[index].get("arguments", {})
    if not (profile["preserve_final_strings"] and isinstance(expected, str)):
        expected = json.dumps(expected, ensure_ascii=False)
    actual = tool.streamed_args_for_tool[index]
    if arguments:
        actual = actual[:-len(arguments)]
    remaining = expected.replace(actual, "", 1)
    identity, function_identity = {}, {}
    if profile["preserve_final_identity"]:
        original = next((call for call in message.tool_calls if call.index == index), None)
        identity = {"id": original.id if original else None, "type": original.type if original else None}
        function_identity = {"name": original.function.name if original and original.function else None}
    return peer_call(protocol.DeltaMessage, tool_calls=[peer_call(protocol.DeltaToolCall,
        index=index, **identity, function=peer_call(protocol.DeltaFunctionCall, arguments=remaining, **function_identity))])


class ModernServingParser:
    """Call the released manager API and preserve server-owned EOF handling."""

    def __init__(self, parser_class, tokenizer, *, tools, chat_template_kwargs=None):
        self.tokenizer = tokenizer
        self.parser = peer_call(parser_class, tokenizer, tools=tools, chat_template_kwargs=chat_template_kwargs)
        self.token_engine = any(base.__module__ == "vllm.parser.engine.parser_engine" for base in parser_class.__mro__)
        self.protocol = resolve_peer_module("vllm", "protocol")
        self.has_finished = "finished" in inspect.signature(self.parser.parse_delta).parameters
        package = importlib.import_module("vllm")
        self.profile = None if self.has_finished else published_serving_profile(Path(package.__file__).resolve().parent, False)

    def parse_delta(self, text, token_ids, request, prompt_ids=None, *, finished=False):
        kwargs = {"finished": finished} if self.has_finished else {}
        message = peer_call(self.parser.parse_delta, text, token_ids, request, prompt_ids, **kwargs)
        if finished and not self.has_finished:
            message = finish_serving_delta(message, self.parser.tool_parser, self.protocol, self.profile)
        return message

    def initialize_streaming(self):
        if hasattr(self.parser, "initialize_streaming"):
            peer_call(self.parser.initialize_streaming)

    def parse(self, text, request, _finished=True, _prompt_ids=None):
        if hasattr(self.parser, "parse"):
            parameters = inspect.signature(self.parser.parse).parameters
            kwargs = {"enable_auto_tools": True}
            if "model_output_token_ids" in parameters:
                kwargs["model_output_token_ids"] = self.tokenizer.generated_ids("", text)
            result = peer_call(self.parser.parse, text, request, **kwargs)
        else:
            reasoning, content = peer_call(self.parser.extract_reasoning, text, request)
            if not request.tools or request.tool_choice == "none":
                result = reasoning, content, []
            else:
                parsed = peer_call(self.parser.extract_tool_calls, content or "", request)
                result = reasoning, parsed.content, parsed.tool_calls
        if not isinstance(result, tuple) or len(result) != 3:
            raise AdapterContractError("Published final-message API returned an unexpected shape")
        if result[2] is not None and not isinstance(result[2], (list, tuple)):
            raise AdapterContractError("Published final-message tool calls have an unexpected shape")
        return result


def capture_parser(parser_class, tokenizer, **kwargs):
    if not isinstance(parser_class, type):
        raise AdapterContractError("Published manager returned a non-class parser factory")
    if issubclass(parser_class, LegacyServingParser):
        return peer_call(parser_class, tokenizer, **kwargs)
    return ModernServingParser(parser_class, tokenizer, **kwargs)


class LegacyServingParser:
    reasoning_class = None
    tool_class = None
    immediate_transition = False
    prompt_end_transition = True
    preserve_final_identity = False
    preserve_final_strings = False
    require_detected_tool = False
    token_engine = False

    def __init__(self, tokenizer, *, tools, chat_template_kwargs=None):
        self.tokenizer = tokenizer
        self.tools = tools
        self.protocol = resolve_peer_module("vllm", "protocol")
        self.initialize_streaming()

    def initialize_streaming(self):
        self.reasoning = peer_call(self.reasoning_class, self.tokenizer)
        self.tool = peer_call(self.tool_class, self.tokenizer) if self.tools else None
        self.reasoning_stream = (self.reasoning.extract_reasoning_streaming if hasattr(self.reasoning, "extract_reasoning_streaming")
                                 else self.reasoning.extract_reasoning_content_streaming)
        self.reasoning_final = (self.reasoning.extract_reasoning if hasattr(self.reasoning, "extract_reasoning")
                                else self.reasoning.extract_reasoning_content)
        self.previous_text = ""
        self.previous_ids = []
        self.reasoning_end = False
        self.added_content = False

    def parse_delta(self, delta_text, delta_ids, request, prompt_ids=None, *, finished=False):
        previous_text, previous_ids = self.previous_text, self.previous_ids
        current_text = previous_text + delta_text
        current_ids = previous_ids + list(delta_ids)
        started_in_tools = self.reasoning_end
        if self.tool is None:
            message = peer_call(self.reasoning_stream,
                previous_text, current_text, delta_text, previous_ids, current_ids, delta_ids)
        elif not self.reasoning_end:
            prompt_ended = self.prompt_end_transition and prompt_ids and self.reasoning.is_reasoning_end(prompt_ids)
            if self.immediate_transition and prompt_ended:
                self.reasoning_end = True
                current_ids = list(delta_ids)
                message = None
            else:
                message = peer_call(self.reasoning_stream,
                    previous_text, current_text, delta_text, previous_ids, current_ids, delta_ids)
                if prompt_ended:
                    self.reasoning_end = True
                    current_ids = list(delta_ids)
                    current_text = message.content if message and message.content else ""
                    if message and message.content:
                        message.content = None
                if self.reasoning.is_reasoning_end(list(delta_ids)):
                    self.reasoning_end = True
                    current_ids = self.reasoning.extract_content_ids(list(delta_ids))
                    current_text = message.content if message and message.content else ""
                    if message and message.content:
                        message.content = None
        if self.tool is not None and (started_in_tools or self.immediate_transition and self.reasoning_end):
            if not self.added_content:
                self.added_content = True
                previous_text, previous_ids = "", []
                delta_text, delta_ids = current_text, current_ids
            message = peer_call(self.tool.extract_tool_calls_streaming,
                previous_text=previous_text, current_text=current_text, delta_text=delta_text,
                previous_token_ids=previous_ids, current_token_ids=current_ids,
                delta_token_ids=list(delta_ids), request=request)
        self.previous_text, self.previous_ids = current_text, current_ids
        # Follow the published serving branch: older releases hold the boundary
        # content until the next delta, while later releases route it immediately.
        if finished:
            message = finish_serving_delta(message, self.tool, self.protocol, {
                "preserve_final_identity": self.preserve_final_identity,
                "preserve_final_strings": self.preserve_final_strings,
                "require_detected_tool": self.require_detected_tool})
        return message

    def parse(self, text, request, _finished, _prompt_ids):
        reasoning, content = peer_call(self.reasoning_final, text, request)
        if self.tool is None:
            return reasoning, content, []
        result = peer_call(self.tool.extract_tool_calls, content or "", request)
        return reasoning, result.content, result.tool_calls


def parser_factory(*, tool_parser_name, reasoning_parser_name, **_options):
    reasoning = resolve_peer_module("vllm", "reasoning").ReasoningParserManager
    tools = resolve_peer_module("vllm", "tools").ToolParserManager
    reasoning_class = reasoning.get_reasoning_parser(reasoning_parser_name)
    tool_class = tools.get_tool_parser(tool_parser_name)
    package = importlib.import_module("vllm")
    profile = published_serving_profile(Path(package.__file__).resolve().parent)
    return type("PublishedLegacyServingParser", (LegacyServingParser,), {
        "reasoning_class": reasoning_class, "tool_class": tool_class,
        **profile})


_REASONING_MARKER_IDS = {
    "<think>": 201,
    "</think>": 202,
    "<mm:think>": 203,
    "</mm:think>": 204,
    "[THINK]": 205,
    "[/THINK]": 206,
    "◁think▷": 207,  # Kimi legacy unicode delimiters
    "◁/think▷": 208,
    # Gemma 4 channel markers (used by the ParserEngine adapter's boundary check).
    "<|channel>": 209,
    "<channel|>": 210,
    # vLLM 0.23.0's Gemma4 parser resolves these three ids in its constructor
    # (self.vocab["<|turn>"] / ["<|tool_call>"] / ["<|tool_response>"]); without them the
    # mock tokenizer raises KeyError and every gemma4 case captures as an error instead of
    # a result. Later vLLM releases dropped the lookups, which is why it only bites 0.23.0.
    "<|turn>": 211,
    "<|tool_call>": 212,
    "<tool_call|>": 213,
    "<|tool_response>": 214,
}


_MARKERS_BY_LENGTH = sorted(_REASONING_MARKER_IDS, key=len, reverse=True)


def _synthetic_token_ids(text):
    ids = []
    pos = 0
    while pos < len(text):
        matches = [
            (text.find(token, pos), token)
            for token in _MARKERS_BY_LENGTH
            if text.find(token, pos) != -1
        ]
        if not matches:
            break
        start, token = min(matches)
        ids.append(_REASONING_MARKER_IDS[token])
        pos = start + len(token)
    return ids


def _vllm_mock_tokenizer():
    vocab = dict(_REASONING_MARKER_IDS)

    class MockTok:
        all_special_tokens = list(vocab)
        vocab_size = 1000

        def get_vocab(self):
            return dict(vocab)

        def convert_ids_to_tokens(self, ids, **kw):
            by_id = {v: k for k, v in vocab.items()}
            return [by_id.get(i, "") for i in ids]

        def convert_tokens_to_ids(self, tokens):
            if isinstance(tokens, str):
                return vocab.get(tokens)
            return [vocab.get(token) for token in tokens]

        def decode(self, ids, **kw):
            return ""

        def encode(self, text, **kw):
            return _synthetic_token_ids(text)

    return MockTok()


def _vllm_chat_template_kwargs(parser_name, case):
    """Mirror tests/parity/reasoning/table.py::_vllm_chat_template_kwargs so the
    capture matches the documented parity-harness flags."""
    kwargs = {}
    if parser_name == "deepseek_v4":
        kwargs["enable_thinking"] = True
    if parser_name == "deepseek_v3":
        kwargs["thinking"] = True
    kwargs.update(case.get("chat_template_kwargs") or {})
    return kwargs


def _vllm_reasoning_parser(parser_name, case):
    ReasoningParserManager = resolve_peer_module(
        "vllm", "reasoning"
    ).ReasoningParserManager

    cls = ReasoningParserManager.get_reasoning_parser(parser_name)
    kwargs = _vllm_chat_template_kwargs(parser_name, case)
    tok = _vllm_mock_tokenizer()
    if kwargs:
        return cls(tok, chat_template_kwargs=kwargs)
    return cls(tok)
