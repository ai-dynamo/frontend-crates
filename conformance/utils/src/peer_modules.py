# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve parser APIs from the installed published package, preserving import errors."""

import importlib
import inspect
import sys
from pathlib import Path


MODULE_PATHS = {
    "sglang": {
        "protocol": ("srt.entrypoints.openai.protocol", "srt.openai_api.protocol"),
        "reasoning": ("srt.parser.reasoning_parser", "srt.reasoning_parser"),
        "tools": ("srt.function_call.function_call_parser", "srt.function_call_parser"),
    },
    "vllm": {
        "protocol": ("entrypoints.openai.chat_completion.protocol", "entrypoints.openai.protocol"),
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
    return importlib.import_module("vllm_stream_compat").parser_factory


class _TextToolVocab(dict):
    def get(self, key, default=None):
        return 1


class VllmTextTokenizer:
    """Synthetic token IDs are only used to construct text-mode tool parsers."""

    def get_vocab(self):
        return _TextToolVocab()
