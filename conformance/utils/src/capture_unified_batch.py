# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the legacy reasoning-then-tools pipeline on an immutable Unified job."""

import argparse
import hashlib
import importlib
import inspect
import json
import sys
from pathlib import Path

import peer_modules
from capture_stimulus import batch_final_events, unavailable_result, validate_family_scoped_cases
from impls import SGLANG_UNIFIED_PIPELINES, VLLM_UNIFIED_PIPELINES, parser_families_path
from peer_modules import VllmTextTokenizer, resolve_peer_module, sglang_reasoning_kwargs, sglang_supports_reasoning_init, parser_error_detail, peer_call


from tables.common import decode_arguments


def capture_cases(job, pipeline, registrations, execute, *, supports_reasoning_init=False):
    validate_family_scoped_cases(job["cases"])
    results = {}
    for case in job["cases"]:
        request = case["request"]
        if request["init"].get("structural_tag") is not None:
            results[case["id"]] = unavailable_result("unsupported_request_mode", "Parser-only batch capture does not execute structural-tag generation constraints.", capture=request, observation=True)
            continue
        names = pipeline.get(case["family"])
        if names is None or not registrations(names):
            result = unavailable_result("unsupported_family", "No registered reasoning + batch tool parser for this family", capture=request, observation=True)
        elif request["init"] not in [{"starting_state": state, "tool_output_mode": "Native", "named_tool": None} for state in (("None", "Reasoning") if supports_reasoning_init else ("None",))] or request["finish_reason"] not in {"stop", "length"}:
            result = unavailable_result("unsupported_request_mode", "Legacy batch API cannot apply this authored initialization or finish reason", capture=request, observation=True)
        else:
            # Parser exceptions are observations. Import/setup failures are fatal and
            # must be handled by the producer launcher, not disguised as parser errors.
            try:
                final = execute(names, request)
                result = {"assembled": batch_final_events(final), "batch_final": final, "capture_input": request}
            except (ValueError, TypeError, KeyError, RuntimeError, IndexError, AssertionError, AttributeError) as error:
                detail = parser_error_detail(error)
                result = {"error": detail, "capture_input": request, "capture_observation": {"error": {"code": "parser_exception", "detail": detail}}}
        results[case["id"]] = result
    return results


def sglang_pipeline():
    protocol = resolve_peer_module("sglang", "protocol")
    reasoning = resolve_peer_module("sglang", "reasoning").ReasoningParser
    tool_parser = resolve_peer_module("sglang", "tools").FunctionCallParser

    def registered(names):
        return names[0] in reasoning.DetectorMap and names[1] in tool_parser.ToolCallParserEnum

    def execute(names, request):
        kwargs = sglang_reasoning_kwargs(reasoning, streaming=False, force_reasoning=request["init"]["starting_state"] == "Reasoning")
        parser = peer_call(reasoning, model_type=names[0], **kwargs)
        reasoning_text, content = peer_call(parser.parse_non_stream, request["input"])
        tools = [protocol.Tool(type="function", function=protocol.Function(**(tool.get("function") or tool))) for tool in request["tools"]]
        parser = peer_call(tool_parser, tools=tools, tool_call_parser=names[1])
        normal_text, calls = peer_call(parser.parse_non_stream, content)
        return {"reasoning_text": reasoning_text or "", "normal_text": normal_text or "", "calls": [{"name": call.name, "arguments": decode_arguments(call.parameters)} for call in calls]}

    return SGLANG_UNIFIED_PIPELINES, registered, execute


def vllm_pipeline():
    reasoning_manager = resolve_peer_module("vllm", "reasoning").ReasoningParserManager
    tool_manager = resolve_peer_module("vllm", "tools").ToolParserManager
    protocol = resolve_peer_module("vllm", "protocol")

    def registered(names):
        return (names[0] in reasoning_manager.reasoning_parsers or names[0] in reasoning_manager.__dict__.get("lazy_parsers", {})) and (names[1] in tool_manager.tool_parsers or names[1] in tool_manager.__dict__.get("lazy_parsers", {}))

    def execute(names, request):
        tools = [protocol.ChatCompletionToolsParam.model_validate(tool if "function" in tool else {"type": "function", "function": tool}) for tool in request["tools"]]
        req = protocol.ChatCompletionRequest.model_construct(model="parser-only-capture", messages=[], tools=tools, tool_choice="auto" if tools else "none")
        parser = peer_modules._vllm_reasoning_parser(names[0], {})
        reasoning_method = parser.extract_reasoning if hasattr(parser, "extract_reasoning") else parser.extract_reasoning_content
        reasoning_text, content = peer_call(reasoning_method, request["input"], req)
        cls = tool_manager.get_tool_parser(names[1])
        kwargs = {"tools": tools} if "tools" in inspect.signature(cls.__init__).parameters else {}
        parser = peer_call(cls, tokenizer=VllmTextTokenizer(), **kwargs)
        result = peer_call(parser.extract_tool_calls, content or "", req)
        return {"reasoning_text": reasoning_text or "", "normal_text": result.content or "", "calls": [{"name": call.function.name, "arguments": decode_arguments(call.function.arguments)} for call in result.tool_calls]}

    return VLLM_UNIFIED_PIPELINES, registered, execute


def source_identity_root(engine, root, version):
    root = Path(root).resolve()
    files = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(root.rglob("*.py"))}
    if not files:
        raise ValueError("producer source inventory is empty")
    adapter_root = Path(__file__).resolve().parent
    config = parser_families_path()
    adapter_files = {config.name: hashlib.sha256(config.read_bytes()).hexdigest()}
    for loaded in tuple(sys.modules.values()):
        path = loaded.__dict__.get("__file__") if loaded is not None else None
        if path:
            path = Path(path).resolve()
            if path.is_relative_to(adapter_root) and path.suffix == ".py":
                adapter_files[str(path.relative_to(adapter_root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"source_sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
            "adapter_sha256": hashlib.sha256(json.dumps(adapter_files, sort_keys=True).encode()).hexdigest(),
            "adapter_files": adapter_files, "package_files": files, "version": version}


def source_identity(engine):
    module = importlib.import_module(engine)
    return source_identity_root(engine, Path(module.__file__).resolve().parent, module.__version__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("vllm", "sglang"), required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pipeline = vllm_pipeline() if args.engine == "vllm" else sglang_pipeline()
    before = source_identity(args.engine)
    args.output.mkdir(parents=True, exist_ok=True)
    for path in sorted(args.jobs.glob("*.json")):
        job = json.loads(path.read_text())
        result = {"version": before["version"], "results": capture_cases(job, *pipeline, supports_reasoning_init=args.engine == "sglang" and sglang_supports_reasoning_init()),
                  "provenance": {"implementation": args.engine + "_python_batch", "version": before["version"], "source_sha256": before["source_sha256"], "adapter_sha256": before["adapter_sha256"], "job_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}}
        (args.output / path.name).write_text(json.dumps(result, ensure_ascii=False))
    if before != source_identity(args.engine):
        raise ValueError("producer source changed during capture")
    (args.output / "source.json").write_text(json.dumps(before, indent=2))


if __name__ == "__main__":
    main()
