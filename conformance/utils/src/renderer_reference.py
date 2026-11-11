#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture pinned model renderers independently of the Dynamo implementation.

Ordinary conformance checks consume the recorded results offline. This command
is the explicit refresh path; it never obtains expectations from Dynamo.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType
from typing import Any

try:
    from jinja2 import TemplateError
except ImportError:
    TemplateError = None


ROOT = Path(__file__).resolve().parents[3]
STORE = "conformance/fixtures-renderer"
Json = Any
TRANSFORMERS_VERSION = "5.12.1"
JINJA_VERSION = "3.1.6"
_ENCODERS: dict[str, ModuleType] = {}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_json(path: Path) -> Json:
    return json.loads(path.read_bytes())


def checked_asset(store: Path, asset: dict[str, Json]) -> Path:
    name = asset["path"]
    if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError("Renderer asset path must stay inside the fixture store")
    target = store / name
    if target.is_symlink() or not target.resolve().is_relative_to(store.resolve()):
        raise ValueError("Renderer asset path escapes the fixture store")
    if sha256(target.read_bytes()) != asset["sha256"]:
        raise ValueError(f"Renderer asset hash mismatch: {name}")
    return target


def assets_for(store: Path, model: dict[str, Json]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for asset in model["assets"]:
        if asset["role"] in result:
            raise ValueError(f"Duplicate renderer asset role: {asset['role']}")
        result[asset["role"]] = checked_asset(store, asset)
    return result


def encoder_module(path: Path) -> ModuleType:
    source = path.read_bytes()
    digest = sha256(source)
    if digest not in _ENCODERS:
        name = f"renderer_model_reference_{digest}"
        module = ModuleType(name)
        module.__file__ = str(path)
        sys.modules[name] = module
        # Compile source bytes directly: importing a vendored module through a
        # filesystem loader would write __pycache__ inside the immutable assets.
        exec(compile(source, str(path), "exec"), module.__dict__)
        _ENCODERS[digest] = module
    return _ENCODERS[digest]


def record_transform(
    transformations: list[dict[str, str]], name: str, authority: str
) -> None:
    entry = {"name": name, "authority": authority}
    if entry not in transformations:
        transformations.append(entry)


def joined_reasoning(
    messages: list[dict[str, Json]], transformations: list[dict[str, str]]
) -> None:
    for message in messages:
        reasoning = message.get("reasoning_content")
        if isinstance(reasoning, list):
            message["reasoning_content"] = "\n".join(
                segment for segment in reasoning if isinstance(segment, str) and segment
            )
            record_transform(
                transformations,
                "join_reasoning_segments_with_newlines",
                "independently-authored renderer API contract; the model reference accepts a string",
            )


def boolean_thinking(args: dict[str, Json], default: bool = True) -> bool:
    for key in ("thinking", "enable_thinking"):
        if isinstance(args.get(key), bool):
            return args[key]
    return default


def template_token(value: Json) -> Json:
    return value.get("content") if isinstance(value, dict) else value


def normalize_hf_messages(
    messages: list[dict[str, Json]],
    transformations: list[dict[str, str]],
    content_format: str,
) -> None:
    for message in messages:
        content = message.get("content")
        if content_format == "openai" and isinstance(content, str):
            message["content"] = [{"type": "text", "text": content}]
            record_transform(transformations, "wrap_string_content_as_parts", "explicit openai content-format API contract")
            content = message["content"]
        if isinstance(content, list):
            projected = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    projected.append({"type": "image"})
                    record_transform(transformations, "project_image_url_to_image_marker", "independently-authored renderer media-input contract; no image bytes fetched")
                else:
                    projected.append(part)
            message["content"] = projected
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function", tool_call)
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    function["arguments"] = json.loads(arguments)
                except json.JSONDecodeError:
                    continue
                record_transform(transformations, "decode_historical_tool_arguments", "Hugging Face apply_chat_template tool-history input contract")
    joined_reasoning(messages, transformations)


def hf_render(
    assets: dict[str, Path], messages: list[dict[str, Json]], tools: Json,
    kwargs: dict[str, Json],
) -> str:
    for distribution, expected in (("transformers", TRANSFORMERS_VERSION), ("Jinja2", JINJA_VERSION)):
        actual = importlib.metadata.version(distribution)
        if actual != expected:
            raise RuntimeError(f"Reference capture requires {distribution}=={expected}, found {actual}")
    # Loaded only for reference capture; the Dynamo dispatch needs no Python ML packages.
    module = importlib.import_module("transformers.utils.chat_template_utils")
    tokenizer = load_json(assets["tokenizer_config"])
    context = {key: template_token(tokenizer.get(key)) for key in ("bos_token", "eos_token", "unk_token")}
    context.update(kwargs)
    rendered, _ = module.render_jinja_template(
        conversations=[messages], tools=tools,
        chat_template=assets["chat_template"].read_text(), **context,
    )
    return rendered[0]


def image_count(messages: list[dict[str, Json]]) -> int:
    return sum(
        1 for message in messages
        for part in (message["content"] if isinstance(message.get("content"), list) else [])
        if isinstance(part, dict) and part.get("type") in ("image", "image_url")
    )


def reference_result(
    store: Path, model: dict[str, Json], case: dict[str, Json]
) -> dict[str, Json]:
    assets = assets_for(store, model)
    request = copy.deepcopy(case["request"])
    settings = case["settings"]
    args = copy.deepcopy(settings.get("chat_template_args", {}))
    messages = request["messages"]
    tools = request.get("tools")
    transformations: list[dict[str, str]] = []
    metadata = case["metadata_expectation"]
    metadata_origin = case["metadata_origin"]
    call: dict[str, Json] = {}
    try:
        if model["formatter"] == "kimi_k3":
            if any(message.get("partial") is True for message in messages):
                return {
                    "result": {"kind": "unsupported", "reason": "The pinned official K3 encoder has no Partial Mode interface; this renderer extension is unmeasured by that reference."},
                    "reference_input": {"messages": messages, "tools": tools, "kwargs": args},
                    "transformations": [], "metadata_origin": metadata_origin,
                }
            joined_reasoning(messages, transformations)
            thinking = boolean_thinking(args)
            kwargs = {
                "add_generation_prompt": settings["add_generation_prompt"],
                "thinking": thinking,
                "thinking_effort": args.get("thinking_effort", args.get("reasoning_effort", "max")),
                "image_prompts": ["<|media_pad|>"] * image_count(messages),
            }
            if "tool_choice" in request:
                kwargs["tool_choice"] = request["tool_choice"]
            if "response_format" in request:
                kwargs["response_format"] = request["response_format"]
            if settings["exclude_tools_when_tool_choice_none"] and request.get("tool_choice") == "none":
                tools = None
                record_transform(transformations, "exclude_tools_when_tool_choice_none", "explicit renderer constructor policy")
            record_transform(transformations, "resolve_k3_default_effort_and_image_prompt_hook", "independently-authored API defaults plus official encoding_k3.py image_prompts hook")
            call = {"messages": messages, "tools": tools, "kwargs": kwargs}
            segments = encoder_module(assets["encoder"]).build_chat_segments(messages, tools=tools, **kwargs)
            encoded_segments = [{"text": segment.text, "allow_special": segment.allow_special} for segment in segments]
            text = "".join(segment.text for segment in segments)
        elif model["formatter"] == "deepseek_v41":
            joined_reasoning(messages, transformations)
            thinking = boolean_thinking(args)
            if not any(isinstance(args.get(key), bool) for key in ("thinking", "enable_thinking")):
                thinking = args.get("thinking_mode", "thinking") != "chat"
            effort = request.get("reasoning_effort", args.get("reasoning_effort", 75))
            if effort == "none":
                thinking, effort = False, 75
            effort = {"low": 50, "medium": 62, "high": 75, "max": 100}.get(effort, effort)
            if effort is None:
                effort = 75
            if request.get("tool_choice") == "none":
                tools = None
                record_transform(transformations, "exclude_tools_when_tool_choice_none", "DeepSeek formatter API tool-choice contract")
            response_format = request.get("response_format")
            if tools or response_format:
                if not messages or messages[0]["role"] not in ("system", "developer"):
                    messages.insert(0, {"role": "system", "content": ""})
                if tools:
                    messages[0]["tools"] = tools
                if response_format:
                    messages[0]["response_format"] = response_format
                record_transform(transformations, "attach_top_level_tools_and_response_format", "official DeepSeek encoder message-level tools and response_format interface")
            for message in messages:
                if message["role"] == "developer":
                    message["role"] = "system"
                    record_transform(transformations, "developer_role_as_system", "independently-authored DeepSeek renderer API contract")
            kwargs = {"thinking_mode": "thinking" if thinking else "chat", "drop_thinking": args.get("drop_thinking", True), "reasoning_effort": effort}
            call = {"messages": messages, "kwargs": kwargs}
            if settings["add_generation_prompt"]:
                text, _ = encoder_module(assets["encoder"]).encode_messages(messages, return_multi_modal_data=True, **kwargs)
            else:
                kwargs["add_generation_prompt"] = False
                call["interface"] = "official_chat_template"
                record_transform(transformations, "use_official_template_generation_flag", "official Python encode_messages has no add_generation_prompt argument")
                text = hf_render(assets, messages, None, kwargs)
            encoded_segments = None
        else:
            if settings["content_format"] == "string":
                return {"result": {"kind": "unsupported", "reason": "The MVP reference adapter does not emulate stock-engine multimodal string flattening."}, "reference_input": {"messages": messages}, "transformations": []}
            normalize_hf_messages(messages, transformations, settings["content_format"])
            kwargs = {"add_generation_prompt": settings["add_generation_prompt"]}
            if "response_format" in request:
                kwargs["response_format"] = request["response_format"]
            kwargs.update(args)
            if settings["exclude_tools_when_tool_choice_none"] and request.get("tool_choice") == "none":
                tools = None
                record_transform(transformations, "exclude_tools_when_tool_choice_none", "explicit renderer constructor policy")
            call = {"messages": messages, "tools": tools, "kwargs": kwargs}
            text = hf_render(assets, messages, tools, kwargs)
            encoded_segments = None
        result = {
            "kind": "prompt", "text": text, "segments": encoded_segments,
            "pending_segments": metadata["pending_segments"],
            "media_message_order": metadata["media_message_order"],
        }
    except (ValueError, AssertionError) as error:
        result = {"kind": "error", "error_type": "invalid_request", "message": str(error)}
    except Exception as error:
        if TemplateError is not None and isinstance(error, TemplateError):
            result = {"kind": "error", "error_type": "invalid_request", "message": str(error)}
        else:
            raise
    return {"result": result, "reference_input": call, "transformations": transformations, "metadata_origin": metadata_origin}


def capture_model(store: Path, manifest_bytes: bytes, model: dict[str, Json]) -> dict[str, Json]:
    input_path = store / "families" / model["id"] / "inputs.json"
    inputs_bytes = input_path.read_bytes()
    inputs = json.loads(inputs_bytes)
    if inputs["schema"] != 1 or inputs["model"] != model["id"]:
        raise ValueError("Renderer input identity mismatch")
    assets = assets_for(store, model)
    identity = assets.get("encoder", assets.get("chat_template"))
    results: dict[str, Json] = {}
    for case in inputs["cases"]:
        if case["id"] in results:
            raise ValueError(f"Duplicate renderer case: {case['id']}")
        results[case["id"]] = reference_result(store, model, case)
    return {
        "schema": 1, "model": model["id"],
        "inputs_sha256": sha256(inputs_bytes), "manifest_sha256": sha256(manifest_bytes),
        "producer": {
            "implementation": "reference", "version": f"{model['repository']}@{model['revision']}",
            "source_sha256": sha256(identity.read_bytes()),
            "adapter_sha256": sha256(Path(__file__).read_bytes()),
            "python": sys.version.split()[0],
            "transformers": TRANSFORMERS_VERSION, "jinja2": JINJA_VERSION,
        },
        "results": results,
    }


def output_directory(root: Path, output: Path) -> Path:
    output = output.resolve()
    store = (root / STORE).resolve()
    if output == root.resolve() or output.is_relative_to(store) or store.is_relative_to(output):
        raise ValueError("Capture output must be separate from the source fixture store")
    return output


def atomic_json(path: Path, value: Json) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as temporary:
        temporary.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        name = temporary.name
    try:
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", required=True, choices=("reference", "dynamo"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--model")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    output = output_directory(root, arguments.output)
    store = root / STORE
    manifest_bytes = (store / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    models = [model for model in manifest["models"] if arguments.model is None or model["id"] == arguments.model]
    if not models:
        parser.error(f"Unknown renderer model: {arguments.model}")
    if arguments.implementation == "dynamo":
        command = ["cargo", "run", "--locked", "-q", "-p", "dynamo-conformance-fixtures-v2", "--example", "renderer_capture", "--", "--root", str(root), "--output", str(output)]
        if arguments.model:
            command += ["--model", arguments.model]
        subprocess.run(command, cwd=root, check=True)
        return
    for model in models:
        capture = capture_model(store, manifest_bytes, model)
        destination = output / "families" / model["id"] / "reference.json"
        atomic_json(destination, capture)
        print(f"{model['id']}: captured {len(capture['results'])} independent reference results")


if __name__ == "__main__":
    main()
