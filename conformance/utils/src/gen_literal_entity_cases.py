#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Author TOOLCALLING 7.y: literal entity text in arguments, for every registered family."""

import json
from pathlib import Path
from typing import Any

import yaml

import gen_unified_golden as golden
from refresh_dynamo_captures import ensure_tree

REGISTRY = yaml.safe_load(Path(__file__).with_name("parser_families.yaml").read_text())["families"]
HEADER = "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n# SPDX-License-Identifier: Apache-2.0\n"
NAME = "inspect"
# Entity spellings are argument data: chat templates render them verbatim, so
# no parser may decode them. `&amp;quot;` catches a second decoding pass.
ARGUMENTS: dict[str, Any] = {
    "text": "a &amp; b &lt;div&gt; &amp;quot;",
    "payload": {"text": "&quot; &amp;"},
    "note": "&lt;b&gt; &#39;x&#39;",
}
TOOLS = [{
    "name": NAME,
    "parameters": {
        "type": "object",
        # `note` is deliberately undeclared, so schema-less conversion is covered.
        "properties": {
            "text": {"type": "string"},
            "payload": {"type": "object", "properties": {"text": {"type": "string"}}},
        },
    },
}]
DESCRIPTION = (
    "Literal entity text in argument values is data. A string, an object member, and an "
    "undeclared argument keep &amp;, &lt;, &gt;, &quot;, &#39;, and the double-encoded "
    "&amp;quot; exactly as written. Constructed native probe; not captured model output."
)


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _raw(value: object) -> str:
    return value if isinstance(value, str) else _json(value)


def wire(family: str) -> str:
    """Render the call in the family's native grammar, mirroring legacy_schema_wire."""
    arguments = _json(ARGUMENTS)
    call = _json({"name": NAME, "arguments": ARGUMENTS})
    if family == "deepseek_v3":
        return f"<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>{NAME}\n```json\n{arguments}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
    if family == "deepseek_v3_1":
        return f"<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>{NAME}<｜tool▁sep｜>{arguments}<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
    if family in ("deepseek_v3_2", "deepseek_v4"):
        envelope = "function_calls" if family == "deepseek_v3_2" else "tool_calls"
        params = "".join(
            f'<｜DSML｜parameter name="{key}" string="{str(isinstance(value, str)).lower()}">{_raw(value)}</｜DSML｜parameter>'
            for key, value in ARGUMENTS.items()
        )
        return f'<｜DSML｜{envelope}><｜DSML｜invoke name="{NAME}">{params}</｜DSML｜invoke></｜DSML｜{envelope}>'
    if family in ("harmony", "harmony_text"):
        return f"<|channel|>commentary to=functions.{NAME} <|constrain|>json<|message|>{arguments}<|call|>"
    if family in ("hermes", "qwen25"):
        return f"<tool_call>{call}</tool_call>"
    if family == "inkling":
        inner = _json({"name": NAME, "args": ARGUMENTS})
        return f"<|message_model|>{NAME}<|content_invoke_tool_json|>{inner}<|end_message|>"
    if family == "jamba":
        return f"<tool_calls>[{call}]</tool_calls>"
    if family == "llama3_json":
        return f"<|python_tag|>{call}"
    if family == "mistral":
        return f"[TOOL_CALLS][{call}][/TOOL_CALLS]"
    if family == "nemotron_deci":
        return f"<TOOLCALL>[{call}]</TOOLCALL>"
    if family == "phi4":
        return f"functools[{call}]"
    if family == "pythonic":
        params = ", ".join(f"{key}={value!r}" for key, value in ARGUMENTS.items())
        return f"[{NAME}({params})]"
    if family in ("qwen3_coder", "nemotron_nano"):
        params = "".join(f"<parameter={key}>\n{_raw(value)}\n</parameter>\n" for key, value in ARGUMENTS.items())
        return f"<tool_call>\n<function={NAME}>\n{params}</function>\n</tool_call>"
    if family == "minimax_m2":
        params = "".join(f'<parameter name="{key}">{_raw(value)}</parameter>' for key, value in ARGUMENTS.items())
        return f'<minimax:tool_call><invoke name="{NAME}">{params}</invoke></minimax:tool_call>'
    if family == "minimax_m3":
        # The template renders objects as nested elements and leaves verbatim.
        marker = "]<]minimax[>["

        def element(key: str, value: Any) -> str:
            body = "".join(element(k, v) for k, v in value.items()) if isinstance(value, dict) else value
            return f"{marker}<{key}>{body}{marker}</{key}>"

        params = "".join(element(key, value) for key, value in ARGUMENTS.items())
        return f'{marker}<tool_call>{marker}<invoke name="{NAME}">{params}{marker}</invoke>{marker}</tool_call>'
    # These grammars have an existing independently authored Unified renderer.
    if family in ("glm47", "gemma4", "kimi_k2", "kimi_k3", "muse_glimmer"):
        raw = {key: _raw(value) for key, value in ARGUMENTS.items()}
        return golden.r_tool_arguments(family, NAME, ARGUMENTS, 0, raw)
    raise ValueError(f"no literal-entity renderer for registered family {family}")


def authored_case(family: str, stream: bool) -> dict[str, dict[str, Any]]:
    text = wire(family)
    case: dict[str, Any] = {
        "description": DESCRIPTION,
        "tools": TOOLS,
        # This is authored independently; recorded implementation output never owns it.
        "golden": {"calls": [{"name": NAME, "arguments": ARGUMENTS}], "normal_text": ""},
    }
    if stream:
        # Three-byte chunks split entities such as `&am|p;` across deltas.
        case["chunks"] = [{"delta_text": text[offset:offset + 3]} for offset in range(0, len(text), 3)]
        case["chunks"][-1]["finish_reason"] = "stop"
        if not REGISTRY[family].get("dynamo_v2"):
            case["unavailable"] = {"dynamo_v2": "No registered Dynamo v2 parser for this family."}
        return {"TOOLCALLING.streamv1.7.y": case}
    case["model_text"] = text
    return {"TOOLCALLING.batch.7.y": case}


def write(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(HEADER + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=100000))


def main() -> None:
    batch = ensure_tree("fixtures-batch-v1") / "inputs"
    stream = ensure_tree("fixtures-stream-v1") / "inputs"
    for family in sorted(REGISTRY):
        # Same coverage as gen_schema_fixtures: Harmony token fixtures need real
        # token IDs, so the text registration carries the stream case instead.
        if family not in ("harmony_text", "muse_glimmer"):
            doc = {"family": family, "mode": "batch", "cases": authored_case(family, False)}
            write(batch / family / "TOOLCALLING.batch.7.entities.yaml", doc)
        if family != "harmony":
            doc = {"family": family, "mode": "streamv1", "cases": authored_case(family, True)}
            write(stream / family / "TOOLCALLING.streamv1.7.entities.yaml", doc)
    print(f"Authored TOOLCALLING 7.y for {len(REGISTRY)} registered families.")


if __name__ == "__main__":
    main()
