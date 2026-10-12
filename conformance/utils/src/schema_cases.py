# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independently authored schema probes shared by all parser conformance paths."""

import copy
import json
from pathlib import Path
from typing import Any, Callable


SCHEMA_PATH = Path(__file__).with_name("glm47_schema_cases.json")
SCHEMA_CASES: list[dict[str, Any]] = json.loads(SCHEMA_PATH.read_text())
# Distinct Rust controls can describe the same authored conformance behavior.
# Keep one immutable case identity per (schema, native spelling, oracle value).
CONFORMANCE_CASES = list({
    json.dumps({key: case.get(key) for key in ("schema", "defs", "value", "raw")}, sort_keys=True): case
    for case in reversed(SCHEMA_CASES) if case.get("conformance", True)
}.values())[::-1]
SCHEMA_GROUPS = {
    "declared": ("o", "17", "Declared JSON argument types"),
    "const": ("p", "18", "String constants retain their literal spelling"),
    "anyof": ("q", "19", "anyOf constant and typed alternatives in both orders"),
    "oneof": ("r", "20", "oneOf constant and typed alternatives in both orders"),
    "reference": ("s", "21", "Local references and sibling type intersections"),
    "intersection": ("t", "22", "Composition intersects union type hints"),
    "enum": ("u", "23", "Enums preserve declared argument types"),
    "nullable": ("v", "24", "Type arrays and nullable argument interpretation"),
    "ambiguous": ("w", "25", "Ambiguous native strings retain established preference"),
}


def schema_group(name: str) -> str:
    if name.startswith("declared_") or name == "id_direct_integer":
        return "declared"
    if name.startswith("const_"):
        return "const"
    if name.startswith("union_anyOf_"):
        return "anyof"
    if name.startswith("union_oneOf_"):
        return "oneof"
    if name.startswith(("ref_", "escaped_ref_", "percent_ref_")):
        return "reference"
    if name.startswith("intersection_"):
        return "intersection"
    if name.startswith("enum_"):
        return "enum"
    if name.startswith(("type_array_", "nullable_")) or name == "null_string_literal":
        return "nullable"
    return "ambiguous"


def schema_case_label(case: dict[str, Any], unified: bool = False) -> str:
    letter, number, _description = SCHEMA_GROUPS[schema_group(case["name"])]
    return f"7-{number}.{case['name']}" if unified else f"7.{letter}.{case['name']}"


def schema_tools(case: dict[str, Any]) -> list[dict[str, Any]]:
    parameters: dict[str, Any] = {
        "type": "object", "properties": {"value": copy.deepcopy(case["schema"])},
        "required": ["value"], "additionalProperties": False,
    }
    if case.get("defs"):
        parameters["$defs"] = copy.deepcopy(case["defs"])
    return [{"name": "schema_probe", "parameters": parameters, "strict": True}]


def schema_description(case: dict[str, Any]) -> str:
    return (
        f"Schema probe {case['name']}: independently authored value "
        f"{json.dumps(case['value'], ensure_ascii=False)} under "
        f"{json.dumps(case['schema'], ensure_ascii=False)}. Native grammar "
        "preserves the intended JSON type; schema-driven grammars must infer it."
    )


def schema_arguments(family: str, case: dict[str, Any]) -> dict[str, Any]:
    # Qwen and MiniMax prefer parsed JSON for these ambiguous constant unions.
    # GLM prefers a matching string hint. Both satisfy these schemas;
    # explicitly typed and JSON grammars carry the authored type on the wire.
    qwen = family in ("qwen3", "qwen3_coder", "nemotron_nano")
    json_union = case["name"] in {"integer_const_ambiguous", "object_const_union"}
    if ((qwen or family in ("minimax_m2", "minimax_m3")) and json_union
            or qwen and case["name"] == "type_array_string_integer"):
        return {"value": json.loads(case["raw"])}
    return {"value": copy.deepcopy(case["value"])}


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def legacy_schema_wire(family: str, case: dict[str, Any], renderer: Callable[..., str]) -> str:
    """Render every registered family's native grammar, never parser output."""
    value, raw = case["value"], case["raw"]
    arguments = _json({"value": value})
    call = _json({"name": "schema_probe", "arguments": {"value": value}})
    if family == "deepseek_v3":
        return f"<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>schema_probe\n```json\n{arguments}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
    if family == "deepseek_v3_1":
        return f"<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>schema_probe<｜tool▁sep｜>{arguments}<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
    if family in ("deepseek_v3_2", "deepseek_v4"):
        envelope = "function_calls" if family == "deepseek_v3_2" else "tool_calls"
        flag = str(isinstance(value, str)).lower()
        return f'<｜DSML｜{envelope}><｜DSML｜invoke name="schema_probe"><｜DSML｜parameter name="value" string="{flag}">{raw}</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜{envelope}>'
    if family in ("harmony", "harmony_text"):
        return f"<|channel|>commentary to=functions.schema_probe <|constrain|>json<|message|>{arguments}<|call|>"
    if family in ("hermes", "qwen25"):
        return f"<tool_call>{call}</tool_call>"
    if family == "inkling":
        inner = _json({"name": "schema_probe", "args": {"value": value}})
        return f"<|message_model|>schema_probe<|content_invoke_tool_json|>{inner}<|end_message|>"
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
        return f"[schema_probe(value={value!r})]"
    if family in ("qwen3_coder", "nemotron_nano"):
        return f"<tool_call>\n<function=schema_probe>\n<parameter=value>\n{raw}\n</parameter>\n</function>\n</tool_call>"
    if family == "minimax_m2":
        return f'<minimax:tool_call><invoke name="schema_probe"><parameter name="value">{raw}</parameter></invoke></minimax:tool_call>'
    if family == "minimax_m3":
        marker = "]<]minimax[>["
        return f'{marker}<tool_call>{marker}<invoke name="schema_probe">{marker}<value>{raw}{marker}</value>{marker}</invoke>{marker}</tool_call>'
    # These grammars have an existing independently authored Unified renderer.
    if family in ("glm47", "gemma4", "kimi_k2", "kimi_k3", "muse_glimmer"):
        return renderer(family, "schema_probe", {"value": value}, 0, {"value": raw})
    raise ValueError(f"no schema grammar renderer for registered family {family}")


def unified_schema_cases(family: str, renderer: Callable[..., str]) -> dict[str, dict[str, Any]]:
    cases = {}
    for probe in CONFORMANCE_CASES:
        name = f"schema_{probe['name']}"
        value, raw = probe["value"], probe["raw"]
        wire = renderer(family, "schema_probe", {"value": value}, 0, {"value": raw})
        cases[f"UNIFIED.{name}.{family}"] = {
            "description": schema_description(probe), "policy": ["I7"],
            "input": wire,
            "golden": [{"kind": "tool_call", "name": "schema_probe", "arguments": schema_arguments(family, probe)}],
            "tools": schema_tools(probe),
            "init": {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None},
            "finish_reason": "stop",
            "expect": {
                "vllm": {"verdict": "diverge", "class": "UNSUPPORTED", "note": "No peer capture recorded for these shared schema probes."},
                "dynamo": {"verdict": "match"},
            },
        }
    return cases
