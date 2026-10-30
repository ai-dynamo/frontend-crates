# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independently authored schema probes shared by all parser conformance paths."""

import copy
import json
from pathlib import Path
from typing import Any, Callable

from impls import unified_grammar_family


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

# The custom type is intentionally outside JSON Schema. Keep these compatibility
# probes and their string controls separate from schema-valid CONFORMANCE_CASES.
XML_COMPATIBILITY_FAMILIES = ("qwen3_coder", "minimax_m2")
XML_COMPATIBILITY_CASES = [
    {
        "name": f"xml_{type_name}_type_{quotes}_quoted_object",
        "schema": {"type": type_name}, "raw": raw,
        "value": {"a": 1} if type_name == "custom" else raw,
        "strict": False,
    }
    for type_name in ("custom", "string")
    for quotes, raw in (("single", "{'a':1}"), ("double", '{"a":1}'))
]


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


# Report sections reuse the schema classification without changing fixture IDs.
ARGUMENT_SECTIONS = {
    "text": "Text preservation",
    "numeric": "TC Argument fidelity",
    "declared": "Declared types and enums",
    "nullable": "Type arrays and null interpretation",
    "const": "String constants",
    "union": "Unions, intersections, and ambiguous values",
    "reference": "References and schema IDs",
}
_SCHEMA_SECTION = {
    "declared": "declared", "enum": "declared", "nullable": "nullable",
    "const": "const", "anyof": "union", "oneof": "union",
    "intersection": "union", "ambiguous": "union", "reference": "reference",
}


def argument_section(scenario: str) -> str:
    scenario = scenario.removesuffix("_pr424")
    if scenario.startswith("xml_"):
        return "declared"
    if scenario.startswith("arg_numeric_"):
        return "numeric"
    if scenario.startswith("schema_"):
        return _SCHEMA_SECTION[schema_group(scenario.removeprefix("schema_"))]
    if scenario in {"arg_unicode", "arg_marker_in_string", "deepseek_v41_mixed_control_text_in_string",
                    "arg_object_quoted_parameter_close", "arg_array_quoted_parameter_close"}:
        return "text"
    if scenario.startswith("arg_"):
        return "nullable"
    return "reference"


def schema_fold(scenario: str) -> tuple[str, str] | None:
    """Existing fixture parent plus the child's display dimension, never a new ID."""
    name = scenario.removeprefix("schema_")
    if not scenario.startswith("schema_"):
        return None
    if schema_group(name) == "const":
        literal, form = name.removeprefix("const_").rsplit("_", 1)
        return f"schema_const_{literal}_bare", {
            "bare": "Direct", "typed": "Explicit string type",
            "allof": "allOf", "ref": "Reference",
        }[form]
    if schema_group(name) in {"anyof", "oneof"}:
        _, keyword, alternative, *dimensions = name.split("_")
        value = "Typed value" if dimensions[-1] == "typed" else "String"
        order = "Reversed" if "reverse" in dimensions else "Forward"
        return f"schema_union_{keyword}_{alternative}_string", f"{value} / {order}"
    return None


def schema_case_label(case: dict[str, Any]) -> str:
    """Legacy tool-calling label; Unified IDs are owned by unified_taxonomy."""
    if case["name"].startswith("xml_"):
        return f"7.x.{case['name']}"
    letter, _number, _description = SCHEMA_GROUPS[schema_group(case["name"])]
    return f"7.{letter}.{case['name']}"


def schema_tools(case: dict[str, Any]) -> list[dict[str, Any]]:
    parameters: dict[str, Any] = {
        "type": "object", "properties": {"value": copy.deepcopy(case["schema"])},
        "required": ["value"], "additionalProperties": False,
    }
    if case.get("defs"):
        parameters["$defs"] = copy.deepcopy(case["defs"])
    return [{"name": "schema_probe", "parameters": parameters, "strict": case.get("strict", True)}]


def schema_description(case: dict[str, Any]) -> str:
    if case["name"].startswith("xml_"):
        return (
            f"Native XML compatibility: {case['raw']} under type {case['schema']['type']} "
            f"must produce {json.dumps(case['value'])}. Custom types use legacy literal "
            "evaluation; an explicit string type preserves the original text."
        )
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
    family = unified_grammar_family(family)
    qwen = family in ("qwen3", "qwen3_coder", "nemotron_nano")
    # Beam's typing follows the model builder's reference: a union's branch types are
    # admitted together, so text that parses as the typed branch is that type.
    beam = family == "beam"
    json_union = case["name"] in {"integer_const_ambiguous", "object_const_union"}
    if ((qwen or beam or family in ("minimax_m2", "minimax_m3")) and json_union
            or (qwen or beam) and case["name"] == "type_array_string_integer"):
        return {"value": json.loads(case["raw"])}
    return {"value": copy.deepcopy(case["value"])}


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def native_call_wire(family: str, name: str, values: dict[str, Any], raw_values: dict[str, str], renderer: Callable[..., str]) -> str:
    """Render every registered family's native grammar, never parser output."""
    arguments = _json(values)
    call = _json({"name": name, "arguments": values})
    if family == "deepseek_v3":
        return f"<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>{name}\n```json\n{arguments}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
    if family == "deepseek_v3_1":
        return f"<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>{name}<｜tool▁sep｜>{arguments}<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
    if family in ("deepseek_v3_2", "deepseek_v4"):
        envelope = "function_calls" if family == "deepseek_v3_2" else "tool_calls"
        params = "".join(f'<｜DSML｜parameter name="{key}" string="{str(isinstance(value, str)).lower()}">{raw_values[key]}</｜DSML｜parameter>' for key, value in values.items())
        return f'<｜DSML｜{envelope}><｜DSML｜invoke name="{name}">{params}</｜DSML｜invoke></｜DSML｜{envelope}>'
    if family in ("harmony", "harmony_text"):
        return f"<|channel|>commentary to=functions.{name} <|constrain|>json<|message|>{arguments}<|call|>"
    if family in ("hermes", "qwen25"):
        return f"<tool_call>{call}</tool_call>"
    if family == "inkling":
        inner = _json({"name": name, "args": values})
        return f"<|message_model|>{name}<|content_invoke_tool_json|>{inner}<|end_message|>"
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
        return f"[{name}({', '.join(f'{key}={value!r}' for key, value in values.items())})]"
    if family in ("qwen3_coder", "nemotron_nano"):
        params = "\n".join(f"<parameter={key}>\n{value}\n</parameter>" for key, value in raw_values.items())
        return f"<tool_call>\n<function={name}>\n{params}\n</function>\n</tool_call>"
    if family == "minimax_m2":
        params = "".join(f'<parameter name="{key}">{value}</parameter>' for key, value in raw_values.items())
        return f'<minimax:tool_call><invoke name="{name}">{params}</invoke></minimax:tool_call>'
    if family == "minimax_m3":
        marker = "]<]minimax[>["
        params = "".join(f"{marker}<{key}>{value}{marker}</{key}>" for key, value in raw_values.items())
        return f'{marker}<tool_call>{marker}<invoke name="{name}">{params}{marker}</invoke>{marker}</tool_call>'
    # These grammars have an existing independently authored Unified renderer.
    if family in ("glm47", "gemma4", "kimi_k2", "kimi_k3", "muse_glimmer", "muse_spark"):
        return renderer(family, name, values, 0, raw_values)
    raise ValueError(f"no schema grammar renderer for registered family {family}")


def legacy_schema_wire(family: str, case: dict[str, Any], renderer: Callable[..., str]) -> str:
    return native_call_wire(family, "schema_probe", {"value": case["value"]}, {"value": case["raw"]}, renderer)


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
