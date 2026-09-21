# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import gen_unified_golden as G
from unified_tools import unified_tools
from unified_taxonomy import numbered_id


def _assert_value(value, schema):
    # This corpus declares only these JSON Schema keywords; fail on additions so
    # a new constraint cannot silently bypass this producer-side check.
    assert schema.keys() <= {"type", "properties", "items"}, schema
    kind = schema["type"]
    if kind == "object":
        assert isinstance(value, dict), value
        for key, item in value.items():
            if key in schema["properties"]:
                _assert_value(item, schema["properties"][key])
    elif kind == "array":
        assert isinstance(value, list), value
        for item in value:
            _assert_value(item, schema["items"])
    elif kind == "number":
        assert type(value) in (int, float), value
    else:
        assert kind == "string", schema
        assert isinstance(value, str), value


def _assert_golden_schemas(cases, tools):
    schemas = {tool["name"]: tool["parameters"] for tool in tools}
    assert len(schemas) == len(tools)
    for case in cases.values():
        for event in case["golden"]:
            if event["kind"] == "tool_call":
                _assert_value(event["arguments"], schemas[event["name"]])


@pytest.mark.parametrize("family", G.FAMILIES)
def test_all_authored_successful_calls_match_offered_schemas(family):
    _assert_golden_schemas(G.build_cases(family), unified_tools())


@pytest.mark.parametrize("family", G.FAMILIES)
def test_schema_guard_rejects_old_array_as_string_declaration(family):
    tools = unified_tools()
    next(tool for tool in tools if tool["name"] == "sum_values")["parameters"]["properties"]["values"] = {"type": "string"}
    with pytest.raises(AssertionError):
        _assert_golden_schemas(G.build_cases(family), tools)


def test_schema_guard_rejects_old_numeric_string_successor():
    cases = copy.deepcopy(G.build_cases("kimi_k3"))
    cases["UNIFIED.kimi_k3_malformed_call_then_valid.kimi_k3"]["golden"][0]["arguments"]["y"] = 2
    with pytest.raises(AssertionError):
        _assert_golden_schemas(cases, unified_tools())


def test_string_arguments_and_open_additional_properties_are_preserved():
    schemas = {tool["name"]: tool["parameters"] for tool in unified_tools()}
    assert set(schemas) == {"get_weather", "f", "g", "run", "log", "sum_values", "functions."}
    for name, key in (("get_weather", "city"), ("f", "x"), ("g", "y"), ("run", "cmd"), ("log", "note")):
        assert schemas[name] == {"type": "object", "properties": {key: {"type": "string"}}}
    assert schemas["sum_values"]["properties"]["values"] == {"type": "array", "items": {"type": "number"}}
    assert schemas["functions."] == {"type": "object", "properties": {}}
    _assert_value({"cmd": "echo ok", "count": 2, "force": True}, schemas["run"])
    _assert_value({}, schemas["get_weather"])


@pytest.mark.parametrize("script", ["capture_vllm_unified.py", "capture_sglang_unified.py"])
def test_peer_request_schema_projection_matches_shared_definition(script):
    tree = ast.parse((SRC / script).read_text())
    assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id in {"TOOLS", "TOOL_SCHEMAS"} for target in node.targets)]
    namespace = {"unified_tools": unified_tools, "Tool": SimpleNamespace, "Function": SimpleNamespace}
    exec(compile(ast.Module(body=assignments, type_ignores=[]), script, "exec"), namespace)
    projected = namespace["TOOLS"]
    if script == "capture_sglang_unified.py":
        actual = [vars(tool.function) for tool in projected]
        assert all(tool.type == "function" for tool in projected)
    else:
        actual = [tool["function"] for tool in projected]
        assert all(tool["type"] == "function" for tool in projected)
    assert actual == unified_tools()


def test_rust_harnesses_consume_the_shared_schema_owner():
    tests = SRC.parents[1] / "tests"
    common = (tests / "common/mod.rs").read_text()
    assert 'include_str!("../../utils/src/unified_tools.json")' in common
    assert 'serde_json::from_value(unified_tool_schemas())' in common
    assert '"tools": common::unified_tool_schemas()' in (tests / "unified_render.rs").read_text()
    for name in ("unified_render.rs", "unified_parity.rs", "capture_cross_version.rs"):
        source = (tests / name).read_text()
        assert "unified_tools as tools" in source
        assert "fn tools()" not in source
    peer = (SRC / "capture_vllm_rust_unified.py").read_text()
    assert 'serde_json::from_str(include_str!("unified_tools.json"))' in peer
    assert '(crate / "src/unified_tools.json").write_bytes(SCHEMA_PATH.read_bytes())' in peer


def test_recovery_successor_respects_string_schema():
    case = G.build_cases("kimi_k3")["UNIFIED.kimi_k3_malformed_call_then_valid.kimi_k3"]
    assert case["golden"] == [{"kind": "tool_call", "name": "g", "arguments": {"y": "2"}}]
    assert G.k3_argument("y", "string", "2") in case["input"]
    assert G.k3_open("call", [("tool", "bad"), ("index", "1")]) + "not-an-argument" in case["input"]


def test_optional_prefix_name_overlap_is_an_explicit_native_declared_tool_contract():
    scenario = "kimi_k2_optional_prefix_name_overlap"
    case = G.build_cases("kimi_k2")[f"UNIFIED.{scenario}.kimi_k2"]
    assert numbered_id(scenario) == "UNIFIED.kimi_k2-1"
    assert G.scenario_families(scenario) == {"kimi_k2"}
    assert case["input"] == (
        "<|tool_calls_section_begin|><|tool_call_begin|>functions.:17"
        "<|tool_call_argument_begin|>{}<|tool_call_end|><|tool_calls_section_end|>"
    )
    assert case["golden"] == [{"kind": "tool_call", "name": "functions.", "arguments": {}}]
    assert case["init"] == {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None}
    assert "functions." in {tool["name"] for tool in unified_tools()}


def test_prefix_overlap_case_is_rejected_without_its_declared_schema():
    case_id = "UNIFIED.kimi_k2_optional_prefix_name_overlap.kimi_k2"
    case = G.build_cases("kimi_k2")[case_id]
    tools = [tool for tool in unified_tools() if tool["name"] != "functions."]
    with pytest.raises(KeyError, match="functions"):
        _assert_golden_schemas({case_id: case}, tools)
