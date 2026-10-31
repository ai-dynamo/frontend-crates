# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import copy
import json
from decimal import Decimal
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from conformance.utils.tests.schema_oracle import matches_schema
from conformance.utils.tests.fixture_test_support import GPT58_NATIVE_REQUEST_DIGESTS, assert_gpt58_native_contract

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import gen_unified_golden as G
from unified_tools import openai_tools, unified_tools
from schema_cases import XML_COMPATIBILITY_CASES, schema_tools


def _assert_value(value, schema):
    assert matches_schema(value, schema), (value, schema)


@pytest.mark.parametrize("kind, valid, invalid", [
    ("integer", [0, -2, 3, 2.0], [True, False, 1.5, "2", [], {}]),
    ("number", [0, -2, 1.5], [True, False, "2", [], {}]),
    ("string", ["", "text"], [0, 1.5, True, [], {}]),
    ("boolean", [True, False], [0, 1, "true", [], {}]),
    ("object", [{}, {"x": 2}], [0, "", True, []]),
    ("array", [[], [2, None]], [0, "", True, {}]),
])
@pytest.mark.parametrize("nullable", [False, True])
def test_schema_guard_valid_and_invalid_values(kind, valid, invalid, nullable):
    schema = {"type": [kind, "null"] if nullable else kind}
    if kind == "object":
        schema["properties"] = {"x": {"type": "integer"}}
    elif kind == "array":
        schema["items"] = {"type": ["integer", "null"]}
    for value in valid + ([None] if nullable else []):
        _assert_value(value, schema)
    for value in invalid + ([] if nullable else [None]):
        with pytest.raises(AssertionError):
            _assert_value(value, schema)


def test_schema_guard_null_type():
    _assert_value(None, {"type": "null"})
    for value in (False, 0, "", [], {}):
        with pytest.raises(AssertionError):
            _assert_value(value, {"type": "null"})


@pytest.mark.parametrize("value, schema", [
    ({"x": True}, {"type": ["null", "object"], "properties": {"x": {"type": "integer"}}}),
    ([1.5], {"type": ["null", "array"], "items": {"type": "integer"}}),
    (None, {"type": ["null", "unsupported"]}),
    (None, {"type": ["null", "integer"], "minimum": 0}),
])
def test_schema_guard_rejects_nested_values_and_unsupported_constraints(value, schema):
    with pytest.raises(AssertionError):
        _assert_value(value, schema)


@pytest.mark.parametrize("schema, valid, invalid", [
    ({"anyOf": [{"type": "string"}, {"type": "null"}]}, ["text", None], [1, False, []]),
    ({"oneOf": [{"type": "integer"}, {"type": "null"}]}, [2, 2.0, None], [1.5, True, "2"]),
    ({"oneOf": [{"type": "number"}, {"type": "integer"}]}, [1.5], [2, 2.0, None]),
    ({"const": None}, [None], ["null", 0, False]),
    ({"enum": [None, "text"]}, [None, "text"], ["other", 0, False]),
    ({"type": "integer", "nullable": True}, [2, None], [1.5, True, "2"]),
    ({"type": "string", "nullable": False}, ["text"], [None, 2]),
    ({"type": ["string", "null"], "minLength": 2}, ["ok", None], ["", "x", 2]),
])
def test_schema_guard_corpus_keywords(schema, valid, invalid):
    # Exercise the same nested argument path as authored tool calls.
    tool_schema = {"type": "object", "properties": {"value": schema}}
    for value in valid:
        _assert_value({"value": value}, tool_schema)
    for value in invalid:
        with pytest.raises(AssertionError):
            _assert_value({"value": value}, tool_schema)


def _assert_golden_schemas(cases, tools):
    for case_id, case in cases.items():
        scenario = case_id.removeprefix("UNIFIED.").rsplit(".", 1)[0]
        family = case_id.rsplit(".", 1)[1]
        custom_probe = next((probe for probe in XML_COMPATIBILITY_CASES
                             if probe["schema"]["type"] == "custom" and probe["name"] == scenario), None)
        if custom_probe is not None:
            # These exact native compatibility requests deliberately use an invalid
            # JSON Schema type. Preserve the strict schema oracle for every other case.
            allowed = ({"qwen3", "nemotron_v3", "glm47", "muse_glimmer", "muse_spark"}
                       if scenario == "xml_custom_type_double_quoted_object"
                       else {"qwen3", "nemotron_v3"})
            assert case_id.rsplit(".", 1)[1] in allowed
            assert case["tools"] == schema_tools(custom_probe)
            assert case["golden"] == [{"kind": "tool_call", "name": "schema_probe",
                                       "arguments": {"value": {"a": 1}}}]
            continue
        if (scenario, family) in GPT58_NATIVE_REQUEST_DIGESTS:
            assert_gpt58_native_contract(family, scenario, case)
        offered_tools = case.get("tools", tools)
        if not offered_tools and scenario in {"gpt_oss_control_marker_inside_argument_string", "gpt_oss_quoted_message_header_literal", "gpt_oss_quoted_truncated_tool_then_final"}:
            allowed = {"gpt_oss", "beam"}
            if scenario == "gpt_oss_control_marker_inside_argument_string":
                allowed.update({"deepseek_v4", "deepseek_v41", "kimi_k2", "kimi_k3", "muse_glimmer", "muse_spark"})
            assert family in allowed, (family, scenario, "schema-free native counterpart")
        schemas = {tool["name"]: tool["parameters"] for tool in offered_tools}
        assert len(schemas) == len(offered_tools)
        for event in case["golden"]:
            if event["kind"] == "tool_call":
                if scenario == "kimi_k3_malformed_call_then_valid":
                    assert family != "glm47"
                    assert offered_tools == unified_tools()
                    assert case["init"] == {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None}
                    # The first invocation retains ownership in these native
                    # grammars; this is deliberate malformed recovery, not a
                    # successful offered-tool/schema assertion.
                    if family in {"deepseek_v4", "qwen3", "nemotron_v3", "muse_glimmer", "muse_spark"}:
                        expected = [{"kind": "tool_call", "name": "bad", "arguments": {"y": "2"}}]
                    elif family == "minimax_m3":
                        expected = [{"kind": "tool_call", "name": "bad", "arguments": {}}]
                    else:
                        expected = [{"kind": "tool_call", "name": "g", "arguments": {"y": "2"}}]
                    assert case["golden"] == expected
                    continue
                if not offered_tools:
                    if (scenario, family) in GPT58_NATIVE_REQUEST_DIGESTS:
                        assert_gpt58_native_contract(family, scenario, case)
                        continue
                    if scenario == "gpt_oss_channel_header_variants":
                        assert family in {"beam", "gpt_oss"}
                        assert case["init"] == {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None}
                        assert case["golden"] == [
                            {"kind": "reasoning", "text": "check"},
                            {"kind": "tool_call", "name": "lookup", "arguments": {"q": "weather"}},
                            {"kind": "text", "text": "found"},
                        ]
                        continue
                    if scenario in {"gpt_oss_implicit_recipient_first_header", "gpt_oss_complete_alternate_message_ends"}:
                        allowed = {"gpt_oss", "muse_glimmer", "muse_spark"}
                        if scenario == "gpt_oss_implicit_recipient_first_header":
                            allowed.add("beam")
                        assert family in allowed
                        assert case["init"] == {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None}
                        values = ["x"] if scenario == "gpt_oss_implicit_recipient_first_header" else ["end", "return"]
                        assert case["golden"] == [{"kind": "tool_call", "name": "lookup", "arguments": {"q": value}} for value in values]
                        continue
                    if scenario in {"guided_json_quoted_bare_header_in_answer_pr424", "guided_json_quoted_bare_header_after_payload_pr424"} and family in {"muse_glimmer", "muse_spark"}:
                        assert case["init"] == {"starting_state": "Response", "tool_output_mode": "GuidedJson", "named_tool": None}
                        text = {"kind": "text", "text": "I mean assistant to=selfself literal"}
                        call = {"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}}
                        assert case["golden"] == ([call, text] if "after_payload" in scenario else [text, call])
                        continue
                    # These lexical probes intentionally have no offered schema;
                    # successful calls elsewhere still pass the strict oracle.
                    assert family == "gpt_oss" or (
                        scenario in {
                            "gpt_oss_multiple_unknown_and_empty_tools",
                            "gpt_oss_control_marker_inside_argument_string",
                            "gpt_oss_invalid_tool_body_then_empty_call",
                            "gpt_oss_invalid_tool_body_then_valid_call",
                            "gpt_oss_truncated_tool_resynchronizes_at_next_message",
                            "gpt_oss_interrupted_reasoning_then_response",
                        }
                        and case["init"]["tool_output_mode"] == "Native"
                    )
                    continue
                if family == "minimax_m3" and scenario == "kimi_k3_guided_native_wrapper":
                    assert offered_tools == unified_tools()
                    assert case["init"]["tool_output_mode"] == "GuidedJson"
                    assert case["golden"] == [
                        {"kind": "tool_call", "name": "ignored", "arguments": {"quoted": "literal"}},
                        {"kind": "text", "text": '[{"name": "get_weather", "arguments": {"city": "Paris"}}]'},
                    ]
                    continue
                if _malformed_native_empty_recovery(family, scenario, case, event):
                    # These exact malformed-input probes deliberately complete
                    # an empty call; successful-call schemas stay strict elsewhere.
                    assert event == {"kind": "tool_call", "name": "write_file", "arguments": {}}
                    continue
                if case_id.startswith("UNIFIED.malformed_json_then_two_valid_calls.") and event["name"] == "bad":
                    grammar_family = G.unified_grammar_family(family)
                    assert grammar_family in {"kimi_k2", "qwen3", "glm47", "deepseek_v4", "muse_glimmer", "muse_spark", "gpt_oss", "beam", "minimax_m3"}
                    assert event["arguments"] == ({} if grammar_family == "kimi_k2" else {"value": '{"x":"unfinished'})
                    continue
                arguments = event["arguments"]
                if isinstance(arguments, str):
                    arguments = json.loads(arguments, parse_float=Decimal)
                assert event["name"] in schemas, (case_id, "successful call must name an offered tool")
                _assert_value(arguments, schemas[event["name"]])


@pytest.mark.parametrize("family,scenario", [
    ("beam", "gpt_oss_channel_header_variants"),
    ("gpt_oss", "gpt_oss_channel_header_variants"),
    ("beam", "gpt_oss_implicit_recipient_first_header"),

    ("muse_glimmer", "gpt_oss_implicit_recipient_first_header"),
    ("muse_spark", "gpt_oss_complete_alternate_message_ends"),
    ("muse_glimmer", "guided_json_quoted_bare_header_in_answer_pr424"),
    ("muse_spark", "guided_json_quoted_bare_header_after_payload_pr424"),
    ("minimax_m3", "kimi_k3_guided_native_wrapper"),
])
@pytest.mark.parametrize("mutation", ["name", "arguments", "mode", "family"])
def test_lexical_schema_exceptions_reject_changed_contracts(family, scenario, mutation):
    case_id = f"UNIFIED.{scenario}.{family}"
    case = copy.deepcopy(G.build_cases(family)[case_id])
    _assert_golden_schemas({case_id: case}, unified_tools())
    call = next(event for event in case["golden"] if event["kind"] == "tool_call")
    if mutation == "name":
        call["name"] = "invented"
    elif mutation == "arguments":
        call["arguments"] = {"unexpected": True}
    elif mutation == "mode":
        case["init"]["tool_output_mode"] = "Native" if case["init"]["tool_output_mode"] == "GuidedJson" else "GuidedJson"
    else:
        case_id = case_id.rsplit(".", 1)[0] + ".qwen3"
    with pytest.raises(AssertionError):
        _assert_golden_schemas({case_id: case}, unified_tools())


@pytest.mark.parametrize("family", ["qwen3", "gpt_oss"])
def test_schema_free_text_does_not_imply_a_schema_free_tool_call(family):
    case = {"tools": [], "golden": [
        {"kind": "reasoning", "text": "analysis"},
        {"kind": "text", "text": "response"},
    ]}
    cases = {f"UNIFIED.no_tools.{family}": case}
    _assert_golden_schemas(cases, unified_tools())
    case["golden"].append({"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}})
    if family == "gpt_oss":
        _assert_golden_schemas(cases, unified_tools())
    else:
        with pytest.raises(AssertionError):
            _assert_golden_schemas(cases, unified_tools())



@pytest.mark.parametrize("family", ["deepseek_v4", "qwen3", "nemotron_v3", "muse_glimmer", "muse_spark", "minimax_m3"])
@pytest.mark.parametrize("mutation", ["name", "arguments", "mode", "schema", "scope"])
def test_native_unclosed_body_schema_exception_is_exact(family, mutation):
    scenario = "kimi_k3_malformed_call_then_valid"
    case_id = f"UNIFIED.{scenario}.{family}"
    case = copy.deepcopy(G.build_cases(family)[case_id])
    _assert_golden_schemas({case_id: case}, unified_tools())
    if mutation == "name":
        case["golden"][0]["name"] = "invented"
    elif mutation == "arguments":
        case["golden"][0]["arguments"] = {"unexpected": True}
    elif mutation == "mode":
        case["init"]["tool_output_mode"] = "GuidedJson"
    elif mutation == "schema":
        case["tools"] = []
    else:
        case_id = case_id.rsplit(".", 1)[0] + ".glm47"
    with pytest.raises(AssertionError):
        _assert_golden_schemas({case_id: case}, unified_tools())


def test_schema_free_lexical_exception_requires_its_scenario_and_native_mode():
    case_id = "UNIFIED.gpt_oss_multiple_unknown_and_empty_tools.qwen3"
    case = copy.deepcopy(G.build_cases("qwen3")[case_id])
    _assert_golden_schemas({case_id: case}, unified_tools())
    with pytest.raises(AssertionError):
        _assert_golden_schemas({"UNIFIED.tool_only.qwen3": case}, unified_tools())
    case["init"]["tool_output_mode"] = "GuidedJson"
    with pytest.raises(AssertionError):
        _assert_golden_schemas({case_id: case}, unified_tools())


@pytest.mark.parametrize("family,quotes", [
    ("qwen3", "single"), ("qwen3", "double"), ("nemotron_v3", "double"),
    ("glm47", "double"), ("muse_glimmer", "double"), ("muse_spark", "double"),
])
@pytest.mark.parametrize("mutation", ["arguments", "schema", "family"])
def test_custom_compatibility_exclusion_rejects_changed_contracts(mutation, family, quotes):
    case_id = f"UNIFIED.xml_custom_type_{quotes}_quoted_object.{family}"
    case = copy.deepcopy(G.build_cases(family)[case_id])
    _assert_golden_schemas({case_id: case}, unified_tools())
    with pytest.raises(AssertionError):
        _assert_value(case["golden"][0]["arguments"], case["tools"][0]["parameters"])
    if mutation == "arguments":
        case["golden"][0]["arguments"]["value"] = "{'a':1}"
    elif mutation == "schema":
        case["tools"][0]["parameters"]["properties"]["value"] = {"type": "object"}
    else:
        case_id = case_id.rsplit(".", 1)[0] + ".beam"
    with pytest.raises(AssertionError):
        _assert_golden_schemas({case_id: case}, unified_tools())


@pytest.mark.parametrize("family", G.FAMILIES)
def test_all_authored_successful_calls_match_offered_schemas(family):
    _assert_golden_schemas(G.build_cases(family), unified_tools())


@pytest.mark.parametrize("family", sorted(G.UNIFIED_FAMILIES))
def test_schema_guard_rejects_old_array_as_string_declaration(family):
    tools = unified_tools()
    next(tool for tool in tools if tool["name"] == "sum_values")["parameters"]["properties"]["values"] = {"type": "string"}
    cases = G.build_cases(family)
    for case in cases.values():
        if not case.get("tools") and any(
            event["kind"] == "tool_call" and event["name"] == "sum_values"
            for event in case["golden"]
        ):
            case["tools"] = copy.deepcopy(tools)
        for tool in case.get("tools", []):
            if tool["name"] == "sum_values":
                tool["parameters"]["properties"]["values"] = {"type": "string"}
    with pytest.raises(AssertionError):
        _assert_golden_schemas(cases, tools)


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


def test_schema_oracle_enforces_required_properties():
    schema = {
        "type": "object",
        "properties": {"count": {"type": "integer"}},
        "required": ["count"],
    }
    _assert_value({"count": 42}, schema)
    with pytest.raises(AssertionError):
        _assert_value({}, schema)


@pytest.mark.parametrize("script", ["capture_vllm_unified.py", "capture_sglang_unified.py"])
def test_peer_request_schema_projection_matches_shared_definition(script):
    tree = ast.parse((SRC / script).read_text())
    assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id in {"TOOLS", "TOOL_SCHEMAS"} for target in node.targets)]
    namespace = {
        "unified_tools": unified_tools,
        "openai_tools": openai_tools,
        "Tool": SimpleNamespace,
        "Function": SimpleNamespace,
    }
    exec(compile(ast.Module(body=assignments, type_ignores=[]), script, "exec"), namespace)
    if script == "capture_sglang_unified.py":
        assert namespace["TOOL_SCHEMAS"] == unified_tools()
        return
    projected = namespace["TOOLS"]
    actual = [tool["function"] for tool in projected]
    assert all(tool["type"] == "function" for tool in projected)
    assert actual == unified_tools()


def test_rust_harnesses_consume_the_shared_and_case_schema_owners():
    tests = SRC.parents[1] / "tests"
    common = (tests / "common/mod.rs").read_text()
    assert 'include_str!("../../utils/src/unified_tools.json")' in common
    assert 'serde_json::from_value(unified_tool_schemas())' in common
    assert "schemas.cloned().unwrap_or_else(unified_tool_schemas)" in common
    for name in ("unified_render.rs", "unified_parity.rs", "capture_cross_version.rs"):
        source = (tests / name).read_text()
        assert "unified_tools as tools" in source or "unified_tools_for_schemas" in source
        assert "fn tools()" not in source
    render = (tests / "unified_render.rs").read_text()
    parity = (tests / "unified_parity.rs").read_text()
    cross_version = (tests / "capture_cross_version.rs").read_text()
    assert "unified_tools_for_schemas(case.tools.as_ref())" in render
    assert "unified_tools_for_schemas(case.tools.as_ref())" in parity
    assert "unified_tool_schemas_for_case(case.tools.as_ref())" in render
    assert "unified_tool_schemas_for_case(tool_schema_json.as_ref())" in cross_version
    peer = (SRC / "capture_vllm_rust_unified.py").read_text()
    assert "tools: Vec<Tool>" in peer
    assert "make_parser(&case.family, &case.tools, &case.init.starting_state)" in peer
    assert "supports_tools=True" in peer
    assert "schema_bytes = SCHEMA_PATH.read_bytes()" in peer


def test_recovery_successor_respects_string_schema():
    case = G.build_cases("kimi_k3")["UNIFIED.kimi_k3_malformed_call_then_valid.kimi_k3"]
    assert case["golden"] == [{"kind": "tool_call", "name": "g", "arguments": {"y": "2"}}]
    assert G.k3_argument("y", "string", "2") in case["input"]
    assert G.k3_open("call", [("tool", "bad"), ("index", "1")]) + "not-an-argument" in case["input"]


def test_required_schema_rejects_missing_value():
    schema = {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]}
    assert matches_schema({"value": "é"}, schema)
    assert not matches_schema({}, schema)
    assert not matches_schema({"value": None}, schema)


def _malformed_native_empty_recovery(family, scenario, case, event):
    probes = {
        ("kimi_k2", "kimi_native_trailing_syntax_whitespace"),
        ("kimi_k2", "kimi_native_incompatible_string_recovery"),
        ("deepseek_v4", "kimi_native_trailing_syntax_whitespace"),
        ("deepseek_v4", "kimi_native_valid_object_trailing_garbage"),
    }
    if (family, scenario) not in probes:
        return False
    tools = ([{"name": "write_file", "parameters": {"type": "object", "properties": {"x": {"type": "string"}}}}]
             if scenario.endswith("incompatible_string_recovery") else G.WRITE_FILE_TOOLS)
    payload = ('{"x":"abc ' if scenario.endswith("incompatible_string_recovery") else
               '{"count":7,"content":"q"} trailing' if scenario.endswith("valid_object_trailing_garbage") else
               '{"count":7,"content": \t\n')
    return (event == {"kind": "tool_call", "name": "write_file", "arguments": {}}
            and case["tools"] == tools and payload in case["input"]
            and case["init"] == {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None})


@pytest.mark.parametrize("family,scenario", [
    ("kimi_k2", "kimi_native_trailing_syntax_whitespace"),
    ("kimi_k2", "kimi_native_incompatible_string_recovery"),
    ("deepseek_v4", "kimi_native_trailing_syntax_whitespace"),
    ("deepseek_v4", "kimi_native_valid_object_trailing_garbage"),
])
@pytest.mark.parametrize("mutation", ["arguments", "schema", "payload", "mode", "family"])
def test_malformed_native_schema_exception_rejects_changed_contracts(family, scenario, mutation):
    case = G.build_cases(family)[f"UNIFIED.{scenario}.{family}"]
    event = case["golden"][0]
    assert _malformed_native_empty_recovery(family, scenario, case, event)
    if mutation == "arguments":
        event["arguments"] = {"unexpected": True}
    elif mutation == "schema":
        case["tools"] = []
    elif mutation == "payload":
        case["input"] = "unrelated"
    elif mutation == "mode":
        case["init"]["tool_output_mode"] = "GuidedJson"
    else:
        family = "qwen3"
    assert not _malformed_native_empty_recovery(family, scenario, case, event)


@pytest.mark.parametrize("scenario,family", sorted(GPT58_NATIVE_REQUEST_DIGESTS))
@pytest.mark.parametrize("mutation", ["name", "arguments", "mode", "family", "schema", "input"])
def test_gpt58_schema_free_native_contracts_reject_changed_requests(scenario, family, mutation):
    case_id = f"UNIFIED.{scenario}.{family}"
    case = copy.deepcopy(G.build_cases(family)[case_id])
    _assert_golden_schemas({case_id: case}, unified_tools())
    if mutation in {"name", "arguments"}:
        if not case["golden"] or case["golden"][0]["kind"] != "tool_call":
            case["golden"].append({"kind": "tool_call", "name": "invented", "arguments": {}})
        else:
            case["golden"][0][mutation] = "invented" if mutation == "name" else {"changed": True}
    elif mutation == "mode":
        case["init"]["tool_output_mode"] = "GuidedJson"
    elif mutation == "family":
        case_id = case_id.rsplit(".", 1)[0] + ".qwen3"
    elif mutation == "schema":
        case["tools"] = [{"name": "lookup", "parameters": {"type": "object", "properties": {"markers": {"type": "integer"}}}}]
    else:
        case["input"] += "drift"
    with pytest.raises(AssertionError):
        _assert_golden_schemas({case_id: case}, unified_tools())


@pytest.mark.parametrize("family", sorted(G.UNIFIED_FAMILIES - {"qwen3_coder"}))
def test_muse_native_counterparts_keep_original_seven_tool_schemas(family):
    cases = {key: case for key, case in G.build_cases(family).items()
             if key.removeprefix("UNIFIED.").rsplit(".", 1)[0] in G.MUSE_NATIVE_REVIEW_SCENARIOS}
    assert all(case["tools"] == unified_tools() for case in cases.values())
    _assert_golden_schemas(cases, unified_tools())


def test_muse_native_opaque_schema_mutation_remains_strict():
    family = "qwen3"
    key = "UNIFIED.muse_opaque_declared_invoke.qwen3"
    case = G.build_cases(family)[key]
    next(tool for tool in case["tools"] if tool["name"] == "f")["parameters"]["properties"]["x"]["type"] = "integer"
    with pytest.raises(AssertionError):
        _assert_golden_schemas({key: case}, unified_tools())
