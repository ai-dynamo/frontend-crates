# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent schema and applicability checks for the shared regression matrix."""

import json
import sys
from pathlib import Path

import pytest

from conformance.utils.tests.schema_oracle import matches_schema

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import gen_unified_golden as unified
import unified_taxonomy
from unified_taxonomy import numbered_id
from schema_cases import CONFORMANCE_CASES, SCHEMA_GROUPS, XML_COMPATIBILITY_CASES, XML_COMPATIBILITY_FAMILIES, schema_case_label, schema_tools

pytestmark = []


@pytest.mark.parametrize("quotes", ["single", "double"])
def test_custom_quote_display_tracks_proven_grammar_without_changing_identity(quotes):
    custom = f"xml_custom_type_{quotes}_quoted_object"
    string = f"xml_string_type_{quotes}_quoted_object"
    assert unified_taxonomy.display_tax(custom)[0] == (7 if quotes == "double" else "qwen3")
    assert unified_taxonomy.display_section(custom)[0] == (
        "unified_g7_declared" if quotes == "double" else "unified_gqwen3")
    assert unified_taxonomy.numbered_id(custom) == f"UNIFIED.7-{custom}"
    assert unified_taxonomy.display_tax(string)[0] == 7
    assert unified.scenario_families(custom) == (
        {"qwen3", "nemotron_v3", "glm47", "muse_glimmer", "muse_spark"}
        if quotes == "double" else {"qwen3", "nemotron_v3"})
    assert unified.scenario_families(string) == set(unified._TEMPLATE_FAMILIES)


@pytest.mark.parametrize("family", ["glm47", "muse_glimmer", "muse_spark"])
def test_custom_double_quote_keeps_original_schema_and_raw_body_ownership(family):
    case = unified.build_cases(family)[f"UNIFIED.xml_custom_type_double_quoted_object.{family}"]
    assert case["tools"] == [{"name": "schema_probe", "parameters": {
        "type": "object", "properties": {"value": {"type": "custom"}},
        "required": ["value"], "additionalProperties": False,
    }, "strict": False}]
    assert case["init"] == {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None}
    assert case["finish_reason"] == "stop" and case["finish_step"] is True
    body = '\n{"a":1}\n'
    assert case["input_chunks"].count(body) == 1
    assert "".join(case["input_chunks"]) == case["input"]
    if family == "glm47":
        assert f"<arg_value>{body}</arg_value>" in case["input"]
    else:
        assert f'<atem:parameter name="value">{body}</atem:parameter>' in case["input"]
    assert case["golden"] == [{"kind": "tool_call", "name": "schema_probe",
                               "arguments": {"value": {"a": 1}}}]
