# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent schema and applicability checks for the shared regression matrix."""

import json
import sys
from decimal import Decimal
from pathlib import Path

import pytest

from conformance.utils.tests.schema_oracle import matches_schema

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import gen_unified_golden as unified
import unified_taxonomy
from unified_taxonomy import numbered_id
from schema_cases import CONFORMANCE_CASES, SCHEMA_CASES, SCHEMA_GROUPS, XML_COMPATIBILITY_CASES, XML_COMPATIBILITY_FAMILIES, schema_case_label, schema_tools

pytestmark = []


@pytest.mark.parametrize("quotes", ["single", "double"])
def test_custom_quote_display_is_family_specific_and_preserves_aliases(quotes):
    custom = f"xml_custom_type_{quotes}_quoted_object"
    string = f"xml_string_type_{quotes}_quoted_object"
    assert unified_taxonomy.display_tax(custom)[0] == "qwen3"
    assert unified_taxonomy.display_section(custom)[0] == "unified_gqwen3"
    assert unified_taxonomy.numbered_id(custom) == f"UNIFIED.7-tool-argument-declared-types-{custom}"
    for family in ("qwen3", "nemotron_v3"):
        assert unified_taxonomy.scenario_for_label(f"UNIFIED.7-{custom}", family=family) == custom
    assert unified_taxonomy.numbered_id(string) == f"UNIFIED.7-tool-argument-declared-types-{string}"
    assert unified_taxonomy.display_tax(string)[0] == 7
    assert unified.scenario_families(custom) == {"qwen3", "nemotron_v3"}
    assert unified.scenario_families(string) == set(unified._TEMPLATE_FAMILIES)


def _json_first(case: dict) -> object:
    """The JSON-first reading of a case's native text, derived by the validator."""
    parameters = {"type": "object", "properties": {"value": case["schema"]}}
    if case.get("defs"):
        parameters["$defs"] = case["defs"]
    try:
        decoded = json.loads(case["raw"].strip(), parse_float=Decimal)
    except json.JSONDecodeError:
        return case["value"]
    # Parsers count an integral float spelling such as 1.0 as a number, which
    # this validator would also accept as an integer, so only these spellings count.
    typed = isinstance(decoded, (bool, int, list, dict)) or (
        isinstance(decoded, Decimal) and decoded != decoded.to_integral_value()
    )
    if typed and matches_schema({"value": decoded}, parameters):
        return float(decoded) if isinstance(decoded, Decimal) else decoded
    return case["value"]


@pytest.mark.parametrize("case", SCHEMA_CASES, ids=lambda case: case["name"])
def test_text_and_v1_values_record_only_real_differences(case):
    for key in ("text_value", "v1_value"):
        if key in case:
            assert case[key] != case["value"] or type(case[key]) is not type(case["value"]), key
    # Rust-only controls such as unresolved references are outside the validator.
    if case.get("conformance", True):
        assert case.get("text_value", case["value"]) == _json_first(case)
