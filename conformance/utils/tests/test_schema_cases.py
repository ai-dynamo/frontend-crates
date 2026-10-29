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
def test_custom_quote_display_is_family_specific_without_changing_identity(quotes):
    custom = f"xml_custom_type_{quotes}_quoted_object"
    string = f"xml_string_type_{quotes}_quoted_object"
    assert unified_taxonomy.display_tax(custom)[0] == "qwen3"
    assert unified_taxonomy.display_section(custom)[0] == "unified_gqwen3"
    assert unified_taxonomy.numbered_id(custom) == f"UNIFIED.7-{custom}"
    assert unified_taxonomy.display_tax(string)[0] == 7
    assert unified.scenario_families(custom) == {"qwen3", "nemotron_v3"}
    assert unified.scenario_families(string) == set(unified._TEMPLATE_FAMILIES)
