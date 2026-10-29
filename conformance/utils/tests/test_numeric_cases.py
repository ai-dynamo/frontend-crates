# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import sys
import tarfile
from decimal import Decimal
from pathlib import Path

import pytest

from conformance.utils.tests.schema_oracle import matches_schema

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import gen_unified_golden as unified
from numeric_cases import NUMERIC_DESCRIPTIONS, NUMERIC_VARIANTS, applicable, canonical_arguments, numeric_expected, numeric_group, unified_numeric_expected
from unified_taxonomy import numbered_id
from conformance.utils.tests.fixture_test_support import run_node_contract


def test_kimi_k2_json_spacing_and_nested_exact_numbers_are_preserved():
    ordinary_arguments = {"city": "Paris", "nested": [1, {"amount": "exact"}]}
    ordinary_json = '{"city": "Paris", "nested": [1, {"amount": "exact"}]}'
    assert unified.r_tool_arguments("kimi_k2", "f", ordinary_arguments, 0).endswith(
        f"{ordinary_json}<|tool_call_end|><|tool_calls_section_end|>"
    )

    exact_arguments = {
        "city": "Paris",
        "nested": [1, {"amount": unified.NumericLiteral("9007199254740992.5")}],
    }
    exact_json = '{"city":"Paris","nested":[1,{"amount":9007199254740992.5}]}'
    assert unified._json_with_numeric_literals(exact_arguments) == exact_json
    assert unified.r_tool_arguments("kimi_k2", "f", exact_arguments, 0).endswith(
        f"{exact_json}<|tool_call_end|><|tool_calls_section_end|>"
    )


def test_numeric_backcaptures_use_their_versioned_archive_path():
    root = Path(__file__).resolve().parents[2] / "fixtures/toolcalling/fixtures-stream-v1"
    measured = 0
    for path in root.glob("dynamo*.tar.gz"):
        with tarfile.open(path) as archive:
            for member in archive:
                if member.isfile() and member.name.endswith("TOOLCALLING.streamv1.7-numeric.yaml"):
                    assert member.name.startswith(f"toolcalling/fixtures-stream-v1/{path.name[:-7]}/")
                    measured += 1
    assert measured > 0


def test_extreme_json_exponents_are_canonicalized_without_expanding_values():
    exponent = "1" + "0" * 40
    shifted_exponent = str(int(exponent) - 1)
    large = canonical_arguments('{"value":1e' + exponent + '}')
    same_value = canonical_arguments('{"value":10e' + shifted_exponent + '}')
    different_value = canonical_arguments('{"value":1e' + shifted_exponent + '}')
    assert large == same_value
    assert large != different_value
    assert large[1][0][1] == ["number", 0, [1], exponent]

    negative_exponent = "-" + exponent
    smaller = str(int(exponent) + 1)
    assert canonical_arguments('{"value":1e' + negative_exponent + '}') == canonical_arguments(
        '{"value":10e-' + smaller + '}'
    )

    long_exponent = "9" * 5000
    result = canonical_arguments('{"value":1e' + long_exponent + '}')
    assert result[1][0][1] == ["number", 0, [1], long_exponent]
    negative = canonical_arguments('{"value":-1e' + long_exponent + '}')
    assert negative[1][0][1] == ["number", 1, [1], long_exponent]
