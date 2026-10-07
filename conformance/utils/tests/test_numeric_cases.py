# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sys
import tarfile
from decimal import Decimal
from pathlib import Path

import pytest

from conformance.utils.tests.schema_oracle import matches_schema

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import gen_unified_golden as unified
import gen_numeric_stream_cases as stream
import generate_conformance_table as report
from case_variants import group_null_variants
from markers import candidate_sig
from numeric_cases import NUMERIC_VARIANTS, applicable, canonical_arguments
from unified_taxonomy import numbered_id
from validate_conformance_status import cell_state


@pytest.mark.parametrize("scenario,label,schema,raw,expected", NUMERIC_VARIANTS)
def test_numeric_oracle_and_native_input_are_independent(scenario, label, schema, raw, expected):
    assert matches_schema(json.loads(expected, parse_float=Decimal), schema)
    assert numbered_id(scenario) == "UNIFIED." + label
    for family in unified.FAMILIES:
        cases = unified.build_cases(family)
        key = f"UNIFIED.{scenario}.{family}"
        assert (key in cases) == applicable(family, label)
        if key in cases:
            assert raw in cases[key]["input"]
            assert cases[key]["golden"][0]["arguments"] == '{"value":' + expected + '}'
            if family != "deepseek_v41":
                stream_family = "qwen3_coder" if family == "qwen3" else family
                other = stream.build_cases(stream_family)["TOOLCALLING.streamv1." + label]
                assert other["golden"]["calls"][0]["arguments"] == cases[key]["golden"][0]["arguments"]
                assert other["tools"] == cases[key]["tools"]


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


@pytest.mark.parametrize("raw,rounded", [
    ("9007199254740992.5", "9007199254740992.0"),
    ("9007199254740993.1", "9007199254740994.0"),
    ("0.10000000000000000001", "0.1"),
    ("1e-400", "0"),
])
def test_rounded_output_is_red_in_both_report_paths(raw, rounded):
    expected = '{"value":' + raw + '}'
    actual = '{"value":' + rounded + '}'
    golden = [{"kind": "tool_call", "name": "f", "arguments": expected}]
    deltas = [[{"kind": "tool_call", "name": "f", "arguments": actual, "complete": True}]]
    events = report._assemble_stream(deltas, preserve_arguments=True)
    assert report._unified_classify("qwen3", golden, events) == "ARG_MISMATCH"
    def block(arguments):
        return {"calls": [{"name": "f", "arguments": arguments}], "normal_text": ""}
    assert candidate_sig(block(expected)) != candidate_sig(block(actual))
    # Positive control: spelling changes with the same exact decimal value agree.
    assert canonical_arguments('{"value":42.0}') == canonical_arguments('{"value":4.2e1}')
    assert canonical_arguments('{"value":"42"}') != canonical_arguments('{"value":42}')


def test_minimax_m2_is_stream_only():
    assert "minimax_m2" not in unified.FAMILIES
    assert len(stream.build_cases("minimax_m2")) == len(NUMERIC_VARIANTS)


def test_integral_conversion_inapplicability_is_distinct_from_missing_capture():
    labels = ["7-14.const_decimal", "7-14.const_exponent"]

    def missing(label, family):
        return {
            "kind": "missing",
            "status": "missing",
            "sub": label,
            "case_id": None,
            "family": family,
            "cmp": None,
            "tooltip": {"head": "missing fixture", "na_note": "No fixture coverage for this case."},
        }

    tab = {
        "id": "tab-toolcalling-streamv1",
        "columns": [{"label": label, "sub": label, "group_key": "7"} for label in labels],
        "rows": [{"family": "deepseek_v4", "cells": {label: missing(label, "deepseek_v4") for label in labels}}],
        "candidates": [{"key": "dynamo_v2-0.7.14", "label": "Dynamo v2 0.7.14"}],
        "column_groups": [{"key": "7", "span": len(labels)}],
        "stats": {},
    }

    group_null_variants(tab)

    cell = tab["rows"][0]["cells"]["7-14.const_decimal"]
    assert cell["status"] == "na"
    assert cell["kind"] == "cell"
    assert cell["cmp"]["dynamo_v2-0.7.14"]["na"] == 1
    assert all(variant["status"] == "na" for variant in cell["variants"])
    assert all(
        variant["tooltip"]["na_note"]
        == "This family does not use the shared integral-decimal conversion contract."
        for variant in cell["variants"]
    )
    assert cell_state(cell, tab["candidates"][0])[0] == "na"

    supported = {
        **tab,
        "columns": [{"label": label, "sub": label, "group_key": "7"} for label in labels],
        "rows": [{"family": "qwen3_coder", "cells": {label: missing(label, "qwen3_coder") for label in labels}}],
        "column_groups": [{"key": "7", "span": len(labels)}],
        "stats": {},
    }
    group_null_variants(supported)
    supported_cell = supported["rows"][0]["cells"]["7-14.const_decimal"]
    assert supported_cell["status"] != "na"
    assert cell_state(supported_cell, supported["candidates"][0])[0] == "empty"
    assert all(variant["kind"] == "missing" for variant in supported_cell["variants"])


def test_exponent_chunks_remain_strings_after_yaml_roundtrip():
    import_value = {"delta_text": "9.0071992547409925e15"}
    encoded = stream.yaml.safe_dump(import_value)
    assert stream.yaml.compose(encoded).value[0][1].style in {"'", '"'}
    assert stream.yaml.safe_load(encoded) == import_value


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


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_numbers_are_invalid_and_never_match_zero(token):
    raw = '{"value":' + token + '}'
    finite = '{"value":0}'
    assert canonical_arguments(raw) == ["invalid_json", raw]
    assert canonical_arguments({"value": float(token)}) != canonical_arguments(finite)
    assert canonical_arguments({"value": Decimal(token)}) != canonical_arguments(finite)
    golden = [{"kind": "tool_call", "name": "f", "arguments": finite}]
    actual = [{"kind": "tool_call", "name": "f", "arguments": raw}]
    assert report._unified_classify("qwen3", golden, actual) == "ARG_MISMATCH"
    assert candidate_sig({"calls": [{"name": "f", "arguments": raw}]}) != candidate_sig(
        {"calls": [{"name": "f", "arguments": finite}]})
