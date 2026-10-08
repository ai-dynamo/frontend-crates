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
import gen_numeric_stream_cases as stream
import generate_conformance_table as report
from case_variants import group_null_variants
from markers import candidate_sig
from numeric_cases import NUMERIC_VARIANTS, applicable, canonical_arguments, numeric_group
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
            assert canonical_arguments(cases[key]["golden"][0]["arguments"]) == canonical_arguments('{"value":' + expected + '}')
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


def test_string_fallback_inapplicability_is_distinct_from_missing_capture():
    labels = ["7-14.fraction_near_integer", "7-14.fraction_fallback"]

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

    cell = tab["rows"][0]["cells"]["7-14.fraction_near_integer"]
    assert cell["status"] == "na"
    assert cell["kind"] == "cell"
    assert cell["cmp"]["dynamo_v2-0.7.14"]["na"] == 1
    assert all(variant["status"] == "na" for variant in cell["variants"])
    assert all(
        variant["tooltip"]["na_note"]
        == "This grammar does not coerce untyped fractional text through an integer/string schema."
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
    supported_cell = supported["rows"][0]["cells"]["7-14.fraction_near_integer"]
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


@pytest.mark.parametrize("label", ["7-14", "7-15", "7-14.unrelated", "7-15.unrelated"])
def test_numeric_groups_do_not_claim_independent_schema_cases(label):
    assert numeric_group(label) is None


def test_every_authored_numeric_leaf_keeps_its_group():
    for _, label, _, _, expected in NUMERIC_VARIANTS:
        assert numeric_group(label) == ("7-14.string" if expected.startswith('"') else label.split(".", 1)[0])


def test_numeric_applicability_separates_numbers_from_schema_string_coercion():
    for family in set(unified.FAMILIES) | set(stream.V2_FAMILIES):
        cases = stream.build_cases("qwen3_coder" if family == "qwen3" else family)
        for _, label, _, raw, expected in NUMERIC_VARIANTS:
            if expected.startswith('"'):
                assert applicable(family, label) == (family in {
                    "qwen3", "qwen3_coder", "minimax_m2", "glm47", "minimax_m3"})
            else:
                assert applicable(family, label)
                case = cases["TOOLCALLING.streamv1." + label]
                assert matches_schema(json.loads(raw, parse_float=Decimal),
                                      case["tools"][0]["parameters"]["properties"]["value"])
                assert canonical_arguments('{"value":' + raw + '}') == canonical_arguments('{"value":' + expected + '}')


def test_new_glm_and_m3_integral_probes_do_not_require_numeric_union_preference():
    for family in ("glm47", "minimax_m3"):
        numeric = stream.build_cases(family)["TOOLCALLING.streamv1.7-14.large_decimal"]
        assert numeric["tools"][0]["parameters"]["properties"]["value"] == {"type": "integer"}
        fallback = stream.build_cases(family)["TOOLCALLING.streamv1.7-14.fraction_fallback"]
        assert fallback["tools"][0]["parameters"]["properties"]["value"] == {"type": ["integer", "string"]}
        assert fallback["golden"]["calls"][0]["arguments"] == '{"value":"42.5"}'
    for family in ("qwen3_coder", "minimax_m2"):
        original = stream.build_cases(family)["TOOLCALLING.streamv1.7-14.large_decimal"]
        assert original["tools"][0]["parameters"]["properties"]["value"] == {"type": ["integer", "string"]}


@pytest.mark.parametrize("family", ["qwen3", "deepseek_v4"])
def test_numeric_display_keeps_bare_schema_results_and_descriptions(family):
    labels = ["7-14", "7-15", "7-14.const_decimal", "7-14.const_exponent",
              "7-15.ordinary", "7-15.round_down"]
    cells = {label: {"sub": label, "case_id": "UNIFIED." + label, "family": family,
                     "kind": "cell", "status": "ok", "red_on_diff": True,
                     "cmp": {"golden": {"sig": 1}, "dynamo": {"sig": 2}},
                     "tooltip": {"head": label, "description": "schema " + label}}
             for label in labels}
    original = copy.deepcopy(cells)
    tab = {"id": "tab-unified", "columns": [{"label": label, "sub": label,
            "desc": "schema " + label, "group_key": "7"} for label in labels],
           "rows": [{"family": family, "cells": cells}], "candidates": [{"key": "dynamo"}],
           "column_groups": [{"key": "7", "span": len(labels)}], "stats": {},
           "glossary": [{"rows": [(label, "schema " + label) for label in labels]}]}
    group_null_variants(tab)
    assert [column["label"] for column in tab["columns"]] == ["7-14", "7-15", "7-14.*", "7-15.*"]
    for label in labels[:2]:
        assert cells[label] == original[label]
        assert next(c for c in tab["columns"] if c["label"] == label)["desc"] == "schema " + label
        assert (label, "schema " + label) in tab["glossary"][0]["rows"]
    for root, prefix in [("7-14.const_decimal", "7-14."), ("7-15.ordinary", "7-15.")]:
        assert cells[root]["case_id"] == "UNIFIED." + prefix + "*"
        assert all(leaf["case_id"].startswith("UNIFIED." + prefix) for leaf in cells[root]["variants"])
    assert cells["7-14.const_decimal"]["status"] != "na"
