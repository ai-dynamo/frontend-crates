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
from numeric_cases import NUMERIC_DESCRIPTIONS, NUMERIC_VARIANTS, applicable, canonical_arguments, numeric_expected, numeric_group, unified_numeric_expected
from unified_taxonomy import numbered_id
from validate_conformance_status import cell_state


@pytest.mark.parametrize("scenario,label,schema,raw,expected", NUMERIC_VARIANTS)
def test_numeric_oracle_and_native_input_are_independent(scenario, label, schema, raw, expected):
    assert matches_schema(json.loads(expected, parse_float=Decimal), schema)
    assert numbered_id(scenario) == f"UNIFIED.7-{scenario}"
    for family in unified.FAMILIES:
        cases = unified.build_cases(family)
        key = f"UNIFIED.{scenario}.{family}"
        assert (key in cases) == applicable(family, label)
        if key in cases:
            assert raw in cases[key]["input"]
            if family == "gpt_oss":
                native_arguments = cases[key]["input"].split("<|message|>", 1)[1].split("<|call|>", 1)[0]
                assert native_arguments == '{"value":' + raw + '}'
            v2_expected = unified_numeric_expected(family, label, raw, expected)
            assert canonical_arguments(cases[key]["golden"][0]["arguments"]) == canonical_arguments('{"value":' + v2_expected + '}')
            if family not in {"deepseek_v41", "beam"}:
                stream_family = "qwen3_coder" if family == "qwen3" else family
                other = stream.build_cases(stream_family)["TOOLCALLING.streamv1." + label]
                assert canonical_arguments(other["golden"]["calls"][0]["arguments"]) == canonical_arguments(
                    cases[key]["golden"][0]["arguments"]
                )


def test_beam_v2_preserves_numeric_text_when_the_request_schema_permits_strings():
    beam_cases = unified.build_cases("beam")
    stream_cases = stream.build_cases("beam")
    for scenario, label, raw, expected in (
        (scenario, label, raw, expected)
        for scenario, label, _schema, raw, expected in NUMERIC_VARIANTS
        if numeric_group(label) == "7-14" and not label.split(".", 1)[1].startswith(("const_", "enum_"))
    ):
        case = beam_cases[f"UNIFIED.{scenario}.beam"]
        assert case["tools"][0]["parameters"]["properties"]["value"]["type"] == ["integer", "string"]
        assert canonical_arguments(case["golden"][0]["arguments"]) == canonical_arguments(
            '{"value":' + json.dumps(raw) + "}"
        )
        stream_case = stream_cases["TOOLCALLING.streamv1." + label]
        assert canonical_arguments(stream_case["golden"]["calls"][0]["arguments"]) == canonical_arguments(
            '{"value":' + numeric_expected("beam", label, raw, expected) + "}"
        )


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


def test_missing_arguments_differ_from_explicit_json_null():
    missing = {"calls": [{"name": "f"}]}
    explicit_null = {"calls": [{"name": "f", "arguments": "null"}]}
    assert candidate_sig(missing) != candidate_sig(explicit_null)


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
    large_json = '{"value":1e' + exponent + '}'
    same_json = '{"value":10e' + shifted_exponent + '}'
    assert candidate_sig({"calls": [{"name": "f", "arguments": large_json}]}) == candidate_sig(
        {"calls": [{"name": "f", "arguments": same_json}]}
    )
    assert report._unified_classify(
        "qwen3",
        [{"kind": "tool_call", "name": "f", "arguments": large_json}],
        [{"kind": "tool_call", "name": "f", "arguments": same_json}],
    ) == "MATCH"


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
                    "beam", "qwen3", "qwen3_coder", "minimax_m2", "glm47", "minimax_m3", "nemotron_v3"})
            else:
                assert applicable(family, label)
                case = cases["TOOLCALLING.streamv1." + label]
                assert matches_schema(json.loads(raw, parse_float=Decimal),
                                      case["tools"][0]["parameters"]["properties"]["value"])
                assert canonical_arguments('{"value":' + raw + '}') == canonical_arguments('{"value":' + expected + '}')


def test_numeric_schema_preservation_keeps_v2_and_v1_requests_independent():
    for family in ("glm47", "minimax_m3", "beam", "nemotron_v3"):
        if family in unified.FAMILIES:
            unified_case = unified.build_cases(family)[f"UNIFIED.arg_numeric_14_large_decimal.{family}"]
            fixture = Path(__file__).resolve().parents[2] / "fixtures-unified-v2" / "families" / family / "inputs_and_golden.yaml"
            authored = stream.yaml.safe_load(fixture.read_text())["cases"].values()
            request = next(case["request"] for case in authored if case["scenario"] == "arg_numeric_14_large_decimal")
            assert unified_case["tools"] == request["tools"]
        numeric = stream.build_cases(family)["TOOLCALLING.streamv1.7-14.large_decimal"]
        assert numeric["tools"][0]["parameters"]["properties"]["value"] == {"type": "integer"}
        fallback = stream.build_cases(family)["TOOLCALLING.streamv1.7-14.fraction_fallback"]
        assert fallback["tools"][0]["parameters"]["properties"]["value"] == {"type": ["integer", "string"]}
        assert fallback["golden"]["calls"][0]["arguments"] == '{"value":"42.5"}'
    for family in ("qwen3_coder", "minimax_m2"):
        original = stream.build_cases(family)["TOOLCALLING.streamv1.7-14.large_decimal"]
        assert original["tools"][0]["parameters"]["properties"]["value"] == {"type": ["integer", "string"]}


def test_unified_nested_numeric_case_preserves_exact_values_for_every_family():
    scenario = "arg_numeric_nested_exact"
    expected = (
        '{"nested":{"fraction":9007199254740992.5,"tiny":1e-400},'
        '"values":[9007199254740993,-9007199254740993]}'
    )
    oracle = json.loads(expected, parse_float=Decimal)
    assert oracle == {
        "nested": {"fraction": Decimal("9007199254740992.5"), "tiny": Decimal("1e-400")},
        "values": [9007199254740993, -9007199254740993],
    }
    assert numbered_id(scenario) == "UNIFIED.7-arg_numeric_nested_exact"

    for family in unified.FAMILIES:
        case = unified.build_cases(family)[f"UNIFIED.{scenario}.{family}"]
        assert matches_schema(oracle, case["tools"][0]["parameters"])
        assert case["golden"][0]["arguments"] == expected
        for token in ("9007199254740992.5", "1e-400", "9007199254740993", "-9007199254740993"):
            assert token in case["input"]


def test_unified_numeric_goldens_use_the_exact_integer_oracle_for_all_families():
    scenario = "arg_numeric_14_zero_underflow"
    label = "7-14.zero_underflow"
    for family in unified.FAMILIES:
        case = unified.build_cases(family).get(f"UNIFIED.{scenario}.{family}")
        if case is not None:
            expected = unified_numeric_expected(family, label, "0.0e-400", "0")
            assert case["golden"][0]["arguments"] == '{"value":' + expected + "}"


def test_numeric_string_fallback_description_names_applicable_families():
    description = NUMERIC_DESCRIPTIONS["7-14.string"]
    for family in ("Qwen3", "Qwen3-Coder", "MiniMax M2", "GLM-4.7", "MiniMax M3", "Beam", "Nemotron v3"):
        assert family in description


@pytest.mark.parametrize("family", ["qwen3", "deepseek_v4"])
def test_numeric_display_keeps_bare_schema_results_and_descriptions(family):
    labels = ["7-14", "7-15", "7-14.const_decimal", "7-14.const_exponent",
              "7-14.fraction_fallback", "7-14.fraction_near_integer",
              "7-14.fraction_underflow", "7-15.ordinary", "7-15.round_down"]
    label_scenarios = {label: scenario for scenario, label, *_ in NUMERIC_VARIANTS}
    label_scenarios.update({
        "7-14": "unused_reference_graph_parameter_types",
        "7-15": "nullable_reference_alias_literals",
    })
    scenarios = [label_scenarios[label] for label in labels]
    cells = {scenario: {"sub": scenario, "case_id": "UNIFIED." + label, "family": family,
                        "kind": "cell", "status": "ok", "red_on_diff": True,
                        "cmp": {"golden": {"sig": 1}, "dynamo": {"sig": 2}},
                        "tooltip": {"head": scenario, "description": "schema " + scenario}}
             for label, scenario in zip(labels, scenarios)}
    original = copy.deepcopy(cells)
    tab = {"id": "tab-unified", "columns": [{"label": label, "sub": scenario,
            "desc": "schema " + label, "group_key": "7"} for label, scenario in zip(labels, scenarios)],
           "rows": [{"family": family, "cells": cells}], "candidates": [{"key": "dynamo"}],
           "column_groups": [{"key": "7", "span": len(labels)}], "stats": {},
           "glossary": [{"rows": [(label, "schema " + label) for label in labels]}]}
    group_null_variants(tab)
    assert [column["label"] for column in tab["columns"]] == [
        "7-14", "7-15", "7-17", "7-19", "7-18"]
    for label, scenario in zip(labels[:2], scenarios[:2]):
        assert cells[scenario] == original[scenario]
        assert next(c for c in tab["columns"] if c["label"] == label)["desc"] == "schema " + label
        assert (label, "schema " + label) in tab["glossary"][0]["rows"]
    for root, prefix, alias in [(label_scenarios["7-14.const_decimal"], "7-14.", "7-17"),
                                (label_scenarios["7-15.ordinary"], "7-15.", "7-18")]:
        assert cells[root]["case_id"] == "UNIFIED." + alias
        assert all(leaf["case_id"].startswith("UNIFIED." + prefix) for leaf in cells[root]["variants"])
    assert cells[label_scenarios["7-14.const_decimal"]]["status"] != "na"
    assert cells[label_scenarios["7-14.fraction_fallback"]]["case_id"] == "UNIFIED.7-19"
