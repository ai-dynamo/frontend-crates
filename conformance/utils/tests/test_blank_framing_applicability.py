import copy
import json
import os
from pathlib import Path
import sys

SNAPSHOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SNAPSHOT / "conformance/utils/src"))

import pytest
import gen_unified_golden as generator
import unified_history as history
import explode_unified_fixtures as explode
from viewer_codec import typed_equal

FAMILIES = sorted(path.parent.name for path in (SNAPSHOT / "conformance/fixtures-unified-v2/families").glob("*/inputs_and_golden.yaml"))
SCENARIOS = [row[0] for row in generator.BLANK_FRAMING] + ["blank_framing_incomplete_call"]
CALL = {"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}}
SECOND = {"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Tokyo"}}
LF = {"kind": "text", "text": "\n"}
# Expected events are independently authored from P2 and the existing DSML
# separator policy. This file never reads any producer observations.
DSML_EXPECTED = {
    "blank_framing_tool_only": [LF, CALL, LF],
    "blank_framing_between_calls_single_chunk": [LF, CALL, SECOND, LF],
    "blank_framing_with_prose": [{"kind": "text", "text": "Pressing it now."}, CALL, LF],
    "blank_framing_after_reasoning": [{"kind": "reasoning", "text": "Check weather."}, CALL, LF],
    "blank_framing_prose_after_call": [LF, CALL, {"kind": "text", "text": "\nDone."}],
}

@pytest.fixture(scope="module")
def built():
    return {family: generator.build_cases(family) for family in FAMILIES}

@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_every_native_family_has_blank_counterpart(built, family, scenario):
    assert family in generator.scenario_families(scenario)
    assert f"UNIFIED.{scenario}.{family}" in built[family]

@pytest.mark.parametrize("family", ["deepseek_v4", "deepseek_v41"])
@pytest.mark.parametrize("scenario,expected", DSML_EXPECTED.items())
def test_dsml_contract_preserves_nonseparator_text(built, family, scenario, expected):
    case = built[family][f"UNIFIED.{scenario}.{family}"]
    assert typed_equal(case["golden"], expected)

@pytest.mark.parametrize("family", FAMILIES)
def test_incomplete_native_call_keeps_lf_and_drops_call(built, family):
    case = built[family][f"UNIFIED.blank_framing_incomplete_call.{family}"]
    assert case["golden"] == [LF]
    assert case["input"].endswith("Par")
    assert "Paris" not in case["input"]
    assert case["init"] == {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None}

@pytest.mark.parametrize("family", FAMILIES)
def test_existing_upstream_requests_and_goldens_unchanged(built, family):
    document = history.load_yaml(SNAPSHOT / f"conformance/fixtures-unified-v2/families/{family}/inputs_and_golden.yaml")
    rows = explode._split_only_cases(family, {"family": family, "cases": built[family]})
    for case in document["cases"].values():
        if case["scenario"] not in SCENARIOS:
            continue
        row = rows[case["display_id"]]
        # The existing generator strips the synthetic finish step; preserve
        # its original historical identity separately from model chunks.
        request = copy.deepcopy(row["request"])
        if case["request"]["chunks"][-1] == {"delta_text": "‹finish›"} and request["chunks"][-1] != {"delta_text": "‹finish›"}:
            request["chunks"].append({"delta_text": "‹finish›"})
        assert typed_equal(request, case["request"])
        assert typed_equal(row["golden"], case["golden"])


def _assert_applicability_contract(cases):
    for family in FAMILIES:
        for scenario in SCENARIOS:
            assert f"UNIFIED.{scenario}.{family}" in cases[family]
        assert cases[family][f"UNIFIED.blank_framing_incomplete_call.{family}"]["golden"] == [LF]
    for family in ("deepseek_v4", "deepseek_v41"):
        for scenario, expected in DSML_EXPECTED.items():
            assert typed_equal(cases[family][f"UNIFIED.{scenario}.{family}"]["golden"], expected)


def test_qualified_contract_accepts_unmodified_cases(built):
    _assert_applicability_contract(built)


def test_scope_mutation_is_detected(built):
    mutated = copy.deepcopy(built)
    del mutated["minimax_m3"]["UNIFIED.blank_framing_incomplete_call.minimax_m3"]
    with pytest.raises(AssertionError):
        _assert_applicability_contract(mutated)


def test_dsml_removed_leading_lf_is_detected(built):
    mutated = copy.deepcopy(built)
    mutated["deepseek_v4"]["UNIFIED.blank_framing_tool_only.deepseek_v4"]["golden"].pop(0)
    with pytest.raises(AssertionError):
        _assert_applicability_contract(mutated)


def test_incomplete_call_completion_is_detected(built):
    mutated = copy.deepcopy(built)
    mutated["beam"]["UNIFIED.blank_framing_incomplete_call.beam"]["golden"].append(CALL)
    with pytest.raises(AssertionError):
        _assert_applicability_contract(mutated)


def test_incidental_city_mutation_is_detected(built):
    mutated = copy.deepcopy(built)
    mutated["deepseek_v41"]["UNIFIED.blank_framing_tool_only.deepseek_v41"]["golden"][1]["arguments"]["city"] = "London"
    with pytest.raises(AssertionError):
        _assert_applicability_contract(mutated)
