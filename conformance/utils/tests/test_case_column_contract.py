# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One active case owns one scenario, request, oracle, and all of its aliases."""

from copy import deepcopy
from pathlib import Path

import pytest

import unified_history as history
import unified_taxonomy as taxonomy


ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def document():
    store = history.load_authored_store(ROOT / "conformance/fixtures-unified-v2")
    original = store.families["qwen3"].document
    return {**deepcopy(original), "cases": {"tool_only": deepcopy(original["cases"]["tool_only"])}}


@pytest.mark.parametrize("mutation", ["unknown_scenario", "array_kind", "wrong_id", "wrong_alias", "duplicate_scenario", "golden_variants", "empty_golden", "nested_events", "request_list", "case_variants"])
def test_yaml_rejects_more_than_one_case_identity_or_payload(document, mutation):
    value = deepcopy(document)
    case = value["cases"]["tool_only"]
    if mutation == "unknown_scenario":
        case["scenario"] = "not_a_scenario"
    elif mutation == "array_kind":
        case["golden"]["assembled"] = [{"kind": ["text"], "text": "bad"}]
    elif mutation == "wrong_id":
        case["display_id"] = "UNIFIED.10-reason_only"
    elif mutation == "wrong_alias":
        case["historical_ids"].append("UNIFIED.10-reason_only")
    elif mutation == "duplicate_scenario":
        value["cases"]["other"] = {**deepcopy(case), "display_id": "UNIFIED.unknown", "historical_ids": []}
    elif mutation == "golden_variants":
        case["golden"]["variants"] = [{"assembled": []}]
    elif mutation == "empty_golden":
        case["golden"] = {}
    elif mutation == "nested_events":
        case["golden"]["assembled"] = [[{"kind": "text", "text": "another case"}]]
    elif mutation == "request_list":
        case["request"] = [case["request"], deepcopy(case["request"])]
    else:
        case["variants"] = [deepcopy(case)]
    with pytest.raises(ValueError):
        history._validate_family(ROOT / "conformance/fixtures-unified-v2/families/qwen3/inputs_and_golden.yaml", value)


@pytest.mark.parametrize("events", [[], [{"kind": "reasoning", "text": "think"}, {"kind": "text", "text": "answer"}], [{"kind": "tool_call", "name": "a", "arguments": {}}, {"kind": "tool_call", "name": "b", "arguments": {}}]])
def test_one_request_can_emit_zero_or_many_ordered_events(document, events):
    value = deepcopy(document)
    value["cases"]["tool_only"]["golden"]["assembled"] = events
    history._validate_family(ROOT / "conformance/fixtures-unified-v2/families/qwen3/inputs_and_golden.yaml", value)


def test_section_seven_previous_and_current_names_resolve_to_one_case():
    scenario = "gpt_oss_control_marker_inside_argument_string"
    for label in ("gpt_oss-control_marker_inside_argument_string", "7-arg_marker_in_string_native_controls_in_arguments", "7-tool-argument-text-preservation-arg_marker_in_string_native_controls_in_arguments"):
        assert taxonomy.scenario_for_label(label) == scenario


@pytest.mark.parametrize("family", ["beam", "deepseek_v4", "gemma4", "glm47", "kimi_k2", "kimi_k3", "muse_glimmer", "qwen3"])
def test_shared_marker_case_old_alias_resolves_across_families(family):
    scenario = "deepseek_v41_mixed_control_text_in_string"
    for label in ("7-3", taxonomy.previous_case_label(scenario), taxonomy.case_label(scenario)):
        assert taxonomy.scenario_for_label(label, family=family) == scenario
