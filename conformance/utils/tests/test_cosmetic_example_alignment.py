# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replacement examples preserve each original request except approved lexical edits."""

from copy import deepcopy
from pathlib import Path

import pytest

import gen_unified_golden as generator
import unified_history as history
import unified_taxonomy as taxonomy
from viewer_codec import typed_equal


ROOT = Path(__file__).resolve().parents[3]
WHITESPACE_FAMILIES = ("gpt_oss", "kimi_k2", "kimi_k3")
RESPONSE_FAMILIES = ("beam", "gpt_oss", "kimi_k3", "muse_glimmer", "muse_spark")
EXAMPLES = (
    *((family, "gpt_oss_tool_body_whitespace", "tool_body_whitespace") for family in WHITESPACE_FAMILIES),
    *((family,
       "kimi_k3_elided_think_close_to_response" if family == "kimi_k3" else "gpt_oss_interrupted_reasoning_then_response",
       "interrupted_reasoning_then_response") for family in RESPONSE_FAMILIES),
    *(("gpt_oss", parent, parent + "_consistent")
      for parent in ("reason_markup_in_arg", "reason_markup_in_arg_with_text")),
)


@pytest.fixture(scope="module")
def authored():
    return history.load_authored_store(ROOT / "conformance/fixtures-unified-v2")


def _assert_cosmetic_replacement(original, replacement, scenario):
    assert original["lifecycle"] == "retired"
    assert original["display_id"] is None
    assert original["historical_ids"]
    assert replacement["lifecycle"] == "active"
    assert replacement["display_id"] == taxonomy.numbered_id(scenario)
    expected_request = deepcopy(original["request"])
    expected_events = deepcopy(original["golden"]["assembled"])
    if scenario == "tool_body_whitespace":
        substitutions = (("lookup", "get_weather"), ('"q"', '"city"'), ('"x"', '"Paris"'))
        for tool in expected_request["tools"]:
            assert tool["name"] == "lookup"
            tool["name"] = "get_weather"
            properties = tool["parameters"]["properties"]
            properties["city"] = properties.pop("q")
        assert expected_events == [{"kind": "tool_call", "name": "lookup", "arguments": {"q": "x"}}]
        expected_events = [{"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}}]
    elif scenario == "interrupted_reasoning_then_response":
        assert expected_events[0] == {"kind": "reasoning", "text": "checking"}
        assert expected_events[1]["text"] in {"done", "The answer is 42."}
        substitutions = (("checking", "let me think"), ("done", "The answer is 42."))
        expected_events = [{"kind": "reasoning", "text": "let me think"},
                           {"kind": "text", "text": "The answer is 42."}]
    else:
        assert scenario in {"reason_markup_in_arg_consistent", "reason_markup_in_arg_with_text_consistent"}
        substitutions = (("secret", "reconsider"),)
        event = next(event for event in expected_events if event["kind"] == "tool_call")
        assert event["arguments"]["note"].count("secret") == 1
        event["arguments"]["note"] = event["arguments"]["note"].replace("secret", "reconsider")
    for old, new in substitutions:
        expected_request["input"] = expected_request["input"].replace(old, new)
        for chunk in expected_request["chunks"]:
            chunk["delta_text"] = chunk["delta_text"].replace(old, new)
    assert typed_equal(replacement["request"], expected_request), "non-cosmetic request change"
    assert typed_equal(replacement["golden"]["assembled"], expected_events), "non-cosmetic expected output change"
    assert replacement["policy"] == original["policy"]


@pytest.mark.parametrize("family,original,active", EXAMPLES)
def test_examples_change_only_incidental_tokens(authored, family, original, active):
    records = authored.families[family].cases
    _assert_cosmetic_replacement(records[original], records[active], active)
    generated = generator.build_cases(family)[f"UNIFIED.{active}.{family}"]
    request = records[active]["request"]
    assert generated["input"] == request["input"]
    assert generated["policy"] == records[original]["policy"]
    assert typed_equal(generated["init"], request["init"])
    assert typed_equal(generated["tools"], request["tools"])
    assert typed_equal(generated["golden"], records[active]["golden"]["assembled"])


def test_replacements_keep_existing_family_coverage(authored):
    assert generator.scenario_families("tool_body_whitespace") == set(WHITESPACE_FAMILIES)
    assert generator.scenario_families("interrupted_reasoning_then_response") == set(RESPONSE_FAMILIES)
    for scenario in ("reason_markup_in_arg", "reason_markup_in_arg_with_text"):
        assert generator.scenario_families(scenario + "_consistent") == {"gpt_oss"}
        assert "gpt_oss" not in generator.scenario_families(scenario)
    for family, old, active in EXAMPLES:
        assert old not in {case["scenario"] for case in authored.families[family].cases.values()
                           if case["lifecycle"] == "active"}
        assert f"UNIFIED.{old}.{family}" not in generator.build_cases(family)


@pytest.mark.parametrize("mutation", ["payload", "type", "marker", "history_binding", "chunks", "schema"])
def test_cosmetic_guard_rejects_semantic_and_binding_mutations(authored, mutation):
    records = authored.families["kimi_k3"].cases
    old = deepcopy(records["gpt_oss_tool_body_whitespace"])
    active = deepcopy(records["tool_body_whitespace"])
    if mutation == "payload":
        active["golden"]["assembled"][0]["arguments"]["city"] = "London"
    elif mutation == "type":
        active["golden"]["assembled"][0]["arguments"]["city"] = 42
    elif mutation == "marker":
        active["request"]["input"] = active["request"]["input"].replace("<|sep|>", "", 1)
    elif mutation == "history_binding":
        old["request"] = deepcopy(active["request"])
    elif mutation == "chunks":
        active["request"]["chunks"] = [{"delta_text": active["request"]["input"]}]
    else:
        active["request"]["tools"][0]["parameters"]["properties"]["city"]["type"] = "number"
    with pytest.raises(AssertionError):
        _assert_cosmetic_replacement(old, active, "tool_body_whitespace")


def test_unmodified_interrupted_tool_variants_match_shared_parents():
    for suffix in ("", "_with_text"):
        cases = generator.build_cases("gpt_oss")
        child = cases[f"UNIFIED.gpt_oss_interrupted_reasoning_then_tool{suffix}.gpt_oss"]
        parent = generator.build_cases("qwen3")[f"UNIFIED.tool_in_reason{suffix}.qwen3"]
        assert child["golden"] == parent["golden"]
