# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Expanded headings retain old identities and keep related names adjacent."""

from pathlib import Path

import pytest

import gen_unified_golden as generator
import unified_history as history
import unified_taxonomy as taxonomy
import viewer_catalog


ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("scenario,previous,expected", [
    ("interrupted_reasoning_then_response", "10-interrupted_reasoning_then_response", "10-reason_then_content_interrupted"),
    ("tool_body_whitespace", "1-tool_body_whitespace", "1-tool_only_leading_json_body_whitespace"),
    ("gpt_oss_interrupted_reasoning_then_tool", "gpt_oss-interrupted_reasoning_then_tool", "12-tool_in_reason_interrupted_harmony_reasoning"),
    ("gpt_oss_quoted_message_header_literal", "gpt_oss-quoted_message_header_literal", "7-tool-argument-text-preservation-arg_marker_in_string_quoted_closer_and_response_header"),
])
def test_related_case_names_preserve_previous_aliases(scenario, previous, expected):
    assert taxonomy.case_label(scenario) == expected
    for label in (previous, expected):
        assert taxonomy.scenario_for_label(label, scenario=scenario) == scenario


def test_every_active_display_identity_matches_its_taxonomy_and_history() -> None:
    store = history.load_authored_store(ROOT / "conformance/fixtures-unified-v2")
    direct_descriptive_scenarios = {
        "schema_enum_inferred_integer",
        "schema_const_numeric_bare",
        "schema_enum_const_narrowing",
    }
    for family, document in store.families.items():
        for key, case in document.cases.items():
            if case["lifecycle"] != "active":
                continue
            assert case["display_id"] == taxonomy.numbered_id(case["scenario"])
            scenario = case["scenario"]
            if scenario in direct_descriptive_scenarios:
                assert case["historical_ids"] == []
                assert taxonomy.scenario_for_label(case["display_id"], scenario=scenario) == scenario
                continue
            previous = f"UNIFIED.{taxonomy.tax(scenario)[0]}-{taxonomy.original_case_description(scenario)}"
            if previous != case["display_id"]:
                assert previous in case["historical_ids"]
                for label in (previous, case["display_id"]):
                    assert taxonomy.scenario_for_label(label, scenario=scenario) == scenario


def test_expanded_single_family_cases_have_named_sections():
    metadata = viewer_catalog._source_metadata(ROOT)
    section_labels = {}
    for scenario, entry in metadata["unified_taxonomy"].items():
        assert entry["section_label"] == section_labels.setdefault(entry["section"], entry["section_label"])
        families = generator.scenario_families(scenario)
        if len(families) == 1:
            family = next(iter(families))
            assert entry["section"] == f"unified_g{family}"
            assert entry["section_label"] == taxonomy.UNIFIED_GROUP_LABEL.get(family, f"Family-specific tests: {family}")


def test_family_section_label_falls_back_for_a_new_family():
    assert taxonomy.display_section("tool_only", {"future_family"}) == (
        "unified_gfuture_family", "Family-specific tests: future_family")


def test_related_shared_case_names_use_the_parent_prefix():
    for scenario, variant in taxonomy.SHARED_CASE_VARIANTS.items():
        if scenario in taxonomy.RETIRED_SCENARIOS:
            continue
        assert taxonomy.case_label(scenario).startswith(taxonomy.case_label(variant.parent) + "_")
