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
        "schema_declared_integer_boolean_number_order",
        # New scenarios have descriptive identities without prior aliases.
        "arg_required_nullable_omitted",
        "arg_required_nullable_type_array_omitted",
        "arg_required_nullable_empty_invoke",
        "arg_optional_nullable_omitted",
        "arg_required_nonnullable_omitted",
        "arg_required_nullable_supplied",
        "arg_required_nullable_ambiguous_oneof_omitted",
        "arg_required_nullable_lookup_budget_omitted",
    }
    # These profiles are first authored with descriptive IDs; their current
    # cases have no prior authored identities to preserve as aliases.
    direct_descriptive_families = {
        "deepseek_v3", "deepseek_v3_1", "deepseek_v3_2", "harmony",
        "harmony_text", "hermes", "inkling", "jamba", "llama3_json",
        "minimax_m2", "mistral", "nemotron_deci", "nemotron_nano",
        "phi4", "pythonic", "qwen25", "qwen3_coder",
    }
    for family, document in store.families.items():
        for key, case in document.cases.items():
            if case["lifecycle"] != "active":
                continue
            assert case["display_id"] == taxonomy.numbered_id(case["scenario"])
            scenario = case["scenario"]
            if scenario in direct_descriptive_scenarios or family in direct_descriptive_families:
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
            label_key = taxonomy.UNIFIED_DISPLAY["family_label_aliases"].get(family, family)
            assert entry["section_label"] == taxonomy.UNIFIED_GROUP_LABEL.get(label_key, f"Family-specific tests: {family}")


@pytest.mark.parametrize("family,label_key", [("gemma4", "gemma"), ("glm47", "glm5"), ("kimi_k3", "kimi")])
def test_family_alias_reaches_curated_section_label(family, label_key):
    section, label = taxonomy.display_section("tool_only", {family})
    assert section == f"unified_g{family}"
    assert label == taxonomy.UNIFIED_GROUP_LABEL[label_key]
    assert label != f"Family-specific tests: {family}"
    heading = label.removeprefix("Single Family Test: ")
    assert f"### {heading}\n" in (ROOT / "conformance/utils/lib/parsers/UNIFIED_CASES.md").read_text()


def test_family_section_label_falls_back_for_a_new_family():
    assert taxonomy.display_section("tool_only", {"future_family"}) == (
        "unified_gfuture_family", "Family-specific tests: future_family")


def test_gpt_oss_historical_route_boundary_reaches_the_viewer():
    generations = viewer_catalog._source_metadata(ROOT)["unified_presentation"]["parser_generations"]
    assert generations["gpt_oss"]["native_unified_since"] == "0.7.28"
    assert generations["harmony"]["native_unified_v2"] is False


def test_related_shared_case_names_use_the_parent_prefix():
    for scenario, variant in taxonomy.SHARED_CASE_VARIANTS.items():
        if scenario in taxonomy.RETIRED_SCENARIOS:
            continue
        assert taxonomy.case_label(scenario).startswith(taxonomy.case_label(variant.parent) + "_")
