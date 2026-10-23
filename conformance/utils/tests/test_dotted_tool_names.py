# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import gen_unified_golden as generator
import generate_conformance_table as table
from case_variants import leaf_cells
from model import hydrate_page
from unified_taxonomy import tax
from validate_conformance_status import load_model


DOTTED_CASES = [
    ("dotted_name_minimal", [("a.b", {})], 1),
    ("dotted_name_mixed_punctuation", [("my-tool.v2", {"location": "Beijing"})], 1),
    ("dotted_name_multiple_segments", [("server.tools.weather", {"location": "Beijing"})], 1),
    ("dotted_name_shared_prefix", [("server.weather", {"location": "Beijing"}),
                                   ("server.time", {"timezone": "Asia/Shanghai"})], 2),
]


@pytest.mark.parametrize("family", sorted(generator.DOTTED_NAME_FAMILIES))
@pytest.mark.parametrize("scenario,calls,group", DOTTED_CASES)
def test_dotted_names_share_exact_oracle_and_tools(family, scenario, calls, group):
    case = generator.build_cases(family)[f"UNIFIED.{scenario}.{family}"]
    assert case["golden"] == [
        {"kind": "tool_call", "name": name, "arguments": arguments} for name, arguments in calls
    ]
    assert [tool["name"] for tool in case["tools"]] == [name for name, _ in calls]
    for tool, (name, arguments) in zip(case["tools"], calls):
        assert name in case["input"]
        assert tool["parameters"] == {
            "type": "object", "properties": {key: {"type": "string"} for key in arguments},
            "required": list(arguments), "additionalProperties": False,
        }
    assert tax(scenario)[0] == group
    assert generator.scenario_families(scenario) == generator.DOTTED_NAME_FAMILIES
    assert case["init"] == {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None}


def test_report_classifier_rejects_truncated_names_and_missing_results():
    golden = [{"kind": "tool_call", "name": "my-tool.v2", "arguments": {"location": "Beijing"}}]
    assert table._unified_classify("gpt_oss", golden, golden) == "MATCH"
    truncated = [{**golden[0], "name": "my-tool"}]
    assert table._unified_classify("gpt_oss", golden, truncated) != "MATCH"
    assert table._unified_capture_failure(None) == {"unavailable": "No captured result", "unavailable_code": "missing_capture"}
    assert table._unified_capture_failure({"assembled": []}) == {}
    assert table._unified_classify("gpt_oss", golden, []) != "MATCH"


def test_dotted_name_scope_covers_native_tool_families():
    expected = frozenset({
        "beam", "deepseek_v4", "deepseek_v41", "gemma4", "glm47", "gpt_oss",
        "kimi_k2", "kimi_k3", "muse_glimmer", "muse_spark", "nemotron_v3", "qwen3",
    })
    assert generator.DOTTED_NAME_FAMILIES == expected
    assert generator.scenario_families("dotted_name_minimal") == expected
    for family in expected:
        assert all(f"UNIFIED.{scenario}.{family}" in generator.build_cases(family)
                   for scenario in generator.DOTTED_NAME_SCENARIOS)


@pytest.mark.parametrize("family,opener", [
    ("kimi_k3", generator.k3_open("tools")),
    ("deepseek_v41", "<｜DSML｜ calls>"),
])
def test_dotted_shared_prefix_reuses_the_baseline_multi_call_envelope(family, opener):
    cases = generator.build_cases(family)
    baseline = cases[f"UNIFIED.two_calls.{family}"]["input"]
    dotted = cases[f"UNIFIED.dotted_name_shared_prefix.{family}"]["input"]
    assert baseline.count(opener) == dotted.count(opener) == 1
    assert dotted.index("server.weather") < dotted.index("server.time")


def test_rendered_report_preserves_dotted_v2_names_and_missing_v1_captures(rendered_page):
    model = hydrate_page(load_model(rendered_page))
    tab = next(tab for tab in model["tabs"] if tab["id"] == "tab-unified")
    for family in sorted(generator.DOTTED_NAME_FAMILIES):
        family_row = next(row for row in tab["rows"] if row["family"] == family)
        cells = leaf_cells(family_row)
        for scenario, calls, _ in DOTTED_CASES:
            candidate = next(candidate for candidate in cells[scenario]["tooltip"]["candidates"] if candidate["key"] == "dynamo")
            assert candidate["block"]["events"] == [
                {"kind": "tool_call", "name": name, "arguments": arguments} for name, arguments in calls
            ], (family_row["family"], scenario)
            assert candidate["block"]["verdict"] == "MATCH"
            assert cells[scenario]["cmp"]["dynamo"]["na"] == 0
            historical_v1 = [
                candidate for candidate in cells[scenario]["tooltip"]["candidates"]
                if candidate["key"].startswith(("dynamo_v1@", "dynamo_v1_jail@"))
            ]
            assert historical_v1
            for candidate in historical_v1:
                assert candidate["block"]["unavailable_code"] == "missing_capture"
                assert "events" not in candidate["block"]
    row = next(row for row in tab["rows"] if row["family"] == "gpt_oss")
    cell = leaf_cells(row)["dotted_name_mixed_punctuation"]
    candidates = {candidate["key"]: candidate for candidate in cell["tooltip"]["candidates"]}
    assert candidates["dynamo"]["block"]["events"] == [
        {"kind": "tool_call", "name": "my-tool.v2", "arguments": {"location": "Beijing"}}
    ]
    assert candidates["dynamo"]["block"]["verdict"] == "MATCH"
    assert candidates["dynamo"]["parse_mode"] == "unified"
    assert "UnifiedParser" in candidates["dynamo"]["label"]
    assert candidates["dynamo@0.7.27"]["parse_mode"] == "split"
    assert "split" in candidates["dynamo@0.7.27"]["label"]
    assert candidates["dynamo@0.7.28"]["parse_mode"] == "unified"
    assert "UnifiedParser" in candidates["dynamo@0.7.28"]["label"]
    current = next(candidate for candidate in tab["candidates"] if candidate["key"] == "dynamo")
    assert table._version_of_label(candidates["dynamo"]["label"]) == current["version"]
    missing = candidates["vllm_python_batch@0.26.0"]["block"]
    assert missing["unavailable_code"] == "missing_capture"
    assert "events" not in missing
