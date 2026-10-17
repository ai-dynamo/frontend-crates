# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import gen_unified_golden as generator
import generate_conformance_table as table
from case_variants import leaf_cells
from unified_taxonomy import tax
from validate_conformance_status import load_model


DOTTED_CASES = [
    ("dotted_name_minimal", [("a.b", {})], 1),
    ("dotted_name_mixed_punctuation", [("my-tool.v2", {"location": "Beijing"})], 1),
    ("dotted_name_multiple_segments", [("server.tools.weather", {"location": "Beijing"})], 1),
    ("dotted_name_shared_prefix", [("server.weather", {"location": "Beijing"}),
                                   ("server.time", {"timezone": "Asia/Shanghai"})], 2),
]


@pytest.mark.parametrize("family", generator.FAMILIES)
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
    assert generator.scenario_families(scenario) == frozenset(generator.FAMILIES)
    assert case["init"] == {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None}


def test_report_classifier_rejects_truncated_names_and_missing_results():
    golden = [{"kind": "tool_call", "name": "my-tool.v2", "arguments": {"location": "Beijing"}}]
    assert table._unified_classify("gpt_oss", golden, golden) == "MATCH"
    truncated = [{**golden[0], "name": "my-tool"}]
    assert table._unified_classify("gpt_oss", golden, truncated) != "MATCH"
    assert table._unified_capture_failure(None) == {"unavailable": "No captured result", "unavailable_code": "missing_capture"}
    assert table._unified_capture_failure({"assembled": []}) == {}
    assert table._unified_classify("gpt_oss", golden, []) != "MATCH"


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


def test_rendered_report_shows_published_harmony_name_boundary(rendered_page):
    model = load_model(rendered_page)
    tab = next(tab for tab in model["tabs"] if tab["id"] == "tab-unified")
    for family_row in tab["rows"]:
        cells = leaf_cells(family_row)
        for scenario, calls, _ in DOTTED_CASES:
            candidate = next(candidate for candidate in cells[scenario]["tooltip"]["candidates"] if candidate["key"] == "dynamo")
            assert candidate["block"]["events"] == [
                {"kind": "tool_call", "name": name, "arguments": arguments} for name, arguments in calls
            ], (family_row["family"], scenario)
            assert candidate["block"]["verdict"] == "MATCH"
            assert cells[scenario]["cmp"]["dynamo"]["na"] == 0
    row = next(row for row in tab["rows"] if row["family"] == "gpt_oss")
    cell = leaf_cells(row)["dotted_name_mixed_punctuation"]
    candidates = {candidate["key"]: candidate for candidate in cell["tooltip"]["candidates"]}
    before = candidates["dynamo_v1@10.0.5"]["block"]
    after = candidates["dynamo_v1@10.0.6"]["block"]
    assert before["events"] == [{"kind": "tool_call", "name": "my-tool", "arguments": {"location": "Beijing"}}]
    assert before["verdict"] != "MATCH"
    assert after["events"] == [{"kind": "tool_call", "name": "my-tool.v2", "arguments": {"location": "Beijing"}}]
    assert after["verdict"] == "MATCH"
    assert candidates["dynamo"]["block"]["events"] == after["events"]
    assert candidates["dynamo"]["block"]["verdict"] == "MATCH"
    assert candidates["dynamo"]["parse_mode"] == "split"
    assert "split" in candidates["dynamo"]["label"]
    assert "UnifiedParser" not in candidates["dynamo"]["label"]
    current = next(candidate for candidate in tab["candidates"] if candidate["key"] == "dynamo")
    assert table._version_of_label(candidates["dynamo"]["label"]) == current["version"]
    missing = candidates["vllm_python_batch@0.26.0"]["block"]
    assert missing["unavailable_code"] == "missing_capture"
    assert "events" not in missing
    assert cell["cmp"]["dynamo_v1@10.0.5"]["sig"] != cell["cmp"]["golden"]["sig"]
    assert cell["cmp"]["dynamo_v1@10.0.6"]["sig"] == cell["cmp"]["golden"]["sig"]
