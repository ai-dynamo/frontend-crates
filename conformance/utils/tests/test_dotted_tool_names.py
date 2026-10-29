# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import gen_unified_golden as generator
from unified_taxonomy import tax


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
