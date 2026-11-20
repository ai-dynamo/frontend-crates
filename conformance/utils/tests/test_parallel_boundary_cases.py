# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import os
import subprocess
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import gen_unified_golden as unified  # noqa: E402


def test_unified_parallel_boundaries_preserve_the_authored_stream_stimulus():
    # These earlier publications split the same bytes at different boundaries.
    retained_lengths = {
        ("adjacent_calls_partial_header", "muse_spark"): (159, 95, 0),
        ("three_calls_partial_header", "beam"): (0, 144, 219, 0),
        ("three_calls_partial_header", "gpt_oss"): (0, 155, 148, 0),
        ("three_calls_partial_header", "minimax_m3"): (0, 187, 218, 0),
        ("three_calls_partial_header", "muse_spark"): (0, 159, 176, 0),
    }
    for scenario, families in unified.PARALLEL_BOUNDARY_CASES.items():
        expected_names = ["f", "g"] if scenario == "adjacent_calls_partial_header" else ["f", "g", "h"]
        assert set(families) == unified.UNIFIED_PARALLEL_BOUNDARY_FAMILIES
        # Spark inherits Glimmer's ATEM inputs, schemas and authored chunk boundaries.
        inherited = {"muse_spark"} if "muse_glimmer" in families else set()
        assert unified.scenario_families(scenario) == set(families) | inherited
        for family in set(families) | inherited:
            source = families["muse_glimmer" if family in inherited else family]
            case = unified.build_cases(family)[f"UNIFIED.{scenario}.{family}"]
            assert case["golden"] == source["golden"]
            assert [event["name"] for event in case["golden"]] == expected_names
            assert [tool["name"] for tool in case["tools"]] == expected_names
            assert case["tools"] == source["tools"]
            assert "".join(case["input_chunks"]) == case["input"]
            expected_chunks = list(source["input_chunks"])
            if family == "nemotron_v3":
                expected_chunks[0] = "</think>" + expected_chunks[0]
            lengths = retained_lengths.get((scenario, family))
            if lengths is not None:
                assert tuple(map(len, case["input_chunks"])) == lengths
                assert case["input"] == "".join(expected_chunks)
            else:
                assert case["input_chunks"] == expected_chunks
            assert case["input_chunks"][-1] == ""
