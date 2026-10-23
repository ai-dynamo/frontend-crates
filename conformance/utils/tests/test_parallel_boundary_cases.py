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
            assert case["input_chunks"] == expected_chunks
            assert case["input_chunks"][-1] == ""


def test_staged_unified_generator_preserves_parallel_boundaries(tmp_path):
    # Direct imports cannot detect a dependency omitted by the renderer's stage.
    probe = """import json
import gen_unified_golden as g
print(json.dumps({f: {k: c for k, c in g.build_cases(f).items()
                     if 'partial_header' in k} for f in g.FAMILIES}, sort_keys=True))
"""
    env = {key: os.environ[key] for key in (
        "PATH", "HOME", "TMPDIR", "TMP", "TEMP", "DYN_SCRATCH_ROOT", "XDG_CACHE_HOME",
    ) if key in os.environ}
    env.update(STAGE=str(tmp_path / "stage"), STAGED_PYTHON=sys.executable,
               STAGED_PROBE=probe, STAGED_COMMON=str(Path(__file__).parents[1] / "src/_common.sh"))
    result = subprocess.run(
        ["bash", "-c", 'source "$STAGED_COMMON" && _build_stage_base && cd "$STAGE/tests/parity" && "$STAGED_PYTHON" -c "$STAGED_PROBE"'],
        env=env, capture_output=True, text=True, check=True,
    )
    expected = {family: {key: case for key, case in unified.build_cases(family).items()
                         if "partial_header" in key} for family in unified.FAMILIES}
    assert json.loads(result.stdout) == expected
