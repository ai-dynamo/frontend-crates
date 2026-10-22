# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from append_stream_regression_cases import _append  # noqa: E402
from stream_regression_cases import (
    PARALLEL_BOUNDARY_CASES,
    UNIFIED_PARALLEL_BOUNDARY_FAMILIES,
    parallel_boundary_source,
)  # noqa: E402
import gen_unified_golden as unified  # noqa: E402


def test_parallel_boundaries_cover_the_registry_with_authored_oracles():
    registry = yaml.safe_load((Path(__file__).parents[1] / "src/parser_families.yaml").read_text())["families"]
    for case_id, families in PARALLEL_BOUNDARY_CASES.items():
        assert set(families) == set(registry)
        expected_names = ["f", "g"] if case_id.endswith(".d") else ["f", "g", "h"]
        for case in families.values():
            assert [call["name"] for call in case["golden"]["calls"]] == expected_names
            assert case["golden"]["normal_text"] == ""
            assert case["chunks"][-1]["delta_text"] == ""
            assert case["chunks"][-1]["finish_reason"] == "tool_calls"
            assert [tool["name"] for tool in case["tools"]] == expected_names


def test_parallel_boundaries_do_not_rewrite_existing_family_members(tmp_path):
    # Kimi and Muse ordinarily share aggregate files: the regression must append
    # a dedicated member so every previously archived YAML byte stays immutable.
    for family in ("kimi_k3", "muse_glimmer", "gemma4"):
        family_root = tmp_path / family
        family_root.mkdir()
        original = family_root / "TOOLCALLING.streamv1.yaml"
        original.write_text(f"family: {family}\nmode: streamv1\ncases: {{}}\n")
        before = original.read_bytes()
        for case_id, cases in PARALLEL_BOUNDARY_CASES.items():
            _append(tmp_path, {family: cases[family]}, case_id)
        added = family_root / "TOOLCALLING.streamv1.50.parallel.yaml"
        first = added.read_bytes()
        for case_id, cases in PARALLEL_BOUNDARY_CASES.items():
            _append(tmp_path, {family: cases[family]}, case_id)
        assert original.read_bytes() == before
        assert added.read_bytes() == first
        assert set(yaml.safe_load(first)["cases"]) == set(PARALLEL_BOUNDARY_CASES)


def test_pythonic_parallel_calls_share_one_list():
    for families in PARALLEL_BOUNDARY_CASES.values():
        text = "".join(chunk["delta_text"] for chunk in families["pythonic"]["chunks"])
        assert text.startswith("[f(") and text.endswith(")]")
        assert text.count("[") == text.count("]") == 1
        assert "), g(" in text


def test_unified_parallel_boundaries_preserve_the_authored_stream_stimulus():
    for family in UNIFIED_PARALLEL_BOUNDARY_FAMILIES:
        cases = unified.build_cases(family)
        for suffix, scenario in (("d", "adjacent_calls_partial_header"), ("e", "three_calls_partial_header")):
            assert unified.scenario_families(scenario) == UNIFIED_PARALLEL_BOUNDARY_FAMILIES
            case = cases[f"UNIFIED.{scenario}.{family}"]
            source = parallel_boundary_source(f"TOOLCALLING.streamv1.50.{suffix}", family)
            assert case["golden"] == [{"kind": "tool_call", **call} for call in source["golden"]["calls"]]
            assert case["tools"] == source["tools"]
            assert "".join(case["input_chunks"]) == case["input"]
            if family == "deepseek_v41":
                assert case["input"].count("<｜DSML｜ calls>") == 1
                assert any(chunk.endswith('<｜DSML｜ invoke na') for chunk in case["input_chunks"])
            else:
                expected_chunks = [chunk["delta_text"] for chunk in source["chunks"]]
                if family == "nemotron_v3":
                    expected_chunks[0] = "</think>" + expected_chunks[0]
                assert case["input_chunks"] == expected_chunks


def test_staged_unified_generator_preserves_parallel_boundaries(tmp_path):
    # Direct imports cannot detect a dependency omitted by the renderer's stage.
    probe = """import json
import gen_unified_golden as g
print(json.dumps({f: {k: c for k, c in g.build_cases(f).items()
                     if 'partial_header' in k} for f in g.FAMILIES}, sort_keys=True))
"""
    env = {key: os.environ[key] for key in (
        "PATH", "HOME", "TMPDIR", "TMP", "TEMP", "DYN_SCRATCH_ROOT",
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
