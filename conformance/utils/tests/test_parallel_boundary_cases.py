# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from append_stream_regression_cases import _append  # noqa: E402
from stream_regression_cases import PARALLEL_BOUNDARY_CASES  # noqa: E402


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
