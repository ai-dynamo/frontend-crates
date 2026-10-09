# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify authored framing stimuli survive YAML serialization and chunking."""
import sys
from pathlib import Path

import yaml
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import gen_unified_golden as generator
import gen_dsml_toolcalling_cases as legacy_generator


def test_new_framing_inputs_survive_golden_yaml_emission() -> None:
    for family in generator.FAMILIES:
        authored = generator.build_cases(family)
        emitted = yaml.safe_load(generator.emit_yaml(family))["cases"]
        for case_id, case in authored.items():
            assert emitted[case_id]["input"] == case["input"]
            if "input_chunks" in case:
                assert emitted[case_id]["input_chunks"] == case["input_chunks"]
                assert "".join(emitted[case_id]["input_chunks"]) == emitted[case_id]["input"]


def test_registered_kimi_k3_legacy_controls_are_available(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tree = tmp_path / "toolcalling"
    for suite in ["fixtures-batch-v1", "fixtures-stream-v1"]:
        (tree / suite / "inputs/kimi_k3").mkdir(parents=True)
    monkeypatch.setattr(legacy_generator, "TREE", tree)
    legacy_generator.main()
    cases = yaml.safe_load((tree / "fixtures-batch-v1/inputs/kimi_k3/TOOLCALLING.batch.framing-controls.yaml").read_text())["cases"]
    assert set(cases) == {"TOOLCALLING.batch.3.a", "TOOLCALLING.batch.7.x"}
    assert all("dynamo_v1" not in case.get("unavailable", {}) for case in cases.values())
