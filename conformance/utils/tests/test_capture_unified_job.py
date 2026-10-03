# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import capture_unified_job


ROOT = Path(__file__).resolve().parents[2] / "fixtures-unified-v2" / "families"


def test_existing_family_job_is_stable_and_preserves_request_shape():
    job = capture_unified_job.build_job(ROOT, "gemma4")
    assert job["family"] == "gemma4"
    assert len(job["cases"]) == 99
    assert [case["id"] for case in job["cases"]].count("UNIFIED.tool_only.gemma4") == 1
    case = next(
        case
        for case in job["cases"]
        if case["init"]["starting_state"] == "Response"
        and case["init"]["tool_output_mode"] == "Native"
    )
    assert case["init"]["starting_state"] == "Response"
    assert case["finish_reason"] == "stop"
    assert case["chunks"]
    assert all(isinstance(chunk, str) for chunk in case["chunks"])


def test_job_builder_rejects_mismatched_family_metadata(tmp_path):
    family_dir = tmp_path / "gemma4"
    family_dir.mkdir()
    (family_dir / "inputs_and_golden.yaml").write_text(
        "input_document:\n  family: qwen3\n  mode: unified\ncases: {}\n"
    )
    with pytest.raises(ValueError, match="does not match"):
        capture_unified_job.build_job(tmp_path, "gemma4")


def test_job_builder_rejects_non_text_chunk(tmp_path):
    family_dir = tmp_path / "gemma4"
    family_dir.mkdir()
    (family_dir / "inputs_and_golden.yaml").write_text(
        """input_document:
  family: gemma4
  mode: unified
cases:
  bad:
    scenario: bad
    request:
      input: hi
      init: {starting_state: None, tool_output_mode: Native, named_tool: null}
      finish_reason: stop
      tools: []
      chunks: [{delta_text: 1}]
"""
    )
    with pytest.raises(ValueError, match="string delta_text"):
        capture_unified_job.build_job(tmp_path, "gemma4")
