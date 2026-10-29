# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import capture_unified_job
import impls


ROOT = Path(__file__).resolve().parents[2] / "fixtures-unified-v2" / "families"


def test_batch_cli_emits_bound_requests_and_golden(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["capture_unified_job.py", "--family", "gemma4", "--mode", "batch"])
    capture_unified_job.main()
    job = json.loads(capsys.readouterr().out)
    assert job == capture_unified_job.build_job(ROOT, "gemma4", mode="batch")
    assert all({"request", "golden", "case_id"} <= case.keys() for case in job["cases"])
    assert any(case["request"]["init"]["starting_state"] == "Reasoning" for case in job["cases"])


def test_existing_family_job_is_stable_and_preserves_request_shape():
    job = capture_unified_job.build_job(ROOT, "gemma4")
    assert job["family"] == "gemma4"
    assert len(job["cases"]) == 270
    assert {
        f"UNIFIED.{scenario}.gemma4"
        for scenario in (
            "guided_json_native_parameter_object_before_payload",
            "guided_json_native_parameter_array_before_payload",
            "guided_json_native_parameter_string_before_payload",
            "guided_json_noncall_before_native_array_before_payload",
            "guided_response_native_wrapper_whitespace_before_prose",
            "gemma4_guided_response_extra_body_opener",
            "adjacent_calls_partial_header",
            "three_calls_partial_header",
            "xml_string_type_single_quoted_object",
            "xml_string_type_double_quoted_object",
            "arg_object_quoted_parameter_close",
            "arg_array_quoted_parameter_close",
        )
    } <= {case["id"] for case in job["cases"]}
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


def test_capture_registry_keeps_historical_pipeline_names_and_jail_scope():
    assert impls.VLLM_UNIFIED_PIPELINES["deepseek_v41"] == ("deepseek_v4", "deepseek_v4")
    assert impls.SGLANG_UNIFIED_PIPELINES["deepseek_v41"] == ("deepseek-v41", "deepseekv41")
    assert impls.DYNAMO_UNIFIED_PIPELINES["glm47"] == ("glm45", "glm47")
    assert "muse_glimmer" not in impls.DYNAMO_UNIFIED_PIPELINES
    assert impls.DYNAMO_JAIL_UNIFIED_PIPELINES["muse_glimmer"] == ("muse", "muse")
    assert impls.VLLM_TOKENIZER_MARKERS.keys() == impls.VLLM_UNIFIED_PIPELINES.keys()
