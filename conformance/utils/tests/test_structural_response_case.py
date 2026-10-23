# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from capture_stimulus import capture_input, capture_peer_results
from capture_unified_batch import capture_cases
from capture_unified_job import _case_job
from gen_unified_golden import build_cases, scenario_families
from unified_taxonomy import numbered_id
import unified_history


SCENARIO = "structured_response_after_reasoning"


def test_structural_case_has_one_shared_id_and_explicit_builder_scope():
    assert scenario_families(SCENARIO) == {"gpt_oss", "qwen3", "glm47", "deepseek_v4"}
    assert numbered_id(SCENARIO) == "UNIFIED.36-structured_response_after_reasoning"
    assert f"UNIFIED.{SCENARIO}.nemotron_v3" not in build_cases("nemotron_v3")
    assert "UNIFIED.guided_response_native_wrapper_whitespace_before_prose.nemotron_v3" in build_cases("nemotron_v3")


def _assert_bound_columns(states, case_id, digest):
    for capture_id, state in states.items():
        assert case_id in state, f"{capture_id}: missing structured request"
        assert state[case_id]["stimulus"]["semantic_sha256"] == digest


def test_structural_request_is_bound_after_each_family_introduces_it(committed_unified_store):
    store = committed_unified_store
    for family in scenario_families(SCENARIO):
        history = store.histories[(family, "dynamo_v2")]
        case_id = next(key for key, case in history.family.cases.items() if case["scenario"] == SCENARIO)
        digest = unified_history._request_digest(history.family.cases[case_id]["request"])
        capture_ids = history.ordered_capture_ids()
        states = {capture_id: history.resolve(capture_id) for capture_id in capture_ids}
        first_capture = next(index for index, capture_id in enumerate(capture_ids)
                             if case_id in states[capture_id])
        assert all(case_id in states[capture_id] for capture_id in capture_ids[first_capture:])
        captured_states = {capture_id: states[capture_id] for capture_id in capture_ids[first_capture:]}
        _assert_bound_columns(captured_states, case_id, digest)
        first_retained = next(iter(captured_states))
        missing = {first_retained: {key: value for key, value in captured_states[first_retained].items()
                                    if key != case_id}}
        with pytest.raises(AssertionError, match="missing structured request"):
            _assert_bound_columns(missing, case_id, digest)


@pytest.mark.parametrize("family", ["gpt_oss", "qwen3", "glm47", "deepseek_v4"])
def test_job_retains_generation_request_and_peers_cannot_claim_parser_only_success(family):
    case = build_cases(family)[f"UNIFIED.{SCENARIO}.{family}"]
    request = capture_input({**case, "chunks": [{"delta_text": case["input"]}]})
    job = _case_job(family, SCENARIO, {
        "scenario": SCENARIO, "request": request, "golden": case["golden"],
    }, mode="batch")
    assert job["init"] == request["init"] == case["init"]
    assert request["tools"] == []
    assert request["init"]["structural_tag"]["rejected"]

    def must_not_execute(*args):
        pytest.fail("parser-only adapter ignored the generation constraint")

    stream = capture_peer_results([job], {family}, must_not_execute, tools=[],
                                  supports_init=True, supports_tools=True, supports_finish=True)
    batch = capture_cases({"cases": [job]}, {family: ("reasoning", "tools")},
                          lambda names: True, must_not_execute, supports_reasoning_init=True)
    for results in [stream, batch]:
        result = results[job["id"]]
        assert result["capture_input"] == request
        assert "structural-tag" in result["unavailable"]
        assert "assembled" not in result and "chunks" not in result
