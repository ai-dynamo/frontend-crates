# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys

from conformance.utils.tests import MODEL_CONTRACTS
from capture_versions import capture_release_sort_key
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from capture_stimulus import capture_input, capture_peer_results
from capture_unified_batch import capture_cases
from capture_unified_job import _case_job
from gen_unified_golden import build_cases, scenario_families
from unified_taxonomy import numbered_id
import unified_history
from test_unified_taxonomy_covers_corpus import retained_unified_snapshots


SCENARIO = "structured_response_after_reasoning"
HISTORY_CONTRACT = MODEL_CONTRACTS["structuralRequestHistory"]


def test_structural_case_has_one_shared_id_and_explicit_builder_scope():
    assert scenario_families(SCENARIO) == {"gpt_oss", "qwen3", "glm47", "deepseek_v4", "muse_spark"}
    assert numbered_id(SCENARIO) == "UNIFIED.36-structured_response_after_reasoning"
    assert f"UNIFIED.{SCENARIO}.nemotron_v3" not in build_cases("nemotron_v3")
    assert "UNIFIED.guided_response_native_wrapper_whitespace_before_prose.nemotron_v3" in build_cases("nemotron_v3")


def _assert_bound_columns(states, case_id, digest):
    for capture_id, state in states.items():
        assert case_id in state, f"{capture_id}: missing structured request"
        assert state[case_id]["stimulus"]["semantic_sha256"] == digest


def _requires_structural_capture(family, version):
    return (version in HISTORY_CONTRACT["versions"][family]
            or capture_release_sort_key(version) >= capture_release_sort_key(HISTORY_CONTRACT["futureFrom"]))


def _assert_required_resources(resources):
    present = {(row["family"], row["version"]) for row in resources
               if row["kind"] == "history" and row["implementation"] == "dynamo_v2"}
    required = {(family, version) for family, versions in HISTORY_CONTRACT["versions"].items() for version in versions}
    assert required <= present, f"missing pinned structural captures: {sorted(required - present)}"


def test_structural_request_is_bound_in_every_retained_dynamo_column(historical_unified_sources):
    root = Path(__file__).resolve().parents[3] / "conformance/fixtures-unified-v2"
    store = unified_history.load_authored_store(root)
    expected = {}
    for family in scenario_families(SCENARIO):
        authored = store.families[family]
        case_id = next(key for key, case in authored.cases.items() if case["scenario"] == SCENARIO)
        expected[family] = (case_id, unified_history._request_digest(authored.cases[case_id]["request"]))
    history_root, resources = historical_unified_sources
    _assert_required_resources(resources)
    checked = set()
    snapshots = 0
    for resource, document in retained_unified_snapshots(store, families=expected, source_root=history_root, resources=resources):
        snapshots += 1
        family = resource["family"]
        if family not in expected:
            continue
        case_id, digest = expected[family]
        states = {resource["path"]: {key: record["value"] for key, record in document["records"].items()}}
        # Expanded archives predate this authored case. Preserve their absent
        # evidence; main's pinned historical captures and future captures stay mandatory.
        if case_id in states[resource["path"]] or _requires_structural_capture(family, resource["version"]):
            _assert_bound_columns(states, case_id, digest)
        checked.add(family)
    assert checked == set(expected)
    print(f"structural history qualification passed: {snapshots} Dynamo snapshots across {len(checked)} families")


def test_structural_request_guard_rejects_missing_oldest_checkpoint():
    with pytest.raises(AssertionError, match="oldest: missing structured request"):
        _assert_bound_columns({"oldest": {}, "newest": {SCENARIO: {"stimulus": {"semantic_sha256": "digest"}}}}, SCENARIO, "digest")


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


def test_structural_history_contract_rejects_missing_resource_or_changed_binding():
    with pytest.raises(AssertionError, match="missing pinned structural captures"):
        _assert_required_resources([])
    assert _requires_structural_capture("deepseek_v4", "0.1.23")
    assert _requires_structural_capture("deepseek_v4", "0.7.29")
    assert not _requires_structural_capture("deepseek_v4", "0.1.0")
    with pytest.raises(AssertionError):
        _assert_bound_columns({"future": {}}, SCENARIO, "digest")
    with pytest.raises(AssertionError):
        _assert_bound_columns({"oldest": {SCENARIO: {"stimulus": {"semantic_sha256": "changed"}}}}, SCENARIO, "digest")
