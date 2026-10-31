# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""An audit must distinguish unsupported evidence from a measured comparison."""

from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
import gzip
import json
from types import SimpleNamespace

import pytest

from audit_unified_consistency import apply_review_evidence, differing_contracts, history_inventory, observation_row, report_family, retained_evidence_inventory


def compare(case, record):
    return observation_row(case, record, implementation="dynamo_v2", version="0.9.0")["status"]


def request_case(events):
    return {"request": {"input": "changed request", "init": {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None}, "tools": [],
                        "finish_reason": "stop", "chunks": []},
            "golden": {"assembled": events}}


def captured(case, events):
    return {"stimulus": {"inline": deepcopy(case["request"])},
            "observation": {"value": {"assembled": events}}, "document": {}}


def test_unbound_old_output_cannot_vote_as_a_matching_observation():
    case = request_case([])
    record = captured(case, [])
    record["stimulus"] = {"unavailable": {"code": "original_request_not_retained"}}
    assert compare(case, record) == "request_binding_unavailable"
    record["stimulus"] = {"inline": {**case["request"], "input": "old request"}}
    assert compare(case, record) == "request_mismatch"
    assert compare(case, captured(case, [])) == "matches_golden"


def test_inherited_observation_keeps_its_measured_producer_and_parser_path():
    case = request_case([])
    record = captured(case, [])
    record["document"] = {"captured_with": {"dynamo_v2": "0.7.30"},
                          "capture_origin": {"crate_version": "0.7.30"},
                          "inherited_from": "0.7.30", "record_metadata": {"parser_path": "split"}}
    row = observation_row(case, record, implementation="dynamo_v2", version="0.7.32")
    assert row["status"] == "matches_golden"
    assert row["selection_version"] == "0.7.32"
    assert row["producer_version"] == row["primary_producer_version"] == "0.7.30"
    assert row["inherited_from"] == "0.7.30" and row["parser_path"] == "split"


def test_legacy_stream_observation_is_not_counted_as_native_batch():
    case = request_case([])
    record = captured(case, [])
    record["document"] = {"captured_with": {"dynamo_v1": "8.0.0"}}
    row = observation_row(case, record, implementation="dynamo_v1", version="8.0.0")
    assert row["selection_mode"] == "batch"
    assert row["mode"] == "legacy-stream"


def test_inherited_checkpoint_labels_do_not_add_measured_executions(tmp_path):
    case = request_case([])
    record = captured(case, [])
    record["document"] = {"captured_with": {"dynamo_v2": "0.7.30"},
                          "capture_origin": {"crate_version": "0.7.30", "job_sha256": "a" * 64}}
    captures = {f"dynamo_v2-{version}": {"runtime_version": version} for version in ("0.7.30", "0.7.32")}
    snapshots = {identity: {"tool_only": deepcopy(record)} for identity in captures}
    snapshots["dynamo_v2-0.7.32"]["tool_only"]["document"]["inherited_from"] = "0.7.30"

    def resolved(identities=None, *, include_members=False):
        return iter((identity, records) for identity, records in snapshots.items() if identities is None or identity in identities)

    history = SimpleNamespace(captures=captures, iter_resolved_snapshots=resolved,
                              snapshot_members={identity: {} for identity in captures})
    store = SimpleNamespace(histories={("beam", "dynamo_v2"): history})
    families = {"beam": {"tool_only": [{"family": "beam", "case_id": "tool_only", "case": case}]}}
    summary = history_inventory(store, families, ["tool_only"], tmp_path / "history.jsonl.gz")
    assert summary["combinations"] == 2
    assert summary["observation_counts"] == {"matches_golden": 2}
    assert summary["unique_observation_evidence"] == summary["unique_bound_executed_evidence"] == 1


def test_distinct_producer_apis_with_identical_results_remain_distinct_evidence(tmp_path):
    case = request_case([])
    histories = {}
    for implementation in ("dynamo_v1", "dynamo_v1_jail"):
        identity = f"{implementation}-10.2.3"
        record = captured(case, [])
        record["document"] = {"captured_with": {implementation: "10.2.3"}}
        history = SimpleNamespace(
            captures={identity: {"runtime_version": "10.2.3"}}, snapshot_members={identity: {}},
            iter_resolved_snapshots=lambda identities=None, record=record, identity=identity, **options:
                iter([(identity, {"tool_only": record})]),
        )
        histories["beam", implementation] = history
    families = {"beam": {"tool_only": [{"family": "beam", "case_id": "tool_only", "case": case}]}}
    summary = history_inventory(SimpleNamespace(histories=histories), families, ["tool_only"], tmp_path / "history.jsonl.gz")
    assert summary["unique_bound_executed_evidence"] == 2


@pytest.mark.parametrize("metadata_owner", ["record_metadata", "document"])
def test_retired_split_records_without_requests_remain_in_retained_inventory(tmp_path, metadata_owner):
    record = {"stimulus": {"unavailable": {"code": "original_request_not_retained"}},
              "observation": {"value": {"assembled": []}},
              "document": {"captured_with": {"dynamo_v2": "0.1.23"}, "record_metadata": {"parser_path": "split"}}}
    if metadata_owner == "document":
        record["document"]["parser_path"] = record["document"].pop("record_metadata")["parser_path"]
    identity = "dynamo_v2-0.1.23"
    history = SimpleNamespace(
        family=SimpleNamespace(cases={"retired__50_a": {"lifecycle": "retired", "scenario": None}}),
        captures={identity: {"runtime_version": "0.1.23"}}, snapshot_members={identity: {"retired__50_a": "original.yaml"}},
        iter_resolved_snapshots=lambda **options: iter([(identity, {"retired__50_a": record})]),
    )
    path = tmp_path / "retained.jsonl.gz"
    summary = retained_evidence_inventory([("history", SimpleNamespace(histories={("deepseek_v4", "dynamo_v2"): history}))], {"deepseek_v4"}, path)
    with gzip.open(path, "rt") as stream:
        rows = [json.loads(line) for line in stream]
    assert summary["records"] == summary["counts"]["retired"] == 1
    assert rows[0]["parser_path"] == "split" and rows[0]["scenario"] is None
    assert rows[0]["disposition"] == "historical_request_not_retained"
    assert rows[0]["request_retained"] is False


def test_verified_recapture_binds_selected_observation_and_producer_separately():
    case = request_case([])
    record = captured(case, [{"kind": "text", "text": "old"}])
    receipt = {"implementation": "dynamo_v2", "version": "0.9.0",
               **{key: "a" * 64 for key in ("source_sha256", "adapter_sha256", "job_sha256")}}
    record["document"] = {"captured_with": {"dynamo_v2": "0.9.0"}, "record_metadata": {
        "verified_recapture": {"result": {"capture_input": case["request"], "assembled": [], "chunks": [], "parser_path": "unified"},
                               "provenance": receipt, "resolution": "fresh_execution"}}}
    row = observation_row(case, record, implementation="dynamo_v2", version="0.9.0")
    assert row["status"] == "matches_golden" and row["verified_recapture"]
    assert row["original_observation_sha256"] != row["selected_observation_sha256"]
    assert row["producer_origin"]["source_sha256"] == receipt["source_sha256"]
    assert row["parser_path"] == "unified" and row["recapture_resolution"] == "fresh_execution"
    record["document"]["record_metadata"]["verified_recapture"]["provenance"]["version"] = "9.9.9"
    with pytest.raises(ValueError, match="both checkpoint and inherited"):
        observation_row(case, record, implementation="dynamo_v2", version="0.9.0")


@pytest.mark.parametrize("outcome,status", [
    ({"error": "ValueError: invalid native argument"}, "parser_error"),
    ({"unavailable": "No public native factory", "unavailable_code": "peer_detector_unavailable"}, "pending_unavailability_review"),
])
def test_selected_recapture_reason_is_distinct_from_original_unavailability(outcome, status):
    case = request_case([])
    record = captured(case, [])
    original = {"code": "peer_prefill_unsupported", "reason": "Earlier adapter rejected this mode"}
    record["observation"] = {"unavailable": original}
    receipt = {"implementation": "dynamo_v2", "version": "0.9.0",
               **{key: "a" * 64 for key in ("source_sha256", "adapter_sha256", "job_sha256")}}
    record["document"] = {"captured_with": {"dynamo_v2": "0.9.0"}, "record_metadata": {
        "verified_recapture": {"result": {"capture_input": case["request"], **outcome},
                               "provenance": receipt, "resolution": "fresh_execution"}}}
    before = deepcopy(record)
    row = observation_row(case, record, implementation="dynamo_v2", version="0.9.0")
    assert row["status"] == status
    assert row["selected_outcome"] == outcome
    assert row["unavailable"] == original
    assert record == before


@pytest.mark.parametrize("changed", [
    [{"kind": "tool_call", "name": "lookup", "arguments": '{"n":9007199254740992}'}],
    [{"kind": "tool_call", "name": "lookup", "arguments": '{"n":"9007199254740993"}'}],
    [{"kind": "text", "text": " done"}],
    [{"kind": "text", "text": "done"}, {"kind": "reasoning", "text": "checking"}],
])
def test_comparison_retains_numeric_precision_types_whitespace_and_order(changed):
    events = [{"kind": "tool_call", "name": "lookup", "arguments": {"n": 9007199254740993}}]
    case = request_case(events)
    assert compare(case, captured(case, changed)) == "differs_from_golden"
    raw = [{"kind": "tool_call", "name": "lookup", "arguments": '{"n":9007199254740993}'}]
    assert compare(case, captured(case, raw)) == "matches_golden"


def test_routing_alias_has_one_family_vote_and_nemotron_keeps_its_own():
    scenario = "tool_only"
    case = request_case([])
    entry = {"case": case, "family": "qwen3"}
    other = {"case": request_case([{"kind": "text", "text": "other"}]), "family": "beam"}
    differences = differing_contracts({"qwen3": {scenario: [entry, {**entry, "family": "qwen3_coder"}]},
                                      "beam": {scenario: [other]}}, [scenario])
    assert len(differences) == 1
    assert [row["family_votes"] for row in differences[0]["outcomes"]] == [1, 1]
    assert report_family("qwen3_coder") == report_family("qwen3")
    assert report_family("nemotron_v3") != report_family("qwen3")


def test_stale_review_inventory_is_rejected_before_applying_dispositions():
    audit = {"family_scenario_inventory": [{"request_sha256": "candidate"}]}
    with pytest.raises(ValueError, match="candidate inventory"):
        apply_review_evidence(audit, {}, {"rows": [{"request_sha256": "old"}]}, {})


def test_semantic_coverage_alias_cannot_claim_complete_historical_execution():
    row = {"family": "beam", "scenario": "retained_variant", "status": "pending_applicability"}
    disposition = {"identity": {"family": "beam", "scenario": "retained_variant"},
                   "classification": "resolved_alias", "resolved": True}
    audit = {"family_scenario_inventory": [deepcopy(row)], "differences": [], "totals": {}}
    evidence = {"resolved_aliases": [disposition], "dispositions": [],
                "final_candidate_snapshot": {"new_coverage_aliases": []}}
    apply_review_evidence(audit, evidence, {"rows": [row], "snapshot": {}}, {"dispositions": []})
    assert audit["totals"]["pending_applicability"] == 0
    assert audit["totals"]["missing_authored_pairs"] == 1
    assert audit["complete"] is False
    assert audit["family_scenario_inventory"][0]["disposition"] == disposition


def test_difference_disposition_must_match_the_exact_typed_group():
    audit = {"family_scenario_inventory": [], "differences": [{"identity": "group", "typed_golden": [42]}]}
    applicability = {"resolved_aliases": [], "dispositions": [],
                     "final_candidate_snapshot": {"new_coverage_aliases": []}}
    differences = {"dispositions": [{"identity": "group", "original_group": {"identity": "group", "typed_golden": ["42"]}}]}
    with pytest.raises(ValueError, match="candidate group"):
        apply_review_evidence(audit, applicability, {"rows": [], "snapshot": {}}, differences)


class DecodedCheckpointMembers(Mapping):
    def __init__(self, values):
        self.values = values
        self.decodes = Counter()

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, identity):
        self.decodes[identity] += 1
        # The routed history reader decodes a complete immutable fragment per lookup.
        return deepcopy(self.values[identity])


@pytest.mark.parametrize("surface", ["active_grid", "retained_evidence"])
def test_each_checkpoint_member_fragment_is_decoded_once_per_inventory(tmp_path, surface):
    cases = {scenario: {**request_case([]), "lifecycle": "active", "scenario": scenario}
             for scenario in ("tool_only", "tool_then_content")}
    captures = {f"dynamo_v2-{version}": {"runtime_version": version}
                for version in ("0.9.0", "0.9.1")}
    snapshots = {identity: {scenario: captured(case, []) for scenario, case in cases.items()}
                 for identity in captures}
    member_values = {identity: {scenario: f"{identity}/{scenario}.yaml" for scenario in cases}
                     for identity in captures}
    members = DecodedCheckpointMembers(member_values)

    def resolved(identities=None, *, include_members=False):
        return iter((identity, records) for identity, records in snapshots.items()
                    if identities is None or identity in identities)

    history = SimpleNamespace(captures=captures, iter_resolved_snapshots=resolved,
                              snapshot_members=members, family=SimpleNamespace(cases=cases))
    store = SimpleNamespace(histories={("beam", "dynamo_v2"): history})
    families = {"beam": {scenario: [{"family": "beam", "case_id": scenario, "case": case}]
                         for scenario, case in cases.items()}}
    destination = tmp_path / "inventory.jsonl.gz"
    if surface == "active_grid":
        summary = history_inventory(store, families, sorted(cases), destination)
        assert summary["combinations"] == 4
        assert summary["observation_counts"] == {"matches_golden": 4}
    else:
        summary = retained_evidence_inventory([("history", store)], families, destination)
        assert summary["records"] == 4
    with gzip.open(destination, "rt") as stream:
        rows = [json.loads(line) for line in stream]
    for row in rows:
        item = row["observations"][0] if surface == "active_grid" else row
        assert item["source_member"] == member_values[row["capture"]][row["scenario"]]
    assert members.decodes == {identity: 1 for identity in captures}
