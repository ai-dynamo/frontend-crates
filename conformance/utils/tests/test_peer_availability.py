# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Peer availability follows the selected observation without decoding browser outputs."""
import copy

import viewer_catalog


def test_index_uses_all_admitted_peer_captures_and_extends_history():
    latest = {"snapshots": {("deepseek_v4", "dynamo_v2"): [{"recorded_cases": ["arg_marker_in_string"]}],
                            ("deepseek_v4", "sglang_python"): [{"recorded_cases": ["text_only"]}]}}
    expected = {"schema": 1, "families": {"deepseek_v4": ["text_only"]}}
    assert viewer_catalog._peer_availability(latest) == expected
    history = copy.deepcopy(latest)
    history["snapshots"]["deepseek_v4", "vllm_rust"] = [{"recorded_cases": ["arg_marker_in_string", "text_only"]}]
    assert viewer_catalog._peer_availability(history)["families"]["deepseek_v4"] == ["arg_marker_in_string", "text_only"]
    assert viewer_catalog._peer_availability(latest) == expected


def test_recorded_cases_follow_effective_recapture_and_keep_errors():
    def record(observation, result=None):
        metadata = {} if result is None else {"verified_recapture": {"result": result}}
        return {"document": {"record_metadata": metadata}, "observation": observation}
    absent = {"unavailable": {"code": "missing_capture"}}
    records = {"ordinary": record({"value": {}}), "error": record({"error": {}}),
               "missing": record(absent), "fresh_error": record(absent, {"error": "failed", "capture_observation": {"error": {}}}),
               "fresh_stream": record(absent, {"chunks": []}), "fresh_batch": record(absent, {"assembled": []}),
               "fresh_missing": record({"value": {}}, {"unavailable": "not recorded"})}
    summary = viewer_catalog._record_snapshot_metadata(records)
    assert summary["recorded_cases"] == ["error", "fresh_batch", "fresh_error", "fresh_stream", "ordinary"]
