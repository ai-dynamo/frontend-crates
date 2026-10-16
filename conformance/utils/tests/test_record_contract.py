# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import capture_stimulus
import unified_history


def test_mixed_legacy_and_adapter_batch_observations_preserve_their_contracts(tmp_path):
    family = unified_history.Family("sample", {"cases": {}}, tmp_path / "inputs_and_golden.yaml")
    final = {"reasoning_text": "", "normal_text": "hello", "calls": []}
    records = {
        "legacy": {"case_key": "UNIFIED.legacy", "stimulus": {"ref": "current"}, "observation": {"value": {"chunks": [[{"kind": "text", "text": "legacy"}]]}}},
        "batch": {"case_key": "UNIFIED.batch", "stimulus": {"ref": "current"}, "observation": {"value": {"assembled": capture_stimulus.batch_final_events(final)}}},
    }
    checkpoint = "dynamo_v1-10.0.1"
    capture = {"runtime_version": "10.0.1", "provenance": {"status": "captured", "origin": {"crate_version": "10.0.1", "source_sha256": "a" * 64, "adapter_sha256": "b" * 64, "job_sha256": "c" * 64}},
               "document": {}, "changes": records, "metadata_changes": {"batch": {"batch_final": final}},
               "document_overrides": {"legacy": {"capture_provenance": {"captured_with": {"dynamo_v1": "10.0.1"}, "capture_origin": None, "inherited_from": None}}}}
    history = unified_history.History(family, "dynamo_v1", {checkpoint: capture}, tmp_path, {checkpoint: tmp_path / (checkpoint + ".yaml")})
    resolved = history.resolve(checkpoint)
    assert resolved["legacy"]["observation"] == records["legacy"]["observation"]
    assert "capture_origin" not in resolved["legacy"]["document"]
    assert capture_stimulus.unified_record_contract("dynamo_v1", resolved["legacy"]["document"]) == "legacy-stream"
    assert capture_stimulus.unified_record_contract("dynamo_v1", resolved["batch"]["document"]) == "batch"
    frozen = copy.deepcopy(capture)
    for change in ("missing_final", "wrong_projection", "chunks"):
        history.captures[checkpoint] = copy.deepcopy(frozen)
        bad = history.captures[checkpoint]
        if change == "missing_final":
            bad["metadata_changes"] = {}
        elif change == "wrong_projection":
            bad["changes"]["batch"]["observation"]["value"]["assembled"] = []
        else:
            bad["changes"]["batch"]["observation"]["value"]["chunks"] = []
        with pytest.raises(ValueError, match="batch"):
            history.resolve(checkpoint)


def test_adapter_batch_contract_rejects_a_partial_producer_identity():
    with pytest.raises(ValueError, match="adapter and job"):
        capture_stimulus.unified_record_contract("dynamo_v1", {"capture_origin": {"adapter_sha256": "a" * 64}})
