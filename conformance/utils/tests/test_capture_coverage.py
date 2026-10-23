# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from capture_coverage import combine_coverage


@pytest.mark.parametrize("corruption", [None, "duplicate", "missing", "version", "case", "status"])
def test_global_accounting_rejects_incomplete_or_misidentified_members(tmp_path, corruption):
    inventory = tmp_path / "inventory.yaml"
    inventory.write_text('targets:\n- implementation: dynamo_v1\n  version: "1.0.0"\n')
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    (jobs / "family.json").write_text(json.dumps({"family": "family", "cases": [
        {"id": "UNIFIED.case.family", "case_id": "case", "scenario": "case"}]}))
    rows = [{"implementation": implementation, "version": "1.0.0", "family": "family",
             "scenario": "case", "case_id": "case", "status": status}
            for implementation, status in [("dynamo_v1", "executed"), ("dynamo_v1_jail", "source_build_blocker")]]
    if corruption == "duplicate":
        rows.append(rows[0])
    elif corruption == "missing":
        rows.pop()
    elif corruption == "version":
        rows[0]["version"] = "2.0.0"
    elif corruption == "case":
        rows[0]["case_id"] = "altered"
    elif corruption == "status":
        rows[0]["status"] = "captured"
    source = tmp_path / "coverage.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "combined.jsonl"
    if corruption:
        output.write_text("original generation")
        with pytest.raises(ValueError):
            combine_coverage(inventory, jobs, [source], output)
        assert output.read_text() == "original generation"
    else:
        summary = combine_coverage(inventory, jobs, [source], output)
        assert summary["combinations"] == 2
        assert summary["counts"] == {"executed": 1, "source_build_blocker": 1}
        assert summary["source_build_blockers"][0]["implementation"] == "dynamo_v1_jail"
        assert all(json.loads(line)["case_id"] == "UNIFIED.case.family" for line in output.read_text().splitlines())
