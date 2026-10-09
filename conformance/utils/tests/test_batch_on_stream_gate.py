# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parity annotations must preserve independent oracles and visible defects."""
import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import generate_conformance_table as table


def result(value):
    return {"calls": [{"name": "probe", "arguments": {"value": value}}], "normal_text": ""}


def configure(monkeypatch, cid, actual, note_key, note):
    monkeypatch.setattr(table, "_load_stream_on_batch_overlay", lambda: {("deepseek_v4", cid): {"dynamo_v2": actual}})
    monkeypatch.setattr(table, "_stream_on_batch_versions", lambda: {"dynamo_v2": "0.7.20"})
    monkeypatch.setattr(table, "_known_divergence_note", lambda family, case, key: note if case == cid and key == note_key else None)


def batch_case(cid, batch, golden=None):
    case = {"__family": "deepseek_v4", "__case_id": cid, "description": "probe", "model_text": "wire",
            "expected": {"dynamo_v1": batch},
            "__ver_status": {"dynamo_v1": {"10-0-1": {"version": "10.0.1", "block": batch}}}}
    if golden is not None:
        case["golden"] = golden
    return {("deepseek_v4", "7.p.probe"): case}


def rendered(case):
    return table._toolcalling_cell_model(case, "batch", "deepseek_v4", "7.p.probe", "batch", "cross_engine", lambda href: href)


def test_baseline_defect_keeps_mismatch_without_intended_difference_badge(monkeypatch):
    cid = "TOOLCALLING.batch.11.b"
    actual = {"calls": [{"name": "", "arguments": {"value": 1}}], "normal_text": ""}
    note = "FIXME (v2 defect): DS4 emits an empty tool name; exact baseline retained."
    configure(monkeypatch, cid, actual, "baseline_defect", note)
    cases = batch_case(cid, {"calls": [], "normal_text": ""})
    table._attach_merged_cmp(cases)
    cell = rendered(next(iter(cases.values())))
    assert not cell["known_divergence"]
    assert cell["cmp"]["dynamo_v1-b-10-0-1"]["sig"] != cell["cmp"]["dynamo_v2-s-0-7-20"]["sig"]
    assert any(note in diagnostic for _, diagnostic in cell["tooltip"]["dynamo_notes"])
    actual_candidate = next(candidate for candidate in cell["tooltip"]["candidates"] if candidate["key"] == "dynamo_v2-s-0-7-20")
    assert not actual_candidate["block"].get("explanation")


@pytest.mark.parametrize("expected,wrong", [("  spaced  ", "spaced"), ("&lt;", "<"), ("null", None), (42, "42")])
def test_documented_difference_does_not_rewrite_golden_or_hide_wrong_capture(monkeypatch, expected, wrong):
    cid = "TOOLCALLING.batch.7.p.probe"
    golden = result(expected)
    # Model a wrong frozen-v1 capture and a correct v2 result.
    configure(monkeypatch, cid, golden, "stream_vs_batch", "FIXME (v1 defect): v2 preserves the declared value.")
    cases = batch_case(cid, result(wrong), copy.deepcopy(golden))
    table._attach_merged_cmp(cases)
    case = next(iter(cases.values()))
    cell = rendered(case)
    assert case["golden"] == golden
    assert cell["red_on_diff"]
    assert cell["known_divergence"]
    assert cell["cmp"]["golden"]["sig"] == cell["cmp"]["dynamo_v2-s-0-7-20"]["sig"]
    assert cell["cmp"]["golden"]["sig"] != cell["cmp"]["dynamo_v1-b-10-0-1"]["sig"]


def test_newly_wrong_v2_capture_still_differs_from_the_authored_golden(monkeypatch):
    cid = "TOOLCALLING.batch.7.p.probe"
    golden = result(42)
    configure(monkeypatch, cid, result("wrong"), "stream_vs_batch", "Documented parity difference.")
    cases = batch_case(cid, result("42"), golden)
    table._attach_merged_cmp(cases)
    cell = rendered(next(iter(cases.values())))
    assert cell["red_on_diff"]
    assert cell["cmp"]["golden"]["sig"] != cell["cmp"]["dynamo_v2-s-0-7-20"]["sig"]
