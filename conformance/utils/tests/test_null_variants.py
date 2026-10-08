# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Schema variants retain independent oracles and observations when grouped."""

import copy
import sys
from pathlib import Path

import pytest

from conformance.utils.tests.schema_oracle import matches_schema

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from case_variants import aggregate_cells, group_null_variants, leaf_cells
from gen_null_stream_cases import FAMILIES, build_cases, native_call
from null_cases import NULL_DESCRIPTIONS, NULL_VARIANTS
from numeric_cases import NumericLiteral
from validate_conformance_status import cell_state


def make_cell(label, sig=1, missing=False):
    return {"kind": "cell", "status": "ok", "sub": label,
            "case_id": "UNIFIED." + label, "family": "glm47",
            "cmp": {"golden": {"sig": 1, "na": 0, "err": 0, "leak": 0},
                    "dynamo": {"sig": sig, "na": int(missing), "err": 0, "leak": 0}},
            "tooltip": {"head": "UNIFIED." + label, "description": label,
                        "input": {"kind": "text", "text": "null"}, "candidates": []}}


@pytest.mark.parametrize("signatures,missing,expected", [
    ([1, 1], [False, False], "green"),
    ([1, 2], [False, False], "red"),
    ([1, 1], [False, True], "empty"),
    ([2, 1], [False, True], "red"),
])
def test_group_does_not_hide_failures_or_missing_captures(signatures, missing, expected):
    cells = [make_cell(label, sig, absent) for label, sig, absent
             in zip(("7-4", "7-4.anyof"), signatures, missing)]
    before = copy.deepcopy(cells)
    result = aggregate_cells(cells, "7-4", NULL_DESCRIPTIONS["7-4"])
    assert cell_state(result, {"key": "dynamo", "label": "Dynamo"})[0] == expected
    assert result["variants"] == before == cells
    assert result["tooltip"]["variants"] == [cell["tooltip"] for cell in before]


@pytest.mark.parametrize("tab_id", ["tab-unified", "tab-toolcalling-streamv1", "tab-toolcalling-batch"])
def test_mixed_probe_is_referenced_twice_but_counted_once(tab_id: str) -> None:
    labels = ["7-3", "7-4", "7-4.anyof", "7-4.mixed_labels", "7-5", "7-5.union"]
    tab = {"id": tab_id, "columns": [{"label": label, "sub": label, "group_key": "7"}
                                    for label in labels],
           "rows": [{"family": "glm47", "cells": {label: make_cell(label) for label in labels}}],
           "column_groups": [{"key": "7", "span": len(labels)}], "stats": {}}
    original = copy.deepcopy(tab["rows"][0])
    group_null_variants(tab)
    assert [column["label"] for column in tab["columns"]] == ["7-3", "7-4", "7-5"]
    row = tab["rows"][0]
    assert leaf_cells(row) == original["cells"]
    assert tab["stats"]["fixture_cases"] == len(labels)
    assert tab["column_groups"][0]["span"] == 3
    for parent in ("7-4", "7-5"):
        assert "7-4.mixed_labels" in {child["sub"] for child in row["cells"][parent]["variants"]}


def test_aggregation_never_moves_a_family_specific_leaf_into_a_generic_section():
    labels = ["7-4", "7-4.anyof", "7-4.ref"]
    tab = {"id": "tab-unified", "columns": [
        {"label": label, "sub": label, "group_key": "glm_specific" if label.endswith(".ref") else "7"}
        for label in labels],
        "rows": [{"family": "glm47", "cells": {label: make_cell(label) for label in labels}}],
        "column_groups": [{"key": "7", "span": 2}, {"key": "glm_specific", "span": 1}], "stats": {}}
    original = copy.deepcopy(tab["rows"][0])
    group_null_variants(tab)
    assert [(column["label"], column["group_key"]) for column in tab["columns"]] == [
        ("7-4", "7"), ("7-4.ref", "glm_specific")]
    assert tab["rows"][0]["cells"]["7-4.ref"] == original["cells"]["7-4.ref"]
    assert "7-4.ref" not in {leaf["sub"] for leaf in tab["rows"][0]["cells"]["7-4"]["variants"]}


@pytest.mark.parametrize("scenario,label,schema,value,description", NULL_VARIANTS)
def test_authored_oracle_satisfies_all_schema_constraints(scenario, label, schema, value, description):
    assert matches_schema(value, schema)
    if label.startswith("7-5"):
        assert not matches_schema(None, schema)
    assert "request tool schema" in description


@pytest.mark.parametrize("family", FAMILIES)
def test_stream_variants_have_distinct_ids_and_independent_goldens(family):
    cases = build_cases(family)
    assert len(cases) == len(NULL_VARIANTS) + 2
    stimuli = []
    for case in cases.values():
        tool = case["tools"][0]
        call = case["golden"]["calls"][0]
        assert matches_schema(call["arguments"], tool["parameters"])
        assert call["name"] == tool["name"]
        assert case["chunks"][-1] == {"delta_text": "", "finish_reason": "stop"}
        stimuli.append((str(case["tools"]), str(case["chunks"])))
    assert len(set(stimuli)) == len(stimuli)


def test_mixed_reproducers_keep_both_field_types_and_strict():
    glm = build_cases("glm47")["TOOLCALLING.streamv1.7-4.mixed_labels"]
    assert glm["golden"]["calls"] == [{"name": "set_labels", "arguments": {
        "label": None, "note": None, "literal": "null"}}]
    m3 = build_cases("minimax_m3")["TOOLCALLING.streamv1.7-4.mixed_grep"]
    assert m3["tools"][0]["strict"] is True
    assert m3["golden"]["calls"] == [{"name": "grep", "arguments": {"pattern": "null", "path": None}}]


@pytest.mark.parametrize("family", ["glm47", "minimax_m2", "minimax_m3", "qwen3_coder"])
def test_native_numeric_call_preserves_each_argument_value(family):
    text = native_call(family, "f", {"first": NumericLiteral("42"), "second": None})
    if family == "minimax_m2":
        assert '<parameter name="first">42</parameter>' in text
        assert '<parameter name="second">null</parameter>' in text
    elif family == "minimax_m3":
        marker = "]<]minimax[>["
        assert f"{marker}<first>42{marker}</first>" in text
        assert f"{marker}<second>null{marker}</second>" in text
    elif family == "glm47":
        assert "<arg_key>first</arg_key><arg_value>42</arg_value>" in text
        assert "<arg_key>second</arg_key><arg_value>null</arg_value>" in text
    else:
        assert "<parameter=first>42</parameter>" in text
        assert "<parameter=second>null</parameter>" in text


def test_missing_fixture_variant_remains_visible_and_incomplete():
    missing = {"kind": "missing", "status": "missing", "sub": "7-4.anyof", "family": "glm47",
               "case_id": None, "cmp": None, "tooltip": {"head": "missing fixture"}}
    for cells in ([make_cell("7-4"), missing], [missing, make_cell("7-4")]):
        result = aggregate_cells(cells, "7-4", NULL_DESCRIPTIONS["7-4"])
        assert cell_state(result, {"key": "dynamo", "label": "Dynamo"})[0] == "empty"
        assert missing in result["variants"]


def test_applicable_families_keep_missing_mixed_probes_as_missing_evidence():
    labels = ["7-4", "7-4.anyof", "7-4.mixed_labels", "7-5"]
    row = {"family": "qwen3", "cells": {label: make_cell(label) for label in labels}}
    row["cells"]["7-4.mixed_labels"].update(kind="missing", status="missing", cmp=None)
    tab = {"id": "tab-unified", "columns": [{"label": label, "sub": label, "group_key": "7"}
                                                for label in labels], "rows": [row],
           "column_groups": [{"key": "7", "span": 4}], "stats": {}}
    group_null_variants(tab)
    assert leaf_cells(row)["7-4.mixed_labels"]["kind"] == "missing"
    assert cell_state(row["cells"]["7-4"], {"key": "dynamo", "label": "Dynamo"})[0] == "empty"
