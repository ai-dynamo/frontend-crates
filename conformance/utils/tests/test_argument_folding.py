# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Display folds retain the authored fixtures and their independent observations."""
import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from case_variants import aggregate_cells, folded_schemas, group_schema_variants, leaf_cells
from schema_cases import ARGUMENT_SECTIONS, CONFORMANCE_CASES, argument_section, schema_fold, schema_tools
import model
import unified_taxonomy as taxonomy


def child(sub, *, sig=1, missing=False, inapplicable=False):
    return {"sub": sub, "case_id": "UNIFIED." + sub, "family": "alpha", "kind": "cell",
            "status": "na" if inapplicable else "ok",
            "cmp": {"golden": {"sig": 1}, "engine": {"na": 1} if missing else {"sig": sig}},
            "tooltip": {"head": sub, "input": {"text": sub}, "init": {"starting_state": "None"},
                        "candidates": [{"key": "engine", "block": {"text": sub}}]}}


@pytest.mark.parametrize("sig,missing,inapplicable,expected", [
    (1, False, False, (False, False)), (2, False, False, (True, False)),
    (1, True, False, (True, True)), (1, True, True, (False, False)),
])
def test_parent_comparison_preserves_failure_missing_and_inapplicability(sig, missing, inapplicable, expected):
    children = [child("a"), child("b", sig=sig, missing=missing, inapplicable=inapplicable)]
    result = aggregate_cells(children, "7-parent", "parent")
    engine, golden = result["cmp"]["engine"], result["cmp"]["golden"]
    assert (engine["sig"] != golden["sig"], bool(engine["na"])) == expected
    assert result["variants"] == children
    failing = aggregate_cells([child("a", sig=2), child("b", missing=True)], "7-parent", "parent")
    assert not failing["cmp"]["engine"]["na"]
    assert failing["cmp"]["engine"]["sig"] != failing["cmp"]["golden"]["sig"]
    assert aggregate_cells([child("a", inapplicable=True)], "7-parent", "parent")["status"] == "na"


def test_exact_fold_dimensions_and_section_order():
    groups = {}
    for case in CONFORMANCE_CASES:
        scenario = "schema_" + case["name"]
        fold = schema_fold(scenario)
        if fold:
            groups.setdefault(fold[0], []).append((scenario, fold[1]))
    assert len(groups) == 22
    for parent, children in groups.items():
        assert len(children) == 4
        expected = ({"Direct", "Explicit string type", "allOf", "Reference"} if "const_" in parent
                    else {"String / Forward", "String / Reversed", "Typed value / Forward", "Typed value / Reversed"})
        assert {label for _, label in children} == expected
        assert parent in {scenario for scenario, _ in children}
    ordered = sorted((scenario for scenario in taxonomy.UNIFIED_TAX if taxonomy.tax(scenario)[0] == 7),
                     key=taxonomy.display_sort_key)
    assert len(ordered) == 139
    assert list(dict.fromkeys(argument_section(s) for s in ordered)) == list(ARGUMENT_SECTIONS)


def test_folded_serialization_preserves_leaves_and_schema_associations():
    probes = [case for case in CONFORMANCE_CASES if case["name"].startswith("union_anyOf_integer_")]
    columns = [{"sub": "schema_" + case["name"], "label": "7-" + case["name"], "desc": "probe",
                "group_key": "union", "schemas": [{"families": ["alpha"], "tools": schema_tools(case)}]}
               for case in probes]
    cells = {column["sub"]: child(column["sub"]) for column in columns}
    original = copy.deepcopy(cells)
    tab = {"id": "tab-unified", "kind": "unified", "label": "Unified", "columns": columns,
           "column_groups": [{"key": "union", "span": 4}], "rows": [{"family": "alpha", "cells": cells}],
           "stats": {}, "candidates": [], "glossary": []}
    group_schema_variants(tab)
    assert len(tab["columns"]) == 1
    schemas = tab["columns"][0]["schemas"]
    assert len(schemas) == 2  # Each branch order is shared by both value types.
    assert sum(len(schema["cases"]) for schema in schemas) == 4
    assert {association["case_id"] for schema in schemas for association in schema["cases"]} == {
        "UNIFIED." + column["label"] for column in columns}
    decoded = model.hydrate_page(json.loads(model.to_script_json(model.build_page({}, [tab]))))
    assert decoded["tabs"][0]["columns"][0]["schemas"] == schemas
    leaves = leaf_cells(decoded["tabs"][0]["rows"][0])
    assert set(leaves) == set(original)
    for key, leaf in leaves.items():
        assert leaf["case_id"] == original[key]["case_id"]
        assert leaf["cmp"] == original[key]["cmp"]
        assert leaf["tooltip"]["input"] == original[key]["tooltip"]["input"]
        assert leaf["tooltip"]["init"] == original[key]["tooltip"]["init"]
        assert leaf["tooltip"]["candidates"] == original[key]["tooltip"]["candidates"]


def test_schema_dedup_retains_case_specific_families():
    tools = [{"name": "probe", "parameters": {"type": "string"}}]
    schemas = folded_schemas([
        {"label": "Direct", "schemas": [{"families": ["alpha"], "tools": tools}]},
        {"label": "Reference", "schemas": [{"families": ["beta"], "tools": tools}]},
    ])
    assert len(schemas) == 1
    assert schemas[0]["families"] == ["alpha", "beta"]
    assert [(item["label"], item["families"]) for item in schemas[0]["cases"]] == [
        ("Direct", ["alpha"]), ("Reference", ["beta"])]


@pytest.mark.parametrize("prefix", ["const_plain_", "union_anyOf_integer_", "union_oneOf_integer_"])
@pytest.mark.parametrize("single", [False, True])
def test_schema_fold_without_direct_variant_preserves_available_fixture(prefix, single):
    probes = [case for case in CONFORMANCE_CASES if case["name"].startswith(prefix)
              and "schema_" + case["name"] != schema_fold("schema_" + case["name"])[0]]
    if single:
        probes = probes[:1]
    columns = [{"sub": "schema_" + case["name"], "label": taxonomy.case_label("schema_" + case["name"]),
                "desc": "probe", "group_key": "schema",
                "schemas": [{"families": ["alpha"], "tools": schema_tools(case)}]} for case in probes]
    cells = {column["sub"]: child(column["sub"]) for column in columns}
    originals = list(cells.values())
    before = copy.deepcopy(originals)
    tab = {"id": "tab-unified", "columns": columns, "rows": [{"family": "alpha", "cells": cells}],
           "column_groups": [{"key": "schema", "span": len(columns)}], "stats": {},
           "glossary": [{"rows": [(column["label"], "probe") for column in columns]}]}
    group_schema_variants(tab)
    assert len(tab["columns"]) == len(cells) == 1
    root = tab["columns"][0]
    result = cells[root["sub"]]
    assert result["case_id"] == "UNIFIED." + root["label"]
    assert set(leaf_cells(tab["rows"][0])) == {column["sub"] for column in columns}
    assert tab["stats"]["fixture_cases"] == len(probes)
    assert tab["glossary"][0]["rows"] == [(root["label"], root["desc"])]
    assert originals == before
    assert all(variant["tooltip"]["head"].startswith(schema_fold(variant["sub"])[1] + " — ")
               for variant in result["variants"])
