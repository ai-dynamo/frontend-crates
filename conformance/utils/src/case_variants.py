# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Group distinct fixture cells without losing their inputs or observations."""

import copy
import hashlib
import json

from null_cases import MIXED_CASE_FAMILIES, NULL_DESCRIPTIONS, null_group


def aggregate_cells(cells, parent, description):
    result = copy.deepcopy(next((cell for cell in cells if cell.get("case_id")), cells[0]))
    result["sub"] = cells[0]["sub"]
    result["case_id"] = (result.get("case_id") or "").split(parent, 1)[0] + parent
    result["kind"] = "cell"
    result["status"] = "ok"
    result["red_on_diff"] = True
    candidates = set().union(*((cell.get("cmp") or {}) for cell in cells))
    comparisons = {}
    for key in sorted(candidates):
        entries = [(cell.get("cmp") or {}).get(key, {"na": 1}) for cell in cells]
        missing = any(entry.get("na") for entry in entries)
        failure = any(
            not entry.get("na") and (entry.get("err") or entry.get("leak") or (
                "golden" in (cell.get("cmp") or {})
                and entry["sig"] != cell["cmp"]["golden"]["sig"]
            )) for cell, entry in zip(cells, entries)
        )
        signatures = [None if entry.get("na") else entry["sig"] for entry in entries]
        comparisons[key] = {
            "sig": int(hashlib.sha256(json.dumps(signatures).encode()).hexdigest()[:8], 16),
            "na": int(missing and not failure),
            "err": int(any(entry.get("err") for entry in entries)),
            "leak": int(any(entry.get("leak") for entry in entries)),
        }
    result["cmp"] = comparisons
    result["variants"] = copy.deepcopy(cells)
    result["tooltip"] = {
        "head": f"{result['case_id']} — {result['family']}",
        "description": description,
        "input": {"kind": None}, "init": None, "candidates": [],
        "variants": [cell["tooltip"] for cell in result["variants"]],
    }
    return result


def leaf_cells(row):
    """Expose each recorded fixture once, including probes shared by two columns."""
    leaves = {}
    for sub, cell in row.get("cells", {}).items():
        for leaf in cell.get("variants", [cell]):
            key = leaf.get("sub", sub)
            if key in leaves and leaves[key] != leaf:
                raise ValueError(f"inconsistent grouped fixture: {key}")
            leaves[key] = leaf
    return leaves


def group_null_variants(tab):
    if tab["id"] not in {"tab-unified", "tab-toolcalling-streamv1"}:
        return
    columns = tab["columns"]
    for parent, description in NULL_DESCRIPTIONS.items():
        members = [column for column in columns if null_group(column["label"]) == parent]
        root = next((column for column in members if column["label"] == parent), None)
        if root is None:
            continue
        # Mixed-field probes exercise both types in one request. Reference their
        # single recorded result from both categories instead of duplicating inputs.
        mixed = [column for column in columns if column["label"].startswith("7-4.mixed_")]
        referenced = members + (mixed if parent == "7-5" else [])
        root["desc"] = description
        for row in tab["rows"]:
            if row.get("section") or root["sub"] not in row["cells"]:
                continue
            children = [row["cells"][column["sub"]] for column in referenced
                        if (column["label"] not in MIXED_CASE_FAMILIES
                            or row["family"] in MIXED_CASE_FAMILIES[column["label"]])
                        and column["sub"] in row["cells"]
                        and row["cells"][column["sub"]].get("kind") in {"cell", "missing"}
                        and row["cells"][column["sub"]].get("status") != "na"]
            if len(children) > 1:
                row["cells"][root["sub"]] = aggregate_cells(children, parent, description)
        for column in members:
            if column is not root:
                column["variant_parent"] = parent
    hidden = {column["sub"] for column in columns if "variant_parent" in column}
    tab["columns"] = [column for column in columns if column["sub"] not in hidden]
    for row in tab["rows"]:
        for sub in hidden:
            row["cells"].pop(sub, None)
    for group in tab["column_groups"]:
        group["span"] = sum(column["group_key"] == group["key"] for column in tab["columns"])
    tab["column_groups"] = [group for group in tab["column_groups"] if group["span"]]
    tab["stats"]["variant_columns"] = len(hidden)
    tab["stats"]["sub_cases"] = len(tab["columns"])
    cells = [cell for row in tab["rows"] for cell in row["cells"].values()]
    tab["stats"]["fixture_cases"] = sum(
        cell.get("kind") == "cell" and cell.get("status") != "na"
        for row in tab["rows"] for cell in leaf_cells(row).values()
    )
    tab["stats"].update(slots=len(cells),
                         real=sum(cell.get("kind") == "cell" and cell.get("status") != "na" for cell in cells),
                         na=sum(cell.get("status") == "na" for cell in cells),
                         missing=sum(cell.get("kind") == "missing" for cell in cells))
    for group in tab.get("glossary", []):
        group["rows"] = [(label, NULL_DESCRIPTIONS.get(label, desc))
                         for label, desc in group["rows"]
                         if null_group(label) is None or label in NULL_DESCRIPTIONS]
