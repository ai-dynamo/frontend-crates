# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Group distinct fixture cells without losing their inputs or observations."""

import copy
import hashlib
import json

from null_cases import MIXED_CASE_FAMILIES, NULL_DESCRIPTIONS, null_group
import unified_taxonomy
from numeric_cases import (
    NUMERIC_DESCRIPTIONS,
    NUMERIC_DISPLAY_LABELS,
    applicable as numeric_applicable,
    numeric_group,
)

VARIANT_DESCRIPTIONS = {**NULL_DESCRIPTIONS, **NUMERIC_DESCRIPTIONS}


def variant_group(label):
    return null_group(label) or numeric_group(label)


def aggregate_cells(cells, parent, description, *, display_label=None):
    result = copy.deepcopy(next((cell for cell in cells if cell.get("case_id")), cells[0]))
    result["sub"] = cells[0]["sub"]
    case_id = result.get("case_id") or ""
    parent_key = parent.split(".", 1)[0]
    prefix = "UNIFIED." if case_id.startswith("UNIFIED.") else case_id.split(parent_key, 1)[0]
    result["case_id"] = prefix + (display_label or parent)
    result["kind"] = "cell"
    result["status"] = "ok"
    result["red_on_diff"] = True
    applicable = [cell for cell in cells if cell.get("status") != "na"]
    if not applicable:
        result["status"] = "na"
    candidates = set().union(*((cell.get("cmp") or {}) for cell in applicable))
    comparisons = {}
    for key in sorted(candidates):
        entries = [(cell.get("cmp") or {}).get(key, {"na": 1}) for cell in applicable]
        missing = any(entry.get("na") for entry in entries)
        failure = any(
            not entry.get("na") and (entry.get("err") or entry.get("leak") or (
                "golden" in (cell.get("cmp") or {})
                and entry["sig"] != cell["cmp"]["golden"]["sig"]
            )) for cell, entry in zip(applicable, entries)
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


def visible_null_groups(labels, family):
    groups = set()
    for label in labels:
        parent = null_group(label)
        if parent is None:
            groups.add(label.split(".", 1)[0])
            continue
        groups.add(parent)
        if label in MIXED_CASE_FAMILIES and family in MIXED_CASE_FAMILIES[label]:
            groups.add("7-5")
    return groups


def group_null_variants(tab: dict) -> None:
    if tab["id"] not in {"tab-unified", "tab-toolcalling-streamv1", "tab-toolcalling-batch"}:
        return
    columns = tab["columns"]
    unified = tab["id"] == "tab-unified"

    def metadata_label(column):
        return unified_taxonomy.legacy_case_label(column["sub"]) if unified else column["label"]

    def display_label(label):
        return unified_taxonomy.historical_case_label(label) if unified else label

    def null_parent(label):
        scenario = unified_taxonomy.scenario_for_label(label) if unified else None
        return null_group(unified_taxonomy.legacy_case_label(scenario) if scenario else label)

    display_labels: dict[str, str] = {}
    display_descriptions: dict[str, str] = {}
    sections = [(parent, description, group_key)
                for parent, description in VARIANT_DESCRIPTIONS.items()
                for group_key in dict.fromkeys(column["group_key"] for column in columns
                                               if variant_group(metadata_label(column)) == parent)]
    for parent, description, group_key in sections:
        members = [column for column in columns
                   if variant_group(metadata_label(column)) == parent and column["group_key"] == group_key]
        if unified:
            # Display sorting must not change the parent oracle or variant order.
            members.sort(key=lambda column: unified_taxonomy.taxonomy_sort_key(column["sub"]))
        if not members:
            continue
        root = next((column for column in members if metadata_label(column) == parent), members[0])
        # A corpus may contain only a named variant; keep its fixture identity.
        display_group = (NUMERIC_DISPLAY_LABELS[parent] if parent in NUMERIC_DESCRIPTIONS
                         else root["label"] if len(members) == 1 else display_label(parent))
        display_labels[root["label"]] = display_group
        if unified:
            root["_grouped_display_label"] = display_group
        display_descriptions[display_group] = description
        # Mixed-field probes exercise both types in one request. Reference their
        # single recorded result from both categories instead of duplicating inputs.
        mixed = [column for column in columns if metadata_label(column).startswith("7-4.mixed_")
                 and column["group_key"] == group_key]
        if unified:
            mixed.sort(key=lambda column: unified_taxonomy.taxonomy_sort_key(column["sub"]))
        referenced = members + (mixed if parent == "7-5" else [])
        root["desc"] = description
        root_schemas = folded_schemas(referenced)
        if root_schemas:
            root["schemas"] = root_schemas
        for row in tab["rows"]:
            if row.get("section") or root["sub"] not in row["cells"]:
                continue
            inapplicable = []
            if parent in NUMERIC_DESCRIPTIONS and not numeric_applicable(row["family"], metadata_label(root)):
                prefix = "UNIFIED" if tab["id"] == "tab-unified" else "TOOLCALLING.streamv1"
                for column in members:
                    cell = row["cells"].get(column["sub"])
                    if cell is None:
                        continue
                    case_id = f"{prefix}.{column['label']}"
                    cell.update(
                        kind="cell",
                        case_id=case_id,
                        status="na",
                        red_on_diff=False,
                        cmp={
                            candidate["key"]: {"sig": 0, "na": 1, "err": 0, "leak": 0}
                            for candidate in tab["candidates"]
                        },
                    )
                    tooltip = copy.deepcopy(cell.get("tooltip") or {})
                    tooltip.update(
                        head=f"{case_id} — {row['family']}",
                        description=description,
                        na_note="This grammar does not coerce untyped fractional text through an integer/string schema.",
                    )
                    cell["tooltip"] = tooltip
                    inapplicable.append(cell)
            children = [row["cells"][column["sub"]] for column in referenced
                        if (unified or metadata_label(column) not in MIXED_CASE_FAMILIES
                            or row["family"] in MIXED_CASE_FAMILIES[metadata_label(column)])
                        and column["sub"] in row["cells"]
                        and row["cells"][column["sub"]].get("kind") in {"cell", "missing"}
                        and row["cells"][column["sub"]].get("status") != "na"]
            if len(children) > 1:
                row["cells"][root["sub"]] = aggregate_cells(
                    children, parent, description, display_label=display_group
                )
            elif inapplicable:
                grouped = copy.deepcopy(inapplicable[0])
                grouped.update(
                    sub=root["sub"],
                    case_id=f"{prefix}.{display_group}",
                    kind="cell",
                    status="na",
                    red_on_diff=False,
                    variants=copy.deepcopy(inapplicable),
                )
                grouped["tooltip"] = {
                    "head": f"{grouped['case_id']} — {row['family']}",
                    "description": description,
                    "input": {"kind": None},
                    "init": None,
                    "candidates": [],
                    "variants": [cell["tooltip"] for cell in inapplicable],
                    "na_note": "This grammar does not coerce untyped fractional text through an integer/string schema.",
                }
                row["cells"][root["sub"]] = grouped
        for column in members:
            if column is not root:
                column["variant_parent"] = parent
    hidden = {column["sub"] for column in columns if "variant_parent" in column}
    tab["columns"] = [column for column in columns if column["sub"] not in hidden]
    for column in tab["columns"]:
        column["label"] = display_labels.get(column["label"], column["label"])
    numeric_labels = set(NUMERIC_DISPLAY_LABELS.values())
    numeric_positions = [index for index, column in enumerate(tab["columns"])
                         if column["label"] in numeric_labels]
    numeric_order = {label: index for index, label in enumerate(NUMERIC_DISPLAY_LABELS.values())}
    ordered_numeric_columns = sorted(
        (tab["columns"][index] for index in numeric_positions),
        key=lambda column: numeric_order[column["label"]],
    )
    for index, column in zip(numeric_positions, ordered_numeric_columns):
        tab["columns"][index] = column
    for row in tab["rows"]:
        for sub in hidden:
            row["cells"].pop(sub, None)
    refresh_variant_stats(tab, len(hidden))
    for group in tab.get("glossary", []):
        group["rows"] = [(display_labels.get(label, label),
                          NULL_DESCRIPTIONS.get(null_parent(label), desc))
                         for label, desc in group["rows"]
                         if null_parent(label) is None or label in display_labels]


def folded_schemas(columns):
    """One JSON block per schema, with every case/family association retained."""
    schemas = {}
    for column in columns:
        for variant in column.get("schemas", []):
            key = json.dumps(variant["tools"], sort_keys=True, separators=(",", ":"))
            merged = schemas.setdefault(key, {"tools": variant["tools"], "families": [], "cases": []})
            for family in variant["families"]:
                if family not in merged["families"]:
                    merged["families"].append(family)
            merged["cases"].append({"label": column.get("variant_label", column["label"]),
                                    "case_id": column.get("case_id", "UNIFIED." + column["label"]),
                                    "families": variant["families"]})
    return list(schemas.values())


def refresh_variant_stats(tab, hidden_count):
    for group in tab["column_groups"]:
        group["span"] = sum(column["group_key"] == group["key"] for column in tab["columns"])
    tab["column_groups"] = [group for group in tab["column_groups"] if group["span"]]
    tab["stats"]["variant_columns"] = tab["stats"].get("variant_columns", 0) + hidden_count
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


def group_unified_variants(tab):
    if tab["id"] != "tab-unified":
        return
    groups = {}
    for column in tab["columns"]:
        fold = unified_taxonomy.case_fold(column["sub"])
        if fold:
            parent, label = fold
            column["variant_label"] = label
            groups.setdefault(parent, []).append(column)
    hidden = set()
    for parent, members in groups.items():
        members.sort(key=lambda column: (column["sub"] != parent, {
            "Direct": 0, "Explicit string type": 1, "allOf": 2, "Reference": 3,
            "String / Forward": 0, "Typed value / Forward": 1,
            "String / Reversed": 2, "Typed value / Reversed": 3,
        }.get(column["variant_label"], 4), column["sub"]))
        # Partial corpora keep the identity of an available fixture.
        root = next((column for column in members if column["sub"] == parent), members[0])
        labels = {column["sub"]: column["variant_label"] for column in members}
        description = root["desc"]
        if parent.startswith("schema_const_"):
            description = "String constant: Direct, Explicit string type, allOf, and Reference."
        elif parent.startswith("schema_union_"):
            description = "Union alternatives: String / Typed value x Forward / Reversed branch order."
        root["desc"] = description
        root["schemas"] = folded_schemas(members)
        for row in tab["rows"]:
            if row.get("section"):
                continue
            children = [row["cells"][column["sub"]] for column in members if column["sub"] in row["cells"]]
            if children:
                aggregated = aggregate_cells(children, unified_taxonomy.case_label(root["sub"]), description)
                aggregated["sub"] = root["sub"]
                for child in aggregated["variants"]:
                    if child.get("tooltip"):
                        child["tooltip"]["head"] = labels[child["sub"]] + " — " + child["tooltip"]["head"]
                row["cells"][root["sub"]] = aggregated
        hidden.update(column["sub"] for column in members if column is not root)
    hidden_labels = {column["label"] for column in tab["columns"] if column["sub"] in hidden}
    tab["columns"] = [column for column in tab["columns"] if column["sub"] not in hidden]
    for row in tab["rows"]:
        for sub in hidden:
            row["cells"].pop(sub, None)
    descriptions = {column["label"]: column["desc"] for column in tab["columns"]}
    for group in tab.get("glossary", []):
        group["rows"] = [(label, descriptions.get(label, desc))
                         for label, desc in group["rows"]
                         if label not in hidden_labels and (variant_group(label) is None or label in descriptions)]
    refresh_variant_stats(tab, len(hidden))
