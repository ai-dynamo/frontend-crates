#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate rendered conformance cells for selected models and tabs.

The HTML model is the source of truth for what the reader sees. This script reads
the inlined ``conformance-model`` JSON after rendering and reports every selected
model/case pair whose default Reference cell is empty or red.

Examples:
  python3 conformance/utils/src/validate_conformance_status.py \
      --html conformance/CONFORMANCE_v2.html --model qwen3 --tab unified
  python3 conformance/utils/src/validate_conformance_status.py \
      --html conformance/CONFORMANCE_v2.html --model qwen3 --tab unified \
      --require-green
"""

import argparse
import html
import json
import re
import sys
from pathlib import Path


MODEL_RE = re.compile(
    r'<script type="application/json" id="conformance-model">(.*?)</script>', re.DOTALL
)


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def load_model(path: Path) -> dict:
    match = MODEL_RE.search(path.read_text())
    if match is None:
        raise ValueError(f"{path}: missing conformance-model JSON")
    return json.loads(html.unescape(match.group(1)))


def select_tabs(model: dict, requested: list[str]) -> list[dict]:
    tabs = model.get("tabs") or []
    if not requested:
        return tabs
    selected = []
    unknown = []
    for name in requested:
        wanted = _normalize(name.removeprefix("tab-"))
        matches = [
            tab
            for tab in tabs
            if wanted
            in {
                _normalize(tab.get("id", "").removeprefix("tab-")),
                _normalize(tab.get("kind", "")),
                _normalize(tab.get("label", "")),
            }
        ]
        if not matches:
            unknown.append(name)
            continue
        for tab in matches:
            if tab not in selected:
                selected.append(tab)
    if unknown:
        choices = ", ".join(tab.get("id", "") for tab in tabs)
        raise ValueError(f"unknown tab(s): {', '.join(unknown)}; choices: {choices}")
    return selected


def select_rows(tab: dict, requested: list[str]) -> list[dict]:
    rows = [row for row in tab.get("rows", []) if row.get("family")]
    if not requested:
        return rows
    selected = []
    unknown = []
    for name in requested:
        wanted = _normalize(name)
        matches = [
            row
            for row in rows
            if wanted in {_normalize(row.get("family", "")), _normalize(row.get("model_label", ""))}
        ]
        if not matches:
            unknown.append(name)
            continue
        for row in matches:
            if row not in selected:
                selected.append(row)
    if unknown:
        choices = ", ".join(row.get("family", "") for row in rows)
        raise ValueError(
            f"{tab.get('id')}: unknown model(s): {', '.join(unknown)}; choices: {choices}"
        )
    return selected


def reference(tab: dict) -> dict:
    candidates = tab.get("candidates") or []
    matches = [candidate for candidate in candidates if candidate.get("default_bucket") == "A"]
    if len(matches) != 1:
        raise ValueError(
            f"{tab.get('id')}: expected exactly one default Reference candidate, found {len(matches)}"
        )
    return matches[0]


def cell_state(cell: dict | None, ref: dict) -> tuple[str, str]:
    if cell is None:
        return "empty", "no cell was emitted for this model/case pair"
    if cell.get("kind") == "cell" and cell.get("status") == "na":
        note = (cell.get("tooltip") or {}).get("na_note") or "not applicable to this family"
        return "na", note

    cmp = cell.get("cmp") or {}
    current = cmp.get(ref["key"])
    if current is None:
        return "empty", f"the default Reference {ref['label']!r} has no comparison entry"
    if current.get("na") == 1:
        return "empty", f"the default Reference {ref['label']!r} has no captured result"
    if current.get("err") == 1:
        return "red", f"the default Reference {ref['label']!r} returned an error"

    if cell.get("red_on_diff"):
        golden = cmp.get("golden")
        if golden is None:
            return "red", "Unified cell has no GOLDEN comparison entry"
        if current.get("sig") != golden.get("sig"):
            return "red", "the default Reference differs from GOLDEN"
    elif current.get("leak") == 1:
        return "red", "the default Reference leaked structured markup"

    return "green", ""


def build_status(model: dict, tabs: list[dict], requested_models: list[str], html_path: Path) -> dict:
    reports = []
    for tab in tabs:
        ref = reference(tab)
        columns = tab.get("columns") or []
        for row in select_rows(tab, requested_models):
            issues = []
            for column in columns:
                sub = column.get("sub", "")
                state, reason = cell_state((row.get("cells") or {}).get(sub), ref)
                if state in {"green", "na"}:
                    continue
                issues.append(
                    {
                        "state": state,
                        "case": column.get("label", sub),
                        "scenario": sub,
                        "reason": reason,
                    }
                )
            reports.append(
                {
                    "tab": tab.get("id"),
                    "model": row.get("family"),
                    "reference": {"key": ref["key"], "label": ref["label"]},
                    "cells": len(columns),
                    "empty": sum(issue["state"] == "empty" for issue in issues),
                    "red": sum(issue["state"] == "red" for issue in issues),
                    "na": sum(
                        (row.get("cells") or {}).get(column.get("sub", ""), {}).get("kind") == "cell"
                        and (row.get("cells") or {}).get(column.get("sub", ""), {}).get("status") == "na"
                        for column in columns
                    ),
                    "issues": issues,
                }
            )
    return {
        "schema": 1,
        "html": str(html_path),
        "generated": model.get("meta", {}),
        "reports": reports,
    }


def print_summary(status: dict) -> None:
    for report in status["reports"]:
        print(
            f"{report['tab']} {report['model']}: "
            f"{report['cells']} cells, {report.get('na', 0)} n/a, "
            f"{report['empty']} empty, {report['red']} red "
            f"(Reference: {report['reference']['label']})"
        )
        for issue in report["issues"]:
            print(f"  {issue['state'].upper()} {issue['case']} ({issue['scenario']}): {issue['reason']}")


def print_totals(status: dict) -> None:
    reports = status["reports"]
    empty = sum(report["empty"] for report in reports)
    red = sum(report["red"] for report in reports)
    print(f"conformance status: {len(reports)} model/tab pairs, {empty} empty, {red} red")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", type=Path, required=True, help="rendered CONFORMANCE_v2.html")
    parser.add_argument("--model", action="append", default=[], help="model family or row label; repeatable")
    parser.add_argument("--tab", action="append", default=[], help="tab id, kind, or label; repeatable")
    parser.add_argument("--status-path", type=Path, help="write the machine-readable status JSON")
    parser.add_argument("--require-green", action="store_true", help="exit 1 when any selected cell is empty or red")
    parser.add_argument("--summary-only", action="store_true", help="print totals without listing each issue")
    args = parser.parse_args(argv)

    try:
        model = load_model(args.html)
        status = build_status(model, select_tabs(model, args.tab), args.model, args.html)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))

    if args.status_path:
        args.status_path.parent.mkdir(parents=True, exist_ok=True)
        args.status_path.write_text(json.dumps(status, indent=2) + "\n")
    if args.summary_only:
        print_totals(status)
    else:
        print_summary(status)
    blocked = any(report["empty"] or report["red"] for report in status["reports"])
    return 1 if args.require_green and blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
