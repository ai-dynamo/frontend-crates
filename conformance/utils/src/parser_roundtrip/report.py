# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verdict counts and failure groups as JSON and Markdown, written to files
only the owner can read: results and parser errors quote model text."""

from __future__ import annotations

import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, TextIO

MAX_LISTED_IDS = 20


def open_private(path: Path) -> TextIO:
    """Open `path` for writing with mode 0600, never through a symlink."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW
    handle = os.fdopen(os.open(path, flags, 0o600), "w", encoding="utf-8")
    os.fchmod(handle.fileno(), 0o600)
    return handle


def summarize(
    header: dict[str, Any], results: list[dict[str, Any]], meta: dict[str, Any]
) -> dict[str, Any]:
    groups: defaultdict[str, list[str]] = defaultdict(list)
    for result in results:
        if result["verdict"] == "fail":
            groups[result["signature"]].append(result["id"])
    notes = Counter(result["note"] for result in results if result.get("note"))
    orders = Counter(result["order"] for result in results if "order" in result)
    return {
        "schema": "parser-roundtrip-summary/v1",
        **meta,
        "parser": header,
        "cases": len(results),
        "counts": dict(sorted(Counter(r["verdict"] for r in results).items())),
        "notes": dict(sorted(notes.items())),
        "orders": dict(orders.most_common()),
        "incomplete_calls": sum(1 for r in results if r.get("incomplete_calls")),
        "failures": [
            {"signature": key, "count": len(ids), "ids": ids}
            for key, ids in sorted(groups.items(), key=lambda item: -len(item[1]))
        ],
    }


def markdown(summary: dict[str, Any]) -> str:
    def cell(text: str) -> str:
        return text.replace("|", "\\|").replace("`", "'")

    def listed(ids: list[str]) -> str:
        more = " …" if len(ids) > MAX_LISTED_IDS else ""
        return ", ".join(ids[:MAX_LISTED_IDS]) + more

    parser = summary["parser"]
    lines = [
        "# Parser round trip",
        "",
        f"{parser.get('parser')} {parser.get('version')}, family "
        f"`{summary.get('family')}`: {summary['cases']} cases, "
        f"{summary['incomplete_calls']} with incomplete calls.",
        "",
        "| verdict | cases |",
        "|---|---|",
        *(f"| {verdict} | {count} |" for verdict, count in summary["counts"].items()),
    ]
    if summary["orders"]:
        lines += [
            "",
            "Parsed event order, runs collapsed:",
            "",
            "| order | cases |",
            "|---|---|",
            *(
                f"| {order or '(none)'} | {n} |"
                for order, n in summary["orders"].items()
            ),
        ]
    if summary["notes"]:
        lines += [
            "",
            "Template-ambiguous turns judged after a fold:",
            "",
            "| note | cases |",
            "|---|---|",
            *(f"| {note} | {count} |" for note, count in summary["notes"].items()),
        ]
    if summary["failures"]:
        lines += [
            "",
            "## Failures by first difference",
            "",
            "| signature | cases | IDs |",
            "|---|---|---|",
            *(
                f"| `{cell(group['signature'])}` | {group['count']} | "
                f"{listed(group['ids'])} |"
                for group in summary["failures"]
            ),
        ]
    return "\n".join(lines) + "\n"
