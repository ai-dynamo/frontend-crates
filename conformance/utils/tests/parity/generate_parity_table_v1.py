#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate parity tables for each parser stage from one common entrypoint.

Examples:
    python3 tests/parity/generate_parity_table.py all --html > tests/parity/PARITY.html
    python3 tests/parity/generate_parity_table.py toolcalling --html > tests/parity/toolcalling/PARITY.html
    python3 tests/parity/generate_parity_table.py toolcalling --mode stream > tests/parity/toolcalling/PARITY.stream.md
    python3 tests/parity/generate_parity_table.py reasoning --html > tests/parity/reasoning/PARITY.html
"""

from __future__ import annotations

import argparse
import datetime
import html as html_lib
import sys
import zoneinfo
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tests.parity import common  # noqa: E402
from tests.parity.reasoning import table as reasoning_table  # noqa: E402
from tests.parity.toolcalling import table as toolcalling_table  # noqa: E402

import model  # noqa: E402  (DIS-2434 JSON model: schema + serialization)
import re as _re

# Published location of the v1 PARITY page (rendered into .stage/, then copied to
# conformance/PARITY.html). Links resolve relative to this destination.
_PUBLISHED_OUTPUT = REPO_ROOT / "conformance" / "PARITY.html"


def _tab_button(panel: dict[str, Any]) -> str:
    active = " active" if panel["active"] else ""
    selected = "true" if panel["active"] else "false"
    panel_id = html_lib.escape(str(panel["id"]))
    label = html_lib.escape(str(panel["label"]))
    title = html_lib.escape(str(panel.get("tab_title", panel["label"])))
    return (
        f'<button class="tab-button{active}" id="{panel_id}-button" '
        f'type="button" role="tab" aria-selected="{selected}" '
        f'aria-label="{title}" title="{title}" '
        f'data-tab-target="{panel_id}">{label}</button>'
    )


def _combined_toolcalling_panels() -> list[dict[str, Any]]:
    panels = []
    for mode in ("batch", "stream"):
        _mode, panel, _has_cases = toolcalling_table._load_html_panel(mode)
        panel.update(
            {
                "id": f"tab-toolcalling-{mode}",
                "label": f"TC {mode}",
                "tab_title": f"Tool Calling {mode}",
                "active": False,
                "case_docs_href": common.LINKS["toolcalling_cases"],
                "case_docs_label": "lib/parsers/TOOLCALLING_CASES.md",
                "case_prefix": f"TOOLCALLING.{mode}.",
                "case_section_id": f"toolcalling-{mode}",
                # Compare model: versioned candidates (impl+version) drive the
                # per-panel Base/Compare buckets on both TC batch and TC stream.
                "candidates": toolcalling_table._candidate_items(mode),
            }
        )
        panels.append(panel)
    return panels


def _combined_reasoning_panels() -> list[dict[str, Any]]:
    rows, columns, refs = reasoning_table._load()
    no_vllm, no_sglang = reasoning_table._derive_no_peer_sets(rows)
    panels = []
    for mode in ("batch", "stream"):
        mode_columns = reasoning_table._columns_for_mode(columns, mode)
        panel = reasoning_table._html_panel(
            rows,
            mode_columns,
            refs,
            no_vllm,
            no_sglang,
            mode=mode,
            active=False,
        )
        panel.update(
            {
                "id": f"tab-reasoning-{mode}",
                "label": f"Reasoning {mode}",
                "active": False,
                "case_docs_href": common.LINKS["reasoning_cases"],
                "case_docs_label": "lib/parsers/REASONING_CASES.md",
                "case_prefix": "REASONING.",
                "case_section_id": f"reasoning-{mode}",
                "legend_html": reasoning_table._legend_html(rows, mode_columns),
            }
        )
        panels.append(panel)
    return panels


_MODE_PAREN_RE = _re.compile(r"\(([^)]*)\)\s*$")
_LABEL_VERSION_RE = _re.compile(r"(\d[\w.]*)\s*\([^)]*\)\s*$")


def _candidate_model(items: list[dict]) -> list[dict]:
    out = []
    for it in items:
        key = it["key"]
        label = it["label"]
        m = _MODE_PAREN_RE.search(label)
        pm = None
        if m:
            pm = "stream" if "stream" in m.group(1) else ("batch" if "batch" in m.group(1) else None)
        vm = _LABEL_VERSION_RE.search(label)
        grp = next((p for p in ("dynamo", "vllm", "sglang") if key.startswith(p)), key)
        out.append({"key": key, "impl": grp, "label": label,
                    "label_html": html_lib.escape(label),
                    "default_bucket": it.get("default_bucket", "C"),
                    "version": it.get("version") or (vm.group(1) if vm else None),
                    "parse_mode": pm})
    return out


def build_combined_model(stamp: str, sha: str | None) -> dict:
    """Whole-page JSON model for the v1 PARITY page (DIS-2434), same schema as v2."""
    r_rows, r_columns, r_refs = reasoning_table._load()
    r_no_vllm, r_no_sglang = reasoning_table._derive_no_peer_sets(r_rows)
    tabs: list[dict] = []
    for mode in ("batch", "stream"):
        tab = toolcalling_table.build_model_panel(mode)
        tab.update({
            "label": f"TC {mode}", "label_html": f"TC {mode}",
            "tab_title": f"Tool Calling {mode}",
            "case_prefix": f"TOOLCALLING.{mode}.", "case_section_id": f"toolcalling-{mode}",
            "case_docs_href": common.LINKS["toolcalling_cases"],
            "case_docs_label": "lib/parsers/TOOLCALLING_CASES.md",
            "candidates": _candidate_model(tab.get("candidates", [])),
            "captured_note": "", "toolbar_desc_html": None, "details_note_html": None,
            "active": False,
        })
        tabs.append(tab)
    for mode in ("batch", "stream"):
        mode_columns = reasoning_table._columns_for_mode(r_columns, mode)
        tab = reasoning_table.build_model_panel(
            r_rows, mode_columns, r_refs, r_no_vllm, r_no_sglang, mode=mode, active=False)
        tab.update({
            "id": f"tab-reasoning-{mode}",
            "label": f"Reasoning {mode}", "label_html": f"Reasoning {mode}",
            "tab_title": f"Reasoning {mode}",
            "case_prefix": "REASONING.", "case_section_id": f"reasoning-{mode}",
            "case_docs_href": common.LINKS["reasoning_cases"],
            "case_docs_label": "lib/parsers/REASONING_CASES.md",
            "candidates": _candidate_model(tab.get("candidates", [])),
            "captured_note": "", "toolbar_desc_html": None, "details_note_html": None,
            "active": False,
        })
        tabs.append(tab)
    tabs[0]["active"] = True
    # The v1 PARITY page keeps its own "match Base" summary legend + stats line (the v2
    # page uses the JS view's default text). data-overview-count spans are filled by
    # applyCtl. Mirrors the summary-only block in parity_table_v1.html.j2.
    v1_summary_legend = (
        '<p class="legend summary-only"><strong>Overview:</strong> '
        '<span class="summary-key ok" aria-hidden="true"></span>green = selected candidates match Base · '
        '<span class="summary-key problem" aria-hidden="true"></span>red = Base leaks parser markup · '
        '<span class="summary-key na" aria-hidden="true"></span>gray = not applicable, unavailable, or missing fixture.</p>'
        '<p class="stats summary-only">Stats for selected comparison: '
        '<span style="color:#0a7d2c" data-overview-count="ok">0</span> green · '
        '<span style="color:#b00" data-overview-count="problem">0</span> red · '
        '<span style="color:#555" data-overview-count="na">0</span> gray.</p>'
    )
    meta = {"title": "Dynamo Parser Parity Table", "stamp": stamp, "sha": sha,
            "short_sha": sha[:12] if sha else "",
            "command": "python3 tests/parity/generate_parity_table.py all --html",
            "output": "tests/parity/PARITY.html",
            "summary_legend_html": v1_summary_legend,
            "generated_by": "generate_parity_table_v1.build_combined_model"}
    return model.build_page(meta, tabs, parser_ni={}, legend_html="")


def render_combined_html() -> str:
    common.set_links(_PUBLISHED_OUTPUT, REPO_ROOT)

    now = datetime.datetime.now(zoneinfo.ZoneInfo("America/Los_Angeles"))
    stamp = now.strftime("%Y-%m-%d %H:%M %Z")
    sha = toolcalling_table._commit_sha()

    # DIS-2434: the page is rendered entirely by the JS view from this JSON model; the
    # template emits a skeleton + the model blob (no server-rendered tabs/panels).
    model_json = model.to_script_json(build_combined_model(stamp, sha))

    return (
        toolcalling_table._make_jinja_env()
        .get_template("parity_table_v1.html.j2")
        .render(
            title="Dynamo Parser Parity Table",
            stamp=stamp,
            sha=sha,
            short_sha=sha[:12] if sha else "",
            command="python3 tests/parity/generate_parity_table.py all --html",
            output="tests/parity/PARITY.html",
            tabs=[],
            panels=[],
            peer_versions=toolcalling_table._peer_version_items(
                toolcalling_table._peer_versions()
            ),
            peer_versions_href=common.LINKS["pyproject_stub"],
            model_json=model_json,
            js_view=True,
        )
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate Dynamo parity tables.",
    )
    parser.add_argument(
        "stage",
        choices=("all", "toolcalling", "reasoning"),
        help="Parity stage to render.",
    )
    args, rest = parser.parse_known_args(argv)

    if args.stage == "all":
        stage_parser = argparse.ArgumentParser(
            description="Generate the combined Dynamo parser parity HTML page.",
        )
        stage_parser.add_argument(
            "--html",
            action="store_true",
            help="Emit the combined HTML page.",
        )
        stage_args = stage_parser.parse_args(rest)
        if not stage_args.html:
            parser.error("stage 'all' currently supports --html only")
        print(render_combined_html())
        return

    stage_table = toolcalling_table if args.stage == "toolcalling" else reasoning_table
    stage_table.main(rest)


if __name__ == "__main__":
    main()
