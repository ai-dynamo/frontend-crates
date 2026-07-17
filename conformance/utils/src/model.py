# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One JSON data model for the conformance page (DIS-2434).

Python computes the MODEL only; a JS view (`assets/conformance_view.js`) renders the
table, tabs, compare bar, and popups from it. Comparison/parity SEMANTICS stay in
Python (`markers.py`/`impls.py` are the single source of truth) — this module
orchestrates them into a clean, documented, greppable structure and serializes it as
one inlined `<script type="application/json">` blob (so `file://` keeps working, no
fetch). The old `D_rb`/`V_ps` marker mini-language is gone: cells carry STRUCTURED
comparison facts and the view decides how to display them.

Schema (top-down)
=================

page = {
  "schema": 1,                       # bumped on breaking model changes
  "meta": {
    "title": str,
    "stamp": str,                    # "YYYY-MM-DD HH:MM PDT"
    "sha": str | None, "short_sha": str,
    "command": str, "output": str,   # provenance line under the title
    "generated_by": str,             # which generator produced this page
  },
  "parser_ni": { cand_key: {"label": str, "families": [str]} },
                                     # parsers with limited family coverage (Dynamo
                                     # v2): drives the reference-aware "not
                                     # implemented for family X" note. Was
                                     # window.__PARSER_NI.
  "legend_html": str,                # shared legend (still HTML; presentation copy)
  "tabs": [ tab, ... ],
}

tab = {
  "id": str,                         # DOM id, e.g. "tab-toolcalling-batch"
  "kind": "toolcalling" | "reasoning",
  "label": str, "label_html": str, "tab_title": str,
  "active": bool,
  "case_prefix": str,                # e.g. "TOOLCALLING.batch." (glossary + I2 keys)
  "case_section_id": str,
  "case_docs_href": str, "case_docs_label": str,
  "toolbar_desc_html": str | None,
  "captured_note": str,              # "captured against ..." provenance line
  "candidates": [ candidate, ... ],  # compare-bar rows (ordered)
  "column_groups": [ {"key","label","span"} ],   # top header row
  "columns": [ {"sub","group_key","band","label","desc"} ],  # sub-case header row
  "rows": [ row, ... ],
  "glossary": [ {"label": str, "rows": [[sub, desc], ...]} ],
  "stats": { ... },                  # families/sub_cases/slots/real/parity/... counts
  "details_note_html": str | None,   # stream tabs only (explainer)
}

candidate = {
  "key": str,                        # compare key, e.g. "vllm_python-b-0-24-0"
  "impl": str,                       # engine group prefix: dynamo|vllm|sglang
  "label": str, "label_html": str,
  "default_bucket": "A"|"B"|"C",     # A=reference, A/B=checked-compare, C=off
  "version": str | None, "parse_mode": "batch"|"stream"|None,
}

row = {
  "model_label": str, "model_label_html": str,
  "family": str | None,
  "section": str | None,             # "Top-N models" / "Others" band banner
  "parser": parser_cell | None,      # the "Tool calling family" / parser column
  "cells": { sub: cell },            # one per column sub-case ("" cell = blank slot)
}

cell = {
  "kind": "cell" | "blank" | "missing",
  "case_id": str | None, "family": str | None, "sub": str,
  "col_group": str,                  # data-col-hide-group value
  "band": str,                       # sub-case band css class
  "fixture_href": str | None,        # link to the fixture yaml
  "status": "ok"|"problem"|"na"|"missing",   # default-reference overview status
  "cmp": { cand_key: {"sig":int,"leak":0|1,"na":0|1} } | None,
                                     # compare payload (was data-cmp) — the view
                                     # colors/counts from this + the picked reference
  "facts": [ fact, ... ],            # structured per-impl comparison facts
  "known_divergence": bool,          # ≠ corner mark (v1-vs-v2 by design)
  "tooltip": tooltip | None,         # raw popup data (view builds the DOM lazily)
}

fact = { "impl","status","present","agrees","intentional","reason","leak","error_kind" }
       # see markers.comparison_facts()

tooltip = {
  "head": str, "description": str,
  "input": {"kind":"text"|"chunks"|None, "text":str|None,
            "chunks":[{"delta_text","delta_token_ids","finish_reason"}]|None,
            "family": str|None},
  "candidates": [ {"key","label","impl","version","parse_mode","is_ref",
                   "block": output_block, "chunk_deltas": [per-chunk], "leak": bool} ],
  "baseline": {"impl","label","block"} | None,   # per-chunk chart baseline column
  "reasons": [ {"impl","label","reason","intentional"} ],  # divergence reasons
  "dynamo_notes": [ [label, text] ], "refs": [ [label, value] ],
  "leak_note": str | None,
  "na_note": str | None,             # n/a-stub cells: the explanation-only note
}

output_block = {"calls":[...], "normal_text":str, "error":..., "unavailable":str,
                "explanation":str} | None
per-chunk = {"deltas":[{index,name?,arguments?}], "normal_text":str}
            # one entry per input chunk, aligned to input.chunks (chart rows)

The model is the ONLY thing Python emits per DIS-2434 phase 3; while phases 1-2 keep
the Python-rendered HTML, `build_page`/`to_script_json` add the blob additively.
"""
from __future__ import annotations

import json
from typing import Any, Callable

SCHEMA_VERSION = 1


def to_script_json(page: dict) -> str:
    """Serialize the page model for an inlined `<script type="application/json">`.

    `\\uXXXX`-escapes the HTML-significant characters `<`, `>`, `&` so fixture text
    containing `</script>`, `<!--`, or `<script>` cannot break out of the script
    element. These are all valid JSON string escapes (`\\u003c` decodes to `<`), so the
    blob still parses AND round-trips to the exact original text — the canonical
    safe-embed. Greppability of the rendered file is an explicit non-goal (DIS-2434)."""
    raw = json.dumps(page, ensure_ascii=False, separators=(",", ":"))
    return (raw.replace("<", "\\u003c")
               .replace(">", "\\u003e")
               .replace("&", "\\u0026"))


def make_cell(**fields: Any) -> dict:
    """Normalize a cell dict to the schema, filling defaults for omitted keys. Both the
    toolcalling and reasoning extractors funnel through this so the cell SHAPE is
    single-sourced even while the two paths compute the fields differently."""
    cell = {
        "kind": "cell",
        "case_id": None,
        "family": None,
        "sub": "",
        "col_group": "",
        "band": "",
        "fixture_href": None,
        "status": "na",
        "cmp": None,
        "facts": [],
        "known_divergence": False,
        "tooltip": None,
    }
    cell.update(fields)
    return cell


def blank_cell(sub: str, col_group: str = "", band: str = "") -> dict:
    return make_cell(kind="blank", sub=sub, col_group=col_group, band=band, status="na")


def missing_cell(sub: str, family: str | None, col_group: str = "", band: str = "",
                 head: str | None = None) -> dict:
    return make_cell(
        kind="missing", sub=sub, family=family, col_group=col_group, band=band,
        status="missing",
        tooltip={"head": head or "", "description": "", "input": {"kind": None},
                 "candidates": [], "baseline": None, "reasons": [],
                 "dynamo_notes": [], "refs": [], "leak_note": None,
                 "na_note": "No fixture coverage for this case."},
    )


def build_page(meta: dict, tabs: list[dict], *, parser_ni: dict | None = None,
               legend_html: str = "") -> dict:
    """Assemble + lightly validate the whole-page model. `tabs` are already-built tab
    dicts from the toolcalling/reasoning extractors."""
    if not tabs:
        raise ValueError("model.build_page: no tabs")
    for i, tab in enumerate(tabs):
        for req in ("id", "kind", "label", "rows", "columns", "candidates", "stats"):
            if req not in tab:
                raise ValueError(f"model tab #{i} missing required key {req!r}")
    return {
        "schema": SCHEMA_VERSION,
        "meta": meta,
        "parser_ni": parser_ni or {},
        "legend_html": legend_html,
        "tabs": tabs,
    }
