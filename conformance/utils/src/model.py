# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One JSON data model for the conformance page (DIS-2434).

Python computes the MODEL only; a JS view (`assets/conformance_view.js`) renders the
table, tabs, compare bar, and popups from it. Comparison/parity SEMANTICS stay in
Python (`markers.py`/`impls.py` are the single source of truth) — this module
orchestrates them into a clean, documented, greppable structure and serializes it as
one inlined `<script type="application/json">` blob (so `file://` keeps working, no
fetch). The old parser-marker shorthand mini-language is gone: cells carry STRUCTURED
comparison facts and the view decides how to display them with full descriptive
labels (e.g. "vLLM Python batch parser", "Dynamo Rust stream parser").

Schema (top-down)
=================

page = {
  "schema": 2,                       # bumped on breaking model changes
  "strings": [str, ...],             # interned-string table (see Compaction below)
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
  "cmp": { cand_key: {"sig":int,"leak":0|1,"na":0|1,"err":0|1} } | None,
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

Compaction (schema 2)
=====================

`build_page` runs `_compact_page` so the inlined blob doesn't repeat itself; the JS
view's `hydratePage` (conformance_view.js) is the exact mirror and restores the
schema-1 shape in memory before anything renders, so no other consumer changes.
Python consumers of the raw blob (tests) call `hydrate_page` below. Four moves:

  page["strings"]            interned string table. Long (>=16 chars) strings that
                             occur 2+ times across the interned slots (see
                             `_iter_intern_slots`) are stored once and the slot holds
                             the int index. These slots are str|None by schema, so an
                             int is unambiguous.
  tab["cand_meta"]           { cand_key: {label/impl/version/parse_mode} } — per-cell
                             tooltip candidates repeat these identically for every
                             cell; they're stored once per tab and stripped from the
                             cells (kept inline only if a cell ever disagrees).
  tab["fixture_href_base"]   common directory prefix of the tab's fixture_href links;
                             cells keep only the suffix.
  tooltip["head"]            dropped when it equals "<case_id> — <family>" (the
                             overwhelmingly common case); hydration recomposes it.
"""
from __future__ import annotations

import json
import os.path
from typing import Any, Callable, Iterator

SCHEMA_VERSION = 2

# Interning: only strings at least this long are table-worthy (shorter ones cost as
# much as the index reference).
_INTERN_MIN_LEN = 16

# Tooltip-candidate fields that repeat identically for every cell of a tab.
_CAND_META_FIELDS = ("label", "impl", "version", "parse_mode")


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
    return _compact_page({
        "schema": SCHEMA_VERSION,
        "meta": meta,
        "parser_ni": parser_ni or {},
        "legend_html": legend_html,
        "tabs": tabs,
    })


# --- Schema-2 compaction (mirrored by hydratePage in conformance_view.js) -----------

def _iter_cells(page: dict) -> Iterator[tuple[dict, dict]]:
    for tab in page["tabs"]:
        for row in tab.get("rows", []):
            for cell in (row.get("cells") or {}).values():
                yield tab, cell


def _iter_intern_slots(page: dict) -> Iterator[tuple[Any, Any]]:
    """Yield every (container, key) slot whose string value may be interned. All of
    these are str|None by schema, so an int index is unambiguous. `refs` values are
    deliberately NOT interned (they may hold arbitrary fixture-provenance types).
    KEEP IN SYNC with hydratePage in conformance_view.js."""
    for _tab, cell in _iter_cells(page):
        for fact in cell.get("facts") or []:
            yield fact, "reason"
        tip = cell.get("tooltip")
        if not tip:
            continue
        yield tip, "description"
        yield tip, "na_note"
        yield tip, "leak_note"
        if tip.get("input"):
            yield tip["input"], "text"
        for r in tip.get("reasons") or []:
            yield r, "label"
            yield r, "reason"
        for pair in tip.get("dynamo_notes") or []:
            yield pair, 0
            yield pair, 1
        blocks = [c.get("block") for c in tip.get("candidates") or []]
        if tip.get("baseline"):
            blocks.append(tip["baseline"].get("block"))
        for b in blocks:
            if isinstance(b, dict):
                yield b, "explanation"
                yield b, "unavailable"
                yield b, "exception"


def _slot_get(container: Any, key: Any) -> Any:
    try:
        return container[key]
    except (KeyError, IndexError, TypeError):
        return None


def _compact_page(page: dict) -> dict:
    # Intern repeated long strings into one page-level table.
    counts: dict[str, int] = {}
    for container, key in _iter_intern_slots(page):
        v = _slot_get(container, key)
        if isinstance(v, str) and len(v) >= _INTERN_MIN_LEN:
            counts[v] = counts.get(v, 0) + 1
    strings: list[str] = []
    index: dict[str, int] = {}
    for container, key in _iter_intern_slots(page):
        v = _slot_get(container, key)
        if isinstance(v, str) and counts.get(v, 0) >= 2:
            if v not in index:
                index[v] = len(strings)
                strings.append(v)
            container[key] = index[v]
    if strings:
        page["strings"] = strings

    for tab in page["tabs"]:
        cells = [cell for row in tab.get("rows", [])
                 for cell in (row.get("cells") or {}).values()]

        # Tooltip-candidate meta: identical for every cell -> once per tab.
        meta: dict[str, dict] = {}
        conflicts: set[tuple[str, str]] = set()
        def _tip_cands(cell: dict) -> list[dict]:
            tip = cell.get("tooltip")
            return (tip.get("candidates") or []) if tip else []
        for cell in cells:
            for cand in _tip_cands(cell):
                key = cand.get("key")
                if not key:
                    continue
                slot = meta.setdefault(key, {})
                for f in _CAND_META_FIELDS:
                    if f not in cand:
                        continue
                    if f in slot and slot[f] != cand[f]:
                        conflicts.add((key, f))
                    else:
                        slot.setdefault(f, cand[f])
        cand_meta = {k: {f: v for f, v in fields.items() if (k, f) not in conflicts}
                     for k, fields in meta.items()}
        cand_meta = {k: fs for k, fs in cand_meta.items() if fs}
        if cand_meta:
            tab["cand_meta"] = cand_meta
            for cell in cells:
                for cand in _tip_cands(cell):
                    for f, v in (cand_meta.get(cand.get("key")) or {}).items():
                        if f in cand and cand[f] == v:
                            del cand[f]

        # fixture_href: hoist the common directory prefix.
        hrefs = [c["fixture_href"] for c in cells if c.get("fixture_href")]
        if len(hrefs) >= 2:
            prefix = os.path.commonprefix(hrefs)
            prefix = prefix[: prefix.rfind("/") + 1]
            if len(prefix) >= 8:
                tab["fixture_href_base"] = prefix
                for c in cells:
                    if c.get("fixture_href"):
                        c["fixture_href"] = c["fixture_href"][len(prefix):]

        # head: drop when it is the standard "<case_id> — <family>" composition.
        for cell in cells:
            tip = cell.get("tooltip")
            if (tip and cell.get("case_id") and cell.get("family")
                    and tip.get("head") == f"{cell['case_id']} — {cell['family']}"):
                del tip["head"]
    return page


def hydrate_page(page: dict) -> dict:
    """Restore the schema-1 shape in place (inverse of `_compact_page`). Python
    consumers of the raw blob (tests) call this; the JS view runs the mirrored
    hydratePage right after JSON.parse."""
    strings = page.get("strings") or []
    for container, key in _iter_intern_slots(page):
        v = _slot_get(container, key)
        if isinstance(v, int) and not isinstance(v, bool):
            container[key] = strings[v]
    for tab in page["tabs"]:
        cand_meta = tab.get("cand_meta") or {}
        base = tab.get("fixture_href_base") or ""
        for row in tab.get("rows", []):
            for cell in (row.get("cells") or {}).values():
                if base and cell.get("fixture_href"):
                    cell["fixture_href"] = base + cell["fixture_href"]
                tip = cell.get("tooltip")
                if not tip:
                    continue
                for cand in tip.get("candidates") or []:
                    for f, v in (cand_meta.get(cand.get("key")) or {}).items():
                        if f not in cand:
                            cand[f] = v
                if ("head" not in tip and cell.get("case_id") and cell.get("family")):
                    tip["head"] = f"{cell['case_id']} — {cell['family']}"
    return page
