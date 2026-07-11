#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate the conformance table (matrix of cell markers) from the YAML fixtures.

================================================================================
EXAMPLE OUTPUT (truncated; illustrative, NOT a snapshot of current fixtures
— run the script for the real table):

    | model          | parser     | 1 | 2.a | 2.b | 2.c | ... | 9 | 10 |
    |---|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
    | **Top-N models** |   |   |   |   |   |   |   |   |
    | Kimi K2.6      | kimi_k2    | = | =   | =   | V_pbS_rb | ... | = | =  |
    | gpt-oss        | harmony †  | S_rb | S_rb | n/a | S_rb? | ... | = | S_rb |
    | **Others** |   |   |   |   |   |   |   |   |
    | Mistral series | mistral    | S_rb | S_rb | n/a | V_pbS_rb | ... | = | S_rb |

================================================================================

Reads every `tests/parity/toolcalling/fixtures/<family>/TOOLCALLING.batch*.yaml` and emits
the conformance table.

Cell markers (Dynamo Rust + vLLM Rust + vLLM Python + SGLang):
  =     peer block matches the Dynamo Rust block (`expected.dynamo_rust`)
  D_rb      Dynamo Rust batch parser output diverges from the selected parser
  D_rs      Dynamo Rust stream parser output diverges from the selected parser
  V_pb      vLLM Python batch parser output diverges from the selected parser
  V_ps      vLLM Python stream parser output diverges from the selected parser
  V_rs      vLLM Rust stream parser output diverges from the selected parser; no V_rb exists
  S_rb      SGLang batch parser output diverges from the selected parser
  S_rs      SGLang stream parser output diverges from the selected parser
  ?         suffix means the divergent block has no `explanation:` yet
        (research-needed; we observed it but haven't classified it)
  !         suffix means the parser has `error: <substring>` (expected to crash)
  Combined markers, for example V_pbS_rb, mean multiple implementations diverge
  ·     Dynamo Rust-only fixture; peer blocks are unavailable or not captured
  n/a   family/case doesn't apply
  —     no fixture entry exists for this family/case yet

Footnote markers `†` (no vLLM peer) and `§` (no SGLang peer) are auto-derived
from `expected.<impl>.unavailable` across each family's cases.

Run:
    # Markdown table to stdout
    python3 tests/parity/generate_conformance_table.py toolcalling \
        > tests/parity/toolcalling/CONFORMANCE.md
    python3 tests/parity/generate_conformance_table.py toolcalling --mode stream \
        > tests/parity/toolcalling/CONFORMANCE.stream.md

    # HTML table with tabs, clickable YAML links, and hover tooltips. Prefer
    # conformance/utils/render_table_v2.sh so links are computed for the output
    # location.
    python3 tests/parity/generate_conformance_table.py toolcalling --html \
        > tests/parity/toolcalling/CONFORMANCE.html

CONFORMANCE.html is for local viewing only; don't check it in.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import html as html_lib
import json
import os
import re
import subprocess
import sys
import tempfile
import zoneinfo
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from tests.parity import common
from tests.parity.common import TOP_N_TOOL_CALLING_FAMILIES as TOP_N_FAMILIES
from tests.parity.common import (
    linkify_text_html,
    parity_cell_class,
)
from tests.parity.markup import colorize_markup, colorize_stream_deltas
from tests.parity.reasoning import table as reasoning_table
from tests.parity.toolcalling import table as toolcalling_table

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests/parity/toolcalling/fixtures"
# Batch-on-stream overlay: each engine's STREAMING parser run over the v1 batch
# fixture text, keyed by the v1 batch case id. The batch-on-stream tab reuses the
# v1 batch taxonomy/input but renders these stream outputs as `expected`.
STREAM_ON_BATCH_FIXTURES = REPO_ROOT / "tests/parity/toolcalling/fixtures-batch-on-stream-v2"
TOOLCALLING_CASES_MD = REPO_ROOT / "lib/parsers/TOOLCALLING_CASES.md"
# Streaming cases use our own doc (renumbered to the batch+10 taxonomy), not the
# dynamo-synced TOOLCALLING_CASES.md.
TOOLCALLING_STREAMING_V2_CASES_MD = REPO_ROOT / "lib/parsers/TOOLCALLING_STREAMING_V2_CASES.md"
PYPROJECT_TOML = REPO_ROOT / "pyproject.toml"
TEMPLATE_DIR = REPO_ROOT / "tests/parity"

RUST_TOOL_CALLING_DIR = REPO_ROOT / "lib/parsers/src/tool_calling"

# Implementation identity (keys, aliases, display, markers) lives in impls.py — the
# single source of truth (audit B1). build_stage_conformance stages impls.py next to
# this file so the import works from the staged tests/parity layout too.
from impls import (  # noqa: E402
    BASELINE_IMPL,
    BATCH_IMPL_KEYS,
    ENGINE_LETTER,
    IMPL_DISPLAY,
    IMPL_KEYS,
    IMPL_LANG_MARKER,
    LEGACY_IMPL_ALIASES,
    PARSER_NOT_CAPTURED,
    PEER_IMPL_KEYS,
    STREAM_IMPL_KEYS,
)

# Comparison + marker semantics live in markers.py (audit B5); re-exported here so the
# rendering code below and the test suite keep referring to them as module attributes.
from markers import (  # noqa: E402,F401
    VLLM_RUST_UNAVAILABLE,
    _BATCH_MODE_MARKER,
    _PARSER_ERROR_RE,
    _STREAM_MODE_MARKER,
    _TOOL_CALL_MARKUP_RE,
    _block_tool_call_leaks,
    _canonical_impl_key,
    _canonical_tool_output,
    _dynamo_tool_call_leak,
    _expected,
    _explanation,
    _impl_get,
    _impl_keys_for_output_kind,
    _impl_mode_label_html,
    _impl_mode_letter,
    _impl_mode_marker_html,
    _impl_mode_suffix,
    _is_parser_error_unavailable,
    _is_todo_unavailable,
    _legacy_impl_keys,
    _marker_html,
    _marker_span_html,
    _norm_calls,
    _normalize_impl_mapping,
    _overview_status,
    _overview_status_attrs,
    _parity_marker,
    _parser_marker,
    _parser_marker_attrs,
    _parser_marker_spans,
    _selected_parity_marker,
    _selected_parity_suffix,
    _sob_calls_consistent,
    _sob_cell_text,
    _sob_marker_attrs,
    _sob_marker_spans,
    _sob_status,
    _sob_status_attrs,
    _stream_cross_suffix,
    _stream_parity_explainer_html,
    _stream_xeng_marker,
    peer_status,
)

# Fixture loading + sub-case taxonomy live in fixtures.py (audit B5); re-exported here
# so the rendering code and tests keep referring to them as module attributes. The
# captured-with map is a shared mutable dict (load_all_cases mutates it in place).
import fixtures  # noqa: E402  (module handle: version radios repoint fixtures.FIXTURES)
from fixtures import (  # noqa: E402,F401
    BATCH_SUB_CASE_GROUPS,
    SPLIT_PARENT_SUBCASES,
    STREAM_SUB_CASE_GROUPS,
    SUB_CASE_GROUPS_BY_MODE,
    _CAPTURED_WITH_BY_MODE,
    _SUB_CASE_GROUP_KEY_BY_LABEL_BY_MODE,
    _SUB_CASE_GROUP_KEY_BY_SUB_BY_MODE,
    _attach_streamv2_batch_expected,
    _build_family_inheritance,
    _build_family_to_rust_ref,
    _derive_no_peer_sets,
    _derive_stream_expected,
    _discover_sub_cases,
    _display_order,
    _group_by_sub,
    _group_index_by_sub,
    _natural_sub_sort_key,
    _normalize_split_parent_cases,
    _sub_sort_key,
    _subcase_band_class,
    _subcase_group_key,
    family_suffix,
    load_all_cases,
)


# Row-label / visibility overrides keyed by tool calling family; ‡ is explained
# by the legend note in conformance_table.html.j2.
_TOOL_CALLING_LABEL_OVERRIDES = {
    "qwen3_coder": "Qwen 3 Coder / Nemotron V3‡",
}
# nemotron_nano: an alias for qwen3_coder, hide to avoid duplicate row
# nemotron_deci: for older v2 nemotron models, hide to avoid confusion with nemotron v3 models
_HIDDEN_TOOL_CALLING_FAMILIES = {"nemotron_deci", "nemotron_nano"}
_V2_TOP_N_TOOL_CALLING_FAMILIES = []
for family in TOP_N_FAMILIES:
    _V2_TOP_N_TOOL_CALLING_FAMILIES.append(family)
    if family == "harmony":
        _V2_TOP_N_TOOL_CALLING_FAMILIES.append("harmony_text")


def _model_label_html(model: str) -> str:
    """Escape a model label, styling any ‡ marker like the †/§ suffixes."""
    return html_lib.escape(model).replace("‡", '<span class="parser-suffix">‡</span>')


def _make_jinja_env() -> Environment:
    return Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        trim_blocks=False,
        lstrip_blocks=True,
        undefined=StrictUndefined,
    )


def _read_asset(name: str) -> str:
    """Inline a static CSS/JS asset (audit B7) into the rendered page.

    The page is a single self-contained HTML file (no external requests), so the
    CSS/JS live in `tests/parity/assets/` as editable files and get inlined at
    render time rather than hard-coded in the Jinja template.
    """
    return (TEMPLATE_DIR / "assets" / name).read_text(encoding="utf-8")


def _impl_status_css() -> str:
    """Generate the per-implementation status/marker CSS (audit B6) from the one
    impl identity table (IMPL_KEYS + the legacy reasoning aliases) instead of
    hand-maintaining one rule per implementation. The v2 keys use `.marker-<impl>`
    show rules; the legacy reasoning keys use `content: attr(data-marker-<legacy>)`
    with a `:not([data-marker-<canonical>])` guard. Colors are unchanged; only the
    source is de-duplicated, so adding an impl is one `ImplSpec` row."""
    lines: list[str] = []

    def group(selectors: list[str], body: str) -> None:
        lines.append(",\n".join(selectors) + f" {{ {body} }}")

    # ----- v2 implementations: status background by selected parser -----
    overview_bg = {"ok": "#76b884", "problem": "#db7777"}
    for status, color in overview_bg.items():
        group([f'.view-overview.parser-{k} td.cell[data-status-{k}="{status}"]' for k in IMPL_KEYS],
              f"background: {color}; color: {color};")
    group([f'.view-overview.parser-{k} td.cell[data-status-{k}="{s}"]'
           for k in IMPL_KEYS for s in ("na", "todo")],
          "background: #d3d8de; color: #d3d8de;")
    details_bg = {"ok": "#bfe3c6", "problem": "#efb3b3"}
    for status, color in details_bg.items():
        group([f'.view-details.parser-{k} td.cell[data-status-{k}="{status}"]' for k in IMPL_KEYS],
              f"background: {color};")
    group([f'.view-details.parser-{k} td.cell[data-status-{k}="{s}"]'
           for k in IMPL_KEYS for s in ("na", "todo")],
          "background: #e4e8ec;")
    for status, color in (("na", "#aeb6bf"), ("todo", "#6b7280")):
        group([f'.view-details.parser-{k} td.cell[data-status-{k}="{status}"]::before' for k in IMPL_KEYS],
              f"color: {color};")
        group([f'.view-details.parser-{k} td.cell[data-status-{k}="{status}"] .cell-marker' for k in IMPL_KEYS],
              f"color: {color};")
    for k in IMPL_KEYS:
        lines.append(f".view-details.parser-{k} td.cell .marker-{k} {{ display: flex; }}")
    # Hide the per-engine status markers in Conformance mode. This MUST stay right
    # after the status-marker show rules above (equal specificity, so source order
    # decides) and before the parity-marker show rules below, or both the status and
    # parity markers render in the same cell and visibly overlap (audit B7 regression:
    # this rule used to live in the template and was wrongly moved into the static
    # asset, which inlines *before* this generated block and lost the tie).
    lines.append(".view-details.parity-mode td.cell .cell-marker { display: none; }")
    for k in IMPL_KEYS:
        lines.append(f".view-details.parity-mode.parser-{k} td.cell .marker-parity-{k} {{ display: flex; }}")

    # ----- legacy reasoning implementations (dynamo/vllm/sglang) -----
    legacy = list(LEGACY_IMPL_ALIASES.items())  # (legacy, canonical)
    legacy_keys = [lg for lg, _ in legacy]
    for status, color in overview_bg.items():
        group([f'.view-overview.parser-{lg} td.cell[data-status-{lg}="{status}"]' for lg in legacy_keys],
              f"background: {color}; color: {color};")
    group([f'.view-overview.parser-{lg} td.cell[data-status-{lg}="{s}"]'
           for lg in legacy_keys for s in ("na", "todo")],
          "background: #d3d8de; color: #d3d8de;")
    for status, color in details_bg.items():
        group([f'.view-details.parser-{lg} td.cell[data-status-{lg}="{status}"]' for lg in legacy_keys],
              f"background: {color};")
    group([f'.view-details.parser-{lg} td.cell[data-status-{lg}="{s}"]'
           for lg in legacy_keys for s in ("na", "todo")],
          "background: #e4e8ec;")
    for status, color in (("na", "#aeb6bf"), ("todo", "#6b7280")):
        group([f'.view-details.parser-{lg} td.cell[data-status-{lg}="{status}"]::before' for lg in legacy_keys],
              f"color: {color};")
    for lg, canon in legacy:
        lines.append(
            f'.view-details.parser-{lg} td.cell:not([data-marker-{canon}])::before '
            f"{{ content: attr(data-marker-{lg}); }}")
    for lg, canon in legacy:
        lines.append(
            f'.view-details.parity-mode.parser-{lg} td.cell:not([data-marker-parity-{canon}])::before '
            f"{{ content: attr(data-marker-parity-{lg}); }}")

    # ----- per-version coloring + tooltip for the TC v1 (batch) tab -----
    # Keyed on parser-<impl> + verv-<impl>-<slug> + data-status-<impl>-<slug>, so
    # these override the pinned rules above when a non-default version is active.
    # Cells without a per-version attr fall back to the pinned data-status-<impl>.
    lines.append(".ttip .verrow { display: none; }")
    for legacy_impl, versions in _batch_impl_versions().items():
        canon = _VERSION_LEGACY_TO_CANON[legacy_impl]
        for version in versions:
            slug = toolcalling_table._version_slug(version)
            # Version-aware tooltip: show only the active version's output block.
            lines.append(
                f"body.verv-{canon}-{slug} .ttip .verrow-{canon}-{slug} "
                "{ display: block; }")
            base = f'.view-overview.parser-{canon}.verv-{canon}-{slug} td.cell[data-status-{canon}-{slug}='
            for status, color in overview_bg.items():
                group([f'{base}"{status}"]'], f"background: {color}; color: {color};")
            group([f'{base}"{s}"]' for s in ("na", "todo")],
                  "background: #d3d8de; color: #d3d8de;")
            dbase = f'.view-details.parser-{canon}.verv-{canon}-{slug} td.cell[data-status-{canon}-{slug}='
            for status, color in details_bg.items():
                group([f'{dbase}"{status}"]'], f"background: {color};")
            group([f'{dbase}"{s}"]' for s in ("na", "todo")], "background: #e4e8ec;")
    return "\n".join(lines)


def _ref_text(value: Any) -> str:
    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _build_conformance_tooltip_html(
    *,
    head: str,
    description: str | None = None,
    input_label: str | None = None,
    input_html: str | None = None,
    output_sections: list[tuple[str, str]] | None = None,
    divergent_reasons: str | None = None,
    divergent_reasons_html: str | None = None,
    leak_label: str | None = None,
    leak_text: str | None = None,
    extra_sections: list[tuple[str, str]] | None = None,
    chart: tuple[str, str] | None = None,
    refs: list[tuple[str, Any]] | None = None,
    html_section_labels: bool = False,
) -> str:
    parts = ['<div class="ttip">']
    if head:
        parts.append(f'<div class="ttip-head">{html_lib.escape(head)}</div>')
    if description:
        parts.append(f'<pre class="ttip-pre">{html_lib.escape(description)}</pre>')

    def add_section(
        label: str, body_html: str, wrap_class: str | None = None, leak: bool = False
    ) -> None:
        # `html_section_labels` callers pass safe HTML (e.g. `D<sub>RS</sub>`); all
        # such labels are generator-controlled, never user input.
        shown = label if html_section_labels else html_lib.escape(label)
        # A leaking candidate gets a red ↯ after the label, e.g. "… (stream): ↯".
        marker = ' <span class="ttip-leak">↯</span>' if leak else ""
        section = (
            f'<div class="ttip-section">{shown}:{marker}</div>'
            f'<pre class="ttip-pre">{body_html}</pre>'
        )
        # Optional wrapper lets a whole labeled section toggle (per-version blocks).
        if wrap_class:
            section = f'<span class="{wrap_class}">{section}</span>'
        parts.append(section)

    if input_label and input_html is not None:
        add_section(input_label, input_html)

    for section in output_sections or []:
        add_section(*section)

    if divergent_reasons_html:
        add_section("Divergent reasons", divergent_reasons_html)
    elif divergent_reasons:
        add_section("Divergent reasons", linkify_text_html(divergent_reasons))

    if leak_label and leak_text:
        add_section(leak_label, linkify_text_html(leak_text))

    for label, body_html in extra_sections or []:
        add_section(label, body_html)

    if chart is not None:
        chart_label, chart_html = chart
        parts.append(
            f'<div class="ttip-section">{html_lib.escape(chart_label)}:</div>'
        )
        parts.append(chart_html)

    for label, value in refs or []:
        if value:
            add_section(label, linkify_text_html(_ref_text(value)))

    parts.append("</div>")
    return "".join(parts)


def _resolve_output_path(
    output_path: Path | None,
    artifact_root: Path,
    default_output: str,
) -> Path:
    path = output_path or Path(default_output)
    if not path.is_absolute():
        path = artifact_root / path
    return path.resolve()


def _display_path(path: Path, artifact_root: Path) -> str:
    try:
        return path.relative_to(artifact_root).as_posix()
    except ValueError:
        return path.as_posix()


# Destination-aware link resolution lives in tests.parity.common
# (`set_links` / `LINKS`), shared by the v1 PARITY and v2 CONFORMANCE generators.


_VISIBLE_CONFORMANCE_REPLACEMENTS = (
    ("All engines parity", "All engines match"),
    ("Parity harness flags used for this result:", "Conformance harness flags used for this result:"),
    ("Not set by this parser-level parity harness:", "Not set by this parser-level conformance harness:"),
    ("parser-level parity harness", "parser-level conformance harness"),
    ("parser-level parity result", "parser-level conformance result"),
    ("captured-peer parity", "captured-peer conformance"),
    ("Dynamo Parser Parity Table", "Dynamo Parser v2 Conformance Table"),
    ("Dynamo Reasoning Parser - Parity Table", "Dynamo Reasoning Parser v2 Conformance Table"),
    ("Dynamo Tool Calling Parser - Parity Table", "Dynamo Tool Calling Parser v2 Conformance Table"),
    ("Parity Table", "Conformance Table"),
    ("parity table", "conformance table"),
    ("tests/parity/README.md", "Dynamo-synced parser fixture README"),
)


def _scrub_visible_conformance_text(text: str) -> str:
    """Keep the v2 page conformance-branded without renaming internal CSS/JS hooks."""
    for old, new in _VISIBLE_CONFORMANCE_REPLACEMENTS:
        text = text.replace(old, new)
    return text


def _commit_sha() -> str | None:
    """HEAD SHA at table-generation time, or None if not in a git tree."""
    try:
        out = (
            subprocess.check_output(
                ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        return out or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _peer_versions() -> dict[str, str]:
    """Extract pinned vLLM Python / SGLang versions from pyproject.toml.

    Matches a line like `"vllm[flashinfer,runai,otel]==X.Y.Z",` (TOML is
    not parsed — the regex is sufficient and avoids a tomllib import on
    older Pythons running this script outside a Python 3.11+ env)."""
    out: dict[str, str] = {}
    if not PYPROJECT_TOML.exists():
        return out
    text = PYPROJECT_TOML.read_text()
    for name in ("vllm", "sglang"):
        m = re.search(rf'"{name}(?:\[[^\]]*\])?==([0-9][^"]*)"', text)
        if m:
            out[_canonical_impl_key(name)] = m.group(1)
    return out


def _build_display_groups(
    cases: dict, labels: dict[str, str]
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Return `(top_n, others)` as `[(label, family), ...]` lists.

    Top-N: families listed in `_V2_TOP_N_TOOL_CALLING_FAMILIES`, in that exact order.
    Others: every YAML-discovered family not in TOP_N, sorted by label.
    Missing labels fall back to the family ID.
    """
    families = {
        fam for fam, _ in cases.keys() if fam not in _HIDDEN_TOOL_CALLING_FAMILIES
    }

    def label_of(fam: str) -> str:
        return _TOOL_CALLING_LABEL_OVERRIDES.get(fam, labels.get(fam, fam))

    top_n = [
        (label_of(f), f) for f in _V2_TOP_N_TOOL_CALLING_FAMILIES if f in families
    ]
    other_fams = sorted(
        families - set(_V2_TOP_N_TOOL_CALLING_FAMILIES),
        key=lambda f: label_of(f).lower(),
    )
    others = [(label_of(f), f) for f in other_fams]
    return top_n, others


def cell_for(
    case: dict | None,
    impl_keys: tuple[str, ...] = BATCH_IMPL_KEYS,
    marker_mode: str | None = _BATCH_MODE_MARKER,
) -> str:
    if case is None:
        return "—"
    expected = _expected(case)
    dyn = _impl_get(expected, BASELINE_IMPL)
    if not isinstance(dyn, dict):
        return "n/a"
    # Dynamo parser v2 unavailable for this case: distinguish "not yet implemented"
    # (TODO, the whole family) from a structural n/a (e.g. a token parser can't
    # consume a character-split fixture per-chunk).
    if "unavailable" in dyn:
        return "…" if _is_todo_unavailable(dyn) else "n/a"
    parts: list[str] = []
    peer_kinds: dict[str, str] = {}
    for impl in (peer for peer in impl_keys if peer != BASELINE_IMPL):
        kind, unknown = peer_status(case, dyn, impl)
        peer_kinds[impl] = kind
        letter = (
            _impl_mode_letter(impl) + _impl_mode_suffix(impl, marker_mode)
            if marker_mode is not None
            else ENGINE_LETTER[impl]
        )
        if kind == "div":
            parts.append(f"{letter}?" if unknown else letter)
        elif kind == "err":
            parts.append(f"{letter}!")

    # `explanation:` on the `expected.dynamo_rust` block flags Dynamo parser v2 output as
    # leaking tool call markup only when it also leaves residual
    # `normal_text`. The Dynamo parser v2 can have non-leak reasons for dropped malformed
    # markup, so don't mark those as `↯`.
    if isinstance(dyn, dict) and _dynamo_tool_call_leak(dyn):
        if all(kind in {"unavail", "na"} for kind in peer_kinds.values()):
            return "↯·"
        if parts:
            return "↯" + "".join(parts)
        return "↯"

    if parts:
        return "".join(parts)
    if all(kind in {"unavail", "na"} for kind in peer_kinds.values()):
        return "·"
    return "="


def render_row(
    model: str,
    family: str,
    cases: dict,
    sub_cases: list[str],
    no_vllm: set[str],
    no_sglang: set[str],
    inheritance: dict[str, dict],
) -> str:
    cells = [cell_for(cases.get((family, sub))) for sub in sub_cases]
    parser_label = _parser_label_markdown(family, no_vllm, no_sglang, inheritance)
    return f"| {model} | {parser_label} | " + " | ".join(cells) + " |"


_LEGEND_MD = (
    "**Legend:** "
    "`=` all captured peers match Dynamo Rust (`expected.dynamo_rust`) · "
    "`·` Dynamo Rust-only fixture (peers unavailable or not captured) · "
    "`D_rb` (Dynamo Rust batch parser) · "
    "`D_rs` (Dynamo Rust stream parser) · "
    "`V_pb` (vLLM Python batch parser) · "
    "`V_ps` (vLLM Python stream parser) · "
    "`V_rs` (vLLM Rust stream parser; no `V_rb` exists) · "
    "`S_rb` (SGLang batch parser) · "
    "`S_rs` (SGLang stream parser) · "
    "a marker means that implementation diverges from the selected parser output; intentional divergences have `explanation:` · "
    "`?` research-needed suffix (e.g. V_pb?, S_rs? — diverges with no `explanation:` yet) · "
    "`↯` selected parser leaks tool call markup into `normal_text` "
    "(captured peer output can legitimately show imperfect engine behavior) · "
    "`!` expected-error suffix (e.g. V_pb!, S_rs! — engine crashes by design) · "
    "`✗` engine parser ran but failed to parse this input (recorded as `unavailable: … parser not captured: …`); distinct from `n/a` (not run) · "
    "`n/a` not applicable (includes families the Dynamo v2 stream parser doesn't implement) · "
    "`—` missing fixture coverage · "
    "`†` (tool calling parser column) = no vLLM Python peer parser for this family · "
    "`§` (tool calling parser column) = no SGLang peer parser for this family."
    "\n\n"
    "`‡` Nemotron V3 (Ultra) reuses the qwen3_coder tool calling parser; "
    "Nemotron V1 / V2 (DeciLM) is removed from the chart for being an older "
    "generation, but the nemotron_deci parser is still supported."
)


def _common_legend_html(
    peer_versions: list[tuple[str, str]] | None = None,
    peer_versions_href: str | None = None,
) -> str:
    versions_html = ""
    if peer_versions:
        versions = " · ".join(
            f"{html_lib.escape(name)} <code>{html_lib.escape(version)}</code>"
            for name, version in peer_versions
        )
        versions_html = (
            "<p>"
            "<strong>Peer parser versions</strong> pinned in "
            f'<a href="{html_lib.escape(peer_versions_href or common.LINKS["pyproject_stub"], quote=True)}">pyproject.toml</a>: '
            f"{versions}."
            "</p>"
        )
    return (
        "<p><strong>Legend:</strong></p>"
        '<ul class="marker-defs">'
        '<li><span style="color:#0a7d2c"><strong>green</strong></span> = the selected <strong>Reference</strong> parser output is clean — no structured markup (tool-call or reasoning) leaked into the visible <code>normal_text</code>. A clean Reference is green whether or not any Compare parser is selected.</li>'
        '<li><span style="color:#b00"><strong>red</strong> (↯)</span> = the Reference parser leaks structured markup (tool-call or reasoning) into the visible <code>normal_text</code>.</li>'
        '<li><span style="color:#aaa"><strong>n/a</strong></span> = the selected Reference is not applicable for this case (for example the Dynamo v2 stream parser is not implemented for this family).</li>'
        '<li><span style="color:#8a6d3b">—</span> missing fixture coverage.</li>'
        '<li>In the <strong>Detailed</strong> view the number on a cell (with a <span style="color:#8a6d3b">Δ</span> suffix, e.g. <span style="color:#8a6d3b">2Δ</span>) = how many selected <strong>Compare</strong> parsers diverge from the Reference (<span style="color:#0a7d2c">=</span> means every selected Compare matches). A divergence with no <code>explanation:</code> yet is flagged <span style="color:#b00">?</span> (research needed); <span style="color:#b00">!</span> marks an engine that errors by design; <span style="color:#b00">✗</span> means the parser ran but failed to parse.</li>'
        '<li><strong>v1</strong> = the stable batch parser crate (<code>parsers/v1/src/...</code>, <code>dynamo-parsers</code>); <strong>v2</strong> = the WIP streaming parser crate (<code>parsers/v2/src/...</code>, <code>dynamo-parsers-v2</code>).</li>'
        '<li><span class="parser-suffix">†</span> no vLLM Python peer parser for this family. &nbsp; <span class="parser-suffix">§</span> no SGLang peer parser for this family. &nbsp; <span class="parser-suffix">‡</span> Nemotron V3 (Ultra) reuses the qwen3_coder parser.</li>'
        "</ul>"
        f"{versions_html}"
    )


def render_markdown(
    cases: dict,
    sub_cases: list[str],
    no_vllm: set[str],
    no_sglang: set[str],
    top_n: list[tuple[str, str]],
    others: list[tuple[str, str]],
) -> str:
    inheritance = _build_family_inheritance(_build_family_to_rust_ref())
    header = "| model | Tool calling family | " + " | ".join(sub_cases) + " |"
    sep = "|---|---|" + ":-:|" * len(sub_cases)
    lines = [header, sep]
    lines.append("| **Top-N models** |   |" + "   |" * len(sub_cases))
    for model, fam in top_n:
        lines.append(
            render_row(model, fam, cases, sub_cases, no_vllm, no_sglang, inheritance)
        )
    lines.append("| **Others** |   |" + "   |" * len(sub_cases))
    for model, fam in others:
        lines.append(
            render_row(model, fam, cases, sub_cases, no_vllm, no_sglang, inheritance)
        )
    lines.append("")
    lines.append(_LEGEND_MD)
    return "\n".join(lines)


_IMPL_DISPLAY = IMPL_DISPLAY


def _short_unavailable(reason: object) -> str:
    """A family the Dynamo v2 stream parser doesn't implement is a plain neutral
    `n/a` (matching the v1 table, which has no "TODO" concept) — collapse the verbose
    "… not yet implemented for this family; …" reason to `n/a`. Other reasons (missing
    peer parser, capture error) stay verbatim since they carry specifics."""
    r = str(reason)
    return "n/a" if "not yet implemented" in r else r


def _format_output_block_html(block, family: str | None = None) -> str:
    """HTML rendering of an `expected.<impl>` block for tooltips.
    Applies _colorize_xml to `normal_text` so raw model output the engine
    failed to parse shows the same tag coloring as the input."""
    if not isinstance(block, dict):
        return html_lib.escape("(no expectation)")
    if block.get("unavailable"):
        # Un-implemented Dynamo v2 family reads as a plain "n/a" (no "unavailable:"
        # prefix, no "not yet implemented" prose) — same clean placeholder as the v1 table.
        if _is_todo_unavailable(block):
            return html_lib.escape("n/a")
        return html_lib.escape(f"unavailable: {_short_unavailable(block['unavailable'])}")
    if "error" in block:
        return html_lib.escape(f"error matching {block['error']!r}")
    nt = block.get("normal_text", "") or ""
    calls = block.get("calls") or []
    if calls:
        rendered = ", ".join(
            f"{c.get('name', '?')}({json.dumps(c.get('arguments', {}), ensure_ascii=False)})"
            for c in calls
        )
        calls_line = html_lib.escape(f"calls=[{rendered}]")
    else:
        calls_line = "calls=[]"
    nt_line = f"normal_text='{colorize_markup(nt, family)}'"
    return f"{nt_line}\n{calls_line}"


def _cand_section_body(block, family: str | None = None) -> str:
    """A compare candidate's tooltip section body: its output block plus its own
    `explanation:` (when present). The note lives INSIDE the candidate's toggleable
    section — so it shows only when that candidate is selected — instead of a global
    cross-engine "Divergent reasons" blob that would name unselected engines."""
    body = _format_output_block_html(block, family)
    note = _explanation(block)
    if note:
        body += "\nexplanation: " + html_lib.escape(str(note))
    return body


def _dynamo_note_sections(case: dict) -> list[tuple[str, str]]:
    """A baseline-side rationale for the Dynamo Rust output, rendered as its own
    tooltip section. `_tooltip_for` only explains PEER divergences (it skips the
    baseline), so a deliberate Dynamo behavior — e.g. dropping an unterminated
    Harmony tool call per dynamo #10366 — has no other surface. Sourced from a
    case-level `dynamo_note:` in the fc-local v2 fixtures; include a full URL so
    `linkify_text_html` makes the PR reference clickable."""
    note = case.get("dynamo_note")
    if not note:
        return []
    return [("Dynamo recovery contract", linkify_text_html(str(note)))]



# Candidate labels look like "vLLM Python 0.23.0 (stream)". Sort sections by engine
# (label prefix) ascending, then version DESCENDING within the engine — matching the
# compare bar's latest-first ordering. Two stable passes: version-desc, then prefix.
_CAND_LABEL_VER_RE = re.compile(r"^(.*?)(\d[\w.\-]*)(\s*\(.*\))?\s*$")


def _sort_candidate_sections(sections: list) -> None:
    def parts(label: str):
        m = _CAND_LABEL_VER_RE.match(label)
        if not m:
            return (label, "0", "")
        return (m.group(1), m.group(2) or "0", m.group(3) or "")

    sections.sort(
        key=lambda sec: toolcalling_table._version_sort_key(parts(sec[0])[1]),
        reverse=True,
    )
    sections.sort(key=lambda sec: (parts(sec[0])[0], parts(sec[0])[2]))


def _build_tooltip_html(case: dict, dyn, output_kind: str = "batch") -> str:
    """Rich HTML hover tooltip: head, input (colorized), per-engine output,
    divergence reasons. Returns the full `<div class="ttip">...</div>`.

    `output_kind` ("batch" or "stream") labels every per-engine output block so the
    reader can tell what produced it — the batch-on-stream tab feeds batch text to
    each engine's STREAMING parser, so its blocks read "Dynamo stream" etc."""
    case_id = case.get("__case_id", "")
    family = case.get("__family")
    desc = case.get("description") or ""
    if case_id and family:
        head = f"{case_id} — {family}"
    else:
        head = case_id or str(family or "")

    input_label = None
    input_html = None
    model_text = case.get("model_text")
    if isinstance(model_text, str) and model_text:
        input_label = "Input"
        input_html = f"input_text='{colorize_markup(model_text, family)}'"
    chunks = case.get("chunks")
    if isinstance(chunks, list) and chunks:
        chunk_lines = []
        chunk_html = colorize_stream_deltas(chunks, family)
        for i, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                continue
            suffix = ""
            if chunk.get("finish_reason"):
                suffix = " finish_reason=" + html_lib.escape(
                    str(chunk["finish_reason"])
                )
            tid = chunk.get("delta_token_ids")
            tid_str = (
                f" delta_token_ids={tid}"
                if family == "harmony" and isinstance(tid, list)
                else ""
            )
            chunk_lines.append(
                f"{i}: delta_text='{chunk_html[i]}'{tid_str}{suffix}"
            )
        input_label = "Input chunks"
        input_html = "\n".join(chunk_lines)

    expected = _expected(case)
    impl_keys = _impl_keys_for_output_kind(output_kind)

    def _norm(b):
        return {
            "calls": b.get("calls") or [],
            "normal_text": b.get("normal_text") or "",
        }

    n_dyn = _norm(dyn) if isinstance(dyn, dict) else None
    all_engines_parity = isinstance(dyn, dict) and all(
        isinstance(_impl_get(expected, i), dict)
        and not _impl_get(expected, i).get("unavailable")
        and "error" not in _impl_get(expected, i)
        and _norm(_impl_get(expected, i)) == n_dyn
        for i in impl_keys
    )

    ver_status = case.get("__ver_status") or {}
    cmp_items = case.get("__cmp") or []

    output_sections: list[tuple] = []
    if cmp_items:
        # Merged tab: one section per candidate (parser flavor + version), each
        # wrapped in cand-<key> and toggled by the Base/Compare selection so the
        # tooltip shows exactly the candidates being compared.
        for item in cmp_items:
            blk = item["block"]
            output_sections.append((
                item["label"],
                _cand_section_body(blk, family),
                f"cand cand-{item['key']}",
                isinstance(blk, dict) and _block_tool_call_leaks(blk),
            ))
    elif ver_status:
        # Compare model (batch tab): one section per candidate (parser+version),
        # each wrapped in cand-<key> and toggled by the Base/Compare selection so
        # the tooltip shows exactly the candidates being compared.
        for impl in ("dynamo_rust", "vllm_rust", "vllm_python", "sglang_python"):
            for slug, info in (ver_status.get(impl) or {}).items():
                key = f"{impl}-{slug}"
                output_sections.append((
                    _full_label(impl, info["version"], "batch"),
                    _cand_section_body(info["block"], family),
                    f"cand cand-{key}",
                    isinstance(info["block"], dict) and _block_tool_call_leaks(info["block"]),
                ))
    elif all_engines_parity:
        output_sections.append(
            (
                f"All available engines match ({output_kind})",
                _format_output_block_html(dyn, family),
            )
        )
    else:
        for impl in impl_keys:
            output_sections.append(
                (
                    f"{_IMPL_DISPLAY[impl]} {output_kind}",
                    _format_output_block_html(_impl_get(expected, impl), family),
                )
            )

    # Engine ascending, version DESCENDING within the engine (latest first),
    # matching the compare-bar ordering.
    if cmp_items or ver_status:
        _sort_candidate_sections(output_sections)

    reasons = _tooltip_for(case, dyn, impl_keys) if isinstance(dyn, dict) else ""

    chart = _per_chunk_chart_html(case, output_kind)
    # A CANDIDATE chart (columns = compare candidates, `data-cand`-keyed) already
    # carries every candidate's output + explanation in its assembled/output row,
    # so the per-candidate list sections above it would repeat the same info —
    # drop them. Legacy impl-column charts keep the sections for versioned cells.
    cand_chart = chart is not None and "data-cand" in chart[1]

    dyn_leak = _dynamo_tool_call_leak(dyn) if isinstance(dyn, dict) else None
    return _build_conformance_tooltip_html(
        head=head,
        description=desc,
        input_label=None if chart else input_label,
        input_html=None if chart else input_html,
        # When the chart is shown it carries a final "assembled" row per impl, so
        # the separate per-engine output blocks would be redundant — drop them.
        # Exception: versioned candidates (__ver_status) keep their per-candidate
        # `cand cand-<impl>-<slug>` sections ONLY while the chart is impl-keyed;
        # a candidate chart replaces them entirely.
        output_sections=(
            None
            if cand_chart
            else (
                output_sections
                if (ver_status or cmp_items)
                else (None if chart else output_sections)
            )
        ),
        # In the compare model each candidate's reason lives in its own section
        # (via _cand_section_body), so suppress the global cross-engine blob that
        # would name engines not in the current Base/Compare selection.
        divergent_reasons=None if (cmp_items or ver_status) else (reasons or None),
        leak_label="↯ Dynamo tool call leaks",
        leak_text=str(dyn_leak) if dyn_leak else None,
        extra_sections=_dynamo_note_sections(case),
        chart=chart,
        refs=[("Ref", case.get("ref")), ("Spec ref", case.get("spec_ref"))],
    )


def _render_chunk_deltas(deltas: list, normal_text: str) -> str:
    """One impl's emit at one chunk, as literal text. `—` when nothing.

    Name deltas show as `name='<n>' (id)`; a chunk's argument fragments are joined
    into one quoted string `args='<assembled>'`, with consecutive fragments in
    alternating colors so a merged multi-delta chunk still shows its per-delta
    boundaries (replaces the old `×N` count). Strings are quoted so whitespace and
    punctuation like `":"` read unambiguously."""
    parts: list[str] = []  # each entry is already-escaped HTML
    arg_frags: list[str] = []
    for d in deltas or []:
        if not isinstance(d, dict):
            continue
        if d.get("name") is not None:
            tag = "name=" + html_lib.escape(repr(d["name"]))
            if d.get("id"):
                tag += " (id)"
            parts.append(tag)
        if d.get("arguments") is not None:
            arg_frags.append(d["arguments"])
    frags = [f for f in arg_frags if f]
    if frags:
        spans = "".join(
            f'<span class="frag{i % 4}">{html_lib.escape(f)}</span>'
            for i, f in enumerate(frags)
        )
        parts.append(f"args='{spans}'")
    if normal_text:
        parts.append("normal_text=" + html_lib.escape(repr(normal_text)))
    return "   ".join(parts) if parts else "—"


def _version_candidate_chart_html(case: dict, ver_status: dict) -> tuple[str, str] | None:
    """Per-chunk grid whose columns are the per-version CANDIDATES from `__ver_status`
    (one per `(impl, version)`, key `{impl}-{slug}` matching the compare-bar). Each
    column shows that version's per-chunk deltas (captured in `_stream_version_status_map`)
    and, in the final `assembled` row, its assembled output. The JS shows only the
    Reference + checked compare-with columns and marks the Reference. First column is the
    shared input (delta_text)."""
    input_chunks = [c for c in (case.get("chunks") or []) if isinstance(c, dict)]
    if not input_chunks:
        return None
    family = case.get("__family")
    chunk_html = colorize_stream_deltas(input_chunks, family)
    candidates = []  # (key, label, info)
    for impl in ("dynamo_rust", "vllm_rust", "vllm_python", "sglang_python"):
        # Engine columns in canonical order; within an engine, LATEST version first.
        entries = sorted(
            (ver_status.get(impl) or {}).items(),
            key=lambda kv: toolcalling_table._version_sort_key(str(kv[1].get("version") or "0")),
            reverse=True,
        )
        for slug, info in entries:
            candidates.append(
                (f"{impl}-{slug}", _full_label(impl, info.get("version"), "stream"), info)
            )
    if not candidates:
        return None
    def _col_header(key: str, label: str, info: dict) -> str:
        # The capture is emission-packed (fewer rows than input chunks): row
        # positions are NOT consumer-visible timing, so per-chunk cells stay
        # empty and the output shows only in the assembled row.
        note = (
            common.timing_note("bursts at end of call; per-chunk timing not recorded")
            if not info.get("aligned", True)
            else ""
        )
        return common.cand_th(key, html_lib.escape(label) + note)

    header = "".join(_col_header(key, label, info) for key, label, info in candidates)
    rows = []
    for i, chunk in enumerate(input_chunks):
        inp = chunk_html[i]
        if chunk.get("finish_reason"):
            inp += (
                ' <span class="fr">finish='
                + html_lib.escape(str(chunk["finish_reason"]))
                + "</span>"
            )
        cells = ""
        for key, _label, info in candidates:
            chs = info.get("chunks")
            body = ""
            if not info.get("aligned", True):
                body = "—"  # timing not recorded; output appears in `assembled` only
            elif isinstance(chs, list) and i < len(chs):
                body = _render_chunk_deltas(
                    chs[i].get("deltas") or [], chs[i].get("normal_text") or ""
                )
            cells += common.cand_td(key, body)
        rows.append(f'<tr><td class="cin">{inp}</td>{cells}</tr>')
    # `_cand_section_body` (not the bare block formatter) so each candidate's
    # `explanation:` note rides in its column — the chart REPLACES the per-candidate
    # list sections, so nothing the list carried may be lost.
    final_cells = "".join(
        common.cand_td(
            key, _cand_section_body(info.get("block"), family).replace(chr(10), "<br>")
        )
        for key, _label, info in candidates
    )
    rows.append(f'<tr class="ttip-final"><td class="cin">assembled</td>{final_cells}</tr>')
    table = common.candidate_chart_table(header, rows)
    return ("Per-chunk emit (recorded from parser = expected)", table)


def _merged_candidate_chart_html(case: dict, cmp_items: list) -> tuple[str, str] | None:
    """Candidate chart for the merged batch tab: same left-to-right layout as the
    stream chart, but batch input is a single text (one "chunk"), so there is one
    `output` row — each candidate's full result side by side. Columns are the
    `__cmp` candidates (`data-cand` keys match the compare bar), toggled/REF-marked
    by the same JS as the stream chart."""
    model_text = case.get("model_text")
    if not isinstance(model_text, str) or not model_text or not cmp_items:
        return None
    family = case.get("__family")
    header = "".join(
        common.cand_th(item["key"], html_lib.escape(item["label"])) for item in cmp_items
    )
    cells = "".join(
        common.cand_td(
            item["key"],
            _cand_section_body(item.get("block"), family).replace(chr(10), "<br>"),
        )
        for item in cmp_items
    )
    input_html = f"input_text='{colorize_markup(model_text, family)}'"
    final_row = f'<tr class="ttip-final"><td class="cin">{input_html}</td>{cells}</tr>'
    table = common.candidate_chart_table(header, [final_row])
    return ("Output (recorded from parser = expected)", table)


def _per_chunk_chart_html(case: dict, output_kind: str = "stream") -> tuple[str, str] | None:
    """Per-chunk breakdown as a compact table: one row per chunk, first column the
    input delta_text, then one column per available impl (Dynamo Rust, vLLM Rust,
    vLLM Python, SGLang)
    showing what it emitted at that chunk. Returns (label, table_html), or None for
    non per-chunk stream cases. No inter-tag whitespace (keeps the markup tight)."""
    # Stream tab: columns are the per-version CANDIDATES (from __ver_status), so the
    # popup shows the Reference + each checked compare-with (e.g. Dynamo v1 vs v2).
    ver_status = case.get("__ver_status")
    if ver_status and case.get("chunks"):
        cand = _version_candidate_chart_html(case, ver_status)
        if cand is not None:
            return cand
    # Merged batch tab: same chart, single-output row (batch input is "one chunk").
    cmp_items = case.get("__cmp")
    if cmp_items and not case.get("chunks"):
        cand = _merged_candidate_chart_html(case, cmp_items)
        if cand is not None:
            return cand
    chunks = case.get("chunks")
    if not (isinstance(chunks, list) and chunks):
        return None
    if not any(isinstance(c, dict) and "expected" in c for c in chunks):
        return None

    family = case.get("__family")
    unavailable = _normalize_impl_mapping(case.get("unavailable") or {})
    for impl in IMPL_KEYS:
        block = _impl_get(case.get("expected") or {}, impl)
        if isinstance(block, dict) and "unavailable" in block:
            unavailable.setdefault(impl, block["unavailable"])
    impls = [i for i in IMPL_KEYS if i not in unavailable]

    chunk_html = colorize_stream_deltas(chunks, family)
    # Baseline column: the Dynamo BATCH result (no chunks — just the result),
    # placed immediately left of the stream columns as a fixed reference. The
    # streaming parser's job is to reconstruct this batch parse, so it sits beside
    # the per-chunk emit for a direct side-by-side. One cell spans every row.
    dyn_batch = _impl_get(case.get("batch_expected") or {}, BASELINE_IMPL)
    show_baseline = isinstance(dyn_batch, dict) and (
        "calls" in dyn_batch or "normal_text" in dyn_batch
    )
    n_body_rows = sum(1 for c in chunks if isinstance(c, dict)) + 1  # + assembled
    baseline_td = ""
    base_header = ""
    if show_baseline:
        base_header = f'<th class="cbase-h">baseline<br>{html_lib.escape(_IMPL_DISPLAY[BASELINE_IMPL])} batch parser</th>'
        baseline_body = _format_output_block_html(dyn_batch, family).replace(
            chr(10), "<br>"
        )
        baseline_td = (
            f'<td class="cbase" rowspan="{n_body_rows}">{baseline_body}</td>'
        )
    mode_marker = _STREAM_MODE_MARKER if output_kind == "stream" else _BATCH_MODE_MARKER
    parse_mode = "stream" if mode_marker == _STREAM_MODE_MARKER else "batch"

    # Full parser names (no cryptic D_RS/V_RS/… tags); the Dynamo column is the
    # Reference the peer columns are compared against, so flag it `← REF`.
    def _col_h(impl: str) -> str:
        name = html_lib.escape(f"{_IMPL_DISPLAY[impl]} {parse_mode} parser")
        if impl == BASELINE_IMPL:
            inner = (
                '<span class="ttip-ref-star">★</span> '
                f'{name} <span class="ttip-ref">← REF</span>'
            )
        else:
            inner = name
        return common.cand_th(impl, inner, attr="data-col-impl")

    header = base_header + "".join(_col_h(i) for i in impls)
    rows = []
    for i, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            continue
        inp = chunk_html[i]
        # Token-id family: show the raw delta_token_ids next to the text. Empty
        # lists are meaningful — a sub-token text fragment carries no whole token
        # yet, so the parser emits nothing on that chunk.
        tid = chunk.get("delta_token_ids")
        if family == "harmony" and isinstance(tid, list):
            inp += ' <span class="parser-base">' + html_lib.escape(str(tid)) + "</span>"
        if chunk.get("finish_reason"):
            inp += (
                ' <span class="fr">finish='
                + html_lib.escape(str(chunk["finish_reason"]))
                + "</span>"
            )
        exp = _normalize_impl_mapping(chunk.get("expected") or {})
        nt = _normalize_impl_mapping(chunk.get("normal_text") or {})
        cells = "".join(
            common.cand_td(
                i2,
                _render_chunk_deltas(
                    _impl_get(exp, i2, []) or [], _impl_get(nt, i2, "") or ""
                ),
                attr="data-col-impl",
            )
            for i2 in impls
        )
        # The baseline cell (rowspan) is emitted once, on the first body row.
        rows.append(f'<tr><td class="cin">{inp}</td>{baseline_td}{cells}</tr>')
        baseline_td = ""
    # Final row: each impl's assembled result (calls + normal_text), derived by
    # concatenating its per-chunk deltas. Compare each column here against the
    # baseline column on the left (assembled X_s vs Dynamo batch).
    derived = _expected(case)
    final_cells = "".join(
        common.cand_td(
            i2,
            _format_output_block_html(_impl_get(derived, i2), family).replace(
                chr(10), "<br>"
            ),
            attr="data-col-impl",
        )
        for i2 in impls
    )
    rows.append(
        f'<tr class="ttip-final"><td class="cin">assembled</td>{baseline_td}{final_cells}</tr>'
    )
    table = common.candidate_chart_table(header, rows)
    if unavailable:
        note = "; ".join(
            f"{_IMPL_DISPLAY[i]}: {_short_unavailable(unavailable[i])}"
            for i in IMPL_KEYS
            if i in unavailable
        )
        table += f'<div class="ttip-unavail">{html_lib.escape(note)}</div>'
    return ("Per-chunk emit (recorded from parser = expected)", table)


def _tooltip_for(case: dict, dyn: dict, impl_keys: tuple[str, ...] = IMPL_KEYS) -> str:
    """Build the hover-tooltip text for a divergent cell.

    Each non-matching, non-unavailable peer contributes one line:
      vllm_python: <reason>                        # `explanation:` field present
      vllm_python: UNKNOWN — divergent ...         # divergent, no reason
      vllm_python: expected error matching '...'   # `error:` field present
    """
    parts: list[str] = []
    n_dyn = {
        "calls": dyn.get("calls") or [],
        "normal_text": dyn.get("normal_text") or "",
    }
    expected = _expected(case)
    for impl in (peer for peer in impl_keys if peer != BASELINE_IMPL):
        block = _impl_get(expected, impl)
        if not isinstance(block, dict) or block is dyn:
            continue
        if "unavailable" in block:
            continue
        name = _IMPL_DISPLAY.get(impl, impl)
        if "error" in block:
            parts.append(f"{name}: expected error matching {block['error']!r}")
            continue
        # Don't rely on PyYAML preserving anchor identity (the `block is dyn`
        # check above is the fast path; value equality is the safety net).
        n_block = {
            "calls": block.get("calls") or [],
            "normal_text": block.get("normal_text") or "",
        }
        if n_block == n_dyn:
            continue
        note = _explanation(block)
        if note:
            parts.append(f"{name}: {note}")
        elif "calls" in block or "normal_text" in block:
            parts.append(f"{name}: (research-needed — no `explanation:` field yet)")
    return "\n".join(parts)


def _build_na_tooltip_html(case: dict) -> str:
    """Tooltip for an n/a stub case (only `explanation:` in YAML, no `expected:`
    block). Renders case id + description + the note. Used when the cell
    is n/a because the scenario doesn't apply to the family's parser syntax."""
    case_id = case.get("__case_id", "")
    desc = case.get("description") or ""
    head = f"{case_id} — {desc}" if (case_id and desc) else (case_id or desc)
    reason = _explanation(case) or "n/a (no explanation given)"
    return _build_conformance_tooltip_html(
        head=head,
        extra_sections=[("Why not applicable", linkify_text_html(str(reason)))],
        refs=[("Ref", case.get("ref")), ("Spec ref", case.get("spec_ref"))],
    )


def _build_missing_tooltip_html(mode: str, family: str, sub: str) -> str:
    """Tooltip for an absent fixture entry.

    This is intentionally distinct from an explicit n/a stub. Missing means
    the table has no fixture data for this family/case; explicit n/a means a
    fixture author recorded why the case does not apply.
    """
    case_id = f"TOOLCALLING.{mode}.{sub}"
    return _build_conformance_tooltip_html(
        head=f"{case_id} — {family}",
        extra_sections=[
            (
                "Missing fixture",
                html_lib.escape(
                    "No fixture entry exists for this family/case. If the case "
                    "is intentionally not applicable, add an explicit n/a stub "
                    "with description: and explanation: so the table can explain it."
                ),
            )
        ],
    )


def render_cell_html(
    case: dict | None,
    mode: str,
    family: str,
    sub: str,
    output_kind: str = "batch",
    comparison: str = "cross_engine",
    marker_context: str | None = None,
) -> str:
    # `comparison` selects how the cell's status/markers are derived — the HTML
    # shape, CSS, and tooltip assembler are shared either way:
    #   "cross_engine"    (batch / stream tabs): per-engine leak status + cross-
    #                     engine conformance (`=`/D_rb/V_pb/S_rb vs the other engines).
    #   "stream_vs_batch" (batch-on-stream tab): each engine's stream parse vs its
    #                     OWN batch parse; diverge -> problem (red) + engine letter.
    sob = comparison == "stream_vs_batch"
    band_cls = _subcase_band_class(mode, sub)
    col_group = html_lib.escape(_subcase_group_key(mode, sub))
    if sob:
        text = _sob_cell_text(case, marker_context)
        status_attrs = _sob_status_attrs(case)
        marker_attrs = _sob_marker_attrs(case, marker_context)
        marker_spans = _sob_marker_spans(case, marker_context)
    else:
        impl_keys = _impl_keys_for_output_kind(output_kind)
        marker_mode = _STREAM_MODE_MARKER if output_kind == "stream" else _BATCH_MODE_MARKER
        text = cell_for(case, impl_keys, marker_mode)
        status_attrs = _overview_status_attrs(case, impl_keys)
        marker_attrs = _parser_marker_attrs(case, impl_keys, marker_mode)
        marker_spans = _parser_marker_spans(case, impl_keys, marker_mode)
    display_text = _marker_html(text)
    cls = parity_cell_class(text)
    # Compare-any-combination model (batch/TC v1 tab): embed the per-candidate
    # signature payload + a JS-filled marker span. JS colors the cell and fills the
    # count from the Base/Compare selection; falls back to the parser-radio view when
    # inactive (e.g. before JS runs, or on other tabs).
    # Compare model on every toolcalling tab: versioned candidates on the batch (v1)
    # tab; per-impl candidates on the stream / stream-on-batch tabs.
    # Versioned candidates (impl×version) when the case carries a __ver_status map
    # (batch tab, and the streamv2 tab once its per-version overlays are wired);
    # otherwise one candidate per impl (stream-on-batch, or a missing map).
    if case is None:
        cmp_json = ""
    elif isinstance(case, dict) and case.get("__cmp"):
        # Merged tab: both parser flavors (batch + stream on batch) in one payload.
        cmp_json = _cmp_json_from_blocks(
            {item["key"]: item["block"] for item in case["__cmp"]}
        )
    elif isinstance(case, dict) and case.get("__ver_status"):
        cmp_json = _candidate_cmp_json(case)
    else:
        cmp_json = _impl_cmp_json(case, STREAM_IMPL_KEYS)
    cmp_attr = f' data-cmp="{cmp_json}"' if cmp_json else ""
    cmp_span = '<span class="cmp-marker"><span class="marker-text"></span></span>' if cmp_json else ""
    marker_spans = cmp_span + marker_spans
    # data-family lets the compare JS tell "not implemented" (the selected Reference
    # parser doesn't support this family) apart from the case-level "not applicable".
    fam_attr = f' data-family="{html_lib.escape(str(family or ""))}"'
    td_open = (
        f'<td class="cell {cls} {band_cls}" data-col-hide-group="{col_group}"{cmp_attr}{fam_attr} '
        f"{status_attrs} {marker_attrs}>"
    )
    if case is None:
        ttip = _build_missing_tooltip_html(mode, family, sub)
        return f"{td_open}{marker_spans}{display_text}{ttip}</td>"

    dyn = _impl_get(case.get("expected") or {}, BASELINE_IMPL)
    if not isinstance(dyn, dict):
        # n/a stub: case has only `explanation:` (no `expected:` block).
        fp = case.get("__fixture_path", "")
        ttip = _build_na_tooltip_html(case)
        if not fp:
            return f"{td_open}{marker_spans}{display_text}{ttip}</td>"
        href = html_lib.escape(common.fixture_href(fp))
        return f'{td_open}{marker_spans}<a href="{href}">{display_text}</a>{ttip}</td>'

    fp = case.get("__fixture_path", "")
    # Case id + description live in the rich CSS tooltip head — don't also
    # set `title=` on the link, or browsers stack a native tooltip on top.
    ttip = _build_sob_tooltip(case, marker_context) if sob else _build_tooltip_html(case, dyn, output_kind)
    if not fp:
        return f"{td_open}{marker_spans}{display_text}{ttip}</td>"
    href = html_lib.escape(common.fixture_href(fp))
    return f'{td_open}{marker_spans}<a href="{href}">{display_text}</a>{ttip}</td>'


def _parser_inheritance_tooltip_html(
    family: str,
    info: dict,
    ctor_ref: tuple[str, int] | None,
    no_vllm: set[str] | None = None,
    no_sglang: set[str] | None = None,
) -> str:
    """Rich `.ttip` tooltip for the tool calling parser column.

    Keep this field-list shape aligned with the reasoning parser column tooltip
    so both tables explain "effective parser/backend -> row family" the same
    way. `ctor_ref` is unused here (was for older field-based layout) — kept
    for API stability with `_parser_cell_html`.
    """
    del ctor_ref

    variant = info["variant"] or "?"
    sub_variant = info["sub_variant"]
    backend_file = info["backend_file"]
    factory = info["factory"]
    alias_of = info.get("alias_of")  # set when this family is an alias-only entry

    head_parts = [f"ParserConfig::{variant}"]
    if sub_variant:
        head_parts[-1] = f"ParserConfig::{variant}::{sub_variant}"
    bf_href = html_lib.escape(f"{common.LINKS['toolcalling_src']}{backend_file}")
    bf_link = f'<a href="{bf_href}">{html_lib.escape(backend_file)}</a>'

    anchor = alias_of or family
    shared_family = sorted([anchor] + info["shared_with"])
    effective_backend = _shared_backend_short(info) or family

    implementation = f"{html_lib.escape(head_parts[0])} -> {bf_link}"
    if factory:
        factory_name = factory.split("(", 1)[0]
        implementation += html_lib.escape(f" (factory: {factory_name})")

    tooltip_lines = [
        "Tool calling parser family from fixture YAML.",
        f"Tool calling parser row: {html_lib.escape(family)}",
        f"Effective parser/backend: {html_lib.escape(effective_backend)}",
        f"Dynamo implementation: {implementation}",
    ]
    if info["shared_with"]:
        tooltip_lines.append(
            "Shared implementation family: " + html_lib.escape(", ".join(shared_family))
        )
    if alias_of:
        tooltip_lines.append(f"Alias of: {html_lib.escape(alias_of)}")
    if info["aliases"]:
        tooltip_lines.append(
            "Registered aliases: " + html_lib.escape(", ".join(info["aliases"]))
        )

    peer_notes: list[str] = []
    if no_vllm and family in no_vllm:
        peer_notes.append("no vLLM Python peer parser")
    if no_sglang and family in no_sglang:
        peer_notes.append("no SGLang peer parser")
    if peer_notes:
        tooltip_lines.append("Peer availability: " + ", ".join(peer_notes))

    if info["filed_under_xml_misleading"]:
        tooltip_lines.append(
            "Note: filed under xml/ but does not use the shared xml::parser; "
            f"it has its own ParserConfig::{html_lib.escape(variant)} variant."
        )
    tooltip_lines.extend(_tool_parser_tree_lines(family, info, effective_backend))

    if effective_backend == family:
        head_text = f"`{family}`"
    else:
        head_text = f"`{effective_backend}` (row: `{family}`)"
    return (
        '<div class="ttip">'
        f'<div class="ttip-head">{html_lib.escape(head_text)}</div>'
        f'<pre class="ttip-pre">{"".join(line + chr(10) for line in tooltip_lines).rstrip()}</pre>'
        "</div>"
    )


_SHARED_BACKEND_SHORT = {
    ("Json", "Basic"): "base_json",
    ("Xml", None): "xml",
    ("Dsml", None): "dsml",
}


def _tool_parser_tree_lines(
    family: str,
    info: dict,
    effective_backend: str,
) -> list[str]:
    alias_of = info.get("alias_of")
    anchor = alias_of or family
    aliases = info["aliases"]
    if not info["shared_with"] and not aliases and effective_backend == family:
        return []

    fam_list = sorted([anchor] + info["shared_with"])
    lines = ["", "Shared implementation tree:"]
    root_label = html_lib.escape(effective_backend)
    if effective_backend == family:
        root_label = f"<strong>{root_label}</strong>"
    lines.append(f"{root_label} (effective parser/backend)")

    for i, fam in enumerate(fam_list):
        is_last_fam = i == len(fam_list) - 1
        branch = "└── " if is_last_fam else "├── "
        fam_label = html_lib.escape(fam)
        if fam == family and not alias_of:
            fam_label = f"<strong>{fam_label}</strong>"
        lines.append(f"{branch}{fam_label}")

        if fam == anchor and aliases:
            cont = "    " if is_last_fam else "│   "
            for j, alias in enumerate(aliases):
                alast = j == len(aliases) - 1
                ab = "└── " if alast else "├── "
                alias_label = html_lib.escape(alias)
                if alias_of and alias == family:
                    alias_label = f"<strong>{alias_label}</strong>"
                lines.append(f"{cont}{ab}{alias_label} (alias)")

    return lines


def _shared_backend_short(info: dict | None) -> str | None:
    if info and info["shared_with"]:
        return _SHARED_BACKEND_SHORT.get(info["key"])
    return None


def _parser_label_markdown(
    family: str,
    no_vllm: set[str],
    no_sglang: set[str],
    inheritance: dict[str, dict],
) -> str:
    suff = family_suffix(family, no_vllm, no_sglang)
    short = _shared_backend_short(inheritance.get(family))
    if short:
        return f"{short} -> {family}{suff}"
    return f"{family}{suff}"


# Dynamo parser v2 stream parsers with a standard `push`/`finish` text path:
# family -> (backend label, source file under parsers/v2/src/tool_calling/, format marker).
# Families with bespoke paths (harmony token-id/text, deepseek_v4 DSML note) keep their
# dedicated branches below; new standard families belong here, not in new if-branches.
_V2_STREAM_PARSER_CELLS: dict[str, tuple[str, str, str]] = {
    "qwen3_coder": ("Qwen3CoderToolStreamParser text path", "qwen3_coder.rs", "Qwen XML"),
}


def _parser_cell_html(
    family: str,
    refs: dict[str, tuple[str, int]],
    no_vllm: set[str],
    no_sglang: set[str],
    inheritance: dict[str, dict],
    stream_context: str | None = None,
) -> str:
    suff = family_suffix(family, no_vllm, no_sglang)
    row_label = html_lib.escape(family)
    if suff:
        row_label += f'<span class="parser-suffix">{html_lib.escape(suff)}</span>'
    if family == "harmony" and stream_context == "streamv2":
        return _v2_parser_cell_html(
            row_label,
            family,
            "HarmonyToolStreamParser token-id path",
            "harmony.rs",
            "parse_tool_call_streaming_incremental",
            "v2 stream fixtures",
            "TC stream token-id row. It consumes `delta_token_ids` from v2 stream fixtures directly.",
        )
    if family == "harmony" and stream_context == "batch_on_stream":
        return _v2_parser_cell_html(
            row_label,
            family,
            "HarmonyToolStreamParser text path",
            "harmony.rs",
            "parse_tool_call_streaming_text",
            "v1 batch fixtures",
            "TC batch-on-stream row. It feeds each v1 batch fixture's full text through the v2 streaming parser.",
        )
    if family == "harmony_text":
        return _v2_parser_cell_html(
            row_label,
            family,
            "HarmonyToolStreamParser text path",
            "harmony.rs",
            "parse_tool_call_streaming_text",
            "v2 stream fixtures",
            "Synthetic v2 row for gpt-oss text streaming. The text path re-tokenizes a held suffix, then feeds the same token-incremental Harmony stream parser used by the token-id row.",
        )
    if family == "deepseek_v4" and stream_context in ("streamv2", "batch_on_stream"):
        fixtures = "v2 stream fixtures" if stream_context == "streamv2" else "v1 batch fixtures"
        note = (
            "TC stream row. It consumes DSML text chunks and emits compact complete-invoke deltas."
            if stream_context == "streamv2"
            else "TC batch-on-stream row. It feeds each v1 batch fixture's full text through the v2 DSML streaming parser."
        )
        return _v2_parser_cell_html(
            row_label,
            family,
            "DeepSeekV4ToolStreamParser text path",
            "dsml.rs",
            "push",
            fixtures,
            note,
        )
    v2_cell = _V2_STREAM_PARSER_CELLS.get(family)
    if v2_cell and stream_context in ("streamv2", "batch_on_stream"):
        backend, source_file, marker = v2_cell
        fixtures = "v2 stream fixtures" if stream_context == "streamv2" else "v1 batch fixtures"
        note = (
            f"TC stream row. It consumes {marker} text chunks and emits per-chunk tool-call deltas."
            if stream_context == "streamv2"
            else f"TC batch-on-stream row. It feeds each v1 batch fixture's full text through the v2 {marker} streaming parser."
        )
        return _v2_parser_cell_html(
            row_label, family, backend, source_file, "push", fixtures, note
        )
    if stream_context in ("streamv2", "batch_on_stream"):
        # No Dynamo parser v2 stream parser for this family yet. Inventory-only
        # row; don't link the v1 batch parser.
        return _v2_missing_stream_parser_cell_html(family)
    ref = refs.get(family)
    info = inheritance.get(family)
    ttip = (
        _parser_inheritance_tooltip_html(family, info, ref, no_vllm, no_sglang)
        if info
        else ""
    )

    # Shared-backend rows should read as implementation -> fixture family,
    # e.g. `xml -> minimax_m2` and `xml -> qwen3_coder`. Standalone parsers
    # keep the public family name as the primary label.
    short = _shared_backend_short(info)
    if short:
        label = html_lib.escape(short)
        base_suffix = f'<span class="parser-base">→ {row_label}</span>'
    else:
        label = row_label
        base_suffix = ""

    # Family-name link points to the **actual parser code** (backend_file from
    # the inheritance map), not to the config-ctor location in config.rs. The
    # ctor location is still referenced in the inheritance tooltip body when
    # useful (factory calls). For families with no inheritance info, fall back
    # to the refs entry (config.rs or parsers.rs).
    if info and info["backend_file"] != "unknown":
        href = f"{common.LINKS['toolcalling_src']}{info['backend_file']}"
    elif ref is not None:
        href = f"{common.LINKS['toolcalling_src']}{ref[0]}"
    else:
        return (
            f'<td class="parser" data-col-hide-group="parser">'
            f"{label}{base_suffix}{ttip}</td>"
        )
    return (
        f'<td class="parser" data-col-hide-group="parser">'
        f'<a href="{href}">{label}</a>{base_suffix}{ttip}</td>'
    )


def _v2_parser_cell_html(
    row_label: str,
    family: str,
    backend: str,
    source_file: str,
    entrypoint: str,
    fixtures: str,
    note: str,
) -> str:
    tooltip = (
        '<div class="ttip">'
        f'<div class="ttip-head">`{html_lib.escape(family)}` (v2 stream)</div>'
        '<pre class="ttip-pre">'
        f"Fixtures: {html_lib.escape(fixtures)}.\n"
        f"Tool calling parser row: {html_lib.escape(family)}\n"
        f"Effective parser/backend: {html_lib.escape(backend)}\n"
        f"Dynamo parser v2 implementation: parsers/v2/src/tool_calling/{html_lib.escape(source_file)} -> "
        f'<a href="{common.LINKS["streaming_src"]}{html_lib.escape(source_file)}">{html_lib.escape(entrypoint)}</a>\n'
        f"Note: {html_lib.escape(note)}"
        "</pre></div>"
    )
    return (
        f'<td class="parser" data-col-hide-group="parser">'
        f'<a href="{common.LINKS["streaming_src"]}{html_lib.escape(source_file)}">{row_label}</a>{tooltip}</td>'
    )


def _v2_missing_stream_parser_cell_html(family: str) -> str:
    label = html_lib.escape(family)
    tooltip = (
        '<div class="ttip">'
        f'<div class="ttip-head">`{label}` (v2 stream)</div>'
        '<pre class="ttip-pre">'
        "Dynamo parser v2 is not implemented for this family yet.\n"
        "This row is inventory only; no v1 parser code runs on this tab."
        "</pre></div>"
    )
    return f'<td class="parser" data-col-hide-group="parser">{label}{tooltip}</td>'


def render_row_html(
    model: str,
    family: str,
    mode: str,
    cases: dict,
    sub_cases: list[str],
    refs: dict[str, tuple[str, int]],
    no_vllm: set[str],
    no_sglang: set[str],
    inheritance: dict[str, dict],
    parser_stream_context: str | None = None,
    comparison: str = "cross_engine",
) -> str:
    # `mode` drives the sub-case taxonomy (column groups / bands / runs). The
    # parser column has its own variant axis (`parser_stream_context`): it is the
    # same as `mode` for the normal batch/stream panels, but batch-on-stream reuses
    # the "batch" taxonomy while showing the v2 stream parser column.
    if parser_stream_context is None:
        parser_stream_context = mode
    # Tooltips label each engine's output block as batch or stream. Both the stream
    # tab and the batch-on-stream tab carry STREAMING parser output; only the plain
    # batch tab carries batch output.
    output_kind = "batch" if parser_stream_context == "batch" else "stream"
    cell_text = (
        (lambda case: _sob_cell_text(case, parser_stream_context))
        if comparison == "stream_vs_batch"
        else cell_for
    )
    cells = [
        f'<tr><td class="model" data-col-hide-group="model">{_model_label_html(model)}</td>',
        _column_placeholder_html("model"),
        _parser_cell_html(
            family,
            refs,
            no_vllm,
            no_sglang,
            inheritance,
            stream_context=parser_stream_context,
        ),
        _column_placeholder_html("parser"),
    ]
    # A family with no parser implemented and no captured peer output renders as
    # an inventory-only row. If peer parser data exists, still render cells so the
    # peer Parser-radio views show the captured stream behavior while the local
    # Dynamo parser v2 remains TODO.
    all_todo = sub_cases and all(
        cell_text(cases.get((family, sub))) == "…" for sub in sub_cases
    )
    peer_output_exists = any(
        _has_peer_output(cases.get((family, sub))) for sub in sub_cases
    )
    for run in _subcase_runs(mode, sub_cases):
        if all_todo and not peer_output_exists:
            cells.extend(_blank_cell_html(mode, sub) for sub in run)
        else:
            cells.extend(
                render_cell_html(
                    cases.get((family, sub)),
                    mode,
                    family,
                    sub,
                    output_kind,
                    comparison,
                    parser_stream_context,
                )
                for sub in run
            )
        cells.append(_column_placeholder_html(_subcase_group_key(mode, run[0])))
    cells.append("</tr>")
    return "".join(cells)


def _has_peer_output(case: dict | None) -> bool:
    if not case:
        return False
    expected = _expected(case)
    for impl in PEER_IMPL_KEYS:
        block = _impl_get(expected, impl)
        if isinstance(block, dict) and "unavailable" not in block:
            return True
    return False


def _blank_cell_html(mode: str, sub: str) -> str:
    """An empty data cell for a not-yet-implemented (TODO) family. No data-status /
    data-marker attributes, so it stays blank under every Parser-radio selection."""
    band_cls = _subcase_band_class(mode, sub)
    col_group = html_lib.escape(_subcase_group_key(mode, sub))
    return f'<td class="cell todo {band_cls}" data-col-hide-group="{col_group}"></td>'


def _parse_subcase_descriptions(mode: str) -> dict[str, str]:
    """Parse `lib/parsers/TOOLCALLING_CASES.md` for per-case descriptions.

    The Quick-reference section has one-liner bullets for top-level cases
    (`TOOLCALLING.<mode>.1` …); the deeper per-case sections
    contain multi-line bullets for sub-cases (`2.a`, `4.c`, etc.). Both
    look like `- **`TOOLCALLING.<mode>.X`** <desc>`, where the bullet body may
    wrap across indented continuation lines. Returns
    `{"1": "...", "2.a": "...", ...}`.
    """
    # Streaming descriptions come from our own renumbered doc; batch/others from
    # the dynamo-synced TOOLCALLING_CASES.md.
    cases_md = TOOLCALLING_STREAMING_V2_CASES_MD if mode == "streamv2" else TOOLCALLING_CASES_MD
    if not cases_md.exists():
        return {}
    pat = re.compile(
        rf"\*\*`TOOLCALLING\.{re.escape(mode)}" rf"\.([0-9]+(?:\.[a-z])?)`\*\*\s+(.+)"
    )
    out: dict[str, str] = {}
    lines = cases_md.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        m = pat.search(lines[i])
        if not m:
            i += 1
            continue
        sub = m.group(1)
        body_parts = [m.group(2).strip()]
        # Join indented continuation lines until blank / next bullet / unindented.
        j = i + 1
        while j < len(lines):
            nxt = lines[j]
            if not nxt.strip():
                break
            if not nxt.startswith(" "):
                break
            if pat.search(nxt):
                break
            body_parts.append(nxt.strip())
            j += 1
        desc = " ".join(body_parts).rstrip(".")
        out.setdefault(sub, desc)
        i = j
    return out


def _subcase_header_html(mode: str, sub: str, descriptions: dict[str, str]) -> str:
    desc = descriptions.get(sub) or descriptions.get(sub.split(".")[0]) or ""
    href = (
        common.LINKS["toolcalling_streaming_cases"]
        if mode == "streamv2"
        else common.LINKS["toolcalling_cases"]
    )
    title = html_lib.escape(desc) if desc else ""
    band_cls = _subcase_band_class(mode, sub)
    col_group = html_lib.escape(_subcase_group_key(mode, sub))
    return (
        f'<th class="case-sub {band_cls}" data-col-hide-group="{col_group}">'
        f'<a href="{href}" title="{title}">{html_lib.escape(sub)}</a></th>'
    )


def _subcase_group_label(mode: str, sub: str) -> str:
    return _group_by_sub(mode).get(sub, "Other")


def _subcase_runs(mode: str, sub_cases: list[str]) -> list[list[str]]:
    runs: list[list[str]] = []
    start = 0
    while start < len(sub_cases):
        label = _subcase_group_label(mode, sub_cases[start])
        end = start + 1
        while (
            end < len(sub_cases) and _subcase_group_label(mode, sub_cases[end]) == label
        ):
            end += 1
        runs.append(sub_cases[start:end])
        start = end
    return runs


def _column_placeholder_html(key: str, tag: str = "td") -> str:
    key_attr = html_lib.escape(key)
    return (
        f'<{tag} class="col-placeholder col-hidden" '
        f'data-col-placeholder-group="{key_attr}"></{tag}>'
    )


def _column_control_header_html(
    key: str,
    label: str,
    *,
    default_visible: bool,
    css_class: str = "",
    colspan: int | None = None,
) -> str:
    key_attr = html_lib.escape(key)
    visible = "true" if default_visible else "false"
    classes = " ".join(part for part in ("column-control", css_class) if part)
    span_size = colspan if colspan is not None else 1
    if colspan is not None:
        span_attr = f'colspan="{colspan}" data-expanded-colspan="{colspan}"'
    else:
        span_attr = 'rowspan="2"'
    return (
        f'<th class="{html_lib.escape(classes)}" data-col-control-group="{key_attr}" '
        f"{span_attr}>"
        f'<button type="button" class="col-toggle" data-col-toggle="{key_attr}" '
        f'data-col-label="{html_lib.escape(label)}" data-col-span="{span_size}" '
        f'data-default-visible="{visible}" aria-pressed="{visible}" '
        f'aria-label="{"Collapse" if default_visible else "Expand"} '
        f'{html_lib.escape(label)} column">'
        '<span class="col-toggle-symbol" aria-hidden="true"></span>'
        f'<span class="col-toggle-label">{html_lib.escape(label)}</span>'
        "</button></th>"
    )


def _subcase_group_headers_html(mode: str, sub_cases: list[str]) -> str:
    """Build semantic group headers spanning the displayed sub-case columns."""
    spans: list[str] = [
        _column_control_header_html("model", "Model", default_visible=True),
        _column_control_header_html(
            "parser", "Tool calling family", default_visible=True
        ),
    ]
    for run in _subcase_runs(mode, sub_cases):
        label = _subcase_group_label(mode, run[0])
        band_cls = _subcase_band_class(mode, run[0])
        col_group = html_lib.escape(_subcase_group_key(mode, run[0]))
        spans.append(
            _column_control_header_html(
                col_group,
                label,
                default_visible=True,
                css_class=f"case-group {band_cls}",
                colspan=len(run),
            )
        )
    return "".join(spans)


def _subcase_headers_html(
    mode: str, sub_cases: list[str], descriptions: dict[str, str]
) -> str:
    headers: list[str] = []
    for run in _subcase_runs(mode, sub_cases):
        headers.extend(_subcase_header_html(mode, sub, descriptions) for sub in run)
        headers.append(
            _column_placeholder_html(_subcase_group_key(mode, run[0]), tag="th")
        )
    return "".join(headers)


def _glossary_groups(
    mode: str, descriptions: dict[str, str], sub_cases: list[str]
) -> list[dict[str, object]]:
    if not descriptions:
        return []
    return [
        {
            "label": _subcase_group_label(mode, run[0]),
            "rows": [
                (
                    sub,
                    descriptions.get(sub) or descriptions.get(sub.split(".")[0]) or "",
                )
                for sub in run
            ],
        }
        for run in _subcase_runs(mode, sub_cases)
    ]


def _peer_version_items(versions: dict[str, str]) -> list[tuple[str, str]]:
    normalized = _normalize_impl_mapping(versions)
    return [
        (_IMPL_DISPLAY[name], normalized[name])
        for name in ("vllm_rust", "vllm_python", "sglang_python")
        if name in normalized
    ]


# --- per-impl version snapshots for the TC v1 (batch) tab -----------------------
# Version dirs use legacy impl prefixes (dynamo/vllm/sglang); map to the canonical
# batch impl keys the cells + radios use. Discovery/slug/sort helpers are shared
# with the parity page via toolcalling_table.
_VERSION_LEGACY_TO_CANON = {
    "dynamo": "dynamo_rust",
    "vllm": "vllm_python",
    "sglang": "sglang_python",
}
_IMPL_VERSION_RADIO_LABEL = {
    "dynamo_rust": "Dynamo Rust",
    "vllm_python": "vLLM Python",
    "sglang_python": "SGLang Python",
}


def _batch_impl_versions() -> dict[str, list[str]]:
    """Legacy-impl -> versions (ascending) for impls present on the batch tab."""
    discovered = toolcalling_table._impl_versions()
    return {
        legacy: vers
        for legacy, vers in discovered.items()
        if _VERSION_LEGACY_TO_CANON.get(legacy) in BATCH_IMPL_KEYS
    }


def _batch_version_status_map() -> dict[tuple[str, str], dict[str, dict[str, str]]]:
    """{(family, sub): {canonical_impl: {version_slug: overview_status}}} for batch.

    Resolve each impl@version (others pinned) and re-run load_all_cases("batch") so
    keys match the rendered table (same normalization); classify with the same
    markers._overview_status used for the pinned cells."""
    impl_versions = _batch_impl_versions()
    if not impl_versions:
        return {}
    resolver = toolcalling_table._RESOLVE_SRC_DIR / "resolve_fixtures.py"
    src = toolcalling_table._SRC_FIXTURES
    if not resolver.exists() or not src.is_dir():
        return {}
    pinned = toolcalling_table._pinned_versions(impl_versions)
    saved_fixtures = fixtures.FIXTURES
    saved_captured = _CAPTURED_WITH_BY_MODE.get("batch")
    result: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    try:
        for legacy, versions in impl_versions.items():
            canon = _VERSION_LEGACY_TO_CANON[legacy]
            for version in versions:
                slug = toolcalling_table._version_slug(version)
                select = [
                    f"{other}-{version if other == legacy else pinned[other]}"
                    for other in impl_versions
                ]
                # Resolve under the staged fixtures parent so load_all_cases's
                # `fp.relative_to(script_dir)` stays valid (script_dir = the module
                # dir, above the fixtures tree).
                with tempfile.TemporaryDirectory(dir=str(saved_fixtures.parent)) as tmp:
                    subprocess.run(
                        [sys.executable, str(resolver),
                         "--fixtures-root", str(src),
                         "--out", tmp, "--select", *select],
                        check=True, capture_output=True,
                    )
                    fixtures.FIXTURES = Path(tmp)
                    cases, _labels = load_all_cases("batch")
                for key, case in cases.items():
                    block = _impl_get(case.get("expected") or {}, canon)
                    result.setdefault(key, {}).setdefault(canon, {})[slug] = {
                        "status": _overview_status(case, canon),
                        "block": block,
                        "version": version,
                        "marker": _parser_marker(case, canon),
                        "parity_marker": _parity_marker(
                            case, canon, BATCH_IMPL_KEYS, _BATCH_MODE_MARKER
                        ),
                    }
    finally:
        fixtures.FIXTURES = saved_fixtures
        if saved_captured is not None:
            _CAPTURED_WITH_BY_MODE["batch"] = saved_captured
    return result


def _impl_version_items() -> list[dict[str, object]]:
    """Version-radio model for the TC v1 tab template (canonical-keyed)."""
    impl_versions = _batch_impl_versions()
    pinned = toolcalling_table._pinned_versions(impl_versions)
    items: list[dict[str, object]] = []
    for legacy, versions in impl_versions.items():
        canon = _VERSION_LEGACY_TO_CANON[legacy]
        default = pinned.get(legacy)
        items.append({
            "impl": canon,
            "label": _IMPL_VERSION_RADIO_LABEL.get(canon, canon),
            "default_slug": toolcalling_table._version_slug(default) if default else "",
            "versions": [
                {
                    "version": v,
                    "slug": toolcalling_table._version_slug(v),
                    "default": v == default,
                }
                for v in versions
            ],
        })
    return items


# --- compare-any-combination model (TC v1 tab) ---------------------------------
# Every (parser, version) is a "candidate". A cell reports how many of the
# user-selected candidates differ from the chosen Base; the tooltip shows Base +
# each selected candidate's output. All of it is computed client-side from the
# compact per-cell `data-cmp` payload below, so any base/compare combination works.
_CANDIDATE_SHORT = {
    "dynamo_rust": "Dynamo",
    "vllm_rust": "vLLM Rust",
    "vllm_python": "vLLM",
    "sglang_python": "SGLang",
}

# Standardized candidate label: "<Engine> <Runtime> <version> (<mode>)", e.g.
# "Dynamo Rust 3.0.0 (batch)", "vLLM Python 0.24.0 (stream)". The runtime is part of
# the engine display so a chip and its tooltip section read identically, and one
# merged-tab cell distinguishes a batch parser from a stream parser on the same text
# purely by the trailing "(mode)". Dynamo's parsers are Rust crates (dynamo-parsers
# v1 3.0.0, dynamo-parsers-v2 0.1.11); the version disambiguates v1 vs v2.
_ENGINE_RUNTIME = {
    "dynamo_rust": "Dynamo Rust",
    "vllm_rust": "vLLM Rust",
    "vllm_python": "vLLM Python",
    "sglang_python": "SGLang Python",
}


def _dynamo_vtag(version: object) -> str | None:
    """v1 vs v2 tag for a Dynamo parser version: the dynamo-parsers-v2 crate (0.x) = v2,
    the dynamo-parsers crate (3.x) = v1. Returns None when the version is unknown so a
    version-less synthetic case just reads '(mode)'.

    Classify by major version rather than string-matching the live parsers_v2/Cargo.toml
    version. The two crates occupy disjoint major ranges (0.x vs 3.x), and a captured
    fixture dir (e.g. dynamo_rust-0.1.11) keeps its capture-time version even after the
    crate is bumped — matching the current Cargo version would silently mis-tag a
    not-yet-recaptured dir as v1 (and, worse, drop it into the v1 reference bucket)."""
    if not version:
        return None
    try:
        major = int(str(version).split(".")[0])
    except ValueError:
        return None
    return "v2" if major == 0 else "v1"


def _full_label(impl: str, version: object, mode: str) -> str:
    base = _ENGINE_RUNTIME.get(impl, _CANDIDATE_SHORT.get(impl, impl))
    # Dynamo carries a v1/v2 crate tag between the engine and runtime, e.g.
    # "Dynamo v1 Rust 3.0.0 (batch)" vs "Dynamo v2 Rust 0.1.11 (stream)". The v1 parser
    # run against stream data goes through the streaming jail (buffer then v1 batch
    # parse), so on the stream tab it reads "(jail+batch)". Peers have no crate
    # split and stay "<Engine> <Runtime> <version> (<mode>)".
    if impl == BASELINE_IMPL:
        vtag = _dynamo_vtag(version)
        if vtag:
            eng, _, rt = base.partition(" ")  # "Dynamo" / "Rust" -> "Dynamo <vtag> Rust"
            base = f"{eng} {vtag} {rt}".strip()
            if vtag == "v1" and mode == "stream":
                mode = "jail+batch"
    ver = f" {version}" if version else ""
    return f"{base}{ver} ({mode})"


def _candidate_label_html(label: str) -> str:
    """Escape a compare candidate label and color the trailing mode parenthetical:
    `batch` maroon, `stream` NVIDIA green (matches the tab-label word coding). The
    plain `label` stays around for tooltips; only the compare bar uses this HTML."""
    esc = html_lib.escape(label)
    m = re.search(r"\(([^)]*)\)\s*$", esc)
    if not m:
        return esc
    s, e = m.span(1)
    inner = m.group(1)
    inner = inner.replace("batch", '<span class="cand-batch">batch</span>')
    inner = inner.replace("stream", '<span class="cand-stream">stream</span>')
    return esc[:s] + inner + esc[e:]


def _dynamo_v2_version() -> str | None:
    """Version label for the Dynamo v2 stream parser, taken from the PUBLISHED fixture
    provenance (the v2-major `dynamo_rust-<ver>` dir, e.g. 0.1.11), NOT the live
    parsers/v2/Cargo.toml.

    Sourcing from the fixtures keeps every "Dynamo v2 Rust … (stream)" label on the page
    consistent (the stream-tab candidates already read the dir version) and matching the
    captured data. Reading the live crate makes the label drift ahead — the page would
    show 0.1.16 in one place and the real captured 0.1.11 in another the moment the crate
    is bumped before a re-capture/republish."""
    for v in _stream_impl_versions().get(BASELINE_IMPL, []):
        if _dynamo_vtag(v) == "v2":
            return v
    return None


def _v2_display_version(impl: str) -> str | None:
    """Display version for a v2-tab candidate: Dynamo -> the v2 crate version;
    peers -> the engine version they were captured against."""
    if impl == BASELINE_IMPL:
        return _dynamo_v2_version()
    return _clean_version((_CAPTURED_WITH_BY_MODE.get("streamv2") or {}).get(impl))


def _cand_label(impl: str, mode: str = "streamv2") -> str:
    """Engine+runtime+version candidate label without the trailing "(mode)", e.g.
    'vLLM Rust 0.23.0' / 'Dynamo Rust 0.1.11'. Callers append "(stream)"/"(batch)"."""
    base = _ENGINE_RUNTIME.get(impl, _CANDIDATE_SHORT.get(impl, impl))
    ver = _v2_display_version(impl)
    return f"{base} {ver}" if ver else base


def _clean_version(v: object) -> str | None:
    """Pull a display version from a captured_with value: 'v0.23.0 <sha>' -> '0.23.0',
    '0.5.12.post1' -> '0.5.12.post1', 'Dynamo parser v2' -> None (no numeric version)."""
    if not v:
        return None
    token = str(v).split()[0].lstrip("v")
    return token if re.match(r"\d", token) else None


def _impl_candidate_items(
    impl_keys: tuple[str, ...], versions: dict[str, str] | None = None
) -> list[dict[str, str]]:
    """Candidates for a non-versioned tab: one per impl key, labeled with the
    captured version when available (e.g. 'vLLM Rust 0.23.0'). First = Base (A),
    the rest default to Compare-with (B)."""
    versions = versions or {}
    out: list[dict[str, str]] = []
    for i, impl in enumerate(impl_keys):
        short = _CANDIDATE_SHORT.get(impl, impl)
        ver = _clean_version(versions.get(impl))
        out.append({
            "key": impl,
            "label": f"{short} {ver}" if ver else short,
            "default_bucket": "A" if i == 0 else "B",
        })
    return out


def _candidate_items() -> list[dict[str, str]]:
    """Ordered comparison candidates for the batch tab: Dynamo, then vLLM/SGLang —
    within each engine versions run LATEST-FIRST (0.24.0 before 0.23.0). Each:
    {key, impl, version, slug, label, short, default_bucket}.

    Default layout: A (reference) = the first candidate (Dynamo's latest); B (compare
    with) = the latest version of each peer impl; C (others) = the older versions."""
    impl_versions = _batch_impl_versions()
    latest = {lg: (vers[-1] if vers else None) for lg, vers in impl_versions.items()}
    out: list[dict[str, str]] = []
    first = True
    for legacy in ("dynamo", "vllm", "sglang"):
        canon = _VERSION_LEGACY_TO_CANON.get(legacy)
        for v in reversed(impl_versions.get(legacy, [])):
            slug = toolcalling_table._version_slug(v)
            if first:
                bucket = "A"
                first = False
            elif v == latest.get(legacy):
                bucket = "B"
            else:
                bucket = "C"
            out.append({
                "key": f"{canon}-{slug}",
                "impl": canon,
                "version": v,
                "slug": slug,
                "short": _ENGINE_RUNTIME.get(canon, canon),
                "label": _full_label(canon, v, "batch"),
                "default_bucket": bucket,
            })
    return out


# --- per-impl version snapshots for the TC v2 (stream) tab ----------------------
# The streamv2 corpus is versioned like batch, but with a different physical layout:
# The stream-v2 corpus is versioned like the batch corpus (no unversioned anchor):
# fixtures-stream-v2/inputs/ (shared per-chunk delta_text) + fixtures-stream-v2/
# <impl>-<version>/ (per-impl expected; lowest version = full anchor, higher =
# changed-only). resolve_stream_fixtures.py reconstructs a flat tree for any selected
# version set — the stream analogue of resolve_fixtures.py + the batch __ver_status map.
# Read from the HuggingFace download cache (fixtures aren't in the repo since DIS-2310);
# _common.sh exports CONFORMANCE_FIXTURES_ROOT. Without this the stream tab's versioned
# candidates come up empty and the Base/Compare parser selector doesn't render.
_STREAM_SRC = (
    toolcalling_table._fixtures_cache_root() / "toolcalling/fixtures-stream-v2"
)


def _stream_impl_versions() -> dict[str, list[str]]:
    """{stream_impl: versions ascending} discovered from the fixtures-stream-v2/
    <impl>-<version>/ dirs (no hardcoded anchor — the baseline is whichever version is
    lowest). Ordered dynamo_rust, vllm_rust, vllm_python, sglang_python (canonical
    stream column order)."""
    found: dict[str, list[str]] = {}
    if _STREAM_SRC.is_dir():
        for d in _STREAM_SRC.iterdir():
            if not d.is_dir() or d.name == "inputs" or "-" not in d.name:
                continue
            impl, ver = d.name.split("-", 1)
            found.setdefault(impl, []).append(ver)
    for impl in list(found):
        found[impl] = sorted(set(found[impl]), key=toolcalling_table._version_sort_key)
    order = ("dynamo_rust", "vllm_rust", "vllm_python", "sglang_python")
    return {i: found[i] for i in order if i in found}


def _stream_candidate_items() -> list[dict[str, str]]:
    """Versioned comparison candidates for the stream tab. Keyed <impl>-<slug> like
    the batch tab. Default layout: A (reference) = Dynamo v1 (jail+batch, 3.0.0) — the
    parser that has stream coverage on every family; B (compare) = Dynamo v2 + the
    latest of each peer; C (others) = older peer versions."""
    impl_versions = _stream_impl_versions()
    latest = {i: (vs[-1] if vs else None) for i, vs in impl_versions.items()}
    out: list[dict[str, str]] = []
    for impl in ("dynamo_rust", "vllm_rust", "vllm_python", "sglang_python"):
        # Within an engine, versions run LATEST-FIRST (0.24.0 before 0.23.0).
        for v in reversed(impl_versions.get(impl, [])):
            slug = toolcalling_table._version_slug(v)
            if impl == BASELINE_IMPL:
                # Dynamo v1 (jail+batch) is the default reference; v2 goes to compare.
                bucket = "A" if _dynamo_vtag(v) == "v1" else "B"
            elif v == latest.get(impl):
                bucket = "B"
            else:
                bucket = "C"
            out.append({
                "key": f"{impl}-{slug}",
                "label": _full_label(impl, v, "stream"),
                "default_bucket": bucket,
            })
    return out


def _stream_version_families(impl: str, version: str) -> set[str] | None:
    """Families the `<impl>-<version>` stream fixture dir actually holds — the
    authoritative coverage for that parser build. `None` if the dir is absent (don't
    gate). Used to mark the Dynamo v2 stream candidate `na` on families its parser
    doesn't implement, since the dir only contains the families it produced output
    for (dynamo_rust-0.1.11 = the v2-supported handful; dynamo_rust-3.0.0 = all)."""
    d = _STREAM_SRC / f"{impl}-{version}"
    if not d.is_dir():
        return None
    return {p.name for p in d.iterdir() if p.is_dir()}


def _parser_ni_map() -> dict:
    """Map candidate key -> {label, families} for parsers with LIMITED family coverage
    (only the Dynamo v2 parser today, which implements a handful of families). The
    compare JS uses it to render a per-family "not implemented" reason when such a
    parser is the selected Reference, instead of the case-level "not applicable"
    (which is about whether the test case fits the family, not whether the parser
    exists). Coverage is the authoritative `dynamo_rust-<v2>` fixture dir family list."""
    v2ver = _dynamo_v2_version()
    if not v2ver:
        return {}
    fams = sorted(_stream_version_families(BASELINE_IMPL, v2ver) or [])
    if not fams:
        return {}
    slug = toolcalling_table._version_slug(v2ver)
    entry = {"label": _full_label(BASELINE_IMPL, v2ver, "stream"), "families": fams}
    # The v2 candidate key differs by tab: "<impl>-s-<slug>" on the batch
    # (stream-on-batch) tab, bare "<impl>-<slug>" on the stream tab.
    return {f"{BASELINE_IMPL}-s-{slug}": entry, f"{BASELINE_IMPL}-{slug}": entry}


def _stream_version_status_map() -> dict[tuple[str, str], dict[str, dict[str, dict]]]:
    """{(family, sub): {impl: {slug: {block, version, status}}}} for the stream tab.

    Resolve each versioned peer @ each of its versions (others pinned) and re-run
    load_all_cases("streamv2") so keys match the rendered table (same assembly +
    split-parent normalization). Single-version impls (dynamo_rust, vllm_rust) are
    recorded once from the pinned resolve. `block` is the assembled per-impl
    {calls, normal_text} used for the per-cell `data-cmp` signature."""
    impl_versions = _stream_impl_versions()
    if not impl_versions:
        return {}
    resolver = toolcalling_table._RESOLVE_SRC_DIR / "resolve_stream_fixtures.py"
    if not resolver.exists() or not _STREAM_SRC.is_dir():
        return {}
    overlaid = {i: vs for i, vs in impl_versions.items() if len(vs) > 1}
    pinned = {i: vs[-1] for i, vs in impl_versions.items()}
    saved_fixtures = fixtures.FIXTURES
    saved_captured = _CAPTURED_WITH_BY_MODE.get("streamv2")
    result: dict[tuple[str, str], dict[str, dict[str, dict]]] = {}

    def _raw_chunk_counts(impl, version):
        """{(family, case_id): n_chunks} straight from the <impl>-<version> dir docs.
        The resolver pads a folded case to the input chunk count, so alignment
        (did this capture record per-input-chunk timing?) is only visible here."""
        counts: dict[tuple[str, str], int] = {}
        vdir = _STREAM_SRC / f"{impl}-{version}"
        if vdir.is_dir():
            for fp in vdir.glob("*/*.yaml"):
                try:
                    doc = yaml.safe_load(fp.read_text()) or {}
                except Exception:
                    continue
                fam = doc.get("family") or fp.parent.name
                for cid, vc in (doc.get("cases") or {}).items():
                    if isinstance(vc, dict) and isinstance(vc.get("chunks"), list):
                        counts[(fam, cid)] = len(vc["chunks"])
        return counts

    def _record(cases, impl, version):
        slug = toolcalling_table._version_slug(version)
        raw_counts = _raw_chunk_counts(impl, version)
        # The Dynamo parser is version-split into two DIFFERENT parsers: v2
        # (dynamo_rust-0.1.11) implements only a handful of families, while the v1
        # jail (dynamo_rust-3.0.0) covers all. The stream assembly defaults an absent
        # impl to an empty-but-present block, which would paint the v2 parser green on
        # families it doesn't implement. Gate on the version dir's actual family list
        # so uncovered families read `na` (not implemented), not a clean empty output.
        covered = _stream_version_families(impl, version) if impl == BASELINE_IMPL else None
        for key, case in cases.items():
            block = _impl_get(case.get("expected") or {}, impl)
            status = _overview_status(case, impl)
            # Capture this impl's per-chunk deltas at THIS version so the tooltip's
            # per-chunk grid can show a column per (impl, version) candidate.
            vchunks = None
            raw = case.get("chunks")
            if isinstance(raw, list):
                vchunks = []
                for ch in raw:
                    if not isinstance(ch, dict):
                        continue
                    exp = _normalize_impl_mapping(ch.get("expected") or {})
                    nt = _normalize_impl_mapping(ch.get("normal_text") or {})
                    vchunks.append({
                        "deltas": _impl_get(exp, impl, []) or [],
                        "normal_text": _impl_get(nt, impl, "") or "",
                    })
            if covered is not None and case.get("__family") not in covered:
                block, status, vchunks = None, "na", None
            # Aligned = the raw capture recorded one row per INPUT chunk, so a row
            # index is real consumer-visible timing. The v1 jail captures are
            # emission-packed (fewer rows than inputs) — timing NOT recorded.
            raw_n = raw_counts.get((key[0], case.get("__case_id") or ""))
            n_input = len(raw) if isinstance(raw, list) else 0
            aligned = raw_n is None or raw_n == n_input
            result.setdefault(key, {}).setdefault(impl, {})[slug] = {
                "status": status,
                "block": block,
                "version": version,
                "chunks": vchunks,
                "aligned": aligned,
            }

    def _resolve_and_load(select):
        # Resolve under the staged fixtures parent so load_all_cases's
        # `fp.relative_to(script_dir)` stays valid (script_dir is above the tree).
        with tempfile.TemporaryDirectory(dir=str(saved_fixtures.parent)) as tmp:
            subprocess.run(
                [sys.executable, str(resolver),
                 "--fixtures-root", str(_STREAM_SRC),
                 "--out", tmp, "--select", *select],
                check=True, capture_output=True,
            )
            fixtures.FIXTURES = Path(tmp)
            cases, _labels = load_all_cases("streamv2")
        return cases

    try:
        pinned_select = [f"{i}-{pinned[i]}" for i in overlaid]
        # Baseline pinned resolve: record the single-version impls once (their block
        # is version-independent — no overlays exist for them).
        cases = _resolve_and_load(pinned_select)
        for impl, vs in impl_versions.items():
            if impl not in overlaid:
                _record(cases, impl, vs[0])
        # Each versioned peer @ each of its versions, other overlaid peers pinned.
        for impl, versions in overlaid.items():
            for v in versions:
                select = [f"{o}-{v if o == impl else pinned[o]}" for o in overlaid]
                cases = _resolve_and_load(select)
                _record(cases, impl, v)
    finally:
        fixtures.FIXTURES = saved_fixtures
        if saved_captured is not None:
            _CAPTURED_WITH_BY_MODE["streamv2"] = saved_captured
    return result


def _candidate_sig(block) -> str:
    """Canonical signature of a candidate's output; equal signatures = same output."""
    if not isinstance(block, dict) or "unavailable" in block:
        return "na"
    if "error" in block:
        return f"err:{block.get('error')}"
    return json.dumps(
        {"calls": block.get("calls") or [], "normal_text": block.get("normal_text") or ""},
        sort_keys=True, ensure_ascii=False,
    )


def _cmp_json_from_blocks(blocks: dict) -> str:
    """Per-cell `data-cmp` payload from {candidate_key: block}: {key: {sig, leak, na}}.
    `sig` is a per-cell group id (candidates with identical output share an id);
    `na` (unavailable) is excluded from the diff count but still shown in the tooltip."""
    if not blocks:
        return ""
    ids: dict[str, int] = {}
    out: dict[str, dict] = {}
    for key, block in blocks.items():
        sig = _candidate_sig(block)
        out[key] = {
            "sig": ids.setdefault(sig, len(ids)),
            "leak": 1 if (isinstance(block, dict) and _block_tool_call_leaks(block)) else 0,
            "na": 1 if sig == "na" else 0,
        }
    return html_lib.escape(json.dumps(out, separators=(",", ":")), quote=True)


def _candidate_cmp_json(case: dict | None) -> str:
    """Versioned (batch tab) payload: candidate key = <impl>-<version_slug>."""
    ver = (case or {}).get("__ver_status") if isinstance(case, dict) else None
    if not ver:
        return ""
    blocks = {
        f"{impl}-{slug}": info.get("block")
        for impl, by_slug in ver.items()
        for slug, info in by_slug.items()
    }
    return _cmp_json_from_blocks(blocks)


def _impl_cmp_json(case: dict | None, impl_keys: tuple[str, ...]) -> str:
    """Non-versioned tabs (streamv2 / stream-on-batch): candidate key = impl key.
    One capture per impl; the block is the impl's expected output for this cell."""
    if not isinstance(case, dict):
        return ""
    expected = _expected(case)
    return _cmp_json_from_blocks({impl: _impl_get(expected, impl) for impl in impl_keys})


# --- merged compare model ("Tool Calling (batch data)" tab) ---------------------
# The merged tab renders the v1 batch grid, but each cell compares BOTH parser
# flavors over the same batch text: the versioned batch parsers (key <impl>-b-<slug>)
# and the stream parsers run on the batch text (key <impl>-s-<slug>). A cell's
# `__cmp` (ordered [{key, label, block}]) drives its data-cmp payload + per-candidate
# tooltip sections; `_merged_candidate_items()` supplies the matching chip list.
def _stream_on_batch_versions() -> dict[str, str]:
    """{impl: display version} for the merged tab's stream candidates. Dynamo -> the
    v2 crate version; peers -> the engine version the batch-on-stream fixtures were
    captured against (their `captured_with`), since those fixtures are the source of
    the stream blocks shown here."""
    out: dict[str, str] = {}
    dynv = _dynamo_v2_version()
    if dynv:
        out[BASELINE_IMPL] = dynv
    for fp in sorted(STREAM_ON_BATCH_FIXTURES.glob("*/TOOLCALLING.batch*.yaml")):
        doc = yaml.safe_load(fp.read_text()) or {}
        for impl, ver in (doc.get("captured_with") or {}).items():
            if impl == BASELINE_IMPL or impl not in STREAM_IMPL_KEYS:
                continue
            cv = _clean_version(ver)
            if cv:
                out.setdefault(impl, cv)
    return out


def _merged_candidate_items() -> list[dict[str, str]]:
    """Chip list for the merged tab: batch parsers (versioned, keyed <impl>-b-<slug>)
    then the stream parsers on batch (keyed <impl>-s-<slug>). Default layout: A =
    Dynamo v1 batch (from `_candidate_items()`); B = latest vLLM Python + SGLang
    batch; C = everything else (older batch versions + all stream candidates)."""
    out: list[dict[str, str]] = []
    for c in _candidate_items():
        impl = c["impl"]
        out.append({
            "key": f"{impl}-b-{c['slug']}",
            "label": _full_label(impl, c['version'], "batch"),
            "default_bucket": c["default_bucket"],
        })
    stream_versions = _stream_on_batch_versions()
    for impl in STREAM_IMPL_KEYS:
        ver = stream_versions.get(impl)
        slug = toolcalling_table._version_slug(ver) if ver else ""
        out.append({
            "key": f"{impl}-s-{slug}" if slug else f"{impl}-s",
            "label": _full_label(impl, ver, "stream"),
            "default_bucket": "C",
        })
    return out


def _attach_merged_cmp(cases: dict) -> None:
    """Attach `case['__cmp']` to each merged-tab batch case: the batch parsers (from
    `__ver_status`) plus the stream parsers run on the same batch text (from the
    batch-on-stream overlay). Keys/labels mirror `_merged_candidate_items()` so the
    compare chips, data-cmp payloads, and `cand-<key>` tooltip sections line up."""
    sob_cases = _build_stream_on_batch_cases(cases)
    stream_versions = _stream_on_batch_versions()
    for key, case in cases.items():
        if not isinstance(case, dict):
            continue
        items: list[dict] = []
        ver_status = case.get("__ver_status") or {}
        for impl in ("dynamo_rust", "vllm_python", "sglang_python"):
            # Within an engine, LATEST version first (matches the compare bar).
            entries = sorted(
                (ver_status.get(impl) or {}).items(),
                key=lambda kv: toolcalling_table._version_sort_key(str(kv[1].get("version") or "0")),
                reverse=True,
            )
            for slug, info in entries:
                items.append({
                    "key": f"{impl}-b-{slug}",
                    "label": _full_label(impl, info['version'], "batch"),
                    "block": info.get("block"),
                })
        sob = sob_cases.get(key)
        if sob is not None:
            expected = _expected(sob)
            for impl in STREAM_IMPL_KEYS:
                ver = stream_versions.get(impl)
                slug = toolcalling_table._version_slug(ver) if ver else ""
                items.append({
                    "key": f"{impl}-s-{slug}" if slug else f"{impl}-s",
                    "label": _full_label(impl, ver, "stream"),
                    "block": _impl_get(expected, impl),
                })
        if items:
            case["__cmp"] = items


def _compute_stats(
    cases: dict, sub_cases: list[str], families: list[str], cell_text=cell_for
) -> dict[str, int]:
    """Aggregate cell outcomes across the (family × sub_case) grid. `cell_text`
    maps a case to its marker text (cross-engine `cell_for` by default;
    `_sob_cell_text` for the batch-on-stream tab)."""
    s = {
        "families": len(families),
        "sub_cases": len(sub_cases),
        "slots": len(families) * len(sub_cases),
        "real": 0,
        "parity": 0,
        "dynamo_only": 0,
        "documented": 0,
        "research": 0,
        "errors": 0,
        "na": 0,
        "missing": 0,
        "todo": 0,
    }
    for fam in families:
        for sub in sub_cases:
            case = cases.get((fam, sub))
            text = cell_text(case)
            if text == "—":
                s["missing"] += 1
                continue
            if text == "n/a":
                s["na"] += 1
                continue
            if text == "…":
                # Un-implemented Dynamo v2 family: counted as plain n/a in the stats,
                # like the v1 table (no distinct "TODO" bucket). The "…" sentinel is
                # kept only to detect all-unimplemented inventory rows (see all_todo).
                s["na"] += 1
                continue
            s["real"] += 1
            if text == "=":
                s["parity"] += 1
            elif text == "·" or text in {"D", "D_rb", "D_rs"}:
                s["dynamo_only"] += 1
            elif "!" in text:
                s["errors"] += 1
            elif "↯" in text:
                s["documented"] += 1
            elif "?" in text:
                s["research"] += 1
            else:
                s["documented"] += 1
    return s


def _mode_label(mode: str) -> str:
    if mode == "batch":
        return "TOOLCALLING.batch.*"
    if mode == "streamv2":
        return "TOOLCALLING.streamv2.*"
    return mode


def render_html_panel(
    mode: str,
    cases: dict,
    sub_cases: list[str],
    no_vllm: set[str],
    no_sglang: set[str],
    top_n: list[tuple[str, str]],
    others: list[tuple[str, str]],
    active: bool = False,
    *,
    taxonomy_mode: str | None = None,
    parser_stream_context: str | None = None,
    comparison: str = "cross_engine",
) -> dict[str, object]:
    """Render one tab panel through the shared cell/row pipeline.

    `mode` is the panel identity (tab id, label, captured-with key). Three axes can
    differ from it so every tab — including batch-on-stream — flows through this
    one function instead of a bespoke builder:
      - `taxonomy_mode`: which sub-case taxonomy drives the column groups, bands,
        descriptions, and glossary. Defaults to `mode`. Batch-on-stream passes
        "batch" (it reuses the v1 batch sub-cases).
      - `parser_stream_context`: which parser-column variant to render. Defaults to
        `mode`. Batch-on-stream passes "batch_on_stream".
      - `comparison`: how cell status/markers are derived — "cross_engine" (batch /
        stream) or "stream_vs_batch" (batch-on-stream).
    """
    taxonomy_mode = taxonomy_mode or mode
    if parser_stream_context is None:
        parser_stream_context = mode
    cell_text = (
        (lambda case: _sob_cell_text(case, parser_stream_context))
        if comparison == "stream_vs_batch"
        else cell_for
    )
    descriptions = _parse_subcase_descriptions(taxonomy_mode)
    refs = _build_family_to_rust_ref()
    inheritance = _build_family_inheritance(refs)

    group_headers = _subcase_group_headers_html(taxonomy_mode, sub_cases)
    sub_headers = _subcase_headers_html(taxonomy_mode, sub_cases, descriptions)
    n_cols = 2 + len(sub_cases)

    body_rows: list[str] = []
    if top_n:
        body_rows.append(
            f'<tr class="section"><td data-section-span colspan="{n_cols}">'
            "Top-N models</td></tr>"
        )
    for model, fam in top_n:
        body_rows.append(
            render_row_html(
                model,
                fam,
                taxonomy_mode,
                cases,
                sub_cases,
                refs,
                no_vllm,
                no_sglang,
                inheritance,
                parser_stream_context=parser_stream_context,
                comparison=comparison,
            )
        )
    if others:
        body_rows.append(
            f'<tr class="section"><td data-section-span colspan="{n_cols}">'
            "Others</td></tr>"
        )
    for model, fam in others:
        body_rows.append(
            render_row_html(
                model,
                fam,
                taxonomy_mode,
                cases,
                sub_cases,
                refs,
                no_vllm,
                no_sglang,
                inheritance,
                parser_stream_context=parser_stream_context,
                comparison=comparison,
            )
        )

    all_families = [fam for _, fam in top_n] + [fam for _, fam in others]
    stats = _compute_stats(cases, sub_cases, all_families, cell_text=cell_text)

    panel_id = f"tab-{mode}"
    captured = _CAPTURED_WITH_BY_MODE.get(mode) or {}
    captured_note = ""
    if captured:
        pairs = ", ".join(
            f"{impl} {ver}" for impl, ver in sorted(captured.items())
        )
        captured_note = (
            f"Peer streaming output captured against: {pairs}. "
            "A divergence is relative to these versions; re-capture when bumping."
        )
    panel = {
        "id": panel_id,
        "mode": mode,
        "label": _mode_label(mode),
        "active": active,
        "group_headers": group_headers,
        "sub_headers": sub_headers,
        "body_rows": body_rows,
        "stats": stats,
        "glossary_groups": _glossary_groups(taxonomy_mode, descriptions, sub_cases),
        "captured_note": captured_note,
        "parser_options": _impl_keys_for_output_kind(
            "batch" if parser_stream_context == "batch" else "stream"
        ),
    }
    if comparison == "stream_vs_batch":
        panel["details_note_html"] = f"<p>{_stream_parity_explainer_html(parser_stream_context)}</p>"
    return panel


def _build_sob_tooltip(case: dict, marker_context: str | None = None) -> str:
    """Tooltip for a stream-tab cell (TC stream v2 + batch-on-stream).

    Two comparisons, both shown:
      - COLOR is stream-vs-own-batch, so the BATCH reference (X_b) is shown for
        every engine.
      - The STREAM side (X_s) is shown either as the per-chunk chart (TC stream v2
        cases carry `chunks:`; its final `assembled` row IS X_s) or, when there is
        no per-chunk breakdown (batch-on-stream overlay), as a plain X_s block.
    """
    family = case.get("__family")
    cid = case.get("__case_id", "")
    head = f"{cid} — {family}" if family else cid
    desc = case.get("description") or ""
    model_text = case.get("model_text")
    input_html = (
        f"input_text='{colorize_markup(model_text, family)}'"
        if isinstance(model_text, str) and model_text
        else None
    )
    expected = _expected(case)
    batch = _normalize_impl_mapping(case.get("batch_expected") or {})
    # The per-chunk chart (chunk-by-chunk emit + an assembled X_s row + a batch X_b
    # row) is the heart of the streaming tab; only TC stream v2 cases carry
    # `chunks:`. When it's present it already shows both X_s and X_b, so the
    # separate per-engine blocks below would be redundant — skip them.
    chart = _per_chunk_chart_html(case, "stream")
    # A CANDIDATE chart (data-cand columns = the compare candidates) already carries
    # each candidate's per-chunk emit + assembled output (with explanation) in its
    # own toggleable/REF-ordered column — the per-candidate list sections would
    # repeat it one by one, so they are dropped when that chart rendered.
    cand_chart = chart is not None and "data-cand" in chart[1]
    sections: list[tuple] = []
    ver_status = case.get("__ver_status") or {}
    if ver_status and not cand_chart:
        # Versioned candidates (impl×engine-version, e.g. vLLM 0.23.0 vs 0.24.0):
        # one toggleable section each showing that version's assembled stream output.
        # Kept alongside a legacy impl-keyed chart so the Base/Compare selection
        # reveals each candidate — the stream analogue of the batch per-version blocks.
        for impl in ("dynamo_rust", "vllm_rust", "vllm_python", "sglang_python"):
            for slug, info in (ver_status.get(impl) or {}).items():
                blk = info["block"]
                sections.append((
                    _full_label(impl, info["version"], "stream"),
                    _cand_section_body(blk, family),
                    f"cand cand-{impl}-{slug}",
                    isinstance(blk, dict) and _block_tool_call_leaks(blk),
                ))
    elif not chart:
        for impl in IMPL_KEYS:
            lbl = _cand_label(impl)  # friendly candidate name, e.g. "vLLM Rust 0.23.0"
            sblk = _impl_get(expected, impl)
            sections.append(
                (f"{lbl} (stream)", _format_output_block_html(sblk, family), f"cand cand-{impl}",
                 isinstance(sblk, dict) and _block_tool_call_leaks(sblk))
            )
            if impl != "vllm_rust":
                bblk = _impl_get(batch, impl)
                sections.append(
                    (f"{lbl} (batch)", _format_output_block_html(bblk, family), f"cand cand-{impl}",
                     isinstance(bblk, dict) and _block_tool_call_leaks(bblk))
                )
    # Engine ascending, version DESCENDING within the engine (latest first).
    _sort_candidate_sections(sections)
    return _build_conformance_tooltip_html(
        head=head,
        description=desc,
        input_label="Input" if input_html else None,
        input_html=input_html,
        output_sections=sections,
        # Coloring is leak-only in the compare model, so the old cross-impl
        # "stream diverges from batch" blob is stale and would name engines not in
        # the current selection; each candidate's own reason is in its section.
        divergent_reasons_html=None,
        extra_sections=_dynamo_note_sections(case),
        chart=chart,
        refs=[("Ref", case.get("ref"))],
        html_section_labels=True,
    )


def _stream_on_batch_expected(overlay_case: dict, has_batch_text: bool = True) -> dict:
    """Build a standard `expected` block — `{impl: {calls, normal_text}}` (or
    `{unavailable}`) — from one batch-on-stream overlay case.

    The overlay records each engine's STREAMING parse of the v1 batch text. Some
    overlay rows are taxonomy placeholders with no batch `model_text`; render
    those as structural unavailability instead of claiming the parser is missing.
    Peer outputs are tagged with a `reason` so the
    conformance marker reads as an intentional divergence (`V_ps`/`S_rs`), not
    research-needed (`V_ps?`/`S_rs?`) — text-vs-token streaming differs by design.
    """
    expected: dict = {}
    overlay_case = _normalize_impl_mapping(overlay_case)
    dynamo = _impl_get(overlay_case, BASELINE_IMPL)
    if isinstance(dynamo, dict) and ("calls" in dynamo or "normal_text" in dynamo):
        expected[BASELINE_IMPL] = {
            "calls": dynamo.get("calls") or [],
            "normal_text": dynamo.get("normal_text") or "",
        }
    elif not has_batch_text:
        expected[BASELINE_IMPL] = {"unavailable": "No batch model_text for this case."}
    else:
        expected[BASELINE_IMPL] = {
            "unavailable": "Dynamo parser v2 stream parser not yet implemented for this family"
        }
    for impl in PEER_IMPL_KEYS:
        block = _impl_get(overlay_case, impl)
        if not isinstance(block, dict):
            expected[impl] = {
                "unavailable": "No batch-on-stream capture for this engine."
            }
        elif "unavailable" in block:
            expected[impl] = {"unavailable": block["unavailable"]}
        else:
            expected[impl] = {
                "calls": block.get("calls") or [],
                "normal_text": block.get("normal_text") or "",
                "explanation": (
                    f"Captured from the {IMPL_DISPLAY[impl]} streaming parser on the batch text. "
                    "Streaming output differs from Dynamo parser v2 token-incremental "
                    "behavior by design (text vs token streaming)."
                ),
            }
    return expected


def _load_stream_on_batch_overlay() -> dict[tuple[str, str], dict]:
    """`{(family, case_id): {impl: stream_block}}` from the batch-on-stream overlay."""
    overlay: dict[tuple[str, str], dict] = {}
    if not STREAM_ON_BATCH_FIXTURES.exists():
        return overlay
    for fp in sorted(STREAM_ON_BATCH_FIXTURES.glob("*/TOOLCALLING.batch*.yaml")):
        doc = yaml.safe_load(fp.read_text()) or {}
        family = doc.get("family") or fp.parent.name
        for cid, block in (doc.get("cases") or {}).items():
            overlay[(family, cid)] = block
    return overlay


def _build_stream_on_batch_cases(batch_cases: dict) -> dict:
    """Standard `{(family, sub): case}` for the batch-on-stream tab.

    Reuses the v1 batch taxonomy and input text. `expected` holds each engine's
    STREAMING output (from the overlay); `batch_expected` holds the v1 batch
    reference. The cell renderer compares the two per engine (stream-vs-batch).
    Sub-cases with no overlay sample are omitted (rendered as `—`).
    """
    overlay = _load_stream_on_batch_overlay()
    cases: dict[tuple[str, str], dict] = {}
    for (family, sub), bcase in batch_cases.items():
        cid = bcase.get("__case_id") or f"TOOLCALLING.batch.{sub}"
        overlay_case = overlay.get((family, cid))
        if overlay_case is None and cid.endswith(".a"):
            # The generator promotes a bare parent id (e.g. `…13`) to `…13.a`; the
            # overlay may still key it by the bare parent id. Fall back to that.
            overlay_case = overlay.get((family, cid[:-2]))
        if overlay_case is None:
            continue
        cases[(family, sub)] = {
            "__family": family,
            "__case_id": cid,
            "__fixture_path": bcase.get("__fixture_path", ""),
            "description": bcase.get("description"),
            "model_text": bcase.get("model_text"),
            "ref": bcase.get("ref"),
            # Baseline rationale lives in the fc-local overlay (sync-safe); fall
            # back to the synced v1 batch fixture if it ever carries one upstream.
            "dynamo_note": overlay_case.get("dynamo_note") or bcase.get("dynamo_note"),
            "expected": _stream_on_batch_expected(
                overlay_case, has_batch_text="model_text" in bcase
            ),
            "batch_expected": _normalize_impl_mapping(bcase.get("expected") or {}),
        }
    return cases


def _relabel_harmony_text_path(
    groups: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Display the batch-on-stream Harmony row as the text path (it runs
    `parse_tool_call_streaming_text` only). Leaves every other row's label
    untouched."""
    return [
        ("gpt-oss (harmony, text)" if fam == "harmony" else label, fam)
        for label, fam in groups
    ]


def build_stream_on_batch_panel(active: bool = False) -> dict[str, object]:
    """The 'TC batch-on-stream (v2)' tab: each engine's STREAMING parser run over
    the v1 batch text, rendered through the shared panel pipeline.

    It reuses the v1 batch taxonomy (`taxonomy_mode="batch"`) and the v2 stream
    parser column (`parser_stream_context="batch_on_stream"`) through the shared
    panel pipeline. Its markers use the `stream_vs_batch` comparison (two
    dimensions per cell): default mode shows only the leak marker; the cell COLOR
    is each engine's stream parse vs its own batch parse (green consistent, red
    divergent), and the Conformance toggle shows cross-engine output agreement
    (`=`, or `D_rs`/`V_ps`/`V_rs`/`S_rs` for stream parser outputs).
    """
    batch_cases, labels = load_all_cases("batch")
    cases = _build_stream_on_batch_cases(batch_cases)
    sub_cases = _discover_sub_cases("batch", batch_cases)
    no_vllm, no_sglang = _derive_no_peer_sets(batch_cases)
    top_n, others = _build_display_groups(batch_cases, labels)
    # Batch-on-stream feeds each batch fixture's full text through the v2 stream
    # parser. For Harmony that is the TEXT path only (parse_tool_call_streaming_text):
    # batch fixtures carry no token IDs, so there is no token-id path on this tab.
    # Relabel the row to say so — matching the TC stream tab's "gpt-oss (harmony,
    # text)" — without touching the shared v1 batch label `gpt-oss`, which the plain
    # batch tab (the batch parser, not the stream parser) still shows.
    top_n, others = _relabel_harmony_text_path(top_n), _relabel_harmony_text_path(others)
    return render_html_panel(
        "stream_on_batch",
        cases,
        sub_cases,
        no_vllm,
        no_sglang,
        top_n,
        others,
        active,
        taxonomy_mode="batch",
        parser_stream_context="batch_on_stream",
        comparison="stream_vs_batch",
    )


def _filter_family(
    cases: dict[tuple[str, str], dict],
    labels: dict[str, str],
    family_filter: str | None,
) -> tuple[dict[tuple[str, str], dict], dict[str, str]]:
    if family_filter is None:
        return cases, labels
    return (
        {k: v for k, v in cases.items() if k[0] == family_filter},
        {k: v for k, v in labels.items() if k == family_filter},
    )


def _load_html_panel(
    mode: str,
    active: bool = False,
    family_filter: str | None = None,
) -> tuple[str, dict[str, object], bool]:
    cases, labels = load_all_cases(mode)
    cases, labels = _filter_family(cases, labels, family_filter)
    # TC v1 (batch) tab: attach per-impl per-version status so cells can emit
    # data-status-<impl>-<slug> for the version radios. Other tabs aren't versioned.
    if mode == "batch":
        ver_status = _batch_version_status_map()
        for key, case in cases.items():
            if isinstance(case, dict) and key in ver_status:
                case["__ver_status"] = ver_status[key]
        # Merged "Tool Calling (batch data)" tab: augment each cell so the compare
        # model spans both the batch parsers (from __ver_status) and the stream
        # parsers run on the same batch text (batch-on-stream overlay).
        _attach_merged_cmp(cases)
    elif mode == "streamv2":
        # Stream analogue of the batch version map: per-cell candidates are the
        # peer engine versions (vLLM 0.23.0/0.24.0, SGLang 0.5.12.post1/0.5.14),
        # plus single-version Dynamo v2 + vLLM Rust.
        ver_status = _stream_version_status_map()
        for key, case in cases.items():
            if isinstance(case, dict) and key in ver_status:
                case["__ver_status"] = ver_status[key]
    has_cases = bool(cases)
    sub_cases = _discover_sub_cases(mode, cases)
    no_vllm, no_sglang = _derive_no_peer_sets(cases)
    top_n, others = _build_display_groups(cases, labels)
    # The streamv2 tab uses the stream comparison: color = stream-vs-own-batch,
    # conformance marker = cross-engine stream agreement (`Y_s`).
    comparison = "stream_vs_batch" if mode == "streamv2" else "cross_engine"
    return (
        mode,
        render_html_panel(
            mode, cases, sub_cases, no_vllm, no_sglang, top_n, others, active,
            comparison=comparison,
        ),
        has_cases,
    )


def render_html(modes: list[str], family_filter: str | None = None) -> str:
    panels = [
        _load_html_panel(mode, active=(i == 0), family_filter=family_filter)
        for i, mode in enumerate(modes)
    ]
    if family_filter and not any(has_cases for _mode, _panel, has_cases in panels):
        raise SystemExit(f"no parser fixtures found for family={family_filter!r}")

    now = datetime.datetime.now(zoneinfo.ZoneInfo("America/Los_Angeles"))
    stamp = now.strftime("%Y-%m-%d %H:%M %Z")
    title = (
        f"Dynamo {family_filter} Tool Calling Parser v2 Conformance Table"
        if family_filter
        else "Dynamo Tool Calling Parser v2 Conformance Table"
    )
    command = "python3 tests/parity/generate_conformance_table.py toolcalling --html"
    output = "tests/parity/toolcalling/CONFORMANCE.html"
    if family_filter:
        command += f" --family {family_filter}"
        output = f"tests/parity/toolcalling/CONFORMANCE.{family_filter}.html"

    sha = _commit_sha()
    legend_html = _common_legend_html(_peer_version_items(_peer_versions()))
    for _mode, panel, _has_cases in panels:
        panel["legend_html"] = legend_html

    if len(modes) == 1:
        command += f" --mode {modes[0]}"
        output = f"tests/parity/toolcalling/CONFORMANCE.{modes[0]}.html"
        if family_filter:
            output = f"tests/parity/toolcalling/CONFORMANCE.{family_filter}.{modes[0]}.html"

    tabs = []
    for i, (mode, _panel, _has_cases) in enumerate(panels):
        panel_id = f"tab-{mode}"
        active = " active" if i == 0 else ""
        selected = "true" if i == 0 else "false"
        tabs.append(
            f'<button class="tab-button{active}" id="{panel_id}-button" '
            f'type="button" role="tab" aria-selected="{selected}" '
            f'data-tab-target="{panel_id}">{html_lib.escape(_mode_label(mode))}</button>'
        )

    html = (
        _make_jinja_env()
        .get_template("conformance_table.html.j2")
        .render(
            title=title,
            title_html=title.replace(
                "frontend-crate",
                '<span class="title-code">frontend-crate</span>',
            ),
            stamp=stamp,
            conformance_css=_read_asset("conformance.css"),
            conformance_js=_read_asset("conformance.js"),
            impl_status_css=_impl_status_css(),
            sha=sha,
            short_sha=sha[:12] if sha else "",
            command=command,
            output=output,
            tabs=tabs,
            panels=[panel for _mode, panel, _has_cases in panels],
            parser_ni_json=json.dumps(_parser_ni_map()),
        )
    )
    return _scrub_visible_conformance_text(html)


# ===== merged combined-renderer orchestrator (was generate_conformance_table.py) =====

def _rewrite_panel_paths(
    panel: dict[str, Any],
    stage_dir: str,
    fixture_href_root: str,
) -> dict[str, Any]:
    """Rebase a panel's fixture cell links onto that panel's real fixtures root.

    This per-panel transform is irreducible: the same loaded case is rendered into
    multiple panels with different fixture roots (e.g. a batch case appears in both
    the `batch` and `batch-on-stream` panels), so the root can't be baked into the
    case at load time. Every other link (case docs, parser source, pyproject) is
    emitted destination-aware via `common.LINKS` and needs no rewrite.
    """
    rewritten = dict(panel)

    def rewrite(text: str) -> str:
        # Fixture cell links are emitted as `{stage_dir}/fixtures/...` (toolcalling
        # builder, merged into this file) or bare `fixtures/...` (reasoning builder).
        # Order matters: the stage_dir form is rewritten first so the bare-`fixtures/`
        # pass can't re-match its `../<dir>/fixtures/` output.
        return text.replace(
            f'href="{stage_dir}/fixtures/', f'href="{fixture_href_root}'
        ).replace('href="fixtures/', f'href="{fixture_href_root}')

    rewritten["group_headers"] = rewrite(str(rewritten["group_headers"]))
    rewritten["sub_headers"] = rewrite(str(rewritten["sub_headers"]))
    rewritten["body_rows"] = [rewrite(str(row)) for row in rewritten["body_rows"]]
    return rewritten


def _tab_label(
    prefix: str,
    data: str,
    parser: str | None,
    v2: bool,
    data_word: bool = True,
    on_parser: bool = True,
) -> tuple[str, str]:
    """Build a tab label as `<prefix> vN (<data> data on <parser>-parser)`.
    Returns (plain, html); the html form wraps the parenthetical in a smaller-font
    span (`tab-sub`) and color-codes the words "batch"/"stream" (`w-batch`/`w-stream`)
    so the two axes are distinguishable. `data` is "batch" or "stream". `parser` is
    "batch"/"stream", or None for a bare "parser" (reasoning has a single parser, not
    a batch/stream split). `data_word=False` drops the literal " data" word.
    `on_parser=False` drops the `on <parser>-parser` clause entirely, so reasoning
    renders `(batch data)` — the parser axis is meaningless there (one parser)."""
    version = "v2" if v2 else "v1"
    dword = " data" if data_word else ""

    def _w(word: str) -> str:
        return f'<span class="w-{word}">{word}</span>'

    if on_parser:
        parser_plain = f"{parser}-parser" if parser else "parser"
        parser_html = f"{_w(parser)}-parser" if parser else "parser"
        on_plain = f" on {parser_plain}"
        on_html = f" on {parser_html}"
    else:
        on_plain = on_html = ""
    plain = f"{prefix} {version} ({data}{dword}{on_plain})"
    sub_html = f"({_w(data)}{dword}{on_html})"
    return plain, f'{prefix} {version} <span class="tab-sub">{sub_html}</span>'


def _tab_button(panel: dict[str, Any]) -> str:
    active = " active" if panel["active"] else ""
    selected = "true" if panel["active"] else "false"
    panel_id = html_lib.escape(str(panel["id"]))
    # label_html (when set) carries the smaller-font parenthetical; fall back to the
    # plain, escaped label otherwise.
    label = (
        str(panel["label_html"])
        if panel.get("label_html")
        else html_lib.escape(str(panel["label"]))
    )
    title = html_lib.escape(str(panel.get("tab_title", panel["label"])))
    # The per-panel provenance banner (toolbar_desc) renders inside each panel
    # after the matrix table; the tab button no longer carries it.
    return (
        f'<button class="tab-button{active}" id="{panel_id}-button" '
        f'type="button" role="tab" aria-selected="{selected}" '
        f'aria-label="{title}" title="{title}" '
        f'data-tab-target="{panel_id}">{label}</button>'
    )


def _apply_common_legend(panels: list[dict[str, Any]], hrefs: dict[str, str]) -> None:
    legend_html = _common_legend_html(
        _peer_version_items(_peer_versions()),
        hrefs["pyproject_stub"],
    )
    # One legend for every tab: the compare model (Reference vs Compare) is identical
    # across tabs, so they all get the same rich legend.
    for panel in panels:
        panel["legend_html"] = legend_html


def _combined_toolcalling_panels(hrefs: dict[str, str]) -> list[dict[str, Any]]:
    panels = []

    # --- Merged "Tool Calling (batch data)" tab ---
    # One tab over the v1 batch input. Each cell's compare model spans BOTH parser
    # flavors on the same batch text: the versioned batch parsers (<impl>-b-<slug>)
    # and the stream parsers run on the batch text (<impl>-s-<slug>). This replaces
    # the two former tabs (batch + batch-on-stream), which shared the same input.
    _mode, batch_panel, _has_cases = _load_html_panel("batch")
    batch_panel = _rewrite_panel_paths(
        batch_panel, "toolcalling",
        fixture_href_root=hrefs["toolcalling_fixtures"],
    )
    batch_panel.update(
        {
            "id": "tab-toolcalling-batch",
            "label": "Tool Calling (batch data)",
            "label_html": (
                'Tool Calling <span class="tab-sub">'
                '(<span class="w-batch">batch</span> data)</span>'
            ),
            "tab_title": (
                "Tool Calling (batch data): v1 batch parsers plus v2 stream parsers "
                "on the same v1 batch fixtures"
            ),
            "active": False,
            "case_docs_href": hrefs["toolcalling_cases"],
            "case_docs_label": "lib/parsers/TOOLCALLING_CASES.md",
            "case_prefix": "TOOLCALLING.batch.",
            "case_section_id": "toolcalling-batch",
            "parser_options": BATCH_IMPL_KEYS,
            "candidates": _merged_candidate_items(),
            "toolbar_desc": (
                f'Parsers: <strong>v1</strong> Dynamo-synced batch '
                f'(<a href="{hrefs["toolcalling_src"]}">parsers/src/tool_calling/</a>) '
                f'plus <strong>v2</strong> streaming on the same batch text '
                f'(<a href="{hrefs["streaming_src"]}">parsers_v2/src/tool_calling/*</a>) · '
                f'Input: <strong>v1</strong> batch fixtures '
                f'(<a href="{hrefs["toolcalling_fixtures"]}">conformance/toolcalling/fixtures-batch-v1/</a>).'
            ),
        }
    )
    panels.append(batch_panel)

    # --- "Tool Calling (stream data)" tab (per-chunk streamv2; data unchanged) ---
    _mode, stream_panel, _has_cases = _load_html_panel("streamv2")
    stream_panel = _rewrite_panel_paths(
        stream_panel, "toolcalling",
        fixture_href_root=hrefs["toolcalling_stream_fixtures"],
    )
    stream_panel.update(
        {
            "id": "tab-toolcalling-streamv2",
            "label": "Tool Calling (stream data)",
            "label_html": (
                'Tool Calling <span class="tab-sub">'
                '(<span class="w-stream">stream</span> data)</span>'
            ),
            "tab_title": "Tool Calling (stream data): Dynamo parser v2 on v2 stream fixtures",
            "active": False,
            "case_docs_href": hrefs["toolcalling_streaming_cases"],
            "case_docs_label": "lib/parsers/TOOLCALLING_STREAMING_V2_CASES.md",
            "case_prefix": "TOOLCALLING.streamv2.",
            "case_section_id": "toolcalling-streamv2",
            "parser_options": STREAM_IMPL_KEYS,
            "candidates": _stream_candidate_items(),
            "toolbar_desc": (
                f'Parser: <strong>v2</strong> Dynamo parser v2 token-incremental streaming '
                f'(<a href="{hrefs["streaming_src"]}">parsers_v2/src/tool_calling/*</a>) · '
                f'Input: <strong>v2</strong> stream fixtures '
                f'(<a href="{hrefs["toolcalling_stream_fixtures"]}">conformance/toolcalling/fixtures-stream-v2/</a>).'
            ),
        }
    )
    panels.append(stream_panel)


    _apply_common_legend(panels, hrefs)
    return panels


def _combined_reasoning_panels(hrefs: dict[str, str]) -> list[dict[str, Any]]:
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
        panel = _rewrite_panel_paths(
            panel,
            "reasoning",
            fixture_href_root=hrefs["reasoning_fixtures"],
        )
        # Reasoning has a single parser (not a batch/stream split), so the parser
        # axis renders as a bare "parser"; only the data axis varies.
        _r_label, _r_label_html = _tab_label(
            "Reasoning", mode, None, False, on_parser=False
        )
        panel.update(
            {
                "id": f"tab-reasoning-{mode}",
                "label": _r_label,
                "label_html": _r_label_html,
                "tab_title": f"Reasoning {mode}: v1 code on v1 fixtures",
                "toolbar_desc": (
                    f'Parser: <strong>v1</strong> Dynamo-synced reasoning parser '
                    f'(<a href="{hrefs["reasoning_src"]}">parsers/v1/src/reasoning/</a>) · '
                    f'Input: <strong>v1</strong> reasoning fixtures '
                    f'(<a href="{hrefs["reasoning_fixtures"]}">conformance/reasoning/fixtures/</a>).'
                ),
                "active": False,
                "case_docs_href": hrefs["reasoning_cases"],
                "case_docs_label": "lib/parsers/REASONING_CASES.md",
                "case_prefix": "REASONING.",
                "case_section_id": f"reasoning-{mode}",
                "parser_options": ("dynamo_rust", "vllm_python", "sglang_python"),
            }
        )
        panels.append(panel)
    _apply_common_legend(panels, hrefs)
    return panels


def render_combined_html(
    output_path: Path | None = None,
    artifact_root: Path | None = None,
) -> str:
    artifact_root = (artifact_root or REPO_ROOT).resolve()
    resolved_output_path = _resolve_output_path(
        output_path,
        artifact_root,
        "tests/parity/CONFORMANCE.html",
    )
    hrefs = common.set_links(resolved_output_path, artifact_root)
    panels = [
        *_combined_toolcalling_panels(hrefs),
        *_combined_reasoning_panels(hrefs),
    ]
    panels[0]["active"] = True
    # Color the trailing (batch)/(stream) mode word in every compare candidate label.
    for panel in panels:
        for cand in panel.get("candidates", []):
            cand["label_html"] = _candidate_label_html(cand["label"])

    now = datetime.datetime.now(zoneinfo.ZoneInfo("America/Los_Angeles"))
    stamp = now.strftime("%Y-%m-%d %H:%M %Z")
    sha = _commit_sha()

    html = (
        _make_jinja_env()
        .get_template("conformance_table.html.j2")
        .render(
            title="Dynamo Parser v2 Conformance Table",
            title_html="Dynamo Parser v2 Conformance Table",
            stamp=stamp,
            conformance_css=_read_asset("conformance.css"),
            conformance_js=_read_asset("conformance.js"),
            impl_status_css=_impl_status_css(),
            sha=sha,
            short_sha=sha[:12] if sha else "",
            command="conformance/utils/render_table_v2.sh",
            output=_display_path(resolved_output_path, artifact_root),
            tabs=[_tab_button(panel) for panel in panels],
            panels=panels,
            impl_versions=_impl_version_items(),
            candidate_items=_candidate_items(),
            parser_ni_json=json.dumps(_parser_ni_map()),
        )
    )
    return _scrub_visible_conformance_text(html)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate frontend-crate conformance tables.",
    )
    parser.add_argument(
        "stage",
        choices=("all", "toolcalling", "reasoning"),
        help="Conformance stage to render.",
    )
    args, rest = parser.parse_known_args(argv)

    if args.stage == "all":
        stage_parser = argparse.ArgumentParser(
            description="Generate the combined frontend-crate parser conformance HTML page.",
        )
        stage_parser.add_argument(
            "--html",
            action="store_true",
            help="Emit the combined HTML page.",
        )
        stage_parser.add_argument(
            "--output-path",
            type=Path,
            help="Output file path used to compute relative links. The HTML is still written to stdout.",
        )
        stage_parser.add_argument(
            "--artifact-root",
            type=Path,
            help="Repo root that output links should target. Defaults to the staged repo root.",
        )
        stage_args = stage_parser.parse_args(rest)
        if not stage_args.html:
            parser.error("stage 'all' currently supports --html only")
        print(
            render_combined_html(
                output_path=stage_args.output_path,
                artifact_root=stage_args.artifact_root,
            )
        )
        return

    stage_table = toolcalling_table if args.stage == "toolcalling" else reasoning_table
    stage_table.main(rest)


if __name__ == "__main__":
    main()
