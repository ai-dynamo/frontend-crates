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
  =     peer block matches the Dynamo baseline block (`expected.dynamo_v1` batch / `expected.dynamo_v2` stream)
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
import functools
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
# PERF: this render loads thousands of fixture YAMLs (the stream version-status map
# re-loads every peer version); PyYAML's pure-Python SafeLoader dominates the wall
# clock. Route safe_load through libyaml's CSafeLoader (identical result, ~15x faster)
# when the C extension is present. fixtures.py / markers.py call `yaml.safe_load` at
# call time, so patching the module here covers them too.
if hasattr(yaml, "CSafeLoader"):
    yaml.safe_load = lambda _s, _loader=yaml.CSafeLoader: yaml.load(_s, Loader=_loader)
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from tests.parity import common
from tests.parity.common import TOP_N_TOOL_CALLING_FAMILIES as TOP_N_FAMILIES
from tests.parity.common import (
    linkify_text_html,
    parity_cell_class,
)
from tests.parity.markup import (
    colorize_markup,
    colorize_stream_deltas,
    declared_markers,
)
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
    BASELINE_BATCH_IMPL,
    BASELINE_IMPLS,
    BASELINE_STREAM_IMPL,
    BATCH_IMPL_KEYS,
    baseline_impl,
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
import markers  # noqa: E402  (module handle: structured comparison model, DIS-2434)
import unified_taxonomy  # noqa: E402  (shared UNIFIED scenario->numbered-id taxonomy)
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
    _impl_mode_letter,
    _impl_mode_suffix,
    _is_parser_error_unavailable,
    _is_todo_unavailable,
    _legacy_impl_keys,
    _norm_calls,
    _normalize_impl_mapping,
    _overview_status,
    _parity_marker,
    _parser_marker,
    _selected_parity_marker,
    _selected_parity_suffix,
    _sob_calls_consistent,
    _sob_cell_text,
    _sob_status,
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
    dyn = _impl_get(expected, baseline_impl(impl_keys))
    if not isinstance(dyn, dict):
        return "n/a"
    # Dynamo parser v2 unavailable for this case: distinguish "not yet implemented"
    # (TODO, the whole family) from a structural n/a (e.g. a token parser can't
    # consume a character-split fixture per-chunk).
    if "unavailable" in dyn:
        return "…" if _is_todo_unavailable(dyn) else "n/a"
    parts: list[str] = []
    peer_kinds: dict[str, str] = {}
    for impl in (peer for peer in impl_keys if peer not in BASELINE_IMPLS):
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

    # `explanation:` on the `expected.dynamo_v2` block flags Dynamo parser v2 output as
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


_IMPL_DISPLAY = IMPL_DISPLAY


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
    # The tree draws with box-drawing characters and only lines up in a fixed-width
    # font, so it opts back into monospace while the prose around it stays proportional.
    tree_lines = _tool_parser_tree_lines(family, info, effective_backend)
    if tree_lines:
        tooltip_lines.append(
            '<span class="ttip-tree">' + "\n".join(tree_lines).strip("\n") + "</span>"
        )

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


# Dynamo parser v2 stream parsers with a standard `push`/`finish` text path:
# family -> (backend label, source file under parsers/v2/src/tool_calling/, format marker).
# Families with bespoke paths (harmony token-id/text, deepseek_v4 DSML note) keep their
# dedicated branches below; new standard families belong here, not in new if-branches.
_V2_STREAM_PARSER_CELLS: dict[str, tuple[str, str, str]] = {
    "gemma4": ("Gemma4ToolStreamParser text path", "gemma4.rs", "Gemma"),
    "glm47": ("Glm47ToolStreamParser text path", "glm47.rs", "GLM XML"),
    "kimi_k2": ("KimiK2ToolStreamParser text path", "kimi_k2.rs", "Kimi XML"),
    "minimax_m2": ("MiniMaxM2ToolStreamParser text path", "minimax_m2.rs", "MiniMax XML"),
    "minimax_m3": ("MiniMaxM3ToolStreamParser text path", "minimax_m3.rs", "MiniMax-M3 XML"),
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
        '<pre class="ttip-pre ttip-note">'
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
        '<pre class="ttip-pre ttip-note">'
        "Dynamo parser v2 is not implemented for this family yet.\n"
        "This row is inventory only; no v1 parser code runs on this tab."
        "</pre></div>"
    )
    return f'<td class="parser" data-col-hide-group="parser">{label}{tooltip}</td>'


def _has_peer_output(case: dict | None) -> bool:
    if not case:
        return False
    expected = _expected(case)
    for impl in PEER_IMPL_KEYS:
        block = _impl_get(expected, impl)
        if isinstance(block, dict) and "unavailable" not in block:
            return True
    return False


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
    "dynamo_v1": "dynamo_v1",
    "dynamo_v2": "dynamo_v2",
    "vllm_python": "vllm_python",
    "vllm_rust": "vllm_rust",
    "sglang_python": "sglang_python",
    # legacy spellings, accepted on read
    "dynamo": "dynamo_v1",
    "dynamo_rust": "dynamo_v2",
    "vllm": "vllm_python",
    "sglang": "sglang_python",
}
_IMPL_VERSION_RADIO_LABEL = {
    "dynamo_v1": "Dynamo v1 Rust",
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


# --- compare-any-combination model (TC v1 tab) ---------------------------------
# Every (parser, version) is a "candidate". A cell reports how many of the
# user-selected candidates differ from the chosen Base; the tooltip shows Base +
# each selected candidate's output. All of it is computed client-side from the
# compact per-cell `data-cmp` payload below, so any base/compare combination works.
_CANDIDATE_SHORT = {
    "dynamo_v1": "Dynamo v1",
    "dynamo_v2": "Dynamo v2",
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
    "dynamo_v1": "Dynamo v1 Rust",
    "dynamo_v2": "Dynamo v2 Rust",
    "vllm_rust": "vLLM Rust",
    "vllm_python": "vLLM Python",
    "sglang_python": "SGLang Python",
}




def _full_label(impl: str, version: object, mode: str) -> str:
    base = _ENGINE_RUNTIME.get(impl, _CANDIDATE_SHORT.get(impl, impl))
    # The v1/v2 generation is part of the impl key (dynamo_v1/dynamo_v2), so the
    # display already reads "Dynamo v1 Rust 3.0.0 (batch)" / "Dynamo v2 Rust
    # 0.1.11 (stream)". The one remaining special case: v1 run against stream
    # data goes through the streaming jail (buffer, then v1 batch parse), so on
    # the stream tab its mode reads "(jail+batch)".
    if impl == BASELINE_BATCH_IMPL and mode == "stream":
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
    provenance (the `dynamo_v2-<ver>` dir, e.g. 0.1.11), NOT the live
    parsers/v2/Cargo.toml.

    Sourcing from the fixtures keeps every "Dynamo v2 Rust … (stream)" label on the page
    consistent (the stream-tab candidates already read the dir version) and matching the
    captured data. Reading the live crate makes the label drift ahead — the page would
    show 0.1.16 in one place and the real captured 0.1.11 in another the moment the crate
    is bumped before a re-capture/republish."""
    # Versions are ascending; the LATEST capture is the current v2 parser build.
    vs = _stream_impl_versions().get(BASELINE_STREAM_IMPL, [])
    return vs[-1] if vs else None


def _v2_display_version(impl: str) -> str | None:
    """Display version for a v2-tab candidate: Dynamo -> the v2 crate version;
    peers -> the engine version they were captured against."""
    if impl == BASELINE_STREAM_IMPL:
        return _dynamo_v2_version()
    return _clean_version((_CAPTURED_WITH_BY_MODE.get("streamv2") or {}).get(impl))


def _clean_version(v: object) -> str | None:
    """Pull a display version from a captured_with value: 'v0.23.0 <sha>' -> '0.23.0',
    '0.5.12.post1' -> '0.5.12.post1', 'Dynamo parser v2' -> None (no numeric version)."""
    if not v:
        return None
    token = str(v).split()[0].lstrip("v")
    return token if re.match(r"\d", token) else None


def _candidate_items() -> list[dict[str, str]]:
    """Ordered comparison candidates for the batch tab: Dynamo, then vLLM/SGLang —
    within each engine versions run LATEST-FIRST (0.24.0 before 0.23.0). Each:
    {key, impl, version, slug, label, short, default_bucket}.

    Default layout: A (reference) = the first candidate (Dynamo's latest);
    everything else starts UNSELECTED (C) — the reader opts into comparisons."""
    impl_versions = _batch_impl_versions()
    out: list[dict[str, str]] = []
    first = True
    for canon in ("dynamo_v1", "vllm_python", "sglang_python"):
        for v in reversed(impl_versions.get(canon, [])):
            slug = toolcalling_table._version_slug(v)
            if first:
                bucket = "A"
                first = False
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
# Read from the fixture extraction cache (loose YAMLs aren't in the repo; the LFS
# tarballs under conformance/fixtures/ are extracted there on first use);
# _common.sh exports CONFORMANCE_FIXTURES_ROOT. Without this the stream tab's versioned
# candidates come up empty and the Base/Compare parser selector doesn't render.
_STREAM_SRC = (
    toolcalling_table._fixtures_cache_root() / "toolcalling/fixtures-stream-v2"
)


_PATCH_SUFFIX_RE = re.compile(r"\.patch\d+$")


def _base_stream_version(ver: str) -> str:
    """A `X.patchN` capture is the SAME parser binary re-run to backfill newer
    cases onto version `X` (e.g. 0.1.11.patch1 = the 0.1.11 binary on streamv2.5.h).
    It folds onto `X` for display — it is not a standalone candidate version. The
    on-disk shard stays separate so the pristine `X` capture is never rewritten;
    the resolver folds the overlay because it sorts equal to `X`."""
    return _PATCH_SUFFIX_RE.sub("", ver)


def _stream_impl_versions() -> dict[str, list[str]]:
    """{stream_impl: versions ascending} discovered from the fixtures-stream-v2/
    <impl>-<version>/ dirs (no hardcoded anchor — the baseline is whichever version is
    lowest). Ordered dynamo_v1, dynamo_v2, vllm_rust, vllm_python, sglang_python (canonical
    stream column order)."""
    found: dict[str, list[str]] = {}
    if _STREAM_SRC.is_dir():
        for d in _STREAM_SRC.iterdir():
            if not d.is_dir() or d.name == "inputs" or "-" not in d.name:
                continue
            impl, ver = d.name.split("-", 1)
            # `.patchN` overlays are NOT standalone candidates — they fold onto their
            # base version (the resolver merges them since they sort equal). Collapse
            # to the base so only real versions become compare columns.
            found.setdefault(impl, []).append(_base_stream_version(ver))
    for impl in list(found):
        found[impl] = sorted(set(found[impl]), key=toolcalling_table._version_sort_key)
    order = ("dynamo_v1", "dynamo_v2", "vllm_rust", "vllm_python", "sglang_python")
    return {i: found[i] for i in order if i in found}


def _stream_candidate_items() -> list[dict[str, str]]:
    """Versioned comparison candidates for the stream tab. Keyed <impl>-<slug> like
    the batch tab. Default layout: A (reference) = the LATEST Dynamo v2 stream
    capture (this is v2's tab); everything else — the v1 jail reference, the
    peers, and older versions — starts UNSELECTED (C), the reader opts in."""
    impl_versions = _stream_impl_versions()
    latest = {i: (vs[-1] if vs else None) for i, vs in impl_versions.items()}
    out: list[dict[str, str]] = []
    for impl in ("dynamo_v1", "dynamo_v2", "vllm_rust", "vllm_python", "sglang_python"):
        # Within an engine, versions run LATEST-FIRST (0.24.0 before 0.23.0).
        for v in reversed(impl_versions.get(impl, [])):
            slug = toolcalling_table._version_slug(v)
            if impl == BASELINE_STREAM_IMPL and v == latest.get(impl):
                bucket = "A"
            else:
                bucket = "C"
            out.append({
                "key": f"{impl}-{slug}",
                "label": _full_label(impl, v, "stream"),
                "default_bucket": bucket,
            })
    return out


@functools.lru_cache(maxsize=1)
def _stream_divergence_notes() -> dict:
    """Hand-maintained sidecar notes for known Dynamo divergences
    (conformance/toolcalling/known-divergences.yaml,
    read from the REAL repo root via FRONTEND_CRATES_ROOT — this module runs
    from the staged copy). Applied at render time so the notes survive capture
    re-records. {family: {case_id: {"v2"|"jail": note}}}."""
    root = os.environ.get("FRONTEND_CRATES_ROOT")
    if not root:
        return {}
    path = Path(root) / "conformance/toolcalling/known-divergences.yaml"
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def _known_divergence_note(family: str, case_id: str, key: str) -> str | None:
    """One note from known-divergences.yaml: `v2`/`jail` for the stream-tab
    Dynamo candidates, `stream_vs_batch` for the batch-tab v1-vs-v2 allowlist."""
    return ((_stream_divergence_notes().get(family) or {}).get(case_id) or {}).get(key)


def _stream_divergence_note(family: str, case_id: str, impl: str) -> str | None:
    """The sidecar note for one (family, case, dynamo candidate) — `v2` = the
    dynamo_v2 stream parser, `jail` = the dynamo_v1 jail+batch reference."""
    gen = "v2" if impl == BASELINE_STREAM_IMPL else "jail"
    return _known_divergence_note(family, case_id, gen)


def _stream_version_families(impl: str, version: str) -> set[str] | None:
    """Families the `<impl>-<version>` stream fixture dir actually holds — the
    authoritative coverage for that parser build. `None` if the dir is absent (don't
    gate). Used to mark the Dynamo v2 stream candidate `na` on families its parser
    doesn't implement, since the dir only contains the families it produced output
    for (dynamo_v2-0.1.11 = the v2-supported handful; dynamo_v1-3.0.0 = all)."""
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
    exists). Coverage is the authoritative `dynamo_v2-<ver>` fixture dir family list."""
    v2ver = _dynamo_v2_version()
    if not v2ver:
        return {}
    fams = sorted(_stream_version_families(BASELINE_STREAM_IMPL, v2ver) or [])
    if not fams:
        return {}
    slug = toolcalling_table._version_slug(v2ver)
    entry = {"label": _full_label(BASELINE_STREAM_IMPL, v2ver, "stream"), "families": fams}
    # The v2 candidate key differs by tab: "<impl>-s-<slug>" on the batch
    # (stream-on-batch) tab, bare "<impl>-<slug>" on the stream tab.
    return {
        f"{BASELINE_STREAM_IMPL}-s-{slug}": entry,
        f"{BASELINE_STREAM_IMPL}-{slug}": entry,
    }


def _stream_version_status_map() -> dict[tuple[str, str], dict[str, dict[str, dict]]]:
    """{(family, sub): {impl: {slug: {block, version, status}}}} for the stream tab.

    Resolve each versioned peer @ each of its versions (others pinned) and re-run
    load_all_cases("streamv2") so keys match the rendered table (same assembly +
    split-parent normalization). Single-version impls (dynamo_v2, vllm_rust) are
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
        # Dynamo v1 and v2 are DIFFERENT parsers: v2 (dynamo_v2-0.1.11)
        # implements only a handful of families, while the v1 jail
        # (dynamo_v1-3.0.0) covers all. The stream assembly defaults an absent
        # impl to an empty-but-present block, which would paint the v2 parser green on
        # families it doesn't implement. Gate on the version dir's actual family list
        # so uncovered families read `na` (not implemented), not a clean empty output.
        covered = _stream_version_families(impl, version) if impl in BASELINE_IMPLS else None
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
            # Attach the hand-maintained sidecar note for a KNOWN v2-vs-jail
            # divergence to this candidate's block, so the popup explains the Δ
            # and the cell drops its `?` research-needed suffix.
            if impl in (BASELINE_BATCH_IMPL, BASELINE_STREAM_IMPL) and isinstance(
                block, dict
            ):
                note = _stream_divergence_note(
                    case.get("__family") or key[0], case.get("__case_id") or "", impl
                )
                if note:
                    block = {**block, "explanation": note}
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


# Comparison-signature semantics moved to markers.py (DIS-2434): single-sourced with
# the structured comparison model the JS view consumes. Re-exported here so existing
# callers and tests keep referring to them as module attributes.
_canon_call_for_sig = markers._canon_call_for_sig
_candidate_sig = markers.candidate_sig


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
        out[BASELINE_STREAM_IMPL] = dynv
    for fp in sorted(STREAM_ON_BATCH_FIXTURES.glob("*/TOOLCALLING.batch*.yaml")):
        doc = yaml.safe_load(fp.read_text()) or {}
        for impl, ver in (doc.get("captured_with") or {}).items():
            if impl == BASELINE_STREAM_IMPL or impl not in STREAM_IMPL_KEYS:
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
        for impl in ("dynamo_v1", "vllm_python", "sglang_python"):
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
            if sob.get("__known_divergence"):
                case["__known_divergence"] = True
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
    dynamo = _impl_get(overlay_case, BASELINE_STREAM_IMPL)
    if isinstance(dynamo, dict) and ("calls" in dynamo or "normal_text" in dynamo):
        expected[BASELINE_STREAM_IMPL] = {
            "calls": dynamo.get("calls") or [],
            "normal_text": dynamo.get("normal_text") or "",
        }
    elif not has_batch_text:
        expected[BASELINE_STREAM_IMPL] = {"unavailable": "No batch model_text for this case."}
    else:
        expected[BASELINE_STREAM_IMPL] = {
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
        # A documented v1-batch vs v2-stream divergence (the batch-via-stream
        # parity allowlist): note the v2 block (popup `explanation:`) and flag
        # the case so the cell renders the `≠` known-divergence suffix. The
        # generator promotes a bare parent id to `.a` — fall back like the
        # overlay lookup above so `…batch.13` matches the promoted `…batch.13.a`.
        note = _known_divergence_note(family, cid, "stream_vs_batch")
        if note is None and cid.endswith(".a"):
            note = _known_divergence_note(family, cid[:-2], "stream_vs_batch")
        if note:
            blk = cases[(family, sub)]["expected"].get(BASELINE_STREAM_IMPL)
            if isinstance(blk, dict) and "unavailable" not in blk:
                blk["explanation"] = note
            cases[(family, sub)]["__known_divergence"] = True
    return cases


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


def _load_panel_cases(
    mode: str, family_filter: str | None = None
) -> dict[str, object]:
    """Load + augment the cases for one toolcalling tab. Shared by the HTML renderer
    (`_load_html_panel`) and the JSON model builder so both see the identical
    ver_status / merged-cmp attachment and display grouping."""
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
    sub_cases = _discover_sub_cases(mode, cases)
    no_vllm, no_sglang = _derive_no_peer_sets(cases)
    top_n, others = _build_display_groups(cases, labels)
    # The streamv2 tab uses the stream comparison: color = stream-vs-own-batch,
    # conformance marker = cross-engine stream agreement (`Y_s`).
    comparison = "stream_vs_batch" if mode == "streamv2" else "cross_engine"
    return {
        "mode": mode,
        "cases": cases,
        "labels": labels,
        "sub_cases": sub_cases,
        "no_vllm": no_vllm,
        "no_sglang": no_sglang,
        "top_n": top_n,
        "others": others,
        "comparison": comparison,
        "has_cases": bool(cases),
    }




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


# ===== Structured JSON model builders (DIS-2434) ================================
# Python computes the model; the JS view renders it. The comparison SEMANTICS stay in
# markers.py (cmp_model / comparison_facts / _overview_status / _sob_status); these
# functions orchestrate them into the schema documented in model.py. No verdict logic
# is reimplemented in JS.
import model as _model  # noqa: E402  (schema + serialization; leaf module)

_MODE_PAREN_RE = re.compile(r"\(([^)]*)\)\s*$")
_LABEL_VERSION_RE = re.compile(r"(\d[\w.]*)\s*\([^)]*\)\s*$")


def _cand_engine_group(key: str) -> str:
    for prefix in ("dynamo", "vllm", "sglang"):
        if key.startswith(prefix):
            return prefix
    return key.split("-")[0]


def _parse_mode_of_label(label: str) -> str | None:
    m = _MODE_PAREN_RE.search(label)
    if not m:
        return None
    inner = m.group(1)
    if "stream" in inner:
        return "stream"
    if "batch" in inner:  # includes "jail+batch"
        return "batch"
    return None


def _version_of_label(label: str) -> str | None:
    m = _LABEL_VERSION_RE.search(label)
    return m.group(1) if m else None


def _candidate_model(items: list[dict]) -> list[dict]:
    """Normalize a compare-bar candidate list to the model schema. The version is taken
    from the source item when present, else parsed from the label ("… 3.0.0 (batch)")
    so every candidate carries a `version` the view + guards can rely on."""
    out = []
    for it in items:
        key = it["key"]
        label = it["label"]
        out.append({
            "key": key,
            "impl": _cand_engine_group(key),
            "label": label,
            "label_html": _candidate_label_html(label),
            "default_bucket": it.get("default_bucket", "C"),
            "version": it.get("version") or _version_of_label(label),
            "parse_mode": _parse_mode_of_label(label),
        })
    return out


def _columns_model(mode: str, sub_cases: list[str]) -> tuple[list[dict], list[dict]]:
    descriptions = _parse_subcase_descriptions(mode)
    groups: list[dict] = []
    cols: list[dict] = []
    for run in _subcase_runs(mode, sub_cases):
        col_group = _subcase_group_key(mode, run[0])
        groups.append({
            "key": col_group,
            "label": _subcase_group_label(mode, run[0]),
            "band": _subcase_band_class(mode, run[0]),
            "span": len(run),
        })
        for sub in run:
            cols.append({
                "sub": sub,
                "group_key": col_group,
                "band": _subcase_band_class(mode, sub),
                "label": sub,
                "desc": descriptions.get(sub) or descriptions.get(sub.split(".")[0]) or "",
            })
    return groups, cols


def _output_block_model(blk: object) -> dict | None:
    """A candidate's expected output block reduced to the model's raw fields."""
    if not isinstance(blk, dict):
        return None
    out: dict[str, Any] = {}
    if "unavailable" in blk:
        out["unavailable"] = blk["unavailable"]
    if "error" in blk:
        out["error"] = blk["error"]
    if "calls" in blk or "normal_text" in blk:
        out["calls"] = blk.get("calls") or []
        out["normal_text"] = blk.get("normal_text") or ""
    expl = _explanation(blk)
    if expl:
        out["explanation"] = expl
    return out or None


def _cell_candidate_meta(case: dict, output_kind: str) -> tuple[dict, list[dict]]:
    """Reproduce render_cell_html's candidate selection as STRUCTURED data: return
    (cmp_blocks, cand_meta). `cmp_blocks` = {cand_key: raw_block} for markers.cmp_model;
    `cand_meta` is the ordered per-candidate tooltip list {key,label,impl,version,block,
    leak}. Mirrors the __cmp / __ver_status / per-impl branches exactly so the compare
    keys line up with the compare-bar candidates and the JS cand-<key> sections."""
    meta: list[dict] = []
    cmp_items = case.get("__cmp")
    ver_status = case.get("__ver_status")
    if cmp_items:
        for item in cmp_items:
            meta.append({"key": item["key"], "label": item["label"],
                         "version": None, "block_raw": item["block"]})
    elif ver_status:
        # The mode is the TAB's, not a constant: this branch also serves the streamv2
        # tab, where dynamo_v2 and vllm_rust are stream-only impls (impls.py IMPL_SPECS)
        # and can never be "(batch)". Hardcoding it made the tooltip header contradict
        # the compare bar, which builds the same candidates via _stream_candidate_items.
        # _full_label still maps dynamo_v1 on stream data to "(jail+batch)".
        for impl in ("dynamo_v1", "dynamo_v2", "vllm_rust", "vllm_python", "sglang_python"):
            for slug, info in (ver_status.get(impl) or {}).items():
                meta.append({"key": f"{impl}-{slug}",
                             "label": _full_label(impl, info["version"], output_kind),
                             "version": info["version"], "block_raw": info["block"]})
    else:
        expected = _expected(case)
        for impl in STREAM_IMPL_KEYS:
            meta.append({"key": impl, "label": f"{_IMPL_DISPLAY[impl]} {output_kind}",
                         "version": _v2_display_version(impl), "block_raw": _impl_get(expected, impl)})
    cmp_blocks = {m["key"]: m["block_raw"] for m in meta}
    for m in meta:
        blk = m.pop("block_raw")
        m["impl"] = _cand_engine_group(m["key"])
        m["parse_mode"] = _parse_mode_of_label(m["label"])
        m["leak"] = isinstance(blk, dict) and _block_tool_call_leaks(blk)
        m["block"] = _output_block_model(blk)
    return cmp_blocks, meta


def _chunk_model(chunk: dict) -> dict:
    return {
        "delta_text": chunk.get("delta_text", ""),
        "delta_token_ids": chunk.get("delta_token_ids"),
        "finish_reason": chunk.get("finish_reason"),
        # Per-impl streamed deltas + residual normal_text drive the per-chunk chart.
        "expected": _normalize_impl_mapping(chunk.get("expected") or {}),
        "normal_text": (
            _normalize_impl_mapping(chunk["normal_text"])
            if isinstance(chunk.get("normal_text"), dict)
            else chunk.get("normal_text")
        ),
    }


def _input_model(case: dict) -> dict:
    family = case.get("__family")
    chunks = case.get("chunks")
    if isinstance(chunks, list) and chunks:
        return {"kind": "chunks", "text": None, "family": family,
                "chunks": [_chunk_model(c) for c in chunks if isinstance(c, dict)]}
    model_text = case.get("model_text")
    # An EMPTY model_text is still a text input, not a missing one — dropping it to
    # `kind: None` made `TOOLCALLING.batch.9.a` ("Empty model text") indistinguishable
    # from a case with no fixture, so the grammar popup reported "no input recorded"
    # for every family instead of "empty input".
    if isinstance(model_text, str):
        return {"kind": "text", "text": model_text, "chunks": None, "family": family}
    return {"kind": None, "text": None, "chunks": None, "family": family}


def _toolcalling_tooltip_model(case: dict, output_kind: str, cand_meta: list[dict],
                               dyn: object) -> dict:
    family = case.get("__family")
    case_id = case.get("__case_id", "")
    head = f"{case_id} — {family}" if (case_id and family) else (case_id or str(family or ""))
    impl_keys = _impl_keys_for_output_kind(output_kind)
    baseline = baseline_impl(impl_keys)
    reasons = [
        {"impl": f["impl"], "label": _IMPL_DISPLAY.get(f["impl"], f["impl"]),
         "reason": f["reason"], "intentional": f["intentional"]}
        for f in markers.comparison_facts(case, impl_keys, baseline)
        if f["impl"] != baseline and f["agrees"] is False and f["reason"]
    ]
    dyn_leak = _dynamo_tool_call_leak(dyn) if isinstance(dyn, dict) else None
    baseline_block = None
    dyn_batch = _impl_get(case.get("batch_expected") or {}, BASELINE_BATCH_IMPL)
    if isinstance(dyn_batch, dict) and ("calls" in dyn_batch or "normal_text" in dyn_batch):
        baseline_block = {"impl": BASELINE_BATCH_IMPL,
                          "label": f"{_IMPL_DISPLAY[BASELINE_BATCH_IMPL]} batch parser",
                          "block": _output_block_model(dyn_batch)}
    return {
        "head": head,
        "description": case.get("description") or "",
        "input": _input_model(case),
        "candidates": cand_meta,
        "baseline": baseline_block,
        "reasons": reasons,
        "dynamo_notes": [[lbl, txt] for lbl, txt in _dynamo_note_sections(case)],
        "refs": [[lbl, val] for lbl, val in (("Ref", case.get("ref")),
                                             ("Spec ref", case.get("spec_ref"))) if val],
        "leak_note": str(dyn_leak) if dyn_leak else None,
        "na_note": None,
    }


def _toolcalling_cell_model(case: dict | None, mode: str, family: str, sub: str,
                            output_kind: str, comparison: str,
                            href_rewrite) -> dict:
    col_group = _subcase_group_key(mode, sub)
    band = _subcase_band_class(mode, sub)
    if case is None:
        return _model.missing_cell(sub, family, col_group, band,
                                   head=f"TOOLCALLING.{mode}.{sub}")
    impl_keys = _impl_keys_for_output_kind(output_kind)
    baseline = baseline_impl(impl_keys)
    sob = comparison == "stream_vs_batch"
    cmp_blocks, cand_meta = _cell_candidate_meta(case, output_kind)
    cmp = markers.cmp_model(cmp_blocks) if cmp_blocks else None
    facts = markers.comparison_facts(case, impl_keys, baseline)
    status = _sob_status(case, BASELINE_STREAM_IMPL) if sob else _overview_status(case, baseline)
    dyn = _impl_get(case.get("expected") or {}, baseline)
    fp = case.get("__fixture_path", "")
    href = href_rewrite(common.fixture_href(fp)) if fp else None
    if not isinstance(dyn, dict):
        # n/a stub: case has only `explanation:` (no `expected:` block).
        tooltip = {"head": f"{case.get('__case_id','')} — {family}",
                   "description": case.get("description") or "",
                   "input": _input_model(case), "candidates": [], "baseline": None,
                   "reasons": [], "dynamo_notes": [], "refs": [], "leak_note": None,
                   "na_note": _explanation(case)}
        return _model.make_cell(kind="cell", case_id=case.get("__case_id"), family=family,
                                sub=sub, col_group=col_group, band=band, fixture_href=href,
                                status=status, cmp=cmp, facts=facts, tooltip=tooltip,
                                known_divergence=bool(case.get("__known_divergence")))
    tooltip = _toolcalling_tooltip_model(case, output_kind, cand_meta, dyn)
    return _model.make_cell(kind="cell", case_id=case.get("__case_id"), family=family,
                            sub=sub, col_group=col_group, band=band, fixture_href=href,
                            status=status, cmp=cmp, facts=facts, tooltip=tooltip,
                            known_divergence=bool(case.get("__known_divergence")))


def _toolcalling_tab_model(spec: dict, href_rewrite, parser_stream_context: str) -> dict:
    mode = spec["mode"]
    cases = spec["cases"]
    sub_cases = spec["sub_cases"]
    comparison = spec["comparison"]
    no_vllm, no_sglang = spec["no_vllm"], spec["no_sglang"]
    output_kind = "batch" if parser_stream_context == "batch" else "stream"
    cell_text = (
        (lambda case: _sob_cell_text(case, parser_stream_context))
        if comparison == "stream_vs_batch"
        else cell_for
    )
    refs = _build_family_to_rust_ref()
    inheritance = _build_family_inheritance(refs)
    column_groups, cols = _columns_model(mode, sub_cases)

    def row_model(model_label: str, family: str) -> dict:
        all_todo = sub_cases and all(
            cell_text(cases.get((family, sub))) == "…" for sub in sub_cases
        )
        peer_output_exists = any(
            _has_peer_output(cases.get((family, sub))) for sub in sub_cases
        )
        cells: dict[str, dict] = {}
        for sub in sub_cases:
            if all_todo and not peer_output_exists:
                cells[sub] = _model.blank_cell(sub, _subcase_group_key(mode, sub),
                                               _subcase_band_class(mode, sub))
            else:
                cells[sub] = _toolcalling_cell_model(
                    cases.get((family, sub)), mode, family, sub, output_kind,
                    comparison, href_rewrite,
                )
        return {
            "section": None,
            "model_label": model_label,
            "model_label_html": _model_label_html(model_label),
            "family": family,
            "parser": {"html": href_rewrite(_parser_cell_html(
                family, refs, no_vllm, no_sglang, inheritance,
                stream_context=parser_stream_context))},
            "cells": cells,
        }

    rows: list[dict] = []
    if spec["top_n"]:
        rows.append({"section": "Top-N models", "model_label": "Top-N models",
                     "model_label_html": "", "family": None, "parser": None, "cells": {}})
        rows.extend(row_model(m, f) for m, f in spec["top_n"])
    if spec["others"]:
        rows.append({"section": "Others", "model_label": "Others",
                     "model_label_html": "", "family": None, "parser": None, "cells": {}})
        rows.extend(row_model(m, f) for m, f in spec["others"])

    all_families = [f for _, f in spec["top_n"]] + [f for _, f in spec["others"]]
    stats = _compute_stats(cases, sub_cases, all_families, cell_text=cell_text)
    return {
        "id": f"tab-{mode}",
        "kind": "toolcalling",
        "mode": mode,
        "column_groups": column_groups,
        "columns": cols,
        "rows": rows,
        "stats": stats,
        "glossary": _glossary_groups(mode, _parse_subcase_descriptions(mode), sub_cases),
    }


def _fixture_href_rewriter(stage_dir: str, fixture_href_root: str):
    def rewrite(text: str) -> str:
        return text.replace(
            f'href="{stage_dir}/fixtures/', f'href="{fixture_href_root}'
        ).replace('href="fixtures/', f'href="{fixture_href_root}')
    return rewrite


def _plain_href_rewriter(stage_dir: str, fixture_href_root: str):
    """A bare fixture href (no href="...") rewriter for cell links."""
    def rewrite(href: str | None) -> str | None:
        if not href:
            return href
        return href.replace(f"{stage_dir}/fixtures/", fixture_href_root).replace(
            "fixtures/", fixture_href_root, 1
        ) if href.startswith((f"{stage_dir}/fixtures/", "fixtures/")) else href
    return rewrite


_UNIFIED_LEAK_MARKERS = ("<|", "|>", "<think>", "</think>", "◁", "<channel", "channel|>")

# Per-family markers whose presence in a visible payload (text/reasoning) is a leak
# but which the shared markers above miss. gemma4's channel opener leaves `thought\n`;
# qwen3's tool envelope has NO `<|...|>` sentinels, so a `<tool_call>...</tool_call>`
# leaking into reasoning_content is invisible to the shared list — enumerate it
# explicitly (kimi's tool/section markers already contain `<|`/`|>`).
_UNIFIED_FAMILY_LEAK_MARKERS = {
    "gemma4": ("thought\n",),
    "qwen3": ("<tool_call>", "</tool_call>", "<function=", "</function>",
              "<parameter=", "</parameter>"),
}


def _unified_classify(family: str, golden: list, got: list) -> str:
    """Classify a captured event list against the golden oracle. Python port of
    the Rust `classify` in tests/unified_render.rs so vLLM (captured) and Dynamo
    (LIVE-in-Rust) are scored the same way against GOLDEN."""
    if golden == got:
        return "MATCH"

    # Shared markers + any family-specific markup that leaks invisibly to them (e.g.
    # gemma4 `thought\n`, qwen3 `<tool_call>`/`<function=`/`<parameter=`).
    markers = _UNIFIED_LEAK_MARKERS + _UNIFIED_FAMILY_LEAK_MARKERS.get(family, ())

    def _leaks(evs):
        return any(e.get("kind") in ("text", "reasoning")
                   and any(m in (e.get("text") or "") for m in markers)
                   for e in evs)

    if _leaks(got):
        return "LEAK"

    def _rcount(evs):
        return sum(1 for e in evs if e.get("kind") == "reasoning")

    if _rcount(got) < _rcount(golden):
        return "MERGE"

    def _calls(evs):
        return [(e.get("name"), json.dumps(e.get("arguments"), sort_keys=True))
                for e in evs if e.get("kind") == "tool_call"]

    gc, tc = _calls(golden), _calls(got)
    if (len(gc) == len(tc) and all(a[0] == b[0] for a, b in zip(gc, tc))
            and any(a[1] != b[1] for a, b in zip(gc, tc))):
        return "ARG_MISMATCH"

    def _cat(evs, want_reason):
        kind = "reasoning" if want_reason else "text"
        return "".join((e.get("text") or "") for e in evs if e.get("kind") == kind)

    if _cat(golden, True) == _cat(got, True) and _cat(golden, False) == _cat(got, False):
        return "ORDER"
    return "LOSS"


def _assemble_stream(chunk_deltas: list) -> list:
    """Assemble the FINAL ordered event list from a parser's per-chunk STREAMED deltas
    (not its batch final message). Coalesces consecutive reasoning/text runs and joins
    per-call tool-argument fragments (a delta with a name starts a new call; nameless
    arg deltas append to the current one). This is what a streaming client actually
    receives — and, unlike the batch assembly, it preserves reasoning<->tool order."""
    events: list = []
    cur_tool = None  # index in `events` of the tool call currently receiving arg deltas
    for deltas in chunk_deltas:
        for dl in deltas or []:
            k = dl.get("kind")
            if k in ("reasoning", "text"):
                cur_tool = None
                if events and events[-1]["kind"] == k:
                    events[-1]["text"] += dl.get("text") or ""
                else:
                    events.append({"kind": k, "text": dl.get("text") or ""})
            elif k == "tool_call":
                name, args = dl.get("name"), dl.get("arguments")
                if name:  # a name delta opens a new call
                    events.append({"kind": "tool_call", "name": name, "_raw": args or ""})
                    cur_tool = len(events) - 1
                elif cur_tool is not None and args:
                    events[cur_tool]["_raw"] += args
    for e in events:
        if e["kind"] == "tool_call":
            raw = e.pop("_raw", "")
            if isinstance(raw, (dict, list)):
                e["arguments"] = raw
            elif not (raw or "").strip():
                e["arguments"] = {}
            else:
                try:
                    e["arguments"] = json.loads(raw)
                except (ValueError, TypeError):
                    e["arguments"] = raw
    return events


def _unified_base(artifact_root: Path) -> Path:
    """Where the unified capture YAMLs live. In a packaged render they come from the
    extracted fixture snapshot (CONFORMANCE_FIXTURES_ROOT/unified — the
    conformance/fixtures/unified/captures.tar.gz shard); locally after a harness run
    they sit in the loose build tree conformance/unified/."""
    snap = os.environ.get("CONFORMANCE_FIXTURES_ROOT")
    if snap and (Path(snap) / "unified").is_dir():
        return Path(snap) / "unified"
    return artifact_root / "conformance/unified"


def _load_capture(artifact_root: Path, name: str, version_key: str) -> tuple[dict, str | None]:
    """Load a persisted LIVE capture artifact. Returns (results_by_id, version).
    Empty if absent."""
    cp = _unified_base(artifact_root) / name
    if not cp.exists():
        return {}, None
    data = yaml.safe_load(cp.read_text())
    return data.get("results", {}), data.get(version_key)


def _load_vllm_capture(artifact_root: Path) -> tuple[dict, str | None]:
    """vLLM Python parser capture (capture_vllm_unified.py, ParserManager combined path)."""
    return _load_capture(artifact_root, "vllm_capture.yaml", "vllm_version")


def _load_vllm_rust_capture(artifact_root: Path) -> tuple[dict, str | None]:
    """vLLM Rust unified capture (capture_vllm_rust_unified.py). gemma4 = native
    Gemma4UnifiedParser; other families = CombinedParser. Each result carries a
    `parser` string ("vLLM Rust (UnifiedParser)" / "(CombinedParser)")."""
    return _load_capture(artifact_root, "vllm_rust_capture.yaml", "vllm_rust_version")


def _load_sglang_capture(artifact_root: Path) -> tuple[dict, str | None]:
    """SGLang Python capture (capture_sglang_unified.py). SGLang has no unified parser —
    always a reasoning detector then a tool detector (Combined/split)."""
    return _load_capture(artifact_root, "sglang_capture.yaml", "sglang_version")


def _load_unified_fixtures(base: Path):
    """Read the exploded per-case / per-family / per-version unified fixtures (same
    layout as toolcalling/fixtures-stream-v2: inputs/ + golden/ + <impl>-<version>/)
    and reconstruct the (cases feed, per-engine caps, versions) the tab model expects.
    Returns None if the tree isn't present."""
    if not (base / "inputs").is_dir():
        return None

    def _read_dir(name):
        out = {}  # (family, case_key) -> case_doc
        for fp in sorted((base / name).glob("*/*.yaml")):
            doc = yaml.safe_load(fp.read_text()) or {}
            for k, cd in (doc.get("cases") or {}).items():
                out[(fp.parent.name, k)] = cd
        return out

    inputs = _read_dir("inputs")
    golden = _read_dir("golden")
    engine_dirs = {}  # impl -> (dirname, version)
    for d in sorted(base.iterdir()):
        if not d.is_dir() or d.name in ("inputs", "golden"):
            continue
        m = re.match(r"^([a-z0-9_]+)-(\d.*)$", d.name)
        if m:
            engine_dirs[m.group(1)] = (d.name, m.group(2))
    engine_cases = {impl: _read_dir(dirname) for impl, (dirname, _v) in engine_dirs.items()}

    cases = []
    caps = {"vllm_python": {}, "vllm_rust": {}, "sglang_python": {}}
    for (fam, key), inp in sorted(inputs.items()):
        scenario = inp.get("scenario") or key
        cid = f"UNIFIED.{scenario}.{fam}"
        gdoc = golden.get((fam, key), {})
        ddoc = engine_cases.get("dynamo_v2", {}).get((fam, key), {})
        in_chunks = inp.get("chunks") or []
        dyn_chunks = ddoc.get("chunks") or []
        cases.append({
            "id": cid, "scenario": scenario, "family": fam,
            "description": inp.get("description", ""),
            "policy": inp.get("policy") or [], "policy_tags": inp.get("policy") or [],
            "input": inp.get("input", ""),
            "golden": gdoc.get("assembled") or [],
            "dynamo": ddoc.get("assembled") or [],
            "dynamo_verdict": None, "vllm_verdict": None, "vllm_note": None,
            "chunks": [
                {"delta_text": ic.get("delta_text", ""),
                 "dynamo": (dyn_chunks[i].get("expected") if i < len(dyn_chunks) else []) or []}
                for i, ic in enumerate(in_chunks)
            ],
        })
        for impl in ("vllm_python", "vllm_rust", "sglang_python"):
            edoc = engine_cases.get(impl, {}).get((fam, key))
            if edoc is None:
                continue
            if edoc.get("error"):
                caps[impl][cid] = {"error": edoc["error"]}
            else:
                caps[impl][cid] = {
                    "assembled": edoc.get("assembled") or [],
                    "chunks": [c.get("expected") or [] for c in (edoc.get("chunks") or [])],
                    "parser": edoc.get("parser"),
                }
    versions = {impl: v for impl, (_d, v) in engine_dirs.items()}
    return cases, caps, versions


def _unified_tab_model(artifact_root: Path, hrefs: dict) -> dict | None:
    """Build the Unified (reasoning + tools) tab from the versioned fixture shards
    (conformance/fixtures/unified/: inputs + golden + one <impl>-<version> shard per
    engine, same convention as every other tab). Reference = the authored GOLDEN
    oracle; Compare = Dynamo (LIVE) + vLLM/SGLang. A cell is red only when a shown
    parser LEAKED markup; ordering/content divergences show NΔ but stay green."""
    import hashlib
    loaded = _load_unified_fixtures(_unified_base(artifact_root))
    if loaded is None:
        return None
    cases, _caps, _vers = loaded
    if not cases:
        return None

    vllm_cap = _caps["vllm_python"]; vllm_version = _vers.get("vllm_python")
    vllm_live = bool(vllm_cap)
    vllm_ver_label = vllm_version or "0.25.x"

    vrust_cap = _caps["vllm_rust"]; vrust_version = _vers.get("vllm_rust")
    vrust_live = bool(vrust_cap)
    vrust_ver_label = vrust_version or "0.25.x"

    sgl_cap = _caps["sglang_python"]; sgl_version = _vers.get("sglang_python")
    sgl_live = bool(sgl_cap)
    sgl_ver_label = sgl_version or "0.5.x"

    def _sig(events) -> int:
        return int(hashlib.md5(json.dumps(events, sort_keys=True).encode()).hexdigest()[:8], 16)

    scenarios: list[str] = []
    families: list[str] = []
    scn_desc: dict[str, str] = {}
    by_key: dict[tuple[str, str], dict] = {}
    for c in cases:
        s, f = c["scenario"], c["family"]
        if s not in scenarios:
            scenarios.append(s)
            scn_desc[s] = c["description"]
        if f not in families:
            families.append(f)
        by_key[(f, s)] = c

    # Numbered taxonomy + axis labels — single source in unified_taxonomy.py (shared
    # with explode_unified_fixtures.py so the numbering can't drift). See UNIFIED_CASES.md.
    UNIFIED_TAX = unified_taxonomy.UNIFIED_TAX
    UNIFIED_GROUP_LABEL = unified_taxonomy.UNIFIED_GROUP_LABEL

    def _tax(s):
        return UNIFIED_TAX.get(s, (9, s))

    def _band(group_num):
        return "case-band-0" if group_num % 2 == 1 else "case-band-1"

    ordered = sorted(scenarios, key=_tax)
    columns = []
    for s in ordered:
        g, sub = _tax(s)
        columns.append({"sub": s, "group_key": f"unified_g{g}", "band": _band(g),
                        "label": f"{g}.{sub}", "desc": scn_desc.get(s, "")})
    column_groups = []
    seen_groups = []
    for s in ordered:
        g, _sub = _tax(s)
        if g not in seen_groups:
            seen_groups.append(g)
            column_groups.append({"key": f"unified_g{g}",
                                  "label": UNIFIED_GROUP_LABEL.get(g, "Other"),
                                  "band": _band(g),
                                  "span": sum(1 for x in ordered if _tax(x)[0] == g)})

    def _cand(key, label, bucket):
        return {"key": key, "impl": key, "label": label, "label_html": label,
                "default_bucket": bucket, "version": None, "parse_mode": "unified"}
    # Alphabetical by label so non-Reference popup columns sort alphabetically
    # (the selected Reference is pulled to the left by the view).
    # NOT "unified" — vLLM 0.25.x parses only gemma4 with a native UnifiedParser; qwen3
    # and kimi_k2 go through its CombinedParser (split). The column is vLLM's LIVE output
    # however it parses internally.
    # Names follow the convention <Engine> [version] (<parser/mode>). The vLLM Rust
    # column's per-family variant (UnifiedParser for gemma4, CombinedParser otherwise)
    # can't fit one fixed column label, so it's shown per family in the tooltip.
    dynamo_ver_label = _dynamo_v2_version() or "0.1.x"
    dynamo_label = f"Dynamo v2 Rust {dynamo_ver_label} (stream, orig)"
    vllm_label = (f"vLLM Python {vllm_ver_label} (batch, Combined)" if vllm_live
                  else "vLLM Python 0.25.x (expected)")
    vrust_label = f"vLLM Rust {vrust_ver_label} (stream, Combined & Unified)"
    candidates = [
        _cand("dynamo", dynamo_label, "B"),  # Compare-on by default
        _cand("golden", "GOLDEN (oracle)", "A"),  # Reference (default)
        _cand("vllm", vllm_label, "B"),  # Compare-on by default
    ]
    if vrust_live:
        # impl="vllm" groups it under the vLLM engine column of the compare bar (a second
        # row next to vLLM Python); key stays "vllm_rust" for the cmp/chunk/chart lookups.
        rc = _cand("vllm_rust", vrust_label, "B")
        rc["impl"] = "vllm"
        candidates.append(rc)
    if sgl_live:
        candidates.append(_cand("sglang", f"SGLang Python {sgl_ver_label} (stream, Combined)", "B"))
    _TODO = ("TODO: adopt a unified parser for this family (Dynamo v2 is moving to a "
             "per-family mixture — native unified where available, split elsewhere). "
             "Today's split parses ALL reasoning first, so reasoning between/after tool "
             "calls is merged up front and loses its position. One state machine per stream "
             "(owning reasoning+content+tools) fixes this by construction.")

    rows = []
    for f in families:
        cells = {}
        for s in scenarios:
            c = by_key.get((f, s))
            if not c:
                continue
            gold = c["golden"]
            # Assemble every engine's FINAL from its STREAMED per-chunk deltas (not its
            # batch final message). For Dynamo this is decisive: streaming preserves the
            # reasoning<->tool order that the batch assembly (detect_and_parse_reasoning)
            # collapses. Verdict is recomputed against GOLDEN on the streamed assembly.
            dyn_chunk_deltas = [ch.get("dynamo") or [] for ch in (c.get("chunks") or [])]
            dyn = _assemble_stream(dyn_chunk_deltas)
            gsig, dsig = _sig(gold), _sig(dyn)
            dverd = _unified_classify(f, gold, dyn)
            # vLLM: LIVE captured events (batch parse() -> assembled), scored against
            # GOLDEN the same way as Dynamo. Falls back to the documented hypothesis
            # from expect.vllm when no capture is present.
            # vLLM Python's streaming parser (ParserEngine) is TOKEN-ID based — its
            # incremental lexer scans real token IDs, not text — so a text-only stub
            # tokenizer can't faithfully drive its stream (reasoning never surfaces, a
            # capture artifact, not a defect). The other engines are text-based and stub
            # faithfully. So use vLLM Python's faithful BATCH parse() assembled here; its
            # per-chunk stream is omitted (can't be captured without the real tokenizer).
            cap = vllm_cap.get(c["id"]) if vllm_live else None
            vllm_chunks = []
            vllm_events = cap.get("assembled") if cap else None
            if vllm_events is not None:
                vverd = _unified_classify(f, gold, vllm_events)
                vsig = _sig(vllm_events)
            else:
                vverd = c.get("vllm_verdict") or "MATCH"
                vsig = gsig if vverd == "MATCH" else (gsig ^ 0x5A5A5A5A)
            # vLLM Rust (native Gemma4UnifiedParser for gemma4, CombinedParser otherwise).
            rcap = vrust_cap.get(c["id"]) if vrust_live else None
            vrust_chunks = (rcap.get("chunks") if rcap else None) or []
            vrust_events = _assemble_stream(vrust_chunks) if rcap else None
            vrust_parser = (rcap.get("parser") if rcap else None) or "vLLM Rust"
            vrust_err = rcap.get("error") if rcap else None
            if vrust_events is not None:
                rverd = "ERROR" if vrust_err else _unified_classify(f, gold, vrust_events)
                # A hard-error assembles to []; golden can also be [] (partial dropped),
                # so _sig would collide and NΔ would mask the ERROR as a match. XOR a
                # sentinel to force a divergence — the verbatim exception shows in the popup.
                rsig = (_sig(vrust_events) ^ 0xE44) if vrust_err else _sig(vrust_events)
            # SGLang Python (Combined: reasoning detector -> tool detector), streamed.
            scap = sgl_cap.get(c["id"]) if sgl_live else None
            sgl_chunks = (scap.get("chunks") if scap else None) or []
            sgl_events = _assemble_stream(sgl_chunks) if scap else None
            sgl_err = scap.get("error") if scap else None
            if sgl_events is not None:
                sverd = "ERROR" if sgl_err else _unified_classify(f, gold, sgl_events)
                ssig = (_sig(sgl_events) ^ 0xE44) if sgl_err else _sig(sgl_events)
            cmp = {
                "golden": {"sig": gsig, "leak": 0, "na": 0},
                "dynamo": {"sig": dsig, "leak": 1 if dverd == "LEAK" else 0, "na": 0},
                "vllm": {"sig": vsig, "leak": 1 if vverd == "LEAK" else 0, "na": 0},
            }
            if vrust_events is not None:
                cmp["vllm_rust"] = {"sig": rsig, "leak": 1 if rverd == "LEAK" else 0, "na": 0}
            if sgl_events is not None:
                cmp["sglang"] = {"sig": ssig, "leak": 1 if sverd == "LEAK" else 0, "na": 0}
            desc = c["description"]
            if c.get("policy_tags"):
                desc = f"{desc}  [policy: {', '.join(c['policy_tags'])}]"
            chunk_rows = [
                {"delta_text": ch["delta_text"], "finish_reason": None,
                 "expected": {"dynamo": ch.get("dynamo") or [],
                              "vllm": (vllm_chunks[i] if i < len(vllm_chunks) else []),
                              "vllm_rust": (vrust_chunks[i] if i < len(vrust_chunks) else []),
                              "sglang": (sgl_chunks[i] if i < len(sgl_chunks) else [])}}
                for i, ch in enumerate(c.get("chunks") or [])
            ]
            reasons = []
            if dverd in ("MERGE", "ORDER"):
                reasons.append({
                    "label": "stream vs final message",
                    "reason": ("per-chunk deltas stream IN ORDER (see the chunk rows), but Dynamo's "
                               "final message merges all reasoning into one reasoning_content field "
                               "(the assembled row) — losing the reasoning/tool interleaving. The "
                               "UnifiedParser keeps one ordered stream, so the final message preserves order."),
                })
            if vllm_live and vverd != "MATCH":
                reasons.append({
                    "label": f"vLLM {vllm_ver_label} diverges too",
                    "reason": ("vLLM's batch parse() assembled message (captured LIVE) is measured "
                               "against the same GOLDEN oracle and also diverges here — so vLLM is not "
                               "the ground truth. Common vLLM failure modes: batch parse() drops content "
                               "after a tool call (LOSS), merges/leaks reasoning channel markers (MERGE/LEAK), "
                               "or truncates a string arg at a marker-looking substring (ARG_MISMATCH)."),
                })
            g_num, g_sub = _tax(s)
            tooltip = {
                "head": f'UNIFIED.{g_num}.{g_sub} ({s}) — {f}',
                "description": desc,
                "input": {"kind": "chunks", "text": c["input"], "chunks": chunk_rows, "family": f},
                "candidates": [
                    {"key": "dynamo", "label": f"{dynamo_label}; v1 reasoning + v2 tool", "impl": "dynamo",
                     "version": None, "parse_mode": "unified", "leak": dverd == "LEAK",
                     "block": {"events": dyn, "verdict": dverd,
                               "todo": _TODO if dverd != "MATCH" else None}},
                    {"key": "golden", "label": "GOLDEN (oracle)", "impl": "golden",
                     "version": None, "parse_mode": "unified", "leak": False,
                     "pin_first": True,  # oracle is always the leftmost popup column
                     "block": {"events": gold}},
                    {"key": "vllm", "label": vllm_label, "impl": "vllm",
                     "version": vllm_ver_label, "parse_mode": "unified", "leak": vverd == "LEAK",
                     "block": ({"events": vllm_events, "verdict": vverd,
                                "note": c.get("vllm_note")}
                               if vllm_events is not None
                               else {"expected": vverd, "note": c.get("vllm_note")})},
                ] + ([
                    {"key": "vllm_rust",
                     "label": f"vLLM Rust {vrust_ver_label} (stream, {vrust_parser.replace('vLLM Rust ', '').strip('()')})",
                     "impl": "vllm_rust", "version": vrust_ver_label, "parse_mode": "unified",
                     "leak": rverd == "LEAK",
                     "block": ({"error": vrust_err} if vrust_err else
                               {"events": vrust_events, "verdict": rverd})},
                ] if vrust_events is not None else []) + ([
                    {"key": "sglang", "label": f"SGLang Python {sgl_ver_label} (stream, Combined)",
                     "impl": "sglang", "version": sgl_ver_label, "parse_mode": "unified",
                     "leak": sverd == "LEAK",
                     "block": ({"error": sgl_err} if sgl_err else
                               {"events": sgl_events, "verdict": sverd})},
                ] if sgl_events is not None else []),
                "baseline": None, "reasons": reasons, "dynamo_notes": [], "refs": [],
                "leak_note": None, "na_note": None,
            }
            cells[s] = {
                "kind": "cell", "case_id": f"UNIFIED.{g_num}.{g_sub}", "family": f, "sub": s,
                "col_group": f"unified_g{g_num}", "band": _band(g_num),
                "status": "ok", "red_on_leak": True,
                "cmp": cmp, "facts": [], "tooltip": tooltip,
            }
        rows.append({"family": f, "model_label": f, "model_label_html": f, "section": None,
                     "parser": None, "cells": cells})

    total = sum(len(r["cells"]) for r in rows)
    stats = {"families": len(families), "sub_cases": len(scenarios), "slots": total,
             "real": total, "parity": 0, "dynamo_only": 0, "documented": 0,
             "research": 0, "errors": 0, "na": 0, "missing": 0}
    cases_href = str(hrefs.get("reasoning_cases", "#")).replace("REASONING_CASES", "UNIFIED_CASES")

    # "Case descriptions" section under the table, grouped by taxonomy — same shape as
    # every other tab's glossary ([{label, rows:[(short_id, description), ...]}]). The
    # view prepends case_prefix ("UNIFIED."), so short_id is the numbered id (e.g. "1.a").
    unified_glossary = []
    for grp in column_groups:
        gnum = int(grp["key"].removeprefix("unified_g"))
        grp_rows = [(f"{_tax(s)[0]}.{_tax(s)[1]}", scn_desc.get(s, ""))
                    for s in ordered if _tax(s)[0] == gnum]
        unified_glossary.append({"label": grp["label"], "rows": grp_rows})

    return {
        "id": "tab-unified", "kind": "unified", "active": False, "mode": "unified",
        "no_parser_col": True,  # parser variant is already encoded in each engine's name
        "label": "Unified (reasoning + tools)",
        "label_html": 'Unified <span class="tab-sub">(reasoning + tools)</span>',
        "tab_title": ("Unified: one ordered event stream (reasoning + content + tool calls) "
                      "measured against the GOLDEN oracle"),
        "columns": columns, "column_groups": column_groups,
        "candidates": candidates, "rows": rows, "stats": stats, "glossary": unified_glossary,
        "case_prefix": "UNIFIED.", "case_section_id": "unified",
        "case_docs_href": cases_href, "case_docs_label": "lib/parsers/UNIFIED_CASES.md",
        "captured_note": (
            (f"vLLM Python {vllm_ver_label} captured LIVE (ParserManager combined path). "
             if vllm_live else "")
            + (f"vLLM Rust {vrust_ver_label} captured LIVE from the `vllm-parser` crate — "
               "gemma4 via the native Gemma4UnifiedParser, other families via CombinedParser. "
               if vrust_live else "")
            + (f"SGLang Python {sgl_ver_label} captured LIVE (reasoning detector -> tool "
               "detector, Combined)." if sgl_live else "")),
        "toolbar_desc_html": (
            'Reference = <strong>GOLDEN</strong> (authored oracle, best-effort recovery) · '
            'Compare = <strong>Dynamo v2 Rust (orig)</strong> (v1 reasoning over the whole '
            'stream, then the v2 tool parser; no unified parser yet), '
            f'<strong>vLLM Python {vllm_ver_label} (Combined)</strong>'
            + (f', and <strong>vLLM Rust {vrust_ver_label}</strong> (native '
               '<strong>UnifiedParser</strong> for gemma4, <strong>CombinedParser</strong> '
               'otherwise)' if vrust_live else '')
            + '. A cell is red only when a shown parser LEAKED markup; ordering/content '
            'divergences show their NΔ count but stay green. The native gemma4 UnifiedParser '
            '(vLLM Rust) is the only column that reproduces the golden order on the '
            'reasoning↔tool cases.'),
        "details_note_html": None,
    }


def build_combined_model(output_path: Path | None = None,
                         artifact_root: Path | None = None,
                         *, stamp: str, sha: str | None) -> dict:
    """Assemble the whole-page JSON model (both toolcalling tabs + both reasoning
    tabs). Same loaded data + comparison semantics as render_combined_html."""
    artifact_root = (artifact_root or REPO_ROOT).resolve()
    resolved_output_path = _resolve_output_path(
        output_path, artifact_root, "tests/parity/CONFORMANCE.html")
    hrefs = common.set_links(resolved_output_path, artifact_root)

    tabs: list[dict] = []

    # --- Tool Calling (batch data): merged v1-batch + v2-stream-on-batch ---
    batch_spec = _load_panel_cases("batch")
    batch_href = _plain_href_rewriter("toolcalling", hrefs["toolcalling_fixtures"])
    batch_tab = _toolcalling_tab_model(batch_spec, batch_href, parser_stream_context="batch")
    batch_tab.update({
        "id": "tab-toolcalling-batch",
        "label": "Tool Calling (batch data)",
        "label_html": ('Tool Calling <span class="tab-sub">'
                       '(<span class="w-batch">batch</span> data)</span>'),
        "tab_title": ("Tool Calling (batch data): v1 batch parsers plus v2 stream "
                      "parsers on the same v1 batch fixtures"),
        "case_prefix": "TOOLCALLING.batch.",
        "case_section_id": "toolcalling-batch",
        "case_docs_href": hrefs["toolcalling_cases"],
        "case_docs_label": "lib/parsers/TOOLCALLING_CASES.md",
        "candidates": _candidate_model(_merged_candidate_items()),
        "captured_note": _captured_note("batch"),
        "toolbar_desc_html": (
            f'Parsers: <strong>v1</strong> Dynamo-synced batch '
            f'(<a href="{hrefs["toolcalling_src"]}">parsers/src/tool_calling/</a>) '
            f'plus <strong>v2</strong> streaming on the same batch text '
            f'(<a href="{hrefs["streaming_src"]}">parsers_v2/src/tool_calling/*</a>) · '
            f'Input: <strong>v1</strong> batch fixtures '
            f'(<a href="{hrefs["toolcalling_fixtures"]}">conformance/toolcalling/fixtures-batch-v1/</a>).'),
        "details_note_html": None,
    })
    tabs.append(batch_tab)

    # --- Tool Calling (stream data): per-chunk streamv2 ---
    stream_spec = _load_panel_cases("streamv2")
    stream_href = _plain_href_rewriter("toolcalling", hrefs["toolcalling_stream_fixtures"])
    stream_tab = _toolcalling_tab_model(stream_spec, stream_href, parser_stream_context="streamv2")
    stream_tab.update({
        "id": "tab-toolcalling-streamv2",
        "label": "Tool Calling (stream data)",
        "label_html": ('Tool Calling <span class="tab-sub">'
                       '(<span class="w-stream">stream</span> data)</span>'),
        "tab_title": "Tool Calling (stream data): Dynamo parser v2 on v2 stream fixtures",
        "case_prefix": "TOOLCALLING.streamv2.",
        "case_section_id": "toolcalling-streamv2",
        "case_docs_href": hrefs["toolcalling_streaming_cases"],
        "case_docs_label": "lib/parsers/TOOLCALLING_STREAMING_V2_CASES.md",
        "candidates": _candidate_model(_stream_candidate_items()),
        "captured_note": _captured_note("streamv2"),
        "toolbar_desc_html": (
            f'Parser: <strong>v2</strong> Dynamo parser v2 token-incremental streaming '
            f'(<a href="{hrefs["streaming_src"]}">parsers_v2/src/tool_calling/*</a>) · '
            f'Input: <strong>v2</strong> stream fixtures '
            f'(<a href="{hrefs["toolcalling_stream_fixtures"]}">conformance/toolcalling/fixtures-stream-v2/</a>).'),
        "details_note_html": f"<p>{_stream_parity_explainer_html('streamv2')}</p>",
    })
    tabs.append(stream_tab)

    # --- Reasoning tabs (delegated to the v1 reasoning module's model builder) ---
    r_rows, r_columns, r_refs = reasoning_table._load()
    r_no_vllm, r_no_sglang = reasoning_table._derive_no_peer_sets(r_rows)
    reasoning_href = _plain_href_rewriter("reasoning", hrefs["reasoning_fixtures"])
    for rmode in ("batch", "stream"):
        mode_columns = reasoning_table._columns_for_mode(r_columns, rmode)
        rtab = reasoning_table.build_model_panel(
            r_rows, mode_columns, r_refs, r_no_vllm, r_no_sglang, mode=rmode, active=False)
        # Rebase fixture cell links + parser-cell links onto the reasoning root.
        for row in rtab["rows"]:
            if row.get("parser") and row["parser"].get("html"):
                row["parser"]["html"] = _fixture_href_rewriter("reasoning", hrefs["reasoning_fixtures"])(row["parser"]["html"])
            for cell in row.get("cells", {}).values():
                if cell.get("fixture_href"):
                    cell["fixture_href"] = reasoning_href(cell["fixture_href"])
        _r_label, _r_label_html = _tab_label("Reasoning", rmode, None, False, on_parser=False)
        rtab.update({
            "id": f"tab-reasoning-{rmode}",
            "label": _r_label,
            "label_html": _r_label_html,
            "tab_title": f"Reasoning {rmode}: v1 code on v1 fixtures",
            "case_prefix": "REASONING.",
            "case_section_id": f"reasoning-{rmode}",
            "case_docs_href": hrefs["reasoning_cases"],
            "case_docs_label": "lib/parsers/REASONING_CASES.md",
            "candidates": _candidate_model(rtab.get("candidates", [])),
            "captured_note": "",
            "toolbar_desc_html": (
                f'Parser: <strong>v1</strong> Dynamo-synced reasoning parser '
                f'(<a href="{hrefs["reasoning_src"]}">parsers/v1/src/reasoning/</a>) · '
                f'Input: <strong>v1</strong> reasoning fixtures '
                f'(<a href="{hrefs["reasoning_fixtures"]}">conformance/reasoning/fixtures/</a>).'),
            "details_note_html": None,
        })
        tabs.append(rtab)

    # --- Unified (reasoning + tools) tab: golden oracle vs Dynamo (LIVE) + vLLM ---
    unified_tab = _unified_tab_model(artifact_root, hrefs)
    if unified_tab:
        tabs.append(unified_tab)

    if tabs:
        tabs[0]["active"] = True

    meta = {
        "title": "Dynamo Parser v2 Conformance Table",
        "stamp": stamp,
        "sha": sha,
        "short_sha": sha[:12] if sha else "",
        "command": "conformance/utils/render_table_v2.sh",
        "output": _display_path(resolved_output_path, artifact_root),
        "generated_by": "generate_conformance_table.build_combined_model",
    }
    legend_html = _common_legend_html(_peer_version_items(_peer_versions()))
    return _model.build_page(meta, tabs, parser_ni=_parser_ni_map(),
                             legend_html=legend_html)


def _captured_note(mode: str) -> str:
    captured = _CAPTURED_WITH_BY_MODE.get(mode) or {}
    if not captured:
        return ""
    pairs = ", ".join(f"{impl} {ver}" for impl, ver in sorted(captured.items()))
    return (f"Peer streaming output captured against: {pairs}. "
            "A divergence is relative to these versions; re-capture when bumping.")


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
    common.set_links(resolved_output_path, artifact_root)

    now = datetime.datetime.now(zoneinfo.ZoneInfo("America/Los_Angeles"))
    stamp = now.strftime("%Y-%m-%d %H:%M %Z")
    sha = _commit_sha()

    # DIS-2434: the page is now rendered ENTIRELY by the JS view from this JSON model —
    # the Python HTML emitters (render_html_panel/render_cell_html/tooltip builders) are
    # gone; the template emits a skeleton and the model blob, the view builds the DOM.
    page_model = build_combined_model(
        output_path=output_path, artifact_root=artifact_root, stamp=stamp, sha=sha)
    # Per-family declared markers (pairs/singletons) for the JS colorizer's declared
    # lookup — the same table markup.py's _declared_lookup consults server-side.
    page_model["family_markers"] = declared_markers()
    model_json = _model.to_script_json(page_model)

    html = (
        _make_jinja_env()
        .get_template("conformance_table.html.j2")
        .render(
            title="Dynamo Parser v2 Conformance Table",
            title_html="Dynamo Parser v2 Conformance Table",
            stamp=stamp,
            conformance_css=_read_asset("conformance.css"),
            conformance_js=_read_asset("conformance.js"),
            colorize_js=_read_asset("colorize.js"),
            conformance_view_js=_read_asset("conformance_view.js"),
            sha=sha,
            short_sha=sha[:12] if sha else "",
            command="conformance/utils/render_table_v2.sh",
            output=_display_path(resolved_output_path, artifact_root),
            parser_ni_json=json.dumps(_parser_ni_map()),
            model_json=model_json,
        )
    )
    return _scrub_visible_conformance_text(html)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate frontend-crate conformance tables.",
    )
    parser.add_argument(
        "stage",
        choices=("all",),
        help="Conformance stage to render.",
    )
    args, rest = parser.parse_known_args(argv)

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


if __name__ == "__main__":
    main()
