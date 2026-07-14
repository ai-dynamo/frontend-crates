# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared contract types + canonical-JSON diff for parity (parser) impls.

Every impl wrapper (parser-mode and e2e-mode) returns ParseResult so the
harness can diff results uniformly.
"""

from __future__ import annotations

import html as html_lib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Destination-aware link resolution, shared by the v1 PARITY and v2 CONFORMANCE
# generators. Each generator calls set_links() with its own output path before
# building, so the table builders emit hrefs that resolve from where the page is
# actually published -- no post-render path rewriting.
# ---------------------------------------------------------------------------


def _href_from_output(output_path: Path, artifact_root: Path, repo_relative: str) -> str:
    trailing_slash = repo_relative.endswith("/")
    target = artifact_root / repo_relative.rstrip("/")
    href = Path(os.path.relpath(target, output_path.parent)).as_posix()
    return f"{href}/" if trailing_slash else href


def _hrefs_for_output(output_path: Path, artifact_root: Path) -> dict[str, str]:
    def h(rel: str) -> str:
        return _href_from_output(output_path, artifact_root, rel)

    return {
        "toolcalling_fixtures": h("conformance/toolcalling/fixtures-v1/"),
        "toolcalling_stream_fixtures": h("conformance/toolcalling/fixtures-stream-v2/"),
        "toolcalling_batch_on_stream_fixtures": h(
            "conformance/toolcalling/fixtures-batch-on-stream-v2/"
        ),
        "reasoning_fixtures": h("conformance/reasoning/fixtures/"),
        "toolcalling_cases": h("conformance/utils/lib/parsers/TOOLCALLING_CASES.md"),
        "toolcalling_streaming_cases": h(
            "conformance/utils/lib/parsers/TOOLCALLING_STREAMING_V2_CASES.md"
        ),
        "reasoning_cases": h("conformance/utils/lib/parsers/REASONING_CASES.md"),
        "toolcalling_src": h("parsers/v1/src/tool_calling/"),
        "reasoning_src": h("parsers/v1/src/reasoning/"),
        "streaming_src": h("parsers/v2/src/tool_calling/"),
        "streaming_harmony_src": h("parsers/v2/src/tool_calling/harmony.rs"),
        "pyproject_stub": h("conformance/utils/src/pyproject.stub.toml"),
    }


# Fixtures aren't in the repo — they're published to HuggingFace as tarballs and
# extracted into the local cache (~/.cache/dynamo/conformance-fixtures/, or
# $CONFORMANCE_FIXTURES_ROOT). HF stores only the tarballs (no per-file blob URL), so a
# per-cell YAML link points at the extracted file in that cache via file://. The
# rendered `__fixture_path` is the flat resolved-tree path the readers use; remap it to
# the versioned cache layout (the shared `inputs/` tree carries the model_text /
# description a viewer wants to see).
def _fixtures_cache_root() -> str:
    env = os.environ.get("CONFORMANCE_FIXTURES_ROOT")
    if env:
        return env.rstrip("/")
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return str(base / "dynamo/conformance-fixtures")


def _fixture_cache_relpath(rel: str) -> str:
    """Map a rendered `<family>/<FILE>` fixture path to its versioned cache location.
    The renderers resolve batch + v1-stream into a single flat `fixtures/` dir, so the
    corpus can't be told from the path prefix — route on the FILE name instead."""
    parts = rel.lstrip("./").split("/")
    fname = parts[-1]
    family = parts[-2] if len(parts) >= 2 else ""
    if fname.startswith("REASONING."):
        corpus = "reasoning/fixtures-v1/inputs"
    elif fname.startswith("TOOLCALLING.streamv2"):
        corpus = "toolcalling/fixtures-stream-v2/inputs"
    elif fname.startswith("TOOLCALLING."):  # batch + v1 stream both live in the v1 corpus
        corpus = "toolcalling/fixtures-batch-v1/inputs"
    else:
        return rel.lstrip("./")
    return f"{corpus}/{family}/{fname}"


def fixture_href(rel: str) -> str:
    """Map a rendered fixture path to a file:// link into the local fixture cache.
    Leaves absolute URLs and empty strings untouched."""
    if not rel or "://" in rel:
        return rel
    return "file://" + _fixtures_cache_root() + "/" + _fixture_cache_relpath(rel)


# Render-context: the active generator sets this for its destination before the
# builders run; the builders read LINKS[...] at emit time.
LINKS: dict[str, str] = {}


def set_links(output_path: Path, artifact_root: Path) -> dict[str, str]:
    """Resolve all link bases for `output_path`, install them as the active
    render context, and return them for callers that also want the dict."""
    global LINKS
    LINKS = _hrefs_for_output(output_path, artifact_root)
    return LINKS


# Best-effort default so code that emits links before a generator has called
# set_links() (e.g. unit tests that invoke a builder helper directly) still
# resolves to the published-page layout. The active generator overrides this for
# its real destination on every render.
try:
    _DEFAULT_ROOT = Path(__file__).resolve().parents[4]
    if (_DEFAULT_ROOT / "parsers" / "v2").is_dir():
        set_links(_DEFAULT_ROOT / "conformance" / "PARITY.html", _DEFAULT_ROOT)
except (IndexError, OSError):
    pass

URL_RE = re.compile(r"https?://[^\s<>'\"]+")
TRAILING_URL_PUNCTUATION = ".,;:)"

# Curated featured-model order shared by the parser and reasoning parity tables.
# The row labels still come from fixture metadata; this list only decides which
# tool calling families lead the table and in what order.
TOP_N_TOOL_CALLING_FAMILIES = [
    "deepseek_v4",
    "gemma4",
    "glm47",
    "harmony",
    "kimi_k2",
    "minimax_m2",
    "minimax_m3",
    "qwen3_coder",
]

# Dynamo reasoning-parser family -> peer parser name. Kept here (a dependency-free
# module) so the parity table generator can read them without importing the sglang
# or vllm wrapper modules, which pull in the heavy engine packages at import time.
_FAMILY_TO_SGLANG_REASONING = {
    "deepseek_r1": "deepseek-r1",
    "deepseek_v3": "deepseek-v3",
    "deepseek_v4": "deepseek-v4",
    "gemma4": "gemma4",
    "gpt_oss": "gpt-oss",
    "kimi": "kimi",
    "kimi_k25": "kimi_k2",
    "minimax_append_think": "minimax-append-think",
    "mistral": "mistral",
    "nemotron_deci": "glm45",
    "qwen3": "qwen3",
}

_FAMILY_TO_VLLM_REASONING = {
    "deepseek_r1": "deepseek_r1",
    "deepseek_v3": "deepseek_v3",
    "deepseek_v4": "deepseek_v4",
    "gemma4": "gemma4",
    "gpt_oss": "openai_gptoss",
    "granite": "granite",
    "kimi_k25": "kimi_k2",
    "mistral": "mistral",
    "minimax_append_think": "minimax_m2_append_think",
    "minimax_m3": "minimax_m3",
    "nemotron_deci": "glm45",
    "qwen3": "qwen3",
}


@dataclass
class ParseResult:
    """Uniform shape returned by every impl wrapper.

    `calls` is a list of {"name": str, "arguments": dict}.  Argument values
    are dicts (parsed from JSON) so canonical comparison ignores whitespace
    differences in the wire encoding.
    """

    calls: list[dict[str, Any]] = field(default_factory=list)
    normal_text: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "normal_text": self.normal_text,
            "error": self.error,
        }


@dataclass
class ReasoningResult:
    """Uniform shape returned by every reasoning impl wrapper."""

    reasoning_text: str | None = None
    normal_text: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "reasoning_text": self.reasoning_text,
            "normal_text": self.normal_text,
            "error": self.error,
        }


def _normalize_normal_text(v: Any) -> Any:
    """Treat any whitespace-only or None value as equivalent for comparison.

    Engines disagree on the carrier for "no narration":
      * Dynamo emits ``""`` (or ``"\\n"`` carried through between back-to-back
        tool-call envelopes — see DSML TOOLCALLING.batch.2.b).
      * vLLM emits ``None``.
      * SGLang emits ``""``.

    All three express the same semantic and should compare equal."""
    if v is None:
        return None
    if isinstance(v, str) and v.strip() == "":
        return None
    return v


def canonical(d: dict[str, Any]) -> str:
    """Canonical JSON for diffing: sorted keys, no whitespace, with empty-string ↔ None
    normalization applied to parser text fields."""
    if "normal_text" in d:
        d = {**d, "normal_text": _normalize_normal_text(d["normal_text"])}
    if "reasoning_text" in d:
        d = {**d, "reasoning_text": _normalize_normal_text(d["reasoning_text"])}
    return json.dumps(d, sort_keys=True, separators=(",", ":"))


def decode_arguments(args: Any) -> Any:
    """Decode a tool-call `arguments` field to a comparable Python value.

    Parsers surface `arguments` slightly differently: JSON-encoded
    string (Dynamo, vLLM), already-parsed dict (some SGLang paths),
    or the truncated raw string verbatim on malformed input. Decode
    JSON if it parses; otherwise return as-is so the canonical-diff
    surfaces the mismatch rather than swallowing it.
    """
    if not args:
        return {}
    if not isinstance(args, str):
        return args
    try:
        return json.loads(args)
    except (json.JSONDecodeError, TypeError):
        return args


def decode_stream_calls(
    stream_calls: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    calls = []
    for _, call in sorted(stream_calls.items()):
        if not call.get("name") and not call.get("arguments"):
            continue
        calls.append(
            {
                "name": call.get("name") or "",
                "arguments": decode_arguments(call.get("arguments") or ""),
            }
        )
    return calls


def linkify_text_html(text: str) -> str:
    """Escape plain text and turn embedded URLs into anchors."""

    parts: list[str] = []
    last = 0
    for match in URL_RE.finditer(text):
        raw_url = match.group(0)
        url = raw_url.rstrip(TRAILING_URL_PUNCTUATION)
        if not url:
            continue
        parts.append(html_lib.escape(text[last : match.start()]))
        href = html_lib.escape(url, quote=True)
        parts.append(
            f'<a href="{href}" target="_blank" rel="noopener noreferrer">'
            f"{html_lib.escape(url)}</a>"
        )
        parts.append(html_lib.escape(raw_url[len(url) :]))
        last = match.end()
    parts.append(html_lib.escape(text[last:]))
    return "".join(parts)


def parity_cell_class(marker: str) -> str:
    if marker == "—":
        return "missing"
    if marker == "n/a":
        return "na"
    if marker in {"D", "·"}:
        return "donly"
    if "!" in marker or "✗" in marker:
        return "err"
    if "↯" in marker:
        return "leak"
    if "?" in marker:
        return "research"
    if marker == "=":
        return "ok"
    return "documented"


def ref_text(value: Any) -> str:
    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


# --- Shared candidate-chart scaffold -----------------------------------------
# Every popup chart (stream per-version, merged batch, reasoning, and the legacy
# per-impl grid) is the same <table class="ttip-chunks"> with a fixed `input`
# column plus per-candidate columns; only the columns/rows differ. These helpers
# own that scaffold so the four builders don't each re-template it (and so the
# `data-cand` / `data-col-impl` attribute the compare JS keys on can't drift).


def cand_th(key: str, inner: str, *, attr: str = "data-cand") -> str:
    """A candidate-chart column header. `key` becomes the `attr` value the compare
    JS toggles/reorders; `inner` is already-rendered (escaped) HTML."""
    return f'<th {attr}="{html_lib.escape(key, quote=True)}">{inner}</th>'


def cand_td(key: str, inner: str, *, attr: str = "data-cand") -> str:
    """A candidate-chart body cell (see cand_th). `inner` is already-rendered HTML."""
    return f'<td {attr}="{html_lib.escape(key, quote=True)}">{inner}</td>'


def timing_note(reason: str) -> str:
    """The muted `(… per-chunk timing not recorded)` header note the stream charts
    add to a column whose capture is not per-input-chunk aligned. `reason` is the
    parenthetical text (e.g. `bursts at end of call; per-chunk timing not recorded`)."""
    return f' <span class="ttip-note">({html_lib.escape(reason)})</span>'


def candidate_chart_table(header_cells: str, body_rows: list[str]) -> str:
    """Assemble the shared chart <table>: fixed `input` column + per-candidate
    columns (header_cells), then body_rows. Callers build the columns/rows;
    the scaffold lives here once."""
    return (
        '<table class="ttip-chunks"><thead><tr><th>input</th>'
        + header_cells
        + "</tr></thead><tbody>"
        + "".join(body_rows)
        + "</tbody></table>"
    )


def build_parity_tooltip_html(
    *,
    head: str,
    description: str | None = None,
    input_label: str | None = None,
    input_html: str | None = None,
    output_sections: list[tuple[str, str]] | None = None,
    divergent_reasons: str | None = None,
    leak_label: str | None = None,
    leak_text: str | None = None,
    extra_sections: list[tuple[str, str]] | None = None,
    chart: tuple[str, str] | None = None,
    refs: list[tuple[str, Any]] | None = None,
) -> str:
    """Build the hover popup used by parser and reasoning parity tables.

    The section order is intentional and shared across the two tables:
    context, input, outputs, divergence explanation, leak warning, harness
    details, and finally provenance refs.
    """

    parts = ['<div class="ttip">']
    if head:
        parts.append(f'<div class="ttip-head">{html_lib.escape(head)}</div>')
    if description:
        parts.append(f'<pre class="ttip-pre">{html_lib.escape(description)}</pre>')

    def add_section(
        label: str, body_html: str, wrap_class: str | None = None, leak: bool = False
    ) -> None:
        # A leaking candidate gets a red ↯ after the label, e.g. "… (stream): ↯".
        marker = ' <span class="ttip-leak">↯</span>' if leak else ""
        section = (
            f'<div class="ttip-section">{html_lib.escape(label)}:{marker}</div>'
            f'<pre class="ttip-pre">{body_html}</pre>'
        )
        # Optional wrapper lets callers toggle a whole labeled section (e.g. the
        # per-version output blocks swapped by the version radios).
        if wrap_class:
            section = f'<span class="{wrap_class}">{section}</span>'
        parts.append(section)

    if input_label and input_html is not None:
        add_section(input_label, input_html)

    for section in output_sections or []:
        add_section(*section)

    # Candidate chart (left-to-right per-candidate output table). Raw table HTML —
    # not a <pre> section — toggled/REF-ordered by the shared compare JS.
    if chart:
        chart_label, chart_html = chart
        parts.append(
            f'<div class="ttip-section">{html_lib.escape(chart_label)}:</div>'
        )
        parts.append(chart_html)

    if divergent_reasons:
        add_section("Divergent reasons", linkify_text_html(divergent_reasons))

    if leak_label and leak_text:
        add_section(leak_label, linkify_text_html(leak_text))

    for label, body_html in extra_sections or []:
        add_section(label, body_html)

    for label, value in refs or []:
        if value:
            add_section(label, linkify_text_html(ref_text(value)))

    parts.append("</div>")
    return "".join(parts)
