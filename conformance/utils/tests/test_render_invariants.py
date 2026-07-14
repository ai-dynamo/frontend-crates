# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Semantic invariants on the FINAL rendered page (not greps of the source code).

Born from a real escape: the resolver fold doubled Dynamo v1 output into
`calls=[get_weatherget_weather(...)]` and it shipped to the rendered page unnoticed
because verification stopped at "the data exists", never "the output is right".

Scoping learned the hard way while writing these:
- Doubled names are only OUR bug in Dynamo-attributed content. Captured peer blocks
  legitimately record imperfect engine behavior (e.g. SGLang really emits
  `get_weatherget_weather` on gemma4's two-call case) — the legend says so.
- Row-slicing the HTML with `<tr>...</tr>` regexes is wrong: tooltips embed nested
  tables. Cells carry `data-family`; panels have ids — use those.
"""
from __future__ import annotations

import os
import re
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

UTILS = Path(__file__).resolve().parents[1]
SRC = UTILS / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from resolve_stream_fixtures import resolve, version_key  # noqa: E402

FIXTURES_ROOT = Path(
    os.environ.get(
        "CONFORMANCE_FIXTURES_ROOT",
        os.path.expanduser("~/.cache/dynamo/conformance-fixtures"),
    )
)
STREAM_SRC = FIXTURES_ROOT / "toolcalling" / "fixtures-stream-v2"

pytestmark = pytest.mark.skipif(
    not STREAM_SRC.is_dir(), reason="conformance fixtures not downloaded"
)

DOUBLED = re.compile(r"calls=\[(\w+?)\1\(")


def _dynamo_version_dirs() -> list[tuple[str, Path]]:
    out = []
    for d in STREAM_SRC.iterdir():
        if d.is_dir() and d.name.startswith("dynamo_rust-"):
            out.append((d.name.split("-", 1)[1], d))
    out.sort(key=lambda t: version_key(t[0]))
    return out


def _panel(html: str, panel_id: str) -> str:
    i = html.find(f'id="{panel_id}"')
    assert i != -1, f"panel {panel_id!r} not found in render"
    j = html.find('id="tab-', i + len(panel_id) + 6)
    return html[i : j if j != -1 else len(html)]


# --- I1: no doubled call name in DYNAMO-attributed output --------------------------
def test_no_doubled_call_names_in_dynamo_output(rendered_page):
    html = rendered_page.read_text()
    bad: set[str] = set()
    # Tooltip candidate sections for any Dynamo candidate (top blocks).
    for m in re.finditer(
        r'class="cand cand-dynamo[^"]*">(.*?)</span>', html, re.S
    ):
        bad.update(DOUBLED.findall(m.group(1)))
    # Per-chunk grid assembled cells for Dynamo candidates.
    for m in re.finditer(r'<td data-cand="dynamo[^"]*">(.*?)</td>', html, re.S):
        bad.update(DOUBLED.findall(m.group(1)))
    assert not bad, f"doubled call names in Dynamo output: {sorted(bad)}"


# --- I2: per-chunk grid columns must be selectable in the compare bar --------------
def test_grid_candidate_keys_match_compare_bar(rendered_page):
    html = rendered_page.read_text()
    grid_keys = set(re.findall(r'<th data-cand="([^"]+)">', html))
    bar_keys = set(re.findall(r'class="cmp-(?:on|ref)"[^>]*value="([^"]+)"', html))
    orphans = grid_keys - bar_keys
    assert not orphans, (
        f"grid columns not toggleable from the compare bar (key mismatch): {sorted(orphans)}"
    )


# --- I3: folding a higher dynamo version reproduces that version's docs exactly ----
def test_fold_reproduces_each_dynamo_version_exactly():
    versions = _dynamo_version_dirs()
    if len(versions) < 2:
        pytest.skip("needs at least two dynamo_rust version dirs")
    top_ver, top_dir = versions[-1]
    with tempfile.TemporaryDirectory() as tmp:
        resolve(STREAM_SRC, tmp, select=[f"dynamo_rust-{top_ver}"])
        for vfp in top_dir.glob("*/*.yaml"):
            folded_fp = Path(tmp) / vfp.parent.name / vfp.name
            if not folded_fp.exists():
                continue
            want_doc = yaml.safe_load(vfp.read_text()) or {}
            got_doc = yaml.safe_load(folded_fp.read_text()) or {}
            for cid, want_case in (want_doc.get("cases") or {}).items():
                got_case = (got_doc.get("cases") or {}).get(cid)
                if got_case is None or "unavailable" in want_case:
                    continue
                got = [
                    (i, d)
                    for i, ch in enumerate(got_case.get("chunks") or [])
                    for d in ((ch.get("expected") or {}).get("dynamo_rust") or [])
                ]
                want = [
                    (i, d)
                    for i, ch in enumerate(want_case.get("chunks") or [])
                    for d in (ch.get("expected") or [])
                ]
                assert got == want, (
                    f"{vfp.parent.name}/{vfp.name} {cid}: folded dynamo_rust deltas "
                    f"differ from the {top_ver} doc (lower-version residue?)\n"
                    f"  got:  {got}\n  want: {want}"
                )


# --- I4: every family the default REF covers renders populated stream cells --------
def test_ref_covered_families_have_populated_stream_cells(rendered_page):
    versions = _dynamo_version_dirs()
    if not versions:
        pytest.skip("no dynamo_rust version dirs")
    _ver, anchor_dir = versions[0]  # lowest version = the default REF
    covered = {p.name for p in anchor_dir.iterdir() if p.is_dir()}
    panel = _panel(rendered_page.read_text(), "tab-toolcalling-streamv2")
    for family in sorted(covered):
        classes = re.findall(
            rf'<td class="cell ([a-z]+)[^"]*"[^>]*data-family="{re.escape(family)}"',
            panel,
        )
        populated = sum(1 for c in classes if c in ("ok", "research", "documented"))
        assert populated > 0, (
            f"family {family!r} is covered by the default REF but renders no populated "
            f"stream cells (cells found: {len(classes)})"
        )


# --- I6: candidate chart and per-candidate list are mutually exclusive -------------
def test_no_tooltip_has_both_chart_and_candidate_list(rendered_page):
    """A tooltip that renders the candidate chart must NOT also render the legacy
    per-candidate list sections — the chart's output row replaces them."""
    html = rendered_page.read_text()
    both = 0
    # Splitting on tooltip openings yields one segment per tooltip (charts and cand
    # sections only exist inside tooltips, so trailing inter-tooltip markup is inert).
    for seg in html.split('<div class="ttip">')[1:]:
        if (
            '<table class="ttip-chunks">' in seg
            and 'data-cand="' in seg
            and 'class="cand cand-' in seg
        ):
            both += 1
    assert both == 0, f"{both} tooltips render BOTH the candidate chart and the list"


# --- I7: within an engine, compare-bar versions run latest-first -------------------
def test_compare_bar_versions_latest_first(rendered_page):
    """Within one engine (e.g. vLLM Python), versions must list DESCENDING
    (0.24.0 before 0.23.0) in the compare bar; engines keep their canonical order."""
    html = rendered_page.read_text()
    for pid in ("tab-toolcalling-batch", "tab-toolcalling-streamv2"):
        panel = _panel(html, pid)
        keys = re.findall(r'class="cmprow-label" data-cand="([^"]+)"', panel)
        groups: dict[str, list[str]] = {}
        for k in keys:
            m = re.match(r"^([a-z_]+(?:-[sb])?)-(\d[\w-]*)$", k)
            if m:
                groups.setdefault(m.group(1), []).append(m.group(2))
        for impl, slugs in groups.items():
            parsed = [version_key(s.replace("-", ".")) for s in slugs]
            assert parsed == sorted(parsed, reverse=True), (
                f"{pid}: {impl} versions not latest-first: {slugs}"
            )


# --- I5: no v2-implemented family renders as "inventory only" ----------------------
def test_implemented_v2_families_not_marked_inventory_only(rendered_page):
    """A family whose parser exists in parsers/v2 must not carry the
    'not implemented / inventory only' parser tooltip (the copy-paste chain gap that
    left gemma4/glm47/kimi_k2/minimax_m2/qwen3_coder mislabeled)."""
    registered = set(
        re.findall(
            r'^\s*"(\w+)"\s*(?:\|\s*"\w+"\s*)*=>',
            (UTILS.parents[1] / "parsers/v2/src/tool_calling/mod.rs").read_text(),
            re.M,
        )
    )
    registered.discard("other")
    panel = _panel(rendered_page.read_text(), "tab-toolcalling-streamv2")
    mislabeled = []
    for m in re.finditer(
        r'<td class="parser"[^>]*>(\w+)(?:<span[^>]*>[^<]*</span>)?'
        r"<div class=\"ttip\">.{0,400}?not implemented for this family yet",
        panel,
        re.S,
    ):
        if m.group(1) in registered:
            mislabeled.append(m.group(1))
    assert not mislabeled, (
        f"families with a real v2 parser labeled 'inventory only': {sorted(set(mislabeled))}"
    )


# --- I8: unaligned captures must not fake per-chunk timing -------------------------
def test_unaligned_candidates_show_no_per_chunk_timing(rendered_page):
    """A candidate column whose header carries the 'timing not recorded' note must
    have NO per-chunk deltas rendered (all cells em-dash outside the assembled row) —
    the v1 jail's emission-packed captures previously displayed at wrong positions."""
    html = rendered_page.read_text()
    noted = 0
    for seg in html.split('<table class="ttip-chunks">')[1:]:
        table = seg.split("</table>")[0]
        m = re.search(
            r'<th data-cand="([^"]+)">(?:(?!</th>).)*?timing not recorded', table, re.S
        )
        if not m:
            continue
        noted += 1
        key = m.group(1)
        for row in table.split("<tr")[1:]:
            if 'class="ttip-final"' in row:
                continue
            cm = re.search(rf'<td data-cand="{re.escape(key)}">(.*?)</td>', row, re.S)
            if cm:
                assert cm.group(1).strip() in ("—", ""), (
                    f"noted column {key} renders per-chunk data: {cm.group(1)[:80]!r}"
                )
    assert noted > 0, "expected at least one 'timing not recorded' column (v1 jail)"
    # The reasoning tab's final-output-only notes alone must NOT satisfy this guard:
    # the TC stream tab's v1 jail column (emission-packed capture) must be noted too.
    html2 = rendered_page.read_text()
    assert re.search(
        r'<th data-cand="dynamo_rust-3-0-0">(?:(?!</th>).)*?timing not recorded',
        html2, re.S,
    ), "the v1 jail (dynamo_rust-3-0-0) stream column lacks the timing note"
