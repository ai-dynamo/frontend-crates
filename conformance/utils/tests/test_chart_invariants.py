# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression guard for the rendered conformance charts.

The PARITY_v1 and CONFORMANCE_v2 HTML pages have repeatedly regressed *silently*:
a fixture path that moved to the HF cache, a Cargo.toml that moved under the
DIS-2310 reorg, a dropped reference fixture — each left the chart still rendering
(exit 0) but with an empty parser selector, a version-less candidate, a whole tab
of "not yet implemented", or missing peer versions. `cargo test` and the plain
render-smoke never caught them because the HTML was still "valid".

This module renders both charts and asserts the structural properties of a *good*
chart, so the next such regression fails a test instead of shipping. Expected peer
versions are DERIVED from the downloaded fixture dirs (not hard-coded), so the
guard keeps working across version bumps.

Skips when fixtures aren't downloaded (the CI conformance-table job downloads them;
locally, run `python3 conformance/utils/src/download_fixtures.py` first).
"""
import os
import re
import subprocess
from pathlib import Path

import pytest

UTILS = Path(__file__).resolve().parents[1]
REPO = UTILS.parents[1]


def _cache_root() -> Path:
    env = os.environ.get("CONFORMANCE_FIXTURES_ROOT")
    if env:
        return Path(env)
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "dynamo/conformance-fixtures"


_HAVE_FIXTURES = (_cache_root() / "toolcalling").is_dir()
pytestmark = pytest.mark.skipif(
    not _HAVE_FIXTURES, reason="fixtures not downloaded (run download_fixtures.py)"
)


def _peer_versions(tree: str) -> dict[str, set[str]]:
    """{impl: {versions}} discovered from the cached fixture dirs of one tree, e.g.
    fixtures-batch-v1 -> {'vllm': {'0.23.0','0.24.0'}, 'sglang': {...}, 'dynamo': {...}}."""
    out: dict[str, set[str]] = {}
    root = _cache_root() / tree
    if root.is_dir():
        for d in root.iterdir():
            if d.is_dir() and d.name != "inputs" and "-" in d.name:
                impl, ver = d.name.split("-", 1)
                out.setdefault(impl, set()).add(ver)
    return out


@pytest.fixture(scope="module")
def charts() -> dict[str, str]:
    """Render both charts once (or reuse the CI job's output if already present)."""
    v1_path = REPO / "conformance/PARITY_v1.html"
    v2_path = REPO / "conformance/CONFORMANCE_v2.html"
    if not v1_path.exists():
        subprocess.run(
            [str(UTILS / "render_table_v1.sh")], check=True, capture_output=True, cwd=REPO
        )
    if not v2_path.exists():
        subprocess.run(
            [str(UTILS / "render_table_v2.sh")], check=True, capture_output=True, cwd=REPO
        )
    return {"v1": v1_path.read_text(encoding="utf-8"), "v2": v2_path.read_text(encoding="utf-8")}


def _panel(html: str, panel_id: str, all_ids: tuple[str, ...]) -> str:
    """The HTML of one tab panel: from this panel's id to the next panel's id.

    Slices between panel ids rather than by the `tab-panel` class — the class and id
    live in the same <div> tag and their order isn't guaranteed, so a class-based cut
    can collapse to nothing."""
    i = html.find(f'id="{panel_id}"')
    assert i >= 0, f"panel {panel_id!r} is missing from the chart"
    ends = [html.find(f'id="{t}"', i + 1) for t in all_ids if t != panel_id]
    ends = [e for e in ends if e > i]
    return html[i : (min(ends) if ends else len(html))]


def _candidates(segment: str) -> list[str]:
    """Compare-bar candidate labels (chips) within a panel segment. A label may carry
    one level of inline markup (the colored (batch)/(stream) mode word), so capture the
    whole label span — allowing a nested <span> — and strip tags for the visible text."""
    raw = re.findall(
        r'data-cand="[^"]*"[^>]*>((?:[^<]|<span[^>]*>[^<]*</span>)*)</span>', segment
    )
    return [re.sub(r"<[^>]+>", "", r).strip() for r in raw]


# A candidate label is "<Engine> <Runtime> <version> (<mode>)". The version token
# (a digit-led run right before the trailing "(mode)") is what regressed to nothing
# when the crate/fixture path was wrong (bare "Dynamo Rust (stream)").
_VERSIONED_LABEL = re.compile(r"\b\d[\w.]*\s+\([^)]+\)\s*$")

_V2_TABS = (
    "tab-toolcalling-batch",
    "tab-toolcalling-streamv2",
    "tab-reasoning-batch",
    "tab-reasoning-stream",
)
_V1_TABS = (
    "tab-toolcalling-batch",
    "tab-toolcalling-stream",
    "tab-reasoning-batch",
    "tab-reasoning-stream",
)


# --------------------------------------------------------------------------- #
# CONFORMANCE_v2
# --------------------------------------------------------------------------- #
def test_v2_all_tabs_present(charts):
    for tab in _V2_TABS:
        assert f'id="{tab}"' in charts["v2"], f"v2 chart missing tab {tab}"


def test_v2_every_tab_has_a_parser_selector(charts):
    # The empty-parser-selector regression (repo-path fixture read): a tab renders
    # but has zero compare candidates, so there is nothing to select.
    for tab in _V2_TABS:
        cands = _candidates(_panel(charts["v2"], tab, _V2_TABS))
        assert cands, f"v2 tab {tab} has no parser candidates (empty selector)"


def test_v2_every_candidate_is_versioned(charts):
    # The bare-"Dynamo Rust (stream)" regression: a candidate label lost its version.
    for tab in _V2_TABS:
        for label in _candidates(_panel(charts["v2"], tab, _V2_TABS)):
            assert _VERSIONED_LABEL.search(label), f"v2 {tab}: candidate not versioned: {label!r}"


def test_v2_batch_tab_has_all_peer_versions(charts):
    seg = _panel(charts["v2"], "tab-toolcalling-batch", _V2_TABS)
    for impl, vers in _peer_versions("toolcalling/fixtures-batch-v1").items():
        for v in vers:
            assert v in seg, f"v2 batch tab missing {impl} {v}"


def test_v2_stream_tab_has_v1jail_reference_and_v2_and_peers(charts):
    seg = _panel(charts["v2"], "tab-toolcalling-streamv2", _V2_TABS)
    assert "Dynamo v1 Rust 3.0.0 (jail+batch)" in seg, "v2 stream tab lost its v1-jail reference"
    assert "Dynamo v2 Rust" in seg and "(stream)" in seg, "v2 stream tab lost the Dynamo v2 candidate"
    for impl, vers in _peer_versions("toolcalling/fixtures-stream-v2").items():
        if impl in ("dynamo_rust",):
            continue
        for v in vers:
            assert v in seg, f"v2 stream tab missing {impl} {v}"


def test_v2_reference_aware_not_implemented_map(charts):
    # The reference-aware "not implemented" note (shown when the Dynamo v2 parser is the
    # selected Reference on a family it doesn't support) is driven by window.__PARSER_NI:
    # the v2 candidate keys -> the exact families its fixture dir covers, plus data-family
    # on every cell so the JS can map a cell to its family. A regression here (empty map,
    # all-families map, or missing data-family) silently reverts to the misleading
    # case-level "not applicable".
    import json

    v2_dir = _cache_root() / "toolcalling/fixtures-stream-v2/dynamo_rust-0.1.11"
    if not v2_dir.is_dir():
        pytest.skip("v2 stream fixture dir not present")
    fams = sorted(d.name for d in v2_dir.iterdir() if d.is_dir())
    html = charts["v2"]
    m = re.search(r"window\.__PARSER_NI = (\{.*?\});", html)
    assert m, "reference-aware __PARSER_NI map is missing"
    ni = json.loads(m.group(1))
    assert ni, "__PARSER_NI is empty — the not-implemented reason won't fire"
    for key, entry in ni.items():
        assert "dynamo_rust" in key, f"unexpected limited-coverage parser {key}"
        assert sorted(entry["families"]) == fams, (
            f"{key} families {sorted(entry['families'])} != fixture dir {fams}"
        )
    assert html.count("data-family=") > 100, "cells lost data-family (NI can't map a cell to its family)"


def test_v2_stream_parser_only_covers_implemented_families(charts):
    # The Dynamo v2 stream parser (dynamo_rust-0.1.11) implements only a handful of
    # families; its fixture dir holds exactly those. The stream assembly defaults an
    # absent impl to an empty-but-present block, which used to paint the v2 candidate
    # green on EVERY family. Guard: the v2 stream candidate must be `na` on the
    # families its parser doesn't implement, so most cells are na (not all present).
    import json
    from html import unescape

    v2_dir = _cache_root() / "toolcalling/fixtures-stream-v2/dynamo_rust-0.1.11"
    if not v2_dir.is_dir():
        pytest.skip("v2 stream fixture dir not present")
    n_v2_families = len([d for d in v2_dir.iterdir() if d.is_dir()])
    n_all_families = len(
        [d for d in (_cache_root() / "toolcalling/fixtures-stream-v2/dynamo_rust-3.0.0").iterdir()
         if d.is_dir()]
    )
    assert n_v2_families < n_all_families, "expected v2 to implement fewer families than v1 jail"

    seg = _panel(charts["v2"], "tab-toolcalling-streamv2", _V2_TABS)
    present = na = 0
    for blob in re.findall(r'data-cmp="([^"]+)"', seg):
        try:
            cmp = json.loads(unescape(blob))
        except json.JSONDecodeError:
            continue
        v = cmp.get("dynamo_rust-0-1-11")
        if v is None:
            continue
        if v.get("na") == 1:
            na += 1
        else:
            present += 1
    assert na > 0, "v2 stream candidate has NO n/a cells — it is claiming coverage it lacks"
    # Only ~4 of ~19 families are implemented, so n/a cells must outnumber present ones.
    assert na > present, (
        f"v2 stream candidate: {present} present vs {na} n/a — it looks implemented for "
        f"too many families (v2 dir has {n_v2_families} of {n_all_families})"
    )


def test_v2_no_not_yet_implemented_text(charts):
    # Un-implemented Dynamo v2 families render as a clean n/a in the visible grid/tooltip
    # HTML — never a verbose TODO baked into a cell. The reference-aware note (shown only
    # when v2 is the selected Reference) is injected by conformance.js at runtime, so
    # strip the inlined <script> before checking; its legitimate note text is allowed.
    visible = re.sub(r"<script>.*?</script>", "", charts["v2"], flags=re.S)
    assert "not yet implemented" not in visible


def test_divergence_notes_use_explanation_label(charts):
    # The divergence-note field renders under the "explanation:" label (renamed from
    # the confusing "reason:", which collided with "reasoning"). Legacy fixtures with a
    # `reason` key are still read, but the displayed label is always "explanation:".
    for page in ("v1", "v2"):
        assert "explanation:" in charts[page], f"{page}: no explanation: note label"
        # No bare "reason: <Text>" divergence label should leak (finish_reason: is fine).
        assert not re.search(r"[^_]reason: [A-Z]", charts[page]), f"{page}: stale reason: label"


def test_v2_reasoning_candidates_are_versioned(charts):
    for tab in ("tab-reasoning-batch", "tab-reasoning-stream"):
        cands = _candidates(_panel(charts["v2"], tab, _V2_TABS))
        assert cands, f"v2 {tab} has no reasoning parser candidates"
        assert any("Dynamo v1 Rust" in c for c in cands), f"v2 {tab} missing Dynamo reasoning candidate"
        for c in cands:
            assert _VERSIONED_LABEL.search(c), f"v2 {tab}: reasoning candidate not versioned: {c!r}"


def test_v2_cells_have_compare_data(charts):
    # data-cmp drives the overview coloring; without it every cell falls back to raw
    # markers (the letters-not-colors regression on the parity page).
    assert charts["v2"].count("data-cmp=") > 100


def test_v2_reasoning_in_sync_with_toolcalling_peers(charts):
    # CONFORMANCE_v2 is the "current" page: its reasoning tab must compare against the
    # SAME current engine versions as its toolcalling tab. The bug was reasoning stuck
    # on the old peers (0.23.0 / 0.5.12.post1) while toolcalling showed 0.24.0 / 0.5.14.
    tc = _panel(charts["v2"], "tab-toolcalling-batch", _V2_TABS)
    rz = _panel(charts["v2"], "tab-reasoning-batch", _V2_TABS)
    for impl, vers in _peer_versions("toolcalling/fixtures-batch-v1").items():
        if impl == "dynamo":
            continue
        newest = max(vers, key=lambda v: tuple(int(x) for x in re.findall(r"\d+", v)))
        assert newest in tc, f"v2 toolcalling missing current {impl} {newest}"
        assert newest in rz, (
            f"v2 reasoning missing current {impl} {newest} — out of sync with toolcalling"
        )


def test_v2_batch_tab_stream_candidates_use_current_peers(charts):
    # The batch tab ("Tool Calling (batch data)") also compares each engine's STREAMING
    # parser run over the batch text (candidate label "<Engine> <ver> (stream)"). Those
    # blocks come from the batch-on-stream fixtures, which were frozen at the old engines
    # (0.23.0 / 0.5.12.post1) long after the batch parsers moved to 0.24.0 / 0.5.14 — so
    # the tab showed a stale "(stream)" version next to the current "(batch)" one. The
    # container-captured peers (vllm_python, sglang_python) must carry the CURRENT engine
    # version on their (stream) candidate. vllm_rust is a source-captured single-version
    # peer and is intentionally excluded.
    seg = _panel(charts["v2"], "tab-toolcalling-batch", _V2_TABS)
    labels = {
        "vllm": "vLLM Python",
        "sglang": "SGLang Python",
    }
    for impl, vers in _peer_versions("toolcalling/fixtures-batch-v1").items():
        if impl not in labels:
            continue
        newest = max(vers, key=lambda v: tuple(int(x) for x in re.findall(r"\d+", v)))
        assert f"{labels[impl]} {newest} (stream)" in seg, (
            f"v2 batch tab stream candidate for {impl} is not the current {newest} "
            f"— batch-on-stream fixtures need re-capturing at the current engines"
        )


def test_dynamo_version_labels_are_consistent_and_from_fixtures(charts):
    # The same Dynamo parser must show ONE version everywhere — a split (e.g. v1 3.0.0
    # in tool-calling but 4.1.0 in reasoning, or v2 0.1.11 vs 0.1.16) means a label is
    # reading the live Cargo.toml, which drifts ahead of the published fixtures, instead
    # of the fixture provenance. The version must also actually exist as a captured dir.
    fixture_v1 = _peer_versions("toolcalling/fixtures-batch-v1").get("dynamo", set())
    fixture_v2 = {
        v for v in _peer_versions("toolcalling/fixtures-stream-v2").get("dynamo_rust", set())
        if v.split(".", 1)[0] == "0"  # v2 crate = 0.x major
    }
    for page in ("v1", "v2"):
        for tag, expected in (("Dynamo v1 Rust", fixture_v1), ("Dynamo v2 Rust", fixture_v2)):
            seen = set(re.findall(rf"{tag} (\S+) \(", charts[page]))
            assert len(seen) <= 1, f"{page}: {tag} has inconsistent versions: {seen}"
            if seen and expected:
                assert seen <= expected, (
                    f"{page}: {tag} version {seen} is not a captured fixture version "
                    f"{expected} — it is likely read from the live Cargo.toml"
                )


# --------------------------------------------------------------------------- #
# PARITY_v1
# --------------------------------------------------------------------------- #
def test_v1_all_tabs_present(charts):
    for tab in _V1_TABS:
        assert f'id="{tab}"' in charts["v1"], f"v1 chart missing tab {tab}"


def test_v1_toolcalling_tabs_have_both_old_and_new_peers(charts):
    # The v1 page is the legacy baseline but must still CONTAIN every captured peer
    # version (both the v1-era engines and the current ones) — not just one.
    want = _peer_versions("toolcalling/fixtures-batch-v1")
    for tab in ("tab-toolcalling-batch", "tab-toolcalling-stream"):
        seg = _panel(charts["v1"], tab, _V1_TABS)
        for impl, vers in want.items():
            if impl == "dynamo":
                continue
            for v in vers:
                assert v in seg, f"v1 {tab} missing {impl} {v}"


def test_v1_cells_have_compare_data(charts):
    assert charts["v1"].count("data-cmp=") > 100


def test_v1_every_candidate_is_versioned(charts):
    for tab in ("tab-toolcalling-batch", "tab-toolcalling-stream"):
        for label in _candidates(_panel(charts["v1"], tab, _V1_TABS)):
            assert _VERSIONED_LABEL.search(label), f"v1 {tab}: candidate not versioned: {label!r}"
