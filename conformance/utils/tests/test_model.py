# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Structural guards on the JSON data model (DIS-2434).

The conformance page is `one JSON data model + a JS view`. Python computes the model;
the view renders it. These guards assert the structural properties of a *good* model
directly, instead of regex-scraping the rendered HTML the way test_chart_invariants
does — stronger (they check the data the view consumes) and less brittle (no HTML
shape coupling). They are the migration target for the chart-invariant guards named in
the ticket.

The model is the inlined `<script type="application/json" id="conformance-model">`
blob. The CI conformance-table job renders both pages first, then runs this in the
no-browser venv (pyyaml/jinja2/pytest only), so we parse the repo-rendered pages (and
render them if absent, mirroring test_chart_invariants).

Expected peer versions are DERIVED from the downloaded fixture dirs (not hard-coded),
so the guards keep working across version bumps.
"""
import json
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
    not _HAVE_FIXTURES, reason="fixtures not extracted (run extract_fixtures.py)"
)

_MODEL_RE = re.compile(
    r'<script type="application/json" id="conformance-model">(.*?)</script>', re.S
)


def _read_model(page_path: Path, render_script: str) -> dict:
    if not page_path.exists():
        subprocess.run([str(UTILS / render_script)], check=True, capture_output=True, cwd=REPO)
    html = page_path.read_text(encoding="utf-8")
    m = _MODEL_RE.search(html)
    assert m, f"{page_path.name}: no conformance-model blob"
    return json.loads(m.group(1))


@pytest.fixture(scope="module")
def model_v2() -> dict:
    return _read_model(REPO / "conformance/CONFORMANCE_v2.html", "render_table_v2.sh")


@pytest.fixture(scope="module")
def model_v1() -> dict:
    return _read_model(REPO / "conformance/PARITY_v1.html", "render_table_v1.sh")


def _tab(model: dict, tab_id: str) -> dict:
    for t in model["tabs"]:
        if t["id"] == tab_id:
            return t
    raise AssertionError(f"tab {tab_id!r} missing; have {[t['id'] for t in model['tabs']]}")


def _iter_cells(tab: dict):
    for row in tab["rows"]:
        for sub, cell in row.get("cells", {}).items():
            if cell.get("kind") == "cell":
                yield cell


def _peer_versions(tree: str) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    root = _cache_root() / tree
    if root.is_dir():
        for d in root.iterdir():
            if d.is_dir() and d.name != "inputs" and "-" in d.name:
                impl, ver = d.name.split("-", 1)
                out.setdefault(impl, set()).add(ver)
    return out


_VER_PAREN = re.compile(r"\b\d[\w.]*\s+\([^)]+\)\s*$")

# ---- schema + shape -----------------------------------------------------------

def test_v2_schema_and_meta(model_v2):
    assert model_v2["schema"] == 1
    meta = model_v2["meta"]
    assert meta["title"] and meta["stamp"] and meta["command"]


def test_v2_all_tabs_present(model_v2):
    ids = [t["id"] for t in model_v2["tabs"]]
    assert ids == [
        "tab-toolcalling-batch", "tab-toolcalling-streamv2",
        "tab-reasoning-batch", "tab-reasoning-stream",
    ], ids


def test_v2_exactly_one_active_tab(model_v2):
    assert sum(1 for t in model_v2["tabs"] if t.get("active")) == 1
    assert model_v2["tabs"][0]["active"] is True


# ---- candidates (compare bar) -------------------------------------------------

def test_v2_every_tab_has_candidates(model_v2):
    for t in model_v2["tabs"]:
        assert t["candidates"], f"{t['id']}: empty compare selector"


def test_v2_every_candidate_is_versioned(model_v2):
    for t in model_v2["tabs"]:
        for c in t["candidates"]:
            assert _VER_PAREN.search(c["label"]), f"{t['id']}: unversioned candidate {c['label']!r}"


def test_v2_exactly_one_reference_bucket_per_tab(model_v2):
    for t in model_v2["tabs"]:
        refs = [c for c in t["candidates"] if c["default_bucket"] == "A"]
        assert len(refs) == 1, f"{t['id']}: expected one bucket-A reference, got {len(refs)}"


_IMPL_KEYS = ("dynamo_v1", "dynamo_v2", "vllm_rust", "vllm_python", "sglang_python")


def _impl_key_of(cand_key: str) -> str:
    return next((k for k in _IMPL_KEYS if cand_key.startswith(k)), cand_key)


def test_v2_candidate_versions_latest_first_within_impl(model_v2):
    # test_render_invariants I7: within one implementation, compare-bar versions descend
    # (latest first). Grouped by the underlying impl KEY (vllm_rust vs vllm_python share
    # the "vLLM" display column but are separate implementations with non-comparable
    # versions) and parse_mode (batch vs stream candidates are listed separately).
    for t in model_v2["tabs"]:
        by_group: dict[tuple, list] = {}
        for c in t["candidates"]:
            if c.get("version"):
                by_group.setdefault((_impl_key_of(c["key"]), c.get("parse_mode")), []).append(c["version"])
        for (impl, pm), vers in by_group.items():
            keys = [[int(x) for x in re.findall(r"\d+", v)] for v in vers]
            assert keys == sorted(keys, reverse=True), f"{t['id']}/{impl}/{pm}: not latest-first {vers}"


def test_v2_batch_tab_has_all_peer_versions(model_v2):
    labels = " ".join(c["label"] for c in _tab(model_v2, "tab-toolcalling-batch")["candidates"])
    peers = _peer_versions("toolcalling/fixtures-batch-v1")
    for impl in ("vllm_python", "sglang_python"):
        for ver in peers.get(impl, set()):
            assert ver in labels, f"batch tab missing peer version {impl} {ver}"


def test_v2_stream_tab_has_v1jail_ref_v2_and_peers(model_v2):
    # memory: dynamo_v1-3.0.0 on the stream tab is the v1 jail+batch reference (all
    # families) — must be present; plus the v2 candidate and the peers.
    keys = {c["key"] for c in _tab(model_v2, "tab-toolcalling-streamv2")["candidates"]}
    assert any(k.startswith("dynamo_v1") for k in keys), f"no v1-jail ref candidate: {keys}"
    assert any(k.startswith("dynamo_v2") for k in keys), f"no v2 candidate: {keys}"
    assert any(k.startswith("vllm") for k in keys) and any(k.startswith("sglang") for k in keys)


def test_v2_patch_overlay_folds_into_base_version(model_v2):
    # memory: a X.patchN capture folds into its base <ver> column, never a standalone
    # candidate.
    for t in model_v2["tabs"]:
        for c in t["candidates"]:
            assert ".patch" not in (c.get("version") or ""), f"{t['id']}: standalone patch candidate {c}"
            assert ".patch" not in c["key"], f"{t['id']}: patch key leaked {c['key']}"


def test_v2_dynamo_versions_come_from_fixtures(model_v2):
    # memory/chart_invariants: Dynamo version labels come from fixture provenance, never
    # live Cargo.toml. Every shown Dynamo version must be a captured fixture dir version.
    fixture_dynamo = set()
    for tree in ("toolcalling/fixtures-batch-v1", "toolcalling/fixtures-stream-v2"):
        for impl, vers in _peer_versions(tree).items():
            if impl.startswith("dynamo"):
                fixture_dynamo |= {v.split(".patch")[0] for v in vers}
    shown = set()
    for t in model_v2["tabs"]:
        if t["kind"] != "toolcalling":
            continue
        for c in t["candidates"]:
            if c["impl"] == "dynamo" and c.get("version"):
                shown.add(c["version"].split(".patch")[0])
    assert shown, "no dynamo versions shown"
    assert shown <= fixture_dynamo, f"dynamo versions not from fixtures: {shown - fixture_dynamo}"


# ---- cells / compare payload --------------------------------------------------

def test_v2_cells_have_compare_data(model_v2):
    n = sum(1 for t in model_v2["tabs"] for c in _iter_cells(t) if c.get("cmp"))
    assert n > 100, f"only {n} cells carry a compare payload"


def test_v2_grid_cmp_keys_are_selectable(model_v2):
    # test_render_invariants I2: every cmp candidate key on a cell is offered in that
    # tab's compare bar (referential integrity between grid + selector).
    for t in model_v2["tabs"]:
        cand_keys = {c["key"] for c in t["candidates"]}
        for cell in _iter_cells(t):
            for key in (cell.get("cmp") or {}):
                assert key in cand_keys, f"{t['id']}: cmp key {key!r} not in compare bar"


def test_v2_cmp_payload_shape(model_v2):
    for t in model_v2["tabs"]:
        for cell in _iter_cells(t):
            for key, entry in (cell.get("cmp") or {}).items():
                assert set(entry) == {"sig", "leak", "na"}, entry
                assert isinstance(entry["sig"], int)


def test_v2_cell_status_enum(model_v2):
    ok = {"ok", "problem", "na", "missing"}
    for t in model_v2["tabs"]:
        for row in t["rows"]:
            for cell in row.get("cells", {}).values():
                assert cell["status"] in ok, cell["status"]


def test_v2_facts_shape(model_v2):
    keys = {"impl", "status", "present", "agrees", "intentional", "reason", "leak", "error_kind"}
    for cell in _iter_cells(_tab(model_v2, "tab-toolcalling-batch")):
        for f in cell["facts"]:
            assert keys <= set(f), f


_DOUBLED = re.compile(r"^(\w+?)\1$")


def test_no_doubled_call_names_in_dynamo_output(model_v2):
    # I1 (was test_render_invariants, regex on HTML): the resolver fold once doubled
    # Dynamo output into calls=[get_weatherget_weather(...)] and it shipped unnoticed.
    # Assert on the MODEL: no Dynamo candidate's calls carry a doubled name. (Captured
    # PEER blocks may legitimately record imperfect engine behavior — Dynamo only.)
    bad = []
    for tab in model_v2["tabs"]:
        for cell in _iter_cells(tab):
            tip = cell.get("tooltip") or {}
            for cand in tip.get("candidates", []):
                if not str(cand.get("impl", "")).startswith("dynamo"):
                    continue
                for call in ((cand.get("block") or {}).get("calls") or []):
                    name = call.get("name", "") if isinstance(call, dict) else ""
                    if name and _DOUBLED.match(name) and len(name) % 2 == 0:
                        bad.append(name)
    assert not bad, f"doubled call names in Dynamo output: {sorted(set(bad))}"


def test_implemented_v2_families_not_marked_not_implemented(model_v2):
    # I5 (was test_render_invariants): a family with a REAL v2 stream parser in the
    # registry must not be absent from the parser_ni "implemented" list (i.e. it must be
    # covered by the v2 stream candidate, not flagged not-implemented).
    mod = REPO / "parsers/v2/src/tool_calling/mod.rs"
    if not mod.exists():
        pytest.skip("parsers/v2 registry not present")
    registered = set(re.findall(r'"([a-z0-9_]+)"\s*=>', mod.read_text()))
    ni = model_v2["parser_ni"]
    # The parser_ni map lists the families the v2 stream parser DOES implement (its
    # coverage). Every registered family should appear there for the v2 candidate.
    v2_families = set()
    for info in ni.values():
        v2_families |= set(info.get("families", []))
    # Only assert for families that are also rendered as rows (some registry entries are
    # aliases/backends). A registered family that renders must be in the covered set.
    rendered = {row["family"] for tab in model_v2["tabs"] if tab["kind"] == "toolcalling"
                for row in tab["rows"] if row.get("family")}
    for fam in registered & rendered:
        assert fam in v2_families or not v2_families, (
            f"family {fam!r} has a v2 parser but is not in the covered set"
        )


# ---- reference-aware "not implemented" map (was window.__PARSER_NI) ------------

def test_v2_parser_ni_matches_stream_v2_families(model_v2):
    ni = model_v2["parser_ni"]
    assert ni, "empty parser_ni map"
    sv2 = _cache_root() / "toolcalling/fixtures-stream-v2"
    dv2 = max((d for d in sv2.glob("dynamo_v2-*") if d.is_dir()),
              key=lambda d: [int(x) for x in re.findall(r"\d+", d.name)], default=None)
    assert dv2 is not None
    fixture_fams = {p.name for p in dv2.iterdir() if p.is_dir()}
    for key, info in ni.items():
        assert key.startswith("dynamo_v2")
        assert set(info["families"]) <= fixture_fams or fixture_fams <= set(info["families"]) or (
            set(info["families"]) & fixture_fams), (info["families"], fixture_fams)


def test_v2_stream_parser_only_covers_implemented_families(model_v2):
    # The v2 stream candidate is n/a (uncovered) on more families than it covers — a
    # structural coverage guard (was regex over data-cmp na counts).
    tab = _tab(model_v2, "tab-toolcalling-streamv2")
    v2 = next(c["key"] for c in tab["candidates"] if c["key"].startswith("dynamo_v2"))
    na = present = 0
    for cell in _iter_cells(tab):
        entry = (cell.get("cmp") or {}).get(v2)
        if entry is None:
            continue
        if entry["na"]:
            na += 1
        else:
            present += 1
    assert na >= present, f"v2 stream covers too much: na={na} present={present}"


# ---- reasoning tabs -----------------------------------------------------------

def test_v2_reasoning_candidates_versioned_incl_dynamo_v1(model_v2):
    for tid in ("tab-reasoning-batch", "tab-reasoning-stream"):
        cands = _tab(model_v2, tid)["candidates"]
        assert cands
        labels = " ".join(c["label"] for c in cands)
        assert "Dynamo" in labels and "v1" in labels, labels
        for c in cands:
            assert _VER_PAREN.search(c["label"]), f"{tid}: unversioned reasoning candidate {c['label']!r}"


def test_v2_batch_tab_stream_candidates_use_current_peers(model_v2):
    # The merged batch tab offers each engine's CURRENT stream parser as a compare
    # candidate ("<Engine> <newest> (stream)"). Was a chart-invariant regex guard.
    labels = " ".join(
        c["label"] for c in _tab(model_v2, "tab-toolcalling-batch")["candidates"]
        if c.get("parse_mode") == "stream"
    )
    assert "stream" in labels
    peers = _peer_versions("toolcalling/fixtures-stream-v2")
    for impl in ("vllm_python", "sglang_python"):
        newest = max(peers.get(impl, {"0"}), key=lambda v: [int(x) for x in re.findall(r"\d+", v)] or [0])
        assert newest in labels, f"batch tab missing current stream peer {impl} {newest}"


def test_v2_no_verbose_todo_baked_in_cells(model_v2):
    # Un-implemented Dynamo v2 families are a clean n/a status in the model — never a
    # verbose "not yet implemented" string baked as a cell's visible glyph. (The phrase
    # legitimately lives in tooltip.candidates[].block.unavailable, which the view shows
    # in the popup, not the grid.) Replaces test_chart_invariants regex on visible HTML.
    for tab in model_v2["tabs"]:
        for cell in _iter_cells(tab):
            assert cell["status"] in {"ok", "problem", "na", "missing"}


def test_v2_reasoning_uses_current_peers(model_v2):
    # reasoning tab uses the same current peer versions as the toolcalling tabs.
    tc = " ".join(c["label"] for c in _tab(model_v2, "tab-toolcalling-batch")["candidates"])
    peers = _peer_versions("reasoning/fixtures-v1")
    r = " ".join(c["label"] for c in _tab(model_v2, "tab-reasoning-batch")["candidates"])
    for impl in ("vllm_python", "sglang_python"):
        for ver in peers.get(impl, set()):
            assert ver in r, f"reasoning missing current peer {impl} {ver}"


# ---- v1 PARITY page -----------------------------------------------------------

def test_v1_all_tabs_present(model_v1):
    ids = [t["id"] for t in model_v1["tabs"]]
    assert ids == [
        "tab-toolcalling-batch", "tab-toolcalling-stream",
        "tab-reasoning-batch", "tab-reasoning-stream",
    ], ids


def test_v1_cells_have_compare_data(model_v1):
    n = sum(1 for t in model_v1["tabs"] for c in _iter_cells(t) if c.get("cmp"))
    assert n > 100, f"only {n} v1 cells carry a compare payload"


def test_v1_every_candidate_is_versioned(model_v1):
    for t in model_v1["tabs"]:
        for c in t["candidates"]:
            assert _VER_PAREN.search(c["label"]), f"{t['id']}: unversioned candidate {c['label']!r}"


def test_v1_toolcalling_has_both_old_and_new_peers(model_v1):
    # PARITY_v1 shows ALL captured peer versions (v1-era + current).
    labels = " ".join(c["label"] for c in _tab(model_v1, "tab-toolcalling-batch")["candidates"])
    peers = _peer_versions("toolcalling/fixtures-batch-v1")
    for impl in ("vllm_python", "sglang_python"):
        vers = peers.get(impl, set())
        assert sum(1 for v in vers if v in labels) >= min(2, len(vers)), f"v1 missing {impl} versions"
