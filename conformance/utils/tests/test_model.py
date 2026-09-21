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
import copy
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

UTILS = Path(__file__).resolve().parents[1]
REPO = UTILS.parents[1]
if str(UTILS / "src") not in sys.path:
    sys.path.insert(0, str(UTILS / "src"))

from fixture_snapshot import fixture_snapshot_root  # noqa: E402
from capture_stimulus import capture_input  # noqa: E402
import model as model_mod  # noqa: E402
import generate_conformance_table as table  # noqa: E402
from dynamo_version import dynamo_v2_label, select_capture_label  # noqa: E402


def _resolve_cache_root() -> Path:
    """Manifest-pinned snapshot dir, never the mutable compatibility links."""
    return fixture_snapshot_root()


_CACHE_ROOT = _resolve_cache_root()


def _cache_root() -> Path:
    return _CACHE_ROOT


_HAVE_FIXTURES = (_cache_root() / "toolcalling").is_dir()
pytestmark = pytest.mark.skipif(
    not _HAVE_FIXTURES, reason="fixtures not extracted (run extract_fixtures.py)"
)

_MODEL_RE = re.compile(
    r'<script type="application/json" id="conformance-model">(.*?)</script>', re.S
)


def _read_model_raw(page_path: Path, render_script: str) -> dict:
    if not page_path.exists():
        subprocess.run([str(UTILS / render_script)], check=True, capture_output=True, cwd=REPO)
    html = page_path.read_text(encoding="utf-8")
    m = _MODEL_RE.search(html)
    assert m, f"{page_path.name}: no conformance-model blob"
    return json.loads(m.group(1))


def _read_model(page_path: Path, render_script: str) -> dict:
    # The blob is compacted (schema 2); every structural guard asserts on the
    # HYDRATED shape — exactly what the JS view renders after hydratePage.
    return model_mod.hydrate_page(_read_model_raw(page_path, render_script))


@pytest.fixture(scope="module")
def model_v2_raw() -> dict:
    return _read_model_raw(REPO / "conformance/CONFORMANCE_v2.html", "render_table_v2.sh")


@pytest.fixture(scope="module")
def model_v2() -> dict:
    return _read_model(REPO / "conformance/CONFORMANCE_v2.html", "render_table_v2.sh")


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


# A version token then a parenthesized mode, e.g. "0.1.24 (stream)". `+` is allowed in
# the token because a capture can be filed under a change-scoped label ("0.1.24+pr163")
# to compare one code state against another within a single release — see
# conformance/utils/src/dynamo_version.py.
_VER_PAREN = re.compile(r"\b\d[\w.+-]*\s+\([^)]+\)\s*$")

# ---- schema + shape -----------------------------------------------------------

def test_v2_schema_and_meta(model_v2):
    assert model_v2["schema"] == 2
    meta = model_v2["meta"]
    assert meta["title"] and meta["stamp"] and meta["command"]


# ---- schema-2 compaction (model.py _compact_page <-> hydrate_page) --------------

def test_compaction_roundtrip_is_identity():
    # compact -> hydrate must reproduce the original page exactly (modulo the added
    # compaction keys), or the JS view would render different VALUES than Python
    # computed. Synthetic page exercising every compaction move: interned strings,
    # cand_meta dedup (incl. a conflicting label kept inline), fixture_href prefix,
    # and the composed-head drop.
    long_a = "explanation text repeated across cells x"
    long_b = "another repeated reason string y"
    def cand(key, label, expl):
        return {"key": key, "label": label, "impl": "dynamo", "version": "1",
                "parse_mode": "batch", "block": {"explanation": expl}}
    def cell(case, fam, href, label2):
        return {
            "kind": "cell", "case_id": case, "family": fam, "sub": "1",
            "col_group": "core", "band": "b", "fixture_href": href, "status": "ok",
            "cmp": None, "known_divergence": False,
            "facts": [{"impl": "dynamo", "reason": long_b}],
            "tooltip": {
                "head": f"{case} — {fam}", "description": long_a,
                "input": {"kind": "text", "text": long_a},
                "candidates": [cand("k1", "L1", long_a), cand("k2", label2, long_b)],
                "baseline": None, "reasons": [{"label": long_b, "reason": long_a}],
                "dynamo_notes": [[long_b, long_a]], "refs": [["r", 7]],
                "leak_note": None, "na_note": None,
            },
        }
    tabs = [{
        "id": "t", "kind": "toolcalling", "label": "T", "rows": [
            {"model_label": "m", "cells": {
                "1": cell("C.1", "famA", "https://x/fixtures/famA/1.yaml", "L2"),
                "2": cell("C.2", "famB", "https://x/fixtures/famB/2.yaml", "L2-conflict"),
            }},
        ], "columns": [], "candidates": [], "stats": {},
    }]
    original = copy.deepcopy(tabs)
    page = model_mod.build_page({"title": "t", "stamp": "s", "command": "c"}, tabs)
    # Compacted on the wire: strings interned, meta hoisted, head dropped.
    assert page["strings"], "expected interned strings"
    assert page["tabs"][0]["cand_meta"]["k1"]["label"] == "L1"
    assert "label" not in page["tabs"][0]["cand_meta"].get("k2", {}), \
        "conflicting label must stay inline"
    assert page["tabs"][0]["fixture_href_base"].endswith("/fixtures/")
    first_tip = page["tabs"][0]["rows"][0]["cells"]["1"]["tooltip"]
    assert "head" not in first_tip
    # Round-trip: hydrate restores every VALUE the producers computed.
    hydrated = model_mod.hydrate_page(json.loads(json.dumps(page)))
    for tab_orig, tab_hyd in zip(original, hydrated["tabs"]):
        for row_orig, row_hyd in zip(tab_orig["rows"], tab_hyd["rows"]):
            assert row_orig["cells"] == row_hyd["cells"]


def test_v2_blob_is_compacted_and_hydrates_clean(model_v2_raw):
    # The real rendered blob actually uses the compaction (page-size guard) ...
    assert model_v2_raw.get("strings"), "blob should carry an interned-string table"
    assert any(t.get("cand_meta") for t in model_v2_raw["tabs"])
    # ... and hydration leaves no unresolved index anywhere (every interned slot is
    # a string again, every tooltip candidate has its meta back).
    hydrated = model_mod.hydrate_page(copy.deepcopy(model_v2_raw))
    for container, key in model_mod._iter_intern_slots(hydrated):
        v = model_mod._slot_get(container, key)
        assert not isinstance(v, int) or isinstance(v, bool), (key, v)
    for tab in hydrated["tabs"]:
        for row in tab["rows"]:
            for cell in (row.get("cells") or {}).values():
                tip = cell.get("tooltip")
                for cand in (tip.get("candidates") or []) if tip else []:
                    assert cand.get("label"), f"candidate {cand.get('key')} lost its label"


def test_v2_all_tabs_present(model_v2):
    ids = [t["id"] for t in model_v2["tabs"]]
    assert ids == [
        "tab-toolcalling-batch", "tab-toolcalling-streamv2",
        "tab-reasoning-batch", "tab-reasoning-stream", "tab-unified",
    ], ids


def test_v2_tab_labels_show_parser_generation(model_v2):
    labels = {tab["id"]: tab["label"] for tab in model_v2["tabs"]}

    assert labels["tab-toolcalling-batch"].startswith("Tool Calling v1")
    assert labels["tab-toolcalling-streamv2"].startswith("Tool Calling v1")
    assert labels["tab-unified"].startswith("Unified v2")


def test_unified_numeric_case_ids_use_dash_everywhere(model_v2):
    """Fixture IDs, headers, columns, and glossary rows share one numeric format."""
    tab = _tab(model_v2, "tab-unified")
    numeric = {
        "guided_json_quoted_bare_header_in_answer": "35-1",
        "guided_json_quoted_bare_tool_header_in_answer": "muse-1",
        "guided_json_quoted_bare_header_after_payload": "35-2",
        "guided_json_bare_tool_header_recovers_inside_a_thought": "34-7",
    }
    columns = {column["sub"]: column["label"] for column in tab["columns"]}
    glossary_ids = {
        short_id
        for group in tab["glossary"]
        for short_id, _description in group["rows"]
    }
    cells = {cell["sub"]: cell for cell in _iter_cells(tab) if cell["sub"] in numeric}

    assert set(cells) == set(numeric)
    for scenario, short_id in numeric.items():
        full_id = f"UNIFIED.{short_id}"
        assert columns[scenario] == short_id
        assert short_id in glossary_ids
        assert cells[scenario]["case_id"] == full_id
        assert cells[scenario]["tooltip"]["head"].startswith(f"{full_id} (")


def test_unified_duplicate_notes_and_deepseek_prefilled_captures(model_v2):
    tab = _tab(model_v2, "tab-unified")
    for row in tab["rows"]:
        if row["family"] != "muse_glimmer":
            cell = row["cells"]["guided_json_quoted_bare_tool_header_in_answer"]
            assert cell["status"] == "na"
            for field in ("description", "na_note"):
                assert cell["tooltip"][field].startswith("This is a duplication of UNIFIED.35-1")
        if row["family"] == "deepseek_v41":
            for scenario in ("prefilled_reasoning_with_tool", "prefilled_reasoning_then_text_then_tool", "prefilled_reasoning_then_text"):
                cell = row["cells"][scenario]
                assert cell["status"] != "na"
                assert cell["tooltip"]["init"]["starting_state"] == "Reasoning"
                assert cell["cmp"]["dynamo"].get("na", 0) == 0
                assert cell["cmp"]["dynamo"]["sig"] == cell["cmp"]["golden"]["sig"]


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
            # The golden oracle is authored, not captured from an engine build, so it
            # carries no version (the unified tab measures every engine against it).
            if c.get("key") == "golden":
                continue
            assert _VER_PAREN.search(c["label"]), f"{t['id']}: unversioned candidate {c['label']!r}"


def test_v2_exactly_one_reference_bucket_per_tab(model_v2):
    for t in model_v2["tabs"]:
        refs = [c for c in t["candidates"] if c["default_bucket"] == "A"]
        assert len(refs) == 1, f"{t['id']}: expected one bucket-A reference, got {len(refs)}"


def test_unified_tab_keeps_every_captured_vllm_parser_version(model_v2):
    """The Unified tab must show both historical Combined and current native captures."""
    tab = _tab(model_v2, "tab-unified")
    labels = [candidate["label"] for candidate in tab["candidates"]]
    assert "vLLM Rust 0.26.0 (stream, Combined & Unified)" in labels
    assert "vLLM Rust 0.25.1 (stream, Combined & Unified)" in labels
    assert "vLLM Python 0.25.1 (batch, Combined)" in labels
    assert "vLLM Python 0.26.0 (batch, Combined)" in labels

    muse = next(row for row in tab["rows"] if row["family"] == "muse_glimmer")
    peer_keys = {candidate["key"] for candidate in tab["candidates"] if candidate["impl"] == "vllm"}
    assert all(
        all(cell["cmp"][key].get("na") == 1 for key in peer_keys)
        for cell in muse["cells"].values()
    )
    muse_tip = next(iter(muse["cells"].values()))["tooltip"]
    native = next(candidate for candidate in muse_tip["candidates"] if candidate["key"] == "vllm_rust@0.26.0")
    assert native["block"]["unavailable"] == "vLLM Rust 0.26.0 (stream, Combined & Unified) has no parser for muse_glimmer"


def test_unified_default_dynamo_keeps_capture_identity_internal_and_release_history_visible(model_v2):
    tab = _tab(model_v2, "tab-unified")
    dynamo = next(candidate for candidate in tab["candidates"] if candidate["key"] == "dynamo")
    release = next(candidate for candidate in tab["candidates"] if candidate["key"] == "dynamo@0.6.0")

    captures = {}
    for path in (_cache_root() / "unified").glob("dynamo_v2-*/*/*.yaml"):
        version = path.parent.parent.name.removeprefix("dynamo_v2-")
        doc = yaml.safe_load(path.read_text())
        layer = captures.setdefault(version, {
            "complete_snapshot": (path.parent.parent / "capture-snapshot.json").is_file(),
            "records": {},
        })
        layer["records"].update({
            f"{path.parent.name}/{key}": doc.get("capture_provenance") for key in doc["cases"]
        })
    requested = select_capture_label(REPO, captures)
    if requested != "0.6.1":
        pytest.skip("the history backfill has no capture produced by this checkout")
    assert requested == "0.6.1"
    assert dynamo["version"] == requested
    assert dynamo["label"] == "Dynamo v2 Rust 0.6.1 (stream, Combined & Unified)"
    assert "+source." not in dynamo["label"]
    assert all("+source." not in candidate["key"] for candidate in tab["candidates"])
    assert dynamo["default_bucket"] == "A"
    assert release["label"] == "Dynamo v2 Rust 0.6.0 (stream, Combined & Unified)"
    assert release["default_bucket"] == "C"


@pytest.mark.parametrize("changed_field,value,missing_family", [
    (None, None, None), ("starting_state", "Reasoning", None),
    ("tool_output_mode", "GuidedJson", None), ("named_tool", "get_weather", None),
    ("starting_state", "Reasoning", "deepseek_v4"),
    ("starting_state", "Reasoning", "deepseek_v41"),
])
def test_unified_grammar_header_preserves_each_family_config(tmp_path, monkeypatch, changed_field, value, missing_family):
    scenario = "reason_only"
    generator = table.gen_unified_golden
    authored = {family: generator.build_cases(family) for family in ("deepseek_v4", "deepseek_v41")}
    monkeypatch.setattr(generator, "build_cases", authored.__getitem__)
    monkeypatch.setattr(table, "_unified_dynamo_label", lambda _captures: "0.6.0")
    monkeypatch.setattr(table, "_unified_base", lambda _root: tmp_path)
    scenarios = {scenario, "text_only"} if missing_family else {scenario}
    monkeypatch.setattr(generator, "CLEAN", [case for case in generator.CLEAN if case[0] in scenarios])
    monkeypatch.setattr(generator, "EDGE", [])
    configurations = {}
    for family in ("deepseek_v4", "deepseek_v41"):
        init = {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None}
        if family == "deepseek_v41" and changed_field:
            init[changed_field] = value
        configurations[family] = init
        if family == missing_family:
            fallback = generator.build_cases(family)[f"UNIFIED.{scenario}.{family}"]
            assert fallback["init"] == init
            key = table.unified_taxonomy.numbered_id("text_only")
            path = tmp_path / "inputs" / family / f"{key}.yaml"
            path.parent.mkdir(parents=True)
            path.write_text(yaml.safe_dump({"family": family, "cases": {key: {
                "scenario": "text_only", "description": "Visible text", "input": "answer",
                "init": init, "chunks": [{"delta_text": "answer"}],
            }}}))
            continue
        key = table.unified_taxonomy.numbered_id(scenario)
        records = {
            "inputs": {"scenario": scenario, "description": "Reasoning", "input": "thought",
                       "init": init, "chunks": [{"delta_text": "thought"}]},
            "golden": {"assembled": [{"kind": "reasoning", "text": "thought"}]},
        }
        for dirname, record in records.items():
            path = tmp_path / dirname / family / f"{key}.yaml"
            path.parent.mkdir(parents=True)
            path.write_text(yaml.safe_dump({"family": family, "cases": {key: record}}))
    tab = table._unified_tab_model(tmp_path, {})
    default = next(candidate for candidate in tab["candidates"] if candidate["default_bucket"] == "A")
    assert default["key"] == "dynamo"
    assert "Default Reference = <strong>Dynamo v2 Rust</strong>" in tab["toolbar_desc_html"]
    assert "Oracle = <strong>GOLDEN</strong>" in tab["toolbar_desc_html"]
    column = next(col for col in tab["columns"] if col["sub"] == scenario)
    assert column["init"] == (None if changed_field else configurations["deepseek_v4"])
    script = r"""
const fs = require('fs');
const vm = require('vm');
const context = {window: {}, document: {cookie: '', documentElement: {setAttribute() {}},
  querySelectorAll() {return [];}, addEventListener() {}, getElementById() {return null;}}};
vm.createContext(context);
const source = fs.readFileSync(process.argv[1], 'utf8');
vm.runInContext(source.replace('// --- Entry point',
  'window.audit = {columnGrammarModel, buildGrammarHtml, hydratePage};\n// --- Entry point'), context);
const page = JSON.parse(fs.readFileSync(0, 'utf8'));
context.window.audit.hydratePage(page);
const tab = page.tabs[0];
const column = tab.columns.find(col => col.sub === 'reason_only');
const model = context.window.audit.columnGrammarModel(tab, column);
process.stdout.write(JSON.stringify({model, html: context.window.audit.buildGrammarHtml(model)}));
"""
    result = subprocess.run(
        ["node", "-e", script, str(UTILS / "src/assets/conformance_view.js")],
        input=json.dumps(model_mod.build_page({}, [tab])), text=True, capture_output=True, check=True,
    )
    rendered = json.loads(result.stdout)
    for row in rendered["model"]["grammar"]:
        assert row["init"] == configurations[row["family"]]
        html_row = next(part for part in rendered["html"].split("<tr") if row["family"] in part)
        for field, setting in row["init"].items():
            label = f"{field}={'null' if setting is None else setting}"
            assert label in (html_row if changed_field else rendered["html"].split("<table")[0])
    assert rendered["html"].count('class="ttip-config"') == (2 if changed_field else 1)


def test_unified_selector_uses_source_checkout_with_or_without_staging(tmp_path, monkeypatch):
    monkeypatch.delenv("CONFORMANCE_DYNAMO_V2_LABEL", raising=False)
    monkeypatch.delenv("FRONTEND_CRATES_ROOT", raising=False)
    expected = dynamo_v2_label(REPO)
    assert table._unified_dynamo_label({}) == expected
    monkeypatch.setenv("FRONTEND_CRATES_ROOT", str(REPO))
    monkeypatch.setattr(table, "__file__", str(tmp_path / "tests/parity/generate_conformance_table.py"))
    assert table._unified_dynamo_label({}) == expected


@pytest.mark.parametrize(
    ("current_present", "capture_failure"),
    [(True, None), (False, None), (True, "error")],
)
def test_unified_source_selection_inherits_previous_family_capture(
    tmp_path, monkeypatch, current_present, capture_failure
):
    selected = "0.6.1"
    previous = "0.6.0"
    scenario, family = "text_only", "gemma4"
    generator = table.gen_unified_golden
    authored = generator.build_cases(family)[f"UNIFIED.{scenario}.{family}"]
    key = table.unified_taxonomy.numbered_id(scenario)
    monkeypatch.setattr(table, "_unified_dynamo_label", lambda _captures: selected)
    monkeypatch.setattr(table, "_unified_base", lambda _root: tmp_path)
    monkeypatch.setattr(generator, "CLEAN", [case for case in generator.CLEAN if case[0] == scenario])
    monkeypatch.setattr(generator, "EDGE", [])

    records = {
        "inputs": {"scenario": scenario, "description": authored["description"],
                   "input": authored["input"], "init": authored["init"], "tools": [],
                   "chunks": [{"delta_text": authored["input"]}]},
        "golden": {"assembled": authored["golden"]},
    }
    for version in [previous] + ([selected] if current_present else []):
        events = [{"kind": "text", "text": version}]
        records[f"dynamo_v2-{version}"] = (
            {capture_failure: "capture could not run"} if capture_failure and version == selected else
            {"assembled": events, "chunks": [{"expected": events}]}
        )
        records[f"dynamo_v2-{version}"]["capture_input"] = capture_input(records["inputs"])
    for dirname, record in records.items():
        path = tmp_path / dirname / family / f"{key}.yaml"
        path.parent.mkdir(parents=True)
        path.write_text(yaml.safe_dump({"family": family, "cases": {key: record}}))

    cases, _caps, versions = table._load_unified_fixtures(tmp_path)
    case = next(case for case in cases if case["family"] == family)
    assert versions["dynamo_v2"] == selected
    assert set(versions["dynamo_v2_all"]) == {previous, selected}
    assert not case["dynamo_missing"]
    expected_version = selected if current_present else previous
    assert case["dynamo"] == (
        [{"kind": "text", "text": expected_version}] if not capture_failure else []
    )

    tab = table._unified_tab_model(tmp_path, {})
    candidates = {candidate["key"]: candidate for candidate in tab["candidates"]}
    assert candidates["dynamo"]["version"] == selected
    assert candidates["dynamo"]["label"] == "Dynamo v2 Rust 0.6.1 (stream, Combined & Unified)"
    assert {key for key in candidates if key.startswith("dynamo@")} == {f"dynamo@{previous}"}
    cell = next(row for row in tab["rows"] if row["family"] == family)["cells"][scenario]
    current = next(candidate for candidate in cell["tooltip"]["candidates"] if candidate["key"] == "dynamo")
    assert current["label"] == candidates["dynamo"]["label"]
    assert current["version"] == selected
    assert {candidate["key"] for candidate in cell["tooltip"]["candidates"]} == set(candidates)
    assert set(cell["cmp"]) == set(candidates)
    assert (tmp_path / f"dynamo_v2-{previous}" / family / f"{key}.yaml").is_file()
    if capture_failure:
        assert current["block"] == {capture_failure: "capture could not run"}
        assert cell["status"] == "problem"
        assert cell["cmp"]["dynamo"]["err"] == 1
    else:
        assert current["block"]["events"] == [{"kind": "text", "text": expected_version}]
        assert "error" not in current["block"]


@pytest.mark.parametrize("impl,version,mode,want", [
    ("dynamo_v2", "0.6.0+source." + "a" * 64, "stream",
     "Dynamo v2 Rust 0.6.0 (stream)"),
    ("dynamo_v2", "0.6.1", "stream", "Dynamo v2 Rust 0.6.1 (stream)"),
    ("dynamo_v1", "8.2.2", "stream", "Dynamo v1 Rust 8.2.2 (jail+batch)"),
    ("vllm_python", "0.26.0", "batch", "vLLM Python 0.26.0 (batch)"),
])
def test_candidate_label_keeps_capture_identity_out_of_display(impl, version, mode, want):
    assert table._full_label(impl, version, mode) == want


def test_unified_tab_marks_uncomparable_vllm_cases_na(model_v2):
    """Historical output without its original request cannot establish parity."""
    tab = _tab(model_v2, "tab-unified")
    peer_keys = {candidate["key"] for candidate in tab["candidates"] if candidate["impl"] == "vllm"}
    assert peer_keys == {"vllm", "vllm_python@0.26.0", "vllm_rust", "vllm_rust@0.26.0"}
    for row in tab["rows"]:
        for key in peer_keys:
            unavailable = [cell["cmp"][key].get("na") == 1 for cell in row["cells"].values()]
            if row["family"] == "muse_glimmer":
                assert all(unavailable), f"{key} must say n/a for Muse"
                continue
            for scenario, is_unavailable in zip(row["cells"], unavailable):
                if not is_unavailable or row["cells"][scenario]["status"] == "na":
                    continue
                peer = next(candidate for candidate in row["cells"][scenario]["tooltip"]["candidates"] if candidate["key"] == key)
                reason = peer["block"]["unavailable"]
                assert (
                    "not captured at" in reason
                    or "has no parser" in reason
                    or reason.startswith(("Capture stimulus unavailable:", "Capture stimulus mismatch ("))
                ), reason
                assert "events" not in peer["block"]
                comparison = row["cells"][scenario]["cmp"][key]
                assert comparison == {"sig": 0, "leak": 0, "na": 1, "err": 0}

    gemma = next(row for row in tab["rows"] if row["family"] == "gemma4")
    for scenario in (
        "gemma4_guided_json_visible_call_prose_before_reasoning",
        "gemma4_guided_json_malformed_call_prefix_before_reasoning",
    ):
        for key in peer_keys:
            cell = gemma["cells"][scenario]
            assert cell["cmp"][key].get("na") == 1
            peer = next(candidate for candidate in cell["tooltip"]["candidates"] if candidate["key"] == key)
            assert "this case postdates that capture" in peer["block"]["unavailable"]


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
    def release_version(version: str) -> str:
        return version.split("+source.", 1)[0].split(".patch", 1)[0]

    fixture_dynamo = set()
    for tree in ("toolcalling/fixtures-batch-v1", "toolcalling/fixtures-stream-v2"):
        for impl, vers in _peer_versions(tree).items():
            if impl.startswith("dynamo"):
                fixture_dynamo |= {release_version(v) for v in vers}
    shown = set()
    for t in model_v2["tabs"]:
        if t["kind"] != "toolcalling":
            continue
        for c in t["candidates"]:
            if c["impl"] == "dynamo" and c.get("version"):
                shown.add(release_version(c["version"]))
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
                # `err` flags a candidate that ran and THREW (an `exception` block): not
                # `na`, so a threw-Reference vs a parsed peer reddens the cell.
                assert set(entry) == {"sig", "leak", "na", "err"}, entry
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


def test_v2_deepseek_v4_streamv2_parser_links_dsml(model_v2):
    # Migrated from test_stream_on_batch.test_dsv4_v2_parser_cell_links_dsml_parser, which
    # called g._parser_cell_html directly. Assert the same fact on the built model: the
    # deepseek_v4 streamv2 parser cell links the DSML parser source and is NOT flagged
    # unimplemented (the DeepSeek-v4 v2 stream parser exists, at dsml.rs).
    tab = _tab(model_v2, "tab-toolcalling-streamv2")
    htmls = [r["parser"]["html"] for r in tab["rows"]
             if r.get("family") == "deepseek_v4" and r.get("parser")]
    assert htmls, "no deepseek_v4 row with a parser cell in the streamv2 tab"
    html = htmls[0]
    assert "DeepSeekV4ToolStreamParser text path" in html
    assert "parsers/v2/src/tool_calling/dsml.rs" in html
    assert "not implemented" not in html


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
    # The registry owns which families Dynamo v2 implements. Keep the rendered model
    # aligned with that declaration instead of relying on a corpus-wide n/a ratio,
    # which changes whenever a supported family or case is added.
    tab = _tab(model_v2, "tab-toolcalling-streamv2")
    v2 = next(c["key"] for c in tab["candidates"] if c["key"].startswith("dynamo_v2"))
    registry = yaml.safe_load((UTILS / "src/parser_families.yaml").read_text())["families"]
    for row in tab["rows"]:
        family = row.get("family")
        if not family:
            continue
        assert family in registry, f"rendered family {family!r} is absent from the registry"
        entries = [
            (cell.get("cmp") or {}).get(v2)
            for cell in row.get("cells", {}).values()
            if cell.get("kind") == "cell"
        ]
        present = any(entry is not None and not entry["na"] for entry in entries)
        implemented = registry[family].get("dynamo_v2") is not None
        assert present == implemented, (
            f"family {family!r}: rendered Dynamo v2 coverage={present}, "
            f"registry implementation={registry[family].get('dynamo_v2')!r}"
        )


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
    peers = _peer_versions("reasoning/fixtures-v1")
    r = " ".join(c["label"] for c in _tab(model_v2, "tab-reasoning-batch")["candidates"])
    for impl in ("vllm_python", "sglang_python"):
        for ver in peers.get(impl, set()):
            assert ver in r, f"reasoning missing current peer {impl} {ver}"
