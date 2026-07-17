# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Browser-level smoke test for the conformance matrix (audit D4).

The recent regressions were browser-behavior regressions (hover tooltips, parser
radios) that the string-level template tests can't catch. This renders the real
table and drives a headless browser: hover a cell and assert its tooltip becomes
visible; assert the vLLM Rust parser option shows on a tool-calling tab and hides
on Reasoning (which has no vLLM Rust column).

Skips when Selenium or headless Chrome aren't available, so it adds no hard test
dependency — it runs where a browser exists and is a no-op otherwise.
"""
import shutil
import subprocess
import time
from pathlib import Path

import pytest

selenium = pytest.importorskip("selenium")
from selenium import webdriver  # noqa: E402
from selenium.webdriver.chrome.options import Options  # noqa: E402

UTILS = Path(__file__).resolve().parents[1]
REPO = UTILS.parents[1]

pytestmark = pytest.mark.skipif(
    not any(shutil.which(b) for b in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser")),
    reason="no headless Chrome available",
)


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    out = tmp_path_factory.mktemp("d4") / "table.html"
    subprocess.run(
        [str(UTILS / "render_table_v2.sh"), "--output", str(out)],
        check=True, cwd=REPO, capture_output=True, text=True,
    )
    return out


@pytest.fixture(scope="module")
def driver(rendered):
    opts = Options()
    for a in ("--headless=new", "--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage", "--window-size=1600,1200"):
        opts.add_argument(a)
    try:
        d = webdriver.Chrome(options=opts)
    except Exception as exc:  # noqa: BLE001 — environment without a usable driver
        pytest.skip(f"could not start Chrome webdriver: {exc}")
    d.get(f"file://{rendered}")
    d.implicitly_wait(2)
    yield d
    d.quit()


def test_hover_shows_tooltip(driver):
    """Hovering a detail cell makes its `.ttip` visible (`.ttip-visible`)."""
    driver.execute_script(
        "const v=document.querySelector('[data-view-detailed]'); if(v && !v.checked){v.checked=true; v.dispatchEvent(new Event('change'));}"
    )
    # Find a cell that actually has a tooltip, fire the hover event the page listens for.
    found = driver.execute_script(
        """
        const tab = document.querySelector('.tab-panel.active') || document;
        for (const cell of tab.querySelectorAll('td.cell')) {
          if (cell.querySelector('.ttip')) {
            cell.dispatchEvent(new PointerEvent('pointerenter', {bubbles: true}));
            return true;
          }
        }
        return false;
        """
    )
    assert found, "no cell with a tooltip found"
    deadline = time.time() + 3
    visible = False
    while time.time() < deadline:
        visible = driver.execute_script(
            "return !!document.querySelector('.ttip.ttip-visible');"
        )
        if visible:
            break
        time.sleep(0.1)
    assert visible, "tooltip did not become visible on hover"


def test_compare_candidates_are_per_tab(driver):
    """Each tab's compare control carries its own candidate rows: the merged Tool
    Calling (batch data) tab offers a vLLM Rust stream candidate; Reasoning does not.
    (The candidates were `.chip` elements before the compare-bar rework (#98/#105)
    replaced them with `.cmprow-label[data-cand]` rows.)"""
    def cand_keys():
        return driver.execute_script(
            "const p=document.querySelector('.tab-panel.active .cmpctl');"
            "return p?Array.from(p.querySelectorAll('.cmprow-label[data-cand]')).map(c=>c.dataset.cand):[];"
        )

    def click_tab(panel_id):
        driver.execute_script(
            "document.querySelector(arguments[0]).click();",
            f'.tab-button[data-tab-target="{panel_id}"]',
        )
        time.sleep(0.2)

    click_tab("tab-toolcalling-batch")
    keys = cand_keys()
    assert any("vllm_rust" in k for k in keys), "merged tab should offer a vLLM Rust candidate"
    click_tab("tab-reasoning-batch")
    keys = cand_keys()
    assert keys and not any("vllm_rust" in k for k in keys), (
        "Reasoning should have candidates but no vLLM Rust"
    )


def test_compare_shows_one_marker_per_cell(driver):
    """In Details view a compare cell shows exactly one marker — the JS-filled
    `.cmp-marker` — and the legacy per-engine `.cell-marker` spans stay hidden, so
    nothing overlaps (the B7 CSS-order regression that garbled markers)."""
    driver.execute_script(
        "const v=document.querySelector('[data-view-detailed]'); if(v && !v.checked){v.checked=true; v.dispatchEvent(new Event('change'));}"
    )
    time.sleep(0.2)
    result = driver.execute_script(
        """
        const tab = document.querySelector('.tab-panel.active') || document;
        const vis = (el) => el && el.offsetParent !== null
            && getComputedStyle(el).display !== 'none';
        let overlap = 0, cmpShown = 0;
        for (const cell of tab.querySelectorAll('td.cell[data-cmp]')) {
          const legacy = Array.from(cell.querySelectorAll('.cell-marker')).some(vis);
          const cmp = cell.querySelector('.cmp-marker');
          if (legacy) overlap++;
          if (vis(cmp) && cmp.textContent.trim()) cmpShown++;
        }
        return {overlap, cmpShown};
        """
    )
    assert result["overlap"] == 0, (
        f"{result['overlap']} cell(s) still show a legacy marker alongside the compare marker"
    )
    assert result["cmpShown"] > 0, "no compare marker is visible in Details view"


def test_overview_hides_compare_column(driver):
    """In Overview (Detailed off) the compare bar shows only the Reference picker;
    the CMP checkboxes + header are hidden, because an overview cell's color is
    leak-only (depends on the Reference, not the Compares). Turning Detailed on
    reveals the CMP column again — the selections themselves are preserved."""
    driver.execute_script(
        "document.querySelector('.tab-button[data-tab-target=\"tab-toolcalling-batch\"]').click();"
    )
    time.sleep(0.2)

    def set_detailed(on):
        driver.execute_script(
            "const v=document.querySelector('[data-view-detailed]');"
            "if(v && v.checked!==arguments[0]){v.checked=arguments[0]; v.dispatchEvent(new Event('change'));}",
            on,
        )
        time.sleep(0.2)

    def cmp_box_visible():
        # offsetParent is null when the element (or an ancestor) is display:none.
        return driver.execute_script(
            "const p=document.querySelector('.tab-panel.active .cmpctl');"
            "const box=p && p.querySelector('.cmprow:not(.cmphd) .cmprow-cmp');"
            "return box ? (box.offsetParent !== null) : null;"
        )

    def ref_box_visible():
        return driver.execute_script(
            "const p=document.querySelector('.tab-panel.active .cmpctl');"
            "const r=p && p.querySelector('.cmprow:not(.cmphd) .cmprow-ref');"
            "return r ? (r.offsetParent !== null) : null;"
        )

    set_detailed(False)
    assert cmp_box_visible() is False, "CMP column should be hidden in Overview"
    assert ref_box_visible() is True, "REF picker must still show in Overview"
    set_detailed(True)
    assert cmp_box_visible() is True, "CMP column should reappear in Details"


def _click_tab(driver, panel_id):
    driver.execute_script(
        "document.querySelector(arguments[0]).click();",
        f'.tab-button[data-tab-target="{panel_id}"]',
    )
    time.sleep(0.2)


def _set_transpose(driver, on):
    driver.execute_script(
        "const t=document.querySelector('[data-transpose-toggle]');"
        "if(t && t.checked!==arguments[0]){t.checked=arguments[0]; t.dispatchEvent(new Event('change'));}",
        on,
    )
    time.sleep(0.2)


def test_transpose_builds_mirror_and_colors(driver):
    """Toggling Transpose builds a mirror in the active panel: models become rotated
    columns (th.tcol-model), cases become rows (th.trow-case), and the cloned cells
    are colored by the SAME compare engine (cmp-eq/cmp-leak/cmp-na) — not left blank.
    This is the DIS-2280 integration with #98's reference/compare model."""
    _click_tab(driver, "tab-toolcalling-batch")
    _set_transpose(driver, True)
    info = driver.execute_script(
        """
        const p = document.querySelector('.tab-panel.active');
        const tt = p.querySelector('table[data-transpose-table]');
        if (!tt) return {built:false};
        const cells = tt.querySelectorAll('td.cell');
        let colored = 0;
        cells.forEach(function (c) {
          if (c.classList.contains('cmp-eq') || c.classList.contains('cmp-leak') || c.classList.contains('cmp-na')) colored++;
        });
        return {
          built: true,
          models: tt.querySelectorAll('th.tcol-model').length,
          rows: tt.querySelectorAll('th.trow-case').length,
          cells: cells.length,
          colored: colored,
        };
        """
    )
    assert info["built"], "transposed mirror table was not built"
    assert info["models"] > 1, "expected multiple rotated model columns"
    assert info["rows"] > 1, "expected multiple case rows"
    assert info["cells"] > 0 and info["colored"] == info["cells"], (
        f"every cloned cell should be colored by applyCtl, got {info['colored']}/{info['cells']}"
    )


def test_transpose_does_not_double_overview_counts(driver):
    """The mirror's cloned cells must not inflate the overview counts (both applyCtl
    and updateOverviewStats skip cells inside [data-transpose-table])."""
    _click_tab(driver, "tab-toolcalling-batch")
    _set_transpose(driver, False)
    counts = "const p=document.querySelector('.tab-panel.active');return Array.from(p.querySelectorAll('[data-overview-count]')).map(function(e){return e.textContent;});"
    before = driver.execute_script(counts)
    _set_transpose(driver, True)
    after = driver.execute_script(counts)
    assert before == after, f"overview counts changed when transposing: {before} -> {after}"


def test_transpose_recolors_on_reference_change(driver):
    """Picking a different Reference recolors the mirror too — applyCtl covers it
    because the mirror lives in the same panel."""
    _click_tab(driver, "tab-toolcalling-batch")
    _set_transpose(driver, True)
    snap = "const tt=document.querySelector('.tab-panel.active table[data-transpose-table]');return Array.from(tt.querySelectorAll('td.cell')).map(function(c){return c.className;});"
    before = driver.execute_script(snap)
    # Try EVERY alternate Reference until one recolors: adjacent capture
    # generations (e.g. Dynamo v1 batch 3.0.0 vs 5.0.0 — old version dirs are
    # kept as history) can be near-identical, so the FIRST alternate may
    # legitimately produce the same colors.
    n_refs = driver.execute_script(
        "const ctl=document.querySelector('.tab-panel.active .cmpctl');"
        "return ctl.querySelectorAll('input.cmp-ref').length;"
    )
    assert n_refs > 1, "no alternate Reference available to select"
    recolored = False
    for idx in range(n_refs):
        changed = driver.execute_script(
            """
            const idx = arguments[0];
            const ctl = document.querySelector('.tab-panel.active .cmpctl');
            const r = Array.from(ctl.querySelectorAll('input.cmp-ref'))[idx];
            if (!r || r.checked || r.disabled) return false;
            r.checked = true;
            r.dispatchEvent(new Event('change', {bubbles: true}));
            return true;
            """,
            idx,
        )
        if not changed:
            continue
        time.sleep(0.3)
        if driver.execute_script(snap) != before:
            recolored = True
            break
    assert recolored, "transposed cells did not recolor for ANY alternate Reference"


def test_transpose_honors_collapsed_case_group(driver):
    """A case group collapsed via the column toggle in the original table stays hidden
    (as rows) in the transposed mirror — the mirror carries data-col-hide-group and
    re-applies the column state on build (regression: #87 review)."""
    _click_tab(driver, "tab-toolcalling-batch")
    _set_transpose(driver, False)
    key = driver.execute_script(
        """
        const p = document.querySelector('.tab-panel.active');
        const subKeys = new Set(Array.from(p.querySelectorAll('th.case-sub[data-col-hide-group]'))
          .map(function (e) { return e.dataset.colHideGroup; }));
        const btn = Array.from(p.querySelectorAll('[data-col-toggle]'))
          .find(function (b) { return subKeys.has(b.dataset.colToggle); });
        if (!btn) return null;
        btn.click();  // collapse this case group
        return btn.dataset.colToggle;
        """
    )
    assert key, "no case-group column toggle found"
    _set_transpose(driver, True)
    hidden = driver.execute_script(
        """
        const key = arguments[0];
        const tt = document.querySelector('.tab-panel.active table[data-transpose-table]');
        const rows = tt.querySelectorAll('tr[data-col-hide-group="' + key + '"]');
        if (!rows.length) return null;
        return Array.from(rows).every(function (r) { return r.classList.contains('col-hidden'); });
        """,
        key,
    )
    assert hidden is True, f"transposed rows for collapsed group {key} should be hidden"
