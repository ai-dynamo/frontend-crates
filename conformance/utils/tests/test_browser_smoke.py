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
import time

import pytest

selenium = pytest.importorskip("selenium")
from selenium import webdriver  # noqa: E402
from selenium.webdriver.chrome.options import Options  # noqa: E402

pytestmark = pytest.mark.skipif(
    not any(shutil.which(b) for b in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser")),
    reason="no headless Chrome available",
)


# Headless Chrome reports `(hover: hover)` as FALSE, so conformance.js takes its touch
# branch and never attaches the `pointerenter` listener that test_hover_shows_tooltip
# drives — the test could not pass here regardless of the page being correct. Force the
# hover-capable branch by patching matchMedia BEFORE any page script runs (CDP
# addScriptToEvaluateOnNewDocument runs ahead of the document's own scripts).
_FORCE_HOVER_JS = """
const _mm = window.matchMedia.bind(window);
window.matchMedia = function (q) {
  if (/hover:\\s*hover|pointer:\\s*fine/.test(q)) {
    return {matches: true, media: q,
            addListener() {}, removeListener() {},
            addEventListener() {}, removeEventListener() {}};
  }
  return _mm(q);
};
"""


def _chrome(rendered_page, force_hover):
    opts = Options()
    for a in ("--headless=new", "--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage", "--window-size=1600,1200"):
        opts.add_argument(a)
    try:
        d = webdriver.Chrome(options=opts)
    except Exception as exc:  # noqa: BLE001 — environment without a usable driver
        pytest.skip(f"could not start Chrome webdriver: {exc}")
    if force_hover:
        d.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {"source": _FORCE_HOVER_JS})
    d.get(f"file://{rendered_page}")
    d.implicitly_wait(2)
    return d


@pytest.fixture(scope="module")
def driver(rendered_page):
    d = _chrome(rendered_page, force_hover=True)
    yield d
    d.quit()


@pytest.fixture(scope="module")
def unforced_driver(rendered_page):
    """Chrome with its REAL media-query values — no matchMedia patch.

    Headless Chrome reports `(hover: hover)` and `(any-hover: hover)` false, which is
    precisely what Chrome reported on the desktop where hovering opened nothing. Every
    other lane forces those true and therefore cannot observe that failure. This is the
    negative control: it must pass while the browser claims no hover capability.
    """
    d = _chrome(rendered_page, force_hover=False)
    yield d
    d.quit()


@pytest.fixture(scope="module")
def touch_driver(rendered_page):
    """A driver on the TOUCH branch — no matchMedia patch, so `(hover: hover)` is false.

    Tap-to-pin only exists here: pinning is modal, so a mouse click must never pin or it
    would disable hover page-wide. The page decides from `PointerEvent.pointerType`, and
    the tap is delivered by `_tap`, which fires a genuine `pointerdown` carrying
    `pointerType: 'touch'` before the click. A bare `.click()` carries no pointerdown and
    would be classified (correctly) as not-touch, testing nothing; CDP mouse-to-touch
    emulation was tried first and deadlocks ActionChains. This fixture is what keeps the
    pin machinery under test.
    """
    d = _chrome(rendered_page, force_hover=False)
    yield d
    d.quit()


def _tap(drv, el):
    """Tap `el` as a finger would: pointerdown(pointerType=touch), then the click."""
    drv.execute_script(
        """
        const el = arguments[0];
        el.dispatchEvent(new PointerEvent('pointerdown',
            {bubbles: true, cancelable: true, pointerType: 'touch', isPrimary: true}));
        el.dispatchEvent(new PointerEvent('pointerup',
            {bubbles: true, cancelable: true, pointerType: 'touch', isPrimary: true}));
        el.click();
        """,
        el,
    )


def _assert_tooltip_actually_visible(drv, where):
    """A human must SEE text — not merely a `.ttip-visible` class somewhere in the DOM.

    The class-only assertion is what let this suite stay green through a page on which
    hovering did nothing: the popup can carry the class and still be zero-size, fully
    transparent, off-viewport, or underneath another element.
    """
    r = drv.execute_script(
        """
        const t = document.querySelector('.tab-panel.active .ttip.ttip-visible')
              || document.querySelector('.ttip.ttip-visible');
        if (!t) return {found: false};
        const b = t.getBoundingClientRect(), cs = getComputedStyle(t);
        const cx = b.x + b.width / 2, cy = b.y + b.height / 2;
        const top = document.elementFromPoint(cx, cy);
        return {found: true, w: b.width, h: b.height,
                text: (t.innerText || '').trim().length,
                display: cs.display, visibility: cs.visibility, opacity: parseFloat(cs.opacity),
                onScreen: b.right > 0 && b.bottom > 0 && b.x < innerWidth && b.y < innerHeight,
                topmostInside: !!(top && t.contains(top))};
        """
    )
    assert r["found"], f"{where}: no visible tooltip at all"
    assert r["text"] > 0, f"{where}: tooltip is visible but has no text: {r}"
    assert r["w"] > 0 and r["h"] > 0, f"{where}: tooltip has zero size: {r}"
    assert r["display"] != "none" and r["visibility"] == "visible", f"{where}: {r}"
    assert r["opacity"] > 0.5, f"{where}: tooltip is transparent: {r}"
    assert r["onScreen"], f"{where}: tooltip is off-viewport: {r}"
    assert r["topmostInside"], f"{where}: something is covering the tooltip: {r}"


def test_real_mouse_hover_opens_a_visible_tooltip(unforced_driver):
    """THE production contract: a real mouse move over a cell shows readable text.

    Runs on `unforced_driver`, which does NOT patch matchMedia — headless Chrome reports
    `(hover: hover)` and `(any-hover: hover)` false there, exactly as Chrome did on the
    desktop where hovering was dead. The page must not consult those queries to decide
    whether to listen for hover, so this lane fails if that gate ever comes back.
    """
    drv = unforced_driver
    assert drv.execute_script(
        "return !window.matchMedia('(hover: hover)').matches"
        " && !window.matchMedia('(any-hover: hover)').matches;"
    ), "this lane is only meaningful while the browser reports NO hover capability"
    drv.execute_script(
        "const v=document.querySelector('[data-view-detailed]'); if(v && !v.checked){v.checked=true; v.dispatchEvent(new Event('change'));}"
    )
    time.sleep(0.5)
    el = drv.execute_script(
        """
        const p = document.querySelector('.tab-panel.active') || document;
        const c = [...p.querySelectorAll('td.cell')].filter(e => e.offsetParent !== null
                                                             && e.querySelector('.ttip'))[20];
        if (!c) return null;
        c.scrollIntoView({block: 'center'});
        return c;
        """
    )
    if el is None:
        pytest.skip("no visible cell with a tooltip on the active tab")
    # A REAL pointer move, not a synthetic dispatch — a synthetic event would fire the
    # listener even if the browser would never deliver one to a human.
    webdriver.ActionChains(drv).move_to_element(el).perform()
    deadline = time.time() + 4
    while time.time() < deadline:
        if drv.execute_script("return !!document.querySelector('.ttip.ttip-visible');"):
            break
        time.sleep(0.1)
    assert drv.execute_script(
        "return arguments[0].matches(':hover');", el
    ), "the browser did not deliver a real hover to the cell; test setup is wrong"
    _assert_tooltip_actually_visible(drv, "real mouse hover, no forced media queries")


def test_keyboard_focus_opens_a_visible_tooltip(unforced_driver):
    """Focus must open the popup too — `focusin` sat behind the same gate as hover."""
    drv = unforced_driver
    opened = drv.execute_script(
        """
        const p = document.querySelector('.tab-panel.active') || document;
        for (const el of p.querySelectorAll('[data-ttip-wired]')) {
          if (!el.querySelector('.ttip') || el.offsetParent === null) continue;
          const f = el.querySelector('a, button, [tabindex]');
          if (!f) continue;
          el.scrollIntoView({block: 'center'});
          f.focus();
          return true;
        }
        return false;
        """
    )
    if not opened:
        pytest.skip("no focusable element inside a wired tooltip host on this tab")
    deadline = time.time() + 4
    while time.time() < deadline:
        if drv.execute_script("return !!document.querySelector('.ttip.ttip-visible');"):
            break
        time.sleep(0.1)
    _assert_tooltip_actually_visible(drv, "keyboard focus")


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


def test_order_divergence_shows_golden_and_candidate_sequences(driver):
    """ORDER/MERGE explanations come from the golden candidate in the model."""
    text = driver.execute_script(
        """
        const el = document.querySelector('[data-sequence-divergence]');
        if (!el) return null;
        window.__buildTooltip(el);
        const tip = el.querySelector('.ttip');
        return tip ? tip.textContent : null;
        """
    )
    assert text, "rendered producer data had no ORDER/MERGE divergence"
    assert "want:" in text and "got:" in text, text
    want = text.split("want:", 1)[1].split("got:", 1)[0].strip()
    got = text.split("got:", 1)[1].splitlines()[0].strip()
    assert want and got and want != got, text


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
    leak-only (depends on the Reference, not the Compares). Leaving Detailed clears
    its comparison, so turning it back on shows exactly the starred Reference."""
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

    set_detailed(True)
    selected = driver.execute_script(
        "const p=document.querySelector('.tab-panel.active .cmpctl');"
        "const x=[...p.querySelectorAll('input.cmp-on:not(:disabled)')][0];"
        "if(!x){return null;} x.checked=true; x.dispatchEvent(new Event('change',{bubbles:true}));"
        "return x.value;"
    )
    assert selected, "test needs a selectable Compare-with checkbox"
    set_detailed(False)
    assert cmp_box_visible() is False, "CMP column should be hidden in Overview"
    assert ref_box_visible() is True, "REF picker must still show in Overview"
    set_detailed(True)
    assert cmp_box_visible() is True, "CMP column should reappear in Details"
    remaining = driver.execute_script(
        "const p=document.querySelector('.tab-panel.active .cmpctl');"
        "return {refs:p.querySelectorAll('input.cmp-ref:checked').length,"
        "extra:[...p.querySelectorAll('input.cmp-on:checked:not(:disabled)')].map(x=>x.value)};"
    )
    assert remaining == {"refs": 1, "extra": []}, (
        "leaving Detailed must clear the compare checkbox and retain one Reference"
    )


def test_overview_url_clears_restored_compare_selection(driver):
    """A shared Overview URL must not carry a hidden comparison into Details."""
    page = driver.current_url.split("?", 1)[0]
    driver.get(
        page
        + "?tab=tab-unified&base_tab-unified=vllm_rust&cmp_tab-unified=vllm_python"
    )
    state = driver.execute_script(
        "const p=document.querySelector('.tab-panel.active'), c=p.querySelector('.cmpctl');"
        "return {overview:document.body.classList.contains('view-overview'),"
        "extra:[...c.querySelectorAll('input.cmp-on:checked:not(:disabled)')].map(x=>x.value)};"
    )
    assert state == {"overview": True, "extra": []}

    driver.execute_script(
        "const v=document.querySelector('[data-view-detailed]');"
        "v.checked=true; v.dispatchEvent(new Event('change'));"
    )
    restored = driver.execute_script(
        "const c=document.querySelector('.tab-panel.active .cmpctl');"
        "return [...c.querySelectorAll('input.cmp-on:checked:not(:disabled)')].map(x=>x.value);"
    )
    assert restored == []


def test_compare_url_restores_legacy_reference_without_self_compare(driver):
    """An existing shared vLLM Rust URL keeps its star and drops cmp=base."""
    page = driver.current_url.split("?", 1)[0]
    driver.get(
        page + "?tab=tab-unified&base_tab-unified=vllm_rust&cmp_tab-unified=vllm_rust"
    )
    state = driver.execute_script(
        "const p=document.querySelector('.tab-panel.active'), c=p.querySelector('.cmpctl');"
        "return {base:c.querySelector('input.cmp-ref:checked')?.value,"
        "extra:[...c.querySelectorAll('input.cmp-on:checked:not(:disabled)')].map(x=>x.value),"
        "url:location.search};"
    )
    assert state["base"] == "vllm_rust"
    assert state["extra"] == []
    assert "base_tab-unified=vllm_rust" in state["url"]
    assert "cmp_tab-unified" not in state["url"]
    assert "base_tab-toolcalling" not in state["url"]


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
    """The mirror's cloned cells must not inflate the overview counts (applyCtl skips
    cells inside [data-transpose-table] when counting)."""
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


def test_click_never_pins_where_hover_exists(driver):
    """Where hover exists, a click must NOT pin — hover stays live afterwards.

    Pinning is modal: `hoverAllowed()` returns false for every other cell while one is
    pinned. When a desktop click also pinned, a single stray click disabled hover for the
    whole page until the pin was released, and in details view "click elsewhere to
    dismiss" almost always lands on another wired cell, which only moves the pin. This
    pins the fix: on a hover-capable device no click pins anything, so hover cannot be
    switched off by clicking.
    """
    driver.execute_script(
        "const v=document.querySelector('[data-view-detailed]'); if(v && !v.checked){v.checked=true; v.dispatchEvent(new Event('change'));}"
    )
    clicked = driver.execute_script(
        """
        const tab = document.querySelector('.tab-panel.active') || document;
        for (const el of tab.querySelectorAll('[data-ttip-wired]')) {
          if (!el.querySelector('.ttip')) continue;
          if (el.offsetParent === null) continue;
          el.click();
          window.__clicked = el;
          return true;
        }
        return false;
        """
    )
    if not clicked:
        pytest.skip("no wired elements with a tooltip on the active tab")
    time.sleep(0.6)
    assert not driver.execute_script(
        "return !!document.querySelector('.ttip.ttip-pinned');"
    ), "a click pinned a popup on a hover-capable device"
    assert not driver.execute_script(
        "return document.body.classList.contains('ttip-pin-mode');"
    ), "a click put the page into modal pin mode on a hover-capable device"

    # And hover still opens a popup on a DIFFERENT cell afterwards — the actual symptom.
    driver.execute_script(
        """
        const tab = document.querySelector('.tab-panel.active') || document;
        for (const el of tab.querySelectorAll('[data-ttip-wired]')) {
          if (!el.querySelector('.ttip')) continue;
          if (el.offsetParent === null || el === window.__clicked) continue;
          el.dispatchEvent(new PointerEvent('pointerenter', {bubbles: false}));
          return;
        }
        """
    )
    deadline = time.time() + 3
    shown = False
    while time.time() < deadline and not shown:
        shown = driver.execute_script("return !!document.querySelector('.ttip.ttip-visible');")
        time.sleep(0.1)
    assert shown, "hover stopped opening popups after a click"


@pytest.mark.parametrize("transposed", [False, True], ids=["normal", "transposed"])
def test_every_wired_element_stays_pinned(touch_driver, transposed):
    """TOUCH: tapping anything with a popup PINS it, and the same tap must not close it.

    Runs on `touch_driver` because tap-to-pin is now touch-only; where hover exists there
    is no click-to-pin at all (see test_click_never_pins_where_hover_exists).

    The document-level outside-click handler decides what counts as "inside". While it
    enumerated classes, it kept drifting from the set `attachTooltip` actually wires: first
    `th.case-sub` was pinnable but not inside, then the transpose headers (`th.tcol-model`,
    `th.trow-case`) — which are built and wired at runtime and so can never appear in a
    static list. In each case a click pinned the popup and then bubbled to that handler,
    which saw no matching ancestor and unpinned it, so the popup could never stay up.

    The handler now asks for `[data-ttip-wired]`, the mark `attachTooltip` leaves on every
    element it wires. This test walks that same set, one element per distinct tag+class, so
    a newly wired kind of element is covered without anyone remembering to add it here.
    """
    touch_driver.execute_script(
        "const v=document.querySelector('[data-view-detailed]'); if(v && !v.checked){v.checked=true; v.dispatchEvent(new Event('change'));}"
    )
    toggled = touch_driver.execute_script(
        """
        const t = document.querySelector('[data-transpose-toggle]');
        if (!t) return false;
        if (t.checked !== arguments[0]) { t.checked = arguments[0]; t.dispatchEvent(new Event('change')); }
        return true;
        """,
        transposed,
    )
    if transposed and not toggled:
        pytest.skip("no transpose toggle on this page")

    # One representative per tag+class, so the test scales with the wired set instead of a
    # hand-kept list, but does not click hundreds of identical data cells.
    kinds = touch_driver.execute_script(
        """
        const tab = document.querySelector('.tab-panel.active') || document;
        const seen = new Set();
        for (const el of tab.querySelectorAll('[data-ttip-wired]')) {
          if (!el.querySelector('.ttip')) continue;
            // Skip anything not actually RENDERED. A collapsed column is
            // `display: none` (`.col-hidden`), so its cells cannot be clicked and
            // have no popup to pin. This remains a product-facing assertion even
            // though the module-scoped driver is reloaded around every test: hidden
            // elements are not user-interactable and cannot own a visible popup.
          if (el.offsetParent === null) continue;
          const key = el.tagName.toLowerCase() + '.' + (el.className || '');
          if (!seen.has(key)) seen.add(key);
        }
        return Array.from(seen);
        """
    )
    if not kinds:
        pytest.skip("no wired elements with a tooltip on the active tab")

    for key in kinds:
        # A real .click() so the event bubbles to the document handler, which is the whole
        # point — a synthetic dispatch on the element alone would never reproduce the bug.
        # A REAL tap through the input pipeline. Under the fixture's CDP touch emulation
        # this produces `pointerdown` with `pointerType == 'touch'`, which is what the page
        # keys tap-to-pin off. A JS `.click()` carries no pointerdown at all, so it would be
        # classified as not-touch and would silently test nothing.
        el = touch_driver.execute_script(
            """
            const tab = document.querySelector('.tab-panel.active') || document;
            for (const el of tab.querySelectorAll('[data-ttip-wired]')) {
              if (!el.querySelector('.ttip')) continue;
              if (el.offsetParent === null) continue;
              if (el.tagName.toLowerCase() + '.' + (el.className || '') !== arguments[0]) continue;
              el.scrollIntoView({block: 'center'});
              return el;
            }
            return null;
            """,
            key,
        )
        if el is None:
            continue
        _tap(touch_driver, el)
        deadline = time.time() + 3
        pinned = False
        while time.time() < deadline:
            pinned = touch_driver.execute_script(
                "return !!document.querySelector('[data-ttip-wired] .ttip.ttip-pinned');"
            )
            if pinned:
                break
            time.sleep(0.1)
        assert pinned, f"popup on {key} did not stay pinned after its own click"
        # Unpin before the next kind, so a stale pin can't make the next assertion pass.
        touch_driver.execute_script(
            "document.querySelectorAll('.ttip.ttip-pinned').forEach(t => { const c = t.querySelector('.ttip-close'); if (c) c.click(); });"
        )


def test_legend_is_detailed_only(driver):
    """The Reference/Compare legend belongs to Detailed, not Overview.

    It explains the compare bar, the NΔ count and the red-cell rule — none of which are
    on screen in Overview, so there it is a wall of text about controls the reader cannot
    see. Asserts both directions so a future CSS change cannot silently bring it back.
    """
    def legend_shown():
        return driver.execute_script(
            """
            const tab = document.querySelector('.tab-panel.active') || document;
            const el = tab.querySelector('.toolbar-desc') || document.querySelector('.toolbar-desc');
            if (!el) { return null; }
            return getComputedStyle(el).display !== 'none';
            """
        )
    def set_detailed(on):
        driver.execute_script(
            "const v=document.querySelector('[data-view-detailed]');"
            "if(v && v.checked!==arguments[0]){v.checked=arguments[0]; v.dispatchEvent(new Event('change'));}",
            on,
        )
        time.sleep(0.3)
    set_detailed(False)
    overview = legend_shown()
    if overview is None:
        pytest.skip("no .toolbar-desc legend on this page")
    set_detailed(True)
    detailed = legend_shown()
    assert not overview, "legend visible in Overview; it should be Detailed-only"
    assert detailed, "legend hidden in Detailed; it should be visible there"


def test_touch_then_keyboard_does_not_pin(driver):
    """A keyboard activation after a touch must NOT inherit the touch's pinning.

    Pointer provenance used to persist until the next `pointerdown`. A keyboard
    activation issues none, so after any touch the next Enter on a focused link
    still read `touch`, called `preventDefault()` and pinned — suppressing the
    navigation the keyboard user asked for. Provenance is now consumed by the click
    it belongs to and cleared.
    """
    drv = driver
    drv.execute_script(
        "const v=document.querySelector('[data-view-detailed]'); if(v && !v.checked){v.checked=true; v.dispatchEvent(new Event('change'));}"
    )
    time.sleep(0.4)

    # 1. A touch tap: pointerdown(touch) then click. This one MAY pin.
    ok = drv.execute_script(
        """
        const p = document.querySelector('.tab-panel.active') || document;
        const els = [...p.querySelectorAll('[data-ttip-wired]')]
            .filter(e => e.offsetParent !== null && e.querySelector('.ttip'));
        if (els.length < 2) return false;
        window.__a = els[0]; window.__b = els[1];
        window.__a.dispatchEvent(new PointerEvent('pointerdown',
            {bubbles: true, cancelable: true, pointerType: 'touch', isPrimary: true}));
        window.__a.click();
        return true;
        """
    )
    if not ok:
        pytest.skip("need two wired elements on the active tab")
    time.sleep(0.4)

    # 2. A KEYBOARD activation on a different cell: a click with NO pointerdown.
    #    It must not pin, and must not have been preventDefault()-ed.
    prevented = drv.execute_script(
        """
        const ev = new MouseEvent('click', {bubbles: true, cancelable: true});
        window.__b.dispatchEvent(ev);
        return ev.defaultPrevented;
        """
    )
    time.sleep(0.4)
    assert not prevented, (
        "a keyboard activation was preventDefault()-ed, so it inherited the earlier "
        "touch's pointer provenance instead of being treated as keyboard"
    )
    pinned_on_b = drv.execute_script(
        "return !!(window.__b && window.__b.querySelector('.ttip.ttip-pinned'));"
    )
    assert not pinned_on_b, "keyboard activation must not pin"

    drv.execute_script(
        "document.querySelectorAll('.ttip.ttip-pinned').forEach(t => { const c = t.querySelector('.ttip-close'); if (c) c.click(); });"
    )

    # 3. A touch tap INSIDE a tooltip takes the `.ttip` early return. That path must also
    #    consume the provenance, or it stays armed for the next keyboard activation —
    #    the same defect one branch over.
    tapped_inside = drv.execute_script(
        """
        const p = document.querySelector('.tab-panel.active') || document;
        for (const el of p.querySelectorAll('[data-ttip-wired]')) {
          const tip = el.querySelector('.ttip');
          if (!tip || el.offsetParent === null) continue;
          tip.dispatchEvent(new PointerEvent('pointerdown',
              {bubbles: true, cancelable: true, pointerType: 'touch', isPrimary: true}));
          tip.dispatchEvent(new MouseEvent('click', {bubbles: true, cancelable: true}));
          return true;
        }
        return false;
        """
    )
    if tapped_inside:
        time.sleep(0.3)
        prevented2 = drv.execute_script(
            """
            const ev = new MouseEvent('click', {bubbles: true, cancelable: true});
            window.__b.dispatchEvent(ev);
            return ev.defaultPrevented;
            """
        )
        assert not prevented2, (
            "a touch tap inside a tooltip left pointer provenance armed, so the next "
            "keyboard activation inherited it"
        )


def test_touch_outside_a_host_does_not_arm_a_pointerdownless_activation(driver):
    """A tap that lands anywhere else must not arm the next pointerdown-less activation.

    Pointer provenance recorded only the KIND of pointer, in one document-wide
    variable. Only a tooltip host's own click consumed it, so a tap on a header —
    or any element that is not a host — left `touch` armed indefinitely. The next
    activation on a host that carries NO pointerdown of its own — keyboard Enter, a
    synthetic click — then read that stale `touch`, pinned, and `preventDefault()`ed
    the parser-source link the user actually asked for. A genuine mouse click is NOT
    the vulnerable case: it brings its own mouse `pointerdown`, which overwrites the
    stale value before the click arrives. Provenance now also records WHERE the
    pointer went down and is only honoured when the gesture started in the same host.
    """
    drv = driver
    drv.execute_script(
        "const v=document.querySelector('[data-view-detailed]'); if(v && !v.checked){v.checked=true; v.dispatchEvent(new Event('change'));}"
    )
    time.sleep(0.4)

    pinned = drv.execute_script(
        """
        const p = document.querySelector('.tab-panel.active') || document;
        const cell = p.querySelector('[data-ttip-wired]');
        if (!cell) { return 'no-cell'; }

        // A touch that goes down on something that is NOT a tooltip host, and whose
        // click therefore never reaches a host's handler to consume the provenance.
        const outside = document.querySelector('h1, h2, header, .legend') || document.body;
        outside.dispatchEvent(new PointerEvent('pointerdown',
            {bubbles: true, cancelable: true, pointerType: 'touch', isPrimary: true}));
        outside.dispatchEvent(new MouseEvent('click', {bubbles: true, cancelable: true}));

        // Now an activation on a host that carries NO pointerdown of its own — a keyboard
        // Enter or a synthetic click. It cannot overwrite the stale provenance, so this is
        // the interaction that actually reads it.
        cell.dispatchEvent(new MouseEvent('click', {bubbles: true, cancelable: true}));

        return !!document.querySelector('.ttip.ttip-pinned');
        """
    )
    assert pinned != 'no-cell', "no wired tooltip host found to exercise"
    assert pinned is False, (
        "a pointerdown-less activation pinned because an unrelated touch elsewhere left "
        "the provenance armed — pinning must require the gesture to start in this host"
    )
