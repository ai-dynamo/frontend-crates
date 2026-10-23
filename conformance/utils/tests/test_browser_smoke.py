# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Browser-level smoke test for the conformance matrix (audit D4).

The recent regressions were browser-behavior regressions (case popups, parser
radios) that the string-level template tests can't catch. This renders the real
table and drives a headless browser: activate a case and assert its popup becomes
visible; assert the vLLM Rust parser option shows on a tool-calling tab and hides
on Reasoning (which has no vLLM Rust column).

Skips when Selenium or headless Chrome aren't available, so it adds no hard test
dependency — it runs where a browser exists and is a no-op otherwise.
"""
import re
import shutil
import time

import pytest

selenium = pytest.importorskip("selenium")
from selenium import webdriver  # noqa: E402
from selenium.webdriver.common.action_chains import ActionChains  # noqa: E402
from selenium.webdriver.common.by import By  # noqa: E402
from selenium.webdriver.chrome.options import Options  # noqa: E402
from selenium.webdriver.support.ui import WebDriverWait  # noqa: E402

pytestmark = pytest.mark.skipif(
    not any(shutil.which(b) for b in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser")),
    reason="no headless Chrome available",
)


# Exercise the click-only contract with both reported pointer capabilities.
# Install this media-query override before any document script runs.
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


_RECORD_FIRST_VISIBLE_MATRIX_JS = """
window.__firstVisibleMatrix = null;
window.__firstMatrixInsertion = null;
function recordFirstVisibleMatrix() {
  const panel = document.querySelector('.tab-panel.active');
  if (!panel) return;
  const original = panel.querySelector('table[data-parity-table]:not([data-transpose-table])');
  if (!window.__firstMatrixInsertion && original) {
    window.__firstMatrixInsertion = {
      pending: document.documentElement.classList.contains('transpose-pending'),
      transposeMode: document.body.classList.contains('transpose-mode'),
      originalDisplay: getComputedStyle(original).display
    };
  }
  if (window.__firstVisibleMatrix) return;
  const table = Array.from(panel.querySelectorAll(
    'table[data-parity-table], table[data-transpose-table]'
  )).find(function (candidate) {
    return getComputedStyle(candidate).display !== 'none';
  });
  if (table) {
    window.__firstVisibleMatrix = {
      kind: table.hasAttribute('data-transpose-table') ? 'transpose' : 'horizontal',
      pending: document.documentElement.classList.contains('transpose-pending'),
      transposeMode: document.body.classList.contains('transpose-mode')
    };
  }
}
const observer = new MutationObserver(recordFirstVisibleMatrix);
observer.observe(document, {childList: true, subtree: true, attributes: true});
"""


def _chrome(reference_page, force_hover):
    opts = Options()
    for a in ("--headless=new", "--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage", "--window-size=1600,1200"):
        opts.add_argument(a)
    try:
        d = webdriver.Chrome(options=opts)
    except Exception as exc:  # noqa: BLE001 — environment without a usable driver
        pytest.skip(f"could not start Chrome webdriver: {exc}")
    if force_hover:
        d.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {"source": _FORCE_HOVER_JS})
    d.get(f"file://{reference_page}?transpose=0")
    d.implicitly_wait(2)
    return d


@pytest.fixture(scope="module")
def driver(reference_page):
    d = _chrome(reference_page, force_hover=True)
    yield d
    d.quit()


@pytest.fixture(scope="module")
def unforced_driver(reference_page):
    """Exercise case activation with the browser's actual pointer capabilities."""
    d = _chrome(reference_page, force_hover=False)
    yield d
    d.quit()


@pytest.fixture(scope="module")
def touch_driver(reference_page):
    """Exercise native case activation after touch pointer events."""
    d = _chrome(reference_page, force_hover=False)
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


def test_real_mouse_hover_does_not_build_case_popup(unforced_driver):
    drv = unforced_driver
    drv.get(drv.current_url.split("?", 1)[0] + "?tab=tab-unified&transpose=0")
    el = drv.find_element(By.CSS_SELECTOR, '.tab-panel.active td.cell[data-ttip-id]')
    drv.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
    webdriver.ActionChains(drv).move_to_element(el).perform()
    assert drv.execute_script("return arguments[0].matches(':hover')", el)
    assert drv.execute_script("return !document.querySelector('.ttip-visible') && !arguments[0]._ttip.querySelector('.ttip-head')", el)
    el.click()
    _assert_tooltip_actually_visible(drv, 'native click after hover')


def test_keyboard_focus_requires_activation_for_case_popup(unforced_driver):
    drv = unforced_driver
    drv.get(drv.current_url.split("?", 1)[0] + "?tab=tab-unified&transpose=0")
    header = drv.find_element(By.CSS_SELECTOR, '.tab-panel.active th.case-sub[data-ttip-id]')
    drv.execute_script("arguments[0].scrollIntoView({block:'center',inline:'center'}); arguments[0].focus();", header)
    assert drv.execute_script("return !document.querySelector('.ttip-visible') && !arguments[0]._ttip.querySelector('.ttip-head')", header)
    header.send_keys('\ue007')
    _assert_tooltip_actually_visible(drv, 'Enter activation')


def test_pointerenter_does_not_construct_case_popup(driver):
    driver.get(driver.current_url.split("?", 1)[0] + "?tab=tab-unified&transpose=0")
    cell = driver.find_element(By.CSS_SELECTOR, '.tab-panel.active td.cell[data-ttip-id]')
    assert driver.execute_script("arguments[0].dispatchEvent(new PointerEvent('pointerenter', {bubbles:true})); return !document.querySelector('.ttip-visible') && !arguments[0]._ttip.querySelector('.ttip-head');", cell)


def test_column_toggle_hint_is_anchored_to_the_button(driver):
    _click_tab(driver, "tab-toolcalling-batch")
    result = driver.execute_script(
        """
        const button = [...document.querySelectorAll('.tab-panel.active [data-col-toggle]')]
          .find(el => el.dataset.colLabel === 'Tool calling family');
        if (!button) return null;
        return {
          title: button.getAttribute('title'),
          tooltip: button.dataset.tooltip,
          tooltipText: button.dataset.ttipText,
          wired: button.dataset.ttipWired,
        };
        """
    )
    assert result, "Tool calling family column control is missing"
    assert result["title"] is None, result
    assert result["tooltip"] == "Collapse Tool calling family column", result
    assert result["tooltipText"] == "Collapse Tool calling family column", result
    assert result["wired"] == "1", result


def test_column_toggle_hover_opens_its_tooltip(driver):
    _click_tab(driver, "tab-toolcalling-batch")
    result = driver.execute_script(
        """
        const button = [...document.querySelectorAll('.tab-panel.active [data-col-toggle]')]
          .find(el => el.dataset.colLabel === 'Tool calling family');
        button.dispatchEvent(new MouseEvent('mouseenter', {bubbles: false, view: window}));
        const tooltip = button._ttip;
        return {
          visible: tooltip.classList.contains('ttip-visible'),
          visibility: getComputedStyle(tooltip).visibility,
        };
        """
    )

    assert result == {"visible": True, "visibility": "visible"}


def test_case_popup_stays_centered_after_document_scroll(driver):
    driver.get(driver.current_url.split("?", 1)[0] + "?tab=tab-unified&transpose=0")
    cell = driver.find_element(By.CSS_SELECTOR, '.tab-panel.active tbody tr:last-child td.cell[data-ttip-id]')
    driver.execute_script("arguments[0].scrollIntoView({block:'center'}); arguments[0].click();", cell)
    result = driver.execute_script("""
        const t=arguments[0]._ttip, r=t.getBoundingClientRect();
        return {portalled:t.parentElement===document.body, dx:Math.abs((r.left+r.right)/2-innerWidth/2),
                dy:Math.abs((r.top+r.bottom)/2-innerHeight/2), width:r.width, height:r.height, vw:innerWidth, vh:innerHeight};
    """, cell)
    assert result['portalled'] and result['dx'] < 2 and result['dy'] < 2, result
    assert result['width'] <= result['vw'] * .97 + 2 and result['height'] <= result['vh'] * .97 + 2, result


def test_clicked_popup_stays_open_after_pointer_leaves(driver):
    driver.get(driver.current_url.split("?", 1)[0] + "?tab=tab-unified&transpose=0")
    cell=driver.find_element(By.CSS_SELECTOR, '.tab-panel.active td.cell[data-ttip-id]')
    driver.execute_script("arguments[0].click(); arguments[0].dispatchEvent(new MouseEvent('mouseleave'));", cell)
    assert driver.execute_script("return arguments[0]._ttip.classList.contains('ttip-visible') && arguments[0]._ttip.classList.contains('ttip-pinned')", cell)


def test_transpose_reopens_the_same_case_with_one_popup(driver):
    driver.get(driver.current_url.split("?", 1)[0] + "?tab=tab-unified&transpose=0")
    cell = driver.find_element(By.CSS_SELECTOR, '.tab-panel.active td.cell[data-ttip-id]')
    result = driver.execute_script("""
        const cell=arguments[0], id=cell.dataset.ttipId;
        cell.click();
        const before=document.querySelectorAll('.ttip-visible').length;
        document.querySelector('[data-transpose-toggle]').click();
        const dismissed=document.querySelectorAll('.ttip-visible').length;
        const clone=document.querySelector('.transpose-table td.cell[data-ttip-id="'+id+'"]');
        clone.click();
        return {before,dismissed,after:document.querySelectorAll('.ttip-visible').length,
                portalled:clone._ttip.parentElement===document.body};
    """, cell)
    assert result == {'before':1,'dismissed':0,'after':1,'portalled':True}



def test_keyboard_popup_restores_host_focus_on_escape(driver):
    driver.get(driver.current_url.split("?", 1)[0] + "?tab=tab-unified&transpose=0")
    cell=driver.find_element(By.CSS_SELECTOR, '.tab-panel.active td.cell[data-ttip-id]')
    driver.execute_script("arguments[0].focus()", cell)
    cell.send_keys(' ')
    assert driver.execute_script("return arguments[0]._ttip.classList.contains('ttip-visible')", cell)
    driver.switch_to.active_element.send_keys('\ue00c')
    assert driver.execute_script("return !document.querySelector('.ttip-visible') && document.activeElement===arguments[0]", cell)


def test_column_toggle_reuses_its_portalled_tooltip(driver):
    _click_tab(driver, "tab-toolcalling-batch")
    result = driver.execute_script(
        """
        const button = [...document.querySelectorAll('.tab-panel.active [data-col-toggle]')]
          .find(el => el.dataset.colLabel === 'Tool calling family');
        const tooltip = button._ttip;
        document.body.appendChild(tooltip);
        button.click();
        return {
          count: button._ttip === tooltip ? 1 : 0,
          owned: button._ttip === tooltip,
          text: tooltip.querySelector('.column-control-tooltip-text').textContent,
        };
        """
    )

    assert result == {
        "count": 1,
        "owned": True,
        "text": "Expand Tool calling family column",
    }


def test_order_divergence_shows_golden_and_candidate_sequences(driver):
    """ORDER/MERGE explanations come from the golden candidate in the model."""
    text = driver.execute_script(
        """
            const el = document.querySelector('[data-sequence-divergence]');
            if (!el) return null;
            window.__buildTooltip(el);
            return el.querySelector('.ttip')?.textContent || null;
        """
    )
    assert text, "rendered producer data had no ORDER/MERGE divergence"
    assert "want:" in text and "got:" in text, text


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


@pytest.mark.parametrize("query,expected_base", [
    ("base_tab-unified=dynamo%400.6.0%2Bsource." + "a" * 64, "dynamo"),
    ("base_tab-unified=removed-parser", "dynamo"),
    ("cmp_tab-unified=vllm_rust", "dynamo"),
    ("base_tab-unified=", None),
    ("base_tab-unified=dynamo%400.6.0", "dynamo@0.6.0"),
])
def test_compare_url_distinguishes_removed_and_empty_reference(driver, query, expected_base):
    page = driver.current_url.split("?", 1)[0]
    try:
        driver.get(page + "?tab=tab-unified&" + query)
        state = driver.execute_script(
            "const c=document.querySelector('#tab-unified .cmpctl');"
            "return {base:c.querySelector('input.cmp-ref:checked')?.value || null,"
            "enabled:c.querySelectorAll('input.cmp-on:not(:disabled)').length};"
        )
        assert state["base"] == expected_base
        assert (state["enabled"] > 0) is (expected_base is not None)
    finally:
        driver.get(page)


def test_unified_release_labels_hide_capture_identity(driver):
    page = driver.current_url.split("?", 1)[0]
    try:
        driver.get(page + "?tab=tab-unified")
        labels = driver.execute_script(
            "return [...document.querySelectorAll('#tab-unified .cmprow')]"
            ".map(row=>row.innerText);"
        )
        assert all("+source." not in label for label in labels)
        assert all("working build" not in label for label in labels)
        driver.execute_script(
            "document.querySelector('#tab-unified input.cmp-ref[value=\"dynamo@0.6.0\"]').click();"
        )
        assert driver.execute_script(
            "return document.querySelector('#tab-unified input.cmp-ref:checked').value;"
        ) == "dynamo@0.6.0"
    finally:
        driver.get(page)


def test_unified_sglang_versions_are_visible_and_selectable(driver):
    page = driver.current_url.split("?", 1)[0]
    try:
        driver.get(page + "?tab=tab-unified&transpose=0")
        assert driver.find_element(By.CSS_SELECTOR, "#tab-unified input.cmp-ref:checked").get_attribute("value") == "dynamo"
        for version in ("0.5.14", "0.5.16", "0.5.21"):
            key = f"sglang_python@{version}"
            labels = driver.find_elements(By.CSS_SELECTOR, f'#tab-unified .cmprow-label[data-cand="{key}"]')
            assert len(labels) == 1 and labels[0].is_displayed()
            assert f"SGLang Python {version} (stream, reasoning + tool detectors)" in labels[0].text
            assert "Dynamo backend version" not in labels[0].text
            driver.find_element(By.CSS_SELECTOR, f'#tab-unified input.cmp-ref[value="{key}"]').click()
            assert driver.find_element(By.CSS_SELECTOR, "#tab-unified input.cmp-ref:checked").get_attribute("value") == key
    finally:
        driver.get(page)


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


@pytest.mark.parametrize("override,expected", [("", False), ("1", True), ("true", True),
                                               ("0", False), ("false", False)])
def test_transpose_default_overrides_toggle_refresh_and_tabs(driver, reference_page, override, expected):
    url = f"file://{reference_page}?tab=tab-unified"
    try:
        driver.get(url + ("&transpose=" + override if override else ""))
        toggle = driver.find_element(By.CSS_SELECTOR, '[data-transpose-toggle]')
        assert toggle.is_selected() == expected
        assert driver.execute_script("return document.body.classList.contains('transpose-mode')") == expected
        toggle.click()
        selected = not expected
        assert toggle.is_selected() == selected
        assert driver.execute_script("return new URL(location.href).searchParams.get('transpose')") == ("1" if selected else None)
        driver.refresh()
        assert driver.find_element(By.CSS_SELECTOR, '[data-transpose-toggle]').is_selected() == selected
        for tab in ("tab-toolcalling-batch", "tab-unified"):
            _click_tab(driver, tab)
            assert driver.find_element(By.CSS_SELECTOR, '[data-transpose-toggle]').is_selected() == selected
            assert driver.execute_script("return document.body.classList.contains('transpose-mode')") == selected
    finally:
        driver.get(f"file://{reference_page}?transpose=0")


@pytest.mark.parametrize("override,expected", [("", "horizontal"), ("1", "transpose"),
                                               ("true", "transpose"), ("0", "horizontal"),
                                               ("false", "horizontal")])
def test_first_visible_matrix_matches_transpose_url(reference_page, override, expected):
    d = _chrome(reference_page, force_hover=False)
    try:
        d.execute_cdp_cmd(
            "Page.addScriptToEvaluateOnNewDocument",
            {"source": _RECORD_FIRST_VISIBLE_MATRIX_JS},
        )
        url = f"file://{reference_page}" + ("?transpose=" + override if override else "")
        d.get(url)
        first = d.execute_script("return window.__firstVisibleMatrix")
        insertion = d.execute_script("return window.__firstMatrixInsertion")
        assert first is not None, "no report matrix became visible during initial rendering"
        assert insertion is not None, "no original report table was observed when inserted"
        assert first["kind"] == expected, f"first exposed matrix was {first}"
        if expected == "transpose":
            assert insertion["pending"] is True
            assert insertion["originalDisplay"] == "none"
            assert first["pending"] is False
            assert first["transposeMode"] is True
        else:
            assert insertion["pending"] is False
            assert insertion["originalDisplay"] != "none"
            assert first["pending"] is False
            assert first["transposeMode"] is False
    finally:
        d.quit()


def test_case_labels_squish_horizontally_and_expand_when_transposed(driver, reference_page):
    try:
        driver.get(f"file://{reference_page}?tab=tab-unified&view=details")
        header = driver.find_element(By.CSS_SELECTOR, "#tab-unified th.case-sub a")
        full_label = header.find_element(By.CSS_SELECTOR, ".case-label-full").get_attribute("textContent")
        short_label = header.find_element(By.CSS_SELECTOR, ".case-label-short").get_attribute("textContent")
        assert full_label.startswith("1-")
        assert short_label == "1-1"
        placeholder = header.find_element(By.CSS_SELECTOR, ".case-label-placeholder")
        assert placeholder.value_of_css_property("color") == "rgba(138, 143, 152, 1)"
        assert float(placeholder.value_of_css_property("font-size")[:-2]) < float(
            header.value_of_css_property("font-size")[:-2]
        )
        assert header.find_element(By.CSS_SELECTOR, ".case-label-full").value_of_css_property("display") == "none"
        assert header.find_element(By.CSS_SELECTOR, ".case-label-short").value_of_css_property("display") != "none"
        group_heading = driver.find_element(By.CSS_SELECTOR, "#tab-unified th.case-group .col-toggle-label")
        assert group_heading.text.startswith("TC ")
        assert group_heading.value_of_css_property("color") == "rgba(138, 143, 152, 1)"
        group_cell = driver.find_element(By.CSS_SELECTOR, "#tab-unified th.case-group")
        assert group_cell.value_of_css_property("text-align") == "left"
        assert group_cell.value_of_css_property("padding-left") == "8px"
        assert 8 <= group_heading.rect["x"] - group_cell.rect["x"] < 35
        label_pairs = driver.execute_script(
            "return Array.from(document.querySelectorAll('#tab-unified th.case-sub a')).map(a=>["
            "a.querySelector('.case-label-full').textContent,a.querySelector('.case-label-short').textContent,"
            "Boolean(a.querySelector('.case-label-placeholder')),a.closest('th.case-sub').dataset.colHideGroup]);"
        )
        assert label_pairs
        has_model_specific_label = False
        for full, short, has_placeholder, _group_key in label_pairs:
            group_name, separator, description = full.partition("-")
            assert separator
            if description:
                compact_prefix = {"deepseek_v4": "ds4", "deepseek_v41": "ds41"}.get(group_name, group_name)
                assert re.fullmatch(re.escape(compact_prefix) + r"-[1-9]\d*", short)
                assert has_placeholder
                has_model_specific_label = has_model_specific_label or not group_name.isdigit()
            else:
                assert short == full
                assert not has_placeholder
        assert has_model_specific_label
        by_prefix = {}
        by_section = {}
        for full, short, _, group_key in label_pairs:
            prefix, number = short.split("-", 1)
            by_prefix.setdefault(prefix, []).append(int(number))
            by_section.setdefault((prefix, group_key), []).append(full.split("-", 1)[1])
        for descriptions in by_section.values():
            assert descriptions == sorted(descriptions)
        for numbers in by_prefix.values():
            assert numbers == list(range(1, len(numbers) + 1))
        assert by_prefix["ds4"][0] == by_prefix["ds41"][0] == 1

        driver.execute_script(
            "arguments[0].closest('th.case-sub').click();",
            header,
        )
        deadline = time.time() + 3
        tip = None
        while time.time() < deadline:
            tips = driver.find_elements(By.CSS_SELECTOR, ".ttip.ttip-visible")
            if tips:
                tip = tips[0]
                break
            time.sleep(0.1)
        assert tip is not None, "case description popup did not open on click"
        assert tip.find_element(By.CSS_SELECTOR, ".ttip-head > span[title]").get_attribute("title") == header.get_attribute("aria-label")
        assert f"{full_label} ({short_label})" in tip.find_element(By.CSS_SELECTOR, ".ttip-head").text
        assert full_label in tip.text
        assert tip.find_element(By.CSS_SELECTOR, ".ttip-head-desc").text

        driver.execute_script("document.documentElement.setAttribute('data-theme', 'dark')")
        _set_transpose(driver, True)
        transposed = driver.find_element(By.CSS_SELECTOR, "#tab-unified table[data-transpose-table] th.trow-case a")
        assert transposed.find_element(By.CSS_SELECTOR, ".case-label-full").value_of_css_property("display") != "none"
        reference = transposed.find_element(By.CSS_SELECTOR, ".case-label-short")
        assert reference.value_of_css_property("display") == "inline-block"
        assert reference.text == short_label
        assert transposed.text == f"{full_label} {short_label}"
        assert float(reference.value_of_css_property("margin-left")[:-2]) > 0
        full_bounds = transposed.find_element(By.CSS_SELECTOR, ".case-label-full").rect
        assert reference.rect["x"] > full_bounds["x"] + full_bounds["width"]
        reference_number = reference.find_element(By.CSS_SELECTOR, ".case-label-placeholder")
        assert reference_number.value_of_css_property("color") == "rgba(138, 143, 152, 1)"
        assert float(reference_number.value_of_css_property("font-size")[:-2]) < float(
            transposed.value_of_css_property("font-size")[:-2]
        )
        transpose_labels = driver.execute_script(
            "return Array.from(document.querySelectorAll('#tab-unified .transpose-table th.trow-case a')).map(a=>["
            "a.querySelector('.case-label-full').textContent,a.querySelector('.case-label-short').textContent,"
            "getComputedStyle(a.querySelector('.case-label-short')).display]);"
        )
        assert [(full, short) for full, short, _ in transpose_labels] == [
            (full, short) for full, short, _, _ in label_pairs
        ]
        assert all(display == "inline-block" for _, _, display in transpose_labels)
        assert transposed.find_element(By.CSS_SELECTOR, ".case-label-full").get_attribute("textContent") == full_label
        model_header = driver.find_element(By.CSS_SELECTOR, "#tab-unified .transpose-table thead .tcol-model")
        assert model_header.value_of_css_property("background-color") == "rgba(15, 23, 42, 1)"
        transpose_group = driver.find_element(By.CSS_SELECTOR, "#tab-unified .transpose-table tr.section td")
        assert transpose_group.value_of_css_property("color") == "rgba(138, 143, 152, 1)"
        assert transpose_group.value_of_css_property("text-align") == "left"
        assert transpose_group.value_of_css_property("padding-left") == "8px"
        text_inset = driver.execute_script(
            "const e=arguments[0],range=document.createRange();range.selectNodeContents(e);"
            "return range.getBoundingClientRect().left-e.getBoundingClientRect().left;", transpose_group
        )
        assert 8 <= text_inset <= 10

        _set_transpose(driver, False)
        assert header.find_element(By.CSS_SELECTOR, ".case-label-full").value_of_css_property("display") == "none"
        assert header.find_element(By.CSS_SELECTOR, ".case-label-short").value_of_css_property("display") != "none"
        expanded_width = group_cell.rect["width"]
        driver.find_element(By.TAG_NAME, "h1").click()
        WebDriverWait(driver, 5).until(lambda d: not d.find_elements(By.CSS_SELECTOR, ".ttip.ttip-visible"))
        group_toggle = group_cell.find_element(By.CSS_SELECTOR, ".col-toggle")
        group_toggle.click()
        assert "col-collapsed" in group_cell.get_attribute("class").split()
        assert group_cell.value_of_css_property("padding-left") == "2px"
        assert group_cell.rect["width"] < expanded_width
        group_toggle.click()
        assert group_cell.value_of_css_property("padding-left") == "8px"
        assert abs(group_cell.rect["width"] - expanded_width) <= 1
        _click_tab(driver, "tab-toolcalling-batch")
        _set_transpose(driver, True)
        other_header = driver.find_element(By.CSS_SELECTOR, "#tab-toolcalling-batch .transpose-table th.trow-case a")
        assert not other_header.find_elements(By.CSS_SELECTOR, ".case-label-reference")
        assert other_header.find_element(By.CSS_SELECTOR, ".case-label-short").value_of_css_property("display") == "none"
        for cell in driver.find_elements(By.CSS_SELECTOR, "#tab-toolcalling-batch .transpose-table tr.section td"):
            assert cell.value_of_css_property("text-align") == "left"
            assert cell.value_of_css_property("padding-left") == "8px"
    finally:
        driver.get(f"file://{reference_page}?transpose=0")


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


def test_desktop_click_pins_until_outside_click(driver):
    driver.get(driver.current_url.split("?", 1)[0] + "?tab=tab-unified&transpose=0")
    cell=driver.find_element(By.CSS_SELECTOR, '.tab-panel.active td.cell[data-ttip-id]')
    driver.execute_script("arguments[0].click()", cell)
    assert driver.execute_script("return arguments[0]._ttip.classList.contains('ttip-pinned') && document.body.classList.contains('ttip-pin-mode')", cell)
    driver.find_element(By.TAG_NAME, 'h1').click()
    assert not driver.find_elements(By.CSS_SELECTOR, '.ttip-visible')


@pytest.mark.parametrize("transposed", [False, True], ids=["normal", "transposed"])
def test_every_wired_element_stays_pinned(touch_driver, transposed):
    """TOUCH: tapping anything with a popup PINS it, and the same tap must not close it.

    Runs on `touch_driver` to prove touch activation follows the same pinned-popup contract as desktop clicks.

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
                "return !!document.querySelector('.ttip.ttip-pinned');"
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


def test_duplicate_markers_are_detailed_only_and_name_the_canonical_case(driver, rendered_page):
    original_url = driver.current_url
    try:
        driver.get(f"file://{rendered_page}?tab=tab-unified")

        def state():
            return driver.execute_script(
                """
                const markers = [...document.querySelectorAll('.duplicate-marker')];
                return {
                  count: markers.length,
                  visible: markers.filter((marker) => getComputedStyle(marker).display !== 'none').length,
                  titles: markers.map((marker) => marker.getAttribute('title')),
                };
                """
            )

        driver.execute_script(
            "const v=document.querySelector('[data-view-detailed]');"
            "if(v.checked){v.checked=false; v.dispatchEvent(new Event('change'));}"
        )
        overview = state()
        driver.execute_script(
            "const v=document.querySelector('[data-view-detailed]');"
            "v.checked=true; v.dispatchEvent(new Event('change'));"
        )
        detailed = state()

        assert overview["count"] > 0, "Unified model rendered no duplicate markers"
        assert overview["visible"] == 0, "duplicate marker leaked into Overview"
        assert detailed["visible"] == detailed["count"], "duplicate marker hidden in Detailed"
        assert all(title and title.startswith("Duplicate of UNIFIED.") for title in detailed["titles"])
    finally:
        driver.get(original_url)


@pytest.mark.parametrize("touch_origin", ["case", "popup", "outside"])
def test_keyboard_case_activation_after_touch_opens_one_popup(driver, touch_origin):
    driver.get(driver.current_url.split("?", 1)[0] + "?tab=tab-unified&transpose=0")
    cell = driver.find_element(By.CSS_SELECTOR, '.tab-panel.active td.cell[data-ttip-id]')
    driver.execute_script("arguments[0].scrollIntoView({block:'center'});", cell)
    if touch_origin == "case":
        _tap(driver, cell)
    elif touch_origin == "popup":
        driver.execute_script("arguments[0].click()", cell)
        tip = driver.execute_script("return arguments[0]._ttip", cell)
        _tap(driver, tip)
    else:
        _tap(driver, driver.find_element(By.TAG_NAME, "h1"))
    driver.execute_script("document.dispatchEvent(new KeyboardEvent('keydown',{key:'Escape',bubbles:true})); arguments[0].focus();", cell)
    cell.send_keys('\ue007')
    assert driver.execute_script("return document.querySelectorAll('.ttip-visible').length===1 && arguments[0]._ttip.classList.contains('ttip-pinned')", cell)


@pytest.mark.parametrize("transposed", [False, True], ids=["normal", "transposed"])
def test_matrix_labels_stay_visible_when_scrolled(driver, reference_page, transposed):
    driver.get(f"file://{reference_page}?tab=tab-toolcalling-batch&view=details&transpose=0")
    if transposed:
        toggled = driver.execute_script(
            """
            const t = document.querySelector('[data-transpose-toggle]');
            if (!t) return false;
            t.checked = true;
            t.dispatchEvent(new Event('change'));
            return true;
            """
        )
        if not toggled:
            pytest.skip("no transpose toggle on this page")

    result = driver.execute_script(
        """
        const table = document.querySelector(arguments[0]
          ? '.tab-panel.active table[data-transpose-table]'
          : '.tab-panel.active table[data-parity-table]');
        if (!table) return {error: 'active matrix not found'};
        const header = arguments[0]
          ? table.querySelector('thead tr.transpose-models th.tcorner-case')
          : table.querySelector('thead tr.matrix-groups th[data-col-control-group="model"]');
        const label = arguments[0]
          ? table.querySelector('tbody tr:not(.section) th.trow-case')
          : table.querySelector('tbody tr:not(.section) td.model');
        if (!header || !label) return {error: 'matrix labels not found'};
        const before = header.getBoundingClientRect();
        window.scrollTo(500, Math.max(0, table.getBoundingClientRect().top + window.scrollY + 500));
        const afterHeader = header.getBoundingClientRect();
        const afterLabel = label.getBoundingClientRect();
        return {
          beforeTop: before.top,
          afterTop: afterHeader.top,
          afterLeft: afterLabel.left,
          headerLeft: afterHeader.left,
          headerPosition: getComputedStyle(header).position,
          labelPosition: getComputedStyle(label).position,
          labelWidth: afterLabel.width,
          headerWidth: afterHeader.width,
          viewportHeight: window.innerHeight,
        };
        """,
        transposed,
    )
    try:
        assert "error" not in result, result.get("error")
        assert result["headerPosition"] == "sticky"
        assert result["labelPosition"] == "sticky"
        assert 0 <= result["afterTop"] < result["viewportHeight"]
        assert abs(result["afterLeft"] - result["headerLeft"]) < 2
        assert result["labelWidth"] > 0 and result["headerWidth"] > 0
    finally:
        driver.get(f"file://{reference_page}")


def test_popup_headings_include_compact_references(driver, reference_page):
    try:
        for transpose in ("0", "1"):
            driver.get(f"file://{reference_page}?tab=tab-unified&transpose={transpose}")
            table = "#tab-unified table[data-transpose-table]" if transpose == "1" else "#tab-unified table:not([data-transpose-table])"
            header_selector = "th.trow-case" if transpose == "1" else "th.case-sub"
            for prefix in ("1-", "deepseek_v41-", "7-arg_json_null"):
                header = driver.execute_script(
                    "return Array.from(document.querySelectorAll(arguments[0])).find(e=>"
                    "e.querySelector('.case-label-full').textContent.startsWith(arguments[1]));",
                    f"{table} {header_selector}", prefix,
                )
                full = header.find_element(By.CSS_SELECTOR, ".case-label-full").get_attribute("textContent")
                short = header.find_element(By.CSS_SELECTOR, ".case-label-short").get_attribute("textContent")
                cell = driver.execute_script(
                    "const h=arguments[0], t=h.closest('table');"
                    "return t.hasAttribute('data-transpose-table') ? h.parentElement.querySelector('td[data-ttip-id]') : "
                    "Array.from(t.querySelectorAll('tbody tr')).map(r=>r.querySelectorAll('td.cell')[Array.from(t.querySelectorAll('th.case-sub')).indexOf(h)]).find(e=>e && e.hasAttribute('data-ttip-id'));",
                    header,
                )
                assert cell is not None
                for target in (header, cell):
                    actions = ActionChains(driver)
                    driver.execute_script("document.dispatchEvent(new KeyboardEvent('keydown',{key:'Escape',bubbles:true}))")
                    WebDriverWait(driver, 5).until(lambda d: not d.find_elements(By.CSS_SELECTOR, ".ttip.ttip-visible"))
                    driver.execute_script("arguments[0].scrollIntoView({block:'center',inline:'center'});", target)
                    driver.execute_script("arguments[0].click()", target)
                    popup = WebDriverWait(driver, 5).until(lambda d: d.execute_script("""
                        const tip = arguments[0]._ttip;
                        return tip && tip.classList.contains('ttip-visible') ? tip : null;
                    """, target))
                    heading = popup.find_element(By.CSS_SELECTOR, ".ttip-head")
                    assert heading.text.startswith(f"{full} ({short})")
                    number = heading.find_element(By.CSS_SELECTOR, ".popup-case-reference .case-label-placeholder")
                    assert number.value_of_css_property("color") == "rgba(138, 143, 152, 1)"
                    assert float(number.value_of_css_property("font-size")[:-2]) < float(heading.value_of_css_property("font-size")[:-2])
                    assert heading.find_element(By.CSS_SELECTOR, "span[title]").get_attribute("title").startswith("UNIFIED.")
    finally:
        driver.get(f"file://{reference_page}?tab=tab-unified&transpose=0")


@pytest.fixture(scope="session")
def rendered_page(legacy_rendered_page):
    return legacy_rendered_page
