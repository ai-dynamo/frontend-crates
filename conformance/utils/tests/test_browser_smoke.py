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
import re
import shutil
import time

import pytest

selenium = pytest.importorskip("selenium")
from selenium import webdriver  # noqa: E402
from selenium.webdriver.common.action_chains import ActionChains  # noqa: E402
from selenium.webdriver.common.by import By  # noqa: E402
from selenium.webdriver.common.keys import Keys  # noqa: E402
from selenium.webdriver.chrome.options import Options  # noqa: E402
from selenium.webdriver.support.ui import WebDriverWait  # noqa: E402
from conformance.utils.tests.browser_helpers import click as _click, navigate as _navigate, script as _script, refresh as _refresh

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


_RECORD_FIRST_VISIBLE_MATRIX_JS = """
window.__firstVisibleMatrix = null;
window.__loadingMatrixInsertions = [];
function matrixGeometry(table) {
  const rect = table.getBoundingClientRect();
  let ancestorsVisible = true;
  for (let node = table; node; node = node.parentElement) {
    const style = getComputedStyle(node);
    if (style.display === 'none' || style.visibility === 'hidden' || style.visibility === 'collapse' || Number(style.opacity) === 0) {
      ancestorsVisible = false;
      break;
    }
  }
  return {width: rect.width, height: rect.height, ancestorsVisible};
}
let framePending = false;
function observeMatrix() {
  const panel = document.querySelector('#tab-unified');
  if (!panel) return;
  if (panel.dataset.loadState !== 'ready') {
    for (const table of panel.querySelectorAll('table[data-parity-table]')) {
      window.__loadingMatrixInsertions.push({cells: table.querySelectorAll('td.cell').length, ...matrixGeometry(table)});
    }
  }
  if (window.__firstVisibleMatrix || framePending) return;
  framePending = true;
  requestAnimationFrame(frameTime => {
    framePending = false;
    const active = document.querySelector('#tab-unified');
    if (!active) return;
    for (const table of active.querySelectorAll('table[data-parity-table], table[data-transpose-table]')) {
      const geometry = matrixGeometry(table);
      const cells = table.querySelectorAll('td.cell').length;
      if (cells && geometry.width > 0 && geometry.height > 0 && geometry.ancestorsVisible) {
        window.__firstVisibleMatrix = {
          kind: table.hasAttribute('data-transpose-table') ? 'transpose' : 'horizontal',
          transposeMode: document.body.classList.contains('transpose-mode'),
          state: active.dataset.loadState, cells, frameTime, ...geometry
        };
        break;
      }
    }
  });
}
const observer = new MutationObserver(observeMatrix);
observer.observe(document, {childList: true, subtree: true, attributes: true});
"""


def _chrome(viewer_url, force_hover, initial_navigation=True):
    opts = Options()
    for a in ("--headless=new", "--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage", "--window-size=1600,1200"):
        opts.add_argument(a)
    d = webdriver.Chrome(options=opts)
    if force_hover:
        d.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {"source": _FORCE_HOVER_JS})
    if initial_navigation:
        _navigate(d, f"{viewer_url}?transpose=0")
    d.implicitly_wait(2)
    return d


@pytest.fixture(scope="module")
def driver(viewer_url):
    d = _chrome(viewer_url, force_hover=True)
    yield d
    d.quit()


@pytest.fixture(scope="module")
def unforced_driver(viewer_url):
    """Chrome with its REAL media-query values — no matchMedia patch.

    Headless Chrome reports `(hover: hover)` and `(any-hover: hover)` false, which is
    precisely what Chrome reported on the desktop where hovering opened nothing. Every
    other lane forces those true and therefore cannot observe that failure. This is the
    negative control: it must pass while the browser claims no hover capability.
    """
    d = _chrome(viewer_url, force_hover=False)
    yield d
    d.quit()


@pytest.fixture(scope="module")
def touch_driver(viewer_url):
    """A driver on the TOUCH branch — no matchMedia patch, so `(hover: hover)` is false.

    Tap-to-pin only exists here: pinning is modal, so a mouse click must never pin or it
    would disable hover page-wide. The page decides from `PointerEvent.pointerType`, and
    the tap is delivered by `_tap`, which fires a genuine `pointerdown` carrying
    `pointerType: 'touch'` before the click. A bare `.click()` carries no pointerdown and
    would be classified (correctly) as not-touch, testing nothing; CDP mouse-to-touch
    emulation was tried first and deadlocks ActionChains. This fixture is what keeps the
    pin machinery under test.
    """
    d = _chrome(viewer_url, force_hover=False)
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
        const t = document.querySelector('#tab-unified .ttip.ttip-visible')
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
        const p = document.querySelector('#tab-unified') || document;
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
        const p = document.querySelector('#tab-unified') || document;
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
    _script(driver, "const v=document.querySelector('[data-view-detailed]'); if(v && !v.checked){v.checked=true; v.dispatchEvent(new Event('change'));}")
    # Find a cell that actually has a tooltip, fire the hover event the page listens for.
    found = _script(driver, """
        const tab = document.querySelector('#tab-unified') || document;
        for (const cell of tab.querySelectorAll('td.cell')) {
          if (cell.querySelector('.ttip')) {
            cell.dispatchEvent(new PointerEvent('pointerenter', {bubbles: true}));
            return true;
          }
        }
        return false;
        """)
    assert found, "no cell with a tooltip found"
    deadline = time.time() + 3
    visible = False
    while time.time() < deadline:
        visible = _script(driver, "return !!document.querySelector('.ttip.ttip-visible');")
        if visible:
            break
        time.sleep(0.1)
    assert visible, "tooltip did not become visible on hover"






def test_portalled_tooltip_tracks_the_document_scroll(driver):
    cell = driver.find_element(By.CSS_SELECTOR, "#tab-unified tbody tr:last-child td.cell")
    _script(driver, "arguments[0].scrollIntoView({block: 'center'});", cell)
    assert _script(driver, "return window.scrollY > 0")
    ActionChains(driver).move_to_element(cell).perform()
    time.sleep(1)
    result = _script(driver, """
        const cell = arguments[0];
        const tooltip = cell._ttip;
        const cellRect = cell.getBoundingClientRect();
        const tooltipRect = tooltip.getBoundingClientRect();
        return {
          portalled: tooltip.parentElement === document.body,
          topDelta: tooltipRect.top - cellRect.bottom,
          leftDelta: tooltipRect.left - cellRect.left,
        };
        """, cell)

    assert result["portalled"], result
    assert abs(result["topDelta"]) < 2, result
    assert result["leftDelta"] <= 0, result


def test_portalled_tooltip_stays_open_while_hovered(driver):
    cell = driver.find_element(By.CSS_SELECTOR, "#tab-unified td.cell")
    result = _script(driver, """
        const cell = arguments[0];
        cell.dispatchEvent(new MouseEvent('mouseenter', {bubbles: false, view: window}));
        return new Promise(resolve => setTimeout(() => {
          const tooltip = cell._ttip;
          cell.dispatchEvent(new MouseEvent('mouseleave', {bubbles: false, view: window}));
          tooltip.dispatchEvent(new MouseEvent('mouseenter', {bubbles: false, view: window}));
          resolve({tooltip});
        }, 850));
        """, cell)
    time.sleep(1)
    visible = _script(driver, "return arguments[0].classList.contains('ttip-visible');", result["tooltip"])

    assert visible


def test_transpose_keeps_one_open_tooltip(driver):
    _script(driver, """
        document.querySelectorAll('.ttip.ttip-visible').forEach(t => t.classList.remove('ttip-visible', 'ttip-pinned'));
        document.body.classList.remove('ttip-pin-mode', 'transpose-mode');
        const toggle = document.querySelector('[data-transpose-toggle]');
        if (toggle.checked) toggle.click();
        """)
    cell = driver.find_element(By.CSS_SELECTOR, "#tab-unified td.cell")
    result = _script(driver, """
        const cell = arguments[0];
        cell.dispatchEvent(new MouseEvent('mouseenter', {bubbles: false, view: window}));
        return new Promise(resolve => setTimeout(() => {
          const toggle = document.querySelector('[data-transpose-toggle]');
          const sourceId = cell.getAttribute('data-ttip-id');
          const before = document.querySelectorAll('.ttip.ttip-visible').length;
          toggle.click();
          const after = document.querySelectorAll('.ttip.ttip-visible').length;
          const clone = document.querySelector(
            '.transpose-table td.cell[data-ttip-id="' + sourceId + '"]'
          );
          resolve({
            before,
            after,
            portalled: [...document.querySelectorAll('.ttip.ttip-visible')]
              .filter(t => t.parentElement === document.body).length,
            cloneHasTooltip: Boolean(clone && clone._ttip),
          });
        }, 850));
        """, cell)

    assert result["before"] == 1, result
    assert result["after"] == 1, result
    assert result["portalled"] == 1, result
    assert result["cloneHasTooltip"], result


def test_focus_tooltip_stays_in_tab_order(driver):
    cell = driver.find_element(By.CSS_SELECTOR, "#tab-unified td.cell")
    result = _script(driver, """
        const cell = arguments[0];
        cell.dispatchEvent(new FocusEvent('focusin', {bubbles: true}));
        return new Promise(resolve => setTimeout(() => {
          const tooltip = cell._ttip;
          resolve({parent: tooltip.parentElement === cell, visible: tooltip.classList.contains('ttip-visible')});
        }, 850));
        """, cell)
    assert result == {"parent": True, "visible": True}






















def _set_transpose(driver, on):
    _script(driver, "const t=document.querySelector('[data-transpose-toggle]');"
        "if(t && t.checked!==arguments[0]){t.checked=arguments[0]; t.dispatchEvent(new Event('change'));}", on)
    time.sleep(0.2)


@pytest.mark.parametrize("override,expected", [("", False), ("1", True), ("true", True),
                                               ("0", False), ("false", False)])
def test_transpose_default_overrides_toggle_and_reload(driver, viewer_url, override, expected):
    url = f"{viewer_url}?tab=tab-unified"
    try:
        _navigate(driver, url + ("&transpose=" + override if override else ""))
        toggle = driver.find_element(By.CSS_SELECTOR, '[data-transpose-toggle]')
        assert toggle.is_selected() == expected
        assert _script(driver, "return document.body.classList.contains('transpose-mode')") == expected
        _click(toggle)
        selected = not expected
        assert toggle.is_selected() == selected
        assert _script(driver, "return new URL(location.href).searchParams.get('transpose')") == ("1" if selected else None)
        _refresh(driver)
        assert driver.find_element(By.CSS_SELECTOR, '[data-transpose-toggle]').is_selected() == selected
        assert _script(driver, "return document.body.classList.contains('transpose-mode')") == selected
    finally:
        _navigate(driver, f"{viewer_url}?transpose=0")


@pytest.mark.parametrize("override,expected", [("", "horizontal"), ("1", "transpose"),
                                               ("true", "transpose"), ("0", "horizontal"),
                                               ("false", "horizontal")])
def test_first_visible_matrix_matches_transpose_url(viewer_url, override, expected):
    d = _chrome(viewer_url, force_hover=False, initial_navigation=False)
    try:
        d.execute_cdp_cmd(
            "Page.addScriptToEvaluateOnNewDocument",
            {"source": _RECORD_FIRST_VISIBLE_MATRIX_JS},
        )
        url = f"{viewer_url}" + ("?transpose=" + override if override else "")
        _navigate(d, url)
        WebDriverWait(d, 10).until(lambda page: page.execute_script("return window.__firstVisibleMatrix !== null"))
        first = _script(d, "return window.__firstVisibleMatrix")
        skeletons = _script(d, "return window.__loadingMatrixInsertions")
        assert first is not None, "no report matrix became visible during initial rendering"
        assert skeletons, "no Unified matrix skeleton was observed before capture resolution"
        assert all(item["cells"] == 0 and item["ancestorsVisible"] and item["width"] > 0 and item["height"] > 0 for item in skeletons), skeletons
        assert first["kind"] == expected, f"first exposed matrix was {first}"
        assert first["cells"] > 0 and first["width"] > 0 and first["height"] > 0
        assert first["ancestorsVisible"] and first["state"] == "ready" and first["frameTime"] > 0
        assert first["transposeMode"] == (expected == "transpose")
    finally:
        d.quit()


def test_case_labels_squish_horizontally_and_expand_when_transposed(driver, viewer_url):
    try:
        _navigate(driver, f"{viewer_url}?tab=tab-unified")
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
        label_pairs = _script(driver, "return Array.from(document.querySelectorAll('#tab-unified th.case-sub a')).map(a=>["
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
        by_prefix_and_section = {}
        by_section = {}
        for full, short, _, group_key in label_pairs:
            prefix, number = short.split("-", 1)
            by_prefix_and_section.setdefault((prefix, group_key), []).append(int(number))
            by_section.setdefault((prefix, group_key), []).append(full.split("-", 1)[1])
        for descriptions in by_section.values():
            assert descriptions == sorted(descriptions)
        for numbers in by_prefix_and_section.values():
            assert numbers == sorted(set(numbers))

        ActionChains(driver).move_to_element(header).perform()
        _script(driver, "const h=arguments[0].closest('th.case-sub');"
            "h.dispatchEvent(new PointerEvent('pointerover',{bubbles:true}));"
            "h.dispatchEvent(new PointerEvent('pointerenter'));", header)
        deadline = time.time() + 3
        tip = None
        while time.time() < deadline:
            tips = driver.find_elements(By.CSS_SELECTOR, ".ttip.ttip-visible")
            if tips:
                tip = tips[0]
                break
            time.sleep(0.1)
        assert tip is not None, "case description tooltip did not open on hover"
        assert tip.find_element(By.CSS_SELECTOR, ".ttip-head > span[title]").get_attribute("title") == header.get_attribute("aria-label")
        assert f"{full_label} ({short_label})" in tip.find_element(By.CSS_SELECTOR, ".ttip-head").text
        assert full_label in tip.text
        assert tip.find_element(By.CSS_SELECTOR, ".ttip-head-desc").text

        _script(driver, "document.documentElement.setAttribute('data-theme', 'dark')")
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
        transpose_labels = _script(driver, "return Array.from(document.querySelectorAll('#tab-unified .transpose-table th.trow-case a')).map(a=>["
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
        text_inset = _script(driver, "const e=arguments[0],range=document.createRange();range.selectNodeContents(e);"
            "return range.getBoundingClientRect().left-e.getBoundingClientRect().left;", transpose_group)
        assert 8 <= text_inset <= 10

        _set_transpose(driver, False)
        assert header.find_element(By.CSS_SELECTOR, ".case-label-full").value_of_css_property("display") == "none"
        assert header.find_element(By.CSS_SELECTOR, ".case-label-short").value_of_css_property("display") != "none"
        expanded_width = group_cell.rect["width"]
        ActionChains(driver).move_to_element(driver.find_element(By.TAG_NAME, "h1")).perform()
        WebDriverWait(driver, 5).until(lambda d: not d.find_elements(By.CSS_SELECTOR, ".ttip.ttip-visible"))
        group_toggle = group_cell.find_element(By.CSS_SELECTOR, ".col-toggle")
        _click(group_toggle)
        assert "col-collapsed" in group_cell.get_attribute("class").split()
        assert group_cell.value_of_css_property("padding-left") == "2px"
        assert group_cell.rect["width"] < expanded_width
        _click(group_toggle)
        assert group_cell.value_of_css_property("padding-left") == "8px"
        assert abs(group_cell.rect["width"] - expanded_width) <= 1
    finally:
        _navigate(driver, f"{viewer_url}?transpose=0")


def test_transpose_builds_mirror_and_colors(driver, viewer_url):
    """Transpose preserves counts, Reference colors, and collapsed case groups."""
    saved_url = f"{viewer_url}?tab=tab-unified&view=details&transpose=0"
    try:
        _navigate(driver, saved_url)
        source = _script(driver, """
            const p = document.querySelector('#tab-unified');
            const table = p.querySelector('table[data-parity-table]');
            const group = table.querySelector('th.case-group[data-col-control-group]');
            return {cells: table.querySelectorAll('td.cell').length,
                    group: group?.dataset.colControlGroup || null,
                    groupLabel: group?.textContent || null};
        """)
        assert source["cells"] > 0 and source["group"], source
        group = driver.find_element(
            By.CSS_SELECTOR,
            f"#tab-unified table[data-parity-table] th.case-group[data-col-control-group='{source['group']}'] .col-toggle",
        )
        _click(group)
        WebDriverWait(driver, 5).until(lambda browser: "col-collapsed" in browser.find_element(
            By.CSS_SELECTOR,
            f"#tab-unified table[data-parity-table] th.case-group[data-col-control-group='{source['group']}']",
        ).get_attribute("class").split())
        _set_transpose(driver, True)
        info = _script(driver, """
            const p = document.querySelector('#tab-unified');
            const tt = p.querySelector('table[data-transpose-table]');
            if (!tt) return {built:false};
            const cells = tt.querySelectorAll('td.cell');
            let colored = 0;
            cells.forEach(function (c) {
              if (c.classList.contains('cmp-eq') || c.classList.contains('cmp-leak') || c.classList.contains('cmp-na')) colored++;
            });
            const hiddenRows = [...tt.querySelectorAll('tbody tr[data-col-hide-group]')]
              .filter(row => getComputedStyle(row).display === 'none');
            return {
              built: true,
              models: tt.querySelectorAll('th.tcol-model').length,
              rows: tt.querySelectorAll('th.trow-case').length,
              cells: cells.length,
              colored: colored,
              hiddenRows: hiddenRows.length,
              hiddenGroupRows: hiddenRows.filter(row => row.dataset.colHideGroup === arguments[0]).length,
            };
            """, source["group"])
        assert info["built"], "transposed mirror table was not built"
        assert info["models"] > 1, "expected multiple rotated model columns"
        assert info["rows"] > 1, "expected multiple case rows"
        assert info["cells"] == source["cells"], (
            f"transpose changed matrix cell count: {source['cells']} -> {info['cells']}"
        )
        assert info["cells"] > 0 and info["colored"] == info["cells"], (
            f"every cloned cell should be colored by applyCtl, got {info['colored']}/{info['cells']}"
        )
        assert info["hiddenGroupRows"] > 0, f"collapsed group {source['group']} was not hidden in the mirror: {info}"
    finally:
        _navigate(driver, saved_url)




def test_click_never_pins_where_hover_exists(driver):
    """Where hover exists, a click must NOT pin — hover stays live afterwards.

    Pinning is modal: `hoverAllowed()` returns false for every other cell while one is
    pinned. When a desktop click also pinned, a single stray click disabled hover for the
    whole page until the pin was released, and in details view "click elsewhere to
    dismiss" almost always lands on another wired cell, which only moves the pin. This
    pins the fix: on a hover-capable device no click pins anything, so hover cannot be
    switched off by clicking.
    """
    _script(driver, "const v=document.querySelector('[data-view-detailed]'); if(v && !v.checked){v.checked=true; v.dispatchEvent(new Event('change'));}")
    clicked = _script(driver, """
        const tab = document.querySelector('#tab-unified') || document;
        for (const el of tab.querySelectorAll('[data-ttip-wired]')) {
          if (!el.querySelector('.ttip')) continue;
          if (el.offsetParent === null) continue;
          el.click();
          window.__clicked = el;
          return true;
        }
        return false;
        """)
    if not clicked:
        pytest.skip("no wired elements with a tooltip on the active tab")
    time.sleep(0.6)
    assert not _script(driver, "return !!document.querySelector('.ttip.ttip-pinned');"), "a click pinned a popup on a hover-capable device"
    assert not _script(driver, "return document.body.classList.contains('ttip-pin-mode');"), "a click put the page into modal pin mode on a hover-capable device"

    # And hover still opens a popup on a DIFFERENT cell afterwards — the actual symptom.
    _script(driver, """
        const tab = document.querySelector('#tab-unified') || document;
        for (const el of tab.querySelectorAll('[data-ttip-wired]')) {
          if (!el.querySelector('.ttip')) continue;
          if (el.offsetParent === null || el === window.__clicked) continue;
          el.dispatchEvent(new PointerEvent('pointerenter', {bubbles: false}));
          return;
        }
        """)
    deadline = time.time() + 3
    shown = False
    while time.time() < deadline and not shown:
        shown = _script(driver, "return !!document.querySelector('.ttip.ttip-visible');")
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
        const tab = document.querySelector('#tab-unified') || document;
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
            const tab = document.querySelector('#tab-unified') || document;
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
        return _script(driver, """
            const tab = document.querySelector('#tab-unified') || document;
            const el = tab.querySelector('.toolbar-desc') || document.querySelector('.toolbar-desc');
            if (!el) { return null; }
            return getComputedStyle(el).display !== 'none';
            """)
    def set_detailed(on):
        _script(driver, "const v=document.querySelector('[data-view-detailed]');"
            "if(v && v.checked!==arguments[0]){v.checked=arguments[0]; v.dispatchEvent(new Event('change'));}", on)
        time.sleep(0.3)
    set_detailed(False)
    overview = legend_shown()
    if overview is None:
        pytest.skip("no .toolbar-desc legend on this page")
    set_detailed(True)
    detailed = legend_shown()
    assert not overview, "legend visible in Overview; it should be Detailed-only"
    assert detailed, "legend hidden in Detailed; it should be visible there"


def _schema_header(driver, transposed):
    return _script(driver, """
        const panel = document.querySelector('#tab-unified');
        const table = panel?.querySelector(arguments[0]
          ? 'table[data-transpose-table]' : 'table[data-parity-table]');
        if (!table) return null;
        const selector = arguments[0] ? 'th.trow-case' : 'th.case-sub';
        return [...table.querySelectorAll(selector)].find(header =>
          header.querySelector('.case-schema-indicator') && header.offsetWidth > 20
        ) || null;
        """, transposed)


def _dismiss_visible_tooltips(driver):
    ActionChains(driver).move_to_element(driver.find_element(By.TAG_NAME, "h1")).perform()
    _script(driver, """
        const owners = new Set([
          ...document.querySelectorAll('[data-ttip-wired]:hover'),
          ...[...document.querySelectorAll('.ttip-visible')].map(tip => tip._ttipOwner),
        ]);
        for (const owner of owners) owner?._ttipDispose?.();
        """)
    WebDriverWait(driver, 10).until(lambda browser: not browser.find_elements(By.CSS_SELECTOR, ".ttip-visible"))


def _open_schema_popup(driver, header):
    _dismiss_visible_tooltips(driver)
    _script(driver, "arguments[0].scrollIntoView({block: 'center', inline: 'center'})", header)
    _script(driver, "window.__buildTooltip(arguments[0])", header)
    ActionChains(driver).move_to_element(header).perform()
    return WebDriverWait(driver, 10).until(lambda browser: browser.execute_script("""
        const tip = arguments[0]._ttip || arguments[0].querySelector('.ttip');
        return tip && tip.classList.contains('ttip-visible') ? tip : null;
        """, header))


def _set_schema_view(driver, transposed):
    detailed = driver.find_element(By.CSS_SELECTOR, "[data-view-detailed]")
    if not detailed.is_selected():
        _click(detailed)
    toggle = driver.find_element(By.CSS_SELECTOR, "[data-transpose-toggle]")
    if toggle.is_selected() != transposed:
        _click(toggle)
    WebDriverWait(driver, 10).until(lambda browser: browser.execute_script("""
        return document.body.classList.contains('transpose-mode') === arguments[0]
          && Boolean(document.querySelector(arguments[0]
            ? '#tab-unified table[data-transpose-table]'
            : '#tab-unified table[data-parity-table]'));
        """, transposed))


def test_schema_popup_survives_themes_viewports_and_transpose(driver, viewer_url):
    saved_url = driver.current_url
    saved_size = driver.get_window_size()
    saved_theme = _script(driver, "return document.documentElement.dataset.theme")
    try:
        _navigate(driver, f"{viewer_url}?tab=tab-unified&view=details&transpose=0")
        for transposed in (False, True):
            _dismiss_visible_tooltips(driver)
            driver.set_window_size(1000, 700)
            _set_schema_view(driver, transposed)
            for theme in ("light", "dark"):
                if _script(driver, "return document.documentElement.dataset.theme") != theme:
                    _click(driver.find_element(By.CSS_SELECTOR, "[data-theme-toggle]"))
                header = _schema_header(driver, transposed)
                assert header is not None, f"no visible schema case in transpose={transposed}"
                tip = _open_schema_popup(driver, header)
                variant_rows = tip.find_elements(By.CSS_SELECTOR, ".ttip-grammar tbody tr")
                assert len(variant_rows) == 27, "schema popup should include each of the 27 captured families with a schema variant"
                assert all(row.find_elements(By.CSS_SELECTOR, "td") for row in variant_rows)
                disclosure = tip.find_element(By.CSS_SELECTOR, "details.case-schema")
                summary = disclosure.find_element(By.CSS_SELECTOR, "summary")
                for viewport_width, viewport_height in ((1000, 700), (480, 500)):
                    driver.set_window_size(viewport_width, viewport_height)
                    assert tip.is_displayed(), f"schema popup closed in {theme} theme at {viewport_width}px"
                    assert disclosure.get_dom_attribute("open") is None
                    collapsed_width = tip.size["width"]
                    _click(summary)
                    WebDriverWait(driver, 10).until(lambda browser: disclosure.get_dom_attribute("open") is not None)
                    WebDriverWait(driver, 10).until(lambda browser: tip.size["width"] >= collapsed_width)
                    rect = _script(driver, """
                        const box = arguments[0].getBoundingClientRect();
                        return {left: box.left, right: box.right, top: box.top, bottom: box.bottom,
                                width: box.width, viewportWidth: innerWidth, viewportHeight: innerHeight};
                        """, tip)
                    if viewport_width >= 800:
                        assert rect["width"] >= 720, rect
                    else:
                        assert rect["left"] >= 32 and rect["right"] <= rect["viewportWidth"] - 32, rect
                    assert 0 <= rect["top"] and rect["bottom"] <= rect["viewportHeight"], rect
                    blocks = disclosure.find_elements(By.CSS_SELECTOR, ".case-schema-json")
                    assert blocks and all(block.is_displayed() for block in blocks)
                    summary.click()
                    WebDriverWait(driver, 10).until(lambda browser: disclosure.get_dom_attribute("open") is None)
                # Resize is a live update; theme and transpose changes close hover popups.
                _dismiss_visible_tooltips(driver)
    finally:
        _navigate(driver, saved_url)
        driver.set_window_size(saved_size["width"], saved_size["height"])
        if _script(driver, "return document.documentElement.dataset.theme") != saved_theme:
            _click(driver.find_element(By.CSS_SELECTOR, "[data-theme-toggle]"))


@pytest.mark.parametrize("transposed", [False, True], ids=["normal", "transposed"])
def test_keyboard_schema_popup_stays_interactive_after_resize(driver, viewer_url, transposed):
    saved_url = driver.current_url
    saved_size = driver.get_window_size()
    try:
        driver.set_window_size(1000, 700)
        _navigate(driver, f"{viewer_url}?tab=tab-unified&view=details&transpose=0")
        _set_schema_view(driver, transposed)
        header = _schema_header(driver, transposed)
        assert header is not None, f"no visible schema case in transpose={transposed}"
        _script(driver, "arguments[0].scrollIntoView({block: 'center', inline: 'center'})", header)
        ActionChains(driver).move_to_element(driver.find_element(By.TAG_NAME, "h1")).perform()
        WebDriverWait(driver, 10).until(lambda browser: not browser.find_elements(By.CSS_SELECTOR, ".ttip-visible"))
        _script(driver, "arguments[0].querySelector('a').focus()", header)
        tip = WebDriverWait(driver, 10).until(lambda browser: browser.execute_script("""
            const tip = arguments[0]._ttip;
            return tip && tip.classList.contains('ttip-visible') ? tip : null;
            """, header))
        disclosure = tip.find_element(By.CSS_SELECTOR, "details.case-schema")
        summary = disclosure.find_element(By.CSS_SELECTOR, "summary")
        _click(summary)
        WebDriverWait(driver, 10).until(lambda browser: disclosure.get_dom_attribute("open") is not None)
        assert _script(driver, "return document.activeElement === arguments[0]", summary)
        driver.set_window_size(480, 500)
        WebDriverWait(driver, 10).until(lambda browser: browser.execute_script("""
            const rect = arguments[0].getBoundingClientRect();
            return rect.left >= 32 && rect.right <= innerWidth - 32
              && rect.top >= 0 && rect.bottom <= innerHeight;
            """, tip), "keyboard-origin schema popup did not fit after resize")
        summary.send_keys(Keys.ENTER)
        WebDriverWait(driver, 10).until(lambda browser: disclosure.get_dom_attribute("open") is None)
        assert tip.is_displayed()
        summary.send_keys(Keys.SPACE)
        WebDriverWait(driver, 10).until(lambda browser: disclosure.get_dom_attribute("open") is not None)
        summary.send_keys(Keys.ENTER)
        WebDriverWait(driver, 10).until(lambda browser: disclosure.get_dom_attribute("open") is None)
    finally:
        _navigate(driver, saved_url)
        driver.set_window_size(saved_size["width"], saved_size["height"])


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
        const p = document.querySelector('#tab-unified') || document;
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
        const p = document.querySelector('#tab-unified') || document;
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
        const p = document.querySelector('#tab-unified') || document;
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


@pytest.mark.parametrize("transposed", [False, True], ids=["normal", "transposed"])
def test_matrix_labels_stay_visible_when_scrolled(driver, viewer_url, transposed):
    _navigate(driver, f"{viewer_url}?view=details&transpose=0")
    if transposed:
        toggled = _script(driver, """
            const t = document.querySelector('[data-transpose-toggle]');
            if (!t) return false;
            t.checked = true;
            t.dispatchEvent(new Event('change'));
            return true;
            """)
        if not toggled:
            pytest.skip("no transpose toggle on this page")

    result = _script(driver, """
        const table = document.querySelector(arguments[0]
          ? '#tab-unified table[data-transpose-table]'
          : '#tab-unified table[data-parity-table]');
        if (!table) return {error: 'active matrix not found'};
        const header = arguments[0]
          ? table.querySelector('thead tr.transpose-models th.tcorner-case')
              : table.querySelector('thead tr.matrix-groups th[data-col-control-group="model"]');
        const label = arguments[0]
          ? table.querySelector('tbody tr:not(.section) th.trow-case')
          : table.querySelector('tbody tr:not(.section) td.model');
        if (!header || !label) return {error: 'matrix labels not found'};
            const before = header.getBoundingClientRect();
            const tableRect = table.getBoundingClientRect();
            const verticalScroll = tableRect.top + window.scrollY + Math.min(100, tableRect.height / 3);
            window.scrollTo(500, Math.max(0, verticalScroll));
        const afterHeader = header.getBoundingClientRect();
        const afterLabel = label.getBoundingClientRect();
        const ancestors = [];
        for (let node = header.parentElement; node && node !== document.documentElement; node = node.parentElement) {
          const style = getComputedStyle(node);
          ancestors.push({tag: node.tagName, id: node.id, className: String(node.className || ''),
            overflowX: style.overflowX, overflowY: style.overflowY,
            position: style.position, transform: style.transform, contain: style.contain,
            willChange: style.willChange, filter: style.filter,
            top: node.getBoundingClientRect().top, bottom: node.getBoundingClientRect().bottom});
        }
        return {
          beforeTop: before.top,
          afterTop: afterHeader.top,
          scrollY: window.scrollY,
          scrollElement: document.scrollingElement.tagName,
          documentScrollTop: document.scrollingElement.scrollTop,
          theadPosition: getComputedStyle(table.tHead).position,
          theadTop: table.tHead.getBoundingClientRect().top,
          stickyTop: getComputedStyle(header).top,
          ancestors,
          afterLeft: afterLabel.left,
          headerLeft: afterHeader.left,
          headerPosition: getComputedStyle(header).position,
          labelPosition: getComputedStyle(label).position,
          labelWidth: afterLabel.width,
          headerWidth: afterHeader.width,
          viewportHeight: window.innerHeight,
        };
        """, transposed)
    try:
        assert "error" not in result, result.get("error")
        assert result["headerPosition"] == "sticky"
        assert result["labelPosition"] == "sticky"
        if not 0 <= result["afterTop"] < result["viewportHeight"]:
            pytest.fail(f"Sticky header geometry: {result}", pytrace=False)
        assert abs(result["afterLeft"] - result["headerLeft"]) < 2, result
        assert result["labelWidth"] > 0 and result["headerWidth"] > 0
    finally:
        _navigate(driver, f"{viewer_url}")


def test_popup_headings_include_compact_references(driver, viewer_url):
    try:
        for transpose in ("0", "1"):
            _navigate(driver, f"{viewer_url}?tab=tab-unified&transpose={transpose}")
            table = "#tab-unified table[data-transpose-table]" if transpose == "1" else "#tab-unified table:not([data-transpose-table])"
            header_selector = "th.trow-case" if transpose == "1" else "th.case-sub"
            for prefix in ("1-", "deepseek_v41-", "7-arg_json_null"):
                header = _script(driver, "return Array.from(document.querySelectorAll(arguments[0])).find(e=>"
                    "e.querySelector('.case-label-full').textContent.startsWith(arguments[1]));", f"{table} {header_selector}", prefix)
                full = header.find_element(By.CSS_SELECTOR, ".case-label-full").get_attribute("textContent")
                short = header.find_element(By.CSS_SELECTOR, ".case-label-short").get_attribute("textContent")
                cell = _script(driver, "const h=arguments[0], t=h.closest('table');"
                    "return t.hasAttribute('data-transpose-table') ? h.parentElement.querySelector('td[data-ttip-id]') : "
                    "Array.from(t.querySelectorAll('tbody tr')).map(r=>r.querySelectorAll('td.cell')[Array.from(t.querySelectorAll('th.case-sub')).indexOf(h)]).find(e=>e && e.hasAttribute('data-ttip-id'));", header)
                assert cell is not None
                for target in (header, cell):
                    actions = ActionChains(driver)
                    actions.w3c_actions.pointer_action.move_to_location(2, 2)
                    actions.perform()
                    WebDriverWait(driver, 5).until(lambda d: not d.find_elements(By.CSS_SELECTOR, ".ttip.ttip-visible"))
                    _script(driver, "arguments[0].scrollIntoView({block:'center',inline:'center'});", target)
                    ActionChains(driver).move_to_element(target).perform()
                    popup = WebDriverWait(driver, 5).until(lambda d: _script(d, """
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
        _navigate(driver, f"{viewer_url}?tab=tab-unified&transpose=0")
