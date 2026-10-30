# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-browser popup, schema, accessibility, and matrix layout checks."""
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
  let ancestorsVisible = table.checkVisibility({checkOpacity:true,checkVisibilityCSS:true});
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
  if (window.__firstVisibleMatrix) return;
  const panel = document.querySelector('#tab-unified');
  if (!panel) return;
  if (panel.dataset.loadState !== 'ready') {
    for (const table of panel.querySelectorAll('table[data-comparison-table]')) {
      window.__loadingMatrixInsertions.push({cells: table.querySelectorAll('.matrix-cell').length, ...matrixGeometry(table)});
    }
  }
  if (window.__firstVisibleMatrix || framePending) return;
  framePending = true;
  requestAnimationFrame(frameTime => {
    framePending = false;
    const active = document.querySelector('#tab-unified');
    if (!active) return;
    for (const table of active.querySelectorAll('table[data-comparison-table]')) {
      const geometry = matrixGeometry(table);
      const cells = table.querySelectorAll('.matrix-cell').length;
      if (cells && geometry.width > 0 && geometry.height > 0 && geometry.ancestorsVisible) {
        window.__firstVisibleMatrix = {
          kind: table.hasAttribute('data-transpose-table') ? 'transpose' : 'horizontal',
          transposeMode: document.body.classList.contains('transpose-mode'),
          state: active.dataset.loadState, cells, frameTime, ...geometry
        };
        observer.disconnect();
        break;
      }
    }
  });
}
const observer = new MutationObserver(observeMatrix);
observer.observe(document, {childList: true, subtree: true, attributes: true});
"""


@pytest.fixture(autouse=True)
def restore_browser_state(request):
    browsers = [request.getfixturevalue(name) for name in ('driver', 'unforced_driver', 'touch_driver')
                if name in request.fixturenames]
    saved = [(d, d.current_url, d.get_window_size(), d.execute_script(
        "return {theme:document.documentElement.dataset.theme,scroll:[scrollX,scrollY]}")) for d in browsers]
    try:
        for d in browsers:
            _dismiss_visible_tooltips(d)
        yield
    finally:
        for d, url, size, state in saved:
            d.set_window_size(size['width'], size['height'])
            _navigate(d, url)
            if d.execute_script("return document.documentElement.dataset.theme") != state['theme']:
                _click(d.find_element(By.CSS_SELECTOR, "[data-theme-toggle]"))
            d.execute_script("scrollTo(...arguments[0])", state['scroll'])


def _chrome(viewer_url, force_hover, initial_navigation=True):
    opts = Options()
    for a in ("--headless=new", "--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage", "--window-size=1600,1200"):
        opts.add_argument(a)
    d = webdriver.Chrome(options=opts)
    if force_hover:
        d.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {"source": _FORCE_HOVER_JS})
    if initial_navigation:
        _navigate(d, f"{viewer_url}?transpose=0")
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
    """Use real media queries and explicit touch events for pointer provenance."""
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


def test_real_mouse_activation_opens_a_visible_pinned_tooltip(unforced_driver):
    """Unified cells use explicit activation so browsing does not build popups on hover."""
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
        const c = [...p.querySelectorAll('.matrix-cell')].filter(e => e.offsetParent !== null
                                                             && e.querySelector('.ttip'))[20];
        if (!c) return null;
        c.scrollIntoView({block: 'center'});
        return c;
        """
    )
    assert el is not None, "no visible primary matrix cell"
    webdriver.ActionChains(drv).move_to_element(el).click().perform()
    _assert_tooltip_actually_visible(drv, "real mouse activation, no forced media queries")
    assert drv.execute_script("return !!document.querySelector('.ttip.ttip-pinned');")


def test_keyboard_activation_opens_a_visible_pinned_tooltip(unforced_driver):
    """Unified popup content is built when a keyboard user activates a focused cell."""
    drv = unforced_driver
    opened = drv.execute_script(
        """
        const p = document.querySelector('#tab-unified') || document;
        for (const el of p.querySelectorAll('[data-ttip-click="pin"]')) {
          if (!el.querySelector('.ttip') || el.offsetParent === null) continue;
          el.scrollIntoView({block: 'center'});
          el.focus();
          el.dispatchEvent(new KeyboardEvent('keydown', {key: 'Enter', bubbles: true}));
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
    _assert_tooltip_actually_visible(drv, "keyboard activation")
    assert drv.execute_script("return !!document.querySelector('.ttip.ttip-pinned');")


def test_cell_activation_shows_tooltip(driver):
    """Activating a detail cell builds and pins its lazy popup."""
    saved_url = driver.current_url
    saved_scroll = _script(driver, "return [window.scrollX, window.scrollY]")
    try:
        _dismiss_visible_tooltips(driver)
        _script(driver, "const v=document.querySelector('[data-view-detailed]'); if(v && !v.checked){v.checked=true; v.dispatchEvent(new Event('change'));}")
        # Activate an actual detail popup after clearing any restored URL pin.
        found = _script(driver, """
            const tab = document.querySelector('#tab-unified') || document;
            for (const cell of tab.querySelectorAll('.matrix-cell')) {
              if (cell.querySelector('.ttip')) {
                cell.click();
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
        assert visible, "tooltip did not become visible on activation"
        assert _script(driver, "return !!document.querySelector('.ttip.ttip-pinned');")
    finally:
        _navigate(driver, saved_url)
        _script(driver, "window.scrollTo(arguments[0], arguments[1])", *saved_scroll)


def test_portalled_tooltip_tracks_the_document_scroll(driver, viewer_url):
    saved_url = driver.current_url
    saved_scroll = _script(driver, "return [window.scrollX, window.scrollY]")
    try:
        # The overview can fit without scrolling; own the detailed matrix state.
        _navigate(driver, f"{viewer_url}?tab=tab-unified&view=details&transpose=0")
        cell = driver.find_element(By.CSS_SELECTOR, "#tab-unified tbody tr:last-child .matrix-cell")
        _script(driver, "arguments[0].scrollIntoView({block: 'center'});", cell)
        assert _script(driver, "return window.scrollY > 0")
        _click(cell)
        result = _script(driver, """
        const cell = arguments[0];
        const tooltip = cell._ttip;
        const cellRect = cell.getBoundingClientRect();
        const tooltipRect = tooltip.getBoundingClientRect();
        return {
          portalled: tooltip.parentElement === document.body,
          top: tooltipRect.top,
          left: tooltipRect.left,
          right: tooltipRect.right,
          viewportWidth: innerWidth,
          controlsBottom: document.querySelector('#tab-unified .capture-slots').getBoundingClientRect().bottom,
        };
        """, cell)

        assert result["portalled"], result
        assert result["top"] >= 0, result
        assert result["left"] >= 0 and result["right"] <= result["viewportWidth"], result
    finally:
        _navigate(driver, saved_url)
        _script(driver, "window.scrollTo(arguments[0], arguments[1])", *saved_scroll)


def test_portalled_tooltip_stays_open_while_hovered(driver):
    saved_url = driver.current_url
    saved_scroll = _script(driver, "return [window.scrollX, window.scrollY]")
    try:
        _dismiss_visible_tooltips(driver)
        cell = driver.find_element(By.CSS_SELECTOR, '#tab-unified .matrix-cell[data-ttip-click="pin"]')
        result = _script(driver, """
            const cell = arguments[0];
            cell.click();
            const tooltip = cell._ttip;
            cell.dispatchEvent(new MouseEvent('mouseleave', {bubbles: false, view: window}));
            tooltip.dispatchEvent(new MouseEvent('mouseenter', {bubbles: false, view: window}));
            return {tooltip};
            """, cell)
        visible = _script(driver, "return arguments[0].classList.contains('ttip-visible');", result["tooltip"])

        assert visible and "ttip-pinned" in result["tooltip"].get_attribute("class")
    finally:
        _navigate(driver, saved_url)
        _script(driver, "window.scrollTo(arguments[0], arguments[1])", *saved_scroll)


def test_transpose_keeps_one_open_tooltip(driver, viewer_url):
    saved_url = driver.current_url
    try:
        _navigate(driver, f"{viewer_url}?tab=tab-unified&transpose=0")
        _dismiss_visible_tooltips(driver)
        cell = driver.find_element(By.CSS_SELECTOR, "#tab-unified .matrix-cell[data-ttip-click='pin']")
        cell.click()
        WebDriverWait(driver, 10).until(lambda page: page.find_element(By.CSS_SELECTOR, ".ttip.ttip-pinned"))
        source_id = cell.get_attribute("data-ttip-key")
        before = driver.find_element(By.CSS_SELECTOR, ".ttip.ttip-pinned .ttip-head").text
        _click(driver.find_element(By.CSS_SELECTOR, "[data-transpose-toggle]"))
        popups = driver.find_elements(By.CSS_SELECTOR, ".ttip.ttip-pinned")
        assert len(popups) == 1 and popups[0].is_displayed()
        assert popups[0].find_element(By.CSS_SELECTOR, '.ttip-head').text == before
        assert source_id == driver.execute_script("return document.querySelector('.ttip-pinned')._ttipOwner.dataset.ttipKey")
    finally:
        _navigate(driver, saved_url)


def test_focus_tooltip_stays_in_tab_order(driver):
    _dismiss_visible_tooltips(driver)
    cell = driver.find_element(By.CSS_SELECTOR, "#tab-unified .matrix-cell[data-ttip-click='pin']")
    result = _script(driver, """
        const cell = arguments[0];
        const wasFocusable = cell.tabIndex === 0;
        cell.focus();
        cell.dispatchEvent(new KeyboardEvent('keydown', {key: 'Enter', bubbles: true}));
        const tooltip = cell._ttip;
        return {parent: tooltip.parentElement === document.body, visible: tooltip.classList.contains('ttip-visible'), pinned: tooltip.classList.contains('ttip-pinned'), wasFocusable};
        """, cell)
    assert result == {"parent": True, "visible": True, "pinned": True, "wasFocusable": True}






















def _set_transpose(driver, on):
    _script(driver, "const t=document.querySelector('[data-transpose-toggle]');"
        "if(t && t.checked!==arguments[0]){t.checked=arguments[0]; t.dispatchEvent(new Event('change'));}", on)
    time.sleep(0.2)


@pytest.mark.parametrize("override,expected", [("", True), ("1", True), ("true", True),
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
        assert _script(driver, "return new URL(location.href).searchParams.get('vertical')") == ("1" if selected else "0")
        _refresh(driver)
        assert driver.find_element(By.CSS_SELECTOR, '[data-transpose-toggle]').is_selected() == selected
        assert _script(driver, "return document.body.classList.contains('transpose-mode')") == selected
    finally:
        _navigate(driver, f"{viewer_url}?transpose=0")


@pytest.mark.parametrize("override,expected", [("", "transpose"), ("1", "transpose"),
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
        assert all(item["cells"] == 0 and not item["ancestorsVisible"] for item in skeletons), skeletons
        assert first["kind"] == expected, f"first exposed matrix was {first}"
        assert first["cells"] > 0 and first["width"] > 0 and first["height"] > 0
        assert first["ancestorsVisible"] and first["state"] == "ready" and first["frameTime"] > 0
        assert first["transposeMode"] == (expected == "transpose")
    finally:
        d.quit()


def _assert_case_popup_heading(driver, target, heading, full_label, short_label):
    model = _script(driver, "return window.__conformanceTooltipModel(arguments[0].closest('[data-ttip-id]'));", target)
    case_id = model.get("popupCase", {}).get("case_id", full_label).removeprefix("UNIFIED.")
    assert heading.text.startswith(f"{case_id} ({short_label})")
    description = model.get("desc") or model.get("description")
    if description:
        descriptions = heading.find_elements(By.CSS_SELECTOR, ".ttip-head-desc")
        assert len(descriptions) == 1
        assert descriptions[0].text == " ".join(description.replace("`", "").split())


def test_case_labels_squish_horizontally_and_expand_when_transposed(driver, viewer_url):
    _navigate(driver, f"{viewer_url}?vertical=0")
    header = driver.find_element(By.CSS_SELECTOR, '.matrix-case-heading[data-ttip-id]')
    full_label = _script(driver, "return window.__conformanceTooltipModel(arguments[0]).popupCase.label", header)
    short_label, case_id = header.text, header.get_attribute('aria-label')
    assert re.fullmatch(r'\d+-\d+', short_label)
    assert full_label not in header.text
    labels = _script(driver, """
        return [...document.querySelectorAll('.matrix-case-heading')].map(header => ({
          id:header.getAttribute('aria-label'), label:header.querySelector('.matrix-case-text').textContent,
          column:window.__conformanceTooltipModel(header).popupCase
        }));
    """)
    assert [item['id'] for item in labels] == _script(driver, 'return window.__conformanceComparison.view.columns.map(c=>c.case_id)')
    for item in labels:
        column = item['column']
        if column.get('display_number') is not None:
            assert item['label'] == f"{column['compact_prefix']}-{column['display_number']}"
    header.send_keys(Keys.ENTER)
    popup = driver.find_element(By.CSS_SELECTOR, '.ttip-pinned .ttip-head')
    _assert_case_popup_heading(driver, header, popup, full_label, short_label)
    _set_transpose(driver, True)
    header = driver.find_element(By.CSS_SELECTOR, f'.matrix-case-heading[aria-label="{case_id}"]')
    assert full_label in header.text and short_label in header.text
    _set_transpose(driver, False)
    assert driver.find_element(By.CSS_SELECTOR, f'.matrix-case-heading[aria-label="{case_id}"]').text == short_label


def test_orientation_preserves_matrix_outcomes_and_counts(driver, viewer_url):
    _navigate(driver, f"{viewer_url}?vertical=0")
    def inventory():
        return _script(driver, """
            const t=document.querySelector('[data-comparison-table]');
            return {cells:t.querySelectorAll('.matrix-cell').length,
              statuses:[...t.querySelectorAll('.matrix-cell[data-ttip-key]')].map(e=>[e.dataset.ttipKey,[...e.querySelectorAll('.matrix-segments i')].map(s=>s.className)]).sort((a,b)=>a[0].localeCompare(b[0])),
              counts:[...document.querySelectorAll('.capture-summary')].map(e=>e.textContent)};
        """)
    before = inventory()
    assert before['cells'] > 0 and before['statuses']
    _set_transpose(driver, True)
    assert inventory() == before
    assert driver.find_element(By.CSS_SELECTOR, '[data-comparison-table]').get_attribute('data-transpose-table') is not None




@pytest.mark.parametrize("transposed", [False, True], ids=["normal", "transposed"])
def test_every_wired_element_stays_pinned(touch_driver, transposed):
    """TOUCH: tapping anything with a popup PINS it, and the same tap must not close it.

    Runs on `touch_driver` to exercise touch provenance as well as explicit activation.

    The document-level outside-click handler decides what counts as "inside". While it
    enumerated classes, it kept drifting from the set `attachTooltip` actually wires: first
    `th.matrix-case-heading` was pinnable but not inside, then the transpose headers (`th.tcol-model`,
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


def test_capture_legend_tracks_visible_statuses(driver):
    detailed = driver.find_element(By.CSS_SELECTOR, "[data-view-detailed]")
    saved_detailed = detailed.is_selected()
    try:
        for enabled in (False, True):
            if detailed.is_selected() != enabled:
                _click(detailed)
            legend = driver.find_element(By.CSS_SELECTOR, "#tab-unified .comparison-legend")
            assert legend.is_displayed()
            assert legend.get_attribute("aria-label") == "Outcome colors"
            assert not driver.find_elements(By.CSS_SELECTOR, "#tab-unified .toolbar-desc")
            assert _script(driver, """
                const panel = document.querySelector('#tab-unified');
                const table = panel.querySelector('[data-comparison-table]');
                const statuses = new Set([...table.querySelectorAll('.matrix-cell .matrix-status')]
                  .map(segment => segment.className.split('status-')[1]));
                const legend = new Set([...panel.querySelectorAll('.comparison-legend [data-status]')].map(item => item.dataset.status));
                return statuses.size > 0 && statuses.size === legend.size && [...statuses].every(status => legend.has(status));
            """)
    finally:
        if detailed.is_selected() != saved_detailed:
            _click(detailed)


def _schema_header(driver, transposed):
    return _script(driver, """
        return [...document.querySelectorAll('.matrix-case-heading[data-ttip-id]')].find(header =>
          window.__conformanceTooltipModel(header)?.popupCase?.schemas?.length && header.offsetWidth > 0
        ) || null;
    """)


def _dismiss_visible_tooltips(driver, move_pointer=True):
    if move_pointer:
        ActionChains(driver).move_to_element(driver.find_element(By.TAG_NAME, "h1")).perform()
    _script(driver, """
        document.dispatchEvent(new KeyboardEvent('keydown', {key:'Escape',bubbles:true}));
        document.querySelectorAll('.ttip-pinned .ttip-close').forEach(b=>b.click());
        document.querySelectorAll('[data-ttip-wired]').forEach(e=>e.dispatchEvent(new PointerEvent('pointerleave')));
        """)
    WebDriverWait(driver, 10).until(lambda browser: not browser.find_elements(By.CSS_SELECTOR, ".ttip-visible"))


def _open_schema_popup(driver, header):
    _dismiss_visible_tooltips(driver)
    _script(driver, "arguments[0].scrollIntoView({block: 'center', inline: 'center'})", header)
    header.send_keys(Keys.ENTER)
    return WebDriverWait(driver, 10).until(lambda browser: browser.execute_script("""
        const tip = arguments[0]._ttip || arguments[0].querySelector('.ttip');
        return tip && tip.classList.contains('ttip-pinned') ? tip : null;
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
          && Boolean(document.querySelector('[data-comparison-table]'));
        """, transposed))


@pytest.mark.parametrize("transposed", [False, True])
def test_recovery_popup_has_contiguous_variant_blocks(driver, viewer_url, transposed):
    saved_url = driver.current_url
    try:
        _navigate(driver, f"{viewer_url}?tab=tab-unified&view=details&transpose={int(transposed)}")
        header = driver.find_element(By.CSS_SELECTOR, '#tab-unified .matrix-case-heading[aria-label="UNIFIED.5-tool_no_close"]')
        tip = _open_schema_popup(driver, header)
        groups = _script(driver, """
            return [...arguments[0].querySelectorAll('.ttip-grammar tbody')].map(group => ({
              key: group.dataset.variant,
              headings: group.querySelectorAll('th[scope="rowgroup"]').length,
              heading: group.querySelector('th[scope="rowgroup"]')?.textContent,
              families: [...group.querySelectorAll('td.grf')].map(cell => cell.firstChild.textContent),
              outputs: [...group.querySelectorAll('tr:not(.gr-variant-heading)')]
                .map(row => row.querySelectorAll('td.gro').length),
            }));
            """, tip)
        assert [group['key'] for group in groups] == [
            'tool_no_close', 'gpt_oss_quoted_truncated_tool_then_final',
            'gpt_oss_truncated_tool_resynchronizes_at_next_message',
        ]
        families = _script(driver, "return window.__conformanceModel.rows.filter(row => !row.section).map(row => row.model_label || row.family)")
        for group in groups:
            assert group['headings'] == 1
            assert 'Not applicable to' not in group['heading']
            assert group['families'] == families
            assert len(set(group['outputs'])) == 1
    finally:
        _navigate(driver, saved_url)


def test_schema_popup_opens_and_fits_across_themes_viewports_and_transpose(driver, viewer_url):
    saved_url = driver.current_url
    saved_size = driver.get_window_size()
    saved_theme = _script(driver, "return document.documentElement.dataset.theme")
    try:
        _navigate(driver, f"{viewer_url}?tab=tab-unified&view=details&transpose=0")
        for transposed in (False, True):
            for theme in ("light", "dark"):
                for viewport_width, viewport_height in ((1000, 700), (480, 500)):
                    driver.set_window_size(viewport_width, viewport_height)
                    _set_schema_view(driver, transposed)
                    if _script(driver, "return document.documentElement.dataset.theme") != theme:
                        _click(driver.find_element(By.CSS_SELECTOR, "[data-theme-toggle]"))
                    header = _schema_header(driver, transposed)
                    assert header is not None, f"no visible schema case in transpose={transposed}"
                    tip = _open_schema_popup(driver, header)
                    rows = _script(driver, """
                        const tab = window.__conformanceModel;
                        const id = arguments[0].getAttribute('aria-label');
                        const column = tab.columns.find(item => (item.case_id || tab.case_prefix + item.label) === id);
                        const groups = new Map();
                        tab.rows.filter(row => !row.section).forEach(row => {
                          const cell = row.cells[column.sub];
                          const variants = cell?.variants || [cell];
                          variants.forEach(variant => {
                            const key = variant?.sub || column.sub;
                            if (!groups.has(key)) groups.set(key, []);
                            groups.get(key).push([key, row.model_label || row.family]);
                          });
                        });
                        const keys = [...groups.keys()].sort((a, b) => Number(b === column.sub) - Number(a === column.sub));
                        const expected = keys.flatMap(key => groups.get(key));
                        const rows = [...arguments[1].querySelectorAll('.ttip-grammar tbody tr:not(.gr-variant-heading)')];
                        return {expected, actual: rows.flatMap(row => [...row.querySelectorAll('td.grf')]
                          .map(cell => [row.parentElement.dataset.variant, cell.firstChild.textContent])),
                          haveCells: rows.length > 0 && rows.every(row => row.querySelector('td'))};
                        """, header, tip)
                    assert rows['actual'] == rows['expected'], "schema popup must retain every selected family's folded variants"
                    assert rows['haveCells']
                    disclosure = tip.find_element(By.CSS_SELECTOR, "details.case-schema")
                    summary = disclosure.find_element(By.CSS_SELECTOR, "summary")
                    assert disclosure.get_dom_attribute("open") is None
                    collapsed_width = tip.size["width"]
                    summary.send_keys(Keys.ENTER)
                    assert _script(driver, "return document.activeElement === arguments[0]", summary)
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
                    assert _script(driver, """
                        const blocks = [...arguments[0].querySelectorAll('.case-schema-json')];
                        return blocks.length > 0 && blocks.every(block =>
                          block.checkVisibility({checkOpacity: true, checkVisibilityCSS: true}));
                    """, disclosure)
                    summary.send_keys(Keys.ENTER)
                    WebDriverWait(driver, 10).until(lambda browser: disclosure.get_dom_attribute("open") is None)
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
        header.send_keys(Keys.ENTER)
        tip = WebDriverWait(driver, 10).until(lambda browser: browser.execute_script("""
            const tip = arguments[0]._ttip;
            return tip && tip.classList.contains('ttip-pinned') ? tip : null;
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

    size = driver.get_window_size()
    try:
        driver.set_window_size(480, 700)
        _script(driver, "document.querySelector('.comparison-matrix-scroll').scrollIntoView({block:'start'});window.scrollBy(200,200)")
        WebDriverWait(driver, 10).until(lambda page: page.execute_script("""
            const t=document.querySelector('[data-comparison-table]'), s=t.parentElement;
            const header=t.tHead.rows[0].cells[0], h=header.getBoundingClientRect();
            const labels=[...t.tBodies[0].querySelectorAll('tr > th:first-child')]
              .filter(cell=>{const r=cell.getBoundingClientRect();return r.top>=h.bottom&&r.bottom<innerHeight});
            return getComputedStyle(header).position==='sticky' && labels.length>0
              && labels.every(cell=>getComputedStyle(cell).position==='sticky'&&Math.abs(cell.getBoundingClientRect().left)<=1)
              && s.scrollTop===0 && s.scrollLeft===0 && scrollY>0 && scrollX>0
              && Math.abs(h.top)<=1 && h.width>0 && h.bottom<=innerHeight;
        """))
    finally:
        driver.set_window_size(size["width"], size["height"])


def test_popup_headings_include_compact_references(driver, viewer_url):
    for transpose in ('0', '1'):
        _navigate(driver, f"{viewer_url}?transpose={transpose}")
        for prefix in ('1-', None, '7-arg_json_null'):
            header = _script(driver, r"""
                return [...document.querySelectorAll('.matrix-case-heading')].find(header => {
                  const column=window.__conformanceTooltipModel(header).popupCase;
                  return arguments[0] === null ? !/^\d+-/.test(column.label) : column.label.startsWith(arguments[0]);
                });
            """, prefix)
            assert header is not None, prefix
            column = _script(driver, 'return window.__conformanceTooltipModel(arguments[0]).popupCase', header)
            short = f"{column['compact_prefix']}-{column['display_number']}" if column.get('display_number') is not None else column['label']
            cell = _script(driver, """
                return [...document.querySelectorAll('.matrix-cell[data-ttip-id]')].find(cell => {
                  const model=window.__conformanceCellModel(cell);
                  return model?.sub===arguments[0] || model?.case_id===arguments[1];
                });
            """, column['sub'], column.get('case_id'))
            assert cell is not None
            for target in (header, cell):
                _dismiss_visible_tooltips(driver)
                target.send_keys(Keys.ENTER)
                popup = driver.find_element(By.CSS_SELECTOR, '.ttip-pinned')
                _assert_case_popup_heading(driver, target, popup.find_element(By.CSS_SELECTOR, '.ttip-head'), column['label'], short)
                assert popup.find_element(By.CSS_SELECTOR, '.ttip-head span[title]').get_attribute('title').startswith('UNIFIED.')
                _dismiss_visible_tooltips(driver)


def test_three_cards_and_toolbar_do_not_overlap(driver, viewer_url):
    _navigate(driver, f"{viewer_url}?tab=tab-unified")
    cards = driver.find_elements(By.CSS_SELECTOR, '.capture-slot')
    assert len(cards) == 3
    assert all(c.find_elements(By.CSS_SELECTOR, '.capture-summary') for c in cards)
    assert _script(driver, """
        const cards=[...document.querySelectorAll('.capture-slot')].map(e=>e.getBoundingClientRect());
        const toolbar=document.querySelector('.comparison-toolbar').getBoundingClientRect();
        return cards.every((a,i)=>cards.slice(i+1).every(b=>a.right<=b.left || b.right<=a.left || a.bottom<=b.top || b.bottom<=a.top))
          && cards.every(a=>a.right<=toolbar.left || toolbar.right<=a.left || a.bottom<=toolbar.top || toolbar.bottom<=a.top);
    """)
