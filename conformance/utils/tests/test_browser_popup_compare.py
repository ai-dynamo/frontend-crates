# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Browser test for the per-chunk popup's candidate columns (real clicks, real DOM).

The popup's grid columns are the compare-bar candidates: selecting REF = Dynamo v2 and
Compare-with = Dynamo v1 must show exactly `input | v2 (REF-marked) | v1` — no vLLM /
SGLang columns unless checked — and unchecking v1 must hide its column live. The JS was
previously shipped without ever being executed; this test exists so that can't recur.

Skips when Selenium or headless Chrome aren't available (same policy as
test_browser_smoke.py).
"""
from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import pytest

selenium = pytest.importorskip("selenium")
from selenium import webdriver  # noqa: E402
from selenium.webdriver.chrome.options import Options  # noqa: E402

pytestmark = pytest.mark.skipif(
    not any(
        shutil.which(b)
        for b in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser")
    ),
    reason="no headless Chrome available",
)

def _dynamo_key(impl: str) -> str:
    """Candidate key of the LATEST stream capture dir for a Dynamo impl
    (`dynamo_v1` = jail reference, `dynamo_v2` = stream parser); older version
    dirs are capture history and render as extra candidates."""
    root = Path(
        os.environ.get("CONFORMANCE_FIXTURES_ROOT")
        or Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
        / "dynamo/conformance-fixtures"
    )
    import re as _re

    dirs = [
        d
        for d in (root / "toolcalling/fixtures-stream-v2").glob(f"{impl}-*")
        if d.is_dir()
    ]
    latest = max(
        dirs, key=lambda d: [int(x) for x in _re.findall(r"\d+", d.name)], default=None
    )
    if latest is None:
        pytest.skip("dynamo stream fixture dirs not cached", allow_module_level=True)
    return latest.name.replace(".", "-")


V2_KEY = _dynamo_key("dynamo_v2")
V1_KEY = _dynamo_key("dynamo_v1")


@pytest.fixture(scope="module")
def driver(rendered_page):
    opts = Options()
    for a in ("--headless=new", "--no-sandbox", "--disable-gpu",
              "--disable-dev-shm-usage", "--window-size=1600,1200"):
        opts.add_argument(a)
    try:
        d = webdriver.Chrome(options=opts)
    except Exception as exc:  # noqa: BLE001 — environment without a usable driver
        pytest.skip(f"could not start Chrome webdriver: {exc}")
    d.get(f"file://{rendered_page}")
    d.implicitly_wait(2)
    yield d
    d.quit()


def _open_stream_tab(driver):
    ok = driver.execute_script(
        """
        for (const b of document.querySelectorAll('.tab-button')) {
          const t = b.getAttribute('data-tab-target') || '';
          if (t.includes('stream') && t.includes('toolcalling')) { b.click(); return t; }
        }
        return null;
        """
    )
    assert ok, "no Tool Calling stream tab button found"
    time.sleep(0.3)


def _select(driver, ref_key, cmp_keys):
    """Set the active panel's Reference radio + Compare-with checkboxes, fire change.
    Reference radios are made exclusive explicitly (the star inputs are per-column,
    so a stale checked radio elsewhere must be cleared)."""
    driver.execute_script(
        """
        const [refKey, cmpKeys] = arguments;
        const ctl = document.querySelector('.tab-panel.active .cmpctl');
        if (!ctl) { return false; }
        for (const r of ctl.querySelectorAll('input.cmp-ref')) {
          const want = r.value === refKey;
          if (r.checked !== want) { r.checked = want; }
          if (want) { r.dispatchEvent(new Event('change', {bubbles: true})); }
        }
        for (const cb of ctl.querySelectorAll('input.cmp-on')) {
          const want = cmpKeys.includes(cb.value) || cb.value === refKey;
          if (!cb.disabled && cb.checked !== want) {
            cb.checked = want; cb.dispatchEvent(new Event('change', {bubbles: true}));
          }
        }
        return true;
        """,
        ref_key, list(cmp_keys),
    )
    time.sleep(0.3)


def _grid_column_order(driver):
    """Visible [data-cand] header keys of the first candidate grid, in DOM order."""
    return driver.execute_script(
        """
        const tab = document.querySelector('.tab-panel.active') || document;
        for (const grid of tab.querySelectorAll('.ttip-chunks')) {
          const ths = Array.from(grid.querySelectorAll('th[data-cand]'));
          if (!ths.length) { continue; }
          return ths.filter(t => !t.classList.contains('col-hidden'))
                    .map(t => t.getAttribute('data-cand'));
        }
        return null;
        """
    )


def _chart_tooltip_has_cand_list(driver):
    """True if any tooltip in the active panel contains BOTH a candidate grid and
    the legacy per-candidate list sections (they must be mutually exclusive)."""
    return driver.execute_script(
        """
        const tab = document.querySelector('.tab-panel.active') || document;
        for (const tip of tab.querySelectorAll('.ttip')) {
          if (tip.querySelector('.ttip-chunks [data-cand]') && tip.querySelector('.cand')) {
            return true;
          }
        }
        return false;
        """
    )


def _grid_state(driver):
    """{key: {hidden, ref}} for the first candidate grid in the active panel."""
    return driver.execute_script(
        """
        const tab = document.querySelector('.tab-panel.active') || document;
        for (const grid of tab.querySelectorAll('.ttip-chunks')) {
          const ths = grid.querySelectorAll('th[data-cand]');
          if (!ths.length) { continue; }
          const out = {};
          ths.forEach(th => {
            out[th.getAttribute('data-cand')] = {
              hidden: th.classList.contains('col-hidden'),
              ref: th.classList.contains('col-ref'),
            };
          });
          return out;
        }
        return null;
        """
    )


def _assembled_text(driver, key):
    return driver.execute_script(
        """
        const key = arguments[0];
        const tab = document.querySelector('.tab-panel.active') || document;
        for (const row of tab.querySelectorAll('.ttip-chunks tr.ttip-final')) {
          const td = row.querySelector(`td[data-cand="${key}"]`);
          if (td) { return td.textContent; }
        }
        return null;
        """,
        key,
    )


def test_popup_columns_follow_ref_and_compare(driver):
    _open_stream_tab(driver)
    _select(driver, V2_KEY, [V1_KEY])

    state = _grid_state(driver)
    assert state, "no candidate-column grid found in the stream tab"
    assert V2_KEY in state and V1_KEY in state, f"expected both Dynamo columns, got {state}"

    visible = sorted(k for k, s in state.items() if not s["hidden"])
    assert visible == sorted([V2_KEY, V1_KEY]), (
        f"popup must show exactly REF + compare-with columns, got visible={visible}"
    )
    assert state[V2_KEY]["ref"] and not state[V1_KEY]["ref"], (
        f"REF marking wrong: {state}"
    )


def test_v1_assembled_is_not_doubled(driver):
    _open_stream_tab(driver)
    _select(driver, V2_KEY, [V1_KEY])
    text = _assembled_text(driver, V1_KEY)
    assert text is not None, "no assembled cell for the Dynamo v1 candidate"
    assert "get_weatherget_weather" not in text, f"doubled v1 output on the page: {text!r}"


def _active_base(driver):
    """The active panel's selected Reference key, or None when no star is chosen."""
    return driver.execute_script(
        """
        const ctl = document.querySelector('.tab-panel.active .cmpctl');
        const r = ctl && ctl.querySelector('input.cmp-ref:checked');
        return r ? r.value : null;
        """
    )


def test_clicking_the_active_star_again_clears_the_reference(driver):
    # A radio has no native uncheck; clicking the already-selected Reference star must
    # toggle it OFF -> no base -> every cell paints cmp-nobase (the panel clears).
    _open_stream_tab(driver)
    _select(driver, V2_KEY, [])
    assert _active_base(driver) == V2_KEY, "precondition: V2 should be the Reference"

    # Click the visible ★ LABEL (what a user actually clicks), NOT the offscreen radio —
    # the radio gets only a synthesized click, so the handler must key off the label.
    clicked = driver.execute_script(
        """
        const ctl = document.querySelector('.tab-panel.active .cmpctl');
        const r = ctl && ctl.querySelector('input.cmp-ref:checked');
        const label = r && (r.closest('label') || r.parentElement);
        const star = label && (label.querySelector('.star') || label);
        if (!star) { return false; }
        star.dispatchEvent(new MouseEvent('mousedown', {bubbles: true}));
        star.dispatchEvent(new MouseEvent('click', {bubbles: true}));
        return true;
        """
    )
    assert clicked, "no checked Reference star to click"
    time.sleep(0.3)

    assert _active_base(driver) is None, "clicking the active star again must clear the Reference"
    # With no base, applyCtl paints every scored cell cmp-nobase and nothing else.
    counts = driver.execute_script(
        """
        const tab = document.querySelector('.tab-panel.active');
        const cells = tab.querySelectorAll('td.cell[data-cmp]');
        let nobase = 0, colored = 0;
        cells.forEach(c => {
          if (c.classList.contains('cmp-nobase')) { nobase++; }
          if (c.classList.contains('cmp-eq') || c.classList.contains('cmp-leak')) { colored++; }
        });
        return {total: cells.length, nobase, colored};
        """
    )
    assert counts["total"] > 0, "no scored cells in the stream tab"
    assert counts["colored"] == 0, f"cells still colored after clearing the Reference: {counts}"
    assert counts["nobase"] == counts["total"], f"not all cells cleared to cmp-nobase: {counts}"


def _click_active_star(driver):
    return driver.execute_script(
        """
        const ctl = document.querySelector('.tab-panel.active .cmpctl');
        const r = ctl && ctl.querySelector('input.cmp-ref:checked');
        const label = r && (r.closest('label') || r.parentElement);
        const star = label && (label.querySelector('.star') || label);
        if (!star) { return false; }
        star.dispatchEvent(new MouseEvent('mousedown', {bubbles: true}));
        star.dispatchEvent(new MouseEvent('click', {bubbles: true}));
        return true;
        """
    )


def test_active_star_stays_when_a_compare_is_selected(driver):
    # With a Compare checkbox active there's still a chart to show, so clicking the
    # active star must NOT clear the Reference (only an empty compare set may unstar).
    _open_stream_tab(driver)
    _select(driver, V2_KEY, [V1_KEY])
    assert _active_base(driver) == V2_KEY, "precondition: V2 is the Reference"

    assert _click_active_star(driver), "no checked Reference star to click"
    time.sleep(0.3)
    assert _active_base(driver) == V2_KEY, (
        "star must stay while a Compare checkbox is selected"
    )

    # Clear the compare, THEN the same click gesture must unstar (empty set path).
    _select(driver, V2_KEY, [])
    assert _click_active_star(driver)
    time.sleep(0.3)
    assert _active_base(driver) is None, "with no compares, clicking the star must clear it"


def test_unchecking_compare_hides_its_column(driver):
    _open_stream_tab(driver)
    _select(driver, V2_KEY, [V1_KEY])
    assert not _grid_state(driver)[V1_KEY]["hidden"]
    _select(driver, V2_KEY, [])
    state = _grid_state(driver)
    assert state[V1_KEY]["hidden"], f"unchecked v1 column still visible: {state}"
    assert not state[V2_KEY]["hidden"], "REF column must stay visible"


def test_ref_column_is_first_and_follows_restar(driver):
    """The REF candidate reads FIRST (leftmost after input), and re-starring another
    candidate moves that one to the front."""
    _open_stream_tab(driver)
    _select(driver, V2_KEY, [V1_KEY])
    order = _grid_column_order(driver)
    assert order and order[0] == V2_KEY, f"REF (v2) must be the first column, got {order}"
    _select(driver, V1_KEY, [V2_KEY])
    order = _grid_column_order(driver)
    assert order and order[0] == V1_KEY, f"after re-star, v1 must lead, got {order}"
    # restore the default-ish selection for subsequent tests
    _select(driver, V2_KEY, [V1_KEY])


def test_chart_tooltips_have_no_candidate_list(driver):
    """Wherever the candidate chart renders, the per-candidate list sections are gone
    (the chart's output row carries the same info)."""
    _open_stream_tab(driver)
    assert not _chart_tooltip_has_cand_list(driver), (
        "stream tab: tooltip shows BOTH the candidate chart and the legacy list"
    )


def _open_tab(driver, target_substr):
    ok = driver.execute_script(
        """
        const want = arguments[0];
        for (const b of document.querySelectorAll('.tab-button')) {
          const t = b.getAttribute('data-tab-target') || '';
          if (t.includes(want)) { b.click(); return t; }
        }
        return null;
        """,
        target_substr,
    )
    assert ok, f"no tab button matching {target_substr!r}"
    time.sleep(0.3)


def test_batch_tab_uses_candidate_chart(driver):
    """The merged TC (batch data) tab renders the same candidate chart (single
    output row), REF first, no duplicate list."""
    _open_tab(driver, "toolcalling-batch")
    order = _grid_column_order(driver)
    assert order, "batch tab: no candidate chart found"
    base = driver.execute_script(
        "const c=document.querySelector('.tab-panel.active .cmpctl input.cmp-ref:checked');"
        "return c?c.value:null;"
    )
    assert base and order[0] == base, f"batch REF {base!r} must lead, got {order}"
    assert not _chart_tooltip_has_cand_list(driver), (
        "batch tab: tooltip shows BOTH chart and legacy list"
    )


def test_reasoning_tabs_use_candidate_chart(driver):
    """Both Reasoning tabs render the candidate chart with the dynamo/vllm/sglang
    columns (stream additionally lists its input chunks as rows)."""
    for target in ("reasoning-batch", "reasoning-stream"):
        _open_tab(driver, target)
        order = _grid_column_order(driver)
        assert order, f"{target}: no candidate chart found"
        assert set(order) <= {"dynamo_v1", "vllm_python", "sglang_python"}, (
            f"{target}: unexpected candidate keys {order}"
        )
        assert not _chart_tooltip_has_cand_list(driver), (
            f"{target}: tooltip shows BOTH chart and legacy list"
        )
