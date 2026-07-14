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

import shutil
import time

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

V2_KEY = "dynamo_rust-0-1-11"
V1_KEY = "dynamo_rust-3-0-0"


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
        assert set(order) <= {"dynamo", "vllm", "sglang"}, (
            f"{target}: unexpected candidate keys {order}"
        )
        assert not _chart_tooltip_has_cand_list(driver), (
            f"{target}: tooltip shows BOTH chart and legacy list"
        )
