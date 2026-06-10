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
        "document.querySelector('input[name=\"parity-view\"][value=\"details\"]').click();"
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


def test_vllm_rust_option_hidden_on_reasoning(driver):
    """The vLLM Rust parser radio shows on a tool-calling tab and hides on Reasoning."""
    def vllm_rust_visible():
        return driver.execute_script(
            """
            const lbl = document.querySelector('label[data-parser-option="vllm_rust"]');
            return !!(lbl && lbl.offsetParent !== null);
            """
        )

    def click_tab(panel_id):
        driver.execute_script(
            "document.querySelector(arguments[0]).click();",
            f'.tab-button[data-tab-target="{panel_id}"]',
        )
        time.sleep(0.2)

    click_tab("tab-toolcalling-stream-on-batch")
    assert vllm_rust_visible(), "vLLM Rust option should show on a tool-calling tab"
    click_tab("tab-reasoning-batch")
    assert not vllm_rust_visible(), "vLLM Rust option should hide on Reasoning"
    click_tab("tab-toolcalling-stream-on-batch")
    assert vllm_rust_visible(), "vLLM Rust option should reappear on a tool-calling tab"


def test_conformance_mode_shows_one_marker_per_cell(driver):
    """In Details + Conformance, a cell must not show BOTH the per-engine status
    marker and the cross-engine parity marker — they're absolutely positioned in
    the same box and visibly overlap if both display (the B7 CSS-order regression
    that produced garbled markers like a struck-through `=`)."""
    driver.execute_script(
        """
        document.querySelector('input[name="parity-view"][value="details"]').click();
        const par = document.querySelector('input[data-parity-toggle]');
        if (par && !par.checked) par.click();
        """
    )
    time.sleep(0.2)
    # For the selected parser, count cells where both its status marker and its
    # parity marker are visibly rendered. Must be zero.
    both_visible = driver.execute_script(
        """
        const sel = (document.body.className.match(/parser-(\\w+)/) || [])[1];
        const tab = document.querySelector('.tab-panel.active') || document;
        const vis = (el) => el && el.offsetParent !== null
            && getComputedStyle(el).display !== 'none';
        let overlap = 0;
        for (const cell of tab.querySelectorAll('td.cell')) {
          const status = cell.querySelector('.marker-' + sel);
          const parity = cell.querySelector('.marker-parity-' + sel);
          if (vis(status) && vis(parity)) overlap++;
        }
        return overlap;
        """
    )
    assert both_visible == 0, (
        f"{both_visible} cell(s) show both the status and parity marker overlapping"
    )
    # And the parity marker IS the one shown for non-trivial cells.
    parity_shown = driver.execute_script(
        """
        const sel = (document.body.className.match(/parser-(\\w+)/) || [])[1];
        const tab = document.querySelector('.tab-panel.active') || document;
        for (const cell of tab.querySelectorAll('td.cell')) {
          const parity = cell.querySelector('.marker-parity-' + sel);
          if (parity && parity.textContent.trim()
              && parity.offsetParent !== null
              && getComputedStyle(parity).display !== 'none') return true;
        }
        return false;
        """
    )
    assert parity_shown, "no parity marker is visible in Conformance mode"
