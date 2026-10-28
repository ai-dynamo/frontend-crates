# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the comparison surface through its real browser controls."""
import json
import re
import threading
from urllib.parse import urlsplit, parse_qs

import pytest
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select, WebDriverWait

from test_browser_viewer import _driver, _server, _ready, _choose_engine, _published_capture_index


def counts(driver):
    return driver.execute_script("return window.__conformanceComparison.counts")


@pytest.mark.parametrize("query,vertical", [("", True), ("?vertical=0", False), ("?transpose=0", False)])
def test_defaults_before_first_paint_and_shared_state(published_viewer, query, vertical):
    server, _ = _server(root=published_viewer)
    driver = _driver()
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {"source": """
      window.firstComparisonPaint = null;
      new MutationObserver(() => {
        if (window.firstComparisonPaint === null && document.querySelector('.comparison-matrix'))
          window.firstComparisonPaint = document.body.classList.contains('transpose-mode');
      }).observe(document, {childList:true,subtree:true});
    """})
    try:
        driver.set_window_size(1600,1000)
        driver.get(f"http://127.0.0.1:{server.server_port}/{query}")
        _ready(driver, "tab-unified",reference=False)
        assert driver.execute_script("return window.firstComparisonPaint") is vertical
        assert len(driver.find_elements("css selector", ".capture-slot")) == 3
        assert len(driver.find_elements("css selector", ".tabs .tab-button")) == 0
        sha = driver.find_element("css selector", ".source-commit")
        assert re.fullmatch(r"[a-f0-9]{12}", sha.text)
        assert sha.get_attribute("title") in sha.get_attribute("href")
        assert sha.get_attribute("title") in sha.get_attribute("aria-label")
        assert driver.find_element("css selector", "[data-layout='stacked']").text == ("Stacked columns" if vertical else "Stacked rows")
        assert "%" not in driver.find_element("id", "viewer").text
        disclosure = driver.find_element("css selector", ".legacy-report-details > summary")
        disclosure.click()
        assert driver.find_element("css selector", ".legacy-report-details table[data-parity-table]").is_displayed()
        driver.find_element("css selector", "[data-transpose-toggle]").click()
        assert driver.find_element("css selector", ".legacy-report-details table[data-parity-table]").is_displayed()
        driver.find_element("css selector", "[data-transpose-toggle]").click()
        disclosure.click()
        before = counts(driver)
        driver.find_element("css selector", "[data-focus='problems']").click()
        assert counts(driver) == before
        driver.find_element("css selector", "[data-layout='stacked']").click()
        assert counts(driver) == before
        search = driver.find_element("css selector", "[data-case-search]")
        search.send_keys("dotted_name_minimal")
        after = counts(driver)
        assert 0 < after["dynamo"]["total"] < before["dynamo"]["total"]
        driver.find_element("css selector", "[data-peer-only]").click()
        peer = counts(driver)
        driver.refresh(); _ready(driver,"tab-unified",reference=False)
        assert counts(driver) == peer
        assert driver.find_element("css selector", "[data-focus='problems']").get_attribute("aria-pressed") == "true"
        assert driver.find_element("css selector", "[data-case-search]").get_attribute("value") == "dotted_name_minimal"
        driver.find_element("css selector", "[data-theme-toggle]").click()
        assert counts(driver) == peer
        driver.set_window_size(480,850)
        assert driver.execute_script("return document.documentElement.scrollWidth <= innerWidth + 1")
        for card in driver.find_elements("css selector", ".capture-slot"):
            assert card.rect["width"] <= 480
    finally:
        driver.quit(); server.shutdown(); server.server_close()


def test_matrix_keyboard_drawer_and_reload(published_viewer):
    server, _ = _server(root=published_viewer)
    driver = _driver()
    try:
        driver.get(f"http://127.0.0.1:{server.server_port}/")
        _ready(driver,"tab-unified",reference=False)
        cell = driver.find_element("css selector", ".comparison-matrix .matrix-cell[data-ttip-id]")
        cell.send_keys(Keys.ENTER)
        popup = WebDriverWait(driver,10).until(lambda d: d.find_element("css selector", ".comparison-output.ttip-pinned"))
        assert popup.find_elements("css selector", ".golden-out-cap, [data-cand=golden]")
        assert parse_qs(urlsplit(driver.current_url).query)["popup"]
        driver.refresh(); _ready(driver,"tab-unified",reference=False)
        assert len(driver.find_elements("css selector", ".comparison-output.ttip-pinned")) == 1
        driver.find_element("tag name","body").send_keys(Keys.ESCAPE)
        assert not driver.find_elements("css selector", ".comparison-output.ttip-pinned")
        assert "popup" not in parse_qs(urlsplit(driver.current_url).query)
    finally:
        driver.quit(); server.shutdown(); server.server_close()


def test_reference_url_same_engine_and_stale_request(historical_published_viewer):
    started, release = threading.Event(), threading.Event()
    server, _ = _server(root=historical_published_viewer, gate=(started,release),
                        gate_path=_published_capture_index(historical_published_viewer,"dynamo_v1","5.1.3"))
    driver = _driver()
    try:
        driver.get(f"http://127.0.0.1:{server.server_port}/?tab=tab-unified&slots_tab-unified=dynamo%2Cdynamo%400.7.18%2C&cmp_tab-unified=dynamo%2Cdynamo%400.7.18")
        _ready(driver,"tab-unified",reference=False)
        selected = driver.execute_script("return window.__conformanceController.selection(document.getElementById('tab-unified')).selected")
        assert selected == ["dynamo","dynamo@0.7.18"]
        assert "0.7.18" in driver.find_element("css selector","[data-slot-capture='1']").get_attribute("value")
        duplicate = driver.find_element("css selector","[data-slot-capture='0'] option[value='dynamo@0.7.18']")
        assert not duplicate.is_enabled()
        before = counts(driver)
        Select(driver.find_element("css selector","[data-slot-capture='0']")).select_by_value("dynamo_v1@5.1.3")
        assert started.wait(20)
        assert counts(driver) == before
        Select(driver.find_element("css selector","[data-slot-capture='0']")).select_by_value("dynamo")
        WebDriverWait(driver,20).until(lambda d: d.execute_script("return document.getElementById('tab-unified').dataset.loadState === 'ready'"))
        expected_url, expected_counts = driver.current_url, counts(driver)
        release.set()
        WebDriverWait(driver,20).until(lambda d: d.execute_script("return window.__conformanceLoader.pending === 0"))
        assert counts(driver) == expected_counts
        assert driver.current_url == expected_url
        assert driver.execute_script("return window.__conformanceModel.candidates.some(c => c.key === 'dynamo')")
    finally:
        release.set(); driver.quit(); server.shutdown(); server.server_close()


def test_peer_filter_missing_index_requests_regeneration(published_viewer):
    server, requests = _server(root=published_viewer)
    driver = _driver()
    try:
        driver.get(f"http://127.0.0.1:{server.server_port}/")
        _ready(driver,"tab-unified",reference=False)
        driver.execute_script("delete window.__conformanceModel.peer_availability")
        request_count = len(requests)
        driver.find_element("css selector","[data-peer-only]").click()
        assert "Regenerate" in driver.find_element("css selector",".comparison-notice").text
        assert counts(driver)["dynamo"]["total"] == 0
        assert not any("/viewer-data/" in path for path in requests[request_count:])
    finally:
        driver.quit(); server.shutdown(); server.server_close()
