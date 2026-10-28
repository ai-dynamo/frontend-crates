# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the comparison surface through its real browser controls."""
import json
import re
import threading
from urllib.parse import urlsplit, parse_qs

import pytest
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import Select, WebDriverWait

from test_browser_viewer import _driver, _server, _ready, _choose_engine, _clear_slot, _published_capture_index


def counts(driver):
    return driver.execute_script("return window.__conformanceComparison.counts")


@pytest.mark.parametrize("width", [1600, 480])
@pytest.mark.parametrize("captures", [1, 2])
def test_long_case_hover_fits_matrix_scroller(published_viewer, width, captures):
    server, _ = _server(root=published_viewer)
    driver = _driver()
    try:
        driver.set_window_size(width, 1000)
        driver.get(f"http://127.0.0.1:{server.server_port}/?vertical=0")
        _ready(driver, "tab-unified", reference=False)
        if captures == 2:
            _choose_engine(driver, 1, "sglang")
            _ready(driver, "tab-unified", reference=False)
            driver.find_element("css selector", "[data-layout='stacked']").click()
        label = driver.execute_script("""
            const headers=[...document.querySelectorAll('.matrix-case-heading')];
            const header=headers.sort((a,b)=>window.__conformanceTooltipModel(b).popupCase.label.length
              -window.__conformanceTooltipModel(a).popupCase.label.length)[0];
            header.scrollIntoView({block:'center',inline:'end'});
            return header.querySelector('[data-hover-hint]');
        """)
        ActionChains(driver).move_to_element(label).perform()
        WebDriverWait(driver, 5).until(lambda d: d.execute_script(
            "return getComputedStyle(arguments[0],'::after').opacity==='1'", label))
        geometry = driver.execute_script("""
            const a=arguments[0], r=a.getBoundingClientRect(), s=getComputedStyle(a,'::after');
            const scroller=a.closest('.comparison-matrix-scroll'),clip=scroller.getBoundingClientRect();
            return {x:r.left+parseFloat(s.left),y:r.top+parseFloat(s.top),w:parseFloat(s.width),h:parseFloat(s.height),
              left:Math.max(0,clip.left+scroller.clientLeft),right:Math.min(innerWidth,clip.left+scroller.clientLeft+scroller.clientWidth),
              top:Math.max(0,clip.top+scroller.clientTop),bottom:Math.min(innerHeight,clip.top+scroller.clientTop+scroller.clientHeight)};
        """, label)
        assert geometry['left'] <= geometry['x'] and geometry['x'] + geometry['w'] <= geometry['right'], geometry
        assert geometry['top'] <= geometry['y'] and geometry['y'] + geometry['h'] <= geometry['bottom'], geometry
    finally:
        driver.quit(); server.shutdown(); server.server_close()


@pytest.mark.parametrize("vertical", [False, True])
@pytest.mark.parametrize("correction", ["layout", "categories", "labels"])
def test_requested_matrix_corrections(published_viewer, vertical, correction):
    server, _ = _server(root=published_viewer)
    driver = _driver()
    try:
        driver.set_window_size(1600, 1000)
        driver.get(f"http://127.0.0.1:{server.server_port}/?vertical={int(vertical)}&layout_tab-unified=stacked")
        _ready(driver, "tab-unified", reference=False)
        if correction == "layout":
            assert not driver.find_element("css selector", "[data-layout='stacked']").is_displayed()
            before = counts(driver)
            _choose_engine(driver, 1, "sglang")
            _ready(driver, "tab-unified", reference=False)
            stacked = driver.find_element("css selector", "[data-layout='stacked']")
            assert stacked.is_displayed()
            assert stacked.text == ("Stacked columns" if vertical else "Stacked rows")
            assert stacked.get_attribute("aria-pressed") == "true"
            driver.find_element("css selector", "[data-layout='split']").click()
            assert counts(driver)["dynamo"] == before["dynamo"]
            _clear_slot(driver, 1)
            _ready(driver, "tab-unified", reference=False)
            assert not driver.find_element("css selector", "[data-layout='stacked']").is_displayed()
        elif correction == "categories":
            assert not driver.find_elements("css selector", ".category-chips")
            button = driver.find_element("css selector", "[data-comparison-table] [data-group]")
            key = button.get_attribute("data-group")
            before = counts(driver)
            button.send_keys(Keys.SPACE)
            selector = f"[data-comparison-table] [data-group='{key}']"
            assert driver.find_element("css selector", selector).get_attribute("aria-pressed") == "false"
            reduced = counts(driver)
            assert reduced["dynamo"]["total"] < before["dynamo"]["total"]
            driver.refresh()
            _ready(driver, "tab-unified", reference=False)
            assert counts(driver) == reduced
            driver.find_element("css selector", selector).send_keys(Keys.SPACE)
            assert counts(driver) == before
            driver.set_window_size(1600, 500)
            driver.execute_script("const s=document.querySelector('.comparison-matrix-scroll');s.scrollIntoView({block:'start'});s.scrollTop=200")
            WebDriverWait(driver, 5).until(lambda d: d.execute_script("""
                const table=document.querySelector('[data-comparison-table]');
                const top=table.tHead.rows[0].cells[0].getBoundingClientRect();
                const next=table.tHead.rows[1].cells[0].getBoundingClientRect();
                return table.parentElement.scrollTop>0 && Math.abs(next.top-top.bottom)<=1;
            """))
        else:
            header = driver.find_element("css selector", ".matrix-case-heading")
            full = driver.execute_script("return window.__conformanceTooltipModel(arguments[0]).popupCase.label", header)
            label = header.find_element("css selector", "[data-hover-hint]")
            assert label.get_attribute("data-hover-hint") == "Click to show " + full
            if vertical:
                assert full in header.text
            else:
                assert re.fullmatch(r"\d+-\d+", header.text)
                assert full not in header.text
                assert header.rect["width"] < 100
                ActionChains(driver).move_to_element(label).perform()
                WebDriverWait(driver, 5).until(lambda d: d.execute_script(
                    "return getComputedStyle(arguments[0],'::after').visibility==='visible' && getComputedStyle(arguments[0],'::after').opacity==='1'", label))
                assert full in driver.execute_script("return getComputedStyle(arguments[0],'::after').content", label)
                assert driver.execute_script("""
                    const a=arguments[0], r=a.getBoundingClientRect(), s=getComputedStyle(a,'::after');
                    const box=a.closest('.comparison-matrix-scroll').getBoundingClientRect();
                    const x=r.left+parseFloat(s.left), y=r.top+parseFloat(s.top);
                    return x>=Math.max(0,box.left) && x+parseFloat(s.width)<=Math.min(innerWidth,box.right)
                      && y>=Math.max(0,box.top) && y+parseFloat(s.height)<=Math.min(innerHeight,box.bottom);
                """, label)
                assert not driver.find_elements("css selector", ".ttip-pinned")
            header.send_keys(Keys.ENTER)
            assert driver.find_element("css selector", ".ttip-pinned .ttip-head").text
    finally:
        driver.quit(); server.shutdown(); server.server_close()


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
        assert not driver.find_element("css selector", "[data-layout='stacked']").is_displayed()
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
