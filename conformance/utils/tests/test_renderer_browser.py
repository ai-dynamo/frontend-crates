# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise renderer conformance through the shared published browser UI."""
from __future__ import annotations

import shutil
import gzip
import hashlib
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

selenium = pytest.importorskip("selenium")
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait

from browser_helpers import raw_view_download, settle
from test_browser_viewer import _driver, _server
from verify_publication_transport import materialize_embedded, publication_metadata, runtime_catalog

pytestmark = pytest.mark.skipif(
    not any(shutil.which(binary) for binary in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"))
    and not Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome").is_file(),
    reason="no headless Chrome available",
)


@pytest.fixture
def renderer_driver():
    driver = _driver()
    driver.set_window_size(1600, 1200)
    try:
        yield driver
    finally:
        driver.quit()


def _open_renderer(driver, viewer_url: str) -> None:
    driver.get(viewer_url + "?tab=tab-renderer")
    settle(driver, "tab-renderer")


def _search(driver, tab: str, value: str) -> None:
    element = driver.find_element(By.CSS_SELECTOR, f"#{tab} [data-case-search]")
    driver.execute_script("arguments[0].value=arguments[1];arguments[0].dispatchEvent(new Event('input',{bubbles:true}));", element, value)
    WebDriverWait(driver, 10).until(lambda page: parse_qs(urlsplit(page.current_url).query).get("search_" + tab, [""])[0] == value)


def _switch(driver, tab: str) -> None:
    driver.find_element(By.ID, tab + "-button").click()
    settle(driver, tab)


def test_renderer_direct_url_has_five_models_two_captures_and_independent_results(renderer_driver, viewer_url):
    driver = renderer_driver
    _open_renderer(driver, viewer_url)
    state = driver.execute_script("""
      const panel=document.getElementById('tab-renderer');
      return {active:panel.classList.contains('active'), models:window.__conformanceModel.rows.map(row=>row.family),
        cards:panel.querySelectorAll('.capture-slot').length,
        selection:window.__conformanceController.selection(panel).selected,
        cells:panel.querySelectorAll('.matrix-cell[data-ttip-id]').length,
        labels:[...panel.querySelectorAll('.capture-slot-engine')].map(node=>node.textContent),
        error:window.__conformanceError||null};
    """)
    assert state["active"]
    assert set(state["models"]) == {"deepseek_v41", "glm53", "kimi_k3", "qwen38", "minimax_m31"}
    assert state["cards"] == 2
    assert state["selection"] == ["renderer_reference", "renderer_dynamo"]
    assert state["labels"] == ["Model reference", "Dynamo renderer"]
    assert state["cells"] > 100
    assert state["error"] is None


def test_tab_filters_and_reset_preserve_the_other_suite(renderer_driver, viewer_url):
    driver = renderer_driver
    _open_renderer(driver, viewer_url)
    _search(driver, "tab-renderer", "model:Kimi")
    _switch(driver, "tab-unified")
    _search(driver, "tab-unified", "model:Qwen")
    _switch(driver, "tab-renderer")
    assert driver.find_element(By.CSS_SELECTOR, "#tab-renderer [data-case-search]").get_attribute("value") == "model:Kimi"
    driver.refresh()
    settle(driver, "tab-renderer")
    assert driver.find_element(By.CSS_SELECTOR, "#tab-renderer [data-case-search]").get_attribute("value") == "model:Kimi"
    driver.find_element(By.CSS_SELECTOR, "[data-reset]").click()
    settle(driver, "tab-renderer")
    query = parse_qs(urlsplit(driver.current_url).query)
    assert "search_tab-renderer" not in query
    assert query["search_tab-unified"] == ["model:Qwen"]
    assert len(driver.execute_script("return window.__conformanceModel.rows")) == 5
    _switch(driver, "tab-unified")
    assert driver.find_element(By.CSS_SELECTOR, "#tab-unified [data-case-search]").get_attribute("value") == "model:Qwen"


def test_renderer_layout_theme_keyboard_popup_and_shared_url(renderer_driver, viewer_url):
    driver = renderer_driver
    _open_renderer(driver, viewer_url)
    driver.find_element(By.CSS_SELECTOR, "#tab-renderer [data-layout=stacked]").click()
    WebDriverWait(driver, 10).until(lambda page: page.find_element(By.CSS_SELECTOR, "#tab-renderer .comparison-ui.layout-stacked"))
    vertical = driver.find_element(By.CSS_SELECTOR, "[data-transpose-toggle]")
    if vertical.is_selected():
        vertical.click()
    WebDriverWait(driver, 10).until(lambda page: not page.execute_script("return document.body.classList.contains('transpose-mode')"))
    before = driver.execute_script("return document.documentElement.dataset.theme")
    driver.find_element(By.CSS_SELECTOR, "[data-theme-toggle]").click()
    assert driver.execute_script("return document.documentElement.dataset.theme") != before
    cell = driver.execute_script("""
      return [...document.querySelectorAll('#tab-renderer .matrix-cell[data-ttip-id]')]
        .find(cell=>window.__conformanceTooltipModel(cell)?.first_difference);
    """)
    assert cell is not None
    driver.execute_script("arguments[0].scrollIntoView({block:'center'});arguments[0].focus();", cell)
    cell.send_keys(Keys.ENTER)
    popup = WebDriverWait(driver, 10).until(lambda page: next((element for element in page.find_elements(By.CSS_SELECTOR, ".ttip-pinned") if element.is_displayed()), False))
    for label in ("Renderer request", "Rendering settings", "Exact prompt / rejection", "Producer provenance", "First different UTF-8 byte", "Metadata expectation authority"):
        assert label in popup.text
    assert not popup.find_elements(By.CSS_SELECTOR, ".ttip-chunks")
    assert "popup" in parse_qs(urlsplit(driver.current_url).query)
    driver.refresh()
    settle(driver, "tab-renderer")
    WebDriverWait(driver, 10).until(lambda page: any(element.is_displayed() for element in page.find_elements(By.CSS_SELECTOR, ".ttip-pinned .renderer-prompt")))
    assert driver.find_element(By.CSS_SELECTOR, "#tab-renderer .comparison-ui.layout-stacked")
    assert driver.execute_script("return !document.body.classList.contains('transpose-mode')")
    driver.find_element(By.TAG_NAME, "body").send_keys(Keys.ESCAPE)


def test_renderer_fixture_download_returns_original_authored_json(renderer_driver, viewer_url):
    driver = renderer_driver
    _open_renderer(driver, viewer_url)
    cell = driver.find_element(By.CSS_SELECTOR, "#tab-renderer .matrix-cell[data-ttip-id]")
    driver.execute_script("arguments[0].scrollIntoView({block:'center'});arguments[0].focus();", cell)
    cell.send_keys(Keys.ENTER)
    link = WebDriverWait(driver, 10).until(lambda page: next((element for element in page.find_elements(By.CSS_SELECTOR, ".ttip-pinned .ttip-fixture a") if element.is_displayed()), False))
    driver.get(link.get_attribute("href"))
    WebDriverWait(driver, 20).until(lambda page: page.find_elements(By.CSS_SELECTOR, "#viewer > pre"))
    downloaded = raw_view_download(driver)
    assert downloaded["filename"] == "inputs.json"
    assert downloaded["hash"] == downloaded["renderedHash"]
    assert downloaded["size"] > 0
    assert '"schema": 1' in driver.find_element(By.CSS_SELECTOR, "#viewer > pre").text


def test_renderer_standalone_file_uses_no_external_fetch(renderer_driver, published_viewer):
    driver = renderer_driver
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {"source": """
      window.__rendererOfflineFetches=[];
      window.fetch=async function(url){window.__rendererOfflineFetches.push(String(url));throw new Error('Offline renderer attempted external fetch');};
    """})
    driver.get((published_viewer / "index.html").resolve().as_uri() + "?tab=tab-renderer")
    settle(driver, "tab-renderer")
    assert driver.execute_script("return window.__rendererOfflineFetches") == []
    assert driver.execute_script("return window.__conformanceModel.rows.length") == 5


def test_history_catalog_upgrade_keeps_the_active_renderer_and_its_selection(renderer_driver, published_viewer, tmp_path):
    # A private transport projection exercises catalog discovery without fabricating captures.
    site = materialize_embedded(published_viewer, tmp_path / "history-discovery")
    catalog = runtime_catalog(site)
    previous_revision = catalog["revision"]
    catalog["revision"] = "b" * 64
    catalog["capture_generation"]["mode"] = "history"
    catalog.pop("runtime_sha256", None)
    raw = (json.dumps(catalog, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
    catalog_bytes = raw[:-2] + b',"runtime_sha256":"' + hashlib.sha256(raw).hexdigest().encode() + b'"}\n'
    (site / "conformance/viewer-resolved-catalog.json.gz").write_bytes(gzip.compress(catalog_bytes, mtime=0))
    metadata = publication_metadata(site)
    metadata.update(catalog_revision=previous_revision, resolved_catalog_revision=catalog["revision"], generation_mode="history")
    (site / "conformance/viewer-publication.json").write_text(json.dumps(metadata))
    server, _requests = _server(root=site)
    driver = renderer_driver
    try:
        _open_renderer(driver, f"http://127.0.0.1:{server.server_port}/index.html")
        settle(driver, "tab-renderer", history_ready=True)
        state = driver.execute_script("""
          return {active:window.__conformanceController.activePanel().id,
            selection:window.__conformanceController.selection(document.getElementById('tab-renderer')).selected,
            revision:document.getElementById('viewer').dataset.catalogRevision,
            history:document.getElementById('viewer').dataset.historyAvailable,
            model:window.__conformanceModel.id,error:window.__conformanceError||null};
        """)
        assert state == {"active": "tab-renderer", "selection": ["renderer_reference", "renderer_dynamo"],
                         "revision": catalog["revision"], "history": "true", "model": "tab-renderer", "error": None}
        _switch(driver, "tab-unified")
        _switch(driver, "tab-renderer")
        assert driver.execute_script("return window.__conformanceModel.rows.length") == 5
    finally:
        server.shutdown()
        server.server_close()
