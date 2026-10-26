# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real Chromium qualification for standalone latest and bundled-history reports."""
import gzip
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
from urllib.parse import urlencode, urlsplit

import pytest
from selenium.webdriver.support.ui import Select, WebDriverWait

from browser_helpers import raw_view_download
from verify_publication_transport import runtime_catalog
from fixture_test_support import capture_releases, historical_source_bytes
from test_browser_viewer import _driver, _ready, _server, _choose_engine
import capture_sources
import unified_history


@pytest.mark.parametrize("protocol", ["file", "http"])
def test_latest_html_alone_compares_and_downloads_without_external_data(published_viewer, tmp_path, protocol):
    report = tmp_path / "CONFORMANCE_v2.html"
    assert sorted(path.relative_to(published_viewer).as_posix() for path in published_viewer.rglob("*") if path.is_file()) == ["index.html"]
    shutil.copyfile(published_viewer / "index.html", report)
    assert list(tmp_path.iterdir()) == [report]
    catalog = runtime_catalog(published_viewer)
    assert catalog["capture_generation"]["mode"] == "latest"
    versions = capture_releases(published_viewer)
    payload = json.loads(re.search(r'id="conformance-embedded">(.*?)</script>', report.read_text(), re.S).group(1))
    assert payload['captures'] == sorted([implementation, version]
                                         for implementation, releases in versions.items() for version in releases)
    server = None
    driver = _driver()
    try:
        driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {"source": """
            window.__externalRequests = [];
            const original = window.fetch;
            window.fetch = (...args) => { if (!String(args[0]).startsWith('blob:')) window.__externalRequests.push(String(args[0])); return original(...args); };
        """})
        if protocol == "file":
            driver.execute_cdp_cmd("Network.enable", {})
            driver.execute_cdp_cmd("Network.emulateNetworkConditions", {"offline": True, "latency": 0, "downloadThroughput": -1, "uploadThroughput": -1})
            url = report.as_uri()
        else:
            server, requests = _server(prefix='/frontend-crates', root=tmp_path)
            url = f"http://127.0.0.1:{server.server_port}/frontend-crates/CONFORMANCE_v2.html"
        driver.get(url)
        _ready(driver, "tab-unified")
        expected_external = [] if protocol == 'file' else [driver.execute_script(
            "return new URL('conformance/viewer-publication.json', document.baseURI).href")]
        descriptors = driver.execute_script("return window.__CONFORMANCE_PAGE.tabs.find(t=>t.id==='tab-unified').candidates.filter(c=>c.identity && c.key!=='golden')")
        assert len(descriptors) == len(versions)
        visible_captures = set()
        for engine in ("vllm", "sglang", "dynamo"):
            _choose_engine(driver, 0, engine)
            _ready(driver, "tab-unified")
            available = driver.execute_script("return [...document.querySelectorAll('[data-slot-capture=\"0\"] option')].filter(o=>!o.disabled).map(o=>o.value)")
            visible_captures.update(available)
            for descriptor in (item for item in descriptors if item['key'] in available):
                implementation, version = descriptor['identity'].split('@')
                assert version == versions[implementation][-1]
                Select(driver.find_element('css selector', "[data-slot-capture='0']")).select_by_value(descriptor['key'])
                _ready(driver, 'tab-unified')
                assert driver.execute_script("return document.getElementById('tab-unified')._comparisonSelection.keys") == [descriptor['key']]
        policy = catalog["metadata"]["selection_policy"]["unified"]
        candidates = [row for row in catalog["sources"]["candidates"] if row["mode"] == "unified"]
        for candidate in candidates:
            newest = max((row["version"] for row in candidates if row["implementation"] == candidate["implementation"]),
                         key=unified_history._capture_release_sort_key)
            identity = f"{candidate['implementation']}@{candidate['version']}"
            if candidate["implementation"] == policy["reference"]:
                key = "dynamo" if candidate["version"] == newest else f"dynamo@{candidate['version']}"
            else:
                key = policy["historical_control_aliases"].get(identity, identity)
            assert (key in visible_captures) == (candidate["version"] == newest), key
        cell = driver.find_element("css selector", "#tab-unified td.cell[data-ttip-id]")
        cell.click()
        popup = WebDriverWait(driver, 15).until(lambda page: next((item for item in page.find_elements("css selector", ".ttip-pinned") if item.is_displayed()), None))
        assert "Golden output" in popup.text
        popup.find_element("css selector", ".ttip-close").click()
        assert driver.execute_script("return window.__externalRequests") == expected_external
        source = next(row for row in catalog["capture_generation"]["current"] if row["surface"] == "unified")
        driver.get(url + "?" + urlencode({"raw": source["source"]}))
        WebDriverWait(driver, 30).until(lambda page: page.find_elements("css selector", "#viewer > a[download]"))
        result = raw_view_download(driver)
        assert result["hash"] == source["sha256"]
        assert result["renderedHash"] == source["sha256"]
        assert driver.execute_script("return window.__externalRequests") == expected_external
        driver.get(url + '?' + urlencode({'startup': 'external', 'raw': 'missing.yaml',
                                           'slots_tab-unified': 'dynamo_v2@0.6.1,none,none'}))
        WebDriverWait(driver, 15).until(lambda page: page.find_elements('css selector', '#viewer .error'))
        assert 'missing.yaml' in driver.find_element('css selector', '#viewer .error').text
        assert driver.execute_script('return window.__externalRequests') == expected_external
        driver.get(url + '?' + urlencode({'startup': 'external', 'slots_tab-unified': 'dynamo_v2@0.6.1,none,none'}))
        _ready(driver, 'tab-unified')
        assert driver.execute_script("return document.getElementById('tab-unified')._comparisonSelection.keys") == ['dynamo']
        assert driver.execute_script("return document.getElementById('viewer').dataset.historyAvailable") == 'false'
        assert not driver.find_elements('css selector', "option[value='dynamo@0.6.1']")
        assert driver.execute_script("return document.querySelectorAll('script[src],link[rel=stylesheet]').length") == 0
        assert driver.execute_script("return [...document.querySelectorAll('#tab-unified td.cell')].every(cell=>!cell.dataset.fixture)")
        assert not driver.execute_script('return window.__conformanceError || null')
        assert driver.execute_script('return window.__externalRequests') == expected_external
        if server:
            assert requests.count(urlsplit(expected_external[0]).path) == 4
            assert not any("/viewer-data/" in path or path.endswith((".js", ".css")) for path in requests)
    finally:
        driver.quit()
        if server:
            server.shutdown()
            server.server_close()


def test_full_history_selection_urls_and_original_downloads(history_source, historical_published_viewer):
    site = historical_published_viewer
    catalog = json.loads(gzip.decompress((site / "conformance/viewer-resolved-catalog.json.gz").read_bytes()))
    assert catalog["capture_generation"]["mode"] == "history"
    server, requests = _server(root=site)
    driver = _driver()
    url = f"http://127.0.0.1:{server.server_port}/conformance/CONFORMANCE_v2.html"
    try:
        driver.get(url)
        _ready(driver, "tab-unified")
        WebDriverWait(driver, 30).until(lambda page: page.find_element("css selector", "#viewer").get_attribute("data-history-available") != "checking")
        assert driver.find_element("css selector", "#viewer").get_attribute("data-history-available") == "true"
        assert not any("/viewer-data/" in path for path in requests)
        select = driver.find_element("css selector", "[data-slot-capture='0']")
        options = [option.get_attribute("value") for option in select.find_elements("tag name", "option")
                   if option.is_enabled()]
        historical_v2 = next(value for value in options if value.startswith("dynamo@"))
        historical_batch = [value for value in options if value.startswith("dynamo_v1@")]
        assert historical_batch
        # The selector retains two recent v2 releases per minor line, plus
        # historical batch majors. Exercise both retained capture variants.
        historical = [historical_batch[-1], historical_v2]
        assert historical[0] != historical[1]
        Select(select).select_by_value(historical[0])
        _ready(driver, "tab-unified")
        assert any("/viewer-data/" in path for path in requests)
        _choose_engine(driver, 1, "dynamo")
        _ready(driver, "tab-unified")
        Select(driver.find_element("css selector", "[data-slot-capture='1']")).select_by_value(historical[1])
        _ready(driver, "tab-unified")
        assert driver.execute_script("return document.getElementById('tab-unified')._comparisonSelection.keys") == historical
        assert driver.execute_script("return [...document.querySelectorAll('#tab-unified .matrix-segments')].some(box => box.children.length === 2)")
        selection_url = driver.current_url
        driver.get(selection_url)
        _ready(driver, "tab-unified")
        assert driver.execute_script("return document.getElementById('tab-unified')._comparisonSelection.keys") == historical
        assert driver.execute_script("return [...document.querySelectorAll('#tab-unified .matrix-segments')].some(box => box.children.length === 2)")
        inventory = capture_sources.verify(history_source, capture_sources.history_pin(Path(__file__).resolve().parents[3], history_source))
        legacy = next(row for row in inventory["files"] if row["path"].startswith("conformance/fixtures/") and row["path"].endswith(".tar.gz"))
        current = next(row for row in catalog["capture_generation"]["current"]
                       if row["surface"] == "unified" and row["family"] == "qwen3" and row["implementation"] == "dynamo_v2")
        older = min((row for row in catalog["sources"]["original_unified"]
                     if row["family"] == "qwen3" and row["implementation"] == "dynamo_v2"),
                    key=lambda row: unified_history._capture_release_sort_key(row["version"]))
        assert unified_history._capture_release_sort_key(older["version"]) < unified_history._capture_release_sort_key(current["version"])
        original = historical_source_bytes(history_source, older["path"])
        lineage = next(row for row in catalog['sources']['original_unified']
                       if row['family'] == 'nemotron_v3' and row['implementation'] == 'vllm_rust'
                       and row['version'] == '0.26.0')
        lineage_original = historical_source_bytes(history_source, lineage['path'])
        checkout = Path(__file__).resolve().parents[3]
        current_original = (checkout / current["yaml_path"]).read_bytes()
        selection = json.loads((checkout / current["yaml_path"]).read_bytes().split(b"\n", 1)[0].removeprefix(b"capture_selection: "))
        latest_payload = (checkout / current["yaml_path"]).read_bytes().split(b"\n", 1)[1]
        assert hashlib.sha256(latest_payload).hexdigest() == selection["payload_sha256"]
        for path, expected, size in ((legacy["path"], legacy["sha256"], legacy["size"]),
                                     (older["path"], hashlib.sha256(original).hexdigest(), len(original)),
                                     (lineage['path'], hashlib.sha256(lineage_original).hexdigest(), len(lineage_original)),
                                     (current["source"], hashlib.sha256(current_original).hexdigest(), len(current_original))):
            before = len(requests)
            driver.get(url + "?" + urlencode({"raw": path}))
            WebDriverWait(driver, 45).until(lambda page: page.find_elements("css selector", "#viewer > a[download]"))
            if path == current["source"]:
                assert not any('/viewer-data/' in request for request in requests[before:]), 'current originals must use embedded resources after history discovery'
            download = raw_view_download(driver)
            assert download["hash"] == expected
            assert download["size"] == size
            if path.endswith(".yaml"):
                assert download["renderedHash"] == expected
    finally:
        driver.quit()
        server.shutdown()
        server.server_close()
