# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qualify the native site's actual Chromium journey under the Pages prefix."""
import hashlib
import base64
import io
import os
import copy
import gzip
import json
import re
import shutil
import tarfile
import threading
from urllib.parse import parse_qs, urlencode, urlsplit

import pytest

selenium = pytest.importorskip("selenium")
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.common.keys import Keys
from browser_helpers import raw_view_download

from test_browser_viewer import ROOT, _driver, _ready, _server, _choose_engine
from unified_archives import read_source
from fixture_test_support import capture_releases, historical_source_bytes
from fixture_test_support import prepare_isolated_publication
from fixture_test_support import publish
from test_capture_sources import native_publisher, split_evidence, pin_history
import capture_sources


@pytest.mark.parametrize("mode", ["latest", "history"])
def test_isolated_modes_select_and_download_without_the_history_source(
        split_evidence, native_publisher, tmp_path, monkeypatch, source_catalog, mode):
    root, history, _capture, _path = split_evidence
    archive_probes = []
    manifest_path = history / "conformance/fixtures-manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    for relative, member_name in (
            ("toolcalling/isolated.tar.gz", "toolcalling/isolated/original.yaml"),
            ("source-evidence/isolated.tar.gz", "source-evidence/isolated/SOURCE.json")):
        archive_path = history / "conformance/fixtures" / relative
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        member_bytes = b'{"recorded": null, "producer": "isolated"}\n'
        with tarfile.open(archive_path, "w:gz") as output:
            member = tarfile.TarInfo(member_name)
            member.size = len(member_bytes)
            output.addfile(member, io.BytesIO(member_bytes))
        data = archive_path.read_bytes()
        manifest["shards"].append({"path": relative, "format": "tar.gz",
                                   "size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        archive_probes.extend([(archive_path.relative_to(history).as_posix(), data),
                               (member_name, member_bytes), ("conformance/" + member_name, member_bytes)])
    manifest_path.write_bytes(capture_sources.encoded(manifest))
    (root / capture_sources.PIN).write_bytes(capture_sources.encoded(pin_history(history)))
    metadata = copy.deepcopy(source_catalog["metadata"])
    metadata["tabs"] = [tab for tab in metadata["tabs"] if tab["suite"] == "unified"]
    metadata["source_hashes"] = {}
    metadata["unified_presentation"]["authored_scenarios"] = ["text_only"]
    metadata["unified_presentation"]["scenario_families"] = {"text_only": ["gemma4"]}
    view = tmp_path / "view"
    prepare_isolated_publication(root, history, view, monkeypatch, native_publisher,
                                 mode=mode, metadata=metadata)
    shutil.copyfile(ROOT / "index.html", view / "index.html")
    shutil.copytree(ROOT / "conformance/viewer", view / "conformance/viewer",
                    ignore=shutil.ignore_patterns("node_modules", "*.test.js"))
    shutil.copytree(ROOT / "conformance/utils/src/assets", view / "conformance/utils/src/assets")
    monkeypatch.setenv("CONFORMANCE_ESBUILD_ROOT", os.environ.get("CONFORMANCE_ESBUILD_ROOT", str(ROOT / "conformance/viewer")))
    site = tmp_path / "published"
    result = publish(view, site, check=False)
    assert result.returncode == 0, result.stderr
    catalog = json.loads((view / "conformance/viewer-catalog.json").read_bytes())
    version = "0.5.0" if mode == "history" else "0.5.2"
    resource = next(row for row in catalog["sources"]["unified"]
                    if row["kind"] == "history" and row["version"] == version)
    expected = read_source(view, resource)
    history.rename(tmp_path / "unavailable-history")
    server, requests = _server(root=site)
    driver = _driver()
    base = f"http://127.0.0.1:{server.server_port}/"
    try:
        driver.get(base)
        _ready(driver, "tab-unified")
        if mode == "history":
            WebDriverWait(driver, 15).until(lambda page: page.execute_script(
                "return document.getElementById('viewer').dataset.historyAvailable === 'true'"))
        assert not any("/viewer-data/" in path for path in requests)
        choices = [option.get_attribute("value") for option in driver.find_elements(
            "css selector", "[data-slot-capture='0'] option")]
        assert ("dynamo@0.5.0" in choices) == (mode == "history")
        if mode == "history":
            Select(driver.find_element("css selector", "[data-slot-capture='0']")).select_by_value("dynamo@0.5.0")
            _ready(driver, "tab-unified")
            _choose_engine(driver, 1, "dynamo")
            _ready(driver, "tab-unified")
            expected_selection = ["dynamo@0.5.0", "dynamo"]
            assert driver.execute_script("return document.getElementById('tab-unified')._comparisonSelection.keys") == expected_selection
            driver.get(driver.current_url)
            _ready(driver, "tab-unified")
            assert driver.execute_script("return document.getElementById('tab-unified')._comparisonSelection.keys") == expected_selection
            assert any("/viewer-data/" in path for path in requests)
        else:
            assert sum(path.endswith("viewer-publication.json") for path in requests) == 1
            assert not any(path.endswith("viewer-resolved-catalog.json.gz") for path in requests)
        driver.get(base + "?" + urlencode({"raw": resource["path"]}))
        WebDriverWait(driver, 15).until(lambda page: page.find_elements("css selector", "#viewer > a[download]"))
        downloaded = raw_view_download(driver)
        assert downloaded["hash"] == downloaded["renderedHash"] == hashlib.sha256(expected).hexdigest()
        if mode == "history":
            for path, data in archive_probes:
                driver.get(base + "?" + urlencode({"raw": path}))
                WebDriverWait(driver, 15).until(lambda page: page.find_elements("css selector", "#viewer > a[download]"))
                downloaded = raw_view_download(driver)
                assert downloaded["hash"] == hashlib.sha256(data).hexdigest()
                assert downloaded["size"] == len(data)
    finally:
        driver.quit()
        server.shutdown()
        server.server_close()


def test_native_publication_prefix_switch_raw_downloads_and_legacy_routes(historical_published_viewer, history_source, source_catalog):
    prefix = "/frontend-crates"
    server, requests = _server(prefix=prefix, root=historical_published_viewer)
    driver = _driver()
    base = f"http://127.0.0.1:{server.server_port}{prefix}/"
    catalog = json.loads(gzip.decompress((historical_published_viewer / "conformance/viewer-resolved-catalog.json.gz").read_bytes()))
    inventory = capture_sources.verify(history_source)
    assert catalog["transport"]["schema"] == 2
    try:
        driver.get(base + "?tab=tab-unified&keep=native-publication")
        _ready(driver, "tab-unified")
        assert not any("/indexes/raw-" in path for path in requests), "default startup must not request original lookup indexes"
        assert driver.execute_script("return window.__conformanceController.selection(document.getElementById('tab-unified')).selected") == ["dynamo"]
        driver.find_element("css selector", "[data-slot-open='0']").click()
        select = WebDriverWait(driver, 15).until(lambda page: page.find_element("css selector", "[data-slot-capture='0']"))
        choices = [option.get_attribute("value") for option in select.find_elements("tag name", "option")]
        historical = next(value for value in choices if value.startswith("dynamo@"))
        Select(select).select_by_value(historical)
        WebDriverWait(driver, 180).until(lambda page: page.execute_script("return window.__conformanceController.selection(document.getElementById('tab-unified')).selected.includes(arguments[0]) && window.__conformanceLoader.pending === 0", historical))
        _ready(driver, "tab-unified")
        restored = driver.current_url
        driver.get(restored)
        _ready(driver, "tab-unified")
        assert driver.execute_script("return window.__conformanceController.selection(document.getElementById('tab-unified')).selected") == [historical]
        assert parse_qs(urlsplit(driver.current_url).query)["keep"] == ["native-publication"]
        hidden = driver.execute_script("""
            return window.__CONFORMANCE_PAGE.tabs.find(tab => tab.id === 'tab-unified').candidates
                .find(item => item.key.startsWith('dynamo@') && !arguments[0].includes(item.key)
                    && !item.source_blocker && item.has_captured_results !== false
                    && (item.capture_families || item.native_unified_families || []).length).key;
        """, choices)
        driver.get(base + "?" + urlencode({"tab": "tab-unified", "slots_tab-unified": hidden}))
        _ready(driver, "tab-unified")
        normalized = driver.execute_script("return window.__conformanceController.selection(document.getElementById('tab-unified')).selected")
        assert normalized[0] in choices
        driver.refresh()
        _ready(driver, "tab-unified")
        assert driver.execute_script("return window.__conformanceController.selection(document.getElementById('tab-unified')).selected") == normalized
        # The picker keeps recent patches plus historical release lines; older data must
        # still load and paint through the same browser resource/model pipeline.
        driver.execute_async_script("""
            const done = arguments[arguments.length - 1];
            window.__conformanceLoader.select('tab-unified', {selected: [arguments[0]], reference: null}).then(done);
        """, hidden)
        _ready(driver, "tab-unified")
        assert driver.execute_script("return document.getElementById('tab-unified')._comparisonSelection.keys") == [hidden]
        originals = catalog["sources"]["original_unified"]
        authored = next(item for item in source_catalog["sources"]["unified"] if item["kind"] == "inputs-and-golden")
        resource = next(item for item in originals if item["kind"] == "history")
        probes = [(authored["path"], (ROOT / authored["path"]).read_bytes()),
                  (resource["path"], historical_source_bytes(history_source, resource["path"]))]
        # Sparse originals retain manifests and receipts as separate downloads.
        for kind in ("execution", "receipt"):
            resources = [item for item in inventory["files"]
                         if item["path"].startswith("conformance/history-datasets/")
                         and f"/{kind}s/" in item["path"]]
            for resource in resources:
                data = historical_source_bytes(history_source, resource["path"])
                assert hashlib.sha256(data).hexdigest() == resource["sha256"]
                probes.append((resource["path"], data))
        archive = next(row for row in inventory["files"]
                       if row["path"].startswith("conformance/fixtures/toolcalling/") and row["path"].endswith(".tar.gz"))
        member = next(item for item in capture_sources.archive_members(history_source / archive["path"]) if item["kind"] == "file")
        with tarfile.open(history_source / archive["path"]) as stream:
            member_bytes = stream.extractfile(member["path"]).read()
            probes.extend([(member["path"], member_bytes), ("conformance/" + member["path"], member_bytes)])
        probes.append((archive["path"], (history_source / archive["path"]).read_bytes()))
        for path, expected in probes:
            driver.get(base + "?" + urlencode({"raw": path}))
            WebDriverWait(driver, 180).until(lambda page: page.find_elements("css selector", "#viewer pre") or page.execute_script("return !!window.__conformanceError"))
            assert not driver.execute_script("return window.__conformanceError || null")
            rendered = driver.find_element("css selector", "#viewer pre").get_property("textContent")
            if path.endswith((".yaml", ".yml", ".json", ".txt")):
                assert rendered.encode() == expected
            else:
                assert rendered == f"Binary resource ({len(expected)} bytes). Use Download to save the original bytes."
            link = driver.find_element("link text", "Download original resource")
            assert link.get_attribute("href").startswith("blob:")
            downloaded = raw_view_download(driver)
            assert downloaded['hash'] == hashlib.sha256(expected).hexdigest()
            assert downloaded['size'] == len(expected)
        driver.get(base + "conformance/CONFORMANCE_v2.html?tab=tab-unified&keep=native#case")
        _ready(driver, "tab-unified")
        assert urlsplit(driver.current_url).path == prefix + "/conformance/CONFORMANCE_v2.html"
        assert urlsplit(driver.current_url).fragment == "case"
        assert parse_qs(urlsplit(driver.current_url).query)["keep"] == ["native"]
        _ready(driver, "tab-unified")
        assert not any(path.endswith((".js", ".css")) for path in requests), 'report assets must be inline'
        assert not [path for path in requests if path.endswith((".yaml", ".yml", ".tar.gz"))]
        assert all(path.startswith(prefix + "/") or path == "/favicon.ico" for path in requests)
    finally:
        driver.quit()
        server.shutdown()
        server.server_close()


def test_checkout_raw_route_retains_original_source_fallback(current_source_view):
    server, requests = _server(prefix="/frontend-crates", root=current_source_view)
    driver = _driver()
    catalog = json.loads((current_source_view / "conformance/viewer-catalog.json").read_bytes())
    resource = next(item for item in catalog["sources"]["unified"] if item["kind"] == "history")
    try:
        driver.get(f"http://127.0.0.1:{server.server_port}/frontend-crates/?" + urlencode({"raw": resource["path"]}))
        WebDriverWait(driver, 180).until(lambda page: page.find_elements("css selector", "#viewer pre") or page.execute_script("return !!window.__conformanceError"))
        assert not driver.execute_script("return window.__conformanceError || null")
        assert driver.find_element("css selector", "#viewer pre").get_property("textContent").encode() == read_source(current_source_view, resource)
        assert "/frontend-crates/" + resource["path"] in requests
    finally:
        driver.quit()
        server.shutdown()
        server.server_close()


def test_startup_url_aliases_use_the_packed_loader(published_viewer):
    server, requests = _server(prefix="/frontend-crates", root=published_viewer)
    driver = _driver()
    try:
        baseline = None
        for alias in ("external", "bootstrap", "inline"):
            requests.clear()
            driver.get(f"http://127.0.0.1:{server.server_port}/frontend-crates/?tab=tab-unified&startup={alias}")
            _ready(driver, "tab-unified")
            current = driver.execute_script("return {selection: window.__conformanceModel.candidates.map(item => item.key), rows: window.__conformanceModel.rows.map(row => row.family || row.section)}")
            if baseline is None:
                baseline = current
            assert current == baseline
            assert not any('viewer-resolved-catalog' in path or '/viewer-data/' in path for path in requests)
            assert not any(path.endswith((".yaml", ".yml", ".tar.gz", "viewer-startup.json")) or "/bootstrap/" in path for path in requests)
    finally:
        driver.quit()
        server.shutdown()
        server.server_close()


@pytest.fixture
def previous_dynamo_capture(historical_published_viewer):
    version = capture_releases(historical_published_viewer)['dynamo_v2'][-2]
    return {'version': version, 'key': f'dynamo@{version}', 'identity': f'dynamo_v2@{version}'}


def test_recent_captures_are_embedded_and_older_captures_still_fetch(historical_published_viewer, previous_dynamo_capture):
    html = (historical_published_viewer / 'index.html').read_text()
    payload = json.loads(re.search(r'id="conformance-embedded">(.*?)</script>', html, re.S).group(1))
    embedded_catalog = json.loads(gzip.decompress(base64.b64decode(payload['catalog']['bytes'])))
    versions = capture_releases(historical_published_viewer)
    current = sorted({(capture['implementation'], capture['version'])
                      for capture in embedded_catalog['capture_generation']['current']})
    assert payload['captures'] == [list(identity) for identity in current]
    latest = sorted([implementation, releases[-1]] for implementation, releases in versions.items())
    server, requests = _server(prefix='/frontend-crates', root=historical_published_viewer)
    driver = _driver()
    base = f'http://127.0.0.1:{server.server_port}/frontend-crates/'
    try:
        for implementation, version in latest:
            requests.clear()
            identity = f'{implementation}@{version}'
            driver.get(base + '?' + urlencode({'tab': 'tab-unified', 'slots_tab-unified': identity + ',,'}))
            _ready(driver, 'tab-unified')
            selected = driver.execute_script("""
                const panel = document.getElementById('tab-unified');
                const keys = window.__conformanceController.selection(panel).selected;
                return window.__CONFORMANCE_PAGE.tabs.find(tab => tab.id === panel.id).candidates
                  .filter(item => keys.includes(item.key)).map(item => item.identity);
            """)
            assert selected == [identity], (identity, selected)
            assert not any('/viewer-data/' in path for path in requests), requests
            assert sum(path.endswith('viewer-resolved-catalog.json.gz') for path in requests) == 1
            assert not driver.execute_script('return window.__conformanceError || null')
        requests.clear()
        driver.get(base + '?' + urlencode({'tab': 'tab-unified', 'slots_tab-unified': 'dynamo@0.6.1,vllm_python@0.31.0,'}))
        _ready(driver, 'tab-unified')
        assert driver.execute_script("return window.__conformanceController.selection(document.getElementById('tab-unified')).selected") == ['dynamo@0.6.1', 'vllm_python@0.31.0']
        assert any('/viewer-data/' in path for path in requests), 'older capture must fetch JSON transport'
        for version, needs_history in [('0.26.0', True), ('0.27.1', False)]:
            requests.clear()
            identity = f'vllm_rust@{version}'
            driver.get(base + '?' + urlencode({'slots_tab-unified': identity + ',,'}))
            _ready(driver, 'tab-unified')
            captured = driver.execute_script("""
                const row = window.__conformanceModel.rows.find(row => row.family === 'nemotron_v3');
                const leaves = cell => cell.variants ? cell.variants.flatMap(leaves) : [cell];
                return Object.values(row.cells).flatMap(leaves).filter(cell => !cell.tooltip?.na_note).map(cell =>
                    cell.tooltip?.candidates.find(candidate => candidate.key === arguments[0])?.block)
                    .filter(block => block && block.unavailable_code !== 'missing_capture').length;
            """, identity)
            assert captured > 0, 'a partial current-version bundle must not hide the historical Nemotron capture'
            assert any('/viewer-data/' in path for path in requests) is needs_history, requests
        # The second recent patch is selectable through the actual control too.
        driver.get(base)
        _ready(driver, 'tab-unified')
        requests.clear()
        driver.find_element('css selector', "[data-slot-open='0']").click()
        Select(driver.find_element('css selector', "[data-slot-capture='0']")).select_by_value(previous_dynamo_capture['key'])
        WebDriverWait(driver, 90).until(lambda page: page.execute_script("return window.__conformanceLoader.pending === 0 && document.getElementById('tab-unified')._comparisonSelection.keys[0] === arguments[0]", previous_dynamo_capture['key']))
        assert any('/viewer-data/' in path for path in requests), 'previous release must fetch JSON transport'
    finally:
        driver.quit()
        server.shutdown()
        server.server_close()


def test_corrupt_embedded_catalog_fails_without_network_fallback(published_viewer):
    def corrupt(html):
        return re.sub(r'(<script type="application/json" id="conformance-embedded">)(.*?)</script>',
                      lambda match: match[1] + json.dumps({**json.loads(match[2]), 'schema': 0}) + '</script>', html, flags=re.S)

    server, requests = _server(root=published_viewer, html_transform=corrupt)
    driver = _driver()
    try:
        driver.get(f'http://127.0.0.1:{server.server_port}/')
        WebDriverWait(driver, 30).until(lambda page: page.find_elements('css selector', '#viewer .error'))
        assert 'Unsupported embedded data' in driver.find_element('css selector', '#viewer .error').text
        assert not any('/viewer-data/' in path or 'viewer-resolved-catalog' in path for path in requests)
        assert not driver.find_elements('css selector', '#tab-unified td.cell')
    finally:
        driver.quit()
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize('change_selection', [False, True], ids=['default', 'changed-selection'])
def test_history_discovery_renders_first_and_preserves_selection(historical_published_viewer, change_selection, previous_dynamo_capture):
    gate = (threading.Event(), threading.Event())
    server, requests = _server(prefix='/frontend-crates', root=historical_published_viewer, gate=gate, gate_path='/conformance/viewer-publication.json')
    driver = _driver()
    try:
        driver.get(f'http://127.0.0.1:{server.server_port}/frontend-crates/')
        assert gate[0].wait(15), 'history discovery did not start'
        WebDriverWait(driver, 30).until(lambda page: page.execute_script("return Boolean(document.querySelector('#tab-unified[data-load-state=ready] td.cell'))"))
        assert driver.execute_script("return window.__conformanceController.selection(document.getElementById('tab-unified')).selected") == ['dynamo']
        assert not driver.find_elements('css selector', f"option[value='{previous_dynamo_capture['key']}']")
        assert not any('/viewer-data/' in path for path in requests)
        if change_selection:
            driver.find_element('css selector', "[data-slot-open='0']").click()
            driver.find_element('css selector', "[data-picker-engine-choice='vllm']").click()
            driver.find_element('css selector', "[data-slot-capture='0']").send_keys(Keys.ESCAPE)
            WebDriverWait(driver, 30).until(lambda page: page.execute_script("return Boolean(document.querySelector('#tab-unified[data-load-state=ready] td.cell'))"))
        expected = driver.execute_script("return window.__conformanceController.selection(document.getElementById('tab-unified')).selected")
        cell = driver.find_element('css selector', '#tab-unified td.cell[data-ttip-click=pin]')
        cell.click()
        WebDriverWait(driver, 10).until(lambda page: page.find_elements('css selector', '.ttip-pinned .ttip-head'))
        generation = driver.execute_script('return window.__conformanceLoader.generation')
        gate[1].set()
        _ready(driver, 'tab-unified')
        assert driver.execute_script("return document.getElementById('viewer').dataset.historyAvailable") == 'true'
        assert driver.execute_script("return document.getElementById('tab-unified')._comparisonSelection.keys") == expected
        assert driver.execute_script('return window.__conformanceLoader.generation') > generation
        assert not any('/viewer-data/' in path for path in requests), 'current selections must still use embedded resources after catalog replacement'
        assert driver.execute_script("return window.__CONFORMANCE_PAGE.tabs.find(tab=>tab.id==='tab-unified').candidates.some(candidate=>candidate.key===arguments[0])", previous_dynamo_capture['key'])
        for transposed in (False, True):
            toggle = driver.find_element('css selector', '[data-transpose-toggle]')
            if toggle.is_selected() != transposed:
                toggle.send_keys(Keys.SPACE)
            assert driver.find_elements('css selector', '.ttip-pinned .ttip-head')
            missing = driver.execute_script("""return [...document.querySelectorAll('#tab-unified [data-ttip-id]')]
                .filter(cell => !window.__conformanceTooltipModel(cell)
                    || (cell.matches('td.cell') && !window.__conformanceCellModel(cell)))
                .map(cell => cell.getAttribute('data-ttip-id'));""")
            assert missing == [], 'replacement and transposed tooltips must retain their registered models'
    finally:
        gate[1].set()
        driver.quit()
        server.shutdown()
        server.server_close()


def test_same_model_replacement_keeps_tooltips_and_retires_old_only_models(published_viewer):
    driver = _driver()
    try:
        driver.get((published_viewer / 'index.html').as_uri())
        _ready(driver, 'tab-unified')
        driver.execute_script("""
            const panel = document.getElementById('tab-unified');
            window.previousTooltipCell = panel.querySelector('td.cell[data-ttip-id]');
            for (let index = 0; index < 2; index++) {
                const commit = window.ConformanceView.prepareUpdate(panel, window.__conformanceModel, window.__CONFORMANCE_PAGE);
                const pin = window.__conformanceController.disposePanel(panel);
                commit();
                window.__conformanceController.updatePanel(panel, pin);
            }
        """)
        cell = driver.find_element('css selector', '#tab-unified td.cell[data-ttip-click=pin]')
        cell.click()
        WebDriverWait(driver, 10).until(lambda page: page.find_elements('css selector', '.ttip-pinned .ttip-head'))
        assert driver.execute_script("return Boolean(window.__conformanceCellModel(window.previousTooltipCell))")
        driver.find_element('css selector', "[data-slot-open='1']").send_keys(Keys.ENTER)
        driver.find_element('css selector', "[data-picker-engine-choice='vllm']").click()
        driver.find_element('css selector', "[data-slot-capture='1']").send_keys(Keys.ESCAPE)
        _ready(driver, 'tab-unified')
        assert driver.find_elements('css selector', '.ttip-pinned .ttip-head')
        assert driver.execute_script("return window.__conformanceCellModel(window.previousTooltipCell) === undefined")
        assert driver.execute_script("return window.__conformanceTooltipModel(window.previousTooltipCell) === undefined")
    finally:
        driver.quit()
