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
import ssl
import sys
from contextlib import ExitStack
from http.client import IncompleteRead
from pathlib import Path
from types import SimpleNamespace
from urllib.error import URLError
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlencode, urlsplit

import pytest

selenium = pytest.importorskip("selenium")
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.common.keys import Keys
from browser_helpers import raw_view_download, settle

from test_browser_viewer import ROOT, _driver, _ready, _server, _choose_engine, _assert_generation_provenance
from verify_publication_transport import publication_metadata, runtime_catalog, update_delivery_report, verify_http_publication, verify_http_publication_identity
from unified_archives import read_source
from fixture_test_support import (assert_viewer_data_matches_history, capture_releases,
                                  historical_source_bytes, identity_for_key)
from fixture_test_support import prepare_isolated_publication
from fixture_test_support import publish
from test_capture_sources import native_publisher, split_evidence, pin_history
import capture_sources
import unified_history
import verify_publication_transport as transport
from capture_versions import capture_release_sort_key

# How long the delivery check may wait for a freshly deployed GitHub Pages site to
# serve the new revision (main runs f9ea1b2e, 6252b6f4 and 7a2e8f96 saw HTTP 503
# right after `deploy-pages` succeeded; the next run was green). Handed to the
# transport through CONFORMANCE_PAGES_PROPAGATION_SECONDS and capped by the
# delivery budget (CONFORMANCE_DELIVERY_DEADLINE).
PAGES_PROPAGATION_SECONDS = 300.0


@pytest.mark.parametrize("family", ["hermes", "llama3_json"])
@pytest.mark.parametrize("implementation", ["dynamo_v1", "dynamo_v1_jail"])
def test_legacy_family_history_selects_both_recent_v1_releases(historical_published_viewer, family, implementation):
    versions = sorted(capture_releases(historical_published_viewer, family=family)[implementation], key=capture_release_sort_key)[-2:]
    assert versions == ["11.0.6", "11.0.7"]
    keys = [f"{implementation}@{version}" for version in versions]
    query = urlencode({"tab": "tab-unified", "slots_tab-unified": ",".join([*keys, "none"]),
                       "search_tab-unified": f"model:{family}"})
    server, _ = _server(root=historical_published_viewer)
    driver = _driver()
    try:
        driver.get(f"http://127.0.0.1:{server.server_port}/index.html?{query}")
        _ready(driver, "tab-unified")
        for _ in range(2):
            observed = driver.execute_script("""
                const comparison = window.__conformanceComparison;
                return {families: comparison.view.rows.map(row => row.family), counts: comparison.counts};
            """)
            assert observed["families"] == [family]
            for key in keys:
                assert observed["counts"][key]["total"] > 0
                assert observed["counts"][key]["missing"] == 0
            assert [Select(driver.find_element("css selector", f'[data-slot-capture="{slot}"]')).first_selected_option.get_attribute("value")
                    for slot in range(2)] == keys
            driver.refresh()
            _ready(driver, "tab-unified")
    finally:
        driver.quit()
        server.shutdown()
        server.server_close()


def test_publication_delivery_matches_qualified_artifact(historical_published_viewer, request, monkeypatch, tmp_path):
    monkeypatch.setenv("CONFORMANCE_DELIVERY_DEADLINE", str(transport.time.monotonic() + transport.HTTP_BUDGET))
    report_path = tmp_path / "delivery.json"
    if not os.environ.get("CONFORMANCE_DELIVERY_REPORT"):
        monkeypatch.setenv("CONFORMANCE_DELIVERY_REPORT", str(report_path))
    report_path = Path(os.environ["CONFORMANCE_DELIVERY_REPORT"])
    site = historical_published_viewer
    configured = request.config.getoption("--publication-url")
    cleanup = ExitStack()
    if configured:
        base = configured.rstrip("/")
        monkeypatch.setenv(transport.PROPAGATION_ENV, str(PAGES_PROPAGATION_SECONDS))
    else:
        server, _requests = _server(prefix="/frontend-crates", root=site)
        cleanup.callback(server.server_close)
        cleanup.callback(server.shutdown)
        base = f"http://127.0.0.1:{server.server_port}/frontend-crates"
    journey_passed = False
    try:
        # Compare the qualified bytes, including all history sidecars, rather than
        # trusting a successful HTTP response or a revision label on an older site.
        # A deployed site may still answer 503/404 while Pages propagates, so the
        # public checks wait bounded; the local server must answer at once.
        # The transport reads the propagation wait from the environment, so the
        # check calls keep their signature; the local server must answer at once.
        verify_http_publication_identity(site, base)
        verify_http_publication(site, base)
        verify_http_publication_identity(site, base)
        driver = _driver()
        cleanup.callback(driver.quit)
        driver.get(f"{base}/index.html?tab=tab-unified")
        settle(driver, "tab-unified", history_ready=True)
        assert driver.execute_script("return document.getElementById('viewer').dataset.historyAvailable") == "true"
        _assert_generation_provenance(driver, publication_metadata(site))
        # Use an offered comparison, never a pinned model or release. A catalog
        # with only one selectable capture still has a valid initial matrix.
        comparison = driver.execute_script("""
          const tab = window.__CONFORMANCE_PAGE.tabs.find(tab => tab.id === 'tab-unified');
          const select = document.querySelector('[data-slot-capture="0"]');
          const option = [...select.options].find(option => !option.disabled && option.value
            && option.value !== 'none' && option.value !== select.value
            && tab.candidates.some(candidate => candidate.identity
              && (candidate.identity === option.value || candidate.key === option.value)));
          const candidate = option && tab.candidates.find(candidate => candidate.identity === option.value || candidate.key === option.value);
          return candidate ? {identity: candidate.identity, key: candidate.key} : null;
        """)
        catalog = runtime_catalog(site)
        producer = catalog["metadata"]["selection_policy"]["unified"]["reference"]
        candidates = [source for source in catalog["sources"]["candidates"] if source["implementation"] == producer]
        current = max(candidates, key=lambda source: capture_release_sort_key(source["version"]))
        native = set(current["native_unified_families"])
        eligible = {source["version"] for source in candidates if not source.get("source_blocker")
                    and source.get("has_captured_results") is not False
                    and set(source["capture_families"]) & native}
        if len(eligible) > 1:
            assert comparison is not None, "qualified history offers older captures but the browser does not"
        if comparison:
            driver.get(f"{base}/index.html?" + urlencode({"tab": "tab-unified", "slots_tab-unified": f"{comparison['identity']},none,none"}))
            for reload in (False, True):
                if reload:
                    driver.refresh()
                settle(driver, "tab-unified", history_ready=True)
                assert driver.execute_script("return document.getElementById('viewer').dataset.historyAvailable") == "true"
                assert driver.execute_script("return window.__conformanceController.selection(document.getElementById('tab-unified')).selected") == [comparison["key"]]
                assert parse_qs(urlsplit(driver.current_url).query)["slots_tab-unified"][0].split(",")[0] == comparison["identity"]
        verify_http_publication_identity(site, base)
        journey_passed = True
    finally:
        try:
            update_delivery_report(report_path, browser_journey={"success": journey_passed, "publication": publication_metadata(site)})
        finally:
            cleanup.close()


@pytest.mark.parametrize("fault", [None, "stale", "missing"])
def test_http_delivery_checks_qualified_bytes_and_rejects_bad_sidecars(tmp_path, fault):
    (tmp_path / "index.html").write_bytes(b"qualified index")
    resource = tmp_path / "history with spaces.json.gz"
    resource.write_bytes(gzip.compress(b'{"history":true}'))
    server, _requests = _server(root=tmp_path,
        fault={"path": "history%20with%20spaces.json.gz", "kind": "corrupt", "used": False} if fault == "stale" else None,
        missing_paths=("/history%20with%20spaces.json.gz",) if fault == "missing" else ())
    try:
        if fault == "stale":
            with pytest.raises(ValueError, match="differs from qualified artifact"):
                verify_http_publication(tmp_path, f"http://127.0.0.1:{server.server_port}")
        elif fault == "missing":
            with pytest.raises(HTTPError):
                verify_http_publication(tmp_path, f"http://127.0.0.1:{server.server_port}")
        else:
            assert verify_http_publication(tmp_path, f"http://127.0.0.1:{server.server_port}")["resources"] == 2
    finally:
        server.shutdown()
        server.server_close()


def test_http_delivery_recovers_only_failed_get(tmp_path):
    (tmp_path / "index.html").write_bytes(b"qualified index")
    (tmp_path / "history.json").write_bytes(b"qualified history")
    server, requests = _server(root=tmp_path, fault={"path": "history.json", "responses": [503]})
    try:
        assert verify_http_publication(tmp_path, f"http://127.0.0.1:{server.server_port}") == {"resources": 2, "bytes": 32}
        assert requests.count("/history.json") == 2
        assert requests.count("/index.html") == 1
    finally:
        server.shutdown()
        server.server_close()


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
    catalog = runtime_catalog(historical_published_viewer)
    inventory = capture_sources.verify(history_source)
    assert catalog["transport"]["schema"] == 2
    try:
        driver.get(base + "?tab=tab-unified&keep=native-publication")
        _ready(driver, "tab-unified")
        assert not any("/indexes/raw-" in path for path in requests), "default startup must not request original lookup indexes"
        assert driver.execute_script("return window.__conformanceController.selection(document.getElementById('tab-unified')).selected") == ["dynamo"]
        assert_viewer_data_matches_history(
            historical_published_viewer, requests,
            [identity_for_key(catalog, "dynamo")])
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
        assert normalized == [hidden]
        assert driver.find_elements("css selector", f"option[value='{hidden}']")
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
        resource = min((item for item in originals if item["kind"] == "history"
                        and item["family"] == "qwen3" and item["implementation"] == "dynamo_v2"),
                       key=lambda item: unified_history._capture_release_sort_key(item["version"]))
        current_capture = next(item for item in catalog["capture_generation"]["current"]
                               if item["family"] == "qwen3" and item["implementation"] == "dynamo_v2")
        probes = [(authored["path"], (ROOT / authored["path"]).read_bytes()),
                  (current_capture["source"], (ROOT / current_capture["yaml_path"]).read_bytes()),
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
                assert hashlib.sha256(rendered.encode()).hexdigest() == hashlib.sha256(expected).hexdigest(), path
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
    catalog = runtime_catalog(historical_published_viewer)
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
            assert_viewer_data_matches_history(historical_published_viewer, requests, [identity])
            assert sum(path.endswith('viewer-resolved-catalog.json.gz') for path in requests) == 1
            assert not driver.execute_script('return window.__conformanceError || null')
        requests.clear()
        driver.get(base + '?' + urlencode({'tab': 'tab-unified', 'slots_tab-unified': 'dynamo@0.6.1,vllm_python@0.31.0,'}))
        _ready(driver, 'tab-unified')
        assert driver.execute_script("return window.__conformanceController.selection(document.getElementById('tab-unified')).selected") == ['dynamo@0.6.1', 'vllm_python@0.31.0']
        assert any('/viewer-data/' in path for path in requests), 'older capture must fetch JSON transport'
        identity = 'vllm_rust@0.27.1'
        requests.clear()
        driver.get(base + '?' + urlencode({'slots_tab-unified': identity + ',,'}))
        _ready(driver, 'tab-unified')
        capture = driver.execute_script("""
            const panel = document.getElementById('tab-unified');
            const key = window.__conformanceController.selection(panel).selected[0];
            const candidate = window.__CONFORMANCE_PAGE.tabs.find(tab => tab.id === panel.id)
              .candidates.find(item => item.key === key);
            const row = window.__conformanceModel.rows.find(row => row.family === 'nemotron_v3');
            const leaves = cell => cell.variants ? cell.variants.flatMap(leaves) : [cell];
            return {key, identity: candidate?.identity, captured: Object.values(row.cells).flatMap(leaves)
                .filter(cell => !cell.tooltip?.na_note).map(cell =>
                    cell.tooltip?.candidates.find(item => item.key === key)?.block)
                .filter(block => block && !block.unavailable && !block.unavailable_code).length};
        """)
        assert capture['identity'] == identity, (identity, capture)
        assert capture['captured'] > 0, (
            f"retained {identity} must keep its recorded Nemotron results")
        assert_viewer_data_matches_history(historical_published_viewer, requests, [identity])
        # The second recent patch is selectable through the actual control too.
        requests.clear()
        driver.get(base)
        _ready(driver, 'tab-unified')
        driver.find_element('css selector', "[data-slot-open='0']").click()
        Select(driver.find_element('css selector', "[data-slot-capture='0']")).select_by_value(previous_dynamo_capture['key'])
        WebDriverWait(driver, 90).until(lambda page: page.execute_script("return window.__conformanceLoader.pending === 0 && document.getElementById('tab-unified')._comparisonSelection.keys[0] === arguments[0]", previous_dynamo_capture['key']))
        expected_fetches = assert_viewer_data_matches_history(
            historical_published_viewer, requests,
            [identity_for_key(catalog, 'dynamo'), previous_dynamo_capture['identity']])
        assert expected_fetches, 'previous release must fetch its retained history resources'
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
        assert not driver.find_elements('css selector', '#tab-unified .matrix-cell')
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
        settle(driver, 'tab-unified')
        assert driver.execute_script("return window.__conformanceController.selection(document.getElementById('tab-unified')).selected") == ['dynamo']
        assert not driver.find_elements('css selector', f"option[value='{previous_dynamo_capture['key']}']")
        assert not any('/viewer-data/' in path for path in requests), (
            'history assets must wait until the publication discovery response is released', requests)
        if change_selection:
            _choose_engine(driver, 0, 'vllm', programmatic=True)
            settle(driver, 'tab-unified')
        expected = driver.execute_script("return window.__conformanceController.selection(document.getElementById('tab-unified')).selected")
        expected_identities = driver.execute_script("""
            const keys = arguments[0], candidates = window.__CONFORMANCE_PAGE.tabs.find(tab=>tab.id==='tab-unified').candidates;
            return candidates.filter(candidate=>keys.includes(candidate.key)).map(candidate=>candidate.identity);
        """, expected)
        cell = driver.find_element('css selector', '#tab-unified .matrix-cell[data-ttip-click=pin]')
        # Establish retained popup state while history loading blocks user input.
        driver.execute_script("arguments[0].click()", cell)
        WebDriverWait(driver, 10).until(lambda page: page.find_elements('css selector', '.ttip-pinned .ttip-head'))
        generation = driver.execute_script('return window.__conformanceLoader.generation')
        gate[1].set()
        _ready(driver, 'tab-unified')
        assert driver.execute_script("return document.getElementById('viewer').dataset.historyAvailable") == 'true'
        assert driver.execute_script("return document.getElementById('tab-unified')._comparisonSelection.keys") == expected
        assert driver.execute_script('return window.__conformanceLoader.generation') > generation
        assert_viewer_data_matches_history(historical_published_viewer, requests, expected_identities)
        assert driver.execute_script("return window.__CONFORMANCE_PAGE.tabs.find(tab=>tab.id==='tab-unified').candidates.some(candidate=>candidate.key===arguments[0])", previous_dynamo_capture['key'])
        for transposed in (False, True):
            toggle = driver.find_element('css selector', '[data-transpose-toggle]')
            if toggle.is_selected() != transposed:
                toggle.send_keys(Keys.SPACE)
            assert driver.find_elements('css selector', '.ttip-pinned .ttip-head')
            missing = driver.execute_script("""return [...document.querySelectorAll('#tab-unified [data-ttip-id]')]
                .filter(cell => !window.__conformanceTooltipModel(cell)
                    || (cell.matches('.matrix-cell') && !window.__conformanceCellModel(cell)))
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
            window.previousTooltipCell = panel.querySelector('.matrix-cell[data-ttip-id]');
            for (let index = 0; index < 2; index++) {
                const commit = window.ConformanceView.prepareUpdate(panel, window.__conformanceModel, window.__CONFORMANCE_PAGE);
                const pin = window.__conformanceController.disposePanel(panel);
                commit();
                window.__conformanceController.updatePanel(panel, pin);
            }
        """)
        cell = driver.find_element('css selector', '#tab-unified .matrix-cell[data-ttip-click=pin]')
        cell.click()
        WebDriverWait(driver, 10).until(lambda page: page.find_elements('css selector', '.ttip-pinned .ttip-head'))
        assert driver.execute_script("return Boolean(window.__conformanceCellModel(window.previousTooltipCell))")
        driver.find_element('css selector', "[data-slot-open='1']").send_keys(Keys.ENTER)
        driver.find_element('css selector', "[data-picker-engine-choice='vllm']").click()
        _ready(driver, 'tab-unified')
        assert driver.find_elements('css selector', '.ttip-pinned .ttip-head')
        assert driver.execute_script("return window.__conformanceCellModel(window.previousTooltipCell) === undefined")
        assert driver.execute_script("return window.__conformanceTooltipModel(window.previousTooltipCell) === undefined")
    finally:
        driver.quit()


@pytest.fixture
def delivery_clock(tmp_path, monkeypatch):
    clock = {"now": 0.0, "sleeps": []}
    lock = threading.Lock()
    def sleep(seconds):
        with lock:
            clock["sleeps"].append(seconds)
            clock["now"] += seconds
    monkeypatch.setattr(transport, "_run_http_scan", transport._scan_http_resources)
    monkeypatch.setattr(transport, "time", SimpleNamespace(monotonic=lambda: clock["now"], time=lambda: 0, sleep=sleep))
    monkeypatch.setattr(transport.random, "uniform", lambda low, high: 0)
    report = tmp_path.parent / (tmp_path.name + "-delivery.json")
    monkeypatch.setenv("CONFORMANCE_DELIVERY_REPORT", str(report))
    clock["report"] = report
    return clock


@pytest.mark.parametrize("status", [429, 502, 503, 504, 401, 403, 404, 500])
def test_http_delivery_status_classification(tmp_path, delivery_clock, status):
    (tmp_path / "index.html").write_bytes(b"qualified")
    server, requests = _server(root=tmp_path, fault={"path": "index.html", "responses": [status] * 4})
    try:
        with pytest.raises(HTTPError) as failure:
            verify_http_publication(tmp_path, f"http://127.0.0.1:{server.server_port}")
        assert failure.value.code == status
        retryable = status in {429, 502, 503, 504}
        assert len(requests) == (4 if retryable else 1)
        assert delivery_clock["sleeps"] == ([1, 2, 4] if retryable else [])
        report = json.loads(delivery_clock["report"].read_bytes())["checks"][0]
        assert not report["success"]
        resource = report["resources"]["index.html"]
        assert resource["outcome"] == "failed"
        assert resource["attempts"][-1]["request_ids"] == {"X-GitHub-Request-Id": "fixture-request"}
        assert resource["url"].endswith("/index.html")
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("header,delay", [("5", 5), ("Thu, 01 Jan 1970 00:00:09 GMT", 9),
                                          ("Wed, 31 Dec 1969 23:59:59 GMT", 1), ("nonsense", 1), ("999999", None)])
def test_http_delivery_retry_after_is_bounded(tmp_path, delivery_clock, header, delay):
    (tmp_path / "index.html").write_bytes(b"qualified")
    server, requests = _server(root=tmp_path, fault={"path": "index.html", "responses": [503], "retry_after": header})
    try:
        if delay is None:
            with pytest.raises(ValueError, match="budget exhausted"):
                verify_http_publication(tmp_path, f"http://127.0.0.1:{server.server_port}")
            assert len(requests) == 1
            assert delivery_clock["sleeps"] == []
        else:
            assert verify_http_publication(tmp_path, f"http://127.0.0.1:{server.server_port}") == {"resources": 1, "bytes": 9}
            assert len(requests) == 2
            assert delivery_clock["sleeps"] == [delay]
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
@pytest.mark.parametrize("destination_status", [200, 503])
def test_http_delivery_rejects_redirect_without_requesting_destination(tmp_path, delivery_clock, status, destination_status):
    qualified, served = tmp_path / "qualified", tmp_path / "served"
    qualified.mkdir()
    served.mkdir()
    for path in (qualified / "index.html", served / "redirected"):
        path.write_bytes(b"qualified")
    server, requests = _server(root=served, fault={"path": "", "location": "/redirected",
        "responses": {"/index.html": [status] * 4, "/redirected": [503] * 4 if destination_status == 503 else []}})
    try:
        with pytest.raises(HTTPError) as failure:
            verify_http_publication(qualified, f"http://127.0.0.1:{server.server_port}")
        assert failure.value.code == status
        assert requests == ["/index.html"]
        assert delivery_clock["sleeps"] == []
        resource = json.loads(delivery_clock["report"].read_bytes())["checks"][0]["resources"]["index.html"]
        assert resource["outcome"] == "failed"
        assert [attempt["status"] for attempt in resource["attempts"]] == [status]
    finally:
        server.shutdown()
        server.server_close()


def test_http_delivery_identity_preserves_budget_failure(tmp_path, monkeypatch):
    metadata = tmp_path / "conformance/viewer-publication.json"
    metadata.parent.mkdir()
    metadata.write_bytes(b'{"source_revision":"qualified"}')
    report = tmp_path.parent / (tmp_path.name + "-delivery.json")
    monkeypatch.setenv("CONFORMANCE_DELIVERY_REPORT", str(report))
    server, requests = _server(root=tmp_path, fault={"path": "viewer-publication.json",
        "responses": [503], "retry_after": "999999"})
    try:
        with pytest.raises(ValueError, match="verification budget exhausted before retry"):
            verify_http_publication_identity(tmp_path, f"http://127.0.0.1:{server.server_port}")
        assert requests == ["/conformance/viewer-publication.json"]
        assert "budget exhausted" in json.loads(report.read_bytes())["checks"][0]["error"]
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("report_fault,browser_failure,cleanup_failure", [
    (None, False, None), (None, True, None),
    ("malformed", False, None), ("malformed", True, None),
    ("write", False, None), ("write", True, None),
    (None, False, "quit"), (None, False, "shutdown"),
])
def test_delivery_report_failure_preserves_browser_error_and_cleanup(tmp_path, monkeypatch, report_fault, browser_failure, cleanup_failure):
    calls = []
    report = tmp_path / "delivery.json"
    report.write_bytes(b"{" if report_fault == "malformed" else b'{"checks": []}')
    monkeypatch.setenv("CONFORMANCE_DELIVERY_REPORT", str(report))
    module = sys.modules[__name__]
    def cleanup(name):
        calls.append(name)
        if name == cleanup_failure:
            raise OSError(f"{name} failure")
    server = SimpleNamespace(shutdown=lambda: cleanup("shutdown"), server_close=lambda: cleanup("close"), server_port=1)
    driver = SimpleNamespace(get=lambda url: None, quit=lambda: cleanup("quit"),
        execute_script=lambda script: None if "const tab =" in script else "true")
    browser_error = RuntimeError("browser failure")
    def settle_page(*args, **kwargs):
        if browser_failure:
            raise browser_error
    monkeypatch.setattr(module, "_server", lambda **kwargs: (server, []))
    monkeypatch.setattr(module, "_driver", lambda: driver)
    monkeypatch.setattr(module, "settle", settle_page)
    monkeypatch.setattr(module, "_assert_generation_provenance", lambda *args: None)
    monkeypatch.setattr(module, "publication_metadata", lambda site: {"revision": "qualified"})
    monkeypatch.setattr(module, "verify_http_publication", lambda *args: None)
    monkeypatch.setattr(module, "verify_http_publication_identity", lambda *args: None)
    monkeypatch.setattr(module, "runtime_catalog", lambda site: {
        "metadata": {"selection_policy": {"unified": {"reference": "dynamo"}}},
        "sources": {"candidates": [{"implementation": "dynamo", "version": "1.0.0",
                                   "native_unified_families": ["family"], "capture_families": ["family"]}]}})
    if report_fault == "write":
        write_text = Path.write_text
        def fail_report_write(path, *args, **kwargs):
            if path == report:
                raise OSError("report write failure")
            return write_text(path, *args, **kwargs)
        monkeypatch.setattr(Path, "write_text", fail_report_write)
    request = SimpleNamespace(config=SimpleNamespace(getoption=lambda name: None))
    def journey():
        test_publication_delivery_matches_qualified_artifact(tmp_path, request, monkeypatch, tmp_path)
    try:
        if browser_failure:
            with pytest.raises(RuntimeError, match="browser failure") as failure:
                journey()
            assert failure.value is browser_error
            if report_fault:
                assert isinstance(failure.value.__cause__, json.JSONDecodeError if report_fault == "malformed" else OSError)
        elif cleanup_failure:
            with pytest.raises(OSError, match=f"{cleanup_failure} failure"):
                journey()
        elif report_fault:
            with pytest.raises(json.JSONDecodeError if report_fault == "malformed" else OSError):
                journey()
        else:
            journey()
        if report_fault is None:
            assert json.loads(report.read_bytes())["browser_journey"] == {
                "success": not browser_failure, "publication": {"revision": "qualified"}}
    finally:
        assert calls == ["quit", "shutdown", "close"]


def test_http_delivery_recovery_still_rejects_corruption(tmp_path, delivery_clock):
    (tmp_path / "index.html").write_bytes(b"qualified")
    server, requests = _server(root=tmp_path, fault={"path": "index.html", "responses": [503, "corrupt"]})
    try:
        with pytest.raises(ValueError, match="differs from qualified artifact"):
            verify_http_publication(tmp_path, f"http://127.0.0.1:{server.server_port}")
        assert len(requests) == 2
        assert not json.loads(delivery_clock["report"].read_bytes())["checks"][0]["success"]
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("error,retryable", [(TimeoutError(), True), (ConnectionResetError(), True),
    (URLError(TimeoutError()), True), (URLError(ConnectionResetError()), True),
    (IncompleteRead(b"partial", 2), True), (URLError(IncompleteRead(b"partial", 2)), True),
    (ssl.SSLCertVerificationError(), False), (URLError(ssl.SSLCertVerificationError()), False),
    (URLError("name resolution failed"), False), (ValueError("corrupt bytes"), False)])
def test_http_delivery_transport_error_classification(error, retryable):
    assert transport.transient_http_error(error) is retryable


def incomplete_http_responses(data):
    length = {"Content-Length": str(len(data))}
    chunked = {"Transfer-Encoding": "chunked"}
    chunk = f"{len(data):x}\r\n".encode()
    partial = data[:-1]
    return {
        "length-empty": {"headers": length, "body": b""},
        "length-partial": {"headers": length, "body": partial},
        "length-complete": {"headers": {"Content-Length": str(len(data) + 1)}, "body": data},
        "chunked-empty": {"headers": chunked, "body": b""},
        "chunked-partial": {"headers": chunked, "body": f"{len(partial):x}\r\n".encode() + partial + b"\r\n"},
        "chunked-complete": {"headers": chunked, "body": chunk + data + b"\r\n"},
        "chunked-midchunk": {"headers": chunked, "body": chunk + partial},
    }


@pytest.mark.parametrize("framing,completion", [
    ("length-empty", "correct"), ("length-partial", "correct"),
    ("length-complete", "correct"),
    ("chunked-empty", "correct"), ("chunked-partial", "correct"),
    ("chunked-complete", "correct"), ("chunked-midchunk", "correct"),
    ("length-partial", "exhausted"), ("chunked-complete", "exhausted"),
    ("length-partial", "corrupt"), ("chunked-complete", "corrupt"),
])
def test_http_delivery_incomplete_framing_retries_only_failed_file(tmp_path, delivery_clock, framing, completion):
    data = b"qualified"
    for name in ("index.html", "sidecar.json"):
        (tmp_path / name).write_bytes(data)
    partial = incomplete_http_responses(data)[framing]
    responses = [partial] * 4 if completion == "exhausted" else [partial, "corrupt" if completion == "corrupt" else None]
    server, requests = _server(root=tmp_path, fault={"path": "index.html", "responses": responses})
    try:
        if completion == "correct":
            assert verify_http_publication(tmp_path, f"http://127.0.0.1:{server.server_port}") == {"resources": 2, "bytes": 18}
            assert requests.count("/sidecar.json") == 1
        else:
            with pytest.raises(IncompleteRead if completion == "exhausted" else ValueError):
                verify_http_publication(tmp_path, f"http://127.0.0.1:{server.server_port}")
            assert requests.count("/sidecar.json") <= 1
        attempts = 4 if completion == "exhausted" else 2
        assert requests.count("/index.html") == attempts
        assert delivery_clock["sleeps"] == ([1, 2, 4] if completion == "exhausted" else [1])
        report = json.loads(delivery_clock["report"].read_bytes())["checks"][0]
        assert report["success"] == (completion == "correct")
        assert set(report["resources"]) == {"index.html", "sidecar.json"}
        resource = report["resources"]["index.html"]
        assert resource["outcome"] == ("recovered" if completion == "correct" else "failed")
        assert len(resource["attempts"]) == attempts
        assert resource["attempts"][0]["error"].startswith("IncompleteRead(")
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("shape", ["short-complete", "short-unframed", "short-chunked", "corrupt", "oversized-length", "oversized-chunked"])
def test_http_delivery_complete_wrong_bytes_never_retry(tmp_path, delivery_clock, shape):
    data = b"qualified"
    (tmp_path / "index.html").write_bytes(data)
    body = data[:-1] if shape.startswith("short") else data[:-1] + b"X" if shape == "corrupt" else data + b"X" * (1 << 20)
    headers = {} if shape == "short-unframed" else {"Content-Length": str(len(body))}
    if shape.endswith("chunked"):
        headers = {"Transfer-Encoding": "chunked"}
        body = f"{len(body):x}\r\n".encode() + body + b"\r\n0\r\n\r\n"
    server, requests = _server(root=tmp_path, fault={"path": "index.html", "responses": [{"headers": headers, "body": body}]})
    try:
        with pytest.raises(ValueError, match="differs from qualified artifact") as failure:
            verify_http_publication(tmp_path, f"http://127.0.0.1:{server.server_port}")
        assert requests == ["/index.html"]
        assert delivery_clock["sleeps"] == []
        assert len(str(failure.value)) < 16384
        report = json.loads(delivery_clock["report"].read_bytes())["checks"][0]
        assert not report["success"] and len(report["resources"]["index.html"]["attempts"]) == 1
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("framing", ["length-partial", "chunked-complete"])
def test_http_delivery_identity_framing_recovery_does_not_scan_other_resources(tmp_path, delivery_clock, framing):
    data = b'{"revision":"qualified"}'
    metadata = tmp_path / "conformance/viewer-publication.json"
    metadata.parent.mkdir()
    metadata.write_bytes(data)
    (tmp_path / "index.html").write_bytes(b"separate resource")
    server, requests = _server(root=tmp_path, fault={"path": "viewer-publication.json", "responses": [incomplete_http_responses(data)[framing]]})
    try:
        assert verify_http_publication_identity(tmp_path, f"http://127.0.0.1:{server.server_port}") == {"resources": 1, "bytes": len(data)}
        assert requests == ["/conformance/viewer-publication.json"] * 2
        assert delivery_clock["sleeps"] == [1]
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("framing", ["length-partial", "chunked-complete"])
def test_http_delivery_spawned_process_recovers_incomplete_framing(tmp_path, monkeypatch, framing):
    data = b"qualified"
    (tmp_path / "index.html").write_bytes(data)
    report = tmp_path.parent / (tmp_path.name + "-delivery.json")
    monkeypatch.setenv("CONFORMANCE_DELIVERY_REPORT", str(report))
    server, requests = _server(root=tmp_path, fault={"path": "index.html", "responses": [incomplete_http_responses(data)[framing]]})
    try:
        assert verify_http_publication(tmp_path, f"http://127.0.0.1:{server.server_port}") == {"resources": 1, "bytes": 9}
        assert requests == ["/index.html"] * 2
        resource = json.loads(report.read_bytes())["checks"][0]["resources"]["index.html"]
        assert resource["outcome"] == "recovered" and len(resource["attempts"]) == 2
        assert not transport.multiprocessing.active_children()
    finally:
        server.shutdown()
        server.server_close()


def test_http_delivery_concurrent_recoveries_account_for_every_file(tmp_path, delivery_clock):
    paths = ["index.html", *[f"history-{i}.json" for i in range(24)]]
    for path in paths:
        (tmp_path / path).write_bytes(b"qualified")
    server, requests = _server(root=tmp_path, fault={"path": "", "responses": {f"/{path}": [503] for path in paths}})
    try:
        assert verify_http_publication(tmp_path, f"http://127.0.0.1:{server.server_port}") == {"resources": len(paths), "bytes": 9 * len(paths)}
        assert all(requests.count(f"/{path}") == 2 for path in paths)
        resources = json.loads(delivery_clock["report"].read_bytes())["checks"][0]["resources"]
        assert set(resources) == set(paths)
        assert all(row["outcome"] == "recovered" and len(row["attempts"]) == 2 for row in resources.values())
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("responses", [["corrupt"], [None, "corrupt"]])
def test_http_delivery_rejects_stale_or_changing_identity(tmp_path, delivery_clock, responses):
    metadata = tmp_path / "conformance/viewer-publication.json"
    metadata.parent.mkdir()
    metadata.write_bytes(b'{"source_revision":"qualified"}')
    server, requests = _server(root=tmp_path, fault={"path": "viewer-publication.json", "responses": responses.copy()})
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        if len(responses) == 2:
            verify_http_publication_identity(tmp_path, base)
        with pytest.raises(ValueError, match="published resource differs from qualified artifact: .*viewer-publication.json"):
            verify_http_publication_identity(tmp_path, base)
        assert len(requests) == len(responses)
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("fault,persistent", [("timeout", False), ("timeout", True), ("reset", False)])
def test_http_delivery_real_transport_failures(tmp_path, monkeypatch, fault, persistent):
    (tmp_path / "index.html").write_bytes(b"qualified")
    monkeypatch.setattr(transport, "HTTP_TIMEOUT", 0.05)
    monkeypatch.setattr(transport, "HTTP_BUDGET", 15)
    report_path = tmp_path.parent / (tmp_path.name + "-report.json")
    monkeypatch.setenv("CONFORMANCE_DELIVERY_REPORT", str(report_path))
    server, requests = _server(root=tmp_path, fault={"path": "index.html", "responses": [fault] * (4 if persistent else 1)})
    try:
        if persistent:
            with pytest.raises(TimeoutError):
                verify_http_publication(tmp_path, f"http://127.0.0.1:{server.server_port}")
        else:
            assert verify_http_publication(tmp_path, f"http://127.0.0.1:{server.server_port}")["resources"] == 1
        assert len(requests) == (4 if persistent else 2)
        report = json.loads(report_path.read_bytes())["checks"][0]
        assert report["success"] is not persistent
        assert len(report["resources"]["index.html"]["attempts"]) == len(requests)
    finally:
        server.shutdown()
        server.server_close()


def test_http_delivery_hard_deadline_stops_a_trickling_body(tmp_path, monkeypatch):
    (tmp_path / "index.html").write_bytes(b"x" * 150)
    monkeypatch.setattr(transport, "HTTP_BUDGET", 0.2)
    monkeypatch.setattr(transport, "HTTP_TIMEOUT", 0.2)
    report_path = tmp_path.parent / (tmp_path.name + "-report.json")
    monkeypatch.setenv("CONFORMANCE_DELIVERY_REPORT", str(report_path))
    server, requests = _server(root=tmp_path, fault={"path": "index.html", "responses": ["trickle"]})
    before = transport.time.monotonic()
    try:
        with pytest.raises(TimeoutError, match="budget"):
            verify_http_publication(tmp_path, f"http://127.0.0.1:{server.server_port}")
        assert transport.time.monotonic() - before < 1
        assert len(requests) == 1
        data = report_path.read_bytes()
        report = json.loads(data)["checks"][0]
        assert not report["success"]
        assert report["resources"]["index.html"]["attempts"][0]["status"] == 200
        assert report["resources"]["index.html"]["outcome"] in {"interrupted", "budget-exhausted"}
        assert not transport.multiprocessing.active_children()
        threading.Event().wait(0.1)
        assert report_path.read_bytes() == data
    finally:
        server.shutdown()
        server.server_close()
