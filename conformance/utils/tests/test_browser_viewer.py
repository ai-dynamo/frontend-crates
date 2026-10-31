from __future__ import annotations

import threading
import hashlib
import gzip
import json
import re
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit
from fixture_test_support import historical_source_bytes, assert_viewer_data_matches_history, identity_for_key
from verify_publication_transport import publication_metadata, runtime_catalog

import pytest

selenium = pytest.importorskip("selenium")
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select, WebDriverWait

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "conformance/utils/src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import unified_history
import capture_sources
from browser_helpers import raw_view_download, settle


def _published_catalog(site):
    return runtime_catalog(site)


def _published_capture(site, implementation=None, version=None):
    catalog = _published_catalog(site)
    implementation = implementation or catalog["metadata"]["selection_policy"]["unified"]["reference"]
    return max((item for item in catalog["sources"]["unified"]
                if item["kind"] == "resolved-capture" and item["implementation"] == implementation
                and (version is None or item["version"] == version)),
               key=lambda item: unified_history._capture_release_sort_key(item["version"]))


def _published_capture_index(site, implementation=None, version=None):
    return _published_capture(site, implementation, version)["recipe_index"]["path"]


def _server(prefix: str = "", gate=None, gate_path: str | None = None, fault=None, root: Path = ROOT, missing_paths=(), external=False, html_transform=None):
    requests = []
    held = False
    gate_lock = threading.Lock()

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(root), **kwargs)

        def log_message(self, *_args):
            return

        def do_GET(self):
            path = urlsplit(self.path).path
            requests.append(path)
            resource_path = path[len(prefix):] if prefix and path.startswith(prefix + "/") else path
            if resource_path in missing_paths:
                self.send_error(404)
                return
            nonlocal held
            should_hold = False
            with gate_lock:
                matches_gate = gate_path is not None and gate_path in path
                if gate and not held and matches_gate:
                    held = True
                    should_hold = True
            if should_hold:
                gate[0].set()
                assert gate[1].wait(60), "delayed resource was never released"
            matches_fault = fault and fault["path"] in path
            if matches_fault and not fault["used"]:
                fault["used"] = True
                if fault["kind"] == "missing":
                    self.send_error(404)
                    return
                data = Path(self.translate_path(self.path)).read_bytes()
                data = b"version https://git-lfs.github.com/spec/v1\n" if fault["kind"] == "pointer" else data[:-1] + bytes([data[-1] ^ 1])
                self.send_response(200)
                self.end_headers()
                self.wfile.write(data)
                return
            if (external or html_transform) and resource_path in ("/", "/index.html"):
                # Exercise on-demand transport faults independently of the HTML
                # cache. Recent embedded selections have separate HTTP proofs.
                html = (root / "index.html").read_text()
                if external:
                    html = re.sub(r'<script type="application/json" id="conformance-embedded">.*?</script>', '', html, flags=re.S)
                    html = re.sub(r'<script type="application/json" id="conformance-publication">.*?</script>', '', html, flags=re.S)
                data = (html_transform(html) if html_transform else html).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            return super().do_GET()

        def translate_path(self, path):
            parsed = urlsplit(path).path
            if prefix and parsed.startswith(prefix + "/"):
                parsed = parsed[len(prefix):]
            return super().translate_path(parsed)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, requests


def _driver():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(options=options)
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
        "source": "Object.defineProperty(window, 'indexedDB', { configurable: true, value: undefined });",
    })
    return driver


def _ready(driver, tab):
    # Race tests inspect the current matrix while an obsolete read is held.
    settle(driver, tab, history_ready=True, idle=False)


def _choose_engine(driver, slot: int, engine: str, *, programmatic=False):
    button = driver.find_element("css selector", f"[data-slot-open='{slot}']")
    if programmatic:
        # Exercise externally triggered selection while the loading modal blocks pointer input.
        driver.execute_script("arguments[0].click()", button)
    else:
        button.click()
    picker = WebDriverWait(driver, 10).until(EC.visibility_of_element_located(("css selector", ".comparison-picker[open]")))
    picker.find_element("css selector", f"[data-picker-engine-choice='{engine}']").click()
    return WebDriverWait(driver, 10).until(lambda page: page.find_element("css selector", f"[data-slot-capture='{slot}']"))


def _clear_slot(driver, slot: int):
    driver.find_element("css selector", f"[data-slot-open='{slot}']").click()
    picker = WebDriverWait(driver, 10).until(EC.visibility_of_element_located(("css selector", ".comparison-picker[open]")))
    picker.find_element("css selector", "[data-picker-engine-choice='none']").click()


def _historical_dynamo_key(driver):
    choices = [option.get_attribute('value') for option in Select(
        driver.find_element('css selector', '[data-slot-capture="0"]')).options
        if option.is_enabled() and option.get_attribute('value').startswith('dynamo@')]
    assert choices, 'the publication has no selectable historical Dynamo capture'
    return choices[0]


def test_browser_viewer_loads_default_slot_and_controls(published_viewer):
    server, requests = _server(root=published_viewer)
    driver = _driver()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        driver.get(f"{base}/index.html?tab=tab-unified&view=details&transpose=0&keep=1")
        _ready(driver, "tab-unified")
        assert driver.title == 'Dynamo Parser Conformance Table'
        assert driver.find_element('css selector', '#viewer > h1').text.startswith('Parser Conformance')
        assert len(driver.find_element('css selector', '.source-commit').text) == 12
        assert driver.execute_script("return Boolean(document.querySelector(arguments[0]))", "#tab-unified table[data-comparison-table] .matrix-cell")
        family_labels = {item.text for item in driver.find_elements("css selector", "#tab-unified .comparison-matrix tbody tr > th:first-child")}
        assert "deepseek_v4" in family_labels
        assert "deepseek_v3" not in family_labels
        assert "muse_glimmer" in family_labels
        assert driver.find_element("id", "viewer").get_attribute("data-catalog-revision")
        assert driver.find_elements('css selector', '#tab-unified .capture-slot')
        assert len(driver.find_elements('css selector', '#tab-unified .capture-slot')) == 3
        select = driver.find_element('css selector', "#tab-unified select[data-slot-capture='0']")
        assert select.get_attribute('value') == 'dynamo'
        assert driver.execute_script("return window.__conformanceController.selection(document.getElementById('tab-unified'))") == {'reference': None, 'selected': ['dynamo']}
        slot_engines = [slot.get_attribute('data-engine') for slot in driver.find_elements('css selector', '#tab-unified .capture-slot')]
        assert slot_engines == ['dynamo', 'none', 'none']
        assert driver.find_elements('css selector', '#viewer > .tabs .tab-button') == []
        loaded = [path for path in requests if "/viewer-data/" in path]
        assert loaded == [], 'default capture data must come entirely from HTML'
        assert not any('viewer-resolved-catalog' in path for path in requests)
        default_capture = _published_capture(published_viewer)
        version_label = default_capture["version"]
        assert select.get_attribute('aria-label') == f"Choose capture for comparison slot 1: Dynamo, v2 Rust {version_label} (stream, UnifiedParser)"
        expected_rows = next(item["native_unified_families"] for item in _published_catalog(published_viewer)["sources"]["candidates"]
                             if item["implementation"] == default_capture["implementation"] and item["version"] == default_capture["version"])
        assert driver.execute_script("return window.__conformanceModel.rows.filter(r => !r.section).length") == len(expected_rows)
        family_labels = driver.execute_script("return window.__conformanceModel.rows.filter(r => !r.section).map(r => r.model_label)")
        assert len(family_labels) == len(expected_rows)
        assert set(driver.execute_script("return window.__conformanceModel.rows.filter(r => !r.section).map(r => r.family)")) == set(expected_rows)
        assert family_labels == sorted(family_labels)
        assert all(" (v" not in label for label in family_labels), family_labels
        driver.find_element("css selector", "[data-transpose-toggle]").click()
        assert "transpose-mode" in driver.find_element("tag name", "body").get_attribute("class")
        assert driver.execute_script("return Boolean(document.querySelector(arguments[0]))", "#tab-unified table[data-comparison-table][data-transpose-table] .matrix-cell")
        driver.find_element("css selector", "[data-theme-toggle]").click()
        assert driver.find_element("tag name", "html").get_attribute("data-theme") == "dark"
        driver.find_element("css selector", "[data-transpose-toggle]").click()
        cell = driver.find_element("css selector", "#tab-unified table[data-comparison-table] .matrix-cell[data-ttip-id]")
        cell.click()
        popup = WebDriverWait(driver, 10).until(EC.visibility_of_element_located(("css selector", ".ttip-visible")))
        assert popup.text and "Golden output" in popup.text and 'Dynamo' in popup.text
        assert "ttip-pinned" in popup.get_attribute("class")
        assert driver.execute_script("return getComputedStyle(arguments[0]).resize", popup) == "both"
        popup.find_element("css selector", ".ttip-close").click()
        WebDriverWait(driver, 10).until(EC.invisibility_of_element_located(("css selector", ".ttip-pinned")))
        rust_select = _choose_engine(driver, 1, 'vllm')
        WebDriverWait(driver, 90).until(lambda _page: rust_select.is_displayed() and rust_select.is_enabled())
        rust = next(option.get_attribute('value') for option in rust_select.find_elements('tag name', 'option') if option.get_attribute('value').startswith('vllm_rust@'))
        Select(rust_select).select_by_value(rust)
        _ready(driver, 'tab-unified')
        assert driver.execute_script("return window.__conformanceController.selection(document.getElementById('tab-unified')).selected.includes(arguments[0])", rust)
        _clear_slot(driver, 1)
        state = driver.execute_script("return window.__conformanceController.selection(document.getElementById('tab-unified'))")
        assert state == {'reference': None, 'selected': ['dynamo']}
        assert 'keep=1' in driver.current_url
        driver.get(f"{base}/index.html?tab=tab-unified&keep=1#case")
        _ready(driver, "tab-unified")
        assert urlsplit(driver.current_url).path == "/index.html"
        parsed = urlsplit(driver.current_url)
        assert "keep=1" in parsed.query.split("&") and parsed.fragment == "case"
    finally:
        driver.quit()
        server.shutdown()


def test_historical_reference_switch_recolors_matrix(historical_published_viewer):
    server, _requests = _server(root=historical_published_viewer)
    driver = _driver()
    try:
        driver.get(f"http://127.0.0.1:{server.server_port}/?tab=tab-unified&transpose=0")
        _ready(driver, 'tab-unified')
        select = driver.find_element('css selector', "[data-slot-capture='0']")
        def color_summary():
            return driver.execute_script("""
                const counts = {};
                for (const segment of document.querySelectorAll('#tab-unified [data-comparison-table] .matrix-status')) {
                  const status = [...segment.classList].find(name => name.startsWith('status-'));
                  counts[status] = (counts[status] || 0) + 1;
                }
                return counts;
            """)
        newest_reference = color_summary()
        assert newest_reference, 'the comparison matrix has no status segments'
        Select(select).select_by_value(_historical_dynamo_key(driver))
        _ready(driver, 'tab-unified')
        older_reference = color_summary()
        assert older_reference != newest_reference, (
            f"changing the Dynamo Reference did not recolor the matrix: {newest_reference}"
        )
        Select(select).select_by_value('dynamo')
        _ready(driver, 'tab-unified')
    finally:
        driver.quit()
        server.shutdown()
        server.server_close()

def test_dynamo_capture_picker_keeps_recent_two_and_latest_per_historical_line(historical_published_viewer):
    server, _requests = _server(root=historical_published_viewer)
    driver = _driver()
    try:
        driver.get(f"http://127.0.0.1:{server.server_port}/index.html?tab=tab-unified")
        _ready(driver, "tab-unified")
        options = {option.get_attribute("value") for option in driver.find_elements(
            "css selector", "#tab-unified select[data-slot-capture='0'] option")}
        candidates = driver.execute_script("""
          return window.__CONFORMANCE_PAGE.tabs.find(item => item.id === 'tab-unified').candidates
            .filter(item => !item.source_blocker && item.has_captured_results !== false
              && (!(item.capture_families || item.native_unified_families)
                || (item.capture_families || item.native_unified_families).length));
        """)
        newest_v2_by_minor = {}
        newest_v1_by_producer_major = {}
        for item in candidates:
            if item.get("identity", "").startswith("dynamo_v2@") or item["key"].startswith("dynamo@"):
                version = item["version"]
                minor = tuple(int(part) for part in version.split(".")[:2])
                current = newest_v2_by_minor.get(minor)
                if current is None or tuple(int(part) for part in version.split(".")) > tuple(int(part) for part in current["version"].split(".")):
                    newest_v2_by_minor[minor] = item
            elif item["key"].startswith(("dynamo_v1@", "dynamo_v1_jail@")):
                producer = item["key"].split("@", 1)[0]
                version = item["version"]
                major = int(version.split(".", 1)[0])
                line = (producer, major)
                newest_v1_by_producer_major[line] = max(newest_v1_by_producer_major.get(line, version), version,
                                                         key=lambda value: tuple(int(part) for part in value.split(".")))
        expected = {item["key"] for item in newest_v2_by_minor.values()}
        expected.update(f"{producer}@{version}" for (producer, _major), version in newest_v1_by_producer_major.items())
        for producer in ("dynamo_v2", "dynamo_v1", "dynamo_v1_jail"):
            recent = sorted((item for item in candidates if item.get("identity", "").startswith(producer + "@")),
                            key=lambda item: unified_history._capture_release_sort_key(item["version"]), reverse=True)[:2]
            expected.update(item["key"] for item in recent)
        assert options & expected == expected
        assert not (options - expected - {"dynamo", ""}), sorted(options - expected - {"dynamo", ""})
    finally:
        driver.quit()
        server.shutdown()


def test_published_viewer_loads_resolved_json_without_yaml_histories(historical_published_viewer):
    site = historical_published_viewer
    catalog = json.loads(gzip.decompress((site / "conformance/viewer-resolved-catalog.json.gz").read_bytes()))
    server, requests = _server(root=site)
    driver = _driver()
    try:
        driver.get(f"http://127.0.0.1:{server.server_port}/index.html?tab=tab-unified&transpose=0")
        _ready(driver, "tab-unified")
        provenance = json.loads((site / "conformance/viewer-publication.json").read_text())
        generated = WebDriverWait(driver, 10).until(lambda page: page.find_element("css selector", ".generated") if "Generated on" in page.find_element("css selector", ".generated").text else False)
        generated_revision = provenance.get('generated_revision', provenance['revision'])
        identity = ' on commit ' if generated_revision == provenance['revision'] and not provenance.get('candidate_tree') else '; source commit '
        expected = f"Generated on {provenance['generated_on']}{identity}{provenance['revision'][:12]}"
        assert generated.text == expected
        assert provenance["command"] not in generated.text
        assert provenance["output"] not in generated.text
        assert generated.find_element("css selector", "a").get_attribute("href").endswith(provenance['revision'])
        assert generated_revision in generated.get_attribute('title')
        assert not driver.find_elements('css selector', 'h1 .source-commit')
        driver.find_element("css selector", "[data-view-detailed]").click()
        assert not any(node.is_displayed() for node in driver.find_elements('css selector', '.generated-command'))
        assert driver.find_element('css selector', '.case-descriptions').is_displayed()
        labels = driver.find_elements("css selector", ".capture-slot-selected-label")
        assert labels and all(driver.execute_script("return arguments[0].scrollHeight <= arguments[0].clientHeight", label) for label in labels)
        slot_tops = [slot.rect["y"] for slot in driver.find_elements("css selector", ".capture-slot")]
        assert max(slot_tops) - min(slot_tops) < 1
        dynamo_options = driver.find_elements("css selector", '[data-slot-capture="0"] option')
        v1_values = [option.get_attribute("value") for option in dynamo_options
                     if option.get_attribute("value").startswith("dynamo_v1")]
        assert v1_values
        available_v1_versions = driver.execute_script("""
          const candidates = window.__CONFORMANCE_PAGE.tabs.find(item => item.id === 'tab-unified').candidates;
            return candidates.filter(item => item.key.startsWith('dynamo_v1') && !item.source_blocker
            && item.has_captured_results !== false
            && (!(item.capture_families || item.native_unified_families)
              || (item.capture_families || item.native_unified_families).length))
            .map(item => item.key);
        """)
        for producer in ("dynamo_v1@", "dynamo_v1_jail@"):
            selected_versions = {value.rsplit("@", 1)[-1] for value in v1_values if value.startswith(producer)}
            available_versions = [value.rsplit("@", 1)[-1] for value in available_v1_versions if value.startswith(producer)]
            newest_by_major = {}
            for version in available_versions:
                major = version.split(".", 1)[0]
                newest_by_major[major] = max(newest_by_major.get(major, version), version,
                                             key=lambda value: tuple(int(part) for part in value.split(".")))
            recent = sorted(set(available_versions), key=unified_history._capture_release_sort_key, reverse=True)[:2]
            assert selected_versions == set(newest_by_major.values()) | set(recent), producer
        v2_values = {option.get_attribute("value") for option in dynamo_options
                     if option.get_attribute("value").startswith("dynamo@") or option.get_attribute("value") == "dynamo"}
        available_v2_versions = driver.execute_script("""
          const candidates = window.__CONFORMANCE_PAGE.tabs.find(item => item.id === 'tab-unified').candidates;
          return candidates.filter(item => (item.identity?.startsWith('dynamo_v2@') || item.key.startsWith('dynamo@')) && !item.source_blocker
            && item.has_captured_results !== false
            && (!(item.capture_families || item.native_unified_families)
              || (item.capture_families || item.native_unified_families).length))
            .map(item => ({key: item.key, version: item.version}));
        """)
        newest_by_minor = {}
        for item in available_v2_versions:
            version = item["version"]
            minor = tuple(int(part) for part in version.split(".")[:2])
            current = newest_by_minor.get(minor)
            if current is None or tuple(int(part) for part in version.split(".")) > tuple(int(part) for part in current["version"].split(".")):
                newest_by_minor[minor] = item
        recent_v2 = sorted(available_v2_versions,
                           key=lambda item: unified_history._capture_release_sort_key(item["version"]), reverse=True)[:2]
        assert v2_values == {item["key"] for item in [*newest_by_minor.values(), *recent_v2]}
        direct_v1_versions = {value.rsplit("@", 1)[-1] for value in v1_values if value.startswith("dynamo_v1@")}
        jail_v1_versions = {value.rsplit("@", 1)[-1] for value in v1_values if value.startswith("dynamo_v1_jail@")}
        assert direct_v1_versions and jail_v1_versions
        assert {"dynamo_v1@5.1.3", "dynamo_v1_jail@5.1.3"} <= set(v1_values)
        selected_identity = driver.execute_script("""
          const tab = window.__CONFORMANCE_PAGE.tabs.find(item => item.id === 'tab-unified');
          return tab.candidates.find(item => item.key === 'dynamo')?.source_id;
        """)
        assert selected_identity
        assert_viewer_data_matches_history(site, requests, [identity_for_key(catalog, 'dynamo')])
        assert not any(path.endswith(".yaml") or "/fixtures-unified-v2/" in path for path in requests)
        resources = {item["recipe_index"]["path"]: item for item in catalog["sources"]["unified"] if item["kind"] == "resolved-capture"}
        assert any(item["capture_identity"] == selected_identity for item in resources.values())
        requests.clear()
        driver.find_element("css selector", "#tab-unified [data-slot-open='1']").click()
        WebDriverWait(driver, 10).until(EC.visibility_of_element_located(("css selector", ".comparison-picker[open]")))
        driver.execute_async_script("const done = arguments[0]; requestAnimationFrame(() => requestAnimationFrame(done));")
        identities_after_picker = {resources[urlsplit(path).path.removeprefix("/")]["capture_identity"]
                                   for path in requests if urlsplit(path).path.removeprefix("/") in resources}
        assert not identities_after_picker, "opening a selector fetched capture payloads"
        driver.find_element("css selector", ".comparison-picker[open]").send_keys("\ue00c")
        second = _choose_engine(driver, 1, "dynamo")
        Select(second).select_by_value("dynamo_v1@10.0.7")
        _ready(driver, "tab-unified")
        third = _choose_engine(driver, 2, "dynamo")
        Select(third).select_by_value("dynamo_v1_jail@10.0.7")
        _ready(driver, "tab-unified")
        visible_families = set(driver.execute_script("return window.__conformanceModel.rows.filter(row => !row.section).map(row => row.model_label)"))
        selected_sources = {selected_identity, "dynamo_v1@unified@10.0.7", "dynamo_v1_jail@unified@10.0.7"}
        expected_families = {family for candidate in catalog["sources"]["candidates"]
                             if f"{candidate['implementation']}@{candidate['mode']}@{candidate['version']}" in selected_sources
                             for family in candidate.get("native_unified_families", candidate["capture_families"])}
        assert visible_families == expected_families
        assert {"qwen3", "muse_glimmer", "muse_spark"} <= visible_families
        driver.execute_script("""
          for (const value of ['dynamo_v1@5.1.3', arguments[0], 'dynamo']) {
            const select = document.querySelector('#tab-unified [data-slot-capture="0"]');
            select.value = value;
            select.dispatchEvent(new Event('change', {bubbles:true}));
          }
        """, _historical_dynamo_key(driver))
        _ready(driver, "tab-unified")
        assert driver.execute_script("return window.__conformanceController.selection(document.getElementById('tab-unified')).selected") == [
            "dynamo", "dynamo_v1@10.0.7", "dynamo_v1_jail@10.0.7"]
        driver.find_element("css selector", "[data-transpose-toggle]").click()
        WebDriverWait(driver, 10).until(lambda page: page.execute_script("return document.body.classList.contains('transpose-mode')"))
        cell = driver.execute_script("return [...document.querySelectorAll('#tab-unified table[data-comparison-table] .matrix-cell[data-ttip-id]')].find(cell => cell.offsetWidth && cell.offsetHeight);")
        assert cell is not None
        driver.execute_script("window.__buildTooltip(arguments[0]); arguments[0].dispatchEvent(new PointerEvent('pointerdown', {bubbles:true, pointerType:'touch'})); arguments[0].click();", cell)
        popup = WebDriverWait(driver, 10).until(EC.visibility_of_element_located(("css selector", ".ttip-pinned")))
        assert "10.0.7" in popup.text
        assert "golden" in popup.text.lower()
        assert not any(path.endswith(".yaml") or "/fixtures-unified-v2/" in path for path in requests)
        fetched = {resources[urlsplit(path).path.removeprefix("/")]["capture_identity"]
                   for path in requests if urlsplit(path).path.removeprefix("/") in resources}
        allowed = {selected_identity, "dynamo_v2@unified@0.7.24", "dynamo_v2@unified@0.6.1",
                   "dynamo_v1@unified@5.1.3", "dynamo_v1@unified@10.0.7", "dynamo_v1_jail@unified@10.0.7"}
        # Switching engines may admit the picker's most recent historical capture
        # before the following explicit selection replaces it.
        allowed.update(f"dynamo_v2@unified@{item['version']}" for item in recent_v2)
        assert fetched <= allowed

    finally:
        driver.quit()
        server.shutdown()


def test_browser_viewer_resolves_nested_http_prefix_and_legacy_empty_reference(published_viewer):
    server, requests = _server("/nested", root=published_viewer)
    driver = _driver()
    try:
        driver.get(f"http://127.0.0.1:{server.server_port}/nested/index.html?tab=tab-unified&base_tab-unified=&transpose=0")
        _ready(driver, "tab-unified")
        assert driver.execute_script("return window.__conformanceController.selection(document.getElementById('tab-unified'))") == {"reference": None, "selected": ["dynamo"]}
        assert not any(path.endswith("viewer-catalog.json") for path in requests)
        assert driver.current_url.startswith(f"http://127.0.0.1:{server.server_port}/nested/")
        assert driver.execute_script("return Boolean(document.querySelector(arguments[0]))", "#tab-unified .matrix-cell .status-pass, #tab-unified .matrix-cell .status-leak")
        driver.refresh()
        _ready(driver, "tab-unified")
        assert driver.execute_script("return window.__conformanceController.selection(document.getElementById('tab-unified'))") == {"reference": None, "selected": ["dynamo"]}
    finally:
        driver.quit()
        server.shutdown()


def test_same_engine_slots_none_geometry_duplicate_prevention_and_reset(historical_published_viewer):
    server, _requests = _server(root=historical_published_viewer)
    driver = _driver()
    try:
        driver.get(f"http://127.0.0.1:{server.server_port}/index.html?tab=tab-unified&transpose=0&keep=1#saved")
        _ready(driver, "tab-unified")
        none_slot = driver.find_element("css selector", "#tab-unified .capture-slot:nth-child(2)")
        assert none_slot.is_displayed()
        bounds = none_slot.rect
        assert bounds["width"] > 0 and bounds["height"] > 0
        assert none_slot.get_attribute("data-engine") == "none"
        assert not none_slot.find_elements("css selector", "[data-slot-capture]")
        assert none_slot.find_element("css selector", "[data-slot-engine]").text == "None"
        assert none_slot.find_element("css selector", "[data-slot-badge]").text == "—"

        second = _choose_engine(driver, 1, "dynamo")
        v1_options = [option.get_attribute("value") for option in second.find_elements("css selector", "option")
                      if option.get_attribute("value").startswith("dynamo_v1@9.")]
        assert len(v1_options) == 1
        Select(second).select_by_value(v1_options[0])
        _ready(driver, "tab-unified")
        assert driver.execute_script("return window.__conformanceController.selection(document.getElementById('tab-unified')).selected") == ["dynamo", v1_options[0]]
        assert not second.find_element("css selector", "option[value='dynamo']").is_enabled()
        assert all("selected in another slot" not in option.text for option in second.find_elements("tag name", "option"))

        _clear_slot(driver, 1)
        assert driver.execute_script("return window.__conformanceController.selection(document.getElementById('tab-unified')).selected") == ["dynamo"]
        assert driver.find_element("css selector", "#tab-unified .capture-slot:nth-child(2)").get_attribute("data-engine") == "none"
        assert not driver.find_elements("css selector", "#tab-unified .capture-slot:nth-child(2) [data-slot-capture]")
        driver.find_element("css selector", "[data-reset]").click()
        _ready(driver, "tab-unified")
        assert driver.execute_script("return window.__conformanceController.selection(document.getElementById('tab-unified')).selected") == ["dynamo"]
        parsed = urlsplit(driver.current_url)
        assert dict(parse_qsl(parsed.query)) == {"tab": "tab-unified", "keep": "1"}
        assert parsed.fragment == "saved"
    finally:
        driver.quit()
        server.shutdown()


def test_empty_slot_and_delayed_case_hints(published_viewer):
    server, _ = _server(root=published_viewer)
    driver = _driver()
    try:
        driver.get(f"http://127.0.0.1:{server.server_port}/?vertical=0")
        _ready(driver, 'tab-unified')
        none_slot = driver.find_element(By.CSS_SELECTOR, '.capture-slot:nth-child(2)')
        assert none_slot.get_attribute('data-engine') == 'none'
        assert not none_slot.find_elements(By.CSS_SELECTOR, '[data-slot-capture]')
        header = driver.find_element(By.CSS_SELECTOR, '.matrix-case-heading[aria-label="UNIFIED.1-tool_only"]')
        label = header.find_element(By.CSS_SELECTOR, '[data-hover-hint]')
        full = driver.execute_script('return window.__conformanceTooltipModel(arguments[0]).popupCase.label', header)
        driver.execute_script("""
            arguments[0].addEventListener('pointerenter', event => {
              const started=document.timeline.currentTime;
              const check=()=>{
                if(getComputedStyle(event.target,'::after').opacity==='1') event.target.hintDelay=document.timeline.currentTime-started;
                else requestAnimationFrame(check);
              };
              requestAnimationFrame(check);
            }, {once:true});
        """, label)
        ActionChains(driver).move_to_element(label).perform()
        WebDriverWait(driver, 5).until(lambda page: page.execute_script('return arguments[0].hintDelay >= 650', label))
        assert full in driver.execute_script("return getComputedStyle(arguments[0],'::after').content", label)
        assert not driver.find_elements(By.CSS_SELECTOR, '.ttip-visible')
    finally:
        driver.quit()
        server.shutdown()
        server.server_close()


def test_delayed_previous_capture_cannot_repaint_current_matrix(historical_published_viewer):
    started, release = threading.Event(), threading.Event()
    server, requests = _server(root=historical_published_viewer, gate=(started, release), gate_path=_published_capture_index(historical_published_viewer), external=True)
    driver = _driver()
    try:
        driver.get(f"http://127.0.0.1:{server.server_port}/?tab=tab-unified&transpose=0")
        assert started.wait(20)
        select = driver.find_element("css selector", "#tab-unified select[data-slot-capture='0']")
        assert select.is_displayed() and select.is_enabled()
        historical = _historical_dynamo_key(driver)
        Select(select).select_by_value(historical)
        _ready(driver, "tab-unified")
        assert driver.execute_script("return window.__conformanceModel.id") == "tab-unified"
        release.set()
        WebDriverWait(driver, 90).until(lambda page: page.execute_script("return window.__conformanceLoader.pending === 0"))
        assert driver.execute_script("return window.__conformanceController.selection(document.getElementById('tab-unified'))") == {"reference": None, "selected": [historical]}
        assert driver.find_element("css selector", ".tab-panel.active").get_attribute("id") == "tab-unified"
        assert not any("/toolcalling/" in path or "/reasoning/fixtures" in path for path in requests)
    finally:
        release.set()
        driver.quit()
        server.shutdown()


@pytest.mark.parametrize("kind", ["missing", "corrupt", "pointer"])
def test_resource_failures_remain_transport_errors_and_retry(kind, transport_published_viewer):
    default = _published_capture(transport_published_viewer)
    index = json.loads(gzip.decompress((transport_published_viewer / default["recipe_index"]["path"]).read_bytes()))
    target = index["resources"][default["path"]]["chunks"][0]["pack"]["path"]
    fault = {"used": False, "kind": kind, "path": target}
    server, _requests = _server(root=transport_published_viewer, fault=fault, external=True)
    driver = _driver()
    try:
        driver.get(f"http://127.0.0.1:{server.server_port}/?tab=tab-unified&transpose=0")
        error = WebDriverWait(driver, 30).until(EC.visibility_of_element_located(("css selector", "#tab-unified .load-status[role=alert]")))
        assert "Fixture load failed" in error.text
        assert not driver.find_elements("css selector", "#tab-unified[data-load-state=ready]")
        assert not driver.execute_script("return Boolean(window.__conformanceModel)")
        assert driver.find_element("css selector", "#tab-unified .capture-slots").is_displayed()
        assert driver.find_element("css selector", "#tab-unified [data-slot-open='0']").is_displayed()
        assert not driver.find_elements("css selector", "#tab-unified table[data-comparison-table]")
        error.find_element("tag name", "button").click()
        _ready(driver, "tab-unified")
        assert driver.execute_script("return Boolean(document.querySelector(arguments[0]))", "#tab-unified .matrix-cell")
    finally:
        driver.quit()
        server.shutdown()


def test_unified_controls_render_before_fixtures_and_detailed_view_stays_on_unified(transport_published_viewer):
    started, release = threading.Event(), threading.Event()
    server, requests = _server(root=transport_published_viewer, gate=(started, release), gate_path=_published_capture_index(transport_published_viewer), external=True)
    driver = _driver()
    try:
        driver.get(f"http://127.0.0.1:{server.server_port}/?tab=tab-reasoning-batch&transpose=0")
        assert started.wait(20)
        assert driver.current_url.startswith(f"http://127.0.0.1:{server.server_port}/?tab=tab-unified")
        assert len(driver.find_elements("css selector", "#tab-unified .capture-slot")) == 3
        assert len(driver.find_elements("css selector", "#tab-unified select[data-slot-capture]")) == 1
        assert [slot.get_attribute("data-engine") for slot in driver.find_elements(
            "css selector", "#tab-unified .capture-slot")] == ['dynamo', 'none', 'none']
        assert driver.find_element("css selector", "#tab-unified .capture-slots").is_displayed()
        assert driver.find_element("css selector", "#tab-unified [data-slot-open='0']").is_displayed()
        assert not driver.find_elements("css selector", "#tab-unified [data-comparison-table]")
        release.set()
        _ready(driver, "tab-unified")
        driver.find_element("css selector", "[data-view-detailed]").click()
        _ready(driver, "tab-unified")
        WebDriverWait(driver, 20).until(lambda page: page.execute_script("return window.__conformanceLoader.pending === 0"))
        assert driver.execute_script("return window.__conformanceModel.id") == "tab-unified"
        assert not any("/toolcalling/" in path or "/reasoning/fixtures" in path for path in requests)
        assert driver.find_element("css selector", "[data-view-detailed]").get_property("checked") is True
    finally:
        driver.quit()
        server.shutdown()


def test_first_capture_slot_keeps_horizontal_size_during_version_switch(historical_published_viewer):
    started, release = threading.Event(), threading.Event()
    server, _requests = _server(root=historical_published_viewer, gate=(started, release), gate_path=_published_capture_index(historical_published_viewer, "dynamo_v1", "5.1.3"))
    driver = _driver()
    driver.set_window_size(2048, 1000)
    try:
        driver.get(f"http://127.0.0.1:{server.server_port}/?tab=tab-unified")
        _ready(driver, "tab-unified")

        def first_slot_size():
            return driver.execute_script("""
              const rect = document.querySelector('#tab-unified .capture-slot[data-engine="dynamo"]').getBoundingClientRect();
              return {width: rect.width, height: rect.height};
            """)

        before = first_slot_size()
        capture_select = driver.find_element("css selector", '#tab-unified [data-slot-capture="0"]')
        Select(capture_select).select_by_value("dynamo_v1@5.1.3")
        assert started.wait(20), "the selected v1 5.1.3 capture request did not start"
        during = first_slot_size()
        assert during["width"] > 400 and during["height"] == before["height"], {"before": before, "during": during}

        release.set()
        _ready(driver, "tab-unified")
        after = first_slot_size()
        assert after["width"] == before["width"], {"before": before, "after": after}
    finally:
        release.set()
        driver.quit()
        server.shutdown()


def test_source_checkout_yaml_catalog_without_generated_snapshots(current_source_view):
    catalog = json.loads((current_source_view / "conformance/viewer-unified-catalog.json").read_bytes())
    assert catalog["sources"]["unified"] and all("format" not in resource for resource in catalog["sources"]["unified"])
    latest = max((item for item in catalog["sources"]["candidates"]
                  if item["implementation"] == "dynamo_v2" and item["mode"] == "unified"),
                 key=lambda item: unified_history._capture_release_sort_key(item["version"]))
    missing = ("/conformance/viewer-resolved-catalog.json.gz", "/conformance/viewer-resolved-catalog.json")
    server, requests = _server(root=current_source_view, missing_paths=missing)
    driver = _driver()
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {"source": """
        const NativeWorker = window.Worker;
        window.sourceCheckoutWorkerAudit = {native: /\\[native code\\]/.test(String(NativeWorker)), urls: [], yamlSent: 0, yamlReplies: 0, snapshotSent: 0};
        window.Worker = class extends NativeWorker {
            constructor(...args) {
                super(...args);
                this.sourceOperations = new Map();
                window.sourceCheckoutWorkerAudit.urls.push(String(args[0]));
                this.addEventListener('message', event => {
                    if (this.sourceOperations.get(event.data.id) === 'yaml' && !event.data.error) window.sourceCheckoutWorkerAudit.yamlReplies++;
                });
            }
            postMessage(message, ...rest) {
                this.sourceOperations.set(message.id, message.operation);
                if (message.operation === 'yaml') window.sourceCheckoutWorkerAudit.yamlSent++;
                if (message.operation === 'snapshot') window.sourceCheckoutWorkerAudit.snapshotSent++;
                return super.postMessage(message, ...rest);
            }
        };
    """})
    try:
        driver.get(f"http://127.0.0.1:{server.server_port}/?tab=tab-unified")
        _ready(driver, "tab-unified")
        assert driver.find_element("id", "viewer").get_attribute("data-catalog-revision") == catalog["revision"]
        assert driver.find_element("css selector", "#tab-unified [data-slot-capture='0']").get_attribute("value") == "dynamo"
        assert driver.execute_script("return window.__conformanceModel.rows.filter(row => !row.section).map(row => row.family)") == sorted(latest["native_unified_families"])
        assert driver.find_element("css selector", "#tab-unified table[data-comparison-table]").is_displayed()
        audit = driver.execute_script("return window.sourceCheckoutWorkerAudit")
        assert audit["native"] and any(url.endswith("/conformance/viewer/development_worker.js") for url in audit["urls"])
        assert audit["yamlSent"] > 0 and audit["yamlReplies"] == audit["yamlSent"] and audit["snapshotSent"] == 0
        assert all(path in requests for path in (missing[0], "/conformance/viewer-unified-catalog.json"))
        assert missing[1] not in requests, "source checkout fallback must not fetch a redundant plain catalog"
        assert any(path.startswith("/conformance/fixtures-unified-v2/") and path.endswith(".yaml") for path in requests)
        assert not any("/viewer-data/" in path or path.endswith("/viewer-startup.json") or "/bootstrap/" in path for path in requests)
        print(json.dumps({"source_checkout_yaml_worker": audit, "requests": len(requests), "passed": True}), flush=True)
    finally:
        driver.quit()
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("vertical", [False, True])
def test_selection_loading_and_failure_retain_results_and_popup(transport_published_viewer, vertical):
    started, release = threading.Event(), threading.Event()
    target = _published_capture_index(transport_published_viewer, "vllm_python")
    server, _requests = _server(root=transport_published_viewer, gate=(started, release), gate_path=target,
                              fault={"path": target, "kind": "missing", "used": False}, external=True)
    driver = _driver()
    driver.set_window_size(1280, 1000)
    try:
        driver.get(f"http://127.0.0.1:{server.server_port}/?tab=tab-unified&view=details&transpose={int(vertical)}")
        _ready(driver, "tab-unified")
        table_selector = "table[data-comparison-table]"
        cell = driver.find_element("css selector", f"#tab-unified {table_selector} .matrix-cell[data-ttip-id]")
        driver.execute_script("window.__buildTooltip(arguments[0]); arguments[0].dispatchEvent(new PointerEvent('pointerdown', {bubbles:true, pointerType:'touch'})); arguments[0].click();", cell)
        popup = WebDriverWait(driver, 10).until(EC.visibility_of_element_located(("css selector", ".ttip-pinned")))
        driver.execute_script("arguments[0].style.width='640px'; arguments[0].style.height='320px';", popup)
        popup_scroll_before = driver.execute_script("arguments[0].scrollTop=100; arguments[0].scrollLeft=40; return [arguments[0].scrollTop, arguments[0].scrollLeft]", popup)
        assert popup_scroll_before[0] > 0
        identity = driver.execute_script("const c=window.__conformanceCellModel(arguments[0]); return [c.case_id,c.family]", cell)
        _choose_engine(driver, 1, 'vllm')
        assert started.wait(20)
        scroll_before = None

        def assert_retained(state):
            panel = driver.find_element("id", "tab-unified")
            assert panel.get_attribute("data-load-state") == state
            if state == "loading":
                assert not driver.find_element(By.CSS_SELECTOR, "dialog.conformance-loading").is_displayed()
            assert driver.execute_script("return Boolean(window.__conformanceModel)")
            assert cell.is_displayed()
            assert popup.is_displayed()
            assert driver.execute_script("return [arguments[0].scrollTop, arguments[0].scrollLeft]", popup) == popup_scroll_before
            assert driver.find_element("css selector", f"#tab-unified {table_selector}").is_displayed()
            assert driver.find_element("css selector", "#tab-unified .capture-slots").is_displayed()
            assert driver.execute_script("""
                const cards=[...document.querySelectorAll('#tab-unified .capture-slot')].map(slot=>slot.getBoundingClientRect());
                const menu=document.querySelector('.comparison-toolbar').getBoundingClientRect();
                return cards.length===3 && cards.every(card=>card.width>0 && card.right<menu.left && Math.abs(card.top-menu.top)<=1)
                  && cards.slice(1).every((card,index)=>cards[index].right<card.left);
            """)
            assert driver.execute_script("return document.getElementById('tab-unified')._comparisonSelection.keys") == ['dynamo']
            assert driver.execute_script("return [...arguments[0].querySelectorAll('.matrix-status')].map(node => node.dataset.capture)", cell) == ['dynamo']
            if state == "error":
                assert driver.find_element("css selector", "#tab-unified .load-status[role=alert]").text
                assert not driver.find_element("css selector", "dialog.conformance-loading").is_displayed()
            assert driver.find_element("css selector", "[data-view-detailed]").get_property("checked") is True
            assert driver.execute_script("return document.body.classList.contains('view-details')")
            if scroll_before is not None:
                assert driver.execute_script("return [window.scrollX, window.scrollY]") == scroll_before

        assert_retained("loading")
        filter_key = driver.execute_script("""
            const cell = window.__conformanceCellModel(arguments[0]);
            const ownGroup = window.__conformanceModel.columns.find(column => column.sub === cell.sub).group_key;
            const button = [...document.querySelectorAll('[data-category-options] [data-group]')].find(node => node.dataset.group !== ownGroup);
            button.click();
            return button.dataset.group;
        """, cell)
        popup = driver.find_element("css selector", ".ttip-pinned")
        cell = driver.execute_script("return [...document.querySelectorAll('.matrix-cell')].find(cell => cell._ttip === arguments[0]);", popup)
        pressed = driver.find_element("css selector", f'[data-category-options] [data-group="{filter_key}"]').get_attribute("aria-pressed")
        scroll_before = driver.execute_script("window.scrollTo(120, 120); return [window.scrollX, window.scrollY]")
        assert any(value > 0 for value in scroll_before), "retention must be checked away from the scroll origin"
        assert_retained("loading")
        release.set()
        WebDriverWait(driver, 10).until(lambda page: page.find_element("id", "tab-unified").get_attribute("data-load-state") == "error")
        assert_retained("error")
        # Invoke the real Retry handler without WebDriver scrolling its target into view.
        driver.execute_script("arguments[0].click()", driver.find_element("css selector", "#tab-unified .load-status button"))
        _ready(driver, "tab-unified")
        restored = WebDriverWait(driver, 10).until(EC.visibility_of_element_located(("css selector", ".ttip-pinned")))
        restored_cell = driver.execute_script("return [...document.querySelectorAll('#tab-unified .matrix-cell')].find(cell => cell._ttip === arguments[0]);", restored)
        assert driver.execute_script("const c=window.__conformanceCellModel(arguments[0]); return [c.case_id,c.family]", restored_cell) == identity
        assert driver.execute_script("return [arguments[0].scrollTop, arguments[0].scrollLeft]", restored) == popup_scroll_before
        assert restored.value_of_css_property("width") == "640px"
        assert restored.value_of_css_property("height") == "320px"
        assert driver.execute_script("return window.__conformanceModel.id") == "tab-unified"
        assert driver.find_element("css selector", f'[data-category-options] [data-group="{filter_key}"]').get_attribute("aria-pressed") == pressed
        assert driver.execute_script("return document.body.classList.contains('transpose-mode')") == vertical
        assert driver.find_element("css selector", "[data-view-detailed]").get_property("checked") is True
        assert driver.execute_script("return document.body.classList.contains('view-details')")
        # A replacement can narrow the page; browser scroll clamps to its new bounds.
        expected_scroll = driver.execute_script("""
            const page = document.scrollingElement;
            return [Math.min(arguments[0][0], page.scrollWidth - page.clientWidth),
                    Math.min(arguments[0][1], page.scrollHeight - page.clientHeight)];
        """, scroll_before)
        assert driver.execute_script("return [window.scrollX, window.scrollY]") == expected_scroll
        assert len(driver.execute_script("return document.getElementById('tab-unified')._comparisonSelection.keys")) == 2
    finally:
        release.set()
        driver.quit()
        server.shutdown()


def test_history_upgrade_repaints_user_selection_before_old_build_finishes(historical_published_viewer):
    gate = (threading.Event(), threading.Event())
    server, _requests = _server(root=historical_published_viewer, gate=gate,
                               gate_path='/conformance/viewer-resolved-catalog.json.gz')
    driver = _driver()
    try:
        driver.get(f"http://127.0.0.1:{server.server_port}/?tab=tab-unified&keep=1")
        settle(driver, 'tab-unified')
        assert gate[0].wait(15), 'history catalog discovery did not start'
        driver.execute_script("""
            const loader = window.__conformanceLoader;
            const build = loader.build;
            let held = false;
            loader.build = async (...args) => {
                const model = await build(...args);
                if (!held) {
                    held = true;
                    await new Promise(resolve => { window.releasePreHistoryBuild = resolve; });
                }
                return model;
            };
        """)
        _choose_engine(driver, 1, 'vllm', programmatic=True)
        WebDriverWait(driver, 30).until(lambda page: page.execute_script(
            "return typeof window.releasePreHistoryBuild === 'function'"))
        selected = driver.execute_script("return window.__conformanceController.selection(document.getElementById('tab-unified')).selected")
        assert len(selected) == 2 and any(key.startswith('vllm_') for key in selected)
        gate[1].set()
        WebDriverWait(driver, 30).until(lambda page: page.execute_script("""
            const panel = document.getElementById('tab-unified');
            return document.getElementById('viewer').dataset.historyAvailable === 'true'
                && panel.dataset.loadState === 'ready'
                && JSON.stringify(panel._comparisonSelection.keys) === JSON.stringify(arguments[0]);
        """, selected))
        assert driver.execute_script("return window.__conformanceLoader.pending") == 1
        assert not driver.execute_script("return window.__conformanceError || document.querySelector('.load-status[role=alert], .error')")
        driver.execute_script("window.historyUpgradeModel = window.__conformanceModel; window.releasePreHistoryBuild();")
        settle(driver, 'tab-unified', history_ready=True)
        assert driver.execute_script("return window.__conformanceModel === window.historyUpgradeModel")
        assert driver.execute_script("return document.getElementById('tab-unified')._comparisonSelection.keys") == selected
        historical = _historical_dynamo_key(driver)
        Select(driver.find_element('css selector', "[data-slot-capture='0']")).select_by_value(historical)
        settle(driver, 'tab-unified', history_ready=True)
        assert driver.execute_script("return document.getElementById('tab-unified')._comparisonSelection.keys") == [historical, selected[1]]
    finally:
        gate[1].set()
        driver.quit()
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize('selection_parameter', ['slots_tab-unified', 'base_tab-unified', 'ref'])
def test_historical_legacy_slug_is_resolved_before_latest_controls_normalize(historical_published_viewer, selection_parameter):
    server, _requests = _server(root=historical_published_viewer)
    driver = _driver()
    try:
        query = urlencode({'tab': 'tab-unified', selection_parameter: 'dynamo_v2-0-6-1', 'keep': '1'})
        driver.get(f"http://127.0.0.1:{server.server_port}/?{query}#saved")
        settle(driver, 'tab-unified', history_ready=True)
        expected = ['dynamo@0.6.1']
        assert driver.execute_script("return document.getElementById('tab-unified')._comparisonSelection.keys") == expected
        assert driver.execute_script("return window.__conformanceController.selection(document.getElementById('tab-unified')).selected") == expected
        saved = driver.current_url
        assert 'keep=1' in saved and saved.endswith('#saved')
        driver.refresh()
        settle(driver, 'tab-unified', history_ready=True)
        assert driver.execute_script("return document.getElementById('tab-unified')._comparisonSelection.keys") == expected
        assert driver.current_url == saved
    finally:
        driver.quit()
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("stale_failure", [False, True])
def test_stale_selection_completion_keeps_latest_loading_notice(published_viewer, stale_failure):
    server, _requests = _server(root=published_viewer)
    driver = _driver()
    try:
        driver.get(f"http://127.0.0.1:{server.server_port}/?tab=tab-unified&view=details")
        _ready(driver, "tab-unified")
        cell = driver.find_element("css selector", "#tab-unified .matrix-cell[data-ttip-id]")
        driver.execute_script("""
            const loader = window.__conformanceLoader;
            const original = loader.build;
            const staleModel = window.__conformanceModel;
            const staleFailure = arguments[0];
            window.retainedModelBeforePending = staleModel;
            let intercepted = 0;
            loader.build = (...args) => {
                const index = intercepted++;
                if (index > 1) return original(...args);
                return new Promise((resolve, reject) => {
                    if (index === 0) {
                        window.releaseStaleComparison = () => staleFailure
                            ? reject(new Error('obsolete transport failure')) : resolve(staleModel);
                    } else {
                        window.releaseCurrentComparison = async () => {
                            loader.build = original;
                            try { resolve(await original(...args)); }
                            catch (error) { reject(error); }
                        };
                    }
                });
            };
        """, stale_failure)
        capture_select = _choose_engine(driver, 1, 'vllm')
        capture_select.send_keys(Keys.ESCAPE)
        rust = next(option.get_attribute('value') for option in capture_select.find_elements('tag name', 'option')
                    if option.get_attribute('value').startswith('vllm_rust@'))
        driver.execute_script("arguments[0].value=arguments[1]; arguments[0].dispatchEvent(new Event('change', {bubbles:true}));", capture_select, rust)
        notice = driver.find_element("css selector", "dialog.conformance-loading")
        latest_message = notice.text
        assert "Rust" in latest_message
        assert driver.execute_script("return window.__conformanceLoader.pending") == 2
        driver.execute_script("window.releaseStaleComparison()")
        WebDriverWait(driver, 10).until(lambda page: page.execute_script("return window.__conformanceLoader.pending === 1"))
        assert driver.find_element("id", "tab-unified").get_attribute("data-load-state") == "loading"
        assert driver.find_element("css selector", "dialog.conformance-loading") == notice
        assert notice.is_displayed()
        assert notice.text == latest_message
        assert cell.is_displayed()
        assert driver.execute_script("return window.__conformanceModel === window.retainedModelBeforePending")
        assert driver.execute_script("return document.getElementById('tab-unified')._comparisonSelection.keys") == ['dynamo']
        assert rust in driver.execute_script("return window.__conformanceController.selection(document.getElementById('tab-unified')).selected")
        driver.execute_script("window.releaseCurrentComparison()")
        _ready(driver, "tab-unified")
        WebDriverWait(driver, 10).until(lambda page: page.execute_script("return window.__conformanceLoader.pending === 0"))
        assert rust in driver.execute_script("return document.getElementById('tab-unified')._comparisonSelection.keys")
    finally:
        driver.quit()
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("stale_failure", [False, True])
def test_rapid_compare_selections_restore_url_and_ignore_late_same_tab_result(published_viewer, stale_failure):
    server, requests = _server(root=published_viewer)
    driver = _driver()
    try:
        driver.get(f"http://127.0.0.1:{server.server_port}/?tab=tab-unified&view=details&transpose=0&keep=1#saved")
        _ready(driver, "tab-unified")
        selected = driver.execute_script("""
            const panel = document.getElementById('tab-unified');
            const loader = window.__conformanceLoader;
            const original = loader.build;
            const staleModel = window.__conformanceModel;
            const staleFailure = arguments[0];
            loader.build = (...args) => {
                loader.build = original;
                return new Promise((resolve, reject) => { window.releaseOldComparison = () => staleFailure ? reject(new Error('stale failure')) : resolve(staleModel); });
            };
        """, stale_failure)
        vllm_select = _choose_engine(driver, 1, 'vllm')
        rust = next(option.get_attribute('value') for option in vllm_select.find_elements('tag name', 'option') if option.get_attribute('value').startswith('vllm_rust@'))
        driver.execute_script("arguments[0].value=arguments[1]; arguments[0].dispatchEvent(new Event('change', {bubbles:true}));", vllm_select, rust)
        _choose_engine(driver, 2, 'sglang', programmatic=True)
        selected = driver.execute_script("return window.__conformanceController.selection(document.getElementById('tab-unified')).selected")
        assert len(selected) == 3
        _ready(driver, "tab-unified")
        WebDriverWait(driver, 90).until(lambda page: page.execute_script("return typeof window.releaseOldComparison === 'function'"))
        driver.execute_script("window.releaseOldComparison()")
        WebDriverWait(driver, 20).until(lambda page: page.execute_script("return window.__conformanceLoader.pending === 0"))

        def assert_current_selection():
            assert driver.execute_script("return window.__conformanceModel.id") == "tab-unified"
            shown = driver.execute_script("return window.__conformanceController.selection(document.getElementById('tab-unified')).selected")
            assert all(key in shown for key in selected)
            missing = driver.execute_script("return window.__conformanceModel.rows.filter(row=>!row.section).flatMap(row=>Object.entries(row.cells).flatMap(([sub,cell])=>arguments[0].filter(key=>!Object.hasOwn(cell.cmp || {},key)).map(key=>[row.family,sub,key])))", selected)
            assert not missing, f"selected comparison keys missing from matrix cells: {missing[:12]}"
            cell = driver.find_element("css selector", "#tab-unified .matrix-cell[data-ttip-id]")
            driver.execute_script("window.__buildTooltip(arguments[0]);", cell)
            rendered = driver.execute_script("return [...arguments[0].querySelectorAll('.ttip [data-cand]')].filter(node=>!node.classList.contains('col-hidden')).map(node=>node.dataset.cand)", cell)
            assert all(key in rendered for key in selected)

        assert_current_selection()
        saved = driver.current_url
        assert "keep=1" in saved and saved.endswith("#saved")
        assert "slots_tab-unified=" in saved
        assert not any("/toolcalling/" in path or "/reasoning/fixtures" in path for path in requests)
        driver.refresh()
        _ready(driver, "tab-unified")
        assert_current_selection()
        assert driver.current_url == saved
    finally:
        driver.quit()
        server.shutdown()


@pytest.mark.parametrize("tab,transpose,selector", [
    ("tab-unified", False, "th.matrix-case-heading[data-ttip-id]"),
    ("tab-unified", True, "th.matrix-case-heading[data-ttip-id]"),
])
def test_grammar_header_pin_survives_asynchronous_model_refresh(tab, transpose, selector, published_viewer):
    server, _requests = _server(root=published_viewer)
    driver = _driver()
    try:
        driver.get(f"http://127.0.0.1:{server.server_port}/?tab={tab}&view=details&transpose={int(transpose)}")
        _ready(driver, tab)
        popup_key = driver.execute_script("""
            const panel = document.getElementById(arguments[0]);
            const headers = [...panel.querySelectorAll(arguments[1])];
            const header = headers.find(node => node.offsetWidth > 20 && node.offsetHeight > 10);
            if (!header) throw new Error('No interactable grammar header');
            header.scrollIntoView();
            window.__buildTooltip(header);
            header.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, pointerType: 'touch' }));
            header.click();
            return header.dataset.popupKey;
        """, tab, selector)
        popup = WebDriverWait(driver, 10).until(EC.visibility_of_element_located(("css selector", ".ttip-pinned")))
        assert popup_key and popup.text
        popup_scroll_before = driver.execute_script("arguments[0].style.width='640px'; arguments[0].style.height='240px'; arguments[0].scrollTop=100; arguments[0].scrollLeft=40; return [arguments[0].scrollTop, arguments[0].scrollLeft]", popup)
        assert popup_scroll_before[0] > 0
        scroll_before = driver.execute_script("return [window.scrollX, window.scrollY]")
        driver.execute_script("const panel=document.getElementById(arguments[0]); window.__conformanceLoader.select(panel.id, window.__conformanceController.selection(panel));", tab)
        _ready(driver, tab)
        WebDriverWait(driver, 20).until(lambda page: page.execute_script("return window.__conformanceLoader.pending === 0"))
        restored = WebDriverWait(driver, 10).until(EC.visibility_of_element_located(("css selector", ".ttip-pinned")))
        assert driver.execute_script("return arguments[0]._ttipOwner.dataset.popupKey", restored) == popup_key
        assert driver.execute_script("return [window.scrollX, window.scrollY]") == scroll_before
        assert driver.execute_script("return [arguments[0].scrollTop, arguments[0].scrollLeft]", restored) == popup_scroll_before
        restored.find_element("css selector", ".ttip-close").click()
        assert not driver.find_elements("css selector", ".ttip-pinned")
    finally:
        driver.quit()
        server.shutdown()


def test_reset_discards_popup_pin_during_retained_selection_load(published_viewer):
    server, _requests = _server(root=published_viewer)
    driver = _driver()
    try:
        driver.get(f"http://127.0.0.1:{server.server_port}/?tab=tab-unified&view=details&transpose=0")
        _ready(driver, "tab-unified")
        cell = driver.find_element("css selector", "#tab-unified .matrix-cell[data-ttip-id]")
        driver.execute_script("window.__buildTooltip(arguments[0]); arguments[0].dispatchEvent(new PointerEvent('pointerdown', {bubbles:true, pointerType:'touch'})); arguments[0].click();", cell)
        popup = WebDriverWait(driver, 10).until(EC.visibility_of_element_located(("css selector", ".ttip-pinned")))
        driver.execute_script("arguments[0].style.width='640px'; arguments[0].style.height='320px';", popup)
        driver.execute_script("""
            const loader = window.__conformanceLoader;
            const original = loader.build;
            loader.build = (...args) => {
                loader.build = original;
                return new Promise((resolve, reject) => {
                    original(...args).then(model => { window.releaseNavigationSelection = () => resolve(model); }, reject);
                });
            };
            const panel = document.getElementById('tab-unified');
            loader.select(panel.id, window.__conformanceController.selection(panel));
        """)
        WebDriverWait(driver, 90).until(lambda page: page.execute_script("return typeof window.releaseNavigationSelection === 'function'"))
        assert driver.find_element("id", "tab-unified").get_attribute("data-load-state") == "loading"
        assert popup.is_displayed()
        driver.execute_script("arguments[0].click()", driver.find_element("css selector", "[data-reset]"))
        _ready(driver, "tab-unified")
        selection = driver.execute_script("return window.__conformanceController.selection(document.getElementById('tab-unified'))")
        assert selection == {'reference': None, 'selected': ['dynamo']}
        assert not driver.find_elements("css selector", ".ttip-pinned")
        assert driver.execute_script("return window.__conformanceModel.id") == "tab-unified"
    finally:
        driver.quit()
        server.shutdown()


def test_original_archive_view_and_blob_download_preserve_source_bytes(historical_published_viewer, history_source):
    member = 'toolcalling/fixtures-stream-v1/inputs/deepseek_v4/TOOLCALLING.streamv1.1.yaml'
    _assert_original_yaml_download(historical_published_viewer, member, historical_source_bytes(history_source, member))


def test_original_yaml_view_and_blob_download_preserve_source_bytes(published_viewer):
    direct = 'conformance/fixtures-unified-v2/families/qwen3/inputs_and_golden.yaml'
    _assert_original_yaml_download(published_viewer, direct, (ROOT / direct).read_bytes())


def _assert_original_yaml_download(published_viewer, path, expected):
    digest = hashlib.sha256(expected).hexdigest()
    server, _requests = _server(root=published_viewer)
    driver = _driver()
    try:
        query = urlencode({'tab': 'tab-unified', 'raw': path, 'keep': 'original-yaml'})
        driver.get(f'http://127.0.0.1:{server.server_port}/?{query}#saved')
        WebDriverWait(driver, 30).until(lambda page: page.find_elements('css selector', '#viewer > pre') or page.find_elements('css selector', '.error'))
        assert not driver.find_elements('css selector', '.error')
        actual = raw_view_download(driver)
        assert actual['hash'] == actual['renderedHash'] == digest
        assert actual['size'] == len(expected)
        assert actual['filename'] == path.split('/')[-1]
        back = urlsplit(actual['back'])
        assert dict(parse_qsl(back.query)) == {'tab': 'tab-unified', 'keep': 'original-yaml'}
        assert back.fragment == 'saved'
    finally:
        driver.quit()
        server.shutdown()


def test_legacy_capture_alias_and_reset_preserve_unrelated_url_state(historical_published_viewer):
    server, _requests = _server(root=historical_published_viewer)
    driver = _driver()
    try:
        current = _published_capture(historical_published_viewer)['version']
        driver.get(f'http://127.0.0.1:{server.server_port}/?tab=tab-unified&ref=dynamo_v2%40{current}&cmp=dynamo_v2%400.6.1&keep=1&transpose=1#saved')
        _ready(driver, 'tab-unified')
        assert driver.execute_script("return window.__conformanceController.selection(document.getElementById('tab-unified'))") == {'reference': None, 'selected': ['dynamo', 'dynamo@0.6.1']}
        saved = urlsplit(driver.current_url)
        query = dict(parse_qsl(saved.query, keep_blank_values=True))
        slots = query['slots_tab-unified'].split(',')
        assert slots == [f"dynamo_v2@{_published_capture(historical_published_viewer)['version']}", 'dynamo_v2@0.6.1', 'none']
        assert 'view' not in query and query['transpose'] == '1' and query['keep'] == '1'
        assert 'base_tab-unified' not in query and 'cmp_tab-unified' not in query
        assert 'ref' not in query and 'cmp' not in query and saved.fragment == 'saved'
        driver.find_element('css selector', '[data-reset]').click()
        _ready(driver, 'tab-unified')
        reset = urlsplit(driver.current_url)
        assert dict(parse_qsl(reset.query)) == {'tab': 'tab-unified', 'keep': '1'} and reset.fragment == 'saved'
        assert driver.execute_script("return window.__conformanceController.selection(document.getElementById('tab-unified'))") == {'reference': None, 'selected': ['dynamo']}
        assert driver.execute_script("return document.body.classList.contains('transpose-mode')")
    finally:
        driver.quit()
        server.shutdown()


@pytest.mark.parametrize('columns', ['model', 'batch_core'])
def test_legacy_column_selection_survives_async_mount_and_reload(columns, published_viewer):
    server, _requests = _server(root=published_viewer)
    driver = _driver()
    try:
        driver.get(f'http://127.0.0.1:{server.server_port}/?tab=tab-unified&base_tab-unified=&base_tab-reasoning-batch=&cols={columns}&transpose=0')
        _ready(driver, 'tab-unified')
        buttons = driver.find_elements('css selector', '[data-category-options] [data-group]')
        assert buttons and all(button.get_attribute('aria-pressed') == 'true' for button in buttons)
        assert 'cols' not in dict(parse_qsl(urlsplit(driver.current_url).query))
        driver.refresh()
        _ready(driver, 'tab-unified')
    finally:
        driver.quit()
        server.shutdown()




@pytest.mark.parametrize('columns', ['', 'obsolete_unknown'])
def test_empty_or_unknown_legacy_columns_keep_defaults(columns, published_viewer):
    server, _requests = _server(root=published_viewer)
    driver = _driver()
    try:
        driver.get(f'http://127.0.0.1:{server.server_port}/?tab=tab-unified&base_tab-unified=&cols={columns}&transpose=0')
        _ready(driver, 'tab-unified')
        buttons = driver.find_elements('css selector', '[data-category-options] [data-group]')
        assert buttons and all(button.get_attribute('aria-pressed') == 'true' for button in buttons)
    finally:
        driver.quit()
        server.shutdown()


def test_disabling_all_categories_restores_empty_matrix_on_reload(published_viewer):
    server, _ = _server(root=published_viewer)
    driver = _driver()
    try:
        driver.get(f'http://127.0.0.1:{server.server_port}/')
        _ready(driver, 'tab-unified')
        driver.find_element('css selector', '.comparison-categories summary').click()
        for button in driver.find_elements('css selector', '[data-category-options] [data-group]'):
            button.click()
        assert driver.execute_script('return window.__conformanceComparison.counts.dynamo.total') == 0
        driver.refresh()
        _ready(driver, 'tab-unified')
        assert not driver.find_elements('css selector', '.matrix-cell')
        buttons = driver.find_elements('css selector', '[data-category-options] [data-group]')
        assert buttons and all(button.get_attribute('aria-pressed') == 'false' for button in buttons)
        driver.find_element('css selector', '.comparison-categories summary').click()
        buttons[0].click()
        assert driver.execute_script('return window.__conformanceComparison.counts.dynamo.total') > 0
    finally:
        driver.quit()
        server.shutdown()
        server.server_close()


def test_unified_slots_migrate_legacy_comparison_url_and_fit_mobile(historical_published_viewer):
    server, _requests = _server(root=historical_published_viewer)
    driver = _driver()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        driver.set_window_size(390, 900)
        driver.get(f"{base}/?tab=tab-reasoning-batch&ref=sglang_python%400.5.21&cmp=dynamo,vllm_python%400.31.0&keep=1#saved")
        _ready(driver, 'tab-unified')
        assert driver.current_url.startswith(base + '/?tab=tab-unified')
        state = driver.execute_script("return window.__conformanceController.selection(document.getElementById('tab-unified'))")
        assert state == {'reference': None, 'selected': ['sglang_python@0.5.21', 'dynamo', 'vllm_python@0.31.0']}
        slots = driver.find_elements('css selector', '#tab-unified .capture-slot')
        assert [slot.get_attribute('data-engine') for slot in slots] == ['sglang', 'dynamo', 'vllm']
        boxes = [slot.rect for slot in slots]
        assert all(box['x'] >= 0 and box['x'] + box['width'] <= 390 for box in boxes)
        assert all(driver.execute_script("return arguments[0].scrollHeight <= arguments[0].clientHeight", label)
                   for label in driver.find_elements('css selector', '.capture-slot-selected-label'))
        assert 'keep=1' in driver.current_url and driver.current_url.endswith('#saved')
        params = dict(parse_qsl(urlsplit(driver.current_url).query))
        assert params['slots_tab-unified'].split(',')[0] == 'sglang_python@0.5.21'
        assert 'ref' not in params and 'cmp' not in params
        _clear_slot(driver, 2)
        assert driver.execute_script("return window.__conformanceController.selection(document.getElementById('tab-unified'))") == {'reference': None, 'selected': ['sglang_python@0.5.21', 'dynamo']}
        assert driver.find_element('css selector', "#tab-unified .capture-slot[data-engine='none']")
        driver.find_element('css selector', '[data-reset]').click()
        _ready(driver, 'tab-unified')
        params = dict(parse_qsl(urlsplit(driver.current_url).query))
        assert params == {'tab': 'tab-unified', 'keep': '1'}
        assert urlsplit(driver.current_url).fragment == 'saved'
        assert driver.execute_script("return window.__conformanceController.selection(document.getElementById('tab-unified'))") == {'reference': None, 'selected': ['dynamo']}
    finally:
        driver.quit()
        server.shutdown()


def test_unified_slots_migrate_tab_scoped_legacy_url_after_tab_canonicalization(historical_published_viewer):
    server, _requests = _server(root=historical_published_viewer)
    driver = _driver()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        driver.get(f"{base}/?tab=tab-reasoning-batch&base_tab-reasoning-batch=sglang_python%400.5.21&cmp_tab-reasoning-batch=dynamo,vllm_python%400.31.0&keep=1#saved")
        _ready(driver, 'tab-unified')
        assert driver.execute_script("return window.__conformanceController.selection(document.getElementById('tab-unified'))") == {
            'reference': None, 'selected': ['sglang_python@0.5.21', 'dynamo', 'vllm_python@0.31.0']
        }
        params = dict(parse_qsl(urlsplit(driver.current_url).query))
        assert params['tab'] == 'tab-unified'
        latest = driver.execute_script("return window.__CONFORMANCE_PAGE.tabs[0].candidates.find(item => item.key === 'dynamo').identity")
        assert params['slots_tab-unified'].split(',') == ['sglang_python@0.5.21', latest, 'vllm_python@0.31.0']
        assert params['keep'] == '1'
        assert not any(key.startswith(('base_', 'cmp_')) for key in params)
        assert urlsplit(driver.current_url).fragment == 'saved'
    finally:
        driver.quit()
        server.shutdown()


def test_unified_capture_slots_survive_reload_with_url_state(published_viewer):
    server, _requests = _server(root=published_viewer)
    driver = _driver()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        driver.get(f"{base}/?tab=tab-unified&slots_tab-unified=dynamo_v2%400.7.24,none,vllm_python%400.31.0&keep=1#saved")
        _ready(driver, 'tab-unified')
        expected = {'reference': None, 'selected': ['dynamo', 'vllm_python@0.31.0']}
        assert driver.execute_script("return window.__conformanceController.selection(document.getElementById('tab-unified'))") == expected
        assert dict(parse_qsl(urlsplit(driver.current_url).query))['slots_tab-unified'] == f"dynamo_v2@{_published_capture(published_viewer)['version']},none,vllm_python@0.31.0"
        driver.refresh()
        _ready(driver, 'tab-unified')
        assert driver.execute_script("return window.__conformanceController.selection(document.getElementById('tab-unified'))") == expected
        params = dict(parse_qsl(urlsplit(driver.current_url).query))
        assert params['slots_tab-unified'] == f"dynamo_v2@{_published_capture(published_viewer)['version']},none,vllm_python@0.31.0"
        assert params['keep'] == '1'
        assert urlsplit(driver.current_url).fragment == 'saved'
    finally:
        driver.quit()
        server.shutdown()


def test_unavailable_jail_capture_deep_link_is_cleared(published_viewer):
    server, _requests = _server(root=published_viewer)
    driver = _driver()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        driver.get(f"{base}/?tab=tab-unified&slots_tab-unified=dynamo_v2%400.7.23,dynamo_v1_jail%400.9.1,none&keep=1#saved")
        _ready(driver, "tab-unified")
        expected = {"reference": None, "selected": ["dynamo"]}
        assert driver.execute_script("return window.__conformanceController.selection(document.getElementById('tab-unified'))") == expected
        params = dict(parse_qsl(urlsplit(driver.current_url).query))
        assert "slots_tab-unified" not in params
        assert params["keep"] == "1"
        assert urlsplit(driver.current_url).fragment == "saved"
        assert driver.find_element("css selector", "#tab-unified .capture-slot[data-engine='dynamo']").is_displayed()
    finally:
        driver.quit()
        server.shutdown()


@pytest.mark.parametrize("stale_capture", ["dynamo_v1@9.2.10", "dynamo_v1@5.0.1"])
def test_primary_v1_deep_link_preserves_available_capture(stale_capture, historical_published_viewer):
    producer, version = stale_capture.split("@")
    major = version.split(".")[0]
    catalog = _published_catalog(historical_published_viewer)
    newest = max((row["version"] for row in catalog["sources"]["candidates"]
                  if row["implementation"] == producer and row["version"].split(".")[0] == major
                  and row.get("capture_families") and not row.get("source_blocker")
                  and row.get("has_captured_results") is not False),
                 key=unified_history._capture_release_sort_key)
    available = any(row["implementation"] == producer and row["version"] == version
                    and row.get("capture_families") and not row.get("source_blocker")
                    and row.get("has_captured_results") is not False for row in catalog["sources"]["candidates"])
    expected_capture = stale_capture if available else f"{producer}@{newest}"
    server, _requests = _server(root=historical_published_viewer)
    driver = _driver()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        slots = urlencode({"tab": "tab-unified", "slots_tab-unified": f"{stale_capture},none,none", "keep": "1"})
        driver.get(f"{base}/?{slots}#saved")
        _ready(driver, "tab-unified")
        expected = {"reference": None, "selected": [expected_capture]}
        assert driver.execute_script("return window.__conformanceController.selection(document.getElementById('tab-unified'))") == expected
        params = dict(parse_qsl(urlsplit(driver.current_url).query))
        assert params["slots_tab-unified"] == f"{expected_capture},none,none"
        assert params["keep"] == "1"
        assert urlsplit(driver.current_url).fragment == "saved"
    finally:
        driver.quit()
        server.shutdown()


def test_detailed_keeps_legend_below_matrix_and_case_descriptions_last(published_viewer):
    server, _ = _server(root=published_viewer)
    driver = _driver()
    try:
        driver.get(f'http://127.0.0.1:{server.server_port}/')
        _ready(driver, 'tab-unified')
        descriptions = driver.find_element('css selector', '.case-descriptions')
        assert not descriptions.is_displayed()
        driver.find_element('css selector', '[data-view-detailed]').click()
        assert descriptions.is_displayed()
        assert not driver.find_elements('css selector', '.capture-legend, .legacy-report-details, .generation-instructions')
        for vertical in (False, True):
            toggle = driver.find_element('css selector', '[data-transpose-toggle]')
            if toggle.is_selected() != vertical:
                toggle.click()
            assert driver.execute_script("""
                const panel=document.getElementById('tab-unified'), table=panel.querySelector('[data-comparison-table]');
                const legend=panel.querySelector('.comparison-legend'), descriptions=panel.querySelector('.case-descriptions');
                return legend.getBoundingClientRect().top >= table.getBoundingClientRect().bottom
                  && descriptions.getBoundingClientRect().top >= legend.getBoundingClientRect().bottom
                  && descriptions===panel.lastElementChild;
            """)
            first = driver.find_element('css selector', '.matrix-case-heading')
            assert first.get_attribute('aria-label') in descriptions.text
    finally:
        driver.quit()
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("vertical", [False, True])
def test_popup_heading_uses_case_id_before_description(published_viewer, vertical):
    server, _requests = _server(root=published_viewer)
    driver = _driver()
    driver.set_window_size(1500, 1000)
    try:
        driver.get(f"http://127.0.0.1:{server.server_port}/?tab=tab-unified&transpose={int(vertical)}")
        _ready(driver, "tab-unified")
        table = "table[data-comparison-table]"
        header = "th.matrix-case-heading[data-ttip-id]"
        tool_header = driver.find_element(By.CSS_SELECTOR, f"#tab-unified {header}[aria-label='UNIFIED.1-tool_only']")
        number = driver.execute_script("const c=window.__conformanceTooltipModel(arguments[0]).popupCase; return c.compact_prefix+'-'+c.display_number", tool_header)
        tool_cell = driver.execute_script("""
            return [...document.querySelectorAll(arguments[0])].find(node =>
                window.__conformanceCellModel(node)?.sub === 'tool_only' && node.getBoundingClientRect().height);
        """, f"#tab-unified {table} .matrix-cell[data-ttip-id]")
        assert tool_cell is not None
        for cell in [tool_header, tool_cell]:
            driver.execute_script("arguments[0].scrollIntoView({block:'center', inline:'end'})", cell)
            cell.click()
            popup = WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.CSS_SELECTOR, ".ttip-pinned")))
            heading = popup.find_element(By.CSS_SELECTOR, ".ttip-head")
            assert heading.text.startswith(f"1-tool_only ({number})"), heading.text
            assert "Single tool call" in heading.text, heading.text
            popup.find_element(By.CSS_SELECTOR, ".ttip-close").click()
    finally:
        driver.quit()
        server.shutdown()
        server.server_close()


def test_unified_slots_migrate_legacy_stream_candidate_keys(published_viewer):
    server, _requests = _server(root=published_viewer)
    driver = _driver()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        driver.get(f"{base}/?tab=tab-toolcalling-streamv1&base_tab-toolcalling-streamv1=dynamo_v2-0-7-22&cmp_tab-toolcalling-streamv1=vllm_python-0-31-0&keep=1#saved")
        _ready(driver, 'tab-unified')
        assert driver.execute_script("return window.__conformanceController.selection(document.getElementById('tab-unified'))") == {
            'reference': None, 'selected': ['dynamo', 'vllm_python@0.31.0']
        }
        params = dict(parse_qsl(urlsplit(driver.current_url).query))
        assert params['tab'] == 'tab-unified'
        assert params['slots_tab-unified'] == f"dynamo_v2@{_published_capture(published_viewer)['version']},none,vllm_python@0.31.0"
        assert params['keep'] == '1'
        assert not any(key.startswith(('base_', 'cmp_')) for key in params)
        assert urlsplit(driver.current_url).fragment == 'saved'
    finally:
        driver.quit()
        server.shutdown()


@pytest.mark.parametrize("identities", [
    ["dynamo_v1@10.0.1"],
    ["dynamo_v2@0.7.24", "dynamo_v1@10.2.2"],
    ["sglang_python@0.5.14"],
    ["dynamo_v1@9.2.13"],
])
def test_selected_capture_family_union_and_box_spacing(identities, historical_published_viewer):
    catalog = _published_catalog(historical_published_viewer)
    candidates = {f"{item['implementation']}@{item['version']}": item for item in catalog["sources"]["candidates"]}
    server, requests = _server(root=historical_published_viewer)
    driver = _driver()
    try:
        query = urlencode({"tab": "tab-unified", "slots_tab-unified": ",".join(identities + ["none"] * (3 - len(identities))), "transpose": "0"})
        driver.get(f"http://127.0.0.1:{server.server_port}/?{query}")
        _ready(driver, "tab-unified")
        effective = driver.execute_script("""
          const panel = document.getElementById('tab-unified');
          const candidates = window.__CONFORMANCE_PAGE.tabs.find(tab => tab.id === panel.id).candidates;
          return window.__conformanceController.selection(panel).selected.map(key =>
            candidates.find(candidate => candidate.key === key).identity);
        """)
        assert len(effective) == len(identities)
        for requested, selected in zip(identities, effective):
            producer, version = requested.split("@")
            actual_producer, actual_version = selected.split("@")
            assert actual_producer == producer
            parts = 2 if producer == "dynamo_v2" else 1
            assert actual_version.split(".")[:parts] == version.split(".")[:parts]
        current = _published_capture(historical_published_viewer)
        native = set(candidates[f"dynamo_v2@{current['version']}"]["native_unified_families"])
        scopes = [set(candidates[identity]["capture_families"]) & native
                  if identity.startswith("dynamo_v2@") else set(candidates[identity]["capture_families"])
                  for identity in effective]
        expected = sorted(set().union(*scopes))
        rows = driver.execute_script("return window.__conformanceModel.rows.map(row => row.family)")
        assert rows == expected
        assert len(rows) == len(set(rows))
        if candidates[identities[0]].get("source_blocker"):
            selected_option = driver.find_element("css selector", "[data-slot-capture='0'] option:checked")
            assert effective[0] != identities[0]
            assert selected_option.get_attribute("value") == effective[0]
            assert selected_option.is_enabled()
            assert not candidates[effective[0]].get("source_blocker")
        assert not driver.find_elements("css selector", ".engine-card, .toolbar-desc")
        for transpose in [False, True]:
            if transpose:
                driver.find_element("css selector", "[data-transpose-toggle]").click()
            for theme in ["light", "dark"]:
                driver.execute_script("document.documentElement.setAttribute('data-theme', arguments[0])", theme)
                geometry = driver.execute_script("""
                  const table = document.querySelector(arguments[0]);
                  const boxes = [...table.querySelectorAll('.matrix-cell .matrix-segments')].filter(box => box.getBoundingClientRect().width);
                  const first = boxes[0], rect = first.getBoundingClientRect();
                  const next = boxes.find(box => { const r = box.getBoundingClientRect(); return Math.abs(r.top - rect.top) < 1 && r.left > rect.right; });
                  const statuses = new Map([...table.querySelectorAll('.matrix-status')]
                    .filter(node => node.getBoundingClientRect().width)
                    .map(node => [node.className.split('status-')[1], node]));
                  const styles = [...statuses].map(([status, node]) => {
                    const legend = document.querySelector('.comparison-legend [data-status="' + status + '"] .legend-swatch');
                    const style = getComputedStyle(node), example = legend && getComputedStyle(legend);
                    return {status, matches: !!example && style.backgroundColor === example.backgroundColor
                      && style.backgroundImage === example.backgroundImage};
                  });
                  return {gap: next.getBoundingClientRect().left - rect.right,
                    width: rect.width, radius: getComputedStyle(first).borderRadius,
                    segments: first.children.length, styles, knownStatuses: Object.keys(window.__conformanceSegmentLabels)};
                """, "#tab-unified table[data-comparison-table]")
                assert geometry["gap"] >= 3, geometry
                assert geometry["width"] == 27
                assert geometry["radius"] == "4px"
                assert geometry["segments"] == len(identities)
                assert geometry["styles"]
                assert {item["status"] for item in geometry["styles"]} <= set(geometry["knownStatuses"])
                assert all(item["matches"] for item in geometry["styles"])
        if len(identities) == 2:
            difference = scopes[0] ^ scopes[1]
            assert difference
            family = sorted(difference)[0]
            missing = effective[0] if family not in scopes[0] else effective[1]
            assert driver.execute_script("""
              const row = window.__conformanceModel.rows.find(row => row.family === arguments[0]);
              const tab = window.__CONFORMANCE_PAGE.tabs.find(tab => tab.id === 'tab-unified');
              const key = tab.candidates.find(candidate => candidate.identity === arguments[1]).key;
              return Object.values(row.cells).some(cell => cell.cmp?.[key]?.na === 1);
            """, family, missing)
    finally:
        driver.quit()
        server.shutdown()
        server.server_close()


def test_checkout_case_hints_fit_both_orientations_and_themes(current_source_view):
    server, _ = _server(root=current_source_view, missing_paths=('/conformance/viewer-publication.json',))
    driver = _driver()
    try:
        driver.set_window_size(1500, 1100)
        driver.get(f'http://127.0.0.1:{server.server_port}/?vertical=0')
        _ready(driver, 'tab-unified')
        assert not driver.find_element('css selector', '.generated').is_displayed()
        for theme in ('light', 'dark'):
            driver.execute_script('document.documentElement.dataset.theme=arguments[0]', theme)
            for vertical in (False, True):
                toggle = driver.find_element('css selector', '[data-transpose-toggle]')
                if toggle.is_selected() != vertical:
                    toggle.click()
                for case_id in ('UNIFIED.1-tool_only', 'UNIFIED.2-two_calls'):
                    header = driver.find_element('css selector', f'.matrix-case-heading[aria-label="{case_id}"]')
                    label = header.find_element('css selector', '[data-hover-hint]')
                    driver.execute_script('arguments[0].scrollIntoView({block:"center",inline:"center"})', header)
                    ActionChains(driver).move_to_element(label).perform()
                    WebDriverWait(driver, 5).until(lambda page: page.execute_script("return getComputedStyle(arguments[0],'::after').opacity==='1'", label))
                    assert driver.execute_script("""
                        const r=arguments[0].getBoundingClientRect(), css=getComputedStyle(arguments[0],'::after');
                        const x=r.left+parseFloat(css.left);
                        return css.content.includes(arguments[0].getAttribute('data-hover-hint'))
                          && x>=0 && x+parseFloat(css.width)<=document.documentElement.clientWidth;
                    """, label)
                    assert not driver.find_elements('css selector', '.ttip-visible')
    finally:
        driver.quit()
        server.shutdown()
        server.server_close()


def test_missing_sidecar_keeps_source_identity_without_inventing_generation(published_viewer):
    server, _requests = _server(root=published_viewer, missing_paths=("/conformance/viewer-publication.json",))
    driver = _driver()
    try:
        driver.get(f"http://127.0.0.1:{server.server_port}/?tab=tab-unified")
        _ready(driver, "tab-unified")
        metadata = publication_metadata(published_viewer)
        generated = driver.find_element(By.CSS_SELECTOR, ".generated")
        assert not generated.is_displayed()
        assert generated.get_attribute("textContent") == ""
        driver.find_element(By.CSS_SELECTOR, "[data-view-detailed]").click()
        assert not driver.find_elements(By.CSS_SELECTOR, ".generated-command")
        assert not driver.find_elements(By.CSS_SELECTOR, ".generation-instructions")
        driver.find_element(By.CSS_SELECTOR, ".matrix-cell[data-ttip-id]").click()
        source = driver.find_element(By.CSS_SELECTOR, "a[href*='github.com/ai-dynamo/frontend-crates/blob/']")
        assert f"/{metadata['revision']}/" in source.get_attribute('href')
    finally:
        driver.quit()
        server.shutdown()
        server.server_close()


def _popup_for_scenario(driver, family, scenario):
    cell = driver.execute_script("""
        return [...document.querySelectorAll('#tab-unified .matrix-cell[data-ttip-id]')].find(node => {
            const model = window.__conformanceCellModel(node);
            return model?.family === arguments[0] && model.sub === arguments[1]
                && node.getBoundingClientRect().height > 0;
        });
    """, family, scenario)
    assert cell is not None
    driver.execute_script('arguments[0].scrollIntoView({block:"center",inline:"center"})', cell)
    cell.click()
    return WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.CSS_SELECTOR, '.ttip-pinned')))


def test_structural_response_popup_shows_generation_constraints(published_viewer):
    server, _requests = _server(root=published_viewer)
    driver = _driver()
    try:
        driver.get(f"http://127.0.0.1:{server.server_port}/?tab=tab-unified")
        _ready(driver, "tab-unified")
        popup = _popup_for_scenario(driver, 'gpt_oss', 'structured_response_after_reasoning')
        config = next(node for node in popup.find_elements(By.CSS_SELECTOR, '.ttip-config code')
                      if node.get_attribute('textContent').startswith('structural_tag='))
        request = json.loads(config.get_attribute('textContent').removeprefix('structural_tag='))
        assert request['schema']['additionalProperties'] is False
        assert request['schema']['required'] == ['location', 'temperature', 'conditions']
        assert len(request['rejected']) == 6
        assert '[object Object]' not in config.text
        assert 'XGrammar' in popup.text
        assert not popup.find_elements(By.CSS_SELECTOR, 'script')
        assert driver.execute_script('return getComputedStyle(arguments[0]).whiteSpace', config) == 'pre-wrap'
        popup.find_element(By.CSS_SELECTOR, '.ttip-close').click()
    finally:
        driver.quit()
        server.shutdown()


def test_muse_spark_popup_preserves_message_phases(published_viewer):
    _assert_muse_spark_message_phases(published_viewer)


def test_historical_muse_spark_popup_preserves_message_phases(historical_published_viewer):
    _assert_muse_spark_message_phases(historical_published_viewer)


def _assert_muse_spark_message_phases(site):
    server, _requests = _server(root=site)
    driver = _driver()
    try:
        entry = "conformance/CONFORMANCE_v2.html" if (site / "conformance/CONFORMANCE_v2.html").is_file() else "index.html"
        driver.get(f"http://127.0.0.1:{server.server_port}/{entry}?tab=tab-unified")
        _ready(driver, "tab-unified")
        commentary = "message_start phase=commentary"
        for scenario, expected in [
            ("spark_phase_tool_final", [commentary, "message_end", "message_start phase=final_answer", "message_end"]),
            ("spark_phase_repeated", [commentary, "message_end", commentary, "message_end"]),
            ("spark_phase_empty", [commentary, "message_end"]),
            ("spark_phase_eof", [commentary]),
        ]:
            popup = _popup_for_scenario(driver, "muse_spark", scenario)
            rendered = popup.get_attribute("textContent")
            assert "message_start phase=commentary" in rendered
            if scenario == "spark_phase_tool_final":
                assert "message_start phase=final_answer" in rendered
            for rows in ["tr.ttip-final", "tr:not(.ttip-final)"]:
                fields = popup.find_elements(By.CSS_SELECTOR, f"tbody {rows} td[data-cand='dynamo'] .fldl")
                boundaries = [node.get_attribute("textContent") for node in fields
                              if node.get_attribute("textContent").startswith("message_")]
                assert boundaries == expected, (scenario, rows, boundaries)
            popup.find_element(By.CSS_SELECTOR, ".ttip-close").click()
        driver.refresh()
        _ready(driver, "tab-unified")
        popup = _popup_for_scenario(driver, "muse_spark", "spark_phase_repeated")
        assert "message_start phase=commentary" in popup.get_attribute("textContent")
    finally:
        driver.quit()
        server.shutdown()
        server.server_close()


def test_nemotron_default_reasoning_popup_describes_family_default(published_viewer):
    server, _requests = _server(root=published_viewer)
    driver = _driver()
    try:
        driver.get(f"http://127.0.0.1:{server.server_port}/?tab=tab-unified")
        _ready(driver, 'tab-unified')
        popup = _popup_for_scenario(driver, 'nemotron_v3', 'default_reasoning_tool_without_closer')
        assert 'use the family default' in popup.text
        assert 'Check.' in popup.text and 'Then.' in popup.text
        assert 'log' in popup.text and '雪' in popup.text
        assert 'prompt opened no channel' not in popup.text
        popup.find_element(By.CSS_SELECTOR, '.ttip-close').click()
    finally:
        driver.quit()
        server.shutdown()


@pytest.mark.parametrize('vertical', [False, True])
def test_shared_parent_displays_missing_child_capture(historical_published_viewer, vertical):
    server, _requests = _server(root=historical_published_viewer)
    driver = _driver()
    try:
        query = urlencode({'tab': 'tab-unified', 'slots_tab-unified': 'dynamo@0.9.0,none,none', 'transpose': int(vertical)})
        driver.get(f"http://127.0.0.1:{server.server_port}/?{query}")
        _ready(driver, 'tab-unified')
        result = driver.execute_script("""
            const host = [...document.querySelectorAll('#tab-unified .matrix-cell[data-ttip-id]')].find(node => {
                const cell = window.__conformanceCellModel(node);
                return cell?.family === 'kimi_k3' && cell.sub === 'tool_only'
                    && node.getBoundingClientRect().height > 0;
            });
            if (!host) return null;
            const cell = window.__conformanceCellModel(host);
            const key = document.querySelector('#tab-unified')._comparisonSelection.keys[0];
            const candidate = window.__CONFORMANCE_PAGE.tabs[0].candidates.find(item=>item.key === key);
            return {identity:candidate.identity,
                    status:window.__conformanceCaptureOutcome(cell, key, 'golden'),
                    painted:host.querySelector('[data-capture="' + key + '"]').className,
                    variants:Object.fromEntries(cell.variants.map(variant=>[
                      variant.sub,window.__conformanceCaptureOutcome(variant, key, 'golden')]))};
        """)
        assert result is not None
        assert result['identity'] == 'dynamo_v2@0.9.0'
        assert result['status'] == 'missing'
        assert 'status-missing' in result['painted']
        assert result['variants'] == {'gpt_oss_tool_body_whitespace': 'missing', 'tool_only': 'pass'}
    finally:
        driver.quit()
        server.shutdown()


def test_structural_response_config_escapes_hostile_rejection_text(published_viewer):
    server, _requests = _server(root=published_viewer)
    driver = _driver()
    attack = '</code><script>window.injectedStructuralConfig = true</script>'
    try:
        driver.get(f"http://127.0.0.1:{server.server_port}/?tab=tab-unified")
        _ready(driver, 'tab-unified')
        driver.execute_script("""
            const host = [...document.querySelectorAll('#tab-unified .matrix-cell[data-ttip-id]')].find(node => {
                const cell = window.__conformanceCellModel(node);
                return cell?.family === 'gpt_oss' && cell.sub === 'structured_response_after_reasoning'
                    && node.getBoundingClientRect().height > 0;
            });
            window.structuralConfigUnderTest = window.__conformanceCellModel(host).tooltip.init.structural_tag;
            window.structuralConfigUnderTest.rejected.push(arguments[0]);
        """, attack)
        popup = _popup_for_scenario(driver, 'gpt_oss', 'structured_response_after_reasoning')
        config = next(node for node in popup.find_elements(By.CSS_SELECTOR, '.ttip-config code')
                      if node.get_attribute('textContent').startswith('structural_tag='))
        assert attack in json.loads(config.get_attribute('textContent').removeprefix('structural_tag='))['rejected']
        assert not popup.find_elements(By.CSS_SELECTOR, 'script')
        assert driver.execute_script('return window.injectedStructuralConfig || false') is False
        popup.find_element(By.CSS_SELECTOR, '.ttip-close').click()
        driver.execute_script("""
            window.structuralConfigUnderTest.rejected.pop();
            delete window.structuralConfigUnderTest;
        """)
    finally:
        driver.quit()
        server.shutdown()




def test_numeric_popup_preserves_exact_argument_text(published_viewer):
    _assert_numeric_argument_popup(published_viewer)


def test_historical_numeric_popup_preserves_exact_argument_text(historical_published_viewer):
    _assert_numeric_argument_popup(historical_published_viewer)


def _assert_numeric_argument_popup(site):
    server, _requests = _server(root=site)
    driver = _driver()
    try:
        driver.get(f"http://127.0.0.1:{server.server_port}/?tab=tab-unified")
        _ready(driver, "tab-unified")
        scenario = driver.execute_script("""
            const cell = [...document.querySelectorAll('#tab-unified .matrix-cell[data-ttip-id]')].find(node => {
                const model = window.__conformanceCellModel(node);
                return model?.family === 'qwen3' && model.variants?.some(leaf => leaf.sub === 'arg_numeric_15_round_down');
            });
            return cell && window.__conformanceCellModel(cell).sub;
        """)
        assert scenario
        popup = _popup_for_scenario(driver, 'qwen3', scenario)
        variants = popup.find_elements(By.CSS_SELECTOR, '.case-variant')
        assert len(variants) == 7
        variant = next(node for node in variants
                       if 'arg_numeric_15_round_down' in node.find_element(By.CSS_SELECTOR, '.ttip-head').text)
        output = variant.find_element(By.CSS_SELECTOR, '.ttip-final [data-cand="dynamo"]')
        assert '9007199254740992.5' in output.text
        assert '9007199254740992.0' not in output.text
        assert 'cand-diverge' not in output.get_attribute('class').split()
    finally:
        driver.quit()
        server.shutdown()
        server.server_close()
