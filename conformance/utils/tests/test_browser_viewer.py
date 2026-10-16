from __future__ import annotations

import threading
import hashlib
import json
import sys
import tarfile
from collections import Counter
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit

import pytest

selenium = pytest.importorskip("selenium")
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select, WebDriverWait

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "conformance/utils/src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
import publish_viewer


def _server(prefix: str = "", gate=None, gate_path: str | None = None, fault=None, root: Path = ROOT):
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
            nonlocal held
            should_hold = False
            with gate_lock:
                matches_gate = gate_path in path if gate_path else "/fixtures-unified-v2/" in path and "dynamo_v2-" in path
                if gate and not held and matches_gate:
                    held = True
                    should_hold = True
            if should_hold:
                gate[0].set()
                assert gate[1].wait(60), "delayed resource was never released"
            if fault and "/fixtures-unified-v2/" in path and "dynamo_v2-" in path and not fault["used"]:
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
    WebDriverWait(driver, 90).until(lambda page: page.find_elements("css selector", f"#{tab}[data-load-state=ready] td.cell") or page.find_elements("css selector", ".error, .load-status[role=alert]"))
    assert not driver.find_elements("css selector", ".error, .load-status[role=alert]")


def _choose_engine(driver, slot: int, engine: str):
    driver.find_element("css selector", f"[data-slot-open='{slot}']").click()
    picker = WebDriverWait(driver, 10).until(EC.visibility_of_element_located(("css selector", ".comparison-picker[open]")))
    picker.find_element("css selector", f"[data-picker-engine-choice='{engine}']").click()
    return WebDriverWait(driver, 10).until(lambda page: page.find_element("css selector", f"[data-slot-capture='{slot}']"))


def _clear_slot(driver, slot: int):
    driver.find_element("css selector", f"[data-slot-open='{slot}']").click()
    picker = WebDriverWait(driver, 10).until(EC.visibility_of_element_located(("css selector", ".comparison-picker[open]")))
    picker.find_element("css selector", "[data-picker-engine-choice='none']").click()


def test_browser_viewer_loads_default_slot_and_controls():
    server, requests = _server()
    driver = _driver()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        driver.get(f"{base}/index.html?tab=tab-unified&view=details&transpose=0&keep=1")
        _ready(driver, "tab-unified")
        assert driver.title == 'Dynamo Parser v2 Conformance Table'
        assert driver.find_element('css selector', '#viewer > h1').text == driver.title
        assert len(driver.find_elements("css selector", "#tab-unified table[data-parity-table] td.cell")) == 5022
        family_labels = {item.text for item in driver.find_elements("css selector", "#tab-unified table[data-parity-table] td.model")}
        assert "deepseek_v4 (v1/v2 split)" in family_labels
        assert "deepseek_v3 (v1)" in family_labels
        assert "muse_glimmer (v2)" in family_labels
        assert driver.find_element("id", "viewer").get_attribute("data-catalog-revision")
        assert driver.find_elements('css selector', '#tab-unified .capture-slot')
        assert len(driver.find_elements('css selector', '#tab-unified .capture-slot')) == 3
        select = driver.find_element('css selector', "#tab-unified select[data-slot-capture='0']")
        assert select.get_attribute('value') == 'dynamo'
        assert driver.execute_script("return window.__conformanceController.selection(document.getElementById('tab-unified'))") == {'reference': None, 'selected': ['dynamo']}
        assert driver.find_element('css selector', "#tab-unified .capture-slot[data-engine='none']")
        assert driver.find_elements('css selector', '#viewer > .tabs .tab-button') == []
        loaded = [path for path in requests if "/fixtures" in path]
        assert loaded and all("/fixtures-unified-v2/" in path for path in loaded)
        assert all(count == 1 for count in Counter(loaded).values())
        assert any('/dynamo_v2-' in path for path in loaded)
        assert driver.execute_script("return window.__conformanceModel.rows.filter(r => !r.section).length") == 27
        family_labels = driver.execute_script("return window.__conformanceModel.rows.filter(r => !r.section).map(r => r.model_label)")
        assert len(family_labels) == 27
        assert all(label.endswith((" (v1)", " (v1/v2 split)", " (v2)")) for label in family_labels), family_labels
        def color_summary():
            return driver.execute_script("""
                const counts = {equal: 0, different: 0, unavailable: 0};
                for (const cell of document.querySelectorAll('#tab-unified table[data-parity-table] td.cell')) {
                  if (cell.classList.contains('cmp-eq')) counts.equal++;
                  if (cell.classList.contains('cmp-leak')) counts.different++;
                  if (cell.classList.contains('cmp-na')) counts.unavailable++;
                }
                return counts;
            """)
        newest_reference = color_summary()
        Select(select).select_by_value('dynamo@0.7.13')
        _ready(driver, 'tab-unified')
        older_reference = color_summary()
        assert older_reference != newest_reference, (
            f"changing the Dynamo Reference did not recolor the matrix: {newest_reference}"
        )
        Select(select).select_by_value('dynamo')
        _ready(driver, 'tab-unified')
        driver.find_element("css selector", "[data-transpose-toggle]").click()
        assert "transpose-mode" in driver.find_element("tag name", "body").get_attribute("class")
        assert driver.find_elements("css selector", "#tab-unified table[data-transpose-table] td.cell")
        driver.find_element("css selector", "[data-theme-toggle]").click()
        assert driver.find_element("tag name", "html").get_attribute("data-theme") == "dark"
        driver.find_element("css selector", "[data-transpose-toggle]").click()
        cell = driver.find_element("css selector", "#tab-unified table[data-parity-table] td.cell[data-ttip-id]")
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
        driver.get(f"{base}/conformance/CONFORMANCE_v2.html?tab=tab-unified&keep=1#case")
        WebDriverWait(driver, 20).until(lambda page: urlsplit(page.current_url).path in ("/", "/index.html"))
        parsed = urlsplit(driver.current_url)
        assert "keep=1" in parsed.query.split("&") and parsed.fragment == "case"
    finally:
        driver.quit()
        server.shutdown()


def test_published_viewer_loads_resolved_json_without_yaml_histories(tmp_path):
    site = tmp_path / "published"
    publish_viewer.publish(site, "a" * 40)
    catalog = json.loads((site / "conformance/viewer-unified-catalog.json").read_text())
    server, requests = _server(root=site)
    driver = _driver()
    try:
        driver.get(f"http://127.0.0.1:{server.server_port}/index.html?tab=tab-unified")
        _ready(driver, "tab-unified")
        selected_identity = driver.execute_script("""
          const tab = window.__CONFORMANCE_PAGE.tabs.find(item => item.id === 'tab-unified');
          return tab.candidates.find(item => item.key === 'dynamo')?.source_id;
        """)
        assert selected_identity
        resource_paths = [urlsplit(path).path.removeprefix("/") for path in requests if "/viewer-data/" in path]
        assert resource_paths
        assert all(path.endswith(".json.gz") for path in resource_paths)
        assert not any(path.endswith(".yaml") or "/fixtures-unified-v2/" in path for path in requests)
        resources = {item["path"]: item for item in catalog["sources"]["unified"]}
        requested = [resources[path] for path in resource_paths]
        identities = {item["capture_identity"] for item in requested if item["kind"] == "resolved-capture"}
        assert selected_identity in identities
        assert identities == {selected_identity}, f"first paint fetched unselected capture versions: {identities - {selected_identity}}"
        assert len(resource_paths) == len(set(resource_paths)), "published shards should be fetched once"
        driver.find_element("css selector", "#tab-unified [data-slot-open='1']").click()
        WebDriverWait(driver, 30).until(lambda _page: len({
            resources[urlsplit(path).path.removeprefix("/")]["capture_identity"]
            for path in requests if "/viewer-data/" in path
            and resources[urlsplit(path).path.removeprefix("/")]["kind"] == "resolved-capture"
        }) == 3)
        prefetched = {resources[urlsplit(path).path.removeprefix("/")]["capture_identity"]
                      for path in requests if "/viewer-data/" in path
                      and resources[urlsplit(path).path.removeprefix("/")]["kind"] == "resolved-capture"}
        assert selected_identity in prefetched and len(prefetched - {selected_identity}) == 2, prefetched
    finally:
        driver.quit()
        server.shutdown()


def test_browser_viewer_resolves_nested_http_prefix_and_legacy_empty_reference():
    server, requests = _server("/nested")
    driver = _driver()
    try:
        driver.get(f"http://127.0.0.1:{server.server_port}/nested/index.html?tab=tab-unified&base_tab-unified=&transpose=0")
        _ready(driver, "tab-unified")
        assert driver.execute_script("return window.__conformanceController.selection(document.getElementById('tab-unified'))") == {"reference": None, "selected": ["dynamo"]}
        assert not any(path.endswith("viewer-catalog.json") for path in requests)
        assert driver.current_url.startswith(f"http://127.0.0.1:{server.server_port}/nested/")
        assert driver.find_elements("css selector", "#tab-unified td.cell.cmp-eq, #tab-unified td.cell.cmp-leak")
        driver.refresh()
        _ready(driver, "tab-unified")
        assert driver.execute_script("return window.__conformanceController.selection(document.getElementById('tab-unified'))") == {"reference": None, "selected": ["dynamo"]}
    finally:
        driver.quit()
        server.shutdown()


def test_same_engine_slots_none_geometry_duplicate_prevention_and_reset():
    server, _requests = _server()
    driver = _driver()
    try:
        driver.get(f"http://127.0.0.1:{server.server_port}/index.html?tab=tab-unified&transpose=0&keep=1#saved")
        _ready(driver, "tab-unified")
        none_slot = driver.find_element("css selector", "#tab-unified .capture-slot[data-engine='none']")
        assert none_slot.is_displayed()
        bounds = none_slot.rect
        assert bounds["width"] > 0 and bounds["height"] > 0

        second = _choose_engine(driver, 1, "dynamo")
        Select(second).select_by_value("dynamo_v1@9.2.10")
        _ready(driver, "tab-unified")
        assert driver.execute_script("return window.__conformanceController.selection(document.getElementById('tab-unified')).selected") == ["dynamo", "dynamo_v1@9.2.10"]
        assert not second.find_element("css selector", "option[value='dynamo']").is_enabled()

        _clear_slot(driver, 1)
        assert driver.execute_script("return window.__conformanceController.selection(document.getElementById('tab-unified')).selected") == ["dynamo"]
        driver.find_element("css selector", "[data-reset]").click()
        _ready(driver, "tab-unified")
        assert driver.execute_script("return window.__conformanceController.selection(document.getElementById('tab-unified')).selected") == ["dynamo"]
        parsed = urlsplit(driver.current_url)
        assert dict(parse_qsl(parsed.query)) == {"tab": "tab-unified", "keep": "1"}
        assert parsed.fragment == "saved"
    finally:
        driver.quit()
        server.shutdown()


def test_delayed_previous_capture_cannot_repaint_current_matrix():
    started, release = threading.Event(), threading.Event()
    server, requests = _server(gate=(started, release), gate_path="dynamo_v2-0.7.22.yaml")
    driver = _driver()
    try:
        driver.get(f"http://127.0.0.1:{server.server_port}/?tab=tab-unified&transpose=0")
        assert started.wait(20)
        select = driver.find_element("css selector", "#tab-unified select[data-slot-capture='0']")
        assert select.is_displayed() and select.is_enabled()
        Select(select).select_by_value("dynamo@0.7.13")
        _ready(driver, "tab-unified")
        assert driver.execute_script("return window.__conformanceModel.id") == "tab-unified"
        release.set()
        WebDriverWait(driver, 90).until(lambda page: page.execute_script("return window.__conformanceLoader.pending === 0"))
        assert driver.execute_script("return window.__conformanceController.selection(document.getElementById('tab-unified'))") == {"reference": None, "selected": ["dynamo@0.7.13"]}
        assert driver.find_element("css selector", ".tab-panel.active").get_attribute("id") == "tab-unified"
        assert not any("/toolcalling/" in path or "/reasoning/fixtures" in path for path in requests)
    finally:
        release.set()
        driver.quit()
        server.shutdown()


@pytest.mark.parametrize("kind", ["missing", "corrupt", "pointer"])
def test_resource_failures_remain_transport_errors_and_retry(kind):
    fault = {"used": False, "kind": kind}
    server, _requests = _server(fault=fault)
    driver = _driver()
    try:
        driver.get(f"http://127.0.0.1:{server.server_port}/?tab=tab-unified&transpose=0")
        error = WebDriverWait(driver, 30).until(EC.visibility_of_element_located(("css selector", "#tab-unified .load-status[role=alert]")))
        assert "Fixture load failed" in error.text
        assert not driver.find_elements("css selector", "#tab-unified[data-load-state=ready]")
        assert not driver.execute_script("return Boolean(window.__conformanceModel)")
        assert driver.find_element("css selector", "#tab-unified .capture-slots").is_displayed()
        assert driver.find_element("css selector", "#tab-unified [data-slot-open='0']").is_displayed()
        assert not driver.find_element("css selector", "#tab-unified table[data-parity-table]").is_displayed()
        error.find_element("tag name", "button").click()
        _ready(driver, "tab-unified")
        assert driver.find_elements("css selector", "#tab-unified td.cell")
    finally:
        driver.quit()
        server.shutdown()


def test_unified_controls_render_before_fixtures_and_detailed_view_stays_on_unified():
    started, release = threading.Event(), threading.Event()
    server, requests = _server(gate=(started, release))
    driver = _driver()
    try:
        driver.get(f"http://127.0.0.1:{server.server_port}/?tab=tab-reasoning-batch&transpose=0")
        assert started.wait(20)
        assert driver.current_url.startswith(f"http://127.0.0.1:{server.server_port}/?tab=tab-unified")
        assert len(driver.find_elements("css selector", "#tab-unified .capture-slot")) == 3
        assert len(driver.find_elements("css selector", "#tab-unified select[data-slot-capture]")) == 1
        assert driver.find_element("css selector", "#tab-unified .capture-slots").is_displayed()
        assert driver.find_element("css selector", "#tab-unified [data-slot-open='0']").is_displayed()
        assert not driver.find_element("css selector", "#tab-unified [data-parity-table]").is_displayed()
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


def test_selection_loading_and_failure_suspend_results_and_restore_pinned_popup():
    server, _requests = _server()
    driver = _driver()
    try:
        driver.get(f"http://127.0.0.1:{server.server_port}/?tab=tab-unified&view=details&transpose=0")
        _ready(driver, "tab-unified")
        cell = driver.find_element("css selector", "#tab-unified table[data-parity-table] td.cell[data-ttip-id]")
        driver.execute_script("window.__buildTooltip(arguments[0]); arguments[0].dispatchEvent(new PointerEvent('pointerdown', {bubbles:true, pointerType:'touch'})); arguments[0].click();", cell)
        popup = WebDriverWait(driver, 10).until(EC.visibility_of_element_located(("css selector", ".ttip-pinned")))
        driver.execute_script("arguments[0].style.width='640px'; arguments[0].style.height='320px';", popup)
        case_id = cell.get_attribute("data-case-id")
        family = cell.get_attribute("data-family")
        driver.execute_script("""
            const loader = window.__conformanceLoader;
            const original = loader.build;
            loader.build = (...args) => {
                loader.build = original;
                return new Promise((resolve, reject) => { window.rejectHeldSelection = reject; });
            };
        """)
        _choose_engine(driver, 1, 'vllm')

        def assert_suspended(state):
            panel = driver.find_element("id", "tab-unified")
            assert panel.get_attribute("data-load-state") == state
            assert not driver.execute_script("return Boolean(window.__conformanceModel)")
            assert not driver.find_elements("css selector", "#tab-unified td.cell.cmp-eq, #tab-unified td.cell.cmp-leak, #tab-unified td.cell.cmp-na")
            assert not driver.find_elements("css selector", ".ttip-visible")
            assert not driver.find_element("css selector", "#tab-unified table[data-parity-table]").is_displayed()
            assert driver.find_element("css selector", "#tab-unified .capture-slots").is_displayed()
            assert driver.find_element("css selector", "#tab-unified [data-slot-open='0']").is_displayed()
            assert not any(node.is_displayed() for node in driver.find_elements("css selector", "#tab-unified .matrix-status"))
            assert {node.get_attribute("textContent") for node in driver.find_elements("css selector", "#tab-unified [data-overview-count]")} == {"—"}

        assert_suspended("loading")
        driver.execute_script("window.rejectHeldSelection(new Error('held transport failure'))")
        WebDriverWait(driver, 10).until(lambda page: page.find_element("id", "tab-unified").get_attribute("data-load-state") == "error")
        assert_suspended("error")
        driver.find_element("css selector", "#tab-unified .load-status button").click()
        _ready(driver, "tab-unified")
        restored = WebDriverWait(driver, 10).until(EC.visibility_of_element_located(("css selector", ".ttip-pinned")))
        restored_cell = driver.execute_script("return [...document.querySelectorAll('#tab-unified td.cell')].find(cell => cell._ttip === arguments[0]);", restored)
        assert restored_cell.get_attribute("data-case-id") == case_id
        assert restored_cell.get_attribute("data-family") == family
        assert restored.value_of_css_property("width") == "640px"
        assert restored.value_of_css_property("height") == "320px"
        assert driver.execute_script("return window.__conformanceModel.id") == "tab-unified"
    finally:
        driver.quit()
        server.shutdown()


def test_rapid_compare_selections_restore_url_and_ignore_late_same_tab_result():
    server, requests = _server()
    driver = _driver()
    try:
        driver.get(f"http://127.0.0.1:{server.server_port}/?tab=tab-unified&view=details&transpose=0&keep=1#saved")
        _ready(driver, "tab-unified")
        selected = driver.execute_script("""
            const panel = document.getElementById('tab-unified');
            const loader = window.__conformanceLoader;
            const original = loader.build;
            const staleModel = window.__conformanceModel;
            loader.build = (...args) => {
                loader.build = original;
                return new Promise(resolve => { window.releaseOldComparison = () => resolve(staleModel); });
            };
        """)
        vllm_select = _choose_engine(driver, 1, 'vllm')
        rust = next(option.get_attribute('value') for option in vllm_select.find_elements('tag name', 'option') if option.get_attribute('value').startswith('vllm_rust@'))
        Select(vllm_select).select_by_value(rust)
        _choose_engine(driver, 2, 'sglang')
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
            cell = driver.find_element("css selector", "#tab-unified td.cell[data-ttip-id]")
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
    ("tab-unified", False, "th.case-sub[data-ttip-id]"),
    ("tab-unified", True, "th.trow-case[data-ttip-id]"),
])
def test_grammar_header_pin_survives_asynchronous_model_refresh(tab, transpose, selector):
    server, _requests = _server()
    driver = _driver()
    try:
        driver.get(f"http://127.0.0.1:{server.server_port}/?tab={tab}&view=details&transpose={int(transpose)}")
        _ready(driver, tab)
        popup_key = driver.execute_script("""
            const panel = document.getElementById(arguments[0]);
            const headers = [...panel.querySelectorAll(arguments[1])];
            let header = headers.find(node => node.offsetWidth > 20 && node.offsetHeight > 10);
            if (!header) {
                const key = headers[0].dataset.colHideGroup || headers[0].closest('[data-col-hide-group]').dataset.colHideGroup;
                [...panel.querySelectorAll('[data-col-toggle]')].find(button => button.dataset.colToggle === key).click();
                header = headers.find(node => node.offsetWidth > 20 && node.offsetHeight > 10);
            }
            if (!header) throw new Error('No interactable grammar header');
            header.scrollIntoView();
            window.__buildTooltip(header);
            header.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, pointerType: 'touch' }));
            header.click();
            return header.dataset.popupKey;
        """, tab, selector)
        popup = WebDriverWait(driver, 10).until(EC.visibility_of_element_located(("css selector", ".ttip-pinned")))
        assert popup_key and popup.text
        driver.execute_script("const panel=document.getElementById(arguments[0]); window.__conformanceLoader.select(panel.id, window.__conformanceController.selection(panel));", tab)
        _ready(driver, tab)
        WebDriverWait(driver, 20).until(lambda page: page.execute_script("return window.__conformanceLoader.pending === 0"))
        restored = WebDriverWait(driver, 10).until(EC.visibility_of_element_located(("css selector", ".ttip-pinned")))
        assert driver.execute_script("return arguments[0]._ttipOwner.dataset.popupKey", restored) == popup_key
        restored.find_element("css selector", ".ttip-close").click()
        assert not driver.find_elements("css selector", ".ttip-pinned")
    finally:
        driver.quit()
        server.shutdown()


def test_reset_discards_popup_pin_suspended_by_selection_load():
    server, _requests = _server()
    driver = _driver()
    try:
        driver.get(f"http://127.0.0.1:{server.server_port}/?tab=tab-unified&view=details&transpose=0")
        _ready(driver, "tab-unified")
        cell = driver.find_element("css selector", "#tab-unified td.cell[data-ttip-id]")
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
        assert not driver.find_elements("css selector", ".ttip-pinned")
        driver.find_element("css selector", "[data-reset]").click()
        _ready(driver, "tab-unified")
        selection = driver.execute_script("return window.__conformanceController.selection(document.getElementById('tab-unified'))")
        assert selection == {'reference': None, 'selected': ['dynamo']}
        assert not driver.find_elements("css selector", ".ttip-pinned")
        assert driver.execute_script("return window.__conformanceModel.id") == "tab-unified"
    finally:
        driver.quit()
        server.shutdown()



@pytest.mark.parametrize('kind', ['archive', 'unified'])
def test_original_yaml_view_and_blob_download_preserve_source_bytes(kind):
    archive_path = ROOT / 'conformance/fixtures/toolcalling/fixtures-stream-v1/inputs.tar.gz'
    member = 'toolcalling/fixtures-stream-v1/inputs/deepseek_v4/TOOLCALLING.streamv1.1.yaml'
    direct = 'conformance/fixtures-unified-v2/families/qwen3/inputs_and_golden.yaml'
    if kind == 'archive':
        with tarfile.open(archive_path) as archive:
            expected = archive.extractfile(member).read()
        path = member
    else:
        expected = (ROOT / direct).read_bytes()
        path = direct
    digest = hashlib.sha256(expected).hexdigest()
    server, _requests = _server()
    driver = _driver()
    try:
        query = urlencode({'tab': 'tab-unified', 'raw': path, 'keep': 'original-yaml'})
        driver.get(f'http://127.0.0.1:{server.server_port}/?{query}#saved')
        WebDriverWait(driver, 30).until(lambda page: page.find_elements('css selector', '#viewer > pre') or page.find_elements('css selector', '.error'))
        assert not driver.find_elements('css selector', '.error')
        actual = driver.execute_async_script("""
            const done = arguments[0];
            (async () => {
                const links = [...document.querySelectorAll('#viewer > a')];
                const download = links.find(anchor => anchor.download);
                const bytes = new Uint8Array(await (await fetch(download.href)).arrayBuffer());
                const { sha256 } = await import('/conformance/viewer/data.js');
                done({ hash: await sha256(bytes), size: bytes.length,
                    renderedHash: await sha256(new TextEncoder().encode(document.querySelector('#viewer > pre').textContent)),
                    filename: download.download, back: links.find(anchor => !anchor.download).href });
            })().catch(error => done({ error: String(error) }));
        """)
        assert 'error' not in actual
        assert actual['hash'] == actual['renderedHash'] == digest
        assert actual['size'] == len(expected)
        assert actual['filename'] == path.split('/')[-1]
        back = urlsplit(actual['back'])
        assert dict(parse_qsl(back.query)) == {'tab': 'tab-unified', 'keep': 'original-yaml'}
        assert back.fragment == 'saved'
    finally:
        driver.quit()
        server.shutdown()


def test_legacy_capture_alias_and_reset_preserve_unrelated_url_state():
    server, _requests = _server()
    driver = _driver()
    try:
        driver.get(f'http://127.0.0.1:{server.server_port}/?tab=tab-unified&ref=dynamo_v2%400.7.18&cmp=dynamo&keep=1&transpose=1#saved')
        _ready(driver, 'tab-unified')
        assert driver.execute_script("return window.__conformanceController.selection(document.getElementById('tab-unified'))") == {'reference': None, 'selected': ['dynamo@0.7.18', 'dynamo']}
        saved = urlsplit(driver.current_url)
        query = dict(parse_qsl(saved.query, keep_blank_values=True))
        slots = query['slots_tab-unified'].split(',')
        assert slots[0] == 'dynamo_v2@0.7.18' and slots[1].startswith('dynamo_v2@') and slots[2] == 'none'
        assert 'view' not in query and query['transpose'] == '1' and query['keep'] == '1'
        assert 'base_tab-unified' not in query and 'cmp_tab-unified' not in query
        assert 'ref' not in query and 'cmp' not in query and saved.fragment == 'saved'
        driver.find_element('css selector', '[data-reset]').click()
        _ready(driver, 'tab-unified')
        reset = urlsplit(driver.current_url)
        assert dict(parse_qsl(reset.query)) == {'tab': 'tab-unified', 'keep': '1'} and reset.fragment == 'saved'
        assert driver.execute_script("return window.__conformanceController.selection(document.getElementById('tab-unified'))") == {'reference': None, 'selected': ['dynamo']}
        assert not driver.execute_script("return document.body.classList.contains('transpose-mode')")
    finally:
        driver.quit()
        server.shutdown()


@pytest.mark.parametrize('columns', ['model', 'batch_core'])
def test_legacy_column_selection_survives_async_mount_and_reload(columns):
    server, _requests = _server()
    driver = _driver()
    try:
        driver.get(f'http://127.0.0.1:{server.server_port}/?tab=tab-unified&base_tab-unified=&base_tab-reasoning-batch=&cols={columns}&transpose=0')
        _ready(driver, 'tab-unified')
        buttons = driver.find_elements('css selector', '#tab-unified [data-col-toggle]')
        assert all((button.get_attribute('aria-pressed') == 'true') == (button.get_attribute('data-col-toggle') == 'model') for button in buttons) if columns == 'model' else all(button.get_attribute('aria-pressed') == 'true' for button in buttons)
        assert 'cols' not in dict(parse_qsl(urlsplit(driver.current_url).query))
        driver.refresh()
        _ready(driver, 'tab-unified')
    finally:
        driver.quit()
        server.shutdown()


def test_unified_column_toggle_preserves_hide_precedence_and_reload():
    server, _requests = _server()
    driver = _driver()
    try:
        driver.get(f'http://127.0.0.1:{server.server_port}/?tab=tab-unified&base_tab-unified=&base_tab-reasoning-batch=&base_tab-reasoning-stream=&hide=batch_core&show=batch_core,stream_chunk_boundaries&transpose=0')
        _ready(driver, 'tab-unified')
        driver.find_element('css selector', '#tab-unified [data-col-toggle=model]').click()
        query = dict(parse_qsl(urlsplit(driver.current_url).query))
        assert 'batch_core' not in query.get('hide', '').split(',')
        assert driver.find_element('css selector', '#tab-unified [data-col-toggle=model]').get_attribute('aria-pressed') == 'false'
        driver.refresh()
        _ready(driver, 'tab-unified')
        assert driver.find_element('css selector', '#tab-unified [data-col-toggle=model]').get_attribute('aria-pressed') == 'false'
    finally:
        driver.quit()
        server.shutdown()


@pytest.mark.parametrize('columns', ['', 'obsolete_unknown'])
def test_empty_or_unknown_legacy_columns_keep_defaults(columns):
    server, _requests = _server()
    driver = _driver()
    try:
        driver.get(f'http://127.0.0.1:{server.server_port}/?tab=tab-unified&base_tab-unified=&cols={columns}&transpose=0')
        _ready(driver, 'tab-unified')
        assert all(button.get_attribute('aria-pressed') == 'true' for button in driver.find_elements('css selector', '#tab-unified [data-col-toggle]'))
    finally:
        driver.quit()
        server.shutdown()


def test_collapsing_last_legacy_column_restores_no_visible_columns_on_reload():
    server, _requests = _server()
    driver = _driver()
    try:
        driver.get(f'http://127.0.0.1:{server.server_port}/?tab=tab-unified&base_tab-unified=&cols=model&transpose=0')
        _ready(driver, 'tab-unified')
        driver.find_element('css selector', '#tab-unified [data-col-toggle=model]').click()
        driver.refresh()
        _ready(driver, 'tab-unified')
        assert all(button.get_attribute('aria-pressed') == 'false' for button in driver.find_elements('css selector', '#tab-unified [data-col-toggle]'))
    finally:
        driver.quit()
        server.shutdown()


def test_unified_slots_migrate_legacy_comparison_url_and_fit_mobile():
    server, _requests = _server()
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


def test_unified_slots_migrate_tab_scoped_legacy_url_after_tab_canonicalization():
    server, _requests = _server()
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
        assert params['slots_tab-unified'] == 'sglang_python@0.5.21,dynamo_v2@0.7.23,vllm_python@0.31.0'
        assert params['keep'] == '1'
        assert not any(key.startswith(('base_', 'cmp_')) for key in params)
        assert urlsplit(driver.current_url).fragment == 'saved'
    finally:
        driver.quit()
        server.shutdown()


def test_unified_capture_slots_survive_reload_with_url_state():
    server, _requests = _server()
    driver = _driver()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        driver.get(f"{base}/?tab=tab-unified&slots_tab-unified=dynamo_v2%400.7.22,none,vllm_python%400.31.0&keep=1#saved")
        _ready(driver, 'tab-unified')
        expected = {'reference': None, 'selected': ['dynamo@0.7.22', 'vllm_python@0.31.0']}
        assert driver.execute_script("return window.__conformanceController.selection(document.getElementById('tab-unified'))") == expected
        assert dict(parse_qsl(urlsplit(driver.current_url).query))['slots_tab-unified'] == 'dynamo_v2@0.7.22,none,vllm_python@0.31.0'
        driver.refresh()
        _ready(driver, 'tab-unified')
        assert driver.execute_script("return window.__conformanceController.selection(document.getElementById('tab-unified'))") == expected
        params = dict(parse_qsl(urlsplit(driver.current_url).query))
        assert params['slots_tab-unified'] == 'dynamo_v2@0.7.22,none,vllm_python@0.31.0'
        assert params['keep'] == '1'
        assert urlsplit(driver.current_url).fragment == 'saved'
    finally:
        driver.quit()
        server.shutdown()


def test_unified_slots_migrate_legacy_stream_candidate_keys():
    server, _requests = _server()
    driver = _driver()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        driver.get(f"{base}/?tab=tab-toolcalling-streamv1&base_tab-toolcalling-streamv1=dynamo_v2-0-7-22&cmp_tab-toolcalling-streamv1=vllm_python-0-31-0&keep=1#saved")
        _ready(driver, 'tab-unified')
        assert driver.execute_script("return window.__conformanceController.selection(document.getElementById('tab-unified'))") == {
            'reference': None, 'selected': ['dynamo@0.7.22', 'vllm_python@0.31.0']
        }
        params = dict(parse_qsl(urlsplit(driver.current_url).query))
        assert params['tab'] == 'tab-unified'
        assert params['slots_tab-unified'] == 'dynamo_v2@0.7.22,none,vllm_python@0.31.0'
        assert params['keep'] == '1'
        assert not any(key.startswith(('base_', 'cmp_')) for key in params)
        assert urlsplit(driver.current_url).fragment == 'saved'
    finally:
        driver.quit()
        server.shutdown()
