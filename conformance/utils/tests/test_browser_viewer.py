from __future__ import annotations

import threading
import hashlib
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


def _server(prefix: str = "", gate=None, fault=None):
    requests = []
    held = False
    gate_lock = threading.Lock()

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(ROOT), **kwargs)

        def log_message(self, *_args):
            return

        def do_GET(self):
            path = urlsplit(self.path).path
            requests.append(path)
            nonlocal held
            should_hold = False
            with gate_lock:
                if gate and not held and "/fixtures-unified-v2/" in path and "dynamo_v2-" in path:
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


def test_browser_viewer_loads_matrix_and_controls():
    server, requests = _server()
    driver = _driver()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        driver.get(f"{base}/index.html?tab=tab-unified&view=details&transpose=0&keep=1")
        _ready(driver, "tab-unified")
        assert driver.title == 'Dynamo Parser v2 Conformance Table'
        assert driver.find_element('css selector', '#viewer > h1').text == driver.title
        assert len(driver.find_elements("css selector", "#tab-unified table[data-parity-table] td.cell")) == 1352
        assert driver.find_element("id", "viewer").get_attribute("data-catalog-revision")
        loaded = [path for path in requests if "/fixtures" in path]
        assert loaded and all("/fixtures-unified-v2/" in path for path in loaded)
        assert all(count == 1 for count in Counter(loaded).values())
        assert all("vllm" not in path and "sglang" not in path for path in loaded)
        assert driver.execute_script("return window.__conformanceModel.rows.filter(r => !r.section).length") == 8
        driver.find_element("css selector", "[data-transpose-toggle]").click()
        assert "transpose-mode" in driver.find_element("tag name", "body").get_attribute("class")
        assert driver.find_elements("css selector", "#tab-unified table[data-transpose-table] td.cell")
        driver.find_element("css selector", "[data-theme-toggle]").click()
        assert driver.find_element("tag name", "html").get_attribute("data-theme") == "dark"
        driver.find_element("css selector", "[data-transpose-toggle]").click()
        cell = driver.find_element("css selector", "#tab-unified table[data-parity-table] td.cell[data-ttip-id]")
        driver.execute_script("window.__buildTooltip(arguments[0]); arguments[0].dispatchEvent(new MouseEvent('mouseenter'));", cell)
        popup = WebDriverWait(driver, 10).until(EC.visibility_of_element_located(("css selector", ".ttip-visible")))
        assert popup.text and "Golden output" in popup.text
        driver.execute_script("arguments[0].dispatchEvent(new PointerEvent('pointerdown', {bubbles:true, pointerType:'touch'})); arguments[0].click();", cell)
        assert "ttip-pinned" in popup.get_attribute("class")
        assert driver.execute_script("return getComputedStyle(arguments[0]).resize", popup) == "both"
        refs = driver.find_elements("css selector", "#tab-unified input.cmp-ref")
        ref = next(item for item in refs if item.is_selected())
        # Native Reference change invokes the shared asynchronous model and preserves URL.
        target = next(item for item in refs if not item.is_selected())
        target_key = target.get_attribute("value")
        driver.execute_script("arguments[0].click()", target)
        _ready(driver, "tab-unified")
        assert "keep=1" in driver.current_url
        assert driver.find_element("css selector", "#tab-unified input.cmp-ref:checked").get_attribute("value") == target_key
        driver.get(f"{base}/conformance/CONFORMANCE_v2.html?tab=tab-unified&keep=1#case")
        WebDriverWait(driver, 20).until(lambda page: urlsplit(page.current_url).path in ("/", "/index.html"))
        parsed = urlsplit(driver.current_url)
        assert "keep=1" in parsed.query.split("&") and parsed.fragment == "case"
    finally:
        driver.quit()
        server.shutdown()


def test_browser_viewer_resolves_nested_http_prefix_and_empty_reference():
    server, requests = _server("/nested")
    driver = _driver()
    try:
        driver.get(f"http://127.0.0.1:{server.server_port}/nested/index.html?tab=tab-unified&base_tab-unified=&transpose=0")
        _ready(driver, "tab-unified")
        assert not driver.find_elements("css selector", "#tab-unified input.cmp-ref:checked")
        assert not any("dynamo_v2-" in path or "vllm_python-" in path for path in requests)
        assert driver.current_url.startswith(f"http://127.0.0.1:{server.server_port}/nested/")
        assert driver.find_elements("css selector", "#tab-unified td.cell.cmp-nobase")
        driver.refresh()
        _ready(driver, "tab-unified")
        assert not driver.find_elements("css selector", "#tab-unified input.cmp-ref:checked")
    finally:
        driver.quit()
        server.shutdown()


def test_delayed_previous_tab_cannot_repaint_current_matrix():
    started, release = threading.Event(), threading.Event()
    server, requests = _server(gate=(started, release))
    driver = _driver()
    try:
        driver.get(f"http://127.0.0.1:{server.server_port}/?tab=tab-unified&transpose=0")
        assert started.wait(20)
        driver.find_element("id", "tab-reasoning-batch-button").click()
        _ready(driver, "tab-reasoning-batch")
        assert driver.execute_script("return window.__conformanceModel.id") == "tab-reasoning-batch"
        release.set()
        WebDriverWait(driver, 90).until(lambda page: page.execute_script("return window.__conformanceLoader.pending === 0"))
        assert driver.execute_script("return window.__conformanceModel.id") == "tab-reasoning-batch"
        assert driver.find_element("css selector", ".tab-panel.active").get_attribute("id") == "tab-reasoning-batch"
        assert not any("/toolcalling/" in path for path in requests)
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
        error.find_element("tag name", "button").click()
        _ready(driver, "tab-unified")
        assert driver.find_elements("css selector", "#tab-unified td.cell")
    finally:
        driver.quit()
        server.shutdown()


def test_leaving_detailed_only_loads_the_active_tab():
    server, requests = _server()
    driver = _driver()
    try:
        driver.get(f"http://127.0.0.1:{server.server_port}/?tab=tab-reasoning-batch&view=details&transpose=0")
        _ready(driver, "tab-reasoning-batch")
        driver.find_element("css selector", "[data-view-detailed]").click()
        _ready(driver, "tab-reasoning-batch")
        WebDriverWait(driver, 20).until(lambda page: page.execute_script("return window.__conformanceLoader.pending === 0"))
        assert driver.execute_script("return window.__conformanceModel.id") == "tab-reasoning-batch"
        assert not any("/toolcalling/" in path or "/fixtures-unified-v2/" in path for path in requests)
        assert not driver.find_elements("css selector", ".tab-panel:not(.active)[data-load-state=loading]")
        geometry = driver.execute_script("""
            const table = document.querySelector('#tab-reasoning-batch [data-parity-table]');
            return ['--sticky-top-row-height', '--sticky-model-width'].map((key, index) => {
                const node = table.querySelector(index === 0 ? 'thead > tr.matrix-groups' : 'thead th[data-col-control-group=model]');
                return [parseFloat(table.style.getPropertyValue(key)), node.getBoundingClientRect()[index === 0 ? 'height' : 'width']];
            });
        """)
        for stored, measured in geometry:
            assert measured > 0 and stored == pytest.approx(measured)
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
            const target = document.querySelector('#tab-unified input.cmp-ref:not(:checked)');
            target.checked = true;
            target.dispatchEvent(new Event('change', { bubbles: true }));
        """)

        def assert_suspended(state):
            panel = driver.find_element("id", "tab-unified")
            assert panel.get_attribute("data-load-state") == state
            assert not driver.execute_script("return Boolean(window.__conformanceModel)")
            assert not driver.find_elements("css selector", "#tab-unified td.cell.cmp-eq, #tab-unified td.cell.cmp-leak, #tab-unified td.cell.cmp-na")
            assert not driver.find_elements("css selector", ".ttip-visible")
            assert not driver.find_element("css selector", "#tab-unified table[data-parity-table]").is_displayed()
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
            const peers = [...panel.querySelectorAll('input.cmp-on')].filter(cb => !cb.disabled && cb.value !== 'golden').slice(0, 2);
            const loader = window.__conformanceLoader;
            const original = loader.build;
            loader.build = (...args) => {
                loader.build = original;
                return new Promise((resolve, reject) => {
                    original(...args).then(model => { window.releaseOldComparison = () => resolve(model); }, reject);
                });
            };
            peers[0].click();
            peers[1].click();
            return peers.map(cb => cb.value);
        """)
        assert len(selected) == 2
        _ready(driver, "tab-unified")
        WebDriverWait(driver, 90).until(lambda page: page.execute_script("return typeof window.releaseOldComparison === 'function'"))
        driver.execute_script("window.releaseOldComparison()")
        WebDriverWait(driver, 20).until(lambda page: page.execute_script("return window.__conformanceLoader.pending === 0"))

        def assert_current_selection():
            assert driver.execute_script("return window.__conformanceModel.id") == "tab-unified"
            shown = driver.execute_script("return window.__conformanceController.selection(document.getElementById('tab-unified')).selected")
            assert all(key in shown for key in selected)
            assert driver.execute_script("return window.__conformanceModel.rows.filter(row=>!row.section).every(row=>Object.values(row.cells).every(cell=>arguments[0].every(key=>Object.hasOwn(cell.cmp || {},key))))", selected)
            cell = driver.find_element("css selector", "#tab-unified td.cell[data-ttip-id]")
            driver.execute_script("window.__buildTooltip(arguments[0]);", cell)
            rendered = driver.execute_script("return [...arguments[0].querySelectorAll('.ttip [data-cand]')].filter(node=>!node.classList.contains('col-hidden')).map(node=>node.dataset.cand)", cell)
            assert all(key in rendered for key in selected)

        assert_current_selection()
        saved = driver.current_url
        assert "keep=1" in saved and saved.endswith("#saved")
        assert "cmp_tab-unified=" in saved
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
    ("tab-reasoning-batch", False, "td.parser[data-ttip-wired]"),
    ("tab-reasoning-batch", True, "th.tcol-model[data-ttip-wired]"),
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


def test_tab_navigation_discards_popup_pin_suspended_by_selection_load():
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
        driver.find_element("css selector", "[data-tab-target=tab-reasoning-batch]").click()
        _ready(driver, "tab-reasoning-batch")
        driver.execute_script("window.releaseNavigationSelection()")
        WebDriverWait(driver, 20).until(lambda page: page.execute_script("return window.__conformanceLoader.pending === 0"))
        assert driver.execute_script("return window.__conformanceModel.id") == "tab-reasoning-batch"
        driver.find_element("css selector", "[data-tab-target=tab-unified]").click()
        _ready(driver, "tab-unified")
        WebDriverWait(driver, 20).until(lambda page: page.execute_script("return window.__conformanceLoader.pending === 0"))
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


def test_short_selection_aliases_and_reset_preserve_tab_contract():
    server, _requests = _server()
    driver = _driver()
    try:
        driver.get(f'http://127.0.0.1:{server.server_port}/?tab=tab-unified&ref=dynamo_v2%400.7.18&cmp=dynamo&keep=1&transpose=1#saved')
        _ready(driver, 'tab-unified')
        assert driver.execute_script("return window.__conformanceController.selection(document.getElementById('tab-unified'))") == {'reference': 'dynamo@0.7.18', 'selected': ['dynamo@0.7.18', 'dynamo']}
        saved = urlsplit(driver.current_url)
        query = dict(parse_qsl(saved.query))
        assert query['base_tab-unified'] == 'dynamo@0.7.18'
        assert query['cmp_tab-unified'] == 'dynamo'
        assert query['view'] == 'details' and query['transpose'] == '1' and query['keep'] == '1'
        assert 'ref' not in query and 'cmp' not in query and saved.fragment == 'saved'
        driver.find_element('css selector', '[data-reset]').click()
        _ready(driver, 'tab-unified')
        reset = urlsplit(driver.current_url)
        assert reset.query == 'tab=tab-unified' and not reset.fragment
        assert driver.execute_script("return window.__conformanceController.selection(document.getElementById('tab-unified'))") == {'reference': 'dynamo', 'selected': ['dynamo']}
        assert not driver.execute_script("return document.body.classList.contains('transpose-mode')")
    finally:
        driver.quit()
        server.shutdown()


@pytest.mark.parametrize('columns', ['model', 'batch_core'])
def test_legacy_column_selection_survives_async_mount_and_other_tab_toggle(columns):
    server, _requests = _server()
    driver = _driver()
    try:
        driver.get(f'http://127.0.0.1:{server.server_port}/?tab=tab-unified&base_tab-unified=&base_tab-reasoning-batch=&cols={columns}&transpose=0')
        _ready(driver, 'tab-unified')
        buttons = driver.find_elements('css selector', '#tab-unified [data-col-toggle]')
        assert all((button.get_attribute('aria-pressed') == 'true') == (button.get_attribute('data-col-toggle') == columns) for button in buttons)
        assert 'cols' not in dict(parse_qsl(urlsplit(driver.current_url).query))
        driver.refresh()
        _ready(driver, 'tab-unified')
        if columns == 'batch_core':
            driver.find_element('css selector', '#tab-unified [data-col-toggle=model]').click()
            driver.find_element('id', 'tab-reasoning-batch-button').click()
            _ready(driver, 'tab-reasoning-batch')
            buttons = driver.find_elements('css selector', '#tab-reasoning-batch [data-col-toggle]')
            assert all((button.get_attribute('aria-pressed') == 'true') == (button.get_attribute('data-col-toggle') in {'batch_core', 'model'}) for button in buttons)
    finally:
        driver.quit()
        server.shutdown()


def test_column_toggle_preserves_unloaded_tab_overrides_and_hide_precedence():
    server, _requests = _server()
    driver = _driver()
    try:
        driver.get(f'http://127.0.0.1:{server.server_port}/?tab=tab-unified&base_tab-unified=&base_tab-reasoning-batch=&base_tab-reasoning-stream=&hide=batch_core&show=batch_core,stream_chunk_boundaries&transpose=0')
        _ready(driver, 'tab-unified')
        driver.find_element('css selector', '#tab-unified [data-col-toggle=model]').click()
        query = dict(parse_qsl(urlsplit(driver.current_url).query))
        assert 'batch_core' in query['hide'].split(',')
        driver.find_element('id', 'tab-reasoning-batch-button').click()
        _ready(driver, 'tab-reasoning-batch')
        assert driver.find_element('css selector', '#tab-reasoning-batch [data-col-toggle=batch_core]').get_attribute('aria-pressed') == 'false'
        driver.find_element('id', 'tab-reasoning-stream-button').click()
        _ready(driver, 'tab-reasoning-stream')
        assert driver.find_element('css selector', '#tab-reasoning-stream [data-col-toggle=stream_chunk_boundaries]').get_attribute('aria-pressed') == 'true'
        driver.refresh()
        _ready(driver, 'tab-reasoning-stream')
        assert driver.find_element('css selector', '#tab-reasoning-stream [data-col-toggle=stream_chunk_boundaries]').get_attribute('aria-pressed') == 'true'
        driver.find_element('id', 'tab-reasoning-batch-button').click()
        _ready(driver, 'tab-reasoning-batch')
        assert driver.find_element('css selector', '#tab-reasoning-batch [data-col-toggle=batch_core]').get_attribute('aria-pressed') == 'false'
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
