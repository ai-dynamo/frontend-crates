# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import shutil
import re
from pathlib import Path
from urllib.parse import quote

import pytest

pytest.importorskip("selenium")
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select, WebDriverWait

from conformance.utils.src.dynamo_version import crate_version

pytestmark = pytest.mark.skipif(
    not any(shutil.which(binary) for binary in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser")),
    reason="no headless Chrome available",
)

DEFAULTS = ["dynamo"]
DEFAULT_SLOTS = [DEFAULTS[0], None, None]
DYNAMO_V2_VERSION = crate_version(Path(__file__).resolve().parents[3] / "parsers/v2/Cargo.toml")


def assert_semver_descending(labels):
    versions = [tuple(map(int, re.search(r"\d+(?:\.\d+)+", label).group().split("."))) for label in labels]
    assert versions == sorted(versions, reverse=True)


def assert_dynamo_order(labels):
    v2 = [label for label in labels if label.startswith("v2 Rust ")]
    v1 = [label for label in labels if label.startswith("v1 Rust ")]
    assert labels == v2 + v1
    assert_semver_descending(v2)
    assert_semver_descending(v1)


def assert_language_order(labels):
    python = [label for label in labels if label.startswith("Python ")]
    rust = [label for label in labels if label.startswith("Rust ")]
    assert python and rust
    assert labels == python + rust
    assert_semver_descending(python)
    assert_semver_descending(rust)


@pytest.fixture(scope="module")
def matrix_driver():
    options = Options()
    for argument in ("--headless=new", "--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage", "--window-size=2400,1200"):
        options.add_argument(argument)
    driver = webdriver.Chrome(options=options)
    yield driver
    driver.quit()


def load(driver, rendered_page: Path, query=""):
    driver.get(rendered_page.resolve().as_uri() + query)
    WebDriverWait(driver, 20).until(lambda browser: browser.find_elements(By.CSS_SELECTOR, ".tab-panel.active .capture-slots"))
    WebDriverWait(driver, 20).until(lambda browser: browser.find_elements(By.CSS_SELECTOR, ".tab-panel.active table[data-parity-table]"))


def test_structural_response_popup_shows_generation_constraints(matrix_driver, rendered_page):
    driver = matrix_driver
    load(driver, rendered_page)
    try:
        host = driver.execute_script("""
            return [...document.querySelectorAll('#tab-unified td.cell[data-family="gpt_oss"]')]
                .find(cell => (window.__conformanceCellModel(cell) || {}).case_id ===
                    'UNIFIED.36-structured_response_after_reasoning');
        """)
        assert host is not None
        driver.execute_script("arguments[0].scrollIntoView({block:'center',inline:'center'})", host)
        host.send_keys(Keys.ENTER)
        popup = WebDriverWait(driver, 10).until(
            lambda browser: browser.find_element(By.CSS_SELECTOR, "body > .ttip-pinned"))
        config = next(node for node in popup.find_elements(By.CSS_SELECTOR, ".ttip-config code")
                      if node.get_attribute("textContent").startswith("structural_tag="))
        request = json.loads(config.get_attribute("textContent").removeprefix("structural_tag="))
        assert request["schema"]["additionalProperties"] is False
        assert request["schema"]["required"] == ["location", "temperature", "conditions"]
        assert len(request["rejected"]) == 6
        assert "[object Object]" not in config.text
        assert driver.execute_script("return getComputedStyle(arguments[0]).whiteSpace", config) == "pre-wrap"
        popup.find_element(By.CSS_SELECTOR, ".ttip-close").click()
    finally:
        load(driver, rendered_page)


def selection(driver):
    return driver.execute_script("const s=document.querySelector('.tab-panel.active')._comparisonSelection;return {keys:s.keys,reference:s.reference}")


def wait_for_popup(driver):
    return WebDriverWait(driver, 10).until(lambda browser: browser.find_element(By.CSS_SELECTOR, ".tab-panel.active .ttip-visible"))


def assert_popup_closed(driver, popup, owner):
    assert driver.execute_script("""
        return {visible: arguments[0].classList.contains('ttip-visible'),
            ownerOpen: arguments[1].classList.contains('ttip-open'),
            pinned: !!document.querySelector('.ttip-pinned'),
            modal: document.body.classList.contains('ttip-pin-mode'),
            focused: document.activeElement === arguments[1]};
    """, popup, owner) == {"visible": False, "ownerOpen": False, "pinned": False, "modal": False, "focused": True}


def choose_slot(driver, slot, engine, capture=None, *, close=True):
    driver.find_element(By.CSS_SELECTOR, f".capture-slot [data-slot-open='{slot}']").click()
    picker = driver.find_element(By.CSS_SELECTOR, ".comparison-picker[open]")
    driver.find_element(By.CSS_SELECTOR, f"[data-picker-engine-choice='{engine}']").click()
    if engine != "none":
        capture_select = driver.find_element(By.CSS_SELECTOR, f"[data-slot-capture='{slot}']")
        if capture is not None:
            Select(capture_select).select_by_value(capture)
        elif close:
            capture_select.send_keys(Keys.ESCAPE)
    return picker


def open_version_picker(driver, slot, engine):
    driver.find_element(By.CSS_SELECTOR, f"[data-slot-open='{slot}']").click()
    driver.find_element(By.CSS_SELECTOR, f"[data-picker-engine-choice='{engine}']").click()
    capture_select = driver.find_element(By.CSS_SELECTOR, f"[data-slot-capture='{slot}']")
    WebDriverWait(driver, 5).until(lambda browser: browser.execute_script("return document.activeElement === arguments[0]", capture_select))
    return capture_select


def test_capture_legend_matches_segments_in_both_themes_and_views(matrix_driver, rendered_page):
    d = matrix_driver
    load(d, rendered_page)
    legend_matches = """
        const actual = [...new Set([...document.querySelectorAll('#tab-unified td.cell .matrix-status')]
            .filter(node => node.getClientRects().length).map(node => node.className.split('status-')[1]))].sort();
        const legend = [...document.querySelectorAll('#tab-unified [data-legend-status]')]
            .map(node => node.dataset.legendStatus).sort();
        return JSON.stringify(actual) === JSON.stringify(legend);
    """
    try:
        for theme in ("light", "dark"):
            d.execute_script("document.documentElement.dataset.theme=arguments[0]", theme)
            for detailed, transpose in ((False, False), (False, True), (True, False), (True, True)):
                d.execute_script("""
                    for (const [selector, on] of [['[data-view-detailed]', arguments[0]], ['[data-transpose-toggle]', arguments[1]]]) {
                        const control = document.querySelector(selector);
                        if (control.checked !== on) { control.checked = on; control.dispatchEvent(new Event('change', {bubbles: true})); }
                    }
                """, detailed, transpose)
                legend = d.find_element(By.CSS_SELECTOR, "#tab-unified .capture-legend")
                assert legend.is_displayed()
                assert all(symbol not in legend.text for symbol in ("↯", "Δ", "†", "§", "‡"))
                assert "slot order" in legend.text
                assert "Not applicable or unsupported" in legend.text
                assert "Order/chunk not applicable for batch" not in legend.text
                assert "Dynamo v1 is the batch parser; Dynamo v2 details identify native UnifiedParser or split capture paths." in legend.text
                assert not legend.find_element(By.CSS_SELECTOR, "[data-output-formats]").is_displayed()
                assert not d.find_elements(By.CSS_SELECTOR, "#tab-unified .legend.summary-only,#tab-unified .legend.details-only")
                styles = d.execute_script("""
                    const segment = document.querySelector('#tab-unified td.cell .matrix-status');
                    const original = segment.className;
                    const styles = [];
                    try {
                        for (const swatch of document.querySelectorAll('#tab-unified .capture-legend .matrix-status')) {
                            segment.className = swatch.className;
                            const actual = getComputedStyle(segment), key = getComputedStyle(swatch);
                            styles.push({status: swatch.className.split('status-')[1],
                                matches: actual.backgroundColor === key.backgroundColor && actual.backgroundImage === key.backgroundImage,
                                width: swatch.getBoundingClientRect().width});
                        }
                    } finally { segment.className = original; }
                    return styles;
                """)
                assert d.execute_script(legend_matches)
                # The legend follows visible captures; a passing selection need not diverge.
                assert styles
                assert {item["status"] for item in styles} <= {"pass", "unsupported", "diverge", "error"}
                assert all(item["matches"] and item["width"] > 0 for item in styles)
        choose_slot(d, 1, "vllm", "vllm_python_batch@0.26.0")
        assert d.execute_script(legend_matches)
        format_note = d.find_element(By.CSS_SELECTOR, "[data-output-formats]")
        assert format_note.is_displayed()
        assert d.find_element(By.CSS_SELECTOR, "[data-batch-timing]").is_displayed()
        assert "calls=[...]" in format_note.text and "tool_call=..." in format_note.text
        assert "Both are compared as ordered events" in format_note.text
        choose_slot(d, 0, "dynamo", "dynamo_v1@10.0.1")
        assert not format_note.is_displayed()
        choose_slot(d, 0, "dynamo", "dynamo")
        assert format_note.is_displayed()
        choose_slot(d, 1, "none")
        assert not format_note.is_displayed()
        assert not d.find_element(By.CSS_SELECTOR, "[data-batch-timing]").is_displayed()
        # Engine/mode notes must update even when the visible status palette is unchanged.
        choose_slot(d, 0, "vllm", "vllm_rust@0.26.0")
        assert not d.find_element(By.CSS_SELECTOR, "[data-dynamo-parsers]").is_displayed()
        assert not d.find_element(By.CSS_SELECTOR, "[data-batch-timing]").is_displayed()
        choose_slot(d, 0, "vllm", "vllm_python@0.31.0")
        assert not d.find_element(By.CSS_SELECTOR, "[data-dynamo-parsers]").is_displayed()
        assert d.find_element(By.CSS_SELECTOR, "[data-batch-timing]").is_displayed()
        assert not format_note.is_displayed()
        choose_slot(d, 0, "sglang", "sglang_python@0.5.21")
        assert not d.find_element(By.CSS_SELECTOR, "[data-dynamo-parsers]").is_displayed()
        assert not d.find_element(By.CSS_SELECTOR, "[data-batch-timing]").is_displayed()
        choose_slot(d, 0, "dynamo", "dynamo")
        assert d.find_element(By.CSS_SELECTOR, "[data-dynamo-parsers]").is_displayed()
        assert d.execute_script(legend_matches)
        d.execute_script("document.querySelector('[data-transpose-toggle]').click()")
        d.execute_script("document.querySelectorAll('[data-col-toggle][aria-pressed=true]').forEach(button=>button.click())")
        assert not d.find_elements(By.CSS_SELECTOR, "#tab-unified [data-legend-status]")
        assert d.execute_script(legend_matches)
        d.execute_script("document.querySelector('[data-col-toggle]').click()")
        assert d.execute_script(legend_matches)
    finally:
        load(d, rendered_page)


def test_unified_report_keeps_the_original_table_and_hides_its_single_tab(matrix_driver, rendered_page):
    d = matrix_driver
    load(d, rendered_page, "?tab=tab-toolcalling-batch")
    assert d.execute_script("return [...document.querySelectorAll('.tab-panel')].map(node=>node.id)") == ["tab-unified"]
    assert not d.find_elements(By.CSS_SELECTOR, ".tab-button")
    assert d.find_element(By.CSS_SELECTOR, ".tab-panel.active table[data-parity-table]").is_displayed()
    assert not d.find_elements(By.CSS_SELECTOR, ".comparison-matrix,.engine-score,.category-chips,.comparison-legend")
    assert d.find_element(By.CSS_SELECTOR, "[data-view-detailed]").is_displayed()
    assert d.find_element(By.CSS_SELECTOR, "[data-transpose-toggle]").is_displayed()


def test_three_capture_slots_drive_the_original_report(matrix_driver, rendered_page):
    d = matrix_driver
    load(d, rendered_page)
    assert selection(d) == {"keys": DEFAULTS, "reference": None}
    assert len(d.find_elements(By.CSS_SELECTOR, ".capture-slot [data-slot-capture]")) == 1
    assert d.find_element(By.CSS_SELECTOR, ".capture-slot [data-slot-capture='0']").get_attribute("value") == DEFAULTS[0]
    assert d.find_element(By.CSS_SELECTOR, ".capture-slot [data-slot-capture='0']").get_attribute("aria-label") == f"Choose capture for comparison slot 1: Dynamo, v2 Rust {DYNAMO_V2_VERSION} [unpublished] (stream, UnifiedParser)"
    assert not d.find_elements(By.CSS_SELECTOR, ".capture-slot [data-slot-capture='0'] option[value='']")
    assert [element.text for element in d.find_elements(By.CSS_SELECTOR, ".capture-slot [data-slot-engine]")] == ["Dynamo", "", ""]
    assert not d.find_elements(By.CSS_SELECTOR, "[data-slot-capture] option[value='']")
    Select(d.find_element(By.CSS_SELECTOR, "[data-slot-capture='0']")).select_by_value("dynamo_v1@9.2.10")
    assert selection(d)["keys"] == ["dynamo_v1@9.2.10"]
    assert not d.find_elements(By.CSS_SELECTOR, ".comparison-picker[open]")
    Select(d.find_element(By.CSS_SELECTOR, "[data-slot-capture='0']")).select_by_value(DEFAULTS[0])
    assert selection(d)["keys"] == DEFAULTS

    choose_slot(d, 0, "dynamo", "dynamo_v1@9.2.10")
    assert selection(d)["keys"] == ["dynamo_v1@9.2.10"]
    assert d.find_element(By.CSS_SELECTOR, ".capture-slot [data-slot-engine='0']").text.upper() == "DYNAMO"
    assert d.find_element(By.CSS_SELECTOR, ".tab-panel.active table[data-parity-table]").is_displayed()
    checked = d.execute_script("return [...document.querySelectorAll('#tab-unified .cmp-on:checked')].map(input=>input.value)")
    assert sorted(checked) == sorted(selection(d)["keys"])
    detailed = d.find_element(By.CSS_SELECTOR, "[data-view-detailed]")
    detailed.click()
    assert selection(d)["keys"] == ["dynamo_v1@9.2.10"]
    checked = d.execute_script("return [...document.querySelectorAll('#tab-unified .cmp-on:checked')].map(input=>input.value)")
    assert sorted(checked) == sorted(selection(d)["keys"])
    detailed.click()
    assert selection(d)["keys"] == ["dynamo_v1@9.2.10"]


def test_default_follows_native_v2_version_and_reset(matrix_driver, rendered_page):
    d = matrix_driver
    script = d.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {"source": """
      Object.defineProperty(window, '__CONFORMANCE_PAGE', {
        configurable: true,
        set(page) {
          const candidate = page.tabs.find(tab => tab.id === 'tab-unified')
            .candidates.find(item => item.key === 'dynamo');
          candidate.label = candidate.label.replace(candidate.version, '0.7.100');
          candidate.version = '0.7.100';
          Object.defineProperty(window, '__CONFORMANCE_PAGE', {value: page, writable: true});
        }
      });
    """})
    try:
        load(d, rendered_page)
        assert selection(d)["keys"] == ["dynamo"]
        assert d.find_element(By.CSS_SELECTOR, "[data-slot-capture='0'] option:checked").text.startswith("v2 Rust 0.7.100 ")
        assert [node.text for node in d.find_elements(By.CSS_SELECTOR, "[data-slot-engine]")] == ["Dynamo", "", ""]
        choose_slot(d, 1, "sglang")
        assert len(selection(d)["keys"]) == 2
        assert not d.find_elements(By.CSS_SELECTOR, "[data-slot-capture] option[value='']")
        d.find_element(By.CSS_SELECTOR, "[data-reset]").click()
        WebDriverWait(d, 10).until(lambda browser: selection(browser)["keys"] == ["dynamo"])
        assert d.find_element(By.CSS_SELECTOR, "[data-slot-capture='0'] option:checked").text.startswith("v2 Rust 0.7.100 ")
        assert [node.text for node in d.find_elements(By.CSS_SELECTOR, "[data-slot-engine]")] == ["Dynamo", "", ""]
        assert not d.execute_script("return new URL(location.href).searchParams.has('slots_tab-unified')")
    finally:
        d.execute_cdp_cmd("Page.removeScriptToEvaluateOnNewDocument", {"identifier": script["identifier"]})
        load(d, rendered_page)


def test_same_engine_versions_none_and_duplicate_prevention(matrix_driver, rendered_page):
    d = matrix_driver
    load(d, rendered_page)
    choose_slot(d, 0, "dynamo", "dynamo_v1@9.2.10")
    choose_slot(d, 1, "dynamo", "dynamo_v1@9.2.7")
    assert selection(d)["keys"][:2] == ["dynamo_v1@9.2.10", "dynamo_v1@9.2.7"]
    d.find_element(By.CSS_SELECTOR, "[data-slot-open='1']").click()
    picker = d.find_element(By.CSS_SELECTOR, ".comparison-picker[open]")
    picker.find_element(By.CSS_SELECTOR, "[data-picker-engine-choice='dynamo']").click()
    assert not d.find_elements(By.CSS_SELECTOR, ".comparison-picker[open]")
    assert not d.find_elements(By.CSS_SELECTOR, "[data-slot-capture] option[value='']")
    duplicate = d.find_element(By.CSS_SELECTOR, "[data-slot-capture='1'] option[value='dynamo_v1@9.2.10']")
    assert not duplicate.is_enabled()
    d.find_element(By.CSS_SELECTOR, "[data-slot-capture='1']").send_keys(Keys.ESCAPE)
    d.find_element(By.CSS_SELECTOR, "[data-slot-open='1']").click()
    picker = d.find_element(By.CSS_SELECTOR, ".comparison-picker[open]")
    picker.find_element(By.CSS_SELECTOR, "[data-picker-engine-choice='none']").click()
    assert selection(d)["keys"] == ["dynamo_v1@9.2.10"]
    assert d.find_element(By.CSS_SELECTOR, ".capture-slot [data-slot-engine='1']").text == ""


def test_primary_slot_cannot_be_empty_and_engine_change_selects_a_version(matrix_driver, rendered_page):
    d = matrix_driver
    load(d, rendered_page, "?tab=tab-unified&slots_tab-unified=%2C%2C")
    assert selection(d)["keys"] == DEFAULTS
    d.find_element(By.CSS_SELECTOR, "[data-slot-open='0']").click()
    picker = d.find_element(By.CSS_SELECTOR, ".comparison-picker[open]")
    picker.find_element(By.CSS_SELECTOR, "[data-picker-engine-choice='sglang']").click()
    capture_select = d.find_element(By.CSS_SELECTOR, "[data-slot-capture='0']")
    assert not d.find_elements(By.CSS_SELECTOR, ".comparison-picker[open]")
    options = [option.text for option in capture_select.find_elements(By.TAG_NAME, "option") if option.get_attribute("value")]
    assert_semver_descending(options)
    assert all(label.startswith("Python ") for label in options)
    selected = next(option.get_attribute("value") for option in capture_select.find_elements(By.TAG_NAME, "option")
                    if option.get_attribute("value") and option.get_attribute("value") != capture_select.get_attribute("value"))
    Select(capture_select).select_by_value(selected)
    selected = selection(d)["keys"]
    assert len(selected) == 1 and selected[0].startswith("sglang_python")
    assert capture_select.get_attribute("value") == selected[0]


def test_none_slot_card_and_picker_wait_for_an_engine_before_showing_versions(matrix_driver, rendered_page):
    d = matrix_driver
    load(d, rendered_page)
    assert not d.execute_script("return new URL(location.href).searchParams.has('slots_tab-unified')")
    choose_slot(d, 2, "none")

    card = d.find_element(By.CSS_SELECTOR, ".capture-slot[data-engine='none']:has([data-slot-open='2'])")
    trigger = card.find_element(By.CSS_SELECTOR, "[data-slot-open='2']")
    assert trigger.get_attribute("aria-label") == "Edit comparison slot 3: None"
    assert trigger.text == "—"
    assert card.find_element(By.CSS_SELECTOR, "[data-slot-engine='2']").text == ""
    assert not card.find_elements(By.CSS_SELECTOR, "[data-slot-none]")
    geometry = d.execute_script("""
      const none = arguments[0].getBoundingClientRect();
      const active = document.querySelector('.capture-slot[data-engine="dynamo"]').getBoundingClientRect();
      return [none.width, none.height, active.width, active.height];
    """, card)
    assert abs(geometry[0] - geometry[2]) <= 2
    assert abs(geometry[1] - geometry[3]) <= 2 and geometry[1] < 120
    slot_styles = d.execute_script("""
      return [...document.querySelectorAll('.capture-slot')].map(slot => {
        const badge = slot.querySelector('.capture-slot-badge');
        const label = slot.querySelector('.capture-slot-engine');
        const select = slot.querySelector('.capture-slot-select');
        const rect = badge.getBoundingClientRect();
        return {badge: [rect.width, rect.height], badgeFont: getComputedStyle(badge).fontSize,
          labelFont: getComputedStyle(label).fontSize,
          selectFont: select ? getComputedStyle(select).fontSize : null};
      });
    """)
    assert all(style["badge"] == [32, 32] and style["badgeFont"] == "16px" for style in slot_styles)
    assert all(style["labelFont"] == "18px" for style in slot_styles)
    assert slot_styles[0]["selectFont"] == "16px"
    theme_colors = d.execute_script("""
      const root = document.documentElement;
      const original = root.getAttribute('data-theme');
      const slot = arguments[0];
      const trigger = slot.querySelector('.capture-slot-engine-trigger');
      const styles = {};
      for (const theme of ['light', 'dark']) {
        root.setAttribute('data-theme', theme);
        styles[theme] = {
          card: getComputedStyle(slot).backgroundColor,
          button: getComputedStyle(trigger).backgroundColor,
          buttonText: getComputedStyle(trigger).color,
          badge: getComputedStyle(slot.querySelector('.capture-slot-badge')).backgroundColor,
          badgeText: getComputedStyle(slot.querySelector('.capture-slot-badge')).color
        };
      }
      if (original === null) root.removeAttribute('data-theme');
      else root.setAttribute('data-theme', original);
      return styles;
    """, card)
    assert theme_colors == {
        "light": {"card": "rgba(0, 0, 0, 0)", "button": "rgba(0, 0, 0, 0)", "buttonText": "rgb(17, 24, 39)", "badge": "rgb(241, 243, 246)", "badgeText": "rgb(100, 116, 139)"},
        "dark": {"card": "rgba(0, 0, 0, 0)", "button": "rgba(0, 0, 0, 0)", "buttonText": "rgb(232, 237, 243)", "badge": "rgb(89, 101, 117)", "badgeText": "rgb(255, 255, 255)"},
    }

    trigger.click()
    picker = d.find_element(By.CSS_SELECTOR, ".comparison-picker[open]")
    engine_buttons = picker.find_elements(By.CSS_SELECTOR, "[data-picker-engine-choice]")
    assert [button.get_attribute("data-picker-engine-choice") for button in engine_buttons] == ["dynamo", "sglang", "vllm", "none"]
    picker.find_element(By.CSS_SELECTOR, "[data-picker-engine-choice='sglang']").click()
    assert not d.find_elements(By.CSS_SELECTOR, ".comparison-picker[open]")
    assert d.find_element(By.CSS_SELECTOR, "[data-slot-capture='2'] option:checked").text.startswith("Python ")
    d.find_element(By.CSS_SELECTOR, "[data-slot-open='2']").click()
    picker = d.find_element(By.CSS_SELECTOR, ".comparison-picker[open]")
    picker.find_element(By.CSS_SELECTOR, "[data-picker-engine-choice='none']").click()
    assert not d.find_elements(By.CSS_SELECTOR, ".comparison-picker[open]")
    assert d.execute_script("return new URL(location.href).searchParams.get('slots_tab-unified')") is None
    d.refresh()
    WebDriverWait(d, 10).until(lambda browser: browser.find_element(By.CSS_SELECTOR, ".capture-slot[data-engine='none'] [data-slot-open='2']"))
    assert selection(d) == {"keys": DEFAULTS, "reference": None}
    assert d.find_element(By.CSS_SELECTOR, ".capture-slot[data-engine='none'] [data-slot-open='2']").get_attribute("aria-label") == "Edit comparison slot 3: None"

    load(d, rendered_page, "?slots_tab-unified=dynamo_v1_jail%4010.0.1%2Cnone%2Cnone")
    assert d.execute_script("return new URL(location.href).searchParams.get('slots_tab-unified')") == "dynamo_v1_jail@10.0.1,none,none"
    assert d.find_element(By.CSS_SELECTOR, ".capture-slot[data-engine='none'] [data-slot-open='2']").get_attribute("aria-label") == "Edit comparison slot 3: None"
    d.set_window_size(640, 1000)
    try:
        geometry = d.execute_script("return [...document.querySelectorAll('.capture-slot')].map(slot => { const r=slot.getBoundingClientRect(); return {engine:slot.dataset.engine,x:r.x,y:r.y,width:r.width,height:r.height}; })")
        assert [slot["engine"] for slot in geometry] == ["dynamo", "none", "none"]
        assert max(slot["width"] for slot in geometry) - min(slot["width"] for slot in geometry) <= 2
        assert max(slot["height"] for slot in geometry) - min(slot["height"] for slot in geometry) <= 2
        assert max(slot["height"] for slot in geometry) < 120
        assert max(slot["x"] for slot in geometry) - min(slot["x"] for slot in geometry) <= 2
        assert geometry[0]["y"] < geometry[1]["y"] < geometry[2]["y"]
    finally:
        d.set_window_size(2400, 1200)


def test_dynamo_capture_labels_name_the_unified_batch_corpus(matrix_driver, rendered_page):
    d = matrix_driver
    load(d, rendered_page)
    option = d.find_element(By.CSS_SELECTOR, "[data-slot-capture='0'] option[value='dynamo_v1@10.0.1']")
    assert option.text == "v1 Rust 10.0.1 (batch)"
    assert "Dynamo v1 Rust parser" in option.get_attribute("title")
    assert "without a jail buffer" in option.get_attribute("title")


def test_engine_choice_opens_the_normal_sorted_version_select(matrix_driver, rendered_page):
    d = matrix_driver
    load(d, rendered_page)
    d.execute_script("""
      window.__showPickerSlots = [];
      const showPicker = HTMLSelectElement.prototype.showPicker;
      HTMLSelectElement.prototype.showPicker = function() {
        window.__showPickerSlots.push(this.dataset.slotCapture);
        if (showPicker) showPicker.call(this);
      };
    """)
    for slot in range(3):
        d.find_element(By.CSS_SELECTOR, f"[data-slot-open='{slot}']").click()
        picker = d.find_element(By.CSS_SELECTOR, ".comparison-picker[open]")
        assert picker.find_element(By.TAG_NAME, "h2").text == f"Comparison slot {slot + 1}"
        choices = [button.get_attribute("data-picker-engine-choice") for button in picker.find_elements(By.CSS_SELECTOR, "[data-picker-engine-choice]")]
        assert choices == ["dynamo", "sglang", "vllm"] + (["none"] if slot else [])
        geometry = d.execute_script("const p=arguments[0],r=p.getBoundingClientRect(),s=getComputedStyle(p.querySelector('button'));return [r.width,r.height,innerWidth,innerHeight,r.left,r.right,r.top,r.bottom,parseFloat(s.fontSize)]", picker)
        assert geometry[0] <= 620 and geometry[1] < 180 and geometry[8] == 16
        assert 0 <= geometry[4] < geometry[5] <= geometry[2]
        assert 0 <= geometry[6] < geometry[7] <= geometry[3]
        picker.send_keys(Keys.ESCAPE)
    assert selection(d)["keys"] == DEFAULTS

    dynamo = open_version_picker(d, 0, "dynamo")
    assert d.execute_script("return window.__showPickerSlots") == ["0"]
    labels = {option.get_attribute("value"): option.text for option in dynamo.find_elements(By.TAG_NAME, "option")}
    assert labels["dynamo@0.6.1"].startswith("v2 Rust 0.6.1 ")
    assert labels["dynamo_v1@9.2.10"] == "v1 Rust 9.2.10 (batch)"
    assert labels[DEFAULTS[0]].startswith(f"v2 Rust {DYNAMO_V2_VERSION} ")
    assert labels["dynamo_v1_jail@10.0.1"] == "v1 Rust 10.0.1 (batch + jail-buffer)"
    assert labels["dynamo_v1@10.0.1"] == "v1 Rust 10.0.1 (batch)"
    jail_option = dynamo.find_element(By.CSS_SELECTOR, "option[value='dynamo_v1_jail@10.0.1']")
    assert "jail-buffer producer" in jail_option.get_attribute("title")
    assert "without a jail buffer" not in jail_option.get_attribute("title")
    listed_keys = set(labels)
    assert "dynamo_v1_jail@3.0.0" not in listed_keys
    assert "dynamo_v1_jail@9.2.13" not in listed_keys
    assert "dynamo_v1@9.2.13" not in labels
    assert selection(d)["keys"] == DEFAULTS
    assert_dynamo_order(list(labels.values()))
    Select(dynamo).select_by_value("dynamo_v1@9.2.10")
    assert not d.find_elements(By.CSS_SELECTOR, ".comparison-picker[open]")
    assert selection(d)["keys"] == ["dynamo_v1@9.2.10"]

    vllm = open_version_picker(d, 1, "vllm")
    labels_by_key = {option.get_attribute("value"): option.text for option in vllm.find_elements(By.TAG_NAME, "option")}
    assert_language_order([label for key, label in labels_by_key.items() if key])
    assert all(label.startswith(("Python ", "Rust ")) for key, label in labels_by_key.items() if key)
    assert vllm.get_attribute("value") == "vllm_python@0.31.0"
    assert labels_by_key["vllm_python@0.31.0"] == "Python 0.31.0 (batch, Combined)"
    assert labels_by_key["vllm_rust@0.26.0"] == "Rust 0.26.0 (stream, Combined & Unified)"
    Select(vllm).select_by_value("vllm_python_batch@0.26.0")
    assert selection(d)["keys"] == ["dynamo_v1@9.2.10", "vllm_python_batch@0.26.0"]
    d.find_element(By.CSS_SELECTOR, "[data-slot-open='1']").click()
    picker = d.find_element(By.CSS_SELECTOR, ".comparison-picker[open]")
    picker.find_element(By.CSS_SELECTOR, "[data-picker-engine-choice='none']").click()
    assert selection(d)["keys"] == ["dynamo_v1@9.2.10"]


def test_legacy_links_are_capped_and_picker_stays_anchored(matrix_driver, rendered_page):
    d = matrix_driver
    keys = ["dynamo_v1@9.2.10", "dynamo_v1@9.2.7", "sglang_python_batch@0.5.14", "vllm_python_batch@0.26.0"]
    query = "?tab=tab-unified&base_tab-unified=" + quote(keys[0]) + "&cmp_tab-unified=" + quote(",".join(keys[1:]))
    load(d, rendered_page, query)
    assert selection(d)["keys"] == keys[:3]
    assert "more than three captures" in d.find_element(By.CSS_SELECTOR, ".capture-slots-notice").text

    trigger = d.find_element(By.CSS_SELECTOR, ".capture-slot [data-slot-open='2']")
    trigger.click()
    picker = d.find_element(By.CSS_SELECTOR, ".comparison-picker[open]")
    geometry = d.execute_script("const a=arguments[0].getBoundingClientRect(),p=arguments[1].getBoundingClientRect();return {left:p.left,right:p.right,top:p.top,bottom:p.bottom,w:innerWidth,h:innerHeight,gap:Math.min(Math.abs(a.right-p.left),Math.abs(p.right-a.left))}", trigger, picker)
    assert geometry["left"] >= 0 and geometry["top"] >= 0
    assert geometry["right"] <= geometry["w"] and geometry["bottom"] <= geometry["h"]
    assert geometry["gap"] <= 9
    picker.send_keys(Keys.ESCAPE)
    assert d.execute_script("return document.activeElement.dataset.slotOpen") == "2"

    trigger.click()
    picker = d.find_element(By.CSS_SELECTOR, ".comparison-picker[open]")
    d.execute_script("arguments[0].dispatchEvent(new MouseEvent('click',{bubbles:true}))", picker)
    assert not d.find_elements(By.CSS_SELECTOR, ".comparison-picker[open]")
    assert d.execute_script("return document.activeElement.dataset.slotOpen") == "2"


@pytest.mark.parametrize("legacy_query", [
    "cmp_tab-unified=vllm_rust",
    "base_tab-unified=removed-parser&cmp_tab-unified=vllm_rust",
    "base_tab-unified=&cmp_tab-unified=vllm_rust",
])
def test_legacy_compare_keeps_default_primary_when_base_is_missing_or_invalid(matrix_driver, rendered_page, legacy_query):
    d = matrix_driver
    load(d, rendered_page, "?tab=tab-unified&" + legacy_query)
    assert selection(d)["keys"] == [DEFAULTS[0], "vllm_rust"]
    assert d.find_element(By.CSS_SELECTOR, "[data-slot-capture='0']").get_attribute("value") == DEFAULTS[0]
    assert d.execute_script("return new URL(location.href).searchParams.get('slots_tab-unified')") == "dynamo,vllm_rust,none"
    d.refresh()
    WebDriverWait(d, 20).until(lambda browser: browser.find_elements(By.CSS_SELECTOR, ".tab-panel.active .capture-slots"))
    assert selection(d)["keys"] == [DEFAULTS[0], "vllm_rust"]


def test_restored_empty_primary_owns_default_without_duplicate(matrix_driver, rendered_page):
    d = matrix_driver
    load(d, rendered_page, "?slots_tab-unified=%2Cdynamo%2C")
    assert selection(d)["keys"] == DEFAULTS
    assert d.find_element(By.CSS_SELECTOR, "[data-slot-engine='1']").text == ""


def test_narrow_picker_opens_the_normal_version_select(matrix_driver, rendered_page):
    d = matrix_driver
    d.set_window_size(430, 700)
    try:
        load(d, rendered_page)
        choose_slot(d, 2, "none")
        d.find_element(By.CSS_SELECTOR, "[data-slot-open='2']").click()
        picker = d.find_element(By.CSS_SELECTOR, ".comparison-picker[open]")
        picker.find_element(By.CSS_SELECTOR, "[data-picker-engine-choice='dynamo']").click()
        assert not d.find_elements(By.CSS_SELECTOR, ".comparison-picker[open]")
        capture_select = d.find_element(By.CSS_SELECTOR, "[data-slot-capture='2']")
        assert d.execute_script("return document.activeElement === arguments[0]", capture_select)
        options = [option.text for option in capture_select.find_elements(By.TAG_NAME, "option") if option.get_attribute("value")]
        assert_dynamo_order(options)
    finally:
        d.set_window_size(2400, 1200)


@pytest.mark.parametrize("close_over_control,viewport_width", [(False, 2400), (True, 2400), (True, 2300)])
def test_segments_click_only_popup_keyboard_and_immediate_selection(matrix_driver, rendered_page, close_over_control, viewport_width):
    d = matrix_driver
    d.set_window_size(viewport_width, 1200)
    try:
        load(d, rendered_page)
        cell = d.find_element(By.CSS_SELECTOR, "#tab-unified td.cell[data-cmp][data-ttip-id]")
        assert len(cell.find_elements(By.CSS_SELECTOR, ".matrix-status")) == 1
        d.execute_script("arguments[0].dispatchEvent(new PointerEvent('pointerover',{bubbles:true}));arguments[0].focus()", cell)
        assert not cell.find_elements(By.CSS_SELECTOR, ".ttip-built")
        assert not d.find_elements(By.CSS_SELECTOR, ".ttip-visible")
        cell.send_keys(Keys.ENTER)
        popup = WebDriverWait(d, 10).until(lambda b: b.find_element(By.CSS_SELECTOR, "body > .ttip-visible"))
        close_styles = d.execute_script("""
            const root = document.documentElement, original = root.dataset.theme;
            const popup = arguments[0], close = popup.querySelector('.ttip-close');
            const styles = ['light', 'dark'].map(theme => {
                root.dataset.theme = theme;
                return {color: getComputedStyle(close).color, ink: getComputedStyle(popup).color,
                        weight: getComputedStyle(close).fontWeight};
            });
            root.dataset.theme = original;
            return styles;
        """, popup)
        assert all(style["color"] == style["ink"] and int(style["weight"]) >= 600 for style in close_styles)
        assert d.execute_script("""
            const backdrop = getComputedStyle(document.body, '::after');
            return backdrop.content !== 'none' && backdrop.backgroundColor === 'rgba(0, 0, 0, 0.18)'
                && Number(backdrop.zIndex) < Number(getComputedStyle(arguments[0]).zIndex)
                && backdrop.pointerEvents === 'none';
        """, popup)
        bounds = d.execute_script("const r=arguments[0].getBoundingClientRect();return [r.width,r.height,r.left,r.top,innerWidth,innerHeight]", popup)
        assert bounds[0] <= bounds[4] * .97 + 1
        assert bounds[1] <= bounds[5] * .97 + 1
        assert abs(bounds[2] - (bounds[4] - bounds[0]) / 2) <= 1
        assert abs(bounds[3] - (bounds[5] - bounds[1]) / 2) <= 1
        d.find_element(By.TAG_NAME, "body").send_keys(Keys.ESCAPE)
        assert_popup_closed(d, popup, cell)
        assert d.execute_script("return getComputedStyle(document.body, '::after').content") == "none"
        for slot, engine, capture in ((1, "sglang", "sglang_python_batch@0.5.14"), (2, "vllm", "vllm_python_batch@0.26.0")):
            choose_slot(d, slot, engine, capture)
            segments = cell.find_elements(By.CSS_SELECTOR, ".matrix-status")
            assert len(segments) == slot + 1
            widths = d.execute_script("return [...arguments[0].querySelectorAll('.matrix-status')].map(n=>n.getBoundingClientRect().width)", cell)
            assert max(widths) - min(widths) <= 1
            assert [segment.get_attribute("data-capture") for segment in segments] == selection(d)["keys"]
        cell.click()
        popup = d.find_element(By.CSS_SELECTOR, "body > .ttip-visible")
        capture_select = d.find_element(By.CSS_SELECTOR, "[data-slot-capture='0']")
        Select(capture_select).select_by_value("dynamo_v1@9.2.10")
        assert d.execute_script("return document.activeElement===arguments[0]", capture_select)
        assert selection(d)["keys"][0] == "dynamo_v1@9.2.10"
        orders = d.execute_script("""
            return [...arguments[0].querySelectorAll('.ttip-chunks')].map(table =>
                [...table.querySelectorAll('th[data-cand]:not(.col-hidden)')].map(node => node.dataset.cand));
        """, popup)
        expected_tables = d.execute_script("""
            const model = window.__conformanceCellModel(arguments[0]);
            return model.variants ? model.variants.length : 1;
        """, cell)
        assert len(orders) == expected_tables
        assert all("dynamo_v1@9.2.10" in order and DEFAULTS[0] not in order for order in orders)
        assert all(order == selection(d)["keys"] for order in orders)
        Select(capture_select).select_by_value("dynamo_v1@9.2.7")
        assert popup.is_displayed()
        assert selection(d)["keys"][0] == "dynamo_v1@9.2.7"
        close = popup.find_element(By.CSS_SELECTOR, ".ttip-close")
        if close_over_control:
            geometry = d.execute_script("""
                const close = arguments[0].getBoundingClientRect(), x = close.x + close.width / 2;
                const control = [...document.querySelectorAll('.tab-panel.active [data-col-toggle]')].find(node => {
                    const r = node.getBoundingClientRect(); return r.width && r.height && r.left + r.width / 2 >= x;
                });
                if (!control) return null;
                // The close button can land in a gap between controls. Pan the
                // matrix to place a real control beneath it before testing clicks.
                const target = control.getBoundingClientRect();
                window.scrollBy(target.x + target.width / 2 - x, 0);
                const r = control.getBoundingClientRect();
                return {control, height: outerHeight + 2 * (r.y + r.height / 2 - close.y - close.height / 2)};
            """, close)
            assert geometry is not None, "active matrix has no column control to align beneath the popup close button"
            d.set_window_size(viewport_width, round(geometry["height"]))
            WebDriverWait(d, 5).until(lambda browser: browser.execute_script("""
                const c = arguments[0].getBoundingClientRect(), r = arguments[1].getBoundingClientRect();
                const x = c.x + c.width / 2, y = c.y + c.height / 2;
                return r.left < x && x < r.right && r.top < y && y < r.bottom
                    && arguments[0].contains(document.elementFromPoint(x, y));
            """, close, geometry["control"]))
        close.click()
        assert_popup_closed(d, popup, cell)
        assert d.execute_script("return getComputedStyle(document.body, '::after').content") == "none"
        if close_over_control:
            WebDriverWait(d, 5).until(lambda browser: browser.execute_script(
                "return arguments[0]._ttip.classList.contains('ttip-visible')", geometry["control"]))
            assert_popup_closed(d, popup, cell)
    finally:
        d.set_window_size(2400, 1200)
        load(d, rendered_page)


@pytest.mark.parametrize("detailed", [False, True])
def test_pinned_case_excludes_background_hover_tooltips(matrix_driver, rendered_page, detailed):
    d = matrix_driver
    load(d, rendered_page)
    try:
        if detailed:
            d.find_element(By.CSS_SELECTOR, "[data-view-detailed]").click()
        control = d.find_element(By.CSS_SELECTOR, "[data-col-toggle]")
        ActionChains(d).move_to_element(control).perform()
        assert d.find_elements(By.CSS_SELECTOR, ".column-control-tooltip.ttip-visible")
        host = d.execute_script("return [...document.querySelectorAll('#tab-unified td.cell[data-ttip-id]')].find(cell => cell.getClientRects().length)")
        host.send_keys(Keys.ENTER)
        popup = d.find_element(By.CSS_SELECTOR, "body > .ttip-pinned")
        assert d.find_elements(By.CSS_SELECTOR, ".ttip-visible") == [popup]
        ActionChains(d).move_to_element(control).perform()
        assert d.find_elements(By.CSS_SELECTOR, ".ttip-visible") == [popup]
        popup.find_element(By.CSS_SELECTOR, ".ttip-close").click()
        assert_popup_closed(d, popup, host)
    finally:
        load(d, rendered_page)


def test_unified_case_columns_have_no_repeated_vertical_rules(matrix_driver, rendered_page):
    d = matrix_driver
    load(d, rendered_page)
    result = d.execute_script("""
      const headers = [...document.querySelectorAll('#tab-unified th.case-sub')];
      const group = headers.map(header => header.dataset.colHideGroup)
        .find(key => headers.filter(header => header.dataset.colHideGroup === key).length > 1);
      const inner = headers.find((header, index) => index > 0
        && header.dataset.colHideGroup === group && !header.classList.contains('case-group-start'));
      const boundary = headers.find(header => header.classList.contains('case-group-start')
        && header.dataset.colHideGroup !== group);
      const cells = [...document.querySelectorAll('#tab-unified tbody td.cell[data-col-hide-group]')];
      const cell = cells.find(candidate => candidate.dataset.colHideGroup === group
        && !candidate.classList.contains('case-group-start'));
      const cellStart = cells.find(candidate => candidate.classList.contains('case-group-start'));
      return {
        innerRight: getComputedStyle(inner).borderRightWidth,
        headerBoundaryLeft: getComputedStyle(boundary).borderLeftWidth,
        headerBoundaryColor: getComputedStyle(boundary).borderLeftColor,
        cellLeft: getComputedStyle(cell).borderLeftWidth,
        cellRight: getComputedStyle(cell).borderRightWidth,
        cellStartLeft: getComputedStyle(cellStart).borderLeftWidth,
        segmentShadow: getComputedStyle(cell).boxShadow
      };
    """)
    assert result == {
        "innerRight": "0px",
        "headerBoundaryLeft": "1px",
        "headerBoundaryColor": "rgb(220, 225, 231)",
        "cellLeft": "0px",
        "cellRight": "0px",
        "cellStartLeft": "0px",
        "segmentShadow": "none",
    }
    dark_color = d.execute_script("""
      const root = document.documentElement;
      const original = root.getAttribute('data-theme');
      root.setAttribute('data-theme', 'dark');
      const header = document.querySelector('#tab-unified table[data-parity-table] th.case-sub.case-group-start');
      const color = getComputedStyle(header).borderLeftColor;
      if (original === null) root.removeAttribute('data-theme');
      else root.setAttribute('data-theme', original);
      return color;
    """)
    assert dark_color == "rgb(71, 85, 105)"


def test_transposed_table_keeps_group_rows_without_vertical_case_rules(matrix_driver, rendered_page):
    d = matrix_driver
    load(d, rendered_page, "?vertical=1")
    result = d.execute_script("""
      const cell = document.querySelector('#tab-unified .transpose-table td.cell.case-group-start');
      const groupRows = document.querySelectorAll('#tab-unified .transpose-table tr.section').length;
      const style = getComputedStyle(cell);
      return {groupRows, borderLeft: style.borderLeftWidth, borderRight: style.borderRightWidth};
    """)
    assert result["groupRows"] > 0
    assert result["borderLeft"] == result["borderRight"] == "0px"


@pytest.mark.parametrize("query,detailed", [("", True), ("?vertical=1", False), ("?vertical=1", True)])
def test_case_headings_and_cells_stay_click_only_in_all_views(matrix_driver, rendered_page, query, detailed):
    d = matrix_driver
    load(d, rendered_page, query)
    try:
        if detailed:
            d.find_element(By.CSS_SELECTOR, "[data-view-detailed]").click()
        for selector in ("td.cell[data-ttip-id]", "th[data-ttip-id]"):
            host = next(element for element in d.find_elements(By.CSS_SELECTOR, "#tab-unified " + selector) if element.is_displayed())
            # Closing the modal can expose a hover-enabled control under the pointer.
            ActionChains(d).move_to_element(host).perform()
            WebDriverWait(d, 5).until(lambda browser: not browser.find_elements(
                By.CSS_SELECTOR, ".column-control-tooltip.ttip-visible"))
            d.execute_script("arguments[0].dispatchEvent(new PointerEvent('pointerover',{bubbles:true}));arguments[0].focus()", host)
            assert not d.find_elements(By.CSS_SELECTOR, ".ttip-visible")
            assert not host.find_elements(By.CSS_SELECTOR, ".ttip-built")
            host.send_keys(Keys.SPACE)
            popup = WebDriverWait(d, 10).until(lambda b: b.find_element(By.CSS_SELECTOR, "body > .ttip-visible"))
            assert popup.is_displayed()
            d.find_element(By.TAG_NAME, "body").send_keys(Keys.ESCAPE)
            assert_popup_closed(d, popup, host)
            host.send_keys(Keys.SPACE)
            popup = d.find_element(By.CSS_SELECTOR, "body > .ttip-pinned")
            popup.find_element(By.CSS_SELECTOR, ".ttip-close").click()
            assert_popup_closed(d, popup, host)

    finally:
        load(d, rendered_page)


def test_status_palette_and_source_blocker_are_distinct(matrix_driver, rendered_page):
    d = matrix_driver
    load(d, rendered_page, "?slots_tab-unified=dynamo_v1%409.2.13%2Cvllm_python_batch%400.26.0%2C")
    machine_statuses = d.execute_script("""
        return [['missing_capture', 'This engine has no parser'],
                ['unsupported_family', 'No captured result'],
                ['unsupported_request_mode', 'No captured result'],
                ['scenario_not_applicable', 'No captured result'],
                ['duplicate_scenario', 'No captured result'],
                ['source_unavailable', 'No captured result']].map(([code, prose]) => {
            const cell = {kind:'cell', cmp:{candidate:{na:1}, golden:{sig:'oracle'}},
                tooltip:{candidates:[{key:'candidate', block:{unavailable_code:code, unavailable:prose}}]}};
            return window.__conformanceOutputStatus({id:'tab-unified'}, cell, {key:'candidate'}, 'golden');
        });
    """)
    assert machine_statuses == ["missing", "unsupported", "unsupported", "unsupported", "unsupported", "blocked"]
    colors_by_theme = {}
    for theme in ("light", "dark"):
        d.execute_script("document.documentElement.dataset.theme=arguments[0]", theme)
        colors_by_theme[theme] = d.execute_script("""
            const panel=document.querySelector('#tab-unified'), holder=document.createElement('div');
            panel.appendChild(holder);
            const statuses=['pass','documented','diverge','leak','error','expected','unsupported','missing','blocked','uncompared'];
            const colors=Object.fromEntries(statuses.map(status=>{
              const node=document.createElement('i');node.className='matrix-status status-'+status;holder.appendChild(node);
              const style=getComputedStyle(node);
              return [status,style.backgroundImage === 'none' ? style.backgroundColor : style.backgroundImage];
            }));
            holder.remove();return colors;
        """)
    assert colors_by_theme["light"]["blocked"] != colors_by_theme["light"]["missing"]
    assert colors_by_theme["dark"]["blocked"] != colors_by_theme["dark"]["missing"]
    for theme in ("light", "dark"):
        stripes = list(dict.fromkeys(re.findall(r"rgb\((\d+), (\d+), (\d+)\)", colors_by_theme[theme]["unsupported"])))
        assert len(stripes) == 2
        assert max(abs(int(light) - int(dark)) for light, dark in zip(*stripes)) >= 32
    assert all(color != "rgba(0, 0, 0, 0)" for row in colors_by_theme.values() for color in row.values())
    assert all(colors_by_theme["light"][status] != colors_by_theme["dark"][status]
               for status in ("pass", "diverge", "leak", "error", "missing", "blocked"))


@pytest.mark.parametrize("vertical", [False, True])
def test_shared_parent_displays_missing_child_capture(matrix_driver, rendered_page, vertical):
    d = matrix_driver
    query = "?slots_tab-unified=vllm_python_batch%400.26.0%2C%2C"
    load(d, rendered_page, query + ("&vertical=1" if vertical else ""))
    result = d.execute_script("""
        const hosts = [...document.querySelectorAll('#tab-unified td.cell[data-ttip-id]')];
        const host = hosts.find(node => {
            const cell = window.__conformanceCellModel(node);
            return node.getBoundingClientRect().height > 0 && cell && cell.family === 'gpt_oss'
                && cell.sub === 'tool_in_reason';
        });
        if (!host) return null;
        const cell = window.__conformanceCellModel(host);
        return {status:window.__conformanceOutputStatus({id:'tab-unified'}, cell,
                    {key:'vllm_python_batch@0.26.0'}, 'golden'),
                painted:host.querySelector('[data-capture="vllm_python_batch@0.26.0"]').className,
                variants:cell.variants.map(variant=>variant.sub)};
    """)
    assert result is not None
    assert result["status"] == "missing"
    assert "status-missing" in result["painted"]
    assert "gpt_oss_interrupted_reasoning_then_tool" in result["variants"]


def test_popup_columns_follow_slot_order_after_swapping_engines(matrix_driver, rendered_page):
    d = matrix_driver
    load(d, rendered_page)
    try:
        for keys in (("dynamo_v1@10.0.1", "sglang_python@0.5.21"),
                     ("sglang_python@0.5.21", "dynamo_v1@10.0.1")):
            load(d, rendered_page, "?slots_tab-unified=" + quote(",".join(keys) + ","))
            for selector in ("td.cell[data-cmp][data-ttip-id]", "th[data-ttip-id]"):
                host = next(node for node in d.find_elements(By.CSS_SELECTOR, "#tab-unified " + selector) if node.is_displayed())
                host.send_keys(Keys.ENTER)
                popup = WebDriverWait(d, 10).until(lambda browser: browser.find_element(By.CSS_SELECTOR, "body > .ttip-visible"))
                orders = d.execute_script("""
                    return [...arguments[0].querySelectorAll('table tr')].map(row => ({
                      grammar: row.closest('table').classList.contains('ttip-grammar'),
                      keys: [...row.querySelectorAll('[data-cand]:not(.col-hidden)')].map(node => node.dataset.cand)
                    })).filter(row => row.keys.length);
                """, popup)
                assert orders
                assert all(row["keys"] == (["golden", *keys] if row["grammar"] else list(keys)) for row in orders)
                d.find_element(By.TAG_NAME, "body").send_keys(Keys.ESCAPE)

    finally:
        load(d, rendered_page)


def test_empty_args_capture_segments_match_popup_outcomes(matrix_driver, rendered_page):
    d = matrix_driver
    keys = ["dynamo_v1@10.0.1", "sglang_python@0.5.21"]
    try:
        load(d, rendered_page, "?slots_tab-unified=" + quote(",".join(keys) + ","))
        cell = next(node for node in d.find_elements(By.CSS_SELECTOR, "#tab-unified td.cell[data-ttip-id]")
                    if d.execute_script("return String(window.__conformanceTooltipModel(arguments[0]).head).endsWith('UNIFIED.6-empty_args (empty_args) — deepseek_v41')", node))
        assert selection(d)["keys"] == keys
        statuses = d.execute_script("return [...arguments[0].querySelectorAll('.matrix-status')].map(n=>[n.dataset.capture,n.className.split('status-')[1]])", cell)
        assert statuses == [[keys[0], "diverge"], [keys[1], "pass"]]
        outcomes = d.execute_script("const m=window.__conformanceTooltipModel(arguments[0]);return Object.fromEntries(m.candidates.filter(c=>['dynamo_v1@10.0.1','sglang_python@0.5.21'].includes(c.key)).map(c=>[c.key,c.block]))", cell)
        assert outcomes[keys[1]]["verdict"] == "MATCH"
        assert outcomes[keys[1]]["events"] == [{"kind": "tool_call", "name": "get_weather", "arguments": {}}]
        assert outcomes[keys[0]]["verdict"] == "LOSS"
        assert outcomes[keys[0]]["batch_final"]["calls"] == []
        assert 'name="get_weather"' in outcomes[keys[0]]["batch_final"]["normal_text"]
        cell.click()
        popup = WebDriverWait(d, 10).until(lambda browser: browser.find_element(By.CSS_SELECTOR, "body > .ttip-visible"))
        orders = d.execute_script("return [...arguments[0].querySelectorAll('table tr')].map(r=>({grammar:r.closest('table').classList.contains('ttip-grammar'),keys:[...r.querySelectorAll('[data-cand]:not(.col-hidden)')].map(n=>n.dataset.cand)})).filter(r=>r.keys.length)", popup)
        assert orders and all(row["keys"] == (["golden", *keys] if row["grammar"] else keys) for row in orders)
        d.find_element(By.TAG_NAME, "body").send_keys(Keys.ESCAPE)
    finally:
        load(d, rendered_page)


def test_engine_picker_choices_match_slot_badge_colors(matrix_driver, rendered_page):
    d = matrix_driver
    load(d, rendered_page)
    try:
        d.find_element(By.CSS_SELECTOR, "[data-slot-open='1']").click()
        picker = d.find_element(By.CSS_SELECTOR, ".comparison-picker[open]")
        colors = d.execute_script("""
            const root = document.documentElement;
            const original = root.getAttribute('data-theme');
            const result = {};
            for (const theme of ['light', 'dark']) {
              root.setAttribute('data-theme', theme);
              const probe = document.createElement('div');
              probe.innerHTML = ['dynamo', 'sglang', 'vllm'].map(engine =>
                `<div class="capture-slot" data-engine="${engine}"><span class="capture-slot-badge"></span></div>`
              ).join('');
              document.body.appendChild(probe);
              result[theme] = Object.fromEntries(
                ['dynamo', 'sglang', 'vllm', 'none'].map(engine => {
                  const choice = document.querySelector(`[data-picker-engine-choice="${engine}"]`);
                  const background = getComputedStyle(choice).backgroundColor;
                  const badge = probe.querySelector(`.capture-slot[data-engine="${engine}"] .capture-slot-badge`);
                  return [engine, {background, badge: badge ? getComputedStyle(badge).backgroundColor : null}];
                })
              );
              probe.remove();
            }
            if (original === null) root.removeAttribute('data-theme');
            else root.setAttribute('data-theme', original);
            return result;
        """)
        for theme in ("light", "dark"):
            for engine in ("dynamo", "sglang", "vllm"):
                assert colors[theme][engine]["background"] == colors[theme][engine]["badge"]
                assert colors[theme][engine]["background"] != "rgba(0, 0, 0, 0)"
            assert colors[theme]["none"]["background"] not in {
                colors[theme][engine]["background"] for engine in ("dynamo", "sglang", "vllm")
            }
    finally:
        load(d, rendered_page)
