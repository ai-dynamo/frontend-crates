# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from urllib.parse import quote

import pytest

pytest.importorskip("selenium")
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait

from conformance.utils.tests.test_browser_comparison_layout import DEFAULTS, DEFAULT_SLOTS, choose_slot, load, matrix_driver, selection
from conformance.utils.src import gen_unified_golden
from conformance.utils.src.validate_conformance_status import load_model, validate_unified_only_report


def test_public_html_and_json_only_contain_unified(rendered_page):
    model = load_model(rendered_page)
    validate_unified_only_report(model)
    assert model["tabs"][0]["label"] == "Unified"
    status = json.loads(rendered_page.with_suffix(".json").read_text())
    assert {row["tab"] for row in status["reports"]} == {"tab-unified"}
    assert status["comparison_defaults"]["tab-unified"] == DEFAULT_SLOTS
    batch = [capture for capture in status["captures"] if capture["mode"] == "batch"]
    applicable_pairs = sum(len(gen_unified_golden.build_cases(family)) for family in gen_unified_golden.FAMILIES)
    assert len(batch) == 37
    assert {"dynamo_v1@10.0.7", "dynamo_v1_jail@10.0.7"} <= {capture["capture"] for capture in batch}
    assert all(capture["considered"] == applicable_pairs for capture in batch)
    assert all(capture["attempted"] == capture["captured"] + capture["parser_exception"] for capture in batch)
    assert any(capture["mismatches"] for capture in batch)
    blocked = next(capture for capture in batch if capture["capture"] == "dynamo_v1@9.2.13")
    assert blocked["source_unavailable"] > 0
    assert blocked["source_unavailable"] + blocked["missing_capture"] == applicable_pairs
    assert blocked["attempted"] == 0


@pytest.mark.parametrize("transpose", [False, True])
def test_shared_popup_url_restores_without_redundant_tab_parameter(matrix_driver, rendered_page, transpose):
    d = matrix_driver
    load(d, rendered_page, f"?tab=tab-unified&transpose={int(transpose)}")
    legend = d.find_element(By.CSS_SELECTOR, ".capture-legend")
    assert "or press Enter/Space" not in legend.text

    selector = "#tab-unified table[data-transpose-table] th.trow-case[data-ttip-id]" if transpose else "#tab-unified td.cell[data-cmp][data-ttip-id]"
    owner = d.find_element(By.CSS_SELECTOR, selector)
    owner.click()
    popup = WebDriverWait(d, 10).until(lambda browser: browser.find_element(By.CSS_SELECTOR, "body > .ttip-pinned"))
    shared_url = d.current_url
    key = d.execute_script("return new URL(arguments[0]).searchParams.get('popup')", shared_url)
    assert key
    assert not d.execute_script("return new URL(arguments[0]).searchParams.has('tab')", shared_url)
    assert d.execute_script("return arguments[0].dataset.ttipKey", owner) == key

    d.get(shared_url)
    restored = WebDriverWait(d, 10).until(lambda browser: browser.find_element(By.CSS_SELECTOR, "body > .ttip-pinned"))
    assert restored.is_displayed()
    assert d.execute_script("""
        const key = new URL(location.href).searchParams.get('popup');
        const owner = [...document.querySelectorAll('[data-ttip-key][data-ttip-click="pin"]')]
            .find(node => node.dataset.ttipKey === key && node.classList.contains('ttip-open'));
        return owner && Boolean(owner.closest('table[data-transpose-table]')) === arguments[0];
    """, transpose)
    restored.find_element(By.CSS_SELECTOR, ".ttip-close").click()
    assert not d.find_elements(By.CSS_SELECTOR, "body > .ttip-pinned")
    assert not d.execute_script("return new URL(location.href).searchParams.has('popup')")


@pytest.mark.parametrize("removed", ["tab-toolcalling-batch", "tab-toolcalling-streamv1", "tab-reasoning-batch", "tab-reasoning-stream"])
def test_removed_tab_links_open_the_unified_report_without_a_tab_button(matrix_driver, rendered_page, removed):
    d = matrix_driver
    load(d, rendered_page, "?tab=" + removed)
    assert d.execute_script("return [...document.querySelectorAll('.tab-panel')].map(node=>node.id)") == ["tab-unified"]
    assert not d.find_elements(By.CSS_SELECTOR, ".tab-button")
    assert selection(d)["keys"] == DEFAULTS


def test_none_slots_and_duplicate_selections(matrix_driver, rendered_page):
    d = matrix_driver
    load(d, rendered_page, "?tab=tab-unified&slots_tab-unified=" + quote(",,"))
    assert selection(d) == {"keys": DEFAULTS, "reference": None}
    assert d.find_element(By.CSS_SELECTOR, "[data-slot-capture='0']").get_attribute("value") == DEFAULTS[0]
    assert [d.find_element(By.CSS_SELECTOR, f"[data-slot-engine='{slot}']").text for slot in range(3)] == ["Dynamo", "", ""]
    assert not d.find_elements(By.CSS_SELECTOR, "[data-slot-capture='1'],[data-slot-capture='2']")
    choose_slot(d, 0, "dynamo", "dynamo_v1@9.2.10")
    assert selection(d)["keys"] == ["dynamo_v1@9.2.10"]
    assert d.find_elements(By.CSS_SELECTOR, "#tab-unified td.cell.cmp-leak")
    d.find_element(By.CSS_SELECTOR, "[data-slot-open='1']").click()
    picker = d.find_element(By.CSS_SELECTOR, ".comparison-picker[open]")
    picker.find_element(By.CSS_SELECTOR, "[data-picker-engine-choice='dynamo']").click()
    duplicate = d.find_element(By.CSS_SELECTOR, "[data-slot-capture='1'] option[value='dynamo_v1@9.2.10']")
    assert not duplicate.is_enabled()
    d.find_element(By.CSS_SELECTOR, "[data-slot-capture='1']").send_keys(Keys.ESCAPE)
    d.find_element(By.CSS_SELECTOR, "[data-slot-open='1']").click()
    picker = d.find_element(By.CSS_SELECTOR, ".comparison-picker[open]")
    picker.find_element(By.CSS_SELECTOR, "[data-picker-engine-choice='none']").click()
    assert selection(d)["keys"] == ["dynamo_v1@9.2.10"]
    d.find_element(By.CSS_SELECTOR, "[data-reset]").click()
    WebDriverWait(d, 10).until(lambda browser: selection(browser)["keys"] == DEFAULTS)


def test_unavailable_capture_is_omitted_and_older_links_can_select_it(matrix_driver, rendered_page):
    d = matrix_driver
    load(d, rendered_page)
    d.find_element(By.CSS_SELECTOR, ".capture-slot [data-slot-open='0']").click()
    d.find_element(By.CSS_SELECTOR, ".comparison-picker [data-picker-engine-choice='dynamo']").click()
    assert not d.find_elements(By.CSS_SELECTOR, "[data-slot-capture='0'] option[value='dynamo_v1@9.2.13']")

    load(d, rendered_page, "?tab=tab-unified&slots_tab-unified=dynamo_v1%409.2.13%2C%2C")
    assert selection(d)["keys"] == ["dynamo_v1@9.2.13"]
    assert d.find_elements(By.CSS_SELECTOR, "#tab-unified td.cell.cmp-na")
    assert not d.find_elements(By.CSS_SELECTOR, "[data-slot-capture='0'] option[value='dynamo_v1@9.2.13']")


def test_gpt_oss_v1_batch_is_scored_against_ordered_golden(rendered_page):
    model = load_model(rendered_page)
    tab = next(item for item in model["tabs"] if item["id"] == "tab-unified")
    row = next(item for item in tab["rows"] if item["family"] == "gpt_oss")
    cells = []

    def collect(cell):
        cells.append(cell)
        for variant in cell.get("variants") or []:
            collect(variant)

    for cell in row["cells"].values():
        collect(cell)
    by_case = {cell["case_id"]: cell for cell in cells}
    captured = [cell for cell in by_case.values() if not cell["cmp"]["dynamo_v1_jail@10.0.1"]["na"]]
    mismatches = [cell for cell in captured
                  if cell["cmp"]["dynamo_v1_jail@10.0.1"]["sig"] != cell["cmp"]["golden"]["sig"]]
    assert len(captured) == 159
    assert mismatches, "ordered batch output differences must remain visible as mismatches"


def test_nemotron_default_reasoning_popup_describes_family_default(matrix_driver, rendered_page):
    d = matrix_driver
    load(d, rendered_page)
    cell = next(
        cell for cell in d.find_elements(By.CSS_SELECTOR, '#tab-unified td.cell[data-family="nemotron_v3"][data-ttip-key]')
        if 'default_reasoning_tool_without_closer' in cell.get_attribute('data-ttip-key') and cell.is_displayed()
    )
    d.execute_script('arguments[0].scrollIntoView({block:"center",inline:"center"})', cell)
    cell.click()
    popup = WebDriverWait(d, 10).until(lambda driver: driver.find_element(By.CSS_SELECTOR, 'body > .ttip-pinned'))
    try:
        assert popup.is_displayed()
        assert 'use the family default' in popup.text
        assert 'Check.' in popup.text and 'Then.' in popup.text
        assert 'log' in popup.text and '雪' in popup.text
        assert 'prompt opened no channel' not in popup.text
    finally:
        popup.find_element(By.CSS_SELECTOR, '.ttip-close').click()
