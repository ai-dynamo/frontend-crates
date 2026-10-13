# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

selenium = pytest.importorskip("selenium")
from selenium import webdriver  # noqa: E402
from selenium.webdriver.chrome.options import Options  # noqa: E402
from selenium.webdriver.common.action_chains import ActionChains  # noqa: E402
from selenium.webdriver.common.by import By  # noqa: E402
from selenium.webdriver.common.keys import Keys  # noqa: E402
from selenium.webdriver.support.ui import WebDriverWait  # noqa: E402

pytestmark = pytest.mark.skipif(
    not any(shutil.which(binary) for binary in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser")),
    reason="no headless Chrome available",
)


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
    WebDriverWait(driver, 20).until(lambda browser: browser.find_elements(By.CSS_SELECTOR, ".tab-panel.active .comparison-matrix"))


def selection(driver):
    return driver.execute_script("const s=document.querySelector('.tab-panel.active')._comparisonSelection;return {keys:s.keys,reference:s.reference}")


def click(driver, selector):
    driver.find_element(By.CSS_SELECTOR, ".tab-panel.active " + selector).click()


def wait_for_popup(driver):
    return WebDriverWait(driver, 20).until(lambda browser: browser.execute_script("""
      const popup=document.querySelector('.comparison-drawer.open');
      if(!popup) return null;
      const style=getComputedStyle(popup.querySelector('.drawer-panel'));
      return style.opacity==='1' && style.transform==='none' ? popup : null;
    """))


def choose_slot(driver, slot, engine, capture=None):
    click(driver, f".engine-card [data-slot-picker='{slot}']")
    driver.execute_script("const e=document.querySelector('dialog[open] [data-picker-engine]');e.value=arguments[0];e.dispatchEvent(new Event('change',{bubbles:true}));", engine)
    if capture:
        driver.execute_script("const v=document.querySelector('dialog[open] [data-picker-version]');v.value=arguments[0];v.dispatchEvent(new Event('change',{bubbles:true}));", capture)
    driver.find_element(By.CSS_SELECTOR, "dialog[open]").send_keys(Keys.ESCAPE)


def test_status_mapping_and_capture_denominators(matrix_driver, rendered_page):
    d = matrix_driver
    load(d, rendered_page, "?tab=tab-toolcalling-batch")
    try:
        coordinates = d.execute_script("""
          const tab=window.__CONFORMANCE_PAGE.tabs.find(tab=>tab.id==='tab-toolcalling-batch');
          return {row:tab.rows.findIndex(row=>row.family==='deepseek_v4'),column:tab.columns.findIndex(column=>column.sub==='9.b')};
        """)
        peer_case = d.find_element(By.CSS_SELECTOR, f".tab-panel.active .matrix-cell[data-row='{coordinates['row']}'][data-column='{coordinates['column']}']")
        assert peer_case.find_elements(By.CSS_SELECTOR, ".status-diverge.matrix-status")
        click(d, "[data-focus='disagreements']")
        peer_case = d.find_element(By.CSS_SELECTOR, f".tab-panel.active .matrix-cell[data-row='{coordinates['row']}'][data-column='{coordinates['column']}']")
        assert peer_case.get_attribute("data-disagreement") == "true"
        assert "focus-dim" not in peer_case.get_attribute("class")
        keys = d.execute_script("""
          const panel=document.querySelector('.tab-panel.active');
          const tab=window.__CONFORMANCE_PAGE.tabs.find(tab=>tab.id===panel.id);
          const keys=panel._comparisonSelection.keys;
          const row=tab.rows.find(row=>!row.section);
          const column=tab.columns[0];
          const cell=row.cells[column.sub];
          tab.rows=[row]; tab.columns=[column];
          cell.kind='cell'; cell.known_divergence=false;
          cell.cmp=Object.fromEntries(keys.map(key=>[key,{sig:1}]));
          cell.cmp[keys[1]].sig=2;
          cell.tooltip={candidates:keys.map(key=>({key,block:{explanation:'Inherited unchanged from an older capture.'}})),reasons:[]};
          return keys;
        """)
        assert len(keys) == 3
        click(d, "[data-focus='all']")
        assert len(d.find_elements(By.CSS_SELECTOR, ".tab-panel.active .status-diverge.matrix-status")) == 1
        assert not d.find_elements(By.CSS_SELECTOR, ".tab-panel.active .status-documented.matrix-status")
        d.execute_script("""
          const panel=document.querySelector('.tab-panel.active');
          const tab=window.__CONFORMANCE_PAGE.tabs.find(tab=>tab.id===panel.id);
          const cell=tab.rows[0].cells[tab.columns[0].sub];
          cell.cmp[panel._comparisonSelection.reference]={sig:0,na:1};
          cell.tooltip.candidates.find(candidate=>candidate.key===panel._comparisonSelection.reference).block={unavailable:'not captured at this version'};
        """)
        for layout in ("split", "stacked"):
            click(d, f"[data-layout='{layout}']")
            assert len(d.find_elements(By.CSS_SELECTOR, ".tab-panel.active .status-uncompared.matrix-status")) == 2
            assert len(d.find_elements(By.CSS_SELECTOR, ".tab-panel.active .status-missing.matrix-status")) == 1
            scores = [card.text for card in d.find_elements(By.CSS_SELECTOR, ".tab-panel.active .engine-score")]
            assert "0 of 0 captured" in scores[0], scores
            assert all("0 of 1 captured" in score for score in scores[1:]), scores
            assert not d.find_elements(By.CSS_SELECTOR, ".tab-panel.active .status-pass.matrix-status")
    finally:
        load(d, rendered_page)


def test_geometry_focus_orientation_and_themes(matrix_driver, rendered_page):
    d = matrix_driver
    load(d, rendered_page)
    click(d, "[data-layout='split']")
    assert d.find_element(By.CSS_SELECTOR, ".tab-button.active").get_attribute("data-tab-target") == "tab-unified"
    assert d.execute_script("return [...document.querySelectorAll('.tabs .tab-button')].at(-1)?.dataset.tabTarget") == "tab-unified"
    assert len(d.find_elements(By.CSS_SELECTOR, ".tab-panel.active .engine-card")) == 3
    assert "matches GOLDEN" in d.find_element(By.CSS_SELECTOR, ".tab-panel.active .engine-score").text
    assert not d.find_elements(By.CSS_SELECTOR, ".model-summary-bar")
    summaries = d.execute_script("""
      const cell=document.querySelector('.tab-panel.active .matrix-summary');
      return [...cell.querySelectorAll('.model-summary')].map(node=>{
        const label=node.querySelector('b'),value=node.querySelector('span');
        const valueText=document.createRange();
        valueText.selectNodeContents(value);
        return {label:label.textContent.trim(),value:value.textContent.trim(),
          labelColor:getComputedStyle(label).color,valueColor:getComputedStyle(value).color,
          gap:valueText.getBoundingClientRect().left-label.getBoundingClientRect().right};
      });
    """)
    assert len(summaries) >= 2, summaries
    assert all(summary["label"] == summary["label"].upper() for summary in summaries), summaries
    assert all(summary["labelColor"] == summary["valueColor"] for summary in summaries), summaries
    colors_by_engine = {summary["label"]: summary["labelColor"] for summary in summaries}
    assert len(set(colors_by_engine.values())) == len(colors_by_engine), summaries
    assert max(summary["gap"] for summary in summaries) <= 5, summaries
    labels = d.execute_script("""
      const p=document.querySelector('.tab-panel.active'),tab=window.__CONFORMANCE_PAGE.tabs.find(t=>t.id===p.id);
      return {actual:[...p.querySelectorAll('.matrix-case-heading')].map(n=>{const c=n.cloneNode(true);c.querySelector('.ttip')?.remove();return c.textContent.trim()}),
        expected:tab.columns.map(c=>c.compact_prefix+'-'+c.display_number)};
    """)
    assert labels["actual"] == labels["expected"]
    group_geometry = d.execute_script("""
      const row=document.querySelector('.tab-panel.active .matrix-group-row');
      const caseHeading=document.querySelector('.tab-panel.active .matrix-case-row th');
      return [...document.querySelectorAll('.tab-panel.active .matrix-group-row th:not(.matrix-corner):not(.matrix-summary-head)')]
        .map(n=>({label:n.textContent.trim(),width:n.getBoundingClientRect().width,span:n.colSpan,
          rowHeight:row.getBoundingClientRect().height,caseTop:getComputedStyle(caseHeading).top}));
    """)
    single_case_groups = [group for group in group_geometry if group["span"] == 1]
    assert single_case_groups and max(group["width"] for group in single_case_groups) <= 200, group_geometry
    short_single_case_groups = [group for group in single_case_groups if len(group["label"]) <= 28]
    assert short_single_case_groups and max(group["width"] for group in short_single_case_groups) < 100, group_geometry
    assert max(group["rowHeight"] for group in group_geometry) <= 52, group_geometry
    assert {group["caseTop"] for group in group_geometry} == {"50px"}, group_geometry
    d.find_element(By.CSS_SELECTOR, "[data-transpose-toggle]").click()
    WebDriverWait(d, 10).until(lambda browser: "layout-vertical" in browser.find_element(By.CSS_SELECTOR, ".tab-panel.active .comparison-ui").get_attribute("class"))
    vertical_labels = d.execute_script("""
      const p=document.querySelector('.tab-panel.active'),tab=window.__CONFORMANCE_PAGE.tabs.find(t=>t.id===p.id);
      const headings=[...p.querySelectorAll('.matrix-case-heading')];
      const sample=headings[1],small=sample.querySelector('.case-label-placeholder');
      return {actual:headings.map(n=>{const c=n.cloneNode(true);c.querySelector('.ttip')?.remove();return c.textContent.trim()}),
        expected:tab.columns.map(c=>c.label+' ('+c.compact_prefix+'-'+c.display_number+')'),
        small:small&&{color:getComputedStyle(small).color,fontSize:parseFloat(getComputedStyle(small).fontSize),parentSize:parseFloat(getComputedStyle(sample).fontSize)}};
    """)
    assert vertical_labels["actual"] == vertical_labels["expected"]
    assert vertical_labels["small"]["color"] == "rgb(138, 143, 152)" and vertical_labels["small"]["fontSize"] < vertical_labels["small"]["parentSize"], vertical_labels
    d.find_element(By.CSS_SELECTOR, "[data-transpose-toggle]").click()
    WebDriverWait(d, 10).until(lambda browser: "layout-vertical" not in browser.find_element(By.CSS_SELECTOR, ".tab-panel.active .comparison-ui").get_attribute("class"))
    load(d, rendered_page, "?tab=tab-toolcalling-batch")
    labels = d.execute_script("""
      const p=document.querySelector('.tab-panel.active'),tab=window.__CONFORMANCE_PAGE.tabs.find(t=>t.id===p.id);
      return {actual:[...p.querySelectorAll('.matrix-case-heading')].map(n=>{const c=n.cloneNode(true);c.querySelector('.ttip')?.remove();return c.textContent.trim()}),
        expected:tab.columns.map(c=>c.display_number||c.label)};
    """)
    assert labels["actual"] == labels["expected"]
    for theme in ("light", "dark"):
        d.execute_script("document.documentElement.dataset.theme=arguments[0]", theme)
        geometry = d.execute_script("""
          const p=document.querySelector('.tab-panel.active');
          const r=p.querySelector('.comparison-matrix-card').getBoundingClientRect();
          return {left:r.left,right:document.documentElement.clientWidth-r.right,tab:getComputedStyle(document.querySelector('.tab-button.active')).backgroundColor,
            tabs:getComputedStyle(document.querySelector('.tabs')).backgroundColor,page:getComputedStyle(document.body).backgroundColor};
        """)
        assert geometry["left"] == pytest.approx(12)
        assert geometry["right"] == pytest.approx(12)
        assert geometry["tab"] == geometry["tabs"] == geometry["page"]
        click(d, "[data-layout='stacked']")
        tints = d.execute_script("return [...document.querySelectorAll('.tab-panel.active .stacked-row')].slice(0,3).map(row=>[...row.children].map(cell=>getComputedStyle(cell).backgroundColor));")
        assert len({row[0] for row in tints}) == 3, [sorted(set(row)) for row in tints]
        assert all(len(set(row)) == 1 for row in tints)
        family_border = d.execute_script("""
          const row=document.querySelector('.tab-panel.active .matrix-model-row.family-start');
          const style=getComputedStyle(row.firstElementChild);
          return {count:document.querySelectorAll('.tab-panel.active .matrix-model-row.family-start').length,width:style.borderTopWidth,color:style.borderTopColor};
        """)
        assert family_border["count"] > 1 and family_border["width"] == "2px", family_border
        assert family_border["color"] == ("rgb(17, 24, 39)" if theme == "light" else "rgb(232, 237, 243)"), family_border
        score = d.find_element(By.CSS_SELECTOR, ".tab-panel.active .engine-score").text
        click(d, "[data-focus='disagreements']")
        cells = d.execute_script("return [...document.querySelectorAll('.tab-panel.active .matrix-cell')].map(c=>({r:c.dataset.row,c:c.dataset.column,disagree:c.dataset.disagreement==='true',dim:c.classList.contains('focus-dim')}));")
        assert any(cell["disagree"] for cell in cells)
        assert all(cell["dim"] != cell["disagree"] for cell in cells)
        assert d.find_element(By.CSS_SELECTOR, ".tab-panel.active .engine-score").text == score
        d.find_element(By.CSS_SELECTOR, "[data-transpose-toggle]").click()
        WebDriverWait(d, 10).until(lambda browser: "layout-vertical" in browser.find_element(By.CSS_SELECTOR, ".tab-panel.active .comparison-ui").get_attribute("class"))
        assert d.find_element(By.CSS_SELECTOR, ".tab-panel.active [data-layout='stacked']").text == "Stacked columns"
        assert d.find_element(By.CSS_SELECTOR, ".tab-panel.active .legacy-report-details").get_dom_attribute("open") is None
        vertical = d.execute_script("return [...document.querySelectorAll('.tab-panel.active .matrix-cell')].map(c=>({r:c.dataset.row,c:c.dataset.column,disagree:c.dataset.disagreement==='true',dim:c.classList.contains('focus-dim')}));")
        assert sorted(cells, key=lambda x: (x["r"], x["c"])) == sorted(vertical, key=lambda x: (x["r"], x["c"]))
        vertical_border = d.execute_script("""
          const cell=document.querySelector('.tab-panel.active .comparison-matrix td.family-start');
          const style=getComputedStyle(cell);
          return {count:document.querySelectorAll('.tab-panel.active .comparison-matrix td.family-start').length,width:style.borderLeftWidth,color:style.borderLeftColor};
        """)
        assert vertical_border["count"] > 1 and vertical_border["width"] == "2px", vertical_border
        assert vertical_border["color"] == ("rgb(17, 24, 39)" if theme == "light" else "rgb(232, 237, 243)"), vertical_border
        click(d, "[data-layout='split']")
        vertical_width = d.execute_script("""
          const scroll=document.querySelector('.tab-panel.active .comparison-matrix-scroll');
          const table=scroll.querySelector('.comparison-matrix');
          const headers=[...table.querySelectorAll('thead .matrix-case-row th')].slice(1);
          return {tableMinWidth:getComputedStyle(table).minWidth,headerMinWidths:headers.map(n=>getComputedStyle(n).minWidth)};
        """)
        assert vertical_width["tableMinWidth"] == "max-content", vertical_width
        assert vertical_width["headerMinWidths"] and set(vertical_width["headerMinWidths"]) == {"0px"}, vertical_width
        assert len(d.find_elements(By.CSS_SELECTOR, ".tab-panel.active .matrix-segments")) * 3 == len(cells)
        d.find_element(By.CSS_SELECTOR, "[data-transpose-toggle]").click()
    d.set_window_size(540, 900)
    assert d.execute_script("return document.documentElement.scrollWidth <= innerWidth + 1")
    d.set_window_size(2400, 1200)
    assert not [entry for entry in d.get_log("browser") if entry["level"] == "SEVERE"]


def test_interchangeable_slots_reference_empty_and_legacy_urls(matrix_driver, rendered_page):
    d = matrix_driver
    load(d, rendered_page)
    defaults = selection(d)
    assert defaults["reference"] is None
    assert not d.find_elements(By.CSS_SELECTOR, ".tab-panel.active [data-slot-reference]")
    card_scores = [node.text.lower() for node in d.find_elements(By.CSS_SELECTOR, ".tab-panel.active .engine-score span")]
    assert card_scores and all(text.startswith("matches golden") for text in card_scores)
    click(d, ".engine-card .engine-mark")
    assert d.find_elements(By.CSS_SELECTOR, "dialog[open]")
    placement = d.execute_script("""
      const trigger=document.querySelector('.tab-panel.active .engine-card .engine-mark').closest('[data-slot-picker]');
      const a=trigger.getBoundingClientRect(),p=document.querySelector('dialog[open]').getBoundingClientRect();
      return {horizontal:Math.min(Math.abs(p.left-a.right),Math.abs(a.left-p.right)),overlapsVertically:p.top<a.bottom&&p.bottom>a.top};
    """)
    assert placement["horizontal"] <= 12, placement
    assert placement["overlapsVertically"], placement
    assert not d.find_elements(By.CSS_SELECTOR, "dialog[open] [data-picker-apply], dialog[open] button[value='cancel']")
    d.find_element(By.CSS_SELECTOR, "dialog[open]").send_keys(Keys.ESCAPE)
    click(d, ".comparison-matrix [data-slot-picker='2']")
    d.execute_script("const e=document.querySelector('dialog[open] [data-picker-engine]');e.value='none';e.dispatchEvent(new Event('change',{bubbles:true}));")
    assert selection(d)["keys"] == defaults["keys"][:1]
    d.set_window_size(2200, 1200)
    placement = d.execute_script("""
      const trigger=document.querySelector('.tab-panel.active .engine-card [data-slot-picker="2"]');
      const a=trigger.getBoundingClientRect(),p=document.querySelector('dialog[open]').getBoundingClientRect();
      return {horizontal:Math.min(Math.abs(p.left-a.right),Math.abs(a.left-p.right)),overlapsVertically:p.top<a.bottom&&p.bottom>a.top};
    """)
    assert placement["horizontal"] <= 12 and placement["overlapsVertically"], placement
    d.find_element(By.CSS_SELECTOR, "dialog[open]").send_keys(Keys.ESCAPE)
    assert d.switch_to.active_element.get_attribute("data-slot-picker") == "2"
    choose_slot(d, 2, "vllm", defaults["keys"][1])
    d.set_window_size(2400, 1200)
    click(d, ".engine-card [data-slot-picker='2']")
    missing = d.execute_script("return [...document.querySelectorAll('dialog[open] [data-picker-version] option')].filter(o=>o.textContent.includes('no captured results')).map(o=>({key:o.value,disabled:o.disabled}))")
    assert missing and all(item["disabled"] for item in missing)
    d.find_element(By.CSS_SELECTOR, "dialog[open]").send_keys(Keys.ESCAPE)
    changed = selection(d)
    assert changed["keys"] == defaults["keys"]
    assert changed["reference"] is None
    d.refresh()
    assert selection(d) == changed
    assert "base_tab-unified" not in d.current_url
    versions = d.execute_script("return window.__CONFORMANCE_PAGE.tabs.find(t=>t.id==='tab-unified').candidates.filter(c=>c.impl.startsWith('dynamo')).map(c=>c.key)")
    other = next(key for key in versions if key not in changed["keys"])
    choose_slot(d, 1, "dynamo", other)
    assert selection(d)["keys"][:2] == [defaults["keys"][0], other]
    assert selection(d)["reference"] is None
    choose_slot(d, 2, "none")
    assert selection(d)["reference"] is None
    click(d, ".engine-card [data-slot-picker='2']")
    d.execute_script("const e=document.querySelector('dialog[open] [data-picker-engine]');e.value='dynamo';e.dispatchEvent(new Event('change',{bubbles:true}));")
    disabled = d.execute_script("return [...document.querySelectorAll('dialog[open] [data-picker-version] option:disabled')].map(o=>o.value)")
    assert set(selection(d)["keys"][:2]) <= set(disabled)
    assert len(selection(d)["keys"]) == 3
    assert selection(d)["keys"][2] not in selection(d)["keys"][:2]
    d.set_window_size(2200, 1200)
    placement = d.execute_script("""
      const trigger=document.querySelector('.tab-panel.active [data-slot-picker="2"]');
      const a=trigger.getBoundingClientRect(),p=document.querySelector('dialog[open]').getBoundingClientRect();
      return {horizontal:Math.min(Math.abs(p.left-a.right),Math.abs(a.left-p.right),Math.abs(p.left-12)),overlapsVertically:p.top<a.bottom&&p.bottom>a.top};
    """)
    assert placement["horizontal"] <= 12 and placement["overlapsVertically"], placement
    d.find_element(By.CSS_SELECTOR, "dialog[open]").send_keys(Keys.ESCAPE)
    assert d.switch_to.active_element.get_attribute("data-slot-picker") == "2"
    click(d, ".engine-card [data-slot-picker='2']")
    d.execute_script("document.querySelector('dialog[open]').dispatchEvent(new MouseEvent('click',{bubbles:true}));")
    assert not d.find_elements(By.CSS_SELECTOR, "dialog[open]")
    assert d.switch_to.active_element.get_attribute("data-slot-picker") == "2"
    d.set_window_size(2400, 1200)
    for slot in range(3):
        choose_slot(d, slot, "none")
    assert selection(d) == {"keys": [], "reference": None}
    assert not d.find_elements(By.CSS_SELECTOR, ".tab-panel.active .matrix-status")
    assert d.find_element(By.CSS_SELECTOR, ".tab-panel.active .comparison-matrix").get_attribute("data-empty") == "true"
    d.refresh()
    assert selection(d) == {"keys": [], "reference": None}
    choose_slot(d, 2, "dynamo", other)
    assert selection(d) == {"keys": [other], "reference": None}
    click(d, "[data-focus='disagreements']")
    assert d.find_elements(By.CSS_SELECTOR, ".tab-panel.active .matrix-cell[data-disagreement='true']:not(.focus-dim)")
    keys = [versions[1], versions[0], defaults["keys"][1], versions[2]]
    query = "?tab=tab-unified&transpose=1&base_tab-unified=" + keys[0] + "&cmp_tab-unified=" + ",".join(keys[1:])
    load(d, rendered_page, query)
    assert selection(d)["keys"] == keys[:3]
    assert selection(d)["reference"] is None
    assert "three-slot limit" in d.find_element(By.CSS_SELECTOR, ".tab-panel.active .comparison-notice").text
    assert d.find_element(By.CSS_SELECTOR, ".tab-panel.active .legacy-report-details").get_dom_attribute("open") is None
    load(d, rendered_page)
    choose_slot(d, 0, "none")
    click(d, ".engine-card [data-slot-picker='0']")
    unavailable = d.execute_script("return [...document.querySelectorAll('dialog[open] [data-picker-engine] option:disabled')].map(o=>o.textContent)")
    assert any("SGLang" in label and "no captures" in label for label in unavailable)
    d.find_element(By.CSS_SELECTOR, "dialog[open]").send_keys(Keys.ESCAPE)
    load(d, rendered_page, "?tab=tab-toolcalling-batch")
    modes = d.execute_script("return window.__CONFORMANCE_PAGE.tabs.find(t=>t.id==='tab-toolcalling-batch').candidates.filter(c=>c.impl.startsWith('dynamo')).map(c=>({key:c.key,mode:c.parse_mode}))")
    first = modes[0]
    alternate = next(item for item in modes if item["mode"] != first["mode"])
    choose_slot(d, 0, "dynamo", first["key"])
    choose_slot(d, 1, "dynamo", alternate["key"])
    selected = selection(d)
    for layout in ("split", "stacked"):
        click(d, f"[data-layout='{layout}']")
        for vertical in (True, False):
            d.find_element(By.CSS_SELECTOR, "[data-transpose-toggle]").click()
            WebDriverWait(d, 10).until(lambda browser: ("layout-vertical" in browser.find_element(By.CSS_SELECTOR, ".tab-panel.active .comparison-ui").get_attribute("class")) == vertical)
            assert selection(d) == selected
            d.refresh()
            assert selection(d) == selected


def test_shared_url_categories_controls_tabs_and_reset(matrix_driver, rendered_page):
    d = matrix_driver
    load(d, rendered_page)
    click(d, ".category-chip")
    click(d, "[data-layout='stacked']")
    click(d, "[data-focus='disagreements']")
    click(d, "[data-peer-only]")
    d.find_element(By.CSS_SELECTOR, ".tab-panel.active [data-case-search]").send_keys("tool_only")
    d.find_element(By.CSS_SELECTOR, "[data-transpose-toggle]").click()
    before = d.execute_script("return document.querySelector('.tab-panel.active .comparison-ui').innerHTML")
    d.refresh()
    assert d.execute_script("return document.querySelector('.tab-panel.active .comparison-ui').innerHTML") == before
    d.execute_script("while(document.querySelector('.tab-panel.active .category-chip[aria-pressed=true]')) document.querySelector('.tab-panel.active .category-chip[aria-pressed=true]').click();")
    assert "groups_tab-unified=" in d.current_url
    d.refresh()
    assert not d.find_elements(By.CSS_SELECTOR, ".tab-panel.active .category-chip[aria-pressed='true']")
    assert not d.find_elements(By.CSS_SELECTOR, ".tab-panel.active .matrix-cell")
    d.find_element(By.CSS_SELECTOR, "[data-tab-target='tab-toolcalling-batch']").click()
    assert d.find_elements(By.CSS_SELECTOR, ".tab-panel.active .category-chip[aria-pressed='true']")
    d.find_element(By.CSS_SELECTOR, "[data-tab-target='tab-unified']").click()
    assert not d.find_elements(By.CSS_SELECTOR, ".tab-panel.active .category-chip[aria-pressed='true']")
    d.find_element(By.CSS_SELECTOR, "[data-reset]").click()
    WebDriverWait(d, 20).until(lambda browser: browser.find_elements(By.CSS_SELECTOR, ".tab-panel.active .category-chip[aria-pressed='true']"))
    assert d.current_url.endswith("?tab=tab-unified")
    assert not d.find_element(By.CSS_SELECTOR, "[data-transpose-toggle]").is_selected()
    assert d.find_element(By.CSS_SELECTOR, ".tab-panel.active [data-layout='split']").get_attribute("aria-pressed") == "true"


def test_cell_and_case_popup_renderer_parity_and_drawer(matrix_driver, rendered_page):
    d = matrix_driver
    load(d, rendered_page)
    # Folded cases exercise variants; case-header tables exercise schemas and all families.
    result = d.execute_script("""
      const panel=document.querySelector('.tab-panel.active');
      const tab=window.__CONFORMANCE_PAGE.tabs.find(t=>t.id===panel.id);
      const rows=tab.rows.filter(r=>!r.section);
      const cell=[...panel.querySelectorAll('.matrix-cell')].find(n=>rows[+n.dataset.row].cells[tab.columns[+n.dataset.column].sub]?.tooltip?.variants);
      cell.focus();window.__buildTooltip(cell);
      const expected=document.createElement('div');
      expected.innerHTML='<div class="ttip">'+window.__renderConformanceTooltip(rows[+cell.dataset.row].cells[tab.columns[+cell.dataset.column].sub].tooltip,rows[+cell.dataset.row].cells[tab.columns[+cell.dataset.column].sub].fixture_href)+'</div>';
      window.__syncConformanceTooltip(expected,panel._comparisonSelection.keys,panel._comparisonSelection.reference,true);
      const tip=cell._ttip;const actual=tip.cloneNode(true);actual.querySelector('.ttip-close')?.remove();
      return {same:expected.querySelector('.ttip').textContent===actual.textContent,variants:tip.querySelectorAll('.case-variant').length,
        golden:tip.querySelectorAll('.cand-golden.cand-on,[data-cand=golden]:not(.col-hidden),.golden-out-cap,.oracle-output').length,fixtures:tip.querySelectorAll('a[href]').length};
    """)
    assert result["same"] and result["variants"] > 1 and result["golden"], result
    cell = d.find_element(By.CSS_SELECTOR, ".tab-panel.active .matrix-cell")
    ActionChains(d).move_to_element(cell).perform()
    hovered_popup = d.execute_async_script("const done=arguments[0];setTimeout(()=>done(!!document.querySelector('.ttip-visible')),900);")
    assert not hovered_popup
    cell.click()
    wait_for_popup(d)
    drawer_geometry = d.execute_script("const r=document.querySelector('.comparison-drawer.open .drawer-panel').getBoundingClientRect();return {width:r.width,left:r.left,right:innerWidth-r.right,viewport:innerWidth,clientWidth:document.documentElement.clientWidth};")
    assert drawer_geometry["width"] == pytest.approx(drawer_geometry["viewport"] * 0.97, abs=2), drawer_geometry
    assert drawer_geometry["left"] + drawer_geometry["right"] == pytest.approx(drawer_geometry["viewport"] - drawer_geometry["width"], abs=1), drawer_geometry
    assert drawer_geometry["left"] == pytest.approx((drawer_geometry["clientWidth"] - drawer_geometry["width"]) / 2, abs=2), drawer_geometry
    assert not d.find_elements(By.CSS_SELECTOR, ".ttip-pinned")
    assert d.find_elements(By.CSS_SELECTOR, ".comparison-drawer.open .cand-golden.cand-on,.comparison-drawer.open [data-cand=golden]:not(.col-hidden),.comparison-drawer.open .golden-out-cap,.comparison-drawer.open .oracle-output")
    scrim = d.find_element(By.CSS_SELECTOR, ".comparison-drawer.open .drawer-scrim")
    scrim_width = d.execute_script("return arguments[0].getBoundingClientRect().width", scrim)
    ActionChains(d).move_to_element_with_offset(scrim, -int(scrim_width / 2) + 2, 0).click().perform()
    assert not d.find_elements(By.CSS_SELECTOR, ".comparison-drawer.open")
    assert d.execute_script("return !document.elementFromPoint(innerWidth/2,innerHeight/2).closest('.comparison-drawer')")
    d.set_window_size(540, 900)
    cell = d.find_element(By.CSS_SELECTOR, ".tab-panel.active .matrix-cell")
    cell.click()
    wait_for_popup(d)
    narrow_geometry = d.execute_script("const r=document.querySelector('.comparison-drawer.open .drawer-panel').getBoundingClientRect();return {width:r.width,left:r.left,right:innerWidth-r.right,viewport:innerWidth,clientWidth:document.documentElement.clientWidth};")
    assert narrow_geometry["width"] == pytest.approx(narrow_geometry["viewport"] * 0.97, abs=2), narrow_geometry
    assert narrow_geometry["left"] + narrow_geometry["right"] == pytest.approx(narrow_geometry["viewport"] - narrow_geometry["width"], abs=1), narrow_geometry
    assert narrow_geometry["left"] == pytest.approx((narrow_geometry["clientWidth"] - narrow_geometry["width"]) / 2, abs=2), narrow_geometry
    d.find_element(By.CSS_SELECTOR, ".comparison-drawer.open .drawer-close").click()
    d.set_window_size(2400, 1200)
    for vertical in (False, True):
        if vertical:
            d.find_element(By.CSS_SELECTOR, "[data-transpose-toggle]").click()
            WebDriverWait(d, 10).until(lambda browser: "layout-vertical" in browser.find_element(By.CSS_SELECTOR, ".tab-panel.active .comparison-ui").get_attribute("class"))
        grammar = d.execute_script("""
          const p=document.querySelector('.tab-panel.active');
          const tab=window.__CONFORMANCE_PAGE.tabs.find(t=>t.id===p.id);
          const header=[...p.querySelectorAll('.matrix-case-heading')].find(n=>tab.columns[+n.dataset.caseColumn].schemas?.length);
          const chosen=header || p.querySelector('.matrix-case-heading');
          chosen.focus();const focusOpened=chosen._ttip.classList.contains('ttip-visible');window.__buildTooltip(chosen);
          const model=window.__conformanceCaseTooltip(tab,tab.columns[+chosen.dataset.caseColumn]);
          const expected=document.createElement('div');expected.innerHTML='<div class="ttip">'+window.__renderConformanceTooltip(model)+'</div>';
          window.__syncConformanceTooltip(expected,p._comparisonSelection.keys,p._comparisonSelection.reference,true);
          const tip=chosen._ttip;const actual=tip.cloneNode(true);actual.querySelector('.ttip-close')?.remove();
          window.__caseHeader=chosen;
          return {same:actual.textContent===expected.querySelector('.ttip').textContent,focusOpened,
            column:chosen.dataset.caseColumn,families:tip.querySelectorAll('table.ttip-grammar tbody tr').length,
            golden:tip.querySelectorAll('[data-cand=golden]:not(.col-hidden)').length};
        """)
        assert grammar["same"] and grammar["families"] > 1 and grammar["golden"], grammar
        assert not grammar["focusOpened"], grammar
        case_header = d.find_element(By.CSS_SELECTOR, f".tab-panel.active .matrix-case-heading[data-case-column='{grammar['column']}']")
        ActionChains(d).move_to_element(case_header).perform()
        hovered_popup = d.execute_async_script("const done=arguments[0];setTimeout(()=>done(!!document.querySelector('.ttip-visible')),900);")
        assert not hovered_popup
        for activation in (Keys.ENTER, Keys.SPACE):
            case_header.send_keys(activation)
            assert d.find_elements(By.CSS_SELECTOR, ".ttip-pinned")
            d.find_element(By.CSS_SELECTOR, ".tab-panel.active .matrix-corner").click()
            assert not d.find_elements(By.CSS_SELECTOR, ".ttip-pinned")
        case_header.click()
        assert d.find_elements(By.CSS_SELECTOR, ".ttip-pinned")
        d.find_element(By.CSS_SELECTOR, ".tab-panel.active .matrix-corner").click()
        assert not d.find_elements(By.CSS_SELECTOR, ".ttip-pinned")
    load(d, rendered_page, "?tab=tab-toolcalling-streamv1")
    chunks = d.execute_script("""
      const p=document.querySelector('.tab-panel.active');
      const tab=window.__CONFORMANCE_PAGE.tabs.find(t=>t.id===p.id),rows=tab.rows.filter(r=>!r.section);
      const cell=[...p.querySelectorAll('.matrix-cell')].find(n=>rows[+n.dataset.row].cells[tab.columns[+n.dataset.column].sub]?.tooltip?.input?.chunks?.length);
      cell.focus();window.__buildTooltip(cell);
      return cell._ttip.querySelectorAll('.ttip-chunks tr').length;
    """)
    assert chunks > 1


def test_native_picker_focus_schema_and_reference_disclosure(matrix_driver, rendered_page):
    d = matrix_driver
    load(d, rendered_page, "?vertical=1")
    initial = selection(d)
    assert initial["reference"] is None
    assert not d.find_elements(By.CSS_SELECTOR, ".tab-panel.active [data-slot-reference]")
    version = d.find_element(By.CSS_SELECTOR, ".tab-panel.active [data-slot-version='0']")
    previous = version.get_attribute("value")
    version.send_keys(Keys.ARROW_DOWN)
    assert d.switch_to.active_element.get_attribute("data-slot-version") == "0"
    assert d.switch_to.active_element.get_attribute("value") != previous
    load(d, rendered_page, "?tab=tab-toolcalling-batch")
    batch_initial = selection(d)
    reference = d.find_element(By.CSS_SELECTOR, ".tab-panel.active [data-slot-reference='1']")
    reference.send_keys(Keys.SPACE)
    assert d.switch_to.active_element.get_attribute("data-slot-reference") == "1"
    assert selection(d)["reference"] == selection(d)["keys"][1]
    chip = d.find_elements(By.CSS_SELECTOR, ".tab-panel.active .category-chip")[1]
    group = chip.get_attribute("data-group")
    chip.send_keys(Keys.ENTER)
    assert d.switch_to.active_element.get_attribute("data-group") == group
    assert d.switch_to.active_element.get_attribute("aria-pressed") == "false"
    trigger = d.find_element(By.CSS_SELECTOR, ".tab-panel.active .engine-card [data-slot-picker='0']")
    trigger.send_keys(Keys.ENTER)
    engine = d.find_element(By.CSS_SELECTOR, "dialog[open] [data-picker-engine]")
    engine.send_keys(Keys.END)
    assert not d.find_elements(By.CSS_SELECTOR, "dialog[open] [data-picker-apply], dialog[open] button[value='cancel']")
    assert selection(d)["keys"] == batch_initial["keys"][1:]
    assert d.switch_to.active_element.get_attribute("data-picker-engine") is not None
    d.find_element(By.CSS_SELECTOR, "dialog[open]").send_keys(Keys.ESCAPE)
    assert d.switch_to.active_element.get_attribute("data-slot-picker") == "0"
    click(d, ".legacy-report-details > summary")
    assert d.find_element(By.CSS_SELECTOR, ".tab-panel.active table[data-parity-table]").is_displayed()
    header = d.execute_script("""
      const p=document.querySelector('.tab-panel.active'),t=window.__CONFORMANCE_PAGE.tabs.find(t=>t.id===p.id);
      return [...p.querySelectorAll('.matrix-case-heading')].find(n=>t.columns[+n.dataset.caseColumn].schemas?.length);
    """)
    assert header is not None
    ActionChains(d).move_to_element(header).perform()
    hover_started = d.execute_script("return performance.now()")
    WebDriverWait(d, 2).until(lambda browser: browser.execute_script("return performance.now()-arguments[0] >= 850", hover_started))
    hover_built = d.execute_script("return {visible:arguments[0]._ttip.classList.contains('ttip-visible'),built:arguments[0]._ttip.classList.contains('ttip-built')}", header)
    assert not hover_built["visible"] and not hover_built["built"], hover_built
    header.click()
    WebDriverWait(d, 10).until(lambda browser: browser.find_elements(By.CSS_SELECTOR, ".ttip-visible .case-schema-toggle"))
    schema = d.find_element(By.CSS_SELECTOR, ".ttip-visible .case-schema-toggle")
    schema.send_keys(Keys.ENTER)
    assert d.find_element(By.CSS_SELECTOR, ".ttip-visible details.case-schema").get_dom_attribute("open") is not None
    assert d.find_element(By.CSS_SELECTOR, ".ttip-visible .case-schema-json").is_displayed()
    d.execute_script("document.querySelector('.tab-panel.active [data-focus=disagreements]').click()")
    assert not d.find_elements(By.CSS_SELECTOR, ".ttip-visible")
    assert d.find_element(By.CSS_SELECTOR, ".tab-panel.active table[data-parity-table]").is_displayed()


def test_first_paint_never_shows_reference_table(matrix_driver, rendered_page):
    d = matrix_driver
    script = """
      window.__firstTable=null;window.__legacyPaint=false;
      function record() {
        const p=document.querySelector('.tab-panel.active');if(!p)return;
        const visible=n=>!n.closest('details:not([open])')&&getComputedStyle(n).display!=='none'&&getComputedStyle(n).visibility!=='hidden'&&n.getClientRects().length;
        for(const t of p.querySelectorAll('table[data-parity-table],.comparison-matrix')) {
          if(visible(t)){if(!window.__firstTable)window.__firstTable=t.className;if(t.hasAttribute('data-parity-table'))window.__legacyPaint=true;}
        }
      }
      new MutationObserver(record).observe(document,{childList:true,subtree:true,attributes:true});
    """
    registration = d.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {"source": script})
    try:
        d.execute_cdp_cmd("Emulation.setCPUThrottlingRate", {"rate": 4})
        for query in ("", "?vertical=1", "?base_tab-unified=vllm_rust&cmp_tab-unified=vllm_rust"):
            load(d, rendered_page, query)
            assert d.execute_script("return window.__firstTable") == "comparison-matrix"
            assert d.execute_script("return window.__legacyPaint") is False
    finally:
        d.execute_cdp_cmd("Emulation.setCPUThrottlingRate", {"rate": 1})
        d.execute_cdp_cmd("Page.removeScriptToEvaluateOnNewDocument", {"identifier": registration["identifier"]})


@pytest.mark.parametrize("layout", ["split", "stacked"])
def test_documented_dynamo_pair_keeps_unrelated_peer_disagreement(matrix_driver, rendered_page, layout):
    d = matrix_driver
    load(d, rendered_page, "?tab=tab-toolcalling-batch")
    try:
        click(d, f"[data-layout='{layout}']")
        case = d.execute_script("""
          const p=document.querySelector('.tab-panel.active'),t=window.__CONFORMANCE_PAGE.tabs.find(t=>t.id===p.id);
          const rows=t.rows.filter(r=>!r.section),row=rows.findIndex(r=>r.family==='deepseek_v4');
          const column=t.columns.findIndex(c=>c.sub==='9.b'),cell=rows[row].cells[t.columns[column].sub];
          const stream=t.candidates.find(c=>c.key.startsWith('dynamo_v2-s-')&&cell.cmp[c.key]&&!cell.cmp[c.key].na);
          return {row,column,stream:stream.key,batch:p._comparisonSelection.keys[0],peer:p._comparisonSelection.keys[2]};
        """)
        choose_slot(d, 1, "dynamo", case["stream"])
        assert selection(d)["keys"] == [case["batch"], case["stream"], case["peer"]]
        for reference_slot in (0, 1):
            click(d, f"[data-slot-reference='{reference_slot}']")
            click(d, "[data-focus='disagreements']")
            actual = d.execute_script("""
              const p=document.querySelector('.tab-panel.active'),t=window.__CONFORMANCE_PAGE.tabs.find(t=>t.id===p.id);
              const coordinates=arguments[0],cell=t.rows.filter(r=>!r.section)[coordinates.row].cells[t.columns[coordinates.column].sub];
              const nodes=[...p.querySelectorAll('.matrix-cell[data-row="'+coordinates.row+'"][data-column="'+coordinates.column+'"]')];
              const reference=p._comparisonSelection.reference;
              return {classes:nodes.flatMap(node=>[...node.querySelectorAll('.matrix-status')].map(n=>n.className)),
                peerMatches:cell.cmp[coordinates.peer].sig===cell.cmp[reference].sig,
                disagreement:nodes.every(node=>node.dataset.disagreement==='true'),dim:nodes.every(node=>node.classList.contains('focus-dim'))};
            """, case)
            expected = ["pass", "documented"] if reference_slot == 0 else ["documented", "pass"]
            expected.append("pass" if actual["peerMatches"] else "diverge")
            assert actual["classes"] == ["matrix-status status-" + status for status in expected], actual
            assert actual["disagreement"] == (not actual["peerMatches"]), actual
            assert actual["dim"] == actual["peerMatches"], actual
    finally:
        load(d, rendered_page)


@pytest.mark.parametrize("tab_id,vertical", [("tab-unified", False), ("tab-unified", True),
                                             ("tab-toolcalling-streamv1", False), ("tab-toolcalling-streamv1", True)])
def test_cell_popup_url_restores_identity_and_selection(matrix_driver, rendered_page, tab_id, vertical):
    d = matrix_driver
    load(d, rendered_page, "?tab=" + tab_id)
    try:
        if vertical:
            d.find_element(By.CSS_SELECTOR, "[data-transpose-toggle]").click()
            click(d, "[data-layout='stacked']")
        click(d, "[data-focus='disagreements']")
        chosen = d.execute_script("""
          const p=document.querySelector('.tab-panel.active'),t=window.__CONFORMANCE_PAGE.tabs.find(t=>t.id===p.id);
          const rows=t.rows.filter(r=>!r.section);
          const node=[...p.querySelectorAll('.matrix-cell')].find(n=>rows[+n.dataset.row].cells[t.columns[+n.dataset.column].sub]?.tooltip);
          return {node,family:rows[+node.dataset.row].family,model:rows[+node.dataset.row].model_label,sub:t.columns[+node.dataset.column].sub,row:node.dataset.row,column:node.dataset.column};
        """)
        expected_selection = selection(d)
        chosen["node"].click()
        popup = wait_for_popup(d)
        expected_text = popup.text
        geometry = d.execute_script("const r=arguments[0].querySelector('.drawer-panel').getBoundingClientRect();return {top:r.top,bottom:innerHeight-r.bottom,height:r.height,viewport:innerHeight};", popup)
        assert geometry["top"] >= geometry["viewport"] * .09, geometry
        assert geometry["bottom"] >= geometry["viewport"] * .09, geometry
        shared_url = d.current_url
        query = d.execute_script("return Object.fromEntries(new URLSearchParams(location.search))")
        assert query["tab"] == tab_id
        assert query["family_" + tab_id] == chosen["family"]
        assert query["model_" + tab_id] == chosen["model"]
        assert query["case_" + tab_id] == chosen["sub"]
        d.get(shared_url)
        assert wait_for_popup(d).text == expected_text
        assert selection(d) == expected_selection
        assert ("layout-vertical" in d.find_element(By.CSS_SELECTOR, ".tab-panel.active .comparison-ui").get_attribute("class")) == vertical
        assert not d.find_element(By.CSS_SELECTOR, ".tab-panel.active .legacy-report-details").get_attribute("open")
        d.find_element(By.CSS_SELECTOR, ".comparison-drawer.open .drawer-close").send_keys(Keys.ESCAPE)
        assert not d.find_elements(By.CSS_SELECTOR, ".comparison-drawer.open")
        closed_query = d.execute_script("return Object.fromEntries(new URLSearchParams(location.search))")
        assert "case_" + tab_id not in closed_query and "family_" + tab_id not in closed_query
        assert "model_" + tab_id not in closed_query
        focus = d.execute_script("return {row:document.activeElement.dataset.row,column:document.activeElement.dataset.column}")
        assert focus == {"row": chosen["row"], "column": chosen["column"]}
        d.get(d.current_url)
        WebDriverWait(d, 20).until(lambda browser: browser.find_elements(By.CSS_SELECTOR, ".tab-panel.active .comparison-matrix"))
        assert not d.find_elements(By.CSS_SELECTOR, ".comparison-drawer.open")
        invalid_url = rendered_page.resolve().as_uri() + "?tab=" + tab_id + "&case_" + tab_id + "=absent&family_" + tab_id + "=absent"
        d.get(invalid_url)
        WebDriverWait(d, 20).until(lambda browser: browser.find_elements(By.CSS_SELECTOR, ".tab-panel.active .comparison-matrix"))
        assert not d.find_elements(By.CSS_SELECTOR, ".comparison-drawer.open")
    finally:
        load(d, rendered_page)


@pytest.mark.parametrize("tab_id", ["tab-reasoning-batch", "tab-reasoning-stream"])
def test_cell_popup_url_preserves_reasoning_alias_row(matrix_driver, rendered_page, tab_id):
    d = matrix_driver
    load(d, rendered_page, "?tab=" + tab_id)
    try:
        chosen = d.execute_script("""
          const p=document.querySelector('.tab-panel.active'),t=window.__CONFORMANCE_PAGE.tabs.find(t=>t.id===p.id);
          const rows=t.rows.filter(r=>!r.section),row=rows.findIndex(r=>r.model_label==='DeepSeek V3.2');
          const node=[...p.querySelectorAll('.matrix-cell[data-row="'+row+'"]')].find(n=>rows[row].cells[t.columns[+n.dataset.column].sub]?.tooltip);
          return {node,row:String(row),column:node.dataset.column,family:rows[row].family,model:rows[row].model_label};
        """)
        chosen["node"].click()
        wait_for_popup(d)
        expected_description = d.find_element(By.CSS_SELECTOR, ".comparison-drawer.open [data-drawer-description]").text
        assert expected_description.startswith(chosen["model"])
        query = d.execute_script("return Object.fromEntries(new URLSearchParams(location.search))")
        assert query["model_" + tab_id] == chosen["model"]
        d.get(d.current_url)
        wait_for_popup(d)
        assert d.find_element(By.CSS_SELECTOR, ".comparison-drawer.open [data-drawer-description]").text == expected_description
        d.find_element(By.CSS_SELECTOR, ".comparison-drawer.open .drawer-close").send_keys(Keys.ESCAPE)
        focus = d.execute_script("return {row:document.activeElement.dataset.row,column:document.activeElement.dataset.column}")
        assert focus == {"row": chosen["row"], "column": chosen["column"]}
        closed_query = d.execute_script("return Object.fromEntries(new URLSearchParams(location.search))")
        assert "model_" + tab_id not in closed_query
        ambiguous = d.execute_script("const u=new URL(location.href);u.searchParams.set(arguments[0],arguments[1]);u.searchParams.set(arguments[2],arguments[3]);return u.href;",
                                     "family_" + tab_id, chosen["family"], "case_" + tab_id, query["case_" + tab_id])
        d.get(ambiguous)
        WebDriverWait(d, 20).until(lambda browser: browser.find_elements(By.CSS_SELECTOR, ".tab-panel.active .comparison-matrix"))
        assert not d.find_elements(By.CSS_SELECTOR, ".comparison-drawer.open")
    finally:
        load(d, rendered_page)
