# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the comparison surface through its real browser controls."""
import json
import re
import threading
from urllib.parse import urlsplit, parse_qs

import pytest
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.actions.wheel_input import ScrollOrigin
from selenium.webdriver.support.ui import Select, WebDriverWait

from test_browser_viewer import _driver, _server, _ready, _choose_engine, _clear_slot, _published_capture_index


def counts(driver):
    return driver.execute_script("return window.__conformanceComparison.counts")


@pytest.mark.parametrize("vertical", [False, True])
@pytest.mark.parametrize("stacked", [False, True])
def test_detailed_toggle_shows_test_names_and_descriptions(published_viewer, vertical, stacked):
    server, _ = _server(root=published_viewer)
    driver = _driver()
    try:
        driver.set_window_size(1200, 900)
        driver.get(f"http://127.0.0.1:{server.server_port}/?vertical={int(vertical)}")
        _ready(driver, "tab-unified", reference=False)
        _choose_engine(driver, 1, "sglang")
        _ready(driver, "tab-unified", reference=False)
        if stacked:
            driver.find_element("css selector", "[data-layout='stacked']").click()
        before = counts(driver)
        expected = driver.execute_script("""
            const model=window.__conformanceModel, pair=model.glossary.flatMap(group=>group.rows).find(pair=>pair[1]);
            const column=model.columns.find(column=>column.label===pair[0]);
            return {name:column?.case_id || model.case_prefix+pair[0],description:pair[1]};
        """)

        def assert_details(visible):
            sections = driver.find_elements("css selector", ".case-descriptions")
            assert len(sections) == 1
            section = sections[0]
            assert section.is_displayed() == visible
            assert expected["name"] in section.get_attribute("textContent")
            assert expected["description"] in section.get_attribute("textContent")
            result = driver.execute_script("""
                const table=document.querySelector('.comparison-matrix'), segments=[...table.querySelectorAll('.matrix-segments > i')];
                return {segments:segments.length, addedContent:segments.some(segment=>segment.childElementCount || segment.textContent.trim()),
                  after:Boolean(table.compareDocumentPosition(document.querySelector('.case-descriptions'))&Node.DOCUMENT_POSITION_FOLLOWING),
                  width:table.querySelector('.matrix-cell').getBoundingClientRect().width};
            """)
            assert result["segments"] > 0 and not result["addedContent"], result
            assert result["after"], result
            assert abs(result["width"] - 30) <= 1, result
            assert counts(driver) == before

        driver.find_element("css selector", "[data-view-detailed]").click()
        assert parse_qs(urlsplit(driver.current_url).query)["view"] == ["details"]
        assert_details(True)
        driver.set_window_size(480, 900)
        assert driver.execute_script("""
            const table=document.querySelector('.case-descriptions .glossary'), rect=table.getBoundingClientRect();
            return rect.left>=0 && rect.right<=innerWidth && table.scrollWidth<=table.clientWidth+1;
        """)
        driver.set_window_size(1200, 900)
        driver.refresh()
        _ready(driver, "tab-unified", reference=False)
        assert driver.find_element("css selector", "[data-view-detailed]").is_selected()
        assert_details(True)
        driver.find_element("css selector", "[data-view-detailed]").click()
        assert "view" not in parse_qs(urlsplit(driver.current_url).query)
        assert_details(False)
    finally:
        driver.quit(); server.shutdown(); server.server_close()


@pytest.mark.parametrize("vertical", [False, True])
@pytest.mark.parametrize("stacked", [False, True])
def test_matrix_uses_page_scrolling(published_viewer, vertical, stacked):
    server, _ = _server(root=published_viewer)
    driver = _driver()
    try:
        driver.set_window_size(800, 600)
        driver.get(f"http://127.0.0.1:{server.server_port}/?vertical={int(vertical)}")
        _ready(driver, "tab-unified", reference=False)
        _choose_engine(driver, 1, "sglang")
        _ready(driver, "tab-unified", reference=False)
        if stacked:
            driver.find_element("css selector", "[data-layout='stacked']").click()
        geometry = driver.execute_script("""
            const wrapper=document.querySelector('.comparison-matrix-scroll');
            wrapper.scrollIntoView({block:'start'});
            wrapper.scrollTop=200;wrapper.scrollLeft=200;
            return {x:wrapper.scrollLeft,y:wrapper.scrollTop};
        """)
        assert geometry["x"] == geometry["y"] == 0, geometry
        before = driver.execute_script("return scrollY")
        ActionChains(driver).scroll_from_origin(ScrollOrigin.from_viewport(400, 300), 0, 200).perform()
        WebDriverWait(driver, 5).until(lambda page: page.execute_script("return scrollY>arguments[0]", before))
        driver.execute_script("window.scrollBy(200,0)")
        geometry = driver.execute_script("""
            const table=document.querySelector('.comparison-matrix'), wrapper=table.parentElement;
            const header=table.tHead.rows[0].cells[0].getBoundingClientRect();
            const labels=[...table.tBodies[0].querySelectorAll('tr > th:first-child')]
              .map(cell=>cell.getBoundingClientRect()).filter(rect=>rect.top>=header.bottom&&rect.bottom<innerHeight);
            return {top:header.top,labels:labels.map(rect=>rect.left),x:scrollX,y:scrollY,
              overflow:document.scrollingElement.scrollWidth>innerWidth,innerX:wrapper.scrollLeft,innerY:wrapper.scrollTop};
        """)
        assert abs(geometry["top"]) <= 1, geometry
        assert geometry["labels"], geometry
        if geometry["x"]:
            assert all(abs(left) <= 1 for left in geometry["labels"]), geometry
        assert bool(geometry["x"]) == geometry["overflow"], geometry
        assert geometry["innerX"] == geometry["innerY"] == 0, geometry
    finally:
        driver.quit(); server.shutdown(); server.server_close()


@pytest.mark.parametrize("vertical", [False, True])
@pytest.mark.parametrize("stacked", [False, True])
def test_matrix_columns_keep_minimal_width(published_viewer, vertical, stacked):
    server, _ = _server(root=published_viewer)
    driver = _driver()
    try:
        driver.get(f"http://127.0.0.1:{server.server_port}/?vertical={int(vertical)}")
        _ready(driver, "tab-unified", reference=False)
        if stacked:
            _choose_engine(driver, 1, "sglang")
            _ready(driver, "tab-unified", reference=False)
            driver.find_element("css selector", "[data-layout='stacked']").click()
        for width, label_width in [(2048, 220), (480, 150)]:
            driver.set_window_size(width, 1000)
            geometry = driver.execute_script("""
                const table=document.querySelector('.comparison-matrix');
                return {label:table.tHead.rows[0].cells[0].getBoundingClientRect().width,
                  cells:[...table.querySelectorAll('.matrix-cell')].map(c=>c.getBoundingClientRect().width),
                  header:table.tHead.getBoundingClientRect().height};
            """)
            assert abs(geometry["label"] - label_width) <= 1, geometry
            assert geometry["cells"] and all(abs(w - 30) <= 1 for w in geometry["cells"]), geometry
            if not vertical:
                assert geometry["header"] <= 100, geometry
            assert not driver.find_elements("css selector", ".matrix-summary")
            assert "Conformance counts" not in driver.find_element("css selector", ".comparison-matrix").text
    finally:
        driver.quit(); server.shutdown(); server.server_close()


@pytest.mark.parametrize("vertical", [False, True])
def test_stacked_families_are_grouped_and_legend_matches_visible_colors(published_viewer, vertical):
    server, _ = _server(root=published_viewer)
    driver = _driver()
    try:
        driver.set_window_size(1600, 1000)
        driver.get(f"http://127.0.0.1:{server.server_port}/?vertical={int(vertical)}&layout_tab-unified=stacked")
        _ready(driver, "tab-unified", reference=False)
        _choose_engine(driver, 1, "sglang")
        _ready(driver, "tab-unified", reference=False)
        _choose_engine(driver, 2, "vllm")
        _ready(driver, "tab-unified", reference=False)

        def assert_groups_and_legend():
            result = driver.execute_script("""
              const table=document.querySelector('.comparison-matrix');
              const familyRows=window.__conformanceModel.rows;
              const counts=new Map();
              for(const row of familyRows){const family=row.family||row.model_label;counts.set(family,(counts.get(family)||0)+1)}
              const statuses=new Set([...table.querySelectorAll('.matrix-segments > i')]
                .flatMap(node=>[...node.classList].filter(name=>name.startsWith('status-')).map(name=>name.slice(7))));
              const legend=new Set([...document.querySelectorAll('.comparison-legend [data-status]')].map(node=>node.dataset.status));
              const after=Boolean(document.querySelector('.comparison-matrix-scroll').compareDocumentPosition(
                document.querySelector('.comparison-legend')) & Node.DOCUMENT_POSITION_FOLLOWING);
              if(!document.body.classList.contains('transpose-mode')){
                const rows=[...table.tBodies[0].querySelectorAll('tr[data-family]')],groups=[];
                for(const row of rows){const family=row.dataset.family,last=groups.at(-1);if(last?.family===family)last.rows++;
                  else groups.push({family,rows:1,start:row.classList.contains('matrix-family-start')})}
                const boundaries=getComputedStyle(table.tBodies[0].querySelector('tr.matrix-family-start').firstElementChild);
                return {statuses:[...statuses].sort(),legend:[...legend].sort(),after,
                  grouped:groups.length===counts.size&&groups.every(group=>group.rows===3*counts.get(group.family)),
                  separators:groups.slice(1).every(group=>group.start),separatorWidth:boundaries.borderTopWidth};
              }
              const heads=[...table.tHead.querySelectorAll('.matrix-family-heading')];
              const expected=[...counts.values()].map(count=>3*count);
              const widths=heads.map(head=>head.colSpan);
              const ends=[...table.tHead.querySelectorAll('.matrix-capture-heading.matrix-family-end')];
              const bodyEnds=[...table.tBodies[0].querySelectorAll('tr:not(.matrix-category-row) td.matrix-family-end')];
              const categoryRows=[...table.tBodies[0].querySelectorAll('.matrix-category-row')];
              return {statuses:[...statuses].sort(),legend:[...legend].sort(),after,
                grouped:heads.length===counts.size&&widths.every((width,index)=>width===expected[index]),
                separators:ends.length===heads.length-1&&bodyEnds.length>0&&categoryRows.length>0
                  &&categoryRows.every(row=>row.querySelectorAll('td.matrix-family-end').length===heads.length-1),
                separatorWidth:getComputedStyle(ends[0]).borderRightWidth};
            """)
            assert result["statuses"] == result["legend"], result
            assert result["after"], result
            assert result["grouped"], result
            assert result["separators"], result
            assert result["separatorWidth"] == "3px", result

        assert_groups_and_legend()
        driver.find_element("css selector", "[data-case-search]").send_keys("dotted_name_minimal")
        assert_groups_and_legend()
    finally:
        driver.quit(); server.shutdown(); server.server_close()


@pytest.mark.parametrize("width", [1600, 480])
@pytest.mark.parametrize("captures", [1, 2])
def test_long_case_hover_fits_viewport(published_viewer, width, captures):
    server, _ = _server(root=published_viewer)
    driver = _driver()
    try:
        driver.set_window_size(width, 1000)
        driver.get(f"http://127.0.0.1:{server.server_port}/?vertical=0")
        _ready(driver, "tab-unified", reference=False)
        if captures == 2:
            _choose_engine(driver, 1, "sglang")
            _ready(driver, "tab-unified", reference=False)
            driver.find_element("css selector", "[data-layout='stacked']").click()
        label = driver.execute_script("""
            const headers=[...document.querySelectorAll('.matrix-case-heading')];
            const header=headers.sort((a,b)=>window.__conformanceTooltipModel(b).popupCase.label.length
              -window.__conformanceTooltipModel(a).popupCase.label.length)[0];
            header.scrollIntoView({block:'center',inline:'end'});
            return header.querySelector('[data-hover-hint]');
        """)
        ActionChains(driver).move_to_element(label).perform()
        WebDriverWait(driver, 5).until(lambda d: d.execute_script(
            "return getComputedStyle(arguments[0],'::after').opacity==='1'", label))
        geometry = driver.execute_script("""
            const a=arguments[0], r=a.getBoundingClientRect(), s=getComputedStyle(a,'::after');
            return {x:r.left+parseFloat(s.left),y:r.top+parseFloat(s.top),w:parseFloat(s.width),h:parseFloat(s.height),
              left:0,right:innerWidth,top:0,bottom:innerHeight};
        """, label)
        assert geometry['left'] <= geometry['x'] and geometry['x'] + geometry['w'] <= geometry['right'], geometry
        assert geometry['top'] <= geometry['y'] and geometry['y'] + geometry['h'] <= geometry['bottom'], geometry
    finally:
        driver.quit(); server.shutdown(); server.server_close()


@pytest.mark.parametrize("vertical", [False, True])
@pytest.mark.parametrize("correction", ["layout", "categories", "labels"])
def test_requested_matrix_corrections(published_viewer, vertical, correction):
    server, _ = _server(root=published_viewer)
    driver = _driver()
    try:
        driver.set_window_size(1600, 1000)
        driver.get(f"http://127.0.0.1:{server.server_port}/?vertical={int(vertical)}&layout_tab-unified=stacked")
        _ready(driver, "tab-unified", reference=False)
        if correction == "layout":
            assert not driver.find_element("css selector", "[data-layout='stacked']").is_displayed()
            before = counts(driver)
            _choose_engine(driver, 1, "sglang")
            _ready(driver, "tab-unified", reference=False)
            stacked = driver.find_element("css selector", "[data-layout='stacked']")
            assert stacked.is_displayed()
            assert stacked.text == ("Stacked columns" if vertical else "Stacked rows")
            assert stacked.get_attribute("aria-pressed") == "true"
            if vertical:
                headings = driver.find_elements("css selector", ".comparison-matrix thead tr:nth-child(2) .matrix-capture-heading")
                assert headings
                assert all("matches GOLDEN" not in heading.text and "cases" not in heading.text for heading in headings)
            driver.find_element("css selector", "[data-layout='split']").click()
            assert counts(driver)["dynamo"] == before["dynamo"]
            _clear_slot(driver, 1)
            _ready(driver, "tab-unified", reference=False)
            assert not driver.find_element("css selector", "[data-layout='stacked']").is_displayed()
        elif correction == "categories":
            assert not driver.find_elements("css selector", ".category-chips")
            button = driver.find_element("css selector", "[data-comparison-table] [data-group]")
            assert driver.execute_script("return getComputedStyle(arguments[0]).borderTopWidth", button) == "0px"
            key = button.get_attribute("data-group")
            before = counts(driver)
            button.send_keys(Keys.SPACE)
            selector = f"[data-comparison-table] [data-group='{key}']"
            assert driver.find_element("css selector", selector).get_attribute("aria-pressed") == "false"
            reduced = counts(driver)
            assert reduced["dynamo"]["total"] < before["dynamo"]["total"]
            driver.refresh()
            _ready(driver, "tab-unified", reference=False)
            assert counts(driver) == reduced
            driver.find_element("css selector", selector).send_keys(Keys.SPACE)
            assert counts(driver) == before
            driver.set_window_size(1600, 500)
            driver.execute_script("document.querySelector('.comparison-matrix-scroll').scrollIntoView({block:'start'});window.scrollBy(0,200)")
            WebDriverWait(driver, 5).until(lambda d: d.execute_script("""
                const table=document.querySelector('[data-comparison-table]');
                const top=table.tHead.rows[0].cells[0].getBoundingClientRect();
                if (Math.abs(top.top)>1) return false;
                if (table.tHead.rows.length === 1) return true;
                const next=table.tHead.rows[1].cells[0].getBoundingClientRect();
                return Math.abs(next.top-top.bottom)<=1;
            """))
        else:
            header = driver.find_element("css selector", ".matrix-case-heading")
            full = driver.execute_script("return window.__conformanceTooltipModel(arguments[0]).popupCase.label", header)
            label = header.find_element("css selector", "[data-hover-hint]")
            assert label.get_attribute("data-hover-hint") == "Click to show " + full
            if vertical:
                assert full in header.text
            else:
                assert re.fullmatch(r"\d+-\d+", header.text)
                assert full not in header.text
                assert header.rect["width"] < 100
                ActionChains(driver).move_to_element(label).perform()
                WebDriverWait(driver, 5).until(lambda d: d.execute_script(
                    "return getComputedStyle(arguments[0],'::after').visibility==='visible' && getComputedStyle(arguments[0],'::after').opacity==='1'", label))
                assert full in driver.execute_script("return getComputedStyle(arguments[0],'::after').content", label)
                assert driver.execute_script("""
                    const a=arguments[0], r=a.getBoundingClientRect(), s=getComputedStyle(a,'::after');
                    const x=r.left+parseFloat(s.left), y=r.top+parseFloat(s.top);
                    return x>=0 && x+parseFloat(s.width)<=innerWidth
                      && y>=0 && y+parseFloat(s.height)<=innerHeight;
                """, label)
                assert not driver.find_elements("css selector", ".ttip-pinned")
            header.send_keys(Keys.ENTER)
            assert driver.find_element("css selector", ".ttip-pinned .ttip-head").text
    finally:
        driver.quit(); server.shutdown(); server.server_close()


@pytest.mark.parametrize("query,vertical", [("", True), ("?vertical=0", False), ("?transpose=0", False)])
def test_defaults_before_first_paint_and_shared_state(published_viewer, query, vertical):
    server, _ = _server(root=published_viewer)
    driver = _driver()
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {"source": """
      window.firstComparisonPaint = null;
      new MutationObserver(() => {
        if (window.firstComparisonPaint === null && document.querySelector('.comparison-matrix'))
          window.firstComparisonPaint = document.body.classList.contains('transpose-mode');
      }).observe(document, {childList:true,subtree:true});
    """})
    try:
        driver.set_window_size(1600,1000)
        driver.get(f"http://127.0.0.1:{server.server_port}/{query}")
        _ready(driver, "tab-unified",reference=False)
        assert driver.execute_script("return window.firstComparisonPaint") is vertical
        assert len(driver.find_elements("css selector", ".capture-slot")) == 3
        assert len(driver.find_elements("css selector", ".tabs .tab-button")) == 0
        sha = driver.find_element("css selector", ".source-commit")
        assert re.fullmatch(r"[a-f0-9]{12}", sha.text)
        assert sha.get_attribute("title") in sha.get_attribute("href")
        assert sha.get_attribute("title") in sha.get_attribute("aria-label")
        assert not driver.find_element("css selector", "[data-layout='stacked']").is_displayed()
        assert "%" not in driver.find_element("id", "viewer").text
        assert not driver.find_elements("css selector", ".legacy-report-details, [data-peer-only]")
        assert driver.execute_script("return [...document.querySelectorAll('.capture-summary [data-count]')].every(node => Number(node.textContent.trim().split(' ')[0]) > 0)")
        before = counts(driver)
        driver.find_element("css selector", "[data-focus='problems']").click()
        assert counts(driver) == before
        search = driver.find_element("css selector", "[data-case-search]")
        search.send_keys("dotted_name_minimal")
        after = counts(driver)
        assert 0 < after["dynamo"]["total"] < before["dynamo"]["total"]
        assert driver.find_element("css selector", "[data-focus='problems']").get_attribute("aria-pressed") == "true"
        assert driver.find_element("css selector", "[data-case-search]").get_attribute("value") == "dotted_name_minimal"
        driver.find_element("css selector", "[data-theme-toggle]").click()
        assert counts(driver) == after
        driver.set_window_size(480,850)
        # The matrix scrolls with the page; the capture/menu header must still fit.
        assert driver.execute_script("""
            const header=document.querySelector('.capture-slots'), rect=header.getBoundingClientRect();
            return rect.left>=0 && rect.right<=innerWidth && header.scrollWidth<=header.clientWidth+1;
        """)
        for card in driver.find_elements("css selector", ".capture-slot"):
            assert card.rect["width"] <= 480
    finally:
        driver.quit(); server.shutdown(); server.server_close()


def test_vertical_stacked_problems_focus_dims_uncaptured_outputs(published_viewer):
    server, _ = _server(root=published_viewer)
    driver = _driver()
    try:
        driver.set_window_size(1920, 1100)
        driver.get(f"http://127.0.0.1:{server.server_port}/?vertical=1&layout_tab-unified=stacked")
        _ready(driver, "tab-unified", reference=False)
        _choose_engine(driver, 1, "dynamo")
        _ready(driver, "tab-unified", reference=False)
        driver.find_element("css selector", "[data-focus='problems']").click()
        result = driver.execute_script("""
            const statuses = ['documented','diverge','error','leak'];
            const cells = [...document.querySelectorAll('.comparison-matrix .matrix-cell:not(.filtered-cell)')];
            return cells.map(cell => {
                const outcomes = [...cell.querySelectorAll('.matrix-segments > i')]
                  .flatMap(node => [...node.classList].filter(name => name.startsWith('status-')).map(name => name.slice(7)));
                return {outcomes, dimmed: cell.classList.contains('focus-dim'), hasProblem: outcomes.some(status => statuses.includes(status))};
            });
        """)
        assert result
        assert all(item["dimmed"] == (not item["hasProblem"]) for item in result), result
        assert any("missing" in item["outcomes"] and item["dimmed"] for item in result)
    finally:
        driver.quit(); server.shutdown(); server.server_close()


def test_capture_cards_and_menu_share_one_row_and_notice_stays_at_bottom(published_viewer):
    server, _ = _server(root=published_viewer)
    driver = _driver()
    try:
        driver.set_window_size(1600, 1000)
        driver.get(f"http://127.0.0.1:{server.server_port}/?tab=tab-unified")
        _ready(driver, "tab-unified", reference=False)
        _choose_engine(driver, 1, "sglang")
        _ready(driver, "tab-unified", reference=False)
        _choose_engine(driver, 2, "vllm")
        _ready(driver, "tab-unified", reference=False)
        notice = WebDriverWait(driver, 10).until(
            lambda d: d.find_element("css selector", ".publication-history-notice"))
        before = counts(driver)
        for width in (2048, 1600, 1024, 480):
            driver.set_window_size(width, 1000)
            for theme in ("light", "dark"):
                driver.execute_script("document.documentElement.dataset.theme=arguments[0]", theme)
                geometry = driver.execute_script("""
                  const rect=node=>{const r=node.getBoundingClientRect();return {x:r.x,y:r.y,right:r.right,bottom:r.bottom}};
                  const menu=document.querySelector('.comparison-toolbar'), cards=[...document.querySelectorAll('.capture-slot')];
                  return {cards:cards.map(rect), menu:rect(menu),
                    matrix:rect(document.querySelector('.comparison-matrix-scroll')),
                    notice:rect(document.querySelector('.publication-history-notice')),
                    overflow:[menu,...cards].some(node=>node.scrollWidth>node.clientWidth+1),
                    optionsParent:menu.querySelector('.tabs-right').parentElement.className,
                    lastChild:document.querySelector('#viewer').lastElementChild.className};
                """)
                cards, menu = geometry["cards"], geometry["menu"]
                assert len(cards) == 3, geometry
                assert geometry["optionsParent"] == "comparison-toolbar", geometry
                assert not geometry["overflow"], geometry
                if width > 900:
                    assert all(abs(card["y"] - menu["y"]) <= 1 for card in cards), geometry
                    assert all(cards[i]["right"] < cards[i + 1]["x"] for i in range(2)), geometry
                    assert cards[-1]["right"] < menu["x"], geometry
                else:
                    assert menu["y"] >= max(card["bottom"] for card in cards), geometry
                    if width <= 700:
                        assert all(cards[i]["bottom"] < cards[i + 1]["y"] for i in range(2)), geometry
                assert geometry["notice"]["y"] > geometry["matrix"]["bottom"], geometry
                assert geometry["lastChild"] == "capture-slots-notice publication-history-notice", geometry
                assert counts(driver) == before
        assert notice.text == "Latest results are available offline. Older results are not included here."
    finally:
        driver.quit(); server.shutdown(); server.server_close()


@pytest.mark.parametrize("vertical", [False, True])
@pytest.mark.parametrize("width", [1600, 480])
def test_matrix_keyboard_popup_and_reload(published_viewer, vertical, width):
    server, _ = _server(root=published_viewer)
    driver = _driver()
    try:
        driver.set_window_size(width, 1000)
        driver.get(f"http://127.0.0.1:{server.server_port}/?vertical={int(vertical)}")
        _ready(driver,"tab-unified",reference=False)
        cell = driver.find_element("css selector", ".comparison-matrix .matrix-cell[data-ttip-id]")
        cell.send_keys(Keys.ENTER)
        popup = WebDriverWait(driver,10).until(lambda d: d.find_element("css selector", ".comparison-output.ttip-pinned"))
        assert driver.execute_script("const r=arguments[0].getBoundingClientRect(); return Math.abs(r.left+r.width/2-innerWidth/2)<=1", popup)
        assert popup.find_elements("css selector", ".golden-out-cap, [data-cand=golden]")
        assert parse_qs(urlsplit(driver.current_url).query)["popup"]
        driver.refresh(); _ready(driver,"tab-unified",reference=False)
        assert len(driver.find_elements("css selector", ".comparison-output.ttip-pinned")) == 1
        driver.find_element("tag name","body").send_keys(Keys.ESCAPE)
        assert not driver.find_elements("css selector", ".comparison-output.ttip-pinned")
        assert "popup" not in parse_qs(urlsplit(driver.current_url).query)
    finally:
        driver.quit(); server.shutdown(); server.server_close()


def test_reference_url_same_engine_and_stale_request(historical_published_viewer):
    started, release = threading.Event(), threading.Event()
    server, _ = _server(root=historical_published_viewer, gate=(started,release),
                        gate_path=_published_capture_index(historical_published_viewer,"dynamo_v1","5.1.3"))
    driver = _driver()
    try:
        driver.get(f"http://127.0.0.1:{server.server_port}/?tab=tab-unified&slots_tab-unified=dynamo%2Cdynamo%400.7.18%2C&cmp_tab-unified=dynamo%2Cdynamo%400.7.18")
        _ready(driver,"tab-unified",reference=False)
        selected = driver.execute_script("return window.__conformanceController.selection(document.getElementById('tab-unified')).selected")
        assert selected == ["dynamo","dynamo@0.7.18"]
        assert "0.7.18" in driver.find_element("css selector","[data-slot-capture='1']").get_attribute("value")
        duplicate = driver.find_element("css selector","[data-slot-capture='0'] option[value='dynamo@0.7.18']")
        assert not duplicate.is_enabled()
        before = counts(driver)
        Select(driver.find_element("css selector","[data-slot-capture='0']")).select_by_value("dynamo_v1@5.1.3")
        assert started.wait(20)
        assert counts(driver) == before
        Select(driver.find_element("css selector","[data-slot-capture='0']")).select_by_value("dynamo")
        WebDriverWait(driver,20).until(lambda d: d.execute_script("return document.getElementById('tab-unified').dataset.loadState === 'ready'"))
        expected_url, expected_counts = driver.current_url, counts(driver)
        release.set()
        WebDriverWait(driver,20).until(lambda d: d.execute_script("return window.__conformanceLoader.pending === 0"))
        assert counts(driver) == expected_counts
        assert driver.current_url == expected_url
        assert driver.execute_script("return window.__conformanceModel.candidates.some(c => c.key === 'dynamo')")
    finally:
        release.set(); driver.quit(); server.shutdown(); server.server_close()
