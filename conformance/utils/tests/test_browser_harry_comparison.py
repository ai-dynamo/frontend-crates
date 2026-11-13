# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the comparison surface through its real browser controls."""
import json
import re
import threading
from urllib.parse import urlsplit, parse_qs, urlencode

import pytest
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.actions.wheel_input import ScrollOrigin
from selenium.webdriver.support.ui import Select, WebDriverWait

from test_browser_viewer import _driver, _server, _ready, _choose_engine, _clear_slot, _published_capture_index, _published_capture


def counts(driver):
    return driver.execute_script("return window.__conformanceComparison.counts")


def test_boolean_search_filters_cases_counts_and_restores_urls(published_viewer):
    server, _ = _server(root=published_viewer)
    driver = _driver()
    try:
        driver.get(f"http://127.0.0.1:{server.server_port}/")
        _ready(driver, "tab-unified")
        before = counts(driver)
        families = driver.execute_script("return window.__conformanceComparison.view.rows.map(row=>row.family)")
        individual_case = driver.execute_script("""
            const model=window.__conformanceModel;
            const column=model.columns.find(column=>column.case_id==='UNIFIED.10-reason_then_content_interrupted');
            const cells=model.rows.map(row=>row.cells[column.sub]).filter(Boolean);
            return {description:column.desc,id:column.case_id,
                independent:cells.length>0&&cells.every(cell=>!cell.variants)};
        """)
        assert individual_case["description"]
        assert individual_case["independent"]

        def search(value):
            field = driver.find_element("css selector", "[data-case-search]")
            field.send_keys(Keys.CONTROL, "a", Keys.NULL)
            field.send_keys(value or Keys.BACKSPACE)
            return field

        search("TC Trun")
        assert driver.execute_script("return window.__conformanceComparison.view.rows.map(row=>row.family)") == families
        assert 0 < counts(driver)["dynamo"]["total"] < before["dynamo"]["total"]
        search('model:beam AND TC Trun')
        assert driver.execute_script("return window.__conformanceComparison.view.rows.map(row=>row.family)") == ["beam"]
        assert counts(driver)["dynamo"]["total"] > 0
        search("beam")
        unscoped = counts(driver)
        search("model:beam")
        assert driver.execute_script("return window.__conformanceComparison.view.rows.map(row=>row.family)") == ["beam"]
        assert counts(driver) == unscoped
        search('model:"beam" AND call')
        assert driver.execute_script("return window.__conformanceComparison.view.rows.map(row=>row.family)") == ["beam"]
        assert counts(driver)["dynamo"]["total"] > 0
        search('"dotted_name_minimal"')
        family = driver.execute_script("return window.__conformanceComparison.view.rows[0].family")
        query = f'model:{family} AND ("dotted_name_minimal" OR __missing_case__)'
        field = search(query)
        after = counts(driver)
        assert 0 < after["dynamo"]["total"] < before["dynamo"]["total"]
        assert driver.execute_script("return window.__conformanceComparison.view.rows.map(row=>row.family)") == [family]
        assert field.get_attribute("aria-invalid") == "false"
        assert parse_qs(urlsplit(driver.current_url).query)["search_tab-unified"] == [query]
        field.send_keys(" (")
        assert field.get_attribute("aria-invalid") == "true"
        assert driver.find_element("css selector", ".comparison-search-error").is_displayed()
        assert counts(driver) == after
        search(query)
        driver.refresh()
        _ready(driver, "tab-unified")
        assert counts(driver) == after
        assert driver.find_element("css selector", "[data-case-search]").get_attribute("value") == query
        driver.find_element("css selector", "[data-transpose-toggle]").click()
        assert counts(driver) == after
        for value in ["description:" + json.dumps(individual_case["description"], ensure_ascii=False), individual_case["id"]]:
            search(value)
            assert counts(driver)["dynamo"]["total"] > 0
        search('"__no_such_test__"')
        assert counts(driver)["dynamo"]["total"] == 0
        search("")
        assert counts(driver) == before
        assert not driver.find_element("css selector", ".comparison-search-error").is_displayed()
        field = search("beam (TC)")
        field.send_keys(Keys.HOME, *([Keys.ARROW_RIGHT] * 6), "Trun ")
        assert field.get_attribute("value") == "beam (Trun TC)"
        assert driver.execute_script("return arguments[0].selectionStart", field) == len("beam (Trun ")
        field = search("beam TC ")
        field.send_keys(Keys.HOME, "-")
        assert field.get_attribute("value") == "-beam TC "
        assert driver.execute_script("return arguments[0].selectionStart", field) == 1
        search("")
        assert counts(driver) == before
    finally:
        driver.quit(); server.shutdown(); server.server_close()


@pytest.mark.parametrize("vertical", [False, True])
@pytest.mark.parametrize("stacked", [False, True])
def test_search_axes_categories_and_numbered_share_links(published_viewer, vertical, stacked):
    server, _ = _server(root=published_viewer)
    driver = _driver()
    try:
        driver.get(f"http://127.0.0.1:{server.server_port}/?vertical={int(vertical)}")
        _ready(driver, "tab-unified")
        _choose_engine(driver, 1, "sglang")
        _ready(driver, "tab-unified")
        if stacked:
            driver.find_element("css selector", "[data-layout='stacked']").click()
        baseline = counts(driver)
        families = driver.execute_script("return window.__conformanceComparison.view.rows.map(row=>row.family)")

        def search(value):
            field = driver.find_element("css selector", "[data-case-search]")
            field.send_keys(Keys.CONTROL, "a", Keys.NULL)
            field.send_keys(value or Keys.BACKSPACE)
            assert field.get_attribute("aria-invalid") == "false", value
            return field

        for query in ["guided decoding", '"guided decoding"', "TC Trun", "TC OR guided"]:
            search(query)
            actual = driver.execute_script("""
                const {view}=window.__conformanceComparison, model=window.__conformanceModel;
                return {families:view.rows.map(row=>row.family),
                  groups:[...document.querySelectorAll('.comparison-matrix [data-group]')].map(b=>b.dataset.group),
                  expected:[...new Set(view.columns.map(c=>c.group_key))],
                  placeholders:[...document.querySelectorAll('.comparison-matrix thead th')].filter(th=>th.textContent==='—').length};
            """)
            assert actual["families"] == families, query
            assert actual["groups"] == actual["expected"], query
            assert not actual["placeholders"], query
            assert 0 < counts(driver)["dynamo"]["total"] < baseline["dynamo"]["total"]
        for query in ["beam", "parser:beam", "beam (guided decoding OR TC)", "beam -col:TC", "deep seek TC"]:
            search(query)
            shown = driver.execute_script("return window.__conformanceComparison.view.rows.map(row=>row.family)")
            assert shown and all(family.startswith("deepseek" if query.startswith("deep seek") else "beam") for family in shown), query
            assert counts(driver)["dynamo"]["total"] > 0, query
        search("beam OR TC")
        assert driver.execute_script("return window.__conformanceComparison.view.rows.map(row=>row.family)") == families

        search("TC Trun")
        before = counts(driver)
        group = driver.execute_script("return window.__conformanceComparison.view.columns[0].group_key")
        driver.set_window_size(480, 900)
        driver.find_element("css selector", ".comparison-categories summary").click()
        assert driver.execute_script("const r=document.querySelector('[data-category-options]').getBoundingClientRect(); return r.left>=0 && r.right<=innerWidth")
        toggle = driver.find_element("css selector", f'[data-category-options] [data-group="{group}"]')
        toggle.click()
        assert counts(driver)["dynamo"]["total"] == 0
        assert toggle.get_attribute("aria-pressed") == "false"
        toggle.click()
        assert counts(driver) == before
        driver.find_element("css selector", ".comparison-categories summary").click()

        driver.set_window_size(1200, 900)
        search("")
        case = driver.execute_script("""
            const column=window.__conformanceModel.columns.find(c=>c.display_number>=10 && /^\\d+$/.test(c.compact_prefix));
            return {alias:column.compact_prefix+'-'+column.display_number,sub:column.sub,id:column.case_id.replace(/^UNIFIED\\./,'')};
        """)
        field = search(case["alias"])
        canonical = json.dumps(case["id"])
        selected = counts(driver)
        assert field.get_attribute("value") == case["alias"]
        assert parse_qs(urlsplit(driver.current_url).query)["search_tab-unified"] == [canonical]
        field.send_keys(Keys.ENTER)
        assert field.get_attribute("value") == canonical
        assert counts(driver) == selected
        expected = counts(driver)
        assert expected["dynamo"]["total"] > 0
        shared_url = driver.current_url
        field = search('"' + case["alias"] + '"')
        field.send_keys(Keys.TAB)
        assert field.get_attribute("value") == '"' + case["id"] + '"'
        assert counts(driver) == expected
        field = search("beam (TC OR nowhere)")
        field.send_keys(Keys.HOME, *([Keys.ARROW_RIGHT] * 6), Keys.SHIFT, Keys.ARROW_RIGHT, Keys.ARROW_RIGHT, Keys.NULL, case["alias"])
        assert field.get_attribute("value") == f'beam ({case["alias"]} OR nowhere)'
        assert driver.execute_script("return arguments[0].selectionStart", field) == len("beam (") + len(case["alias"])
        field.send_keys(Keys.ENTER)
        assert field.get_attribute("value") == f'beam ({canonical} OR nowhere)'
        driver.get(shared_url)
        _ready(driver, "tab-unified")
        assert counts(driver) == expected
        assert driver.find_element("css selector", "[data-case-search]").get_attribute("value") == canonical
        search("999-999")
        assert counts(driver)["dynamo"]["total"] == 0
        assert "not numbered" in driver.find_element("css selector", ".comparison-search-error").text
        search("")
        assert counts(driver) == baseline
    finally:
        driver.quit(); server.shutdown(); server.server_close()


@pytest.mark.parametrize("vertical", [False, True])
@pytest.mark.parametrize("stacked", [False, True])
def test_detailed_toggle_shows_test_names_and_descriptions(published_viewer, vertical, stacked):
    server, _ = _server(root=published_viewer)
    driver = _driver()
    try:
        driver.set_window_size(1200, 900)
        driver.get(f"http://127.0.0.1:{server.server_port}/?vertical={int(vertical)}")
        _ready(driver, "tab-unified")
        _choose_engine(driver, 1, "sglang")
        _ready(driver, "tab-unified")
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
                  width:table.querySelector('.matrix-cell').getBoundingClientRect().width,
                  markWidth:table.querySelector('.matrix-segments').getBoundingClientRect().width,
                  fontSize:parseFloat(getComputedStyle(table).fontSize)};
            """)
            assert result["segments"] > 0 and not result["addedContent"], result
            assert result["after"], result
            if vertical and stacked:
                assert 0 <= result['width'] - result['markWidth'] <= result['fontSize'], result
            else:
                assert abs(result["width"] - (42 if vertical else 30)) <= 1, result
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
        _ready(driver, "tab-unified")
        assert driver.find_element("css selector", "[data-view-detailed]").is_selected()
        assert_details(True)
        driver.find_element("css selector", "[data-view-detailed]").click()
        assert "view" not in parse_qs(urlsplit(driver.current_url).query)
        assert_details(False)
    finally:
        driver.quit(); server.shutdown(); server.server_close()


@pytest.mark.parametrize("vertical", [False, True])
@pytest.mark.parametrize("stacked", [False, True])
@pytest.mark.parametrize("width", [1600, 800, 480])
def test_matrix_uses_page_scrolling(published_viewer, vertical, stacked, width):
    server, _ = _server(root=published_viewer)
    driver = _driver()
    try:
        driver.set_window_size(width, 600)
        driver.get(f"http://127.0.0.1:{server.server_port}/?vertical={int(vertical)}")
        _ready(driver, "tab-unified")
        _choose_engine(driver, 1, "sglang")
        _ready(driver, "tab-unified")
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
              .map(cell=>({rect:cell.getBoundingClientRect(),position:getComputedStyle(cell).position}))
              .filter(({rect})=>rect.top>=header.bottom&&rect.bottom<innerHeight);
            return {top:header.top,labels:labels.map(({rect})=>rect.left),x:scrollX,y:scrollY,
              positions:labels.map(label=>label.position),tableLeft:table.getBoundingClientRect().left,
              overflow:document.scrollingElement.scrollWidth>document.documentElement.clientWidth,
              innerX:wrapper.scrollLeft,innerY:wrapper.scrollTop};
        """)
        assert abs(geometry["top"]) <= 1, geometry
        assert geometry["labels"], geometry
        sticky = not vertical or width > 1280
        assert all(position == ('sticky' if sticky else 'static') for position in geometry['positions']), geometry
        label_left = max(0, geometry['tableLeft']) if sticky else geometry['tableLeft']
        assert all(abs(left - label_left) <= 1 for left in geometry["labels"]), geometry
        assert bool(geometry["x"]) == geometry["overflow"], geometry
        assert geometry["innerX"] == geometry["innerY"] == 0, geometry
    finally:
        driver.quit(); server.shutdown(); server.server_close()


@pytest.mark.parametrize("vertical", [False, True])
@pytest.mark.parametrize("stacked", [False, True])
def test_matrix_columns_use_orientation_spacing(published_viewer, vertical, stacked):
    server, _ = _server(root=published_viewer)
    driver = _driver()
    try:
        driver.get(f"http://127.0.0.1:{server.server_port}/?vertical={int(vertical)}")
        _ready(driver, "tab-unified")
        if stacked:
            _choose_engine(driver, 1, "sglang")
            _ready(driver, "tab-unified")
            driver.find_element("css selector", "[data-layout='stacked']").click()
        for width in (2048, 480):
            driver.set_window_size(width, 1000)
            if vertical and stacked:
                WebDriverWait(driver, 10).until(lambda page: page.execute_script("""
                    const table=document.querySelector('.comparison-matrix');
                    return table.querySelector('.matrix-cell').getBoundingClientRect().width
                      - table.querySelector('.matrix-segments').getBoundingClientRect().width
                      <= parseFloat(getComputedStyle(table).fontSize);
                """))
            geometry = driver.execute_script("""
                const table=document.querySelector('.comparison-matrix');
                return {label:table.tHead.rows[0].cells[0].getBoundingClientRect().width,
                  labelTrack:parseFloat(getComputedStyle(table.querySelector('.matrix-label-column')).width),
                  cellTrack:parseFloat(getComputedStyle(table).getPropertyValue('--matrix-cell-width')),
                  tracks:[...table.querySelectorAll('.matrix-result-column')].map(col=>col.getBoundingClientRect().width),
                  cells:[...table.querySelectorAll('.matrix-cell')].map(c=>c.getBoundingClientRect().width),
                  width:table.getBoundingClientRect().width,
                  resultColumns:parseInt(table.style.getPropertyValue('--matrix-result-columns')),
                  categoryButtons:[...table.querySelectorAll('.matrix-category-heading')].map(cell=>{
                    const box=cell.getBoundingClientRect(), button=cell.querySelector('button').getBoundingClientRect();
                    return {left:button.left-box.left,top:button.top-box.top,
                      right:box.right-button.right,bottom:box.bottom-button.bottom};
                  }),
                  header:table.tHead.getBoundingClientRect().height};
            """)
            label_width = geometry['labelTrack'] if vertical else (220 if width > 700 else 150)
            assert abs(geometry["label"] - label_width) <= 1, geometry
            assert geometry['cells'], geometry
            if vertical and stacked:
                assert all(abs(w - geometry['tracks'][index % len(geometry['tracks'])]) <= 1
                           for index, w in enumerate(geometry['cells'])), geometry
            else:
                assert all(abs(w - geometry['cellTrack']) <= 1 for w in geometry['cells']), geometry
                assert abs(geometry['cellTrack'] - (42 if vertical else 30)) <= 1, geometry
            if not vertical:
                assert abs(geometry['width'] - label_width - geometry['cellTrack'] * geometry['resultColumns']) <= 1, geometry
                assert geometry['categoryButtons'], geometry
                assert all(all(abs(offset) <= 1 for offset in button.values())
                           for button in geometry['categoryButtons']), geometry
                assert geometry["header"] <= 100, geometry
            assert not driver.find_elements("css selector", ".matrix-summary")
            assert "Conformance counts" not in driver.find_element("css selector", ".comparison-matrix").text
            for theme in ["light", "dark"]:
                offsets = driver.execute_script("""
                    document.documentElement.dataset.theme=arguments[0];
                    return [...document.querySelectorAll('.comparison-matrix .filtered-cell')].map(cell=>{
                        const range=document.createRange(); range.selectNodeContents(cell);
                        const glyph=range.getBoundingClientRect(), box=cell.getBoundingClientRect(), css=getComputedStyle(cell);
                        const left=box.left+parseFloat(css.borderLeftWidth)+parseFloat(css.paddingLeft);
                        const right=box.right-parseFloat(css.borderRightWidth)-parseFloat(css.paddingRight);
                        const top=box.top+parseFloat(css.borderTopWidth)+parseFloat(css.paddingTop);
                        const bottom=box.bottom-parseFloat(css.borderBottomWidth)-parseFloat(css.paddingBottom);
                        return {x:(glyph.left+glyph.right-left-right)/2,y:(glyph.top+glyph.bottom-top-bottom)/2};
                    });
                """, theme)
                assert offsets and all(abs(offset["x"]) <= 1 and abs(offset["y"]) <= 1 for offset in offsets), offsets[:5]
    finally:
        driver.quit(); server.shutdown(); server.server_close()


@pytest.mark.parametrize("vertical", [False, True])
def test_stacked_families_are_grouped_and_legend_matches_visible_colors(published_viewer, vertical):
    server, _ = _server(root=published_viewer)
    driver = _driver()
    try:
        driver.set_window_size(1600, 1000)
        driver.get(f"http://127.0.0.1:{server.server_port}/?vertical={int(vertical)}&layout_tab-unified=stacked")
        _ready(driver, "tab-unified")
        _choose_engine(driver, 1, "sglang")
        _ready(driver, "tab-unified")
        _choose_engine(driver, 2, "vllm")
        _ready(driver, "tab-unified")

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

            def compact_spacing(browser):
                return browser.execute_script("""
                  const table = document.querySelector('.comparison-matrix');
                  const rows = [...table.tBodies[0].querySelectorAll('tr:not(.matrix-category-row)')];
                  const vertical = document.body.classList.contains('transpose-mode');
                  let cells;
                  if (vertical) {
                    const row = rows.find(row => [...row.querySelectorAll('.matrix-cell')].slice(0,3)
                      .every(cell => cell.querySelector('.matrix-segments')));
                    cells = [...row.querySelectorAll('.matrix-cell')].slice(0,3);
                  } else {
                    const group = rows.slice(0,3);
                    const index = [...group[0].querySelectorAll('.matrix-cell')].findIndex((cell,index) =>
                      group.every(row => row.querySelectorAll('.matrix-cell')[index].querySelector('.matrix-segments')));
                    cells = group.map(row => row.querySelectorAll('.matrix-cell')[index]);
                  }
                  const boxes = cells.map(cell => cell.querySelector('.matrix-segments').getBoundingClientRect());
                  const gaps = boxes.slice(1).map((box,index) => vertical
                    ? box.left-boxes[index].right : box.top-boxes[index].bottom);
                  const limit = parseFloat(getComputedStyle(table).fontSize)/2 + 1;
                  return gaps.every(gap => gap >= 0 && gap <= limit) && {gaps,
                    contained: boxes.every((box,index) => {
                      const cell=cells[index].getBoundingClientRect();
                      return box.left>=cell.left-1 && box.right<=cell.right+1
                        && box.top>=cell.top-1 && box.bottom<=cell.bottom+1;
                    })};
                """)

            spacing = WebDriverWait(driver, 10).until(compact_spacing)
            assert spacing['contained'], spacing

            separator = driver.execute_script("""
              const table=document.querySelector('.comparison-matrix');
              const font=parseFloat(getComputedStyle(table).fontSize);
              if(document.body.classList.contains('transpose-mode')) {
                const cell=[...table.querySelectorAll('td.matrix-cell.matrix-family-end')].find(cell =>
                  cell.querySelector('.matrix-segments') && cell.nextElementSibling?.querySelector('.matrix-segments'));
                const edge=cell.getBoundingClientRect().right, border=parseFloat(getComputedStyle(cell).borderRightWidth);
                return {font,before:edge-border-cell.querySelector('.matrix-segments').getBoundingClientRect().right,
                  after:cell.nextElementSibling.querySelector('.matrix-segments').getBoundingClientRect().left-edge};
              }
              const row=[...table.querySelectorAll('tbody tr.matrix-family-end')].find(row =>
                [...row.querySelectorAll('.matrix-cell')].some((cell,index) => cell.querySelector('.matrix-segments')
                  && row.nextElementSibling.querySelectorAll('.matrix-cell')[index]?.querySelector('.matrix-segments')));
              const cells=[...row.querySelectorAll('.matrix-cell')], next=row.nextElementSibling;
              const index=cells.findIndex((cell,index) => cell.querySelector('.matrix-segments')
                && next.querySelectorAll('.matrix-cell')[index]?.querySelector('.matrix-segments'));
              const nextCell=next.querySelectorAll('.matrix-cell')[index], edge=nextCell.getBoundingClientRect().top;
              return {font,before:edge-cells[index].querySelector('.matrix-segments').getBoundingClientRect().bottom,
                after:nextCell.querySelector('.matrix-segments').getBoundingClientRect().top-edge
                  -parseFloat(getComputedStyle(nextCell).borderTopWidth)};
            """)
            assert separator['before'] >= separator['font'] / 2, separator
            assert separator['after'] >= separator['font'] / 2, separator

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
        _ready(driver, "tab-unified")
        if captures == 2:
            _choose_engine(driver, 1, "sglang")
            _ready(driver, "tab-unified")
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
        _ready(driver, "tab-unified")
        if correction == "layout":
            assert not driver.find_element("css selector", "[data-layout='stacked']").is_displayed()
            before = counts(driver)
            _choose_engine(driver, 1, "sglang")
            _ready(driver, "tab-unified")
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
            _ready(driver, "tab-unified")
            assert not driver.find_element("css selector", "[data-layout='stacked']").is_displayed()
        elif correction == "categories":
            assert not driver.find_elements("css selector", ".category-chips")
            button = driver.find_element("css selector", "[data-comparison-table] [data-group]")
            assert driver.execute_script("return getComputedStyle(arguments[0]).borderTopWidth", button) == "0px"
            key = button.get_attribute("data-group")
            before = counts(driver)
            button.send_keys(Keys.SPACE)
            assert not driver.find_elements("css selector", f"[data-comparison-table] [data-group='{key}']")
            selector = f"[data-category-options] [data-group='{key}']"
            assert driver.find_element("css selector", selector).get_attribute("aria-pressed") == "false"
            reduced = counts(driver)
            assert reduced["dynamo"]["total"] < before["dynamo"]["total"]
            driver.refresh()
            _ready(driver, "tab-unified")
            assert counts(driver) == reduced
            driver.find_element("css selector", ".comparison-categories summary").click()
            driver.find_element("css selector", selector).send_keys(Keys.SPACE)
            assert counts(driver) == before
            driver.find_element("css selector", ".comparison-categories summary").click()
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
                column = driver.execute_script("return window.__conformanceTooltipModel(arguments[0]).popupCase", header)
                marker = '{}' if column.get('schemas') else ''
                assert re.fullmatch(r"\d+-\d+" + re.escape(marker), header.text)
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
        _ready(driver, "tab-unified")
        assert driver.execute_script("return window.firstComparisonPaint") is vertical
        assert len(driver.find_elements("css selector", ".capture-slot")) == 3
        assert len(driver.find_elements("css selector", ".tabs .tab-button")) == 0
        sha = driver.find_element("css selector", ".source-commit")
        assert driver.find_element("css selector", "#viewer > h1").text == "Parser Conformance"
        assert driver.find_elements("css selector", ".generated .source-commit") == [sha]
        assert re.fullmatch(r"[a-f0-9]{12}", sha.text)
        assert sha.get_attribute("title") in sha.get_attribute("href")
        assert sha.get_attribute("title") in sha.get_attribute("aria-label")
        assert not driver.find_element("css selector", "[data-layout='stacked']").is_displayed()
        assert "%" not in driver.find_element("id", "viewer").text
        assert not driver.find_elements("css selector", ".legacy-report-details, [data-peer-only]")
        assert driver.execute_script("return [...document.querySelectorAll('.capture-summary [data-count]')].every(node => Number(node.textContent.trim().split(' ')[0]) > 0)")
        before = counts(driver)
        driver.find_element("css selector", "[data-focus='errors']").click()
        assert counts(driver) == before
        search = driver.find_element("css selector", "[data-case-search]")
        search.send_keys("dotted_name_minimal")
        after = counts(driver)
        assert 0 < after["dynamo"]["total"] < before["dynamo"]["total"]
        assert not driver.find_element("css selector", "[data-focus='errors']").is_enabled()
        assert driver.find_element("css selector", "[data-focus='all']").get_attribute('aria-pressed') == 'true'
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


def test_source_sha_appears_once_on_second_line(published_viewer):
    server, _ = _server(root=published_viewer)
    driver = _driver()
    try:
        driver.get(f"http://127.0.0.1:{server.server_port}/")
        _ready(driver, "tab-unified")
        source = driver.find_element("css selector", ".source-commit").get_attribute("title")
        for generated in (None, source, 'a' * 40 if source != 'a' * 40 else 'b' * 40):
            if generated:
                driver.execute_script("""
                    window.ConformanceView.updateGeneration(document.getElementById('viewer'), {
                      revision: arguments[0], generated_revision: arguments[1],
                      generated_on: '2026-10-30 12:00 PDT', command: 'conformance-publish', output: 'site'
                    });
                """, source, generated)
            heading = driver.find_element("css selector", "#viewer > h1")
            line = driver.find_element("css selector", "#viewer > .generated")
            assert heading.text == 'Parser Conformance'
            assert re.findall(r'\b[a-f0-9]{12}\b', line.text) == [source[:12]]
            assert line.rect['y'] >= heading.rect['y'] + heading.rect['height']
            assert len(driver.find_elements("css selector", ".source-commit")) == 1
            link = line.find_element("css selector", ".source-commit")
            assert link.get_attribute('href').endswith(source)
            assert source in link.get_attribute('aria-label')
            if generated:
                assert generated in line.get_attribute('title')
            else:
                assert line.text == 'Source commit ' + source[:12]
    finally:
        driver.quit(); server.shutdown(); server.server_close()


def test_errors_and_all_green_results_follow_focus(published_viewer):
    server, _ = _server(root=published_viewer)
    driver = _driver()
    try:
        driver.get(f"http://127.0.0.1:{server.server_port}/?tab=tab-unified")
        _ready(driver, "tab-unified")
        passed = driver.find_element('css selector', '.engine-status-bar > .status-pass')
        assert float(driver.execute_script('return getComputedStyle(arguments[0]).opacity', passed)) == 1
        _choose_engine(driver, 1, 'sglang')
        _ready(driver, 'tab-unified')
        mixed = driver.find_element('css selector', '.matrix-segments:has(> .status-pass):has(> :not(.status-pass)) > .status-pass')
        assert float(driver.execute_script('return getComputedStyle(arguments[0]).opacity', mixed)) == 1
        family = driver.execute_script("""
            const keys=window.__conformanceController.selection(document.getElementById('tab-unified')).selected;
            const leaves=c=>c?.variants ? c.variants.flatMap(leaves) : c ? [c] : [];
            return window.__conformanceModel.rows.find(row=>{
                const cells=Object.values(row.cells).flatMap(leaves).filter(c=>c.applicable ?? c.status!=='na');
                return cells.length && cells.every(c=>c.cmp?.dynamo && !c.cmp.dynamo.na && !c.cmp.dynamo.err && !c.cmp.dynamo.leak
                    && JSON.stringify(c.cmp.dynamo.sig)===JSON.stringify(c.cmp.golden?.sig))
                    && cells.some(c=>{const peer=c.cmp?.[keys[1]];return peer && !peer.na && JSON.stringify(peer.sig)!==JSON.stringify(c.cmp.dynamo.sig)});
            })?.family;
        """)
        assert family
        field = driver.find_element('css selector', '[data-case-search]')
        field.send_keys('model:' + family)
        driver.find_element('css selector', '[data-layout=stacked]').click()
        driver.find_element('css selector', '[data-focus=difference]').click()
        assert driver.find_element('css selector', '[data-focus=difference]').get_attribute('aria-pressed') == 'true'
        passed = driver.find_element('css selector', '.engine-status-bar > .status-pass:only-child')
        assert float(driver.execute_script('return getComputedStyle(arguments[0]).opacity', passed)) == .2
        green = driver.find_element('css selector', '.matrix-status.status-pass')
        assert float(driver.execute_script('return getComputedStyle(arguments[0]).opacity', green)) == .2
        driver.find_element('css selector', '[data-focus=all]').click()
        passed = driver.find_element('css selector', '.engine-status-bar > .status-pass:only-child')
        assert float(driver.execute_script('return getComputedStyle(arguments[0]).opacity', passed)) == 1
        green = driver.find_element('css selector', '.matrix-status.status-pass')
        assert float(driver.execute_script('return getComputedStyle(arguments[0]).opacity', green)) == 1
        _clear_slot(driver, 1)
        _ready(driver, 'tab-unified')
        field = driver.find_element('css selector', '[data-case-search]')
        field.send_keys(Keys.CONTROL, 'a', Keys.NULL)
        field.send_keys('model:gpt_oss AND "gpt_oss_native_builtin_python_raw"')
        golden = driver.find_element('css selector', '[data-focus=disagreements]')
        errors = driver.find_element('css selector', '[data-focus=errors]')
        assert golden.is_enabled()
        assert errors.text == 'Errors'
        assert not errors.is_enabled()
        assert float(driver.execute_script('return getComputedStyle(arguments[0]).opacity', errors)) < 1
        field.send_keys(Keys.CONTROL, 'a', Keys.NULL)
        field.send_keys('model:beam AND "tool_only"')
        assert counts(driver)['dynamo']['pass'] == counts(driver)['dynamo']['total'] > 0
        passed = driver.find_element('css selector', '.engine-status-bar > .status-pass:only-child')
        assert float(driver.execute_script('return getComputedStyle(arguments[0]).opacity', passed)) == 1
        green = driver.find_element('css selector', '.matrix-segments:not(:has(> :not(.status-pass))) > .status-pass')
        assert float(driver.execute_script('return getComputedStyle(arguments[0]).opacity', green)) == 1
        assert not driver.find_element('css selector', '[data-focus=errors]').is_enabled()
        assert not driver.find_element('css selector', '[data-focus=disagreements]').is_enabled()
        assert not driver.find_elements('css selector', '[data-focus=problems]')
    finally:
        driver.quit(); server.shutdown(); server.server_close()


@pytest.mark.parametrize('vertical', [False, True])
def test_stacked_dynamo_labels_include_generation(published_viewer, vertical):
    implementations = ['dynamo_v2', 'dynamo_v1', 'dynamo_v1_jail']
    versions = {impl: _published_capture(published_viewer, impl)['version'] for impl in implementations}
    query = urlencode({'tab': 'tab-unified', 'vertical': int(vertical), 'layout_tab-unified': 'stacked',
                       'search_tab-unified': 'model:gpt_oss AND "tool_only"',
                       'slots_tab-unified': ','.join(impl + '@' + versions[impl] for impl in implementations)})
    expected = ['Dynamo v2 ' + versions['dynamo_v2'], 'Dynamo v1 ' + versions['dynamo_v1'],
                'Dynamo v1 ' + versions['dynamo_v1_jail']]
    server, _ = _server(root=published_viewer)
    driver = _driver()
    try:
        driver.get(f'http://127.0.0.1:{server.server_port}/?{query}')
        _ready(driver, 'tab-unified')
        selector = '.matrix-capture-heading' if vertical else '.comparison-matrix tbody tr:not(.matrix-category-row) > th:first-child'
        labels = [node.get_attribute('textContent').strip().rsplit(' / ', 1)[-1]
                  for node in driver.find_elements('css selector', selector)]
        assert sorted(labels) == sorted(expected)
    finally:
        driver.quit(); server.shutdown(); server.server_close()


@pytest.mark.parametrize("vertical", [False, True])
@pytest.mark.parametrize("stacked", [False, True])
def test_focus_filters_axes_and_restores_shared_urls(published_viewer, vertical, stacked):
    server, _ = _server(root=published_viewer)
    driver = _driver()
    try:
        driver.get(f"http://127.0.0.1:{server.server_port}/?vertical={int(vertical)}&layout_tab-unified={'stacked' if stacked else 'split'}")
        _ready(driver, "tab-unified")
        assert not driver.find_element('css selector', '[data-focus=difference]').is_displayed()
        _choose_engine(driver, 1, "sglang")
        _ready(driver, "tab-unified")
        before = counts(driver)
        inventory = driver.execute_script("""
            return [...document.querySelectorAll('.matrix-cell[role=button]')].map(cell => ({
                label: cell.getAttribute('aria-label').split('; ').slice(0, 2),
                statuses: [...cell.querySelectorAll('.matrix-segments > i')].map(node =>
                    [...node.classList].find(c => c.startsWith('status-')).slice(7))
            }));
        """)
        assert inventory
        all_button = driver.find_element('css selector', "[data-focus='all']")
        assert all_button.tag_name == 'button'
        assert all_button.get_attribute('aria-pressed') == 'true'
        assert len(driver.find_elements('css selector', '[data-focus]')) == 4
        assert driver.execute_script("return [...document.querySelectorAll('[data-focus][aria-pressed=true]')].length") == 1
        all_button.click()
        assert all_button.get_attribute('aria-pressed') == 'true'
        for focus, statuses in [('disagreements', ['diverge']), ('errors', ['error', 'leak'])]:
            expected = [cell for cell in inventory if any(status in statuses for status in cell['statuses'])]
            button = driver.find_element("css selector", f"[data-focus='{focus}']")
            assert button.text == ('Different than GOLDEN' if focus == 'disagreements' else 'Errors')
            assert button.is_enabled() == bool(expected)
            if not expected:
                continue
            button.click()
            assert driver.find_element('css selector', "[data-focus='all']").get_attribute('aria-pressed') == 'false'
            assert button.get_attribute('aria-pressed') == 'true'
            assert driver.execute_script("return [...document.querySelectorAll('[data-focus][aria-pressed=true]')].length") == 1
            assert counts(driver) == before
            labels = driver.execute_script("""
                return [...document.querySelectorAll('.matrix-cell[role=button]')]
                    .map(cell => cell.getAttribute('aria-label').split('; ').slice(0, 2));
            """)
            assert {pair[0] for pair in labels} == {cell['label'][0] for cell in expected}
            assert {pair[1] for pair in labels} == {cell['label'][1] for cell in expected}
            assert len(driver.find_elements('css selector', '.matrix-cell[role=button]')) < len(inventory)
            if stacked:
                # Every retained capture/model axis has at least one matching result.
                assert driver.execute_script("""
                    const allowed = arguments[0], vertical = arguments[1];
                    const rows = [...document.querySelectorAll('.comparison-matrix tbody tr:not(.matrix-category-row)')];
                    const cells = rows.map(row => [...row.querySelectorAll('.matrix-cell')]);
                    const matches = cell => [...cell.querySelectorAll('.matrix-segments > i')].some(node =>
                        allowed.some(status => node.classList.contains('status-' + status)));
                    return vertical ? cells[0].every((_,i) => cells.some(row => matches(row[i])))
                        : cells.every(row => row.some(matches));
                """, statuses, vertical)
            if focus == 'errors':
                driver.get(driver.current_url.replace('focus_tab-unified=errors', 'focus_tab-unified=problems'))
                _ready(driver, 'tab-unified')
            driver.refresh()
            _ready(driver, 'tab-unified')
            assert driver.find_element('css selector', f"[data-focus='{focus}']").get_attribute('aria-pressed') == 'true'
            assert driver.find_element('css selector', "[data-focus='all']").get_attribute('aria-pressed') == 'false'
            assert counts(driver) == before
            driver.find_element('css selector', f"[data-focus='{focus}']").click()
            assert driver.find_element('css selector', "[data-focus='all']").get_attribute('aria-pressed') == 'true'
            driver.find_element('css selector', '[data-focus=all]').click()
            assert driver.find_element('css selector', "[data-focus='all']").get_attribute('aria-pressed') == 'true'
            assert len(driver.find_elements('css selector', '.matrix-cell[role=button]')) == len(inventory)
        for parser_count in [2, 3]:
            if parser_count == 3:
                _choose_engine(driver, 2, "vllm")
                _ready(driver, "tab-unified")
            difference = driver.find_element('css selector', '[data-focus=difference]')
            assert difference.is_displayed()
            expected = driver.execute_script("""
                const keys = window.__conformanceController.selection(document.getElementById('tab-unified')).selected.filter(k => k !== 'golden');
                const leaves = c => c?.variants ? c.variants.flatMap(leaves) : c ? [c] : [];
                const differs = c => {
                    const entries = keys.map(k => c.cmp?.[k]).filter(e => e && e.na !== 1 && e.sig != null);
                    return entries.length > 1 && entries.some(e => JSON.stringify(e.sig) !== JSON.stringify(entries[0].sig));
                };
                const model = window.__conformanceModel;
                return model.rows.flatMap(row => model.columns.filter(col => leaves(row.cells[col.sub]).some(differs))
                    .map(col => ({family: row.family, sub: col.sub})));
            """)
            assert expected
            difference.click()
            assert difference.get_attribute('aria-pressed') == 'true'
            visible = driver.execute_script("return {families:window.__conformanceComparison.view.rows.map(r=>r.family),cases:window.__conformanceComparison.view.columns.map(c=>c.sub)}")
            assert set(visible['families']) == {entry['family'] for entry in expected}
            assert set(visible['cases']) == {entry['sub'] for entry in expected}
            assert driver.execute_script("""
                const selected = window.__conformanceController.selection(document.getElementById('tab-unified')).selected;
                const painted = new Set([...document.querySelectorAll('.matrix-status')].map(n=>n.dataset.capture));
                return selected.length === arguments[0] && selected.every(key=>painted.has(key));
            """, parser_count)
            geometry = driver.execute_script("return ['.layout-control','.focus-control','.comparison-filter-row'].map(s=>{const r=document.querySelector(s).getBoundingClientRect();return {top:r.top,bottom:r.bottom}})")
            assert geometry[0]['bottom'] <= geometry[1]['top']
            assert geometry[1]['bottom'] <= geometry[2]['top']
            driver.refresh()
            _ready(driver, 'tab-unified')
            assert driver.find_element('css selector', '[data-focus=difference]').get_attribute('aria-pressed') == 'true'
            driver.find_element('css selector', '[data-focus=difference]').click()
            assert driver.find_element('css selector', '[data-focus=all]').get_attribute('aria-pressed') == 'true'
        driver.find_element('css selector', '[data-focus=difference]').click()
        _clear_slot(driver, 2)
        _clear_slot(driver, 1)
        _ready(driver, 'tab-unified')
        assert not driver.find_element('css selector', '[data-focus=difference]').is_displayed()
        assert driver.find_element('css selector', '[data-focus=all]').get_attribute('aria-pressed') == 'true'
        _choose_engine(driver, 1, 'sglang')
        _ready(driver, 'tab-unified')
        driver.find_element('css selector', '[data-focus=errors]').click()
        field = driver.find_element('css selector', '[data-case-search]')
        field.send_keys('__no_matching_case__')
        assert driver.find_element('css selector', '[data-focus=all]').get_attribute('aria-pressed') == 'true'
        assert not driver.find_element('css selector', '[data-focus=errors]').is_enabled()
        assert not driver.find_element('css selector', '[data-focus=disagreements]').is_enabled()
        assert not driver.find_elements('css selector', '.matrix-category-heading, .matrix-case-heading')
        assert 'focus_tab-unified' not in parse_qs(urlsplit(driver.current_url).query)
    finally:
        driver.quit(); server.shutdown(); server.server_close()


@pytest.mark.parametrize('vertical', [False, True])
@pytest.mark.parametrize('stacked', [False, True])
def test_explicit_schema_markers_follow_case_descriptions(published_viewer, vertical, stacked):
    server, _ = _server(root=published_viewer)
    driver = _driver()
    try:
        driver.set_window_size(1600, 1000)
        driver.get(f"http://127.0.0.1:{server.server_port}/?vertical={int(vertical)}&view=details")
        _ready(driver, 'tab-unified')
        if stacked:
            _choose_engine(driver, 1, 'sglang')
            _ready(driver, 'tab-unified')
            driver.find_element('css selector', '[data-layout="stacked"]').click()
        result = driver.execute_script("""
            const model = window.__conformanceModel, view = window.__conformanceComparison.view;
            const check = (node, column) => {
                const markers = node.querySelectorAll('.case-schema-indicator');
                return markers.length === (column.schemas?.length ? 1 : 0)
                    && [...markers].every(marker => marker.textContent === '{}'
                        && marker.getAttribute('aria-label') === 'Has explicit schema.'
                        && marker.getBoundingClientRect().width > 0);
            };
            const headings = [...document.querySelectorAll('.matrix-case-heading')];
            const descriptions = [...document.querySelectorAll('.case-descriptions .gloss-desc')];
            const columns = model.glossary.flatMap(group => group.rows).map(pair => model.columns.find(column => column.label === pair[0]));
            return {withSchema: view.columns.some(column => column.schemas?.length),
                withoutSchema: view.columns.some(column => !column.schemas?.length),
                headings: headings.length === view.columns.length && headings.every((node, i) => check(node, view.columns[i])),
                descriptions: descriptions.length === columns.length && descriptions.every((node, i) => check(node, columns[i]))};
        """)
        assert all(result.values()), result
    finally:
        driver.quit(); server.shutdown(); server.server_close()


@pytest.mark.parametrize('width', [480, 1600])
def test_complete_case_names_wrap_in_label_column(published_viewer, width):
    server, _ = _server(root=published_viewer)
    driver = _driver()
    try:
        driver.set_window_size(width, 1000)
        driver.get(f"http://127.0.0.1:{server.server_port}/?vertical=1")
        _ready(driver, 'tab-unified')
        for font_size in (10, 11, 12):
            before = driver.execute_script("""
                const table = document.querySelector('.comparison-matrix');
                return {fontSize: parseFloat(getComputedStyle(table).fontSize),
                    width: table.style.getPropertyValue('--matrix-label-width')};
            """)
            driver.execute_script("document.querySelector('.comparison-matrix').style.fontSize=arguments[0]+'px'", font_size)
            WebDriverWait(driver, 10).until(lambda browser: browser.execute_script("""
                const width = document.querySelector('.comparison-matrix').style.getPropertyValue('--matrix-label-width');
                return Boolean(width) && (arguments[0].fontSize === arguments[1] || width !== arguments[0].width);
            """, before, font_size))
            geometry = driver.execute_script("""
                return [...document.querySelectorAll('.matrix-case-heading')].map(th => {
                    const table = th.closest('table'), text = th.querySelector('.matrix-case-text'), rect = text.getBoundingClientRect();
                    const range = document.createRange(); range.selectNodeContents(text);
                    const boxes = [...range.getClientRects()];
                    const lineTops = [];
                    for (const box of boxes) if (!lineTops.some(top => Math.abs(top - box.top) < 4)) lineTops.push(box.top);
                    return {label: text.textContent, sticky: getComputedStyle(th).position,
                        width: th.getBoundingClientRect().width,
                        labelWidth: parseFloat(getComputedStyle(table.querySelector('.matrix-label-column')).width),
                        lineCount: lineTops.length,
                        lineLefts: lineTops.map(top => Math.min(...boxes.filter(box => Math.abs(box.top - top) < 4).map(box => box.left))),
                        indent: parseFloat(getComputedStyle(text).fontSize),
                        clipped: text.scrollWidth > text.clientWidth || text.scrollHeight > text.clientHeight,
                        contained: boxes.every(box => box.left >= rect.left - 1 && box.right <= rect.right + 1
                            && box.top >= rect.top - 1 && box.bottom <= rect.bottom + 1)};
                });
            """)
            assert geometry
            assert any('7-tool-argument-text-preservation' in item['label'] for item in geometry)
            assert any('dotted_name_minimal' in item['label'] for item in geometry)
            expected_position = 'sticky' if width > 1280 else 'static'
            assert all(item['sticky'] == expected_position and abs(item['width'] - item['labelWidth']) <= 1 for item in geometry), geometry
            assert all(not item['clipped'] and item['contained'] for item in geometry), geometry
            assert any(item['lineCount'] == 2 for item in geometry), (font_size, geometry)
            assert any(item['lineLefts'][0] + item['indent'] - 2 <= min(item['lineLefts'][1:]) for item in geometry if len(item['lineLefts']) > 1), (font_size, geometry)
            assert all(item['lineCount'] <= 2 for item in geometry), [(item['label'], item['lineCount'], item['width']) for item in geometry if item['lineCount'] > 2]
    finally:
        driver.quit(); server.shutdown(); server.server_close()


def test_capture_cards_and_menu_share_one_row_and_notice_stays_at_bottom(published_viewer):
    server, _ = _server(root=published_viewer)
    driver = _driver()
    try:
        driver.set_window_size(1600, 1000)
        driver.get(f"http://127.0.0.1:{server.server_port}/?tab=tab-unified")
        _ready(driver, "tab-unified")
        _choose_engine(driver, 1, "sglang")
        _ready(driver, "tab-unified")
        _choose_engine(driver, 2, "vllm")
        _ready(driver, "tab-unified")
        notice = WebDriverWait(driver, 10).until(
            lambda d: d.find_element("css selector", ".publication-history-notice"))
        before = counts(driver)
        for width in (2048, 1600, 1024, 480, 390):
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
                assert geometry["optionsParent"] == "comparison-filter-row", geometry
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
        _ready(driver,"tab-unified")
        cell = driver.find_element("css selector", ".comparison-matrix .matrix-cell[data-ttip-id]")
        cell.send_keys(Keys.ENTER)
        popup = WebDriverWait(driver,10).until(lambda d: d.find_element("css selector", ".comparison-output.ttip-pinned"))
        assert driver.execute_script("const r=arguments[0].getBoundingClientRect(); return Math.abs(r.left+r.width/2-document.documentElement.clientWidth/2)<=1 && Math.abs(r.top+r.height/2-document.documentElement.clientHeight/2)<=1", popup)
        assert popup.find_elements("css selector", ".golden-out-cap, [data-cand=golden]")
        assert parse_qs(urlsplit(driver.current_url).query)["popup"]
        driver.refresh(); _ready(driver,"tab-unified")
        assert len(driver.find_elements("css selector", ".comparison-output.ttip-pinned")) == 1
        driver.find_element("tag name","body").send_keys(Keys.ESCAPE)
        assert not driver.find_elements("css selector", ".comparison-output.ttip-pinned")
        assert "popup" not in parse_qs(urlsplit(driver.current_url).query)
    finally:
        driver.quit(); server.shutdown(); server.server_close()


@pytest.mark.parametrize("vertical", [False, True])
@pytest.mark.parametrize("width", [2048, 480])
def test_popup_centers_in_viewport_without_closed_schema_overflow(published_viewer, vertical, width):
    server, _ = _server(root=published_viewer)
    driver = _driver()
    try:
        driver.set_window_size(width, 1194)
        driver.get(f"http://127.0.0.1:{server.server_port}/?vertical={int(vertical)}")
        _ready(driver, "tab-unified")
        baseline = driver.execute_script("return document.documentElement.scrollWidth")
        header = driver.execute_script("return [...document.querySelectorAll('.matrix-case-heading')].find(h=>h.getAttribute('aria-label')?.includes('arg_object_quoted_parameter_close'))")
        header.click()
        driver.find_element("tag name", "body").send_keys(Keys.ESCAPE)
        assert driver.execute_script("return document.documentElement.scrollWidth") == baseline
        cell = driver.execute_script("return [...document.querySelectorAll('.matrix-cell[data-ttip-id]')].find(c=>c.getAttribute('aria-label').startsWith('qwen3; 6-empty_args;'))")
        cell.click()
        popup = driver.find_element("css selector", ".comparison-output.ttip-pinned")

        def centered():
            return driver.execute_script("""
                const r=arguments[0].getBoundingClientRect(),width=document.documentElement.clientWidth,height=document.documentElement.clientHeight;
                return Math.abs(r.left+r.width/2-width/2)<=1 && Math.abs(r.top+r.height/2-height/2)<=1
                  && r.left>=31 && r.right<=width-31 && r.top>=31 && r.bottom<=height-31;
            """, popup)

        assert centered()
        assert driver.execute_script("return document.documentElement.scrollWidth") == baseline
        driver.execute_script("window.scrollBy(0,200)")
        WebDriverWait(driver, 5).until(lambda _: centered())
        driver.set_window_size(width, 800)
        WebDriverWait(driver, 5).until(lambda _: centered())
        driver.find_element("tag name", "body").send_keys(Keys.ESCAPE)
        assert driver.execute_script("return document.documentElement.scrollWidth") == baseline
    finally:
        driver.quit(); server.shutdown(); server.server_close()


def test_reference_url_same_engine_and_stale_request(historical_published_viewer):
    started, release = threading.Event(), threading.Event()
    server, _ = _server(root=historical_published_viewer, gate=(started,release),
                        gate_path=_published_capture_index(historical_published_viewer,"dynamo_v1","5.1.3"))
    driver = _driver()
    try:
        driver.get(f"http://127.0.0.1:{server.server_port}/?tab=tab-unified&slots_tab-unified=dynamo%2Cdynamo%400.7.18%2C&cmp_tab-unified=dynamo%2Cdynamo%400.7.18")
        _ready(driver,"tab-unified")
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
