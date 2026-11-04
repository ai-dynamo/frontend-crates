// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { escapeHtml, isApplicableCell, selectedCaptureOutcome, captureStatusLabels as statusLabels } from './model.js';
import { captureSegmentHtml } from './presentation.js';
import { compileSearch, searchContext } from './search.js';
export const verticalDefault = params => !['0', 'false'].includes(params.get('vertical') ?? params.get('transpose'));
const html = value => escapeHtml(value ?? '');
const leavesOf = cell => cell?.variants ? cell.variants.flatMap(leavesOf) : cell ? [cell] : [];
const familyOf = row => row.family || row.model_label || '';
const problemOutcomes = new Set(['documented', 'diverge', 'error', 'leak']);
const fittedCaseLabelColumns = new WeakMap();

export const isProblemOutcome = outcome => problemOutcomes.has(outcome);

export function comparisonView(tab, state, matches = compileSearch(state.search, searchContext(tab))) {
  const groupLabels = new Map(tab.column_groups.map(group => [group.key, group.label]));
  const entries = [], visibleRows = new Set(), visibleColumns = new Set();
  const seen = new Set();
  for (const row of tab.rows) {
    for (const column of tab.columns) {
      if (!state.groups.has(column.group_key)) continue;
      const matching = leavesOf(row.cells[column.sub]).filter(leaf => matches(
        [groupLabels.get(column.group_key), column.label, column.case_id, column.sub, leaf.sub, leaf.case_id],
        [row.family, row.model_label], [column.desc, leaf.tooltip?.description]));
      if (matching.length) { visibleRows.add(row); visibleColumns.add(column); }
      const selected = matching.filter(isApplicableCell);
      if (selected.length) entries.push({row, column, leaves: selected, countedLeaves: selected.filter(leaf => {
        const key = JSON.stringify([row.family, leaf.sub || leaf.case_id]);
        if (seen.has(key)) return false;
        seen.add(key); return true;
      }), cell: row.cells[column.sub]});
    }
  }
  return {entries, rows: tab.rows.filter(row => visibleRows.has(row)),
    columns: tab.columns.filter(column => visibleColumns.has(column))};
}

// Apply focus after search/categories so availability and card totals share the base view.
export function focusComparison(view, keys, requested) {
  const matches = (entry, key, focus) => entry.leaves.some(leaf => {
    const outcome = selectedCaptureOutcome(leaf, key);
    return focus === 'disagreements' ? outcome === 'diverge' : isProblemOutcome(outcome);
  });
  const available = Object.fromEntries(['disagreements', 'problems'].map(focus =>
    [focus, view.entries.some(entry => keys.some(key => matches(entry, key, focus)))]));
  const focus = requested === 'all' || available[requested] ? requested : 'all';
  const matching = focus === 'all' ? view.entries : view.entries.filter(entry => keys.some(key => matches(entry, key, focus)));
  const rows = focus === 'all' ? view.rows : view.rows.filter(row => matching.some(entry => entry.row === row));
  const columns = focus === 'all' ? view.columns : view.columns.filter(column => matching.some(entry => entry.column === column));
  const rowKeys = new Map(rows.map(row => [row, focus === 'all' ? keys : keys.filter(key =>
    matching.some(entry => entry.row === row && matches(entry, key, focus)))]));
  return {focus, available, rowKeys, view: {rows, columns,
    entries: view.entries.filter(entry => rows.includes(entry.row) && columns.includes(entry.column))}};
}

export function captureCounts(view, key) {
  const counts = Object.fromEntries(Object.keys(statusLabels).map(status => [status, 0]));
  for (const entry of view.entries) for (const leaf of entry.countedLeaves || entry.leaves) counts[selectedCaptureOutcome(leaf, key)]++;
  counts.total = Object.values(counts).reduce((a, b) => a + b, 0);
  return counts;
}

function horizontalCellExtra(cell) {
  const style = getComputedStyle(cell);
  return ['paddingLeft', 'paddingRight', 'borderLeftWidth', 'borderRightWidth']
    .reduce((sum, property) => sum + parseFloat(style[property]), 0);
}

function fitStackedResultColumns(table) {
  const mark = table.querySelector(':scope > tbody > tr > td.matrix-cell > .matrix-segments');
  if (!mark) return;
  const cells = [...mark.closest('tr').cells].filter(cell => cell.classList.contains('matrix-cell'));
  const tracks = [...table.querySelectorAll(':scope > colgroup > .matrix-result-column')];
  const markWidth = mark.getBoundingClientRect().width;
  let total = 0;
  tracks.forEach((track, index) => {
    const width = markWidth + horizontalCellExtra(cells[index]);
    track.style.width = width + 'px'; total += width;
  });
  table.style.setProperty('--matrix-result-width', total + 'px');
}

function fitVerticalCaseLabels(table) {
  const labels = [...table.querySelectorAll('tbody tr > th:first-child .matrix-case-text')];
  if (!labels.length) return;
  const cellExtra = horizontalCellExtra(labels[0].closest('th'));
  const signature = cellExtra + labels.map(label => {
    const style = getComputedStyle(label);
    return `${label.textContent}\u0000${style.font}\u0000${style.letterSpacing}\u0000${style.lineHeight}`;
  }).join('\u0001');
  const cached = fittedCaseLabelColumns.get(table);
  if (cached?.signature === signature) {
    table.style.setProperty('--matrix-label-width', cached.width + 'px');
    return;
  }
  const measurements = labels.map(label => {
    const style = getComputedStyle(label), clone = label.cloneNode(true);
    Object.assign(clone.style, {position: 'fixed', left: '-100000px', top: '0', visibility: 'hidden',
      display: 'block', width: 'max-content', whiteSpace: 'nowrap', padding: '0', textIndent: '0',
      overflow: 'visible', overflowWrap: 'normal', font: style.font, letterSpacing: style.letterSpacing,
      lineHeight: style.lineHeight});
    document.body.appendChild(clone);
    const rect = clone.getBoundingClientRect();
    clone.remove();
    return {width: rect.width, lineHeight: rect.height};
  });
  const fitsInTwoLines = width => {
    table.style.setProperty('--matrix-label-width', width + 'px');
    return labels.every((label, index) =>
      measurements[index].lineHeight > 0
        && Math.round(label.getBoundingClientRect().height / measurements[index].lineHeight) <= 2);
  };
  let high = Math.max(...measurements.map(item => item.width)) + cellExtra;
  for (let attempt = 0; attempt < 4 && !fitsInTwoLines(high); attempt++) high *= 2;
  let low = 0;
  for (let attempt = 0; attempt < 32 && high - low > 1 / window.devicePixelRatio; attempt++) {
    const middle = (low + high) / 2;
    if (fitsInTwoLines(middle)) high = middle;
    else low = middle;
  }
  table.style.setProperty('--matrix-label-width', high + 'px');
  fittedCaseLabelColumns.set(table, {signature, width: high});
}

function countMarkup(counts) {
  const present = Object.entries(statusLabels).filter(([s]) => counts[s] > 0);
  return '<div class="engine-status-bar" role="img" aria-label="' + html(present.map(([s,l]) => `${l}: ${counts[s]}`).join(', ')) + '">'
    + present.map(([s]) => '<i class="status-' + s + '" style="flex-grow:' + counts[s] + '"></i>').join('') + '</div>'
    + '<div class="engine-counts">' + present.map(([s,l]) => '<span data-count="' + s + '"><i class="legend-swatch status-' + s + '"></i>' + counts[s] + ' ' + html(l) + '</span>').join('') + '</div>';
}

export function mountComparison(panel, controller, familyNames = () => []) {
  const params = new URLSearchParams(location.search);
  const param = name => name + '_' + panel.id;
  const state = {layout: params.get(param('layout')) === 'stacked' ? 'stacked' : 'split',
    focus: ['disagreements','problems'].includes(params.get(param('focus'))) ? params.get(param('focus')) : 'all',
    search: params.get(param('search')) || '', groups: null};
  const ui = document.createElement('div'); ui.className = 'comparison-ui';
  ui.innerHTML = '<div class="comparison-toolbar"><div class="layout-control"><span class="control-label">Layout</span><div class="segmented">'
    + '<button type="button" data-layout="split">Split cells</button><button type="button" data-layout="stacked">Stacked columns</button></div></div>'
    + '<div class="focus-control"><span class="control-label">Focus</span><div class="segmented" role="group" aria-label="Focus filter">'
    + '<button type="button" data-focus="all">All</button><button type="button" data-focus="disagreements">Disagreements</button>'
    + '<button type="button" data-focus="problems">Problems</button></div></div>'
    + '<details class="comparison-categories"><summary>Categories</summary><div data-category-options></div></details>'
    + '<input type="search" data-case-search placeholder="Filter models or cases..." aria-label="Filter models or cases"'
    + ' title=\'Spaces require all words. Use OR, quotes, -exclusions and parentheses: beam ("guided decoding" OR TC). Scopes: parser:, col:, description:. Case numbers expand to stable names in shared URLs.\''
    + ' aria-describedby="' + panel.id + '-search-error"><span class="comparison-search-error" id="' + panel.id + '-search-error" role="status" hidden></span></div>'
    + '<p class="comparison-notice" role="status" hidden></p><div class="comparison-matrix-scroll"></div>'
    + '<div class="comparison-legend" aria-label="Outcome colors"></div>';
  panel.appendChild(ui);
  const toolbar = ui.querySelector('.comparison-toolbar');
  const viewActions = panel.querySelector('.capture-slots .tabs-right');
  if (viewActions) {
    viewActions.classList.remove('capture-slot-actions');
    toolbar.appendChild(viewActions);
  }
  panel.querySelector('.capture-slots').appendChild(toolbar);
  toolbar.querySelector('[data-case-search]').value = state.search;
  let search, context, matches = compileSearch('');
  let tab;
  let observedMatrix;
  const labelResizeObserver = new ResizeObserver(() => {
    if (observedMatrix?.isConnected && ui.classList.contains('layout-vertical')) {
      if (ui.classList.contains('layout-stacked')) fitStackedResultColumns(observedMatrix);
      fitVerticalCaseLabels(observedMatrix);
    }
  });
  const persist = () => {
    const url = new URL(location.href);
    for (const [key,value,defaultValue] of [['layout',state.layout,'split'],['focus',state.focus,'all'],['search',matches.canonicalQuery,''],['groups',[...state.groups].join(','),tab.column_groups.map(g => g.key).join(',')]]) {
      if (value === defaultValue) url.searchParams.delete(param(key)); else url.searchParams.set(param(key),value);
    }
    history.replaceState(null, '', url);
  };
  function draw() {
    if (!tab) return;
    if (search !== state.search) {
      search = state.search;
      const error = toolbar.querySelector('.comparison-search-error');
      let invalid = false;
      try { matches = compileSearch(search, context); error.textContent = matches.notices.join(' '); error.hidden = !matches.notices.length; }
      catch (failure) { invalid = true; error.textContent = failure.message + ' Showing the last valid filter.'; error.hidden = false; }
      toolbar.querySelector('[data-case-search]').setAttribute('aria-invalid', String(invalid));
    }
    const scroll = ui.querySelector('.comparison-matrix-scroll');
    const pin = controller.disposePanel(ui);
    const vertical = document.body.classList.contains('transpose-mode');
    ui.classList.toggle('layout-vertical', vertical);
    toolbar.querySelector('[data-layout="stacked"]').textContent = vertical ? 'Stacked columns' : 'Stacked rows';
    toolbar.querySelectorAll('[data-layout]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.layout === state.layout)));
    let view;
    const notice = ui.querySelector('.comparison-notice');
    try { view = comparisonView(tab,state,matches); notice.hidden = true; }
    catch (error) { notice.textContent = error.message; notice.hidden = false; view = {entries: [],rows: [],columns: []}; }
    const keys = panel._comparisonSelection.keys;
    const baseView = view;
    const focused = focusComparison(baseView, keys, state.focus);
    if (state.focus !== focused.focus) { state.focus = focused.focus; persist(); }
    view = focused.view;
    toolbar.querySelectorAll('[data-focus]').forEach(button => {
      button.disabled = button.dataset.focus !== 'all' && !focused.available[button.dataset.focus];
      button.setAttribute('aria-pressed', String(button.dataset.focus === state.focus));
    });
    const rowKeys = row => focused.rowKeys.get(row);
    toolbar.querySelector('.layout-control').hidden = keys.length < 2;
    const stacked = keys.length > 1 && state.layout === 'stacked';
    ui.classList.toggle('layout-stacked', stacked);
    const familyGroups = [];
    for (const row of view.rows) {
      const key = familyOf(row), previous = familyGroups[familyGroups.length - 1];
      if (previous?.key === key) previous.rows.push(row);
      else familyGroups.push({key, rows: [row]});
    }
    const entriesByRow = new Map(view.rows.map(row => [row,new Map(view.entries.filter(e => e.row === row).map(e => [e.column,e]))]));
    panel.querySelectorAll('.capture-slot').forEach((card, i) => {
      const key = card.querySelector('[data-slot-capture]')?.value;
      let summary = card.querySelector('.capture-summary');
      if (!summary) { summary = document.createElement('div'); summary.className = 'capture-summary'; card.appendChild(summary); }
      const selected = keys.includes(key);
      summary.dataset.capture = selected ? key : '';
      summary.innerHTML = selected ? '<div class="summary-capture-label">Results: ' + html(tab.candidates.find(c => c.key === key)?.label || key) + '</div>' + countMarkup(captureCounts(baseView,key)) : '<p>' + (key ? 'Selected capture loading' : 'No comparison selected') + '</p>';
    });
    const table = document.createElement('table'); table.className = 'comparison-matrix'; table.dataset.comparisonTable = '';
    if (vertical) table.dataset.transposeTable = '';
    const thead = table.createTHead(), tbody = table.createTBody();
    const entryFor = (row,column) => entriesByRow.get(row)?.get(column);
    const candidate = key => tab.candidates.find(c => c.key === key);
    const captureLabel = key => {
      const item = candidate(key);
      return [item?.impl, item?.version].filter(Boolean).join(' ') || key;
    };
    const outcomes = (entry,key) => entry ? selectedCaptureOutcome({...entry.cell, variants: entry.leaves},key) : 'unsupported';
    const heading = (tr,label,span=1) => { const th = document.createElement('th'); th.textContent = label; th.colSpan = span; tr.appendChild(th); return th; };
    const rotatedHeading = (tr,label,span=1) => {
      const th = heading(tr,'',span);
      th.innerHTML = '<span class="tcol-model-name">' + html(label) + '</span>';
      return th;
    };
    const groupHeading = (tr,group,span=1) => {
      const th = heading(tr,'',span);
      th.className = 'matrix-category-heading';
      th.innerHTML = '<button type="button" data-group="' + html(group.key) + '" aria-pressed="' + state.groups.has(group.key)
        + '" title="' + html(group.label) + '" aria-label="Include ' + html(group.label) + '">' + html(group.label) + '</button>';
    };
    const caseHeading = (tr,column) => {
      const th = heading(tr, column.label); th.className = 'matrix-case-heading';
      const compact = column.display_number == null ? html(column.label) : window.__compactCaseLabelHtml(column);
      th.innerHTML = '<span class="matrix-case-label" data-hover-hint="Click to show ' + html(column.label) + '"><span class="matrix-case-text">'
        + (vertical ? html(column.label) + (column.display_number == null ? '' : ' (' + compact + ')') : compact)
        + window.__conformanceSchemaIndicatorHtml(column)
        + '</span></span><div class="ttip"></div>';
      th.setAttribute('aria-label', column.case_id || column.label);
      th.dataset.ttipCenter = '1';
      window.__registerConformanceTooltip(th,window.__conformanceCaseTooltip(tab,column),null,JSON.stringify(['grammar',tab.id,column.case_id || column.sub]));
      window.__attachConformanceTooltip(th);
    };
    const result = (tr,row,column,shown=keys) => {
      const entry = entryFor(row,column), td = tr.insertCell(); td.className = 'matrix-cell';
      if (!entry) { td.classList.add('filtered-cell'); td.textContent = '—'; td.setAttribute('aria-label', isApplicableCell(row.cells[column.sub]) ? 'Excluded by filters' : 'Not applicable to this family'); return td; }
      td.innerHTML = '<span class="matrix-segments">' + shown.map(k => captureSegmentHtml(outcomes(entry,k),k)).join('') + '</span><div class="ttip comparison-output"></div>';
      td.setAttribute('role','button');
      td.setAttribute('aria-label',[row.model_label || row.family,column.label,...shown.map(k => (candidate(k)?.label || k) + ': ' + statusLabels[outcomes(entry,k)])].join('; '));
      if (entry?.cell.tooltip) {
        const cell = entry.cell;
        window.__registerConformanceTooltip(td,cell.tooltip,cell.fixture_href,JSON.stringify(['cell',tab.id,cell.case_id || column.sub,cell.family || '']),cell);
        window.__attachConformanceTooltip(td);
      }
      return td;
    };
    if (vertical) {
      const top = thead.insertRow(); heading(top,'Case / model family');
      const sub = stacked ? thead.insertRow() : null;
      if (sub) heading(sub,'');
      if (stacked) {
        for (const [groupIndex,group] of familyGroups.entries()) {
          const th = rotatedHeading(top,group.rows[0].model_label || group.key,group.rows.reduce((sum,row) => sum + rowKeys(row).length, 0));
          th.className = 'matrix-family-heading';
          th.dataset.family = group.key;
          if (groupIndex < familyGroups.length - 1) th.classList.add('matrix-family-end');
          for (const row of group.rows) for (const [index,k] of rowKeys(row).entries()) {
            const item = candidate(k);
            const capture = rotatedHeading(sub,captureLabel(k));
            capture.className = 'matrix-capture-heading';
            capture.title = item?.label || k;
            capture.setAttribute('aria-label', item?.label || k);
            if (row === group.rows[group.rows.length - 1] && index === rowKeys(row).length - 1 && groupIndex < familyGroups.length - 1) capture.classList.add('matrix-family-end');
          }
        }
      } else for (const row of view.rows) {
        rotatedHeading(top,row.model_label || row.family);
      }
      for (const group of tab.column_groups.filter(group => view.columns.some(column => column.group_key === group.key))) {
        const category = tbody.insertRow(); category.className = 'matrix-category-row';
        groupHeading(category,group);
        if (stacked) {
          for (const [index,family] of familyGroups.entries()) {
            const cell = category.insertCell();
            cell.colSpan = family.rows.reduce((sum,row) => sum + rowKeys(row).length, 0);
            cell.dataset.family = family.key;
            if (index < familyGroups.length - 1) cell.classList.add('matrix-family-end');
          }
        } else {
          const width = view.rows.length;
          if (width) category.insertCell().colSpan = width;
        }
        for (const column of view.columns.filter(c => c.group_key === group.key)) {
          const tr = tbody.insertRow(); caseHeading(tr,column);
          if (stacked) for (const [groupIndex,family] of familyGroups.entries()) for (const row of family.rows) {
            for (const [index,k] of rowKeys(row).entries()) {
              const cell = result(tr,row,column,[k]);
              if (row === family.rows[family.rows.length - 1] && index === rowKeys(row).length - 1 && groupIndex < familyGroups.length - 1) cell.classList.add('matrix-family-end');
            }
          }
          else for (const row of view.rows) result(tr,row,column);
        }
      }
    } else {
      const groups = thead.insertRow(); heading(groups,'Model family');
      const columns = tab.column_groups.map(group => ({group, columns: view.columns.filter(c => c.group_key === group.key)})).filter(item => item.columns.length);
      for (const {group,columns: cases} of columns) groupHeading(groups,group,cases.length);
      const sub = thead.insertRow(); heading(sub,'');
      for (const {columns: cases} of columns) for (const c of cases) caseHeading(sub,c);
      const results = (tr,row,shown=keys) => {
        for (const {columns: cases} of columns) for (const c of cases) result(tr,row,c,shown);
      };
      if (stacked) for (const [groupIndex,family] of familyGroups.entries()) for (const row of family.rows) for (const [index,k] of rowKeys(row).entries()) {
        const tr = tbody.insertRow(); tr.dataset.family = family.key;
        if (groupIndex > 0 && row === family.rows[0] && index === 0) tr.classList.add('matrix-family-start');
        if (groupIndex < familyGroups.length - 1 && row === family.rows.at(-1) && index === rowKeys(row).length - 1) tr.classList.add('matrix-family-end');
        const label = (row.model_label || row.family) + ' / ';
        const th = heading(tr,label + captureLabel(k));
        th.title = label + candidate(k)?.label; th.setAttribute('aria-label', th.title);
        results(tr,row,[k]);
      }
      else for (const row of view.rows) {
        const tr = tbody.insertRow(); heading(tr,row.model_label || row.family); results(tr,row);
      }
    }
    const visibleStatuses = new Set();
    for (const entry of view.entries) for (const key of (stacked ? rowKeys(entry.row) : keys)) visibleStatuses.add(outcomes(entry,key));
    const legend = ui.querySelector('.comparison-legend');
    legend.replaceChildren(...Object.entries(statusLabels).filter(([status]) => visibleStatuses.has(status)).map(([status,label]) => {
      const item = document.createElement('span'); item.dataset.status = status;
      const swatch = document.createElement('i'); swatch.className = 'legend-swatch status-' + status;
      item.append(swatch,label); return item;
    }));
    // Explicit tracks keep long headings and a wide viewport from stretching result cells.
    const resultColumns = vertical ? view.rows.reduce((sum,row) => sum + (stacked ? rowKeys(row).length : 1), 0) : thead.rows[1].cells.length - 1;
    table.style.setProperty('--matrix-result-columns', resultColumns);
    const tracks = document.createElement('colgroup');
    const labelTrack = document.createElement('col'); labelTrack.className = 'matrix-label-column';
    tracks.appendChild(labelTrack);
    if (resultColumns) {
      if (vertical && stacked) for (let index = 0; index < resultColumns; index++) {
        const track = document.createElement('col'); track.className = 'matrix-result-column'; tracks.appendChild(track);
      }
      else {
        const resultTracks = document.createElement('col'); resultTracks.className = 'matrix-result-column'; resultTracks.span = resultColumns;
        tracks.appendChild(resultTracks);
      }
    }
    table.insertBefore(tracks,thead);
    if (observedMatrix) labelResizeObserver.unobserve(observedMatrix);
    observedMatrix = table;
    scroll.replaceChildren(table);
    labelResizeObserver.observe(table);
    controller.restorePopup(panel,pin);
    window.__conformanceComparison = {state,view,availability: focused.available,search: {ast: matches.ast, canonicalQuery: matches.canonicalQuery, context},counts: Object.fromEntries(keys.map(k => [k,captureCounts(baseView,k)]))};
  }
  toolbar.addEventListener('click', event => {
    const group = event.target.closest('[data-group]')?.dataset.group;
    if (group) { toggleGroup(group); return; }
    const button = event.target.closest('[data-layout],[data-focus]'); if (!button || button.disabled) return;
    if (button.dataset.layout) state.layout = button.dataset.layout;
    else if (button.dataset.focus === 'all') state.focus = 'all';
    else state.focus = state.focus === button.dataset.focus ? 'all' : button.dataset.focus;
    draw(); persist();
  });
  toolbar.addEventListener('input', event => {
    if (event.target.matches('[data-case-search]')) state.search = event.target.value;
    else return;
    draw(); persist();
    if (searchField.selectionStart === state.search.length && searchField.selectionEnd === state.search.length
      && /\s$|\)$/.test(state.search)) expandSearch();
  });
  const searchField = toolbar.querySelector('[data-case-search]');
  function expandSearch() {
    if (searchField.getAttribute('aria-invalid') === 'true') return;
    if (state.search === matches.canonicalQuery) return;
    state.search = matches.canonicalQuery; searchField.value = state.search;
    draw(); persist();
  }
  searchField.addEventListener('blur', expandSearch);
  searchField.addEventListener('keydown', event => { if (event.key === 'Enter') expandSearch(); });
  function toggleGroup(group) {
    if (!state.groups.delete(group)) state.groups.add(group);
    draw(); persist();
    toolbar.querySelectorAll('[data-group]').forEach(button => button.setAttribute('aria-pressed', String(state.groups.has(button.dataset.group))));
  }
  ui.querySelector('.comparison-matrix-scroll').addEventListener('click', event => {
    const group = event.target.closest('[data-group]')?.dataset.group; if (!group) return;
    toggleGroup(group);
    [...ui.querySelectorAll('[data-group]')].find(button => button.dataset.group === group)?.focus({preventScroll: true});
  });
  new MutationObserver(() => {
    const vertical = document.body.classList.contains('transpose-mode');
    if (vertical !== ui.classList.contains('layout-vertical')) draw();
  }).observe(document.body,{attributes: true,attributeFilter: ['class']});
  return {update(model) {
    tab = model;
    context = searchContext(tab, familyNames()); search = undefined;
    if (!state.groups) state.groups = new Set(params.has(param('groups')) ? (params.get(param('groups')) || '').split(',') : tab.column_groups.map(g => g.key));
    const categories = toolbar.querySelector('[data-category-options]');
    categories.innerHTML = tab.column_groups.map(group => '<button type="button" data-group="' + html(group.key)
      + '" aria-pressed="' + state.groups.has(group.key) + '">' + html(group.label) + '</button>').join('');
    panel.append(ui);
    const descriptions = panel.querySelector(':scope > .case-descriptions');
    if (descriptions) panel.append(descriptions);
    for (const child of [...panel.children]) if (!child.matches('.cmpctl,.capture-slots,.comparison-ui,.case-descriptions,.load-status')) child.remove();
    draw(); expandSearch();
  }};
}
