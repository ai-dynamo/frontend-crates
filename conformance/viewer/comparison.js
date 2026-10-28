// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { isApplicableCell, selectedCaptureOutcome, captureStatusLabels as statusLabels } from './model.js';

export { captureStatusLabels as statusLabels } from './model.js';
export const verticalDefault = params => !['0', 'false'].includes(params.get('vertical') ?? params.get('transpose'));
const text = value => String(value ?? '').toLowerCase();
const html = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[c]));
const leavesOf = cell => cell?.variants ? cell.variants.flatMap(leavesOf) : cell ? [cell] : [];

export function comparisonView(tab, state) {
  const index = tab.peer_availability;
  if (state.peerOnly && (index?.schema !== 1 || !index.families)) throw new Error('Regenerate this publication to index peer availability.');
  const query = text(state.search).trim();
  const entries = [];
  const seen = new Set();
  for (const row of tab.rows) {
    const modelMatch = !query || text([row.family, row.model_label].join(' ')).includes(query);
    for (const column of tab.columns) {
      if (!state.groups.has(column.group_key)) continue;
      const columnMatch = text([column.label, column.case_id, column.desc].join(' ')).includes(query);
      const selected = leavesOf(row.cells[column.sub]).filter(leaf => isApplicableCell(leaf)
        && (modelMatch || columnMatch || text([leaf.sub, leaf.case_id, leaf.tooltip?.description].join(' ')).includes(query))
        && (!state.peerOnly || index.families[row.family]?.includes(leaf.sub)));
      if (selected.length) entries.push({row, column, leaves: selected, countedLeaves: selected.filter(leaf => {
        const key = JSON.stringify([row.family, leaf.sub || leaf.case_id]);
        if (seen.has(key)) return false;
        seen.add(key); return true;
      }), cell: row.cells[column.sub]});
    }
  }
  return {entries, rows: tab.rows.filter(row => entries.some(e => e.row === row)),
    columns: tab.columns.filter(column => entries.some(e => e.column === column))};
}

export function captureCounts(view, key) {
  const counts = Object.fromEntries(Object.keys(statusLabels).map(status => [status, 0]));
  for (const entry of view.entries) for (const leaf of entry.countedLeaves || entry.leaves) counts[selectedCaptureOutcome(leaf, key)]++;
  counts.total = Object.values(counts).reduce((a, b) => a + b, 0);
  return counts;
}

function countMarkup(counts) {
  return '<div class="engine-status-bar" role="img" aria-label="' + html(Object.entries(statusLabels).map(([s,l]) => `${l}: ${counts[s]}`).join(', ')) + '">'
    + Object.keys(statusLabels).filter(s => counts[s]).map(s => '<i class="status-' + s + '" style="flex-grow:' + counts[s] + '"></i>').join('') + '</div>'
    + '<div class="engine-counts">' + Object.entries(statusLabels).map(([s,l]) => '<span data-count="' + s + '"><i class="legend-swatch status-' + s + '"></i>' + counts[s] + ' ' + html(l) + '</span>').join('') + '</div>';
}

export function mountComparison(panel, controller) {
  const params = new URLSearchParams(location.search);
  const param = name => name + '_' + panel.id;
  const state = {layout: params.get(param('layout')) === 'stacked' ? 'stacked' : 'split',
    focus: ['disagreements','problems'].includes(params.get(param('focus'))) ? params.get(param('focus')) : 'all',
    search: params.get(param('search')) || '', peerOnly: params.get(param('peer')) === '1',
    groups: null};
  const ui = document.createElement('div'); ui.className = 'comparison-ui';
  ui.innerHTML = '<div class="comparison-toolbar"><span class="control-label">Layout</span><div class="segmented">'
    + '<button type="button" data-layout="split">Split cells</button><button type="button" data-layout="stacked">Stacked columns</button></div>'
    + '<span class="control-label">Focus</span><div class="segmented"><button type="button" data-focus="all">All</button>'
    + '<button type="button" data-focus="disagreements">Disagreements</button><button type="button" data-focus="problems">Problems</button></div>'
    + '<label class="peer-filter"><input type="checkbox" data-peer-only> Only cases with peer data</label>'
    + '<input type="search" data-case-search placeholder="Filter models or cases..." aria-label="Filter models or cases"></div>'
    + '<p class="comparison-notice" role="status" hidden></p><div class="comparison-legend">'
    + Object.entries(statusLabels).map(([s,l]) => '<span><i class="legend-swatch status-' + s + '"></i>' + l + '</span>').join('')
    + '</div><div class="category-chips" aria-label="Case categories"></div><div class="comparison-matrix-scroll"></div>';
  panel.appendChild(ui);
  const toolbar = ui.querySelector('.comparison-toolbar');
  toolbar.querySelector('[data-peer-only]').checked = state.peerOnly;
  toolbar.querySelector('[data-case-search]').value = state.search;
  const details = document.createElement('details'); details.className = 'legacy-report-details';
  details.innerHTML = '<summary>Detailed case reference table</summary>';
  details.open = document.body.classList.contains('view-details'); panel.appendChild(details);
  let tab;
  const persist = () => {
    const url = new URL(location.href);
    for (const [key,value,defaultValue] of [['layout',state.layout,'split'],['focus',state.focus,'all'],['search',state.search,''],['peer',state.peerOnly ? '1' : '0','0'],['groups',[...state.groups].join(','),tab.column_groups.map(g => g.key).join(',')]]) {
      if (value === defaultValue) url.searchParams.delete(param(key)); else url.searchParams.set(param(key),value);
    }
    history.replaceState(null, '', url);
  };
  function draw() {
    if (!tab) return;
    const pin = controller.disposePanel(ui);
    const vertical = document.body.classList.contains('transpose-mode');
    ui.classList.toggle('layout-vertical', vertical);
    toolbar.querySelector('[data-layout="stacked"]').textContent = vertical ? 'Stacked columns' : 'Stacked rows';
    toolbar.querySelectorAll('[data-layout]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.layout === state.layout)));
    toolbar.querySelectorAll('[data-focus]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.focus === state.focus)));
    let view;
    const notice = ui.querySelector('.comparison-notice');
    try { view = comparisonView(tab,state); notice.hidden = true; }
    catch (error) { notice.textContent = error.message; notice.hidden = false; view = {entries: [],rows: [],columns: []}; }
    ui.querySelector('.category-chips').innerHTML = tab.column_groups.map(g => '<button type="button" data-group="' + html(g.key) + '" aria-pressed="' + state.groups.has(g.key) + '">' + html(g.label) + '</button>').join('');
    const keys = panel._comparisonSelection.keys;
    const entriesByRow = new Map(view.rows.map(row => [row,new Map(view.entries.filter(e => e.row === row).map(e => [e.column,e]))]));
    panel.querySelectorAll('.capture-slot').forEach((card, i) => {
      const key = card.querySelector('[data-slot-capture]')?.value;
      let summary = card.querySelector('.capture-summary');
      if (!summary) { summary = document.createElement('div'); summary.className = 'capture-summary'; card.appendChild(summary); }
      const selected = keys.includes(key);
      summary.dataset.capture = selected ? key : '';
      summary.innerHTML = selected ? '<div class="summary-capture-label">Results: ' + html(tab.candidates.find(c => c.key === key)?.label || key) + '</div>' + countMarkup(captureCounts(view,key)) : '<p>' + (key ? 'Selected capture loading' : 'No comparison selected') + '</p>';
    });
    const table = document.createElement('table'); table.className = 'comparison-matrix'; table.dataset.comparisonTable = '';
    if (vertical) table.dataset.transposeTable = '';
    const thead = table.createTHead(), tbody = table.createTBody();
    const entryFor = (row,column) => entriesByRow.get(row)?.get(column);
    const candidate = key => tab.candidates.find(c => c.key === key);
    const outcomes = (entry,key) => entry ? selectedCaptureOutcome({...entry.cell, variants: entry.leaves},key) : 'unsupported';
    const countsLabel = (row,key) => {
      const counts = captureCounts({entries: view.entries.filter(e => e.row === row)},key);
      return html(candidate(key)?.label || key) + ': ' + Object.entries(statusLabels).filter(([s]) => counts[s]).map(([s,label]) => counts[s] + ' ' + html(label.toLowerCase())).join(', ') + '; ' + counts.total + ' cases';
    };
    const heading = (tr,label,span=1) => { const th = document.createElement('th'); th.textContent = label; th.colSpan = span; tr.appendChild(th); return th; };
    const caseHeading = (tr,column) => {
      const th = heading(tr, column.label); th.className = 'matrix-case-heading';
      th.innerHTML = html(column.label) + (column.display_number == null ? '' : ' (' + window.__compactCaseLabelHtml(column) + ')') + '<div class="ttip"></div>';
      th.dataset.ttipCenter = '1';
      window.__registerConformanceTooltip(th,window.__conformanceCaseTooltip(tab,column),null,JSON.stringify(['grammar',tab.id,column.case_id || column.sub]));
      window.__attachConformanceTooltip(th);
    };
    const result = (tr,row,column,shown=keys) => {
      const entry = entryFor(row,column), td = tr.insertCell(); td.className = 'matrix-cell';
      if (!entry) { td.classList.add('filtered-cell'); td.textContent = '—'; td.setAttribute('aria-label', 'Excluded by filters'); return; }
      const codes = keys.map(k => outcomes(entry,k));
      const focused = state.focus === 'all' || (state.focus === 'disagreements' ? codes.includes('diverge') : codes.some(s => ['error','leak','expected','missing','blocked','unsupported','uncompared'].includes(s)));
      td.classList.toggle('focus-dim',!focused);
      td.innerHTML = '<span class="matrix-segments">' + shown.map(k => '<i class="status-' + outcomes(entry,k) + '"></i>').join('') + '</span><div class="ttip comparison-output"></div>';
      td.setAttribute('role','button');
      td.setAttribute('aria-label',[row.model_label || row.family,column.label,...shown.map(k => (candidate(k)?.label || k) + ': ' + statusLabels[outcomes(entry,k)])].join('; '));
      if (entry?.cell.tooltip) {
        const cell = entry.cell;
        window.__registerConformanceTooltip(td,cell.tooltip,cell.fixture_href,JSON.stringify(['cell',tab.id,cell.case_id || column.sub,cell.family || '']),cell);
        window.__attachConformanceTooltip(td);
      }
    };
    const summary = (tr,row) => { const td = tr.insertCell(); td.className = 'matrix-summary'; td.innerHTML = keys.map(k => '<span title="' + html(countsLabel(row,k)) + '">' + countsLabel(row,k) + '</span>').join(''); };
    if (vertical) {
      const top = thead.insertRow(), sub = thead.insertRow(); heading(top,'Case / model family'); heading(sub,'Conformance counts');
      for (const row of view.rows) {
        heading(top,row.model_label || row.family,state.layout === 'stacked' ? Math.max(1,keys.length) : 1);
        if (state.layout === 'stacked') for (const k of keys) heading(sub,countsLabel(row,k));
        else heading(sub,keys.map(k => countsLabel(row,k)).join('\n'));
      }
      for (const column of view.columns) {
        const tr = tbody.insertRow(); caseHeading(tr,column);
        for (const row of view.rows) if (state.layout === 'stacked') for (const k of keys) result(tr,row,column,[k]); else result(tr,row,column);
      }
    } else {
      const groups = thead.insertRow(); heading(groups,'Model family');
      for (const g of tab.column_groups) { const n = view.columns.filter(c => c.group_key === g.key).length; if (n) heading(groups,g.label,n); }
      heading(groups,'Conformance counts'); const sub = thead.insertRow(); heading(sub,'Engine'); for (const c of view.columns) caseHeading(sub,c); heading(sub,'Counts');
      for (const row of view.rows) {
        if (state.layout === 'stacked') for (const k of keys) { const tr = tbody.insertRow(); heading(tr,(row.model_label || row.family) + ' / ' + candidate(k)?.label); for (const c of view.columns) result(tr,row,c,[k]); summary(tr,row); }
        else { const tr = tbody.insertRow(); heading(tr,row.model_label || row.family); for (const c of view.columns) result(tr,row,c); summary(tr,row); }
      }
    }
    ui.querySelector('.comparison-matrix-scroll').replaceChildren(table);
    controller.restorePopup(panel,pin);
    window.__conformanceComparison = {state,view,counts: Object.fromEntries(keys.map(k => [k,captureCounts(view,k)]))};
  }
  toolbar.addEventListener('click', event => {
    const button = event.target.closest('[data-layout],[data-focus]'); if (!button) return;
    if (button.dataset.layout) state.layout = button.dataset.layout; else state.focus = button.dataset.focus;
    draw(); persist();
  });
  toolbar.addEventListener('input', event => {
    if (event.target.matches('[data-case-search]')) state.search = event.target.value;
    else if (event.target.matches('[data-peer-only]')) state.peerOnly = event.target.checked;
    else return;
    draw(); persist();
  });
  ui.querySelector('.category-chips').addEventListener('click', event => {
    const group = event.target.closest('[data-group]')?.dataset.group; if (!group) return;
    if (!state.groups.delete(group)) state.groups.add(group); draw(); persist();
  });
  let detailed = document.body.classList.contains('view-details');
  new MutationObserver(() => {
    const vertical = document.body.classList.contains('transpose-mode');
    if (vertical !== ui.classList.contains('layout-vertical')) draw();
    const requestedDetails = document.body.classList.contains('view-details');
    if (requestedDetails !== detailed) { detailed = requestedDetails; details.open = detailed; }
  }).observe(document.body,{attributes: true,attributeFilter: ['class']});
  return {update(model) {
    tab = model;
    if (!state.groups) state.groups = new Set(params.has(param('groups')) ? (params.get(param('groups')) || '').split(',') : tab.column_groups.map(g => g.key));
    panel.append(ui,details);
    for (const child of [...details.children]) if (child.tagName !== 'SUMMARY') child.remove();
    for (const child of [...panel.children]) if (!child.matches('.cmpctl,.capture-slots,.comparison-ui,.legacy-report-details')) details.appendChild(child);
    draw();
  }};
}
