// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
(function () {
  'use strict';

  const page = window.__CONFORMANCE_PAGE;
  if (!page || !Array.isArray(page.tabs)) return;

  const engineOrder = ['dynamo', 'sglang', 'vllm'];
  const engineNames = { dynamo: 'Dynamo', sglang: 'SGLang', vllm: 'vLLM' };
  const engineColors = { dynamo: 'var(--engine-dynamo)', sglang: 'var(--engine-sglang)', vllm: 'var(--engine-vllm)' };
  const statusNames = {
    pass: 'Matches reference', documented: 'Documented difference', diverge: 'Disagreement',
    leak: 'Markup leak', error: 'Parser error', expected: 'Expected error',
    unsupported: 'No parser for family', missing: 'No captured result', uncompared: 'Reference unavailable',
  };
  const statusClasses = ['pass', 'documented', 'diverge', 'leak', 'error', 'expected', 'unsupported', 'missing', 'uncompared'];

  function escapeHtml(value) {
    return String(value == null ? '' : value).replace(/[&<>"']/g, ch => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    })[ch]);
  }

  function engineKey(candidate) {
    const impl = String(candidate.impl || candidate.key || '').toLowerCase();
    if (impl.startsWith('dynamo')) return 'dynamo';
    if (impl.startsWith('sglang')) return 'sglang';
    if (impl.startsWith('vllm')) return 'vllm';
    return impl;
  }

  function candidateVersion(candidate) {
    const mode = candidate.parse_mode || '';
    return [candidate.version, mode].filter(Boolean).join(' · ') || candidate.label || candidate.key;
  }

  function candidatesFor(tab) {
    return (tab.candidates || []).filter(candidate => candidate && candidate.key && candidate.impl !== 'golden');
  }

  function selectedKeys(panel) {
    const ctl = panel.querySelector('.cmpctl');
    if (!ctl) return [];
    return Array.from(ctl.querySelectorAll('input.cmp-on:checked')).map(input => input.value);
  }

  function referenceKey(panel) {
    const selected = panel.querySelector('.cmp-ref:checked');
    return selected ? selected.value : null;
  }

  function tooltipCandidate(cell, key) {
    const candidates = (cell && cell.tooltip && cell.tooltip.candidates) || [];
    return candidates.find(candidate => candidate.key === key) || null;
  }

  function sourceMatches(key, impl) {
    return typeof key === 'string' && typeof impl === 'string'
      && (key === impl || key.startsWith(impl + '-'));
  }

  function captureIssue(cell, key) {
    if (!cell || cell.kind === 'blank' || cell.kind === 'missing') return 'missing';
    const entry = (cell.cmp || {})[key];
    const candidateTip = tooltipCandidate(cell, key);
    const block = candidateTip && candidateTip.block;
    if (!entry || entry.na === 1) {
      const reason = String((block && block.unavailable) || '');
      return /\bno(?: [\w.-]+)* parser\b|\bnot (?:yet )?implemented\b/i.test(reason) ? 'unsupported' : 'missing';
    }
    if (entry.err === 1 || (block && (block.exception || (block.error && typeof block.error === 'object')))) return 'error';
    if (block && typeof block.error === 'string') return 'expected';
    if (entry.leak === 1 || (candidateTip && candidateTip.leak)) return 'leak';
    return null;
  }

  function outputStatus(tab, cell, candidate, reference) {
    const issue = captureIssue(cell, candidate.key);
    if (issue) return issue;
    const entry = cell.cmp[candidate.key];
    const oracleTab = tab.id === 'tab-unified';
    const compareKey = oracleTab ? 'golden' : reference;
    const compareEntry = (cell.cmp || {})[compareKey];
    if (['missing', 'unsupported', 'error', 'expected'].includes(captureIssue(cell, compareKey))) return 'uncompared';
    if (entry.sig === compareEntry.sig) return 'pass';
    const intentional = ((cell.tooltip && cell.tooltip.reasons) || []).some(reason => {
      if (reason.intentional !== true) return false;
      const forward = sourceMatches(candidate.key, reason.impl) && sourceMatches(compareKey, reason.reference_impl)
        && entry.sig === reason.candidate_sig && compareEntry.sig === reason.reference_sig;
      const reverse = sourceMatches(compareKey, reason.impl) && sourceMatches(candidate.key, reason.reference_impl)
        && compareEntry.sig === reason.candidate_sig && entry.sig === reason.reference_sig;
      const block = (tooltipCandidate(cell, forward ? candidate.key : compareKey) || {}).block;
      return (forward || reverse) && block && reason.reason === (block.explanation || block.reason);
    });
    if (intentional) return 'documented';
    return 'diverge';
  }

  function eligibleData(cell, candidate) {
    const entry = cell && cell.cmp && cell.cmp[candidate.key];
    return !!(entry && entry.na !== 1);
  }

  function makeHeader() {
    const originalTitle = document.querySelector('h1');
    const generated = document.querySelector('p.generated');
    const meta = page.meta || {};
    const shortSha = meta.short_sha || (meta.sha ? meta.sha.slice(0, 12) : '');
    const commitHref = meta.sha ? 'https://github.com/ai-dynamo/frontend-crates/commit/' + encodeURIComponent(meta.sha) : '';
    const header = document.createElement('header');
    header.className = 'report-header';
    header.innerHTML = '<div><div class="report-crumb"><b>ai-dynamo / frontend-crates</b> · conformance</div>'
      + '<h1>Parser Conformance</h1></div><div class="report-meta">'
      + (shortSha ? '<span class="meta-pill">commit <a href="' + escapeHtml(commitHref) + '"><code>' + escapeHtml(shortSha) + '</code></a></span>' : '')
      + (meta.stamp ? '<span class="meta-pill">generated ' + escapeHtml(meta.stamp) + '</span>' : '')
      + '</div>';
    const themeButton = document.querySelector('[data-theme-toggle]');
    if (themeButton) {
      themeButton.classList.add('header-theme-toggle');
      header.querySelector('.report-meta').appendChild(themeButton);
    }
    if (originalTitle && originalTitle.parentNode) originalTitle.remove();
    if (generated && generated.parentNode) generated.remove();
    document.body.insertBefore(header, document.body.firstChild);
  }

  function statusFor(tab, row, column, candidate, reference) {
    return outputStatus(tab, row.cells && row.cells[column.sub], candidate, reference);
  }

  function peerHasData(tab, row, column) {
    return candidatesFor(tab).some(candidate => engineKey(candidate) !== 'dynamo'
      && eligibleData(row.cells && row.cells[column.sub], candidate));
  }

  function createUi(panel, tab) {
    const goldenOracle = tab.id === 'tab-unified';
    const statusLabel = status => goldenOracle && status === 'pass' ? 'Matches GOLDEN'
      : goldenOracle && status === 'uncompared' ? 'GOLDEN unavailable' : statusNames[status];
    const candidates = candidatesFor(tab);
    const query = new URLSearchParams(window.location.search);
    const capturedKeys = new Set(candidates.filter(candidate => (tab.rows || []).some(row =>
      Object.values(row.cells || {}).some(cell => eligibleData(cell, candidate)))).map(candidate => candidate.key));
    const candidatesByEngine = new Map(engineOrder.map(key => [key, candidates.filter(candidate => engineKey(candidate) === key)]));
    const byKey = new Map(candidates.map(candidate => [candidate.key, candidate]));
    const defaults = engineOrder.map(key => {
      const items = candidatesByEngine.get(key).filter(candidate => capturedKeys.has(candidate.key));
      const preferred = items.find(candidate => candidate.default_bucket === 'A') || items[0];
      return preferred ? preferred.key : null;
    });
    const param = name => name + '_' + tab.id;
    const available = keys => [...new Set(keys.filter(key => byKey.has(key)))];
    const legacyReference = query.has(param('base')) ? query.get(param('base')) : referenceKey(panel);
    const legacyKeys = available([legacyReference, ...(query.get(param('cmp')) || '').split(',')]);
    const hasLegacy = query.has(param('base')) || query.has(param('cmp'));
    const initialKeys = query.has(param('slots')) ? query.get(param('slots')).split(',').map(key => byKey.has(key) ? key : null)
      : hasLegacy ? legacyKeys : defaults;
    const groupKeys = (tab.column_groups || []).map(group => group.key);
    const state = {
      slots: [null, null, null], reference: goldenOracle ? null : legacyReference,
      layout: query.get(param('layout')) === 'stacked' ? 'stacked' : 'split',
      focus: ['disagreements', 'problems'].includes(query.get(param('focus'))) ? query.get(param('focus')) : 'all',
      peerOnly: query.get(param('peer')) === '1', search: query.get(param('search')) || '',
      vertical: ['1', 'true'].includes(query.get('vertical') || query.get('transpose')),
      groups: new Set(query.has(param('groups')) ? query.get(param('groups')).split(',').filter(key => groupKeys.includes(key)) : groupKeys),
      selectedCell: null,
      popup: query.has(param('family')) && query.has(param('case'))
        ? { family: query.get(param('family')), model: query.get(param('model')), sub: query.get(param('case')) } : null,
    };
    const initialSeen = new Set();
    initialKeys.slice(0, 3).forEach((key, index) => {
      if (key && !initialSeen.has(key)) { state.slots[index] = key; initialSeen.add(key); }
    });
    if (!goldenOracle && !state.slots.includes(state.reference)) state.reference = state.slots.find(Boolean) || null;
    const caseTooltips = new Map();
    function activeCandidates() { return state.slots.filter(Boolean).map(key => byKey.get(key)); }
    function engineAvailable(key) { return candidatesByEngine.get(key).some(candidate => capturedKeys.has(candidate.key)); }
    function captureOption(candidate, selectedKey, slot) {
      const missing = !capturedKeys.has(candidate.key);
      const duplicate = state.slots.includes(candidate.key) && state.slots[slot] !== candidate.key;
      return '<option value="' + escapeHtml(candidate.key) + '"' + (candidate.key === selectedKey ? ' selected' : '')
        + (missing || duplicate ? ' disabled' : '') + '>' + escapeHtml(candidateVersion(candidate))
        + (missing ? ' — no captured results for this tab' : duplicate ? ' — selected in another slot' : '') + '</option>';
    }
    function engineButton(candidate, slot, withMark = false) {
      const key = candidate ? engineKey(candidate) : null;
      return '<button type="button" class="engine-picker-trigger" data-slot-picker="' + slot + '" aria-label="Comparison slot ' + (slot + 1)
        + ': ' + escapeHtml(key ? engineNames[key] : 'None') + '">'
        + (withMark ? '<span class="engine-mark">' + escapeHtml(engineNames[key].slice(0, 1)) + '</span>' : '')
        + escapeHtml(key ? engineNames[key] : 'None') + '</button>';
    }
    const ui = document.createElement('div');
    ui.className = 'comparison-ui';
    const engines = document.createElement('div');
    engines.className = 'engine-cards';
    engines.setAttribute('aria-label', 'Engine comparisons');
    const toolbar = document.createElement('div');
    toolbar.className = 'comparison-toolbar';
    const legend = document.createElement('div');
    legend.className = 'comparison-legend';
    legend.innerHTML = statusClasses.map(status => '<span><i class="legend-swatch status-' + status + '"></i>'
      + escapeHtml(statusLabel(status)) + '</span>').join('');
    const categories = document.createElement('div');
    categories.className = 'category-chips';
    categories.setAttribute('aria-label', 'Case categories');
    const matrixCard = document.createElement('div');
    matrixCard.className = 'comparison-matrix-card';
    const scroll = document.createElement('div');
    scroll.className = 'comparison-matrix-scroll';
    const matrix = document.createElement('table');
    matrix.className = 'comparison-matrix';
    matrix.setAttribute('aria-label', 'Parser conformance comparison matrix');
    scroll.appendChild(matrix);
    matrixCard.appendChild(scroll);
    ui.append(engines, toolbar, legend, categories, matrixCard);
    if (!query.has(param('slots')) && legacyKeys.length > 3) {
      const notice = document.createElement('p');
      notice.className = 'comparison-notice';
      notice.textContent = 'This link selected more than three captures. Showing the reference and the first two other captures (three-slot limit).';
      ui.prepend(notice);
    }

    const legacy = document.createElement('details');
    legacy.className = 'legacy-report-details';
    legacy.innerHTML = '<summary>Detailed case reference table</summary>';
    const legacyRequested = query.get('view') === 'details';
    legacy.open = legacyRequested;
    const legacyBody = document.createElement('div');
    legacyBody.className = 'legacy-report-body';
    while (panel.firstChild) legacyBody.appendChild(panel.firstChild);
    legacy.appendChild(legacyBody);
    panel.append(ui, legacy);

    const control = document.createElement('div');
    control.className = 'comparison-controls';
    control.innerHTML = '<span class="control-label">Layout</span><div class="segmented" data-layout-control>'
      + '<button type="button" data-layout="split" aria-pressed="true">Split cells</button>'
      + '<button type="button" data-layout="stacked" aria-pressed="false">Stacked rows</button></div>'
      + '<span class="control-divider"></span><span class="control-label">Focus</span>'
      + '<div class="segmented" data-focus-control><button type="button" data-focus="all" aria-pressed="true">All</button>'
      + '<button type="button" data-focus="disagreements" aria-pressed="false">Disagreements</button>'
      + '<button type="button" data-focus="problems" aria-pressed="false">Problems</button></div>'
      + '<span class="control-divider"></span><label class="peer-filter"><input type="checkbox" data-peer-only> Only cases with peer data</label>'
      + '<label class="model-search"><span aria-hidden="true">⌕</span><input type="search" data-case-search placeholder="Filter models or cases…" aria-label="Filter models or cases"></label>';
    toolbar.appendChild(control);

    const drawer = document.createElement('aside');
    drawer.className = 'comparison-drawer';
    drawer.setAttribute('aria-hidden', 'true');
    drawer.innerHTML = '<div class="drawer-scrim" data-drawer-close></div><section class="drawer-panel" role="dialog" aria-modal="true" aria-labelledby="drawer-title">'
      + '<header class="drawer-header"><div><div class="drawer-case-id" data-drawer-id></div><h2 id="drawer-title" data-drawer-title>Case output</h2><div data-drawer-description></div></div>'
      + '<button type="button" class="drawer-close" data-drawer-close aria-label="Close case output">×</button></header>'
      + '<div class="drawer-content" data-drawer-content></div></section>';
    document.body.appendChild(drawer);

    function renderCards(columns, rows) {
      const reference = state.reference;
      engines.innerHTML = state.slots.map((selectedKey, slot) => {
        const selected = byKey.get(selectedKey);
        const key = selected ? engineKey(selected) : null;
        const items = key ? candidatesByEngine.get(key) : [];
        if (!selected) return '<article class="engine-card empty-slot"><div class="engine-card-head">' + engineButton(null, slot)
          + '</div><p>No comparison selected</p></article>';
        const counts = Object.fromEntries(statusClasses.map(status => [status, 0]));
        const totalized = new Set();
        let captured = 0;
        for (const row of rows) {
          for (const column of columns) {
            const cell = row.cells && row.cells[column.sub];
            const identity = (row.model_label || row.family || '') + '|' + (column.case_id || column.sub);
            if (totalized.has(identity)) continue;
            totalized.add(identity);
            const status = statusFor(tab, row, column, selected, reference);
            counts[status] += 1;
            if (eligibleData(cell, selected)) captured += 1;
          }
        }
        const denominator = captured;
        const totalCases = statusClasses.reduce((total, status) => total + counts[status], 0);
        const success = counts.pass;
        const percent = denominator ? Math.round(success / denominator * 100) : null;
        const bar = statusClasses.map(status => totalCases
          ? '<span class="bar-' + status + '" style="width:' + (counts[status] / totalCases * 100) + '%"></span>' : '').join('');
        const options = items.map(item => captureOption(item, selectedKey, slot)).join('');
        const referenceControl = goldenOracle ? '' : '<div class="engine-card-controls"><label><input type="radio" name="reference-'
          + escapeHtml(tab.id) + '" data-slot-reference="' + slot + '"'
          + (reference === selected.key ? ' checked' : '') + '> Reference</label></div>';
        return '<article class="engine-card" style="--engine-color:' + (engineColors[key] || 'var(--accent)') + '">'
          + '<div class="engine-card-head"><div class="engine-name">'
          + engineButton(selected, slot, true) + '</div><select data-slot-version="' + slot + '" aria-label="Slot ' + (slot + 1) + ' capture version">' + options + '</select></div>'
          + '<div class="engine-score"><strong>' + (percent == null ? '—' : percent + '%') + '</strong><span>'
          + escapeHtml(tab.id === 'tab-unified' ? 'matches GOLDEN' : 'matches reference') + ' · ' + success + ' of ' + captured + ' captured</span></div>'
          + '<div class="engine-status-bar" role="img" aria-label="' + statusClasses.map(status => statusLabel(status) + ': ' + counts[status]).join(', ') + '">' + bar + '</div>'
          + '<div class="engine-counts">' + statusClasses.filter(status => counts[status] > 0).map(status => '<span><i class="legend-swatch status-' + status + '"></i>'
            + counts[status] + ' ' + escapeHtml(statusLabel(status).toLowerCase()) + '</span>').join('') + '</div>'
          + referenceControl + '</article>';
      }).join('');
    }

    function renderCategories(columns, categoryColumns) {
      categories.innerHTML = (tab.column_groups || []).map(group => {
        const count = categoryColumns.filter(column => column.group_key === group.key).length;
        return '<button type="button" class="category-chip" data-group="' + escapeHtml(group.key) + '" aria-pressed="'
          + state.groups.has(group.key) + '"><strong>' + escapeHtml(group.label) + '</strong><span>' + count + '</span></button>';
      }).join('');
    }

    function filteredView() {
      const query = state.search.trim().toLowerCase();
      const allRows = (tab.rows || []).filter(row => !row.section);
      let categoryColumns = tab.columns || [];
      const matchingColumns = query ? categoryColumns.filter(column => (column.label + ' ' + (column.case_id || '') + ' '
        + (column.desc || '')).toLowerCase().includes(query)) : [];
      const matchingRows = query ? allRows.filter(row => (row.model_label + ' ' + row.family + ' '
        + ((row.parser && row.parser.text) || '')).toLowerCase().includes(query)) : allRows;
      const rows = query && matchingColumns.length === 0 ? matchingRows : allRows;
      if (query && matchingColumns.length > 0) categoryColumns = matchingColumns;
      if (state.peerOnly) categoryColumns = categoryColumns.filter(column => allRows.some(row => peerHasData(tab, row, column)));
      const columns = categoryColumns.filter(column => state.groups.has(column.group_key));
      return { columns, categoryColumns, rows };
    }

    function cellCodes(row, column, visibleCandidates, reference) {
      const cell = row.cells && row.cells[column.sub];
      return visibleCandidates.map(candidate => ({
        candidate,
        status: statusFor(tab, row, column, candidate, reference),
        entry: cell && cell.cmp && cell.cmp[candidate.key],
      }));
    }

    function rowSummaries(row, columns, visibleCandidates, reference) {
      return visibleCandidates.map(candidate => {
        let pass = 0;
        let denominator = 0;
        for (const column of columns) {
          const status = statusFor(tab, row, column, candidate, reference);
          if (eligibleData(row.cells && row.cells[column.sub], candidate)) denominator += 1;
          if (status === 'pass') pass += 1;
        }
        const percentage = denominator ? Math.round(pass / denominator * 100) : null;
        const key = engineKey(candidate);
        const oracleName = goldenOracle ? 'GOLDEN' : 'reference';
        return '<div class="model-summary" style="--engine-color:' + (engineColors[key] || 'var(--accent)') + '" title="'
          + escapeHtml(candidate.label + ': ' + pass + ' of ' + denominator + ' captured match ' + oracleName) + '"><b>'
          + escapeHtml((engineNames[key] || candidate.label).slice(0, 1).toUpperCase()) + '</b><span>'
          + (percentage == null ? '—' : percentage + '%') + '</span></div>';

      });
    }

    function comparisonFocus(row, column, visibleCandidates, reference) {
      const codes = cellCodes(row, column, visibleCandidates, reference);
      const cell = row.cells && row.cells[column.sub];
      const entries = codes.filter(code => ['pass', 'diverge', 'uncompared'].includes(code.status)).map(code => code.entry);
      if (tab.id === 'tab-unified' && entries.length) entries.push(cell && cell.cmp && cell.cmp.golden);
      const signatures = entries.filter(entry => entry && entry.na !== 1 && entry.err !== 1).map(entry => entry.sig);
      return {
        disagreement: new Set(signatures).size > 1,
        problem: codes.some(code => ['leak', 'error', 'expected', 'missing', 'unsupported', 'uncompared'].includes(code.status)),
      };
    }

    function statusCell(row, column, visibleCandidates, reference, rowIndex, columnIndex, engineOnly, focus, familyStart = false) {
      const codes = cellCodes(row, column, visibleCandidates, reference);
      const focused = state.focus === 'all' || (state.focus === 'disagreements' ? focus.disagreement : focus.problem);
      const segments = codes.map(code => '<i class="matrix-status status-' + code.status + '"></i>').join('');
      const label = tab.columns[columnIndex];
      return '<td class="matrix-cell' + (focused ? '' : ' focus-dim') + (engineOnly ? ' stacked-cell' : '') + (familyStart ? ' family-start' : '') + '" tabindex="0" role="button"'
        + ' data-disagreement="' + focus.disagreement + '" data-row="' + rowIndex + '" data-column="' + columnIndex + '" aria-label="' + escapeHtml((row.model_label || row.family) + ', ' + label.label + ': '
        + codes.map(code => (code.candidate.label || code.candidate.key) + ' ' + statusNames[code.status]).join('; ')) + '">'
        + '<span class="matrix-segments">' + segments + '</span><div class="ttip"></div></td>';
    }

    function caseHeading(column, columnIndex, rowHeading) {
      const compactReference = tab.id === 'tab-unified' && column.compact_prefix != null && column.display_number != null
        ? window.__compactCaseLabelHtml(column) : '';
      const label = state.vertical
        ? escapeHtml(column.label) + (compactReference ? ' (' + compactReference + ')' : '')
        : compactReference || escapeHtml(column.display_number || column.label);
      return '<th class="matrix-case-heading' + (rowHeading ? ' matrix-model' : '') + '" tabindex="0" role="button" data-case-column="' + columnIndex
        + '" aria-label="' + escapeHtml(column.case_id || column.label) + '">' + label
        + '<div class="ttip"></div></th>';
    }

    function renderMatrix(columns, rows) {
      matrix.querySelectorAll('[data-ttip-wired]').forEach(node => node._ttipDispose());
      const visibleCandidates = activeCandidates();
      const reference = state.reference;
      const allRows = (tab.rows || []).filter(row => !row.section);
      const coordinates = (row, column, projection, stacked, familyStart = false) => statusCell(row, column, projection, reference,
        allRows.indexOf(row), tab.columns.indexOf(column), stacked, comparisonFocus(row, column, visibleCandidates, reference), familyStart);
      const familyName = row => row.model_label || row.family || '';
      const startsFamily = index => index === 0 || familyName(rows[index - 1]) !== familyName(rows[index]);
      const engineLabel = candidate => engineButton(candidate, state.slots.indexOf(candidate.key));
      let header = '';
      let body = '';
      if (state.vertical) {
        const span = state.layout === 'stacked' ? Math.max(1, visibleCandidates.length) : 1;
        header = '<tr class="matrix-group-row"><th class="matrix-corner">Case / model family</th>'
          + rows.map((row, index) => '<th class="' + (startsFamily(index) ? 'family-start' : '') + '" colspan="' + span + '">' + escapeHtml(row.model_label || row.family) + '</th>').join('') + '</tr>';
        header += '<tr class="matrix-case-row"><th class="matrix-corner">Conformance</th>'
          + rows.map((row, rowIndex) => {
            const summaries = rowSummaries(row, columns, visibleCandidates, reference);
            return state.layout === 'stacked' ? visibleCandidates.map((candidate, index) => '<th class="engine-tint-' + engineKey(candidate) + (index === 0 && startsFamily(rowIndex) ? ' family-start' : '') + '">'
              + engineLabel(candidate) + summaries[index] + '</th>').join('') : '<th class="' + (startsFamily(rowIndex) ? 'family-start' : '') + '">' + summaries.join('') + '</th>';
          }).join('') + '</tr>';
        body = columns.map(column => '<tr>' + caseHeading(column, tab.columns.indexOf(column), true)
          + rows.map((row, rowIndex) => state.layout === 'stacked' ? visibleCandidates.map((candidate, index) => coordinates(row, column, [candidate], true, index === 0 && startsFamily(rowIndex)).replace('class="matrix-cell', 'class="engine-tint-' + engineKey(candidate) + ' matrix-cell')).join('')
            : coordinates(row, column, visibleCandidates, false, startsFamily(rowIndex))).join('') + '</tr>').join('');
      } else {
        const groupRuns = [];
        columns.forEach(column => {
          const last = groupRuns[groupRuns.length - 1];
          if (last && last.key === column.group_key) last.span += 1;
          else groupRuns.push({ key: column.group_key, span: 1 });
        });
        header = '<tr class="matrix-group-row"><th class="matrix-corner">Model family</th>'
          + groupRuns.map(run => {
            const label = (tab.column_groups || []).find(group => group.key === run.key)?.label || run.key;
            const labelClass = label.length > 28 ? ' matrix-group-label-long' : '';
            return '<th colspan="' + run.span + '"><span class="matrix-group-label' + labelClass + '">'
              + escapeHtml(label) + '</span></th>';
          }).join('')
          + '<th class="matrix-summary-head">Conformance</th></tr><tr class="matrix-case-row"><th class="matrix-corner">'
          + (state.layout === 'split' ? visibleCandidates.map(engineLabel).join(' · ') : 'Engine') + '</th>'
          + columns.map(column => caseHeading(column, tab.columns.indexOf(column), false)).join('') + '<th class="matrix-summary-head"></th></tr>';
        rows.forEach((row, rowIndex) => {
          const familyStart = startsFamily(rowIndex);
          const summaries = rowSummaries(row, columns, visibleCandidates, reference);
          if (state.layout === 'split') {
            body += '<tr class="matrix-model-row' + (familyStart ? ' family-start' : '') + '"><th scope="row" class="matrix-model"><strong>' + escapeHtml(row.model_label || row.family)
              + '</strong><small>' + escapeHtml(row.parser_label || row.family) + '</small></th>'
              + columns.map(column => coordinates(row, column, visibleCandidates, false)).join('') + '<td class="matrix-summary">' + summaries.join('') + '</td></tr>';
          } else {
            visibleCandidates.forEach((candidate, index) => {
              body += '<tr class="matrix-model-row stacked-row engine-tint-' + engineKey(candidate) + (familyStart && index === 0 ? ' family-start' : '') + '"><th scope="row" class="matrix-model">'
                + (index === 0 ? '<strong>' + escapeHtml(row.model_label || row.family) + '</strong>' : '') + engineLabel(candidate) + '</th>'
                + columns.map(column => coordinates(row, column, [candidate], true)).join('') + '<td class="matrix-summary">' + summaries[index] + '</td></tr>';
            });
          }
        });
      }
      matrix.innerHTML = '<thead>' + header + '</thead><tbody>' + body + '</tbody>';
      matrix.dataset.empty = String(columns.length === 0 || rows.length === 0 || visibleCandidates.length === 0);
      matrix.querySelectorAll('.matrix-cell').forEach(node => {
        node.dataset.ttipPortal = '1';
        const row = allRows[Number(node.dataset.row)];
        const column = tab.columns[Number(node.dataset.column)];
        const tip = row.cells && row.cells[column.sub] && row.cells[column.sub].tooltip;
        if (!tip) return;
        node.dataset.ttipClick = 'drawer';
        window.__registerConformanceTooltip(node, tip, row.cells[column.sub].fixture_href);
        window.__attachConformanceTooltip(node);
      });
      matrix.querySelectorAll('[data-case-column]').forEach(node => {
        node.dataset.ttipPortal = '1';
        node.dataset.ttipClick = 'pin';
        const index = Number(node.dataset.caseColumn);
        if (!caseTooltips.has(index)) caseTooltips.set(index, window.__conformanceCaseTooltip(tab, tab.columns[index]));
        window.__registerConformanceTooltip(node, caseTooltips.get(index));
        window.__attachConformanceTooltip(node);
      });
    }

    function persist() {
      const url = new URL(window.location.href);
      const set = (name, value, isDefault) => {
        url.searchParams.delete(param(name));
        if (!isDefault) url.searchParams.set(param(name), value);
      };
      const defaultReference = goldenOracle ? null : (defaults.find(Boolean) || null);
      const defaultSelection = state.slots.every((key, index) => key === defaults[index]) && state.reference === defaultReference;
      set('slots', state.slots.map(key => key || '').join(','), defaultSelection);
      if (goldenOracle) url.searchParams.delete(param('base'));
      else set('base', state.reference || '', state.reference === defaultReference);
      set('cmp', state.slots.filter(key => key && key !== state.reference).join(','), defaultSelection);
      set('groups', groupKeys.filter(key => state.groups.has(key)).join(','), state.groups.size === groupKeys.length);
      set('layout', state.layout, state.layout === 'split');
      set('focus', state.focus, state.focus === 'all');
      set('search', state.search, !state.search);
      set('peer', '1', !state.peerOnly);
      set('family', state.popup ? state.popup.family : '', !state.popup);
      set('model', state.popup ? state.popup.model : '', !state.popup || !state.popup.model);
      set('case', state.popup ? state.popup.sub : '', !state.popup);
      if (state.popup && panel.classList.contains('active')) url.searchParams.set('tab', tab.id);
      window.history.replaceState(null, '', url.toString());
    }

    function update(writeUrl = true, pickerFocus = null) {
      const focused = document.activeElement;
      const attribute = ui.contains(focused) && ['data-slot-picker', 'data-slot-version', 'data-slot-reference', 'data-group']
        .find(name => focused.hasAttribute(name));
      const focusTarget = pickerFocus !== null ? { attribute: 'data-slot-picker', value: String(pickerFocus) }
        : attribute ? { attribute, value: focused.getAttribute(attribute) } : null;
      const keys = state.slots.filter(Boolean);
      if (goldenOracle) state.reference = null;
      else if (!keys.includes(state.reference)) state.reference = keys[0] || null;
      panel._comparisonSelection.keys = keys;
      panel._comparisonSelection.reference = state.reference;
      window.__applyConformanceSelection(panel, keys, goldenOracle ? referenceKey(panel) : state.reference);
      const { columns, categoryColumns, rows } = filteredView();
      renderCards(columns, rows);
      renderCategories(columns, categoryColumns);
      renderMatrix(columns, rows);
      if (drawer.classList.contains('open')) window.__syncConformanceTooltip(drawer, keys, state.reference, tab.id === 'tab-unified');
      ui.classList.toggle('layout-stacked', state.layout === 'stacked');
      ui.classList.toggle('layout-vertical', state.vertical);
      control.querySelector('[data-layout="stacked"]').textContent = state.vertical ? 'Stacked columns' : 'Stacked rows';
      control.querySelectorAll('[data-layout]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.layout === state.layout)));
      control.querySelectorAll('[data-focus]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.focus === state.focus)));
      control.querySelector('[data-peer-only]').checked = state.peerOnly;
      const search = control.querySelector('[data-case-search]');
      if (search.value !== state.search) search.value = state.search;
      if (writeUrl) persist();
      if (focusTarget) {
        const replacement = Array.from(ui.querySelectorAll('[' + focusTarget.attribute + ']'))
          .find(node => node.getAttribute(focusTarget.attribute) === focusTarget.value);
        if (replacement) replacement.focus({ preventScroll: true });
      }
    }

    function selectSlot(slot, key, restorePicker = false) {
      if (key && (!capturedKeys.has(key) || state.slots.some((other, index) => index !== slot && other === key))) return;
      state.slots[slot] = key;
      update(true, restorePicker ? slot : null);
    }
    panel._comparisonSelection = {
      keys: state.slots.filter(Boolean), reference: state.reference,
      setVertical(enabled) {
        if (state.vertical === enabled) return;
        state.vertical = enabled;
        dismissVisibleTooltips();
        update();
      },
      fromControls(target) {
        const keys = available(selectedKeys(panel));
        if (target.matches('.cmp-ref') && !keys.includes(target.value)) keys.unshift(target.value);
        state.slots = [...keys.slice(0, 3), null, null, null].slice(0, 3);
        state.reference = goldenOracle ? null : referenceKey(panel);
        update();
      },
    };

    const picker = document.createElement('dialog');
    picker.className = 'comparison-picker';
    picker.setAttribute('aria-label', 'Choose comparison engine and capture');
    document.body.appendChild(picker);
    let pickerSlot = 0;
    let pickerTrigger = null;
    let pickerTriggerSurface = '.engine-cards';
    let pickerTriggerIndex = 0;
    function positionPicker() {
      if (!picker.open || !pickerTrigger) return;
      const anchor = pickerTrigger.getBoundingClientRect();
      const popup = picker.getBoundingClientRect();
      const margin = 12;
      const gap = 8;
      let left = anchor.right + gap;
      if (left + popup.width > window.innerWidth - margin) left = anchor.left - popup.width - gap;
      left = Math.max(margin, Math.min(left, window.innerWidth - popup.width - margin));
      const top = Math.max(margin, Math.min(anchor.top + (anchor.height - popup.height) / 2,
        window.innerHeight - popup.height - margin));
      picker.style.left = left + 'px';
      picker.style.top = top + 'px';
    }
    function pickerVersions(engine) {
      const items = candidatesByEngine.get(engine) || [];
      return items.map(candidate => captureOption(candidate, state.slots[pickerSlot], pickerSlot)).join('');
    }
    function syncPickerTrigger() {
      if (!picker.open) return;
      const surface = ui.querySelector(pickerTriggerSurface);
      const matchingTriggers = surface ? surface.querySelectorAll('[data-slot-picker="' + pickerSlot + '"]') : [];
      const replacement = matchingTriggers[Math.min(pickerTriggerIndex, matchingTriggers.length - 1)]
        || engines.querySelector('[data-slot-picker="' + pickerSlot + '"]');
      if (replacement) pickerTrigger = replacement;
    }
    function openPicker(slot, trigger) {
      pickerSlot = slot;
      pickerTrigger = trigger;
      pickerTriggerSurface = trigger.closest('.comparison-matrix') ? '.comparison-matrix' : '.engine-cards';
      const surface = trigger.closest(pickerTriggerSurface);
      pickerTriggerIndex = surface ? Array.from(surface.querySelectorAll('[data-slot-picker="' + slot + '"]')).indexOf(trigger) : 0;
      const selected = byKey.get(state.slots[slot]);
      const engine = selected ? engineKey(selected) : 'none';
      picker.innerHTML = '<form method="dialog"><h2>Comparison slot ' + (slot + 1) + '</h2><label>Engine <select data-picker-engine>'
        + engineOrder.map(key => '<option value="' + key + '"' + (engine === key ? ' selected' : '') + (!engineAvailable(key) ? ' disabled' : '') + '>'
          + engineNames[key] + (!engineAvailable(key) ? ' — no captures for this tab' : '') + '</option>').join('')
        + '<option value="none"' + (engine === 'none' ? ' selected' : '') + '>None</option></select></label>'
        + '<label>Capture <select data-picker-version' + (engine === 'none' ? ' disabled' : '') + '>' + pickerVersions(engine) + '</select></label></form>';
      picker.showModal();
      window.requestAnimationFrame(positionPicker);
    }
    ui.addEventListener('click', event => {
      const trigger = event.target.closest('[data-slot-picker]');
      if (trigger) openPicker(Number(trigger.dataset.slotPicker), trigger);
    });
    window.addEventListener('resize', positionPicker);
    picker.addEventListener('click', event => {
      if (event.target === picker) picker.close();
    });
    picker.addEventListener('close', () => {
      if (pickerTrigger && pickerTrigger.isConnected) pickerTrigger.focus({ preventScroll: true });
    });
    picker.addEventListener('change', event => {
      if (event.target.matches('[data-picker-engine]')) {
        const engine = event.target.value;
        const version = picker.querySelector('[data-picker-version]');
        if (engine === 'none') {
          version.innerHTML = '';
          version.disabled = true;
          selectSlot(pickerSlot, null);
          syncPickerTrigger();
          return;
        }
        const current = byKey.get(state.slots[pickerSlot]);
        const choices = (candidatesByEngine.get(engine) || []).filter(candidate => capturedKeys.has(candidate.key)
          && !state.slots.some((key, index) => index !== pickerSlot && key === candidate.key));
        const selected = choices.find(candidate => current && candidate.key === current.key) || choices[0];
        if (!selected) {
          const restoredEngine = current ? engineKey(current) : 'none';
          event.target.value = restoredEngine;
          version.innerHTML = pickerVersions(restoredEngine);
          version.disabled = restoredEngine === 'none';
          return;
        }
        version.innerHTML = pickerVersions(engine);
        version.value = selected.key;
        version.disabled = false;
        selectSlot(pickerSlot, selected.key);
        syncPickerTrigger();
      } else if (event.target.matches('[data-picker-version]')) {
        const candidate = byKey.get(event.target.value);
        if (!candidate || !capturedKeys.has(candidate.key)
          || state.slots.some((key, index) => index !== pickerSlot && key === candidate.key)) return;
        selectSlot(pickerSlot, candidate.key);
        syncPickerTrigger();
      }
    });
    engines.addEventListener('change', event => {
      if (event.target.matches('[data-slot-version]')) selectSlot(Number(event.target.dataset.slotVersion), event.target.value);
      if (event.target.matches('[data-slot-reference]')) { state.reference = state.slots[Number(event.target.dataset.slotReference)]; update(); }
    });
    control.addEventListener('click', event => {
      const layoutButton = event.target.closest('[data-layout]');
      const focusButton = event.target.closest('[data-focus]');
      if (layoutButton) { state.layout = layoutButton.dataset.layout; update(); }
      if (focusButton) { state.focus = focusButton.dataset.focus; update(); }
    });
    control.addEventListener('change', event => {
      if (event.target.matches('[data-peer-only]')) { state.peerOnly = event.target.checked; update(); }
    });
    control.addEventListener('input', event => {
      if (event.target.matches('[data-case-search]')) { state.search = event.target.value; update(); }
    });
    categories.addEventListener('click', event => {
      const button = event.target.closest('[data-group]');
      if (!button) return;
      const key = button.dataset.group;
      if (state.groups.has(key)) state.groups.delete(key);
      else state.groups.add(key);
      update();
    });

    function closeDrawer(restoreFocus = true) {
      drawer.classList.remove('open');
      drawer.setAttribute('aria-hidden', 'true');
      const coordinates = popupCoordinates();
      const selectedCell = state.selectedCell && state.selectedCell.isConnected ? state.selectedCell
        : coordinates && matrix.querySelector('.matrix-cell[data-row="' + coordinates.row + '"][data-column="' + coordinates.column + '"]');
      state.selectedCell = null;
      state.popup = null;
      persist();
      if (restoreFocus && selectedCell) selectedCell.focus({ preventScroll: true });
    }
    function popupCoordinates() {
      if (!state.popup) return null;
      const matchingRows = (tab.rows || []).filter(item => !item.section).flatMap((item, index) =>
        item.family === state.popup.family && (!state.popup.model || item.model_label === state.popup.model) ? [index] : []);
      const row = matchingRows.length === 1 ? matchingRows[0] : -1;
      const column = (tab.columns || []).findIndex(item => item.sub === state.popup.sub);
      return row >= 0 && column >= 0 ? { row, column } : null;
    }
    function openDrawer(rowIndex, columnIndex, cellNode = null, writeUrl = true) {
      const row = (tab.rows || []).filter(item => !item.section)[rowIndex];
      const column = (tab.columns || [])[columnIndex];
      if (!row || !column) return;
      const cell = row && column && row.cells && row.cells[column.sub];
      state.selectedCell = cellNode;
      state.popup = { family: row.family, model: row.model_label, sub: column.sub };
      drawer.querySelector('[data-drawer-id]').textContent = (cell && cell.case_id) || column.case_id || column.label;
      drawer.querySelector('[data-drawer-title]').textContent = column.desc || column.label;
      drawer.querySelector('[data-drawer-description]').textContent = [row.model_label, row.family].filter(Boolean).join(' · ');
      const content = drawer.querySelector('[data-drawer-content]');
      content.innerHTML = cell && cell.tooltip && window.__renderConformanceTooltip
        ? '<div class="ttip">' + window.__renderConformanceTooltip(cell.tooltip, cell.fixture_href) + '</div>'
        : '<p class="drawer-empty">' + escapeHtml(cell && cell.kind === 'missing' ? 'No fixture was captured for this case.' : 'No output details are available for this case.') + '</p>';
      dismissVisibleTooltips();
      window.__syncConformanceTooltip(drawer, state.slots.filter(Boolean), state.reference, tab.id === 'tab-unified');
      drawer.classList.add('open');
      drawer.setAttribute('aria-hidden', 'false');
      if (writeUrl) persist();
      const closeButton = drawer.querySelector('.drawer-close');
      closeButton.focus();
    }
    matrix.addEventListener('click', event => {
      if (event.target.closest('.ttip')) return;
      const cell = event.target.closest('.matrix-cell');
      if (cell) openDrawer(Number(cell.dataset.row), Number(cell.dataset.column), cell);
    });
    matrix.addEventListener('keydown', event => {
      if (event.target.closest('.ttip')) return;
      const cell = event.target.closest('.matrix-cell,.matrix-case-heading');
      if (cell && (event.key === 'Enter' || event.key === ' ')) { event.preventDefault(); cell.click(); }
    });
    drawer.addEventListener('click', event => { if (event.target.closest('[data-drawer-close]')) closeDrawer(); });
    drawer.addEventListener('keydown', event => {
      if (event.key !== 'Tab' || !drawer.classList.contains('open')) return;
      const focusable = Array.from(drawer.querySelectorAll('button:not([disabled]),a[href],input:not([disabled]),select:not([disabled]),[tabindex="0"]'));
      if (!focusable.length) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
    });
    document.addEventListener('keydown', event => { if (event.key === 'Escape' && drawer.classList.contains('open')) closeDrawer(); });
    document.addEventListener('click', event => {
      if (event.target.closest('[data-tab-target]') && state.popup) closeDrawer(false);
    });

    function dismissVisibleTooltips() {
      document.querySelectorAll('.ttip.ttip-visible').forEach(tip => {
        const owner = tip._ttipOwner;
        if (owner && owner._ttipUnpin) owner._ttipUnpin();
        else tip.classList.remove('ttip-visible');
      });
    }
    let detailed = legacyRequested;
    const transposeObserver = new MutationObserver(() => {
      const requestedDetails = document.body.classList.contains('view-details');
      if (detailed !== requestedDetails) { detailed = requestedDetails; legacy.open = detailed; }
    });
    transposeObserver.observe(document.body, { attributes: true, attributeFilter: ['class'] });
    update(false);
    if (panel.classList.contains('active')) {
      const coordinates = popupCoordinates();
      if (coordinates) openDrawer(coordinates.row, coordinates.column, null, false);
    }

  }

  function install() {
    makeHeader();
    document.querySelectorAll('.tab-panel').forEach(panel => {
      const tab = page.tabs.find(item => item.id === panel.id);
      if (tab) createUi(panel, tab);
    });
    document.documentElement.classList.remove('comparison-pending');
    document.body.classList.remove('cmp-loading');
  }

  try { install(); }
  catch (error) {
    document.documentElement.classList.remove('comparison-pending');
    document.body.classList.remove('cmp-loading');
    console.error('comparison_layout: failed to build comparison UI', error);
  }
})();
