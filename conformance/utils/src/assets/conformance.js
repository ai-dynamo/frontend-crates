// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Maintained conformance matrix controller for asynchronously loaded tab models.
window.initializeConformanceController = function (hooks = {}) {
  // Gutter kept between a popup and the viewport edge (2em at the 16px root size).
  // The tooltip is width:max-content and the widest ones run to the full cap below, so
  // without this they sit flush against the screen edge and read as clipped.
  const margin = 32;
  let placeScheduled = false;
  const visibleTooltips = new Set();
  const showDelayMs = 750;
  const columnShowDelayMs = 1000;
  const hideDelayMs = 750;
  const columnHideDelayMs = 180;
  let columnButtons = Array.from(document.querySelectorAll('[data-col-toggle]'));
  let columnKeys = Array.from(new Set([...(hooks.columnKeys || []), ...columnButtons.map(function (button) {
    return button.dataset.colToggle;
  })]));
  const viewCheckboxes = Array.from(document.querySelectorAll('[data-view-detailed]'));

  // Compare-any-combination model (TC v1 batch tab): pick one Base and any number
  // of Compare candidates; each cell shows how many selected candidates differ
  // from Base ("=" if none), plus ↯ when Base leaks markup. Tooltip shows Base +
  // each selected candidate. All client-side from each cell's data-cmp payload.
  // Compare is per-panel: each tab's .cmpctl has its own Reference radios + Compare checkboxes.
  function activePanel() { return document.querySelector('.tab-panel.active'); }
  function panelCtl(panel) { return panel ? panel.querySelector('.cmpctl') : null; }
  // Reference = the single checked radio; Compare-with = every checked checkbox.
  function ctlBase(ctl) { if (hooks.engineSelection) return hooks.engineSelection.reference; const r = ctl && ctl.querySelector('input.cmp-ref:checked'); return r ? r.value : null; }
  function ctlShown(ctl) {
    if (hooks.engineSelection) return hooks.engineSelection.selection().selected;
    return ctl ? Array.from(ctl.querySelectorAll('input.cmp-on:checked')).map(function (x) { return x.value; }) : [];
  }
  // The Reference parser can't compare to itself: disable (and flag) its own
  // Compare checkbox, re-enable every other row's.
  function syncRefDisable(ctl, base, allowNoReference = false) {
    ctl.querySelectorAll('input.cmp-on').forEach(function (cb) {
      const isRef = cb.value === base;
      // The Reference is always part of the comparison (it's the base everything is
      // compared against), so its Compare box shows pressed + locked. It's filtered
      // out of the compare set in applyCtl, so it isn't double-counted.
      if (isRef) { cb.checked = true; }
      // Invariant: a Compare-with can never exist without a Reference. With no base
      // (the cleared state) every box is unchecked + locked, so the only possible
      // action is starring a new Reference. With a base, the ref's box is the locked
      // one and every other row is free to toggle.
      if (!base && !allowNoReference) { cb.checked = false; }
      cb.disabled = isRef || (!base && !allowNoReference);
      const row = cb.closest('.cmprow');
      if (row) { row.classList.toggle('is-ref', isRef); }
    });
  }
  // Compare keys are `<impl>-s-<version>` / `<impl>-b-<version>` (or `<impl>-s`); the
  // per-chunk grid columns are keyed by bare impl. Strip the mode+version suffix.
  function implOf(key) { return key ? key.replace(/-[sb](-.*)?$/, '') : key; }
  // All popup grids follow the slot selection, with the oracle first.
  function orderCandidateColumns(table, active, base) {
    const keys = Array.from(active);
    function rank(element) {
      const key = element.getAttribute('data-cand');
      if (element.getAttribute('data-cand-pin') === '1' || key === 'golden') return -2;
      if (key === base) return -1;
      const index = keys.indexOf(key);
      return index < 0 ? keys.length : index;
    }
    table.querySelectorAll('tr').forEach(function (row) {
      const columns = Array.from(row.querySelectorAll('[data-cand]'));
      columns.sort((a, b) => rank(a) - rank(b)
        || Number(a.dataset.candOrder || 0) - Number(b.dataset.candOrder || 0));
      columns.forEach(column => row.appendChild(column));
    });
  }
  function toggleCands(cell, active, base) {
    let baseSec = null;
    const tip = cell._ttip || cell.querySelector('.ttip');
    if (!tip) return;
    tip.querySelectorAll('.cand').forEach(function (sec) {
      const cls = Array.from(sec.classList).find(function (c) { return c.indexOf('cand-') === 0; });
      const key = cls ? cls.slice(5) : null;
      sec.classList.toggle('cand-on', key !== null && active.has(key));
      // Mark the Base reference's section so the tooltip flags which output the
      // others are being compared against.
      const isBase = key !== null && key === base;
      sec.classList.toggle('cand-base', isBase);
      if (isBase) { baseSec = sec; }
    });
    // The Reference reads FIRST: move its section ahead of its sibling candidates.
    if (baseSec) {
      const first = baseSec.parentNode.querySelector('.cand');
      if (first && first !== baseSec) { baseSec.parentNode.insertBefore(baseSec, first); }
    }
    // Per-chunk grid: show only the columns in the Reference + Compare-with selection.
    tip.querySelectorAll('.ttip-chunks').forEach(function (grid) {
      const cands = grid.querySelectorAll('[data-cand]');
      if (cands.length) {
        // Candidate-column grid: columns ARE the (impl, version) candidates. Show the
        // Reference + each checked compare-with, flag the Reference column, and move
        // it first (right after the input column) in every row.
        cands.forEach(function (el) {
          const key = el.getAttribute('data-cand');
          el.classList.toggle('col-hidden', !(el.getAttribute('data-cand-pin') === '1' || active.has(key)));
          el.classList.toggle('col-ref', key === base);
        });
        orderCandidateColumns(grid, active, base);
      } else {
        // Legacy impl-column grid: show columns whose engine is active.
        const activeImpls = new Set(Array.from(active).map(implOf));
        grid.querySelectorAll('[data-col-impl]').forEach(function (el) {
          el.classList.toggle('col-hidden', !activeImpls.has(el.getAttribute('data-col-impl')));
        });
      }
    });
  }
  // Parsers with limited family coverage (Dynamo v2): key -> {label, families}. Drives
  // the reference-aware "not implemented" reason. Empty on pages without such a parser.
  const PARSER_NI = window.__PARSER_NI || {};
  // Show/clear a JS-driven "why n/a" line at the top of a cell's tooltip. Built in JS
  // (not the server-rendered tooltip) so the reason can change with the Reference.
  function setWhy(cell, text) {
    const tip = cell._ttip || cell.querySelector('.ttip');
    if (!tip) return;
    let el = tip.querySelector('.cmp-why');
    if (!text) { if (el) { el.remove(); } return; }
    if (!el) {
      el = document.createElement('div');
      el.className = 'cmp-why';
      tip.insertBefore(el, tip.firstChild);
    }
    el.textContent = text;
  }
  const cmpDefaults = {};  // panel id -> {base, shown} captured before URL restore
  function _sameLayout(pid, base, shown) {
    const d = cmpDefaults[pid];
    if (!d) { return false; }
    if ((d.base || '') !== (base || '')) { return false; }
    if (d.shown.length !== shown.length) { return false; }
    const set = new Set(d.shown);
    return shown.every(function (k) { return set.has(k); });
  }
  function updateCompareUrl(panel, ctl) {
    if (hooks.engineSelection) return;
    const url = new URL(window.location.href);
    const pid = panel.id;
    const base = ctlBase(ctl);
    const b = ctlShown(ctl).filter(function (key) { return key !== base; });
    url.searchParams.delete('base_' + pid); url.searchParams.delete('cmp_' + pid);
    // Only record params when this panel differs from its default layout, so the
    // default state keeps a clean URL (and Reset lands on an empty query).
    if (!_sameLayout(pid, base, b)) {
      // A null base is a deliberate "no reference" state (the star was toggled off),
      // distinct from the default layout — record it as an empty sentinel so a reload
      // restores no-ref instead of falling back to the default reference.
      url.searchParams.set('base_' + pid, base || '');
      if (b.length) { url.searchParams.set('cmp_' + pid, b.join(',')); }
    }
    window.history.replaceState(null, '', url.toString());
  }
  function restoreCtlFromUrl(panel, ctl) {
    if (hooks.engineSelection) return;
    const params = new URLSearchParams(window.location.search);
    const pid = panel.id;
    if (!params.has('base_' + pid) && !params.has('cmp_' + pid)) { return; }  // keep defaults
    const base = params.has('base_' + pid) ? params.get('base_' + pid) : ctlBase(ctl);
    const refs = Array.from(ctl.querySelectorAll('input.cmp-ref'));
    // A removed capture in an old link is not an intentional empty reference.
    if (base && !refs.some(function (r) { return r.value === base; })) { return; }
    const inB = new Set((params.get('cmp_' + pid) || '').split(',').filter(Boolean));
    refs.forEach(function (r) { r.checked = (r.value === base); });
    ctl.querySelectorAll('input.cmp-on').forEach(function (cb) {
      cb.checked = cb.value !== base && inB.has(cb.value);
    });
  }
  const segmentLabels = window.__conformanceSegmentLabels;
  function renderSegments(cell, keys) {
    const model = window.__conformanceTooltipModel && window.__conformanceTooltipModel(cell);
    const cmp = JSON.parse(cell.dataset.cmp || '{}');
    const data = (window.__conformanceCellModel && window.__conformanceCellModel(cell))
      || { kind: 'cell', cmp, tooltip: model };
    let segments = cell.querySelector('.matrix-segments');
    if (!segments) {
      segments = document.createElement('span');
      segments.className = 'matrix-segments';
      const marker = cell.querySelector('.cmp-marker');
      if (marker) marker.replaceWith(segments); else cell.insertBefore(segments, cell.firstChild);
    }
    segments.replaceChildren();
    const labels = keys.map(key => {
      const status = window.__conformanceCaptureOutcome(data, key, 'golden');
      const segment = document.createElement('i');
      segment.className = 'matrix-status status-' + status;
      segment.dataset.capture = key;
      segment.setAttribute('aria-label', key + ': ' + segmentLabels[status]);
      segments.appendChild(segment);
      return key + ': ' + segmentLabels[status];
    });
    cell.setAttribute('aria-label', labels.join('; '));
    const tip = cell._ttip || cell.querySelector('.ttip');
    if (tip && tip.classList.contains('ttip-built')) {
      let summary = tip.querySelector('.capture-status-summary');
      if (!summary) {
        summary = document.createElement('div');
        summary.className = 'capture-status-summary';
        tip.appendChild(summary);
      }
      summary.textContent = labels.join('; ');
    }
  }
  function applyCtl(panel) {
    const ctl = panelCtl(panel);
    const oracle = panel.id === 'tab-unified' && !!panel._comparisonSelection;
    if (!ctl && !oracle) { return; }
    const base = oracle ? (panel._comparisonSelection.keys.length ? 'golden' : null) : ctlBase(ctl);
    if (ctl) syncRefDisable(ctl, oracle ? null : base, oracle);
    const shown = (oracle ? panel._comparisonSelection.keys : ctlShown(ctl)).filter(function (k) { return k !== base; });
    const active = new Set((base ? [base] : []).concat(shown));
    const counts = { ok: 0, problem: 0, na: 0 };
    // If the selected Reference is a parser with limited family coverage (e.g. the
    // Dynamo v2 parser), ni holds {label, families}. A cell whose family is not in
    // that list is "not implemented" — a different reason than case-level "not
    // applicable", and it wins.
    // Cells carry data-cmp (compare payload) and, on the v2 page, data-family (for the
    // reference-aware "not implemented" note). The v1 parity page has data-cmp only, so
    // match either — otherwise v1 cells never get colored.
    panel.querySelectorAll('td.cell[data-cmp], td.cell[data-family]').forEach(function (cell) {
      // Cloned cells in the transposed mirror get colored like any other cell,
      // but must not be tallied into the overview counts (they'd double them).
      const countThis = !cell.closest('[data-transpose-table]');
      let cmp = {};
      const raw = cell.getAttribute('data-cmp');
      if (raw) { try { cmp = JSON.parse(raw); } catch (e) { return; } }
      if (oracle) renderSegments(cell, shown);
      cell.classList.remove('cmp-eq', 'cmp-leak', 'cmp-na', 'cmp-nobase', 'cmp-donly');
      const marker = cell.querySelector('.cmp-marker .marker-text');
      const fam = cell.getAttribute('data-family') || '';
      const outcome = hooks.cellOutcome({ cmp, base, shown, family: fam,
        parserNi: PARSER_NI, redOnDiff: cell.getAttribute('data-red-on-diff') === '1' });
      cell.classList.add(outcome.className);
      if (marker) marker.textContent = outcome.marker;
      setWhy(cell, outcome.why);
      if (countThis) counts[outcome.category]++;
      toggleCands(cell, active, base);
    });
    panel.querySelectorAll('[data-overview-count]').forEach(function (el) {
      const k = el.dataset.overviewCount;
      el.textContent = String(k === 'todo' ? 0 : (counts[k] || 0));
    });
    window.__conformanceUpdateLegend(panel);
    repositionVisibleTooltips();
    // Column grammar popups (header activation): output columns are one-per-candidate. golden
    // is PINNED (always shown, leftmost); the rest show only when active (Reference +
    // selected compares). Flag the Reference column and order golden -> REF -> the rest,
    // the same rule the cell-popup candidate grid uses.
    panel.querySelectorAll('[data-ttip-id]').forEach(cell => syncTooltip(cell, [...active], base, panel.id === 'tab-unified'));
    if (!panel._comparisonSelection) updateCompareUrl(panel, ctl);
  }
  // Cell popups, case grammar tables and drawers use the same capture visibility policy.
  function syncTooltip(host, keys, base, oracle) {
    const active = new Set(keys);
    if (oracle) active.add('golden');
    toggleCands(host, active, base);
  }
  window.__syncConformanceTooltip = syncTooltip;
  window.__applyConformanceSelection = function (panel, keys, reference) {
    const ctl = panelCtl(panel);
    if (!ctl) return;
    ctl.querySelectorAll('input.cmp-ref').forEach(input => { input.checked = input.value === reference; });
    ctl.querySelectorAll('input.cmp-on').forEach(input => { input.checked = keys.includes(input.value); });
    ctl.dataset.prevBase = reference || '';
    applyCtl(panel);
  };
  function applyCompare() { const p = activePanel(); if (p) { applyCtl(p); } }
  function _boxFor(ctl, val) {
    return Array.prototype.find.call(
      ctl.querySelectorAll('input.cmp-on'), function (x) { return x.value === val; }) || null;
  }
  // Picking a new Reference (starring a row) clears the previous comparison. The
  // starred row is the only visible version until the user explicitly checks another
  // Compare box.
  function handleRefChange(ctl) {
    const newBase = ctlBase(ctl);
    ctl.querySelectorAll('input.cmp-on').forEach(function (box) { box.checked = false; });
    const newBox = _boxFor(ctl, newBase);
    if (newBox) { newBox.checked = true; }
    ctl.dataset.prevBase = newBase || '';
  }
  // Toggling the current Reference star OFF (clicking the already-selected star):
  // drop the previous reference's pressed Compare box and clear the base. With no
  // base, applyCtl paints every cell cmp-nobase, so the whole panel clears.
  function handleRefUncheck(ctl) {
    const oldBase = ctl.dataset.prevBase || '';
    if (oldBase) {
      const oldBox = _boxFor(ctl, oldBase);
      if (oldBox) { oldBox.checked = false; }
    }
    ctl.dataset.prevBase = '';
  }
  // Re-color the panel whenever a Reference star or Compare checkbox toggles.
  function initCompareInputs() {
    document.querySelectorAll('.cmpctl').forEach(function (ctl) {
      ctl.dataset.prevBase = ctlBase(ctl) || '';
      // Radios don't fire `change` when you click the already-checked one, and there's
      // no native "uncheck". The visible control is the ★ label; the radio is offscreen,
      // so the real pointer lands on the LABEL and the radio gets only a synthesized
      // click (never a mousedown). Wire both listeners to the label: record the pre-click
      // state on mousedown, and on click, if it was already the Reference, prevent the
      // re-activation, uncheck it, and drive the update ourselves.
      ctl.querySelectorAll('input.cmp-ref').forEach(function (r) {
        const label = r.closest('label') || r.parentElement;
        if (!label) { return; }
        label.addEventListener('mousedown', function () { r.dataset.wasChecked = r.checked ? '1' : '0'; });
        label.addEventListener('click', function () {
          if (ctl.closest('.tab-panel')._comparisonSelection) return;
          if (r.dataset.wasChecked !== '1') { return; }
          r.dataset.wasChecked = '0';
          // Only clear the Reference when nothing else is selected to compare against.
          // With one or more Compare checkboxes active (its own pressed box excepted)
          // there's still a chart to show, so the star stays put.
          const otherCompares = Array.prototype.filter.call(
            ctl.querySelectorAll('input.cmp-on:checked'),
            function (cb) { return cb.value !== r.value; });
          if (otherCompares.length > 0) { return; }
          // The browser (re)activates the radio as the click's default action AFTER this
          // listener, so an inline uncheck gets clobbered. Defer it to the next tick so
          // it lands after activation: uncheck, clear the base, repaint cmp-nobase.
          setTimeout(function () {
            r.checked = false;
            handleRefUncheck(ctl);
            const panel = ctl.closest('.tab-panel');
            if (panel) { applyCtl(panel); hooks.selectionChanged?.(panel); }
          }, 0);
        });
      });
      ctl.addEventListener('change', function (e) {
        const selection = ctl.closest('.tab-panel')._comparisonSelection;
        if (selection) { selection.fromControls(e.target); return; }
        if (e.target.matches('input.cmp-ref')) { handleRefChange(ctl); }
        if (e.target.matches('input.cmp-ref, input.cmp-on')) {
          const panel = ctl.closest('.tab-panel');
          if (panel) { applyCtl(panel); hooks.selectionChanged?.(panel); }
        }
      });
    });
  }
  function initCompare() {
    document.querySelectorAll('.tab-panel').forEach(function (panel) {
      const ctl = panelCtl(panel);
      if (!ctl) { return; }
      // Record the server-rendered default layout BEFORE applying any URL state,
      // so updateCompareUrl can keep the URL clean while at defaults.
      const base = ctlBase(ctl);
      cmpDefaults[panel.id] = {
        base: base,
        shown: ctlShown(ctl).filter(function (key) { return key !== base; }),
      };
      restoreCtlFromUrl(panel, ctl);
    });
    initCompareInputs();
    document.querySelectorAll('.tab-panel').forEach(function (panel) {
      if (panelCtl(panel)) { applyCtl(panel); }
    });
    // Compare model applied — reveal the real colors (see body.cmp-loading CSS).
    document.body.classList.remove('cmp-loading');
  }

  function readDetailed() {
    // Overview (non-detailed) is the default; ?view=details turns it on.
    return new URLSearchParams(window.location.search).get('view') === 'details';
  }

  function updateViewUrl(detailed) {
    const url = new URL(window.location.href);
    url.searchParams.delete('view');
    if (detailed) {
      url.searchParams.set('view', 'details');
    }
    window.history.replaceState(null, '', url.toString());
  }

  function applyView(detailed, shouldUpdateUrl) {
    document.body.classList.toggle('view-overview', !detailed);
    document.body.classList.toggle('view-details', detailed);
    viewCheckboxes.forEach(function (cb) { cb.checked = detailed; });
    // Legacy tabs keep their existing compare reset. Unified's capture selector owns
    // this selection across Overview and Detailed, so its hidden controls must stay in sync.
    if (!detailed && Object.keys(cmpDefaults).length > 0) {
      document.querySelectorAll('.tab-panel').forEach(function (panel) {
        if (panel._comparisonSelection) { return; }
        const ctl = panelCtl(panel);
        if (!ctl) { return; }
        ctl.querySelectorAll('input.cmp-on:not(:disabled)').forEach(function (cb) {
          cb.checked = false;
        });
        applyCtl(panel);
        if (shouldUpdateUrl && panel.classList.contains('active')) hooks.selectionChanged?.(panel);
      });
    }
    document.querySelectorAll('.tab-panel').forEach(window.__conformanceUpdateLegend);
    if (shouldUpdateUrl) {
      updateViewUrl(detailed);
    }
  }

  initCompare();
  applyView(readDetailed(), false);
  viewCheckboxes.forEach(function (cb) {
    cb.addEventListener('change', function () { applyView(cb.checked, true); });
  });
  // Reset: drop every URL param (compare/view state) and reload at defaults, but
  // stay on the current tab.
  document.querySelectorAll('[data-reset]').forEach(function (btn) {
    btn.addEventListener('click', function () {
      const tab = new URLSearchParams(window.location.search).get('tab');
      const u = new URL(window.location.href);
      for (const key of [...u.searchParams.keys()]) if (/^(?:base_|cmp_)/.test(key) || ['slots_tab-unified', 'tab', 'ref', 'cmp', 'view', 'transpose', 'cols', 'show', 'hide', 'popup'].includes(key)) u.searchParams.delete(key);
      if (tab) { u.searchParams.set('tab', tab); }
      window.location.href = u.href;  // reload at defaults, same tab
    });
  });

  function defaultVisibleColumns() {
    const visible = new Set(hooks.columnKeys || []);
    columnButtons.forEach(function (button) {
      if (button.dataset.defaultVisible === 'true') {
        visible.add(button.dataset.colToggle);
      } else visible.delete(button.dataset.colToggle);
    });
    return visible;
  }

  function parseColumnList(raw) {
    const columns = new Set();
    if (raw === null) {
      return columns;
    }
    raw.split(',').forEach(function (key) {
      if (columnKeys.includes(key)) {
        columns.add(key);
      }
    });
    return columns;
  }

  function readVisibleColumns() {
    const params = new URLSearchParams(window.location.search);
    const legacyRaw = params.get('cols');
    if (legacyRaw !== null) {
      const legacyVisible = parseColumnList(legacyRaw);
      return legacyVisible.size > 0 ? legacyVisible : defaultVisibleColumns();
    }
    const visible = defaultVisibleColumns();
    parseColumnList(params.get('show')).forEach(function (key) {
      visible.add(key);
    });
    parseColumnList(params.get('hide')).forEach(function (key) {
      visible.delete(key);
    });
    return visible;
  }

  function updateUrl(visible) {
    const url = new URL(window.location.href);
    const defaults = defaultVisibleColumns();
    const show = columnKeys.filter(function (key) {
      return visible.has(key) && !defaults.has(key);
    });
    const hide = columnKeys.filter(function (key) {
      return !visible.has(key) && defaults.has(key);
    });
    url.searchParams.delete('cols');
    url.searchParams.delete('show');
    url.searchParams.delete('hide');
    if (show.length > 0) {
      url.searchParams.set('show', show.join(','));
    }
    if (hide.length > 0) {
      url.searchParams.set('hide', hide.join(','));
    }
    window.history.replaceState(null, '', url.toString());
  }

  function applyColumnState(visible, shouldUpdateUrl) {
    columnButtons.forEach(function (button) {
      const key = button.dataset.colToggle;
      const isVisible = visible.has(key);
      button.setAttribute('aria-pressed', isVisible ? 'true' : 'false');
      button.setAttribute(
        'aria-label',
        (isVisible ? 'Collapse ' : 'Expand ') + button.dataset.colLabel + ' column'
      );
      // Preserve the accessible name while exposing the same text in the page tooltip.
      button.dataset.tooltip = button.getAttribute('aria-label');
      button.dataset.ttipText = button.dataset.tooltip;
      let ttip = button._ttip || button.querySelector('.ttip');
      if (!ttip) {
        ttip = document.createElement('span');
        ttip.className = 'ttip column-control-tooltip';
        const label = document.createElement('span');
        label.className = 'column-control-tooltip-text';
        label.textContent = button.dataset.ttipText;
        ttip.appendChild(label);
        button.appendChild(ttip);
      } else {
        ttip.querySelector('.column-control-tooltip-text').textContent = button.dataset.ttipText;
      }
      attachTooltip(button);
      document.querySelectorAll('[data-col-control-group="' + key + '"]').forEach(function (el) {
        el.classList.toggle('col-collapsed', !isVisible);
        if (el.dataset.expandedColspan) {
          el.colSpan = isVisible ? Number(el.dataset.expandedColspan) : 1;
        }
      });
      document.querySelectorAll('[data-col-hide-group="' + key + '"]').forEach(function (el) {
        el.classList.toggle('col-hidden', !isVisible);
      });
      document.querySelectorAll('[data-col-placeholder-group="' + key + '"]').forEach(function (el) {
        el.classList.toggle('col-hidden', isVisible);
      });
    });
    document.querySelectorAll('table[data-parity-table]').forEach(function (table) {
      let visibleColumnCount = 0;
      table.querySelectorAll('[data-col-toggle]').forEach(function (button) {
        const key = button.dataset.colToggle;
        const span = Number(button.dataset.colSpan || '1');
        visibleColumnCount += visible.has(key) ? span : 1;
      });
      table.querySelectorAll('[data-section-span]').forEach(function (el) {
        el.colSpan = visibleColumnCount;
      });
    });
    document.querySelectorAll('.tab-panel').forEach(window.__conformanceUpdateLegend);
    updateStickyOffsets();
    if (shouldUpdateUrl) {
      updateUrl(visible);
    }
  }

  // Label widths and header heights vary by tab, view, and collapsed column group. Measure
  // them after each layout change so CSS sticky positioning does not hide either axis.
  function updateStickyOffsets() {
    document.querySelectorAll('table[data-parity-table], table[data-transpose-table]')
      .forEach(function (table) {
        const topRow = table.querySelector(
          'thead > tr.matrix-groups, thead > tr.transpose-sections'
        );
        table.style.setProperty(
          '--sticky-top-row-height',
          topRow ? topRow.getBoundingClientRect().height + 'px' : '0px'
        );
        if (table.matches('[data-parity-table]')) {
          const modelHeader = table.querySelector(
            'thead th[data-col-control-group="model"]'
          );
          const parserHeader = table.querySelector(
            'thead th[data-col-control-group="parser"]'
          );
          const modelWidth = modelHeader && modelHeader.offsetParent !== null
            ? modelHeader.getBoundingClientRect().width : 0;
          const parserWidth = parserHeader && parserHeader.offsetParent !== null
            ? parserHeader.getBoundingClientRect().width : 0;
          table.style.setProperty('--sticky-model-width', modelWidth + 'px');
          table.style.setProperty('--sticky-parser-width', parserWidth + 'px');
        } else {
          const caseHeaders = Array.from(table.querySelectorAll(
            'thead th.tcorner-case, tbody th.trow-case'
          )).filter(function (header) { return header.offsetParent !== null; });
          const caseWidth = caseHeaders.reduce(function (width, header) {
            return Math.max(width, header.getBoundingClientRect().width);
          }, 0);
          table.style.setProperty('--sticky-case-width', caseWidth + 'px');
        }
      });
  }

  let visibleColumns = readVisibleColumns();
  applyColumnState(visibleColumns, false);
  if (new URLSearchParams(window.location.search).has('cols')) {
    updateUrl(visibleColumns);
  }
  document.addEventListener('click', function (event) {
    const button = event.target.closest('[data-col-toggle]');
    if (!button) return;
    const key = button.dataset.colToggle;
    visibleColumns = new Set(visibleColumns);
    if (visibleColumns.has(key)) visibleColumns.delete(key); else visibleColumns.add(key);
    applyColumnState(visibleColumns, true);
  });

  const tabButtons = Array.from(document.querySelectorAll('.tab-button'));
  const tabPanels = Array.from(document.querySelectorAll('.tab-panel'));

  function readActiveTab() {
    const params = new URLSearchParams(window.location.search);
    const requested = params.get('tab');
    const validTargets = new Set(tabButtons.map(function (button) {
      return button.dataset.tabTarget;
    }));
    if (requested && validTargets.has(requested)) {
      return requested;
    }
    const active = tabButtons.find(function (button) {
      return button.classList.contains('active');
    });
    return active ? active.dataset.tabTarget : (tabButtons[0] && tabButtons[0].dataset.tabTarget);
  }

  function updateTabUrl(id) {
    const url = new URL(window.location.href);
    url.searchParams.set('tab', id);
    window.history.replaceState(null, '', url.toString());
  }

  function activateTab(id, shouldUpdateUrl) {
    if (!id) return;
    if (pinnedCell) unpinCell(pinnedCell);
    visibleTooltips.forEach(function (ttip) {
      const owner = ttip._ttipOwner;
      if (owner && owner.closest('.tab-panel')?.id !== id) {
        owner._ttipUnpin ? owner._ttipUnpin() : ttip.classList.remove('ttip-visible');
      }
    });
    tabButtons.forEach(function (button) {
      const selected = button.dataset.tabTarget === id;
      button.classList.toggle('active', selected);
      button.setAttribute('aria-selected', selected ? 'true' : 'false');
    });
    tabPanels.forEach(function (panel) {
      panel.classList.toggle('active', panel.id === id);
    });
    // Each tab keeps its own Base/Compare/Others (its own URL state or default) —
    // no carry-over from the previously-viewed tab.
    applyCompare();
    if (shouldUpdateUrl) {
      updateTabUrl(id);
    }
    updateStickyOffsets();
  }
  // Equal columns, sized to what the WIDEST column actually needs.
  //
  // Pure CSS gives one or the other, not both: `table-layout: fixed; width: 100%`
  // makes columns equal but always consumes the whole popup (a chart of `–` cells
  // took as much room as one full of arguments), while auto layout sizes the table
  // to its content but leaves every column a different width, which reads as messy
  // when the point is to compare engines side by side.
  //
  // So measure, then pin. Under auto + `max-content` each rendered column width IS
  // that column's max-content (the width at which its longest line does not wrap);
  // take the widest, then switch to fixed layout with `width = n * widest`, which
  // fixed layout divides into n equal columns of exactly that width. If that exceeds
  // the popup's viewport cap, `max-width: 100%` shrinks it — columns stay equal
  // because fixed layout keeps dividing evenly.
  //
  // Memoized per table: the result only depends on content, which does not change.
  function equalizeColumns(ttip) {
    const tables = ttip.querySelectorAll('.ttip-chunks, .ttip-grammar');
    for (const t of tables) {
      if (t.dataset.eqCols === '1') continue;
      const row = t.querySelector('tr');
      if (!row || !row.children.length) continue;
      t.style.tableLayout = 'auto';
      t.style.width = 'max-content';
      const natural = [];
      for (const c of row.children) natural.push(c.getBoundingClientRect().width);

      // The grammar popup's first column is only a model name ("qwen3", "kimi_k2"),
      // so equalizing it just pads every row by a couple hundred pixels. Leave it at
      // its natural width and equalize only the columns actually being compared. The
      // chunk chart has no such column — its first column is the input/golden stream,
      // which belongs in the comparison — so there it stays equal with the rest.
      const keepFirst = t.classList.contains('ttip-grammar') && natural.length > 1;
      const compared = keepFirst ? natural.slice(1) : natural;
      const widest = Math.max.apply(null, compared);

      if (widest > 0) {
        const firstW = keepFirst ? Math.ceil(natural[0]) : 0;
        t.style.tableLayout = 'fixed';
        t.style.width = Math.ceil(firstW + widest * compared.length) + 'px';
        t.style.maxWidth = '100%';
        // Under fixed layout the FIRST row's explicit widths pin the columns; the
        // rest split what is left, which is exactly `widest` each.
        if (keepFirst) row.children[0].style.width = firstW + 'px';
      }
      t.dataset.eqCols = '1';
    }
  }

  function boundedPopupLeft(left, width) {
    return Math.max(margin, Math.min(left, window.innerWidth - margin - width));
  }

  function placeCaseHint(cell) {
    const anchor = cell.matches('[data-hover-hint]') ? cell : cell.querySelector('[data-hover-hint]');
    if (!anchor) return;
    const hint = getComputedStyle(anchor, '::after');
    const width = parseFloat(hint.width);
    const height = parseFloat(hint.height);
    const box = anchor.getBoundingClientRect();
    anchor.style.setProperty('--case-hint-left', (boundedPopupLeft(box.left, width) - box.left) + 'px');
    // Keep the label outside the hand cursor, with a below-header fallback near the viewport top.
    anchor.style.setProperty('--case-hint-top', (box.top >= height + margin + 12 ? -height - 12 : box.height + 24) + 'px');
  }

  function place(cell) {
    const ttip = cell._ttip || cell.querySelector('.ttip');
    if (!ttip) return;
    visibleTooltips.add(ttip);
    equalizeColumns(ttip);
    ttip.style.visibility = 'hidden';
    ttip.style.opacity = '0';
    ttip.classList.add('ttip-visible');
    const isPortalled = ttip.parentNode === document.body;
    ttip.style.left = '0px';
    ttip.style.top = '0px';
    ttip.style.right = 'auto';
    ttip.style.bottom = 'auto';
    // Cap the width at the viewport MINUS both gutters — that is the most a popup may
    // use, so on a narrow screen it grows to fill the window while still leaving the
    // left/right padding. This is only a CEILING: the popup is `width: max-content`, so
    // a chart that needs less stays narrow rather than padding itself out to the cap.
    // (The old cap also took 90% of the viewport, which on a small screen threw away a
    // gutter's worth of usable room on top of the two real gutters.)
    ttip.style.maxWidth = Math.max(240, window.innerWidth - 2 * margin) + 'px';
    const cellRect = cell.getBoundingClientRect();
    if (cell.dataset.ttipClick === 'pin') {
      ttip.style.maxWidth = Math.max(240, window.innerWidth - 2 * margin) + 'px';
      const controls = cell.closest('#tab-unified')?.querySelector('.capture-slots');
      const requestedTop = controls ? controls.getBoundingClientRect().bottom + margin : margin;
      const resized = ttip.style.height && ttip.style.height !== 'auto';
      const controlsBottom = resized
        ? requestedTop
        : Math.max(margin, Math.min(requestedTop, Math.max(margin, window.innerHeight - 160 - margin)));
      const availableHeight = Math.max(160, window.innerHeight - controlsBottom - margin);
      ttip.style.maxHeight = (resized ? window.innerHeight * .97 : availableHeight) + 'px';
      const rect = ttip.getBoundingClientRect();
      ttip.style.left = ((window.innerWidth - rect.width) / 2 + window.scrollX) + 'px';
      const top = resized ? Math.min(controlsBottom, Math.max(margin, window.innerHeight - rect.height - margin)) : controlsBottom;
      ttip.style.top = (top + window.scrollY) + 'px';
      ttip.style.visibility = '';
      ttip.style.opacity = '';
      return;
    }
    // An expanded schema may fit on neither side; scroll it within the larger space.
    const schemaOpen = ttip.querySelector('.case-schema[open]');
    ttip.style.maxHeight = schemaOpen
      ? Math.max(0, Math.min(window.innerHeight - 2 * margin,
        Math.max(cellRect.top - margin, window.innerHeight - cellRect.bottom - margin))) + 'px'
      : '';
    const tipRect = ttip.getBoundingClientRect();
    const vh = window.innerHeight;
    const shiftX = boundedPopupLeft(cellRect.left, tipRect.width) - cellRect.left;
    const placeAbove = cellRect.bottom + tipRect.height > vh - margin
      && cellRect.top - tipRect.height >= margin;
    let top = placeAbove ? cellRect.top - tipRect.height : cellRect.bottom;
    if (schemaOpen) {
      // Resizing can move the focused owner outside the viewport; keep its disclosure readable.
      top = Math.max(margin, Math.min(top, vh - margin - tipRect.height));
    }
    if (isPortalled) {
      ttip.style.left = (cellRect.left + window.scrollX + shiftX) + 'px';
      ttip.style.top = (top + window.scrollY) + 'px';
    } else {
      // Measure the containing origin so table-cell borders count toward the viewport gutters.
      ttip.style.left = (cellRect.left + shiftX - tipRect.left) + 'px';
      ttip.style.top = (top - tipRect.top) + 'px';
    }
    ttip.style.visibility = '';
    ttip.style.opacity = '';
  }

  function repositionVisibleTooltips() {
    if (placeScheduled) return;
    placeScheduled = true;
    window.requestAnimationFrame(function () {
      placeScheduled = false;
      document.querySelectorAll('[data-hover-hint]:hover, [data-hover-hint]:focus-visible').forEach(placeCaseHint);
      visibleTooltips.forEach(function (ttip) {
        if (ttip.classList.contains('ttip-visible') && ttip._ttipOwner) {
          place(ttip._ttipOwner);
        } else {
          visibleTooltips.delete(ttip);
        }
      });
    });
  }
  window.addEventListener('scroll', repositionVisibleTooltips, true);
  window.addEventListener('resize', repositionVisibleTooltips);
  document.addEventListener('toggle', function (event) {
    if (!event.target.matches('.case-schema')) return;
    const ttip = event.target.closest('.ttip');
    if (ttip && ttip._ttipOwner && ttip.classList.contains('ttip-visible')) {
      place(ttip._ttipOwner);
    }
  }, true);

  // Only one tooltip is pinned at a time.
  // NEVER gate event REGISTRATION on a media query. `(hover: hover)` and
  // `(any-hover: hover)` are advisory descriptions of a device; they are not a statement
  // about which events the browser will deliver. Chrome on Keiven's desktop reports BOTH
  // false while delivering real hover — the cell matched CSS `:hover` and the lazy content
  // builder ran on `pointerover`, but no `pointerenter` listener existed to open the popup,
  // so hovering did nothing at all. The same gate silently removed keyboard access, because
  // `focusin`/`focusout` sat behind it too.
  //
  // So: listeners are always attached (below), and the ONE thing that genuinely needs to
  // know the input device — whether a tap should pin — is decided from the actual
  // `PointerEvent.pointerType` of the interaction in hand, which is a fact rather than a
  // guess. `pendingPointerType` is written on `pointerdown`, which always precedes the click
  // it belongs to.
  // CONSUMED by the click it belongs to, then cleared. Holding it until the next
  // `pointerdown` made it stale rather than click-scoped: a keyboard activation issues
  // no `pointerdown` at all, so after any touch the next Enter on a focused link would
  // still read `touch`, call `preventDefault()` and pin — suppressing the link the
  // keyboard user asked for. `pointercancel` clears it too, so an aborted gesture
  // cannot arm a later click.
  // Both the KIND of pointer and WHERE it went down. The kind alone is a document-wide
  // global: a tap that lands anywhere else — a header, a scroll, any element that is not a
  // tooltip host — leaves `touch` armed, because only a tooltip host's click consumes it.
  // The next MOUSE click on a host would then read that stale `touch`, pin, and
  // `preventDefault()` the parser-source link the mouse user actually asked for. Requiring
  // the pointerdown to have landed inside the same host makes the provenance belong to
  // this interaction rather than to whatever touched the page last.
  let pendingPointerType = '';
  let pendingPointerTarget = null;
  document.addEventListener('pointerdown', function (e) {
    pendingPointerType = e.pointerType || '';
    pendingPointerTarget = e.target;
  }, true);
  document.addEventListener('pointercancel', function () {
    pendingPointerType = '';
    pendingPointerTarget = null;
  }, true);
  // Hosts without explicit activation must not gain modal ownership from a mouse
  // click: their hover path already opens the tooltip. Touch/pen needs pinning because
  // those pointers cannot keep a hover tooltip open.
  const consumeClickShouldPin = function (cell) {
    const t = pendingPointerType;
    const target = pendingPointerTarget;
    pendingPointerType = '';
    pendingPointerTarget = null;
    if (t !== 'touch' && t !== 'pen') { return false; }
    // The gesture must have STARTED in this host, or it belongs to something else.
    return !!(cell && target && cell.contains(target));
  };
  let pinnedCell = null;
  function unpinCell(c) { if (c && c._ttipUnpin) { c._ttipUnpin(); } }
  function updatePinnedPopupUrl(cell) {
    const url = new URL(window.location.href);
    if (cell) {
      const panels = document.querySelectorAll('.tab-panel');
      const panel = cell.closest('.tab-panel');
      if (panels.length > 1 && panel) { url.searchParams.set('tab', panel.id); }
      else { url.searchParams.delete('tab'); }
      if (cell.dataset.ttipKey) { url.searchParams.set('popup', cell.dataset.ttipKey); }
      else { url.searchParams.delete('popup'); }
    } else {
      url.searchParams.delete('popup');
    }
    window.history.replaceState(null, '', url.toString());
  }
  // "Inside a pinnable element" must mean EXACTLY the set `attachTooltip` wired, so ask
  // for the mark `attachTooltip` itself leaves rather than re-listing the classes. A class
  // list cannot express that set: the transpose headers are wired directly (see below) and
  // never pass through WIRE_ON_LOAD, so every hand-written list drifted from it. Headers
  // ended up pinnable but not counted as inside — a header click pinned it and the SAME
  // click bubbled here and unpinned it, so the popup could never stay up.
  const WIRED = '[data-ttip-wired]';
  const pinsAtClickStart = new WeakMap();
  // A cached Retry may restore a different owner before this click bubbles.
  // Only the pin present at event entry belongs to the outside-click gesture.
  const popupControls = '.capture-slots, .comparison-picker, [data-col-toggle], [data-transpose-toggle], [data-theme-toggle], [data-view-detailed]';
  // A Retry handler may replace its own notice before this click bubbles. Retain
  // the original control ownership instead of classifying its detached target.
  document.addEventListener('click', event => {
    pinsAtClickStart.set(event, { pin: pinnedCell, control: Boolean(event.target.closest(popupControls)) });
  }, true);
  document.addEventListener('click', function (e) {
    const started = pinsAtClickStart.get(e);
    if (started?.control) return;
    // A click anywhere outside a pinnable element (and not on a pinned tooltip) closes the pin.
    if (pinnedCell && pinnedCell === started?.pin && !e.target.closest(WIRED) && !e.target.closest('.ttip')) { unpinCell(pinnedCell); }
  });
  document.addEventListener('keydown', function (event) {
    if (event.key === 'Escape' && pinnedCell && !document.querySelector('dialog[open]')) { event.preventDefault(); unpinCell(pinnedCell); }
  });

  function attachTooltip(cell) {
    const ttip = cell.querySelector('.ttip');
    if (!ttip) return;
    if (cell.hasAttribute('data-ttip-id') && !cell.classList.contains('col-toggle')) {
      cell.dataset.ttipClick = 'pin';
      cell.dataset.ttipPortal = '1';
      cell.tabIndex = 0;
    }
    if (!cell.dataset.popupKey && cell.dataset.colToggle) cell.dataset.popupKey = `control:${cell.dataset.colToggle}`;
    cell._ttip = ttip;
    ttip._ttipOwner = cell;
    // cloneNode copies the wired flag; guard so re-wiring a clone is a no-op and
    // originals aren't double-wired.
    if (cell.dataset.ttipWired === '1') return;
    cell.dataset.ttipWired = '1';

    function showFocusedTooltip() {
      cell._focusTooltip = true;
      if (window.__buildTooltip) window.__buildTooltip(cell);
    }

    let showTimer = null;
    let hideTimer = null;
    let isActive = false;
    let isVisible = false;
    let portalParent = null;

    cell._ttipDispose = function () {
      clearTimeout(showTimer);
      clearTimeout(hideTimer);
      unpin();
      ttip.remove();
    };

    function portalTooltip() {
      if (portalParent) return;
      if (cell.dataset.ttipPortal !== '1' && (cell._focusTooltip || (document.activeElement && cell.contains(document.activeElement)))) return;
      portalParent = ttip.parentNode;
      document.body.appendChild(ttip);
    }

    function restoreTooltip() {
      if (!portalParent) return;
      portalParent.appendChild(ttip);
      portalParent = null;
    }

    // ✕ close button (shown only while pinned) — inserted once per tooltip.
    let closeButton = ttip.querySelector('.ttip-close');
    if (!closeButton) {
      closeButton = document.createElement('button');
      closeButton.type = 'button';
      closeButton.className = 'ttip-close';
      closeButton.setAttribute('aria-label', 'Close');
      closeButton.textContent = '✕';
      ttip.insertBefore(closeButton, ttip.firstChild);
    }
    if (!closeButton.dataset.ttipCloseWired) {
      closeButton.addEventListener('click', function (e) { e.stopPropagation(); unpin(); });
      closeButton.dataset.ttipCloseWired = '1';
    }

    function pin() {
      if (pinnedCell && pinnedCell !== cell) { unpinCell(pinnedCell); }
      visibleTooltips.forEach(function (other) {
        if (other !== ttip) { unpinCell(other._ttipOwner); }
      });
      clearTimers();
      if (window.__buildTooltip) window.__buildTooltip(cell);
      portalTooltip();
      place(cell);
      ttip.classList.add('ttip-visible', 'ttip-pinned');
      isVisible = true;
      isActive = true;
      pinnedCell = cell;
      updatePinnedPopupUrl(cell);
      // The pressed look must mark the cell whose popup is actually up. While a pin
      // is held, hover no longer opens anything, so hover must not press either —
      // `ttip-pin-mode` switches the press off for :hover and onto this cell alone.
      cell.classList.add('ttip-open');
      document.body.classList.add('ttip-pin-mode');
    }
    function unpin() {
      const wasPinned = pinnedCell === cell;
      clearTimers();
      ttip.classList.remove('ttip-visible', 'ttip-pinned');
      visibleTooltips.delete(ttip);
      restoreTooltip();
      isVisible = false;
      isActive = false;
      cell.classList.remove('ttip-open');
      if (wasPinned) {
        if (cell.isConnected) cell.focus({ preventScroll: true });
        pinnedCell = null;
        document.body.classList.remove('ttip-pin-mode');
        updatePinnedPopupUrl(null);
      }
    }
    cell._ttipPin = pin;
    cell._ttipUnpin = unpin;
    cell.addEventListener('keydown', function (event) {
      if (event.target !== cell || cell.dataset.ttipClick !== 'pin') return;
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        if (ttip.classList.contains('ttip-pinned')) unpin(); else pin();
      }
    });
    cell._ttipPin = pin;
    cell._ttipDispose = function () { clearTimers(); unpin(); };

    function clearTimers() {
      if (showTimer !== null) {
        window.clearTimeout(showTimer);
        showTimer = null;
      }
      if (hideTimer !== null) {
        window.clearTimeout(hideTimer);
        hideTimer = null;
      }
    }

    function scheduleShow() {
      isActive = true;
      clearTimers();
      showTimer = window.setTimeout(function () {
        showTimer = null;
        if (!isActive || !hoverAllowed()) return;
        if (cell._focusTooltip && window.__buildTooltip) window.__buildTooltip(cell);
        portalTooltip();
        place(cell);
        ttip.classList.add('ttip-visible');
        isVisible = true;
      }, cell.classList.contains('col-toggle') ? columnShowDelayMs : showDelayMs);
    }

    function scheduleHide() {
      // A portalled disclosure still belongs to this popup while it owns focus.
      if (ttip.contains(document.activeElement)) { return; }
      isActive = false;
      if (showTimer !== null) {
        window.clearTimeout(showTimer);
        showTimer = null;
      }
      if (!isVisible) {
        ttip.classList.remove('ttip-visible');
        visibleTooltips.delete(ttip);
        restoreTooltip();
        return;
      }
      if (hideTimer !== null) {
        window.clearTimeout(hideTimer);
      }
      hideTimer = window.setTimeout(function () {
        hideTimer = null;
        if (isActive) return;
        ttip.classList.remove('ttip-visible');
        visibleTooltips.delete(ttip);
        restoreTooltip();
        isVisible = false;
      }, cell.classList.contains('col-toggle') ? columnHideDelayMs : hideDelayMs);
    }

    function keepTooltipOpen() {
      isActive = true;
      clearTimers();
    }

    cell.addEventListener('click', function (e) {
      // Consume FIRST, unconditionally. Every early return below still ends this click,
      // so leaving the provenance unconsumed on any of them hands a stale `touch` to the
      // NEXT click — the defect consume-on-use exists to prevent, one branch over. A tap
      // on the popup's own links or its ✕ takes this return, and used to leave it armed.
      const shouldPin = consumeClickShouldPin(cell);
      if (e.target.closest('.ttip')) { return; }
      if (cell.dataset.ttipClick === 'drawer') {
        unpin();
        return;
      }
      if (cell.dataset.ttipClick === 'pin') {
        e.preventDefault();
        if (ttip.classList.contains('ttip-pinned')) { unpin(); } else { pin(); }
        return;
      }
      if (!shouldPin) { return; }
      e.preventDefault();
      if (ttip.classList.contains('ttip-pinned')) { unpin(); } else { pin(); }
    });
    // Hover show/hide only where hover exists; on touch it just flickers. A pinned
    // tooltip ignores hover leave so it stays until ✕ / outside tap.
    //
    // Pinning is also MODAL: while any tooltip is pinned, other hover hosts do
    // nothing. Clicking a cell is a deliberate "keep this one open so I can read it",
    // and hover popups firing over the top of it defeat that — you cannot move the
    // mouse toward the pinned popup without crossing cells that each try to open
    // their own. Hover hosts resume when the pin is released (✕, clicking the pinned cell
    // again, or clicking outside).
    const hoverAllowed = function () { return pinnedCell === null || pinnedCell === cell; };
    const clickOnly = cell.dataset.ttipClick === 'drawer' || cell.dataset.ttipClick === 'pin';
    // Media queries cannot establish which pointer events a device delivers, so register
    // listeners unconditionally. Only non-click-only hosts may activate on hover or focus;
    // case hosts require explicit activation to avoid constructing popups during navigation.
    const onEnter = function () {
      placeCaseHint(cell);
      if (!clickOnly && hoverAllowed() && !ttip.classList.contains('ttip-pinned')) { scheduleShow(); }
    };
    const onLeave = function () {
      if (!ttip.classList.contains('ttip-pinned')) { scheduleHide(); }
    };
    cell.addEventListener('pointerenter', onEnter);
    cell.addEventListener('pointerleave', onLeave);
    // ALSO the classic mouse events, deliberately not instead. `pointerenter` is the
    // modern spelling, but it does not fire on movement when the browser classifies the
    // input as a touch pointer — and a device can report no hover capability, deliver a
    // real mouse, and still surprise us about which family of events it sends. These two
    // are what every browser has emitted for a moving mouse for twenty years. Both paths
    // funnel into the same scheduleShow/scheduleHide, which are idempotent (they clear
    // their own timers), so a browser sending BOTH opens exactly one popup.
    cell.addEventListener('mouseenter', onEnter);
    cell.addEventListener('mouseleave', onLeave);
    cell.addEventListener('focusin', function () {
      placeCaseHint(cell);
      if (!clickOnly && hoverAllowed()) {
        restoreTooltip();
        showFocusedTooltip();
        scheduleShow();
      }
    });
    const focusWithinTooltip = function (target) {
      return target && (cell.contains(target) || ttip.contains(target));
    };
    const onFocusOut = function (event) {
      if (focusWithinTooltip(event.relatedTarget)) { return; }
      if (!ttip.classList.contains('ttip-pinned')) { scheduleHide(); }
      cell._focusTooltip = false;
    };
    cell.addEventListener('focusout', onFocusOut);
    // Portal placement changes ancestry, so the popup owns its own focus events.
    ttip.addEventListener('focusin', keepTooltipOpen);
    ttip.addEventListener('focusout', onFocusOut);
    ttip.addEventListener('pointerenter', keepTooltipOpen);
    ttip.addEventListener('mouseenter', keepTooltipOpen);
    ttip.addEventListener('pointerleave', onLeave);
    ttip.addEventListener('mouseleave', onLeave);

  }
  window.__attachConformanceTooltip = attachTooltip;
  // The elements present at load. `th.case-sub` carries the per-column grammar popup (the
  // same case in every family's grammar); legacy transpose builds its headers later and
  // wires them itself.
  const WIRE_ON_LOAD = 'td.cell, td.parser, th.case-sub';
  document.querySelectorAll(WIRE_ON_LOAD).forEach(attachTooltip);

  function restorePinnedTooltip() {
    const stateKey = new URLSearchParams(window.location.search).get('popup');
    if (!stateKey) return;
    const panel = activePanel();
    const transposed = document.body.classList.contains('transpose-mode');
    const cell = panel && Array.from(panel.querySelectorAll('[data-ttip-key][data-ttip-click="pin"]'))
      .find(function (candidate) {
        return candidate.dataset.ttipKey === stateKey
          && Boolean(candidate.closest('table[data-transpose-table]')) === transposed;
      });
    if (cell && cell._ttipPin) { cell._ttipPin(); }
    else if (!(panel && panel.id === 'tab-unified' && panel.dataset.loadState !== 'ready')) {
      const url = new URL(window.location.href);
      url.searchParams.delete('popup');
      window.history.replaceState(null, '', url.toString());
    }
  }

  activateTab(readActiveTab(), false);
  tabButtons.forEach(function (button) {
    button.addEventListener('click', function () {
      activateTab(button.dataset.tabTarget, true);
      hooks.tabChanged?.(button.dataset.tabTarget);
    });
  });

  // ---- Transpose view (DIS-2280) ----
  // Build a transposed mirror of each panel's table on demand: models become
  // columns (rotated CCW headers), test cases become rows. Data cells are cloned
  // from the server-rendered table, so they keep their data-cmp payload and the
  // compare/coloring engine (applyCtl) recolors them for free — the mirror lives
  // in the same panel, so applyCtl's cell query already covers it.
  const transposeToggle = document.querySelector('[data-transpose-toggle]');

  function readTransposeMode() {
    const params = new URLSearchParams(window.location.search);
    const requested = params.get('vertical') || params.get('transpose');
    return requested === '1' || requested === 'true';
  }

  function updateTransposeUrl(enabled) {
    const url = new URL(window.location.href);
    url.searchParams.delete('transpose');
    url.searchParams.delete('vertical');
    if (enabled) {
      url.searchParams.set('transpose', '1');
    }
    window.history.replaceState(null, '', url.toString());
  }

  function cloneTooltip(source) {
    if (!source) return null;
    const clone = source.cloneNode(true);
    clone.classList.remove('ttip-visible', 'ttip-pinned');
    clone.removeAttribute('data-ttip-wired');
    clone.querySelectorAll('.ttip-close').forEach(button => { button.removeAttribute('data-ttip-close-wired'); });
    return clone;
  }

  function buildTransposed(table) {
    const head = table.tHead;
    const headRows = head ? Array.from(head.rows) : [];
    const body = table.tBodies[0];
    if (headRows.length < 2 || !body) {
      return null;
    }

    // Case-group key -> human label (from the group-header toggle buttons).
    const groupLabel = {};
    Array.from(headRows[0].querySelectorAll('th.case-group')).forEach(function (th) {
      const btn = th.querySelector('[data-col-label]');
      groupLabel[th.dataset.colControlGroup] = btn ? btn.dataset.colLabel : th.textContent.trim();
    });
    // Ordered sub-cases (the new rows). Each carries its case-group key.
    const subThs = Array.from(headRows[1].querySelectorAll('th.case-sub'));

    // Models (the new columns), grouped by the body's section banners.
    const models = [];
    let curSection = null;
    Array.from(body.rows).forEach(function (row) {
      if (row.classList.contains('section')) {
        curSection = row.textContent.trim();
        return;
      }
      const modelTd = row.querySelector('td.model');
      if (!modelTd) return;
      models.push({
        name: modelTd.textContent.trim(),
        popupKey: modelTd.dataset.popupKey,
        section: curSection,
        parserTd: row.querySelector('td.parser'),
        cells: Array.from(row.querySelectorAll('td.cell'))
      });
    });
    if (!models.length) {
      return null;
    }
    const nModels = models.length;

    const out = document.createElement('table');
    out.className = 'transpose-table';
    out.setAttribute('data-transpose-table', '');

    const outHead = out.createTHead();

    // Optional model-section banner row (e.g. "Top-N models" / "Others").
    const sections = [];
    models.forEach(function (m) {
      const last = sections[sections.length - 1];
      if (last && last.label === m.section) {
        last.count += 1;
      } else {
        sections.push({label: m.section, count: 1});
      }
    });
    if (sections.some(function (s) { return s.label; })) {
      const r0 = outHead.insertRow();
      r0.className = 'transpose-sections';
      const corner = document.createElement('th');
      corner.className = 'tcorner-case';
      r0.appendChild(corner);
      sections.forEach(function (s) {
        const th = document.createElement('th');
        th.className = 'tsection-col';
        th.colSpan = s.count;
        th.textContent = s.label || '';
        r0.appendChild(th);
      });
    }

    // Rotated model-name header row. The corner cell labels the case axis with
    // the panel's case prefix (e.g. "Case TOOLCALLING.batch.*"), rotated the same
    // way as the model names.
    const r1 = outHead.insertRow();
    r1.className = 'transpose-models';
    const cornerCase = document.createElement('th');
    cornerCase.className = 'tcol-model tcorner-case';
    let prefix = (table.dataset.casePrefix || '').trim();
    const mode = (table.dataset.mode || '').trim();
    // Tool-calling prefixes already carry the case namespace ("TOOLCALLING.batch.");
    // reasoning's is family-only ("REASONING."), where the namespace is the mode.
    // Splice the mode in only for that family-only case.
    if (mode && prefix.split('.').filter(Boolean).length === 1) {
      prefix = prefix + mode + '.';
    }
    const caseLabel = document.createElement('span');
    caseLabel.className = 'tcol-model-label';
    const caseLabelLine = document.createElement('span');
    caseLabelLine.className = 'tcol-model-line';
    caseLabelLine.textContent = prefix ? ('Case ' + prefix + '*') : 'Case';
    caseLabel.appendChild(caseLabelLine);
    const cornerInner = document.createElement('div');
    cornerInner.className = 'tcol-model-inner';
    cornerInner.appendChild(caseLabel);
    cornerCase.appendChild(cornerInner);
    r1.appendChild(cornerCase);
    models.forEach(function (m) {
      const th = document.createElement('th');
      th.className = 'tcol-model';
      th.dataset.popupKey = m.popupKey;
      const label = document.createElement('span');
      label.className = 'tcol-model-label';
      const nameSpan = document.createElement('span');
      nameSpan.className = 'tcol-model-name';
      nameSpan.textContent = m.name;
      label.appendChild(nameSpan);
      // Second line: the tool-calling family, lifted from the original parser
      // cell (without its tooltip). Clone the markup (not just text) so the
      // parser-source link stays clickable, exactly like the non-transposed view.
      if (m.parserTd) {
        const famClone = m.parserTd.cloneNode(true);
        const famTip = famClone.querySelector('.ttip');
        if (famTip) famTip.remove();
        if (famClone.textContent.trim()) {
          const famSpan = document.createElement('span');
          famSpan.className = 'tcol-model-family';
          while (famClone.firstChild) {
            famSpan.appendChild(famClone.firstChild);
          }
          label.appendChild(famSpan);
        }
      }
      const inner = document.createElement('div');
      inner.className = 'tcol-model-inner';
      inner.appendChild(label);
      th.appendChild(inner);
      // Same hover tooltip as the original parser cell.
      const srcTip = m.parserTd && (m.parserTd._ttip || m.parserTd.querySelector('.ttip'));
      if (srcTip) {
        const tipClone = cloneTooltip(srcTip);
        th.appendChild(tipClone);
        attachTooltip(th);
      }
      r1.appendChild(th);
    });

    // One body row per sub-case, with a section banner when the group changes.
    const outBody = out.createTBody();
    let curGroup = null;
    subThs.forEach(function (subTh, idx) {
      const group = subTh.dataset.colHideGroup;
      if (group !== curGroup) {
        curGroup = group;
        const sr = outBody.insertRow();
        sr.className = 'section';
        if (group) { sr.setAttribute('data-col-hide-group', group); }
        const td = document.createElement('td');
        td.colSpan = 1 + nModels;
        td.textContent = groupLabel[group] || group || '';
        sr.appendChild(td);
      }
      const tr = outBody.insertRow();
      // Carry the case-group key so applyColumnState hides this row (and its section
      // banner) when that group is collapsed via the column toggles — keeps the
      // transposed view in sync with the original table's show/hide state.
      if (group) { tr.setAttribute('data-col-hide-group', group); }
      const caseTh = document.createElement('th');
      caseTh.className = 'trow-case';
      caseTh.dataset.popupKey = subTh.dataset.popupKey;
      const link = subTh.querySelector('a');
      caseTh.appendChild(link ? link.cloneNode(true) : document.createTextNode(subTh.textContent.trim()));
      // Carry the per-case grammar popup into the transposed view. Transposing turns the
      // case COLUMN header into a ROW header, and this branch previously cloned only the
      // link — so the popup silently disappeared whenever the table was transposed.
      // Copy the tooltip identity and URL key so this heading can build lazily and
      // restore the same pinned popup after sharing a transposed view.
      const caseTipId = subTh.getAttribute('data-ttip-id');
      if (caseTipId) {
        caseTh.setAttribute('data-ttip-id', caseTipId);
        const caseTipKey = subTh.getAttribute('data-ttip-key');
        if (caseTipKey) { caseTh.setAttribute('data-ttip-key', caseTipKey); }
        const caseTip = document.createElement('div');
        caseTip.className = 'ttip';
        caseTh.appendChild(caseTip);
        attachTooltip(caseTh);
      }
      tr.appendChild(caseTh);
      models.forEach(function (m) {
        const src = m.cells[idx];
        if (src) {
          const sourceTip = src._ttip || src.querySelector('.ttip');
          const clone = src.cloneNode(true);
          const cloneTip = cloneTooltip(sourceTip);
          const embeddedTip = clone.querySelector('.ttip');
          if (cloneTip) {
            if (embeddedTip) embeddedTip.remove();
            clone.appendChild(cloneTip);
          }
          clone.classList.remove('col-hidden');
          clone.removeAttribute('data-col-hide-group');
          // cloneNode copies the "already wired" flag; clear it so the clone
          // gets its own tooltip listeners.
          clone.removeAttribute('data-ttip-wired');
          delete clone.dataset.ttipWired;
          tr.appendChild(clone);
          attachTooltip(clone);
        } else {
          const td = document.createElement('td');
          td.className = 'cell na';
          tr.appendChild(td);
        }
      });
    });

    return out;
  }

  function ensureTransposed(panel) {
    if (panel.querySelector('table[data-transpose-table]')) return;
    const orig = panel.querySelector('table[data-parity-table]');
    if (!orig) return;
    const transposed = buildTransposed(orig);
    if (transposed) {
      orig.insertAdjacentElement('afterend', transposed);
      // Color the freshly-cloned cells with the panel's current Reference/Compare
      // selection (applyCtl covers the mirror since it lives in the panel).
      if (panelCtl(panel)) { applyCtl(panel); }
      // Apply the current column-collapse state so a case group hidden in the
      // original table stays hidden (as rows) in the freshly-built mirror.
      applyColumnState(visibleColumns, false);
      updateStickyOffsets();
    }
  }

  function applyTransposeMode(enabled, shouldUpdateUrl) {
    document.body.classList.toggle('transpose-mode', Boolean(enabled));
    if (transposeToggle) {
      transposeToggle.checked = Boolean(enabled);
    }
    if (enabled) {
      document.querySelectorAll('.tab-panel').forEach(ensureTransposed);
    }
    document.documentElement.classList.remove('transpose-pending');
    document.querySelectorAll('.tab-panel').forEach(window.__conformanceUpdateLegend);
    if (shouldUpdateUrl) {
      updateTransposeUrl(Boolean(enabled));
    }
  }

  applyTransposeMode(readTransposeMode(), false);
  restorePinnedTooltip();
  updateStickyOffsets();
  window.addEventListener('resize', updateStickyOffsets);
  if (transposeToggle) {
    transposeToggle.addEventListener('change', function () {
      applyTransposeMode(transposeToggle.checked, true);
    });
  }
  function disposePanel(panel) {
    const selected = pinnedCell?.closest('.tab-panel') === panel ? {
      popupKey: pinnedCell.dataset.popupKey,
      transposed: Boolean(pinnedCell.closest('[data-transpose-table]')),
      width: pinnedCell._ttip.style.width, height: pinnedCell._ttip.style.height,
      scrollTop: pinnedCell._ttip.scrollTop, scrollLeft: pinnedCell._ttip.scrollLeft,
    } : null;
    panel.querySelectorAll('[data-ttip-wired]').forEach(cell => cell._ttipDispose?.());
    return selected;
  }
  function refreshPanel(panel) { applyCtl(panel); }
  function updatePanel(panel, pin) {
    columnButtons = Array.from(document.querySelectorAll('[data-col-toggle]'));
    columnKeys = [...new Set([...(hooks.columnKeys || []), ...columnButtons.map(button => button.dataset.colToggle)])];
    visibleColumns = readVisibleColumns();
    panel.querySelectorAll(WIRE_ON_LOAD).forEach(attachTooltip);
    applyColumnState(visibleColumns, false);
    if (document.body.classList.contains('transpose-mode')) ensureTransposed(panel);
    applyCtl(panel);
    if (pin?.popupKey) {
      const table = panel.querySelector(pin.transposed ? '[data-transpose-table]' : '[data-parity-table]');
      const cell = [...(table?.querySelectorAll('[data-popup-key]') || [])].find(cell => cell.dataset.popupKey === pin.popupKey);
      if (cell) {
        window.__buildTooltip(cell);
        cell._ttip.style.width = pin.width; cell._ttip.style.height = pin.height;
        cell._ttipPin?.();
        cell._ttip.scrollTop = pin.scrollTop;
        cell._ttip.scrollLeft = pin.scrollLeft;
      }
    }
    restorePinnedTooltip();
    updateStickyOffsets();
  }
  return { refreshPanel, updatePanel, disposePanel, activePanel, refreshPopupLayout: repositionVisibleTooltips,
    selection(panel) {
      const ctl = panelCtl(panel), reference = ctlBase(ctl);
      return hooks.engineSelection ? hooks.engineSelection.selection() : { reference, selected: reference ? [reference, ...ctlShown(ctl).filter(key => key !== reference)] : [] };
    },
  };
};

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
window.mountConformanceCaptureSlots = function (page, panel, onChange = () => {}, { availableIdentities = null } = {}) {
  const tab = page && page.tabs && page.tabs.find(item => item.id === 'tab-unified');
  if (!tab || !panel) return null;

  const engines = tab.comparison_slot_engines || ['dynamo', 'sglang', 'vllm'];
  const engineNames = { dynamo: 'Dynamo', sglang: 'SGLang', vllm: 'vLLM' };
  const engineLetters = { dynamo: 'D', sglang: 'S', vllm: 'V' };
  const candidates = [];
  const byKey = new Map();
  const byIdentity = new Map();
  let selectionTouched = false;
  const engineName = engine => engine ? engineNames[engine] : 'None';
  const engineOf = item => {
    const impl = String(item.impl || item.key).toLowerCase();
    return impl.startsWith('dynamo') ? 'dynamo' : impl.startsWith('sglang') ? 'sglang' : 'vllm';
  };
  // Catalog descriptors define availability before any selected history is loaded.
  const available = [];
  const languageOrder = tab.comparison_languages || [];
  const languageRank = item => {
    const rank = languageOrder.indexOf(item.language);
    return rank < 0 ? languageOrder.length : rank;
  };
  const versionParts = item => String(item.version || '').match(/\d+|[A-Za-z]+/g) || [];
  const compareVersions = (left, right) => {
    const a = versionParts(left);
    const b = versionParts(right);
    for (let index = 0; index < Math.max(a.length, b.length); index += 1) {
      if (a[index] === undefined) return 1;
      if (b[index] === undefined) return -1;
      const leftNumber = /^\d+$/.test(a[index]);
      const rightNumber = /^\d+$/.test(b[index]);
      if (leftNumber && rightNumber) {
        const difference = Number(b[index]) - Number(a[index]);
        if (difference) return difference;
      } else if (a[index] !== b[index]) {
        return a[index] < b[index] ? 1 : -1;
      }
    }
    return 0;
  };
  const compareCaptures = (left, right, engine) => {
    const languageDifference = languageRank(left) - languageRank(right);
    if (languageDifference) return languageDifference;
    if (engine === 'dynamo') {
      const leftGeneration = /^dynamo_v1(?:_|@)/.test(left.key) ? 1 : 2;
      const rightGeneration = /^dynamo_v1(?:_|@)/.test(right.key) ? 1 : 2;
      if (leftGeneration !== rightGeneration) return rightGeneration - leftGeneration;
    }
    return compareVersions(left, right)
      || (left.key < right.key ? -1 : left.key > right.key ? 1 : 0);
  };
  const byEngine = new Map();
  const setCandidates = items => {
    candidates.splice(0, candidates.length, ...items.filter(item => item && item.key && item.impl !== 'golden'));
    byKey.clear(); byIdentity.clear();
    for (const item of candidates) {
      byKey.set(item.key, item);
      for (const identity of [item.key, item.source_id, item.identity].filter(Boolean)) byIdentity.set(identity, item.key);
    }
    available.splice(0, available.length, ...candidates.filter(item => {
      const families = item.capture_families || item.native_unified_families;
      return !item.source_blocker && item.has_captured_results !== false && (!families || families.length);
    }));
    for (const engine of engines) byEngine.set(engine, available.filter(item => engineOf(item) === engine)
      .sort((left, right) => compareCaptures(left, right, engine)));
  };
  setCandidates(tab.candidates || []);
  const isDynamoV2Choice = item => Boolean(item)
    && (item.identity?.startsWith('dynamo_v2@') || item.key.startsWith('dynamo@'));
  const captureChoices = engine => {
    const choices = (byEngine.get(engine) || []).filter(item => !availableIdentities || availableIdentities.has(item.identity));
    if (engine !== 'dynamo') return choices;
    const versioned = choices.filter(item => /^dynamo_v1(?:_|@)/.test(item.key) || isDynamoV2Choice(item));
    const latestByLine = new Map();
    const recentByProducer = new Map();
    for (const item of [...versioned].sort(compareVersions)) {
      const isV2 = isDynamoV2Choice(item);
      const producer = isV2 ? 'dynamo_v2' : item.key.split('@')[0];
      const recent = recentByProducer.get(producer) || new Set();
      if (recent.size < 2) recent.add(item.version);
      recentByProducer.set(producer, recent);
      const parts = versionParts(item);
      const line = isV2 ? parts.slice(0, 2).join('.') : parts[0];
      const key = producer + ':' + line;
      if (!latestByLine.has(key)) latestByLine.set(key, item.version);
    }
    return choices.filter(item => {
      const producer = item.key.split('@')[0];
      const isV2 = isDynamoV2Choice(item);
      if (!/^dynamo_v1(?:_|@)/.test(item.key) && !isV2) return true;
      const parts = versionParts(item);
      const line = isV2 ? parts.slice(0, 2).join('.') : parts[0];
      const identity = isV2 ? 'dynamo_v2' : producer;
      return recentByProducer.get(identity)?.has(item.version) || latestByLine.get(identity + ':' + line) === item.version;
    });
  };
  const preferredByEngine = engines.map(engine => {
    return captureChoices(engine)[0]?.key || null;
  });
  const preferred = [preferredByEngine[0], null, null];
  const params = new URLSearchParams(window.__CONFORMANCE_INITIAL_SEARCH || window.location.search);
  const slotsParam = 'slots_' + tab.id;
  const initialTab = params.get('tab');
  const legacyCandidates = new Map(Object.entries(page.legacy_candidate_identities || {}));
  for (const item of page.tabs.find(item => item.id === initialTab)?.candidates || []) {
    if (item.key && item.identity) legacyCandidates.set(item.key, item.identity);
  }
  const resolveSlotKey = key => {
    if (!key || key === 'golden') return '';
    if (byKey.has(key)) return key;
    const identity = legacyCandidates.get(key) || key;
    return byIdentity.get(identity) || byIdentity.get(key) || '';
  };
  const legacyTabs = [...new Set([initialTab, tab.id].filter(Boolean))];
  const legacyBaseParam = legacyTabs.map(id => 'base_' + id).find(key => params.has(key));
  const legacyCompareParam = legacyTabs.map(id => 'cmp_' + id).find(key => params.has(key));
  const legacyRef = (legacyBaseParam && params.get(legacyBaseParam)) ?? params.get('ref');
  const legacyCmp = (legacyCompareParam && params.get(legacyCompareParam)) ?? params.get('cmp');
  const hasLegacy = Boolean(legacyBaseParam || legacyCompareParam || params.has('ref') || params.has('cmp'));
  const legacySelection = hasLegacy
    ? [(legacyBaseParam || params.has('ref')) ? legacyRef : preferred[0], ...(legacyCmp || '').split(',')]
      .map(resolveSlotKey)
    : null;
  // Stream-tab links encode engine columns; other legacy comparisons preserve their authored order.
  const legacy = initialTab === 'tab-toolcalling-streamv1' && legacySelection
    ? legacySelection.reduce((slots, key) => {
      if (!key) return slots;
      const engineIndex = engines.indexOf(engineOf(byKey.get(key)));
      const target = engineIndex >= 0 && !slots[engineIndex] ? engineIndex : slots.findIndex(slot => !slot);
      if (target >= 0) slots[target] = key;
      return slots;
    }, [null, null, null])
    : legacySelection;
  const requested = params.has(slotsParam) ? (params.get(slotsParam) || '').split(',').map(key => key && key !== 'none' ? resolveSlotKey(key) : '') : legacy;
  const initial = requested === null ? preferred : requested;
  const overflow = initial.filter(Boolean).length > 3;
  const slots = [null, null, null];
  const slotEngines = [...engines];
  // Restoration and updates share primary ownership and duplicate prevention.
  function normalizeSlots() {
    const seen = new Set();
    slots.forEach((key, index) => {
      let current = byKey.get(key);
      const currentIsV2 = isDynamoV2Choice(current);
      if (current && (/^dynamo_v1(?:_|@)/.test(current.key) || currentIsV2)
        && !captureChoices('dynamo').includes(current)) {
        const producer = currentIsV2 ? 'dynamo_v2' : current.key.split('@')[0];
        const parts = versionParts(current);
        const line = currentIsV2 ? parts.slice(0, 2).join('.') : parts[0];
        const latest = captureChoices('dynamo').find(item => (currentIsV2 ? isDynamoV2Choice(item) : item.key.startsWith(producer + '@'))
          && (currentIsV2
            ? versionParts(item).slice(0, 2).join('.') === line
            : versionParts(item)[0] === line));
        if (latest) slots[index] = key = latest.key;
        else {
          slots[index] = key = '';
          slotEngines[index] = engineOf(current);
        }
        current = byKey.get(key);
      }
      if (!current || !captureChoices(engineOf(current)).includes(current) || seen.has(key)) {
        if (index === 0) {
          slots[index] = preferred[0];
          current = byKey.get(preferred[0]);
        } else {
          slots[index] = null;
          slotEngines[index] = null;
          if (current) slotEngines[index] = engineOf(current);
          return;
        }
      }
      seen.add(slots[index]);
      slotEngines[index] = engineOf(current);
    });
  }
  initial.slice(0, 3).forEach((key, index) => {
    slots[index] = key || null;
  });
  normalizeSlots();

  const selector = document.createElement('section');
  selector.className = 'capture-slots';
  selector.setAttribute('aria-label', 'Parser captures compared with GOLDEN');
  if (overflow) {
    const notice = document.createElement('p');
    notice.className = 'capture-slots-notice';
    notice.textContent = 'This link selected more than three captures. Showing the first three.';
    selector.appendChild(notice);
  }
  const controls = document.createElement('div');
  controls.className = 'capture-slot-controls';
  const escapeHtml = value => String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  const escapeAttr = value => escapeHtml(value).replace(/"/g, '&quot;');
  const shortCaptureLabel = (capture, engine) => {
    const label = String(capture && (capture.label || capture.version || capture.key) || 'None');
    return label.startsWith(engineName(engine) + ' ') ? label.slice(engineName(engine).length + 1) : label;
  };
  const captureTitle = (capture, engine) => {
    const label = String(capture && (capture.label || capture.version || capture.key) || 'None');
    if (engine !== 'dynamo' || !capture || capture.parse_mode !== 'batch') return label;
    if (capture.source_blocker) return capture.source_blocker;
    return capture.key.startsWith('dynamo_v1_jail@')
      ? 'Unified v2 batch corpus; produced through the Dynamo v1 Rust jail-buffer producer.'
      : 'Unified v2 batch corpus; produced by the Dynamo v1 Rust parser using legacy batch APIs, without a jail buffer.';
  };
  function candidateOptions(engine, slot, selectedKey) {
    if (!engine) return '';
    const choices = captureChoices(engine);
    return choices.map(item => {
      const duplicate = slots.some((key, index) => index !== slot && key === item.key);
      const label = shortCaptureLabel(item, engine);
      return '<option value="' + item.key.replace(/&/g, '&amp;').replace(/"/g, '&quot;') + '"'
        + ' title="' + escapeAttr(captureTitle(item, engine)) + '"'
        + (selectedKey === item.key ? ' selected' : '') + (duplicate || item.source_blocker ? ' disabled' : '') + '>'
        + label.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        + (duplicate ? ' — selected in another slot' : '') + '</option>';
    }).join('');
  }
  controls.innerHTML = slots.map((_key, index) => {
    const engine = slotEngines[index];
    return '<div class="capture-slot" data-engine="' + (engine || 'none') + '">'
      + '<button type="button" class="capture-slot-engine-trigger" data-slot-open="' + index + '"'
      + ' aria-haspopup="dialog" aria-expanded="false" aria-controls="unified-capture-picker"'
      + ' aria-label="Edit comparison slot ' + (index + 1) + ': ' + engineName(engine) + '">'
      + '<span class="capture-slot-badge" data-slot-badge="' + index + '" aria-hidden="true">' + (engineLetters[engine] || '—') + '</span>'
      + '<span class="capture-slot-engine" data-slot-engine="' + index + '">' + engineName(engine) + '</span></button>'
      + '<div class="capture-slot-capture-control" data-slot-capture-control="' + index + '"></div></div>';
  }).join('');
  selector.appendChild(controls);
  const actions = document.querySelector('.tabs-right');
  if (actions) {
    actions.classList.add('capture-slot-actions');
    selector.appendChild(actions);
  }
  panel.insertBefore(selector, panel.querySelector('.cmpctl'));

  const state = { slots, engines: slotEngines };
  // Controls describe the requested captures. Only a completed model changes the
  // displayed selection, which also owns cell colors, legend and lazy popups.
  panel._comparisonSelection = { keys: [], reference: null };
  function persist() {
    const url = new URL(window.location.href);
    const current = state.slots.map(key => key ? (byKey.get(key)?.identity || key) : 'none').join(',');
    const isDefault = state.slots.every((key, index) => key === preferred[index]);
    if (isDefault) url.searchParams.delete(slotsParam);
    else url.searchParams.set(slotsParam, current);
    for (const key of [...url.searchParams.keys()]) {
      if (/^(?:base_|cmp_)/.test(key)) url.searchParams.delete(key);
    }
    url.searchParams.delete('ref');
    url.searchParams.delete('cmp');
    window.history.replaceState(null, '', url.toString());
  }
  function update(notify = true) {
    normalizeSlots();
    state.slots.forEach((key, index) => {
      const capture = byKey.get(key);
      const engine = state.engines[index];
      const slot = controls.querySelector('.capture-slot:nth-child(' + (index + 1) + ')');
      slot.dataset.engine = engine || 'none';
      slot.querySelector('[data-slot-engine]').textContent = engineName(engine);
      slot.querySelector('[data-slot-badge]').textContent = engineLetters[engine] || '—';
      const captureControl = slot.querySelector('.capture-slot-capture-control');
      if (engine) {
        let captureSelect = captureControl.querySelector('[data-slot-capture]');
        if (!captureSelect) {
          captureSelect = document.createElement('select');
          captureSelect.className = 'capture-slot-select';
          captureSelect.dataset.slotCapture = index;
          const selectedLabel = document.createElement('span');
          selectedLabel.className = 'capture-slot-selected-label';
          selectedLabel.setAttribute('aria-hidden', 'true');
          captureControl.replaceChildren(captureSelect, selectedLabel);
        }
        captureSelect.innerHTML = candidateOptions(engine, index, key);
        captureSelect.value = key || '';
        captureControl.querySelector('.capture-slot-selected-label').textContent = shortCaptureLabel(capture, engine);
        captureSelect.disabled = false;
        captureSelect.setAttribute('aria-label', 'Choose capture for comparison slot '
          + (index + 1) + ': ' + engineName(engine) + ', ' + shortCaptureLabel(capture, engine));
      } else {
        captureControl.replaceChildren();
      }
      slot.querySelector('[data-slot-open]').setAttribute('aria-label', 'Edit comparison slot ' + (index + 1) + ': ' + engineName(engine));
    });
    persist();
    if (notify) onChange();
  }
  update();

  const picker = document.createElement('dialog');
  picker.className = 'comparison-picker';
  picker.id = 'unified-capture-picker';
  picker.setAttribute('aria-label', 'Choose comparison engine or capture');
  document.body.appendChild(picker);
  let activeSlot = 0;
  let activeTrigger = null;
  let skipPickerFocusRestore = false;
  function positionPicker() {
    if (!picker.open || !activeTrigger) return;
    const anchor = activeTrigger.getBoundingClientRect();
    const rect = picker.getBoundingClientRect();
    const margin = 12;
    const gap = 8;
    let left = anchor.right + gap;
    if (left + rect.width > window.innerWidth - margin) left = anchor.left - rect.width - gap;
    picker.style.left = Math.max(margin, Math.min(left, window.innerWidth - rect.width - margin)) + 'px';
    picker.style.top = Math.max(margin, Math.min(anchor.top, window.innerHeight - rect.height - margin)) + 'px';
  }
  function focusPicker(selector) {
    if (picker.open) positionPicker();
    window.requestAnimationFrame(() => {
      positionPicker();
      picker.querySelector(selector)?.focus();
    });
  }
  function showEngineChoices() {
    const choices = activeSlot === 0 ? engines : [...engines, 'none'];
    picker.dataset.stage = 'engine';
    picker.innerHTML = '<h2>Comparison slot ' + (activeSlot + 1) + '</h2>'
      + '<div class="picker-choices" role="group" aria-label="Choose engine">'
      + choices.map(engine => {
        const selected = engine === (state.engines[activeSlot] || 'none');
        return '<button type="button" data-picker-engine-choice="' + engine + '"'
          + (selected ? ' aria-current="true"' : '') + '>'
          + engineName(engine === 'none' ? null : engine) + '</button>';
      }).join('') + '</div>';
    focusPicker('[data-picker-engine-choice]');
  }
  function chooseEngine(engine) {
    selectionTouched = true;
    const choices = captureChoices(engine).filter(item =>
      !state.slots.some((key, index) => index !== activeSlot && key === item.key));
    const current = byKey.get(state.slots[activeSlot]);
    const selected = current && engineOf(current) === engine && choices.some(item => item.key === current.key)
      ? current
      : choices.find(item => item.key === preferredByEngine[engines.indexOf(engine)]) || choices[0];
    if (!selected) return;
    state.engines[activeSlot] = engine;
    state.slots[activeSlot] = selected.key;
    update();
    const captureSelect = controls.querySelector('[data-slot-capture="' + activeSlot + '"]');
    skipPickerFocusRestore = true;
    picker.close();
    captureSelect.focus({ preventScroll: true });
    if (typeof captureSelect.showPicker === 'function') captureSelect.showPicker();
    else captureSelect.click();
  }
  function openPicker(slot, trigger) {
    activeSlot = slot;
    activeTrigger = trigger;
    picker.dataset.slot = slot;
    showEngineChoices();
    trigger.setAttribute('aria-expanded', 'true');
    picker.showModal();
    window.requestAnimationFrame(positionPicker);
  }
  controls.addEventListener('click', event => {
    const trigger = event.target.closest('[data-slot-open]');
    if (trigger) openPicker(Number(trigger.dataset.slotOpen), trigger);
  });
  controls.addEventListener('change', event => {
    if (!event.target.matches('[data-slot-capture]')) return;
    const slot = Number(event.target.dataset.slotCapture);
    const key = event.target.value || null;
    if (key && state.slots.some((value, index) => index !== slot && value === key)) return;
    selectionTouched = true;
    state.slots[slot] = key;
    update();
  });
  picker.addEventListener('click', event => {
    if (event.target === picker) picker.close();
    const engineButton = event.target.closest('[data-picker-engine-choice]');
    if (engineButton) {
      const engine = engineButton.dataset.pickerEngineChoice;
      if (engine === 'none') {
        selectionTouched = true;
        state.engines[activeSlot] = null;
        state.slots[activeSlot] = null;
        update();
        picker.close();
      } else chooseEngine(engine);
    }
  });
  picker.addEventListener('close', () => {
    if (activeTrigger && activeTrigger.isConnected) {
      activeTrigger.setAttribute('aria-expanded', 'false');
      if (!skipPickerFocusRestore) activeTrigger.focus({ preventScroll: true });
    }
    skipPickerFocusRestore = false;
  });
  window.addEventListener('resize', positionPicker);
  window.addEventListener('scroll', positionPicker, true);
  return {
    setCandidates,
    setAvailableIdentities(identities, { notify = true } = {}) {
      const previous = state.slots.join(',');
      availableIdentities = identities;
      // Discovery may finish after a user changes a slot. Preserve that choice;
      // only an untouched initial historical URL can restore its deferred slots.
      if (!selectionTouched) initial.slice(0, 3).forEach((key, index) => { state.slots[index] = key || null; });
      update(false);
      const changed = state.slots.join(',') !== previous;
      if (changed && notify) onChange();
      return changed;
    },
    selection() { return { reference: null, selected: state.slots.filter(Boolean) }; },
    alternatives() {
      const selected = state.slots.filter(Boolean);
      const activeEngine = state.engines.find(Boolean) || engines[0];
      const engineOrder = [activeEngine, ...engines.filter(engine => engine !== activeEngine)];
      return engineOrder.flatMap(engine => captureChoices(engine).filter(item => !selected.includes(item.key)).map(item => {
        const slot = state.slots.findIndex(key => key && engineOf(byKey.get(key)) === engine);
        const index = slot >= 0 ? slot : state.slots.findIndex(key => !key);
        if (index < 0) return null;
        const next = [...state.slots]; next[index] = item.key;
        return next.filter(Boolean);
      })).filter(Boolean);
    },
    dispose() { selector.remove(); picker.remove(); window.removeEventListener('resize', positionPicker); window.removeEventListener('scroll', positionPicker, true); },
  };
};
if (window.__CONFORMANCE_PAGE) {
  window.mountConformanceCaptureSlots(window.__CONFORMANCE_PAGE, document.getElementById('tab-unified'));
}
