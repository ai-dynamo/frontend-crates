// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Conformance matrix behavior (audit B7). Inlined at render by generate_conformance_table.py.
(function () {
  const margin = 8;
  const showDelayMs = 750;
  const hideDelayMs = 750;
  const columnButtons = Array.from(document.querySelectorAll('[data-col-toggle]'));
  const columnKeys = Array.from(new Set(columnButtons.map(function (button) {
    return button.dataset.colToggle;
  })));
  const viewCheckboxes = Array.from(document.querySelectorAll('[data-view-detailed]'));
  const parserRadios = Array.from(document.querySelectorAll('[data-parser-toggle]'));
  const parserKeys = new Set(parserRadios.map(function (radio) {
    return radio.value;
  }));
  const legacyParserAliases = {
    dynamo_rust: 'dynamo',
    vllm_python: 'vllm',
    sglang_python: 'sglang'
  };
  const parityToggle = document.querySelector('[data-parity-toggle]');

  // Per-impl version radios (TC v1 tab). impl -> { slugs: [...], default: slug }.
  const versionRadios = Array.from(document.querySelectorAll('[data-version-toggle]'));
  const versionImpls = {};
  versionRadios.forEach(function (radio) {
    const impl = radio.dataset.versionImpl;
    if (!versionImpls[impl]) { versionImpls[impl] = { slugs: [], default: null }; }
    versionImpls[impl].slugs.push(radio.value);
    if (radio.checked) { versionImpls[impl].default = radio.value; }
  });
  const activeVersion = {};
  Object.keys(versionImpls).forEach(function (impl) {
    activeVersion[impl] = versionImpls[impl].default || versionImpls[impl].slugs[0];
  });

  // Compare-any-combination model (TC v1 batch tab): pick one Base and any number
  // of Compare candidates; each cell shows how many selected candidates differ
  // from Base ("=" if none), plus ↯ when Base leaks markup. Tooltip shows Base +
  // each selected candidate. All client-side from each cell's data-cmp payload.
  // Compare is per-panel: each tab's .cmpctl has its own Reference radios + Compare checkboxes.
  function activePanel() { return document.querySelector('.tab-panel.active'); }
  function panelCtl(panel) { return panel ? panel.querySelector('.cmpctl') : null; }
  // Reference = the single checked radio; Compare-with = every checked checkbox.
  function ctlBase(ctl) { const r = ctl && ctl.querySelector('input.cmp-ref:checked'); return r ? r.value : null; }
  function ctlShown(ctl) {
    return ctl ? Array.from(ctl.querySelectorAll('input.cmp-on:checked')).map(function (x) { return x.value; }) : [];
  }
  // The Reference parser can't compare to itself: disable (and flag) its own
  // Compare checkbox, re-enable every other row's.
  function syncRefDisable(ctl, base) {
    ctl.querySelectorAll('input.cmp-on').forEach(function (cb) {
      const isRef = cb.value === base;
      // The Reference is always part of the comparison (it's the base everything is
      // compared against), so its Compare box shows pressed + locked. It's filtered
      // out of the compare set in applyCtl, so it isn't double-counted.
      if (isRef) { cb.checked = true; }
      cb.disabled = isRef;
      const row = cb.closest('.cmprow');
      if (row) { row.classList.toggle('is-ref', isRef); }
    });
  }
  // Compare keys are `<impl>-s-<version>` / `<impl>-b-<version>` (or `<impl>-s`); the
  // per-chunk grid columns are keyed by bare impl. Strip the mode+version suffix.
  function implOf(key) { return key ? key.replace(/-[sb](-.*)?$/, '') : key; }
  function toggleCands(cell, active, base) {
    let baseSec = null;
    cell.querySelectorAll('.ttip .cand').forEach(function (sec) {
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
    const grid = cell.querySelector('.ttip-chunks');
    if (grid) {
      const cands = grid.querySelectorAll('[data-cand]');
      if (cands.length) {
        // Candidate-column grid: columns ARE the (impl, version) candidates. Show the
        // Reference + each checked compare-with, flag the Reference column, and move
        // it first (right after the input column) in every row.
        cands.forEach(function (el) {
          const key = el.getAttribute('data-cand');
          el.classList.toggle('col-hidden', !active.has(key));
          el.classList.toggle('col-ref', key === base);
        });
        if (base) {
          grid.querySelectorAll('tr').forEach(function (tr) {
            const el = tr.querySelector('[data-cand="' + base + '"]');
            const first = tr.querySelector('[data-cand]');
            if (el && first && first !== el) { tr.insertBefore(el, first); }
          });
        }
      } else {
        // Legacy impl-column grid: show columns whose engine is active.
        const activeImpls = new Set(Array.from(active).map(implOf));
        grid.querySelectorAll('[data-col-impl]').forEach(function (el) {
          el.classList.toggle('col-hidden', !activeImpls.has(el.getAttribute('data-col-impl')));
        });
      }
    }
  }
  // Parsers with limited family coverage (Dynamo v2): key -> {label, families}. Drives
  // the reference-aware "not implemented" reason. Empty on pages without such a parser.
  const PARSER_NI = window.__PARSER_NI || {};
  // Show/clear a JS-driven "why n/a" line at the top of a cell's tooltip. Built in JS
  // (not the server-rendered tooltip) so the reason can change with the Reference.
  function setWhy(cell, text) {
    let el = cell.querySelector('.cmp-why');
    if (!text) { if (el) { el.remove(); } return; }
    if (!el) {
      const tip = cell.querySelector('.ttip');
      if (!tip) { return; }
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
    const url = new URL(window.location.href);
    const pid = panel.id;
    const base = ctlBase(ctl), b = ctlShown(ctl);
    url.searchParams.delete('base_' + pid); url.searchParams.delete('cmp_' + pid);
    // Only record params when this panel differs from its default layout, so the
    // default state keeps a clean URL (and Reset lands on an empty query).
    if (!_sameLayout(pid, base, b)) {
      if (base) { url.searchParams.set('base_' + pid, base); }
      if (b.length) { url.searchParams.set('cmp_' + pid, b.join(',')); }
    }
    window.history.replaceState(null, '', url.toString());
  }
  function restoreCtlFromUrl(panel, ctl) {
    const params = new URLSearchParams(window.location.search);
    const pid = panel.id;
    if (!params.has('base_' + pid) && !params.has('cmp_' + pid)) { return; }  // keep defaults
    const base = params.get('base_' + pid);
    const inB = new Set((params.get('cmp_' + pid) || '').split(',').filter(Boolean));
    ctl.querySelectorAll('input.cmp-ref').forEach(function (r) { r.checked = (r.value === base); });
    ctl.querySelectorAll('input.cmp-on').forEach(function (cb) { cb.checked = inB.has(cb.value); });
  }
  function applyCtl(panel) {
    const ctl = panelCtl(panel);
    if (!ctl) { return; }
    const base = ctlBase(ctl);
    syncRefDisable(ctl, base);
    const shown = ctlShown(ctl).filter(function (k) { return k !== base; });
    const active = new Set((base ? [base] : []).concat(shown));
    const counts = { ok: 0, problem: 0, na: 0 };
    // If the selected Reference is a parser with limited family coverage (e.g. the
    // Dynamo v2 parser), ni holds {label, families}. A cell whose family is not in
    // that list is "not implemented" — a different reason than case-level "not
    // applicable", and it wins.
    const ni = base ? PARSER_NI[base] : null;
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
      cell.classList.remove('cmp-eq', 'cmp-leak', 'cmp-na', 'cmp-nobase', 'cmp-donly');
      const marker = cell.querySelector('.cmp-marker .marker-text');
      const fam = cell.getAttribute('data-family') || '';
      if (base && ni && ni.families.indexOf(fam) === -1) {
        cell.classList.add('cmp-na'); if (marker) { marker.textContent = 'x'; }
        setWhy(cell, ni.label + ' is not yet implemented for family “' + fam + '”');
        if (countThis) { counts.na++; } toggleCands(cell, active, base); return;
      }
      setWhy(cell, '');
      const bd = base ? cmp[base] : null;
      if (!base) {
        cell.classList.add('cmp-nobase'); if (marker) { marker.textContent = ''; }
        if (countThis) { counts.na++; } toggleCands(cell, active, base); return;
      }
      if (!bd || bd.na === 1) {
        cell.classList.add('cmp-na'); if (marker) { marker.textContent = 'n/a'; }
        if (countThis) { counts.na++; } toggleCands(cell, active, base); return;
      }
      // Unavailable candidates never count toward the diff; still shown in tooltip.
      const avail = shown.map(function (k) { return cmp[k]; }).filter(function (o) { return o && o.na !== 1; });
      const diffs = avail.filter(function (o) { return o.sig !== bd.sig; }).length;
      const leak = bd.leak === 1;
      // GREEN = the Reference output is clean (no leaked tool-call markup). That holds
      // whether or not any Compare is selected, so a lone Reference with 0 Compares is
      // green too. The marker count is the number of selected Compares that diverge;
      // with no comparable peer there is simply nothing to count (blank), not gray.
      const donly = avail.length === 0;
      // Δ suffix marks the number as a count of diverging Compare-with parsers,
      // e.g. "2Δ"; "=" (all agree) and a lone leak "↯" carry no count.
      const txt = donly ? '' : (diffs === 0 ? '=' : String(diffs) + 'Δ');
      if (marker) { marker.textContent = (leak ? '↯' : '') + txt; }
      // Color = leak only: red = Reference leaks markup, green = clean.
      cell.classList.add(leak ? 'cmp-leak' : 'cmp-eq');
      if (countThis) { if (leak) { counts.problem++; } else { counts.ok++; } }
      toggleCands(cell, active, base);
    });
    panel.querySelectorAll('[data-overview-count]').forEach(function (el) {
      const k = el.dataset.overviewCount;
      el.textContent = String(k === 'todo' ? 0 : (counts[k] || 0));
    });
    updateCompareUrl(panel, ctl);
  }
  function applyCompare() { const p = activePanel(); if (p) { applyCtl(p); } }
  function _boxFor(ctl, val) {
    return Array.prototype.find.call(
      ctl.querySelectorAll('input.cmp-on'), function (x) { return x.value === val; }) || null;
  }
  // Picking a new Reference (starring a row): the starred row is the reference, so its
  // Compare box shows pressed. If the row you just starred was the ONLY compare, the
  // previous reference becomes the compare (a 1-vs-1 swap); otherwise the previous
  // reference drops out of the comparison.
  function handleRefChange(ctl) {
    const newBase = ctlBase(ctl);
    const oldBase = ctl.dataset.prevBase || '';
    const checked = Array.prototype.map.call(
      ctl.querySelectorAll('input.cmp-on:checked'), function (x) { return x.value; });
    // Active compares before the switch = checked boxes minus the old ref's (its box is
    // checked only as the reference's pressed state, not as a real compare).
    const priorCompares = checked.filter(function (v) { return v !== oldBase; });
    const isSwap = priorCompares.length === 1 && priorCompares[0] === newBase;
    const newBox = _boxFor(ctl, newBase);
    if (newBox) { newBox.checked = true; }
    if (oldBase && oldBase !== newBase && !isSwap) {
      const oldBox = _boxFor(ctl, oldBase);
      if (oldBox) { oldBox.checked = false; }
    }
    ctl.dataset.prevBase = newBase || '';
  }
  // Re-color the panel whenever a Reference star or Compare checkbox toggles.
  function initCompareInputs() {
    document.querySelectorAll('.cmpctl').forEach(function (ctl) {
      ctl.dataset.prevBase = ctlBase(ctl) || '';
      ctl.addEventListener('change', function (e) {
        if (e.target.matches('input.cmp-ref')) { handleRefChange(ctl); }
        if (e.target.matches('input.cmp-ref, input.cmp-on')) {
          const panel = ctl.closest('.tab-panel');
          if (panel) { applyCtl(panel); }
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
      cmpDefaults[panel.id] = { base: ctlBase(ctl), shown: ctlShown(ctl).slice() };
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

  function readActiveParser() {
    const params = new URLSearchParams(window.location.search);
    const requested = params.get('parser') || params.get('perspective');
    return parserKeys.has(requested) ? requested : 'dynamo_rust';
  }

  function readParityMode() {
    const requested = new URLSearchParams(window.location.search).get('parity');
    return requested === '1' || requested === 'true';
  }

  function updateViewUrl(detailed) {
    const url = new URL(window.location.href);
    url.searchParams.delete('view');
    if (detailed) {
      url.searchParams.set('view', 'details');
    }
    window.history.replaceState(null, '', url.toString());
  }

  function updateParserUrl(parser) {
    const url = new URL(window.location.href);
    url.searchParams.delete('parser');
    url.searchParams.delete('perspective');
    if (parser !== 'dynamo_rust') {
      url.searchParams.set('parser', parser);
    }
    window.history.replaceState(null, '', url.toString());
  }

  function updateParityUrl(enabled) {
    const url = new URL(window.location.href);
    url.searchParams.delete('parity');
    if (enabled) {
      url.searchParams.set('parity', '1');
    }
    window.history.replaceState(null, '', url.toString());
  }

  function readActiveVersion(impl) {
    const info = versionImpls[impl];
    if (!info) { return null; }
    const requested = new URLSearchParams(window.location.search).get('ver-' + impl);
    if (requested && info.slugs.indexOf(requested) !== -1) { return requested; }
    return info.default || info.slugs[0];
  }

  function updateVersionUrl(impl, slug) {
    const info = versionImpls[impl];
    const url = new URL(window.location.href);
    url.searchParams.delete('ver-' + impl);
    if (info && slug !== info.default) {
      url.searchParams.set('ver-' + impl, slug);
    }
    window.history.replaceState(null, '', url.toString());
  }

  function activeParser() {
    const checked = parserRadios.find(function (radio) {
      return radio.checked;
    });
    return checked ? checked.value : 'dynamo_rust';
  }

  function updateOverviewStats() {
    const parser = activeParser();
    const slug = activeVersion[parser];
    document.querySelectorAll('.tab-panel').forEach(function (panel) {
      const versioned = panel.dataset.hasVersions === 'true';
      const counts = {ok: 0, problem: 0, todo: 0, na: 0};
      panel.querySelectorAll('td.cell').forEach(function (cell) {
        // Skip cloned cells in the transposed mirror; they'd double the tally.
        if (cell.closest('[data-transpose-table]')) { return; }
        const alias = legacyParserAliases[parser];
        // On a versioned tab, prefer the active version's status; fall back to the
        // pinned attr (and legacy alias) for cells without per-version data.
        const status = (versioned && slug && cell.getAttribute('data-status-' + parser + '-' + slug))
          || cell.getAttribute('data-status-' + parser)
          || (alias ? cell.getAttribute('data-status-' + alias) : null) || 'na';
        counts[status] = (counts[status] || 0) + 1;
      });
      panel.querySelectorAll('[data-overview-count]').forEach(function (el) {
        el.textContent = String(counts[el.dataset.overviewCount] || 0);
      });
    });
  }

  function applyVersion(impl, slug, shouldUpdateUrl) {
    const info = versionImpls[impl];
    if (!info) { return; }
    const active = info.slugs.indexOf(slug) !== -1 ? slug : (info.default || info.slugs[0]);
    activeVersion[impl] = active;
    info.slugs.forEach(function (s) {
      document.body.classList.toggle('verv-' + impl + '-' + s, s === active);
    });
    versionRadios.forEach(function (radio) {
      if (radio.dataset.versionImpl === impl) {
        radio.checked = radio.value === active;
      }
    });
    updateOverviewStats();
    if (shouldUpdateUrl) {
      updateVersionUrl(impl, active);
    }
  }

  function applyView(detailed, shouldUpdateUrl) {
    document.body.classList.toggle('view-overview', !detailed);
    document.body.classList.toggle('view-details', detailed);
    viewCheckboxes.forEach(function (cb) { cb.checked = detailed; });
    if (shouldUpdateUrl) {
      updateViewUrl(detailed);
    }
  }

  function applyParser(parser, shouldUpdateUrl) {
    const active = parserKeys.has(parser) ? parser : 'dynamo_rust';
    parserKeys.forEach(function (key) {
      document.body.classList.toggle('parser-' + key, active === key);
    });
    Object.keys(legacyParserAliases).forEach(function (key) {
      document.body.classList.toggle('parser-' + legacyParserAliases[key], active === key);
    });
    parserRadios.forEach(function (radio) {
      radio.checked = radio.value === active;
    });
    updateOverviewStats();
    if (shouldUpdateUrl) {
      updateParserUrl(active);
    }
  }

  function applyParityMode(enabled, shouldUpdateUrl) {
    document.body.classList.toggle('parity-mode', Boolean(enabled));
    if (parityToggle) {
      parityToggle.checked = Boolean(enabled);
    }
    if (shouldUpdateUrl) {
      updateParityUrl(Boolean(enabled));
    }
  }

  applyView(readDetailed(), false);
  applyParser(readActiveParser(), false);
  Object.keys(versionImpls).forEach(function (impl) {
    applyVersion(impl, readActiveVersion(impl), false);
  });
  applyParityMode(readParityMode(), false);
  versionRadios.forEach(function (radio) {
    radio.addEventListener('change', function () {
      if (radio.checked) {
        applyVersion(radio.dataset.versionImpl, radio.value, true);
      }
    });
  });
  initCompare();
  viewCheckboxes.forEach(function (cb) {
    cb.addEventListener('change', function () { applyView(cb.checked, true); });
  });
  // Reset: drop every URL param (compare/view state) and reload at defaults, but
  // stay on the current tab.
  document.querySelectorAll('[data-reset]').forEach(function (btn) {
    btn.addEventListener('click', function () {
      const tab = new URLSearchParams(window.location.search).get('tab');
      const u = new URL(window.location.href); u.search = ''; u.hash = '';
      if (tab) { u.searchParams.set('tab', tab); }
      window.location.href = u.href;  // reload at defaults, same tab
    });
  });
  parserRadios.forEach(function (radio) {
    radio.addEventListener('change', function () {
      if (radio.checked) {
        applyParser(radio.value, true);
      }
    });
  });
  if (parityToggle) {
    parityToggle.addEventListener('change', function () {
      applyParityMode(parityToggle.checked, true);
    });
  }

  function defaultVisibleColumns() {
    const visible = new Set();
    columnButtons.forEach(function (button) {
      if (button.dataset.defaultVisible === 'true') {
        visible.add(button.dataset.colToggle);
      }
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
      button.title = button.getAttribute('aria-label');
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
    if (shouldUpdateUrl) {
      updateUrl(visible);
    }
  }

  let visibleColumns = readVisibleColumns();
  applyColumnState(visibleColumns, false);
  if (new URLSearchParams(window.location.search).has('cols')) {
    updateUrl(visibleColumns);
  }
  columnButtons.forEach(function (button) {
    button.addEventListener('click', function () {
      const key = button.dataset.colToggle;
      visibleColumns = new Set(visibleColumns);
      if (visibleColumns.has(key)) {
        visibleColumns.delete(key);
      } else {
        visibleColumns.add(key);
      }
      applyColumnState(visibleColumns, true);
    });
  });

  const tabButtons = Array.from(document.querySelectorAll('.tab-button'));
  const tabPanels = Array.from(document.querySelectorAll('.tab-panel'));

  function activePanelParserOptions() {
    const activePanel = tabPanels.find(function (panel) {
      return panel.classList.contains('active');
    });
    const raw = activePanel ? activePanel.dataset.parserOptions || '' : '';
    const options = raw.split(',').filter(function (key) {
      return parserKeys.has(key);
    });
    return options.length > 0 ? options : Array.from(parserKeys);
  }

  function syncParserOptions(shouldUpdateUrl) {
    const allowed = new Set(activePanelParserOptions());
    parserRadios.forEach(function (radio) {
      const isAllowed = allowed.has(radio.value);
      radio.disabled = !isAllowed;
      const label = radio.closest('[data-parser-option]');
      if (label) {
        label.hidden = !isAllowed;
      }
    });
    const current = activeParser();
    if (!allowed.has(current)) {
      const fallback = allowed.has('dynamo_rust') ? 'dynamo_rust' : Array.from(allowed)[0];
      applyParser(fallback, shouldUpdateUrl);
    } else {
      updateOverviewStats();
    }
  }

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
  }
  activateTab(readActiveTab(), false);
  tabButtons.forEach(function (button) {
    button.addEventListener('click', function () {
      activateTab(button.dataset.tabTarget, true);
    });
  });

  function place(cell) {
    const ttip = cell.querySelector('.ttip');
    if (!ttip) return;
    ttip.style.visibility = 'hidden';
    ttip.style.opacity = '0';
    ttip.classList.add('ttip-visible');
    ttip.style.left = '0px';
    ttip.style.top = '100%';
    ttip.style.right = 'auto';
    ttip.style.bottom = 'auto';
    ttip.style.maxWidth = Math.round(window.innerWidth * 0.9) + 'px';
    const cellRect = cell.getBoundingClientRect();
    const tipRect = ttip.getBoundingClientRect();
    const vw = window.innerWidth, vh = window.innerHeight;
    let shiftX = 0;
    const overflowRight = (cellRect.left + tipRect.width) - (vw - margin);
    if (overflowRight > 0) shiftX = -overflowRight;
    const absLeft = cellRect.left + shiftX;
    if (absLeft < margin) shiftX += (margin - absLeft);
    ttip.style.left = shiftX + 'px';
    if (cellRect.bottom + tipRect.height > vh - margin
        && cellRect.top - tipRect.height > margin) {
      ttip.style.top = 'auto';
      ttip.style.bottom = '100%';
    }
    ttip.style.visibility = '';
    ttip.style.opacity = '';
  }

  function attachTooltip(cell) {
    const ttip = cell.querySelector('.ttip');
    if (!ttip) return;
    // cloneNode copies the wired flag; guard so re-wiring a clone is a no-op and
    // originals aren't double-wired.
    if (cell.dataset.ttipWired === '1') return;
    cell.dataset.ttipWired = '1';

    let showTimer = null;
    let hideTimer = null;
    let isActive = false;
    let isVisible = false;

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
        if (!isActive) return;
        place(cell);
        ttip.classList.add('ttip-visible');
        isVisible = true;
      }, showDelayMs);
    }

    function scheduleHide() {
      isActive = false;
      if (showTimer !== null) {
        window.clearTimeout(showTimer);
        showTimer = null;
      }
      if (!isVisible) {
        ttip.classList.remove('ttip-visible');
        return;
      }
      if (hideTimer !== null) {
        window.clearTimeout(hideTimer);
      }
      hideTimer = window.setTimeout(function () {
        hideTimer = null;
        if (isActive) return;
        ttip.classList.remove('ttip-visible');
        isVisible = false;
      }, hideDelayMs);
    }

    cell.addEventListener('pointerenter', scheduleShow);
    cell.addEventListener('pointerleave', scheduleHide);
    cell.addEventListener('focusin', scheduleShow);
    cell.addEventListener('focusout', scheduleHide);
  }
  document.querySelectorAll('td.cell, td.parser').forEach(attachTooltip);

  // ---- Transpose view (DIS-2280) ----
  // Build a transposed mirror of each panel's table on demand: models become
  // columns (rotated CCW headers), test cases become rows. Data cells are cloned
  // from the server-rendered table, so they keep their data-cmp payload and the
  // compare/coloring engine (applyCtl) recolors them for free — the mirror lives
  // in the same panel, so applyCtl's cell query already covers it.
  const transposeToggle = document.querySelector('[data-transpose-toggle]');

  function readTransposeMode() {
    const requested = new URLSearchParams(window.location.search).get('transpose');
    return requested === '1' || requested === 'true';
  }

  function updateTransposeUrl(enabled) {
    const url = new URL(window.location.href);
    url.searchParams.delete('transpose');
    if (enabled) {
      url.searchParams.set('transpose', '1');
    }
    window.history.replaceState(null, '', url.toString());
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
      const srcTip = m.parserTd && m.parserTd.querySelector('.ttip');
      if (srcTip) {
        const tipClone = srcTip.cloneNode(true);
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
      const link = subTh.querySelector('a');
      caseTh.appendChild(link ? link.cloneNode(true) : document.createTextNode(subTh.textContent.trim()));
      tr.appendChild(caseTh);
      models.forEach(function (m) {
        const src = m.cells[idx];
        if (src) {
          const clone = src.cloneNode(true);
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
    if (shouldUpdateUrl) {
      updateTransposeUrl(Boolean(enabled));
    }
  }

  applyTransposeMode(readTransposeMode(), false);
  if (transposeToggle) {
    transposeToggle.addEventListener('change', function () {
      applyTransposeMode(transposeToggle.checked, true);
    });
  }
})();
