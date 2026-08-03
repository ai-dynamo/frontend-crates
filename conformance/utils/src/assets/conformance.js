// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Conformance matrix behavior (audit B7). Inlined at render by generate_conformance_table.py.
(function () {
  // Gutter kept between a popup and the viewport edge (2em at the 16px root size).
  // The tooltip is width:max-content and the widest ones run to the full cap below, so
  // without this they sit flush against the screen edge and read as clipped.
  const margin = 32;
  const showDelayMs = 750;
  const hideDelayMs = 750;
  const columnButtons = Array.from(document.querySelectorAll('[data-col-toggle]'));
  const columnKeys = Array.from(new Set(columnButtons.map(function (button) {
    return button.dataset.colToggle;
  })));
  const viewCheckboxes = Array.from(document.querySelectorAll('[data-view-detailed]'));

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
      // Invariant: a Compare-with can never exist without a Reference. With no base
      // (the cleared state) every box is unchecked + locked, so the only possible
      // action is starring a new Reference. With a base, the ref's box is the locked
      // one and every other row is free to toggle.
      if (!base) { cb.checked = false; }
      cb.disabled = isRef || !base;
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
        // Stable re-sort: Reference first, then the rest in their declaration order
        // (data-cand-order). Using a stable key avoids the cumulative scramble that a
        // move-to-front produces when the Reference is toggled repeatedly.
        // Order: a pinned column (the golden oracle) is always leftmost, THEN the
        // selected Reference, THEN the rest in declaration order (alphabetical here).
        function _colRank(el) {
          if (el.getAttribute('data-cand-pin') === '1') { return 0; }
          if (el.getAttribute('data-cand') === base) { return 1; }
          return 2;
        }
        grid.querySelectorAll('tr').forEach(function (tr) {
          const cols = Array.prototype.slice.call(tr.querySelectorAll('[data-cand]'));
          if (!cols.length) { return; }
          cols.sort(function (a, b) {
            const ra = _colRank(a), rb = _colRank(b);
            if (ra !== rb) { return ra - rb; }
            return (parseInt(a.getAttribute('data-cand-order'), 10) || 0)
              - (parseInt(b.getAttribute('data-cand-order'), 10) || 0);
          });
          cols.forEach(function (el) { tr.appendChild(el); });
        });
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
      // A null base is a deliberate "no reference" state (the star was toggled off),
      // distinct from the default layout — record it as an empty sentinel so a reload
      // restores no-ref instead of falling back to the default reference.
      url.searchParams.set('base_' + pid, base || '');
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
      // ...but they must not be SILENT either. A compare candidate that has no data for
      // this case (an older capture taken before the case existed) is not agreement, and
      // folding it into "=" claims the two builds produced the same output when one of
      // them never ran it. Counted separately so the glyph can say "no data" instead.
      const naShown = shown.filter(function (k) {
        return k !== base && cmp[k] && cmp[k].na === 1;
      }).length;
      // NΔ counts shown COMPARE engines that diverge from GOLDEN (the fixed reference),
      // independent of which engine is starred. When there is no golden (other tabs), fall
      // back to the selected reference. `avail` excludes the base (the starred REF).
      const goldenSig = cmp.golden ? cmp.golden.sig : null;
      const refSig = (goldenSig != null ? goldenSig : bd.sig);
      const diffs = avail.filter(function (o) { return o.sig !== refSig; }).length;
      const leak = bd.leak === 1;
      const redOnDiff = cell.getAttribute('data-red-on-diff') === '1';
      // Unified tab (data-red-on-diff): GOLDEN is the fixed oracle, and the cell color
      // reflects ONLY the REF — the starred engine (`base`, default Dynamo) — measured
      // against golden. A red cell means the REF's own output diverges from golden. The
      // Compare-with engines still surface their divergence count as NΔ, but never color
      // the cell (a compare diverging is the compare's business, not the REF's). golden as
      // REF never diverges from itself, so star an engine to evaluate it.
      const refDiverges = redOnDiff && base !== 'golden' && goldenSig != null
        && bd && bd.na !== 1 && bd.sig !== goldenSig;
      const donly = avail.length === 0;
      const leaked = leak;  // the REF's own leak drives ↯, same as every other tab
      // Marker: ✗ when the REF diverges (the red signal); otherwise NΔ = how many Compare
      // engines diverge from golden (informational, stays green), "=" when all agree.
      // "=" is only honest when every shown candidate actually produced output. When a
      // selected compare candidate has no capture for this case, say so ("?") rather
      // than reporting agreement it cannot support. A real divergence still wins.
      const eqTxt = naShown > 0 ? '?' : '=';
      let txt;
      if (redOnDiff) {
        txt = refDiverges ? '✗' : (diffs > 0 ? String(diffs) + 'Δ' : eqTxt);
      } else {
        txt = donly ? '' : (diffs === 0 ? eqTxt : String(diffs) + 'Δ');
      }
      if (marker) { marker.textContent = (leaked ? '↯' : '') + txt; }
      const problem = redOnDiff ? refDiverges : leaked;
      cell.classList.add(problem ? 'cmp-leak' : 'cmp-eq');
      if (countThis) { if (problem) { counts.problem++; } else { counts.ok++; } }
      toggleCands(cell, active, base);
    });
    panel.querySelectorAll('[data-overview-count]').forEach(function (el) {
      const k = el.dataset.overviewCount;
      el.textContent = String(k === 'todo' ? 0 : (counts[k] || 0));
    });
    // Column grammar popups (header hover): output columns are one-per-candidate. golden
    // is PINNED (always shown, leftmost); the rest show only when active (Reference +
    // selected compares). Flag the Reference column and order golden -> REF -> the rest,
    // the same rule the cell-popup candidate grid uses.
    panel.querySelectorAll('table.ttip-grammar').forEach(function (tbl) {
      tbl.querySelectorAll('[data-cand]').forEach(function (el) {
        const key = el.getAttribute('data-cand');
        const pinned = el.getAttribute('data-cand-pin') === '1';
        el.classList.toggle('col-hidden', !(pinned || active.has(key)));
        el.classList.toggle('col-ref', key === base);
      });
      function _gRank(el) {
        if (el.getAttribute('data-cand-pin') === '1') { return 0; }
        if (el.getAttribute('data-cand') === base) { return 1; }
        return 2;
      }
      tbl.querySelectorAll('tr').forEach(function (tr) {
        const cols = Array.prototype.slice.call(tr.querySelectorAll('[data-cand]'));
        if (!cols.length) { return; }
        cols.sort(function (a, b) {
          const ra = _gRank(a), rb = _gRank(b);
          if (ra !== rb) { return ra - rb; }
          return (parseInt(a.getAttribute('data-cand-order'), 10) || 0)
            - (parseInt(b.getAttribute('data-cand-order'), 10) || 0);
        });
        cols.forEach(function (el) { tr.appendChild(el); });
      });
    });
    updateCompareUrl(panel, ctl);
  }
  function applyCompare() { const p = activePanel(); if (p) { applyCtl(p); } }
  function _boxFor(ctl, val) {
    return Array.prototype.find.call(
      ctl.querySelectorAll('input.cmp-on'), function (x) { return x.value === val; }) || null;
  }
  // Picking a new Reference (starring a row): the starred row is the reference, so its
  // Compare box shows pressed. The PREVIOUS reference stays in the comparison as a normal
  // checked Compare box, so moving the star to a new row keeps the old one visible
  // (you're now comparing new-ref vs old-ref). syncRefDisable re-enables the old ref's
  // box on the next applyCtl; leaving it checked here is what turns it into a compare.
  function handleRefChange(ctl) {
    const newBase = ctlBase(ctl);
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
            if (panel) { applyCtl(panel); }
          }, 0);
        });
      });
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
    if (shouldUpdateUrl) {
      updateViewUrl(detailed);
    }
  }

  applyView(readDetailed(), false);
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

  function place(cell) {
    const ttip = cell.querySelector('.ttip');
    if (!ttip) return;
    equalizeColumns(ttip);
    ttip.style.visibility = 'hidden';
    ttip.style.opacity = '0';
    ttip.classList.add('ttip-visible');
    ttip.style.left = '0px';
    ttip.style.top = '100%';
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

  // Touch devices have no hover, so the tooltip is opened by TAP and pinned open
  // (with an ✕ to close) rather than shown on pointerenter. Desktop keeps hover;
  // a click there also pins (handy for reading a big tooltip without holding the
  // mouse still). Only one tooltip is pinned at a time.
  const hoverCapable = window.matchMedia('(hover: hover)').matches;
  let pinnedCell = null;
  function unpinCell(c) { if (c && c._ttipUnpin) { c._ttipUnpin(); } }
  // "Inside a pinnable element" must mean EXACTLY the set `attachTooltip` wired, so ask
  // for the mark `attachTooltip` itself leaves rather than re-listing the classes. A class
  // list cannot express that set: the transpose headers are wired directly (see below) and
  // never pass through WIRE_ON_LOAD, so every hand-written list drifted from it. Headers
  // ended up pinnable but not counted as inside — a header click pinned it and the SAME
  // click bubbled here and unpinned it, so the popup could never stay up.
  const WIRED = '[data-ttip-wired]';
  document.addEventListener('click', function (e) {
    // A click anywhere outside a pinnable element (and not on a pinned tooltip) closes the pin.
    if (pinnedCell && !e.target.closest(WIRED)) { unpinCell(pinnedCell); }
  });

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

    // ✕ close button (shown only while pinned) — inserted once per tooltip.
    if (!ttip.querySelector('.ttip-close')) {
      const x = document.createElement('button');
      x.type = 'button';
      x.className = 'ttip-close';
      x.setAttribute('aria-label', 'Close');
      x.textContent = '✕';
      x.addEventListener('click', function (e) { e.stopPropagation(); unpin(); });
      ttip.insertBefore(x, ttip.firstChild);
    }

    function pin() {
      if (pinnedCell && pinnedCell !== cell) { unpinCell(pinnedCell); }
      clearTimers();
      place(cell);
      ttip.classList.add('ttip-visible', 'ttip-pinned');
      isVisible = true;
      isActive = true;
      pinnedCell = cell;
      // The pressed look must mark the cell whose popup is actually up. While a pin
      // is held, hover no longer opens anything, so hover must not press either —
      // `ttip-pin-mode` switches the press off for :hover and onto this cell alone.
      cell.classList.add('ttip-open');
      document.body.classList.add('ttip-pin-mode');
    }
    function unpin() {
      ttip.classList.remove('ttip-visible', 'ttip-pinned');
      isVisible = false;
      isActive = false;
      cell.classList.remove('ttip-open');
      if (pinnedCell === cell) {
        pinnedCell = null;
        document.body.classList.remove('ttip-pin-mode');
      }
    }
    cell._ttipUnpin = unpin;

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

    // Tap (or click) toggles a pinned tooltip. Taps INSIDE the tooltip (its links,
    // the ✕) behave normally. On touch there's no hover, so a tap on the cell must
    // open the tooltip instead of following the cell's own parser-source link —
    // preventDefault blocks that navigation. On desktop, a click on the cell link
    // still navigates; a click on the cell body pins.
    cell.addEventListener('click', function (e) {
      if (e.target.closest('.ttip')) { return; }
      if (!hoverCapable) {
        e.preventDefault();
      } else if (e.target.closest('a, button, input, label')) {
        return;
      }
      if (ttip.classList.contains('ttip-pinned')) { unpin(); } else { pin(); }
    });
    // Hover show/hide only where hover exists; on touch it just flickers. A pinned
    // tooltip ignores hover leave so it stays until ✕ / outside tap.
    //
    // Pinning is also MODAL: while any tooltip is pinned, hover on OTHER cells does
    // nothing. Clicking a cell is a deliberate "keep this one open so I can read it",
    // and hover popups firing over the top of it defeat that — you cannot move the
    // mouse toward the pinned popup without crossing cells that each try to open
    // their own. Hover resumes when the pin is released (✕, clicking the pinned cell
    // again, or clicking outside).
    const hoverAllowed = function () { return pinnedCell === null || pinnedCell === cell; };
    if (hoverCapable) {
      cell.addEventListener('pointerenter', function () {
        if (hoverAllowed() && !ttip.classList.contains('ttip-pinned')) { scheduleShow(); }
      });
      cell.addEventListener('pointerleave', function () {
        if (!ttip.classList.contains('ttip-pinned')) { scheduleHide(); }
      });
      cell.addEventListener('focusin', function () {
        if (hoverAllowed()) { scheduleShow(); }
      });
      cell.addEventListener('focusout', function () {
        if (!ttip.classList.contains('ttip-pinned')) { scheduleHide(); }
      });
    }
  }
  // The elements present at load. `th.case-sub` carries the per-column grammar popup (the
  // same case in every family's grammar); it uses the identical hover/pin machinery as a
  // data cell. Transpose builds its headers later and wires them itself — this list is only
  // the starting set, never the definition of "pinnable" (that is `WIRED` above).
  const WIRE_ON_LOAD = 'td.cell, td.parser, th.case-sub';
  document.querySelectorAll(WIRE_ON_LOAD).forEach(attachTooltip);

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
      // Carry the per-case grammar popup into the transposed view. Transposing turns the
      // case COLUMN header into a ROW header, and this branch previously cloned only the
      // link — so the popup silently disappeared whenever the table was transposed.
      // Copy the `data-ttip-id` (the model key) and attach an empty `.ttip`; it then
      // builds lazily exactly like the upright header.
      const caseTipId = subTh.getAttribute('data-ttip-id');
      if (caseTipId) {
        caseTh.setAttribute('data-ttip-id', caseTipId);
        const caseTip = document.createElement('div');
        caseTip.className = 'ttip';
        caseTh.appendChild(caseTip);
        attachTooltip(caseTh);
      }
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
