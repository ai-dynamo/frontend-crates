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
  // Compare is per-panel: each tab's .cmpctl has its own candidate chips + buckets.
  function activePanel() { return document.querySelector('.tab-panel.active'); }
  function panelCtl(panel) { return panel ? panel.querySelector('.cmpctl') : null; }
  function ctlBase(ctl) { const c = ctl && ctl.querySelector('.bucket-A .chip'); return c ? c.dataset.cand : null; }
  function ctlShown(ctl) {
    return ctl ? Array.from(ctl.querySelectorAll('.bucket-B .chip')).map(function (x) { return x.dataset.cand; }) : [];
  }
  function toggleCands(cell, active, base) {
    cell.querySelectorAll('.ttip .cand').forEach(function (sec) {
      const cls = Array.from(sec.classList).find(function (c) { return c.indexOf('cand-') === 0; });
      const key = cls ? cls.slice(5) : null;
      sec.classList.toggle('cand-on', key !== null && active.has(key));
      // Mark the Base reference's section so the tooltip flags which output the
      // others are being compared against.
      sec.classList.toggle('cand-base', key !== null && key === base);
    });
  }
  // Keep each bucket's chips in lexical order regardless of drag/restore order.
  function sortChips(ctl) {
    ctl.querySelectorAll('.chips').forEach(function (zone) {
      Array.from(zone.querySelectorAll('.chip'))
        .sort(function (a, b) { return a.textContent.trim().localeCompare(b.textContent.trim()); })
        .forEach(function (chip) { zone.appendChild(chip); });
    });
  }
  // Size Compare-with and Others proportionally to how many chips each holds, so the
  // fuller bucket gets more width. Base holds one chip and stays content-sized. The
  // grow factor is floored at 1 so an empty bucket keeps a usable, droppable width.
  function resizeBuckets(ctl) {
    const b = ctl.querySelector('.bucket-B');
    const c = ctl.querySelector('.bucket-C');
    const nB = ctl.querySelectorAll('.bucket-B .chip').length;
    const nC = ctl.querySelectorAll('.bucket-C .chip').length;
    if (b) { b.style.flexGrow = String(Math.max(1, nB)); }
    if (c) { c.style.flexGrow = String(Math.max(1, nC)); }
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
    const A = ctl.querySelector('.bucket-A .chips');
    const B = ctl.querySelector('.bucket-B .chips');
    const C = ctl.querySelector('.bucket-C .chips');
    ctl.querySelectorAll('.chip').forEach(function (chip) {
      const k = chip.dataset.cand;
      if (k === base) { A.appendChild(chip); }
      else if (inB.has(k)) { B.appendChild(chip); }
      else { C.appendChild(chip); }
    });
  }
  function updateSwap(ctl) {
    const btn = ctl.querySelector('.cmp-swap');
    if (!btn) { return; }
    btn.hidden = !(ctl.querySelectorAll('.bucket-A .chip').length === 1
      && ctl.querySelectorAll('.bucket-B .chip').length === 1);
  }
  function applyCtl(panel) {
    const ctl = panelCtl(panel);
    if (!ctl) { return; }
    sortChips(ctl);
    resizeBuckets(ctl);
    const base = ctlBase(ctl);
    const shown = ctlShown(ctl).filter(function (k) { return k !== base; });
    const active = new Set((base ? [base] : []).concat(shown));
    const counts = { ok: 0, problem: 0, na: 0 };
    panel.querySelectorAll('td.cell[data-cmp]').forEach(function (cell) {
      let cmp;
      try { cmp = JSON.parse(cell.getAttribute('data-cmp')); } catch (e) { return; }
      cell.classList.remove('cmp-eq', 'cmp-leak', 'cmp-na', 'cmp-nobase', 'cmp-donly');
      const marker = cell.querySelector('.cmp-marker .marker-text');
      const bd = base ? cmp[base] : null;
      if (!base) {
        cell.classList.add('cmp-nobase'); if (marker) { marker.textContent = ''; }
        counts.na++; toggleCands(cell, active, base); return;
      }
      if (!bd || bd.na === 1) {
        cell.classList.add('cmp-na'); if (marker) { marker.textContent = 'n/a'; }
        counts.na++; toggleCands(cell, active, base); return;
      }
      // Unavailable candidates never count toward the diff; still shown in tooltip.
      const avail = shown.map(function (k) { return cmp[k]; }).filter(function (o) { return o && o.na !== 1; });
      const diffs = avail.filter(function (o) { return o.sig !== bd.sig; }).length;
      const leak = bd.leak === 1;
      // No available peer to compare against = a subtle "·" (Base-only), rendered
      // neutral gray (cmp-donly) rather than green — there is nothing to conform to.
      const donly = avail.length === 0;
      const txt = donly ? '·' : (diffs === 0 ? '=' : String(diffs));
      if (marker) { marker.textContent = (leak ? '↯' : '') + txt; }
      // Color = leak only: red = Base leaks markup, green = clean. Count is the number.
      cell.classList.add(leak ? 'cmp-leak' : (donly ? 'cmp-donly' : 'cmp-eq'));
      if (leak) { counts.problem++; } else if (donly) { counts.na++; } else { counts.ok++; }
      toggleCands(cell, active, base);
    });
    panel.querySelectorAll('[data-overview-count]').forEach(function (el) {
      const k = el.dataset.overviewCount;
      el.textContent = String(k === 'todo' ? 0 : (counts[k] || 0));
    });
    updateSwap(ctl);
    updateCompareUrl(panel, ctl);
  }
  function applyCompare() { const p = activePanel(); if (p) { applyCtl(p); } }
  function swapChips(x, y) {
    const xp = x.parentNode, xn = x.nextSibling, yp = y.parentNode, yn = y.nextSibling;
    yp.insertBefore(x, yn === x ? xn : yn);
    xp.insertBefore(y, xn === y ? yn : xn);
  }
  function initCompareDnd() {
    let dragged = null;
    document.querySelectorAll('.cmpctl .chip').forEach(function (chip) {
      chip.addEventListener('dragstart', function (e) {
        dragged = chip; chip.classList.add('dragging');
        e.dataTransfer.effectAllowed = 'move'; e.dataTransfer.setData('text/plain', chip.dataset.cand || '');
      });
      chip.addEventListener('dragend', function () {
        chip.classList.remove('dragging'); dragged = null;
        document.querySelectorAll('.chip-over').forEach(function (c) { c.classList.remove('chip-over'); });
      });
      // Drop a chip directly onto another chip in the same control = swap them —
      // except in Others, which always just receives the chip (no swap).
      const inOthers = function () { return chip.closest('.bucket').dataset.bucket === 'C'; };
      chip.addEventListener('dragover', function (e) {
        if (!inOthers() && dragged && dragged !== chip && chip.closest('.cmpctl') === dragged.closest('.cmpctl')) {
          e.preventDefault(); e.stopPropagation(); chip.classList.add('chip-over');
        }
      });
      chip.addEventListener('dragleave', function () { chip.classList.remove('chip-over'); });
      chip.addEventListener('drop', function (e) {
        if (inOthers() || !dragged || dragged === chip || chip.closest('.cmpctl') !== dragged.closest('.cmpctl')) { return; }
        e.preventDefault(); e.stopPropagation(); chip.classList.remove('chip-over');
        swapChips(dragged, chip); applyCompare();
      });
    });
    document.querySelectorAll('.cmpctl .bucket').forEach(function (b) {
      const zone = b.querySelector('.chips');
      b.addEventListener('dragover', function (e) { e.preventDefault(); e.dataTransfer.dropEffect = 'move'; b.classList.add('drop-hover'); });
      b.addEventListener('dragleave', function () { b.classList.remove('drop-hover'); });
      b.addEventListener('drop', function (e) {
        e.preventDefault(); b.classList.remove('drop-hover');
        if (!dragged || dragged.closest('.cmpctl') !== b.closest('.cmpctl')) { return; }
        if (b.dataset.bucket === 'A') {
          const cur = b.querySelector('.chip');
          if (cur && cur !== dragged) { b.closest('.cmpctl').querySelector('.bucket-B .chips').appendChild(cur); }
        }
        zone.appendChild(dragged); applyCompare();
      });
    });
    document.querySelectorAll('.cmpctl .cmp-swap').forEach(function (btn) {
      btn.addEventListener('click', function () {
        const ctl = btn.closest('.cmpctl');
        const a = ctl.querySelector('.bucket-A .chip'), bch = ctl.querySelector('.bucket-B .chip');
        if (a && bch) { swapChips(a, bch); applyCompare(); }
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
    initCompareDnd();
    document.querySelectorAll('.tab-panel').forEach(function (panel) {
      if (panelCtl(panel)) { applyCtl(panel); }
    });
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

  document.querySelectorAll('td.cell, td.parser').forEach(function (cell) {
    const ttip = cell.querySelector('.ttip');
    if (!ttip) return;

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
  });
})();
