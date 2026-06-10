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
  const viewRadios = Array.from(document.querySelectorAll('[data-view-toggle]'));
  const viewKeys = new Set(viewRadios.map(function (radio) {
    return radio.value;
  }));
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

  function readActiveView() {
    const requested = new URLSearchParams(window.location.search).get('view');
    return viewKeys.has(requested) ? requested : 'overview';
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

  function updateViewUrl(view) {
    const url = new URL(window.location.href);
    url.searchParams.delete('view');
    if (view !== 'overview') {
      url.searchParams.set('view', view);
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

  function activeParser() {
    const checked = parserRadios.find(function (radio) {
      return radio.checked;
    });
    return checked ? checked.value : 'dynamo_rust';
  }

  function updateOverviewStats() {
    const parser = activeParser();
    document.querySelectorAll('.tab-panel').forEach(function (panel) {
      const counts = {ok: 0, problem: 0, todo: 0, na: 0};
      panel.querySelectorAll('td.cell').forEach(function (cell) {
        const alias = legacyParserAliases[parser];
        const status = cell.getAttribute('data-status-' + parser) || (alias ? cell.getAttribute('data-status-' + alias) : null) || 'na';
        counts[status] = (counts[status] || 0) + 1;
      });
      panel.querySelectorAll('[data-overview-count]').forEach(function (el) {
        el.textContent = String(counts[el.dataset.overviewCount] || 0);
      });
    });
  }

  function applyView(view, shouldUpdateUrl) {
    const activeView = viewKeys.has(view) ? view : 'overview';
    document.body.classList.toggle('view-overview', activeView === 'overview');
    document.body.classList.toggle('view-details', activeView === 'details');
    viewRadios.forEach(function (radio) {
      radio.checked = radio.value === activeView;
    });
    if (shouldUpdateUrl) {
      updateViewUrl(activeView);
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

  applyView(readActiveView(), false);
  applyParser(readActiveParser(), false);
  applyParityMode(readParityMode(), false);
  viewRadios.forEach(function (radio) {
    radio.addEventListener('change', function () {
      if (radio.checked) {
        applyView(radio.value, true);
      }
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
    syncParserOptions(shouldUpdateUrl);
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
    ttip.style.maxWidth = (window.innerWidth - 2 * margin) + 'px';
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
