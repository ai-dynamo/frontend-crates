// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
(function () {
  'use strict';

  const page = window.__CONFORMANCE_PAGE;
  const tab = page && page.tabs && page.tabs.find(item => item.id === 'tab-unified');
  const panel = document.getElementById('tab-unified');
  if (!tab || !panel) return;

  const engines = tab.comparison_slot_engines || ['dynamo', 'sglang', 'vllm'];
  const engineNames = { dynamo: 'Dynamo', sglang: 'SGLang', vllm: 'vLLM' };
  const engineLetters = { dynamo: 'D', sglang: 'S', vllm: 'V' };
  const candidates = (tab.candidates || []).filter(item => item && item.key && item.impl !== 'golden');
  const byKey = new Map(candidates.map(item => [item.key, item]));
  const engineName = engine => engine ? engineNames[engine] : 'None';
  const engineOf = item => {
    const impl = String(item.impl || item.key).toLowerCase();
    return impl.startsWith('dynamo') ? 'dynamo' : impl.startsWith('sglang') ? 'sglang' : 'vllm';
  };
  const hasCapture = key => (tab.rows || []).some(row => Object.values(row.cells || {}).some(cell => {
    const value = cell.cmp && cell.cmp[key];
    return value && value.na !== 1;
  }));
  const available = candidates.filter(item => hasCapture(item.key));
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
  const byEngine = new Map(engines.map(engine => [engine, available
    .filter(item => engineOf(item) === engine)
    .sort((left, right) => compareCaptures(left, right, engine))]));
  const captureChoices = engine => byEngine.get(engine) || [];
  const preferredByEngine = engines.map(engine => {
    return captureChoices(engine)[0]?.key || null;
  });
  const preferred = tab.comparison_defaults || [preferredByEngine[0], null, null];
  const params = new URLSearchParams(window.__CONFORMANCE_INITIAL_SEARCH || window.location.search);
  const slotsParam = 'slots_' + tab.id;
  const legacyBaseParam = 'base_' + tab.id;
  const legacyCompareParam = 'cmp_' + tab.id;
  const legacy = params.has(legacyBaseParam) || params.has(legacyCompareParam)
    ? [params.has(legacyBaseParam) ? params.get(legacyBaseParam) : preferred[0],
      ...(params.get(legacyCompareParam) || '').split(',')]
      .map(key => key && key !== 'golden' && byKey.has(key) ? key : '')
    : null;
  const requested = params.has(slotsParam) ? (params.get(slotsParam) || '').split(',') : legacy;
  const initial = requested === null ? preferred : requested;
  const overflow = initial.filter(Boolean).length > 3;
  const slots = [null, null, null];
  const slotEngines = [...engines];
  // Restoration and updates share primary ownership and duplicate prevention.
  function normalizeSlots() {
    if (!slots[0] || !byKey.has(slots[0])) slots[0] = preferred[0];
    const seen = new Set();
    slots.forEach((key, index) => {
      if (!byKey.has(key) || seen.has(key)) {
        slots[index] = null;
        slotEngines[index] = null;
      } else { seen.add(key); slotEngines[index] = engineOf(byKey.get(key)); }
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
    return captureChoices(engine).map(item => {
      const duplicate = slots.some((key, index) => index !== slot && key === item.key);
      const label = shortCaptureLabel(item, engine);
      return '<option value="' + item.key.replace(/&/g, '&amp;').replace(/"/g, '&quot;') + '"'
        + ' title="' + escapeAttr(captureTitle(item, engine)) + '"'
        + (selectedKey === item.key ? ' selected' : '') + (duplicate ? ' disabled' : '') + '>'
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
      + '<span class="capture-slot-engine" data-slot-engine="' + index + '">' + (engine ? engineName(engine) : '') + '</span></button>'
      + '<div class="capture-slot-capture-control" data-slot-capture-control="' + index + '"></div></div>';
  }).join('');
  selector.appendChild(controls);
  panel.insertBefore(selector, panel.querySelector('.cmpctl'));

  const state = { slots, engines: slotEngines };
  const selection = { keys: state.slots.filter(Boolean), reference: null };
  panel._comparisonSelection = selection;
  function persist() {
    const url = new URL(window.location.href);
    const current = state.slots.map(key => key || 'none').join(',');
    const isDefault = state.slots.every((key, index) => key === preferred[index]);
    if (isDefault) url.searchParams.delete(slotsParam);
    else url.searchParams.set(slotsParam, current);
    url.searchParams.delete(legacyBaseParam);
    url.searchParams.delete(legacyCompareParam);
    window.history.replaceState(null, '', url.toString());
  }
  function update() {
    normalizeSlots();
    selection.keys = state.slots.filter(Boolean);
    window.__applyConformanceSelection(panel, selection.keys, null);
    state.slots.forEach((key, index) => {
      const capture = byKey.get(key);
      const engine = state.engines[index];
      const slot = controls.querySelector('.capture-slot:nth-child(' + (index + 1) + ')');
      slot.dataset.engine = engine || 'none';
      slot.querySelector('[data-slot-engine]').textContent = engine ? engineName(engine) : '';
      slot.querySelector('[data-slot-badge]').textContent = engineLetters[engine] || '—';
      const captureControl = slot.querySelector('.capture-slot-capture-control');
      if (engine) {
        let captureSelect = captureControl.querySelector('[data-slot-capture]');
        if (!captureSelect) {
          captureSelect = document.createElement('select');
          captureSelect.className = 'capture-slot-select';
          captureSelect.dataset.slotCapture = index;
          captureControl.replaceChildren(captureSelect);
        }
        captureSelect.innerHTML = candidateOptions(engine, index, key);
        captureSelect.value = key || '';
        captureSelect.disabled = false;
        captureSelect.setAttribute('aria-label', 'Choose capture for comparison slot '
          + (index + 1) + ': ' + engineName(engine) + ', ' + shortCaptureLabel(capture, engine));
      } else {
        captureControl.replaceChildren();
      }
      slot.querySelector('[data-slot-open]').setAttribute('aria-label', 'Edit comparison slot ' + (index + 1) + ': ' + engineName(engine));
    });
    persist();
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
    state.slots[slot] = key;
    update();
  });
  picker.addEventListener('click', event => {
    if (event.target === picker) picker.close();
    const engineButton = event.target.closest('[data-picker-engine-choice]');
    if (engineButton) {
      const engine = engineButton.dataset.pickerEngineChoice;
      if (engine === 'none') {
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
})();
