// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
window.mountConformanceCaptureSlots = function (page, panel, onChange = () => {}, { availableIdentities = null } = {}) {
  const tab = page && page.tabs && page.tabs.find(item => item.id === 'tab-unified');
  if (!tab || !panel) return null;

  const engines = tab.comparison_slot_engines || ['dynamo', 'sglang', 'vllm'];
  const engineNames = { dynamo: 'Dynamo', sglang: 'SGLang', vllm: 'vLLM' };
  const engineLetters = { dynamo: 'D', sglang: 'S', vllm: 'V' };
  const candidates = (tab.candidates || []).filter(item => item && item.key && item.impl !== 'golden');
  const byKey = new Map(candidates.map(item => [item.key, item]));
  const byIdentity = new Map(candidates.flatMap(item => [[item.key, item.key], [item.source_id, item.key], [item.identity, item.key]]).filter(([identity]) => identity));
  let selectionTouched = false;
  const engineName = engine => engine ? engineNames[engine] : 'None';
  const engineOf = item => {
    const impl = String(item.impl || item.key).toLowerCase();
    return impl.startsWith('dynamo') ? 'dynamo' : impl.startsWith('sglang') ? 'sglang' : 'vllm';
  };
  // Catalog descriptors define availability before any selected history is loaded.
  const available = candidates.filter(item => {
    const families = item.native_unified_families || item.capture_families;
    return !item.source_blocker && item.has_captured_results !== false && (!families || families.length);
  });
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
    setAvailableIdentities(identities) {
      const previous = state.slots.join(',');
      availableIdentities = identities;
      // Discovery may finish after a user changes a slot. Preserve that choice;
      // only an untouched initial historical URL can restore its deferred slots.
      if (!selectionTouched) initial.slice(0, 3).forEach((key, index) => { state.slots[index] = key || null; });
      update(false);
      const changed = state.slots.join(',') !== previous;
      if (changed) onChange();
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
