// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
export const engineOrder = ['dynamo', 'sglang', 'vllm'];
const referenceParam = 'base_tab-unified';
const comparisonParam = 'cmp_tab-unified';

// This owner supplies controls, URL state and resource demand from the same identities.
export class EngineSelection {
  constructor(descriptors, url) {
    this.descriptors = descriptors;
    this.defaults = Object.fromEntries(engineOrder.map(engine => [engine, descriptors.find(item => item.control.impl === engine && (engine !== 'vllm' || item.source.implementation === 'vllm_python'))?.control.key || '']));
    this.values = { ...this.defaults };
    this.reference = this.defaults.dynamo || null;
    this.lastEngine = null;
    this.notices = [];
    const explicit = [referenceParam, comparisonParam, 'ref', 'cmp'].some(key => url.searchParams.has(key));
    if (explicit) {
      this.values = Object.fromEntries(engineOrder.map(engine => [engine, '']));
      const resolve = identity => descriptors.find(item => item.control.key === identity || item.source.id === identity || `${item.source.implementation}@${item.source.version}` === identity);
      const rawReference = url.searchParams.get(referenceParam) ?? url.searchParams.get('ref');
      let reference = rawReference === null ? null : resolve(rawReference);
      if (rawReference && !reference) {
        this.notices.push(`Unknown Reference ${rawReference}; using newest Dynamo.`);
        reference = resolve(this.defaults.dynamo);
      }
      this.reference = reference?.control.key || null;
      const listed = (url.searchParams.get(comparisonParam) ?? url.searchParams.get('cmp') ?? '').split(',').filter(Boolean);
      for (const identity of listed) if (!resolve(identity)) this.notices.push(`Unknown comparison capture ${identity} omitted.`);
      for (const descriptor of [reference, ...listed.map(resolve)].filter(Boolean)) {
        const engine = descriptor.control.impl;
        if (engineOrder.includes(engine) && !this.values[engine]) this.values[engine] = descriptor.control.key;
      }
    }
    url.searchParams.set('tab', 'tab-unified');
    url.searchParams.delete('ref'); url.searchParams.delete('cmp');
    this.serialize(url);
  }
  selection() { return { reference: this.reference, selected: engineOrder.map(engine => this.values[engine]).filter(Boolean) }; }
  change(engine, key) {
    if (key && !this.descriptors.some(item => item.control.impl === engine && item.control.key === key)) throw new Error('Invalid engine selection');
    const old = this.values[engine];
    this.values[engine] = key;
    this.lastEngine = engine;
    if (engine === 'dynamo') this.reference = key || null;
    else if (this.reference === old) this.reference = key || null;
  }
  serialize(url) {
    url.searchParams.set('tab', 'tab-unified');
    const selected = this.selection().selected;
    const defaults = engineOrder.every(engine => this.values[engine] === this.defaults[engine]) && this.reference === this.defaults.dynamo;
    url.searchParams.delete(referenceParam); url.searchParams.delete(comparisonParam);
    url.searchParams.delete('ref'); url.searchParams.delete('cmp');
    if (!defaults) {
      url.searchParams.set(referenceParam, this.identity(this.reference));
      url.searchParams.set(comparisonParam, selected.filter(key => key !== this.reference).map(key => this.identity(key)).join(','));
    }
    return url;
  }
  identity(key) { return this.descriptors.find(item => item.control.key === key)?.source.id || ''; }
  alternatives() {
    const priority = this.lastEngine ? [this.lastEngine, ...engineOrder.filter(engine => engine !== this.lastEngine)] : ['vllm', 'dynamo', 'sglang'];
    return priority.flatMap(engine => {
      const options = this.descriptors.filter(item => item.control.impl === engine && item.control.key !== this.values[engine]);
      if (!this.lastEngine && engine === 'vllm') options.sort((a, b) => Number(b.source.implementation === 'vllm_rust') - Number(a.source.implementation === 'vllm_rust'));
      return options.map(item => engineOrder.map(current => current === engine ? item.control.key : this.values[current]).filter(Boolean));
    });
  }
}

export function mountEngineControls(ctl, owner, onChange) {
  const status = document.createElement('p'); status.className = 'reference-status'; ctl.append(status);
  const sync = () => {
    for (const select of ctl.querySelectorAll('select[data-engine]')) select.value = owner.values[select.dataset.engine];
    const reference = owner.descriptors.find(item => item.control.key === owner.reference);
    status.textContent = `Reference: ${reference?.control.label || 'None'}`;
    const url = owner.serialize(new URL(location.href)); history.replaceState(null, '', url);
  };
  for (const select of ctl.querySelectorAll('select[data-engine]')) select.addEventListener('change', () => {
    owner.change(select.dataset.engine, select.value); sync(); onChange();
  });
  for (const message of owner.notices) { const notice = document.createElement('p'); notice.setAttribute('role', 'alert'); notice.textContent = message; ctl.append(notice); }
  sync();
}
