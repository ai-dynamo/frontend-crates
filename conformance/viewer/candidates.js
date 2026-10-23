// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { compareLegacyVersions, compareVersions } from './model.js';
import { compareText } from './values.js';

export function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#x27;' }[char]));
}

export function candidateLabelHtml(label) {
  return escapeHtml(label).replace(/\(([^)]*)\)$/, (_, inner) => `(${inner.replaceAll('batch', '<span class="cand-batch">batch</span>').replaceAll('stream', '<span class="cand-stream">stream</span>')})`);
}

export function fullLabel(implementation, version, mode, metadata) {
  const label = metadata.implementations.find((item) => item.key === implementation)?.label || implementation;
  const parseMode = mode === 'stream' && implementation === 'dynamo_v2' ? 'stream, Combined & Unified' : mode === 'stream' && implementation === 'dynamo_v1' ? 'jail+batch' : mode;
  return `${label}${version ? ` ${version}` : ''} (${parseMode})`;
}

const slug = (version) => String(version || '').replace(/[^a-zA-Z0-9]+/g, '-').replace(/^-|-$/g, '');

export function sortCandidates(items) {
  const parserName = (item) => item.label.replace(/\s\d[\w.+]*\s*(?:\[unpublished sha256:[a-f0-9]+\]\s*)?(?:\([^)]*\))?\s*$/, '').replace(/\s*\([^)]*\)\s*$/, '').toLowerCase();
  return [...items].sort((a, b) => (a.version && b.version ? compareLegacyVersions(b.version, a.version) : Number(Boolean(b.version)) - Number(Boolean(a.version))))
    .sort((a, b) => compareText(parserName(a), parserName(b)));
}

// The source descriptor carries implementation + data mode independently of
// display parse_mode (the v1 jail uses stream data but displays batch parsing).
export function tabCandidates(catalog, tab, { golden = false } = {}) {
  const metadata = catalog.metadata;
  const mode = tab.suite === 'reasoning' ? 'reasoning' : tab.mode;
  const policy = metadata.selection_policy[mode];
  if (!policy) throw new Error(`missing selection policy for ${mode}`);
  const raw = catalog.sources.candidates.filter((item) => item.mode === mode || (tab.id === 'tab-toolcalling-batch' && item.mode === 'batch-on-stream'));
  const unique = new Map();
  for (const source of raw) {
    const version = policy.fold_patch_suffix ? source.version.replace(/\.patch\d+$/, '') : source.version;
    const identity = `${source.implementation}@${source.mode}@${version}`;
    if (!unique.has(identity)) unique.set(identity, { ...source, id: identity, version });
  }
  const sources = [...unique.values()];
  const compare = mode === 'unified' ? compareVersions : compareLegacyVersions;
  const latest = (implementation, dataMode = mode) => sources.filter((item) => item.implementation === implementation && item.mode === dataMode).sort((a, b) => compare(b.version, a.version))[0];
  const control = (source, key, label, bucket = 'C') => ({
    source,
    control: { key, impl: source.engine || metadata.unified_capture_policy?.[source.implementation]?.engine || metadata.implementations.find((item) => item.key === source.implementation)?.engine || source.implementation,
      label, label_html: mode === 'unified' ? label : candidateLabelHtml(label), default_bucket: bucket,
      version: source.version, parse_mode: mode === 'unified' ? (source.capture_mode || metadata.unified_capture_policy?.[source.implementation]?.mode || 'stream') : /\([^)]*stream/.test(label) ? 'stream' : /\([^)]*batch/.test(label) ? 'batch' : null,
      language: source.language || metadata.unified_capture_policy?.[source.implementation]?.language || null,
      capture_families: source.capture_families,
      native_unified_families: source.native_unified_families,
      has_captured_results: source.has_captured_results,
      source_blocker: source.source_blocker || null },
  });
  const oracle = (parseMode, bucket) => ({ source: { implementation: 'golden', mode: parseMode, version: null, id: `golden@${parseMode}` },
    control: { key: 'golden', impl: 'golden', label: 'GOLDEN (oracle)', label_html: 'GOLDEN (oracle)', default_bucket: bucket, version: null, parse_mode: parseMode } });
  if (mode === 'unified') {
    const reference = latest(policy.reference);
    if (!reference) throw new Error('Unified reference history missing');
    const implementationOrder = [...new Set([...policy.implementation_order, ...sources.map(item => item.implementation)])];
    const labelFor = source => {
      const capture = metadata.unified_capture_policy?.[source.implementation] || {};
      const producer = metadata.implementations.find(item => item.key === capture.producer);
      const engine = capture.engine || source.engine;
      const producerLabel = producer?.label || (engine === 'dynamo' ? 'Dynamo v1 Rust' : engine === 'vllm' ? 'vLLM Python' : engine === 'sglang' ? 'SGLang Python' : source.implementation);
      const flavor = source.implementation === 'dynamo_v2' && source.id !== reference.id ? 'stream, recorded history'
        : source.implementation === 'dynamo_v1_jail' ? 'batch + jail-buffer'
        : capture.mode === 'batch' ? 'batch'
          : policy.capture_contracts?.[source.implementation]?.label || 'stream';
      return `${producerLabel} ${source.version} (${flavor})`;
    };
    const makeControl = (source, key, bucket = 'C') => control(source, key, labelFor(source), bucket);
    const result = [makeControl(reference, 'dynamo', 'A'), oracle('unified', policy.golden_bucket)];
    for (const implementation of implementationOrder) {
      const versions = sources.filter((item) => item.implementation === implementation && item.id !== reference.id)
        .sort((a, b) => compare(b.version, a.version));
      for (const source of versions) {
        const identity = `${implementation}@${source.version}`;
        const key = implementation === policy.reference ? `dynamo@${source.version}`
          : policy.historical_control_aliases[identity] || identity;
        result.push(makeControl(source, key));
      }
    }
    return result;
  }
  if (mode === 'reasoning') return policy.implementation_order.flatMap((implementation) => {
    const source = latest(implementation);
    const label = `${metadata.implementations.find((item) => item.key === implementation).label} ${source?.version} (${tab.mode})`;
    return source ? [control(source, implementation, label, implementation === policy.reference ? 'A' : 'C')] : [];
  });
  const reference = latest(policy.reference);
  const result = sources.map((source) => {
    const batch = tab.id === 'tab-toolcalling-batch';
    const dataMode = source.mode === 'batch-on-stream' ? 'stream' : source.mode;
    const key = `${source.implementation}-${batch ? dataMode === 'batch' ? 'b-' : 's-' : ''}${slug(source.version)}`;
    return control(source, key, fullLabel(source.implementation, source.version, dataMode, metadata), source.id === reference?.id ? 'A' : 'C');
  });
  const ordered = sortCandidates(result.map((item) => ({ ...item.control, descriptor: item }))).map((item) => item.descriptor);
  if (golden) ordered.push(oracle(tab.mode, policy.golden_bucket));
  return ordered;
}

// Catalog presence controls are shared by demand loading and row construction.
export function sourceCapturesFamily(source, family, { nativeOnly = true } = {}) {
  if (nativeOnly && source.implementation === 'dynamo_v2' && source.mode === 'unified') {
    if (!Array.isArray(source.native_unified_families)) throw new Error('Dynamo v2 capture lacks version-specific UnifiedParser eligibility');
    if (!source.native_unified_families.includes(family)) return false;
  }
  return !source.capture_families || source.capture_families.includes(family);
}

export function unifiedDisplayFamilyPredicate(catalog, descriptors = []) {
  const current = (catalog.sources.candidates || descriptors.map(item => item.source))
    .filter(source => source.implementation === 'dynamo_v2' && (!source.mode || source.mode === 'unified'))
    .sort((left, right) => compareVersions(right.version, left.version))[0];
  // Current native eligibility bounds the table's families. Historical Combined
  // observations remain displayable without claiming a native route for that release.
  return (source, family) => sourceCapturesFamily(source, family, { nativeOnly: false })
    && (source.implementation !== 'dynamo_v2' || sourceCapturesFamily(current || source, family));
}

export function selectedUnifiedFamilies(catalog, descriptors) {
  const capturesFamily = unifiedDisplayFamilyPredicate(catalog, descriptors);
  return new Set(catalog.sources.unified.filter(resource => resource.kind === 'inputs-and-golden')
    .flatMap(resource => resource.families || [resource.family])
    .filter(family => family && descriptors.some(({ source }) => capturesFamily(source, family))));
}
