// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { canonicalArguments, sameValue, isMapping, mappingEntries, getField, hasField, setField, deleteField, mergeMappings, compareText, parseJson } from './values.js';

export const captureStatusLabels = {pass: 'Matches GOLDEN', documented: 'Documented differences', diverge: 'Different than GOLDEN',
  error: 'Parser errors', leak: 'Markup leaks', missing: 'Missing captures', unsupported: 'Unsupported cases',
  blocked: 'Blocked sources', expected: 'Expected errors', uncompared: 'GOLDEN unavailable'};

const has = hasField;
const parserError = (value) => /parser not captured|parsing failed|parse error|panicked|exception|traceback/i.test(value);


// IDs are local equivalence groups assigned after full typed-value comparison.
// There is no truncated digest capable of merging two unequal outputs.


export function equivalenceEntries(rows) {
  const values = [];
  const entries = {};
  for (const [key, { value, leak = 0, na = 0, err = 0 }] of Object.entries(rows)) {
    let index = values.findIndex((prior) => sameValue(prior, value));
    if (index < 0) { index = values.length; values.push(value); }
    setField(entries, key, { sig: index, leak, na, err });
  }
  return entries;
}

export function equivalenceGroups(entries) {
  const groups = new Map();
  for (const [key, entry] of Object.entries(entries)) {
    if (!groups.has(entry.sig)) groups.set(entry.sig, []);
    groups.get(entry.sig).push(key);
  }
  return Object.fromEntries(Object.entries(entries).map(([key, entry]) => [key, { ...entry, sig: [...groups.get(entry.sig)].sort() }]));
}


// Preserve the historical statistical buckets without shipping its private
// concatenated marker alphabet. Cells, facts and statistics share typed outputs.


export function canonicalUnifiedEvents(events) {
  return events.map((event) => getField(event, 'kind') === 'tool_call' ? decodedCall(event) : event);
}

export function classifyUnified(family, golden, observed, leakMarkers) {
  if (!Array.isArray(golden) || !Array.isArray(observed) || ![...golden, ...observed].every(isMapping)) throw new Error('Unified events must be lists of mappings');
  golden = canonicalUnifiedEvents(golden);
  observed = canonicalUnifiedEvents(observed);
  if (sameValue(golden, observed)) return 'MATCH';
  const markers = [...leakMarkers.common, ...(getField(leakMarkers.families, family) || [])];
  if (observed.some((event) => ['text', 'reasoning'].includes(getField(event, 'kind')) && markers.some((marker) => String(getField(event, 'text') || '').includes(marker)))) return 'LEAK';
  const count = (events, kind) => events.filter((event) => getField(event, 'kind') === kind).length;
  if (count(observed, 'reasoning') < count(golden, 'reasoning')) return 'MERGE';
  const calls = (events) => events.filter((event) => getField(event, 'kind') === 'tool_call');
  const expectedCalls = calls(golden), actualCalls = calls(observed);
  if (expectedCalls.length === actualCalls.length && expectedCalls.every((call, index) => getField(call, 'name') === getField(actualCalls[index], 'name')) && expectedCalls.some((call, index) => !sameValue(getField(call, 'arguments'), getField(actualCalls[index], 'arguments')))) return 'ARG_MISMATCH';
  const text = (events, kind) => events.filter((event) => getField(event, 'kind') === kind).map((event) => getField(event, 'text') || '').join('');
  if (text(golden, 'reasoning') === text(observed, 'reasoning') && text(golden, 'text') === text(observed, 'text')) return 'ORDER';
  return 'LOSS';
}

const pythonWhitespace = /^[\t\n\v\f\r \u001c-\u001f\u0085\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]*$/;


export const isApplicableCell = cell => cell.applicable ?? cell.status !== 'na';

export function leafCells(row) {
  const leaves = new Map();
  for (const [sub, cell] of Object.entries(row.cells)) {
    for (const leaf of cell.variants || [cell]) {
      const key = leaf.sub || sub;
      if (leaves.has(key) && !sameValue(leaves.get(key), leaf)) throw new Error(`inconsistent grouped fixture: ${key}`);
      leaves.set(key, leaf);
    }
  }
  return leaves;
}

export function explicitSchemas(tools) {
  if (!Array.isArray(tools)) return null;
  const schemas = tools.filter((tool) => typeof getField(tool, 'name') === 'string' && hasField(tool, 'parameters') && getField(tool, 'parameters') !== null)
    .map((tool) => ({ name: getField(tool, 'name'), parameters: getField(tool, 'parameters') }));
  return schemas.length ? schemas : null;
}

export function caseSchemas(authored, scenario, sharedTools = []) {
  const shared = explicitSchemas(sharedTools) || [];
  const variants = [];
  for (const [family, cases] of authored) {
    const tools = explicitSchemas(cases.get(scenario)?.request.tools);
    if (!tools || sameValue(tools, shared)) continue;
    let variant = variants.find(entry => sameValue(entry.tools, tools));
    if (!variant) { variant = {families:[],tools}; variants.push(variant); }
    variant.families.push(family);
  }
  return variants;
}

export function validateCaseCell(cell, column) {
  if (!cell || Object.hasOwn(cell, 'variants') || Object.hasOwn(cell.tooltip || {}, 'variants')) {
    throw new Error(`${column.sub}: one case per displayed cell; nested variants are forbidden`);
  }
  if (cell.case_id !== column.case_id) throw new Error(`${column.sub}: heading owns exactly its case`);
  return [cell];
}

export function validateCaseColumns(tab, authored, sharedTools = []) {
  const expected = new Map();
  for (const cases of authored.values()) for (const [scenario, input] of cases) {
    const prior = expected.get(scenario);
    if (prior && prior !== input.case_id) throw new Error(`${scenario}: conflicting case identity across families`);
    expected.set(scenario, input.case_id);
  }
  const columns = new Map(), ids = new Set();
  if (tab.columns.length !== expected.size) throw new Error('every authored case needs exactly one column');
  for (const column of tab.columns) {
    if (columns.has(column.sub) || ids.has(column.case_id) || expected.get(column.sub) !== column.case_id) {
      throw new Error(`${column.sub}: duplicate, extra, or conflicting case column`);
    }
    if (['variants','variant_parent','variant_labels'].some(key => Object.hasOwn(column, key))
        || (column.schemas || []).some(schema => Object.hasOwn(schema, 'cases'))) {
      throw new Error(`${column.sub}: aggregated case columns are forbidden`);
    }
    if (!sameValue(column.schemas || [], caseSchemas(authored, column.sub, sharedTools))) {
      throw new Error(`${column.sub}: displayed schemas differ from the authored request tools`);
    }
    columns.set(column.sub, column); ids.add(column.case_id);
  }
  const families = new Set();
  for (const row of tab.rows) {
    if (families.has(row.family) || !authored.has(row.family)) throw new Error('duplicate or extra family row');
    families.add(row.family);
    if (Object.keys(row.cells).length !== columns.size) throw new Error(`${row.family}: incomplete case cells`);
    const cases = authored.get(row.family);
    for (const [scenario, column] of columns) {
      const cell = row.cells[scenario];
      validateCaseCell(cell, column);
      if (cell.kind !== 'cell' || cell.sub !== scenario || cell.family !== row.family) {
        throw new Error(`${row.family}/${scenario}: misplaced case cell identity`);
      }
      const input = cases.get(scenario);
      if (!input) {
        if (isApplicableCell(cell)) throw new Error(`${row.family}/${scenario}: unexpected applicable case`);
        continue;
      }
      if (!isApplicableCell(cell)) throw new Error(`${row.family}/${scenario}: authored case is hidden`);
      const candidates = cell.tooltip?.candidates || [];
      const oracles = candidates.filter(candidate => candidate.key === 'golden');
      if (oracles.length !== 1 || new Set(candidates.map(candidate => candidate.key)).size !== candidates.length) {
        throw new Error(`${row.family}/${scenario}: exactly one GOLDEN and one observation per producer selection are required`);
      }
      const oracle = oracles[0].block?.events;
      const init = {starting_state:'None',tool_output_mode:'Native',named_tool:null,...Object.fromEntries(mappingEntries(input.request.init))};
      const chunks = (cell.tooltip?.input?.chunks || []).map(chunk => getField(chunk, 'delta_text', ''));
      if (!sameValue(cell.tooltip?.init, init) || cell.tooltip?.finish_reason !== (input.request.finish_reason || 'stop')
          || !sameValue(chunks, input.request.chunks.map(chunk => getField(chunk, 'delta_text', '')))) {
        throw new Error(`${row.family}/${scenario}: column must retain its one authored request configuration`);
      }
      if (cell.tooltip?.input?.text !== input.request.input || !sameValue(oracle, input.golden)) {
        throw new Error(`${row.family}/${scenario}: column must retain its one authored input and GOLDEN`);
      }
    }
  }
  if (families.size !== authored.size) throw new Error('every authored family needs exactly one row');
  return tab;
}

export function rankUnifiedColumns(tab, metadata) {
  if (tab.id !== 'tab-unified') return;
  const counts = new Map();
  const labels = new Map();
  const columns = [...tab.columns].sort((a, b) => metadata.unified_taxonomy[a.sub].display_order - metadata.unified_taxonomy[b.sub].display_order);
  for (const column of columns) {
    const taxonomy = metadata.unified_taxonomy[column.sub];
    const prefix = taxonomy.compact_prefix;
    counts.set(prefix, (counts.get(prefix) || 0) + 1);
    const label = taxonomy.label;
    labels.set(column.label, label);
    column.case_id = `UNIFIED.${column.label}`;
    Object.assign(column, { label, compact_prefix: prefix, display_number: counts.get(prefix) });
  }
  for (const group of tab.glossary || []) group.rows = group.rows.map(([label, description]) => [labels.get(label) || label, description]);
}

export function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#x27;' }[char]));
}


const slug = (version) => String(version || '').replace(/[^a-zA-Z0-9]+/g, '-').replace(/^-|-$/g, '');

export function sortCandidates(items) {
  const parserName = (item) => item.label.replace(/\s\d[\w.+]*\s*(?:\([^)]*\))?\s*$/, '').replace(/\s*\([^)]*\)\s*$/, '').toLowerCase();
  return [...items].sort((a, b) => (a.version && b.version ? compareLegacyVersions(b.version, a.version) : Number(Boolean(b.version)) - Number(Boolean(a.version))))
    .sort((a, b) => compareText(parserName(a), parserName(b)));
}

// The source descriptor carries implementation + data mode independently of
// display parse_mode (the v1 jail uses stream data but displays batch parsing).
export function tabCandidates(catalog, tab, { golden = false } = {}) {
  const metadata = catalog.metadata;
  const mode = tab.mode;
  if (mode !== 'unified') throw new Error(`unsupported conformance mode: ${mode}`);
  const policy = metadata.selection_policy[mode];
  if (!policy) throw new Error(`missing selection policy for ${mode}`);
  const raw = catalog.sources.candidates.filter((item) => item.mode === mode || (tab.id === 'tab-toolcalling-batch' && item.mode === 'batch-on-stream'));
  const unique = new Map();
  for (const source of raw) {
    if (typeof source.version !== 'string' || !unifiedCaptureVersionPattern.test(source.version)) throw new Error(`invalid capture version: ${source.version}`);
    const version = policy.fold_patch_suffix ? source.version.replace(/\.patch\d+$/, '') : source.version;
    const identity = `${source.implementation}@${source.mode}@${version}`;
    if (!unique.has(identity)) unique.set(identity, { ...source, id: identity, version });
  }
  const recordedSources = [...unique.values()];
  const compare = mode === 'unified' ? compareVersions : compareLegacyVersions;
  const latest = (implementation, dataMode = mode) => recordedSources.filter((item) => item.implementation === implementation && item.mode === dataMode).sort((a, b) => compare(b.version, a.version))[0];
  // Only the public Unified selector narrows family-local current evidence.
  const sources = mode === 'unified' && catalog.capture_generation?.mode === 'latest'
    ? recordedSources.filter(source => source.id === latest(source.implementation, source.mode).id)
    : recordedSources;
  const control = (source, key, label, bucket = 'C') => ({
    source,
    control: { key, implementation: source.implementation, impl: source.engine || metadata.unified_capture_policy?.[source.implementation]?.engine || metadata.implementations.find((item) => item.key === source.implementation)?.engine || source.implementation,
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
  throw new Error(`unsupported conformance mode: ${mode}`);
}

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

// Matrix paint and status reports consume the same recorded comparison relation.
// The report rejects Reference errors even when the legacy matrix has no concrete
// Compare result with which to show a delta; expose both facts explicitly.
export function selectedCellOutcome({ cmp = {}, base = null, shown = [], redOnDiff = false, family = '', parserNi = {} } = {}) {
  const entry = (key) => Object.hasOwn(cmp, key) ? cmp[key] : null;
  const reference = base ? entry(base) : null;
  const golden = entry('golden');
  const compares = shown.filter((key) => key !== base);
  const outcomes = compares.map((key) => ({ key, value: entry(key), outcome: selectedCaptureOutcome({ kind: 'cell', cmp }, key, base || 'golden') }));
  const available = outcomes.filter(({ value }) => value && value.na !== 1 && value.err !== 1);
  const unavailable = compares.filter((key) => entry(key)?.na === 1).length;
  const signature = golden?.sig != null ? golden.sig : reference?.sig;
  const differences = available.filter(({ value }) => !sameValue(value.sig, signature)).length;
  const errors = outcomes.filter(({ outcome }) => outcome === 'error').length;
  const leaks = outcomes.filter(({ outcome }) => outcome === 'leak').length;
  const result = { className: 'cmp-na', marker: 'n/a', category: 'na', why: '', referenceState: 'missing',
    missingGolden: golden == null, referenceDiffersFromGolden: reference && golden ? !sameValue(reference.sig, golden.sig) : null,
    referenceLeaks: reference?.leak === 1, availableComparisons: available.length, unavailableComparisons: unavailable,
    comparisonDifferences: differences, errorComparisons: errors, leakedComparisons: leaks };
  const ni = base && Object.hasOwn(parserNi, base) ? parserNi[base] : null;
  if (base && ni && !ni.families.includes(family)) {
    return { ...result, marker: 'x', referenceState: 'notImplemented', why: `${ni.label} is not yet implemented for family “${family}”` };
  }
  if (!base) return { ...result, className: 'cmp-nobase', marker: '', referenceState: 'none' };
  if (base === 'golden' && !golden) return { ...result, className: 'cmp-leak', marker: '✗', category: 'problem', referenceState: 'missing' };
  if (!reference) return result;
  if (reference.na === 1) return { ...result, referenceState: 'unavailable' };
  if (reference.err === 1) {
    if (redOnDiff) return { ...result, className: 'cmp-leak', marker: '✗', category: 'problem', referenceState: 'error' };
    if (available.length) return { ...result, className: 'cmp-leak', marker: `${available.length}Δ`, category: 'problem', referenceState: 'error' };
    return { ...result, referenceState: 'error' };
  }
  if (errors || leaks) return { ...result, className: 'cmp-leak', marker: `${errors ? '✗' : ''}${leaks ? '↯' : ''}`, category: 'problem', referenceState: errors ? 'error' : 'leak' };
  if (redOnDiff && !available.length) return { ...result, className: 'cmp-na', marker: unavailable ? '?' : 'n/a', category: 'na', referenceState: 'unavailable' };
  const diverges = redOnDiff && base === 'golden' && golden?.sig != null && differences > 0;
  const agreement = unavailable ? '?' : '=';
  const marker = redOnDiff ? diverges ? '✗' : differences ? `${differences}Δ` : agreement
    : !available.length ? '' : differences ? `${differences}Δ` : agreement;
  const problem = redOnDiff ? Boolean(diverges || golden == null) : result.referenceLeaks;
  return { ...result, className: problem ? 'cmp-leak' : 'cmp-eq', marker: (result.referenceLeaks ? '↯' : '') + marker,
    category: problem ? 'problem' : 'ok', referenceState: 'value' };
}

function tooltipCandidate(cell, key) {
  const candidates = cell?.tooltip?.candidates || [];
  return candidates.find(candidate => candidate.key === key) || null;
}

function sourceMatches(key, impl) {
  return typeof key === 'string' && typeof impl === 'string'
    && (key === impl || key.startsWith(impl + '-') || key.startsWith(impl + '@'));
}

function captureIssue(cell, key) {
  if (!cell || cell.kind === 'blank' || cell.kind === 'missing') return 'missing';
  const entry = (cell.cmp || {})[key];
  const candidateTip = tooltipCandidate(cell, key);
  const block = candidateTip?.block;
  if (!entry || entry.na === 1) {
    const reason = String((block && block.unavailable) || '');
    const code = block && (block.unavailable_code || (block.capture_observation && block.capture_observation.unavailable && block.capture_observation.unavailable.code));
    if (code === 'missing_capture') return 'missing';
    if (['unsupported_family', 'unsupported_request_mode', 'unsupported_capture_mode', 'scenario_not_applicable', 'duplicate_scenario'].includes(code)) return 'unsupported';
    if (code === 'source_unavailable' || /source unavailable|source blocked|historical source/i.test(reason)) return 'blocked';
    return /\bno(?: [\w.-]+)* parser\b|\bnot (?:yet )?implemented\b|unsupported (?:family|request|mode)|cannot apply/i.test(reason) ? 'unsupported' : 'missing';
  }
  if (entry.err === 1 || (block && (block.exception || (block.error && typeof block.error === 'object')))) return 'error';
  if (block && typeof block.error === 'string') return 'expected';
  if (entry.leak === 1 || candidateTip?.leak) return 'leak';
  return null;
}

export function selectedCapturesDiffer(cell, keys) {
  const entries = keys.filter(key => key !== 'golden').map(key => cell.cmp?.[key])
    .filter(entry => entry && entry.na !== 1 && entry.sig != null);
  return entries.length > 1 && entries.some(entry => !sameValue(entry.sig, entries[0].sig));
}

export function selectedCaptureOutcome(cell, key, reference = 'golden') {
  if (cell?.variants) {
    const outcomes = cell.variants.filter(isApplicableCell).map(leaf => selectedCaptureOutcome(leaf, key, reference));
    return ['error', 'leak', 'diverge', 'expected', 'missing', 'blocked', 'unsupported', 'uncompared', 'documented', 'pass']
      .find(outcome => outcomes.includes(outcome)) || 'unsupported';
  }
  const issue = captureIssue(cell, key);
  if (issue) return issue;
  const entry = cell.cmp[key];
  const compareEntry = (cell.cmp || {})[reference];
  if (['missing', 'unsupported', 'error', 'expected'].includes(captureIssue(cell, reference))) return 'uncompared';
  if (sameValue(entry.sig, compareEntry.sig)) return 'pass';
  const intentional = ((cell.tooltip && cell.tooltip.reasons) || []).some(reason => {
    if (reason.intentional !== true) return false;
    const forward = sourceMatches(key, reason.impl) && sourceMatches(reference, reason.reference_impl)
      && entry.sig === reason.candidate_sig && compareEntry.sig === reason.reference_sig;
    const reverse = sourceMatches(reference, reason.impl) && sourceMatches(key, reason.reference_impl)
      && compareEntry.sig === reason.candidate_sig && entry.sig === reason.reference_sig;
    const block = (tooltipCandidate(cell, forward ? key : reference) || {}).block;
    return (forward || reverse) && block && reason.reason === (block.explanation || block.reason);
  });
  return intentional ? 'documented' : 'diverge';
}

export const unifiedCaptureVersionPattern = /^\d+(?:\.\d+){2,}(?:\.post\d+|-[0-9A-Za-z.-]+)?$/;

export function provenanceReleaseVersion(version) {
  if (typeof version !== 'string') throw new Error('invalid provenance version');
  const release = version.replace(/\+source\.[0-9a-f]{64}$/, '');
  if (!unifiedCaptureVersionPattern.test(release)) throw new Error('invalid provenance version');
  return release;
}

export function compareVersions(left, right) {
  for (const value of [left, right]) {
    if (String(value).includes('+')) throw new Error(`invalid capture version: ${value}`);
  }
  if ([left, right].some((value) => !String(value).includes('-') && /\.patch\d+$/.test(value))) return compareLegacyVersions(left, right);
  const key = (value) => {
    const match = String(value).match(/^(\d+(?:\.\d+)*)(?:\.post(\d+)|-([0-9A-Za-z.-]+)|((?:a|b|rc|alpha|beta|dev)\d*))?$/);
    if (!match) throw new Error(`invalid capture version: ${value}`);
    const prerelease = match[3] || match[4] || null;
    return [match[1].split('.').map(BigInt), match[2] !== undefined ? 2 : prerelease === null ? 1 : 0, match[2] !== undefined ? BigInt(match[2]) : prerelease];
  };
  const a = key(left), b = key(right);
  for (let i = 0; i < Math.max(a[0].length, b[0].length); i++) {
    const x = a[0][i] ?? 0n, y = b[0][i] ?? 0n;
    if (x !== y) return x < y ? -1 : 1;
  }
  for (const index of [1, 2]) if (a[index] !== b[index]) return a[index] < b[index] ? -1 : 1;
  return 0;
}
export function compareLegacyVersions(left, right) {
  const key = (value) => {
    const patch = String(value).match(/^(.+)\.patch(\d+)$/);
    const match = (patch?.[1] || String(value)).match(/^(\d+(?:\.\d+)*)(?:[.-]?post(\d+))?/);
    return [match ? match[1].split('.').map(BigInt) : [], BigInt(match?.[2] || 0), BigInt(patch?.[2] || 0)];
  };
  const a = key(left), b = key(right);
  for (let index = 0; index < Math.min(a[0].length, b[0].length); index++) {
    if (a[0][index] !== b[0][index]) return a[0][index] < b[0][index] ? -1 : 1;
  }
  if (a[0].length !== b[0].length) return a[0].length < b[0].length ? -1 : 1;
  for (const index of [1, 2]) if (a[index] !== b[index]) return a[index] < b[index] ? -1 : 1;
  return 0;
}

export function selectedUnifiedSnapshot(records, checkpointVersion, selectedVersion) {
  const result = structuredClone(records);
  if (checkpointVersion !== selectedVersion) for (const record of result.values()) {
    const document = getField(getField(record, 'value'), 'document');
    const metadata = getField(document, 'record_metadata');
    deleteField(metadata, 'verified_recapture');
    if (metadata !== undefined && mappingEntries(metadata).length === 0) deleteField(document, 'record_metadata');
    if (!hasField(document, 'inherited_from') && !getField(record, 'suppress_inherited_from')) setField(document, 'inherited_from', checkpointVersion);
  }
  return result;
}

export function resolveUnified(captures, version) {
  return resolveUnifiedSeries(captures, [version]).next().value[1];
}

export function* resolveUnifiedSeries(captures, versions) {
  const ordered = [...captures].sort((left, right) => compareVersions(left.version, right.version));
  const stateByCase = new Map();
  const originByCase = new Map();
  const metadataOriginByCase = new Map();
  let index = 0;
  for (const version of [...versions].sort(compareVersions)) {
    while (index < ordered.length && compareVersions(ordered[index].version, version) <= 0) {
      const capture = ordered[index];
      if (Object.hasOwn(capture.document, 'parent') && capture.document.parent !== `${capture.implementation}-${ordered[index - 1]?.version}`) throw new Error(`capture parent is not the preceding checkpoint: ${capture.path}`);
      for (const [caseId, change] of Object.entries(capture.document.changes || {})) {
        if (change?.absent === true && Object.keys(change).length === 1) {
          stateByCase.delete(caseId);
          originByCase.delete(caseId);
          metadataOriginByCase.delete(caseId);
          continue;
        }
        const prior = stateByCase.get(caseId);
        if (!prior && ['case_key', 'stimulus', 'observation'].some(key => !Object.hasOwn(change, key))) throw new Error(`first capture change must provide a complete record: ${caseId} (${capture.path})`);
        const next = { ...structuredClone(prior || {}), ...structuredClone(change) };
        if (!prior || Object.hasOwn(change, 'observation')) {
          originByCase.set(caseId, capture);
          delete next._document_overrides;
        }
        if (prior?._record_metadata !== undefined) next._record_metadata = prior._record_metadata;
        if (prior?._document_overrides?.parser_path !== undefined) {
          next._document_overrides = Object.hasOwn(change, 'observation')
            ? { parser_path: prior._document_overrides.parser_path }
            : structuredClone(prior._document_overrides);
        }
        stateByCase.set(caseId, next);
      }
      for (const [caseId, metadata] of Object.entries(capture.document.metadata_changes || {})) {
        if (!stateByCase.has(caseId)) throw new Error(`metadata change has no observation: ${caseId} (${capture.path})`);
        const record = stateByCase.get(caseId);
        if (metadata instanceof Map ? metadata.size : Object.keys(metadata || {}).length) {
          record._record_metadata = structuredClone(metadata);
          metadataOriginByCase.set(caseId, capture.version);
        } else {
          delete record._record_metadata;
          metadataOriginByCase.delete(caseId);
        }
      }
      for (const [caseId, override] of Object.entries(capture.document.document_overrides || {})) {
        if (!stateByCase.has(caseId)) throw new Error(`document override has no observation: ${caseId} (${capture.path})`);
        const record = stateByCase.get(caseId);
        record._document_overrides = { ...(record._document_overrides || {}), ...structuredClone(override || {}) };
      }
      index += 1;
    }
    yield [version, new Map([...stateByCase].map(([caseId, record]) => {
      const { _record_metadata, _document_overrides, ...observation } = structuredClone(record);
      const __origin = originByCase.get(caseId);
      const document = mergeMappings(structuredClone(__origin.document.document), { captured_with: { [__origin.implementation]: __origin.version } });
      const origin = __origin.document.provenance?.origin;
      if (__origin.document.provenance?.status === 'captured' && origin) setField(document, 'capture_origin', structuredClone(origin));
      if (version !== __origin.version) setField(document, 'inherited_from', __origin.version);
      const provenanceOverride = _document_overrides?.capture_provenance;
      if (provenanceOverride) {
        for (const [key, value] of Object.entries(provenanceOverride)) {
          if (value === null) deleteField(document, key);
          else setField(document, key, structuredClone(value));
        }
      }
      for (const [key, value] of Object.entries(_document_overrides || {})) {
        if (key === 'parser_path' && value === null) deleteField(document, key);
        else if (key !== 'capture_provenance') setField(document, key, structuredClone(value));
      }
      if (_record_metadata instanceof Map ? _record_metadata.size : _record_metadata && Object.keys(_record_metadata).length) {
        const metadata = structuredClone(_record_metadata);
        if (metadataOriginByCase.get(caseId) !== version) deleteField(metadata, 'verified_recapture');
        setField(document, 'record_metadata', metadata);
      }
      observation.document = document;
      return [caseId, { value: observation, member: __origin.path, sourceBytes: __origin.bytes }];
    }))];
  }
}

// Historical typed captures rounded numbers, but lost fields and changed values
// are parser results. Restore raw precision only across the same projection.
function matchesRecordedProjection(raw, recorded) {
  const numeric = value => typeof value === 'number' || typeof value === 'bigint';
  if (typeof recorded === 'bigint') return raw === recorded;
  if (numeric(raw) && typeof recorded === 'number') return Number.isFinite(recorded) && Number(raw) === recorded;
  if (Array.isArray(raw)) return Array.isArray(recorded) && raw.length === recorded.length && raw.every((value, index) => matchesRecordedProjection(value, recorded[index]));
  if (isMapping(raw)) {
    if (!isMapping(recorded) || Array.isArray(recorded)) return false;
    const entries = mappingEntries(raw);
    return entries.length === mappingEntries(recorded).length && entries.every(([key, value]) => hasField(recorded, key) && matchesRecordedProjection(value, getField(recorded, key)));
  }
  return sameValue(raw, recorded);
}

export function assembleStream(chunks, recordedAssembly = null, preserveArguments = false) {
  const events = [];
  let currentTool = null;
  for (const deltas of chunks || []) {
    for (const delta of deltas || []) {
      if (!isMapping(delta)) throw new Error('Unified stream delta must be a mapping');
      const kind = getField(delta, 'kind');
      const name = getField(delta, 'name');
      const args = getField(delta, 'arguments');
      const complete = getField(delta, 'complete', true);
      if (kind === 'message_start' || kind === 'message_end') {
        currentTool = null;
        events.push(kind === 'message_start' ? { kind, phase: getField(delta, 'phase') } : { kind });
      } else if (kind === 'reasoning' || kind === 'text') {
        currentTool = null;
        const text = getField(delta, 'text') || '';
        if (typeof text !== 'string') throw new Error('Unified text delta must contain text');
        events.push({ kind, text });
      } else if (kind === 'tool_call') {
        if (name != null) { // a name delta opens a new call, an empty name included
          events.push({ kind: 'tool_call', name, arguments: args || '', complete });
          currentTool = events.length - 1;
        } else if (currentTool !== null && args) {
          const before = events[currentTool].arguments;
          const part = args;
          events[currentTool].arguments = typeof before === 'string' && typeof part === 'string' ? before + part : part;
        }
        if (currentTool !== null && complete) events[currentTool].complete = true;
      }
    }
  }
  const completed = [];
  for (const event of events) {
    if (event.kind === 'tool_call' && !event.complete) continue;
    if ((event.kind === 'text' || event.kind === 'reasoning') && completed.at(-1)?.kind === event.kind) completed.at(-1).text += event.text;
    else completed.push(event);
  }
  const malformed = new Set();
  const projections = new Map();
  for (const event of completed) {
    if (event.kind !== 'tool_call') continue;
    delete event.complete;
    if (isMapping(event.arguments) || Array.isArray(event.arguments)) { projections.set(event, event.arguments); continue; }
    if (typeof event.arguments !== 'string') throw new Error('Unified argument fragment must be text, a mapping, or a list');
    if (!event.arguments.trim()) { event.arguments = {}; projections.set(event, event.arguments); }
    else {
      try { const decoded = parseJson(event.arguments); projections.set(event, decoded); if (!preserveArguments) event.arguments = decoded; }
      catch (error) {
        if (!(error instanceof SyntaxError)) throw error;
        malformed.add(event);
      }
    }
  }
  if (recordedAssembly) {
    const calls = completed.filter(event => event.kind === 'tool_call');
    const recorded = recordedAssembly.filter(event => getField(event, 'kind') === 'tool_call');
    if (calls.length === recorded.length && calls.every((event, index) => event.name === getField(recorded[index], 'name'))) {
      calls.forEach((event, index) => {
        const args = getField(recorded[index], 'arguments');
        let projected = args;
        if (typeof args === 'string') {
          try { projected = parseJson(args); } catch (error) { if (!(error instanceof SyntaxError)) throw error; }
        }
        if (malformed.has(event)) {
          if (sameValue(projected, {})) event.arguments = preserveArguments ? args : projected;
          return;
        }
        if (!projections.has(event)) return;
        const matches = typeof args === 'string'
          ? sameValue(canonicalArguments(event.arguments), canonicalArguments(args))
          : matchesRecordedProjection(projections.get(event), projected);
        if (!matches) event.arguments = preserveArguments ? args : projected;
      });
    }
  }
  return completed;
}

export function unifiedChunkDeltas(value) {
  const chunks = getField(value, 'chunks');
  if (!Array.isArray(chunks)) return [];
  return chunks.map((chunk) => Array.isArray(chunk) ? chunk : getField(chunk, 'expected', []));
}

export function unifiedEvents(value, { stream = true, recordedFallback = true } = {}) {
  if (!value || typeof value !== 'object') return [];
  const chunks = getField(value, 'chunks');
  const assembled = getField(value, 'assembled');
  if (stream && Array.isArray(chunks)) return assembleStream(unifiedChunkDeltas(value), recordedFallback ? assembled : null, true);
  if (Array.isArray(assembled)) return assembled;
  return [];
}


export function unifiedResult(record) {
  if (!record) return { state: 'missing' };
  const value = record.value;
  const observation = value?.observation;
  if (observation?.unavailable) return { state: 'unavailable', reason: getField(observation.unavailable, 'detail') ?? getField(observation.unavailable, 'message') ?? getField(observation.unavailable, 'code') };
  if (observation?.error) return { state: 'error', reason: getField(observation.error, 'detail') ?? getField(observation.error, 'message') ?? getField(observation.error, 'code') };
  if (observation?.state && observation.state !== 'value') return { state: observation.state, reason: observation.code || '' };
  const payload = observation?.value || value?.value || value;
  return { state: 'value', payload, events: unifiedEvents(payload) };
}
export { sameValue } from './values.js';

function decodedCall(call) {
  if (!isMapping(call) || !hasField(call, 'arguments')) return call;
  const result = structuredClone(call);
  setField(result, 'arguments', canonicalArguments(getField(call, 'arguments')));
  return result;
}
