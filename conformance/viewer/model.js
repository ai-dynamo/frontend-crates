// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { canonicalArguments, sameValue, isMapping, mappingEntries, getField, hasField, setField, deleteField, mergeMappings, compareText, parseJson } from './values.js';

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

export function aggregateCells(cells, parent, description) {
  if (!cells.length) throw new Error(`empty fold: ${parent}`);
  const result = structuredClone(cells.find((cell) => cell.case_id) || cells[0]);
  result.sub = cells[0].sub;
  const id = result.case_id || '';
  result.case_id = (id.startsWith('UNIFIED.') ? 'UNIFIED.' : id.replace(/(?:7-14|7-15)\..*$/, '').split(parent)[0]) + parent;
  result.kind = 'cell';
  result.status = cells.some(isApplicableCell) ? 'ok' : 'na';
  if (cells.some(cell => Object.hasOwn(cell, 'applicable'))) result.applicable = cells.some(isApplicableCell);
  result.red_on_diff = true;
  const applicable = cells.filter(isApplicableCell);
  // Keep comparison identities even when every folded leaf is inapplicable.
  // The aggregate still needs NA entries for selected engines so rendering and
  // status checks can distinguish an unavailable case from a missing engine.
  const keys = [...new Set(cells.flatMap((cell) => Object.keys(cell.cmp || {})))].sort(compareText);
  const signatures = [];
  result.cmp = {};
  for (const key of keys) {
    const entries = applicable.map((cell) => cell.cmp?.[key] || { na: 1 });
    const missing = !applicable.length || entries.some((entry) => entry.na);
    const failure = entries.some((entry, index) => !entry.na && (entry.err || entry.leak || (applicable[index].cmp?.golden && !sameValue(entry.sig, applicable[index].cmp.golden.sig))));
    const signature = entries.map((entry) => entry.na ? null : entry.sig);
    let group = signatures.findIndex((prior) => sameValue(prior, signature));
    if (group < 0) { group = signatures.length; signatures.push(signature); }
    setField(result.cmp, key, { sig: group, na: Number(missing && !failure), err: Number(entries.some((entry) => entry.err)), leak: Number(entries.some((entry) => entry.leak)) });
  }
  result.variants = structuredClone(cells);
  result.tooltip = { head: `${result.case_id} — ${result.family}`, description,
    input: { kind: null }, init: null, candidates: [], variants: result.variants.map((cell) => cell.tooltip) };
  return result;
}

export function foldedSchemas(columns) {
  const variants = [];
  for (const column of columns) for (const variant of column.schemas || []) {
    let merged = variants.find((prior) => sameValue(prior.tools, variant.tools));
    if (!merged) { merged = { tools: variant.tools, families: [], cases: [] }; variants.push(merged); }
    for (const family of variant.families) if (!merged.families.includes(family)) merged.families.push(family);
    merged.cases.push({ label: column.variant_label || column.label, case_id: column.case_id || `UNIFIED.${column.label}`, families: variant.families });
  }
  return variants;
}

function refreshStats(tab, hidden) {
  for (const group of tab.column_groups) group.span = tab.columns.filter((column) => column.group_key === group.key).length;
  tab.column_groups = tab.column_groups.filter((group) => group.span);
  tab.stats.variant_columns = (tab.stats.variant_columns || 0) + hidden.size;
  tab.stats.sub_cases = tab.columns.length;
  const visible = tab.rows.flatMap((row) => Object.values(row.cells));
  tab.stats.fixture_cases = tab.rows.reduce((total, row) => total + [...leafCells(row).values()].filter((cell) => cell.kind === 'cell' && isApplicableCell(cell)).length, 0);
  Object.assign(tab.stats, { slots: visible.length, real: visible.filter((cell) => cell.kind === 'cell' && cell.status !== 'na').length,
    na: visible.filter((cell) => cell.status === 'na').length, missing: visible.filter((cell) => cell.kind === 'missing').length });
}

function removeColumns(tab, hidden) {
  tab.columns = tab.columns.filter((column) => !hidden.has(column.sub));
  for (const row of tab.rows) for (const sub of hidden) delete row.cells[sub];
  refreshStats(tab, hidden);
}

function compareTuple(left, right) {
  for (let index = 0; index < Math.min(left.length, right.length); index++) {
    const a = left[index], b = right[index];
    const difference = Array.isArray(a) ? compareTuple(a, b) : typeof a === 'string' ? compareText(a, b) : a - b;
    if (difference) return difference;
  }
  return left.length - right.length;
}

export function foldNullVariants(tab, metadata) {
  if (tab.id !== 'tab-unified') return;
  const config = metadata.folds;
  const unified = tab.id === 'tab-unified';
  const identity = (column) => unified ? config.unified[column.sub].legacy_label : column.label;
  const descriptions = config.variant_descriptions || config.null_descriptions;
  const parentOf = (label) => config.variant_groups?.[label] || (Object.hasOwn(config.null_descriptions, label.split('.')[0]) ? label.split('.')[0] : null);
  const canonical = (label) => unified ? Object.values(config.unified).find((item) => item.legacy_label === label)?.canonical_label || label : label;
  const sorted = (columns) => unified ? columns.sort((a, b) => compareTuple(config.unified[a.sub].taxonomy_order, config.unified[b.sub].taxonomy_order)) : columns;
  const displayLabels = new Map();
  for (const [parent, description] of Object.entries(descriptions)) {
    const sections = [...new Set(tab.columns.filter((column) => parentOf(identity(column)) === parent).map((column) => column.group_key))];
    for (const section of sections) {
      const members = sorted(tab.columns.filter((column) => parentOf(identity(column)) === parent && column.group_key === section));
      const root = members.find((column) => identity(column) === parent) || members[0];
      const display = config.display_labels?.[parent] || (members.length === 1 ? root.label : canonical(parent));
      displayLabels.set(root.label, display);
      if (config.display_labels?.[parent]) root.grouped_display_label = display;
      const mixed = sorted(tab.columns.filter((column) => identity(column).startsWith('7-4.mixed_') && column.group_key === section));
      const referenced = [...members, ...(parent === '7-5' ? mixed : [])];
      root.desc = description;
      const schemas = foldedSchemas(referenced);
      if (schemas.length) root.schemas = schemas;
      for (const row of tab.rows) {
        if (row.section || !Object.hasOwn(row.cells, root.sub)) continue;
        const inapplicable = config.numeric_applicability?.[parent] && !config.numeric_applicability[parent].includes(row.family);
        if (inapplicable) for (const column of members) {
          const cell = row.cells[column.sub];
          if (!cell) continue;
          Object.assign(cell, { kind: 'cell', status: 'na', applicable: false, red_on_diff: false,
            cmp: Object.fromEntries(tab.candidates.map(({key}) => [key, {sig: 0, na: 1, err: 0, leak: 0}])) });
          cell.tooltip = {...cell.tooltip, description, na_note: 'This grammar does not coerce untyped fractional text through an integer/string schema.'};
        }
        const children = referenced.map((column) => row.cells[column.sub]).filter((cell) => cell && ['cell', 'missing'].includes(cell.kind) && (inapplicable || isApplicableCell(cell)));
        if (children.length && (children.length > 1 || children[0] !== row.cells[root.sub] || inapplicable)) {
          row.cells[root.sub] = aggregateCells(children, display, description);
          row.cells[root.sub].sub = root.sub;
          if (inapplicable) row.cells[root.sub].tooltip.na_note = children[0].tooltip.na_note;
        }
      }
      for (const column of members) if (column !== root) column.variant_parent = parent;
    }
  }
  const hidden = new Set(tab.columns.filter((column) => Object.hasOwn(column, 'variant_parent')).map((column) => column.sub));
  removeColumns(tab, hidden);
  for (const column of tab.columns) column.label = displayLabels.get(column.label) || column.label;
  const numericOrder = Object.values(config.display_labels || {});
  const positions = tab.columns.map((column, index) => numericOrder.includes(column.label) ? index : -1).filter((index) => index >= 0);
  const numericColumns = positions.map((index) => tab.columns[index]).sort((a,b) => numericOrder.indexOf(a.label) - numericOrder.indexOf(b.label));
  positions.forEach((position, index) => { tab.columns[position] = numericColumns[index]; });
  const labels = new Set(tab.columns.map((column) => column.label));
  const nullParent = (label) => parentOf(unified ? Object.values(config.unified).find((item) => item.canonical_label === label)?.legacy_label || label : label);
  for (const group of tab.glossary || []) group.rows = group.rows.filter(([label]) => !nullParent(label) || displayLabels.has(label))
    .map(([label, description]) => [displayLabels.get(label) || label, descriptions[nullParent(label)] || description])
    .filter(([label]) => !numericOrder.includes(label) || labels.has(label));
}

export function foldSchemaVariants(tab, metadata) {
  if (tab.id !== 'tab-unified') return;
  const groups = new Map();
  const order = ['Direct', 'Explicit string type', 'allOf', 'Reference'];
  const unionOrder = ['String / Forward', 'Typed value / Forward', 'String / Reversed', 'Typed value / Reversed'];
  for (const column of tab.columns) {
    const fold = metadata.folds.unified[column.sub].schema || metadata.folds.unified[column.sub].case;
    if (!fold) continue;
    const [parent, label] = fold;
    column.variant_label = label;
    if (!groups.has(parent)) groups.set(parent, []);
    groups.get(parent).push(column);
  }
  const hidden = new Set();
  for (const [parent, members] of groups) {
    const rank = (column) => Math.max(order.indexOf(column.variant_label), unionOrder.indexOf(column.variant_label));
    members.sort((a, b) => rank(a) - rank(b));
    const root = members.find((column) => column.sub === parent) || members[0];
    const labels = new Map(members.map((column) => [column.sub, column.variant_label]));
    const description = members.some(column => metadata.folds.unified[column.sub].case) ? root.desc : parent.includes('const_') ? 'String constant: Direct, Explicit string type, allOf, and Reference.' : 'Union alternatives: String / Typed value x Forward / Reversed branch order.';
    root.desc = description;
    root.schemas = foldedSchemas(members);
    for (const row of tab.rows) {
      if (row.section) continue;
      const children = members.map((column) => row.cells[column.sub]).filter(Boolean);
      if (!children.length) continue;
      const aggregated = aggregateCells(children, metadata.folds.unified[root.sub].canonical_label, description);
      aggregated.sub = root.sub;
      for (const child of aggregated.variants) if (child.tooltip) child.tooltip.head = `${labels.get(child.sub)} — ${child.tooltip.head}`;
      row.cells[root.sub] = aggregated;
    }
    for (const column of members) if (column !== root) hidden.add(column.sub);
  }
  const hiddenLabels = new Set(tab.columns.filter((column) => hidden.has(column.sub)).map((column) => column.label));
  removeColumns(tab, hidden);
  const descriptions = new Map(tab.columns.map((column) => [column.label, column.desc]));
  for (const group of tab.glossary || []) group.rows = group.rows.filter(([label]) => !hiddenLabels.has(label)).map(([label, description]) => [label, descriptions.get(label) || description]);
}

export function rankUnifiedColumns(tab, metadata) {
  if (tab.id !== 'tab-unified') return;
  const counts = new Map();
  const labels = new Map();
  const columns = [...tab.columns].sort((a, b) => metadata.unified_taxonomy[a.sub].display_order - metadata.unified_taxonomy[b.sub].display_order);
  for (const column of columns) {
    const taxonomy = metadata.unified_taxonomy[column.sub];
    const group = taxonomy.display_group;
    counts.set(group, (counts.get(group) || 0) + 1);
    const label = column.grouped_display_label || taxonomy.label;
    labels.set(column.label, label);
    column.case_id = `UNIFIED.${column.label}`;
    Object.assign(column, { label, compact_prefix: taxonomy.compact_prefix, display_number: counts.get(group) });
  }
  for (const group of tab.glossary || []) group.rows = group.rows.map(([label, description]) => [labels.get(label) || label, description]);
}

export function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#x27;' }[char]));
}


function captureVersionLabel(implementation, version, metadata) {
  const producer = metadata.dynamo_v2_producer;
  return implementation === 'dynamo_v2' && producer?.kind === 'unpublished' && version === producer.crate_version
    ? `${version} [unpublished]` : version;
}


const slug = (version) => String(version || '').replace(/[^a-zA-Z0-9]+/g, '-').replace(/^-|-$/g, '');

export function sortCandidates(items) {
  const parserName = (item) => item.label.replace(/\s\d[\w.+]*\s*(?:\[unpublished(?: sha256:[a-f0-9]+)?\]\s*)?(?:\([^)]*\))?\s*$/, '').replace(/\s*\([^)]*\)\s*$/, '').toLowerCase();
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
      return `${producerLabel} ${captureVersionLabel(source.implementation, source.version, metadata)} (${flavor})`;
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

export function selectedCaptureOutcome(cell, key, reference = 'golden') {
  if (cell?.variants) {
    const outcomes = cell.variants.filter(isApplicableCell).map(leaf => selectedCaptureOutcome(leaf, key, reference));
    return ['error', 'leak', 'diverge', 'missing', 'blocked', 'unsupported', 'uncompared', 'documented', 'pass']
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
        if (events.at(-1)?.kind === kind) events.at(-1).text += text;
        else events.push({ kind, text });
      } else if (kind === 'tool_call') {
        if (name) {
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
  const completed = events.filter((event) => event.kind !== 'tool_call' || event.complete);
  const malformed = new Set();
  const projections = new Map();
  for (const event of completed) {
    if (event.kind !== 'tool_call') continue;
    delete event.complete;
    if (isMapping(event.arguments) || Array.isArray(event.arguments)) continue;
    if (typeof event.arguments !== 'string') throw new Error('Unified argument fragment must be text, a mapping, or a list');
    if (!event.arguments.trim()) event.arguments = {};
    else {
      try { const decoded = parseJson(event.arguments); projections.set(event, decoded); if (!preserveArguments) event.arguments = decoded; }
      catch (error) {
        if (!(error instanceof SyntaxError)) throw error;
        malformed.add(event);
      }
    }
  }
  if (malformed.size && recordedAssembly) {
    const calls = completed.filter((event) => event.kind === 'tool_call');
    const recorded = recordedAssembly.filter((event) => getField(event, 'kind') === 'tool_call');
    // Old assembled records contain binary-float projections. Use them only to
    // align malformed fallback calls; exact raw bytes still drive comparison.
    const projected = recorded.map((event) => {
      const args = getField(event, 'arguments');
      if (typeof args !== 'string') return args;
      try { return parseJson(args); } catch (error) { if (!(error instanceof SyntaxError)) throw error; return args; }
    });
    if (calls.length === recorded.length && calls.every((event, index) => event.name === getField(recorded[index], 'name') && (sameValue(projections.has(event) ? projections.get(event) : event.arguments, projected[index]) || (malformed.has(event) && sameValue(projected[index], {}))))) {
      calls.forEach((event, index) => { if (malformed.has(event)) event.arguments = preserveArguments ? getField(recorded[index], 'arguments') : projected[index]; });
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
