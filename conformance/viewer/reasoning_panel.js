// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { tabCandidates, escapeHtml } from './candidates.js';
import { makeCell, missingCell } from './toolcalling.js';
import { getField, hasField, setField, isMapping, mappingEntries } from './mappings.js';
import { compareText } from './values.js';
import { implementationValue, replaceImplementation, normalizeImplementations, explanation, referenceRecord, reasoningFacts, reasoningComparisonEntries, reasoningLeakReason, reasoningOverviewStatus, reasoningStatisticsCategory } from './comparison.js';
import { unavailablePeers, reasoningParserCell } from './presentation.js';
import { catalogPeerAvailability, catalogFixtureApplicability } from './availability.js';
import { columnGroupKey } from './column_state.js';

function caseMeta(id, config) {
  const short = id.replace(/^REASONING\./, '');
  const index = config.groups.findIndex(([, cases]) => cases.includes(short));
  const label = index < 0 ? 'Other' : config.groups[index][0];
  const mode = short.split('.')[0];
  return { index, order: index < 0 ? 0 : config.groups[index][1].indexOf(short), label,
    key: columnGroupKey(label, mode), band: `case-band-${(index < 0 ? config.groups.length : index) % 2}` };
}

function outputBlock(block) {
  if (!isMapping(block)) return null;
  const result = {};
  for (const key of ['unavailable', 'error']) if (hasField(block, key)) result[key] = getField(block, key);
  if (hasField(block, 'reasoning_text') || hasField(block, 'normal_text')) Object.assign(result, { reasoning_text: getField(block, 'reasoning_text') || '', normal_text: getField(block, 'normal_text') || '' });
  const reason = explanation(block);
  if (reason) result.explanation = reason;
  return Object.keys(result).length ? result : null;
}

export async function reasoningPanel(catalog, reader, spec, { selected = null, reference = undefined } = {}) {
  const metadata = catalog.metadata;
  const config = metadata.reasoning_display;
  const descriptors = tabCandidates(catalog, spec);
  const chosen = selected === null ? descriptors : descriptors.filter(({ control }) => selected.includes(control.key));
  const referenceKey = reference === undefined ? selected?.[0] : reference;
  const referenceDescriptor = selected === null ? null : descriptors.find(({ control }) => control.key === referenceKey);
  if (selected !== null && referenceKey != null && !referenceDescriptor) throw new Error(`unknown reasoning reference: ${referenceKey}`);
  if (referenceDescriptor && !chosen.includes(referenceDescriptor)) throw new Error(`reference is not loaded: ${referenceKey}`);
  const baseline = selected === null ? 'dynamo_v1' : referenceDescriptor?.source.implementation ?? null;
  const selectedImplementations = selected === null ? null : chosen.map(({ source }) => source.implementation);
  const resource = catalog.sources.archives.find((item) => item.path === `conformance/fixtures/${spec.root}/inputs.tar.gz`);
  if (!resource) throw new Error('reasoning inputs are absent from catalog');
  const documents = new Map([...(await reader.legacyDocuments(resource))].map(([key, source]) => [key, { ...source, value: structuredClone(source.value) }]));
  for (const document of documents.values()) for (const [, record] of mappingEntries(getField(document.value, 'cases'))) {
    if (hasField(record, 'expected')) setField(record, 'expected', normalizeImplementations(getField(record, 'expected'), metadata.implementations));
  }
  for (const { source } of chosen) {
    const captured = await reader.legacy({ root: spec.root, mode: 'reasoning', implementation: source.implementation, version: source.version });
    for (const [key, capture] of captured) {
      const target = documents.get(key);
      if (!target) continue;
      for (const [id, record] of mappingEntries(getField(capture.value, 'cases'))) {
        const existing = getField(getField(target.value, 'cases'), id);
        const block = implementationValue(getField(record, 'expected'), source.implementation, metadata.implementations);
        if (!existing || block === undefined) continue;
        const expected = getField(existing, 'expected') || {};
        replaceImplementation(expected, source.implementation, block, metadata.implementations);
        setField(existing, 'expected', expected);
      }
    }
  }
  const families = new Map();
  const columnSet = new Set();
  for (const [, source] of [...documents].sort(([a], [b]) => compareText(a, b))) {
    const family = getField(source.value, 'family');
    if (!families.has(family)) families.set(family, { label: getField(source.value, 'model_label', family), cases: new Map() });
    for (const [id, record] of mappingEntries(getField(source.value, 'cases'))) {
      columnSet.add(id);
      families.get(family).cases.set(id, { record, source });
    }
  }
  const subs = [...columnSet].filter((id) => id.startsWith(`REASONING.${spec.mode}.`)).sort((a, b) => {
    const x = caseMeta(a, config), y = caseMeta(b, config);
    if ((x.index < 0) !== (y.index < 0)) return x.index < 0 ? 1 : -1;
    if (x.index >= 0) return x.index - y.index || x.order - y.order;
    const natural = (id) => { const [mode, number, sub = ''] = id.replace(/^REASONING\./, '').split('.'); return [mode === 'batch' ? 0 : 1, Number(number), sub]; };
    const left = natural(a), right = natural(b);
    return left[0] - right[0] || left[1] - right[1] || compareText(left[2], right[2]);
  });
  const parserInputs = catalog.sources.archives.find((item) => item.path === 'conformance/fixtures/toolcalling/fixtures-batch-v1/inputs.tar.gz');
  const parserLabels = parserInputs.model_labels;
  const canDisplay = (family) => Object.hasOwn(parserLabels, family) && families.has(config.parser_families[family]) && !config.hidden.includes(family);
  const top = [...config.top.filter(canDisplay), ...config.top_append.filter((family) => canDisplay(family) && !config.top.includes(family))];
  const excluded = new Set([...config.top, ...config.top_append, ...config.hidden]);
  const other = Object.keys(parserLabels).filter((family) => !excluded.has(family) && families.has(config.parser_families[family])).sort((a, b) => compareText(parserLabels[a].toLowerCase(), parserLabels[b].toLowerCase()));
  const mapped = new Set(Object.values(config.parser_families).filter((family) => families.has(family)));
  const only = [...families.keys()].filter((family) => !mapped.has(family)).sort(compareText);
  const groups = [];
  const columns = subs.map((id) => {
    const meta = caseMeta(id, config);
    if (groups.at(-1)?.key === meta.key) groups.at(-1).span++;
    else groups.push({ key: meta.key, label: meta.label, band: meta.band, span: 1 });
    const docId = id.replace(/^REASONING\./, '');
    return { sub: id, group_key: meta.key, band: meta.band, label: docId.split('.').slice(1).join('.'), desc: config.descriptions[docId] || config.descriptions[docId.split('.').slice(0, -1).join('.')] || '' };
  });
  const stats = { families: top.length + other.length + only.length, sub_cases: subs.length, slots: (top.length + other.length + only.length) * subs.length, real: 0, parity: 0, dynamo_only: 0, documented: 0, research: 0, errors: 0, na: 0, missing: 0 };
  const rows = [];
  const unavailable = await catalogPeerAvailability(catalog, spec) || unavailablePeers(new Map([...families].map(([family, value]) => [family, [...value.cases.values()].map((entry) => entry.record)])), metadata, true);
  const fixtureApplicability = selected === null ? null : await catalogFixtureApplicability(catalog, spec);
  for (const [section, rowFamilies] of [['Top-N models', top], ['Others', other], ['Reasoning-only', only]]) {
    if (!rowFamilies.length) continue;
    rows.push({ section, model_label: section, model_label_html: '', family: null, parser: null, cells: {} });
    for (const rowFamily of rowFamilies) {
      const family = section === 'Reasoning-only' ? rowFamily : config.parser_families[rowFamily];
      const displayFamily = rowFamily === 'glm47' && family === 'nemotron_deci' ? 'glm45' : family;
      const label = section === 'Reasoning-only' ? families.get(family).label : config.labels[rowFamily] || parserLabels[rowFamily] || rowFamily;
      const cells = {};
      for (const id of subs) {
        const meta = caseMeta(id, config);
        const entry = families.get(family).cases.get(id);
        const blocks = Object.fromEntries(chosen.map(({ control, source }) => [control.key, implementationValue(getField(entry?.record, 'expected'), source.implementation, metadata.implementations, null)]));
        const judgedRecord = entry ? referenceRecord(entry.record, referenceDescriptor, blocks) : null;
        const category = reasoningStatisticsCategory(judgedRecord, family, metadata, baseline, selectedImplementations);
        stats[category]++;
        if (!['na', 'missing'].includes(category)) stats.real++;
        if (!entry) { cells[id] = missingCell(id, family, meta.key, meta.band, `${id} — ${displayFamily}`); continue; }
        const record = entry.record;
        const expected = normalizeImplementations(getField(record, 'expected'), metadata.implementations);
        const facts = reasoningFacts(judgedRecord, family, metadata, baseline, selectedImplementations);
        const candidates = chosen.filter(({ control }) => hasField(expected, control.key)).map(({ control }) => ({
          key: control.key, label: control.label, impl: control.impl, version: control.version, parse_mode: spec.mode, block: outputBlock(getField(expected, control.key)), leak: Boolean(reasoningLeakReason(getField(expected, control.key), family, metadata)),
        }));
        let text = getField(record, 'model_text', null), chunks = null;
        if (!text && Array.isArray(getField(record, 'chunks')) && getField(record, 'chunks').every((chunk) => typeof chunk === 'string')) {
          text = getField(record, 'chunks').join('');
          chunks = getField(record, 'chunks').map((chunk) => ({ delta_text: chunk }));
        }
        const tooltip = { head: `${id} — ${displayFamily}`, description: getField(record, 'description') || '',
          input: { kind: text ? 'text' : null, text, chunks, family }, candidates, baseline: null,
          reasons: facts.filter((fact) => fact.impl !== baseline && fact.agrees === false && fact.reason).map((fact) => ({ impl: fact.impl, label: fact.impl === 'dynamo_v1' ? 'Dynamo Rust' : metadata.implementations.find((item) => item.key === fact.impl).label, reason: fact.reason, intentional: fact.intentional })),
          dynamo_notes: [], refs: [['Ref', getField(record, 'ref')], ['Spec ref', getField(record, 'spec_ref')]].filter(([, value]) => value), leak_note: null, na_note: hasField(record, 'expected') ? null : explanation(record) };
        cells[id] = makeCell({ case_id: id, family, sub: id, col_group: meta.key, band: meta.band, fixture_href: entry.source.member,
          status: baseline === null ? 'na' : reasoningOverviewStatus(judgedRecord, family, baseline, metadata), cmp: reasoningComparisonEntries(record, family, metadata), facts, tooltip });
        if (fixtureApplicability) cells[id].applicable = fixtureApplicability.get(`${family}\0${id}`)?.applicable ?? false;
      }
      rows.push({ section: null, model_label: label, model_label_html: escapeHtml(label).replaceAll('‡', '<span class="parser-suffix">‡</span>'), family, parser: { html: reasoningParserCell(section === 'Reasoning-only' ? null : rowFamily, family, unavailable, metadata) }, cells });
    }
  }
  const glossary = [];
  for (const column of columns) {
    const meta = caseMeta(column.sub, config);
    if (!glossary.length || glossary.at(-1).key !== meta.key) glossary.push({ key: meta.key, label: meta.label, rows: [] });
    const docId = column.sub.replace(/^REASONING\./, '');
    glossary.at(-1).rows.push([docId, config.descriptions[docId] || '']);
  }
  return { id: spec.id, kind: 'reasoning', mode: spec.mode, columns, column_groups: groups, rows, stats, glossary: glossary.map(({ key, ...group }) => group), candidates: descriptors.map((descriptor) => descriptor.control) };
}
