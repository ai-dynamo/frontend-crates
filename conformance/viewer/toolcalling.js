// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { legacyCases } from './legacy.js';
import { tabCandidates, escapeHtml, sortCandidates } from './candidates.js';
import { implementationValue, replaceImplementation, comparisonEntries, comparisonFacts, blockLeaks, declaredLeakTokens, explanation, normalizeImplementations, canonicalToolOutput, legacyOverviewStatus, legacyStatisticsCategory, referenceRecord } from './comparison.js';
import { getField, hasField, setField, isMapping, mappingEntries } from './mappings.js';
import { compareLegacyVersions } from './model.js';
import { sameValue, compareText } from './values.js';
import { foldNullVariants } from './folds.js';
import { unavailablePeers, toolParserCell } from './presentation.js';
import { catalogPeerAvailability, catalogFixtureApplicability } from './availability.js';
import { columnGroupKey } from './column_state.js';
import { clearLegacyImplementation } from './resources.js';

export function makeCell(fields = {}) {
  return { kind: 'cell', case_id: null, family: null, sub: '', col_group: '', band: '', fixture_href: null, status: 'na', cmp: null, facts: [], known_divergence: false, tooltip: null, ...fields };
}

export function missingCell(sub, family, colGroup, band, head) {
  return makeCell({ kind: 'missing', sub, family, col_group: colGroup, band, status: 'missing',
    tooltip: { head, description: '', input: { kind: null }, candidates: [], baseline: null, reasons: [], dynamo_notes: [], refs: [], leak_note: null, na_note: 'No fixture coverage for this case.' } });
}

function metadataForSub(metadata, mode, sub) {
  const parent = sub.split('.')[0];
  const key = metadata.legacy_null_groups.includes(parent) ? parent : sub;
  const groups = metadata.legacy_case_groups[mode];
  const index = groups.findIndex(([, members]) => members.includes(key));
  const label = index < 0 ? 'Other' : groups[index][0];
  return { label, group_key: columnGroupKey(label), band: `case-band-${(index < 0 ? groups.length : index) % 2}`, index, order: index < 0 ? -1 : groups[index][1].indexOf(key) };
}

function subOrder(metadata, mode, left, right) {
  const a = metadataForSub(metadata, mode, left), b = metadataForSub(metadata, mode, right);
  if ((a.index < 0) !== (b.index < 0)) return a.index < 0 ? 1 : -1;
  if (a.index >= 0) return a.index - b.index || a.order - b.order || compareText(left, right);
  const natural = (sub) => { const [parent, ...suffix] = sub.split('.'); const [group, number = '0'] = parent.split('-'); return [Number(group), Number(number), suffix.join('.')]; };
  const x = natural(left), y = natural(right);
  return x[0] - y[0] || x[1] - y[1] || compareText(x[2], y[2]);
}

export function explicitSchemas(tools) {
  if (!Array.isArray(tools)) return null;
  const schemas = tools.filter((tool) => typeof getField(tool, 'name') === 'string' && hasField(tool, 'parameters') && getField(tool, 'parameters') !== null)
    .map((tool) => ({ name: getField(tool, 'name'), parameters: getField(tool, 'parameters') }));
  return schemas.length ? schemas : null;
}

function schemasByFamily(families, records, sub) {
  const variants = [];
  for (const family of families) {
    const tools = explicitSchemas(getField(records.get(`${family}\0${sub}`)?.record, 'tools'));
    if (!tools) continue;
    let variant = variants.find((entry) => sameValue(entry.tools, tools));
    if (!variant) { variant = { families: [], tools }; variants.push(variant); }
    variant.families.push(family);
  }
  return variants;
}

function outputBlock(block) {
  if (!isMapping(block)) return null;
  const result = {};
  for (const key of ['unavailable', 'exception', 'error']) if (hasField(block, key)) result[key] = getField(block, key);
  if (hasField(block, 'calls') || hasField(block, 'normal_text')) Object.assign(result, { calls: getField(block, 'calls') || [], normal_text: getField(block, 'normal_text') || '' });
  const reason = explanation(block);
  if (reason) result.explanation = reason;
  return Object.keys(result).length ? result : null;
}

function inputModel(record, metadata) {
  const family = getField(record, '__family');
  const chunks = getField(record, 'chunks');
  if (Array.isArray(chunks) && chunks.length) return { kind: 'chunks', text: null, family, chunks: chunks.filter(isMapping).map((chunk) => ({
    delta_text: getField(chunk, 'delta_text', ''), delta_token_ids: getField(chunk, 'delta_token_ids', null), finish_reason: getField(chunk, 'finish_reason', null),
    expected: normalizeImplementations(getField(chunk, 'expected') || {}, metadata.implementations),
    normal_text: isMapping(getField(chunk, 'normal_text')) ? normalizeImplementations(getField(chunk, 'normal_text'), metadata.implementations) : getField(chunk, 'normal_text', null),
  })) };
  const text = getField(record, 'model_text');
  return { kind: typeof text === 'string' ? 'text' : null, text: typeof text === 'string' ? text : null, chunks: null, family };
}

function linkify(text) {
  let offset = 0, result = '';
  for (const match of String(text).matchAll(/https?:\/\/[^\s<>"']+/g)) {
    const url = match[0].replace(/[.,;:!?\)\]\}]+$/, '');
    result += escapeHtml(text.slice(offset, match.index));
    result += `<a href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(url)}</a>${escapeHtml(match[0].slice(url.length))}`;
    offset = match.index + match[0].length;
  }
  return result + escapeHtml(text.slice(offset));
}

function overlayExpected(record, hasText, metadata) {
  const source = normalizeImplementations(record, metadata.implementations);
  const result = {};
  const dynamo = getField(source, 'dynamo_v2');
  result.dynamo_v2 = isMapping(dynamo) && (hasField(dynamo, 'calls') || hasField(dynamo, 'normal_text'))
    ? { calls: getField(dynamo, 'calls') || [], normal_text: getField(dynamo, 'normal_text') || '' }
    : { unavailable: hasText ? 'Dynamo parser v2 stream parser not yet implemented for this family' : 'No batch model_text for this case.' };
  for (const impl of metadata.implementations.filter((item) => !item.key.startsWith('dynamo_'))) {
    const block = getField(source, impl.key);
    result[impl.key] = !isMapping(block) ? { unavailable: 'No batch-on-stream capture for this engine.' } : hasField(block, 'unavailable') ? { unavailable: getField(block, 'unavailable') } : {
      calls: getField(block, 'calls') || [], normal_text: getField(block, 'normal_text') || '',
      explanation: `Captured from the ${impl.label} streaming parser on the batch text. Streaming output differs from Dynamo parser v2 token-incremental behavior by design (text vs token streaming).`,
    };
  }
  return result;
}

function noteFor(metadata, family, id, field) { return metadata.divergence_notes?.[family]?.[id]?.[field]; }

// Input records are shared; each version's observed block is resolved separately.
// This permits a browser selection to fetch only its requested histories while
// Node asks for the complete candidate set through the same builder.
export async function toolcallingPanel(catalog, reader, spec, { selected = null, reference = undefined } = {}) {
  const metadata = catalog.metadata;
  const mode = spec.mode === 'stream' ? 'streamv1' : spec.mode;
  const inputResource = catalog.sources.archives.find((item) => item.path === `conformance/fixtures/${spec.root}/inputs.tar.gz`);
  if (!inputResource) throw new Error(`missing input archive: ${spec.id}`);
  const inputDocuments = await reader.legacyDocuments(inputResource);
  const inputs = legacyCases(inputDocuments, mode, metadata);
  const hasGolden = [...inputs.cases.values()].some((entry) => Boolean(getField(entry.record, 'golden')));
  const descriptors = tabCandidates(catalog, spec, { golden: hasGolden });
  const chosen = descriptors.filter((item) => item.source.implementation !== 'golden' && (!selected || selected.includes(item.control.key)));
  if (spec.mode === 'stream') {
    const batchSpec = metadata.tabs.find((item) => item.id === 'tab-toolcalling-batch');
    const batchCandidates = tabCandidates(catalog, batchSpec).filter((item) => item.source.mode === 'batch');
    const implementations = new Set(['dynamo_v1', ...chosen.map((item) => item.source.implementation)]);
    const batchDocuments = new Map();
    for (const impl of implementations) {
      const versions = batchCandidates.filter((item) => item.source.implementation === impl).sort((a, b) => compareLegacyVersions(b.source.version, a.source.version));
      if (!versions.length) continue;
      const docs = await reader.legacy({ root: batchSpec.root, mode: 'batch', implementation: impl, version: versions[0].source.version });
      for (const [key, value] of docs) {
        if (!batchDocuments.has(key)) batchDocuments.set(key, { ...value, value: structuredClone(value.value) });
        else {
          const base = batchDocuments.get(key).value;
          for (const [id, item] of mappingEntries(getField(value.value, 'cases'))) {
            const block = implementationValue(getField(item, 'expected'), impl, metadata.implementations);
            if (block === undefined) continue;
            const target = getField(getField(base, 'cases'), id);
            if (!target) continue;
            const expected = getField(target, 'expected') || {};
            replaceImplementation(expected, impl, block, metadata.implementations); setField(target, 'expected', expected);
          }
        }
      }
    }
    const withBatchReference = legacyCases(new Map([...batchDocuments, ...inputDocuments]), mode, metadata);
    for (const [key, entry] of inputs.cases) setField(entry.record, 'batch_expected', getField(withBatchReference.cases.get(key)?.record, 'batch_expected') || {});
  }
  const snapshots = new Map();
  const overlay = new Map();
  for (const descriptor of chosen) {
    const { source, control } = descriptor;
    if (source.mode === 'batch-on-stream') {
      const resource = catalog.sources.archives.find((item) => item.path === source.archive);
      for (const document of (await reader.legacyDocuments(resource)).values()) {
        const family = getField(document.value, 'family');
        for (const [id, block] of mappingEntries(getField(document.value, 'cases'))) overlay.set(`${family}\0${id}`, block);
      }
      continue;
    }
    const docs = await reader.legacy({ root: spec.root, mode: spec.mode, implementation: source.implementation, version: source.version });
    snapshots.set(control.key, legacyCases(docs, mode, metadata, mode === 'streamv1' ? [source.implementation] : null));
  }
  const current = new Map([...inputs.cases].map(([key, value]) => [key, { ...value, record: structuredClone(value.record) }]));
  const latest = new Map();
  for (const descriptor of chosen.filter((item) => item.source.mode !== 'batch-on-stream')) {
    const prior = latest.get(descriptor.source.implementation);
    const oldest = selected === null && metadata.selection_policy[spec.mode].inventory_defaults?.[descriptor.source.implementation] === 'oldest';
    if (!prior || compareLegacyVersions(descriptor.source.version, prior.source.version) * (oldest ? -1 : 1) > 0) latest.set(descriptor.source.implementation, descriptor);
  }
  for (const [impl, descriptor] of latest) {
    for (const [key, captured] of snapshots.get(descriptor.control.key).cases) {
      const target = current.get(key);
      if (!target) continue;
      const chunks = getField(target.record, 'chunks'), capturedChunks = getField(captured.record, 'chunks');
      if (spec.mode === 'stream' || Array.isArray(chunks) && Array.isArray(capturedChunks)) clearLegacyImplementation(target.record, impl, metadata.implementations);
      const block = getField(getField(captured.record, 'expected'), impl);
      const expected = getField(target.record, 'expected') || {};
      if (block !== undefined) { replaceImplementation(expected, impl, block, metadata.implementations); setField(target.record, 'expected', expected); }
      if (Array.isArray(chunks) && Array.isArray(capturedChunks)) chunks.forEach((chunk, index) => {
        const capturedChunk = capturedChunks[index];
        for (const field of ['expected', 'normal_text']) {
          if (isMapping(getField(capturedChunk, field)) && !hasField(chunk, field)) setField(chunk, field, {});
          const block = getField(getField(capturedChunk, field), impl);
          if (block !== undefined) {
            const value = getField(chunk, field) || {};
            setField(value, impl, block); setField(chunk, field, value);
          }
        }
      });
    }
  }
  const suite = mode === 'streamv1' ? 'toolcalling.stream' : 'toolcalling.batch';
  const retired = metadata.legacy_taxonomy.retired_subcases?.[suite] || [];
  const hiddenGroups = metadata.legacy_taxonomy.hidden_subcase_groups?.[suite] || [];
  const subs = [...new Set([...current.values()].map((entry) => entry.sub))].filter((sub) => !retired.includes(sub) && !hiddenGroups.includes(sub.split('.')[0])).sort((a, b) => subOrder(metadata, mode, a, b));
  const families = [...new Set([...current.values()].map((entry) => entry.family))].filter((family) => !metadata.legacy_display.hidden.includes(family));
  const label = (family) => metadata.legacy_display.labels[family] || inputs.labels[family] || family;
  const top = metadata.legacy_display.top.filter((family) => families.includes(family));
  const other = families.filter((family) => !top.includes(family)).sort((a, b) => compareText(label(a).toLowerCase(), label(b).toLowerCase()));
  const tokens = declaredLeakTokens(metadata.markers);
  const impls = metadata.implementations.filter((item) => item.modes.includes(spec.mode));
  const defaultBaseline = spec.mode === 'batch' ? 'dynamo_v1' : 'dynamo_v2';
  const referenceKey = selected === null ? null : reference === undefined ? selected[0] ?? null : reference;
  const referenceDescriptor = descriptors.find((item) => item.control.key === referenceKey);
  if (referenceKey !== null && !referenceDescriptor) throw new Error(`unknown toolcalling reference: ${referenceKey}`);
  if (referenceDescriptor && !selected.includes(referenceKey)) throw new Error(`reference is not loaded: ${referenceKey}`);
  const baseline = referenceDescriptor ? referenceDescriptor.source.implementation : defaultBaseline;
  const groups = [];
  const columns = subs.map((sub) => {
    const meta = metadataForSub(metadata, mode, sub);
    if (groups.at(-1)?.key === meta.group_key) groups.at(-1).span++;
    else groups.push({ key: meta.group_key, label: meta.label, band: meta.band, span: 1 });
    const descriptions = metadata.legacy_display.descriptions[mode];
    const column = { sub, group_key: meta.group_key, band: meta.band, label: sub, desc: descriptions[sub] || descriptions[sub.split('.')[0]] || '' };
    const schemas = schemasByFamily([...top, ...other], current, sub);
    if (schemas.length) column.schemas = schemas;
    return column;
  });
  const rows = [];
  const judgedRecords = new Map();
  const recordsByFamily = new Map();
  for (const entry of current.values()) {
    if (!recordsByFamily.has(entry.family)) recordsByFamily.set(entry.family, []);
    recordsByFamily.get(entry.family).push(entry.record);
  }
  const unavailable = await catalogPeerAvailability(catalog, spec) || unavailablePeers(recordsByFamily, metadata);
  const fixtureApplicability = selected === null ? null : await catalogFixtureApplicability(catalog, spec);
  for (const [section, familyList] of [['Top-N models', top], ['Others', other]]) {
    if (!familyList.length) continue;
    rows.push({ section, model_label: section, model_label_html: '', family: null, parser: null, cells: {} });
    for (const family of familyList) {
      const cells = {};
      for (const sub of subs) {
        const entry = current.get(`${family}\0${sub}`);
        const meta = metadataForSub(metadata, mode, sub);
        if (!entry) { setField(cells, sub, missingCell(sub, family, meta.group_key, meta.band, `TOOLCALLING.${mode}.${sub}`)); continue; }
        const record = entry.record;
        const blocks = {};
        const candidates = [];
        let missingReference = false;
        let knownDivergence = false;
        const defect = noteFor(metadata, family, getField(record, '__case_id'), 'baseline_defect');
        if (defect) setField(record, 'dynamo_note', `${getField(record, 'dynamo_note') ? `${getField(record, 'dynamo_note')}\n` : ''}${defect}`);
        for (const descriptor of chosen) {
          const { source, control } = descriptor;
          let block;
          if (source.mode === 'batch-on-stream') {
            const id = getField(record, '__case_id');
            const sample = overlay.get(`${family}\0${id}`) || (id.endsWith('.a') ? overlay.get(`${family}\0${id.slice(0, -2)}`) : null);
            if (!sample) {
              // An absent observation is recorded missing; selection loading was
              // validated above and must not become a transport failure here.
              if (referenceDescriptor?.control.key === control.key) missingReference = true;
              continue;
            }
            const expected = overlayExpected(sample, hasField(record, 'model_text'), metadata);
            block = expected[source.implementation];
            const note = noteFor(metadata, family, id, 'stream_vs_batch') || (id.endsWith('.a') ? noteFor(metadata, family, id.slice(0, -2), 'stream_vs_batch') : null);
            if (note) {
              knownDivergence = true;
              if (source.implementation === 'dynamo_v2' && !hasField(block, 'unavailable')) block = { ...block, explanation: note };
            }
          } else {
            block = getField(getField(snapshots.get(control.key).cases.get(`${family}\0${sub}`)?.record, 'expected'), source.implementation, null);
            if (spec.mode === 'stream') {
              const resources = catalog.sources.archives.filter((item) => item.path.startsWith(`conformance/fixtures/${spec.root}/${source.implementation}-`));
              const covered = resources.filter((item) => compareLegacyVersions(item.path.split('/').at(-1).slice(source.implementation.length + 1, -7), source.version) <= 0).flatMap((item) => item.families);
              if (source.implementation.startsWith('dynamo_') && !covered.includes(family)) block = null;
              const note = noteFor(metadata, family, getField(record, '__case_id'), source.implementation === 'dynamo_v2' ? 'v2' : 'jail');
              if (source.implementation.startsWith('dynamo_') && isMapping(block) && note) block = { ...block, explanation: note };
            }
          }
          blocks[control.key] = block;
          candidates.push({ key: control.key, label: control.label, impl: control.impl, version: spec.mode === 'batch' ? null : source.version, parse_mode: control.parse_mode, leak: blockLeaks(block, tokens), block: outputBlock(block) });
        }
        const golden = getField(record, 'golden');
        if (golden) {
          blocks.golden = golden;
          candidates.push({ key: 'golden', label: 'GOLDEN (oracle)', impl: 'golden', version: null, parse_mode: null, leak: blockLeaks(golden, tokens), block: outputBlock(golden) });
        } else if (referenceDescriptor?.source.implementation === 'golden') {
          missingReference = true;
        }
        const expected = getField(record, 'expected') || {};
        const factRecord = referenceRecord(record, selected !== null ? referenceDescriptor : null, blocks);
        judgedRecords.set(`${family}\0${sub}`, factRecord);
        const dyn = getField(getField(factRecord, 'expected'), baseline);
        const facts = comparisonFacts(factRecord, impls, baseline, tokens, metadata.recording.rustUnavailable);
        const status = legacyOverviewStatus(factRecord, baseline, tokens, spec.mode === 'stream');
        const input = inputModel(record, metadata);
        const applicable = fixtureApplicability ? Boolean(fixtureApplicability.get(`${family}\0${sub}`)?.present || golden)
          : isMapping(dyn) || Boolean(golden) || (selected !== null && candidates.some((candidate) => candidate.block !== null));
        const baselineBlock = getField(getField(record, 'batch_expected'), 'dynamo_v1');
        const note = getField(record, 'dynamo_note');
        const tooltip = { head: `${getField(record, '__case_id')} — ${family}`, description: getField(record, 'description') || '', input,
          candidates: applicable ? sortCandidates(candidates) : [], baseline: applicable && canonicalToolOutput(baselineBlock) ? { impl: 'dynamo_v1', label: 'Dynamo v1 Rust batch parser', block: outputBlock(baselineBlock) } : null,
          reasons: applicable ? facts.filter((fact) => fact.impl !== baseline && fact.agrees === false && fact.reason).map((fact) => ({ impl: fact.impl, label: metadata.implementations.find((impl) => impl.key === fact.impl).label, reason: fact.reason, intentional: fact.intentional })) : [],
          dynamo_notes: applicable && note ? [['Dynamo recovery contract', linkify(String(note))]] : [],
          refs: applicable ? [['Ref', getField(record, 'ref')], ['Spec ref', getField(record, 'spec_ref')]].filter(([, value]) => value) : [],
          leak_note: applicable && blockLeaks(dyn, tokens) && explanation(dyn) ? String(explanation(dyn)) : null,
          na_note: applicable ? null : explanation(record) };
        const cell = makeCell({ case_id: getField(record, '__case_id'), family, sub, col_group: meta.group_key, band: meta.band,
          fixture_href: entry.source.member, status: missingReference ? 'missing' : status, cmp: Object.keys(blocks).length ? comparisonEntries(blocks, tokens) : null, facts, known_divergence: knownDivergence, tooltip });
        if (fixtureApplicability) cell.applicable = fixtureApplicability.get(`${family}\0${sub}`)?.applicable ?? false;
        if (applicable) cell.red_on_diff = Boolean(golden);
        setField(cells, sub, cell);
      }
      rows.push({ section: null, model_label: label(family), model_label_html: escapeHtml(label(family)).replaceAll('‡', '<span class="parser-suffix">‡</span>'), family, parser: { html: toolParserCell(family, mode, unavailable, metadata) }, cells });
    }
  }
  const glossary = [];
  for (const column of columns) {
    const label = metadataForSub(metadata, mode, column.sub).label;
    let group = glossary.find((item) => item.label === label);
    if (!group) { group = { label, rows: [] }; glossary.push(group); }
    group.rows.push([column.sub, column.desc]);
  }
  const stats = { families: families.length, sub_cases: subs.length, slots: families.length * subs.length, real: 0, parity: 0, dynamo_only: 0, documented: 0, research: 0, errors: 0, na: 0, missing: 0, todo: 0 };
  for (const family of [...top, ...other]) for (const sub of subs) {
    const category = legacyStatisticsCategory(judgedRecords.get(`${family}\0${sub}`), spec.mode === 'stream' ? metadata.implementations : impls, baseline, tokens, spec.mode === 'stream');
    stats[category]++;
    if (!['na', 'missing'].includes(category)) stats.real++;
  }
  const tab = { id: spec.id, kind: 'toolcalling', mode, columns, column_groups: groups, rows, candidates: descriptors.map((item) => item.control), glossary, stats };
  foldNullVariants(tab, metadata);
  return tab;
}
