// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createNodeResourceStore } from './data.js';
import { FixtureResources } from './resources.js';
import { buildPage, buildTab } from './page_model.js';
import { buildStatus } from './status.js';
import { tabCandidates } from './candidates.js';
import { compareLegacyVersions, resolveUnified } from './model.js';
import { leafCells } from './folds.js';
import { columnKeysFor } from './column_state.js';
const root = new URL('../../', import.meta.url);
const catalog = JSON.parse(await readFile(new URL('conformance/viewer-catalog.json', root)));
const reader = new FixtureResources(catalog, await createNodeResourceStore(root));
const page = await buildPage(catalog, reader);
const columnKeys = new Set(columnKeysFor(catalog.metadata));
assert.equal(page.schema, 1);
assert.equal(Object.hasOwn(page, 'strings'), false);
assert.deepEqual(page.tabs.map(tab => tab.id), catalog.metadata.tabs.map(tab => tab.id));
assert.equal(page.tabs.filter(tab => tab.active).length, 1);
let cells = 0, leaves = 0;
for (const tab of page.tabs) {
  const spec = catalog.metadata.tabs.find(spec => spec.id === tab.id);
  const descriptors = tabCandidates(catalog, spec, { golden: tab.candidates.some(candidate => candidate.key === 'golden') });
  assert.deepEqual(tab.candidates, descriptors.map(descriptor => descriptor.control));
  assert.equal(tab.candidates.filter(candidate => candidate.default_bucket === 'A').length, 1);
  assert.equal(new Set(tab.candidates.map(candidate => candidate.key)).size, tab.candidates.length);
  assert.equal(new Set(tab.columns.map(column => column.sub)).size, tab.columns.length);
  assert.equal(tab.column_groups.reduce((sum, group) => sum + group.span, 0), tab.columns.length);
  for (const group of tab.column_groups) assert.ok(columnKeys.has(group.key), `${tab.id}: column group ${group.key} must exist before the tab loads`);
  assert.equal(Object.hasOwn(tab, 'cand_meta'), false);
  const selectable = new Set(tab.candidates.map(candidate => candidate.key));
  selectable.add('golden');
  const sources = new Map();
  for (const descriptor of descriptors) {
    if (descriptor.source.implementation === 'golden') continue;
    assert.ok(descriptor.source.version);
    const owner = `${descriptor.source.implementation}@${descriptor.source.mode}`;
    const prior = sources.get(owner);
    if (prior) assert.ok(compareLegacyVersions(prior, descriptor.source.version) >= 0);
    sources.set(owner, descriptor.source.version);
    assert.ok(!descriptor.control.label.includes('+source.'));
    if (spec.mode === 'stream') assert.ok(!descriptor.source.version.match(/\.patch\d+$/));
  }
  for (const row of tab.rows) {
    if (row.section) continue;
    assert.ok(row.family && row.model_label);
    assert.equal(Object.keys(row.cells).length, tab.columns.length);
    for (const column of tab.columns) {
      const cell = row.cells[column.sub]; cells++;
      assert.ok(cell);
      assert.ok(['ok', 'problem', 'na', 'missing', 'todo'].includes(cell.status) || cell.kind === 'blank');
      for (const [key, value] of Object.entries(cell.cmp || {})) {
        assert.ok(selectable.has(key), `${tab.id}:${row.family}:${column.sub}: unselectable comparison ${key}`);
        for (const field of ['na', 'err', 'leak']) assert.ok([0, 1].includes(value[field]));
        assert.ok(Object.hasOwn(value, 'sig'));
      }
      const list = [...leafCells({ cells: { selected: cell } }).values()]; leaves += list.length;
      const ids = list.filter(cell => cell.case_id).map(cell => cell.case_id);
      assert.equal(new Set(ids).size, ids.length, `${tab.id}:${row.family}:${column.sub}: repeated folded leaf`);
      for (const leaf of list) {
        for (const fact of leaf.facts || []) {
          assert.ok(fact.impl && typeof fact.status === 'string');
          assert.ok(typeof fact.leak === 'boolean');
        }
        for (const candidate of leaf.tooltip?.candidates || []) {
          assert.ok(selectable.has(candidate.key));
          assert.ok(candidate.label);
          const block = candidate.block;
          if (!block) continue;
          if ((candidate.impl || candidate.key).startsWith('dynamo')) for (const call of block.calls || []) assert.ok(!/^(.+)\1$/.test(call.name || ''));
        }
      }
    }
  }
}
const stream = page.tabs.find(tab => tab.id === 'tab-toolcalling-streamv1');
assert.ok(stream.rows.find(row => row.family === 'deepseek_v4').parser.html.includes('dsml.rs'));
assert.ok(stream.rows.find(row => row.family === 'harmony').parser.html.includes('harmony.rs'));
for (const tab of page.tabs.filter(tab => tab.kind === 'toolcalling')) {
  assert.ok(tab.columns.every(column => !/^null/.test(column.sub)));
  if (tab.id === 'tab-toolcalling-streamv1') for (const sub of ['7.g','7.h','7.i','7.j','7.k','7.l']) assert.equal(tab.columns.find(column => column.sub === sub)?.group_key, 'args');
  assert.ok(tab.columns.some(column => column.schemas?.length));
}
const unified = page.tabs.find(tab => tab.kind === 'unified');
assert.ok(unified.columns.every(column => column.label.includes('-')));
assert.ok(unified.columns.some(column => column.variant_labels?.length || column.schemas?.length));
assert.ok(unified.candidates.some(candidate => candidate.key === 'dynamo@0.6.0'));
assert.ok(unified.candidates.some(candidate => candidate.key === 'vllm_rust'));
const sparseProvenance = resolveUnified([
  { implementation: 'dynamo', version: '0.9.0', path: 'first.yaml', bytes: new Uint8Array(), document: {
    changes: { 'UNIFIED.provenance': { case_key: 'provenance', stimulus: {}, observation: { events: [] } } },
    document: {}, document_overrides: { 'UNIFIED.provenance': { parser_path: 'unified', capture_provenance: { captured_with: { dynamo: '0.8.0' }, inherited_from: null } } },
  } },
  { implementation: 'dynamo', version: '1.0.0', path: 'next.yaml', bytes: new Uint8Array(), document: {
    changes: { 'UNIFIED.provenance': { stimulus: { updated: true } } }, document: {},
  } },
], '1.0.0');
assert.equal(sparseProvenance.get('UNIFIED.provenance').value.document.captured_with.dynamo, '0.8.0');
assert.equal(Object.hasOwn(sparseProvenance.get('UNIFIED.provenance').value.document, 'inherited_from'), false);
for (const original of page.tabs) {
  const reference = original.candidates.find(candidate => candidate.default_bucket === 'A').key;
  const selected = await buildTab(catalog, reader, original.id, { selected: [reference], reference });
  assert.deepEqual(selected.rows.map(row => [row.section, row.family, row.model_label_html, row.parser?.html]), original.rows.map(row => [row.section, row.family, row.model_label_html, row.parser?.html]), `${original.id}: selected loading must preserve presentation`);
  const leafIdentities = tab => tab.rows.filter(row => !row.section).map(row => [row.model_label, [...leafCells(row).keys()]]);
  assert.deepEqual(leafIdentities(selected), leafIdentities(original), `${original.id}: default selection must retain folded leaves`);
  assert.equal(selected.stats.fixture_cases, original.stats.fixture_cases, `${original.id}: default selection must preserve fixture accounting`);
  assert.deepEqual(buildStatus({ ...page, tabs: [selected] }).reports, buildStatus({ ...page, tabs: [original] }).reports, `${original.id}: selected Reference status parity`);
  const spec = catalog.metadata.tabs.find(spec => spec.id === original.id);
  const descriptors = tabCandidates(catalog, spec, { golden: original.candidates.some(candidate => candidate.key === 'golden') });
  const groups = new Map();
  for (const descriptor of descriptors.filter(candidate => candidate.source.version)) {
    const identity = `${descriptor.source.implementation}@${descriptor.source.mode}`;
    if (!groups.has(identity)) groups.set(identity, []);
    groups.get(identity).push(descriptor);
  }
  const selectedDescriptors = new Map();
  for (const versions of groups.values()) {
    versions.sort((a, b) => compareLegacyVersions(b.source.version, a.source.version));
    for (const descriptor of [versions[0], versions.at(-1), versions[Math.floor(versions.length / 2)]]) {
      if (descriptor.control.key !== reference) selectedDescriptors.set(descriptor.control.key, descriptor);
    }
  }
  for (const { control: historical } of selectedDescriptors.values()) {
    const historicalTab = await buildTab(catalog, reader, original.id, { selected: [historical.key], reference: historical.key });
    const label = `${original.id}:${historical.key}`;
    assert.deepEqual(historicalTab.rows.map(row => [row.section, row.family, row.model_label_html, row.parser?.html]), original.rows.map(row => [row.section, row.family, row.model_label_html, row.parser?.html]), `${label}: historical selection must preserve presentation`);
    assert.deepEqual(leafIdentities(historicalTab), leafIdentities(original), `${label}: historical selection must retain folded leaves`);
    assert.equal(historicalTab.stats.fixture_cases, original.stats.fixture_cases, `${label}: capture availability cannot change fixture accounting`);
    const statusWithReference = tab => buildStatus({ ...page, tabs: [{ ...tab, candidates: tab.candidates.map(candidate => ({ ...candidate, default_bucket: candidate.key === historical.key ? 'A' : 'C' })) }] }).reports;
    assert.deepEqual(statusWithReference(historicalTab), statusWithReference(original), `${label}: historical Reference status parity`);
    const relation = cell => {
      const referenceEntry = cell.cmp?.[historical.key], goldenEntry = cell.cmp?.golden;
      const flags = entry => entry ? [entry.na, entry.err, entry.leak] : null;
      return [flags(referenceEntry), flags(goldenEntry), referenceEntry && goldenEntry ? JSON.stringify(referenceEntry.sig) === JSON.stringify(goldenEntry.sig) : null, Boolean(cell.red_on_diff)];
    };
    for (let rowIndex = 0; rowIndex < original.rows.length; rowIndex++) {
      const sourceRow = original.rows[rowIndex], selectedRow = historicalTab.rows[rowIndex];
      if (sourceRow.section) continue;
      for (const sub of Object.keys(sourceRow.cells)) assert.deepEqual(relation(selectedRow.cells[sub]), relation(sourceRow.cells[sub]), `${label}:${sourceRow.family}:${sub}: folded comparison relation`);
      const sourceLeaves = leafCells(sourceRow), selectedLeaves = leafCells(selectedRow);
      for (const [sub, sourceCell] of sourceLeaves) assert.deepEqual(relation(selectedLeaves.get(sub)), relation(sourceCell), `${label}:${sourceRow.family}:${sub}: leaf comparison relation`);
    }
  }
  const empty = await buildTab(catalog, reader, original.id, { selected: [], reference: null });
  assert.deepEqual(empty.rows.map(row => [row.section, row.family, row.model_label_html, row.parser?.html]), original.rows.map(row => [row.section, row.family, row.model_label_html, row.parser?.html]), `${original.id}: empty Reference must preserve presentation`);
  assert.deepEqual(leafIdentities(empty), leafIdentities(original), `${original.id}: empty Reference must retain folded leaves`);
  assert.equal(empty.stats.fixture_cases, original.stats.fixture_cases, `${original.id}: empty Reference must preserve fixture accounting`);
}
reader.dispose();
console.log(`Shared matrix structural contracts passed: ${page.tabs.length} tabs, ${cells} visible cells, ${leaves} leaf appearances`);
