// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createNodeResourceStore } from './data.js';
import './development_parser.js';
import { FixtureResources } from './resources.js';
import { buildPage, buildTab, pageMetadata } from './page_model.js';
import { buildStatus } from './status.js';
import { tabCandidates, selectedUnifiedFamilies } from './candidates.js';
import { classifyUnified, compareLegacyVersions, resolveUnified } from './model.js';
import { leafCells } from './folds.js';
import { columnKeysFor } from './column_state.js';
import { tabPresentation } from './presentation.js';
import { sameValue } from './values.js';
import { getField } from './mappings.js';
import { selectedCaptureOutcome } from './selection_comparison.js';
const modelContracts = JSON.parse(await readFile(new URL('../utils/tests/fixtures/model_contracts.json', import.meta.url)));
const root = process.argv[3] || new URL('../../', import.meta.url);
const catalog = JSON.parse(await readFile(process.argv[2] || new URL('conformance/viewer-catalog.json', root)));
// The retained batch presentation flag is separate from the browser's active tab.
// Validate metadata contracts before constructing the expensive matrix.
const legacyDefaultIds = catalog.metadata.tabs.filter(spec => spec.id === 'tab-toolcalling-batch').map(spec => spec.id);
for (const spec of catalog.metadata.tabs) {
  const descriptors = tabCandidates(catalog, spec);
  assert.equal(descriptors.filter(descriptor => descriptor.control.default_bucket === 'A').length,
    1, `${spec.id}: retained tabs need exactly one Reference`);
  assert.equal(Boolean(tabPresentation(spec, catalog, descriptors).active), spec.id === 'tab-toolcalling-batch');
}
async function newReader() {
  return new FixtureResources(catalog, await createNodeResourceStore(root, { allowSourceFiles: !process.argv[3] }));
}
async function checkPageOrchestration() {
  const reader = await newReader();
  try {
    const page = await buildPage(catalog, reader, { selected: [], reference: null });
    assert.equal(page.schema, 1);
    assert.equal(Object.hasOwn(page, 'strings'), false);
    assert.deepEqual(page.tabs.map(tab => tab.id), catalog.metadata.tabs.map(tab => tab.id));
    assert.deepEqual(page.tabs.filter(tab => tab.active).map(tab => tab.id), legacyDefaultIds);
  } finally {
    reader.dispose();
  }
}
await checkPageOrchestration();
const columnKeys = new Set(columnKeysFor(catalog.metadata));
let cells = 0, leaves = 0;
function validateTab(tab, loadedKeys = null) {
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
      if (loadedKeys) assert.deepEqual(Object.keys(cell.cmp).sort(), ['golden', ...loadedKeys].sort(), `${tab.id}:${row.family}:${column.sub}: every loaded comparison`);
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
        if (loadedKeys) {
          assert.deepEqual(Object.keys(leaf.cmp).sort(), ['golden', ...loadedKeys].sort(), `${leaf.sub}: every loaded leaf comparison`);
          assert.deepEqual(leaf.tooltip.candidates.map(candidate => candidate.key).sort(), ['golden', ...loadedKeys].sort(), `${leaf.sub}: every loaded tooltip`);
        }
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
function validateUnifiedColumns(unified) {
  assert.ok(unified.columns.every(column => column.label.includes('-')));
  assert.ok(unified.columns.some(column => column.variant_labels?.length || column.schemas?.length));
  if (catalog.capture_generation?.mode === 'latest') {
    const local = new Set(catalog.capture_generation.current.filter(row => row.surface === 'unified')
      .map(row => `${row.implementation}@${row.version}`));
    assert.ok(catalog.sources.candidates.filter(row => row.mode === 'unified')
      .every(row => local.has(`${row.implementation}@${row.version}`)), 'latest choices must name local current observations');
  } else {
    assert.ok(unified.candidates.some(candidate => candidate.key === 'dynamo@0.6.0'));
  }
}
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
async function checkLegacy() {
  const legacyCatalog = { ...catalog, metadata: { ...catalog.metadata, tabs: catalog.metadata.tabs.filter(spec => spec.suite !== 'unified') } };
  const reader = await newReader();
  try {
    const page = await buildPage(legacyCatalog, reader);
    assert.equal(page.schema, 1);
    assert.equal(Object.hasOwn(page, 'strings'), false);
    assert.deepEqual(page.tabs.map(tab => tab.id), legacyCatalog.metadata.tabs.map(tab => tab.id));
    assert.deepEqual(page.tabs.filter(tab => tab.active).map(tab => tab.id), legacyDefaultIds);
    for (const tab of page.tabs) validateTab(tab);
    const stream = page.tabs.find(tab => tab.id === 'tab-toolcalling-streamv1');
    if (stream) {
      assert.ok(stream.rows.find(row => row.family === 'deepseek_v4').parser.html.includes('dsml.rs'));
      assert.ok(stream.rows.find(row => row.family === 'harmony').parser.html.includes('harmony.rs'));
    }
    for (const tab of page.tabs.filter(tab => tab.kind === 'toolcalling')) {
      assert.ok(tab.columns.every(column => !/^null/.test(column.sub)));
      if (tab.id === 'tab-toolcalling-streamv1') for (const sub of ['7.g','7.h','7.i','7.j','7.k','7.l']) assert.equal(tab.columns.find(column => column.sub === sub)?.group_key, 'args');
      assert.ok(tab.columns.some(column => column.schemas?.length));
    }
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
  } finally {
    reader.dispose();
  }
}
await checkLegacy();
async function checkUnifiedSelections(original, reader, loadedKeys) {
  const spec = catalog.metadata.tabs.find(item => item.id === original.id);
  const currentNativeFamilies = tabCandidates(catalog, spec).find(item => item.control.key === 'dynamo').source.native_unified_families;
  const descriptors = tabCandidates(catalog, spec).filter(item => item.source.version && loadedKeys.has(item.control.key) && singletonKeys.has(item.control.key));
  const implementations = [...new Set(descriptors.map(item => item.source.implementation))];
  for (const implementation of implementations) {
    const versions = descriptors.filter(item => item.source.implementation === implementation);
    for (const descriptor of [versions[0], versions.at(-1)]) {
      const key = descriptor.control.key;
      const selected = await buildTab(catalog, reader, original.id, { selected: [key], reference: null });
      const expectedFamilies = descriptor.source.implementation === 'dynamo_v2'
        ? currentNativeFamilies.filter(family => !descriptor.source.capture_families || descriptor.source.capture_families.includes(family))
        : descriptor.source.capture_families;
      assert.deepEqual(selected.rows.map(row => row.family), expectedFamilies,
        `${key}: historical observations retain the current native family scope`);
      for (const row of selected.rows) {
        const baseline = leafCells(original.rows.find(item => item.family === row.family));
        for (const [scenario, cell] of leafCells(row)) {
          const originalCell = baseline.get(scenario);
          const relation = entry => { const cmp = entry.cmp[key]; return [cmp.na, cmp.err, cmp.leak, cmp.sig === entry.cmp.golden.sig]; };
          assert.deepEqual(relation(cell), relation(originalCell), `${key}:${row.family}:${scenario}: GOLDEN comparison preservation`);
          // Not-applicable prose may describe a different first visible family; it has no captured output.
          if (cell.status !== 'na') assert.deepEqual(cell.tooltip.candidates.find(item => item.key === key).block, originalCell.tooltip.candidates.find(item => item.key === key).block, `${key}:${row.family}:${scenario}: full captured output preservation`);
        }
      }
    }
  }
  for (const descriptor of descriptors) singletonKeys.delete(descriptor.control.key);
  const empty = await buildTab(catalog, reader, original.id, { selected: [], reference: null });
  assert.deepEqual(empty.rows, [], 'None contributes no families');
}
const unifiedSpec = catalog.metadata.tabs.find(spec => spec.suite === 'unified');
assert.ok(unifiedSpec);
const allUnifiedDescriptors = tabCandidates(catalog, unifiedSpec).filter(item => item.source.version);
const selectedResources = process.argv[4] ? JSON.parse(await readFile(process.argv[4])) : null;
const unifiedDescriptors = allUnifiedDescriptors.filter(item => !selectedResources || selectedResources.some(resource => resource.implementation === item.source.implementation && resource.version === item.source.version));
const defaultKey = allUnifiedDescriptors.find(item => item.control.default_bucket === 'A').control.key;
const defaultDescriptor = allUnifiedDescriptors.find(item => item.control.key === defaultKey);
if (!unifiedDescriptors.includes(defaultDescriptor)) unifiedDescriptors.push(defaultDescriptor);
for (const implementation of new Set(allUnifiedDescriptors.map(item => item.source.implementation))) {
  const earliest = allUnifiedDescriptors.filter(item => item.source.implementation === implementation).at(-1);
  if (earliest && !unifiedDescriptors.includes(earliest)) unifiedDescriptors.push(earliest);
}
console.log(`model selection: ${JSON.stringify(unifiedDescriptors.map(item => item.source.id).sort())}`);
const singletonKeys = new Set();
for (const implementation of new Set(unifiedDescriptors.map(item => item.source.implementation))) {
  const versions = unifiedDescriptors.filter(item => item.source.implementation === implementation);
  for (const descriptor of [versions[0], versions.at(-1)]) singletonKeys.add(descriptor.control.key);
}
const testedKeys = new Set(), testedFamilies = new Set(), testedScenarios = new Set();
let checkedRebaseContracts = false;
async function checkRebaseContracts(tab, reader) {
  if (checkedRebaseContracts) return;
  const columns = new Map(tab.columns.map(column => [column.sub, column]));
  const standalone = tab.columns.filter(column => catalog.metadata.unified_taxonomy[column.sub].display_group === 'gpt_oss');
  assert.deepEqual(standalone.map(column => column.sub).sort(), modelContracts.gptOssStandaloneHeaderScenarios, 'only the exact standalone Harmony header scenarios retain their own section');
  for (const [scenario, config] of Object.entries(catalog.metadata.folds.unified)) {
    if (config.case && config.case[0] !== scenario) assert.equal(columns.has(scenario), false, `shared variant must fold: ${scenario}`);
  }
  for (const row of tab.rows.filter(row => !row.section)) {
    const leaves = leafCells(row);
    const actual = [...leaves].filter(([, cell]) => cell.status !== 'na').map(([scenario]) => scenario).sort();
    const expected = Object.entries(catalog.metadata.unified_presentation.scenario_families)
      .filter(([, families]) => families.includes(row.family)).map(([scenario]) => scenario).sort();
    assert.deepEqual(actual, expected, `${row.family}: every applicable shared variant remains a leaf`);
  }
  const groups = new Map(tab.column_groups.map(group => [group.key, group]));
  assert.equal(groups.get('unified_gqwen3').label, 'Family-specific tests: Qwen3 and Nemotron v3 XML');
  for (const quotes of ['single', 'double']) {
    const custom = columns.get(`xml_custom_type_${quotes}_quoted_object`);
    const string = columns.get(`xml_string_type_${quotes}_quoted_object`);
    assert.equal(custom.group_key, 'unified_gqwen3');
    assert.ok(custom.label.startsWith('qwen3-'));
    assert.equal(string.group_key, 'unified_g7_declared');
    for (const row of tab.rows.filter(row => !row.section)) {
      const cell = leafCells(row).get(custom.sub);
      assert.equal(cell.case_id, `UNIFIED.7-${custom.sub}`);
      assert.equal(cell.status === 'na', !['qwen3', 'nemotron_v3'].includes(row.family));
      if (['qwen3', 'nemotron_v3'].includes(row.family)) assert.equal(selectedCaptureOutcome(cell, defaultKey), 'pass');
    }
  }
  assert.equal(columns.get('arg_marker_in_string').group_key, 'unified_g7_text');
  assert.equal(columns.get('reason_only').schemas?.length || 0, 0);
  const toolSchemas = columns.get('tool_only').schemas;
  assert.equal(toolSchemas.length, 1);
  const schema = toolSchemas[0];
  assert.deepEqual(schema.tools, [{ name: 'lookup', parameters: { type: 'object', properties: { q: { type: 'string' } } } }]);
  assert.deepEqual([...schema.families].sort(), ['kimi_k2', 'kimi_k3']);
  assert.equal(schema.cases.length, 1);
  assert.equal(schema.cases[0].case_id, 'UNIFIED.gpt_oss-tool_body_whitespace');
  assert.equal(schema.cases[0].label, 'Leading JSON body whitespace');
  assert.deepEqual([...schema.cases[0].families].sort(), ['kimi_k2', 'kimi_k3']);
  assert.ok(columns.get('schema_declared_integer').schemas.length);

  for (const family of ['gpt_oss', 'qwen3', 'glm47', 'deepseek_v4']) {
    const row = tab.rows.find(row => row.family === family);
    const cell = leafCells(row).get('structured_response_after_reasoning');
    const inputs = (await reader.unifiedInputs(family)).value;
    const authored = Object.values(inputs.cases).find(item => item.scenario === 'structured_response_after_reasoning');
    assert(sameValue(cell.tooltip.init, getField(authored.request, 'init')), `${family}: exact structural request init reaches popup model`);
    assert.equal(selectedCaptureOutcome(cell, defaultKey), 'pass', `${family}: current structured response matches GOLDEN`);
    assert.equal(Object.hasOwn(leafCells(row).get('tool_only').tooltip.init, 'structural_tag'), false);
  }
  checkedRebaseContracts = true;
}
function checkDottedNameContracts(tab, loadedKeys) {
  const cases = {
    dotted_name_minimal: [{ kind: 'tool_call', name: 'a.b', arguments: {} }],
    dotted_name_mixed_punctuation: [{ kind: 'tool_call', name: 'my-tool.v2', arguments: { location: 'Beijing' } }],
    dotted_name_multiple_segments: [{ kind: 'tool_call', name: 'server.tools.weather', arguments: { location: 'Beijing' } }],
    dotted_name_shared_prefix: [{ kind: 'tool_call', name: 'server.weather', arguments: { location: 'Beijing' } }, { kind: 'tool_call', name: 'server.time', arguments: { timezone: 'Asia/Shanghai' } }],
  };
  const families = ['beam', 'deepseek_v4', 'deepseek_v41', 'gemma4', 'glm47', 'gpt_oss', 'kimi_k2', 'kimi_k3', 'muse_glimmer', 'nemotron_v3', 'qwen3'];
  for (const family of families) {
    const row = tab.rows.find(item => item.family === family);
    assert.ok(row, `${family}: dotted-name family is rendered`);
    for (const [scenario, expected] of Object.entries(cases)) {
      const cell = leafCells(row).get(scenario);
      const candidate = cell.tooltip.candidates.find(item => item.key === defaultKey);
      assert.deepEqual(candidate.block.events, expected, `${family}/${scenario}: exact dotted tool names`);
      assert.equal(candidate.block.verdict, 'MATCH');
      assert.equal(cell.cmp[defaultKey].na, 0);
      for (const other of cell.tooltip.candidates.filter(item => loadedKeys.has(item.key) && /^(dynamo_v1(?:_jail)?@|vllm_python_batch@0\.26\.0$)/.test(item.key))) {
        assert.equal(other.block.unavailable_code, 'missing_capture');
        assert.equal(Object.hasOwn(other.block, 'events'), false);
      }
      if (family === 'gpt_oss') {
        assert.equal(candidate.parse_mode, 'unified');
        assert.ok(candidate.label.includes('UnifiedParser'));
        for (const [key, mode] of [['dynamo@0.7.27', 'split'], ['dynamo@0.7.28', 'unified']]) {
          if (!loadedKeys.has(key)) continue;
          const historical = cell.tooltip.candidates.find(item => item.key === key);
          assert.equal(historical.parse_mode, mode);
          assert.ok(historical.label.includes(mode === 'split' ? 'split' : 'UnifiedParser'));
        }
      }
    }
  }
  const golden = cases.dotted_name_mixed_punctuation;
  assert.equal(classifyUnified('gpt_oss', golden, golden, catalog.metadata.unified_leak_markers), 'MATCH');
  assert.notEqual(classifyUnified('gpt_oss', golden, [{ ...golden[0], name: 'my-tool' }], catalog.metadata.unified_leak_markers), 'MATCH');
  assert.notEqual(classifyUnified('gpt_oss', golden, [], catalog.metadata.unified_leak_markers), 'MATCH');
}
async function checkUnifiedBatch(batch) {
  const loadedKeys = new Set([defaultKey, ...batch.map(item => item.control.key)]);
  const reader = await newReader();
  try {
    const tab = await buildTab(catalog, reader, unifiedSpec.id, { selected: [...loadedKeys], reference: defaultKey });
    validateTab(tab, loadedKeys);
    validateUnifiedColumns(tab);
    await checkRebaseContracts(tab, reader);
    checkDottedNameContracts(tab, loadedKeys);
    const beam = tab.rows.find(row => row.family === 'beam');
    if (beam) {
      for (const scenario of ['glm_ref_object', 'glm_ref_encoded_targets', 'glm_ref_json_looking_strings', 'glm_ref_scalar_types']) {
        const cell = leafCells(beam).get(scenario);
        assert.equal(cell.cmp[defaultKey].na, 0, `Beam ${scenario}: current capture is measured`);
        for (const descriptor of batch.filter(item => catalog.capture_generation?.mode !== 'latest' && item.source.implementation === 'dynamo_v2' && compareLegacyVersions(item.source.version, '0.7.29') < 0)) {
          const comparison = cell.cmp[descriptor.control.key];
          assert.equal(comparison.na, 1, `Beam ${scenario}: ${descriptor.source.version} predates its introduction`);
          assert.equal(comparison.sig, 0);
        }
      }
    }
    assert.deepEqual(tab.rows.map(row => row.family), [...selectedUnifiedFamilies(catalog, unifiedDescriptors.filter(item => loadedKeys.has(item.control.key)))].sort());
    await checkUnifiedSelections(tab, reader, loadedKeys);
    for (const row of tab.rows) {
      testedFamilies.add(row.family);
      for (const [scenario, cell] of leafCells(row)) {
        testedScenarios.add(scenario);
        for (const key of Object.keys(cell.cmp)) if (key !== 'golden') testedKeys.add(key);
      }
    }
    console.log(`Unified candidate coverage: ${testedKeys.size}/${unifiedDescriptors.length}`);
  } finally {
    reader.dispose();
  }
}
// Each group owns its parsed resources and projected model. Retaining all
// historical candidates at once exceeds V8's heap without adding a cross-group assertion.
for (let offset = 0; offset < unifiedDescriptors.length; offset += 16) {
  await checkUnifiedBatch(unifiedDescriptors.slice(offset, offset + 16));
}
assert.deepEqual([...testedKeys].sort(), unifiedDescriptors.map(item => item.control.key).sort());
assert.deepEqual([...testedFamilies].sort(), [...selectedUnifiedFamilies(catalog, unifiedDescriptors)].sort());
assert.deepEqual([...testedScenarios].sort(), catalog.metadata.unified_presentation.authored_scenarios.filter(scenario => catalog.metadata.unified_presentation.scenario_families[scenario]?.some(family => testedFamilies.has(family))).sort());
assert.equal(singletonKeys.size, 0);
assert.equal(pageMetadata(catalog).schema, 1);
console.log(`Shared matrix structural contracts passed: ${catalog.metadata.tabs.length} tabs, ${testedKeys.size} Unified candidates, ${cells} visible cells, ${leaves} leaf appearances`);
