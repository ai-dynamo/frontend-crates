// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { aggregateCells, foldedSchemas, foldSchemaVariants, foldNullVariants, leafCells, isApplicableCell } from './folds.js';
import { assembleStream, resolveUnified, unifiedResult, compareVersions } from './model.js';
import { deriveStreamExpected } from './legacy.js';
import { canonicalUnifiedLabel } from './unified.js';
import { toolcallingPanel } from './toolcalling.js';
import { cellState, selectTabs, selectRows, defaultReference } from './status.js';

const args = process.argv.slice(2);
if (args.includes('--help')) {
  console.log('usage: node conformance/viewer/retirement_contract.test.js --suite assembly|folds|overlay|batch|status');
  process.exit(0);
}
assert.equal(args.length, 2);
assert.equal(args[0], '--suite');
const catalog = JSON.parse(await readFile(new URL('../viewer-catalog.json', import.meta.url), 'utf8'));
const metadata = catalog.metadata;
const entry = (sig, fields = {}) => ({ sig, na: 0, err: 0, leak: 0, ...fields });
const child = (sub, sig = 1, missing = false, inapplicable = false) => ({ sub, case_id: `UNIFIED.${sub}`, family: 'alpha', kind: 'cell', status: inapplicable ? 'na' : 'ok',
  cmp: { golden: entry(1), reference: entry(sig, { na: Number(missing) }) }, tooltip: { head: sub, input: { text: sub }, init: { starting_state: 'None' }, candidates: [{ key: 'reference', block: { text: sub } }] } });

if (args[1] === 'assembly') {
  assert.deepEqual(assembleStream([[{ kind: 'tool_call', name: 'f', arguments: '{', complete: false }]]), []);
  assert.deepEqual(assembleStream([[{ kind: 'tool_call', name: 'f', arguments: '{"x":', complete: false }], [{ kind: 'tool_call', arguments: '1}', complete: true }]]), [{ kind: 'tool_call', name: 'f', arguments: { x: 1n } }]);
  const malformed = [[{ kind: 'tool_call', name: 'f', arguments: '{', complete: true }]];
  assert.deepEqual(assembleStream(malformed, [{ kind: 'tool_call', name: 'f', arguments: {} }]), [{ kind: 'tool_call', name: 'f', arguments: {} }]);
  assert.equal(assembleStream(malformed)[0].arguments, '{');
  assert.equal(assembleStream(malformed, [{ kind: 'tool_call', name: 'different', arguments: {} }])[0].arguments, '{');
  assert.equal(assembleStream(malformed, [])[0].arguments, '{');
  assert.equal(assembleStream([[{ kind: 'tool_call', name: 'f', arguments: '"literal"', complete: true }]], [{ kind: 'tool_call', name: 'f', arguments: {} }])[0].arguments, 'literal');
  const raw = { chunks: [
    { expected: { vllm: [{ index: 1n, name: 'ec', arguments: '{"x":' }] }, normal_text: { vllm: 'old', vllm_python: 'new' } },
    { expected: { vllm_python: [{ index: 1n, name: 'ho', arguments: '2}' }] }, normal_text: { vllm: '!' } },
  ] };
  const before = structuredClone(raw);
  assert.deepEqual(deriveStreamExpected(raw, metadata, ['vllm_python']).vllm_python.calls, [{ name: 'echo', arguments: { x: 2n } }]);
  assert.equal(deriveStreamExpected(raw, metadata, ['vllm_python']).vllm_python.normal_text, 'new!');
  assert.deepEqual(raw, before);
} else if (args[1] === 'folds') {
  const nullableTab = (cells) => ({ id: 'tab-toolcalling-batch', columns: cells.map(cell => ({ sub: cell.sub, label: cell.sub, group_key: 'args' })),
    column_groups: [{ key: 'args', span: cells.length }], rows: [{ family: 'alpha', cells: Object.fromEntries(cells.map(cell => [cell.sub, cell])) }], stats: {}, glossary: [] });
  const scope = { folds: { null_descriptions: { '7-4': 'nullable', '7-5': 'omitted' }, mixed_families: { '7-4.mixed_a': ['alpha'] } } };
  const fixture = (sub, applicable, missingCapture = false) => ({ ...child(sub, 1, missingCapture, missingCapture || !applicable), case_id: `TOOLCALLING.batch.${sub}`, applicable });
  // Hiding a sole nonroot member must preserve its identity and popup when the
  // visible root itself is inapplicable.
  const exact = nullableTab([fixture('7-4', false), fixture('7-4.a', true)]);
  foldNullVariants(exact, scope);
  assert.deepEqual([...leafCells(exact.rows[0]).keys()], ['7-4.a']);
  assert.equal(exact.rows[0].cells['7-4'].sub, '7-4');
  assert.equal(exact.stats.fixture_cases, 1);
  assert.equal(exact.rows[0].cells['7-4'].variants[0].tooltip.head, '7-4.a');
  for (const rootState of ['eligible', 'ineligible', 'missing']) for (const count of [0, 1, 3]) for (const missingCapture of [false, true]) {
    for (let mask = 0; mask < 2 ** count; mask++) {
      const root = fixture('7-4', rootState === 'eligible', missingCapture);
      if (rootState === 'missing') { root.kind = 'missing'; root.case_id = null; }
      const cells = [root, ...Array.from({ length: count }, (_, index) => fixture(`7-4.${String.fromCharCode(97 + index)}`, Boolean(mask & 2 ** index), missingCapture))];
      const expected = cells.filter(cell => cell.case_id && cell.applicable).map(cell => cell.sub);
      const tab = nullableTab(cells);
      foldNullVariants(tab, scope);
      const actual = [...leafCells(tab.rows[0]).values()].filter(cell => cell.case_id && isApplicableCell(cell)).map(cell => cell.sub);
      assert.deepEqual(actual, expected, `${rootState}/${count}/${mask}/${missingCapture}: every applicable authored leaf survives once`);
      assert.equal(tab.stats.fixture_cases, expected.length, `${rootState}/${count}/${mask}/${missingCapture}: selected absence does not change fixture count`);
    }
  }
  const shared = nullableTab([fixture('7-4', false), fixture('7-4.mixed_a', true, true), fixture('7-5', false), fixture('7-5.a', true, true)]);
  foldNullVariants(shared, scope);
  assert.deepEqual([...leafCells(shared.rows[0]).keys()], ['7-4.mixed_a', '7-5.a']);
  assert.equal(shared.stats.fixture_cases, 2);
  for (const [sig, missing, inapplicable, expected] of [[1, false, false, [false, false]], [2, false, false, [true, false]], [1, true, false, [true, true]], [1, true, true, [false, false]]]) {
    const children = [child('a'), child('b', sig, missing, inapplicable)];
    const before = structuredClone(children);
    const result = aggregateCells(children, '7-parent', 'parent');
    assert.deepEqual([result.cmp.reference.sig !== result.cmp.golden.sig, Boolean(result.cmp.reference.na)], expected);
    assert.deepEqual(result.variants, children);
    assert.deepEqual(children, before);
  }
  const failure = aggregateCells([child('a', 2), child('b', 1, true)], '7-parent', 'parent');
  assert.equal(failure.cmp.reference.na, 0);
  assert.equal(cellState(failure, { key: 'reference', label: 'Reference' })[0], 'red');
  assert.equal(aggregateCells([child('a', 1, false, true)], '7-parent', 'parent').status, 'na');
  const tools = [{ name: 'probe', parameters: { type: 'string' } }];
  const schemas = foldedSchemas([{ label: 'Direct', schemas: [{ families: ['alpha'], tools }] }, { label: 'Reference', schemas: [{ families: ['beta'], tools }] }]);
  assert.equal(schemas.length, 1);
  assert.deepEqual(schemas[0].families, ['alpha', 'beta']);
  assert.deepEqual(schemas[0].cases.map(({ label, families }) => [label, families]), [['Direct', ['alpha']], ['Reference', ['beta']]]);
  const groups = new Map();
  for (const [scenario, spec] of Object.entries(metadata.folds.unified)) if (spec.schema) {
    if (!groups.has(spec.schema[0])) groups.set(spec.schema[0], []);
    groups.get(spec.schema[0]).push([scenario, spec.schema[1]]);
  }
  assert.equal(groups.size, 22);
  for (const [parent, variants] of groups) {
    assert.equal(variants.length, 4);
    assert.ok(variants.some(([scenario]) => scenario === parent));
  }
  for (const prefix of ['schema_const_plain_', 'schema_union_anyOf_integer_', 'schema_union_oneOf_integer_']) for (const single of [false, true]) {
    const members = [...groups].flatMap(([, variants]) => variants).filter(([scenario]) => scenario.startsWith(prefix) && metadata.folds.unified[scenario].schema[0] !== scenario);
    const chosen = single ? members.slice(0, 1) : members;
    const columns = chosen.map(([sub]) => ({ sub, label: metadata.folds.unified[sub].canonical_label, desc: 'probe', group_key: 'schema', schemas: [{ families: ['alpha'], tools }] }));
    const originals = chosen.map(([sub]) => child(sub));
    const before = structuredClone(originals);
    const tab = { id: 'tab-unified', columns, rows: [{ family: 'alpha', cells: Object.fromEntries(originals.map((value) => [value.sub, value])) }], column_groups: [{ key: 'schema', span: columns.length }], stats: {}, glossary: [{ rows: columns.map((column) => [column.label, 'probe']) }] };
    foldSchemaVariants(tab, metadata);
    assert.equal(tab.columns.length, 1);
    const root = tab.columns[0], result = tab.rows[0].cells[root.sub];
    assert.equal(result.case_id, `UNIFIED.${root.label}`);
    assert.equal(tab.stats.fixture_cases, chosen.length);
    assert.deepEqual(new Set(leafCells(tab.rows[0]).keys()), new Set(chosen.map(([sub]) => sub)));
    assert.deepEqual(originals, before);
    for (const variant of result.variants) assert.ok(variant.tooltip.head.startsWith(metadata.folds.unified[variant.sub].schema[1] + ' — '));
  }
} else if (args[1] === 'overlay') {
  const first = { implementation: 'dynamo_v2', version: '0.6.0', path: 'first.yaml', bytes: new Uint8Array(), document: { provenance: { status: 'legacy', captured_with: { dynamo_v2: '0.6.0' } }, document: {}, changes: { id: { case_key: 'UNIFIED.3', observation: { value: { assembled: [{ kind: 'text', text: 'captured' }] } } } }, metadata_changes: {}, document_overrides: {} } };
  const carried = resolveUnified([first], '0.6.1').get('id');
  assert.equal(unifiedResult(carried).events[0].text, 'captured');
  assert.equal(carried.value.document.captured_with.dynamo_v2, '0.6.0');
  const error = { ...first, version: '0.6.1', document: { ...first.document, changes: { id: { case_key: 'UNIFIED.3', observation: { error: { code: 'capture_failure', detail: 'capture failed' } } } } } };
  const current = resolveUnified([first, error], '0.6.1').get('id');
  assert.equal(unifiedResult(current).state, 'error');
  assert.equal(unifiedResult(current).reason, 'capture failed');
  assert.equal(Object.hasOwn(current.value.document, 'inherited_from'), false);
  const before = structuredClone(first);
  resolveUnified([first], '0.6.1');
  assert.deepEqual(first, before);
  assert.throws(() => compareVersions('0.6.1+source.' + 'a'.repeat(64), '0.6.1'), /invalid capture version/);
  assert.equal(canonicalUnifiedLabel('UNIFIED.31.a', 'qwen3', metadata.unified_aliases), 'UNIFIED.31-guided_json_invalid_call');
  assert.equal(canonicalUnifiedLabel('UNIFIED.31-29', 'qwen3', metadata.unified_aliases), 'UNIFIED.31-29');
} else if (args[1] === 'batch') {
  const testCatalog = structuredClone(catalog);
  testCatalog.sources.candidates = [
    { implementation: 'dynamo_v1', mode: 'batch', version: '10.0.1', archive: 'batch.tar.gz' },
    { implementation: 'dynamo_v2', mode: 'batch-on-stream', version: '0.7.20', archive: 'conformance/fixtures/toolcalling/fixtures-batch-on-stream-v1.tar.gz' },
  ];
  const spec = testCatalog.metadata.tabs.find((tab) => tab.id === 'tab-toolcalling-batch');
  const result = (value) => ({ calls: [{ name: 'probe', arguments: { value } }], normal_text: '' });
  for (const [expected, wrong] of [['  spaced  ', 'spaced'], ['&lt;', '<'], ['null', null], [42n, '42']]) {
    const id = 'TOOLCALLING.batch.7.p.probe';
    const golden = result(expected), before = structuredClone(golden);
    testCatalog.metadata.divergence_notes = { deepseek_v4: { [id]: { stream_vs_batch: 'Documented parity difference.' } } };
    const document = (block = null) => new Map([['deepseek_v4/TOOLCALLING.batch.7.p.yaml', { member: 'fixture.yaml', value: { family: 'deepseek_v4', mode: 'batch', cases: { [id]: { description: 'probe', model_text: 'wire', golden, ...(block ? { expected: { dynamo_v1: block } } : {}) } } } }]]);
    let observed = golden;
    const reader = { legacy: async () => document(result(wrong)), legacyDocuments: async (resource) => resource.path.includes('batch-on-stream') ? new Map([['overlay.yaml', { value: { family: 'deepseek_v4', cases: { [id]: { dynamo_v2: observed } } } }]]) : document() };
    const panel = await toolcallingPanel(testCatalog, reader, spec);
    const cell = panel.rows.find((row) => row.family === 'deepseek_v4').cells['7.p.probe'];
    assert.equal(cell.red_on_diff, true);
    assert.equal(cell.known_divergence, true);
    assert.equal(cell.cmp.golden.sig, cell.cmp['dynamo_v2-s-0-7-20'].sig);
    assert.notEqual(cell.cmp.golden.sig, cell.cmp['dynamo_v1-b-10-0-1'].sig);
    assert.deepEqual(golden, before);
    observed = result('wrong');
    const wrongPanel = await toolcallingPanel(testCatalog, reader, spec);
    const wrongCell = wrongPanel.rows.find((row) => row.family === 'deepseek_v4').cells['7.p.probe'];
    assert.notEqual(wrongCell.cmp.golden.sig, wrongCell.cmp['dynamo_v2-s-0-7-20'].sig);
  }
  const id = 'TOOLCALLING.batch.11.b';
  const defect = 'FIXME (v2 defect): empty tool name; exact baseline retained.';
  const wrong = { calls: [{ name: '', arguments: { value: 1n } }], normal_text: '' };
  testCatalog.metadata.divergence_notes = { deepseek_v4: { [id]: { baseline_defect: defect } } };
  const document = (capture = false) => new Map([['deepseek_v4/TOOLCALLING.batch.11.yaml', { member: 'fixture.yaml', value: { family: 'deepseek_v4', mode: 'batch', cases: { [id]: { description: 'probe', model_text: 'wire', ...(capture ? { expected: { dynamo_v1: { calls: [], normal_text: '' } } } : {}) } } } }]]);
  const reader = { legacy: async () => document(true), legacyDocuments: async (resource) => resource.path.includes('batch-on-stream') ? new Map([['overlay.yaml', { value: { family: 'deepseek_v4', cases: { [id]: { dynamo_v2: wrong } } } }]]) : document() };
  const panel = await toolcallingPanel(testCatalog, reader, spec);
  const cell = panel.rows.find((row) => row.family === 'deepseek_v4').cells['11.b'];
  assert.equal(cell.known_divergence, false);
  assert.notEqual(cell.cmp['dynamo_v1-b-10-0-1'].sig, cell.cmp['dynamo_v2-s-0-7-20'].sig);
  assert.ok(cell.tooltip.dynamo_notes.some(([, note]) => note.includes(defect)));
  assert.equal(cell.tooltip.candidates.find((candidate) => candidate.key === 'dynamo_v2-s-0-7-20').block.explanation, undefined);
  for (const absence of ['document', 'case']) {
    const missingReader = { ...reader, legacyDocuments: async (resource) => resource.path.includes('batch-on-stream')
      ? absence === 'document' ? new Map() : new Map([['overlay.yaml', { value: { family: 'deepseek_v4', cases: {} } }]])
      : document() };
    const reference = 'dynamo_v2-s-0-7-20';
    const missing = await toolcallingPanel(testCatalog, missingReader, spec, { selected: [reference], reference });
    const missingCell = missing.rows.find((row) => row.family === 'deepseek_v4').cells['11.b'];
    assert.equal(missingCell.kind, 'cell', `${absence}: an authored fixture keeps its identity when the loaded Reference has no capture`);
    assert.equal(missingCell.status, 'missing');
    assert.equal(cellState(missingCell, missing.candidates.find((candidate) => candidate.key === reference))[0], 'empty');
  }
  await assert.rejects(toolcallingPanel(testCatalog, reader, spec, { selected: ['dynamo_v1-b-10-0-1'], reference: 'dynamo_v2-s-0-7-20' }), /reference is not loaded/);
  await assert.rejects(toolcallingPanel(testCatalog, reader, spec, { selected: [], reference: 'unknown' }), /unknown toolcalling reference/);
} else if (args[1] === 'status') {
  const tabs = metadata.tabs.map((spec) => ({ ...spec, kind: spec.suite }));
  assert.equal(selectTabs({ tabs }, ['toolcalling']).length, 2);
  assert.equal(selectTabs({ tabs }, ['reasoning']).length, 2);
  assert.equal(selectTabs({ tabs }, ['tab-unified', 'unified']).length, 1);
  assert.throws(() => selectTabs({ tabs }, ['unknown']), /unknown tab/);
  assert.throws(() => selectRows({ id: 'tab-unified', rows: [] }, ['qwen3']), /unknown model/);
  assert.throws(() => defaultReference({ id: 'tab-unified', candidates: [{ default_bucket: 'A' }, { default_bucket: 'A' }] }), /found 2/);
} else throw new Error(`unknown retirement suite: ${args[1]}`);
console.log(`retired Python ${args[1]} contracts passed through shared JS owners`);
