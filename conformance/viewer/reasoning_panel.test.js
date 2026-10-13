// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { reasoningComparisonEntries, reasoningFacts, reasoningOverviewStatus } from './comparison.js';
import { buildTab } from './page_model.js';

const catalog = JSON.parse(await readFile(new URL('../viewer-catalog.json', import.meta.url), 'utf8'));
catalog.sources.candidates = [
  { implementation: 'dynamo_v1', mode: 'reasoning', version: '3.0.0', archive: 'dynamo.tar.gz' },
  { implementation: 'vllm_python', mode: 'reasoning', version: '0.25.0', archive: 'peer.tar.gz' },
];
const documents = (peer = false) => new Map([['basic/REASONING.yaml', { member: 'basic/REASONING.yaml', value: {
  family: 'basic', model_label: 'Basic', cases: { 'REASONING.batch.1.a': {
    description: 'sample', model_text: 'sample', expected: {
      dynamo_v1: { reasoning_text: '', normal_text: 'sample' },
      ...(peer ? { vllm_python: { error: 'parser failed' } } : {}),
    },
  } },
} }]]);
const loaded = [];
const reader = {
  legacyDocuments: async () => documents(),
  legacy: async ({ implementation }) => { loaded.push(implementation); return documents(implementation === 'vllm_python'); },
};
const build = async (selected, reference) => {
  const tab = await buildTab(catalog, reader, 'tab-reasoning-batch', { selected, reference });
  return { tab, cell: tab.rows.find((row) => row.family === 'basic').cells['REASONING.batch.1.a'] };
};

let result = await build(['vllm_python'], 'vllm_python');
assert.deepEqual(loaded, ['vllm_python']);
assert.equal(result.cell.status, 'problem');
assert.equal(result.cell.tooltip.candidates.length, 1);
assert.equal(result.cell.tooltip.candidates[0].block.error, 'parser failed');
assert.equal(result.cell.facts[0].impl, 'vllm_python');
assert.equal(result.cell.facts[0].status, 'problem');
assert.equal(result.cell.facts[0].agrees, true);
assert.equal(result.tab.stats.errors, result.tab.stats.real);
assert.ok(result.tab.stats.errors > 0);

result = await build(['dynamo_v1', 'vllm_python'], 'dynamo_v1');
assert.equal(result.cell.status, 'ok');
result = await build(['dynamo_v1', 'vllm_python'], 'vllm_python');
assert.equal(result.cell.status, 'problem');

result = await build(['vllm_python'], null);
assert.equal(result.cell.status, 'na');
assert.equal(result.cell.facts[0].agrees, null);
assert.equal(result.tab.stats.real, 0);
assert.equal(result.tab.stats.na, result.tab.stats.slots);
loaded.length = 0;
result = await build([], null);
assert.equal(result.cell.status, 'na');
assert.equal(result.cell.tooltip.candidates.length, 0);
assert.deepEqual(loaded, []);
await assert.rejects(build(['dynamo_v1'], 'vllm_python'), /reference is not loaded/);
await assert.rejects(build(['dynamo_v1'], 'unknown'), /unknown reasoning reference/);

result = await build(null, undefined);
assert.equal(result.cell.status, 'ok');
assert.equal(result.cell.tooltip.candidates.length, 2);
assert.equal(result.cell.facts[0].impl, 'dynamo_v1');
console.log('reasoning selected-reference status, facts, summaries, empty Reference, and transport passed');

// Alias-only input and capture maps must reach every reasoning presentation owner.
reader.legacyDocuments = async () => documents(true);
reader.legacy = async () => {
  const result = documents(true);
  for (const entry of result.values()) for (const record of Object.values(entry.value.cases)) {
    record.expected.vllm = { reasoning_text: 'fresh', normal_text: 'alias-only' };
    delete record.expected.vllm_python;
  }
  return result;
};
result = await build(['vllm_python'], 'vllm_python');
assert.equal(result.cell.status, 'ok');
assert.equal(result.cell.tooltip.candidates.length, 1);
assert.equal(result.cell.tooltip.candidates[0].block.normal_text, 'alias-only');
assert.equal(result.cell.facts[0].agrees, true);
console.log('alias-only reasoning captures replace canonical input and remain visible');

// Duplicate legacy keys keep frozen first-key precedence across all consumers.
for (const aliasFirst of [true, false]) {
  const expected = aliasFirst
    ? { vllm: { error: 'first alias' }, vllm_python: { reasoning_text: '', normal_text: 'second canonical' } }
    : { vllm_python: { reasoning_text: '', normal_text: 'first canonical' }, vllm: { error: 'second alias' } };
  const dual = () => {
    const result = documents();
    for (const entry of result.values()) for (const record of Object.values(entry.value.cases)) record.expected = structuredClone(expected);
    return result;
  };
  reader.legacyDocuments = async () => dual();
  reader.legacy = async () => new Map();
  result = await build(['vllm_python'], 'vllm_python');
  const raw = { expected };
  assert.equal(reasoningOverviewStatus(raw, 'basic', 'vllm_python', catalog.metadata), aliasFirst ? 'problem' : 'ok');
  assert.deepEqual(result.cell.cmp, reasoningComparisonEntries(raw, 'basic', catalog.metadata));
  assert.equal(reasoningFacts(raw, 'basic', catalog.metadata, 'vllm_python', ['vllm_python'])[0].status, result.cell.status);
  assert.equal(result.cell.status, aliasFirst ? 'problem' : 'ok');
  assert.equal(result.cell.facts[0].status, result.cell.status);
  assert.deepEqual(result.cell.tooltip.candidates[0].block, aliasFirst ? { error: 'first alias' } : { reasoning_text: '', normal_text: 'first canonical' });
}
console.log('duplicate alias precedence agrees across reasoning status, facts and popup');

for (const aliasFirst of [true, false]) for (const first of [{ unavailable: 'unsupported' }, { error: 'failed' }, { reasoning_text: '', normal_text: 'good' }]) {
  const second = { normal_text: 'other' };
  const expected = aliasFirst ? { vllm: first, vllm_python: second } : { vllm_python: first, vllm: second };
  const dual = () => {
    const result = documents();
    for (const entry of result.values()) for (const record of Object.values(entry.value.cases)) record.expected = structuredClone(expected);
    return result;
  };
  reader.legacyDocuments = async () => dual();
  reader.legacy = async () => new Map();
  result = await build(['vllm_python'], 'vllm_python');
  const status = first.unavailable ? 'na' : first.error ? 'problem' : 'ok';
  assert.equal(result.cell.status, status);
  assert.equal(result.cell.facts[0].status, status);
  assert.deepEqual(result.cell.cmp, reasoningComparisonEntries({ expected }, 'basic', catalog.metadata));
  assert.deepEqual(result.cell.tooltip.candidates[0].block, first);
}
