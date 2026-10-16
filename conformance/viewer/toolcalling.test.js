// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { toolcallingPanel } from './toolcalling.js';
import { tabCandidates } from './candidates.js';

const catalog = JSON.parse(await readFile(new URL('../viewer-catalog.json', import.meta.url), 'utf8'));
// Synthetic records own their inventory; real corpus controls describe other cases.
for (const archive of catalog.sources.archives) delete archive.availability_documents;
const spec = catalog.metadata.tabs.find((tab) => tab.id === 'tab-toolcalling-batch');
catalog.sources.candidates = [
  { implementation: 'dynamo_v1', mode: 'batch', version: '3.0.0', archive: 'older.tar.gz' },
  { implementation: 'dynamo_v1', mode: 'batch', version: '3.1.0', archive: 'newer.tar.gz' },
  { implementation: 'vllm_python', mode: 'batch', version: '0.25.0', archive: 'peer.tar.gz' },
  { implementation: 'vllm_python', mode: 'batch-on-stream', version: '0.25.0', archive: 'conformance/fixtures/toolcalling/fixtures-batch-on-stream-v1.tar.gz' },
];
const docs = (implementation, version) => new Map([['qwen3/TOOLCALLING.batch.1.yaml', { member: 'source.yaml', value: {
  family: 'qwen3', mode: 'batch', cases: { 'TOOLCALLING.batch.1': { description: 'sample', model_text: 'sample',
    ...(implementation ? { expected: { [implementation]: version === '3.0.0' ? { unavailable: 'no parser' } : { calls: [], normal_text: 'captured' } } } : {}) },
  },
} }]]);
const loaded = [];
const reader = {
  legacyDocuments: async (resource) => resource.path.includes('batch-on-stream') ? new Map([['qwen3/TOOLCALLING.batch.1.yaml', { member: 'overlay.yaml', value: { family: 'qwen3', cases: { 'TOOLCALLING.batch.1': { vllm_python: { calls: [], normal_text: 'streamed' } } } } }]]) : docs(),
  legacy: async ({ implementation, version }) => { loaded.push(`${implementation}@${version}`); return docs(implementation, version); },
};
const controls = tabCandidates(catalog, spec);
const key = (implementation, mode, version) => controls.find((item) => item.source.implementation === implementation && item.source.mode === mode && item.source.version === version).control.key;
const peer = key('vllm_python', 'batch', '0.25.0');
const older = key('dynamo_v1', 'batch', '3.0.0');
const newer = key('dynamo_v1', 'batch', '3.1.0');
const stream = key('vllm_python', 'batch-on-stream', '0.25.0');
const panelCell = async (selected, reference) => {
  const panel = await toolcallingPanel(catalog, reader, spec, { selected, reference });
  return { panel, cell: panel.rows.find((row) => row.family === 'qwen3').cells['1'] };
};
let result = await panelCell([peer], peer);
assert.equal(result.cell.status, 'ok');
assert.equal(result.cell.tooltip.candidates.length, 1);
assert.equal(result.panel.stats.fixture_cases, 1);
assert.deepEqual(loaded, ['vllm_python@0.25.0']);
result = await panelCell([older, newer], older);
assert.equal(result.cell.status, 'na');
assert.equal(result.cell.cmp[older].na, 1);
assert.equal(result.cell.facts.find((fact) => fact.impl === 'dynamo_v1').present, false);
assert.equal(result.cell.tooltip.candidates.find((candidate) => candidate.key === older).block.unavailable, 'no parser');
assert.equal(result.cell.tooltip.candidates.find((candidate) => candidate.key === newer).block.normal_text, 'captured');
result = await panelCell([stream], stream);
assert.equal(result.cell.status, 'ok');
assert.equal(result.cell.cmp[stream].na, 0);
assert.equal(result.cell.facts.find((fact) => fact.impl === 'vllm_python').present, true);
assert.equal(result.cell.tooltip.candidates[0].block.normal_text, 'streamed');
assert.equal(result.panel.stats.fixture_cases, 1);
assert.equal(result.panel.stats.parity + result.panel.stats.dynamo_only + result.panel.stats.documented + result.panel.stats.research + result.panel.stats.errors, result.panel.stats.real);
console.log('selected peer, historical-version, and cross-mode reference identities passed');
