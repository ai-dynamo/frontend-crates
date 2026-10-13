// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { catalogPeerAvailability, catalogFixtureApplicability } from './availability.js';
import { legacyOutcome } from './comparison.js';

const catalog = JSON.parse(await readFile(new URL('../viewer-catalog.json', import.meta.url), 'utf8'));
assert.equal(legacyOutcome({ unavailable: 'no parser', parserError: true }, []).status, 'na', 'ordinary captured fields cannot become source-control overrides');
const root = 'toolcalling/fixtures-stream-v1';
const spec = catalog.metadata.tabs.find(tab => tab.id === 'tab-toolcalling-streamv1');
const ids = ['TOOLCALLING.streamv1.1.a', 'TOOLCALLING.streamv1.1.b'];
const archive = (name, cases) => ({ path: `conformance/fixtures/${root}/${name}.tar.gz`, availability_documents: {
  'probe/TOOLCALLING.streamv1.yaml': { family: 'probe', mode: 'streamv1', cases },
} });
catalog.sources.archives = [
  archive('inputs', Object.fromEntries(ids.map(id => [id, { unavailable: { vllm: true }, chunk_runs: [[2, {}]] }]))),
  archive('vllm_python-1.0.0', Object.fromEntries(ids.map(id => [id, { chunk_runs: [[1, { expected: true }]] }]))),
  archive('vllm_python-2.0.0', { [ids[0]]: { unavailable: true } }),
  archive('vllm_python-2.0.0.patch1', { [ids[1]]: { unavailable: true } }),
  archive('sglang_python-1.0.0', Object.fromEntries(ids.map(id => [id, { unavailable: true }]))),
  archive('sglang_python-2.0.0', { [ids[0]]: { chunk_runs: [[1, { expected: true }]] } }),
  archive('dynamo_v2-2.0.0', { [ids[0]]: { chunk_runs: [[1, { expected: true }]] }, [ids[1]]: { unavailable: true } }),
];
catalog.sources.candidates = [
  { implementation: 'dynamo_v2', mode: 'stream', version: '2.0.0' },
  ...['vllm_python', 'sglang_python'].flatMap(implementation => ['1.0.0', '2.0.0'].map(version => ({ implementation, mode: 'stream', version }))),
];
const before = JSON.stringify(catalog);
let availability = await catalogPeerAvailability(catalog, spec);
assert.equal(availability.vllm_python.has('probe'), true, 'patch replacement clears the last available case');
assert.equal(availability.sglang_python.has('probe'), false, 'a newer sparse capture clears unavailable state and replaces chunk state');
assert.equal(JSON.stringify(catalog), before, 'resolving presence controls must not mutate the catalog');
const applicability = await catalogFixtureApplicability(catalog, spec);
assert.deepEqual(applicability.get('probe\0' + ids[0].replace('TOOLCALLING.streamv1.', '')), { applicable: true, present: true });
assert.deepEqual(applicability.get('probe\0' + ids[1].replace('TOOLCALLING.streamv1.', '')), { applicable: false, present: true }, 'recorded unavailable retains block presence separately from fixture eligibility');
const failed = structuredClone(catalog);
failed.sources.archives.find(resource => resource.path.endsWith('dynamo_v2-2.0.0.tar.gz')).availability_documents['probe/TOOLCALLING.streamv1.yaml'].cases[ids[1]].unavailable = 'parser-error';
assert.deepEqual((await catalogFixtureApplicability(failed, spec)).get('probe\0' + ids[1].replace('TOOLCALLING.streamv1.', '')), { applicable: true, present: true }, 'parser failure controls remain failures rather than becoming not-applicable leaves');
const oldest = structuredClone(catalog);
oldest.metadata.selection_policy.stream.inventory_defaults.sglang_python = 'oldest';
availability = await catalogPeerAvailability(oldest, spec);
assert.equal(availability.sglang_python.has('probe'), true, 'inventory defaults select oldest independently of UI selection');
const alias = structuredClone(catalog);
alias.metadata.legacy_aliases[ids[0]] = 'TOOLCALLING.streamv1.1.alias';
availability = await catalogPeerAvailability(alias, spec);
assert.equal(availability.vllm_python.has('probe'), true, 'case aliases use the shared legacy owner');
console.log('Selection-independent availability controls preserve sparse/patch/alias/state replacement and oldest policy');

const reasoningCatalog = structuredClone(catalog);
const reasoningSpec = reasoningCatalog.metadata.tabs.find(tab => tab.id === 'tab-reasoning-batch');
reasoningCatalog.sources.archives = [{ path: `conformance/fixtures/${reasoningSpec.root}/inputs.tar.gz`, availability_documents: {
  'probe/REASONING.yaml': { family: 'probe', mode: 'reasoning', cases: { 'REASONING.batch.1.a': { expected: { dynamo_v1: 'present', vllm: 'unavailable' } } } },
} }];
reasoningCatalog.sources.candidates = [{ implementation: 'dynamo_v1', mode: 'reasoning', version: '1.0.0' }];
assert.equal((await catalogPeerAvailability(reasoningCatalog, reasoningSpec)).vllm_python.has('probe'), true, 'alias-only reasoning controls without overlays retain unavailable peer');
console.log('alias-only reasoning input controls retain peer availability');

for (const aliasFirst of [true, false]) for (const first of ['present', 'unavailable', 'unavailable-error']) {
  const input = structuredClone(reasoningCatalog);
  input.sources.archives[0].availability_documents['probe/REASONING.yaml'].cases['REASONING.batch.1.a'].expected = aliasFirst
    ? { vllm: first, vllm_python: 'present', dynamo_v1: 'present' }
    : { vllm_python: first, vllm: 'present', dynamo_v1: 'present' };
  assert.equal((await catalogPeerAvailability(input, reasoningSpec)).vllm_python.has('probe'), first !== 'present', `controls preserve first key ${aliasFirst}/${first}`);
}
