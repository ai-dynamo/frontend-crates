// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { deriveStreamExpected, legacyCases } from './legacy.js';

const { metadata } = JSON.parse(await readFile(new URL('../viewer-catalog.json', import.meta.url), 'utf8'));
const stream = { chunks: [
  { expected: { dynamo_v1: [{ index: 0n, name: 'get', arguments: '{"large":' }] }, normal_text: { dynamo_v1: 'before ' } },
  { expected: { dynamo_v1: [{ index: 0n, name: 'weather', arguments: '9007199254740993}' }] }, normal_text: { dynamo_v1: 'after' } },
] };
assert.deepEqual(deriveStreamExpected(stream, metadata, ['dynamo_v1']).dynamo_v1, { calls: [{ name: 'getweather', arguments: { large: 9007199254740993n } }], normal_text: 'before after' });
assert.equal(deriveStreamExpected({ chunks: [] }, metadata, ['dynamo_v1']).dynamo_v1.unavailable, 'No recorded capture for this case.');
assert.deepEqual(deriveStreamExpected({ chunks: [{ expected: { dynamo_v1: [] } }] }, metadata, ['dynamo_v1']).dynamo_v1, { calls: [], normal_text: '' });
assert.equal(deriveStreamExpected({ unavailable: { dynamo_v1: 'no parser' }, chunks: stream.chunks }, metadata, ['dynamo_v1']).dynamo_v1.unavailable, 'no parser');
assert.equal(deriveStreamExpected({ exception: { dynamo_v1: 'failed' }, chunks: stream.chunks }, metadata, ['dynamo_v1']).dynamo_v1.exception, 'failed');
assert.equal(deriveStreamExpected({ chunks: [{ expected: { dynamo_v1: [{ index: 1n, name: 'f', arguments: '{' }] } }] }, metadata, ['dynamo_v1']).dynamo_v1.calls[0].arguments, '{');
assert.throws(() => deriveStreamExpected({ chunks: [{ expected: { dynamo_v1: [{ name: 'f' }] } }] }, metadata, ['dynamo_v1']), /no index/);
assert.equal(deriveStreamExpected({ chunks: [{ expected: { dynamo_v1: [{ index: 1n, name: 'get' }, { index: 1, name: 'weather' }] } }] }, metadata, ['dynamo_v1']).dynamo_v1.calls[0].name, 'getweather');
assert.equal(deriveStreamExpected({ chunks: [{ expected: { dynamo_v1: [{ index: '1', name: 'string' }, { index: 1n, name: 'number' }] } }] }, metadata, ['dynamo_v1']).dynamo_v1.calls.length, 2);

const documents = new Map([
  ['sample/TOOLCALLING.batch.yaml', { member: 'source.yaml', value: { mode: 'batch', family: 'sample', cases: { 'TOOLCALLING.batch.30': { description: 'parent reason', expected: { dynamo: { unavailable: 'grammar exclusion' } } } } } }],
  ['other/TOOLCALLING.batch.yaml', { member: 'other.yaml', value: { mode: 'batch', family: 'other', cases: { 'TOOLCALLING.batch.30.b': { description: 'leaf' } } } }],
]);
const cases = legacyCases(documents, 'batch', metadata).cases;
assert.equal(cases.has('sample\0' + '30'), false);
assert.equal(cases.get('sample\0' + '30.a').record.__synthetic_from_case_id, 'TOOLCALLING.batch.30');
assert.equal(cases.get('sample\0' + '30.c').record.expected.dynamo_v1.unavailable, 'grammar exclusion');
assert.equal(cases.get('sample\0' + '30.a').source.member, 'source.yaml');
assert.equal(documents.get('sample/TOOLCALLING.batch.yaml').value.cases['TOOLCALLING.batch.30'].__case_id, undefined);
const aliases = new Map([['sample/TOOLCALLING.streamv1.yaml', { value: { mode: 'streamv1', family: 'sample', cases: { 'TOOLCALLING.streamv1.7-1': { chunks: [] } } } }]]);
assert.equal(legacyCases(aliases, 'streamv1', metadata, ['dynamo_v1']).cases.get('sample\0' + '7-5').record.__case_id, 'TOOLCALLING.streamv1.7-5');
aliases.set('sample/TOOLCALLING.batch.yaml', { value: { mode: 'batch', family: 'sample', cases: { 'TOOLCALLING.batch.7-5': { expected: { dynamo: { calls: [], normal_text: 'batch text' } } } } } });
assert.equal(legacyCases(aliases, 'streamv1', metadata, ['dynamo_v1']).cases.get('sample\0' + '7-5').record.batch_expected.dynamo_v1.normal_text, 'batch text');
console.log('legacy assembly, aliases, split-parent normalization, and source preservation passed');
