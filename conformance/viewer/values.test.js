// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { parseJson, sameValue, stringifyValue, canonicalJson, temporalText } from './values.js';
import { hasField, getField, setField, deleteField } from './mappings.js';
import { parseYaml, HttpResourceStore, sha256 } from './data.js';
import { assembleStream, compareLegacyVersions, compareVersions, unifiedEvents } from './model.js';
import { mergeLegacyDocument } from './resources.js';
import { deriveStreamExpected } from './legacy.js';

const json = parseJson('{"large":9007199254740993,"int":1,"float":1.0,"null":null,"s":"1","bool":true,"a":[1,2]}');
assert.equal(json.large, 9007199254740993n);
assert.equal(typeof json.float, 'number');
assert.equal(typeof json.int, 'bigint');
assert.equal(sameValue(json, { ...json, large: 9007199254740992n }), false);
for (const [a, b] of [[undefined, null], [false, 0n], ['1', 1n], [1, 1n], [[1n, 2n], [2n, 1n]], [{}, { absent: undefined }]]) assert.equal(sameValue(a, b), false);
assert.equal(sameValue(json, parseJson(stringifyValue(json))), true);
assert.equal(stringifyValue(json, 0).includes('9007199254740993'), true);
assert.equal(parseJson('{"__proto__":{"polluted":true}}').__proto__.polluted, true);
assert.equal({}.polluted, undefined);
assert.equal(parseJson('"\\u0041\\n\\\\\\\""'), 'A\n\\"');
for (const source of ['', 'undefined', 'NaN', 'Infinity', '+1', '01', '-01', '1.', '.1', '1e', '1e+', '1 2', '[1,]', '{"x":1,}', '{x:1}', '/*x*/1', '"\\x00"', '"\\u123"', '"\n"', '"unterminated', '[', '{', 'truex', '\u00a01']) assert.throws(() => parseJson(source), SyntaxError, source);

const yaml = parseYaml('values: [null, "null", 1, "1", 1.0, false, 9007199254740993]\n');
assert.deepEqual(yaml.values, [null, 'null', 1n, '1', 1, false, 9007199254740993n]);
assert.equal(parseYaml('key: 1\nkey: 2').key, 2n);
assert.equal(parseYaml('a: &value {key: 1}\nb: *value').b.key, 1n);
const typedKeys = parseYaml('1: integer key\n"1": string key\n');
assert.equal(typedKeys.size, 2);
assert.equal(typedKeys.get(1n), 'integer key');
assert.equal(typedKeys.get('1'), 'string key');
assert.equal(sameValue(typedKeys, new Map([[1n, 'integer key'], ['1', 'string key']])), true);
assert.equal(sameValue(typedKeys, new Map([['1', 'string key']])), false);
const equivalentKeys = parseYaml('1: first\n1.0: second\ntrue: third\n');
assert.equal(equivalentKeys.size, 1);
assert.equal(equivalentKeys.get(1n), 'third');
assert.equal(hasField(equivalentKeys, true), true);
assert.equal(getField(equivalentKeys, 1), 'third');
setField(equivalentKeys, true, 'updated');
assert.equal(equivalentKeys.size, 1);
deleteField(equivalentKeys, 1);
assert.equal(equivalentKeys.size, 0);
assert.equal(getField(typedKeys, '1'), 'string key');
assert.equal(getField({ '1': 'string' }, 1n), undefined);
const temporal = parseYaml('date: 2026-10-12\ntime: 2026-10-12T00:00:00Z\nfirst: 2026-10-12T00:00:00.000001Z\nsecond: 2026-10-12T00:00:00.000002Z\nlocal: 2026-10-12T00:00:00\noffset: 2026-10-12T01:00:00+01:00\n');
assert.equal(sameValue(temporal.date, temporal.time), false);
assert.equal(sameValue(temporal.first, temporal.second), false);
assert.equal(sameValue(temporal.local, temporal.time), false);
assert.equal(sameValue(temporal.offset, temporal.time), true);
assert.equal(sameValue(structuredClone(temporal.first), temporal.first), true);
assert.equal(temporalText(temporal.first), '2026-10-12T00:00:00.000001+00:00');
assert.equal(stringifyValue(temporal.date), '"2026-10-12"');
const binary = parseYaml('value: !!binary ZGF0ZToyMDI2LTEwLTEy').value;
assert.equal(sameValue(binary, temporal.date), false);
assert.equal(stringifyValue(parseYaml('value: !!binary eA==').value), '"eA=="');
assert.equal(temporalText(parseYaml('value: 9999-12-31T23:00:00-01:00').value), '+010000-01-01T00:00:00.000000+00:00');
assert.equal(sameValue(parseYaml('value: 2020-01-01T00:00:00+01:99').value, parseYaml('value: 2020-01-01T00:00:00+02:39').value), true);
assert.throws(() => parseYaml('value: 2020-01-01T00:00:00+23:99'), /offset/);
for (const source of ['2026-02-30', '2026-10-12T24:00:00Z']) assert.throws(() => parseYaml(`value: ${source}`), /timestamp/);
assert.equal(canonicalJson({ '\u{10000}': 1n, '\uE000': 2n }), '{"\uE000":2,"\u{10000}":1}');
assert.equal(canonicalJson(new Map([['\u{10000}', 1n], ['\uE000', 2n]])), '{"\uE000":2,"\u{10000}":1}');
assert.throws(() => parseYaml('1: first\n1.0: second\n', { unified: true }), /unique|duplicate/i);
assert.throws(() => parseYaml('? [unhashable]\n: value\n'), /hashable/);
assert.deepEqual(parseYaml('n: n\ny: y\nvalues: [09, 1e3, 1.0e3, 1.0e+3, 01, 1:30, 0:30, .5, -.5]'), { n: 'n', y: 'y', values: ['09', '1e3', '1.0e3', 1000, 1n, 90n, '0:30', 0.5, '-.5'] });
for (const source of ['a: &x 1', 'a: *x', 'a: !!int 1', 'a: 1\na: 2', '[1,2]', 'null']) assert.throws(() => parseYaml(source, { unified: true }), /alias|anchor|tag|unique|mapping/i);

assert.equal(compareVersions('0.7.9', '0.7.10'), -1);
assert.equal(compareVersions('0.7.0rc1', '0.7.0'), -1);
assert.equal(compareVersions('0.7.0-beta', '0.7.0'), -1);
assert.equal(compareVersions('0.1.0-0', '0.1.0'), -1);
assert.equal(compareVersions('0.1.0-1.test', '0.1.0'), -1);
for (const version of ['0.1.0-post1', '0.1.0-rc.post1', '0.1.0-alpha.patch1']) assert.equal(compareVersions(version, '0.1.0'), -1);
for (const [a, b] of [['0.7.0', '0.7.0.post1'], ['0.7.0.post1', '0.7.0.post2'], ['0.7.0.patch1', '0.7.0.patch2']]) assert.equal(compareLegacyVersions(a, b), -1);

assert.deepEqual(assembleStream([[{ kind: 'tool_call', name: 'f', arguments: '{"n":9007199254740993}', complete: false }]]), []);
assert.deepEqual(assembleStream([[{ kind: 'tool_call', name: 'f', arguments: '{"n":', complete: false }], [{ kind: 'tool_call', arguments: '9007199254740993}', complete: true }]]), [{ kind: 'tool_call', name: 'f', arguments: { n: 9007199254740993n } }]);
const broken = [[{ kind: 'tool_call', name: 'f', arguments: 'bad' }]];
assert.equal(assembleStream(broken)[0].arguments, 'bad');
assert.deepEqual(assembleStream(broken, [{ kind: 'tool_call', name: 'f', arguments: {} }])[0].arguments, {});
assert.equal(assembleStream(broken, [{ kind: 'tool_call', name: 'other', arguments: {} }])[0].arguments, 'bad');
assert.deepEqual(assembleStream([[{ kind: 'tool_call', name: 'f', arguments: '{' }, { kind: 'tool_call', arguments: { n: 1n } }]]), [{ kind: 'tool_call', name: 'f', arguments: { n: 1n } }]);
assert.deepEqual(assembleStream([[new Map([[1n, 'metadata'], ['kind', 'text'], ['text', 'content']])]]), [{ kind: 'text', text: 'content' }]);
assert.throws(() => assembleStream([[{ kind: 'tool_call', name: 'f', arguments: 1n }]]), /argument fragment/);
const unifiedPayload = { chunks: [{ expected: [{ kind: 'text', text: 'stream' }] }], assembled: [{ kind: 'text', text: 'batch' }] };
assert.deepEqual(unifiedEvents(unifiedPayload), [{ kind: 'text', text: 'stream' }]);
assert.deepEqual(unifiedEvents(unifiedPayload, { stream: false }), [{ kind: 'text', text: 'batch' }]);

const base = { cases: { sample: { chunks: [{ expected: { test: ['old'], peer: ['kept'] } }, { expected: { test: ['tail'] }, normal_text: { test: 'tail' } }] } } };
mergeLegacyDocument(base, { cases: { sample: { chunks: [{ expected: ['new'] }] } } }, 'test', 'stream');
assert.deepEqual(base.cases.sample.chunks[0].expected, { test: ['new'], peer: ['kept'] });
assert.equal(Object.hasOwn(base.cases.sample.chunks[1].expected, 'test'), false);
mergeLegacyDocument(base, { cases: { sample: { unavailable: 'unsupported' } } }, 'test', 'stream');
assert.equal(base.cases.sample.unavailable.test, 'unsupported');
assert.equal(Object.hasOwn(base.cases.sample.chunks[0].expected, 'test'), false);
mergeLegacyDocument(base, { cases: { sample: { exception: 'error' } } }, 'test', 'stream');
assert.equal(Object.hasOwn(base.cases.sample.unavailable, 'test'), false);
assert.equal(base.cases.sample.exception.test, 'error');

const metadata = JSON.parse(await readFile(new URL('../viewer-catalog.json', import.meta.url), 'utf8')).metadata;
for (const owner of metadata.implementations.filter(implementation => implementation.legacy_key)) for (const selectedKey of [owner.key, owner.legacy_key]) {
  const legacy = owner.legacy_key;
  const record = { unavailable: { [legacy]: 'old unavailable', other: 'keep' }, exception: { [owner.key]: 'old exception' }, chunks: [
    { expected: { [legacy]: [{ index: 0n, name: 'stale', arguments: '{}' }], [owner.key]: [], other: ['keep'] }, normal_text: { [legacy]: 'staleHead', [owner.key]: 'staleCanonical' } },
    { expected: { [legacy]: [{ index: 0n, name: 'staleTail', arguments: '{}' }] }, normal_text: { [legacy]: 'staleTail', [owner.key]: 'staleCanonicalTail' } },
  ] };
  const document = { cases: { sample: record } };
  const apply = value => mergeLegacyDocument(document, { cases: { sample: value } }, selectedKey, 'stream', metadata.implementations);
  apply({ chunks: [{ expected: [{ index: 0n, name: 'fresh', arguments: '{}' }], normal_text: 'fresh' }] });
  const result = deriveStreamExpected(record, metadata, [owner.key])[owner.key];
  assert.equal(result.normal_text, 'fresh', `${selectedKey}: newer short capture clears stale alias tail`);
  assert.deepEqual(result.calls, [{ name: 'fresh', arguments: {} }]);
  assert.deepEqual(record.chunks[0].expected.other, ['keep']);
  assert.equal(record.unavailable.other, 'keep');
  for (const key of [owner.key, legacy]) {
    assert.equal(Object.hasOwn(record.chunks[1].normal_text, key), false);
    assert.equal(Object.hasOwn(record.chunks[1].expected, key), false);
    assert.equal(Object.hasOwn(record.unavailable, key), false);
    assert.equal(Object.hasOwn(record.exception, key), false);
  }
  apply({ unavailable: 'new unavailable' });
  assert.equal(deriveStreamExpected(record, metadata, [owner.key])[owner.key].unavailable, 'new unavailable');
  apply({ exception: 'new exception' });
  assert.equal(deriveStreamExpected(record, metadata, [owner.key])[owner.key].exception, 'new exception');
  apply({ chunks: [{ expected: [], normal_text: 'recovered' }] });
  assert.equal(deriveStreamExpected(record, metadata, [owner.key])[owner.key].normal_text, 'recovered');
}

const bytes = new TextEncoder().encode('validated');
const resource = { path: 'a.yaml', size: bytes.length, sha256: await sha256(bytes) };
Object.defineProperty(globalThis, 'indexedDB', { configurable: true, value: { open() { throw new Error('storage blocked'); } } });
let fetches = 0;
const store = new HttpResourceStore({ baseUrl: new URL('http://localhost/'), fetchImpl: async () => { fetches++; return { ok: true, arrayBuffer: async () => bytes.buffer }; } });
await Promise.all([store.read(resource), store.read(resource)]);
assert.equal(fetches, 1);
const failing = new HttpResourceStore({ baseUrl: new URL('http://localhost/'), fetchImpl: async () => ({ ok: false, status: 404 }) });
const missing = { ...resource, sha256: '0'.repeat(64) };
await assert.rejects(failing.read(missing), /404/);
await assert.rejects(failing.read(missing), /404/);
console.log('lossless values, YAML contracts, stream replacement, and storage fallback passed');

// Each incoming capture replaces the equivalent identity, preserving peer state.
for (const owner of metadata.implementations.filter(item => item.legacy_key)) {
  for (const mode of ['batch', 'reasoning']) for (const [oldKey, newKey] of [[owner.key, owner.legacy_key], [owner.legacy_key, owner.key]]) {
    const document = { cases: { sample: { expected: { [oldKey]: { normal_text: 'old' }, other: { normal_text: 'keep' } } } } };
    mergeLegacyDocument(document, { cases: { sample: { expected: { [newKey]: { normal_text: 'fresh' } } } } }, owner.key, mode, metadata.implementations);
    assert.deepEqual(document.cases.sample.expected, { other: { normal_text: 'keep' }, [owner.key]: { normal_text: 'fresh' } }, `${mode}: ${oldKey} -> ${newKey}`);
  }
}
