// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { test } from 'node:test';

await test('values contracts', async () => {
const savedGlobals = Object.getOwnPropertyDescriptors(globalThis);
try {
const { default: assert } = await import('node:assert/strict');
const { parseJson, sameValue, stringifyValue, canonicalJson, temporalText } = await import('./values.js');
const { hasField, getField, setField, deleteField } = await import('./values.js');
const { HttpResourceStore, sha256 } = await import('./data.js');
const { parseYaml } = await import('./yaml_data.js');
const { assembleStream, compareLegacyVersions, compareVersions, unifiedEvents } = await import('./model.js');


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

const bytes = new TextEncoder().encode('validated');
const resource = { path: 'a.yaml', size: bytes.length, sha256: await sha256(bytes) };
Object.defineProperty(globalThis, 'indexedDB', { configurable: true, value: { open() { throw new Error('storage blocked'); } } });
let fetches = 0;
const store = new HttpResourceStore({ allowSourceFiles: true, baseUrl: new URL('http://localhost/'), fetchImpl: async url => { fetches++; return { ok: true, arrayBuffer: async () => bytes.buffer }; } });
await Promise.all([store.read(resource), store.read(resource)]);
assert.equal(fetches, 1);
const failing = new HttpResourceStore({ allowSourceFiles: true, baseUrl: new URL('http://localhost/'), fetchImpl: async () => ({ ok: false, status: 404 }) });
const missing = { ...resource, sha256: '0'.repeat(64) };
await assert.rejects(failing.read(missing), /404/);
await assert.rejects(failing.read(missing), /404/);
console.log('lossless values, YAML contracts, stream replacement, and storage fallback passed');


} finally {
  for (const key of Object.getOwnPropertyNames(globalThis)) {
    if (!Object.hasOwn(savedGlobals, key) && Object.getOwnPropertyDescriptor(globalThis, key).configurable) delete globalThis[key];
  }
  for (const [key, descriptor] of Object.entries(savedGlobals)) {
    if (descriptor.configurable) Object.defineProperty(globalThis, key, descriptor);
  }
}

});

await test('typed_json contracts', async () => {
const savedGlobals = Object.getOwnPropertyDescriptors(globalThis);
try {
const { default: assert } = await import('node:assert/strict');
const { decodeBase64, decodeTyped } = await import('./values.js');
const { temporalText } = await import('./values.js');


assert.equal(decodeTyped(['int', '9007199254740993']), 9007199254740993n);
assert.equal(decodeTyped(['float', '1.0']), 1);
assert.equal(Object.is(decodeTyped(['float', '-0.0']), -0), true);
assert.equal(Number.isNaN(decodeTyped(['float', 'nan'])), true);
assert.equal(decodeTyped(['float', 'inf']), Infinity);
assert.equal(decodeTyped(['float', '-inf']), -Infinity);
assert.deepEqual(decodeTyped(['object', [[['int', '1'], ['str', 'integer']], [['str', '1'], ['null']]]]), new Map([[1n, 'integer'], ['1', null]]));
assert.throws(() => decodeTyped(['object', [[['str', 'same'], ['null']], [['str', 'same'], ['null']]]]), /duplicate typed mapping key/);
assert.throws(() => decodeTyped(['object', [[['int', '1'], ['null']], [['float', '1.0'], ['null']]]]), /duplicate typed mapping key/);
assert.throws(() => decodeTyped(['object', [[['bool', true], ['null']], [['int', '1'], ['null']]]]), /duplicate typed mapping key/);
assert.throws(() => decodeTyped(['object', [[['binary', 'AQI='], ['null']], [['binary', 'AQI='], ['null']]]]), /duplicate typed mapping key/);
assert.deepEqual(decodeTyped(['object', [[['binary', 'AQI='], ['str', 'first']], [['binary', 'AQM='], ['str', 'second']]]]), new Map([[new Uint8Array([1, 2]), 'first'], [new Uint8Array([1, 3]), 'second']]));
const largeMapping = (entries) => ['object', [...entries, ...Array.from({ length: 8 }, (_, index) => [['str', `filler-${index}`], ['null']])]];
assert.throws(() => decodeTyped(largeMapping([[['str', 'same'], ['null']], [['str', 'same'], ['null']]])), /duplicate typed mapping key/);
assert.throws(() => decodeTyped(largeMapping([[['bool', true], ['null']], [['int', '1'], ['null']], [['float', '1.0'], ['null']]])), /duplicate typed mapping key/);
assert.throws(() => decodeTyped(largeMapping([[['date', '2026-10-15'], ['null']], [['date', '2026-10-15'], ['null']]])), /duplicate typed mapping key/);
assert.throws(() => decodeTyped(largeMapping([[['datetime', '2026-10-15T00:00:00'], ['null']], [['datetime', '2026-10-15T00:00:00'], ['null']]])), /duplicate typed mapping key/);
assert.throws(() => decodeTyped(largeMapping([[['binary', 'AQI='], ['null']], [['binary', 'AQI='], ['null']]])), /duplicate typed mapping key/);
assert.equal(decodeTyped(largeMapping([[['str', 'integer:1'], ['null']], [['int', '1'], ['null']]])).size, 10);
assert.equal(decodeTyped(largeMapping([[['binary', 'AQI='], ['null']], [['binary', 'AQM='], ['null']]])).size, 10);
assert.equal(decodeTyped(largeMapping([[['date', '2026-10-15'], ['null']], [['datetime', '2026-10-15T00:00:00'], ['null']]])).size, 10);
const object = decodeTyped(['object', [[['str', '__proto__'], ['bool', true]], [['str', 'explicit'], ['null']]]]);
assert.equal(Object.hasOwn(object, '__proto__'), true);
assert.equal(Object.hasOwn(object, 'missing'), false);
assert.equal(object.explicit, null);
assert.deepEqual(decodeTyped(['binary', 'eA==']), new Uint8Array([120]));
const binary = decodeTyped(['binary', 'AQID'.repeat(1024 * 1024)]);
assert.equal(binary.length, 3 * 1024 * 1024, 'large binary values must not exhaust the regex stack');
for (let index = 0; index < binary.length; index++) assert.equal(binary[index], index % 3 + 1);
const priorBase64 = /^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/;
const alphabet = ['A', 'B', '+', '/', '=', '!', '\n'];
const checkBase64Language = (value, remaining) => {
  if (priorBase64.test(value)) assert.deepEqual(decodeBase64(value), new Uint8Array(Buffer.from(value, 'base64')));
  else assert.throws(() => decodeBase64(value), /invalid typed binary/);
  if (remaining) for (const character of alphabet) checkBase64Language(value + character, remaining - 1);
};
checkBase64Language('', 4);
for (const value of [null, 1, [], 'A'.repeat(4 * 1024 * 1024 - 1) + '!', 'AA=A', '=AAA', 'A===', '====', 'AA==AAAA']) {
  assert.throws(() => decodeTyped(['binary', value]), /invalid typed binary/);
}
assert.equal(temporalText(decodeTyped(['datetime', '2026-10-15T01:00:00.123456+01:00'])), '2026-10-15T00:00:00.123456+00:00');
assert.equal(temporalText(decodeTyped(['date', '2026-10-15'])), '2026-10-15');
for (const node of [null, {}, [], ['unknown'], ['null', null], ['int', 1], ['int', '01'], ['int', '1.0'], ['float', 'Infinity'], ['float', '1e999'], ['float', ''], ['str', false], ['bool', 1], ['array', {}], ['array', [1]], ['object', [] , 0], ['object', [[['str', 'key']]]], ['object', [[['array', []], ['null']]]], ['object', [[['str', 'x'], ['null']], [['str', 'x'], ['null']]]], ['object', [[['int', '1'], ['null']], [['bool', true], ['null']]]], ['date', '2026-02-30'], ['date', '2026-01-01T00:00:00'], ['binary', '!']]) {
  assert.throws(() => decodeTyped(node), /typed|timestamp|kind/);
}
console.log('typed JSON values, missing fields, temporal values, mapping keys, and malformed nodes passed');

} finally {
  for (const key of Object.getOwnPropertyNames(globalThis)) {
    if (!Object.hasOwn(savedGlobals, key) && Object.getOwnPropertyDescriptor(globalThis, key).configurable) delete globalThis[key];
  }
  for (const [key, descriptor] of Object.entries(savedGlobals)) {
    if (descriptor.configurable) Object.defineProperty(globalThis, key, descriptor);
  }
}

});
