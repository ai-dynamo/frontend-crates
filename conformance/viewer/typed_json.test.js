// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import assert from 'node:assert/strict';
import { decodeBase64, decodeTyped } from './typed_json.js';
import { temporalText } from './values.js';

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
