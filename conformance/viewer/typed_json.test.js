// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import assert from 'node:assert/strict';
import { decodeTyped } from './typed_json.js';
import { temporalText } from './values.js';

assert.equal(decodeTyped(['int', '9007199254740993']), 9007199254740993n);
assert.equal(decodeTyped(['float', '1.0']), 1);
assert.equal(Object.is(decodeTyped(['float', '-0.0']), -0), true);
assert.equal(Number.isNaN(decodeTyped(['float', 'nan'])), true);
assert.equal(decodeTyped(['float', 'inf']), Infinity);
assert.equal(decodeTyped(['float', '-inf']), -Infinity);
assert.deepEqual(decodeTyped(['object', [[['int', '1'], ['str', 'integer']], [['str', '1'], ['null']]]]), new Map([[1n, 'integer'], ['1', null]]));
const object = decodeTyped(['object', [[['str', '__proto__'], ['bool', true]], [['str', 'explicit'], ['null']]]]);
assert.equal(Object.hasOwn(object, '__proto__'), true);
assert.equal(Object.hasOwn(object, 'missing'), false);
assert.equal(object.explicit, null);
assert.deepEqual(decodeTyped(['binary', 'eA==']), new Uint8Array([120]));
assert.equal(temporalText(decodeTyped(['datetime', '2026-10-15T01:00:00.123456+01:00'])), '2026-10-15T00:00:00.123456+00:00');
assert.equal(temporalText(decodeTyped(['date', '2026-10-15'])), '2026-10-15');
for (const node of [null, {}, [], ['unknown'], ['null', null], ['int', 1], ['int', '01'], ['int', '1.0'], ['float', 'Infinity'], ['float', '1e999'], ['float', ''], ['str', false], ['bool', 1], ['array', {}], ['array', [1]], ['object', [] , 0], ['object', [[['str', 'key']]]], ['object', [[['array', []], ['null']]]], ['object', [[['str', 'x'], ['null']], [['str', 'x'], ['null']]]], ['object', [[['int', '1'], ['null']], [['bool', true], ['null']]]], ['date', '2026-02-30'], ['date', '2026-01-01T00:00:00'], ['binary', '!']]) {
  assert.throws(() => decodeTyped(node), /typed|timestamp|kind/);
}
console.log('typed JSON values, missing fields, temporal values, mapping keys, and malformed nodes passed');
