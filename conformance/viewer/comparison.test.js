// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import assert from 'node:assert/strict';
import { comparisonEntries, comparisonFacts, declaredLeakTokens, equivalenceGroups, legacyOutcome } from './comparison.js';

const tokens = declaredLeakTokens({ family: { pairs: [['<tool_call>', '</tool_call>']], leak: ['<invoke'] } });
const first = { calls: [{ name: 'f', arguments: '{"large":9007199254740993,"n":1}' }], normal_text: '' };
const reordered = { calls: [{ name: 'f', arguments: '{"n":1,"large":9007199254740993}' }], normal_text: '', explanation: 'captured difference' };
const different = { calls: [{ name: 'f', arguments: '{"large":9007199254740992,"n":1}' }], normal_text: '' };
const typed = { calls: [{ name: 'f', arguments: '{"large":9007199254740993,"n":1.0}' }], normal_text: '' };
const blocks = { first, reordered, different, typed, unavailable: { unavailable: 'no parser' }, absent: null, exception: { exception: 'failed' }, otherException: { exception: 'other failure' }, error: { error: 'declared error' }, leak: { calls: [], normal_text: '<tool_call' } };
const entries = comparisonEntries(blocks, tokens);
assert.equal(entries.first.sig, entries.reordered.sig);
assert.notEqual(entries.first.sig, entries.different.sig);
assert.notEqual(entries.first.sig, entries.typed.sig);
assert.equal(entries.unavailable.sig, entries.absent.sig);
assert.notEqual(entries.exception.sig, entries.otherException.sig);
assert.equal(entries.exception.err, 1);
assert.equal(entries.error.err, 0);
assert.equal(entries.leak.leak, 1);
assert.deepEqual(equivalenceGroups(entries).first.sig, ['first', 'reordered']);
assert.equal(legacyOutcome({ unavailable: 'parser not captured: panic' }, tokens).status, 'problem');
assert.equal(legacyOutcome({ unavailable: 'no parser' }, tokens).status, 'na');
assert.equal(legacyOutcome({ error: 'expected error' }, tokens).errorKind, 'expected_error');
assert.equal(legacyOutcome({ error: { message: 'panic' } }, tokens).errorKind, 'parser_error');
const implementations = [{ key: 'base', legacy_key: 'oldBase' }, { key: 'peer', legacy_key: 'oldPeer' }];
const facts = comparisonFacts({ expected: { oldBase: first, peer: first, oldPeer: different } }, implementations, 'base', tokens);
assert.equal(facts[0].agrees, true);
assert.equal(facts[1].agrees, true);
assert.equal(facts[1].present, true);
assert.equal(comparisonFacts(null, implementations, 'base', tokens)[0].agrees, null);
const prototypeExpected = JSON.parse('{"__proto__":{"base":{"calls":[],"normal_text":"phantom"}}}');
assert.equal(comparisonFacts({ expected: prototypeExpected }, implementations, 'base', tokens)[0].present, false);
const typedBlock = new Map([[1n, 'metadata'], ['calls', []], ['normal_text', 'captured']]);
assert.equal(comparisonFacts({ expected: { base: typedBlock } }, implementations, 'base', tokens)[0].present, true);
assert.equal(legacyOutcome(new Date(), tokens).status, 'na');
console.log('typed comparison equivalence, leaks, errors, missing captures, and aliases passed');
