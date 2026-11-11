// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { parseYaml } from './yaml_data.js';
import { sameValue, getField } from './values.js';
import { validateUnifiedInputs, validateUnifiedSnapshot, validateResolvedUnified } from './resources.js';
import { expandFamily, expandSnapshot } from './normalization.js';

await test('normalized sources preserve typed metadata and frozen request identity', async () => {
  const fixture = parseYaml(await readFile(new URL('../utils/tests/fixtures/normalized.yaml', import.meta.url), 'utf8'), { unified: true });
  for (const variant of fixture.variants) assert(sameValue(await expandFamily(variant.authored), variant.expected));
  const authored = fixture.authored;
  const snapshot = fixture.snapshot;
  assert(sameValue(await expandFamily(authored), fixture.expected_family));
  assert(sameValue(await expandSnapshot(snapshot, authored), fixture.expected_snapshot));
  const expanded = await expandSnapshot(snapshot, authored);
  expanded.records.one.value.stimulus.inline.tools.length = 0;
  assert(sameValue(await expandSnapshot(snapshot, authored), fixture.expected_snapshot));
  for (const mutate of [
    value => { value.tools = {}; },
    value => { value.request_defaults.finish_reason = 'length'; },
    value => { Object.values(value.requests)[0].chunks = null; },
  ]) {
    const invalid = structuredClone(authored); mutate(invalid);
    await assert.rejects(expandFamily(invalid));
  }
  for (const mutate of [
    value => { value.producers = {}; },
    value => { value.records.one.context = null; },
    value => { value.records.one.error = { code: 'extra outcome' }; },
    value => { value.records.one.stimulus = { ref: 'current' }; },
  ]) {
    const invalid = structuredClone(snapshot); mutate(invalid);
    await assert.rejects(expandSnapshot(invalid, authored));
  }
});

await test('malformed context definitions fail through full reader admission', async () => {
  const fixture = parseYaml(await readFile(new URL('../utils/tests/fixtures/normalized.yaml', import.meta.url), 'utf8'), { unified: true });
  const resource = { path: 'conformance/fixtures-unified-v2/families/sample/dynamo_v2-0.1.0.yaml', family: 'sample', implementation: 'dynamo_v2', version: '0.1.0', kind: 'history', record_contract: 'stream' };
  const admit = async stored => {
    const authored = await expandFamily(fixture.authored);
    validateUnifiedInputs(authored, 'sample', 'authored');
    const snapshot = await expandSnapshot(stored, fixture.authored);
    snapshot.records.one.member = resource.path;
    await validateUnifiedSnapshot(snapshot, authored, resource);
    await validateResolvedUnified(snapshot.records, authored, resource, [resource]);
    return snapshot;
  };
  await admit(fixture.snapshot);
  const typed = structuredClone(fixture.snapshot);
  typed.contexts[typed.record_defaults.context] = new Map([[1n, 'integer key'], ['1', 'string key']]);
  const expanded = await admit(typed);
  const metadata = expanded.records.one.value.document;
  assert.equal(metadata.get(1n), 'integer key');
  assert.equal(metadata.get('1'), 'string key');
  assert.equal(getField(metadata, 'captured_with').dynamo_v2, '0.1.0');
  for (const badContext of [['untrusted-array'], 'untrusted-string']) {
    const invalid = structuredClone(fixture.snapshot);
    const context = invalid.record_defaults.context;
    invalid.contexts[context] = badContext;
    await assert.rejects(admit(invalid));
  }
});

await test('normalized authored identities remain ordinary mapping keys', async () => {
  const fixture = parseYaml(await readFile(new URL('../utils/tests/fixtures/normalized.yaml', import.meta.url), 'utf8'), { unified: true });
  for (const id of ['__proto__', 'constructor', 'toString']) {
    const raw = structuredClone(fixture.authored);
    const stored = raw.cases.one;
    raw.cases = Object.fromEntries([[id, stored]]);
    const expanded = await expandFamily(raw);
    validateUnifiedInputs(expanded, 'sample', 'authored');
    assert(Object.hasOwn(expanded.cases, id));
    assert.equal(expanded.cases[id].scenario, id);
    const snapshot = structuredClone(fixture.snapshot);
    snapshot.records = Object.fromEntries([[id, snapshot.records.one]]);
    const resolved = await expandSnapshot(snapshot, raw);
    assert(Object.hasOwn(resolved.records, id));
  }
});
