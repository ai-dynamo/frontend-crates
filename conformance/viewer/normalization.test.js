// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { parseYaml } from './yaml_data.js';
import { sameValue, getField, canonicalJson } from './values.js';
import { sha256 } from './data.js';
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

await test('archived result references preserve full payloads and independent receipts', async () => {
  const fixture = parseYaml(await readFile(new URL('../utils/tests/fixtures/normalized-results.yaml', import.meta.url), 'utf8'), { unified: true });
  const expanded = await expandSnapshot(fixture.snapshot, fixture.authored);
  assert(sameValue(expanded, fixture.expected_snapshot));
  const metadata = expanded.records.one.value.document.record_metadata;
  assert.deepEqual(Object.keys(metadata.verified_recaptures[0]), ['result', 'provenance', 'resolution']);
  metadata.verified_recaptures[0].result.chunks.length = 0;
  assert.equal(metadata.verified_recapture.result.chunks.length, 2);
  assert.equal(metadata.verified_recaptures[1].result.chunks.length, 2);
  assert(sameValue(await expandSnapshot(fixture.snapshot, fixture.authored), fixture.expected_snapshot));
  for (const mutate of [
    m => { m.verified_recaptures[0].result_ref = '0'.repeat(64); },
    m => { delete m.verified_recapture; },
    m => { m.verified_recapture.result_ref = fixture.payload_sha256; },
    m => { m.verified_recaptures[0].result = {}; },
    m => { m.verified_recapture.result.chunks.reverse(); },
    m => { m.verified_recapture.result.literal.float = 1000000000000000.25; },
    m => { m.verified_recapture.result.literal = new Map([[1n, 'typed key']]); },
    m => { m.verified_recapture.result.literal.date = new Uint16Array([100]); },
  ]) {
    const invalid = structuredClone(fixture.snapshot);
    mutate(invalid.contexts.shared.record_metadata);
    await assert.rejects(expandSnapshot(invalid, fixture.authored));
  }
  for (const payload of [{ value: 1.0 }, { value: Infinity }, new Map([[1n, 'typed key']]),
    { value: new Uint8Array([97]) }]) {
    const invalid = structuredClone(fixture.snapshot);
    const metadata = invalid.contexts.shared.record_metadata;
    metadata.verified_recapture.result = payload;
    const digest = await sha256(new TextEncoder().encode(canonicalJson(payload)));
    for (const archive of metadata.verified_recaptures) archive.result_ref = digest;
    await assert.rejects(expandSnapshot(invalid, fixture.authored));
  }
  const unused = structuredClone(fixture.snapshot);
  unused.contexts.unused = { record_metadata: { verified_recaptures: [{ result_ref: 'bad' }] } };
  await assert.rejects(expandSnapshot(unused, fixture.authored));
});
