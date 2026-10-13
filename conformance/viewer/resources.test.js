// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import assert from 'node:assert/strict';
import { createWorkerParser, HttpResourceStore, sha256 } from './data.js';
import { FixtureResources } from './resources.js';

const yamlBytes = new TextEncoder().encode('answer: 42\n');
const descriptor = { path: 'fixtures/input.yaml', sha256: await sha256(yamlBytes), size: yamlBytes.length };
const response = (bytes, status = 200) => ({ ok: status === 200, status, arrayBuffer: async () => bytes.slice().buffer });
let downloads = 0;
let parses = 0;
const store = new HttpResourceStore({ baseUrl: 'http://localhost/nested/', fetchImpl: async (url, options) => {
  downloads++;
  assert.equal(url.origin + url.pathname, downloads === 1 ? 'http://localhost/nested/fixtures/input.yaml' : 'http://localhost/nested/fixtures/alias.yaml');
  assert.equal(url.searchParams.get('sha256'), descriptor.sha256);
  assert.equal(options.cache, 'no-store');
  return response(yamlBytes);
} });
const resources = new FixtureResources({}, store, {
  parseYaml: async () => { parses++; return { answer: 42 }; },
  parseArchive: async () => new Map([['family/input.yaml', yamlBytes]]),
});
const alias = { ...descriptor, path: 'fixtures/alias.yaml' };
const [original, aliased] = await Promise.all([resources.yaml(descriptor), resources.yaml(alias)]);
assert.equal(downloads, 1);
assert.equal(parses, 1);
assert.equal(original.resource, descriptor.path);
assert.equal(original.member, descriptor.path);
assert.equal(aliased.resource, alias.path);
assert.equal(aliased.member, alias.path);
assert.deepEqual(aliased.sourceBytes, yamlBytes);
await assert.rejects(resources.yaml({ ...alias, size: descriptor.size + 1 }), /size mismatch/);
for (const path of ['../outside.yaml', '/absolute.yaml', 'javascript:invalid', 'a\\b.yaml', 'a//b.yaml']) {
  const invalid = { ...descriptor, path };
  await assert.rejects(store.read(invalid), /unsafe|repository-relative/);
  await assert.rejects(resources.yaml(invalid), /unsafe|repository-relative/);
  await assert.rejects(resources.archive(invalid), /unsafe|repository-relative/);
}
for (const invalid of [{ ...descriptor, sha256: 'invalid' }, { ...descriptor, size: -1 }, { ...descriptor, size: '11' }]) {
  await assert.rejects(store.read(invalid), /invalid resource hash or size/);
}
assert.equal(downloads, 2, 'only the wrong-size valid descriptor may reach transport');

// Archive aliases share extraction, while each document retains caller provenance.
const archive = { ...descriptor, path: 'fixtures/archive.tar.gz' };
const archiveAlias = { ...archive, path: 'fixtures/archive-alias.tar.gz' };
const archived = await resources.yaml(archive, 'family/input.yaml');
const archivedAlias = await resources.yaml(archiveAlias, 'family/input.yaml');
assert.equal(archived.member, 'family/input.yaml');
assert.equal(archivedAlias.resource, archiveAlias.path);
assert.equal(archived.resource, archive.path);
await assert.rejects(resources.archive({ ...archive, path: '../cached-archive.tar.gz' }), /unsafe/);
await assert.rejects(resources.yaml({ ...archive, path: '../cached-archive.tar.gz' }, 'family/input.yaml'), /unsafe/);
await assert.rejects(resources.yaml(archive, 'family/missing.yaml'), /archive member missing/);

for (const [label, bad, expected] of [
  ['corrupt', response(new Uint8Array(yamlBytes.length)), /SHA-256 mismatch/],
  ['missing', response(new Uint8Array(), 404), /fetch failed \(404\)/],
  ['LFS pointer', response(new TextEncoder().encode('version https://git-lfs.github.com/spec/v1\noid sha256:abc\n')), /not hydrated/],
]) {
  let attempts = 0;
  const retryStore = new HttpResourceStore({ baseUrl: 'http://localhost/', fetchImpl: async () => ++attempts === 1 ? bad : response(yamlBytes) });
  const retryResources = new FixtureResources({}, retryStore, { parseYaml: async () => ({ answer: 42 }) });
  await assert.rejects(retryResources.yaml(descriptor), expected, label);
  assert.deepEqual((await retryResources.yaml(descriptor)).sourceBytes, yamlBytes);
  assert.equal(attempts, 2, `${label} failures must be evicted for Retry`);
}

// Plain HTTP and inaccessible persistent storage must still use validated memory.
globalThis.indexedDB = { open() { throw new Error('storage denied'); } };
try {
  let attempts = 0;
  const deniedStore = new HttpResourceStore({ baseUrl: 'http://localhost/', fetchImpl: async () => { attempts++; return response(yamlBytes); } });
  const [firstBytes, secondBytes] = await Promise.all([deniedStore.read(descriptor), deniedStore.read(alias)]);
  assert.equal(attempts, 1);
  firstBytes[0] = 0;
  assert.deepEqual(secondBytes, yamlBytes, 'callers must not share mutable byte arrays');
  assert.deepEqual(await deniedStore.read(descriptor), yamlBytes);
  deniedStore.memory.set(descriptor.sha256, new Uint8Array(yamlBytes.length));
  assert.deepEqual(await deniedStore.read(descriptor), yamlBytes);
  assert.equal(attempts, 2, 'corrupt cached bytes must be revalidated');
} finally {
  delete globalThis.indexedDB;
}
console.log('resource descriptors, aliases, deduplication and transport retry passed');

const instances = [];
class TestWorker {
  constructor() { this.listeners = new Map(); instances.push(this); }
  addEventListener(type, callback) { this.listeners.set(type, callback); }
  terminate() { this.terminated = true; }
  postMessage(message) { this.message = message; }
  reply(result) { this.listeners.get('message')({ data: { id: this.message.id, result } }); }
  fail() { this.listeners.get('error')({ message: 'worker import failed', preventDefault() {} }); }
}
globalThis.Worker = TestWorker;
const parser = createWorkerParser();
const first = parser.parseYaml(new Uint8Array([1]), {});
const rejected = assert.rejects(first, /worker import failed/);
instances[0].fail();
await rejected;
assert.equal(instances[0].terminated, true);
const retry = parser.parseYaml(new Uint8Array([2]), {});
assert.equal(instances.length, 2);
instances[1].reply({ large: 9007199254740993n });
assert.deepEqual(await retry, { large: 9007199254740993n });
const pending = parser.parseYaml(new Uint8Array([3]), {});
const disposed = assert.rejects(pending, /disposed/);
parser.terminate();
await disposed;
assert.equal(instances[1].terminated, true);
await assert.rejects(parser.parseYaml(new Uint8Array(), {}), /disposed/);
delete globalThis.Worker;
console.log('worker failure retry and disposal passed');

// Provenance replacement follows the same alias identity as observation replacement.
for (const mode of ['stream', 'reasoning']) {
  const root = 'probe';
  const input = { path: `conformance/fixtures/${root}/inputs.tar.gz` };
  const capture = { path: `conformance/fixtures/${root}/vllm_python-2.0.0.tar.gz` };
  const registry = [{ key: 'vllm_python', legacy_key: 'vllm' }];
  const reader = new FixtureResources({ metadata: { implementations: registry }, sources: { archives: [input, capture] } }, null, {});
  reader.legacyDocuments = async resource => new Map([['probe/case.yaml', { value: resource === input
    ? { captured_with: { vllm: '1.0.0', other: 'keep' }, cases: { sample: { expected: { vllm: { normal_text: 'old' } }, chunks: [{}] } } }
    : { cases: { sample: mode === 'stream' ? { chunks: [{ expected: [], normal_text: 'fresh' }] } : { expected: { vllm: { normal_text: 'fresh' } } } } } }]]);
  const records = await reader.legacy({ root, implementation: 'vllm_python', version: '2.0.0', mode });
  assert.deepEqual(records.get('probe/case.yaml').value.captured_with, { other: 'keep', vllm_python: '2.0.0' });
  if (mode === 'reasoning') assert.deepEqual(records.get('probe/case.yaml').value.cases.sample.expected, { vllm_python: { normal_text: 'fresh' } });
}
console.log('stream and reasoning provenance replace aliases without stale stamps');
