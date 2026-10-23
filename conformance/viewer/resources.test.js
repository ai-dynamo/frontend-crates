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
const store = new HttpResourceStore({ allowSourceFiles: true, baseUrl: 'http://localhost/nested/', fetchImpl: async (url, options) => {
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
  const retryStore = new HttpResourceStore({ allowSourceFiles: true, baseUrl: 'http://localhost/', fetchImpl: async url => ++attempts === 1 ? bad : response(yamlBytes) });
  const retryResources = new FixtureResources({}, retryStore, { parseYaml: async () => ({ answer: 42 }) });
  await assert.rejects(retryResources.yaml(descriptor), expected, label);
  assert.deepEqual((await retryResources.yaml(descriptor)).sourceBytes, yamlBytes);
  assert.equal(attempts, 2, `${label} failures must be evicted for Retry`);
}

// Plain HTTP and inaccessible persistent storage must still use validated memory.
globalThis.indexedDB = { open() { throw new Error('storage denied'); } };
try {
  let attempts = 0;
  const deniedStore = new HttpResourceStore({ allowSourceFiles: true, baseUrl: 'http://localhost/', fetchImpl: async url => { attempts++; return response(yamlBytes); } });
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

const inputDocument = { input_document: { family: 'sample' }, golden_document: { family: 'sample' }, cases: {
  id: { lifecycle: 'active', scenario: 'sample', description: 'sample', policy: [], display_id: 'UNIFIED.1', historical_ids: [],
    request: { input: 'sample', init: {}, finish_reason: 'stop', tools: [], chunks: [] }, golden: { assembled: [] } },
} };
const latestPath = 'conformance/fixtures-unified-v2/families/sample/dynamo_v2-0.2.0.yaml';
const inputSource = { path: 'conformance/fixtures-unified-v2/families/sample/inputs_and_golden.yaml', sha256: '1'.repeat(64), size: 1,
  kind: 'inputs-and-golden', family: 'sample' };
const latestSource = { path: latestPath, sha256: '2'.repeat(64), size: 1, kind: 'history', family: 'sample',
  implementation: 'dynamo_v2', version: '0.2.0' };
const snapshotSource = { schema_version: 2, provenance: { status: 'legacy', captured_with: { dynamo_v2: '0.2.0' } }, document: {},
  records: { id: { value: { case_key: 'UNIFIED.1', stimulus: { ref: 'current' }, observation: { value: { assembled: [] } },
    document: { captured_with: { dynamo_v2: '0.2.0' }, record_metadata: { retained: null, verified_recapture: { receipt: 'checkpoint' } } } }, member: latestPath } } };
const sourceReads = [];
const checkoutStore = { demand() {}, async read(resource) { sourceReads.push(resource.path); return new Uint8Array([1]); } };
const directReader = new FixtureResources({
  sources: { unified: [inputSource, latestSource], original_unified: [latestSource] },
  metadata: { unified_aliases: null, unified_capture_policy: { dynamo_v2: { mode: 'stream' } } },
}, checkoutStore, { parseYaml: async (_bytes, { path }) => path.endsWith('inputs_and_golden.yaml') ? inputDocument : snapshotSource });
const selectionDescriptor = { source: { implementation: 'dynamo_v2', version: '0.2.0', mode: 'unified', native_unified_families: ['sample'] } };
await directReader.prepareUnified([selectionDescriptor]);
const directResult = await directReader.unified('sample', 'dynamo_v2', '0.2.0');
assert.equal(directResult.records.size, 1);
assert.ok(directResult.records.get('id').value.document.record_metadata.verified_recapture);
const derivedResult = await directReader.unified('sample', 'dynamo_v2', '0.2.1');
assert.equal(Object.hasOwn(derivedResult.records.get('id').value.document.record_metadata, 'verified_recapture'), false);
assert.equal(derivedResult.records.get('id').value.document.record_metadata.retained, null);
assert.ok(snapshotSource.records.id.value.document.record_metadata.verified_recapture, 'projection cannot mutate permanent evidence');
assert.ok(sourceReads.includes(latestPath));
assert.ok(!sourceReads.some(path => path.includes('0.1.0')));
console.log('complete snapshots load selected versions without earlier history');

const instances = [];
class TestWorker {
  constructor(url, options) { this.url = url; this.options = options; this.listeners = new Map(); instances.push(this); }
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

const createObjectUrl = URL.createObjectURL;
const revokeObjectUrl = URL.revokeObjectURL;
const revokedWorkerUrls = [];
let workerUrls = 0;
globalThis.__CONFORMANCE_WORKER_SOURCE__ = 'self.onmessage = () => {};';
globalThis.Worker = TestWorker;
URL.createObjectURL = () => `blob:worker-${++workerUrls}`;
URL.revokeObjectURL = url => revokedWorkerUrls.push(url);
try {
  const embeddedParser = createWorkerParser();
  const read = embeddedParser.parseYaml(new Uint8Array([1]), {});
  const failure = assert.rejects(read, /worker import failed/);
  assert.equal(instances.at(-1).options.type, 'classic');
  instances.at(-1).fail();
  await failure;
  assert.deepEqual(revokedWorkerUrls, ['blob:worker-1']);
  const next = embeddedParser.parseYaml(new Uint8Array([2]), {});
  const cancellation = assert.rejects(next, /disposed/);
  embeddedParser.terminate();
  await cancellation;
  assert.deepEqual(revokedWorkerUrls, ['blob:worker-1', 'blob:worker-2']);
  globalThis.Worker = class { constructor() { throw new Error('worker construction failed'); } };
  await assert.rejects(createWorkerParser().parseYaml(new Uint8Array([3]), {}), /construction failed/);
  assert.deepEqual(revokedWorkerUrls, ['blob:worker-1', 'blob:worker-2', 'blob:worker-3']);
} finally {
  URL.createObjectURL = createObjectUrl;
  URL.revokeObjectURL = revokeObjectUrl;
  delete globalThis.__CONFORMANCE_WORKER_SOURCE__;
  delete globalThis.Worker;
}
console.log('embedded worker URLs release on failure, disposal and constructor rejection');

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

// The selected capture union controls authored inputs and version payloads together.
const unionCatalog = { sources: { unified: ['A', 'B', 'C', 'unselected'].flatMap(family => [
  { family, kind: 'inputs-and-golden', path: `${family}/inputs` },
  ...['v1', 'v2', 'peer'].map(implementation => ({ family, implementation, version: '1.0.0', format: 'typed-json-v1', kind: 'resolved-capture', path: `${family}/${implementation}` })),
]) } };
const unionDescriptors = [
  { source: { implementation: 'v1', version: '1.0.0', capture_families: ['A', 'B'] } },
  { source: { implementation: 'v2', version: '1.0.0', capture_families: ['B', 'C'] } },
];
const unionResources = new FixtureResources(unionCatalog, {}, {});
assert.deepEqual(unionResources.unifiedDependencies(unionDescriptors).map(item => item.path),
  ['A/inputs', 'A/v1', 'B/inputs', 'B/v1', 'B/v2', 'C/inputs', 'C/v2']);
assert.deepEqual(unionResources.unifiedDependencies([unionDescriptors[1]]).map(item => item.path),
  ['B/inputs', 'B/v2', 'C/inputs', 'C/v2']);
assert.deepEqual(unionResources.unifiedDependencies([]), []);
console.log('selected family union loads only available selected capture shards');

const bundledInputs = ['A', 'B', 'C'].map(family => ({ family, kind: 'inputs-and-golden', path: `${family}/inputs` }));
const captureBundle = { kind: 'resolved-capture', families: ['A', 'B'], implementation: 'dynamo_v1', version: '2.0.0', path: 'bundles/dynamo_v1-2.0.0' };
const bundledReader = new FixtureResources({ sources: { unified: [...bundledInputs, captureBundle] } }, {
  demand() {}, read: async () => new Uint8Array(),
}, {});
const bundledSelection = [{ source: { implementation: 'dynamo_v1', version: '2.0.0', capture_families: ['A', 'B'] } }];
assert.deepEqual(bundledReader.unifiedDependencies(bundledSelection).map(item => item.path),
  ['A/inputs', 'B/inputs', 'bundles/dynamo_v1-2.0.0']);
await bundledReader.prepareUnified(bundledSelection);
assert.equal(bundledReader.inputResources.get('A'), bundledInputs[0]);
assert.equal(bundledReader.inputResources.get('B'), bundledInputs[1]);
assert.deepEqual(bundledReader.unifiedDependencies([{ source: { implementation: 'dynamo_v1', version: '2.0.0', capture_families: ['A'] } }]).map(item => item.path),
  ['A/inputs', 'bundles/dynamo_v1-2.0.0']);
console.log('version bundles remain demand-scoped to selected parser families');

const historicalSource = { implementation: 'dynamo_v2', version: '0.1.0', mode: 'unified', native_unified_families: [], capture_families: ['sample'] };
const historicalResource = { ...latestSource, version: '0.1.0', path: 'sample/dynamo_v2-0.1.0.yaml' };
const historicalResources = new FixtureResources({ sources: {
  candidates: [selectionDescriptor.source, historicalSource],
  unified: [inputSource, latestSource, historicalResource],
} }, {}, {});
assert.deepEqual(historicalResources.unifiedDependencies([{ source: historicalSource }]).map(item => item.path),
  [inputSource.path, historicalResource.path], 'historical observations must be reserved and fetched even without native-route proof');
assert.deepEqual(historicalSource.native_unified_families, [], 'loading cannot change historical native eligibility');
