// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { test } from 'node:test';

await test('resources contracts', async () => {
const savedGlobals = Object.getOwnPropertyDescriptors(globalThis);
try {
const { default: assert } = await import('node:assert/strict');
const { createWorkerParser, HttpResourceStore, sha256 } = await import('./data.js');
const { FixtureResources } = await import('./resources.js');


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

const bootstrapCatalog = { ...unionCatalog, capture_generation: { current: [
  { implementation: 'v1', version: '0.9.0' }, { implementation: 'v1', version: '1.0.0' },
] },
  sources: { unified: unionCatalog.sources.unified.map(item => ({ ...item, path: `embedded/${item.path}` })) } };
const admitted = [];
const dualReader = new FixtureResources(unionCatalog, {
  demand(dependencies) { admitted.push(dependencies); }, read: async () => new Uint8Array(),
}, {}, bootstrapCatalog);
assert.equal(dualReader.isEmbedded('v1', '0.9.0'), false, 'older family-local snapshots cannot shadow complete historical bundles');
assert.deepEqual(dualReader.unifiedDependencies([unionDescriptors[0]]).map(item => item.path),
  ['embedded/A/inputs', 'embedded/A/v1', 'embedded/B/inputs', 'embedded/B/v1']);
await dualReader.prepareUnified(unionDescriptors);
assert.equal(admitted.length, 1, 'mixed captures must reserve both catalog owners together');
assert.deepEqual(admitted[0].map(item => item.path),
  ['embedded/A/inputs', 'embedded/A/v1', 'embedded/B/inputs', 'embedded/B/v1', 'B/inputs', 'B/v2', 'C/inputs', 'C/v2']);
assert.equal(dualReader.inputResources.get('B').path, 'B/inputs', 'mixed display uses historical authored union');
dualReader.embedded.unified = async (...args) => args;
assert.deepEqual(await dualReader.unified('A', 'v1', '1.0.0', { wrong: 'history fingerprint' }), ['A', 'v1', '1.0.0'],
  'embedded records must validate against their own authored input');
console.log('history discovery retains embedded latest dependencies and input validation');

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

} finally {
  for (const key of Object.getOwnPropertyNames(globalThis)) {
    if (!Object.hasOwn(savedGlobals, key) && Object.getOwnPropertyDescriptor(globalThis, key).configurable) delete globalThis[key];
  }
  for (const [key, descriptor] of Object.entries(savedGlobals)) {
    if (descriptor.configurable) Object.defineProperty(globalThis, key, descriptor);
  }
}

});

await test('unified contracts', async () => {
const savedGlobals = Object.getOwnPropertyDescriptors(globalThis);
try {
const { default: assert } = await import('node:assert/strict');
const { validateUnifiedInputs, validateUnifiedCapture, validateResolvedUnified, materializeUnifiedRecord, unifiedRecordContract, validateResolvedRecordContracts } = await import('./resources.js');
const { canonicalJson } = await import('./values.js');
const { sha256 } = await import('./data.js');
const { resolveUnified, resolveUnifiedSeries, selectedUnifiedSnapshot, compareVersions, provenanceReleaseVersion, unifiedCaptureVersionPattern } = await import('./model.js');

const qualifiedProducer = '0.7.25+source.' + 'a'.repeat(64);
assert.equal(provenanceReleaseVersion(qualifiedProducer), '0.7.25');
assert.equal(unifiedCaptureVersionPattern.test(qualifiedProducer), false);
for (const invalid of ['0.7.25+source.a', '0.7.25+other.abc', '0.7.25+source./bad']) assert.throws(() => provenanceReleaseVersion(invalid));


assert.equal(canonicalJson({ z: 9007199254740993n, a: [1, 1n, 1e-7, 1e16, -0] }), '{"a":[1.0,1,1e-07,1e+16,-0.0],"z":9007199254740993}');
const request = { input: 'sample', init: {}, finish_reason: 'stop', tools: [], chunks: [] };
const inputs = { input_document: { family: 'sample' }, golden_document: { family: 'sample' }, cases: { id: { lifecycle: 'active', scenario: 'sample', description: 'sample case', policy: [], display_id: 'UNIFIED.1', historical_ids: [], request, golden: { assembled: [] } } } };
validateUnifiedInputs(inputs, 'sample', 'inputs.yaml');

const batchFinal = { reasoning_text: 'think', normal_text: 'answer', calls: [{ name: 'lookup', arguments: '{"q":1}' }] };
const batchEvents = [
  { kind: 'reasoning', text: 'think' },
  { kind: 'text', text: 'answer' },
  { kind: 'tool_call', name: 'lookup', arguments: '{"q":1}' },
];
const batchDocument = { capture_origin: { adapter_sha256: 'a'.repeat(64), job_sha256: 'b'.repeat(64) }, record_metadata: { batch_final: batchFinal } };
const batchRecord = { value: { document: batchDocument, observation: { value: { assembled: batchEvents } } } };
const legacyStreamRecord = { value: { document: {}, observation: { value: { chunks: [[{ kind: 'text', text: 'legacy' }]] } } } };
assert.equal(unifiedRecordContract('dynamo_v1', batchDocument, 'batch'), 'batch');
assert.equal(unifiedRecordContract('dynamo_v1', {}, 'batch'), 'legacy-stream');
assert.equal(unifiedRecordContract('dynamo_v2', {}, 'stream'), 'stream');
assert.equal(validateResolvedRecordContracts(new Map([['batch', batchRecord], ['legacy', legacyStreamRecord]]), 'dynamo_v1', 'batch').size, 2);
assert.throws(() => unifiedRecordContract('dynamo_v1', { capture_origin: { adapter_sha256: 'a'.repeat(64) } }, 'batch'), /adapter and job/);
assert.throws(() => validateResolvedRecordContracts(new Map([['batch', { ...batchRecord, value: { ...batchRecord.value, observation: { value: { assembled: [] } } } }]]), 'dynamo_v1', 'batch'), /does not match its final fields/);
const noGolden = structuredClone(inputs);
delete noGolden.cases.id.golden;
assert.throws(() => validateUnifiedInputs(noGolden, 'sample', 'inputs.yaml'), /missing golden/);
const emptyOracle = structuredClone(inputs);
emptyOracle.cases.id.golden = {};
assert.throws(() => validateUnifiedInputs(emptyOracle, 'sample', 'inputs.yaml'), /assembled event list/);
const badRequest = structuredClone(inputs);
delete badRequest.cases.id.request.init;
assert.throws(() => validateUnifiedInputs(badRequest, 'sample', 'inputs.yaml'), /missing init/);
const unknownField = { ...inputs, unexpected: {} };
assert.throws(() => validateUnifiedInputs(unknownField, 'sample', 'inputs.yaml'), /unknown field/);
const digest = await sha256(new TextEncoder().encode(canonicalJson(request)));
const capture = { provenance: { status: 'captured', origin: { crate_version: '0.1.0', source_sha256: 'a'.repeat(64) } }, document: {}, changes: { id: { case_key: 'UNIFIED.1', stimulus: { ref: 'current', semantic_sha256: digest }, observation: { value: { chunks: [] } } } }, metadata_changes: {}, document_overrides: {} };
const identity = { path: 'sample-0.1.0.yaml', implementation: 'sample', version: '0.1.0' };
await validateUnifiedCapture(capture, inputs, identity);
const snapshotMember = 'conformance/fixtures-unified-v2/families/sample/dynamo_v2-0.1.0.yaml';
const snapshotRecord = { value: {
  case_key: 'UNIFIED.1', stimulus: { ref: 'current' }, observation: { value: { assembled: [] } },
  document: { captured_with: { dynamo_v2: '0.1.0' } },
}, member: snapshotMember };
const snapshot = {
  schema_version: 2,
  provenance: { status: 'legacy', captured_with: { dynamo_v2: '0.1.0' } },
  document: {}, records: { id: snapshotRecord },
};
const snapshotResource = { path: snapshotMember, family: 'sample', implementation: 'dynamo_v2', version: '0.1.0', record_contract: 'stream' };
await validateUnifiedCapture(snapshot, inputs, snapshotResource);
const qualifiedRecord = structuredClone(snapshotRecord);
qualifiedRecord.value.document = {
  captured_with: { dynamo_v2: '0.1.0+source.' + 'a'.repeat(64) },
  capture_origin: { crate_version: '0.1.0', source_sha256: 'a'.repeat(64) },
};
const qualifiedSources = [{ ...snapshotResource, kind: 'history' }];
await validateResolvedUnified({ id: qualifiedRecord }, inputs, snapshotResource, qualifiedSources);
for (const badLabel of ['0.1.0+source.' + 'b'.repeat(64), '0.0.9+source.' + 'a'.repeat(64)]) {
  const invalid = structuredClone(qualifiedRecord);
  invalid.value.document.captured_with.dynamo_v2 = badLabel;
  await assert.rejects(validateResolvedUnified({ id: invalid }, inputs, snapshotResource, qualifiedSources));
}
assert.equal(qualifiedRecord.value.document.captured_with.dynamo_v2, '0.1.0+source.' + 'a'.repeat(64));
const frozenProofSnapshot = structuredClone(snapshot);
const frozenOrigin = { crate_version: '0.1.0', source_sha256: 'a'.repeat(64), adapter_sha256: 'b'.repeat(64), job_sha256: 'c'.repeat(64) };
frozenProofSnapshot.provenance = { status: 'captured', origin: frozenOrigin };
frozenProofSnapshot.records.id.value.document.capture_origin = structuredClone(frozenOrigin);
frozenProofSnapshot.records.id.value.stimulus.semantic_sha256 = digest;
frozenProofSnapshot.execution_proof = { receipt: { ...frozenOrigin, implementation: 'dynamo_v2', version: '0.1.0', results_sha256: 'd'.repeat(64) }, requests: { id: digest } };
const grownInputs = structuredClone(inputs);
grownInputs.cases.later = structuredClone(inputs.cases.id);
await validateUnifiedCapture(frozenProofSnapshot, grownInputs, snapshotResource);
const deduplicatedProofSnapshot = structuredClone(frozenProofSnapshot);
deduplicatedProofSnapshot.records.id.value.document.capture_origin.source_sha256 = 'f'.repeat(64);
await validateUnifiedCapture(deduplicatedProofSnapshot, grownInputs, snapshotResource);
for (const mutation of ['missing', 'changed', 'extra', 'partialBad', 'partialHex', 'emptyExecution']) {
  const broken = structuredClone(frozenProofSnapshot);
  if (mutation === 'missing') delete broken.execution_proof.requests.id;
  if (mutation === 'changed') broken.execution_proof.requests.id = 'e'.repeat(64);
  if (mutation === 'extra') broken.records.later = structuredClone(broken.records.id);
  if (mutation === 'emptyExecution') { broken.records = {}; broken.execution_proof.requests = {}; }
  if (mutation === 'partialBad' || mutation === 'partialHex') {
    const badDigest = mutation === 'partialBad' ? 'bad' : 'e'.repeat(64);
    broken.records.id.value.stimulus = { partial: { input: 'x' }, semantic_sha256: badDigest };
    broken.execution_proof.requests.id = badDigest;
  }
  await assert.rejects(validateUnifiedCapture(broken, grownInputs, snapshotResource), /request membership mismatch|sealed request digests|complete request/);
}

const historicalEvidence = {
  kind: 'execution_manifest', schema_version: 1, family: 'sample', implementation: 'dynamo_v2', version: '0.1.0',
  checkpoint: {}, cases: ['id'], defaults: {}, exceptions: {}, metadata_changes: { id: {} }, receipt_membership: [],
};
for (const field of ['checkpoint', 'defaults', 'exceptions']) {
  const invalidEvidence = structuredClone(historicalEvidence);
  invalidEvidence[field] = [];
  await assert.rejects(validateUnifiedCapture({ ...snapshot, historical_evidence: [invalidEvidence] }, inputs, snapshotResource), new RegExp(`${field} must be a mapping`));
}
await validateResolvedUnified({ id: snapshotRecord }, inputs, snapshotResource, [
  { path: snapshotMember, kind: 'history', family: 'sample', implementation: 'dynamo_v2', version: '0.1.0' },
]);
await validateResolvedUnified({ id: { ...snapshotRecord, value: { ...snapshotRecord.value,
  document: { captured_with: { dynamo_v2: '0.1.1' } },
} } }, inputs, { ...snapshotResource, version: '0.1.1' }, [
  { path: snapshotMember, kind: 'history', family: 'sample', implementation: 'dynamo_v2', version: '0.1.0' },
]);
await assert.rejects(validateResolvedUnified({ id: { ...snapshotRecord, value: { ...snapshotRecord.value, stimulus: { ref: 'wrong' } } } }, inputs, snapshotResource, [
  { path: snapshotMember, kind: 'history', family: 'sample', implementation: 'dynamo_v2', version: '0.1.0' },
]), /invalid current stimulus reference/);
await assert.rejects(validateResolvedUnified({ id: { ...snapshotRecord, value: { ...snapshotRecord.value, document: { captured_with: { dynamo_v2: '0.1.1' } } } } }, inputs, snapshotResource, [
  { path: snapshotMember, kind: 'history', family: 'sample', implementation: 'dynamo_v2', version: '0.1.0' },
]), /false producer identity/);
await assert.rejects(validateUnifiedCapture({ ...snapshot, records: { id: { ...snapshotRecord, member: '../outside.yaml' } } }, inputs, snapshotResource), /invalid source member/);
const postVersions = ['0.5.4', '0.5.4.post1', '0.5.4.post2', '0.5.5', '0.9.0.1', '0.10.1.1'];
assert.deepEqual([...postVersions].reverse().sort(compareVersions), postVersions);
for (const version of postVersions) {
  assert.equal(unifiedCaptureVersionPattern.test(version), true);
  await validateUnifiedCapture({ ...capture, provenance: { status: 'captured', origin: { ...capture.provenance.origin, crate_version: version } } }, inputs, { ...identity, version });
}
for (const [implementation, versions] of [
  ['vllm_python', ['0.8.5', '0.8.5.post1']],
  ['sglang_python', ['0.4.4', '0.4.4.post2', '0.4.4.post3']],
]) {
  const history = [];
  for (const [index, version] of versions.entries()) {
    const document = { ...capture, provenance: { status: 'legacy', captured_with: { [implementation]: version } },
      changes: { id: { ...capture.changes.id, observation: { value: { assembled: [{ kind: 'text', text: version }] } } } },
      ...(index ? { parent: `${implementation}-${versions[index - 1]}` } : {}) };
    const resource = { path: `${implementation}-${version}.yaml`, implementation, version };
    await validateUnifiedCapture(document, inputs, resource);
    history.push({ ...resource, document });
  }
  for (const version of versions) {
    const record = resolveUnified([...history].reverse(), version).get('id').value;
    assert.equal(record.observation.value.assembled[0].text, version);
    assert.equal(record.document.captured_with[implementation], version);
  }
}
for (const version of ['0.5.4.post', '0.5.4.post2junk', '0.5.4.patch1', '0.5.4+source.abcd', '1.2']) assert.equal(unifiedCaptureVersionPattern.test(version), false);
for (const version of ['0.5.4.post', '0.5.4.post2junk', '0.5.4+source.abcd']) assert.throws(() => compareVersions(version, '0.5.4'), /invalid capture version/);
const executionManifest = {
  schema_version: 1n, family: 'sample', implementation: 'sample', version: '0.1.0',
  checkpoint: { provenance: capture.provenance, document: {}, document_overrides: {} },
  cases: ['id'], defaults: { receipt: { source_sha256: 'a'.repeat(64), nullable: null } }, exceptions: { id: { note: 'retained' } },
};
const executionBytes = new TextEncoder().encode('verified execution manifest bytes');
const execution = { value: executionManifest, sourceBytes: executionBytes };
const compactCapture = { execution: "executions/sample--sample-0.1.0.yaml", changes: capture.changes };
const expandedCapture = await validateUnifiedCapture(compactCapture, inputs, { ...identity, family: 'sample' }, execution);
assert.deepEqual(expandedCapture.metadata_changes, { id: { receipt: { source_sha256: 'a'.repeat(64), nullable: null }, note: 'retained' } });
const compactResolved = resolveUnified([{ ...identity, document: expandedCapture }], '0.1.0');
assert.equal(compactResolved.get('id').value.document.record_metadata.receipt.nullable, null);
for (const invalid of [
  { ...execution, value: { ...executionManifest, version: '0.2.0' } },
  { ...execution, value: { ...executionManifest, cases: [] } },
  { ...execution, value: { ...executionManifest, cases: ['id', 'id'] } },
  { ...execution, value: { ...executionManifest, exceptions: { other: {} } } },
]) await assert.rejects(validateUnifiedCapture(compactCapture, inputs, { ...identity, family: 'sample' }, invalid));
assert.throws(() => resolveUnified([{ ...identity, document: { ...expandedCapture, parent: 'sample-0.0.1' } }], '0.1.0'), /preceding checkpoint/);
const receiptBytes = new TextEncoder().encode('global execution receipt bytes');
const receiptHash = 'receipts/sample--sample-0.1.0--receipt-001.yaml';
const receiptValue = { version: '0.1.0', job_sha256: 'a'.repeat(64), adapter_sha256: 'b'.repeat(64) };
const pooledManifest = { ...executionManifest, defaults: { sample_capture_provenance: receiptHash }, exceptions: {}, receipts: { [receiptHash]: ['id'] } };
const pooled = { ...execution, value: pooledManifest, receipts: new Map([[receiptHash, { value: receiptValue, sourceBytes: receiptBytes }]]) };
const pooledExpanded = await validateUnifiedCapture(compactCapture, inputs, { ...identity, family: 'sample' }, pooled);
assert.deepEqual(pooledExpanded.metadata_changes.id.sample_capture_provenance, receiptValue);
await assert.rejects(validateUnifiedCapture(compactCapture, inputs, { ...identity, family: 'sample' }, { ...pooled, receipts: new Map() }), /receipt is missing/);
await assert.rejects(validateUnifiedCapture(compactCapture, inputs, { ...identity, family: 'sample' }, { ...pooled, value: { ...pooledManifest, receipts: { [receiptHash]: [] } } }), /membership mismatch/);
await assert.rejects(validateUnifiedCapture(compactCapture, inputs, { ...identity, family: 'sample' }, { ...pooled, value: { ...pooledManifest, receipts: {} } }), /body or membership missing/);
for (const mutate of [
  (value) => { value.changes.id.stimulus.semantic_sha256 = '0'.repeat(64); },
  (value) => { value.changes.id.stimulus.semantic_sha256 = ''; },
  (value) => { value.changes.id.stimulus.inline = request; },
  (value) => { value.changes.id.observation.error = { code: 'error' }; },
  (value) => { value.changes.id.observation.value = {}; },
  (value) => { value.changes.other = value.changes.id; },
  (value) => { value.document.captured_with = {}; },
  (value) => { value.provenance.origin.crate_version = '0.2.0'; },
  (value) => { value.document_overrides.id = { family: 'other' }; },
  (value) => { value.changes = new Map([[1n, { absent: true }]]); },
  (value) => { value.metadata_changes = new Map([[1n, {}]]); },
  (value) => { value.document_overrides = new Map([[1n, {}]]); },
  (value) => { value.changes.constructor = { absent: true }; },
  (value) => { value.metadata_changes.toString = {}; },
  (value) => { value.document_overrides.constructor = {}; },
]) {
  const invalid = structuredClone(capture);
  mutate(invalid);
  await assert.rejects(validateUnifiedCapture(invalid, inputs, identity));
}
const timestampRequest = structuredClone(inputs);
timestampRequest.cases.id.request.init = new Date();
assert.throws(() => validateUnifiedInputs(timestampRequest, 'sample', 'inputs.yaml'), /mapping/);
const duplicateAlias = structuredClone(inputs);
duplicateAlias.cases.id.display_id = 'UNIFIED.1-tool_only';
duplicateAlias.cases.other = { ...duplicateAlias.cases.id, display_id: 'UNIFIED.1.a' };
const aliases = { owners: { '1-tool_only': ['tool_only'], '1.a': ['tool_only'] }, canonical: { tool_only: '1-tool_only' }, families: {} };
assert.throws(() => validateUnifiedInputs(duplicateAlias, 'sample', 'inputs.yaml', aliases), /duplicate canonical ID/);
const legacy = structuredClone(capture);
legacy.provenance = { status: 'legacy', captured_with: { sample: '0.1.0' } };
await validateUnifiedCapture(legacy, inputs, identity);
const typedMetadata = structuredClone(capture);
typedMetadata.document = new Map([[1n, 'document metadata']]);
typedMetadata.metadata_changes.id = new Map([[2n, 'record metadata']]);
await validateUnifiedCapture(typedMetadata, inputs, identity);
const typedResolved = resolveUnified([{ ...identity, bytes: new Uint8Array(), document: typedMetadata }], '0.1.0').get('id');
assert.equal(typedResolved.value.document.get(1n), 'document metadata');
const materialized = materializeUnifiedRecord(inputs.cases.id, typedResolved, 'sample');
assert.equal(materialized.get(1n), 'document metadata');
assert.equal(materialized.get('cases')['UNIFIED.1'].get(2n), 'record metadata');
const metadata = { ...identity, bytes: new Uint8Array(), document: capture };
assert.throws(() => resolveUnified([{ ...metadata, document: { ...capture, changes: {}, metadata_changes: { id: {} } } }], '0.1.0'), /no observation/);
assert.throws(() => resolveUnified([{ ...metadata, document: { ...capture, changes: {}, document_overrides: { id: {} } } }], '0.1.0'), /no observation/);
const first = { ...metadata, document: { ...capture, metadata_changes: { id: { reason: 'old' } }, document_overrides: { id: { parser_path: 'path' } } } };
const second = { ...metadata, version: '0.2.0', document: { ...capture, changes: {}, metadata_changes: { id: {} }, document_overrides: {} } };
const resolved = resolveUnified([first, second], '0.2.0').get('id').value.document;
assert.equal(Object.hasOwn(resolved, 'record_metadata'), false);
assert.equal(resolved.parser_path, 'path');
assert.equal(resolved.inherited_from, '0.1.0');
const tombstone = { ...second, document: { ...second.document, changes: { id: { absent: true } }, metadata_changes: {} } };
assert.equal(resolveUnified([first, tombstone], '0.2.0').size, 0);
const originCheckpoint = {
  ...first,
  bytes: new Uint8Array([1, 2, 3]),
  document: { ...first.document, document: { nested: { value: 'original' } } },
};
const overlayCheckpoint = {
  ...second,
  document: { ...second.document, changes: { id: { stimulus: { ref: 'updated' } } } },
};
const frozenHistory = structuredClone([originCheckpoint, overlayCheckpoint]);
const overlayRecord = resolveUnified([originCheckpoint, overlayCheckpoint], '0.2.0').get('id');
assert.equal(overlayRecord.sourceBytes, originCheckpoint.bytes, 'sparse overlays must share original YAML bytes rather than copy the checkpoint');
assert.equal(overlayRecord.member, originCheckpoint.path);
assert.equal(overlayRecord.value.document.inherited_from, '0.1.0');
overlayRecord.value.observation.value.chunks.push([]);
overlayRecord.value.stimulus.ref = 'mutated';
overlayRecord.value.document.nested.value = 'mutated';
assert.deepEqual([originCheckpoint, overlayCheckpoint], frozenHistory, 'mutable resolved records must not change checkpoint data');
assert.deepEqual(resolveUnified([originCheckpoint, overlayCheckpoint], '0.2.0').get('id').value.observation, capture.changes.id.observation);
const replaced = resolveUnified([originCheckpoint, { ...overlayCheckpoint, document: { ...overlayCheckpoint.document, changes: { id: { observation: { value: { chunks: [] } } } } } }], '0.2.0').get('id');
assert.equal(replaced.sourceBytes, overlayCheckpoint.bytes, 'a new observation must select its own checkpoint bytes');
const receiptCheckpoint = { ...first, document: { ...first.document, metadata_changes: { id: { verified_recapture: { receipt: 'first' }, retained: true } } } };
assert.equal(resolveUnified([receiptCheckpoint], '0.1.0').get('id').value.document.record_metadata.verified_recapture.receipt, 'first');
for (const history of [[receiptCheckpoint], [receiptCheckpoint, { ...second, document: { ...second.document, metadata_changes: {} } }]]) {
  const later = resolveUnified(history, '0.2.0').get('id').value.document.record_metadata;
  assert.equal(Object.hasOwn(later, 'verified_recapture'), false);
  assert.equal(later.retained, true);
}
assert.equal(receiptCheckpoint.document.metadata_changes.id.verified_recapture.receipt, 'first');
const removedPath = resolveUnified([first, { ...second, document: { ...second.document, document_overrides: { id: { parser_path: null } } } }], '0.2.0');
assert.equal(Object.hasOwn(removedPath.get('id').value.document, 'parser_path'), false);

const seriesVersions = ['0.1.0', '0.2.0', '0.2.0.post2', '0.10.0', '0.11.0'];
const seriesHistory = [
  receiptCheckpoint,
  { ...overlayCheckpoint, document: { ...overlayCheckpoint.document, metadata_changes: {}, document_overrides: { id: { parser_path: null } } } },
  { ...overlayCheckpoint, document: { ...overlayCheckpoint.document, changes: { id: { observation: { value: { chunks: [['replacement']] } } } }, metadata_changes: {}, document_overrides: {} } },
  tombstone,
  receiptCheckpoint,
].map((entry, index) => ({ ...entry, version: seriesVersions[index], path: `history/${seriesVersions[index]}.yaml`, bytes: new Uint8Array([index]),
  document: { ...entry.document, ...(index ? { parent: `${entry.implementation}-${seriesVersions[index - 1]}` } : {}) } }));
const requestedVersions = ['0.0.0', '0.1.0', '0.1.0.post1', '0.2.0', '0.2.0.post1', '0.2.0.post2', '0.2.0.post3', '0.3.0', '0.10.0', '0.11.0', '0.12.0'];
const series = new Map(resolveUnifiedSeries([...seriesHistory].reverse(), [...requestedVersions].reverse()));
assert.deepEqual([...series.keys()], requestedVersions, 'series uses numeric and post-release ordering');
for (const version of requestedVersions) assert.deepEqual(series.get(version), resolveUnified(seriesHistory, version));
assert.equal(series.get('0.0.0').size, 0);
assert.equal(series.get('0.1.0').get('id').value.document.record_metadata.verified_recapture.receipt, 'first');
assert.equal(Object.hasOwn(series.get('0.1.0.post1').get('id').value.document.record_metadata, 'verified_recapture'), false);
assert.equal(Object.hasOwn(series.get('0.2.0').get('id').value.document, 'parser_path'), false);
assert.equal(series.get('0.2.0.post1').get('id').member, 'history/0.1.0.yaml');
assert.equal(series.get('0.2.0.post2').get('id').sourceBytes, seriesHistory[2].bytes);
assert.equal(series.get('0.10.0').size, 0);
assert.equal(series.get('0.11.0').get('id').member, 'history/0.11.0.yaml');
const mutableSeries = resolveUnifiedSeries(seriesHistory, ['0.1.0', '0.1.0.post1', '0.2.0']);
const mutableFirst = mutableSeries.next().value[1];
mutableFirst.get('id').value.stimulus.ref = 'mutated yield';
mutableFirst.get('id').value.observation.value.chunks.push(['mutated yield']);
mutableFirst.get('id').value.document.record_metadata.retained = false;
mutableFirst.clear();
assert.deepEqual(mutableSeries.next().value[1], series.get('0.1.0.post1'), 'mutating an earlier yield cannot alter a later record');
assert.deepEqual(mutableSeries.next().value[1], series.get('0.2.0'));
const typedSeries = resolveUnifiedSeries([{ ...identity, bytes: new Uint8Array(), document: typedMetadata }], ['0.1.0', '0.2.0']);
const typedFirst = typedSeries.next().value[1].get('id').value;
typedFirst.document.set(1n, 'mutated document map');
typedFirst.document.get('record_metadata').set(2n, 'mutated record map');
const typedLater = typedSeries.next().value[1].get('id').value;
assert.equal(typedLater.document.get(1n), 'document metadata');
assert.equal(typedLater.document.get('record_metadata').get(2n), 'record metadata');
let checkpointReads = 0;
const measuredHistory = seriesHistory.map(entry => ({ ...entry, document: { ...entry.document,
  get changes() { checkpointReads += 1; return entry.document.changes; },
} }));
[...resolveUnifiedSeries(measuredHistory, requestedVersions)];
assert.equal(checkpointReads, seriesHistory.length, 'series visits each checkpoint once across all requested versions');
const invalidParentHistory = [...seriesHistory.slice(0, 2), { ...seriesHistory[2], document: { ...seriesHistory[2].document, parent: 'sample-9.9.9' } }];
assert.throws(() => [...resolveUnifiedSeries(invalidParentHistory, ['0.1.0', '0.2.0.post2'])], /preceding checkpoint/);
assert.throws(() => [...resolveUnifiedSeries([{ ...metadata, document: { ...capture, changes: {}, metadata_changes: { id: {} } } }], ['0.1.0'])], /no observation/);
console.log('Unified shape, stimulus, provenance, metadata, and tombstone contracts passed');

// A fresh execution is selected without rewriting the historical source record.
const freshResult = { assembled: [{ kind: 'text', text: 'fresh' }], chunks: [{ expected: [{ kind: 'text', text: 'fresh' }] }], capture_input: request };
const freshProvenance = { implementation: 'dynamo_v2', version: '0.7.24', source_sha256: 'a'.repeat(64), adapter_sha256: 'b'.repeat(64), job_sha256: 'c'.repeat(64) };
const recapturedRecord = { value: {
  case_key: 'UNIFIED.1', stimulus: { inline: request }, observation: { value: { assembled: [{ kind: 'text', text: 'original' }] } },
  document: { captured_with: { dynamo_v2: '0.7.24' }, record_metadata: {
    verified_recapture: { result: freshResult, provenance: freshProvenance, resolution: 'fresh_execution' },
  } },
} };
const selectedRecapture = materializeUnifiedRecord(inputs.cases.id, recapturedRecord, 'sample', { implementation: 'dynamo_v2', version: '0.7.24', captureMode: 'stream' });
assert.deepEqual(selectedRecapture.cases['UNIFIED.1'].assembled, freshResult.assembled);
assert.equal(recapturedRecord.value.observation.value.assembled[0].text, 'original');
for (const implementation of ['sglang_python', 'vllm_python', 'vllm_rust']) {
  const peerRecord = structuredClone(recapturedRecord);
  peerRecord.value.document.record_metadata.verified_recapture.provenance.implementation = implementation;
  const selected = materializeUnifiedRecord(inputs.cases.id, peerRecord, 'sample', { implementation, version: '0.7.24', captureMode: 'stream' });
  assert.deepEqual(selected.cases['UNIFIED.1'].assembled, freshResult.assembled);
  assert.equal(peerRecord.value.observation.value.assembled[0].text, 'original');
  const unavailableRecord = structuredClone(peerRecord);
  unavailableRecord.value.document.record_metadata.verified_recapture.result = { capture_input: request, unavailable: 'No published parser API' };
  const unavailable = materializeUnifiedRecord(inputs.cases.id, unavailableRecord, 'sample', { implementation, version: '0.7.24', captureMode: 'stream' });
  assert.match(unavailable.cases['UNIFIED.1'].historical_capture_note, /remains unavailable/);
  delete peerRecord.value.document.record_metadata.verified_recapture.provenance.implementation;
  assert.throws(() => materializeUnifiedRecord(inputs.cases.id, peerRecord, 'sample', { implementation, version: '0.7.24', captureMode: 'stream' }), /implementation differs/);
}
for (const [implementation, version] of [['dynamo_v2', '0.7.23'], ['vllm_rust', '0.7.24']]) {
  assert.throws(() => materializeUnifiedRecord(inputs.cases.id, recapturedRecord, 'sample', { implementation, version, captureMode: 'stream' }), /differs from candidate/);
}
const alteredInput = structuredClone(inputs.cases.id);
alteredInput.request.input = 'altered';
assert.throws(() => materializeUnifiedRecord(alteredInput, recapturedRecord, 'sample', { implementation: 'dynamo_v2', version: '0.7.24', captureMode: 'stream' }), /request differs/);
const staleReceipt = structuredClone(recapturedRecord);
staleReceipt.value.document.record_metadata.verified_recapture.provenance.adapter_sha256 = 'stale';
assert.throws(() => materializeUnifiedRecord(inputs.cases.id, staleReceipt, 'sample', { implementation: 'dynamo_v2', version: '0.7.24', captureMode: 'stream' }), /invalid adapter_sha256/);

const completeSelection = new Map([
  ['plain', { value: { document: { record_metadata: { verified_recapture: { version: '1.0.0' }, kept: true } } } }],
  ['noMetadata', { value: { document: {} } }],
  ['emptyMetadata', { value: { document: { record_metadata: {} } } }],
  ['onlyVerified', { value: { document: { record_metadata: { verified_recapture: { version: '1.0.0' } } } } }],
  ['onlySibling', { value: { document: { record_metadata: { kept: true } } } }],
  ['suppressed', { value: { document: {} }, suppress_inherited_from: true }],
  ['override', { value: { document: { inherited_from: 'explicit' } } }],
  ['nullOverride', { value: { document: { inherited_from: null } } }],
]);
const derivedSelection = selectedUnifiedSnapshot(completeSelection, '1.0.0', '1.1.0');
for (const name of ['plain', 'noMetadata']) assert.equal(derivedSelection.get(name).value.document.inherited_from, '1.0.0');
assert.deepEqual(derivedSelection.get('plain').value.document.record_metadata, { kept: true });
assert.deepEqual(derivedSelection.get('onlySibling').value.document.record_metadata, { kept: true });
for (const name of ['noMetadata', 'emptyMetadata', 'onlyVerified']) assert.equal(Object.hasOwn(derivedSelection.get(name).value.document, 'record_metadata'), false);
assert.equal(derivedSelection.get('override').value.document.inherited_from, 'explicit');
assert.equal(derivedSelection.get('nullOverride').value.document.inherited_from, null);
assert.equal(Object.hasOwn(completeSelection.get('plain').value.document, 'inherited_from'), false);
assert.deepEqual(selectedUnifiedSnapshot(completeSelection, '1.0.0', '1.0.0'), completeSelection);
assert.equal(resolveUnified([receiptCheckpoint], '0.2.0').get('id').value.document.inherited_from, '0.1.0');

assert.equal(Object.hasOwn(derivedSelection.get('suppressed').value.document, 'inherited_from'), false);

{
  const { assembleStream } = await import('./model.js');
  const phases = [
    { kind: 'message_start', phase: 'commentary' }, { kind: 'message_end' },
    { kind: 'message_start', phase: 'commentary' }, { kind: 'text', text: 'One.' }, { kind: 'message_end' },
    { kind: 'message_start', phase: 'commentary' }, { kind: 'text', text: 'Two.' }, { kind: 'message_end' },
    { kind: 'message_start', phase: 'final_answer' }, { kind: 'text', text: 'Done.' }, { kind: 'message_end' },
  ];
  assert.deepEqual(assembleStream(phases.map(event => [event])), phases);
  assert.deepEqual(assembleStream([phases]), phases);
  assert.deepEqual(assembleStream([phases.slice(0, -1)]), phases.slice(0, -1));
  assert.deepEqual(assembleStream([[{ kind: 'tool_call', name: 'f', arguments: '{', complete: false }]]), []);
  assert.deepEqual(assembleStream([[{ kind: 'tool_call', name: 'f', arguments: '{"x":', complete: false }], [{ kind: 'tool_call', arguments: '1}', complete: true }]]), [{ kind: 'tool_call', name: 'f', arguments: { x: 1n } }]);
  const malformed = [[{ kind: 'tool_call', name: 'f', arguments: '{', complete: true }]];
  assert.deepEqual(assembleStream(malformed, [{ kind: 'tool_call', name: 'f', arguments: {} }]), [{ kind: 'tool_call', name: 'f', arguments: {} }]);
  assert.equal(assembleStream(malformed)[0].arguments, '{');
  assert.equal(assembleStream(malformed, [{ kind: 'tool_call', name: 'different', arguments: {} }])[0].arguments, '{');
  assert.equal(assembleStream(malformed, [])[0].arguments, '{');
  assert.equal(assembleStream([[{ kind: 'tool_call', name: 'f', arguments: '"literal"', complete: true }]], [{ kind: 'tool_call', name: 'f', arguments: {} }])[0].arguments, 'literal');

}

{
const metadata = JSON.parse(await (await import('node:fs/promises')).readFile(new URL('../viewer-catalog.json', import.meta.url))).metadata;
const { unifiedResult, resolveUnified, compareVersions } = await import('./model.js');
const { canonicalUnifiedLabel } = await import('./resources.js');
  const first = { implementation: 'dynamo_v2', version: '0.6.0', path: 'first.yaml', bytes: new Uint8Array(), document: { provenance: { status: 'legacy', captured_with: { dynamo_v2: '0.6.0' } }, document: {}, changes: { id: { case_key: 'UNIFIED.3', stimulus: { unavailable: { code: 'original_request_not_retained' } }, observation: { value: { assembled: [{ kind: 'text', text: 'captured' }] } } } }, metadata_changes: {}, document_overrides: {} } };
  const carried = resolveUnified([first], '0.6.1').get('id');
  assert.equal(unifiedResult(carried).events[0].text, 'captured');
  assert.equal(carried.value.document.captured_with.dynamo_v2, '0.6.0');
  const error = { ...first, version: '0.6.1', document: { ...first.document, changes: { id: { case_key: 'UNIFIED.3', observation: { error: { code: 'capture_failure', detail: 'capture failed' } } } } } };
  const current = resolveUnified([first, error], '0.6.1').get('id');
  assert.equal(unifiedResult(current).state, 'error');
  assert.equal(unifiedResult(current).reason, 'capture failed');
  assert.equal(Object.hasOwn(current.value.document, 'inherited_from'), false);
  const before = structuredClone(first);
  resolveUnified([first], '0.6.1');
  assert.deepEqual(first, before);
  assert.throws(() => compareVersions('0.6.1+source.' + 'a'.repeat(64), '0.6.1'), /invalid capture version/);
  assert.equal(canonicalUnifiedLabel('UNIFIED.31.a', 'qwen3', metadata.unified_aliases), 'UNIFIED.31-guided_json_invalid_call');
  assert.equal(canonicalUnifiedLabel('UNIFIED.31-29', 'qwen3', metadata.unified_aliases), 'UNIFIED.31-29');

}

} finally {
  for (const key of Object.getOwnPropertyNames(globalThis)) {
    if (!Object.hasOwn(savedGlobals, key) && Object.getOwnPropertyDescriptor(globalThis, key).configurable) delete globalThis[key];
  }
  for (const [key, descriptor] of Object.entries(savedGlobals)) {
    if (descriptor.configurable) Object.defineProperty(globalThis, key, descriptor);
  }
}

});
