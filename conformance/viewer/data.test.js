// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { test } from 'node:test';

await test('transport contracts', async () => {
const savedGlobals = Object.getOwnPropertyDescriptors(globalThis);
try {
const { default: assert } = await import('node:assert/strict');
const { gzipSync } = await import('node:zlib');
const { HttpResourceStore, sha256 } = await import('./data.js');
const { boundedGunzip, ChunkTransport, MAX_PACK_SIZE } = await import('./data.js');


const first = new Uint8Array([0xef, 0xbb, 0xbf, 97, 13, 10]);
const second = new Uint8Array([0, 255, 42]);
const bytes = new Uint8Array([...first, ...second]);
const compressed = new Uint8Array(gzipSync(bytes));
const pack = { path: 'packs/example.gz', sha256: await sha256(compressed), size: compressed.length, decoded_sha256: await sha256(bytes), decoded_size: bytes.length };
const recipe = { schema: 2, sha256: await sha256(bytes), size: bytes.length, chunks: [
  { sha256: await sha256(first), size: first.length, offset: 0, pack },
  { sha256: await sha256(second), size: second.length, offset: first.length, pack },
] };
const resource = { path: 'source/original.yaml', sha256: recipe.sha256, size: recipe.size, transport: 'chunked-v2', recipe };
const other = { path: 'source/second.yaml', sha256: recipe.chunks[1].sha256, size: second.length, transport: 'chunked-v2', recipe: { schema: 2, sha256: recipe.chunks[1].sha256, size: second.length, chunks: [recipe.chunks[1]] } };
function storeFor(fetchImpl) {
  return new HttpResourceStore({ baseUrl: 'http://test/', fetchImpl });
}
let requests = 0;
const store = storeFor(async url => {
  requests++;
  assert.equal(url.pathname, '/packs/example.gz');
  return new Response(compressed);
});
const values = await Promise.all([store.read(resource), store.read(other), store.read(resource)]);
assert.deepEqual(values, [bytes, second, bytes]);
assert.equal(requests, 1, 'concurrent resources share a single pack fetch');
await assert.rejects(store.read({ ...resource, size: 1024 * 1024 * 1024 + 1,
  recipe: { ...recipe, size: 1024 * 1024 * 1024 + 1 } }), /size|limit|large|recipe/,
  'oversized logical content must fail before output allocation');
values[0][0] = 0;
assert.deepEqual(await store.read(resource), bytes, 'reader cannot mutate cached bytes');

for (const change of [
  r => { r.schema = 1; },
  r => { r.size++; },
  r => { r.sha256 = '0'.repeat(64); },
  r => { r.chunks[0].offset = -1; },
  r => { r.chunks[0].offset = 999; },
  r => { r.chunks[0].size++; },
  r => { r.chunks[0].pack.decoded_size = MAX_PACK_SIZE + 1; },
]) {
  const invalidRecipe = structuredClone(recipe);
  change(invalidRecipe);
  await assert.rejects(store.read({ ...resource, recipe: invalidRecipe }), /recipe|chunk/, 'cached bytes must not bypass recipe admission');
}
for (const change of [
  r => { r.chunks.reverse(); },
  r => { r.chunks[0].sha256 = '0'.repeat(64); },
  r => { r.chunks.forEach(chunk => { chunk.pack.decoded_sha256 = '0'.repeat(64); }); },
  r => { r.chunks.forEach(chunk => { chunk.pack.decoded_size++; }); },
]) {
  const invalidRecipe = structuredClone(recipe);
  change(invalidRecipe);
  const invalidStore = storeFor(async () => new Response(compressed));
  await assert.rejects(invalidStore.read({ ...resource, recipe: invalidRecipe }), /hash|length/);
}
for (const initial of ['missing', 'corrupt']) {
  let attempt = 0;
  const retryStore = storeFor(async () => {
    attempt++;
    if (attempt === 1) return initial === 'missing' ? new Response(null, { status: 404 }) : new Response(new Uint8Array(compressed.length));
    return new Response(compressed);
  });
  await assert.rejects(retryStore.read(resource), /fetch failed|SHA-256 mismatch/);
  assert.deepEqual(await retryStore.read(resource), bytes);
  assert.equal(attempt, 2);
}
const cachedStore = storeFor(async () => new Response(compressed));
cachedStore.memory.set(pack.sha256, new Uint8Array(compressed.length));
assert.deepEqual(await cachedStore.read(resource), bytes, 'corrupt cache entries must be replaced');

const indexBytes = new TextEncoder().encode(JSON.stringify({ schema: 2, resources: { [resource.path]: recipe, [other.path]: other.recipe } }));
const indexCompressed = new Uint8Array(gzipSync(indexBytes));
const index = { path: 'indexes/captures.gz', sha256: await sha256(indexCompressed), size: indexCompressed.length, decoded_size: indexBytes.length, decoded_sha256: await sha256(indexBytes) };
let indexRequests = 0;
let packRequests = 0;
const indexStore = storeFor(async url => {
  if (url.pathname === '/indexes/captures.gz') { indexRequests++; return new Response(indexCompressed); }
  packRequests++;
  return new Response(compressed);
});
const indexed = ({ recipe: ignored, ...descriptor }) => ({ ...descriptor, recipe_index: index });
assert.deepEqual(await Promise.all([indexStore.read(indexed(resource)), indexStore.read(indexed(other))]), [bytes, second]);
assert.equal(indexRequests, 1);
assert.equal(packRequests, 1);
await assert.rejects(indexStore.read({ ...indexed(resource), path: 'missing.yaml' }), /recipe missing/);
let indexAttempts = 0;
const retryIndexStore = storeFor(async url => {
  if (url.pathname !== '/indexes/captures.gz') return new Response(compressed);
  indexAttempts++;
  return indexAttempts === 1 ? new Response(null, { status: 404 }) : new Response(indexCompressed);
});
await assert.rejects(retryIndexStore.read(indexed(resource)), /fetch failed/);
assert.deepEqual(await retryIndexStore.read(indexed(resource)), bytes);
assert.equal(indexAttempts, 2, 'failed lazy indexes are evicted for retry');
await assert.rejects(indexStore.read({ ...indexed(resource), recipe_index: { ...index, decoded_sha256: '0'.repeat(64) } }), /decoded hash/);
await assert.rejects(indexStore.read({ ...indexed(resource), recipe_index: { ...index, decoded_size: 64 * 1024 * 1024 + 1 } }), /decoded size/);
const oversized = new Uint8Array(gzipSync(new Uint8Array(1000)));
new DataView(oversized.buffer).setUint32(oversized.length - 4, 1, true);
assert.throws(() => boundedGunzip(oversized, 1), /length mismatch/, 'forged ISIZE cannot truncate oversized output silently');
const largePacks = new Map();
const largeResources = [];
for (const value of [1, 2, 3]) {
  const data = new Uint8Array(600 * 1024).fill(value);
  const encoded = new Uint8Array(gzipSync(data));
  const descriptor = { path: `packs/${value}.gz`, sha256: await sha256(encoded), size: encoded.length, decoded_sha256: await sha256(data), decoded_size: data.length };
  const chunks = [];
  for (let offset = 0; offset < data.length; offset += 128 * 1024) {
    const piece = data.subarray(offset, Math.min(data.length, offset + 128 * 1024));
    chunks.push({ sha256: await sha256(piece), size: piece.length, offset, pack: descriptor });
  }
  largePacks.set(descriptor.path, encoded);
  largeResources.push({ path: `source/${value}`, sha256: await sha256(data), size: data.length,
    recipe: { schema: 2, sha256: await sha256(data), size: data.length, chunks } });
}
let decodedLoads = 0;
const boundedTransport = new ChunkTransport({ sha256, validateResource() {}, maxDecodedPackBytes: MAX_PACK_SIZE,
  async readBytes(descriptor) {
    decodedLoads++;
    const encoded = largePacks.get(descriptor.path);
    assert.equal(await sha256(encoded), descriptor.sha256);
    return encoded;
  },
});
for (const item of largeResources) {
  const decoded = await boundedTransport.read(item);
  assert.equal(await sha256(decoded), item.sha256);
  assert.ok(boundedTransport.decodedPackBytes <= MAX_PACK_SIZE);
  assert.equal(boundedTransport.packs.size, 1);
}
assert.equal(decodedLoads, 3);
await Promise.all([boundedTransport.read(largeResources[0]), boundedTransport.read(largeResources[0])]);
assert.equal(decodedLoads, 4, 'evicted packs reload once for concurrent readers');
await boundedTransport.read(largeResources[1]);
largePacks.set('packs/1.gz', new Uint8Array(largePacks.get('packs/1.gz').length));
await assert.rejects(boundedTransport.read(largeResources[0]), /assert|equal/i, 'evicted packs cannot bypass compressed-byte validation');
assert.ok(boundedTransport.decodedPackBytes <= MAX_PACK_SIZE);
console.log('chunk transport integrity, concurrency, cache admission, eviction, and retry checks passed');

// Distinct logical resources can share a held pack. A selection switch must
// preserve that pending transport even when its original consumer is obsolete.
let releaseShared;
let sharedFetches = 0;
const sharingStore = storeFor(async (_url, {signal}) => {
  sharedFetches++;
  return new Promise((resolve, reject) => {
    releaseShared = () => resolve(new Response(compressed));
    signal.addEventListener('abort', () => reject(signal.reason), {once: true});
  });
});
await sharingStore.demand([resource]);
const firstSelection = sharingStore.read(resource, {selection: true});
while (!releaseShared) await new Promise(resolve => setImmediate(resolve));
sharingStore.demand([other]);
const rejectedSelection = assert.rejects(firstSelection, /Obsolete/);
const secondSelection = sharingStore.read(other, {selection: true});
releaseShared();
await rejectedSelection;
assert.deepEqual(await secondSelection, second);
assert.equal(sharedFetches, 1, 'a pack needed by the new selection must remain in flight');

let releaseRaw;
let rawFetches = 0;
const rawSharingStore = storeFor(async (_url, {signal}) => {
  rawFetches++;
  return new Promise((resolve, reject) => {
    releaseRaw = () => resolve(new Response(compressed));
    signal.addEventListener('abort', () => reject(signal.reason), {once: true});
  });
});
await rawSharingStore.demand([resource]);
const staleResource = rawSharingStore.read(resource, {selection: true});
while (!releaseRaw) await new Promise(resolve => setImmediate(resolve));
const rawRead = rawSharingStore.read(other);
// Ensure the separate Raw logical resource attaches its pack consumer.
while (rawSharingStore.transport.packs.values().next().value.consumers < 2) await new Promise(resolve => setImmediate(resolve));
await rawSharingStore.demand([]);
await assert.rejects(staleResource, /Obsolete/);
releaseRaw();
assert.deepEqual(await rawRead, second);
assert.equal(rawFetches, 1, 'Raw ownership must outlive obsolete selection ownership');
console.log('selection-shared pack and distinct Raw consumer cancellation passed');

const common = new Uint8Array([1]);
const commonCompressed = new Uint8Array(gzipSync(common));
const commonPack = {path: 'packs/common.gz', sha256: await sha256(commonCompressed), size: commonCompressed.length,
  decoded_sha256: await sha256(common), decoded_size: 1};
const commonChunk = {sha256: commonPack.decoded_sha256, size: 1, offset: 0, pack: commonPack};
const multiPackResources = await Promise.all(Array.from({length: 8}, async (_, i) => {
  const tail = new Uint8Array([i + 2]), encoded = new Uint8Array(gzipSync(tail));
  const tailPack = {path: `packs/tail-${i}.gz`, sha256: await sha256(encoded), size: encoded.length,
    decoded_sha256: await sha256(tail), decoded_size: 1};
  const value = new Uint8Array([1, i + 2]), hash = await sha256(value);
  return {path: `multi/${i}`, sha256: hash, size: 2, recipe: {schema: 2, sha256: hash, size: 2,
    chunks: [commonChunk, {sha256: tailPack.decoded_sha256, size: 1, offset: 0, pack: tailPack}]}};
}));
let releaseCommon;
let tailReads = 0;
const multiStore = storeFor(async (url, {signal}) => {
  if (url.pathname !== '/packs/common.gz') { tailReads++; throw new Error('obsolete tail downloaded'); }
  return new Promise((resolve, reject) => {
    releaseCommon = () => resolve(new Response(commonCompressed));
    signal.addEventListener('abort', () => reject(signal.reason), {once: true});
  });
});
multiStore.demand(multiPackResources);
const obsoleteMulti = multiPackResources.map(item => multiStore.read(item, {selection: true}));
const rejectedMulti = Promise.all(obsoleteMulti.map(task => assert.rejects(task, /Obsolete/)));
while (!releaseCommon) await new Promise(resolve => setImmediate(resolve));
const commonResource = {path: 'current/common', sha256: commonChunk.sha256, size: 1,
  recipe: {schema: 2, sha256: commonChunk.sha256, size: 1, chunks: [commonChunk]}};
const commonRevision = multiStore.demand([commonResource]);
const currentCommon = multiStore.read(commonResource, {selection: commonRevision});
releaseCommon();
assert.deepEqual(await currentCommon, common);
await rejectedMulti;
assert.equal(tailReads, 0, 'retaining a shared first pack must not retain obsolete logical tails or slots');
multiStore.releaseReservations(commonRevision);
assert.equal(multiStore.reservations.size, 0);
console.log('eight obsolete multi-pack resources cancel without fetching their unique tails');

let cachedPackSignal;
const cachedSelectionStore = storeFor(async (_url, {signal}) => {
  cachedPackSignal = signal;
  return new Promise((resolve, reject) => signal.addEventListener('abort', () => reject(signal.reason), {once: true}));
});
cachedSelectionStore.demand([resource]);
const cachedObsolete = cachedSelectionStore.read(resource, {selection: true});
const cachedRejected = assert.rejects(cachedObsolete, /Obsolete/);
while (!cachedPackSignal) await new Promise(resolve => setImmediate(resolve));
cachedSelectionStore.memory.set(other.sha256, second);
const cachedRevision = cachedSelectionStore.demand([other]);
assert.deepEqual(await cachedSelectionStore.read(other, {selection: cachedRevision}), second);
assert.equal(cachedPackSignal.aborted, false, 'the pending shared pack is reserved until preparation finishes');
cachedSelectionStore.releaseReservations(cachedRevision - 1);
assert.equal(cachedPackSignal.aborted, false, 'an obsolete preparation cannot release current reservations');
cachedSelectionStore.releaseReservations(cachedRevision);
assert.equal(cachedPackSignal.aborted, true, 'a logical cache hit must release its unused transport reservation');
await cachedRejected;
assert.equal(cachedSelectionStore.reservations.size, 0);
await assert.rejects(cachedSelectionStore.read(other, {selection: cachedRevision - 1}), /Obsolete/);
console.log('cache-hit reservation cleanup and stale preparation exclusion passed');

let releaseDecodedHash;
const abortDuringHash = new AbortController();
const hashingTransport = new ChunkTransport({validateResource() {}, readBytes: async () => compressed,
  sha256: async value => {
    if (!releaseDecodedHash) await new Promise(resolve => { releaseDecodedHash = resolve; });
    return sha256(value);
  },
});
const hashingRead = hashingTransport.read(resource, undefined, abortDuringHash.signal);
const hashingRejected = assert.rejects(hashingRead, /Obsolete/);
while (!releaseDecodedHash) await new Promise(resolve => setImmediate(resolve));
abortDuringHash.abort(new Error('Obsolete fixture selection'));
releaseDecodedHash();
await hashingRejected;
await new Promise(resolve => setImmediate(resolve));
assert.equal(hashingTransport.decodedPackBytes, 0, 'canceled hash continuations must not populate decoded-cache accounting');
assert.equal(hashingTransport.packSizes.size, 0);
console.log('cancellation during async decoded hashing cannot mutate cache accounting');

} finally {
  for (const key of Object.getOwnPropertyNames(globalThis)) {
    if (!Object.hasOwn(savedGlobals, key) && Object.getOwnPropertyDescriptor(globalThis, key).configurable) delete globalThis[key];
  }
  for (const [key, descriptor] of Object.entries(savedGlobals)) {
    if (descriptor.configurable) Object.defineProperty(globalThis, key, descriptor);
  }
}

});

await test('download_scheduler contracts', async () => {
const savedGlobals = Object.getOwnPropertyDescriptors(globalThis);
try {
const { default: assert } = await import('node:assert/strict');
const { DownloadScheduler } = await import('./data.js');


let active = 0, maxActive = 0, starts = 0;
const scheduler = new DownloadScheduler();
const releases = [];
const create = () => { starts++; active++; maxActive = Math.max(maxActive, active); return new Promise(resolve => releases.push(() => { active--; resolve('bytes'); })); };
const tasks = Array.from({ length: 12 }, (_, i) => scheduler.read(String(i), create));
await Promise.resolve(); assert.equal(starts, 8);
const shared = scheduler.read('0', create); assert.equal(shared, tasks[0]);
while (scheduler.tasks.size) { releases.splice(0).forEach(release => release()); await new Promise(resolve => setImmediate(resolve)); }
await Promise.all(tasks); assert.equal(maxActive, 8); assert.equal(starts, 12);
const retry = new DownloadScheduler();
await assert.rejects(retry.read('fail', async () => { throw new Error('transport'); }), /transport/);
await new Promise(resolve => setImmediate(resolve));
assert.equal(await retry.read('fail', async () => 'recovered'), 'recovered');

const stale = new DownloadScheduler({limit: 2});
let obsoleteStarted = false;
const old = stale.read('old', signal => new Promise((resolve, reject) => {
  signal.addEventListener('abort', () => reject(signal.reason), {once: true});
}), {selection: true});
const raw = stale.read('raw', () => new Promise(resolve => { globalThis.releaseRaw = resolve; }));
const obsolete = stale.read('queued', async () => { obsoleteStarted = true; }, {selection: true});
await Promise.resolve();
stale.demand(['new']);
const rejected = Promise.all([assert.rejects(old, /Obsolete/), assert.rejects(obsolete, /Obsolete/)]);
assert.equal(await stale.read('new', async () => 'current', {selection: true}), 'current');
await rejected;
assert.equal(obsoleteStarted, false);
globalThis.releaseRaw('raw');
assert.equal(await raw, 'raw');
console.log('Bounded scheduler, obsolete cancellation, pinned Raw, and retry contracts passed');

} finally {
  for (const key of Object.getOwnPropertyNames(globalThis)) {
    if (!Object.hasOwn(savedGlobals, key) && Object.getOwnPropertyDescriptor(globalThis, key).configurable) delete globalThis[key];
  }
  for (const [key, descriptor] of Object.entries(savedGlobals)) {
    if (descriptor.configurable) Object.defineProperty(globalThis, key, descriptor);
  }
}

});

await test('json_resources contracts', async () => {
const savedGlobals = Object.getOwnPropertyDescriptors(globalThis);
try {
const { default: assert } = await import('node:assert/strict');
const { execFileSync } = await import('node:child_process');
const { createWorkerParser, HttpResourceStore, sha256 } = await import('./data.js');
const { FixtureResources } = await import('./resources.js');
const { gzipSync } = await import('./vendor/fflate/esm/browser.js');


const fingerprint = 'a'.repeat(64);
const payloads = new Map();
async function snapshot(kind, value, version = null, expectedMetadataHash = null) {
  const encoded = JSON.parse(execFileSync('python3', ['-c', `
import hashlib
import json
import sys
sys.path.insert(0, 'conformance/utils/src')
from viewer_codec import encode_typed
value = json.load(sys.stdin)
metadata = None
if all(isinstance(record, dict) and 'member' in record and 'value' in record for record in value.values()):
    records = {key: {'document': record['value']['document'], 'member': record['member']}
               for key, record in value.items()}
    data = (json.dumps(encode_typed(records), ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\\n').encode()
    metadata = hashlib.sha256(data).hexdigest()
print(json.dumps({'typed': encode_typed(value), 'metadata': metadata}))
`], {
    input: JSON.stringify(value), cwd: new URL('../../', import.meta.url), encoding: 'utf8',
    env: Object.fromEntries(['PATH', 'LANG', 'LC_ALL'].filter(key => process.env[key]).map(key => [key, process.env[key]])),
  }));
  const resource = { path: `conformance/viewer-data/sample/${version || 'inputs'}.json.gz`, kind, family: 'sample', implementation: version ? 'dynamo_v2' : null, version, record_contract: version ? 'stream' : null, format: 'typed-json-v1', compression: 'gzip', source_fingerprint: fingerprint };
  const bytes = gzipSync(new TextEncoder().encode(JSON.stringify({ schema: 1, kind, family: resource.family, implementation: resource.implementation, version, source_fingerprint: fingerprint, value: encoded.typed })));
  if (kind === 'resolved-capture') resource.resolved_metadata_sha256 = expectedMetadataHash || encoded.metadata;
  resource.sha256 = await sha256(bytes); resource.size = bytes.length;
  payloads.set(resource.path, bytes);
  return resource;
}
const request = { input: 'hello', init: {}, finish_reason: 'stop', tools: [], chunks: [] };
const inputs = await snapshot('inputs-and-golden', { input_document: { family: 'sample' }, golden_document: { family: 'sample' }, cases: { one: { lifecycle: 'active', scenario: 'one', description: 'one', policy: [], display_id: 'UNIFIED.1', historical_ids: [], request, golden: { assembled: [] } } } });
const original = { path: 'conformance/fixtures-unified-v2/families/sample/dynamo_v2-1.0.0.yaml', kind: 'history', family: 'sample', implementation: 'dynamo_v2', version: '1.0.0' };
const records = { one: { member: original.path, value: { case_key: 'UNIFIED.1', stimulus: { ref: 'current' }, observation: { value: { assembled: [] } }, document: { captured_with: { dynamo_v2: '1.0.0' } } } } };
const first = await snapshot('resolved-capture', records, '1.0.0');
const other = await snapshot('resolved-capture', records, '2.0.0');
const catalog = { metadata: {}, sources: { unified: [inputs, first, other], original_unified: [original] } };
const downloads = [];
let corrupt = false;
const store = new HttpResourceStore({ allowSourceFiles: true, baseUrl: 'http://localhost/', fetchImpl: async url => {
  const path = decodeURI(url.pathname.slice(1));
  downloads.push(path);
  const bytes = payloads.get(path).slice();
  if (corrupt && path === first.path) bytes[bytes.length - 1] ^= 1;
  return { ok: true, arrayBuffer: async () => bytes.buffer };
} });
const parser = createWorkerParser();
parser.parseYaml = async () => { throw new Error('normal JSON browsing attempted YAML parsing'); };
const reader = new FixtureResources(catalog, store, parser);
const selected = [{ source: { implementation: 'dynamo_v2', version: '1.0.0' } }];
assert.deepEqual(reader.unifiedDependencies(selected), [inputs, first]);
corrupt = true;
await assert.rejects(reader.unified('sample', 'dynamo_v2', '1.0.0'), /SHA-256/);
corrupt = false;
const resolved = await reader.unified('sample', 'dynamo_v2', '1.0.0');
assert.deepEqual(resolved.records.get('one'), records.one);
resolved.records.get('one').value.observation.value.assembled.push({ kind: 'text', text: 'mutated' });
assert.deepEqual((await reader.unified('sample', 'dynamo_v2', '1.0.0')).records.get('one'), records.one, 'cached snapshots must isolate mutable case overlays');
await reader.prepareUnified(selected);
assert.equal(downloads.includes(other.path), false, 'selection demand must not fetch another version');
assert.equal(downloads.some(path => path.endsWith('.yaml')), false, 'resolved JSON must never replay YAML history');
assert.equal(downloads.filter(path => path === first.path).length, 2, 'failed integrity checks must remain retryable');
const invalid = await snapshot('resolved-capture', { one: { ...records.one, member: 'outside.yaml' } }, '3.0.0');
catalog.sources.unified.push(invalid);
await assert.rejects(reader.unified('sample', 'dynamo_v2', '3.0.0'), /invalid source/);
const forged = structuredClone(records);
forged.one.value.document.captured_with = { vllm_python: '999.0.0' };
forged.one.value.document.record_metadata = { verified_recapture: { receipt: 'forged' } };
const falseProducer = await snapshot('resolved-capture', forged, '4.0.0', first.resolved_metadata_sha256);
catalog.sources.unified.push(falseProducer);
await assert.rejects(reader.unified('sample', 'dynamo_v2', '4.0.0'), /metadata binding/);
const falseProducerWithBoundMetadata = await snapshot('resolved-capture', forged, '5.0.0');
catalog.sources.unified.push(falseProducerWithBoundMetadata);
await assert.rejects(reader.unified('sample', 'dynamo_v2', '5.0.0'), /false producer identity/);
const overridden = structuredClone(records);
overridden.one.value.document.captured_with = { dynamo_v2: '1.0.1' };
catalog.sources.unified.push(await snapshot('resolved-capture', overridden, '1.0.1'));
assert.deepEqual((await reader.unified('sample', 'dynamo_v2', '1.0.1')).records.get('one'), overridden.one);
const futureProducer = structuredClone(records);
futureProducer.one.value.document.captured_with = { dynamo_v2: '99.0.0' };
catalog.sources.unified.push(await snapshot('resolved-capture', futureProducer, '6.0.0'));
await assert.rejects(reader.unified('sample', 'dynamo_v2', '6.0.0'), /false producer identity/);
const futureMember = { ...original, path: original.path.replace('1.0.0', '99.0.0'), version: '99.0.0' };
catalog.sources.original_unified = [...catalog.sources.original_unified, futureMember];
catalog.sources.unified.push(await snapshot('resolved-capture', { one: { ...records.one, member: futureMember.path } }, '7.0.0'));
await assert.rejects(reader.unified('sample', 'dynamo_v2', '7.0.0'), /invalid source/);
console.log('selected JSON demand, zero YAML reads, full records, source validation, integrity rejection, and retry passed');

} finally {
  for (const key of Object.getOwnPropertyNames(globalThis)) {
    if (!Object.hasOwn(savedGlobals, key) && Object.getOwnPropertyDescriptor(globalThis, key).configurable) delete globalThis[key];
  }
  for (const [key, descriptor] of Object.entries(savedGlobals)) {
    if (descriptor.configurable) Object.defineProperty(globalThis, key, descriptor);
  }
}

});

await test('originals contracts', async () => {
const savedGlobals = Object.getOwnPropertyDescriptors(globalThis);
try {
const { default: assert } = await import('node:assert/strict');
const { mkdtemp, mkdir, writeFile, rm } = await import('node:fs/promises');
const { tmpdir } = await import('node:os');
const { join } = await import('node:path');
const { HttpResourceStore, createNodeResourceStore, sha256 } = await import('./data.js');


const bytes = new Uint8Array([0xef, 0xbb, 0xbf, 97, 58, 32, 49, 13, 10, 0, 255]);
const resource = { path: 'conformance/source.yaml', sha256: await sha256(bytes), size: bytes.length };
for (const path of [resource.path, 'conformance/archive.tar.gz', 'conformance/capture.json.gz']) {
  let requests = 0;
  const published = new HttpResourceStore({ baseUrl: 'http://local/', fetchImpl: async () => { requests++; return new Response(bytes); } });
  // Admission must run before cache reuse, including another path with identical bytes.
  published.memory.set(resource.sha256, bytes);
  await assert.rejects(published.read({ ...resource, path }), /Source files disabled/);
  assert.equal(requests, 0);
}
const requests = [];
const checkout = new HttpResourceStore({ baseUrl: 'http://local/frontend-crates/', allowSourceFiles: true, fetchImpl: async url => {
  requests.push(url.pathname);
  return new Response(bytes);
} });
assert.deepEqual(await checkout.read(resource), bytes);
assert.deepEqual(requests, ['/frontend-crates/conformance/source.yaml']);
assert.deepEqual(await checkout.read(resource), bytes);
assert.equal(requests.length, 1);
checkout.memory.set(resource.sha256, new Uint8Array(bytes.length));
assert.deepEqual(await checkout.read(resource), bytes);
assert.equal(requests.length, 2, 'checkout cache hits still require exact-byte validation');
await assert.rejects(checkout.read({ ...resource, sha256: '0'.repeat(64) }), /SHA-256 mismatch/);
const filesystemRoot = await mkdtemp(join(tmpdir(), 'source-admission-'));
try {
  await mkdir(join(filesystemRoot, 'conformance'));
  await writeFile(join(filesystemRoot, resource.path), bytes);
  await assert.rejects((await createNodeResourceStore(filesystemRoot)).read(resource), /Source files disabled/);
  const source = await createNodeResourceStore(filesystemRoot, { allowSourceFiles: true });
  assert.deepEqual(await source.read(resource), bytes);
  assert.deepEqual(await source.read(resource), bytes);
  assert.equal(source.memory.size, 0);
} finally {
  await rm(filesystemRoot, { recursive: true });
}
console.log('explicit checkout reads, exact bytes, cache validation, and production source rejection passed');

} finally {
  for (const key of Object.getOwnPropertyNames(globalThis)) {
    if (!Object.hasOwn(savedGlobals, key) && Object.getOwnPropertyDescriptor(globalThis, key).configurable) delete globalThis[key];
  }
  for (const [key, descriptor] of Object.entries(savedGlobals)) {
    if (descriptor.configurable) Object.defineProperty(globalThis, key, descriptor);
  }
}

});

await test('raw_lookup contracts', async () => {
const savedGlobals = Object.getOwnPropertyDescriptors(globalThis);
try {
const { default: assert } = await import('node:assert/strict');
const { gzipSync } = await import('node:zlib');
const { HttpResourceStore, sha256 } = await import('./data.js');
const { rawResource } = await import('./resources.js');


const bytes = new Uint8Array([0xef, 0xbb, 0xbf, 97, 13, 10, 0, 255]);
const compressed = new Uint8Array(gzipSync(bytes));
const pack = { path: 'packs/original.gz', size: compressed.length, sha256: await sha256(compressed), decoded_size: bytes.length, decoded_sha256: await sha256(bytes) };
const resource = { path: 'conformance/archive.tar.gz', size: bytes.length, sha256: await sha256(bytes), recipe: { schema: 2, size: bytes.length, sha256: await sha256(bytes), chunks: [{ sha256: await sha256(bytes), size: bytes.length, offset: 0, pack }] } };
const unified = { ...resource, path: 'conformance/Unified/family/1.0.0.yaml' };
const entries = {
  [resource.path]: { resource, member: null },
  [unified.path]: { resource: unified, member: null },
  'family/outputs/case.yaml': { path: resource.path, member: 'family/outputs/case.yaml' },
  'conformance/family/outputs/case.yaml': { path: resource.path, member: 'family/outputs/case.yaml' },
};
const buckets = new Map();
for (const [path, entry] of Object.entries(entries)) {
  const key = (await sha256(new TextEncoder().encode(path))).slice(0, 2);
  if (!buckets.has(key)) buckets.set(key, {});
  buckets.get(key)[path] = entry;
}
const catalog = { transport: { schema: 2, raw_buckets: {} } };
const files = new Map([[pack.path, compressed]]);
for (const [key, resources] of buckets) {
  const decoded = new TextEncoder().encode(JSON.stringify({ schema: 2, resources }));
  const encoded = new Uint8Array(gzipSync(decoded));
  const descriptor = { path: `indexes/raw-${key}.gz`, sha256: await sha256(encoded), size: encoded.length, decoded_size: decoded.length, decoded_sha256: await sha256(decoded) };
  catalog.transport.raw_buckets[key] = descriptor;
  files.set(descriptor.path, encoded);
}
const requests = [];
const store = new HttpResourceStore({ baseUrl: 'http://local/prefix/', fetchImpl: async url => {
  const path = url.pathname.slice('/prefix/'.length);
  requests.push(path);
  const data = files.get(path);
  return data ? new Response(data) : new Response(null, { status: 404 });
} });
assert.equal(requests.length, 0, 'original lookup is fully lazy');
for (const path of Object.keys(entries)) {
  const resolved = await rawResource(catalog, store, path);
  assert.equal(resolved.resource.path, path === unified.path ? unified.path : resource.path);
  assert.equal(resolved.member, entries[path].member);
  assert.deepEqual(await store.read(resolved.resource), bytes);
}
assert.equal(requests.filter(path => path === pack.path).length, 1, 'all aliases and originals share the pack');
assert.equal(requests.some(path => path.endsWith('.yaml') || path.endsWith('.tar.gz')), false);
await assert.rejects(rawResource(catalog, store, 'unknown.yaml'), /not indexed/);
const hash = (await sha256(new TextEncoder().encode(resource.path))).slice(0, 2);
const badCatalog = structuredClone(catalog);
badCatalog.transport.raw_buckets[hash].sha256 = '0'.repeat(64);
await assert.rejects(rawResource(badCatalog, store, resource.path), /SHA-256 mismatch/);
const fakeStore = entry => ({ transport: { index: async () => ({ resources: { [resource.path]: entry } }) } });
await assert.rejects(rawResource(catalog, fakeStore({ member: 1 }), resource.path), /Invalid original alias/);
await assert.rejects(rawResource(catalog, fakeStore({ path: resource.path, member: 'member.yaml' }), resource.path), /Invalid original alias target/);
console.log('lazy raw buckets, Unified paths, archive aliases, exact bytes, shared packs, and descriptor integrity passed');

} finally {
  for (const key of Object.getOwnPropertyNames(globalThis)) {
    if (!Object.hasOwn(savedGlobals, key) && Object.getOwnPropertyDescriptor(globalThis, key).configurable) delete globalThis[key];
  }
  for (const [key, descriptor] of Object.entries(savedGlobals)) {
    if (descriptor.configurable) Object.defineProperty(globalThis, key, descriptor);
  }
}

});

await test('source_archives contracts', async () => {
const savedGlobals = Object.getOwnPropertyDescriptors(globalThis);
try {
const { default: assert } = await import('node:assert/strict');
const { gzipSync } = await import('node:zlib');
const { HttpResourceStore, sha256, readTarGzip } = await import('./data.js');
const { initializeDevelopmentParser } = await import('./yaml_data.js');


initializeDevelopmentParser();

function archiveBytes(files, trailingZeros = 1024) {
  const blocks = [];
  for (const [name, data, type = 48] of files) {
    const header = Buffer.alloc(512);
    header.write(name);
    header.write('0000644\0', 100);
    header.write('0000000\0', 108);
    header.write('0000000\0', 116);
    header.write(data.length.toString(8).padStart(11, '0') + '\0', 124);
    header.write('00000000000\0', 136);
    header.fill(32, 148, 156);
    header[156] = type;
    const sum = header.reduce((total, byte) => total + byte, 0);
    header.write(sum.toString(8).padStart(6, '0') + '\0 ', 148);
    blocks.push(header, Buffer.from(data), Buffer.alloc((512 - data.length % 512) % 512));
  }
  return new Uint8Array(gzipSync(Buffer.concat([...blocks, Buffer.alloc(trailingZeros)])));
}

const name = 'families/qwen3/vllm_python-1.2.3.yaml';
const bytes = new TextEncoder().encode('typed: [null, "null", 1, 1.0, true]\r\n');
const compressed = archiveBytes([[name, bytes]]);
const archive = { path: 'conformance/fixtures-unified-v2/versions/vllm-1.2.3.tar.gz', sha256: await sha256(compressed), size: compressed.length };
const resource = { path: `conformance/fixtures-unified-v2/${name}`, sha256: await sha256(bytes), size: bytes.length, archive };
let requests = 0;
const store = new HttpResourceStore({ baseUrl: 'http://local/', allowSourceFiles: true, retainBytes: false,
  fetchImpl: async url => { requests++; assert.equal(url.pathname, '/' + archive.path); return new Response(compressed); } });
const readers = await Promise.all(Array.from({ length: 12 }, () => store.read(resource)));
assert.equal(requests, 1, 'concurrent members share their archive download without nesting scheduler slots');
for (const result of readers) assert.deepEqual(result, bytes);
readers[0][0] = 0;
assert.deepEqual(await store.read(resource), bytes);
await assert.rejects(store.read({ ...resource, sha256: '0'.repeat(64) }), /integrity mismatch/);
await assert.rejects(store.read({ ...resource, path: resource.path.replace(name, 'families/qwen3/missing.yaml') }), /member missing/);
await assert.rejects(store.read({ ...resource, path: 'outside/file.yaml' }), /binding/);
const production = new HttpResourceStore({ baseUrl: 'http://local/', fetchImpl: async () => { throw new Error('must not fetch'); } });
production.memory.set(resource.sha256, bytes);
await assert.rejects(production.read(resource), /Source files disabled/);
let attempts = 0;
const retry = new HttpResourceStore({ baseUrl: 'http://local/', allowSourceFiles: true,
  fetchImpl: async () => ++attempts === 1 ? new Response(new Uint8Array(compressed.length)) : new Response(compressed) });
await assert.rejects(retry.read(resource), /SHA-256 mismatch/);
assert.deepEqual(await retry.read(resource), bytes);
assert.equal(attempts, 2);
console.log('engine/version archive checkout reads, exact bytes, concurrent readers, retry, and production rejection passed');

function paxComment(size) {
  const prefix = `${size} comment=`;
  return Buffer.from(prefix + 'x'.repeat(size - prefix.length - 1) + '\n');
}
for (const type of [76, 120, 103]) {
  const extension = type === 76 ? Buffer.from('safe.yaml\0') : paxComment(65536);
  const accepted = archiveBytes([['extension', extension, type], ['safe.yaml', bytes]]);
  assert.deepEqual(readTarGzip(accepted).get('safe.yaml'), bytes);
  const oversized = type === 76 ? Buffer.alloc(65537, 120) : paxComment(65537);
  assert.throws(() => readTarGzip(archiveBytes([['extension', oversized, type], ['safe.yaml', bytes]])), /extension metadata exceeds limit/);
}
assert.throws(() => readTarGzip(archiveBytes([
  ['first', paxComment(32768), 120], ['second', paxComment(32769), 120], ['safe.yaml', bytes],
])), /extension metadata exceeds limit/, 'several small extensions share one member budget');
assert.throws(() => readTarGzip(archiveBytes([
  ...Array.from({length: 129}, () => ['extension', Buffer.alloc(0), 120]), ['safe.yaml', bytes],
])), /extension metadata exceeds limit/, 'empty extension headers cannot evade the metadata budget');
const padding = archiveBytes([['safe.yaml', bytes]], 1024 * 1024);
assert.deepEqual(readTarGzip(padding, {maximumDecoded: 2 * 1024 * 1024}).get('safe.yaml'), bytes);
assert.throws(() => readTarGzip(padding, {maximumDecoded: 1024}), /decoded limit/);
console.log('checkout GNU/PAX metadata boundaries, cumulative/header budgets, and bounded zero padding passed');

} finally {
  for (const key of Object.getOwnPropertyNames(globalThis)) {
    if (!Object.hasOwn(savedGlobals, key) && Object.getOwnPropertyDescriptor(globalThis, key).configurable) delete globalThis[key];
  }
  for (const [key, descriptor] of Object.entries(savedGlobals)) {
    if (descriptor.configurable) Object.defineProperty(globalThis, key, descriptor);
  }
}

});

await test('snapshots contracts', async () => {
const savedGlobals = Object.getOwnPropertyDescriptors(globalThis);
try {
const { default: assert } = await import('node:assert/strict');
const { gzipSync } = await import('./vendor/fflate/esm/browser.js');
const { parseSnapshot } = await import('./data.js');
const { sha256 } = await import('./data.js');


const resource = { path: 'family/capture.json.gz', format: 'typed-json-v1', compression: 'gzip', kind: 'inputs-and-golden', family: 'family', implementation: 'dynamo_v2', version: '1.0.0', source_fingerprint: 'a'.repeat(64) };
const envelope = { schema: 1, kind: resource.kind, family: resource.family, implementation: resource.implementation, version: resource.version, source_fingerprint: resource.source_fingerprint, value: ['object', [[['str', 'case'], ['int', '9007199254740993']]]] };
const encode = value => gzipSync(new TextEncoder().encode(JSON.stringify(value)));
assert.deepEqual(await parseSnapshot(encode(envelope), resource), { case: 9007199254740993n });
const canonicalBytes = value => new TextEncoder().encode(JSON.stringify(value) + '\n');
const canonicalResource = { ...resource, compression: 'none', transport: 'chunked-v2' };
assert.deepEqual(await parseSnapshot(canonicalBytes(envelope), canonicalResource), { case: 9007199254740993n });
await assert.rejects(() => parseSnapshot(canonicalBytes(envelope), { ...canonicalResource, transport: undefined }), /format/);
for (const field of ['kind', 'family', 'implementation', 'version', 'source_fingerprint', 'schema']) await assert.rejects(() => parseSnapshot(canonicalBytes({ ...envelope, [field]: 'wrong' }), canonicalResource), /identity/);
for (const field of ['kind', 'family', 'implementation', 'version', 'source_fingerprint', 'schema']) await assert.rejects(() => parseSnapshot(encode({ ...envelope, [field]: 'wrong' }), resource), /identity/);
await assert.rejects(() => parseSnapshot(encode({ ...envelope, value: ['int', '1.5'] }), resource), /typed integer/);
await assert.rejects(() => parseSnapshot(new Uint8Array([1, 2]), resource));
await assert.rejects(() => parseSnapshot(encode(envelope), { ...resource, format: 'unknown' }), /format/);
await assert.rejects(() => parseSnapshot(encode(envelope), { ...resource, compression: 'unknown' }), /format/);

const groupedResource = { ...resource, path: 'bundles/dynamo_v2-1.0.0.json.gz', kind: 'resolved-capture', family: null,
  families: ['sample'], record_contract: 'stream', resolved_metadata_sha256: '' };
const document = ['object', []];
const record = ['object', [
  [['str', 'value'], ['object', [[['str', 'document'], document]]]],
  [['str', 'member'], ['str', 'source.yaml']],
]];
const records = ['object', [[['str', 'one'], record]]];
const groupedValue = ['object', [[['str', 'sample'], records]]];
const caseMetadataValue = ['object', [
  [['str', 'document'], document], [['str', 'member'], ['str', 'source.yaml']],
]];
const caseMetadata = ['object', [[['str', 'one'], caseMetadataValue]]];
const groupedMetadata = ['object', [[['str', 'sample'], caseMetadata]]];
groupedResource.resolved_metadata_sha256 = await sha256(new TextEncoder().encode(JSON.stringify(groupedMetadata) + '\n'));
const groupedEnvelope = { ...envelope, kind: 'resolved-capture', family: null, families: ['sample'], value: groupedValue };
assert.deepEqual(await parseSnapshot(encode(groupedEnvelope), groupedResource), { sample: { one: { value: { document: {} }, member: 'source.yaml' } } });
assert.deepEqual(await parseSnapshot(canonicalBytes(groupedEnvelope), { ...groupedResource, compression: 'none', transport: 'chunked-v2' }), { sample: { one: { value: { document: {} }, member: 'source.yaml' } } });
await assert.rejects(() => parseSnapshot(canonicalBytes(groupedEnvelope), { ...groupedResource, compression: 'none', transport: 'chunked-v2', resolved_metadata_sha256: 'b'.repeat(64) }), /metadata binding/);
await assert.rejects(() => parseSnapshot(encode({ ...groupedEnvelope, families: ['other'] }), groupedResource), /identity/);
await assert.rejects(() => parseSnapshot(encode(groupedEnvelope), { ...groupedResource, resolved_metadata_sha256: 'b'.repeat(64) }), /metadata binding/);
console.log('snapshot format, identity, fingerprint, malformed payload, and compression guards passed');

} finally {
  for (const key of Object.getOwnPropertyNames(globalThis)) {
    if (!Object.hasOwn(savedGlobals, key) && Object.getOwnPropertyDescriptor(globalThis, key).configurable) delete globalThis[key];
  }
  for (const [key, descriptor] of Object.entries(savedGlobals)) {
    if (descriptor.configurable) Object.defineProperty(globalThis, key, descriptor);
  }
}

});
