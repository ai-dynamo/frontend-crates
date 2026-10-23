// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import assert from 'node:assert/strict';
import { gzipSync } from 'node:zlib';
import { HttpResourceStore, sha256 } from './data.js';
import { boundedGunzip, ChunkTransport, MAX_PACK_SIZE } from './transport.js';

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
