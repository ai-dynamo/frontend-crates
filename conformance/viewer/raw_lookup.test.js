// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import assert from 'node:assert/strict';
import { gzipSync } from 'node:zlib';
import { HttpResourceStore, sha256 } from './data.js';
import { rawResource } from './raw_lookup.js';

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
