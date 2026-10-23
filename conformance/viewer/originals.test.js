// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { HttpResourceStore, createNodeResourceStore, sha256 } from './data.js';

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
