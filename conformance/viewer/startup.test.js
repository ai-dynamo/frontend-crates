// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { test } from 'node:test';

await test('startup contracts', async () => {
const savedGlobals = Object.getOwnPropertyDescriptors(globalThis);
try {
const { default: assert } = await import('node:assert/strict');
const { sha256 } = await import('./data.js');
const { configureDevelopmentCatalog, discoverHistory, loadStartupCatalog, validateStartupCatalog } = await import('./app.js');
const { gzipSync } = await import('./vendor/fflate/esm/browser.js');


const base = new URL('http://localhost/prefix/');
const catalog = { schema: 1, revision: 'a'.repeat(64), metadata: { tabs: [] },
  sources: { archives: [], unified: [{ path: 'capture', format: 'typed-json-v1' }] },
  transport: { schema: 2, raw_buckets: {} } };
catalog.runtime_sha256 = await sha256(new TextEncoder().encode(JSON.stringify(catalog) + '\n'));
const responseForText = value => new Response(gzipSync(new TextEncoder().encode(value)));
const responseFor = value => responseForText(JSON.stringify(value) + '\n');
const unorderedSource = JSON.stringify({ ...catalog, runtime_sha256: undefined }).replace('"tabs":[]', '"tabs":[],"group_labels":{"v1_split":"Split","1":"One","2":"Two","10":"Ten"}') + '\n';
const unorderedHash = await sha256(new TextEncoder().encode(unorderedSource));
const canonicalSource = unorderedSource.slice(0, -2) + `,"runtime_sha256":"${unorderedHash}"}\n`;
const orderedResult = await loadStartupCatalog(base, new URLSearchParams(), { fetchImpl: async () => responseForText(canonicalSource) });
assert.equal(orderedResult.catalog.metadata.group_labels.v1_split, 'Split');
assert.equal(orderedResult.catalog.metadata.group_labels['10'], 'Ten');
assert.notEqual(await sha256(new TextEncoder().encode(JSON.stringify(JSON.parse(unorderedSource)) + '\n')), unorderedHash,
  'fixture must exercise JavaScript integer-key reordering relative to Rust source');
for (const malformed of [canonicalSource.trimEnd(), canonicalSource.replace(',"runtime_sha256"', ', "runtime_sha256"'),
  canonicalSource.replace('"Ten"', '"Changed"'), canonicalSource.replace(`,"runtime_sha256":"${unorderedHash}"`, '')]) {
  await assert.rejects(loadStartupCatalog(base, new URLSearchParams(), { fetchImpl: async () => responseForText(malformed) }), /integrity/);
}
for (const query of ['', 'startup=compressed', 'startup=external', 'startup=bootstrap', 'startup=inline',
  'startup=bootstrap&raw=conformance/source.yaml', 'startup=external&slots_tab-unified=old']) {
  const requests = [];
  const result = await loadStartupCatalog(base, new URLSearchParams(query), {
    fetchImpl: async (url, options) => { requests.push(url.pathname); assert.equal(options.cache, 'no-store'); return responseFor(catalog); },
  });
  assert.deepEqual(result.catalog, catalog);
  assert.deepEqual(requests, ['/prefix/conformance/viewer-resolved-catalog.json.gz']);
  assert.equal(result.store.memory.size, 0, 'catalog load must not preload captures or originals');
  assert.equal(result.store.allowSourceFiles, false);
}
for (const mutate of [
  value => { value.revision = 'b'.repeat(64); },
  value => { value.runtime_sha256 = '0'.repeat(64); },
  value => { delete value.runtime_sha256; },
  value => { value.transport.raw_buckets.ff = {}; },
  value => { value.sources.unified[0].path = 'changed'; },
]) {
  const changed = structuredClone(catalog); mutate(changed);
  await assert.rejects(loadStartupCatalog(base, new URLSearchParams(), { fetchImpl: async () => responseFor(changed) }), /integrity/);
}
configureDevelopmentCatalog(null);
await assert.rejects(loadStartupCatalog(base, new URLSearchParams(), { fetchImpl: async () => new Response(null, { status: 404 }) }), /404/);
const yamlCatalog = structuredClone(catalog);
delete yamlCatalog.runtime_sha256;
delete yamlCatalog.transport;
delete yamlCatalog.sources.unified[0].format;
let fallbackCalls = 0;
configureDevelopmentCatalog(async (actualBase, fetchImpl) => {
  fallbackCalls++;
  assert.equal(actualBase, base);
  const response = await fetchImpl(new URL('conformance/viewer-unified-catalog.json', actualBase));
  return response.json();
});
try {
  const requests = [];
  const loaded = await loadStartupCatalog(base, new URLSearchParams('startup=inline'), { fetchImpl: async url => {
    requests.push(url.pathname);
    return url.pathname.endsWith('.gz') ? new Response(null, { status: 404 }) : Response.json(yamlCatalog);
  } });
  assert.deepEqual(loaded.catalog, yamlCatalog);
  assert.equal(loaded.store.allowSourceFiles, true);
  assert.equal(fallbackCalls, 1);
  assert.deepEqual(requests, ['/prefix/conformance/viewer-resolved-catalog.json.gz', '/prefix/conformance/viewer-unified-catalog.json']);
  for (const response of [new Response(null, { status: 503 }), responseFor({}), responseFor({ ...catalog, runtime_sha256: '0'.repeat(64) }), new Response(new Uint8Array([1, 2]))]) {
    await assert.rejects(loadStartupCatalog(base, new URLSearchParams(), { fetchImpl: async () => response }));
    assert.equal(fallbackCalls, 1, 'development fallback must not hide corrupt or failed published data');
  }
} finally { configureDevelopmentCatalog(null); }
assert.throws(() => validateStartupCatalog(yamlCatalog), /regeneration/);
assert.deepEqual(validateStartupCatalog(yamlCatalog, { resolved: false }), yamlCatalog);
const unknown = structuredClone(yamlCatalog);
unknown.sources.unified[0].format = 'unknown';
assert.throws(() => validateStartupCatalog(unknown, { resolved: false }), /Unsupported/);
for (const invalid of [null, false, 0, '', [], {}]) {
  assert.throws(() => validateStartupCatalog(invalid), /Unsupported/);
}
console.log('canonical startup modes, runtime identity, lazy loading, fail-closed publication, and development fallback passed');

// Embedded transport must exercise normal recipe admission without any HTTP.
const encode = bytes => Buffer.from(bytes).toString('base64');
const packBytes = new TextEncoder().encode('recent capture');
const packed = gzipSync(packBytes);
const pack = { path: 'packs/recent.gz', size: packed.length, sha256: await sha256(packed), decoded_size: packBytes.length, decoded_sha256: await sha256(packBytes) };
const recent = { path: 'recent.json', format: 'typed-json-v1', transport: 'chunked-v2', size: packBytes.length, sha256: await sha256(packBytes) };
const recipe = { schema: 2, size: recent.size, sha256: recent.sha256, chunks: [{ pack, offset: 0, size: recent.size, sha256: recent.sha256 }] };
const indexBytes = new TextEncoder().encode(JSON.stringify({ schema: 2, resources: { [recent.path]: recipe } }));
const indexCompressed = gzipSync(indexBytes);
recent.recipe_index = { path: 'indexes/recent.gz', size: indexCompressed.length, sha256: await sha256(indexCompressed), decoded_size: indexBytes.length, decoded_sha256: await sha256(indexBytes) };
const embeddedCatalog = structuredClone(catalog);
delete embeddedCatalog.runtime_sha256;
embeddedCatalog.sources.unified = [recent];
embeddedCatalog.capture_generation = { mode: 'latest', current: [{ sha256: 'a'.repeat(64) }] };
embeddedCatalog.runtime_sha256 = await sha256(new TextEncoder().encode(JSON.stringify(embeddedCatalog) + '\n'));
const catalogText = new TextEncoder().encode(JSON.stringify(embeddedCatalog) + '\n');
const catalogCompressed = gzipSync(catalogText);
const payload = { schema: 1, captures: [['sample', '1.0.0']], catalog: { bytes: encode(catalogCompressed), size: catalogCompressed.length, sha256: await sha256(catalogCompressed), decoded_size: catalogText.length },
  resources: [{ descriptor: pack, bytes: encode(packed) }, { descriptor: recent.recipe_index, bytes: encode(indexCompressed) }] };
const element = value => ({ textContent: JSON.stringify(value), remove() { this.removed = true; } });
payload.resources_sha256 = await sha256(new TextEncoder().encode(JSON.stringify(payload.resources.map(({descriptor}) => [descriptor.path, descriptor.size, descriptor.sha256])) + '\n'));
let embeddedRequests = 0;
const embeddedElement = element(payload);
const embeddedStore = await loadStartupCatalog(base, new URLSearchParams(), { embedded: embeddedElement, fetchImpl: async () => { embeddedRequests++; throw new Error('unexpected HTTP'); } });
assert.equal(embeddedElement.removed, true);
assert.deepEqual(await embeddedStore.store.read(recent), packBytes);
assert.equal(embeddedRequests, 0, 'catalog, index and capture must all be served from HTML');
assert.equal(embeddedStore.store.transport.packs.size, 1, 'normal transport validates and decodes on demand');
await assert.rejects(embeddedStore.store.read({ ...recent, recipe_index: { ...recent.recipe_index, path: 'older.gz', sha256: 'f'.repeat(64) } }), /standalone/);
assert.equal(embeddedRequests, 0, 'external reads stay disabled until history discovery succeeds');
const historyCatalog = structuredClone(embeddedCatalog);
delete historyCatalog.runtime_sha256;
historyCatalog.revision = 'b'.repeat(64);
historyCatalog.capture_generation.mode = 'history';
historyCatalog.runtime_sha256 = await sha256(new TextEncoder().encode(JSON.stringify(historyCatalog) + '\n'));
const historyPublication = { schema: 1, generation_mode: 'history', catalog_revision: embeddedCatalog.revision,
  resolved_catalog_revision: historyCatalog.revision };
const historyFetch = async url => url.pathname.endsWith('viewer-publication.json')
  ? Response.json(historyPublication) : responseFor(historyCatalog);
assert.deepEqual(await discoverHistory(base, embeddedCatalog, { fetchImpl: historyFetch }), historyCatalog);
embeddedStore.store.allowExternalResources = Boolean(await discoverHistory(base, embeddedCatalog, { fetchImpl: historyFetch }));
await assert.rejects(embeddedStore.store.read({ ...recent, recipe_index: { ...recent.recipe_index, path: 'older.gz', sha256: 'f'.repeat(64) } }), /unexpected HTTP/);
assert.equal(embeddedRequests, 1, 'older capture still fetches its recipe');
for (const mutate of [
  value => { value.schema++; },
  value => { value.resources = []; },
  value => { value.resources.pop(); },
  value => { value.catalog.sha256 = '0'.repeat(64); },
  value => { value.catalog.decoded_size++; },
  value => { value.resources[0].descriptor.sha256 = '0'.repeat(64); },
  value => { value.resources[0].descriptor.path = '../unsafe'; },
  value => { value.resources[0].bytes = ''; },
  value => { value.resources[0].descriptor.size = 256 * 1024 * 1024 + 1; },
]) {
  const corrupt = structuredClone(payload); mutate(corrupt);
  await assert.rejects(loadStartupCatalog(base, new URLSearchParams(), { embedded: element(corrupt), fetchImpl: async () => { throw new Error('must not hide corrupt embedded data with HTTP'); } }));
}
console.log('embedded startup, lazy normal transport, old capture fetch and fail-closed integrity passed');
for (const response of [new Response(null, { status: 404 }), new Response(null, { status: 503 }), Response.json({}),
  Response.json({ schema: 1, catalog_revision: 'wrong' }), new Response('not JSON')]) {
  assert.equal(await discoverHistory(base, embeddedCatalog, { fetchImpl: async () => response }), false);
}
assert.equal(await discoverHistory(base, embeddedCatalog, { fetchImpl: async () => { throw new TypeError('network unavailable'); } }), false);
await assert.rejects(discoverHistory(base, embeddedCatalog, { fetchImpl: async () => { throw new Error('programming error'); } }), /programming error/);
const fileBase = new URL('file:///isolated/');
const noFileFetch = async () => { throw new Error('file mode attempted external fetch'); };
assert.equal(await discoverHistory(fileBase, embeddedCatalog, { fetchImpl: noFileFetch }), false);
await assert.rejects(loadStartupCatalog(fileBase, new URLSearchParams(), { fetchImpl: noFileFetch }), /must contain embedded/);
const fileStartup = await loadStartupCatalog(fileBase, new URLSearchParams(), { embedded: element(payload), fetchImpl: noFileFetch });
assert.deepEqual(await fileStartup.store.read(recent), packBytes);
assert.deepEqual(fileStartup.embeddedCaptures, payload.captures);
for (const mutate of [
  value => { value.revision = 'c'.repeat(64); },
  value => { value.capture_generation.current = []; },
  value => { value.capture_generation.mode = 'latest'; },
]) {
  const stale = structuredClone(historyCatalog); mutate(stale); delete stale.runtime_sha256;
  stale.runtime_sha256 = await sha256(new TextEncoder().encode(JSON.stringify(stale) + '\n'));
  await assert.rejects(discoverHistory(base, embeddedCatalog, { fetchImpl: async url =>
    url.pathname.endsWith('viewer-publication.json') ? Response.json(historyPublication) : responseFor(stale) }), /does not match/);
}
console.log('file startup makes no external reads; HTTP history requires a matching publication manifest');
for (const mutate of [
  value => { value.resources[0].descriptor.path = '../unsafe'; },
  value => { value.resources[0].descriptor.sha256 = '0'.repeat(64); },
  value => { value.resources[0].descriptor.size++; },
  value => { value.resources[0].descriptor.size = 256 * 1024 * 1024 + 1; },
]) {
  const corrupt = structuredClone(payload); mutate(corrupt);
  corrupt.resources_sha256 = await sha256(new TextEncoder().encode(JSON.stringify(corrupt.resources.map(({descriptor}) => [descriptor.path, descriptor.size, descriptor.sha256])) + '\n'));
  await assert.rejects(loadStartupCatalog(base, new URLSearchParams(), { embedded: element(corrupt), fetchImpl: async () => { throw new Error('unexpected HTTP'); } }), /unsafe|integrity mismatch|Invalid embedded|memory limit/);
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

await test('selection contracts', async () => {
const savedGlobals = Object.getOwnPropertyDescriptors(globalThis);
try {
const { default: assert } = await import('node:assert/strict');
const { SelectionLoader } = await import('./app.js');


const pending = [], paints = [], states = [];
const loader = new SelectionLoader({ build: (tab, selection) => new Promise((resolve, reject) => pending.push({ tab, selection, resolve, reject })), paint: (tab, model) => paints.push([tab, model]), state: (...args) => states.push(args) });
const old = loader.select('tab-a', { reference: 'old', selected: ['old'] });
const newer = loader.select('tab-a', { reference: 'new', selected: ['new', 'peer'] });
pending[1].resolve('new-model'); await newer;
pending[0].resolve('old-model'); await old;
assert.deepEqual(paints, [['tab-a', 'new-model']]);
const abandoned = loader.select('tab-a', { selected: ['slow'] });
const active = loader.select('tab-b', { selected: [] });
pending[3].resolve('empty-reference-model'); await active;
pending[2].reject(new Error('old load failed')); await abandoned;
assert.equal(states.filter(([, state]) => state === 'error').length, 0);
assert.deepEqual(paints.at(-1), ['tab-b', 'empty-reference-model']);
const failed = loader.select('tab-b', { selected: ['missing'] });
pending[4].reject(new Error('transport')); await failed;
assert.equal(states.at(-1)[1], 'error');
const retry = loader.select('tab-b', { selected: ['missing'] });
pending[5].resolve('recovered'); await retry;
assert.deepEqual(paints.at(-1), ['tab-b', 'recovered']);
const obsolete = loader.select('tab-b', { selected: ['obsolete'] });
const latestSelection = { reference: null, selected: ['latest'] };
const latest = loader.select('tab-b', latestSelection);
const loadingState = states.at(-1);
assert.equal(loadingState[1], 'loading');
assert.equal(loadingState[3], latestSelection);
pending[6].reject(new Error('obsolete failure during current load')); await obsolete;
assert.equal(states.at(-1), loadingState, 'stale failures must not replace the current loading notice');
assert.deepEqual(paints.at(-1), ['tab-b', 'recovered'], 'the displayed model remains until replacement succeeds');
pending[7].resolve('latest-model'); await latest;
assert.deepEqual(paints.at(-1), ['tab-b', 'latest-model']);
console.log('Selection generations reject stale paints and failures; empty Reference and Retry recover');

} finally {
  for (const key of Object.getOwnPropertyNames(globalThis)) {
    if (!Object.hasOwn(savedGlobals, key) && Object.getOwnPropertyDescriptor(globalThis, key).configurable) delete globalThis[key];
  }
  for (const [key, descriptor] of Object.entries(savedGlobals)) {
    if (descriptor.configurable) Object.defineProperty(globalThis, key, descriptor);
  }
}

});
