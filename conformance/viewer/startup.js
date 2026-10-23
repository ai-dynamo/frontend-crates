// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { HttpResourceStore, gunzipSync, sha256, resourceMetrics } from './data.js';
import { boundedGunzip } from './transport.js';
const decoder = new TextDecoder('utf-8', { fatal: true });
let developmentFallback = null;
export function configureDevelopmentCatalog(loader) { developmentFallback = loader; }

export const startupMetrics = { phases: {}, resources: resourceMetrics };

export async function timedStartupPhase(name, operation) {
  const start = performance.now();
  try { return await operation(); }
  finally { startupMetrics.phases[name] = (startupMetrics.phases[name] || 0) + performance.now() - start; }
}

export function validateStartupCatalog(catalog, { resolved = true } = {}) {
  if (!catalog || catalog.schema !== 1 || !Array.isArray(catalog.sources?.archives)) throw new Error('Unsupported conformance catalog');
  if (!Array.isArray(catalog.sources.unified) || catalog.sources.unified.some(resource =>
    resource.format !== 'typed-json-v1' && (resolved || resource.format !== undefined))) {
    throw new Error(resolved ? 'Browser data needs regeneration' : 'Unsupported conformance resource format');
  }
  return catalog;
}

async function readCatalog(text) {
  const catalog = validateStartupCatalog(JSON.parse(text));
  if (catalog.transport?.schema === 2) {
    const trailer = /,"runtime_sha256":"([a-f0-9]{64})"}\n$/.exec(text);
    const content = trailer ? text.slice(0, trailer.index) + '}\n' : '';
    if (!trailer || trailer[1] !== catalog.runtime_sha256 || await sha256(new TextEncoder().encode(content)) !== trailer[1]) throw new Error('Runtime catalog integrity mismatch');
  }
  return catalog;
}

function base64Bytes(text, size) {
  if (typeof text !== 'string' || !Number.isSafeInteger(size) || size < 0 || size > 64 * 1024 * 1024 || text.length !== 4 * Math.ceil(size / 3)) throw new Error('Invalid embedded resource length');
  const bytes = Uint8Array.from(atob(text), char => char.charCodeAt(0));
  if (bytes.length !== size) throw new Error('Invalid embedded resource length');
  return bytes;
}

export async function loadStartupCatalog(baseUrl, params, { fetchImpl = globalThis.fetch.bind(globalThis), embedded = globalThis.document?.getElementById('conformance-embedded') } = {}) {
  const store = new HttpResourceStore({ baseUrl, fetchImpl });
  const mode = params.get('startup') || 'compressed';
  startupMetrics.mode = mode;
  if (embedded) {
    const payload = JSON.parse(embedded.textContent);
    if (payload.schema !== 1 || !Array.isArray(payload.resources) || payload.resources.length > 4096) throw new Error('Unsupported embedded data');
    if (!Array.isArray(payload.captures) || payload.captures.some(capture => !Array.isArray(capture) || capture.length !== 2 || capture.some(value => typeof value !== 'string' || !value))) throw new Error('Invalid embedded capture identities');
    const inventory = payload.resources.map(({descriptor}) => [descriptor.path, descriptor.size, descriptor.sha256]);
    if (await sha256(new TextEncoder().encode(JSON.stringify(inventory) + '\n')) !== payload.resources_sha256) throw new Error('Embedded resource membership mismatch');
    const descriptor = payload.catalog;
    const bytes = base64Bytes(descriptor.bytes, descriptor.size);
    if (await sha256(bytes) !== descriptor.sha256 || !Number.isSafeInteger(descriptor.decoded_size) || descriptor.decoded_size > 64 * 1024 * 1024) throw new Error('Embedded catalog integrity mismatch');
    const catalog = await readCatalog(decoder.decode(boundedGunzip(bytes, descriptor.decoded_size)));
    if (catalog.transport?.schema !== 2) throw new Error('Unsupported embedded transport');
    let total = 0;
    for (const resource of payload.resources) {
      total += resource.descriptor.size;
      if (!Number.isSafeInteger(total) || total > 256 * 1024 * 1024) throw new Error('Embedded resources exceed memory limit');
      await store.seed(resource.descriptor, base64Bytes(resource.bytes, resource.descriptor.size));
    }
    // Release the DOM copy; only compressed bytes are retained. Envelope JSON
    // remains lazy and follows the normal recipe, hash and worker validation.
    embedded.remove?.();
    store.allowExternalResources = false;
    return { catalog, store, embeddedCaptures: payload.captures };
  }
  if ((globalThis.location?.protocol || baseUrl.protocol) === 'file:') throw new Error('Standalone HTML must contain embedded results');
  const response = await fetchImpl(new URL('conformance/viewer-resolved-catalog.json.gz', baseUrl), { cache: 'no-store' });
  if (response.ok) {
    const text = decoder.decode(gunzipSync(new Uint8Array(await response.arrayBuffer())));
    const catalog = await readCatalog(text);
    store.allowSourceFiles = catalog.transport?.schema !== 2 && Boolean(developmentFallback);
    return { catalog, store };
  }
  if (response.status !== 404 || !developmentFallback) throw new Error(`Catalog fetch failed (${response.status})`);
  store.allowSourceFiles = true;
  return { catalog: validateStartupCatalog(await developmentFallback(baseUrl, fetchImpl), { resolved: false }), store };
}

export async function discoverHistory(baseUrl, catalog, { fetchImpl = globalThis.fetch.bind(globalThis) } = {}) {
  if (!['http:', 'https:'].includes(globalThis.location?.protocol || baseUrl.protocol)) return false;
  let response;
  try { response = await fetchImpl(new URL('conformance/viewer-publication.json', baseUrl), { cache: 'no-store' }); }
  catch (error) {
    if (error instanceof TypeError) return false;
    throw error;
  }
  if (!response.ok) return false;
  let publication;
  try { publication = await response.json(); }
  catch (error) {
    if (error instanceof SyntaxError || error instanceof TypeError) return false;
    throw error;
  }
  return publication?.schema === 1 && publication.catalog_revision === catalog.revision;
}
