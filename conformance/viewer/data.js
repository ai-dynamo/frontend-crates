// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { gunzipSync } from './vendor/fflate/esm/browser.js';
import { ChunkTransport, boundedGunzip } from './transport.js';

export const resourceMetrics = { decoding: 0, validation: 0 };

const CACHE_DB = 'frontend-crates-conformance-v1';
const CACHE_STORE = 'resources';
import { DownloadScheduler, sharedRead } from './download_scheduler.js';

export function bytesOf(value) {
  if (value instanceof Uint8Array) return value;
  if (value instanceof ArrayBuffer) return new Uint8Array(value);
  if (ArrayBuffer.isView(value)) return new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
  throw new TypeError('resource bytes must be an ArrayBuffer or Uint8Array');
}

function sha256Software(bytes) {
  const K = new Uint32Array([
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
  ]);
  const H = new Uint32Array([0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]);
  const bitLength = bytes.length * 8;
  const paddedLength = Math.ceil((bytes.length + 9) / 64) * 64;
  const block = new Uint8Array(paddedLength);
  block.set(bytes);
  block[bytes.length] = 0x80;
  const view = new DataView(block.buffer);
  view.setUint32(paddedLength - 8, Math.floor(bitLength / 0x100000000));
  view.setUint32(paddedLength - 4, bitLength >>> 0);
  const w = new Uint32Array(64);
  const rotr = (x, n) => (x >>> n) | (x << (32 - n));
  for (let offset = 0; offset < paddedLength; offset += 64) {
    for (let i = 0; i < 16; i++) w[i] = view.getUint32(offset + i * 4);
    for (let i = 16; i < 64; i++) {
      const x = w[i - 15], y = w[i - 2];
      const s0 = rotr(x, 7) ^ rotr(x, 18) ^ (x >>> 3);
      const s1 = rotr(y, 17) ^ rotr(y, 19) ^ (y >>> 10);
      w[i] = (w[i - 16] + s0 + w[i - 7] + s1) >>> 0;
    }
    let [a, b, c, d, e, f, g, h] = H;
    for (let i = 0; i < 64; i++) {
      const s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      const ch = (e & f) ^ (~e & g);
      const t1 = (h + s1 + ch + K[i] + w[i]) >>> 0;
      const s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      const maj = (a & b) ^ (a & c) ^ (b & c);
      const t2 = (s0 + maj) >>> 0;
      [h, g, f, e, d, c, b, a] = [g, f, e, (d + t1) >>> 0, c, b, a, (t1 + t2) >>> 0];
    }
    H[0] = (H[0] + a) >>> 0; H[1] = (H[1] + b) >>> 0;
    H[2] = (H[2] + c) >>> 0; H[3] = (H[3] + d) >>> 0;
    H[4] = (H[4] + e) >>> 0; H[5] = (H[5] + f) >>> 0;
    H[6] = (H[6] + g) >>> 0; H[7] = (H[7] + h) >>> 0;
  }
  return [...H].map((x) => x.toString(16).padStart(8, '0')).join('');
}

export async function sha256(bytes) {
  const data = bytesOf(bytes);
  if (globalThis.crypto?.subtle) {
    const digest = await globalThis.crypto.subtle.digest('SHA-256', data);
    return [...new Uint8Array(digest)].map((x) => x.toString(16).padStart(2, '0')).join('');
  }
  return sha256Software(data);
}

function openCache() {
  if (!globalThis.indexedDB) return Promise.resolve(null);
  return new Promise((resolve) => {
    const request = indexedDB.open(CACHE_DB, 1);
    request.onupgradeneeded = () => request.result.createObjectStore(CACHE_STORE, { keyPath: 'sha256' });
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => resolve(null);
    request.onblocked = () => resolve(null);
  });
}

async function readCachedBytes(hash) {
  const db = await openCache();
  if (!db) return null;
  return new Promise((resolve) => {
    const request = db.transaction(CACHE_STORE).objectStore(CACHE_STORE).get(hash);
    request.onsuccess = () => { db.close(); resolve(request.result?.bytes ? new Uint8Array(request.result.bytes) : null); };
    request.onerror = () => { db.close(); resolve(null); };
  });
}

async function writeCachedBytes(hash, bytes) {
  const db = await openCache();
  if (!db) return;
  await new Promise((resolve) => {
    const tx = db.transaction(CACHE_STORE, 'readwrite');
    tx.objectStore(CACHE_STORE).put({ sha256: hash, bytes: bytes.slice().buffer });
    tx.oncomplete = tx.onerror = tx.onabort = () => { db.close(); resolve(); };
  });
}

// Persistent storage is optional on plain HTTP, private browsing, and restricted
// profiles. Only storage failures are ignored; transport/hash errors propagate.
async function cachedBytes(hash) { return readCachedBytes(hash).catch(() => null); }
async function storeBytes(hash, bytes) { return writeCachedBytes(hash, bytes).catch(() => undefined); }

function checkLfsPointer(bytes, path) {
  const head = new TextDecoder().decode(bytes.subarray(0, Math.min(bytes.length, 128)));
  if (head.startsWith('version https://git-lfs.github.com/spec/v1\n')) {
    throw new Error(`Git LFS object is not hydrated: ${path}`);
  }
}

function resourceUrl(path, base) {
  assertSafeMember(path);
  if (path.includes(':')) throw new Error(`resource path must be repository-relative: ${path}`);
  return new URL(path.split('/').map(encodeURIComponent).join('/'), base);
}

export function validateResource(resource) {
  assertSafeMember(resource.path);
  if (resource.path.includes(':')) throw new Error(`resource path must be repository-relative: ${resource.path}`);
  if (!/^[0-9a-f]{64}$/.test(resource.sha256) || !Number.isSafeInteger(resource.size) || resource.size < 0) {
    throw new Error(`invalid resource hash or size: ${resource.path}`);
  }
}

export class HttpResourceStore {
  constructor({ baseUrl = new URL('.', document.baseURI), fetchImpl = globalThis.fetch.bind(globalThis), allowSourceFiles = false, retainBytes = true } = {}) {
    this.baseUrl = baseUrl instanceof URL ? baseUrl : new URL(baseUrl, globalThis.document?.baseURI);
    this.fetchImpl = fetchImpl;
    this.allowExternalResources = true;
    this.memory = new Map();
    this.retainBytes = retainBytes;
    this.scheduler = new DownloadScheduler();
    this.allowSourceFiles = allowSourceFiles;
    this.downloads = new Map();
    this.selectionRevision = 0;
    this.reservations = new Map();
    this.transport = new ChunkTransport({ readBytes: (resource, signal) => this.readTransportBytes(resource, signal), sha256, validateResource });
  }

  key(resource) { return `${resource.sha256}:${resource.size}`; }
  async seed(resource, bytes) {
    validateResource(resource);
    if (bytes.byteLength !== resource.size || await sha256(bytes) !== resource.sha256) throw new Error(`Embedded resource integrity mismatch: ${resource.path}`);
    this.memory.set(resource.sha256, bytes);
  }
  taskKey(resource) { return JSON.stringify([this.key(resource), resource.path, resource.transport, resource.recipe, resource.recipe_index, resource.archive]); }
  demand(resources) {
    const revision = ++this.selectionRevision;
    const transportKeys = new Set();
    for (const resource of resources) {
      validateResource(resource);
      transportKeys.add(this.key(resource.archive || resource));
      if (resource.recipe_index) transportKeys.add(this.key(resource.recipe_index));
      const recipe = this.transport.knownRecipe(resource);
      if (recipe) for (const chunk of this.transport.validate(resource, recipe).chunks) transportKeys.add(this.key(chunk.pack));
    }
    // Reserve only existing transport, never wait for a cold recipe index before
    // canceling obsolete downloads: those downloads can own every HTTP slot.
    for (const key of transportKeys) if (this.downloads.has(key) && !this.reservations.has(key)) {
      const controller = new AbortController();
      this.reservations.set(key, controller);
      sharedRead(this.downloads, key, () => { throw new Error('missing reserved transport'); }, controller.signal).catch(() => {});
    }
    for (const [key, controller] of this.reservations) if (!transportKeys.has(key)) {
      this.reservations.delete(key);
      controller.abort();
    }
    this.scheduler.demand(resources.map(resource => this.taskKey(resource)));
    return revision;
  }
  releaseReservations(revision) {
    if (revision !== this.selectionRevision) return;
    for (const controller of this.reservations.values()) controller.abort();
    this.reservations.clear();
  }
  async admit(resource, signal) {
    validateResource(resource);
    if (resource.transport === 'chunked-v2' || resource.recipe || resource.recipe_index) {
      this.transport.validate(resource, await this.transport.recipe(resource, signal));
      return null;
    }
    if (!this.allowSourceFiles) throw new Error(`Source files disabled in published site: ${resource.path}`);
    return null;
  }
  async readTransportBytes(resource, signal) {
    validateResource(resource);
    const key = this.key(resource);
    const pending = sharedRead(this.downloads, key, sharedSignal => this._readTransportBytes(resource, sharedSignal), signal, {retain: false});
    this.reservations.get(key)?.abort();
    this.reservations.delete(key);
    const bytes = await pending;
    return bytes;
  }
  async _readTransportBytes(resource, signal) {
    let bytes = this.memory.get(resource.sha256) || await cachedBytes(resource.sha256);
    if (!bytes || bytes.byteLength !== resource.size || await sha256(bytes) !== resource.sha256) {
      bytes = await this.fetchValidated(resource, signal);
      await storeBytes(resource.sha256, bytes);
    }
    if (this.retainBytes) this.memory.set(resource.sha256, bytes);
    return bytes;
  }
  async readRecipe(resource, recipe) {
    return this.read({ ...resource, transport: 'chunked-v2', recipe });
  }
  async read(resource, {selection = false} = {}) {
    if (selection && selection !== true && selection !== this.selectionRevision) throw new Error('Obsolete fixture selection');
    return this.scheduler.read(this.taskKey(resource), signal => this._read(resource, signal), {selection}).then(bytes => bytes.slice());
  }

  async _read(resource, signal) {
    await this.admit(resource, signal);
    signal.throwIfAborted();
    let bytes = this.memory.get(resource.sha256) || await cachedBytes(resource.sha256);
    if (bytes && bytes.byteLength === resource.size && await sha256(bytes) === resource.sha256) {
      if (this.retainBytes) this.memory.set(resource.sha256, bytes);
      return bytes;
    }
    const chunked = resource.transport === 'chunked-v2' || resource.recipe || resource.recipe_index;
    bytes = chunked ? await this.transport.read(resource, undefined, signal)
      : resource.archive ? await (await developmentParser()).readSourceMember(this, resource, signal) : await this.readTransportBytes(resource, signal);
    checkLfsPointer(bytes, resource.path);
    if (bytes.byteLength !== resource.size || await sha256(bytes) !== resource.sha256) throw new Error(`original resource integrity mismatch: ${resource.path}`);
    if (this.retainBytes) this.memory.set(resource.sha256, bytes);
    await storeBytes(resource.sha256, bytes);
    return bytes;
  }


  async fetchValidated(resource, signal) {
    if (!this.allowExternalResources) throw new Error('This standalone report contains only embedded latest results');
    validateResource(resource);
    const url = resourceUrl(resource.path, this.baseUrl);
    url.searchParams.set('sha256', resource.sha256);
    const response = await this.fetchImpl(url, { cache: 'no-store', signal });
    if (!response.ok) throw new Error(`fetch failed (${response.status}) for ${resource.path}`);
    const bytes = new Uint8Array(await response.arrayBuffer());
    checkLfsPointer(bytes, resource.path);
    if (bytes.byteLength !== resource.size) throw new Error(`size mismatch for ${resource.path}`);
    const digest = await sha256(bytes);
    if (digest !== resource.sha256) throw new Error(`SHA-256 mismatch for ${resource.path}`);
    return bytes;
  }
}

export async function createNodeResourceStore(root, { allowSourceFiles = false } = {}) {
  const [{ readFile }, { resolve, sep }, { pathToFileURL }] = await Promise.all([
    import('node:fs/promises'), import('node:path'), import('node:url'),
  ]);
  const base = root instanceof URL ? root : pathToFileURL(resolve(root) + sep);
  // Explicit checkout readers share validation with the published transport.
  return new HttpResourceStore({ baseUrl: base, allowSourceFiles, retainBytes: false, fetchImpl: async (url, {signal} = {}) => {
    try {
      return new Response(await readFile(url, {signal}));
    } catch (error) {
      if (error.code === 'ENOENT') return new Response(null, { status: 404 });
      throw error;
    }
  } });
}

function failIfLfs(bytes, path) {
  checkLfsPointer(bytes, path);
}

function assertSafeMember(path) {
  if (!path || path.startsWith('/') || path.includes('\\') || path.split('/').some((part) => !part || part === '.' || part === '..')) {
    throw new Error(`unsafe tar member path: ${JSON.stringify(path)}`);
  }
}

function parseOctal(bytes, start, length, field) {
  const raw = new TextDecoder().decode(bytes.subarray(start, start + length)).replace(/\0.*$/s, '').trim();
  if (!raw) return 0;
  if (!/^[0-7]+$/.test(raw)) throw new Error(`invalid tar ${field}`);
  const value = Number.parseInt(raw, 8);
  if (!Number.isSafeInteger(value)) throw new Error(`tar ${field} exceeds safe integer range`);
  return value;
}

function textField(bytes, start, length) {
  const field = bytes.subarray(start, start + length);
  const end = field.indexOf(0);
  return new TextDecoder('utf-8', { fatal: true }).decode(end < 0 ? field : field.subarray(0, end));
}

function parsePax(bytes) {
  const out = {};
  let offset = 0;
  while (offset < bytes.length) {
    let space = offset;
    while (space < bytes.length && bytes[space] !== 0x20) space++;
    if (space === bytes.length) throw new Error('malformed PAX record length');
    const lengthText = new TextDecoder().decode(bytes.subarray(offset, space));
    if (!/^\d+$/.test(lengthText)) throw new Error('invalid PAX record length');
    const length = Number(lengthText);
    if (length <= space - offset + 2 || offset + length > bytes.length || bytes[offset + length - 1] !== 0x0a) {
      throw new Error('truncated PAX record');
    }
    const record = new TextDecoder('utf-8', { fatal: true }).decode(bytes.subarray(space + 1, offset + length - 1));
    const split = record.indexOf('=');
    if (split < 1) throw new Error('malformed PAX key/value');
    out[record.slice(0, split)] = record.slice(split + 1);
    offset += length;
  }
  return out;
}

export function readTarGzip(input, { selectedPaths = null, path = '<archive>', maximumDecoded = null } = {}) {
  const compressed = bytesOf(input);
  failIfLfs(compressed, path);
  if (maximumDecoded !== null && (compressed.length < 18
      || new DataView(compressed.buffer, compressed.byteOffset).getUint32(compressed.length - 4, true) > maximumDecoded)) {
    throw new Error('source archive exceeds decoded limit');
  }
  const tar = maximumDecoded === null ? gunzipSync(compressed)
    : boundedGunzip(compressed, new DataView(compressed.buffer, compressed.byteOffset).getUint32(compressed.length - 4, true));
  const found = new Map();
  const seen = new Set();
  const wanted = selectedPaths ? new Set(selectedPaths) : null;
  let offset = 0;
  let paxGlobal = {};
  let paxNext = {};
  let longName = null;
  let zeroBlocks = 0;
  let extensionBytes = 0;
  let extensionHeaders = 0;
  while (offset < tar.length) {
    if (offset + 512 > tar.length) throw new Error(`truncated tar header in ${path}`);
    const header = tar.subarray(offset, offset + 512);
    if (header.every((byte) => byte === 0)) {
      zeroBlocks++;
      offset += 512;
      if (zeroBlocks === 2) break;
      continue;
    }
    if (zeroBlocks) throw new Error(`nonzero tar data after end marker in ${path}`);
    const storedSum = parseOctal(header, 148, 8, 'checksum');
    let actualSum = 0;
    for (let i = 0; i < 512; i++) actualSum += (i >= 148 && i < 156) ? 32 : header[i];
    if (actualSum !== storedSum) throw new Error(`tar checksum mismatch in ${path}`);
    const headerName = textField(header, 0, 100);
    const prefix = textField(header, 345, 155);
    const type = header[156] || 48;
    const size = parseOctal(header, 124, 12, 'size');
    if ([120, 103, 76, 75].includes(type)) {
      extensionBytes += size;
      extensionHeaders++;
      if (extensionBytes > 64 * 1024 || extensionHeaders > 128) throw new Error(`tar extension metadata exceeds limit in ${path}`);
    } else {
      extensionBytes = 0;
      extensionHeaders = 0;
    }
    const dataStart = offset + 512;
    const dataEnd = dataStart + size;
    if (!Number.isSafeInteger(dataEnd) || dataEnd > tar.length) throw new Error(`truncated tar member in ${path}`);
    const payload = tar.subarray(dataStart, dataEnd);
    const paddedEnd = dataStart + Math.ceil(size / 512) * 512;
    if (paddedEnd > tar.length) throw new Error(`truncated tar padding in ${path}`);
    offset = paddedEnd;
    if (type === 120 || type === 103) { // PAX local/global headers
      const parsed = parsePax(payload);
      const pax = Object.fromEntries(['path', 'size', 'linkpath'].filter(key => Object.hasOwn(parsed, key)).map(key => [key, parsed[key]]));
      if (type === 103) paxGlobal = { ...paxGlobal, ...pax };
      else paxNext = { ...paxNext, ...pax };
      continue;
    }
    if (type === 76 || type === 75) { // GNU long name/link name
      const longValue = new TextDecoder('utf-8', { fatal: true }).decode(payload).replace(/\0.*$/s, '').replace(/\n$/, '');
      if (type === 76) longName = longValue;
      else throw new Error(`tar links are not supported in ${path}`);
      continue;
    }
    if (paxNext.linkpath || paxGlobal.linkpath) throw new Error(`tar links are not supported in ${path}`);
    const paxSize = paxNext.size ?? paxGlobal.size;
    if (paxSize !== undefined && (!/^\d+$/.test(paxSize) || Number(paxSize) !== size)) {
      throw new Error(`PAX member size differs from tar header in ${path}`);
    }
    const member = longName || paxNext.path || paxGlobal.path || (prefix ? `${prefix}/${headerName}` : headerName);
    longName = null;
    paxNext = {};
    if (type === 53) { // directory
      const normalized = member.endsWith('/') ? member.slice(0, -1) : member;
      assertSafeMember(normalized);
      if (seen.has(normalized)) throw new Error(`duplicate tar member ${normalized} in ${path}`);
      seen.add(normalized);
      continue;
    }
    if (type !== 48 && type !== 0 && type !== 55) throw new Error(`unsupported tar member type ${type} in ${path}`);
    assertSafeMember(member);
    if (seen.has(member)) throw new Error(`duplicate tar member ${member} in ${path}`);
    seen.add(member);
    if (!wanted || wanted.has(member)) found.set(member, payload.slice());
  }
  if (zeroBlocks !== 2) throw new Error(`missing tar end marker in ${path}`);
  if (tar.subarray(offset).some((byte) => byte !== 0)) throw new Error(`nonzero data after tar end marker in ${path}`);
  for (const member of wanted || []) {
    if (!found.has(member)) throw new Error(`archive member missing: ${member} in ${path}`);
  }
  return found;
}

let developmentParser = null;
let developmentWorker = null;
export function configureDevelopmentParser(loader, workerUrl) { developmentParser = loader; developmentWorker = workerUrl; }

export function createWorkerParser() {
  if (typeof Worker === 'undefined') {
    return {
      parseArchive: async (bytes, opts) => readTarGzip(bytes, opts),
      parseSnapshot: async (bytes, opts) => (await import('./snapshots.js')).parseSnapshot(bytes, opts),
      parseYaml: async (bytes, opts) => (await developmentParser()).parseYamlBytes(bytes, opts),
      parseArchiveCases: async (bytes, opts) => (await developmentParser()).readArchiveCases(bytes, opts),
    };
  }
  let worker = null;
  let workerObjectUrl = null;
  let disposed = false;
  let sequence = 0;
  const pending = new Map();
  const failWorker = (instance, error) => {
    for (const [id, waiter] of pending) if (waiter.worker === instance) { pending.delete(id); waiter.reject(error); }
    instance.terminate();
    if (worker === instance) {
      worker = null;
      if (workerObjectUrl) URL.revokeObjectURL(workerObjectUrl);
      workerObjectUrl = null;
    }
  };
  const getWorker = () => {
    if (disposed) throw new Error('conformance parser is disposed');
    if (worker) return worker;
    const source = typeof __CONFORMANCE_WORKER_SOURCE__ === 'string' ? __CONFORMANCE_WORKER_SOURCE__ : null;
    const objectUrl = source ? URL.createObjectURL(new Blob([source], { type: 'text/javascript' })) : null;
    let instance;
    try { instance = new Worker(objectUrl || developmentWorker || new URL('./worker.js', import.meta.url), { type: objectUrl ? 'classic' : 'module' }); }
    catch (error) { if (objectUrl) URL.revokeObjectURL(objectUrl); throw error; }
    workerObjectUrl = objectUrl;
    instance.addEventListener('message', (event) => {
      const waiter = pending.get(event.data.id);
      if (!waiter || waiter.worker !== instance) return;
      pending.delete(event.data.id);
      if (event.data.error) waiter.reject(new Error(event.data.error));
      else {
        for (const [key, value] of Object.entries(event.data.metrics || {})) resourceMetrics[key] = (resourceMetrics[key] || 0) + value;
        waiter.resolve(event.data.result);
      }
    });
    instance.addEventListener('error', (event) => {
      event.preventDefault?.();
      failWorker(instance, new Error(event.message || 'conformance worker failed'));
    });
    worker = instance;
    return worker;
  };
  const send = (operation, bytes, options) => new Promise((resolve, reject) => {
    const instance = getWorker();
    const id = ++sequence;
    const copy = bytesOf(bytes).slice();
    pending.set(id, { resolve, reject, worker: instance });
    try { instance.postMessage({ id, operation, bytes: copy.buffer, options }, [copy.buffer]); }
    catch (error) { pending.delete(id); reject(error); }
  });
  return {
    parseArchive: (bytes, options) => send('archive', bytes, options),
    parseSnapshot: (bytes, options) => send('snapshot', bytes, options),
    parseYaml: (bytes, options) => send('yaml', bytes, options),
    parseArchiveCases: (bytes, options) => send('archiveCases', bytes, options),
    terminate: () => { disposed = true; if (worker) failWorker(worker, new Error('conformance parser is disposed')); },
  };
}

export { gunzipSync };
