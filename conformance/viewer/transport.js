// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { sharedRead } from './download_scheduler.js';
import { gunzipSync } from './vendor/fflate/esm/browser.js';

export const MAX_PACK_SIZE = 1024 * 1024;
const MAX_RESOURCE_SIZE = 1024 * 1024 * 1024;
const hashPattern = /^[0-9a-f]{64}$/;

function identity(value) {
  return value && hashPattern.test(value.sha256) && Number.isSafeInteger(value.size) && value.size >= 0;
}

export function boundedGunzip(bytes, size) {
  if (!Number.isSafeInteger(size) || size < 0 || bytes.length < 18) throw new Error('invalid compressed transport length');
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (view.getUint32(bytes.length - 4, true) !== size) throw new Error('compressed transport decoded length mismatch');
  // Supply the output allocation: an untrusted gzip header cannot allocate memory.
  const decoded = gunzipSync(bytes, { out: new Uint8Array(size + 1) });
  if (decoded.length !== size) throw new Error('transport decoded length mismatch');
  return decoded;
}

export class ChunkTransport {
  constructor({ readBytes, sha256, validateResource, maxDecodedPackBytes = 64 * MAX_PACK_SIZE }) {
    if (!Number.isSafeInteger(maxDecodedPackBytes) || maxDecodedPackBytes < MAX_PACK_SIZE) throw new Error('invalid decoded pack cache limit');
    this.readBytes = readBytes;
    this.sha256 = sha256;
    this.validateResource = validateResource;
    this.packs = new Map();
    this.packSizes = new Map();
    this.decodedPackBytes = 0;
    this.maxDecodedPackBytes = maxDecodedPackBytes;
    this.indexes = new Map();
  }

  rememberPack(key, bytes) {
    this.packSizes.set(key, bytes.byteLength);
    this.decodedPackBytes += bytes.byteLength;
    // Compressed bytes remain in the store cache; eviction only drops decoding.
    while (this.decodedPackBytes > this.maxDecodedPackBytes) {
      const [oldest, size] = this.packSizes.entries().next().value;
      this.packSizes.delete(oldest);
      this.packs.delete(oldest);
      this.decodedPackBytes -= size;
    }
    return bytes;
  }

  shared(map, key, load, signal) {
    return sharedRead(map, key, load, signal);
  }

  indexKey(descriptor) { return `${descriptor.sha256}:${descriptor.size}:${descriptor.decoded_size}:${descriptor.decoded_sha256 || ''}`; }

  knownRecipe(resource) {
    return resource.recipe || (resource.recipe_index && this.indexes.get(this.indexKey(resource.recipe_index))?.value?.resources?.[resource.path]);
  }

  async index(descriptor, signal) {
    this.validateResource(descriptor);
    return this.shared(this.indexes, this.indexKey(descriptor), async sharedSignal => {
      const compressed = await this.readBytes(descriptor, sharedSignal);
      sharedSignal.throwIfAborted();
      // Index size is publisher-declared and bounded independently of gzip ISIZE.
      const limit = descriptor.decoded_size;
      if (!Number.isSafeInteger(limit) || limit < 0 || limit > 64 * 1024 * 1024) throw new Error('invalid recipe index decoded size');
      const bytes = boundedGunzip(compressed, limit);
      if (descriptor.decoded_sha256 && await this.sha256(bytes) !== descriptor.decoded_sha256) throw new Error('recipe index decoded hash mismatch');
      const value = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes));
      if (value.schema !== 2 || !value.resources || typeof value.resources !== 'object' || Array.isArray(value.resources)) throw new Error('invalid recipe index');
      return value;
    }, signal);
  }

  async recipe(resource, signal) {
    if (resource.recipe) return resource.recipe;
    const index = (await this.index(resource.recipe_index, signal)).resources;
    if (!Object.hasOwn(index, resource.path)) throw new Error(`recipe missing for ${resource.path}`);
    return index[resource.path];
  }

  validate(resource, recipe) {
    if (recipe.schema !== 2 || !identity(recipe) || recipe.size >= MAX_RESOURCE_SIZE || recipe.sha256 !== resource.sha256 || recipe.size !== resource.size || !Array.isArray(recipe.chunks)) throw new Error(`recipe identity mismatch: ${resource.path}`);
    let total = 0;
    for (const chunk of recipe.chunks) {
      const pack = chunk.pack;
      if (!identity(chunk) || chunk.size > 128 * 1024 || !Number.isSafeInteger(chunk.offset) || chunk.offset < 0 || !identity(pack) || !hashPattern.test(pack.decoded_sha256) || !Number.isSafeInteger(pack.decoded_size) || pack.decoded_size < 0 || pack.decoded_size > MAX_PACK_SIZE || chunk.offset + chunk.size > pack.decoded_size) throw new Error(`invalid chunk location: ${resource.path}`);
      this.validateResource(pack);
      total += chunk.size;
      if (!Number.isSafeInteger(total) || total > recipe.size) throw new Error(`chunk lengths exceed resource: ${resource.path}`);
    }
    if (total !== recipe.size) throw new Error(`chunk lengths mismatch: ${resource.path}`);
    return recipe;
  }

  async read(resource, suppliedRecipe, signal) {
    const recipe = this.validate(resource, suppliedRecipe || await this.recipe(resource, signal));
    const output = new Uint8Array(recipe.size);
    let offset = 0;
    for (const chunk of recipe.chunks) {
      const pack = chunk.pack;
      const key = `${pack.sha256}:${pack.size}:${pack.decoded_sha256}:${pack.decoded_size}`;
      const bytes = await this.shared(this.packs, key, async sharedSignal => {
        const compressed = await this.readBytes(pack, sharedSignal);
        sharedSignal.throwIfAborted();
        const decoded = boundedGunzip(compressed, pack.decoded_size);
        if (await this.sha256(decoded) !== pack.decoded_sha256) throw new Error(`pack decoded hash mismatch: ${pack.path}`);
        sharedSignal.throwIfAborted();
        return this.rememberPack(key, decoded);
      }, signal);
      if (this.packSizes.delete(key)) this.packSizes.set(key, bytes.byteLength);
      const bytesChunk = bytes.subarray(chunk.offset, chunk.offset + chunk.size);
      if (await this.sha256(bytesChunk) !== chunk.sha256) throw new Error(`chunk hash mismatch: ${resource.path}`);
      output.set(bytesChunk, offset);
      offset += chunk.size;
    }
    if (await this.sha256(output) !== recipe.sha256) throw new Error(`reconstructed resource hash mismatch: ${resource.path}`);
    return output;
  }
}
