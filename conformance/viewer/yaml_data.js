// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import * as YAML from './vendor/yaml/index.js';
import { sameMappingKey, getField, mappingEntries } from './values.js';
import { yamlTemporal } from './values.js';
import { sharedRead } from './data.js';
import { bytesOf, readTarGzip, validateResource, configureDevelopmentParser } from './data.js';

// yaml's YAML 1.1 schema includes y/n and unsigned/no-dot exponents. PyYAML's
// frozen readers do not. Adapt tag tests, preserving the library's resolvers.
function pythonYamlTags(tags) {
  return tags.map((tag) => {
    if (tag.tag === 'tag:yaml.org,2002:timestamp') return { ...tag, test: /^(?:\d{4}-\d{2}-\d{2}|\d{4}-\d{1,2}-\d{1,2}(?:[Tt]|[ \t]+)\d{1,2}:\d{2}:\d{2}(?:\.\d*)?(?:[ \t]*(?:Z|[-+]\d{1,2}(?::\d{2})?))?)$/, resolve: (source) => yamlTemporal(source) };
    if (tag.tag === 'tag:yaml.org,2002:bool') return { ...tag, test: tag.test.test('y') ? /^(?:yes|Yes|YES|true|True|TRUE|on|On|ON)$/ : /^(?:no|No|NO|false|False|FALSE|off|Off|OFF)$/ };
    if (tag.tag === 'tag:yaml.org,2002:int' && !tag.format) return { ...tag, test: /^[-+]?(?:0|[1-9][0-9_]*)$/ };
    if (tag.tag === 'tag:yaml.org,2002:int' && tag.format === 'TIME') return { ...tag, test: /^[-+]?[1-9][0-9_]*(?::[0-5]?[0-9])+$/ };
    if (tag.tag === 'tag:yaml.org,2002:float' && tag.format === 'EXP') return { ...tag, test: /^(?:[-+]?[0-9][0-9_]*\.[0-9_]*|\.[0-9][0-9_]*)[eE][-+][0-9]+$/ };
    if (tag.tag === 'tag:yaml.org,2002:float' && !tag.format && !tag.test.test('.inf')) return { ...tag, test: /^(?:[-+]?[0-9][0-9_]*\.[0-9_]*|\.[0-9][0-9_]*)$/ };
    return tag;
  });
}
function yamlMappings(value, seen = new WeakMap()) {
  if (!value || typeof value !== 'object' || value instanceof Date) return value;
  if (seen.has(value)) return seen.get(value);
  if (value instanceof Map) {
    const stringKeys = [...value.keys()].every((key) => typeof key === 'string');
    const result = stringKeys ? {} : new Map();
    seen.set(value, result);
    for (const [key, item] of value) {
      if (key !== null && typeof key === 'object' && !(key instanceof Date) && !(key instanceof Uint8Array) && !(key instanceof Uint16Array)) throw new Error('YAML mapping key is not hashable');
      const mapped = yamlMappings(item, seen);
      if (stringKeys) Object.defineProperty(result, key, { value: mapped, enumerable: true, writable: true, configurable: true });
      else {
        const existing = [...result.keys()].findIndex((prior) => sameMappingKey(prior, key));
        result.set(existing < 0 ? key : [...result.keys()][existing], mapped);
      }
    }
    return result;
  }
  if (Array.isArray(value)) {
    const result = [];
    seen.set(value, result);
    for (const item of value) result.push(yamlMappings(item, seen));
    return result;
  }
  return value;
}

export function parseYaml(source, { path = '<yaml>', unified = false } = {}) {
  const doc = YAML.parseDocument(source, {
    version: '1.1',
    schema: 'yaml-1.1',
    customTags: pythonYamlTags,
    uniqueKeys: unified ? (left, right) => YAML.isScalar(left) && YAML.isScalar(right) ? sameMappingKey(left.value, right.value) : left === right : false,
    merge: !unified,
    intAsBigInt: true,
    maxAliasCount: 100,
    prettyErrors: false,
    strict: true,
  });
  if (doc.errors.length) throw new Error(`${path}: ${doc.errors.map((item) => item.message).join('; ')}`);
  if (unified) {
    YAML.visit(doc, (_, node) => {
      if (YAML.isAlias(node)) throw new Error(`${path}: YAML aliases are not allowed`);
      if (node?.anchor) throw new Error(`${path}: YAML anchors are not allowed`);
      if (node?.tag) throw new Error(`${path}: YAML tags are not allowed`);
    });
    if (!YAML.isMap(doc.contents)) throw new Error(`${path}: YAML document must be a mapping`);
  }
  return yamlMappings(doc.toJS({ mapAsMap: true, maxAliasCount: 100 }));
}

export function parseYamlBytes(bytes, options = {}) {
  return parseYaml(new TextDecoder('utf-8', { fatal: true }).decode(bytesOf(bytes)), options);
}

export function readArchiveCases(input, options = {}) {
  const members = readTarGzip(input, options);
  const cases = new Map();
  for (const [member, bytes] of members) {
    if (!member.endsWith('.yaml')) continue;
    const document = parseYamlBytes(bytes, { path: member });
    for (const [caseId, value] of mappingEntries(getField(document, 'cases'))) {
      cases.set(caseId, { value, member, sourceBytes: bytes });
    }
  }
  return cases;
}


const sourceArchives = new WeakMap();

export async function readSourceMember(store, resource, signal) {
  // Only the explicit checkout reader reaches this path. Published originals
  // retain their own recipes and never fall back to source archives.
  if (!store.allowSourceFiles) throw new Error(`Source files disabled in published site: ${resource.path}`);
  if (!sourceArchives.has(store)) sourceArchives.set(store, { archives: new Map(), sizes: new Map(), bytes: 0 });
  const cache = sourceArchives.get(store);
  const archive = resource.archive;
  validateResource(archive);
  const prefix = archive.path.replace(/\/versions\/[^/]+$/, '/');
  if (prefix === archive.path || !resource.path.startsWith(prefix)) throw new Error('invalid source archive binding');
  const member = resource.path.slice(prefix.length);
  const key = store.key(archive);
  const members = await sharedRead(cache.archives, key, async sharedSignal => {
    const bytes = await store.readTransportBytes(archive, sharedSignal);
    sharedSignal.throwIfAborted();
    if (bytes.length < 18 || new DataView(bytes.buffer, bytes.byteOffset).getUint32(bytes.length - 4, true) > 512 * 1024 * 1024) {
      throw new Error('source archive exceeds decoded limit');
    }
    const members = readTarGzip(bytes, { path: archive.path, maximumDecoded: 512 * 1024 * 1024 });
    const size = [...members.values()].reduce((total, memberBytes) => total + memberBytes.length, 0);
    cache.sizes.set(key, size);
    cache.bytes += size;
    while (cache.bytes > 64 * 1024 * 1024) {
      const [oldest, evictedSize] = cache.sizes.entries().next().value;
      cache.sizes.delete(oldest);
      cache.archives.delete(oldest);
      cache.bytes -= evictedSize;
    }
    return members;
  }, signal);
  const bytes = members.get(member);
  if (!bytes) throw new Error(`archive member missing: ${member} in ${archive.path}`);
  return bytes;
}

export function initializeDevelopmentParser() {
  configureDevelopmentParser(() => import('./yaml_data.js'), new URL('./development_worker.js', import.meta.url));
}
