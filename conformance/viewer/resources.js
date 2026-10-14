// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { createWorkerParser, validateResource } from './data.js';
import { compareLegacyVersions, compareVersions, resolveUnified } from './model.js';
import { validateUnifiedInputs, validateUnifiedCapture } from './unified.js';
import { isMapping, getField, hasField, setField, deleteField, mappingEntries, ensureMappingField } from './mappings.js';
import { implementationKeys, implementationValue, replaceImplementation, normalizeImplementations } from './comparison.js';

function cachedTask(cache, key, create) {
  if (!cache.has(key)) {
    const task = create().catch((error) => { cache.delete(key); throw error; });
    cache.set(key, task);
  }
  return cache.get(key);
}

// Transport errors stay outside recorded outcomes. Failed reads are evicted so
// Retry can recover without weakening hash validation or changing parser state.
export class FixtureResources {
  constructor(catalog, store, parser = createWorkerParser()) {
    this.catalog = catalog;
    this.store = store;
    this.parser = parser;
    this.archives = new Map();
    this.documents = new Map();
    this.captures = new Map();
  }

  unifiedDependencies(descriptors) {
    return this.catalog.sources.unified.filter(resource => resource.kind === 'inputs-and-golden' || descriptors.some(item => item.source.implementation === resource.implementation && compareVersions(resource.version, item.source.version) <= 0));
  }

  async prepareUnified(descriptors) {
    const dependencies = this.unifiedDependencies(descriptors);
    this.store.demand?.(dependencies);
    await Promise.all(dependencies.map(resource => this.store.read(resource)));
  }

  async prefetchUnified(alternatives, descriptors, isCurrent = () => true) {
    if (globalThis.navigator?.connection?.saveData) return;
    this.store.scheduler?.ready();
    let count = 0;
    for (const selected of alternatives) {
      if (!isCurrent()) return;
      const dependencies = this.unifiedDependencies(descriptors.filter(item => selected.includes(item.control.key))).filter(resource => !this.store.has?.(resource));
      if (!dependencies.length) continue;
      await Promise.allSettled(dependencies.map(resource => this.store.read(resource, { background: true })));
      if (++count === 2) break;
    }
  }

  async archive(resource) {
    validateResource(resource);
    return cachedTask(this.archives, `${resource.sha256}:${resource.size}`, async () => this.parser.parseArchive(await this.store.read(resource), { path: resource.path }));
  }

  async yaml(resource, member = null, unified = false) {
    validateResource(resource);
    const key = `${resource.sha256}:${resource.size}:${member || ''}:${unified}`;
    const parsed = await cachedTask(this.documents, key, async () => {
      const bytes = member ? (await this.archive(resource)).get(member) : await this.store.read(resource);
      if (!bytes) throw new Error(`archive member missing: ${member} in ${resource.path}`);
      const value = await this.parser.parseYaml(bytes, { path: member || resource.path, unified });
      return { value, sourceBytes: bytes };
    });
    return { ...parsed, member: member || resource.path, resource: resource.path };
  }

  async legacyDocuments(resource, families = null) {
    const rows = resource.members.filter((member) => member.kind === 'file' && member.path.endsWith('.yaml') && (!families || families.includes(member.path.split('/').at(-2))));
    return new Map(await Promise.all(rows.map(async (member) => {
      const identity = member.path.split('/').slice(-2).join('/');
      return [identity, await this.yaml(resource, member.path)];
    })));
  }

  async unified(family, implementation, version) {
    const input = this.catalog.sources.unified.find((item) => item.family === family && item.kind === 'inputs-and-golden');
    if (!input) throw new Error(`Unified input and GOLDEN document missing for ${family}`);
    const inputs = await this.yaml(input, null, true);
    validateUnifiedInputs(inputs.value, family, input.path, this.catalog.metadata.unified_aliases);
    const files = this.catalog.sources.unified.filter((item) => item.family === family && item.implementation === implementation && compareVersions(item.version, version) <= 0);
    const captures = await Promise.all(files.map((file) => cachedTask(this.captures, `${input.sha256}:${input.size}:${file.sha256}:${file.size}`, async () => {
      const parsed = await this.yaml(file, null, true);
      await validateUnifiedCapture(parsed.value, inputs.value, file);
      return { ...file, document: parsed.value, bytes: parsed.sourceBytes };
    })));
    return { inputs, records: resolveUnified(captures, version) };
  }

  async legacy({ root, implementation, version, mode }, families = null) {
    const prefix = `conformance/fixtures/${root}/`;
    const input = this.catalog.sources.archives.find((resource) => resource.path === `${prefix}inputs.tar.gz`);
    if (!input) throw new Error(`input archive missing: ${root}`);
    const inputs = await this.legacyDocuments(input, families);
    const documents = new Map([...inputs].map(([key, record]) => [key, { ...record, value: structuredClone(record.value) }]));
    const expression = new RegExp(`^${prefix.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}${implementation}-([^/]+)\\.tar\\.gz$`);
    const available = this.catalog.sources.archives.flatMap((resource) => {
      const match = resource.path.match(expression);
      return match ? [{ resource, version: match[1] }] : [];
    });
    let target = version;
    if (mode === 'stream' && !/\.patch\d+$/.test(version)) {
      for (const layer of available) if (layer.version.startsWith(`${version}.patch`) && compareLegacyVersions(layer.version, target) > 0) target = layer.version;
    }
    const layers = available.filter((layer) => compareLegacyVersions(layer.version, target) <= 0).sort((left, right) => compareLegacyVersions(left.version, right.version));
    if (mode === 'reasoning' && layers.length) {
      for (const record of documents.values()) {
        if (mappingEntries(getField(record.value, 'cases')).some(([, item]) => isMapping(getField(item, 'expected')) && isMapping(implementationValue(getField(item, 'expected'), implementation, this.catalog.metadata.implementations)) && !hasField(implementationValue(getField(item, 'expected'), implementation, this.catalog.metadata.implementations), 'unavailable'))) {
          replaceImplementation(ensureMappingField(record.value, 'captured_with'), implementation, layers.at(-1).version, this.catalog.metadata.implementations);
        }
      }
    }
    for (const layer of layers) {
      for (const [key, overlay] of await this.legacyDocuments(layer.resource, families)) {
        const record = documents.get(key);
        if (!record) continue;
        mergeLegacyDocument(record.value, overlay.value, implementation, mode, this.catalog.metadata.implementations);
        if (mode === 'stream') {
          replaceImplementation(ensureMappingField(record.value, 'captured_with'), implementation, version, this.catalog.metadata.implementations);
        }
      }
    }
    return documents;
  }

  dispose() { this.parser.terminate?.(); }
}

export function clearLegacyImplementation(record, implementation, implementations = []) {
  const keys = implementationKeys(implementation, implementations);
  for (const chunk of getField(record, 'chunks') || []) {
    for (const field of ['expected', 'normal_text']) for (const key of keys) deleteField(getField(chunk, field), key);
  }
  for (const field of ['unavailable', 'exception']) for (const key of keys) deleteField(getField(record, field), key);
}

export function mergeLegacyDocument(base, overlay, implementation, mode, implementations = []) {
  for (const [caseId, source] of mappingEntries(getField(overlay, 'cases'))) {
    const target = getField(getField(base, 'cases'), caseId);
    if (!target) continue;
    if (mode !== 'stream') {
      const expected = getField(source, 'expected');
      if (expected) for (const [key, value] of Object.entries(normalizeImplementations(expected, implementations))) {
        replaceImplementation(ensureMappingField(target, 'expected'), key, structuredClone(value), implementations);
      }
      continue;
    }
    // New capture state replaces this implementation in every chunk, including
    // trailing input rows which a shorter emission-packed capture never reaches.
    clearLegacyImplementation(target, implementation, implementations);
    const state = hasField(source, 'unavailable') ? 'unavailable' : hasField(source, 'exception') ? 'exception' : null;
    if (state) {
      replaceImplementation(ensureMappingField(target, state), implementation, structuredClone(getField(source, state)), implementations);
      continue;
    }
    (getField(source, 'chunks') || []).forEach((chunk, index) => {
      const baseChunk = getField(target, 'chunks')?.[index];
      if (!baseChunk) return;
      replaceImplementation(ensureMappingField(baseChunk, 'expected'), implementation, structuredClone(getField(chunk, 'expected') || []), implementations);
      if (getField(chunk, 'normal_text')) {
        replaceImplementation(ensureMappingField(baseChunk, 'normal_text'), implementation, getField(chunk, 'normal_text'), implementations);
      }
    });
  }
}
