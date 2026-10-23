// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { selectedUnifiedFamilies, unifiedDisplayFamilyPredicate } from './candidates.js';
import { createWorkerParser, validateResource, resourceMetrics } from './data.js';
import { compareLegacyVersions, compareVersions, selectedUnifiedSnapshot } from './model.js';
import { validateUnifiedInputs, validateUnifiedSnapshot, validateResolvedUnified, validateResolvedRecordContracts } from './unified.js';
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
    this.inputResources = new Map();
  }

  unifiedDependencies(descriptors) {
    const families = selectedUnifiedFamilies(this.catalog, descriptors);
    const capturesFamily = unifiedDisplayFamilyPredicate(this.catalog, descriptors);
    const inputs = this.catalog.sources.unified.filter(resource => resource.kind === 'inputs-and-golden' && families.has(resource.family));
    const histories = new Map();
    for (const descriptor of descriptors) {
      for (const family of families) {
        if (!capturesFamily(descriptor.source, family)) continue;
        const packed = this.catalog.sources.unified.find(resource => resource.kind === 'resolved-capture'
          && resource.implementation === descriptor.source.implementation
          && resource.version === descriptor.source.version
          && (resource.family === family || resource.families?.includes(family)));
        if (packed) {
          histories.set(packed.path, packed);
          continue;
        }
        const selected = this.catalog.sources.unified
          .filter(resource => resource.kind === 'history' && resource.family === family
            && resource.implementation === descriptor.source.implementation
            && compareVersions(resource.version, descriptor.source.version) <= 0)
          .sort((left, right) => compareVersions(right.version, left.version))[0];
        if (selected) histories.set(selected.path, selected);
      }
    }
    const selected = new Set([...inputs, ...histories.values()].map(resource => resource.path));
    return this.catalog.sources.unified.filter(resource => selected.has(resource.path));
  }

  async prepareUnified(descriptors) {
    const dependencies = this.unifiedDependencies(descriptors);
    this.inputResources = new Map();
    for (const resource of dependencies.filter(item => item.kind === 'inputs-and-golden')) {
      this.inputResources.set(resource.family, resource);
    }
    const selection = this.store.demand?.(dependencies) || true;
    try {
      await Promise.all(dependencies.map(resource => this.store.read(resource, {selection})));
    } finally {
      this.store.releaseReservations?.(selection);
    }
  }

  async admitResource(resource) {
    validateResource(resource);
    await this.store.admit?.(resource);
  }

  async archive(resource) {
    await this.admitResource(resource);
    return cachedTask(this.archives, `${resource.sha256}:${resource.size}`, async () => this.parser.parseArchive(await this.store.read(resource), { path: resource.path }));
  }

  async yaml(resource, member = null, unified = false) {
    if (resource.format === 'typed-json-v1') {
      if (member || !unified || resource.kind !== 'inputs-and-golden') throw new Error('snapshot is not an authored Unified document');
      return this.snapshot(resource);
    }
    await this.admitResource(resource);
    const key = `${resource.sha256}:${resource.size}:${member || ''}:${unified}`;
    const parsed = await cachedTask(this.documents, key, async () => {
      const bytes = member ? (await this.archive(resource)).get(member) : await this.store.read(resource);
      if (!bytes) throw new Error(`archive member missing: ${member} in ${resource.path}`);
      const value = await this.parser.parseYaml(bytes, { path: member || resource.path, unified });
      return { value, sourceBytes: bytes };
    });
    return { ...parsed, member: member || resource.path, resource: resource.path };
  }

  async snapshot(resource) {
    await this.admitResource(resource);
    const value = await cachedTask(this.documents, `${resource.sha256}:${resource.size}:snapshot`, async () => this.parser.parseSnapshot(await this.store.read(resource), resource));
    return { value, member: resource.path, resource: resource.path };
  }

  async legacyDocuments(resource, families = null) {
    const rows = resource.members.filter((member) => member.kind === 'file' && member.path.endsWith('.yaml') && (!families || families.includes(member.path.split('/').at(-2))));
    return new Map(await Promise.all(rows.map(async (member) => {
      const identity = member.path.split('/').slice(-2).join('/');
      return [identity, await this.yaml(resource, member.path)];
    })));
  }

  async unifiedInputs(family) {
    const input = this.inputResources.get(family) || this.catalog.sources.unified.find(item => item.kind === 'inputs-and-golden' && item.family === family);
    if (!input) throw new Error(`Unified input and GOLDEN document missing for ${family}`);
    const parsedInputs = input.format === 'typed-json-v1' ? await this.snapshot(input) : await this.yaml(input, null, true);
    const inputDocument = parsedInputs.value;
    if (!inputDocument) throw new Error(`Unified input and GOLDEN document missing for ${family}`);
    validateUnifiedInputs(inputDocument, family, input.path, this.catalog.metadata.unified_aliases);
    return { ...parsedInputs, value: inputDocument, sourceResource: input };
  }

  async unified(family, implementation, version, inputDocument = undefined) {
    const inputs = inputDocument === undefined ? await this.unifiedInputs(family)
      : { value: inputDocument, sourceResource: this.inputResources.get(family) || this.catalog.sources.unified.find(item => item.kind === 'inputs-and-golden'
        && item.family === family) };
    const { sourceResource: input, ...inputResult } = inputs;
    inputDocument = inputResult.value;
    if (input.format === 'typed-json-v1') {
      const resource = this.catalog.sources.unified.find(item => item.kind === 'resolved-capture'
        && item.implementation === implementation && item.version === version
        && (item.family === family || item.families?.includes(family)));
      if (!resource) return { inputs: inputResult, records: new Map() };
      if (resource.format !== input.format || resource.source_fingerprint !== input.source_fingerprint) throw new Error('inconsistent Unified resource format or fingerprint');
      const resolved = await this.snapshot(resource);
      const familyRecords = Array.isArray(resource.families) ? resolved.value[family] : resolved.value;
      if (!familyRecords) return { inputs: inputResult, records: new Map() };
      const familyResource = Array.isArray(resource.families) ? { ...resource, family } : resource;
      const validationStart = performance.now();
      await validateResolvedUnified(familyRecords, inputDocument, familyResource, this.catalog.sources.original_unified || []);
      resourceMetrics.validation += performance.now() - validationStart;
      return { inputs: inputResult, records: new Map(Object.entries(structuredClone(familyRecords))) };
    }
    const captureMode = this.catalog.metadata.unified_capture_policy?.[implementation]?.mode;
    if (!captureMode) throw new Error(`Unified capture mode missing for ${implementation}`);
    const file = this.catalog.sources.unified
      .filter(item => item.kind === 'history' && item.family === family && item.implementation === implementation
        && compareVersions(item.version, version) <= 0)
      .sort((left, right) => compareVersions(right.version, left.version))[0];
    if (!file) return { inputs: inputResult, records: new Map() };
    const capture = await this.unifiedCapture(file, inputDocument, input);
    await validateResolvedUnified(capture.document.records, inputDocument, { ...file, record_contract: captureMode },
      this.catalog.sources.original_unified || this.catalog.sources.unified);
    return { inputs: inputResult, records: validateResolvedRecordContracts(
      selectedUnifiedSnapshot(new Map(Object.entries(capture.document.records)), file.version, version), implementation, captureMode) };
  }

  async unifiedCapture(file, inputDocument, input) {
    const identity = JSON.stringify([input, file].map(resource => [resource.path, resource.sha256, resource.size]));
    return cachedTask(this.captures, identity, async () => {
      const parsed = await this.yaml(file, null, true);
      const document = await validateUnifiedSnapshot(parsed.value, inputDocument, file);
      return { ...file, document, bytes: parsed.sourceBytes };
    });
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
