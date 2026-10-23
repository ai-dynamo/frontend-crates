// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { FixtureResources } from './resources.js';
import { tabCandidates } from './candidates.js';
import { legacyCases } from './legacy.js';
import { compareLegacyVersions } from './model.js';
import { getField, hasField, isMapping, setField, mappingEntries } from './mappings.js';
import { unavailablePeers } from './presentation.js';
import { legacyOutcome, implementationValue, replaceImplementation } from './comparison.js';

const catalogs = new WeakMap();

function expandDocument(control) {
  const document = { family: control.family, mode: control.mode, cases: {} };
  for (const [id, source] of Object.entries(control.cases)) {
    const record = structuredClone(source);
    if (source.expected) record.expected = Object.fromEntries(Object.entries(source.expected).map(([implementation, state]) => [implementation,
      state === 'unavailable' ? { unavailable: true } : state === 'unavailable-error' ? { unavailable: true, parserError: true } : state === 'present' ? {} : null]));
    if (source.chunk_runs) {
      record.chunks = source.chunk_runs.flatMap(([count, shape]) => Array.from({ length: count }, () => {
        const chunk = {};
        if (Object.hasOwn(shape, 'expected')) chunk.expected = Array.isArray(shape.expected) ? Object.fromEntries(shape.expected.map(implementation => [implementation, []])) : [];
        if (Object.hasOwn(shape, 'normal_text')) chunk.normal_text = structuredClone(shape.normal_text);
        return chunk;
      }));
      delete record.chunk_runs;
    }
    document.cases[id] = record;
  }
  return document;
}

class AvailabilityResources extends FixtureResources {
  constructor(catalog) { super(catalog, null, {}); }

  async legacyDocuments(resource, families = null) {
    if (!resource.availability_documents) throw new Error(`availability controls missing: ${resource.path}`);
    return new Map(Object.entries(resource.availability_documents)
      .filter(([, value]) => !families || families.includes(value.family))
      .map(([member, value]) => [member.split('/').slice(-2).join('/'), { member, value: expandDocument(value) }]));
  }
}

// Replay presence/state controls through the same history owner as real captures.
// Presentation describes the inventory, so selected or historical candidates
// cannot make an unloaded peer look as though it has no parser.
export async function catalogPeerAvailability(catalog, spec) {
  return controlTask(catalog, spec.root, () => resolveAvailability(catalog, spec));
}

function controlTask(catalog, key, create) {
  if (!catalog.sources.archives.some((resource) => resource.availability_documents)) return null;
  if (!catalogs.has(catalog)) catalogs.set(catalog, new Map());
  const tasks = catalogs.get(catalog);
  if (!tasks.has(key)) tasks.set(key, create().catch((error) => {
    tasks.delete(key);
    throw error;
  }));
  return tasks.get(key);
}

export async function catalogFixtureApplicability(catalog, spec) {
  return controlTask(catalog, `${spec.root}:fixtures`, async () => {
    const reader = new AvailabilityResources(catalog);
    const descriptor = tabCandidates(catalog, spec).find(candidate => candidate.control.default_bucket === 'A');
    const source = descriptor.source;
    const reasoning = spec.suite === 'reasoning';
    const documents = await reader.legacy({ root: spec.root, mode: reasoning ? 'reasoning' : spec.mode, implementation: source.implementation, version: source.version });
    if (reasoning) return new Map([...documents.values()].flatMap(entry => mappingEntries(getField(entry.value, 'cases')).map(([id, record]) => {
      const block = implementationValue(getField(record, 'expected'), source.implementation, catalog.metadata.implementations);
      return [`${getField(entry.value, 'family')}\0${id}`, { applicable: isMapping(block) && !hasField(block, 'unavailable'), present: isMapping(block) }];
    })));
    const captured = legacyCases(documents, spec.mode === 'stream' ? 'streamv1' : spec.mode, catalog.metadata, spec.mode === 'stream' ? [source.implementation] : null, { controls: true });
    return new Map([...captured.cases].map(([key, entry]) => {
      const block = getField(getField(entry.record, 'expected'), source.implementation);
      return [key, { applicable: legacyOutcome(block, [], { unavailableParserError: block?.parserError === true }).status !== 'na', present: isMapping(block) }];
    }));
  });
}

async function resolveAvailability(catalog, spec) {
  const reader = new AvailabilityResources(catalog);
  const metadata = catalog.metadata;
  const resource = catalog.sources.archives.find((item) => item.path === `conformance/fixtures/${spec.root}/inputs.tar.gz`);
  if (!resource) throw new Error(`availability inputs missing: ${spec.root}`);
  const inputs = await reader.legacyDocuments(resource);
  const reasoning = spec.suite === 'reasoning';
  const mode = reasoning ? 'reasoning' : spec.mode;
  const descriptors = tabCandidates(catalog, spec);
  const versions = new Map();
  for (const descriptor of metadata.selection_policy[mode].inventory_availability === 'inputs' ? [] : descriptors) {
    const source = descriptor.source;
    if (source.mode !== mode || !['vllm_python', 'sglang_python'].includes(source.implementation)) continue;
    const prior = versions.get(source.implementation);
    const oldest = metadata.selection_policy[mode].inventory_defaults?.[source.implementation] === 'oldest';
    if (!prior || compareLegacyVersions(source.version, prior.version) * (oldest ? -1 : 1) > 0) versions.set(source.implementation, source);
  }
  const current = reasoning ? inputs : legacyCases(inputs, mode === 'stream' ? 'streamv1' : mode, metadata, null, { controls: true }).cases;
  for (const [implementation, source] of versions) {
    const documents = await reader.legacy({ root: spec.root, mode, implementation, version: source.version });
    if (reasoning) {
      for (const [key, capture] of documents) {
        const target = current.get(key);
        if (!target) continue;
        for (const [id, record] of mappingEntries(getField(capture.value, 'cases'))) {
          const existing = getField(getField(target.value, 'cases'), id);
          const block = implementationValue(getField(record, 'expected'), implementation, metadata.implementations);
          if (!existing || block === undefined) continue;
          const expected = getField(existing, 'expected') || {};
          replaceImplementation(expected, implementation, block, metadata.implementations);
          setField(existing, 'expected', expected);
        }
      }
    } else {
      const captured = legacyCases(documents, mode === 'stream' ? 'streamv1' : mode, metadata, mode === 'stream' ? [implementation] : null, { controls: true });
      for (const [key, entry] of captured.cases) {
        const target = current.get(key);
        if (!target) continue;
        const block = getField(getField(entry.record, 'expected'), implementation);
        if (block === undefined) continue;
        const expected = getField(target.record, 'expected') || {};
        replaceImplementation(expected, implementation, block, metadata.implementations);
        setField(target.record, 'expected', expected);
      }
    }
  }
  const families = new Map();
  const add = (family, record) => {
    if (!families.has(family)) families.set(family, []);
    families.get(family).push(record);
  };
  if (reasoning) for (const entry of current.values()) for (const [, record] of mappingEntries(getField(entry.value, 'cases'))) add(getField(entry.value, 'family'), record);
  else for (const entry of current.values()) add(entry.family, entry.record);
  return unavailablePeers(families, metadata, reasoning);
}
