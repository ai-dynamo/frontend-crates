// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { parseSnapshot } from './snapshots.js';
import { createNodeResourceStore } from './data.js';
import './development_parser.js';
import { FixtureResources } from './resources.js';
import { sourceCapturesFamily } from './candidates.js';
import { compareVersions, resolveUnifiedSeries } from './model.js';
import { sameValue } from './values.js';
import { deleteField, getField, hasField, setField } from './mappings.js';
import { sourceReference, validateResolvedUnified, validateUnifiedCapture, validateUnifiedInputs } from './unified.js';

const [root, site, catalogPath, selectionPath] = process.argv.slice(2);
const selectedPaths = selectionPath ? new Set(JSON.parse(readFileSync(selectionPath, 'utf8'))) : null;
const catalog = JSON.parse(readFileSync(catalogPath, 'utf8'));
const originals = catalog.proof_original_unified;
assert(Array.isArray(originals), 'proof requires source catalog descriptors');
const originalCatalog = { ...catalog, sources: { ...catalog.sources, unified: originals } };
const sourceStore = await createNodeResourceStore(root, { allowSourceFiles: true });
const authoredReader = new FixtureResources(originalCatalog, sourceStore);
const snapshotStore = await createNodeResourceStore(site);
let authoredComparisons = 0;
let snapshotCount = 0;
let recordCount = 0;
let checkpointVisits = 0;
let decodedCaptureBundles = 0;
let retainedCaptureBundles = 0;
let peakRetainedCaptureBundles = 0;
let peakHeapBytes = 0;
async function readSnapshot(resource) {
  return parseSnapshot(await snapshotStore.read(resource), resource);
}
async function readOracleCapture(reader, file, input) {
  const parsed = await reader.yaml(file, null, true);
  let execution = null;
  if (parsed.value.execution) {
    const sourcePath = (resource, reference, directory) => `${resource.path.replace(/\/(?:families|executions|receipts)\/.*$/, '')}/${sourceReference(reference, directory)}`;
    const manifest = originals.find(item => item.kind === 'execution' && item.path === sourcePath(file, parsed.value.execution, 'executions'));
    if (!manifest || file.execution !== parsed.value.execution) throw new Error(`execution manifest not indexed for ${file.path}`);
    execution = await reader.yaml(manifest, null, true);
    const receiptIds = Object.keys(execution.value.receipts || {}).sort();
    assert.deepEqual(receiptIds, [...(manifest.receipts || [])].sort(), 'execution receipt dependencies not indexed');
    const receipts = new Map(await Promise.all(receiptIds.map(async digest => {
      const resource = originals.find(item => item.kind === 'receipt' && item.path === sourcePath(manifest, digest, 'receipts'));
      if (!resource) throw new Error(`execution receipt not indexed: ${digest}`);
      return [digest, await reader.yaml(resource, null, true)];
    })));
    execution = { ...execution, receipts };
  }
  const document = await validateUnifiedCapture(parsed.value, input, file, execution);
  return { ...file, document, bytes: parsed.sourceBytes };
}
const authoredFamilies = new Set(originals.filter(item => item.kind === 'inputs-and-golden').map(item => item.family));
const inputResources = catalog.sources.unified.filter(item => item.kind === 'inputs-and-golden');
const publishedInputFamilies = new Set(inputResources.map(item => item.family));
const missingInputFamilies = [...authoredFamilies].filter(family => !publishedInputFamilies.has(family));
const extraInputFamilies = [...publishedInputFamilies].filter(family => !authoredFamilies.has(family));
assert.equal(missingInputFamilies.length + extraInputFamilies.length, 0,
  `published authored family membership differs: missing ${JSON.stringify(missingInputFamilies)}; extra ${JSON.stringify(extraInputFamilies)}`);
for (const inputResource of inputResources) {
  const family = inputResource.family;
  const snapshot = await readSnapshot(inputResource);
  const original = originals.find(item => item.family === family && item.kind === 'inputs-and-golden');
  const { value: yaml } = await authoredReader.yaml(original, null, true);
  assert(sameValue(snapshot, yaml), `authored JSON differs from YAML: ${family}`);
  validateUnifiedInputs(snapshot, family, inputResource.path, catalog.metadata.unified_aliases);
  authoredComparisons += 1;
}

const histories = new Map();
const earliestVersions = new Map();
for (const resource of originals.filter(item => item.kind === 'history')) {
  const key = `${resource.family}\0${resource.implementation}`;
  if (!histories.has(key)) histories.set(key, []);
  histories.get(key).push(resource);
  if (!earliestVersions.has(key) || compareVersions(resource.version, earliestVersions.get(key)) < 0) earliestVersions.set(key, resource.version);
}
const expectedSelections = new Set();
const selectionIdentity = (implementation, version, family) => JSON.stringify([implementation, version, family]);
for (const candidate of catalog.sources.candidates.filter(item => item.mode === 'unified')) {
  for (const family of authoredFamilies) {
    const earliest = earliestVersions.get(`${family}\0${candidate.implementation}`);
    if (sourceCapturesFamily(candidate, family, { nativeOnly: false }) && earliest && compareVersions(earliest, candidate.version) <= 0) {
      expectedSelections.add(selectionIdentity(candidate.implementation, candidate.version, family));
    }
  }
}
const implementations = new Map();
const publishedSelections = new Set();
for (const resource of catalog.sources.unified.filter(item => item.kind === 'resolved-capture')) {
  for (const family of resource.families || [resource.family]) {
    const identity = selectionIdentity(resource.implementation, resource.version, family);
    assert(!publishedSelections.has(identity), `duplicate published capture selection: ${identity}`);
    publishedSelections.add(identity);
  }
  if (!implementations.has(resource.implementation)) implementations.set(resource.implementation, []);
  implementations.get(resource.implementation).push(resource);
}
const missingSelections = [...expectedSelections].filter(identity => !publishedSelections.has(identity));
const extraSelections = [...publishedSelections].filter(identity => !expectedSelections.has(identity));
assert.equal(missingSelections.length + extraSelections.length, 0,
  `published capture selection membership differs: ${missingSelections.length} missing ${JSON.stringify(missingSelections.slice(0, 10))}; ${extraSelections.length} extra ${JSON.stringify(extraSelections.slice(0, 10))}`);
// Validate complete publication membership above, then partition comparison work.
if (selectedPaths) {
  const known = new Set(catalog.sources.unified.filter(item => item.kind === 'resolved-capture').map(item => item.path));
  for (const path of selectedPaths) assert(known.has(path), `unknown selected capture: ${path}`);
}
console.log(`parity selection: ${JSON.stringify(selectedPaths ? [...selectedPaths].sort() : catalog.sources.unified.filter(item => item.kind === 'resolved-capture').map(item => item.path).sort())}`);
for (const [implementation, assigned] of implementations) {
  // Release expanded YAML and receipt caches before the next implementation.
  const reader = new FixtureResources(originalCatalog, sourceStore);
  const families = new Map();
  let resources = assigned;
  const allResources = [...resources];
  if (selectedPaths) resources = resources.filter(item => selectedPaths.has(item.path));
  if (!resources.length) continue;
  resources.sort((left, right) => compareVersions(left.version, right.version));
  for (const resource of resources) {
    // A bundle is decoded once, compared for every family, then released.
    const bundle = await readSnapshot(resource);
    decodedCaptureBundles += 1;
    retainedCaptureBundles += 1;
    peakRetainedCaptureBundles = Math.max(peakRetainedCaptureBundles, retainedCaptureBundles);
    for (const family of resource.families || [resource.family]) {
      if (!families.has(family)) {
        const { value: input, sourceResource: originalInput } = await reader.unifiedInputs(family);
        const descriptors = [...(histories.get(`${family}\0${implementation}`) || [])]
          .sort((left, right) => compareVersions(left.version, right.version));
        const first = await readOracleCapture(reader, descriptors[0], input);
        const versions = allResources.filter(item => (item.families || [item.family]).includes(family)).map(item => item.version);
        if (first.document.schema_version === 2 || first.document.schema_version === 2n) {
          families.set(family, { input, originalInput, descriptors, complete: true, originIndex: 0 });
        } else {
          const yamlHistory = await Promise.all(descriptors.map(item => readOracleCapture(reader, item, input)));
          families.set(family, { input, history: yamlHistory, byPath: new Map(yamlHistory.map(item => [item.path, item])),
            series: resolveUnifiedSeries(yamlHistory, versions), originByCase: new Map(), originIndex: 0 });
        }
      }
      const context = families.get(family);
      let resolved;
      if (context.complete) {
        while (context.originIndex < context.descriptors.length
            && compareVersions(context.descriptors[context.originIndex].version, resource.version) <= 0) {
          context.originIndex += 1;
          checkpointVisits += 1;
        }
        assert(context.originIndex, `no YAML snapshot for ${resource.capture_identity}`);
        const descriptor = context.descriptors[context.originIndex - 1];
        const capture = await readOracleCapture(reader, descriptor, context.input);
        // This oracle reads complete source records directly, independently of
        // the publisher and checkout reader's selected-snapshot projection.
        resolved = new Map(Object.entries(structuredClone(capture.document.records)));
        if (descriptor.version !== resource.version) for (const record of resolved.values()) {
          const document = getField(record.value, 'document');
          deleteField(getField(document, 'record_metadata'), 'verified_recapture');
          if (!hasField(document, 'inherited_from') && record.suppress_inherited_from !== true) setField(document, 'inherited_from', descriptor.version);
        }
        reader.documents.clear();
        reader.captures.clear();
      } else {
        let next = context.series.next();
        while (!next.done && compareVersions(next.value[0], resource.version) < 0) next = context.series.next();
        assert.equal(next.value?.[0], resource.version, 'incremental version differs from published resource');
        resolved = next.value[1];
        while (context.originIndex < context.history.length && compareVersions(context.history[context.originIndex].version, resource.version) <= 0) {
          const capture = context.history[context.originIndex++];
          checkpointVisits += 1;
          for (const [caseId, change] of Object.entries(capture.document.changes || {})) {
            if (change?.absent === true && Object.keys(change).length === 1) context.originByCase.delete(caseId);
            else if (!context.originByCase.has(caseId) || Object.hasOwn(change, 'observation')) context.originByCase.set(caseId, capture.path);
          }
        }
      }
      assert(context.originIndex, `no YAML history for ${resource.capture_identity}`);
      const snapshot = resource.families ? bundle[family] : bundle;
      await validateResolvedUnified(snapshot, context.input, { ...resource, family }, originals);
      assert.equal(resolved.size, Object.keys(snapshot).length, `record count differs: ${resource.capture_identity}:${family}`);
      for (const [caseId, value] of resolved) {
        assert(sameValue(value.value, snapshot[caseId].value), `resolved record differs: ${resource.capture_identity}:${family}:${caseId}`);
        assert.equal(value.member, snapshot[caseId].member, `resolved source differs: ${resource.capture_identity}:${caseId}`);
        if (!context.complete) {
          assert.equal(context.originByCase.get(caseId), snapshot[caseId].member, `source identity differs: ${resource.capture_identity}:${caseId}`);
          assert.deepEqual(value.sourceBytes, context.byPath.get(snapshot[caseId].member)?.bytes, `source bytes differ: ${resource.capture_identity}:${caseId}`);
        }
        recordCount += 1;
      }
      snapshotCount += 1;
    }
    peakHeapBytes = Math.max(peakHeapBytes, process.memoryUsage().heapUsed);
    retainedCaptureBundles -= 1;
  }
}

assert(authoredComparisons > 0);
assert(selectedPaths?.size === 0 || (snapshotCount > 0 && recordCount > 0));
console.log(`full corpus parity passed: ${authoredComparisons} authored family comparisons, ${snapshotCount} resolved captures, ${recordCount} complete records`);
console.log(`parity work: ${checkpointVisits} checkpoint visits, ${decodedCaptureBundles} capture bundles decoded, ${peakRetainedCaptureBundles} retained capture bundle, ${peakHeapBytes} peak sampled heap bytes`);
