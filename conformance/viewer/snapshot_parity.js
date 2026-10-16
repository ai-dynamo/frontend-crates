// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseSnapshot } from './snapshots.js';
import { parseYamlBytes } from './yaml_data.js';
import { compareVersions, resolveUnified } from './model.js';
import { sameValue } from './values.js';
import { validateResolvedUnified, validateUnifiedInputs } from './unified.js';

const [root, site, catalogPath] = process.argv.slice(2);
const catalog = JSON.parse(readFileSync(catalogPath, 'utf8'));
const originals = catalog.sources.original_unified;
let inputCount = 0;
let snapshotCount = 0;
let recordCount = 0;
for (const inputResource of catalog.sources.unified.filter(item => item.kind === 'inputs-and-golden')) {
  const original = originals.find(item => item.family === inputResource.family && item.kind === 'inputs-and-golden');
  const yaml = parseYamlBytes(readFileSync(join(root, original.path)), { path: original.path, unified: true });
  const snapshot = await parseSnapshot(readFileSync(join(site, inputResource.path)), inputResource);
  assert(sameValue(snapshot, yaml), `authored JSON differs from YAML: ${inputResource.family}`);
  validateUnifiedInputs(snapshot, inputResource.family, inputResource.path, catalog.metadata.unified_aliases);
  inputCount += 1;
}

const groups = new Map();
for (const resource of catalog.sources.unified.filter(item => item.kind === 'resolved-capture')) {
  const key = `${resource.family}\0${resource.implementation}`;
  if (!groups.has(key)) groups.set(key, []);
  groups.get(key).push(resource);
}
for (const resources of groups.values()) {
  const { family, implementation } = resources[0];
  const yamlHistory = originals.filter(item => item.kind === 'history' && item.family === family && item.implementation === implementation)
    .sort((left, right) => compareVersions(left.version, right.version))
    .map(item => {
      const bytes = readFileSync(join(root, item.path));
      return { ...item, bytes, document: parseYamlBytes(bytes, { path: item.path, unified: true }) };
    });
  const historyByPath = new Map(yamlHistory.map(item => [item.path, item]));
  const inputResource = catalog.sources.unified.find(item => item.kind === 'inputs-and-golden' && item.family === family);
  const input = await parseSnapshot(readFileSync(join(site, inputResource.path)), inputResource);
  for (const resource of resources) {
    const history = yamlHistory.filter(item => compareVersions(item.version, resource.version) <= 0);
    assert(history.length, `no YAML history for ${resource.capture_identity}`);
    const resolved = resolveUnified(history, resource.version);
    const originByCase = new Map();
    for (const capture of history) {
      for (const [caseId, change] of Object.entries(capture.document.changes || {})) {
        if (change?.absent === true && Object.keys(change).length === 1) originByCase.delete(caseId);
        else if (!originByCase.has(caseId) || Object.hasOwn(change, 'observation')) originByCase.set(caseId, capture.path);
      }
    }
    const snapshot = await parseSnapshot(readFileSync(join(site, resource.path)), resource);
    await validateResolvedUnified(snapshot, input, resource, originals);
    assert.equal(resolved.size, Object.keys(snapshot).length, `record count differs: ${resource.capture_identity}`);
    for (const [caseId, value] of resolved) {
      assert(sameValue(value.value, snapshot[caseId].value), `resolved record differs: ${resource.capture_identity}:${caseId}`);
      assert.equal(value.member, snapshot[caseId].member, `resolved source differs: ${resource.capture_identity}:${caseId}`);
      assert.equal(originByCase.get(caseId), snapshot[caseId].member, `source identity differs: ${resource.capture_identity}:${caseId}`);
      assert.deepEqual(value.sourceBytes, historyByPath.get(snapshot[caseId].member)?.bytes, `source bytes differ: ${resource.capture_identity}:${caseId}`);
      recordCount += 1;
    }
    snapshotCount += 1;
  }
}

assert(inputCount > 0 && snapshotCount > 0 && recordCount > 0);
console.log(`full corpus parity passed: ${inputCount} authored resources, ${snapshotCount} resolved captures, ${recordCount} complete records`);
