// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import assert from 'node:assert/strict';
import { validateUnifiedInputs, validateUnifiedCapture, materializeUnifiedRecord } from './unified.js';
import { canonicalJson } from './values.js';
import { sha256 } from './data.js';
import { resolveUnified } from './model.js';

assert.equal(canonicalJson({ z: 9007199254740993n, a: [1, 1n, 1e-7, 1e16, -0] }), '{"a":[1.0,1,1e-07,1e+16,-0.0],"z":9007199254740993}');
const request = { input: 'sample', init: {}, finish_reason: 'stop', tools: [], chunks: [] };
const inputs = { input_document: { family: 'sample' }, golden_document: { family: 'sample' }, cases: { id: { lifecycle: 'active', scenario: 'sample', description: 'sample case', policy: [], display_id: 'UNIFIED.1', historical_ids: [], request, golden: { assembled: [] } } } };
validateUnifiedInputs(inputs, 'sample', 'inputs.yaml');
const noGolden = structuredClone(inputs);
delete noGolden.cases.id.golden;
assert.throws(() => validateUnifiedInputs(noGolden, 'sample', 'inputs.yaml'), /missing golden/);
const emptyOracle = structuredClone(inputs);
emptyOracle.cases.id.golden = {};
assert.throws(() => validateUnifiedInputs(emptyOracle, 'sample', 'inputs.yaml'), /assembled event list/);
const badRequest = structuredClone(inputs);
delete badRequest.cases.id.request.init;
assert.throws(() => validateUnifiedInputs(badRequest, 'sample', 'inputs.yaml'), /missing init/);
const unknownField = { ...inputs, unexpected: {} };
assert.throws(() => validateUnifiedInputs(unknownField, 'sample', 'inputs.yaml'), /unknown field/);
const digest = await sha256(new TextEncoder().encode(canonicalJson(request)));
const capture = { provenance: { status: 'captured', origin: { crate_version: '0.1.0', source_sha256: 'a'.repeat(64) } }, document: {}, changes: { id: { case_key: 'UNIFIED.1', stimulus: { ref: 'current', semantic_sha256: digest }, observation: { value: { chunks: [] } } } }, metadata_changes: {}, document_overrides: {} };
const identity = { path: 'sample-0.1.0.yaml', implementation: 'sample', version: '0.1.0' };
await validateUnifiedCapture(capture, inputs, identity);
for (const mutate of [
  (value) => { value.changes.id.stimulus.semantic_sha256 = '0'.repeat(64); },
  (value) => { value.changes.id.stimulus.semantic_sha256 = ''; },
  (value) => { value.changes.id.stimulus.inline = request; },
  (value) => { value.changes.id.observation.error = { code: 'error' }; },
  (value) => { value.changes.id.observation.value = {}; },
  (value) => { value.changes.other = value.changes.id; },
  (value) => { value.document.captured_with = {}; },
  (value) => { value.provenance.origin.crate_version = '0.2.0'; },
  (value) => { value.document_overrides.id = { family: 'other' }; },
  (value) => { value.changes = new Map([[1n, { absent: true }]]); },
  (value) => { value.metadata_changes = new Map([[1n, {}]]); },
  (value) => { value.document_overrides = new Map([[1n, {}]]); },
  (value) => { value.changes.constructor = { absent: true }; },
  (value) => { value.metadata_changes.toString = {}; },
  (value) => { value.document_overrides.constructor = {}; },
]) {
  const invalid = structuredClone(capture);
  mutate(invalid);
  await assert.rejects(validateUnifiedCapture(invalid, inputs, identity));
}
const timestampRequest = structuredClone(inputs);
timestampRequest.cases.id.request.init = new Date();
assert.throws(() => validateUnifiedInputs(timestampRequest, 'sample', 'inputs.yaml'), /mapping/);
const duplicateAlias = structuredClone(inputs);
duplicateAlias.cases.id.display_id = 'UNIFIED.1-tool_only';
duplicateAlias.cases.other = { ...duplicateAlias.cases.id, display_id: 'UNIFIED.1.a' };
const aliases = { owners: { '1-tool_only': ['tool_only'], '1.a': ['tool_only'] }, canonical: { tool_only: '1-tool_only' }, families: {} };
assert.throws(() => validateUnifiedInputs(duplicateAlias, 'sample', 'inputs.yaml', aliases), /duplicate canonical ID/);
const legacy = structuredClone(capture);
legacy.provenance = { status: 'legacy', captured_with: { sample: '0.1.0' } };
await validateUnifiedCapture(legacy, inputs, identity);
const typedMetadata = structuredClone(capture);
typedMetadata.document = new Map([[1n, 'document metadata']]);
typedMetadata.metadata_changes.id = new Map([[2n, 'record metadata']]);
await validateUnifiedCapture(typedMetadata, inputs, identity);
const typedResolved = resolveUnified([{ ...identity, bytes: new Uint8Array(), document: typedMetadata }], '0.1.0').get('id');
assert.equal(typedResolved.value.document.get(1n), 'document metadata');
const materialized = materializeUnifiedRecord(inputs.cases.id, typedResolved, 'sample');
assert.equal(materialized.get(1n), 'document metadata');
assert.equal(materialized.get('cases')['UNIFIED.1'].get(2n), 'record metadata');
const metadata = { ...identity, bytes: new Uint8Array(), document: capture };
assert.throws(() => resolveUnified([{ ...metadata, document: { ...capture, changes: {}, metadata_changes: { id: {} } } }], '0.1.0'), /no observation/);
assert.throws(() => resolveUnified([{ ...metadata, document: { ...capture, changes: {}, document_overrides: { id: {} } } }], '0.1.0'), /no observation/);
const first = { ...metadata, document: { ...capture, metadata_changes: { id: { reason: 'old' } }, document_overrides: { id: { parser_path: 'path' } } } };
const second = { ...metadata, version: '0.2.0', document: { ...capture, changes: {}, metadata_changes: { id: {} }, document_overrides: {} } };
const resolved = resolveUnified([first, second], '0.2.0').get('id').value.document;
assert.equal(Object.hasOwn(resolved, 'record_metadata'), false);
assert.equal(resolved.parser_path, 'path');
assert.equal(resolved.inherited_from, '0.1.0');
const tombstone = { ...second, document: { ...second.document, changes: { id: { absent: true } }, metadata_changes: {} } };
assert.equal(resolveUnified([first, tombstone], '0.2.0').size, 0);
console.log('Unified shape, stimulus, provenance, metadata, and tombstone contracts passed');
