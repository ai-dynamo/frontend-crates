// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import assert from 'node:assert/strict';
import { gzipSync } from './vendor/fflate/esm/browser.js';
import { parseSnapshot } from './snapshots.js';
import { sha256 } from './data.js';

const resource = { path: 'family/capture.json.gz', format: 'typed-json-v1', compression: 'gzip', kind: 'inputs-and-golden', family: 'family', implementation: 'dynamo_v2', version: '1.0.0', source_fingerprint: 'a'.repeat(64) };
const envelope = { schema: 1, kind: resource.kind, family: resource.family, implementation: resource.implementation, version: resource.version, source_fingerprint: resource.source_fingerprint, value: ['object', [[['str', 'case'], ['int', '9007199254740993']]]] };
const encode = value => gzipSync(new TextEncoder().encode(JSON.stringify(value)));
assert.deepEqual(await parseSnapshot(encode(envelope), resource), { case: 9007199254740993n });
const canonicalBytes = value => new TextEncoder().encode(JSON.stringify(value) + '\n');
const canonicalResource = { ...resource, compression: 'none', transport: 'chunked-v2' };
assert.deepEqual(await parseSnapshot(canonicalBytes(envelope), canonicalResource), { case: 9007199254740993n });
await assert.rejects(() => parseSnapshot(canonicalBytes(envelope), { ...canonicalResource, transport: undefined }), /format/);
for (const field of ['kind', 'family', 'implementation', 'version', 'source_fingerprint', 'schema']) await assert.rejects(() => parseSnapshot(canonicalBytes({ ...envelope, [field]: 'wrong' }), canonicalResource), /identity/);
for (const field of ['kind', 'family', 'implementation', 'version', 'source_fingerprint', 'schema']) await assert.rejects(() => parseSnapshot(encode({ ...envelope, [field]: 'wrong' }), resource), /identity/);
await assert.rejects(() => parseSnapshot(encode({ ...envelope, value: ['int', '1.5'] }), resource), /typed integer/);
await assert.rejects(() => parseSnapshot(new Uint8Array([1, 2]), resource));
await assert.rejects(() => parseSnapshot(encode(envelope), { ...resource, format: 'unknown' }), /format/);
await assert.rejects(() => parseSnapshot(encode(envelope), { ...resource, compression: 'unknown' }), /format/);

const groupedResource = { ...resource, path: 'bundles/dynamo_v2-1.0.0.json.gz', kind: 'resolved-capture', family: null,
  families: ['sample'], record_contract: 'stream', resolved_metadata_sha256: '' };
const document = ['object', []];
const record = ['object', [
  [['str', 'value'], ['object', [[['str', 'document'], document]]]],
  [['str', 'member'], ['str', 'source.yaml']],
]];
const records = ['object', [[['str', 'one'], record]]];
const groupedValue = ['object', [[['str', 'sample'], records]]];
const caseMetadataValue = ['object', [
  [['str', 'document'], document], [['str', 'member'], ['str', 'source.yaml']],
]];
const caseMetadata = ['object', [[['str', 'one'], caseMetadataValue]]];
const groupedMetadata = ['object', [[['str', 'sample'], caseMetadata]]];
groupedResource.resolved_metadata_sha256 = await sha256(new TextEncoder().encode(JSON.stringify(groupedMetadata) + '\n'));
const groupedEnvelope = { ...envelope, kind: 'resolved-capture', family: null, families: ['sample'], value: groupedValue };
assert.deepEqual(await parseSnapshot(encode(groupedEnvelope), groupedResource), { sample: { one: { value: { document: {} }, member: 'source.yaml' } } });
assert.deepEqual(await parseSnapshot(canonicalBytes(groupedEnvelope), { ...groupedResource, compression: 'none', transport: 'chunked-v2' }), { sample: { one: { value: { document: {} }, member: 'source.yaml' } } });
await assert.rejects(() => parseSnapshot(canonicalBytes(groupedEnvelope), { ...groupedResource, compression: 'none', transport: 'chunked-v2', resolved_metadata_sha256: 'b'.repeat(64) }), /metadata binding/);
await assert.rejects(() => parseSnapshot(encode({ ...groupedEnvelope, families: ['other'] }), groupedResource), /identity/);
await assert.rejects(() => parseSnapshot(encode(groupedEnvelope), { ...groupedResource, resolved_metadata_sha256: 'b'.repeat(64) }), /metadata binding/);
console.log('snapshot format, identity, fingerprint, malformed payload, and compression guards passed');
