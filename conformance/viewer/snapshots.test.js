// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import assert from 'node:assert/strict';
import { gzipSync } from './vendor/fflate/esm/browser.js';
import { parseSnapshot } from './snapshots.js';

const resource = { path: 'family/capture.json.gz', format: 'typed-json-v1', compression: 'gzip', kind: 'inputs-and-golden', family: 'family', implementation: 'dynamo_v2', version: '1.0.0', source_fingerprint: 'a'.repeat(64) };
const envelope = { schema: 1, kind: resource.kind, family: resource.family, implementation: resource.implementation, version: resource.version, source_fingerprint: resource.source_fingerprint, value: ['object', [[['str', 'case'], ['int', '9007199254740993']]]] };
const encode = value => gzipSync(new TextEncoder().encode(JSON.stringify(value)));
assert.deepEqual(await parseSnapshot(encode(envelope), resource), { case: 9007199254740993n });
for (const field of ['kind', 'family', 'implementation', 'version', 'source_fingerprint', 'schema']) await assert.rejects(() => parseSnapshot(encode({ ...envelope, [field]: 'wrong' }), resource), /identity/);
await assert.rejects(() => parseSnapshot(encode({ ...envelope, value: ['int', '1.5'] }), resource), /typed integer/);
await assert.rejects(() => parseSnapshot(new Uint8Array([1, 2]), resource));
await assert.rejects(() => parseSnapshot(encode(envelope), { ...resource, format: 'unknown' }), /format/);
await assert.rejects(() => parseSnapshot(encode(envelope), { ...resource, compression: 'unknown' }), /format/);
console.log('snapshot format, identity, fingerprint, malformed payload, and compression guards passed');
