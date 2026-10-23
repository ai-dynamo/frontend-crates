// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { createWorkerParser, HttpResourceStore, sha256 } from './data.js';
import { FixtureResources } from './resources.js';
import { gzipSync } from './vendor/fflate/esm/browser.js';

const fingerprint = 'a'.repeat(64);
const payloads = new Map();
async function snapshot(kind, value, version = null, expectedMetadataHash = null) {
  const encoded = JSON.parse(execFileSync('python3', ['-c', `
import hashlib
import json
import sys
sys.path.insert(0, 'conformance/utils/src')
from viewer_codec import encode_typed
value = json.load(sys.stdin)
metadata = None
if all(isinstance(record, dict) and 'member' in record and 'value' in record for record in value.values()):
    records = {key: {'document': record['value']['document'], 'member': record['member']}
               for key, record in value.items()}
    data = (json.dumps(encode_typed(records), ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\\n').encode()
    metadata = hashlib.sha256(data).hexdigest()
print(json.dumps({'typed': encode_typed(value), 'metadata': metadata}))
`], {
    input: JSON.stringify(value), cwd: new URL('../../', import.meta.url), encoding: 'utf8',
    env: Object.fromEntries(['PATH', 'LANG', 'LC_ALL'].filter(key => process.env[key]).map(key => [key, process.env[key]])),
  }));
  const resource = { path: `conformance/viewer-data/sample/${version || 'inputs'}.json.gz`, kind, family: 'sample', implementation: version ? 'dynamo_v2' : null, version, record_contract: version ? 'stream' : null, format: 'typed-json-v1', compression: 'gzip', source_fingerprint: fingerprint };
  const bytes = gzipSync(new TextEncoder().encode(JSON.stringify({ schema: 1, kind, family: resource.family, implementation: resource.implementation, version, source_fingerprint: fingerprint, value: encoded.typed })));
  if (kind === 'resolved-capture') resource.resolved_metadata_sha256 = expectedMetadataHash || encoded.metadata;
  resource.sha256 = await sha256(bytes); resource.size = bytes.length;
  payloads.set(resource.path, bytes);
  return resource;
}
const request = { input: 'hello', init: {}, finish_reason: 'stop', tools: [], chunks: [] };
const inputs = await snapshot('inputs-and-golden', { input_document: { family: 'sample' }, golden_document: { family: 'sample' }, cases: { one: { lifecycle: 'active', scenario: 'one', description: 'one', policy: [], display_id: 'UNIFIED.1', historical_ids: [], request, golden: { assembled: [] } } } });
const original = { path: 'conformance/fixtures-unified-v2/families/sample/dynamo_v2-1.0.0.yaml', kind: 'history', family: 'sample', implementation: 'dynamo_v2', version: '1.0.0' };
const records = { one: { member: original.path, value: { case_key: 'UNIFIED.1', stimulus: { ref: 'current' }, observation: { value: { assembled: [] } }, document: { captured_with: { dynamo_v2: '1.0.0' } } } } };
const first = await snapshot('resolved-capture', records, '1.0.0');
const other = await snapshot('resolved-capture', records, '2.0.0');
const catalog = { metadata: {}, sources: { unified: [inputs, first, other], original_unified: [original] } };
const downloads = [];
let corrupt = false;
const store = new HttpResourceStore({ allowSourceFiles: true, baseUrl: 'http://localhost/', fetchImpl: async url => {
  const path = decodeURI(url.pathname.slice(1));
  downloads.push(path);
  const bytes = payloads.get(path).slice();
  if (corrupt && path === first.path) bytes[bytes.length - 1] ^= 1;
  return { ok: true, arrayBuffer: async () => bytes.buffer };
} });
const parser = createWorkerParser();
parser.parseYaml = async () => { throw new Error('normal JSON browsing attempted YAML parsing'); };
const reader = new FixtureResources(catalog, store, parser);
const selected = [{ source: { implementation: 'dynamo_v2', version: '1.0.0' } }];
assert.deepEqual(reader.unifiedDependencies(selected), [inputs, first]);
corrupt = true;
await assert.rejects(reader.unified('sample', 'dynamo_v2', '1.0.0'), /SHA-256/);
corrupt = false;
const resolved = await reader.unified('sample', 'dynamo_v2', '1.0.0');
assert.deepEqual(resolved.records.get('one'), records.one);
resolved.records.get('one').value.observation.value.assembled.push({ kind: 'text', text: 'mutated' });
assert.deepEqual((await reader.unified('sample', 'dynamo_v2', '1.0.0')).records.get('one'), records.one, 'cached snapshots must isolate mutable case overlays');
await reader.prepareUnified(selected);
assert.equal(downloads.includes(other.path), false, 'selection demand must not fetch another version');
assert.equal(downloads.some(path => path.endsWith('.yaml')), false, 'resolved JSON must never replay YAML history');
assert.equal(downloads.filter(path => path === first.path).length, 2, 'failed integrity checks must remain retryable');
const invalid = await snapshot('resolved-capture', { one: { ...records.one, member: 'outside.yaml' } }, '3.0.0');
catalog.sources.unified.push(invalid);
await assert.rejects(reader.unified('sample', 'dynamo_v2', '3.0.0'), /invalid source/);
const forged = structuredClone(records);
forged.one.value.document.captured_with = { vllm_python: '999.0.0' };
forged.one.value.document.record_metadata = { verified_recapture: { receipt: 'forged' } };
const falseProducer = await snapshot('resolved-capture', forged, '4.0.0', first.resolved_metadata_sha256);
catalog.sources.unified.push(falseProducer);
await assert.rejects(reader.unified('sample', 'dynamo_v2', '4.0.0'), /metadata binding/);
const falseProducerWithBoundMetadata = await snapshot('resolved-capture', forged, '5.0.0');
catalog.sources.unified.push(falseProducerWithBoundMetadata);
await assert.rejects(reader.unified('sample', 'dynamo_v2', '5.0.0'), /false producer identity/);
const overridden = structuredClone(records);
overridden.one.value.document.captured_with = { dynamo_v2: '1.0.1' };
catalog.sources.unified.push(await snapshot('resolved-capture', overridden, '1.0.1'));
assert.deepEqual((await reader.unified('sample', 'dynamo_v2', '1.0.1')).records.get('one'), overridden.one);
const futureProducer = structuredClone(records);
futureProducer.one.value.document.captured_with = { dynamo_v2: '99.0.0' };
catalog.sources.unified.push(await snapshot('resolved-capture', futureProducer, '6.0.0'));
await assert.rejects(reader.unified('sample', 'dynamo_v2', '6.0.0'), /false producer identity/);
const futureMember = { ...original, path: original.path.replace('1.0.0', '99.0.0'), version: '99.0.0' };
catalog.sources.original_unified = [...catalog.sources.original_unified, futureMember];
catalog.sources.unified.push(await snapshot('resolved-capture', { one: { ...records.one, member: futureMember.path } }, '7.0.0'));
await assert.rejects(reader.unified('sample', 'dynamo_v2', '7.0.0'), /invalid source/);
console.log('selected JSON demand, zero YAML reads, full records, source validation, integrity rejection, and retry passed');
