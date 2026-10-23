// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { gunzipSync } from './vendor/fflate/esm/browser.js';
import { decodeTyped } from './typed_json.js';
import { sha256 } from './data.js';

function typedField(node, name) {
  if (!Array.isArray(node) || node[0] !== 'object' || !Array.isArray(node[1])) throw new Error('invalid snapshot metadata mapping');
  const fields = node[1].filter(pair => Array.isArray(pair) && pair[0]?.[0] === 'str' && pair[0]?.[1] === name);
  if (fields.length !== 1 || fields[0].length !== 2) throw new Error('invalid snapshot metadata field');
  return fields[0][1];
}

export async function parseSnapshot(bytes, resource, metrics = { decoding: 0, validation: 0 }) {
  if (resource.format !== 'typed-json-v1' || !['gzip', 'none'].includes(resource.compression) || (resource.compression === 'none' && resource.transport !== 'chunked-v2')) throw new Error('unsupported viewer snapshot format');
  const decodeStart = performance.now();
  const input = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  const payload = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(resource.compression === 'none' ? input : gunzipSync(input)));
  metrics.decoding += performance.now() - decodeStart;
  const validationStart = performance.now();
  const bundled = Array.isArray(resource.families);
  if (payload.schema !== 1 || payload.kind !== resource.kind || (bundled
      ? payload.family !== null || JSON.stringify(payload.families) !== JSON.stringify(resource.families)
      : payload.family !== resource.family)
      || payload.implementation !== resource.implementation || payload.version !== resource.version
      || payload.source_fingerprint !== resource.source_fingerprint || !/^[0-9a-f]{64}$/.test(payload.source_fingerprint)) throw new Error(`viewer snapshot identity mismatch: ${resource.path}`);
  if (payload.kind === 'resolved-capture') {
    if (!Array.isArray(payload.value) || payload.value[0] !== 'object' || !Array.isArray(payload.value[1])) throw new Error('invalid resolved snapshot mapping');
    const recordsMetadata = records => ['object', records[1].map(([key, record]) => [key, ['object', [
      [['str', 'document'], typedField(typedField(record, 'value'), 'document')],
      [['str', 'member'], typedField(record, 'member')],
    ]]])];
    const metadata = bundled
      ? ['object', payload.value[1].map(([family, records]) => [family, recordsMetadata(records)])]
      : recordsMetadata(payload.value);
    if (!/^[0-9a-f]{64}$/.test(resource.resolved_metadata_sha256 || '') || await sha256(new TextEncoder().encode(JSON.stringify(metadata) + '\n')) !== resource.resolved_metadata_sha256) throw new Error(`resolved snapshot metadata binding mismatch: ${resource.path}`);
  }
  metrics.validation += performance.now() - validationStart;
  const typedStart = performance.now();
  const result = decodeTyped(payload.value);
  metrics.decoding += performance.now() - typedStart;
  return result;
}
