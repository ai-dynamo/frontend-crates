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

export async function parseSnapshot(bytes, resource) {
  if (resource.format !== 'typed-json-v1' || resource.compression !== 'gzip') throw new Error('unsupported viewer snapshot format');
  const payload = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(gunzipSync(bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes))));
  if (payload.schema !== 1 || payload.kind !== resource.kind || payload.family !== resource.family
      || payload.implementation !== resource.implementation || payload.version !== resource.version
      || payload.source_fingerprint !== resource.source_fingerprint || !/^[0-9a-f]{64}$/.test(payload.source_fingerprint)) throw new Error(`viewer snapshot identity mismatch: ${resource.path}`);
  if (payload.kind === 'resolved-capture') {
    if (!Array.isArray(payload.value) || payload.value[0] !== 'object' || !Array.isArray(payload.value[1])) throw new Error('invalid resolved snapshot mapping');
    const metadata = ['object', payload.value[1].map(([key, record]) => [key, ['object', [
      [['str', 'document'], typedField(typedField(record, 'value'), 'document')],
      [['str', 'member'], typedField(record, 'member')],
    ]]])];
    if (!/^[0-9a-f]{64}$/.test(resource.resolved_metadata_sha256 || '') || await sha256(new TextEncoder().encode(JSON.stringify(metadata) + '\n')) !== resource.resolved_metadata_sha256) throw new Error(`resolved snapshot metadata binding mismatch: ${resource.path}`);
  }
  return decodeTyped(payload.value);
}
