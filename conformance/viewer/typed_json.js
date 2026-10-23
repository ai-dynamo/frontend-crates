// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { yamlTemporal } from './values.js';
import { sameMappingKey } from './mappings.js';

export function decodeBase64(value, errorMessage = 'invalid typed binary') {
  if (typeof value !== 'string' || value.length % 4 !== 0 || /[^A-Za-z0-9+/=]/.test(value)) throw new Error(errorMessage);
  const padding = value.indexOf('=');
  if (padding !== -1 && padding !== value.length - 1
      && !(padding === value.length - 2 && value.endsWith('=='))) throw new Error(errorMessage);
  // Scan alphabet and padding separately: grouped repetition exhausts V8's stack on large blobs.
  return Uint8Array.from(atob(value), char => char.charCodeAt(0));
}

function mappingKeyFingerprint(key) {
  if (typeof key === 'boolean') return `integer:${key ? 1 : 0}`;
  if (typeof key === 'bigint') return `integer:${key}`;
  if (typeof key === 'number') {
    if (Number.isInteger(key)) return `integer:${BigInt(key)}`;
    if (Number.isNaN(key)) return 'number:nan';
    return `number:${key}`;
  }
  if (typeof key === 'string') return `string:${key}`;
  if (key instanceof Uint8Array) return `binary:${Array.from(key).join(',')}`;
  if (key instanceof Uint16Array) return `temporal:${Array.from(key).join(',')}`;
  if (key === null) return 'null';
  throw new Error('invalid typed mapping key');
}

export function decodeTyped(node) {
  if (!Array.isArray(node) || typeof node[0] !== 'string') throw new Error('invalid typed JSON node');
  const [type, value] = node;
  if (node.length !== (type === 'null' ? 1 : 2)) throw new Error('invalid typed JSON arity');
  switch (type) {
    case 'null': return null;
    case 'bool':
      if (typeof value !== 'boolean') throw new Error('invalid typed boolean');
      return value;
    case 'str':
      if (typeof value !== 'string') throw new Error('invalid typed string');
      return value;
    case 'int':
      if (typeof value !== 'string' || !/^-?(?:0|[1-9]\d*)$/.test(value)) throw new Error('invalid typed integer');
      return BigInt(value);
    case 'float': {
      if (typeof value !== 'string') throw new Error('invalid typed float');
      if (value === 'nan') return NaN;
      if (value === 'inf') return Infinity;
      if (value === '-inf') return -Infinity;
      if (!/^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?$/.test(value) || !Number.isFinite(Number(value))) throw new Error('invalid typed float');
      return Number(value);
    }
    case 'date': case 'datetime':
      if (typeof value !== 'string' || (type === 'date' ? !/^\d{4}-\d{2}-\d{2}$/.test(value) : !value.includes('T'))) throw new Error('invalid typed temporal');
      return yamlTemporal(value, type);
    case 'binary':
      return decodeBase64(value);
    case 'array':
      if (!Array.isArray(value)) throw new Error('invalid typed array');
      return value.map(decodeTyped);
    case 'object': {
      if (!Array.isArray(value)) throw new Error('invalid typed mapping');
      const pairs = [];
      const seenKeys = value.length > 8 ? new Set() : null;
      for (const pair of value) {
        if (!Array.isArray(pair) || pair.length !== 2) throw new Error('invalid typed mapping entry');
        const key = decodeTyped(pair[0]);
        if (key !== null && typeof key === 'object' && !(key instanceof Uint16Array) && !(key instanceof Uint8Array)) throw new Error('invalid typed mapping key');
        if (seenKeys) {
          const fingerprint = mappingKeyFingerprint(key);
          if (seenKeys.has(fingerprint) && pairs.some(([prior]) => sameMappingKey(prior, key))) throw new Error('duplicate typed mapping key');
          seenKeys.add(fingerprint);
        } else if (pairs.some(([prior]) => sameMappingKey(prior, key))) throw new Error('duplicate typed mapping key');
        pairs.push([key, decodeTyped(pair[1])]);
      }
      return pairs.every(([key]) => typeof key === 'string') ? Object.fromEntries(pairs) : new Map(pairs);
    }
    default: throw new Error(`unsupported typed JSON kind: ${type}`);
  }
}
