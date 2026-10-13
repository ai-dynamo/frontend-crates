// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { sameValue } from './values.js';

export function isMapping(value) {
  return value instanceof Map || (value !== null && typeof value === 'object' && [Object.prototype, null].includes(Object.getPrototypeOf(value)));
}
export function mappingEntries(value) {
  return value instanceof Map ? [...value] : isMapping(value) ? Object.entries(value) : [];
}
export function hasField(value, key) {
  return value instanceof Map ? [...value.keys()].some((prior) => sameMappingKey(prior, key)) : isMapping(value) && typeof key === 'string' && Object.hasOwn(value, key);
}
export function getField(value, key, fallback = undefined) {
  if (value instanceof Map) {
    const entry = [...value].find(([prior]) => sameMappingKey(prior, key));
    return entry ? entry[1] : fallback;
  }
  return hasField(value, key) ? value[key] : fallback;
}
export function setField(value, key, item) {
  if (value instanceof Map) {
    const entry = [...value].find(([prior]) => sameMappingKey(prior, key));
    value.set(entry ? entry[0] : key, item);
  } else {
    if (typeof key !== 'string') throw new Error('typed key requires a Map');
    Object.defineProperty(value, key, { value: item, enumerable: true, writable: true, configurable: true });
  }
}
export function deleteField(value, key) {
  if (value instanceof Map) {
    const entry = [...value].find(([prior]) => sameMappingKey(prior, key));
    if (entry) value.delete(entry[0]);
  }
  else if (isMapping(value)) delete value[key];
}
export function mergeMappings(left, right) {
  const result = left instanceof Map || right instanceof Map ? new Map(mappingEntries(left)) : Object.fromEntries(mappingEntries(left));
  for (const [key, value] of mappingEntries(right)) {
    setField(result, key, value);
  }
  return result;
}
export function ensureMappingField(owner, key) {
  if (!hasField(owner, key)) setField(owner, key, {});
  const value = getField(owner, key);
  if (!isMapping(value)) throw new Error(`${key} must be a mapping`);
  return value;
}

// Python dictionaries and legacy streamed call indices identify 1, 1.0 and
// true as the same key. Argument values retain their separate scalar kinds.
export function mappingKeyIdentity(value) {
  if (typeof value === 'boolean') return value ? 1n : 0n;
  if (typeof value === 'number' && Number.isFinite(value) && Number.isInteger(value)) return BigInt(value);
  return value;
}
export function sameMappingKey(left, right) { return sameValue(mappingKeyIdentity(left), mappingKeyIdentity(right)); }

export function mappingKeyText(value) {
  if (value === null) return 'None';
  if (typeof value === 'boolean') return value ? 'True' : 'False';
  if (typeof value === 'number' && Number.isInteger(value)) return `${value}.0`;
  return String(value);
}
