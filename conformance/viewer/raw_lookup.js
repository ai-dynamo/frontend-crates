// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { sha256 } from './data.js';

export async function rawResource(catalog, store, path) {
  const lookup = async key => {
    const bucket = (await sha256(new TextEncoder().encode(key))).slice(0, 2);
    const descriptor = catalog.transport.raw_buckets[bucket];
    if (!descriptor) throw new Error(`Original resource is not indexed: ${key}`);
    const index = await store.transport.index(descriptor);
    if (!Object.hasOwn(index.resources, key)) throw new Error(`Original resource is not indexed: ${key}`);
    return index.resources[key];
  };
  const entry = await lookup(path);
  if (entry.resource) return entry;
  if (typeof entry.path !== 'string' || typeof entry.member !== 'string') throw new Error('Invalid original alias');
  const target = await lookup(entry.path);
  if (!target.resource || target.member !== null) throw new Error('Invalid original alias target');
  return { resource: target.resource, member: entry.member };
}
