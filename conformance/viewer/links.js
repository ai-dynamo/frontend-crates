// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
const sourceAliases = {
  'conformance/toolcalling/fixtures-v1': 'conformance/fixtures/toolcalling/fixtures-batch-v1',
  'conformance/toolcalling/fixtures-stream-v1': 'conformance/fixtures/toolcalling/fixtures-stream-v1',
  'conformance/reasoning/fixtures': 'conformance/fixtures/reasoning/fixtures-v1',
};
export function createSourceLink(baseUrl, publication = null, catalogRevision = null) {
  if (publication && (publication.schema !== 1 || !/^[0-9a-f]{40}$/.test(publication.revision) || publication.catalog_revision !== catalogRevision)) {
    throw new Error('Publication metadata does not match the resource catalog');
  }
  return path => {
    const normalized = path.replace(/\/$/, '');
    if (!normalized || normalized.startsWith('/') || normalized.includes(':') || normalized.includes('\\') || normalized.split('/').some(part => !part || part === '.' || part === '..')) throw new Error(`Invalid source target: ${path}`);
    const target = sourceAliases[normalized] || normalized;
    if (publication) {
      const kind = /\.[a-z0-9]+$/i.test(target) ? 'blob' : 'tree';
      return `https://github.com/ai-dynamo/frontend-crates/${kind}/${publication.revision}/${target}`;
    }
    return new URL(target, baseUrl).href;
  };
}
