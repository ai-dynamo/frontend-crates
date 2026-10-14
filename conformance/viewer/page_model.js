// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { toolcallingPanel } from './toolcalling.js';
import { reasoningPanel } from './reasoning_panel.js';
import { unifiedPanel } from './unified_panel.js';
import { tabCandidates, fullLabel } from './candidates.js';
import { tabPresentation, legendHtml } from './presentation.js';
import { compareLegacyVersions } from './model.js';

export async function buildTab(catalog, reader, tabId, options = {}) {
  const spec = catalog.metadata.tabs.find((tab) => tab.id === tabId);
  if (!spec) throw new Error(`unknown tab: ${tabId}`);
  const tab = spec.suite === 'unified' ? await unifiedPanel(catalog, reader, options) : spec.suite === 'reasoning' ? await reasoningPanel(catalog, reader, spec, options) : await toolcallingPanel(catalog, reader, spec, options);
  return Object.assign(tab, tabPresentation(spec, catalog, tabCandidates(catalog, spec), options.sourceLink));
}

export function pageMetadata(catalog, { commitSha = null, sourceLink = null } = {}) {
  if (commitSha !== null && !/^[0-9a-f]{40}$/.test(commitSha)) throw new Error('commitSha must be an immutable full SHA');
  const streamSpec = catalog.metadata.tabs.find(tab => tab.id === 'tab-toolcalling-streamv1');
  const latest = streamSpec ? tabCandidates(catalog, streamSpec).find(descriptor => descriptor.control.default_bucket === 'A') : null;
  const parserNi = {};
  if (latest) {
    const version = latest.source.version;
    const families = [...new Set(catalog.sources.archives.filter((resource) => resource.path.startsWith('conformance/fixtures/toolcalling/fixtures-stream-v1/dynamo_v2-') && compareLegacyVersions(resource.path.split('/').at(-1).slice('dynamo_v2-'.length, -7), version) <= 0).flatMap((resource) => resource.families))].sort();
    if (families.length) {
      const entry = { label: fullLabel('dynamo_v2', version, 'stream', catalog.metadata), families };
      parserNi[latest.control.key] = entry;
      parserNi[latest.control.key.replace('dynamo_v2-', 'dynamo_v2-s-')] = entry;
    }
  }
  const markers = Object.fromEntries(Object.entries(catalog.metadata.markers).map(([family, declaration]) => [family, Object.fromEntries(['pairs', 'singletons', 'opaque'].filter((key) => declaration[key]?.length).map((key) => [key, declaration[key]]))]).filter(([, declaration]) => Object.keys(declaration).length));
  return { schema: 1, meta: { title: 'Dynamo Parser v2 Conformance Table', sha: commitSha, short_sha: commitSha?.slice(0, 12) || '' },
    parser_ni: parserNi, legend_html: legendHtml(catalog.metadata, sourceLink), family_markers: markers };
}

export async function buildPage(catalog, reader, options = {}) {
  const tabs = [];
  for (const spec of catalog.metadata.tabs) tabs.push(await buildTab(catalog, reader, spec.id, options));
  return { ...pageMetadata(catalog, options), tabs };
}
