// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { HttpResourceStore } from './data.js';
import { FixtureResources } from './resources.js';
import { buildTab, pageMetadata } from './page_model.js';
import { tabCandidates } from './candidates.js';
import { tabPresentation } from './presentation.js';
import { stringifyValue } from './values.js';
import { SelectionLoader } from './selection.js';
import { selectedCellOutcome } from './selection_comparison.js';
import { createSourceLink } from './links.js';
import { columnKeysFor } from './column_state.js';
import { EngineSelection, mountEngineControls } from './engine_selection.js';
import '../utils/src/assets/colorize.js';
import '../utils/src/assets/conformance_view.js';
import '../utils/src/assets/conformance.js';

const root = document.querySelector('#viewer');
const baseUrl = new URL('.', document.baseURI);
window.__formatConformanceValue = stringifyValue;

function resourceFor(catalog, path) {
  const direct = catalog.sources.unified.find(resource => resource.path === path);
  if (direct) return { resource: direct, member: null };
  const logical = path.replace(/^conformance\//, '');
  for (const resource of catalog.sources.archives) {
    const member = resource.members.find(member => member.kind === 'file' && (member.path === logical || member.path === path));
    if (member) return { resource, member: member.path };
  }
  throw new Error(`Original YAML resource is not in the catalog: ${path}`);
}

async function rawView(catalog, reader, path) {
  const { resource, member } = resourceFor(catalog, path);
  const bytes = member ? (await reader.archive(resource)).get(member) : await reader.store.read(resource);
  if (!bytes) throw new Error(`Missing archive member: ${member}`);
  const header = document.createElement('h1'); header.textContent = path;
  const download = document.createElement('a'); download.textContent = 'Download original YAML';
  const objectUrl = URL.createObjectURL(new Blob([bytes], { type: 'application/yaml' }));
  download.href = objectUrl; download.download = path.split('/').at(-1);
  const back = document.createElement('a'); back.textContent = 'Back to matrix';
  const url = new URL(location.href); url.searchParams.delete('raw'); back.href = url.href;
  const pre = document.createElement('pre'); pre.textContent = new TextDecoder('utf-8', { fatal: true }).decode(bytes);
  root.replaceChildren(header, download, document.createTextNode(' · '), back, pre);
  window.addEventListener('pagehide', () => URL.revokeObjectURL(objectUrl), { once: true });
}

function fixtureLinks(tab) {
  for (const row of tab.rows) for (const cell of Object.values(row.cells || {})) {
    if (cell.fixture_href) {
      const url = new URL(location.href); url.searchParams.set('raw', cell.fixture_href);
      cell.fixture_href = url.href;
    }
  }
  return tab;
}

async function initialize() {
  const response = await fetch(new URL('conformance/viewer-unified-catalog.json', baseUrl), { cache: 'no-store' });
  if (!response.ok) throw new Error(`Catalog fetch failed (${response.status})`);
  let catalog = await response.json();
  if (catalog.schema !== 1 || !Array.isArray(catalog.sources?.archives)) throw new Error('Unsupported conformance catalog');
  const reader = new FixtureResources(catalog, new HttpResourceStore({ baseUrl }));
  window.addEventListener('pagehide', () => reader.dispose(), { once: true });
  const params = new URLSearchParams(location.search);
  if (params.has('raw')) {
    if (!catalog.sources.unified.some(resource => resource.path === params.get('raw'))) {
      const fullResponse = await fetch(new URL('conformance/viewer-catalog.json', baseUrl), { cache: 'no-store' });
      if (!fullResponse.ok) throw new Error(`Catalog fetch failed (${fullResponse.status})`);
      const full = await fullResponse.json();
      if (full.revision !== catalog.revision) throw new Error('Legacy catalog revision mismatch');
      catalog = full; reader.catalog = full;
    }
    await rawView(catalog, reader, params.get('raw')); return;
  }
  const active = catalog.metadata.tabs.find(tab => tab.id === 'tab-unified');
  const descriptors = tabCandidates(catalog, active, { golden: true });
  const owner = new EngineSelection(descriptors, new URL(location.href));
  let publication = null;
  let sourceLink = createSourceLink(baseUrl, null, catalog.revision);
  const page = pageMetadata(catalog, { commitSha: null, sourceLink: path => `source:${path}` });
  window.__PARSER_NI = page.parser_ni;
  page.tabs = catalog.metadata.tabs.map(spec => ({
    ...tabPresentation(spec, catalog, tabCandidates(catalog, spec), path => `source:${path}`),
    id: spec.id, kind: spec.suite === 'toolcalling' ? 'toolcalling' : spec.suite, mode: spec.mode === 'stream' && spec.suite === 'toolcalling' ? 'streamv1' : spec.mode,
    active: spec.id === active.id, no_parser_col: spec.suite === 'unified',
    rows: [], columns: [], column_groups: [], stats: {}, glossary: [],
    candidates: tabCandidates(catalog, spec, { golden: true }).map(item => item.control),
  }));
  const resolveSources = () => root.querySelectorAll('a').forEach(anchor => {
    if (anchor.getAttribute('href')?.startsWith('source:')) anchor.dataset.sourcePath = anchor.getAttribute('href').slice(7);
    if (anchor.dataset.sourcePath) anchor.href = sourceLink(anchor.dataset.sourcePath);
  });
  window.ConformanceView.initialize(page, root);
  resolveSources();
  root.dataset.catalogRevision = catalog.revision;
  let controller;
  const suspendedPins = new Map();
  const loader = new SelectionLoader({
    async build(tabId, selection) {
      const generation = loader.generation;
      await reader.prepareUnified(descriptors.filter(item => selection.selected.includes(item.control.key)));
      if (generation !== loader.generation) throw new Error('Obsolete fixture selection');
      return buildTab(catalog, reader, tabId, { ...selection, sourceLink: path => `source:${path}` });
    },
    state(tabId, state, error) {
      if (error) window.__conformanceError = error.stack;
      const panel = document.getElementById(tabId);
      panel.dataset.loadState = state;
      if (state !== 'ready') {
        const pin = controller.disposePanel(panel);
        if (pin) suspendedPins.set(tabId, pin);
        panel.querySelectorAll('td.cell').forEach(cell => {
          cell.classList.remove('cmp-eq', 'cmp-leak', 'cmp-na', 'cmp-nobase', 'cmp-donly');
          const marker = cell.querySelector('.marker-text'); if (marker) marker.textContent = '';
        });
        panel.querySelectorAll('[data-overview-count]').forEach(count => { count.textContent = '—'; });
        window.__conformanceModel = null;
      }
      panel.setAttribute('aria-busy', String(state === 'loading'));
      panel.querySelector('.load-status')?.remove();
      if (state !== 'ready') {
        const notice = document.createElement('div'); notice.className = 'load-status'; notice.setAttribute('role', state === 'error' ? 'alert' : 'status');
        notice.textContent = state === 'error' ? `Fixture load failed: ${error.message} ` : 'Loading selected fixtures…';
        if (state === 'error') {
          const retry = document.createElement('button'); retry.textContent = 'Retry'; retry.addEventListener('click', () => load(panel)); notice.appendChild(retry);
        }
        panel.appendChild(notice);
      }
    },
    paint(tabId, model) {
      const panel = document.getElementById(tabId), pin = suspendedPins.get(tabId) || controller.disposePanel(panel);
      suspendedPins.delete(tabId);
      window.ConformanceView.update(panel, fixtureLinks(model), page);
      resolveSources();
      panel.dataset.loadState = 'ready';
      controller.updatePanel(panel, pin);
      window.__conformanceModel = model;
      const generation = loader.generation;
      reader.prefetchUnified(owner.alternatives(), descriptors, () => generation === loader.generation);
    },
  });
  window.__conformanceLoader = loader;
  const load = panel => { if (panel.classList.contains('active')) return loader.select(panel.id, controller.selection(panel)); };
  controller = window.initializeConformanceController({ engineSelection: owner, columnKeys: columnKeysFor(catalog.metadata), cellOutcome: selectedCellOutcome, selectionChanged: load, tabChanged: id => { suspendedPins.clear(); return load(document.getElementById(id)); } });
  window.__conformanceController = controller;
  mountEngineControls(root.querySelector('.cmpctl'), owner, () => { controller.refreshPanel(controller.activePanel()); load(controller.activePanel()); });
  load(controller.activePanel());
  // Publication is optional and never gates controls or fixture resolution.
  fetch(new URL('conformance/viewer-publication.json', baseUrl), { cache: 'no-store' }).then(async response => {
    if (!response.ok) return;
    const metadata = await response.json();
    const upgraded = createSourceLink(baseUrl, metadata, catalog.revision);
    sourceLink = upgraded; publication = metadata; resolveSources();
  }).catch(() => {});

}

initialize().catch(error => {
  const alert = document.createElement('p'); alert.className = 'error'; alert.setAttribute('role', 'alert'); alert.textContent = error.message;
  const retry = document.createElement('button'); retry.textContent = 'Retry'; retry.addEventListener('click', () => location.reload());
  root.replaceChildren(alert, retry);
});
