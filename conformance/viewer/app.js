// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { rawResource } from './raw_lookup.js';
import { discoverHistory, loadStartupCatalog, startupMetrics, timedStartupPhase } from './startup.js';
import { FixtureResources } from './resources.js';
import { buildTab, pageMetadata } from './page_model.js';
import { tabCandidates } from './candidates.js';
import { tabPresentation } from './presentation.js';
import { stringifyValue } from './values.js';
import { SelectionLoader } from './selection.js';
import { selectedCellOutcome, selectedCaptureOutcome } from './selection_comparison.js';
import { createSourceLink } from './links.js';
import { columnKeysFor } from './column_state.js';
import '../utils/src/assets/colorize.js';
import '../utils/src/assets/conformance_view.js';
import '../utils/src/assets/conformance.js';
import '../utils/src/assets/comparison_layout.js';

const root = document.querySelector('#viewer');
const baseUrl = new URL('.', document.baseURI);
window.__CONFORMANCE_INITIAL_SEARCH = location.search;
window.__formatConformanceValue = stringifyValue;
window.__conformanceCaptureOutcome = selectedCaptureOutcome;

function resourceFor(catalog, path) {
  const direct = [...catalog.sources.unified, ...(catalog.sources.original_unified || [])].find(resource => resource.path === path);
  if (direct) return { resource: direct, member: null };
  const logical = path.replace(/^conformance\//, '');
  for (const resource of catalog.sources.archives) {
    const member = resource.members.find(member => member.kind === 'file' && (member.path === logical || member.path === path));
    if (member) return { resource, member: member.path };
  }
  throw new Error(`Original YAML resource is not in the catalog: ${path}`);
}

async function rawView(catalog, reader, path) {
  const { resource, member } = catalog.transport?.schema === 2 ? await rawResource(catalog, reader.store, path) : resourceFor(catalog, path);
  const bytes = member ? (await reader.archive(resource)).get(member) : await reader.store.read(resource);
  if (!bytes) throw new Error(`Missing archive member: ${member}`);
  const header = document.createElement('h1'); header.textContent = path;
  const download = document.createElement('a'); download.textContent = 'Download original resource';
  const objectUrl = URL.createObjectURL(new Blob([bytes], { type: 'application/yaml' }));
  download.href = objectUrl; download.download = path.split('/').at(-1);
  const back = document.createElement('a'); back.textContent = 'Back to matrix';
  const url = new URL(location.href); url.searchParams.delete('raw'); back.href = url.href;
  const pre = document.createElement('pre'); pre.textContent = /\.(?:ya?ml|json|txt)$/.test(path) ? new TextDecoder('utf-8').decode(bytes) : `Binary resource (${bytes.length} bytes). Use Download to save the original bytes.`;
  root.replaceChildren(header, download, document.createTextNode(' · '), back, pre);
  window.addEventListener('pagehide', () => URL.revokeObjectURL(objectUrl), { once: true });
}

function fixtureLinks(tab, historyAvailable) {
  const rewrite = cell => {
    if (cell.fixture_href) {
      if (historyAvailable) {
        const url = new URL(location.href); url.searchParams.set('raw', cell.fixture_href);
        cell.fixture_href = url.href;
      } else cell.fixture_href = null;
    }
    for (const variant of cell.variants || []) rewrite(variant);
  };
  for (const row of tab.rows) for (const cell of Object.values(row.cells || {})) rewrite(cell);
  return tab;
}

async function initialize() {
  window.__conformanceStartup = startupMetrics;
  const params = new URLSearchParams(location.search);
  const startup = await timedStartupPhase('catalog', () => loadStartupCatalog(baseUrl, params));
  let catalog = startup.catalog;
  const embeddedIdentities = startup.embeddedCaptures ? new Set(startup.embeddedCaptures.map(capture => capture.join('@'))) : null;
  let historyAvailable = !embeddedIdentities && ['http:', 'https:'].includes(baseUrl.protocol);
  root.dataset.historyAvailable = embeddedIdentities ? 'checking' : String(historyAvailable);
  let historyDiscovery;
  const readHistory = () => historyDiscovery ||= discoverHistory(baseUrl, catalog);
  if (params.has('raw') && embeddedIdentities) {
    historyAvailable = await readHistory();
    startup.store.allowExternalResources = historyAvailable;
    if (!historyAvailable) params.delete('raw');
  }
  const reader = new FixtureResources(catalog, startup.store);
  window.addEventListener('pagehide', () => reader.dispose(), { once: true });
  if (params.has('raw')) {
    if (catalog.transport?.schema !== 2 && ![...catalog.sources.unified, ...(catalog.sources.original_unified || [])].some(resource => resource.path === params.get('raw'))) {
      const fullResponse = await fetch(new URL('conformance/viewer-catalog.json', baseUrl), { cache: 'no-store' });
      if (!fullResponse.ok) throw new Error(`Catalog fetch failed (${fullResponse.status})`);
      const full = await fullResponse.json();
      if (full.revision !== catalog.revision && full.revision !== catalog.source_revision) throw new Error('Legacy catalog revision mismatch');
      catalog = full; reader.catalog = full;
    }
    await rawView(catalog, reader, params.get('raw')); return;
  }
  if (params.get('tab') && params.get('tab') !== 'tab-unified') {
    const url = new URL(location.href);
    url.searchParams.set('tab', 'tab-unified');
    history.replaceState(null, '', url.toString());
  }
  const active = catalog.metadata.tabs.find(tab => tab.id === 'tab-unified');
  const descriptors = tabCandidates(catalog, active, { golden: true });
  const publicationElement = document.getElementById('conformance-publication');
  let publication = publicationElement ? JSON.parse(publicationElement.textContent) : null;
  let sourceLink = createSourceLink(baseUrl, publication, catalog.revision);
  const page = pageMetadata(catalog, { commitSha: null, sourceLink: path => `source:${path}` });
  page.legacy_candidate_identities = catalog.metadata.legacy_candidate_identities || {};
  window.__PARSER_NI = page.parser_ni;
  page.tabs = catalog.metadata.tabs.map(spec => ({
    ...tabPresentation(spec, catalog, tabCandidates(catalog, spec), path => `source:${path}`),
    id: spec.id, kind: spec.suite === 'toolcalling' ? 'toolcalling' : spec.suite, mode: spec.mode === 'stream' && spec.suite === 'toolcalling' ? 'streamv1' : spec.mode,
    active: spec.id === active.id, no_parser_col: spec.suite === 'unified',
    comparison_defaults: catalog.metadata.unified_display?.comparison_defaults,
    comparison_slot_engines: catalog.metadata.unified_display?.comparison_slot_engines,
    comparison_languages: catalog.metadata.unified_display?.comparison_languages,
    rows: [], columns: [], column_groups: [], stats: {}, glossary: [],
    candidates: tabCandidates(catalog, spec, { golden: true }).map(item => ({ ...item.control, source_id: item.source.id, identity: `${item.source.implementation}@${item.source.version}`, language: item.source.language })),
  }));
  window.__CONFORMANCE_PAGE = page;
  const resolveSources = () => root.querySelectorAll('a').forEach(anchor => {
    if (anchor.getAttribute('href')?.startsWith('source:')) anchor.dataset.sourcePath = anchor.getAttribute('href').slice(7);
    if (anchor.dataset.sourcePath) anchor.href = sourceLink(anchor.dataset.sourcePath);
  });
  window.ConformanceView.initialize(page, root);
  const showGeneration = metadata => {
    if (['generated_on', 'command', 'output'].every(key => typeof metadata[key] === 'string' && metadata[key])) {
      window.ConformanceView.updateGeneration(root, metadata);
    }
  };
  if (publication) showGeneration(publication);
  resolveSources();
  root.dataset.catalogRevision = catalog.revision;
  let controller;
  let slotOwner;
  const loader = new SelectionLoader({
    async build(tabId, selection) {
      const generation = loader.generation;
      // Cached selections can rebuild synchronously; let the loading notice paint first.
      await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
      if (generation !== loader.generation) throw new Error('Obsolete fixture selection');
      await timedStartupPhase('resourcePreparation', () => reader.prepareUnified(descriptors.filter(item => selection.selected.includes(item.control.key))));
      if (generation !== loader.generation) throw new Error('Obsolete fixture selection');
      return timedStartupPhase('modelConstruction', () => buildTab(catalog, reader, tabId, { ...selection, sourceLink: path => `source:${path}` }));
    },
    state(tabId, state, error, selection) {
      if (error) window.__conformanceError = error.stack;
      const panel = document.getElementById(tabId);
      const scroll = { x: window.scrollX, y: window.scrollY };
      panel.dataset.loadState = state;
      panel.setAttribute('aria-busy', String(state === 'loading'));
      panel.querySelector('.load-status')?.remove();
      if (state !== 'ready') {
        const notice = document.createElement('div'); notice.className = 'load-status'; notice.setAttribute('role', state === 'error' ? 'alert' : 'status');
        const labels = selection?.selected.map(key => descriptors.find(item => item.control.key === key)?.control.label || key);
        notice.textContent = state === 'error' ? `Fixture load failed: ${error.message} ` : `Loading... ${labels?.join(', ') || 'selected fixtures'}`;
        if (state === 'error') {
          const retry = document.createElement('button'); retry.textContent = 'Retry'; retry.addEventListener('click', () => load(panel)); notice.appendChild(retry);
        }
        panel.querySelector('.capture-slots').appendChild(notice);
      }
      controller.refreshPopupLayout();
      window.scrollTo(scroll.x, scroll.y);
    },
    paint(tabId, model, selection) {
      const started = performance.now();
      const panel = document.getElementById(tabId);
      // Build the replacement before releasing the current popup or model. No
      // asynchronous work runs between disposal and the complete replacement.
      const commit = window.ConformanceView.prepareUpdate(panel, fixtureLinks(model, historyAvailable), page);
      const scroll = { x: window.scrollX, y: window.scrollY };
      const pin = controller.disposePanel(panel);
      panel._comparisonSelection = { keys: [...selection.selected], reference: selection.reference };
      panel.querySelector('.load-status')?.remove();
      commit();
      resolveSources();
      panel.dataset.hasModel = 'true';
      controller.updatePanel(panel, pin);
      window.__conformanceModel = model;
      window.scrollTo(scroll.x, scroll.y);
      startupMetrics.phases.domUpdate = performance.now() - started;
      requestAnimationFrame(() => requestAnimationFrame(() => {
        startupMetrics.paintedReady = performance.now();
        startupMetrics.painted = true;
      }));
    },
  });
  window.__conformanceLoader = loader;
  const load = panel => { if (panel.classList.contains('active')) return loader.select(panel.id, controller.selection(panel)); };
  controller = window.initializeConformanceController({ engineSelection: { selection: () => slotOwner?.selection() || { reference: null, selected: [] } }, columnKeys: columnKeysFor(catalog.metadata), cellOutcome: selectedCellOutcome, selectionChanged: load, tabChanged: id => load(document.getElementById(id)) });
  window.__conformanceController = controller;
  const panel = document.getElementById('tab-unified');
  panel.querySelector('.cmpctl')?.replaceChildren();
  slotOwner = window.mountConformanceCaptureSlots(page, panel, () => {
    if (!slotOwner) return;
    load(panel);
  }, { availableIdentities: historyAvailable ? null : embeddedIdentities });
  const firstLoad = load(controller.activePanel());
  async function discoverOlderResults() {
    await firstLoad;
    if (!embeddedIdentities) return;
    historyAvailable = await readHistory();
    startup.store.allowExternalResources = historyAvailable;
    root.dataset.historyAvailable = String(historyAvailable);
    if (historyAvailable) slotOwner.setAvailableIdentities(null);
    else {
      const notice = document.createElement('p');
      notice.className = 'capture-slots-notice';
      notice.textContent = 'Standalone report: latest embedded releases only. Older results and original YAML are unavailable here.';
      panel.querySelector('.capture-slots').appendChild(notice);
    }
  }
  discoverOlderResults().catch(error => {
    window.__conformanceError = error.stack;
    const notice = document.createElement('p'); notice.className = 'error'; notice.textContent = error.message;
    panel.querySelector('.capture-slots').appendChild(notice);
  });
  // Publication is optional and never gates controls or fixture resolution.
  if (!publication) fetch(new URL('conformance/viewer-publication.json', baseUrl), { cache: 'no-store' }).then(async response => {
    if (!response.ok) return;
    const metadata = await response.json();
    const upgraded = createSourceLink(baseUrl, metadata, metadata.source_checkout ? catalog.source_revision || catalog.revision : catalog.revision);
    sourceLink = metadata.source_checkout ? createSourceLink(baseUrl, null, catalog.revision) : upgraded;
    publication = metadata; resolveSources();
    showGeneration(metadata);
  }).catch(() => {
    // A checkout's sidecar can belong to an older catalog; keep the explicit
    // checkout explanation rather than attributing this view to that generation.
  });

}

initialize().catch(error => {
  const alert = document.createElement('p'); alert.className = 'error'; alert.setAttribute('role', 'alert'); alert.textContent = error.message;
  const retry = document.createElement('button'); retry.textContent = 'Retry'; retry.addEventListener('click', () => location.reload());
  root.replaceChildren(alert, retry);
});
