// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { HttpResourceStore, gunzipSync, sha256, resourceMetrics, boundedGunzip } from './data.js';
import { rawResource, FixtureResources } from './resources.js';
import { buildTab, pageMetadata, tabPresentation, columnKeysFor, createSourceLink } from './presentation.js';
import { tabCandidates, selectedCellOutcome, selectedCaptureOutcome } from './model.js';
import { stringifyValue } from './values.js';

const decoder = new TextDecoder('utf-8', { fatal: true });
let developmentFallback = null;
export function configureDevelopmentCatalog(loader) { developmentFallback = loader; }

export const startupMetrics = { phases: {}, resources: resourceMetrics };

export async function timedStartupPhase(name, operation) {
  const start = performance.now();
  try { return await operation(); }
  finally { startupMetrics.phases[name] = (startupMetrics.phases[name] || 0) + performance.now() - start; }
}

export function validateStartupCatalog(catalog, { resolved = true } = {}) {
  if (!catalog || catalog.schema !== 1 || !Array.isArray(catalog.sources?.archives)) throw new Error('Unsupported conformance catalog');
  if (!Array.isArray(catalog.sources.unified) || catalog.sources.unified.some(resource =>
    resource.format !== 'typed-json-v1' && (resolved || resource.format !== undefined))) {
    throw new Error(resolved ? 'Browser data needs regeneration' : 'Unsupported conformance resource format');
  }
  return catalog;
}

async function readCatalog(text) {
  const catalog = validateStartupCatalog(JSON.parse(text));
  if (catalog.transport?.schema === 2) {
    const trailer = /,"runtime_sha256":"([a-f0-9]{64})"}\n$/.exec(text);
    const content = trailer ? text.slice(0, trailer.index) + '}\n' : '';
    if (!trailer || trailer[1] !== catalog.runtime_sha256 || await sha256(new TextEncoder().encode(content)) !== trailer[1]) throw new Error('Runtime catalog integrity mismatch');
  }
  return catalog;
}

function base64Bytes(text, size) {
  if (typeof text !== 'string' || !Number.isSafeInteger(size) || size < 0 || size > 64 * 1024 * 1024 || text.length !== 4 * Math.ceil(size / 3)) throw new Error('Invalid embedded resource length');
  const bytes = Uint8Array.from(atob(text), char => char.charCodeAt(0));
  if (bytes.length !== size) throw new Error('Invalid embedded resource length');
  return bytes;
}

export async function loadStartupCatalog(baseUrl, params, { fetchImpl = globalThis.fetch.bind(globalThis), embedded = globalThis.document?.getElementById('conformance-embedded') } = {}) {
  const store = new HttpResourceStore({ baseUrl, fetchImpl });
  const mode = params.get('startup') || 'compressed';
  startupMetrics.mode = mode;
  if (embedded) {
    const payload = JSON.parse(embedded.textContent);
    if (payload.schema !== 1 || !Array.isArray(payload.resources) || payload.resources.length > 4096) throw new Error('Unsupported embedded data');
    if (!Array.isArray(payload.captures) || payload.captures.some(capture => !Array.isArray(capture) || capture.length !== 2 || capture.some(value => typeof value !== 'string' || !value))) throw new Error('Invalid embedded capture identities');
    const inventory = payload.resources.map(({descriptor}) => [descriptor.path, descriptor.size, descriptor.sha256]);
    if (await sha256(new TextEncoder().encode(JSON.stringify(inventory) + '\n')) !== payload.resources_sha256) throw new Error('Embedded resource membership mismatch');
    const descriptor = payload.catalog;
    const bytes = base64Bytes(descriptor.bytes, descriptor.size);
    if (await sha256(bytes) !== descriptor.sha256 || !Number.isSafeInteger(descriptor.decoded_size) || descriptor.decoded_size > 64 * 1024 * 1024) throw new Error('Embedded catalog integrity mismatch');
    const catalog = await readCatalog(decoder.decode(boundedGunzip(bytes, descriptor.decoded_size)));
    if (catalog.transport?.schema !== 2) throw new Error('Unsupported embedded transport');
    let total = 0;
    for (const resource of payload.resources) {
      total += resource.descriptor.size;
      if (!Number.isSafeInteger(total) || total > 256 * 1024 * 1024) throw new Error('Embedded resources exceed memory limit');
      await store.seed(resource.descriptor, base64Bytes(resource.bytes, resource.descriptor.size));
    }
    // Release the DOM copy; only compressed bytes are retained. Envelope JSON
    // remains lazy and follows the normal recipe, hash and worker validation.
    embedded.remove?.();
    store.allowExternalResources = false;
    return { catalog, store, embeddedCaptures: payload.captures };
  }
  if ((globalThis.location?.protocol || baseUrl.protocol) === 'file:') throw new Error('Standalone HTML must contain embedded results');
  const response = await fetchImpl(new URL('conformance/viewer-resolved-catalog.json.gz', baseUrl), { cache: 'no-store' });
  if (response.ok) {
    const text = decoder.decode(gunzipSync(new Uint8Array(await response.arrayBuffer())));
    const catalog = await readCatalog(text);
    store.allowSourceFiles = catalog.transport?.schema !== 2 && Boolean(developmentFallback);
    return { catalog, store };
  }
  if (response.status !== 404 || !developmentFallback) throw new Error(`Catalog fetch failed (${response.status})`);
  store.allowSourceFiles = true;
  return { catalog: validateStartupCatalog(await developmentFallback(baseUrl, fetchImpl), { resolved: false }), store };
}

export async function loadPublication(baseUrl, { fetchImpl = globalThis.fetch.bind(globalThis) } = {}) {
  if (!['http:', 'https:'].includes(globalThis.location?.protocol || baseUrl.protocol)) return null;
  try {
    const response = await fetchImpl(new URL('conformance/viewer-publication.json', baseUrl), { cache: 'no-store' });
    return response.ok ? await response.json() : null;
  } catch (error) {
    if (error instanceof TypeError || error instanceof SyntaxError) return null;
    throw error;
  }
}

export async function discoverHistory(baseUrl, catalog, { fetchImpl = globalThis.fetch.bind(globalThis), publication } = {}) {
  if (!['http:', 'https:'].includes(globalThis.location?.protocol || baseUrl.protocol)) return false;
  const metadata = await (publication === undefined ? loadPublication(baseUrl, { fetchImpl }) : publication);
  if (metadata?.schema !== 1 || metadata.catalog_revision !== catalog.revision
      || metadata.generation_mode !== 'history') return false;
  const resolved = await fetchImpl(new URL('conformance/viewer-resolved-catalog.json.gz', baseUrl), { cache: 'no-store' });
  if (!resolved.ok) return false;
  const history = await readCatalog(decoder.decode(gunzipSync(new Uint8Array(await resolved.arrayBuffer()))));
  if (history.revision !== metadata.resolved_catalog_revision || history.capture_generation?.mode !== 'history'
      || JSON.stringify(history.capture_generation.current) !== JSON.stringify(catalog.capture_generation?.current)) {
    throw new Error('History catalog does not match embedded latest captures');
  }
  return history;
}

// Cache population may finish after navigation, but only the latest selection paints.
export class SelectionLoader {
  constructor({ build, paint, state }) { this.build = build; this.paint = paint; this.state = state; this.generation = 0; this.pending = 0; }
  async select(tabId, selection) {
    const generation = ++this.generation;
    this.pending++;
    this.state(tabId, 'loading', null, selection);
    try {
      const model = await this.build(tabId, selection);
      if (generation !== this.generation) return;
      this.paint(tabId, model, selection);
      this.state(tabId, 'ready');
    } catch (error) {
      if (generation === this.generation) this.state(tabId, 'error', error);
    } finally { this.pending--; }
  }
}

export async function startApp() {
  await import('../utils/src/assets/colorize.js');
  await import('../utils/src/assets/conformance_view.js');
  await import('../utils/src/assets/conformance.js');


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

function captureControls(catalog, spec) {
  return tabCandidates(catalog, spec, { golden: true }).map(item => ({ ...item.control, source_id: item.source.id,
    identity: `${item.source.implementation}@${item.source.version}`, language: item.source.language }));
}

async function initialize() {
  window.__conformanceStartup = startupMetrics;
  const params = new URLSearchParams(location.search);
  const startup = await timedStartupPhase('catalog', () => loadStartupCatalog(baseUrl, params));
  let catalog = startup.catalog;
  const bootstrapRevision = catalog.revision;
  const embeddedIdentities = startup.embeddedCaptures ? new Set(startup.embeddedCaptures.map(capture => capture.join('@'))) : null;
  let historyAvailable = !embeddedIdentities && ['http:', 'https:'].includes(baseUrl.protocol);
  root.dataset.historyAvailable = embeddedIdentities ? 'checking' : String(historyAvailable);
  let publicationRequest;
  const readPublication = () => publicationRequest ||= loadPublication(baseUrl);
  let historyDiscovery;
  const readHistory = () => historyDiscovery ||= discoverHistory(baseUrl, catalog, { publication: readPublication() });
  // Resolve explicit historical URLs before controls normalize unknown choices.
  const explicitHistory = params.has('raw') || [...params.keys()].some(key => /^(slots_|base_|cmp_|ref$|cmp$)/.test(key));
  if (embeddedIdentities && explicitHistory) {
    const history = await readHistory();
    if (history) { catalog = history; historyAvailable = true; }
    startup.store.allowExternalResources = historyAvailable;
  }
  const reader = new FixtureResources(catalog, startup.store, undefined, embeddedIdentities ? startup.catalog : null);
  window.addEventListener('pagehide', () => reader.dispose(), { once: true });
  if (params.has('raw')) {
    if (catalog.transport?.schema !== 2 && ![...catalog.sources.unified, ...(catalog.sources.original_unified || [])].some(resource => resource.path === params.get('raw'))) {
      const fullResponse = await fetch(new URL('conformance/viewer-catalog.json', baseUrl), { cache: 'no-store' });
      if (!fullResponse.ok) throw new Error(`Catalog fetch failed (${fullResponse.status})`);
      const full = await fullResponse.json();
      if (full.revision !== catalog.revision && full.revision !== catalog.source_revision) throw new Error('Legacy catalog revision mismatch');
      catalog = full; reader.catalog = full;
    }
    const path = params.get('raw');
    const rawCatalog = startup.catalog.capture_generation?.current?.some(capture => capture.source === path) ? startup.catalog : catalog;
    await rawView(rawCatalog, reader, path); return;
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
  let sourceLink = createSourceLink(baseUrl, publication, bootstrapRevision);
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
    candidates: captureControls(catalog, spec),
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
    const history = await readHistory();
    historyAvailable = Boolean(history);
    startup.store.allowExternalResources = historyAvailable;
    root.dataset.historyAvailable = String(historyAvailable);
    if (historyAvailable) {
      // Catalog replacement invalidates every build admitted against the old reader.
      loader.generation++;
      catalog = history; reader.catalog = catalog;
      descriptors.splice(0, descriptors.length, ...tabCandidates(catalog, active, { golden: true }));
      page.legacy_candidate_identities = catalog.metadata.legacy_candidate_identities || {};
      for (const tab of page.tabs) {
        const spec = catalog.metadata.tabs.find(item => item.id === tab.id);
        tab.candidates = captureControls(catalog, spec);
      }
      root.dataset.catalogRevision = catalog.revision;
      slotOwner.setCandidates(page.tabs.find(tab => tab.id === 'tab-unified').candidates);
      slotOwner.setAvailableIdentities(null, { notify: false });
      await load(panel);
    }
    else {
      const notice = document.createElement('p');
      notice.className = 'capture-slots-notice';
      notice.textContent = 'Latest results are available offline. Older results are not included here.';
      panel.querySelector('.capture-slots').appendChild(notice);
    }
  }
  discoverOlderResults().catch(error => {
    window.__conformanceError = error.stack;
    const notice = document.createElement('p'); notice.className = 'error'; notice.textContent = error.message;
    panel.querySelector('.capture-slots').appendChild(notice);
  });
  // Publication is optional and never gates controls or fixture resolution.
  firstLoad.then(readPublication).then(metadata => {
    if (!metadata) return;
    const upgraded = createSourceLink(baseUrl, metadata, metadata.source_checkout ? catalog.source_revision || catalog.revision : bootstrapRevision);
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

}

if (typeof __CONFORMANCE_WORKER_SOURCE__ !== 'undefined') await startApp();
