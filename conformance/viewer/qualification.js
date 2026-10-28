// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { readFile, mkdir, writeFile, access, mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { resolve, dirname } from 'node:path';
import { createNodeResourceStore } from './data.js';
import { initializeDevelopmentParser } from './yaml_data.js';
initializeDevelopmentParser();
import { FixtureResources } from './resources.js';
import { buildTab, pageMetadata } from './presentation.js';
import { selectedCellOutcome } from './model.js';
import { leafCells, isApplicableCell } from './model.js';
import { canonicalUnifiedLabel } from './resources.js';

const execFileAsync = promisify(execFile);
const normalize = (value) => value.toLowerCase().replace(/[^a-z0-9]+/g, '');

export function selectTabs(model, requested = []) {
  const tabs = model.tabs || [];
  if (!requested.length) return tabs;
  const selected = [], unknown = [];
  for (const name of requested) {
    const wanted = normalize(name.replace(/^tab-/, ''));
    const matches = tabs.filter((tab) => [tab.id?.replace(/^tab-/, '') || '', tab.kind || '', tab.label || ''].map(normalize).includes(wanted));
    if (!matches.length) unknown.push(name);
    for (const tab of matches) if (!selected.includes(tab)) selected.push(tab);
  }
  if (unknown.length) throw new Error(`unknown tab(s): ${unknown.join(', ')}; choices: ${tabs.map((tab) => tab.id || '').join(', ')}`);
  return selected;
}

export function selectRows(tab, requested = []) {
  const rows = (tab.rows || []).filter((row) => row.family);
  if (!requested.length) return rows;
  const selected = [], unknown = [];
  for (const name of requested) {
    const wanted = normalize(name);
    const matches = rows.filter((row) => [row.family || '', row.model_label || ''].map(normalize).includes(wanted));
    if (!matches.length) unknown.push(name);
    for (const row of matches) if (!selected.includes(row)) selected.push(row);
  }
  if (unknown.length) throw new Error(`${tab.id}: unknown model(s): ${unknown.join(', ')}; choices: ${rows.map((row) => row.family || '').join(', ')}`);
  return selected;
}

export function defaultReference(tab) {
  const matches = (tab.candidates || []).filter((candidate) => candidate.default_bucket === 'A');
  if (matches.length !== 1) throw new Error(`${tab.id}: expected exactly one default Reference candidate, found ${matches.length}`);
  return matches[0];
}

function quotedLabel(label) {
  const quote = label.includes("'") && !label.includes('"') ? '"' : "'";
  const escaped = label.replace(/\\/g, '\\\\').replace(/\n/g, '\\n').replace(/\r/g, '\\r').replace(/\t/g, '\\t').split(quote).join(`\\${quote}`);
  return quote + escaped + quote;
}

// Recorded outcome classification consumes the shared model's comparison groups.
// Loading/resource failures never enter this function as missing parser captures.
export function cellState(cell, reference) {
  if (cell == null) return ['empty', 'no cell was emitted for this model/case pair'];
  if (cell.applicable === false || cell.kind === 'cell' && !isApplicableCell(cell)) return ['na', cell.tooltip?.na_note || 'not applicable to this family'];
  if (cell.red_on_diff) {
    const outcome = selectedCellOutcome({ cmp: cell.cmp || {}, base: 'golden', shown: [reference.key], redOnDiff: true });
    const label = quotedLabel(reference.label);
    if (outcome.missingGolden) return ['red', 'Unified cell has no GOLDEN comparison entry'];
    if (outcome.category === 'na') return ['empty', `the default Reference ${label} has no captured result`];
    if (outcome.errorComparisons) return ['red', 'the default Reference returned an error'];
    if (outcome.leakedComparisons) return ['red', 'the default Reference leaked structured markup'];
    if (outcome.category === 'problem') return ['red', 'the default Reference differs from GOLDEN'];
    return ['green', ''];
  }
  const outcome = selectedCellOutcome({ cmp: cell.cmp || {}, base: reference.key, redOnDiff: Boolean(cell.red_on_diff) });
  const label = quotedLabel(reference.label);
  if (outcome.referenceState === 'missing') return ['empty', `the default Reference ${label} has no comparison entry`];
  if (outcome.referenceState === 'unavailable') return ['empty', `the default Reference ${label} has no captured result`];
  if (outcome.referenceState === 'error') return ['red', `the default Reference ${label} returned an error`];
  if (cell.red_on_diff) {
    if (outcome.missingGolden) return ['red', 'Unified cell has no GOLDEN comparison entry'];
    if (outcome.referenceDiffersFromGolden) return ['red', 'the default Reference differs from GOLDEN'];
  } else if (outcome.referenceLeaks) return ['red', 'the default Reference leaked structured markup'];
  return ['green', ''];
}

export function buildStatus(model, tabs = model.tabs, requestedModels = [], reportPath = 'index.html') {
  const reports = [];
  for (const tab of tabs) {
    const reference = defaultReference(tab);
    const columns = tab.columns || [];
    for (const row of selectRows(tab, requestedModels)) {
      const issues = [];
      let na = 0;
      for (const column of columns) {
        const sub = column.sub || '';
        const [state, reason] = cellState(row.cells?.[sub], reference);
        if (state === 'na') na++;
        if (!['green', 'na'].includes(state)) issues.push({ state, case: column.label ?? sub, scenario: sub, reason });
      }
      reports.push({ tab: tab.id, model: row.family, reference: { key: reference.key, label: reference.label }, cells: columns.length,
        empty: issues.filter((issue) => issue.state === 'empty').length, red: issues.filter((issue) => issue.state === 'red').length,
        na, issues });
    }
  }
  return { schema: 1, html: reportPath, generated: model.meta || {}, reports };
}

export function statusTotals(status) {
  return { reports: status.reports.length, empty: status.reports.reduce((sum, report) => sum + report.empty, 0), red: status.reports.reduce((sum, report) => sum + report.red, 0) };
}

export function validateUnifiedKnownDivergences(model, registry, families, aliases, v1SplitScenarios = []) {
  const tabs = selectTabs(model, ['unified']);
  if (tabs.length !== 1) throw new Error('expected exactly one Unified tab');
  const tab = tabs[0];
  const rows = selectRows(tab);
  const actualFamilies = rows.map((row) => row.family);
  if (new Set(actualFamilies).size !== actualFamilies.length) throw new Error('Unified has duplicate family rows');
  const missing = families.filter((family) => !actualFamilies.includes(family)).sort();
  const extra = actualFamilies.filter((family) => !families.includes(family)).sort();
  if (missing.length || extra.length) throw new Error(`Unified display families differ from the generator: missing=${JSON.stringify(missing)} extra=${JSON.stringify(extra)}`);
  if (!registry || typeof registry !== 'object' || Array.isArray(registry) || registry instanceof Map) throw new Error('known divergences: expected a family mapping');
  const unknown = Object.keys(registry).filter((family) => !actualFamilies.includes(family)).sort();
  if (unknown.length) throw new Error(`known divergences: unknown families: ${JSON.stringify(unknown)}`);
  const reference = defaultReference(tab);
  for (const row of rows) {
    const expected = new Set();
    for (const [caseId, checks] of Object.entries(registry[row.family] || {})) {
      const canonical = canonicalUnifiedLabel(caseId, row.family, aliases);
      const scenario = Object.entries(aliases.canonical).find(([, label]) => canonical === `UNIFIED.${label}`)?.[0];
      if (!caseId.startsWith('UNIFIED.') || (!scenario && !caseId.endsWith(`.${row.family}`))) throw new Error(`known divergences: invalid case ID for ${row.family}: ${caseId}`);
      if (Object.hasOwn(checks, 'golden')) expected.add(scenario || caseId.slice('UNIFIED.'.length, -(row.family.length + 1)));
    }
    const states = [...leafCells(row)].map(([sub, cell]) => [sub, cellState(cell, reference)[0], cell]);
    if (!states.length || states.every(([, state]) => state === 'na')) throw new Error(`Unified ${row.family}: no applicable fixture leaves`);
    const empty = states.filter(([sub, state, cell]) => {
      if (state !== 'empty') return false;
      const reason = cell.tooltip?.candidates?.find((candidate) => candidate.key === reference.key)?.block?.unavailable;
      const expectedUnavailable = `Dynamo v2 does not run the Dynamo v1 split parser for ${row.family};`;
      return !(v1SplitScenarios.includes(sub) && typeof reason === 'string' && reason.startsWith(expectedUnavailable));
    }).map(([sub]) => sub).sort();
    const actual = new Set(states.filter(([, state]) => state === 'red').map(([sub]) => sub));
    if (empty.length || actual.size !== expected.size || [...actual].some((sub) => !expected.has(sub))) throw new Error(`Unified display differs from known divergences for ${row.family}: empty=${JSON.stringify(empty)} expected_red=${JSON.stringify([...expected].sort())} actual_red=${JSON.stringify([...actual].sort())}`);
  }
}

export function printStatus(status, summaryOnly = false, output = console.log) {
  if (summaryOnly) {
    const totals = statusTotals(status);
    output(`conformance status: ${totals.reports} model/tab pairs, ${totals.empty} empty, ${totals.red} red`);
    return;
  }
  for (const report of status.reports) {
    output(`${report.tab} ${report.model}: ${report.cells} cells, ${report.na || 0} n/a, ${report.empty} empty, ${report.red} red (Reference: ${report.reference.label})`);
    for (const issue of report.issues) output(`  ${issue.state.toUpperCase()} ${issue.case} (${issue.scenario}): ${issue.reason}`);
  }
}

export function parseStatusArguments(argv) {
  const options = { root: fileURLToPath(new URL('../../', import.meta.url)), models: [], tabs: [], requireGreen: false, summaryOnly: false };
  const valued = new Map([['--root', 'root'], ['--status-path', 'statusPath'], ['--report-path', 'reportPath'], ['--html', 'reportPath'], ['--commit-sha', 'commitSha'], ['--known-divergences', 'knownDivergences']]);
  for (let index = 0; index < argv.length; index++) {
    const argument = argv[index];
    if (['--help', '-h'].includes(argument)) options.help = true;
    else if (argument === '--require-green') options.requireGreen = true;
    else if (argument === '--summary-only') options.summaryOnly = true;
    else if (argument === '--model' || argument === '--tab' || valued.has(argument)) {
      const value = argv[++index];
      if (value === undefined || value.startsWith('--')) throw new Error(`${argument} requires a value`);
      if (argument === '--model') options.models.push(value);
      else if (argument === '--tab') options.tabs.push(value);
      else options[valued.get(argument)] = value;
    } else throw new Error(`invalid argument: ${argument}`);
  }
  return options;
}

export async function runStatus(options, output = console.log, errorOutput = console.error) {
  let reader;
  let currentView;
  try {
    const root = resolve(options.root);
    let split = true;
    try {
      const resolver = resolve(root, 'conformance/utils/src/capture_sources.py');
      await access(resolver);
      const result = await execFileAsync(process.env.CONFORMANCE_PYTHON || 'python3', [resolver, '--root', root, '--has-latest']);
      split = JSON.parse(result.stdout);
    } catch (error) {
      if (error.code !== 'ENOENT') throw error;
      split = false;
    }
    if (split) {
      currentView = await mkdtemp(resolve(process.env.TMPDIR || tmpdir(), 'current-source-status-'));
      await execFileAsync(process.env.CONFORMANCE_PYTHON || 'python3', [resolve(root, 'conformance/utils/src/capture_sources.py'),
        '--root', root, '--stage-current-source', currentView]);
    }
    const sourceRoot = currentView || root;
    const catalog = JSON.parse(await readFile(resolve(sourceRoot, 'conformance/viewer-catalog.json'), 'utf8'));
    const selected = selectTabs({ tabs: catalog.metadata.tabs.map((spec) => ({ ...spec, kind: spec.suite })) }, options.tabs);
    const commitSha = options.commitSha ?? (await execFileAsync('git', ['-C', root, 'rev-parse', 'HEAD'])).stdout.trim();
    const page = pageMetadata(catalog, { commitSha });
    reader = new FixtureResources(catalog, await createNodeResourceStore(sourceRoot, { allowSourceFiles: true }));
    if (options.knownDivergences && !selected.some((tab) => tab.id === 'tab-unified')) throw new Error('--known-divergences requires selecting the Unified tab');
    const known = options.knownDivergences ? await reader.parser.parseYaml(await readFile(options.knownDivergences), { path: options.knownDivergences }) : null;
    const status = { schema: 1, html: options.reportPath || resolve(root, 'index.html'), generated: page.meta, reports: [] };
    for (const spec of selected) {
      const tab = await buildTab(catalog, reader, spec.id);
      if (options.knownDivergences && tab.id === 'tab-unified') validateUnifiedKnownDivergences({ tabs: [tab] }, known, Object.keys(catalog.metadata.unified_presentation.parser_generations), catalog.metadata.unified_aliases, catalog.metadata.unified_presentation.v1_split_scenarios);
      status.reports.push(...buildStatus(page, [tab], options.models, status.html).reports);
    }
    const totals = statusTotals(status);
    if (options.requireGreen && (totals.empty || totals.red)) {
      errorOutput('ERROR: selected conformance cells are empty or red');
      printStatus(status, false, output);
      return 1;
    }
    if (totals.empty || totals.red) errorOutput(`WARNING: selected Reference cells include ${totals.empty} empty and ${totals.red} red; use --require-green to reject them`);
    if (options.statusPath) {
      await mkdir(dirname(options.statusPath), { recursive: true });
      await writeFile(options.statusPath, JSON.stringify(status, null, 2) + '\n');
    }
    printStatus(status, options.summaryOnly, output);
    return 0;
  } catch (error) {
    errorOutput(`ERROR: ${error.message}`);
    return 2;
  } finally {
    reader?.dispose();
    if (currentView) await rm(currentView, { recursive: true, force: true });
  }
}


export async function runSnapshots(argv) {
const args = ['node', 'qualification.js', ...argv];
const { default: assert } = await import('node:assert/strict');
const { readFileSync } = await import('node:fs');
const { parseSnapshot } = await import('./data.js');
const { createNodeResourceStore } = await import('./data.js');
const { initializeDevelopmentParser } = await import('./yaml_data.js');
const { FixtureResources } = await import('./resources.js');
const { sourceCapturesFamily } = await import('./model.js');
const { compareVersions, resolveUnifiedSeries } = await import('./model.js');
const { sameValue } = await import('./values.js');
const { mappingEntries, deleteField, getField, hasField, setField } = await import('./values.js');
const { boundOracleRecord, sourceReference, validateResolvedUnified, validateUnifiedCapture, validateUnifiedInputs } = await import('./resources.js');


initializeDevelopmentParser();


const [root, site, catalogPath, selectionPath] = args.slice(2);
const selectedPaths = selectionPath ? new Set(JSON.parse(readFileSync(selectionPath, 'utf8'))) : null;
const catalog = JSON.parse(readFileSync(catalogPath, 'utf8'));
const originals = catalog.proof_original_unified;
assert(Array.isArray(originals), 'proof requires source catalog descriptors');
const originalCatalog = { ...catalog, sources: { ...catalog.sources, unified: originals } };
const sourceStore = await createNodeResourceStore(root, { allowSourceFiles: true });
const authoredReader = new FixtureResources(originalCatalog, sourceStore);
const snapshotStore = await createNodeResourceStore(site);
let authoredComparisons = 0;
let snapshotCount = 0;
let recordCount = 0;
let checkpointVisits = 0;
let decodedCaptureBundles = 0;
let retainedCaptureBundles = 0;
let peakRetainedCaptureBundles = 0;
let peakHeapBytes = 0;
async function readSnapshot(resource) {
  return parseSnapshot(await snapshotStore.read(resource), resource);
}
async function readOracleCapture(reader, file, input) {
  const parsed = await reader.yaml(file, null, true);
  let execution = null;
  if (parsed.value.execution) {
    const sourcePath = (resource, reference, directory) => `${resource.path.replace(/\/(?:families|executions|receipts)\/.*$/, '')}/${sourceReference(reference, directory)}`;
    const manifest = originals.find(item => item.kind === 'execution' && item.path === sourcePath(file, parsed.value.execution, 'executions'));
    if (!manifest || file.execution !== parsed.value.execution) throw new Error(`execution manifest not indexed for ${file.path}`);
    execution = await reader.yaml(manifest, null, true);
    const receiptIds = Object.keys(execution.value.receipts || {}).sort();
    assert.deepEqual(receiptIds, [...(manifest.receipts || [])].sort(), 'execution receipt dependencies not indexed');
    const receipts = new Map(await Promise.all(receiptIds.map(async digest => {
      const resource = originals.find(item => item.kind === 'receipt' && item.path === sourcePath(manifest, digest, 'receipts'));
      if (!resource) throw new Error(`execution receipt not indexed: ${digest}`);
      return [digest, await reader.yaml(resource, null, true)];
    })));
    execution = { ...execution, receipts };
  }
  const document = await validateUnifiedCapture(parsed.value, input, file, execution);
  return { ...file, document, bytes: parsed.sourceBytes };
}
const authoredFamilies = new Set(originals.filter(item => item.kind === 'inputs-and-golden').map(item => item.family));
const inputResources = catalog.sources.unified.filter(item => item.kind === 'inputs-and-golden');
const publishedInputFamilies = new Set(inputResources.map(item => item.family));
const missingInputFamilies = [...authoredFamilies].filter(family => !publishedInputFamilies.has(family));
const extraInputFamilies = [...publishedInputFamilies].filter(family => !authoredFamilies.has(family));
assert.equal(missingInputFamilies.length + extraInputFamilies.length, 0,
  `published authored family membership differs: missing ${JSON.stringify(missingInputFamilies)}; extra ${JSON.stringify(extraInputFamilies)}`);
for (const inputResource of inputResources) {
  const family = inputResource.family;
  const snapshot = await readSnapshot(inputResource);
  const original = originals.find(item => item.family === family && item.kind === 'inputs-and-golden');
  const { value: yaml } = await authoredReader.yaml(original, null, true);
  assert(sameValue(snapshot, yaml), `authored JSON differs from YAML: ${family}`);
  validateUnifiedInputs(snapshot, family, inputResource.path, catalog.metadata.unified_aliases);
  authoredComparisons += 1;
}

const histories = new Map();
const earliestVersions = new Map();
for (const resource of originals.filter(item => item.kind === 'history')) {
  const key = `${resource.family}\0${resource.implementation}`;
  if (!histories.has(key)) histories.set(key, []);
  histories.get(key).push(resource);
  if (!earliestVersions.has(key) || compareVersions(resource.version, earliestVersions.get(key)) < 0) earliestVersions.set(key, resource.version);
}
const expectedSelections = new Set();
const selectionIdentity = (implementation, version, family) => JSON.stringify([implementation, version, family]);
for (const candidate of catalog.sources.candidates.filter(item => item.mode === 'unified')) {
  for (const family of authoredFamilies) {
    const earliest = earliestVersions.get(`${family}\0${candidate.implementation}`);
    if (sourceCapturesFamily(candidate, family, { nativeOnly: false }) && earliest && compareVersions(earliest, candidate.version) <= 0) {
      expectedSelections.add(selectionIdentity(candidate.implementation, candidate.version, family));
    }
  }
}
const implementations = new Map();
const publishedSelections = new Set();
for (const resource of catalog.sources.unified.filter(item => item.kind === 'resolved-capture')) {
  for (const family of resource.families || [resource.family]) {
    const identity = selectionIdentity(resource.implementation, resource.version, family);
    assert(!publishedSelections.has(identity), `duplicate published capture selection: ${identity}`);
    publishedSelections.add(identity);
  }
  if (!implementations.has(resource.implementation)) implementations.set(resource.implementation, []);
  implementations.get(resource.implementation).push(resource);
}
const missingSelections = [...expectedSelections].filter(identity => !publishedSelections.has(identity));
const extraSelections = [...publishedSelections].filter(identity => !expectedSelections.has(identity));
assert.equal(missingSelections.length + extraSelections.length, 0,
  `published capture selection membership differs: ${missingSelections.length} missing ${JSON.stringify(missingSelections.slice(0, 10))}; ${extraSelections.length} extra ${JSON.stringify(extraSelections.slice(0, 10))}`);
// Validate complete publication membership above, then partition comparison work.
if (selectedPaths) {
  const known = new Set(catalog.sources.unified.filter(item => item.kind === 'resolved-capture').map(item => item.path));
  for (const path of selectedPaths) assert(known.has(path), `unknown selected capture: ${path}`);
}
console.log(`parity selection: ${JSON.stringify(selectedPaths ? [...selectedPaths].sort() : catalog.sources.unified.filter(item => item.kind === 'resolved-capture').map(item => item.path).sort())}`);
for (const [implementation, assigned] of implementations) {
  // Release expanded YAML and receipt caches before the next implementation.
  const reader = new FixtureResources(originalCatalog, sourceStore);
  const families = new Map();
  let resources = assigned;
  const allResources = [...resources];
  if (selectedPaths) resources = resources.filter(item => selectedPaths.has(item.path));
  if (!resources.length) continue;
  resources.sort((left, right) => compareVersions(left.version, right.version));
  for (const resource of resources) {
    // A bundle is decoded once, compared for every family, then released.
    const bundle = await readSnapshot(resource);
    decodedCaptureBundles += 1;
    retainedCaptureBundles += 1;
    peakRetainedCaptureBundles = Math.max(peakRetainedCaptureBundles, retainedCaptureBundles);
    for (const family of resource.families || [resource.family]) {
      if (!families.has(family)) {
        const { value: input, sourceResource: originalInput } = await reader.unifiedInputs(family);
        const descriptors = [...(histories.get(`${family}\0${implementation}`) || [])]
          .sort((left, right) => compareVersions(left.version, right.version));
        const first = await readOracleCapture(reader, descriptors[0], input);
        const versions = allResources.filter(item => (item.families || [item.family]).includes(family)).map(item => item.version);
        if (first.document.schema_version === 2 || first.document.schema_version === 2n) {
          families.set(family, { input, originalInput, descriptors, complete: true, originIndex: 0 });
        } else {
          const yamlHistory = await Promise.all(descriptors.map(item => readOracleCapture(reader, item, input)));
          families.set(family, { input, history: yamlHistory, byPath: new Map(yamlHistory.map(item => [item.path, item])),
            series: resolveUnifiedSeries(yamlHistory, versions), originByCase: new Map(), originIndex: 0 });
        }
      }
      const context = families.get(family);
      let resolved;
      if (context.complete) {
        while (context.originIndex < context.descriptors.length
            && compareVersions(context.descriptors[context.originIndex].version, resource.version) <= 0) {
          context.originIndex += 1;
          checkpointVisits += 1;
        }
        assert(context.originIndex, `no YAML snapshot for ${resource.capture_identity}`);
        const descriptor = context.descriptors[context.originIndex - 1];
        const capture = await readOracleCapture(reader, descriptor, context.input);
        // This oracle reads complete source records directly, independently of
        // the publisher and checkout reader's selected-snapshot projection.
        resolved = new Map(Object.entries(structuredClone(capture.document.records)));
        if (descriptor.version !== resource.version) for (const record of resolved.values()) {
          const document = getField(record.value, 'document');
          const metadata = getField(document, 'record_metadata');
          deleteField(metadata, 'verified_recapture');
          if (metadata !== undefined && mappingEntries(metadata).length === 0) deleteField(document, 'record_metadata');
          if (!hasField(document, 'inherited_from') && record.suppress_inherited_from !== true) setField(document, 'inherited_from', descriptor.version);
        }
        reader.documents.clear();
        reader.captures.clear();
      } else {
        let next = context.series.next();
        while (!next.done && compareVersions(next.value[0], resource.version) < 0) next = context.series.next();
        assert.equal(next.value?.[0], resource.version, 'incremental version differs from published resource');
        resolved = next.value[1];
        while (context.originIndex < context.history.length && compareVersions(context.history[context.originIndex].version, resource.version) <= 0) {
          const capture = context.history[context.originIndex++];
          checkpointVisits += 1;
          for (const [caseId, change] of Object.entries(capture.document.changes || {})) {
            if (change?.absent === true && Object.keys(change).length === 1) context.originByCase.delete(caseId);
            else if (!context.originByCase.has(caseId) || Object.hasOwn(change, 'observation')) context.originByCase.set(caseId, capture.path);
          }
        }
      }
      assert(context.originIndex, `no YAML history for ${resource.capture_identity}`);
      const snapshot = resource.families ? bundle[family] : bundle;
      await validateResolvedUnified(snapshot, context.input, { ...resource, family }, catalog.sources.original_unified || originals);
      assert.equal(resolved.size, Object.keys(snapshot).length, `record count differs: ${resource.capture_identity}:${family}`);
      for (const [caseId, value] of resolved) {
        const expected = boundOracleRecord(value.value, getField(context.input.cases, caseId));
        assert(sameValue(expected, snapshot[caseId].value), `resolved record differs: ${resource.capture_identity}:${family}:${caseId}`);
        assert.equal(value.member, snapshot[caseId].member, `resolved source differs: ${resource.capture_identity}:${caseId}`);
        if (!context.complete) {
          assert.equal(context.originByCase.get(caseId), snapshot[caseId].member, `source identity differs: ${resource.capture_identity}:${caseId}`);
          assert.deepEqual(value.sourceBytes, context.byPath.get(snapshot[caseId].member)?.bytes, `source bytes differ: ${resource.capture_identity}:${caseId}`);
        }
        recordCount += 1;
      }
      snapshotCount += 1;
    }
    peakHeapBytes = Math.max(peakHeapBytes, process.memoryUsage().heapUsed);
    retainedCaptureBundles -= 1;
  }
}

assert(authoredComparisons > 0);
assert(selectedPaths?.size === 0 || (snapshotCount > 0 && recordCount > 0));
console.log(`full corpus parity passed: ${authoredComparisons} authored family comparisons, ${snapshotCount} resolved captures, ${recordCount} complete records`);
console.log(`parity work: ${checkpointVisits} checkpoint visits, ${decodedCaptureBundles} capture bundles decoded, ${peakRetainedCaptureBundles} retained capture bundle, ${peakHeapBytes} peak sampled heap bytes`);
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0




}

export async function runModels(argv) {
const args = ['node', 'qualification.js', ...argv];
const { default: assert } = await import('node:assert/strict');
const { readFile } = await import('node:fs/promises');
const { createNodeResourceStore } = await import('./data.js');
const { initializeDevelopmentParser } = await import('./yaml_data.js');
const { FixtureResources } = await import('./resources.js');
const { buildPage, buildTab, pageMetadata } = await import('./presentation.js');

const { tabCandidates, selectedUnifiedFamilies } = await import('./model.js');
const { classifyUnified, compareLegacyVersions, resolveUnified } = await import('./model.js');
const { leafCells } = await import('./model.js');
const { columnKeysFor, tabPresentation } = await import('./presentation.js');
const { sameValue } = await import('./values.js');
const { getField } = await import('./values.js');
const { selectedCaptureOutcome } = await import('./model.js');
const { canonicalUnifiedEvents } = await import('./model.js');


initializeDevelopmentParser();


const modelContracts = JSON.parse(await readFile(new URL('../utils/tests/fixtures/model_contracts.json', import.meta.url)));
const root = args[3] || new URL('../../', import.meta.url);
const catalog = JSON.parse(await readFile(args[2] || new URL('conformance/viewer-catalog.json', root)));
// The retained batch presentation flag is separate from the browser's active tab.
// Validate metadata contracts before constructing the expensive matrix.
const activeIds = [];
for (const spec of catalog.metadata.tabs) {
  const descriptors = tabCandidates(catalog, spec);
  assert.equal(descriptors.filter(descriptor => descriptor.control.default_bucket === 'A').length,
    1, `${spec.id}: retained tabs need exactly one Reference`);
  assert.equal(Boolean(tabPresentation(spec, catalog, descriptors).active), false);
}
async function newReader() {
  return new FixtureResources(catalog, await createNodeResourceStore(root, { allowSourceFiles: !args[3] }));
}
async function checkPageOrchestration() {
  const reader = await newReader();
  try {
    const page = await buildPage(catalog, reader, { selected: [], reference: null });
    assert.equal(page.schema, 1);
    assert.equal(Object.hasOwn(page, 'strings'), false);
    assert.deepEqual(page.tabs.map(tab => tab.id), catalog.metadata.tabs.map(tab => tab.id));
    assert.deepEqual(page.tabs.filter(tab => tab.active).map(tab => tab.id), activeIds);
  } finally {
    reader.dispose();
  }
}
await checkPageOrchestration();
const columnKeys = new Set(columnKeysFor(catalog.metadata));
let cells = 0, leaves = 0;
function validateTab(tab, loadedKeys = null) {
  const spec = catalog.metadata.tabs.find(spec => spec.id === tab.id);
  const descriptors = tabCandidates(catalog, spec, { golden: tab.candidates.some(candidate => candidate.key === 'golden') });
  assert.deepEqual(tab.candidates, descriptors.map(descriptor => descriptor.control));
  assert.equal(tab.candidates.filter(candidate => candidate.default_bucket === 'A').length, 1);
  assert.equal(new Set(tab.candidates.map(candidate => candidate.key)).size, tab.candidates.length);
  assert.equal(new Set(tab.columns.map(column => column.sub)).size, tab.columns.length);
  assert.equal(tab.column_groups.reduce((sum, group) => sum + group.span, 0), tab.columns.length);
  for (const group of tab.column_groups) assert.ok(columnKeys.has(group.key), `${tab.id}: column group ${group.key} must exist before the tab loads`);
  assert.equal(Object.hasOwn(tab, 'cand_meta'), false);
  const selectable = new Set(tab.candidates.map(candidate => candidate.key));
  selectable.add('golden');
  const sources = new Map();
  for (const descriptor of descriptors) {
    if (descriptor.source.implementation === 'golden') continue;
    assert.ok(descriptor.source.version);
    const owner = `${descriptor.source.implementation}@${descriptor.source.mode}`;
    const prior = sources.get(owner);
    if (prior) assert.ok(compareLegacyVersions(prior, descriptor.source.version) >= 0);
    sources.set(owner, descriptor.source.version);
    assert.ok(!descriptor.control.label.includes('+source.'));
    if (spec.mode === 'stream') assert.ok(!descriptor.source.version.match(/\.patch\d+$/));
  }
  for (const column of tab.columns) {
    if (!/^unified_g\d+(?:_|$)/.test(column.group_key)) continue;
    const families = new Set(tab.rows.filter(row => row.family && row.cells[column.sub]
      && row.cells[column.sub].status !== 'na' && ['cell', 'missing'].includes(row.cells[column.sub].kind)).map(row => row.family));
    assert.ok(families.size > 1, `${column.sub}: single-family generic column: ${[...families]}`);
  }
  for (const row of tab.rows) {
    if (row.section) continue;
    assert.ok(row.family && row.model_label);
    assert.equal(Object.keys(row.cells).length, tab.columns.length);
    for (const column of tab.columns) {
      const cell = row.cells[column.sub]; cells++;
      assert.ok(cell);
      if (loadedKeys) assert.deepEqual(Object.keys(cell.cmp).sort(), ['golden', ...loadedKeys].sort(), `${tab.id}:${row.family}:${column.sub}: every loaded comparison`);
      assert.ok(['ok', 'problem', 'na', 'missing', 'todo'].includes(cell.status) || cell.kind === 'blank');
      for (const [key, value] of Object.entries(cell.cmp || {})) {
        assert.ok(selectable.has(key), `${tab.id}:${row.family}:${column.sub}: unselectable comparison ${key}`);
        for (const field of ['na', 'err', 'leak']) assert.ok([0, 1].includes(value[field]));
        assert.ok(Object.hasOwn(value, 'sig'));
      }
      const list = [...leafCells({ cells: { selected: cell } }).values()]; leaves += list.length;
      const ids = list.filter(cell => cell.case_id).map(cell => cell.case_id);
      assert.equal(new Set(ids).size, ids.length, `${tab.id}:${row.family}:${column.sub}: repeated folded leaf`);
      for (const leaf of list) {
        if (loadedKeys) {
          assert.deepEqual(Object.keys(leaf.cmp).sort(), ['golden', ...loadedKeys].sort(), `${leaf.sub}: every loaded leaf comparison`);
          assert.deepEqual(leaf.tooltip.candidates.map(candidate => candidate.key).sort(), ['golden', ...loadedKeys].sort(), `${leaf.sub}: every loaded tooltip`);
        }
        for (const fact of leaf.facts || []) {
          assert.ok(fact.impl && typeof fact.status === 'string');
          assert.ok(typeof fact.leak === 'boolean');
        }
        for (const candidate of leaf.tooltip?.candidates || []) {
          assert.ok(selectable.has(candidate.key));
          assert.ok(candidate.label);
          const block = candidate.block;
          if (!block) continue;
          if ((candidate.impl || candidate.key).startsWith('dynamo')) for (const call of block.calls || []) assert.ok(!/^(.+)\1$/.test(call.name || ''));
        }
      }
    }
  }
}
function validateUnifiedColumns(unified) {
  assert.ok(unified.columns.every(column => column.label.includes('-')));
  assert.ok(unified.columns.some(column => column.variant_labels?.length || column.schemas?.length));
  if (catalog.capture_generation?.mode === 'latest') {
    const local = new Set(catalog.capture_generation.current.filter(row => row.surface === 'unified')
      .map(row => `${row.implementation}@${row.version}`));
    assert.ok(catalog.sources.candidates.filter(row => row.mode === 'unified')
      .every(row => local.has(`${row.implementation}@${row.version}`)), 'latest choices must name local current observations');
  } else {
    assert.ok(unified.candidates.some(candidate => candidate.key === 'dynamo@0.6.0'));
  }
}
const sparseProvenance = resolveUnified([
  { implementation: 'dynamo', version: '0.9.0', path: 'first.yaml', bytes: new Uint8Array(), document: {
    changes: { 'UNIFIED.provenance': { case_key: 'provenance', stimulus: {}, observation: { events: [] } } },
    document: {}, document_overrides: { 'UNIFIED.provenance': { parser_path: 'unified', capture_provenance: { captured_with: { dynamo: '0.8.0' }, inherited_from: null } } },
  } },
  { implementation: 'dynamo', version: '1.0.0', path: 'next.yaml', bytes: new Uint8Array(), document: {
    changes: { 'UNIFIED.provenance': { stimulus: { updated: true } } }, document: {},
  } },
], '1.0.0');
assert.equal(sparseProvenance.get('UNIFIED.provenance').value.document.captured_with.dynamo, '0.8.0');
assert.equal(Object.hasOwn(sparseProvenance.get('UNIFIED.provenance').value.document, 'inherited_from'), false);


async function checkUnifiedSelections(original, reader, loadedKeys) {
  const spec = catalog.metadata.tabs.find(item => item.id === original.id);
  const currentNativeFamilies = tabCandidates(catalog, spec).find(item => item.control.key === 'dynamo').source.native_unified_families;
  const descriptors = tabCandidates(catalog, spec).filter(item => item.source.version && loadedKeys.has(item.control.key) && singletonKeys.has(item.control.key));
  const implementations = [...new Set(descriptors.map(item => item.source.implementation))];
  for (const implementation of implementations) {
    const versions = descriptors.filter(item => item.source.implementation === implementation);
    for (const descriptor of [versions[0], versions.at(-1)]) {
      const key = descriptor.control.key;
      const selected = await buildTab(catalog, reader, original.id, { selected: [key], reference: null });
      const expectedFamilies = descriptor.source.implementation === 'dynamo_v2'
        ? currentNativeFamilies.filter(family => !descriptor.source.capture_families || descriptor.source.capture_families.includes(family))
        : descriptor.source.capture_families;
      assert.deepEqual(selected.rows.map(row => row.family), expectedFamilies,
        `${key}: historical observations retain the current native family scope`);
      for (const row of selected.rows) {
        const baseline = leafCells(original.rows.find(item => item.family === row.family));
        for (const [scenario, cell] of leafCells(row)) {
          const originalCell = baseline.get(scenario);
          const relation = entry => { const cmp = entry.cmp[key]; return [cmp.na, cmp.err, cmp.leak, cmp.sig === entry.cmp.golden.sig]; };
          assert.deepEqual(relation(cell), relation(originalCell), `${key}:${row.family}:${scenario}: GOLDEN comparison preservation`);
          // Not-applicable prose may describe a different first visible family; it has no captured output.
          if (cell.status !== 'na') assert.deepEqual(cell.tooltip.candidates.find(item => item.key === key).block, originalCell.tooltip.candidates.find(item => item.key === key).block, `${key}:${row.family}:${scenario}: full captured output preservation`);
        }
      }
    }
  }
  for (const descriptor of descriptors) singletonKeys.delete(descriptor.control.key);
  const empty = await buildTab(catalog, reader, original.id, { selected: [], reference: null });
  assert.deepEqual(empty.rows, [], 'None contributes no families');
}
const unifiedSpec = catalog.metadata.tabs.find(spec => spec.suite === 'unified');
assert.ok(unifiedSpec);
const allUnifiedDescriptors = tabCandidates(catalog, unifiedSpec).filter(item => item.source.version);
const selectedResources = args[4] ? JSON.parse(await readFile(args[4])) : null;
const unifiedDescriptors = allUnifiedDescriptors.filter(item => !selectedResources || selectedResources.some(resource => resource.implementation === item.source.implementation && resource.version === item.source.version));
const defaultKey = allUnifiedDescriptors.find(item => item.control.default_bucket === 'A').control.key;
const defaultDescriptor = allUnifiedDescriptors.find(item => item.control.key === defaultKey);
if (!unifiedDescriptors.includes(defaultDescriptor)) unifiedDescriptors.push(defaultDescriptor);
for (const implementation of new Set(allUnifiedDescriptors.map(item => item.source.implementation))) {
  const earliest = allUnifiedDescriptors.filter(item => item.source.implementation === implementation).at(-1);
  if (earliest && !unifiedDescriptors.includes(earliest)) unifiedDescriptors.push(earliest);
}
console.log(`model selection: ${JSON.stringify(unifiedDescriptors.map(item => item.source.id).sort())}`);
const singletonKeys = new Set();
for (const implementation of new Set(unifiedDescriptors.map(item => item.source.implementation))) {
  const versions = unifiedDescriptors.filter(item => item.source.implementation === implementation);
  for (const descriptor of [versions[0], versions.at(-1)]) singletonKeys.add(descriptor.control.key);
}
const testedKeys = new Set(), testedFamilies = new Set(), testedScenarios = new Set();
let checkedRebaseContracts = false;
async function checkRebaseContracts(tab, reader) {
  if (checkedRebaseContracts) return;
  const columns = new Map(tab.columns.map(column => [column.sub, column]));
  const standalone = tab.columns.filter(column => catalog.metadata.unified_taxonomy[column.sub].display_group === 'gpt_oss');
  assert.deepEqual(standalone.map(column => column.sub).sort(), modelContracts.gptOssStandaloneHeaderScenarios, 'only the exact standalone Harmony header scenarios retain their own section');
  for (const [scenario, config] of Object.entries(catalog.metadata.folds.unified)) {
    if (config.case && config.case[0] !== scenario) assert.equal(columns.has(scenario), false, `shared variant must fold: ${scenario}`);
  }
  for (const row of tab.rows.filter(row => !row.section)) {
    const leaves = leafCells(row);
    const actual = [...leaves].filter(([, cell]) => cell.status !== 'na').map(([scenario]) => scenario).sort();
    const expected = Object.entries(catalog.metadata.unified_presentation.scenario_families)
      .filter(([, families]) => families.includes(row.family)).map(([scenario]) => scenario).sort();
    assert.deepEqual(actual, expected, `${row.family}: every applicable shared variant remains a leaf`);
  }
  for (const row of tab.rows.filter(row => !row.section)) {
    for (const parent of ['arg_json_null', 'arg_string_null']) {
      const expected = Object.keys(catalog.metadata.folds.unified).filter(scenario => scenario.startsWith(parent));
      expected.push('arg_null_mixed_labels');
      assert.deepEqual(row.cells[parent].variants.map(leaf => leaf.sub).sort(), expected.sort(), `${row.family}: complete null fold`);
    }
  }
  for (const scenario of ['default_reasoning_without_opener', 'default_reasoning_tool_without_closer']) {
    assert.equal(columns.get(scenario).group_key, 'unified_gnemotron_v3');
  }
  const groups = new Map(tab.column_groups.map(group => [group.key, group]));
  const argumentGroups = tab.column_groups.filter(group => group.key.startsWith('unified_g7_'));
  assert.deepEqual(argumentGroups.map(group => group.span), [5, 4, 12, 6, 10, 17, 14]);
  const argumentColumns = tab.columns.filter(column => column.group_key.startsWith('unified_g7_'));
  const numericColumns=tab.columns.filter(column=>['7-17','7-19','7-18'].includes(column.label));
  assert.deepEqual(numericColumns.map(column=>column.label),['7-17','7-19','7-18']);
  assert.equal(new Set(numericColumns.map(column=>column.group_key)).size,1);
  assert.equal(new Set(numericColumns.map(column=>column.band)).size,1);
  assert.equal(tab.column_groups.find(group=>group.key===numericColumns[0].group_key).label,'TC Argument fidelity');
  for (const row of tab.rows.filter(row=>!row.section)) for (const column of numericColumns) {
    const cell=row.cells[column.sub];
    assert.equal(cell.variants.length,{'7-17':9,'7-19':3,'7-18':7}[column.label]);
    assert.equal(cell.status==='na',column.label==='7-19'&&!catalog.metadata.folds.numeric_applicability['7-14.string'].includes(row.family));
  }
  assert.equal(argumentColumns.length, 68);
  assert.deepEqual(argumentColumns.map(column => column.display_number), Array.from({ length: 68 }, (_, index) => index + 1));
  for (const row of tab.rows.filter(row => !row.section)) {
    for (const [scenario, cell] of leafCells(row)) {
      if (!['arg_object_quoted_parameter_close', 'arg_array_quoted_parameter_close'].includes(scenario)
          && !scenario.startsWith('spark_phase_') && !scenario.startsWith('muse_')) continue;
      if (cell.status === 'na') continue;
      const golden = cell.tooltip.candidates.find(candidate => candidate.key === 'golden');
      const current = cell.tooltip.candidates.find(candidate => candidate.key === defaultKey);
      assert.deepEqual(canonicalUnifiedEvents(current.block.events), canonicalUnifiedEvents(golden.block.events), `${row.family}/${scenario}: exact current event values`);
      assert.equal(selectedCaptureOutcome(cell, defaultKey), 'pass', `${row.family}/${scenario}`);
    }
  }
  assert.equal(groups.get('unified_gqwen3').label, 'Family-specific tests: Qwen3 and Nemotron v3 XML');
  for (const quotes of ['single', 'double']) {
    const custom = columns.get(`xml_custom_type_${quotes}_quoted_object`);
    const string = columns.get(`xml_string_type_${quotes}_quoted_object`);
    assert.equal(custom.group_key, 'unified_gqwen3');
    assert.ok(custom.label.startsWith('qwen3-'));
    assert.equal(string.group_key, 'unified_g7_declared');
    for (const row of tab.rows.filter(row => !row.section)) {
      const cell = leafCells(row).get(custom.sub);
      assert.equal(cell.case_id, `UNIFIED.7-${custom.sub}`);
      assert.equal(cell.status === 'na', !['qwen3', 'nemotron_v3'].includes(row.family));
      if (['qwen3', 'nemotron_v3'].includes(row.family)) assert.equal(selectedCaptureOutcome(cell, defaultKey), 'pass');
    }
  }
  assert.equal(columns.get('arg_marker_in_string').group_key, 'unified_g7_text');
  assert.equal(columns.get('reason_only').schemas?.length || 0, 0);
  const toolSchemas = columns.get('tool_only').schemas;
  assert.equal(toolSchemas.length, 1);
  const schema = toolSchemas[0];
  assert.deepEqual(schema.tools, [{ name: 'lookup', parameters: { type: 'object', properties: { q: { type: 'string' } } } }]);
  assert.deepEqual([...schema.families].sort(), ['kimi_k2', 'kimi_k3']);
  assert.equal(schema.cases.length, 1);
  assert.equal(schema.cases[0].case_id, 'UNIFIED.gpt_oss-tool_body_whitespace');
  assert.equal(schema.cases[0].label, 'Leading JSON body whitespace');
  assert.deepEqual([...schema.cases[0].families].sort(), ['kimi_k2', 'kimi_k3']);
  assert.ok(columns.get('schema_declared_integer').schemas.length);

  for (const family of ['gpt_oss', 'qwen3', 'glm47', 'deepseek_v4']) {
    const row = tab.rows.find(row => row.family === family);
    const cell = leafCells(row).get('structured_response_after_reasoning');
    const inputs = (await reader.unifiedInputs(family)).value;
    const authored = Object.values(inputs.cases).find(item => item.scenario === 'structured_response_after_reasoning');
    assert(sameValue(cell.tooltip.init, getField(authored.request, 'init')), `${family}: exact structural request init reaches popup model`);
    assert.equal(selectedCaptureOutcome(cell, defaultKey), 'pass', `${family}: current structured response matches GOLDEN`);
    assert.equal(Object.hasOwn(leafCells(row).get('tool_only').tooltip.init, 'structural_tag'), false);
  }
  checkedRebaseContracts = true;
}
function checkDottedNameContracts(tab, loadedKeys) {
  const cases = {
    dotted_name_minimal: [{ kind: 'tool_call', name: 'a.b', arguments: {} }],
    dotted_name_mixed_punctuation: [{ kind: 'tool_call', name: 'my-tool.v2', arguments: { location: 'Beijing' } }],
    dotted_name_multiple_segments: [{ kind: 'tool_call', name: 'server.tools.weather', arguments: { location: 'Beijing' } }],
    dotted_name_shared_prefix: [{ kind: 'tool_call', name: 'server.weather', arguments: { location: 'Beijing' } }, { kind: 'tool_call', name: 'server.time', arguments: { timezone: 'Asia/Shanghai' } }],
  };
  const families = ['beam', 'deepseek_v4', 'deepseek_v41', 'gemma4', 'glm47', 'gpt_oss', 'kimi_k2', 'kimi_k3', 'muse_glimmer', 'muse_spark', 'nemotron_v3', 'qwen3'];
  for (const family of families) {
    const row = tab.rows.find(item => item.family === family);
    assert.ok(row, `${family}: dotted-name family is rendered`);
    for (const [scenario, expected] of Object.entries(cases)) {
      const cell = leafCells(row).get(scenario);
      const candidate = cell.tooltip.candidates.find(item => item.key === defaultKey);
      assert.deepEqual(canonicalUnifiedEvents(candidate.block.events), canonicalUnifiedEvents(expected), `${family}/${scenario}: exact dotted tool names and argument values`);
      assert.equal(candidate.block.verdict, 'MATCH');
      assert.equal(cell.cmp[defaultKey].na, 0);
      for (const other of cell.tooltip.candidates.filter(item => loadedKeys.has(item.key) && /^(dynamo_v1(?:_jail)?@|vllm_python_batch@0\.26\.0$)/.test(item.key))) {
        const descriptor = allUnifiedDescriptors.find(item => item.control.key === other.key);
        const measured = descriptor.source.implementation.startsWith('dynamo_v1')
          && catalog.capture_generation?.current.some(item => item.family === family
          && item.implementation === descriptor.source.implementation && item.version === descriptor.source.version);
        if (measured) {
          assert.notEqual(other.block.unavailable_code, 'missing_capture', `${family}/${scenario}: refreshed capture is retained`);
          assert.ok(Array.isArray(other.block.events) || other.block.unavailable === `No parser for ${family}`);
        } else {
          assert.equal(other.block.unavailable_code, 'missing_capture');
          assert.equal(Object.hasOwn(other.block, 'events'), false);
        }
      }
      if (family === 'gpt_oss') {
        assert.equal(candidate.parse_mode, 'unified');
        assert.ok(candidate.label.includes('UnifiedParser'));
        for (const [key, mode] of [['dynamo@0.7.27', 'split'], ['dynamo@0.7.28', 'unified']]) {
          if (!loadedKeys.has(key)) continue;
          const historical = cell.tooltip.candidates.find(item => item.key === key);
          assert.equal(historical.parse_mode, mode);
          assert.ok(historical.label.includes(mode === 'split' ? 'split' : 'UnifiedParser'));
        }
      }
    }
  }
  const golden = cases.dotted_name_mixed_punctuation;
  assert.equal(classifyUnified('gpt_oss', golden, golden, catalog.metadata.unified_leak_markers), 'MATCH');
  assert.notEqual(classifyUnified('gpt_oss', golden, [{ ...golden[0], name: 'my-tool' }], catalog.metadata.unified_leak_markers), 'MATCH');
  assert.notEqual(classifyUnified('gpt_oss', golden, [], catalog.metadata.unified_leak_markers), 'MATCH');
}
async function checkUnifiedBatch(batch) {
  const loadedKeys = new Set([defaultKey, ...batch.map(item => item.control.key)]);
  const reader = await newReader();
  try {
    const tab = await buildTab(catalog, reader, unifiedSpec.id, { selected: [...loadedKeys], reference: defaultKey });
    validateTab(tab, loadedKeys);
    validateUnifiedColumns(tab);
    await checkRebaseContracts(tab, reader);
    checkDottedNameContracts(tab, loadedKeys);
    const beam = tab.rows.find(row => row.family === 'beam');
    if (beam) {
      for (const scenario of ['glm_ref_object', 'glm_ref_encoded_targets', 'glm_ref_json_looking_strings', 'glm_ref_scalar_types']) {
        const cell = leafCells(beam).get(scenario);
        assert.equal(cell.cmp[defaultKey].na, 0, `Beam ${scenario}: current capture is measured`);
        for (const descriptor of batch.filter(item => catalog.capture_generation?.mode !== 'latest' && item.source.implementation === 'dynamo_v2' && compareLegacyVersions(item.source.version, '0.7.29') < 0)) {
          const comparison = cell.cmp[descriptor.control.key];
          assert.equal(comparison.na, 1, `Beam ${scenario}: ${descriptor.source.version} predates its introduction`);
          assert.equal(comparison.sig, 0);
        }
      }
    }
    assert.deepEqual(tab.rows.map(row => row.family), [...selectedUnifiedFamilies(catalog, unifiedDescriptors.filter(item => loadedKeys.has(item.control.key)))].sort());
    await checkUnifiedSelections(tab, reader, loadedKeys);
    for (const row of tab.rows) {
      testedFamilies.add(row.family);
      for (const [scenario, cell] of leafCells(row)) {
        testedScenarios.add(scenario);
        for (const key of Object.keys(cell.cmp)) if (key !== 'golden') testedKeys.add(key);
      }
    }
    console.log(`Unified candidate coverage: ${testedKeys.size}/${unifiedDescriptors.length}`);
  } finally {
    reader.dispose();
  }
}
// Each group owns its parsed resources and projected model. Retaining all
// historical candidates at once exceeds V8's heap without adding a cross-group assertion.
for (let offset = 0; offset < unifiedDescriptors.length; offset += 16) {
  await checkUnifiedBatch(unifiedDescriptors.slice(offset, offset + 16));
}
assert.deepEqual([...testedKeys].sort(), unifiedDescriptors.map(item => item.control.key).sort());
assert.deepEqual([...testedFamilies].sort(), [...selectedUnifiedFamilies(catalog, unifiedDescriptors)].sort());
assert.deepEqual([...testedScenarios].sort(), catalog.metadata.unified_presentation.authored_scenarios.filter(scenario => catalog.metadata.unified_presentation.scenario_families[scenario]?.some(family => testedFamilies.has(family))).sort());
assert.equal(singletonKeys.size, 0);
assert.equal(pageMetadata(catalog).schema, 1);
console.log(`Shared matrix structural contracts passed: ${catalog.metadata.tabs.length} tabs, ${testedKeys.size} Unified candidates, ${cells} visible cells, ${leaves} leaf appearances`);

}

export async function runMarkers(argv) {
const args = ['node', 'qualification.js', ...argv];
const { readFileSync } = await import('node:fs');
const { createRequire } = await import('node:module');


const require = createRequire(import.meta.url);
const { colorizeMarkup } = require('../utils/src/assets/colorize.js');
const { family, markers } = JSON.parse(readFileSync(0, 'utf8'));
const declared = markers[family] || {};
const failures = [];
for (const [open, close] of declared.pairs || []) {
  if (colorizeMarkup(`${open}x${close}`, family, markers).includes('tt-orphan')) {
    failures.push(`pair ${JSON.stringify(open)}/${JSON.stringify(close)} renders as tt-orphan`);
  }
}
for (const token of declared.singletons || []) {
  if (colorizeMarkup(String(token), family, markers).includes('tt-orphan')) {
    failures.push(`singleton ${JSON.stringify(token)} renders as tt-orphan`);
  }
}
process.stdout.write(JSON.stringify(failures));

}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  const [command, ...argv] = process.argv.slice(2);
  try {
    if (command === '--help') console.log('usage: node conformance/viewer/qualification.js status|snapshots|models|markers [arguments]');
    else if (command === 'status') {
      const options = parseStatusArguments(argv);
      if (options.help) console.log('status: --root CHECKOUT --model FAMILY --tab ID --status-path JSON --report-path PAGE --commit-sha SHA --known-divergences YAML --require-green --summary-only');
      else process.exitCode = await runStatus(options);
    } else if (command === 'snapshots') await runSnapshots(argv);
    else if (command === 'models') await runModels(argv);
    else if (command === 'markers') await runMarkers(argv);
    else throw new Error(`unknown qualification command: ${command}`);
  } catch (error) {
    console.error(error.message);
    process.exitCode = 2;
  }
}
