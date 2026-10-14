// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { resolve, dirname } from 'node:path';
import { createNodeResourceStore } from './data.js';
import { FixtureResources } from './resources.js';
import { buildTab, pageMetadata } from './page_model.js';
import { selectedCellOutcome } from './selection_comparison.js';
import { leafCells, isApplicableCell } from './folds.js';
import { canonicalUnifiedLabel } from './unified.js';

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
  try {
    const root = resolve(options.root);
    const catalog = JSON.parse(await readFile(resolve(root, 'conformance/viewer-catalog.json'), 'utf8'));
    const selected = selectTabs({ tabs: catalog.metadata.tabs.map((spec) => ({ ...spec, kind: spec.suite })) }, options.tabs);
    const commitSha = options.commitSha ?? (await execFileAsync('git', ['-C', root, 'rev-parse', 'HEAD'])).stdout.trim();
    const page = pageMetadata(catalog, { commitSha });
    reader = new FixtureResources(catalog, await createNodeResourceStore(root));
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
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  try {
    const options = parseStatusArguments(process.argv.slice(2));
    if (options.help) console.log('usage: node conformance/viewer/status.js [--root CHECKOUT] [--model FAMILY] [--tab ID|KIND|LABEL] [--status-path JSON] [--report-path PAGE] [--commit-sha SHA] [--known-divergences YAML] [--require-green] [--summary-only]\n--html is a compatibility alias for --report-path; status reads maintained source resources directly.');
    else process.exitCode = await runStatus(options);
  } catch (error) {
    console.error(`ERROR: ${error.message}`);
    process.exitCode = 2;
  }
}
