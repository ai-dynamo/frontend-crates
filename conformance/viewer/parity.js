// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { readFile } from 'node:fs/promises';
import { createReadStream } from 'node:fs';
import { createInterface } from 'node:readline';
import { fileURLToPath } from 'node:url';
import { createNodeResourceStore } from './data.js';
import { FixtureResources } from './resources.js';
import { parseJson, sameValue, yamlTemporal, stringifyValue } from './values.js';
import { materializeUnifiedRecord } from './unified.js';
import { classifyUnified } from './model.js';
import { comparisonEntries, comparisonFacts, declaredLeakTokens, equivalenceGroups, legacyOutcome } from './comparison.js';
import { legacyCases } from './legacy.js';
import { tabCandidates } from './candidates.js';
import { foldNullVariants, foldSchemaVariants, rankUnifiedColumns } from './folds.js';
import { toolcallingPanel } from './toolcalling.js';
import { unifiedPanel } from './unified_panel.js';
import { reasoningPanel } from './reasoning_panel.js';
import { buildTab, pageMetadata } from './page_model.js';

function firstDifference(expected, actual, path = '$') {
  if (sameValue(expected, actual)) return null;
  if (typeof expected !== typeof actual || expected === null || actual === null || typeof expected !== 'object' || Array.isArray(expected) !== Array.isArray(actual)) return path;
  if (expected instanceof Map || actual instanceof Map) return `${path} (typed mapping)`;
  for (const key of new Set([...Object.keys(expected), ...Object.keys(actual)])) {
    if (!Object.hasOwn(expected, key) || !Object.hasOwn(actual, key)) return `${path}.${key} (missing field)`;
    const difference = firstDifference(expected[key], actual[key], `${path}.${key}`);
    if (difference) return difference;
  }
  return path;
}

function decodeTyped(value) {
  const [type, content] = value;
  switch (type) {
    case 'int': return BigInt(content);
    case 'float': return content === 'nan' ? NaN : content === 'inf' ? Infinity : content === '-inf' ? -Infinity : Number(content);
    case 'null': return null;
    case 'str': case 'bool': return content;
    case 'array': return content.map(decodeTyped);
    case 'object': {
      const pairs = content.map(([key, child]) => [decodeTyped(key), decodeTyped(child)]);
      return pairs.every(([key]) => typeof key === 'string') ? Object.fromEntries(pairs) : new Map(pairs);
    }
    case 'date': case 'datetime': return yamlTemporal(content, type);
    default: throw new Error(`unsupported oracle scalar type: ${type}`);
  }
}

function normalizeModelNumbers(tab) {
  delete tab.cand_meta;
  delete tab.fixture_href_base;
  for (const key of Object.keys(tab.stats || {})) tab.stats[key] = Number(tab.stats[key]);
  for (const group of tab.column_groups || []) group.span = Number(group.span);
  for (const column of tab.columns || []) if (Object.hasOwn(column, 'display_number')) column.display_number = Number(column.display_number);
  const visit = (cell) => {
    for (const entry of Object.values(cell.cmp || {})) for (const key of ['na', 'err', 'leak']) entry[key] = Number(entry[key]);
    for (const child of cell.variants || []) visit(child);
  };
  for (const row of tab.rows) for (const cell of Object.values(row.cells)) visit(cell);
  return tab;
}

function normalizeModelGroups(tab) {
  const visit = (cell) => {
    if (cell.cmp) cell.cmp = equivalenceGroups(cell.cmp);
    for (const child of cell.variants || []) visit(child);
  };
  for (const row of tab.rows) for (const cell of Object.values(row.cells)) visit(cell);
  return tab;
}

const args = process.argv.slice(2);
if (args.includes('--help')) {
  console.log('usage: node conformance/viewer/parity.js --baseline EVIDENCE_DIRECTORY [--root CHECKOUT] [--parsed | --resolved | --processed | --comparison | --controls | --folds | --cells | --models | --unified-cells | --reasoning-cells]');
} else {
  try {
    const options = {};
    for (let index = 0; index < args.length; index++) {
      if (['--parsed', '--resolved', '--processed', '--comparison', '--controls', '--folds', '--cells', '--models', '--unified-cells', '--reasoning-cells'].includes(args[index])) options[args[index].slice(2)] = true;
      else if (['--baseline', '--root'].includes(args[index]) && args[index + 1]) options[args[index].slice(2)] = args[++index];
      else throw new Error(`invalid argument: ${args[index]}`);
    }
    if (!options.baseline) throw new Error('--baseline is required');
    if (!Object.keys(options).some((key) => ['parsed', 'resolved', 'processed', 'comparison', 'controls', 'folds', 'cells', 'models', 'unified-cells', 'reasoning-cells'].includes(key))) Object.assign(options, { parsed: true, resolved: true, processed: true, comparison: true, controls: true });
    const root = options.root || fileURLToPath(new URL('../../', import.meta.url));
    const catalog = JSON.parse(await readFile(`${root}/conformance/viewer-catalog.json`, 'utf8'));
    const reader = new FixtureResources(catalog, await createNodeResourceStore(root));
    let parsed = 0;
    let snapshots = 0;
    if (options.folds) {
      const preFold = createInterface({ input: createReadStream(`${options.baseline}/pre-fold-tabs.jsonl`), crlfDelay: Infinity });
      const final = createInterface({ input: createReadStream(`${options.baseline}/folded-tabs-original.jsonl`), crlfDelay: Infinity })[Symbol.asyncIterator]();
      let count = 0;
      for await (const line of preFold) {
        const actual = normalizeModelNumbers(parseJson(line));
        const expected = normalizeModelNumbers(parseJson((await final.next()).value));
        foldNullVariants(actual, catalog.metadata);
        foldSchemaVariants(actual, catalog.metadata);
        rankUnifiedColumns(actual, catalog.metadata);
        const difference = firstDifference(normalizeModelGroups(expected), normalizeModelGroups(actual));
        if (difference) throw new Error(`fold model mismatch ${actual.id}:${difference}`);
        count++;
      }
      console.log(`exact independently folded model parity: ${count} tabs`);
    }
    if (options.models) {
      const expected = parseJson(await readFile(`${options.baseline}/page-model-meta.json`, 'utf8'));
      expected.schema = Number(expected.schema);
      const actual = pageMetadata(catalog, { commitSha: expected.meta.sha });
      const markers = parseJson(await readFile(`${options.baseline}/family-markers.json`, 'utf8'));
      const { family_markers, ...base } = actual;
      const difference = firstDifference(expected, base) || firstDifference(markers, family_markers, '$.family_markers');
      if (difference) throw new Error(`page metadata mismatch:${difference}`);
      console.log('exact page metadata and family marker parity');
    }
    if (options.cells || options.models || options['unified-cells'] || options['reasoning-cells']) {
      const lines = createInterface({ input: createReadStream(`${options.baseline}/page-model-tabs.jsonl`), crlfDelay: Infinity });
      let count = 0;
      for await (const line of lines) {
        const expected = normalizeModelNumbers(parseJson(line));
        if (options['unified-cells'] && expected.kind !== 'unified') continue;
        if (options['reasoning-cells'] && expected.kind !== 'reasoning') continue;
        const spec = catalog.metadata.tabs.find((tab) => tab.id === expected.id);
        const actual = normalizeModelGroups(options.models ? await buildTab(catalog, reader, spec.id) : expected.kind === 'unified' ? await unifiedPanel(catalog, reader) : expected.kind === 'reasoning' ? await reasoningPanel(catalog, reader, spec) : await toolcallingPanel(catalog, reader, spec));
        let tabCount = 0;
        for (const row of expected.rows) {
          if (row.section) continue;
          const candidate = actual.rows.find((entry) => entry.family === row.family);
          for (const [sub, cell] of Object.entries(row.cells)) {
            const target = candidate?.cells[sub];
            if (cell.fixture_href?.startsWith('file://SNAPSHOT_ROOT/')) cell.fixture_href = cell.fixture_href.slice('file://SNAPSHOT_ROOT/'.length);
            const difference = firstDifference(cell, target);
            const detail = difference?.match(/\.tooltip\.candidates\.(\d+)\.block\.(unavailable|explanation)$/);
            if (detail) {
              const index = Number(detail[1]), field = detail[2];
              console.error(`candidate ${cell.tooltip.candidates[index].key}: ${field}: expected ${stringifyValue(cell.tooltip.candidates[index].block[field])}; actual ${stringifyValue(target.tooltip.candidates[index].block[field])}`);
            }
            if (difference) throw new Error(`cell model mismatch ${expected.id}:${row.family}:${sub}:${difference}`);
            count++;
            tabCount++;
          }
        }
        console.log(`exact independently constructed cell-model parity: ${expected.id}: ${tabCount} cells`);
        const rowDifference = firstDifference(expected.rows, actual.rows);
        if (rowDifference) throw new Error(`row/parser presentation mismatch ${expected.id}:${rowDifference}`);
        for (const field of ['columns', 'column_groups', 'candidates', 'glossary', 'stats']) {
          const difference = firstDifference(expected[field], actual[field]);
          if (difference && field === 'stats') console.error(`statistics: expected ${JSON.stringify(expected.stats)}, actual ${JSON.stringify(actual.stats)}`);
          if (difference) throw new Error(`matrix axes mismatch ${expected.id}:${field}:${difference}`);
        }
        console.log(`exact matrix axes/statistics parity: ${expected.id}`);
        if (options.models) {
          const difference = firstDifference(expected, actual);
          if (difference) throw new Error(`complete tab model mismatch ${expected.id}:${difference}`);
          console.log(`exact complete tab model parity: ${expected.id}`);
        }
      }
      console.log(`exact independently constructed cell-model parity: ${count} cells`);
    }
    if (options.controls) {
      const lines = createInterface({ input: createReadStream(`${options.baseline}/page-model-tabs.jsonl`), crlfDelay: Infinity });
      let count = 0;
      for await (const line of lines) {
        const expected = JSON.parse(line);
        const spec = catalog.metadata.tabs.find((tab) => tab.id === expected.id);
        const actual = tabCandidates(catalog, spec, { golden: expected.candidates.some((item) => item.key === 'golden') }).map((item) => item.control);
        const difference = firstDifference(expected.candidates, actual);
        if (difference) throw new Error(`candidate controls mismatch ${expected.id}:${difference}`);
        count++;
      }
      console.log(`exact candidate control parity: ${count} tabs`);
    }
    if (options.parsed) {
      const lines = createInterface({ input: createReadStream(`${options.baseline}/parsed-scalars.jsonl`), crlfDelay: Infinity });
      for await (const line of lines) {
        const oracle = JSON.parse(line);
        const resource = [...catalog.sources.archives, ...catalog.sources.unified].find((item) => item.path === oracle.resource);
        if (!resource) throw new Error(`oracle resource absent from catalog: ${oracle.resource}`);
        const actual = (await reader.yaml(resource, oracle.member || null, !oracle.member)).value;
        const difference = firstDifference(decodeTyped(oracle.value), actual);
        if (difference) throw new Error(`parsed mismatch ${oracle.resource}:${oracle.member || ''}:${difference}`);
        parsed++;
      }
      console.log(`exact parsed-value parity: ${parsed} documents`);
    }
    if (options.resolved) {
      const lines = createInterface({ input: createReadStream(`${options.baseline}/resolved-snapshots.jsonl`), crlfDelay: Infinity });
      for await (const line of lines) {
        const record = parseJson(line);
        const mode = record.mode;
        if (mode === 'batch-on-stream') {
          const resource = catalog.sources.archives.find((item) => item.path === 'conformance/fixtures/toolcalling/fixtures-batch-on-stream-v1.tar.gz');
          if (!resource) throw new Error('batch-on-stream archive missing');
          const actual = await reader.legacyDocuments(resource);
          if (actual.size !== record.documents.length) throw new Error('batch-on-stream document count mismatch');
          for (const expected of record.documents) {
            const difference = firstDifference(expected.document, actual.get(`${expected.family}/${expected.member}`)?.value);
            if (difference) throw new Error(`resolved mismatch ${mode}:${expected.family}:${expected.member}:${difference}`);
          }
          snapshots++;
          continue;
        }
        if (mode !== 'unified') {
          const rootPath = mode === 'reasoning' ? 'reasoning/fixtures-v1' : `toolcalling/fixtures-${mode}-v1`;
          const documents = await reader.legacy({ root: rootPath, mode, implementation: record.implementation, version: record.version });
          if (documents.size !== record.documents.length) throw new Error(`resolved document count mismatch: ${mode}:${record.implementation}:${record.version}`);
          for (const expected of record.documents) {
            const actual = documents.get(`${expected.family}/${expected.member}`)?.value;
            const difference = firstDifference(expected.document, actual);
            if (difference) throw new Error(`resolved mismatch ${mode}:${expected.family}:${expected.member}:${record.implementation}:${record.version}:${difference}`);
          }
          snapshots++;
          continue;
        }
        const { records } = await reader.unified(record.family, record.implementation, record.version);
        const resolved = Object.fromEntries([...records].map(([key, item]) => [key, item.value]));
        const difference = firstDifference(record.records, resolved);
        if (difference) throw new Error(`resolved mismatch unified:${record.family}:${record.implementation}:${record.version}:${difference}`);
        snapshots++;
      }
      console.log(`exact resolved-record parity: ${snapshots} snapshots`);
      const views = parseJson(await readFile(`${options.baseline}/unified-release-views.json`, 'utf8'));
      const families = [...new Set(catalog.sources.unified.filter((file) => file.kind === 'inputs-and-golden').map((file) => file.family))];
      for (const view of views) {
        const documents = new Map();
        for (const family of families) {
          const { inputs, records } = await reader.unified(family, view.implementation, view.version);
          for (const [caseId, record] of records) {
            const document = materializeUnifiedRecord(inputs.value.cases[caseId], record, family);
            documents.set(`${family}/${Object.keys(document.cases)[0]}.yaml`, document);
          }
        }
        if (documents.size !== view.documents.length) throw new Error(`release-view document count mismatch: ${view.implementation}:${view.version}`);
        for (const expected of view.documents) {
          const difference = firstDifference(expected.document, documents.get(`${expected.family}/${expected.member}`));
          if (difference) throw new Error(`release-view mismatch ${expected.family}:${expected.member}:${view.implementation}:${view.version}:${difference}`);
        }
      }
      console.log(`exact cross-family carry-forward parity: ${views.length} release views`);
    }
    if (options.processed) {
      const tokens = declaredLeakTokens(catalog.metadata.markers);
      const lines = createInterface({ input: createReadStream(`${options.baseline}/processed-cases.jsonl`), crlfDelay: Infinity });
      let snapshots = 0;
      let caseCount = 0;
      for await (const line of lines) {
        const expected = parseJson(line);
        const stream = ['stream', 'streamv1'].includes(expected.mode);
        const documents = await reader.legacy({ root: `toolcalling/fixtures-${stream ? 'stream' : 'batch'}-v1`, mode: stream ? 'stream' : 'batch', implementation: expected.implementation, version: expected.version });
        const actual = legacyCases(documents, stream ? 'streamv1' : 'batch', catalog.metadata, stream ? [expected.implementation] : null);
        if (actual.cases.size !== expected.cases.length) throw new Error(`processed case count mismatch ${expected.mode}:${expected.implementation}:${expected.version}`);
        for (const field of ['labels', 'capturedWith']) {
          const difference = firstDifference(expected[field === 'capturedWith' ? 'captured_with' : field], actual[field]);
          if (difference) throw new Error(`processed metadata mismatch ${expected.mode}:${expected.implementation}:${expected.version}:${field}:${difference}`);
        }
        const implementations = expected.comparison_implementations.map((key) => catalog.metadata.implementations.find((item) => item.key === key));
        for (const entry of expected.cases) {
          const candidate = actual.cases.get(`${entry.family}\0${entry.sub}`);
          const difference = firstDifference(entry.record, candidate?.record);
          if (difference) throw new Error(`processed mismatch ${expected.mode}:${entry.family}:${entry.sub}:${expected.implementation}:${expected.version}:${difference}`);
          const sourceTarget = expected.source_targets[entry.source_target]?.input_target;
          if (sourceTarget !== candidate.source.member) throw new Error(`source target mismatch ${expected.mode}:${entry.family}:${entry.sub}:${expected.implementation}:${expected.version}:input_target`);
          const facts = comparisonFacts(candidate.record, implementations, expected.comparison_baseline, tokens, catalog.metadata.recording.rustUnavailable);
          const factDifference = firstDifference(entry.comparison_facts, facts);
          if (factDifference) throw new Error(`comparison facts mismatch ${expected.mode}:${entry.family}:${entry.sub}:${expected.implementation}:${expected.version}:${factDifference}`);
          const status = legacyOutcome(candidate.record.expected?.[expected.implementation], tokens).status;
          if (status !== entry.status) throw new Error(`status mismatch ${expected.mode}:${entry.family}:${entry.sub}:${expected.implementation}:${expected.version}`);
          caseCount++;
        }
        snapshots++;
      }
      console.log(`exact independently processed case/fact/status parity: ${caseCount} cases across ${snapshots} snapshots`);
    }
    if (options.comparison) {
      const tokens = declaredLeakTokens(catalog.metadata.markers);
      const lines = createInterface({ input: createReadStream(`${options.baseline}/page-model-tabs.jsonl`), crlfDelay: Infinity });
      let legacyCells = 0;
      let unifiedVerdicts = 0;
      for await (const line of lines) {
        const tab = parseJson(line);
        for (const row of tab.rows) for (const cell of Object.values(row.cells)) {
          if (!cell.cmp || !cell.tooltip) continue;
          if (tab.kind === 'toolcalling' && tab.mode !== 'unified') {
            const blocks = Object.fromEntries(cell.tooltip.candidates.map((candidate) => [candidate.key, candidate.block]));
            for (const [key, entry] of Object.entries(cell.cmp)) {
              if (!Object.hasOwn(blocks, key) && entry.na) blocks[key] = null;
            }
            const actual = equivalenceGroups(comparisonEntries(blocks, tokens));
            const expected = Object.fromEntries(Object.entries(cell.cmp).map(([key, entry]) => [key, { ...entry, na: Number(entry.na), err: Number(entry.err), leak: Number(entry.leak) }]));
            const difference = firstDifference(expected, actual);
            if (difference) throw new Error(`comparison mismatch ${tab.id}:${row.family}:${cell.case_id}:${difference}`);
            legacyCells++;
          }
          if (tab.mode === 'unified') {
            const golden = cell.tooltip.candidates.find((candidate) => candidate.key === 'golden')?.block?.events;
            if (!golden) continue;
            for (const candidate of cell.tooltip.candidates) {
              if (!candidate.block?.verdict || !candidate.block.events) continue;
              const verdict = classifyUnified(row.family, golden, candidate.block.events, catalog.metadata.unified_leak_markers);
              if (verdict !== candidate.block.verdict) throw new Error(`classification mismatch ${tab.id}:${row.family}:${cell.case_id}:${candidate.key}:verdict`);
              unifiedVerdicts++;
            }
          }
        }
      }
      console.log(`exact legacy comparison parity: ${legacyCells} cells; exact Unified classification parity: ${unifiedVerdicts} verdicts`);
    }
    reader.dispose();
  } catch (error) {
    console.error(error.message);
    process.exitCode = 2;
  }
}
