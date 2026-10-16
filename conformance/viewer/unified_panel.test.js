// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { unifiedPanel, unifiedComparisonFailure } from './unified_panel.js';
import { validateUnifiedCapture } from './unified.js';
import { resolveUnified } from './model.js';
import { tabCandidates } from './candidates.js';

const catalog = JSON.parse(await readFile(new URL('../viewer-catalog.json', import.meta.url), 'utf8'));
catalog.metadata.unified_presentation.authored_scenarios = ['tool_only'];
catalog.sources.unified = catalog.sources.unified.filter((source) => source.family === 'qwen3' && source.kind === 'inputs-and-golden');
catalog.sources.candidates = [{ mode: 'unified', implementation: 'dynamo_v2', version: '1.0.0', source: 'capture.yaml' }];
const request = { input: 'sample', init: { starting_state: 'None', tool_output_mode: 'Native', named_tool: null }, finish_reason: 'stop', tools: [], chunks: [{ delta_text: 'sample' }] };
const input = { lifecycle: 'active', scenario: 'tool_only', description: 'sample', policy: [], display_id: 'UNIFIED.1-tool_only', historical_ids: [], request, golden: { assembled: [] } };
const document = { input_document: { family: 'qwen3' }, golden_document: { family: 'qwen3' }, cases: { id: input } };
const identity = { path: 'capture.yaml', implementation: 'dynamo_v2', version: '1.0.0' };
for (const observation of [{ error: { code: 'panic', detail: 'parser failed' } }, { value: { assembled: [], chunks: [{ expected: [] }] } }]) {
  const capture = { provenance: { status: 'captured', origin: { crate_version: '1.0.0', source_sha256: 'a'.repeat(64) } },
    document: new Map([[1n, 'document metadata']]), changes: { id: { case_key: input.display_id, stimulus: { ref: 'current' }, observation } },
    metadata_changes: { id: new Map([[1n, 'record metadata']]) }, document_overrides: {} };
  await validateUnifiedCapture(capture, document, identity);
  const reader = { yaml: async () => ({ value: document }), unified: async () => ({ records: resolveUnified([{ ...identity, document: capture }], '1.0.0') }) };
  const panel = await unifiedPanel(catalog, reader);
  assert.equal(panel.rows[0].model_label, 'qwen3 (v2)');
  const cell = panel.rows[0].cells.tool_only;
  const candidate = cell.tooltip.candidates.find((candidate) => candidate.key === 'dynamo');
  if (observation.error) {
    assert.equal(cell.status, 'problem');
    assert.equal(cell.cmp.dynamo.err, 1);
    assert.notEqual(cell.cmp.dynamo.sig, cell.cmp.golden.sig);
    assert.equal(candidate.block.error, 'parser failed');
  } else {
    assert.equal(cell.status, 'ok');
    assert.equal(cell.cmp.dynamo.err, 0);
    assert.equal(candidate.block.verdict, 'MATCH');
  }
}
assert.equal(unifiedComparisonFailure(request, { capture_input: request }), null);
assert.match(unifiedComparisonFailure(request, {}), /original input/);
assert.match(unifiedComparisonFailure(request, { capture_input: { ...request, tools: null } }), /original tool schema/);
assert.match(unifiedComparisonFailure(request, { capture_input: { ...request, input: 'different' } }), /mismatch \(input\)/);
assert.throws(() => unifiedComparisonFailure(request, { capture_input: { input: 'sample', tools: [] } }), /incomplete/);
const missingCaptureReader = { yaml: async () => ({ value: document }), unified: async () => ({ records: new Map() }) };
const missingCapturePanel = await unifiedPanel(catalog, missingCaptureReader);
assert.equal(missingCapturePanel.rows[0].cells.tool_only.cmp.dynamo.err, 1);
assert.match(missingCapturePanel.rows[0].cells.tool_only.tooltip.candidates.find(({ key }) => key === 'dynamo').block.error, /Missing Unified capture/);
const qwen3Generation = catalog.metadata.unified_presentation.parser_generations.qwen3;
for (const label of ['v1', 'v1/v2 split', 'v2']) {
  catalog.metadata.unified_presentation.parser_generations.qwen3 = { label };
  const labeledPanel = await unifiedPanel(catalog, missingCaptureReader);
  assert.equal(labeledPanel.rows[0].model_label, `qwen3 (${label})`);
}
catalog.metadata.unified_presentation.parser_generations.qwen3 = qwen3Generation;
catalog.metadata.unified_presentation.v1_split_scenarios = ['tool_only'];
const v1SplitPanel = await unifiedPanel(catalog, missingCaptureReader);
assert.equal(v1SplitPanel.rows[0].cells.tool_only.cmp.dynamo.na, 1);
assert.match(v1SplitPanel.rows[0].cells.tool_only.tooltip.candidates.find(({ key }) => key === 'dynamo').block.unavailable, /Dynamo v1 split parser/);
catalog.metadata.unified_presentation.v1_split_scenarios = [];
const retiredDocument = structuredClone(document);
retiredDocument.cases.retired = { ...structuredClone(input), lifecycle: 'retired', display_id: null, historical_ids: [], request: null, golden: null, description: null };
const good = { provenance: { status: 'captured', origin: { crate_version: '1.0.0', source_sha256: 'a'.repeat(64) } }, document: {},
  changes: { id: { case_key: input.display_id, stimulus: { ref: 'current' }, observation: { value: { assembled: [], chunks: [] } } },
    retired: { case_key: 'UNIFIED.unrelated', stimulus: { inline: request }, observation: { value: { assembled: [{ kind: 'text', text: 'BAD' }], chunks: [{ expected: [{ kind: 'text', text: 'BAD' }] }] } } } }, metadata_changes: {}, document_overrides: {} };
await validateUnifiedCapture(good, retiredDocument, identity);
const retiredReader = { yaml: async () => ({ value: retiredDocument }), unified: async () => ({ records: resolveUnified([{ ...identity, document: good }], '1.0.0') }) };
let panel = await unifiedPanel(catalog, retiredReader);
assert.equal(panel.rows[0].cells.tool_only.tooltip.candidates.find(candidate => candidate.key === 'dynamo').block.verdict, 'MATCH');
delete good.changes.id;
retiredDocument.cases.id.historical_ids = ['UNIFIED.1.a'];
good.changes.retired.case_key = 'UNIFIED.1.a';
good.changes.retired.stimulus.inline = { ...request, input: 'old', chunks: [{ delta_text: 'old' }] };
await validateUnifiedCapture(good, retiredDocument, identity);
panel = await unifiedPanel(catalog, retiredReader);
assert.match(panel.rows[0].cells.tool_only.tooltip.candidates.find(candidate => candidate.key === 'dynamo').block.unavailable, /mismatch \(input, chunks\)/);
assert.equal(panel.rows[0].cells.tool_only.cmp.dynamo.err, 0);
assert.equal(panel.rows[0].cells.tool_only.cmp.dynamo.na, 1);

catalog.sources.candidates.push(
  { mode: 'unified', implementation: 'dynamo_v2', version: '0.9.0', source: 'older.yaml' },
  { mode: 'unified', implementation: 'vllm_python', version: '0.26.0', source: 'peer.yaml' },
);
const controls = tabCandidates(catalog, catalog.metadata.tabs.find((tab) => tab.id === 'tab-unified'));
const peerKey = controls.find(({ source }) => source.implementation === 'vllm_python').control.key;
const olderKey = controls.find(({ source }) => source.implementation === 'dynamo_v2' && source.version === '0.9.0').control.key;
const loaded = [];
const selectionReader = {
  yaml: async () => ({ value: document }),
  unified: async (family, implementation, version) => {
    loaded.push(`${implementation}@${version}`);
    const observation = implementation === 'vllm_python' || version === '0.9.0'
      ? { error: { code: 'panic', detail: 'selected parser failed' } }
      : { value: { assembled: [], chunks: [] } };
    const capture = { provenance: { status: 'legacy', captured_with: { [implementation]: version } }, document: {},
      changes: { id: { case_key: input.display_id, stimulus: { ref: 'current' }, observation } }, metadata_changes: {}, document_overrides: {} };
    const source = { path: `${implementation}-${version}.yaml`, implementation, version };
    await validateUnifiedCapture(capture, document, source);
    return { records: resolveUnified([{ ...source, document: capture }], version) };
  },
};
const selectedCell = async (selected, reference) => {
  const result = await unifiedPanel(catalog, selectionReader, { selected, reference });
  return { panel: result, cell: result.rows[0].cells.tool_only };
};
let selection = await selectedCell([peerKey], peerKey);
assert.deepEqual(loaded, ['vllm_python@0.26.0']);
assert.equal(selection.cell.status, 'problem');
assert.equal(selection.cell.cmp[peerKey].err, 1);
assert.equal(selection.cell.tooltip.candidates.find(({ key }) => key === peerKey).block.error, 'selected parser failed');
selection = await selectedCell([olderKey, 'dynamo'], olderKey);
assert.equal(selection.cell.status, 'problem');
selection = await selectedCell([olderKey, 'dynamo'], 'dynamo');
assert.equal(selection.cell.status, 'ok');
selection = await selectedCell([peerKey], 'golden');
assert.equal(selection.cell.status, 'ok');
selection = await selectedCell([peerKey], null);
assert.equal(selection.cell.status, 'ok');
assert.equal(selection.panel.stats.fixture_cases, 1);
loaded.length = 0;
selection = await selectedCell([], null);
assert.deepEqual(loaded, []);
assert.equal(selection.panel.stats.fixture_cases, 1);
await assert.rejects(selectedCell([peerKey], 'dynamo'), /reference is not loaded/);
await assert.rejects(selectedCell([peerKey], 'unknown'), /unknown Unified reference/);
console.log('Unified projected mapping values, recorded errors, and stimulus admission passed');
