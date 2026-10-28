// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { test } from 'node:test';

await test('unified_panel contracts', async () => {
const savedGlobals = Object.getOwnPropertyDescriptors(globalThis);
try {
const { default: assert } = await import('node:assert/strict');
const { readFile } = await import('node:fs/promises');
const { unifiedPanel, unifiedComparisonFailure } = await import('./presentation.js');
const { leafCells } = await import('./model.js');
const { validateUnifiedCapture } = await import('./resources.js');
const { resolveUnified } = await import('./model.js');
const { tabCandidates, sourceCapturesFamily } = await import('./model.js');
const { selectedCaptureOutcome } = await import('./model.js');


const toolOnlyCell = (panel, family = panel.rows[0].family) => leafCells(panel.rows.find(row => row.family === family)).get('tool_only');
const catalog = JSON.parse(await readFile(new URL('../viewer-catalog.json', import.meta.url), 'utf8'));
catalog.capture_generation = { mode: 'history' };
catalog.metadata.unified_presentation.authored_scenarios = ['tool_only'];
catalog.sources.unified = catalog.sources.unified.filter((source) => source.family === 'qwen3' && source.kind === 'inputs-and-golden');
catalog.sources.candidates = [{ mode: 'unified', implementation: 'dynamo_v2', version: '1.0.0', source: 'capture.yaml', native_unified_families: ['qwen3'] }];
const postCatalog = structuredClone(catalog);
for (const [implementation, versions] of [
  ['vllm_python', ['0.8.5', '0.8.5.post1']],
  ['sglang_python', ['0.4.4.post2', '0.4.4', '0.4.4.post3']],
]) postCatalog.sources.candidates.push(...versions.map(version => ({ mode: 'unified', implementation, version })));
const postCandidates = tabCandidates(postCatalog, { id: 'tab-unified', suite: 'toolcalling', mode: 'unified' });
assert.deepEqual(postCandidates.filter(item => item.source.implementation === 'vllm_python').map(item => item.source.version), ['0.8.5.post1', '0.8.5']);
assert.deepEqual(postCandidates.filter(item => item.source.implementation === 'sglang_python').map(item => item.source.version), ['0.4.4.post3', '0.4.4.post2', '0.4.4']);
const latestCatalog = structuredClone(postCatalog);
latestCatalog.capture_generation = { mode: 'latest' };
latestCatalog.sources.candidates.push({ mode: 'unified', implementation: 'dynamo_v2', version: '0.9.0' });
const latestCandidates = tabCandidates(latestCatalog, { id: 'tab-unified', suite: 'toolcalling', mode: 'unified' });
assert.deepEqual(latestCandidates.filter(item => item.source.implementation === 'dynamo_v2').map(item => item.source.version), ['1.0.0']);
assert.deepEqual(latestCandidates.filter(item => item.source.implementation === 'vllm_python').map(item => item.source.version), ['0.8.5.post1']);
assert.deepEqual(latestCandidates.filter(item => item.source.implementation === 'sglang_python').map(item => item.source.version), ['0.4.4.post3']);
const request = { input: 'sample', init: { starting_state: 'None', tool_output_mode: 'Native', named_tool: null }, finish_reason: 'stop', tools: [], chunks: [{ delta_text: 'sample' }] };
const input = { lifecycle: 'active', scenario: 'tool_only', description: 'sample', policy: [], display_id: 'UNIFIED.1-tool_only', historical_ids: [], request, golden: { assembled: [] } };
const document = { input_document: { family: 'qwen3' }, golden_document: { family: 'qwen3' }, cases: { id: input } };
const identity = { path: 'capture.yaml', implementation: 'dynamo_v2', version: '1.0.0' };
for (const observation of [{ error: { code: 'panic', detail: 'parser failed' } }, { value: { assembled: [], chunks: [{ expected: [] }] } },
  ...['unsupported_request_mode', 'unsupported_capture_mode', 'unsupported_family', 'source_unavailable'].map(code => ({ unavailable: { code, detail: 'Capability unavailable' } }))]) {
  const capture = { provenance: { status: 'captured', origin: { crate_version: '1.0.0', source_sha256: 'a'.repeat(64) } },
    document: new Map([[1n, 'document metadata']]), changes: { id: { case_key: input.display_id, stimulus: { ref: 'current' }, observation } },
    metadata_changes: { id: new Map([[1n, 'record metadata']]) }, document_overrides: {} };
  await validateUnifiedCapture(capture, document, identity);
  const reader = { unifiedInputs: async () => ({ value: document }), unified: async () => ({ records: resolveUnified([{ ...identity, document: capture }], '1.0.0') }) };
  const panel = await unifiedPanel(catalog, reader);
  assert.equal(panel.rows[0].model_label, 'qwen3');
  const cell = toolOnlyCell(panel);
  const candidate = cell.tooltip.candidates.find((candidate) => candidate.key === 'dynamo');
  if (observation.error) {
    assert.equal(cell.status, 'problem');
    assert.equal(cell.cmp.dynamo.err, 1);
    assert.notEqual(cell.cmp.dynamo.sig, cell.cmp.golden.sig);
    assert.equal(candidate.block.error, 'parser failed');
  } else if (observation.unavailable) {
    assert.equal(cell.cmp.dynamo.na, 1);
    assert.equal(candidate.block.unavailable_code, observation.unavailable.code);
    assert.equal(selectedCaptureOutcome(cell, 'dynamo'), observation.unavailable.code === 'source_unavailable' ? 'blocked' : 'unsupported');
  } else {
    assert.equal(cell.status, 'ok');
    assert.equal(cell.cmp.dynamo.err, 0);
    assert.equal(candidate.block.verdict, 'MATCH');
  }
}
for (const [expected,actual,verdict] of [
  ['9007199254740992.5','9007199254740992.5','MATCH'],
  ['9007199254740992.5','9007199254740992.0','ARG_MISMATCH'],
  ['0.10000000000000000001','0.1','ARG_MISMATCH'],
  ['42.0','4.2e1','MATCH'], ['1e-400','0','ARG_MISMATCH'],
]) {
  const numericDocument=structuredClone(document);
  const golden={kind:'tool_call',name:'f',arguments:`{"value":${expected}}`};
  numericDocument.cases.id.golden.assembled=[golden];
  const delta={kind:'tool_call',name:'f',arguments:`{"value":${actual}}`,complete:true};
  const capture={provenance:{status:'captured',origin:{crate_version:'1.0.0',source_sha256:'a'.repeat(64)}},document:{},
    changes:{id:{case_key:input.display_id,stimulus:{ref:'current'},observation:{value:{assembled:[golden],chunks:[{expected:[delta]}]}}}},metadata_changes:{},document_overrides:{}};
  await validateUnifiedCapture(capture,numericDocument,identity);
  const reader={unifiedInputs:async()=>({value:numericDocument}),unified:async()=>({records:resolveUnified([{...identity,document:capture}],'1.0.0')})};
  const numericPanel=await unifiedPanel(catalog,reader), cell=toolOnlyCell(numericPanel);
  const output=cell.tooltip.candidates.find(candidate=>candidate.key==='dynamo').block;
  assert.equal(output.events[0].arguments,delta.arguments);
  assert.equal(output.verdict,verdict);
  assert.equal(cell.cmp.dynamo.sig===cell.cmp.golden.sig,verdict==='MATCH');
}
assert.equal(unifiedComparisonFailure(request, { capture_input: request }), null);
assert.match(unifiedComparisonFailure(request, {}), /Original input/);
assert.match(unifiedComparisonFailure(request, { capture_input: { ...request, tools: null } }), /Original tool schema/);
assert.match(unifiedComparisonFailure(request, { capture_input: { ...request, input: 'different' } }), /mismatch \(input\)/);
assert.throws(() => unifiedComparisonFailure(request, { capture_input: { input: 'sample', tools: [] } }), /incomplete/);
const missingCaptureReader = { unifiedInputs: async () => ({ value: document }), unified: async () => ({ records: new Map() }) };
const missingCapturePanel = await unifiedPanel(catalog, missingCaptureReader);
assert.equal(toolOnlyCell(missingCapturePanel).cmp.dynamo.err, 0);
assert.equal(selectedCaptureOutcome(toolOnlyCell(missingCapturePanel), 'dynamo'), 'missing');
assert.deepEqual(toolOnlyCell(missingCapturePanel).tooltip.candidates.find(({ key }) => key === 'dynamo').block, {
  unavailable: 'Output not yet captured', unavailable_code: 'missing_capture',
});
const qwen3Generation = catalog.metadata.unified_presentation.parser_generations.qwen3;
for (const label of ['v1', 'v1/v2 split', 'v2']) {
  catalog.metadata.unified_presentation.parser_generations.qwen3 = { label };
  const labeledPanel = await unifiedPanel(catalog, missingCaptureReader);
  assert.equal(labeledPanel.rows[0].model_label, 'qwen3');
}
catalog.metadata.unified_presentation.parser_generations.qwen3 = qwen3Generation;
catalog.metadata.unified_presentation.v1_split_scenarios = ['tool_only'];
const v1SplitPanel = await unifiedPanel(catalog, missingCaptureReader);
assert.equal(toolOnlyCell(v1SplitPanel).cmp.dynamo.err, 0);
assert.equal(selectedCaptureOutcome(toolOnlyCell(v1SplitPanel), 'dynamo'), 'missing');
assert.equal(toolOnlyCell(v1SplitPanel).tooltip.candidates.find(({ key }) => key === 'dynamo').block.unavailable, 'Output not yet captured');
catalog.metadata.unified_presentation.v1_split_scenarios = [];
const retiredDocument = structuredClone(document);
retiredDocument.cases.retired = { ...structuredClone(input), lifecycle: 'retired', display_id: null, historical_ids: [], request: null, golden: null, description: null };
const good = { provenance: { status: 'captured', origin: { crate_version: '1.0.0', source_sha256: 'a'.repeat(64) } }, document: {},
  changes: { id: { case_key: input.display_id, stimulus: { ref: 'current' }, observation: { value: { assembled: [], chunks: [] } } },
    retired: { case_key: 'UNIFIED.unrelated', stimulus: { inline: request }, observation: { value: { assembled: [{ kind: 'text', text: 'BAD' }], chunks: [{ expected: [{ kind: 'text', text: 'BAD' }] }] } } } }, metadata_changes: {}, document_overrides: {} };
await validateUnifiedCapture(good, retiredDocument, identity);
const retiredReader = { unifiedInputs: async () => ({ value: retiredDocument }), unified: async () => ({ records: resolveUnified([{ ...identity, document: good }], '1.0.0') }) };
let panel = await unifiedPanel(catalog, retiredReader);
assert.equal(toolOnlyCell(panel).tooltip.candidates.find(candidate => candidate.key === 'dynamo').block.verdict, 'MATCH');
delete good.changes.id;
retiredDocument.cases.id.historical_ids = ['UNIFIED.1.a'];
good.changes.retired.case_key = 'UNIFIED.1.a';
good.changes.retired.stimulus.inline = { ...request, input: 'old', chunks: [{ delta_text: 'old' }] };
await validateUnifiedCapture(good, retiredDocument, identity);
panel = await unifiedPanel(catalog, retiredReader);
assert.match(toolOnlyCell(panel).tooltip.candidates.find(candidate => candidate.key === 'dynamo').block.unavailable, /mismatch \(input, chunks\)/);
assert.equal(toolOnlyCell(panel).cmp.dynamo.err, 0);
assert.equal(toolOnlyCell(panel).cmp.dynamo.na, 1);

catalog.sources.candidates.push(
  { mode: 'unified', implementation: 'dynamo_v2', version: '0.9.0', source: 'older.yaml', native_unified_families: ['qwen3'] },
  { mode: 'unified', implementation: 'vllm_python', version: '0.26.0', source: 'peer.yaml' },
);
const controls = tabCandidates(catalog, catalog.metadata.tabs.find((tab) => tab.id === 'tab-unified'));
const peerKey = controls.find(({ source }) => source.implementation === 'vllm_python').control.key;
const olderKey = controls.find(({ source }) => source.implementation === 'dynamo_v2' && source.version === '0.9.0').control.key;
const loaded = [];
const selectionReader = {
  unifiedInputs: async () => ({ value: document }),
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
  return { panel: result, cell: toolOnlyCell(result) };
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
const emptyPanel = await unifiedPanel(catalog, selectionReader, { selected: [], reference: null });
assert.deepEqual(loaded, []);
assert.deepEqual(emptyPanel.rows, []);
await assert.rejects(selectedCell([peerKey], 'dynamo'), /reference is not loaded/);
await assert.rejects(selectedCell([peerKey], 'unknown'), /unknown Unified reference/);
console.log('Unified projected mapping values, recorded errors, and stimulus admission passed');

const unionCatalog = structuredClone(catalog);
unionCatalog.sources.unified = ['qwen3', 'glm47', 'kimi_k2'].map(family => ({ family, kind: 'inputs-and-golden', path: `${family}/inputs.yaml` }));
unionCatalog.sources.candidates = [
  { mode: 'unified', implementation: 'dynamo_v1', version: '1.0.0', capture_families: ['glm47', 'qwen3'] },
  { mode: 'unified', implementation: 'dynamo_v2', version: '1.0.0', capture_families: ['glm47', 'kimi_k2'], native_unified_families: ['glm47', 'kimi_k2'] },
];
const unionReads = [];
const unionReader = {
  unifiedInputs: async family => ({ value: { ...structuredClone(document), input_document: { family }, golden_document: { family } } }),
  unified: async (family, implementation) => { unionReads.push(`${family}:${implementation}`); return { records: new Map() }; },
};
const unionPanel = await unifiedPanel(unionCatalog, unionReader, { selected: ['dynamo', 'dynamo_v1@1.0.0'], reference: null });
assert.deepEqual(unionPanel.rows.map(row => row.family), ['glm47', 'kimi_k2', 'qwen3']);
assert.equal(toolOnlyCell(unionPanel, 'qwen3').cmp.dynamo.na, 1);
assert.equal(toolOnlyCell(unionPanel, 'kimi_k2').cmp['dynamo_v1@1.0.0'].na, 1);
assert.equal(selectedCaptureOutcome(toolOnlyCell(unionPanel, 'glm47'), 'dynamo'), 'missing');
assert.equal(selectedCaptureOutcome(toolOnlyCell(unionPanel, 'qwen3'), 'dynamo'), 'unsupported');
assert.deepEqual(unionReads.sort(), ['glm47:dynamo_v1', 'glm47:dynamo_v2', 'kimi_k2:dynamo_v2', 'qwen3:dynamo_v1']);
console.log('A/B plus B/C renders the union once with distinct unsupported and missing-capture segments');

const scopedCatalog = structuredClone(unionCatalog);
scopedCatalog.metadata.unified_presentation.scenario_families.tool_only = ['glm47', 'kimi_k2'];
const scopedReader = { ...unionReader, unifiedInputs: async family => {
  const result = await unionReader.unifiedInputs(family);
  if (family === 'qwen3') result.value.cases = {};
  return result;
} };
const scopedPanel = await unifiedPanel(scopedCatalog, scopedReader, { selected: ['dynamo', 'dynamo_v1@1.0.0'], reference: null });
const scopedCell = toolOnlyCell(scopedPanel, 'qwen3');
assert.equal(selectedCaptureOutcome(scopedCell, 'dynamo_v1@1.0.0'), 'unsupported');
assert.deepEqual(scopedCell.tooltip.candidates.find(({ key }) => key === 'dynamo_v1@1.0.0').block, {
  unavailable: 'Not applicable to qwen3', unavailable_code: 'scenario_not_applicable',
});

const inventoryCatalog = structuredClone(scopedCatalog);
inventoryCatalog.sources.candidates.find(item => item.implementation === 'dynamo_v2').capture_families.push('qwen3');
inventoryCatalog.sources.candidates.find(item => item.implementation === 'dynamo_v2').native_unified_families.push('qwen3');
for (let index = 0; index < 826; index++) {
  inventoryCatalog.sources.candidates.push({ mode: 'unified', implementation: 'dynamo_v1', version: `2.0.${index}`, capture_families: ['qwen3', 'glm47'] });
}
for (const implementation of ['sglang_python', 'vllm_python']) {
  inventoryCatalog.sources.candidates.push({ mode: 'unified', implementation, version: '1.0.0', capture_families: ['qwen3', 'glm47'] });
}
const inventoryDescriptors = tabCandidates(inventoryCatalog, { id: 'tab-unified', suite: 'unified', mode: 'unified' });
for (const options of [
  { selected: ['dynamo'] },
  { selected: ['dynamo_v1@1.0.0'], reference: 'dynamo_v1@1.0.0' },
  { selected: ['dynamo_v1@1.0.0'], reference: null },
  { selected: ['sglang_python@1.0.0'], reference: 'golden' },
  { selected: ['dynamo', 'sglang_python@1.0.0', 'vllm_python@1.0.0'], reference: null },
  {},
]) {
  const panel = await unifiedPanel(inventoryCatalog, scopedReader, options);
  const cell = toolOnlyCell(panel, 'qwen3');
  const expected = inventoryDescriptors.filter(item => item.control.key === 'golden'
    || !options.selected || options.selected.includes(item.control.key)).map(item => item.control.key);
  assert.deepEqual(Object.keys(cell.cmp), expected);
  assert.deepEqual(cell.tooltip.candidates.map(item => item.key), expected);
  assert.deepEqual(panel.candidates, inventoryDescriptors.map(item => item.control), 'selectors retain the full registered inventory');
  assert.equal(cell.status, 'na');
  for (const key of expected) {
    assert.equal(selectedCaptureOutcome(cell, key), 'unsupported');
    assert.deepEqual(cell.tooltip.candidates.find(item => item.key === key).block, {
      unavailable: 'Not applicable to qwen3', unavailable_code: 'scenario_not_applicable',
    });
  }
}
console.log('NA comparisons and popup candidates scale with selection across the full release inventory');

// Historical observations cannot introduce families outside current native support.
const nativeOnlyCatalog = structuredClone(unionCatalog);
const v2 = nativeOnlyCatalog.sources.candidates.find(item => item.implementation === 'dynamo_v2');
v2.capture_families.push('qwen3', 'gpt_oss');
nativeOnlyCatalog.sources.unified.push({ family: 'gpt_oss', kind: 'inputs-and-golden', path: 'gpt_oss/inputs.yaml' });
assert.equal(sourceCapturesFamily(v2, 'gpt_oss'), false);
assert.equal(sourceCapturesFamily(v2, 'gpt_oss', { nativeOnly: false }), true, 'archive parity retains split historical records');
unionReads.length = 0;
const nativeOnly = await unifiedPanel(nativeOnlyCatalog, unionReader, { selected: ['dynamo'], reference: null });
assert.deepEqual(nativeOnly.rows.map(row => row.family), ['glm47', 'kimi_k2']);
assert.deepEqual(unionReads.sort(), ['glm47:dynamo_v2', 'kimi_k2:dynamo_v2']);
v2.native_unified_families = [];
assert.deepEqual((await unifiedPanel(nativeOnlyCatalog, unionReader, { selected: ['dynamo'], reference: null })).rows, [], 'current native eligibility bounds Dynamo v2 rows');
delete v2.native_unified_families;
await assert.rejects(unifiedPanel(nativeOnlyCatalog, unionReader, { selected: ['dynamo'], reference: null }), /version-specific UnifiedParser eligibility/);
console.log('Dynamo v2 rows and reads exclude split captures and reject missing route eligibility');

// Main retains historical Combined observations beside current native results.
// Missing historical route proof must not replace recorded errors with "No parser".
for (const historicalNative of [[], ['qwen3']]) for (const observedError of [true, false]) {
  const historicalCatalog = structuredClone(catalog);
  const historical = historicalCatalog.sources.candidates.find(item => item.implementation === 'dynamo_v2' && item.version === '0.9.0');
  historical.native_unified_families = historicalNative;
  loaded.length = 0;
  const historicalReader = { ...selectionReader, async unified(...args) {
    const result = await selectionReader.unified(...args);
    if (!observedError) for (const record of result.records.values()) record.value.observation = { value: { assembled: [], chunks: [] } };
    return result;
  } };
  const historicalPanel = await unifiedPanel(historicalCatalog, historicalReader, { selected: [olderKey], reference: null });
  assert.deepEqual(historicalPanel.rows.map(row => row.family), ['qwen3'], 'recorded historical results use current family eligibility without claiming an old native route');
  assert.equal(selectedCaptureOutcome(toolOnlyCell(historicalPanel), olderKey), observedError ? 'error' : 'pass');
  assert.deepEqual(loaded, ['dynamo_v2@0.9.0']);
  assert.deepEqual(historical.native_unified_families, historicalNative, 'display cannot rewrite historical native support');
  assert.match(historicalPanel.candidates.find(item => item.key === olderKey).label, /recorded history/);
}
console.log('historical Dynamo results remain visible independently of native-route proof');

const aliasName = `${input.display_id}.retired`;
const aliasInputs = structuredClone(document);
aliasInputs.cases.id.historical_ids = [aliasName];
aliasInputs.cases.retired = { ...structuredClone(input), lifecycle: 'retired', display_id: null, historical_ids: [], request: null, golden: null };
const aliasReader = (exactError, reverse, mismatch, extraRetired = false) => ({
  unifiedInputs: async () => ({ value: aliasInputs }),
  unified: async () => {
    const oldRequest = structuredClone(request);
    if (mismatch === 'chunks') oldRequest.chunks.push({ delta_text: 'finish' });
    if (mismatch === 'input') oldRequest.input = 'older input';
    const capture = { provenance: { status: 'legacy', captured_with: { dynamo_v2: '1.0.0' } }, document: {},
      changes: {
        id: { case_key: input.display_id, stimulus: { ref: 'current' }, observation: exactError ? { error: { code: 'panic', detail: 'recorded error' } } : { value: { assembled: [], chunks: [] } } },
        retired: { case_key: aliasName, stimulus: { inline: oldRequest }, observation: { value: { assembled: [{ kind: 'text', text: 'old result' }], chunks: [] } } },
      }, metadata_changes: {}, document_overrides: {} };
    await validateUnifiedCapture(capture, aliasInputs, identity);
    const entries = [...resolveUnified([{ ...identity, document: capture }], '1.0.0')];
    if (extraRetired) {
      aliasInputs.cases.retired2 = structuredClone(aliasInputs.cases.retired);
      const other = structuredClone(entries.find(([id]) => id === 'retired')[1]);
      other.value.observation = { value: { assembled: [], chunks: [] } };
      entries.push(['retired2', other]);
    }
    return { records: new Map(reverse ? entries.reverse() : entries) };
  },
});
for (const exactError of [false, true]) for (const reverse of [false, true]) for (const mismatch of ['chunks', 'input']) for (const extraRetired of [false, true]) {
  const result = await unifiedPanel(catalog, aliasReader(exactError, reverse, mismatch, extraRetired), { selected: ['dynamo'], reference: null });
  assert.equal(selectedCaptureOutcome(toolOnlyCell(result), 'dynamo'), exactError ? 'error' : 'pass', 'retired request mismatches cannot replace or conflict with an exact capture binding');
}
for (const reverse of [false, true]) {
  await assert.rejects(unifiedPanel(catalog, aliasReader(false, reverse, null), { selected: ['dynamo'], reference: null }),
    /conflicting historical aliases/, 'contradictory exact request bindings must still fail');
}
console.log('historical alias resolution preserves exact request bindings and rejects real conflicts');

} finally {
  for (const key of Object.getOwnPropertyNames(globalThis)) {
    if (!Object.hasOwn(savedGlobals, key) && Object.getOwnPropertyDescriptor(globalThis, key).configurable) delete globalThis[key];
  }
  for (const [key, descriptor] of Object.entries(savedGlobals)) {
    if (descriptor.configurable) Object.defineProperty(globalThis, key, descriptor);
  }
}

});

await test('links contracts', async () => {
const savedGlobals = Object.getOwnPropertyDescriptors(globalThis);
try {
const { default: assert } = await import('node:assert/strict');
const { access, readFile } = await import('node:fs/promises');
const { createSourceLink } = await import('./presentation.js');


const root = new URL('../../', import.meta.url);
const local = createSourceLink('http://host.test/nested/');
assert.equal(local('parsers/v1/src/tool_calling/config.rs'), 'http://host.test/nested/parsers/v1/src/tool_calling/config.rs');
const catalog = JSON.parse(await readFile(new URL('conformance/viewer-catalog.json', root)));
for (const target of Object.values(catalog.metadata.documentation)) await access(new URL(target, root));
for (const [target, current] of [
  ['conformance/toolcalling/fixtures-v1/', 'conformance/fixtures/toolcalling/fixtures-batch-v1/inputs.tar.gz'],
  ['conformance/toolcalling/fixtures-stream-v1/', 'conformance/fixtures/toolcalling/fixtures-stream-v1/inputs.tar.gz'],
  ['conformance/reasoning/fixtures/', 'conformance/fixtures/reasoning/fixtures-v1/inputs.tar.gz'],
]) {
  assert.equal(local(target), `http://host.test/nested/${current}`);
  assert.equal(local(`${current}/`), local(target));
  await access(new URL(current, root));
}
const publication = { schema: 1, revision: 'a'.repeat(40), catalog_revision: catalog.revision };
const web = createSourceLink('http://host.test/nested/', publication, catalog.revision);
assert.equal(web('parsers/v1/src/tool_calling/config.rs'), `https://github.com/ai-dynamo/frontend-crates/blob/${publication.revision}/parsers/v1/src/tool_calling/config.rs`);
assert.ok(web('parsers/v1/src/tool_calling/').includes(`/tree/${publication.revision}/`));
assert.equal(web('conformance/toolcalling/fixtures-v1/'), `https://github.com/ai-dynamo/frontend-crates/blob/${publication.revision}/conformance/fixtures/toolcalling/fixtures-batch-v1/inputs.tar.gz`);
const cachedPublication = { ...publication, generated_revision: 'b'.repeat(40) };
const cachedWeb = createSourceLink('http://host.test/nested/', cachedPublication, catalog.revision);
assert.ok(cachedWeb('parsers/v1/src/tool_calling/config.rs').includes(`/${cachedPublication.revision}/`));
for (const revision of ['main', 'short', 'a'.repeat(39), null]) assert.throws(() => createSourceLink('http://host.test/', { ...publication, revision }, catalog.revision), /Publication/);
for (const revision of ['main', 'short', 'b'.repeat(39), null]) assert.throws(() => createSourceLink('http://host.test/', { ...publication, generated_revision: revision }, catalog.revision), /Publication/);
assert.throws(() => createSourceLink('http://host.test/', publication, 'other'), /Publication/);
for (const target of ['../outside', '/etc/passwd', 'javascript:invalid', 'a\\b', 'a//b']) assert.throws(() => local(target), /Invalid source/);
console.log('Source targets resolve under root/nested hosting and immutable publication identity');

} finally {
  for (const key of Object.getOwnPropertyNames(globalThis)) {
    if (!Object.hasOwn(savedGlobals, key) && Object.getOwnPropertyDescriptor(globalThis, key).configurable) delete globalThis[key];
  }
  for (const [key, descriptor] of Object.entries(savedGlobals)) {
    if (descriptor.configurable) Object.defineProperty(globalThis, key, descriptor);
  }
}

});
