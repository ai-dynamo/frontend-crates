// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { test } from 'node:test';

await test('latest and history selectors accept only version numbers', async () => {
  const { default: assert } = await import('node:assert/strict');
  const { tabCandidates } = await import('./model.js');
  for (const mode of ['latest', 'history']) {
    const catalog = {
      capture_generation: { mode },
      metadata: {
        selection_policy: { unified: { reference: 'dynamo_v2', implementation_order: ['dynamo_v2'], historical_control_aliases: {}, golden_bucket: 'B' } },
        implementations: [], unified_capture_policy: {},
      },
      sources: { candidates: [{ implementation: 'dynamo_v2', mode: 'unified', version: '0.9.7' }] },
    };
    const tab = { mode: 'unified' };
    assert.equal(tabCandidates(catalog, tab)[0].control.version, '0.9.7');
    for (const suffix of ['+source.' + 'a'.repeat(64), '+sha.abc', '+local']) {
      catalog.sources.candidates[0].version = '0.9.7' + suffix;
      assert.throws(() => tabCandidates(catalog, tab), /invalid capture version/);
    }
  }
});

await test('columns cannot omit, duplicate, replace, or combine authored cases', async () => {
  const { default: assert }=await import('node:assert/strict');
  const { validateCaseColumns }=await import('./model.js');
  const id='UNIFIED.1-tool_only',events=[{kind:'tool_call',name:'a',arguments:{}},{kind:'tool_call',name:'b',arguments:{}}];
  const authored=new Map([['sample',new Map([['one',{case_id:id,request:{input:'input',init:{},finish_reason:'stop',tools:[],chunks:[]},golden:events}]])]]);
  const tab={columns:[{sub:'one',case_id:id}],rows:[{family:'sample',cells:{one:{kind:'cell',sub:'one',family:'sample',case_id:id,status:'ok',
    tooltip:{input:{text:'input',chunks:[]},init:{starting_state:'None',tool_output_mode:'Native',named_tool:null},finish_reason:'stop',candidates:[{key:'golden',block:{events}}]}}}}]};
  validateCaseColumns(tab,authored);
  for(const mutate of [
    t=>{t.columns=[];},t=>{t.columns.push({...t.columns[0]});},
    t=>{t.columns[0].sub='extra';},t=>{t.rows[0].cells.one.case_id='different';},
    t=>{t.rows[0].cells.one.variants=[];},t=>{t.rows[0].cells.one.tooltip.variants=[];},
    t=>{t.columns[0].schemas=[{cases:[]}];},t=>{t.rows[0].cells.one.tooltip.input.text='other input';},
    t=>{t.rows[0].cells.one.tooltip.candidates[0].block.events=[];},
    t=>{t.rows[0].cells.one.sub='other';},t=>{t.rows[0].cells.one.family='other';},
    t=>{t.rows[0].cells.one.tooltip.candidates.push({key:'golden',block:{events:[]}});},
    t=>{t.columns[0].schemas=[{families:['sample'],tools:[{name:'wrong',parameters:{}}]}];},
    t=>{t.rows=[];},t=>{t.rows[0].cells.one.status='na';},
  ]){const bad=structuredClone(tab);mutate(bad);assert.throws(()=>validateCaseColumns(bad,authored));}
});

await test('numeric contracts', async () => {
const savedGlobals = Object.getOwnPropertyDescriptors(globalThis);
try {
const { equivalenceEntries, canonicalUnifiedEvents } = await import('./model.js');
const { default: assert } = await import('node:assert/strict');
const { canonicalArguments, sameValue } = await import('./values.js');
const { classifyUnified } = await import('./model.js');
const { assembleStream } = await import('./model.js');


const comparisonEntries = blocks => equivalenceEntries(Object.fromEntries(Object.entries(blocks).map(([key, block]) => [key, {value: canonicalUnifiedEvents((block.calls || []).map(call => ({kind: 'tool_call', ...call})))}])));
const markers = {common: [], families: {}};
const event = (argumentsValue) => ({kind: 'tool_call', name: 'f', arguments: argumentsValue});
const block = (argumentsValue) => ({calls: [{name: 'f', arguments: argumentsValue}], normal_text: ''});
const classify = (a,b) => classifyUnified('qwen3', [event(a)], [event(b)], markers);
const pairs = [
  ['9007199254740992.5', '9007199254740992.0'], ['9007199254740993.1', '9007199254740994.0'],
  ['0.10000000000000000001', '0.1'], ['1e-400','0'], ['-9007199254740992.5','-9007199254740992.0'],
];
for (const [raw, rounded] of pairs) {
  const expected = `{"value":${raw}}`, actual = `{"value":${rounded}}`;
  const stream = assembleStream([[{...event(actual), complete: true}]], null, true);
  assert.equal(classifyUnified('qwen3',[event(expected)],stream,markers),'ARG_MISMATCH');
  const entries = comparisonEntries({expected: block(expected), actual: block(actual)},[]);
  assert.notEqual(entries.expected.sig,entries.actual.sig);
  assert.equal(classify(expected,expected),'MATCH');
  assert.equal(classify(expected,{value:Number(raw)}),'ARG_MISMATCH');
}
for (const [a,b] of [['42.0','4.2e1'], ['4200e-2','42'], ['-0.0e-400','0']]) {
  assert.equal(classify(`{"value":${a}}`,`{"value":${b}}`),'MATCH');
  const entries = comparisonEntries({a:block(`{"value":${a}}`),b:block(`{"value":${b}}`)},[]);
  assert.equal(entries.a.sig,entries.b.sig);
}
assert.equal(classify('{"value":"42"}','{"value":42}'),'ARG_MISMATCH');
assert.equal(classify('{"value":42.5}',{value:42.5}),'MATCH');
for (const token of ['NaN','Infinity','-Infinity']) {
  assert.deepEqual(canonicalArguments(`{"value":${token}}`),['invalid_json',`{"value":${token}}`]);
  assert.equal(classify('{"value":0}',`{"value":${token}}`),'ARG_MISMATCH');
  assert.equal(sameValue(canonicalArguments({value:Number(token)}),canonicalArguments('{"value":0}')),false);
}
for (const exponent of ['1'+'0'.repeat(40),'9'.repeat(5000)]) for (const sign of ['', '-']) {
  const adjusted = String(BigInt(exponent)+(sign ? 1n : -1n));
  const a = `{"value":1e${sign}${exponent}}`, b = `{"value":10e${sign}${adjusted}}`;
  assert.equal(classify(a,b),'MATCH');
  assert.equal(classify(a,`{"value":1e${sign}${adjusted}}`),'ARG_MISMATCH');
}
const nullFallback=assembleStream([[{...event('null'),complete:true},{kind:'tool_call',name:'bad',arguments:'{',complete:true}]],[event(null),{kind:'tool_call',name:'bad',arguments:{}}],true);
assert.deepEqual(nullFallback[1].arguments,{});
const missing = {calls:[{name:'f'}]}, empty = {calls:[{name:'f',arguments:{}}]}, explicitNull=block('null');
const identities=comparisonEntries({missing,empty,explicitNull},[]);
assert.equal(new Set(Object.values(identities).map(row=>row.sig)).size,3);
for (const preserve of [false,true]) for (const misaligned of [false,true]) for (const encoded of [false,true]) {
  const exact='{"value":9007199254740992.5}', malformed='{"broken":';
  const recorded=[{kind:'tool_call',name:'number',arguments:encoded?exact:{value:9007199254740992.0}},
    {kind:'tool_call',name:'bad',arguments:encoded?'{}':{}}];
  if (misaligned) recorded.reverse();
  const result=assembleStream([[{kind:'tool_call',name:'number',arguments:exact,complete:true},{kind:'tool_call',name:'bad',arguments:malformed,complete:true}]],recorded,preserve);
  const aligned=!misaligned;
  if (preserve) assert.equal(result[0].arguments,exact);
  else assert.equal(result[0].arguments.value,9007199254740992.0);
  assert.deepEqual(result[1].arguments,aligned?(preserve && encoded?'{}':{}):malformed);
}
// Recorded assembly must expose data loss even when the stream was valid JSON.
for (const preserve of [false, true]) {
  for (const [raw, recorded] of [
    ['{"city":"Paris","value":1e400}', {}],
    ['{"city":"Paris","value":-1E+400}', {}],
    ['{"nested":{"city":"Paris","value":1e400}}', {nested: {}}],
    ['[1,2]', [1n]],
    ['{"city":"Paris"}', {city: 'Rome'}],
    ['{"value":1.5}', {value: 2.5}],
    ['{"city":"Paris"}', 'wrong'],
  ]) {
    const result = assembleStream([[{...event(raw), complete: true}]], [event(recorded)], preserve);
    assert.deepEqual(result[0].arguments, recorded);
  }
}
for (const [raw, recorded] of [['9007199254740993', '9007199254740992'], ['1e-400', '0'], ['1e400', '1e401']]) {
  assert.equal(assembleStream([[{...event(raw), complete: true}]], [event(recorded)], true)[0].arguments, recorded);
}
for (const [raw, recorded] of [[{city: 'Paris'}, {}], [[1n, 2n], [1n]], [9007199254740993n, 9007199254740992n]]) {
  if (typeof raw === 'bigint') {
    assert.equal(assembleStream([[{...event(String(raw)), complete: true}]], [event(recorded)], true)[0].arguments, recorded);
  } else assert.deepEqual(assembleStream([[{...event(raw), complete: true}]], [event(recorded)], true)[0].arguments, recorded);
}
assert.deepEqual(assembleStream([[{...event('{"value":1e400}'), complete: true}]], [event({value: Infinity})], true)[0].arguments, {value: Infinity});
assert.deepEqual(assembleStream([[{...event('  '), complete: true}]], [event({city: 'Paris'})], true)[0].arguments, {city: 'Paris'});
for (const preserve of [false, true]) {
  const result = assembleStream([[{...event('{"value":1.5}'),complete:true},{kind:'tool_call',name:'bad',arguments:'{bad',complete:true}]], [event({value:2.5}),{kind:'tool_call',name:'bad',arguments:{}}],preserve);
  assert.deepEqual(result.map(call=>call.arguments),[{value:2.5},{}]);
}
console.log('numeric report contracts passed');

} finally {
  for (const key of Object.getOwnPropertyNames(globalThis)) {
    if (!Object.hasOwn(savedGlobals, key) && Object.getOwnPropertyDescriptor(globalThis, key).configurable) delete globalThis[key];
  }
  for (const [key, descriptor] of Object.entries(savedGlobals)) {
    if (descriptor.configurable) Object.defineProperty(globalThis, key, descriptor);
  }
}

});

await test('status contracts', async () => {
const savedGlobals = Object.getOwnPropertyDescriptors(globalThis);
try {
const { default: assert } = await import('node:assert/strict');
const { spawnSync } = await import('node:child_process');
const { fileURLToPath } = await import('node:url');
const { buildStatus, cellState, defaultReference, parseStatusArguments, printStatus, runStatus, selectRows, selectTabs, statusTotals, validateUnifiedKnownDivergences } = await import('./qualification.js');
const { selectedCellOutcome, selectedCaptureOutcome } = await import('./model.js');


const reference = { key: 'reference', label: 'Dynamo sample', default_bucket: 'A' };
const comparison = { reference: { sig: 0, na: 0, err: 0, leak: 0 }, golden: { sig: 0, na: 0, err: 0, leak: 0 } };
const cell = { kind: 'cell', status: 'ok', cmp: comparison };
assert.deepEqual(cellState(null, reference), ['empty', 'no cell was emitted for this model/case pair']);
assert.deepEqual(cellState({ ...cell, status: 'na', tooltip: { na_note: 'not supported' } }, reference), ['na', 'not supported']);
assert.equal(cellState({ ...cell, cmp: {} }, reference)[1], "the default Reference 'Dynamo sample' has no comparison entry");
assert.equal(cellState({ ...cell, cmp: { reference: { ...comparison.reference, na: 1, err: 1 } } }, reference)[0], 'empty');
assert.equal(cellState({ ...cell, cmp: { reference: { ...comparison.reference, err: 1 } } }, reference)[0], 'red');
assert.equal(cellState({ ...cell, red_on_diff: true, cmp: { reference: comparison.reference } }, reference)[1], 'Unified cell has no GOLDEN comparison entry');
assert.equal(cellState({ ...cell, red_on_diff: true, cmp: { ...comparison, golden: { ...comparison.golden, sig: 1 } } }, reference)[1], 'the default Reference differs from GOLDEN');
assert.deepEqual(cellState({ ...cell, red_on_diff: true, cmp: { ...comparison, reference: { ...comparison.reference, err: 1 } } }, reference), ['red', 'the default Reference returned an error']);
assert.deepEqual(cellState({ ...cell, red_on_diff: true, cmp: { ...comparison, reference: { ...comparison.reference, leak: 1 } } }, reference), ['red', 'the default Reference leaked structured markup']);
assert.equal(cellState({ ...cell, cmp: { ...comparison, reference: { ...comparison.reference, leak: 1 } } }, reference)[1], 'the default Reference leaked structured markup');
assert.equal(cellState({ ...cell, red_on_diff: true, cmp: { reference: { ...comparison.reference, sig: ['golden', 'reference'] }, golden: { ...comparison.golden, sig: ['golden', 'reference'] } } }, reference)[0], 'green');
const clean = { sig: 0, na: 0, err: 0, leak: 0 };
const choices = { base: clean, equal: clean, different: { ...clean, sig: 1 }, missing: { ...clean, na: 1 }, failed: { ...clean, err: 1 } };
const outcome = (overrides = {}) => selectedCellOutcome({ cmp: choices, base: 'base', shown: ['equal'], ...overrides });
assert.equal(outcome().marker, '=');
assert.equal(outcome({ shown: ['base', 'equal', 'missing'] }).marker, '?');
assert.equal(outcome({ shown: ['different', 'missing', 'failed'] }).marker, '✗');
assert.equal(outcome({ shown: ['failed', 'missing'] }).marker, '✗');
assert.equal(outcome({ base: null }).className, 'cmp-nobase');
assert.equal(outcome({ base: 'unknown' }).referenceState, 'missing');
assert.equal(outcome({ base: 'missing' }).referenceState, 'unavailable');
assert.equal(outcome({ base: 'failed', shown: [] }).category, 'na');
assert.equal(outcome({ base: 'failed', shown: ['equal'] }).marker, '1Δ');
assert.equal(outcome({ base: 'failed', shown: [], redOnDiff: true }).marker, '✗');
assert.equal(outcome({ family: 'unsupported', parserNi: { base: { label: 'Selected parser', families: ['supported'] } } }).marker, 'x');
assert.equal(outcome({ cmp: { ...choices, golden: clean }, base: 'golden', redOnDiff: true, shown: ['different'] }).category, 'problem');
assert.equal(outcome({ cmp: choices, base: 'golden', redOnDiff: true, shown: ['different'] }).category, 'problem');
assert.equal(outcome({ cmp: { ...choices, golden: clean }, redOnDiff: true, shown: ['different'] }).marker, '1Δ');
assert.equal(outcome({ cmp: { ...choices, base: choices.different, golden: clean }, redOnDiff: true }).marker, '=');
assert.equal(outcome({ cmp: { ...choices, base: { ...clean, leak: 1 }, golden: clean }, redOnDiff: true }).marker, '↯=');
assert.equal(outcome({ cmp: { ...choices, base: { ...clean, leak: 1 } } }).category, 'problem');
const goldenComparisons = { golden: clean, good: clean, bad: { ...clean, err: 1 }, leaky: { ...clean, leak: 1 } };
const goldenOutcome = (shown) => selectedCellOutcome({ cmp: goldenComparisons, base: 'golden', shown, redOnDiff: true });
assert.equal(goldenOutcome(['good', 'bad']).category, 'problem');
assert.equal(goldenOutcome(['bad']).referenceState, 'error');
assert.equal(goldenOutcome(['leaky']).category, 'problem');
assert.equal(goldenOutcome(['leaky']).leakedComparisons, 1);
assert.equal(selectedCaptureOutcome({ kind: 'cell', cmp: { ...comparison, selected: { ...clean, sig: 1 } } }, 'selected'), 'diverge');
assert.equal(selectedCaptureOutcome({ kind: 'cell', cmp: { ...comparison, selected: clean } }, 'selected'), 'pass');
assert.equal(selectedCaptureOutcome({ kind: 'cell', cmp: { ...comparison, selected: { ...clean, err: 1 } } }, 'selected'), 'error');
assert.equal(selectedCaptureOutcome({ kind: 'cell', cmp: { ...comparison, selected: { ...clean, leak: 1 } } }, 'selected'), 'leak');
for (const [code, expected] of [
  ['unsupported_capture_mode', 'unsupported'], ['unsupported_request_mode', 'unsupported'],
  ['unsupported_family', 'unsupported'], ['scenario_not_applicable', 'unsupported'],
  ['duplicate_scenario', 'unsupported'], ['missing_capture', 'missing'],
  ['source_unavailable', 'blocked'], ['original_request_not_retained', 'missing'],
  ['unknown_capture_code', 'missing'],
]) {
  for (const nested of [false, true]) for (const absent of [false, true]) {
    const block = { unavailable: 'This published crate does not expose a JailedStream parser',
      ...(nested ? { capture_observation: { unavailable: { code } } } : { unavailable_code: code }) };
    const capture = { kind: 'cell', cmp: { golden: clean, ...(absent ? {} : { selected: { ...clean, na: 1 } }) },
      tooltip: { candidates: [{ key: 'selected', block }] } };
    assert.equal(selectedCaptureOutcome(capture, 'selected'), expected, `${code}: nested=${nested}, absent=${absent}`);
  }
}

const tab = { id: 'tab-unified', kind: 'unified', label: 'Unified v2 (reasoning + tools)', candidates: [reference],
  columns: [{ sub: 'present', label: '1' }, { sub: 'missing', label: '2' }, { sub: 'absent', label: '3' }],
  rows: [{ family: null, section: 'heading' }, { family: 'qwen3_coder', model_label: 'Qwen 3 Coder / Nemotron V3', cells: { present: cell, absent: { ...cell, status: 'na' } } }] };
assert.deepEqual(selectTabs({ tabs: [tab] }, ['tab-unified', 'Unified v2 reasoning tools', 'unified']), [tab]);
assert.deepEqual(selectRows(tab, ['qwen3-coder', 'Qwen 3 Coder Nemotron V3']), [tab.rows[1]]);
assert.throws(() => selectTabs({ tabs: [tab] }, ['unknown']), /unknown tab/);
assert.throws(() => selectRows(tab, ['unknown']), /unknown model/);
assert.throws(() => defaultReference({ ...tab, candidates: [] }), /found 0/);
assert.throws(() => defaultReference({ ...tab, candidates: [reference, reference] }), /found 2/);
const status = buildStatus({ meta: { title: 'sample' }, tabs: [tab] }, [tab], [], 'index.html');
assert.deepEqual(statusTotals(status), { reports: 1, empty: 1, red: 0 });
assert.deepEqual(status.reports[0].issues, [{ state: 'empty', case: '2', scenario: 'missing', reason: 'no cell was emitted for this model/case pair' }]);
assert.equal(status.reports[0].na, 1);
const leaf = (sub, red) => ({ sub, kind: 'cell', status: 'ok', red_on_diff: true, cmp: { reference: { ...clean, sig: red ? 1 : 0 }, golden: clean } });
const folded = { ...tab, rows: [{ family: 'qwen3', cells: { folded: { variants: [leaf('nullable', true), leaf('union', false)] } } }] };
const aliases = { canonical: { nullable: '7-nullable', union: '8-union' }, owners: { '7-nullable': ['nullable'], '8-union': ['union'] }, families: {} };
const registry = { qwen3: { 'UNIFIED.7-nullable': { golden: { actual: 'recorded' } } } };
validateUnifiedKnownDivergences({ tabs: [folded] }, registry, ['qwen3'], aliases);
const unavailableSplit = { ...leaf('v1_split_parallel_tool_calls', false), cmp: {
  reference: { ...clean, na: 1 }, golden: clean,
}, tooltip: { candidates: [{ key: reference.key, block: {
  unavailable: 'Dynamo v2 does not run the Dynamo v1 split parser for deepseek_v3; select a Dynamo v1 capture to inspect this case.',
} }] } };
const expanded = { ...folded, rows: [
  ...folded.rows,
  { family: 'deepseek_v3', cells: { folded: { variants: [leaf('nullable', false), leaf('union', false), unavailableSplit] } } },
] };
validateUnifiedKnownDivergences({ tabs: [expanded] }, registry, ['qwen3', 'deepseek_v3'], aliases, ['v1_split_parallel_tool_calls']);
assert.throws(() => validateUnifiedKnownDivergences({ tabs: [expanded] }, registry, ['qwen3', 'deepseek_v3'], aliases), /empty=.*v1_split_parallel_tool_calls/);
assert.throws(() => validateUnifiedKnownDivergences({ tabs: [folded] }, {}, ['qwen3'], aliases), /differs from known divergences/);
assert.throws(() => validateUnifiedKnownDivergences({ tabs: [folded] }, { unknown: {} }, ['qwen3'], aliases), /unknown families/);
folded.rows[0].cells.folded.variants[1] = leaf('union', true);
assert.throws(() => validateUnifiedKnownDivergences({ tabs: [folded] }, registry, ['qwen3'], aliases), /actual_red/);
folded.rows[0].cells.folded.variants[1] = leaf('union', false);
folded.rows[0].cells.folded.variants[0] = leaf('nullable', false);
assert.throws(() => validateUnifiedKnownDivergences({ tabs: [folded] }, registry, ['qwen3'], aliases), /expected_red/);
const output = [];
printStatus(status, true, (line) => output.push(line));
assert.deepEqual(output, ['conformance status: 1 model/tab pairs, 1 empty, 0 red']);
assert.deepEqual(parseStatusArguments(['--model', 'qwen3', '--model', 'glm47', '--tab', 'unified']).models, ['qwen3', 'glm47']);
assert.equal(parseStatusArguments(['--html', 'legacy-path.html']).reportPath, 'legacy-path.html');
assert.throws(() => parseStatusArguments(['--model']), /requires a value/);
assert.throws(() => parseStatusArguments(['--tab', '--summary-only']), /requires a value/);
assert.throws(() => parseStatusArguments(['--unknown']), /invalid argument/);
const cli = fileURLToPath(new URL('./qualification.js', import.meta.url));
assert.equal(spawnSync(process.execPath, [cli, 'status', '--help']).status, 0);
assert.equal(spawnSync(process.execPath, [cli, 'status', '--unknown']).status, 2);
const errors = [];
assert.equal(await runStatus({ ...parseStatusArguments([]), root: '/missing/conformance-status-test-root' }, () => {}, (line) => errors.push(line)), 2);
assert.match(errors[0], /^ERROR: /);
console.log('status schemas, issue reasons, aliases, missing/error precedence, CLI arguments and resource failures passed');

} finally {
  for (const key of Object.getOwnPropertyNames(globalThis)) {
    if (!Object.hasOwn(savedGlobals, key) && Object.getOwnPropertyDescriptor(globalThis, key).configurable) delete globalThis[key];
  }
  for (const [key, descriptor] of Object.entries(savedGlobals)) {
    if (descriptor.configurable) Object.defineProperty(globalThis, key, descriptor);
  }
}

});

await test('sparse history and classification contracts', async () => {
const assert = await import('node:assert/strict');
const { classifyUnified, compareVersions, resolveUnified, sameValue, unifiedEvents } = await import('./model.js');




assert.equal(compareVersions('0.7.9', '0.7.10'), -1);
assert.equal(compareVersions('0.7.10.patch1', '0.7.10'), 1);
assert.equal(sameValue(null, 'null'), false);
assert.equal(sameValue(1, '1'), false);
const captures = [
  { version: '0.1.0', implementation: 'dynamo_v2', path: 'first.yaml', bytes: new Uint8Array([1]), document: {
    provenance: { status: 'captured', origin: { crate_version: '0.1.0', source_sha256: 'a'.repeat(64) } },
    document: { mode: 'unified' },
    changes: { case_a: { case_key: 'UNIFIED.3', stimulus: { unavailable: { code: 'original_request_not_retained' } }, observation: { value: { assembled: [{ kind: 'text', text: 'old' }] } } }, case_removed: { case_key: 'UNIFIED.3', stimulus: { unavailable: { code: 'original_request_not_retained' } }, observation: { value: { assembled: [] } } } },
    metadata_changes: { case_a: { source: 'old' } },
    document_overrides: { case_a: { parser_path: 'old/parser' } },
  } },
  { version: '0.2.0', implementation: 'dynamo_v2', path: 'second.yaml', bytes: new Uint8Array([2]), document: {
    provenance: { status: 'captured', origin: { crate_version: '0.2.0', source_sha256: 'b'.repeat(64) } },
    document: { mode: 'unified' },
    changes: { case_a: { observation: { value: { chunks: [[{ kind: 'text', text: 'new' }]] } } }, case_removed: { absent: true } },
    metadata_changes: { case_a: { source: 'new' } },
    document_overrides: { case_a: { parser_path: 'new/parser', capture_provenance: { captured_with: { dynamo_v2: '0.2.0' }, capture_origin: null } } },
  } },
  { version: '0.3.0', implementation: 'dynamo_v2', path: 'third.yaml', bytes: new Uint8Array([3]), document: {
    provenance: { status: 'captured', origin: { crate_version: '0.3.0', source_sha256: 'c'.repeat(64) } },
    document: { mode: 'unified' }, changes: {}, metadata_changes: {}, document_overrides: {},
  } },
];
const resolved = resolveUnified(captures, '0.3.0');
assert.equal(resolved.has('case_removed'), false);
const inherited = resolved.get('case_a');
assert.deepEqual(unifiedEvents(inherited.value.observation.value), [{ kind: 'text', text: 'new' }]);
assert.equal(inherited.value.document.inherited_from, '0.2.0');
assert.equal(inherited.value.document.parser_path, 'new/parser');
assert.equal(inherited.value.document.record_metadata.source, 'new');
assert.equal(Object.hasOwn(inherited.value.document, 'capture_origin'), false);

const markers = { common: ['<control>'], families: { sample: ['<tool>'] } };
assert.equal(classifyUnified('sample', [{ kind: 'text', text: 'x' }], [{ kind: 'text', text: 'x' }], markers), 'MATCH');
assert.equal(classifyUnified('sample', [{ kind: 'reasoning', text: 'x' }], [{ kind: 'text', text: 'x' }], markers), 'MERGE');
assert.equal(classifyUnified('sample', [{ kind: 'text', text: '' }], [{ kind: 'text', text: '<tool>' }], markers), 'LEAK');
assert.equal(classifyUnified('sample', [{ kind: 'text', text: 'quoted <tool> marker' }], [{ kind: 'text', text: 'quoted ' }, { kind: 'text', text: '<tool> marker' }], markers), 'ORDER');
assert.equal(classifyUnified('sample', [{ kind: 'text', text: 'quoted <tool> marker' }], [{ kind: 'text', text: 'quoted <tool> marker <tool>' }], markers), 'LEAK');
const harmonyMarkers = { common: ['<|start|>'], families: {} };
assert.equal(classifyUnified('gpt_oss', [{ kind: 'text', text: '{"visible":"<|start|>"}Done.' }], [{ kind: 'text', text: '{"visible":"<|start|>"}' }, { kind: 'text', text: 'Done.' }], harmonyMarkers), 'ORDER');
assert.equal(classifyUnified('gpt_oss', [{ kind: 'text', text: '{"visible":"<|start|>"}Done.' }], [{ kind: 'text', text: '{"visible":"<|start|>"}Done.<|start|>' }], harmonyMarkers), 'LEAK');
assert.equal(classifyUnified('sample', [{ kind: 'tool_call', name: 'f', arguments: { x: null } }], [{ kind: 'tool_call', name: 'f', arguments: { x: 'null' } }], markers), 'ARG_MISMATCH');
assert.equal(classifyUnified('sample', [{ kind: 'reasoning', text: 'x' }, { kind: 'text', text: 'y' }], [{ kind: 'text', text: 'y' }, { kind: 'reasoning', text: 'x' }], markers), 'ORDER');
console.log('viewer model semantics passed');

});
