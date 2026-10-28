// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { test } from 'node:test';

await test('folds contracts', async () => {
const savedGlobals = Object.getOwnPropertyDescriptors(globalThis);
try {
const { default: assert } = await import('node:assert/strict');
const { aggregateCells, leafCells, foldSchemaVariants, foldNullVariants } = await import('./model.js');
const { selectedCaptureOutcome } = await import('./model.js');
const { equivalenceGroups } = await import('./model.js');


const cell = (sub, cmp, status = 'ok') => ({ kind: 'cell', sub, family: 'sample', case_id: `TOOLCALLING.batch.${sub}`, status, cmp, tooltip: { head: sub } });
const entry = (sig, fields = {}) => ({ sig, na: 0, err: 0, leak: 0, ...fields });
const children = [cell('7-4', { golden: entry(1), reference: entry(1), peer: entry(2) }), cell('7-4.anyof', { golden: entry(4), reference: entry(4), peer: entry(5) })];
const aggregate = aggregateCells(children, '7-4', 'nullable');
assert.deepEqual(equivalenceGroups(aggregate.cmp).reference.sig, ['golden', 'reference']);
assert.deepEqual(aggregate.variants.map((leaf) => leaf.sub), ['7-4', '7-4.anyof']);
const missing = aggregateCells([children[0], cell('7-4.anyof', { golden: entry(4), reference: entry(0, { na: 1 }), peer: entry(0, { na: 1 }) })], '7-4', 'nullable');
assert.equal(missing.cmp.reference.na, 1);
const failure = aggregateCells([cell('7-4', { golden: entry(1), reference: entry(2) }), cell('7-4.anyof', { golden: entry(4), reference: entry(0, { na: 1 }) })], '7-4', 'nullable');
assert.equal(failure.cmp.reference.na, 0);
assert.notEqual(failure.cmp.reference.sig, failure.cmp.golden.sig);
const mixed = cell('7-4.mixed_labels', { golden: entry(1), reference: entry(1) });
assert.equal(leafCells({ cells: { null: { variants: [...children, mixed] }, string: { variants: [mixed] } } }).size, 3);
assert.throws(() => leafCells({ cells: { null: { variants: [mixed] }, string: { variants: [{ ...mixed, status: 'problem' }] } } }), /inconsistent/);
for (const family of ['beam', 'glm47']) {
  const labels = ['7-4', '7-4.anyof', '7-4.mixed_labels', '7-5', '7-5.union'];
  const config = { folds: { null_descriptions: { '7-4': 'JSON null', '7-5': 'String null' },
    mixed_families: { '7-4.mixed_labels': [] }, unified: Object.fromEntries(labels.map((label, index) =>
      [label, { legacy_label: label, canonical_label: label, taxonomy_order: [7, index] }])) } };
  const tab = { id: 'tab-unified', candidates: [{ key: 'golden' }],
    columns: labels.map(label => ({ sub: label, label, group_key: 'unified_g7' })),
    column_groups: [{ key: 'unified_g7', span: labels.length }], stats: {}, glossary: [],
    rows: [{ family, cells: Object.fromEntries(labels.map(label =>
      [label, { ...cell(label, { golden: entry(1) }), family }])) }] };
  const original = structuredClone(tab.rows[0].cells);
  foldNullVariants(tab, config);
  for (const parent of ['7-4', '7-5']) assert.ok(tab.rows[0].cells[parent].variants.some(leaf => leaf.sub === '7-4.mixed_labels'));
  assert.deepEqual(Object.fromEntries(leafCells(tab.rows[0])), original);
  assert.equal(tab.stats.fixture_cases, labels.length);
}
const allInapplicable = aggregateCells([
  cell('schema_a', { golden: entry(0, { na: 1 }), dynamo: entry(0, { na: 1 }), sglang: entry(0, { na: 1 }) }, 'na'),
  cell('schema_b', { golden: entry(0, { na: 1 }), dynamo: entry(0, { na: 1 }), sglang: entry(0, { na: 1 }) }, 'na'),
], 'schema', 'schema variants');
assert.deepEqual(Object.keys(allInapplicable.cmp), ['dynamo', 'golden', 'sglang']);
assert.equal(allInapplicable.cmp.dynamo.na, 1);
assert.equal(allInapplicable.cmp.sglang.na, 1);
for (const [parent, variant] of [
  ['arg_marker_in_string', 'arg_marker_in_string_pr424'],
  ['tool_in_reason', 'gpt_oss_interrupted_reasoning_then_tool'],
  ['tool_no_close', 'gpt_oss_truncated_tool_resynchronizes_at_next_message'],
]) for (const missingParent of [false, true]) {
  const leaves = [cell(parent, { golden: entry(1), engine: entry(1) }, missingParent ? 'na' : 'ok'), cell(variant, { golden: entry(1), engine: entry(1) })];
  const before = structuredClone(leaves);
  const metadata = { folds: { unified: Object.fromEntries([parent, variant].map(sub => [sub, { canonical_label: sub, case: [parent, sub] }])) } };
  const folded = { id: 'tab-unified', columns: [parent, variant].map(sub => ({ sub, label: sub, desc: 'shared behavior', group_key: 'group' })), rows: [{ family: 'sample', cells: Object.fromEntries(leaves.map(leaf => [leaf.sub, leaf])) }], column_groups: [{ key: 'group', span: 2 }], stats: {}, glossary: [] };
  foldSchemaVariants(folded, metadata);
  assert.deepEqual(folded.columns.map(column => column.sub), [parent]);
  assert.deepEqual([...leafCells(folded.rows[0]).keys()], [parent, variant]);
  assert.deepEqual(leaves, before, 'folding must preserve the original leaves');
  const expected = before.map(leaf => ({ ...leaf, tooltip: { ...leaf.tooltip, head: `${leaf.sub} — ${leaf.tooltip.head}` } }));
  assert.deepEqual(folded.rows[0].cells[parent].variants, expected);
  assert.equal(folded.stats.fixture_cases, missingParent ? 1 : 2);
}
for (const [parentNa, code, signature, expected] of [
  [true, 'missing_capture', 1, 'missing'], [true, 'source_unavailable', 1, 'blocked'],
  [true, 'unsupported_request_mode', 1, 'unsupported'], [false, 'missing_capture', 1, 'missing'],
  [false, 'missing_capture', 2, 'diverge'], [true, null, 1, 'pass'],
]) {
  const parent = cell('parent', { golden: entry(1), engine: entry(signature) }, parentNa ? 'na' : 'ok');
  const variant = cell('variant', { golden: entry(1), engine: entry(1, { na: Number(code !== null) }) });
  variant.tooltip.candidates = [{ key: 'engine', block: code ? { unavailable_code: code } : {} }];
  assert.equal(selectedCaptureOutcome(aggregateCells([parent, variant], 'parent', ''), 'engine'), expected);
}

} finally {
  for (const key of Object.getOwnPropertyNames(globalThis)) {
    if (!Object.hasOwn(savedGlobals, key) && Object.getOwnPropertyDescriptor(globalThis, key).configurable) delete globalThis[key];
  }
  for (const [key, descriptor] of Object.entries(savedGlobals)) {
    if (descriptor.configurable) Object.defineProperty(globalThis, key, descriptor);
  }
}

});

await test('numeric contracts', async () => {
const savedGlobals = Object.getOwnPropertyDescriptors(globalThis);
try {
const { equivalenceEntries, canonicalUnifiedEvents } = await import('./model.js');
const { default: assert } = await import('node:assert/strict');
const { readFile } = await import('node:fs/promises');
const { canonicalArguments, sameValue } = await import('./values.js');
const { classifyUnified } = await import('./model.js');
const { assembleStream } = await import('./model.js');
const { foldNullVariants, rankUnifiedColumns } = await import('./model.js');


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
const metadata=JSON.parse(await readFile(new URL('../viewer-catalog.json',import.meta.url),'utf8')).metadata;
const numeric=Object.entries(metadata.folds.unified).filter(([,item])=>metadata.folds.variant_groups[item.legacy_label]);
assert.equal(numeric.length,19);
const makeCell=(sub,label,family,status='ok')=>({sub,family,case_id:`UNIFIED.${label}`,kind:status==='missing'?'missing':'cell',status,red_on_diff:true,
  cmp:{golden:{sig:1,na:0},dynamo:{sig:2,na:0}},tooltip:{head:sub,description:'schema '+label}});
for (const family of ['qwen3','deepseek_v4']) {
  const entries=[['unused_reference_graph_parameter_types',metadata.folds.unified.unused_reference_graph_parameter_types],['nullable_reference_alias_literals',metadata.folds.unified.nullable_reference_alias_literals],...numeric];
  const columns=entries.map(([sub,item])=>({sub,label:item.canonical_label,desc:'schema '+item.legacy_label,group_key:'args'}));
  const cells=Object.fromEntries(entries.map(([sub,item])=>[sub,makeCell(sub,item.legacy_label,family)]));
  const bare=structuredClone([cells.unused_reference_graph_parameter_types,cells.nullable_reference_alias_literals]);
  const tab={id:'tab-unified',columns,rows:[{family,cells}],candidates:[{key:'dynamo'}],column_groups:[{key:'args',span:columns.length}],stats:{},glossary:[{rows:entries.map(([,item])=>[item.canonical_label,'description'])}]};
  foldNullVariants(tab,metadata);
  rankUnifiedColumns(tab,metadata);
  assert.deepEqual(tab.columns.slice(-3).map(column=>column.label),['7-17','7-19','7-18']);
  assert.deepEqual(tab.glossary[0].rows.slice(-3).map(([label])=>label),['7-17','7-19','7-18']);
  assert.deepEqual([cells.unused_reference_graph_parameter_types,cells.nullable_reference_alias_literals],bare);
  for (const [label,count] of [['7-17',9],['7-19',3],['7-18',7]]) {
    const column=tab.columns.find(column=>column.label===label), cell=cells[column.sub];
    assert.equal(cell.case_id,`UNIFIED.${label}`);assert.equal(cell.variants.length,count);
    assert.equal(cell.status==='na',label==='7-19'&&family==='deepseek_v4');
  }
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
assert.equal(classifyUnified('sample', [{ kind: 'tool_call', name: 'f', arguments: { x: null } }], [{ kind: 'tool_call', name: 'f', arguments: { x: 'null' } }], markers), 'ARG_MISMATCH');
assert.equal(classifyUnified('sample', [{ kind: 'reasoning', text: 'x' }, { kind: 'text', text: 'y' }], [{ kind: 'text', text: 'y' }, { kind: 'reasoning', text: 'x' }], markers), 'ORDER');
console.log('viewer model semantics passed');

});
