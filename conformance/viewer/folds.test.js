// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import assert from 'node:assert/strict';
import { aggregateCells, leafCells, foldNullVariants, foldSchemaVariants } from './folds.js';
import { selectedCaptureOutcome } from './selection_comparison.js';
import { equivalenceGroups } from './comparison.js';

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
const allInapplicable = aggregateCells([
  cell('schema_a', { golden: entry(0, { na: 1 }), dynamo: entry(0, { na: 1 }), sglang: entry(0, { na: 1 }) }, 'na'),
  cell('schema_b', { golden: entry(0, { na: 1 }), dynamo: entry(0, { na: 1 }), sglang: entry(0, { na: 1 }) }, 'na'),
], 'schema', 'schema variants');
assert.deepEqual(Object.keys(allInapplicable.cmp), ['dynamo', 'golden', 'sglang']);
assert.equal(allInapplicable.cmp.dynamo.na, 1);
assert.equal(allInapplicable.cmp.sglang.na, 1);
const tab = { id: 'tab-toolcalling-batch', columns: [
  { sub: '7-4', label: '7-4', group_key: 'args' }, { sub: '7-4.anyof', label: '7-4.anyof', group_key: 'args' },
  { sub: '7-5', label: '7-5', group_key: 'args' }, { sub: '7-4.mixed_labels', label: '7-4.mixed_labels', group_key: 'args' },
], rows: [{ family: 'sample', cells: Object.fromEntries([...children, cell('7-5', { golden: entry(1), reference: entry(1) }), mixed].map((item) => [item.sub, item])) }], column_groups: [{ key: 'args', span: 4 }], stats: {}, glossary: [] };
foldNullVariants(tab, { folds: { null_descriptions: { '7-4': 'nullable', '7-5': 'string' }, mixed_families: { '7-4.mixed_labels': ['sample'] } } });
assert.equal(tab.columns.length, 2);
assert.equal(tab.stats.fixture_cases, 4);
assert.equal(tab.stats.variant_columns, 2);
assert.equal(leafCells(tab.rows[0]).size, 4);
assert.deepEqual(tab.rows[0].cells['7-5'].variants.map((item) => item.sub), ['7-5', '7-4.mixed_labels']);
console.log('folded comparison, ordered leaves, and shared-fixture accounting passed');

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
