// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import assert from 'node:assert/strict';
import { aggregateCells, leafCells, foldNullVariants } from './folds.js';
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
