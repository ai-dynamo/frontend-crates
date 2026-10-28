// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { selectedCaptureOutcome } from './model.js';
import { comparisonView, captureCounts, statusLabels, verticalDefault } from './comparison.js';
const cell = (id, status = 'pass', extra = {}) => ({ kind: 'cell', sub: id, case_id: id, family: 'family', status: 'ok',
  cmp: { golden: {sig: 1}, d: {sig: status === 'diverge' || status === 'documented' ? 2 : 1,
    ...(status === 'error' ? {err: 1} : status === 'leak' ? {leak: 1} : ['missing', 'unsupported', 'blocked'].includes(status) ? {na: 1} : {})} },
  tooltip: { candidates: [{key: 'd', block: status === 'expected' ? {error: 'expected'} : status === 'unsupported' ? {unavailable_code: 'unsupported_request_mode'} : status === 'blocked' ? {unavailable_code: 'source_unavailable'} : status === 'documented' ? {explanation: 'intentional'} : {}}],
    reasons: status === 'documented' ? [{intentional: true, impl: 'd', reference_impl: 'golden', candidate_sig: 2, reference_sig: 1, reason: 'intentional'}] : [] }, ...extra });
const leaves = Object.keys(statusLabels).filter(s => s !== 'uncompared').map(s => cell(s, s));
leaves.push(cell('uncompared', 'pass', {cmp: {d: {sig: 1}, golden: {na: 1}}}));
const columns = leaves.map(c => ({sub: c.sub, label: c.sub, desc: 'case ' + c.sub, group_key: 'core'}));
const tab = {rows: [{family: 'family', model_label: 'Model', cells: Object.fromEntries(leaves.map(c => [c.sub, c]))}], columns,
  column_groups: [{key: 'core', label: 'Core'}]};
const state = {groups: new Set(['core']), search: '', focus: 'all'};
test('each status is counted separately without scores', () => {
  const counts = captureCounts(comparisonView(tab, state), 'd');
  for (const status of Object.keys(statusLabels)) assert.equal(counts[status], 1, status);
  assert.equal(counts.total, Object.keys(statusLabels).length);
  assert.equal(Object.hasOwn(counts, 'score'), false);
  assert.equal(captureCounts({entries: []}, 'd').total, 0);
});
test('filters and focus use the same applicable leaf inventory', () => {
  assert.deepEqual(captureCounts(comparisonView(tab, {...state, focus: 'problems'}), 'd'), captureCounts(comparisonView(tab, state), 'd'));
  assert.equal(captureCounts(comparisonView(tab, {...state, search: 'Model'}), 'd').total, leaves.length);
  assert.equal(captureCounts(comparisonView(tab, {...state, search: 'case leak'}), 'd').leak, 1);
  assert.equal(captureCounts(comparisonView(tab, {...state, groups: new Set()}), 'd').total, 0);
});
test('folding deduplicates underlying cases per family and skips inapplicable leaves', () => {
  const a = cell('a'), b = cell('b', 'leak'), na = cell('na', 'pass', {status: 'na'});
  const folded = {...tab, columns: [{sub: 'fold', group_key: 'core'}, {sub: 'a', group_key: 'core'}],
    rows: [{family: 'family', cells: {fold: {variants: [a,b,na]}, a}}]};
  const counts = captureCounts(comparisonView(folded, state), 'd');
  assert.equal(counts.total, 2); assert.equal(counts.pass, 1); assert.equal(counts.leak, 1);
});
test('Vertical default preserves explicit horizontal URL overrides', () => {
  for (const query of ['', 'vertical=1', 'transpose=true']) assert.equal(verticalDefault(new URLSearchParams(query)), true);
  for (const query of ['vertical=0', 'transpose=0', 'vertical=false', 'vertical=0&transpose=1']) assert.equal(verticalDefault(new URLSearchParams(query)), false);
});

test('folded matrix statuses preserve every leaf outcome', () => {
  for (const leaf of leaves) assert.equal(selectedCaptureOutcome({...leaf,variants: [leaf]}, 'd'), selectedCaptureOutcome(leaf,'d'));
});
