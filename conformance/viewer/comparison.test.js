// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { selectedCaptureOutcome } from './model.js';
import { comparisonView, captureCounts, isProblemOutcome, statusLabels, verticalDefault } from './comparison.js';
import { captureSegmentHtml } from './presentation.js';
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
test('scoped search keeps family mentions in descriptions out of bare family queries', () => {
  const a = cell('a', 'pass', {tooltip: {description: 'Beam and GLM single call'}});
  const scoped = {...tab, columns: [{sub: 'a', label: 'TC Truncation', group_key: 'core'}], rows: [
    {family: 'beam', cells: {a}}, {family: 'glm47', cells: {a}},
  ]};
  for (const query of ['beam', 'parser:beam', 'model:beam', 'beam (TC OR nowhere)']) {
    const view = comparisonView(scoped, {...state, search: query});
    assert.deepEqual(view.rows.map(row => row.family), ['beam'], query);
    assert.equal(captureCounts(view, 'd').total, 1);
  }
  assert.equal(comparisonView(scoped, {...state, search: 'description:Beam'}).rows.length, 2);
  assert.equal(comparisonView(scoped, {...state, search: 'TC Trun'}).rows.length, 2);
});
test('Boolean cross-axis union counts each applicable leaf once including folded overlaps', () => {
  const a = cell('tc'), b = cell('other', 'leak');
  const folded = {...tab, columns: [
    {sub: 'fold', label: 'Grouped cases', group_key: 'core'},
    {sub: 'tc', label: 'TC', group_key: 'core'},
    {sub: 'other', label: 'Other', group_key: 'core'},
  ], rows: ['beam','glm47'].map(family => ({family, cells: {fold: {variants: [a,b]}, tc: a, other: b}}))};
  for (const focus of ['all', 'disagreements', 'problems']) {
    const view = comparisonView(folded, {...state, focus, search: 'beam OR TC'});
    assert.equal(captureCounts(view, 'd').total, 3);
    assert.equal(captureCounts(view, 'd').leak, 1);
    assert.deepEqual(view.rows.map(row => row.family), ['beam','glm47']);
  }
});
test('column searches retain all families even when their matching cases are inapplicable', () => {
  const a = cell('a'), na = cell('a', 'pass', {status: 'na'});
  const sparse = {...tab, columns: [{sub: 'a', label: 'TC Truncation', group_key: 'core'}],
    rows: [{family: 'beam', cells: {a}}, {family: 'glm47', cells: {a: na}}]};
  const view = comparisonView(sparse, {...state, search: 'TC Trun'});
  assert.equal(view.rows.length, 2);
  assert.equal(captureCounts(view, 'd').total, 1);
});
test('Capture boxes stay empty and keep accessible outcome labels', () => {
  for (const status of Object.keys(statusLabels)) {
    const segment = captureSegmentHtml(status, 'capture');
    assert.match(segment, /^<i [^>]+><\/i>$/, status);
    assert.ok(segment.includes('capture: ' + statusLabels[status]), status);
    assert.ok(!segment.includes('%'), status);
  }
  assert.ok(captureSegmentHtml('pass', '<capture>').includes('&lt;capture&gt;'));
});
test('each status is counted separately without scores', () => {
  const counts = captureCounts(comparisonView(tab, state), 'd');
  for (const status of Object.keys(statusLabels)) assert.equal(counts[status], 1, status);
  assert.equal(counts.total, Object.keys(statusLabels).length);
  assert.equal(Object.hasOwn(counts, 'score'), false);
  assert.equal(captureCounts({entries: []}, 'd').total, 0);
});
test('Problems focus includes mismatches, parser errors, and leaks only', () => {
  for (const status of ['documented', 'diverge', 'error', 'leak']) assert.equal(isProblemOutcome(status), true, status);
  for (const status of ['expected', 'missing', 'unsupported', 'blocked', 'uncompared', 'pass']) assert.equal(isProblemOutcome(status), false, status);
});
test('filters and focus use the same applicable leaf inventory', () => {
  assert.deepEqual(captureCounts(comparisonView(tab, {...state, focus: 'problems'}), 'd'), captureCounts(comparisonView(tab, state), 'd'));
  assert.equal(captureCounts(comparisonView(tab, {...state, search: 'Model'}), 'd').total, leaves.length);
  assert.equal(captureCounts(comparisonView(tab, {...state, search: 'description:case description:leak'}), 'd').leak, 1);
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
