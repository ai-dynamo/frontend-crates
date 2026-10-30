// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { selectedCaptureOutcome } from './model.js';
import { compileSearch, comparisonView, captureCounts, isProblemOutcome, statusLabels, verticalDefault } from './comparison.js';
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
test('search expressions support precedence, groups, phrases and whitespace shorthand', () => {
  for (const [query, fields, expected] of [
    ['', [], true],
    ['deepseek AND (call OR blah)', ['DeepSeek', 'tool call'], true],
    ['deepseek AND (call OR blah)', ['llama', 'tool call'], false],
    ['deepseek AND (call OR blah)', ['deepseek', 'reasoning'], false],
    ['deepseek OR llama AND call', ['deepseek'], true],
    ['deepseek OR llama AND call', ['llama'], false],
    ['(deepseek OR llama) AND call', ['deepseek'], false],
    ['deepseek and ((call or blah))', ['DeepSeek', 'blah'], true],
    ['deep seek', ['deepseek'], true],
    ['deep seek', ['deep seek'], true],
    ['deep seek', ['deep', 'seek'], false],
    ['deep seek AND call', ['deepseek', 'call'], true],
    ['deep seek AND call', ['deepseek'], false],
    ['"deep seek"', ['deepseek'], false],
    ['"deep seek"', ['DEEP SEEK'], true],
    ['"AND"', ['and'], true],
    ['"call (tool)"', ['call (tool)'], true],
    ['"say \\"hi\\""', ['say "hi"'], true],
    ['[a].*', ['[a].*'], true],
    ['[a].*', ['anything'], false],
  ]) assert.equal(compileSearch(query)(fields), expected, query);
  for (const query of ['"', '"unterminated', '""', '()', '(call', 'call)', 'AND call', 'call OR', 'call AND OR blah', 'call (blah)', '"call" blah']) {
    assert.throws(() => compileSearch(query), SyntaxError, query);
  }
});
test('boolean filters count only matching underlying cases across model and case fields', () => {
  const a = cell('a', 'pass', {tooltip: {description: 'single call'}});
  const b = cell('b', 'leak', {tooltip: {description: 'reasoning'}});
  const folded = {...tab, columns: [{sub: 'fold', label: 'group', case_id: 'folded_id', desc: 'Union alternatives', group_key: 'core'}, {sub: 'a', group_key: 'core'}],
    rows: [{family: 'deepseek', cells: {fold: {variants: [a,b]}, a}}, {family: 'llama', cells: {fold: {variants: [a,b]}, a}}]};
  const view = comparisonView(folded, {...state, search: 'deep seek AND (call OR nowhere)'});
  assert.deepEqual(view.rows.map(row => row.family), ['deepseek']);
  assert.equal(captureCounts(view, 'd').total, 1);
  assert.equal(captureCounts(view, 'd').pass, 1);
  assert.equal(captureCounts(view, 'd').leak, 0);
  for (const query of ['"union alternatives"', 'folded_id']) {
    assert.equal(captureCounts(comparisonView(folded, {...state, search: query}), 'd').total, 4, query);
  }
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
