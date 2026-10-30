// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { compileSearch, searchContext } from './search.js';

const context = {families: ['beam', 'glm47', 'qwen3', 'qwen3_coder', 'deepseek_v4'],
  aliases: new Map([['7-17','7-arg_numeric_14_const_decimal'], ['5-2','5-tool_no_close']])};
const search = (query, columns = [], models = [], descriptions = []) => compileSearch(query, context)(columns, models, descriptions);

test('Google-style words, phrases, alternatives, exclusions and explicit scopes', () => {
  for (const [query, columns, models, expected] of [
    ['', [], [], true],
    ['beam', ['Beam appears here'], ['glm47'], false],
    ['beam', ['Other'], ['beam'], true],
    ['parser:beam', [], ['beam'], true],
    ['model:beam', [], ['beam'], true],
    ['parser:unknown', ['unknown'], ['beam'], false],
    ['parser:!', [], ['beam'], false],
    ['.', [], ['beam'], false],
    ['.', ['.'], ['beam'], true],
    ['qwen3', [], ['qwen3_coder'], true],
    ['parser:qwen3', [], ['qwen3_coder'], true],
    ['col:beam', ['Beam directed'], ['glm47'], true],
    ['TC Trun', ['TC Truncation / recovery'], ['glm47'], true],
    ['TC', ['mismatch'], ['glm47'], false],
    ['call', ['Tool calls'], [], true],
    ['call', ['recall'], [], false],
    ['numeric', ['arg_numeric_14'], [], true],
    ['guided decoding', ['Guided argument fidelity', 'Decoding tests'], [], true],
    ['"guided decoding"', ['Guided argument fidelity', 'Decoding tests'], [], false],
    ['"guided decoding"', ['Guided decoding tests'], [], true],
    ['"call"', ['calls'], [], false],
    ['beam (guided decoding OR TC)', ['TC Truncation'], ['beam'], true],
    ['beam (guided decoding OR TC)', ['Guided Decoding'], ['beam'], true],
    ['beam (guided decoding OR TC)', ['Guided Decoding'], ['glm47'], false],
    ['beam (guided decoding OR TC)', ['Other'], ['beam'], false],
    ['beam OR TC', ['Other'], ['beam'], true],
    ['beam OR TC', ['TC'], ['glm47'], true],
    ['beam OR TC', ['Other'], ['glm47'], false],
    ['beam OR qwen AND TC', ['Other'], ['beam'], true],
    ['(beam OR qwen) AND TC', ['Other'], ['beam'], false],
    ['(beam OR qwen) TC', ['TC'], ['qwen3_coder'], true],
    ['beam -TC', ['Other'], ['beam'], true],
    ['beam -col:TC', ['TC Truncation'], ['beam'], false],
    ['-parser:beam TC', ['TC'], ['glm47'], true],
    ['beam -(TC OR "guided decoding")', ['TC'], ['beam'], false],
    ['beam -"guided decoding"', ['Guided Decoding'], ['beam'], false],
    ['deep seek', [], ['deepseek_v4'], true],
    ['deep seek TC', ['TC'], ['deepseek_v4'], true],
    ['parser:deep seek TC', ['TC'], ['deepseek_v4'], true],
    ['"deep seek"', ['Deep Seek'], ['glm47'], true],
    ['parser:"deep seek"', [], ['deepseek_v4'], false],
    ['"OR"', ['OR'], [], true],
    ['"parser:beam"', ['parser:beam'], ['glm47'], true],
    ['"call (tool)"', ['call (tool)'], [], true],
    ['"say \\"hi\\""', ['say "hi"'], [], true],
    ['[a].*', ['[a].*'], [], true],
    ['[a].*', ['anything'], [], false],
  ]) assert.equal(search(query, columns, models), expected, query);
  assert.equal(search('leak', [], [], ['markup leak']), false);
  assert.equal(search('description:leak', [], [], ['markup leak']), true);
  assert.equal(search('description:"markup leak"', [], [], ['markup leak']), true);
  for (const query of ['"', '"unterminated', '""', '()', '(call', 'call)', 'AND call', 'call OR',
    'call AND OR blah', 'parser:', 'parser: AND call', 'model:(beam OR qwen)', '-', 'beam -', 'col:"']) {
    assert.throws(() => compileSearch(query, context), SyntaxError, query);
  }
});

test('numbered aliases expand atomically without damaging operators, qualifiers or quoted text', () => {
  for (const [query, expected] of [
    ['7-17', '7-arg_numeric_14_const_decimal'],
    ['"7-17"', '"7-arg_numeric_14_const_decimal"'],
    ['UNIFIED.7-17', '7-arg_numeric_14_const_decimal'],
    ['beam (7-17 OR col:5-2)', 'beam (7-arg_numeric_14_const_decimal OR col:5-tool_no_close)'],
    ['-7-17 AND 5-2', '-7-arg_numeric_14_const_decimal AND 5-tool_no_close'],
    ['"see 7-17"', '"see 7-17"'],
    ['parser:7-17', 'parser:7-17'],
  ]) assert.equal(compileSearch(query, context).canonicalQuery, expected);
  const match = compileSearch('7-17', context);
  assert.equal(match(['UNIFIED.7-arg_numeric_14_const_decimal']), true);
  assert.equal(match(['arg_numeric_14_const_decimal']), true);
  assert.equal(match(['7-17']), false);
  assert.equal(match(['arg_numeric_14_const_decimal_extra']), false);
  const unknown = compileSearch('99-99', context);
  assert.equal(unknown(['99-99']), false);
  assert.equal(unknown.notices.length, 1);
});

test('shared canonical queries survive numbering, group prefixes, ordering and family changes', () => {
  const makeTab = (number, prefix = '7') => ({columns: [{sub: 'arg_numeric_14_const_decimal',
    case_id: `${prefix}-arg_numeric_14_const_decimal`, compact_prefix: '7', display_number: number}], rows: []});
  const oldContext = searchContext(makeTab(17), ['beam']);
  const newContext = searchContext(makeTab(2, '77'), ['glm47']);
  const canonical = compileSearch('7-17', oldContext).canonicalQuery;
  assert.equal(compileSearch(canonical, newContext)(['77-arg_numeric_14_const_decimal']), true);
  assert.equal(compileSearch(canonical, newContext)(['7-17', 'other_case']), false);
  assert.equal(compileSearch(canonical, newContext).canonicalQuery, canonical);
  assert.equal(compileSearch('7-17', newContext)(['other_case']), false);
});

test('Boolean laws hold across family/case combinations and negation', () => {
  for (const family of ['beam','glm47','qwen3']) for (const column of ['TC Truncation','Guided Decoding','Other']) {
    const a = search('beam', [column], [family]), b = search('TC', [column], [family]);
    assert.equal(search('beam OR TC', [column], [family]), a || b);
    assert.equal(search('beam TC', [column], [family]), a && b);
    assert.equal(search('-(beam OR TC)', [column], [family]), !a && !b);
    assert.equal(search('beam OR (beam TC)', [column], [family]), a);
  }
});
