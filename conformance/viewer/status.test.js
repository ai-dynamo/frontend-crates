// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { buildStatus, cellState, defaultReference, parseStatusArguments, printStatus, runStatus, selectRows, selectTabs, statusTotals, validateUnifiedKnownDivergences } from './status.js';
import { selectedCellOutcome } from './selection_comparison.js';

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
assert.equal(cellState({ ...cell, red_on_diff: true, cmp: { ...comparison, reference: { ...comparison.reference, leak: 1 } } }, reference)[0], 'green');
assert.equal(cellState({ ...cell, cmp: { ...comparison, reference: { ...comparison.reference, leak: 1 } } }, reference)[1], 'the default Reference leaked structured markup');
assert.equal(cellState({ ...cell, red_on_diff: true, cmp: { reference: { ...comparison.reference, sig: ['golden', 'reference'] }, golden: { ...comparison.golden, sig: ['golden', 'reference'] } } }, reference)[0], 'green');
const clean = { sig: 0, na: 0, err: 0, leak: 0 };
const choices = { base: clean, equal: clean, different: { ...clean, sig: 1 }, missing: { ...clean, na: 1 }, failed: { ...clean, err: 1 } };
const outcome = (overrides = {}) => selectedCellOutcome({ cmp: choices, base: 'base', shown: ['equal'], ...overrides });
assert.equal(outcome().marker, '=');
assert.equal(outcome({ shown: ['base', 'equal', 'missing'] }).marker, '?');
assert.equal(outcome({ shown: ['different', 'missing', 'failed'] }).marker, '1Δ');
assert.equal(outcome({ shown: ['failed', 'missing'] }).marker, '');
assert.equal(outcome({ base: null }).className, 'cmp-nobase');
assert.equal(outcome({ base: 'unknown' }).referenceState, 'missing');
assert.equal(outcome({ base: 'missing' }).referenceState, 'unavailable');
assert.equal(outcome({ base: 'failed', shown: [] }).category, 'na');
assert.equal(outcome({ base: 'failed', shown: ['equal'] }).marker, '1Δ');
assert.equal(outcome({ base: 'failed', shown: [], redOnDiff: true }).marker, '✗');
assert.equal(outcome({ family: 'unsupported', parserNi: { base: { label: 'Selected parser', families: ['supported'] } } }).marker, 'x');
assert.equal(outcome({ cmp: { ...choices, golden: clean }, redOnDiff: true, shown: ['different'] }).category, 'ok');
assert.equal(outcome({ cmp: { ...choices, golden: clean }, redOnDiff: true, shown: ['different'] }).marker, '1Δ');
assert.equal(outcome({ cmp: { ...choices, base: choices.different, golden: clean }, redOnDiff: true }).marker, '✗');
assert.equal(outcome({ cmp: { ...choices, base: { ...clean, leak: 1 }, golden: clean }, redOnDiff: true }).marker, '↯=');
assert.equal(outcome({ cmp: { ...choices, base: { ...clean, leak: 1 } } }).category, 'problem');

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
const cli = fileURLToPath(new URL('./status.js', import.meta.url));
assert.equal(spawnSync(process.execPath, [cli, '--help']).status, 0);
assert.equal(spawnSync(process.execPath, [cli, '--unknown']).status, 2);
const errors = [];
assert.equal(await runStatus({ ...parseStatusArguments([]), root: '/missing/conformance-status-test-root' }, () => {}, (line) => errors.push(line)), 2);
assert.match(errors[0], /^ERROR: /);
console.log('status schemas, issue reasons, aliases, missing/error precedence, CLI arguments and resource failures passed');
