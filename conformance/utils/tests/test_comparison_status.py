# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import subprocess
from pathlib import Path


def test_comparison_status_preserves_capture_and_reference_failures():
    source = Path(__file__).resolve().parents[1] / "src/assets/comparison_layout.js"
    script = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync(process.argv[1], 'utf8');
const context = vm.createContext({});
vm.runInContext(source.slice(source.indexOf('  function tooltipCandidate('),
  source.indexOf('  function makeHeader(')), context);
const candidate = {key: 'candidate', impl: 'dynamo'};
function cell(block = {}, reference = {sig: 1}) {
  return {kind: 'cell', cmp: {candidate: {sig: 2}, reference, golden: {sig: 1}},
    tooltip: {candidates: [{key: 'candidate', block}], reasons: []}};
}
function status(value, unified = true, selected = candidate) {
  return context.outputStatus({id: unified ? 'tab-unified' : 'tab-batch'}, value, selected, 'reference');
}
assert.equal(status(cell({explanation: 'Inherited unchanged from Dynamo v2 0.7.18.'})), 'diverge');
const intentional = cell({explanation: 'intentional recovery difference'});
intentional.tooltip.reasons = [{impl: 'candidate', intentional: true, reason: 'intentional recovery difference',
  candidate_sig: 2, reference_impl: 'golden', reference_sig: 1}];
assert.equal(status(intentional), 'documented');
intentional.tooltip.reasons[0].impl = 'vllm';
assert.equal(status(intentional), 'diverge');
intentional.known_divergence = true;
assert.equal(status(intentional), 'diverge');
intentional.tooltip.reasons[0].impl = 'candidate';
intentional.tooltip.reasons[0].reference_sig = 99;
assert.equal(status(intentional), 'diverge');
intentional.tooltip.reasons[0].reference_sig = 1;
intentional.tooltip.reasons[0].reference_impl = 'vllm';
assert.equal(status(intentional), 'diverge');
for (const key of ['vllm_python', 'vllm_python-b-0-26-0', 'vllm_python-0-25-1']) {
  const peer = {key, impl: 'vllm'};
  const value = cell({explanation: 'intentional recovery difference'});
  value.cmp[key] = value.cmp.candidate;
  value.tooltip.candidates[0].key = key;
  value.tooltip.reasons = [{impl: 'vllm_python', intentional: true, reason: 'intentional recovery difference',
    candidate_sig: 2, reference_impl: 'golden', reference_sig: 1}];
  assert.equal(status(value, true, peer), 'documented', key);
  value.tooltip.candidates[0].block.explanation = 'Inherited unchanged from an older capture';
  assert.equal(status(value, true, peer), 'diverge', key);
  value.tooltip.candidates[0].block.explanation = 'intentional recovery difference';
  value.tooltip.reasons[0].impl = 'vllm_rust';
  assert.equal(status(value, true, peer), 'diverge', key);
}
const scoped = cell();
scoped.known_divergence = true;
scoped.cmp = {dynamo_v1: {sig: 1}, dynamo_v2: {sig: 2}, vllm_python: {sig: 3}};
scoped.tooltip.candidates = [{key: 'dynamo_v1', block: {}},
  {key: 'dynamo_v2', block: {explanation: 'known batch/stream difference'}}, {key: 'vllm_python', block: {}}];
scoped.tooltip.reasons = [{impl: 'dynamo_v2', intentional: true, reason: 'known batch/stream difference',
  candidate_sig: 2, reference_impl: 'dynamo_v1', reference_sig: 1}];
for (const [selected, reference, expected] of [
  ['dynamo_v2', 'dynamo_v1', 'documented'], ['dynamo_v1', 'dynamo_v2', 'documented'],
  ['vllm_python', 'dynamo_v1', 'diverge'], ['vllm_python', 'dynamo_v2', 'diverge'],
]) {
  assert.equal(context.outputStatus({id: 'tab-batch'}, scoped, {key: selected}, reference), expected);
}
for (const reference of [undefined, {sig: 0, na: 1}, {sig: 1, err: 1}]) {
  const value = cell({}, reference);
  if (reference === undefined) delete value.cmp.reference;
  assert.equal(status(value, false), 'uncompared', JSON.stringify(reference));
  assert.equal(context.eligibleData(value, candidate), true);
}
const matching = cell();
matching.cmp.candidate.sig = 1;
assert.equal(status(matching), 'pass');
for (const reason of ['not captured at 0.7.1', 'not applicable for this case',
  'No batch model_text for this case.', 'No batch-on-stream capture for this engine.']) {
  const value = cell({unavailable: reason});
  value.cmp.candidate = {sig: 0, na: 1};
  assert.equal(status(value), 'missing', reason);
  assert.equal(context.eligibleData(value, candidate), false);
}
for (const reason of ['vLLM has no parser for family', 'no harmony parser in Dynamo v2 0.7.1',
  'No registered Dynamo v2 parser for this family.',
  'Dynamo parser v2 stream parser not yet implemented for this family']) {
  const value = cell({unavailable: reason});
  value.cmp.candidate = {sig: 0, na: 1};
  assert.equal(status(value), 'unsupported', reason);
}
for (const [entry, block, expected] of [
  [{sig: 2, err: 1}, {}, 'error'], [{sig: 2}, {exception: 'failed'}, 'error'],
  [{sig: 2}, {error: {message: 'failed'}}, 'error'],
  [{sig: 2}, {error: 'intentional invalid input'}, 'expected'],
  [{sig: 2, leak: 1}, {}, 'leak'],
]) {
  const value = cell(block);
  value.cmp.candidate = entry;
  assert.equal(status(value), expected);
  assert.equal(context.eligibleData(value, candidate), true);
}
"""
    result = subprocess.run(["node", "-e", script, str(source)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
