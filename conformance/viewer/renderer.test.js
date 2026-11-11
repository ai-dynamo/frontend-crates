// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';
import { firstPromptDifference, rendererOutcome, rendererPanel, validateRendererSuite } from './renderer.js';
import { selectedCaptureOutcome, tabCandidates } from './model.js';
import { buildTab, tabPresentation } from './presentation.js';
import { FixtureResources } from './resources.js';
import { sha256 } from './data.js';
import { discoverHistory } from './app.js';
import { gzipSync } from './vendor/fflate/esm/browser.js';

const prompt = (text = 'Hello', extra = {}) => ({ kind: 'prompt', text, segments: null, pending_segments: 0, media_message_order: null, ...extra });
const input = { id: 'RENDERER.single_turn', group: 'conversations', description: 'One user turn.',
  request: { model: 'sample', messages: [{ role: 'user', content: 'Hello' }] }, settings: { add_generation_prompt: true, request_mode: 'typed' },
  metadata_expectation: { pending_segments: 0, media_message_order: null }, metadata_origin: 'authored-contract' };
const makeSuite = () => {
  const model = { id: 'sample', label: 'Sample', repository: 'sample/model', revision: 'a'.repeat(40), assets: [] };
  const capture = (implementation, result) => ({ schema: 1, model: 'sample', inputs_sha256: 'b'.repeat(64), manifest_sha256: 'c'.repeat(64),
    producer: { implementation, version: '1.0.0', source_sha256: 'd'.repeat(64) }, results: { [input.id]: { result,
      ...(implementation === 'reference' ? { reference_input: structuredClone(input.request), metadata_origin: 'authored-contract', transformations: [] } : {}) } } });
  return { schema: 1, manifest: { schema: 1, models: [model] }, manifest_sha256: 'c'.repeat(64), models: [{ model,
    inputs: { schema: 1, model: 'sample', cases: [structuredClone(input)] }, inputs_sha256: 'b'.repeat(64),
    reference: capture('reference', prompt()), dynamo: capture('dynamo', prompt()) }] };
};
const catalog = { metadata: { renderer: { dynamo_version: '10.3.2' }, tabs: [{ id: 'tab-renderer', suite: 'renderer', label: 'Renderer' }] } };

test('renderer comparisons preserve prompt bytes and metadata boundaries', () => {
  assert.equal(rendererOutcome(prompt(), prompt(), input), 'pass');
  assert.equal(rendererOutcome(prompt('Hello\n'), prompt('Hello'), input), 'diverge');
  assert.equal(rendererOutcome(prompt('é'), prompt('ê'), input), 'diverge');
  assert.equal(rendererOutcome(prompt(), prompt('Hello', { media_message_order: [0] }), input), 'pass', 'implicit identity media order is equivalent');
  assert.equal(rendererOutcome(prompt(), prompt('Hello', { segments: [] }), input), 'diverge', 'segment absence remains distinct from empty segments');
  const segment = { text: '<|open|>', allow_special: true };
  const segmented = prompt(segment.text, { segments: [segment], pending_segments: 1 });
  assert.equal(rendererOutcome(segmented, { ...segmented, segments: [{ ...segment, allow_special: false }] }, input), 'diverge');
  assert.equal(rendererOutcome(segmented, { ...segmented, pending_segments: 0 }, input), 'diverge');
  const twoMessages = { ...input, request: { messages: [...input.request.messages, { role: 'tool', content: 'Result' }] } };
  assert.equal(rendererOutcome(prompt(), prompt('Hello', { media_message_order: [1, 0] }), twoMessages), 'diverge');
});

test('renderer errors distinguish expected rejection, missing evidence, and unexpected failures', () => {
  const invalid = { kind: 'error', error_type: 'invalid_request', message: 'Python rejection' };
  assert.equal(rendererOutcome(invalid, { ...invalid, message: 'Rust rejection' }, input), 'expected');
  const broken = { kind: 'error', error_type: 'render_error', message: 'Unexpected failure' };
  assert.equal(rendererOutcome(broken, broken, input), 'error');
  assert.equal(rendererOutcome(prompt(), invalid, input), 'error');
  assert.equal(rendererOutcome(invalid, prompt(), input), 'diverge');
  assert.equal(rendererOutcome(prompt(), undefined, input), 'missing');
  assert.equal(rendererOutcome(undefined, prompt(), input), 'uncompared');
  assert.equal(rendererOutcome(prompt(), { kind: 'unsupported', reason: 'Absent capability' }, input), 'unsupported');
});

test('first prompt difference uses UTF-8 byte offsets and preserves escaped context', () => {
  assert.equal(firstPromptDifference('same', 'same'), null);
  assert.equal(firstPromptDifference(null, 'same'), null);
  assert.deepEqual(firstPromptDifference('é', 'ê'), { offset: 1, reference_byte: 169, dynamo_byte: 170, reference_context: '"é"', dynamo_context: '"ê"' });
  assert.equal(firstPromptDifference('ab', 'a').offset, 1);
  assert.equal(firstPromptDifference('ab', 'a').dynamo_byte, null);
  assert.equal(firstPromptDifference('a\n', 'a ').reference_context, '"a\\n"');
});

test('renderer admission rejects broken bindings, duplicate cases, and invalid metadata', () => {
  assert.equal(validateRendererSuite(makeSuite()).models.length, 1);
  for (const mutate of [
    suite => { suite.models[0].reference.inputs_sha256 = 'e'.repeat(64); },
    suite => { suite.models[0].dynamo.manifest_sha256 = 'e'.repeat(64); },
    suite => { suite.models[0].inputs.cases.push(structuredClone(input)); },
    suite => { suite.models.push(structuredClone(suite.models[0])); },
    suite => { suite.models = []; },
    suite => { suite.models[0].reference.results.extra = { result: prompt() }; },
    suite => { suite.models[0].reference.results[input.id].reference_input = null; },
    suite => { suite.models[0].reference.results[input.id].metadata_origin = null; },
    suite => { suite.models[0].reference.results[input.id].transformations = [{ name: 'join', authority: null }]; },
    suite => { suite.models[0].dynamo.results[input.id].result.pending_segments = 1; },
    suite => { suite.models[0].dynamo.results[input.id].result.segments = [{ text: 'other', allow_special: true }]; },
    suite => { suite.models[0].dynamo.results[input.id].result.media_message_order = [1]; },
  ]) {
    const suite = makeSuite(); mutate(suite);
    assert.throws(() => validateRendererSuite(suite), /Renderer conformance/);
  }
});

test('renderer panel shares the matrix contract without parser leak classification', async () => {
  const suite = makeSuite();
  suite.models[0].reference.results[input.id].result = prompt('<think>reason</think>');
  suite.models[0].dynamo.results[input.id].result = prompt('<think>different</think>');
  const panel = await buildTab(catalog, { renderer: async () => suite }, 'tab-renderer');
  assert.equal(panel.rows.length, 1);
  assert.equal(panel.columns.length, 1);
  assert.deepEqual(panel.candidates.map(item => item.key), ['renderer_reference', 'renderer_dynamo']);
  const cell = panel.rows[0].cells[input.id];
  assert.equal(selectedCaptureOutcome(cell, 'renderer_dynamo'), 'diverge');
  assert.equal(selectedCaptureOutcome(cell, 'renderer_reference'), 'pass');
  assert.equal(cell.tooltip.renderer, true);
  assert.equal(cell.tooltip.first_difference.offset, 7);
  assert.equal(panel.comparison_slot_count, 2);
  assert.deepEqual(panel.comparison_slot_engines, ['reference', 'dynamo']);
  assert.equal(panel.status_labels.error, 'Unexpected rendering errors');
  assert.deepEqual(tabCandidates(catalog, catalog.metadata.tabs[0]).map(item => item.control.label), ['Model reference', 'Dynamo renderer 10.3.2']);
  assert.equal(tabPresentation(catalog.metadata.tabs[0], catalog, []).case_prefix, 'RENDERER.');
});

test('missing captures remain missing evidence with all authored columns and model rows', async () => {
  const suite = makeSuite(); delete suite.models[0].dynamo.results[input.id];
  const panel = await rendererPanel(catalog, { renderer: async () => suite });
  assert.equal(panel.columns.length, 1);
  assert.equal(panel.rows.length, 1);
  assert.equal(panel.rows[0].cells[input.id].applicable, true);
  assert.equal(selectedCaptureOutcome(panel.rows[0].cells[input.id], 'renderer_dynamo'), 'missing');
});

test('renderer transport validates envelope and supports parser-only catalogs', async () => {
  const suite = makeSuite();
  const bytes = new TextEncoder().encode(JSON.stringify({ schema: 1, kind: 'renderer-suite', renderer: suite }));
  const descriptor = { kind: 'renderer-suite', format: 'typed-json-v1', compression: 'none',
    path: 'conformance/renderer.json', sha256: await sha256(bytes), size: bytes.length };
  const reader = new FixtureResources({ sources: { renderer: [descriptor] } }, { read: async () => bytes }, {});
  assert.deepEqual(await reader.renderer(), suite);
  await assert.rejects(new FixtureResources({ sources: {} }, {}, {}).renderer(), /missing/);
  const corrupt = new TextEncoder().encode(JSON.stringify({ schema: 1, kind: 'parser', renderer: suite }));
  await assert.rejects(new FixtureResources({ sources: { renderer: [descriptor] } }, { read: async () => corrupt }, {}).renderer(), /envelope/);
});

test('renderer popup escapes prompt text and displays provenance without parser event charts', async () => {
  const window = { localStorage: { getItem: () => null, setItem() {} } };
  const document = { cookie: '', documentElement: { setAttribute() {} }, querySelectorAll: () => [], addEventListener() {} };
  const context = vm.createContext({ window, document });
  vm.runInContext(await readFile(new URL('../utils/src/assets/conformance_view.js', import.meta.url), 'utf8'), context);
  const suite = makeSuite();
  const hostile = '<script>alert("prompt")</script>\n';
  suite.models[0].reference.results[input.id].result = prompt(hostile);
  suite.models[0].dynamo.results[input.id].result = prompt('different');
  const panel = await rendererPanel(catalog, { renderer: async () => suite });
  const markup = window.__renderConformanceTooltip(panel.rows[0].cells[input.id].tooltip);
  assert.ok(markup.includes('&lt;script&gt;alert(&quot;prompt&quot;)&lt;/script&gt;'));
  assert.ok(!markup.includes('<script>'));
  for (const label of ['Renderer request', 'Rendering settings', 'Dynamo renderer', 'First different UTF-8 byte', 'Producer provenance', 'Metadata expectation authority']) assert.ok(markup.includes(label), label);
  assert.ok(!markup.includes('ttip-chunks'));
});

test('the pinned renderer corpus admits every authored case across all five models', async () => {
  const root = new URL('../fixtures-renderer/', import.meta.url);
  const bytes = await readFile(new URL('manifest.json', root));
  const manifest = JSON.parse(bytes);
  const models = await Promise.all(manifest.models.map(async model => {
    const inputs = await readFile(new URL(`families/${model.id}/inputs.json`, root));
    return { model, inputs: JSON.parse(inputs), inputs_sha256: await sha256(inputs),
      reference: JSON.parse(await readFile(new URL(`families/${model.id}/reference.json`, root))),
      dynamo: JSON.parse(await readFile(new URL(`families/${model.id}/dynamo.json`, root))) };
  }));
  const suite = { schema: 1, manifest, manifest_sha256: await sha256(bytes), models };
  const panel = await rendererPanel(catalog, { renderer: async () => suite });
  assert.equal(panel.rows.length, 5);
  assert.equal(panel.stats.real, models.reduce((sum, model) => sum + model.inputs.cases.length, 0));
  for (const row of panel.rows) for (const cell of Object.values(row.cells).filter(cell => cell.applicable)) {
    assert.notEqual(selectedCaptureOutcome(cell, 'renderer_dynamo'), 'missing', cell.case_id);
    assert.notEqual(selectedCaptureOutcome(cell, 'renderer_dynamo'), 'leak', 'rendered markers must not be parsed as leaked output');
    assert.equal(cell.tooltip.renderer, true);
    assert.ok(cell.fixture_href.endsWith('/inputs.json'));
  }
  assert.equal(panel.columns.length, new Set(models.flatMap(model => model.inputs.cases.map(input => input.id))).size);
});

test('history discovery preserves the renderer suite while upgrading parser selections', async () => {
  const base = new URL('http://localhost/frontend-crates/');
  const recent = { schema: 1, revision: 'a'.repeat(64), metadata: { ...catalog.metadata },
    sources: { archives: [], unified: [], renderer: [{ kind: 'renderer-suite', format: 'typed-json-v1', source_fingerprint: 'c'.repeat(64) }] },
    capture_generation: { mode: 'latest', current: [] } };
  const history = structuredClone(recent); history.revision = 'b'.repeat(64); history.capture_generation.mode = 'history';
  const publication = { schema: 1, catalog_revision: recent.revision, generation_mode: 'history', resolved_catalog_revision: history.revision };
  const fetchFor = value => async () => new Response(gzipSync(new TextEncoder().encode(JSON.stringify(value))));
  assert.deepEqual(await discoverHistory(base, recent, { publication, fetchImpl: fetchFor(history) }), history);
  for (const mutate of [
    value => { delete value.sources.renderer; },
    value => { value.metadata.tabs = []; },
    value => { value.metadata.renderer.dynamo_version = '0.0.0'; },
    value => { value.sources.renderer[0].source_fingerprint = 'd'.repeat(64); },
  ]) {
    const changed = structuredClone(history); mutate(changed);
    await assert.rejects(discoverHistory(base, recent, { publication, fetchImpl: fetchFor(changed) }), /preserve.*renderer/);
  }
});
