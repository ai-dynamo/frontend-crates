// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { sameValue } from './values.js';

export const rendererStatusLabels = {
  pass: 'Matches model reference', documented: 'Documented differences', diverge: 'Disagreements',
  error: 'Unexpected rendering errors', leak: 'Markup leaks', missing: 'Missing evidence',
  unsupported: 'Unsupported cases', blocked: 'Blocked sources', expected: 'Expected rejections',
  uncompared: 'Reference unavailable',
};

const mapping = value => value && typeof value === 'object' && !Array.isArray(value);
const hash = value => typeof value === 'string' && /^[a-f0-9]{64}$/.test(value);
const own = (object, key) => Object.hasOwn(object, key);
const fail = message => { throw new Error(`Renderer conformance: ${message}`); };
const title = value => value.replace(/[_-]+/g, ' ').replace(/\b\w/g, char => char.toUpperCase());

function validateResult(result, request, where) {
  if (!mapping(result)) fail(`${where}: missing result`);
  if (result.kind === 'unsupported') {
    if (typeof result.reason !== 'string' || !result.reason) fail(`${where}: missing unsupported reason`);
    return;
  }
  if (result.kind === 'error') {
    if (!['invalid_request', 'render_error'].includes(result.error_type) || typeof result.message !== 'string') fail(`${where}: invalid error outcome`);
    return;
  }
  if (result.kind !== 'prompt' || typeof result.text !== 'string') fail(`${where}: invalid prompt`);
  if (result.segments !== null && (!Array.isArray(result.segments) || result.segments.some(segment =>
    !mapping(segment) || typeof segment.text !== 'string' || typeof segment.allow_special !== 'boolean'))) fail(`${where}: invalid segments`);
  if (result.segments !== null && result.segments.map(segment => segment.text).join('') !== result.text) fail(`${where}: segment text differs from prompt`);
  if (!Number.isSafeInteger(result.pending_segments) || result.pending_segments < 0
      || result.pending_segments > (result.segments?.length || 0)) fail(`${where}: invalid pending segments`);
  const order = result.media_message_order;
  if (order !== null && (!Array.isArray(order) || order.length !== request.messages.length
      || new Set(order).size !== order.length || order.some(index => !Number.isSafeInteger(index) || index < 0 || index >= order.length))) fail(`${where}: invalid media order`);
}

export function validateRendererSuite(suite) {
  if (!mapping(suite) || suite.schema !== 1 || !mapping(suite.manifest) || suite.manifest.schema !== 1
      || !hash(suite.manifest_sha256) || !Array.isArray(suite.manifest.models) || !Array.isArray(suite.models)) fail('invalid suite');
  const manifest = new Map();
  for (const model of suite.manifest.models) {
    if (!mapping(model) || typeof model.id !== 'string' || !/^[a-z0-9_]+$/.test(model.id)
        || typeof model.label !== 'string' || !model.label || manifest.has(model.id)) fail('invalid or duplicate model');
    manifest.set(model.id, model);
  }
  const seen = new Set();
  for (const bundle of suite.models) {
    const model = bundle.model;
    if (!model || !sameValue(manifest.get(model.id), model) || seen.has(model.id)) fail('model manifest mismatch');
    seen.add(model.id);
    const inputs = bundle.inputs;
    if (!mapping(inputs) || inputs.schema !== 1 || inputs.model !== model.id || !Array.isArray(inputs.cases)
        || !inputs.cases.length || !hash(bundle.inputs_sha256)) fail(`${model.id}: invalid authored inputs`);
    const ids = new Set();
    for (const input of inputs.cases) {
      if (!mapping(input) || typeof input.id !== 'string' || !/^RENDERER\.[A-Za-z0-9_.-]+$/.test(input.id)
          || ids.has(input.id) || typeof input.group !== 'string' || !input.group
          || typeof input.description !== 'string' || !mapping(input.request)
          || !Array.isArray(input.request.messages) || !mapping(input.settings)) fail(`${model.id}: invalid or duplicate case`);
      ids.add(input.id);
    }
    for (const implementation of ['reference', 'dynamo']) {
      const capture = bundle[implementation];
      // Missing observations stay visible; publication requires complete recordings.
      if (capture == null) continue;
      if (!mapping(capture) || capture.schema !== 1 || capture.model !== model.id
          || capture.inputs_sha256 !== bundle.inputs_sha256 || capture.manifest_sha256 !== suite.manifest_sha256
          || !mapping(capture.producer) || capture.producer.implementation !== implementation
          || typeof capture.producer.version !== 'string' || !capture.producer.version
          || !hash(capture.producer.source_sha256) || !mapping(capture.results)) fail(`${model.id}/${implementation}: capture binding mismatch`);
      for (const [id, record] of Object.entries(capture.results)) {
        if (!ids.has(id) || !mapping(record)) fail(`${model.id}/${implementation}: unknown case ${id}`);
        if (implementation === 'reference' && (!mapping(record.reference_input)
            || typeof record.metadata_origin !== 'string' || !record.metadata_origin.trim()
            || !Array.isArray(record.transformations) || record.transformations.some(change => !mapping(change)
              || typeof change.name !== 'string' || !change.name.trim() || typeof change.authority !== 'string' || !change.authority.trim()))) fail(`${model.id}/${id}: reference input or authority missing`);
        validateResult(record.result, inputs.cases.find(input => input.id === id).request, `${model.id}/${implementation}/${id}`);
      }
    }
  }
  if (seen.size !== manifest.size) fail('missing model row');
  return suite;
}

export function rendererCandidates(catalog) {
  const version = catalog.metadata.renderer?.dynamo_version || '0.0.0';
  return [
    ['renderer_reference', 'reference', 'Model reference', '1.0.0'],
    ['renderer_dynamo', 'dynamo', `Dynamo renderer ${version}`, version],
  ].map(([key, engine, label, release]) => ({
    source: { id: `${key}@renderer@${release}`, implementation: key, mode: 'renderer', version: release, language: engine === 'dynamo' ? 'Rust' : 'Python' },
    control: { key, impl: engine, label, label_html: label, version: release, parse_mode: 'renderer', language: engine === 'dynamo' ? 'Rust' : 'Python', default_bucket: 'B' },
  }));
}

function comparable(result, input) {
  if (result?.kind === 'error') return { kind: 'error', error_type: result.error_type };
  if (result?.kind !== 'prompt') return result;
  return { ...result, media_message_order: result.media_message_order ?? input.request.messages.map((_, index) => index) };
}

export function rendererOutcome(reference, observed, input) {
  if (!observed) return 'missing';
  if (observed.kind === 'unsupported') return 'unsupported';
  if (observed.kind === 'error' && observed.error_type === 'render_error') return 'error';
  if (!reference || reference.kind === 'unsupported') return 'uncompared';
  if (reference.kind === 'error' && reference.error_type === 'render_error') return 'uncompared';
  if (sameValue(comparable(reference, input), comparable(observed, input))) return observed.kind === 'error' ? 'expected' : 'pass';
  return observed.kind === 'error' && reference.kind !== 'error' ? 'error' : 'diverge';
}

export function firstPromptDifference(reference, observed) {
  if (typeof reference !== 'string' || typeof observed !== 'string' || reference === observed) return null;
  const encoder = new TextEncoder(), decoder = new TextDecoder();
  const expected = encoder.encode(reference), actual = encoder.encode(observed);
  let offset = 0;
  while (offset < Math.min(expected.length, actual.length) && expected[offset] === actual[offset]) offset++;
  const context = bytes => JSON.stringify(decoder.decode(bytes.slice(Math.max(0, offset - 24), offset + 25)));
  return { offset, reference_byte: expected[offset] ?? null, dynamo_byte: actual[offset] ?? null,
    reference_context: context(expected), dynamo_context: context(actual) };
}

export async function rendererPanel(catalog, reader, { selected = null } = {}) {
  const suite = validateRendererSuite(await reader.renderer());
  const descriptors = rendererCandidates(catalog);
  const chosen = selected ?? descriptors.map(item => item.control.key);
  if (chosen.some(key => !descriptors.some(item => item.control.key === key))) fail('unknown selected capture');
  const cases = new Map(), groupLabels = new Map();
  for (const bundle of suite.models) for (const input of bundle.inputs.cases) {
    const prior = cases.get(input.id);
    if (prior && prior.group !== input.group) fail(`${input.id}: conflicting case category`);
    if (!prior) cases.set(input.id, input);
    groupLabels.set(input.group, title(input.group));
  }
  const columns = [...cases.values()].map(input => ({ sub: input.id, case_id: input.id,
    label: input.id.replace(/^RENDERER\./, ''), desc: input.description,
    group_key: `renderer_${input.group}`, group: input.group, band: 'case-band-0' }));
  const order = ['conversations', 'reasoning', 'tools', 'controls', 'images', ...suite.manifest.models.map(model => model.id)];
  const groups = [...groupLabels].sort(([left], [right]) => {
    const rank = key => order.includes(key) ? order.indexOf(key) : order.length;
    return rank(left) - rank(right) || left.localeCompare(right);
  }).map(([key, label], index) => {
    const model = suite.manifest.models.find(model => model.id === key);
    const group = { key: `renderer_${key}`, label: model?.label || label,
      span: columns.filter(column => column.group === key).length, band: `case-band-${index % 2}` };
    columns.filter(column => column.group === key).forEach((column, number) => {
      column.band = group.band; column.display_number = number + 1;
      column.compact_prefix = model ? key : String(index + 1);
    });
    return group;
  });
  const groupedColumns = groups.flatMap(group => columns.filter(column => column.group_key === group.key));
  const rows = suite.models.map(bundle => {
    const authored = new Map(bundle.inputs.cases.map(input => [input.id, input]));
    const cells = Object.fromEntries(groupedColumns.map(column => {
      const input = authored.get(column.sub);
      if (!input) return [column.sub, { kind: 'cell', case_id: column.case_id, family: bundle.model.id, sub: column.sub,
        status: 'na', applicable: false, cmp: {}, capture_outcomes: { renderer_reference: 'unsupported', renderer_dynamo: 'unsupported' },
        tooltip: { renderer: true, head: column.case_id, description: 'This model does not expose this model-specific contract.', model: bundle.model, applicable: false } }];
      const reference = bundle.reference?.results[input.id], dynamo = bundle.dynamo?.results[input.id];
      const expected = reference?.result, actual = dynamo?.result;
      const outcome = rendererOutcome(expected, actual, input);
      const referenceOutcome = !expected ? 'missing' : expected.kind === 'unsupported' ? 'unsupported'
        : expected.kind === 'error' ? expected.error_type === 'render_error' ? 'error' : 'expected' : 'pass';
      const diff = firstPromptDifference(expected?.text, actual?.text);
      const expectedComparable = comparable(expected, input), actualComparable = comparable(actual, input);
      const metadataDifferences = ['segments', 'pending_segments', 'media_message_order'].filter(field => expected?.kind === 'prompt' && actual?.kind === 'prompt'
        && !sameValue(expectedComparable[field], actualComparable[field]));
      const values = [expectedComparable, actualComparable];
      const equal = sameValue(values[0], values[1]);
      const cmp = { golden: { sig: 0 }, renderer_reference: { sig: 0 }, renderer_dynamo: { sig: equal ? 0 : 1 } };
      const notes = [reference?.documented_difference, dynamo?.documented_difference, reference?.note, dynamo?.note].filter(value => typeof value === 'string');
      return [column.sub, { kind: 'cell', case_id: column.case_id, family: bundle.model.id, sub: column.sub,
        col_group: column.group_key, band: column.band, status: ['error', 'diverge'].includes(outcome) ? 'problem' : 'ok', applicable: true,
        cmp, red_on_diff: true, capture_outcomes: { golden: referenceOutcome, renderer_reference: referenceOutcome, renderer_dynamo: notes.length && outcome === 'diverge' ? 'documented' : outcome },
        fixture_href: `conformance/fixtures-renderer/families/${bundle.model.id}/inputs.json`,
        tooltip: { renderer: true, head: column.case_id, description: input.description, model: bundle.model,
          input: { kind: 'renderer', request: input.request }, settings: input.settings,
          reference, dynamo, metadata_expectation: input.metadata_expectation, metadata_origin: input.metadata_origin,
          reference_producer: bundle.reference?.producer, dynamo_producer: bundle.dynamo?.producer,
          first_difference: diff, metadata_differences: metadataDifferences, notes,
          candidates: [
            { key: 'renderer_reference', label: 'Model reference', block: expected || { unavailable: 'Missing reference capture' } },
            { key: 'renderer_dynamo', label: 'Dynamo renderer', block: actual || { unavailable: 'Missing Dynamo capture' } },
          ] } }];
    }));
    return { family: bundle.model.id, model_label: bundle.model.label, model_label_html: bundle.model.label.replace(/[&<>"']/g, value => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[value]), cells };
  });
  const real = rows.reduce((count, row) => count + Object.values(row.cells).filter(cell => cell.applicable).length, 0);
  return { id: 'tab-renderer', kind: 'renderer', mode: 'renderer', no_parser_col: true,
    columns: groupedColumns, column_groups: groups, rows,
    stats: { families: rows.length, sub_cases: columns.length, slots: rows.length * columns.length, real, na: rows.length * columns.length - real },
    glossary: groups.map(group => ({ label: group.label, rows: groupedColumns.filter(column => column.group_key === group.key).map(column => [column.label, column.desc]) })),
    candidates: descriptors.map(item => item.control), status_labels: rendererStatusLabels };
}
