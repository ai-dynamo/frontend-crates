// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { sha256 } from './data.js';
import { canonicalJson, isMapping, hasField, mergeMappings } from './values.js';

export const isNormalized = value => value?.schema_version === 3 || value?.schema_version === 3n;
const clone = value => structuredClone(value);
function fields(value, required, optional = []) {
  if (!isMapping(value) || value instanceof Map
      || required.some(key => !Object.hasOwn(value, key))
      || Object.keys(value).some(key => !required.includes(key) && !optional.includes(key))) throw new Error('invalid normalized YAML fields');
}
function reference(table, identity) {
  if (typeof identity !== 'string' || !Object.hasOwn(table, identity)) throw new Error(`unknown normalized reference: ${identity}`);
  return clone(table[identity]);
}

export async function expandRequests(document) {
  fields(document, ['schema_version', 'family', 'request_defaults', 'tools', 'requests', 'cases']);
  if (!isNormalized(document)) throw new Error('invalid normalized authored schema');
  fields(document.request_defaults, ['init', 'finish_reason']);
  fields(document.tools, Object.keys(document.tools));
  fields(document.requests, Object.keys(document.requests));
  const requests = {};
  for (const [identity, stored] of Object.entries(document.requests)) {
    fields(stored, ['input', 'tools'], ['init', 'finish_reason', 'chunks', 'chunk_sizes', 'finish_step']);
    const request = { ...clone(document.request_defaults), ...clone(stored), tools: reference(document.tools, stored.tools) };
    let chunks;
    if (Object.hasOwn(stored, 'chunk_sizes')) {
      if (Object.hasOwn(stored, 'chunks') || !Array.isArray(stored.chunk_sizes)
          || Object.hasOwn(stored, 'finish_step') && typeof stored.finish_step !== 'boolean') throw new Error('ambiguous normalized chunk schedule');
      const characters = [...stored.input];
      let offset = 0;
      chunks = stored.chunk_sizes.map(size => {
        const valid = typeof size === 'bigint' || typeof document.schema_version === 'number' && Number.isInteger(size);
        if (!valid || size < 0 || size > characters.length - offset) throw new Error('invalid normalized chunk size');
        const text = characters.slice(offset, offset + Number(size)).join('');
        offset += Number(size);
        return text;
      });
      if (offset !== characters.length) throw new Error('normalized chunk sizes do not cover input');
      if (stored.finish_step === true) chunks.push('‹finish›');
    } else {
      if (Object.hasOwn(stored, 'finish_step')) throw new Error('finish step requires a chunk schedule');
      chunks = Object.hasOwn(stored, 'chunks') ? stored.chunks : [stored.input];
    }
    if (!Array.isArray(chunks)) throw new Error('invalid normalized chunk schedule');
    delete request.chunk_sizes; delete request.finish_step;
    request.chunks = chunks.map(chunk => typeof chunk === 'string' ? { delta_text: chunk } : clone(chunk));
    if (await sha256(new TextEncoder().encode(canonicalJson(request))) !== identity) throw new Error('normalized request digest mismatch');
    requests[identity] = request;
  }
  return requests;
}

export async function expandFamily(document) {
  if (!isNormalized(document)) return document;
  const requests = await expandRequests(document);
  fields(document.cases, Object.keys(document.cases));
  const cases = Object.create(null);
  for (const [identity, stored] of Object.entries(document.cases)) {
    fields(stored, ['description', 'display_id', 'request', 'golden'], ['lifecycle', 'scenario', 'policy', 'historical_ids']);
    cases[identity] = { lifecycle: 'active', scenario: identity, policy: [], historical_ids: [], ...clone(stored),
      request: stored.request === null ? null : reference(requests, stored.request),
      golden: stored.golden === null ? null : { assembled: clone(stored.golden) } };
  }
  return { input_document: { family: document.family }, golden_document: { family: document.family }, cases };
}

export async function expandSnapshot(document, authored) {
  if (!isNormalized(document)) return document;
  fields(document, ['schema_version', 'provenance', 'document', 'origins', 'producers', 'contexts', 'record_defaults', 'records'],
    ['capture_selection', 'execution_proof', 'historical_evidence']);
  const requests = await expandRequests(authored);
  for (const key of ['origins', 'producers', 'contexts', 'records']) fields(document[key], Object.keys(document[key]));
  const provenance = clone(document.provenance);
  if (Object.hasOwn(provenance, 'origin')) provenance.origin = reference(document.origins, provenance.origin);
  fields(document.record_defaults, ['member', 'producer', 'context']);
  const records = Object.create(null);
  for (const [identity, stored] of Object.entries(document.records)) {
    fields(stored, ['case_key'], ['request', 'stimulus', 'output', 'error', 'unavailable', 'producer', 'context', 'member', 'suppress_inherited_from']);
    if (Object.hasOwn(stored, 'request') === Object.hasOwn(stored, 'stimulus')) throw new Error('normalized record needs exactly one stimulus');
    const states = ['output', 'error', 'unavailable'].filter(key => Object.hasOwn(stored, key));
    if (states.length !== 1) throw new Error('normalized record needs exactly one observation');
    const producer = reference(document.producers, Object.hasOwn(stored, 'producer') ? stored.producer : document.record_defaults.producer);
    fields(producer, [], ['captured_with', 'capture_origin']);
    if (Object.hasOwn(producer, 'capture_origin') && producer.capture_origin !== null) producer.capture_origin = reference(document.origins, producer.capture_origin);
    const context = reference(document.contexts, Object.hasOwn(stored, 'context') ? stored.context : document.record_defaults.context);
    if (!isMapping(context)) throw new Error('normalized context must be a mapping');
    if (hasField(context, 'captured_with') || hasField(context, 'capture_origin')) throw new Error('normalized context repeats producer fields');
    const stimulus = Object.hasOwn(stored, 'request')
      ? { inline: reference(requests, stored.request), semantic_sha256: stored.request } : clone(stored.stimulus);
    const state = states[0];
    records[identity] = { member: Object.hasOwn(stored, 'member') ? stored.member : document.record_defaults.member, value: {
      case_key: stored.case_key, document: mergeMappings(context, producer), stimulus,
      observation: { [state === 'output' ? 'value' : state]: clone(stored[state]) } } };
    if (Object.hasOwn(stored, 'suppress_inherited_from')) records[identity].suppress_inherited_from = stored.suppress_inherited_from;
  }
  return { ...Object.fromEntries(Object.entries(document).filter(([key]) => !['origins', 'producers', 'contexts', 'record_defaults', 'records'].includes(key))),
    schema_version: typeof document.schema_version === 'bigint' ? 2n : 2, provenance, records };
}
