// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { normalizeImplementations } from './comparison.js';
import { parseJson } from './values.js';
import { isMapping, getField, hasField, setField, mappingEntries, mappingKeyIdentity } from './mappings.js';

const own = (mapping, key, fallback = undefined) => Object.hasOwn(mapping, key) ? mapping[key] : fallback;

export function deriveStreamExpected(record, metadata, selected = null, { controls = false } = {}) {
  const implementations = metadata.implementations;
  const unavailable = normalizeImplementations(getField(record, 'unavailable'), implementations);
  const exception = normalizeImplementations(getField(record, 'exception'), implementations);
  const chunks = (getField(record, 'chunks') || []).filter(isMapping).map((chunk) => {
    const text = getField(chunk, 'normal_text');
    const normal = normalizeImplementations(text, implementations);
    for (const implementation of implementations) if (hasField(text, implementation.key)) normal[implementation.key] = getField(text, implementation.key);
    return { expected: normalizeImplementations(getField(chunk, 'expected'), implementations), normal };
  });
  const expected = {};
  for (const implementation of implementations) {
    const key = implementation.key;
    if (selected && !selected.includes(key)) continue;
    if (Object.hasOwn(unavailable, key)) { expected[key] = controls ? { unavailable: true, parserError: unavailable[key] === 'parser-error' } : { unavailable: unavailable[key] }; continue; }
    if (Object.hasOwn(exception, key)) { expected[key] = { exception: exception[key] }; continue; }
    if (!chunks.some((chunk) => Object.hasOwn(chunk.expected, key) || Object.hasOwn(chunk.normal, key))) {
      expected[key] = { unavailable: controls ? true : key === 'vllm_rust' ? metadata.recording.rustUnavailable : 'No recorded capture for this case.' };
      continue;
    }
    if (controls) { expected[key] = {}; continue; }
    const calls = new Map();
    let normalText = '';
    for (const chunk of chunks) {
      for (const delta of own(chunk.expected, key, []) || []) {
        if (!hasField(delta, 'index')) throw new Error(`stream delta has no index: ${key}`);
        const index = mappingKeyIdentity(getField(delta, 'index'));
        if (!calls.has(index)) calls.set(index, { name: '', arguments: '' });
        const call = calls.get(index);
        for (const field of ['name', 'arguments']) {
          const fragment = getField(delta, field);
          if (fragment === undefined || fragment === null) continue;
          if (typeof fragment !== 'string') throw new Error(`legacy stream ${field} fragment must be text: ${key}`);
          call[field] += fragment;
        }
      }
      const text = own(chunk.normal, key);
      if (text) {
        if (typeof text !== 'string') throw new Error(`legacy stream normal_text must be text: ${key}`);
        normalText += text;
      }
    }
    for (const call of calls.values()) {
      if (!call.arguments) call.arguments = {};
      else {
        try { call.arguments = parseJson(call.arguments); }
        catch (error) { if (!(error instanceof SyntaxError)) throw error; }
      }
    }
    const block = { calls: [...calls.values()], normal_text: normalText };
    if (!key.startsWith('dynamo_')) block.explanation = metadata.recording.streamExplanation.replace('{label}', implementation.label);
    expected[key] = block;
  }
  return expected;
}

export function legacyCases(documents, mode, metadata, selected = null, options = {}) {
  const cases = new Map();
  const labels = {};
  const capturedWith = {};
  const prefix = `TOOLCALLING.${mode}`;
  const batchExpected = new Map();
  if (mode === 'streamv1') for (const [path, source] of documents) {
    const document = source.value;
    if (!path.split('/').at(-1).startsWith('TOOLCALLING.batch') || getField(document, 'mode') !== 'batch') continue;
    for (const [caseId, record] of mappingEntries(getField(document, 'cases'))) {
      if (isMapping(getField(record, 'expected'))) batchExpected.set(`${getField(document, 'family')}\0${caseId.replace('TOOLCALLING.batch.', '')}`, normalizeImplementations(getField(record, 'expected'), metadata.implementations));
    }
  }
  for (const [path, source] of [...documents].sort(([left], [right]) => left < right ? -1 : left > right ? 1 : 0)) {
    if (!path.split('/').at(-1).startsWith(prefix)) continue;
    const document = source.value;
    if (getField(document, 'mode') !== mode) continue;
    const family = getField(document, 'family');
    if (hasField(document, 'model_label') && !Object.hasOwn(labels, family)) setField(labels, family, getField(document, 'model_label'));
    const provenance = normalizeImplementations(getField(document, 'captured_with'), metadata.implementations);
    for (const [key, version] of Object.entries(provenance)) if (!Object.hasOwn(capturedWith, key)) setField(capturedWith, key, String(version));
    if (!isMapping(getField(document, 'cases'))) throw new Error(`legacy cases must be a mapping with string IDs: ${path}`);
    for (const [recordedId, value] of mappingEntries(getField(document, 'cases'))) {
      if (typeof recordedId !== 'string') throw new Error(`legacy case ID must be text: ${path}`);
      const caseId = own(metadata.legacy_aliases, recordedId, recordedId);
      const sub = caseId.replace(`${prefix}.`, '');
      const identity = `${family}\0${sub}`;
      if (cases.has(identity)) throw new Error(`duplicate case after ID normalization: ${family}/${caseId}`);
      const record = structuredClone(value);
      if (!isMapping(record)) throw new Error(`legacy case must be a mapping: ${path}:${caseId}`);
      setField(record, '__family', family);
      setField(record, '__case_id', caseId);
      if (sub.includes('.') && !metadata.legacy_null_groups.includes(sub.split('.')[0])) {
        const [group, ...suffix] = sub.split('.');
        const suite = mode === 'streamv1' ? 'toolcalling.stream' : 'toolcalling.batch';
        const groups = metadata.legacy_taxonomy.suites[suite].groups;
        const description = own(groups, group)?.cases?.[suffix.join('.')]?.desc;
        if (description !== undefined) setField(record, 'description', description);
      }
      if (mode === 'streamv1') {
        setField(record, 'expected', deriveStreamExpected(record, metadata, selected, options));
        setField(record, 'batch_expected', batchExpected.get(identity) || {});
      } else if (isMapping(getField(record, 'expected'))) setField(record, 'expected', normalizeImplementations(getField(record, 'expected'), metadata.implementations));
      cases.set(identity, { family, sub, record, source });
    }
  }
  const subs = new Set([...cases.values()].map((entry) => entry.sub));
  const families = new Set([...cases.values()].map((entry) => entry.family));
  for (const [parent, children] of Object.entries(metadata.legacy_split_parents)) {
    if (!children.some((child) => subs.has(child))) continue;
    for (const family of families) {
      const parentKey = `${family}\0${parent}`;
      const source = cases.get(parentKey);
      if (!source) continue;
      for (const child of children) {
        const key = `${family}\0${child}`;
        if (cases.has(key)) continue;
        const record = structuredClone(source.record);
        setField(record, '__case_id', `TOOLCALLING.batch.${child}`);
        setField(record, '__synthetic_from_case_id', getField(source.record, '__case_id'));
        cases.set(key, { ...source, sub: child, record });
      }
      cases.delete(parentKey);
    }
  }
  return { cases, labels, capturedWith };
}
