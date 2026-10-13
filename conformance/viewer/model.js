// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { parseJson, sameValue } from './values.js';
import { isMapping, getField, hasField, setField, deleteField, mergeMappings } from './mappings.js';
export { sameValue } from './values.js';

export function compareVersions(left, right) {
  if ([left, right].some((value) => !String(value).includes('-') && /(?:\.?post\d+|\.patch\d+)$/.test(value))) return compareLegacyVersions(left, right);
  const key = (value) => {
    const match = String(value).match(/^(\d+(?:\.\d+)*)(?:-([0-9A-Za-z.-]+)|((?:a|b|rc|alpha|beta|dev)\d*))?$/);
    if (!match) throw new Error(`invalid capture version: ${value}`);
    return [match[1].split('.').map(BigInt), match[2] || match[3] || null];
  };
  const a = key(left), b = key(right);
  for (let i = 0; i < Math.max(a[0].length, b[0].length); i++) {
    const x = a[0][i] ?? 0n, y = b[0][i] ?? 0n;
    if (x !== y) return x < y ? -1 : 1;
  }
  if (a[1] !== b[1]) {
    if (a[1] === null) return 1;
    if (b[1] === null) return -1;
    return a[1] < b[1] ? -1 : 1;
  }
  return 0;
}
export function compareLegacyVersions(left, right) {
  const key = (value) => {
    const patch = String(value).match(/^(.+)\.patch(\d+)$/);
    const match = (patch?.[1] || String(value)).match(/^(\d+(?:\.\d+)*)(?:[.-]?post(\d+))?/);
    return [match ? match[1].split('.').map(BigInt) : [], BigInt(match?.[2] || 0), BigInt(patch?.[2] || 0)];
  };
  const a = key(left), b = key(right);
  for (let index = 0; index < Math.min(a[0].length, b[0].length); index++) {
    if (a[0][index] !== b[0][index]) return a[0][index] < b[0][index] ? -1 : 1;
  }
  if (a[0].length !== b[0].length) return a[0].length < b[0].length ? -1 : 1;
  for (const index of [1, 2]) if (a[index] !== b[index]) return a[index] < b[index] ? -1 : 1;
  return 0;
}

export function resolveUnified(captures, version) {
  const ordered = captures.filter((entry) => compareVersions(entry.version, version) <= 0)
    .sort((left, right) => compareVersions(left.version, right.version));
  const stateByCase = new Map();
  for (const capture of ordered) {
    for (const [caseId, change] of Object.entries(capture.document.changes || {})) {
      if (change?.absent === true && Object.keys(change).length === 1) {
        stateByCase.delete(caseId);
        continue;
      }
      const prior = stateByCase.get(caseId);
      const next = structuredClone(change);
      next.__origin = capture;
      if (prior?._record_metadata !== undefined) next._record_metadata = prior._record_metadata;
      if (prior?._document_overrides?.parser_path !== undefined) next._document_overrides = { parser_path: prior._document_overrides.parser_path };
      stateByCase.set(caseId, next);
    }
    for (const [caseId, metadata] of Object.entries(capture.document.metadata_changes || {})) {
      if (!stateByCase.has(caseId)) throw new Error(`metadata change has no observation: ${caseId} (${capture.path})`);
      const record = stateByCase.get(caseId);
      if (metadata instanceof Map ? metadata.size : Object.keys(metadata || {}).length) record._record_metadata = structuredClone(metadata);
      else delete record._record_metadata;
    }
    for (const [caseId, override] of Object.entries(capture.document.document_overrides || {})) {
      if (!stateByCase.has(caseId)) throw new Error(`document override has no observation: ${caseId} (${capture.path})`);
      const record = stateByCase.get(caseId);
      record._document_overrides = { ...(record._document_overrides || {}), ...structuredClone(override || {}) };
    }
  }
  return new Map([...stateByCase].map(([caseId, record]) => {
    const { __origin, _record_metadata, _document_overrides, ...observation } = record;
    const document = mergeMappings(__origin.document.document, { captured_with: { [__origin.implementation]: __origin.version } });
    const origin = __origin.document.provenance?.origin;
    if (__origin.document.provenance?.status === 'captured' && origin) setField(document, 'capture_origin', structuredClone(origin));
    if (ordered.at(-1)?.version !== __origin.version) setField(document, 'inherited_from', __origin.version);
    const provenanceOverride = _document_overrides?.capture_provenance;
    if (provenanceOverride) {
      for (const [key, value] of Object.entries(provenanceOverride)) {
        if (value === null) deleteField(document, key);
        else setField(document, key, structuredClone(value));
      }
    }
    for (const [key, value] of Object.entries(_document_overrides || {})) {
      if (key !== 'capture_provenance') setField(document, key, structuredClone(value));
    }
    if (_record_metadata instanceof Map ? _record_metadata.size : _record_metadata && Object.keys(_record_metadata).length) setField(document, 'record_metadata', _record_metadata);
    observation.document = document;
    return [caseId, { value: observation, member: __origin.path, sourceBytes: __origin.bytes }];
  }));
}

export function assembleStream(chunks, recordedAssembly = null) {
  const events = [];
  let currentTool = null;
  for (const deltas of chunks || []) {
    for (const delta of deltas || []) {
      if (!isMapping(delta)) throw new Error('Unified stream delta must be a mapping');
      const kind = getField(delta, 'kind');
      const name = getField(delta, 'name');
      const args = getField(delta, 'arguments');
      const complete = getField(delta, 'complete', true);
      if (kind === 'reasoning' || kind === 'text') {
        currentTool = null;
        const text = getField(delta, 'text') || '';
        if (typeof text !== 'string') throw new Error('Unified text delta must contain text');
        if (events.at(-1)?.kind === kind) events.at(-1).text += text;
        else events.push({ kind, text });
      } else if (kind === 'tool_call') {
        if (name) {
          events.push({ kind: 'tool_call', name, arguments: args || '', complete });
          currentTool = events.length - 1;
        } else if (currentTool !== null && args) {
          const before = events[currentTool].arguments;
          const part = args;
          events[currentTool].arguments = typeof before === 'string' && typeof part === 'string' ? before + part : part;
        }
        if (currentTool !== null && complete) events[currentTool].complete = true;
      }
    }
  }
  const completed = events.filter((event) => event.kind !== 'tool_call' || event.complete);
  const malformed = new Set();
  for (const event of completed) {
    if (event.kind !== 'tool_call') continue;
    delete event.complete;
    if (isMapping(event.arguments) || Array.isArray(event.arguments)) continue;
    if (typeof event.arguments !== 'string') throw new Error('Unified argument fragment must be text, a mapping, or a list');
    if (!event.arguments.trim()) event.arguments = {};
    else {
      try { event.arguments = parseJson(event.arguments); }
      catch (error) {
        if (!(error instanceof SyntaxError)) throw error;
        malformed.add(event);
      }
    }
  }
  if (malformed.size && recordedAssembly) {
    const calls = completed.filter((event) => event.kind === 'tool_call');
    const recorded = recordedAssembly.filter((event) => getField(event, 'kind') === 'tool_call');
    if (calls.length === recorded.length && calls.every((event, index) => event.name === getField(recorded[index], 'name') && (sameValue(event.arguments, getField(recorded[index], 'arguments')) || (malformed.has(event) && sameValue(getField(recorded[index], 'arguments'), {}))))) {
      calls.forEach((event, index) => { if (malformed.has(event)) event.arguments = getField(recorded[index], 'arguments'); });
    }
  }
  return completed;
}

export function unifiedChunkDeltas(value) {
  const chunks = getField(value, 'chunks');
  if (!Array.isArray(chunks)) return [];
  return chunks.map((chunk) => Array.isArray(chunk) ? chunk : getField(chunk, 'expected', []));
}

export function unifiedEvents(value, { stream = true, recordedFallback = true } = {}) {
  if (!value || typeof value !== 'object') return [];
  const chunks = getField(value, 'chunks');
  const assembled = getField(value, 'assembled');
  if (stream && Array.isArray(chunks)) return assembleStream(unifiedChunkDeltas(value), recordedFallback ? assembled : null);
  if (Array.isArray(assembled)) return assembled;
  return [];
}

export { classifyUnified } from './comparison.js';

export function unifiedResult(record) {
  if (!record) return { state: 'missing' };
  const value = record.value;
  const observation = value?.observation;
  if (observation?.unavailable) return { state: 'unavailable', reason: getField(observation.unavailable, 'detail') ?? getField(observation.unavailable, 'message') ?? getField(observation.unavailable, 'code') };
  if (observation?.error) return { state: 'error', reason: getField(observation.error, 'detail') ?? getField(observation.error, 'message') ?? getField(observation.error, 'code') };
  if (observation?.state && observation.state !== 'value') return { state: observation.state, reason: observation.code || '' };
  const payload = observation?.value || value?.value || value;
  return { state: 'value', payload, events: unifiedEvents(payload) };
}
