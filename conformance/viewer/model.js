// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { parseJson, sameValue } from './values.js';
import { isMapping, getField, hasField, setField, deleteField, mergeMappings } from './mappings.js';
export { sameValue } from './values.js';
export const unifiedCaptureVersionPattern = /^\d+(?:\.\d+){2,}(?:\.post\d+|-[0-9A-Za-z.-]+)?$/;

export function provenanceReleaseVersion(version) {
  if (typeof version !== 'string') throw new Error('invalid provenance version');
  const release = version.replace(/\+source\.[0-9a-f]{64}$/, '');
  if (!unifiedCaptureVersionPattern.test(release)) throw new Error('invalid provenance version');
  return release;
}

export function compareVersions(left, right) {
  if ([left, right].some((value) => !String(value).includes('-') && /\.patch\d+$/.test(value))) return compareLegacyVersions(left, right);
  const key = (value) => {
    const match = String(value).match(/^(\d+(?:\.\d+)*)(?:\.post(\d+)|-([0-9A-Za-z.-]+)|((?:a|b|rc|alpha|beta|dev)\d*))?$/);
    if (!match) throw new Error(`invalid capture version: ${value}`);
    const prerelease = match[3] || match[4] || null;
    return [match[1].split('.').map(BigInt), match[2] !== undefined ? 2 : prerelease === null ? 1 : 0, match[2] !== undefined ? BigInt(match[2]) : prerelease];
  };
  const a = key(left), b = key(right);
  for (let i = 0; i < Math.max(a[0].length, b[0].length); i++) {
    const x = a[0][i] ?? 0n, y = b[0][i] ?? 0n;
    if (x !== y) return x < y ? -1 : 1;
  }
  for (const index of [1, 2]) if (a[index] !== b[index]) return a[index] < b[index] ? -1 : 1;
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

export function selectedUnifiedSnapshot(records, checkpointVersion, selectedVersion) {
  const result = structuredClone(records);
  if (checkpointVersion !== selectedVersion) for (const record of result.values()) {
    const document = getField(getField(record, 'value'), 'document');
    deleteField(getField(document, 'record_metadata'), 'verified_recapture');
    if (!hasField(document, 'inherited_from') && !getField(record, 'suppress_inherited_from')) setField(document, 'inherited_from', checkpointVersion);
  }
  return result;
}

export function resolveUnified(captures, version) {
  return resolveUnifiedSeries(captures, [version]).next().value[1];
}

export function* resolveUnifiedSeries(captures, versions) {
  const ordered = [...captures].sort((left, right) => compareVersions(left.version, right.version));
  const stateByCase = new Map();
  const originByCase = new Map();
  const metadataOriginByCase = new Map();
  let index = 0;
  for (const version of [...versions].sort(compareVersions)) {
    while (index < ordered.length && compareVersions(ordered[index].version, version) <= 0) {
      const capture = ordered[index];
      if (Object.hasOwn(capture.document, 'parent') && capture.document.parent !== `${capture.implementation}-${ordered[index - 1]?.version}`) throw new Error(`capture parent is not the preceding checkpoint: ${capture.path}`);
      for (const [caseId, change] of Object.entries(capture.document.changes || {})) {
        if (change?.absent === true && Object.keys(change).length === 1) {
          stateByCase.delete(caseId);
          originByCase.delete(caseId);
          metadataOriginByCase.delete(caseId);
          continue;
        }
        const prior = stateByCase.get(caseId);
        if (!prior && ['case_key', 'stimulus', 'observation'].some(key => !Object.hasOwn(change, key))) throw new Error(`first capture change must provide a complete record: ${caseId} (${capture.path})`);
        const next = { ...structuredClone(prior || {}), ...structuredClone(change) };
        if (!prior || Object.hasOwn(change, 'observation')) {
          originByCase.set(caseId, capture);
          delete next._document_overrides;
        }
        if (prior?._record_metadata !== undefined) next._record_metadata = prior._record_metadata;
        if (prior?._document_overrides?.parser_path !== undefined) {
          next._document_overrides = Object.hasOwn(change, 'observation')
            ? { parser_path: prior._document_overrides.parser_path }
            : structuredClone(prior._document_overrides);
        }
        stateByCase.set(caseId, next);
      }
      for (const [caseId, metadata] of Object.entries(capture.document.metadata_changes || {})) {
        if (!stateByCase.has(caseId)) throw new Error(`metadata change has no observation: ${caseId} (${capture.path})`);
        const record = stateByCase.get(caseId);
        if (metadata instanceof Map ? metadata.size : Object.keys(metadata || {}).length) {
          record._record_metadata = structuredClone(metadata);
          metadataOriginByCase.set(caseId, capture.version);
        } else {
          delete record._record_metadata;
          metadataOriginByCase.delete(caseId);
        }
      }
      for (const [caseId, override] of Object.entries(capture.document.document_overrides || {})) {
        if (!stateByCase.has(caseId)) throw new Error(`document override has no observation: ${caseId} (${capture.path})`);
        const record = stateByCase.get(caseId);
        record._document_overrides = { ...(record._document_overrides || {}), ...structuredClone(override || {}) };
      }
      index += 1;
    }
    yield [version, new Map([...stateByCase].map(([caseId, record]) => {
      const { _record_metadata, _document_overrides, ...observation } = structuredClone(record);
      const __origin = originByCase.get(caseId);
      const document = mergeMappings(structuredClone(__origin.document.document), { captured_with: { [__origin.implementation]: __origin.version } });
      const origin = __origin.document.provenance?.origin;
      if (__origin.document.provenance?.status === 'captured' && origin) setField(document, 'capture_origin', structuredClone(origin));
      if (version !== __origin.version) setField(document, 'inherited_from', __origin.version);
      const provenanceOverride = _document_overrides?.capture_provenance;
      if (provenanceOverride) {
        for (const [key, value] of Object.entries(provenanceOverride)) {
          if (value === null) deleteField(document, key);
          else setField(document, key, structuredClone(value));
        }
      }
      for (const [key, value] of Object.entries(_document_overrides || {})) {
        if (key === 'parser_path' && value === null) deleteField(document, key);
        else if (key !== 'capture_provenance') setField(document, key, structuredClone(value));
      }
      if (_record_metadata instanceof Map ? _record_metadata.size : _record_metadata && Object.keys(_record_metadata).length) {
        const metadata = structuredClone(_record_metadata);
        if (metadataOriginByCase.get(caseId) !== version) deleteField(metadata, 'verified_recapture');
        setField(document, 'record_metadata', metadata);
      }
      observation.document = document;
      return [caseId, { value: observation, member: __origin.path, sourceBytes: __origin.bytes }];
    }))];
  }
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
