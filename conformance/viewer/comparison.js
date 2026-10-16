// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { parseJson, sameValue } from './values.js';
import { isMapping, mappingEntries, mappingKeyText, getField, hasField, setField, deleteField, mergeMappings } from './mappings.js';

const has = hasField;
const parserError = (value) => /parser not captured|parsing failed|parse error|panicked|exception|traceback/i.test(value);

export function declaredLeakTokens(markers) {
  const tokens = new Set();
  for (const declaration of Object.values(markers)) {
    for (const token of [...(declaration.pairs || []).flat(), ...(declaration.singletons || [])]) {
      const chars = [...token];
      tokens.add(token);
      for (let length = 7; length <= chars.length; length++) tokens.add(chars.slice(0, length).join(''));
    }
    for (const token of declaration.leak || []) tokens.add(token);
  }
  if (!tokens.size) throw new Error('parser metadata declares no family markers');
  return [...tokens];
}
export function blockLeaks(block, tokens) {
  const text = getField(block, 'normal_text');
  return typeof text === 'string' && tokens.some((token) => text.includes(token));
}

export function explanation(block) {
  return getField(block, 'explanation') ?? getField(block, 'reason') ?? null;
}

export function legacyOutcome(block, tokens, controlContext = null) {
  const leak = blockLeaks(block, tokens);
  const unavailable = !isMapping(block) || has(block, 'unavailable');
  const unavailableReason = getField(block, 'unavailable');
  const errorUnavailable = controlContext?.unavailableParserError === true || typeof unavailableReason === 'string' && parserError(unavailableReason);
  const exception = has(block, 'exception');
  const error = has(block, 'error');
  return {
    state: unavailable ? 'unavailable' : exception || error ? 'error' : 'value',
    status: unavailable ? errorUnavailable ? 'problem' : 'na' : exception || error || leak ? 'problem' : 'ok',
    leak,
    errorKind: errorUnavailable ? 'parser_error' : unavailable && isMapping(block) ? 'unavailable' : error ? isMapping(getField(block, 'error')) ? 'parser_error' : 'expected_error' : null,
    reason: explanation(block),
  };
}

export function canonicalToolOutput(block) {
  if (!isMapping(block) || has(block, 'unavailable') || has(block, 'error') || (!has(block, 'calls') && !has(block, 'normal_text'))) return null;
  return { calls: getField(block, 'calls') || [], normal_text: getField(block, 'normal_text') || '' };
}

function decodedCall(call) {
  if (!isMapping(call) || typeof getField(call, 'arguments') !== 'string') return call;
  try { const result = structuredClone(call); setField(result, 'arguments', parseJson(getField(call, 'arguments'))); return result; }
  catch (error) {
    if (!(error instanceof SyntaxError)) throw error;
    return call;
  }
}

export function legacyComparisonValue(block) {
  if (!isMapping(block) || has(block, 'unavailable')) return ['na'];
  if (has(block, 'exception')) return ['exception', getField(block, 'exception')];
  if (has(block, 'error')) return ['error', getField(block, 'error')];
  return ['value', { calls: (getField(block, 'calls') || []).map(decodedCall), normal_text: getField(block, 'normal_text') || '' }];
}

// IDs are local equivalence groups assigned after full typed-value comparison.
// There is no truncated digest capable of merging two unequal outputs.
export function comparisonEntries(blocks, tokens) {
  return equivalenceEntries(Object.fromEntries(Object.entries(blocks).map(([key, block]) => {
    const value = legacyComparisonValue(block);
    return [key, { value, leak: Number(blockLeaks(block, tokens)), na: Number(value[0] === 'na'), err: Number(value[0] === 'exception') }];
  })));
}

export function equivalenceEntries(rows) {
  const values = [];
  const entries = {};
  for (const [key, { value, leak = 0, na = 0, err = 0 }] of Object.entries(rows)) {
    let index = values.findIndex((prior) => sameValue(prior, value));
    if (index < 0) { index = values.length; values.push(value); }
    setField(entries, key, { sig: index, leak, na, err });
  }
  return entries;
}

export function equivalenceGroups(entries) {
  const groups = new Map();
  for (const [key, entry] of Object.entries(entries)) {
    if (!groups.has(entry.sig)) groups.set(entry.sig, []);
    groups.get(entry.sig).push(key);
  }
  return Object.fromEntries(Object.entries(entries).map(([key, entry]) => [key, { ...entry, sig: [...groups.get(entry.sig)].sort() }]));
}

export function implementationKeys(implementation, implementations) {
  const owner = implementations.find(item => item.key === implementation || item.legacy_key === implementation);
  return owner ? [owner.key, ...(owner.legacy_key ? [owner.legacy_key] : [])] : [implementation];
}

export function implementationValue(mapping, implementation, implementations, fallback = undefined) {
  const canonical = implementationKeys(implementation, implementations)[0];
  return getField(normalizeImplementations(mapping, implementations), canonical, fallback);
}

export function replaceImplementation(mapping, implementation, value, implementations) {
  const keys = implementationKeys(implementation, implementations);
  for (const key of keys) deleteField(mapping, key);
  setField(mapping, keys[0], value);
}

export function normalizeImplementations(mapping, implementations) {
  if (!isMapping(mapping)) return {};
  const aliases = new Map(implementations.flatMap(item => implementationKeys(item.key, implementations).map(key => [key, item.key])));
  const result = {};
  for (const [key, value] of mappingEntries(mapping)) {
    const text = typeof key === 'string' ? key : mappingKeyText(key);
    const canonical = aliases.get(text) || text;
    if (!Object.hasOwn(result, canonical)) setField(result, canonical, value);
  }
  return result;
}

export function comparisonFacts(record, implementations, baseline, tokens, rustUnavailable = null) {
  const keys = implementations.map((item) => item.key);
  if (!isMapping(record) || !has(record, 'expected')) return keys.map((impl) => ({ impl, status: 'na', present: false, agrees: null, intentional: false, reason: null, leak: false, error_kind: null }));
  const expected = normalizeImplementations(getField(record, 'expected'), implementations);
  const statusBlocks = { ...expected };
  if (Object.keys(expected).length && !Object.hasOwn(expected, 'vllm_rust') && rustUnavailable) expected.vllm_rust = { unavailable: rustUnavailable };
  const base = canonicalToolOutput(getField(expected, baseline));
  return keys.map((impl) => {
    const block = getField(expected, impl);
    const value = canonicalToolOutput(block);
    const outcome = legacyOutcome(block, tokens);
    return {
      impl, status: legacyOutcome(getField(statusBlocks, impl), tokens).status, present: value !== null,
      agrees: value === null || base === null ? impl === baseline ? true : null : sameValue(value, base),
      intentional: outcome.reason !== null, reason: outcome.reason, leak: outcome.leak, error_kind: outcome.errorKind,
    };
  });
}

export function streamCallsConsistent(record, implementation) {
  const stream = getField(getField(record, 'expected'), implementation);
  const batch = getField(getField(record, 'batch_expected'), implementation);
  if (!isMapping(stream) || !isMapping(batch) || (!hasField(batch, 'calls') && !hasField(batch, 'normal_text'))) return null;
  const calls = (block) => (getField(block, 'calls') || []).map((call) => [getField(call, 'name', ''), getField(call, 'arguments', {})]);
  return sameValue(calls(stream), calls(batch));
}

export function legacyOverviewStatus(record, implementation, tokens, stream = false) {
  const outcome = legacyOutcome(getField(getField(record, 'expected'), implementation), tokens);
  return stream && outcome.status === 'ok' && streamCallsConsistent(record, implementation) === false ? 'problem' : outcome.status;
}

// Preserve the historical statistical buckets without shipping its private
// concatenated marker alphabet. Cells, facts and statistics share typed outputs.
export function legacyStatisticsCategory(record, implementations, baseline, tokens, stream = false) {
  if (!record) return 'missing';
  const expected = normalizeImplementations(getField(record, 'expected'), implementations);
  const base = getField(expected, baseline);
  if (!isMapping(base) || hasField(base, 'unavailable')) {
    return stream && legacyOutcome(base, tokens).status === 'problem' ? 'documented' : 'na';
  }
  if (stream) {
    const output = canonicalToolOutput(base);
    const differing = output === null ? [] : implementations.filter((item) => item.key !== baseline && canonicalToolOutput(getField(expected, item.key)) !== null && !sameValue(output, canonicalToolOutput(getField(expected, item.key))));
    if (blockLeaks(base, tokens)) return 'documented';
    if (streamCallsConsistent(record, baseline) === false) return differing.length ? 'documented' : 'dynamo_only';
    if (differing.length === 1 && differing[0].key === 'dynamo_v1') return 'dynamo_only';
    return differing.length ? 'documented' : 'parity';
  }
  const peers = implementations.filter((item) => !item.key.startsWith('dynamo_')).map((item) => getField(expected, item.key));
  const unavailable = peers.every((peer) => !isMapping(peer) || hasField(peer, 'unavailable'));
  if (blockLeaks(base, tokens) && explanation(base)) return 'documented';
  if (unavailable) return 'dynamo_only';
  if (peers.some((peer) => isMapping(peer) && !hasField(peer, 'unavailable') && hasField(peer, 'error'))) return 'errors';
  const output = canonicalToolOutput(base) || { calls: getField(base, 'calls') || [], normal_text: getField(base, 'normal_text') || '' };
  const differing = peers.filter((peer) => isMapping(peer) && !hasField(peer, 'unavailable') && canonicalToolOutput(peer) !== null && !sameValue(output, canonicalToolOutput(peer)));
  if (differing.some((peer) => explanation(peer) === null)) return 'research';
  return differing.length ? 'documented' : 'parity';
}

export function classifyUnified(family, golden, observed, leakMarkers) {
  if (!Array.isArray(golden) || !Array.isArray(observed) || ![...golden, ...observed].every(isMapping)) throw new Error('Unified events must be lists of mappings');
  if (sameValue(golden, observed)) return 'MATCH';
  const markers = [...leakMarkers.common, ...(getField(leakMarkers.families, family) || [])];
  if (observed.some((event) => ['text', 'reasoning'].includes(getField(event, 'kind')) && markers.some((marker) => String(getField(event, 'text') || '').includes(marker)))) return 'LEAK';
  const count = (events, kind) => events.filter((event) => getField(event, 'kind') === kind).length;
  if (count(observed, 'reasoning') < count(golden, 'reasoning')) return 'MERGE';
  const calls = (events) => events.filter((event) => getField(event, 'kind') === 'tool_call');
  const expectedCalls = calls(golden), actualCalls = calls(observed);
  if (expectedCalls.length === actualCalls.length && expectedCalls.every((call, index) => getField(call, 'name') === getField(actualCalls[index], 'name')) && expectedCalls.some((call, index) => !sameValue(getField(call, 'arguments'), getField(actualCalls[index], 'arguments')))) return 'ARG_MISMATCH';
  const text = (events, kind) => events.filter((event) => getField(event, 'kind') === kind).map((event) => getField(event, 'text') || '').join('');
  if (text(golden, 'reasoning') === text(observed, 'reasoning') && text(golden, 'text') === text(observed, 'text')) return 'ORDER';
  return 'LOSS';
}

const pythonWhitespace = /^[\t\n\v\f\r \u001c-\u001f\u0085\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]*$/;

export function canonicalReasoning(block) {
  if (!isMapping(block)) throw new Error('reasoning output must be a mapping');
  const normalize = (value) => value == null || typeof value === 'string' && pythonWhitespace.test(value) ? null : value;
  const result = mergeMappings(block, { normal_text: normalize(getField(block, 'normal_text')), reasoning_text: normalize(getField(block, 'reasoning_text')) });
  deleteField(result, 'reason'); deleteField(result, 'explanation');
  return result;
}

export function reasoningLeakReason(block, family, metadata) {
  if (!isMapping(block)) return null;
  const config = metadata.reasoning_display;
  const pattern = new RegExp(config.markup_patterns[family] || config.default_markup_pattern);
  for (const field of ['reasoning_text', 'normal_text']) {
    const text = getField(block, field);
    const handoff = family === 'gpt_oss' && field === 'normal_text' && typeof text === 'string' && (text.includes('<|channel|>commentary to=functions.') || text.includes('<|channel|>analysis to=functions.')) && text.includes('<|call|>');
    if (typeof text === 'string' && pattern.test(text) && !handoff) return String(explanation(block) || 'Dynamo leaks reasoning markup or final-answer text.');
  }
  return null;
}

export function reasoningOverviewStatus(record, family, implementation, metadata) {
  if (!record) return 'na';
  if (!hasField(record, 'expected')) {
    const stub = mappingEntries(record).every(([key]) => ['description', 'reason', 'explanation', 'ref', 'spec_ref'].includes(key)) && explanation(record) !== null;
    return stub && Object.hasOwn(metadata.reasoning_display.peer_families[implementation] || {}, family) ? 'problem' : 'na';
  }
  const block = implementationValue(getField(record, 'expected'), implementation, metadata.implementations);
  if (!isMapping(block) || hasField(block, 'unavailable')) return 'na';
  return hasField(block, 'error') || reasoningLeakReason(block, family, metadata) ? 'problem' : 'ok';
}

export function reasoningComparisonEntries(record, family, metadata) {
  const expected = hasField(record, 'expected') ? normalizeImplementations(getField(record, 'expected'), metadata.implementations) : null;
  if (!isMapping(expected)) return null;
  const rows = {};
  for (const impl of metadata.selection_policy.reasoning.implementation_order) {
    if (!hasField(expected, impl)) continue;
    const block = getField(expected, impl);
    const unavailable = block === null || hasField(block, 'unavailable');
    rows[impl] = { value: unavailable ? ['absent'] : hasField(block, 'error') ? ['error', getField(block, 'error')] : ['value', canonicalReasoning(block)], leak: Number(Boolean(reasoningLeakReason(block, family, metadata))), na: Number(unavailable) };
  }
  return Object.keys(rows).length ? equivalenceEntries(rows) : null;
}

export function reasoningFacts(record, family, metadata, baseline = 'dynamo_v1', selected = null) {
  const expected = normalizeImplementations(getField(record, 'expected'), metadata.implementations);
  const base = getField(expected, baseline);
  const implementations = selected ?? metadata.selection_policy.reasoning.implementation_order;
  const concrete = (block) => isMapping(block) && !hasField(block, 'unavailable') && !hasField(block, 'error');
  return implementations.map((impl) => {
    const block = getField(expected, impl);
    const reason = explanation(block);
    return { impl, status: reasoningOverviewStatus(record, family, impl, metadata), present: concrete(block), agrees: impl === baseline ? true : !concrete(block) || !concrete(base) ? null : sameValue(canonicalReasoning(block), canonicalReasoning(base)),
      intentional: reason !== null, reason, leak: Boolean(reasoningLeakReason(block, family, metadata)), error_kind: hasField(block, 'error') ? 'expected_error' : hasField(block, 'unavailable') ? 'unavailable' : null };
  });
}

export function reasoningStatisticsCategory(record, family, metadata, baseline = 'dynamo_v1', selected = null) {
  if (!record) return 'missing';
  if (baseline === null) return 'na';
  if (!hasField(record, 'expected')) {
    const stub = mappingEntries(record).every(([key]) => ['description', 'reason', 'explanation', 'ref', 'spec_ref'].includes(key)) && explanation(record) !== null;
    return stub ? 'na' : 'research';
  }
  const expected = hasField(record, 'expected') ? normalizeImplementations(getField(record, 'expected'), metadata.implementations) : null;
  const base = getField(expected, baseline);
  if (selected !== null && (!isMapping(base) || hasField(base, 'unavailable'))) return 'na';
  if (selected !== null && hasField(base, 'error')) return 'errors';
  const leakReason = reasoningLeakReason(base, family, metadata);
  const leaked = Boolean((baseline === 'dynamo_v1' && getField(record, 'dynamo_leak')) || leakReason);
  const peers = (selected ?? metadata.selection_policy.reasoning.implementation_order).filter((key) => key !== baseline).map((key) => getField(expected, key));
  if (peers.every((block) => hasField(block, 'unavailable'))) return leaked ? 'documented' : 'dynamo_only';
  if (peers.some((block) => !hasField(block, 'unavailable') && hasField(block, 'error'))) return 'errors';
  const differing = peers.filter((block) => !hasField(block, 'unavailable') && !sameValue(canonicalReasoning(block), canonicalReasoning(base)));
  if (leaked) return !leakReason || !differing.length ? 'research' : 'documented';
  if (differing.some((block) => !explanation(block))) return 'research';
  return differing.length ? 'documented' : 'parity';
}

export function referenceRecord(record, descriptor, blocks) {
  if (!descriptor) return record;
  const key = descriptor.control.key;
  // Panels validate the selected candidate's loading before evaluating records.
  // A loaded sparse capture may still have no observation for this fixture.
  const result = structuredClone(record);
  const expected = getField(result, 'expected') || {};
  setField(expected, descriptor.source.implementation, blocks[key] ?? null);
  setField(result, 'expected', expected);
  return result;
}
