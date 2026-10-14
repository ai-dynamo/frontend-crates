// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { sha256 } from './data.js';
import { canonicalJson, sameValue } from './values.js';
import { getField, hasField, deleteField, mergeMappings } from './mappings.js';

function mapping(value, where) {
  if (!value || typeof value !== 'object' || (!(value instanceof Map) && ![Object.prototype, null].includes(Object.getPrototypeOf(value)))) throw new Error(`${where} must be a mapping`);
  return value;
}
function keys(value, required, allowed, where) {
  mapping(value, where);
  if (value instanceof Map) throw new Error(`${where} must have string field names`);
  for (const key of required) if (!Object.hasOwn(value, key)) throw new Error(`${where} is missing ${key}`);
  for (const key of Object.keys(value)) if (!allowed.includes(key)) throw new Error(`${where} has unknown field ${key}`);
}
function stringMapping(value, where) {
  mapping(value, where);
  if (value instanceof Map) throw new Error(`${where} must have string keys`);
  return value;
}
function string(value, where) {
  if (typeof value !== 'string') throw new Error(`${where} must be a string`);
}
function component(value, where) {
  string(value, where);
  if (!value || ['.', '..'].includes(value) || /[/\\\0]/.test(value)) throw new Error(`${where} must be one safe path component`);
}
function request(value, where) {
  const fields = ['input', 'init', 'finish_reason', 'tools', 'chunks'];
  keys(value, fields, fields, where);
  string(value.input, `${where}.input`);
  mapping(value.init, `${where}.init`);
  string(value.finish_reason, `${where}.finish_reason`);
  if (!Array.isArray(value.tools)) throw new Error(`${where}.tools must be a list`);
  if (!Array.isArray(value.chunks)) throw new Error(`${where}.chunks must be a list`);
  value.chunks.forEach((chunk) => mapping(chunk, `${where}.chunks`));
}

export function canonicalUnifiedLabel(label, family, metadata) {
  if (!metadata) return label;
  let key = label.replace(/^UNIFIED\./, '');
  if (key.endsWith(`.${family}`)) key = key.slice(0, -(family.length + 1));
  if (Object.hasOwn(metadata.canonical, key)) return `UNIFIED.${metadata.canonical[key]}`;
  const familyAliases = Object.hasOwn(metadata.families, family) ? metadata.families[family] : {};
  key = Object.hasOwn(familyAliases, key) ? familyAliases[key] : key;
  const dotted = key.match(/^31\.([a-x])$/);
  if (dotted) key = `31-${dotted[1].charCodeAt(0) - 'a'.charCodeAt(0) + 1}`;
  const owners = Object.hasOwn(metadata.owners, key) ? metadata.owners[key] : [];
  if (owners.length > 1) throw new Error(`conflicting Unified alias ownership: ${label}`);
  return owners.length ? `UNIFIED.${metadata.canonical[owners[0]]}` : `UNIFIED.${key}`;
}

export function validateUnifiedInputs(document, family, path, aliasMetadata = null) {
  const fields = ['input_document', 'golden_document', 'cases'];
  keys(document, fields, fields, path);
  for (const field of ['input_document', 'golden_document']) {
    mapping(document[field], `${path}.${field}`);
    if (document[field].family !== family) throw new Error(`${path}.${field} family differs from directory`);
  }
  stringMapping(document.cases, `${path}.cases`);
  const aliases = new Set();
  const canonicalOwners = new Map();
  for (const [caseId, value] of Object.entries(document.cases)) {
    const where = `${path}:${caseId}`;
    const fields = ['lifecycle', 'scenario', 'description', 'policy', 'display_id', 'historical_ids', 'request', 'golden'];
    keys(value, fields, fields, where);
    if (!['active', 'retired'].includes(value.lifecycle)) throw new Error(`${where} has invalid lifecycle`);
    if (value.description !== null) string(value.description, `${where}.description`);
    if (value.lifecycle === 'active' && !value.description) throw new Error(`${where} needs a description`);
    if (!Array.isArray(value.policy) || value.policy.some((tag) => typeof tag !== 'string')) throw new Error(`${where}.policy must be a list of strings`);
    if (!Array.isArray(value.historical_ids)) throw new Error(`${where}.historical_ids must be a list`);
    for (const alias of [...(value.display_id === null ? [] : [value.display_id]), ...value.historical_ids]) {
      component(alias, `${where}.display_id/historical_ids`);
      if (aliases.has(alias)) throw new Error(`${where} has duplicate display/historical ID: ${alias}`);
      aliases.add(alias);
      const canonical = canonicalUnifiedLabel(alias, family, aliasMetadata);
      if (canonicalOwners.has(canonical) && canonicalOwners.get(canonical) !== caseId) throw new Error(`${where} has duplicate canonical ID: ${alias}`);
      canonicalOwners.set(canonical, caseId);
    }
    if (value.lifecycle === 'active') {
      component(value.display_id, `${where}.display_id`);
      string(value.scenario, `${where}.scenario`);
      request(value.request, `${where}.request`);
      mapping(value.golden, `${where}.golden`);
      if (!hasField(value.golden, 'assembled') || !Array.isArray(getField(value.golden, 'assembled'))) throw new Error(`${where}.golden needs an assembled event list`);
      if (!Array.isArray(getField(value.golden, 'assembled')) && !Array.isArray(getField(value.golden, 'chunks'))) throw new Error(`${where} has no GOLDEN events`);
    } else if (value.request !== null) request(value.request, `${where}.request`);
  }
  return document;
}

function origin(value, where, version = null) {
  keys(value, ['crate_version', 'source_sha256'], ['crate_version', 'source_sha256', 'git_commit'], where);
  if (typeof value.crate_version !== 'string' || !/^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$/.test(value.crate_version) || (version !== null && value.crate_version !== version)) throw new Error(`${where} has invalid crate_version`);
  if (!/^[0-9a-f]{64}$/.test(value.source_sha256 || '')) throw new Error(`${where} has invalid source hash`);
  if (Object.hasOwn(value, 'git_commit') && !/^[0-9a-f]{40}$/.test(value.git_commit)) throw new Error(`${where} has invalid git commit`);
}

export async function validateUnifiedCapture(document, inputDocument, { path, implementation, version }) {
  const fields = ['provenance', 'document', 'changes', 'metadata_changes', 'document_overrides'];
  keys(document, fields, fields, path);
  const provenance = mapping(document.provenance, `${path}.provenance`);
  if (provenance.status === 'captured') {
    keys(provenance, ['status', 'origin'], ['status', 'origin'], `${path}.provenance`);
    origin(provenance.origin, `${path}.provenance.origin`, version);
  } else if (provenance.status === 'legacy') {
    keys(provenance, ['status', 'captured_with'], ['status', 'captured_with'], `${path}.provenance`);
    if (!sameValue(provenance.captured_with, { [implementation]: version })) throw new Error(`${path} has inconsistent capture identity`);
  } else throw new Error(`${path} has invalid capture provenance status`);
  mapping(document.document, `${path}.document`);
  if (['capture_origin', 'captured_with'].some((field) => hasField(document.document, field))) throw new Error(`${path} repeats derived provenance`);
  for (const field of ['changes', 'metadata_changes', 'document_overrides']) stringMapping(document[field], `${path}.${field}`);
  for (const [caseId, change] of Object.entries(document.changes)) {
    const where = `${path}:${caseId}`;
    if (!Object.hasOwn(inputDocument.cases, caseId)) throw new Error(`${where} is an unknown case`);
    const input = inputDocument.cases[caseId];
    if (sameValue(change, { absent: true })) continue;
    keys(change, [], ['case_key', 'stimulus', 'observation'], where);
    if (!Object.keys(change).length) throw new Error(`${where} has empty change`);
    if (Object.hasOwn(change, 'case_key')) component(change.case_key, `${where}.case_key`);
    if (Object.hasOwn(change, 'stimulus')) {
    const stimulus = mapping(change.stimulus, `${where}.stimulus`);
    const states = ['ref', 'inline', 'partial', 'error', 'unavailable'].filter((state) => Object.hasOwn(stimulus, state));
    if (states.length !== 1) throw new Error(`${where} must contain exactly one stimulus state`);
    keys(stimulus, states, [...states, 'semantic_sha256'], `${where}.stimulus`);
    if (states[0] === 'ref') {
      if (stimulus.ref !== 'current' || input.request === null) throw new Error(`${where} has invalid current stimulus reference`);
      if (stimulus.semantic_sha256 !== undefined && stimulus.semantic_sha256 !== null && stimulus.semantic_sha256 !== await sha256(new TextEncoder().encode(canonicalJson(input.request)))) throw new Error(`${where} stimulus digest differs from the canonical request`);
    } else if (states[0] === 'inline') request(stimulus.inline, `${where}.stimulus.inline`);
    else if (states[0] === 'partial') mapping(stimulus.partial, `${where}.stimulus.partial`);
    else string(getField(mapping(stimulus[states[0]], `${where}.stimulus.${states[0]}`), 'code'), `${where}.stimulus.${states[0]}.code`);
    }
    if (Object.hasOwn(change, 'observation')) {
    const observation = mapping(change.observation, `${where}.observation`);
    const outcomes = ['value', 'error', 'unavailable'].filter((state) => Object.hasOwn(observation, state));
    if (outcomes.length !== 1) throw new Error(`${where} must contain exactly one observation state`);
    keys(observation, outcomes, outcomes, `${where}.observation`);
    const payload = mapping(observation[outcomes[0]], `${where}.observation.${outcomes[0]}`);
    if (outcomes[0] === 'value') {
      keys(payload, [], ['assembled', 'chunks'], `${where}.observation.value`);
      if (!Object.hasOwn(payload, 'assembled') && !Object.hasOwn(payload, 'chunks')) throw new Error(`${where}.value needs assembled or chunks`);
    } else string(getField(payload, 'code'), `${where}.observation.${outcomes[0]}.code`);
  }
    }
  for (const field of ['metadata_changes', 'document_overrides']) {
    for (const [caseId, value] of Object.entries(document[field])) {
      if (!Object.hasOwn(inputDocument.cases, caseId)) throw new Error(`${path}:${caseId} is an unknown metadata case`);
      mapping(value, `${path}:${caseId}.${field}`);
      if (field !== 'document_overrides') continue;
      keys(value, [], ['parser_path', 'capture_provenance'], `${path}:${caseId}.${field}`);
      if (Object.hasOwn(value, 'parser_path')) string(value.parser_path, `${path}:${caseId}.parser_path`);
      if (Object.hasOwn(value, 'capture_provenance')) {
        const provenance = value.capture_provenance;
        keys(provenance, ['captured_with', 'capture_origin', 'inherited_from'], ['captured_with', 'capture_origin', 'inherited_from'], `${path}:${caseId}.capture_provenance`);
        mapping(provenance.captured_with, `${path}:${caseId}.captured_with`);
        if (!Object.keys(provenance.captured_with).length || Object.values(provenance.captured_with).some((version) => typeof version !== 'string')) throw new Error(`${path}:${caseId} has invalid captured_with`);
        if (provenance.capture_origin !== null) origin(provenance.capture_origin, `${path}:${caseId}.capture_origin`);
        if (provenance.inherited_from !== null) string(provenance.inherited_from, `${path}:${caseId}.inherited_from`);
      }
    }
  }
  return document;
}

export function materializeUnifiedRecord(input, record, family) {
  const observation = record.value.observation;
  const state = Object.keys(observation)[0];
  const payload = observation[state];
  let value = state === 'value' ? structuredClone(payload) : { [state]: getField(payload, 'detail') ?? getField(payload, 'message') ?? getField(payload, 'code') };
  if (record.value.stimulus.ref) value.capture_input = input.request;
  else if (record.value.stimulus.inline) value.capture_input = record.value.stimulus.inline;
  const metadata = mergeMappings(record.value.document, { capture_provenance: { format: 'schema_v3' } });
  value = mergeMappings(value, getField(metadata, 'record_metadata'));
  deleteField(metadata, 'record_metadata');
  const caseId = input.lifecycle === 'active' ? input.display_id : record.value.case_key;
  return mergeMappings(metadata, { family, cases: { [caseId]: value } });
}
