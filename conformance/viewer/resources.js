// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { sha256, createWorkerParser, validateResource, resourceMetrics } from './data.js';
import { canonicalJson, sameValue, getField, hasField, setField, deleteField, mergeMappings, mappingEntries, isMapping } from './values.js';
import { compareVersions, provenanceReleaseVersion, unifiedCaptureVersionPattern, selectedUnifiedFamilies, unifiedDisplayFamilyPredicate, selectedUnifiedSnapshot } from './model.js';

export function sourceReference(value, directory) {
  if (typeof value !== 'string' || !new RegExp(`^${directory}/[A-Za-z0-9][A-Za-z0-9_.+-]*\\.yaml$`).test(value)) throw new Error(`invalid ${directory} filename reference`);
  return value;
}

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
  const original = label.replace(/^UNIFIED\./, '');
  let key = original;
  if (key.endsWith(`.${family}`)) key = key.slice(0, -(family.length + 1));
  const familyScenario = metadata.family_scenarios?.[family]?.[key];
  if (familyScenario) return `UNIFIED.${metadata.canonical[familyScenario]}`;
  if (Object.hasOwn(metadata.canonical, key)) return `UNIFIED.${metadata.canonical[key]}`;
  const familyAliases = Object.hasOwn(metadata.families, family) ? metadata.families[family] : {};
  key = Object.hasOwn(familyAliases, key) ? familyAliases[key] : key;
  const dotted = key.match(/^31\.([a-x])$/);
  if (dotted) key = `31-${dotted[1].charCodeAt(0) - 'a'.charCodeAt(0) + 1}`;
  const owners = Object.hasOwn(metadata.owners, key) ? metadata.owners[key] : [];
  if (owners.length > 1) throw new Error(`conflicting Unified alias ownership: ${label}`);
  if (owners.length && metadata.owner_families?.[owners[0]] && metadata.owner_families[owners[0]] !== family) return `UNIFIED.${original}`;
  return owners.length ? `UNIFIED.${metadata.canonical[owners[0]]}` : `UNIFIED.${original}`;
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
  keys(value, ['crate_version', 'source_sha256'], ['crate_version', 'source_sha256', 'git_commit', 'adapter_sha256', 'job_sha256'], where);
  if (typeof value.crate_version !== 'string' || !unifiedCaptureVersionPattern.test(provenanceReleaseVersion(value.crate_version)) || (version !== null && value.crate_version !== provenanceReleaseVersion(version))) throw new Error(`${where} has invalid crate_version`);
  if (version !== null) {
    const suffix = version.slice(provenanceReleaseVersion(version).length);
    if (suffix && suffix !== `+source.${value.source_sha256}`) throw new Error(`${where} has invalid source identity`);
  }
  if (!/^[0-9a-f]{64}$/.test(value.source_sha256 || '')) throw new Error(`${where} has invalid source hash`);
  for (const field of ['adapter_sha256', 'job_sha256']) if (Object.hasOwn(value, field) && !/^[0-9a-f]{64}$/.test(value[field])) throw new Error(`${where} has invalid ${field}`);
  if (Object.hasOwn(value, 'adapter_sha256') !== Object.hasOwn(value, 'job_sha256')) throw new Error(`${where} needs both adapter and job identities`);
  if (Object.hasOwn(value, 'git_commit') && !/^[0-9a-f]{40}$/.test(value.git_commit)) throw new Error(`${where} has invalid git commit`);
}

function validateCaptureProvenance(provenance, implementation, version, where) {
  if (provenance.status === 'captured') {
    keys(provenance, ['status', 'origin'], ['status', 'origin'], `${where}.provenance`);
    origin(provenance.origin, `${where}.provenance.origin`, version);
  } else if (provenance.status === 'legacy') {
    keys(provenance, ['status', 'captured_with'], ['status', 'captured_with'], `${where}.provenance`);
    if (!sameValue(provenance.captured_with, { [implementation]: version })) throw new Error(`${where} has inconsistent capture identity`);
  } else throw new Error(`${where} has invalid capture provenance status`);
}

async function validateStimulus(stimulus, input, where) {
  mapping(stimulus, `${where}.stimulus`);
  const states = ['ref', 'inline', 'partial', 'error', 'unavailable'].filter((state) => Object.hasOwn(stimulus, state));
  if (states.length !== 1) throw new Error(`${where} must contain exactly one stimulus state`);
  keys(stimulus, states, [...states, 'semantic_sha256'], `${where}.stimulus`);
  if (states[0] === 'ref') {
    if (stimulus.ref !== 'current' || input.request === null) throw new Error(`${where} has invalid current stimulus reference`);
    if (stimulus.semantic_sha256 !== undefined && stimulus.semantic_sha256 !== null && stimulus.semantic_sha256 !== await sha256(new TextEncoder().encode(canonicalJson(input.request)))) throw new Error(`${where} stimulus digest differs from the canonical request`);
  } else if (states[0] === 'inline') {
    request(stimulus.inline, `${where}.stimulus.inline`);
    if (stimulus.semantic_sha256 !== undefined && stimulus.semantic_sha256 !== null && stimulus.semantic_sha256 !== await sha256(new TextEncoder().encode(canonicalJson(stimulus.inline)))) throw new Error(`${where} stimulus digest differs from its inline request`);
  } else if (states[0] === 'partial') mapping(stimulus.partial, `${where}.stimulus.partial`);
  else string(getField(mapping(stimulus[states[0]], `${where}.stimulus.${states[0]}`), 'code'), `${where}.stimulus.${states[0]}.code`);
}

async function validateExecutionProof(proof, inputDocument, provenance, implementation, version, where, proofRequests = null) {
  keys(proof, ['receipt', 'requests'], ['receipt', 'requests'], `${where}.execution_proof`);
  const receipt = mapping(proof.receipt, `${where}.execution_proof.receipt`);
  if (receipt.implementation !== implementation || receipt.version !== version) throw new Error(`${where} execution proof producer identity mismatch`);
  if (provenance.status !== 'captured' || Object.entries(provenance.origin).some(([key, value]) => key !== 'crate_version' && receipt[key] !== value)) throw new Error(`${where} execution proof differs from checkpoint origin`);
  if (['source_sha256', 'adapter_sha256', 'job_sha256', 'results_sha256'].some(key => typeof receipt[key] !== 'string' || !/^[0-9a-f]{64}$/.test(receipt[key]))) throw new Error(`${where} execution proof lacks sealed receipt hashes`);
  stringMapping(proof.requests, `${where}.execution_proof.requests`);
  if (!Object.keys(proof.requests).length || Object.values(proof.requests).some(digest => typeof digest !== 'string' || !/^[0-9a-f]{64}$/.test(digest))) throw new Error(`${where} execution proof requires sealed request digests`);
  const expected = proofRequests || {};
  if (!proofRequests) {
    for (const [caseId, input] of Object.entries(inputDocument.cases)) if (input.lifecycle !== 'retired') expected[caseId] = await sha256(new TextEncoder().encode(canonicalJson(input.request)));
  }
  if (!sameValue(proof.requests, expected)) throw new Error(`${where} execution proof request membership mismatch`);
}

export async function validateUnifiedSnapshot(snapshot, inputDocument, resource) {
  const { path, implementation, version } = resource;
  const fields = ['schema_version', 'provenance', 'document', 'records'];
  keys(snapshot, fields, [...fields, 'execution_proof', 'historical_evidence', 'capture_selection'], path);
  if (snapshot.schema_version !== 2n && snapshot.schema_version !== 2) throw new Error(`${path} invalid Unified snapshot schema`);
  if (typeof version !== 'string' || !unifiedCaptureVersionPattern.test(version)) throw new Error(`${path} invalid Unified capture version`);
  const provenance = mapping(snapshot.provenance, `${path}.provenance`);
  validateCaptureProvenance(provenance, implementation, version, path);
  mapping(snapshot.document, `${path}.document`);
  if (['capture_origin', 'captured_with'].some(field => hasField(snapshot.document, field))) throw new Error(`${path} repeats derived provenance`);
  stringMapping(snapshot.records, `${path}.records`);
  for (const [caseId, record] of Object.entries(snapshot.records)) {
    if (!Object.hasOwn(inputDocument.cases, caseId)) throw new Error(`${path}:${caseId} is an unknown case`);
    keys(record, ['value', 'member'], ['value', 'member', 'suppress_inherited_from'], `${path}:${caseId}`);
    if (typeof record.member !== 'string' || !/^conformance\/fixtures-unified-v2\/families\/[a-z0-9_]+\/[A-Za-z0-9_.+-]+\.yaml$/.test(record.member)) throw new Error(`${path}:${caseId} has an invalid source member`);
    if (Object.hasOwn(record, 'suppress_inherited_from') && record.suppress_inherited_from !== true) throw new Error('invalid snapshot inheritance projection');
    const value = record.value;
    keys(value, ['case_key', 'stimulus', 'observation', 'document'], ['case_key', 'stimulus', 'observation', 'document'], `${path}:${caseId}.value`);
    component(value.case_key, `${path}:${caseId}.case_key`);
    await validateStimulus(value.stimulus, inputDocument.cases[caseId], `${path}:${caseId}`);
    mapping(value.document, `${path}:${caseId}.value.document`);
  }
  if (Object.hasOwn(snapshot, 'execution_proof')) {
    const executed = {};
    for (const [caseId, wrapped] of Object.entries(snapshot.records)) {
      if (!Object.hasOwn(wrapped.value.stimulus, 'ref') && !Object.hasOwn(wrapped.value.stimulus, 'inline')) throw new Error(`${path}:${caseId} execution proof requires a complete request`);
      executed[caseId] = wrapped.value.stimulus.semantic_sha256;
    }
    await validateExecutionProof(snapshot.execution_proof, inputDocument, provenance, implementation, version, path, executed);
  }
  if (Object.hasOwn(snapshot, 'historical_evidence')) {
    if (!Array.isArray(snapshot.historical_evidence)) throw new Error(`${path}.historical_evidence must be a list`);
    for (const [index, evidence] of snapshot.historical_evidence.entries()) {
      const where = `${path}.historical_evidence[${index}]`;
      keys(evidence, ['kind', 'schema_version', 'family', 'implementation', 'version', 'checkpoint', 'cases', 'defaults', 'exceptions', 'metadata_changes', 'receipt_membership'], ['kind', 'schema_version', 'family', 'implementation', 'version', 'checkpoint', 'cases', 'defaults', 'exceptions', 'metadata_changes', 'receipt_membership', 'execution_proof'], where);
      if (evidence.kind !== 'execution_manifest' || evidence.schema_version !== 1n && evidence.schema_version !== 1
          || evidence.family !== resource.family || evidence.implementation !== implementation || evidence.version !== version) throw new Error(`${where} producer identity mismatch`);
      mapping(evidence.checkpoint, `${where}.checkpoint`);
      mapping(evidence.defaults, `${where}.defaults`);
      mapping(evidence.exceptions, `${where}.exceptions`);
      if (!Array.isArray(evidence.cases) || new Set(evidence.cases).size !== evidence.cases.length) throw new Error(`${where} has invalid metadata membership`);
      stringMapping(evidence.metadata_changes, `${where}.metadata_changes`);
      if (!sameValue(Object.keys(evidence.metadata_changes).sort(), [...evidence.cases].sort())) throw new Error(`${where} metadata scope differs from its cases`);
      if (!Array.isArray(evidence.receipt_membership)) throw new Error(`${where} has invalid receipt membership`);
      for (const [receiptIndex, entry] of evidence.receipt_membership.entries()) {
        const receiptWhere = `${where}.receipt_membership[${receiptIndex}]`;
        keys(entry, ['members', 'body'], ['members', 'body'], receiptWhere);
        if (!Array.isArray(entry.members) || new Set(entry.members).size !== entry.members.length) throw new Error(`${receiptWhere} has invalid members`);
        mapping(entry.body, `${receiptWhere}.body`);
      }
    }
  }
  return snapshot;
}

export function unifiedRecordContract(implementation, document, captureMode) {
  if (!['batch', 'stream'].includes(captureMode)) throw new Error(`invalid Unified capture mode for ${implementation}`);
  if (captureMode !== 'batch') return 'stream';
  const origin = getField(document, 'capture_origin') || {};
  const adapter = getField(origin, 'adapter_sha256');
  const job = getField(origin, 'job_sha256');
  if (Boolean(adapter) !== Boolean(job)) throw new Error(`${implementation} adapter-backed batch observation needs adapter and job identities`);
  if (adapter && job) return 'batch';
  return hasField(getField(document, 'record_metadata') || {}, 'batch_final') ? 'batch' : 'legacy-stream';
}

function batchFinalEvents(final, where) {
  const fields = mappingEntries(final);
  if (fields.length !== 3 || !['reasoning_text', 'normal_text', 'calls'].every(key => hasField(final, key))) throw new Error(`${where} needs reasoning_text, normal_text, and calls`);
  const reasoning = getField(final, 'reasoning_text');
  const text = getField(final, 'normal_text');
  const calls = getField(final, 'calls');
  if (typeof reasoning !== 'string' || typeof text !== 'string' || !Array.isArray(calls)) throw new Error(`${where} has invalid batch final field types`);
  const events = [];
  if (reasoning) events.push({ kind: 'reasoning', text: reasoning });
  if (text) events.push({ kind: 'text', text });
  for (const [index, call] of calls.entries()) {
    if (mappingEntries(call).length !== 2 || typeof getField(call, 'name') !== 'string' || !hasField(call, 'arguments')) throw new Error(`${where}.calls[${index}] is invalid`);
    events.push({ kind: 'tool_call', name: getField(call, 'name'), arguments: getField(call, 'arguments') });
  }
  return events;
}

export function validateResolvedRecordContracts(records, implementation, captureMode, where = implementation) {
  for (const [caseId, record] of records) {
    const contract = unifiedRecordContract(implementation, getField(getField(record, 'value'), 'document'), captureMode);
    const observation = getField(getField(record, 'value'), 'observation');
    if (contract !== 'batch' || !hasField(observation, 'value')) continue;
    const value = getField(observation, 'value');
    const metadata = getField(getField(record, 'value'), 'document');
    const recordMetadata = getField(metadata, 'record_metadata') || {};
    const final = getField(recordMetadata, 'batch_final');
    if (!final || hasField(value, 'chunks') || !sameValue(getField(value, 'assembled'), batchFinalEvents(final, `${where}:${caseId}.batch_final`))) {
      throw new Error(`${where}:${caseId} batch observation does not match its final fields`);
    }
  }
  return records;
}

function mergeExecutionMetadata(defaults, exception) {
  const result = defaults instanceof Map || exception instanceof Map ? new Map(mappingEntries(structuredClone(defaults))) : structuredClone(defaults);
  for (const [key, value] of mappingEntries(exception)) {
    const prior = getField(result, key);
    setField(result, key, isMapping(prior) && isMapping(value) ? mergeExecutionMetadata(prior, value) : structuredClone(value));
  }
  return result;
}

function* executionReceiptSlots(metadata) {
  for (const [key] of mappingEntries(metadata)) if (typeof key === 'string' && (key === 'capture_provenance' || key.endsWith('_capture_provenance'))) yield [metadata, key];
  const owners = [getField(metadata, 'verified_recapture')];
  const archived = getField(metadata, 'verified_recaptures');
  if (Array.isArray(archived)) owners.push(...archived);
  for (const owner of owners) if (isMapping(owner) && hasField(owner, 'provenance')) yield [owner, 'provenance'];
}

export async function expandUnifiedExecution(document, executionManifest, resource) {
  const { path, implementation, version, family } = resource;
  keys(document, ['execution'], ['execution', 'parent', 'changes'], path);
  sourceReference(document.execution, 'executions');
  if (!executionManifest) throw new Error(`${path} execution manifest is missing`);
  const manifest = executionManifest.value;
  const fields = ['schema_version', 'family', 'implementation', 'version', 'checkpoint', 'cases', 'defaults', 'exceptions'];
  keys(manifest, fields, [...fields, 'receipts', 'proof'], `${path}.execution`);
  if (manifest.schema_version !== 1n && manifest.schema_version !== 1) throw new Error(`${path} invalid execution schema`);
  if (manifest.family !== family || manifest.implementation !== implementation || manifest.version !== version) throw new Error(`${path} execution producer identity mismatch`);
  const checkpointFields = ['provenance', 'document', 'document_overrides'];
  keys(manifest.checkpoint, checkpointFields, checkpointFields, `${path}.execution.checkpoint`);
  if (!Array.isArray(manifest.cases) || (!manifest.cases.length && !Object.hasOwn(manifest, 'proof')) || manifest.cases.some(item => typeof item !== 'string') || new Set(manifest.cases).size !== manifest.cases.length) throw new Error(`${path} invalid execution case membership`);
  mapping(manifest.defaults, `${path}.execution.defaults`);
  if (!manifest.cases.length && mappingEntries(manifest.defaults).length) throw new Error(`${path} execution defaults require metadata case membership`);
  stringMapping(manifest.exceptions, `${path}.execution.exceptions`);
  for (const [caseId, value] of Object.entries(manifest.exceptions)) {
    if (!manifest.cases.includes(caseId)) throw new Error(`${path} execution exception outside membership`);
    mapping(value, `${path}.execution.exceptions.${caseId}`);
  }
  const metadata = Object.fromEntries(manifest.cases.map(caseId => [caseId, mergeExecutionMetadata(manifest.defaults, Object.hasOwn(manifest.exceptions, caseId) ? manifest.exceptions[caseId] : {})]));
  const memberships = manifest.receipts || {};
  stringMapping(memberships, `${path}.execution.receipts`);
  const proof = Object.hasOwn(manifest, 'proof') ? structuredClone(manifest.proof) : null;
  if (Object.hasOwn(manifest, 'proof')) {
    keys(proof, ['receipt', 'requests'], ['receipt', 'requests'], `${path}.execution.proof`);
    stringMapping(proof.requests, `${path}.execution.proof.requests`);
    sourceReference(proof.receipt, 'receipts');
  }
  const allowedCases = new Set([...manifest.cases, ...Object.keys(proof?.requests || {})]);
  const receipts = new Map();
  for (const [digest, members] of Object.entries(memberships)) {
    sourceReference(digest, 'receipts');
    if (!Array.isArray(members) || members.some(caseId => typeof caseId !== 'string' || !allowedCases.has(caseId)) || new Set(members).size !== members.length) throw new Error(`${path} invalid receipt membership`);
    const receipt = executionManifest.receipts?.get(digest);
    if (!receipt) throw new Error(`${path} execution receipt is missing`);
    mapping(receipt.value, `${path}.execution.receipt`);
    receipts.set(digest, receipt.value);
  }
  const observed = new Map();
  if (proof !== null) {
    const digest = proof.receipt;
    if (!receipts.has(digest)) throw new Error(`${path} execution proof receipt body or membership missing`);
    observed.set(digest, new Set(Object.keys(proof.requests)));
    proof.receipt = structuredClone(receipts.get(digest));
  }
  for (const [caseId, value] of Object.entries(metadata)) {
    for (const [owner, key] of executionReceiptSlots(value)) {
      const reference = getField(owner, key);
      if (typeof reference !== 'string' || !reference.startsWith('receipts/')) continue;
      const digest = sourceReference(reference, 'receipts');
      if (!receipts.has(digest)) throw new Error(`${path} execution receipt body or membership missing`);
      if (!observed.has(digest)) observed.set(digest, new Set());
      observed.get(digest).add(caseId);
      setField(owner, key, structuredClone(receipts.get(digest)));
    }
  }
  for (const [digest, members] of Object.entries(memberships)) {
    const actual = observed.get(digest) || new Set();
    if (!actual.size || actual.size !== members.length || members.some(caseId => !actual.has(caseId))) throw new Error(`${path} execution receipt case membership mismatch`);
  }
  return { ...structuredClone(manifest.checkpoint), changes: structuredClone(document.changes || {}), metadata_changes: metadata,
    ...(proof !== null ? { execution_proof: proof } : {}),
    ...(Object.hasOwn(document, 'parent') ? { parent: document.parent } : {}) };
}

export async function validateUnifiedCapture(document, inputDocument, resource, executionManifest = null) {
  const { path, implementation, version } = resource;
  if (typeof version !== 'string' || !unifiedCaptureVersionPattern.test(version)) throw new Error(`${path} invalid Unified capture version`);
  if (Object.hasOwn(document, 'schema_version')) return validateUnifiedSnapshot(document, inputDocument, resource);
  if (Object.hasOwn(document, 'execution')) document = await expandUnifiedExecution(document, executionManifest, resource);
  return validateCapturePayload(document, inputDocument, resource);
}

async function validateCapturePayload(document, inputDocument, resource) {
  const { path, implementation, version } = resource;
  if (typeof version !== 'string' || !unifiedCaptureVersionPattern.test(version)) throw new Error(`${path} invalid Unified capture version`);
  const fields = ['provenance', 'document', 'changes', 'metadata_changes', 'document_overrides'];
  keys(document, fields, [...fields, 'parent', 'execution_proof'], path);
  if (Object.hasOwn(document, 'parent') && (typeof document.parent !== 'string' || !document.parent.startsWith(`${implementation}-`)
      || compareVersions(document.parent.slice(implementation.length + 1), version) >= 0)) throw new Error(`${path} invalid capture parent`);
  const provenance = mapping(document.provenance, `${path}.provenance`);
  if (provenance.status === 'captured') {
    keys(provenance, ['status', 'origin'], ['status', 'origin'], `${path}.provenance`);
    origin(provenance.origin, `${path}.provenance.origin`, version);
  } else if (provenance.status === 'legacy') {
    keys(provenance, ['status', 'captured_with'], ['status', 'captured_with'], `${path}.provenance`);
    if (!sameValue(provenance.captured_with, { [implementation]: version })) throw new Error(`${path} has inconsistent capture identity`);
  } else throw new Error(`${path} has invalid capture provenance status`);
  if (Object.hasOwn(document, 'execution_proof')) await validateExecutionProof(document.execution_proof, inputDocument, provenance, implementation, version, path);
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
    await validateStimulus(change.stimulus, input, where);
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
      if (Object.hasOwn(value, 'parser_path') && value.parser_path !== null) string(value.parser_path, `${path}:${caseId}.parser_path`);
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

const originalSourcesByPath = new WeakMap();

export async function validateResolvedUnified(records, inputDocument, resource, originals) {
  stringMapping(records, resource.path);
  let sourceByPath = originalSourcesByPath.get(originals);
  if (!sourceByPath) {
    sourceByPath = new Map(originals.map(item => [item.path, item]));
    originalSourcesByPath.set(originals, sourceByPath);
  }
  const changes = {};
  for (const [caseId, record] of Object.entries(records)) {
    keys(record, ['value', 'member'], ['value', 'member', 'suppress_inherited_from'], `${resource.path}:${caseId}`);
    const source = sourceByPath.get(record.member);
    if (!source || source.kind !== 'history' || source.family !== resource.family || source.implementation !== resource.implementation || compareVersions(source.version, resource.version) > 0) throw new Error(`resolved observation has an invalid source: ${caseId}`);
    if (Object.hasOwn(record, 'suppress_inherited_from') && record.suppress_inherited_from !== true) throw new Error('invalid snapshot inheritance projection');
    const value = record.value;
    keys(value, ['case_key', 'stimulus', 'observation', 'document'], ['case_key', 'stimulus', 'observation', 'document'], `${resource.path}:${caseId}.value`);
    mapping(value.document, `${resource.path}:${caseId}.document`);
    const capturedWith = getField(value.document, 'captured_with');
    if (!hasField(capturedWith, resource.implementation)
        || mappingEntries(capturedWith).some(([implementation, version]) => typeof implementation !== 'string' || typeof version !== 'string' || !unifiedCaptureVersionPattern.test(provenanceReleaseVersion(version)))
        || compareVersions(provenanceReleaseVersion(getField(capturedWith, resource.implementation)), resource.version) > 0) {
      throw new Error(`resolved observation has a false producer identity: ${caseId}`);
    }
    if (hasField(value.document, 'capture_origin')) origin(getField(value.document, 'capture_origin'), `${resource.path}:${caseId}.capture_origin`, getField(capturedWith, resource.implementation));
    Object.defineProperty(changes, caseId, { value: { case_key: value.case_key, stimulus: value.stimulus, observation: value.observation }, enumerable: true });
  }
  await validateCapturePayload({ provenance: { status: 'legacy', captured_with: { [resource.implementation]: resource.version } }, document: {}, changes, metadata_changes: {}, document_overrides: {} }, inputDocument, resource);
  validateResolvedRecordContracts(new Map(Object.entries(records)), resource.implementation, resource.record_contract, resource.path);
  return records;
}

export function selectVerifiedRecapture(value, input, stimulus, implementation, version, captureMode) {
  const recapture = getField(value, 'verified_recapture');
  if (recapture == null) return value;
  keys(recapture, ['result', 'provenance', 'resolution'], ['result', 'provenance', 'resolution'], 'verified recapture');
  const provenance = recapture.provenance;
  const batch = captureMode === 'batch';
  if (!['sglang_python', 'vllm_python', 'vllm_rust', 'dynamo_v2'].includes(implementation) && !batch) throw new Error('unsupported verified recapture implementation');
  if (implementation !== 'sglang_python' && getField(provenance, 'implementation') !== implementation) throw new Error('verified recapture implementation differs from candidate');
  if (getField(provenance, 'version') !== version) throw new Error('verified recapture version differs from candidate');
  for (const field of ['source_sha256', 'adapter_sha256', 'job_sha256']) {
    if (!/^[0-9a-f]{64}$/.test(getField(provenance, field) || '')) throw new Error(`verified recapture has invalid ${field}`);
  }
  const resolution = recapture.resolution;
  if (resolution === 'historical_request_not_retained') {
    if (!sameValue(stimulus, { unavailable: { code: 'original_request_not_retained' } })) throw new Error('verified recapture requires an unbound historical observation');
  } else if (resolution === 'historical_adapter_unsupported' && implementation !== 'sglang_python') {
    if (!hasField(value, 'unavailable') || !sameValue(getField(value, 'capture_input'), input)) throw new Error('adapter recapture requires a bound historical unavailable result');
  } else if (resolution === 'fresh_execution') {
    if (getField(provenance, 'implementation') !== implementation) throw new Error('verified recapture implementation differs from candidate');
  } else {
    throw new Error('unresolved historical recapture disagreement');
  }
  const result = recapture.result;
  if (!sameValue(getField(result, 'capture_input'), input)) throw new Error('verified recapture request differs from displayed request');
  if (['error', 'unavailable', batch ? 'assembled' : 'chunks'].filter(key => hasField(result, key)).length !== 1) throw new Error('verified recapture must carry exactly one result state');
  if (batch && (hasField(result, 'chunks') || (hasField(result, 'assembled') && !sameValue(getField(result, 'assembled'), batchFinalEvents(getField(result, 'batch_final'), 'verified recapture'))))) throw new Error('invalid verified batch final fields');
  return mergeMappings(result, {
    producer_origin: mergeMappings(provenance, { crate_version: version }),
    historical_capture_note: resolution === 'fresh_execution' && hasField(result, 'unavailable')
      ? 'The displayed request remains unavailable in this capture; the original observation and origin remain preserved in YAML.'
      : 'Fresh execution of the displayed request; the original observation and origin remain preserved in YAML.',
  });
}

export function materializeUnifiedRecord(input, record, family, identity = null) {
  const observation = record.value.observation;
  const state = Object.keys(observation)[0];
  const payload = observation[state];
  let value = state === 'value' ? structuredClone(payload) : { [state]: getField(payload, 'detail') ?? getField(payload, 'message') ?? getField(payload, 'code') };
  if (state === 'unavailable') value.unavailable_code = getField(payload, 'code');
  if (record.value.stimulus.ref) value.capture_input = input.request;
  else if (record.value.stimulus.inline) value.capture_input = record.value.stimulus.inline;
  const metadata = mergeMappings(record.value.document, { capture_provenance: { format: 'schema_v3' } });
  value = mergeMappings(value, getField(metadata, 'record_metadata'));
  deleteField(metadata, 'record_metadata');
  if (identity) value = selectVerifiedRecapture(value, input.request, record.value.stimulus, identity.implementation, identity.version, identity.captureMode);
  const caseId = input.lifecycle === 'active' ? input.display_id : record.value.case_key;
  return mergeMappings(metadata, { family, cases: { [caseId]: value } });
}

function cachedTask(cache, key, create) {
  if (!cache.has(key)) {
    const task = create().catch((error) => { cache.delete(key); throw error; });
    cache.set(key, task);
  }
  return cache.get(key);
}

// Transport errors stay outside recorded outcomes. Failed reads are evicted so
// Retry can recover without weakening hash validation or changing parser state.
export class FixtureResources {
  constructor(catalog, store, parser = createWorkerParser(), embeddedCatalog = null) {
    this.catalog = catalog;
    this.store = store;
    this.parser = parser;
    this.archives = new Map();
    this.documents = new Map();
    this.captures = new Map();
    this.inputResources = new Map();
    this.embedded = embeddedCatalog ? new FixtureResources(embeddedCatalog, store, parser) : null;
    const latestVersions = new Map();
    for (const capture of embeddedCatalog?.capture_generation?.current || []) {
      const prior = latestVersions.get(capture.implementation);
      if (!prior || compareVersions(capture.version, prior) > 0) latestVersions.set(capture.implementation, capture.version);
    }
    this.embeddedIdentities = new Set([...latestVersions].map(([implementation, version]) => `${implementation}@${version}`));
  }

  unifiedDependencies(descriptors) {
    if (this.embedded) {
      const latest = descriptors.filter(item => this.isEmbedded(item.source.implementation, item.source.version));
      const historical = descriptors.filter(item => item.source.implementation !== 'golden' && !latest.includes(item));
      return [...(latest.length ? this.embedded.unifiedDependencies(latest) : []),
        ...(historical.length ? this.catalogDependencies(historical) : [])];
    }
    return this.catalogDependencies(descriptors);
  }

  isEmbedded(implementation, version) {
    return this.embeddedIdentities.has(`${implementation}@${version}`);
  }

  catalogDependencies(descriptors) {
    const families = selectedUnifiedFamilies(this.catalog, descriptors);
    const capturesFamily = unifiedDisplayFamilyPredicate(this.catalog, descriptors);
    const inputs = this.catalog.sources.unified.filter(resource => resource.kind === 'inputs-and-golden' && families.has(resource.family));
    const histories = new Map();
    for (const descriptor of descriptors) {
      for (const family of families) {
        if (!capturesFamily(descriptor.source, family)) continue;
        const packed = this.catalog.sources.unified.find(resource => resource.kind === 'resolved-capture'
          && resource.implementation === descriptor.source.implementation
          && resource.version === descriptor.source.version
          && (resource.family === family || resource.families?.includes(family)));
        if (packed) {
          histories.set(packed.path, packed);
          continue;
        }
        const selected = this.catalog.sources.unified
          .filter(resource => resource.kind === 'history' && resource.family === family
            && resource.implementation === descriptor.source.implementation
            && compareVersions(resource.version, descriptor.source.version) <= 0)
          .sort((left, right) => compareVersions(right.version, left.version))[0];
        if (selected) histories.set(selected.path, selected);
      }
    }
    const selected = new Set([...inputs, ...histories.values()].map(resource => resource.path));
    return this.catalog.sources.unified.filter(resource => selected.has(resource.path));
  }

  async prepareUnified(descriptors) {
    const dependencies = this.unifiedDependencies(descriptors);
    this.inputResources = new Map();
    for (const resource of dependencies.filter(item => item.kind === 'inputs-and-golden')) {
      this.inputResources.set(resource.family, resource);
    }
    const selection = this.store.demand?.(dependencies) || true;
    try {
      await Promise.all(dependencies.map(resource => this.store.read(resource, {selection})));
    } finally {
      this.store.releaseReservations?.(selection);
    }
  }

  async admitResource(resource) {
    validateResource(resource);
    await this.store.admit?.(resource);
  }

  async archive(resource) {
    await this.admitResource(resource);
    return cachedTask(this.archives, `${resource.sha256}:${resource.size}`, async () => this.parser.parseArchive(await this.store.read(resource), { path: resource.path }));
  }

  async yaml(resource, member = null, unified = false) {
    if (resource.format === 'typed-json-v1') {
      if (member || !unified || resource.kind !== 'inputs-and-golden') throw new Error('snapshot is not an authored Unified document');
      return this.snapshot(resource);
    }
    await this.admitResource(resource);
    const key = `${resource.sha256}:${resource.size}:${member || ''}:${unified}`;
    const parsed = await cachedTask(this.documents, key, async () => {
      const bytes = member ? (await this.archive(resource)).get(member) : await this.store.read(resource);
      if (!bytes) throw new Error(`archive member missing: ${member} in ${resource.path}`);
      const value = await this.parser.parseYaml(bytes, { path: member || resource.path, unified });
      return { value, sourceBytes: bytes };
    });
    return { ...parsed, member: member || resource.path, resource: resource.path };
  }

  async snapshot(resource) {
    await this.admitResource(resource);
    const value = await cachedTask(this.documents, `${resource.sha256}:${resource.size}:snapshot`, async () => this.parser.parseSnapshot(await this.store.read(resource), resource));
    return { value, member: resource.path, resource: resource.path };
  }



  async unifiedInputs(family) {
    const input = this.inputResources.get(family) || this.catalog.sources.unified.find(item => item.kind === 'inputs-and-golden' && item.family === family);
    if (!input) throw new Error(`Unified input and GOLDEN document missing for ${family}`);
    const parsedInputs = input.format === 'typed-json-v1' ? await this.snapshot(input) : await this.yaml(input, null, true);
    const inputDocument = parsedInputs.value;
    if (!inputDocument) throw new Error(`Unified input and GOLDEN document missing for ${family}`);
    validateUnifiedInputs(inputDocument, family, input.path, this.catalog.metadata.unified_aliases);
    return { ...parsedInputs, value: inputDocument, sourceResource: input };
  }

  async unified(family, implementation, version, inputDocument = undefined) {
    // Latest envelopes are bound to the embedded catalog's input fingerprint.
    // Keep their normal validation and recipes when history adds another catalog.
    if (this.isEmbedded(implementation, version)) return this.embedded.unified(family, implementation, version);
    const inputs = inputDocument === undefined ? await this.unifiedInputs(family)
      : { value: inputDocument, sourceResource: this.inputResources.get(family) || this.catalog.sources.unified.find(item => item.kind === 'inputs-and-golden'
        && item.family === family) };
    const { sourceResource: input, ...inputResult } = inputs;
    inputDocument = inputResult.value;
    if (input.format === 'typed-json-v1') {
      const resource = this.catalog.sources.unified.find(item => item.kind === 'resolved-capture'
        && item.implementation === implementation && item.version === version
        && (item.family === family || item.families?.includes(family)));
      if (!resource) return { inputs: inputResult, records: new Map() };
      if (resource.format !== input.format || resource.source_fingerprint !== input.source_fingerprint) throw new Error('inconsistent Unified resource format or fingerprint');
      const resolved = await this.snapshot(resource);
      const familyRecords = Array.isArray(resource.families) ? resolved.value[family] : resolved.value;
      if (!familyRecords) return { inputs: inputResult, records: new Map() };
      const familyResource = Array.isArray(resource.families) ? { ...resource, family } : resource;
      const validationStart = performance.now();
      await validateResolvedUnified(familyRecords, inputDocument, familyResource, this.catalog.sources.original_unified || []);
      resourceMetrics.validation += performance.now() - validationStart;
      return { inputs: inputResult, records: new Map(Object.entries(structuredClone(familyRecords))) };
    }
    const captureMode = this.catalog.metadata.unified_capture_policy?.[implementation]?.mode;
    if (!captureMode) throw new Error(`Unified capture mode missing for ${implementation}`);
    const file = this.catalog.sources.unified
      .filter(item => item.kind === 'history' && item.family === family && item.implementation === implementation
        && compareVersions(item.version, version) <= 0)
      .sort((left, right) => compareVersions(right.version, left.version))[0];
    if (!file) return { inputs: inputResult, records: new Map() };
    const capture = await this.unifiedCapture(file, inputDocument, input);
    await validateResolvedUnified(capture.document.records, inputDocument, { ...file, record_contract: captureMode },
      this.catalog.sources.original_unified || this.catalog.sources.unified);
    return { inputs: inputResult, records: validateResolvedRecordContracts(
      selectedUnifiedSnapshot(new Map(Object.entries(capture.document.records)), file.version, version), implementation, captureMode) };
  }

  async unifiedCapture(file, inputDocument, input) {
    const identity = JSON.stringify([input, file].map(resource => [resource.path, resource.sha256, resource.size]));
    return cachedTask(this.captures, identity, async () => {
      const parsed = await this.yaml(file, null, true);
      const document = await validateUnifiedSnapshot(parsed.value, inputDocument, file);
      return { ...file, document, bytes: parsed.sourceBytes };
    });
  }



  dispose() { this.parser.terminate?.(); }
}


export async function rawResource(catalog, store, path) {
  const lookup = async key => {
    const bucket = (await sha256(new TextEncoder().encode(key))).slice(0, 2);
    const descriptor = catalog.transport.raw_buckets[bucket];
    if (!descriptor) throw new Error(`Original resource is not indexed: ${key}`);
    const index = await store.transport.index(descriptor);
    if (!Object.hasOwn(index.resources, key)) throw new Error(`Original resource is not indexed: ${key}`);
    return index.resources[key];
  };
  const entry = await lookup(path);
  if (entry.resource) return entry;
  if (typeof entry.path !== 'string' || typeof entry.member !== 'string') throw new Error('Invalid original alias');
  const target = await lookup(entry.path);
  if (!target.resource || target.member !== null) throw new Error('Invalid original alias target');
  return { resource: target.resource, member: entry.member };
}

export function boundOracleRecord(value, authoredCase) {
  const result = structuredClone(value);
  const stimulus = getField(result, 'stimulus');
  // A source-local reference binds to that source's authored request, even when
  // the published capture is displayed beside a newer checkout request.
  if (getField(stimulus, 'ref') === 'current') {
    deleteField(stimulus, 'ref');
    setField(stimulus, 'inline', structuredClone(getField(authoredCase, 'request')));
  }
  return result;
}
