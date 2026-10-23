// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { sameValue } from './values.js';
import { isApplicableCell } from './folds.js';

// Matrix paint and status reports consume the same recorded comparison relation.
// The report rejects Reference errors even when the legacy matrix has no concrete
// Compare result with which to show a delta; expose both facts explicitly.
export function selectedCellOutcome({ cmp = {}, base = null, shown = [], redOnDiff = false, family = '', parserNi = {} } = {}) {
  const entry = (key) => Object.hasOwn(cmp, key) ? cmp[key] : null;
  const reference = base ? entry(base) : null;
  const golden = entry('golden');
  const compares = shown.filter((key) => key !== base);
  const outcomes = compares.map((key) => ({ key, value: entry(key), outcome: selectedCaptureOutcome({ kind: 'cell', cmp }, key, base || 'golden') }));
  const available = outcomes.filter(({ value }) => value && value.na !== 1 && value.err !== 1);
  const unavailable = compares.filter((key) => entry(key)?.na === 1).length;
  const signature = golden?.sig != null ? golden.sig : reference?.sig;
  const differences = available.filter(({ value }) => !sameValue(value.sig, signature)).length;
  const errors = outcomes.filter(({ outcome }) => outcome === 'error').length;
  const leaks = outcomes.filter(({ outcome }) => outcome === 'leak').length;
  const result = { className: 'cmp-na', marker: 'n/a', category: 'na', why: '', referenceState: 'missing',
    missingGolden: golden == null, referenceDiffersFromGolden: reference && golden ? !sameValue(reference.sig, golden.sig) : null,
    referenceLeaks: reference?.leak === 1, availableComparisons: available.length, unavailableComparisons: unavailable,
    comparisonDifferences: differences, errorComparisons: errors, leakedComparisons: leaks };
  const ni = base && Object.hasOwn(parserNi, base) ? parserNi[base] : null;
  if (base && ni && !ni.families.includes(family)) {
    return { ...result, marker: 'x', referenceState: 'notImplemented', why: `${ni.label} is not yet implemented for family “${family}”` };
  }
  if (!base) return { ...result, className: 'cmp-nobase', marker: '', referenceState: 'none' };
  if (base === 'golden' && !golden) return { ...result, className: 'cmp-leak', marker: '✗', category: 'problem', referenceState: 'missing' };
  if (!reference) return result;
  if (reference.na === 1) return { ...result, referenceState: 'unavailable' };
  if (reference.err === 1) {
    if (redOnDiff) return { ...result, className: 'cmp-leak', marker: '✗', category: 'problem', referenceState: 'error' };
    if (available.length) return { ...result, className: 'cmp-leak', marker: `${available.length}Δ`, category: 'problem', referenceState: 'error' };
    return { ...result, referenceState: 'error' };
  }
  if (errors || leaks) return { ...result, className: 'cmp-leak', marker: `${errors ? '✗' : ''}${leaks ? '↯' : ''}`, category: 'problem', referenceState: errors ? 'error' : 'leak' };
  if (redOnDiff && !available.length) return { ...result, className: 'cmp-na', marker: unavailable ? '?' : 'n/a', category: 'na', referenceState: 'unavailable' };
  const diverges = redOnDiff && base === 'golden' && golden?.sig != null && differences > 0;
  const agreement = unavailable ? '?' : '=';
  const marker = redOnDiff ? diverges ? '✗' : differences ? `${differences}Δ` : agreement
    : !available.length ? '' : differences ? `${differences}Δ` : agreement;
  const problem = redOnDiff ? Boolean(diverges || golden == null) : result.referenceLeaks;
  return { ...result, className: problem ? 'cmp-leak' : 'cmp-eq', marker: (result.referenceLeaks ? '↯' : '') + marker,
    category: problem ? 'problem' : 'ok', referenceState: 'value' };
}

function tooltipCandidate(cell, key) {
  const candidates = cell?.tooltip?.candidates || [];
  return candidates.find(candidate => candidate.key === key) || null;
}

function sourceMatches(key, impl) {
  return typeof key === 'string' && typeof impl === 'string'
    && (key === impl || key.startsWith(impl + '-') || key.startsWith(impl + '@'));
}

function captureIssue(cell, key) {
  if (!cell || cell.kind === 'blank' || cell.kind === 'missing') return 'missing';
  const entry = (cell.cmp || {})[key];
  const candidateTip = tooltipCandidate(cell, key);
  const block = candidateTip?.block;
  if (!entry || entry.na === 1) {
    const reason = String((block && block.unavailable) || '');
    const code = block && (block.unavailable_code || (block.capture_observation && block.capture_observation.unavailable && block.capture_observation.unavailable.code));
    if (code === 'missing_capture') return 'missing';
    if (['unsupported_family', 'unsupported_request_mode', 'unsupported_capture_mode', 'scenario_not_applicable', 'duplicate_scenario'].includes(code)) return 'unsupported';
    if (code === 'source_unavailable' || /source unavailable|source blocked|historical source/i.test(reason)) return 'blocked';
    return /\bno(?: [\w.-]+)* parser\b|\bnot (?:yet )?implemented\b|unsupported (?:family|request|mode)|cannot apply/i.test(reason) ? 'unsupported' : 'missing';
  }
  if (entry.err === 1 || (block && (block.exception || (block.error && typeof block.error === 'object')))) return 'error';
  if (block && typeof block.error === 'string') return 'expected';
  if (entry.leak === 1 || candidateTip?.leak) return 'leak';
  return null;
}

export function selectedCaptureOutcome(cell, key, reference = 'golden') {
  if (cell?.variants) {
    const outcomes = cell.variants.filter(isApplicableCell).map(leaf => selectedCaptureOutcome(leaf, key, reference));
    return ['error', 'leak', 'diverge', 'missing', 'blocked', 'unsupported', 'uncompared', 'documented', 'pass']
      .find(outcome => outcomes.includes(outcome)) || 'unsupported';
  }
  const issue = captureIssue(cell, key);
  if (issue) return issue;
  const entry = cell.cmp[key];
  const compareEntry = (cell.cmp || {})[reference];
  if (['missing', 'unsupported', 'error', 'expected'].includes(captureIssue(cell, reference))) return 'uncompared';
  if (sameValue(entry.sig, compareEntry.sig)) return 'pass';
  const intentional = ((cell.tooltip && cell.tooltip.reasons) || []).some(reason => {
    if (reason.intentional !== true) return false;
    const forward = sourceMatches(key, reason.impl) && sourceMatches(reference, reason.reference_impl)
      && entry.sig === reason.candidate_sig && compareEntry.sig === reason.reference_sig;
    const reverse = sourceMatches(reference, reason.impl) && sourceMatches(key, reason.reference_impl)
      && compareEntry.sig === reason.candidate_sig && entry.sig === reason.reference_sig;
    const block = (tooltipCandidate(cell, forward ? key : reference) || {}).block;
    return (forward || reverse) && block && reason.reason === (block.explanation || block.reason);
  });
  return intentional ? 'documented' : 'diverge';
}
