// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { sameValue } from './values.js';

// Matrix paint and status reports consume the same recorded comparison relation.
// The report rejects Reference errors even when the legacy matrix has no concrete
// Compare result with which to show a delta; expose both facts explicitly.
export function selectedCellOutcome({ cmp = {}, base = null, shown = [], redOnDiff = false, family = '', parserNi = {} } = {}) {
  const entry = (key) => Object.hasOwn(cmp, key) ? cmp[key] : null;
  const reference = base ? entry(base) : null;
  const golden = entry('golden');
  const compares = shown.filter((key) => key !== base);
  const available = compares.map(entry).filter((value) => value && value.na !== 1 && value.err !== 1);
  const unavailable = compares.filter((key) => entry(key)?.na === 1).length;
  const signature = golden?.sig != null ? golden.sig : reference?.sig;
  const differences = available.filter((value) => !sameValue(value.sig, signature)).length;
  const result = { className: 'cmp-na', marker: 'n/a', category: 'na', why: '', referenceState: 'missing',
    missingGolden: golden == null, referenceDiffersFromGolden: reference && golden ? !sameValue(reference.sig, golden.sig) : null,
    referenceLeaks: reference?.leak === 1, availableComparisons: available.length, unavailableComparisons: unavailable, comparisonDifferences: differences };
  const ni = base && Object.hasOwn(parserNi, base) ? parserNi[base] : null;
  if (base && ni && !ni.families.includes(family)) {
    return { ...result, marker: 'x', referenceState: 'notImplemented', why: `${ni.label} is not yet implemented for family “${family}”` };
  }
  if (!base) return { ...result, className: 'cmp-nobase', marker: '', referenceState: 'none' };
  if (!reference) return result;
  if (reference.na === 1) return { ...result, referenceState: 'unavailable' };
  if (reference.err === 1) {
    if (redOnDiff) return { ...result, className: 'cmp-leak', marker: '✗', category: 'problem', referenceState: 'error' };
    if (available.length) return { ...result, className: 'cmp-leak', marker: `${available.length}Δ`, category: 'problem', referenceState: 'error' };
    return { ...result, referenceState: 'error' };
  }
  const diverges = redOnDiff && base !== 'golden' && golden?.sig != null && result.referenceDiffersFromGolden;
  const agreement = unavailable ? '?' : '=';
  const marker = redOnDiff ? diverges ? '✗' : differences ? `${differences}Δ` : agreement
    : !available.length ? '' : differences ? `${differences}Δ` : agreement;
  const problem = redOnDiff ? Boolean(diverges) : result.referenceLeaks;
  return { ...result, className: problem ? 'cmp-leak' : 'cmp-eq', marker: (result.referenceLeaks ? '↯' : '') + marker,
    category: problem ? 'problem' : 'ok', referenceState: 'value' };
}
