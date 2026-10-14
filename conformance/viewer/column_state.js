// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
export function columnGroupKey(label, mode = null) {
  const key = label.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, '');
  return mode ? `${mode}_${key}` : key;
}

export function columnKeysFor(metadata) {
  // Fixed columns are already mounted by the shell; only case groups arrive late.
  const keys = new Set();
  for (const groups of Object.values(metadata.legacy_case_groups || {})) {
    for (const [label] of groups) keys.add(columnGroupKey(label));
  }
  keys.add(columnGroupKey('Other'));
  for (const mode of ['batch', 'stream']) {
    for (const [label, cases] of metadata.reasoning_display?.groups || []) {
      if (cases.some(id => id.startsWith(`${mode}.`))) keys.add(columnGroupKey(label, mode));
    }
    keys.add(columnGroupKey('Other', mode));
  }
  for (const spec of Object.values(metadata.unified_taxonomy)) keys.add(spec.section);
  return [...keys];
}
