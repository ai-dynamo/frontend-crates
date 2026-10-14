// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { sameValue, compareText } from './values.js';
import { setField } from './mappings.js';

export const isApplicableCell = cell => cell.applicable ?? cell.status !== 'na';

export function leafCells(row) {
  const leaves = new Map();
  for (const [sub, cell] of Object.entries(row.cells)) {
    for (const leaf of cell.variants || [cell]) {
      const key = leaf.sub || sub;
      if (leaves.has(key) && !sameValue(leaves.get(key), leaf)) throw new Error(`inconsistent grouped fixture: ${key}`);
      leaves.set(key, leaf);
    }
  }
  return leaves;
}

export function aggregateCells(cells, parent, description) {
  if (!cells.length) throw new Error(`empty fold: ${parent}`);
  const result = structuredClone(cells.find((cell) => cell.case_id) || cells[0]);
  result.sub = cells[0].sub;
  const id = result.case_id || '';
  result.case_id = (id.startsWith('UNIFIED.') ? 'UNIFIED.' : id.split(parent)[0]) + parent;
  result.kind = 'cell';
  result.status = cells.some(isApplicableCell) ? 'ok' : 'na';
  if (cells.some(cell => Object.hasOwn(cell, 'applicable'))) result.applicable = cells.some(isApplicableCell);
  result.red_on_diff = true;
  const applicable = cells.filter(isApplicableCell);
  // Keep comparison identities even when every folded leaf is inapplicable.
  // The aggregate still needs NA entries for selected engines so rendering and
  // status checks can distinguish an unavailable case from a missing engine.
  const keys = [...new Set(cells.flatMap((cell) => Object.keys(cell.cmp || {})))].sort(compareText);
  const signatures = [];
  result.cmp = {};
  for (const key of keys) {
    const entries = applicable.map((cell) => cell.cmp?.[key] || { na: 1 });
    const missing = !applicable.length || entries.some((entry) => entry.na);
    const failure = entries.some((entry, index) => !entry.na && (entry.err || entry.leak || (applicable[index].cmp?.golden && !sameValue(entry.sig, applicable[index].cmp.golden.sig))));
    const signature = entries.map((entry) => entry.na ? null : entry.sig);
    let group = signatures.findIndex((prior) => sameValue(prior, signature));
    if (group < 0) { group = signatures.length; signatures.push(signature); }
    setField(result.cmp, key, { sig: group, na: Number(missing && !failure), err: Number(entries.some((entry) => entry.err)), leak: Number(entries.some((entry) => entry.leak)) });
  }
  result.variants = structuredClone(cells);
  result.tooltip = { head: `${result.case_id} — ${result.family}`, description,
    input: { kind: null }, init: null, candidates: [], variants: result.variants.map((cell) => cell.tooltip) };
  return result;
}

export function foldedSchemas(columns) {
  const variants = [];
  for (const column of columns) for (const variant of column.schemas || []) {
    let merged = variants.find((prior) => sameValue(prior.tools, variant.tools));
    if (!merged) { merged = { tools: variant.tools, families: [], cases: [] }; variants.push(merged); }
    for (const family of variant.families) if (!merged.families.includes(family)) merged.families.push(family);
    merged.cases.push({ label: column.variant_label || column.label, case_id: column.case_id || `UNIFIED.${column.label}`, families: variant.families });
  }
  return variants;
}

function refreshStats(tab, hidden) {
  for (const group of tab.column_groups) group.span = tab.columns.filter((column) => column.group_key === group.key).length;
  tab.column_groups = tab.column_groups.filter((group) => group.span);
  tab.stats.variant_columns = (tab.stats.variant_columns || 0) + hidden.size;
  tab.stats.sub_cases = tab.columns.length;
  const visible = tab.rows.flatMap((row) => Object.values(row.cells));
  tab.stats.fixture_cases = tab.rows.reduce((total, row) => total + [...leafCells(row).values()].filter((cell) => cell.kind === 'cell' && isApplicableCell(cell)).length, 0);
  Object.assign(tab.stats, { slots: visible.length, real: visible.filter((cell) => cell.kind === 'cell' && cell.status !== 'na').length,
    na: visible.filter((cell) => cell.status === 'na').length, missing: visible.filter((cell) => cell.kind === 'missing').length });
}

function removeColumns(tab, hidden) {
  tab.columns = tab.columns.filter((column) => !hidden.has(column.sub));
  for (const row of tab.rows) for (const sub of hidden) delete row.cells[sub];
  refreshStats(tab, hidden);
}

function compareTuple(left, right) {
  for (let index = 0; index < Math.min(left.length, right.length); index++) {
    const a = left[index], b = right[index];
    const difference = Array.isArray(a) ? compareTuple(a, b) : typeof a === 'string' ? compareText(a, b) : a - b;
    if (difference) return difference;
  }
  return left.length - right.length;
}

export function foldNullVariants(tab, metadata) {
  if (!['tab-unified', 'tab-toolcalling-streamv1', 'tab-toolcalling-batch'].includes(tab.id)) return;
  const config = metadata.folds;
  const unified = tab.id === 'tab-unified';
  const identity = (column) => unified ? config.unified[column.sub].legacy_label : column.label;
  const parentOf = (label) => Object.hasOwn(config.null_descriptions, label.split('.')[0]) ? label.split('.')[0] : null;
  const canonical = (label) => unified ? Object.values(config.unified).find((item) => item.legacy_label === label)?.canonical_label || label : label;
  const sorted = (columns) => unified ? columns.sort((a, b) => compareTuple(config.unified[a.sub].taxonomy_order, config.unified[b.sub].taxonomy_order)) : columns;
  const displayLabels = new Map();
  for (const [parent, description] of Object.entries(config.null_descriptions)) {
    const members = sorted(tab.columns.filter((column) => parentOf(identity(column)) === parent));
    if (!members.length) continue;
    const root = members.find((column) => identity(column) === parent) || members[0];
    displayLabels.set(root.label, canonical(parent));
    const mixed = sorted(tab.columns.filter((column) => identity(column).startsWith('7-4.mixed_')));
    const referenced = [...members, ...(parent === '7-5' ? mixed : [])];
    root.desc = description;
    const schemas = foldedSchemas(referenced);
    if (schemas.length) root.schemas = schemas;
    for (const row of tab.rows) {
      if (row.section || !Object.hasOwn(row.cells, root.sub)) continue;
      const children = referenced.filter((column) => !Object.hasOwn(config.mixed_families, identity(column)) || config.mixed_families[identity(column)].includes(row.family))
        .map((column) => row.cells[column.sub]).filter((cell) => cell && ['cell', 'missing'].includes(cell.kind) && isApplicableCell(cell));
      if (children.length && (children.length > 1 || children[0] !== row.cells[root.sub])) {
        row.cells[root.sub] = aggregateCells(children, canonical(parent), description);
        row.cells[root.sub].sub = root.sub;
      }
    }
    for (const column of members) if (column !== root) column.variant_parent = parent;
  }
  const hidden = new Set(tab.columns.filter((column) => Object.hasOwn(column, 'variant_parent')).map((column) => column.sub));
  removeColumns(tab, hidden);
  for (const column of tab.columns) column.label = displayLabels.get(column.label) || column.label;
  const nullParent = (label) => parentOf(unified ? Object.values(config.unified).find((item) => item.canonical_label === label)?.legacy_label || label : label);
  for (const group of tab.glossary || []) group.rows = group.rows.filter(([label]) => !nullParent(label) || displayLabels.has(label))
    .map(([label, description]) => [displayLabels.get(label) || label, config.null_descriptions[nullParent(label)] || description]);
}

export function foldSchemaVariants(tab, metadata) {
  if (tab.id !== 'tab-unified') return;
  const groups = new Map();
  const order = ['Direct', 'Explicit string type', 'allOf', 'Reference'];
  const unionOrder = ['String / Forward', 'Typed value / Forward', 'String / Reversed', 'Typed value / Reversed'];
  for (const column of tab.columns) {
    const fold = metadata.folds.unified[column.sub].schema;
    if (!fold) continue;
    const [parent, label] = fold;
    column.variant_label = label;
    if (!groups.has(parent)) groups.set(parent, []);
    groups.get(parent).push(column);
  }
  const hidden = new Set();
  for (const [parent, members] of groups) {
    const rank = (column) => Math.max(order.indexOf(column.variant_label), unionOrder.indexOf(column.variant_label));
    members.sort((a, b) => rank(a) - rank(b));
    const root = members.find((column) => column.sub === parent) || members[0];
    const labels = new Map(members.map((column) => [column.sub, column.variant_label]));
    const description = parent.includes('const_') ? 'String constant: Direct, Explicit string type, allOf, and Reference.' : 'Union alternatives: String / Typed value x Forward / Reversed branch order.';
    root.desc = description;
    root.schemas = foldedSchemas(members);
    for (const row of tab.rows) {
      if (row.section) continue;
      const children = members.map((column) => row.cells[column.sub]).filter(Boolean);
      if (!children.length) continue;
      const aggregated = aggregateCells(children, metadata.folds.unified[root.sub].canonical_label, description);
      aggregated.sub = root.sub;
      for (const child of aggregated.variants) if (child.tooltip) child.tooltip.head = `${labels.get(child.sub)} — ${child.tooltip.head}`;
      row.cells[root.sub] = aggregated;
    }
    for (const column of members) if (column !== root) hidden.add(column.sub);
  }
  const hiddenLabels = new Set(tab.columns.filter((column) => hidden.has(column.sub)).map((column) => column.label));
  removeColumns(tab, hidden);
  const descriptions = new Map(tab.columns.map((column) => [column.label, column.desc]));
  for (const group of tab.glossary || []) group.rows = group.rows.filter(([label]) => !hiddenLabels.has(label)).map(([label, description]) => [label, descriptions.get(label) || description]);
}

export function rankUnifiedColumns(tab, metadata) {
  if (tab.id !== 'tab-unified') return;
  const counts = new Map();
  const labels = new Map();
  const columns = [...tab.columns].sort((a, b) => metadata.unified_taxonomy[a.sub].display_order - metadata.unified_taxonomy[b.sub].display_order);
  for (const column of columns) {
    const taxonomy = metadata.unified_taxonomy[column.sub];
    const group = taxonomy.display_group;
    counts.set(group, (counts.get(group) || 0) + 1);
    labels.set(column.label, taxonomy.label);
    column.case_id = `UNIFIED.${column.label}`;
    Object.assign(column, { label: taxonomy.label, compact_prefix: taxonomy.compact_prefix, display_number: counts.get(group) });
  }
  for (const group of tab.glossary || []) group.rows = group.rows.map(([label, description]) => [labels.get(label) || label, description]);
}
