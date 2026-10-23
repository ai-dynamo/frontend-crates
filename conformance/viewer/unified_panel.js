// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { tabCandidates, selectedUnifiedFamilies, unifiedDisplayFamilyPredicate } from './candidates.js';
import { explicitSchemas } from './toolcalling.js';
import { materializeUnifiedRecord, canonicalUnifiedLabel } from './unified.js';
import { getField, hasField, isMapping, mappingEntries, mergeMappings, deleteField } from './mappings.js';
import { sameValue, compareText } from './values.js';
import { assembleStream, unifiedChunkDeltas } from './model.js';
import { classifyUnified, equivalenceEntries } from './comparison.js';
import { foldNullVariants, foldSchemaVariants, rankUnifiedColumns } from './folds.js';

const missingCapture = { unavailable: 'Output not yet captured', unavailable_code: 'missing_capture' };

function requestIdentity(request) {
  return { input: getField(request, 'input', ''), init: getField(request, 'init') || { starting_state: 'None', tool_output_mode: 'Native', named_tool: null },
    finish_reason: getField(request, 'finish_reason') || 'stop', tools: getField(request, 'tools'),
    chunks: (getField(request, 'chunks') || []).map((chunk) => Object.fromEntries(mappingEntries(chunk).filter(([key]) => ['delta_text', 'token_ids', 'finish_reason'].includes(key)))) };
}

export function unifiedComparisonFailure(request, captured) {
  const original = getField(captured, 'capture_input');
  if (!original) return 'Original input missing';
  const expected = requestIdentity(request);
  if (isMapping(original) && getField(original, 'tools') == null) return 'Original tool schema missing';
  if (expected.tools == null) return 'Tool schema missing';
  const originalKeys = mappingEntries(original).map(([key]) => key);
  if (!isMapping(original) || originalKeys.some((key) => typeof key !== 'string') || !sameValue(originalKeys.sort(compareText), Object.keys(expected).sort(compareText))) throw new Error('incomplete capture input binding');
  const changed = Object.keys(expected).filter((key) => !sameValue(getField(original, key), expected[key]));
  return changed.length ? `Request mismatch (${changed.join(', ')})` : null;
}

function schemaVariants(families, byFamily, scenario, shared) {
  const variants = [];
  for (const family of families) {
    const input = byFamily.get(family)?.get(scenario)?.input;
    const tools = explicitSchemas(input?.request.tools);
    if (!tools || sameValue(tools, shared)) continue;
    let variant = variants.find((entry) => sameValue(entry.tools, tools));
    if (!variant) { variant = { families: [], tools }; variants.push(variant); }
    variant.families.push(family);
  }
  return variants;
}

export async function unifiedPanel(catalog, reader, { selected = null, reference = undefined } = {}) {
  const metadata = catalog.metadata;
  const spec = metadata.tabs.find((tab) => tab.id === 'tab-unified');
  const descriptors = tabCandidates(catalog, spec);
  const chosen = descriptors.filter((entry) => entry.source.implementation !== 'golden' && (!selected || selected.includes(entry.control.key)));
  const referenceKey = selected === null ? 'dynamo' : reference === undefined ? selected[0] ?? null : reference;
  const referenceDescriptor = descriptors.find(({ control }) => control.key === referenceKey);
  if (referenceKey !== null && !referenceDescriptor) throw new Error(`unknown Unified reference: ${referenceKey}`);
  if (referenceDescriptor && referenceDescriptor.source.implementation !== 'golden' && !chosen.includes(referenceDescriptor)) throw new Error(`reference is not loaded: ${referenceKey}`);
  const byFamily = new Map();
  const descriptions = new Map();
  const histories = new Map();
  const visibleFamilies = selectedUnifiedFamilies(catalog, chosen);
  const capturesFamily = unifiedDisplayFamilyPredicate(catalog, chosen);
  for (const family of [...visibleFamilies].sort(compareText)) {
    const { value: document } = await reader.unifiedInputs(family);
    const cases = new Map();
    const activeAliases = new Map();
    for (const [id, input] of Object.entries(document.cases).sort(([a], [b]) => compareText(a, b))) {
      if (input.lifecycle !== 'active') continue;
      if (cases.has(input.scenario)) throw new Error(`duplicate active Unified scenario: ${family}/${input.scenario}`);
      cases.set(input.scenario, { id, input });
      for (const alias of [input.display_id, ...input.historical_ids]) activeAliases.set(canonicalUnifiedLabel(alias, family, metadata.unified_aliases), { id, input });
      if (!descriptions.has(input.scenario)) descriptions.set(input.scenario, input.description);
    }
    byFamily.set(family, cases);
    for (const descriptor of chosen) {
      const { records } = capturesFamily(descriptor.source, family)
        ? await reader.unified(family, descriptor.source.implementation, descriptor.source.version, document) : { records: new Map() };
      const captures = new Map();
      for (const [id, record] of records) {
        const input = document.cases[id];
        const identity = input.lifecycle === 'active' ? input.display_id : record.value.case_key;
        const current = activeAliases.get(canonicalUnifiedLabel(identity, family, metadata.unified_aliases));
        if (!current) continue;
        const projected = materializeUnifiedRecord(input, record, family, { implementation: descriptor.source.implementation, version: descriptor.source.version, captureMode: metadata.unified_capture_policy[descriptor.source.implementation].mode });
        let value = mappingEntries(getField(projected, 'cases'))[0][1];
        const reason = unifiedComparisonFailure(current.input.request, value);
        if (reason) { value = mergeMappings(value, { unavailable: reason }); deleteField(value, 'error'); deleteField(value, 'unavailable_code'); }
        else if (hasField(value, 'unavailable')) {
          const observation = getField(value, 'capture_observation') || getField(value, 'capture_stimulus');
          const code = getField(getField(value, 'unavailable'), 'code') || getField(value, 'unavailable_code') || getField(getField(observation, 'unavailable'), 'code');
          if (code) value = mergeMappings(value, { unavailable_code: metadata.unavailable_categories?.[code] || code });
        }
        const prior = captures.get(current.input.scenario);
        // Retired aliases can retain an older stimulus under the same display ID.
        // Prefer an exact request binding without weakening real conflict checks.
        if (prior && !prior.bindingFailure && reason) continue;
        const equalRank = prior && Boolean(prior.bindingFailure) === Boolean(reason);
        const conflict = equalRank && (prior.conflict || (!sameValue(prior.value, value)
          && new Error(`conflicting historical aliases in ${descriptor.control.key}: ${family}/${current.id}`)));
        captures.set(current.input.scenario, { value, bindingFailure: reason, conflict, inherited: getField(value, 'inherited_from') });
      }
      // A later exact binding can supersede conflicts among retired stimuli.
      for (const capture of captures.values()) if (capture.conflict) throw capture.conflict;
      histories.set(`${family}\0${descriptor.control.key}`, captures);
    }
  }
  const missingScenarios = metadata.unified_presentation.authored_scenarios.filter(scenario =>
    metadata.unified_presentation.scenario_families[scenario]?.some(family => visibleFamilies.has(family)) && !descriptions.has(scenario));
  if (missingScenarios.length) throw new Error(`Unified fixture inputs are missing authored scenario(s): ${missingScenarios.sort(compareText).join(', ')}`);
  const families = [...byFamily.keys()];
  const scenarios = [...descriptions.keys()];
  for (const scenario of scenarios) if (!Object.hasOwn(metadata.unified_taxonomy, scenario)) throw new Error(`unknown Unified scenario: ${scenario}`);
  const ordered = [...scenarios].sort((a, b) => metadata.unified_taxonomy[a].display_order - metadata.unified_taxonomy[b].display_order);
  const band = (scenario) => { const group = metadata.unified_taxonomy[scenario].display_group; return `case-band-${typeof group === 'number' && group % 2 === 0 ? 1 : 0}`; };
  const shared = explicitSchemas(metadata.unified_presentation.shared_tools) || [];
  const columns = ordered.map((scenario) => {
    const taxonomy = metadata.unified_taxonomy[scenario];
    const column = { sub: scenario, group_key: taxonomy.section, band: band(scenario), label: metadata.folds.unified[scenario].canonical_label, desc: descriptions.get(scenario) || '', init: null };
    const schemas = schemaVariants(families, byFamily, scenario, shared);
    if (schemas.length) column.schemas = schemas;
    return column;
  });
  const groups = [];
  for (const column of columns) {
    let group = groups.find((entry) => entry.key === column.group_key);
    if (!group) { group = { key: column.group_key, label: metadata.unified_taxonomy[column.sub].section_label, band: column.band, span: 0 }; groups.push(group); }
    group.span++;
  }
  const selectedDescriptors = descriptors.filter(descriptor => descriptor.control.key === 'golden' || chosen.includes(descriptor));
  const tooltipOrder = [selectedDescriptors.find((descriptor) => descriptor.control.key === 'golden'), ...['dynamo', 'sglang', 'vllm'].flatMap(engine => selectedDescriptors.filter(descriptor => descriptor.control.impl === engine))];
  const rows = [];
  for (const family of families) {
    const cells = {};
    for (const scenario of scenarios) {
      const entry = byFamily.get(family).get(scenario);
      const taxonomy = metadata.unified_taxonomy[scenario];
      const caseId = `UNIFIED.${metadata.folds.unified[scenario].canonical_label}`;
      const applicable = metadata.unified_presentation.scenario_families[scenario];
      const markerFamily = metadata.unified_presentation.marker_families[family] || family;
      if (!entry && !applicable?.includes(family)) {
        const note = `Not applicable to ${family}`;
        cells[scenario] = { kind: 'cell', case_id: caseId, family, sub: scenario, col_group: taxonomy.section, band: band(scenario), status: 'na', red_on_diff: false,
          cmp: equivalenceEntries(Object.fromEntries(selectedDescriptors.map((descriptor) => [descriptor.control.key, { value: ['absent'], na: 1 }]))), facts: [],
          tooltip: { head: `${caseId} (${scenario}) — ${family}`, description: note, init: null, finish_reason: null,
            input: { kind: null, text: null, chunks: null, family: markerFamily }, candidates: selectedDescriptors.map(({ control }) => ({ key: control.key, label: control.label, impl: control.impl, version: control.version, parse_mode: control.parse_mode, leak: false, block: { unavailable: note, unavailable_code: 'scenario_not_applicable' } })),
            baseline: null, reasons: [], dynamo_notes: [], refs: [], leak_note: null, na_note: note } };
        continue;
      }
      if (!entry) throw new Error(`Unified applicable input missing: ${family}/${scenario}`);
      const input = entry.input;
      const request = input.request;
      const gold = input.golden.assembled;
      const results = new Map();
      const comparisons = { golden: { value: ['events', gold] } };
      for (const descriptor of chosen) {
        const { source, control } = descriptor;
        const history = histories.get(`${family}\0${control.key}`);
        const capture = history.get(scenario);
        let failure = capture ? Object.fromEntries(['unavailable', 'unavailable_code', 'error'].filter((key) => hasField(capture.value, key)).map((key) => [key, getField(capture.value, key)])) : {};
        const isDynamo = control.impl === 'dynamo';
        if (!capturesFamily(source, family)) {
          failure = { unavailable: `No parser for ${family}` };
        } else if (!capture) {
          failure = missingCapture;
        }
        const failed = Object.hasOwn(failure, 'error');
        const hasFailure = Object.keys(failure).length > 0;
        const chunks = capture && !hasFailure ? unifiedChunkDeltas(capture.value) : [];
        const stream = metadata.selection_policy.unified.capture_contracts?.[source.implementation]?.assembled === 'stream';
        const events = !capture || hasFailure ? null : getField(capture.value, 'parser_path') === 'split' ? getField(capture.value, 'assembled', []) : stream ? assembleStream(chunks, source.implementation === 'dynamo_v2' ? getField(capture.value, 'assembled') : null) : getField(capture.value, 'assembled', []);
        const verdict = failed ? 'ERROR' : events === null ? null : classifyUnified(family, gold, events, metadata.unified_leak_markers);
        let block;
        if (events === null) {
          block = hasFailure ? failure : missingCapture;
        } else {
          block = { events, verdict };
          if (isDynamo) {
            if (control.key === 'dynamo') block.todo = null;
            block.explanation = getField(capture.value, 'parser_path') === 'split' ? 'v1 reasoning + v2 tool parsing.' : capture.inherited ? `Inherited from v2 ${capture.inherited}.` : null;
          }
        }
        results.set(control.key, { chunks, events, verdict, failure, block });
        comparisons[control.key] = { value: events !== null ? ['events', events] : failed && (isDynamo || Object.hasOwn(failure, 'error')) ? ['failure'] : ['absent'],
          leak: Number(verdict === 'LEAK'), err: Number(control.key === 'dynamo' ? failed : Object.hasOwn(failure, 'error')),
          na: Number(events === null && !Object.hasOwn(failure, 'error')) };
      }
      const init = getField(request, 'init') || {};
      const chunks = request.chunks.map((chunk, index) => ({ delta_text: getField(chunk, 'delta_text', ''), finish_reason: null,
        expected: Object.fromEntries(chosen.map(({ control }) => [control.key, results.get(control.key).chunks[index] || []])) }));
      const current = results.get('dynamo');
      const reasons = current && ['MERGE', 'ORDER'].includes(current.verdict) ? [{ label: 'stream vs final message', reason: "Final output changes event order or combines separate events." }] : [];
      // Scope applicability stays independent of selection, so an empty Reference
      // keeps every authored variant available in the folded popup.
      cells[scenario] = { kind: 'cell', case_id: caseId, family, sub: scenario, col_group: taxonomy.section, band: band(scenario), status: referenceKey !== null && comparisons[referenceKey]?.err ? 'problem' : 'ok', red_on_diff: true,
        cmp: equivalenceEntries(comparisons), facts: [], tooltip: { head: `${caseId} (${scenario}) — ${family}`, description: input.description + (input.policy.length ? `  [policy: ${input.policy.join(', ')}]` : ''),
          init: { starting_state: 'None', tool_output_mode: 'Native', named_tool: null, ...Object.fromEntries(mappingEntries(init)) }, finish_reason: request.finish_reason || 'stop',
          input: { kind: chunks.length ? 'chunks' : 'text', text: request.input, chunks, family: markerFamily },
          candidates: tooltipOrder.map(({ control }) => control.key === 'golden'
            ? { key: 'golden', label: control.label, impl: 'golden', version: null, parse_mode: 'unified', leak: false, pin_first: true, block: { events: gold } }
            : { key: control.key, label: control.label, impl: control.impl, version: control.version, parse_mode: 'unified', leak: results.get(control.key).verdict === 'LEAK', block: results.get(control.key).block }),
          baseline: null, reasons, dynamo_notes: [], refs: [], leak_note: null, na_note: null } };
    }
    const modelLabel = family;
    rows.push({ family, model_label: modelLabel, model_label_html: modelLabel, section: null, parser: null, cells });
  }
  for (const column of columns) {
    const configs = rows.map((row) => row.cells[column.sub]).filter((cell) => cell.status !== 'na').map((cell) => cell.tooltip.init);
    if (configs.length && configs.every((config) => sameValue(config, configs[0]))) column.init = configs[0];
  }
  const all = rows.flatMap((row) => Object.values(row.cells));
  const na = all.filter((cell) => cell.status === 'na').length;
  const stats = { families: families.length, sub_cases: scenarios.length, slots: all.length, real: all.length - na, parity: 0, dynamo_only: 0, documented: 0, research: 0, errors: 0, na, missing: 0 };
  const glossary = groups.map((group) => ({ label: group.label, rows: ordered.filter((scenario) => metadata.unified_taxonomy[scenario].section === group.key).map((scenario) => [metadata.folds.unified[scenario].canonical_label, descriptions.get(scenario) || '']) }));
  const tab = { id: 'tab-unified', kind: 'unified', mode: 'unified', no_parser_col: true, columns, column_groups: groups, rows, stats, glossary, candidates: descriptors.map((descriptor) => descriptor.control) };
  foldNullVariants(tab, metadata);
  foldSchemaVariants(tab, metadata);
  rankUnifiedColumns(tab, metadata);
  return tab;
}
