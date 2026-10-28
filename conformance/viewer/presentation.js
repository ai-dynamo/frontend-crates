// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { escapeHtml, tabCandidates, selectedUnifiedFamilies, unifiedDisplayFamilyPredicate, unifiedEvents, unifiedChunkDeltas, classifyUnified, canonicalUnifiedEvents, equivalenceEntries, foldNullVariants, foldSchemaVariants, rankUnifiedColumns } from './model.js';
import { isMapping, hasField, getField, compareText, mappingEntries, mergeMappings, deleteField, sameValue } from './values.js';
import { materializeUnifiedRecord, canonicalUnifiedLabel } from './resources.js';

const esc = escapeHtml;
const sourceHref = (path, link) => esc(link ? link(path) : `source:${path.replace(/\/$/, '')}`);


export function tabPresentation(spec, catalog, descriptors, link = null) {
  const href = (path) => link ? link(path) : `source:${path.replace(/\/$/, '')}`;
  const source = (path, label) => `<a href="${esc(href(path))}">${label}</a>`;
  const base = { label: spec.label, captured_note: '', details_note_html: null };
  return { ...base, active: false,
    label_html: 'Unified v2 <span class="tab-sub">(reasoning + tools)</span>', tab_title: 'Unified: one ordered event stream (reasoning + content + tool calls) measured against the GOLDEN oracle',
    case_prefix: 'UNIFIED.', case_section_id: 'unified', case_docs_href: href(catalog.metadata.documentation.unified), case_docs_label: 'lib/parsers/UNIFIED_CASES.md',
    captured_note: 'vLLM captures include every packaged 0.25.1 and 0.26.x parser version. vLLM Rust 0.26.0 uses the native UnifiedParser for Gemma4 and CombinedParser for Qwen3/Kimi K2.',
    toolbar_desc_html: null,
  };
}

export function legendHtml(metadata, link = null) {
  const versionItems = ['vllm_rust', 'vllm_python', 'sglang_python'].filter((impl) => Object.hasOwn(metadata.peer_versions, impl)).map((impl) => `${esc(metadata.implementations.find((item) => item.key === impl).label)} <code>${esc(metadata.peer_versions[impl])}</code>`);
  const versions = versionItems.length ? `<p><strong>Peer parser versions</strong> pinned in <a href="${sourceHref('conformance/utils/src/pyproject.stub.toml', link)}">pyproject.toml</a>: ${versionItems.join(' · ')}.</p>` : '';
  return '<p><strong>Legend:</strong></p><ul class="marker-defs">'
    + '<li><span style="color:#0a7d2c"><strong>green</strong></span> = the selected <strong>Reference</strong> parser output is clean — no structured markup (tool-call or reasoning) leaked into the visible <code>normal_text</code>. A clean Reference is green whether or not any Compare parser is selected.</li>'
    + '<li><span style="color:#b00"><strong>red</strong> (↯)</span> = the Reference parser leaks structured markup (tool-call or reasoning) into the visible <code>normal_text</code>.</li>'
    + '<li><span style="color:#aaa"><strong>n/a</strong></span> = the selected Reference is not applicable for this case (for example the Dynamo v2 stream parser is not implemented for this family).</li>'
    + '<li><span style="color:#8a6d3b">—</span> missing fixture coverage.</li>'
    + '<li>In the <strong>Detailed</strong> view the number on a cell (with a <span style="color:#8a6d3b">Δ</span> suffix, e.g. <span style="color:#8a6d3b">2Δ</span>) = how many selected <strong>Compare</strong> parsers diverge from the Reference (<span style="color:#0a7d2c">=</span> means every selected Compare matches). A divergence with no <code>explanation:</code> yet is flagged <span style="color:#b00">?</span> (research needed); <span style="color:#b00">!</span> marks an engine that errors by design; <span style="color:#b00">✗</span> means the parser ran but failed to parse.</li>'
    + '<li><strong>v1</strong> = the stable batch parser crate (<code>parsers/v1/src/...</code>, <code>dynamo-parsers</code>); <strong>v2</strong> = the WIP streaming parser crate (<code>parsers/v2/src/...</code>, <code>dynamo-parsers-v2</code>).</li>'
    + '<li><span class="parser-suffix">†</span> no vLLM Python peer parser for this family. &nbsp; <span class="parser-suffix">§</span> no SGLang peer parser for this family. &nbsp; <span class="parser-suffix">‡</span> Nemotron V3 (Ultra) reuses the qwen3_coder parser.</li></ul>' + versions;
}

export function columnGroupKey(label, mode = null) {
  const key = label.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, '');
  return mode ? `${mode}_${key}` : key;
}

export function columnKeysFor(metadata) {
  // Fixed columns are already mounted by the shell; only case groups arrive late.
  const keys = new Set();
  for (const spec of Object.values(metadata.unified_taxonomy)) keys.add(spec.section);
  return [...keys];
}

const missingCapture = { unavailable: 'Output not yet captured', unavailable_code: 'missing_capture' };


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
      const comparisons = { golden: { value: ['events', canonicalUnifiedEvents(gold)] } };
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
        const events = !capture || hasFailure ? null : getField(capture.value, 'parser_path') === 'split' ? getField(capture.value, 'assembled', []) : stream ? unifiedEvents(capture.value, {recordedFallback: source.implementation === 'dynamo_v2'}) : getField(capture.value, 'assembled', []);
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
        comparisons[control.key] = { value: events !== null ? ['events', canonicalUnifiedEvents(events)] : failed && (isDynamo || Object.hasOwn(failure, 'error')) ? ['failure'] : ['absent'],
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

export async function buildTab(catalog, reader, tabId, options = {}) {
  const spec = catalog.metadata.tabs.find((tab) => tab.id === tabId);
  if (!spec) throw new Error(`unknown tab: ${tabId}`);
  if (spec.suite !== 'unified') throw new Error(`unsupported report suite: ${spec.suite}`);
  const tab = await unifiedPanel(catalog, reader, options);
  return Object.assign(tab, tabPresentation(spec, catalog, tabCandidates(catalog, spec), options.sourceLink));
}

export function pageMetadata(catalog, { commitSha = null, sourceLink = null } = {}) {
  if (commitSha !== null && !/^[0-9a-f]{40}$/.test(commitSha)) throw new Error('commitSha must be an immutable full SHA');
  const markers = Object.fromEntries(Object.entries(catalog.metadata.markers).map(([family, declaration]) => [family, Object.fromEntries(['pairs', 'singletons', 'opaque'].filter((key) => declaration[key]?.length).map((key) => [key, declaration[key]]))]).filter(([, declaration]) => Object.keys(declaration).length));
  return { schema: 1, meta: { title: 'Dynamo Parser Conformance Table', sha: commitSha, short_sha: commitSha?.slice(0, 12) || '' },
    parser_ni: {}, legend_html: legendHtml(catalog.metadata, sourceLink), family_markers: markers };
}

export async function buildPage(catalog, reader, options = {}) {
  const tabs = [];
  for (const spec of catalog.metadata.tabs) tabs.push(await buildTab(catalog, reader, spec.id, options));
  return { ...pageMetadata(catalog, options), tabs };
}

const sourceAliases = {
  'conformance/toolcalling/fixtures-v1': 'conformance/fixtures/toolcalling/fixtures-batch-v1/inputs.tar.gz',
  'conformance/toolcalling/fixtures-stream-v1': 'conformance/fixtures/toolcalling/fixtures-stream-v1/inputs.tar.gz',
  'conformance/reasoning/fixtures': 'conformance/fixtures/reasoning/fixtures-v1/inputs.tar.gz',
};


export function createSourceLink(baseUrl, publication = null, catalogRevision = null) {
  if (publication) validatePublication(publication, catalogRevision);
  return path => {
    const normalized = path.replace(/\/$/, '');
    if (!normalized || normalized.startsWith('/') || normalized.includes(':') || normalized.includes('\\') || normalized.split('/').some(part => !part || part === '.' || part === '..')) throw new Error(`Invalid source target: ${path}`);
    const target = sourceAliases[normalized] || normalized;
    if (publication) {
      const kind = /\.[a-z0-9]+$/i.test(target) ? 'blob' : 'tree';
      return `https://github.com/ai-dynamo/frontend-crates/${kind}/${publication.revision}/${target}`;
    }
    return new URL(target, baseUrl).href;
  };
}

export function explicitSchemas(tools) {
  if (!Array.isArray(tools)) return null;
  const schemas = tools.filter((tool) => typeof getField(tool, 'name') === 'string' && hasField(tool, 'parameters') && getField(tool, 'parameters') !== null)
    .map((tool) => ({ name: getField(tool, 'name'), parameters: getField(tool, 'parameters') }));
  return schemas.length ? schemas : null;
}

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

export function validatePublication(publication, catalogRevision, resolvedCatalogRevision) {
  if (!publication || publication.schema !== 1 || !/^[0-9a-f]{40}$/.test(publication.revision) || publication.catalog_revision !== catalogRevision
      || (publication.generated_revision !== undefined && !/^[0-9a-f]{40}$/.test(publication.generated_revision))
      || (resolvedCatalogRevision !== undefined && publication.resolved_catalog_revision !== resolvedCatalogRevision)) {
    throw new Error('Publication metadata does not match the resource catalog');
  }
}
