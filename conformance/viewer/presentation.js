// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { implementationValue } from './comparison.js';
import { escapeHtml } from './candidates.js';
import { getField, isMapping, hasField } from './mappings.js';
import { compareText } from './values.js';

const esc = escapeHtml;
const sourceHref = (path, link) => esc(link ? link(path) : `source:${path.replace(/\/$/, '')}`);

export function unavailablePeers(families, metadata, reasoning = false) {
  const unavailable = { vllm_python: new Set(), sglang_python: new Set() };
  for (const [family, records] of families) {
    const expected = records.map((record) => getField(record, 'expected')).filter(isMapping);
    if (!expected.length) continue;
    for (const implementation of Object.keys(unavailable)) {
      if (reasoning && Object.hasOwn(metadata.reasoning_display.peer_families[implementation], family)) continue;
      if (expected.every((mapping) => isMapping(implementationValue(mapping, implementation, metadata.implementations)) && hasField(implementationValue(mapping, implementation, metadata.implementations), 'unavailable'))) unavailable[implementation].add(family);
    }
  }
  return unavailable;
}

function familySuffix(family, unavailable) {
  return `${unavailable.vllm_python.has(family) ? '†' : ''}${unavailable.sglang_python.has(family) ? '§' : ''}`;
}

function sharedBackend(info, metadata) {
  if (!info?.shared_with.length) return null;
  return metadata.parser_sources.shared_backend_labels.find((item) => item.variant === info.key[0] && item.sub_variant === info.key[1])?.label || null;
}

function toolTree(family, info, backend) {
  const aliasOf = info.alias_of;
  const anchor = aliasOf || family;
  if (!info.shared_with.length && !info.aliases.length && backend === family) return [];
  const families = [anchor, ...info.shared_with].sort(compareText);
  const root = backend === family ? `<strong>${esc(backend)}</strong>` : esc(backend);
  const lines = ['', 'Shared implementation tree:', `${root} (effective parser/backend)`];
  families.forEach((name, index) => {
    const last = index === families.length - 1;
    lines.push(`${last ? '└── ' : '├── '}${name === family && !aliasOf ? `<strong>${esc(name)}</strong>` : esc(name)}`);
    if (name === anchor) info.aliases.forEach((alias, aliasIndex) => {
      lines.push(`${last ? '    ' : '│   '}${aliasIndex === info.aliases.length - 1 ? '└── ' : '├── '}${aliasOf && alias === family ? `<strong>${esc(alias)}</strong>` : esc(alias)} (alias)`);
    });
  });
  return lines;
}

function inheritanceTooltip(family, info, unavailable, metadata, link) {
  const variant = info.variant || '?';
  const backend = sharedBackend(info, metadata) || family;
  const config = `ParserConfig::${variant}${info.sub_variant ? `::${info.sub_variant}` : ''}`;
  const href = sourceHref(`parsers/v1/src/tool_calling/${info.backend_file}`, link);
  const implementation = `${esc(config)} -> <a href="${href}">${esc(info.backend_file)}</a>${info.factory ? esc(` (factory: ${info.factory.split('(')[0]})`) : ''}`;
  const lines = ['Tool calling parser family from fixture YAML.', `Tool calling parser row: ${esc(family)}`, `Effective parser/backend: ${esc(backend)}`, `Dynamo implementation: ${implementation}`];
  if (info.shared_with.length) lines.push(`Shared implementation family: ${esc([info.alias_of || family, ...info.shared_with].sort(compareText).join(', '))}`);
  if (info.alias_of) lines.push(`Alias of: ${esc(info.alias_of)}`);
  if (info.aliases.length) lines.push(`Registered aliases: ${esc(info.aliases.join(', '))}`);
  const peers = [];
  if (unavailable.vllm_python.has(family)) peers.push('no vLLM Python peer parser');
  if (unavailable.sglang_python.has(family)) peers.push('no SGLang peer parser');
  if (peers.length) lines.push(`Peer availability: ${peers.join(', ')}`);
  if (info.filed_under_xml_misleading) lines.push(`Note: filed under xml/ but does not use the shared xml::parser; it has its own ParserConfig::${esc(variant)} variant.`);
  const tree = toolTree(family, info, backend);
  if (tree.length) lines.push(`<span class="ttip-tree">${tree.join('\n').replace(/^\n|\n$/g, '')}</span>`);
  const head = backend === family ? `\`${family}\`` : `\`${backend}\` (row: \`${family}\`)`;
  return `<div class="ttip"><div class="ttip-head">${esc(head)}</div><pre class="ttip-pre">${lines.join('\n').trimEnd()}</pre></div>`;
}

function streamParserCell(label, family, backend, file, entrypoint, fixtures, note, link) {
  const href = sourceHref(`parsers/v2/src/tool_calling/${file}`, link);
  const tooltip = `<div class="ttip"><div class="ttip-head">\`${esc(family)}\` (v2 stream)</div><pre class="ttip-pre ttip-note">Fixtures: ${esc(fixtures)}.\nTool calling parser row: ${esc(family)}\nEffective parser/backend: ${esc(backend)}\nDynamo parser v2 implementation: parsers/v2/src/tool_calling/${esc(file)} -> <a href="${href}">${esc(entrypoint)}</a>\nNote: ${esc(note)}</pre></div>`;
  return `<td class="parser" data-col-hide-group="parser"><a href="${href}">${label}</a>${tooltip}</td>`;
}

export function toolParserCell(family, mode, unavailable, metadata, link = null) {
  const suffix = familySuffix(family, unavailable);
  const label = esc(family) + (suffix ? `<span class="parser-suffix">${esc(suffix)}</span>` : '');
  if (family === 'harmony' && mode === 'streamv1') return streamParserCell(label, family, 'HarmonyToolStreamParser token-id path', 'harmony.rs', 'parse_tool_call_streaming_incremental', 'legacy stream fixtures', 'TC stream token-id row. It consumes `delta_token_ids` from legacy, non-Unified stream fixtures directly.', link);
  if (family === 'harmony_text') return streamParserCell(label, family, 'HarmonyToolStreamParser text path', 'harmony.rs', 'parse_tool_call_streaming_text', 'legacy stream fixtures', 'Synthetic v2 row for gpt-oss text streaming. The text path re-tokenizes a held suffix, then feeds the same token-incremental Harmony stream parser used by the token-id row.', link);
  if (mode === 'streamv1') {
    if (family === 'deepseek_v4') return streamParserCell(label, family, 'DeepSeekV4ToolStreamParser text path', 'dsml.rs', 'push', 'legacy stream fixtures', 'TC stream row. It consumes DSML text chunks and emits compact complete-invoke deltas.', link);
    const info = metadata.parser_sources.stream_cells[family];
    if (info) return streamParserCell(label, family, info[0], info[1], 'push', 'legacy stream fixtures', `TC stream row. It consumes ${info[2]} text chunks and emits per-chunk tool-call deltas.`, link);
    return `<td class="parser" data-col-hide-group="parser">${esc(family)}<div class="ttip"><div class="ttip-head">\`${esc(family)}\` (v2 stream)</div><pre class="ttip-pre ttip-note">Dynamo parser v2 is not implemented for this family yet.\nThis row is inventory only; no v1 parser code runs on this tab.</pre></div></td>`;
  }
  const info = metadata.parser_sources.inheritance[family];
  const ref = metadata.parser_sources.references[family];
  const tooltip = info ? inheritanceTooltip(family, info, unavailable, metadata, link) : '';
  const short = sharedBackend(info, metadata);
  const text = short ? esc(short) : label;
  const base = short ? `<span class="parser-base">→ ${label}</span>` : '';
  const path = info && info.backend_file !== 'unknown' ? info.backend_file : ref ? ref[0] : null;
  const anchor = path ? `<a href="${sourceHref(`parsers/v1/src/tool_calling/${path}`, link)}">${text}</a>` : text;
  return `<td class="parser" data-col-hide-group="parser">${anchor}${base}${tooltip}</td>`;
}

function reasoningImplementation(family, metadata) {
  if (!family) return 'n/a';
  const implementation = metadata.reasoning_display.families[family]?.implementation || family;
  const name = implementation.split(' ')[0];
  const marker = implementation.match(/`([^`]+)`/);
  return name === 'BasicReasoningParser' && marker ? `${name} ${marker[1]}` : name;
}

export function reasoningParserCell(toolFamily, family, unavailable, metadata) {
  const config = metadata.reasoning_display;
  const info = config.families[family] || {};
  const displayFamily = toolFamily === 'glm47' && family === 'nemotron_deci' ? 'glm45' : family;
  const implementation = reasoningImplementation(family, metadata);
  const mode = config.modes[family] || { label: family ? 'custom' : 'n/a', control: family ? 'mostly static' : 'not configured', static: [] };
  const lines = [esc(`This is activated via "--dyn-reasoning-parser ${displayFamily || 'n/a'}".`)];
  if (displayFamily) lines.push(`Parser family: ${esc(displayFamily)}`);
  if (toolFamily && toolFamily !== displayFamily) lines.push(`Tool calling row: ${esc(toolFamily)}`);
  if (displayFamily && displayFamily !== family) lines.push(`Fixture family: ${esc(family || '')}`);
  if (!displayFamily) lines.push('Reasoning parser family: n/a');
  if (info.models) lines.push(`Models: ${esc(info.models.join(', '))}`);
  if (info.rust_enum) lines.push(`Rust enum: ${esc(info.rust_enum)}`);
  if (info.implementation) lines.push(`Implementation: ${esc(info.implementation)}`);
  const peers = [];
  if (unavailable.vllm_python.has(family)) peers.push('no vLLM peer reasoning parser');
  if (unavailable.sglang_python.has(family)) peers.push('no SGLang peer reasoning parser');
  if (peers.length) lines.push(`Peer availability: ${esc(peers.join(', '))}`);
  lines.push('', 'Mode:', `- ${esc(`${mode.label} / ${mode.control}`)}`);
  const configs = (mode.static || []).filter((line) => !/^(?:BasicReasoningParser|Gemma4ReasoningParser|GptOssReasoningParser|GraniteReasoningParser|MiniMaxAppendThinkParser)/.test(line));
  if (configs.length) lines.push('', 'Config:', ...configs.map((line) => `- ${esc(line)}`));
  if (family) {
    const shared = (info.shared_with || []).filter((name) => name !== family);
    const aliases = (info.aliases || []).filter((name) => name !== family && !shared.includes(name));
    const other = toolFamily && toolFamily !== family && ![...shared, ...aliases].includes(toolFamily) ? [toolFamily] : [];
    const children = [[family, ''], ...other.sort(compareText).map((name) => [name, '']), ...shared.sort(compareText).map((name) => [name, '']), ...aliases.sort(compareText).map((name) => [name, ' (alias)'])];
    lines.push('', 'Shared implementation tree:', esc(implementation));
    children.forEach(([name, suffix], index) => lines.push(`${index === children.length - 1 ? '└── ' : '├── '}${toolFamily === name || !toolFamily && name === family ? `<strong>${esc(name)}</strong>` : esc(name)}${esc(suffix)}`));
  }
  const suffix = familySuffix(family, unavailable);
  const tooltip = `<div class="ttip-head">${esc(family ? `\`${implementation}\`${suffix}` : 'n/a')}</div><pre class="ttip-pre">${lines.join('\n')}</pre>`;
  return `<td class="parser" data-col-hide-group="parser"><code>${esc(implementation)}</code>${suffix ? `<span class="parser-suffix">${esc(suffix)}</span>` : ''}<div class="ttip">${tooltip}</div></td>`;
}

export function tabPresentation(spec, catalog, descriptors, link = null) {
  const href = (path) => link ? link(path) : `source:${path.replace(/\/$/, '')}`;
  const source = (path, label) => `<a href="${esc(href(path))}">${label}</a>`;
  const base = { label: spec.label, captured_note: '', details_note_html: null };
  if (spec.id === 'tab-toolcalling-batch') return { ...base, active: true,
    label_html: 'Tool Calling v1 <span class="tab-sub">(<span class="w-batch">batch</span> data)</span>',
    tab_title: 'Tool Calling (batch data): v1 batch parsers plus v2 stream parsers on the same v1 batch fixtures',
    case_prefix: 'TOOLCALLING.batch.', case_section_id: 'toolcalling-batch', case_docs_href: href(catalog.metadata.documentation.toolcalling), case_docs_label: 'lib/parsers/TOOLCALLING_CASES.md',
    toolbar_desc_html: `Parsers: <strong>v1</strong> batch (${source('parsers/v1/src/tool_calling/', 'parsers/src/tool_calling/')}) plus <strong>v2</strong> streaming on the same batch text (${source('parsers/v2/src/tool_calling/', 'parsers_v2/src/tool_calling/*')}) · Input: <strong>v1</strong> batch fixtures (${source('conformance/toolcalling/fixtures-v1/', 'conformance/fixtures/toolcalling/fixtures-batch-v1/')}).`,
  };
  if (spec.id === 'tab-toolcalling-streamv1') {
    const versions = new Map();
    for (const descriptor of descriptors) {
      if (descriptor.source.implementation === 'golden') continue;
      if (!versions.has(descriptor.source.implementation)) versions.set(descriptor.source.implementation, []);
      versions.get(descriptor.source.implementation).push(descriptor.source.version);
    }
    const captures = [...versions].sort(([a], [b]) => compareText(a, b)).map(([impl, values]) => {
      // Controls already run newest first within an implementation.
      const oldest = catalog.metadata.selection_policy.stream.inventory_defaults?.[impl] === 'oldest';
      return `${impl} ${oldest ? values.at(-1) : values[0]}`;
    });
    return { ...base, label_html: 'Tool Calling legacy stream <span class="tab-sub">(<span class="w-stream">stream</span> data)</span>',
      tab_title: 'Tool Calling (legacy stream corpus): Dynamo parser v2', case_prefix: 'TOOLCALLING.streamv1.', case_section_id: 'toolcalling-streamv1', case_docs_href: href(catalog.metadata.documentation.stream), case_docs_label: 'lib/parsers/TOOLCALLING_STREAMING_V1_CASES.md',
      captured_note: `Peer streaming output captured against: ${captures.join(', ')}. A divergence is relative to these versions; re-capture when bumping.`,
      toolbar_desc_html: `Parser: <strong>v2</strong> Dynamo parser v2 token-incremental streaming (${source('parsers/v2/src/tool_calling/', 'parsers_v2/src/tool_calling/*')}) · Input: <strong>legacy v1</strong> non-Unified stream fixtures (${source('conformance/toolcalling/fixtures-stream-v1/', 'conformance/fixtures/toolcalling/fixtures-stream-v1/')}).`,
      details_note_html: "<p>Red means that engine's stream parser diverges from its batch parser. A <code>≠</code> corner mark is a KNOWN v1-vs-v2 divergence (known-divergences.yaml): calls agree, normal_text differs by design — the popup's v2 block carries the explanation. There is no <code>V_rb</code>; vLLM Rust has stream parser capture only. Harmony captured against vLLM 0.23.0 / SGLang 0.5.12.post1.</p>",
    };
  }
  if (spec.suite === 'reasoning') return { ...base, active: false,
    label_html: `Reasoning v1 <span class="tab-sub">(<span class="w-${spec.mode}">${spec.mode}</span> data)</span>`, tab_title: `Reasoning ${spec.mode}: v1 code on v1 fixtures`,
    case_prefix: 'REASONING.', case_section_id: `reasoning-${spec.mode}`, case_docs_href: href(catalog.metadata.documentation.reasoning), case_docs_label: 'lib/parsers/REASONING_CASES.md',
    toolbar_desc_html: `Parser: <strong>v1</strong> reasoning parser (${source('parsers/v1/src/reasoning/', 'parsers/v1/src/reasoning/')}) · Input: <strong>v1</strong> reasoning fixtures (${source('conformance/reasoning/fixtures/', 'conformance/fixtures/reasoning/fixtures-v1/')}).`,
  };
  return { ...base, active: false,
    label_html: 'Unified v2 <span class="tab-sub">(reasoning + tools)</span>', tab_title: 'Unified: one ordered event stream (reasoning + content + tool calls) measured against the GOLDEN oracle',
    case_prefix: 'UNIFIED.', case_section_id: 'unified', case_docs_href: href(catalog.metadata.documentation.unified), case_docs_label: 'lib/parsers/UNIFIED_CASES.md',
    captured_note: 'vLLM captures include every packaged 0.25.1 and 0.26.x parser version. vLLM Rust 0.26.0 uses the native UnifiedParser for Gemma4 and CombinedParser for Qwen3/Kimi K2.',
    toolbar_desc_html: 'Oracle = <strong>GOLDEN</strong> (authored, best-effort recovery) · Default Reference = <strong>Dynamo v2 Rust</strong>. Compare with historical Dynamo and captured peer versions. A parser that does not exist for a family is shown as n/a. A cell is red when the REF — the starred engine (default Dynamo) — DIVERGES from golden in any class: leaked markup (↯), merged/reordered events, or dropped content (✗). A green cell means the REF matches golden exactly; the NΔ count is how many Compare-with engines diverge (informational — they never color the cell). Star a different engine to judge it instead.',
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
