// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
const lower = value => String(value ?? '').toLowerCase();
const compact = value => lower(value).replace(/[^\p{L}\p{N}]/gu, '');
const numericAlias = /^(?:UNIFIED\.)?\d+-\d+$/i;
const unprefix = value => value.replace(/^UNIFIED\./i, '');
const escapeRegex = value => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
const prefixMatch = (value, fields) => {
  const pattern = new RegExp('(?:^|[^\\p{L}\\p{N}])' + escapeRegex(lower(value)), 'u');
  return fields.some(field => pattern.test(lower(field)));
};
const phraseMatch = (value, fields) => {
  const pattern = new RegExp('(?:^|[^\\p{L}\\p{N}])' + escapeRegex(lower(value)) + '(?=$|[^\\p{L}\\p{N}])', 'u');
  return fields.some(field => pattern.test(lower(field)));
};

export function searchContext(tab, families = []) {
  const aliases = new Map();
  for (const column of tab.columns) {
    if (column.display_number == null) continue;
    const alias = `${column.compact_prefix}-${column.display_number}`;
    const stable = unprefix(column.case_id || `${column.compact_prefix}-${column.sub}`);
    if (!numericAlias.test(stable)) aliases.set(lower(alias), stable);
  }
  return {families: [...new Set([...families, ...tab.rows.flatMap(row => [row.family, row.model_label]).filter(Boolean)])], aliases};
}

// Offsets allow URL canonicalization without changing the user's spacing or operators.
function tokenize(query) {
  const tokens = [];
  const pattern = /\s+|"(?:\\.|[^"\\])*"|[()]|-|(?:parser|model|col|description):|[^\s()"]+|"/giy;
  let position = 0;
  while (position < query.length) {
    pattern.lastIndex = position;
    const match = pattern.exec(query);
    if (!match) throw new SyntaxError('Check the search expression.');
    position = pattern.lastIndex;
    if (!/^\s+$/.test(match[0])) tokens.push({value: match[0], start: match.index, end: position});
  }
  return tokens;
}

export function compileSearch(query = '', context = {families: [], aliases: new Map()}) {
  const tokens = tokenize(query), replacements = [], notices = [];
  const families = context.families || [];
  const familyMatch = (value, quoted) => families.some(family => quoted ? phraseMatch(value, [family])
    : Boolean(compact(value)) && compact(family).startsWith(compact(value)));
  let position = 0;
  const peek = () => tokens[position]?.value.toUpperCase();
  const isBoundary = value => !value || ['AND', 'OR', '(', ')', '-'].includes(value) || /^(PARSER|MODEL|COL|DESCRIPTION):$/.test(value);
  function term() {
    if (peek() === '-') { position++; return {type: 'not', child: term()}; }
    if (peek() === '(') {
      position++;
      const node = expression('or');
      if (peek() !== ')') throw new SyntaxError('Close the parenthesis.');
      position++;
      return node;
    }
    let scope;
    if (/^(PARSER|MODEL|COL|DESCRIPTION):$/.test(peek() || '')) scope = lower(tokens[position++].value.slice(0, -1));
    const token = tokens[position];
    if (!token || isBoundary(peek())) throw new SyntaxError('Enter a search term' + (scope ? ` after ${scope}:` : ' after the operator') + '.');
    if (token.value === '"') throw new SyntaxError('Close the quoted phrase.');
    const quoted = token.value.startsWith('"');
    let value = quoted ? token.value.slice(1, -1).replace(/\\(["\\])/g, '$1') : token.value;
    if (!value) throw new SyntaxError('Enter text inside the quotes.');
    position++;
    // Only recognized family names join spaced words; column words remain implicit AND.
    if (!quoted && (!scope || scope === 'parser' || scope === 'model')) {
      let end = position, joined = value, best = familyMatch(value, false) ? {end, value} : null;
      while (end < tokens.length && !isBoundary(tokens[end].value.toUpperCase()) && !tokens[end].value.startsWith('"')) {
        joined += tokens[end++].value;
        if (familyMatch(joined, false)) best = {end, value: joined};
      }
      if (best) { position = best.end; value = best.value; }
    }
    scope = scope === 'model' ? 'parser' : scope;
    if (!scope) scope = familyMatch(value, quoted) ? 'parser' : 'col';
    let stable = false;
    if (scope === 'col' && numericAlias.test(value)) {
      const resolved = context.aliases?.get(lower(unprefix(value)));
      if (resolved) {
        value = resolved;
        replacements.push({start: token.start, end: token.end, value: quoted ? JSON.stringify(value) : value});
      } else notices.push(`Case ${value} is not numbered in this report.`);
      stable = true;
    }
    if (scope === 'col' && /^(?:UNIFIED\.)?\d+-[a-z_]/i.test(value)) stable = true;
    return {type: 'term', scope, value, quoted, stable};
  }
  function expression(operator) {
    const next = operator === 'or' ? () => expression('and') : term;
    const children = [next()];
    while (position < tokens.length && peek() !== ')') {
      if (operator === 'or') {
        if (peek() !== 'OR') break;
        position++;
      } else {
        if (peek() === 'OR') break;
        if (peek() === 'AND') position++;
      }
      children.push(next());
    }
    return children.length === 1 ? children[0] : {type: operator, children};
  }
  const ast = tokens.length ? expression('or') : {type: 'all'};
  if (position < tokens.length) throw new SyntaxError('Check the parentheses.');
  const evaluate = (node, fields, models, descriptions) => {
    if (node.type === 'all') return true;
    if (node.type === 'not') return !evaluate(node.child, fields, models, descriptions);
    if (node.children) return node.type === 'or' ? node.children.some(child => evaluate(child, fields, models, descriptions))
      : node.children.every(child => evaluate(child, fields, models, descriptions));
    const values = node.scope === 'parser' ? models : node.scope === 'description' ? descriptions : fields;
    if (numericAlias.test(node.value)) return false;
    if (node.stable) {
      const identity = value => lower(unprefix(String(value))).replace(/^\d+-/, '');
      return values.some(value => identity(value) === identity(node.value));
    }
    return node.quoted ? phraseMatch(node.value, values) : node.scope === 'parser'
      ? Boolean(compact(node.value)) && values.some(value => compact(value).startsWith(compact(node.value))) : prefixMatch(node.value, values);
  };
  const match = (fields, models = [], descriptions = []) => evaluate(ast, fields.filter(value => value != null && !numericAlias.test(value)), models, descriptions);
  match.ast = ast;
  match.notices = notices;
  match.canonicalQuery = replacements.reverse().reduce((result, item) => result.slice(0, item.start) + item.value + result.slice(item.end), query);
  return match;
}
