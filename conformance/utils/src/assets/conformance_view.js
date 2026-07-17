// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Conformance page VIEW (DIS-2434): build the tabs + table + compare bar + popups
// from the inlined JSON model (`<script id="conformance-model">`). This runs at
// parse time, BEFORE conformance.js, and produces the exact DOM (classes/attrs)
// that conformance.js queries — so the interactivity script wires against it
// unmodified. If the model blob is absent, we leave the server-rendered DOM in
// place (phases 1-2 still server-render the panels; we swap them for model-built
// ones so the two stay byte-compatible from conformance.js's point of view).
(function () {
  // --- Escaping helpers ------------------------------------------------------
  // Every plain-text value from the model is escaped before insertion. Fields
  // whose name ends in `_html` (label_html, model_label_html, legend_html,
  // toolbar_desc_html, parser.html, candidate.label_html) are already-safe HTML
  // produced by Python and are inserted verbatim.
  function escapeHtml(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }
  function escapeAttr(s) { return escapeHtml(s); }
  function num(x) { return String(x == null ? 0 : x); }

  var URL_RE = /(https?:\/\/[^\s<>"']+)/g;
  function refText(v) {
    if (v == null) { return ''; }
    if (Object.prototype.toString.call(v) === '[object Array]') {
      return v.map(function (x) { return String(x); }).join('\n');
    }
    if (typeof v === 'object') { return JSON.stringify(v); }
    return String(v);
  }
  // Escape text and turn embedded URLs into anchors (mirrors common.linkify_text_html).
  function linkify(v) {
    var text = refText(v);
    var out = '';
    var last = 0;
    var m;
    URL_RE.lastIndex = 0;
    while ((m = URL_RE.exec(text)) !== null) {
      out += escapeHtml(text.slice(last, m.index));
      var url = m[0];
      out += '<a href="' + escapeAttr(url) + '" target="_blank" rel="noopener noreferrer">'
        + escapeHtml(url) + '</a>';
      last = m.index + url.length;
    }
    out += escapeHtml(text.slice(last));
    return out;
  }

  // COLORIZED: the server colorizes tool-call markup + whitespace (markup.py's
  // colorize_markup / _mark_ws). First-draft: HTML-escape only, never crash.
  /* TODO: port colorize_markup whitespace chips + markup coloring from markup.py */
  function colorize(text) { return escapeHtml(text == null ? '' : String(text)); }

  // --- Lazy tooltip building -------------------------------------------------
  // conformance.js's attachTooltip queries `cell.querySelector('.ttip')` at wire
  // time, so every popup cell needs a `.ttip` element present up front — but the
  // expensive CONTENT (per-candidate output blocks, per-chunk chart) is deferred:
  // we attach an EMPTY `.ttip` now and populate it on first interaction. The cell
  // keeps its raw tooltip model on `_ttipModel` until then.
  // The tooltip model lives in a JS map keyed by a `data-ttip-id` ATTRIBUTE (which
  // survives cloneNode), NOT a JS property (which does not) — so transpose clones
  // (built by conformance.js via cloneNode) can still build their tooltip. One
  // DELEGATED document listener handles every cell (originals + clones) instead of
  // thousands of per-cell listeners.
  var _ttipModels = {};
  var _ttipSeq = 0;
  // Whether this page carries a reference-aware not-implemented map (v2 only); gates
  // `data-family` emission so the v1 overview count matches its server render.
  var _usesFamily = false;
  // Per-page summary legend (the "green(N)…" strip under the grid); each generator
  // supplies its own HTML so the legacy v1 page keeps its "match Base" wording.
  var _summaryLegendHtml = null;

  function registerTooltip(td, model) {
    var id = String(++_ttipSeq);
    td.setAttribute('data-ttip-id', id);
    _ttipModels[id] = model;
  }

  function buildTooltipInto(td) {
    if (!td) { return; }
    var ttip = td.querySelector('.ttip');
    if (!ttip || ttip.classList.contains('ttip-built')) { return; }
    // A clone of an ALREADY-built tooltip carries the built content + this class, so
    // it's skipped above. A fresh (unbuilt) cell/clone builds from the keyed model.
    var m = _ttipModels[td.getAttribute('data-ttip-id')];
    ttip.classList.add('ttip-built');
    if (!m) { return; }
    // Append (don't overwrite): conformance.js may already have inserted its
    // `.ttip-close` button / `.cmp-why` line into the empty `.ttip`; keep them.
    ttip.insertAdjacentHTML('beforeend', buildTooltipHtml(m));
    // The freshly-built `.cand` sections + chart columns start hidden — CSS shows
    // them only with `.cand-on`/visible-column, which conformance.js's toggleCands
    // adds during applyCtl. Nudge applyCtl so the current selection is reflected.
    refreshCompare(td);
  }
  // Expose so conformance.js (or tests) can force-build a specific cell if needed.
  window.__buildTooltip = buildTooltipInto;

  function delegatedBuild(e) {
    var t = e.target;
    var td = t && t.closest ? t.closest('td.cell[data-ttip-id]') : null;
    if (td) { buildTooltipInto(td); }
  }
  // pointerover/focusin/click all bubble, so one document listener covers every
  // cell (and any cloned mirror cell) with no per-cell wiring.
  document.addEventListener('pointerover', delegatedBuild, false);
  document.addEventListener('focusin', delegatedBuild, false);
  document.addEventListener('click', delegatedBuild, false);

  // Re-run the panel's compare/coloring pass by dispatching a synthetic `change`
  // on its `.cmpctl` — conformance.js's delegated listener calls applyCtl, which
  // re-applies cand visibility to every cell (including our just-built tooltip).
  function refreshCompare(td) {
    var panel = td.closest ? td.closest('.tab-panel') : null;
    if (!panel) { return; }
    var ctl = panel.querySelector('.cmpctl');
    if (!ctl) { return; }
    var box = ctl.querySelector('input.cmp-on');
    if (!box) { return; }
    var ev;
    try {
      ev = new Event('change', { bubbles: true });
    } catch (e) {
      ev = document.createEvent('Event');
      ev.initEvent('change', true, false);
    }
    box.dispatchEvent(ev);
  }

  // --- Output block rendering (mirrors _format_output_block_html) ------------
  function outputBlock(b) {
    if (!b) { return '—'; }
    if (b.unavailable != null) {
      return 'unavailable: ' + escapeHtml(String(b.unavailable));
    }
    if (b.error != null) {
      var e = (typeof b.error === 'string') ? b.error : JSON.stringify(b.error);
      return 'error: ' + escapeHtml(e);
    }
    var out;
    if (b.reasoning_text != null) {
      // Reasoning cell: reasoning_text + normal_text (no tool calls).
      out = '<span class="fldl">reasoning_text=\'</span>' + colorize(b.reasoning_text)
        + '<span class="fldl">\'</span>'
        + '\n<span class="fldl">normal_text=\'</span>' + colorize(b.normal_text || '')
        + '<span class="fldl">\'</span>';
    } else {
      var nt = b.normal_text || '';
      var calls = b.calls || [];
      out = '<span class="fldl">normal_text=\'</span>' + colorize(nt)
        + '<span class="fldl">\'</span>'
        + '\n<span class="fldl">calls=</span>' + escapeHtml(JSON.stringify(calls));
    }
    if (b.explanation) {
      out += '\n<span class="expl">explanation: ' + escapeHtml(String(b.explanation)) + '</span>';
    }
    return out;
  }

  // --- Candidate chart (the popup grid) --------------------------------------
  // Columns are keyed by `data-cand` = the compare-bar candidate key, so
  // conformance.js's toggleCands shows/hides/REF-orders them for free. The chart
  // is the SINGLE per-candidate output surface (the legacy per-candidate `.cand`
  // list is never emitted alongside it — chart XOR list). Two shapes:
  //   text input  -> one `ttip-final` row: input_text + each candidate's assembled block
  //   chunk input -> one row per input chunk (per-candidate deltas) + a `ttip-final`
  //                  assembled row.
  var _IMPL_KEYS = ['dynamo_v1', 'dynamo_v2', 'vllm_rust', 'vllm_python', 'sglang_python'];
  function implKeyOf(candKey) {
    for (var i = 0; i < _IMPL_KEYS.length; i++) {
      if (candKey.indexOf(_IMPL_KEYS[i]) === 0) { return _IMPL_KEYS[i]; }
    }
    return candKey;
  }

  // One candidate's emitted deltas at one chunk, as compact text (mirrors the
  // server's _render_chunk_deltas: name deltas as name='<n>', arg fragments joined).
  function renderDeltas(deltas) {
    if (!deltas || !deltas.length) { return '<span class="parser-base">—</span>'; }
    var parts = [];
    var args = '';
    deltas.forEach(function (d) {
      if (d == null) { return; }
      if (d.name != null) { parts.push("name='" + escapeHtml(String(d.name)) + "'"); }
      if (d.arguments != null) { args += String(d.arguments); }
    });
    if (args) { parts.push("args='" + escapeHtml(args) + "'"); }
    return parts.length ? parts.join(' ') : '<span class="parser-base">—</span>';
  }

  function inputTextCell(input) {
    if (input && input.text != null) {
      return '<span class="fldl">input_text=\'</span>' + colorize(input.text)
        + '<span class="fldl">\'</span>';
    }
    return '';
  }

  function buildChartHtml(m) {
    var cands = m.candidates || [];
    if (!cands.length) { return ''; }
    var input = m.input || { kind: null };
    var header = '';
    cands.forEach(function (c) {
      header += '<th data-cand="' + escapeAttr(c.key) + '">' + escapeHtml(c.label) + '</th>';
    });
    var body = '';
    if (input.kind === 'chunks' && input.chunks && input.chunks.length) {
      input.chunks.forEach(function (ch, i) {
        var row = '<tr><td class="cin">' + colorize(ch.delta_text || '');
        if (ch.finish_reason) {
          row += '<span class="fr"> finish=' + escapeHtml(String(ch.finish_reason)) + '</span>';
        }
        row += '</td>';
        cands.forEach(function (c) {
          var impl = implKeyOf(c.key);
          var d = (ch.expected && ch.expected[impl]) || [];
          row += '<td data-cand="' + escapeAttr(c.key) + '">' + renderDeltas(d) + '</td>';
        });
        body += row + '</tr>';
      });
    }
    // Assembled row: each candidate's final block, compared against the input.
    var fin = '<tr class="ttip-final"><td class="cin">'
      + (body ? 'assembled' : inputTextCell(input)) + '</td>';
    cands.forEach(function (c) {
      fin += '<td data-cand="' + escapeAttr(c.key) + '">'
        + outputBlock(c.block).replace(/\n/g, '<br>') + '</td>';
    });
    fin += '</tr>';
    // The table carries class `ttip-chunks` — conformance.js keys the popup grid on it.
    return '<table class="ttip-chunks"><thead><tr><th>input</th>' + header
      + '</tr></thead><tbody>' + body + fin + '</tbody></table>';
  }

  // --- Tooltip content (built lazily into the empty .ttip) -------------------
  function buildTooltipHtml(m) {
    var h = '';
    if (m.head) { h += '<div class="ttip-head">' + escapeHtml(m.head) + '</div>'; }
    var cands = m.candidates || [];
    var chart = cands.length ? buildChartHtml(m) : '';
    // Description shown only when there's no chart (the chart's input cell carries it).
    if (m.description && !chart) {
      h += '<pre class="ttip-pre">' + escapeHtml(m.description) + '</pre>';
    }
    if (chart) {
      // Chart is the per-candidate output surface — NEVER also emit the `.cand`
      // list (test_chart_tooltips_have_no_candidate_list: chart XOR list).
      h += chart;
    } else {
      var input = m.input || { kind: null };
      if (input.kind === 'text' && input.text) {
        h += '<div class="ttip-section">Input:</div>'
          + '<pre class="ttip-pre">' + colorize(input.text) + '</pre>';
      }
    }
    // Divergence reasons (structured).
    (m.reasons || []).forEach(function (r) {
      h += '<div class="ttip-reason">' + escapeHtml(r.label) + ': ' + escapeHtml(r.reason) + '</div>';
    });
    // Provenance refs ([label, value] pairs; values may contain URLs).
    (m.refs || []).forEach(function (pair) {
      h += '<div class="ttip-ref">' + escapeHtml(pair[0]) + ': ' + linkify(pair[1]) + '</div>';
    });
    // n/a-stub explanation.
    if (m.na_note) {
      h += '<pre class="ttip-pre">' + escapeHtml(m.na_note) + '</pre>';
    }
    return h;
  }

  // --- Compare bar (mirrors _compare_bar.html.j2) ----------------------------
  // Three engine columns (Dynamo / vLLM / SGLang); one .cmprow per candidate whose
  // impl matches. Bucket A = reference (radio checked, own compare box locked-on);
  // bucket A or B = compare-with checked; C = off.
  function compareBarHtml(tab) {
    var groups = [['Dynamo', 'dynamo'], ['vLLM', 'vllm'], ['SGLang', 'sglang']];
    var html = '<div class="cmpctl" role="group" aria-label="Pick one Reference parser'
      + ' (radio) and any number of Compare-with parsers (checkbox)">';
    groups.forEach(function (g) {
      var engine = g[0];
      var impl = g[1];
      html += '<div class="cmpcol" data-engine="' + escapeAttr(engine) + '">'
        + '<div class="cmprow cmphd"><span class="cmphd-ref" title="Reference (pick one)">ref</span>'
        + '<span class="cmphd-cmp" title="Compare-with">compare with</span></div>';
      (tab.candidates || []).forEach(function (c) {
        if (c.impl !== impl) { return; }
        var isA = c.default_bucket === 'A';
        var isAB = c.default_bucket === 'A' || c.default_bucket === 'B';
        html += '<div class="cmprow' + (isA ? ' is-ref' : '') + '">'
          + '<label class="cmprow-ref" title="Set as the Reference parser">'
          + '<input type="radio" class="cmp-ref" name="ref_' + escapeAttr(tab.id) + '"'
          + ' value="' + escapeAttr(c.key) + '"' + (isA ? ' checked' : '') + '>'
          + '<span class="star" aria-hidden="true">★</span></label>'
          + '<label class="cmprow-cmp" title="Include in Compare-with">'
          + '<input type="checkbox" class="cmp-on" value="' + escapeAttr(c.key) + '"'
          + (isAB ? ' checked' : '') + (isA ? ' disabled' : '') + '></label>'
          + '<span class="cmprow-label" data-cand="' + escapeAttr(c.key) + '">'
          + (c.label_html || escapeHtml(c.label || '')) + '</span>'
          + '</div>';
      });
      html += '</div>';
    });
    html += '</div>';
    return html;
  }

  // --- Column headers (mirror _column_control_header_html / _subcase_headers) --
  function fixedColumnHeader(key, label) {
    return '<th class="column-control" data-col-control-group="' + escapeAttr(key) + '" rowspan="2">'
      + '<button type="button" class="col-toggle" data-col-toggle="' + escapeAttr(key) + '"'
      + ' data-col-label="' + escapeAttr(label) + '" data-col-span="1" data-default-visible="true"'
      + ' aria-pressed="true" aria-label="Collapse ' + escapeAttr(label) + ' column">'
      + '<span class="col-toggle-symbol" aria-hidden="true"></span>'
      + '<span class="col-toggle-label">' + escapeHtml(label) + '</span></button></th>';
  }
  function groupColumnHeader(g) {
    return '<th class="column-control case-group ' + escapeAttr(g.band) + '"'
      + ' data-col-control-group="' + escapeAttr(g.key) + '" colspan="' + g.span + '"'
      + ' data-expanded-colspan="' + g.span + '">'
      + '<button type="button" class="col-toggle" data-col-toggle="' + escapeAttr(g.key) + '"'
      + ' data-col-label="' + escapeAttr(g.label) + '" data-col-span="' + g.span + '"'
      + ' data-default-visible="true" aria-pressed="true"'
      + ' aria-label="Collapse ' + escapeAttr(g.label) + ' column">'
      + '<span class="col-toggle-symbol" aria-hidden="true"></span>'
      + '<span class="col-toggle-label">' + escapeHtml(g.label) + '</span></button></th>';
  }
  function groupHeadersHtml(tab) {
    // The parser column is labeled per tab kind: "Tool calling family" vs
    // "Reasoning family" (matches _subcase_group_headers_html / _case_group_headers_html).
    var parserLabel = tab.kind === 'reasoning' ? 'Reasoning family' : 'Tool calling family';
    var h = fixedColumnHeader('model', 'Model') + fixedColumnHeader('parser', parserLabel);
    (tab.column_groups || []).forEach(function (g) { h += groupColumnHeader(g); });
    return h;
  }
  function subHeadersHtml(tab) {
    var cols = tab.columns || [];
    var href = escapeAttr(tab.case_docs_href || '');
    var h = '';
    for (var i = 0; i < cols.length; i++) {
      var c = cols[i];
      // Server always emits a title attr (empty string when no description).
      h += '<th class="case-sub ' + escapeAttr(c.band) + '" data-col-hide-group="'
        + escapeAttr(c.group_key) + '"><a href="' + href + '" title="' + escapeAttr(c.desc || '')
        + '">' + escapeHtml(c.label) + '</a></th>';
      // A hidden placeholder cell closes each contiguous group run.
      var next = cols[i + 1];
      if (!next || next.group_key !== c.group_key) {
        h += '<th class="col-placeholder col-hidden" data-col-placeholder-group="'
          + escapeAttr(c.group_key) + '"></th>';
      }
    }
    return h;
  }

  // --- Body rows + cells -----------------------------------------------------
  function placeholderTd(group) {
    var td = document.createElement('td');
    td.className = 'col-placeholder col-hidden';
    td.setAttribute('data-col-placeholder-group', group == null ? '' : group);
    return td;
  }

  // Parse a full `<td class="parser">...</td>` string (row.parser.html) into a
  // node — wrap in a table so the td parses in a legal context. Its own eager
  // `.ttip` (if any) is left intact for conformance.js to wire.
  function parseTdHtml(html) {
    var t = document.createElement('table');
    t.innerHTML = '<tbody><tr>' + html + '</tr></tbody>';
    return t.querySelector('td');
  }

  function buildCellNode(cell, col) {
    var td = document.createElement('td');
    if (!cell) {
      // Defensive: a column with no matching cell renders as an empty blank slot.
      td.className = ('cell todo ' + (col.band || '')).trim();
      td.setAttribute('data-col-hide-group', col.group_key || '');
      return td;
    }
    if (cell.kind === 'blank') {
      td.className = ('cell todo ' + (cell.band || '')).trim();
      td.setAttribute('data-col-hide-group', cell.col_group || '');
      return td;
    }
    // kind 'cell' or 'missing': a real (comparable) cell.
    td.className = ('cell ' + (cell.band || '')).trim();
    td.setAttribute('data-col-hide-group', cell.col_group || '');
    // data-family drives ONLY the reference-aware "not implemented for family X" note
    // (the PARSER_NI path). Emit it only on pages that carry a parser_ni map (the v2
    // page) — on the v1 page it has no consumer and would make applyCtl count every
    // no-cmp missing cell as n/a (inflating the overview count vs the server render).
    if (_usesFamily) {
      td.setAttribute('data-family', cell.family || '');
    }
    if (cell.cmp) {
      // setAttribute stores the raw JSON; getAttribute (conformance.js) returns it
      // verbatim for JSON.parse — no manual attribute escaping needed.
      td.setAttribute('data-cmp', JSON.stringify(cell.cmp));
      var marker = document.createElement('span');
      marker.className = 'cmp-marker';
      var markerText = document.createElement('span');
      markerText.className = 'marker-text';
      marker.appendChild(markerText);
      td.appendChild(marker);
    }
    if (cell.known_divergence) {
      var kd = document.createElement('span');
      kd.className = 'kdiv';
      kd.setAttribute('title', 'Known v1-vs-v2 divergence: calls agree, normal_text differs'
        + ' by design — see the popup\'s explanation');
      kd.textContent = '≠';
      td.appendChild(kd);
    }
    if (cell.fixture_href) {
      // Empty anchor: the visible glyph is written into .marker-text by applyCtl;
      // the link still points at the fixture yaml.
      var a = document.createElement('a');
      a.setAttribute('href', cell.fixture_href);
      td.appendChild(a);
    }
    if (cell.tooltip) {
      var ttip = document.createElement('div');
      ttip.className = 'ttip';
      td.appendChild(ttip);
      // Empty .ttip now (so conformance.js's attachTooltip finds it); content built
      // lazily on first interaction from the keyed model (survives transpose clones).
      registerTooltip(td, cell.tooltip);
    }
    return td;
  }

  function buildDataRow(tab, row) {
    var tr = document.createElement('tr');
    var modelTd = document.createElement('td');
    modelTd.className = 'model';
    modelTd.setAttribute('data-col-hide-group', 'model');
    modelTd.innerHTML = row.model_label_html || '';
    tr.appendChild(modelTd);
    tr.appendChild(placeholderTd('model'));
    // Parser cell: row.parser.html is a ready `<td class="parser">...</td>` string.
    if (row.parser && row.parser.html) {
      var ptd = parseTdHtml(row.parser.html);
      if (ptd) {
        tr.appendChild(ptd);
      } else {
        tr.appendChild(emptyParserTd());
      }
    } else {
      tr.appendChild(emptyParserTd());
    }
    tr.appendChild(placeholderTd('parser'));
    var cols = tab.columns || [];
    var cells = row.cells || {};
    for (var i = 0; i < cols.length; i++) {
      var c = cols[i];
      tr.appendChild(buildCellNode(cells[c.sub], c));
      var next = cols[i + 1];
      if (!next || next.group_key !== c.group_key) {
        tr.appendChild(placeholderTd(c.group_key));
      }
    }
    return tr;
  }
  function emptyParserTd() {
    var td = document.createElement('td');
    td.className = 'parser';
    td.setAttribute('data-col-hide-group', 'parser');
    return td;
  }

  function buildTable(tab) {
    var table = document.createElement('table');
    table.setAttribute('data-parity-table', '');
    table.setAttribute('data-case-prefix', tab.case_prefix || '');
    table.setAttribute('data-mode', tab.mode || '');
    var thead = document.createElement('thead');
    thead.innerHTML = '<tr>' + groupHeadersHtml(tab) + '</tr><tr>' + subHeadersHtml(tab) + '</tr>';
    table.appendChild(thead);
    var tbody = document.createElement('tbody');
    // Section banner colspan = model + parser + one per sub-case column. The
    // hidden group placeholders don't count (they're display:none by default);
    // conformance.js recomputes this via applyColumnState on load anyway.
    var nCols = 2 + (tab.columns ? tab.columns.length : 0);
    (tab.rows || []).forEach(function (row) {
      if (row.section) {
        var tr = document.createElement('tr');
        tr.className = 'section';
        var td = document.createElement('td');
        td.setAttribute('data-section-span', '');
        td.colSpan = nCols;
        td.textContent = row.section;
        tr.appendChild(td);
        tbody.appendChild(tr);
      } else {
        tbody.appendChild(buildDataRow(tab, row));
      }
    });
    table.appendChild(tbody);
    return table;
  }

  // --- Static per-panel sections (legends, stats, glossary) ------------------
  // Copied verbatim from conformance_table.html.j2 so conformance.js finds the
  // same data-overview-count spans it writes tallies into.
  function summaryLegendHtml() {
    // Prefer the page-supplied summary legend (each generator emits its own so the v1
    // "match Base" wording + its stats line survive); fall back to the v2 text.
    if (_summaryLegendHtml) { return _summaryLegendHtml; }
    return '<p class="legend summary-only">'
      + '<span class="summary-key ok" aria-hidden="true"></span>green(<span data-overview-count="ok">0</span>)'
      + ' = selected implementation output is clean · '
      + '<span class="summary-key problem" aria-hidden="true"></span>red(<span data-overview-count="problem">0</span>)'
      + ' = leaks parser markup, has an expected error, or the engine parser failed to parse'
      + ' (<span style="font-family:ui-monospace,monospace">✗</span>) · '
      + '<span class="summary-key na" aria-hidden="true"></span>gray(<span data-overview-count="na">0</span>)'
      + ' = not applicable, unavailable, missing fixture, or a family the Dynamo v2 stream parser'
      + ' doesn\'t implement.</p>';
  }
  function detailsLegendHtml(tab, page) {
    var h = '<div class="legend details-only">' + (page.legend_html || '');
    if (tab.captured_note) {
      h += '<p class="versions">' + escapeHtml(tab.captured_note) + '</p>';
    }
    h += '</div>';
    return h;
  }
  function statsHtml(tab) {
    var s = tab.stats || {};
    return '<p class="stats details-only">Stats: ' + num(s.families) + ' families × '
      + num(s.sub_cases) + ' sub-cases = ' + num(s.slots) + ' grid slots ('
      + num(s.na) + ' n/a, ' + num(s.missing) + ' missing). <strong>' + num(s.real)
      + '</strong> real cases: <span style="color:#0a7d2c">' + num(s.parity)
      + ' captured-peer conformance</span> · <span style="color:#555">' + num(s.dynamo_only)
      + ' Dynamo Rust-only</span> · <span style="color:#555">' + num(s.documented)
      + ' documented divergences</span> (have <code>reason:</code>) · '
      + '<span style="color:#b00">' + num(s.research) + ' research-needed</span>'
      + ' (no <code>reason:</code> yet) · <span style="color:#b00">' + num(s.errors)
      + ' parser errors</span>.</p>';
  }
  function glossaryHtml(tab) {
    var sid = tab.case_section_id || tab.mode || '';
    var prefix = tab.case_prefix || '';
    var h = '<div class="details-only"><h2 id="case-descriptions-' + escapeAttr(sid) + '">'
      + 'Case descriptions</h2><p>Full case definitions: <a href="'
      + escapeAttr(tab.case_docs_href || '') + '">' + escapeHtml(tab.case_docs_label || '')
      + '</a>.</p><table class="glossary"><tbody>';
    (tab.glossary || []).forEach(function (group) {
      h += '<tr class="category"><td colspan="2">' + escapeHtml(group.label) + '</td></tr>';
      (group.rows || []).forEach(function (pair) {
        h += '<tr><td class="sub">' + escapeHtml(prefix + pair[0]) + '</td><td>'
          + escapeHtml(pair[1]) + '</td></tr>';
      });
    });
    h += '</tbody></table></div>';
    return h;
  }

  // --- Panel + tabs bar ------------------------------------------------------
  function buildPanel(tab, page, multiTab) {
    var section = document.createElement('section');
    section.id = tab.id;
    section.className = 'tab-panel' + (tab.active ? ' active' : '');
    section.setAttribute('role', 'tabpanel');
    if (multiTab) { section.setAttribute('aria-labelledby', tab.id + '-button'); }
    var hasCands = tab.candidates && tab.candidates.length;
    if (hasCands) {
      section.setAttribute('data-cmp-panel', 'true');
      section.insertAdjacentHTML('beforeend', compareBarHtml(tab));
    }
    section.appendChild(buildTable(tab));
    // toolbar-desc (stream tabs carry a parser/input explainer).
    if (tab.toolbar_desc_html) {
      section.insertAdjacentHTML('beforeend', '<div class="toolbar-desc">' + tab.toolbar_desc_html + '</div>');
    }
    section.insertAdjacentHTML('beforeend', summaryLegendHtml());
    section.insertAdjacentHTML('beforeend', detailsLegendHtml(tab, page));
    section.insertAdjacentHTML('beforeend', statsHtml(tab));
    if (tab.glossary && tab.glossary.length) {
      section.insertAdjacentHTML('beforeend', glossaryHtml(tab));
    }
    // NOTE: tab.details_note_html exists in the model but the current template
    // does not render it into the DOM, so we omit it too (byte-parity with server).
    return section;
  }

  function buildTabsBar(page) {
    var html = '<div class="tabs" role="tablist">';
    (page.tabs || []).forEach(function (t) {
      html += '<button class="tab-button' + (t.active ? ' active' : '') + '"'
        + ' id="' + escapeAttr(t.id) + '-button" type="button" role="tab"'
        + ' aria-selected="' + (t.active ? 'true' : 'false') + '"'
        + ' aria-label="' + escapeAttr(t.tab_title || '') + '"'
        + ' title="' + escapeAttr(t.tab_title || '') + '"'
        + ' data-tab-target="' + escapeAttr(t.id) + '">'
        + (t.label_html || escapeHtml(t.label || '')) + '</button>';
    });
    html += '<span class="tabs-right">'
      + '<label class="checkbox-option cmp-detailed"><input type="checkbox" data-view-detailed> Detailed</label>'
      + '<label class="checkbox-option cmp-transpose"><input type="checkbox" data-transpose-toggle> Transpose</label>'
      + '<button type="button" class="cmp-reset" data-reset title="Clear all selections and reload defaults">Reset</button>'
      + '</span></div>';
    var tmp = document.createElement('div');
    tmp.innerHTML = html;
    return tmp.firstChild;
  }

  // --- Entry point -----------------------------------------------------------
  try {
    var modelEl = document.getElementById('conformance-model');
    if (!modelEl) { return; }  // server HTML already present; nothing to build.
    var page = JSON.parse(modelEl.textContent);
    if (!page || !page.tabs || !page.tabs.length) { return; }

    _usesFamily = !!(page.parser_ni && Object.keys(page.parser_ni).length);
    _summaryLegendHtml = (page.meta && page.meta.summary_legend_html) || null;

    var multiTab = page.tabs.length > 1;
    var newNodes = [];
    if (multiTab) { newNodes.push(buildTabsBar(page)); }
    page.tabs.forEach(function (tab) { newNodes.push(buildPanel(tab, page, multiTab)); });

    // Replace the server-rendered tabs bar + panels in place with model-built
    // ones. Insert the new nodes ahead of the first existing tabs/panel (or the
    // model script if the server rendered neither), then drop the old ones.
    var oldTabs = document.querySelector('.tabs');
    var oldPanels = Array.prototype.slice.call(document.querySelectorAll('.tab-panel'));
    var anchor = oldTabs || oldPanels[0] || modelEl;
    var parent = anchor ? anchor.parentNode : document.body;
    newNodes.forEach(function (n) { parent.insertBefore(n, anchor); });
    if (oldTabs && oldTabs.parentNode) { oldTabs.parentNode.removeChild(oldTabs); }
    oldPanels.forEach(function (p) { if (p.parentNode) { p.parentNode.removeChild(p); } });
  } catch (err) {
    // On any failure, leave the server-rendered DOM intact — never blank the page.
    console.error('conformance_view: failed to build DOM from model', err);
  }
})();
