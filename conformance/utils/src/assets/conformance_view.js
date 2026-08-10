// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Conformance page VIEW (DIS-2434): build the tabs + table + compare bar + popups
// from the inlined JSON model (`<script id="conformance-model">`). This runs at
// parse time, BEFORE conformance.js, and produces the exact DOM (classes/attrs)
// that conformance.js queries — so the interactivity script wires against it
// unmodified. The template emits only the page skeleton + the model blob; this
// view is the sole renderer of the tabs bar and panels.
(function () {
  // --- Theme (light / dark) --------------------------------------------------
  // The page was light while every popup was dark, so the two surfaces disagreed. One
  // switch drives both via `data-theme` on <html>; CSS carries the per-surface overrides.
  // Persisted in a COOKIE, not a URL param — the query string is reserved for the
  // click-driven compare/selection state, and the theme must not be shareable noise.
  var THEME_COOKIE = 'conformance_theme';
  var THEME_GLYPH = { light: '\u25D1', dark: '\u25D0' };  // ◑ / ◐

  function readCookie(name) {
    var parts = String(document.cookie || '').split(';');
    for (var i = 0; i < parts.length; i++) {
      var kv = parts[i].split('=');
      if (kv[0].trim() === name) { return decodeURIComponent((kv[1] || '').trim()); }
    }
    return null;
  }
  // localStorage backs the cookie because a `file://` page cannot set one — the rendered
  // HTML is opened both ways (served over http, and straight off disk), and a preference
  // that silently forgets itself in one of them is worse than no preference.
  function readStored() {
    try { return window.localStorage.getItem(THEME_COOKIE); } catch (e) { return null; }
  }
  function currentTheme() {
    var t = readCookie(THEME_COOKIE) || readStored();
    return t === 'dark' || t === 'light' ? t : 'light';
  }
  function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    // 1 year, path=/ so it holds for every rendered page in the tree.
    document.cookie = THEME_COOKIE + '=' + theme + ';path=/;max-age=31536000;samesite=lax';
    try { window.localStorage.setItem(THEME_COOKIE, theme); } catch (e) { /* private mode */ }
    var btns = document.querySelectorAll('[data-theme-toggle]');
    for (var i = 0; i < btns.length; i++) { btns[i].textContent = THEME_GLYPH[theme]; }
  }
  // Apply before the table is built so there is no flash of the wrong theme.
  applyTheme(currentTheme());
  document.addEventListener('click', function (e) {
    var b = e.target && e.target.closest ? e.target.closest('[data-theme-toggle]') : null;
    if (!b) { return; }
    applyTheme(currentTheme() === 'dark' ? 'light' : 'dark');
  }, false);

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

  // Per-family declared markers (page.family_markers) for the colorizer's declared
  // lookup — the JS analogue of markup.py's _declared_lookup. Set at entry.
  var _familyMarkers = {};
  // COLORIZED: delegate to colorize.js (__markupColorize). `ctx` is the per-tooltip link
  // context so identical content shares a background across input + output cells. If the
  // module is somehow absent, fall back to plain escaping so the page never crashes.
  // Markers -> background color (pair-matched; unmatched -> red); user text -> per-word
  // foreground color (same word == same color everywhere in this tooltip).
  function colorize(text, family, ctx) {
    var mc = (typeof window !== 'undefined') && window.__markupColorize;
    if (!mc) { return escapeHtml(text == null ? '' : String(text)); }
    return mc.colorizeLinked(text, family == null ? null : family, _familyMarkers, ctx);
  }
  // Word coloring only, no marker parsing — for the `calls=` JSON blob. The parsed
  // arguments there are the SAME values the input carried (mode, fast, ...), so they
  // must share the tooltip's word hues; but the blob is JSON, not model markup, so
  // running the marker matcher over it would be wrong.
  function colorizeWords(text, ctx) {
    var mc = (typeof window !== 'undefined') && window.__markupColorize;
    if (!mc) { return escapeHtml(text == null ? '' : String(text)); }
    return mc.colorizeWords(text == null ? '' : String(text), ctx);
  }
  // In an OUTPUT content payload (text=/reasoning=) a control marker is LEAKED markup:
  // a correct parse consumes every marker, so anything marker-shaped left inside content
  // is a leak and must read as such (red), NOT as a legit marker pair. The generic
  // colorizer pairs `<think>...</think>` and colors it like real structure, so flag
  // markers here instead: any angle-bracket token (`<think>`, `</think>`, `<|channel>`,
  // `<tool_call|>`, `<|tool_call_end|>`, ...) plus gemma4's bare `thought` label (the
  // opener with its brackets already stripped).
  function colorizeUnifiedLeak(text, family, ctx) {
    var s = String(text == null ? '' : text);
    var re = /<\|?[^<>]*\|?>|thought\n/g;
    var out = '', last = 0, m;
    while ((m = re.exec(s)) !== null) {
      if (m.index > last) { out += colorize(s.slice(last, m.index), family, ctx); }
      var tok = m[0];
      if (tok === 'thought\n') { out += '<span class="tt-orphan">thought</span>\n'; }
      else { out += '<span class="tt-orphan">' + escapeHtml(tok) + '</span>'; }
      last = m.index + tok.length;
    }
    if (last < s.length) { out += colorize(s.slice(last), family, ctx); }
    return out;
  }

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
    // Anything carrying a `data-ttip-id` builds the same lazy way a data cell does —
    // headers included, which is why this asks for the attribute rather than listing the
    // element types that happen to have one today. A list here had already missed
    // `th.tcol-model`, so hovering a transposed model header showed an empty box.
    var td = t && t.closest ? t.closest('[data-ttip-id]') : null;
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
  function outputBlock(b, family, ctx) {
    if (!b) { return '—'; }
    if (b.unavailable != null) {
      // Prose, not payload: these carry n/a rationale and TODO notes.
      return '<span class="expl">unavailable: ' + escapeHtml(String(b.unavailable)) + '</span>';
    }
    if (b.exception != null) {
      // The parser ran and threw; surface the exception verbatim (e.g. the named
      // vLLM Rust crate variant `ToolParserError::ParsingFailed (...)`).
      return 'exception: ' + escapeHtml(String(b.exception));
    }
    if (b.error != null) {
      var e = (typeof b.error === 'string') ? b.error : JSON.stringify(b.error);
      return 'error: ' + escapeHtml(e);
    }
    // Unified tab: an ordered event stream (reasoning | text | tool_call).
    if (b.events != null) {
      var evlines = b.events.map(function (ev) {
        if (ev.kind === 'reasoning') {
          return '<span class="fldl">reasoning=\'</span>' + colorizeUnifiedLeak(ev.text || '', family, ctx) + '<span class="fldl">\'</span>';
        }
        if (ev.kind === 'text') {
          return '<span class="fldl">text=\'</span>' + colorizeUnifiedLeak(ev.text || '', family, ctx) + '<span class="fldl">\'</span>';
        }
        if (ev.kind === 'tool_call') {
          // Colorize the name + args like the streaming deltas do (word hues shared with
          // the input), instead of plain-escaping — so the assembled row matches.
          return '<span class="fldl">tool_call=</span>' + colorizeWords(ev.name || '', ctx)
            + '(' + colorizeWords(JSON.stringify(ev.arguments || {}), ctx) + ')';
        }
        return '';
      });
      var uout = evlines.join('\n') || '<i>(no events)</i>';
      // A blank line separates the monospaced event stream from the explanation
      // lines (verdict / TODO / note) so they don't read as another event.
      var expl = [];
      if (b.verdict && b.verdict !== 'MATCH') { expl.push('diverges: ' + escapeHtml(b.verdict)); }
      if (b.todo) { expl.push(escapeHtml(String(b.todo))); }
      if (b.note) { expl.push(escapeHtml(String(b.note))); }
      if (expl.length) {
        uout += '\n\n' + expl.map(function (x) { return '<span class="expl">' + x + '</span>'; }).join('\n');
      }
      return uout;
    }
    // Unified tab: vLLM column is a documented expectation (no live events yet).
    if (b.expected != null) {
      var xout = '<span class="fldl">expected: </span>' + escapeHtml(b.expected)
        + ' <span class="parser-base">(live capture pending — U1)</span>';
      if (b.note) { xout += '\n<span class="expl">' + escapeHtml(String(b.note)) + '</span>'; }
      return xout;
    }
    var out;
    if (b.reasoning_text != null) {
      // Reasoning cell: reasoning_text + normal_text (markers -> bg, words -> fg).
      out = '<span class="fldl">reasoning_text=\'</span>' + colorize(b.reasoning_text, family, ctx)
        + '<span class="fldl">\'</span>'
        + '\n<span class="fldl">normal_text=\'</span>' + colorize(b.normal_text || '', family, ctx)
        + '<span class="fldl">\'</span>';
    } else {
      var nt = b.normal_text || '';
      var calls = b.calls || [];
      out = '<span class="fldl">normal_text=\'</span>' + colorize(nt, family, ctx)
        + '<span class="fldl">\'</span>'
        + '\n<span class="fldl">calls=</span>' + colorizeWords(JSON.stringify(calls), ctx);
    }
    if (b.explanation) {
      // Blank line before the prose: `explanation:` ran straight on from the `calls=`
      // JSON, so a long note read as a continuation of the data rather than a note.
      out += '\n\n<span class="expl">explanation: ' + escapeHtml(String(b.explanation)) + '</span>';
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
  // The emitted name/arguments are values the INPUT carried (get_weather, NYC, EST),
  // so they colorize against the tooltip vocabulary like every other output surface.
  // The `name='`/`args='` labels around them are chrome and stay plain.
  // Deltas are bucketed by `index` (ToolCallDelta.tool_index) FIRST. One chunk can
  // carry deltas for two different calls, and concatenating their `arguments` into a
  // single string renders two independent calls as one call with malformed JSON —
  // `name='get_weather' name='get_time' args='{"location":"NYC"}{"timezone":"EST"}'`.
  // Fragments are only meant to join WITHIN one index; `index` is exactly the field a
  // client uses to demultiplex concurrent calls in the OpenAI stream.
  function renderDeltas(deltas, ctx) {
    if (!deltas || !deltas.length) { return '<span class="parser-base">—</span>'; }
    var order = [];
    var byIndex = {};
    var unified = [];  // Unified tab: reasoning / text / tool_call event deltas (one ordered stream)
    deltas.forEach(function (d) {
      if (d == null) { return; }
      if (d.kind === 'reasoning') { unified.push("<span class=\"fldl\">reasoning='</span>" + colorize(String(d.text || ''), null, ctx) + "<span class=\"fldl\">'</span>"); return; }
      if (d.kind === 'text') { unified.push("<span class=\"fldl\">text='</span>" + colorize(String(d.text || ''), null, ctx) + "<span class=\"fldl\">'</span>"); return; }
      if (d.kind === 'tool_call') {
        var tc = "<span class=\"fldl\">tool_call=</span>" + escapeHtml(String(d.name || ''));
        if (d.arguments != null) { tc += " args='" + colorizeWords(String(d.arguments), ctx) + "'"; }
        unified.push(tc);
        return;
      }
      var key = (d.index == null) ? '' : String(d.index);
      if (!Object.prototype.hasOwnProperty.call(byIndex, key)) {
        byIndex[key] = { names: [], args: '' };
        order.push(key);
      }
      if (d.name != null) { byIndex[key].names.push(String(d.name)); }
      if (d.arguments != null) { byIndex[key].args += String(d.arguments); }
    });
    if (unified.length) { return unified.join('\n'); }
    // The `#N` marker is shown only when there is more than one call to tell apart, so
    // single-call chunks — the overwhelming majority — render exactly as before.
    var showIndex = order.length > 1;
    var groups = order.map(function (key) {
      var g = byIndex[key];
      var parts = [];
      if (showIndex && key !== '') { parts.push('<span class="fldl">#' + escapeHtml(key) + '</span>'); }
      g.names.forEach(function (n) { parts.push("name='" + colorizeWords(n, ctx) + "'"); });
      if (g.args) { parts.push("args='" + colorizeWords(g.args, ctx) + "'"); }
      return parts.join(' ');
    }).filter(function (s) { return s !== ''; });
    return groups.length ? groups.join('\n') : '<span class="parser-base">—</span>';
  }

  function inputTextCell(input, ctx) {
    if (input && input.text != null) {
      return '<span class="fldl">input_text=\'</span>' + colorize(input.text, input.family, ctx)
        + '<span class="fldl">\'</span>';
    }
    return '';
  }

  // Per-chunk linked delta_text, computed over the JOINED stream so a tag spanning a
  // chunk boundary keeps one underline color (mirrors markup.colorize_stream_deltas).
  function chunkDeltaHtml(input, ctx) {
    var mc = (typeof window !== 'undefined') && window.__markupColorize;
    var chunks = input.chunks || [];
    if (!mc) {
      return chunks.map(function (ch) { return escapeHtml(ch.delta_text || ''); });
    }
    return mc.colorizeLinkedStreamDeltas(chunks, input.family == null ? null : input.family, _familyMarkers, ctx);
  }

  function buildChartHtml(m, ctx) {
    var allCands = m.candidates || [];
    if (!allCands.length) { return ''; }
    var input = m.input || { kind: null };
    var family = input.family;
    // Golden is the fixed reference: render its output at the bottom of the INPUT column,
    // not as a selectable column. Every engine is measured against it.
    var golden = null;
    var cands = allCands.filter(function (c) {
      if (c.key === 'golden') { golden = c; return false; }
      return true;
    });
    var header = '';
    cands.forEach(function (c, ci) {
      header += '<th data-cand="' + escapeAttr(c.key) + '" data-cand-order="' + ci + '">'
        + escapeHtml(c.label) + '</th>';
    });
    var body = '';
    if (input.kind === 'chunks' && input.chunks && input.chunks.length) {
      var deltaHtml = chunkDeltaHtml(input, ctx);
      input.chunks.forEach(function (ch, i) {
        var row = '<tr><td class="cin">' + deltaHtml[i];
        if (ch.finish_reason) {
          row += '<span class="fr"> finish=' + escapeHtml(String(ch.finish_reason)) + '</span>';
        }
        row += '</td>';
        cands.forEach(function (c, ci) {
          var impl = implKeyOf(c.key);
          var d = (ch.expected && ch.expected[impl]) || [];
          row += '<td data-cand="' + escapeAttr(c.key) + '" data-cand-order="' + ci + '">'
            + renderDeltas(d, ctx) + '</td>';
        });
        body += row + '</tr>';
      });
    }
    // Assembled row: the input cell shows the GOLDEN output (the reference); each engine
    // column shows its own assembled block.
    var inputCell = golden
      ? outputBlock(golden.block, family, ctx).replace(/\n/g, '<br>')
        + '<br><br><span class="golden-out-cap">Golden output</span>'
      : (body ? 'assembled' : inputTextCell(input, ctx));
    var fin = '<tr class="ttip-final"><td class="cin">' + inputCell + '</td>';
    cands.forEach(function (c, ci) {
      // Red when this candidate's assembled output DIVERGES from the golden oracle
      // (its verdict, classified against golden, is anything but MATCH) — the same
      // "red = doesn't match golden" rule the matrix cell uses. golden itself is the
      // input-column reference and never appears here, so it is never flagged.
      var diverges = c.block && c.block.verdict && c.block.verdict !== 'MATCH';
      fin += '<td data-cand="' + escapeAttr(c.key) + '" data-cand-order="' + ci + '"'
        + (diverges ? ' class="cand-diverge"' : '') + '>'
        + outputBlock(c.block, family, ctx).replace(/\n/g, '<br>') + '</td>';
    });
    fin += '</tr>';
    var inputHdr = golden ? 'input / golden output' : 'input';
    // The table carries class `ttip-chunks` — conformance.js keys the popup grid on it.
    return '<table class="ttip-chunks"><thead><tr><th>' + escapeHtml(inputHdr) + '</th>' + header
      + '</tr></thead><tbody>' + body + fin + '</tbody></table>';
  }

  // --- Parser configuration bullet list ---------------------------------------
  // The case description says WHAT the case is; this says HOW the parser was
  // configured to run it. Every knob is listed on its own line, always — a knob the
  // case did not set still shows, as `<not specified>` plus the default that applied,
  // so "not set" and "set to the default" stay distinguishable when reading a popup.
  //
  // ONLY real request-scoped parser inputs belong here, i.e. the arguments of
  // `UnifiedParser::initialize_with_output_mode`. `finish_reason` is deliberately
  // NOT one: `finish()` takes no argument, so the parser cannot see it (the same is
  // true of vLLM's parsers) — it is a stream property, not a knob. Listing it here
  // would claim the parser behaves differently per value, which it does not.
  // value -> what that value MEANS, so a reader does not have to know the Rust enum.
  // `None` and `Native` are real `#[default]` variants, NOT absent values — the gloss
  // says so, because "None" otherwise reads as null/unset.
  var CONFIG_GLOSS = {
    prefill: {
      'None': 'prompt opened no channel; the model emits its own markers (enum default)',
      'Reasoning': 'prompt opened reasoning, so the stream begins INSIDE a thought',
      'Response': 'prompt opened the visible response channel'
    },
    tool_output_mode: {
      'Native': "the model's own tool-call markup (enum default)",
      'GuidedJson': 'guided decoding emits bare JSON instead of native markup'
    }
  };

  var CONFIG_KEYS = [
    { key: 'prefill', dflt: 'None', values: ['None', 'Reasoning', 'Response'] },
    { key: 'tool_output_mode', dflt: 'Native', values: ['Native', 'GuidedJson'] },
    {
      key: 'named_tool',
      dflt: 'null',
      // Not an enum: any tool name from the request, or null.
      values: ['null', '<a tool name>'],
      // Only meaningful under GuidedJson, where null is the "required" choice.
      gloss: function (v, init) {
        if (init.tool_output_mode !== 'GuidedJson') { return 'only applies to GuidedJson'; }
        return v == null
          ? 'required choice: payload is one call object, or an array of them'
          : 'named choice: payload is this tool\'s arguments alone';
      }
    }
  ];

  function buildConfigHtml(init) {
    if (!init) { return ''; }
    var items = CONFIG_KEYS.map(function (spec) {
      var v = init[spec.key];
      var unset = (v === undefined || v === '');
      // Distinguish "the case did not set this" from "the case set it to the
      // variant that happens to be named None".
      var shown = unset ? '<unset>' : (v === null ? 'null' : String(v));
      var why = spec.gloss
        ? spec.gloss(unset ? null : v, init)
        : (CONFIG_GLOSS[spec.key] || {})[shown];
      if (unset) {
        why = 'not set by this case; parser default is ' + spec.dflt
          + (why ? ' — ' + why : '');
      }
      // The value set is part of reading a knob: without it "Response" gives no
      // hint that "Reasoning" and "None" are the alternatives. Values are literals,
      // so each is mono even though the surrounding gloss is prose.
      var vals = (spec.values || [])
        .map(function (x) { return '<code>' + escapeHtml(x) + '</code>'; })
        .join(', ');
      return '<li><code>' + escapeHtml(spec.key + '=' + shown) + '</code>'
        + (why ? '<span class="ttip-config-why"> — ' + escapeHtml(why) + '</span>' : '')
        + (vals ? '<span class="ttip-config-vals"> (possible: ' + vals + ')</span>' : '')
        + '</li>';
    });
    if (!items.length) { return ''; }
    return '<ul class="ttip-config">' + items.join('') + '</ul>';
  }

  // --- Tooltip content (built lazily into the empty .ttip) -------------------
  function buildTooltipHtml(m) {
    if (m && m.grammar) { return buildGrammarHtml(m); }
    var h = '';
    // Id + description on ONE line, matching the column grammar popup: the id keeps its
    // accent color, the description follows inline in the normal tooltip text color (not
    // the loud section blue). Falls back to its own line only when there is no id/head.
    if (m.head) {
      h += '<div class="ttip-head">' + escapeHtml(m.head)
        + (m.description ? ' <span class="ttip-head-desc">' + escapeHtml(m.description) + '</span>' : '')
        + '</div>';
    } else if (m.description) {
      h += '<div class="ttip-casedesc"><span class="ttip-head-desc">' + escapeHtml(m.description) + '</span></div>';
    }
    // Parser configuration for THIS case, one knob per line, directly under the
    // description it qualifies.
    h += buildConfigHtml(m.init);
    // One link context per tooltip, shared by the input and every output cell. Harvest
    // the vocabulary from the INPUT first, then seal it: the input's tokens and their
    // colors are the only ones that exist, and every later render (the input itself,
    // each candidate's block, the `calls=` JSON) colors by matching against them. That
    // is what makes an output value carry its input color, and what lets a concatenated
    // output like `bodyanswer` come back as `body` + `answer` in the input's two colors.
    var mc = (typeof window !== 'undefined') && window.__markupColorize;
    var ctx = mc ? mc.newLinkCtx() : null;
    if (ctx) {
      var reg = m.input || {};
      var regFamily = reg.family == null ? null : reg.family;
      if (reg.kind === 'chunks' && reg.chunks && reg.chunks.length) {
        mc.colorizeLinkedStreamDeltas(reg.chunks, regFamily, _familyMarkers, ctx);
      } else if (reg.text != null) {
        mc.colorizeLinked(reg.text, regFamily, _familyMarkers, ctx);
      }
      mc.sealLinkCtx(ctx);
    }
    var cands = m.candidates || [];
    var chart = cands.length ? buildChartHtml(m, ctx) : '';
    // Description shown only when there's no chart (the chart's input cell carries it).
    // (description already emitted under the head, for chart and non-chart alike)
    if (chart) {
      // Chart is the per-candidate output surface — NEVER also emit the `.cand`
      // list (test_chart_tooltips_have_no_candidate_list: chart XOR list).
      h += chart;
    } else {
      var input = m.input || { kind: null };
      if (input.kind === 'text' && input.text) {
        h += '<div class="ttip-section">Input:</div>'
          + '<pre class="ttip-pre ttip-code">' + colorize(input.text, input.family, ctx) + '</pre>';
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
      h += '<pre class="ttip-pre ttip-note">' + escapeHtml(m.na_note) + '</pre>';
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
    // Golden is the oracle shown in the input column, not a selectable engine. Make it the
    // hidden comparison base ONLY when no engine is the default Reference — otherwise an
    // engine (Dynamo on the Unified tab) is the starred REF and golden is just the fixed
    // NΔ baseline (cmp.golden). A golden REF would color every cell green (it never diverges
    // from itself), which is exactly what the Unified red-on-diff rule must avoid.
    var hasGolden = (tab.candidates || []).some(function (c) { return c.key === 'golden'; });
    var engineRef = (tab.candidates || []).some(function (c) {
      return c.key !== 'golden' && c.default_bucket === 'A';
    });
    if (hasGolden && !engineRef) {
      html += '<input type="radio" class="cmp-ref" name="ref_' + escapeAttr(tab.id)
        + '" value="golden" checked style="display:none" aria-hidden="true">';
    }
    groups.forEach(function (g) {
      var engine = g[0];
      var impl = g[1];
      // Only render an engine column when this tab actually has a candidate for it,
      // so the Golden column appears only on the Unified tab and other tabs are unchanged.
      if (!(tab.candidates || []).some(function (c) { return c.impl === impl; })) { return; }
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
    // `no_parser_col` drops it entirely (the Unified tab — parser architecture is
    // per-engine, not a per-family column).
    var h = fixedColumnHeader('model', 'Model');
    if (!tab.no_parser_col) {
      var parserLabel = tab.parser_col_label
        || (tab.kind === 'reasoning' ? 'Reasoning family' : 'Tool calling family');
      h += fixedColumnHeader('parser', parserLabel);
    }
    (tab.column_groups || []).forEach(function (g) { h += groupColumnHeader(g); });
    return h;
  }
  // --- Per-column grammar popup ----------------------------------------------
  // Hovering a sub-case header shows THE SAME test case in every family's grammar,
  // side by side, so the envelope differences are readable at a glance. Streaming
  // inputs are joined back into one text: the chunk split is a streaming concern,
  // not a grammar one, and chunk boundaries only obscure the shape here.
  function caseInputText(tip) {
    if (!tip) { return null; }
    var inp = tip.input || {};
    if (inp.kind === 'chunks' && inp.chunks && inp.chunks.length) {
      return inp.chunks.map(function (ch) { return (ch && ch.delta_text) || ''; }).join('');
    }
    // An empty `model_text` is dropped from the model (falsy), so a text-kind input with
    // no `text` is EMPTY, not missing — `TOOLCALLING.batch.9.a` is the empty-model-text
    // case and was reporting "no input recorded" for every family.
    if (inp.kind === 'text' || inp.text != null) { return String(inp.text == null ? '' : inp.text); }
    return null;
  }

  // A family with no input for this case still gets a row, carrying WHY — an empty
  // row would read as "identical grammar" rather than "case does not apply here".
  function naReason(cell, tip) {
    if (tip && tip.na_note) { return tip.na_note; }
    if (!cell) { return 'no cell for this case'; }
    if (cell.kind === 'blank') { return 'case not defined for this family'; }
    if (cell.kind === 'missing') { return 'fixture missing'; }
    if (cell.status === 'na') { return 'not applicable to this family'; }
    return 'no input recorded';
  }

  function columnGrammarModel(tab, col) {
    var rows = [];
    var caseId = null;   // numbered id (e.g. "UNIFIED.7.b") — the cells' own id, not the slug
    var colDefs = null;  // shared output-candidate columns [{key,label,pin}], first row wins
    (tab.rows || []).forEach(function (row) {
      if (!row || row.section) { return; }              // section banners are not families
      var cell = (row.cells || {})[col.sub];
      if (cell && cell.case_id && !caseId) { caseId = cell.case_id; }
      var tip = cell && cell.tooltip;
      var text = caseInputText(tip);
      // An input that EXISTS but is empty is not the same as a missing one — case 9.a
      // ("Empty model text") is empty on purpose, and calling that "no input recorded"
      // would read as a gap in the corpus.
      // OUTPUT columns are one-per-candidate, keyed by candidate key. golden is PINNED
      // (always shown, leftmost output); the others show only when active, with the
      // Reference flagged/ordered first — exactly like the cell popups. applyCtl
      // (conformance.js) drives visibility + ordering off the compare selection.
      var cands = (tip && tip.candidates) || [];
      if (!colDefs && cands.length) {
        colDefs = [];
        for (var k = 0; k < cands.length; k++) {
          var cd = cands[k];
          if (cd && cd.key) {
            colDefs.push({ key: cd.key, label: cd.label || cd.key, pin: !!cd.pin_first || cd.key === 'golden' });
          }
        }
      }
      var blocks = {};
      for (var i = 0; i < cands.length; i++) {
        var cc = cands[i];
        if (cc && cc.key && cc.block) { blocks[cc.key] = cc.block; }
      }
      // Keep the raw chunk list for stream cases: the popup joins them for coloring
      // (markers and values must resolve over the WHOLE stream), but the reader still
      // needs to see where one chunk ended and the next began.
      var inp = (tip && tip.input) || {};
      rows.push({
        family: row.family || '',
        label: row.model_label || row.family || '',
        text: text ? text : null,
        chunks: (inp.chunks && inp.chunks.length) ? inp.chunks : null,
        blocks: blocks,
        reason: text ? null : (text === '' ? 'empty input — this case tests empty model text'
                                           : 'n/a — ' + naReason(cell, tip)),
      });
    });
    // `init` rides along so the column popup lists the SAME parser inputs as the
    // cells under it — the config list is built from this by the shared builder.
    return { head: caseId || fullCaseId(tab, col), desc: col.desc || '', init: col.init,
             grammar: rows, cands: colDefs || [] };
  }

  // The header shows the FULL case id, matching the cell popups and the fixture YAML, so
  // a column is identifiable without counting across the header row. Tool-calling
  // columns carry a bare sub ("1", "9.a") against a mode-qualified prefix
  // ("TOOLCALLING.batch."); reasoning columns already carry the whole id.
  function fullCaseId(tab, col) {
    var prefix = tab.case_prefix || '';
    var sub = String(col.sub == null ? col.label : col.sub);
    if (!prefix) { return sub; }
    return sub.indexOf(prefix) === 0 ? sub : prefix + sub;
  }

  function buildGrammarHtml(m) {
    // Id + description on ONE line: the id keeps its accent color, the description follows
    // inline in the normal tooltip text color (not the loud section blue).
    var h = '<div class="ttip-head">' + escapeHtml(m.head || '')
      + (m.desc ? ' <span class="ttip-head-desc">' + escapeHtml(m.desc) + '</span>' : '')
      + '</div>';
    // Same builder as the cell popup: a column header and the cells under it describe
    // one configuration, so they cannot drift into showing different knobs.
    h += buildConfigHtml(m.init);
    var cands = m.cands || [];
    var body = '';
    (m.grammar || []).forEach(function (r) {
      var cls = r.text ? '' : ' class="gr-na"';
      var cell, outCell;
      // ONE context per row, seeded from that row's input: every family has its own
      // marker vocabulary, and seeding from the input is what makes a value carry the
      // same color into the output — the same convention as the cell popups.
      var mc = (typeof window !== 'undefined') && window.__markupColorize;
      var ctx = null;
      if (mc) {
        ctx = mc.newLinkCtx();
        if (r.text) { mc.colorizeLinked(r.text, r.family || null, _familyMarkers, ctx); }
        mc.sealLinkCtx(ctx);
      }
      if (r.text && r.chunks && mc) {
        // Slice the JOINED render back apart at the chunk boundaries and mark each seam,
        // so cross-chunk coloring stays correct while the chunking stays visible.
        var parts = mc.colorizeLinkedStreamDeltas(r.chunks, r.family || null, _familyMarkers, ctx);
        // Fold a whitespace-only delta (e.g. a bare "\n" streamed on its own) into the
        // previous part: otherwise it draws a chunk-boundary marker on BOTH sides and reads
        // as a double return. Real token boundaries keep their single marker.
        var folded = [];
        for (var di = 0; di < parts.length; di++) {
          var dt = (r.chunks[di] && r.chunks[di].delta_text) || '';
          if (folded.length && /^\s*$/.test(dt)) { folded[folded.length - 1] += parts[di]; }
          else { folded.push(parts[di]); }
        }
        cell = folded.join('<span class="gr-chunk-sep" title="chunk boundary">\u2B90</span>');
      } else if (r.text) {
        cell = mc ? mc.colorizeLinked(r.text, r.family || null, _familyMarkers, ctx)
                  : escapeHtml(r.text);
      } else {
        cell = '<span class="parser-base">' + escapeHtml(r.reason || '') + '</span>';
      }
      // One OUTPUT column per candidate (golden pinned first, then Reference, then the
      // rest). data-cand/-order/-pin let applyCtl show only golden + the active columns
      // and order them REF-first, exactly like the cell popup's candidate columns.
      outCell = cands.map(function (c, ci) {
        var blk = r.blocks && r.blocks[c.key];
        var inner = blk ? outputBlock(blk, r.family || null, ctx).replace(/\n/g, '<br>')
                        : '<span class="parser-base">—</span>';
        return '<td class="gro" data-cand="' + escapeAttr(c.key) + '" data-cand-order="' + ci + '"'
          + (c.pin ? ' data-cand-pin="1"' : '') + '>' + inner + '</td>';
      }).join('');
      // Key the row by MODEL, not by parser family. DeepSeek V3, V3.1 and V3.2 are
      // distinct models that happen to share one parser family, and labelling all three
      // `deepseek_v3` made them read as duplicate rows. Every model gets its own row,
      // exactly like V4; the family is kept alongside since it names the grammar.
      var fam = (r.family && r.family !== r.label)
        ? '<span class="grfam">' + escapeHtml(r.family) + '</span>' : '';
      body += '<tr' + cls + '><td class="grf">' + escapeHtml(r.label || r.family)
        + fam + '</td><td class="gri">' + cell + '</td>' + outCell + '</tr>';
    });
    // One output header per candidate; applyCtl shows golden + the active columns,
    // flags the Reference (★) and orders golden -> REF -> rest.
    var header = cands.map(function (c, ci) {
      return '<th class="gro-hd" data-cand="' + escapeAttr(c.key) + '" data-cand-order="' + ci + '"'
        + (c.pin ? ' data-cand-pin="1"' : '') + '>' + escapeHtml(c.label) + '</th>';
    }).join('');
    return h + '<table class="ttip-chunks ttip-grammar"><thead><tr><th>model</th>'
      + '<th>input</th>' + header + '</tr></thead><tbody>' + body + '</tbody></table>';
  }

  function subHeadersHtml(tab) {
    var cols = tab.columns || [];
    var href = escapeAttr(tab.case_docs_href || '');
    var h = '';
    for (var i = 0; i < cols.length; i++) {
      var c = cols[i];
      // The rich grammar popup replaces the old native `title` tooltip (which could
      // only carry the one-line description, and rendered alongside the new popup).
      h += '<th class="case-sub ' + escapeAttr(c.band) + '" data-col-hide-group="'
        + escapeAttr(c.group_key) + '"><a href="' + href + '">'
        + escapeHtml(c.label) + '</a><div class="ttip"></div></th>';
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
    if (cell.red_on_diff) {
      // Unified tab: GOLDEN is the fixed oracle, so a cell is red when a SHOWN parser's
      // output DIVERGES from golden (any class — leaked markup, merged/reordered events,
      // dropped content), not just on a leak. Green = every shown parser matches golden.
      td.setAttribute('data-red-on-diff', '1');
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
    // `no_parser_col` drops the whole column (Unified tab).
    if (!tab.no_parser_col) {
      if (row.parser && row.parser.html) {
        var ptd = parseTdHtml(row.parser.html);
        tr.appendChild(ptd || emptyParserTd());
      } else {
        tr.appendChild(emptyParserTd());
      }
      tr.appendChild(placeholderTd('parser'));
    }
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
    // The sub-case headers were emitted as a string; key their grammar popups now that
    // they are real nodes. Built lazily like every other tooltip (a grammar table for
    // ~20 families is far too much to render up front for every column).
    var subThs = thead.querySelectorAll('th.case-sub');
    (tab.columns || []).forEach(function (col, i) {
      if (subThs[i]) { registerTooltip(subThs[i], columnGrammarModel(tab, col)); }
    });
    table.appendChild(thead);
    var tbody = document.createElement('tbody');
    // Section banner colspan = model + parser + one per sub-case column. The
    // hidden group placeholders don't count (they're display:none by default);
    // conformance.js recomputes this via applyColumnState on load anyway.
    var nCols = (tab.no_parser_col ? 1 : 2) + (tab.columns ? tab.columns.length : 0);
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
        h += '<tr><td class="sub">' + escapeHtml(prefix + pair[0]) + '</td>'
          + '<td class="gloss-desc">' + escapeHtml(pair[1]) + '</td></tr>';
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
      + '<button type="button" class="theme-toggle" data-theme-toggle'
      + ' title="Switch between light and dark">' + THEME_GLYPH[currentTheme()] + '</button>'
      + '<label class="checkbox-option cmp-transpose"><input type="checkbox" data-transpose-toggle> Transpose</label>'
      + '<button type="button" class="cmp-reset" data-reset title="Clear all selections and reload defaults">Reset</button>'
      + '</span></div>';
    var tmp = document.createElement('div');
    tmp.innerHTML = html;
    return tmp.firstChild;
  }

  // --- Schema-2 hydration (mirror of model.py hydrate_page) -------------------
  // The blob is COMPACTED (model.py _compact_page): repeated long strings live once
  // in page.strings (slots hold int indexes), per-cell tooltip-candidate meta lives
  // once per tab in tab.cand_meta, fixture_hrefs keep only their suffix under
  // tab.fixture_href_base, and the standard "<case_id> — <family>" head is dropped.
  // Hydrating once right after JSON.parse restores the schema-1 shape in memory, so
  // every builder below (and conformance.js) is untouched by the compaction.
  // KEEP THE SLOT LIST IN SYNC with model.py _iter_intern_slots.
  function hydratePage(page) {
    var strings = page.strings || [];
    function S(container, key) {
      if (typeof container[key] === 'number') { container[key] = strings[container[key]]; }
    }
    (page.tabs || []).forEach(function (tab) {
      var meta = tab.cand_meta || {};
      var base = tab.fixture_href_base || '';
      (tab.rows || []).forEach(function (row) {
        var cells = row.cells || {};
        Object.keys(cells).forEach(function (sub) {
          var cell = cells[sub];
          if (base && cell.fixture_href) { cell.fixture_href = base + cell.fixture_href; }
          (cell.facts || []).forEach(function (f) { S(f, 'reason'); });
          var tip = cell.tooltip;
          if (!tip) { return; }
          S(tip, 'description'); S(tip, 'na_note'); S(tip, 'leak_note');
          if (tip.input) { S(tip.input, 'text'); }
          (tip.reasons || []).forEach(function (r) { S(r, 'label'); S(r, 'reason'); });
          (tip.dynamo_notes || []).forEach(function (pair) { S(pair, 0); S(pair, 1); });
          var blocks = (tip.candidates || []).map(function (c) { return c.block; });
          if (tip.baseline) { blocks.push(tip.baseline.block); }
          blocks.forEach(function (b) { if (b) { S(b, 'explanation'); S(b, 'unavailable'); S(b, 'exception'); } });
          (tip.candidates || []).forEach(function (c) {
            var fields = meta[c.key] || {};
            for (var f in fields) {
              if (!(f in c)) { c[f] = fields[f]; }
            }
          });
          if (!('head' in tip) && cell.case_id && cell.family) {
            tip.head = cell.case_id + ' — ' + cell.family;
          }
        });
      });
    });
    return page;
  }

  // --- Entry point -----------------------------------------------------------
  try {
    var modelEl = document.getElementById('conformance-model');
    if (!modelEl) { return; }  // no model blob; nothing to build.
    var page = hydratePage(JSON.parse(modelEl.textContent));
    if (!page || !page.tabs || !page.tabs.length) { return; }

    _usesFamily = !!(page.parser_ni && Object.keys(page.parser_ni).length);
    _familyMarkers = page.family_markers || {};
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
