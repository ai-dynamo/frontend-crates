// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Tool-call markup colorizer — a faithful JS port of tests/parity/markup.py
// (colorize_markup / colorize_stream_deltas). Moving page rendering into
// the browser but shipped a stub `colorize()` that only HTML-escaped, dropping the
// server-side coloring of special tokens (`<|python_tag|>`, `<tool_call>...`,
// harmony `<|channel|>`, MiniMax namespace, gemma `<|"|>` quotes). This restores it.
//
// PARITY CONTRACT: for any single call this produces byte-identical HTML to
// markup.py with its module color state reset first (markup._color_seq = 0,
// markup._singleton_classes = {}). markup.py carried that palette counter as a
// process-global across every call in one render; here each top-level call resets it,
// so a tooltip's coloring is deterministic from its own text alone (an intentional
// simplification — cross-tooltip palette continuity was never visible to a reader).
// tests/parity/test_colorize_parity.py pins the equality via node.
//
// Loads in the browser (sets `self.__markupColorize`) and in node (module.exports).
(function (root, factory) {
  var mod = factory();
  if (typeof module !== 'undefined' && module.exports) { module.exports = mod; }
  root.__markupColorize = mod;
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  // html.escape(s, quote=True): & < > " ', with `'` -> &#x27; (NOT &#39;) to match
  // Python byte-for-byte.
  function escapeHtml(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#x27;');
  }

  // MiniMax M3 namespace special token — matched whole, before the generic `<...>`
  // rule, colored as muted namespace decoration (no open/close semantics).
  var MINIMAX_NS_TOKEN = ']<]minimax[>[';
  var NS_CLASS = 'tt-ns';
  // MiniMax ns token | `<...>` | Mistral-style `[NAME]`/`[/NAME]` (ALL-CAPS only, so
  // JSON arrays like `[{...}]` are not false-matched).
  function tagRe() { return /\]<\]minimax\[>\[|<[^<>]+>|\[\/?[A-Z][A-Z0-9_]*\]/g; }

  var PIPES = ['|', '｜'];  // ASCII and FULLWIDTH VERTICAL LINE
  var BEGIN_SUFFIXES = ['_begin', '▁begin'];  // ASCII underscore, LOWER ONE EIGHTH BLOCK
  var END_SUFFIXES = ['_end', '▁end'];

  // Harmony (gpt-oss) linear state-machine delimiters.
  var HARMONY_TURN_OPEN = 'start';
  var HARMONY_TURN_CLOSE = { end: true, 'return': true, call: true };
  var HARMONY_SECTION_MARKERS = { channel: true, constrain: true, message: true };
  function harmonyRe() { return /<\|([A-Za-z_]+)\|>/g; }
  var HARMONY_SEGMENT_CLASS = {
    start: 'tt-h-start',
    channel: 'tt-h-channel',
    constrain: 'tt-h-constrain',
    message: 'tt-h-message',
    end: 'tt-h-stop',
    'return': 'tt-h-stop',
    call: 'tt-h-call',
  };

  var PAIRED_PALETTE_SIZE = 8;

  // Per-call color state (reset at each top-level entry — see PARITY CONTRACT).
  function State() {
    this.seq = 0;
    this.singletons = {};
  }
  State.prototype.nextColorClass = function () {
    var cls = 'tt-c' + (this.seq % PAIRED_PALETTE_SIZE);
    this.seq += 1;
    return cls;
  };
  State.prototype.singletonClassFor = function (name) {
    var cls = this.singletons[name];
    if (cls == null) {
      cls = this.nextColorClass();
      this.singletons[name] = cls;
    }
    return cls;
  };

  // family markers: { family: {pairs: [[open, close], ...], singletons: [tok, ...]} }
  // -> { fullToken: [kind, pairId] }. Mirrors markup._declared_lookup.
  function declaredLookup(family, markersMap) {
    if (!family || !markersMap || !markersMap[family]) { return {}; }
    var decl = markersMap[family];
    var table = {};
    (decl.pairs || []).forEach(function (pair) {
      table[pair[0]] = ['open', pair[0]];
      table[pair[1]] = ['close', pair[0]];
    });
    (decl.singletons || []).forEach(function (tok) {
      table[tok] = ['singleton', tok];
    });
    return table;
  }

  function stripSuffix(s, suffixes) {
    for (var i = 0; i < suffixes.length; i++) {
      var suf = suffixes[i];
      if (s.length > suf.length && s.slice(-suf.length) === suf) {
        return s.slice(0, s.length - suf.length);
      }
    }
    return null;
  }

  function nameOf(s) {
    // First piece before whitespace / `/` / `>` / `=`, trailing ASCII `|` stripped.
    if (!s) { return ''; }
    return s.split(/[\s/>=]/)[0].replace(/\|+$/, '');
  }

  // Classify `<...>` inner text -> [kind, pairId, colorOverride].
  //   kind: 'open' | 'close' | 'singleton' | 'toggle' | null
  function tagKindAndName(inner) {
    if (inner.charAt(0) === '/') {
      return ['close', nameOf(inner.slice(1)), null];
    }
    var first = inner.slice(0, 1);
    var lastCh = inner.slice(-1);
    var startsPipe = PIPES.indexOf(first) >= 0;
    var endsPipe = PIPES.indexOf(lastCh) >= 0;
    if (startsPipe && endsPipe && inner.length >= 2) {
      var middle = inner.slice(1, -1);
      var stripped = stripSuffix(middle, BEGIN_SUFFIXES);
      if (stripped !== null) { return ['open', stripped, null]; }
      stripped = stripSuffix(middle, END_SUFFIXES);
      if (stripped !== null) { return ['close', stripped, null]; }
      if (middle === HARMONY_TURN_OPEN) { return ['open', '__harmony_turn', null]; }
      if (HARMONY_TURN_CLOSE[middle]) {
        return ['close', '__harmony_turn', '__harmony_pair_' + middle];
      }
      if (HARMONY_SECTION_MARKERS[middle]) {
        return ['singleton', '__harmony_section', null];
      }
      // gemma4 `<|"|>`: self-paired quote token (open/close decided by the stack).
      if (middle === '"') { return ['toggle', '__gemma_quote', null]; }
      return [null, '', null];
    }
    if (startsPipe && first === '|') { return ['open', nameOf(inner.slice(1)), null]; }
    if (endsPipe && lastCh === '|') { return ['close', nameOf(inner.slice(0, -1)), null]; }
    return ['open', nameOf(inner), null];
  }

  // Harmony: color each `<|token|>related-text` segment (not paired XML).
  function colorizeHarmony(text) {
    var pieces = [];
    var last = 0;
    var re = harmonyRe();
    var m;
    while ((m = re.exec(text)) !== null) {
      if (m.index > last) { pieces.push(escapeHtml(text.slice(last, m.index))); }
      var tokenName = m[1];
      var end;
      if (HARMONY_TURN_CLOSE[tokenName]) {
        end = re.lastIndex;
      } else {
        var re2 = harmonyRe();
        re2.lastIndex = re.lastIndex;
        var nextM = re2.exec(text);
        end = nextM ? nextM.index : text.length;
      }
      var cls = HARMONY_SEGMENT_CLASS[tokenName] || 'tt-h-other';
      var token = escapeHtml(m[0]);
      var related = escapeHtml(text.slice(re.lastIndex, end));
      pieces.push('<span class="tt-h ' + cls + '"><span class="tt-h-token">'
        + token + '</span>' + related + '</span>');
      last = end;
      re.lastIndex = end;
    }
    if (last < text.length) { pieces.push(escapeHtml(text.slice(last))); }
    return pieces.join('');
  }

  // XML/pipe-marker path: escape text, wrap each `<...>` token in a span; stack-match
  // open/close (fresh palette color per pair), undeclared/unmatched -> tt-orphan.
  function colorizeXml(text, family, markersMap) {
    var declared = declaredLookup(family, markersMap);
    var st = new State();
    var pieces = [];
    var stack = [];  // [pairId, pieceIndex]
    var last = 0;
    var re = tagRe();
    var m;
    while ((m = re.exec(text)) !== null) {
      if (m.index > last) { pieces.push(escapeHtml(text.slice(last, m.index))); }
      var tok = m[0];
      if (tok === MINIMAX_NS_TOKEN) {
        pieces.push('<span class="' + NS_CLASS + '">' + escapeHtml(tok) + '</span>');
        last = re.lastIndex;
        continue;
      }
      var kind, pairId, colorOverride;
      var hit = declared[tok];
      if (hit != null) {
        kind = hit[0]; pairId = hit[1]; colorOverride = null;
      } else {
        var k = tagKindAndName(tok.slice(1, -1));
        kind = k[0]; pairId = k[1]; colorOverride = k[2];
      }
      var esc = escapeHtml(tok);
      if (kind === null) {
        pieces.push('<span class="tt-orphan">' + esc + '</span>');
      } else if (kind === 'singleton') {
        pieces.push('<span class="' + st.singletonClassFor(pairId) + '">' + esc + '</span>');
      } else if (kind === 'toggle') {
        // Self-paired delimiter: close nearest matching open, else treat as open.
        var matchAtT = findFromTop(stack, pairId);
        if (matchAtT >= 0) {
          orphanAbove(pieces, stack, matchAtT);
          var openIdxT = stack[matchAtT][1];
          var clsT = st.nextColorClass();
          pieces[openIdxT] = '<span class="' + clsT + '">' + pieces[openIdxT] + '</span>';
          pieces.push('<span class="' + clsT + '">' + esc + '</span>');
          stack.length = matchAtT;
        } else {
          pieces.push(esc);
          stack.push([pairId, pieces.length - 1]);
        }
      } else if (kind === 'close') {
        var matchAt = findFromTop(stack, pairId);
        if (matchAt >= 0) {
          orphanAbove(pieces, stack, matchAt);
          var openIdx = stack[matchAt][1];
          // Per-instance color: every matched pair gets a fresh palette index.
          void colorOverride;
          var cls = st.nextColorClass();
          pieces[openIdx] = '<span class="' + cls + '">' + pieces[openIdx] + '</span>';
          pieces.push('<span class="' + cls + '">' + esc + '</span>');
          stack.length = matchAt;
        } else {
          pieces.push('<span class="tt-orphan">' + esc + '</span>');
        }
      } else {  // open
        pieces.push(esc);
        stack.push([pairId, pieces.length - 1]);
      }
      last = re.lastIndex;
    }
    for (var i = 0; i < stack.length; i++) {
      var idx = stack[i][1];
      pieces[idx] = '<span class="tt-orphan">' + pieces[idx] + '</span>';
    }
    if (last < text.length) { pieces.push(escapeHtml(text.slice(last))); }
    return pieces.join('');
  }

  function findFromTop(stack, pairId) {
    for (var i = stack.length - 1; i >= 0; i--) {
      if (stack[i][0] === pairId) { return i; }
    }
    return -1;
  }
  // Everything left un-closed above the matched open is orphaned.
  function orphanAbove(pieces, stack, matchAt) {
    for (var i = matchAt + 1; i < stack.length; i++) {
      var ui = stack[i][1];
      pieces[ui] = '<span class="tt-orphan">' + pieces[ui] + '</span>';
    }
  }

  function colorizeMarkup(text, family, markersMap) {
    text = text == null ? '' : String(text);
    if (family === 'harmony' && harmonyRe().test(text)) {
      return colorizeHarmony(text);
    }
    return colorizeXml(text, family, markersMap);
  }

  // --- Stream deltas: intervals computed over the JOINED text, then sliced per chunk,
  // so a tag spanning a chunk boundary keeps one color (mirrors colorize_stream_deltas).
  function xmlTokenIntervals(text, family, markersMap) {
    var declared = declaredLookup(family, markersMap);
    var st = new State();
    var intervals = [];
    var stack = [];
    var re = tagRe();
    var m;
    while ((m = re.exec(text)) !== null) {
      var tok = m[0];
      if (tok === MINIMAX_NS_TOKEN) {
        intervals.push({ start: m.index, end: re.lastIndex, cls: NS_CLASS });
        continue;
      }
      var idx = intervals.length;
      var kind, pairId;
      var hit = declared[tok];
      if (hit != null) {
        kind = hit[0]; pairId = hit[1];
      } else {
        var k = tagKindAndName(tok.slice(1, -1));
        kind = k[0]; pairId = k[1];
      }
      intervals.push({ start: m.index, end: re.lastIndex, cls: null });
      if (kind === null) {
        intervals[idx].cls = 'tt-orphan';
      } else if (kind === 'singleton') {
        intervals[idx].cls = st.singletonClassFor(pairId);
      } else if (kind === 'toggle') {
        var matchAtT = findFromTop(stack, pairId);
        if (matchAtT >= 0) {
          orphanAboveIntervals(intervals, stack, matchAtT);
          var clsT = st.nextColorClass();
          intervals[stack[matchAtT][1]].cls = clsT;
          intervals[idx].cls = clsT;
          stack.length = matchAtT;
        } else {
          stack.push([pairId, idx]);
        }
      } else if (kind === 'close') {
        var matchAt = findFromTop(stack, pairId);
        if (matchAt >= 0) {
          orphanAboveIntervals(intervals, stack, matchAt);
          var cls = st.nextColorClass();
          intervals[stack[matchAt][1]].cls = cls;
          intervals[idx].cls = cls;
          stack.length = matchAt;
        } else {
          intervals[idx].cls = 'tt-orphan';
        }
      } else {  // open
        stack.push([pairId, idx]);
      }
    }
    for (var i = 0; i < stack.length; i++) {
      intervals[stack[i][1]].cls = 'tt-orphan';
    }
    return intervals;
  }
  function orphanAboveIntervals(intervals, stack, matchAt) {
    for (var i = matchAt + 1; i < stack.length; i++) {
      intervals[stack[i][1]].cls = 'tt-orphan';
    }
  }

  function harmonyIntervals(text) {
    var intervals = [];
    var re = harmonyRe();
    var m;
    while ((m = re.exec(text)) !== null) {
      var tokenName = m[1];
      var end;
      if (HARMONY_TURN_CLOSE[tokenName]) {
        end = re.lastIndex;
      } else {
        var re2 = harmonyRe();
        re2.lastIndex = re.lastIndex;
        var nextM = re2.exec(text);
        end = nextM ? nextM.index : text.length;
      }
      intervals.push({
        start: m.index,
        end: end,
        cls: HARMONY_SEGMENT_CLASS[tokenName] || 'tt-h-other',
        tokenStart: m.index,
        tokenEnd: re.lastIndex,
        tokenName: tokenName,
        harmony: true,
      });
    }
    return intervals;
  }

  function markupIntervals(text, family, markersMap) {
    if (family === 'harmony' && harmonyRe().test(text)) {
      return harmonyIntervals(text);
    }
    return xmlTokenIntervals(text, family, markersMap);
  }

  function renderHarmonyIntervalSlice(text, interval, start, end) {
    var tokenStart = interval.tokenStart;
    var tokenEnd = interval.tokenEnd;
    var parts = [];
    var cursor = start;
    var tokenSliceStart = Math.max(start, tokenStart);
    var tokenSliceEnd = Math.min(end, tokenEnd);
    if (cursor < tokenSliceStart) {
      parts.push(escapeHtml(text.slice(cursor, tokenSliceStart)));
      cursor = tokenSliceStart;
    }
    if (tokenSliceStart < tokenSliceEnd) {
      parts.push('<span class="tt-h-token">'
        + escapeHtml(text.slice(tokenSliceStart, tokenSliceEnd)) + '</span>');
      cursor = tokenSliceEnd;
    }
    if (cursor < end) { parts.push(escapeHtml(text.slice(cursor, end))); }
    return '<span class="tt-h ' + interval.cls + '">' + parts.join('') + '</span>';
  }

  function renderIntervalSlice(text, interval, start, end) {
    if (interval.harmony) {
      return renderHarmonyIntervalSlice(text, interval, start, end);
    }
    return '<span class="' + (interval.cls || 'tt-orphan') + '">'
      + escapeHtml(text.slice(start, end)) + '</span>';
  }

  function colorizeMarkupSlice(text, intervals, start, end) {
    var pieces = [];
    var cursor = start;
    for (var i = 0; i < intervals.length; i++) {
      var interval = intervals[i];
      if (interval.end <= start) { continue; }
      if (interval.start >= end) { break; }
      var overlapStart = Math.max(start, interval.start);
      var overlapEnd = Math.min(end, interval.end);
      if (cursor < overlapStart) { pieces.push(escapeHtml(text.slice(cursor, overlapStart))); }
      pieces.push(renderIntervalSlice(text, interval, overlapStart, overlapEnd));
      cursor = overlapEnd;
    }
    if (cursor < end) { pieces.push(escapeHtml(text.slice(cursor, end))); }
    return pieces.join('');
  }

  // chunks: [{delta_text}, ...] -> one colored HTML string per chunk.
  function colorizeStreamDeltas(chunks, family, markersMap) {
    var deltas = (chunks || []).map(function (ch) {
      return (ch && ch.delta_text != null) ? String(ch.delta_text) : '';
    });
    var text = deltas.join('');
    var intervals = markupIntervals(text, family, markersMap);
    var rendered = [];
    var cursor = 0;
    for (var i = 0; i < deltas.length; i++) {
      var end = cursor + deltas[i].length;
      rendered.push(colorizeMarkupSlice(text, intervals, cursor, end));
      cursor = end;
    }
    return rendered;
  }

  // --- Linked rendering ------------------------------------------------------
  // Presentation the page actually uses (the glyph-coloring colorizeMarkup above is
  // kept as the tested reference of the pairing algorithm). Two ideas:
  //   * MARKER tokens (`<think>`, `<|tool_call|>`, harmony tokens, ...) get a BACKGROUND
  //     color in their pair color — open and close share it, distinct pairs differ, so
  //     the markers are told apart. An UNMATCHED / error marker gets a bright red bg.
  //   * USER TEXT gets a per-WORD FOREGROUND color keyed by the word, so the SAME word
  //     always renders the same color everywhere in one tooltip — across the input cell,
  //     every output cell, and every stream chunk (adjacent distinct words alternate as
  //     the palette cycles). That is the left<->right / chunk<->assembled "words match".
  var HUE_PALETTE_SIZE = 8;

  // THE INPUT DRIVES THE COLOR SCHEME. A tooltip is colored in two phases against one
  // shared ctx:
  //   1. REGISTER — tokenize the INPUT's user text and give each token a hue. This is
  //      the tooltip's whole vocabulary; nothing else ever mints a color.
  //   2. MATCH (after sealLinkCtx) — every render, input and output alike, scans its
  //      text left-to-right taking the LONGEST vocabulary token that matches at each
  //      position. Output text the input never contained stays plain.
  // So an output value carries exactly the color it had on the input side, and a parser
  // that concatenates two inputs shows both source colors in the joined result.
  function newLinkCtx() { return { hue: {}, seq: 0, vocab: [], markers: {}, sealed: false }; }

  // Token rule: a content run splits on STRUCTURAL characters — anything outside
  // [A-Za-z0-9_ ] (quotes, braces, colons, commas, newlines). Spaces stay INSIDE a
  // token, so prose stays whole while structured text splits into its values:
  //   `partial reasoning` -> one token `partial reasoning`      (one color)
  //   `{"mode": "fast"}`  -> tokens `mode` and `fast`           (two colors)
  //   `preamble`/`body`/`answer` -> three tokens, which is what lets the output's
  //   concatenated `bodyanswer` match back as `body` + `answer` in the input's colors.
  // A STRING runs until a QUOTING/STRUCTURAL character — `"` `{` `}` `[` `]` `:`.
  // Sentence punctuation (`,` `.` `!` `'` `?` `*`) stays INSIDE the phrase, so
  // `Let's go! How's it going? Yeah, maybe.` is one continuous string rather than six.
  // Those are what separate one value from the next, so `{"mode": "fast"}` still splits
  // into `mode` and `fast`. Everything else, including sentence punctuation AND the
  // whitespace around it, stays INSIDE the string: the run before harmony's `<|start|>`
  // is `I need both. ` — one string whose trailing space belongs to it, not a detached
  // `I need both` + `.` + ` `. Whitespace inside renders with a background so it shows.
  var TOKEN_RUN_RE = /[^"{}\[\]:]+/g;
  // A WORD inside a run: letters, digits, underscore, hyphen. Each run registers both
  // the whole phrase AND its individual words, so `partial reasoning` stays one color
  // when it moves as a unit, while `NYC` / `EST` / `get_weather` still match on their
  // own wherever an output mentions only one of them.
  var WORD_RE = /[A-Za-z0-9_-]+/g;

  // `' NYC '` and `'NYC'` are the SAME string, differing only in padding a parser chose
  // to keep or trim — so the TRIMMED form owns the hue and every surface form reuses it.
  // That is what lets vLLM Rust's `" NYC "` and Dynamo's `"NYC"` read as one value.
  function addToken(tok, ctx) {
    if (!tok || !/\S/.test(tok) || (tok in ctx.hue)) { return; }
    var key = tok.replace(/^\s+|\s+$/g, '');
    var hue = ctx.hue[key];
    if (hue == null) { hue = ctx.seq % HUE_PALETTE_SIZE; ctx.seq += 1; ctx.hue[key] = hue; }
    ctx.hue[tok] = hue;
    ctx.vocab.push(tok);
  }

  function registerTokens(sub, ctx) {
    var m;
    TOKEN_RUN_RE.lastIndex = 0;
    while ((m = TOKEN_RUN_RE.exec(sub)) !== null) {
      var raw = m[0];
      var run = raw.replace(/^\s+|\s+$/g, '');
      if (!run) { continue; }
      addToken(run, ctx);   // hue owner: the string without its padding
      addToken(raw, ctx);   // the padded form matches too, at the SAME hue
      // ALWAYS index the individual words too, not just for whitespace-separated
      // phrases: `to=functions.get_weather` is one run with no spaces, and an output
      // that reports only `get_weather` still has to match it. Longest-match keeps the
      // whole run winning wherever it applies.
      var w;
      WORD_RE.lastIndex = 0;
      while ((w = WORD_RE.exec(run)) !== null) { addToken(w[0], ctx); }
      WORD_RE.lastIndex = 0;
    }
  }

  // Freeze the vocabulary and order it longest-first, so the match scan below is greedy:
  // `partial reasoning` wins over a bare `reasoning` wherever both could apply.
  function sealLinkCtx(ctx) {
    if (!ctx) { return; }
    ctx.vocab.sort(function (a, b) { return b.length - a.length; });
    ctx.sealed = true;
  }

  // THE match primitive, shared by the plain and the chunk-sliced renderers: greedy
  // longest vocabulary match, returned as {start,end,hue} offsets relative to `sub`.
  // Characters no token claims are left out — they render plain.
  function isWordChar(ch) { return ch !== '' && /[A-Za-z0-9_-]/.test(ch); }

  // Does a vocabulary token start exactly at `pos`? Used for the right-boundary test.
  function startsMatch(sub, pos, ctx) {
    for (var k = 0; k < ctx.vocab.length; k++) {
      var t = ctx.vocab[k];
      if (t.length <= sub.length - pos && sub.substr(pos, t.length) === t) { return true; }
    }
    return false;
  }

  function matchSegments(sub, ctx) {
    var out = [];
    var i = 0;
    var prevEnd = -1;
    while (i < sub.length) {
      var hit = null;
      for (var k = 0; k < ctx.vocab.length; k++) {
        var t = ctx.vocab[k];
        if (t.length > sub.length - i || sub.substr(i, t.length) !== t) { continue; }
        // Don't light up a WORD FRAGMENT: `weather` must not match inside `get_weather`,
        // and `at` must not match inside `analysis`. A match may still begin or end
        // mid-word when it butts against another match — that is what resolves a
        // concatenation like `bodyanswer` into `body` + `answer`.
        var end = i + t.length;
        if (isWordChar(t.charAt(0)) && i > 0 && i !== prevEnd
            && isWordChar(sub.charAt(i - 1))) { continue; }
        if (isWordChar(t.charAt(t.length - 1)) && end < sub.length
            && isWordChar(sub.charAt(end)) && !startsMatch(sub, end, ctx)) { continue; }
        hit = t;
        break;
      }
      if (hit) {
        out.push({ start: i, end: i + hit.length, hue: ctx.hue[hit] });
        i += hit.length;
        prevEnd = i;
      } else {
        i += 1;
      }
    }
    return out;
  }

  // User text with no markup around it (the `calls=` blob, emitted name/arguments).
  // Before sealing this only harvests the vocabulary (the caller discards the markup);
  // after sealing it colors by matching against it.
  function contentSpan(sub, ctx) {
    if (!sub) { return ''; }
    if (!ctx) { return escapeHtml(sub); }
    if (!ctx.sealed) { registerTokens(sub, ctx); return escapeHtml(sub); }
    return renderSegments(sub, matchSegments(sub, ctx), 0, sub.length, false);
  }
  // interval class (tt-c3 / tt-orphan / tt-ns / tt-h-start...) -> marker background class.
  // tt-orphan (unmatched / error) -> the bright-red bg class.
  function markerClass(cls) { return 'tt-mbg tt-mbg-' + String(cls).replace(/^tt-/, ''); }

  // A harmony HEADER token carries a keyword that belongs to the tag, not to the
  // content: `<|start|>assistant`, `<|channel|>analysis`, `<|constrain|>json`. Verified
  // across the harmony + harmony_text fixtures — `<|start|>` is followed by `assistant`
  // 63/63, `<|constrain|>` by `json` 104/104, `<|channel|>` by a channel name. Absorb
  // that leading keyword into the marker so the tag and its value share one color.
  //
  // Deliberately NOT `<|message|>`: its payload is the message BODY (`{"location":
  // "NYC"}`, prose), which has to stay ordinary content so it still matches the parser
  // output. Nor `<|call|>`/`<|end|>`, which are turn terminators whose trailing text is
  // narration (` Done.`). Only `<|channel|>`'s leading keyword is taken, so the
  // recipient in `<|channel|>commentary to=functions.get_weather` stays matchable.
  // Keyed on the token LITERAL, not the family: `harmony` runs through the state machine
  // (harmonyIntervals) while `harmony_text` carries the same tokens as declared markers
  // through xmlTokenIntervals. Both must absorb, so match the text of the marker itself.
  // Gemma 4's `<|channel>` (pipe on the LEFT only — not harmony's `<|channel|>`) carries a
  // role label that the parser treats as part of the envelope: gemma4_parser.rs declares
  // `START_TOKEN = "<|channel>"` alongside `THOUGHT_PREFIX = "thought\n"` and calls that
  // label "a structural artefact", stripping it before the reasoning text begins. So
  // `<|channel>thought` is ONE unit and is grouped here.
  //
  // Gemma's `<|tool_call>call:` is deliberately NOT grouped: the tool parser declares
  // `TOOL_CALL_START = "<|tool_call>"` and `CALL_PREFIX = "call:"` as SEPARATE constants,
  // and `detect_tool_call_start_gemma4("call:get_weather{")` is true — `call:` can open a
  // call on its own, so fusing the two would misrepresent the grammar.
  var HARMONY_HEADER_LITERALS = {
    '<|start|>': true, '<|channel|>': true, '<|constrain|>': true, '<|channel>': true,
  };

  function absorbHeaderKeywords(text, intervals) {
    return intervals.map(function (iv, i) {
      var b = markerBounds(iv);
      if (!HARMONY_HEADER_LITERALS[text.slice(b[0], b[1])]) { return iv; }
      // Never grow past the next marker.
      var limit = text.length;
      for (var j = i + 1; j < intervals.length; j++) {
        var nb = markerBounds(intervals[j]);
        if (nb[0] >= b[1]) { limit = nb[0]; break; }
      }
      var kw = /^[A-Za-z0-9_]+/.exec(text.slice(b[1], limit));
      if (!kw) { return iv; }
      var grown = {};
      for (var k in iv) {
        if (Object.prototype.hasOwnProperty.call(iv, k)) { grown[k] = iv[k]; }
      }
      if (iv.harmony) { grown.tokenEnd = b[1] + kw[0].length; } else { grown.end = b[1] + kw[0].length; }
      return grown;
    });
  }

  // NOTE: only the LINKED renderer absorbs keywords. markupIntervals/colorizeMarkup stay
  // byte-identical to markup.py (test_colorize_parity pins them).
  // Any harmony-transport family takes the harmony state machine, not just the family
  // literally named `harmony`. `harmony_text` is the SAME gpt-oss envelope shipped as text
  // instead of token-ids, but it was falling through to the generic XML/pipe matcher, which
  // has no idea what these tokens mean: it read `<|call|>` as a close tag, found no opener,
  // and painted it ORPHAN RED, while the harmony path knows it is a turn terminator
  // (`h-call`). Same bytes, two different renderings, and the red one was a false alarm.
  //
  // Matched on the family PREFIX rather than on the token shape: `harmonyRe()` also matches
  // kimi's `<|tool_calls_section_begin|>`, so dispatching on the text alone would drag
  // unrelated families onto the harmony path.
  function isHarmonyFamily(family) { return /^harmony/.test(family || ''); }

  function linkedTokenize(text, family, markersMap) {
    var intervals = (isHarmonyFamily(family) && harmonyRe().test(text))
      ? harmonyIntervals(text)
      : xmlTokenIntervals(text, family, markersMap);
    return absorbHeaderKeywords(text, intervals);
  }

  // The MARKER span of an interval: for harmony that is just the token, the related
  // text after it is ordinary user content.
  function markerBounds(iv) {
    return iv.harmony ? [iv.tokenStart, iv.tokenEnd] : [iv.start, iv.end];
  }

  // The content (non-marker) ranges of the whole text, in order.
  function contentGaps(text, intervals) {
    var gaps = [];
    var cursor = 0;
    for (var i = 0; i < intervals.length; i++) {
      var b = markerBounds(intervals[i]);
      if (cursor < b[0]) { gaps.push([cursor, b[0]]); }
      if (b[1] > cursor) { cursor = b[1]; }
    }
    if (cursor < text.length) { gaps.push([cursor, text.length]); }
    return gaps;
  }

  function registerContent(text, intervals, ctx) {
    contentGaps(text, intervals).forEach(function (g) {
      registerTokens(text.slice(g[0], g[1]), ctx);  // the WHOLE run, never a chunk slice
    });
    // Remember which marker literals the INPUT actually contained. An unmatched marker
    // in OUTPUT text that the input never had was INJECTED by the parser, not leaked
    // through it — those are opposite findings and must not share the alarm color.
    for (var i = 0; i < intervals.length; i++) {
      var b = markerBounds(intervals[i]);
      if (b[0] < b[1]) { ctx.markers[text.slice(b[0], b[1])] = true; }
    }
  }

  // Segment the ENTIRE text once — marker spans plus matched word spans, in order.
  // Rendering any sub-range then just CLIPS these segments, which is what keeps a word
  // straddling a chunk boundary one color: `comment` at the end of one chunk and `ary`
  // at the start of the next are two clips of the single `commentary` segment, so they
  // carry the same hue. Matching per chunk-slice instead would score them as two
  // different words.
  function segmentText(text, intervals, ctx) {
    var segs = [];
    for (var i = 0; i < intervals.length; i++) {
      var b = markerBounds(intervals[i]);
      if (b[0] < b[1]) {
        var cls = intervals[i].cls;
        if (cls === 'tt-orphan' && ctx && ctx.sealed && !ctx.markers[text.slice(b[0], b[1])]) {
          cls = 'tt-injected';
        }
        segs.push({ start: b[0], end: b[1], cls: markerClass(cls) });
      }
    }
    contentGaps(text, intervals).forEach(function (g) {
      matchSegments(text.slice(g[0], g[1]), ctx).forEach(function (s) {
        segs.push({ start: g[0] + s.start, end: g[0] + s.end, hue: s.hue });
      });
    });
    segs.sort(function (a, b2) { return a.start - b2.start; });
    return segs;
  }

  // One matched string, split so its WHITESPACE is visible: word characters take the
  // hue as a foreground color, the whitespace between/around them takes the same hue as
  // a background. Same hue index either way — it is one string, shown as one color.
  function hueSpans(sub, hue) {
    var out = '';
    var re = /\s+|\S+/g;
    var m;
    while ((m = re.exec(sub)) !== null) {
      var cls = /\s/.test(m[0]) ? ('tt-ws tt-ws' + hue) : ('tt-fg tt-fg' + hue);
      out += '<span class="' + cls + '">' + escapeHtml(m[0]) + '</span>';
    }
    return out;
  }

  // Whitespace that belongs to no string is still whitespace the MODEL emitted: between
  // two markers a space, a newline and a tab all render identically, and the YAML folding
  // bug proved that difference matters. Give it a neutral background so it is visible —
  // a hue would be wrong, because it is not part of any matched value.
  function plainSpans(sub, markWs) {
    if (!markWs) { return escapeHtml(sub); }
    var out = '';
    var re = /\s+|\S+/g;
    var m;
    while ((m = re.exec(sub)) !== null) {
      out += /\s/.test(m[0])
        ? '<span class="tt-wsp">' + escapeHtml(m[0]) + '</span>'
        : escapeHtml(m[0]);
    }
    return out;
  }

  // `markWs` is on for model INPUT text (where spacing is part of the grammar) and off
  // for the emitted `calls=`/deltas JSON, where every separator space would be noise.
  function renderSegments(text, segs, start, end, markWs) {
    var out = '';
    var cursor = start;
    for (var i = 0; i < segs.length; i++) {
      var s = segs[i];
      if (s.end <= start) { continue; }
      if (s.start >= end) { break; }
      var a = Math.max(start, s.start);
      var b = Math.min(end, s.end);
      if (cursor < a) { out += plainSpans(text.slice(cursor, a), markWs); }
      if (s.cls) {
        out += '<span class="' + s.cls + '">' + escapeHtml(text.slice(a, b)) + '</span>';
      } else {
        out += hueSpans(text.slice(a, b), s.hue);
      }
      cursor = b;
    }
    if (cursor < end) { out += plainSpans(text.slice(cursor, end), markWs); }
    return out;
  }

  function colorizeLinked(text, family, markersMap, ctx) {
    text = text == null ? '' : String(text);
    var intervals = linkedTokenize(text, family, markersMap);
    if (ctx && !ctx.sealed) { registerContent(text, intervals, ctx); return escapeHtml(text); }
    return renderSegments(text, segmentText(text, intervals, ctx), 0, text.length, true);
  }

  // Stream analogue: markers AND words are resolved over the JOINED text, then sliced
  // per chunk — so anything split across a chunk boundary keeps one color.
  function colorizeLinkedStreamDeltas(chunks, family, markersMap, ctx) {
    var deltas = (chunks || []).map(function (ch) {
      return (ch && ch.delta_text != null) ? String(ch.delta_text) : '';
    });
    var text = deltas.join('');
    var intervals = linkedTokenize(text, family, markersMap);
    if (ctx && !ctx.sealed) {
      registerContent(text, intervals, ctx);
      return deltas.map(escapeHtml);
    }
    var segs = segmentText(text, intervals, ctx);
    var rendered = [];
    var cursor = 0;
    for (var i = 0; i < deltas.length; i++) {
      var end = cursor + deltas[i].length;
      rendered.push(renderSegments(text, segs, cursor, end, true));
      cursor = end;
    }
    return rendered;
  }

  return {
    escapeHtml: escapeHtml,
    colorizeMarkup: colorizeMarkup,
    colorizeStreamDeltas: colorizeStreamDeltas,
    newLinkCtx: newLinkCtx,
    sealLinkCtx: sealLinkCtx,
    colorizeLinked: colorizeLinked,
    colorizeLinkedStreamDeltas: colorizeLinkedStreamDeltas,
    // Word coloring with NO marker parsing — for text that carries no markup but must
    // still match the rest of the tooltip word-for-word (the `calls=` JSON blob).
    colorizeWords: contentSpan,
  };
});
