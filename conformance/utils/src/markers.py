# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parser-comparison and marker semantics for the conformance table (audit B5).

This module owns the *comparison* logic — given the captured `expected:` blocks for
each implementation, decide each cell's status (`ok`/`problem`/`na`/`todo`), the
per-engine parser marker (`=`, `↯`, `!`, `✗`, `n/a`, `…`, `·`), and the cross-engine
conformance markers (`D_rb`, `V_ps`, `S_rs`, …). It is deliberately split out from the
HTML rendering so a UI change cannot accidentally change parser comparison logic.

Identity (keys/aliases/display/letters) comes from `impls.py`; this module has no
dependency on the rendering, fixture-loading, or tooltip code, so it is the leaf of
the generator's import graph.
"""
import html as html_lib
import json
import re
from pathlib import Path
from typing import Any

import yaml

from impls import (
    BASELINE_IMPLS,
    BASELINE_STREAM_IMPL,
    BATCH_IMPL_KEYS,
    ENGINE_LETTER,
    IMPL_DISPLAY,
    IMPL_KEYS,
    IMPL_LANG_MARKER,
    LEGACY_IMPL_ALIASES,
    PARSER_NOT_CAPTURED,
    PEER_IMPL_KEYS,
    STREAM_IMPL_KEYS,
)

_STREAM_MODE_MARKER = "s"
_BATCH_MODE_MARKER = "b"
VLLM_RUST_UNAVAILABLE = (
    "vLLM Rust source not available; set VLLM_RUST_SOURCE and run the Rust capture probe."
)

_IMPL_DISPLAY = IMPL_DISPLAY


def _canonical_impl_key(impl: str) -> str:
    return LEGACY_IMPL_ALIASES.get(impl, impl)


def _legacy_impl_keys(impl: str) -> list[str]:
    return [old for old, new in LEGACY_IMPL_ALIASES.items() if new == impl]


def _impl_get(mapping: object, impl: str, default: Any = None) -> Any:
    if not isinstance(mapping, dict):
        return default
    if impl in mapping:
        return mapping[impl]
    for legacy in _legacy_impl_keys(impl):
        if legacy in mapping:
            return mapping[legacy]
    return default


def _normalize_impl_mapping(mapping: object) -> dict:
    if not isinstance(mapping, dict):
        return {}
    normalized: dict = {}
    for key, value in mapping.items():
        canonical = _canonical_impl_key(str(key))
        if canonical not in normalized:
            normalized[canonical] = value
    return normalized


def _expected(case: dict | None) -> dict:
    if not isinstance(case, dict):
        return {}
    expected = _normalize_impl_mapping(case.get("expected") or {})
    if expected and "vllm_rust" not in expected:
        expected["vllm_rust"] = {"unavailable": VLLM_RUST_UNAVAILABLE}
    return expected


def peer_status(case: dict, dyn: dict, impl: str) -> tuple[str, bool]:
    """Returns (kind, is_unknown).

    kind:
      'na'      — peer key missing from `expected:` (block not recorded)
      'match'   — peer is anchor ref to Dynamo Rust, or value-equal to Dynamo Rust
      'unavail' — peer block is `{unavailable: <msg>}`
      'err'     — peer block is `{error: <substring>}`
      'div'     — peer block is a concrete divergent {calls, normal_text}
    is_unknown is True iff kind == 'div' AND block has no `explanation:`.
    """
    block = _impl_get(case.get("expected") or {}, impl)
    if block is None:
        return ("na", False)
    if block is dyn:
        return ("match", False)
    if not isinstance(block, dict):
        return ("na", False)
    if "unavailable" in block:
        return ("unavail", False)
    if "error" in block:
        return ("err", False)
    if "calls" in block or "normal_text" in block:
        # Value-equal to Dynamo Rust (non-anchor)? Treat as match.
        n_block = {
            "calls": block.get("calls") or [],
            "normal_text": block.get("normal_text") or "",
        }
        n_dyn = {
            "calls": dyn.get("calls") or [],
            "normal_text": dyn.get("normal_text") or "",
        }
        if n_block == n_dyn:
            return ("match", False)
        return ("div", _explanation(block) is None)
    return ("na", False)


# Tool-call markup leak detector (the `↯` marker): derived from the union of every
# family's declared grammar tokens in parser_families.yaml `markers:` (DIS-2442) —
# registering a family's tokens there is the ONLY step to make its leaks detectable.
#
# Truncation cases (taxonomy group 5) leak PARTIAL markers (`<tool_call`,
# `[/TOOL_CALLS`, `<｜tool▁sep｜`), so each pair/singleton token also matches by any
# prefix of at least _LEAK_PREFIX_MIN chars — the stem semantics the old hardcoded
# regex had (`</?tool_call`, `TOOL_CALLS`, ...). Tokens shorter than the minimum
# match whole only (short stems are false-positive-prone). `leak:` entries are
# already stems of parameterized tags (e.g. "<invoke" for `<invoke name="x">`) and
# are matched verbatim. Substring semantics (no anchors), longest-first alternation.
_LEAK_PREFIX_MIN = 7


def parser_families_path() -> Path:
    """The registry lives at src/parser_families.yaml in the repo; the render stage
    copies this module flat into tests/parity/ and the registry to <stage>/src/.

    Public because every consumer of the registry has to resolve BOTH layouts, and a
    second copy of this walk is a second thing to get wrong -- `unified_taxonomy` did
    exactly that and broke the render, which only runs from the stage."""
    here = Path(__file__).resolve()
    for cand in (
        here.parent / "parser_families.yaml",
        here.parents[2] / "src" / "parser_families.yaml",
    ):
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"parser_families.yaml not found relative to {here}")


def _declared_leak_patterns() -> list[str]:
    spec = yaml.safe_load(parser_families_path().read_text())
    patterns: set[str] = set()
    for decl in (spec.get("markers") or {}).values():
        tokens = [t for pair in decl.get("pairs") or [] for t in pair]
        tokens += decl.get("singletons") or []
        for tok in tokens:
            patterns.update(tok[:n] for n in range(_LEAK_PREFIX_MIN, len(tok) + 1))
            patterns.add(tok)
        patterns.update(decl.get("leak") or [])
    if not patterns:
        raise ValueError("parser_families.yaml declares no family markers")
    return sorted(patterns, key=len, reverse=True)


_TOOL_CALL_MARKUP_RE = re.compile(
    "|".join(re.escape(p) for p in _declared_leak_patterns())
)


def _explanation(block: object) -> str | None:
    """The intentional-divergence note on an expected block. `explanation` is the
    current key; `reason` is the legacy spelling still present in older fixtures and
    Dynamo-synced code. Read both (explanation wins); new fixtures/captures write
    `explanation`."""
    if not isinstance(block, dict):
        return None
    v = block.get("explanation")
    return v if v is not None else block.get("reason")


def _dynamo_tool_call_leak(dyn: dict) -> str | None:
    normal_text = dyn.get("normal_text")
    note = _explanation(dyn)
    if not note or not isinstance(normal_text, str):
        return None
    if not _TOOL_CALL_MARKUP_RE.search(normal_text):
        return None
    return str(note)


def _block_tool_call_leaks(block: dict) -> bool:
    normal_text = block.get("normal_text")
    return isinstance(normal_text, str) and bool(
        _TOOL_CALL_MARKUP_RE.search(normal_text)
    )


def _is_exception(block: object) -> bool:
    """The parser RAN and RAISED. Distinct from `unavailable` (no such parser exists):
    an exception is a real, attributable outcome and always colours as a problem, while
    an absent parser is neutral. Every classifier below goes through this predicate so
    the exception contract lives in exactly one place.
    """
    return isinstance(block, dict) and "exception" in block


def _overview_status(case: dict | None, impl: str) -> str:
    if case is None or "expected" not in case:
        return "na"
    block = _impl_get(case.get("expected") or {}, impl)
    if not isinstance(block, dict) or "unavailable" in block:
        if _is_parser_error_unavailable(block):
            return "problem"
        # A family the Dynamo v2 stream parser doesn't implement is a plain neutral
        # n/a (like the v1 table, which has no "TODO" concept) — not a distinct
        # orange "todo" state.
        return "na"
    if _is_exception(block) or "error" in block or _block_tool_call_leaks(block):
        return "problem"
    return "ok"


def _impl_keys_for_output_kind(output_kind: str) -> tuple[str, ...]:
    return BATCH_IMPL_KEYS if output_kind == "batch" else STREAM_IMPL_KEYS


def _canonical_tool_output(block: object) -> dict | None:
    if not isinstance(block, dict) or "unavailable" in block or "error" in block:
        return None
    if "calls" not in block and "normal_text" not in block:
        return None
    return {
        "calls": block.get("calls") or [],
        "normal_text": block.get("normal_text") or "",
    }


def _selected_parity_marker(
    case: dict | None,
    impl: str,
    impl_keys: tuple[str, ...] = BATCH_IMPL_KEYS,
    marker_mode: str | None = _BATCH_MODE_MARKER,
) -> str | None:
    """Cross-engine conformance marker (batch / stream tabs): the letters of the
    other engines whose canonical output differs from the selected one (`=` when
    all three agree). Returns None — the caller falls back to the per-engine status
    marker — when any engine lacks output. (The stream tabs do NOT use this; their
    color carries stream-vs-own-batch (`_sob_status`) and their marker carries
    cross-engine STREAM agreement (`_stream_xeng_marker`).)
    """
    if case is None or "expected" not in case:
        return None
    if impl not in impl_keys:
        return None
    expected = _expected(case)
    outputs = {
        eng: _canonical_tool_output(_impl_get(expected, eng))
        for eng in impl_keys
    }
    if outputs.get(impl) is None:
        return None
    available = {eng: out for eng, out in outputs.items() if out is not None}
    if len(available) < 2:
        return None
    if len({json.dumps(out, ensure_ascii=False, sort_keys=True) for out in available.values()}) == 1:
        return "="
    selected = outputs[impl]
    marker = "".join(
        (
            _impl_mode_letter(peer) + _impl_mode_suffix(peer, marker_mode)
            if marker_mode is not None
            else ENGINE_LETTER[peer]
        )
        for peer in impl_keys
        if peer != impl and outputs[peer] is not None and outputs[peer] != selected
    )
    return marker or "="


def _selected_parity_suffix(case: dict | None, impl: str) -> str:
    if case is None or "expected" not in case:
        return ""
    block = _impl_get(case.get("expected") or {}, impl)
    if isinstance(block, dict) and _block_tool_call_leaks(block):
        return "↯"
    return ""


def _parity_marker(
    case: dict | None,
    impl: str,
    impl_keys: tuple[str, ...] = BATCH_IMPL_KEYS,
    marker_mode: str | None = _BATCH_MODE_MARKER,
) -> str:
    marker = _selected_parity_marker(case, impl, impl_keys, marker_mode)
    if marker is None:
        return _parser_marker(case, impl)
    return _selected_parity_suffix(case, impl) + marker


def _is_todo_unavailable(block: object) -> bool:
    """True when a dynamo unavailable block is a not-yet-implemented TODO
    (v2 streaming work), not a structural n/a."""
    if not isinstance(block, dict):
        return False
    msg = block.get("unavailable", "")
    return isinstance(msg, str) and "not yet implemented" in msg


# An engine `unavailable` block whose reason shows the engine's parser was actually
# invoked and FAILED (threw) — the capture records these as "<impl> parser not
# captured: <error>" or a "parsing failed"/"parse error" message. This is real
# signal (the engine can't parse this input) and gets the `✗` error marker, distinct
# from benign unavailables (no model_text, no parser for the family, Rust source not
# set up), which stay a neutral `n/a`. The primary marker is the shared
# PARSER_NOT_CAPTURED contract the capture wrapper stamps (B11 — not a private
# guess); the rest cover common runtime-throw phrasings any probe may emit (F2).
_PARSER_ERROR_RE = re.compile(
    "|".join(
        re.escape(p)
        for p in (PARSER_NOT_CAPTURED, "parsing failed", "parse error", "panicked", "exception", "traceback")
    ),
    re.I,
)


def _is_parser_error_unavailable(block: object) -> bool:
    if not isinstance(block, dict):
        return False
    msg = block.get("unavailable")
    return isinstance(msg, str) and bool(_PARSER_ERROR_RE.search(msg))


def _parser_marker(case: dict | None, impl: str) -> str:
    if case is None:
        return "—"
    if "expected" not in case:
        return "n/a"
    expected = _expected(case)
    block = _impl_get(expected, impl)
    if not isinstance(block, dict) or "unavailable" in block:
        if _is_parser_error_unavailable(block):
            return "✗"
        # Un-implemented Dynamo v2 family: plain neutral n/a, no distinct "…" TODO
        # marker (matches the v1 table's clean look; see _overview_status).
        return "n/a"
    # A captured `exception` is the parser having RUN and RAISED — same class as a
    # structured error, and never `n/a` (which means the parser does not exist).
    if _is_exception(block):
        return "✗"
    if "error" in block:
        # B11: a structured (dict) error = a peer parser ran and threw -> `✗`;
        # a plain-string error is a declared expected-error -> `!`.
        return "✗" if isinstance(block["error"], dict) else "!"
    if _block_tool_call_leaks(block):
        return "↯"
    if impl in BASELINE_IMPLS:
        peers = [_impl_get(expected, peer) for peer in PEER_IMPL_KEYS]
        if all(
            peer is None or (isinstance(peer, dict) and "unavailable" in peer)
            for peer in peers
        ):
            return "·"
    return ""


def _norm_calls(calls: list) -> list[tuple]:
    """Normalize a calls list to [(name, canonical-json-args)] for equality."""
    out = []
    for c in calls or []:
        out.append(
            (c.get("name", ""), json.dumps(c.get("arguments", {}), sort_keys=True, ensure_ascii=False))
        )
    return out


# --- Stream-tab comparison (TC stream v2 + batch-on-stream), two dimensions per cell:
#   COLOR (data-status): each engine's STREAM parse vs its OWN BATCH parse — green if
#     the stream reconstructs the batch result, red if it diverges (mirrors the
#     `conformance_toolcalling_batch_via_stream` Rust test).
#   MARKER (Conformance toggle): each engine's output vs the OTHER engines' outputs —
#     `=` when the available streams agree, else the differing engines' letters with a
#     two-letter suffix. The suffix is implementation language (`r` Rust, `p` Python)
#     plus parse mode (`s` stream, `b` batch). The default marker (toggle off) stays
#     leak-only.


def _impl_mode_suffix(impl: str, mode: str) -> str:
    return f"_{IMPL_LANG_MARKER[impl]}{mode}"


def _impl_mode_letter(impl: str) -> str:
    # vLLM Python and vLLM Rust share the visible `V` prefix; the subscript carries
    # the implementation language (`p`/`r`).
    return "V" if impl == "vllm_rust" else ENGINE_LETTER[impl]


def _stream_cross_suffix(impl: str, marker_context: str | None) -> str:
    # Batch-on-stream still reports streaming parser output. Batch markers are
    # reserved for the batch reference shown in the tooltip/reason text.
    return _impl_mode_suffix(impl, _STREAM_MODE_MARKER)


def _stream_parity_explainer_html(marker_context: str | None) -> str:
    del marker_context
    return (
        "Red means that engine's stream parser diverges from its batch parser. "
        "A <code>≠</code> corner mark is a KNOWN v1-vs-v2 divergence "
        "(known-divergences.yaml): calls agree, normal_text differs by design — "
        "the popup's v2 block carries the explanation. "
        "There is no <code>V_rb</code>; vLLM Rust has stream parser capture only. "
        "Harmony captured against vLLM 0.23.0 / SGLang 0.5.12.post1."
    )


def _sob_calls_consistent(case: dict, impl: str) -> bool | None:
    """True/False if the engine's stream calls match its batch calls; None when
    there's nothing to compare (no stream output or no batch reference)."""
    stream = _impl_get(case.get("expected") or {}, impl)
    batch = _impl_get(case.get("batch_expected") or {}, impl)
    if not isinstance(stream, dict) or not isinstance(batch, dict):
        return None
    if "calls" not in batch and "normal_text" not in batch:
        return None
    return _norm_calls(stream.get("calls")) == _norm_calls(batch.get("calls"))


def _sob_status(case: dict | None, impl: str) -> str:
    if case is None:
        return "na"
    stream = _impl_get(case.get("expected") or {}, impl)
    if not isinstance(stream, dict) or "unavailable" in stream:
        if _is_parser_error_unavailable(stream):
            return "problem"
        return "na"
    if _is_exception(stream) or "error" in stream or _block_tool_call_leaks(stream):
        return "problem"
    consistent = _sob_calls_consistent(case, impl)
    if consistent is None:
        return "ok"
    return "ok" if consistent else "problem"


def _stream_xeng_marker(case: dict | None, impl: str, marker_context: str | None = None) -> str:
    """Conformance marker for the stream tabs, two parts concatenated:
      - own-batch: `X_rs`/`X_ps` when this engine's stream diverges from its OWN batch
        parse (the same condition that reddens the cell — e.g. `D_rs` for Dynamo).
      - cross-engine: the OTHER engines' letters with a context suffix (`V_ps` for
        vLLM Python stream output, including batch-on-stream) for engines whose output differs
        from this one (needs >=2 available outputs).
    Returns the `↯` leak prefix + own-batch token + cross-engine tokens, `=` when
    none, or the per-engine status marker (`n/a`) when this engine has no
    stream output."""
    if case is None:
        return "—"
    expected = _expected(case)
    sel_block = _impl_get(expected, impl)
    if not isinstance(sel_block, dict) or "unavailable" in sel_block:
        return _parser_marker(case, impl)
    leak = "↯" if _block_tool_call_leaks(sel_block) else ""
    # own-batch divergence (X_rs/X_ps): this engine's stream != its own batch parse.
    own = (
        _impl_mode_letter(impl) + _impl_mode_suffix(impl, _STREAM_MODE_MARKER)
        if _sob_calls_consistent(case, impl) is False
        else ""
    )
    # cross-engine (Y_rs/Y_ps or Y_rb/Y_pb): other engines whose output differs from this one.
    outputs = {
        e: _canonical_tool_output(_impl_get(expected, e)) for e in IMPL_KEYS
    }
    available = {e: o for e, o in outputs.items() if o is not None}
    selected = available.get(impl)
    cross = ""
    if selected is not None and len(available) >= 2:
        cross = "".join(
            _impl_mode_letter(e) + _stream_cross_suffix(e, marker_context)
            for e in IMPL_KEYS
            if e in available and e != impl and available[e] != selected
        )
    return leak + (own + cross or "=")


def _sob_cell_text(case: dict | None, marker_context: str | None = None) -> str:
    """Static/overview cell text: the Dynamo cross-engine marker (=, V_ps/V_rs/S_rs, …)."""
    return _stream_xeng_marker(case, BASELINE_STREAM_IMPL, marker_context)


# --- Structured comparison model (DIS-2434) --------------------------------------
# The JSON data model + JS view replace the `D_rb`/`V_ps` marker mini-language: the
# model carries STRUCTURED comparison facts and the view decides how to display them.
# These functions are the single source of the per-cell comparison payload; the old
# glyph/attr emitters above are retired once every page renders from the model.


def _canon_call_for_sig(call: object) -> object:
    """A call with its `arguments` decoded when it is a JSON string, so the signature
    compares argument VALUES, not serialization bytes. The v1 parser serializes
    arguments from a HashMap (key order varies per capture) while the v2 stream parser
    pins source order — byte-comparing the strings flagged a divergence on every
    multi-arg call even when the decoded values were identical. `sort_keys=True` in the
    dump then makes key order irrelevant; genuine value/type differences (e.g. `"2"` vs
    `2`) still differ."""
    if not isinstance(call, dict):
        return call
    args = call.get("arguments")
    if isinstance(args, str):
        try:
            return {**call, "arguments": json.loads(args)}
        except (json.JSONDecodeError, ValueError):
            return call
    return call


def candidate_sig(block: object) -> str:
    """Canonical signature of a candidate's output; equal signatures = same output."""
    if not isinstance(block, dict) or "unavailable" in block:
        return "na"
    # A parser that RAN and RAISED is not the same as one that does not exist. Its
    # signature carries the verbatim message so two different exceptions stay distinct
    # (and differ from any real output), instead of collapsing into one "error" bucket.
    if "exception" in block:
        return f"exc:{block.get('exception')}"
    if "error" in block:
        return f"err:{block.get('error')}"
    calls = [_canon_call_for_sig(c) for c in block.get("calls") or []]
    return json.dumps(
        {"calls": calls, "normal_text": block.get("normal_text") or ""},
        sort_keys=True, ensure_ascii=False,
    )


def cmp_entry(sig: int, *, leak: int = 0, na: int = 0, err: int = 0) -> dict:
    """One compare-payload entry. EVERY producer builds entries through this so the key
    set cannot drift: the Unified tab assembles its entries by hand from verdicts rather
    than from expected-blocks, and it silently lacked `err` when the JS view started
    reading it."""
    return {"sig": sig, "leak": leak, "na": na, "err": err}


def cmp_model(blocks: dict) -> dict[str, dict]:
    """Per-cell compare payload from {candidate_key: block}: {key: {sig, leak, na, err}}.
    `sig` is a per-cell group id (candidates with identical output share an id); `na`
    (unavailable) is excluded from the diff count but still shown in the tooltip. This
    is the structured form the JS view consumes directly (the old page HTML-escaped a
    `json.dumps` of the same dict into `data-cmp`).

    `err` marks a parser that RAN and THREW — distinct from `na` (no such parser). The
    difference is what the cell colour means: a thrown Reference beside a peer that
    parsed is a real disagreement (red), while a missing parser is simply absent (grey).
    Both flags are produced HERE so the generator and assets/conformance.js consume one
    contract rather than each inferring exceptions from the block shape."""
    ids: dict[str, int] = {}
    out: dict[str, dict] = {}
    for key, block in blocks.items():
        sig = candidate_sig(block)
        out[key] = cmp_entry(
            ids.setdefault(sig, len(ids)),
            leak=1 if (isinstance(block, dict) and _block_tool_call_leaks(block)) else 0,
            na=1 if sig == "na" else 0,
            err=1 if sig.startswith("exc:") else 0,
        )
    return out


def _error_kind(block: object) -> str | None:
    """Classify an expected block's error/unavailable signal for the model facts:
    `parser_error` (peer parser ran and threw), `expected_error` (declared error),
    `unavailable` (benign: no parser/text/source), or None."""
    if not isinstance(block, dict):
        return None
    if "unavailable" in block:
        return "parser_error" if _is_parser_error_unavailable(block) else "unavailable"
    if "error" in block:
        return "parser_error" if isinstance(block["error"], dict) else "expected_error"
    return None


def comparison_facts(
    case: dict | None,
    impl_keys: tuple[str, ...],
    baseline: str,
) -> list[dict]:
    """Structured per-impl comparison facts for one cell — the retired marker strings'
    replacement. One entry per impl in `impl_keys`:
      impl        — implementation key
      status      — overview status: ok | problem | na
      present     — the impl recorded a concrete output block for this case
      agrees      — output value-equals the baseline impl's output (None if either side
                    has no concrete output to compare)
      intentional — the divergence is documented (block has an `explanation`/`reason`)
      reason      — that explanation text, if any
      leak        — the block leaks tool-call markup into normal_text
      error_kind  — parser_error | expected_error | unavailable | None
    The VIEW turns these into whatever glyph/label it wants; Python keeps deciding the
    agree/intentional/status semantics (single source, no JS reimplementation)."""
    if not isinstance(case, dict) or "expected" not in case:
        return [
            {"impl": impl, "status": "na", "present": False, "agrees": None,
             "intentional": False, "reason": None, "leak": False, "error_kind": None}
            for impl in impl_keys
        ]
    expected = _expected(case)
    base_block = _impl_get(expected, baseline)
    base_out = _canonical_tool_output(base_block)
    facts: list[dict] = []
    for impl in impl_keys:
        block = _impl_get(expected, impl)
        out = _canonical_tool_output(block)
        agrees: bool | None
        if out is None or base_out is None:
            agrees = None if impl != baseline else True
        else:
            agrees = out == base_out
        reason = _explanation(block) if isinstance(block, dict) else None
        facts.append({
            "impl": impl,
            "status": _overview_status(case, impl),
            "present": out is not None,
            "agrees": agrees,
            "intentional": reason is not None,
            "reason": reason,
            "leak": isinstance(block, dict) and _block_tool_call_leaks(block),
            "error_kind": _error_kind(block),
        })
    return facts
