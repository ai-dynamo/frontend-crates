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
from typing import Any

from impls import (
    BASELINE_IMPL,
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


_TOOL_CALL_MARKUP_RE = re.compile(
    r"</?tool_call|</?tool_calls|<\|tool_call|<\|tool_calls|"
    r"<\|(?:channel|message|call|python_tag)\|>|"
    r"</?TOOLCALL|TOOL_CALLS|<｜(?:DSML｜)?(?:tool|tool▁call|tool▁calls)|"
    r"<｜DSML｜|</?minimax:tool_call|</?invoke|</?arg_key|</?arg_value"
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
    if "error" in block or _block_tool_call_leaks(block):
        return "problem"
    return "ok"


def _impl_keys_for_output_kind(output_kind: str) -> tuple[str, ...]:
    return BATCH_IMPL_KEYS if output_kind == "batch" else STREAM_IMPL_KEYS


def _overview_status_attrs(case: dict | None, impl_keys: tuple[str, ...] = BATCH_IMPL_KEYS) -> str:
    parts = [
        f'data-status-{impl}="{_overview_status(case, impl)}"'
        for impl in impl_keys
    ]
    # Per-version status (data-status-<impl>-<slug>) powers the TC v1 version radios.
    # Only batch cases carry __ver_status; other cells fall back to the pinned attr.
    ver_status = case.get("__ver_status") if isinstance(case, dict) else None
    if ver_status:
        for impl, by_slug in ver_status.items():
            for slug, info in by_slug.items():
                parts.append(f'data-status-{impl}-{slug}="{info["status"]}"')
    return " ".join(parts)


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
    if "error" in block:
        # B11: a structured (dict) error = a peer parser ran and threw -> `✗`;
        # a plain-string error is a declared expected-error -> `!`.
        return "✗" if isinstance(block["error"], dict) else "!"
    if _block_tool_call_leaks(block):
        return "↯"
    if impl == BASELINE_IMPL:
        peers = [_impl_get(expected, peer) for peer in PEER_IMPL_KEYS]
        if all(
            peer is None or (isinstance(peer, dict) and "unavailable" in peer)
            for peer in peers
        ):
            return "·"
    return ""


def _parser_marker_attrs(
    case: dict | None,
    impl_keys: tuple[str, ...] = BATCH_IMPL_KEYS,
    marker_mode: str | None = _BATCH_MODE_MARKER,
) -> str:
    attrs = [
        f'data-marker-{impl}="{html_lib.escape(_parser_marker(case, impl))}"'
        for impl in impl_keys
    ]
    attrs.extend(
        f'data-marker-parity-{impl}="{html_lib.escape(_parity_marker(case, impl, impl_keys, marker_mode))}"'
        for impl in impl_keys
    )
    return " ".join(attrs)


def _marker_html(marker: str) -> str:
    """Render marker suffixes like `V_ps` and `D_rb` with real HTML subscript."""
    parts: list[str] = []
    i = 0
    while i < len(marker):
        ch = marker[i]
        if ch in set(ENGINE_LETTER.values()):
            suffix = None
            marker_len = 1
            match = re.match(r"_(?:[rp][sb])", marker[i + 1 :])
            if match:
                suffix = match.group(0)[1:]
                marker_len += len(match.group(0))
            if suffix is not None:
                parts.append(f"{html_lib.escape(ch)}<sub>{html_lib.escape(suffix.upper())}</sub>")
                i += marker_len
                continue
        parts.append(html_lib.escape(ch))
        i += 1
    return "".join(parts)


def _marker_span_html(
    markers: dict[str, str],
    parity_markers: dict[str, str],
    impl_keys: tuple[str, ...] = BATCH_IMPL_KEYS,
) -> str:
    parts: list[str] = []
    for impl in impl_keys:
        parts.append(
            f'<span class="cell-marker marker-{impl}"><span class="marker-text">{_marker_html(markers[impl])}</span></span>'
        )
    for impl in impl_keys:
        parts.append(
            f'<span class="cell-marker marker-parity-{impl}"><span class="marker-text">{_marker_html(parity_markers[impl])}</span></span>'
        )
    return "".join(parts)


def _parser_marker_spans(
    case: dict | None,
    impl_keys: tuple[str, ...] = BATCH_IMPL_KEYS,
    marker_mode: str | None = _BATCH_MODE_MARKER,
) -> str:
    return _marker_span_html(
        {impl: _parser_marker(case, impl) for impl in impl_keys},
        {impl: _parity_marker(case, impl, impl_keys, marker_mode) for impl in impl_keys},
        impl_keys,
    )


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
#     `parity_toolcalling_batch_via_stream` Rust test).
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


def _impl_mode_marker_html(impl: str, mode: str) -> str:
    return f"{_impl_mode_letter(impl)}<sub>{html_lib.escape(IMPL_LANG_MARKER[impl].upper() + mode.upper())}</sub>"


def _impl_mode_label_html(impl: str, mode: str) -> str:
    parse_mode = "stream" if mode == _STREAM_MODE_MARKER else "batch"
    return f"{_impl_mode_marker_html(impl, mode)} ({_IMPL_DISPLAY[impl]} {parse_mode} parser)"


def _stream_cross_suffix(impl: str, marker_context: str | None) -> str:
    # Batch-on-stream still reports streaming parser output. Batch markers are
    # reserved for the batch reference shown in the tooltip/reason text.
    return _impl_mode_suffix(impl, _STREAM_MODE_MARKER)


def _stream_parity_explainer_html(marker_context: str | None) -> str:
    del marker_context
    return (
        "Red means that engine's stream parser diverges from its batch parser. "
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
    if "error" in stream or _block_tool_call_leaks(stream):
        return "problem"
    consistent = _sob_calls_consistent(case, impl)
    if consistent is None:
        return "ok"
    return "ok" if consistent else "problem"


def _sob_status_attrs(case: dict | None) -> str:
    return " ".join(
        f'data-status-{impl}="{_sob_status(case, impl)}"'
        for impl in IMPL_KEYS
    )


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


def _sob_marker_attrs(case: dict | None, marker_context: str | None = None) -> str:
    # Default marker (Conformance OFF) = leak-only, same as every tab. Conformance
    # marker (Conformance ON) = cross-engine output agreement; the color
    # (data-status) carries the stream-vs-own-batch result.
    attrs = [
        f'data-marker-{impl}="{html_lib.escape(_parser_marker(case, impl))}"'
        for impl in IMPL_KEYS
    ]
    attrs.extend(
        f'data-marker-parity-{impl}="{html_lib.escape(_stream_xeng_marker(case, impl, marker_context))}"'
        for impl in IMPL_KEYS
    )
    return " ".join(attrs)


def _sob_marker_spans(case: dict | None, marker_context: str | None = None) -> str:
    return _marker_span_html(
        {impl: _parser_marker(case, impl) for impl in IMPL_KEYS},
        {impl: _stream_xeng_marker(case, impl, marker_context) for impl in IMPL_KEYS},
        IMPL_KEYS,
    )


def _sob_cell_text(case: dict | None, marker_context: str | None = None) -> str:
    """Static/overview cell text: the Dynamo cross-engine marker (=, V_ps/V_rs/S_rs, …)."""
    return _stream_xeng_marker(case, BASELINE_IMPL, marker_context)
