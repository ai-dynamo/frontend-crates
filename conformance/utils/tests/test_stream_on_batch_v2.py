# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the conformance-table marker semantics.

Contract pinned here:

  * DEFAULT view (Conformance toggle OFF), every tab: per-engine status only —
    empty when clean, `↯` when the engine leaks tool-call markup, plus
    `…`/`n/a`/`·`/`!`/`—`. NEVER `=` or a divergence letter.
  * CONFORMANCE view (toggle ON):
      - batch tab: cross-engine `=`/`D`/`V`/`S`.
      - stream tabs (batch-on-stream + TC stream v2): two dimensions per cell —
        COLOR (`data-status`) is each engine's STREAM parse vs its OWN BATCH parse
        (green consistent, red "problem" on divergence); MARKER
        (`data-marker-parity`) concatenates an own-batch token (`Xᵦ` when the
        engine's stream differs from its own batch) and cross-engine STREAM
        tokens (`Yₛ` for other engines whose stream differs), `=` when all agree.

The stream tabs flow through the same render_cell_html/render_row_html/
render_html_panel pipeline as the others; only the `comparison` strategy differs.
"""

from __future__ import annotations

import itertools
import re
import sys
from pathlib import Path

UTILS = Path(__file__).resolve().parents[1]
if str(UTILS) not in sys.path:
    sys.path.insert(0, str(UTILS))

import generate_conformance_table_v2 as g  # noqa: E402

_TODO_MSG = "Dynamo parser v2 stream parser not yet implemented for this family"


def _markers(html: str) -> dict[str, str]:
    return dict(re.findall(r'(data-(?:status|marker(?:-parity)?)-\w+)="([^"]*)"', html))


def _xcase(expected: dict) -> dict:
    """Cross-engine case (batch/stream tab shape)."""
    return {
        "__family": "fam",
        "__case_id": "TOOLCALLING.batch.1",
        "__fixture_path": "toolcalling/fixtures/fam/TOOLCALLING.batch.1.yaml",
        "description": "demo",
        "model_text": "hello",
        "expected": expected,
    }


def _sobcase(stream: dict, batch: dict) -> dict:
    """Batch-on-stream case: stream output (`expected`) + batch reference."""
    return {
        "__family": "harmony",
        "__case_id": "TOOLCALLING.batch.5.d",
        "__fixture_path": "toolcalling/fixtures/harmony/TOOLCALLING.batch.5.yaml",
        "description": "demo",
        "model_text": "hello",
        "expected": stream,
        "batch_expected": batch,
    }


def _calls(*names):
    return {"calls": [{"name": n, "arguments": {}} for n in names], "normal_text": ""}


# --------------------------------------------------------------------------- #
# _stream_on_batch_expected: overlay -> standard expected block
# --------------------------------------------------------------------------- #
def test_expected_dynamo_absent_renders_as_todo() -> None:
    exp = g._stream_on_batch_expected({"vllm": {"calls": []}, "sglang": {"calls": []}})
    assert g._is_todo_unavailable(exp["dynamo"])
    assert "reason" in exp["vllm"] and "reason" in exp["sglang"]


def test_expected_dynamo_present_and_peer_unavailable() -> None:
    exp = g._stream_on_batch_expected(
        {
            "dynamo": {"calls": [{"name": "f", "arguments": {}}], "normal_text": ""},
            "vllm": {"calls": []},
            "sglang": {"unavailable": "SGLang has no detector for family"},
        }
    )
    assert exp["dynamo"]["calls"] == [{"name": "f", "arguments": {}}]
    assert "reason" not in exp["dynamo"]
    assert exp["sglang"] == {"unavailable": "SGLang has no detector for family"}


# --------------------------------------------------------------------------- #
# DEFAULT marker is leak-only on every tab (the reported bug)
# --------------------------------------------------------------------------- #
def test_parser_marker_never_emits_equals_or_divergence_letters() -> None:
    allowed = {"", "↯", "!", "·", "…", "n/a"}
    blocks = [
        {"calls": [], "normal_text": ""},
        {"calls": [{"name": "f", "arguments": {"a": 1}}], "normal_text": ""},
        {"unavailable": "structural n/a"},
        {"unavailable": _TODO_MSG},
        {"error": "boom"},
        {"calls": [], "normal_text": "<tool_call>leak"},
    ]
    for combo in itertools.product(blocks, repeat=3):
        case = _xcase(dict(zip(("dynamo", "vllm", "sglang"), combo)))
        for impl in ("dynamo", "vllm", "sglang"):
            assert g._parser_marker(case, impl) in allowed


# --------------------------------------------------------------------------- #
# stream tabs: 2-D cell — COLOR = stream-vs-own-batch, MARKER = cross-engine streams
# --------------------------------------------------------------------------- #
def test_sob_cell_two_dimensions_color_and_cross_engine_marker() -> None:
    # Stream outputs: dynamo got nothing, vllm/sglang each got [f].
    # Batch refs:     dynamo recovered [f], vllm [f], sglang got nothing.
    # COLOR (stream-vs-own-batch): dynamo stream([]) != batch([f]) -> problem;
    #   vllm stream([f]) == batch([f]) -> ok; sglang stream([f]) != batch([]) -> problem.
    # MARKER (cross-engine streams): dynamo([]) differs from vllm & sglang -> VₛSₛ;
    #   vllm([f]) differs only from dynamo -> Dₛ; sglang([f]) differs only from dynamo -> Dₛ.
    case = _sobcase(
        stream={
            "dynamo": _calls(),  # stream got nothing
            "vllm": _calls("f"),
            "sglang": _calls("f"),
        },
        batch={
            "dynamo": _calls("f"),  # batch recovered it
            "vllm": _calls("f"),
            "sglang": _calls(),
        },
    )
    html = g.render_cell_html(case, "batch", "harmony", "5.d", "stream", "stream_vs_batch")
    m = _markers(html)
    # DEFAULT: leak-only (all clean) -> empty, never a letter
    assert m["data-marker-dynamo"] == ""
    assert m["data-marker-vllm"] == ""
    assert m["data-marker-sglang"] == ""
    # COLOR: stream-vs-own-batch divergence is red (problem); agreement is green (ok)
    assert m["data-status-dynamo"] == "problem"
    assert m["data-status-vllm"] == "ok"
    assert m["data-status-sglang"] == "problem"
    # MARKER (Conformance): own-batch token (Xᵦ when stream != own batch) +
    # cross-engine streams (Yₛ). dynamo: diverges from its batch (Dᵦ) and from both
    # peer streams (VₛSₛ). vllm: matches its batch, differs only from dynamo (Dₛ).
    # sglang: diverges from its batch (Sᵦ), differs only from dynamo stream (Dₛ).
    assert m["data-marker-parity-dynamo"] == "DᵦVₛSₛ"
    assert m["data-marker-parity-vllm"] == "Dₛ"
    assert m["data-marker-parity-sglang"] == "SᵦDₛ"


def test_sob_marker_all_consistent_is_equals_and_green() -> None:
    case = _sobcase(
        stream={i: _calls("f") for i in ("dynamo", "vllm", "sglang")},
        batch={i: _calls("f") for i in ("dynamo", "vllm", "sglang")},
    )
    m = _markers(g.render_cell_html(case, "batch", "harmony", "1", "stream", "stream_vs_batch"))
    for impl in ("dynamo", "vllm", "sglang"):
        assert m[f"data-marker-{impl}"] == ""  # default: nothing
        assert m[f"data-marker-parity-{impl}"] == "="  # conformance: consistent
        assert m[f"data-status-{impl}"] == "ok"  # green


def test_sob_dynamo_todo_when_no_v2_parser() -> None:
    case = _sobcase(
        stream={
            "dynamo": {"unavailable": _TODO_MSG},
            "vllm": _calls("f"),
            "sglang": _calls("f"),
        },
        batch={i: _calls("f") for i in ("dynamo", "vllm", "sglang")},
    )
    assert g._sob_status(case, "dynamo") == "todo"
    # dynamo unavailable -> marker falls back to the per-engine status (…)
    assert g._stream_xeng_marker(case, "dynamo") == "…"
    # vllm/sglang streams both present and agree -> cross-engine '='
    assert g._stream_xeng_marker(case, "vllm") == "="


def test_sob_leak_marks_red_and_lightning() -> None:
    case = _sobcase(
        stream={
            "dynamo": {"calls": [], "normal_text": "<tool_call>leaked"},
            "vllm": _calls("f"),
            "sglang": _calls("f"),
        },
        batch={i: _calls("f") for i in ("dynamo", "vllm", "sglang")},
    )
    m = _markers(g.render_cell_html(case, "batch", "harmony", "1", "stream", "stream_vs_batch"))
    assert m["data-marker-dynamo"] == "↯"  # leak shows in default
    assert m["data-status-dynamo"] == "problem"  # red


def test_sob_tooltip_labels_stream_and_batch() -> None:
    case = _sobcase(
        stream={i: _calls("f") for i in ("dynamo", "vllm", "sglang")},
        batch={i: _calls("f") for i in ("dynamo", "vllm", "sglang")},
    )
    ttip = g._build_sob_tooltip(case)
    for impl in ("Dynamo", "vLLM", "SGLang"):
        assert f"{impl} stream:" in ttip
        assert f"{impl} batch:" in ttip


# --------------------------------------------------------------------------- #
# cross-engine conformance (batch / stream tabs) — unchanged, all 3 required
# --------------------------------------------------------------------------- #
def test_cross_engine_all_three_agree() -> None:
    case = _xcase({i: {"calls": [], "normal_text": ""} for i in ("dynamo", "vllm", "sglang")})
    assert g._selected_parity_marker(case, "dynamo") == "="


def test_cross_engine_requires_all_three_present() -> None:
    case = _xcase(
        {
            "dynamo": {"unavailable": "x"},
            "vllm": {"calls": [{"name": "f", "arguments": {}}], "normal_text": ""},
            "sglang": {"calls": [], "normal_text": ""},
        }
    )
    # One engine missing -> no cross-engine comparison (caller falls back to status).
    assert g._selected_parity_marker(case, "vllm") is None


def test_cross_engine_divergence_letters() -> None:
    case = _xcase(
        {
            "dynamo": {"calls": [{"name": "f", "arguments": {}}], "normal_text": ""},
            "vllm": {"calls": [], "normal_text": ""},
            "sglang": {"calls": [], "normal_text": ""},
        }
    )
    assert g._selected_parity_marker(case, "dynamo") == "VS"
    assert g._selected_parity_marker(case, "vllm") == "D"


# --------------------------------------------------------------------------- #
# _build_stream_on_batch_cases: overlay + batch reference
# --------------------------------------------------------------------------- #
def test_build_cases_carries_stream_and_batch(monkeypatch) -> None:
    batch_cases = {
        ("fam", "1"): {
            "__case_id": "TOOLCALLING.batch.1",
            "__fixture_path": "toolcalling/fixtures/fam/TOOLCALLING.batch.1.yaml",
            "description": "d",
            "model_text": "x",
            "expected": {"dynamo": _calls("f"), "vllm": _calls("f"), "sglang": _calls("f")},
        },
        ("fam", "3"): {
            "__case_id": "TOOLCALLING.batch.3",
            "__fixture_path": "toolcalling/fixtures/fam/TOOLCALLING.batch.3.yaml",
        },
    }
    overlay = {("fam", "TOOLCALLING.batch.1"): {"vllm": {"calls": []}, "sglang": {"calls": []}}}
    monkeypatch.setattr(g, "_load_stream_on_batch_overlay", lambda: overlay)

    cases = g._build_stream_on_batch_cases(batch_cases)
    assert ("fam", "1") in cases and ("fam", "3") not in cases
    built = cases[("fam", "1")]
    assert built["model_text"] == "x"
    assert g._is_todo_unavailable(built["expected"]["dynamo"])  # dynamo absent -> todo
    assert built["batch_expected"]["vllm"] == _calls("f")  # batch reference carried
    # COLOR: vllm stream calls=[] vs batch calls=[f] -> diverge -> problem (red)
    assert g._sob_status(built, "vllm") == "problem"
    # MARKER: vllm diverges from its own batch (Vᵦ); dynamo todo, sglang stream also
    # empty so no cross-engine token.
    assert g._stream_xeng_marker(built, "vllm") == "Vᵦ"
