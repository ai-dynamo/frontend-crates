# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fold semantics of resolve_stream_fixtures.py.

Regression for the doubled-call bug: `_merge_impl`'s docstring promises that when a
version doc lists a case it "replaces the impl's prior state entirely", but the code
only overwrote chunk indices `0..len(overlay.chunks)`. A higher-version doc with FEWER
chunks than the base (Dynamo v1 3.0.0 records 2 chunks against a 6-chunk input, while
the v2 0.1.11 anchor emits in chunk 3) left the lower version's deltas in the tail
chunks, so assembly concatenated both versions:
`get_weatherget_weather("{"location":"NYC"}{"location":"NYC"}")`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

UTILS = Path(__file__).resolve().parents[1]
SRC = UTILS / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from resolve_stream_fixtures import resolve  # noqa: E402

FAMILY = "famx"
CASE = "TOOLCALLING.streamv2.1"
NAME = "TOOLCALLING.streamv2.1.yaml"


def _write(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(doc, sort_keys=False))


def _make_tree(root: Path) -> None:
    # Shared inputs: 6 chunks (the real deepseek streamv2.1 shape).
    _write(root / "inputs" / FAMILY / NAME, {
        "family": FAMILY,
        "mode": "streamv2",
        "cases": {CASE: {"chunks": [
            {"delta_text": "<tc> <invoke"},
            {"delta_text": ' name="get_weather">'},
            {"delta_text": ' <param name="location">'},
            {"delta_text": "NYC</param> </invoke>"},
            {"delta_text": " </tc>"},
            {"delta_text": "", "finish_reason": "tool_calls"},
        ]}},
    })
    # v2 anchor (lowest version): emits name then args in chunk 3.
    _write(root / "dynamo_rust-0.1.11" / FAMILY / NAME, {
        "family": FAMILY,
        "mode": "streamv2",
        "captured_with": {"dynamo_rust": "0.1.11"},
        "cases": {CASE: {"chunks": [
            {"expected": []},
            {"expected": []},
            {"expected": []},
            {"expected": [
                {"index": 0, "name": "get_weather", "arguments": ""},
                {"index": 0, "arguments": '{"location":"NYC"}'},
            ]},
            {"expected": []},
            {"expected": []},
        ]}},
    })
    # v1 overlay (higher version): a DIFFERENT parser, and its doc carries FEWER
    # chunks than the base — the shape that triggered the bug.
    _write(root / "dynamo_rust-3.0.0" / FAMILY / NAME, {
        "family": FAMILY,
        "mode": "streamv2",
        "captured_with": {"dynamo_rust": "3.0.0"},
        "cases": {CASE: {"chunks": [
            {"expected": [
                {"index": 0, "id": True, "name": "get_weather",
                 "arguments": '{"location":"NYC"}'},
            ]},
            {"expected": []},
        ]}},
    })


def _dynamo_deltas(folded_case: dict) -> list[tuple[int, dict]]:
    """(chunk_index, delta) for every dynamo_rust delta in the folded case."""
    out = []
    for i, ch in enumerate(folded_case.get("chunks") or []):
        for delta in ((ch.get("expected") or {}).get("dynamo_rust") or []):
            out.append((i, delta))
    return out


def test_higher_version_with_fewer_chunks_fully_replaces_lower(tmp_path):
    root = tmp_path / "sv2"
    out = tmp_path / "out"
    _make_tree(root)

    resolve(root, out, select=["dynamo_rust-3.0.0"])

    folded = yaml.safe_load((out / FAMILY / NAME).read_text())
    case = folded["cases"][CASE]
    deltas = _dynamo_deltas(case)

    # Exactly the 3.0.0 doc's content: ONE delta, in chunk 0 — no v2 residue anywhere.
    assert [i for i, _ in deltas] == [0], (
        f"v2 anchor deltas leaked into tail chunks: indices {[i for i, _ in deltas]}"
    )
    # Assembly must be a single, un-doubled call.
    name = "".join(d.get("name") or "" for _, d in deltas)
    args = "".join(d.get("arguments") or "" for _, d in deltas)
    assert name == "get_weather", f"doubled/mangled name: {name!r}"
    assert args == '{"location":"NYC"}', f"doubled/mangled arguments: {args!r}"


def test_lower_target_keeps_anchor_untouched(tmp_path):
    # Selecting the anchor version itself must reproduce the anchor exactly.
    root = tmp_path / "sv2"
    out = tmp_path / "out"
    _make_tree(root)

    resolve(root, out, select=["dynamo_rust-0.1.11"])

    folded = yaml.safe_load((out / FAMILY / NAME).read_text())
    deltas = _dynamo_deltas(folded["cases"][CASE])
    assert [i for i, _ in deltas] == [3, 3]
    assert "".join(d.get("name") or "" for _, d in deltas) == "get_weather"
