# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`build_stream_fixtures._q` must round-trip a scalar through YAML unchanged.

A single-quoted YAML scalar FOLDS an embedded newline into a space when read back, so
the previous `_q` silently corrupted every `delta_text` containing one. The concrete
damage: DeepSeek V3's tool-call payload is delimited by `V3_JSON_START = "\\n```json\\n"`
(vLLM `rust/src/tool-parser/src/deepseek_json/mod.rs`). Once folded to
`get_weather ```json {...}` the delimiter no longer matches, the parser never enters the
JSON state, and `finish()` reports "incomplete DeepSeek V3 tool call" -- turning 19
previously-passing conformance cases into errors that look like parser regressions.
"""

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import build_stream_fixtures  # noqa: E402

# The real DeepSeek V3 shape is first: it is the case that actually broke.
ROUND_TRIP_CASES = [
    'get_weather\n```json\n{"location": "NYC"}\n```',
    "plain text, no quoting hazard",
    "it's got an apostrophe",
    'it has "double quotes"',
    "back\\slash",
    "tab\tseparated",
    "trailing newline\n",
    "\nleading newline",
    "multi\nline\nvalue",
    "<｜tool▁calls▁begin｜>unicode markers<｜tool▁call▁end｜>",
    "",
]


def _round_trip(value: str):
    """Emit `value` the way the fixture writer does, then read it back with YAML."""
    doc = yaml.safe_load("key: " + build_stream_fixtures._q(value) + "\n")
    return doc["key"]


def test_q_round_trips_every_scalar():
    for value in ROUND_TRIP_CASES:
        assert _round_trip(value) == value, f"_q corrupted {value!r}"


def test_q_preserves_the_deepseek_v3_json_fence():
    """The specific delimiter vLLM's DeepSeek V3 parser matches on must survive."""
    text = 'function<｜tool▁sep｜>get_weather\n```json\n{"location": "NYC"}\n```'
    assert "\n```json\n" in _round_trip(text)


def test_single_quoted_folding_is_why_this_test_exists():
    """Guard the assumption: a single-quoted scalar really does fold newlines."""
    folded = yaml.safe_load("key: 'a\nb'\n")["key"]
    assert folded == "a b"
