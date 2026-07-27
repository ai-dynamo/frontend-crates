# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Byte-parity between the Python markup colorizer (markup.py) and its JS port
(src/assets/colorize.js). markup.py is the single source of truth; the JS view calls
the port to color tooltip text in the browser. This pins them equal via node so the
two copies can't silently drift.

markup.py carries its palette counter as a process-global across all calls in one
render; the JS port resets per top-level call. We reset the Python globals before each
call here so both sides color a string from its own text alone — the contract the port
documents.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import markup  # noqa: E402

COLORIZE_JS = Path(__file__).resolve().parents[2] / "src" / "assets" / "colorize.js"
MARKERS = markup.declared_markers()

# (family, text) — cover every colorizer path: plain JSON + special token, paired XML,
# nested/per-instance colors, unmatched orphan, harmony segments, MiniMax namespace,
# gemma self-paired quote, declared deepseek markers, and HTML-escape edge chars.
CASES = [
    ("llama3_json", '<|python_tag|>{"name": "get_weather", "arguments": {"location": "NYC"}}'),
    ("hermes", "<tool_call>{\"name\": \"f\"}</tool_call>"),
    ("hermes", "<tool_call>a</tool_call> mid <tool_call>b</tool_call>"),
    ("hermes", "text <tool_call>unclosed here"),
    ("hermes", "orphan close </tool_call> after"),
    ("harmony", "<|start|>assistant<|channel|>final<|message|>hi there<|end|>"),
    ("harmony", "<|start|>x<|channel|>commentary<|call|>run<|return|>"),
    ("minimax_m2", "]<]minimax[>[<tool_call>{\"a\":1}</tool_call>"),
    ("gemma4", '<|"|>quoted value<|"|> and <|"|>dangling'),
    ("deepseek_v3", "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>x<｜tool▁sep｜>y<｜tool▁call▁end｜><｜tool▁calls▁end｜>"),
    ("mistral", "[TOOL_CALLS][{\"name\": \"f\"}]"),
    ("hermes", "escape & < > \" ' chars"),
    (None, "no family, plain <tool_call>t</tool_call> & 'text'"),
]

# (family, chunks) — cross-chunk slicing: a tag split across chunk boundaries must keep
# one color, so per-chunk independent coloring would diverge from the joined-then-sliced
# reference.
STREAM_CASES = [
    ("hermes", [{"delta_text": "<tool_"}, {"delta_text": "call>{\"n"}, {"delta_text": "\":1}</tool_call>"}]),
    ("harmony", [{"delta_text": "<|chan"}, {"delta_text": "nel|>fin"}, {"delta_text": "al<|message|>hi"}]),
    ("hermes", [{"delta_text": "plain "}, {"delta_text": "text & "}, {"delta_text": "more"}]),
]


def _reset():
    markup._color_seq = 0
    markup._singleton_classes.clear()


def _py_markup(text, family):
    _reset()
    return markup.colorize_markup(text, family)


def _py_stream(chunks, family):
    _reset()
    return markup.colorize_stream_deltas(chunks, family)


def _node(fn_body: str, payload: dict) -> list:
    import json

    driver = (
        "const mc = require(" + json.dumps(str(COLORIZE_JS)) + ");"
        "const inp = JSON.parse(require('fs').readFileSync(0, 'utf8'));"
        + fn_body
        + "process.stdout.write(JSON.stringify(out));"
    )
    node = shutil.which("node")
    if not node:
        pytest.skip("node not available")
    res = subprocess.run(
        [node, "-e", driver],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(res.stdout)


def test_colorize_markup_parity():
    js = _node(
        "const out = inp.cases.map(c => mc.colorizeMarkup(c.text, c.family, inp.markers));",
        {"cases": [{"text": t, "family": f} for f, t in CASES], "markers": MARKERS},
    )
    for (family, text), got in zip(CASES, js):
        assert got == _py_markup(text, family), f"family={family!r} text={text!r}"


def test_colorize_stream_deltas_parity():
    js = _node(
        "const out = inp.cases.map(c => mc.colorizeStreamDeltas(c.chunks, c.family, inp.markers));",
        {"cases": [{"chunks": ch, "family": f} for f, ch in STREAM_CASES], "markers": MARKERS},
    )
    for (family, chunks), got in zip(STREAM_CASES, js):
        assert got == _py_stream(chunks, family), f"family={family!r} chunks={chunks!r}"
