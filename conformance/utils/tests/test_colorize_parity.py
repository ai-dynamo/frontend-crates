# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Maintained JavaScript colorizer matches frozen #385 Python outputs."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

COLORIZE_JS = Path(__file__).resolve().parents[1] / "src" / "assets" / "colorize.js"
REGISTRY = Path(__file__).resolve().parents[1] / "src" / "parser_families.yaml"
MARKERS = yaml.safe_load(REGISTRY.read_text())["markers"]

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
    # Opaque argument-value regions (parser_families.yaml `opaque:`): a
    # marker-looking substring INSIDE a value is DATA, so it must not be colored as a
    # control token and must not steal the real pair. One per region shape.
    # qwen3: `<parameter=K>` .. `</parameter>`, an unquoted raw value.
    (
        "qwen3_coder",
        "<tool_call>\n<function=run>\n<parameter=cmd>\n"
        "git log </tool_call> --oneline\n</parameter>\n</function>\n</tool_call>",
    ),
    # qwen3: a reasoning marker in a value stays data too (the parser's I7 case).
    (
        "qwen3_coder",
        "<tool_call><function=log><parameter=note><think>reconsider</think>"
        "</parameter></function></tool_call>",
    ),
    # qwen3: two values in one call, and an unterminated value at EOF.
    (
        "qwen3_coder",
        "<tool_call><function=f><parameter=a>1</parameter>"
        "<parameter=b>2</parameter></function></tool_call>",
    ),
    ("qwen3_coder", "<tool_call>\n<function=run>\n<parameter=cmd>unterminated value"),
    # gemma4: the `<|"|>` quote toggle wraps the value; the `<tool_call|>` inside it
    # would otherwise close the real `<|tool_call>` and orphan the true closer.
    ("gemma4", '<|tool_call>call:run{cmd:<|"|>git log }<tool_call|> --oneline<|"|>}<tool_call|>'),
    ("gemma4", '<|tool_call>call:f{a:<|"|>x<|"|>,b:<|"|>y<|"|>}<tool_call|>'),
    # kimi: the argument blob is JSON and its terminator is the SAME token that can
    # appear inside a string value, so only JSON-string awareness separates them.
    (
        "kimi_k2",
        '<|tool_calls_section_begin|><|tool_call_begin|>functions.run:0'
        '<|tool_call_argument_begin|>{"cmd": "git log <|tool_call_end|> --oneline"}'
        '<|tool_call_end|><|tool_calls_section_end|>',
    ),
    (
        "kimi_k2",
        '<|tool_calls_section_begin|><|tool_call_begin|>functions.f:0'
        '<|tool_call_argument_begin|>{"a": "esc \\" quote <|tool_call_end|>"}'
        '<|tool_call_end|><|tool_calls_section_end|>',
    ),
    # Kimi K3 declarations are composite tokens. Parameterized headers and the
    # canonical/space-padded spellings must win before generic `<|...|>` tokenization.
    (
        "kimi_k3",
        '<|open|>think<|sep|>private<|close|>think<|sep|>'
        '<|open|> call tool="run" index="1" <|sep|>'
        '<|open|> argument key="cmd" type="string" <|sep|>'
        "literal <|close|>call<|sep|> text"
        '<|close|> argument <|sep|><|close|> call <|sep|>',
    ),
    (
        "kimi_k3",
        '<|open|>json type="object"<|sep|>'
        '{"cmd":"literal <|close|>json<|sep|> text"}'
        '<|close|>json<|sep|>'
        '<|open|> message role="assistant" <|sep|>visible'
        '<|close|> message <|sep|>',
    ),
    (None, "no family, plain <tool_call>t</tool_call> & 'text'"),
]

# (family, chunks) — cross-chunk slicing: a tag split across chunk boundaries must keep
# one color, so per-chunk independent coloring would diverge from the joined-then-sliced
# reference.
STREAM_CASES = [
    ("hermes", [{"delta_text": "<tool_"}, {"delta_text": "call>{\"n"}, {"delta_text": "\":1}</tool_call>"}]),
    ("harmony", [{"delta_text": "<|chan"}, {"delta_text": "nel|>fin"}, {"delta_text": "al<|message|>hi"}]),
    ("hermes", [{"delta_text": "plain "}, {"delta_text": "text & "}, {"delta_text": "more"}]),
    # An opaque region split across chunk boundaries: the value (and the
    # marker-looking substring in it) must stay data even though the opener,
    # the embedded token and the closer arrive in different chunks.
    (
        "qwen3_coder",
        [
            {"delta_text": "<tool_call><function=run><parameter=cmd>git log "},
            {"delta_text": "</tool_call> --one"},
            {"delta_text": "line</parameter></function></tool_call>"},
        ],
    ),
    (
        "kimi_k3",
        [
            {"delta_text": '<|open|> call tool="run" index="1" <|'},
            {"delta_text": 'sep|><|open|> argument key="cmd" type="string" <|sep|>'},
            {"delta_text": "literal <|close|>call<|sep|> text<|close|> argument <|sep|>"},
            {"delta_text": "<|close|> call <|sep|>"},
        ],
    ),
]


# Frozen #385 Python-renderer outputs; the production colorizer lives in JavaScript.
EXPECTED_MARKUP = ['<span class="tt-c0">&lt;|python_tag|&gt;</span>{&quot;name&quot;: &quot;get_weather&quot;, &quot;arguments&quot;: {&quot;location&quot;: &quot;NYC&quot;}}', '<span class="tt-c0">&lt;tool_call&gt;</span>{&quot;name&quot;: &quot;f&quot;}<span class="tt-c0">&lt;/tool_call&gt;</span>', '<span class="tt-c0">&lt;tool_call&gt;</span>a<span class="tt-c0">&lt;/tool_call&gt;</span> mid <span class="tt-c1">&lt;tool_call&gt;</span>b<span class="tt-c1">&lt;/tool_call&gt;</span>', 'text <span class="tt-orphan">&lt;tool_call&gt;</span>unclosed here', 'orphan close <span class="tt-orphan">&lt;/tool_call&gt;</span> after', '<span class="tt-h tt-h-start"><span class="tt-h-token">&lt;|start|&gt;</span>assistant</span><span class="tt-h tt-h-channel"><span class="tt-h-token">&lt;|channel|&gt;</span>final</span><span class="tt-h tt-h-message"><span class="tt-h-token">&lt;|message|&gt;</span>hi there</span><span class="tt-h tt-h-stop"><span class="tt-h-token">&lt;|end|&gt;</span></span>', '<span class="tt-h tt-h-start"><span class="tt-h-token">&lt;|start|&gt;</span>x</span><span class="tt-h tt-h-channel"><span class="tt-h-token">&lt;|channel|&gt;</span>commentary</span><span class="tt-h tt-h-call"><span class="tt-h-token">&lt;|call|&gt;</span></span>run<span class="tt-h tt-h-stop"><span class="tt-h-token">&lt;|return|&gt;</span></span>', '<span class="tt-ns">]&lt;]minimax[&gt;[</span><span class="tt-c0">&lt;tool_call&gt;</span>{&quot;a&quot;:1}<span class="tt-c0">&lt;/tool_call&gt;</span>', '<span class="tt-c0">&lt;|&quot;|&gt;</span>quoted value<span class="tt-c0">&lt;|&quot;|&gt;</span> and <span class="tt-orphan">&lt;|&quot;|&gt;</span>dangling', '<span class="tt-c2">&lt;｜tool▁calls▁begin｜&gt;</span><span class="tt-c1">&lt;｜tool▁call▁begin｜&gt;</span>x<span class="tt-c0">&lt;｜tool▁sep｜&gt;</span>y<span class="tt-c1">&lt;｜tool▁call▁end｜&gt;</span><span class="tt-c2">&lt;｜tool▁calls▁end｜&gt;</span>', '<span class="tt-orphan">[TOOL_CALLS]</span>[{&quot;name&quot;: &quot;f&quot;}]', 'escape &amp; <span class="tt-orphan">&lt; &gt;</span> &quot; &#x27; chars', '<span class="tt-c2">&lt;tool_call&gt;</span>\n<span class="tt-c1">&lt;function=run&gt;</span>\n<span class="tt-c0">&lt;parameter=cmd&gt;</span>\ngit log &lt;/tool_call&gt; --oneline\n<span class="tt-c0">&lt;/parameter&gt;</span>\n<span class="tt-c1">&lt;/function&gt;</span>\n<span class="tt-c2">&lt;/tool_call&gt;</span>', '<span class="tt-c2">&lt;tool_call&gt;</span><span class="tt-c1">&lt;function=log&gt;</span><span class="tt-c0">&lt;parameter=note&gt;</span>&lt;think&gt;reconsider&lt;/think&gt;<span class="tt-c0">&lt;/parameter&gt;</span><span class="tt-c1">&lt;/function&gt;</span><span class="tt-c2">&lt;/tool_call&gt;</span>', '<span class="tt-c3">&lt;tool_call&gt;</span><span class="tt-c2">&lt;function=f&gt;</span><span class="tt-c0">&lt;parameter=a&gt;</span>1<span class="tt-c0">&lt;/parameter&gt;</span><span class="tt-c1">&lt;parameter=b&gt;</span>2<span class="tt-c1">&lt;/parameter&gt;</span><span class="tt-c2">&lt;/function&gt;</span><span class="tt-c3">&lt;/tool_call&gt;</span>', '<span class="tt-orphan">&lt;tool_call&gt;</span>\n<span class="tt-orphan">&lt;function=run&gt;</span>\n<span class="tt-orphan">&lt;parameter=cmd&gt;</span>unterminated value', '<span class="tt-c1">&lt;|tool_call&gt;</span>call:run{cmd:<span class="tt-c0">&lt;|&quot;|&gt;</span>git log }&lt;tool_call|&gt; --oneline<span class="tt-c0">&lt;|&quot;|&gt;</span>}<span class="tt-c1">&lt;tool_call|&gt;</span>', '<span class="tt-c2">&lt;|tool_call&gt;</span>call:f{a:<span class="tt-c0">&lt;|&quot;|&gt;</span>x<span class="tt-c0">&lt;|&quot;|&gt;</span>,b:<span class="tt-c1">&lt;|&quot;|&gt;</span>y<span class="tt-c1">&lt;|&quot;|&gt;</span>}<span class="tt-c2">&lt;tool_call|&gt;</span>', '<span class="tt-c2">&lt;|tool_calls_section_begin|&gt;</span><span class="tt-c1">&lt;|tool_call_begin|&gt;</span>functions.run:0<span class="tt-c0">&lt;|tool_call_argument_begin|&gt;</span>{&quot;cmd&quot;: &quot;git log &lt;|tool_call_end|&gt; --oneline&quot;}<span class="tt-c1">&lt;|tool_call_end|&gt;</span><span class="tt-c2">&lt;|tool_calls_section_end|&gt;</span>', '<span class="tt-c2">&lt;|tool_calls_section_begin|&gt;</span><span class="tt-c1">&lt;|tool_call_begin|&gt;</span>functions.f:0<span class="tt-c0">&lt;|tool_call_argument_begin|&gt;</span>{&quot;a&quot;: &quot;esc \\&quot; quote &lt;|tool_call_end|&gt;&quot;}<span class="tt-c1">&lt;|tool_call_end|&gt;</span><span class="tt-c2">&lt;|tool_calls_section_end|&gt;</span>', '<span class="tt-c0">&lt;|open|&gt;think&lt;|sep|&gt;</span>private<span class="tt-c0">&lt;|close|&gt;think&lt;|sep|&gt;</span><span class="tt-c2">&lt;|open|&gt; call tool=&quot;run&quot; index=&quot;1&quot; &lt;|sep|&gt;</span><span class="tt-c1">&lt;|open|&gt; argument key=&quot;cmd&quot; type=&quot;string&quot; &lt;|sep|&gt;</span>literal &lt;|close|&gt;call&lt;|sep|&gt; text<span class="tt-c1">&lt;|close|&gt; argument &lt;|sep|&gt;</span><span class="tt-c2">&lt;|close|&gt; call &lt;|sep|&gt;</span>', '<span class="tt-c0">&lt;|open|&gt;json type=&quot;object&quot;&lt;|sep|&gt;</span>{&quot;cmd&quot;:&quot;literal &lt;|close|&gt;json&lt;|sep|&gt; text&quot;}<span class="tt-c0">&lt;|close|&gt;json&lt;|sep|&gt;</span><span class="tt-c1">&lt;|open|&gt; message role=&quot;assistant&quot; &lt;|sep|&gt;</span>visible<span class="tt-c1">&lt;|close|&gt; message &lt;|sep|&gt;</span>', 'no family, plain <span class="tt-c0">&lt;tool_call&gt;</span>t<span class="tt-c0">&lt;/tool_call&gt;</span> &amp; &#x27;text&#x27;']
EXPECTED_STREAM = [['<span class="tt-c0">&lt;tool_</span>', '<span class="tt-c0">call&gt;</span>{&quot;n', '&quot;:1}<span class="tt-c0">&lt;/tool_call&gt;</span>'], ['<span class="tt-h tt-h-channel"><span class="tt-h-token">&lt;|chan</span></span>', '<span class="tt-h tt-h-channel"><span class="tt-h-token">nel|&gt;</span>fin</span>', '<span class="tt-h tt-h-channel">al</span><span class="tt-h tt-h-message"><span class="tt-h-token">&lt;|message|&gt;</span>hi</span>'], ['plain ', 'text &amp; ', 'more'], ['<span class="tt-c2">&lt;tool_call&gt;</span><span class="tt-c1">&lt;function=run&gt;</span><span class="tt-c0">&lt;parameter=cmd&gt;</span>git log ', '&lt;/tool_call&gt; --one', 'line<span class="tt-c0">&lt;/parameter&gt;</span><span class="tt-c1">&lt;/function&gt;</span><span class="tt-c2">&lt;/tool_call&gt;</span>'], ['<span class="tt-c1">&lt;|open|&gt; call tool=&quot;run&quot; index=&quot;1&quot; &lt;|</span>', '<span class="tt-c1">sep|&gt;</span><span class="tt-c0">&lt;|open|&gt; argument key=&quot;cmd&quot; type=&quot;string&quot; &lt;|sep|&gt;</span>', 'literal &lt;|close|&gt;call&lt;|sep|&gt; text<span class="tt-c0">&lt;|close|&gt; argument &lt;|sep|&gt;</span>', '<span class="tt-c1">&lt;|close|&gt; call &lt;|sep|&gt;</span>']]

def _node(fn_body: str, payload: dict) -> list:
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
        env={key: os.environ[key] for key in ("PATH", "LANG", "LC_ALL") if key in os.environ},
    )
    return json.loads(res.stdout)


def test_colorize_markup_parity():
    js = _node(
        "const out = inp.cases.map(c => mc.colorizeMarkup(c.text, c.family, inp.markers));",
        {"cases": [{"text": t, "family": f} for f, t in CASES], "markers": MARKERS},
    )
    for (family, text), got, expected in zip(CASES, js, EXPECTED_MARKUP):
        assert got == expected, f"family={family!r} text={text!r}"


def test_colorize_stream_deltas_parity():
    js = _node(
        "const out = inp.cases.map(c => mc.colorizeStreamDeltas(c.chunks, c.family, inp.markers));",
        {"cases": [{"chunks": ch, "family": f} for f, ch in STREAM_CASES], "markers": MARKERS},
    )
    for (family, chunks), got, expected in zip(STREAM_CASES, js, EXPECTED_STREAM):
        assert got == expected, f"family={family!r} chunks={chunks!r}"


def test_kimi_k3_linked_renderer_has_no_orphan_composite_markers():
    text = (
        '<|open|> message role="assistant" <|sep|>'
        '<|open|>think<|sep|>private<|close|>think<|sep|>'
        '<|open|> call tool="run" index="1" <|sep|>'
        '<|open|> argument key="cmd" type="string" <|sep|>'
        "literal <|close|>call<|sep|> text"
        '<|close|> argument <|sep|><|close|> call <|sep|>'
        '<|close|> message <|sep|>'
    )
    [rendered] = _node(
        "const ctx = mc.newLinkCtx();"
        "mc.colorizeLinked(inp.text, 'kimi_k3', inp.markers, ctx);"
        "mc.sealLinkCtx(ctx);"
        "const out = [mc.colorizeLinked(inp.text, 'kimi_k3', inp.markers, ctx)];",
        {"text": text, "markers": MARKERS},
    )
    assert "tt-mbg-orphan" not in rendered
    assert rendered.count('<span class="tt-mbg ') == 8


def test_output_string_whitespace_is_not_rendered_as_input_grammar_whitespace():
    [input_html, output_html] = _node(
        "const ctx = mc.newLinkCtx();"
        "const input = '<｜DSML｜invoke name=\"get_weather\"[{\"name\":\"get_weather\",\"arguments\":{\"city\":\"a > b\"}}]';"
        "mc.colorizeLinked(input, 'deepseek_v4', inp.markers, ctx);"
        "mc.sealLinkCtx(ctx);"
        "const out = [mc.colorizeLinked(input, 'deepseek_v4', inp.markers, ctx),"
        "mc.colorizeWords('{\"city\":\"a > b\"}', ctx)];",
        {"markers": MARKERS},
    )
    assert "tt-ws" in input_html
    assert "tt-ws" not in output_html
    assert "tt-fg" in output_html
