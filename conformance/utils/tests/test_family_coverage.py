# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Marker registration, leak recognition, and rendered token contracts."""
import os
import re
import sys
from pathlib import Path

import yaml

UTILS = Path(__file__).resolve().parents[1]
SRC = UTILS / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(UTILS) not in sys.path:
    sys.path.insert(0, str(UTILS))

import check_family_coverage as cfc  # noqa: E402
import markers as markers_mod  # noqa: E402
from conformance.utils.tests.fixture_test_support import colorize_markup  # noqa: E402


def _registry() -> dict:
    return yaml.safe_load((SRC / "parser_families.yaml").read_text())


def test_marker_lint_checks_registered_families() -> None:
    registry = _registry()
    report = cfc.Report()
    for family in registry["markers"]:
        cfc.check_markers(report, registry, family)
    assert report.failures == []
    cfc.check_markers(report, registry, "unregistered")
    assert any("no `markers:` entry" in failure for failure in report.failures)


def test_markers_declared_for_every_family() -> None:
    """Tool-only and unified parser families must declare their grammar tokens."""
    registry = _registry()
    unified = {row.get("registry", family) for family, row in registry["unified"].items()}
    assert set(registry["markers"]) == set(registry["families"]) | unified


def test_leak_regex_superset_of_retired_hardcoded_regex() -> None:
    """The YAML-derived leak detector matches everywhere the retired hardcoded
    regex did, on every normal_text in the committed corpus — token registration
    can only ADD detections, never silently lose them."""
    old = re.compile(
        r"</?tool_call|</?tool_calls|<\|tool_call|<\|tool_calls|"
        r"<\|(?:channel|message|call|python_tag)\|>|"
        r"</?TOOLCALL|TOOL_CALLS|<｜(?:DSML｜)?(?:tool|tool▁call|tool▁calls)|"
        r"<｜DSML｜|</?minimax:tool_call|</?invoke|</?arg_key|</?arg_value"
    )
    root = UTILS.parent / "fixtures-unified-v2/families"
    texts: set[str] = set()

    def walk(node) -> None:
        if isinstance(node, dict):
            nt = node.get("normal_text", node.get("text"))
            if isinstance(nt, str):
                texts.add(nt)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    for path in root.glob("*/inputs_and_golden.yaml"):
        walk(yaml.safe_load(path.read_text()))
    assert texts, "no authored Unified text outcomes found"
    lost = [t for t in texts if old.search(t) and not markers_mod._TOOL_CALL_MARKUP_RE.search(t)]
    assert lost == []


def test_declared_tokens_render_non_orphan() -> None:
    """Every declared pair/singleton renders without tt-orphan, including tokens the
    heuristic classifier cannot type (deepseek `<｜tool▁sep｜>` was a red orphan
    before the declared-marker lookup; Inkling's `<|message_model|>` class)."""
    registry = _registry()
    for fam, decl in registry["markers"].items():
        for open_tok, close_tok in decl.get("pairs") or []:
            assert "tt-orphan" not in colorize_markup(f"{open_tok}x{close_tok}", family=fam, markers=_registry()["markers"]), (fam, open_tok)
        for tok in decl.get("singletons") or []:
            assert "tt-orphan" not in colorize_markup(str(tok), family=fam, markers=_registry()["markers"]), (fam, tok)


def test_kimi_k3_composite_markers_render_as_longest_declared_tokens() -> None:
    cases = [
        "<|open|>think<|sep|>private<|close|>think<|sep|>",
        "<|open|> response <|sep|>visible<|close|> response <|sep|>",
        '<|open|>call tool="get_weather" index="1"<|sep|>'
        '<|open|>argument key="city" type="string"<|sep|>Paris'
        "<|close|>argument<|sep|><|close|>call<|sep|>",
        '<|open|> message role="assistant" <|sep|>visible'
        "<|close|> message <|sep|>",
    ]
    for text in cases:
        rendered = colorize_markup(text, family="kimi_k3", markers=_registry()["markers"])
        assert "tt-orphan" not in rendered, text
        expected_markers = text.count("<|open|>") + text.count("<|close|>")
        assert rendered.count("<span") == expected_markers


def test_kimi_k3_opaque_composite_regions_keep_marker_text_as_data() -> None:
    typed = (
        '<|open|> call tool="run" index="1" <|sep|>'
        '<|open|> argument key="cmd" type="string" <|sep|>'
        "literal <|close|>call<|sep|> text"
        "<|close|> argument <|sep|><|close|> call <|sep|>"
    )
    raw_json = (
        '<|open|>json type="object"<|sep|>'
        '{"cmd":"literal <|close|>json<|sep|> text"}'
        "<|close|>json<|sep|>"
    )
    for text in (typed, raw_json):
        rendered = colorize_markup(text, family="kimi_k3", markers=_registry()["markers"])
        assert "tt-orphan" not in rendered
        assert rendered.count("&lt;|close|&gt;") == text.count("<|close|>")


def test_undeclared_pipe_token_still_orphans() -> None:
    """The declared-marker lookup is per-family and exact: an Inkling-style token
    that is NOT declared keeps rendering as tt-orphan (the lint, not the colorizer,
    is what forces registration)."""
    assert "tt-orphan" in colorize_markup("<|message_model|>hello<|end_message|>", family="hermes", markers=_registry()["markers"])
