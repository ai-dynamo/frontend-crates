# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import smg  # noqa: E402


def test_capture_accessors(monkeypatch, tmp_path: Path) -> None:
    doc = {
        "schema": "smg-conformance/v1",
        "tool_parser_version": "1.6.0",
        "reasoning_parser_version": "1.6.0",
        "toolcalling": {"batch": {"qwen25:case": {"calls": []}}},
        "reasoning": {"stream": {"qwen3:case": {"reasoning_text": "r"}}},
        "unified": {"qwen3:scenario": {"assembled": []}},
        "coverage": {"tool_batch_inputs": 1},
    }
    path = tmp_path / "capture.json"
    path.write_text(json.dumps(doc))
    monkeypatch.setenv("SMG_CAPTURE_PATH", str(path))
    smg.capture.cache_clear()

    assert smg.tool_version() == "1.6.0"
    assert smg.reasoning_version() == "1.6.0"
    assert smg.tool("batch", "qwen25", "case") == {"calls": []}
    assert smg.reasoning("stream", "qwen3", "case") == {"reasoning_text": "r"}
    assert smg.unified("qwen3", "scenario") == {"assembled": []}
    assert smg.coverage() == {"tool_batch_inputs": 1}

    smg.capture.cache_clear()


def test_capture_is_optional(monkeypatch) -> None:
    monkeypatch.delenv("SMG_CAPTURE_PATH", raising=False)
    smg.capture.cache_clear()
    assert smg.capture() is None
    assert smg.tool("batch", "qwen25", "case") is None
