# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import explode_unified_fixtures as explode  # noqa: E402


def test_peer_error_wins_over_partial_output() -> None:
    result = {
        "error": "UnifiedParserError::ParsingFailed",
        "assembled": [{"kind": "reasoning", "text": "partial"}],
        "chunks": [[{"kind": "reasoning", "text": "partial"}]],
    }

    assert explode._peer_cell(result) == {"error": "UnifiedParserError::ParsingFailed"}


def test_regeneration_removes_all_materialized_capture_directories(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(explode, "BUILD", tmp_path)
    (tmp_path / "inputs").mkdir()
    (tmp_path / "golden").mkdir()
    (tmp_path / "vllm_python-0.25.1").mkdir()

    explode._clear_generated_dirs()

    assert not (tmp_path / "inputs").exists()
    assert not (tmp_path / "golden").exists()
    assert not (tmp_path / "vllm_python-0.25.1").exists()
