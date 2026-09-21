# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

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


def test_missing_scratch_peer_record_reuses_matching_history() -> None:
    request = {
        "init": {"starting_state": "Response"},
        "finish_reason": "stop",
        "input": "payload",
        "tools": [],
        "chunks": [],
    }
    case = {"scenario": "restored_case", "request": request}
    change = {
        "case_key": "UNIFIED.31-39",
        "stimulus": {"ref": "current"},
        "observation": {
            "unavailable": {"code": "unsupported", "detail": "not captured"}
        },
        "document": {"captured_with": {"vllm_python": "0.27.1"}},
    }
    history = SimpleNamespace(
        family=SimpleNamespace(cases={"retired__31_39": case}),
        captures={"vllm_python-0.27.1": {}},
        resolve=lambda _capture_id: {"retired__31_39": change},
    )
    store = SimpleNamespace(histories={("gemma4", "vllm_python"): history})

    assert explode._history_peer_cell(
        store,
        "vllm_python",
        "vllm_python-0.27.1",
        "gemma4",
        "restored_case",
        request,
    ) == {
        "unavailable": "not captured",
        "capture_input": request,
    }
    assert explode._history_peer_cell(
        store,
        "vllm_python",
        "vllm_python-0.27.1",
        "gemma4",
        "restored_case",
        {**request, "input": "changed"},
    ) is None


def test_regeneration_removes_all_materialized_capture_directories(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(explode, "BUILD", tmp_path)
    (tmp_path / "inputs").mkdir()
    (tmp_path / "golden").mkdir()
    (tmp_path / "vllm_python-0.25.1").mkdir()

    explode._clear_generated_dirs()

    assert not (tmp_path / "inputs").exists()
    assert not (tmp_path / "golden").exists()
    assert not (tmp_path / "vllm_python-0.25.1").exists()
