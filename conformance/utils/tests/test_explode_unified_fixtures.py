# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

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


def test_peer_error_retains_partial_chunks_as_separate_evidence() -> None:
    events = [{"kind": "tool_call", "name": "f", "arguments": '{"partial":'}]
    result = {"error": "ValueError: invalid chunk", "partial_chunks": [events]}
    assert explode._peer_cell(result) == {
        "error": "ValueError: invalid chunk", "partial_chunks": [{"expected": events}],
    }
    with pytest.raises(ValueError, match="requires a parser error"):
        explode._peer_cell({"chunks": [], "partial_chunks": [events]})


def test_peer_value_is_wrapped_once_and_preserves_empty_events() -> None:
    result = {
        "assembled": [],
        "chunks": [[], [{"kind": "text", "text": "answer"}]],
    }

    assert explode._peer_cell(result, expected_chunk_count=2) == {
        "assembled": [],
        "chunks": [
            {"expected": []},
            {"expected": [{"kind": "text", "text": "answer"}]},
        ],
    }


def test_peer_value_rejects_canonical_double_wrapping() -> None:
    result = {
        "assembled": [],
        "chunks": [[{"expected": []}]],
    }

    with pytest.raises(ValueError, match="double-wrapping"):
        explode._peer_cell(result, expected_chunk_count=1)


def test_peer_value_rejects_authored_chunk_count_mismatch() -> None:
    result = {"assembled": [], "chunks": [[]]}

    with pytest.raises(ValueError, match="chunk count differs"):
        explode._peer_cell(result, expected_chunk_count=2)


def test_vllm_peer_value_requires_assembled_events() -> None:
    with pytest.raises(ValueError, match="assembled must be a list"):
        explode._peer_cell(
            {"chunks": [[]]}, expected_chunk_count=1, require_assembled=True
        )


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
        "capture_observation": {
            "unavailable": {"code": "unsupported", "detail": "not captured"}
        },
    }
    assert explode._history_peer_cell(
        store,
        "vllm_python",
        "vllm_python-0.27.1",
        "gemma4",
        "restored_case",
        {**request, "input": "changed"},
    ) is None


def test_split_only_family_preserves_authored_request_and_golden() -> None:
    init = {"starting_state": "Reasoning", "tool_output_mode": "Native", "named_tool": None}
    tools = [{"name": "f", "parameters": {"type": "object"}}]
    golden = [{"kind": "reasoning", "text": "thinking"}, {"kind": "text", "text": "done"}]
    document = {
        "family": "gpt_oss",
        "cases": {
            "UNIFIED.prefilled_reasoning_then_text.gpt_oss": {
                "input": "authored Harmony request",
                "init": init,
                "finish_reason": "length",
                "tools": tools,
                "golden": golden,
            }
        },
    }

    rows = explode._split_only_cases("gpt_oss", document)

    row = rows["UNIFIED.40-prefilled_reasoning_then_text"]
    assert row["request"] == {
        "input": "authored Harmony request",
        "init": init,
        "finish_reason": "length",
        "tools": tools,
        "chunks": [{"delta_text": "authored Harmony request"}],
    }
    assert row["golden"] == {"assembled": golden}


def test_split_only_family_rejects_mismatched_family_scope() -> None:
    with pytest.raises(ValueError, match="belongs to qwen3"):
        explode._split_only_cases(
            "gpt_oss",
            {"family": "gpt_oss", "cases": {"UNIFIED.tool_only.qwen3": {}}},
        )


def test_split_only_family_preserves_an_empty_offered_tool_set() -> None:
    rows = explode._split_only_cases(
        "gpt_oss",
        {
            "family": "gpt_oss",
            "cases": {
                "UNIFIED.text_only.gpt_oss": {
                    "input": "authored text",
                    "init": {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None},
                    "finish_reason": "stop",
                    "tools": None,
                    "golden": [{"kind": "text", "text": "authored text"}],
                }
            },
        },
    )

    assert rows["UNIFIED.3-text_only"]["request"]["tools"] == []


def test_regeneration_removes_all_materialized_capture_directories(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(explode, "BUILD", tmp_path)
    (tmp_path / "inputs").mkdir()
    (tmp_path / "golden").mkdir()
    (tmp_path / "vllm_python-0.25.1").mkdir()

    explode._clear_generated_dirs()

    assert not (tmp_path / "inputs").exists()
    assert not (tmp_path / "golden").exists()
    assert not (tmp_path / "vllm_python-0.25.1").exists()
