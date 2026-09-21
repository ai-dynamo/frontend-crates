# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pytest
import yaml

UTILS_SRC = Path(__file__).resolve().parents[1] / "src"
if str(UTILS_SRC) not in sys.path:
    sys.path.insert(0, str(UTILS_SRC))

import capture_stimulus  # noqa: E402
import generate_conformance_table as table  # noqa: E402


def _write_case(root, directory, key, body):
    path = root / directory / "gemma4" / f"{key}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    if directory not in ("inputs", "golden"):
        input_document = yaml.safe_load((root / "inputs" / "gemma4" / f"{key}.yaml").read_text())
        body = {**body, "capture_input": capture_stimulus.capture_input(input_document["cases"][key])}
    path.write_text(
        yaml.safe_dump({"family": "gemma4", "mode": "unified", "cases": {key: body}}, sort_keys=False),
        encoding="utf-8",
    )


def _write_input_and_golden(root, key="UNIFIED.1-1"):
    _write_case(
        root,
        "inputs",
        key,
        {"scenario": "text_only", "chunks": [{"delta_text": "text"}], "tools": []},
    )
    _write_case(root, "golden", key, {"assembled": [{"kind": "text", "text": "text"}]})


def test_sparse_semantic_checkpoints_carry_a_family_forward_to_current_release(tmp_path, monkeypatch):
    monkeypatch.setattr(table, "_unified_dynamo_label", lambda _captures: "0.6.1")
    _write_input_and_golden(tmp_path)
    _write_case(
        tmp_path,
        "dynamo_v2-0.6.0",
        "UNIFIED.1-1",
        {"assembled": [{"kind": "text", "text": "text"}], "chunks": []},
    )

    cases, _caps, versions = table._load_unified_fixtures(tmp_path)

    assert versions["dynamo_v2"] == "0.6.1"
    assert versions["dynamo_v2_all"] == ["0.6.0", "0.6.1"]
    current = cases[0]["dynamo_by_ver"]["0.6.1"]
    assert current["assembled"] == [{"kind": "text", "text": "text"}]
    assert current["inherited_from"] == "0.6.0"


def test_explicit_current_error_is_not_replaced_by_an_older_success(tmp_path, monkeypatch):
    monkeypatch.setattr(table, "_unified_dynamo_label", lambda _captures: "0.6.1")
    _write_input_and_golden(tmp_path)
    _write_case(
        tmp_path,
        "dynamo_v2-0.6.0",
        "UNIFIED.1-1",
        {"assembled": [{"kind": "text", "text": "text"}], "chunks": []},
    )
    _write_case(tmp_path, "dynamo_v2-0.6.1", "UNIFIED.1-1", {"error": "capture failed"})

    cases, _caps, _versions = table._load_unified_fixtures(tmp_path)

    assert cases[0]["dynamo_failure"] == {"error": "capture failed"}
    assert cases[0]["dynamo_by_ver"]["0.6.1"]["inherited_from"] is None


@pytest.mark.parametrize("directory", ["dynamo_v2-0.6.1.patch1", "dynamo_v2-0.6.1+source." + "a" * 64])
def test_renderer_ignores_nonsemantic_capture_directories(tmp_path, monkeypatch, directory):
    monkeypatch.setattr(table, "_unified_dynamo_label", lambda _captures: "0.6.1")
    _write_input_and_golden(tmp_path)
    _write_case(tmp_path, directory, "UNIFIED.1-1", {"assembled": [], "chunks": []})

    cases, _caps, versions = table._load_unified_fixtures(tmp_path)

    assert versions["dynamo_v2_all"] == ["0.6.1"]
    assert cases[0]["dynamo_missing"] is True
