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


def test_sparse_semantic_checkpoints_use_the_latest_actual_capture(tmp_path):
    _write_input_and_golden(tmp_path)
    _write_case(
        tmp_path,
        "dynamo_v2-0.6.0",
        "UNIFIED.1-1",
        {"assembled": [{"kind": "text", "text": "text"}], "chunks": []},
    )

    cases, _caps, versions = table._load_unified_fixtures(tmp_path)

    assert versions["dynamo_v2"] == {"gemma4": "0.6.0"}
    assert versions["dynamo_v2_all"] == ["0.6.0"]
    assert cases[0]["dynamo_missing"] is False
    assert cases[0]["dynamo"] == [{"kind": "text", "text": "text"}]


def test_new_family_without_a_capture_is_missing(tmp_path):
    _write_input_and_golden(tmp_path)
    family = tmp_path / "inputs" / "gemma4" / "UNIFIED.1-1.yaml"
    document = yaml.safe_load(family.read_text())
    document["family"] = "future_family"
    for directory in ("inputs", "golden"):
        source = tmp_path / directory / "gemma4" / "UNIFIED.1-1.yaml"
        target = tmp_path / directory / "future_family" / source.name
        target.parent.mkdir(parents=True)
        content = yaml.safe_load(source.read_text())
        content["family"] = "future_family"
        target.write_text(yaml.safe_dump(content), encoding="utf-8")
        source.unlink()

    cases, _caps, _versions = table._load_unified_fixtures(tmp_path)

    assert cases[0]["family"] == "future_family"
    assert cases[0]["dynamo_missing"] is True


def test_explicit_current_error_is_not_replaced_by_an_older_success(tmp_path, monkeypatch):
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
    _write_input_and_golden(tmp_path)
    _write_case(tmp_path, directory, "UNIFIED.1-1", {"assembled": [], "chunks": []})

    cases, _caps, versions = table._load_unified_fixtures(tmp_path)

    assert versions["dynamo_v2_all"] == []
    assert cases[0]["dynamo_missing"] is True


def test_latest_capture_is_selected_independently_for_each_family(tmp_path):
    key = "UNIFIED.1-1"
    for family in ("family_a", "family_b"):
        for directory, body in (
            ("inputs", {"scenario": "text_only", "chunks": [{"delta_text": family}], "tools": []}),
            ("golden", {"assembled": [{"kind": "text", "text": family}]}),
        ):
            path = tmp_path / directory / family / f"{key}.yaml"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump({"family": family, "cases": {key: body}}))
        for version in (["1.0.0", "2.0.0-rc.2", "2.0.0-rc.10"] if family == "family_a" else ["1.0.0", "3.0.0"]):
            path = tmp_path / f"dynamo_v2-{version}" / family / f"{key}.yaml"
            path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "capture_input": capture_stimulus.capture_input(
                    {"input": family, "chunks": [{"delta_text": family}], "tools": []}
                ),
                "assembled": [{"kind": "text", "text": version}],
                "chunks": [],
            }
            path.write_text(yaml.safe_dump({"family": family, "cases": {key: record}}))

    cases, _caps, versions = table._load_unified_fixtures(tmp_path)

    assert versions["dynamo_v2"] == {"family_a": "2.0.0-rc.10", "family_b": "3.0.0"}
    assert {case["family"]: case["dynamo"][0]["text"] for case in cases} == {
        "family_a": "2.0.0-rc.10",
        "family_b": "3.0.0",
    }
