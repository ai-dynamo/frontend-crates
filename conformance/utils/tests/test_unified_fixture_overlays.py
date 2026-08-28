# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import sys

import yaml

UTILS_SRC = Path(__file__).resolve().parents[1] / "src"
if str(UTILS_SRC) not in sys.path:
    sys.path.insert(0, str(UTILS_SRC))

import generate_conformance_table as table  # noqa: E402


def _write_case(root, dirname, family, key, body, **metadata):
    path = root / dirname / family / f"{key}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {"family": family, "mode": "unified", **metadata, "cases": {key: body}},
            sort_keys=False,
        )
    )


def test_sparse_unified_patch_merges_with_base_and_overrides_in_order(tmp_path):
    family = "gemma4"
    base_key = "UNIFIED.1.a"
    patch_key = "UNIFIED.1.b"
    for key, scenario in (
        (base_key, "base_case"),
        (patch_key, "patch_case"),
    ):
        _write_case(
            tmp_path,
            "inputs",
            family,
            key,
            {"scenario": scenario, "chunks": [{"delta_text": scenario}]},
            model_label=family,
        )
        _write_case(
            tmp_path,
            "golden",
            family,
            key,
            {"assembled": [{"kind": "text", "text": scenario}]},
            captured_with={"golden": "v1"},
        )

    _write_case(
        tmp_path,
        "dynamo_v2-0.2.1",
        family,
        base_key,
        {"assembled": [{"kind": "text", "text": "base"}], "chunks": []},
        captured_with={"dynamo_v2": "0.2.1"},
    )
    _write_case(
        tmp_path,
        "dynamo_v2-0.2.1.patch1",
        family,
        base_key,
        {"assembled": [{"kind": "text", "text": "patched"}], "chunks": []},
        captured_with={"dynamo_v2": "0.2.1.patch1"},
    )
    _write_case(
        tmp_path,
        "dynamo_v2-0.2.1.patch1",
        family,
        patch_key,
        {"assembled": [{"kind": "text", "text": "patch-only"}], "chunks": []},
        captured_with={"dynamo_v2": "0.2.1.patch1"},
    )

    cases, _caps, versions = table._load_unified_fixtures(tmp_path)
    by_scenario = {case["scenario"]: case for case in cases}

    assert versions["dynamo_v2_all"] == ["0.2.1"]
    assert by_scenario["base_case"]["dynamo_by_ver"]["0.2.1"]["assembled"][0]["text"] == "patched"
    assert by_scenario["patch_case"]["dynamo_by_ver"]["0.2.1"]["assembled"][0]["text"] == "patch-only"
