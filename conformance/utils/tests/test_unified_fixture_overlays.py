# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import sys

import pytest
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


def test_pr_overlay_merges_shared_inputs_and_golden_without_rewriting_releases(tmp_path):
    family = "gemma4"
    base_key = "UNIFIED.1.a"
    pr_key = "UNIFIED.1.b"
    for dirname, key, body, metadata in (
        ("inputs", base_key, {"scenario": "released", "chunks": [{"delta_text": "base"}]}, {"model_label": family}),
        ("golden", base_key, {"assembled": [{"kind": "text", "text": "base"}]}, {"captured_with": {"golden": "v1"}}),
        ("inputs+pr166.patch1", pr_key, {"scenario": "pr_only", "chunks": [{"delta_text": "overlay"}]}, {"model_label": family}),
        ("golden+pr166.patch1", pr_key, {"assembled": [{"kind": "text", "text": "overlay"}]}, {"captured_with": {"golden": "v1"}}),
    ):
        _write_case(tmp_path, dirname, family, key, body, **metadata)
    for key, text in ((base_key, "base"), (pr_key, "overlay")):
        _write_case(
            tmp_path,
            "dynamo_v2-0.3.4+pr166.patch1",
            family,
            key,
            {"assembled": [{"kind": "text", "text": text}], "chunks": []},
            captured_with={"dynamo_v2": "0.3.4+pr166"},
        )

    cases, _caps, _versions = table._load_unified_fixtures(tmp_path)

    assert {case["scenario"] for case in cases} == {"released", "pr_only"}
    assert next(case for case in cases if case["scenario"] == "pr_only")["golden"] == [{"kind": "text", "text": "overlay"}]


@pytest.mark.parametrize(
    ("dirname", "base_body", "overlay_body", "kind"),
    [
        (
            "inputs",
            {"scenario": "released", "chunks": [{"delta_text": "base"}]},
            {"scenario": "changed", "chunks": [{"delta_text": "base"}]},
            "input",
        ),
        (
            "golden",
            {"assembled": [{"kind": "text", "text": "base"}]},
            {"assembled": [{"kind": "text", "text": "changed"}]},
            "golden",
        ),
    ],
)
def test_shared_overlay_rejects_conflicting_duplicate_records(
    tmp_path, dirname, base_body, overlay_body, kind
):
    family = "gemma4"
    key = "UNIFIED.1.a"
    if dirname == "golden":
        _write_case(
            tmp_path,
            "inputs",
            family,
            key,
            {"scenario": "released", "chunks": [{"delta_text": "base"}]},
            model_label=family,
        )
    else:
        _write_case(
            tmp_path,
            "golden",
            family,
            key,
            {"assembled": [{"kind": "text", "text": "base"}]},
            captured_with={"golden": "v1"},
        )
    metadata = {"model_label": family} if dirname == "inputs" else {"captured_with": {"golden": "v1"}}
    _write_case(tmp_path, dirname, family, key, base_body, **metadata)
    _write_case(tmp_path, f"{dirname}+pr166.patch1", family, key, overlay_body, **metadata)

    with pytest.raises(ValueError, match=rf"conflicting shared {kind} record gemma4/UNIFIED\.1\.a"):
        table._load_unified_fixtures(tmp_path)


@pytest.mark.parametrize("dirname, body", [
    ("inputs", {"scenario": "same", "chunks": [{"delta_text": "same"}]}),
    ("golden", {"assembled": [{"kind": "text", "text": "same"}]}),
])
def test_shared_overlay_accepts_byte_identical_duplicate_records(tmp_path, dirname, body):
    family = "gemma4"
    key = "UNIFIED.1.a"
    if dirname == "golden":
        _write_case(
            tmp_path,
            "inputs",
            family,
            key,
            {"scenario": "same", "chunks": [{"delta_text": "same"}]},
            model_label=family,
        )
    else:
        _write_case(
            tmp_path,
            "golden",
            family,
            key,
            {"assembled": [{"kind": "text", "text": "same"}]},
            captured_with={"golden": "v1"},
        )
    metadata = {"model_label": family} if dirname == "inputs" else {"captured_with": {"golden": "v1"}}
    _write_case(tmp_path, dirname, family, key, body, **metadata)
    _write_case(tmp_path, f"{dirname}+pr166.patch1", family, key, body, **metadata)
    _write_case(
        tmp_path,
        "dynamo_v2-0.3.4+pr166",
        family,
        key,
        {"assembled": [{"kind": "text", "text": "same"}], "chunks": []},
        captured_with={"dynamo_v2": "0.3.4+pr166"},
    )

    cases, _caps, _versions = table._load_unified_fixtures(tmp_path)

    assert len(cases) == 1


def test_shared_overlay_rejects_semantically_equal_but_byte_different_record(tmp_path):
    family = "gemma4"
    key = "UNIFIED.1.a"
    _write_case(
        tmp_path,
        "inputs",
        family,
        key,
        {"scenario": "same", "chunks": [{"delta_text": "same"}]},
        model_label=family,
    )
    overlay = tmp_path / "inputs+pr166.patch1" / family / f"{key}.yaml"
    overlay.parent.mkdir(parents=True)
    overlay.write_text(
        "family: gemma4\nmode: unified\nmodel_label: gemma4\ncases:\n"
        "  UNIFIED.1.a: {chunks: [{delta_text: same}], scenario: same}\n"
    )

    with pytest.raises(ValueError, match=r"conflicting shared input record gemma4/UNIFIED\.1\.a"):
        table._load_unified_fixtures(tmp_path)


def test_unified_qualified_capture_is_current_and_keeps_the_release(tmp_path):
    family = "gemma4"
    key = "UNIFIED.1.a"
    _write_case(
        tmp_path,
        "inputs",
        family,
        key,
        {"scenario": "capture_history", "chunks": [{"delta_text": "x"}]},
        model_label=family,
    )
    _write_case(
        tmp_path,
        "golden",
        family,
        key,
        {"assembled": [{"kind": "text", "text": "x"}]},
        captured_with={"golden": "v1"},
    )
    for version, text in (("0.3.4", "release"), ("0.3.4+pr166", "branch")):
        _write_case(
            tmp_path,
            f"dynamo_v2-{version}",
            family,
            key,
            {"assembled": [{"kind": "text", "text": text}], "chunks": []},
            captured_with={"dynamo_v2": version},
        )

    cases, _caps, versions = table._load_unified_fixtures(tmp_path)

    assert versions["dynamo_v2"] == "0.3.4+pr166"
    assert versions["dynamo_v2_all"] == ["0.3.4", "0.3.4+pr166"]
    assert cases[0]["dynamo_by_ver"]["0.3.4"]["assembled"][0]["text"] == "release"
    assert cases[0]["dynamo_by_ver"]["0.3.4+pr166"]["assembled"][0]["text"] == "branch"


def test_selected_current_capture_requires_every_input_or_a_sparse_overlay(tmp_path):
    family = "gemma4"
    captured_key = "UNIFIED.1.a"
    missing_key = "UNIFIED.1.b"
    for key in (captured_key, missing_key):
        _write_case(
            tmp_path,
            "inputs",
            family,
            key,
            {"scenario": "tool_only" if key == captured_key else "text_only", "chunks": [{"delta_text": key}]},
            model_label=family,
        )
        _write_case(
            tmp_path,
            "golden",
            family,
            key,
            {"assembled": [{"kind": "text", "text": key}]},
            captured_with={"golden": "v1"},
        )
    _write_case(
        tmp_path,
        "dynamo_v2-0.3.4+pr166",
        family,
        captured_key,
        {"assembled": [{"kind": "text", "text": "branch"}], "chunks": []},
        captured_with={"dynamo_v2": "0.3.4+pr166"},
    )

    with pytest.raises(ValueError, match=r"0\.3\.4\+pr166 lacks input case\(s\): gemma4/UNIFIED\.1\.b"):
        table._load_unified_fixtures(tmp_path)

    _write_case(
        tmp_path,
        "dynamo_v2-0.3.4+pr166.patch1",
        family,
        missing_key,
        {"assembled": [{"kind": "text", "text": "overlay"}], "chunks": []},
        captured_with={"dynamo_v2": "0.3.4+pr166.patch1"},
    )

    cases, _caps, versions = table._load_unified_fixtures(tmp_path)

    assert versions["dynamo_v2"] == "0.3.4+pr166"
    assert {case["scenario"] for case in cases} == {"tool_only", "text_only"}


def test_sparse_peer_patch_overrides_base_case_without_changing_release(tmp_path):
    family = "gemma4"
    base_key = "UNIFIED.1.a"
    patch_key = "UNIFIED.1.b"
    for key, scenario in ((base_key, "base_case"), (patch_key, "patch_case")):
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
        "vllm_python-0.25.1",
        family,
        base_key,
        {"assembled": [{"kind": "text", "text": "base"}], "chunks": []},
        captured_with={"vllm_python": "0.25.1"},
    )
    _write_case(
        tmp_path,
        "vllm_python-0.25.1.patch1",
        family,
        base_key,
        {"assembled": [{"kind": "text", "text": "patched"}], "chunks": []},
        captured_with={"vllm_python": "0.25.1.patch1"},
    )
    _write_case(
        tmp_path,
        "vllm_python-0.25.1.patch1",
        family,
        patch_key,
        {"assembled": [{"kind": "text", "text": "patch-only"}], "chunks": []},
        captured_with={"vllm_python": "0.25.1.patch1"},
    )

    cases, caps, versions = table._load_unified_fixtures(tmp_path)
    by_scenario = {case["scenario"]: case for case in cases}

    assert versions["vllm_python"] == "0.25.1"
    assert versions["vllm_python_all"] == ["0.25.1"]
    assert caps["vllm_python"]["UNIFIED.base_case.gemma4"]["assembled"][0]["text"] == "patched"
    assert by_scenario["base_case"]["peer_by_ver"]["vllm_python"]["0.25.1"]["assembled"][0]["text"] == "patched"
    assert by_scenario["patch_case"]["peer_by_ver"]["vllm_python"]["0.25.1"]["assembled"][0]["text"] == "patch-only"
