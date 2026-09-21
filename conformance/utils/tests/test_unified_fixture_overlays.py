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
import capture_stimulus
import validate_conformance_status as status  # noqa: E402


def _write_case(root, dirname, family, key, body, **metadata):
    if dirname == "inputs" or dirname.startswith("inputs+"):
        body = {"tools": [], **body}
    if dirname not in ("inputs", "golden") and not dirname.startswith(("inputs+", "golden+")):
        sources = sorted(root.glob(f"inputs*/{family}/{key}.yaml"))
        if sources:
            stimulus = yaml.safe_load(sources[-1].read_text())["cases"][key]
            body = {**body, "capture_input": capture_stimulus.capture_input(stimulus)}
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
    base_key = "UNIFIED.1-1"
    patch_key = "UNIFIED.1-2"
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
    base_key = "UNIFIED.1-1"
    pr_key = "UNIFIED.1-2"
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
    key = "UNIFIED.1-1"
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

    with pytest.raises(ValueError, match=rf"conflicting shared {kind} record gemma4/UNIFIED\.1-1"):
        table._load_unified_fixtures(tmp_path)


@pytest.mark.parametrize("dirname, body", [
    ("inputs", {"scenario": "same", "chunks": [{"delta_text": "same"}]}),
    ("golden", {"assembled": [{"kind": "text", "text": "same"}]}),
])
def test_shared_overlay_accepts_byte_identical_duplicate_records(tmp_path, dirname, body):
    family = "gemma4"
    key = "UNIFIED.1-1"
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
    key = "UNIFIED.1-1"
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
        "  UNIFIED.1-1: {chunks: [{delta_text: same}], scenario: same}\n"
    )

    with pytest.raises(ValueError, match=r"conflicting shared input record gemma4/UNIFIED\.1-1"):
        table._load_unified_fixtures(tmp_path)


def test_selected_qualified_capture_keeps_the_release(tmp_path, monkeypatch):
    monkeypatch.setattr(table, "_unified_dynamo_label", lambda captures: "0.3.4+pr166")
    family = "gemma4"
    key = "UNIFIED.1-1"
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


def test_selected_current_capture_requires_every_input_or_a_sparse_overlay(tmp_path, monkeypatch):
    monkeypatch.setattr(table, "_unified_dynamo_label", lambda captures: "0.3.4+pr166")
    family = "gemma4"
    captured_key = "UNIFIED.1-1"
    missing_key = "UNIFIED.1-2"
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

    cases, _caps, versions = table._load_unified_fixtures(tmp_path)

    missing = next(case for case in cases if case["scenario"] == "text_only")
    assert missing["dynamo_missing"] is True
    assert versions["dynamo_v2"] == "0.3.4+pr166"

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
    assert next(case for case in cases if case["scenario"] == "text_only")["dynamo_missing"] is False


@pytest.mark.parametrize("scenario", ["text_only", "tool_markup_only_emits_nothing"])
@pytest.mark.parametrize("current_present", [False, True])
def test_missing_current_capture_preserves_other_candidates(tmp_path, monkeypatch, scenario, current_present):
    monkeypatch.setattr(table, "_unified_dynamo_label", lambda captures: "0.6.0")
    family = "qwen3"
    generator = table.gen_unified_golden
    authored = generator.build_cases(family)[f"UNIFIED.{scenario}.{family}"]
    key = table.unified_taxonomy.numbered_id(scenario)
    golden = authored["golden"]
    divergent = [{"kind": "text", "text": "incorrect output"}]
    monkeypatch.setattr(generator, "CLEAN", [case for case in generator.CLEAN if case[0] == scenario])
    monkeypatch.setattr(generator, "EDGE", [case for case in generator.EDGE if case[0] == scenario])
    monkeypatch.setattr(table, "_unified_base", lambda root: tmp_path)
    _write_case(tmp_path, "inputs", family, key, {
        "scenario": scenario, "description": authored["description"],
        "input": authored["input"], "init": authored["init"],
        "chunks": [{"delta_text": authored["input"]}],
    })
    _write_case(tmp_path, "golden", family, key, {"assembled": golden})
    for dirname, events in (
        ("dynamo_v2-0.5.1", golden),
        ("dynamo_v2-0.4.0", divergent),
        ("vllm_python-0.26.0", golden),
        ("vllm_rust-0.26.0", divergent),
    ):
        _write_case(tmp_path, dirname, family, key, {
            "assembled": events, "chunks": [{"expected": events}],
        })
    _write_case(tmp_path, "vllm_python-0.25.1", family, key, {
        "assembled": divergent, "chunks": [{"expected": divergent}],
    })
    # Another case establishes each version without claiming a result for this one.
    for dirname in ("dynamo_v2-0.3.4", "vllm_rust-0.25.1", "dynamo_v2-0.6.0"):
        _write_case(tmp_path, dirname, family, "UNIFIED.1-1", {"assembled": [], "chunks": []})
    if current_present:
        _write_case(tmp_path, "dynamo_v2-0.6.0", family, key, {
            "assembled": golden, "chunks": [{"expected": golden}],
        })

    model = table._unified_tab_model(tmp_path, {})
    cell = model["rows"][0]["cells"][scenario]
    candidates = {candidate["key"]: candidate for candidate in model["candidates"]}
    popup = {candidate["key"]: candidate["block"] for candidate in cell["tooltip"]["candidates"]}
    assert set(cell["cmp"]) == set(popup) == set(candidates)
    for candidate, expected in {
        "dynamo": "green" if current_present else "red",
        "dynamo@0.5.1": "green", "dynamo@0.4.0": "red", "dynamo@0.3.4": "empty",
        "vllm_python@0.26.0": "green", "vllm_rust@0.26.0": "red",
        "vllm": "red", "vllm_rust": "empty",
    }.items():
        assert status.cell_state(cell, candidates[candidate])[0] == expected, candidate
    assert popup["dynamo@0.5.1"]["events"] == golden
    assert popup["vllm_python@0.26.0"]["events"] == golden
    assert popup["vllm"]["events"] == divergent
    assert "postdates" in popup["dynamo@0.3.4"]["unavailable"]
    assert "postdates" in popup["vllm_rust"]["unavailable"]
    chunks = cell["tooltip"]["input"]["chunks"]
    assert chunks[0]["expected"]["dynamo@0.5.1"] == golden
    assert chunks[0]["expected"]["vllm_rust@0.26.0"] == divergent
    if current_present:
        assert popup["dynamo"]["events"] == golden
    else:
        assert cell["status"] == "problem"
        assert "Missing Unified capture" in popup["dynamo"]["error"]
        assert cell["cmp"]["dynamo"]["sig"] != cell["cmp"]["golden"]["sig"]


@pytest.mark.parametrize("scenario, expected_state", [
    ("tool_only", "red"),
    ("gemma4_guided_json_visible_call_prose_before_reasoning", "na"),
])
def test_absent_input_distinguishes_missing_capture_from_inapplicable(tmp_path, monkeypatch, scenario, expected_state):
    generator = table.gen_unified_golden
    for family, selected in (("gemma4", scenario), ("qwen3", "text_only")):
        authored = generator.build_cases(family)[f"UNIFIED.{selected}.{family}"]
        key = table.unified_taxonomy.numbered_id(selected)
        _write_case(tmp_path, "inputs", family, key, {
            **authored, "scenario": selected, "chunks": [{"delta_text": authored["input"]}],
        })
        _write_case(tmp_path, "golden", family, key, {"assembled": authored["golden"]})
        for dirname in ("dynamo_v2-0.6.0", "dynamo_v2-0.5.1", "vllm_python-0.26.0"):
            _write_case(tmp_path, dirname, family, key, {"assembled": [], "chunks": []})
    monkeypatch.setattr(generator, "CLEAN", [case for case in generator.CLEAN if case[0] in (scenario, "text_only")])
    monkeypatch.setattr(generator, "EDGE", [case for case in generator.EDGE if case[0] == scenario])
    monkeypatch.setattr(table, "_unified_base", lambda root: tmp_path)

    model = table._unified_tab_model(tmp_path, {})
    row = next(row for row in model["rows"] if row["family"] == "qwen3")
    cell = row["cells"][scenario]
    candidates = {candidate["key"]: candidate for candidate in model["candidates"]}
    popup = {candidate["key"]: candidate["block"] for candidate in cell["tooltip"]["candidates"]}
    assert set(cell["cmp"]) == set(popup) == set(candidates)
    assert status.cell_state(cell, candidates["dynamo"])[0] == expected_state
    if expected_state == "red":
        assert "Missing Unified capture" in popup["dynamo"]["error"]
        assert cell["tooltip"]["input"]["kind"] == "text"
        assert cell["tooltip"]["input"]["text"]
        for key in ("dynamo@0.5.1", "vllm_python@0.26.0"):
            assert status.cell_state(cell, candidates[key])[0] == "empty"
    else:
        assert cell["status"] == "na"


def test_taxonomy_rename_uses_the_overlay_case_key(tmp_path):
    family = "gemma4"
    _write_case(
        tmp_path,
        "inputs",
        family,
        "UNIFIED.31-29",
        {"scenario": "gemma4_guided_json_visible_call_prose_before_reasoning", "chunks": []},
        model_label=family,
    )
    _write_case(
        tmp_path,
        "inputs+pr191.patch2",
        family,
        "UNIFIED.g4-1",
        {"scenario": "gemma4_guided_json_visible_call_prose_before_reasoning", "chunks": []},
        model_label=family,
    )
    _write_case(
        tmp_path,
        "golden",
        family,
        "UNIFIED.31-29",
        {"assembled": []},
        captured_with={"golden": "v1"},
    )
    _write_case(
        tmp_path,
        "golden+pr191.patch2",
        family,
        "UNIFIED.g4-1",
        {"assembled": []},
        captured_with={"golden": "v1"},
    )
    _write_case(
        tmp_path,
        "dynamo_v2-0.6.0",
        family,
        "UNIFIED.g4-1",
        {"assembled": [], "chunks": []},
        captured_with={"dynamo_v2": "0.6.0"},
    )

    cases, _caps, _versions = table._load_unified_fixtures(tmp_path)

    assert len(cases) == 1
    assert cases[0]["scenario"] == "gemma4_guided_json_visible_call_prose_before_reasoning"


def test_sparse_peer_patch_overrides_base_case_without_changing_release(tmp_path):
    family = "gemma4"
    base_key = "UNIFIED.1-1"
    patch_key = "UNIFIED.1-2"
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
