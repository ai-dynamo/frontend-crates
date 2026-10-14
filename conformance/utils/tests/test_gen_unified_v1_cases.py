# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pytest
import yaml

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import gen_unified_golden
import gen_unified_v1_cases
import unified_taxonomy


ROOT = Path(__file__).resolve().parents[3]


def test_v1_split_inputs_and_captures_cover_every_family_with_a_v1_capture_path():
    documents = gen_unified_v1_cases.generated_documents(ROOT)
    expected_families = set(gen_unified_golden.V1_SPLIT_FAMILIES)
    registered_families = set(gen_unified_golden.yaml.safe_load(
        gen_unified_golden.markers.parser_families_path().read_text()
    )["families"])
    input_documents = {}
    capture_documents = {}
    for path, content in documents.items():
        family = path.parent.name
        document = yaml.safe_load(content)
        if path.name == "inputs_and_golden.yaml":
            input_documents[family] = document
        else:
            capture_documents[family] = document

    assert set(input_documents) == expected_families
    assert set(capture_documents) == expected_families
    assert expected_families == registered_families - {"muse_glimmer"} | {"gpt_oss"}
    assert len(expected_families) == 24
    for family in sorted(expected_families):
        input_case = input_documents[family]["cases"]["v1_split_parallel_tool_calls"]
        capture = capture_documents[family]
        assert input_case["scenario"] == "v1_split_parallel_tool_calls"
        assert input_case["display_id"] == unified_taxonomy.numbered_id(input_case["scenario"])
        assert len(input_case["golden"]["assembled"]) == 2
        assert capture["provenance"]["status"] == "legacy"
        assert capture["document"]["mode"] == "unified"
        assert capture["changes"]["v1_split_parallel_tool_calls"]["case_key"] == input_case["display_id"]
        assert capture["changes"]["v1_split_parallel_tool_calls"]["observation"]["value"] == input_case["golden"]
        assert (ROOT / "conformance/fixtures-unified-v2/families" / family / "inputs_and_golden.yaml").read_text() == documents[ROOT / "conformance/fixtures-unified-v2/families" / family / "inputs_and_golden.yaml"]
        capture_path = ROOT / "conformance/fixtures-unified-v2/families" / family / f"dynamo_v1-{capture['provenance']['captured_with']['dynamo_v1']}.yaml"
        assert capture_path.read_text() == documents[capture_path]


def test_incomplete_archived_input_reports_its_selected_legacy_case(monkeypatch):
    monkeypatch.setattr(gen_unified_v1_cases, "_capture_archives", lambda _root: [((1,), "1.0.0", Path("archive.tar.gz"))])
    monkeypatch.setattr(gen_unified_v1_cases, "_captured_call_map", lambda _archives, _families: {("sample", "1.0.0", "UNIFIED.1"): {}})
    monkeypatch.setattr(gen_unified_v1_cases, "_input_cases", lambda _root, _family: {"UNIFIED.1": {"model_text": None, "tools": []}})
    monkeypatch.setattr(gen_unified_v1_cases.gen_unified_golden, "V1_SPLIT_FAMILIES", frozenset({"sample"}))
    monkeypatch.setattr(gen_unified_v1_cases, "FAMILY_ALIASES", {})

    with pytest.raises(ValueError, match=r"sample/UNIFIED\.1 has an incomplete archived input"):
        gen_unified_v1_cases.generated_documents(Path("fixture-root"))
