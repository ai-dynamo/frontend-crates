# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent schema and applicability checks for the shared regression matrix."""

import json
import sys
from pathlib import Path

import pytest

from conformance.utils.tests.schema_oracle import matches_schema

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import gen_schema_fixtures as legacy
import gen_unified_golden as unified
import unified_taxonomy
from unified_taxonomy import numbered_id
from schema_cases import CONFORMANCE_CASES, SCHEMA_GROUPS, XML_COMPATIBILITY_CASES, XML_COMPATIBILITY_FAMILIES, schema_case_label, schema_tools

pytestmark = []


@pytest.mark.parametrize("family", sorted(legacy.REGISTRY))
def test_every_registered_family_has_independent_native_schema_inputs(family: str) -> None:
    cases = legacy.authored_legacy_cases(family, False)
    assert len(cases) == len(CONFORMANCE_CASES)
    assert len(set(cases)) == len(cases)
    for case in cases.values():
        assert case["model_text"]
        calls = case["golden"]["calls"]
        assert len(calls) == 1 and calls[0]["name"] == "schema_probe"
        assert matches_schema(calls[0]["arguments"], case["tools"][0]["parameters"])


@pytest.mark.parametrize("family", unified.FAMILIES)
def test_every_unified_family_covers_every_schema_group(family: str) -> None:
    cases = unified.build_cases(family)
    for probe in CONFORMANCE_CASES:
        case = cases[f"UNIFIED.schema_{probe['name']}.{family}"]
        assert case["tools"] == schema_tools(probe)
        assert matches_schema(case["golden"][0]["arguments"], case["tools"][0]["parameters"])
        assert case["init"]["tool_output_mode"] == "Native"
    assert len(SCHEMA_GROUPS) == 9


def test_schema_variants_have_unique_stable_identities() -> None:
    assert len({schema_case_label(case) for case in CONFORMANCE_CASES}) == len(CONFORMANCE_CASES)
    assert len({numbered_id("schema_" + case["name"]) for case in CONFORMANCE_CASES}) == len(CONFORMANCE_CASES)
    identities = [json.dumps({key: case.get(key) for key in ("schema", "defs", "value", "raw")}, sort_keys=True)
                  for case in CONFORMANCE_CASES]
    assert len(set(identities)) == len(identities)


def test_streaming_schema_spelling_is_lossless() -> None:
    for family in legacy.REGISTRY:
        batch, stream = legacy.authored_legacy_cases(family, False), legacy.authored_legacy_cases(family, True)
        for case_id, case in stream.items():
            batch_id = case_id.replace("TOOLCALLING.streamv1.", "TOOLCALLING.batch.")
            assert "".join(chunk["delta_text"] for chunk in case["chunks"]) == batch[batch_id]["model_text"]
            assert case["golden"] == batch[batch_id]["golden"]


@pytest.mark.parametrize("probe", XML_COMPATIBILITY_CASES, ids=lambda probe: probe["name"])
def test_xml_quote_compatibility_has_the_same_oracle_in_both_corpora(probe) -> None:
    raw = "{'a':1}" if "single_quoted" in probe["name"] else '{"a":1}'
    type_name = probe["schema"]["type"]
    expected = {"value": {"a": 1} if type_name == "custom" else raw}
    for family in XML_COMPATIBILITY_FAMILIES:
        case = legacy.authored_legacy_cases(family, True, [probe])[f"TOOLCALLING.streamv1.{schema_case_label(probe)}"]
        assert raw in "".join(chunk["delta_text"] for chunk in case["chunks"])
        assert case["golden"] == {"calls": [{"name": "schema_probe", "arguments": expected}], "normal_text": ""}
        assert case["tools"] == schema_tools(probe)
    applicable = set(unified.FAMILIES) if type_name == "string" else {"qwen3", "nemotron_v3"}
    assert unified.scenario_families(probe["name"]) == applicable
    for family in unified.FAMILIES:
        case_id = f"UNIFIED.{probe['name']}.{family}"
        cases = unified.build_cases(family)
        if family not in applicable:
            assert case_id not in cases
            continue
        case = cases[case_id]
        assert raw in case["input"] or json.dumps(raw, ensure_ascii=False) in case["input"]
        assert case["tools"] == schema_tools(probe)
        assert case["golden"] == [{"kind": "tool_call", "name": "schema_probe", "arguments": expected}]
        assert numbered_id(probe["name"]).startswith("UNIFIED.7-")


@pytest.mark.parametrize("family", sorted(legacy.REGISTRY))
@pytest.mark.parametrize("quotes,raw", [("single", "{'a':1}"), ("double", '{"a":1}')])
def test_string_quote_controls_preserve_payload_in_every_native_grammar(family, quotes, raw):
    probe = next(probe for probe in XML_COMPATIBILITY_CASES
                 if probe["name"] == f"xml_string_type_{quotes}_quoted_object")
    case_id = schema_case_label(probe)
    batch = legacy.authored_legacy_cases(family, False, [probe])[f"TOOLCALLING.batch.{case_id}"]
    stream = legacy.authored_legacy_cases(family, True, [probe])[f"TOOLCALLING.streamv1.{case_id}"]
    wire = "".join(chunk["delta_text"] for chunk in stream["chunks"])
    assert wire == batch["model_text"]
    # JSON/Python grammars escape the containing string; raw XML carries it directly.
    assert any(spelling in wire for spelling in (raw, json.dumps(raw), repr(raw)))
    assert batch["golden"] == stream["golden"] == {
        "calls": [{"name": "schema_probe", "arguments": {"value": raw}}], "normal_text": ""}
    assert matches_schema({"value": raw}, stream["tools"][0]["parameters"])
    assert stream["chunks"][-1]["finish_reason"] == "stop"


def test_staged_quote_generator_reads_history_from_the_supplied_checkout(monkeypatch, tmp_path):
    expected = unified.build_cases("qwen3")
    monkeypatch.setenv("FRONTEND_CRATES_ROOT", str(SRC.parents[2]))
    monkeypatch.setattr(unified, "__file__", str(tmp_path / "stage/tests/parity/gen_unified_golden.py"))
    assert unified.build_cases("qwen3") == expected


@pytest.mark.parametrize("quotes", ["single", "double"])
def test_custom_quote_display_is_family_specific_without_changing_identity(quotes):
    custom = f"xml_custom_type_{quotes}_quoted_object"
    string = f"xml_string_type_{quotes}_quoted_object"
    assert unified_taxonomy.display_tax(custom)[0] == "qwen3"
    assert unified_taxonomy.display_section(custom)[0] == "unified_gqwen3"
    assert unified_taxonomy.numbered_id(custom) == f"UNIFIED.7-{custom}"
    assert unified_taxonomy.display_tax(string)[0] == 7
    assert unified.scenario_families(custom) == {"qwen3", "nemotron_v3"}
    assert unified.scenario_families(string) == set(unified.FAMILIES)
