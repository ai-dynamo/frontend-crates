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
from schema_cases import CONFORMANCE_CASES, SCHEMA_GROUPS, schema_case_label, schema_tools

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
    assert len({schema_case_label(case, unified=True) for case in CONFORMANCE_CASES}) == len(CONFORMANCE_CASES)
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
