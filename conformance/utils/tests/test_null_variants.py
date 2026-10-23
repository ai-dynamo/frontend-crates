# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Schema variants retain independent oracles and observations when grouped."""

import sys
from pathlib import Path

import pytest

from conformance.utils.tests.schema_oracle import matches_schema

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gen_null_stream_cases import FAMILIES, build_cases
from null_cases import NULL_DESCRIPTIONS, NULL_VARIANTS


@pytest.mark.parametrize("scenario,label,schema,value,description", NULL_VARIANTS)
def test_authored_oracle_satisfies_all_schema_constraints(scenario, label, schema, value, description):
    assert matches_schema(value, schema)
    if label.startswith("7-5"):
        assert not matches_schema(None, schema)
    assert "request tool schema" in description


@pytest.mark.parametrize("family", FAMILIES)
def test_stream_variants_have_distinct_ids_and_independent_goldens(family):
    cases = build_cases(family)
    assert len(cases) == len(NULL_VARIANTS) + int(family in {"glm47", "minimax_m3"})
    stimuli = []
    for case in cases.values():
        tool = case["tools"][0]
        call = case["golden"]["calls"][0]
        assert matches_schema(call["arguments"], tool["parameters"])
        assert call["name"] == tool["name"]
        assert case["chunks"][-1] == {"delta_text": "", "finish_reason": "stop"}
        stimuli.append((str(case["tools"]), str(case["chunks"])))
    assert len(set(stimuli)) == len(stimuli)


def test_mixed_reproducers_keep_both_field_types_and_strict():
    glm = build_cases("glm47")["TOOLCALLING.streamv1.7-4.mixed_labels"]
    assert glm["golden"]["calls"] == [{"name": "set_labels", "arguments": {
        "label": None, "note": None, "literal": "null"}}]
    m3 = build_cases("minimax_m3")["TOOLCALLING.streamv1.7-4.mixed_grep"]
    assert m3["tools"][0]["strict"] is True
    assert m3["golden"]["calls"] == [{"name": "grep", "arguments": {"pattern": "null", "path": None}}]
