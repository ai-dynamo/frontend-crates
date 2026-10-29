# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Schema variants retain independent oracles and observations when grouped."""

import sys
from pathlib import Path

import pytest

from conformance.utils.tests.schema_oracle import matches_schema

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from null_cases import NULL_DESCRIPTIONS, NULL_VARIANTS


@pytest.mark.parametrize("scenario,label,schema,value,description", NULL_VARIANTS)
def test_authored_oracle_satisfies_all_schema_constraints(scenario, label, schema, value, description):
    assert matches_schema(value, schema)
    if label.startswith("7-5"):
        assert not matches_schema(None, schema)
    assert "request tool schema" in description
