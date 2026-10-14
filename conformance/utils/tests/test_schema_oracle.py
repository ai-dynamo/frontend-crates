# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Guard independent schema expectations against Python's equality shortcuts."""

import pytest

from conformance.utils.tests.schema_oracle import matches_schema

pytestmark = []


@pytest.mark.parametrize("keyword", ["const", "enum"])
@pytest.mark.parametrize("value, expected", [
    (True, 1), (False, 0), (1, True), (0, False),
    ({"flag": True}, {"flag": 1}), ([False], [0]),
])
def test_constants_distinguish_booleans_from_numbers(
    keyword: str, value: object, expected: object,
) -> None:
    schema = {keyword: [expected] if keyword == "enum" else expected}
    assert not matches_schema(value, schema)
    assert matches_schema(expected, schema)


@pytest.mark.parametrize("value, expected", [(1, 1.0), ([1], [1.0]), ({"x": 1}, {"x": 1.0})])
def test_constants_use_json_numeric_equality(value: object, expected: object) -> None:
    assert matches_schema(value, {"const": expected})


def test_additional_properties_and_required_keys() -> None:
    schema = {
        "type": "object", "properties": {"value": {"type": "integer"}},
        "required": ["value"], "additionalProperties": False,
    }
    assert matches_schema({"value": 42}, schema)
    assert not matches_schema({}, schema)
    assert not matches_schema({"value": 42, "extra": "text"}, schema)
    assert not matches_schema({"value": True}, schema)


def test_typed_additional_properties() -> None:
    schema = {"type": "object", "additionalProperties": {"type": "integer"}}
    assert matches_schema({"x": 42}, schema)
    assert not matches_schema({"x": "42"}, schema)
    assert not matches_schema({"x": True}, schema)
