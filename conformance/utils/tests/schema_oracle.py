# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent value validator for the schema keywords authored by this corpus."""

import re
from decimal import Decimal
from urllib.parse import unquote


def _json_equal(left: object, right: object) -> bool:
    """JSON numbers compare mathematically; booleans never compare as numbers."""
    if type(left) in (int, float) and type(right) in (int, float):
        return left == right
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _json_equal(value, right[key]) for key, value in left.items()
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _json_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def _local_reference(root_schema: dict, reference: str) -> dict:
    assert isinstance(reference, str) and reference.startswith("#"), reference
    fragment = unquote(reference[1:], errors="strict")
    assert not re.search(r"%(?![0-9a-fA-F]{2})", reference[1:]), reference
    assert not fragment or fragment.startswith("/"), reference
    target = root_schema
    for part in fragment.split("/")[1:]:
        assert not re.search(r"~(?![01])", part), reference
        key = part.replace("~1", "/").replace("~0", "~")
        if isinstance(target, dict):
            assert key in target, reference
            target = target[key]
        else:
            assert isinstance(target, list) and re.fullmatch(r"0|[1-9][0-9]*", key), reference
            index = int(key)
            assert index < len(target), reference
            target = target[index]
    assert isinstance(target, dict), reference
    return target


def matches_schema(
    value: object, schema: dict, root_schema: dict | None = None,
    _references: tuple[str, ...] = (),
    _inherited_nullable: bool = False,
) -> bool:
    root_schema = schema if root_schema is None else root_schema
    nullable = _inherited_nullable or schema.get("nullable") is True
    if "$id" in schema:
        # Corpus $id controls are scalar declarations; reference scopes remain unsupported.
        assert schema.keys() <= {"$id", "type", "const", "enum"}, schema
    assert schema.keys() <= {
        "$id", "type", "properties", "items", "anyOf", "oneOf", "const", "enum",
        "nullable", "minLength", "required", "$ref", "$defs", "definitions",
        "allOf", "minimum", "additionalProperties",
    }, schema
    if "$ref" in schema:
        reference = schema["$ref"]
        assert reference not in _references, ("cyclic schema reference", reference)
        target = _local_reference(root_schema, reference)
        if not matches_schema(
            value, target, root_schema, _references + (reference,), nullable
        ):
            return False
        siblings = {key: item for key, item in schema.items() if key != "$ref"}
        if not matches_schema(value, siblings, root_schema, _references, nullable):
            return False
    kind = schema.get("type")
    kinds = kind if isinstance(kind, list) else [kind] if kind else []
    if nullable and kinds:
        kinds = kinds + ["null"]
    types = {"string": isinstance(value, str), "null": value is None,
             "number": type(value) in (int, float, Decimal),
             "integer": type(value) is int or (type(value) is float and value.is_integer()) or (type(value) is Decimal and value == value.to_integral_value()),
             "boolean": type(value) is bool,
             "object": isinstance(value, dict), "array": isinstance(value, list)}
    # Reject unknown types even when another union member matches the value.
    assert all(isinstance(k, str) and k in types for k in kinds), schema
    if kinds and not any(types[k] for k in kinds):
        return False
    if "const" in schema and not _json_equal(value, schema["const"]):
        return False
    if "enum" in schema and not any(_json_equal(value, item) for item in schema["enum"]):
        return False
    if "minimum" in schema and (type(value) not in (int, float) or value < schema["minimum"]):
        return False
    if "anyOf" in schema and not any(
        matches_schema(value, branch, root_schema, _references, nullable)
        for branch in schema["anyOf"]
    ):
        return False
    if "oneOf" in schema and sum(
        matches_schema(value, branch, root_schema, _references, nullable)
        for branch in schema["oneOf"]
    ) != 1:
        return False
    if "allOf" in schema and not all(
        matches_schema(value, branch, root_schema, _references, nullable)
        for branch in schema["allOf"]
    ):
        return False
    if isinstance(value, str) and len(value) < schema.get("minLength", 0):
        return False
    if isinstance(value, dict):
        if not set(schema.get("required", [])) <= value.keys():
            return False
        properties = schema.get("properties", {})
        additional = schema.get("additionalProperties", True)
        assert isinstance(additional, (bool, dict)), schema
        for key, item in value.items():
            if key not in properties:
                if additional is False:
                    return False
                if isinstance(additional, dict) and not matches_schema(
                    item, additional, root_schema, _references
                ):
                    return False
        return all(
            matches_schema(item, properties[key], root_schema, _references)
            for key, item in value.items() if key in properties
        )
    if isinstance(value, list) and "items" in schema:
        return all(matches_schema(item, schema["items"], root_schema, _references) for item in value)
    return True
