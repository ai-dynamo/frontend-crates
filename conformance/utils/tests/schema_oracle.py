# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent value validator for the schema keywords authored by this corpus."""

import re
from urllib.parse import unquote


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
) -> bool:
    root_schema = schema if root_schema is None else root_schema
    assert schema.keys() <= {"type", "properties", "items", "anyOf", "oneOf", "const",
                             "enum", "nullable", "minLength", "$ref", "$defs", "definitions"}, schema
    if "$ref" in schema:
        reference = schema["$ref"]
        assert reference not in _references, ("cyclic schema reference", reference)
        target = _local_reference(root_schema, reference)
        if not matches_schema(value, target, root_schema, _references + (reference,)):
            return False
    kind = schema.get("type")
    kinds = kind if isinstance(kind, list) else [kind] if kind else []
    if schema.get("nullable") is True:
        kinds = kinds + ["null"]
    types = {"string": isinstance(value, str), "null": value is None,
             "number": type(value) in (int, float),
             "integer": type(value) is int or (type(value) is float and value.is_integer()),
             "boolean": type(value) is bool,
             "object": isinstance(value, dict), "array": isinstance(value, list)}
    # Reject unknown types even when another union member matches the value.
    assert all(isinstance(k, str) and k in types for k in kinds), schema
    if kinds and not any(types[k] for k in kinds):
        return False
    if "const" in schema and value != schema["const"]:
        return False
    if "enum" in schema and value not in schema["enum"]:
        return False
    if "anyOf" in schema and not any(matches_schema(value, branch, root_schema, _references) for branch in schema["anyOf"]):
        return False
    if "oneOf" in schema and sum(matches_schema(value, branch, root_schema, _references) for branch in schema["oneOf"]) != 1:
        return False
    if isinstance(value, str) and len(value) < schema.get("minLength", 0):
        return False
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        return all(matches_schema(item, properties[key], root_schema, _references) for key, item in value.items() if key in properties)
    if isinstance(value, list) and "items" in schema:
        return all(matches_schema(item, schema["items"], root_schema, _references) for item in value)
    return True
