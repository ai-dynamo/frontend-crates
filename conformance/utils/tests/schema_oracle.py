# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent value validator for the schema keywords authored by this corpus."""


def matches_schema(value, schema):
    assert schema.keys() <= {"type", "properties", "items", "anyOf", "oneOf", "const",
                             "enum", "nullable", "minLength"}, schema
    kind = schema.get("type")
    kinds = kind if isinstance(kind, list) else [kind] if kind else []
    if schema.get("nullable") is True:
        kinds = kinds + ["null"]
    types = {"string": isinstance(value, str), "null": value is None,
             "number": type(value) in (int, float), "integer": type(value) is int,
             "object": isinstance(value, dict), "array": isinstance(value, list)}
    if kinds and not any(types[k] for k in kinds):
        return False
    if "const" in schema and value != schema["const"]:
        return False
    if "enum" in schema and value not in schema["enum"]:
        return False
    if "anyOf" in schema and not any(matches_schema(value, branch) for branch in schema["anyOf"]):
        return False
    if "oneOf" in schema and sum(matches_schema(value, branch) for branch in schema["oneOf"]) != 1:
        return False
    if isinstance(value, str) and len(value) < schema.get("minLength", 0):
        return False
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        return all(matches_schema(item, properties[key]) for key, item in value.items() if key in properties)
    if isinstance(value, list) and "items" in schema:
        return all(matches_schema(item, schema["items"]) for item in value)
    return True
