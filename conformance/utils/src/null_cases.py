# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Request schemas and oracle types shared by Unified and legacy stream cases."""

NULL_DESCRIPTIONS = {
    "7-4": 'The request tool schema permits JSON null. Bare parameter text `null` must produce JSON `null`; grammars with explicit types use native null syntax. Variants cover nullable type arrays, anyOf, oneOf, nullable, and const. Mixed-field probes also assert that non-nullable fields remain strings. Refs #251, #268, #269.',
    "7-5": 'The request tool schema requires a string for the tested value. Bare parameter text `null` must remain JSON string `"null"`; grammars with explicit types use native string syntax. Variants cover non-nullable unions and intersecting sibling constraints. Mixed-field probes also assert that nullable fields become null. Refs #251, #268, #269.',
}

# Each schema gets its own immutable fixture identity, even when its wire text is
# identical to another variant. Display grouping must never merge captured inputs.
NULL_VARIANTS = (
    ("arg_json_null", "7-4", {"type": ["string", "null"]}, None,
     'The request tool schema declares `city` as `string | null` using a type array.'),
    ("arg_json_null_anyof", "7-4.anyof", {"anyOf": [{"type": "string"}, {"type": "null"}]}, None,
     'The request tool schema declares `city` as `string | null` using anyOf.'),
    ("arg_json_null_oneof", "7-4.oneof", {"oneOf": [{"type": "string"}, {"type": "null"}]}, None,
     'The request tool schema declares `city` as `string | null` using oneOf.'),
    ("arg_json_null_nullable", "7-4.nullable", {"type": "string", "nullable": True}, None,
     'The request tool schema declares `city` as string with nullable: true.'),
    ("arg_json_null_const", "7-4.const", {"anyOf": [{"const": None}, {"type": "string"}]}, None,
     'The request tool schema permits `city` to be null through an anyOf const-null branch.'),
    ("arg_string_null", "7-5", {"type": "string"}, "null",
     'The request tool schema declares `city` as non-nullable `string`.'),
    ("arg_string_null_union", "7-5.union", {"anyOf": [{"type": "string"}, {"type": "integer"}]}, "null",
     'The request tool schema permits string or integer through anyOf, but excludes null.'),
    ("arg_string_null_sibling_anyof", "7-5.sibling_anyof", {"type": "string", "anyOf": [{"type": "string"}, {"type": "null"}]}, "null",
     'The request tool schema has type string alongside a nullable anyOf; both constraints apply, so null is excluded.'),
    ("arg_string_null_sibling_oneof", "7-5.sibling_oneof", {"type": ["string", "null"], "oneOf": [{"type": "string"}, {"type": "integer"}]}, "null",
     'The request tool schema has a nullable type array alongside a non-nullable oneOf; their intersection permits string.'),
    ("arg_string_null_untyped_branch", "7-5.untyped_branch", {"type": "string", "anyOf": [{"minLength": 1}, {"type": "null"}]}, "null",
     'The request tool schema requires string; an anyOf branch without a type does not remove that requirement.'),
    ("arg_string_null_const", "7-5.const", {"anyOf": [{"const": "null"}, {"type": "integer"}]}, "null",
     'The request tool schema permits the literal string "null" through const, or an integer.'),
    ("arg_string_null_enum", "7-5.enum", {"type": "string", "enum": ["null", None]}, "null",
     'The request tool schema requires string alongside an enum containing string "null" and JSON null; the type excludes JSON null.'),
)

MIXED_CASE_FAMILIES = {
    "7-4.mixed_labels": {"glm47"},
    "7-4.mixed_grep": {"minimax_m3"},
}

MIXED_LABELS_SCHEMA = {"type": "object", "properties": {
    "label": {"anyOf": [{"type": "string"}, {"type": "null"}]},
    "note": {"type": ["string", "null"]},
    "literal": {"type": "string"},
}}
MIXED_LABELS_ARGS = {"label": None, "note": None, "literal": "null"}
MIXED_GREP_SCHEMA = {"properties": {
    "pattern": {"type": "string"},
    "path": {"anyOf": [{"type": "string"}, {"type": "null"}]},
}}
MIXED_GREP_ARGS = {"pattern": "null", "path": None}


def null_group(label):
    parent = label.split(".", 1)[0]
    return parent if parent in NULL_DESCRIPTIONS else None


def null_description(label, detail=""):
    return NULL_DESCRIPTIONS[null_group(label)] + (" " + detail if detail else "")
