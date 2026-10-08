# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Authored decimal tokens; never construct the oracle through binary floats."""

import json
from decimal import Decimal


class NumericLiteral(str):
    """Native numeric token, distinct from a quoted string argument."""


NUMERIC_DESCRIPTIONS = {
    "7-14": "Integer-valued decimal/exponent fidelity: preserve mathematically integral numeric values, including large integers, negative exponents, and zero. Schema-directed grammars must permit an integer; already-typed grammars preserve native numbers. See numeric-failures.md.",
    "7-14.string": "Schema-directed fractional string fallback: Qwen3-Coder, MiniMax M2, GLM, and MiniMax M3 preserve fractional text as a string under an integer/string schema. JSON-native grammars do not apply this coercion. See numeric-failures.md.",
    "7-15": "Fractional numeric preservation: native number syntax preserves ordinary fractions, upward/downward rounding boundaries, negative values, and exponent notation. See numeric-failures.md.",
}

# (variant, schema, input token, expected JSON token). Expectations are decimal
# arithmetic, independent of parser output; the spelling also drives unit tests.
INTEGRAL_VARIANTS = [
    (f"{keyword}_{form}", {"type": ["integer", "null"], keyword: 42 if keyword == "const" else [42]}, raw, "42")
    for keyword in ("const", "enum")
    for form, raw in (("decimal", "42.0"), ("exponent", "4.2e1"))
] + [
    (name, {"type": ["integer", "string"]}, raw, expected)
    for name, raw, expected in (
        ("large_decimal", "9007199254740993.0", "9007199254740993"),
        ("large_exponent", "9.007199254740993e15", "9007199254740993"),
        ("negative", "-4.2E+1", "-42"),
        ("negative_exponent", "4200e-2", "42"),
        ("zero_underflow", "0.0e-400", "0"),
        ("fraction_near_integer", "42.0000000000000001", '\"42.0000000000000001\"'),
        ("fraction_underflow", "1e-400", '\"1e-400\"'),
        ("fraction_fallback", "42.5", '\"42.5\"'),
    )
]
FRACTIONAL_VARIANTS = [
    (name, {"type": "number"}, raw, raw)
    for name, raw in (
        ("ordinary", "42.5"),
        ("round_down", "9007199254740992.5"),
        ("round_up", "9007199254740993.1"),
        ("quarter", "9007199254740993.25"),
        ("exponent", "9.0071992547409925e15"),
        ("small", "0.10000000000000000001"),
        ("negative", "-9007199254740992.5"),
    )
]
NUMERIC_VARIANTS = [
    (f"arg_numeric_{group}_{name}", f"7-{group}.{name}", schema, raw, expected)
    for group, variants in ((14, INTEGRAL_VARIANTS), (15, FRACTIONAL_VARIANTS))
    for name, schema, raw, expected in variants
]


# Bare IDs are independent schema cases; only authored leaves belong here.
_STRING_FALLBACK_LABELS = {label for _, label, _, _, expected in NUMERIC_VARIANTS
                         if expected.startswith('"')}
_NUMERIC_GROUPS = {label: "7-14.string" if label in _STRING_FALLBACK_LABELS
                  else label.split(".", 1)[0] for _, label, *_ in NUMERIC_VARIANTS}
SCHEMA_DIRECTED_FAMILIES = {"qwen3", "qwen3_coder", "minimax_m2", "glm47", "minimax_m3"}


def numeric_group(label):
    return _NUMERIC_GROUPS.get(label)


def applicable(family, label):
    group = numeric_group(label)
    return group is not None and (group != "7-14.string" or family in SCHEMA_DIRECTED_FAMILIES)


def numeric_schema(family, label, schema):
    # The original Qwen/M2 union probes are immutable. GLM/M3 prefer strings in
    # that union, so their new numeric-fidelity probes require an integer alone.
    if numeric_group(label) == "7-14" and family in {"glm47", "minimax_m3"} and schema.get("type") == ["integer", "string"]:
        return {"type": "integer"}
    return schema


def numeric_description(label):
    return NUMERIC_DESCRIPTIONS[numeric_group(label)]


def numeric_expected(family, label, raw, expected):
    # Normalization is an existing Qwen/M2 contract. Other grammars may retain
    # the numeric spelling; the independent decimal oracle compares its value.
    if numeric_group(label) == "7-14" and family not in {"qwen3", "qwen3_coder", "minimax_m2"}:
        return raw
    return expected


def arguments_json(token):
    return '{"value":' + token + '}'


def _reject_json_constant(value):
    raise ValueError(f"non-JSON numeric constant: {value}")


def canonical_arguments(arguments):
    """Typed decimal comparison without conflating JSON strings and numbers."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments, parse_float=Decimal, parse_int=Decimal,
                                   parse_constant=_reject_json_constant)
        except ValueError:
            return ["invalid_json", arguments]
    return canonical_value(arguments)


def canonical_value(value):
    if isinstance(value, dict):
        return ["object", [[key, canonical_value(item)] for key, item in sorted(value.items())]]
    if isinstance(value, list):
        return ["array", [canonical_value(item) for item in value]]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return [type(value).__name__, value]
    number = Decimal(str(value))
    if not number.is_finite():
        return ["invalid_number", str(number)]
    # Decimal.normalize() uses the current precision and can itself round.
    sign, digits, exponent = number.as_tuple()
    digits = list(digits)
    while digits and digits[-1] == 0:
        digits.pop()
        exponent += 1
    return ["number", sign if digits else 0, digits, exponent if digits else 0]


def canonical_events(events):
    return [dict(event, arguments=canonical_arguments(event["arguments"]))
            if event.get("kind") == "tool_call" else event for event in events]
