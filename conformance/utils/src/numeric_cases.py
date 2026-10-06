# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Authored decimal tokens; never construct the oracle through binary floats."""

import json
from decimal import Decimal


class NumericLiteral(str):
    """Native numeric token, distinct from a quoted string argument."""


NUMERIC_DESCRIPTIONS = {
    "7-14": "Integral decimal/exponent conversion: Qwen3-Coder and MiniMax-M2 convert mathematically integral decimal tokens under literal or integer/string schemas; fractional tokens fall back to strings. See numeric-failures.md.",
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


def numeric_group(label):
    parent = label.split(".", 1)[0]
    return parent if parent in NUMERIC_DESCRIPTIONS else None


def applicable(family, label):
    return numeric_group(label) == "7-15" or family in {"qwen3", "qwen3_coder", "minimax_m2"}


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
