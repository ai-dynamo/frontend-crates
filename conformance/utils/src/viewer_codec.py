# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lossless wire values for the viewer and its independent Python oracle."""
from __future__ import annotations

import base64
import datetime
import json
import math


def encode_typed(value, *, canonical=False):
    if value is None:
        return ["null"]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", str(value)]
    if isinstance(value, float):
        return ["float", "nan" if math.isnan(value) else repr(value)]
    if isinstance(value, str):
        return ["str", value]
    if isinstance(value, datetime.datetime):
        return ["datetime", value.isoformat()]
    if isinstance(value, datetime.date):
        return ["date", value.isoformat()]
    if isinstance(value, bytes):
        return ["binary", base64.b64encode(value).decode("ascii")]
    if isinstance(value, list):
        return ["array", [encode_typed(item, canonical=canonical) for item in value]]
    if isinstance(value, dict):
        entries = [[encode_typed(key, canonical=canonical), encode_typed(item, canonical=canonical)]
                   for key, item in value.items()]
        if canonical:
            entries.sort(key=lambda entry: json.dumps(entry[0], ensure_ascii=False, separators=(",", ":")))
        return ["object", entries]
    raise TypeError(f"unsupported viewer value: {type(value).__name__}")


def canonical_typed(value):
    """Compare YAML values without stringifying keys or conflating scalar types."""
    return json.dumps(encode_typed(value, canonical=True), ensure_ascii=False, separators=(",", ":"))


def typed_key_identity(value):
    return ("str", value) if type(value) is str else ("typed", canonical_typed(value))


def typed_equal(left, right):
    if type(left) is not type(right):
        return False
    if left is right:
        return True
    if isinstance(left, dict):
        if all(type(key) is str for key in left) and all(type(key) is str for key in right):
            return left.keys() == right.keys() and all(typed_equal(value, right[key]) for key, value in left.items())
        right_items = {typed_key_identity(key): value for key, value in right.items()}
        if len(left) != len(right):
            return False
        for key, value in left.items():
            identity = typed_key_identity(key)
            if identity not in right_items or not typed_equal(value, right_items[identity]):
                return False
        return True
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(typed_equal(a, b) for a, b in zip(left, right))
    return encode_typed(left) == encode_typed(right)
