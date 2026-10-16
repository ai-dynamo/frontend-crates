# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lossless wire values for the viewer and its independent Python oracle."""
from __future__ import annotations

import base64
import datetime
import math


def encode_typed(value):
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
        return ["array", [encode_typed(item) for item in value]]
    if isinstance(value, dict):
        return ["object", [[encode_typed(key), encode_typed(item)] for key, item in value.items()]]
    raise TypeError(f"unsupported viewer value: {type(value).__name__}")
