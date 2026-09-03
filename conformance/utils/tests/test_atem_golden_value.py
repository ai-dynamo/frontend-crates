# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""An ATEM parameter body must type back to the value the golden records.

`golden_of` writes the RAW segment value into the golden tool-call arguments, and the
muse parser reads a parameter body with `value_parser: json` and `allow_non_json: true`.
So the body `_atem_value` emits has to survive that read unchanged, or the corpus
authors a golden no correct parser can ever emit.

`unified_parity` cannot catch the gap on its own: it only bites once a scenario carries
a value of the failing shape, and the corpus carries none today.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gen_unified_golden import _atem_value  # noqa: E402


def _typed(body: str):
    """How the parser types one parameter body."""
    try:
        return json.loads(body)
    except ValueError:
        return body


@pytest.mark.parametrize(
    "val",
    [
        # Not JSON at all, so the bare body already types as this exact string.
        "Paris", "São Paulo 東京", "  ", "-",
        # JSON scalars and containers: bare, each types as something that is not a
        # string, so each needs the quoted spelling.
        "1", "0", "1.5", "true", "false", "null", "[1,2]", '{"a":1}',
        # A value that is itself a JSON string. It parses to a string, but to the
        # UNQUOTED one, so leaving it bare loses the quotes the golden records.
        '"hi"', '""', '"1"',
    ],
)
def test_the_emitted_body_types_back_to_the_value(val):
    assert _typed(_atem_value(val)) == val
