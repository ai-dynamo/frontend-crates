# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Request schemas shared by the Dynamo and peer Unified capture harnesses."""

import json
from pathlib import Path

SCHEMA_PATH = Path(__file__).with_suffix(".json")


def unified_tools() -> list[dict]:
    return json.loads(SCHEMA_PATH.read_text())


def openai_tools(schemas: list[dict] | None = None) -> list[dict]:
    return [
        {"type": "function", "function": schema}
        for schema in (unified_tools() if schemas is None else schemas)
    ]
