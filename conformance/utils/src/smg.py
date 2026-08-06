# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the live SMG parser capture produced before a conformance render."""

from __future__ import annotations

import functools
import json
import os
from pathlib import Path
from typing import Any


@functools.lru_cache(maxsize=1)
def capture() -> dict[str, Any] | None:
    raw_path = os.environ.get("SMG_CAPTURE_PATH")
    if not raw_path:
        return None
    path = Path(raw_path)
    if not path.is_file():
        raise RuntimeError(f"SMG capture does not exist: {path}")
    doc = json.loads(path.read_text())
    if doc.get("schema") != "smg-conformance/v1":
        raise RuntimeError(f"unsupported SMG capture schema in {path}: {doc.get('schema')!r}")
    return doc


def tool_version() -> str | None:
    doc = capture()
    return str(doc["tool_parser_version"]) if doc else None


def reasoning_version() -> str | None:
    doc = capture()
    return str(doc["reasoning_parser_version"]) if doc else None


def tool(mode: str, family: str, case_id: str) -> dict[str, Any] | None:
    doc = capture()
    if not doc:
        return None
    return ((doc.get("toolcalling") or {}).get(mode) or {}).get(f"{family}:{case_id}")


def reasoning(mode: str, family: str, case_id: str) -> dict[str, Any] | None:
    doc = capture()
    if not doc:
        return None
    return ((doc.get("reasoning") or {}).get(mode) or {}).get(f"{family}:{case_id}")


def unified(family: str, scenario: str) -> dict[str, Any] | None:
    doc = capture()
    if not doc:
        return None
    return (doc.get("unified") or {}).get(f"{family}:{scenario}")


def coverage() -> dict[str, int]:
    doc = capture()
    return dict(doc.get("coverage") or {}) if doc else {}
