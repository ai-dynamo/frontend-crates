# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Original request bindings shared by report equivalence and current-source checks."""
import hashlib
import json
from pathlib import Path


def read_bindings(directory: Path) -> dict:
    path = directory / "capture-inputs.json"
    if not path.exists():
        return {}
    doc = json.loads(path.read_text())
    if doc.get("schema_version") != 1 or not isinstance(doc.get("records"), dict):
        raise ValueError(f"invalid capture input bindings: {path}")
    return doc["records"]


def original_capture_input(record: dict, raw: bytes, relative: str, bindings: dict) -> dict | None:
    original = record.get("capture_input")
    if original is None and relative in bindings:
        binding = bindings[relative]
        if binding["capture_sha256"] != hashlib.sha256(raw).hexdigest():
            raise ValueError(f"capture input binding does not match capture bytes: {relative}")
        original = binding["capture_input"]
    return original
