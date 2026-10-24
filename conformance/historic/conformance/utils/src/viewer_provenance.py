# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Disposable viewer generation metadata, separate from deterministic catalogs."""
from datetime import datetime
from zoneinfo import ZoneInfo


def generation_metadata(revision: str, catalog_revision: str, command: str, output: str) -> dict:
    return {
        "schema": 1,
        "revision": revision,
        "generated_revision": revision,
        "catalog_revision": catalog_revision,
        "generated_on": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M %Z"),
        "command": command,
        "output": output,
    }
