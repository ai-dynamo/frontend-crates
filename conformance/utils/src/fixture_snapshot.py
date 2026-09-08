# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve the immutable fixture snapshot used by Python conformance tools."""

import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path


@lru_cache
def fixture_snapshot_root() -> Path:
    """Return one immutable snapshot path, never the mutable cache symlinks."""
    env = os.environ.get("CONFORMANCE_FIXTURES_ROOT")
    root = Path(env) if env else None
    if root is None or any((root / name).is_symlink() for name in ("toolcalling", "reasoning", "unified")):
        script = Path(__file__).resolve().parent / "extract_fixtures.py"
        proc = subprocess.run(
            [sys.executable, str(script)],
            check=True,
            capture_output=True,
            text=True,
        )
        printed = proc.stdout.strip().splitlines()
        if not printed:
            raise RuntimeError("extract_fixtures.py returned no snapshot path")
        root = Path(printed[-1])
    if not root.is_dir():
        raise RuntimeError(f"fixture snapshot path is not a directory: {root}")
    return root
