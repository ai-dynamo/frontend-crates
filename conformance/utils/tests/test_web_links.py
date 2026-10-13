# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Maintained-viewer source links have one JavaScript owner."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

def test_browser_source_links():
    result = subprocess.run(["node", str(ROOT / "conformance/viewer/links.test.js")], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "immutable publication identity" in result.stdout
