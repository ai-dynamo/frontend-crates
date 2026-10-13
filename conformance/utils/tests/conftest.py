# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures: ONE real render per test session, reused by every browser test.

Both browser modules must take the page from here. test_browser_smoke.py used to
declare its own module-scoped copy of this fixture, so a full run rendered the table
twice — and the two renders were the bulk of the suite's wall clock."""
import subprocess
from pathlib import Path

import pytest

UTILS = Path(__file__).resolve().parents[1]
REPO = UTILS.parents[1]


@pytest.fixture(scope="session")
def rendered_page(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("render") / "CONFORMANCE_v2.html"
    subprocess.run(
        [str(UTILS / "render_table_v2.sh"), "--output", str(out)],
        check=True, cwd=REPO, capture_output=True, text=True,
    )
    return out


@pytest.fixture(scope="session")
def reference_page(rendered_page: Path, tmp_path_factory) -> Path:
    """Exercise the retained reference renderer without the comparison projection.

    The comparison browser tests use rendered_page. Reference popup tests keep
    their original controls/orientation so renderer content can be compared independently.
    """
    page = rendered_page.read_text()
    layout = (UTILS / "src/assets/comparison_layout.js").read_text()
    script = "<script>\n" + layout + "\n</script>"
    assert page.count(script) == 1
    page = page.replace(script, "")
    page = page.replace("  window.__comparisonLayoutEnabled = true;", "")
    page = page.replace("  document.documentElement.classList.add('comparison-page', 'comparison-pending');", "")
    out = tmp_path_factory.mktemp("reference-render") / "CONFORMANCE_reference.html"
    out.write_text(page)
    return out
