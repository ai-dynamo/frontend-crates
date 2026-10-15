# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures: ONE real render per test session, reused by every browser test.

Both browser modules must take the page from here. test_browser_smoke.py used to
declare its own module-scoped copy of this fixture, so a full run rendered the table
twice — and the two renders were the bulk of the suite's wall clock."""
import subprocess
import sys
from pathlib import Path

import pytest

UTILS = Path(__file__).resolve().parents[1]
REPO = UTILS.parents[1]
sys.path.insert(0, str(UTILS / "src"))


@pytest.fixture(scope="session")
def rendered_page(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("render") / "CONFORMANCE_v2.html"
    subprocess.run(
        [str(UTILS / "render_table_v2.sh"), "--output", str(out)],
        check=True, cwd=REPO, capture_output=True, text=True,
    )
    return out


@pytest.fixture(scope="session")
def reference_page(legacy_rendered_page: Path, tmp_path_factory) -> Path:
    """Exercise the retained reference renderer without the comparison projection.

    The comparison browser tests use rendered_page. Reference popup tests keep
    their original controls/orientation so renderer content can be compared independently.
    """
    page = legacy_rendered_page.read_text()
    layout = (UTILS / "src/assets/comparison_layout.js").read_text()
    script = "<script>\n" + layout + "\n</script>"
    assert page.count(script) == 1
    page = page.replace(script, "")
    # The reference harness retains the original controls alongside archived tabs.
    hidden_controls = ".tab-panel#tab-unified > .cmpctl { display: none; }"
    assert page.count(hidden_controls) == 1
    page = page.replace(hidden_controls, "")
    out = tmp_path_factory.mktemp("reference-render") / "CONFORMANCE_reference.html"
    out.write_text(page)
    return out


@pytest.fixture(scope="session")
def legacy_rendered_page(tmp_path_factory) -> Path:
    """Keep archived-corpus regression coverage outside the public report."""
    out = tmp_path_factory.mktemp("legacy-render") / "CONFORMANCE_legacy.html"
    command = """set -e
source conformance/utils/src/_common.sh
build_stage_conformance
cd "$STAGE"
PYTHONPATH="$STAGE/tests/parity" python3 -c 'import sys; from pathlib import Path; import generate_conformance_table as t; print(t.render_combined_html(output_path=Path(sys.argv[1]), artifact_root=Path(sys.argv[2]), include_legacy=True))' "$1" "$2"
"""
    with out.open("w") as output:
        subprocess.run(["bash", "-c", command, "legacy-render", str(out), str(REPO)], cwd=REPO, stdout=output, check=True)
    return out
