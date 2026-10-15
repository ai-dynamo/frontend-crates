# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path


def test_capture_selector_delegates_comparison_to_the_existing_table_owner():
    assets = Path(__file__).resolve().parents[1] / "src/assets"
    selector = (assets / "comparison_layout.js").read_text()
    table = (assets / "conformance_view.js").read_text()
    comparison = (assets / "conformance.js").read_text()

    assert "window.__applyConformanceSelection(panel, selection.keys, null)" in selector
    assert "function applyCtl(panel)" in comparison
    assert "const oracle = panel.id === 'tab-unified' && !!panel._comparisonSelection" in comparison
    assert "compareBarHtml(tab)" in table
    assert "data-red-on-diff" in table
    assert "function outputStatus(" not in selector
