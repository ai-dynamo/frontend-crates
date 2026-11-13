# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path


def test_capture_selector_leaves_loaded_comparison_as_selection_authority():
    assets = Path(__file__).resolve().parents[1] / "src/assets"
    selector = (assets / "conformance.js").read_text()
    table = (assets / "conformance_view.js").read_text()
    comparison = (assets / "conformance.js").read_text()
    outcome = (Path(__file__).resolve().parents[3] / "conformance/viewer/model.js").read_text()

    assert "panel._comparisonSelection = { keys: [], reference: null };" in selector
    assert "__applyConformanceSelection(panel, selection.keys, null)" not in selector
    assert "function applyCtl(panel)" in comparison
    assert "const oracle = panel.id === 'tab-unified' && !!panel._comparisonSelection" in comparison
    assert "compareBarHtml(tab)" in table
    assert "data-red-on-diff" in table
    assert "function outputStatus(" not in comparison
    assert "window.__conformanceCaptureOutcome(data, key, 'golden')" in comparison
    assert "export function selectedCaptureOutcome" in outcome
