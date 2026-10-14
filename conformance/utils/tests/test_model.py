# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The production JavaScript engine owns matrix assertions; no generated HTML fixture."""
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]

def test_shared_javascript_matrix_contracts():
    # Rebuilding alternate selections for the full matrix can exceed 300s under xdist load.
    result = subprocess.run(["node", str(ROOT / "conformance/viewer/models.test.js")], cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr
    assert "Shared matrix structural contracts passed:" in result.stdout


@pytest.mark.parametrize(("unavailable", "reason"), [
    ({"code": "missing_result"}, "missing_result"),
    ({"code": "missing_result", "detail": ""}, "missing_result"),
    ({"code": "missing_result", "detail": "No retained capture."}, "No retained capture."),
    ("legacy explanation", "legacy explanation"),
    ({"code": "<missing>&", "detail": ""}, "&lt;missing&gt;&amp;"),
    ({"code": "missing", "detail": "<script>&\\\"'"}, "&lt;script&gt;&amp;\\&quot;&#39;"),
])
def test_unavailable_popup_renders_detail_or_code(unavailable, reason):
    script = r"""
const fs = require('fs');
const vm = require('vm');
const context = {window: {}, document: {cookie: '', documentElement: {setAttribute() {}},
  querySelectorAll() {return [];}, addEventListener() {}, getElementById() {return null;}}};
vm.createContext(context);
const source = fs.readFileSync(process.argv[1], 'utf8');
const marker = '// --- Output block rendering (mirrors _format_output_block_html) ------------';
vm.runInContext(source.replace(marker, 'window.audit = {outputBlock};\n' + marker), context);
const unavailable = JSON.parse(fs.readFileSync(0, 'utf8'));
process.stdout.write(context.window.audit.outputBlock({unavailable}, 'gemma4', {}, []));
"""
    rendered = subprocess.run(
        ["node", "-e", script, str(ROOT / "conformance/utils/src/assets/conformance_view.js")],
        input=json.dumps(unavailable), text=True, capture_output=True, check=True,
    ).stdout
    assert rendered == f'<span class="expl">unavailable: {reason}</span>'
