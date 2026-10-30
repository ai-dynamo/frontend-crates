# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The production JavaScript engine owns matrix assertions; no generated HTML fixture."""
import json
import subprocess
from pathlib import Path

import pytest
import ci_coverage
from verify_publication_transport import materialize_embedded, runtime_catalog

ROOT = Path(__file__).resolve().parents[3]

def test_shared_javascript_matrix_contracts(published_viewer, tmp_path):
    catalog = tmp_path / "runtime-catalog.json"
    runtime = runtime_catalog(published_viewer)
    catalog.write_text(json.dumps(runtime))
    transport = (published_viewer if (published_viewer / "conformance/viewer-resolved-catalog.json.gz").is_file()
                 else materialize_embedded(published_viewer, tmp_path / "transport"))
    # Candidate controls also include versions without retained measurements.
    # Partition those controls, rather than dropping them through bundle selection.
    rows = [{**row, "path": row["id"]} for row in runtime["sources"]["candidates"]
            if row["mode"] == "unified"]
    selection = tmp_path / "model-selection.json"
    selected = ci_coverage.select_resources(rows)
    selection.write_text(json.dumps(selected))
    selection_args = [str(selection)] if len(selected) < len(rows) else []
    result = subprocess.run(["node", str(ROOT / "conformance/viewer/qualification.js"), "models",
                             str(catalog), str(transport), *selection_args],
                            cwd=ROOT, capture_output=True, text=True, timeout=60 * 60)
    assert result.returncode == 0, result.stderr
    assert "Shared matrix structural contracts passed:" in result.stdout
    print(result.stdout)


@pytest.mark.parametrize(("unavailable", "reason"), [
    ({"code": "missing_result"}, "missing_result"),
    ({"code": "missing_result", "detail": ""}, "missing_result"),
    ({"code": "missing_result", "detail": "No retained capture."}, "No retained capture."),
    ("legacy explanation", "legacy explanation"),
    ({"code": "<missing>&", "detail": ""}, "&lt;missing&gt;&amp;"),
    ({"code": "missing", "detail": "<script>&\\\"'"}, "&lt;script&gt;&amp;\\&quot;&#39;"),
    ({"code": "missing_capture", "detail": "Output not yet captured"}, "Output not yet captured"),
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
    prefix = "" if isinstance(unavailable, dict) and unavailable.get("code") == "missing_capture" else "unavailable: "
    assert rendered == f'<span class="expl">{prefix}{reason}</span>'
