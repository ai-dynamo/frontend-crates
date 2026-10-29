# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import report_capture_verification


@pytest.mark.parametrize("fault", [None, "golden", "recorded", "missing", "empty", "wrong_severity", "stale"])
def test_executed_receipts_warn_for_golden_and_fail_for_unverified_yaml(tmp_path, fault, monkeypatch):
    monkeypatch.delenv("CONFORMANCE_CANDIDATE_TREE", raising=False)
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    if fault == "stale":
        monkeypatch.setenv("GITHUB_SHA", "new-head")
    for suite, blocking in report_capture_verification.SUITES.items():
        if fault == "missing" and blocking:
            continue
        differences = ["case: expected differs from actual"] if fault == ("recorded" if blocking else "golden") else []
        document = {"schema": 1, "suite": suite, "checked": 0 if fault == "empty" else 1,
                    "blocking": False if fault == "wrong_severity" else blocking, "differences": differences}
        (tmp_path / f"{suite}.json").write_text(json.dumps(document))
    script = Path(report_capture_verification.__file__)
    result = subprocess.run([sys.executable, str(script), "--directory", str(tmp_path), "--summary", str(tmp_path / "summary")],
                            capture_output=True, text=True)
    assert (result.returncode == 0) == (fault in {None, "golden"}), result.stderr
    if fault == "golden":
        assert "::warning::" in result.stdout
        assert "::error::" not in result.stdout
    if fault in {"recorded", "missing"}:
        assert "::error::" in result.stdout
    if fault in {None, "golden", "recorded", "missing"}:
        assert "recorded-capture" in (tmp_path / "summary").read_text()


def test_required_rust_job_publishes_executed_comparisons_even_after_failure():
    root = Path(__file__).resolve().parents[3]
    workflow = yaml.safe_load((root / ".github/workflows/ci.yml").read_text())
    steps = workflow["jobs"]["rust"]["steps"]
    execute = next(step for step in steps if step.get("name") == "Run tests")
    assert "cargo nextest run --workspace --all-targets --locked" in execute["run"]
    assert "CONFORMANCE_CAPTURE_REPORT_DIR" in execute["env"]
    publish = next(step for step in steps if step.get("name") == "Publish capture verification and GOLDEN warnings")
    assert publish["if"] == "${{ always() }}"
    assert "report_capture_verification.py" in publish["run"]
    assert not (root / ".github/workflows/capture-approval.yml").exists()
