import importlib.util
import json
import os
import shlex
import subprocess
from pathlib import Path

import pytest


def _load_validator():
    path = Path(__file__).parents[1] / "src" / "validate_conformance_status.py"
    spec = importlib.util.spec_from_file_location("validate_conformance_status", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _model(cell):
    return {
        "meta": {},
        "tabs": [{
            "id": "tab-unified",
            "kind": "unified",
            "label": "Unified",
            "candidates": [
                {"key": "dynamo", "label": "Dynamo current", "default_bucket": "A"},
                {"key": "golden", "label": "GOLDEN", "default_bucket": "C"},
            ],
            "columns": [{"sub": "case", "label": "1.a"}],
            "rows": [{"family": "qwen3", "model_label": "qwen3", "cells": cell}],
        }],
    }


def test_reports_empty_cells():
    validator = _load_validator()
    status = validator.build_status(_model({}), _model({})["tabs"], ["qwen3"], Path("report.html"))

    assert status["reports"][0]["empty"] == 1
    assert status["reports"][0]["issues"][0]["reason"] == "no cell was emitted for this model/case pair"


def test_reports_unified_reference_mismatches_as_red():
    validator = _load_validator()
    cell = {
        "case": {
            "red_on_diff": True,
            "cmp": {
                "golden": {"sig": 1, "leak": 0, "na": 0, "err": 0},
                "dynamo": {"sig": 2, "leak": 0, "na": 0, "err": 0},
            },
        }
    }
    model = _model(cell)
    status = validator.build_status(model, model["tabs"], ["qwen3"], Path("report.html"))

    assert status["reports"][0]["red"] == 1
    assert status["reports"][0]["issues"][0]["reason"] == "the default Reference differs from GOLDEN"


def test_green_unified_cell_has_no_issues():
    validator = _load_validator()
    cell = {
        "case": {
            "red_on_diff": True,
            "cmp": {
                "golden": {"sig": 1, "leak": 0, "na": 0, "err": 0},
                "dynamo": {"sig": 1, "leak": 0, "na": 0, "err": 0},
            },
        }
    }
    model = _model(cell)
    status = validator.build_status(model, model["tabs"], ["qwen3"], Path("report.html"))

    assert status["reports"][0]["empty"] == 0
    assert status["reports"][0]["red"] == 0
    assert status["reports"][0]["issues"] == []


def test_explicit_unified_na_is_not_empty():
    validator = _load_validator()
    cell = {
        "case": {
            "kind": "cell",
            "status": "na",
            "tooltip": {"na_note": "requires a different grammar"},
            "cmp": {"dynamo": {"sig": 0, "leak": 0, "na": 1, "err": 0}},
        }
    }
    model = _model(cell)
    status = validator.build_status(model, model["tabs"], ["qwen3"], Path("report.html"))

    report = status["reports"][0]
    assert report["empty"] == 0
    assert report["red"] == 0
    assert report["na"] == 1
    assert report["issues"] == []


@pytest.fixture
def inventory_report(tmp_path):
    fixtures = tmp_path / "unified"
    sources = ("dynamo_v2-0.6.1", "dynamo_v2-0.6.0", "vllm_python-0.26.0", "vllm_rust-0.26.0")
    keys = ("dynamo", "dynamo@0.6.0", "vllm_python@0.26.0", "vllm_rust@0.26.0")
    model = _model({})
    tab = model["tabs"][0]
    tab["candidates"][0]["version"] = "0.6.1"
    for source, key in zip(sources, keys):
        (fixtures / source).mkdir(parents=True)
        if key != "dynamo":
            tab["candidates"].append({"key": key, "label": source, "version": source.split("-")[1]})
    inputs = fixtures / "inputs" / "qwen3" / "UNIFIED.case.yaml"
    inputs.parent.mkdir(parents=True)
    inputs.write_text("cases:\n  sample:\n    scenario: case\n")
    tab["rows"][0]["cells"]["case"] = {
        "kind": "cell", "status": "ok", "red_on_diff": True,
        "cmp": {candidate["key"]: {"sig": 1, "leak": 0, "na": 0, "err": 0} for candidate in tab["candidates"]},
        "tooltip": {"candidates": [{"key": candidate["key"], "block": {"events": []}} for candidate in tab["candidates"]]},
    }
    return fixtures, model


def _html(model):
    return '<script type="application/json" id="conformance-model">' + json.dumps(model) + '</script>'


def test_accepts_snapshot_inventory(inventory_report):
    fixtures, model = inventory_report
    assert _load_validator().validate_unified_inventory(model, fixtures) == []


@pytest.mark.parametrize("key", ["dynamo@0.6.0", "vllm_python@0.26.0", "vllm_rust@0.26.0"])
def test_rejects_missing_recorded_column(inventory_report, key):
    fixtures, model = inventory_report
    tab = model["tabs"][0]
    tab["candidates"] = [candidate for candidate in tab["candidates"] if candidate["key"] != key]
    with pytest.raises(ValueError, match="missing recorded version columns"):
        _load_validator().validate_unified_inventory(model, fixtures)


@pytest.mark.parametrize("damage, message", [
    ("family", "missing recorded family"),
    ("scenario", "missing recorded scenario"),
    ("cell", "missing cell"),
    ("comparison", "missing comparison or popup data"),
    ("popup", "missing comparison or popup data"),
    ("output", "missing captured output"),
    ("field", "incomplete comparison"),
    ("duplicate_candidate", "duplicate candidate keys"),
    ("duplicate_family", "duplicate family rows"),
    ("golden", "missing GOLDEN comparison column"),
])
def test_inventory_rejects_incomplete_report(inventory_report, damage, message):
    fixtures, model = inventory_report
    tab = model["tabs"][0]
    cell = tab["rows"][0]["cells"]["case"]
    if damage == "family":
        tab["rows"].clear()
    elif damage == "scenario":
        tab["columns"][0]["sub"] = "different"
    elif damage == "cell":
        tab["rows"][0]["cells"].clear()
    elif damage == "comparison":
        del cell["cmp"]["dynamo"]
    elif damage == "popup":
        cell["tooltip"]["candidates"].pop()
    elif damage == "output":
        cell["tooltip"]["candidates"][0]["block"] = {"unavailable": "missing"}
    elif damage == "field":
        del cell["cmp"]["dynamo"]["sig"]
    elif damage == "duplicate_candidate":
        tab["candidates"].append(tab["candidates"][0])
    elif damage == "duplicate_family":
        tab["rows"].append(tab["rows"][0])
    elif damage == "golden":
        tab["candidates"] = [candidate for candidate in tab["candidates"] if candidate["key"] != "golden"]
    with pytest.raises(ValueError, match=message):
        _load_validator().validate_unified_inventory(model, fixtures)


def test_cli_warns_for_unavailable_peer(inventory_report, tmp_path, capsys):
    fixtures, model = inventory_report
    cell = model["tabs"][0]["rows"][0]["cells"]["case"]
    key = "vllm_rust@0.26.0"
    cell["cmp"][key]["na"] = 1
    cell["tooltip"]["candidates"][-1]["block"] = {"unavailable": "not captured"}
    report = tmp_path / "report.html"
    report.write_text(_html(model))
    assert _load_validator().main(["--html", str(report), "--unified-fixtures", str(fixtures), "--summary-only"]) == 0
    assert "WARNING: Unified vllm_rust-0.26.0: 1 applicable cells unavailable; 0 captured results" in capsys.readouterr().err


def test_cli_warns_without_require_green(inventory_report, tmp_path, capsys):
    _, model = inventory_report
    model["tabs"][0]["rows"][0]["cells"]["case"]["cmp"]["dynamo"]["sig"] = 2
    report = tmp_path / "report.html"
    report.write_text(_html(model))
    assert _load_validator().main(["--html", str(report), "--summary-only"]) == 0
    assert "WARNING: selected Reference cells include 0 empty and 1 red" in capsys.readouterr().err


@pytest.mark.parametrize("invalid", [True, False])
def test_cli_error_preserves_status(inventory_report, tmp_path, capsys, invalid):
    fixtures, model = inventory_report
    report = tmp_path / "report.html"
    status = tmp_path / "report.json"
    status.write_text("previous status")
    if invalid:
        report.write_text("no model")
    else:
        model["tabs"][0]["rows"][0]["cells"]["case"]["cmp"]["dynamo"]["sig"] = 2
        report.write_text(_html(model))
    rc = _load_validator().main([
        "--html", str(report), "--status-path", str(status),
        "--unified-fixtures", str(fixtures), "--require-green",
    ])
    assert rc == (2 if invalid else 1)
    assert "ERROR:" in capsys.readouterr().err
    assert status.read_text() == "previous status"


@pytest.mark.parametrize("failure", ["missing_history", "generator", None])
def test_renderer_validates_before_publishing(inventory_report, tmp_path, failure):
    # Use the real wrapper and validator with a tiny generator so failure paths
    # do not rebuild fixtures or overwrite the developer's published report.
    fixtures, model = inventory_report
    utils = Path(__file__).parents[1]
    sandbox = tmp_path / "utils"
    tools = sandbox / "src"
    tools.mkdir(parents=True)
    stage = tmp_path / "stage"
    (stage / "tests" / "parity").mkdir(parents=True)
    wrapper = sandbox / "render_table_v2.sh"
    wrapper.write_bytes((utils / "render_table_v2.sh").read_bytes())
    for module in ("validate_conformance_status.py", "case_variants.py", "null_cases.py"):
        (tools / module).write_bytes((utils / "src" / module).read_bytes())
    (tools / "_common.sh").write_text(
        "set -euo pipefail\n"
        + f"ROOT={shlex.quote(str(tmp_path))}\n"
        + f"TOOLS={shlex.quote(str(tools))}\n"
        + f"STAGE={shlex.quote(str(stage))}\n"
        + f"FIXTURES_SNAP={shlex.quote(str(fixtures.parent))}\n"
        + "build_stage_conformance() { :; }\n"
    )
    if failure == "missing_history":
        model["tabs"][0]["candidates"].pop()
    document = _html(model)
    generator = stage / "tests" / "parity" / "generate_conformance_table.py"
    generator.write_text(f"print({document!r})\nraise SystemExit({7 if failure == 'generator' else 0})\n")
    out = tmp_path / "report.html"
    status = tmp_path / "report.json"
    out.write_text("previous HTML")
    status.write_text("previous status")
    result = subprocess.run(
        ["bash", str(wrapper), "--output", str(out)], capture_output=True, text=True,
        env={"PATH": os.environ["PATH"]},
    )
    if failure:
        assert result.returncode == (7 if failure == "generator" else 2), result.stderr
        assert "ERROR:" in result.stderr
        assert "untouched" in result.stderr
        assert out.read_text() == "previous HTML"
        assert status.read_text() == "previous status"
    else:
        assert result.returncode == 0, result.stderr
        assert out.read_text() == document + "\n"
        assert json.loads(status.read_text())["html"] == str(out)
        assert out.stat().st_mode & 0o004
    assert not list(tmp_path.glob("report.html.*"))
    assert not list(tmp_path.glob("report.json.*"))
