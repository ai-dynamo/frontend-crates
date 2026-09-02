import importlib.util
from pathlib import Path


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
