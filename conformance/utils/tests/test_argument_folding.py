# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Display folds retain the authored fixtures and their independent observations."""
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from schema_cases import ARGUMENT_SECTIONS, CONFORMANCE_CASES, argument_section, schema_fold, schema_tools
import unified_taxonomy as taxonomy
from conformance.utils.tests.viewer_test_support import run_node_contract


def test_exact_fold_dimensions_and_section_order():
    groups = {}
    for case in CONFORMANCE_CASES:
        scenario = "schema_" + case["name"]
        fold = schema_fold(scenario)
        if fold:
            groups.setdefault(fold[0], []).append((scenario, fold[1]))
    assert len(groups) == 22
    for parent, children in groups.items():
        assert len(children) == 4
        expected = ({"Direct", "Explicit string type", "allOf", "Reference"} if "const_" in parent
                    else {"String / Forward", "String / Reversed", "Typed value / Forward", "Typed value / Reversed"})
        assert {label for _, label in children} == expected
        assert parent in {scenario for scenario, _ in children}
    ordered = sorted((scenario for scenario in taxonomy.UNIFIED_TAX if taxonomy.tax(scenario)[0] == 7),
                     key=taxonomy.display_sort_key)
    assert len(ordered) == 139
    assert list(dict.fromkeys(argument_section(s) for s in ordered)) == list(ARGUMENT_SECTIONS)


def test_display_fold_reader_contracts():
    output = run_node_contract("retirement_contract.test.js", "--suite", "folds")
    assert "folds contracts passed" in output
