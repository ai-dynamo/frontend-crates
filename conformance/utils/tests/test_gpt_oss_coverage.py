# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import copy
from pathlib import Path

import yaml_fast  # noqa: F401
import yaml

import gen_unified_golden
import normalized_yaml
from conformance.utils.tests.fixture_test_support import semantic_key


ROOT = Path(__file__).resolve().parents[3]


def coverage_key(case, *, ignore_schema=False):
    case = copy.deepcopy(case)
    if ignore_schema:
        assert case["request"]["init"]["tool_output_mode"] == "Native"
        case["request"]["tools"] = []
    return semantic_key(case)


def authored_cases():
    path = ROOT / "conformance/fixtures-unified-v2/families/gpt_oss/inputs_and_golden.yaml"
    return normalized_yaml.expand_family(yaml.safe_load(path.read_text()))["cases"]


def test_native_schema_retirements_preserve_exact_request_and_oracle():
    cases = authored_cases()
    exclusions = gen_unified_golden._MANIFEST["gpt_oss"]["scenario_exclusions"]
    by_scenario = {case["scenario"]: case for case in cases.values()}
    checked = 0
    for scenario, reason in exclusions.items():
        marker = "identical request and oracle retained as "
        if marker not in reason:
            continue
        parent = reason.split(marker)[1].removesuffix(".")
        assert by_scenario[scenario]["lifecycle"] == "retired"
        assert cases[parent]["lifecycle"] == "active"
        assert coverage_key(by_scenario[scenario], ignore_schema=True) == coverage_key(cases[parent], ignore_schema=True)
        checked += 1
    assert checked == 109


def test_duplicate_guard_detects_schema_repetitions_and_keeps_delivery():
    case = next(case for case in authored_cases().values() if case["lifecycle"] == "active" and case["request"]["init"]["tool_output_mode"] == "Native")
    duplicate = copy.deepcopy(case)
    assert coverage_key(case) == coverage_key(duplicate)
    duplicate["request"]["tools"] = [{"name": "changed_schema", "parameters": {"type": "object"}}]
    assert coverage_key(case) != coverage_key(duplicate)
    assert coverage_key(case, ignore_schema=True) == coverage_key(duplicate, ignore_schema=True)
    duplicate["request"]["chunks"].append({"delta_text": ""})
    assert coverage_key(case, ignore_schema=True) != coverage_key(duplicate, ignore_schema=True)
    duplicate = copy.deepcopy(case)
    duplicate["request"]["init"]["starting_state"] = "Reasoning"
    assert coverage_key(case, ignore_schema=True) != coverage_key(duplicate, ignore_schema=True)


def test_active_gpt_oss_requests_are_not_exact_or_schema_only_duplicates():
    seen = {}
    for scenario, case in authored_cases().items():
        if case["lifecycle"] != "active":
            continue
        key = coverage_key(case, ignore_schema=case["request"]["init"]["tool_output_mode"] == "Native")
        assert key not in seen, (scenario, seen.get(key))
        seen[key] = scenario


def test_corrected_oracle_is_generated_without_changing_requests():
    cases = authored_cases()
    generated = gen_unified_golden.build_cases("gpt_oss")
    for scenario in gen_unified_golden.GPT_OSS_ORACLE_CORRECTIONS:
        assert generated[f"UNIFIED.{scenario}.gpt_oss"]["golden"] == cases[scenario]["golden"]["assembled"]
    assert {case["scenario"] for case in cases.values() if case["lifecycle"] == "active"} == {
        key.removeprefix("UNIFIED.").removesuffix(".gpt_oss") for key in generated
    }
