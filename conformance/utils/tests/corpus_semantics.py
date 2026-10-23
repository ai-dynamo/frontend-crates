# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from collections import defaultdict

from unified_tools import unified_tools


def semantic_key(case):
    request = case.get("request", case)
    golden = case["golden"]
    if isinstance(golden, dict):
        golden = golden["assembled"]
    return json.dumps({
        "input": request["input"],
        "init": request["init"],
        "tools": request.get("tools") if request.get("tools") is not None else unified_tools(),
        "golden": golden,
        "chunks": request.get("chunks", case.get("input_chunks")),
        "finish_reason": request.get("finish_reason", "stop"),
    }, sort_keys=True)


def assert_unique_behaviour(family, cases, *, retained_archives=None):
    retained_archives = retained_archives or {}
    entries = list(cases.items()) if isinstance(cases, dict) else list(cases)
    seen_ids = set()
    seen_semantics = defaultdict(list)
    active_ids = set()
    retired_ids = []
    for case_id, case in entries:
        assert case_id not in seen_ids, f"{family}: reused authored case ID {case_id}"
        seen_ids.add(case_id)
        if case.get("lifecycle") == "retired":
            retired_ids.append(case_id)
            continue
        active_ids.add(case_id)
        seen_semantics[semantic_key(case)].append(case_id)
    assert set(retained_archives) <= active_ids, f"{family}: exception has no active authored case"
    for case_id, binding in retained_archives.items():
        assert all(binding.get(field) for field in ("purpose", "input_source", "capture_source", "version")), (
            f"{family}/{case_id}: retained archive exception requires source and version binding")
    duplicates = [names for names in seen_semantics.values() if len(names) > 1]
    for names in duplicates:
        canonical = [name for name in names if name not in retained_archives]
        assert len(canonical) <= 1, f"{family}: scenarios with identical behaviour: {names}"
    return {"family": family, "active_cases": len(active_ids), "retired_cases": retired_ids, "exact_duplicates": duplicates,
            "retained_archive_exceptions": retained_archives}
