# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model-fold contract on the resolver (pure logic, no rendered HTML).

Born from a real escape: the resolver fold doubled Dynamo v1 output into
`calls=[get_weatherget_weather(...)]` and it shipped to the rendered page unnoticed
because verification stopped at "the data exists", never "the output is right".

DIS-2434: the page is now rendered by the JS view from the JSON model, so the former
HTML-regex guards here (grid↔compare-bar keys, versions-latest-first, no doubled
names, chart-vs-list exclusivity, v2 registry cross-check) moved to structural
assertions on the model (test_model.py) and DOM smokes (test_browser_popup_compare.py)
— stronger and less brittle. What remains is the one guard with no HTML at all: folding
a higher Dynamo version reproduces that version's docs exactly (the fold contract the
rendered stream cells depend on).
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

UTILS = Path(__file__).resolve().parents[1]
SRC = UTILS / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from resolve_stream_fixtures import resolve, version_key  # noqa: E402
from fixture_snapshot import fixture_snapshot_root  # noqa: E402
import fixtures  # noqa: E402
from case_variants import visible_null_groups  # noqa: E402

FIXTURES_ROOT = fixture_snapshot_root()
STREAM_SRC = FIXTURES_ROOT / "toolcalling" / "fixtures-stream-v1"


def test_visible_null_issue_groups_follow_display_grouping():
    assert visible_null_groups({"7-4.mixed_labels"}, "glm47") == {"7-4", "7-5"}
    assert visible_null_groups({"7-4.mixed_grep"}, "minimax_m3") == {"7-4", "7-5"}
    assert visible_null_groups({"7-4"}, "qwen3") == {"7-4"}
    assert visible_null_groups({"7-7"}, "glm47") == {"7-7"}


def test_stream_regression_cases_share_their_parent_case_bands():
    assert fixtures._subcase_group_key("streamv1", "7.g") == "args"
    assert fixtures._subcase_group_key("streamv1", "7.h") == "args"
    for case_id in ("7.i", "7.j", "7.k", "7.l"):
        assert fixtures._subcase_group_key("streamv1", case_id) == "args"
    assert fixtures._subcase_group_key("streamv1", "51.a") == "reasoning_projection"
    assert fixtures._subcase_group_key("streamv1", "51.b") == "reasoning_projection"
    assert "50.a" not in fixtures._discover_sub_cases("streamv1", {("deepseek_v4", "50.a"): {}, ("deepseek_v4", "51.a"): {}})

pytestmark = pytest.mark.skipif(
    not STREAM_SRC.is_dir(), reason="conformance fixtures not downloaded"
)


def _dynamo_version_dirs() -> list[tuple[str, Path]]:
    out = []
    for d in STREAM_SRC.iterdir():
        if d.is_dir() and d.name.startswith("dynamo_v2-"):
            out.append((d.name.split("-", 1)[1], d))
    out.sort(key=lambda t: version_key(t[0]))
    return out


def test_stream_regression_inputs_reference_prs_without_null_output_cases():
    expected = {
        "TOOLCALLING.streamv1.7.g": (
            {"glm47", "qwen3_coder", "minimax_m2", "minimax_m3"},
            "https://github.com/ai-dynamo/frontend-crates/pull/248",
        ),
        "TOOLCALLING.streamv1.7.h": (
            {"deepseek_v4", "glm47", "gemma4", "harmony", "kimi_k2", "kimi_k3", "minimax_m3", "muse_glimmer"},
            "https://github.com/ai-dynamo/frontend-crates/pull/247",
        ),
        "TOOLCALLING.streamv1.51.a": (
            {"deepseek_v4", "kimi_k3", "muse_glimmer"},
            "https://github.com/ai-dynamo/frontend-crates/pull/253",
        ),
        "TOOLCALLING.streamv1.7.i": (
            {"glm47"},
            "https://github.com/ai-dynamo/frontend-crates/pull/249",
        ),
        "TOOLCALLING.streamv1.7.j": (
            {"minimax_m3"},
            "https://github.com/ai-dynamo/frontend-crates/pull/270",
        ),
        "TOOLCALLING.streamv1.7.k": (
            {"glm47"},
            "https://github.com/ai-dynamo/frontend-crates/pull/271",
        ),
        "TOOLCALLING.streamv1.7.l": (
            {"minimax_m3"},
            "https://github.com/ai-dynamo/frontend-crates/pull/273",
        ),
        "TOOLCALLING.streamv1.51.b": (
            {"deepseek_v4"},
            "https://github.com/ai-dynamo/frontend-crates/pull/255",
        ),
    }
    found = {case_id: {} for case_id in expected}
    for path in (STREAM_SRC / "inputs").glob("*/*.yaml"):
        document = yaml.safe_load(path.read_text()) or {}
        for case_id in expected:
            case = (document.get("cases") or {}).get(case_id)
            if case is not None:
                found[case_id][document["family"]] = case

    for case_id, (families, reference) in expected.items():
        assert set(found[case_id]) == families
        for case in found[case_id].values():
            assert case["ref"] == reference
            assert all("null" not in chunk.get("delta_text", "") for chunk in case["chunks"]), case_id

    nested_union = found["TOOLCALLING.streamv1.7.j"]["minimax_m3"]
    pagination = nested_union["tools"][0]["parameters"]["properties"]["pagination"]
    assert pagination["anyOf"][-1] == {"type": "null"}
    assert nested_union["tools"][0]["strict"] is True
    assert "null" not in "".join(chunk["delta_text"] for chunk in nested_union["chunks"])

    scalar_case = found["TOOLCALLING.streamv1.7.g"]["glm47"]
    scalar_properties = scalar_case["tools"][0]["parameters"]["properties"]
    assert scalar_properties == {
        "count": {"anyOf": [{"type": "integer"}]},
        "ratio": {"type": ["number"]},
        "enabled": {"type": ["boolean"]},
    }
    string_case = found["TOOLCALLING.streamv1.7.h"]["glm47"]
    string_properties = string_case["tools"][0]["parameters"]["properties"]
    assert set(string_properties) == {"spaced", "blank", "empty"}
    assert all(value["type"] == "string" for value in string_properties.values())
    harmony_chunks = found["TOOLCALLING.streamv1.7.h"]["harmony"]["chunks"]
    assert all("delta_token_ids" in chunk for chunk in harmony_chunks if chunk.get("delta_text"))


@pytest.mark.parametrize(("case_id", "families", "arguments"), [
    ("TOOLCALLING.streamv1.7.g", {"glm47", "qwen3_coder", "minimax_m2", "minimax_m3"},
     {"count": 42, "ratio": 1.25, "enabled": False}),
    ("TOOLCALLING.streamv1.7.h", {"deepseek_v4", "glm47", "gemma4", "harmony", "kimi_k2", "kimi_k3", "minimax_m3", "muse_glimmer"},
     {"spaced": "  café\n", "blank": "\t\r\n ", "empty": ""}),
    ("TOOLCALLING.streamv1.51.a", {"deepseek_v4", "kimi_k3", "muse_glimmer"},
     {"location": "Paris"}),
])
def test_retained_stream_regression_captures_preserve_semantics(case_id, families, arguments):
    versions = _dynamo_version_dirs()
    assert versions, "no retained Dynamo stream captures are available"
    captures_found = 0
    for version, root in versions:
        found = {}
        for path in root.glob("*/*.yaml"):
            document = yaml.safe_load(path.read_text())
            case = document["cases"].get(case_id)
            if case is None:
                continue
            complete = [
                event
                for chunk in case["chunks"]
                for event in chunk.get("expected", [])
                if event.get("complete") and "arguments" in event
            ]
            assert len(complete) == 1, (path, case_id)
            captured_arguments = json.loads(complete[0]["arguments"])
            assert captured_arguments == arguments, (path, case_id)
            found[path.parent.name] = "".join(chunk.get("normal_text", "") for chunk in case["chunks"])
        if (version, root) == versions[-1]:
            assert families <= set(found), (version, case_id, families - set(found))
        if not found:
            continue
        captures_found += 1
        assert set(found) <= families
        if case_id == "TOOLCALLING.streamv1.51.a":
            expected_normal_text = {
                "deepseek_v4": "before<think>reason</think>after",
                "kimi_k3": "reasonafter",
                "muse_glimmer": "reasonafter",
            }
            assert found == {family: expected_normal_text[family] for family in found}
    assert captures_found > 0


# --- folding a higher dynamo version reproduces that version's docs exactly ----
def test_fold_reproduces_each_dynamo_version_exactly():
    versions = _dynamo_version_dirs()
    if len(versions) < 2:
        pytest.skip("needs at least two dynamo_v2 version dirs")
    top_ver, top_dir = versions[-1]
    with tempfile.TemporaryDirectory() as tmp:
        resolve(STREAM_SRC, tmp, select=[f"dynamo_v2-{top_ver}"])
        for vfp in top_dir.glob("*/*.yaml"):
            folded_fp = Path(tmp) / vfp.parent.name / vfp.name
            if not folded_fp.exists():
                continue
            want_doc = yaml.safe_load(vfp.read_text()) or {}
            got_doc = yaml.safe_load(folded_fp.read_text()) or {}
            for cid, want_case in (want_doc.get("cases") or {}).items():
                got_case = (got_doc.get("cases") or {}).get(cid)
                if got_case is None or "unavailable" in want_case:
                    continue
                got = [
                    (i, d)
                    for i, ch in enumerate(got_case.get("chunks") or [])
                    for d in ((ch.get("expected") or {}).get("dynamo_v2") or [])
                ]
                want = [
                    (i, d)
                    for i, ch in enumerate(want_case.get("chunks") or [])
                    for d in (ch.get("expected") or [])
                ]
                assert got == want, (
                    f"{vfp.parent.name}/{vfp.name} {cid}: folded dynamo_v2 deltas "
                    f"differ from the {top_ver} doc (lower-version residue?)\n"
                    f"  got:  {got}\n  want: {want}"
                )
