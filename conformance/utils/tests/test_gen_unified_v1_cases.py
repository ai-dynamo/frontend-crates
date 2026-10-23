# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pytest
import yaml

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import gen_unified_golden
import gen_unified_v1_cases
import unified_taxonomy
import unified_history
import unified_archives
from viewer_codec import typed_equal
from conformance.utils.tests.test_execution_manifests import compact_store
from conformance.utils.tests.corpus_semantics import assert_unique_behaviour


ROOT = Path(__file__).resolve().parents[3]


def test_compact_merge_preserves_parent_receipts_and_unrelated_records(tmp_path):
    store, first, second, records = compact_store(tmp_path)
    family_path = second.parent / unified_history.FAMILY_DOCUMENT_NAME
    family = unified_history._validate_family(family_path, unified_history.load_yaml(family_path))
    _, _, _, expanded = unified_history._validate_capture_file(second, unified_history.load_yaml(second), {family.name: family})
    original = second.read_text()
    unchanged = {field: expanded[field] for field in ("changes", "metadata_changes", "document_overrides")}
    assert gen_unified_v1_cases._merge_generated_case(store, second.relative_to(store), unified_history.dump_yaml(unchanged)) == original
    addition = {"changes": {}, "metadata_changes": {"text_only": {**expanded["metadata_changes"]["text_only"], "new": None}}, "document_overrides": {}}
    merged = gen_unified_v1_cases._merge_generated_case(store, second.relative_to(store), unified_history.dump_yaml(addition))
    document = unified_history.load_yaml(second)
    assert set(document) == {"parent", "execution"}
    second.write_text(merged)
    _, _, _, restored = unified_history._validate_capture_file(second, unified_history.load_yaml(second), {family.name: family})
    assert restored["parent"] == document["parent"]
    assert typed_equal(restored["metadata_changes"]["text_only"], addition["metadata_changes"]["text_only"])
    assert typed_equal(restored["document_overrides"], expanded["document_overrides"])
    after = dict(unified_history.load_store(store).histories[("gemma4", "dynamo_v2")].iter_resolved_snapshots())
    assert typed_equal(after[first.stem], records[first.stem])
    expected = records[second.stem]
    expected["text_only"]["document"]["record_metadata"]["new"] = None
    assert typed_equal(after[second.stem], expected)
    unified_history.compact_execution_store(store)
    assert typed_equal(dict(unified_history.load_store(store).histories[("gemma4", "dynamo_v2")].iter_resolved_snapshots()), after)


def _resolved_retained_regression(path, document, family):
    key = gen_unified_v1_cases.CASE_KEY
    if document.get("schema_version") == 2:
        identity = unified_history.CAPTURE_DIRECTORY_RE.fullmatch(path.stem)
        assert identity is not None
        unified_history._validate_complete_snapshot(
            path, document, family, identity["implementation"], identity["runtime_version"],
        )
        return document, document["records"][key]["value"]
    _, implementation, capture_id, capture = unified_history._validate_capture_file(
        path, document, {family.name: family},
    )
    focused = {**capture, "parent": None, **{field: {key: capture[field][key]} if key in capture[field] else {}
                             for field in ("changes", "metadata_changes", "document_overrides")}}
    history = unified_history.History(family, implementation, {capture_id: focused}, path.parent, {capture_id: path})
    return capture, history.resolve(capture_id)[key]


def _retained_regression_record(capture):
    key = gen_unified_v1_cases.CASE_KEY
    if capture.get("schema_version") == 2:
        return capture["records"][key]["value"]
    return capture["changes"][key]


def _retained_regression_provenance(capture, record):
    if capture.get("schema_version") == 2:
        return {key: record["document"].get(key) for key in (
            "captured_with", "capture_origin", "inherited_from",
        )}
    return capture["document_overrides"][gen_unified_v1_cases.CASE_KEY]["capture_provenance"]


def test_retained_archived_split_regressions_preserve_complete_records():
    archives = {f"dynamo-{version}.tar.gz" for _, version, _ in gen_unified_v1_cases._capture_archives(ROOT)}
    with unified_archives.expanded(ROOT / "conformance/fixtures-unified-v2", archive_names=archives) as source:
        _assert_retained_archived_split_regressions(source)


def _assert_retained_archived_split_regressions(source):
    documents = gen_unified_v1_cases._generated_documents(ROOT, source)
    expected_families = set(gen_unified_golden.V1_SPLIT_FAMILIES)
    registered_families = set(gen_unified_golden.yaml.safe_load(
        gen_unified_golden.markers.parser_families_path().read_text()
    )["families"])
    input_documents = {}
    capture_documents = {}
    for path, content in documents.items():
        family = path.parent.name
        document = yaml.safe_load(content)
        if path.name == "inputs_and_golden.yaml":
            input_documents[family] = document
        else:
            capture_documents[family] = document

    assert set(input_documents) == expected_families
    assert set(capture_documents) == expected_families
    assert expected_families == registered_families - {"muse_glimmer"} | {"gpt_oss"}
    assert len(expected_families) == 24
    for family in sorted(expected_families):
        input_case = input_documents[family]["cases"]["v1_split_parallel_tool_calls"]
        capture = capture_documents[family]
        capture_path = next(path for path in documents if path.parent.name == family and path.name.startswith("dynamo_v1-"))
        version = capture_path.stem.removeprefix("dynamo_v1-")
        input_path = source / "families" / family / "inputs_and_golden.yaml"
        capture_path = source / "families" / family / capture_path.name
        current_inputs = yaml.safe_load(input_path.read_text())
        assert current_inputs == input_documents[family]
        family_record = unified_history._validate_family(input_path, current_inputs)
        capture, expected_record = _resolved_retained_regression(capture_path, capture, family_record)
        assert input_case["scenario"] == "v1_split_parallel_tool_calls"
        assert input_case["display_id"] == unified_taxonomy.numbered_id(input_case["scenario"])
        assert len(input_case["golden"]["assembled"]) == 2
        assert capture["document"]["mode"] == "unified"
        record = _retained_regression_record(capture)
        assert record["case_key"] == input_case["display_id"]
        assert record["observation"]["value"] == input_case["golden"]
        record_provenance = _retained_regression_provenance(capture, record)
        assert record_provenance == {
            "captured_with": {"dynamo_v1": version},
            "capture_origin": None,
            "inherited_from": None,
        }
        current_capture = yaml.safe_load(capture_path.read_text())
        _current, actual_record = _resolved_retained_regression(capture_path, current_capture, family_record)
        assert actual_record == expected_record
        source_case = input_case["policy"][0].removeprefix("input: ")
        inventory = assert_unique_behaviour(family, input_documents[family]["cases"], retained_archives={
            gen_unified_v1_cases.CASE_KEY: {
                "purpose": "Retain the exact archived request and version-bound result alongside the shared two-call scenario",
                "input_source": f"{gen_unified_v1_cases.INPUT_ARCHIVE}:{source_case}",
                "capture_source": str(Path("conformance/fixtures-unified-v2") / capture_path.relative_to(source)),
                "version": version,
            },
        })
        assert set(inventory["retained_archive_exceptions"]) == {gen_unified_v1_cases.CASE_KEY}



def test_incomplete_archived_input_reports_its_selected_legacy_case(monkeypatch):
    monkeypatch.setattr(gen_unified_v1_cases, "_capture_archives", lambda _root: [((1,), "1.0.0", Path("archive.tar.gz"))])
    monkeypatch.setattr(gen_unified_v1_cases, "_captured_call_map", lambda _archives, _families: {("sample", "1.0.0", "UNIFIED.1"): {}})
    monkeypatch.setattr(gen_unified_v1_cases, "_input_cases", lambda _root, _family: {"UNIFIED.1": {"model_text": None, "tools": []}})
    monkeypatch.setattr(gen_unified_v1_cases.gen_unified_golden, "V1_SPLIT_FAMILIES", frozenset({"sample"}))
    monkeypatch.setattr(gen_unified_v1_cases, "FAMILY_ALIASES", {})

    with pytest.raises(ValueError, match=r"sample/UNIFIED\.1 has an incomplete archived input"):
        gen_unified_v1_cases.generated_documents(Path("fixture-root"))
