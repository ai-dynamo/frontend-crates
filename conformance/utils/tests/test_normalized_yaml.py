# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import copy
import hashlib
import json
from pathlib import Path
import shutil

import yaml

import pytest

import normalized_yaml as normalized
import normalize_current
import unified_history as history
import capture_sources
from test_unified_history import _store, _write_capture, _change
from viewer_codec import encode_typed, typed_equal


def fixture():
    request = {"input": "literal $ref ‹finish›", "init": {"starting_state": "None"},
               "finish_reason": "stop", "tools": [{"name": "f", "parameters": {"$ref": "#/$defs/Literal"}}],
               "chunks": [{"delta_text": "literal $ref "}, {"delta_text": ""}, {"delta_text": "‹finish›"}]}
    family = {"input_document": {"family": "sample"}, "golden_document": {"family": "sample"}, "cases": {
        "one": {"lifecycle": "active", "scenario": "one", "description": "one", "policy": [], "historical_ids": [],
                "display_id": "UNIFIED.one", "request": request, "golden": {"assembled": []}}}}
    origin = {"crate_version": "0.1.0", "source_sha256": "a" * 64}
    value = {"case_key": "UNIFIED.one", "document": {"captured_with": {"dynamo_v2": "0.1.0"}, "capture_origin": origin,
             "mode": "unified", "record_metadata": {"typed": {1: "integer", "1": "string", None: False}, "nullable": None}},
             "stimulus": {"inline": request, "semantic_sha256": normalized.request_digest(request)},
             "observation": {"value": {"assembled": [], "chunks": [{"expected": []}]}}}
    snapshot = {"schema_version": 2, "provenance": {"status": "captured", "origin": origin}, "document": {"mode": "unified"},
                "records": {"one": {"member": "sample/0.1.0.yaml", "value": value}}}
    return family, snapshot


def test_roundtrip_preserves_typed_metadata_literal_refs_and_exact_delivery():
    family, snapshot = fixture()
    encoded_family = normalized.encode_family(family)
    encoded_snapshot = normalized.encode_snapshot(snapshot, encoded_family)
    assert typed_equal(normalized.expand_family(encoded_family), family)
    expanded = normalized.expand_snapshot(encoded_snapshot, encoded_family)
    assert typed_equal(expanded, snapshot)
    expanded["records"]["one"]["value"]["stimulus"]["inline"]["tools"].clear()
    assert typed_equal(normalized.expand_snapshot(encoded_snapshot, encoded_family), snapshot)
    assert "golden_document" not in encoded_family
    assert "capture_origin" not in encoded_snapshot["records"]["one"]


def recapture_fixture(value=42):
    family, snapshot = fixture()
    result = {"capture_input": copy.deepcopy(family["cases"]["one"]["request"]),
              "assembled": [], "chunks": [{"expected": []}, {"expected": [], "literal": {"result_ref": "literal", "$ref": "literal"}}],
              "error_detail": {"value": value}}
    def receipt(source):
        return {"result": copy.deepcopy(result), "provenance": {"source_sha256": source * 64},
                "resolution": "fresh_execution"}
    snapshot["records"]["one"]["value"]["document"]["record_metadata"].update(
        verified_recapture=receipt("a"), verified_recaptures=[receipt("b"), receipt("c")])
    return family, snapshot


def test_identical_complete_results_share_content_without_merging_receipts():
    family, snapshot = recapture_fixture()
    authored = normalized.encode_family(family)
    compact = normalized.encode_snapshot(snapshot, authored)
    metadata = next(iter(compact["contexts"].values()))["record_metadata"]
    active = metadata["verified_recapture"]
    digest = hashlib.sha256(json.dumps(active["result"], ensure_ascii=False, sort_keys=True,
                                      separators=(",", ":")).encode()).hexdigest()
    assert [r["result_ref"] for r in metadata["verified_recaptures"]] == [digest, digest]
    assert all("result" not in r for r in metadata["verified_recaptures"])
    expanded = normalized.expand_snapshot(compact, authored)
    assert typed_equal(expanded, snapshot)
    restored = expanded["records"]["one"]["value"]["document"]["record_metadata"]
    assert encode_typed(restored["verified_recaptures"]) == encode_typed(
        snapshot["records"]["one"]["value"]["document"]["record_metadata"]["verified_recaptures"])
    restored["verified_recaptures"][0]["result"]["chunks"].clear()
    assert len(restored["verified_recapture"]["result"]["chunks"]) == 2
    assert len(restored["verified_recaptures"][1]["result"]["chunks"]) == 2
    assert typed_equal(normalized.expand_snapshot(compact, authored), snapshot)
    assert typed_equal(normalized.encode_snapshot(normalized.expand_snapshot(compact, authored), authored), compact)


def test_unused_context_cannot_hide_an_invalid_result_reference():
    family, snapshot = fixture()
    authored = normalized.encode_family(family)
    compact = normalized.encode_snapshot(snapshot, authored)
    compact["contexts"]["unused"] = {"record_metadata": {"verified_recaptures": [{"result_ref": "bad"}]}}
    with pytest.raises(ValueError):
        normalized.expand_snapshot(compact, authored)


@pytest.mark.parametrize("compact", [False, True])
def test_canonical_check_accepts_existing_inline_or_shared_results(tmp_path, monkeypatch, compact):
    source = Path(__file__).resolve().parents[2] / "fixtures-unified-v2/families/nemotron_v3"
    destination = tmp_path / "conformance/fixtures-unified-v2/families/nemotron_v3"
    destination.mkdir(parents=True)
    for name in ("inputs_and_golden.yaml", "dynamo_v2.yaml"):
        shutil.copyfile(source / name, destination / name)
    with monkeypatch.context() as patch:
        if not compact:
            patch.setattr(normalized, "_compact_result_refs", lambda context: None)
        normalize_current.normalize(tmp_path)
    before = {path.name: path.read_bytes() for path in destination.iterdir() if path.is_file()}
    normalize_current.normalize(tmp_path, check=True)
    assert before == {path.name: path.read_bytes() for path in destination.iterdir() if path.is_file()}


def test_shared_reader_fixture_and_archive_only_legacy_payloads():
    shared = yaml.safe_load((Path(__file__).parent / "fixtures/normalized-results.yaml").read_text())
    assert typed_equal(normalized.expand_snapshot(shared["snapshot"], shared["authored"]), shared["expected_snapshot"])
    family, snapshot = recapture_fixture()
    metadata = snapshot["records"]["one"]["value"]["document"]["record_metadata"]
    del metadata["verified_recapture"]
    authored = normalized.encode_family(family)
    compact = normalized.encode_snapshot(snapshot, authored)
    assert typed_equal(normalized.expand_snapshot(compact, authored), snapshot)


@pytest.mark.parametrize("change", ["request", "tools", "chunks", "error", "type", "mapping_order"])
def test_equal_final_output_does_not_merge_different_complete_results(change):
    family, snapshot = recapture_fixture(1)
    metadata = snapshot["records"]["one"]["value"]["document"]["record_metadata"]
    other = metadata["verified_recaptures"][0]["result"]
    if change == "request":
        other["capture_input"]["input"] += "different"
    elif change == "tools":
        other["capture_input"]["tools"] = []
    elif change == "chunks":
        other["chunks"].reverse()
    elif change == "error":
        other["error_detail"]["message"] = "different error"
    elif change == "type":
        other["error_detail"]["value"] = True
    else:
        other["error_detail"] = {"value": 1, "first": "before"}
        metadata["verified_recapture"]["result"]["error_detail"] = {"first": "before", "value": 1}
    authored = normalized.encode_family(family)
    compact = normalized.encode_snapshot(snapshot, authored)
    encoded = next(iter(compact["contexts"].values()))["record_metadata"]
    assert "result_ref" not in encoded["verified_recaptures"][0]
    assert typed_equal(normalized.expand_snapshot(compact, authored), snapshot)


@pytest.mark.parametrize("value", [1.0, 1000000000000000.25, float("nan"), float("inf"), float("-inf"), b"bytes", {1: "typed key"}])
def test_nonportable_payload_types_stay_inline(value):
    family, snapshot = recapture_fixture(value)
    authored = normalized.encode_family(family)
    compact = normalized.encode_snapshot(snapshot, authored)
    metadata = next(iter(compact["contexts"].values()))["record_metadata"]
    assert all("result_ref" not in r for r in metadata["verified_recaptures"])
    assert typed_equal(normalized.expand_snapshot(compact, authored), snapshot)


@pytest.mark.parametrize("mutation", ["digest", "changed_payload", "missing_active", "active_ref", "both", "float", "nonstring_key"])
def test_invalid_archived_result_references_fail_closed(mutation):
    family, snapshot = recapture_fixture()
    authored = normalized.encode_family(family)
    compact = normalized.encode_snapshot(snapshot, authored)
    metadata = next(iter(compact["contexts"].values()))["record_metadata"]
    active = metadata["verified_recapture"]
    archived = metadata["verified_recaptures"][0]
    if mutation == "digest":
        archived["result_ref"] = "0" * 64
    elif mutation == "changed_payload":
        active["result"]["chunks"].reverse()
    elif mutation == "missing_active":
        del metadata["verified_recapture"]
    elif mutation == "active_ref":
        active["result_ref"] = archived["result_ref"]
    elif mutation == "both":
        archived["result"] = copy.deepcopy(active["result"])
    else:
        active["result"]["error_detail"]["value"] = 1.0 if mutation == "float" else {1: "typed key"}
        archived["result_ref"] = hashlib.sha256(json.dumps(active["result"], ensure_ascii=False, sort_keys=True,
                                                        separators=(",", ":")).encode()).hexdigest()
    with pytest.raises(ValueError):
        normalized.expand_snapshot(compact, authored)


def test_frozen_request_remains_bound_after_current_authored_request_changes():
    family, snapshot = fixture()
    old = copy.deepcopy(family["cases"]["one"]["request"])
    family["cases"]["one"]["request"]["input"] = "new request"
    family["cases"]["one"]["request"]["chunks"] = [{"delta_text": "new request"}]
    encoded = normalized.encode_family(family, [old])
    captured = normalized.encode_snapshot(snapshot, encoded)
    assert len(encoded["requests"]) == 2
    assert typed_equal(normalized.expand_snapshot(captured, encoded), snapshot)


def test_producer_exceptions_preserve_missing_and_null_without_deep_merge():
    family, snapshot = fixture()
    other = copy.deepcopy(snapshot["records"]["one"])
    other["value"]["document"] = {"captured_with": {"dynamo_v2": "0.0.9"}, "capture_origin": None}
    other["suppress_inherited_from"] = True
    snapshot["records"]["other"] = other
    encoded = normalized.encode_family(family)
    compact = normalized.encode_snapshot(snapshot, encoded)
    assert len(compact["producers"]) == 2
    assert typed_equal(normalized.expand_snapshot(compact, encoded), snapshot)


@pytest.mark.parametrize("mutation", [
    lambda f, s: f["request_defaults"].update(finish_reason="length"),
    lambda f, s: f["tools"].clear(),
    lambda f, s: f["requests"].clear(),
    lambda f, s: s["origins"].clear(),
    lambda f, s: s["producers"].clear(),
    lambda f, s: s["contexts"].clear(),
    lambda f, s: s["record_defaults"].pop("producer"),
    lambda f, s: s["records"]["one"].update(context=None),
    lambda f, s: s["records"]["one"].update(error={"code": "extra outcome"}),
    lambda f, s: s["records"]["one"].update(stimulus={"ref": "current"}),
    lambda f, s: s["records"]["one"].update(unknown=True),
])
def test_malformed_definitions_and_ambiguous_states_fail_closed(mutation):
    family, snapshot = fixture()
    family = normalized.encode_family(family)
    snapshot = normalized.encode_snapshot(snapshot, family)
    mutation(family, snapshot)
    with pytest.raises(ValueError):
        normalized.expand_snapshot(snapshot, family)


@pytest.mark.parametrize("delta", [42, 42.0, True, None, "42", "on", "-0", "4.2e1"])
def test_chunk_shorthand_never_changes_non_string_scalar_types(delta):
    family, _snapshot = fixture()
    family["cases"]["one"]["request"]["chunks"] = [{"delta_text": delta}]
    compact = normalized.encode_family(family)
    expanded = normalized.expand_family(compact)
    assert typed_equal(expanded, family)


def test_selected_materialization_resolves_its_supplied_frozen_library(tmp_path):
    root = _store(tmp_path / "store")
    _write_capture(root, "0.5.2", {"text_only": _change()})
    store = history.load_store(root)
    family = store.families["gemma4"]
    member = "conformance/fixtures-unified-v2/families/gemma4/dynamo_v2-0.5.2.yaml"
    snapshot = store.histories["gemma4", "dynamo_v2"].complete_snapshot("dynamo_v2-0.5.2", source_member=member)
    snapshot = capture_sources.independent_snapshot(snapshot, family.document)
    authored = normalized.encode_family(family.document)
    encoded = normalized.encode_snapshot(snapshot, authored)
    input_path = "families/gemma4/inputs_and_golden.yaml"
    capture_path = "families/gemma4/dynamo_v2-0.5.2.yaml"
    resources = [{**history.source_identity(path), "path": "conformance/fixtures-unified-v2/" + path}
                 for path in [input_path, capture_path]]
    selected = history.selected_store(root, resources, {"dynamo_v2": "0.5.2"}, source_data={
        input_path: history.dump_yaml(authored).encode(), capture_path: history.dump_yaml(encoded).encode()})
    actual = selected.histories["gemma4", "dynamo_v2"].resolve("dynamo_v2-0.5.2")
    assert typed_equal(actual, {key: wrapped["value"] for key, wrapped in snapshot["records"].items()})


def test_authored_writer_retains_frozen_requests_when_case_changes(tmp_path):
    family, _snapshot = fixture()
    encoded = normalized.encode_family(family)
    path = tmp_path / "inputs_and_golden.yaml"
    path.write_text(history.dump_yaml(encoded))
    current = normalized.expand_family(encoded)
    current["cases"]["one"]["request"]["input"] = "new"
    current["cases"]["one"]["request"]["chunks"] = [{"delta_text": "new"}]
    _, text = history._render_document((path, current))
    saved = yaml.safe_load(text)
    assert saved["schema_version"] == 3
    assert len(saved["requests"]) == 2
    assert typed_equal(normalized.expand_family(saved), current)
