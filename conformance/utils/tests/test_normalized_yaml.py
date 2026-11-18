# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import copy

import yaml

import pytest

import normalized_yaml as normalized
import unified_history as history
import capture_sources
from test_unified_history import _store, _write_capture, _change
from viewer_codec import typed_equal


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
