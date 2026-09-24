# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retained evidence is invariant under checkpoint migration and removal."""

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import legacy_checkpoint
import legacy_history
import unified_history
from test_legacy_history import _loose_capture, _store_bytes
from test_unified_history import _change, _store, _write_capture


def test_unified_migration_binds_historical_request(tmp_path):
    root = _store(tmp_path / "store")
    before = unified_history.resolved_inventory(root)
    unified_history.migrate_store(root)
    assert unified_history.resolved_inventory(root) == before
    assert unified_history.migrate_store(root) == []
    ledger = unified_history.load_yaml(root / "families/gemma4/observations.yaml")
    assert len(ledger["observations"]) == 1
    latest = unified_history.load_yaml(root / "families/gemma4/dynamo_v2-0.5.2.yaml")
    assert latest["changes"] == {}
    family = root / "families/gemma4/inputs_and_golden.yaml"
    document = unified_history.load_yaml(family)
    document["cases"]["text_only"]["request"]["input"] = "edited today"
    family.write_text(unified_history.dump_yaml(document))
    assert unified_history.resolved_inventory(root) == before


@pytest.mark.parametrize("versions", [["0.5.0"], ["0.5.2"], ["0.6.0"], ["0.7.0"], ["0.5.0", "0.5.2", "0.6.0"]])
def test_unified_removal_preserves_provenance_absence_and_restored_output(tmp_path, versions):
    root = _store(tmp_path / "store")
    _write_capture(root, "0.6.0", {"text_only": {"absent": True}})
    _write_capture(root, "0.7.0", {"text_only": _change()}, metadata_changes={"text_only": {"note": "restored"}})
    before = unified_history.resolved_inventory(root)
    for version in versions:
        unified_history.remove_capture(root, f"dynamo_v2-{version}")
        before.pop(f"gemma4/dynamo_v2-{version}")
        assert unified_history.resolved_inventory(root) == before
    ledger = unified_history.load_yaml(root / "families/gemma4/observations.yaml")
    assert len(ledger["observations"]) == 1


def test_unified_only_removal_and_new_checkpoint_roundtrip(tmp_path):
    root = _store(tmp_path / "store")
    unified_history.remove_capture(root, "dynamo_v2-0.5.0")
    unified_history.remove_capture(root, "dynamo_v2-0.5.2")
    assert unified_history.resolved_inventory(root) == {}
    assert unified_history.migrate_store(root) == []


@pytest.mark.parametrize("mutation,pattern", [("cycle", "cycle"), ("base", "missing capture base"), ("reference", "dangling"), ("payload", "fingerprint")])
def test_unified_rejects_broken_references(tmp_path, mutation, pattern):
    root = _store(tmp_path / "store")
    unified_history.migrate_store(root)
    path = root / "families/gemma4/dynamo_v2-0.5.0.yaml"
    doc = unified_history.load_yaml(path)
    if mutation == "cycle":
        doc["base"] = "dynamo_v2-0.5.2"
    elif mutation == "base":
        doc["base"] = "dynamo_v2-9.9.9"
    elif mutation == "reference":
        doc["changes"]["text_only"]["observation"] = "missing"
    else:
        path = path.parent / "observations.yaml"
        doc = unified_history.load_yaml(path)
        next(iter(doc["observations"].values()))["stimulus"]["inline"]["input"] = "corrupt"
    path.write_text(unified_history.dump_yaml(doc))
    with pytest.raises(ValueError, match=pattern):
        unified_history.load_store(root)


def _legacy(root, corpus="stream"):
    loose, store = root / "loose", root / "store"
    for version, value in (("1.0.0", "first"), ("1.0.0+current", "qualified"), ("1.1.0", "restored"), ("1.2.0", "restored")):
        _loose_capture(loose, corpus, f"vllm_python-{version}", value)
    legacy_history.update_from_loose(store, loose)
    return store


@pytest.mark.parametrize("versions", [["1.0.0"], ["1.0.0+current"], ["1.1.0"], ["1.2.0"], ["1.0.0", "1.0.0+current", "1.1.0", "1.2.0"]])
@pytest.mark.parametrize("corpus", ["stream", "batch", "reasoning"])
def test_legacy_removals_keep_every_retained_view(tmp_path, versions, corpus):
    root = _legacy(tmp_path, corpus)
    before = legacy_history.resolved_inventory(root)
    legacy_history.migrate_store(root)
    assert legacy_history.resolved_inventory(root) == before
    assert legacy_history.migrate_store(root) == []
    for version in versions:
        legacy_history.remove_capture(root, f"vllm_python-{version}", corpus=corpus)
        before.pop(f"{corpus}/qwen3/vllm_python-{version}")
        assert legacy_history.resolved_inventory(root) == before


def test_legacy_stream_views_and_packaging_stay_separate(tmp_path):
    root = _legacy(tmp_path)
    for version in ("1.1.0", "1.2.0"):
        path = root / f"stream/families/qwen3/vllm_python-{version}.yaml"
        document = unified_history.load_yaml(path)
        document["documents"]["one.yaml"]["cases"] = {}
        path.write_text(unified_history.dump_yaml(document))
    legacy_history.migrate_store(root)
    evidence = legacy_history.resolved_inventory(root)
    first = evidence["stream/qwen3/vllm_python-1.1.0"]["views"]
    key = json.dumps(["one.yaml", "A"], separators=(",", ":"))
    assert first["python"][key]["payload"]["value"]["chunks"][0]["expected"][0]["name"] == "qualified"
    assert first["rust"][key]["payload"]["value"]["chunks"][0]["expected"][0]["name"] == "first"
    output = tmp_path / "output"
    legacy_history.materialize_store(root, output)
    assert json.loads((output / ".reader-views/legacy-checkpoints.json").read_text()) == {"schema_version": 1, "complete_family_snapshots": True}
    rust = output / ".reader-views/rust/toolcalling/fixtures-stream-v1/vllm_python-1.0.0/qwen3/one.yaml"
    assert unified_history.load_yaml(rust)["cases"]["A"]["chunks"][0]["expected"][0]["name"] == "first"
    before = _store_bytes(root)
    assert legacy_history.update_from_loose(root, output) == []
    assert _store_bytes(root) == before


@pytest.mark.parametrize("mutation,pattern", [("cycle", "cycle"), ("base", "missing legacy capture base"), ("reference", "dangling"), ("payload", "fingerprint")])
def test_legacy_rejects_broken_references(tmp_path, mutation, pattern):
    root = _legacy(tmp_path)
    legacy_history.migrate_store(root)
    path = root / "stream/families/qwen3/vllm_python-1.0.0.yaml"
    doc = unified_history.load_yaml(path)
    if mutation == "cycle":
        doc["views"]["python"]["base"] = "vllm_python-1.2.0"
    elif mutation == "base":
        doc["views"]["python"]["base"] = "vllm_python-9.9.9"
    elif mutation == "reference":
        next(iter(doc["views"]["python"]["changes"].values()))["observation"] = "missing"
    else:
        path = path.parent / "observations.yaml"
        doc = unified_history.load_yaml(path)
        next(iter(doc["observations"].values()))["request"]["model_text"] = "corrupt"
    path.write_text(unified_history.dump_yaml(doc))
    with pytest.raises(ValueError, match=pattern):
        legacy_history.resolved_inventory(root)


def test_legacy_unchanged_append_after_removal_keeps_original_producer(tmp_path):
    root = _legacy(tmp_path, "batch")
    legacy_history.migrate_store(root)
    legacy_history.remove_capture(root, "vllm_python-1.1.0")
    before = legacy_history.resolved_inventory(root)
    output = tmp_path / "output"
    legacy_history.materialize_store(root, output)
    _loose_capture(output, "batch", "vllm_python-1.3.0", "restored")
    legacy_history.update_from_loose(root, output)
    after = legacy_history.resolved_inventory(root)
    for identity, snapshot in before.items():
        assert after[identity] == snapshot
    assert after["batch/qwen3/vllm_python-1.3.0"]["views"] == before["batch/qwen3/vllm_python-1.2.0"]["views"]
    checkpoint = unified_history.load_yaml(root / "batch/families/qwen3/vllm_python-1.3.0.yaml")
    assert checkpoint["views"]["python"]["changes"] == {}


def test_legacy_modified_bound_input_fails_load(tmp_path):
    root = _legacy(tmp_path, "batch")
    path = root / "batch/families/qwen3/inputs_and_golden.yaml"
    document = {"corpus": "batch", "family": "qwen3", "capture": "inputs", "documents": {
        "one.yaml": {"family": "qwen3", "cases": {"A": {"model_text": "original"}}},
    }}
    path.write_text(unified_history.dump_yaml(document))
    legacy_history.migrate_store(root)
    document["documents"]["one.yaml"]["cases"]["A"]["model_text"] = "edited"
    path.write_text(unified_history.dump_yaml(document))
    with pytest.raises(ValueError, match="input fingerprint mismatch"):
        legacy_history.resolved_inventory(root)


def _snapshot_store(root):
    records = legacy_history._split_snapshot_records(root, "qwen3", {
        "one.yaml": {"family": "qwen3", "captured_with": {"vllm_python": "1.0.0"},
                     "cases": {"A": {"vllm_python": {"tool_calls": []}}}},
    })
    for path, record in records.items():
        legacy_history._write(path, record)
    source = root / "batch_on_stream/families/qwen3/vllm_python-1.0.0.yaml"
    later = copy.deepcopy(unified_history.load_yaml(source))
    later["capture"], later["provenance"] = legacy_history._snapshot_capture("vllm_python", "1.1.0")
    legacy_history._write(source.with_name("vllm_python-1.1.0.yaml"), later)
    legacy_history.migrate_store(root)
    return root


def test_selected_snapshot_removal_requires_explicit_disposition(tmp_path):
    root = _snapshot_store(tmp_path / "store")
    before = _store_bytes(root)
    with pytest.raises(ValueError, match="requires an equivalent replacement"):
        legacy_history.remove_capture(root, "vllm_python-1.0.0")
    assert _store_bytes(root) == before
    legacy_history.remove_capture(root, "vllm_python-1.0.0", replacement="vllm_python-1.1.0")
    inventory = legacy_history.resolved_inventory(root)
    assert inventory["batch_on_stream/qwen3/inputs"]["record"]["documents"]["one.yaml"]["capture_selection"] == {"vllm_python": "vllm_python-1.1.0"}
    legacy_history.remove_capture(root, "vllm_python-1.1.0", unavailable="capture removed")
    output = tmp_path / "output"
    legacy_history.materialize_store(root, output)
    snapshot = unified_history.load_yaml(output / "toolcalling/fixtures-batch-on-stream-v1/qwen3/one.yaml")
    assert snapshot["cases"]["A"]["vllm_python"] == {"unavailable": "capture removed"}


def test_annotation_only_legacy_row_keeps_prior_measurement(tmp_path):
    root = _legacy(tmp_path, "batch")
    path = root / "batch/families/qwen3/vllm_python-1.2.0.yaml"
    document = unified_history.load_yaml(path)
    document["documents"]["one.yaml"]["cases"] = {"A": {"note": "explanation only"}}
    path.write_text(unified_history.dump_yaml(document))
    legacy_history.migrate_store(root)
    output = tmp_path / "output"
    legacy_history.materialize_store(root, output)
    case = unified_history.load_yaml(output / "toolcalling/fixtures-batch-v1/vllm_python-1.2.0/qwen3/one.yaml")["cases"]["A"]
    assert case["expected"]["vllm_python"]["tool_calls"][0]["name"] == "restored"
    assert case["note"] == "explanation only"


def test_legacy_writer_does_not_emit_aliases(tmp_path):
    shared = [{"argument": "value"}]
    path = tmp_path / "observations.yaml"
    document = {"first": shared, "second": shared}
    legacy_history._write(path, document)
    assert unified_history.load_yaml(path) == document


def test_unified_can_capture_again_after_removing_only_history(tmp_path):
    root = _store(tmp_path / "store")
    output = tmp_path / "output"
    unified_history.materialize_store(root, output)
    unified_history.remove_capture(root, "dynamo_v2-0.5.0")
    unified_history.remove_capture(root, "dynamo_v2-0.5.2")
    unified_history.update_from_loose(root, output, excluded_capture_dirs={"dynamo_v2-0.5.2"})
    assert "gemma4/dynamo_v2-0.5.0" in unified_history.resolved_inventory(root)


def test_native_stream_partial_chunks_and_unavailability_preserve_tail(tmp_path):
    root = tmp_path / "store"
    versions = {
        "1.0.0": {"chunks": [{"expected": [{"index": 0, "name": "first"}], "normal_text": "old"}, {"expected": [{"index": 0, "name": "tail"}], "normal_text": "tail text"}]},
        "2.0.0": {"chunks": [{"expected": [{"index": 0, "name": "second"}]}]},
        "3.0.0": {"unavailable": "not supported"},
        "4.0.0": {"chunks": [{}]},
        "5.0.0": {"chunks": []},
    }
    for version, case in versions.items():
        legacy_history._write(root / f"stream/families/qwen3/dynamo_v2-{version}.yaml", {
            "corpus": "stream", "family": "qwen3", "capture": f"dynamo_v2-{version}",
            "documents": {"one.yaml": {"family": "qwen3", "cases": {"A": case}}},
        })
    before = legacy_history.resolved_inventory(root)
    key = json.dumps(["one.yaml", "A"], separators=(",", ":"))
    def value(version):
        return before[f"stream/qwen3/dynamo_v2-{version}"]["views"]["rust"][key]["payload"]["value"]
    assert value("2.0.0")["chunks"][1] == versions["1.0.0"]["chunks"][1]
    assert "normal_text" not in value("2.0.0")["chunks"][0]
    assert value("3.0.0")["chunks"] == value("2.0.0")["chunks"]
    assert value("3.0.0")["unavailable"] == "not supported"
    assert "unavailable" not in value("4.0.0")
    assert value("4.0.0")["chunks"][0] == {"expected": []}
    assert value("5.0.0")["chunks"] == value("4.0.0")["chunks"]
    legacy_history.migrate_store(root)
    for version in ("1.0.0", "2.0.0", "3.0.0", "4.0.0"):
        legacy_history.remove_capture(root, f"dynamo_v2-{version}")
        before.pop(f"stream/qwen3/dynamo_v2-{version}")
        assert legacy_history.resolved_inventory(root) == before
    producer = before["stream/qwen3/dynamo_v2-5.0.0"]["views"]["rust"][key]["producer"]
    assert "dynamo_v2-1.0.0" in producer["contributors"]


def test_new_release_inherits_native_tail_after_packaging(tmp_path):
    root = _legacy(tmp_path)
    for version in ("1.1.0", "1.2.0"):
        path = root / f"stream/families/qwen3/vllm_python-{version}.yaml"
        document = unified_history.load_yaml(path)
        document["documents"]["one.yaml"]["cases"] = {}
        path.write_text(unified_history.dump_yaml(document))
    legacy_history.migrate_store(root)
    output = tmp_path / "output"
    legacy_history.materialize_store(root, output)
    legacy_history._write(output / "toolcalling/fixtures-stream-v1/vllm_python-1.3.0/qwen3/one.yaml", {
        "family": "qwen3", "cases": {"A": {"chunks": []}},
    })
    before = legacy_history.resolved_inventory(root)
    legacy_history.update_from_loose(root, output)
    after = legacy_history.resolved_inventory(root)
    for identity, snapshot in before.items():
        assert after[identity] == snapshot
    key = json.dumps(["one.yaml", "A"], separators=(",", ":"))
    newest = after["stream/qwen3/vllm_python-1.3.0"]["views"]
    assert newest["python"][key]["payload"]["value"]["chunks"] == []
    assert newest["rust"][key]["payload"]["value"]["chunks"][0]["expected"][0]["name"] == "first"


def test_snapshot_replacement_with_different_annotations_is_rejected(tmp_path):
    root = _snapshot_store(tmp_path / "store")
    inventory = legacy_history.resolved_inventory(root)
    target = inventory["batch_on_stream/qwen3/vllm_python-1.1.0"]["views"]["python"]
    next(iter(target.values()))["annotations"]["note"] = "different interpretation"
    legacy_history._publish_compact(root, inventory)
    before = _store_bytes(root)
    with pytest.raises(ValueError, match="not equivalent"):
        legacy_history.remove_capture(root, "vllm_python-1.0.0", replacement="vllm_python-1.1.0")
    assert _store_bytes(root) == before


def test_reasoning_checkpoint_keeps_anchor_input_measurements(tmp_path):
    root = tmp_path / "store"
    family = root / "reasoning/families/qwen3"
    legacy_history._write(family / "inputs_and_golden.yaml", {
        "corpus": "reasoning", "family": "qwen3", "capture": "inputs", "documents": {
            "one.yaml": {"family": "qwen3", "captured_with": {"vllm_python": "1.0.0"}, "cases": {
                "A": {"model_text": "reason", "expected": {"vllm_python": {"reasoning_content": "original"}}},
            }},
        },
    })
    for version in ("1.0.0", "2.0.0"):
        legacy_history._write(family / f"vllm_python-{version}.yaml", {
            "corpus": "reasoning", "family": "qwen3", "capture": f"vllm_python-{version}",
            "documents": {"one.yaml": {"family": "qwen3", "cases": {}}},
        })
    legacy_history.migrate_store(root)
    before = legacy_history.resolved_inventory(root)
    key = json.dumps(["one.yaml", "A"], separators=(",", ":"))
    state = before["reasoning/qwen3/vllm_python-2.0.0"]["views"]["python"]
    assert state[key]["payload"]["value"]["expected"]["vllm_python"] == {"reasoning_content": "original"}
    assert state[key]["producer"]["capture"] == "inputs"
    legacy_history.remove_capture(root, "vllm_python-1.0.0")
    before.pop("reasoning/qwen3/vllm_python-1.0.0")
    assert legacy_history.resolved_inventory(root) == before


def test_stream_input_unavailable_reason_is_a_retained_observation(tmp_path):
    root = tmp_path / "store"
    family = root / "stream/families/qwen3"
    legacy_history._write(family / "inputs_and_golden.yaml", {
        "corpus": "stream", "family": "qwen3", "capture": "inputs", "documents": {
            "one.yaml": {"family": "qwen3", "cases": {
                "A": {"chunks": [{"delta_text": "input"}], "unavailable": {"dynamo_v2": "token IDs required"}},
            }},
        },
    })
    for version in ("1.0.0", "2.0.0"):
        legacy_history._write(family / f"dynamo_v2-{version}.yaml", {
            "corpus": "stream", "family": "qwen3", "capture": f"dynamo_v2-{version}",
            "documents": {"one.yaml": {"family": "qwen3", "cases": {}}},
        })
    legacy_history.migrate_store(root)
    before = legacy_history.resolved_inventory(root)
    key = json.dumps(["one.yaml", "A"], separators=(",", ":"))
    for state in before["stream/qwen3/dynamo_v2-2.0.0"]["views"].values():
        assert state[key]["payload"]["value"] == {"unavailable": "token IDs required"}
        assert state[key]["producer"]["capture"] == "inputs"
    legacy_history.remove_capture(root, "dynamo_v2-1.0.0")
    before.pop("stream/qwen3/dynamo_v2-1.0.0")
    assert legacy_history.resolved_inventory(root) == before


@pytest.mark.parametrize("field,first,changed", [("calls", [], [{"name": "tool"}]), ("tool_calls", [], [{"name": "tool"}]), ("error", "first error", "second error")])
def test_batch_on_stream_outputs_deduplicate_changed_then_restored_payload(tmp_path, field, first, changed):
    root = tmp_path / "store"
    shared = legacy_history._split_snapshot_records(root, "qwen3", {
        "one.yaml": {"family": "qwen3", "captured_with": {"vllm_python": "1.0.0"},
                     "cases": {"A": {"vllm_python": {field: first, "note": "annotation"}}}},
    })
    for path, record in shared.items():
        legacy_history._write(path, record)
    original = unified_history.load_yaml(root / "batch_on_stream/families/qwen3/vllm_python-1.0.0.yaml")
    for version, value in (("1.1.0", changed), ("1.2.0", first)):
        record = copy.deepcopy(original)
        record["capture"], record["provenance"] = legacy_history._snapshot_capture("vllm_python", version)
        record["documents"]["one.yaml"]["cases"]["A"] = {field: value, "note": "annotation"}
        legacy_history._write(root / f"batch_on_stream/families/qwen3/vllm_python-{version}.yaml", record)
    legacy_history.migrate_store(root)
    ledger = unified_history.load_yaml(root / "batch_on_stream/families/qwen3/observations.yaml")
    assert len(ledger["observations"]) == 2
    assert all(field in payload["value"] for payload in ledger["observations"].values())
    inventory = legacy_history.resolved_inventory(root)
    for identity, snapshot in inventory.items():
        if "views" not in snapshot:
            continue
        for observation in snapshot["views"]["python"].values():
            assert observation["annotations"] == {"note": "annotation"}
    assert legacy_history.migrate_store(root) == []



def test_materialized_checkpoint_keeps_own_measurement_and_original_producer(tmp_path):
    root = _store(tmp_path / "store")
    for version, digest in (("0.5.0", "a" * 64), ("0.5.2", "b" * 64)):
        path = root / f"families/gemma4/dynamo_v2-{version}.yaml"
        document = unified_history.load_yaml(path)
        document["provenance"] = {"status": "captured", "origin": {"crate_version": version, "source_sha256": digest}}
        path.write_text(unified_history.dump_yaml(document))
    unified_history.migrate_store(root)
    output = tmp_path / "output"
    unified_history.materialize_store(root, output)
    capture = output / "dynamo_v2-0.5.2"
    observation = capture / "gemma4/UNIFIED.1-1.yaml"
    assert unified_history.load_yaml(observation)["capture_origin"]["source_sha256"] == "a" * 64
    checkpoint = json.loads((capture / "capture-checkpoint.json").read_text())
    assert checkpoint["families"]["gemma4"]["origin"]["source_sha256"] == "b" * 64
    assert checkpoint["records"] == {"gemma4/UNIFIED.1-1.yaml": hashlib.sha256(observation.read_bytes()).hexdigest()}


@pytest.mark.parametrize("corruption", ["version", "foreign"])
def test_unified_rejects_misbound_producer_identity(tmp_path, corruption):
    root = _store(tmp_path / "store")
    unified_history.migrate_store(root)
    path = root / "families/gemma4/observations.yaml"
    ledger = unified_history.load_yaml(path)
    if corruption == "version":
        ledger["producers"]["dynamo_v2-0.5.0"]["runtime_version"] = "9.9.9"
    else:
        producer = copy.deepcopy(ledger["producers"]["dynamo_v2-0.5.0"])
        producer["provenance"] = {"status": "legacy", "captured_with": {"vllm_python": "0.5.0"}}
        ledger["producers"]["vllm_python-0.5.0"] = producer
        checkpoint_path = path.parent / "dynamo_v2-0.5.0.yaml"
        checkpoint = unified_history.load_yaml(checkpoint_path)
        checkpoint["changes"]["text_only"]["producer"] = "vllm_python-0.5.0"
        checkpoint_path.write_text(unified_history.dump_yaml(checkpoint))
    path.write_text(unified_history.dump_yaml(ledger))
    with pytest.raises(ValueError, match="producer identity mismatch|foreign observation producer"):
        unified_history.load_store(root)


def test_legacy_rejects_foreign_producer_reference(tmp_path):
    root = _legacy(tmp_path, "batch")
    legacy_history.migrate_store(root)
    family = root / "batch/families/qwen3"
    checkpoint_path = family / "vllm_python-1.0.0.yaml"
    checkpoint = unified_history.load_yaml(checkpoint_path)
    reference = next(iter(checkpoint["views"]["python"]["changes"].values()))
    ledger_path = family / "observations.yaml"
    ledger = unified_history.load_yaml(ledger_path)
    producer = copy.deepcopy(ledger["producers"][reference["producer"]])
    producer["capture"] = "sglang_python-1.0.0"
    reference["producer"] = unified_history._request_digest(producer)
    ledger["producers"][reference["producer"]] = producer
    checkpoint_path.write_text(unified_history.dump_yaml(checkpoint))
    ledger_path.write_text(unified_history.dump_yaml(ledger))
    with pytest.raises(ValueError, match="foreign legacy observation producer"):
        legacy_history.resolved_inventory(root)
