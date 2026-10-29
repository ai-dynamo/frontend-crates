# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coverage for canonical Unified capture checkpoints."""
import copy
from collections.abc import Mapping
from dataclasses import replace
from contextlib import contextmanager
import hashlib
import json
import shutil
import sys
from pathlib import Path
import pytest
import yaml
import viewer_catalog
import unified_history
import unified_archives
import migrate_capture_sources
import publish_current_capture
import capture_sources
from capture_stimulus import normalize_unavailable, select_peer_observation
from viewer_codec import encode_typed


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.mark.parametrize("case_count", [1, 8])
@pytest.mark.parametrize("derived", [False, True])
def test_lazy_inheritance_metadata_is_loaded_once_per_snapshot(tmp_path, case_count, derived):
    class CountingFragments(Mapping):
        reads = 0

        def __getitem__(self, key):
            assert key == "dynamo_v2-0.1.0"
            self.reads += 1
            return {"case-0"}

        def __iter__(self):
            return iter(["dynamo_v2-0.1.0"])

        def __len__(self):
            return 1

    capture = "dynamo_v2-0.1.0"
    records = {f"case-{index}": {"document": {"record_metadata": {"verified_recapture": {"result": {}}}}}
               for index in range(case_count)}
    suppressed = CountingFragments()
    history = unified_history.History(
        unified_history.Family("sample", {}, tmp_path / "inputs_and_golden.yaml"),
        "dynamo_v2", {capture: {"runtime_version": "0.1.0"}}, tmp_path, {},
        {capture: records}, {}, suppressed)
    resolved = history.resolve(capture, view_capture_id="dynamo_v2-0.2.0" if derived else capture)
    assert suppressed.reads == int(derived)
    for case, record in resolved.items():
        assert ("verified_recapture" in record["document"].get("record_metadata", {})) == (not derived)
        assert ("inherited_from" in record["document"]) == (derived and case != "case-0")
        if derived and case != "case-0":
            assert record["document"]["inherited_from"] == "0.1.0"
        assert "verified_recapture" in records[case]["document"]["record_metadata"]
        assert "inherited_from" not in records[case]["document"]


@pytest.mark.parametrize("metadata", [None, {}, {"verified_recapture": {"result": {}}},
    {"verified_recapture": {"result": {}}, "retained": None}, {"retained": False}])
@pytest.mark.parametrize("derived", [False, True])
def test_complete_projection_matches_original_sparse_metadata(tmp_path, metadata, derived):
    root = tmp_path / "store"
    _write_family(root)
    _write_capture(root, "0.1.0", {"text_only": _change()},
                   metadata_changes={"text_only": metadata} if metadata is not None else {})
    sparse = unified_history.load_store(root).histories["gemma4", "dynamo_v2"]
    checkpoint = "dynamo_v2-0.1.0"
    snapshot = sparse.complete_snapshot(checkpoint)
    records = {case: wrapped["value"] for case, wrapped in snapshot["records"].items()}
    complete = replace(sparse, resolved_snapshots={checkpoint: records})
    selected = "dynamo_v2-0.2.0" if derived else checkpoint
    expected = sparse.resolve(checkpoint, view_capture_id=selected)
    actual = complete.resolve(checkpoint, view_capture_id=selected)
    assert actual == expected
    if metadata is not None and "verified_recapture" in metadata:
        assert "verified_recapture" in records["text_only"]["document"]["record_metadata"]


def _request(text: str = "hello") -> dict:
    return {
        "input": text,
        "init": {
            "starting_state": "None",
            "tool_output_mode": "Native",
            "named_tool": None,
        },
        "finish_reason": "stop",
        "tools": [],
        "chunks": [{"delta_text": text}],
    }


@pytest.mark.parametrize("operation", ["update_store_from_loose", "update_from_loose", "sync_current_corpus"])
def test_no_loose_sources_validate_store_without_reserializing(tmp_path, monkeypatch, operation):
    root = _store(tmp_path / "store")
    loose = tmp_path / "loose"
    (loose / "golden_spec").mkdir(parents=True)
    (loose / "golden_spec-1271640-0").mkdir()
    (loose / "unified_results.yaml").write_text("generated: true\n")
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*.yaml")}
    original_load = unified_history._load_store_unlocked
    loads = []

    def validated(store_root):
        loads.append(store_root)
        return original_load(store_root)

    def unexpected_serialization(*args, **kwargs):
        raise AssertionError("source-free update must not serialize the store")

    monkeypatch.setattr(unified_history, "_load_store_unlocked", validated)
    monkeypatch.setattr(unified_history, "_store_documents", unexpected_serialization)
    kwargs = {"complete_snapshot": False} if operation == "update_store_from_loose" else {}
    operations = {
        "update_store_from_loose": unified_history.update_store_from_loose,
        "update_from_loose": unified_history.update_from_loose,
        "sync_current_corpus": unified_history.sync_current_corpus,
    }
    assert operations[operation](root, loose, **kwargs) == []
    assert loads == [root]
    assert {path.relative_to(root): path.read_bytes() for path in root.rglob("*.yaml")} == before


def test_no_loose_sources_still_reject_invalid_store(tmp_path):
    root = _store(tmp_path / "store")
    (root / "unexpected.yaml").write_text("unexpected: true\n")
    with pytest.raises(ValueError, match="unknown Unified root entry"):
        unified_history.update_store_from_loose(root, tmp_path / "loose", complete_snapshot=False)


@pytest.mark.parametrize("invalid", ["complete", "excluded", "malformed", "required"])
def test_no_loose_sources_preserve_source_validation(tmp_path, invalid):
    root = _store(tmp_path / "store")
    loose = tmp_path / "loose"
    (loose / "golden_spec").mkdir(parents=True)
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*.yaml")}
    if invalid == "complete":
        with pytest.raises(ValueError, match="needs inputs and golden roots"):
            unified_history.update_store_from_loose(root, loose, complete_snapshot=True)
    elif invalid == "excluded":
        with pytest.raises(ValueError, match="absent from the loose corpus"):
            unified_history.update_store_from_loose(
                root, loose, complete_snapshot=False, excluded_capture_dirs={"dynamo_v2-0.5.0"},
            )
    elif invalid == "malformed":
        (loose / "dynamo_v2-0.5").mkdir()
        with pytest.raises(ValueError, match="must use <implementation>-<semantic-version>"):
            unified_history.update_store_from_loose(root, loose, complete_snapshot=False)
    else:
        with pytest.raises(ValueError, match="missing required captures"):
            unified_history._mutate_store(
                root,
                lambda store: unified_history._update_from_loose(
                    store, loose, complete_snapshot=True, required_capture_dirs={"dynamo_v2-0.5.0"},
                ),
                skip_no_sources=True,
            )
    assert {path.relative_to(root): path.read_bytes() for path in root.rglob("*.yaml")} == before


@pytest.mark.parametrize("updated,skip_no_sources", [(None, False), ({}, True)])
def test_mutation_no_source_optimization_requires_explicit_marker_and_opt_in(tmp_path, monkeypatch, updated, skip_no_sources):
    root = _store(tmp_path / "store")
    original_documents = unified_history._store_documents
    serialized = []

    def documents(store, **kwargs):
        serialized.append(store.root)
        return original_documents(store, **kwargs)

    monkeypatch.setattr(unified_history, "_store_documents", documents)
    unified_history._mutate_store(root, lambda store: updated, skip_no_sources=skip_no_sources)
    assert serialized == [root]


def test_no_source_optimization_preserves_expected_inventory_precondition(tmp_path):
    root = _store(tmp_path / "store")
    expected = unified_history.store_inventory(root)
    path = root / "families/gemma4/dynamo_v2-0.5.2.yaml"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="store generation changed before mutation"):
        unified_history._mutate_store(root, lambda store: None, skip_no_sources=True, expected_inventory=expected)


def test_structured_parser_error_survives_store_and_resolved_record(tmp_path):
    _write_family(tmp_path)
    change = _change()
    detail = {"message": "parser threw", "offset": 17}
    change["observation"] = {"error": {"code": "legacy_unclassified", "detail": detail}}
    path = _write_capture(tmp_path, "0.26.0", {"text_only": change})
    original = path.read_bytes()
    store = unified_history.load_store(tmp_path)
    resolved = store.histories[("gemma4", "dynamo_v2")].resolve("dynamo_v2-0.26.0")["text_only"]
    record, _document = unified_history.materialized_record(store.families["gemma4"].cases["text_only"], resolved)
    assert record["error"] == detail
    assert record["capture_observation"] == {"error": {"code": "legacy_unclassified", "detail": detail}}
    assert encode_typed(record)[0] == "object"
    assert path.read_bytes() == original


def test_materialized_unavailable_reimport_does_not_add_view_metadata(tmp_path):
    root = tmp_path / "store"
    _write_family(root)
    change = _change()
    change["stimulus"]["semantic_sha256"] = unified_history._request_digest(_request())
    payload = {"code": "vllm_rust_guided_json_unsupported", "detail": "no guided API"}
    change["observation"] = {"unavailable": payload}
    path = _write_capture(root, "0.26.0", {"text_only": change})
    original = path.read_bytes()
    store = unified_history.load_store(root)
    resolved = store.histories[("gemma4", "dynamo_v2")].resolve("dynamo_v2-0.26.0")["text_only"]
    record, _document = unified_history.materialized_record(store.families["gemma4"].cases["text_only"], resolved)
    assert "unavailable_code" not in record
    assert record["capture_observation"] == {"unavailable": payload}
    stimulus, observation, metadata = unified_history._legacy_state(record, b"", "case.yaml", {}, _request())
    assert observation == change["observation"]
    assert metadata == {}
    assert stimulus["ref"] == "current"
    loose = tmp_path / "loose"
    unified_history.materialize_store(root, loose, store=store)
    unified_history.update_store_from_loose(root, loose, complete_snapshot=True)
    after = unified_history.load_store(root).histories[("gemma4", "dynamo_v2")].resolve("dynamo_v2-0.26.0")
    assert after["text_only"] == resolved
    assert path.read_bytes() == original


def _change(text: str = "hello") -> dict:
    return {
        "case_key": "UNIFIED.1-1",
        "stimulus": {"ref": "current"},
        "observation": {
            "value": {
                "assembled": [{"kind": "text", "text": text}],
                "chunks": [{"expected": [{"kind": "text", "text": text}]}],
            }
        },
    }


def _write_family(root: Path) -> None:
    path = root / "families/gemma4/inputs_and_golden.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(
        unified_history.dump_yaml(
            {
                "input_document": {"family": "gemma4", "mode": "unified"},
                "golden_document": {"family": "gemma4", "mode": "unified"},
                "cases": {
                    "text_only": {
                        "lifecycle": "active",
                        "scenario": "text_only",
                        "description": "plain text",
                        "policy": [],
                        "display_id": "UNIFIED.1-1",
                        "historical_ids": [],
                        "request": _request(),
                        "golden": {"assembled": [{"kind": "text", "text": "hello"}]},
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_complete_snapshot_reader_correction_and_archive_roundtrip(tmp_path, monkeypatch):
    root = tmp_path / "store"
    _write_family(root)
    first = _write_capture(root, "0.26.0", {"text_only": _change()})
    _write_capture(root, "0.27.0", {})
    legacy = unified_history.load_store(root).histories[("gemma4", "dynamo_v2")]
    snapshots = dict(legacy.iter_resolved_snapshots())
    for capture_id, records in snapshots.items():
        version = capture_id.split("-", 1)[1]
        path = root / "families/gemma4" / f"{capture_id}.yaml"
        capture = legacy.captures[capture_id]
        document = {
            "schema_version": 2,
            "provenance": capture["provenance"],
            "document": capture["document"],
            "records": {
                case_id: {
                    "value": value,
                    "member": f"conformance/fixtures-unified-v2/families/gemma4/dynamo_v2-{version}.yaml"
                    if capture_id == "dynamo_v2-0.26.0"
                    else f"conformance/fixtures-unified-v2/families/gemma4/dynamo_v2-0.26.0.yaml",
                }
                for case_id, value in records.items()
            },
        }
        path.write_text(unified_history.dump_yaml(document), encoding="utf-8")
    direct = unified_history.load_store(root).histories[("gemma4", "dynamo_v2")]
    before_later = direct.resolve("dynamo_v2-0.27.0")
    corrected = copy.deepcopy(direct.resolve("dynamo_v2-0.26.0"))
    corrected["text_only"]["observation"]["value"]["assembled"][0]["text"] = "corrected"
    unified_history._replace_capture_snapshot(direct, "dynamo_v2-0.26.0", corrected)
    assert direct.resolve("dynamo_v2-0.27.0") == before_later
    documents = unified_history._store_documents(
        unified_history.Store(root, {"gemma4": direct.family}, {("gemma4", "dynamo_v2"): direct})
    )
    assert documents[first]["schema_version"] == 2
    assert documents[first]["records"]["text_only"]["member"].endswith("dynamo_v2-0.26.0.yaml")
    for path, document in documents.items():
        path.write_text(unified_history.dump_yaml(document), encoding="utf-8")
    assert unified_archives.pack(root, unified_history.CAPTURE_DIRECTORY_RE,
                                 unified_history._capture_release_sort_key) is None
    assert not list((root / "families/gemma4").glob("dynamo_v2-*.yaml"))
    loaded = unified_history.load_store(root).histories[("gemma4", "dynamo_v2")]
    assert loaded.resolve("dynamo_v2-0.26.0")["text_only"]["observation"]["value"]["assembled"][0]["text"] == "corrected"

    # A comment is permanent source data even when its parsed value is unchanged.
    authored = root / "families/gemma4/inputs_and_golden.yaml"
    authored.write_bytes(b"# retained source comment\n" + authored.read_bytes())
    before = {path.relative_to(root): (path.read_bytes(), path.stat().st_ino)
              for path in root.rglob("*") if path.is_file()}
    store = unified_history.load_store(root)
    assert unified_history._commit_store_documents(root, unified_history._store_documents(store), expected_store=store) == []
    assert before == {path.relative_to(root): (path.read_bytes(), path.stat().st_ino)
                      for path in root.rglob("*") if path.is_file()}

    history = store.histories[("gemma4", "dynamo_v2")]
    target = history.resolve("dynamo_v2-0.26.0")
    target["text_only"]["observation"]["value"]["assembled"][0]["text"] = "second correction"
    unified_history._replace_capture_snapshot(history, "dynamo_v2-0.26.0", target)
    unified_history._commit_store_documents(root, unified_history._store_documents(store), expected_store=store)
    later = root / "versions/dynamo-0.27.0.tar.gz"
    assert (later.read_bytes(), later.stat().st_ino) == before[later.relative_to(root)]
    assert authored.read_bytes() == before[authored.relative_to(root)][0]
    final = unified_history.load_store(root)
    assert final.histories[("gemma4", "dynamo_v2")].resolve("dynamo_v2-0.27.0") == before_later

    # Selecting a self-contained version must not pull its producer's history.
    from_resources = viewer_catalog._unified_records(unified=root)
    selected = unified_history.selected_store(root, from_resources, {"dynamo_v2": "0.27.1"})
    assert set(selected.histories[("gemma4", "dynamo_v2")].captures) == {"dynamo_v2-0.27.0"}
    output = tmp_path / "selected"
    unified_history.materialize_store(root, output, store=selected,
                                      selected_capture_ids={"dynamo_v2-0.27.1"},
                                      derived_release_versions={"dynamo_v2": "0.27.1"})
    assert {path.name for path in output.iterdir()} == {"inputs", "golden", "dynamo_v2-0.27.1"}

    admissions = []
    original_members = unified_archives.members

    def admitted_once(path, **kwargs):
        admissions.append(path.name)
        yield from original_members(path, **kwargs)

    def unexpected_reread(*args, **kwargs):
        raise AssertionError("selected members must reuse the inventory admission")

    monkeypatch.setattr(unified_archives, "members", admitted_once)
    monkeypatch.setattr(unified_archives, "read_sources", unexpected_reread)
    cold, targets = unified_history.selected_materialization_store(root, ["latest:dynamo_v2"])
    assert targets == {"dynamo_v2": "0.27.0"}
    assert cold.histories[("gemma4", "dynamo_v2")].resolve("dynamo_v2-0.27.0") == before_later
    assert sorted(admissions) == ["dynamo-0.26.0.tar.gz", "dynamo-0.27.0.tar.gz"]
    admissions.clear()
    cold, targets = unified_history.selected_materialization_store(root, ["dynamo_v2-0.26.1"])
    assert targets == {"dynamo_v2": "0.26.1"}
    assert set(cold.histories[("gemma4", "dynamo_v2")].captures) == {"dynamo_v2-0.26.0"}
    assert len(admissions) == len(set(admissions)) == 2


@pytest.mark.parametrize("implementation", ["dynamo_v1", "dynamo_v2"])
def test_current_materialization_excludes_v1_only_cases_only_for_v2(tmp_path, implementation):
    root = tmp_path / "store"
    _write_family(root)
    authored = root / "families/gemma4/inputs_and_golden.yaml"
    document = unified_history.load_yaml(authored)
    document["cases"]["text_only"]["scenario"] = "v1_split_parallel_tool_calls"
    authored.write_text(unified_history.dump_yaml(document))
    capture = _write_capture(root, "0.26.0", {"text_only": _change()}, implementation=implementation)
    original = capture.read_bytes()
    historical = tmp_path / "historical"
    unified_history.materialize_store(root, historical)
    current = tmp_path / "current"
    unified_history.materialize_store(root, current, current_capture_id=capture.stem)
    assert list((historical / capture.stem).rglob("*.yaml"))
    assert bool(list((current / capture.stem).rglob("*.yaml"))) == (implementation == "dynamo_v1")
    assert capture.read_bytes() == original


@pytest.mark.parametrize("invalid", ["origin", "batch"])
def test_complete_snapshot_validates_resolved_record_contracts(tmp_path, invalid):
    root = tmp_path / "store"
    _write_family(root)
    _write_capture(root, "0.5.0", {"text_only": _change()}, implementation="dynamo_v1")
    history = unified_history.load_store(root).histories[("gemma4", "dynamo_v1")]
    record = history.resolve("dynamo_v1-0.5.0")["text_only"]
    if invalid == "origin":
        record["document"]["capture_origin"] = {"crate_version": "0.5.0", "source_sha256": "bad"}
    else:
        record["document"]["record_metadata"] = {
            "batch_final": {"reasoning_text": "", "normal_text": "different", "calls": []}
        }
    capture = history.captures["dynamo_v1-0.5.0"]
    path = root / "families/gemma4/dynamo_v1-0.5.0.yaml"
    path.write_text(unified_history.dump_yaml({
        "schema_version": 2,
        "provenance": capture["provenance"],
        "document": capture["document"],
        "records": {"text_only": {
            "value": record,
            "member": "conformance/fixtures-unified-v2/families/gemma4/dynamo_v1-0.5.0.yaml",
        }},
    }))

    expected = "invalid source hash" if invalid == "origin" else "batch observation does not match its final fields"
    with pytest.raises(ValueError, match=expected):
        unified_history.load_store(root)


@pytest.mark.parametrize("field", ["checkpoint", "defaults", "exceptions"])
def test_complete_snapshot_validates_historical_evidence_mappings(tmp_path, field):
    root = tmp_path / "store"
    _write_family(root)
    _write_capture(root, "0.5.0", {"text_only": _change()}, implementation="dynamo_v2")
    history = unified_history.load_store(root).histories[("gemma4", "dynamo_v2")]
    capture = history.captures["dynamo_v2-0.5.0"]
    record = history.resolve("dynamo_v2-0.5.0")["text_only"]
    evidence = {
        "kind": "execution_manifest",
        "schema_version": 1,
        "family": "gemma4",
        "implementation": "dynamo_v2",
        "version": "0.5.0",
        "checkpoint": {},
        "cases": ["text_only"],
        "defaults": {},
        "exceptions": {},
        "metadata_changes": {"text_only": {}},
        "receipt_membership": [],
    }
    evidence[field] = []
    snapshot = {
        "schema_version": 2,
        "provenance": capture["provenance"],
        "document": capture["document"],
        "records": {"text_only": {
            "value": record,
            "member": "conformance/fixtures-unified-v2/families/gemma4/dynamo_v2-0.5.0.yaml",
        }},
        "historical_evidence": [evidence],
    }
    (root / "families/gemma4/dynamo_v2-0.5.0.yaml").write_text(unified_history.dump_yaml(snapshot), encoding="utf-8")

    with pytest.raises(ValueError, match=rf"historical_evidence\[0\]\.{field} must be a mapping"):
        unified_history.load_store(root)


def test_current_corpus_adds_a_registered_family(tmp_path):
    loose = tmp_path / "loose"
    input_dir = loose / "inputs/gpt_oss"
    golden_dir = loose / "golden/gpt_oss"
    input_dir.mkdir(parents=True)
    golden_dir.mkdir(parents=True)
    case_key = "UNIFIED.1-tool_only"
    request = _request("authored request")
    input_document = {
        "family": "gpt_oss",
        "mode": "unified",
        "model_label": "gpt_oss",
        "cases": {
            case_key: {
                "scenario": "tool_only",
                "description": "single tool call",
                "policy": [],
                **request,
            }
        },
    }
    golden_document = {
        "family": "gpt_oss",
        "mode": "unified",
        "captured_with": {"golden": "v1"},
        "cases": {case_key: {"assembled": [{"kind": "tool_call", "name": "f", "arguments": {}}]}},
    }
    (input_dir / f"{case_key}.yaml").write_text(unified_history.dump_yaml(input_document))
    (golden_dir / f"{case_key}.yaml").write_text(unified_history.dump_yaml(golden_document))
    store = unified_history.Store(tmp_path / "store", {}, {})

    documents = unified_history._sync_current_corpus(store, loose, complete_snapshot=True)

    family = store.families["gpt_oss"]
    assert family.path == tmp_path / "store/families/gpt_oss/inputs_and_golden.yaml"
    assert len(family.cases) == 1
    case = next(iter(family.cases.values()))
    assert case["request"] == request
    assert case["golden"] == golden_document["cases"][case_key]
    assert documents[family.path] == family.document


def test_first_capture_for_existing_family_is_published_and_idempotent(tmp_path):
    root = tmp_path / "store"
    _write_family(root)
    prior_path = _write_capture(root, "0.26.0", {"text_only": _change()}, implementation="vllm_python")
    prior_bytes = prior_path.read_bytes()
    loose = tmp_path / "loose"
    path = loose / "dynamo_v2-0.7.27/gemma4/case.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(unified_history.dump_yaml({
        "family": "gemma4", "mode": "unified",
        "captured_with": {"dynamo_v2": "0.7.27"},
        "cases": {"UNIFIED.1-1": {"assembled": [{"kind": "text", "text": "hello"}]}},
    }))
    changed = unified_history.update_from_loose(root, loose)
    capture_path = root / "families/gemma4/dynamo_v2-0.7.27.yaml"
    assert changed == [capture_path]
    store = unified_history.load_store(root)
    record = store.histories[("gemma4", "dynamo_v2")].resolve("dynamo_v2-0.7.27")["text_only"]
    assert record["observation"]["value"]["assembled"] == [{"kind": "text", "text": "hello"}]
    original = capture_path.read_bytes()
    assert unified_history.update_from_loose(root, loose) == []
    assert capture_path.read_bytes() == original
    assert prior_path.read_bytes() == prior_bytes


def test_descriptive_migration_preserves_capture_bytes_and_is_idempotent(tmp_path):
    _write_family(tmp_path)
    family_path = tmp_path / "families/gemma4/inputs_and_golden.yaml"
    before = unified_history.load_yaml(family_path)
    before["cases"]["text_only"]["display_id"] = "UNIFIED.3-1"
    family_path.write_text(unified_history.dump_yaml(before))
    capture = _write_capture(tmp_path, "0.6.0", {"text_only": {
        **_change(), "case_key": "UNIFIED.3-1",
    }})
    capture_bytes = capture.read_bytes()
    assert unified_history.migrate_descriptive_ids(tmp_path) == [family_path]
    after = unified_history.load_yaml(family_path)
    expected = copy.deepcopy(before)
    expected["cases"]["text_only"]["display_id"] = "UNIFIED.3-text_only"
    expected["cases"]["text_only"]["historical_ids"] = ["UNIFIED.3-1"]
    assert after == expected
    assert capture.read_bytes() == capture_bytes
    assert unified_history.migrate_descriptive_ids(tmp_path) == []


def _write_capture(
    root: Path,
    version: str,
    changes: dict,
    *,
    implementation: str = "dynamo_v2",
    metadata_changes: dict | None = None,
    document_overrides: dict | None = None,
    provenance: dict | None = None,
) -> Path:
    path = root / f"families/gemma4/{implementation}-{version}.yaml"
    path.write_text(
        unified_history.dump_yaml(
            {
                "provenance": provenance
                or {"status": "legacy", "captured_with": {implementation: version}},
                "document": {"mode": "unified"},
                "changes": changes,
                "metadata_changes": metadata_changes or {},
                "document_overrides": document_overrides or {},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_first_capture_creates_a_new_implementation_history(tmp_path):
    root = tmp_path / "store"
    _write_family(root)
    _write_capture(root, "0.5.14", {"text_only": _change()}, implementation="sglang_python")
    loose = tmp_path / "loose"
    path = loose / "dynamo_v2-0.26.0/gemma4/UNIFIED.1-1.yaml"
    path.parent.mkdir(parents=True)
    events = [{"kind": "text", "text": "hello"}]
    path.write_text(unified_history.dump_yaml({
        "family": "gemma4", "mode": "unified",
        "captured_with": {"dynamo_v2": "0.26.0"},
        "cases": {"UNIFIED.1-1": {
            "capture_input": _request(), "assembled": events,
            "chunks": [{"expected": events}],
        }},
    }))
    unified_history.update_store_from_loose(root, loose, complete_snapshot=False)
    history = unified_history.load_store(root).histories[("gemma4", "dynamo_v2")]
    assert history.resolve("dynamo_v2-0.26.0")["text_only"]["observation"]["value"]["assembled"] == events


@pytest.mark.parametrize("later_changes", [False, True])
def test_correcting_older_placeholder_preserves_later_observation(tmp_path, later_changes):
    store = tmp_path / "store"
    _write_family(store)
    old = _change()
    old["stimulus"]["semantic_sha256"] = unified_history._request_digest(_request())
    old["observation"] = {"unavailable": {
        "code": "vllm_tool_schema_unsupported", "detail": "old adapter cannot send tools",
    }}
    _write_capture(store, "0.25.1", {"text_only": old}, implementation="vllm_python")
    _write_capture(
        store, "0.26.0", {"text_only": _change("later")} if later_changes else {},
        implementation="vllm_python",
    )
    history = unified_history.load_store(store).histories[("gemma4", "vllm_python")]
    later = history.resolve("vllm_python-0.26.0")
    loose = tmp_path / "loose"
    path = loose / "vllm_python-0.25.1/gemma4/UNIFIED.1-1.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(unified_history.dump_yaml({
        "family": "gemma4", "mode": "unified", "captured_with": {"vllm_python": "0.25.1"},
        "cases": {"UNIFIED.1-1": {
            "capture_input": _request(), "assembled": [{"kind": "text", "text": "new capture"}],
            "chunks": [{"expected": [{"kind": "text", "text": "new capture"}]}],
        }},
    }))
    unified_history.update_store_from_loose(store, loose, complete_snapshot=False)
    history = unified_history.load_store(store).histories[("gemma4", "vllm_python")]
    assert history.resolve("vllm_python-0.26.0") == later
    assert history.resolve("vllm_python-0.25.1")["text_only"]["observation"] == {
        "value": {"assembled": [{"kind": "text", "text": "new capture"}],
                  "chunks": [{"expected": [{"kind": "text", "text": "new capture"}]}]},
    }


def _store(root: Path) -> Path:
    _write_family(root)
    _write_capture(root, "0.5.0", {"text_only": _change()})
    _write_capture(root, "0.5.2", {})
    return root


def test_materialization_reuses_supplied_store_without_mutating_it(tmp_path, monkeypatch):
    root = _store(tmp_path / "store")
    store = unified_history.load_store(root)
    before = copy.deepcopy(store)
    default = tmp_path / "default"
    supplied = tmp_path / "supplied"
    unified_history.materialize_store(root, default)

    def unexpected_load(_root):
        pytest.fail("materialization reread an already supplied store")

    monkeypatch.setattr(unified_history, "load_store", unexpected_load)
    unified_history.materialize_store(root, supplied, store=store)
    assert {
        path.relative_to(default): path.read_bytes() for path in default.rglob("*.yaml")
    } == {
        path.relative_to(supplied): path.read_bytes() for path in supplied.rglob("*.yaml")
    }
    assert store == before


def test_input_only_materialization_preserves_default_input_and_golden_bytes(tmp_path):
    root = _store(tmp_path / "store")
    store = unified_history.load_store(root)
    before = copy.deepcopy(store)
    default = tmp_path / "default"
    inputs_only = tmp_path / "inputs-only"
    unified_history.materialize_store(root, default, store=store)
    unified_history.materialize_store(root, inputs_only, store=replace(store, histories={}))
    expected = {
        path.relative_to(default): path.read_bytes()
        for directory in ("inputs", "golden")
        for path in (default / directory).rglob("*.yaml")
    }
    assert expected
    assert expected == {
        path.relative_to(inputs_only): path.read_bytes()
        for path in inputs_only.rglob("*.yaml")
    }
    for capture_id in store.histories[("gemma4", "dynamo_v2")].captures:
        assert list((default / capture_id).rglob("*.yaml"))
        assert not (inputs_only / capture_id).exists()
    assert store == before


def _add_backfill_case(root: Path) -> None:
    path = root / "families/gemma4/inputs_and_golden.yaml"
    document = unified_history.load_yaml(path)
    request = _request("new")
    document["cases"]["new_case"] = {
        "lifecycle": "active",
        "scenario": "new_case",
        "description": "new case",
        "policy": [],
        "display_id": "UNIFIED.1-2",
        "historical_ids": [],
        "request": request,
        "golden": {"assembled": [{"kind": "text", "text": "new"}]},
    }
    path.write_text(unified_history.dump_yaml(document), encoding="utf-8")


def _add_loose_backfill_record(
    loose: Path, version: str, output: str, *, implementation: str = "dynamo_v2"
) -> None:
    request = _request("new")
    event = {"kind": "text", "text": output}
    path = loose / f"{implementation}-{version}/gemma4/UNIFIED.1-2.yaml"
    document = unified_history.load_yaml(path.parent / "UNIFIED.1-1.yaml")
    document["cases"] = {
        "UNIFIED.1-2": {
            "capture_input": unified_history.capture_stimulus.capture_input(request),
            "assembled": [event],
            "chunks": [{"expected": [event]}],
        }
    }
    path.write_text(unified_history.dump_yaml(document), encoding="utf-8")


def test_history_accepts_partial_peer_backfill_and_preserves_inherited_rows(tmp_path):
    root = _store(tmp_path / "store")
    _write_capture(root, "0.5.0", {"text_only": _change()}, implementation="vllm_python")
    _add_backfill_case(root)
    loose = tmp_path / "loose"
    unified_history.materialize_store(root, loose)
    _add_loose_backfill_record(loose, "0.5.0", "captured", implementation="vllm_python")

    unified_history.update_store_from_loose(root, loose, complete_snapshot=False)

    history = unified_history.load_store(root).histories[("gemma4", "vllm_python")]
    resolved = history.resolve("vllm_python-0.5.0")
    assert set(resolved) == {"text_only", "new_case"}
    assert resolved["text_only"]["observation"] == _change()["observation"]
    assert resolved["new_case"]["observation"]["value"]["assembled"] == [
        {"kind": "text", "text": "captured"}
    ]


def test_legacy_checkpoint_appends_bound_case_with_matching_source_origin(tmp_path):
    root = _store(tmp_path / "store")
    _write_capture(root, "0.5.0", {"text_only": _change()})
    history = unified_history.load_store(root).histories[("gemma4", "dynamo_v2")]
    original = history.resolve("dynamo_v2-0.5.0")["text_only"]
    _add_backfill_case(root)
    loose = tmp_path / "loose"
    unified_history.materialize_store(root, loose)
    _add_loose_backfill_record(loose, "0.5.0", "captured")
    path = loose / "dynamo_v2-0.5.0/gemma4/UNIFIED.1-2.yaml"
    document = unified_history.load_yaml(path)
    document["capture_provenance"] = {
        "kind": "unpublished",
        "crate_version": "0.5.0",
        "source_sha256": "a" * 64,
        "source_id": "sha256:" + "a" * 64,
    }
    path.write_text(unified_history.dump_yaml(document), encoding="utf-8")

    unified_history.update_store_from_loose(root, loose, complete_snapshot=False)

    updated = unified_history.load_store(root).histories[("gemma4", "dynamo_v2")]
    resolved = updated.resolve("dynamo_v2-0.5.0")
    assert resolved["text_only"] == original
    assert resolved["new_case"]["document"]["capture_origin"] == {
        "crate_version": "0.5.0",
        "source_sha256": "a" * 64,
    }
    assert updated.captures["dynamo_v2-0.5.0"]["provenance"]["status"] == "legacy"


def test_source_capture_origin_is_pinned_for_an_inherited_unchanged_observation(tmp_path):
    root = _store(tmp_path / "store")
    _write_capture(root, "0.5.0", {"text_only": _change()})
    _write_capture(root, "0.5.1", {})
    original = unified_history.load_store(root).histories[("gemma4", "dynamo_v2")].resolve(
        "dynamo_v2-0.5.0"
    )["text_only"]
    before = tmp_path / "before"
    unified_history.materialize_store(root, before, include_current_inputs=False)
    loose = tmp_path / "loose" / "dynamo_v2-0.5.1" / "gemma4"
    loose.mkdir(parents=True)
    document = unified_history.load_yaml(
        before / "dynamo_v2-0.5.0/gemma4/UNIFIED.1-1.yaml"
    )
    document["captured_with"] = {"dynamo_v2": "0.5.1"}
    document["capture_provenance"] = {
        "kind": "unpublished",
        "crate_version": "0.5.1",
        "source_sha256": "a" * 64,
        "source_id": "sha256:" + "a" * 64,
        "git_commit": "b" * 40,
        "label": "0.5.1+source." + "a" * 64,
    }
    (loose / "UNIFIED.1-1.yaml").write_text(
        unified_history.dump_yaml(document), encoding="utf-8"
    )

    unified_history.update_store_from_loose(root, tmp_path / "loose", complete_snapshot=False)

    updated = unified_history.load_store(root).histories[("gemma4", "dynamo_v2")]
    assert updated.resolve("dynamo_v2-0.5.0")["text_only"] == original
    latest = updated.resolve("dynamo_v2-0.5.1")["text_only"]
    assert latest["observation"] == original["observation"]
    assert latest["document"]["capture_origin"] == {
        "crate_version": "0.5.1",
        "source_sha256": "a" * 64,
        "git_commit": "b" * 40,
    }


@pytest.mark.parametrize("later_absent", [False, True])
def test_sparse_historical_insertion_keeps_unchanged_sibling_inherited(tmp_path, later_absent):
    root = tmp_path / "store"
    _write_family(root)
    _add_backfill_case(root)
    sibling = {**_change("new"), "case_key": "UNIFIED.1-2"}
    _write_capture(root, "0.5.0", {"text_only": _change(), "new_case": sibling})
    _write_capture(root, "0.5.2", {"new_case": {"absent": True}} if later_absent else {})
    history = unified_history.load_store(root).histories[("gemma4", "dynamo_v2")]
    snapshots = {key: history.resolve(key) for key in history.ordered_capture_ids()}

    source = tmp_path / "source"
    unified_history.materialize_store(root, source)
    loose = tmp_path / "loose"
    middle = loose / "dynamo_v2-0.5.1/gemma4"
    middle.mkdir(parents=True)
    document = unified_history.load_yaml(source / "dynamo_v2-0.5.0/gemma4/UNIFIED.1-1.yaml")
    document["captured_with"] = {"dynamo_v2": "0.5.1"}
    document["cases"]["UNIFIED.1-1"]["assembled"] = [{"kind": "text", "text": "middle"}]
    document["cases"]["UNIFIED.1-1"]["chunks"] = [
        {"expected": [{"kind": "text", "text": "middle"}]}
    ]
    (middle / "UNIFIED.1-1.yaml").write_text(unified_history.dump_yaml(document))

    unified_history.update_store_from_loose(root, loose, complete_snapshot=False)

    history = unified_history.load_store(root).histories[("gemma4", "dynamo_v2")]
    assert {key: history.resolve(key) for key in snapshots} == snapshots
    assert history.resolve("dynamo_v2-0.5.1")["new_case"]["observation"] == snapshots["dynamo_v2-0.5.0"]["new_case"]["observation"]
    assert "observation" not in history.captures["dynamo_v2-0.5.2"]["changes"].get("new_case", {})


def test_history_partial_new_peer_capture_inherits_omitted_rows(tmp_path):
    root = _store(tmp_path / "store")
    _write_capture(root, "0.5.0", {"text_only": _change()}, implementation="vllm_python")
    _add_backfill_case(root)
    loose = tmp_path / "loose"
    unified_history.materialize_store(root, loose)
    _add_loose_backfill_record(loose, "0.5.0", "captured", implementation="vllm_python")
    unified_history.update_store_from_loose(root, loose, complete_snapshot=False)

    prior_capture = loose / "vllm_python-0.5.0/gemma4"
    new_capture = loose / "vllm_python-0.5.3/gemma4"
    new_capture.mkdir(parents=True)
    for path in prior_capture.glob("*.yaml"):
        (new_capture / path.name).write_bytes(path.read_bytes())
    (new_capture / "UNIFIED.1-2.yaml").unlink()
    path = new_capture / "UNIFIED.1-1.yaml"
    document = unified_history.load_yaml(path)
    document["cases"]["UNIFIED.1-1"]["assembled"] = [
        {"kind": "text", "text": "updated"}
    ]
    document["cases"]["UNIFIED.1-1"]["chunks"] = [
        {"expected": [{"kind": "text", "text": "updated"}]}
    ]
    path.write_text(unified_history.dump_yaml(document), encoding="utf-8")

    unified_history.update_store_from_loose(root, loose, complete_snapshot=False)

    history = unified_history.load_store(root).histories[("gemma4", "vllm_python")]
    prior = history.resolve("vllm_python-0.5.0")
    current = history.resolve("vllm_python-0.5.3")
    assert set(current) == set(prior) == {"text_only", "new_case"}
    assert current["new_case"]["observation"] == prior["new_case"]["observation"]
    assert current["text_only"]["observation"]["value"]["assembled"] == [
        {"kind": "text", "text": "updated"}
    ]


def test_history_rejects_excluded_capture_with_unrecorded_case(tmp_path):
    root = _store(tmp_path / "store")
    _add_backfill_case(root)
    loose = tmp_path / "loose"
    unified_history.materialize_store(root, loose)
    omitted = loose / "dynamo_v2-0.5.1/gemma4"
    omitted.mkdir(parents=True)
    prior = loose / "dynamo_v2-0.5.0/gemma4/UNIFIED.1-1.yaml"
    (omitted / prior.name).write_bytes(prior.read_bytes())
    _add_loose_backfill_record(loose, "0.5.1", "captured")

    with pytest.raises(ValueError, match="unrecorded cases: new_case"):
        unified_history.update_store_from_loose(
            root,
            loose,
            complete_snapshot=False,
            excluded_capture_dirs={"dynamo_v2-0.5.1"},
        )


def test_history_excludes_sparse_capture_against_latest_prior_checkpoint(tmp_path):
    root = _store(tmp_path / "store")
    loose = tmp_path / "loose"
    unified_history.materialize_store(root, loose)
    omitted = loose / "dynamo_v2-0.5.1/gemma4"
    omitted.mkdir(parents=True)
    prior = loose / "dynamo_v2-0.5.0/gemma4/UNIFIED.1-1.yaml"
    (omitted / prior.name).write_bytes(prior.read_bytes())

    unified_history.update_store_from_loose(
        root,
        loose,
        complete_snapshot=False,
        excluded_capture_dirs={"dynamo_v2-0.5.1"},
    )

    history = unified_history.load_store(root).histories[("gemma4", "dynamo_v2")]
    assert "dynamo_v2-0.5.1" not in history.captures
    assert history.resolve("dynamo_v2-0.5.2")["text_only"]["observation"] == _change()[
        "observation"
    ]


def test_history_excluded_capture_keeps_yaml_result_for_known_case(tmp_path):
    root = _store(tmp_path / "store")
    _write_capture(root, "0.5.0", {"text_only": _change()}, implementation="vllm_python")
    loose = tmp_path / "loose"
    unified_history.materialize_store(root, loose)
    path = loose / "vllm_python-0.5.0/gemma4/UNIFIED.1-1.yaml"
    document = unified_history.load_yaml(path)
    document["cases"]["UNIFIED.1-1"]["assembled"] = [
        {"kind": "text", "text": "new local output"}
    ]
    document["cases"]["UNIFIED.1-1"]["chunks"] = [
        {"expected": [{"kind": "text", "text": "new local output"}]}
    ]
    path.write_text(unified_history.dump_yaml(document), encoding="utf-8")

    unified_history.update_store_from_loose(
        root,
        loose,
        complete_snapshot=False,
        excluded_capture_dirs={"vllm_python-0.5.0"},
    )

    resolved = unified_history.load_store(root).histories[("gemma4", "vllm_python")].resolve(
        "vllm_python-0.5.0"
    )
    assert resolved["text_only"]["observation"] == _change()["observation"]


def test_history_excluded_capture_does_not_parse_capture_stimulus(tmp_path):
    root = _store(tmp_path / "store")
    loose = tmp_path / "loose"
    unified_history.materialize_store(root, loose)
    path = loose / "dynamo_v2-0.5.0/gemma4/UNIFIED.1-1.yaml"
    document = unified_history.load_yaml(path)
    document["cases"]["UNIFIED.1-1"]["capture_stimulus"] = {"invalid": {}}
    path.write_text(unified_history.dump_yaml(document), encoding="utf-8")

    unified_history.update_store_from_loose(
        root,
        loose,
        complete_snapshot=False,
        excluded_capture_dirs={"dynamo_v2-0.5.0"},
    )

    resolved = unified_history.load_store(root).histories[("gemma4", "dynamo_v2")].resolve(
        "dynamo_v2-0.5.0"
    )
    assert resolved["text_only"]["observation"] == _change()["observation"]


def test_history_imports_explicit_partial_capture_stimulus(tmp_path):
    root = _store(tmp_path / "store")
    loose = tmp_path / "loose"
    unified_history.materialize_store(root, loose)
    new_capture = loose / "dynamo_v2-0.5.1/gemma4"
    new_capture.mkdir(parents=True)
    prior = loose / "dynamo_v2-0.5.0/gemma4/UNIFIED.1-1.yaml"
    path = new_capture / prior.name
    path.write_bytes(prior.read_bytes())
    document = unified_history.load_yaml(path)
    record = document["cases"]["UNIFIED.1-1"]
    record["capture_input"] = {"input": "older request"}
    record["capture_stimulus"] = {"partial": {"input": "older request"}}
    path.write_text(unified_history.dump_yaml(document), encoding="utf-8")

    unified_history.update_store_from_loose(root, loose, complete_snapshot=False)

    resolved = unified_history.load_store(root).histories[("gemma4", "dynamo_v2")].resolve("dynamo_v2-0.5.1")
    assert resolved["text_only"]["stimulus"] == {
        "partial": {"input": "older request"}
    }


def test_history_complete_peer_snapshot_rejects_omitted_recorded_active_rows(tmp_path):
    root = _store(tmp_path / "store")
    _write_capture(root, "0.5.0", {"text_only": _change()}, implementation="vllm_python")
    _add_backfill_case(root)
    loose = tmp_path / "loose"
    unified_history.materialize_store(root, loose)
    _add_loose_backfill_record(loose, "0.5.0", "captured", implementation="vllm_python")
    (loose / "vllm_python-0.5.0/gemma4/UNIFIED.1-1.yaml").unlink()

    with pytest.raises(ValueError, match="missing previously recorded active cases: gemma4/text_only"):
        unified_history.update_store_from_loose(root, loose, complete_snapshot=True)


@pytest.mark.parametrize(
    ("create_loose_dir", "message"),
    [
        (False, "absent from the loose corpus"),
        (True, "absent from that family's YAML history"),
    ],
)
def test_history_rejects_unknown_excluded_capture_identities(
    tmp_path, create_loose_dir, message
):
    root = _store(tmp_path / "store")
    loose = tmp_path / "loose"
    loose.mkdir()
    if create_loose_dir:
        (loose / "vllm_python-0.5.0/gemma4").mkdir(parents=True)

    with pytest.raises(ValueError, match=message):
        unified_history.update_store_from_loose(
            root,
            loose,
            complete_snapshot=False,
            excluded_capture_dirs={"vllm_python-0.5.0"},
        )


def test_history_rejects_exclusion_missing_from_one_family_history(tmp_path):
    root = _store(tmp_path / "store")
    _write_capture(root, "0.5.0", {"text_only": _change()}, implementation="vllm_python")
    loose = tmp_path / "loose"
    (loose / "vllm_python-0.5.0/gemma4").mkdir(parents=True)
    (loose / "vllm_python-0.5.0/qwen3").mkdir()

    with pytest.raises(
        ValueError,
        match="cannot exclude Unified capture vllm_python-0.5.0/qwen3: absent from that family's YAML history",
    ):
        unified_history.update_store_from_loose(
            root,
            loose,
            complete_snapshot=False,
            excluded_capture_dirs={"vllm_python-0.5.0"},
        )


def test_history_complete_snapshot_cannot_exclude_a_required_capture(tmp_path):
    root = _store(tmp_path / "store")
    loose = tmp_path / "loose"
    (loose / "dynamo_v2-0.5.0/gemma4").mkdir(parents=True)

    with pytest.raises(ValueError, match="cannot exclude required Unified captures: dynamo_v2-0.5.0"):
        unified_history._update_from_loose(
            unified_history.load_store(root),
            loose,
            complete_snapshot=True,
            excluded_capture_dirs={"dynamo_v2-0.5.0"},
            required_capture_dirs={"dynamo_v2-0.5.0"},
        )


def test_history_carries_a_missing_checkpoint_forward(tmp_path):
    history = unified_history.load_store(_store(tmp_path)).histories[("gemma4", "dynamo_v2")]

    resolved = history.resolve("dynamo_v2-0.5.2")

    assert history.ordered_capture_ids() == ["dynamo_v2-0.5.0", "dynamo_v2-0.5.2"]
    assert resolved["text_only"]["observation"] == _change()["observation"]
    assert resolved["text_only"]["document"]["captured_with"] == {"dynamo_v2": "0.5.0"}
    assert resolved["text_only"]["document"]["inherited_from"] == "0.5.0"


@pytest.mark.parametrize("transition", ["document", "metadata", "stimulus", "case_key", "observation", "readded"])
def test_snapshot_migration_preserves_observation_member_after_provenance_override(tmp_path, transition):
    root = _store(tmp_path / "store")
    (root / "executions").mkdir(exist_ok=True)
    changes = {}
    metadata = {}
    if transition == "readded":
        _write_capture(root, "0.5.1", {"text_only": {"absent": True}})
        changes = {"text_only": _change()}
    elif transition in {"stimulus", "case_key", "observation"}:
        changes = {"text_only": {transition: _change()[transition]}}
    elif transition == "metadata":
        metadata = {"text_only": {"note": "metadata does not replace an observation"}}
    _write_capture(root, "0.5.2", changes, metadata_changes=metadata, document_overrides={"text_only": {
        "capture_provenance": {"captured_with": {"dynamo_v2": "0.5.2"},
                               "capture_origin": None, "inherited_from": None},
    }})
    destination = tmp_path / "converted"
    migrate_capture_sources.convert_snapshots(root, destination, tmp_path)
    history = unified_history.load_store(destination).histories[("gemma4", "dynamo_v2")]
    capture_id = "dynamo_v2-0.5.2"
    assert history.resolve(capture_id)["text_only"]["document"]["captured_with"] == {
        "dynamo_v2": "0.5.2",
    }
    origin_version = "0.5.2" if transition in {"observation", "readded"} else "0.5.0"
    assert history.snapshot_members[capture_id]["text_only"] == (
        f"conformance/fixtures-unified-v2/families/gemma4/dynamo_v2-{origin_version}.yaml"
    )
    snapshot = {
        "schema_version": 2,
        "provenance": history.captures[capture_id]["provenance"],
        "document": history.captures[capture_id]["document"],
        "records": {"text_only": {
            "member": history.snapshot_members[capture_id]["text_only"],
            "value": history.resolve(capture_id)["text_only"],
        }},
    }
    future_member = copy.deepcopy(snapshot)
    future_member["records"]["text_only"]["member"] = (
        "conformance/fixtures-unified-v2/families/gemma4/dynamo_v2-99.0.0.yaml"
    )
    with pytest.raises(ValueError, match="source member is too new"):
        unified_history._validate_complete_snapshot(
            history.capture_path(capture_id), future_member, history.family, "dynamo_v2", "0.5.2",
        )
    snapshot["records"]["text_only"]["value"]["document"]["captured_with"] = {"dynamo_v2": "99.0.0"}
    with pytest.raises(ValueError, match="producer version"):
        unified_history._validate_complete_snapshot(
            history.capture_path(capture_id), snapshot, history.family, "dynamo_v2", "0.5.2",
        )


def test_history_partial_case_key_change_inherits_the_observation(tmp_path):
    root = _store(tmp_path)
    _write_capture(root, "0.5.2", {"text_only": {"case_key": "UNIFIED.1-2"}})

    resolved = unified_history.load_store(root).histories[("gemma4", "dynamo_v2")].resolve(
        "dynamo_v2-0.5.2"
    )["text_only"]

    assert resolved["case_key"] == "UNIFIED.1-2"
    assert resolved["observation"] == _change()["observation"]
    assert resolved["document"]["inherited_from"] == "0.5.0"


def test_history_delta_omits_an_unchanged_observation():
    before = _change()
    after = _change()
    after["case_key"] = "UNIFIED.1-2"
    after["stimulus"]["semantic_sha256"] = "a" * 64

    assert unified_history._capture_changes(
        {"text_only": before}, {"text_only": after}
    ) == {"text_only": {"case_key": "UNIFIED.1-2", "stimulus": after["stimulus"]}}


def test_history_preserves_version_specific_unregistered_peer_detail():
    before = _change()
    after = _change()
    before["observation"] = {
        "unavailable": {
            "code": "vllm_rust_parser_not_registered",
            "detail": "No vLLM Rust Unified parser is registered for gemma4 at 0.25.1.",
        }
    }
    after["observation"] = {
        "unavailable": {
            "code": "vllm_rust_parser_not_registered",
            "detail": "No vLLM Rust Unified parser is registered for gemma4 at 0.26.0.",
        }
    }

    assert unified_history._capture_changes(
        {"text_only": before}, {"text_only": after}
    ) == {"text_only": {"observation": after["observation"]}}

    after["observation"]["unavailable"]["code"] = "vllm_tool_schema_unsupported"
    assert unified_history._capture_changes(
        {"text_only": before}, {"text_only": after}
    ) == {"text_only": {"observation": after["observation"]}}


def test_history_reingestion_rejects_changed_unavailable_evidence(tmp_path):
    root = tmp_path / "store"
    _write_family(root)
    change = _change()
    change["observation"] = {"unavailable": {
        "code": "vllm_rust_parser_not_registered", "detail": "original explanation",
    }}
    _write_capture(root, "0.25.1", {"text_only": change}, implementation="vllm_rust")
    loose = tmp_path / "loose"
    unified_history.materialize_store(root, loose)
    path = loose / "vllm_rust-0.25.1/gemma4/UNIFIED.1-1.yaml"
    document = unified_history.load_yaml(path)
    record = document["cases"]["UNIFIED.1-1"]
    record["unavailable"] = "different explanation"
    record["capture_observation"]["unavailable"]["detail"] = "different explanation"
    path.write_text(unified_history.dump_yaml(document))
    with pytest.raises(ValueError, match="capture is immutable"):
        unified_history.update_store_from_loose(root, loose, complete_snapshot=False)


def test_history_rejects_a_partial_first_observation(tmp_path):
    root = tmp_path
    _write_family(root)
    _write_capture(root, "0.5.0", {"text_only": {"case_key": "UNIFIED.1-1"}})

    with pytest.raises(ValueError, match="first capture change must provide a complete record"):
        unified_history.load_store(root)


def test_history_compacts_repeated_observations_when_a_capture_is_reingested(tmp_path):
    root = _store(tmp_path / "store")
    _write_capture(root, "0.5.2", {"text_only": _change()})
    loose = tmp_path / "loose"
    unified_history.materialize_store(root, loose)

    changed = unified_history.update_store_from_loose(root, loose, complete_snapshot=True)

    assert changed == [root / "families/gemma4/dynamo_v2-0.5.2.yaml"]
    document = unified_history.load_yaml(
        root / "families/gemma4/dynamo_v2-0.5.2.yaml"
    )
    assert document["changes"] == {}
    resolved = unified_history.load_store(root).histories[("gemma4", "dynamo_v2")].resolve(
        "dynamo_v2-0.5.2"
    )["text_only"]
    assert resolved["observation"] == _change()["observation"]
    assert resolved["document"]["captured_with"] == {"dynamo_v2": "0.5.2"}
    assert "inherited_from" not in resolved["document"]


@pytest.mark.parametrize("operation", ["rewrite", "reingest", "exclude"])
@pytest.mark.parametrize("unavailable", [False, True])
@pytest.mark.parametrize("captured", [False, True])
def test_history_compaction_preserves_complete_resolved_snapshots(
    tmp_path, operation, unavailable, captured,
):
    root = tmp_path / "store"
    _write_family(root)
    for index, version in enumerate(("0.25.1", "0.26.0")):
        change = _change()
        if unavailable:
            change["observation"] = {"unavailable": {
                "code": "vllm_rust_parser_not_registered",
                "detail": f"No parser at {version}.",
            }}
        provenance = None
        if captured:
            provenance = {"status": "captured", "origin": {
                "crate_version": version,
                "source_sha256": str(index + 1) * 64,
                "git_commit": str(index + 1) * 40,
            }}
        _write_capture(
            root, version, {"text_only": change}, implementation="vllm_rust",
            provenance=provenance, metadata_changes={"text_only": {"attempt": index}},
            document_overrides={"text_only": {"parser_path": f"parser-{index}"}},
        )
    history = unified_history.load_store(root).histories[("gemma4", "vllm_rust")]
    before = {capture_id: history.resolve(capture_id) for capture_id in history.ordered_capture_ids()}

    if operation == "rewrite":
        unified_history.rewrite_store(root)
    else:
        loose = tmp_path / "loose"
        unified_history.materialize_store(root, loose)
        unified_history.update_store_from_loose(
            root, loose, complete_snapshot=False,
            excluded_capture_dirs={"vllm_rust-0.26.0"} if operation == "exclude" else frozenset(),
        )

    history = unified_history.load_store(root).histories[("gemma4", "vllm_rust")]
    assert {capture_id: history.resolve(capture_id) for capture_id in history.ordered_capture_ids()} == before
    later = history.captures["vllm_rust-0.26.0"]
    assert later["changes"] == ({"text_only": {"observation": before["vllm_rust-0.26.0"]["text_only"]["observation"]}} if unavailable else {})
    assert unified_history.rewrite_store(root) is None
    again = unified_history.load_store(root).histories[("gemma4", "vllm_rust")]
    assert {capture_id: again.resolve(capture_id) for capture_id in again.ordered_capture_ids()} == before


def test_history_older_insertion_preserves_later_snapshots_and_new_metadata(tmp_path):
    _write_family(tmp_path)
    for version in ("0.26.0", "0.27.1"):
        change = _change()
        change["observation"] = {"unavailable": {
            "code": "vllm_rust_parser_not_registered", "detail": f"No parser at {version}.",
        }}
        _write_capture(tmp_path, version, {"text_only": change}, implementation="vllm_rust")
    history = unified_history.load_store(tmp_path).histories[("gemma4", "vllm_rust")]
    before = {capture_id: history.resolve(capture_id) for capture_id in history.ordered_capture_ids()}
    record = copy.deepcopy(before["vllm_rust-0.26.0"]["text_only"])
    record["document"] = {"mode": "unified", "captured_with": {"vllm_rust": "0.25.1"},
                          "record_metadata": {"attempt": 1}, "parser_path": "Combined"}
    assert unified_history._insert_historical_capture(
        history, "vllm_rust-0.25.1", "0.25.1", {"text_only": record},
    )
    assert history.resolve("vllm_rust-0.25.1") == {"text_only": record}
    assert {capture_id: history.resolve(capture_id) for capture_id in before} == before
    for capture_id in history.ordered_capture_ids():
        history.capture_path(capture_id).write_text(
            unified_history.dump_yaml(unified_history._capture_document(history, capture_id)),
        )
    reloaded = unified_history.load_store(tmp_path).histories[("gemma4", "vllm_rust")]
    assert reloaded.resolve("vllm_rust-0.25.1") == {"text_only": record}
    assert {capture_id: reloaded.resolve(capture_id) for capture_id in before} == before


@pytest.mark.parametrize("parser_path", [0, [], {}])
def test_history_parser_path_tombstone_rejects_other_non_string_values(tmp_path, parser_path):
    _write_family(tmp_path)
    _write_capture(
        tmp_path, "0.25.1", {"text_only": _change()},
        document_overrides={"text_only": {"parser_path": parser_path}},
    )
    with pytest.raises(ValueError, match="parser_path must be a string or null"):
        unified_history.load_store(tmp_path)


def test_history_bound_addition_preserves_later_snapshots_and_new_metadata(tmp_path):
    _write_family(tmp_path)
    for version in ("0.25.1", "0.26.0"):
        change = _change()
        change["observation"] = {"unavailable": {
            "code": "vllm_rust_parser_not_registered", "detail": f"No parser at {version}.",
        }}
        _write_capture(tmp_path, version, {"text_only": change}, implementation="vllm_rust")
    _add_backfill_case(tmp_path)
    history = unified_history.load_store(tmp_path).histories[("gemma4", "vllm_rust")]
    before = {capture_id: history.resolve(capture_id) for capture_id in history.ordered_capture_ids()}
    record = _change("new")
    record["case_key"] = "UNIFIED.1-2"
    record["stimulus"]["semantic_sha256"] = unified_history._request_digest(_request("new"))
    record["document"] = {"mode": "unified", "captured_with": {"vllm_rust": "0.25.1"},
                          "record_metadata": {"attempt": 1}, "parser_path": "Combined"}
    assert unified_history._add_bound_legacy_cases(
        history, "vllm_rust-0.25.1", {"new_case": record}, ["new_case"],
    )
    assert history.resolve("vllm_rust-0.25.1") == {**before["vllm_rust-0.25.1"], "new_case": record}
    assert history.resolve("vllm_rust-0.26.0") == before["vllm_rust-0.26.0"]


def test_history_applies_metadata_and_parser_path_without_replacing_observation(tmp_path):
    root = _store(tmp_path)
    _write_capture(
        root,
        "0.5.3",
        {},
        metadata_changes={"text_only": {"attempt": 2}},
        document_overrides={"text_only": {"parser_path": "unified"}},
    )

    resolved = unified_history.load_store(root).histories[("gemma4", "dynamo_v2")].resolve(
        "dynamo_v2-0.5.3"
    )["text_only"]

    assert resolved["observation"] == _change()["observation"]
    assert resolved["document"]["record_metadata"] == {"attempt": 2}
    assert resolved["document"]["parser_path"] == "unified"
    assert resolved["document"]["inherited_from"] == "0.5.0"


def test_history_preserves_one_origin_for_a_captured_checkpoint(tmp_path):
    root = _store(tmp_path)
    source_sha256 = "a" * 64
    _write_capture(
        root,
        "0.5.3",
        {"text_only": _change("updated")},
        provenance={
            "status": "captured",
            "origin": {
                "crate_version": "0.5.3",
                "source_sha256": source_sha256,
                "git_commit": "b" * 40,
            },
        },
    )

    record = unified_history.load_store(root).histories[("gemma4", "dynamo_v2")].resolve(
        "dynamo_v2-0.5.3"
    )["text_only"]

    assert record["document"]["capture_origin"] == {
        "crate_version": "0.5.3",
        "source_sha256": source_sha256,
        "git_commit": "b" * 40,
    }
    assert "inherited_from" not in record["document"]


def test_history_keeps_the_first_origin_when_an_unchanged_capture_is_rerun(tmp_path):
    root = _store(tmp_path / "store")
    original_origin = {
        "crate_version": "0.5.3",
        "source_sha256": "a" * 64,
        "git_commit": "b" * 40,
    }
    _write_capture(
        root,
        "0.5.3",
        {"text_only": _change("updated")},
        provenance={"status": "captured", "origin": original_origin},
    )
    before = (root / "families/gemma4/dynamo_v2-0.5.3.yaml").read_bytes()
    loose = tmp_path / "loose"
    unified_history.materialize_store(root, loose)
    path = loose / "dynamo_v2-0.5.3/gemma4/UNIFIED.1-1.yaml"
    document = unified_history.load_yaml(path)
    document["capture_origin"] = {
        "crate_version": "0.5.3",
        "source_sha256": "c" * 64,
        "git_commit": "d" * 40,
    }
    path.write_text(unified_history.dump_yaml(document), encoding="utf-8")

    capture_path = root / "families/gemma4/dynamo_v2-0.5.3.yaml"
    assert unified_history.update_store_from_loose(root, loose, complete_snapshot=True) == [
        capture_path
    ]
    compacted = unified_history.load_yaml(capture_path)
    assert compacted["provenance"]["origin"] == original_origin
    assert compacted["changes"] == {
        "text_only": {"observation": _change("updated")["observation"]}
    }
    before = capture_path.read_bytes()
    assert unified_history.update_store_from_loose(root, loose, complete_snapshot=True) == []
    assert capture_path.read_bytes() == before

    document["cases"]["UNIFIED.1-1"]["assembled"] = [{"kind": "text", "text": "changed"}]
    path.write_text(unified_history.dump_yaml(document), encoding="utf-8")
    with pytest.raises(ValueError, match="capture is immutable"):
        unified_history.update_store_from_loose(root, loose, complete_snapshot=True)


def test_schema_v3_backfills_an_older_capture_without_changing_later_results(tmp_path):
    root = _store(tmp_path / "store")
    _write_capture(
        root,
        "0.5.0",
        {"text_only": _change()},
        document_overrides={"text_only": {"parser_path": "split"}},
    )
    _write_capture(
        root,
        "0.5.2",
        {},
    )
    before = unified_history.load_store(root).histories[("gemma4", "dynamo_v2")]
    original_later = before.resolve("dynamo_v2-0.5.2")["text_only"]
    loose = tmp_path / "loose" / "dynamo_v2-0.5.1" / "gemma4"
    loose.mkdir(parents=True)
    source = tmp_path / "before" / "dynamo_v2-0.5.0"
    unified_history.materialize_store(root, tmp_path / "before", include_current_inputs=False)
    document = unified_history.load_yaml(source / "gemma4/UNIFIED.1-1.yaml")
    document["captured_with"] = {"dynamo_v2": "0.5.1"}
    document["capture_origin"] = {
        "crate_version": "0.5.1",
        "source_sha256": "a" * 64,
        "git_commit": "b" * 40,
    }
    document["cases"]["UNIFIED.1-1"]["assembled"] = [
        {"kind": "text", "text": "middle"}
    ]
    (loose / "UNIFIED.1-1.yaml").write_text(
        unified_history.dump_yaml(document), encoding="utf-8"
    )

    unified_history.update_store_from_loose(root, tmp_path / "loose", complete_snapshot=False)

    updated = unified_history.load_store(root).histories[("gemma4", "dynamo_v2")]
    inserted = updated.resolve("dynamo_v2-0.5.1")["text_only"]
    later = updated.resolve("dynamo_v2-0.5.2")["text_only"]
    assert inserted["observation"]["value"]["assembled"] == [
        {"kind": "text", "text": "middle"}
    ]
    assert inserted["document"]["capture_origin"]["crate_version"] == "0.5.1"
    assert later["observation"] == original_later["observation"]
    assert later["stimulus"] == original_later["stimulus"]
    assert later["document"] == original_later["document"]


def test_schema_v3_new_observation_drops_inherited_capture_provenance_override(tmp_path):
    root = _store(tmp_path / "store")
    _write_capture(
        root,
        "0.5.0",
        {"text_only": _change()},
        document_overrides={"text_only": {"parser_path": "split"}},
    )
    loose = tmp_path / "loose" / "dynamo_v2-0.5.1" / "gemma4"
    loose.mkdir(parents=True)
    source = tmp_path / "before" / "dynamo_v2-0.5.0"
    unified_history.materialize_store(root, tmp_path / "before", include_current_inputs=False)
    document = unified_history.load_yaml(source / "gemma4/UNIFIED.1-1.yaml")
    document["captured_with"] = {"dynamo_v2": "0.5.1"}
    document["capture_origin"] = {
        "crate_version": "0.5.1",
        "source_sha256": "a" * 64,
        "git_commit": "b" * 40,
    }
    document["cases"]["UNIFIED.1-1"]["assembled"] = [
        {"kind": "text", "text": "middle"}
    ]
    (loose / "UNIFIED.1-1.yaml").write_text(
        unified_history.dump_yaml(document), encoding="utf-8"
    )
    unified_history.update_store_from_loose(root, tmp_path / "loose", complete_snapshot=False)
    _write_capture(
        root,
        "0.5.3",
        {"text_only": _change("latest")},
        provenance={
            "status": "captured",
            "origin": {
                "crate_version": "0.5.3",
                "source_sha256": "c" * 64,
                "git_commit": "d" * 40,
            },
        },
    )

    latest = unified_history.load_store(root).histories[("gemma4", "dynamo_v2")].resolve(
        "dynamo_v2-0.5.3"
    )["text_only"]

    assert latest["observation"]["value"]["assembled"] == [
        {"kind": "text", "text": "latest"}
    ]
    assert latest["document"]["captured_with"] == {"dynamo_v2": "0.5.3"}
    assert latest["document"]["capture_origin"] == {
        "crate_version": "0.5.3",
        "source_sha256": "c" * 64,
        "git_commit": "d" * 40,
    }
    assert "inherited_from" not in latest["document"]
    assert latest["document"]["parser_path"] == "split"


def test_schema_v3_origin_only_change_inherits_without_a_checkpoint(tmp_path):
    root = _store(tmp_path / "store")
    loose = tmp_path / "loose" / "dynamo_v2-0.5.3" / "gemma4"
    unified_history.materialize_store(root, tmp_path / "loose")
    loose.mkdir(parents=True)
    source = tmp_path / "before" / "dynamo_v2-0.5.0"
    unified_history.materialize_store(root, tmp_path / "before", include_current_inputs=False)
    document = unified_history.load_yaml(source / "gemma4/UNIFIED.1-1.yaml")
    document["captured_with"] = {"dynamo_v2": "0.5.3"}
    document["capture_origin"] = {
        "crate_version": "0.5.3",
        "source_sha256": "c" * 64,
        "git_commit": "d" * 40,
    }
    (loose / "UNIFIED.1-1.yaml").write_text(
        unified_history.dump_yaml(document), encoding="utf-8"
    )

    changed = unified_history.update_store_from_loose(
        root, tmp_path / "loose", complete_snapshot=False
    )

    assert changed == []
    assert not (root / "families/gemma4/dynamo_v2-0.5.3.yaml").exists()
    history = unified_history.load_store(root).histories[("gemma4", "dynamo_v2")]
    assert history.ordered_capture_ids() == ["dynamo_v2-0.5.0", "dynamo_v2-0.5.2"]
    output = tmp_path / "after"
    unified_history.materialize_store(root, output, derived_release_versions={"dynamo_v2": "0.5.3"})
    inherited = unified_history.load_yaml(output / "dynamo_v2-0.5.3/gemma4/UNIFIED.1-1.yaml")
    assert inherited["cases"] == document["cases"]
    assert inherited["inherited_from"] == "0.5.0"
    assert inherited["captured_with"] == {"dynamo_v2": "0.5.0"}
@pytest.mark.parametrize("version", ["0.5.1.patch1", "0.5.1+source." + "a" * 64])
def test_history_rejects_patch_and_source_qualified_filenames(tmp_path, version):
    _write_family(tmp_path)
    _write_capture(tmp_path, version, {"text_only": _change()})

    with pytest.raises(ValueError, match="capture identity differs"):
        unified_history.load_store(tmp_path)


@pytest.mark.parametrize(
    ("filename", "header"),
    [
        ("inputs_and_golden.yaml", {"family": "gemma4"}),
        ("dynamo_v2-0.5.0.yaml", {"implementation": "dynamo_v2"}),
    ],
)
def test_history_rejects_identity_headers_owned_by_the_path(tmp_path, filename, header):
    root = _store(tmp_path)
    path = root / "families/gemma4" / filename
    document = unified_history.load_yaml(path)
    document.update(header)
    path.write_text(unified_history.dump_yaml(document), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown fields"):
        unified_history.load_store(root)


def test_history_accepts_explicit_preceding_parent(tmp_path):
    root = _store(tmp_path)
    path = root / "families/gemma4/dynamo_v2-0.5.2.yaml"
    document = unified_history.load_yaml(path)
    document["parent"] = "dynamo_v2-0.5.0"
    path.write_text(unified_history.dump_yaml(document), encoding="utf-8")

    history = unified_history.load_store(root).histories[("gemma4", "dynamo_v2")]
    assert history.captures["dynamo_v2-0.5.2"]["parent"] == "dynamo_v2-0.5.0"


def test_history_rejects_a_capture_origin_for_another_version(tmp_path):
    _write_family(tmp_path)
    _write_capture(
        tmp_path,
        "0.5.0",
        {"text_only": _change()},
        provenance={
            "status": "captured",
            "origin": {"crate_version": "0.4.9", "source_sha256": "a" * 64},
        },
    )

    with pytest.raises(ValueError, match="capture origin differs"):
        unified_history.load_store(tmp_path)


@pytest.mark.parametrize("corruption", [None, "release", "source", "origin_version"])
def test_source_qualified_record_origin_binds_release_and_source(tmp_path, corruption):
    _write_family(tmp_path)
    origin = {"crate_version": "0.5.0", "source_sha256": "a" * 64}
    if corruption == "origin_version":
        origin["crate_version"] += "+source." + "a" * 64
    release = "0.5.1" if corruption == "release" else "0.5.0"
    source_hash = "b" * 64 if corruption == "source" else "a" * 64
    label = f"{release}+source.{source_hash}"
    _write_capture(tmp_path, "0.5.0", {"text_only": _change()}, document_overrides={
        "text_only": {"capture_provenance": {
            "captured_with": {"dynamo_v2": label}, "capture_origin": origin,
            "inherited_from": None,
        }},
    })
    if corruption:
        with pytest.raises(ValueError, match="capture origin differs"):
            unified_history.load_store(tmp_path)
    else:
        history = unified_history.load_store(tmp_path).histories[("gemma4", "dynamo_v2")]
        document = history.resolve("dynamo_v2-0.5.0")["text_only"]["document"]
        assert document["captured_with"] == {"dynamo_v2": label}
        assert document["capture_origin"] == origin


def test_history_rewrite_is_byte_deterministic(tmp_path):
    root = _store(tmp_path)
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*.yaml")}

    unified_history.rewrite_store(root)
    unified_history.rewrite_store(root)

    assert {path.relative_to(root): path.read_bytes() for path in root.rglob("*.yaml")} == before


def test_history_rewrite_compacts_unchanged_observations(tmp_path):
    root = _store(tmp_path)
    _write_capture(root, "0.5.2", {"text_only": _change()})

    unified_history.rewrite_store(root)

    capture = unified_history.load_yaml(root / "families/gemma4/dynamo_v2-0.5.2.yaml")
    assert capture["changes"] == {}
    resolved = unified_history.load_store(root).histories[("gemma4", "dynamo_v2")].resolve(
        "dynamo_v2-0.5.2"
    )["text_only"]
    assert resolved["observation"] == _change()["observation"]
    assert resolved["document"]["captured_with"] == {"dynamo_v2": "0.5.2"}
    assert "inherited_from" not in resolved["document"]


def test_history_materializes_a_derived_release_view_without_a_checkpoint(tmp_path):
    root = _store(tmp_path / "store")
    loose = tmp_path / "loose"

    unified_history.materialize_store(
        root,
        loose,
        derived_release_versions={"dynamo_v2": "0.6.1"},
    )

    assert not (root / "families/gemma4/dynamo_v2-0.6.1.yaml").exists()
    document = unified_history.load_yaml(loose / "dynamo_v2-0.6.1/gemma4/UNIFIED.1-1.yaml")
    assert document["capture_provenance"] == {"format": "unified_history"}
    assert document["captured_with"] == {"dynamo_v2": "0.5.0"}
    assert document["inherited_from"] == "0.5.0"


def test_extraction_tracks_crate_release_without_repackaging(tmp_path, monkeypatch, capsys):
    import extract_fixtures

    root = _store(tmp_path / "store")
    digest, size = unified_history.store_digest(root)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "snapshot": "20260101_000000",
        "crates": {"dynamo-parsers-v2": "0.5.2"},
        "shards": [{
            "path": "unified-history", "format": "unified-history",
            "sha256": digest, "size": size,
        }],
    }))
    crate = tmp_path / "parsers/v2/Cargo.toml"
    crate.parent.mkdir(parents=True)
    monkeypatch.setattr(extract_fixtures, "ROOT", tmp_path)
    monkeypatch.setattr(extract_fixtures, "HISTORY_DIR", root)
    monkeypatch.setattr(extract_fixtures, "MANIFEST_PATH", manifest)
    monkeypatch.setattr(extract_fixtures, "get_cache_root", lambda: tmp_path / "cache")
    monkeypatch.setattr(sys, "argv", ["extract_fixtures.py"])
    monkeypatch.delenv("CONFORMANCE_DYNAMO_V2_LABEL", raising=False)

    crate.write_text('[package]\nversion = "0.5.2"\n')
    extract_fixtures.main()
    original = Path(capsys.readouterr().out.strip())
    prior = original / "unified/dynamo_v2-0.5.2/gemma4/UNIFIED.1-1.yaml"
    prior_bytes = prior.read_bytes()

    crate.write_text('[package]\nversion = "0.6.1"\n')
    extract_fixtures.main()
    current = Path(capsys.readouterr().out.strip())
    assert current != original
    document = unified_history.load_yaml(
        current / "unified/dynamo_v2-0.6.1/gemma4/UNIFIED.1-1.yaml"
    )
    assert document["inherited_from"] == "0.5.0"
    assert prior.read_bytes() == prior_bytes
    assert unified_history.store_digest(root) == (digest, size)

    extract_fixtures.main()
    cached = capsys.readouterr()
    assert Path(cached.out.strip()) == current
    assert "Cache hit" in cached.err


def test_history_derived_release_view_is_not_reingested_as_a_checkpoint(tmp_path):
    root = _store(tmp_path / "store")
    loose = tmp_path / "loose"
    family_path = root / "families/gemma4/inputs_and_golden.yaml"
    family = unified_history.load_yaml(family_path)
    family["cases"]["text_only"]["display_id"] = "UNIFIED.1-2"
    family["cases"]["text_only"]["historical_ids"] = ["UNIFIED.1-1"]
    family["cases"]["text_only"]["request"]["init"] = {
        "starting_state": "None",
        "tool_output_mode": "Native",
        "named_tool": None,
    }
    family_path.write_text(unified_history.dump_yaml(family), encoding="utf-8")
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*.yaml")}

    unified_history.materialize_store(
        root,
        loose,
        derived_release_versions={"dynamo_v2": "0.6.1"},
    )
    changed = unified_history.update_store_from_loose(root, loose, complete_snapshot=True)

    assert changed == []
    assert {path.relative_to(root): path.read_bytes() for path in root.rglob("*.yaml")} == before
    assert not (root / "families/gemma4/dynamo_v2-0.6.1.yaml").exists()


def test_schema_v3_ignores_generated_oracle_directories_but_rejects_malformed_captures(tmp_path):
    root = _store(tmp_path / "store")
    loose = tmp_path / "loose"
    unified_history.materialize_store(root, loose)
    (loose / "golden_spec-1271640-0").mkdir()

    assert unified_history.update_store_from_loose(root, loose, complete_snapshot=True) == []

    (loose / "dynamo_v2-0.5").mkdir()
    with pytest.raises(ValueError, match="Unified capture directories must use"):
        unified_history.update_store_from_loose(root, loose, complete_snapshot=True)


def test_history_request_rewrite_directs_contributors_to_latest_tests(tmp_path):
    root = _store(tmp_path / "store")
    loose = tmp_path / "loose"
    unified_history.materialize_store(root, loose)
    input_path = loose / "inputs/gemma4/UNIFIED.1-1.yaml"
    document = unified_history.load_yaml(input_path)
    document["cases"]["UNIFIED.1-1"]["input"] = "changed request"
    input_path.write_text(unified_history.dump_yaml(document), encoding="utf-8")

    with pytest.raises(ValueError, match="Run latest parser tests.*repository administrator"):
        unified_history.sync_current_corpus(root, loose, complete_snapshot=True)


def test_materialized_null_labels_follow_stable_owners_after_number_swap(tmp_path):
    root = tmp_path / "store"
    _write_family(root)
    path = root / "families/gemma4/inputs_and_golden.yaml"
    doc = unified_history.load_yaml(path)
    template = doc["cases"]["text_only"]
    doc["cases"] = {}
    changes = {}
    for owner, current, historical, value in (
        ("arg_json_null", "UNIFIED.7-4", "UNIFIED.7-5", None),
        ("arg_string_null", "UNIFIED.7-5", "UNIFIED.7-4", "null"),
    ):
        event = {"kind": "tool_call", "name": "get_weather", "arguments": {"city": value}}
        case = copy.deepcopy(template)
        case.update(scenario=owner, display_id=current, golden={"assembled": [event]})
        doc["cases"][owner] = case
        change = _change()
        change["case_key"] = historical
        change["observation"]["value"] = {"assembled": [event], "chunks": [{"expected": [event]}]}
        changes[owner] = change
    path.write_text(unified_history.dump_yaml(doc))
    capture_path = _write_capture(root, "0.7.4", changes)
    before = capture_path.read_bytes()
    out = tmp_path / "view"
    unified_history.materialize_store(root, out)
    for label, value in (("7-4", None), ("7-5", "null")):
        display = "UNIFIED." + label
        capture = unified_history.load_yaml(out / "dynamo_v2-0.7.4/gemma4" / (display + ".yaml"))
        assert capture["cases"][display]["assembled"][0]["arguments"]["city"] == value
    assert capture_path.read_bytes() == before


@pytest.mark.parametrize(("family", "original_digests"), [
    ("gemma4", ("4ab8dacd248094c3a57d64002663fa8852fb589343ab51ed99d6821a20941c15",
                "f6995d2089b187275dfd0645d98acbb38d60d9379f65ea52c64893553c23a8a7")),
    ("kimi_k2", ("96c861d60198686d18f2803dce68fb6322caf71def79a35fc1aa00f0441cd08e",
                 "95741377967864cd2b0bd55e24c7b430cb1641a0e63f2d751b9a7b505312650f")),
    ("qwen3", ("be1520f44605ed60b906aa03f7847a9210631d43cec8c967f55ac99186d496bd",
               "697ee3feadda33be06413278e6f213f8361a060793518381713cc19a771c2f4d")),
])
def test_vllm_rust_guided_json_recaptures_preserve_unbound_history(tmp_path, family, original_digests, history_source):
    store_root = history_source / "conformance/fixtures-unified-v2"
    source = store_root / "families" / family
    destination = tmp_path / "families" / family
    destination.mkdir(parents=True)
    # Sparse snapshots can depend on every preceding checkpoint for this history.
    with unified_history._store_read_lock(store_root), unified_archives.expanded(store_root) as expanded_root:
        source = expanded_root / "families" / family
        for path in [source / "inputs_and_golden.yaml", *sorted(source.glob("vllm_rust-*.yaml"))]:
            shutil.copyfile(path, destination / path.name)
            reference = unified_history.load_yaml(path).get("execution")
            if reference is not None:
                manifest_path = expanded_root / reference
                manifest = unified_history.load_yaml(manifest_path)
                dependencies = [manifest_path, *(expanded_root / digest for digest in manifest.get("receipts", {}))]
                for dependency in dependencies:
                    target = tmp_path / dependency.relative_to(expanded_root)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(dependency, target)
    store = unified_history.load_store(tmp_path)
    history = store.histories[(family, "vllm_rust")]
    historical = history.resolve("vllm_rust-0.25.1")
    recaptured = history.resolve("vllm_rust-0.26.0")
    case_ids = [
        case_id
        for case_id, record in recaptured.items()
        if record["document"].get("record_metadata", {}).get("verified_recapture", {}).get("resolution")
        == "historical_request_not_retained"
        and "unavailable" in record["document"]["record_metadata"]["verified_recapture"]["result"]
    ]

    assert len(case_ids) == 39
    # Oracle 78b53357 retains different chunk schedules at these two releases;
    # comparing them to each other would bless the loss of version-specific evidence.
    for version, expected in zip(("0.25.1", "0.26.0"), original_digests, strict=True):
        snapshot = history.resolve(f"vllm_rust-{version}")
        observations = {case_id: snapshot[case_id]["observation"] for case_id in case_ids}
        encoded = json.dumps(observations, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        assert hashlib.sha256(encoded).hexdigest() == expected
    for case_id in case_ids:
        original = historical[case_id]
        current = recaptured[case_id]
        assert original["stimulus"] == {"unavailable": {"code": "original_request_not_retained"}}
        assert next(iter(original["observation"])) in {"value", "error"}
        assert current["stimulus"] == original["stimulus"]
        assert next(iter(current["observation"])) in {"value", "error"}
        case = store.families[family].cases[case_id]
        record, document = unified_history.materialized_record(case, current)
        record.update(document["record_metadata"])
        selected = select_peer_observation(
            record,
            case["request"],
            implementation="vllm_rust",
            version="0.26.0",
        )
        typed_unavailable = selected["unavailable"]
        assert typed_unavailable["code"].startswith("vllm_rust_")
        normalized = normalize_unavailable(selected)
        assert normalized["unavailable"] == typed_unavailable["detail"]
        assert normalized["unavailable_code"] == "unsupported_request_mode"
        assert "original observation is preserved in YAML" in selected["historical_capture_note"]


@pytest.mark.parametrize("implementation", ["vllm_python", "vllm_rust", "sglang_python"])
@pytest.mark.parametrize("observation_version", ["0.24.0", "0.25.1"])
def test_sparse_checkpoint_does_not_inherit_version_scoped_recapture(tmp_path, implementation, observation_version):
    _write_family(tmp_path)
    change = _change("old output")
    change["stimulus"] = {"unavailable": {"code": "original_request_not_retained"}}
    _write_capture(tmp_path, observation_version, {"text_only": change}, implementation=implementation)
    supplement = {
        "result": {"capture_input": _request(), "assembled": [{"kind": "text", "text": "hello"}],
                   "chunks": [{"expected": [{"kind": "text", "text": "hello"}]}]},
        "provenance": {"implementation": implementation, "version": "0.25.1",
                       "source_sha256": "a" * 64, "adapter_sha256": "b" * 64, "job_sha256": "c" * 64},
        "resolution": "historical_request_not_retained",
    }
    _write_capture(tmp_path, "0.25.1", {"text_only": change} if observation_version == "0.25.1" else {},
                   implementation=implementation,
                   metadata_changes={"text_only": {"verified_recapture": supplement, "audit_note": "preserve me"}})
    history = unified_history.load_store(tmp_path).histories[("gemma4", implementation)]
    prior = history.resolve(f"{implementation}-0.25.1")
    derived = tmp_path.parent / f"{tmp_path.name}-derived"
    unified_history.materialize_store(
        tmp_path, derived, derived_release_versions={implementation: "0.26.0"},
    )
    derived_paths = list((derived / f"{implementation}-0.26.0/gemma4").glob("*.yaml"))
    assert len(derived_paths) == 1
    derived_document = unified_history.load_yaml(derived_paths[0])
    derived_record = next(iter(derived_document["cases"].values()))
    assert "verified_recapture" not in derived_record
    assert derived_record["audit_note"] == "preserve me"
    selected = select_peer_observation(
        derived_record, _request(), implementation=implementation, version="0.26.0",
    )
    assert selected["assembled"] == [{"kind": "text", "text": "old output"}]
    _write_capture(tmp_path, "0.26.0", {}, implementation=implementation)
    _write_capture(tmp_path, "0.27.0", {}, implementation=implementation)
    history = unified_history.load_store(tmp_path).histories[("gemma4", implementation)]
    assert history.resolve(f"{implementation}-0.25.1") == prior
    for version in ["0.25.1", "0.26.0", "0.27.0"]:
        resolved = history.resolve(f"{implementation}-{version}")["text_only"]
        assert resolved["observation"] == prior["text_only"]["observation"]
        assert resolved["stimulus"] == prior["text_only"]["stimulus"]
        metadata = resolved["document"]["record_metadata"]
        assert metadata["audit_note"] == "preserve me"
        assert ("verified_recapture" in metadata) == (version == "0.25.1")
        record, document = unified_history.materialized_record(history.family.cases["text_only"], resolved)
        record.update(document["record_metadata"])
        selected = select_peer_observation(record, _request(), implementation=implementation, version=version)
        expected = [{"kind": "text", "text": "hello"}] if version == "0.25.1" else [{"kind": "text", "text": "old output"}]
        assert selected["assembled"] == expected
    destination = tmp_path / "materialized"
    unified_history.materialize_store(tmp_path, destination, store=unified_history.load_store(tmp_path))
    later_files = list((destination / f"{implementation}-0.26.0/gemma4").glob("*.yaml"))
    assert len(later_files) == 1
    later_document = unified_history.load_yaml(later_files[0])
    later_record = next(iter(later_document["cases"].values()))
    assert "verified_recapture" not in later_record


@pytest.mark.parametrize("implementation", ["vllm_python", "vllm_rust", "sglang_python"])
@pytest.mark.parametrize("difference", ["finish", "assembled", "error", "unchanged"])
def test_restore_historical_snapshot_preserves_later_execution_and_additions(tmp_path, implementation, difference):
    _write_family(tmp_path)
    original = _change("original")
    original["stimulus"] = {"unavailable": {"code": "original_request_not_retained"}}
    _write_capture(tmp_path, "0.25.1", {"text_only": original}, implementation=implementation)
    _write_capture(tmp_path, "0.26.0", {}, implementation=implementation)
    history = unified_history.load_store(tmp_path).histories[("gemma4", implementation)]
    oracle = history.resolve(f"{implementation}-0.25.1")
    execution = copy.deepcopy(oracle)
    if difference == "finish":
        execution["text_only"]["observation"]["value"]["chunks"].append({"expected": []})
    elif difference == "assembled":
        execution["text_only"]["observation"]["value"]["assembled"] = [{"kind": "text", "text": "changed"}]
    elif difference == "error":
        execution["text_only"]["observation"] = {"error": {"code": "parser_error", "detail": "changed error"}}
    execution["text_only"]["stimulus"] = {"ref": "current"}
    unified_history._replace_capture_snapshot(history, f"{implementation}-0.25.1", execution)
    _add_backfill_case(tmp_path)
    history.family = unified_history.load_store(tmp_path).families["gemma4"]
    addition = _change("new")
    addition["case_key"] = "UNIFIED.1-2"
    addition["stimulus"]["semantic_sha256"] = unified_history._request_digest(_request("new"))
    addition["document"] = {"mode": "unified", "captured_with": {implementation: "0.25.1"}}
    execution["new_case"] = addition
    unified_history._replace_capture_snapshot(history, f"{implementation}-0.25.1", execution)
    later = history.resolve(f"{implementation}-0.26.0")
    restored = {**copy.deepcopy(execution), **oracle}
    unified_history._replace_capture_snapshot(history, f"{implementation}-0.25.1", restored)
    assert history.resolve(f"{implementation}-0.25.1") == restored
    assert restored["text_only"] == oracle["text_only"]
    assert restored["new_case"] == addition
    assert history.resolve(f"{implementation}-0.26.0") == later
    for capture_id in history.ordered_capture_ids():
        history.capture_path(capture_id).write_text(
            unified_history.dump_yaml(unified_history._capture_document(history, capture_id)),
        )
    reloaded = unified_history.load_store(tmp_path).histories[("gemma4", implementation)]
    assert reloaded.resolve(f"{implementation}-0.25.1") == restored
    assert reloaded.resolve(f"{implementation}-0.26.0") == later


@pytest.mark.parametrize('corruption', [None, 'deduplicated_origin', 'missing_member', 'changed_digest', 'unproved_record', 'partial_bad', 'partial_hex', 'empty_execution'])
def test_frozen_snapshot_proof_keeps_its_executed_members_after_authored_growth(tmp_path, corruption):
    root = tmp_path / 'store'
    _write_family(root)
    path = _write_capture(root, '0.26.0', {'text_only': _change()})
    store = unified_history.load_store(root)
    family = store.families['gemma4']
    history = store.histories[('gemma4', 'dynamo_v2')]
    records = history.resolve('dynamo_v2-0.26.0')
    origin = {'crate_version': '0.26.0', 'source_sha256': 'a' * 64,
              'adapter_sha256': 'b' * 64, 'job_sha256': 'c' * 64}
    record = records['text_only']
    record['document']['capture_origin'] = copy.deepcopy(origin)
    record['stimulus']['semantic_sha256'] = unified_history._request_digest(family.cases['text_only']['request'])
    proof = {'receipt': {**origin, 'implementation': 'dynamo_v2', 'version': '0.26.0', 'results_sha256': 'd' * 64},
             'requests': {'text_only': record['stimulus']['semantic_sha256']}}
    snapshot = {'schema_version': 2, 'provenance': {'status': 'captured', 'origin': origin}, 'document': {},
                'records': {'text_only': {'value': record, 'member': 'conformance/fixtures-unified-v2/families/gemma4/dynamo_v2-0.26.0.yaml'}},
                'execution_proof': proof}
    family.cases['later'] = copy.deepcopy(family.cases['text_only'])
    if corruption == 'deduplicated_origin':
        record['document']['capture_origin']['source_sha256'] = 'f' * 64
    elif corruption in {'partial_bad', 'partial_hex'}:
        digest = 'bad' if corruption == 'partial_bad' else 'e' * 64
        record['stimulus'] = {'partial': {'input': 'x'}, 'semantic_sha256': digest}
        proof['requests']['text_only'] = digest
    elif corruption == 'empty_execution':
        snapshot['records'].clear()
        proof['requests'].clear()
    elif corruption == 'missing_member':
        proof['requests'].clear()
    elif corruption == 'changed_digest':
        proof['requests']['text_only'] = 'e' * 64
    elif corruption == 'unproved_record':
        snapshot['records']['later'] = copy.deepcopy(snapshot['records']['text_only'])
    if corruption and corruption != 'deduplicated_origin':
        with pytest.raises(ValueError, match='(proof request membership mismatch|sealed request digests|complete request)'):
            unified_history._validate_complete_snapshot(path, snapshot, family, 'dynamo_v2', '0.26.0')
    else:
        unified_history._validate_complete_snapshot(path, snapshot, family, 'dynamo_v2', '0.26.0')


def test_retained_kimi_sparse_history_preserves_observation_origins(history_source):
    root = history_source / 'conformance/fixtures-unified-v2'
    store = unified_history.load_store(root)
    history = store.histories['kimi_k3', 'sglang_python_batch']
    capture_id = max(history.captures, key=lambda name: unified_history._capture_release_sort_key(name.removeprefix('sglang_python_batch-')))
    snapshot = history.complete_snapshot(capture_id)
    assert snapshot['records']
    members = history.snapshot_members[capture_id]
    assert set(snapshot['records']) == set(members)
    assert any(not unified_history.typed_equal(record['value']['document']['capture_origin'], snapshot['provenance']['origin'])
               for record in snapshot['records'].values())
    resolved = history.resolve(capture_id)
    for case, wrapped in snapshot['records'].items():
        assert wrapped['member'] == members[case]
        assert wrapped['value'] == resolved[case]
    family = store.families['kimi_k3']
    version = capture_id.removeprefix('sglang_python_batch-')
    unified_history._validate_complete_snapshot(root / f'families/kimi_k3/{capture_id}.yaml', snapshot, family, 'sglang_python_batch', version)


def test_routed_sparse_history_keeps_requests_precedence_and_lazy_fragments(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    root = _store(checkout / "conformance/fixtures-unified-v2")
    original = unified_history.load_store(root)
    authored = root / "families/gemma4/inputs_and_golden.yaml"
    authored_bytes = authored.read_bytes()
    for path in authored.parent.glob("dynamo*.yaml"):
        path.unlink()
    cache = tmp_path / "cache"
    cache.mkdir()
    datasets = []
    for dataset, version in [("imported", "0.5.0"), ("imported", "0.5.2"), ("base", "0.5.0")]:
        prefix = f"conformance/history-datasets/{dataset}"
        retained = checkout / prefix / "conformance/fixtures-unified-v2/families/gemma4/inputs_and_golden.yaml"
        retained.parent.mkdir(parents=True, exist_ok=True)
        retained.write_bytes(authored_bytes)
        history = original.histories["gemma4", "dynamo_v2"]
        capture_id = f"dynamo_v2-{version}"
        records = history.resolve(capture_id)
        records["text_only"]["stimulus"] = {"inline": _request(dataset)}
        wrapped = {case: {"value": record, "member": f"conformance/fixtures-unified-v2/families/gemma4/{capture_id}.yaml"}
                   for case, record in records.items()}
        payload = encode_typed(wrapped)
        key = hashlib.sha256((dataset + version).encode()).hexdigest()
        data = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        (cache / f"{key}.json").write_text(json.dumps({"identity": key, "sha256": hashlib.sha256(data).hexdigest(), "payload": payload}))
        descriptor = next((row for row in datasets if row["id"] == dataset), None)
        if descriptor is None:
            descriptor = {"id": dataset, "root": prefix, "checkpoints": []}
            datasets.append(descriptor)
        capture = history.captures[capture_id]
        descriptor["checkpoints"].append({"family": "gemma4", "implementation": "dynamo_v2", "version": version,
            "fragment": key, "capture": {field: encode_typed(capture[field]) for field in ("provenance", "document")}})
    (checkout / "conformance/history-datasets.json").write_text(json.dumps([{key: row[key] for key in ("id", "root")} for row in datasets]))
    (checkout / "conformance/viewer-catalog.json").write_text(json.dumps({"sources": {"history_datasets": datasets}}))
    (checkout / "conformance/history-reader-cache.json").write_text(json.dumps({"directory": str(cache)}))
    store = unified_history.load_store(root)
    history = store.histories["gemma4", "dynamo_v2"]
    assert history.resolve("dynamo_v2-0.5.0")["text_only"]["stimulus"]["inline"]["input"] == "base"
    assert history.resolve("dynamo_v2-0.5.2")["text_only"]["stimulus"]["inline"]["input"] == "imported"
    assert "/base/" in history.snapshot_members["dynamo_v2-0.5.0"]["text_only"]
    selected, targets = unified_history.selected_materialization_store(root, ["dynamo_v2-0.5.1"])
    assert targets == {"dynamo_v2": "0.5.1"}
    assert set(selected.histories) == set(store.histories)
    @contextmanager
    def retained_view(_root, _history):
        yield checkout

    monkeypatch.setattr(capture_sources, "history_view", retained_view)
    monkeypatch.setattr(capture_sources, "verify", lambda *_: {"files": []})
    monkeypatch.setattr(capture_sources, "history_pin", lambda *_: {})
    monkeypatch.setattr(migrate_capture_sources, "convert", lambda *_args, **_kwargs: {"rows": []})
    retained_history = tmp_path / "history"
    (retained_history / capture_sources.DATASETS).parent.mkdir(parents=True)
    (retained_history / capture_sources.DATASETS).write_bytes(capture_sources.encoded([]))
    publish_current_capture.reconcile(checkout, retained_history)
    current = capture_sources.current(checkout)["captures"][0]
    latest = unified_history.load_yaml(checkout / current["path"])
    assert current["version"] == "0.5.2"
    assert latest["records"]["text_only"]["member"] == current["source"]
    assert latest["records"]["text_only"]["value"]["stimulus"]["inline"]["input"] == "imported"
    # Corruption is detected at demand time, including after a successful read.
    row = datasets[-1]["checkpoints"][0]
    (cache / (row["fragment"] + ".json")).write_text('{}')
    with pytest.raises((ValueError, KeyError)):
        history.resolve("dynamo_v2-0.5.0")
