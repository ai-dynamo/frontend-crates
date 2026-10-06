# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coverage for canonical Unified capture checkpoints."""

import copy
from dataclasses import replace
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import unified_history
from capture_stimulus import select_peer_observation
import generate_conformance_table as table
import markers
from tables.reasoning import table as reasoning_table


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


def test_structured_parser_error_survives_store_and_view_projection(tmp_path):
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
    for reader in (table._output_block_model, reasoning_table._reasoning_output_model):
        projected = reader(record)
        assert projected["error"] == detail
        assert markers._error_kind(projected) == "parser_error"
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
    assert table._unified_capture_failure(record)["unavailable_code"] == "unsupported_request_mode"
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


def test_history_rejects_parent_chain_fields(tmp_path):
    root = _store(tmp_path)
    path = root / "families/gemma4/dynamo_v2-0.5.2.yaml"
    document = unified_history.load_yaml(path)
    document["parent"] = "dynamo_v2-0.5.0"
    path.write_text(unified_history.dump_yaml(document), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown fields"):
        unified_history.load_store(root)


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


def test_history_rejects_input_changes_without_recapturing_prior_versions(tmp_path):
    root = _store(tmp_path / "store")
    loose = tmp_path / "loose"
    unified_history.materialize_store(root, loose)
    input_path = loose / "inputs/gemma4/UNIFIED.1-1.yaml"
    document = unified_history.load_yaml(input_path)
    document["cases"]["UNIFIED.1-1"]["input"] = "changed request"
    input_path.write_text(unified_history.dump_yaml(document), encoding="utf-8")

    with pytest.raises(ValueError, match="requires recapturing and updating every prior semantic version"):
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
def test_vllm_rust_guided_json_recaptures_preserve_unbound_history(family, original_digests, committed_unified_store):
    store = committed_unified_store
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
        assert selected["unavailable"]["code"].startswith("vllm_rust_")
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
