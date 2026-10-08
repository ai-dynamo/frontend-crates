# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coverage for canonical Unified capture checkpoints."""

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import unified_history


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


def _store(root: Path) -> Path:
    _write_family(root)
    _write_capture(root, "0.5.0", {"text_only": _change()})
    _write_capture(root, "0.5.2", {})
    return root


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
    ) == {"text_only": {"case_key": "UNIFIED.1-2"}}


def test_history_omits_version_only_unregistered_peer_detail():
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
    ) == {}

    after["observation"]["unavailable"]["code"] = "vllm_tool_schema_unsupported"
    assert unified_history._capture_changes(
        {"text_only": before}, {"text_only": after}
    ) == {"text_only": {"observation": after["observation"]}}


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
    assert resolved["document"]["inherited_from"] == "0.5.0"


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


def test_schema_v3_records_source_origin_for_unchanged_output(tmp_path):
    root = _store(tmp_path / "store")
    loose = tmp_path / "loose" / "dynamo_v2-0.5.3" / "gemma4"
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

    capture_path = root / "families/gemma4/dynamo_v2-0.5.3.yaml"
    assert changed == [capture_path]
    history = unified_history.load_store(root).histories[("gemma4", "dynamo_v2")]
    assert history.captures["dynamo_v2-0.5.3"]["provenance"]["origin"] == {
        "crate_version": "0.5.3",
        "source_sha256": "c" * 64,
        "git_commit": "d" * 40,
    }


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
    assert resolved["document"]["inherited_from"] == "0.5.0"


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


def test_schema_v3_retains_a_new_captured_version_when_all_results_match(tmp_path):
    root = _store(tmp_path / "store")
    loose = tmp_path / "loose"
    unified_history.materialize_store(root, loose)
    source = loose / "dynamo_v2-0.5.2/gemma4/UNIFIED.1-1.yaml"
    document = unified_history.load_yaml(source)
    document["captured_with"] = {"dynamo_v2": "0.5.3"}
    document["capture_origin"] = {
        "crate_version": "0.5.3",
        "source_sha256": "a" * 64,
        "git_commit": "b" * 40,
    }
    target = loose / "dynamo_v2-0.5.3/gemma4/UNIFIED.1-1.yaml"
    target.parent.mkdir(parents=True)
    target.write_text(unified_history.dump_yaml(document), encoding="utf-8")

    changed = unified_history.update_store_from_loose(root, loose, complete_snapshot=True)

    history = unified_history.load_store(root).histories[("gemma4", "dynamo_v2")]
    assert changed == [history.capture_path("dynamo_v2-0.5.3")]
    assert history.captures["dynamo_v2-0.5.3"]["provenance"] == {
        "status": "captured",
        "origin": {
        "crate_version": "0.5.3",
        "source_sha256": "a" * 64,
        "git_commit": "b" * 40,
        },
    }


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
