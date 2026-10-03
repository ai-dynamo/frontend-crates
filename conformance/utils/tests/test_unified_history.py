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
    _write_capture(root, "0.5.0", {"text_only": _change()}, implementation="vllm_python")
    _add_backfill_case(root)
    loose = tmp_path / "loose"
    unified_history.materialize_store(root, loose)
    _add_loose_backfill_record(loose, "0.5.0", "captured", implementation="vllm_python")

    with pytest.raises(ValueError, match="unrecorded cases: new_case"):
        unified_history.update_store_from_loose(
            root,
            loose,
            complete_snapshot=False,
            excluded_capture_dirs={"vllm_python-0.5.0"},
        )


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


def test_history_rejects_replacement_of_an_unbound_legacy_capture(tmp_path):
    root = _store(tmp_path / "store")
    first = unified_history.load_yaml(root / "families/gemma4/dynamo_v2-0.5.0.yaml")
    first["changes"]["text_only"]["stimulus"] = {
        "unavailable": {"code": "request_not_retained"},
    }
    first["changes"]["text_only"]["observation"]["value"]["assembled"] = [
        {"kind": "text", "text": "old"}
    ]
    (root / "families/gemma4/dynamo_v2-0.5.0.yaml").write_text(
        unified_history.dump_yaml(first), encoding="utf-8"
    )
    later = unified_history.load_yaml(root / "families/gemma4/dynamo_v2-0.5.2.yaml")
    later["changes"] = {"text_only": _change("later")}
    (root / "families/gemma4/dynamo_v2-0.5.2.yaml").write_text(
        unified_history.dump_yaml(later), encoding="utf-8"
    )
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*.yaml")}

    loose = tmp_path / "loose"
    unified_history.materialize_store(root, loose)
    path = loose / "dynamo_v2-0.5.0/gemma4/UNIFIED.1-1.yaml"
    document = unified_history.load_yaml(path)
    record = document["cases"]["UNIFIED.1-1"]
    record["assembled"] = [{"kind": "text", "text": "recaptured"}]
    record["capture_input"] = _request()
    path.write_text(unified_history.dump_yaml(document), encoding="utf-8")

    with pytest.raises(ValueError, match="capture is immutable"):
        unified_history.update_store_from_loose(root, loose, complete_snapshot=True)
    assert {path.relative_to(root): path.read_bytes() for path in root.rglob("*.yaml")} == before


def test_history_does_not_replace_a_legacy_parser_error(tmp_path):
    root = _store(tmp_path / "store")
    first_path = root / "families/gemma4/dynamo_v2-0.5.0.yaml"
    first = unified_history.load_yaml(first_path)
    first["changes"]["text_only"]["observation"] = {
        "error": {"code": "vllm_rust_parser_error", "detail": "parser failed"}
    }
    first_path.write_text(unified_history.dump_yaml(first), encoding="utf-8")

    loose = tmp_path / "loose"
    unified_history.materialize_store(root, loose)
    path = loose / "dynamo_v2-0.5.0/gemma4/UNIFIED.1-1.yaml"
    document = unified_history.load_yaml(path)
    record = document["cases"]["UNIFIED.1-1"]
    record.pop("error")
    record["assembled"] = [{"kind": "text", "text": "recaptured"}]
    record["capture_input"] = _request()
    path.write_text(unified_history.dump_yaml(document), encoding="utf-8")

    with pytest.raises(ValueError, match="capture is immutable"):
        unified_history.update_store_from_loose(root, loose, complete_snapshot=True)


def test_history_inserts_an_older_live_capture_without_changing_newer_results(tmp_path):
    root = _store(tmp_path / "store")
    before_later = unified_history.load_store(root).histories[("gemma4", "dynamo_v2")].resolve(
        "dynamo_v2-0.5.2"
    )["text_only"]["observation"]
    loose = tmp_path / "loose"
    unified_history.materialize_store(root, loose)
    path = loose / "dynamo_v2-0.4.0/gemma4/UNIFIED.1-1.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(
        unified_history.dump_yaml(
            {
                "family": "gemma4",
                "mode": "unified",
                "captured_with": {"dynamo_v2": "0.4.0"},
                "cases": {
                    "UNIFIED.1-1": {
                        "capture_input": _request(),
                        "assembled": [{"kind": "text", "text": "historical"}],
                        "chunks": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    unified_history.update_store_from_loose(root, loose, complete_snapshot=True)
    history = unified_history.load_store(root).histories[("gemma4", "dynamo_v2")]
    assert history.ordered_capture_ids() == ["dynamo_v2-0.4.0", "dynamo_v2-0.5.0", "dynamo_v2-0.5.2"]
    assert history.resolve("dynamo_v2-0.5.2")["text_only"]["observation"] == before_later


def test_history_backfills_only_bound_additions_and_preserves_later_absence(tmp_path):
    root = _store(tmp_path / "store")
    _add_backfill_case(root)
    history = unified_history.load_store(root).histories[("gemma4", "dynamo_v2")]
    original = {
        capture_id: {
            case_id: {
                "semantic": unified_history._capture_semantic(record, case_id),
                "document": {
                    key: value
                    for key, value in record["document"].items()
                    if key != "inherited_from"
                },
            }
            for case_id, record in history.resolve(capture_id).items()
        }
        for capture_id in history.ordered_capture_ids()
    }
    loose = tmp_path / "loose"
    unified_history.materialize_store(root, loose)
    _add_loose_backfill_record(loose, "0.5.0", "captured")

    unified_history.update_store_from_loose(root, loose, complete_snapshot=True)

    history = unified_history.load_store(root).histories[("gemma4", "dynamo_v2")]
    assert "new_case" in history.resolve("dynamo_v2-0.5.0")
    assert "new_case" not in history.resolve("dynamo_v2-0.5.2")
    assert history.captures["dynamo_v2-0.5.2"]["changes"]["new_case"] == {"absent": True}
    for capture_id, observations in original.items():
        resolved = history.resolve(capture_id)
        for case_id, expected in observations.items():
            record = resolved[case_id]
            assert unified_history._capture_semantic(record, case_id) == expected["semantic"]
            assert {
                key: value
                for key, value in record["document"].items()
                if key != "inherited_from"
            } == expected["document"]


@pytest.mark.parametrize("later_metadata", [{}, {"parser": "NewParser"}])
@pytest.mark.parametrize("later_overrides", [{}, {"parser_path": "new/parser.py"}])
def test_backfill_preserves_later_metadata_and_parser_path(
    tmp_path, later_metadata, later_overrides
):
    root = _store(tmp_path / "store")
    _add_backfill_case(root)
    later = _change("later")
    later["case_key"] = "UNIFIED.1-2"
    _write_capture(
        root, "0.5.2", {"new_case": later},
        metadata_changes={"new_case": later_metadata},
        document_overrides={"new_case": later_overrides},
    )
    history = unified_history.load_store(root).histories[("gemma4", "dynamo_v2")]
    # Remove explicit empty resets: an originally absent field must also stay absent.
    if not later_metadata:
        history.captures["dynamo_v2-0.5.2"]["metadata_changes"] = {}
    if not later_overrides:
        history.captures["dynamo_v2-0.5.2"]["document_overrides"] = {}
    before = history.resolve("dynamo_v2-0.5.2")
    record = _change("earlier")
    record["case_key"] = "UNIFIED.1-2"
    record["stimulus"]["semantic_sha256"] = "bound-request"
    record["document"] = {
        "record_metadata": {"parser": "OldParser"},
        "parser_path": "old/parser.py",
    }
    assert unified_history._add_bound_legacy_cases(
        history, "dynamo_v2-0.5.0", {"new_case": record}, ["new_case"]
    )
    after = history.resolve("dynamo_v2-0.5.2")
    assert after.keys() == before.keys()
    for case_id in before:
        assert after[case_id]["document"] == before[case_id]["document"]
        assert unified_history._capture_semantic(after[case_id], case_id) == unified_history._capture_semantic(before[case_id], case_id)


@pytest.mark.parametrize(
    ("later_output", "expected_delta", "origin"),
    [
        ("captured", None, "dynamo_v2-0.5.0"),
        ("changed", {"observation"}, "dynamo_v2-0.5.2"),
    ],
)
def test_history_compacts_explicit_backfill_by_version_delta(
    tmp_path, later_output, expected_delta, origin
):
    root = _store(tmp_path / "store")
    _add_backfill_case(root)
    loose = tmp_path / "loose"
    unified_history.materialize_store(root, loose)
    _add_loose_backfill_record(loose, "0.5.0", "captured")
    _add_loose_backfill_record(loose, "0.5.2", later_output)

    unified_history.update_store_from_loose(root, loose, complete_snapshot=True)

    history = unified_history.load_store(root).histories[("gemma4", "dynamo_v2")]
    record = history.resolve("dynamo_v2-0.5.2")["new_case"]
    if expected_delta is None:
        assert record["document"]["inherited_from"] == origin.removeprefix("dynamo_v2-")
        assert "new_case" not in history.captures["dynamo_v2-0.5.2"]["changes"]
    else:
        assert "inherited_from" not in record["document"]
        assert set(history.captures["dynamo_v2-0.5.2"]["changes"]["new_case"]) == expected_delta

    before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*.yaml")}
    assert unified_history.update_store_from_loose(root, loose, complete_snapshot=True) == []
    assert {path.relative_to(root): path.read_bytes() for path in root.rglob("*.yaml")} == before


def test_history_rejects_unbound_additions_to_a_legacy_capture(tmp_path):
    root = _store(tmp_path / "store")
    _add_backfill_case(root)
    loose = tmp_path / "loose"
    unified_history.materialize_store(root, loose)
    _add_loose_backfill_record(loose, "0.5.0", "captured")
    path = loose / "dynamo_v2-0.5.0/gemma4/UNIFIED.1-2.yaml"
    document = unified_history.load_yaml(path)
    document["cases"]["UNIFIED.1-2"].pop("capture_input")
    path.write_text(unified_history.dump_yaml(document), encoding="utf-8")

    with pytest.raises(ValueError, match="only bound, conflict-free active-case additions"):
        unified_history.update_store_from_loose(root, loose, complete_snapshot=True)


def test_history_rejects_additions_to_captured_provenance(tmp_path):
    root = _store(tmp_path / "store")
    _add_backfill_case(root)
    path = root / "families/gemma4/dynamo_v2-0.5.0.yaml"
    capture = unified_history.load_yaml(path)
    capture["provenance"] = {
        "status": "captured",
        "origin": {"crate_version": "0.5.0", "source_sha256": "a" * 64},
    }
    path.write_text(unified_history.dump_yaml(capture), encoding="utf-8")
    loose = tmp_path / "loose"
    unified_history.materialize_store(root, loose)
    _add_loose_backfill_record(loose, "0.5.0", "captured")

    with pytest.raises(ValueError, match="only bound, conflict-free active-case additions"):
        unified_history.update_store_from_loose(root, loose, complete_snapshot=True)


def test_history_rejects_addition_when_an_existing_row_conflicts(tmp_path):
    root = _store(tmp_path / "store")
    _add_backfill_case(root)
    loose = tmp_path / "loose"
    unified_history.materialize_store(root, loose)
    _add_loose_backfill_record(loose, "0.5.0", "captured")
    existing_path = loose / "dynamo_v2-0.5.0/gemma4/UNIFIED.1-1.yaml"
    existing = unified_history.load_yaml(existing_path)
    existing["cases"]["UNIFIED.1-1"]["assembled"] = [
        {"kind": "text", "text": "changed"}
    ]
    existing_path.write_text(unified_history.dump_yaml(existing), encoding="utf-8")

    with pytest.raises(ValueError, match="capture is immutable"):
        unified_history.update_store_from_loose(root, loose, complete_snapshot=True)


def test_history_preserves_typed_peer_unavailable_state():
    marker = {
        "unavailable": {
            "code": "vllm_rust_guided_json_unsupported",
            "detail": "request mode is not exposed by the Rust API",
        }
    }
    stimulus, observation, metadata = unified_history._legacy_state(
        {
            "unavailable": marker["unavailable"]["detail"],
            "capture_stimulus": marker,
        },
        b"capture",
        "vllm_rust-0.26.0/gemma4/UNIFIED.30-1.yaml",
        {},
        _request(),
    )

    assert stimulus == marker
    assert observation == marker
    assert metadata == {}


@pytest.mark.parametrize("state", ["error", "unavailable"])
def test_history_materialized_typed_states_round_trip_without_reclassification(
    tmp_path, state
):
    root = _store(tmp_path / "store")
    history = unified_history.load_store(root).histories[("gemma4", "dynamo_v2")]
    case = history.family.cases["text_only"]
    stimulus = {state: {"code": "request_not_retained"}}
    observation = {
        state: {
            "code": "peer_guided_json_unsupported",
            "detail": "request mode is not exposed by the peer API",
        }
    }
    change = {
        "stimulus": stimulus,
        "observation": observation,
        "document": {},
    }

    record, document = unified_history.materialized_record(case, change)
    parsed_stimulus, parsed_observation, metadata = unified_history._legacy_state(
        record,
        b"capture",
        "vllm_rust-0.26.0/gemma4/UNIFIED.1-1.yaml",
        {},
        case["request"],
    )

    assert parsed_stimulus == stimulus
    assert parsed_observation == observation
    assert metadata == document == {}


def test_history_accepts_typed_error_stimuli_in_store_round_trip(tmp_path):
    root = _store(tmp_path / "store")
    stimulus = {
        "error": {
            "code": "request_not_retained",
            "detail": "the original request could not be recovered",
        }
    }
    change = _change()
    change["stimulus"] = stimulus
    change["observation"] = {
        "error": {"code": "peer_capture_error", "detail": "capture failed"}
    }
    _write_capture(root, "0.5.3", {"text_only": change})

    resolved = unified_history.load_store(root).histories[("gemma4", "dynamo_v2")].resolve(
        "dynamo_v2-0.5.3"
    )["text_only"]
    assert resolved["stimulus"] == stimulus

    loose = tmp_path / "loose"
    unified_history.materialize_store(root, loose)
    unified_history.update_store_from_loose(root, loose, complete_snapshot=True)
    reingested = unified_history.load_store(root).histories[("gemma4", "dynamo_v2")].resolve(
        "dynamo_v2-0.5.3"
    )["text_only"]
    assert reingested["stimulus"] == stimulus
    assert reingested["observation"] == change["observation"]


def test_history_binds_typed_unavailable_state_when_request_is_retained():
    marker = {
        "unavailable": {
            "code": "vllm_rust_guided_json_unsupported",
            "detail": "request mode is not exposed by the Rust API",
        }
    }
    request = _request()

    stimulus, observation, metadata = unified_history._legacy_state(
        {
            "unavailable": marker["unavailable"]["detail"],
            "capture_stimulus": marker,
            "capture_input": request,
        },
        b"capture",
        "vllm_rust-0.26.0/gemma4/UNIFIED.30-1.yaml",
        {},
        request,
    )

    assert stimulus == {
        "ref": "current",
        "semantic_sha256": unified_history._request_digest(request),
    }
    assert observation == marker
    assert metadata == {}


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


def test_history_ignores_generated_oracle_directories_but_rejects_malformed_captures(tmp_path):
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
