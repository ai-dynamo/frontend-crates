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
    metadata_changes: dict | None = None,
    document_overrides: dict | None = None,
    provenance: dict | None = None,
) -> Path:
    path = root / f"families/gemma4/dynamo_v2-{version}.yaml"
    path.write_text(
        unified_history.dump_yaml(
            {
                "provenance": provenance
                or {"status": "legacy", "captured_with": {"dynamo_v2": version}},
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


def test_schema_v3_carries_a_missing_checkpoint_forward(tmp_path):
    history = unified_history.load_store(_store(tmp_path)).histories[("gemma4", "dynamo_v2")]

    resolved = history.resolve("dynamo_v2-0.5.2")

    assert history.ordered_capture_ids() == ["dynamo_v2-0.5.0", "dynamo_v2-0.5.2"]
    assert resolved["text_only"]["observation"] == _change()["observation"]
    assert resolved["text_only"]["document"]["captured_with"] == {"dynamo_v2": "0.5.0"}
    assert resolved["text_only"]["document"]["inherited_from"] == "0.5.0"


def test_schema_v3_applies_metadata_and_parser_path_without_replacing_observation(tmp_path):
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


def test_schema_v3_preserves_one_origin_for_a_captured_checkpoint(tmp_path):
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


def test_schema_v3_keeps_the_first_origin_when_an_unchanged_capture_is_rerun(tmp_path):
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

    assert unified_history.update_store_from_loose(root, loose, complete_snapshot=True) == []
    assert (root / "families/gemma4/dynamo_v2-0.5.3.yaml").read_bytes() == before

    document["cases"]["UNIFIED.1-1"]["assembled"] = [{"kind": "text", "text": "changed"}]
    path.write_text(unified_history.dump_yaml(document), encoding="utf-8")
    with pytest.raises(ValueError, match="capture is immutable"):
        unified_history.update_store_from_loose(root, loose, complete_snapshot=True)


def test_existing_capture_accepts_only_additive_historical_backfill(tmp_path):
    root = _store(tmp_path / "store")
    family_path = root / "families/gemma4/inputs_and_golden.yaml"
    family_document = unified_history.load_yaml(family_path)
    added_case = dict(family_document["cases"]["text_only"])
    added_case["scenario"] = "new_text_case"
    added_case["display_id"] = "UNIFIED.1-2"
    added_case["historical_ids"] = []
    family_document["cases"]["new_text_case"] = added_case
    family_path.write_text(unified_history.dump_yaml(family_document), encoding="utf-8")
    origin = {"crate_version": "0.5.3", "source_sha256": "a" * 64, "git_commit": "b" * 40}
    _write_capture(root, "0.5.3", {"text_only": _change("existing")}, provenance={"status": "captured", "origin": origin})
    before = unified_history.load_store(root).histories[("gemma4", "dynamo_v2")].resolve("dynamo_v2-0.5.3")
    loose = tmp_path / "loose"
    unified_history.materialize_store(root, loose)
    capture_dir = loose / "dynamo_v2-0.5.3/gemma4"
    document = unified_history.load_yaml(capture_dir / "UNIFIED.1-1.yaml")
    document["capture_origin"] = origin
    document["cases"] = {"UNIFIED.1-2": {"assembled": [{"kind": "text", "text": "new"}]}}
    (capture_dir / "UNIFIED.1-2.yaml").write_text(unified_history.dump_yaml(document), encoding="utf-8")

    changed = unified_history.update_store_from_loose(root, loose, complete_snapshot=False)
    assert len(changed) == 1
    history = unified_history.load_store(root).histories[("gemma4", "dynamo_v2")]
    resolved = history.resolve("dynamo_v2-0.5.3")
    assert resolved["text_only"] == before["text_only"]
    assert resolved["new_text_case"]["observation"] == {"value": {"assembled": [{"kind": "text", "text": "new"}]}}


@pytest.mark.parametrize("mixed_origin_batch", [False, True])
def test_existing_capture_rejects_originless_measured_backfill(tmp_path, mixed_origin_batch):
    root = _store(tmp_path / "store")
    family_path = root / "families/gemma4/inputs_and_golden.yaml"
    family_document = unified_history.load_yaml(family_path)
    for scenario, display_id in (("new_text_case", "UNIFIED.1-2"), ("second_text_case", "UNIFIED.1-3")):
        added_case = dict(family_document["cases"]["text_only"])
        added_case["scenario"] = scenario
        added_case["display_id"] = display_id
        added_case["historical_ids"] = []
        family_document["cases"][scenario] = added_case
    family_path.write_text(unified_history.dump_yaml(family_document), encoding="utf-8")
    origin = {"crate_version": "0.5.3", "source_sha256": "a" * 64, "git_commit": "b" * 40}
    _write_capture(root, "0.5.3", {"text_only": _change("existing")}, provenance={"status": "captured", "origin": origin})
    loose = tmp_path / "loose"
    unified_history.materialize_store(root, loose)
    capture_dir = loose / "dynamo_v2-0.5.3/gemma4"
    document = unified_history.load_yaml(capture_dir / "UNIFIED.1-1.yaml")
    document["capture_origin"] = origin
    missing_origin = dict(document)
    missing_origin.pop("capture_origin")
    missing_origin["cases"] = {"UNIFIED.1-2": {"assembled": [{"kind": "text", "text": "new"}]}}
    (capture_dir / "UNIFIED.1-2.yaml").write_text(unified_history.dump_yaml(missing_origin), encoding="utf-8")
    if mixed_origin_batch:
        valid_origin = dict(document)
        valid_origin["cases"] = {"UNIFIED.1-3": {"assembled": [{"kind": "text", "text": "also new"}]}}
        (capture_dir / "UNIFIED.1-3.yaml").write_text(unified_history.dump_yaml(valid_origin), encoding="utf-8")

    with pytest.raises(ValueError, match="back-capture records are missing capture origin"):
        unified_history.update_store_from_loose(root, loose, complete_snapshot=False)


@pytest.mark.parametrize("complete_snapshot", [False, True])
def test_mixed_backfill_and_new_checkpoints_keep_family_release_order(tmp_path, complete_snapshot):
    root = _store(tmp_path / "store")
    family_path = root / "families/gemma4/inputs_and_golden.yaml"
    family = unified_history.load_yaml(family_path)
    added = copy.deepcopy(family["cases"]["text_only"])
    added.update(scenario="new_text_case", display_id="UNIFIED.1-2", historical_ids=[])
    family["cases"]["new_text_case"] = added
    family_path.write_text(unified_history.dump_yaml(family))
    for version in ("0.5.3", "0.5.4"):
        origin = {"crate_version": version, "source_sha256": "a" * 64, "git_commit": "b" * 40}
        _write_capture(root, version, {"text_only": _change(version)},
                       provenance={"status": "captured", "origin": origin})
    other = root / "families/qwen3"
    other.mkdir()
    other_family = copy.deepcopy(family)
    other_family["input_document"]["family"] = "qwen3"
    other_family["golden_document"]["family"] = "qwen3"
    (other / "inputs_and_golden.yaml").write_text(unified_history.dump_yaml(other_family))
    for version in ("0.5.0", "0.5.2", "0.5.3"):
        source = root / f"families/gemma4/dynamo_v2-{version}.yaml"
        (other / source.name).write_bytes(source.read_bytes())
    loose = tmp_path / "loose"
    unified_history.materialize_store(root, loose)
    for family_name in ("gemma4", "qwen3"):
        for version in ("0.5.3", "0.5.4", "0.5.5", "0.5.6"):
            directory = loose / f"dynamo_v2-{version}" / family_name
            directory.mkdir(parents=True, exist_ok=True)
            existing_path = directory / "UNIFIED.1-1.yaml"
            template = unified_history.load_yaml(existing_path if existing_path.exists() else
                loose / "dynamo_v2-0.5.3" / family_name / "UNIFIED.1-1.yaml")
            template.pop("inherited_from", None)
            template.pop("capture_provenance", None)
            template["capture_origin"] = {"crate_version": version, "source_sha256": "a" * 64, "git_commit": "b" * 40}
            template["captured_with"] = {"dynamo_v2": version}
            if not existing_path.exists():
                template["cases"] = {"UNIFIED.1-1": {"assembled": [{"kind": "text", "text": version}]}}
            (directory / "UNIFIED.1-1.yaml").write_text(unified_history.dump_yaml(template))
            template["cases"] = {"UNIFIED.1-2": {"assembled": [{"kind": "text", "text": "added " + version}]}}
            (directory / "UNIFIED.1-2.yaml").write_text(unified_history.dump_yaml(template))

    unified_history.update_store_from_loose(root, loose, complete_snapshot=complete_snapshot)

    histories = unified_history.load_store(root).histories
    for family_name in ("gemma4", "qwen3"):
        history = histories[(family_name, "dynamo_v2")]
        assert history.ordered_capture_ids() == [f"dynamo_v2-{version}" for version in (
            "0.5.0", "0.5.2", "0.5.3", "0.5.4", "0.5.5", "0.5.6")]
        for version in ("0.5.3", "0.5.4", "0.5.5", "0.5.6"):
            record = history.resolve("dynamo_v2-" + version)["new_text_case"]
            assert record["observation"]["value"]["assembled"] == [{"kind": "text", "text": "added " + version}]
            assert record["document"]["capture_origin"]["crate_version"] == version


@pytest.mark.parametrize("version", ["0.5.1.patch1", "0.5.1+source." + "a" * 64])
def test_schema_v3_rejects_patch_and_source_qualified_filenames(tmp_path, version):
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
def test_schema_v3_rejects_identity_headers_owned_by_the_path(tmp_path, filename, header):
    root = _store(tmp_path)
    path = root / "families/gemma4" / filename
    document = unified_history.load_yaml(path)
    document.update(header)
    path.write_text(unified_history.dump_yaml(document), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown fields"):
        unified_history.load_store(root)


def test_schema_v3_rejects_parent_chain_fields(tmp_path):
    root = _store(tmp_path)
    path = root / "families/gemma4/dynamo_v2-0.5.2.yaml"
    document = unified_history.load_yaml(path)
    document["parent"] = "dynamo_v2-0.5.0"
    path.write_text(unified_history.dump_yaml(document), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown fields"):
        unified_history.load_store(root)


def test_schema_v3_rejects_a_capture_origin_for_another_version(tmp_path):
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


def test_schema_v3_rewrite_is_byte_deterministic(tmp_path):
    root = _store(tmp_path)
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*.yaml")}

    unified_history.rewrite_store(root)
    unified_history.rewrite_store(root)

    assert {path.relative_to(root): path.read_bytes() for path in root.rglob("*.yaml")} == before


def test_schema_v3_materializes_a_derived_release_view_without_a_checkpoint(tmp_path):
    root = _store(tmp_path / "store")
    loose = tmp_path / "loose"

    unified_history.materialize_store(
        root,
        loose,
        derived_release_versions={"dynamo_v2": "0.6.1"},
    )

    assert not (root / "families/gemma4/dynamo_v2-0.6.1.yaml").exists()
    document = unified_history.load_yaml(loose / "dynamo_v2-0.6.1/gemma4/UNIFIED.1-1.yaml")
    assert document["capture_provenance"] == {"format": "schema_v3"}
    assert document["captured_with"] == {"dynamo_v2": "0.5.0"}
    assert document["inherited_from"] == "0.5.0"


def test_schema_v3_derived_release_view_is_not_reingested_as_a_checkpoint(tmp_path):
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


def test_schema_v3_rejects_input_changes_without_recapturing_prior_versions(tmp_path):
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
