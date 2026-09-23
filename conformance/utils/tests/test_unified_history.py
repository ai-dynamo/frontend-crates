# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coverage for canonical Unified capture checkpoints."""

import copy
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import capture_stimulus
import package_fixtures
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


def test_capture_order_uses_numeric_prerelease_identifiers():
    assert unified_history._capture_release_sort_key("0.7.0-rc.2") < (
        unified_history._capture_release_sort_key("0.7.0-rc.10")
    )
    assert unified_history._capture_release_sort_key("0.7.0-rc.10") < (
        unified_history._capture_release_sort_key("0.7.0")
    )


@pytest.mark.parametrize("versions", [
    ("0.9.0", "0.10.0"),
    ("0.7.0-rc.2", "0.7.0-rc.10"),
    ("0.7.0-rc.10", "0.7.0"),
])
def test_ingest_multiple_new_captures_in_semantic_version_order(tmp_path, versions):
    root = _store(tmp_path / "store")
    loose = tmp_path / "loose"
    unified_history.materialize_store(root, loose)
    for version in reversed(versions):
        target = loose / f"dynamo_v2-{version}"
        shutil.copytree(loose / "dynamo_v2-0.5.2", target)
        path = target / "gemma4/UNIFIED.1-1.yaml"
        document = unified_history.load_yaml(path)
        document["captured_with"] = {"dynamo_v2": version}
        record = document["cases"]["UNIFIED.1-1"]
        record["assembled"] = [{"kind": "text", "text": version}]
        record["chunks"] = [{"expected": record["assembled"]}]
        path.write_text(unified_history.dump_yaml(document), encoding="utf-8")

    unified_history.update_store_from_loose(root, loose, complete_snapshot=True)

    history = unified_history.load_store(root).histories[("gemma4", "dynamo_v2")]
    assert history.ordered_capture_ids() == [
        "dynamo_v2-0.5.0", "dynamo_v2-0.5.2", *(f"dynamo_v2-{version}" for version in versions),
    ]
    for version in versions:
        observation = history.resolve(f"dynamo_v2-{version}")["text_only"]["observation"]["value"]
        assert observation["assembled"] == [{"kind": "text", "text": version}]
    assert unified_history.update_store_from_loose(root, loose, complete_snapshot=True) == []


def test_schema_v3_does_not_materialize_an_uncaptured_release(tmp_path):
    root = _store(tmp_path / "store")
    loose = tmp_path / "loose"

    unified_history.materialize_store(root, loose)

    assert not (root / "families/gemma4/dynamo_v2-0.6.1.yaml").exists()
    assert not (loose / "dynamo_v2-0.6.1").exists()


def test_schema_v3_materialization_does_not_add_an_uncaptured_checkpoint(tmp_path):
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

    unified_history.materialize_store(root, loose)
    changed = unified_history.update_store_from_loose(root, loose, complete_snapshot=True)

    assert changed == []
    assert {path.relative_to(root): path.read_bytes() for path in root.rglob("*.yaml")} == before
    assert not (root / "families/gemma4/dynamo_v2-0.6.1.yaml").exists()


@pytest.mark.parametrize("stimulus_kind", ["inline", "ref", "partial"])
@pytest.mark.parametrize("mutation", [None, "request", "observation"])
def test_recorded_capture_roundtrip_compares_resolved_request(tmp_path, stimulus_kind, mutation):
    root = _store(tmp_path / "store")
    capture_path = root / "families/gemma4/dynamo_v2-0.5.0.yaml"
    capture = unified_history.load_yaml(capture_path)
    capture["changes"]["text_only"]["stimulus"] = (
        {"inline": _request()} if stimulus_kind == "inline" else {"ref": "current"}
    )
    if stimulus_kind == "partial":
        capture["changes"]["text_only"]["stimulus"] = {
            "partial": {key: value for key, value in _request().items() if key != "tools"},
        }
    capture_path.write_text(unified_history.dump_yaml(capture), encoding="utf-8")
    loose = tmp_path / "loose"
    unified_history.materialize_store(root, loose)
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*.yaml")}
    if stimulus_kind == "partial":
        path = loose / "dynamo_v2-0.5.0/gemma4/UNIFIED.1-1.yaml"
        record = unified_history.load_yaml(path)["cases"]["UNIFIED.1-1"]
        assert record["capture_input"] == capture["changes"]["text_only"]["stimulus"]["partial"]
        reason = capture_stimulus.comparison_failure(record, _request(), path.read_bytes(), path.name, {})
        assert "original tool schema was not retained" in reason
    if mutation is not None:
        path = loose / "dynamo_v2-0.5.0/gemma4/UNIFIED.1-1.yaml"
        document = unified_history.load_yaml(path)
        record = document["cases"]["UNIFIED.1-1"]
        if mutation == "request":
            record["capture_input"]["input"] = "changed input"
        else:
            record["assembled"][0]["text"] = "changed observation"
        path.write_text(unified_history.dump_yaml(document), encoding="utf-8")
        with pytest.raises(ValueError, match="capture is immutable"):
            unified_history.update_store_from_loose(root, loose, complete_snapshot=True)
    else:
        assert unified_history.update_store_from_loose(root, loose, complete_snapshot=True) == []
    assert {path.relative_to(root): path.read_bytes() for path in root.rglob("*.yaml")} == before


@pytest.mark.parametrize("provenance_kind", ["origin", "captured_with", "inherited"])
@pytest.mark.parametrize("stimulus_kind", ["ref", "inline"])
def test_unchanged_new_capture_retains_only_actual_version_provenance(tmp_path, provenance_kind, stimulus_kind):
    root = _store(tmp_path / "store")
    if stimulus_kind == "inline":
        anchor = root / "families/gemma4/dynamo_v2-0.5.0.yaml"
        document = unified_history.load_yaml(anchor)
        document["changes"]["text_only"]["stimulus"] = {"inline": _request()}
        anchor.write_text(unified_history.dump_yaml(document), encoding="utf-8")
    loose = tmp_path / "loose"
    unified_history.materialize_store(root, loose)
    capture_dir = loose / "dynamo_v2-0.5.3"
    shutil.copytree(loose / "dynamo_v2-0.5.2", capture_dir)
    path = capture_dir / "gemma4/UNIFIED.1-1.yaml"
    document = unified_history.load_yaml(path)
    origin = {"crate_version": "0.5.3", "source_sha256": "c" * 64, "git_commit": "d" * 40}
    if provenance_kind != "inherited":
        document.pop("inherited_from", None)
        document["captured_with"] = {"dynamo_v2": "0.5.3"}
    if provenance_kind == "origin":
        document["capture_origin"] = origin
    path.write_text(unified_history.dump_yaml(document), encoding="utf-8")

    changed = unified_history.update_store_from_loose(root, loose, complete_snapshot=True)

    checkpoint = root / "families/gemma4/dynamo_v2-0.5.3.yaml"
    if provenance_kind == "inherited":
        assert changed == []
        assert not checkpoint.exists()
        return
    assert checkpoint in changed
    capture = unified_history.load_yaml(checkpoint)
    assert capture["changes"] == {}
    assert capture["metadata_changes"] == {}
    assert capture["document_overrides"] == {}
    assert capture["provenance"] == (
        {"status": "captured", "origin": origin} if provenance_kind == "origin"
        else {"status": "legacy", "captured_with": {"dynamo_v2": "0.5.3"}}
    )
    rematerialized = tmp_path / "rematerialized"
    unified_history.materialize_store(root, rematerialized)
    assert (rematerialized / "dynamo_v2-0.5.3/gemma4/UNIFIED.1-1.yaml").is_file()
    before = checkpoint.read_bytes()
    assert unified_history.update_store_from_loose(root, loose, complete_snapshot=True) == []
    assert checkpoint.read_bytes() == before


@pytest.mark.parametrize("mutation", [None, "new_capture", "missing_family", "missing_case"])
def test_packaging_preserves_historical_capture_coverage(tmp_path, mutation):
    root = tmp_path / "store"
    _write_family(root)
    _write_capture(root, "0.5.0", {"text_only": _change()})
    second_family = root / "families/qwen3"
    shutil.copytree(root / "families/gemma4", second_family)
    family_path = second_family / "inputs_and_golden.yaml"
    family = unified_history.load_yaml(family_path)
    for field in ("input_document", "golden_document"):
        family[field]["family"] = "qwen3"
    family_path.write_text(unified_history.dump_yaml(family), encoding="utf-8")
    original_capture = second_family / "dynamo_v2-0.5.0.yaml"
    capture = unified_history.load_yaml(original_capture)
    capture["provenance"]["captured_with"]["dynamo_v2"] = "0.6.0"
    original_capture.unlink()
    (second_family / "dynamo_v2-0.6.0.yaml").write_text(unified_history.dump_yaml(capture), encoding="utf-8")
    loose = tmp_path / "loose"
    unified_history.materialize_store(root, loose / "unified")
    legacy = tmp_path / "legacy/batch/families/gemma4/inputs_and_golden.yaml"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("corpus: batch\nfamily: gemma4\ncapture: inputs\ndocuments: {}\n")
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*.yaml")}
    expected_error = None
    if mutation == "new_capture":
        shutil.copytree(loose / "unified/dynamo_v2-0.5.0", loose / "unified/dynamo_v2-0.5.1")
        expected_error = "complete capture dynamo_v2-0.5.1 is missing active families: qwen3"
    elif mutation == "missing_family":
        shutil.rmtree(loose / "unified/dynamo_v2-0.6.0/qwen3")
        expected_error = "complete capture dynamo_v2-0.6.0 is missing active families: qwen3"
    elif mutation == "missing_case":
        (loose / "unified/dynamo_v2-0.5.0/gemma4/UNIFIED.1-1.yaml").unlink()
        expected_error = "complete capture dynamo_v2-0.5.0 is missing active cases: gemma4/text_only"
    if expected_error is None:
        package_fixtures.build_shards(loose, tmp_path / "blobs", history_root=root, legacy_root=tmp_path / "legacy")
    else:
        with pytest.raises(ValueError, match=expected_error):
            package_fixtures.build_shards(loose, tmp_path / "blobs", history_root=root, legacy_root=tmp_path / "legacy")
    assert {path.relative_to(root): path.read_bytes() for path in root.rglob("*.yaml")} == before


@pytest.mark.parametrize("retained_cases", [0, 1])
def test_packaging_requires_complete_new_family_version_pair(tmp_path, retained_cases):
    root = _store(tmp_path / "store")
    second_family = root / "families/qwen3"
    shutil.copytree(root / "families/gemma4", second_family)
    (second_family / "dynamo_v2-0.5.2.yaml").unlink()
    family_path = second_family / "inputs_and_golden.yaml"
    family = unified_history.load_yaml(family_path)
    for field in ("input_document", "golden_document"):
        family[field]["family"] = "qwen3"
    other = copy.deepcopy(family["cases"]["text_only"])
    other.update(scenario="second", display_id="UNIFIED.1-2", request=_request("second"))
    other["golden"] = {"assembled": [{"kind": "text", "text": "second"}]}
    family["cases"]["second"] = other
    family_path.write_text(unified_history.dump_yaml(family), encoding="utf-8")
    capture_path = second_family / "dynamo_v2-0.5.0.yaml"
    capture = unified_history.load_yaml(capture_path)
    capture["changes"]["second"] = {**_change("second"), "case_key": "UNIFIED.1-2"}
    capture_path.write_text(unified_history.dump_yaml(capture), encoding="utf-8")
    loose = tmp_path / "loose"
    unified_history.materialize_store(root, loose / "unified")
    incoming = loose / "unified/dynamo_v2-0.5.2/qwen3"
    incoming.mkdir()
    if retained_cases:
        document = unified_history.load_yaml(loose / "unified/dynamo_v2-0.5.0/qwen3/UNIFIED.1-1.yaml")
        document["captured_with"] = {"dynamo_v2": "0.5.2"}
        (incoming / "UNIFIED.1-1.yaml").write_text(unified_history.dump_yaml(document), encoding="utf-8")
    legacy = tmp_path / "legacy/batch/families/gemma4/inputs_and_golden.yaml"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("corpus: batch\nfamily: gemma4\ncapture: inputs\ndocuments: {}\n")
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*.yaml")}

    with pytest.raises(ValueError, match="complete capture dynamo_v2-0.5.2 is missing active cases: qwen3/second"):
        package_fixtures.build_shards(loose, tmp_path / "blobs", history_root=root, legacy_root=tmp_path / "legacy")

    assert {path.relative_to(root): path.read_bytes() for path in root.rglob("*.yaml")} == before


def test_required_snapshot_cannot_use_a_later_prerelease_for_a_missing_family(tmp_path):
    root = tmp_path / "store"
    _write_family(root)
    _write_capture(root, "1.0.0-rc.1", {"text_only": _change()})
    second_family = root / "families/qwen3"
    shutil.copytree(root / "families/gemma4", second_family)
    family_path = second_family / "inputs_and_golden.yaml"
    family = unified_history.load_yaml(family_path)
    for field in ("input_document", "golden_document"):
        family[field]["family"] = "qwen3"
    family_path.write_text(unified_history.dump_yaml(family), encoding="utf-8")
    original_capture = second_family / "dynamo_v2-1.0.0-rc.1.yaml"
    capture = unified_history.load_yaml(original_capture)
    capture["provenance"]["captured_with"]["dynamo_v2"] = "1.0.0-rc.10"
    original_capture.unlink()
    (second_family / "dynamo_v2-1.0.0-rc.10.yaml").write_text(
        unified_history.dump_yaml(capture), encoding="utf-8"
    )
    loose = tmp_path / "loose"
    unified_history.materialize_store(root, loose)
    (loose / "dynamo_v2-1.0.0-rc.1").rename(loose / "dynamo_v2-1.0.0-rc.2")
    shutil.rmtree(loose / "dynamo_v2-1.0.0-rc.10")
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*.yaml")}

    with pytest.raises(ValueError, match="missing active families: qwen3"):
        unified_history.update_store_from_loose(
            root, loose, complete_snapshot=True,
            required_capture_dirs=frozenset({"dynamo_v2-1.0.0-rc.2"}),
        )

    assert {path.relative_to(root): path.read_bytes() for path in root.rglob("*.yaml")} == before


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
