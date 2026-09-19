# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import sys

import pytest
import yaml

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import unified_history  # noqa: E402


def _request(text: str) -> dict:
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


def _write_family(root: Path, family: str, cases: dict) -> None:
    path = root / "families" / f"{family}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        unified_history.dump_yaml(
            {"schema_version": 1, "family": family, "cases": cases}
        )
    )


def _write_history(root: Path, family: str, implementation: str, captures: dict) -> None:
    path = root / "capture_history" / family / f"{implementation}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        unified_history.dump_yaml(
            {
                "schema_version": 1,
                "family": family,
                "implementation": implementation,
                "captures": captures,
            }
        )
    )


def _case(display_id: str, scenario: str, text: str, *, aliases=()) -> dict:
    return {
        "lifecycle": "active",
        "scenario": scenario,
        "display_id": display_id,
        "historical_ids": list(aliases),
        "request": _request(text),
        "golden": {"assembled": [{"kind": "text", "text": text}]},
    }


def _change(text: str, *, stimulus=None, state="value") -> dict:
    if stimulus is None:
        stimulus = {"ref": "current"}
    observation = {
        state: (
            {"assembled": [{"kind": "text", "text": text}], "chunks": []}
            if state == "value"
            else {"code": f"{state}_code", "message": text}
        )
    }
    return {
        "case_key": "UNIFIED.1-1",
        "stimulus": stimulus,
        "observation": observation,
        "document": {"captured_with": {"dynamo_v2": "0.1.0"}},
    }


def _capture(parent, completeness: str, changes: dict, *, version="0.1.0") -> dict:
    return {
        "parent": parent,
        "runtime_version": version,
        "provenance": {"status": "legacy", "captured_with": {"dynamo_v2": version}},
        "completeness": completeness,
        "import_lineage": [],
        "changes": changes,
        "metadata_changes": {},
    }


def _store(root: Path) -> Path:
    _write_family(
        root,
        "gemma4",
        {
            "text_only": _case("UNIFIED.1-1", "text_only", "hello", aliases=("UNIFIED.1.a",)),
            "retired__31_40": {
                "lifecycle": "retired",
                "scenario": None,
                "display_id": None,
                "historical_ids": ["UNIFIED.31-40"],
                "request": None,
                "golden": None,
            },
        },
    )
    _write_family(root, "qwen3", {"text_only": _case("UNIFIED.1-1", "text_only", "hi")})
    _write_history(
        root,
        "gemma4",
        "dynamo_v2",
        {
            "dynamo_v2-0.1.0": _capture(
                None,
                "snapshot",
                {
                    "text_only": _change("hello"),
                    "retired__31_40": {
                        "case_key": "UNIFIED.31-40",
                        "stimulus": {"unavailable": {"code": "not_retained"}},
                        "observation": {
                            "unavailable": {
                                "code": "legacy_unclassified",
                                "message": "not captured",
                            }
                        },
                        "document": {"captured_with": {"dynamo_v2": "0.1.0"}},
                    },
                },
            ),
            "dynamo_v2-0.2.0": _capture(
                "dynamo_v2-0.1.0",
                "delta",
                {},
                version="0.2.0",
            ),
            "dynamo_v2-0.3.0": _capture(
                "dynamo_v2-0.2.0",
                "delta",
                {
                    "text_only": _change(
                        "changed",
                        stimulus={"inline": _request("different request")},
                    ),
                    "retired__31_40": {"absent": True},
                },
                version="0.3.0",
            ),
        },
    )
    _write_history(
        root,
        "qwen3",
        "vllm_python",
        {
            "vllm_python-1.0.0": _capture(
                None,
                "snapshot",
                {
                    "text_only": {
                        **_change("peer error", state="error"),
                        "document": {"captured_with": {"vllm_python": "1.0.0"}},
                    }
                },
                version="1.0.0",
            )
        },
    )
    return root


def test_snapshot_delta_tombstone_and_unchanged_capture_resolution(tmp_path):
    store = unified_history.load_store(_store(tmp_path))
    history = store.histories[("gemma4", "dynamo_v2")]

    first = history.resolve("dynamo_v2-0.1.0")
    unchanged = history.resolve("dynamo_v2-0.2.0")
    changed = history.resolve("dynamo_v2-0.3.0")

    assert unchanged == first
    assert history.captures["dynamo_v2-0.2.0"]["provenance"]["captured_with"] == {
        "dynamo_v2": "0.2.0"
    }
    assert set(changed) == {"text_only"}
    assert changed["text_only"]["observation"]["value"]["assembled"][0]["text"] == "changed"
    assert changed["text_only"]["stimulus"]["inline"]["input"] == "different request"


def test_error_and_unavailable_are_distinct_from_empty_success(tmp_path):
    store = unified_history.load_store(_store(tmp_path))
    dynamo = store.histories[("gemma4", "dynamo_v2")].resolve("dynamo_v2-0.1.0")
    peer = store.histories[("qwen3", "vllm_python")].resolve("vllm_python-1.0.0")

    assert set(dynamo["text_only"]["observation"]) == {"value"}
    assert set(dynamo["retired__31_40"]["observation"]) == {"unavailable"}
    assert set(peer["text_only"]["observation"]) == {"error"}


@pytest.mark.parametrize(
    "text, message",
    [
        ("schema_version: 1\nfamily: gemma4\nfamily: qwen3\ncases: {}\n", "duplicate key"),
        ("schema_version: 1\nfamily: &family gemma4\ncases: {}\n", "anchors"),
        ("schema_version: 1\nfamily: gemma4\ncases: !!map {}\n", "tags"),
    ],
)
def test_rejects_ambiguous_yaml(tmp_path, text, message):
    path = tmp_path / "families/gemma4.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(text)

    with pytest.raises(ValueError, match=message):
        unified_history.load_store(tmp_path)


@pytest.mark.parametrize("parent", ["missing", "self"])
def test_rejects_missing_parent_and_cycle(tmp_path, parent):
    _write_family(tmp_path, "gemma4", {"text_only": _case("UNIFIED.1-1", "text_only", "x")})
    capture_id = "dynamo_v2-0.1.0"
    _write_history(
        tmp_path,
        "gemma4",
        "dynamo_v2",
        {
            capture_id: _capture(
                capture_id if parent == "self" else parent,
                "delta",
                {},
            )
        },
    )

    with pytest.raises(ValueError, match="parent cycle|missing parent"):
        unified_history.load_store(tmp_path)


def test_rejects_duplicate_display_id_or_alias(tmp_path):
    duplicate = _case("UNIFIED.1-1", "other", "other", aliases=("UNIFIED.1.a",))
    _write_family(
        tmp_path,
        "gemma4",
        {
            "text_only": _case("UNIFIED.1-1", "text_only", "x", aliases=("UNIFIED.1.a",)),
            "other": duplicate,
        },
    )

    with pytest.raises(ValueError, match="duplicate display ID|duplicate historical ID"):
        unified_history.load_store(tmp_path)


@pytest.mark.parametrize("display_id", ["/tmp/escaped", "../escaped", "nested/escaped", "nested\\escaped"])
def test_rejects_path_bearing_display_id(tmp_path, display_id):
    root = _store(tmp_path)
    path = root / "families/gemma4.yaml"
    family = yaml.safe_load(path.read_text())
    family["cases"]["text_only"]["display_id"] = display_id
    path.write_text(unified_history.dump_yaml(family))

    with pytest.raises(ValueError, match="safe path component"):
        unified_history.load_store(root)


@pytest.mark.parametrize(
    "observation",
    [
        {"value": {}, "error": {"code": "bad", "message": "bad"}},
        {"unknown": {}},
        None,
    ],
)
def test_rejects_invalid_observation_state(tmp_path, observation):
    root = _store(tmp_path)
    history_path = root / "capture_history/gemma4/dynamo_v2.yaml"
    history = yaml.safe_load(history_path.read_text())
    history["captures"]["dynamo_v2-0.1.0"]["changes"]["text_only"]["observation"] = observation
    history_path.write_text(unified_history.dump_yaml(history))

    with pytest.raises(ValueError, match="observation"):
        unified_history.load_store(root)


def test_writer_and_materializer_are_byte_deterministic(tmp_path):
    source = _store(tmp_path / "source")
    first = tmp_path / "first"
    second = tmp_path / "second"

    unified_history.rewrite_store(source)
    once = {path.relative_to(source): path.read_bytes() for path in source.rglob("*.yaml")}
    unified_history.rewrite_store(source)
    twice = {path.relative_to(source): path.read_bytes() for path in source.rglob("*.yaml")}
    unified_history.materialize_store(source, first)
    unified_history.materialize_store(source, second)

    assert once == twice
    assert {path.relative_to(first): path.read_bytes() for path in first.rglob("*") if path.is_file()} == {
        path.relative_to(second): path.read_bytes()
        for path in second.rglob("*")
        if path.is_file()
    }
    assert (first / "inputs/gemma4/UNIFIED.1-1.yaml").is_file()
    assert (first / "dynamo_v2-0.2.0/gemma4/UNIFIED.1-1.yaml").is_file()
    assert not (first / "dynamo_v2-0.3.0/gemma4/UNIFIED.31-40.yaml").exists()


def test_materializer_preserves_capture_provenance_from_document_history(tmp_path):
    source = _store(tmp_path / "source")
    history_path = source / "capture_history/gemma4/dynamo_v2.yaml"
    history = yaml.safe_load(history_path.read_text())
    history["captures"]["dynamo_v2-0.1.0"]["provenance"] = {
        "status": "verified",
        "record": {"label": "0.1.0", "source_sha256": "a" * 64},
    }
    history["captures"]["dynamo_v2-0.1.0"]["changes"]["text_only"]["document"] = {
        "capture_provenance": {"label": "0.1.0", "source_sha256": "a" * 64}
    }
    history_path.write_text(unified_history.dump_yaml(history))

    destination = tmp_path / "materialized"
    unified_history.materialize_store(source, destination)

    document = yaml.safe_load(
        (destination / "dynamo_v2-0.1.0/gemma4/UNIFIED.1-1.yaml").read_text()
    )
    assert document["capture_provenance"] == {"label": "0.1.0", "source_sha256": "a" * 64}


def test_materializer_does_not_inject_capture_provenance_into_inherited_document(tmp_path):
    source = _store(tmp_path / "source")
    history_path = source / "capture_history/gemma4/dynamo_v2.yaml"
    history = yaml.safe_load(history_path.read_text())
    capture = history["captures"]["dynamo_v2-0.2.0"]
    capture["provenance"] = {"status": "captured", "record": {"label": "0.2.0"}}
    history_path.write_text(unified_history.dump_yaml(history))

    destination = tmp_path / "materialized"
    unified_history.materialize_store(source, destination)

    document = yaml.safe_load(
        (destination / "dynamo_v2-0.2.0/gemma4/UNIFIED.1-1.yaml").read_text()
    )
    assert document["captured_with"] == {"dynamo_v2": "0.1.0"}
    assert "capture_provenance" not in document


def test_materializer_does_not_replace_inherited_captured_with(tmp_path):
    source = _store(tmp_path / "source")
    destination = tmp_path / "materialized"

    unified_history.materialize_store(source, destination)

    document = yaml.safe_load(
        (destination / "dynamo_v2-0.2.0/gemma4/UNIFIED.1-1.yaml").read_text()
    )
    assert document["captured_with"] == {"dynamo_v2": "0.1.0"}


def test_materializer_applies_document_metadata_delta(tmp_path):
    source = _store(tmp_path / "source")
    history_path = source / "capture_history/gemma4/dynamo_v2.yaml"
    history = yaml.safe_load(history_path.read_text())
    history["captures"]["dynamo_v2-0.2.0"]["document_metadata_changes"] = {
        "text_only": {"mode": "changed-mode"}
    }
    history_path.write_text(unified_history.dump_yaml(history))

    destination = tmp_path / "materialized"
    unified_history.materialize_store(source, destination)

    document = yaml.safe_load(
        (destination / "dynamo_v2-0.2.0/gemma4/UNIFIED.1-1.yaml").read_text()
    )
    assert document["mode"] == "changed-mode"
    assert "capture_provenance" not in document
    assert "captured_with" not in document


def test_current_reference_requires_exact_canonical_request(tmp_path):
    root = _store(tmp_path)
    path = root / "capture_history/gemma4/dynamo_v2.yaml"
    history = yaml.safe_load(path.read_text())
    history["captures"]["dynamo_v2-0.1.0"]["changes"]["text_only"]["stimulus"] = {
        "ref": "current",
        "semantic_sha256": "0" * 64,
    }
    path.write_text(unified_history.dump_yaml(history))

    with pytest.raises(ValueError, match="stimulus digest"):
        unified_history.load_store(root)


def test_update_from_loose_adds_only_the_affected_history(tmp_path):
    root = _store(tmp_path / "store")
    loose = tmp_path / "loose"
    capture = loose / "dynamo_v2-0.4.0/gemma4"
    capture.mkdir(parents=True)
    (capture / "UNIFIED.1-1.yaml").write_text(
        unified_history.dump_yaml(
            {
                "family": "gemma4",
                "mode": "unified",
                "capture_provenance": {"label": "0.4.0"},
                "cases": {
                    "UNIFIED.1-1": {
                        "capture_input": _request("hello"),
                        "assembled": [{"kind": "text", "text": "new capture"}],
                        "chunks": [],
                    }
                },
            }
        )
    )
    before = {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*.yaml")
    }

    changed = unified_history.update_from_loose(root, loose)

    history_path = root / "capture_history/gemma4/dynamo_v2.yaml"
    assert changed == [history_path]
    assert unified_history.load_store(root).histories[("gemma4", "dynamo_v2")].resolve(
        "dynamo_v2-0.4.0"
    )["text_only"]["observation"]["value"]["assembled"][0]["text"] == "new capture"
    assert yaml.safe_load(history_path.read_text())["captures"]["dynamo_v2-0.4.0"][
        "provenance"
    ] == {"status": "captured", "record": {"label": "0.4.0"}}
    assert all(
        path == history_path or path.read_bytes() == before[path.relative_to(root)]
        for path in root.rglob("*.yaml")
    )


def test_update_from_loose_rejects_mixed_provenance_in_new_capture(tmp_path):
    root = _store(tmp_path / "store")
    family_path = root / "families/gemma4.yaml"
    family = yaml.safe_load(family_path.read_text())
    family["cases"]["new_case"] = _case("UNIFIED.1-2", "new_case", "new")
    family_path.write_text(unified_history.dump_yaml(family))
    capture = tmp_path / "loose/dynamo_v2-0.4.0/gemma4"
    capture.mkdir(parents=True)
    for case_key, text, metadata in (
        ("UNIFIED.1-1", "hello", {"capture_provenance": {"label": "0.4.0"}}),
        ("UNIFIED.1-2", "new", {}),
    ):
        (capture / f"{case_key}.yaml").write_text(
            unified_history.dump_yaml(
                {
                    "family": "gemma4",
                    **metadata,
                    "cases": {
                        case_key: {
                            "capture_input": _request(text),
                            "assembled": [{"kind": "text", "text": text}],
                            "chunks": [],
                        }
                    },
                }
            )
        )

    with pytest.raises(ValueError, match="one complete provenance identity"):
        unified_history.update_from_loose(root, capture.parents[1])


def test_update_from_loose_appends_new_case_to_existing_source_identity(tmp_path):
    root = _store(tmp_path / "store")
    loose = tmp_path / "loose"
    family_path = root / "families/gemma4.yaml"
    family = yaml.safe_load(family_path.read_text())
    family["cases"]["new_case"] = _case("UNIFIED.1-2", "new_case", "new")
    family_path.write_text(unified_history.dump_yaml(family))

    capture = loose / "dynamo_v2-0.3.0/gemma4"
    capture.mkdir(parents=True)
    for case_key, text, request_text in (
        ("UNIFIED.1-1", "changed", "different request"),
        ("UNIFIED.1-2", "new", "new"),
    ):
        captured_with = "0.1.0" if case_key == "UNIFIED.1-1" else "0.3.0"
        (capture / f"{case_key}.yaml").write_text(
            unified_history.dump_yaml(
                {
                    "family": "gemma4",
                    "captured_with": {"dynamo_v2": captured_with},
                    "cases": {
                        case_key: {
                            "capture_input": _request(request_text),
                            "assembled": [{"kind": "text", "text": text}],
                            "chunks": [],
                        }
                    },
                }
            )
        )

    changed = unified_history.update_from_loose(root, loose)

    history_path = root / "capture_history/gemma4/dynamo_v2.yaml"
    assert changed == [history_path]
    resolved = unified_history.load_store(root).histories[("gemma4", "dynamo_v2")].resolve(
        "dynamo_v2-0.3.0"
    )
    assert set(resolved) == {"text_only", "new_case"}


def test_update_from_loose_preserves_document_metadata_change(tmp_path):
    root = _store(tmp_path / "store")
    loose = tmp_path / "loose/dynamo_v2-0.3.0/gemma4"
    loose.mkdir(parents=True)
    (loose / "UNIFIED.1-1.yaml").write_text(
        unified_history.dump_yaml(
            {
                "family": "gemma4",
                "mode": "changed-mode",
                "captured_with": {"dynamo_v2": "0.1.0"},
                "cases": {
                    "UNIFIED.1-1": {
                        "capture_input": _request("different request"),
                        "assembled": [{"kind": "text", "text": "changed"}],
                        "chunks": [],
                    }
                },
            }
        )
    )

    changed = unified_history.update_from_loose(root, loose.parents[1])

    assert changed == [root / "capture_history/gemma4/dynamo_v2.yaml"]
    destination = tmp_path / "materialized"
    unified_history.materialize_store(root, destination)
    document = yaml.safe_load(
        (destination / "dynamo_v2-0.3.0/gemma4/UNIFIED.1-1.yaml").read_text()
    )
    assert document["mode"] == "changed-mode"


def test_update_from_loose_rejects_existing_case_provenance_rewrite(tmp_path):
    root = _store(tmp_path / "store")
    loose = tmp_path / "loose/dynamo_v2-0.3.0/gemma4"
    loose.mkdir(parents=True)
    (loose / "UNIFIED.1-1.yaml").write_text(
        unified_history.dump_yaml(
            {
                "family": "gemma4",
                "captured_with": {"dynamo_v2": "different-producer"},
                "cases": {
                    "UNIFIED.1-1": {
                        "capture_input": _request("different request"),
                        "assembled": [{"kind": "text", "text": "changed"}],
                        "chunks": [],
                    }
                },
            }
        )
    )

    with pytest.raises(ValueError, match="provenance is immutable"):
        unified_history.update_from_loose(root, loose.parents[1])


def test_update_from_loose_keeps_first_checkout_provenance_for_same_source_identity(tmp_path):
    root = _store(tmp_path / "store")
    history_path = root / "capture_history/gemma4/dynamo_v2.yaml"
    family_path = root / "families/gemma4.yaml"
    family = yaml.safe_load(family_path.read_text())
    family["cases"]["same_source_new_case"] = _case("UNIFIED.1-2", "same_source_new_case", "new")
    family_path.write_text(unified_history.dump_yaml(family))
    history = yaml.safe_load(history_path.read_text())
    original = {
        "kind": "unpublished",
        "crate_version": "0.1.0",
        "source_id": "sha256:" + "a" * 64,
        "source_sha256": "a" * 64,
        "source_paths": ["parsers/v2/src"],
        "git_commit": "old-commit",
        "git_head_tree": "old-tree",
    }
    rerun = {**original, "git_commit": "new-commit", "git_head_tree": "new-tree"}
    capture = history["captures"]["dynamo_v2-0.3.0"]
    capture["provenance"] = {"status": "captured", "record": original}
    capture["changes"]["text_only"]["document"] = {"capture_provenance": original}
    history_path.write_text(unified_history.dump_yaml(history))

    loose = tmp_path / "loose/dynamo_v2-0.3.0/gemma4"
    loose.mkdir(parents=True)
    (loose / "UNIFIED.1-1.yaml").write_text(
        unified_history.dump_yaml(
            {
                "family": "gemma4",
                "capture_provenance": rerun,
                "cases": {
                    "UNIFIED.1-1": {
                        "capture_input": _request("different request"),
                        "assembled": [{"kind": "text", "text": "changed"}],
                        "chunks": [],
                    }
                },
            }
        )
    )
    (loose / "UNIFIED.1-2.yaml").write_text(
        unified_history.dump_yaml(
            {
                "family": "gemma4",
                "capture_provenance": rerun,
                "cases": {
                    "UNIFIED.1-2": {
                        "capture_input": _request("new"),
                        "assembled": [{"kind": "text", "text": "new"}],
                        "chunks": [],
                    }
                },
            }
        )
    )

    assert unified_history.update_from_loose(root, loose.parents[1]) == [history_path]
    stored = unified_history.load_store(root).histories[("gemma4", "dynamo_v2")]
    assert stored.captures["dynamo_v2-0.3.0"]["provenance"]["record"] == original
    assert "same_source_new_case" in stored.resolve("dynamo_v2-0.3.0")


def test_update_from_loose_rejects_changed_source_identity_for_unpublished_capture(tmp_path):
    root = _store(tmp_path / "store")
    history_path = root / "capture_history/gemma4/dynamo_v2.yaml"
    history = yaml.safe_load(history_path.read_text())
    original = {
        "kind": "unpublished",
        "crate_version": "0.1.0",
        "source_id": "sha256:" + "a" * 64,
        "source_sha256": "a" * 64,
        "source_paths": ["parsers/v2/src"],
    }
    changed_source = {
        **original,
        "source_id": "sha256:" + "b" * 64,
        "source_sha256": "b" * 64,
    }
    capture = history["captures"]["dynamo_v2-0.3.0"]
    capture["provenance"] = {"status": "captured", "record": original}
    capture["changes"]["text_only"]["document"] = {"capture_provenance": original}
    history_path.write_text(unified_history.dump_yaml(history))

    loose = tmp_path / "loose/dynamo_v2-0.3.0/gemma4"
    loose.mkdir(parents=True)
    (loose / "UNIFIED.1-1.yaml").write_text(
        unified_history.dump_yaml(
            {
                "family": "gemma4",
                "capture_provenance": changed_source,
                "cases": {
                    "UNIFIED.1-1": {
                        "capture_input": _request("different request"),
                        "assembled": [{"kind": "text", "text": "changed"}],
                        "chunks": [],
                    }
                },
            }
        )
    )

    with pytest.raises(ValueError, match="provenance is immutable"):
        unified_history.update_from_loose(root, loose.parents[1])


def test_update_from_loose_rejects_new_case_with_different_capture_identity(tmp_path):
    root = _store(tmp_path / "store")
    loose = tmp_path / "loose"
    family_path = root / "families/gemma4.yaml"
    family = yaml.safe_load(family_path.read_text())
    family["cases"]["new_case"] = _case("UNIFIED.1-2", "new_case", "new")
    family_path.write_text(unified_history.dump_yaml(family))

    capture = loose / "dynamo_v2-0.3.0/gemma4"
    capture.mkdir(parents=True)
    (capture / "UNIFIED.1-2.yaml").write_text(
        unified_history.dump_yaml(
            {
                "family": "gemma4",
                "captured_with": {"dynamo_v2": "different-producer"},
                "cases": {
                    "UNIFIED.1-2": {
                        "capture_input": _request("new"),
                        "assembled": [{"kind": "text", "text": "new"}],
                        "chunks": [],
                    }
                },
            }
        )
    )

    with pytest.raises(ValueError, match="provenance differs"):
        unified_history.update_from_loose(root, loose)


@pytest.mark.parametrize(
    "identity",
    [
        {},
        {"capture_provenance": None},
        {"captured_with": None},
    ],
)
def test_update_from_loose_rejects_new_case_without_capture_identity(tmp_path, identity):
    root = _store(tmp_path / "store")
    loose = tmp_path / "loose"
    family_path = root / "families/gemma4.yaml"
    family = yaml.safe_load(family_path.read_text())
    family["cases"]["new_case"] = _case("UNIFIED.1-2", "new_case", "new")
    family_path.write_text(unified_history.dump_yaml(family))

    capture = loose / "dynamo_v2-0.3.0/gemma4"
    capture.mkdir(parents=True)
    (capture / "UNIFIED.1-2.yaml").write_text(
        unified_history.dump_yaml(
            {
                "family": "gemma4",
                **identity,
                "cases": {
                    "UNIFIED.1-2": {
                        "capture_input": _request("new"),
                        "assembled": [{"kind": "text", "text": "new"}],
                        "chunks": [],
                    }
                },
            }
        )
    )

    with pytest.raises(ValueError, match="provenance differs"):
        unified_history.update_from_loose(root, loose)


def test_update_from_loose_keeps_existing_capture_immutable_and_ignores_missing_root(tmp_path):
    root = _store(tmp_path / "store")
    assert unified_history.update_from_loose(root, tmp_path / "missing") == []

    loose = tmp_path / "loose/dynamo_v2-0.1.0/gemma4"
    loose.mkdir(parents=True)
    (loose / "UNIFIED.1-1.yaml").write_text(
        unified_history.dump_yaml(
            {
                "family": "gemma4",
                "cases": {"UNIFIED.1-1": {"assembled": [{"kind": "text", "text": "changed"}]}},
            }
        )
    )

    with pytest.raises(ValueError, match="capture is immutable"):
        unified_history.update_from_loose(root, loose.parents[1])


def test_sync_current_corpus_ignores_missing_tree_without_rewriting_store(tmp_path):
    root = _store(tmp_path / "store")
    before = {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*.yaml")
    }

    assert unified_history.sync_current_corpus(root, tmp_path / "missing") == []
    assert {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*.yaml")
    } == before
