# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import itertools
import multiprocessing
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import capture_stimulus  # noqa: E402
import generate_conformance_table as table  # noqa: E402
import package_fixtures  # noqa: E402
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
    path = root / "families" / family / "inputs_and_golden.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        unified_history.dump_yaml(
            {
                "schema_version": 2,
                "family": family,
                "input_document": {"family": family, "mode": "unified"},
                "golden_document": {"family": family, "mode": "unified"},
                "cases": cases,
            }
        )
    )


def _write_history(root: Path, family: str, implementation: str, captures: dict) -> None:
    directory = root / "families" / family
    directory.mkdir(parents=True, exist_ok=True)
    for capture_id, capture in captures.items():
        document = {
            "schema_version": 2,
            "family": family,
            "implementation": implementation,
            **capture,
        }
        (directory / f"{capture_id}.yaml").write_text(unified_history.dump_yaml(document))


def _family_path(root: Path, family: str) -> Path:
    return root / "families" / family / "inputs_and_golden.yaml"


def _capture_path(root: Path, family: str, implementation: str, version: str) -> Path:
    return root / "families" / family / f"{implementation}-{version}.yaml"


def _case(display_id: str, scenario: str, text: str, *, aliases=()) -> dict:
    return {
        "lifecycle": "active",
        "scenario": scenario,
        "description": f"Description for {scenario}",
        "policy": ["test-policy"],
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


def _capture(
    parent,
    completeness: str,
    changes: dict,
    *,
    version="0.1.0",
    implementation="dynamo_v2",
) -> dict:
    document_by_case = {
        case_id: change.pop("document", {})
        for case_id, change in ((case_id, dict(change)) for case_id, change in changes.items())
        if change != {"absent": True}
    }
    normalized_changes = {
        case_id: change
        for case_id, change in (
            (case_id, dict(change)) for case_id, change in changes.items()
        )
    }
    for case_id, change in normalized_changes.items():
        if change != {"absent": True}:
            change.pop("document", None)
    provenance_documents = [
        document
        for document in document_by_case.values()
        if document
    ]
    identities = {
        unified_history._canonical_json(
            unified_history._document_provenance(document)
        ): unified_history._document_provenance(document)
        for document in provenance_documents
    }
    if len(identities) == 1:
        provenance = {
            "status": "legacy",
            "captured_with": {implementation: version},
        }
        document_overrides = {}
    elif identities:
        provenance = {"status": "mixed", "record_count": len(identities)}
        document_overrides = {
            case_id: document
            for case_id, document in document_by_case.items()
            if document
        }
    else:
        provenance = {"status": "legacy", "captured_with": {implementation: version}}
        document_overrides = {}
    return {
        "parent": parent,
        "runtime_version": version,
        "provenance": provenance,
        "completeness": completeness,
        "document": {},
        "import_lineage": [],
        "changes": normalized_changes,
        "metadata_changes": {},
        "document_overrides": document_overrides,
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
                "description": None,
                "policy": [],
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
                implementation="vllm_python",
            )
        },
    )
    return root


def _write_loose_current(root: Path, family: str, cases: list[tuple[str, str, str]]) -> None:
    for case_key, scenario, text in cases:
        input_path = root / "inputs" / family / f"{case_key}.yaml"
        golden_path = root / "golden" / family / f"{case_key}.yaml"
        input_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        request = _request(text)
        input_path.write_text(
            unified_history.dump_yaml(
                {
                    "family": family,
                    "cases": {
                        case_key: {
                            "scenario": scenario,
                            "description": f"Description for {scenario}",
                            "policy": ["test-policy"],
                            "init": request["init"],
                            "finish_reason": request["finish_reason"],
                            "input": request["input"],
                            "tools": request["tools"],
                            "chunks": request["chunks"],
                        }
                    },
                }
            )
        )
        golden_path.write_text(
            unified_history.dump_yaml(
                {
                    "family": family,
                    "cases": {
                        case_key: {
                            "assembled": [{"kind": "text", "text": text}],
                        }
                    },
                }
            )
        )


def _ingest_loose_source(root: Path, loose: Path, source: str) -> list[Path]:
    if source == "capture":
        return unified_history.update_from_loose(root, loose)
    return unified_history.sync_current_corpus(root, loose)


def _add_new_gemma_case(root: Path) -> None:
    family_path = _family_path(root, "gemma4")
    family = yaml.safe_load(family_path.read_text())
    family["cases"]["new_case"] = _case("UNIFIED.1-2", "new_case", "new")
    family_path.write_text(unified_history.dump_yaml(family))


def _write_new_gemma_capture(
    loose_root: Path,
    version: str,
    output: str,
    *,
    display_version: str | None = None,
    capture_provenance: dict | None = None,
) -> None:
    capture = loose_root / f"dynamo_v2-{version}/gemma4"
    capture.mkdir(parents=True, exist_ok=True)
    document = {
        "family": "gemma4",
        "captured_with": {"dynamo_v2": display_version or version},
        "cases": {
            "UNIFIED.1-2": {
                "capture_input": _request("new"),
                "assembled": [{"kind": "text", "text": output}],
                "chunks": [],
            }
        },
    }
    if capture_provenance is not None:
        document["capture_provenance"] = capture_provenance
    (capture / "UNIFIED.1-2.yaml").write_text(unified_history.dump_yaml(document))


def _set_capture_record_provenance(
    root: Path,
    version: str,
    provenance: dict,
) -> Path:
    history_path = _capture_path(root, "gemma4", "dynamo_v2", version)
    history = yaml.safe_load(history_path.read_text())
    status = "legacy"
    history["provenance"] = {
        "status": status,
        "record": provenance,
    }
    if status == "legacy":
        history.setdefault("document_overrides", {})
        for case_id, change in history["changes"].items():
            if change != {"absent": True}:
                history["document_overrides"][case_id] = {
                    "capture_provenance": provenance,
                }
    history_path.write_text(unified_history.dump_yaml(history))
    return history_path


def _unpublished_provenance(version: str, marker: str = "a") -> dict:
    source_sha256 = marker * 64
    return {
        "kind": "unpublished",
        "crate_version": version,
        "source_id": f"sha256:{source_sha256}",
        "source_sha256": source_sha256,
        "source_paths": ["parsers/v2/src"],
    }


def test_snapshot_delta_tombstone_and_unchanged_capture_resolution(tmp_path):
    store = unified_history.load_store(_store(tmp_path))
    history = store.histories[("gemma4", "dynamo_v2")]

    first = history.resolve("dynamo_v2-0.1.0")
    unchanged = history.resolve("dynamo_v2-0.2.0")
    changed = history.resolve("dynamo_v2-0.3.0")

    assert {
        case_id: record["observation"] for case_id, record in unchanged.items()
    } == {
        case_id: record["observation"] for case_id, record in first.items()
    }
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
        ("schema_version: 1\nfamily: *family\ncases: {}\n", "aliases"),
        ("schema_version: 1\nfamily: gemma4\ncases: !!map {}\n", "tags"),
    ],
)
def test_rejects_ambiguous_yaml(tmp_path, text, message):
    path = _family_path(tmp_path, "gemma4")
    path.parent.mkdir(parents=True)
    path.write_text(text)

    with pytest.raises(ValueError, match=message):
        unified_history.load_store(tmp_path)


def test_load_yaml_parses_once_without_a_scanner_pass(tmp_path, monkeypatch):
    path = tmp_path / "document.yaml"
    path.write_text("schema_version: 1\n")
    monkeypatch.setattr(yaml, "scan", lambda _text: pytest.fail("unexpected scanner pass"))

    assert unified_history.load_yaml(path) == {"schema_version": 1}


def test_strict_loader_uses_c_parser_with_python_security_composer():
    assert issubclass(unified_history.StrictLoader, unified_history.CParser)
    assert (
        unified_history.StrictLoader.get_single_node
        is unified_history.Composer.get_single_node
    )


def test_load_yaml_reports_malformed_document_path(tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text("schema_version: [\n")

    with pytest.raises(ValueError, match=rf"invalid YAML: {re.escape(str(path))}"):
        unified_history.load_yaml(path)


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

    with pytest.raises(ValueError, match="parent cycle|missing parent|one graph root"):
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


def test_rejects_different_cases_with_ids_that_share_one_canonical_id(tmp_path):
    root = _store(tmp_path)
    path = _family_path(root, "gemma4")
    family = yaml.safe_load(path.read_text())
    family["cases"]["text_only"]["historical_ids"] = []
    family["cases"]["legacy"] = _case("UNIFIED.1.a", "legacy", "legacy")
    path.write_text(unified_history.dump_yaml(family))

    with pytest.raises(ValueError, match="duplicate canonical ID"):
        unified_history.load_store(root)


@pytest.mark.parametrize("display_id", ["/tmp/escaped", "../escaped", "nested/escaped", "nested\\escaped"])
def test_rejects_path_bearing_display_id(tmp_path, display_id):
    root = _store(tmp_path)
    path = _family_path(root, "gemma4")
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
    history_path = _capture_path(root, "gemma4", "dynamo_v2", "0.1.0")
    history = yaml.safe_load(history_path.read_text())
    history["changes"]["text_only"]["observation"] = observation
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
    input_document = yaml.safe_load(
        (first / "inputs/gemma4/UNIFIED.1-1.yaml").read_text()
    )
    input_case = input_document["cases"]["UNIFIED.1-1"]
    assert list(input_case) == [
        "scenario",
        "description",
        "policy",
        "init",
        "finish_reason",
        "input",
        "tools",
        "chunks",
    ]
    assert input_case["description"] == "Description for text_only"
    assert input_case["policy"] == ["test-policy"]
    rendered_cases, _capabilities, _versions = table._load_unified_fixtures(first)
    rendered = next(
        case
        for case in rendered_cases
        if case["family"] == "gemma4" and case["scenario"] == "text_only"
    )
    assert rendered["description"] == "Description for text_only"
    assert rendered["policy_tags"] == ["test-policy"]


def test_materializer_is_deterministic_across_hash_seeds(tmp_path):
    source = _store(tmp_path / "source")
    materializer = "\n".join(
        [
            "from pathlib import Path",
            "import sys",
            "sys.path.insert(0, sys.argv[1])",
            "import unified_history",
            "unified_history.materialize_store(Path(sys.argv[2]), Path(sys.argv[3]))",
        ]
    )
    outputs = []
    for seed in ("1", "2"):
        destination = tmp_path / f"materialized-{seed}"
        subprocess.run(
            [sys.executable, "-c", materializer, str(SRC), str(source), str(destination)],
            check=True,
            env=os.environ | {"PYTHONHASHSEED": seed},
        )
        outputs.append(
            {
                path.relative_to(destination): path.read_bytes()
                for path in destination.rglob("*")
                if path.is_file()
            }
        )

    assert outputs[0] == outputs[1]


def test_materialized_writer_uses_serial_path_in_daemon_process(tmp_path, monkeypatch):
    documents = {
        tmp_path / f"case-{index}.yaml": {"case": index}
        for index in range(256)
    }
    monkeypatch.setattr(
        unified_history.multiprocessing,
        "current_process",
        lambda: SimpleNamespace(daemon=True),
    )

    def reject_child_pool(_worker_count):
        raise AssertionError("daemon process attempted to create a child pool")

    monkeypatch.setattr(unified_history.multiprocessing, "Pool", reject_child_pool)

    unified_history._write_materialized_documents(documents)

    assert len(list(tmp_path.glob("*.yaml"))) == len(documents)


def test_store_digest_hashes_bytes_without_reparsing(tmp_path, monkeypatch):
    root = _store(tmp_path)
    expected = unified_history.store_digest(root)

    def unexpected_parse(_root):
        raise AssertionError("store_digest must not parse the complete YAML store")

    monkeypatch.setattr(unified_history, "load_store", unexpected_parse)
    assert unified_history.store_digest(root) == expected


def test_materializer_preserves_capture_provenance_from_document_history(tmp_path):
    source = _store(tmp_path / "source")
    history_path = _capture_path(source, "gemma4", "dynamo_v2", "0.1.0")
    history = yaml.safe_load(history_path.read_text())
    history["provenance"] = {
        "status": "legacy",
        "record": {"label": "0.1.0", "source_sha256": "a" * 64},
    }
    history["document_overrides"]["text_only"] = {
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
    history_path = _capture_path(source, "gemma4", "dynamo_v2", "0.2.0")
    history = yaml.safe_load(history_path.read_text())
    capture = history
    capture["provenance"] = {"status": "captured", "record": {"label": "0.2.0"}}
    history_path.write_text(unified_history.dump_yaml(history))

    destination = tmp_path / "materialized"
    unified_history.materialize_store(source, destination)

    document = yaml.safe_load(
        (destination / "dynamo_v2-0.2.0/gemma4/UNIFIED.1-1.yaml").read_text()
    )
    assert document["captured_with"] == {"dynamo_v2": "0.2.0"}
    assert document["capture_provenance"] == {"label": "0.2.0"}


def test_materializer_does_not_replace_inherited_captured_with(tmp_path):
    source = _store(tmp_path / "source")
    destination = tmp_path / "materialized"

    unified_history.materialize_store(source, destination)

    document = yaml.safe_load(
        (destination / "dynamo_v2-0.2.0/gemma4/UNIFIED.1-1.yaml").read_text()
    )
    assert document["captured_with"] == {"dynamo_v2": "0.2.0"}


def test_materializer_applies_document_metadata_delta(tmp_path):
    source = _store(tmp_path / "source")
    history_path = _capture_path(source, "gemma4", "dynamo_v2", "0.2.0")
    history = yaml.safe_load(history_path.read_text())
    history["document_overrides"] = {
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
    assert document["captured_with"] == {"dynamo_v2": "0.2.0"}


def test_materializer_keeps_partial_stimulus_unavailable_to_complete_input_consumer(tmp_path):
    source = _store(tmp_path / "source")
    history_path = _capture_path(source, "gemma4", "dynamo_v2", "0.3.0")
    history = yaml.safe_load(history_path.read_text())
    tools = [
        {
            "type": "function",
            "function": {"name": "get_weather", "parameters": {"type": "object"}},
        }
    ]
    history["changes"]["text_only"]["stimulus"] = {
        "partial": {"input": "different request", "tools": tools}
    }
    history_path.write_text(unified_history.dump_yaml(history))

    destination = tmp_path / "materialized"
    unified_history.materialize_store(source, destination)
    path = destination / "dynamo_v2-0.3.0/gemma4/UNIFIED.1-1.yaml"
    document = yaml.safe_load(path.read_text())
    record = document["cases"]["UNIFIED.1-1"]

    assert "capture_input" not in record
    assert capture_stimulus.comparison_failure(
        record,
        _request("hello"),
        path.read_bytes(),
        "gemma4/UNIFIED.1-1.yaml",
        {},
    ).startswith("Capture stimulus unavailable")


def test_current_reference_requires_exact_canonical_request(tmp_path):
    root = _store(tmp_path)
    path = _capture_path(root, "gemma4", "dynamo_v2", "0.1.0")
    history = yaml.safe_load(path.read_text())
    history["changes"]["text_only"]["stimulus"] = {
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

    history_path = _capture_path(root, "gemma4", "dynamo_v2", "0.4.0")
    assert changed == [history_path]
    assert unified_history.load_store(root).histories[("gemma4", "dynamo_v2")].resolve(
        "dynamo_v2-0.4.0"
    )["text_only"]["observation"]["value"]["assembled"][0]["text"] == "new capture"
    assert yaml.safe_load(history_path.read_text())["provenance"] == {
        "status": "captured",
        "record": {"label": "0.4.0"},
    }
    history = unified_history.load_store(root).histories[("gemma4", "dynamo_v2")]
    assert set(history.captures) == {
        "dynamo_v2-0.1.0",
        "dynamo_v2-0.2.0",
        "dynamo_v2-0.3.0",
        "dynamo_v2-0.4.0",
    }
    assert history.captures["dynamo_v2-0.2.0"]["parent"] == "dynamo_v2-0.1.0"
    assert all(
        path == history_path or path.read_bytes() == before[path.relative_to(root)]
        for path in root.rglob("*.yaml")
    )


@pytest.mark.parametrize(
    ("implementation", "runtime_version", "identity"),
    [
        (
            "dynamo_v2",
            "0.4.0",
            {
                "capture_provenance": {
                    "label": "0.4.0",
                    "kind": "release",
                    "crate_version": "0.4.0",
                    "source_id": "sha256:" + "a" * 64,
                    "source_sha256": "a" * 64,
                    "source_paths": ["parsers/v2/src"],
                }
            },
        ),
        (
            "dynamo_v2",
            "0.4.0.patch2",
            {
                "capture_provenance": {
                    "label": "0.4.0",
                    "kind": "release",
                    "crate_version": "0.4.0",
                    "source_id": "sha256:" + "b" * 64,
                    "source_sha256": "b" * 64,
                    "source_paths": ["parsers/v2/src"],
                },
                "captured_with": {"dynamo_v2": "0.4.0"},
            },
        ),
        (
            "dynamo_v2",
            "0.4.0+source." + "c" * 64,
            {
                "capture_provenance": {
                    "label": "0.4.0+source." + "c" * 64,
                    "kind": "unpublished",
                    "crate_version": "0.4.0",
                    "source_id": "sha256:" + "c" * 64,
                    "source_sha256": "c" * 64,
                    "source_paths": ["parsers/v2/src"],
                },
                "captured_with": {"dynamo_v2": "0.4.0+source." + "c" * 64},
            },
        ),
        (
            "vllm_python",
            "0.26.0.patch1",
            {"captured_with": {"vllm_python": "0.26.0"}},
        ),
    ],
)
def test_new_capture_provenance_matches_normalized_runtime_identity(
    implementation,
    runtime_version,
    identity,
):
    records = {"case": {"document": identity}}

    provenance = unified_history._capture_provenance(
        records,
        f"{implementation}-{runtime_version}",
        implementation,
        runtime_version,
        strict=True,
    )

    assert provenance["status"] == "captured"


@pytest.mark.parametrize(
    ("implementation", "runtime_version", "identity"),
    [
        (
            "dynamo_v2",
            "0.4.0",
            {"captured_with": {"dynamo_v2": "9.9.9"}},
        ),
        (
            "vllm_python",
            "0.26.0.patch1",
            {"captured_with": {"vllm_python": "0.26.0.patch1"}},
        ),
        (
            "dynamo_v2",
            "0.4.0",
            {
                "capture_provenance": {
                    "label": "9.9.9",
                    "kind": "release",
                    "crate_version": "0.4.0",
                    "source_id": "sha256:" + "a" * 64,
                    "source_sha256": "a" * 64,
                    "source_paths": ["parsers/v2/src"],
                }
            },
        ),
        (
            "dynamo_v2",
            "0.4.0+source." + "a" * 64,
            {
                "capture_provenance": {
                    "label": "0.4.0+source." + "a" * 64,
                    "kind": "unpublished",
                    "crate_version": "0.4.0",
                    "source_id": "sha256:" + "a" * 64,
                    "source_sha256": "a" * 64,
                    "source_paths": [],
                }
            },
        ),
        (
            "dynamo_v2",
            "0.4.0+source." + "a" * 64,
            {
                "capture_provenance": {
                    "label": "0.4.0+source." + "a" * 64,
                    "kind": "unpublished",
                    "crate_version": "0.4.0",
                    "source_id": "sha256:" + "b" * 64,
                    "source_sha256": "b" * 64,
                    "source_paths": ["parsers/v2/src"],
                }
            },
        ),
    ],
)
def test_new_capture_provenance_rejects_mismatched_or_malformed_identity(
    implementation,
    runtime_version,
    identity,
):
    records = {"case": {"document": identity}}

    with pytest.raises(ValueError, match="runtime identity|invalid producer provenance"):
        unified_history._capture_provenance(
            records,
            f"{implementation}-{runtime_version}",
            implementation,
            runtime_version,
            strict=True,
        )


def test_update_from_loose_rejects_new_capture_with_wrong_display_version(tmp_path):
    root = _store(tmp_path / "store")
    _add_new_gemma_case(root)
    loose = tmp_path / "loose"
    _write_new_gemma_capture(
        loose,
        "0.4.0",
        "new",
        display_version="9.9.9",
    )

    with pytest.raises(ValueError, match="runtime identity"):
        unified_history.update_from_loose(root, loose)


@pytest.mark.parametrize(
    "fields",
    [
        ("assembled",),
        ("chunks",),
        ("assembled", "chunks"),
        ("error",),
        ("unavailable",),
    ],
)
def test_legacy_record_accepts_exactly_one_observation_state(fields):
    record = {"capture_input": _request("hello")}
    for field in fields:
        record[field] = [] if field in {"assembled", "chunks"} else field

    _stimulus, observation, _metadata = unified_history._legacy_state(
        record,
        b"",
        "gemma4/UNIFIED.1-1.yaml",
        {},
        None,
    )

    expected = "value" if set(fields) <= {"assembled", "chunks"} else fields[0]
    assert set(observation) == {expected}


@pytest.mark.parametrize(
    "fields",
    [
        fields
        for size in range(2, 5)
        for fields in itertools.combinations(
            ("assembled", "chunks", "error", "unavailable"), size
        )
        if not set(fields) <= {"assembled", "chunks"}
    ],
)
def test_legacy_record_rejects_conflicting_observation_states(fields):
    record = {"capture_input": _request("hello")}
    for field in fields:
        record[field] = [] if field in {"assembled", "chunks"} else field

    with pytest.raises(ValueError, match="exactly one observation state"):
        unified_history._legacy_state(
            record,
            b"",
            "gemma4/UNIFIED.1-1.yaml",
            {},
            None,
        )


def test_update_from_loose_uses_graph_leaf_after_history_reordering(tmp_path):
    root = _store(tmp_path / "store")
    loose = tmp_path / "loose/dynamo_v2-0.4.0/gemma4"
    loose.mkdir(parents=True)
    (loose / "UNIFIED.1-1.yaml").write_text(
        unified_history.dump_yaml(
            {
                "family": "gemma4",
                "captured_with": {"dynamo_v2": "0.4.0"},
                "cases": {
                    "UNIFIED.1-1": {
                        "assembled": [{"kind": "text", "text": "changed again"}],
                        "chunks": [],
                        "capture_input": _request("different request"),
                    }
                },
            }
        )
    )

    unified_history.update_from_loose(root, tmp_path / "loose")

    capture = yaml.safe_load(
        _capture_path(root, "gemma4", "dynamo_v2", "0.4.0").read_text()
    )
    assert capture["parent"] == "dynamo_v2-0.3.0"
    assert set(capture["changes"]) == {"text_only"}


def test_rejects_history_with_multiple_graph_leaves(tmp_path):
    root = _store(tmp_path / "store")
    history_path = _capture_path(root, "gemma4", "dynamo_v2", "0.3.0")
    history = yaml.safe_load(history_path.read_text())
    history["parent"] = "dynamo_v2-0.1.0"
    history_path.write_text(unified_history.dump_yaml(history))

    with pytest.raises(ValueError, match="one graph leaf"):
        unified_history.load_store(root)


@pytest.mark.parametrize(
    ("capture_id", "runtime_version"),
    [
        ("dynamo_v2-1.0.0", "1.0.0"),
        ("vllm_python-1.0.0", "2.0.0"),
    ],
)
def test_rejects_capture_identity_that_differs_from_history_owner(
    tmp_path,
    capture_id,
    runtime_version,
):
    root = _store(tmp_path / "store")
    history_path = _capture_path(root, "qwen3", "vllm_python", "1.0.0")
    history = yaml.safe_load(history_path.read_text())
    history["runtime_version"] = runtime_version
    history["implementation"] = capture_id.split("-", 1)[0]
    history_path.write_text(unified_history.dump_yaml(history))

    with pytest.raises(ValueError, match="capture identity differs"):
        unified_history.load_store(root)


def test_cross_family_capture_metadata_is_rejected_by_store_and_package(tmp_path):
    root = _store(tmp_path / "store")
    capture = _capture(None, "snapshot", {"text_only": _change("hi")})
    capture["provenance"] = {"status": "legacy", "record": {"label": "0.1.0"}}
    _write_history(
        root,
        "qwen3",
        "dynamo_v2",
        {"dynamo_v2-0.1.0": capture},
    )

    with pytest.raises(ValueError, match="capture metadata differs across families"):
        unified_history.load_store(root)

    digest, size = unified_history.store_digest(root)
    manifest = {
        "shards": [
            {
                "path": "unified-history",
                "format": "unified-history",
                "sha256": digest,
                "size": size,
            }
        ],
        "inactive_shards": [],
    }
    with pytest.raises(ValueError, match="capture metadata differs across families"):
        package_fixtures._validate_candidate_package(manifest, tmp_path / "fixtures", root)


def test_candidate_package_retains_every_import_lineage_archive(tmp_path):
    root = _store(tmp_path / "store")
    history_path = _capture_path(root, "gemma4", "dynamo_v2", "0.1.0")
    history = yaml.safe_load(history_path.read_text())
    archive_bytes = b"retained archive bytes"
    archive_hash = hashlib.sha256(archive_bytes).hexdigest()
    history["import_lineage"] = [
        {
            "archive": "dynamo_v2-0.1.0.tar.gz",
            "sha256": archive_hash,
            "mode": "snapshot",
        }
    ]
    history_path.write_text(unified_history.dump_yaml(history))
    fixtures = tmp_path / "fixtures"
    archive = fixtures / "unified/dynamo_v2-0.1.0.tar.gz"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(archive_bytes)
    digest, size = unified_history.store_digest(root)
    manifest = {
        "shards": [
            {
                "path": "unified-history",
                "format": "unified-history",
                "sha256": digest,
                "size": size,
            }
        ],
        "inactive_shards": [],
    }

    with pytest.raises(ValueError, match="import lineage archive is not retained"):
        package_fixtures._validate_candidate_package(manifest, fixtures, root)

    manifest["inactive_shards"] = [
        {
            "path": "unified/dynamo_v2-0.1.0.tar.gz",
            "sha256": archive_hash,
            "size": len(archive_bytes),
            "disposition": "superseded",
            "reason": "Canonical Unified YAML history supersedes this imported archive.",
        }
    ]
    package_fixtures._validate_candidate_package(manifest, fixtures, root)


@pytest.mark.parametrize("second_output", ["same", "different"])
def test_update_from_loose_rejects_duplicate_canonical_records(tmp_path, second_output):
    root = _store(tmp_path / "store")
    capture = tmp_path / "loose/dynamo_v2-0.4.0/gemma4"
    capture.mkdir(parents=True)
    for filename, case_key, output in (
        ("current", "UNIFIED.1-1", "same"),
        ("legacy", "UNIFIED.1.a", second_output),
    ):
        (capture / f"{filename}.yaml").write_text(
            unified_history.dump_yaml(
                {
                    "family": "gemma4",
                    "capture_provenance": {"label": "0.4.0"},
                    "cases": {
                        case_key: {
                            "capture_input": _request("hello"),
                            "assembled": [{"kind": "text", "text": output}],
                            "chunks": [],
                        }
                    },
                }
            )
        )

    with pytest.raises(ValueError, match="duplicate Unified records resolve to"):
        unified_history.update_from_loose(root, capture.parents[1])


def test_update_from_loose_rejects_mixed_provenance_in_new_capture(tmp_path):
    root = _store(tmp_path / "store")
    family_path = _family_path(root, "gemma4")
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


def test_update_from_loose_checks_provenance_on_unchanged_records(tmp_path):
    root = _store(tmp_path / "store")
    family_path = _family_path(root, "gemma4")
    family = yaml.safe_load(family_path.read_text())
    family["cases"]["new_case"] = _case("UNIFIED.1-2", "new_case", "new")
    family_path.write_text(unified_history.dump_yaml(family))
    capture = tmp_path / "loose/dynamo_v2-0.4.0/gemma4"
    capture.mkdir(parents=True)
    for case_key, request_text, output_text, identity in (
        ("UNIFIED.1-1", "different request", "changed", "identity-b"),
        ("UNIFIED.1-2", "new", "new", "identity-a"),
    ):
        (capture / f"{case_key}.yaml").write_text(
            unified_history.dump_yaml(
                {
                    "family": "gemma4",
                    "capture_provenance": {"label": identity},
                    "cases": {
                        case_key: {
                            "capture_input": _request(request_text),
                            "assembled": [{"kind": "text", "text": output_text}],
                            "chunks": [],
                        }
                    },
                }
            )
        )

    with pytest.raises(ValueError, match="one complete provenance identity"):
        unified_history.update_from_loose(root, capture.parents[1])


def test_update_from_loose_skips_exact_excluded_capture_directory(tmp_path):
    root = _store(tmp_path / "store")
    loose = tmp_path / "loose"
    capture = loose / "dynamo_v2-0.4.0/gemma4"
    capture.mkdir(parents=True)
    (capture / "UNIFIED.1-1.yaml").write_text(
        unified_history.dump_yaml(
            {
                "family": "gemma4",
                "capture_provenance": {"label": "inactive"},
                "cases": {
                    "UNIFIED.1-1": {
                        "capture_input": _request("hello"),
                        "assembled": [{"kind": "text", "text": "inactive"}],
                        "chunks": [],
                    }
                },
            }
        )
    )

    changed = unified_history.update_from_loose(
        root,
        loose,
        excluded_capture_dirs={"dynamo_v2-0.4.0"},
    )

    assert changed == []
    history = unified_history.load_store(root).histories[("gemma4", "dynamo_v2")]
    assert "dynamo_v2-0.4.0" not in history.captures


@pytest.mark.parametrize("source", ["inputs", "golden", "capture"])
def test_loose_ingress_rejects_duplicate_mapping_keys(tmp_path, source):
    root = _store(tmp_path / "store")
    loose = tmp_path / "loose"
    _write_loose_current(loose, "gemma4", [("UNIFIED.1-1", "text_only", "updated")])
    if source == "capture":
        path = loose / "dynamo_v2-0.4.0/gemma4/UNIFIED.1-1.yaml"
        path.parent.mkdir(parents=True)
    else:
        path = loose / source / "gemma4/UNIFIED.1-1.yaml"
    path.write_text("family: gemma4\ncases: {}\ncases: {}\n")
    before = {item.relative_to(root): item.read_bytes() for item in root.rglob("*.yaml")}

    with pytest.raises(ValueError, match="duplicate key"):
        _ingest_loose_source(root, loose, source)

    assert {item.relative_to(root): item.read_bytes() for item in root.rglob("*.yaml")} == before


@pytest.mark.parametrize("source", ["inputs", "golden", "capture"])
def test_loose_ingress_rejects_wrong_family_owner(tmp_path, source):
    root = _store(tmp_path / "store")
    loose = tmp_path / "loose"
    _write_loose_current(loose, "gemma4", [("UNIFIED.1-1", "text_only", "updated")])
    if source == "capture":
        path = loose / "dynamo_v2-0.4.0/gemma4/UNIFIED.1-1.yaml"
        path.parent.mkdir(parents=True)
        path.write_text("family: qwen3\ncases: {}\n")
    else:
        path = loose / source / "gemma4/UNIFIED.1-1.yaml"
        document = yaml.safe_load(path.read_text())
        document["family"] = "qwen3"
        path.write_text(unified_history.dump_yaml(document))
    before = {item.relative_to(root): item.read_bytes() for item in root.rglob("*.yaml")}

    with pytest.raises(ValueError, match="family differs from directory"):
        _ingest_loose_source(root, loose, source)

    assert {item.relative_to(root): item.read_bytes() for item in root.rglob("*.yaml")} == before


@pytest.mark.parametrize("source", ["inputs", "golden", "capture"])
def test_loose_ingress_requires_cases_mapping(tmp_path, source):
    root = _store(tmp_path / "store")
    loose = tmp_path / "loose"
    _write_loose_current(loose, "gemma4", [("UNIFIED.1-1", "text_only", "updated")])
    if source == "capture":
        path = loose / "dynamo_v2-0.4.0/gemma4/UNIFIED.1-1.yaml"
        path.parent.mkdir(parents=True)
    else:
        path = loose / source / "gemma4/UNIFIED.1-1.yaml"
    path.write_text("family: gemma4\ncases: []\n")

    with pytest.raises(ValueError, match="cases must be a mapping"):
        _ingest_loose_source(root, loose, source)


def test_update_from_loose_appends_new_case_to_existing_source_identity(tmp_path):
    root = _store(tmp_path / "store")
    loose = tmp_path / "loose"
    family_path = _family_path(root, "gemma4")
    family = yaml.safe_load(family_path.read_text())
    family["cases"]["new_case"] = _case("UNIFIED.1-2", "new_case", "new")
    family_path.write_text(unified_history.dump_yaml(family))
    provenance = _unpublished_provenance("0.3.0")
    history_path = _set_capture_record_provenance(root, "0.3.0", provenance)
    history = yaml.safe_load(history_path.read_text())
    history["document_overrides"]["text_only"] = {
        "capture_provenance": provenance
    }
    history_path.write_text(unified_history.dump_yaml(history))

    capture = loose / "dynamo_v2-0.3.0/gemma4"
    capture.mkdir(parents=True)
    for case_key, text, request_text in (
        ("UNIFIED.1-1", "changed", "different request"),
        ("UNIFIED.1-2", "new", "new"),
    ):
        (capture / f"{case_key}.yaml").write_text(
            unified_history.dump_yaml(
                {
                    "family": "gemma4",
                    "capture_provenance": provenance,
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

    assert changed == [history_path]
    resolved = unified_history.load_store(root).histories[("gemma4", "dynamo_v2")].resolve(
        "dynamo_v2-0.3.0"
    )
    assert set(resolved) == {"text_only", "new_case"}


@pytest.mark.parametrize("second_completeness", ["delta"])
def test_update_from_loose_backfills_each_capture_without_descendant_inheritance(
    tmp_path,
    second_completeness,
):
    root = _store(tmp_path / "store")
    loose = tmp_path / "loose"
    _add_new_gemma_case(root)
    provenances = {
        version: _unpublished_provenance(version, marker)
        for version, marker in zip(("0.1.0", "0.2.0", "0.3.0"), ("a", "b", "c"))
    }
    for version, provenance in provenances.items():
        _set_capture_record_provenance(root, version, provenance)
    if second_completeness == "snapshot":
        history_path = _capture_path(root, "gemma4", "dynamo_v2", "0.2.0")
        history = yaml.safe_load(history_path.read_text())
        history["completeness"] = "snapshot"
        history_path.write_text(unified_history.dump_yaml(history))

    def backfill(version: str, output: str) -> None:
        _write_new_gemma_capture(
            loose,
            version,
            output,
            capture_provenance=provenances[version],
        )
        unified_history.update_from_loose(root, loose)

    def resolved_output(version: str):
        history = unified_history.load_store(root).histories[("gemma4", "dynamo_v2")]
        change = history.resolve(f"dynamo_v2-{version}").get("new_case")
        if change is None:
            return None
        return change["observation"]["value"]["assembled"][0]["text"]

    backfill("0.1.0", "from-0.1.0")
    assert [resolved_output(version) for version in ("0.1.0", "0.2.0", "0.3.0")] == [
        "from-0.1.0",
        None,
        None,
    ]

    backfill("0.2.0", "from-0.2.0")
    assert [resolved_output(version) for version in ("0.1.0", "0.2.0", "0.3.0")] == [
        "from-0.1.0",
        "from-0.2.0",
        None,
    ]

    backfill("0.3.0", "from-0.3.0")
    assert [resolved_output(version) for version in ("0.1.0", "0.2.0", "0.3.0")] == [
        "from-0.1.0",
        "from-0.2.0",
        "from-0.3.0",
    ]


def test_update_from_loose_backfills_distinct_capture_results_in_one_update(tmp_path):
    root = _store(tmp_path / "store")
    loose = tmp_path / "loose"
    _add_new_gemma_case(root)
    for version, marker in zip(("0.1.0", "0.2.0", "0.3.0"), ("a", "b", "c")):
        provenance = _unpublished_provenance(version, marker)
        _set_capture_record_provenance(root, version, provenance)
        _write_new_gemma_capture(
            loose,
            version,
            f"from-{version}",
            capture_provenance=provenance,
        )

    unified_history.update_from_loose(root, loose)

    history = unified_history.load_store(root).histories[("gemma4", "dynamo_v2")]
    for version in ("0.1.0", "0.2.0", "0.3.0"):
        change = history.resolve(f"dynamo_v2-{version}")["new_case"]
        assert change["observation"]["value"]["assembled"][0]["text"] == f"from-{version}"


@pytest.mark.parametrize("child_has_observation", [False, True])
def test_update_from_loose_preserves_explicit_descendant_case_state(
    tmp_path,
    child_has_observation,
):
    root = _store(tmp_path / "store")
    _add_new_gemma_case(root)
    history_path = _capture_path(root, "gemma4", "dynamo_v2", "0.2.0")
    history = yaml.safe_load(history_path.read_text())
    child_change = _change("from-0.2.0") if child_has_observation else {"absent": True}
    if child_has_observation:
        child_change["case_key"] = "UNIFIED.1-2"
        child_change.pop("document", None)
    history["changes"]["new_case"] = child_change
    provenance = _unpublished_provenance("0.1.0")
    history_path = _capture_path(root, "gemma4", "dynamo_v2", "0.1.0")
    root_capture = yaml.safe_load(history_path.read_text())
    root_capture["provenance"] = {"status": "legacy", "record": provenance}
    history_path.write_text(unified_history.dump_yaml(root_capture))
    history_path = _capture_path(root, "gemma4", "dynamo_v2", "0.2.0")
    history_path.write_text(unified_history.dump_yaml(history))
    loose = tmp_path / "loose"
    _write_new_gemma_capture(
        loose,
        "0.1.0",
        "from-0.1.0",
        capture_provenance=provenance,
    )

    unified_history.update_from_loose(root, loose)

    stored = unified_history.load_store(root).histories[("gemma4", "dynamo_v2")]
    child = stored.resolve("dynamo_v2-0.2.0").get("new_case")
    descendant = stored.resolve("dynamo_v2-0.3.0").get("new_case")
    if child_has_observation:
        assert child["observation"]["value"]["assembled"][0]["text"] == "from-0.2.0"
        assert unified_history._semantic_change(descendant) == unified_history._semantic_change(child)
    else:
        assert child is None
        assert descendant is None
        assert stored.captures["dynamo_v2-0.2.0"]["changes"]["new_case"] == {
            "absent": True
        }


def test_update_from_loose_accepts_matching_producer_provenance_for_existing_capture(
    tmp_path,
):
    root = _store(tmp_path / "store")
    _add_new_gemma_case(root)
    provenance = {
        "kind": "unpublished",
        "crate_version": "0.3.0",
        "source_id": "sha256:" + "a" * 64,
        "source_sha256": "a" * 64,
        "source_paths": ["parsers/v2/src"],
    }
    history_path = _set_capture_record_provenance(root, "0.3.0", provenance)
    loose = tmp_path / "loose"
    _write_new_gemma_capture(
        loose,
        "0.3.0",
        "new",
        capture_provenance=provenance,
    )

    assert unified_history.update_from_loose(root, loose) == [history_path]
    stored = unified_history.load_store(root).histories[("gemma4", "dynamo_v2")]
    assert "new_case" in stored.resolve("dynamo_v2-0.3.0")


@pytest.mark.parametrize("mutation", ["addition", "record_metadata", "document_metadata"])
def test_update_from_loose_rejects_mutation_of_released_capture(tmp_path, mutation):
    root = _store(tmp_path / "store")
    provenance = {
        "kind": "release",
        "crate_version": "0.3.0",
        "source_id": "sha256:" + "a" * 64,
        "source_sha256": "a" * 64,
        "source_paths": ["parsers/v2/src"],
    }
    history_path = _set_capture_record_provenance(root, "0.3.0", provenance)
    history = yaml.safe_load(history_path.read_text())
    history["document_overrides"]["text_only"] = {
        "capture_provenance": provenance
    }
    history_path.write_text(unified_history.dump_yaml(history))
    loose_root = tmp_path / "loose"
    if mutation == "addition":
        _add_new_gemma_case(root)
        _write_new_gemma_capture(
            loose_root,
            "0.3.0",
            "new",
            capture_provenance=provenance,
        )
    else:
        capture = loose_root / "dynamo_v2-0.3.0/gemma4"
        capture.mkdir(parents=True)
        document = {
            "family": "gemma4",
            "capture_provenance": provenance,
            "cases": {
                "UNIFIED.1-1": {
                    "capture_input": _request("different request"),
                    "assembled": [{"kind": "text", "text": "changed"}],
                    "chunks": [],
                }
            },
        }
        if mutation == "record_metadata":
            document["cases"]["UNIFIED.1-1"]["engine_note"] = "changed"
        else:
            document["mode"] = "changed-mode"
        (capture / "UNIFIED.1-1.yaml").write_text(unified_history.dump_yaml(document))

    with pytest.raises(ValueError, match=r"released capture is immutable; use a new \.patchN"):
        unified_history.update_from_loose(root, loose_root)


@pytest.mark.parametrize("mutation", ["addition", "record_metadata", "document_metadata"])
def test_update_from_loose_rejects_mutation_of_versioned_peer_capture(tmp_path, mutation):
    root = _store(tmp_path / "store")
    history_path = _capture_path(root, "qwen3", "vllm_python", "1.0.0")
    history = yaml.safe_load(history_path.read_text())
    stored_change = history["changes"]["text_only"]
    stored_change["stimulus"] = {
        "ref": "current",
        "semantic_sha256": unified_history._request_digest(_request("hi")),
    }
    stored_change["observation"] = {
        "error": {"code": "legacy_unclassified", "detail": "peer error"}
    }
    history_path.write_text(unified_history.dump_yaml(history))
    if mutation == "addition":
        family_path = _family_path(root, "qwen3")
        family = yaml.safe_load(family_path.read_text())
        family["cases"]["new_case"] = _case("UNIFIED.1-2", "new_case", "new")
        family_path.write_text(unified_history.dump_yaml(family))

    loose_root = tmp_path / "loose"
    capture = loose_root / "vllm_python-1.0.0/qwen3"
    capture.mkdir(parents=True)
    path = capture / "UNIFIED.1-1.yaml"
    path.write_text(
        unified_history.dump_yaml(
            {
                "family": "qwen3",
                "captured_with": {"vllm_python": "1.0.0"},
                "cases": {
                    "UNIFIED.1-1": {
                        "capture_input": _request("hi"),
                        "error": "peer error",
                    }
                },
            }
        )
    )
    if mutation == "addition":
        (capture / "UNIFIED.1-2.yaml").write_text(
            unified_history.dump_yaml(
                {
                    "family": "qwen3",
                    "captured_with": {"vllm_python": "1.0.0"},
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
    else:
        document = yaml.safe_load(path.read_text())
        if mutation == "record_metadata":
            document["cases"]["UNIFIED.1-1"]["engine_note"] = "changed"
        else:
            document["mode"] = "changed-mode"
        path.write_text(unified_history.dump_yaml(document))

    with pytest.raises(ValueError, match=r"versioned capture is immutable; use a new \.patchN"):
        unified_history.update_from_loose(root, loose_root)


def test_update_from_loose_rejects_unverified_unpublished_identity(tmp_path):
    root = _store(tmp_path / "store")
    _add_new_gemma_case(root)
    incomplete = {"kind": "unpublished"}
    _set_capture_record_provenance(root, "0.3.0", incomplete)
    loose = tmp_path / "loose"
    _write_new_gemma_capture(
        loose,
        "0.3.0",
        "new",
        capture_provenance=incomplete,
    )

    with pytest.raises(ValueError, match=r"versioned capture is immutable; use a new \.patchN"):
        unified_history.update_from_loose(root, loose)


def test_update_from_loose_rejects_conflicting_producer_display_provenance(tmp_path):
    root = _store(tmp_path / "store")
    _add_new_gemma_case(root)
    provenance = {
        "kind": "unpublished",
        "crate_version": "0.3.0",
        "source_id": "sha256:" + "a" * 64,
        "source_sha256": "a" * 64,
        "source_paths": ["parsers/v2/src"],
    }
    _set_capture_record_provenance(root, "0.3.0", provenance)
    loose = tmp_path / "loose"
    _write_new_gemma_capture(
        loose,
        "0.3.0",
        "new",
        display_version="different-producer",
        capture_provenance=provenance,
    )

    with pytest.raises(ValueError, match="provenance differs"):
        unified_history.update_from_loose(root, loose)


def test_update_from_loose_preserves_document_metadata_change(tmp_path):
    root = _store(tmp_path / "store")
    provenance = _unpublished_provenance("0.3.0")
    history_path = _capture_path(root, "gemma4", "dynamo_v2", "0.3.0")
    history = yaml.safe_load(history_path.read_text())
    capture = history
    capture["provenance"] = {"status": "legacy", "record": provenance}
    capture["document_overrides"]["text_only"] = {"capture_provenance": provenance}
    history_path.write_text(unified_history.dump_yaml(history))
    loose = tmp_path / "loose/dynamo_v2-0.3.0/gemma4"
    loose.mkdir(parents=True)
    (loose / "UNIFIED.1-1.yaml").write_text(
        unified_history.dump_yaml(
            {
                "family": "gemma4",
                "mode": "changed-mode",
                "capture_provenance": provenance,
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

    assert changed == [history_path]
    destination = tmp_path / "materialized"
    unified_history.materialize_store(root, destination)
    document = yaml.safe_load(
        (destination / "dynamo_v2-0.3.0/gemma4/UNIFIED.1-1.yaml").read_text()
    )
    assert document["mode"] == "changed-mode"
    assert document["capture_provenance"] == provenance


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
    history_path = _capture_path(root, "gemma4", "dynamo_v2", "0.3.0")
    family_path = _family_path(root, "gemma4")
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
    capture = history
    capture["provenance"] = {"status": "legacy", "record": original}
    capture["document_overrides"]["text_only"] = {"capture_provenance": original}
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
    history_path = _capture_path(root, "gemma4", "dynamo_v2", "0.3.0")
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
    capture = history
    capture["provenance"] = {"status": "legacy", "record": original}
    capture["document_overrides"]["text_only"] = {"capture_provenance": original}
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
    _set_capture_record_provenance(root, "0.3.0", _unpublished_provenance("0.3.0"))
    loose = tmp_path / "loose"
    family_path = _family_path(root, "gemma4")
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
    _set_capture_record_provenance(root, "0.3.0", _unpublished_provenance("0.3.0"))
    loose = tmp_path / "loose"
    family_path = _family_path(root, "gemma4")
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


def test_update_from_loose_ignores_display_alias_change_for_existing_capture(tmp_path):
    root = _store(tmp_path / "store")
    family_path = _family_path(root, "gemma4")
    family = yaml.safe_load(family_path.read_text())
    family["cases"]["text_only"]["display_id"] = "UNIFIED.1-2"
    family["cases"]["text_only"]["historical_ids"] = ["UNIFIED.1-1", "UNIFIED.1.a"]
    family_path.write_text(unified_history.dump_yaml(family))

    loose_root = tmp_path / "loose/dynamo_v2-0.1.0/gemma4"
    loose_root.mkdir(parents=True)
    (loose_root / "UNIFIED.1-2.yaml").write_text(
        unified_history.dump_yaml(
            {
                "family": "gemma4",
                "captured_with": {"dynamo_v2": "0.1.0"},
                "cases": {
                    "UNIFIED.1-2": {
                        "capture_input": _request("hello"),
                        "assembled": [{"kind": "text", "text": "hello"}],
                        "chunks": [],
                    }
                },
            }
        )
    )
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*.yaml")}

    assert unified_history.update_from_loose(root, loose_root.parents[1]) == []
    assert {path.relative_to(root): path.read_bytes() for path in root.rglob("*.yaml")} == before


def test_complete_update_rejects_omitted_active_existing_capture_case(tmp_path):
    root = _store(tmp_path / "store")
    loose = tmp_path / "loose"
    _write_loose_current(loose, "gemma4", [("UNIFIED.1-1", "text_only", "hello")])
    _write_loose_current(loose, "qwen3", [("UNIFIED.1-1", "text_only", "hi")])
    (loose / "dynamo_v2-0.1.0/gemma4").mkdir(parents=True)
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*.yaml")}

    with pytest.raises(
        ValueError,
        match=r"complete capture dynamo_v2-0\.1\.0 is missing active cases: gemma4/text_only",
    ):
        unified_history.update_store_from_loose(root, loose, complete_snapshot=True)

    assert {path.relative_to(root): path.read_bytes() for path in root.rglob("*.yaml")} == before


def test_complete_update_rejects_missing_existing_capture_family(tmp_path):
    root = _store(tmp_path / "store")
    loose = tmp_path / "loose"
    _write_loose_current(loose, "gemma4", [("UNIFIED.1-1", "text_only", "hello")])
    _write_loose_current(loose, "qwen3", [("UNIFIED.1-1", "text_only", "hi")])
    (loose / "dynamo_v2-0.1.0").mkdir()
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*.yaml")}

    with pytest.raises(
        ValueError,
        match=r"complete capture dynamo_v2-0\.1\.0 is missing active families: gemma4",
    ):
        unified_history.update_store_from_loose(root, loose, complete_snapshot=True)

    assert {path.relative_to(root): path.read_bytes() for path in root.rglob("*.yaml")} == before


def test_complete_update_rejects_missing_required_capture_root(tmp_path):
    root = _store(tmp_path / "store")
    loose = tmp_path / "loose"
    _write_loose_current(loose, "gemma4", [("UNIFIED.1-1", "text_only", "hello")])
    _write_loose_current(loose, "qwen3", [("UNIFIED.1-1", "text_only", "hi")])
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*.yaml")}

    with pytest.raises(
        ValueError,
        match=r"complete snapshot is missing required captures: dynamo_v2-0\.3\.0",
    ):
        unified_history.update_store_from_loose(
            root,
            loose,
            complete_snapshot=True,
            required_capture_dirs={"dynamo_v2-0.3.0"},
        )

    assert {path.relative_to(root): path.read_bytes() for path in root.rglob("*.yaml")} == before


def test_complete_update_rejects_empty_required_capture_root(tmp_path):
    root = _store(tmp_path / "store")
    loose = tmp_path / "loose"
    _write_loose_current(loose, "gemma4", [("UNIFIED.1-1", "text_only", "hello")])
    _write_loose_current(loose, "qwen3", [("UNIFIED.1-1", "text_only", "hi")])
    capture = "dynamo_v2-9.9.9+source.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    (loose / capture).mkdir()
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*.yaml")}

    with pytest.raises(
        ValueError,
        match=rf"complete capture {re.escape(capture)} is missing active families: gemma4",
    ):
        unified_history.update_store_from_loose(
            root,
            loose,
            complete_snapshot=True,
            required_capture_dirs={capture},
        )

    assert {path.relative_to(root): path.read_bytes() for path in root.rglob("*.yaml")} == before


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


def test_sync_current_corpus_rejects_duplicate_scenario_owners(tmp_path):
    root = _store(tmp_path / "store")
    loose = tmp_path / "loose"
    _write_loose_current(
        loose,
        "gemma4",
        [
            ("UNIFIED.1-1", "text_only", "first"),
            ("UNIFIED.9-9", "text_only", "second"),
        ],
    )
    _write_loose_current(loose, "qwen3", [("UNIFIED.1-1", "text_only", "hi")])
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*.yaml")}

    with pytest.raises(ValueError, match="duplicate current Unified scenario"):
        unified_history.sync_current_corpus(root, loose, complete_snapshot=True)

    assert {path.relative_to(root): path.read_bytes() for path in root.rglob("*.yaml")} == before


def test_store_lock_serializes_real_and_symlinked_paths(tmp_path):
    real_store = _store(tmp_path / "real/store")
    alias_parent = tmp_path / "alias"
    alias_parent.mkdir()
    alias_store = alias_parent / "store-link"
    alias_store.symlink_to(real_store, target_is_directory=True)
    context = multiprocessing.get_context("fork")
    ready = context.Event()
    begin = context.Event()
    acquired = context.Event()

    def acquire_alias():
        ready.set()
        begin.wait()
        with unified_history._store_mutation_lock(alias_store):
            acquired.set()

    process = context.Process(target=acquire_alias)
    try:
        process.start()
        assert ready.wait(timeout=2)
        with unified_history._store_mutation_lock(real_store):
            begin.set()
            assert not acquired.wait(timeout=2)
        assert acquired.wait(timeout=2)
        process.join(timeout=2)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=2)


def test_sync_current_corpus_merges_partial_family_without_retiring_omissions(tmp_path):
    root = _store(tmp_path / "store")
    family_path = _family_path(root, "gemma4")
    family = yaml.safe_load(family_path.read_text())
    family["cases"]["second"] = _case("UNIFIED.1-2", "second", "second")
    family_path.write_text(unified_history.dump_yaml(family))
    loose = tmp_path / "loose"
    _write_loose_current(loose, "gemma4", [("UNIFIED.1-1", "text_only", "updated")])

    unified_history.sync_current_corpus(root, loose)

    stored = unified_history.load_store(root).families["gemma4"].cases
    assert stored["text_only"]["request"]["input"] == "updated"
    assert stored["second"]["lifecycle"] == "active"
    assert stored["second"]["display_id"] == "UNIFIED.1-2"


def test_sync_current_corpus_retires_omissions_only_for_complete_snapshot(tmp_path):
    root = _store(tmp_path / "store")
    family_path = _family_path(root, "gemma4")
    family = yaml.safe_load(family_path.read_text())
    family["cases"]["second"] = _case("UNIFIED.1-2", "second", "second")
    family_path.write_text(unified_history.dump_yaml(family))
    loose = tmp_path / "loose"
    _write_loose_current(loose, "gemma4", [("UNIFIED.1-1", "text_only", "updated")])
    _write_loose_current(loose, "qwen3", [("UNIFIED.1-1", "text_only", "hi")])

    unified_history.sync_current_corpus(root, loose, complete_snapshot=True)

    stored = unified_history.load_store(root).families["gemma4"].cases
    assert stored["second"]["lifecycle"] == "retired"
    assert stored["second"]["display_id"] is None
    assert stored["second"]["historical_ids"] == ["UNIFIED.1-2"]


def test_sync_current_corpus_restores_case_retired_by_complete_snapshot(tmp_path):
    root = _store(tmp_path / "store")
    family_path = _family_path(root, "gemma4")
    family = yaml.safe_load(family_path.read_text())
    family["cases"]["second"] = _case("UNIFIED.1-2", "second", "second")
    family_path.write_text(unified_history.dump_yaml(family))
    loose = tmp_path / "loose"
    _write_loose_current(loose, "gemma4", [("UNIFIED.1-1", "text_only", "updated")])
    _write_loose_current(loose, "qwen3", [("UNIFIED.1-1", "text_only", "hi")])

    unified_history.sync_current_corpus(root, loose, complete_snapshot=True)

    _write_loose_current(
        loose,
        "gemma4",
        [
            ("UNIFIED.1-1", "text_only", "updated"),
            ("UNIFIED.1-2", "second", "restored"),
        ],
    )
    unified_history.sync_current_corpus(root, loose, complete_snapshot=True)

    stored = unified_history.load_store(root).families["gemma4"].cases
    assert "second_2" not in stored
    assert stored["second"]["lifecycle"] == "active"
    assert stored["second"]["display_id"] == "UNIFIED.1-2"
    assert stored["second"]["historical_ids"] == []


@pytest.mark.parametrize(
    "restored_display_id, expected_aliases",
    [
        ("UNIFIED.1-1", []),
        ("UNIFIED.9-9", ["UNIFIED.1-1"]),
    ],
)
def test_sync_current_corpus_reactivates_retired_scenario(
    tmp_path,
    restored_display_id,
    expected_aliases,
):
    root = _store(tmp_path / "store")
    family_path = _family_path(root, "gemma4")
    family = yaml.safe_load(family_path.read_text())
    family["cases"]["text_only"].update(
        {
            "lifecycle": "retired",
            "display_id": None,
            "historical_ids": ["UNIFIED.1-1"],
        }
    )
    family_path.write_text(unified_history.dump_yaml(family))
    loose = tmp_path / "loose"
    _write_loose_current(loose, "gemma4", [(restored_display_id, "text_only", "restored")])

    unified_history.sync_current_corpus(root, loose)

    stored = unified_history.load_store(root).families["gemma4"].cases
    assert "text_only_2" not in stored
    assert stored["text_only"]["lifecycle"] == "active"
    assert stored["text_only"]["display_id"] == restored_display_id
    assert stored["text_only"]["historical_ids"] == expected_aliases


def test_complete_snapshot_rejects_omitted_family_without_mutation(tmp_path):
    root = _store(tmp_path / "store")
    loose = tmp_path / "loose"
    _write_loose_current(loose, "gemma4", [("UNIFIED.1-1", "text_only", "updated")])
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*.yaml")}

    with pytest.raises(ValueError, match="complete current snapshot is missing family: qwen3"):
        unified_history.sync_current_corpus(root, loose, complete_snapshot=True)

    assert {path.relative_to(root): path.read_bytes() for path in root.rglob("*.yaml")} == before


def test_sync_current_corpus_preserves_old_request_for_historical_capture(tmp_path):
    root = _store(tmp_path / "store")
    loose = tmp_path / "loose"
    _write_loose_current(loose, "gemma4", [("UNIFIED.1-1", "text_only", "updated")])

    changed = unified_history.sync_current_corpus(root, loose)

    store = unified_history.load_store(root)
    history = store.histories[("gemma4", "dynamo_v2")]
    stimulus = history.captures["dynamo_v2-0.1.0"]["changes"]["text_only"]["stimulus"]
    assert stimulus["inline"]["input"] == "hello"
    assert store.families["gemma4"].cases["text_only"]["request"]["input"] == "updated"
    assert set(changed) == {
        _family_path(root, "gemma4"),
        _capture_path(root, "gemma4", "dynamo_v2", "0.1.0"),
    }


def test_update_store_from_loose_serializes_concurrent_publishers(tmp_path, monkeypatch):
    root = _store(tmp_path / "store")
    loose_a = tmp_path / "loose_a"
    loose_b = tmp_path / "loose_b"
    _write_loose_current(loose_a, "gemma4", [("UNIFIED.1-2", "new_a", "a")])
    _write_loose_current(loose_b, "gemma4", [("UNIFIED.1-3", "new_b", "b")])

    sync_current_corpus = unified_history._sync_current_corpus
    first_derived = threading.Event()
    release_first = threading.Event()
    second_derived = threading.Event()

    def controlled_sync(store, loose_root, *, complete_snapshot):
        documents = sync_current_corpus(
            store,
            loose_root,
            complete_snapshot=complete_snapshot,
        )
        if Path(loose_root) == loose_a:
            first_derived.set()
            assert release_first.wait(timeout=5)
        else:
            second_derived.set()
        return documents

    monkeypatch.setattr(unified_history, "_sync_current_corpus", controlled_sync)
    errors = []

    def publish(loose_root):
        try:
            unified_history.update_store_from_loose(
                root,
                loose_root,
                complete_snapshot=False,
            )
        except Exception as error:
            errors.append(error)

    first = threading.Thread(target=publish, args=(loose_a,))
    second = threading.Thread(target=publish, args=(loose_b,))
    first.start()
    assert first_derived.wait(timeout=5)
    second.start()
    second_blocked_before_derivation = not second_derived.wait(timeout=0.2)
    release_first.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    assert second_blocked_before_derivation
    cases = unified_history.load_store(root).families["gemma4"].cases
    assert {"new_a", "new_b"} <= cases.keys()


@pytest.mark.parametrize(
    ("read_operation", "read_store"),
    [
        ("load_store", unified_history.load_store),
        ("store_inventory", unified_history.store_inventory),
    ],
)
def test_store_readers_wait_for_complete_publication(
    tmp_path,
    monkeypatch,
    read_operation,
    read_store,
):
    root = _store(tmp_path / "store")
    loose = tmp_path / "loose"
    _write_loose_current(loose, "gemma4", [("UNIFIED.1-1", "text_only", "updated")])

    replace = unified_history.os.replace
    first_replacement = threading.Event()
    release_writer = threading.Event()
    reader_finished = threading.Event()
    replacement_count = 0

    def pause_after_first_replacement(source, destination):
        nonlocal replacement_count
        replace(source, destination)
        replacement_count += 1
        if replacement_count == 1:
            first_replacement.set()
            assert release_writer.wait(timeout=5)

    monkeypatch.setattr(unified_history.os, "replace", pause_after_first_replacement)
    writer_errors = []
    reader_errors = []
    observed = []

    def publish():
        try:
            unified_history.sync_current_corpus(root, loose)
        except Exception as error:
            writer_errors.append(error)

    def read():
        try:
            observed.append(read_store(root))
        except Exception as error:
            reader_errors.append(error)
        finally:
            reader_finished.set()

    writer = threading.Thread(target=publish)
    reader_thread = threading.Thread(target=read)
    writer.start()
    assert first_replacement.wait(timeout=5)
    reader_thread.start()
    reader_blocked_during_publication = not reader_finished.wait(timeout=0.2)
    release_writer.set()
    writer.join(timeout=5)
    reader_thread.join(timeout=5)

    assert not writer.is_alive()
    assert not reader_thread.is_alive()
    assert writer_errors == []
    assert reader_errors == []
    assert reader_blocked_during_publication
    assert len(observed) == 1
    if read_operation == "load_store":
        store = observed[0]
        assert store.families["gemma4"].cases["text_only"]["request"]["input"] == "updated"
        stimulus = store.histories[("gemma4", "dynamo_v2")].captures["dynamo_v2-0.1.0"][
            "changes"
        ]["text_only"]["stimulus"]
        assert stimulus["inline"]["input"] == "hello"
    else:
        assert {
            "families/gemma4/inputs_and_golden.yaml",
            "families/gemma4/dynamo_v2-0.1.0.yaml",
        } <= {item["path"] for item in observed[0]}


def test_sync_current_corpus_validation_failure_leaves_store_unchanged(tmp_path, monkeypatch):
    root = _store(tmp_path / "store")
    loose = tmp_path / "loose"
    _write_loose_current(loose, "gemma4", [("UNIFIED.1-1", "text_only", "updated")])
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*.yaml")}
    load_store = unified_history._load_store_unlocked

    def reject_candidate(candidate):
        if Path(candidate) != root:
            raise ValueError("candidate rejected")
        return load_store(candidate)

    monkeypatch.setattr(unified_history, "_load_store_unlocked", reject_candidate)
    with pytest.raises(ValueError, match="candidate rejected"):
        unified_history.sync_current_corpus(root, loose)

    assert {path.relative_to(root): path.read_bytes() for path in root.rglob("*.yaml")} == before


def test_sync_current_corpus_replace_failure_rolls_back_store(tmp_path, monkeypatch):
    root = _store(tmp_path / "store")
    loose = tmp_path / "loose"
    _write_loose_current(loose, "gemma4", [("UNIFIED.1-1", "text_only", "updated")])
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*.yaml")}
    replace = unified_history.os.replace
    def fail_second_replace(source, destination):
        if Path(destination) == root and Path(source).parent.name.startswith("unified-history-"):
            raise OSError("injected second replacement failure")
        replace(source, destination)

    monkeypatch.setattr(unified_history.os, "replace", fail_second_replace)
    with pytest.raises(OSError, match="injected second replacement failure"):
        unified_history.sync_current_corpus(root, loose)

    assert {path.relative_to(root): path.read_bytes() for path in root.rglob("*.yaml")} == before
    unified_history.load_store(root)


def test_sync_current_corpus_interrupt_rolls_back_store(tmp_path, monkeypatch):
    root = _store(tmp_path / "store")
    loose = tmp_path / "loose"
    _write_loose_current(loose, "gemma4", [("UNIFIED.1-1", "text_only", "updated")])
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*.yaml")}
    replace = unified_history.os.replace

    def interrupt_second_replace(source, destination):
        if Path(destination) == root and Path(source).parent.name.startswith("unified-history-"):
            raise KeyboardInterrupt
        replace(source, destination)

    monkeypatch.setattr(unified_history.os, "replace", interrupt_second_replace)
    with pytest.raises(KeyboardInterrupt):
        unified_history.sync_current_corpus(root, loose)

    assert {path.relative_to(root): path.read_bytes() for path in root.rglob("*.yaml")} == before
    unified_history.load_store(root)


def test_sync_current_corpus_post_replace_interrupt_rolls_back_store(tmp_path, monkeypatch):
    root = _store(tmp_path / "store")
    loose = tmp_path / "loose"
    _write_loose_current(loose, "gemma4", [("UNIFIED.1-1", "text_only", "updated")])
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*.yaml")}
    replace = unified_history.os.replace

    def interrupt_after_first_replace(source, destination):
        replace(source, destination)
        if Path(destination) == root and Path(source).parent.name.startswith("unified-history-"):
            raise KeyboardInterrupt

    monkeypatch.setattr(unified_history.os, "replace", interrupt_after_first_replace)
    with pytest.raises(KeyboardInterrupt):
        unified_history.sync_current_corpus(root, loose)

    assert {path.relative_to(root): path.read_bytes() for path in root.rglob("*.yaml")} == before
    unified_history.load_store(root)


def test_sync_current_corpus_retains_backup_when_rollback_fails(tmp_path, monkeypatch):
    root = _store(tmp_path / "store")
    loose = tmp_path / "loose"
    _write_loose_current(loose, "gemma4", [("UNIFIED.1-1", "text_only", "updated")])
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*.yaml")}
    replace = unified_history.os.replace
    published = False

    def fail_publish_and_rollback(source, destination):
        nonlocal published
        source = Path(source)
        destination = Path(destination)
        if destination == root and source.parent.name.startswith("unified-history-"):
            replace(source, destination)
            published = True
            raise OSError("injected second replacement failure")
        if (
            published
            and destination == root
            and source.parent.name.startswith(".conformance-transaction-backup-")
        ):
            raise OSError("injected rollback failure")
        replace(source, destination)

    monkeypatch.setattr(unified_history.os, "replace", fail_publish_and_rollback)
    with pytest.raises(OSError, match="injected second replacement failure") as error:
        unified_history.sync_current_corpus(root, loose)

    journal = tmp_path / unified_history.TRANSACTION_JOURNAL
    backup_roots = list(tmp_path.glob(".conformance-transaction-backup-*"))
    assert journal.is_file()
    assert len(backup_roots) == 1
    assert any(str(journal) in note for note in error.value.__notes__)

    monkeypatch.setattr(unified_history.os, "replace", replace)
    unified_history.load_store(root)
    assert {path.relative_to(root): path.read_bytes() for path in root.rglob("*.yaml")} == before
    assert not journal.exists()
    assert not list(tmp_path.glob(".conformance-transaction-backup-*"))


def test_generation_publication_fsyncs_payloads_before_prepared_journal(
    tmp_path,
    monkeypatch,
):
    staged = tmp_path / "staged"
    candidate_directory = staged / "history"
    candidate_nested = candidate_directory / "nested"
    candidate_nested.mkdir(parents=True)
    root_payload = candidate_directory / "root.yaml"
    nested_payload = candidate_nested / "payload.yaml"
    candidate_file = staged / "manifest.json"
    root_payload.write_text("root")
    nested_payload.write_text("nested")
    candidate_file.write_text("manifest")
    events = []
    write_journal = unified_history._write_transaction_journal

    def observe_file(path):
        events.append(("file", Path(path)))

    def observe_directory(path):
        events.append(("directory", Path(path)))

    def observe_journal(path, document):
        if document["phase"] == "prepared":
            assert {root_payload, nested_payload, candidate_file} <= {
                path for kind, path in events if kind == "file"
            }
            assert events.index(("directory", candidate_nested)) < events.index(
                ("directory", candidate_directory)
            )
            events.append(("prepared", Path(path)))
        write_journal(path, document)

    monkeypatch.setattr(unified_history, "_fsync_regular_file", observe_file)
    monkeypatch.setattr(unified_history, "_fsync_directory", observe_directory)
    monkeypatch.setattr(unified_history, "_write_transaction_journal", observe_journal)

    unified_history.publish_paths_transactionally(
        [
            (candidate_directory, tmp_path / "history"),
            (candidate_file, tmp_path / "manifest.json"),
        ],
        backup_parent=tmp_path,
    )

    first_prepared = next(index for index, event in enumerate(events) if event[0] == "prepared")
    assert all(
        next(index for index, event in enumerate(events) if event == ("file", payload))
        < first_prepared
        for payload in (root_payload, nested_payload, candidate_file)
    )


def test_generation_publication_cleans_backup_if_initial_journal_write_fails(
    tmp_path,
    monkeypatch,
):
    candidate = tmp_path / "staged/manifest.json"
    destination = tmp_path / "manifest.json"
    candidate.parent.mkdir()
    candidate.write_text("new")
    destination.write_text("old")

    def reject_initial_journal(_path, document):
        assert document["phase"] == "prepared"
        raise OSError("injected initial journal failure")

    monkeypatch.setattr(
        unified_history,
        "_write_transaction_journal",
        reject_initial_journal,
    )
    with pytest.raises(OSError, match="injected initial journal failure"):
        unified_history.publish_paths_transactionally(
            [(candidate, destination)],
            backup_parent=tmp_path,
        )

    assert candidate.read_text() == "new"
    assert destination.read_text() == "old"
    assert not (tmp_path / unified_history.TRANSACTION_JOURNAL).exists()
    assert not list(tmp_path.glob(".conformance-transaction-backup-*"))


@pytest.mark.parametrize("failure_type", [OSError, KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("failure_target", ["history", "archives", "manifest"])
def test_generation_publication_rolls_back_every_post_effect_failure(
    tmp_path,
    monkeypatch,
    failure_type,
    failure_target,
):
    live = tmp_path / "live"
    staged = tmp_path / "staged"
    live.mkdir()
    staged.mkdir()
    replacements = []
    for name in ("history", "archives"):
        destination = live / name
        candidate = staged / name
        destination.mkdir()
        candidate.mkdir()
        (destination / "generation").write_text("old")
        (candidate / "generation").write_text("new")
        replacements.append((candidate, destination))
    destination_manifest = live / "manifest"
    candidate_manifest = staged / "manifest"
    destination_manifest.write_text("old")
    candidate_manifest.write_text("new")
    replacements.append((candidate_manifest, destination_manifest))

    replace = unified_history.os.replace
    injected = False

    def fail_after_effect(source, destination):
        nonlocal injected
        replace(source, destination)
        if not injected and Path(destination) == live / failure_target:
            injected = True
            raise failure_type("injected post-effect failure")

    monkeypatch.setattr(unified_history.os, "replace", fail_after_effect)
    with pytest.raises(failure_type):
        unified_history.publish_paths_transactionally(
            replacements,
            backup_parent=tmp_path,
        )

    assert (live / "history/generation").read_text() == "old"
    assert (live / "archives/generation").read_text() == "old"
    assert (live / "manifest").read_text() == "old"
    assert not list(tmp_path.glob(".conformance-transaction-backup-*"))


def test_generation_publication_retains_backup_when_rollback_fails(tmp_path, monkeypatch):
    live = tmp_path / "live"
    staged = tmp_path / "staged"
    destination = live / "history"
    candidate = staged / "history"
    destination.mkdir(parents=True)
    candidate.mkdir(parents=True)
    (destination / "generation").write_text("old")
    (candidate / "generation").write_text("new")

    replace = unified_history.os.replace
    published = False

    def fail_publish_and_rollback(source, target):
        nonlocal published
        source = Path(source)
        target = Path(target)
        if source == candidate and target == destination:
            replace(source, target)
            published = True
            raise OSError("injected publication failure")
        if (
            published
            and target == destination
            and source.parent.name.startswith(".conformance-transaction-backup-")
        ):
            raise OSError("injected rollback failure")
        replace(source, target)

    monkeypatch.setattr(unified_history.os, "replace", fail_publish_and_rollback)
    with pytest.raises(OSError, match="injected publication failure") as error:
        unified_history.publish_paths_transactionally(
            [(candidate, destination)],
            backup_parent=tmp_path,
        )

    backup_roots = list(tmp_path.glob(".conformance-transaction-backup-*"))
    assert len(backup_roots) == 1
    backup = backup_roots[0] / "00-history"
    assert (backup / "generation").read_text() == "old"
    journal = tmp_path / unified_history.TRANSACTION_JOURNAL
    assert journal.is_file()
    assert any(str(journal) in note for note in error.value.__notes__)

    monkeypatch.setattr(unified_history.os, "replace", replace)
    unified_history._recover_publication(tmp_path)
    assert (destination / "generation").read_text() == "old"
    assert not candidate.exists()
    assert not journal.exists()
    assert not backup_roots[0].exists()


def test_generation_recovery_does_not_require_expired_staging_directory(tmp_path, monkeypatch):
    live = tmp_path / "live"
    destination_history = live / "history"
    destination_manifest = live / "manifest"
    destination_history.mkdir(parents=True)
    (destination_history / "generation").write_text("old")
    destination_manifest.write_text("old")
    replace = unified_history.os.replace
    published = False

    with tempfile.TemporaryDirectory(dir=tmp_path) as staging_directory:
        candidate_history = Path(staging_directory) / "history"
        candidate_manifest = Path(staging_directory) / "manifest"
        candidate_history.mkdir()
        (candidate_history / "generation").write_text("new")
        candidate_manifest.write_text("new")

        def fail_publish_and_rollback(source, target):
            nonlocal published
            source = Path(source)
            target = Path(target)
            if source == candidate_history and target == destination_history:
                replace(source, target)
                published = True
                raise OSError("injected publication failure")
            if (
                published
                and target == destination_history
                and source.parent.name.startswith(".conformance-transaction-backup-")
            ):
                raise OSError("injected rollback failure")
            replace(source, target)

        monkeypatch.setattr(unified_history.os, "replace", fail_publish_and_rollback)
        with pytest.raises(OSError, match="injected publication failure"):
            unified_history.publish_paths_transactionally(
                [
                    (candidate_history, destination_history),
                    (candidate_manifest, destination_manifest),
                ],
                backup_parent=tmp_path,
            )

    assert not Path(staging_directory).exists()
    monkeypatch.setattr(unified_history.os, "replace", replace)
    unified_history._recover_publication(tmp_path)

    assert (destination_history / "generation").read_text() == "old"
    assert destination_manifest.read_text() == "old"
    assert not (tmp_path / unified_history.TRANSACTION_JOURNAL).exists()
    assert not list(tmp_path.glob(".conformance-transaction-backup-*"))


def test_generation_reader_recovers_publication_interrupted_by_sigterm(tmp_path):
    staged = tmp_path / "staged"
    replacements = []
    for name in ("history", "archives"):
        destination = tmp_path / name
        candidate = staged / name
        destination.mkdir()
        candidate.mkdir(parents=True)
        (destination / "generation").write_text("old")
        (candidate / "generation").write_text("new")
        replacements.append((candidate, destination))
    destination_manifest = tmp_path / "manifest"
    candidate_manifest = staged / "manifest"
    destination_manifest.write_text("old")
    candidate_manifest.write_text("new")
    replacements.append((candidate_manifest, destination_manifest))

    child = os.fork()
    if child == 0:
        replace = unified_history.os.replace

        def terminate_after_history_publish(source, destination):
            replace(source, destination)
            if Path(source) == staged / "history" and Path(destination) == tmp_path / "history":
                os.kill(os.getpid(), signal.SIGTERM)

        unified_history.os.replace = terminate_after_history_publish
        unified_history.publish_paths_transactionally(replacements, backup_parent=tmp_path)
        os._exit(99)

    _pid, status = os.waitpid(child, 0)
    assert os.WIFSIGNALED(status)
    assert os.WTERMSIG(status) == signal.SIGTERM

    with unified_history._store_read_lock(tmp_path / "history"):
        pass

    assert (tmp_path / "history/generation").read_text() == "old"
    assert (tmp_path / "archives/generation").read_text() == "old"
    assert destination_manifest.read_text() == "old"
    assert not (tmp_path / unified_history.TRANSACTION_JOURNAL).exists()
    assert not list(tmp_path.glob(".conformance-transaction-backup-*"))


def test_generation_cleanup_failure_is_committed_and_recovered_later(tmp_path, monkeypatch):
    destination = tmp_path / "history"
    candidate = tmp_path / "staged/history"
    destination.mkdir()
    candidate.mkdir(parents=True)
    (destination / "generation").write_text("old")
    (candidate / "generation").write_text("new")
    rmtree = unified_history.shutil.rmtree

    def reject_backup_cleanup(path):
        if Path(path).name.startswith(".conformance-transaction-backup-"):
            raise OSError("injected cleanup failure")
        rmtree(path)

    monkeypatch.setattr(unified_history.shutil, "rmtree", reject_backup_cleanup)
    with pytest.warns(RuntimeWarning, match="publication committed; cleanup remains pending"):
        unified_history.publish_paths_transactionally(
            [(candidate, destination)],
            backup_parent=tmp_path,
        )

    journal = tmp_path / unified_history.TRANSACTION_JOURNAL
    assert (destination / "generation").read_text() == "new"
    assert journal.is_file()
    assert len(list(tmp_path.glob(".conformance-transaction-backup-*"))) == 1

    reader_entered = threading.Event()
    reader_finished = threading.Event()
    reader_errors = []

    def read_generation():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                with unified_history._store_read_lock(destination):
                    assert (destination / "generation").read_text() == "new"
                    reader_entered.set()
        except Exception as error:
            reader_errors.append(error)
        finally:
            reader_finished.set()

    reader = threading.Thread(target=read_generation)
    reader.start()
    try:
        entered_while_cleanup_fails = reader_entered.wait(timeout=1)
        assert reader_finished.wait(timeout=1)
        assert journal.is_file()
    finally:
        monkeypatch.setattr(unified_history.shutil, "rmtree", rmtree)
        reader.join(timeout=2)

    assert entered_while_cleanup_fails
    assert not reader.is_alive()
    assert not reader_errors
    assert journal.is_file()

    with unified_history._store_mutation_lock(destination):
        assert (destination / "generation").read_text() == "new"
    assert not journal.exists()
    assert not list(tmp_path.glob(".conformance-transaction-backup-*"))


def test_committed_journal_fsync_failure_retains_indeterminate_recovery_state(
    tmp_path,
    monkeypatch,
):
    destination = tmp_path / "history"
    candidate = tmp_path / "staged/history"
    destination.mkdir()
    candidate.mkdir(parents=True)
    (destination / "generation").write_text("old")
    (candidate / "generation").write_text("new")
    fsync_directory = unified_history._fsync_directory
    committed_journal_installed = False
    replace = unified_history.os.replace

    def observe_committed_journal(source, target):
        nonlocal committed_journal_installed
        replace(source, target)
        if Path(target) == tmp_path / unified_history.TRANSACTION_JOURNAL:
            document = unified_history._load_transaction_journal(Path(target))
            committed_journal_installed = document["phase"] == "committed"

    def fail_committed_journal_directory_sync(path):
        if committed_journal_installed and Path(path) == tmp_path:
            raise OSError("injected fsync failure after committed journal install")
        fsync_directory(path)

    monkeypatch.setattr(unified_history.os, "replace", observe_committed_journal)
    monkeypatch.setattr(unified_history, "_fsync_directory", fail_committed_journal_directory_sync)

    with pytest.raises(
        unified_history.PublicationCommitIndeterminate,
        match="commit is visible but not durably confirmed",
    ):
        unified_history.publish_paths_transactionally(
            [(candidate, destination)],
            backup_parent=tmp_path,
        )

    journal = tmp_path / unified_history.TRANSACTION_JOURNAL
    assert (destination / "generation").read_text() == "new"
    assert unified_history._load_transaction_journal(journal)["phase"] == "committed"
    assert len(list(tmp_path.glob(".conformance-transaction-backup-*"))) == 1

    monkeypatch.setattr(unified_history.os, "replace", replace)
    monkeypatch.setattr(unified_history, "_fsync_directory", fsync_directory)
    unified_history._recover_publication(tmp_path)
    assert (destination / "generation").read_text() == "new"
    assert not journal.exists()
    assert not list(tmp_path.glob(".conformance-transaction-backup-*"))


@pytest.mark.parametrize("flag", ["--import-archives", "--compare-import"])
def test_removed_archive_migration_flags_fail_at_argument_boundary(tmp_path, capsys, flag):
    with pytest.raises(SystemExit) as error:
        unified_history.main(["--validate", flag, "--store", str(tmp_path)])

    assert error.value.code == 2
    assert f"unrecognized arguments: {flag}" in capsys.readouterr().err
