# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import unified_history  # noqa: E402


def _request(text: str) -> dict:
    return {
        "input": text,
        "init": {},
        "finish_reason": "stop",
        "tools": [],
        "chunks": [{"delta_text": text}],
    }


def _case(display_id: str, text: str) -> dict:
    return {
        "lifecycle": "active",
        "scenario": "text_only",
        "description": text,
        "policy": ["test-policy"],
        "display_id": display_id,
        "historical_ids": [],
        "request": _request(text),
        "golden": {"assembled": [{"kind": "text", "text": text}]},
    }


def _write_family(root: Path, *, extra_cases: dict | None = None) -> Path:
    path = root / "families/gemma4/inputs_and_golden.yaml"
    path.parent.mkdir(parents=True)
    cases = {"text_only": _case("UNIFIED.1-1", "hello")}
    cases.update(extra_cases or {})
    path.write_text(
        unified_history.dump_yaml(
            {
                "schema_version": 2,
                "family": "gemma4",
                "input_document": {"family": "gemma4", "mode": "unified"},
                "golden_document": {"family": "gemma4", "mode": "unified"},
                "cases": cases,
            }
        )
    )
    return path


def _change(
    text: str = "hello",
    *,
    stimulus: dict | None = None,
    case_key: str = "UNIFIED.1-1",
) -> dict:
    return {
        "case_key": case_key,
        "stimulus": stimulus or {"ref": "current"},
        "observation": {
            "value": {
                "assembled": [{"kind": "text", "text": text}],
                "chunks": [],
            }
        },
    }


def _write_capture(
    root: Path,
    version: str,
    parent: str | None,
    changes: dict,
    *,
    completeness: str | None = None,
    implementation: str = "dynamo_v2",
    metadata_changes: dict | None = None,
    document_overrides: dict | None = None,
    provenance: dict | None = None,
) -> Path:
    capture_id = f"{implementation}-{version}"
    base_version, separator, patch = version.rpartition(".patch")
    captured_version = base_version if separator and patch.isdigit() else version
    path = root / "families/gemma4" / f"{capture_id}.yaml"
    path.write_text(
        unified_history.dump_yaml(
            {
                "schema_version": 2,
                "family": "gemma4",
                "implementation": implementation,
                "runtime_version": version,
                "parent": parent,
                "provenance": provenance
                or {
                    "status": "legacy",
                    "captured_with": {implementation: captured_version},
                },
                "completeness": completeness or ("snapshot" if parent is None else "delta"),
                "document": {"mode": "unified"},
                "import_lineage": [],
                "changes": changes,
                "metadata_changes": metadata_changes or {},
                "document_overrides": document_overrides or {},
            }
        )
    )
    return path


def _store(root: Path) -> Path:
    _write_family(root)
    _write_capture(root, "0.5.0", None, {"text_only": _change()})
    _write_capture(root, "0.5.1", "dynamo_v2-0.5.0", {})
    _write_capture(root, "0.5.2", "dynamo_v2-0.5.1", {})
    return root


def test_schema_v2_resolves_consecutive_unchanged_capture_files(tmp_path):
    store = unified_history.load_store(_store(tmp_path))
    history = store.histories[("gemma4", "dynamo_v2")]

    first = history.resolve("dynamo_v2-0.5.0")
    second = history.resolve("dynamo_v2-0.5.1")
    third = history.resolve("dynamo_v2-0.5.2")

    assert first["text_only"]["observation"] == second["text_only"]["observation"]
    assert second["text_only"]["observation"] == third["text_only"]["observation"]
    assert first["text_only"]["document"]["captured_with"] == {"dynamo_v2": "0.5.0"}
    assert second["text_only"]["document"]["captured_with"] == {"dynamo_v2": "0.5.1"}
    assert third["text_only"]["document"]["captured_with"] == {"dynamo_v2": "0.5.2"}
    assert history.capture_path("dynamo_v2-0.5.2").name == "dynamo_v2-0.5.2.yaml"


def test_schema_v2_resolves_add_tombstone_and_readd(tmp_path):
    _write_family(tmp_path)
    _write_capture(tmp_path, "0.5.0", None, {"text_only": _change("first")})
    _write_capture(
        tmp_path,
        "0.5.1",
        "dynamo_v2-0.5.0",
        {"text_only": {"absent": True}},
    )
    _write_capture(
        tmp_path,
        "0.5.2",
        "dynamo_v2-0.5.1",
        {"text_only": _change("readded")},
    )

    history = unified_history.load_store(tmp_path).histories[("gemma4", "dynamo_v2")]

    assert set(history.resolve("dynamo_v2-0.5.0")) == {"text_only"}
    assert history.resolve("dynamo_v2-0.5.1") == {}
    assert (
        history.resolve("dynamo_v2-0.5.2")["text_only"]["observation"]["value"]
        == {"assembled": [{"kind": "text", "text": "readded"}], "chunks": []}
    )


def test_schema_v2_retains_changed_stimulus_with_identical_output(tmp_path):
    root = _store(tmp_path)
    path = root / "families/gemma4/dynamo_v2-0.5.1.yaml"
    document = unified_history.load_yaml(path)
    changed_stimulus = {"inline": _request("different request")}
    document["changes"] = {
        "text_only": _change(stimulus=changed_stimulus),
    }
    path.write_text(unified_history.dump_yaml(document))

    history = unified_history.load_store(root).histories[("gemma4", "dynamo_v2")]
    resolved = history.resolve("dynamo_v2-0.5.2")["text_only"]

    assert history.captures["dynamo_v2-0.5.1"]["changes"]["text_only"][
        "stimulus"
    ] == changed_stimulus
    assert resolved["stimulus"] == changed_stimulus
    assert resolved["observation"] == _change()["observation"]


def test_schema_v2_metadata_only_delta_keeps_changes_empty_and_materializes_record_metadata(
    tmp_path,
):
    root = _store(tmp_path)
    path = root / "families/gemma4/dynamo_v2-0.5.1.yaml"
    document = unified_history.load_yaml(path)
    record_metadata = {"engine_note": "metadata-only", "attempt": 2}
    document["changes"] = {}
    document["metadata_changes"] = {"text_only": record_metadata}
    path.write_text(unified_history.dump_yaml(document))

    history = unified_history.load_store(root).histories[("gemma4", "dynamo_v2")]
    resolved = history.resolve("dynamo_v2-0.5.2")["text_only"]

    assert history.captures["dynamo_v2-0.5.1"]["changes"] == {}
    assert history.captures["dynamo_v2-0.5.1"]["metadata_changes"] == {
        "text_only": record_metadata
    }
    assert resolved["document"]["record_metadata"] == record_metadata


def test_schema_v2_mixed_provenance_overrides_round_trip(tmp_path):
    _write_family(
        tmp_path,
        extra_cases={"second": _case("UNIFIED.1-2", "world")},
    )
    text_provenance = {"label": "legacy-text", "archive": "text.tar"}
    second_captured_with = {"dynamo_v2": "0.5.0"}
    _write_capture(
        tmp_path,
        "0.5.0",
        None,
        {
            "text_only": _change(),
            "second": _change("world", case_key="UNIFIED.1-2"),
        },
        provenance={"status": "mixed", "record_count": 2},
        document_overrides={
            "text_only": {"capture_provenance": text_provenance},
            "second": {"captured_with": second_captured_with},
        },
    )

    store = unified_history.load_store(tmp_path)
    history = store.histories[("gemma4", "dynamo_v2")]

    def documents() -> dict:
        resolved = history.resolve("dynamo_v2-0.5.0")
        return {
            case_id: record["document"]
            for case_id, record in resolved.items()
        }

    assert documents() == {
        "text_only": {
            "mode": "unified",
            "capture_provenance": text_provenance,
        },
        "second": {
            "mode": "unified",
            "captured_with": second_captured_with,
        },
    }

    unified_history.rewrite_store(tmp_path)
    round_tripped = unified_history.load_store(tmp_path).histories[
        ("gemma4", "dynamo_v2")
    ]
    assert {
        case_id: record["document"]
        for case_id, record in round_tripped.resolve("dynamo_v2-0.5.0").items()
    } == documents()


def test_schema_v2_loads_source_qualified_and_patch_capture_filenames(tmp_path):
    source_version = "0.6.1+source." + "a" * 64
    patch_version = source_version + ".patch1"
    _write_family(tmp_path)
    _write_capture(tmp_path, source_version, None, {"text_only": _change("source")})
    _write_capture(
        tmp_path,
        patch_version,
        f"dynamo_v2-{source_version}",
        {"text_only": _change("patched")},
    )

    history = unified_history.load_store(tmp_path).histories[("gemma4", "dynamo_v2")]

    assert history.capture_path(f"dynamo_v2-{source_version}").name == (
        f"dynamo_v2-{source_version}.yaml"
    )
    assert history.capture_path(f"dynamo_v2-{patch_version}").name == (
        f"dynamo_v2-{patch_version}.yaml"
    )
    assert (
        history.resolve(f"dynamo_v2-{patch_version}")["text_only"]["observation"][
            "value"
        ]["assembled"][0]["text"]
        == "patched"
    )


def test_schema_v2_resolution_cache_is_invalidated_before_reuse(tmp_path):
    store = unified_history.load_store(_store(tmp_path))
    history = store.histories[("gemma4", "dynamo_v2")]

    first = history.resolve("dynamo_v2-0.5.2")
    assert first["text_only"]["observation"]["value"]["assembled"][0]["text"] == "hello"
    assert set(history._resolved_states) == {
        "dynamo_v2-0.5.0",
        "dynamo_v2-0.5.1",
        "dynamo_v2-0.5.2",
    }

    history.captures["dynamo_v2-0.5.1"]["changes"] = {"text_only": _change("updated")}
    history._invalidate_resolution_cache()

    second = history.resolve("dynamo_v2-0.5.2")
    assert second["text_only"]["observation"]["value"]["assembled"][0]["text"] == "updated"


def test_capture_document_metadata_preserves_legacy_provenance_order():
    record = {"label": "0.5.0"}
    base = {
        "document": {"mode": "unified"},
        "provenance": {"status": "captured", "record": record},
        "runtime_version": "0.5.0",
    }

    imported = unified_history._capture_document_metadata(
        {**base, "import_lineage": [{"archive": "capture.tar.gz"}]},
        "dynamo_v2",
    )
    current = unified_history._capture_document_metadata(
        {**base, "import_lineage": []},
        "dynamo_v2",
    )

    assert tuple(imported) == ("mode", "capture_provenance", "captured_with")
    assert tuple(current) == ("mode", "captured_with", "capture_provenance")


def test_schema_v2_inventory_hashes_only_family_bundle_yaml(tmp_path):
    root = _store(tmp_path)
    inventory = unified_history.store_inventory(root)

    assert [item["path"] for item in inventory] == [
        "families/gemma4/dynamo_v2-0.5.0.yaml",
        "families/gemma4/dynamo_v2-0.5.1.yaml",
        "families/gemma4/dynamo_v2-0.5.2.yaml",
        "families/gemma4/inputs_and_golden.yaml",
    ]


@pytest.mark.parametrize("field", ["input_document", "golden_document"])
def test_schema_v2_requires_input_and_golden_documents(tmp_path, field):
    root = _store(tmp_path)
    family_path = root / "families/gemma4/inputs_and_golden.yaml"
    document = unified_history.load_yaml(family_path)
    del document[field]
    family_path.write_text(unified_history.dump_yaml(document))

    with pytest.raises(ValueError, match=rf"missing required fields.*{field}"):
        unified_history.load_store(root)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda root: (root / "capture_history").mkdir(), "legacy Unified capture_history"),
        (
            lambda root: (root / "families/gemma4.yaml").write_text("schema_version: 1\n"),
            "legacy Unified family layout",
        ),
        (
            lambda root: (root / "families/gemma4/index.yaml").write_text("schema_version: 2\n"),
            "unknown Unified family file",
        ),
        (
            lambda root: (root / "families/gemma4/dynamo_v2-0.5.1.yaml").unlink(),
            "missing parent",
        ),
    ],
)
def test_schema_v2_rejects_legacy_or_incomplete_layout(tmp_path, mutation, message):
    root = _store(tmp_path)
    mutation(root)

    with pytest.raises(ValueError, match=message):
        unified_history.load_store(root)


def test_schema_v2_rejects_filename_identity_mismatch(tmp_path):
    root = _store(tmp_path)
    source = root / "families/gemma4/dynamo_v2-0.5.2.yaml"
    source.rename(root / "families/gemma4/dynamo_v2-0.5.3.yaml")

    with pytest.raises(ValueError, match="capture identity differs"):
        unified_history.load_store(root)


def test_schema_v2_rejects_unhashable_capture_family(tmp_path):
    root = _store(tmp_path)
    path = root / "families/gemma4/dynamo_v2-0.5.0.yaml"
    document = unified_history.load_yaml(path)
    document["family"] = []
    path.write_text(unified_history.dump_yaml(document))

    with pytest.raises(ValueError, match="capture identity differs"):
        unified_history.load_store(root)


@pytest.mark.parametrize(
    ("provenance", "message"),
    [
        ({"status": "unknown", "record_count": 0}, "invalid capture provenance status"),
        ({"status": "mixed", "record_count": "0"}, "record_count must be"),
        (
            {
                "status": "captured",
                "record": {"label": "0.5.0"},
                "captured_with": {"dynamo_v2": "0.5.0"},
            },
            "exactly one of record or captured_with",
        ),
    ],
)
def test_schema_v2_rejects_invalid_capture_provenance_shape(
    tmp_path, provenance, message
):
    root = _store(tmp_path)
    path = root / "families/gemma4/dynamo_v2-0.5.0.yaml"
    document = unified_history.load_yaml(path)
    document["provenance"] = provenance
    path.write_text(unified_history.dump_yaml(document))

    with pytest.raises(ValueError, match=message):
        unified_history.load_store(root)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        (
            {"capture_provenance": "not-a-mapping"},
            "must be a mapping|capture provenance differs",
        ),
        (
            {"captured_with": ["dynamo_v2", "0.5.0"]},
            "must be a mapping|capture provenance differs",
        ),
        ({"family": "gemma4"}, "reserved metadata"),
    ],
)
def test_schema_v2_rejects_invalid_document_overrides(tmp_path, override, message):
    root = _store(tmp_path)
    path = root / "families/gemma4/dynamo_v2-0.5.0.yaml"
    document = unified_history.load_yaml(path)
    document["document_overrides"] = {"text_only": override}
    path.write_text(unified_history.dump_yaml(document))

    with pytest.raises(ValueError, match=message):
        unified_history.load_store(root)


def test_schema_v2_rejects_non_string_document_override_keys(tmp_path):
    root = _store(tmp_path)
    path = root / "families/gemma4/dynamo_v2-0.5.0.yaml"
    document = unified_history.load_yaml(path)
    document["document_overrides"] = {1: {"parser_path": "legacy"}}
    path.write_text(unified_history.dump_yaml(document))

    with pytest.raises(ValueError, match="document override case ID must be a string"):
        unified_history.load_store(root)


def test_schema_v2_rejects_unknown_root_entries(tmp_path):
    root = _store(tmp_path)
    (root / "unexpected.yaml").write_text("schema_version: 2\n")

    with pytest.raises(ValueError, match="unknown Unified root entry"):
        unified_history.load_store(root)


def test_schema_v2_rejects_multiple_graph_roots(tmp_path):
    root = _store(tmp_path)
    _write_capture(root, "0.6.0", None, {"text_only": _change("other")})

    with pytest.raises(ValueError, match="one graph root"):
        unified_history.load_store(root)


def test_schema_v2_rejects_multiple_graph_leaves(tmp_path):
    root = _store(tmp_path)
    source = root / "families/gemma4/dynamo_v2-0.5.2.yaml"
    document = unified_history.load_yaml(source)
    document["parent"] = "dynamo_v2-0.5.0"
    source.write_text(unified_history.dump_yaml(document))

    with pytest.raises(ValueError, match="one graph leaf"):
        unified_history.load_store(root)


def test_schema_v2_rejects_cross_implementation_parent(tmp_path):
    root = _store(tmp_path)
    _write_capture(
        root,
        "0.4.0",
        None,
        {"text_only": _change("peer")},
        implementation="vllm_python",
    )
    source = root / "families/gemma4/dynamo_v2-0.5.2.yaml"
    document = unified_history.load_yaml(source)
    document["parent"] = "vllm_python-0.4.0"
    source.write_text(unified_history.dump_yaml(document))

    with pytest.raises(
        ValueError,
        match=r"missing parent.*vllm_python-0\.4\.0",
    ):
        unified_history.load_store(root)


def test_schema_v2_rewrite_is_byte_deterministic(tmp_path):
    root = _store(tmp_path)
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*.yaml")}

    unified_history.rewrite_store(root)
    unified_history.rewrite_store(root)

    assert {path.relative_to(root): path.read_bytes() for path in root.rglob("*.yaml")} == before
