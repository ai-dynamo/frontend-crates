# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import hashlib
import io
import shutil
import sys
import tarfile
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import fixture_corpus  # noqa: E402
import fixture_disposition  # noqa: E402
import legacy_history  # noqa: E402
import package_fixtures  # noqa: E402
import resolve_fixtures  # noqa: E402
import resolve_reasoning_fixtures  # noqa: E402
import resolve_stream_fixtures  # noqa: E402


def _archive(fixtures: Path, relative: str, documents: dict[str, dict]) -> dict:
    path = fixtures / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as archive:
        for name, document in documents.items():
            data = yaml.safe_dump(document, sort_keys=False).encode()
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    return {"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size": path.stat().st_size}


def _store_bytes(store: Path) -> dict[str, bytes]:
    return {path.relative_to(store).as_posix(): path.read_bytes() for path in store.rglob("*.yaml")}


def _loose_capture(loose: Path, corpus: str, capture: str, value: str) -> Path:
    prefix = legacy_history.CORPORA[corpus]
    path = loose / prefix / capture / "qwen3" / "one.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    if corpus == "stream":
        case = {"chunks": [{"expected": [{"name": value}]}]}
    else:
        case = {"expected": {"vllm_python": {"tool_calls": [{"name": value}]}}}
    path.write_text(yaml.safe_dump({"family": "qwen3", "cases": {"A": case}}))
    return path


@pytest.mark.parametrize("corpus", ["batch", "reasoning"])
@pytest.mark.parametrize("annotation", [{"note": "corrected"}, {"provenance": None}, {"unavailable": "not supported"}])
def test_unchanged_expectations_retain_capture_annotations(tmp_path: Path, corpus: str, annotation: dict) -> None:
    loose, store = tmp_path / "loose", tmp_path / "store"
    _loose_capture(loose, corpus, "vllm_python-1.0.0", "same")
    later = _loose_capture(loose, corpus, "vllm_python-1.1.0", "same")
    doc = yaml.safe_load(later.read_text())
    doc["cases"]["A"].update(annotation)
    later.write_text(yaml.safe_dump(doc))
    legacy_history.update_from_loose(store, loose)
    output = tmp_path / "output"
    legacy_history.materialize_store(store, output)
    assert yaml.safe_load((output / later.relative_to(loose)).read_text()) == doc


@pytest.mark.parametrize("corpus", ["batch", "reasoning"])
@pytest.mark.parametrize("annotation", [{"note": "corrected"}, {"provenance": None}, {"unavailable": "not supported"}])
def test_unchanged_expectations_retain_annotation_removal(tmp_path: Path, corpus: str, annotation: dict) -> None:
    loose, store = tmp_path / "loose", tmp_path / "store"
    earlier = _loose_capture(loose, corpus, "vllm_python-1.0.0", "same")
    doc = yaml.safe_load(earlier.read_text())
    doc["cases"]["A"].update(annotation)
    earlier.write_text(yaml.safe_dump(doc))
    later = _loose_capture(loose, corpus, "vllm_python-1.1.0", "same")
    expected = yaml.safe_load(later.read_text())

    legacy_history.update_from_loose(store, loose)
    output = tmp_path / "output"
    legacy_history.materialize_store(store, output)

    assert yaml.safe_load((output / later.relative_to(loose)).read_text()) == expected
    assert legacy_history.update_from_loose(store, loose) == []


def test_duplicate_capture_identity_is_rejected_before_materialization(tmp_path: Path, monkeypatch) -> None:
    store = tmp_path / "fixtures-v1"
    family = store / "batch/families/qwen3"
    family.mkdir(parents=True)
    for version, value in (("0.1.0", "old"), ("0.2.0", "new")):
        record = {
            "corpus": "batch", "family": "qwen3", "capture": "vllm_python-0.1.0",
            "documents": {"one.yaml": {"family": "qwen3", "cases": {"A": {
                "expected": {"vllm_python": {"normal_text": value}},
            }}}},
        }
        (family / f"vllm_python-{version}.yaml").write_text(yaml.safe_dump(record))
    before = _store_bytes(store)
    destination = tmp_path / "materialized"

    with pytest.raises(ValueError, match="legacy history identity mismatch"):
        legacy_history.materialize_store(store, destination)
    assert not list(destination.rglob("*.yaml"))
    assert _store_bytes(store) == before

    digest, size = legacy_history.store_digest(store)
    manifest = {"shards": [{
        "path": legacy_history.HISTORY_PATH, "format": legacy_history.HISTORY_FORMAT,
        "sha256": digest, "size": size,
    }]}
    monkeypatch.setattr(package_fixtures.unified_history, "load_store", lambda _root: {})
    with pytest.raises(ValueError, match="legacy history identity mismatch"):
        package_fixtures._validate_candidate_package(manifest, tmp_path / "fixtures", tmp_path, store)


@pytest.mark.parametrize("change", ["corpus", "family", "capture", "document_family", "bad_version"])
def test_store_record_identity_is_checked_before_update(tmp_path: Path, change: str) -> None:
    store = tmp_path / "fixtures-v1"
    path = store / "batch/families/qwen3/vllm_python-0.1.0.yaml"
    path.parent.mkdir(parents=True)
    record = {
        "corpus": "batch", "family": "qwen3", "capture": "vllm_python-0.1.0",
        "documents": {"one.yaml": {"family": "qwen3", "cases": {}}},
    }
    if change == "document_family":
        record["documents"]["one.yaml"]["family"] = "other"
    elif change == "bad_version":
        record["capture"] = "vllm_python-0bogus"
        path = path.with_name("vllm_python-0bogus.yaml")
    else:
        record[change] = "other"
    path.write_text(yaml.safe_dump(record))
    before = path.read_bytes()

    with pytest.raises(ValueError, match="legacy history (identity mismatch|document)"):
        legacy_history.update_from_loose(store, tmp_path / "loose")
    assert path.read_bytes() == before


@pytest.mark.parametrize("corpus", ["batch", "reasoning"])
def test_versioned_capture_rejects_foreign_expected_results(tmp_path: Path, corpus: str) -> None:
    loose = tmp_path / "loose"
    corpus_root = loose / legacy_history.CORPORA[corpus]
    shared = corpus_root / "inputs/qwen3/one.yaml"
    shared.parent.mkdir(parents=True)
    shared.write_text(yaml.safe_dump({"family": "qwen3", "cases": {"A": {"model_text": "input"}}}))
    for version, foreign in (("0.1.0", "old"), ("0.2.0", "new")):
        capture = f"vllm_python-{version}"
        document = {"family": "qwen3", "cases": {"A": {"expected": {
            "vllm_python": {"normal_text": "same"},
            "sglang_python": {"normal_text": foreign},
        }}}}
        source = corpus_root / capture / "qwen3/one.yaml"
        source.parent.mkdir(parents=True)
        source.write_text(yaml.safe_dump(document))
    if corpus == "batch":
        resolved = resolve_fixtures.resolve_docs(corpus_root, ["vllm_python-0.2.0"])[0]
    else:
        resolved_root = tmp_path / "resolved"
        resolve_reasoning_fixtures.resolve(corpus_root, resolved_root, ["vllm_python-0.2.0"])
        resolved = {("qwen3", "one.yaml"): yaml.safe_load((resolved_root / "qwen3/one.yaml").read_text())}
    assert resolved[("qwen3", "one.yaml")]["cases"]["A"]["expected"]["sglang_python"]["normal_text"] == "new"
    store = tmp_path / "fixtures-v1"

    with pytest.raises(ValueError, match="foreign expected results"):
        legacy_history.update_from_loose(store, loose)
    assert _store_bytes(store) == {}

    stored = store / corpus / "families/qwen3/vllm_python-0.2.0.yaml"
    stored.parent.mkdir(parents=True)
    stored.write_text(yaml.safe_dump({
        "corpus": corpus, "family": "qwen3", "capture": "vllm_python-0.2.0",
        "documents": {"one.yaml": document},
    }))
    destination = tmp_path / "materialized"
    with pytest.raises(ValueError, match="foreign expected results"):
        legacy_history.materialize_store(store, destination)
    assert not list(destination.rglob("*.yaml"))


@pytest.mark.parametrize("corpus", ["batch", "stream", "reasoning"])
def test_release_overlay_follows_its_prerelease(corpus: str, tmp_path: Path) -> None:
    for version_key in (
        fixture_corpus.version_key,
        resolve_fixtures.version_key,
        resolve_stream_fixtures.version_key,
        resolve_reasoning_fixtures.version_key,
    ):
        assert version_key is fixture_disposition.version_sort_key
        assert version_key("1.0.0-rc1") < version_key("1.0.0")
        assert version_key("1.0.0-rc1") < version_key("1.0.0-rc1+build-branch")
        assert version_key("1.0.0-rc1+build-branch") < version_key("1.0.0")
        assert version_key("1.0.0") < version_key("1.0.0+current-branch")

    root = tmp_path / legacy_history.CORPORA[corpus]
    inputs = root / "inputs/qwen3/one.yaml"
    inputs.parent.mkdir(parents=True)
    if corpus == "stream":
        base_case = {"chunks": [{"delta_text": "x"}]}
    else:
        base_case = {"model_text": "x"}
    input_cases = {"A": base_case, "B": copy.deepcopy(base_case)}
    inputs.write_text(yaml.safe_dump({"family": "qwen3", "cases": input_cases}))

    for version, value in (
        ("1.0.0", "release"),
        ("1.0.0-rc1", "prerelease"),
        ("1.0.0+current-branch", "qualified"),
    ):
        capture = root / f"vllm_python-{version}/qwen3/one.yaml"
        capture.parent.mkdir(parents=True)
        if corpus == "stream":
            overlay_case = {"chunks": [{"expected": [{"name": value}]}]}
        else:
            overlay_case = {"expected": {"vllm_python": {"normal_text": value}}}
        case_id = "B" if version == "1.0.0+current-branch" else "A"
        capture_doc = {"family": "qwen3", "cases": {case_id: overlay_case}}
        capture.write_text(yaml.safe_dump(capture_doc))

    for target, expected in (
        ("1.0.0", "release"),
        ("1.0.0-rc1", "prerelease"),
        ("1.0.0+current-branch", "release"),
    ):
        selection = [f"vllm_python-{target}"]
        if corpus == "batch":
            docs = resolve_fixtures.resolve_docs(root, selection)[0]
        elif corpus == "stream":
            docs = resolve_stream_fixtures.resolve_docs(root, selection)[0]
        else:
            output = tmp_path / f"resolved-{target}"
            resolve_reasoning_fixtures.resolve(root, output, selection)
            docs = {("qwen3", "one.yaml"): yaml.safe_load((output / "qwen3/one.yaml").read_text())}

        case = docs[("qwen3", "one.yaml")]["cases"]["A"]
        if corpus == "stream":
            assert case["chunks"][0]["expected"]["vllm_python"] == [{"name": expected}]
        else:
            assert case["expected"]["vllm_python"]["normal_text"] == expected
        if target == "1.0.0+current-branch":
            case_b = docs[("qwen3", "one.yaml")]["cases"]["B"]
            if corpus == "stream":
                assert case_b["chunks"][0]["expected"]["vllm_python"] == [{"name": "qualified"}]
            else:
                assert case_b["expected"]["vllm_python"]["normal_text"] == "qualified"


def test_import_preserves_sparse_layers_chunks_and_mixed_snapshot_labels(tmp_path: Path) -> None:
    fixtures = tmp_path / "fixtures"
    prefix = "toolcalling/fixtures-stream-v1"
    filename = "TOOLCALLING.streamv1.1.yaml"
    shared = {"family": "qwen3", "mode": "streamv1", "cases": {"A": {"chunks": [{"delta_text": "x"}, {"delta_text": "y"}]}}}
    anchor = {"family": "qwen3", "mode": "streamv1", "captured_with": {"dynamo_v2": "0.1.11"}, "cases": {"A": {"chunks": [{"expected": []}, {"expected": [{"index": 0, "arguments": "{}"}]}]}}}
    patch = {"family": "qwen3", "mode": "streamv1", "captured_with": {"dynamo_v2": "0.1.11.patch1"}, "cases": {"A": {"unavailable": {"dynamo_v2": "no parser"}}}}
    snapshot = {
        "family": "qwen3", "mode": "batch-on-stream",
        "captured_with": {"dynamo_v2": "Dynamo parser v2", "vllm_python": "0.26.0"},
        "cases": {"A": {"vllm_python": {"error": "parse failed"}, "dynamo_v2": {"tool_calls": []}}},
    }
    shards = [
        _archive(fixtures, f"{prefix}/inputs.tar.gz", {f"{prefix}/inputs/qwen3/{filename}": shared}),
        _archive(fixtures, f"{prefix}/dynamo_v2-0.1.11.tar.gz", {f"{prefix}/dynamo_v2-0.1.11/qwen3/{filename}": anchor}),
        _archive(fixtures, f"{prefix}/dynamo_v2-0.1.11.patch1.tar.gz", {f"{prefix}/dynamo_v2-0.1.11.patch1/qwen3/{filename}": patch}),
        _archive(fixtures, "toolcalling/fixtures-batch-on-stream-v1.tar.gz", {"toolcalling/fixtures-batch-on-stream-v1/qwen3/TOOLCALLING.batch.1.yaml": snapshot}),
    ]
    store = tmp_path / "fixtures-v1"
    legacy_history.import_archives(fixtures, shards, store)
    extracted = tmp_path / "extracted"
    legacy_history.materialize_store(store, extracted)

    assert yaml.safe_load((extracted / prefix / "inputs/qwen3" / filename).read_text()) == shared
    assert yaml.safe_load((extracted / prefix / "dynamo_v2-0.1.11.patch1/qwen3" / filename).read_text()) == patch
    assert yaml.safe_load((extracted / "toolcalling/fixtures-batch-on-stream-v1/qwen3/TOOLCALLING.batch.1.yaml").read_text()) == snapshot
    assert (store / "stream/families/qwen3/dynamo_v2-0.1.11.patch1.yaml").is_file()
    batch_on_stream = store / "batch_on_stream/families/qwen3/inputs_and_golden.yaml"
    assert batch_on_stream.is_file()
    assert yaml.safe_load(batch_on_stream.read_text())["documents"]["TOOLCALLING.batch.1.yaml"]["capture_order"] == list(snapshot["captured_with"])


def test_import_compacts_unchanged_cases_and_keeps_empty_capture_provenance(tmp_path: Path) -> None:
    fixtures = tmp_path / "fixtures"
    prefix = "toolcalling/fixtures-batch-v1"
    filename = "TOOLCALLING.batch.1.yaml"
    shared = {
        "family": "qwen3", "mode": "batch",
        "cases": {"A": {"model_text": "a"}, "B": {"model_text": "b"}},
    }

    def capture(case_a: str, case_b: str) -> dict:
        return {
            "family": "qwen3", "mode": "batch",
            "cases": {
                "A": {"expected": {"vllm_python": {"tool_calls": [{"name": case_a}]}}},
                "B": {"expected": {"vllm_python": {"tool_calls": [{"name": case_b}]}}},
            },
        }

    shards = [
        _archive(fixtures, f"{prefix}/inputs.tar.gz", {f"{prefix}/inputs/qwen3/{filename}": shared}),
        _archive(fixtures, f"{prefix}/vllm_python-0.24.0.tar.gz", {f"{prefix}/vllm_python-0.24.0/qwen3/{filename}": capture("a", "b")}),
        _archive(fixtures, f"{prefix}/vllm_python-0.25.1.tar.gz", {f"{prefix}/vllm_python-0.25.1/qwen3/{filename}": capture("a", "b2")}),
        _archive(fixtures, f"{prefix}/vllm_python-0.26.0.tar.gz", {f"{prefix}/vllm_python-0.26.0/qwen3/{filename}": capture("a", "b2")}),
    ]
    store = tmp_path / "fixtures-v1"
    legacy_history.import_archives(fixtures, shards, store)

    anchor = yaml.safe_load((store / "batch/families/qwen3/vllm_python-0.24.0.yaml").read_text())
    delta = yaml.safe_load((store / "batch/families/qwen3/vllm_python-0.25.1.yaml").read_text())
    unchanged = yaml.safe_load((store / "batch/families/qwen3/vllm_python-0.26.0.yaml").read_text())
    assert set(anchor["documents"][filename]["cases"]) == {"A", "B"}
    assert set(delta["documents"][filename]["cases"]) == {"B"}
    assert unchanged["documents"][filename]["cases"] == {}

    loose = tmp_path / "loose"
    legacy_history.materialize_store(store, loose)
    resolved = resolve_fixtures.resolve_docs(
        loose / prefix, ["vllm_python-0.26.0"]
    )[0][("qwen3", filename)]
    assert resolved["cases"]["A"]["expected"]["vllm_python"]["tool_calls"][0]["name"] == "a"
    assert resolved["cases"]["B"]["expected"]["vllm_python"]["tool_calls"][0]["name"] == "b2"
    assert legacy_history.update_from_loose(store, loose) == []


def test_import_rejects_out_of_order_capture_without_changing_later_resolution(tmp_path: Path) -> None:
    fixtures = tmp_path / "fixtures"
    prefix = "toolcalling/fixtures-batch-v1"
    filename = "TOOLCALLING.batch.1.yaml"
    family = "qwen3"
    shared = {"family": family, "mode": "batch", "cases": {"A": {"model_text": "a"}}}

    def archive(capture: str, value: str) -> dict:
        document = {
            "family": family,
            "mode": "batch",
            "cases": {"A": {"expected": {"vllm_python": {"tool_calls": [{"name": value}]}}}},
        }
        return _archive(
            fixtures,
            f"{prefix}/{capture}.tar.gz",
            {f"{prefix}/{capture}/{family}/{filename}": document},
        )

    inputs = _archive(fixtures, f"{prefix}/inputs.tar.gz", {f"{prefix}/inputs/{family}/{filename}": shared})
    store = tmp_path / "fixtures-v1"
    legacy_history.import_archives(fixtures, [inputs, archive("vllm_python-0.1.0", "A"), archive("vllm_python-0.3.0", "A")], store)
    before = _store_bytes(store)

    with pytest.raises(ValueError, match="cannot add out-of-order legacy capture"):
        legacy_history.import_archives(fixtures, [archive("vllm_python-0.2.0", "B")], store)
    assert _store_bytes(store) == before

    extracted = tmp_path / "extracted"
    legacy_history.materialize_store(store, extracted)
    resolved = resolve_fixtures.resolve_docs(
        extracted / prefix, ["vllm_python-0.3.0"]
    )[0][(family, filename)]
    assert resolved["cases"]["A"]["expected"]["vllm_python"]["tool_calls"] == [{"name": "A"}]


def test_partial_snapshot_update_keeps_unmentioned_documents(tmp_path: Path) -> None:
    store = tmp_path / "fixtures-v1"
    loose = tmp_path / "loose"
    family = loose / "toolcalling/fixtures-batch-on-stream-v1/qwen3"
    family.mkdir(parents=True)
    first = {"family": "qwen3", "mode": "batch-on-stream", "captured_with": {"vllm_python": "0.25.1"}, "cases": {"A": {"vllm_python": {"tool_calls": []}}}}
    second = {"family": "qwen3", "mode": "batch-on-stream", "captured_with": {"vllm_python": "0.26.0"}, "cases": {"B": {"vllm_python": {"tool_calls": [{"name": "f"}]}}}}
    (family / "one.yaml").write_text(yaml.safe_dump(first))
    (family / "two.yaml").write_text(yaml.safe_dump(second))
    legacy_history.update_from_loose(store, loose)
    snapshot_file = store / "batch_on_stream/families/qwen3/inputs_and_golden.yaml"
    assert snapshot_file.is_file()
    assert len(list(snapshot_file.parent.glob("*.yaml"))) == 3
    (family / "two.yaml").unlink()
    first["captured_with"]["vllm_python"] = "0.26.0"
    (family / "one.yaml").write_text(yaml.safe_dump(first))
    legacy_history.update_from_loose(store, loose)

    extracted = tmp_path / "extracted"
    legacy_history.materialize_store(store, extracted)
    root = extracted / "toolcalling/fixtures-batch-on-stream-v1/qwen3"
    assert yaml.safe_load((root / "one.yaml").read_text()) == first
    assert yaml.safe_load((root / "two.yaml").read_text()) == second


def test_split_snapshot_preserves_labels_and_order(tmp_path: Path) -> None:
    store = tmp_path / "fixtures-v1"
    family = store / "batch_on_stream/families/qwen3"
    family.mkdir(parents=True)
    filename = "TOOLCALLING.batch.1.yaml"
    document = {
        "family": "qwen3", "mode": "batch-on-stream",
        "captured_with": {"vllm_python": "0.26.0", "dynamo_v2": "Dynamo parser v2", "vllm_rust": "v0.23.0 " + "a" * 40},
        "cases": {"A": {"vllm_python": {"tool_calls": []}, "dynamo_v2": {"tool_calls": [{"name": "f"}]}, "vllm_rust": {"error": None}}},
    }
    old = family / "legacy-snapshot.yaml"
    old.write_text(yaml.safe_dump({"corpus": "batch_on_stream", "family": "qwen3", "capture": "legacy-snapshot", "documents": {filename: document}}, sort_keys=False))

    changed = legacy_history.split_batch_on_stream_store(store)

    assert len(changed) == 5
    assert sorted(path.name for path in family.glob("*.yaml")) == ["dynamo_v2-unversioned.yaml", "inputs_and_golden.yaml", "vllm_python-0.26.0.yaml", "vllm_rust-0.23.0.yaml"]
    rust = yaml.safe_load((family / "vllm_rust-0.23.0.yaml").read_text())
    assert rust["provenance"] == {"captured_with": "v0.23.0 " + "a" * 40, "runtime_version": "0.23.0", "git_commit": "a" * 40}
    unknown = yaml.safe_load((family / "dynamo_v2-unversioned.yaml").read_text())
    assert unknown["provenance"] == {"captured_with": "Dynamo parser v2", "runtime_version": None}
    extracted = tmp_path / "extracted"
    legacy_history.materialize_store(store, extracted)
    materialized = yaml.safe_load((extracted / "toolcalling/fixtures-batch-on-stream-v1/qwen3" / filename).read_text())
    assert materialized == document
    assert list(materialized["cases"]["A"]) == list(document["cases"]["A"])
    assert list(materialized["captured_with"]) == list(document["captured_with"])
    assert legacy_history.split_batch_on_stream_store(store) == []


@pytest.mark.parametrize("corruption", ["missing_capture", "wrong_origin", "duplicate_implementation", "changed_observation"])
def test_split_snapshot_rejects_corruption_without_publishing(tmp_path: Path, corruption: str) -> None:
    store, loose = tmp_path / "store", tmp_path / "loose"
    capture = loose / legacy_history.CORPORA["batch_on_stream"] / "qwen3/one.yaml"
    capture.parent.mkdir(parents=True)
    document = {"family": "qwen3", "captured_with": {"dynamo_v2": "Dynamo parser v2"},
                "cases": {"A": {"dynamo_v2": {"tool_calls": []}}}}
    capture.write_text(yaml.safe_dump(document))
    legacy_history.update_from_loose(store, loose)
    family = store / "batch_on_stream/families/qwen3"
    if corruption == "changed_observation":
        before = _store_bytes(store)
        document["cases"]["A"]["dynamo_v2"]["tool_calls"] = [{"name": "changed"}]
        capture.write_text(yaml.safe_dump(document))
        with pytest.raises(ValueError, match="immutable"):
            legacy_history.update_from_loose(store, loose)
        assert _store_bytes(store) == before
        return
    if corruption == "missing_capture":
        (family / "dynamo_v2-unversioned.yaml").unlink()
        message = "missing batch-on-stream capture"
    elif corruption == "wrong_origin":
        path = family / "dynamo_v2-unversioned.yaml"
        record = yaml.safe_load(path.read_text())
        record["provenance"]["runtime_version"] = "9.9.9"
        path.write_text(yaml.safe_dump(record))
        message = "capture origin mismatch"
    else:
        path = family / "inputs_and_golden.yaml"
        record = yaml.safe_load(path.read_text())
        record["documents"]["one.yaml"]["case_order"]["A"].append("dynamo_v2")
        path.write_text(yaml.safe_dump(record))
        message = "duplicate batch-on-stream implementation"
    output = tmp_path / "extracted"
    with pytest.raises(ValueError, match=message):
        legacy_history.materialize_store(store, output)
    assert not list(output.rglob("*.yaml"))


def test_versioned_capture_cannot_be_rewritten(tmp_path: Path) -> None:
    store = tmp_path / "fixtures-v1"
    loose = tmp_path / "loose"
    family = loose / "toolcalling/fixtures-batch-v1/vllm_python-0.26.0/qwen3"
    family.mkdir(parents=True)
    path = family / "TOOLCALLING.batch.1.yaml"
    path.write_text(yaml.safe_dump({"family": "qwen3", "mode": "batch", "cases": {"A": {"expected": {"vllm_python": {"tool_calls": []}}}}}))
    legacy_history.update_from_loose(store, loose)
    path.write_text(yaml.safe_dump({"family": "qwen3", "mode": "batch", "cases": {"A": {"expected": {"vllm_python": {"tool_calls": [{"name": "f"}]}}}}}))
    with pytest.raises(ValueError, match="immutable"):
        legacy_history.update_from_loose(store, loose)


@pytest.mark.parametrize("qualified", ["+current", ".patch1"])
@pytest.mark.parametrize("release_value", ["A", "B"])
def test_plain_stream_capture_survives_both_consumer_histories(
    tmp_path: Path, release_value: str, qualified: str,
) -> None:
    store = tmp_path / "fixtures-v1"
    loose = tmp_path / "loose"
    prefix = legacy_history.CORPORA["stream"]
    inputs = loose / prefix / "inputs/qwen3/one.yaml"
    inputs.parent.mkdir(parents=True, exist_ok=True)
    inputs.write_text(yaml.safe_dump({"family": "qwen3", "cases": {"A": {"chunks": [{"delta_text": "x"}]}}}))
    _loose_capture(loose, "stream", "vllm_python-0.1.0", "A")
    _loose_capture(loose, "stream", f"vllm_python-0.1.0{qualified}", "B")
    _loose_capture(loose, "stream", "vllm_python-0.2.0", release_value)

    legacy_history.update_from_loose(store, loose)

    release = yaml.safe_load((store / "stream/families/qwen3/vllm_python-0.2.0.yaml").read_text())
    assert release["documents"]["one.yaml"]["cases"]["A"]["chunks"][0]["expected"] == [{"name": release_value}]
    extracted = tmp_path / "extracted"
    legacy_history.materialize_store(store, extracted)
    root = extracted / prefix
    selected = ["vllm_python-0.2.0"]
    resolved = resolve_stream_fixtures.resolve_docs(root, selected)[0][("qwen3", "one.yaml")]
    assert resolved["cases"]["A"]["chunks"][0]["expected"]["vllm_python"] == [{"name": release_value}]
    shutil.rmtree(root / f"vllm_python-0.1.0{qualified}")
    release_only = resolve_stream_fixtures.resolve_docs(root, selected)[0][("qwen3", "one.yaml")]
    assert release_only["cases"]["A"]["chunks"][0]["expected"]["vllm_python"] == [{"name": release_value}]
    assert legacy_history.update_from_loose(store, loose) == []


def test_prune_can_replace_history_when_later_checkpoint_is_removed(tmp_path: Path) -> None:
    store = tmp_path / "fixtures-v1"
    loose = tmp_path / "loose"
    _loose_capture(loose, "batch", "vllm_python-0.1.0", "A")
    later = _loose_capture(loose, "batch", "vllm_python-0.3.0", "A")
    legacy_history.update_from_loose(store, loose)
    later.unlink()
    _loose_capture(loose, "batch", "vllm_python-0.2.0", "B")

    legacy_history.update_from_loose(store, loose, prune=True)

    family = store / "batch/families/qwen3"
    assert not (family / "vllm_python-0.3.0.yaml").exists()
    assert yaml.safe_load((family / "vllm_python-0.2.0.yaml").read_text())["documents"]["one.yaml"]["cases"]["A"]["expected"]["vllm_python"]["tool_calls"] == [{"name": "B"}]


def test_prune_rejects_removing_anchor_of_retained_sparse_checkpoint(tmp_path: Path) -> None:
    store = tmp_path / "fixtures-v1"
    loose = tmp_path / "loose"
    anchor = _loose_capture(loose, "batch", "vllm_python-0.1.0", "A")
    later = _loose_capture(loose, "batch", "vllm_python-0.2.0", "A")
    legacy_history.update_from_loose(store, loose)
    before = _store_bytes(store)
    anchor.unlink()
    later.write_text(yaml.safe_dump({"family": "qwen3", "cases": {}}))
    _loose_capture(loose, "batch", "vllm_python-0.3.0", "B")

    with pytest.raises(ValueError, match="prune changes resolved legacy capture output"):
        legacy_history.update_from_loose(store, loose, prune=True)
    assert _store_bytes(store) == before


def test_import_rejects_overwriting_existing_capture_without_writes(tmp_path: Path) -> None:
    fixtures = tmp_path / "fixtures"
    prefix = "toolcalling/fixtures-batch-v1"
    store = tmp_path / "fixtures-v1"

    def archive(value: str) -> dict:
        document = {"family": "qwen3", "cases": {"A": {"expected": {"vllm_python": {"tool_calls": [{"name": value}]}}}}}
        return _archive(fixtures, f"{prefix}/vllm_python-0.1.0.tar.gz", {f"{prefix}/vllm_python-0.1.0/qwen3/one.yaml": document})

    legacy_history.import_archives(fixtures, [archive("A")], store)
    before = _store_bytes(store)

    with pytest.raises(ValueError, match="immutable"):
        legacy_history.import_archives(fixtures, [archive("B")], store)
    assert _store_bytes(store) == before


def test_failed_update_preserves_changed_input_and_capture_bytes(tmp_path: Path) -> None:
    store = tmp_path / "fixtures-v1"
    loose = tmp_path / "loose"
    prefix = legacy_history.CORPORA["batch"]
    inputs = loose / prefix / "inputs/qwen3/one.yaml"
    inputs.parent.mkdir(parents=True, exist_ok=True)
    inputs.write_text(yaml.safe_dump({"family": "qwen3", "cases": {"A": {"model_text": "original"}}}))
    capture = _loose_capture(loose, "batch", "vllm_python-0.1.0", "A")
    legacy_history.update_from_loose(store, loose)
    before = _store_bytes(store)
    inputs.write_text(yaml.safe_dump({"family": "qwen3", "cases": {"A": {"model_text": "changed"}}}))
    capture.write_text(yaml.safe_dump({"family": "qwen3", "cases": {"A": {"expected": {"vllm_python": {"tool_calls": [{"name": "B"}]}}}}}))

    with pytest.raises(ValueError, match="immutable"):
        legacy_history.update_from_loose(store, loose)
    assert _store_bytes(store) == before


def test_prune_removes_unstaged_legacy_captures(tmp_path: Path) -> None:
    store = tmp_path / "fixtures-v1"
    loose = tmp_path / "loose"
    family = loose / "toolcalling/fixtures-batch-v1/inputs/qwen3"
    family.mkdir(parents=True)
    (family / "one.yaml").write_text(yaml.safe_dump({"family": "qwen3", "cases": {"A": {}}}))
    legacy_history.update_from_loose(store, loose)
    old_version = store / "batch/families/qwen3/vllm_python-0.24.0.yaml"
    old_version.write_text(yaml.safe_dump({"corpus": "batch", "family": "qwen3", "capture": "vllm_python-0.24.0", "documents": {"one.yaml": {"family": "qwen3"}}}))
    assert old_version.is_file()

    legacy_history.update_from_loose(store, loose, prune=True)

    assert not old_version.exists()
    assert (store / "batch/families/qwen3/inputs_and_golden.yaml").is_file()


@pytest.mark.parametrize("corpus", ["batch", "stream", "reasoning"])
@pytest.mark.parametrize("change", ["text", "tools", "document_tools", "chunks", "token_ids", "finish", "init", "reasoning", "template", "mode", "remove"])
def test_retained_capture_prevents_changing_its_request(tmp_path: Path, corpus: str, change: str) -> None:
    store, loose = tmp_path / "store", tmp_path / "loose"
    inputs = loose / legacy_history.CORPORA[corpus] / "inputs/qwen3/one.yaml"
    inputs.parent.mkdir(parents=True)
    case = {"model_text": "original", "chunks": [{"delta_text": "original"}]}
    document = {"family": "qwen3", "mode": corpus, "cases": {"A": case}}
    inputs.write_text(yaml.safe_dump(document))
    _loose_capture(loose, corpus, "vllm_python-0.1.0", "original")
    legacy_history.update_from_loose(store, loose)
    before = _store_bytes(store)
    if change == "text":
        case["model_text"] = "changed"
    elif change == "tools":
        case["tools"] = [{"name": "new_tool"}]
    elif change == "document_tools":
        document["tools"] = [{"name": "new_tool"}]
    elif change == "chunks":
        case["chunks"] = [{"delta_text": "orig"}, {"delta_text": "inal"}]
    elif change == "token_ids":
        case["chunks"][0]["delta_token_ids"] = [42]
    elif change == "finish":
        case["chunks"][0]["finish_reason"] = "length"
    elif change == "init":
        case["init"] = {"starting_state": "Reasoning"}
    elif change == "reasoning":
        case["force_reasoning"] = True
    elif change == "template":
        document["chat_template_kwargs"] = {"enable_thinking": True}
    elif change == "mode":
        document["mode"] = "other"
    else:
        document["cases"].pop("A")
    inputs.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match="captured legacy input is immutable"):
        legacy_history.update_from_loose(store, loose)
    assert _store_bytes(store) == before


@pytest.mark.parametrize("corpus", ["batch", "stream", "reasoning"])
def test_captured_inputs_allow_metadata_and_new_cases(tmp_path: Path, corpus: str) -> None:
    store, loose = tmp_path / "store", tmp_path / "loose"
    inputs = loose / legacy_history.CORPORA[corpus] / "inputs/qwen3/one.yaml"
    inputs.parent.mkdir(parents=True)
    document = {"family": "qwen3", "cases": {"A": {"model_text": "same", "tools": []}}}
    inputs.write_text(yaml.safe_dump(document))
    _loose_capture(loose, corpus, "vllm_python-0.1.0", "same")
    legacy_history.update_from_loose(store, loose)
    capture = store / corpus / "families/qwen3/vllm_python-0.1.0.yaml"
    before = capture.read_bytes()
    document["cases"]["A"]["description"] = "Clearer description"
    document["cases"]["B"] = {"model_text": "new"}
    document["refs"] = ["updated source"]
    inputs.write_text(yaml.safe_dump(document))
    legacy_history.update_from_loose(store, loose)
    assert capture.read_bytes() == before
    assert legacy_history.update_from_loose(store, loose) == []


@pytest.mark.parametrize("owner", ["reasoning_input", "batch_on_stream"])
def test_input_guard_includes_observations_outside_versioned_corpus(tmp_path: Path, owner: str) -> None:
    corpus = "reasoning" if owner == "reasoning_input" else "batch"
    store, loose = tmp_path / "store", tmp_path / "loose"
    inputs = loose / legacy_history.CORPORA[corpus] / "inputs/qwen3/one.yaml"
    inputs.parent.mkdir(parents=True)
    case = {"model_text": "original"}
    if owner == "reasoning_input":
        case["expected"] = {"dynamo_v1": {"reasoning": "original"}}
    else:
        snapshot = loose / legacy_history.CORPORA["batch_on_stream"] / "qwen3/one.yaml"
        snapshot.parent.mkdir(parents=True)
        snapshot.write_text(yaml.safe_dump({"family": "qwen3", "captured_with": {"dynamo_v2": "0.7.0"},
                                            "cases": {"A": {"dynamo_v2": {"calls": []}}}}))
    document = {"family": "qwen3", "cases": {"A": case}}
    inputs.write_text(yaml.safe_dump(document))
    legacy_history.update_from_loose(store, loose)
    before = _store_bytes(store)
    case["model_text"] = "changed"
    inputs.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match="captured legacy input is immutable"):
        legacy_history.update_from_loose(store, loose)
    assert _store_bytes(store) == before


@pytest.mark.parametrize("corpus", ["batch", "stream", "reasoning"])
@pytest.mark.parametrize("field,value", [("tools", [{"name": "get_weather"}]), ("model_text", "hello"),
                                         ("chat_template_kwargs", {"enable_thinking": True}), ("force_reasoning", True)])
def test_document_field_cannot_replace_captured_case_field(tmp_path: Path, corpus: str, field: str, value) -> None:
    store, loose = tmp_path / "store", tmp_path / "loose"
    inputs = loose / legacy_history.CORPORA[corpus] / "inputs/qwen3/one.yaml"
    inputs.parent.mkdir(parents=True)
    case = {field: value}
    document = {"family": "qwen3", field: value, "cases": {"A": case}}
    inputs.write_text(yaml.safe_dump(document))
    _loose_capture(loose, corpus, "vllm_python-0.1.0", "recorded")
    legacy_history.update_from_loose(store, loose)
    before = _store_bytes(store)
    case.pop(field)
    inputs.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match="captured legacy input is immutable"):
        legacy_history.update_from_loose(store, loose)
    assert _store_bytes(store) == before
