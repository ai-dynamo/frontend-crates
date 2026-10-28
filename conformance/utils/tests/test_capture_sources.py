# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The evidence boundary rejects damaged pins and keeps current reads local."""
import io
import json
from pathlib import Path
import tarfile
import shutil
import gzip
import subprocess
import os
import sys
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import pytest

import capture_sources as sources
from publish_current_capture import choose_current
import publish_current_capture
import extract_fixtures
import unified_archives
import unified_history
import migrate_capture_sources
from test_unified_history import _store
from fixture_test_support import publish, publisher_binary
from fixture_test_support import prepare_isolated_publication


def pin_history(root):
    inventory = {**sources.inventory(root), "source_revision": "a" * 40}
    data = sources.encoded(inventory)
    (root / sources.INVENTORY).write_bytes(data)
    return {"schema": 1, "source_revision": "a" * 40, "inventory_sha256": sources.digest(data)}


def test_history_verifies_original_bytes_members_and_missing_versus_null(tmp_path):
    history = tmp_path / "history"
    history.mkdir()
    archive = history / "capture.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        for name, data in (("family/absent.yaml", b"{}\n"), ("family/null.yaml", b"value: null\n")):
            member = tarfile.TarInfo(name)
            member.size = len(data)
            output.addfile(member, io.BytesIO(data))
    pin = pin_history(history)
    inventory = sources.verify(history, pin)
    assert inventory["files"][0]["members"][0]["sha256"] != inventory["files"][0]["members"][1]["sha256"]
    archive.write_bytes(archive.read_bytes()[:-8])
    with pytest.raises((ValueError, EOFError, tarfile.TarError)):
        sources.verify(history, pin)


@pytest.mark.parametrize("damage", ["pin", "missing", "extra", "nested-inventory", "revision", "symlink"])
def test_history_rejects_incorrect_or_incomplete_evidence(tmp_path, damage):
    (tmp_path / "one.yaml").write_bytes(b"one: null\n")
    pin = pin_history(tmp_path)
    if damage == "pin":
        pin["inventory_sha256"] = "0" * 64
    elif damage == "missing":
        (tmp_path / "one.yaml").unlink()
    elif damage == "extra":
        (tmp_path / "extra.yaml").write_bytes(b"extra: true\n")
    elif damage == "nested-inventory":
        (tmp_path / "nested").mkdir()
        (tmp_path / "nested" / sources.INVENTORY).write_bytes(b"unindexed evidence\n")
    elif damage == "revision":
        pin["source_revision"] = "b" * 40
    else:
        (tmp_path / "one.yaml").rename(tmp_path / "original.yaml")
        (tmp_path / "one.yaml").symlink_to("original.yaml")
    with pytest.raises(ValueError):
        sources.verify(tmp_path, pin)


def test_current_interpretation_never_opens_history(tmp_path, monkeypatch):
    snapshot = "conformance/fixtures-unified-v2/families/sample/dynamo_v2.yaml"
    data = b"schema_version: 2\nrecords: {}\n"
    path = tmp_path / snapshot
    path.parent.mkdir(parents=True)
    path.write_bytes(data)
    authored = tmp_path / sources.STORE / "families/sample/inputs_and_golden.yaml"
    authored.parent.mkdir(parents=True, exist_ok=True)
    authored.write_bytes(b"cases: {}\n")
    row = {**sources.identity(snapshot, data), "source": sources.STORE + "/families/sample/dynamo_v2-1.0.0.yaml",
           "family": "sample", "implementation": "dynamo_v2", "version": "1.0.0", "surface": "unified", "mode": "unified"}
    path.write_bytes(sources.selected_bytes(data, row, []))
    data = path.read_bytes()
    monkeypatch.setattr(sources, "verify", lambda *_: pytest.fail("current read attempted history retrieval"))
    with sources.unified_view(tmp_path) as view:
        assert (view / "families/sample/dynamo_v2-1.0.0.yaml").read_bytes() == data
        assert (view / "families/sample/inputs_and_golden.yaml").read_bytes() == authored.read_bytes()
    path.write_bytes(data + b"corruption: true\n")
    with pytest.raises(ValueError, match="identity mismatch"):
        sources.current(tmp_path)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a//b", "a/./b", "a\\b", "a:b"])
def test_capture_paths_cannot_escape_source(name):
    with pytest.raises(ValueError):
        sources.relative(name)


def test_release_gaps_modes_and_backfills_do_not_change_a_newer_family_capture():
    older = {"surface": "unified", "family": "one", "implementation": "dynamo_v2", "mode": "unified", "version": "1.0.0", "sha256": "a"}
    newer = {**older, "family": "two", "version": "2.0.0"}
    separate_mode = {**older, "implementation": "dynamo_v1_batch", "version": "3.0.0"}
    rows, advanced = choose_current([older, newer, separate_mode], {**older, "version": "1.5.0", "sha256": "b"})
    assert advanced
    assert [row["version"] for row in rows if row["family"] == "two"] == ["2.0.0"]
    assert separate_mode in rows
    backfilled, advanced = choose_current(rows, {**older, "version": "1.2.0"})
    assert not advanced and backfilled == rows
    with pytest.raises(ValueError, match="frozen"):
        choose_current(rows, {**older, "version": "1.5.0", "sha256": "changed"})


@pytest.fixture
def split_evidence(tmp_path):
    sparse = _store(tmp_path / "sparse")
    history = tmp_path / "history"
    migrate_capture_sources.convert_snapshots(sparse, history / sources.STORE, tmp_path)
    root = tmp_path / "code"
    target_store = root / sources.STORE
    shutil.copytree(history / sources.STORE / "families", target_store / "families")
    resources = unified_archives.inventory(history / sources.STORE)
    resource = next(row for row in resources if row["path"].endswith("dynamo_v2-0.5.2.yaml"))
    data = unified_archives.read_source(history / sources.STORE, resource)
    path = "conformance/fixtures-unified-v2/families/gemma4/dynamo_v2.yaml"
    (root / path).parent.mkdir(parents=True, exist_ok=True)
    (root / path).write_bytes(data)
    row = {**sources.identity(path, data), "source": sources.STORE + "/" + resource["path"],
           "family": "gemma4", "implementation": "dynamo_v2", "version": "0.5.2", "mode": "unified", "surface": "unified"}
    lineage = [{"path": sources.STORE + "/" + item["path"], **unified_history.source_identity(item["path"])}
               for item in resources if item["path"].endswith(".yaml") and "dynamo_v2-" in item["path"]]
    (root / path).write_bytes(sources.selected_bytes(data, row, lineage))
    digest, size = unified_history.store_digest(history / sources.STORE)
    manifest = history / "conformance/fixtures-manifest.json"
    manifest.write_bytes(sources.encoded({"shards": [{"path": "unified-history", "format": "unified-history", "sha256": digest, "size": size}]}))
    (history / "conformance/fixtures").mkdir()
    pin = pin_history(history)
    (root / sources.PIN).write_bytes(sources.encoded(pin))
    snapshot = unified_history.yaml.load(data, Loader=unified_history.StrictLoader)
    snapshot["provenance"]["captured_with"]["dynamo_v2"] = "0.5.3"
    capture = tmp_path / "complete.yaml"
    capture.write_text(unified_history.dump_yaml(snapshot))
    return root, history, capture, path


@pytest.mark.parametrize("failure_call", [None, 1, 2, 3])
def test_actual_capture_publish_and_recovery_preserve_evidence(split_evidence, monkeypatch, failure_call):
    root, history, capture, path = split_evidence
    original = publish_current_capture.atomic_write
    calls = []

    def fail(target, data, **kwargs):
        if target.name.startswith(".capture-update-"):
            return original(target, data, **kwargs)
        calls.append(target)
        if len(calls) == failure_call:
            raise OSError("injected pointer publication failure")
        return original(target, data, **kwargs)

    monkeypatch.setattr(publish_current_capture, "atomic_write", fail)
    if failure_call is None:
        assert publish_current_capture.publish(root, history, capture, "gemma4", "dynamo_v2", "0.5.3")["current_advanced"]
    else:
        with pytest.raises(OSError, match="injected"):
            publish_current_capture.publish(root, history, capture, "gemma4", "dynamo_v2", "0.5.3")
        with pytest.raises(ValueError, match="pending"):
            sources.current(root)
        monkeypatch.setattr(publish_current_capture, "atomic_write", original)
        assert publish_current_capture.recover(root, history)
    pin = json.loads((root / sources.PIN).read_bytes())
    sources.verify(history, pin)
    assert unified_history.typed_equal(unified_history.load_yaml(root / path)["records"], sources.independent_snapshot(unified_history.load_yaml(capture), unified_history.load_yaml(root / sources.STORE / "families/gemma4/inputs_and_golden.yaml"), source_member=sources.current(root)["captures"][0]["source"])["records"])
    assert sources.current(root)["captures"][0]["version"] == "0.5.3"
    assert not (root / sources.UPDATE_JOURNAL).exists()
    assert not publish_current_capture.publish(root, history, capture, "gemma4", "dynamo_v2", "0.5.3")["archived"]


def test_authored_copy_guard_uses_the_copied_bytes(tmp_path):
    root = tmp_path / "code"
    name = sources.STORE + "/families/sample/inputs_and_golden.yaml"
    path = root / name
    path.parent.mkdir(parents=True)
    path.write_bytes(b"revision: before\n")
    view = tmp_path / "view"
    identities = sources.copy_authored(root, view / sources.STORE)
    path.write_bytes(b"revision: after\n")
    guard = {"root": str(root), "inputs": identities, "controls": [], "history_source": None}
    (view / "conformance/capture-source-guard.json").write_bytes(sources.encoded(guard))
    with pytest.raises(ValueError, match="source changed"):
        sources.verify_prepared(view)


@pytest.fixture
def native_publisher():
    return publisher_binary()


def test_native_publication_failure_reports_captured_stderr(tmp_path, native_publisher):
    with pytest.raises(subprocess.CalledProcessError) as failure:
        publish(tmp_path, tmp_path / "published", extra=("--invalid-publication-option",))
    assert "unexpected argument '--invalid-publication-option'" in failure.value.stderr
    assert failure.value.__notes__ == [failure.value.stderr]
    result = publish(tmp_path, tmp_path / "published", extra=("--invalid-publication-option",), check=False)
    assert result.returncode != 0
    assert result.stderr == failure.value.stderr


def test_publication_identity_excludes_installed_dependencies():
    for name in ("conformance/viewer/node_modules/.bin/esbuild", "conformance/viewer/node_modules/esbuild/lib/main.js"):
        assert not sources.ci_publication_cache.is_publication_input(name)
    for name in ("conformance/viewer/package.json", "conformance/viewer/package-lock.json"):
        assert sources.ci_publication_cache.is_publication_input(name)


@pytest.mark.parametrize("mode", ["latest", "history"])
def test_isolated_report_modes_share_native_publication(split_evidence, tmp_path, monkeypatch, native_publisher, mode):
    root, history, _capture, _path = split_evidence
    view = tmp_path / "view"
    prepare_isolated_publication(root, history, view, monkeypatch, native_publisher, mode=mode)
    output = tmp_path / "published"
    result = subprocess.run([str(native_publisher), "--root", str(view), "--output", str(output), "--revision", "a" * 40],
                          capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    catalog = json.loads(gzip.decompress((output / "conformance/viewer-resolved-catalog.json.gz").read_bytes()))
    assert catalog["capture_generation"]["mode"] == mode
    publication = json.loads((output / "conformance/viewer-publication.json").read_bytes())
    with monkeypatch.context() as patch:
        paths = [path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()]
        patch.setattr(subprocess, "check_output", lambda *_args, **_kwargs: "\0".join(paths).encode())
        expected = sources.publication_inputs_digest(root, include_history=mode == "history", history_source=history if mode == "history" else None)
        assert publication["publisher_inputs_sha256"] == expected
        original = (root / "index.html").read_bytes()
        (root / "index.html").write_bytes(b"changed publisher input")
        assert sources.publication_inputs_digest(root, include_history=mode == "history", history_source=history if mode == "history" else None) != expected
        (root / "index.html").write_bytes(original)
    versions = {item["version"] for item in catalog["sources"]["unified"] if item["kind"] == "resolved-capture"}
    assert versions == ({"0.5.0", "0.5.2"} if mode == "history" else {"0.5.2"})
    assert (output / "index.html").is_file()
    before = (output / "index.html").read_bytes()
    (history / sources.INVENTORY).write_bytes(b"corrupt")
    if mode == "history":
        with pytest.raises(ValueError, match="pin mismatch"):
            sources.verify_prepared(view)
        rejected = subprocess.run([str(native_publisher), "--root", str(view), "--output", str(output), "--revision", "a" * 40],
                                capture_output=True, text=True, timeout=60)
        assert rejected.returncode != 0
        assert "capture source changed during publication" in rejected.stderr
    else:
        sources.verify_prepared(view)
    assert (output / "index.html").read_bytes() == before


@pytest.mark.parametrize("mutation", ["changed", "deleted"])
def test_history_requests_keep_the_retained_authored_binding(split_evidence, tmp_path, monkeypatch, native_publisher, mutation):
    root, history, _capture, _path = split_evidence
    name = sources.STORE + "/families/gemma4/inputs_and_golden.yaml"
    original = sources.read(history, name)
    authored = root / name
    document = unified_history.load_yaml(authored)
    if mutation == "deleted":
        authored.unlink()
    else:
        document["cases"]["text_only"]["request"]["input"] = "replacement request"
        authored.write_text(unified_history.dump_yaml(document))
    before = {path.relative_to(history): path.read_bytes() for path in history.rglob("*") if path.is_file()}
    view = tmp_path / "retained-history-view"
    prepare_isolated_publication(root, history, view, monkeypatch, native_publisher, mode="history")
    assert sources.read(view, name) == original
    # Loading performs sealed stimulus-digest validation against the retained
    # request as well as resolving references that predate digest seals.
    unified_history.load_store(view / sources.STORE)
    sources.verify_prepared(view)
    assert {path.relative_to(history): path.read_bytes() for path in history.rglob("*") if path.is_file()} == before


def test_independent_latest_request_survives_authored_edits(split_evidence):
    root, _history, _capture, path = split_evidence
    snapshot = unified_history.load_yaml(root / path)
    authored_path = root / sources.STORE / "families/gemma4/inputs_and_golden.yaml"
    authored = unified_history.load_yaml(authored_path)
    independent = sources.independent_snapshot(snapshot, authored)
    old_request = independent["records"]["text_only"]["value"]["stimulus"]["inline"]
    assert old_request == authored["cases"]["text_only"]["request"]
    authored["cases"]["text_only"]["request"]["input"] = "changed current input"
    family = unified_history._validate_family(authored_path, authored)
    unified_history._validate_complete_snapshot(root / path, independent, family, "dynamo_v2", "0.5.2")
    assert old_request["input"] != authored["cases"]["text_only"]["request"]["input"]


def test_republishing_unsealed_baseline_is_an_idempotent_backfill(split_evidence, tmp_path):
    root, history, _capture, path = split_evidence
    resource = next(row for row in unified_archives.inventory(history / sources.STORE)
                    if row["path"].endswith("dynamo_v2-0.5.0.yaml"))
    capture = tmp_path / "baseline.yaml"
    capture.write_bytes(unified_archives.read_source(history / sources.STORE, resource))
    before = sources.read(root, path)
    result = publish_current_capture.publish(root, history, capture, "gemma4", "dynamo_v2", "0.5.0")
    assert result == {"archived": False, "current_advanced": False}
    assert sources.read(root, path) == before
    assert not (history / "additions").exists()


def test_capture_workflow_noop_keeps_current_bytes(split_evidence, tmp_path):
    root, history, _capture, path = split_evidence
    before = (root / path).read_bytes()
    loose = tmp_path / "loose"
    loose.mkdir()
    assert publish_current_capture.publish_loose(root, history, loose) == []
    assert (root / path).read_bytes() == before


def test_capture_workflow_archives_and_advances_new_loose_capture(split_evidence, tmp_path):
    root, history, _capture, path = split_evidence
    before = (root / path).read_bytes()
    loose = tmp_path / "loose"
    with sources.unified_view(root) as view:
        store, _targets = unified_history.selected_materialization_store(view, ["latest:dynamo_v2"])
        unified_history.materialize_store(view, loose, store=store)
    capture_directory = loose / "dynamo_v2-0.5.3"
    (loose / "dynamo_v2-0.5.2").rename(capture_directory)
    capture_path = capture_directory / "gemma4/UNIFIED.1-1.yaml"
    document = unified_history.load_yaml(capture_path)
    document["captured_with"]["dynamo_v2"] = "0.5.3"
    document.pop("inherited_from", None)
    document["cases"]["UNIFIED.1-1"]["assembled"][0]["text"] = "changed"
    document["cases"]["UNIFIED.1-1"]["chunks"][0]["expected"][0]["text"] = "changed"
    capture_path.write_text(unified_history.dump_yaml(document))

    result = publish_current_capture.publish_loose(root, history, loose)
    assert result == [{"archived": True, "current_advanced": True,
                       "source": sources.STORE + "/families/gemma4/dynamo_v2-0.5.3.yaml"}]
    row = sources.current(root)["captures"][0]
    assert row["version"] == "0.5.3"
    assert (root / path).read_bytes() != before
    snapshot = unified_history.load_yaml(root / path)
    assert snapshot["records"]["text_only"]["value"]["observation"]["value"]["assembled"][0]["text"] == "changed"
    sources.verify(history, json.loads((root / sources.PIN).read_bytes()))
    descriptors = list((history / "additions").glob("*/*.json"))
    assert len(descriptors) == 1
    descriptor = json.loads(descriptors[0].read_bytes())
    assert descriptor["selection"]["source"] == row["source"]


def test_current_extraction_invalidates_unified_authored_cache(split_evidence, tmp_path, monkeypatch, capsys):
    root, _history, _capture, _path = split_evidence
    monkeypatch.setattr(extract_fixtures, "ROOT", root)
    monkeypatch.setattr(extract_fixtures, "HISTORY_DIR", root / sources.STORE)
    monkeypatch.setattr(extract_fixtures, "get_cache_root", lambda: tmp_path / "cache")
    args = SimpleNamespace(info=False, dry_run=False, full_refresh=False)
    extract_fixtures._extract(args)
    first = Path(capsys.readouterr().out.strip().splitlines()[-1])
    authored = root / sources.STORE / "families/gemma4/inputs_and_golden.yaml"
    document = unified_history.load_yaml(authored)
    document["cases"]["text_only"]["description"] = "updated authored description"
    authored.write_text(unified_history.dump_yaml(document))

    extract_fixtures._extract(args)
    second = Path(capsys.readouterr().out.strip().splitlines()[-1])
    materialized = unified_history.load_yaml(second / "unified/inputs/gemma4/UNIFIED.1-1.yaml")
    assert second != first
    assert materialized["cases"]["UNIFIED.1-1"]["description"] == "updated authored description"
    original = unified_history.load_yaml(first / "unified/inputs/gemma4/UNIFIED.1-1.yaml")
    assert original["cases"]["UNIFIED.1-1"]["description"] == "plain text"


def test_current_extraction_rejects_authored_mutation_during_materialization(split_evidence, tmp_path, monkeypatch):
    root, _history, _capture, _path = split_evidence
    cache = tmp_path / "cache"
    monkeypatch.setattr(extract_fixtures, "ROOT", root)
    monkeypatch.setattr(extract_fixtures, "HISTORY_DIR", root / sources.STORE)
    monkeypatch.setattr(extract_fixtures, "get_cache_root", lambda: cache)
    original = unified_history.materialize_store
    authored = root / sources.STORE / "families/gemma4/inputs_and_golden.yaml"

    def materialize_then_edit(*args, **kwargs):
        original(*args, **kwargs)
        document = unified_history.load_yaml(authored)
        document["cases"]["text_only"]["description"] = "changed during materialization"
        authored.write_text(unified_history.dump_yaml(document))

    monkeypatch.setattr(unified_history, "materialize_store", materialize_then_edit)
    with pytest.raises(ValueError):
        extract_fixtures._extract(SimpleNamespace(info=False, dry_run=False, full_refresh=False))
    assert not list(cache.glob("current-*"))


def test_current_extraction_keeps_refreshed_generation_selected(split_evidence, tmp_path, monkeypatch, capsys):
    root, _history, _capture, _path = split_evidence
    monkeypatch.setattr(extract_fixtures, "ROOT", root)
    monkeypatch.setattr(extract_fixtures, "HISTORY_DIR", root / sources.STORE)
    monkeypatch.setattr(extract_fixtures, "get_cache_root", lambda: tmp_path / "cache")
    args = SimpleNamespace(info=False, dry_run=False, full_refresh=False)
    extract_fixtures._extract(args)
    initial = Path(capsys.readouterr().out.strip().splitlines()[-1])
    input_path = "unified/inputs/gemma4/UNIFIED.1-1.yaml"
    initial_bytes = (initial / input_path).read_bytes()

    args.full_refresh = True
    extract_fixtures._extract(args)
    refreshed = Path(capsys.readouterr().out.strip().splitlines()[-1])
    assert refreshed != initial
    assert (initial / input_path).read_bytes() == initial_bytes
    assert (refreshed / input_path).read_bytes() == initial_bytes

    def unexpected_materialization(*_args, **_kwargs):
        pytest.fail("normal extraction rebuilt instead of selecting the refreshed generation")

    monkeypatch.setattr(unified_history, "materialize_store", unexpected_materialization)
    args.full_refresh = False
    extract_fixtures._extract(args)
    selected = Path(capsys.readouterr().out.strip().splitlines()[-1])
    assert selected == refreshed


def test_current_source_status_needs_no_history_or_publisher_and_cleans_view(tmp_path, source_catalog):
    root = Path(sources.__file__).resolve().parents[3]
    temporary = tmp_path / "status-temporary"
    temporary.mkdir()
    environment = dict(os.environ)
    for key in ("CONFORMANCE_HISTORY_SOURCE", "CONFORMANCE_HISTORY_SITE", "CONFORMANCE_PUBLISHER"):
        environment.pop(key, None)
    environment["TMPDIR"] = str(temporary)
    destination = tmp_path / "status.json"
    result = subprocess.run(["node", str(root / "conformance/viewer/qualification.js"), "status", "--root", str(root),
                             "--model", "qwen3", "--tab", "unified", "--summary-only", "--status-path", str(destination)],
                            env=environment, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    assert "conformance status: 1 model/tab pairs" in result.stdout
    status = json.loads(destination.read_bytes())
    assert status["html"] == str(root / "index.html")
    assert status["reports"][0]["model"] == "qwen3"
    assert not list(temporary.glob("current-source-status-*"))


def test_current_source_status_rejects_stale_catalog_and_cleans_view(split_evidence, tmp_path):
    root, _history, _capture, _path = split_evidence
    original_root = Path(sources.__file__).resolve().parents[3]
    scripts = Path(sources.__file__).parent
    for name in ("capture_sources.py", "ci_publication_cache.py", "verify_publication_transport.py", "yaml_fast.py", "unified_archives.py", "execution_manifests.py", "impls.py", "viewer_codec.py", "parser_families.yaml"):
        target = root / "conformance/utils/src" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(scripts / name, target)
    (root / "conformance/viewer-catalog.json").write_bytes(sources.encoded({"capture_generation": {"current": []}}))
    temporary = tmp_path / "status-temporary"
    temporary.mkdir()
    environment = dict(os.environ)
    environment["TMPDIR"] = str(temporary)
    destination = tmp_path / "status.json"
    result = subprocess.run(["node", str(original_root / "conformance/viewer/qualification.js"), "status", "--root", str(root),
                             "--model", "qwen3", "--tab", "unified", "--summary-only", "--status-path", str(destination)],
                            env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 2
    assert "current source catalog is stale" in result.stderr
    assert not destination.exists()
    assert not list(temporary.glob("current-source-status-*"))


@pytest.mark.parametrize("crash_call", [1, 2, 3])
def test_capture_writer_recovers_process_exit_before_transaction_journal(split_evidence, monkeypatch, crash_call):
    root, history, capture, path = split_evidence
    original = publish_current_capture.atomic_write
    calls = []

    def crash(target, data, **kwargs):
        if target.name.startswith(".capture-update-"):
            return original(target, data, **kwargs)
        calls.append(target)
        if len(calls) == crash_call:
            os._exit(93)
        return original(target, data, **kwargs)

    monkeypatch.setattr(publish_current_capture, "atomic_write", crash)
    pid = os.fork()
    if pid == 0:
        publish_current_capture.publish(root, history, capture, "gemma4", "dynamo_v2", "0.5.3")
        os._exit(0)
    _pid, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 93
    monkeypatch.setattr(publish_current_capture, "atomic_write", original)
    assert publish_current_capture.recover(root, history)
    sources.verify(history, json.loads((root / sources.PIN).read_bytes()))
    assert unified_history.typed_equal(unified_history.load_yaml(root / path)["records"], sources.independent_snapshot(unified_history.load_yaml(capture), unified_history.load_yaml(root / sources.STORE / "families/gemma4/inputs_and_golden.yaml"), source_member=sources.current(root)["captures"][0]["source"])["records"])


def test_reconciliation_restores_conflicted_latest_from_baseline_and_additions(split_evidence):
    root, history, capture, path = split_evidence
    baseline_inventory = sources.read(history, sources.INVENTORY)
    publish_current_capture.publish(root, history, capture, "gemma4", "dynamo_v2", "0.5.3")
    expected = (root / path).read_bytes()
    (root / path).write_bytes(b"<<<<<<< branch-one\nconflicted YAML\n=======\nbranch-two\n>>>>>>> branch-two\n")
    publish_current_capture.reconcile(root, history)
    assert (root / path).read_bytes() == expected
    assert sources.read(history, sources.INVENTORY) == baseline_inventory
    assert not (root / "conformance/capture-current.json").exists()


def test_reconciliation_rejects_same_release_disagreement_before_replacing_latest(split_evidence, tmp_path):
    root, history, capture, path = split_evidence
    branch = tmp_path / "branch"
    branch_history = tmp_path / "branch-history"
    shutil.copytree(root, branch)
    shutil.copytree(history, branch_history)
    publish_current_capture.publish(root, history, capture, "gemma4", "dynamo_v2", "0.5.3")
    conflicting = unified_history.load_yaml(capture)
    conflicting["document"]["diagnostic"] = None
    capture.write_text(unified_history.dump_yaml(conflicting))
    publish_current_capture.publish(branch, branch_history, capture, "gemma4", "dynamo_v2", "0.5.3")
    for directory in (branch_history / "additions").iterdir():
        shutil.copytree(directory, history / "additions" / directory.name)
    before = (root / path).read_bytes()
    with pytest.raises(ValueError, match="conflicting frozen"):
        publish_current_capture.reconcile(root, history)
    assert (root / path).read_bytes() == before


def test_reconciliation_backfill_does_not_downgrade_baseline(split_evidence):
    root, history, capture, path = split_evidence
    # The baseline retains 0.5.2 even when a conflicted output cannot be read.
    document = unified_history.load_yaml(capture)
    document["provenance"]["captured_with"]["dynamo_v2"] = "0.5.1"
    capture.write_text(unified_history.dump_yaml(document))
    result = publish_current_capture.publish(root, history, capture, "gemma4", "dynamo_v2", "0.5.1")
    assert not result["current_advanced"]
    (root / path).write_bytes(b"<<<<<<< unresolved\n")
    publish_current_capture.reconcile(root, history)
    assert sources.current(root)["captures"][0]["version"] == "0.5.2"


def test_archive_and_descriptor_become_visible_together(split_evidence, monkeypatch):
    root, history, capture, _path = split_evidence
    original = publish_current_capture.atomic_write
    inspected = []

    def verify_during_staging(path, data, **kwargs):
        original(path, data, **kwargs)
        if path.suffix == ".gz":
            sources.verify(history, sources.history_pin(root, history))
            inspected.append(path)

    monkeypatch.setattr(publish_current_capture, "atomic_write", verify_during_staging)
    publish_current_capture.publish(root, history, capture, "gemma4", "dynamo_v2", "0.5.3")
    assert len(inspected) == 1
    sources.verify(history, sources.history_pin(root, history))


def test_latest_discovery_and_cache_ignore_history_changes(split_evidence, monkeypatch):
    root, history, _capture, _path = split_evidence
    paths = [path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()]
    monkeypatch.setattr(subprocess, "check_output", lambda *_args, **_kwargs: "\0".join(paths).encode())
    before = sources.publication_inputs_digest(root, include_history=False)
    historic = root / sources.HISTORIC
    historic.mkdir()
    (historic / "unrelated.yaml").write_bytes(b"invalid: historical-only\n")
    paths.append(sources.HISTORIC + "/unrelated.yaml")
    assert sources.publication_inputs_digest(root, include_history=False) == before
    assert not sources.ci_publication_cache.is_publication_input(paths[-1])


def test_publication_guard_rejects_added_capture_discovery(split_evidence, tmp_path):
    root, _history, _capture, path = split_evidence
    view = tmp_path / "guard-view"
    (view / "conformance").mkdir(parents=True)
    document = sources.current(root)
    guard = {"root": str(root), "inputs": [], "controls": [], "history_source": None,
             "current": sources.current_identities(document)}
    (view / "conformance/capture-source-guard.json").write_bytes(sources.encoded(guard))
    data = (root / path).read_bytes().split(b"\n", 1)[1]
    row = {**document["captures"][0], "family": "another",
           "source": sources.STORE + "/families/another/dynamo_v2-0.5.2.yaml"}
    target = root / "conformance/fixtures-unified-v2/families/another/dynamo_v2.yaml"
    target.parent.mkdir()
    target.write_bytes(sources.selected_bytes(data, row, []))
    with pytest.raises(ValueError, match="discovery changed"):
        sources.verify_prepared(view)


def test_independent_family_captures_publish_concurrently_without_shared_mutations(split_evidence, tmp_path):
    root, history, _capture, _path = split_evidence
    baseline = sources.read(history, sources.INVENTORY)
    pin = sources.read(root, sources.PIN)
    jobs = []
    for family in ("one", "two"):
        authored = root / "conformance/inputs/stream" / family / "case.yaml"
        authored.parent.mkdir(parents=True, exist_ok=True)
        authored.write_bytes(b"cases: {case: {model_text: hello}}\n")
        capture = tmp_path / (family + ".yaml")
        capture.write_text(unified_history.dump_yaml({"schema_version": 1, "surface": "stream", "family": family,
            "implementation": "vllm_python", "version": "1.0.0", "source_records": [],
            "documents": {"case.yaml": {"family": family, "cases": {"case": {"chunks": []}}}}}))
        jobs.append((root, history, capture, family, "vllm_python", "1.0.0", "stream"))
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda args: publish_current_capture.publish(*args), jobs))
    assert all(row["archived"] and row["current_advanced"] for row in results)
    assert len(list((history / "additions").glob("*/*.json"))) == 2
    assert sources.read(history, sources.INVENTORY) == baseline
    assert sources.read(root, sources.PIN) == pin
    assert len(sources.current(root)["captures"]) == 3
    view = tmp_path / "history-view"
    manifest = sources.stage_history(root, view, history)
    assert not sources.is_current(view)
    assert len(manifest["shards"]) == 3
    for shard in manifest["shards"]:
        if shard.get("format") == "unified-history":
            continue
        with tarfile.open(view / "conformance/fixtures" / shard["path"], "r:gz") as archive:
            assert len(archive.getmembers()) == 1
            assert "/vllm_python-1.0.0/" in archive.getmembers()[0].name


@pytest.mark.parametrize("damage", ["pending-unrelated", "corrupt-latest"])
def test_retained_history_admin_view_ignores_mutable_latest(split_evidence, tmp_path, monkeypatch, native_publisher, damage):
    root, history, _capture, path = split_evidence
    prefix = "conformance/history-datasets/base"
    shutil.copytree(tmp_path / "sparse", history / prefix / sources.STORE)
    (history / sources.DATASETS).write_bytes(sources.encoded([{"id": "base", "root": prefix}]))
    (root / sources.PIN).write_bytes(sources.encoded(pin_history(history)))
    if damage == "pending-unrelated":
        pending = root / "conformance/fixtures-unified-v2/families/another/.capture-update-dynamo_v2.json"
        pending.parent.mkdir(parents=True)
        pending.write_bytes(b"{}\n")
    else:
        (root / path).write_bytes(b"conflicted latest\n")
    with pytest.raises(ValueError):
        sources.current(root)
    view = tmp_path / "retained-view"
    prepare_isolated_publication(root, history, view, monkeypatch, native_publisher,
                                 mode="history", retained_history_only=True)
    retained = unified_history.load_store(view / sources.STORE)
    assert retained.histories["gemma4", "dynamo_v2"].resolve("dynamo_v2-0.5.2")
    sources.verify_prepared(view)


def test_added_snapshot_is_in_combined_history_without_rewriting_baseline(split_evidence, tmp_path):
    root, history, capture, _path = split_evidence
    publish_current_capture.publish(root, history, capture, "gemma4", "dynamo_v2", "0.5.3")
    view = tmp_path / "combined"
    sources.stage_history(root, view, history)
    store = unified_history.load_store(view / sources.STORE)
    assert "dynamo_v2-0.5.3" in store.histories[("gemma4", "dynamo_v2")].captures
    descriptor = next((history / "additions").glob("*/*.json"))
    descriptor.unlink()
    with pytest.raises(ValueError, match="missing capture descriptor"):
        sources.verify(history, sources.history_pin(root, history))


def test_legacy_backfill_conflict_checks_older_baseline_release(tmp_path):
    history = tmp_path / "history"
    (history / sources.STORE).mkdir(parents=True)
    archive_path = history / "conformance/fixtures/toolcalling/fixtures-batch-v1/evidence.tar.gz"
    archive_path.parent.mkdir(parents=True)
    documents = {
        "inputs": {"family": "demo", "cases": {"one": {"input": "hello"}}},
        "vllm_python-1.0.0": {"family": "demo", "cases": {"one": {"expected": {"vllm_python": {"value": "original"}}}}},
        "vllm_python-2.0.0": {"family": "demo", "cases": {"one": {"expected": {"vllm_python": {"value": "newer"}}}}},
    }
    with tarfile.open(archive_path, "w:gz") as archive:
        for version, document in documents.items():
            body = unified_history.dump_yaml(document).encode()
            member = tarfile.TarInfo(f"toolcalling/fixtures-batch-v1/{version}/demo/case.yaml")
            member.size = len(body)
            archive.addfile(member, io.BytesIO(body))
    (history / "conformance/fixtures-manifest.json").write_bytes(sources.encoded({"shards": []}))
    pin = pin_history(history)
    root = tmp_path / "code"
    migrate = publish_current_capture.migrate_capture_sources
    migrate.convert(root, history)
    (root / sources.PIN).write_bytes(sources.encoded(pin))
    target = root / "conformance/captures/batch/demo/vllm_python.yaml"
    before = target.read_bytes()
    snapshot = unified_history.load_yaml(target)
    snapshot.pop(sources.SELECTION)
    snapshot["version"] = "1.0.0"
    snapshot["documents"]["case.yaml"]["cases"]["one"]["expected"]["vllm_python"]["value"] = "CONFLICT"
    capture = tmp_path / "conflict.yaml"
    capture.write_text(unified_history.dump_yaml(snapshot))
    with pytest.raises(ValueError, match="conflicting frozen legacy"):
        publish_current_capture.publish(root, history, capture, "demo", "vllm_python", "1.0.0", "batch")
    assert target.read_bytes() == before
    assert not (history / "additions").exists()


def test_history_staging_rejects_conflicting_legacy_branch_additions(split_evidence, tmp_path):
    root, history, _capture, _path = split_evidence
    authored = root / "conformance/inputs/batch/demo/case.yaml"
    authored.parent.mkdir(parents=True, exist_ok=True)
    authored.write_bytes(b"family: demo\ncases: {one: {input: hello}}\n")
    branch = tmp_path / "branch"
    other_history = tmp_path / "other-history"
    shutil.copytree(root, branch)
    shutil.copytree(history, other_history)
    for checkout, evidence, value in ((root, history, "A"), (branch, other_history, "B")):
        capture = tmp_path / (value + ".yaml")
        capture.write_text(unified_history.dump_yaml({"schema_version": 1, "surface": "batch", "family": "demo",
            "implementation": "vllm_python", "version": "1.0.0", "source_records": [],
            "documents": {"case.yaml": {"family": "demo", "cases": {"one": {"input": "hello", "expected": {"vllm_python": {"value": value}}}}}}}))
        publish_current_capture.publish(checkout, evidence, capture, "demo", "vllm_python", "1.0.0", "batch")
    for directory in (other_history / "additions").iterdir():
        shutil.copytree(directory, history / "additions" / directory.name)
    with pytest.raises(ValueError, match="conflicting frozen legacy"):
        publish_current_capture.validate_legacy_baseline(root, history)
    with pytest.raises(subprocess.CalledProcessError):
        sources.stage_history(root, tmp_path / "rejected-view", history)


def test_capture_archives_have_portable_headers_and_stable_member_order():
    first = unified_archives.capture_archive_bytes({"b.yaml": b"b: null\n", "a.yaml": b"a: 1\n"})
    second = unified_archives.capture_archive_bytes({"a.yaml": b"a: 1\n", "b.yaml": b"b: null\n"})
    assert first == second
    assert first[9] == 255
    with tarfile.open(fileobj=io.BytesIO(first), mode="r:gz") as archive:
        assert [member.name for member in archive] == ["a.yaml", "b.yaml"]


def test_selected_release_is_independent_of_inherited_producer_identity(split_evidence):
    root, _history, _capture, path = split_evidence
    row = sources.current(root)["captures"][0]
    snapshot = unified_history.load_yaml(root / path)
    inherited = snapshot["records"]["text_only"]
    assert row["version"] == "0.5.2"
    assert inherited["member"].endswith("dynamo_v2-0.5.0.yaml")
    assert inherited["value"]["document"]["captured_with"]["dynamo_v2"] == "0.5.0"
    with sources.unified_view(root) as view:
        assert (view / "families/gemma4/dynamo_v2-0.5.2.yaml").is_file()


def test_reconciliation_selects_newest_of_two_branch_releases(split_evidence, tmp_path):
    root, history, capture, path = split_evidence
    branch = tmp_path / "second-branch"
    other_history = tmp_path / "second-history"
    shutil.copytree(root, branch)
    shutil.copytree(history, other_history)
    publish_current_capture.publish(root, history, capture, "gemma4", "dynamo_v2", "0.5.3")
    document = unified_history.load_yaml(capture)
    document["provenance"]["captured_with"]["dynamo_v2"] = "0.5.4"
    capture.write_text(unified_history.dump_yaml(document))
    publish_current_capture.publish(branch, other_history, capture, "gemma4", "dynamo_v2", "0.5.4")
    for directory in (other_history / "additions").iterdir():
        shutil.copytree(directory, history / "additions" / directory.name)
    (root / path).write_bytes(b"<<<<<<< first-branch\n=======\nsecond-branch\n>>>>>>>\n")
    publish_current_capture.reconcile(root, history)
    assert sources.current(root)["captures"][0]["version"] == "0.5.4"
    assert len(list((history / "additions").glob("*/*.json"))) == 2


def test_latest_staging_rejects_transient_capture_replacement(split_evidence, tmp_path, monkeypatch):
    root, _history, _capture, path = split_evidence
    original_read = sources.read
    row = sources.current(root)["captures"][0]
    original = unified_history.load_yaml(root / path)
    selection = original.pop(sources.SELECTION)
    original["document"]["transient"] = True
    changed = sources.selected_bytes(unified_history.dump_yaml(original).encode(), row, selection["lineage"])
    reads = 0

    def replace_during_copy(parent, name):
        nonlocal reads
        if parent == root and name == path:
            reads += 1
            if reads == 2:
                return changed
        return original_read(parent, name)

    monkeypatch.setattr(sources, "read", replace_during_copy)
    with pytest.raises(ValueError, match="changed during staging"):
        sources.stage_unified(root, tmp_path / "view")


def test_history_guard_binds_the_evidence_actually_staged(split_evidence, tmp_path, monkeypatch, native_publisher):
    root, history, capture, _path = split_evidence
    document = unified_history.load_yaml(capture)
    document["provenance"]["captured_with"]["dynamo_v2"] = "0.5.1"
    capture.write_text(unified_history.dump_yaml(document))
    original = sources.stage_history

    def add_after_copy(*args):
        manifest = original(*args)
        assert not publish_current_capture.publish(root, history, capture, "gemma4", "dynamo_v2", "0.5.1")["current_advanced"]
        return manifest

    monkeypatch.setattr(sources, "stage_history", add_after_copy)
    view = tmp_path / "view"
    prepare_isolated_publication(root, history, view, monkeypatch, native_publisher, mode="history")
    with pytest.raises(ValueError, match="source changed"):
        sources.verify_prepared(view)


def test_history_staging_rejects_transient_manifest_replacement(split_evidence, tmp_path, monkeypatch):
    root, history, _capture, _path = split_evidence
    original_read = sources.read
    manifest_name = "conformance/fixtures-manifest.json"

    def replace_manifest(parent, name):
        if parent == history and name == manifest_name:
            return sources.encoded({"shards": []})
        return original_read(parent, name)

    monkeypatch.setattr(sources, "read", replace_manifest)
    with pytest.raises(ValueError, match="changed during staging"):
        sources.stage_history(root, tmp_path / "transient-manifest-view", history)


def test_history_staging_rejects_transient_descriptor_replacement(split_evidence, tmp_path, monkeypatch):
    root, history, capture, _path = split_evidence
    publish_current_capture.publish(root, history, capture, "gemma4", "dynamo_v2", "0.5.3")
    descriptor = next((history / "additions").glob("*/*.json")).relative_to(history).as_posix()
    original_read = sources.read
    reads = 0

    def replace_descriptor(parent, name):
        nonlocal reads
        data = original_read(parent, name)
        if parent == history and name == descriptor:
            reads += 1
            if reads > 1:
                changed = json.loads(data)
                changed["archive"]["members"] = []
                return sources.encoded(changed)
        return data

    monkeypatch.setattr(sources, "read", replace_descriptor)
    with pytest.raises(ValueError, match="changed during staging"):
        sources.stage_history(root, tmp_path / "transient-descriptor-view", history)


def test_publication_guard_rejects_added_authored_family_during_staging(split_evidence, tmp_path, monkeypatch):
    root, _history, _capture, _path = split_evidence
    (root / "conformance/fixtures-manifest.json").write_bytes(sources.encoded({"shards": []}))
    original_stage = sources.stage_unified

    def add_authored_after_copy(*args, **kwargs):
        result = original_stage(*args, **kwargs)
        authored = root / sources.STORE / "families/another/inputs_and_golden.yaml"
        authored.parent.mkdir()
        authored.write_bytes(b"cases: {}\n")
        return result

    monkeypatch.setattr(sources, "publication_input_paths", lambda _root: ["conformance/fixtures-manifest.json"])
    monkeypatch.setattr(sources, "stage_unified", add_authored_after_copy)
    monkeypatch.setattr(sources.subprocess, "run", lambda *_args, **_kwargs: None)
    view = tmp_path / "added-authored-view"
    with pytest.raises(ValueError, match="membership changed"):
        sources.prepare(root, view)
        sources.verify_prepared(view)


def test_history_staging_copies_only_admitted_baseline_files(split_evidence, tmp_path, monkeypatch):
    root, history, _capture, _path = split_evidence
    relative = sources.STORE + "/families/gemma4/dynamo_v2-9.9.9.yaml"
    extra = history / relative
    original_verify = sources.verify

    def add_after_admission(*args, **kwargs):
        evidence = original_verify(*args, **kwargs)
        extra.write_bytes(b"schema_version: 2\nrecords: {}\n")
        return evidence

    monkeypatch.setattr(sources, "verify", add_after_admission)
    view = tmp_path / "admitted-baseline-view"
    try:
        sources.stage_history(root, view, history)
    finally:
        extra.unlink(missing_ok=True)
    assert not (view / relative).exists()


def test_latest_preparation_uses_the_copied_manifest(split_evidence, tmp_path, monkeypatch):
    root, _history, _capture, _path = split_evidence
    manifest_name = "conformance/fixtures-manifest.json"
    (root / manifest_name).write_bytes(sources.encoded({"snapshot": "admitted", "shards": []}))
    original_read = sources.read
    root_reads = 0

    def replace_on_second_read(parent, name):
        nonlocal root_reads
        if parent == root and name == manifest_name:
            root_reads += 1
            if root_reads > 1:
                return sources.encoded({"snapshot": "unadmitted", "shards": []})
        return original_read(parent, name)

    monkeypatch.setattr(sources, "publication_input_paths", lambda _root: [manifest_name])
    monkeypatch.setattr(sources, "read", replace_on_second_read)
    monkeypatch.setattr(sources.subprocess, "run", lambda *_args, **_kwargs: None)
    view = tmp_path / "copied-manifest-view"
    sources.prepare(root, view)
    assert json.loads((view / manifest_name).read_bytes())["snapshot"] == "admitted"
    assert root_reads == 1
    monkeypatch.setattr(sources, "read", original_read)
    sources.verify_prepared(view)


@pytest.mark.parametrize("mode", ["batch", "stream", "reasoning", "batch-on-stream"])
def test_legacy_publication_rejects_added_authored_document_before_finish(split_evidence, tmp_path, monkeypatch, mode):
    root, history, _capture, _path = split_evidence
    input_mode = "batch" if mode == "batch-on-stream" else mode
    authored = root / "conformance/inputs" / input_mode / "demo/case.yaml"
    authored.parent.mkdir(parents=True, exist_ok=True)
    authored.write_bytes(b"family: demo\ncases: {one: {input: hello}}\n")
    snapshot = {"schema_version": 1, "surface": mode, "family": "demo", "implementation": "vllm_python",
                "version": "0.9.0", "source_records": [],
                "documents": {"case.yaml": {"family": "demo", "cases": {"one": {"input": "hello"}}}}}
    row = {"surface": mode, "mode": mode, "family": "demo", "implementation": "vllm_python", "version": "0.9.0",
           "source": f"conformance/captures/{mode}/demo/vllm_python-0.9.0.yaml"}
    latest = root / f"conformance/captures/{mode}/demo/vllm_python.yaml"
    latest.parent.mkdir(parents=True)
    before = sources.selected_bytes(unified_history.dump_yaml(snapshot).encode(), row, [])
    latest.write_bytes(before)
    snapshot["version"] = "1.0.0"
    capture = tmp_path / "legacy-next.yaml"
    capture.write_text(unified_history.dump_yaml(snapshot))
    original_write = publish_current_capture.atomic_write

    def add_document_after_journal(path, data, **kwargs):
        original_write(path, data, **kwargs)
        if path.name.startswith(".capture-update-"):
            authored.with_name("new-case.yaml").write_bytes(b"family: demo\ncases: {two: {input: new}}\n")

    monkeypatch.setattr(publish_current_capture, "atomic_write", add_document_after_journal)
    with pytest.raises(ValueError, match="authored.*changed"):
        publish_current_capture.publish(root, history, capture, "demo", "vllm_python", "1.0.0", mode)
    assert latest.read_bytes() == before


@pytest.mark.parametrize("mode", ["batch", "stream", "reasoning", "batch-on-stream"])
@pytest.mark.parametrize("replacement", ["capture", "authored"])
def test_current_extraction_rejects_transient_legacy_source_bytes(tmp_path, monkeypatch, mode, replacement):
    root = tmp_path / "code"
    input_mode = "batch" if mode == "batch-on-stream" else mode
    authored_name = f"conformance/inputs/{input_mode}/demo/case.yaml"
    authored = root / authored_name
    authored.parent.mkdir(parents=True, exist_ok=True)
    authored.write_bytes(b"family: demo\ncases: {one: {input: original}}\n")
    capture_name = f"conformance/captures/{mode}/demo/vllm_python.yaml"
    capture = root / capture_name
    capture.parent.mkdir(parents=True)
    row = {"surface": mode, "mode": mode, "family": "demo", "implementation": "vllm_python", "version": "1.0.0",
           "source": f"conformance/captures/{mode}/demo/vllm_python-1.0.0.yaml"}
    snapshot = {"schema_version": 1, "surface": mode, "family": "demo", "implementation": "vllm_python",
                "version": "1.0.0", "source_records": [],
                "documents": {"case.yaml": {"family": "demo", "cases": {"one": {"expected": {"vllm_python": "original"}}}}}}
    original = sources.selected_bytes(unified_history.dump_yaml(snapshot).encode(), row, [])
    capture.write_bytes(original)
    snapshot["documents"]["case.yaml"]["cases"]["one"]["expected"]["vllm_python"] = "transient"
    replacement_bytes = (sources.selected_bytes(unified_history.dump_yaml(snapshot).encode(), row, [])
                         if replacement == "capture" else b"family: demo\ncases: {one: {input: transient}}\n")
    replaced_name = capture_name if replacement == "capture" else authored_name
    original_read = sources.read
    reads = 0

    def replace_only_consumed_bytes(parent, name):
        nonlocal reads
        if parent == root and name == replaced_name:
            reads += 1
            if reads == 2:
                return replacement_bytes
        return original_read(parent, name)

    cache = tmp_path / "cache"
    monkeypatch.setattr(sources, "read", replace_only_consumed_bytes)
    monkeypatch.setattr(extract_fixtures, "ROOT", root)
    monkeypatch.setattr(extract_fixtures, "get_cache_root", lambda: cache)
    with pytest.raises(ValueError, match="changed during staging"):
        extract_fixtures._extract(SimpleNamespace(info=False, dry_run=False, full_refresh=False), legacy_only=True)
    assert reads == 2
    assert capture.read_bytes() == original
    assert not list(cache.glob("current-*"))


def test_capture_owner_lock_is_shared_across_bulk_job_tmpdirs(tmp_path):
    root = tmp_path / "code"
    root.mkdir()
    script = (
        "from pathlib import Path\n"
        "import sys\n"
        "import publish_current_capture as publication\n"
        "print('waiting', flush=True)\n"
        "with publication.capture_lock(Path(sys.argv[1]), 'demo', 'vllm_python', 'batch'):\n"
        " print('acquired', flush=True)\n"
        " sys.stdin.readline()\n"
    )

    def start(scratch):
        scratch.mkdir()
        environment = {**os.environ, "TMPDIR": str(scratch), "TMP": str(scratch), "TEMP": str(scratch),
                       "PYTHONPATH": str(Path(sources.__file__).parent)}
        return subprocess.Popen([sys.executable, "-c", script, str(root)], env=environment,
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    first = start(tmp_path / "bulk-job-one")
    second = None
    try:
        assert first.stdout.readline().strip() == "waiting"
        assert first.stdout.readline().strip() == "acquired"
        second = start(tmp_path / "bulk-job-two")
        assert second.stdout.readline().strip() == "waiting"
        with pytest.raises(subprocess.TimeoutExpired):
            second.communicate(input="\n", timeout=0.3)
    finally:
        first.communicate(input="\n", timeout=5)
        if second is not None and second.poll() is None:
            second.communicate(timeout=5)
    assert first.returncode == 0
    assert second.returncode == 0


def test_recovery_accepts_journal_completed_after_discovery(split_evidence, monkeypatch):
    root, history, capture, path = split_evidence
    original_write = publish_current_capture.atomic_write

    def pause_after_journal(target, data, **kwargs):
        original_write(target, data, **kwargs)
        if target.name.startswith(".capture-update-"):
            raise SystemExit("paused capture owner")

    with monkeypatch.context() as patch:
        patch.setattr(publish_current_capture, "atomic_write", pause_after_journal)
        with pytest.raises(SystemExit, match="paused capture owner"):
            publish_current_capture.publish(root, history, capture, "gemma4", "dynamo_v2", "0.5.3")
    journal = sources.pending_updates(root)[0]
    original_lock = publish_current_capture.capture_lock
    completed = False

    @contextmanager
    def finish_before_recovery_lock(*args):
        nonlocal completed
        if not completed:
            completed = True
            with original_lock(root, "gemma4", "dynamo_v2", "unified"):
                publish_current_capture.finish(root, history, journal)
        with original_lock(*args):
            yield

    monkeypatch.setattr(publish_current_capture, "capture_lock", finish_before_recovery_lock)
    assert not publish_current_capture.recover(root, history)
    assert completed
    assert not journal.exists()
    assert sources.current(root)["captures"][0]["version"] == "0.5.3"
    assert unified_history.load_yaml(root / path)["records"] == sources.independent_snapshot(unified_history.load_yaml(capture), unified_history.load_yaml(root / sources.STORE / "families/gemma4/inputs_and_golden.yaml"), source_member=sources.current(root)["captures"][0]["source"])["records"]


def test_latest_parser_store_reads_versionless_files_without_staging(split_evidence, monkeypatch):
    root, _history, _capture, _path = split_evidence
    before = {path: path.read_bytes() for path in (root / sources.STORE / "families").glob("*/*.yaml")}
    monkeypatch.setattr(sources, "unified_view", lambda *_: pytest.fail("latest parser staged versioned capture files"))
    monkeypatch.setattr(unified_archives, "inventory", lambda *_args, **_kwargs: pytest.fail("latest parser inventoried history"))
    monkeypatch.setattr(sources, "verify", lambda *_args: pytest.fail("latest parser verified history"))
    store, targets = unified_history.selected_materialization_store(root / sources.STORE, ["latest:dynamo_v2"])
    assert targets == {"dynamo_v2": "0.5.2"}
    assert set(store.histories) == {("gemma4", "dynamo_v2")}
    assert {path: path.read_bytes() for path in before} == before


def test_explicit_alternate_dataset_has_its_own_pin(tmp_path):
    root = tmp_path / "root"
    (root / "conformance").mkdir(parents=True)
    bundled = root / sources.HISTORIC
    bundled.mkdir()
    bundled_pin = pin_history(bundled)
    (root / sources.PIN).write_bytes(sources.encoded(bundled_pin))
    alternate = tmp_path / "alternate"
    alternate.mkdir()
    (alternate / "evidence.yaml").write_bytes(b"original: true\n")
    alternate_pin = pin_history(alternate)
    assert sources.history_pin(root) == bundled_pin
    assert sources.history_pin(root, bundled) == bundled_pin
    assert sources.history_pin(root, alternate) == alternate_pin
    assert alternate_pin != bundled_pin
    sources.verify(alternate, alternate_pin)


def test_published_latest_downloads_its_complete_source_and_keeps_producer(split_evidence):
    root, history, capture, path = split_evidence
    before = unified_history.load_yaml(capture)["records"]["text_only"]["value"]["document"]
    published = publish_current_capture.publish(root, history, capture, "gemma4", "dynamo_v2", "0.5.3")
    latest = unified_history.load_yaml(root / path)
    assert latest["records"]["text_only"]["member"] == published["source"]
    assert unified_history.typed_equal(latest["records"]["text_only"]["value"]["document"], before)
    assert latest["records"]["text_only"]["value"]["document"]["captured_with"]["dynamo_v2"] == "0.5.0"
