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
from types import SimpleNamespace

import pytest

import capture_sources as sources
from publish_current_capture import choose_current
import publish_current_capture
import extract_fixtures
import unified_archives
import unified_history
import migrate_unified_snapshots
from test_unified_history import _store
from native_publication import publisher_binary, fixture_catalog


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
    snapshot = "conformance/captures/unified/sample/dynamo_v2.yaml"
    data = b"schema_version: 2\nrecords: {}\n"
    path = tmp_path / snapshot
    path.parent.mkdir(parents=True)
    path.write_bytes(data)
    authored = tmp_path / sources.STORE / "families/sample/inputs_and_golden.yaml"
    authored.parent.mkdir(parents=True)
    authored.write_bytes(b"cases: {}\n")
    row = {**sources.identity(snapshot, data), "source": sources.STORE + "/families/sample/dynamo_v2-1.0.0.yaml",
           "family": "sample", "implementation": "dynamo_v2", "version": "1.0.0", "surface": "unified", "mode": "unified"}
    (tmp_path / sources.CURRENT).write_bytes(sources.encoded({"schema": 1, "captures": [row], "lineage": []}))
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
    migrate_unified_snapshots.convert(sparse, history / sources.STORE, tmp_path)
    root = tmp_path / "code"
    target_store = root / sources.STORE
    shutil.copytree(history / sources.STORE / "families", target_store / "families")
    resources = unified_archives.inventory(history / sources.STORE)
    resource = next(row for row in resources if row["path"].endswith("dynamo_v2-0.5.2.yaml"))
    data = unified_archives.read_source(history / sources.STORE, resource)
    path = "conformance/captures/unified/gemma4/dynamo_v2.yaml"
    (root / path).parent.mkdir(parents=True)
    (root / path).write_bytes(data)
    row = {**sources.identity(path, data), "source": sources.STORE + "/" + resource["path"],
           "family": "gemma4", "implementation": "dynamo_v2", "version": "0.5.2", "mode": "unified", "surface": "unified"}
    lineage = [{"path": sources.STORE + "/" + item["path"], **unified_history.source_identity(item["path"])}
               for item in resources if item["path"].endswith(".yaml") and "dynamo_v2-" in item["path"]]
    (root / sources.CURRENT).write_bytes(sources.encoded({"schema": 1, "captures": [row], "lineage": lineage}))
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


@pytest.mark.parametrize("failure_call", [None, 1, 2])
def test_actual_capture_publish_and_recovery_preserve_evidence(split_evidence, monkeypatch, failure_call):
    root, history, capture, path = split_evidence
    original = publish_current_capture.unified_history.publish_paths_transactionally
    calls = []

    def fail(replacements, **kwargs):
        calls.append(replacements)
        if len(calls) == failure_call:
            raise OSError("injected pointer publication failure")
        return original(replacements, **kwargs)

    monkeypatch.setattr(publish_current_capture.unified_history, "publish_paths_transactionally", fail)
    if failure_call is None:
        assert publish_current_capture.publish(root, history, capture, "gemma4", "dynamo_v2", "0.5.3")["current_advanced"]
    else:
        with pytest.raises(OSError, match="injected"):
            publish_current_capture.publish(root, history, capture, "gemma4", "dynamo_v2", "0.5.3")
        with pytest.raises(ValueError, match="pending"):
            sources.current(root)
        monkeypatch.setattr(publish_current_capture.unified_history, "publish_paths_transactionally", original)
        assert publish_current_capture.recover(root, history)
    pin = json.loads((root / sources.PIN).read_bytes())
    sources.verify(history, pin)
    assert (root / path).read_bytes() == capture.read_bytes()
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


def test_publication_identity_excludes_installed_dependencies():
    for name in ("conformance/viewer/node_modules/.bin/esbuild", "conformance/viewer/node_modules/esbuild/lib/main.js"):
        assert not sources.ci_publication_cache.is_publication_input(name)
    for name in ("conformance/viewer/package.json", "conformance/viewer/package-lock.json"):
        assert sources.ci_publication_cache.is_publication_input(name)


@pytest.mark.parametrize("mode", ["latest", "history"])
def test_isolated_report_modes_share_native_publication(split_evidence, tmp_path, monkeypatch, native_publisher, mode):
    root, history, _capture, _path = split_evidence
    original_run = sources.subprocess.run
    originals = Path(sources.__file__).parent
    for name in ("capture_sources.py", "ci_publication_cache.py", "yaml_fast.py"):
        target = root / "conformance/utils/src" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(originals / name, target)
    (root / "index.html").write_bytes(b"<head></head><body></body>")
    (root / "conformance/CONFORMANCE_v2.html").write_bytes(b'<a href="../">Viewer</a>')
    (root / "conformance/fixtures-manifest.json").write_bytes(b'{"shards":[]}\n')
    source_files = [path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()]
    monkeypatch.setattr(sources.subprocess, "check_output", lambda *_args, **_kwargs: "\0".join(source_files).encode())
    view = tmp_path / "view"
    view.mkdir()

    def build_catalog(command, **kwargs):
        if any(str(value).endswith("viewer_catalog.py") for value in command):
            generation = json.loads((view / "conformance/capture-generation.json").read_bytes())
            if not unified_archives.paths(view / sources.STORE):
                unified_archives.pack(view / sources.STORE, unified_history.CAPTURE_DIRECTORY_RE,
                                      unified_history._capture_release_sort_key)
            versions = ["0.5.0", "0.5.2"] if mode == "history" else ["0.5.2"]
            fixture_catalog(view, {"capture_generation": generation,
                "metadata": {"unified_capture_policy": {"dynamo_v2": {"mode": "stream"}}},
                "sources": {"lineage": generation["lineage"], "candidates": [
                    {"mode": "unified", "implementation": "dynamo_v2", "version": version} for version in versions]}})
            return SimpleNamespace(returncode=0)
        return original_run(command, **kwargs)

    monkeypatch.setattr(sources.subprocess, "run", build_catalog)
    monkeypatch.delenv("CONFORMANCE_DOCUMENT_LOCK", raising=False)
    sources.prepare(root, view, include_history=mode == "history", history_source=history if mode == "history" else None,
                    publisher=str(native_publisher))
    output = tmp_path / "published"
    result = original_run([str(native_publisher), "--root", str(view), "--output", str(output), "--revision", "a" * 40],
                          capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    catalog = json.loads(gzip.decompress((output / "conformance/viewer-resolved-catalog.json.gz").read_bytes()))
    assert catalog["capture_generation"]["mode"] == mode
    versions = {item["version"] for item in catalog["sources"]["unified"] if item["kind"] == "resolved-capture"}
    assert versions == ({"0.5.0", "0.5.2"} if mode == "history" else {"0.5.2"})
    assert (output / "index.html").is_file()
    before = (output / "index.html").read_bytes()
    (history / sources.INVENTORY).write_bytes(b"corrupt")
    if mode == "history":
        with pytest.raises(ValueError, match="pin mismatch"):
            sources.verify_prepared(view)
        rejected = original_run([str(native_publisher), "--root", str(view), "--output", str(output), "--revision", "a" * 40],
                                capture_output=True, text=True, timeout=60)
        assert rejected.returncode != 0
        assert "capture source changed during publication" in rejected.stderr
    else:
        sources.verify_prepared(view)
    assert (output / "index.html").read_bytes() == before


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
    archived = next(item for item in unified_archives.inventory(history / sources.STORE)
                    if item["path"] == row["source"].removeprefix(sources.STORE + "/"))
    assert unified_archives.read_source(history / sources.STORE, archived) == (root / path).read_bytes()


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
    result = subprocess.run(["node", str(root / "conformance/viewer/status.js"), "--root", str(root),
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
    for name in ("capture_sources.py", "ci_publication_cache.py", "yaml_fast.py"):
        target = root / "conformance/utils/src" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(scripts / name, target)
    (root / "conformance/viewer-catalog.json").write_bytes(sources.encoded({"capture_generation": {"current": []}}))
    temporary = tmp_path / "status-temporary"
    temporary.mkdir()
    environment = dict(os.environ)
    environment["TMPDIR"] = str(temporary)
    destination = tmp_path / "status.json"
    result = subprocess.run(["node", str(original_root / "conformance/viewer/status.js"), "--root", str(root),
                             "--model", "qwen3", "--tab", "unified", "--summary-only", "--status-path", str(destination)],
                            env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 2
    assert "current source catalog is stale" in result.stderr
    assert not destination.exists()
    assert not list(temporary.glob("current-source-status-*"))


@pytest.mark.parametrize("crash_call", [1, 2])
def test_capture_writer_recovers_process_exit_before_transaction_journal(split_evidence, monkeypatch, crash_call):
    root, history, capture, path = split_evidence
    original = publish_current_capture.unified_history.publish_paths_transactionally
    calls = []

    def crash(replacements, **kwargs):
        calls.append(replacements)
        if len(calls) == crash_call:
            os._exit(93)
        return original(replacements, **kwargs)

    monkeypatch.setattr(publish_current_capture.unified_history, "publish_paths_transactionally", crash)
    pid = os.fork()
    if pid == 0:
        publish_current_capture.publish(root, history, capture, "gemma4", "dynamo_v2", "0.5.3")
        os._exit(0)
    _pid, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 93
    monkeypatch.setattr(publish_current_capture.unified_history, "publish_paths_transactionally", original)
    assert publish_current_capture.recover(root, history)
    sources.verify(history, json.loads((root / sources.PIN).read_bytes()))
    assert (root / path).read_bytes() == capture.read_bytes()
