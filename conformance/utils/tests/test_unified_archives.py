# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import gzip
import io
import os
import shutil
import tarfile
from types import SimpleNamespace
import pytest
import unified_archives
import unified_history
from viewer_catalog import _unified_records
from test_unified_history import _change, _store, _write_capture
from pathlib import Path
import migrate_capture_sources
from conformance.utils.tests.test_unified_history import _write_family, _write_capture, _change


"""Version archives preserve source bytes and the existing history contracts."""





def test_archive_keeps_latest_two_and_exact_history_bytes(tmp_path):
    root = _store(tmp_path / "store")
    _write_capture(root, "0.5.3", {"text_only": _change("new")})
    before = unified_history.load_store(root)
    inventory = unified_history.store_inventory(root)
    original = (root / "families/gemma4/dynamo_v2-0.5.0.yaml").read_bytes()
    result = unified_history.archive_versions(root)
    assert result["archives"] == 1
    assert result["logical_documents"] == len(inventory)
    archive = root / "versions/dynamo-0.5.0.tar.gz"
    assert list(unified_archives.members(archive)) == [("families/gemma4/dynamo_v2-0.5.0.yaml", original)]
    assert not (root / "families/gemma4/dynamo_v2-0.5.0.yaml").exists()
    assert (root / "families/gemma4/dynamo_v2-0.5.2.yaml").is_file()
    assert (root / "families/gemma4/dynamo_v2-0.5.3.yaml").is_file()
    assert unified_history.store_inventory(root) == inventory
    after = unified_history.load_store(root)
    unified_history._validate_store_parity(before, after)
    assert after.root == root.resolve()
    assert after.families["gemma4"].path == root / "families/gemma4/inputs_and_golden.yaml"
    records = _unified_records(unified=root, store=after)
    archived = next(row for row in records if row.get("archive"))
    assert archived["archive"]["path"] == "conformance/fixtures-unified-v2/versions/dynamo-0.5.0.tar.gz"
    assert archived["sha256"] != archived["archive"]["sha256"]
    compressed = archive.read_bytes()
    assert unified_history.archive_versions(root) == result
    assert archive.read_bytes() == compressed
    # Existing mutations must preserve the reduced layout and rotate retention.
    unified_history.rewrite_store(root)
    unified_history._validate_store_parity(after, unified_history.load_store(root))
    assert archive.is_file()
    assert not (root / "families/gemma4/dynamo_v2-0.5.0.yaml").exists()


def test_members_spools_archive_and_enforces_decoded_limit(tmp_path, monkeypatch):
    archive_path = tmp_path / "dynamo-0.1.0.tar.gz"
    with tarfile.open(archive_path, "w:gz") as output:
        payload = b"x" * 1024
        member = tarfile.TarInfo("families/sample/dynamo_v2-0.1.0.yaml")
        member.size = len(payload)
        output.addfile(member, io.BytesIO(payload))

    spool_directories = []
    original_temporary_file = unified_archives.tempfile.TemporaryFile

    def tracked_temporary_file(*args, **kwargs):
        spool_directories.append(kwargs.get("dir"))
        return original_temporary_file(*args, **kwargs)

    monkeypatch.setattr(unified_archives.tempfile, "TemporaryFile", tracked_temporary_file)
    assert list(unified_archives.members(archive_path, temporary_parent=tmp_path)) == [
        ("families/sample/dynamo_v2-0.1.0.yaml", payload)
    ]
    assert spool_directories == [tmp_path]

    monkeypatch.setattr(unified_archives, "MAX_DECODED", 512)
    with pytest.raises(ValueError, match="decoded limit"):
        list(unified_archives.members(archive_path, temporary_parent=tmp_path))
    assert set(path.name for path in tmp_path.iterdir()) == {archive_path.name}


def test_archive_read_rejects_source_mutation(tmp_path, monkeypatch):
    archive_path = tmp_path / "dynamo-0.1.0.tar.gz"
    archive_path.write_bytes(gzip.compress(b"archive bytes"))
    original_fstat = os.fstat
    calls = 0

    def mutate_identity(fd):
        nonlocal calls
        calls += 1
        stat = original_fstat(fd)
        return SimpleNamespace(
            st_dev=stat.st_dev,
            st_ino=stat.st_ino,
            st_size=stat.st_size,
            st_mtime_ns=stat.st_mtime_ns + int(calls == 2),
            st_ctime_ns=stat.st_ctime_ns,
        )

    monkeypatch.setattr(unified_archives.os, "fstat", mutate_identity)
    with pytest.raises(ValueError, match="changed while being read"):
        unified_archives._read_archive(archive_path)


@pytest.mark.parametrize("kind", [tarfile.GNUTYPE_LONGNAME, tarfile.GNUTYPE_LONGLINK,
                                 tarfile.XHDTYPE, tarfile.XGLTYPE])
def test_members_checks_extension_declaration_before_tarfile_reads_body(tmp_path, monkeypatch, kind):
    member = tarfile.TarInfo("extension")
    member.type = kind
    member.size = 65537
    archive_path = tmp_path / "dynamo-0.1.0.tar.gz"
    archive_path.write_bytes(gzip.compress(member.tobuf() + bytes(66048) + bytes(1024)))

    def forbidden_open(*args, **kwargs):
        pytest.fail("tarfile was opened before oversized extension admission")

    monkeypatch.setattr(unified_archives.tarfile, "open", forbidden_open)
    with pytest.raises(ValueError, match="extension metadata"):
        list(unified_archives.members(archive_path))


def test_members_bounds_combined_extensions_per_logical_member(tmp_path, monkeypatch):
    extensions = []
    for kind in [tarfile.GNUTYPE_LONGNAME, tarfile.GNUTYPE_LONGLINK]:
        member = tarfile.TarInfo("extension")
        member.type = kind
        member.size = 32769
        extensions.extend([member.tobuf(), bytes(33280)])
    archive_path = tmp_path / "dynamo-0.1.0.tar.gz"
    archive_path.write_bytes(gzip.compress(b"".join(extensions) + bytes(1024)))

    def forbidden_open(*args, **kwargs):
        pytest.fail("tarfile was opened before combined extension admission")

    monkeypatch.setattr(unified_archives.tarfile, "open", forbidden_open)
    with pytest.raises(ValueError, match="extension metadata"):
        list(unified_archives.members(archive_path))


def test_members_accepts_boundary_gnu_extension_and_streams_large_padding(tmp_path):
    name = "families/sample/dynamo_v2-0.1.0.yaml"
    extension = tarfile.TarInfo("././@LongLink")
    extension.type = tarfile.GNUTYPE_LONGNAME
    extension.size = 65536
    member = tarfile.TarInfo(name)
    member.size = 1
    raw = (extension.tobuf() + name.encode() + bytes(65536 - len(name))
           + member.tobuf() + b"x" + bytes(511) + bytes(2 * 1024 * 1024))
    archive_path = tmp_path / "dynamo-0.1.0.tar.gz"
    archive_path.write_bytes(gzip.compress(raw))
    assert list(unified_archives.members(archive_path)) == [(name, b"x")]


def test_members_bounds_tiny_extension_chains(tmp_path, monkeypatch):
    extension = tarfile.TarInfo("././@LongLink")
    extension.type = tarfile.GNUTYPE_LONGNAME
    extension.size = 0
    archive_path = tmp_path / "dynamo-0.1.0.tar.gz"
    archive_path.write_bytes(gzip.compress(extension.tobuf() * 129 + bytes(1024)))

    def forbidden_open(*args, **kwargs):
        pytest.fail("tarfile was opened before extension chain admission")

    monkeypatch.setattr(unified_archives.tarfile, "open", forbidden_open)
    with pytest.raises(ValueError, match="extension metadata"):
        list(unified_archives.members(archive_path))


def test_archived_mutation_keeps_new_latest_receipt_readable(tmp_path):
    root = _store(tmp_path / "store")
    _write_capture(root, "0.5.3", {})
    unified_history.compact_execution_store(root)
    unified_history.archive_versions(root)

    def add_receipt(store):
        capture = store.histories["gemma4", "dynamo_v2"].captures["dynamo_v2-0.5.3"]
        capture["metadata_changes"] = {"text_only": {"verified_recaptures": [{
            "provenance": {"job_sha256": "a" * 64, "adapter_sha256": "b" * 64, "source_sha256": "c" * 64, "version": "0.5.3"}, "result": {"old": None}}]}}

    unified_history._mutate_store(root, add_receipt, compact_executions=True)
    store = unified_history.load_store(root)
    manifest = store.resources[store.resources["families/gemma4/dynamo_v2-0.5.3.yaml"]["execution"]]
    assert manifest["receipts"]
    assert all((root / reference).is_file() for reference in manifest["receipts"])
    assert unified_history.compact_execution_store(root) == []


def test_engine_version_archive_groups_families(tmp_path, monkeypatch):
    root = _store(tmp_path / "store")
    _write_capture(root, "0.5.3", {})
    family = root / "families/gemma4"
    other = root / "families/qwen3"
    shutil.copytree(family, other)
    # Packing is byte-preserving and independent of the semantic family reader.
    unified_archives.pack(root, unified_history.CAPTURE_DIRECTORY_RE, unified_history._capture_release_sort_key)
    assert len(unified_archives.paths(root)) == 1
    assert {name for name, _ in unified_archives.members(unified_archives.paths(root)[0])} == {
        "families/gemma4/dynamo_v2-0.5.0.yaml", "families/qwen3/dynamo_v2-0.5.0.yaml"}

    rows = [row for row in unified_archives.inventory(root) if row.get("archive")]
    calls = []
    original = unified_archives.members

    def measured(path, **options):
        calls.append(path)
        yield from original(path, **options)

    monkeypatch.setattr(unified_archives, "members", measured)
    assert len(list(unified_archives.read_sources(root, rows))) == 2
    assert calls == unified_archives.paths(root)


def test_complete_snapshot_pack_uses_canonical_header_without_yaml_parse(tmp_path, monkeypatch):
    source = _store(tmp_path / "source")
    root = tmp_path / "store"
    family = root / "families/gemma4"
    family.mkdir(parents=True)
    shutil.copyfile(source / "families/gemma4/inputs_and_golden.yaml", family / "inputs_and_golden.yaml")
    capture = root / "families/gemma4/dynamo_v2-0.5.0.yaml"
    capture.write_text(unified_history.dump_yaml({
        "schema_version": 2,
        "provenance": {},
        "document": {},
        "records": {},
    }), encoding="utf-8")

    def reject_yaml_parse(_data):
        raise AssertionError("canonical complete snapshot headers should avoid a full YAML parse")

    monkeypatch.setattr(unified_archives.yaml, "safe_load", reject_yaml_parse)
    unified_archives.pack(root, unified_history.CAPTURE_DIRECTORY_RE, unified_history._capture_release_sort_key)

    assert len(unified_archives.paths(root)) == 1


@pytest.mark.parametrize("name,kind", [("../escape.yaml", tarfile.REGTYPE),
                                     ("receipts/link.yaml", tarfile.SYMTYPE),
                                     ("families/gemma4/inputs_and_golden.yaml", tarfile.REGTYPE)])
def test_archives_reject_unsafe_members(tmp_path, name, kind):
    archive = tmp_path / "dynamo-0.1.0.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        member = tarfile.TarInfo(name)
        member.type = kind
        member.size = 2
        output.addfile(member, io.BytesIO(b"{}"))
    with pytest.raises(ValueError, match="invalid"):
        list(unified_archives.members(archive))


def test_duplicate_loose_and_archived_source_is_rejected(tmp_path):
    root = _store(tmp_path / "store")
    _write_capture(root, "0.5.3", {})
    path = root / "families/gemma4/dynamo_v2-0.5.0.yaml"
    original = path.read_bytes()
    unified_history.archive_versions(root)
    path.write_bytes(original)
    with pytest.raises(ValueError, match="duplicate"):
        unified_history.store_inventory(root)
    with pytest.raises(FileExistsError):
        unified_history.load_store(root)


def test_corrupt_archive_is_not_hidden_by_loose_files(tmp_path):
    root = _store(tmp_path / "store")
    _write_capture(root, "0.5.3", {})
    unified_history.archive_versions(root)
    archive = unified_archives.paths(root)[0]
    archive.write_bytes(archive.read_bytes()[:-5])
    with pytest.raises((EOFError, gzip.BadGzipFile, ValueError)):
        unified_history.load_store(root)


@pytest.mark.parametrize("tail", [b"garbage", gzip.compress(b"", mtime=0), gzip.compress(b"\0" * 1024, mtime=0)],
                         ids=["trailing-bytes", "extra-empty-member", "extra-data-member"])
def test_archives_reject_extra_gzip_members_and_bytes(tmp_path, tail):
    root = _store(tmp_path / "store")
    _write_capture(root, "0.5.3", {})
    unified_history.archive_versions(root)
    archive = unified_archives.paths(root)[0]
    archive.write_bytes(archive.read_bytes() + tail)
    with pytest.raises(ValueError, match="trailing"):
        unified_history.load_store(root)


def test_two_capture_store_remains_readable_without_empty_versions(tmp_path):
    root = _store(tmp_path / "store")
    before = unified_history.load_store(root)
    assert unified_history.archive_versions(root)["archives"] == 0
    assert not (root / "versions").exists()
    unified_history._validate_store_parity(before, unified_history.load_store(root))


def test_archiving_rejects_linked_family_directory(tmp_path):
    root = _store(tmp_path / "store")
    original = root / "families/gemma4"
    outside = tmp_path / "outside"
    original.rename(outside)
    original.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="links forbidden"):
        unified_history.archive_versions(root)
    assert original.is_symlink()
    assert not (root / "versions").exists()


def test_archive_bytes_are_deterministic_and_members_are_readable(tmp_path: Path) -> None:
    member = "families/sample/dynamo_v2-1.2.3.yaml"
    payload = b"schema_version: 2\nrecords: {}\n"
    archives = []
    for name in ("first", "second"):
        root = tmp_path / name
        family = root / "families/sample"
        family.mkdir(parents=True)
        (family / "inputs_and_golden.yaml").write_text("family: sample\ncases: {}\n")
        (root / member).write_bytes(payload)
        unified_archives.pack(root, unified_history.CAPTURE_DIRECTORY_RE,
                              unified_history._capture_release_sort_key)
        archives.append(next((root / "versions").iterdir()))

    assert archives[0].read_bytes() == archives[1].read_bytes()
    assert list(unified_archives.members(archives[0])) == [(member, payload)]


@pytest.mark.parametrize('override', [None, '0.9.0', 'absent'])
def test_conversion_preserves_derived_inheritance_overrides(tmp_path, override):
    source = tmp_path / 'source'
    _write_family(source)
    _write_capture(source, '1.0.0', {'text_only': _change()})
    path = _write_capture(source, '1.1.0', {})
    document = unified_history.load_yaml(path)
    if override != 'absent':
        document['document_overrides'] = {'text_only': {'capture_provenance': {'inherited_from': override, 'captured_with': {'dynamo_v2': '1.1.0'}, 'capture_origin': None}}}
        path.write_text(unified_history.dump_yaml(document))
    sparse = unified_history.load_store(source).histories[('gemma4', 'dynamo_v2')]
    expected = sparse.resolve('dynamo_v2-1.1.0', view_capture_id='dynamo_v2-1.2.0')
    complete = tmp_path / 'complete'
    migrate_capture_sources.convert_snapshots(source, complete, tmp_path)
    history = unified_history.load_store(complete).histories[('gemma4', 'dynamo_v2')]
    actual = history.resolve('dynamo_v2-1.1.0', view_capture_id='dynamo_v2-1.2.0')
    assert unified_history.typed_equal(actual, expected)
    assert ('text_only' in history.snapshot_suppressed_inheritance['dynamo_v2-1.1.0']) == (override is None)
    rewritten = unified_history._store_documents(unified_history.load_store(complete))
    converted = rewritten[history.capture_path('dynamo_v2-1.1.0')]
    if override is None:
        assert converted['records']['text_only']['suppress_inherited_from'] is True
        converted['records']['text_only']['suppress_inherited_from'] = False
        with pytest.raises(ValueError, match='invalid snapshot inheritance projection'):
            unified_history._validate_complete_snapshot(path, converted, history.family, 'dynamo_v2', '1.1.0')


def test_complete_snapshot_updates_keep_and_clear_projection_markers(tmp_path):
    source = tmp_path / 'source'
    _write_family(source)
    path = _write_capture(source, '1.0.0', {'text_only': _change()})
    document = unified_history.load_yaml(path)
    document['document_overrides'] = {'text_only': {'capture_provenance': {
        'captured_with': {'dynamo_v2': '1.0.0'}, 'capture_origin': None, 'inherited_from': None,
    }}}
    path.write_text(unified_history.dump_yaml(document))
    complete = tmp_path / 'complete'
    migrate_capture_sources.convert_snapshots(source, complete, tmp_path)
    history = unified_history.load_store(complete).histories[('gemma4', 'dynamo_v2')]
    first = 'dynamo_v2-1.0.0'
    later = 'dynamo_v2-1.1.0'
    history.captures[later] = {**history.captures[first], 'runtime_version': '1.1.0'}
    history.capture_paths[later] = history.family.path.parent / (later + '.yaml')
    original = history.resolve(first)
    unified_history._set_capture_snapshot(history, later, original, original)
    assert history.snapshot_suppressed_inheritance[later] == {'text_only'}
    assert 'inherited_from' not in history.resolve(later, view_capture_id='dynamo_v2-1.2.0')['text_only']['document']
    changed = history.resolve(first)
    changed['text_only']['observation'] = {'value': {'assembled': [{'kind': 'text', 'text': 'new'}]}}
    unified_history._set_capture_snapshot(history, first, changed, {})
    assert history.snapshot_suppressed_inheritance[first] == set()
    unified_history._set_capture_snapshot(history, later, {}, original)
    assert history.snapshot_suppressed_inheritance[later] == set()
