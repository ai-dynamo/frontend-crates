# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native publication admission, lossless transport, and failure preservation."""
import fcntl
import gzip
import hashlib
import json
import os
import signal
import shlex
import subprocess
from pathlib import Path

import pytest
import viewer_catalog

from native_publication import fixture_catalog, publish, publisher_binary, runtime_catalog, snapshot_bytes
from verify_publication_transport import Decoder, compare, read_index
from test_unified_history import _store


def _fixture(tmp_path):
    root = tmp_path / "source"
    _store(root / "conformance/fixtures-unified-v2")
    catalog = fixture_catalog(root, {"sources": {"candidates": [{"mode": "unified", "implementation": "dynamo_v2", "version": "0.5.2"}]},
                                     "metadata": {"unified_capture_policy": {"dynamo_v2": {"mode": "stream"}}}})
    output = tmp_path / "site"
    output.mkdir()
    (output / "previous.bin").write_bytes(b"previous publication")
    return root, output, catalog


def _save(root, catalog):
    catalog.pop("revision", None)
    catalog["revision"] = hashlib.sha256(json.dumps(catalog, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    (root / "conformance/viewer-catalog.json").write_text(json.dumps(catalog))
    (root / "conformance/viewer-unified-catalog.json").write_text(json.dumps(viewer_catalog.unified_projection(catalog)))


def _preserved(root, output, *, extra=()):
    result = publish(root, output, extra=extra, check=False)
    assert result.returncode != 0, result.stderr
    assert (output / "previous.bin").read_bytes() == b"previous publication"
    return result.stderr


def test_publication_source_integrity_and_revision(published_viewer):
    site = published_viewer
    catalog = runtime_catalog(site)
    provenance = json.loads((site / "conformance/viewer-publication.json").read_bytes())
    command = shlex.split(provenance['command'])
    assert command[command.index('--revision') + 1] == provenance['generated_revision']
    assert provenance["catalog_revision"] == catalog["revision"]
    assert provenance["resolved_catalog_revision"] == catalog["revision"]
    assert provenance["generated_on"].endswith((" PST", " PDT"))
    assert "conformance-publish" in provenance["command"]
    assert ".native-publication-stage-" not in provenance["command"]
    assert not (site / "conformance/viewer/worker.js").exists()
    assert '<style>' in (site / 'index.html').read_text()
    assert not (site / "conformance/viewer-original-resources.json.gz").exists()
    assert 'id="conformance-publication"' in (site / "index.html").read_text()
    assert not [path for path in site.rglob("*") if path.is_file() and path.name.endswith((".yaml", ".yml", ".tar.gz", ".md", ".txt", ".test.js"))]
    decoder = Decoder(site.resolve())
    for resource in catalog["sources"]["unified"]:
        data = snapshot_bytes(site, resource, decoder)
        assert len(data) == resource["size"]
        assert hashlib.sha256(data).hexdigest() == resource["sha256"]
        assert data.endswith(b"\n")
    source_root = Path(__file__).resolve().parents[3]
    source_catalog = json.loads((source_root / "conformance/viewer-catalog.json").read_bytes())
    originals = {resource["path"]: resource for kind in ("unified", "archives") for resource in source_catalog["sources"][kind]}
    seen = set()
    for descriptor in catalog["transport"]["raw_buckets"].values():
        for path, entry in read_index(site, descriptor).items():
            if "resource" in entry:
                resource = entry["resource"]
                assert resource["sha256"] == originals[path]["sha256"]
                assert resource["size"] == originals[path]["size"]
                seen.add(path)
    assert seen == originals.keys()


@pytest.mark.parametrize("path", ["../outside.yaml", "/outside.yaml", "a//b.yaml", "javascript:invalid", "a\\b.yaml"])
def test_publication_rejects_outside_source_paths(tmp_path, path):
    root, output, catalog = _fixture(tmp_path)
    catalog["sources"]["unified"][0]["path"] = path
    _save(root, catalog)
    _preserved(root, output)


@pytest.mark.parametrize("fault", ["revision", "resource", "document", "projection", "manifest", "directory"])
def test_invalid_source_cannot_overwrite_site(tmp_path, fault):
    root, output, catalog = _fixture(tmp_path)
    if fault == "revision":
        catalog["revision"] = "0" * 64
        (root / "conformance/viewer-catalog.json").write_text(json.dumps(catalog))
    elif fault == "resource":
        (root / catalog["sources"]["unified"][0]["path"]).write_bytes(b"corrupt: true\n")
    elif fault == "document":
        (root / "document.bin").write_bytes(b"changed")
        catalog["metadata"]["source_hashes"]["documentation"] = {"document.bin": "0" * 64}
        _save(root, catalog)
    elif fault == "projection":
        (root / "conformance/viewer-unified-catalog.json").write_bytes(b"{}")
    elif fault == "manifest":
        (root / "conformance/fixtures-manifest.json").write_bytes(b'{"shards":[{"path":"missing.tar.gz"}]}')
    else:
        (root / "conformance/fixtures-unified-v2/unknown").mkdir()
    _preserved(root, output)


def test_data_only_refresh_preserves_checkout_sources(tmp_path):
    root, _output, catalog = _fixture(tmp_path)
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    publish(root, root, workers=1, extra=("--data-only",))
    assert all(path.read_bytes() == data for path, data in before.items())
    old_catalog = runtime_catalog(root)
    old_resource = next(item for item in old_catalog["sources"]["unified"] if item["kind"] == "inputs-and-golden")
    old_path = root / old_resource["path"]
    old_bytes = old_path.read_bytes()

    authored = next(item for item in catalog["sources"]["unified"] if item["kind"] == "inputs-and-golden")
    with (root / authored["path"]).open("ab") as stream:
        stream.write(b"# immutable generation transition\n")
    fixture_catalog(root, catalog)
    publish(root, root, workers=1, extra=("--data-only",))

    new_catalog = runtime_catalog(root)
    new_resource = next(item for item in new_catalog["sources"]["unified"] if item["kind"] == "inputs-and-golden")
    assert new_resource["path"] != old_resource["path"]
    assert old_path.read_bytes() == old_bytes
    assert hashlib.sha256(old_path.read_bytes()).hexdigest() == old_resource["sha256"]
    metadata = json.loads((root / "conformance/viewer-publication.json").read_bytes())
    assert metadata["source_checkout"] is True
    assert "--data-only" in metadata["command"]
    assert (root / "conformance/viewer-data/generations").is_dir()


def test_native_validation_waits_for_source_writer(tmp_path):
    root, _output, _catalog = _fixture(tmp_path)
    result_path = tmp_path / "validation.json"
    lease = os.open(root / "conformance", os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(lease, fcntl.LOCK_EX)
        process = subprocess.Popen(
            [str(publisher_binary()), "--root", str(root), "--output", str(result_path),
             "--validate-unified", "--workers", "1"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        try:
            with pytest.raises(subprocess.TimeoutExpired):
                process.communicate(timeout=1)
            assert not result_path.exists()
            fcntl.flock(lease, fcntl.LOCK_UN)
            stdout, stderr = process.communicate(timeout=30)
            assert process.returncode == 0, (stdout, stderr)
            assert json.loads(result_path.read_bytes())["resources"] > 0
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()
    finally:
        os.close(lease)


@pytest.mark.parametrize("name", ["viewer-data", "viewer-resolved-catalog.json", "viewer-resolved-catalog.json.gz", "viewer-publication.json"])
def test_data_refresh_rejects_symlink_outputs(tmp_path, name):
    root, output, _catalog = _fixture(tmp_path)
    (root / "conformance" / name).symlink_to(output)
    result = publish(root, root, extra=("--data-only",), check=False)
    assert result.returncode != 0
    assert (output / "previous.bin").read_bytes() == b"previous publication"


def test_symlinks_overlaps_and_gigabyte_sources_are_rejected(tmp_path):
    root, output, catalog = _fixture(tmp_path)
    linked = tmp_path / "linked"
    linked.symlink_to(output, target_is_directory=True)
    assert publish(root, linked / "site", check=False).returncode != 0
    original_manifest=(root / "conformance/fixtures-manifest.json").read_bytes()
    for target in (root, root / "conformance/viewer", root.parent, root / "conformance/fixtures-manifest.json"):
        assert publish(root, target, check=False).returncode != 0
    assert (root / "conformance/fixtures-manifest.json").read_bytes() == original_manifest
    huge = root / "huge.bin"
    with huge.open("wb") as stream:
        stream.truncate(1_000_000_000)
    catalog["metadata"]["source_hashes"]["documentation"] = {"huge.bin": "0" * 64}
    _save(root, catalog)
    assert "1 GB" in _preserved(root, output)


def test_killed_staging_process_keeps_previous_publication(tmp_path):
    root, output, catalog = _fixture(tmp_path)
    authored = next(item for item in catalog["sources"]["unified"] if item["kind"] == "inputs-and-golden")
    path = root / authored["path"]
    with path.open("ab") as stream:
        stream.write(b"# staging interruption probe\n" * 400_000)
    fixture_catalog(root, catalog)
    command = [str(publisher_binary()), "--root", str(root), "--output", str(output), "--revision", "a" * 40, "--workers", "1"]
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
        assert '"phase":"admission"' in process.stderr.readline()
        process.kill()
        assert process.wait(timeout=10) == -signal.SIGKILL
    assert (output / "previous.bin").read_bytes() == b"previous publication"
    publish(root, output, workers=1)
    assert (output / "conformance/viewer-resolved-catalog.json.gz").is_file()


def test_source_inventory_mutation_during_publication_preserves_output(tmp_path):
    root, output, catalog = _fixture(tmp_path)
    authored = next(item for item in catalog["sources"]["unified"] if item["kind"] == "inputs-and-golden")
    with (root / authored["path"]).open("ab") as stream:
        stream.write(b"# concurrent inventory probe\n" * 400_000)
    fixture_catalog(root, catalog)
    command = [str(publisher_binary()), "--root", str(root), "--output", str(output), "--revision", "a" * 40, "--workers", "1"]
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
        assert '"phase":"admission"' in process.stderr.readline()
        os.kill(process.pid, signal.SIGSTOP)
        try:
            assert process.poll() is None
            added = (root / authored["path"]).with_name("concurrent.yaml")
            added.write_bytes(b"schema_version: 1\n")
        finally:
            os.kill(process.pid, signal.SIGCONT)
        _stdout, stderr = process.communicate(timeout=60)
        assert process.returncode != 0, stderr
        assert "source inventory differs from catalog" in stderr
    assert (output / "previous.bin").read_bytes() == b"previous publication"
    added.unlink()
    publish(root, output, workers=1)


def test_packed_baseline_verifier_compares_logical_bytes_and_metadata(tmp_path):
    root, baseline, _catalog = _fixture(tmp_path)
    candidate = tmp_path / "candidate"
    publish(root, baseline, workers=1)
    publish(root, candidate, workers=2)
    result = compare(baseline.resolve(), candidate.resolve())
    assert result["equal"] is True
    assert result["capture_selections"] == 1
    assert result["family_inputs"] == 1
    assert result["original_resources"] > 0
    catalog = runtime_catalog(candidate)
    catalog["metadata"]["markers"] = {"changed": True}
    catalog_path = candidate / "conformance/viewer-resolved-catalog.json.gz"
    catalog_path.write_bytes(gzip.compress(json.dumps(catalog).encode(), mtime=0))
    with pytest.raises(ValueError, match="report metadata differs"):
        compare(baseline.resolve(), candidate.resolve())
