# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Publication copies maintained sources and validates resources before writing."""
import hashlib
import json
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "conformance/utils/src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
import publish_viewer
import viewer_catalog
import unified_history


def test_publication_source_integrity_and_revision(tmp_path, monkeypatch):
    copied_documents = []
    copy = publish_viewer.copy_validated
    def copy_source(source, target, *args, **kwargs):
        # Document publication is verified separately under the shared document lock.
        if Path(target).suffix == ".md":
            copied_documents.append(str(Path(source).relative_to(ROOT)))
            return target
        return copy(source, target, *args, **kwargs)
    monkeypatch.setattr(publish_viewer, "copy_validated", copy_source)
    revision = "a" * 40
    publish_viewer.publish(tmp_path / "site", revision)
    site = tmp_path / "site"
    catalog = json.loads((site / "conformance/viewer-catalog.json").read_text())
    assert json.loads((site / "conformance/viewer-publication.json").read_text()) == {"schema": 1, "revision": revision, "catalog_revision": catalog["revision"]}
    assert (site / "index.html").read_bytes() == (ROOT / "index.html").read_bytes()
    assert (site / "conformance/utils/src/assets/conformance_view.js").is_file()
    assert not (site / "fixtures").exists()
    for resource in (*catalog["sources"]["archives"], *catalog["sources"]["unified"]):
        data = (site / resource["path"]).read_bytes()
        assert len(data) == resource["size"]
        assert hashlib.sha256(data).hexdigest() == resource["sha256"]
    assert set(catalog["metadata"]["source_hashes"]["documentation"]).issubset(copied_documents)


def test_invalid_revision_or_resource_cannot_overwrite_site(tmp_path, monkeypatch):
    output = tmp_path / "site"
    with pytest.raises(ValueError, match="immutable"):
        publish_viewer.publish(output, "main")
    assert not output.exists()
    catalog = json.loads(publish_viewer.CATALOG.read_text())
    catalog["sources"]["archives"][0]["sha256"] = "0" * 64
    invalid = tmp_path / "viewer-catalog.json"
    invalid.write_text(json.dumps(catalog))
    (tmp_path / "viewer-unified-catalog.json").write_text(json.dumps(viewer_catalog.unified_projection(catalog)))
    monkeypatch.setattr(publish_viewer, "CATALOG", invalid)
    with pytest.raises(ValueError, match="does not match catalog"):
        publish_viewer.publish(output, "a" * 40)
    assert not output.exists()


@pytest.mark.parametrize("path", ["../outside.yaml", "/outside.yaml", "a//b.yaml", "javascript:invalid", "a\\b.yaml"])
def test_publication_rejects_outside_source_paths(path):
    with pytest.raises(ValueError, match="repository-relative"):
        publish_viewer.source_path(path)


def test_stale_document_hash_rejects_before_publication(tmp_path, monkeypatch):
    catalog = json.loads(publish_viewer.CATALOG.read_text())
    name = next(iter(catalog["metadata"]["source_hashes"]["documentation"]))
    catalog["metadata"]["source_hashes"]["documentation"][name] = "0" * 64
    source = tmp_path / "viewer-catalog.json"
    source.write_text(json.dumps(catalog))
    (tmp_path / "viewer-unified-catalog.json").write_text(json.dumps(viewer_catalog.unified_projection(catalog)))
    monkeypatch.setattr(publish_viewer, "CATALOG", source)
    target = tmp_path / "site"
    with pytest.raises(ValueError, match="document does not match catalog"):
        publish_viewer.publish(target, "a" * 40)
    assert not target.exists()


def _private_publication(tmp_path, monkeypatch):
    root = tmp_path / "source"
    conformance = root / "conformance"
    (conformance / "viewer").mkdir(parents=True)
    (conformance / "utils/src/assets").mkdir(parents=True)
    for name, data in [("index.html", b"index"), ("conformance/CONFORMANCE_v2.html", b"redirect"),
                       ("conformance/viewer/app.js", b"app"), ("conformance/utils/src/assets/view.js", b"view"),
                       ("conformance/resource.yaml", b"resource"), ("conformance/doc.json", b"document"),
                       ("conformance/readme.bin", b"readme")]:
        (root / name).write_bytes(data)
    catalog = {"revision": "probe", "sources": {"archives": [{"path": "conformance/resource.yaml", "sha256": hashlib.sha256(b"resource").hexdigest(), "size": 8}], "unified": []},
               "metadata": {"source_hashes": {"documentation": {"conformance/doc.json": hashlib.sha256(b"document").hexdigest()}}}}
    (conformance / "viewer-catalog.json").write_text(json.dumps(catalog))
    (conformance / "viewer-unified-catalog.json").write_text(json.dumps(viewer_catalog.unified_projection(catalog)))
    monkeypatch.setattr(publish_viewer, "ROOT", root)
    monkeypatch.setattr(publish_viewer, "CATALOG", conformance / "viewer-catalog.json")
    source_path = publish_viewer.source_path
    monkeypatch.setattr(publish_viewer, "source_path", lambda relative: root / "conformance/readme.bin" if relative == "conformance/utils/README.md" else source_path(relative))
    copy = publish_viewer.copy_validated
    monkeypatch.setattr(publish_viewer, "copy_validated", lambda source, target, *args: None if target.suffix == ".md" else copy(source, target, *args))
    return root, catalog


def test_publication_copy_failure_preserves_existing_and_unrelated_output(tmp_path, monkeypatch):
    _private_publication(tmp_path, monkeypatch)
    site = tmp_path / "site"
    site.mkdir()
    (site / "index.html").write_bytes(b"old index")
    (site / "unrelated.json").write_bytes(b"keep")
    copy = publish_viewer.shutil.copyfile
    def fail(source, destination, **kwargs):
        if Path(source).name == "resource.yaml":
            raise OSError("injected resource copy failure")
        return copy(source, destination, **kwargs)
    monkeypatch.setattr(publish_viewer.shutil, "copyfile", fail)
    with pytest.raises(OSError, match="injected"):
        publish_viewer.publish(site, "a" * 40)
    assert (site / "index.html").read_bytes() == b"old index"
    assert (site / "unrelated.json").read_bytes() == b"keep"
    assert not (site / "conformance").exists()


def test_publication_checks_copied_resource_and_document_bytes(tmp_path, monkeypatch):
    root, _ = _private_publication(tmp_path, monkeypatch)
    for resource in ["conformance/resource.yaml", "conformance/doc.json"]:
        original = (root / resource).read_bytes()
        site = tmp_path / Path(resource).stem
        site.mkdir()
        (site / "keep.json").write_bytes(b"old")
        copy = publish_viewer.shutil.copyfile
        def mutate(source, destination, **kwargs):
            if Path(source) == root / "index.html":
                (root / resource).write_bytes(b"mutated")
            return copy(source, destination, **kwargs)
        with monkeypatch.context() as patch:
            patch.setattr(publish_viewer.shutil, "copyfile", mutate)
            with pytest.raises(ValueError, match="copied.*does not match catalog"):
                publish_viewer.publish(site, "a" * 40)
        assert list(site.iterdir()) == [site / "keep.json"]
        assert (site / "keep.json").read_bytes() == b"old"
        (root / resource).write_bytes(original)


def test_publication_holds_source_generation_through_all_copies(tmp_path, monkeypatch):
    root, _ = _private_publication(tmp_path, monkeypatch)
    copying = threading.Event()
    release_copy = threading.Event()
    writer_entered = threading.Event()
    copy = publish_viewer.shutil.copyfile
    def hold(source, target, **kwargs):
        if Path(source) == root / "index.html":
            copying.set()
            assert release_copy.wait(5)
        return copy(source, target, **kwargs)
    monkeypatch.setattr(publish_viewer.shutil, "copyfile", hold)
    errors = []
    def publish():
        try:
            publish_viewer.publish(tmp_path / "site", "a" * 40)
        except Exception as error:
            errors.append(error)
    def writer():
        with viewer_catalog.catalog_generation(root):
            writer_entered.set()
    publisher = threading.Thread(target=publish)
    mutation = threading.Thread(target=writer)
    publisher.start()
    assert copying.wait(5)
    mutation.start()
    assert not writer_entered.wait(0.2)
    release_copy.set()
    publisher.join(5)
    mutation.join(5)
    assert not publisher.is_alive() and not mutation.is_alive()
    assert errors == [] and writer_entered.is_set()
    assert (tmp_path / "site/conformance/resource.yaml").read_bytes() == b"resource"


@pytest.mark.parametrize("tree", ["site/conformance/viewer/app.js", "site/conformance/resource.yaml", "source/conformance/viewer/link.json", "source/conformance/utils/src/assets/link"])
def test_publication_rejects_source_and_destination_links_before_writes(tmp_path, monkeypatch, tree):
    _private_publication(tmp_path, monkeypatch)
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"untouched")
    target = tmp_path / tree
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(outside)
    site = tmp_path / "site"
    with pytest.raises(ValueError, match="non-regular|non-directory"):
        publish_viewer.publish(site, "a" * 40)
    assert outside.read_bytes() == b"untouched"
    assert target.is_symlink()
    if tree.startswith("site/"):
        assert not (site / "index.html").exists()
    else:
        assert not site.exists()


def test_publication_cli_rejects_link_and_ancestor_link(tmp_path, monkeypatch):
    _private_publication(tmp_path, monkeypatch)
    target = tmp_path / "existing"
    target.mkdir()
    (target / "keep.json").write_bytes(b"keep")
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    for output in [link, link / "child"]:
        monkeypatch.setattr(sys, "argv", ["publish_viewer.py", "--output", str(output), "--revision", "a" * 40])
        with pytest.raises(ValueError, match="symlink"):
            publish_viewer.main()
    assert list(target.iterdir()) == [target / "keep.json"]


def test_publication_preserves_unknown_files_and_supports_nested_site(tmp_path, monkeypatch):
    root, _ = _private_publication(tmp_path, monkeypatch)
    site = root / "site"
    site.mkdir()
    (site / "unrelated.json").write_bytes(b"keep")
    publish_viewer.publish(site, "a" * 40)
    assert (site / "unrelated.json").read_bytes() == b"keep"
    assert (site / "index.html").read_bytes() == b"index"
    with pytest.raises(ValueError, match="overlaps"):
        publish_viewer.publish(root, "a" * 40)
    with pytest.raises(ValueError, match="overlaps"):
        publish_viewer.publish(root / "conformance/viewer", "a" * 40)


@pytest.mark.parametrize("location", ["source/conformance/viewer/linkdir", "site/unrelated-linkdir"])
def test_publication_rejects_directory_links_preserving_unknown_target(tmp_path, monkeypatch, location):
    _private_publication(tmp_path, monkeypatch)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.json").write_bytes(b"keep")
    link = tmp_path / location
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="non-directory"):
        publish_viewer.publish(tmp_path / "site", "a" * 40)
    assert link.is_symlink()
    assert list(outside.iterdir()) == [outside / "keep.json"]
    assert (outside / "keep.json").read_bytes() == b"keep"


def test_publication_commit_failure_restores_old_site(tmp_path, monkeypatch):
    _private_publication(tmp_path, monkeypatch)
    site = tmp_path / "site"
    site.mkdir()
    (site / "index.html").write_bytes(b"old")
    (site / "unrelated.json").write_bytes(b"keep")
    replace = unified_history.os.replace
    def fail(source, destination):
        if Path(source).name == "site" and Path(source).parent.name.startswith(".viewer-site-stage-"):
            raise OSError("injected site commit failure")
        return replace(source, destination)
    monkeypatch.setattr(unified_history.os, "replace", fail)
    with pytest.raises(OSError, match="injected site commit"):
        publish_viewer.publish(site, "a" * 40)
    assert (site / "index.html").read_bytes() == b"old"
    assert (site / "unrelated.json").read_bytes() == b"keep"
    assert len(list(site.iterdir())) == 2
