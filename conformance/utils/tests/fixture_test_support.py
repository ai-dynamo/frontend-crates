# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Isolated checkout identity for synthetic fixture-extraction tests."""
from pathlib import Path
import copy
import json
import shutil
import subprocess
from types import SimpleNamespace

import capture_sources
import extract_fixtures
import unified_archives
import unified_history
import viewer_catalog
from native_publication import fixture_catalog


REPO = Path(__file__).resolve().parents[3]


def prepare_isolated_publication(root, history, view, monkeypatch, publisher, *, mode, metadata=None,
                                 retained_history_only=False):
    """Exercise both source modes against the same tiny, verified synthetic dataset."""
    tools = Path(capture_sources.__file__).parent
    for name in ("capture_sources.py", "ci_publication_cache.py", "yaml_fast.py", "unified_archives.py", "execution_manifests.py", "impls.py", "viewer_codec.py", "parser_families.yaml"):
        target = root / "conformance/utils/src" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(tools / name, target)
    (root / "index.html").write_bytes(b"<head></head><body></body>")
    (root / "conformance/CONFORMANCE_v2.html").write_bytes(b'<a href="../">Viewer</a>')
    (root / "conformance/fixtures-manifest.json").write_bytes(b'{"shards":[]}\n')
    subprocess.run(["git", "init", str(root)], check=True, capture_output=True)
    original_run = subprocess.run
    view.mkdir()

    def build_catalog(command, **kwargs):
        if any(str(value).endswith("viewer_catalog.py") for value in command):
            generation = json.loads((view / "conformance/capture-generation.json").read_bytes())
            if not unified_archives.paths(view / capture_sources.STORE):
                unified_archives.pack(view / capture_sources.STORE, unified_history.CAPTURE_DIRECTORY_RE,
                                      unified_history._capture_release_sort_key)
            versions = ["0.5.0", "0.5.2"] if mode == "history" else ["0.5.2"]
            manifest_path = view / "conformance/fixtures-manifest.json"
            manifest_bytes = manifest_path.read_bytes()
            manifest = json.loads(manifest_bytes)
            archives = viewer_catalog._archive_records(manifest, view)
            catalog = fixture_catalog(view, {"capture_generation": generation,
                "metadata": copy.deepcopy(metadata or {"unified_capture_policy": {"dynamo_v2": {"mode": "stream"}}}),
                "sources": {"archives": archives, "lineage": generation["lineage"], "candidates": [
                    {"mode": "unified", "implementation": "dynamo_v2", "version": version,
                     "capture_families": ["gemma4"], "native_unified_families": ["gemma4"],
                     "has_captured_results": True} for version in versions]}})
            if (view / capture_sources.DATASETS).is_file():
                _inventory, _documents, indexed = viewer_catalog._native_catalog_document_index(view / capture_sources.STORE)
                catalog["sources"]["history_datasets"] = viewer_catalog._history_dataset_sources(
                    view, catalog["sources"]["unified"], indexed, catalog["metadata"])
                (view / "conformance/viewer-catalog.json").write_bytes(capture_sources.encoded(catalog))
            manifest["shards"] = [row for row in manifest["shards"] if row.get("format") != "unified-history"]
            manifest_path.write_bytes(capture_sources.encoded(manifest))
            return SimpleNamespace(returncode=0)
        return original_run(command, **kwargs)

    with monkeypatch.context() as patch:
        # Supply synthetic catalog metadata; source discovery and admission stay real.
        patch.setattr(subprocess, "run", build_catalog)
        patch.delenv("CONFORMANCE_DOCUMENT_LOCK", raising=False)
        patch.setenv("CONFORMANCE_CONVERSION_CACHE", str(root / "build/conformance-cache"))
        capture_sources.prepare(root, view, include_history=mode == "history",
                                history_source=history if mode == "history" else None, publisher=str(publisher),
                                retained_history_only=retained_history_only)


def configure_extractor_root(root, monkeypatch, *, manifest_path=None):
    # ROOT selects current versus historical storage and supplies release metadata.
    # Partial path patches can accidentally materialize the real checkout instead.
    package = root / "parsers/v2/Cargo.toml"
    package.parent.mkdir(parents=True, exist_ok=True)
    package.write_bytes((REPO / "parsers/v2/Cargo.toml").read_bytes())
    conformance = root / "conformance"
    fixtures = conformance / "fixtures"
    history = conformance / "fixtures-unified-v2"
    fixtures.mkdir(parents=True, exist_ok=True)
    history.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(extract_fixtures, "ROOT", root)
    monkeypatch.setattr(extract_fixtures, "MANIFEST_PATH", manifest_path or conformance / "fixtures-manifest.json")
    monkeypatch.setattr(extract_fixtures, "FIXTURES_DIR", fixtures)
    monkeypatch.setattr(extract_fixtures, "HISTORY_DIR", history)
