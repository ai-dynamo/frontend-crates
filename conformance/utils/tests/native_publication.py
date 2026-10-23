# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Invoke the native publisher; synthetic catalogs belong only to test roots."""
import copy
import gzip
import hashlib
import json
import os
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import migrate_unified_snapshots
import unified_archives
import viewer_catalog
from verify_publication_transport import Decoder, read_index

ROOT = Path(__file__).resolve().parents[3]


@lru_cache(maxsize=1)
def publisher_binary():
    configured = os.environ.get("CONFORMANCE_PUBLISHER")
    if configured:
        binary = Path(configured).resolve(strict=True)
    else:
        subprocess.run(["cargo", "build", "--locked", "--release", "-p", "conformance-publisher"], cwd=ROOT, check=True, capture_output=True)
        binary = Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target")) / "release/conformance-publish"
    return binary


def publish(root, destination, *, workers="auto", extra=(), check=True):
    return subprocess.run([str(publisher_binary()), "--root", str(root), "--output", str(destination), "--revision", "a" * 40,
                           "--workers", str(workers), *extra], cwd=ROOT, capture_output=True, text=True, check=check, timeout=4500)


def fixture_catalog(root, catalog):
    if root.resolve() == ROOT:
        raise ValueError("synthetic catalogs cannot replace maintained catalogs")
    store = root / "conformance/fixtures-unified-v2"
    if not unified_archives.paths(store):
        # Historical test builders remain sparse; publication fixtures use the permanent format.
        with tempfile.TemporaryDirectory(prefix="native-fixture-", dir=root.parent) as temporary:
            sparse = Path(temporary) / "sparse"
            store.rename(sparse)
            migrate_unified_snapshots.convert(sparse, store, Path(temporary))
    catalog = copy.deepcopy(catalog)
    catalog.setdefault("schema", 1)
    catalog.setdefault("snapshot", "fixture")
    catalog["sources"].setdefault("archives", [])
    catalog["sources"].setdefault("candidates", [])
    catalog["sources"]["unified"] = viewer_catalog._unified_records(root)
    catalog.setdefault("metadata", {}).setdefault("tabs", [])
    catalog["metadata"].setdefault("source_hashes", {})
    catalog.pop("revision", None)
    catalog["revision"] = hashlib.sha256(json.dumps(catalog, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    conformance = root / "conformance"
    (root / "index.html").write_bytes(b"<head></head><body></body>")
    (conformance / "CONFORMANCE_v2.html").write_bytes(b'<a href="../">Viewer</a>')
    (conformance / "fixtures-manifest.json").write_text('{"shards":[]}\n')
    (conformance / "viewer-catalog.json").write_text(json.dumps(catalog))
    (conformance / "viewer-unified-catalog.json").write_text(json.dumps(viewer_catalog.unified_projection(catalog)))
    return catalog


def native_snapshots(root, destination, catalog, *, workers="auto"):
    root, destination = Path(root), Path(destination)
    if root.resolve() != ROOT:
        fixture_catalog(root, catalog)
    publish(root, destination, workers=workers)
    return proof_catalog(root, destination)


def runtime_catalog(destination):
    return json.loads(gzip.decompress((Path(destination) / "conformance/viewer-resolved-catalog.json.gz").read_bytes()))


def proof_catalog(root, destination):
    catalog = runtime_catalog(destination)
    source_catalog = json.loads((Path(root) / "conformance/viewer-catalog.json").read_bytes())
    catalog["proof_original_unified"] = source_catalog["sources"]["unified"]
    return catalog


def snapshot_bytes(destination, resource, decoder=None):
    decoder = decoder or Decoder(Path(destination).resolve())
    recipe = read_index(Path(destination).resolve(), resource["recipe_index"])[resource["path"]]
    return decoder.reconstruct(recipe)
