# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native snapshot output and browser decoding share the publication contract."""
import hashlib
import subprocess
from pathlib import Path

from native_publication import native_snapshots, runtime_catalog, snapshot_bytes
from verify_publication_transport import read_index
from test_unified_history import _store

ROOT = Path(__file__).resolve().parents[3]


def test_snapshot_generation_resolves_history_and_explicit_empty_scope(tmp_path):
    root = tmp_path / "source"
    _store(root / "conformance/fixtures-unified-v2")
    catalog = {"sources": {"candidates": [{"mode": "unified", "implementation": "dynamo_v2", "version": version,
                                           **({"capture_families": []} if version == "0.5.3" else {})}
                                          for version in ("0.5.0", "0.5.2", "0.5.3")]},
               "metadata": {"unified_capture_policy": {"dynamo_v2": {"mode": "stream"}}}}
    site = tmp_path / "site"
    published = native_snapshots(root, site, catalog)
    resources = published["sources"]["unified"]
    assert len(resources) == 3
    assert not any(resource["version"] == "0.5.3" for resource in resources)
    for resource in resources:
        data = snapshot_bytes(site, resource)
        assert len(data) == resource["size"]
        assert hashlib.sha256(data).hexdigest() == resource["sha256"]
        decoded = data
        assert decoded.endswith(b"\n") and not decoded.endswith(b"\n\n")
        assert b"sourceBytes" not in decoded
    assert not (site / "conformance/viewer-startup.json").exists()


def test_snapshot_and_startup_integrity_negative_controls():
    for name in ("snapshots.test.js", "startup.test.js", "originals.test.js", "source_archives.test.js"):
        subprocess.run(["node", str(ROOT / "conformance/viewer" / name)], cwd=ROOT, capture_output=True, check=True)


def test_startup_publication_is_content_addressed_and_default_scoped(published_viewer):
    site = published_viewer
    catalog = runtime_catalog(site)
    assert catalog["transport"]["schema"] == 2
    for path in ("viewer-catalog.json", "viewer-unified-catalog.json", "viewer-resolved-catalog.json", "viewer-startup.json", "viewer-original-resources.json.gz"):
        assert not (site / "conformance" / path).exists()
    reference = catalog["metadata"]["selection_policy"]["unified"]["reference"]
    candidates = [candidate for candidate in catalog["sources"]["candidates"] if candidate["implementation"] == reference]
    selected = max(candidates, key=lambda candidate: tuple(int(part) for part in candidate["version"].split(".")))
    families = selected["native_unified_families" if reference == "dynamo_v2" else "capture_families"]
    defaults = [resource for resource in catalog["sources"]["unified"] if (resource["kind"] == "inputs-and-golden" and resource["family"] in families) or (resource["implementation"] == reference and resource["version"] == selected["version"])]
    default_packs = {}
    for resource in defaults:
        recipe = read_index(site, resource["recipe_index"])[resource["path"]]
        for chunk in recipe["chunks"]:
            pack = chunk["pack"]
            entry = default_packs.setdefault(pack["path"], {"size": pack["decoded_size"], "ranges": set()})
            entry["ranges"].add((chunk["offset"], chunk["size"]))
    assert default_packs
    for pack in default_packs.values():
        end = 0
        for offset, length in sorted(pack["ranges"]):
            assert offset == end, "default pack contains bytes outside default-view dependencies"
            end += length
        assert end == pack["size"]
