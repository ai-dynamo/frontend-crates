#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Copy the maintained browser viewer and its active source data to a site tree."""
from __future__ import annotations

import argparse
import json
import hashlib
import re
import shutil
import tempfile
from contextlib import nullcontext

import unified_history
from viewer_catalog import catalog_generation
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CATALOG = ROOT / "conformance" / "viewer-catalog.json"


def source_path(relative: str) -> Path:
    if not isinstance(relative, str) or not relative or relative.startswith("/") or ":" in relative or "\\" in relative or any(part in ("", ".", "..") for part in relative.split("/")):
        raise ValueError("publication source must be a repository-relative path")
    source = (ROOT / relative).resolve()
    if not source.is_relative_to(ROOT.resolve()) or not source.is_file():
        raise ValueError(f"publication source is outside the checkout or missing: {relative}")
    return source


def copy_validated(source: Path, target: Path, expected_hash: str | None = None, size: int | None = None) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    if expected_hash is not None:
        data = target.read_bytes()
        if (size is not None and len(data) != size) or hashlib.sha256(data).hexdigest() != expected_hash:
            raise ValueError(f"copied publication resource/document does not match catalog: {source}")


def _copy_site(destination: Path, revision: str, catalog: dict) -> None:
    for relative in ("index.html", "conformance/CONFORMANCE_v2.html", "conformance/viewer-catalog.json"):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path(relative), target)
    shutil.copytree(ROOT / "conformance/viewer", destination / "conformance/viewer", dirs_exist_ok=True)
    shutil.copytree(ROOT / "conformance/utils/src/assets", destination / "conformance/utils/src/assets", dirs_exist_ok=True)
    (destination / "conformance/viewer-publication.json").write_text(json.dumps({"schema": 1, "revision": revision, "catalog_revision": catalog["revision"]}, sort_keys=True) + "\n")
    for resource in (*catalog["sources"]["archives"], *catalog["sources"]["unified"]):
        copy_validated(source_path(resource["path"]), destination / resource["path"], resource["sha256"], resource["size"])
    for relative in ("conformance/utils/README.md", *catalog["metadata"]["source_hashes"]["documentation"]):
        copy_validated(source_path(relative), destination / relative, catalog["metadata"]["source_hashes"]["documentation"].get(relative))


def publish(destination: Path, revision: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("publication revision must be a full immutable commit SHA")
    destination = destination.absolute()
    if any(path.is_symlink() for path in [destination, *destination.parents]):
        raise ValueError("publication destination cannot be a symlink")
    destination = destination.resolve()
    root = ROOT.resolve()
    if root.is_relative_to(destination):
        raise ValueError("publication destination overlaps the source checkout")
    with catalog_generation(root):
        catalog = json.loads(CATALOG.read_text())
        source_roots = [ROOT / "conformance/viewer", ROOT / "conformance/utils/src/assets"]
        relatives = ["index.html", "conformance/CONFORMANCE_v2.html", "conformance/viewer-catalog.json", "conformance/utils/README.md",
                     *catalog["metadata"]["source_hashes"]["documentation"],
                     *(resource["path"] for resource in (*catalog["sources"]["archives"], *catalog["sources"]["unified"]))]
        if any(source.resolve().is_relative_to(destination) or destination.is_relative_to(source.resolve()) for source in [*source_roots, *(ROOT / relative for relative in relatives)]):
            raise ValueError("publication destination overlaps maintained source paths")
        for source in source_roots:
            unified_history._fsync_candidate_payload(source)
        for resource in (*catalog["sources"]["archives"], *catalog["sources"]["unified"]):
            data = source_path(resource["path"]).read_bytes()
            if len(data) != resource["size"] or hashlib.sha256(data).hexdigest() != resource["sha256"]:
                raise ValueError(f"publication resource does not match catalog: {resource['path']}")
        for relative, expected_hash in catalog["metadata"]["source_hashes"]["documentation"].items():
            if hashlib.sha256(source_path(relative).read_bytes()).hexdigest() != expected_hash:
                raise ValueError(f"publication document does not match catalog: {relative}")
        source_path("conformance/utils/README.md")
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Both paths use the existing directory lease; avoid reacquiring it when
        # an allowed site directory shares the source generation's lock parent.
        output_lease = nullcontext() if destination.parent == root / "conformance" else unified_history._store_mutation_lock(destination)
        with output_lease:
            if destination.exists():
                unified_history._fsync_candidate_payload(destination)
            with tempfile.TemporaryDirectory(prefix=".viewer-site-stage-", dir=destination.parent) as temporary:
                candidate = Path(temporary) / "site"
                if destination.exists():
                    shutil.copytree(destination, candidate, symlinks=True)
                else:
                    candidate.mkdir()
                _copy_site(candidate, revision, catalog)
                unified_history.publish_paths_transactionally([(candidate, destination)], backup_parent=destination.parent)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="site output directory")
    parser.add_argument("--revision", required=True, help="full immutable publication commit SHA")
    args = parser.parse_args()
    publish(args.output, args.revision)
    print(f"published browser viewer to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
