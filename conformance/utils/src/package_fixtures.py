#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Publish Unified capture YAML and its manifest transactionally."""

import argparse
import datetime
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from zoneinfo import ZoneInfo

import capture_sources
import fixture_disposition
import generation_workers
import publish_current_capture
import unified_history
import viewer_catalog

ROOT = Path(__file__).resolve().parents[3]
MANIFEST_REL = Path("conformance/fixtures-manifest.json")
UNIFIED_HISTORY_DIR = ROOT / "conformance/fixtures-unified-v2"

def read_versions():
    """Read crate versions from Cargo.toml files and peer versions from pyproject.stub.toml."""
    crates = {}
    for crate_name, cargo_path in [
        ("dynamo-parsers", ROOT / "parsers" / "v1" / "Cargo.toml"),
        ("dynamo-parsers-v2", ROOT / "parsers" / "v2" / "Cargo.toml"),
    ]:
        if cargo_path.exists():
            m = re.search(r'^version\s*=\s*"([^"]+)"', cargo_path.read_text(), re.MULTILINE)
            if m:
                crates[crate_name] = m.group(1)

    peers = {}
    pyproject = ROOT / "conformance" / "utils" / "src" / "pyproject.stub.toml"
    if pyproject.exists():
        text = pyproject.read_text()
        for pkg in ["vllm", "sglang"]:
            # Match vllm[extras]==X.Y.Z or sglang[extras]==X.Y.Z
            m = re.search(rf'{pkg}(?:\[[^\]]*\])?==([\d][^">,\s]*)', text)
            if m:
                peers[pkg] = m.group(1)
    return crates, peers


def stage_fixtures(conformance_root, tmpdir):
    source = conformance_root / "unified"
    if source.is_dir():
        shutil.copytree(source, tmpdir / "unified")


def build_shards(tmpdir, *, history_root=None,
                 excluded_unified_capture_dirs=frozenset(), required_unified_capture_dirs=None):
    history_root = Path(history_root or UNIFIED_HISTORY_DIR)
    if not history_root.is_dir():
        raise ValueError(f"Unified YAML history is missing: {history_root}")
    capture_root = tmpdir / "unified"
    complete_snapshot = (
        (capture_root / "inputs").is_dir()
        or (capture_root / "golden").is_dir()
    )
    required_capture_dirs = frozenset()
    if complete_snapshot and required_unified_capture_dirs is None:
        required_capture_dirs = frozenset(
            path.name
            for path in capture_root.iterdir()
            if path.is_dir()
            and path.name.startswith("dynamo_v2-")
            and "+pr" not in path.name
        )
    elif complete_snapshot:
        required_capture_dirs = frozenset(required_unified_capture_dirs)
    # Keep explicitly excluded versions out of the completeness set; the
    # history updater separately verifies that each exclusion already exists.
    required_capture_dirs -= excluded_unified_capture_dirs
    update_options = {
        "complete_snapshot": complete_snapshot,
        "required_capture_dirs": required_capture_dirs,
    }
    if excluded_unified_capture_dirs:
        update_options["excluded_capture_dirs"] = excluded_unified_capture_dirs
    changed = unified_history.update_store_from_loose(
        history_root, capture_root, **update_options
    )
    for path in changed:
        display_path = Path(fixture_disposition.UNIFIED_HISTORY_PATH) / path.relative_to(
            history_root
        )
        print(f"  updated {display_path}")
    source_root = history_root
    digest, size = unified_history.store_digest(source_root)
    return [{
            "path": fixture_disposition.UNIFIED_HISTORY_PATH,
            "format": "unified-history",
            "sha256": digest,
            "size": size,
        }]


def _validate_candidate_package(manifest, history_dir, *, generation=None):
    shards = fixture_disposition.active_shards(manifest)
    if len(shards) != 1 or shards[0].get("format") != "unified-history":
        raise ValueError("manifest must contain only Unified history")
    if generation is not None:
        generation.validated_store(history_dir)
    else:
        unified_history.load_store(history_dir)
    digest, size = unified_history.store_digest(history_dir)
    if digest != shards[0]["sha256"] or size != shards[0]["size"]:
        raise ValueError("candidate Unified history differs from manifest")


def package_snapshot(stamp, created_pt, crates, peers, *, dry_run, prune=False,
                     excluded_unified_capture_dirs=frozenset(),
                     required_unified_capture_dirs=None):
    if prune:
        raise ValueError("Unified packaging cannot prune historical evidence")
    conformance_root = ROOT / "conformance"
    manifest_path = ROOT / MANIFEST_REL
    with tempfile.TemporaryDirectory(prefix=".dyn-fixtures-stage-", dir=conformance_root) as temporary:
        transaction_root = Path(temporary)
        loose_root = transaction_root / "loose"
        loose_root.mkdir()
        candidate_history = transaction_root / "fixtures-unified-v2"
        candidate_manifest = transaction_root / "fixtures-manifest.json"
        with viewer_catalog.catalog_generation(ROOT) as generation:
            stage_fixtures(conformance_root, loose_root)
            shutil.copytree(UNIFIED_HISTORY_DIR, candidate_history)
            shards = build_shards(loose_root, history_root=candidate_history,
                                  excluded_unified_capture_dirs=excluded_unified_capture_dirs,
                                  required_unified_capture_dirs=required_unified_capture_dirs)
            manifest = {"snapshot": stamp, "created_pt": created_pt, "crates": crates,
                        "peers": peers, "shards": shards, "inactive_shards": []}
            if manifest_path.is_file():
                previous = json.loads(manifest_path.read_text())
                if all(previous.get(key) == manifest[key] for key in ("crates", "peers", "shards", "inactive_shards")):
                    manifest = previous
            candidate_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
            _validate_candidate_package(manifest, candidate_history, generation=generation)
            if dry_run:
                print(f"[dry-run] validated candidate manifest for: {manifest_path}")
                return
            candidate_catalog = transaction_root / "viewer-catalog.json"
            viewer_catalog.refresh_catalog(ROOT, destination=candidate_catalog,
                                           unified=candidate_history, manifest=candidate_manifest,
                                           generation=generation, unified_only=True)
            generation.validate_inventories()
            unified_history.publish_paths_transactionally([
                (candidate_history, UNIFIED_HISTORY_DIR),
                (candidate_manifest, manifest_path),
                (candidate_catalog, conformance_root / "viewer-catalog.json"),
                (transaction_root / "viewer-unified-catalog.json", conformance_root / "viewer-unified-catalog.json"),
            ], backup_parent=conformance_root)
    print(f"Manifest written: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--unified-only", action="store_true", help="Unified is the only supported corpus")
    parser.add_argument("--history-source", type=Path, default=os.environ.get("CONFORMANCE_HISTORY_SOURCE"))
    parser.add_argument("--exclude-unified-capture-dir", action="append", default=[])
    parser.add_argument("--require-unified-capture-dir", action="append")
    parser.add_argument("--allow-partial-unified-captures", action="store_true")
    generation_workers.add_worker_argument(parser)
    args = parser.parse_args()
    if args.allow_partial_unified_captures and args.require_unified_capture_dir is not None:
        parser.error("--allow-partial-unified-captures cannot be combined with --require-unified-capture-dir")
    if capture_sources.is_current(ROOT):
        if args.dry_run:
            parser.error("latest capture publication does not support --dry-run")
        history = args.history_source or ROOT / capture_sources.HISTORIC
        print(json.dumps(publish_current_capture.publish_loose(ROOT, Path(history).resolve(), ROOT / "conformance/unified")))
        return
    now = datetime.datetime.now(tz=ZoneInfo("America/Los_Angeles"))
    stamp = args.snapshot or now.strftime("%Y%m%d_%H%M%S")
    created_pt = now.strftime("%Y-%m-%d %H:%M:%S America/Los_Angeles")
    crates, peers = read_versions()
    with generation_workers.worker_policy(args.workers):
        package_snapshot(stamp, created_pt, crates, peers, dry_run=args.dry_run,
                         excluded_unified_capture_dirs=frozenset(args.exclude_unified_capture_dir),
                         required_unified_capture_dirs=(frozenset(args.require_unified_capture_dir)
                             if args.require_unified_capture_dir is not None else frozenset()
                             if args.allow_partial_unified_captures else None))


if __name__ == "__main__":
    main()
