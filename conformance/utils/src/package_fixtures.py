#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Update the reviewable legacy and Unified YAML stores from loose captures, then
write the manifest that pins both sources. No external service is involved.

Usage:
  python3 package_fixtures.py [--dry-run] [--snapshot YYYYMMDD_HHMMSS]

Source trees are the loose capture outputs in conformance/{toolcalling,reasoning}/
(written by capture.sh / capture_driver.py; not committed to git).
"""

import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

import capture_policy
import fixture_disposition
import legacy_history
import unified_history

# conformance/utils/src/ -> repo root: 4 .parent calls (strip filename, then 3 dirs)
ROOT = Path(__file__).resolve().parent.parent.parent.parent
MANIFEST_REL = Path("conformance") / "fixtures-manifest.json"
FIXTURES_DIR = ROOT / "conformance" / "fixtures"
UNIFIED_HISTORY_DIR = ROOT / "conformance" / "fixtures-unified-v2"
LEGACY_HISTORY_DIR = ROOT / "conformance" / "fixtures-v1"

# Stage Unified loose captures before updating its YAML history store.
PER_SUBDIR_TREES = [
    "unified",
]
LEGACY_TREES = tuple(legacy_history.CORPORA.values())
NON_ARCHIVE_FORMATS = {"unified-history", legacy_history.HISTORY_FORMAT,
                       fixture_disposition.CAPTURE_POLICY_FORMAT}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


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
    """Copy all fixture trees into tmpdir, preserving the relative layout."""
    all_trees = list(PER_SUBDIR_TREES) + list(LEGACY_TREES)
    for tree_rel in all_trees:
        src = conformance_root / tree_rel
        dst = tmpdir / tree_rel
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(str(src), str(dst))
        else:
            print(f"  warn: {tree_rel} not found, skipping", file=sys.stderr)


def build_shards(
    tmpdir,
    blobs_dir,
    prune=False,
    *,
    history_root=None,
    legacy_root=None,
):
    """Update the YAML stores and return their pinned manifest entries."""
    history_root = Path(history_root or UNIFIED_HISTORY_DIR)
    if not history_root.is_dir():
        raise ValueError(f"Unified YAML history is missing: {history_root}")
    capture_root = tmpdir / "unified"
    complete_snapshot = (capture_root / "inputs").is_dir() or (capture_root / "golden").is_dir()
    required_capture_dirs = frozenset()
    if complete_snapshot:
        required_capture_dirs = frozenset(
            path.name for path in capture_root.iterdir()
            if path.is_dir() and path.name.startswith("dynamo_v2-") and "+pr" not in path.name
        )
        if required_capture_dirs:
            # Recorded checkpoints retain the families/cases measured then;
            # only new captures must satisfy the current corpus requirements.
            recorded = {
                capture
                for history in unified_history.load_store(history_root).histories.values()
                for capture in history.captures
            }
            required_capture_dirs -= recorded
    changed = unified_history.update_store_from_loose(
        history_root,
        capture_root,
        complete_snapshot=complete_snapshot,
        required_capture_dirs=required_capture_dirs,
    )
    for path in changed:
        display_path = Path(fixture_disposition.UNIFIED_HISTORY_PATH) / path.relative_to(history_root)
        print(f"  updated {display_path}")
    digest, size = unified_history.store_digest(history_root)
    shards = [{
        "path": fixture_disposition.UNIFIED_HISTORY_PATH,
        "format": "unified-history",
        "sha256": digest,
        "size": size,
    }]
    print(f"  {fixture_disposition.UNIFIED_HISTORY_PATH:<60s} {size:>9,} B  {digest[:12]}…")

    legacy_root = Path(legacy_root or LEGACY_HISTORY_DIR)
    for changed_path in legacy_history.split_batch_on_stream_store(legacy_root):
        print(f"  updated {changed_path.relative_to(legacy_root)}")
    for changed_path in legacy_history.update_from_loose(legacy_root, tmpdir, prune=prune):
        print(f"  updated {changed_path.relative_to(legacy_root)}")
    digest, size = legacy_history.store_digest(legacy_root)
    shards.append({
        "path": legacy_history.HISTORY_PATH,
        "format": legacy_history.HISTORY_FORMAT,
        "sha256": digest,
        "size": size,
    })
    print(f"  {legacy_history.HISTORY_PATH:<60s} {size:>9,} B  {digest[:12]}…")

    unique = {}
    for shard in shards:
        if shard["path"] in unique and unique[shard["path"]] != shard:
            raise ValueError(f"conflicting staged capture layer: {shard['path']}")
        unique[shard["path"]] = shard
    return list(unique.values())


def preserved_evidence(*, manifest_path=None, fixtures_dir=None):
    manifest_path = Path(manifest_path or ROOT / MANIFEST_REL)
    fixtures_dir = Path(fixtures_dir or FIXTURES_DIR)
    if not manifest_path.exists():
        return {}
    manifest = json.loads(manifest_path.read_text())
    fixture_disposition.active_shards(manifest)
    return fixture_disposition.verify_inactive_shards(manifest, fixtures_dir)


def sync_store(
    blobs_dir,
    shards,
    dry_run,
    prune,
    *,
    fixtures_dir=None,
    manifest_path=None,
):
    """Copy built shards into conformance/fixtures/.

    Store files not in the new shard set are KEPT unless --prune is passed:
    the local capture trees are often partial (one family recaptured, the rest
    absent), and mirroring a partial tree would silently drop shards. Capture
    versions are additive by design — a re-record ADDS a version subdir, so
    its shard joins the set; pruning is only for deliberately retired trees.
    """
    fixtures_dir = Path(fixtures_dir or FIXTURES_DIR)
    new_paths = {s["path"] for s in shards}
    inactive = preserved_evidence(manifest_path=manifest_path, fixtures_dir=fixtures_dir)
    if new_paths & inactive.keys():
        raise ValueError(f"cannot overwrite inactive evidence: {sorted(new_paths & inactive.keys())}")
    # Historical archive fixtures remain immutable when packaging older manifests.
    # Unified captures use the YAML store and need a new semantic version when changed.
    for shard in shards:
        if shard.get("format") in NON_ARCHIVE_FORMATS:
            continue
        destination = fixtures_dir / shard["path"]
        if re.match(r"^[a-z0-9_]+-\d", destination.name) and destination.exists():
            if sha256_file(destination) != shard["sha256"]:
                raise ValueError(f"versioned capture is immutable; use a new semantic version: {shard['path']}")
    stale = [
        p
        for p in fixtures_dir.rglob("*.tar.gz")
        if str(p.relative_to(fixtures_dir)) not in new_paths | inactive.keys()
    ]
    if dry_run:
        archive_shards = [shard for shard in shards if shard.get("format") not in NON_ARCHIVE_FORMATS]
        print(
            f"  [dry-run] would write {len(archive_shards)} archive shard(s) to {fixtures_dir} "
            f"and update {len(shards) - len(archive_shards)} history pin(s)"
        )
        for p in stale:
            verb = "remove stale" if prune else "keep (not in this package run)"
            print(f"  [dry-run] would {verb} {p.relative_to(fixtures_dir)}")
        return
    for s in shards:
        if s.get("format") in NON_ARCHIVE_FORMATS:
            continue
        src = blobs_dir / s["path"]
        dst = fixtures_dir / s["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.unlink(missing_ok=True)
        shutil.copy2(str(src), str(dst))
    for p in stale:
        if prune or any(shard.get("format") == legacy_history.HISTORY_FORMAT for shard in shards):
            print(f"  removing stale {p.relative_to(fixtures_dir)}")
            p.unlink()
            parent = p.parent
            while parent != fixtures_dir and not any(parent.iterdir()):
                parent.rmdir()
                parent = parent.parent
        else:
            print(f"  keeping {p.relative_to(fixtures_dir)} (not in this package run; --prune removes)")


def merge_shards(
    built,
    prune,
    *,
    fixtures_dir=None,
    history_dir=None,
    legacy_dir=None,
    manifest_path=None,
):
    """Final manifest shard set: built shards, plus prior-manifest entries whose
    store file was kept (partial capture trees update only their own shards).
    With --prune the built set stands alone."""
    fixtures_dir = Path(fixtures_dir or FIXTURES_DIR)
    history_dir = Path(history_dir or UNIFIED_HISTORY_DIR)
    legacy_dir = Path(legacy_dir or LEGACY_HISTORY_DIR)
    manifest_path = Path(manifest_path or ROOT / MANIFEST_REL)
    inactive = preserved_evidence(manifest_path=manifest_path, fixtures_dir=fixtures_dir)
    if any(shard["path"] in inactive for shard in built):
        raise ValueError("cannot activate inactive evidence")
    if prune:
        return built
    built_paths = {s["path"] for s in built}
    merged = list(built)
    if manifest_path.exists():
        prior = fixture_disposition.active_shards(json.loads(manifest_path.read_text()))
        for s in prior:
            if s["path"].endswith(".tar.gz") and legacy_history.HISTORY_PATH in built_paths:
                continue
            if s["path"].startswith("unified/") and unified_history.is_generated_oracle_directory(
                Path(s["path"]).name.removesuffix(".tar.gz")
            ):
                continue
            fp = fixtures_dir / s["path"]
            if s.get("format") == "unified-history":
                if s["path"] not in built_paths:
                    digest, size = unified_history.store_digest(history_dir)
                    merged.append({**s, "sha256": digest, "size": size})
                continue
            if s.get("format") == legacy_history.HISTORY_FORMAT:
                if s["path"] not in built_paths:
                    digest, size = legacy_history.store_digest(legacy_dir)
                    merged.append({**s, "sha256": digest, "size": size})
                continue
            if s.get("format") == fixture_disposition.CAPTURE_POLICY_FORMAT:
                if s["path"] not in built_paths:
                    merged.append(fixture_disposition.capture_policy_pin(
                        history_dir.parent / fixture_disposition.CAPTURE_POLICY_PATH))
                continue
            if s["path"] not in built_paths and fp.exists():
                # RECOMPUTE the sha/size from the on-disk file — never trust the prior
                # manifest's value. A kept shard's store file can change between runs
                # (git restore, a re-pin, a manual swap); copying the old sha would
                # publish a manifest that lies about the content and makes
                # extract_fixtures' sha-verify fail or serve stale data.
                merged.append(
                    {"path": s["path"], "sha256": sha256_file(fp), "size": fp.stat().st_size}
                )
    merged.sort(key=lambda s: s["path"])
    return merged


def _validate_candidate_package(manifest, fixtures_dir, history_dir, legacy_dir):
    shards = fixture_disposition.active_shards(manifest)
    fixture_disposition.verify_inactive_shards(manifest, fixtures_dir)
    unified_history.load_store(history_dir)
    for shard in shards:
        if shard.get("format") == "unified-history":
            digest, size = unified_history.store_digest(history_dir)
        elif shard.get("format") == legacy_history.HISTORY_FORMAT:
            digest, size = legacy_history.store_digest(legacy_dir)
        elif shard.get("format") == fixture_disposition.CAPTURE_POLICY_FORMAT:
            pin = fixture_disposition.capture_policy_pin(
                history_dir.parent / fixture_disposition.CAPTURE_POLICY_PATH)
            digest, size = pin["sha256"], pin["size"]
        else:
            path = fixtures_dir / shard["path"]
            if not path.is_file():
                raise FileNotFoundError(f"candidate package shard is missing: {shard['path']}")
            digest, size = sha256_file(path), path.stat().st_size
        if digest != shard["sha256"] or size != shard["size"]:
            raise ValueError(f"candidate package shard differs from manifest: {shard['path']}")
    if any(shard.get("format") == legacy_history.HISTORY_FORMAT for shard in shards):
        with tempfile.TemporaryDirectory(prefix="dyn-legacy-validate-") as temporary:
            legacy_history.materialize_store(legacy_dir, Path(temporary))
            if any(s.get("format") == fixture_disposition.CAPTURE_POLICY_FORMAT for s in shards):
                unified_history.materialize_store(history_dir, Path(temporary) / "unified")
                policy = capture_policy.load_policy(
                    history_dir.parent / fixture_disposition.CAPTURE_POLICY_PATH)
                capture_policy.validate_materialized_policy(policy, Path(temporary))


def package_snapshot(stamp, created_pt, crates, peers, *, dry_run, prune):
    conformance_root = ROOT / "conformance"
    manifest_path = ROOT / MANIFEST_REL
    with tempfile.TemporaryDirectory(
        prefix=".dyn-fixtures-stage-", dir=conformance_root
    ) as temporary:
        transaction_root = Path(temporary)
        loose_root = transaction_root / "loose"
        blobs_dir = transaction_root / "blobs"
        candidate_fixtures = transaction_root / "fixtures"
        candidate_history = transaction_root / "fixtures-unified-v2"
        candidate_legacy = transaction_root / "fixtures-v1"
        candidate_manifest = transaction_root / "fixtures-manifest.json"
        policy_path = conformance_root / fixture_disposition.CAPTURE_POLICY_PATH
        candidate_policy = transaction_root / fixture_disposition.CAPTURE_POLICY_PATH
        loose_root.mkdir()
        blobs_dir.mkdir()

        with unified_history._store_mutation_lock(UNIFIED_HISTORY_DIR):
            print("\nStaging fixture trees…")
            stage_fixtures(conformance_root, loose_root)
            shutil.copytree(FIXTURES_DIR, candidate_fixtures, copy_function=os.link)
            shutil.copytree(UNIFIED_HISTORY_DIR, candidate_history)
            shutil.copytree(LEGACY_HISTORY_DIR, candidate_legacy)
            if policy_path.is_file():
                shutil.copyfile(policy_path, candidate_policy)
            original_history_digest = unified_history.store_digest(candidate_history)
            original_legacy_digest = legacy_history.store_digest(candidate_legacy)

            print("\nBuilding shards…")
            shards = build_shards(
                loose_root,
                blobs_dir,
                prune,
                history_root=candidate_history,
                legacy_root=candidate_legacy,
            )
            if candidate_policy.is_file():
                shards.append(fixture_disposition.capture_policy_pin(candidate_policy))
            source_changed = (
                original_history_digest != unified_history.store_digest(candidate_history)
                or original_legacy_digest != legacy_history.store_digest(candidate_legacy)
            )

            print(f"\nStaging store candidate for: {FIXTURES_DIR}")
            sync_store(
                blobs_dir,
                shards,
                False,
                prune,
                fixtures_dir=candidate_fixtures,
                manifest_path=manifest_path,
            )

            inactive_shards = list(
                preserved_evidence(
                    manifest_path=manifest_path,
                    fixtures_dir=candidate_fixtures,
                ).values()
            )
            manifest = {
                "snapshot": stamp,
                "created_pt": created_pt,
                "crates": crates,
                "peers": peers,
                "shards": merge_shards(
                    shards,
                    prune,
                    fixtures_dir=candidate_fixtures,
                    history_dir=candidate_history,
                    legacy_dir=candidate_legacy,
                    manifest_path=manifest_path,
                ),
                "inactive_shards": inactive_shards,
            }
            if manifest_path.is_file():
                previous = json.loads(manifest_path.read_text())
                if not source_changed:
                    for key in ("snapshot", "created_pt", "crates", "peers"):
                        manifest[key] = previous.get(key, manifest[key])
                if (
                    previous.get("crates") == manifest["crates"]
                    and previous.get("peers") == manifest["peers"]
                    and previous.get("shards") == manifest["shards"]
                    and previous.get("inactive_shards") == manifest["inactive_shards"]
                ):
                    manifest = previous
            candidate_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
            _validate_candidate_package(manifest, candidate_fixtures, candidate_history, candidate_legacy)

            if dry_run:
                print(f"\n[dry-run] validated candidate manifest for: {manifest_path}")
                return

            publications = [
                (candidate_history, UNIFIED_HISTORY_DIR),
                (candidate_legacy, LEGACY_HISTORY_DIR),
                (candidate_fixtures, FIXTURES_DIR),
                (candidate_manifest, manifest_path),
            ]
            if candidate_policy.is_file():
                publications.insert(-1, (candidate_policy, policy_path))
            unified_history.publish_paths_transactionally(
                publications,
                backup_parent=conformance_root,
            )

    print(f"\nManifest written: {manifest_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Package conformance fixtures into the in-repo YAML stores"
    )
    ap.add_argument("--snapshot", default=None, help="Snapshot stamp override (YYYYMMDD_HHMMSS)")
    ap.add_argument("--dry-run", action="store_true", help="Validate without publishing the YAML stores")
    ap.add_argument(
        "--prune",
        action="store_true",
        help="Remove store shards (and manifest entries) not rebuilt by this run. "
        "Default keeps them: local capture trees are often partial.",
    )
    args = ap.parse_args()

    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        try:
            from backports.zoneinfo import ZoneInfo
        except ImportError:
            sys.exit("Python 3.9+ required for zoneinfo (or install backports.zoneinfo)")

    now_pt = datetime.datetime.now(tz=ZoneInfo("America/Los_Angeles"))
    if args.snapshot:
        stamp = args.snapshot
        created_pt = f"{stamp} (stamp override) America/Los_Angeles"
    else:
        stamp = now_pt.strftime("%Y%m%d_%H%M%S")
        created_pt = now_pt.strftime("%Y-%m-%d %H:%M:%S") + " America/Los_Angeles"

    print(f"Snapshot: {stamp}")

    crates, peers = read_versions()
    print(f"Crates:   {crates}")
    print(f"Peers:    {peers}")

    package_snapshot(
        stamp,
        created_pt,
        crates,
        peers,
        dry_run=args.dry_run,
        prune=args.prune,
    )


if __name__ == "__main__":
    main()
