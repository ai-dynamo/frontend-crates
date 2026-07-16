#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Extract conformance fixtures from the in-repo LFS shard store into the local cache.

Shard tarballs live in git at conformance/fixtures/ (tracked via git-lfs; see
.gitattributes). The manifest (conformance/fixtures-manifest.json) pins the
active snapshot and the sha256 of every shard. No network access: extraction
reads the checked-out shard files directly.

Cache location (fixed, same contract as the old HF downloader):
  ${XDG_CACHE_HOME:-~/.cache}/dynamo/conformance-fixtures/

Usage:
  # extract (or verify cache is current) and print the snapshot dir
  python3 conformance/utils/src/extract_fixtures.py

  # force re-extraction ignoring existing cache
  python3 conformance/utils/src/extract_fixtures.py --full-refresh

  # show plan without extracting anything
  python3 conformance/utils/src/extract_fixtures.py --dry-run

  # show manifest info and local cache state
  python3 conformance/utils/src/extract_fixtures.py --info
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
from pathlib import Path

# conformance/utils/src/extract_fixtures.py -> repo root: 4 .parent calls
ROOT = Path(__file__).resolve().parent.parent.parent.parent
MANIFEST_PATH = ROOT / "conformance" / "fixtures-manifest.json"
FIXTURES_DIR = ROOT / "conformance" / "fixtures"

LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def get_cache_root():
    xdg = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    return Path(xdg) / "dynamo" / "conformance-fixtures"


def read_state(snap_dir):
    state_file = snap_dir / ".fixtures-state.json"
    if state_file.exists():
        try:
            return json.loads(state_file.read_text())
        except Exception:
            return {}
    return {}


def write_state(snap_dir, snapshot, shards):
    state = {
        "snapshot": snapshot,
        "shards": {s["path"]: s["sha256"] for s in shards},
    }
    tmp = snap_dir / ".fixtures-state.json.tmp"
    tmp.write_text(json.dumps(state, indent=2) + "\n")
    tmp.rename(snap_dir / ".fixtures-state.json")


def list_cached_snapshots(cache_root):
    """Return cached snapshot dirs sorted newest-first (timestamp sort = lexicographic sort)."""
    if not cache_root.exists():
        return []
    return sorted(
        [d for d in cache_root.iterdir() if d.is_dir() and (d / ".fixtures-state.json").exists()],
        key=lambda d: d.name,
        reverse=True,
    )


def shard_file(shard):
    """Resolve a manifest shard entry to its checked-out file, failing actionably.

    A git-lfs pointer file (checkout without `git lfs pull`) is the common
    failure: the file exists but holds ~130 bytes of pointer text, not the
    tarball.
    """
    path = FIXTURES_DIR / shard["path"]
    if not path.exists():
        sys.exit(
            f"Shard missing: {path}\n"
            "The fixture store is part of the git checkout. If this is a fresh\n"
            "clone, ensure git-lfs is installed and run: git lfs pull"
        )
    with open(path, "rb") as f:
        head = f.read(len(LFS_POINTER_PREFIX))
    if head == LFS_POINTER_PREFIX:
        sys.exit(
            f"{path} is a git-lfs pointer, not the shard itself.\n"
            "Install git-lfs and fetch the objects: git lfs install && git lfs pull"
        )
    actual = sha256_file(path)
    if actual != shard["sha256"]:
        sys.exit(
            f"SHA256 mismatch for {shard['path']}: expected {shard['sha256'][:12]}…, "
            f"got {actual[:12]}…\n"
            "The checked-out shard does not match the manifest pin. Re-run\n"
            "package_fixtures.py (which rewrites both together) or restore the file."
        )
    return path


def update_symlinks(cache_root, snap_dir, verbose=False):
    """Create/retarget relative symlinks cache_root/{toolcalling,reasoning} -> snap_dir/..."""
    for name in ("toolcalling", "reasoning"):
        src = snap_dir / name
        if not src.exists():
            continue
        link = cache_root / name
        # Relative target: <snapshot>/<name>  (e.g. 20260707_215709/toolcalling)
        target = Path(snap_dir.name) / name
        tmp = cache_root / f".{name}.tmp"
        if tmp.is_symlink() or tmp.exists():
            tmp.unlink()
        tmp.symlink_to(target)
        tmp.rename(link)
        if verbose:
            print(f"  [symlink] {link} -> {target}", file=sys.stderr)


def extract_tarball(tarball_path, dest_dir, verbose=False):
    dest_dir.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"  [extract] {tarball_path.name} -> {dest_dir}", file=sys.stderr)
    with tarfile.open(str(tarball_path), "r:gz") as tf:
        # filter="data" (PEP 706) rejects absolute paths, "..", links pointing
        # outside the destination, and device entries — shard tarballs come
        # from PR-controlled files, so never trust member paths.
        tf.extractall(str(dest_dir), filter="data")


def show_info(manifest, cache_root):
    pin = manifest["snapshot"]
    print(f"Snapshot: {pin}")
    print(f"Created:  {manifest.get('created_pt', 'unknown')}")
    print(f"Store:    {FIXTURES_DIR}")
    if manifest.get("crates"):
        print(f"Crates:   {', '.join(f'{k}={v}' for k, v in manifest['crates'].items())}")
    if manifest.get("peers"):
        print(f"Peers:    {', '.join(f'{k}={v}' for k, v in manifest['peers'].items())}")
    shards = manifest.get("shards", [])
    total_shard_bytes = sum(s.get("size", 0) for s in shards)
    print(f"Shards:   {len(shards)}  ({total_shard_bytes:,} B total)")

    cached = list_cached_snapshots(cache_root)
    if cached:
        print(f"\nCached snapshots in {cache_root}:")
        for d in cached:
            marker = "  <- current pin" if d.name == pin else ""
            print(f"  {d.name}{marker}")
    else:
        print(f"\nNo cached snapshots in {cache_root}")


def main():
    ap = argparse.ArgumentParser(
        description="Extract conformance fixtures from the in-repo LFS shard store"
    )
    ap.add_argument("--full-refresh", action="store_true", help="Ignore existing cache, re-extract all")
    ap.add_argument("--dry-run", action="store_true", help="Show plan without extracting")
    ap.add_argument("--info", action="store_true", help="Show manifest info and cache state, then exit")
    ap.add_argument("-v", "--verbose", action="store_true", help="Print per-shard details")
    args = ap.parse_args()

    if not MANIFEST_PATH.exists():
        sys.exit(
            f"Manifest not found: {MANIFEST_PATH}\n"
            "Run package_fixtures.py to create one, then commit it."
        )

    manifest = json.loads(MANIFEST_PATH.read_text())
    pin = manifest["snapshot"]
    shards = manifest.get("shards", [])

    cache_root = get_cache_root()

    if args.info:
        show_info(manifest, cache_root)
        return

    snap_dir = cache_root / pin

    # Clean up an incomplete partial extraction (snap_dir present but no state marker)
    if snap_dir.exists() and not (snap_dir / ".fixtures-state.json").exists():
        print(f"  incomplete extraction at {snap_dir}, removing", file=sys.stderr)
        shutil.rmtree(str(snap_dir))

    state = read_state(snap_dir)

    pinned_shards = {s["path"]: s["sha256"] for s in shards}
    if (
        state
        and state.get("snapshot") == pin
        # Compare the full shard-hash map, not just the stamp: a shard can be
        # re-pinned IN PLACE under an unchanged snapshot (it happened to
        # fixtures-batch-on-stream-v2), and a stamp-only check would keep
        # serving the stale extracted tree forever.
        and state.get("shards") == pinned_shards
        and not args.full_refresh
    ):
        # Retarget the stable symlinks even on a hit: switching between two
        # already-cached snapshots (e.g. a pin rollback) must repoint
        # toolcalling/ + reasoning/ or readers keep using the other snapshot.
        update_symlinks(cache_root, snap_dir, verbose=args.verbose)
        print(f"Cache hit: {snap_dir}", file=sys.stderr)
        print(snap_dir)
        return

    if args.dry_run:
        for s in shards:
            print(f"  [dry-run] would extract {s['path']}", file=sys.stderr)
        print(f"[dry-run] shards: {len(shards)}", file=sys.stderr)
        print(snap_dir)
        return

    if snap_dir.exists():
        # Reaching here means the cache-hit check failed for this dir (forced
        # refresh, or the manifest's shard pins moved under the same stamp) —
        # the extraction is stale either way.
        print(f"  stale extraction at {snap_dir}, re-extracting", file=sys.stderr)
        shutil.rmtree(str(snap_dir))

    print(f"Extracting {len(shards)} shard(s) into {snap_dir}", file=sys.stderr)
    for s in shards:
        extract_tarball(shard_file(s), snap_dir, verbose=args.verbose)
    write_state(snap_dir, pin, shards)
    update_symlinks(cache_root, snap_dir, verbose=args.verbose)

    print(snap_dir)


if __name__ == "__main__":
    main()
