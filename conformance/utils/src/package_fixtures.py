#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Package conformance fixtures into per-version shard tarballs inside the repo's
LFS store (conformance/fixtures/) and write conformance/fixtures-manifest.json.
Publishing a snapshot = committing both to git (the shards are LFS-tracked via
.gitattributes); no external service is involved.

Shard layout (relative to conformance/fixtures/):
  toolcalling/fixtures-batch-v1/inputs.tar.gz
  toolcalling/fixtures-batch-v1/<impl>-<ver>.tar.gz   (one per immediate subdir)
  toolcalling/fixtures-stream-v2/inputs.tar.gz
  toolcalling/fixtures-stream-v2/<impl>-<ver>.tar.gz
  toolcalling/fixtures-batch-on-stream-v2.tar.gz      (whole tree as one tarball)
  reasoning/fixtures-v1/inputs.tar.gz

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
import tarfile
import tempfile
from pathlib import Path

# conformance/utils/src/ -> repo root: 4 .parent calls (strip filename, then 3 dirs)
ROOT = Path(__file__).resolve().parent.parent.parent.parent
MANIFEST_REL = Path("conformance") / "fixtures-manifest.json"
FIXTURES_DIR = ROOT / "conformance" / "fixtures"

# Fixture trees that get one shard tarball per immediate subdir. Unified follows the
# same convention: inputs/ + golden/ + <impl>-<version>/ each become a versioned shard
# (unified/inputs.tar.gz, unified/golden.tar.gz, unified/vllm_python-0.25.1.tar.gz, ...).
PER_SUBDIR_TREES = [
    "toolcalling/fixtures-batch-v1",
    "toolcalling/fixtures-stream-v2",
    "reasoning/fixtures-v1",
    "unified",
]
# Fixture trees bundled as a single tarball (whole tree, no per-version sharding)
# Tuple: (source rel-path in conformance/, shard path in the store)
WHOLE_TREE_SHARDS = [
    (
        "toolcalling/fixtures-batch-on-stream-v2",
        "toolcalling/fixtures-batch-on-stream-v2.tar.gz",
    ),
]


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


def _tar_dir(src_abs, arcname, out_path):
    """Create a deterministic gzip tarball (mtime=0, uid/gid=0) for reproducible sha256."""
    import gzip as _gzip

    def _normalize(ti):
        ti.mtime = 0
        ti.uid = 0
        ti.gid = 0
        ti.uname = ""
        ti.gname = ""
        return ti

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with _gzip.GzipFile(str(out_path), "wb", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w") as tf:
            tf.add(str(src_abs), arcname=str(arcname), filter=_normalize)
    return sha256_file(out_path), out_path.stat().st_size


def stage_fixtures(conformance_root, tmpdir):
    """Copy all fixture trees into tmpdir, preserving the relative layout."""
    all_trees = list(PER_SUBDIR_TREES) + [src for src, _ in WHOLE_TREE_SHARDS]
    for tree_rel in all_trees:
        src = conformance_root / tree_rel
        dst = tmpdir / tree_rel
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(str(src), str(dst))
        else:
            print(f"  warn: {tree_rel} not found, skipping", file=sys.stderr)


def _extracted_snapshot_dir():
    """The current extracted snapshot in the fixture cache, or None. Used to
    protect whole-tree shards from partial local capture trees."""
    xdg = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    cache_root = Path(xdg) / "dynamo" / "conformance-fixtures"
    manifest_path = ROOT / MANIFEST_REL
    if not manifest_path.exists():
        return None
    snap = json.loads(manifest_path.read_text()).get("snapshot")
    d = cache_root / snap if snap else None
    return d if d and d.is_dir() else None


def build_shards(tmpdir, blobs_dir, prune=False):
    """Build per-version shard tarballs and whole-tree shards. Returns list of shard dicts."""
    shards = []

    for tree_rel in PER_SUBDIR_TREES:
        tree_abs = tmpdir / tree_rel
        if not tree_abs.exists():
            continue
        for subdir in sorted(d for d in tree_abs.iterdir() if d.is_dir()):
            # Only the documented layout becomes a shard: inputs/ or
            # <impl>-<version>/. Anything else (a stray family dir, an
            # overlays/ nest from a raw capture) would produce a tarball the
            # resolvers ignore — reject it loudly instead.
            if subdir.name not in ("inputs", "golden") and not re.match(r"^[a-z0-9_]+-\d", subdir.name):
                print(
                    f"  warn: skipping {tree_rel}/{subdir.name} — not inputs/ or "
                    "<impl>-<version>/ (normalize the capture output first)",
                    file=sys.stderr,
                )
                continue
            rel = f"{tree_rel}/{subdir.name}"
            shard_path = rel + ".tar.gz"
            out = blobs_dir / shard_path
            sha, size = _tar_dir(tmpdir / rel, rel, out)
            shards.append({"path": shard_path, "sha256": sha, "size": size})
            print(f"  {shard_path:<60s} {size:>9,} B  {sha[:12]}…")

    for src_rel, shard_path in WHOLE_TREE_SHARDS:
        src_abs = tmpdir / src_rel
        if not src_abs.exists():
            continue
        if not prune:
            # A whole-tree shard is rebuilt from whatever local tree exists, so
            # a partial capture (one family re-recorded) would silently DROP
            # every uncaptured family from the stored shard. Merge families
            # that exist in the current extracted snapshot but not locally;
            # --prune opts into exact mirroring.
            snap = _extracted_snapshot_dir()
            prior = (snap / src_rel) if snap else None
            if prior and prior.is_dir():
                for fam in sorted(prior.iterdir()):
                    if fam.is_dir() and not (src_abs / fam.name).exists():
                        shutil.copytree(str(fam), str(src_abs / fam.name))
                        print(f"  merged {src_rel}/{fam.name} from extracted snapshot (absent locally)")
            elif prior is None:
                print(f"  warn: no extracted snapshot to verify {src_rel} completeness", file=sys.stderr)
        out = blobs_dir / shard_path
        sha, size = _tar_dir(src_abs, src_rel, out)
        shards.append({"path": shard_path, "sha256": sha, "size": size})
        print(f"  {shard_path:<60s} {size:>9,} B  {sha[:12]}…")

    return shards


def sync_store(blobs_dir, shards, dry_run, prune):
    """Copy built shards into conformance/fixtures/.

    Store files not in the new shard set are KEPT unless --prune is passed:
    the local capture trees are often partial (one family recaptured, the rest
    absent), and mirroring a partial tree would silently drop shards. Capture
    versions are additive by design — a re-record ADDS a version subdir, so
    its shard joins the set; pruning is only for deliberately retired trees.
    """
    new_paths = {s["path"] for s in shards}
    stale = [
        p
        for p in FIXTURES_DIR.rglob("*.tar.gz")
        if str(p.relative_to(FIXTURES_DIR)) not in new_paths
    ]
    if dry_run:
        print(f"  [dry-run] would write {len(shards)} shard(s) to {FIXTURES_DIR}")
        for p in stale:
            verb = "remove stale" if prune else "keep (not in this package run)"
            print(f"  [dry-run] would {verb} {p.relative_to(FIXTURES_DIR)}")
        return
    for s in shards:
        src = blobs_dir / s["path"]
        dst = FIXTURES_DIR / s["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))
    for p in stale:
        if prune:
            print(f"  removing stale {p.relative_to(FIXTURES_DIR)}")
            p.unlink()
        else:
            print(f"  keeping {p.relative_to(FIXTURES_DIR)} (not in this package run; --prune removes)")


def merge_shards(built, prune):
    """Final manifest shard set: built shards, plus prior-manifest entries whose
    store file was kept (partial capture trees update only their own shards).
    With --prune the built set stands alone."""
    if prune:
        return built
    built_paths = {s["path"] for s in built}
    merged = list(built)
    manifest_path = ROOT / MANIFEST_REL
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text()).get("shards", [])
        for s in prior:
            fp = FIXTURES_DIR / s["path"]
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


def main():
    ap = argparse.ArgumentParser(
        description="Package conformance fixtures into the in-repo LFS store"
    )
    ap.add_argument("--snapshot", default=None, help="Snapshot stamp override (YYYYMMDD_HHMMSS)")
    ap.add_argument("--dry-run", action="store_true", help="Build tarballs but don't touch the store")
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

    conformance_root = ROOT / "conformance"

    with tempfile.TemporaryDirectory(prefix="dyn-fixtures-stage-") as _tmpdir:
        tmpdir = Path(_tmpdir)
        blobs_dir = tmpdir / "_blobs"
        blobs_dir.mkdir()

        print("\nStaging fixture trees…")
        stage_fixtures(conformance_root, tmpdir)

        print("\nBuilding shards…")
        shards = build_shards(tmpdir, blobs_dir, args.prune)

        print(f"\nSyncing store: {FIXTURES_DIR}")
        sync_store(blobs_dir, shards, args.dry_run, args.prune)

        manifest = {
            "snapshot": stamp,
            "created_pt": created_pt,
            "crates": crates,
            "peers": peers,
            "shards": merge_shards(shards, args.prune),
        }

        manifest_path = ROOT / MANIFEST_REL
        if args.dry_run:
            print(f"\n[dry-run] would write manifest: {manifest_path}")
        else:
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            print(f"\nManifest written: {manifest_path}")
            print("\nNext: commit the store + manifest to pin this snapshot:")
            print("  git add conformance/fixtures conformance/fixtures-manifest.json")
            print(f'  git commit -s -m "fixtures: snapshot {stamp}"')


if __name__ == "__main__":
    main()
