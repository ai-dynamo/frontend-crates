#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Package stream captures as per-family YAML, retain historical archives, update
the Unified YAML store, and write the manifest that pins these sources.
Publishing a snapshot means committing the stores and the manifest to git;
no external service is involved.

Shard layout (relative to conformance/fixtures/):
  toolcalling/fixtures-batch-v1/inputs.tar.gz
  toolcalling/fixtures-batch-v1/<impl>-<ver>.tar.gz   (one per immediate subdir)
  toolcalling/fixtures-stream-v2/inputs.tar.gz
  toolcalling/fixtures-stream-v2/families/<family>/<impl>-<ver>.yaml
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

import extract_fixtures  # sibling script, same dir on sys.path (matches capture_driver's import pattern)
import fixture_disposition
import stream_history
import unified_history

# conformance/utils/src/ -> repo root: 4 .parent calls (strip filename, then 3 dirs)
ROOT = Path(__file__).resolve().parent.parent.parent.parent
MANIFEST_REL = Path("conformance") / "fixtures-manifest.json"
FIXTURES_DIR = ROOT / "conformance" / "fixtures"
UNIFIED_HISTORY_DIR = ROOT / "conformance" / "fixtures-unified-v2"

# Fixture trees that get one archive per immediate subdirectory. Unified is listed so
# its loose capture tree is staged, then is written to the separate YAML history store.
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
    protect whole-tree shards from partial local capture trees.

    Same directory-naming contract as `extract_fixtures.py`: cached
    extractions are keyed by `{pin}-{fixtures_identity(shards)}`, not bare
    `pin` (a shard set can be re-pinned in place under an unchanged pin) --
    reusing `fixtures_identity` here instead of a second, independent
    identity computation keeps the two scripts from silently drifting apart
    on what "the same content" means.

    Resolution itself routes through `extract_fixtures.resolve_current_generation`
    -- the same single owner `extract_fixtures.py`'s own cache-hit check
    uses -- instead of reconstructing the bare `{pin}-{fid}` path directly.
    A `--full-refresh` publishes later generations at `{pin}-{fid}.refreshN`
    without ever touching the original; a direct reconstruction here would
    keep resolving to the abandoned (possibly corrupted) original generation
    forever, even after a refresh fixed it.
    """
    manifest_path = ROOT / MANIFEST_REL
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text())
    snap = manifest.get("snapshot")
    shards = fixture_disposition.active_shards(manifest)
    if not snap or not shards:
        return None
    cache_root = extract_fixtures.get_cache_root()
    inactive = manifest.get("inactive_shards", [])
    fid = extract_fixtures.fixtures_identity(shards, inactive)
    pinned_shards = extract_fixtures.shard_hash_map(shards)
    d, _generation = extract_fixtures.resolve_current_generation(cache_root, snap, fid, pinned_shards, inactive)
    return d


def build_shards(
    tmpdir,
    blobs_dir,
    prune=False,
    dry_run=False,
    *,
    history_root=None,
):
    """Build capture shards and return their manifest entries."""
    shards = []
    inactive = preserved_evidence()

    for tree_rel in PER_SUBDIR_TREES:
        tree_abs = tmpdir / tree_rel
        if tree_rel == "unified":
            history_root = Path(history_root or UNIFIED_HISTORY_DIR)
            if not history_root.is_dir():
                raise ValueError(f"Unified YAML history is missing: {history_root}")
            if dry_run:
                dry_run_history_root = tmpdir / "_unified-history"
                shutil.copytree(history_root, dry_run_history_root)
                history_root = dry_run_history_root
            capture_root = tmpdir / tree_rel
            complete_snapshot = (
                (capture_root / "inputs").is_dir()
                or (capture_root / "golden").is_dir()
            )
            required_capture_dirs = frozenset()
            if complete_snapshot:
                required_capture_dirs = frozenset(
                    path.name
                    for path in capture_root.iterdir()
                    if path.is_dir()
                    and path.name.startswith("dynamo_v2-")
                    and "+pr" not in path.name
                )
            changed = unified_history.update_store_from_loose(
                history_root,
                capture_root,
                complete_snapshot=complete_snapshot,
                required_capture_dirs=required_capture_dirs,
            )
            for path in changed:
                display_path = Path(fixture_disposition.UNIFIED_HISTORY_PATH) / path.relative_to(
                    history_root
                )
                print(f"  updated {display_path}")
            source_root = history_root
            digest, size = unified_history.store_digest(source_root)
            shards.append(
                {
                    "path": fixture_disposition.UNIFIED_HISTORY_PATH,
                    "format": "unified-history",
                    "sha256": digest,
                    "size": size,
                }
            )
            print(f"  {fixture_disposition.UNIFIED_HISTORY_PATH:<60s} {size:>9,} B  {digest[:12]}…")
            continue
        if not tree_abs.exists():
            continue
        for subdir in sorted(d for d in tree_abs.iterdir() if d.is_dir()):
            # golden_spec/ is an authored Unified oracle build tree, not a v1 shard.
            if unified_history.is_generated_oracle_directory(subdir.name):
                continue
            # Only the documented layout becomes a shard: inputs/ or
            # <impl>-<version>/. Anything else (a stray family dir, an
            # overlays/ nest from a raw capture) would produce a tarball the
            # resolvers ignore — reject it loudly instead.
            shared_overlay = re.match(r"^(inputs|golden)\+pr\d+\.patch\d+$", subdir.name)
            if subdir.name not in ("inputs", "golden") and not shared_overlay and not re.match(r"^[a-z0-9_]+-\d", subdir.name):
                print(
                    f"  warn: skipping {tree_rel}/{subdir.name} — not inputs/ or "
                    "<impl>-<version>/ (normalize the capture output first)",
                    file=sys.stderr,
                )
                continue
            rel = f"{tree_rel}/{subdir.name}"
            shard_path = rel + ".tar.gz"
            if shard_path in inactive:
                print(f"  preserving inactive evidence {shard_path}; not rebuilding")
                continue
            if tree_rel == str(stream_history.TREE) and subdir.name not in ("inputs", "golden"):
                # Existing archives remain historical evidence, never a publication target.
                if (FIXTURES_DIR / shard_path).exists():
                    prior = _extracted_snapshot_dir()
                    if prior is None:
                        raise ValueError("extract fixtures before packaging historical stream captures")
                    for path in subdir.glob("*/*.yaml"):
                        previous = prior / tree_rel / subdir.name / path.relative_to(subdir)
                        if not previous.is_file() or unified_history.load_yaml(path) != unified_history.load_yaml(previous):
                            raise ValueError(f"historical stream capture changed; use a new semantic version: {path}")
                    continue
                prior = _extracted_snapshot_dir()
                for output in stream_history.package(
                    subdir, blobs_dir,
                    prior_root=prior / tree_rel if prior else None,
                ):
                    shards.append({
                        "path": str(output.relative_to(blobs_dir)),
                        "sha256": sha256_file(output),
                        "size": output.stat().st_size,
                    })
                continue
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
    # Versioned archive fixtures remain immutable. Unified captures are stored only
    # in the canonical YAML history and must use a new semantic version when changed.
    for shard in shards:
        if shard.get("format") == "unified-history":
            continue
        destination = fixtures_dir / shard["path"]
        if re.match(r"^[a-z0-9_]+-\d", destination.name) and destination.exists():
            if sha256_file(destination) != shard["sha256"]:
                raise ValueError(f"versioned capture is immutable; use a new semantic version: {shard['path']}")
    stale = [
        p
        for p in fixtures_dir.rglob("*.tar.gz")
        if str(p.relative_to(fixtures_dir)) not in new_paths | inactive.keys()
        and not str(p.relative_to(fixtures_dir)).startswith("unified/")
    ]
    if dry_run:
        archive_shards = [shard for shard in shards if shard.get("format") != "unified-history"]
        print(
            f"  [dry-run] would write {len(archive_shards)} archive shard(s) to {fixtures_dir} "
            f"and update {len(shards) - len(archive_shards)} history pin(s)"
        )
        for p in stale:
            verb = "remove stale" if prune else "keep (not in this package run)"
            print(f"  [dry-run] would {verb} {p.relative_to(fixtures_dir)}")
        return
    for s in shards:
        if s.get("format") == "unified-history":
            continue
        src = blobs_dir / s["path"]
        dst = fixtures_dir / s["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.unlink(missing_ok=True)
        shutil.copy2(str(src), str(dst))
    for p in stale:
        if prune:
            print(f"  removing stale {p.relative_to(fixtures_dir)}")
            p.unlink()
        else:
            print(f"  keeping {p.relative_to(fixtures_dir)} (not in this package run; --prune removes)")


def merge_shards(
    built,
    prune,
    *,
    fixtures_dir=None,
    history_dir=None,
    manifest_path=None,
):
    """Final manifest shard set: built shards, plus prior-manifest entries whose
    store file was kept (partial capture trees update only their own shards).
    With --prune the built set stands alone."""
    fixtures_dir = Path(fixtures_dir or FIXTURES_DIR)
    history_dir = Path(history_dir or UNIFIED_HISTORY_DIR)
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


def _validate_candidate_package(manifest, fixtures_dir, history_dir):
    shards = fixture_disposition.active_shards(manifest)
    inactive = fixture_disposition.verify_inactive_shards(manifest, fixtures_dir)
    store = unified_history.load_store(history_dir)
    for shard in shards:
        if shard.get("format") == "unified-history":
            digest, size = unified_history.store_digest(history_dir)
        else:
            path = fixtures_dir / shard["path"]
            if not path.is_file():
                raise FileNotFoundError(f"candidate package shard is missing: {shard['path']}")
            digest, size = sha256_file(path), path.stat().st_size
        if digest != shard["sha256"] or size != shard["size"]:
            raise ValueError(f"candidate package shard differs from manifest: {shard['path']}")


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
        candidate_manifest = transaction_root / "fixtures-manifest.json"
        loose_root.mkdir()
        blobs_dir.mkdir()

        with unified_history._store_mutation_lock(UNIFIED_HISTORY_DIR):
            print("\nStaging fixture trees…")
            stage_fixtures(conformance_root, loose_root)
            shutil.copytree(FIXTURES_DIR, candidate_fixtures, copy_function=os.link)
            shutil.copytree(UNIFIED_HISTORY_DIR, candidate_history)

            print("\nBuilding shards…")
            shards = build_shards(
                loose_root,
                blobs_dir,
                prune,
                history_root=candidate_history,
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
                    manifest_path=manifest_path,
                ),
                "inactive_shards": inactive_shards,
            }
            if manifest_path.is_file():
                previous = json.loads(manifest_path.read_text())
                if (
                    previous.get("crates") == manifest["crates"]
                    and previous.get("peers") == manifest["peers"]
                    and previous.get("shards") == manifest["shards"]
                    and previous.get("inactive_shards") == manifest["inactive_shards"]
                ):
                    manifest = previous
            candidate_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
            _validate_candidate_package(manifest, candidate_fixtures, candidate_history)

            if dry_run:
                print(f"\n[dry-run] validated candidate manifest for: {manifest_path}")
                return

            unified_history.publish_paths_transactionally(
                [
                    (candidate_history, UNIFIED_HISTORY_DIR),
                    (candidate_fixtures, FIXTURES_DIR),
                    (candidate_manifest, manifest_path),
                ],
                backup_parent=conformance_root,
            )

    print(f"\nManifest written: {manifest_path}")
    print("\nNext: commit the store + manifest to pin this snapshot:")
    print(
        "  git add conformance/fixtures conformance/fixtures-unified-v2 "
        "conformance/fixtures-manifest.json"
    )
    print(f'  git commit -s -m "fixtures: snapshot {stamp}"')


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
