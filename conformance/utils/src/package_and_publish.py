#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Package conformance fixtures into per-version shard tarballs + all-<stamp>.tar.gz monolith,
upload blobs to ai-dynamo/conformance-fixtures on HuggingFace, and write
conformance/fixtures-manifest.json into the working tree for the maintainer to commit.

Shard layout (relative to HF repo root):
  all-<stamp>.tar.gz
  toolcalling/fixtures-batch-v1/inputs.tar.gz
  toolcalling/fixtures-batch-v1/<impl>-<ver>.tar.gz   (one per immediate subdir)
  toolcalling/fixtures-stream-v2/inputs.tar.gz
  toolcalling/fixtures-stream-v2/<impl>-<ver>.tar.gz
  toolcalling/fixtures-batch-on-stream-v2.tar.gz      (whole tree as one tarball)
  reasoning/fixtures-v1/inputs.tar.gz

Usage:
  export HF_TOKEN=<your-write-token>
  python3 package_and_publish.py [--dry-run] [--snapshot YYYYMMDD_HHMMSS]
  python3 package_and_publish.py --cleanup-old   # delete old loose files from HF repo

HF_TOKEN must be set to a write-capable token before running. The script does not
search personal token caches — callers are responsible for exporting the right token.
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
DEFAULT_REPO = "ai-dynamo/conformance-fixtures"

# Fixture trees that get one shard tarball per immediate subdir
PER_SUBDIR_TREES = [
    "toolcalling/fixtures-batch-v1",
    "toolcalling/fixtures-stream-v2",
    "reasoning/fixtures-v1",
]
# Fixture trees bundled as a single tarball (whole tree, no per-version sharding)
# Tuple: (source rel-path in conformance/, HF blob path)
WHOLE_TREE_SHARDS = [
    (
        "toolcalling/fixtures-batch-on-stream-v2",
        "toolcalling/fixtures-batch-on-stream-v2.tar.gz",
    ),
]


def find_token(cli_token=None):
    """Resolve token from --token flag or HF_TOKEN env only. No personal cache fallback."""
    if cli_token:
        return cli_token
    return os.environ.get("HF_TOKEN") or None


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


def build_shards(tmpdir, blobs_dir):
    """Build per-version shard tarballs and whole-tree shards. Returns list of shard dicts."""
    shards = []

    for tree_rel in PER_SUBDIR_TREES:
        tree_abs = tmpdir / tree_rel
        if not tree_abs.exists():
            continue
        for subdir in sorted(d for d in tree_abs.iterdir() if d.is_dir()):
            rel = f"{tree_rel}/{subdir.name}"
            hf_path = rel + ".tar.gz"
            out = blobs_dir / hf_path
            sha, size = _tar_dir(tmpdir / rel, rel, out)
            shards.append({"path": hf_path, "sha256": sha, "size": size})
            print(f"  {hf_path:<60s} {size:>9,} B  {sha[:12]}…")

    for src_rel, hf_path in WHOLE_TREE_SHARDS:
        src_abs = tmpdir / src_rel
        if not src_abs.exists():
            continue
        out = blobs_dir / hf_path
        sha, size = _tar_dir(src_abs, src_rel, out)
        shards.append({"path": hf_path, "sha256": sha, "size": size})
        print(f"  {hf_path:<60s} {size:>9,} B  {sha[:12]}…")

    return shards


def build_all_tarball(tmpdir, stamp, blobs_dir):
    """Build all-<stamp>.tar.gz from the same staged tree used for shards (deterministic)."""
    import gzip as _gzip

    def _normalize(ti):
        ti.mtime = 0
        ti.uid = 0
        ti.gid = 0
        ti.uname = ""
        ti.gname = ""
        return ti

    all_name = f"all-{stamp}.tar.gz"
    out = blobs_dir / all_name
    with _gzip.GzipFile(str(out), "wb", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w") as tf:
            for subdir in ["toolcalling", "reasoning"]:
                src = tmpdir / subdir
                if src.exists():
                    tf.add(str(src), arcname=subdir, filter=_normalize)
    sha = sha256_file(out)
    size = out.stat().st_size
    print(f"  {all_name:<60s} {size:>9,} B  {sha[:12]}…")
    return all_name, sha, size


def upload_blobs(token, repo_id, blobs_dir, commit_msg, dry_run):
    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("huggingface_hub is not installed. Run: pip install huggingface_hub")

    blobs = sorted(blobs_dir.rglob("*.tar.gz"))
    total_bytes = sum(f.stat().st_size for f in blobs)
    print(f"  {len(blobs)} blobs, {total_bytes:,} B total")

    if dry_run:
        print("  [dry-run] skipping upload (blobs built successfully)")
        return

    # HF's LFS batch-preupload endpoint returns 403 for multi-file commits on org
    # repos when the files are new LFS objects. upload_file (one commit per blob)
    # uses a single-file preupload path that consistently works. ~13 commits is fine
    # for a dumb blob store (LFS dedup prevents re-upload of unchanged shards).
    api = HfApi(token=token)
    for i, blob in enumerate(blobs, 1):
        path_in_repo = str(blob.relative_to(blobs_dir))
        is_last = i == len(blobs)
        msg = commit_msg if is_last else f"fixtures: upload {path_in_repo}"
        url = api.upload_file(
            path_or_fileobj=str(blob),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=msg,
        )
        print(f"  [{i}/{len(blobs)}] {path_in_repo}")
    print(f"  Upload complete.")


def cleanup_old_loose(token, repo_id, dry_run):
    """Delete non-tarball files from the HF repo (clears the old loose YAML mirror)."""
    try:
        from huggingface_hub import HfApi, CommitOperationDelete
    except ImportError:
        sys.exit("huggingface_hub is not installed.")

    api = HfApi(token=token)
    entries = list(api.list_repo_tree(repo_id=repo_id, repo_type="dataset", recursive=True))
    keep = {".gitattributes", "README.md"}
    to_delete = [
        CommitOperationDelete(path_in_repo=e.path)
        for e in entries
        if hasattr(e, "size")  # RepoFile (not RepoFolder)
        and not e.path.endswith(".tar.gz")
        and e.path not in keep
    ]

    if not to_delete:
        print("  No old loose files found.")
        return

    print(f"  {len(to_delete)} old files to delete")
    for e in to_delete[:8]:
        print(f"    {e.path_in_repo}")
    if len(to_delete) > 8:
        print(f"    … and {len(to_delete) - 8} more")

    if dry_run:
        print("  [dry-run] skipping delete")
        return

    api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=to_delete,
        commit_message="fixtures: remove old loose files, replaced by tarball layout",
    )
    print(f"  Deleted {len(to_delete)} old loose files.")


def main():
    ap = argparse.ArgumentParser(
        description="Package and publish conformance fixtures to HuggingFace"
    )
    ap.add_argument("--snapshot", default=None, help="Snapshot stamp override (YYYYMMDD_HHMMSS)")
    ap.add_argument("--repo", default=DEFAULT_REPO, help="HF dataset repo ID")
    ap.add_argument("--token", default=None, help="HuggingFace token (overrides env/cache)")
    ap.add_argument("--dry-run", action="store_true", help="Build tarballs but skip upload")
    ap.add_argument(
        "--cleanup-old",
        action="store_true",
        help="Delete old non-tarball files from HF repo after publishing",
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

    token = find_token(args.token)
    if not token and not args.dry_run:
        sys.exit(
            "HF_TOKEN is not set. Export a write-capable token before publishing:\n"
            "  export HF_TOKEN=<your-write-token>\n"
            "  python3 package_and_publish.py ..."
        )
    if token and not args.dry_run:
        try:
            from huggingface_hub import HfApi as _HfApi
            _role = _HfApi(token=token).whoami().get("auth", {}).get("accessToken", {}).get("role", "")
            if _role and _role != "write":
                sys.exit(
                    f"HF_TOKEN has role='{_role}' — a write token is required for publish."
                )
        except Exception:
            pass  # whoami failure is non-fatal; upload will fail with a clear 403 if needed

    conformance_root = ROOT / "conformance"

    with tempfile.TemporaryDirectory(prefix="dyn-fixtures-stage-") as _tmpdir:
        tmpdir = Path(_tmpdir)
        blobs_dir = tmpdir / "_blobs"
        blobs_dir.mkdir()

        print("\nStaging fixture trees…")
        stage_fixtures(conformance_root, tmpdir)

        print("\nBuilding shards…")
        shards = build_shards(tmpdir, blobs_dir)

        print("\nBuilding monolith…")
        all_name, all_sha, all_size = build_all_tarball(tmpdir, stamp, blobs_dir)

        print(f"\nUploading to {args.repo}…")
        upload_blobs(token, args.repo, blobs_dir, f"fixtures: snapshot {stamp}", args.dry_run)

        if args.cleanup_old:
            print(f"\nCleaning up old loose files in {args.repo}…")
            cleanup_old_loose(token, args.repo, args.dry_run)

        manifest = {
            "snapshot": stamp,
            "created_pt": created_pt,
            "hf_repo": args.repo,
            "all_tarball": all_name,
            "all_sha256": all_sha,
            "crates": crates,
            "peers": peers,
            "shards": shards,
        }

        manifest_path = ROOT / MANIFEST_REL
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"\nManifest written: {manifest_path}")
        if not args.dry_run:
            print("\nNext: commit the manifest to pin this snapshot:")
            print(f"  git add {MANIFEST_REL}")
            print(f"  git commit -m 'fixtures: snapshot {stamp}'")


if __name__ == "__main__":
    main()
