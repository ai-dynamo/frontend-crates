#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Download conformance fixtures from HuggingFace using the in-git manifest as the pin.

The manifest (conformance/fixtures-manifest.json) is read from the git checkout —
0 HuggingFace metadata calls. HF is contacted only to fetch blobs (resolver bucket):

  Cache hit  (snapshot dir already extracted): 0 resolver calls
  Cold       (no cached snapshot):             1 resolver call  (all-<pin>.tar.gz)
  Warm       (older snapshot cached):          1 per changed shard (usually 0-1)

Cache location (fixed):
  ${XDG_CACHE_HOME:-~/.cache}/dynamo/conformance-fixtures/

Usage:
  # download (or verify cache is current) and print the snapshot dir
  python3 conformance/utils/src/download_fixtures.py

  # force re-download ignoring existing cache
  python3 conformance/utils/src/download_fixtures.py --full-refresh

  # show plan without fetching anything
  python3 conformance/utils/src/download_fixtures.py --dry-run

  # show manifest info and local cache state
  python3 conformance/utils/src/download_fixtures.py --info

Token resolution:
  --token flag > HF_TOKEN env > ~/.cache/huggingface/token.write > ~/.cache/huggingface/token
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
from pathlib import Path

# conformance/utils/src/download_fixtures.py -> repo root: 4 .parent calls
ROOT = Path(__file__).resolve().parent.parent.parent.parent
MANIFEST_PATH = ROOT / "conformance" / "fixtures-manifest.json"
DEFAULT_REPO = "ai-dynamo/conformance-fixtures"


def find_token(cli_token=None):
    if cli_token:
        return cli_token
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    for path in [
        os.path.expanduser("~/.cache/huggingface/token.write"),
        os.path.expanduser("~/.cache/huggingface/token"),
    ]:
        if os.path.exists(path):
            content = open(path).read().strip()
            if content:
                return content
    return None


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


def download_blob(token, repo_id, filename, verbose=False):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("huggingface_hub is not installed. Run: pip install huggingface_hub")
    if verbose:
        print(f"  [resolver] {filename}", file=sys.stderr)
    local = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=filename,
        token=token,
        force_download=False,
    )
    return Path(local)


def verify_sha256(path, expected, label=""):
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"SHA256 mismatch{' for ' + label if label else ''}: "
            f"expected {expected[:12]}…, got {actual[:12]}…"
        )


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
        tf.extractall(str(dest_dir))


def show_info(manifest, cache_root):
    pin = manifest["snapshot"]
    print(f"Snapshot: {pin}")
    print(f"Created:  {manifest.get('created_pt', 'unknown')}")
    print(f"Repo:     {manifest.get('hf_repo', DEFAULT_REPO)}")
    if manifest.get("crates"):
        print(f"Crates:   {', '.join(f'{k}={v}' for k, v in manifest['crates'].items())}")
    if manifest.get("peers"):
        print(f"Peers:    {', '.join(f'{k}={v}' for k, v in manifest['peers'].items())}")
    shards = manifest.get("shards", [])
    total_shard_bytes = sum(s.get("size", 0) for s in shards)
    print(f"Shards:   {len(shards)}  ({total_shard_bytes:,} B total)")
    all_size = manifest.get("all_tarball_size") or "?"
    print(f"All:      {manifest['all_tarball']}")

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
        description="Download conformance fixtures from HuggingFace (manifest-driven)"
    )
    ap.add_argument("--repo", default=None, help="HF dataset repo ID (overrides manifest)")
    ap.add_argument("--token", default=None, help="HuggingFace token (overrides env/cache)")
    ap.add_argument("--full-refresh", action="store_true", help="Ignore existing cache, re-download all")
    ap.add_argument("--dry-run", action="store_true", help="Show plan without fetching")
    ap.add_argument("--info", action="store_true", help="Show manifest info and cache state, then exit")
    ap.add_argument("-v", "--verbose", action="store_true", help="Print per-resolver-call details")
    args = ap.parse_args()

    if not MANIFEST_PATH.exists():
        sys.exit(
            f"Manifest not found: {MANIFEST_PATH}\n"
            "Run package_and_publish.py to create one, then commit it."
        )

    manifest = json.loads(MANIFEST_PATH.read_text())
    pin = manifest["snapshot"]
    shards = manifest.get("shards", [])
    repo_id = args.repo or manifest.get("hf_repo", DEFAULT_REPO)
    all_tarball = manifest["all_tarball"]
    all_sha256 = manifest.get("all_sha256")

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
    resolver_calls = 0

    # ── Cache hit ──────────────────────────────────────────────────────────────
    if state and state.get("snapshot") == pin and not args.full_refresh:
        print(f"Cache hit: {snap_dir}", file=sys.stderr)
        print(f"Resolver calls: 0  (cache hit)", file=sys.stderr)
        print(snap_dir)
        return

    token = find_token(args.token)
    if not token:
        print(
            "Warning: no HuggingFace token found; anonymous downloads may be rate-limited.",
            file=sys.stderr,
        )

    cached = list_cached_snapshots(cache_root)

    # Fall back to cold if all shards would be stale (no usable warm base)
    warm_base = None
    if cached and not args.full_refresh:
        prev_dir = cached[0]
        prev_state = read_state(prev_dir)
        prev_hashes = prev_state.get("shards", {})
        stale = [s for s in shards if prev_hashes.get(s["path"]) != s["sha256"]]
        if len(stale) < len(shards) or not shards:
            warm_base = (prev_dir, prev_state, stale)

    # ── Cold path ──────────────────────────────────────────────────────────────
    if warm_base is None:
        print(f"Cold: downloading {all_tarball}", file=sys.stderr)
        if args.dry_run:
            print(f"[dry-run] would download {all_tarball} from {repo_id}", file=sys.stderr)
            print(f"[dry-run] resolver calls: 1", file=sys.stderr)
            print(snap_dir)
            return
        local = download_blob(token, repo_id, all_tarball, verbose=args.verbose)
        resolver_calls += 1
        if all_sha256:
            verify_sha256(local, all_sha256, label=all_tarball)
        extract_tarball(local, snap_dir, verbose=args.verbose)
        write_state(snap_dir, pin, shards)
        update_symlinks(cache_root, snap_dir, verbose=args.verbose)
        print(f"Resolver calls: {resolver_calls}  (cold: 1 monolith)", file=sys.stderr)

    # ── Warm path ──────────────────────────────────────────────────────────────
    else:
        prev_dir, _prev_state, stale = warm_base
        print(
            f"Warm: base={prev_dir.name}  changed={len(stale)}/{len(shards)} shards",
            file=sys.stderr,
        )

        if args.dry_run:
            for s in stale:
                print(f"  [dry-run] would download shard {s['path']}", file=sys.stderr)
            print(f"[dry-run] resolver calls: {len(stale)}", file=sys.stderr)
            print(snap_dir)
            return

        # Copy previous extraction tree to new snapshot dir
        snap_dir.mkdir(parents=True, exist_ok=True)
        for item in prev_dir.iterdir():
            if item.name == ".fixtures-state.json":
                continue  # written fresh below
            dst = snap_dir / item.name
            if item.is_dir():
                shutil.copytree(str(item), str(dst))
            else:
                shutil.copy2(str(item), str(dst))

        # Download and extract only changed shards
        for s in stale:
            local = download_blob(token, repo_id, s["path"], verbose=args.verbose)
            resolver_calls += 1
            verify_sha256(local, s["sha256"], label=s["path"])
            extract_tarball(local, snap_dir, verbose=args.verbose)

        write_state(snap_dir, pin, shards)
        update_symlinks(cache_root, snap_dir, verbose=args.verbose)
        print(
            f"Resolver calls: {resolver_calls}  (warm: {len(stale)} shard(s))",
            file=sys.stderr,
        )

    print(snap_dir)


if __name__ == "__main__":
    main()
