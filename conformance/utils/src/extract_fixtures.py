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
import errno
import hashlib
import json
import os
import secrets
import shutil
import sys
import tarfile
from pathlib import Path

# The only errnos `Path.rename()` onto an existing directory is expected to
# raise for "the destination is already occupied" -- confirmed ENOTEMPTY on
# ext4; EEXIST kept for portability to other POSIX filesystems/kernels. Any
# other errno (permission, I/O, cross-device EXDEV, disk full, ...) is a real
# failure and must propagate, never be treated as "someone published first."
RENAME_DEST_EXISTS_ERRNOS = (errno.ENOTEMPTY, errno.EEXIST)

# Bound on retrying a `.refreshN` publish name after a collision with a
# concurrent `--full-refresh` run (see the retry loop in `main()`). A bounded
# retry, not a `while .exists()` probe-then-rename: the probe alone is
# check-then-act and a genuine concurrent racer can occupy the exact name
# between the check and the rename.
REFRESH_RENAME_RETRY_LIMIT = 20

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


def shard_hash_map(shards):
    """The one owner of `{shard path: sha256}` -- `write_state`, the cache-hit
    check, and `fixtures_identity` all need the exact same map, and a second
    independent construction of it is exactly the divergent-copy class that
    caused the prior stale-check bug this file already documents (see the
    `fixtures_identity` docstring below)."""
    return {s["path"]: s["sha256"] for s in shards}


def fixtures_identity(shards):
    """Content identity for a shard set: sha256 of the canonical (sorted)
    shard-hash map, truncated to 16 hex chars.

    NOT the same thing as `snapshot`/`pin`: `pin` is a human-readable stamp
    that can stay fixed while the shards under it are re-pinned in place
    (it happened once, to fixtures-batch-on-stream-v2 -- see the comment in
    `main()`). Naming the cache directory by `pin` alone let two DIFFERENT
    shard-hash sets collide on one path; whichever extraction ran second
    called `shutil.rmtree()` on the directory a concurrent reader (a test in
    another git worktree, or `_common.sh`, which has no lock at all) could
    still be reading through the `toolcalling`/`reasoning`/`unified`
    symlinks -- a real, reproduced `ENOENT` race. Keying the directory name
    by this identity instead means two different shard sets simply never
    share a path, so nothing is ever deleted out from under a live reader.
    """
    pinned = sorted(shard_hash_map(shards).items())
    return hashlib.sha256(json.dumps(pinned).encode()).hexdigest()[:16]


def read_state(snap_dir):
    state_file = snap_dir / ".fixtures-state.json"
    if state_file.exists():
        try:
            return json.loads(state_file.read_text())
        except Exception:
            return {}
    return {}


def resolve_current_generation(cache_root, pin, fid, pinned_shards):
    """The one owner of "which published directory is current for this
    identity" -- generation 0 is the bare `{pin}-{fid}`, and `--full-refresh`
    publishes later generations at `{pin}-{fid}.refresh{N}` (N >= 1) without
    ever touching an earlier one (see the `--full-refresh` branch in
    `main()`). A caller that reconstructs the bare `{pin}-{fid}` path
    directly instead of calling this function is blind to every refresh: it
    treats the abandoned original generation as the cache hit forever,
    silently undoing what `--full-refresh` was run to fix. `main()`'s
    cache-hit check and `package_fixtures.py`'s `_extracted_snapshot_dir()`
    both route through this one function for exactly that reason.

    Returns `(path, generation)` for the highest-generation directory whose
    recorded state still matches `pinned_shards`, or `(None, -1)` if none
    does.
    """
    if not cache_root.exists():
        return None, -1
    base = f"{pin}-{fid}"
    best_dir, best_n = None, -1
    for d in cache_root.iterdir():
        if not d.is_dir():
            continue
        if d.name == base:
            n = 0
        elif d.name.startswith(base + ".refresh"):
            suffix = d.name[len(base + ".refresh") :]
            if not suffix.isdigit():
                continue
            n = int(suffix)
        else:
            continue
        if n <= best_n:
            continue
        if read_state(d).get("shards") != pinned_shards:
            continue
        best_dir, best_n = d, n
    return best_dir, best_n


def write_state(snap_dir, snapshot, shards):
    state = {
        "snapshot": snapshot,
        "shards": shard_hash_map(shards),
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
    for name in ("toolcalling", "reasoning", "unified"):
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
        fid = fixtures_identity(shards)
        pinned_shards = shard_hash_map(shards)
        current_dir, _ = resolve_current_generation(cache_root, pin, fid, pinned_shards)
        print(f"\nCached snapshots in {cache_root}:")
        for d in cached:
            # Cached dirs are named `{pin}-{fid}` or `{pin}-{fid}.refreshN`
            # (see `fixtures_identity` / `resolve_current_generation`), never
            # bare `pin` -- an exact-equality check here always missed. A
            # `startswith(f"{pin}-")` check marks EVERY generation of the
            # current pin, not just the one readers actually resolve through
            # the symlinks -- route through `resolve_current_generation`, the
            # same single owner `main()`'s cache-hit check uses, instead.
            if d == current_dir:
                marker = "  <- current pin (active generation)"
            elif d.name.startswith(f"{pin}-"):
                marker = "  <- current pin (older generation)"
            else:
                marker = ""
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

    pinned_shards = shard_hash_map(shards)
    fid = fixtures_identity(shards)
    # `{pin}-{fid}`, not bare `pin`: two different shard-hash sets under the
    # SAME pin (a re-pin in place) must never share a directory name -- see
    # `fixtures_identity`'s docstring for the race this closes. This is only
    # the generation-0 build target; a cache HIT must check every generation
    # via `resolve_current_generation`, not just this bare name, or a prior
    # `--full-refresh` becomes invisible to every later plain run (see that
    # function's docstring).
    snap_dir = cache_root / f"{pin}-{fid}"

    current_dir, current_n = resolve_current_generation(cache_root, pin, fid, pinned_shards)
    if current_dir and not args.full_refresh:
        # A directory only ever exists at this exact content-addressed name
        # once it is fully built (see the atomic publish below), so its
        # presence alone is already sufficient; `resolve_current_generation`'s
        # state-file check is a cheap extra sanity check against out-of-band
        # interference, not the thing establishing correctness.
        #
        # Retarget the stable symlinks even on a hit: switching between two
        # already-cached identities/generations (e.g. a pin rollback) must
        # repoint toolcalling/ + reasoning/ or readers keep using the other
        # snapshot.
        update_symlinks(cache_root, current_dir, verbose=args.verbose)
        print(f"Cache hit: {current_dir}", file=sys.stderr)
        print(current_dir)
        return

    if args.dry_run:
        for s in shards:
            print(f"  [dry-run] would extract {s['path']}", file=sys.stderr)
        print(f"[dry-run] shards: {len(shards)}", file=sys.stderr)
        print(snap_dir)
        return

    # No orphaned-`.tmp.*`-dir sweep here (deliberately removed after review):
    # an mtime-based age gate only reflects a directory's own DIRECT children
    # changing, not writes deep inside an already-created subtree, so a
    # genuinely still-running long build (or one with a large gap between
    # its last top-level entry and its last file write) could be swept by a
    # CONCURRENT process's own sweep step. `extract_tarball`'s unconditional
    # `mkdir(parents=True, exist_ok=True)` means the victim wouldn't even
    # notice -- it would silently recreate the directory, keep writing, and
    # ultimately publish a `.fixtures-state.json` claiming completeness over
    # data that was actually wiped mid-build. A leftover `.tmp.*` dir is
    # never referenced by any symlink and can never be mistaken for a valid
    # cache (see `list_cached_snapshots`'s state-file check), so leaving
    # orphans in place is purely a disk-hygiene cost, not a correctness one
    # -- not worth trading for that failure mode. Clean up manually if disk
    # usage becomes a real problem.

    # Build into a private, unique temp dir -- never referenced by any
    # symlink, so a concurrent reader can never observe it, and a crash here
    # only ever corrupts this process's own scratch directory. Publish with a
    # single atomic rename once the build (including the state file) is
    # complete, so `snap_dir` is either fully absent or fully populated —
    # never observed mid-extraction by a concurrent reader.
    tmp_dir = cache_root / f".tmp.{fid}.{os.getpid()}.{secrets.token_hex(4)}"
    if tmp_dir.exists():
        shutil.rmtree(str(tmp_dir))
    print(f"Extracting {len(shards)} shard(s) into {tmp_dir}", file=sys.stderr)
    for s in shards:
        extract_tarball(shard_file(s), tmp_dir, verbose=args.verbose)
    write_state(tmp_dir, pin, shards)

    try:
        tmp_dir.rename(snap_dir)
    except OSError as exc:
        # Only "the destination already exists" may mean a concurrent process
        # published the SAME identity first (two uncoordinated first-time
        # builds of one new identity, e.g. via the unlocked `_common.sh`
        # path). Any other errno -- permission, I/O, disk full, cross-device
        # EXDEV -- is a real failure with nothing to reconcile against and
        # must propagate with its original cause, not be swallowed here.
        if exc.errno not in RENAME_DEST_EXISTS_ERRNOS:
            raise
        # Even a "destination exists" error must be verified, not trusted:
        # the directory occupying this name could be a stray/partial one
        # from manual intervention or a bug, not a genuine prior publish.
        published = read_state(snap_dir)
        if not (snap_dir.is_dir() and published.get("shards") == pinned_shards):
            raise OSError(
                exc.errno,
                f"{snap_dir} is occupied but is not a valid published extraction "
                f"for identity {fid} (state mismatch) -- refusing to treat it as "
                "authoritative or overwrite it",
            ) from exc
        if args.full_refresh:
            # `--full-refresh` means "rebuild regardless of cache validity" --
            # e.g. to recover from local corruption that doesn't change the
            # shard-source hash this identity is keyed on (state only records
            # the shard SOURCES' hashes, never the extracted output content).
            # Silently discarding the rebuild here (the plain branch below)
            # defeats the flag's purpose. But renaming/deleting the EXISTING
            # `snap_dir` to make room -- an earlier version of this fix did
            # exactly that -- reopens the identical hazard the rest of this
            # function exists to close: a reader that resolved a symlink
            # pointing at `snap_dir` before this run started may still be
            # reading files through that exact path for the rest of ITS OWN
            # lifetime, with no bound on how long that takes. There is no
            # safe moment to rename or delete a path a reader might still be
            # using -- "briefly" is still unsafe.
            #
            # So a forced rebuild of an identity that already exists NEVER
            # touches `snap_dir` at all. It publishes to a new, disambiguated
            # generation path instead; `snap_dir` itself is simply
            # abandoned, exactly like any other published directory (see the
            # "no orphan sweep, never delete a published dir" decision
            # above) -- a reader already using it keeps working, unaffected,
            # for its entire lifetime. The symlinks (the one mutable
            # "current" locator, already updated via its own atomic
            # temp-then-rename) retarget to the new generation for whoever
            # resolves them AFTER this run.
            # `resolve_current_generation` (not a fresh `.exists()` probe loop
            # from 1) picks the starting number: another concurrent
            # `--full-refresh` racing on the SAME base identity could have
            # already published `.refresh1`, and starting from 1 again would
            # just re-collide. The `.exists()` check on the next candidate is
            # still check-then-act, so the actual rename below is wrapped in
            # the same narrowed-errno retry as the generation-0 publish path
            # above -- two runs computing the identical candidate name is a
            # real, reachable race, not a hypothetical one.
            _, current_n = resolve_current_generation(cache_root, pin, fid, pinned_shards)
            refresh_n = max(current_n, 0) + 1
            published_refresh = False
            for _attempt in range(REFRESH_RENAME_RETRY_LIMIT):
                candidate = cache_root / f"{pin}-{fid}.refresh{refresh_n}"
                try:
                    tmp_dir.rename(candidate)
                    published_refresh = True
                    break
                except OSError as exc2:
                    if exc2.errno not in RENAME_DEST_EXISTS_ERRNOS:
                        raise
                    refresh_n += 1
            if not published_refresh:
                raise OSError(
                    f"could not publish a --full-refresh generation for identity "
                    f"{fid} in {cache_root} after {REFRESH_RENAME_RETRY_LIMIT} "
                    "name collisions -- persistent concurrent refreshers?"
                )
            snap_dir = candidate
            print(
                f"  --full-refresh: identity {fid} already published; built a new "
                f"generation at {snap_dir} instead of mutating the existing one "
                "(a reader still using the prior generation is unaffected)",
                file=sys.stderr,
            )
        else:
            print(
                f"  identity {fid} already published by a concurrent extraction, "
                f"discarding redundant build",
                file=sys.stderr,
            )
            shutil.rmtree(str(tmp_dir))

    update_symlinks(cache_root, snap_dir, verbose=args.verbose)

    print(snap_dir)


if __name__ == "__main__":
    main()
