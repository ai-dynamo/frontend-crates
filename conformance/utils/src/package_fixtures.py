#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Package tool-calling and reasoning fixtures into per-version tarballs in the
repo's LFS store, update the reviewable Unified YAML store, and write the
manifest that pins both sources. Publishing a snapshot means committing both
stores and the manifest to git; no external service is involved.

Shard layout (relative to conformance/fixtures/):
  toolcalling/fixtures-batch-v1/inputs.tar.gz
  toolcalling/fixtures-batch-v1/<impl>-<ver>.tar.gz   (one per immediate subdir)
  toolcalling/fixtures-stream-v1/inputs.tar.gz
  toolcalling/fixtures-stream-v1/<impl>-<ver>.tar.gz
  toolcalling/fixtures-batch-on-stream-v1.tar.gz      (whole tree as one tarball)
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
from dynamo_version import capture_source_fingerprint
import fixture_disposition
import stream_capture_archive
import unified_history

# conformance/utils/src/ -> repo root: 4 .parent calls (strip filename, then 3 dirs)
ROOT = Path(__file__).resolve().parent.parent.parent.parent
MANIFEST_REL = Path("conformance") / "fixtures-manifest.json"
FIXTURES_DIR = ROOT / "conformance" / "fixtures"
UNIFIED_HISTORY_DIR = ROOT / "conformance" / "fixtures-unified-v2"
STREAM_CASE_ID_RE = re.compile(r"TOOLCALLING\.streamv1\.\d+(?:-\d+)?(?:\.[a-z][a-z0-9_-]*)*")


# Fixture trees that get one archive per immediate subdirectory. Unified is listed so
# its loose capture tree is staged, then is written to the separate YAML history store.
PER_SUBDIR_TREES = [
    "toolcalling/fixtures-batch-v1",
    # Legacy, non-Unified streaming uses the v1 corpus convention. This does not
    # rename the parser crate or its implementation versions.
    "toolcalling/fixtures-stream-v1",
    "reasoning/fixtures-v1",
    "unified",
]
# Fixture trees bundled as a single tarball (whole tree, no per-version sharding)
# Tuple: (source rel-path in conformance/, shard path in the store)
WHOLE_TREE_SHARDS = [
    (
        "toolcalling/fixtures-batch-on-stream-v1",
        "toolcalling/fixtures-batch-on-stream-v1.tar.gz",
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


def stage_fixtures(conformance_root, tmpdir, *, trees=None):
    """Copy all fixture trees into tmpdir, preserving the relative layout."""
    all_trees = trees if trees is not None else list(PER_SUBDIR_TREES) + [src for src, _ in WHOLE_TREE_SHARDS]
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
    trees=None,
):
    """Build per-version shard tarballs and whole-tree shards. Returns list of shard dicts."""
    shards = []
    inactive = preserved_evidence()

    for tree_rel in (trees if trees is not None else PER_SUBDIR_TREES):
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


def _require_stream_capture_receipt(
    receipt,
    blobs_dir,
    fixtures_dir,
    capture_path,
    capture_root,
    required_cases,
    *,
    context,
):
    if receipt is None:
        raise ValueError(f"{context} requires a matching capture receipt: {capture_path}")
    label = capture_root.removeprefix("dynamo_v2-")
    try:
        expected_source_sha256 = capture_source_fingerprint(ROOT, label)
    except ValueError as exc:
        raise ValueError(f"{context} requires a verifiable parser source: {capture_path}") from exc
    candidate_inputs = blobs_dir / "toolcalling/fixtures-stream-v1/inputs.tar.gz"
    inputs_path = candidate_inputs if candidate_inputs.exists() else fixtures_dir / candidate_inputs.relative_to(blobs_dir)
    if not stream_capture_archive.validate_capture_receipt(
        receipt,
        inputs_path,
        blobs_dir / capture_path,
        capture_root,
        required_cases,
        expected_source_sha256,
    ):
        raise ValueError(f"{context} requires a matching capture receipt: {capture_path}")


def sync_store(
    blobs_dir,
    shards,
    dry_run,
    prune,
    *,
    fixtures_dir=None,
    manifest_path=None,
    approved_stream_replacements=frozenset(),
    stream_capture_receipt=None,
):
    """Copy built shards into conformance/fixtures/.

    Store files not in the new shard set are KEPT unless --prune is passed:
    the local capture trees are often partial (one family recaptured, the rest
    absent), and mirroring a partial tree would silently drop shards. Capture
    versions are additive by design — a re-record ADDS a version subdir, so
    its shard joins the set. Existing stream results can change only for case
    IDs explicitly approved by the caller; pruning is for deliberately retired trees.
    """
    fixtures_dir = Path(fixtures_dir or FIXTURES_DIR)
    new_paths = {s["path"] for s in shards}
    manifest_path = Path(manifest_path or ROOT / MANIFEST_REL)
    inactive = preserved_evidence(manifest_path=manifest_path, fixtures_dir=fixtures_dir)
    if new_paths & inactive.keys():
        raise ValueError(f"cannot overwrite inactive evidence: {sorted(new_paths & inactive.keys())}")
    stream_inputs_path = "toolcalling/fixtures-stream-v1/inputs.tar.gz"
    stream_inputs = next((s for s in shards if s["path"] == stream_inputs_path), None)
    allowed_stream_cases = set()
    corrected_stream_cases = set()
    known_stream_cases = set()
    if stream_inputs is not None:
        existing_inputs = fixtures_dir / stream_inputs_path
        candidate_inputs = blobs_dir / stream_inputs_path
        known_stream_cases = stream_capture_archive.stream_input_case_ids(candidate_inputs)
        if existing_inputs.exists():
            if sha256_file(existing_inputs) != stream_inputs["sha256"]:
                replacements = {
                    case
                    for case in known_stream_cases
                    if case[1] in approved_stream_replacements
                }
                changes = stream_capture_archive.stream_input_changes(
                    existing_inputs,
                    candidate_inputs,
                    replacements,
                )
                if changes is None:
                    raise ValueError(f"stream input archive has unapproved changes: {stream_inputs_path}")
                allowed_stream_cases, corrected_stream_cases = changes
                unmatched_approvals = approved_stream_replacements - {
                    case_id for _family, case_id in corrected_stream_cases
                }
                if unmatched_approvals:
                    raise ValueError(f"approved stream cases were not corrected: {sorted(unmatched_approvals)}")
        else:
            allowed_stream_cases = stream_capture_archive.stream_input_case_ids(candidate_inputs)
    if stream_inputs is None and (fixtures_dir / stream_inputs_path).exists():
        known_stream_cases = stream_capture_archive.stream_input_case_ids(fixtures_dir / stream_inputs_path)
    if corrected_stream_cases and manifest_path.exists():
        prior = fixture_disposition.active_shards(json.loads(manifest_path.read_text()))
        affected_stream_archives = {}
        affected_peer_versions = {}
        for shard in prior:
            path = shard["path"]
            if (
                not path.startswith("toolcalling/fixtures-stream-v1/")
                or path == stream_inputs_path
                or not path.endswith(".tar.gz")
            ):
                continue
            existing_capture = fixtures_dir / path
            if not existing_capture.exists():
                continue
            capture_root = Path(path).name.removesuffix(".tar.gz")
            affected_cases = stream_capture_archive.stream_recorded_case_ids(existing_capture, capture_root)
            corrected_for_archive = affected_cases & corrected_stream_cases
            if corrected_for_archive:
                if not capture_root.startswith("dynamo_v2-"):
                    if (
                        path in new_paths
                        and sha256_file(blobs_dir / path) != sha256_file(existing_capture)
                    ):
                        raise ValueError(
                            f"stream capture archives are append-only; add a patch overlay instead: {path}"
                        )
                    base, _patch = fixture_disposition.capture_layer_sort_key(capture_root)
                    version = affected_peer_versions.setdefault(
                        base, {"cases": set(), "layers": {}}
                    )
                    version["cases"].update(corrected_for_archive)
                    for candidate_shard in prior:
                        candidate_path = candidate_shard["path"]
                        if (
                            not candidate_path.startswith("toolcalling/fixtures-stream-v1/")
                            or candidate_path == stream_inputs_path
                            or not candidate_path.endswith(".tar.gz")
                        ):
                            continue
                        candidate_root = Path(candidate_path).name.removesuffix(".tar.gz")
                        candidate_base, _candidate_patch = (
                            fixture_disposition.capture_layer_sort_key(candidate_root)
                        )
                        candidate_archive = fixtures_dir / candidate_path
                        if candidate_base == base and candidate_archive.exists():
                            version["layers"][candidate_root] = candidate_archive
                    continue
                if path not in new_paths:
                    raise ValueError(f"corrected stream input requires recapturing active archive: {path}")
                affected_stream_archives[path] = (capture_root, corrected_for_archive)
        for base, version in affected_peer_versions.items():
            base_archive = version["layers"].get(base)
            existing_patches = {
                root: path for root, path in version["layers"].items() if root != base
            }
            peer_patches = [
                candidate["path"]
                for candidate in shards
                if candidate["path"].startswith("toolcalling/fixtures-stream-v1/")
                and candidate["path"].endswith(".tar.gz")
                and (blobs_dir / candidate["path"]).is_file()
                and stream_capture_archive.stream_capture_patch_allowed(
                    base_archive,
                    blobs_dir / candidate["path"],
                    base,
                    Path(candidate["path"]).name.removesuffix(".tar.gz"),
                    version["cases"],
                    existing_patches,
                )
            ] if base_archive is not None else []
            if len(peer_patches) != 1:
                raise ValueError(
                    f"corrected stream input requires one append-only capture patch or explicit retirement: {base}"
                )
        for shard in shards:
            path = shard["path"]
            if not path.startswith("toolcalling/fixtures-stream-v1/dynamo_v2-") or not path.endswith(".tar.gz"):
                continue
            capture_root = Path(path).name.removesuffix(".tar.gz")
            candidate_capture = blobs_dir / path
            if not candidate_capture.exists():
                continue
            candidate_cases = stream_capture_archive.stream_recorded_case_ids(
                candidate_capture, capture_root
            )
            corrected_for_archive = candidate_cases & corrected_stream_cases
            if corrected_for_archive:
                prior_cases = affected_stream_archives.get(path, (capture_root, set()))[1]
                affected_stream_archives[path] = (
                    capture_root,
                    prior_cases | corrected_for_archive,
                )
        for path, (capture_root, corrected_for_archive) in affected_stream_archives.items():
            _require_stream_capture_receipt(
                stream_capture_receipt,
                blobs_dir,
                fixtures_dir,
                path,
                capture_root,
                corrected_for_archive,
                context="corrected stream input",
            )
    for shard in shards:
        path = shard["path"]
        if not path.startswith("toolcalling/fixtures-stream-v1/dynamo_v2-"):
            continue
        existing_capture = fixtures_dir / path
        candidate_capture = blobs_dir / path
        if not candidate_capture.exists():
            continue
        capture_root = Path(path).name.removesuffix(".tar.gz")
        candidate_capture_cases = stream_capture_archive.stream_recorded_case_ids(
            candidate_capture, capture_root
        )
        if not existing_capture.exists():
            if not candidate_capture_cases <= known_stream_cases:
                raise ValueError(f"stream capture includes cases missing from shared inputs: {path}")
            receipt_cases = candidate_capture_cases
            receipt_context = "new stream capture"
        else:
            receipt_cases = candidate_capture_cases & allowed_stream_cases
            receipt_context = "new stream backfill"
        if receipt_cases:
            _require_stream_capture_receipt(
                stream_capture_receipt,
                blobs_dir,
                fixtures_dir,
                path,
                capture_root,
                receipt_cases,
                context=receipt_context,
            )
        if not existing_capture.exists():
            continue
        if sha256_file(existing_capture) == shard["sha256"]:
            if allowed_stream_cases and not stream_capture_archive.merge_stream_capture_additions(
                existing_capture,
                candidate_capture,
                Path(path).name.removesuffix(".tar.gz"),
                allowed_stream_cases,
                corrected_stream_cases,
                known_stream_cases,
            ):
                raise ValueError(f"stream capture is missing approved additions: {path}")
            continue
        if stream_capture_archive.merge_stream_capture_additions(
            existing_capture,
            candidate_capture,
            capture_root,
            allowed_stream_cases,
            corrected_stream_cases,
            known_stream_cases,
        ):
            shard["sha256"] = sha256_file(candidate_capture)
            shard["size"] = candidate_capture.stat().st_size
    # New cases need matching input additions; result corrections need explicit case IDs.
    # Producer metadata and non-YAML members remain immutable.
    for shard in shards:
        if shard.get("format") == "unified-history":
            continue
        destination = fixtures_dir / shard["path"]
        if re.match(r"^[a-z0-9_]+-\d", destination.name) and destination.exists():
            if sha256_file(destination) != shard["sha256"]:
                stream_update_allowed = (
                    shard["path"].startswith("toolcalling/fixtures-stream-v1/dynamo_v2-")
                    and stream_capture_archive.stream_capture_updates_allowed(
                        destination,
                        blobs_dir / shard["path"],
                        Path(shard["path"]).name.removesuffix(".tar.gz"),
                        allowed_stream_cases,
                        corrected_stream_cases,
                        known_stream_cases,
                    )
                )
                if not stream_update_allowed:
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


def package_snapshot(
    stamp,
    created_pt,
    crates,
    peers,
    *,
    dry_run,
    prune,
    stream_only=False,
    unified_only=False,
    approved_stream_replacements=frozenset(),
    stream_capture_receipt=None,
):
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
            trees = (
                ["toolcalling/fixtures-stream-v1"]
                if stream_only
                else ["unified"]
                if unified_only
                else None
            )
            if trees is not None:
                stage_fixtures(conformance_root, loose_root, trees=trees)
            else:
                stage_fixtures(conformance_root, loose_root)
            shutil.copytree(FIXTURES_DIR, candidate_fixtures, copy_function=os.link)
            shutil.copytree(UNIFIED_HISTORY_DIR, candidate_history)

            print("\nBuilding shards…")
            if trees is not None:
                shards = build_shards(loose_root, blobs_dir, prune, history_root=candidate_history, trees=trees)
            else:
                shards = build_shards(loose_root, blobs_dir, prune, history_root=candidate_history)

            print(f"\nStaging store candidate for: {FIXTURES_DIR}")
            sync_store(
                blobs_dir,
                shards,
                False,
                prune,
                fixtures_dir=candidate_fixtures,
                manifest_path=manifest_path,
                approved_stream_replacements=approved_stream_replacements,
                stream_capture_receipt=stream_capture_receipt,
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

            publish_paths = [
                (candidate_fixtures, FIXTURES_DIR),
                (candidate_manifest, manifest_path),
            ]
            if not stream_only:
                publish_paths.insert(0, (candidate_history, UNIFIED_HISTORY_DIR))
            unified_history.publish_paths_transactionally(
                publish_paths,
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
    ap.add_argument("--stream-only", action="store_true", help="Package only the loose tool-calling stream tree")
    ap.add_argument("--unified-only", action="store_true", help="Package only the Unified capture history")
    ap.add_argument(
        "--replace-stream-case",
        action="append",
        default=[],
        metavar="CASE_ID",
        help="Allow an approved stream input and capture correction for this full case ID",
    )
    ap.add_argument(
        "--stream-capture-receipt",
        type=Path,
        default=None,
        help="validate this capture receipt when packaging new or amended Dynamo stream captures",
    )
    ap.add_argument(
        "--prune",
        action="store_true",
        help="Remove store shards (and manifest entries) not rebuilt by this run. "
        "Default keeps them: local capture trees are often partial.",
    )
    args = ap.parse_args()
    if args.stream_only and args.unified_only:
        ap.error("--stream-only and --unified-only cannot be combined")
    if (args.stream_only or args.unified_only) and args.prune:
        ap.error("scoped package modes cannot be combined with --prune")
    if args.replace_stream_case and not args.stream_only:
        ap.error("--replace-stream-case requires --stream-only")
    invalid_replacements = [
        case_id
        for case_id in args.replace_stream_case
        if STREAM_CASE_ID_RE.fullmatch(case_id) is None
    ]
    if invalid_replacements:
        ap.error(f"invalid stream case IDs: {invalid_replacements}")

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

    receipt = None
    if args.stream_capture_receipt is not None:
        receipt = json.loads(args.stream_capture_receipt.read_text())

    package_snapshot(
        stamp,
        created_pt,
        crates,
        peers,
        dry_run=args.dry_run,
        prune=args.prune,
        stream_only=args.stream_only,
        unified_only=args.unified_only,
        approved_stream_replacements=frozenset(args.replace_stream_case),
        stream_capture_receipt=receipt,
    )


if __name__ == "__main__":
    main()
