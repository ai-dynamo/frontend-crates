# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep inactive evidence in the store without admitting it to active fixtures."""

import hashlib
import json
import re
import tarfile
from pathlib import Path, PurePosixPath

CAPTURE_SNAPSHOT = "capture-snapshot.json"
DYNAMO_VERSION_RE = re.compile(r"\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?")


def is_source_capture(name: str) -> bool:
    if not name.startswith("dynamo_v2-"):
        return False
    version, separator, digest = name.removeprefix("dynamo_v2-").partition("+source.")
    return bool(separator and DYNAMO_VERSION_RE.fullmatch(version) and re.fullmatch(r"[0-9a-f]{64}", digest))


def capture_snapshot_members(data: bytes | None, available) -> set[str] | None:
    if data is None:
        return None
    doc = json.loads(data)
    members = doc.get("records")
    if doc.get("schema_version") != 1 or not isinstance(members, list) or any(not isinstance(key, str) for key in members):
        raise ValueError("invalid complete capture snapshot")
    paths = set(members)
    if len(paths) != len(members) or paths != {key for key in available if key.endswith(".yaml")}:
        raise ValueError("complete capture snapshot differs from its capture files")
    return paths


def capture_layer_sort_key(label: str) -> tuple[str, int]:
    match = re.fullmatch(r"(.*)\.patch(\d+)", label)
    return (match[1], int(match[2])) if match else (label, 0)


def historical_unified_case_key(family: str, key: str) -> str:
    # Only these taxonomy renames preserve the stimulus; 30.m is not a number.
    match = re.fullmatch(r"UNIFIED\.31\.([a-x])", key)
    if match:
        key = f"UNIFIED.31-{ord(match[1]) - ord('a') + 1}"
    if family == "gemma4":
        key = {"UNIFIED.31-29": "UNIFIED.g4-1", "UNIFIED.31-30": "UNIFIED.g4-2"}.get(key, key)
    return key


def capture_archive_layers(store: Path, relative: str) -> list[Path]:
    base = store / relative
    candidates = [base.with_name(base.name + ".tar.gz")]
    candidates.extend(base.parent.glob(base.name + ".patch*.tar.gz"))
    return sorted(
        (path for path in candidates if path.is_file()
         and capture_layer_sort_key(path.name.removesuffix(".tar.gz"))[0] == base.name),
        key=lambda path: capture_layer_sort_key(path.name.removesuffix(".tar.gz")),
    )


def capture_archive_files(path: Path, relative: str) -> dict[str, bytes]:
    prefix = relative + "/"
    files = {}
    with tarfile.open(path) as archive:
        for member in archive.getmembers():
            if member.isdir():
                continue
            if not member.isfile() or not member.name.startswith(prefix):
                raise ValueError(f"unexpected capture archive member: {path}: {member.name}")
            key = member.name[len(prefix):]
            if ".." in PurePosixPath(key).parts or key in files:
                raise ValueError(f"ambiguous capture archive member: {path}: {member.name}")
            with archive.extractfile(member) as source:
                files[key] = source.read()
    return files


def inactive_shards(manifest: dict) -> dict[str, dict]:
    result = {}
    for shard in manifest.get("inactive_shards", []):
        path = shard["path"]
        parts = PurePosixPath(path)
        if parts.is_absolute() or ".." in parts.parts or str(parts) != path or not path.endswith(".tar.gz"):
            raise ValueError(f"invalid inactive shard path: {path}")
        if path in result:
            raise ValueError(f"duplicate inactive shard: {path}")
        if not re.fullmatch(r"[0-9a-f]{64}", shard["sha256"]):
            raise ValueError(f"invalid inactive shard hash: {path}")
        if shard.get("disposition") not in {"quarantined", "superseded"} or not shard.get("reason"):
            raise ValueError(f"inactive shard needs a disposition and reason: {path}")
        if type(shard.get("size")) is not int or shard["size"] < 0:
            raise ValueError(f"invalid inactive shard size: {path}")
        result[path] = shard
    return result


def active_shards(manifest: dict) -> list[dict]:
    inactive = inactive_shards(manifest)
    shards = manifest.get("shards", [])
    for shard in shards:
        if shard["path"] in inactive:
            raise ValueError(f"inactive shard is also active: {shard['path']}")
    return shards


def canonical_inactive_shards(entries) -> list[dict]:
    return sorted(inactive_shards({"inactive_shards": list(entries)}).values(), key=lambda shard: shard["path"])


def verify_inactive_shards(manifest: dict, store: Path) -> dict[str, dict]:
    inactive = inactive_shards(manifest)
    for path, shard in inactive.items():
        data = (store / path).read_bytes()
        if len(data) != shard["size"] or hashlib.sha256(data).hexdigest() != shard["sha256"]:
            raise ValueError(f"inactive evidence differs from its pinned bytes: {path}")
    return inactive


def inactive_fixture_dirs(base: Path) -> set[str]:
    # Loose trees have the repository manifest; extracted trees retain its disposition
    # in their immutable state file. Neither path interprets a version-wide wildcard.
    manifest_path = base.parent / "fixtures-manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        active_shards(manifest)
        inactive = verify_inactive_shards(manifest, base.parent / "fixtures")
    else:
        state = base.parent / ".fixtures-state.json"
        inactive = inactive_shards(json.loads(state.read_text())) if state.is_file() else {}
    prefix = f"{base.name}/"
    return {path[len(prefix):-len('.tar.gz')] for path in inactive
            if path.startswith(prefix) and "/" not in path[len(prefix):]}
