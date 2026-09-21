# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep inactive evidence in the store without admitting it to active fixtures."""

import hashlib
import json
import re
import tarfile
from pathlib import Path, PurePosixPath

import unified_taxonomy
from unified_taxonomy import historical_case_label

CAPTURE_SNAPSHOT = "capture-snapshot.json"
DYNAMO_VERSION_RE = re.compile(r"\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?")
UNIFIED_HISTORY_PATH = "unified-history"


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
    """Read historical archive IDs through the scenario-owned taxonomy aliases."""
    label = key.removeprefix("UNIFIED.")
    return f"UNIFIED.{historical_case_label(label, family)}"


def canonical_unified_case_key(family: str, key: str, scenario: str | None = None) -> str:
    """Return the shared key for new and historical Unified fixture records."""
    if scenario in unified_taxonomy.UNIFIED_TAX:
        return unified_taxonomy.numbered_id(scenario)
    return historical_unified_case_key(family, key)


def canonicalize_unified_inputs(records: dict[tuple[str, str], dict]) -> tuple[dict, dict]:
    """Apply scenario-owned input IDs once, retaining aliases for immutable captures."""
    canonical = {}
    aliases = {}
    scenario_keys = {}
    for (family, key), record in records.items():
        scenario = record.get("scenario")
        ident = (family, canonical_unified_case_key(family, key, scenario))
        if scenario:
            previous = scenario_keys.get((family, scenario))
            if (
                previous is not None
                and previous != key
                and historical_unified_case_key(family, previous)
                != historical_unified_case_key(family, key)
            ):
                raise ValueError(
                    f"duplicate Unified scenario ownership for {family}/{scenario}: "
                    f"{previous}, {key}"
                )
            scenario_keys[(family, scenario)] = key
        canonical[ident] = record
        historical = historical_unified_case_key(family, key)
        aliases[(family, key)] = ident
        aliases[(family, historical)] = ident
        if key.startswith("UNIFIED."):
            aliases[(family, key.removeprefix("UNIFIED."))] = ident
        else:
            aliases[(family, f"UNIFIED.{key}")] = ident
    return canonical, aliases


def canonical_unified_record_key(family: str, key: str, input_aliases: dict) -> tuple[str, str]:
    """Resolve capture and golden records through the shared input ownership map."""
    historical = historical_unified_case_key(family, key)
    return input_aliases.get(
        (family, key),
        input_aliases.get((family, historical), (family, historical)),
    )


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
        if shard.get("format") == "unified-history":
            if shard.get("path") != UNIFIED_HISTORY_PATH:
                raise ValueError(f"invalid Unified history path: {shard.get('path')}")
        elif shard["path"] in inactive:
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
        active = active_shards(manifest)
        unified_history_active = any(
            shard.get("format") == "unified-history" for shard in active
        )
        inactive = verify_inactive_shards(manifest, base.parent / "fixtures")
    else:
        state = base.parent / ".fixtures-state.json"
        document = json.loads(state.read_text()) if state.is_file() else {}
        unified_history_active = UNIFIED_HISTORY_PATH in document.get("shards", {})
        inactive = inactive_shards(document)
    if base.name == "unified" and unified_history_active:
        return set()
    prefix = f"{base.name}/"
    return {path[len(prefix):-len('.tar.gz')] for path in inactive
            if path.startswith(prefix) and "/" not in path[len(prefix):]}
