#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Archive a complete Unified snapshot before transactionally advancing current YAML and its pin."""
import argparse
import gzip
import io
import json
from pathlib import Path
import shutil
import tarfile
import tempfile

import yaml

import capture_sources as sources
from capture_versions import capture_release_sort_key
import unified_archives
import unified_history


def choose_current(rows, candidate):
    key = tuple(candidate[name] for name in ("surface", "family", "implementation", "mode"))
    previous = next((row for row in rows if tuple(row[name] for name in ("surface", "family", "implementation", "mode")) == key), None)
    if previous is not None and capture_release_sort_key(candidate["version"]) <= capture_release_sort_key(previous["version"]):
        if candidate["version"] == previous["version"] and candidate["sha256"] != previous["sha256"]:
            raise ValueError("a frozen current release cannot be overwritten")
        return list(rows), False
    return [row for row in rows if row is not previous] + [candidate], True


PENDING = sources.UPDATE_JOURNAL


def install(payloads, parent, root, owner):
    stage = Path(tempfile.mkdtemp(prefix=".capture-install-", dir=parent))
    pending = json.loads(sources.read(root, PENDING))
    pending["stages"].append({"owner": owner, "path": stage.relative_to(parent).as_posix()})
    # Register the empty stage before creating evidence-shaped files. Recovery
    # must remove these candidates even if the transaction journal never existed.
    unified_history._write_transaction_journal(root / PENDING, pending)
    try:
        pairs = []
        for index, (name, data) in enumerate(payloads):
            destination = parent / sources.relative(name)
            sources.relative(destination.relative_to(parent).as_posix())
            destination.parent.mkdir(parents=True, exist_ok=True)
            candidate = stage / str(index)
            candidate.write_bytes(data)
            pairs.append((candidate, destination))
        unified_history.publish_paths_transactionally(pairs, backup_parent=parent)
    finally:
        if not (parent / unified_history.TRANSACTION_JOURNAL).exists():
            shutil.rmtree(stage)


def recover_locked(root, history):
    pending_path = root / PENDING
    if not pending_path.exists():
        return False
    pending = json.loads(sources.read(root, PENDING))
    if pending["history_source"] != str(history):
        raise ValueError("pending capture publication belongs to another history source")
    payload_root = root / sources.relative(pending["payload"])
    if payload_root.parent != root / "conformance" or not payload_root.name.startswith(".capture-pending-"):
        raise ValueError("invalid pending capture payload")
    bodies = {}
    for row in pending["files"]:
        data = sources.read(payload_root, row["path"])
        if sources.identity(row["path"], data) != row:
            raise ValueError("pending capture payload changed")
        bodies[row["path"]] = data
    unified_history._recover_publication(history)
    unified_history._recover_publication(root / "conformance")
    for entry in pending["stages"]:
        if entry["owner"] not in {"history", "current"}:
            raise ValueError("invalid pending staging owner")
        parent = history if entry["owner"] == "history" else root / "conformance"
        stage = parent / sources.relative(entry["path"])
        if stage.parent != parent or not stage.name.startswith(".capture-install-") or stage.is_symlink():
            raise ValueError("invalid pending staging path")
        if stage.exists():
            shutil.rmtree(stage)
    for row in pending["authored"]:
        if sources.identity(row["path"], sources.read(root, row["path"])) != row:
            raise ValueError("authored inputs changed during capture publication")
    old_pin, new_pin = pending["old_pin"], json.loads(bodies["pin.json"])
    current_pin = json.loads(sources.read(root, sources.PIN))
    current_index_digest = sources.digest(sources.read(root, sources.CURRENT))
    if current_pin not in (old_pin, new_pin) or current_index_digest not in (pending["old_current_sha256"], sources.digest(bodies["current.json"])):
        raise ValueError("current capture pointers changed during pending publication")
    external_digest = sources.digest(sources.read(history, sources.INVENTORY))
    if external_digest == old_pin["inventory_sha256"]:
        sources.verify(history, old_pin)
        install([(pending["archive"], bodies["archive.tar.gz"]),
                 ("conformance/fixtures-manifest.json", bodies["manifest.json"]),
                 (sources.INVENTORY, bodies["inventory.json"])], history, root, "history")
    elif external_digest != new_pin["inventory_sha256"]:
        raise ValueError("history changed during pending capture publication")
    sources.verify(history, new_pin)
    if current_pin != new_pin or current_index_digest != sources.digest(bodies["current.json"]):
        local = [("capture-current.json", bodies["current.json"]), ("capture-history-pin.json", bodies["pin.json"])]
        if pending["current_target"] is not None:
            local.insert(0, (pending["current_target"].removeprefix("conformance/"), bodies["capture.yaml"]))
        install(local, root / "conformance", root, "current")
    pending_path.unlink()
    unified_history._fsync_directory(pending_path.parent)
    shutil.rmtree(payload_root)
    return True


def recover(root, history):
    with unified_history._store_mutation_lock(root / sources.STORE), unified_history._store_mutation_lock(history / sources.STORE):
        return recover_locked(root, history)


def publish_loose(root: Path, history: Path, loose: Path) -> list[dict]:
    """Use the existing capture importer, then archive each newly recorded snapshot."""
    recover(root, history)
    current = sources.current(root)
    previous = {row["source"]: row for row in current["captures"] if row["surface"] == "unified"}
    results = []
    with sources.unified_view(root) as view:
        implementations = sorted({row["implementation"] for row in current["captures"] if row["surface"] == "unified"})
        store, _ = unified_history.selected_materialization_store(view, [f"latest:{implementation}" for implementation in implementations])
        updated = unified_history._update_from_loose(store, loose, complete_snapshot=False)
        if updated is None:
            return []
        for path, document in unified_history._store_documents(store).items():
            identity = unified_history.source_identity(path.relative_to(view).as_posix())
            if identity["kind"] != "history":
                continue
            logical = sources.STORE + "/" + path.relative_to(view).as_posix()
            prior = previous.get(logical)
            if prior is not None and unified_history.typed_equal(unified_history.load_yaml(root / prior["path"]), document):
                continue
            candidate = view / "candidate.yaml"
            candidate.write_text(unified_history.dump_yaml(document))
            results.append(publish(root, history, candidate, identity["family"], identity["implementation"], identity["version"]))
    return results


def publish(root: Path, history: Path, capture: Path, family: str, implementation: str, version: str) -> dict:
    capture_release_sort_key(version)
    sources.relative(family)
    if "/" in family:
        raise ValueError("family must be one component")
    engine = unified_archives.engine(implementation)
    with unified_history._store_mutation_lock(root / sources.STORE), unified_history._store_mutation_lock(history / sources.STORE):
        recover_locked(root, history)
        data = capture.read_bytes()
        authored_name = f"{sources.STORE}/families/{family}/inputs_and_golden.yaml"
        authored_bytes = sources.read(root, authored_name)
        authored_path = root / authored_name
        authored = unified_history._validate_family(authored_path, yaml.load(authored_bytes, Loader=unified_history.StrictLoader))
        document = yaml.load(data, Loader=unified_history.StrictLoader)
        unified_history._validate_complete_snapshot(capture, document, authored, implementation, version)
        logical = f"{sources.STORE}/families/{family}/{implementation}-{version}.yaml"
        path = f"conformance/captures/unified/{family}/{implementation}.yaml"
        row = {**sources.identity(path, data), "source": logical, "surface": "unified", "mode": "unified",
               "family": family, "implementation": implementation, "version": version}
        current = sources.current(root)
        pin = json.loads(sources.read(root, sources.PIN))
        inventory = sources.verify(history, pin)
        rows, advanced = choose_current(current["captures"], row)
        existing = {item["path"]: item for item in unified_archives.inventory(history / sources.STORE)}
        relative = logical.removeprefix(sources.STORE + "/")
        if relative in existing:
            if unified_archives.read_source(history / sources.STORE, existing[relative]) != data:
                raise ValueError("frozen historical capture differs from the proposed bytes")
            return {"archived": False, "current_advanced": False}
        lineage = {item["path"]: item for item in current["lineage"]}
        for item in existing.values():
            identity = unified_history.source_identity(item["path"])
            if identity["kind"] == "history":
                lineage[sources.STORE + "/" + item["path"]] = {"path": sources.STORE + "/" + item["path"], **identity}
        lineage[logical] = {"path": logical, "kind": "history", "family": family, "implementation": implementation, "version": version}
        if any(record["member"] not in lineage for record in document["records"].values()):
            raise ValueError("capture references unknown observation lineage")
        raw = io.BytesIO()
        with tarfile.open(fileobj=raw, mode="w", format=tarfile.USTAR_FORMAT) as archive:
            member = tarfile.TarInfo(relative)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
        compressed = gzip.compress(raw.getvalue(), mtime=0)
        archive_name = f"{sources.STORE}/versions/{engine}-{version}.capture{sources.digest(data)}.tar.gz"
        logical_inventory = [{key: item[key] for key in ("path", "size", "sha256")} for item in existing.values()]
        logical_inventory.append(sources.identity(relative, data))
        logical_inventory.sort(key=lambda item: item["path"])
        digest = sources.digest(json.dumps(logical_inventory, sort_keys=True, separators=(",", ":")).encode())
        size = sum(item["size"] for item in logical_inventory)
        manifest_path = "conformance/fixtures-manifest.json"
        manifest = json.loads(sources.read(history, manifest_path))
        for shard in manifest["shards"]:
            if shard.get("format") == "unified-history":
                shard.update(sha256=digest, size=size)
        manifest_bytes = sources.encoded(manifest)
        inventory["files"] = [item for item in inventory["files"] if item["path"] != manifest_path]
        inventory["files"].extend([sources.identity(manifest_path, manifest_bytes),
                                   {**sources.identity(archive_name, compressed), "members": [{**sources.identity(relative, data), "kind": "file"}]}])
        inventory["files"].sort(key=lambda item: item["path"])
        inventory_bytes = sources.encoded(inventory)
        new_pin = {**pin, "inventory_sha256": sources.digest(inventory_bytes)}
        current["captures"] = sorted(rows, key=lambda item: item["path"])
        if advanced:
            members = {record["member"] for record in document["records"].values()}
            current["lineage"] = sorted({**{item["path"]: item for item in current["lineage"]},
                                         **{member: lineage[member] for member in members}}.values(), key=lambda item: item["path"])
        payloads = {"archive.tar.gz": compressed, "manifest.json": manifest_bytes, "inventory.json": inventory_bytes,
                    "current.json": sources.encoded(current), "pin.json": sources.encoded(new_pin), "capture.yaml": data}
        payload_root = Path(tempfile.mkdtemp(prefix=".capture-pending-", dir=root / "conformance"))
        for name, body in payloads.items():
            (payload_root / name).write_bytes(body)
            unified_history._fsync_regular_file(payload_root / name)
        unified_history._fsync_directory(payload_root)
        pending = {"schema": 1, "history_source": str(history), "payload": payload_root.relative_to(root).as_posix(),
                   "archive": archive_name, "current_target": path if advanced else None,
                   "old_pin": pin, "old_current_sha256": sources.digest(sources.read(root, sources.CURRENT)),
                   "authored": [sources.identity(authored_name, authored_bytes)],
                   "files": [sources.identity(name, body) for name, body in payloads.items()], "stages": []}
        unified_history._write_transaction_journal(root / PENDING, pending)
        recover_locked(root, history)
    return {"archived": True, "current_advanced": advanced, "source": logical}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--history-source", type=Path, required=True)
    parser.add_argument("--recover", action="store_true")
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--family")
    parser.add_argument("--implementation")
    parser.add_argument("--version")
    args = parser.parse_args()
    if args.recover:
        print(json.dumps({"recovered": recover(args.root.resolve(), args.history_source.resolve())}))
        return
    if args.capture is None or not all((args.family, args.implementation, args.version)):
        parser.error("publication requires --capture, --family, --implementation, and --version")
    print(json.dumps(publish(args.root.resolve(), args.history_source.resolve(), args.capture.resolve(), args.family, args.implementation, args.version)))


if __name__ == "__main__":
    main()
