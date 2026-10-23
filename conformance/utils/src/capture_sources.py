#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned capture evidence and local current snapshots, independent of publication."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
import tempfile

import yaml
import yaml_fast  # noqa: F401
import ci_publication_cache

CURRENT = "conformance/capture-current.json"
PIN = "conformance/capture-history-pin.json"
INVENTORY = "capture-inventory.json"
UPDATE_JOURNAL = "conformance/.capture-update.json"
STORE = "conformance/fixtures-unified-v2"
LEGACY_CORPORA = {"batch": "toolcalling/fixtures-batch-v1", "stream": "toolcalling/fixtures-stream-v1",
                  "reasoning": "reasoning/fixtures-v1", "batch-on-stream": "toolcalling/fixtures-batch-on-stream-v1"}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encoded(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def relative(name: str) -> str:
    if (not name or "\\" in name or ":" in name or PurePosixPath(name).is_absolute()
            or any(part in ("", ".", "..") for part in name.split("/"))):
        raise ValueError(f"unsafe capture path: {name!r}")
    return name


def read(root: Path, name: str) -> bytes:
    path = root / relative(name)
    if any(parent.is_symlink() for parent in (path, *path.parents)) or not path.is_file():
        raise ValueError(f"capture evidence must be a regular file: {path}")
    return path.read_bytes()


def identity(name: str, data: bytes) -> dict:
    return {"path": name, "size": len(data), "sha256": digest(data)}


def archive_members(path: Path) -> list[dict]:
    result = []
    seen = set()
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            relative(member.name.rstrip("/") if member.isdir() else member.name)
            if member.name in seen or not (member.isdir() or member.isfile()):
                raise ValueError(f"duplicate or unsupported archive member: {path}:{member.name}")
            seen.add(member.name)
            row = {"path": member.name, "kind": "file" if member.isfile() else "directory", "size": member.size}
            if member.isfile():
                row["sha256"] = digest(archive.extractfile(member).read())
            result.append(row)
    return result


def inventory(root: Path) -> dict:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"history links forbidden: {path}")
        if not path.is_file() or path == root / INVENTORY:
            continue
        row = identity(path.relative_to(root).as_posix(), path.read_bytes())
        if path.name.endswith(".tar.gz"):
            row["members"] = archive_members(path)
        rows.append(row)
    return {"schema": 1, "files": rows}


def verify(root: Path, pin: dict) -> dict:
    root = root.absolute()
    raw = read(root, INVENTORY)
    if digest(raw) != pin["inventory_sha256"]:
        raise ValueError("capture history inventory pin mismatch")
    expected = json.loads(raw)
    if expected.get("schema") != 1 or expected.get("source_revision") != pin["source_revision"]:
        raise ValueError("capture history revision or schema mismatch")
    actual = inventory(root)
    if actual["files"] != expected["files"]:
        raise ValueError("capture history missing, corrupt, or unindexed evidence")
    return expected


def current(root: Path) -> dict:
    if (root / UPDATE_JOURNAL).exists():
        raise ValueError("capture publication is pending; run publish_current_capture.py --recover with its explicit history source")
    document = json.loads(read(root, CURRENT))
    if document.get("schema") != 1:
        raise ValueError("unsupported current capture index")
    owners = set()
    for row in document["captures"]:
        key = row["surface"], row["family"], row["implementation"], row["mode"]
        if key in owners:
            raise ValueError(f"duplicate current capture: {key}")
        owners.add(key)
        data = read(root, row["path"])
        if identity(row["path"], data) != {key: row[key] for key in ("path", "size", "sha256")}:
            raise ValueError(f"current capture identity mismatch: {row['path']}")
    return document


def copy_authored(root: Path, destination: Path) -> list[dict]:
    inputs = []
    for path in sorted((root / STORE / "families").glob("*/inputs_and_golden.yaml")):
        name = path.relative_to(root).as_posix()
        data = read(root, name)
        target = destination / path.relative_to(root / STORE)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        inputs.append(identity(name, data))
    return inputs


def stage_unified(root: Path, destination: Path) -> list[dict]:
    document = current(root)
    inputs = copy_authored(root, destination)
    for row in document["captures"]:
        if row["surface"] != "unified":
            continue
        target = destination / relative(row["source"].removeprefix(STORE + "/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(read(root, row["path"]))
    return inputs


def current_identities(document: dict) -> list[dict]:
    return [{"yaml_path": row["path"], **{key: value for key, value in row.items() if key != "path"}}
            for row in document["captures"]]


def stage_current_sources(root: Path, destination: Path) -> None:
    """Expose current YAML to source readers without publication or history retrieval."""
    document = current(root)
    catalog = json.loads(read(root, "conformance/viewer-catalog.json"))
    if catalog.get("capture_generation", {}).get("current") != current_identities(document):
        raise ValueError("current source catalog is stale; regenerate viewer_catalog.py")
    authored = stage_unified(root, destination / STORE)
    catalog["sources"]["original_unified"] = catalog["sources"]["unified"] + document["lineage"]
    catalog["metadata"]["tabs"] = [tab for tab in catalog["metadata"]["tabs"] if tab["suite"] == "unified"]
    if current(root) != document or any(identity(row["path"], read(root, row["path"])) != row for row in authored):
        raise ValueError("current source inputs changed during staging")
    target = destination / "conformance/viewer-catalog.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(encoded(catalog))


@contextmanager
def unified_view(root: Path):
    """Expose logical version identities to existing semantic readers without history I/O."""
    with tempfile.TemporaryDirectory(prefix="current-captures-") as temporary:
        destination = Path(temporary) / "fixtures-unified-v2"
        stage_unified(root, destination)
        (destination / "lineage.json").write_bytes(encoded(current(root)["lineage"]))
        yield destination


def prepare(root: Path, destination: Path, *, include_history: bool = False,
            history_source: Path | None = None, publisher: str | None = None) -> None:
    """Build a private logical source view for the existing catalog and Rust pipeline."""
    if history_source is not None and not include_history:
        raise ValueError("--history-source requires --include-history")
    current_bytes = read(root, CURRENT)
    document = current(root)
    pin = json.loads(read(root, PIN)) if include_history else None
    if include_history:
        if history_source is None:
            raise ValueError("history mode requires an explicit --history-source directory")
        verify(history_source, pin)
    files = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root).decode().split("\0")
    inputs = []
    for name in sorted(set(files) - {"", CURRENT, PIN}):
        if (not ci_publication_cache.is_publication_input(name)
                or name.startswith((STORE + "/", "conformance/fixtures/", "conformance/captures/"))
                or name.startswith("conformance/viewer-") or not (root / name).is_file()):
            continue
        data = read(root, name)
        target = destination / relative(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        lock_helper = os.environ.get("CONFORMANCE_DOCUMENT_LOCK")
        if target.suffix in {".md", ".txt"} and lock_helper:
            # Documentation is also an input to catalog metadata. Use the workspace
            # lease helper for these temporary copies, just like maintained docs.
            subprocess.run([sys.executable, lock_helper,
                            "run", str(target), "--agent", "capture-source-view", "--",
                            sys.executable, "-c", "import pathlib,sys; pathlib.Path(sys.argv[2]).write_bytes(pathlib.Path(sys.argv[1]).read_bytes())",
                            str(root / name), str(target)], check=True, capture_output=True)
        else:
            target.write_bytes(data)
        inputs.append(identity(name, data))
    if include_history:
        for name in (STORE, "conformance/fixtures"):
            shutil.copytree(history_source / name, destination / name)
        manifest = json.loads(read(history_source, "conformance/fixtures-manifest.json"))
        # Authored requests remain owned by this checkout.
        copied_authored = copy_authored(root, destination / STORE)
    else:
        copied_authored = stage_unified(root, destination / STORE)
        manifest = json.loads(read(root, "conformance/fixtures-manifest.json"))
        manifest["shards"] = []
    (destination / "conformance/fixtures-manifest.json").write_bytes(encoded(manifest))
    identities = current_identities(document)
    generation = {"mode": "history" if include_history else "latest", "current": identities,
                  "lineage": document["lineage"], "history_pin": pin}
    (destination / "conformance/capture-generation.json").write_bytes(encoded(generation))
    if current_bytes != read(root, CURRENT):
        raise ValueError("current capture index changed during source preparation")
    guard = {"root": str(root), "inputs": inputs,
             "controls": [identity(CURRENT, read(root, CURRENT))],
             "history_source": str(history_source) if include_history else None, "history_pin": pin}
    guard["inputs"].extend(copied_authored)
    if include_history:
        guard["controls"].append(identity(PIN, read(root, PIN)))
    (destination / "conformance/capture-source-guard.json").write_bytes(encoded(guard))
    command = [sys.executable, str(destination / "conformance/utils/src/viewer_catalog.py"), "--root", str(destination)]
    environment = dict(os.environ)
    if publisher:
        environment["CONFORMANCE_PUBLISHER"] = publisher
    subprocess.run(command, check=True, env=environment)


def verify_prepared(root: Path) -> None:
    guard = json.loads(read(root, "conformance/capture-source-guard.json"))
    original = Path(guard["root"])
    for row in guard["inputs"] + guard["controls"]:
        if identity(row["path"], read(original, row["path"])) != row:
            raise ValueError(f"publication source changed: {row['path']}")
    current(original)
    if guard["history_source"] is not None:
        verify(Path(guard["history_source"]), guard["history_pin"])


def materialize_legacy(root: Path, destination: Path) -> None:
    document = current(root)
    for mode, corpus in LEGACY_CORPORA.items():
        authored = root / "conformance/inputs" / mode
        if authored.is_dir():
            shutil.copytree(authored, destination / corpus / "inputs")
    for row in document["captures"]:
        if row["surface"] == "unified":
            continue
        snapshot = yaml.safe_load(read(root, row["path"]))
        mode, implementation = row["mode"], row["implementation"]
        if any(snapshot[key] != row[key] for key in ("family", "implementation", "version", "surface")):
            raise ValueError("legacy snapshot identity mismatch")
        corpus = destination / LEGACY_CORPORA[mode]
        for name, original in snapshot["documents"].items():
            document = copy.deepcopy(original)
            if mode == "stream":
                for case in document.get("cases", {}).values():
                    for key in ("unavailable", "exception"):
                        if isinstance(case.get(key), dict):
                            value = case.pop(key).get(implementation)
                            if value is not None:
                                case[key] = value
                    for chunk in case.get("chunks", []):
                        chunk["expected"] = (chunk.get("expected") or {}).get(implementation, [])
                        text = chunk.pop("normal_text", {})
                        if isinstance(text, dict) and implementation in text:
                            chunk["normal_text"] = text[implementation]
            folder = corpus if mode == "batch-on-stream" else corpus / f"{implementation}-{row['version']}"
            target = folder / row["family"] / relative(name)
            target.parent.mkdir(parents=True, exist_ok=True)
            data = yaml.safe_dump(document, sort_keys=False, allow_unicode=True, width=4096).encode()
            if target.exists() and target.read_bytes() != data:
                raise ValueError(f"conflicting legacy current documents: {target}")
            target.write_bytes(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--verify-history", type=Path)
    parser.add_argument("--prepare", type=Path)
    parser.add_argument("--stage-current-source", type=Path)
    parser.add_argument("--include-history", action="store_true")
    parser.add_argument("--history-source", type=Path)
    parser.add_argument("--publisher")
    parser.add_argument("--verify-prepared", action="store_true")
    args = parser.parse_args()
    if args.history_source is not None and not args.include_history:
        parser.error("--history-source requires --include-history")
    if args.stage_current_source is not None:
        stage_current_sources(args.root.resolve(), args.stage_current_source.resolve())
    elif args.prepare is not None:
        prepare(args.root.resolve(), args.prepare.resolve(), include_history=args.include_history,
                history_source=args.history_source.resolve() if args.history_source else None, publisher=args.publisher)
    elif args.verify_prepared:
        verify_prepared(args.root)
    elif args.verify_history is not None:
        pin = json.loads(read(args.root, PIN))
        result = verify(args.verify_history, pin)
        print(json.dumps({"verified": True, "files": len(result["files"]), **pin}))
    else:
        print(json.dumps(current(args.root)))


if __name__ == "__main__":
    main()
