#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned capture evidence and local current snapshots, independent of publication."""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
import tempfile

import yaml
import yaml_fast  # noqa: F401
import ci_publication_cache
import unified_archives
from viewer_codec import typed_equal

PIN = "conformance/historic-baseline.json"
HISTORIC = "conformance/historic"
SELECTION = "capture_selection"
INVENTORY = "capture-inventory.json"
UPDATE_JOURNAL = "conformance/.capture-update.json"
STORE = "conformance/fixtures-unified-v2"
DATASETS = "conformance/history-datasets.json"
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


def inventory(root: Path, *, baseline_only: bool = False) -> dict:
    rows = []
    for path in sorted(root.rglob("*")):
        if baseline_only and path.relative_to(root).parts[0] == "additions":
            continue
        if path.is_symlink():
            raise ValueError(f"history links forbidden: {path}")
        if not path.is_file() or path == root / INVENTORY:
            continue
        row = identity(path.relative_to(root).as_posix(), path.read_bytes())
        if path.name.endswith(".tar.gz"):
            row["members"] = archive_members(path)
        rows.append(row)
    return {"schema": 1, "files": rows}


def history_pin(root: Path, history: Path | None = None) -> dict:
    history = history or default_history(root)
    if not (history / INVENTORY).is_file():
        evidence = retained_inventory(history)
        return {"schema": 2, "source_revision": evidence["source_revision"],
                "inventory_sha256": digest(encoded(evidence))}
    bundled = history.resolve() == default_history(root).resolve()
    if bundled and (root / PIN).is_file():
        return json.loads(read(root, PIN))
    history = history or root / HISTORIC
    raw = read(history, INVENTORY)
    baseline = json.loads(raw)
    return {"schema": 1, "source_revision": baseline["source_revision"], "inventory_sha256": digest(raw)}


def verify(root: Path, pin: dict | None = None) -> dict:
    root = root.absolute()
    if not (root / INVENTORY).is_file():
        evidence = retained_inventory(root)
        if pin is not None and digest(encoded(evidence)) != pin["inventory_sha256"]:
            raise ValueError("retained history changed during publication")
        return evidence
    raw = read(root, INVENTORY)
    pin = pin or history_pin(root, root)
    if digest(raw) != pin["inventory_sha256"]:
        raise ValueError("capture history inventory pin mismatch")
    expected = json.loads(raw)
    if expected.get("schema") != 1 or expected.get("source_revision") != pin["source_revision"]:
        raise ValueError("capture history revision or schema mismatch")
    if (root / "additions").is_symlink():
        raise ValueError("history links forbidden")
    actual = inventory(root, baseline_only=True)
    if actual["files"] != expected["files"]:
        raise ValueError("capture history missing, corrupt, or unindexed evidence")
    additions = []
    # A complete evidence directory is installed with one rename. Snapshot the
    # directory names once so independent publishers do not invalidate readers.
    for directory in sorted((root / "additions").glob("*")):
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("unindexed capture evidence")
        descriptors = sorted(directory.glob("*.json"))
        if len(descriptors) != 1:
            raise ValueError("missing capture descriptor")
        path = descriptors[0]
        descriptor_bytes = read(root, path.relative_to(root).as_posix())
        descriptor = json.loads(descriptor_bytes)
        if path.stem != digest(descriptor_bytes) or descriptor.get("schema") != 1:
            raise ValueError("invalid capture descriptor identity")
        archive = descriptor["archive"]
        relative(archive["path"])
        archive_path = root / archive["path"]
        if archive_path.parent != directory or set(directory.iterdir()) != {path, archive_path}:
            raise ValueError("capture history missing, corrupt, or unindexed evidence")
        measured = {**identity(archive["path"], read(root, archive["path"])), "members": archive_members(archive_path)}
        if measured != archive:
            raise ValueError("capture history missing, corrupt, or unindexed evidence")
        additions.extend([identity(path.relative_to(root).as_posix(), descriptor_bytes), measured])
    combined = sorted(expected["files"] + additions, key=lambda row: row["path"])
    return {**expected, "files": combined}


def retained_inventory(root: Path) -> dict:
    """Inventory existing history in place, without expanding or rewriting archives."""
    paths = sorted(path for path in (root / STORE / "families").glob("*/*.yaml")
                   if path.name == "inputs_and_golden.yaml" or "-" in path.stem)
    paths.extend(sorted((root / "conformance/fixtures").rglob("*.tar.gz")))
    paths.append(root / "conformance/fixtures-manifest.json")
    rows = [identity(path.relative_to(root).as_posix(), read(root, path.relative_to(root).as_posix()))
            for path in paths]
    if not any("-" in Path(row["path"]).stem and row["path"].endswith(".yaml") for row in rows):
        raise ValueError("no retained historical captures")
    return {"schema": 2, "source_revision": "retained-paths", "files": sorted(rows, key=lambda row: row["path"])}


def latest_unified_paths(root: Path):
    """Discover latest files by name without opening any versioned capture."""
    return sorted(path for path in (root / STORE / "families").glob("*/*.yaml")
                  if path.stem != "inputs_and_golden" and "-" not in path.stem)


def default_history(root: Path) -> Path:
    if any("-" in path.stem for path in (root / STORE / "families").glob("*/*.yaml")):
        return root
    return root / HISTORIC


def latest_path(mode: str, family: str, implementation: str) -> str:
    if mode == "unified":
        return f"{STORE}/families/{family}/{implementation}.yaml"
    return f"conformance/captures/{mode}/{family}/{implementation}.yaml"


def pending_updates(root: Path):
    return sorted([*(root / STORE / "families").glob("*/.capture-update-*.json"),
                   *(root / "conformance/captures").glob("*/*/.capture-update-*.json")])


def is_current(root: Path) -> bool:
    return bool(latest_unified_paths(root)) or any((root / "conformance/captures").glob("*/*/*.yaml"))


def independent_snapshot(document: dict, authored: dict, *, source_member: str | None = None) -> dict:
    """Seal current references so later authored edits cannot change a capture."""
    document = copy.deepcopy(document)
    for case_id, wrapped in document["records"].items():
        if source_member is not None:
            wrapped["member"] = source_member
        value = wrapped["value"]
        stimulus = value["stimulus"]
        if stimulus.get("ref") == "current":
            request = authored["cases"][case_id]["request"]
            value["stimulus"] = {**{key: item for key, item in stimulus.items() if key != "ref"},
                                 "inline": copy.deepcopy(request)}
    return document


def selected_bytes(data: bytes, row: dict, lineage: list[dict], *, authored: dict | None = None) -> bytes:
    # Latest downloads use the supplied complete snapshot; producer metadata stays intact.
    body = data.split(b"\n", 1)[1] if data.startswith(b"capture_selection: ") else data
    if authored is not None:
        body = yaml.safe_dump(independent_snapshot(yaml.safe_load(body), authored, source_member=row["source"]),
                              sort_keys=True, allow_unicode=True, width=4096).encode()
    selection = {key: value for key, value in row.items() if key not in {"path", "size", "sha256"}}
    selection.update(payload_sha256=digest(body), lineage=sorted(lineage, key=lambda item: item["path"]))
    return b"capture_selection: " + encoded(selection).rstrip(b"\n") + b"\n" + body


def current(root: Path) -> dict:
    if pending_updates(root):
        raise ValueError("capture publication is pending; run publish_current_capture.py --recover")
    captures, lineage = [], {}
    latest = latest_unified_paths(root)
    for path in sorted([*latest, *(root / "conformance/captures").glob("*/*/*.yaml")]):
        if list(path.parent.glob(".capture-update-*.json")):
            raise ValueError("capture publication is pending; run publish_current_capture.py --recover")
        name = path.relative_to(root).as_posix()
        data = read(root, name)
        if not data.startswith(b"capture_selection: "):
            raise ValueError(f"missing capture selection: {name}")
        header, body = data.split(b"\n", 1)
        selection = json.loads(header.removeprefix(b"capture_selection: "))
        if selection.pop("payload_sha256") != digest(body):
            raise ValueError(f"current capture identity mismatch: {name}")
        for item in selection.pop("lineage"):
            previous = lineage.setdefault(item["path"], item)
            if previous != item:
                raise ValueError("conflicting observation lineage")
        row = {**selection, **identity(name, data)}
        surface = "unified" if path in latest else path.parent.parent.name
        if (row["surface"] != surface or row["family"] != path.parent.name
                or row["implementation"] != path.stem or row["mode"] != row["surface"]):
            raise ValueError(f"current capture identity mismatch: {name}")
        if row["surface"] == "unified":
            expected = f"{STORE}/families/{row['family']}/{row['implementation']}-{row['version']}.yaml"
            if row["source"] != expected:
                raise ValueError("selected release source mismatch")
        captures.append(row)
    if not captures:
        raise ValueError("no current capture YAML found")
    return {"schema": 1, "captures": captures, "lineage": sorted(lineage.values(), key=lambda item: item["path"])}


def authored_sources(root: Path):
    for path in sorted((root / STORE / "families").glob("*/inputs_and_golden.yaml")):
        name = path.relative_to(root).as_posix()
        yield name, read(root, name)


def copy_authored(root: Path, destination: Path) -> list[dict]:
    inputs = []
    for name, data in authored_sources(root):
        target = destination / relative(name.removeprefix(STORE + "/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        inputs.append(identity(name, data))
    return inputs


def checked_bytes(root: Path, row: dict, *, label: str = "current capture") -> bytes:
    data = read(root, row["path"])
    if identity(row["path"], data) != {key: row[key] for key in ("path", "size", "sha256")}:
        raise ValueError(f"{label} changed during staging")
    return data


def stage_unified(root: Path, destination: Path, *, document: dict | None = None) -> list[dict]:
    document = document or current(root)
    inputs = copy_authored(root, destination)
    for row in document["captures"]:
        if row["surface"] != "unified":
            continue
        target = destination / relative(row["source"].removeprefix(STORE + "/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(checked_bytes(root, row))
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
    authored = stage_unified(root, destination / STORE, document=document)
    catalog["sources"]["original_unified"] = catalog["sources"]["unified"] + document["lineage"]
    catalog["metadata"]["tabs"] = [tab for tab in catalog["metadata"]["tabs"] if tab["suite"] == "unified"]
    if current(root) != document or [identity(name, data) for name, data in authored_sources(root)] != authored:
        raise ValueError("current source inputs changed during staging")
    target = destination / "conformance/viewer-catalog.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(encoded(catalog))


@contextmanager
def unified_view(root: Path):
    """Expose logical version identities to existing semantic readers without history I/O."""
    with tempfile.TemporaryDirectory(prefix="current-captures-") as temporary:
        destination = Path(temporary) / "fixtures-unified-v2"
        document = current(root)
        authored = stage_unified(root, destination, document=document)
        (destination / "lineage.json").write_bytes(encoded(document["lineage"]))
        if current(root) != document or [identity(name, data) for name, data in authored_sources(root)] != authored:
            raise ValueError("current capture changed during staging")
        yield destination


def publication_input_paths(root: Path) -> list[str]:
    files = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "--exclude=conformance/historic/", "-z"], cwd=root).decode().split("\0")
    return [name for name in sorted(set(files) - {"", PIN})
            if ci_publication_cache.is_publication_input(name)
            and not name.startswith((STORE + "/", "conformance/fixtures/", "conformance/captures/", HISTORIC + "/", "conformance/viewer-"))
            and (root / name).is_file()]


def publication_inputs_digest(root: Path, *, include_history: bool, history_source: Path | None = None) -> str:
    """Bind reusable reports to the same bytes admitted by source preparation."""
    inputs = [identity(name, read(root, name)) for name in publication_input_paths(root)]
    inputs.extend(identity(name, data) for name, data in authored_sources(root))
    controls = [identity(row["path"], read(root, row["path"])) for row in current(root)["captures"]]
    if include_history:
        controls.append(identity("history-catalog", encoded(verify(history_source or default_history(root)))))
    # Rust preserves the parsed guard key order, then inserts inputs before controls.
    guard = json.loads(encoded({"inputs": inputs, "controls": controls}))
    value = {"inputs": guard["inputs"], "controls": guard["controls"]}
    return digest((json.dumps(value, separators=(",", ":"), ensure_ascii=False) + "\n").encode())


def stage_history(root: Path, destination: Path, history: Path, *, separate: bool = True) -> dict:
    evidence = verify(history, history_pin(root, history))
    if evidence["schema"] == 2:
        # Latest complete snapshots and historical sparse overlays have separate
        # conversion owners; neither may become a parent of the other.
        prefix = "conformance/history-datasets/retained" if separate else ""
        for row in evidence["files"]:
            name = row["path"]
            target = destination / (prefix if name.startswith(STORE + "/") else "") / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(checked_bytes(history, row, label="retained history"))
        if separate:
            (destination / DATASETS).write_bytes(encoded([{"id": "retained", "root": prefix}]))
        (destination / "conformance/history-source-identity.json").write_bytes(encoded(evidence))
        return json.loads(read(history, "conformance/fixtures-manifest.json"))
    admitted = {row["path"]: row for row in evidence["files"]}

    def admitted_bytes(name):
        return checked_bytes(history, admitted[name], label="history")

    # Copy only admitted members so transient unindexed files cannot enter the view.
    for row in evidence["files"]:
        if row["path"].startswith((STORE + "/", "conformance/fixtures/", "conformance/history-datasets/")) or row["path"] == DATASETS:
            target = destination / relative(row["path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(admitted_bytes(row["path"]))
    manifest = json.loads(admitted_bytes("conformance/fixtures-manifest.json"))
    descriptors = [json.loads(admitted_bytes(row["path"])) for row in evidence["files"]
                   if row["path"].startswith("additions/") and row["path"].endswith(".json")]
    if any(descriptor["selection"]["mode"] != "unified" for descriptor in descriptors):
        subprocess.run([sys.executable, str(Path(__file__).with_name("publish_current_capture.py")),
                        "--root", str(root), "--history-source", str(history), "--validate-history"], check=True, capture_output=True)
    baseline = {row["path"]: row for row in unified_archives.inventory(destination / STORE)}
    for descriptor in descriptors:
        archive_row = descriptor["archive"]
        archive_bytes = admitted_bytes(archive_row["path"])
        if identity(archive_row["path"], archive_bytes) != {key: archive_row[key] for key in ("path", "size", "sha256")}:
            raise ValueError("history changed during staging")
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
            for member in descriptor["archive"]["members"]:
                name = relative(member["path"])
                data = archive.extractfile(name).read()
                if descriptor["selection"]["mode"] != "unified":
                    snapshot = yaml.safe_load(data)
                    compressed = unified_archives.capture_archive_bytes(dict(legacy_payloads(snapshot, descriptor["selection"])))
                    selection = descriptor["selection"]
                    derived_name = f"additions/{digest(compressed)}/{LEGACY_CORPORA[selection['mode']]}/{selection['implementation']}-{selection['version']}.tar.gz"
                    target = destination / "conformance/fixtures" / derived_name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(compressed)
                    manifest["shards"].append({"path": derived_name, "size": len(compressed), "sha256": digest(compressed)})
                    continue
                baseline_resource = baseline.get(name.removeprefix(STORE + "/"))
                if baseline_resource is not None:
                    original = unified_archives.read_source(destination / STORE, baseline_resource)
                    authored = yaml.safe_load(read(destination, f"{STORE}/families/{descriptor['selection']['family']}/inputs_and_golden.yaml"))
                    if not typed_equal(independent_snapshot(yaml.safe_load(original), authored),
                                       independent_snapshot(yaml.safe_load(data), authored)):
                        raise ValueError("conflicting frozen release evidence; explicit correction decision required")
                    continue
                target = destination / name
                if target.exists() and target.read_bytes() != data:
                    old = yaml.safe_load(target.read_bytes())
                    authored = yaml.safe_load(read(destination, f"{STORE}/families/{descriptor['selection']['family']}/inputs_and_golden.yaml"))
                    if not typed_equal(independent_snapshot(old, authored),
                                       independent_snapshot(yaml.safe_load(data), authored)):
                        raise ValueError("conflicting frozen release evidence; explicit correction decision required")
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
    # The combined manifest is generated privately; baseline interpretation stays immutable.
    resources = {row["path"]: {key: row[key] for key in ("path", "sha256", "size")}
                 for row in unified_archives.inventory(destination / STORE)}
    updates = dict(resources)
    if (destination / DATASETS).is_file():
        for dataset in json.loads(read(destination, DATASETS)):
            prefix = relative(dataset["root"])
            for row in unified_archives.inventory(destination / prefix / STORE):
                if not row["path"].endswith("/inputs_and_golden.yaml"):
                    resources[row["path"]] = {key: row[key] for key in ("path", "sha256", "size")}
        resources.update(updates)
    resources = [resources[name] for name in sorted(resources)]
    combined_digest = digest(json.dumps(resources, sort_keys=True, separators=(",", ":")).encode())
    size = sum(row["size"] for row in resources)
    for shard in manifest["shards"]:
        if shard.get("format") == "unified-history":
            shard.update(sha256=combined_digest, size=size)
    (destination / "conformance/fixtures-manifest.json").write_bytes(encoded(manifest))
    (destination / "conformance/history-source-identity.json").write_bytes(encoded(evidence))
    return manifest


@contextmanager
def history_view(root: Path, history: Path):
    with tempfile.TemporaryDirectory(prefix="capture-history-") as temporary:
        destination = Path(temporary)
        if (history / DATASETS).is_file():
            prepare(root, destination, include_history=True, history_source=history, retained_history_only=True)
        else:
            stage_history(root, destination, history, separate=False)
        yield destination


def publisher_binary(root: Path, configured: str | None = None) -> Path:
    configured = configured or os.environ.get("CONFORMANCE_PUBLISHER")
    if configured:
        return Path(configured).resolve(strict=True)
    build_root = Path(os.environ.get("CARGO_TARGET_DIR", root / "target")).resolve()
    subprocess.run(["cargo", "build", "--locked", "--release", "-p", "conformance-publisher"],
                   cwd=root, check=True)
    return build_root / "release/conformance-publish"


def prepare(root: Path, destination: Path, *, include_history: bool = False,
            history_source: Path | None = None, publisher: str | None = None,
            cache_dir: Path | None = None, retained_history_only: bool = False) -> None:
    """Build a private logical source view for the existing catalog and Rust pipeline."""
    if history_source is not None and not include_history:
        raise ValueError("--history-source requires --include-history")
    if retained_history_only and not include_history:
        raise ValueError("retained history requires --include-history")
    # Admin recovery and disjoint publishers cannot depend on mutable latest YAML.
    document = {"captures": [], "lineage": []} if retained_history_only else current(root)
    history_source = history_source or default_history(root) if include_history else None
    pin = history_pin(root, history_source) if include_history else None
    if include_history:
        verify(history_source, pin)
    inputs = []
    input_paths = publication_input_paths(root)
    for name in input_paths:
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
        if read(destination, name) != data:
            raise ValueError("publication input changed during staging")
        inputs.append(identity(name, data))
    if include_history:
        # Stage latest first; sparse retained history is a separate dataset.
        if pin["schema"] == 2:
            stage_unified(root, destination / STORE, document=document)
        manifest = stage_history(root, destination, history_source)
        # A historical `ref: current` names the authored document retained with
        # that evidence, not a request subsequently edited in this checkout.
        copied_authored = (copy_authored(root, destination / STORE)
                           if (destination / DATASETS).is_file() else
                           [identity(name, data) for name, data in authored_sources(root)])
    else:
        copied_authored = stage_unified(root, destination / STORE, document=document)
        manifest = json.loads(read(destination, "conformance/fixtures-manifest.json"))
        manifest["shards"] = []
    # The publisher emits this report from the shared index template. Keep an
    # asset-graph anchor without consuming a previously generated local report.
    (destination / "conformance/CONFORMANCE_v2.html").write_bytes(b"<!doctype html><head></head><body></body>\n")
    (destination / "conformance/fixtures-manifest.json").write_bytes(encoded(manifest))
    if include_history and pin["schema"] == 2:
        # The publication inventory describes the selected logical union, while
        # the source guard below retains the unchanged physical evidence hashes.
        resources = {row["path"].removeprefix(STORE + "/"): {"path": row["path"].removeprefix(STORE + "/"),
                     "size": row["size"], "sha256": row["sha256"]}
                     for row in verify(history_source, pin)["files"] if row["path"].startswith(STORE + "/")}
        for path in (destination / STORE / "families").glob("*/*.yaml"):
            name = path.relative_to(destination / STORE).as_posix()
            resources[name] = identity(name, path.read_bytes())
        rows = [resources[name] for name in sorted(resources)]
        for shard in manifest["shards"]:
            if shard.get("format") == "unified-history":
                shard.update(sha256=digest(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()),
                             size=sum(row["size"] for row in rows))
        (destination / "conformance/fixtures-manifest.json").write_bytes(encoded(manifest))
    identities = current_identities(document)
    generation = {"mode": "history" if include_history else "latest", "current": identities,
                  "lineage": document["lineage"], "history_pin": pin}
    (destination / "conformance/capture-generation.json").write_bytes(encoded(generation))
    if not retained_history_only and document != current(root):
        raise ValueError("current capture YAML changed during source preparation")
    guard = {"root": str(root), "inputs": inputs,
             "controls": [identity(row["path"], read(root, row["path"])) for row in document["captures"]],
             "current": identities, "publication_paths": input_paths,
             "authored_paths": [row["path"] for row in copied_authored], "history_source": str(history_source) if include_history else None, "history_pin": pin}
    guard["inputs"].extend(copied_authored)
    if retained_history_only:
        del guard["current"]
    if include_history:
        guard["controls"].append(identity("history-catalog", read(destination, "conformance/history-source-identity.json")))
    (destination / "conformance/capture-source-guard.json").write_bytes(encoded(guard))
    command = [sys.executable, str(destination / "conformance/utils/src/viewer_catalog.py"), "--root", str(destination)]
    environment = dict(os.environ)
    cache_dir = Path(cache_dir or environment.get("CONFORMANCE_CONVERSION_CACHE", root / "build/conformance-cache")).resolve()
    environment["CONFORMANCE_CONVERSION_CACHE"] = str(cache_dir)
    (destination / "conformance/history-reader-cache.json").write_bytes(encoded({"directory": str(cache_dir)}))
    environment["CONFORMANCE_PUBLISHER"] = str(publisher_binary(root, publisher))
    subprocess.run(command, check=True, env=environment)


def verify_prepared(root: Path) -> None:
    guard = json.loads(read(root, "conformance/capture-source-guard.json"))
    original = Path(guard["root"])
    for row in guard["inputs"] + guard["controls"]:
        data = encoded(verify(Path(guard["history_source"]), guard["history_pin"])) if row["path"] == "history-catalog" else read(original, row["path"])
        if identity(row["path"], data) != row:
            raise ValueError(f"publication source changed: {row['path']}")
    if "publication_paths" in guard and publication_input_paths(original) != guard["publication_paths"]:
        raise ValueError("publication source membership changed")
    if "authored_paths" in guard and [name for name, _data in authored_sources(original)] != guard["authored_paths"]:
        raise ValueError("publication authored source membership changed")
    if "current" in guard and current_identities(current(original)) != guard["current"]:
        raise ValueError("publication current capture discovery changed")
    if guard["history_source"] is not None:
        verify(Path(guard["history_source"]), guard["history_pin"])


def materialize_legacy(root: Path, destination: Path, *, document: dict, authored: list[dict]) -> None:
    for mode, corpus in LEGACY_CORPORA.items():
        prefix = f"conformance/inputs/{mode}/"
        for row in authored:
            if not row["path"].startswith(prefix):
                continue
            target = destination / corpus / "inputs" / relative(row["path"].removeprefix(prefix))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(checked_bytes(root, row, label="authored inputs"))
    for row in document["captures"]:
        if row["surface"] == "unified":
            continue
        snapshot = yaml.safe_load(checked_bytes(root, row))
        for name, data in legacy_payloads(snapshot, row):
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.read_bytes() != data:
                raise ValueError(f"conflicting legacy current documents: {target}")
            target.write_bytes(data)


def legacy_payloads(snapshot, row):
    mode, implementation = row["mode"], row["implementation"]
    if any(snapshot[key] != row[key] for key in ("family", "implementation", "version", "surface")):
        raise ValueError("legacy snapshot identity mismatch")
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
        corpus = Path(LEGACY_CORPORA[mode])
        folder = corpus if mode == "batch-on-stream" else corpus / f"{implementation}-{row['version']}"
        name = (folder / row["family"] / relative(name)).as_posix()
        yield name, yaml.safe_dump(document, sort_keys=False, allow_unicode=True, width=4096).encode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--verify-history", type=Path)
    parser.add_argument("--prepare", type=Path)
    parser.add_argument("--stage-current-source", type=Path)
    parser.add_argument("--include-history", action="store_true")
    parser.add_argument("--history-source", type=Path)
    parser.add_argument("--publisher")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--verify-prepared", action="store_true")
    parser.add_argument("--has-latest", action="store_true")
    args = parser.parse_args()
    if args.history_source is not None and not args.include_history:
        parser.error("--history-source requires --include-history")
    if args.has_latest:
        print(json.dumps(is_current(args.root)))
    elif args.stage_current_source is not None:
        stage_current_sources(args.root.resolve(), args.stage_current_source.resolve())
    elif args.prepare is not None:
        prepare(args.root.resolve(), args.prepare.resolve(), include_history=args.include_history,
                history_source=args.history_source.resolve() if args.history_source else None, publisher=args.publisher, cache_dir=args.cache_dir)
    elif args.verify_prepared:
        verify_prepared(args.root)
    elif args.verify_history is not None:
        pin = history_pin(args.root, args.verify_history)
        result = verify(args.verify_history, pin)
        print(json.dumps({"verified": True, "files": len(result["files"]), "catalog_sha256": digest(encoded(result)), **pin}))
    else:
        print(json.dumps(current(args.root)))


if __name__ == "__main__":
    main()
