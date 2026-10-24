#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Archive immutable capture evidence before publishing one latest YAML."""
import argparse
import base64
import fcntl
import json
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
import tarfile
import tempfile

import yaml

import capture_sources as sources
from capture_versions import capture_release_sort_key
import unified_archives
import unified_history
import migrate_capture_sources


def choose_current(rows, candidate):
    key = tuple(candidate[name] for name in ("surface", "family", "implementation", "mode"))
    previous = next((row for row in rows if tuple(row[name] for name in ("surface", "family", "implementation", "mode")) == key), None)
    if previous is not None and capture_release_sort_key(candidate["version"]) <= capture_release_sort_key(previous["version"]):
        if candidate["version"] == previous["version"] and candidate["sha256"] != previous["sha256"]:
            raise ValueError("a frozen current release cannot be overwritten")
        return list(rows), False
    return [row for row in rows if row is not previous] + [candidate], True


@contextmanager
def capture_lock(root, family, implementation, mode):
    key = sources.digest(sources.encoded([str(root.resolve()), family, implementation, mode]))
    # Bulk jobs have private TMPDIRs; runtime locks must share one namespace.
    directory = Path("/tmp") / f"capture-locks-{os.getuid()}"
    directory.mkdir(mode=0o700, exist_ok=True)
    with (directory / key).open("a+b") as lease:
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX)
        yield


def atomic_write(path, data, *, immutable=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        if sources.read(path.parent, path.name) != data:
            raise ValueError(f"immutable capture evidence differs: {path}")
        return
    fd, temporary = tempfile.mkstemp(prefix=".capture-stage-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        unified_history._fsync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def authored_paths(root, family, mode):
    if mode == "unified":
        return [root / sources.STORE / "families" / family / "inputs_and_golden.yaml"]
    input_mode = "batch" if mode == "batch-on-stream" else mode
    return sorted((root / "conformance/inputs" / input_mode / family).glob("*.yaml"))


def retained_authored(root, history, family):
    name = f"{sources.STORE}/families/{family}/inputs_and_golden.yaml"
    owner = history if (history / name).is_file() else root
    return yaml.load(sources.read(owner, name), Loader=unified_history.StrictLoader)


def finish(root, history, journal):
    pending = json.loads(sources.read(root, journal.relative_to(root).as_posix()))
    if pending["history_source"] != str(history.resolve()):
        raise ValueError("pending capture publication belongs to another history source")
    paths = [path.relative_to(root).as_posix() for path in authored_paths(root, pending["family"], pending["mode"])]
    if paths != [row["path"] for row in pending["authored"]]:
        raise ValueError("authored inputs changed during capture publication")
    for row in pending["authored"]:
        if sources.identity(row["path"], sources.read(root, row["path"])) != row:
            raise ValueError("authored inputs changed during capture publication")
    payloads = []
    for item in pending["files"]:
        if item["owner"] not in {"history", "current"}:
            raise ValueError("invalid pending payload owner")
        data = base64.b64decode(item["data"], validate=True)
        if sources.digest(data) != item["sha256"]:
            raise ValueError("pending capture payload changed")
        parent = history if item["owner"] == "history" else root
        target = parent / sources.relative(item["path"])
        if item["owner"] == "current":
            existing = sources.digest(target.read_bytes()) if target.exists() else None
            if existing not in (pending["previous"], sources.digest(data)):
                raise ValueError("current YAML changed during pending publication")
        payloads.append((item["owner"], target, data))
    evidence = [(target, data) for owner, target, data in payloads if owner == "history"]
    if len(evidence) != 2 or evidence[0][0].parent != evidence[1][0].parent:
        raise ValueError("invalid evidence transaction")
    directory = evidence[0][0].parent
    if directory.parent != history / "additions":
        raise ValueError("invalid evidence transaction destination")
    if directory.exists():
        for target, data in evidence:
            if sources.read(history, target.relative_to(history).as_posix()) != data:
                raise ValueError("immutable capture evidence differs")
    else:
        # Stage outside verified history; one rename publishes archive + descriptor.
        stage = history.parent / f".capture-evidence-{sources.digest(str(root.resolve()).encode())}-{directory.name}"
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir()
        try:
            for target, data in evidence:
                atomic_write(stage / target.name, data)
            directory.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.rename(stage, directory)
            except FileExistsError:
                if any(sources.read(history, target.relative_to(history).as_posix()) != data for target, data in evidence):
                    raise ValueError("immutable capture evidence differs")
            unified_history._fsync_directory(directory.parent)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
    for owner, target, data in payloads:
        if owner == "current":
            atomic_write(target, data)
    journal.unlink()
    unified_history._fsync_directory(journal.parent)


def recover(root, history):
    recovered = False
    for journal in sources.pending_updates(root):
        mode = "unified" if journal.parent.parent == root / sources.STORE / "families" else journal.parent.parent.name
        family = journal.parent.name
        implementation = journal.name.removeprefix(".capture-update-").removesuffix(".json")
        with capture_lock(root, family, implementation, mode):
            if journal.exists():
                finish(root, history, journal)
                recovered = True
    return recovered


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
            if prior is not None:
                prior_document = unified_history.load_yaml(root / prior["path"])
                prior_document.pop(sources.SELECTION, None)
                if unified_history.typed_equal(prior_document, document):
                    continue
            candidate = view / "candidate.yaml"
            candidate.write_text(unified_history.dump_yaml(document))
            results.append(publish(root, history, candidate, identity["family"], identity["implementation"], identity["version"]))
    return results


def publish(root: Path, history: Path, capture: Path, family: str, implementation: str, version: str,
            mode: str = "unified") -> dict:
    if sources.history_pin(root, history)["schema"] == 2:
        raise ValueError("retained history is read-only; historical archival and backfill are deferred to a later PR")
    for component in (family, implementation, mode, version):
        sources.relative(component)
        if "/" in component:
            raise ValueError("capture identity must use single components")
    if mode != "unified" and mode not in sources.LEGACY_CORPORA:
        raise ValueError("unsupported capture mode")
    release_order(version, mode)
    path = sources.latest_path(mode, family, implementation)
    target = root / path
    journal = target.parent / f".capture-update-{implementation}.json"
    with capture_lock(root, family, implementation, mode):
        if journal.exists():
            finish(root, history, journal)
        data = capture.read_bytes()
        if data.startswith(b"capture_selection: "):
            data = data.split(b"\n", 1)[1]
        document = yaml.load(data, Loader=unified_history.StrictLoader)
        # Stable serialization makes independent captures of the same records idempotent.
        data = yaml.safe_dump(document, sort_keys=True, allow_unicode=True, width=4096).encode()
        authored_rows, lineage = [], []
        if mode == "unified":
            authored_name = authored_paths(root, family, mode)[0].relative_to(root).as_posix()
            authored_bytes = sources.read(root, authored_name)
            authored = unified_history._validate_family(root / authored_name, yaml.load(authored_bytes, Loader=unified_history.StrictLoader))
            unified_history._validate_complete_snapshot(capture, document, authored, implementation, version)
            document = sources.independent_snapshot(document, authored.document)
            data = yaml.safe_dump(document, sort_keys=True, allow_unicode=True, width=4096).encode()
            authored_rows.append(sources.identity(authored_name, authored_bytes))
            logical = f"{sources.STORE}/families/{family}/{implementation}-{version}.yaml"
            for member in sorted({record["member"] for record in document["records"].values()}):
                identity = unified_history.source_identity(member.removeprefix(sources.STORE + "/"))
                lineage.append({"path": member, **identity})
        else:
            if any(document.get(key) != value for key, value in
                   (("surface", mode), ("family", family), ("implementation", implementation), ("version", version))):
                raise ValueError("legacy snapshot identity mismatch")
            if not isinstance(document.get("documents"), dict) or not document["documents"]:
                raise ValueError("legacy snapshot requires complete documents")
            logical = f"conformance/captures/{mode}/{family}/{implementation}-{version}.yaml"
            input_sources = authored_paths(root, family, mode)
            if not input_sources or set(document["documents"]) != {source.name for source in input_sources}:
                raise ValueError("legacy snapshot must cover every authored family document")
            for source in input_sources:
                name = source.relative_to(root).as_posix()
                authored_rows.append(sources.identity(name, sources.read(root, name)))
        row = {"source": logical, "surface": mode, "mode": mode, "family": family,
               "implementation": implementation, "version": version}
        selected = sources.selected_bytes(data, row, lineage,
                                          authored=authored.document if mode == "unified" else None)
        previous_data = sources.read(root, path) if target.exists() else None
        previous_selection = json.loads(previous_data.split(b"\n", 1)[0].removeprefix(b"capture_selection: ")) if previous_data else None
        if previous_selection and previous_selection["version"] == version:
            previous_document = yaml.load(previous_data.split(b"\n", 1)[1], Loader=unified_history.StrictLoader)
            comparable = sources.independent_snapshot(document, authored.document, source_member=logical) if mode == "unified" else document
            previous_comparable = sources.independent_snapshot(previous_document, authored.document, source_member=logical) if mode == "unified" else previous_document
            if not unified_history.typed_equal(previous_comparable, comparable):
                raise ValueError("a frozen current release cannot be overwritten")
            return {"archived": False, "current_advanced": False}
        # Check every retained identity, including older backfills and branch additions.
        inventory = sources.verify(history, sources.history_pin(root, history))
        if mode == "unified":
            known = {sources.STORE + "/" + member["path"] if not member["path"].startswith("conformance/") else member["path"]
                     for entry in inventory["files"] for member in entry.get("members", [])}
            for dataset in json.loads(sources.read(history, sources.DATASETS)) if (history / sources.DATASETS).is_file() else []:
                prefix = dataset["root"] + "/"
                known.update(entry["path"].removeprefix(prefix) for entry in inventory["files"]
                             if entry["path"].startswith(prefix + sources.STORE + "/families/"))
            if any(item["path"] not in known and item["path"] != logical for item in lineage):
                raise ValueError("capture references unknown observation lineage")
        if mode == "unified" and (history / sources.DATASETS).is_file():
            with sources.history_view(root, history) as view:
                retained = unified_history.load_store(view / sources.STORE)
                existing = retained.histories.get((family, implementation))
                capture_id = f"{implementation}-{version}"
                if existing is not None and capture_id in existing.captures:
                    original = existing.complete_snapshot(capture_id, source_member=logical)
                    proposed = sources.independent_snapshot(document, authored.document, source_member=logical)
                    if not unified_history.typed_equal(original, proposed):
                        raise ValueError("frozen historical capture differs from the proposed records")
                    return {"archived": False, "current_advanced": False}
        for entry in inventory["files"]:
            for member in entry.get("members", []):
                if member["path"] != logical and not (mode == "unified" and member["path"] == logical.removeprefix(sources.STORE + "/")):
                    continue
                with tarfile.open(history / entry["path"], "r:gz") as archive:
                    original = yaml.load(archive.extractfile(member["path"]).read(), Loader=unified_history.StrictLoader)
                original.pop(sources.SELECTION, None)
                if mode == "unified":
                    original = sources.independent_snapshot(original, retained_authored(root, history, family), source_member=logical)
                proposed = sources.independent_snapshot(document, authored.document, source_member=logical) if mode == "unified" else document
                if not unified_history.typed_equal(original, proposed):
                    raise ValueError("frozen historical capture differs from the proposed records")
                return {"archived": False, "current_advanced": False}
        if mode != "unified":
            validate_legacy_baseline(root, history, candidate=(row, document))
        advanced = previous_selection is None or release_order(version, mode) > release_order(previous_selection["version"], mode)
        compressed = unified_archives.capture_archive_bytes({logical: data})
        group = sources.digest(sources.encoded(row) + data)
        archive_name = f"additions/{group}/{sources.digest(compressed)}.tar.gz"
        archive_row = {**sources.identity(archive_name, compressed),
                       "members": [{**sources.identity(logical, data), "kind": "file"}]}
        descriptor = sources.encoded({"schema": 1, "selection": row, "lineage": lineage, "archive": archive_row})
        descriptor_name = f"additions/{group}/{sources.digest(descriptor)}.json"
        payloads = [("history", archive_name, compressed), ("history", descriptor_name, descriptor)]
        if advanced:
            payloads.append(("current", path, selected))
        pending = {"schema": 1, "history_source": str(history.resolve()), "family": family,
                   "implementation": implementation, "mode": mode, "authored": authored_rows,
                   "previous": sources.digest(previous_data) if previous_data else None,
                   "files": [{"owner": owner, "path": name, "data": base64.b64encode(body).decode(),
                              "sha256": sources.digest(body)} for owner, name, body in payloads]}
        atomic_write(journal, sources.encoded(pending))
        finish(root, history, journal)
        return {"archived": True, "current_advanced": advanced, "source": logical}


def retained_candidates(root, history, evidence):
    for item in evidence["files"]:
        if item["path"].startswith("additions/") and item["path"].endswith(".json"):
            descriptor = json.loads(sources.read(history, item["path"]))
            with tarfile.open(history / descriptor["archive"]["path"], "r:gz") as archive:
                data = archive.extractfile(descriptor["selection"]["source"]).read()
            yield descriptor, data
    store = history / sources.STORE
    if (history / sources.DATASETS).is_file():
        with sources.history_view(root, history) as view:
            retained = unified_history.load_store(view / sources.STORE)
            for (family, implementation), capture_history in retained.histories.items():
                capture_id = capture_history.ordered_capture_ids()[-1]
                version = capture_history.captures[capture_id]["runtime_version"]
                logical = f"{sources.STORE}/families/{family}/{capture_id}.yaml"
                row = {"family": family, "implementation": implementation, "version": version,
                       "surface": "unified", "mode": "unified", "source": logical}
                lineage = [{"path": logical, **unified_history.source_identity(logical.removeprefix(sources.STORE + "/"))}]
                yield {"selection": row, "lineage": lineage}, unified_history.dump_yaml(capture_history.complete_snapshot(capture_id, source_member=logical), shared_metadata=True).encode()
    elif store.exists():
        resources = unified_archives.inventory(store)
        for resource, data in unified_archives.read_sources(store, resources):
            identity = unified_history.source_identity(resource["path"])
            if identity["kind"] != "history":
                continue
            row = {key: identity[key] for key in ("family", "implementation", "version")}
            row.update(surface="unified", mode="unified", source=sources.STORE + "/" + resource["path"])
            document = yaml.load(data, Loader=unified_history.StrictLoader)
            lineage = [{"path": name, **unified_history.source_identity(name.removeprefix(sources.STORE + "/"))}
                       for name in sorted({record["member"] for record in document["records"].values()})]
            yield {"selection": row, "lineage": lineage}, data
    with tempfile.TemporaryDirectory(prefix="baseline-current-") as temporary:
        target = Path(temporary)
        baseline = migrate_capture_sources.convert(target, history, inventory=evidence, all_versions=True)
        for row in baseline["rows"]:
            data = sources.read(target, row["path"]).split(b"\n", 1)[1]
            yield {"selection": row, "lineage": []}, data


def validate_legacy_baseline(root, history, *, candidate=None):
    evidence = sources.verify(history, sources.history_pin(root, history))
    proposed = []
    if candidate is not None:
        proposed.append(candidate)
    else:
        for entry in evidence["files"]:
            if not entry["path"].startswith("additions/") or not entry["path"].endswith(".json"):
                continue
            descriptor = json.loads(sources.read(history, entry["path"]))
            if descriptor["selection"]["mode"] == "unified":
                continue
            with tarfile.open(history / descriptor["archive"]["path"], "r:gz") as archive:
                data = archive.extractfile(descriptor["selection"]["source"]).read()
            proposed.append((descriptor["selection"], yaml.load(data, Loader=unified_history.StrictLoader)))
    by_identity = {}
    for selection, document in proposed:
        key = tuple(selection[name] for name in ("mode", "family", "implementation", "version"))
        if key in by_identity and not unified_history.typed_equal(by_identity[key], document):
            raise ValueError("conflicting frozen legacy release evidence; explicit correction decision required")
        by_identity[key] = document
    with tempfile.TemporaryDirectory(prefix="baseline-legacy-") as temporary:
        target = Path(temporary)
        baseline = migrate_capture_sources.convert(target, history, inventory=evidence, all_versions=True)
        originals = {(row["mode"], row["family"], row["implementation"], row["version"]): row for row in baseline["rows"]}
        for selection, document in proposed:
            key = tuple(selection[name] for name in ("mode", "family", "implementation", "version"))
            if key not in originals:
                continue
            original = unified_history.load_yaml(target / originals[key]["path"])
            original.pop(sources.SELECTION, None)
            if not unified_history.typed_equal(original, document):
                raise ValueError("conflicting frozen legacy release evidence; explicit correction decision required")


def reconcile(root, history):
    """Rebuild latest outputs from the immutable baseline and both branches' evidence."""
    evidence = sources.verify(history, sources.history_pin(root, history))
    grouped = {}
    for descriptor, data in retained_candidates(root, history, evidence):
        row = descriptor["selection"]
        key = row["mode"], row["family"], row["implementation"]
        versions = grouped.setdefault(key, {})
        if row["version"] in versions:
            previous = yaml.load(versions[row["version"]][1], Loader=unified_history.StrictLoader)
            current = yaml.load(data, Loader=unified_history.StrictLoader)
            previous.pop(sources.SELECTION, None)
            current.pop(sources.SELECTION, None)
            if row["mode"] == "unified":
                authored = retained_authored(root, history, row["family"])
                previous = sources.independent_snapshot(previous, authored, source_member=row["source"])
                current = sources.independent_snapshot(current, authored, source_member=row["source"])
            if not unified_history.typed_equal(previous, current):
                raise ValueError("conflicting frozen release evidence; explicit correction decision required")
        versions[row["version"]] = descriptor, data
    replacements = []
    for (mode, family, implementation), versions in sorted(grouped.items()):
        newest = max(versions, key=lambda version: release_order(version, mode))
        descriptor, data = versions[newest]
        replacements.append((mode, family, implementation, descriptor, data))
    for mode, family, implementation, descriptor, data in replacements:
        path = root / sources.latest_path(mode, family, implementation)
        with capture_lock(root, family, implementation, mode):
            if list(path.parent.glob(".capture-update-*.json")):
                raise ValueError("capture publication is pending; recover before reconciliation")
            if path.exists() and path.read_bytes().startswith(b"capture_selection: "):
                selection = json.loads(path.read_bytes().split(b"\n", 1)[0].removeprefix(b"capture_selection: "))
                if release_order(selection["version"], mode) > release_order(descriptor["selection"]["version"], mode):
                    continue
            authored = retained_authored(root, history, family) if mode == "unified" else None
            atomic_write(path, sources.selected_bytes(data, descriptor["selection"], descriptor["lineage"],
                                                     authored=authored))
    return {"reconciled": len(grouped)}


def release_order(version, mode):
    if mode == "batch-on-stream" and version == "Dynamo parser v2":
        return ((), 0, "")
    token = version.split()[0].lstrip("v") if mode == "batch-on-stream" else version
    return capture_release_sort_key(token)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--history-source", type=Path)
    parser.add_argument("--recover", action="store_true")
    parser.add_argument("--reconcile", action="store_true")
    parser.add_argument("--validate-history", action="store_true")
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--family")
    parser.add_argument("--implementation")
    parser.add_argument("--version")
    parser.add_argument("--mode", default="unified", choices=["unified", *sources.LEGACY_CORPORA])
    args = parser.parse_args()
    root = args.root.resolve()
    history = args.history_source.resolve() if args.history_source else root / sources.HISTORIC
    if args.validate_history:
        validate_legacy_baseline(root, history)
        print(json.dumps({"verified": True}))
    elif args.recover:
        print(json.dumps({"recovered": recover(root, history)}))
    elif args.reconcile:
        print(json.dumps(reconcile(root, history)))
    else:
        if args.capture is None or not all((args.family, args.implementation, args.version)):
            parser.error("publication requires --capture, --family, --implementation, and --version")
        print(json.dumps(publish(root, history, args.capture.resolve(), args.family, args.implementation, args.version, args.mode)))


if __name__ == "__main__":
    main()
