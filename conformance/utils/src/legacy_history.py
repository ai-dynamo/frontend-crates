#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reviewable YAML storage for the four non-Unified conformance corpora.

Each family has one shared input document and one document per capture layer.
The existing sparse archive layers stay sparse; materialization recreates the
loose layout consumed by the existing resolvers and Rust fixture harnesses.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import tarfile
from collections import defaultdict
from pathlib import Path

import yaml
from fixture_corpus import split_sel, version_key
from fixture_disposition import LEGACY_HISTORY_FORMAT, LEGACY_HISTORY_PATH


CORPORA = {
    "batch": "toolcalling/fixtures-batch-v1",
    "stream": "toolcalling/fixtures-stream-v1",
    "reasoning": "reasoning/fixtures-v1",
    "batch_on_stream": "toolcalling/fixtures-batch-on-stream-v1",
}
HISTORY_PATH = LEGACY_HISTORY_PATH
HISTORY_FORMAT = LEGACY_HISTORY_FORMAT
SHARED_NAME = "inputs_and_golden.yaml"
SNAPSHOT_NAME = "legacy-snapshot"
COMPONENT_RE = re.compile(r"^[A-Za-z0-9_.+-]+$")
CAPTURE_RE = re.compile(
    r"^[A-Za-z0-9_]+-\d+(?:\.\d+){2,}"
    r"(?:\.(?:post|patch)\d+|\+[A-Za-z0-9_.-]+|-[A-Za-z0-9.-]+)?$"
)


def _component(value: str) -> str:
    if not COMPONENT_RE.fullmatch(value) or value in {".", ".."}:
        raise ValueError(f"unsafe fixture path component: {value!r}")
    return value


def _family_dir(store: Path, corpus: str, family: str) -> Path:
    return store / corpus / "families" / _component(family)


def _store_file(store: Path, corpus: str, family: str, capture: str) -> Path:
    name = SHARED_NAME if capture == "inputs" else f"{_component(capture)}.yaml"
    return _family_dir(store, corpus, family) / name


def _load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return value


def _validate_record(store: Path, path: Path, record: dict) -> dict:
    corpus = record.get("corpus")
    family = record.get("family")
    capture = record.get("capture")
    if (not isinstance(corpus, str) or corpus not in CORPORA
            or not isinstance(family, str) or not isinstance(capture, str)):
        raise ValueError(f"legacy history identity mismatch: {path}")
    if corpus == "batch_on_stream":
        valid_capture = capture == "inputs" or CAPTURE_RE.fullmatch(capture) is not None or re.fullmatch(r"[A-Za-z0-9_]+-unversioned", capture) is not None
    elif capture == "inputs":
        valid_capture = True
    else:
        valid_capture = CAPTURE_RE.fullmatch(capture) is not None
    if not valid_capture or _store_file(store, corpus, family, capture) != path:
        raise ValueError(f"legacy history identity mismatch: {path}")
    documents = record.get("documents")
    if not isinstance(documents, dict):
        raise ValueError(f"invalid legacy history documents: {path}")
    if corpus == "batch_on_stream" and capture != "inputs":
        implementation, _version = split_sel(capture)
        provenance = record.get("provenance")
        if not isinstance(provenance, dict) or not isinstance(provenance.get("captured_with"), str):
            raise ValueError(f"invalid batch-on-stream provenance: {path}")
        expected_capture, expected_provenance = _snapshot_capture(implementation, provenance["captured_with"])
        if capture != expected_capture or provenance != expected_provenance:
            raise ValueError(f"batch-on-stream capture origin mismatch: {path}")
    for filename, document in documents.items():
        if (not isinstance(filename, str) or not filename.endswith(".yaml")
                or _component(filename) != filename or not isinstance(document, dict)
                or document.get("family") != family):
            raise ValueError(f"invalid legacy history document: {path}/{filename}")
        if corpus in {"batch", "reasoning"} and capture != "inputs":
            implementation, _version = split_sel(capture)
            for case_id, case in (document.get("cases") or {}).items():
                expected = case.get("expected")
                if expected is not None and (
                    not isinstance(expected, dict) or expected.keys() - {implementation}
                ):
                    raise ValueError(
                        f"legacy capture contains foreign expected results: "
                        f"{path}/{filename}/{case_id}"
                    )
    return record


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False, allow_unicode=True, width=4096))


_MISSING = object()


def _capture_order(capture: str) -> tuple:
    _impl, version = split_sel(capture)
    # A qualified or patch capture follows its plain release when numeric versions
    # tie. Both may be read by Python, while release-only Rust readers skip them.
    qualified = "+" in version or ".patch" in version
    return version_key(version), qualified, f"{capture}.yaml"


def _captured_value(corpus: str, impl: str, case: dict):
    if corpus == "stream":
        return case
    expected = case.get("expected")
    if not isinstance(expected, dict):
        return _MISSING
    return expected.get(impl, _MISSING)


def _seed_reasoning_state(records: dict[Path, dict], family_dir: Path, impl: str) -> dict:
    state = {}
    inputs_record = records.get(family_dir / SHARED_NAME)
    if inputs_record is None:
        return state
    inputs = inputs_record.get("documents") or {}
    for filename, document in inputs.items():
        for case_id, case in (document.get("cases") or {}).items():
            value = _captured_value("reasoning", impl, case)
            if value is not _MISSING:
                state[(filename, case_id)] = copy.deepcopy(value)
    return state


def _compact_capture(
    record: dict, corpus: str, impl: str, state: dict, annotations_state: dict,
    release_state: dict | None = None,
) -> dict:
    compacted = copy.deepcopy(record)
    release_capture = "+" not in record["capture"] and ".patch" not in record["capture"]
    for filename, document in compacted["documents"].items():
        kept = {}
        for case_id, case in (document.get("cases") or {}).items():
            value = _captured_value(corpus, impl, case)
            key = (filename, case_id)
            previous = state.get(key, _MISSING)
            prior_release = release_state.get(key, _MISSING) if release_state is not None else value
            # Batch/reasoning readers only fold expectations. Preserve both
            # capture annotations and their removal from a later observation.
            annotations = {k: v for k, v in case.items() if k != "expected"} if corpus != "stream" else {}
            annotated = bool(annotations) or annotations != annotations_state.get(key, {})
            if (value is _MISSING or previous is _MISSING or value != previous
                    or prior_release is _MISSING or value != prior_release or annotated):
                kept[case_id] = case
            annotations_state[key] = copy.deepcopy(annotations)
            if value is not _MISSING:
                state[key] = copy.deepcopy(value)
                if release_state is not None and release_capture:
                    release_state[key] = copy.deepcopy(value)
        document["cases"] = kept
    return compacted


def _store_records(store: Path) -> dict[Path, dict]:
    return {
        path: _validate_record(store, path, _load(path))
        for path in sorted(store.glob("*/families/*/*.yaml"))
    }


def _compact_records(records: dict[Path, dict]) -> tuple[dict[Path, dict], dict[Path, tuple]]:
    """Resolve and compact both Python and release-only stream histories in memory."""
    compacted = dict(records)
    states = {}
    grouped = defaultdict(list)
    for path, record in records.items():
        corpus = record["corpus"]
        capture = record["capture"]
        # Batch-on-stream selects independent versions in its header; its full
        # observations cannot inherit from the neighboring filename's version.
        if corpus in {"batch", "stream", "reasoning"} and capture != "inputs":
            impl, _version = split_sel(capture)
            grouped[(corpus, path.parent, impl)].append((_capture_order(capture), path, record))
    for (corpus, family_dir, impl), captures in grouped.items():
        state = _seed_reasoning_state(records, family_dir, impl) if corpus == "reasoning" else {}
        annotations_state = {}
        # Rust skips qualified stream captures, so its release anchors must survive.
        release_state = {} if corpus == "stream" else None
        for _order, path, record in sorted(captures):
            compacted[path] = _compact_capture(record, corpus, impl, state, annotations_state, release_state)
            states[path] = (
                copy.deepcopy(state),
                copy.deepcopy(release_state) if release_state is not None else None,
                copy.deepcopy(annotations_state),
            )
    return compacted, states


def _commit_records(before: dict[Path, dict], after: dict[Path, dict]) -> list[Path]:
    changed = []
    for path in sorted(before.keys() - after.keys()):
        path.unlink()
        changed.append(path)
    for path, record in sorted(after.items()):
        if before.get(path) != record:
            _write(path, record)
            changed.append(path)
    return changed


def _legacy_capture_input(document: dict, case: dict) -> dict:
    fields = (
        "model_text", "tools", "init", "finish_reason", "chat_template_kwargs", "force_reasoning",
    )
    request = {key: case.get(key) for key in fields}
    # Native readers consume case fields without inheriting document defaults.
    request["document"] = {key: document.get(key) for key in fields}
    request.update(family=document["family"], mode=document.get("mode"))
    request["chunks"] = [
        {key: value for key, value in chunk.items()
         if key in {"delta_text", "delta_token_ids", "token_ids", "finish_reason"}}
        if isinstance(chunk, dict) else chunk
        for chunk in case.get("chunks", [])
    ]
    return request


def _validate_captured_inputs(before: dict[Path, dict], candidate: dict[Path, dict]) -> None:
    captured = set()
    for path, record in before.items():
        if path not in candidate:
            continue
        corpus = record["corpus"]
        for filename, document in record["documents"].items():
            for case_id, case in document.get("cases", {}).items():
                if record["capture"] != "inputs" or (corpus == "reasoning" and case.get("expected")):
                    captured.add(("batch" if corpus == "batch_on_stream" else corpus,
                                  record["family"], filename, case_id))
    for path, record in before.items():
        if record["capture"] != "inputs" or record["corpus"] == "batch_on_stream":
            continue
        documents = candidate.get(path, {}).get("documents", {})
        for filename, document in record["documents"].items():
            replacement = documents.get(filename, {})
            for case_id, case in document.get("cases", {}).items():
                if (record["corpus"], record["family"], filename, case_id) not in captured:
                    continue
                current = replacement.get("cases", {}).get(case_id)
                if current is None or _legacy_capture_input(document, case) != _legacy_capture_input(replacement, current):
                    # Immutable sparse observations cannot be recaptured by this importer.
                    raise ValueError(f"captured legacy input is immutable; add a new case: {path}/{filename}/{case_id}")


def _apply_records(
    store: Path, staged: dict[Path, dict], *, prune: bool, archive_import: bool = False,
) -> list[Path]:
    """Check the complete candidate before changing any history file."""
    before = _store_records(store)
    desired = set(staged)
    candidate = {path: record for path, record in before.items() if not prune or path in desired}
    for path, record in staged.items():
        incoming = copy.deepcopy(_validate_record(store, path, record))
        old = before.get(path)
        if old is not None and incoming["corpus"] == "batch_on_stream" and incoming["capture"] != "inputs":
            if incoming["provenance"] != old["provenance"]:
                raise ValueError(f"versioned legacy capture is immutable: {path}")
            merged = copy.deepcopy(old["documents"])
            for filename, document in incoming["documents"].items():
                previous = merged.get(filename)
                if previous is not None:
                    for case_id in previous["cases"].keys() & document["cases"].keys():
                        if previous["cases"][case_id] != document["cases"][case_id]:
                            raise ValueError(f"versioned legacy capture is immutable: {path}/{filename}/{case_id}")
                    document["cases"] = previous["cases"] | document["cases"]
                merged[filename] = document
            incoming["documents"] = merged
        elif old is not None and incoming["capture"] == "inputs":
            if archive_import and old != incoming:
                raise ValueError(f"cannot overwrite imported legacy capture: {path}")
            if not archive_import and not prune:
                incoming["documents"] = old["documents"] | incoming["documents"]
        candidate[path] = incoming

    for path, record in candidate.items():
        _validate_record(store, path, record)
    _validate_captured_inputs(before, candidate)

    for path, record in staged.items():
        if path in before or record["capture"] == "inputs" or record["corpus"] == "batch_on_stream":
            continue
        impl, _version = split_sel(record["capture"])
        later = [
            existing["capture"] for other, existing in before.items()
            if other in candidate and other.parent == path.parent and other != path
            and split_sel(existing["capture"])[0] == impl
            and _capture_order(existing["capture"]) > _capture_order(record["capture"])
        ]
        if later:
            newest = max(later, key=_capture_order)
            raise ValueError(
                f"cannot add out-of-order legacy capture {record['capture']} after sparse "
                f"checkpoint {newest}; later resolved outputs cannot be reconstructed"
            )

    compacted, new_states = _compact_records(candidate)
    _old_compacted, old_states = _compact_records(before)
    for path in before.keys() & compacted.keys():
        capture = compacted[path]["capture"]
        if capture != "inputs" and compacted[path]["corpus"] != "batch_on_stream" and compacted[path] != before[path]:
            raise ValueError(f"versioned legacy capture is immutable: {path}")
        if path in old_states and old_states[path] != new_states[path]:
            raise ValueError(f"prune changes resolved legacy capture output: {path}")
    _snapshot_documents(compacted)
    return _commit_records(before, compacted)


def store_digest(store: Path) -> tuple[str, int]:
    inventory = []
    for path in sorted(store.glob("*/families/*/*.yaml")):
        data = path.read_bytes()
        inventory.append({
            "path": path.relative_to(store).as_posix(),
            "sha256": hashlib.sha256(data).hexdigest(),
            "size": len(data),
        })
    if not inventory:
        raise ValueError(f"empty legacy YAML history: {store}")
    encoded = json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), sum(item["size"] for item in inventory)


def _archive_identity(relative: str) -> tuple[str, str, str]:
    for corpus, prefix in CORPORA.items():
        if corpus == "batch_on_stream":
            if relative == f"{prefix}.tar.gz":
                return corpus, prefix, SNAPSHOT_NAME
        elif relative.startswith(prefix + "/") and relative.endswith(".tar.gz"):
            capture = relative[len(prefix) + 1 : -len(".tar.gz")]
            if "/" not in capture:
                return corpus, prefix, _component(capture)
    raise ValueError(f"unrecognized legacy archive: {relative}")


def _snapshot_capture(implementation: str, label: str) -> tuple[str, dict]:
    """Separate a recorded version from its exact producer label without guessing."""
    _component(implementation)
    if not isinstance(label, str) or not label:
        raise ValueError("batch-on-stream capture label must be a nonempty string")
    provenance = {"captured_with": label, "runtime_version": None}
    match = re.fullmatch(r"v?(\d+\.\d+\.\d+) ([0-9a-f]{40})", label)
    if match:
        version, commit = match.groups()
        provenance.update(runtime_version=version, git_commit=commit)
    elif CAPTURE_RE.fullmatch(f"{implementation}-{label}"):
        provenance["runtime_version"] = label
    version = provenance["runtime_version"] or "unversioned"
    return f"{implementation}-{version}", provenance


def _validate_uncaptured_block(block: dict, location: str) -> None:
    if (not isinstance(block, dict) or block.keys() != {"unavailable"}
            or not isinstance(block["unavailable"], str) or not block["unavailable"]):
        raise ValueError(f"batch-on-stream result has no capture label: {location}")


def _split_snapshot_records(store: Path, family: str, documents: dict) -> dict[Path, dict]:
    records = {}
    header_path = _store_file(store, "batch_on_stream", family, "inputs")
    records[header_path] = {"corpus": "batch_on_stream", "family": family, "capture": "inputs", "documents": {}}
    for filename, document in documents.items():
        if document.keys() & {"capture_order", "case_order", "capture_selection", "uncaptured"}:
            raise ValueError(f"reserved batch-on-stream header field: {family}/{filename}")
        captured = document["captured_with"]
        cases = document["cases"]
        selection = {}
        for implementation, label in captured.items():
            capture, provenance = _snapshot_capture(implementation, label)
            selection[implementation] = capture
            target = _store_file(store, "batch_on_stream", family, capture)
            record = records.setdefault(target, {"corpus": "batch_on_stream", "family": family,
                                                "capture": capture, "provenance": provenance, "documents": {}})
            if record["provenance"] != provenance:
                raise ValueError(f"conflicting batch-on-stream capture labels: {target}")
            record["documents"][filename] = {"family": family, "cases": {
                case_id: copy.deepcopy(row[implementation]) for case_id, row in cases.items() if implementation in row
            }}
        uncaptured = {}
        for case_id, row in cases.items():
            for implementation, block in row.items():
                if implementation not in captured:
                    _validate_uncaptured_block(block, f"{family}/{filename}/{case_id}/{implementation}")
                    uncaptured.setdefault(case_id, {})[implementation] = copy.deepcopy(block)
        records[header_path]["documents"][filename] = {
            **{key: copy.deepcopy(value) for key, value in document.items() if key not in {"captured_with", "cases"}},
            "capture_order": list(captured),
            "case_order": {case_id: list(row) for case_id, row in cases.items()},
            "capture_selection": selection,
        }
        if uncaptured:
            # An absent optional runtime has no versioned observation to archive.
            records[header_path]["documents"][filename]["uncaptured"] = uncaptured
    return records


def _snapshot_documents(records: dict[Path, dict]) -> dict[tuple[str, str], dict]:
    """Reconstruct selected observations, retaining source map order and labels."""
    groups = defaultdict(dict)
    for record in records.values():
        if record["corpus"] == "batch_on_stream":
            groups[record["family"]][record["capture"]] = record
    documents = {}
    for family, parts in groups.items():
        if "inputs" not in parts:
            raise ValueError(f"missing batch-on-stream input headers: {family}")
        for filename, header in parts["inputs"]["documents"].items():
            selection = header["capture_selection"]
            capture_order = header["capture_order"]
            if len(capture_order) != len(set(capture_order)) or set(capture_order) != selection.keys():
                raise ValueError(f"invalid batch-on-stream capture order: {family}/{filename}")
            captured, fragments = {}, {}
            for implementation in capture_order:
                capture = selection[implementation]
                if capture not in parts or split_sel(capture)[0] != implementation:
                    raise ValueError(f"missing batch-on-stream capture: {family}/{filename}/{implementation}")
                record = parts[capture]
                if filename not in record["documents"]:
                    raise ValueError(f"missing batch-on-stream document: {family}/{filename}/{implementation}")
                fragments[implementation] = record["documents"][filename]["cases"]
                captured[implementation] = record["provenance"]["captured_with"]
            uncaptured = header.get("uncaptured", {})
            expected_uncaptured = {
                (case_id, implementation)
                for case_id, implementations in header["case_order"].items()
                for implementation in implementations if implementation not in selection
            }
            if {(case_id, implementation) for case_id, row in uncaptured.items()
                    for implementation in row} != expected_uncaptured:
                raise ValueError(f"invalid batch-on-stream uncaptured blocks: {family}/{filename}")
            cases = {}
            for case_id, implementations in header["case_order"].items():
                if len(implementations) != len(set(implementations)):
                    raise ValueError(f"duplicate batch-on-stream implementation: {family}/{filename}/{case_id}")
                row = {}
                for implementation in implementations:
                    if implementation not in selection:
                        block = uncaptured[case_id][implementation]
                        _validate_uncaptured_block(block, f"{family}/{filename}/{case_id}/{implementation}")
                        row[implementation] = copy.deepcopy(block)
                        continue
                    if implementation not in fragments or case_id not in fragments[implementation]:
                        raise ValueError(f"missing batch-on-stream block: {family}/{filename}/{case_id}/{implementation}")
                    row[implementation] = copy.deepcopy(fragments[implementation][case_id])
                cases[case_id] = row
            documents[(family, filename)] = {
                **{key: copy.deepcopy(value) for key, value in header.items()
                   if key not in {"capture_order", "case_order", "capture_selection", "uncaptured"}},
                "captured_with": captured, "cases": cases,
            }
    return documents


def split_batch_on_stream_store(store: Path) -> list[Path]:
    """Replace old mixed snapshots only after every new record reconstructs exactly."""
    before = {}
    after = {}
    original = {}
    for path in sorted((store / "batch_on_stream/families").glob(f"*/{SNAPSHOT_NAME}.yaml")):
        record = _load(path)
        family = path.parent.name
        if record.get("corpus") != "batch_on_stream" or record.get("family") != family or record.get("capture") != SNAPSHOT_NAME:
            raise ValueError(f"legacy history identity mismatch: {path}")
        before[path] = record
        for filename, document in record["documents"].items():
            original[(family, filename)] = document
        for target, split in _split_snapshot_records(store, family, record["documents"]).items():
            if target.exists():
                raise ValueError(f"refusing to overwrite existing split capture: {target}")
            after[target] = _validate_record(store, target, split)
    if _snapshot_documents(after) != original:
        raise ValueError("split batch-on-stream observations differ from original snapshots")
    return _commit_records(before, after)


def import_archives(fixtures: Path, shards: list[dict], store: Path) -> None:
    """Import manifest-pinned archive members without resolving away history."""
    grouped: dict[tuple[str, str, str], dict[str, dict]] = defaultdict(dict)
    for shard in shards:
        relative = shard["path"]
        if not relative.endswith(".tar.gz"):
            continue
        corpus, prefix, capture = _archive_identity(relative)
        path = fixtures / relative
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != shard["sha256"]:
            raise ValueError(f"archive differs from manifest: {relative}")
        with tarfile.open(path, "r:gz") as archive:
            for member in archive.getmembers():
                if not member.isfile():
                    continue
                parts = Path(member.name).parts
                expected = tuple(Path(prefix).parts) + (() if corpus == "batch_on_stream" else (capture,))
                if parts[: len(expected)] != expected or len(parts) != len(expected) + 2:
                    raise ValueError(f"unexpected archive member: {relative}/{member.name}")
                family, filename = map(_component, parts[-2:])
                if not filename.endswith(".yaml"):
                    raise ValueError(f"unexpected archive member: {relative}/{member.name}")
                doc = yaml.safe_load(archive.extractfile(member))
                if not isinstance(doc, dict) or doc.get("family") != family:
                    raise ValueError(f"invalid fixture document: {relative}/{member.name}")
                key = (corpus, family, capture)
                if filename in grouped[key]:
                    raise ValueError(f"duplicate fixture document: {relative}/{member.name}")
                grouped[key][filename] = doc
    staged = {}
    for (corpus, family, capture), documents in sorted(grouped.items()):
        if corpus == "batch_on_stream":
            staged.update(_split_snapshot_records(store, family, documents))
            continue
        staged[_store_file(store, corpus, family, capture)] = {
            "corpus": corpus, "family": family, "capture": capture,
            "documents": documents,
        }
    _apply_records(store, staged, prune=False, archive_import=True)


def materialize_store(store: Path, destination: Path) -> None:
    planned = {}
    records = _store_records(store)
    for source, record in records.items():
        corpus = record["corpus"]
        capture = record["capture"]
        if corpus == "batch_on_stream":
            continue
        dirname = capture
        for filename, document in record["documents"].items():
            target = destination / CORPORA[corpus] / dirname / record["family"] / filename
            if target in planned:
                raise ValueError(f"duplicate materialized legacy document: {source} and {planned[target][0]}")
            planned[target] = (source, document)
    for (family, filename), document in _snapshot_documents(records).items():
        planned[destination / CORPORA["batch_on_stream"] / family / filename] = (store, document)
    for target, (_source, document) in planned.items():
        _write(target, document)


def update_from_loose(store: Path, loose: Path, *, prune: bool = False) -> list[Path]:
    """Import captures; prune only when the loose tree is a complete replacement."""
    staged = {}
    for corpus, prefix in CORPORA.items():
        root = loose / prefix
        if not root.is_dir():
            continue
        if corpus == "batch_on_stream":
            for family_dir in sorted(root.iterdir()):
                if not family_dir.is_dir():
                    continue
                family = _component(family_dir.name)
                documents = {_component(p.name): _load(p) for p in family_dir.glob("*.yaml")}
                if not documents:
                    continue
                staged.update(_split_snapshot_records(store, family, documents))
            continue
        capture_dirs = [path for path in root.iterdir() if path.is_dir()]
        capture_dirs.sort(
            key=lambda path: (0, "") if path.name == "inputs" else (1, _capture_order(path.name))
        )
        for capture_dir in capture_dirs:
            if not capture_dir.is_dir():
                continue
            capture = "inputs" if capture_dir.name == "inputs" else _component(capture_dir.name)
            for family_dir in sorted(capture_dir.iterdir()):
                if not family_dir.is_dir():
                    continue
                family = _component(family_dir.name)
                documents = {_component(p.name): _load(p) for p in family_dir.glob("*.yaml")}
                if not documents:
                    continue
                target = _store_file(store, corpus, family, capture)
                staged[target] = {"corpus": corpus, "family": family, "capture": capture, "documents": documents}
    return _apply_records(store, staged, prune=prune)
