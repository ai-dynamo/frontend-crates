# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load, validate, update, and materialize the Unified YAML history store."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import multiprocessing
import os
import re
import shutil
import tempfile
import warnings
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from yaml.composer import Composer
from yaml.constructor import SafeConstructor
from yaml.cyaml import CParser
from yaml.resolver import BaseResolver, Resolver

import capture_stimulus
import fixture_disposition


FAMILY_DOCUMENT_NAME = "inputs_and_golden.yaml"
FAMILY_KEYS = {
    "input_document",
    "golden_document",
    "cases",
}
CAPTURE_DOCUMENT_KEYS = {
    "provenance",
    "document",
    "changes",
    "metadata_changes",
    "document_overrides",
}
CAPTURE_PROVENANCE_KEYS = {"status", "origin", "captured_with"}
CAPTURE_PROVENANCE_STATUSES = {"captured", "legacy"}
CASE_KEY_ORDER = (
    "lifecycle",
    "scenario",
    "description",
    "policy",
    "display_id",
    "historical_ids",
    "request",
    "golden",
)
CASE_KEYS = set(CASE_KEY_ORDER)
CHANGE_KEYS = {"case_key", "stimulus", "observation"}
OBSERVATION_STATES = {"value", "error", "unavailable"}
STIMULUS_STATES = {"ref", "inline", "partial", "unavailable"}
REQUEST_KEYS = {"input", "init", "finish_reason", "tools", "chunks"}
CAPTURE_DIRECTORY_RE = re.compile(
    r"(?P<implementation>[a-z0-9_]+)-(?P<runtime_version>\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?)"
)
GENERATED_ORACLE_DIRECTORY = "golden_spec"
DOCUMENT_METADATA_EXCLUDED = {
    "family",
    "record_metadata",
}
DOCUMENT_DERIVED_KEYS = (
    "capture_origin",
    "captured_with",
)
TRANSACTION_JOURNAL = ".conformance-transaction.json"
TRANSACTION_SCHEMA_VERSION = 1


def is_generated_oracle_directory(name: str) -> bool:
    return name == GENERATED_ORACLE_DIRECTORY or name.startswith(
        f"{GENERATED_ORACLE_DIRECTORY}-"
    )


class StrictLoader(CParser, Composer, SafeConstructor, Resolver):
    def __init__(self, stream):
        CParser.__init__(self, stream)
        Composer.__init__(self)
        SafeConstructor.__init__(self)
        Resolver.__init__(self)

    get_single_node = Composer.get_single_node

    def compose_node(self, parent, index):
        if self.check_event(yaml.events.AliasEvent):
            raise ValueError("YAML aliases are not allowed")
        event = self.peek_event()
        if event.anchor is not None:
            raise ValueError("YAML anchors are not allowed")
        if event.tag is not None:
            raise ValueError("YAML tags are not allowed")
        return super().compose_node(parent, index)


def _construct_mapping(loader: StrictLoader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"duplicate key: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


StrictLoader.add_constructor(BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def load_yaml(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    try:
        value = yaml.load(text, Loader=StrictLoader)
    except ValueError as exc:
        raise ValueError(f"{exc}: {path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"YAML document must be a mapping: {path}")
    return value


def dump_yaml(value: dict) -> str:
    class NoAliasDumper(yaml.SafeDumper):
        def ignore_aliases(self, data):
            return True

    return yaml.dump(
        value,
        Dumper=NoAliasDumper,
        sort_keys=False,
        allow_unicode=True,
        width=4096,
        default_flow_style=False,
    )


def _require_keys(value: dict, required: set[str], allowed: set[str], where: str) -> None:
    missing = required - value.keys()
    unknown = value.keys() - allowed
    if missing:
        raise ValueError(f"{where} is missing required fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{where} has unknown fields: {sorted(unknown)}")


def _mapping(value: Any, where: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be a mapping")
    return value


def _safe_component(value: Any, where: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or "\0" in value
    ):
        raise ValueError(f"{where} must be one safe path component")
    return value


def _request_digest(request: dict) -> str:
    encoded = json.dumps(
        request,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_request(request: Any, where: str) -> None:
    request = _mapping(request, where)
    _require_keys(request, REQUEST_KEYS, REQUEST_KEYS, where)
    if not isinstance(request["input"], str):
        raise ValueError(f"{where}.input must be a string")
    if not isinstance(request["init"], dict):
        raise ValueError(f"{where}.init must be a mapping")
    if not isinstance(request["finish_reason"], str):
        raise ValueError(f"{where}.finish_reason must be a string")
    if not isinstance(request["tools"], list):
        raise ValueError(f"{where}.tools must be a list")
    if not isinstance(request["chunks"], list) or any(
        not isinstance(chunk, dict) for chunk in request["chunks"]
    ):
        raise ValueError(f"{where}.chunks must be a list of mappings")


def _validate_case_metadata(
    description: Any,
    policy: Any,
    where: str,
    *,
    active: bool,
) -> None:
    if description is not None and not isinstance(description, str):
        raise ValueError(f"{where}.description must be a string or null")
    if active and not description:
        raise ValueError(f"{where}.description must be a non-empty string for an active case")
    if not isinstance(policy, list) or any(not isinstance(tag, str) for tag in policy):
        raise ValueError(f"{where}.policy must be a list of strings")


def _validate_stimulus(stimulus: Any, case: dict, where: str) -> None:
    stimulus = _mapping(stimulus, where)
    states = stimulus.keys() & STIMULUS_STATES
    allowed = states | {"semantic_sha256"}
    if len(states) != 1 or stimulus.keys() - allowed:
        raise ValueError(f"{where} must contain exactly one stimulus state")
    state = next(iter(states))
    if state == "ref":
        if stimulus["ref"] != "current" or case["request"] is None:
            raise ValueError(f"{where} has an invalid current stimulus reference")
        expected = _request_digest(case["request"])
        digest = stimulus.get("semantic_sha256")
        if digest is not None and digest != expected:
            raise ValueError(f"{where} stimulus digest differs from the canonical request")
    elif state == "inline":
        _validate_request(stimulus["inline"], f"{where}.inline")
    elif state == "partial":
        if not isinstance(stimulus["partial"], dict):
            raise ValueError(f"{where}.partial must be a mapping")
    else:
        unavailable = _mapping(stimulus["unavailable"], f"{where}.unavailable")
        if not isinstance(unavailable.get("code"), str):
            raise ValueError(f"{where}.unavailable needs a string code")


def _validate_observation(observation: Any, where: str) -> None:
    observation = _mapping(observation, where)
    states = observation.keys() & OBSERVATION_STATES
    if len(states) != 1 or observation.keys() != states:
        raise ValueError(f"{where} must contain exactly one observation state")
    state = next(iter(states))
    payload = _mapping(observation[state], f"{where}.{state}")
    if state == "value":
        if set(payload) - {"assembled", "chunks"}:
            raise ValueError(f"{where}.value has unknown fields")
        if "assembled" not in payload and "chunks" not in payload:
            raise ValueError(f"{where}.value needs assembled or chunks")
    if state != "value" and not isinstance(payload.get("code"), str):
        raise ValueError(f"{where}.{state} needs a string code")


@dataclass
class Family:
    name: str
    document: dict
    path: Path

    @property
    def cases(self) -> dict:
        return self.document["cases"]


@dataclass
class History:
    family: Family
    implementation: str
    captures: dict
    directory: Path
    capture_paths: dict[str, Path]
    @property
    def path(self) -> Path:
        """Compatibility diagnostic path for callers that name the history owner."""
        return self.directory

    def capture_path(self, capture_id: str) -> Path:
        return self.capture_paths[capture_id]

    def ordered_capture_ids(self) -> list[str]:
        return sorted(
            self.captures,
            key=lambda capture_id: _capture_release_sort_key(
                self.captures[capture_id]["runtime_version"]
            ),
        )

    def resolve(self, capture_id: str) -> dict:
        if capture_id not in self.captures:
            raise ValueError(f"unknown capture in {self.directory}: {capture_id}")
        state = {}
        for current in self.ordered_capture_ids():
            capture = self.captures[current]
            for case_id, change in capture["changes"].items():
                if change == {"absent": True}:
                    state.pop(case_id, None)
                    continue
                inherited_metadata = state.get(case_id, {}).get("_record_metadata")
                inherited_overrides = state.get(case_id, {}).get("_document_overrides")
                state[case_id] = copy.deepcopy(change)
                state[case_id]["_origin_capture_id"] = current
                if inherited_metadata is not None:
                    state[case_id]["_record_metadata"] = inherited_metadata
                if inherited_overrides is not None:
                    state[case_id]["_document_overrides"] = inherited_overrides
            for case_id, metadata in capture["metadata_changes"].items():
                if case_id not in state:
                    raise ValueError(
                        f"metadata change has no observation in {self.capture_path(current)}: "
                        f"{case_id}"
                    )
                if metadata:
                    state[case_id]["_record_metadata"] = copy.deepcopy(metadata)
                else:
                    state[case_id].pop("_record_metadata", None)
            for case_id, override in capture["document_overrides"].items():
                if case_id not in state:
                    raise ValueError(
                        f"document override has no observation in {self.capture_path(current)}: "
                        f"{case_id}"
                    )
                state[case_id]["_document_overrides"] = copy.deepcopy(override)
            if current == capture_id:
                break
        else:
            raise ValueError(f"capture order does not include {capture_id}: {self.directory}")
        materialized = {}
        for case_id, change in state.items():
            record = copy.deepcopy(change)
            record_metadata = record.pop("_record_metadata", None)
            overrides = record.pop("_document_overrides", {})
            origin_capture_id = record.pop("_origin_capture_id")
            origin_capture = self.captures[origin_capture_id]
            case_document = {
                **_capture_document_metadata(origin_capture, self.implementation),
                **overrides,
            }
            if origin_capture_id != capture_id:
                case_document["inherited_from"] = origin_capture["runtime_version"]
            if record_metadata:
                case_document["record_metadata"] = record_metadata
            record["document"] = case_document
            materialized[case_id] = record
        return materialized


@dataclass
class Store:
    root: Path
    families: dict[str, Family]
    histories: dict[tuple[str, str], History]


class PublicationCommitIndeterminate(OSError):
    """The new generation is installed, but its commit marker was not durably synced."""


def _validate_family(path: Path, document: dict) -> Family:
    _require_keys(document, FAMILY_KEYS, FAMILY_KEYS, str(path))
    family = path.parent.name
    if (
        path.name != FAMILY_DOCUMENT_NAME
        or re.fullmatch(r"[a-z0-9_]+", family) is None
    ):
        raise ValueError(f"family differs from filename: {path}")
    for field in ("input_document", "golden_document"):
        metadata = _mapping(document[field], f"{path}.{field}")
        _require_keys(metadata, {"family"}, set(metadata), f"{path}.{field}")
        if metadata["family"] != family:
            raise ValueError(f"{field} family differs from directory: {path}")
    cases = _mapping(document["cases"], f"{path}.cases")
    display_ids: set[str] = set()
    aliases: set[str] = set()
    canonical_owners: dict[str, tuple[str, str]] = {}

    def claim_canonical_id(case_id: str, external_id: str) -> None:
        canonical_id = fixture_disposition.historical_unified_case_key(family, external_id)
        owner = canonical_owners.get(canonical_id)
        if owner is not None and owner[0] != case_id:
            raise ValueError(
                f"duplicate canonical ID in {path}: {owner[1]} and {external_id} resolve to "
                f"{canonical_id} for different cases"
            )
        canonical_owners[canonical_id] = (case_id, external_id)

    for case_id, case in cases.items():
        if not isinstance(case_id, str):
            raise ValueError(f"case ID must be a string: {path}")
        case = _mapping(case, f"{path}:{case_id}")
        _require_keys(case, CASE_KEYS, CASE_KEYS, f"{path}:{case_id}")
        if case["lifecycle"] not in {"active", "retired"}:
            raise ValueError(f"invalid lifecycle in {path}:{case_id}")
        _validate_case_metadata(
            case["description"],
            case["policy"],
            f"{path}:{case_id}",
            active=case["lifecycle"] == "active",
        )
        display_id = case["display_id"]
        if display_id is not None:
            _safe_component(display_id, f"{path}:{case_id}.display_id")
            if display_id in display_ids or display_id in aliases:
                raise ValueError(f"duplicate display ID in {path}: {display_id}")
            display_ids.add(display_id)
            claim_canonical_id(case_id, display_id)
        historical_ids = case["historical_ids"]
        if not isinstance(historical_ids, list) or any(
            not isinstance(alias, str) for alias in historical_ids
        ):
            raise ValueError(f"historical_ids must be strings in {path}:{case_id}")
        for alias in historical_ids:
            _safe_component(alias, f"{path}:{case_id}.historical_ids")
            if alias in aliases or alias in display_ids:
                raise ValueError(f"duplicate historical ID in {path}: {alias}")
            aliases.add(alias)
            claim_canonical_id(case_id, alias)
        if case["lifecycle"] == "active":
            if display_id is None or not isinstance(case["scenario"], str):
                raise ValueError(f"active case needs display_id and scenario: {path}:{case_id}")
            _validate_request(case["request"], f"{path}:{case_id}.request")
            _mapping(case["golden"], f"{path}:{case_id}.golden")
        elif case["request"] is not None:
            _validate_request(case["request"], f"{path}:{case_id}.request")
    return Family(family, document, path)


def _capture_document_metadata(capture: dict, implementation: str) -> dict:
    document = copy.deepcopy(capture["document"])
    provenance = capture["provenance"]
    document["captured_with"] = {implementation: capture["runtime_version"]}
    if provenance.get("status") == "captured":
        document["capture_origin"] = copy.deepcopy(provenance["origin"])
    return document


def _validate_capture_provenance(
    provenance: Any,
    path: Path,
    implementation: str,
    runtime_version: str,
) -> dict:
    provenance = _mapping(provenance, f"{path}.provenance")
    _require_keys(
        provenance,
        {"status"},
        CAPTURE_PROVENANCE_KEYS,
        f"{path}.provenance",
    )
    status = provenance["status"]
    if status not in CAPTURE_PROVENANCE_STATUSES:
        raise ValueError(f"invalid capture provenance status in {path}: {status!r}")

    if status == "captured":
        _require_keys(provenance, {"status", "origin"}, {"status", "origin"}, f"{path}.provenance")
        origin = _mapping(provenance["origin"], f"{path}.provenance.origin")
        _require_keys(origin, {"crate_version", "source_sha256"}, {"crate_version", "source_sha256", "git_commit"}, f"{path}.provenance.origin")
        if origin["crate_version"] != runtime_version or not re.fullmatch(r"[0-9a-f]{64}", origin["source_sha256"]):
            raise ValueError(f"capture origin differs from runtime identity: {path}")
        if "git_commit" in origin and not re.fullmatch(r"[0-9a-f]{40}", origin["git_commit"]):
            raise ValueError(f"capture origin has invalid git commit: {path}")
    else:
        _require_keys(provenance, {"status", "captured_with"}, {"status", "captured_with"}, f"{path}.provenance")
        captured_with = _mapping(
            provenance["captured_with"], f"{path}.provenance.captured_with"
        )
        if captured_with != {implementation: runtime_version}:
            raise ValueError(f"capture provenance differs from runtime identity: {path}")
    return provenance


def _validate_capture_file(
    path: Path,
    document: dict,
    families: dict[str, Family],
) -> tuple[str, str, str, dict]:
    _require_keys(document, CAPTURE_DOCUMENT_KEYS, CAPTURE_DOCUMENT_KEYS, str(path))
    family_name = path.parent.name
    identity = CAPTURE_DIRECTORY_RE.fullmatch(path.stem)
    if (
        family_name not in families
        or identity is None
    ):
        raise ValueError(f"capture identity differs from its path: {path}")
    capture_id = path.stem
    _safe_component(capture_id, str(path))
    implementation = identity["implementation"]
    runtime_version = identity["runtime_version"]
    capture = {**document, "runtime_version": runtime_version}
    provenance = _validate_capture_provenance(
        capture["provenance"], path, implementation, runtime_version
    )
    shared_document = _mapping(capture["document"], f"{path}.document")
    repeated = set(DOCUMENT_DERIVED_KEYS) & shared_document.keys()
    if repeated:
        raise ValueError(f"capture document repeats derived provenance in {path}: {sorted(repeated)}")
    changes = _mapping(capture["changes"], f"{path}.changes")
    for case_id, change in changes.items():
        if case_id not in families[family_name].cases:
            raise ValueError(f"unknown case in {path}: {case_id}")
        if change == {"absent": True}:
            continue
        change = _mapping(change, f"{path}:{case_id}")
        _require_keys(change, CHANGE_KEYS, CHANGE_KEYS, f"{path}:{case_id}")
        _safe_component(change["case_key"], f"{path}:{case_id}.case_key")
        _validate_stimulus(
            change["stimulus"],
            families[family_name].cases[case_id],
            f"{path}:{case_id}.stimulus",
        )
        _validate_observation(change["observation"], f"{path}:{case_id}.observation")
    metadata_changes = _mapping(capture["metadata_changes"], f"{path}.metadata_changes")
    for case_id, metadata in metadata_changes.items():
        if case_id not in families[family_name].cases:
            raise ValueError(f"unknown metadata case in {path}: {case_id}")
        _mapping(metadata, f"{path}:{case_id}.metadata_changes")
    document_overrides = _mapping(
        capture["document_overrides"], f"{path}.document_overrides"
    )
    for case_id, metadata in document_overrides.items():
        if not isinstance(case_id, str):
            raise ValueError(f"document override case ID must be a string in {path}")
        if case_id not in families[family_name].cases:
            raise ValueError(f"unknown document override case in {path}: {case_id}")
        metadata = _mapping(metadata, f"{path}:{case_id}.document_overrides")
        if "family" in metadata or "record_metadata" in metadata:
            raise ValueError(
                f"document override contains reserved metadata in {path}:{case_id}"
            )
        if set(metadata) - {"parser_path"}:
            raise ValueError(f"unsupported document override in {path}:{case_id}")
        if "parser_path" in metadata and not isinstance(metadata["parser_path"], str):
            raise ValueError(f"{path}:{case_id}.document_overrides.parser_path must be a string")
    return family_name, implementation, capture_id, capture


def _validate_history(
    family: Family,
    implementation: str,
    captures: dict,
    capture_paths: dict[str, Path],
) -> History:
    where = family.path.parent
    if not captures:
        raise ValueError(f"history has no captures: {where}/{implementation}")
    history = History(family, implementation, captures, where, capture_paths)
    for capture_id, capture in captures.items():
        state = history.resolve(capture_id)
        missing = sorted(set(capture["document_overrides"]) - set(state))
        if missing:
            raise ValueError(
                f"document override has no observation in {capture_paths[capture_id]}: {missing}"
            )
    return history


def _canonical_store_root(root: Path) -> Path:
    return Path(root).resolve(strict=False)


def _load_store_unlocked(root: Path) -> Store:
    root = _canonical_store_root(root)
    families_root = root / "families"
    legacy_family_paths = sorted(families_root.glob("*.yaml"))
    if legacy_family_paths:
        raise ValueError(f"legacy Unified family layout is not supported: {legacy_family_paths[0]}")
    legacy_history = root / "capture_history"
    if legacy_history.exists():
        raise ValueError(f"legacy Unified capture_history layout is not supported: {legacy_history}")
    if not families_root.is_dir():
        raise ValueError(f"Unified history has no families directory: {root}")
    unknown_root_entries = sorted(path for path in root.iterdir() if path.name != "families")
    if unknown_root_entries:
        raise ValueError(f"unknown Unified root entry: {unknown_root_entries[0]}")

    family_directories = sorted(path for path in families_root.iterdir() if path.is_dir())
    unknown_family_entries = sorted(path for path in families_root.iterdir() if not path.is_dir())
    if unknown_family_entries:
        raise ValueError(f"unknown Unified family entry: {unknown_family_entries[0]}")
    family_paths = [directory / FAMILY_DOCUMENT_NAME for directory in family_directories]
    if not family_paths:
        raise ValueError(f"Unified history has no family files: {root}")
    families = {}
    for path in family_paths:
        if not path.is_file():
            raise ValueError(f"Unified family has no {FAMILY_DOCUMENT_NAME}: {path.parent}")
        family = _validate_family(path, load_yaml(path))
        if family.name in families:
            raise ValueError(f"duplicate family: {family.name}")
        families[family.name] = family

    grouped: dict[tuple[str, str], dict[str, dict]] = {}
    grouped_paths: dict[tuple[str, str], dict[str, Path]] = {}
    family_capture_owners: dict[str, tuple[str, str]] = {}
    for family_name, family in sorted(families.items()):
        entries = sorted(family.path.parent.iterdir())
        for path in entries:
            if path == family.path:
                continue
            if path.is_dir() or path.suffix != ".yaml" or path.name == "index.yaml":
                raise ValueError(f"unknown Unified family file: {path}")
            capture_family, implementation, capture_id, capture = _validate_capture_file(
                path,
                load_yaml(path),
                families,
            )
            if capture_family != family_name:
                raise ValueError(f"capture family differs from directory: {path}")
            owner = family_capture_owners.setdefault(capture_id, (implementation, family_name))
            if owner[0] != implementation:
                raise ValueError(f"capture implementation differs across paths: {capture_id}")
            key = (family_name, implementation)
            captures = grouped.setdefault(key, {})
            paths = grouped_paths.setdefault(key, {})
            if capture_id in captures:
                raise ValueError(f"duplicate capture: {family_name}/{capture_id}")
            captures[capture_id] = capture
            paths[capture_id] = path

    histories = {}
    for key, captures in sorted(grouped.items()):
        family_name, implementation = key
        histories[key] = _validate_history(
            families[family_name],
            implementation,
            captures,
            grouped_paths[key],
        )
    if not histories:
        raise ValueError(f"Unified history has no history files: {root}")
    return Store(root, families, histories)


def load_store(root: Path) -> Store:
    root = _canonical_store_root(root)
    with _store_read_lock(root):
        return _load_store_unlocked(root)


def _store_inventory_unlocked(root: Path) -> list[dict]:
    root = Path(root)
    inventory = []
    paths = list(root.glob("families/*/*.yaml"))
    for path in sorted(paths):
        data = path.read_bytes()
        inventory.append(
            {
                "path": str(path.relative_to(root)),
                "sha256": hashlib.sha256(data).hexdigest(),
                "size": len(data),
            }
        )
    return inventory


def store_inventory(root: Path) -> list[dict]:
    root = _canonical_store_root(root)
    with _store_read_lock(root):
        return _store_inventory_unlocked(root)


def store_digest(root: Path) -> tuple[str, int]:
    inventory = store_inventory(root)
    encoded = json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), sum(item["size"] for item in inventory)


def rewrite_store(root: Path) -> None:
    _mutate_store(root, lambda _store: None)


def _capture_document(history: History, capture_id: str) -> dict:
    capture = history.captures[capture_id]
    return {
        key: value for key, value in capture.items() if key != "runtime_version"
    }


def _store_documents(store: Store) -> dict[Path, dict]:
    documents = {family.path: family.document for family in store.families.values()}
    for history in store.histories.values():
        for capture_id in history.captures:
            documents[history.capture_path(capture_id)] = _capture_document(history, capture_id)
    return documents


def _commit_store_documents(store_root: Path, documents: dict[Path, dict]) -> list[Path]:
    """Validate and publish one complete store generation."""
    store_root = Path(store_root)
    rendered: dict[Path, str] = {}
    for path, document in documents.items():
        rendered[path] = dump_yaml(document)
    current_paths = {
        path
        for path in store_root.rglob("*")
        if path.is_file()
    }
    desired_paths = set(rendered)
    changed = {
        path
        for path, text in rendered.items()
        if not path.is_file() or path.read_text(encoding="utf-8") != text
    }
    changed.update(current_paths - desired_paths)
    if not changed:
        return []

    with tempfile.TemporaryDirectory(prefix="unified-history-", dir=store_root.parent) as temp:
        staged_root = Path(temp) / store_root.name
        staged_root.mkdir()
        for path, text in sorted(rendered.items()):
            staged_path = staged_root / path.relative_to(store_root)
            staged_path.parent.mkdir(parents=True, exist_ok=True)
            staged_path.write_text(text, encoding="utf-8", newline="\n")
        _load_store_unlocked(staged_root)
        publish_paths_transactionally(
            [(staged_root, store_root)],
            backup_parent=store_root.parent,
        )
    return sorted(changed)


@contextmanager
def _store_lock(store_root: Path, operation: int):
    store_root = _canonical_store_root(store_root)
    descriptor = os.open(store_root.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(descriptor, operation)
        yield
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_regular_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_candidate_payload(candidate: Path) -> None:
    if candidate.is_file():
        _fsync_regular_file(candidate)
    elif candidate.is_dir():
        directories = []
        for root, names, files in os.walk(candidate, topdown=False):
            root_path = Path(root)
            for name in names:
                path = root_path / name
                if not path.is_dir() or path.is_symlink():
                    raise ValueError(
                        f"transaction candidate has a non-directory component: {path}"
                    )
            for name in files:
                path = root_path / name
                if not path.is_file() or path.is_symlink():
                    raise ValueError(f"transaction candidate has a non-regular payload: {path}")
                _fsync_regular_file(path)
            directories.append(root_path)
        for directory in directories:
            _fsync_directory(directory)
    else:
        raise ValueError(f"transaction candidate must be a regular file or directory: {candidate}")
    _fsync_directory(candidate.parent)


def _transaction_path(parent: Path, relative: str, where: str) -> Path:
    path = parent / relative
    try:
        path.resolve(strict=False).relative_to(parent.resolve(strict=True))
    except ValueError as error:
        raise ValueError(f"transaction {where} escapes its root: {relative}") from error
    return path


def _write_transaction_journal(path: Path, document: dict) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(document, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    except (Exception, KeyboardInterrupt, SystemExit):
        temporary_path.unlink(missing_ok=True)
        raise


def _load_transaction_journal(path: Path) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != TRANSACTION_SCHEMA_VERSION:
        raise ValueError(f"unsupported transaction journal schema: {path}")
    if document.get("phase") not in {"prepared", "committed"}:
        raise ValueError(f"invalid transaction journal phase: {path}")
    states = document.get("states")
    if not isinstance(states, list) or not states:
        raise ValueError(f"transaction journal has no states: {path}")
    return document


def _cleanup_transaction(backup_parent: Path, document: dict, journal_path: Path) -> None:
    backup_root = _transaction_path(backup_parent, document["backup_root"], "backup root")
    if backup_root.exists():
        shutil.rmtree(backup_root)
    journal_path.unlink(missing_ok=True)
    _fsync_directory(backup_parent)


def _warn_pending_cleanup(journal_path: Path, error: BaseException) -> None:
    warnings.warn(
        f"fixture publication committed; cleanup remains pending in {journal_path}: {error}",
        RuntimeWarning,
        stacklevel=2,
    )


def _transaction_path_exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _remove_transaction_path(path: Path) -> None:
    if path.is_symlink() or not path.is_dir():
        path.unlink(missing_ok=True)
    else:
        shutil.rmtree(path)


def _recover_publication(backup_parent: Path) -> None:
    backup_parent = Path(backup_parent).resolve(strict=True)
    journal_path = backup_parent / TRANSACTION_JOURNAL
    if not journal_path.exists():
        return
    document = _load_transaction_journal(journal_path)
    if document["phase"] == "committed":
        try:
            _cleanup_transaction(backup_parent, document, journal_path)
        except Exception as cleanup_error:
            _warn_pending_cleanup(journal_path, cleanup_error)
        return

    errors = []
    for state in reversed(document["states"]):
        _transaction_path(backup_parent, state["candidate"], "candidate")
        destination = _transaction_path(backup_parent, state["destination"], "destination")
        backup = _transaction_path(backup_parent, state["backup"], "backup")
        had_original = state.get("had_original")
        if not isinstance(had_original, bool):
            raise ValueError(f"transaction journal has invalid original state: {journal_path}")
        try:
            if had_original:
                if _transaction_path_exists(backup):
                    _remove_transaction_path(destination)
                    os.replace(backup, destination)
                elif not _transaction_path_exists(destination):
                    raise FileNotFoundError(f"transaction original is unavailable: {destination}")
            else:
                if _transaction_path_exists(backup):
                    raise RuntimeError(f"transaction has a backup for a new destination: {backup}")
                _remove_transaction_path(destination)
            _fsync_directory(destination.parent)
            _fsync_directory(backup.parent)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as rollback_error:
            errors.append(f"{destination}: {rollback_error}")
    if errors:
        raise RuntimeError(
            f"fixture publication recovery failed; retain {journal_path}: " + "; ".join(errors)
        )
    try:
        _cleanup_transaction(backup_parent, document, journal_path)
    except Exception as cleanup_error:
        _warn_pending_cleanup(journal_path, cleanup_error)


@contextmanager
def _store_read_lock(store_root: Path):
    store_root = _canonical_store_root(store_root)
    while True:
        with _store_lock(store_root, fcntl.LOCK_SH):
            journal_path = store_root.parent / TRANSACTION_JOURNAL
            if not journal_path.exists() or (
                _load_transaction_journal(journal_path)["phase"] == "committed"
            ):
                yield
                return
        with _store_lock(store_root, fcntl.LOCK_EX):
            _recover_publication(store_root.parent)


@contextmanager
def _store_mutation_lock(store_root: Path):
    """Keep readers and writers on one complete store generation."""
    store_root = _canonical_store_root(store_root)
    with _store_lock(store_root, fcntl.LOCK_EX):
        _recover_publication(Path(store_root).parent)
        yield


def publish_paths_transactionally(
    replacements: list[tuple[Path, Path]],
    *,
    backup_parent: Path,
) -> None:
    """Publish staged paths with crash recovery through one durable commit journal."""
    backup_parent = Path(backup_parent).resolve(strict=True)
    _recover_publication(backup_parent)
    journal_path = backup_parent / TRANSACTION_JOURNAL
    if journal_path.exists():
        raise RuntimeError(f"prior fixture publication cleanup is still pending: {journal_path}")
    pairs = [
        (Path(candidate).resolve(strict=True), Path(destination).resolve(strict=False))
        for candidate, destination in replacements
    ]
    destinations = [destination for _candidate, destination in pairs]
    if len(set(destinations)) != len(destinations):
        raise ValueError("transaction has duplicate destinations")
    for candidate, destination in pairs:
        if not candidate.exists():
            raise FileNotFoundError(f"transaction candidate is missing: {candidate}")
        if not destination.parent.is_dir():
            raise FileNotFoundError(f"transaction destination parent is missing: {destination.parent}")

    for candidate, destination in pairs:
        try:
            candidate.relative_to(backup_parent)
            destination.relative_to(backup_parent)
        except ValueError as error:
            raise ValueError("transaction paths must stay under the backup parent") from error

    for candidate, _destination in pairs:
        _fsync_candidate_payload(candidate)

    backup_root = Path(tempfile.mkdtemp(prefix=".conformance-transaction-backup-", dir=backup_parent))
    states = []
    for index, (candidate, destination) in enumerate(pairs):
        backup = backup_root / f"{index:02d}-{destination.name}"
        states.append(
            {
                "candidate": str(candidate.relative_to(backup_parent)),
                "destination": str(destination.relative_to(backup_parent)),
                "backup": str(backup.relative_to(backup_parent)),
                "had_original": destination.exists(),
            }
        )
    document = {
        "schema_version": TRANSACTION_SCHEMA_VERSION,
        "phase": "prepared",
        "backup_root": str(backup_root.relative_to(backup_parent)),
        "states": states,
    }
    try:
        _write_transaction_journal(journal_path, document)
        for state in states:
            candidate = _transaction_path(backup_parent, state["candidate"], "candidate")
            destination = _transaction_path(backup_parent, state["destination"], "destination")
            backup = _transaction_path(backup_parent, state["backup"], "backup")
            if state["had_original"]:
                os.replace(destination, backup)
                _fsync_directory(destination.parent)
                _fsync_directory(backup.parent)
            os.replace(candidate, destination)
            _fsync_directory(candidate.parent)
            _fsync_directory(destination.parent)
        document["phase"] = "committed"
        _write_transaction_journal(journal_path, document)
    except (Exception, KeyboardInterrupt, SystemExit) as publish_error:
        if not journal_path.exists():
            try:
                if backup_root.exists():
                    shutil.rmtree(backup_root)
                _fsync_directory(backup_parent)
            except (Exception, KeyboardInterrupt, SystemExit) as cleanup_error:
                publish_error.add_note(
                    f"fixture publication pre-journal cleanup also failed: {cleanup_error}; "
                    f"backup root: {backup_root}"
                )
            raise
        if journal_path.exists() and _load_transaction_journal(journal_path)["phase"] == "committed":
            if isinstance(publish_error, Exception):
                raise PublicationCommitIndeterminate(
                    f"fixture publication commit is visible but not durably confirmed; "
                    f"retain recovery journal: {journal_path}"
                ) from publish_error
            publish_error.add_note(
                f"fixture publication commit is visible; retain recovery journal: {journal_path}"
            )
            raise
        try:
            _recover_publication(backup_parent)
        except (Exception, KeyboardInterrupt, SystemExit) as rollback_error:
            publish_error.add_note(
                f"fixture publication rollback also failed: {rollback_error}; "
                f"recovery journal: {journal_path}"
            )
        raise
    try:
        _cleanup_transaction(backup_parent, document, journal_path)
    except Exception as cleanup_error:
        _warn_pending_cleanup(journal_path, cleanup_error)


def _mutate_store(
    store_root: Path,
    mutate: Callable[[Store], Any],
) -> list[Path]:
    store_root = _canonical_store_root(store_root)
    with _store_mutation_lock(store_root):
        store = _load_store_unlocked(store_root)
        mutate(store)
        return _commit_store_documents(store.root, _store_documents(store))


def _case_document(metadata: dict, family: str, case_key: str, record: dict) -> dict:
    return {**metadata, "family": family, "cases": {case_key: record}}


def _materialized_case_path(
    destination: Path,
    directory: str,
    family: str,
    case_key: str,
) -> Path:
    _safe_component(directory, "materialized directory")
    _safe_component(family, "materialized family")
    _safe_component(case_key, "materialized case key")
    return destination / directory / family / f"{case_key}.yaml"


def _write_materialized_document(item: tuple[Path, dict]) -> None:
    path, document = item
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_yaml(document), encoding="utf-8", newline="\n")


def _write_materialized_documents(documents: dict[Path, dict]) -> None:
    items = sorted(documents.items())
    worker_count = min(32, os.cpu_count() or 1, max(1, len(items) // 128))
    if multiprocessing.current_process().daemon:
        worker_count = 1
    if worker_count == 1:
        for item in items:
            _write_materialized_document(item)
        return
    with multiprocessing.Pool(worker_count) as pool:
        pool.map(
            _write_materialized_document,
            items,
            chunksize=max(1, len(items) // (worker_count * 4)),
        )


def _materialized_record(case: dict, change: dict) -> tuple[dict, dict]:
    observation = change["observation"]
    state = next(iter(observation))
    if state == "value":
        record = dict(observation[state])
    else:
        payload = observation[state]
        record = {state: payload.get("detail", payload.get("message", payload["code"]))}
    stimulus = change["stimulus"]
    if "ref" in stimulus:
        record["capture_input"] = case["request"]
    elif "inline" in stimulus:
        record["capture_input"] = stimulus["inline"]
    return record, change["document"]


def _capture_release_sort_key(runtime_version: str) -> tuple:
    release_version = runtime_version
    release, separator, prerelease = release_version.partition("-")
    numeric = tuple(int(part) for part in release.split("."))
    return numeric, 0 if separator else 1, prerelease


def materialize_store(
    root: Path,
    destination: Path,
    *,
    include_current_inputs: bool = True,
    derived_release_versions: dict[str, str] | None = None,
) -> None:
    store = load_store(root)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    written: set[Path] = set()
    documents: dict[Path, dict] = {}

    def add_document(
        directory: str,
        family_name: str,
        case_key: str,
        document: dict,
    ) -> None:
        path = _materialized_case_path(
            destination,
            directory,
            family_name,
            case_key,
        )
        relative = path.relative_to(destination)
        if relative in written:
            raise ValueError(f"duplicate materialized path: {relative}")
        written.add(relative)
        documents[path] = document

    for family_name, family in sorted(store.families.items()):
        if not include_current_inputs:
            continue
        input_metadata = family.document.get("input_document") or {
            "family": family_name,
            "mode": "unified",
            "model_label": family_name,
        }
        golden_metadata = family.document.get("golden_document") or {
            "family": family_name,
            "mode": "unified",
            "captured_with": {"golden": "v1"},
        }
        for case in family.cases.values():
            if case["lifecycle"] != "active":
                continue
            case_key = case["display_id"]
            add_document(
                "inputs",
                family_name,
                case_key,
                _case_document(
                    input_metadata,
                    family_name,
                    case_key,
                    {
                        "scenario": case["scenario"],
                        "description": case["description"],
                        "policy": case["policy"],
                        "init": case["request"]["init"],
                        "finish_reason": case["request"]["finish_reason"],
                        "input": case["request"]["input"],
                        "tools": case["request"]["tools"],
                        "chunks": case["request"]["chunks"],
                    },
                ),
            )
            add_document(
                "golden",
                family_name,
                case_key,
                _case_document(golden_metadata, family_name, case_key, case["golden"]),
            )

    def add_capture_state(directory: str, history: History, state: dict) -> None:
        family_name = history.family.name
        for case_id, change in sorted(state.items()):
            case = history.family.cases[case_id]
            case_key = change["case_key"]
            record, document_metadata = _materialized_record(case, change)
            document_metadata = dict(document_metadata)
            # Rust readers pass this field to the shared Python selector. Keep a
            # generated schema marker so an omitted legacy provenance and an
            # intentional schema-v3 semantic view do not collapse into null.
            document_metadata["capture_provenance"] = {"format": "schema_v3"}
            record_metadata = document_metadata.pop("record_metadata", {})
            record.update(record_metadata)
            document = _case_document(document_metadata, family_name, case_key, record)
            relative = Path(directory) / family_name / f"{case_key}.yaml"
            if relative not in written:
                add_document(directory, family_name, case_key, document)

    capture_ids_by_implementation: dict[str, set[str]] = {}
    for (family_name, _implementation), history in sorted(store.histories.items()):
        capture_ids_by_implementation.setdefault(history.implementation, set()).update(
            history.captures
        )
        for capture_id, capture in history.captures.items():
            add_capture_state(capture_id, history, history.resolve(capture_id))

    # The YAML store remains sparse: a release with no changed family output has
    # no checkpoint file. Consumers still need a complete directory for the
    # released version, so extraction may request a derived release view.
    for implementation, runtime_version in (derived_release_versions or {}).items():
        if not isinstance(implementation, str) or not isinstance(runtime_version, str):
            raise ValueError("derived release versions must map strings to strings")
        if not fixture_disposition.DYNAMO_VERSION_RE.fullmatch(runtime_version):
            raise ValueError(f"invalid derived release version: {runtime_version}")
        if implementation not in capture_ids_by_implementation:
            raise ValueError(f"no Unified captures for derived implementation: {implementation}")
        capture_ids_by_implementation[implementation].add(
            f"{implementation}-{runtime_version}"
        )

    # A capture directory is a complete release view. If only one family was
    # recaptured for a source identity, carry every other family forward from its
    # newest capture at the same or an earlier crate version.
    for implementation, target_ids in sorted(capture_ids_by_implementation.items()):
        histories = [
            history
            for (_family, impl), history in sorted(store.histories.items())
            if impl == implementation
        ]
        for target_id in sorted(
            target_ids,
            key=lambda capture_id: _capture_release_sort_key(
                capture_id.removeprefix(f"{implementation}-")
            ),
        ):
            target_version = target_id.removeprefix(f"{implementation}-")
            target_release = _capture_release_sort_key(target_version)[:3]
            for history in histories:
                if target_id in history.captures:
                    continue
                eligible = [
                    capture_id
                    for capture_id, capture in history.captures.items()
                    if _capture_release_sort_key(capture["runtime_version"])[:3]
                    <= target_release
                ]
                if not eligible:
                    continue
                source_id = max(
                    eligible,
                    key=lambda capture_id: _capture_release_sort_key(
                        history.captures[capture_id]["runtime_version"]
                    ),
                )
                add_capture_state(target_id, history, history.resolve(source_id))

    _write_materialized_documents(documents)

def _internal_case_id(case_key: str, scenario: str | None, used: set[str]) -> str:
    stem = scenario or "retired__" + case_key.removeprefix("UNIFIED.")
    stem = re.sub(r"[^a-zA-Z0-9_]+", "_", stem).strip("_").lower()
    candidate = stem or "case"
    suffix = 2
    while candidate in used:
        candidate = f"{stem}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def _legacy_state(record: dict, raw: bytes, relative: str, bindings: dict, current: dict | None) -> tuple[dict, dict, dict]:
    original = capture_stimulus.original_capture_input(record, raw, relative, bindings)
    if original is None:
        stimulus = {"unavailable": {"code": "original_request_not_retained"}}
    elif current is not None and original == capture_stimulus.capture_input(current):
        stimulus = {
            "ref": "current",
            "semantic_sha256": _request_digest(original),
        }
    elif isinstance(original, dict) and original.keys() == REQUEST_KEYS and original.get("tools") is not None:
        stimulus = {"inline": original, "semantic_sha256": _request_digest(original)}
    else:
        stimulus = {"partial": original}

    value = dict(record)
    value.pop("capture_input", None)
    value_fields = value.keys() & {"assembled", "chunks"}
    error_fields = value.keys() & {"error", "unavailable"}
    if bool(value_fields) + len(error_fields) != 1:
        raise ValueError(
            f"capture record must contain exactly one observation state: {relative}"
        )
    metadata = {
        key: value.pop(key)
        for key in tuple(value)
        if key not in {"assembled", "chunks", "error", "unavailable"}
    }
    if "error" in value:
        detail = value.pop("error")
        observation = {"error": {"code": "legacy_unclassified", "detail": detail}}
    elif "unavailable" in value:
        detail = value.pop("unavailable")
        observation = {"unavailable": {"code": "legacy_unclassified", "detail": detail}}
    else:
        observation = {"value": value}
    return stimulus, observation, metadata


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _semantic_stimulus(value: dict) -> dict:
    return {key: item for key, item in value.items() if key != "semantic_sha256"}


def _capture_semantic(value: dict, case_id: str) -> dict:
    """Return immutable capture content keyed by the canonical case identity.

    A current loose corpus can use a newer display ID for a case whose historical
    capture still stores an older alias. The canonical case ID is stable across
    that rename; the display key is retained in new deltas but must not make an
    identical re-ingestion look like a mutation of a released capture.
    """
    return {
        "case_id": case_id,
        "stimulus": _semantic_stimulus(value["stimulus"]),
        "observation": value["observation"],
    }


def _stored_change(value: dict) -> dict:
    return {key: copy.deepcopy(value[key]) for key in CHANGE_KEYS}


def _load_loose_document(path: Path, family_name: str) -> dict:
    document = load_yaml(path)
    if document.get("family") != family_name:
        raise ValueError(f"loose Unified family differs from directory: {path}")
    _mapping(document.get("cases"), f"{path}.cases")
    return document


def _update_from_loose(
    store: Store,
    loose_root: Path,
    *,
    complete_snapshot: bool,
    excluded_capture_dirs: set[str] | frozenset[str] = frozenset(),
    required_capture_dirs: set[str] | frozenset[str] = frozenset(),
) -> dict[Path, dict]:
    loose_root = Path(loose_root)
    documents = {}
    if not loose_root.is_dir():
        return documents
    excluded_capture_dirs = {
        _safe_component(directory, "excluded capture directory")
        for directory in excluded_capture_dirs
    }
    required_capture_dirs = {
        _safe_component(directory, "required capture directory")
        for directory in required_capture_dirs
    }
    capture_dirs = {
        path.name
        for path in loose_root.iterdir()
        if (
            path.is_dir()
            and not is_generated_oracle_directory(path.name)
            and re.match(r"^[a-z0-9_]+-\d", path.name)
        )
    }
    malformed_capture_dirs = sorted(
        name for name in capture_dirs if CAPTURE_DIRECTORY_RE.fullmatch(name) is None
    )
    if malformed_capture_dirs:
        raise ValueError(
            "Unified capture directories must use <implementation>-<semantic-version>; "
            f"found: {', '.join(malformed_capture_dirs)}"
        )
    if complete_snapshot:
        missing_required_capture_dirs = sorted(required_capture_dirs - capture_dirs)
        if missing_required_capture_dirs:
            missing = ", ".join(missing_required_capture_dirs)
            raise ValueError(f"complete snapshot is missing required captures: {missing}")
    for capture_dir in sorted(
        path
        for path in loose_root.iterdir()
        if (
            path.is_dir()
            and CAPTURE_DIRECTORY_RE.fullmatch(path.name) is not None
            and path.name not in excluded_capture_dirs
        )
    ):
        match = CAPTURE_DIRECTORY_RE.fullmatch(capture_dir.name)
        assert match is not None
        implementation = match["implementation"]
        runtime_version = match["runtime_version"]
        families = sorted(path.name for path in capture_dir.iterdir() if path.is_dir())
        required_capture = capture_dir.name in required_capture_dirs
        if complete_snapshot:
            if required_capture:
                expected_active_families = sorted(
                    family_name
                    for (family_name, history_implementation), history in store.histories.items()
                    if (
                        history_implementation == implementation
                        and any(
                            case["lifecycle"] == "active"
                            for case in history.family.cases.values()
                        )
                    )
                )
            else:
                expected_active_families = sorted(
                    family_name
                    for (family_name, history_implementation), history in store.histories.items()
                    if (
                        history_implementation == implementation
                        and capture_dir.name in history.captures
                        and any(
                            history.family.cases[case_id]["lifecycle"] == "active"
                            for case_id in history.resolve(capture_dir.name)
                        )
                    )
                )
            missing_active_families = sorted(set(expected_active_families) - set(families))
            if required_capture and families:
                target_release = _capture_release_sort_key(runtime_version)[:3]
                missing_active_families = [
                    family_name
                    for family_name in missing_active_families
                    if not any(
                        _capture_release_sort_key(capture["runtime_version"])[:3]
                        <= target_release
                        for capture in store.histories[
                            (family_name, implementation)
                        ].captures.values()
                    )
                ]
            if missing_active_families:
                missing = ", ".join(missing_active_families)
                raise ValueError(
                    f"complete capture {capture_dir.name} is missing active families: {missing}"
                )
        for family_name in families:
            key = (family_name, implementation)
            history = store.histories.get(key)
            if history is None:
                raise ValueError(f"no history file owns {capture_dir.name}/{family_name}")
            case_by_external = {}
            for case_id, case in history.family.cases.items():
                for external_id in [case["display_id"], *case["historical_ids"]]:
                    if external_id is not None:
                        case_by_external[
                            (
                                family_name,
                                fixture_disposition.historical_unified_case_key(
                                    family_name, external_id
                                ),
                            )
                        ] = case_id
            records = {}
            for path in sorted((capture_dir / family_name).glob("*.yaml")):
                raw = path.read_bytes()
                document = _load_loose_document(path, family_name)
                metadata = {name: value for name, value in document.items() if name != "cases"}
                # A schema-v3 marker belongs only to the extracted compatibility
                # view. It must not become canonical capture metadata when that
                # view is ingested again.
                if metadata.get("capture_provenance") == {"format": "schema_v3"}:
                    metadata.pop("capture_provenance")
                for case_key, record in document["cases"].items():
                    external = (
                        family_name,
                        fixture_disposition.historical_unified_case_key(family_name, case_key),
                    )
                    case_id = case_by_external.get(external)
                    if case_id is None:
                        raise ValueError(
                            f"new Unified case {family_name}/{case_key} needs a canonical family entry"
                        )
                    current = history.family.cases[case_id]["request"]
                    stimulus, observation, record_metadata = _legacy_state(
                        record,
                        raw,
                        str(path.relative_to(capture_dir)),
                        capture_stimulus.read_bindings(capture_dir),
                        current,
                    )
                    if case_id in records:
                        previous_key = records[case_id]["case_key"]
                        raise ValueError(
                            f"duplicate Unified records resolve to {family_name}/{case_id}: "
                            f"{previous_key}, {case_key}"
                        )
                    records[case_id] = {
                        "case_key": case_key,
                        "stimulus": stimulus,
                        "observation": observation,
                        "document": {
                            **metadata,
                            **({"record_metadata": record_metadata} if record_metadata else {}),
                        },
                    }

            captures = history.captures
            if complete_snapshot and required_capture:
                missing_active_cases = [
                    case_id
                    for case_id, case in sorted(history.family.cases.items())
                    if case["lifecycle"] == "active" and case_id not in records
                ]
                if missing_active_cases:
                    missing = ", ".join(
                        f"{family_name}/{case_id}" for case_id in missing_active_cases
                    )
                    raise ValueError(
                        f"complete capture {capture_dir.name} is missing active cases: {missing}"
                    )
            if capture_dir.name in captures:
                resolved = history.resolve(capture_dir.name)
                missing_active_cases = [
                    case_id
                    for case_id in sorted(set(resolved) - set(records))
                    if history.family.cases[case_id]["lifecycle"] == "active"
                ]
                if complete_snapshot and not required_capture and missing_active_cases:
                    missing = ", ".join(
                        f"{family_name}/{case_id}" for case_id in missing_active_cases
                    )
                    raise ValueError(
                        f"complete capture {capture_dir.name} is missing active cases: {missing}"
                    )
                conflicts = [
                    case_id
                    for case_id in sorted(set(resolved) & set(records))
                    if _canonical_json(_capture_semantic(resolved[case_id], case_id))
                    != _canonical_json(_capture_semantic(records[case_id], case_id))
                ]
                if conflicts:
                    raise ValueError(f"capture is immutable; use a new identity: {capture_dir.name}")
                additions = sorted(set(records) - set(resolved))
                if not additions:
                    continue
                raise ValueError(
                    f"capture {capture_dir.name} is already recorded; add a new semantic "
                    "version after back-capturing any new case across prior versions"
                )
            prior_id = history.ordered_capture_ids()[-1]
            if _capture_release_sort_key(runtime_version) <= _capture_release_sort_key(
                history.captures[prior_id]["runtime_version"]
            ):
                raise ValueError(
                    f"capture {capture_dir.name} must use a new semantic version after "
                    f"{prior_id}"
                )
            prior = history.resolve(prior_id)
            changes = {}
            metadata_changes = {}
            for case_id in sorted(set(prior) | set(records)):
                before = prior.get(case_id)
                after = records.get(case_id)
                # Display IDs are renumbered independently of capture semantics.
                # Compare by the stable case ID so a renamed case is inherited,
                # while a changed observation or stimulus still creates a capture.
                before_key = (
                    None
                    if before is None
                    else _canonical_json(_capture_semantic(before, case_id))
                )
                after_key = (
                    None
                    if after is None
                    else _canonical_json(_capture_semantic(after, case_id))
                )
                if before_key != after_key:
                    changes[case_id] = (
                        {"absent": True} if after is None else _stored_change(after)
                    )
                before_metadata = (
                    {} if before is None else before["document"].get("record_metadata", {})
                )
                after_metadata = (
                    {} if after is None else after["document"].get("record_metadata", {})
                )
                if after is not None and _canonical_json(before_metadata) != _canonical_json(
                    after_metadata
                ):
                    metadata_changes[case_id] = after_metadata
            origins = {
                _canonical_json(record["document"]["capture_origin"])
                for record in records.values()
                if "capture_origin" in record["document"]
            }
            if origins:
                if len(origins) != 1:
                    raise ValueError(f"new capture has mixed source origins: {capture_dir.name}")
                provenance = {"status": "captured", "origin": json.loads(next(iter(origins)))}
            else:
                provenance = {
                    "status": "legacy",
                    "captured_with": {implementation: runtime_version},
                }
            document_overrides = {
                case_id: {"parser_path": record["document"]["parser_path"]}
                for case_id, record in records.items()
                if "parser_path" in record["document"]
            }
            # Extraction may derive a complete semantic release directory from
            # an earlier checkpoint. That inherited view is not a new capture.
            if not changes and not metadata_changes and not document_overrides:
                continue
            capture = {
                "runtime_version": runtime_version,
                "provenance": provenance,
                "document": {"mode": "unified"},
                "changes": changes,
                "metadata_changes": metadata_changes,
                "document_overrides": document_overrides,
            }
            captures[capture_dir.name] = capture
            history.capture_paths[capture_dir.name] = (
                history.family.path.parent / f"{capture_dir.name}.yaml"
            )
    return documents


def update_from_loose(
    store_root: Path,
    loose_root: Path,
    *,
    excluded_capture_dirs: set[str] | frozenset[str] = frozenset(),
) -> list[Path]:
    return _mutate_store(
        store_root,
        lambda store: _update_from_loose(
            store,
            loose_root,
            complete_snapshot=False,
            excluded_capture_dirs=excluded_capture_dirs,
        ),
    )


def _preserve_historical_stimulus(
    store: Store,
    family_name: str,
    case_id: str,
    request: dict,
    documents: dict[Path, dict],
) -> None:
    raise ValueError(
        "Unified test inputs are immutable after capture; changing "
        f"{family_name}/{case_id} requires recapturing and updating every prior semantic version"
    )


def _shared_corpus_roots(loose_root: Path, base: str) -> list[Path]:
    """The canonical authored corpus has one current root per document type."""
    return [loose_root / base]


def _sync_current_corpus(
    store: Store,
    loose_root: Path,
    *,
    complete_snapshot: bool,
) -> dict[Path, dict]:
    loose_root = Path(loose_root)
    documents = {}
    inputs_root = loose_root / "inputs"
    golden_root = loose_root / "golden"
    if not inputs_root.is_dir() or not golden_root.is_dir():
        if complete_snapshot:
            raise ValueError("complete current snapshot needs inputs and golden roots")
        return documents
    input_roots = _shared_corpus_roots(loose_root, "inputs")
    golden_roots = _shared_corpus_roots(loose_root, "golden")

    family_documents = {}
    for family_name in sorted(store.families):
        input_paths = [
            path
            for root in input_roots
            for path in sorted((root / family_name).glob("*.yaml"))
        ]
        golden_paths = [
            path
            for root in golden_roots
            for path in sorted((root / family_name).glob("*.yaml"))
        ]
        if complete_snapshot and (not input_paths or not golden_paths):
            raise ValueError(f"complete current snapshot is missing family: {family_name}")
        input_documents = [
            (path, _load_loose_document(path, family_name)) for path in input_paths
        ]
        golden_documents = {}
        for path in golden_paths:
            document = _load_loose_document(path, family_name)
            for case_key in document["cases"]:
                if case_key in golden_documents:
                    raise ValueError(
                        f"duplicate current Unified golden case for {family_name}/{case_key}"
                    )
                golden_documents[case_key] = (path, document)
        if complete_snapshot:
            input_case_keys = {
                case_key
                for _path, document in input_documents
                for case_key in document["cases"]
            }
            if input_case_keys != golden_documents.keys():
                raise ValueError(
                    f"complete current snapshot input/golden cases differ for {family_name}"
                )
        family_documents[family_name] = (input_documents, golden_documents)

    known_families = set(store.families)
    for root in (*input_roots, *golden_roots):
        unknown = sorted(
            path.name for path in root.iterdir() if path.is_dir() and path.name not in known_families
        )
        if unknown:
            raise ValueError(f"current Unified corpus has unknown families under {root}: {unknown}")

    for family_name, family in sorted(store.families.items()):
        cases = copy.deepcopy(family.cases)
        by_scenario = {
            case["scenario"]: case_id
            for case_id, case in cases.items()
            if isinstance(case["scenario"], str)
        }
        current_scenarios = set()
        scenario_owners = {}
        input_documents, golden_documents = family_documents[family_name]
        if not input_documents:
            continue
        for input_path, input_document in input_documents:
            for case_key, request in input_document["cases"].items():
                scenario = request.get("scenario")
                if not isinstance(scenario, str):
                    raise ValueError(f"current Unified input has no scenario: {input_path}")
                description = request.get("description")
                policy = request.get("policy")
                _validate_case_metadata(
                    description,
                    policy,
                    f"{input_path}:{case_key}",
                    active=True,
                )
                previous_owner = scenario_owners.get(scenario)
                if previous_owner is not None:
                    raise ValueError(
                        f"duplicate current Unified scenario for {family_name}/{scenario}: "
                        f"{previous_owner}, {input_path}:{case_key}"
                    )
                scenario_owners[scenario] = f"{input_path}:{case_key}"
                current_scenarios.add(scenario)
                case_id = by_scenario.get(scenario)
                if case_id is None:
                    case_id = _internal_case_id(case_key, scenario, set(cases))
                    cases[case_id] = {
                        "lifecycle": "active",
                        "scenario": scenario,
                        "description": description,
                        "policy": policy,
                        "display_id": case_key,
                        "historical_ids": [],
                        "request": None,
                        "golden": None,
                    }
                golden_entry = golden_documents.get(case_key)
                if golden_entry is None:
                    raise ValueError(f"current Unified input has no golden: {family_name}/{case_key}")
                _golden_path, golden_document = golden_entry
                if case_key not in golden_document["cases"]:
                    raise ValueError(
                        f"current Unified golden has no matching case: {family_name}/{case_key}"
                    )
                case = cases[case_id]
                old_display = case["display_id"]
                aliases = set(case["historical_ids"])
                if old_display and old_display != case_key:
                    aliases.add(old_display)
                aliases.discard(case_key)
                old_request = case["request"]
                new_request = capture_stimulus.capture_input(request)
                new_request["tools"] = request.get("tools")
                if old_request is not None and _canonical_json(old_request) != _canonical_json(
                    new_request
                ):
                    _preserve_historical_stimulus(
                        store,
                        family_name,
                        case_id,
                        old_request,
                        documents,
                    )
                cases[case_id] = {
                    "lifecycle": "active",
                    "scenario": scenario,
                    "description": description,
                    "policy": policy,
                    "display_id": case_key,
                    "historical_ids": sorted(aliases),
                    "request": new_request,
                    "golden": golden_document["cases"][case_key],
                }
                family.document["input_document"] = {
                    key: value for key, value in input_document.items() if key != "cases"
                }
                family.document["golden_document"] = {
                    key: value for key, value in golden_document.items() if key != "cases"
                }
        for case in cases.values():
            case.setdefault("description", None)
            case.setdefault("policy", [])
            if (
                complete_snapshot
                and case["lifecycle"] == "active"
                and case["scenario"] not in current_scenarios
            ):
                aliases = set(case["historical_ids"])
                if case["display_id"]:
                    aliases.add(case["display_id"])
                case.update(
                    {
                        "lifecycle": "retired",
                        "display_id": None,
                        "historical_ids": sorted(aliases),
                    }
                )
        family.document["cases"] = {
            case_id: {key: case[key] for key in CASE_KEY_ORDER}
            for case_id, case in cases.items()
        }
        documents[family.path] = family.document
    return documents


def sync_current_corpus(
    store_root: Path,
    loose_root: Path,
    *,
    complete_snapshot: bool = False,
) -> list[Path]:
    return _mutate_store(
        store_root,
        lambda store: _sync_current_corpus(
            store,
            loose_root,
            complete_snapshot=complete_snapshot,
        ),
    )


def update_store_from_loose(
    store_root: Path,
    loose_root: Path,
    *,
    complete_snapshot: bool,
    excluded_capture_dirs: set[str] | frozenset[str] = frozenset(),
    required_capture_dirs: set[str] | frozenset[str] = frozenset(),
) -> list[Path]:
    def documents(store: Store) -> dict[Path, dict]:
        updated = _sync_current_corpus(
            store,
            loose_root,
            complete_snapshot=complete_snapshot,
        )
        updated.update(
            _update_from_loose(
                store,
                loose_root,
                complete_snapshot=complete_snapshot,
                excluded_capture_dirs=excluded_capture_dirs,
                required_capture_dirs=required_capture_dirs,
            )
        )
        return updated

    return _mutate_store(store_root, documents)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--materialize", action="store_true")
    action.add_argument("--validate", action="store_true")
    action.add_argument("--update-from-loose", action="store_true")
    action.add_argument("--sync-current-corpus", action="store_true")
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--history-only", action="store_true")
    parser.add_argument(
        "--complete-snapshot",
        action="store_true",
        help="retire active cases omitted from a complete loose corpus",
    )
    args = parser.parse_args(argv)
    if args.materialize:
        if args.output is None:
            parser.error("--materialize requires --output")
        materialize_store(args.store, args.output, include_current_inputs=not args.history_only)
    elif args.update_from_loose:
        if args.output is None:
            parser.error("--update-from-loose requires --output")
        print(json.dumps([str(path) for path in update_from_loose(args.store, args.output)], indent=2))
    elif args.sync_current_corpus:
        if args.output is None:
            parser.error("--sync-current-corpus requires --output")
        print(
            json.dumps(
                [
                    str(path)
                    for path in sync_current_corpus(
                        args.store,
                        args.output,
                        complete_snapshot=args.complete_snapshot,
                    )
                ],
                indent=2,
            )
        )
    else:
        store = load_store(args.store)
        digest, size = store_digest(args.store)
        print(
            json.dumps(
                {
                    "families": len(store.families),
                    "histories": len(store.histories),
                    "sha256": digest,
                    "size": size,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
