# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load, validate, update, and materialize the Unified YAML history store."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import re
import shutil
import tempfile
import warnings
from collections import ChainMap
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from yaml.composer import Composer
from yaml.constructor import SafeConstructor
from yaml.cyaml import CParser
from yaml.resolver import BaseResolver, Resolver

import capture_stimulus
import capture_sources
from capture_versions import provenance_release_version, UNIFIED_VERSION_PATTERN, UNIFIED_VERSION_RE, capture_release_sort_key as _capture_release_sort_key
import impls
import fixture_disposition
import execution_manifests
import unified_archives
import generation_workers
from viewer_codec import canonical_typed, decode_typed, typed_equal


FAMILY_DOCUMENT_NAME = "inputs_and_golden.yaml"
MATERIALIZED_FORMAT = "unified_history"
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
CORRECTABLE_CAPTURE_UNAVAILABLE_CODES = {"vllm_tool_schema_unsupported"}
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
CHANGE_KEY_ORDER = ("case_key", "stimulus", "observation")
CHANGE_KEYS = set(CHANGE_KEY_ORDER)
OBSERVATION_STATES = {"value", "error", "unavailable"}
STIMULUS_STATES = {"ref", "inline", "partial", "error", "unavailable"}
REQUEST_KEYS = {"input", "init", "finish_reason", "tools", "chunks"}
CAPTURE_DIRECTORY_RE = re.compile(
    rf"(?P<implementation>[a-z0-9_]+)-(?P<runtime_version>{UNIFIED_VERSION_PATTERN})"
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


def dump_yaml(value: dict, *, shared_metadata: bool = False) -> str:
    class NoAliasDumper(yaml.CSafeDumper if shared_metadata else yaml.SafeDumper):
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
        digest = stimulus.get("semantic_sha256")
        if digest is not None and digest != _request_digest(stimulus["inline"]):
            raise ValueError(f"{where} stimulus digest differs from its inline request")
    elif state == "partial":
        if not isinstance(stimulus["partial"], dict):
            raise ValueError(f"{where}.partial must be a mapping")
    elif state in {"error", "unavailable"}:
        typed_state = _mapping(stimulus[state], f"{where}.{state}")
        if not isinstance(typed_state.get("code"), str):
            raise ValueError(f"{where}.{state} needs a string code")


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


def _validate_resolved_record_contract(
    implementation: str,
    document: dict,
    observation: dict,
    where: str,
    *,
    source_version: str | None = None,
) -> None:
    if "capture_origin" in document:
        _validate_source_origin(document["capture_origin"], f"{where}.capture_origin", source_version)
    if capture_stimulus.unified_record_contract(implementation, document) != "batch" or "value" not in observation:
        return
    value = observation["value"]
    metadata = _mapping(document.get("record_metadata", {}), f"{where}.record_metadata")
    final = metadata.get("batch_final", {})
    if "chunks" in value or not typed_equal(value.get("assembled"), capture_stimulus.batch_final_events(final)):
        raise ValueError(f"{where} batch observation does not match its final fields")


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
    resolved_snapshots: dict[str, dict] = field(default_factory=dict)
    snapshot_members: dict[str, dict[str, str]] = field(default_factory=dict)
    snapshot_suppressed_inheritance: dict[str, set[str]] = field(default_factory=dict)
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

    def resolve(self, capture_id: str, *, view_capture_id: str | None = None) -> dict:
        if capture_id not in self.captures:
            raise ValueError(f"unknown capture in {self.directory}: {capture_id}")
        return next(self.iter_resolved_snapshots({capture_id}, view_capture_id=view_capture_id))[1]

    def complete_snapshot(self, capture_id: str, *, records: dict | None = None,
                          source_member: str | None = None) -> dict:
        if records is None:
            records = next(self.iter_resolved_snapshots({capture_id}, include_members=True))[1]
        capture = self.captures[capture_id]
        members = self.snapshot_members[capture_id] if source_member is None else {}
        suppressed = self.snapshot_suppressed_inheritance.get(capture_id, set())
        document = {"schema_version": 2, "provenance": copy.deepcopy(capture["provenance"]),
                    "document": copy.deepcopy(capture["document"]), "records": {}}
        for case, value in records.items():
            document["records"][case] = {
                "value": copy.deepcopy(value), "member": source_member or members[case],
                **({"suppress_inherited_from": True} if case in suppressed else {}),
            }
        for field in ("execution_proof", "historical_evidence"):
            if field in capture:
                document[field] = copy.deepcopy(capture[field])
        return document

    def _apply_capture(self, state: dict, current: str) -> None:
        capture = self.captures[current]
        for case_id, change in capture["changes"].items():
            if change == {"absent": True}:
                state.pop(case_id, None)
                continue
            previous = state.get(case_id)
            if previous is None:
                if set(change) != CHANGE_KEYS:
                    raise ValueError(
                        f"first capture change must provide a complete record in "
                        f"{self.capture_path(current)}: {case_id}"
                    )
                state[case_id] = copy.deepcopy(change)
                state[case_id]["_origin_capture_id"] = current
            else:
                state[case_id] = copy.deepcopy(previous)
                state[case_id].update(copy.deepcopy(change))
                if "observation" in change:
                    state[case_id].pop("_document_overrides", None)
                    state[case_id]["_origin_capture_id"] = current
            inherited_metadata = previous.get("_record_metadata") if previous else None
            inherited_overrides = previous.get("_document_overrides") if previous else None
            if inherited_metadata is not None:
                state[case_id]["_record_metadata"] = inherited_metadata
            if inherited_overrides is not None and "parser_path" in inherited_overrides:
                state[case_id].setdefault("_document_overrides", {})[
                    "parser_path"
                ] = copy.deepcopy(inherited_overrides["parser_path"])
        for case_id, metadata in capture["metadata_changes"].items():
            if case_id not in state:
                raise ValueError(
                    f"metadata change has no observation in {self.capture_path(current)}: "
                    f"{case_id}"
                )
            if metadata:
                state[case_id]["_record_metadata"] = copy.deepcopy(metadata)
                state[case_id]["_record_metadata_capture_id"] = current
            else:
                state[case_id].pop("_record_metadata", None)
                state[case_id].pop("_record_metadata_capture_id", None)
        for case_id, override in capture["document_overrides"].items():
            if case_id not in state:
                raise ValueError(
                    f"document override has no observation in {self.capture_path(current)}: "
                    f"{case_id}"
                )
            state.setdefault(case_id, {}).setdefault("_document_overrides", {}).update(
                copy.deepcopy(override)
            )

    def _materialize_state(self, state: dict, current: str, *, validate_batch: bool = True, view_capture_id: str | None = None) -> dict:
        view_capture_id = view_capture_id or current
        materialized = {}
        for case_id, change in state.items():
            record = copy.deepcopy(change)
            record_metadata = record.pop("_record_metadata", None)
            metadata_capture_id = record.pop("_record_metadata_capture_id", None)
            if record_metadata and metadata_capture_id != view_capture_id:
                # Peer import and report readers must not treat an earlier recapture
                # as a verified execution of a later checkpoint or derived view.
                record_metadata.pop("verified_recapture", None)
            overrides = record.pop("_document_overrides", {})
            origin_capture_id = record.pop("_origin_capture_id")
            origin_capture = self.captures[origin_capture_id]
            case_document = _capture_document_metadata(origin_capture, self.implementation)
            if origin_capture_id != view_capture_id:
                case_document["inherited_from"] = origin_capture["runtime_version"]
            provenance_override = overrides.pop("capture_provenance", None)
            if provenance_override is not None:
                for key, value in provenance_override.items():
                    if value is None:
                        case_document.pop(key, None)
                    else:
                        case_document[key] = copy.deepcopy(value)
            if "parser_path" in overrides and overrides["parser_path"] is None:
                overrides.pop("parser_path")
                case_document.pop("parser_path", None)
            case_document.update(overrides)
            if record_metadata:
                case_document["record_metadata"] = record_metadata
            if validate_batch:
                _validate_resolved_record_contract(
                    self.implementation,
                    case_document,
                    record["observation"],
                    f"{case_id} batch history",
                    source_version=case_document["captured_with"][self.implementation],
                )
            record["document"] = case_document
            materialized[case_id] = record
        return materialized

    def iter_resolved_snapshots(self, capture_ids: set[str] | None = None, *, include_members: bool = False, view_capture_id: str | None = None):
        """Resolve checkpoints in one pass, with independently owned yielded records."""
        if self.resolved_snapshots:
            for current in self.ordered_capture_ids():
                if capture_ids is None or current in capture_ids:
                    records = copy.deepcopy(self.resolved_snapshots[current])
                    if view_capture_id and view_capture_id != current:
                        suppressed = self.snapshot_suppressed_inheritance.get(current, set())
                        for case_id, record in records.items():
                            document = record["document"]
                            if "record_metadata" in document:
                                document["record_metadata"].pop("verified_recapture", None)
                                if not document["record_metadata"]:
                                    document.pop("record_metadata")
                            if case_id not in suppressed:
                                document.setdefault("inherited_from", self.captures[current]["runtime_version"])
                    yield current, records
            return
        state = {}
        for current in self.ordered_capture_ids():
            self._apply_capture(state, current)
            if capture_ids is None or current in capture_ids:
                if include_members:
                    # Observation lineage survives document provenance overrides.
                    self.snapshot_members[current] = {
                        case_id: f"conformance/fixtures-unified-v2/families/{self.family.name}/{record['_origin_capture_id']}.yaml"
                        for case_id, record in state.items()
                    }
                records = self._materialize_state(state, current, view_capture_id=view_capture_id)
                if include_members:
                    self.snapshot_suppressed_inheritance[current] = {
                        case_id for case_id, record in state.items()
                        if "inherited_from" not in records[case_id]["document"]
                        and record.get("_document_overrides", {}).get("capture_provenance", {}).get("inherited_from", False) is None
                    }
                yield current, records


class _HistoryCursor:
    """Preview a rebuilt checkpoint before committing its provenance overrides."""

    def __init__(self, history: History):
        self.history = history
        self.state = {}

    def preview(self, capture_id: str) -> dict:
        state = copy.deepcopy(self.state)
        self.history._apply_capture(state, capture_id)
        # Natural defaults may temporarily give retained stream data a batch
        # origin. Validate only after the required provenance override is restored.
        return self.history._materialize_state(state, capture_id, validate_batch=False)

    def advance(self, capture_id: str) -> None:
        self.history._apply_capture(self.state, capture_id)
        self.history._materialize_state(self.state, capture_id)


@dataclass
class Store:
    root: Path
    families: dict[str, Family]
    histories: dict[tuple[str, str], History]
    resources: dict[str, dict] = field(default_factory=dict)

    def history_for_capture(self, family_name: str, implementation: str) -> History:
        key = (family_name, implementation)
        if key not in self.histories:
            family = self.families[family_name]
            self.histories[key] = History(family, implementation, {}, family.path.parent, {})
        return self.histories[key]


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


def _validate_source_origin(origin: Any, where: str, runtime_version: str | None = None) -> dict:
    origin = _mapping(origin, where)
    _require_keys(origin, {"crate_version", "source_sha256"}, {"crate_version", "source_sha256", "git_commit", "adapter_sha256", "job_sha256"}, where)
    if not isinstance(origin["crate_version"], str):
        raise ValueError(f"capture origin has invalid crate_version: {where}")
    provenance_release_version(origin["crate_version"])
    if runtime_version is not None:
        release = provenance_release_version(runtime_version)
        if origin["crate_version"] != release:
            raise ValueError(f"capture origin differs from runtime identity: {where}")
        suffix = runtime_version[len(release):]
        if suffix and suffix != f"+source.{origin['source_sha256']}":
            raise ValueError(f"capture origin differs from source identity: {where}")
    for key in ("source_sha256", "adapter_sha256", "job_sha256"):
        if key in origin and not re.fullmatch(r"[0-9a-f]{64}", origin[key]):
            raise ValueError(f"capture origin has invalid source hash ({key}): {where}")
    if ("adapter_sha256" in origin) != ("job_sha256" in origin):
        raise ValueError(f"capture origin needs both adapter and job identities: {where}")
    if "git_commit" in origin and not re.fullmatch(r"[0-9a-f]{40}", origin["git_commit"]):
        raise ValueError(f"capture origin has invalid git commit: {where}")
    return origin


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
        _validate_source_origin(provenance["origin"], f"{path}.provenance.origin", runtime_version)
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
    *, resource_index=None, proof_requests=None,
) -> tuple[str, str, str, dict]:
    parent = document.get("parent")
    if "parent" in document and parent is None:
        raise ValueError(f"invalid null capture parent in {path}")
    if "execution" in document:
        _require_keys(document, {"execution"}, {"execution", "parent", "changes"}, str(path))
        identity = CAPTURE_DIRECTORY_RE.fullmatch(path.stem)
        execution = document["execution"]
        if identity is None:
            raise ValueError(f"invalid execution manifest identity in {path}")
        manifest_path = path.parents[2] / execution_manifests.source_reference(execution, "executions")
        if manifest_path.is_symlink():
            raise ValueError(f"execution manifest must be a regular file: {manifest_path}")
        manifest = load_yaml(manifest_path)
        if resource_index is not None:
            resource_index[manifest_path.relative_to(path.parents[2]).as_posix()] = {"receipts": sorted(manifest.get("receipts", {}))}
        receipt_documents = {}
        memberships = manifest.get("receipts", {})
        if not isinstance(memberships, dict):
            raise ValueError("execution receipt membership must be a mapping")
        for digest in memberships:
            receipt_path = path.parents[2] / execution_manifests.source_reference(digest, "receipts")
            if receipt_path.is_symlink():
                raise ValueError(f"execution receipt must be a regular file: {receipt_path}")
            receipt_documents[digest] = load_yaml(receipt_path)
            if resource_index is not None:
                resource_index[digest] = {"receipt_identity": canonical_typed(receipt_documents[digest])}
        expanded = execution_manifests.expand(manifest, family=path.parent.name,
            implementation=identity["implementation"], version=identity["runtime_version"], receipt_documents=receipt_documents)
        document = {**expanded, "changes": document.get("changes", {})}
    else:
        document = {key: value for key, value in document.items() if key != "parent"}
    _require_keys(document, CAPTURE_DOCUMENT_KEYS, CAPTURE_DOCUMENT_KEYS | {"execution_proof"}, str(path))
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
    if parent is not None:
        if not isinstance(parent, str) or CAPTURE_DIRECTORY_RE.fullmatch(parent) is None:
            raise ValueError(f"invalid capture parent in {path}")
        capture["parent"] = parent
    provenance = _validate_capture_provenance(
        capture["provenance"], path, implementation, runtime_version
    )
    if "execution_proof" in capture:
        proof = _mapping(capture["execution_proof"], f"{path}.execution_proof")
        _require_keys(proof, {"receipt", "requests"}, {"receipt", "requests"}, str(path))
        receipt = _mapping(proof["receipt"], f"{path}.execution_proof.receipt")
        if receipt.get("implementation") != implementation or receipt.get("version") != runtime_version:
            raise ValueError(f"execution proof producer identity mismatch: {path}")
        origin = provenance.get("origin", {})
        if provenance["status"] != "captured" or any(receipt.get(key) != value for key, value in origin.items() if key != "crate_version"):
            raise ValueError(f"execution proof differs from checkpoint origin: {path}")
        if any(not isinstance(receipt.get(key), str) or re.fullmatch(r"[0-9a-f]{64}", receipt[key]) is None for key in ("source_sha256", "adapter_sha256", "job_sha256", "results_sha256")):
            raise ValueError(f"execution proof lacks sealed receipt hashes: {path}")
        requests = _mapping(proof["requests"], f"{path}.execution_proof.requests")
        if not requests or any(not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None for digest in requests.values()):
            raise ValueError(f"execution proof requires sealed request digests: {path}")
        expected = proof_requests if proof_requests is not None else {key: _request_digest(case["request"]) for key, case in families[family_name].cases.items() if case.get("lifecycle", "active") != "retired"}
        if not typed_equal(requests, expected):
            raise ValueError(f"execution proof request membership mismatch: {path}")
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
        if not change or set(change) - CHANGE_KEYS:
            raise ValueError(f"invalid change fields in {path}:{case_id}: {sorted(change)}")
        if "case_key" in change:
            _safe_component(change["case_key"], f"{path}:{case_id}.case_key")
        if "stimulus" in change:
            _validate_stimulus(
                change["stimulus"],
                families[family_name].cases[case_id],
                f"{path}:{case_id}.stimulus",
            )
        if "observation" in change:
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
        if set(metadata) - {"parser_path", "capture_provenance"}:
            raise ValueError(f"unsupported document override in {path}:{case_id}")
        if "parser_path" in metadata and metadata["parser_path"] is not None and not isinstance(metadata["parser_path"], str):
            raise ValueError(f"{path}:{case_id}.document_overrides.parser_path must be a string or null")
        if "capture_provenance" in metadata:
            provenance = _mapping(
                metadata["capture_provenance"],
                f"{path}:{case_id}.document_overrides.capture_provenance",
            )
            if set(provenance) != {"captured_with", "capture_origin", "inherited_from"}:
                raise ValueError(
                    f"{path}:{case_id}.document_overrides.capture_provenance has invalid keys"
                )
            captured_with = _mapping(
                provenance["captured_with"],
                f"{path}:{case_id}.document_overrides.capture_provenance.captured_with",
            )
            if not captured_with or any(
                not isinstance(implementation, str)
                or not isinstance(version, str)
                for implementation, version in captured_with.items()
            ):
                raise ValueError(
                    f"{path}:{case_id}.document_overrides.capture_provenance.captured_with is invalid"
                )
            origin = provenance["capture_origin"]
            if origin is not None:
                _validate_source_origin(origin, f"{path}:{case_id}.document_overrides.capture_provenance.capture_origin")
            if provenance["inherited_from"] is not None and not isinstance(
                provenance["inherited_from"], str
            ):
                raise ValueError(
                    f"{path}:{case_id}.document_overrides.capture_provenance has invalid inherited_from"
                )
    return family_name, implementation, capture_id, capture


def _validate_complete_snapshot(path: Path, snapshot: dict, family: Family,
                                implementation: str, version: str) -> None:
    where = str(path)
    _require_keys(snapshot, {"schema_version", "provenance", "document", "records"},
                  {"schema_version", "provenance", "document", "records", "execution_proof", "historical_evidence", "capture_selection"}, where)
    if type(snapshot["schema_version"]) is not int or snapshot["schema_version"] != 2:
        raise ValueError(f"invalid Unified snapshot schema: {where}")
    provenance = _validate_capture_provenance(snapshot["provenance"], path, implementation, version)
    document = _mapping(snapshot["document"], f"{where}.document")
    if set(document) & set(DOCUMENT_DERIVED_KEYS):
        raise ValueError(f"snapshot document repeats derived provenance: {where}")
    records = _mapping(snapshot["records"], f"{where}.records")
    for case_id, wrapped in records.items():
        case = family.cases.get(case_id)
        if case is None:
            raise ValueError(f"unknown case in {where}: {case_id}")
        wrapped = _mapping(wrapped, f"{where}:{case_id}")
        _require_keys(wrapped, {"value", "member"}, {"value", "member", "suppress_inherited_from"}, f"{where}:{case_id}")
        if "suppress_inherited_from" in wrapped and wrapped["suppress_inherited_from"] is not True:
            raise ValueError(f"invalid snapshot inheritance projection: {where}:{case_id}")
        member = wrapped["member"]
        if not isinstance(member, str):
            raise ValueError(f"invalid snapshot source member: {where}:{case_id}")
        match = re.fullmatch(
            rf"conformance/fixtures-unified-v2/families/{re.escape(family.name)}/"
            r"(?P<implementation>[A-Za-z0-9_.+-]+)-(?P<version>[0-9][A-Za-z0-9.+-]*)\.yaml",
            member,
        )
        if match is None or match["implementation"] != implementation:
            raise ValueError(f"invalid snapshot source member: {where}:{case_id}")
        if _capture_release_sort_key(match["version"]) > _capture_release_sort_key(version):
            raise ValueError(f"snapshot source member is too new: {where}:{case_id}")
        value = _mapping(wrapped["value"], f"{where}:{case_id}.value")
        _require_keys(value, {"case_key", "stimulus", "observation", "document"},
                      {"case_key", "stimulus", "observation", "document"}, f"{where}:{case_id}.value")
        _safe_component(value["case_key"], f"{where}:{case_id}.case_key")
        _validate_stimulus(value["stimulus"], case, f"{where}:{case_id}.stimulus")
        _validate_observation(value["observation"], f"{where}:{case_id}.observation")
        record_document = _mapping(value["document"], f"{where}:{case_id}.document")
        captured_with = _mapping(record_document.get("captured_with"), f"{where}:{case_id}.captured_with")
        producer_version = captured_with.get(implementation)
        if (not isinstance(producer_version, str)
                or _capture_release_sort_key(provenance_release_version(producer_version)) > _capture_release_sort_key(version)):
            raise ValueError(f"invalid snapshot record producer version: {where}:{case_id}")
        if "record_metadata" in record_document:
            _mapping(record_document["record_metadata"], f"{where}:{case_id}.record_metadata")
        _validate_resolved_record_contract(
            implementation,
            record_document,
            value["observation"],
            f"{where}:{case_id}",
            source_version=producer_version,
        )
    historical = snapshot.get("historical_evidence", [])
    if not isinstance(historical, list):
        raise ValueError(f"{where}.historical_evidence must be a list")
    for index, evidence in enumerate(historical):
        evidence_where = f"{where}.historical_evidence[{index}]"
        evidence = _mapping(evidence, evidence_where)
        required = {"kind", "schema_version", "family", "implementation", "version", "checkpoint",
                    "cases", "defaults", "exceptions", "metadata_changes", "receipt_membership"}
        _require_keys(evidence, required, required | {"execution_proof"}, evidence_where)
        if (evidence["kind"] != "execution_manifest" or evidence["schema_version"] != 1
                or evidence["family"] != family.name or evidence["implementation"] != implementation
                or evidence["version"] != version):
            raise ValueError(f"historical execution evidence identity mismatch: {evidence_where}")
        _mapping(evidence["checkpoint"], f"{evidence_where}.checkpoint")
        _mapping(evidence["defaults"], f"{evidence_where}.defaults")
        _mapping(evidence["exceptions"], f"{evidence_where}.exceptions")
        cases = evidence["cases"]
        if not isinstance(cases, list) or any(not isinstance(case_id, str) for case_id in cases) or len(cases) != len(set(cases)):
            raise ValueError(f"invalid historical evidence case membership: {evidence_where}")
        metadata = _mapping(evidence["metadata_changes"], f"{evidence_where}.metadata_changes")
        if set(metadata) != set(cases):
            raise ValueError(f"historical evidence metadata membership mismatch: {evidence_where}")
        for case_id, value in metadata.items():
            if case_id not in family.cases:
                raise ValueError(f"unknown historical evidence case: {evidence_where}:{case_id}")
            _mapping(value, f"{evidence_where}:{case_id}.metadata_changes")
        if not isinstance(evidence["receipt_membership"], list):
            raise ValueError(f"invalid historical receipt membership: {evidence_where}")
        for receipt_index, receipt in enumerate(evidence["receipt_membership"]):
            receipt_where = f"{evidence_where}.receipt_membership[{receipt_index}]"
            receipt = _mapping(receipt, receipt_where)
            _require_keys(receipt, {"members", "body"}, {"members", "body"}, receipt_where)
            members = receipt["members"]
            if not isinstance(members, list) or any(not isinstance(case_id, str) for case_id in members) or len(members) != len(set(members)):
                raise ValueError(f"invalid historical receipt case membership: {receipt_where}")
            _mapping(receipt["body"], f"{receipt_where}.body")
    if "execution_proof" in snapshot:
        proof_capture = {"provenance": provenance, "document": document,
                         "changes": {}, "metadata_changes": {}, "document_overrides": {},
                         "execution_proof": snapshot["execution_proof"]}
        # The receipt binds this checkpoint's complete request set. Observation
        # deduplication can retain older origins without changing execution membership.
        executed = {}
        for key, wrapped in records.items():
            stimulus = wrapped["value"]["stimulus"]
            if "ref" not in stimulus and "inline" not in stimulus:
                raise ValueError(f"execution proof requires a complete request: {path}:{key}")
            executed[key] = stimulus.get("semantic_sha256")
        _validate_capture_file(path, proof_capture, {family.name: family}, proof_requests=executed)


def _validate_history(
    family: Family,
    implementation: str,
    captures: dict,
    capture_paths: dict[str, Path],
    snapshot_callback=None,
    resolved_snapshots: dict[str, dict] | None = None,
    snapshot_members: dict[str, dict[str, str]] | None = None,
    snapshot_suppressed_inheritance: dict[str, set[str]] | None = None,
    external_captures: set[str] | None = None,
) -> History:
    where = family.path.parent
    if not captures:
        raise ValueError(f"history has no captures: {where}/{implementation}")
    history = History(family, implementation, captures, where, capture_paths,
                      resolved_snapshots or {}, snapshot_members or {}, snapshot_suppressed_inheritance or {})
    if history.resolved_snapshots:
        for capture_id in history.ordered_capture_ids():
            capture = captures[capture_id]
            for case_id, member in history.snapshot_members[capture_id].items():
                source_name = Path(member).name.removesuffix(".yaml")
                source = CAPTURE_DIRECTORY_RE.fullmatch(source_name)
                if (source is None or source["implementation"] != implementation
                        or source_name not in captures and source_name not in (external_captures or set())
                        or _capture_release_sort_key(source["runtime_version"])
                        > _capture_release_sort_key(capture["runtime_version"])):
                    raise ValueError(f"snapshot source member is unavailable or too new: {capture_paths[capture_id]}:{case_id}")
            if snapshot_callback is not None:
                snapshot_callback(history, capture_id, history.resolved_snapshots[capture_id])
        return history
    previous = None
    for capture_id in history.ordered_capture_ids():
        capture = captures[capture_id]
        if "parent" in capture and capture["parent"] != previous:
            raise ValueError(f"capture parent is not the preceding checkpoint: {capture_paths[capture_id]}")
        previous = capture_id
    for capture_id, state in history.iter_resolved_snapshots(include_members=snapshot_callback is not None):
        capture = captures[capture_id]
        missing = sorted(set(capture["document_overrides"]) - set(state))
        if missing:
            raise ValueError(
                f"document override has no observation in {capture_paths[capture_id]}: {missing}"
            )
        if snapshot_callback is not None:
            snapshot_callback(history, capture_id, state)
    return history


def _canonical_store_root(root: Path) -> Path:
    return Path(root).resolve(strict=False)


def _load_family(item, snapshot_callback=None, *, external_captures=None):
    root, family = item
    grouped: dict[tuple[str, str], dict[str, dict]] = {}
    grouped_paths: dict[tuple[str, str], dict[str, Path]] = {}
    family_capture_owners: dict[str, tuple[str, str]] = {}
    resources = {}
    resolved_snapshots = {}
    snapshot_members = {}
    snapshot_suppressed_inheritance = {}
    family_name = family.name
    entries = sorted(family.path.parent.iterdir())
    for path in entries:
        if path == family.path:
            continue
        if path.is_dir() or path.suffix != ".yaml" or path.name == "index.yaml":
            raise ValueError(f"unknown Unified family file: {path}")
        document = load_yaml(path)
        relative = path.relative_to(root).as_posix()
        resources[relative] = {"execution": document.get("execution")}
        if document.get("schema_version") == 2:
            family_name = family.name
            identity = CAPTURE_DIRECTORY_RE.fullmatch(path.stem)
            if identity is None:
                raise ValueError(f"invalid Unified snapshot identity in {path}")
            implementation = identity["implementation"]
            capture_id = path.stem
            _validate_complete_snapshot(path, document, family, implementation, identity["runtime_version"])
            owner = family_capture_owners.setdefault(capture_id, (implementation, family_name))
            if owner[0] != implementation:
                raise ValueError(f"capture implementation differs across paths: {capture_id}")
            key = (family_name, implementation)
            captures = grouped.setdefault(key, {})
            paths = grouped_paths.setdefault(key, {})
            if capture_id in captures:
                raise ValueError(f"duplicate capture: {family_name}/{capture_id}")
            captures[capture_id] = {
                "runtime_version": identity["runtime_version"],
                "provenance": document["provenance"],
                "document": document["document"],
                "changes": {},
                "metadata_changes": {},
                "document_overrides": {},
                **({"execution_proof": document["execution_proof"]} if "execution_proof" in document else {}),
                **({"historical_evidence": document["historical_evidence"]} if "historical_evidence" in document else {}),
            }
            paths[capture_id] = path
            resolved_snapshots[capture_id] = {
                case_id: copy.deepcopy(record["value"])
                for case_id, record in document["records"].items()
            }
            snapshot_members[capture_id] = {
                case_id: record["member"] for case_id, record in document["records"].items()
            }
            snapshot_suppressed_inheritance[capture_id] = {
                case_id for case_id, record in document["records"].items() if record.get("suppress_inherited_from")
            }
            continue
        capture_family, implementation, capture_id, capture = _validate_capture_file(
            path,
            document,
            {family.name: family},
            resource_index=resources,
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
        snapshot_ids = set(captures) & set(resolved_snapshots)
        if snapshot_ids and snapshot_ids != set(captures):
            raise ValueError(f"mixed sparse and complete snapshots in {family_name}/{implementation}")
        histories[key] = _validate_history(
            family,
            implementation,
            captures,
            grouped_paths[key],
            snapshot_callback,
            {capture_id: resolved_snapshots[capture_id] for capture_id in snapshot_ids},
            {capture_id: snapshot_members[capture_id] for capture_id in snapshot_ids},
            {capture_id: snapshot_suppressed_inheritance[capture_id] for capture_id in snapshot_ids},
            external_captures=external_captures,
        )
    return family.name, histories, family_capture_owners, resources


def _load_store_unlocked(root: Path) -> Store:
    root = _canonical_store_root(root)
    with unified_archives.expanded(root) as expanded:
        store = _load_expanded_store_unlocked(expanded)
        if expanded != root:
            for family in store.families.values():
                family.path = root / family.path.relative_to(expanded)
            for history in store.histories.values():
                history.directory = root / history.directory.relative_to(expanded)
                history.capture_paths = {key: root / path.relative_to(expanded)
                                         for key, path in history.capture_paths.items()}
            store.root = root
        return store


def _load_expanded_store_unlocked(root: Path) -> Store:
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
    unknown_root_entries = sorted(path for path in root.iterdir() if path.name not in {"families", "executions", "receipts"})
    if unknown_root_entries:
        raise ValueError(f"unknown Unified root entry: {unknown_root_entries[0]}")
    resource_index = {}
    for directory in ("executions", "receipts"):
        execution_root = root / directory
        if not execution_root.exists():
            continue
        if execution_root.is_symlink() or not execution_root.is_dir():
            raise ValueError("execution manifests and receipts must be regular directories")
        for path in execution_root.iterdir():
            execution_manifests.source_reference(f"{directory}/{path.name}", directory)
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"invalid execution manifest path: {path}")

    families = _load_families_unlocked(root)

    histories = {}
    owners = {}
    for name, family_histories, capture_owners, resources in generation_workers.bounded_map(
        _load_family, ((root, family) for family in families.values()),
        phase="validate-families", jobs=len(families), ordered=False,
    ):
        for capture_id, owner in capture_owners.items():
            if capture_id in owners and owners[capture_id][0] != owner[0]:
                raise ValueError(f"capture implementation differs across paths: {capture_id}")
            owners[capture_id] = owner
        for history in family_histories.values():
            history.family = families[name]
        histories.update(family_histories)
        resource_index.update(resources)
    if not histories:
        raise ValueError(f"Unified history has no history files: {root}")
    return Store(root, families, dict(sorted(histories.items())), resource_index)


def _load_families_unlocked(root: Path) -> dict[str, Family]:
    families_root = root / "families"
    if families_root.is_symlink() or not families_root.is_dir():
        raise ValueError(f"Unified families must be a regular directory: {families_root}")
    family_directories = sorted(path for path in families_root.iterdir() if path.is_dir())
    unknown_family_entries = sorted(path for path in families_root.iterdir() if not path.is_dir())
    if unknown_family_entries:
        raise ValueError(f"unknown Unified family entry: {unknown_family_entries[0]}")
    if any(directory.is_symlink() for directory in family_directories):
        raise ValueError("Unified family directories must not be links")
    family_paths = [directory / FAMILY_DOCUMENT_NAME for directory in family_directories]
    if not family_paths:
        raise ValueError(f"Unified history has no family files: {root}")
    families = {}
    for path in family_paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Unified family has no {FAMILY_DOCUMENT_NAME}: {path.parent}")
        family = _validate_family(path, load_yaml(path))
        if family.name in families:
            raise ValueError(f"duplicate family: {family.name}")
        families[family.name] = family

    return families


def load_authored_store(root: Path) -> Store:
    """Validate authored inputs without opening or retaining historical captures."""
    root = _canonical_store_root(root)
    with _store_read_lock(root):
        return Store(root, _load_families_unlocked(root), {})


def mutate_authored_store(root: Path, mutate: Callable[[Store], Any]) -> list[Path]:
    """Append authored cases without rewriting the current observation snapshots."""
    root = _canonical_store_root(root)
    with _store_mutation_lock(root), tempfile.TemporaryDirectory(prefix=".authored-stage-", dir=root.parent) as temporary:
        store = Store(root, _load_families_unlocked(root), {})
        before = {family.path: load_yaml(family.path) for family in store.families.values()}
        mutate(store)
        pairs = []
        for index, family in enumerate(store.families.values()):
            if typed_equal(before[family.path], family.document):
                continue
            _validate_family(family.path, family.document)
            candidate = Path(temporary) / f"{index}.yaml"
            candidate.write_text(dump_yaml(family.document))
            pairs.append((candidate, family.path))
        if pairs:
            publish_paths_transactionally(pairs, backup_parent=root.parent)
        return [destination for _candidate, destination in pairs]


def load_store(root: Path) -> Store:
    root = _canonical_store_root(root)
    checkout = root.parent.parent
    if capture_sources.is_current(checkout):
        document = capture_sources.current(checkout)
        implementations = sorted({row["implementation"] for row in document["captures"] if row["surface"] == "unified"})
        store, _ = selected_materialization_store(root, [f"latest:{implementation}" for implementation in implementations])
        return store
    if (checkout / capture_sources.DATASETS).is_file():
        return _routed_history_store(root)
    with _store_read_lock(root):
        return _load_store_unlocked(root)


class _FragmentMapping(Mapping):
    """Decode one validated checkpoint at a time instead of expanding the corpus."""

    def __init__(self, directory: Path, checkpoints: dict, field: str):
        self.directory = directory
        self.checkpoints = checkpoints
        self.field = field

    def __len__(self):
        return len(self.checkpoints)

    def __iter__(self):
        return iter(self.checkpoints)

    def __getitem__(self, capture_id):
        row = self.checkpoints[capture_id]
        key = row["fragment"]
        if re.fullmatch(r"[0-9a-f]{64}", key) is None:
            raise ValueError("invalid history fragment identity")
        path = self.directory / f"{key}.json"
        if path.is_symlink():
            raise ValueError("history fragment must be a regular file")
        entry = json.loads(path.read_bytes())
        payload = entry["payload"]
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        if entry["identity"] != key or entry["sha256"] != hashlib.sha256(encoded).hexdigest():
            raise ValueError("history fragment hash mismatch")
        records = decode_typed(payload)
        if self.field == "records":
            return {case: wrapped["value"] for case, wrapped in records.items()}
        if self.field == "suppressed":
            return {case for case, wrapped in records.items() if wrapped.get("suppress_inherited_from")}
        return {case: f"conformance/history-datasets/{row['dataset']}/{wrapped['member']}"
                for case, wrapped in records.items()}


def _routed_history_store(root: Path) -> Store:
    checkout = root.parent.parent
    catalog = json.loads((checkout / "conformance/viewer-catalog.json").read_bytes())
    cache = Path(json.loads((checkout / "conformance/history-reader-cache.json").read_bytes())["directory"])
    families = _load_families_unlocked(root)
    checkpoints = {}
    for dataset in catalog["sources"]["history_datasets"]:
        retained = _load_families_unlocked(checkout / dataset["root"] / capture_sources.STORE)
        for name, family in retained.items():
            current = families[name]
            for case_id, case in family.cases.items():
                if case_id not in current.cases:
                    current.cases[case_id] = {**copy.deepcopy(case), "lifecycle": "retired", "display_id": None, "historical_ids": []}
        for row in dataset["checkpoints"]:
            capture_id = f"{row['implementation']}-{row['version']}"
            checkpoints.setdefault((row["family"], row["implementation"]), {})[capture_id] = {**row, "dataset": dataset["id"]}
    histories = {}
    for (family, implementation), rows in checkpoints.items():
        captures = {identity: {
            "runtime_version": row["version"],
            "provenance": decode_typed(row["capture"]["provenance"]),
            "document": decode_typed(row["capture"]["document"]),
            "changes": {}, "metadata_changes": {}, "document_overrides": {},
            **{field: decode_typed(row["capture"][field]) for field in ("execution_proof", "historical_evidence")
               if field in row["capture"]},
        } for identity, row in rows.items()}
        paths = {identity: checkout / f"conformance/history-datasets/{row['dataset']}/{capture_sources.STORE}/families/{family}/{identity}.yaml"
                 for identity, row in rows.items()}
        histories[family, implementation] = History(families[family], implementation, captures,
            families[family].path.parent, paths, _FragmentMapping(cache, rows, "records"),
            _FragmentMapping(cache, rows, "members"), _FragmentMapping(cache, rows, "suppressed"))
    # Administratively added complete checkpoints take precedence at their exact identity.
    for family in families.values():
        if any(path.name != FAMILY_DOCUMENT_NAME for path in family.path.parent.iterdir()):
            external = {capture_id for (name, _), history in histories.items() if name == family.name
                        for capture_id in history.captures}
            _, updates, _, _ = _load_family((root, family), external_captures=external)
            for identity, update in updates.items():
                history = histories.get(identity)
                if history is None:
                    histories[identity] = update
                    continue
                # ChainMap retains lazy fragments without copying every snapshot.
                history.captures.update(update.captures)
                history.capture_paths.update(update.capture_paths)
                history.resolved_snapshots = ChainMap(update.resolved_snapshots, history.resolved_snapshots)
                history.snapshot_members = ChainMap(update.snapshot_members, history.snapshot_members)
                history.snapshot_suppressed_inheritance = ChainMap(update.snapshot_suppressed_inheritance, history.snapshot_suppressed_inheritance)
    return Store(root, families, histories)


def _store_inventory_unlocked(root: Path) -> list[dict]:
    return [{key: row[key] for key in ("path", "sha256", "size")}
            for row in unified_archives.inventory(Path(root))]


def store_inventory(root: Path) -> list[dict]:
    root = _canonical_store_root(root)
    with _store_read_lock(root):
        return _store_inventory_unlocked(root)


def store_digest(root: Path) -> tuple[str, int]:
    inventory = store_inventory(root)
    encoded = json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), sum(item["size"] for item in inventory)


def rewrite_store(root: Path) -> None:
    def compact(store: Store) -> None:
        for history in store.histories.values():
            _compact_history(history)

    _mutate_store(root, compact)


def _capture_document(history: History, capture_id: str) -> dict:
    capture = history.captures[capture_id]
    return {
        key: value for key, value in capture.items() if key not in {"runtime_version", "parent"}
    }


def _store_documents(store: Store, *, compact_executions: bool | None = None) -> dict[Path, dict]:
    if compact_executions is None:
        compact_executions = (store.root / "executions").exists()
    documents = {family.path: family.document for family in store.families.values()}
    receipts = {}
    receipt_names = {}
    for name, value in sorted(store.resources.items()):
        if "receipt_identity" in value:
            receipt_names.setdefault(value["receipt_identity"], name)
    for history in store.histories.values():
        if history.resolved_snapshots:
            for capture_id in history.ordered_capture_ids():
                documents[history.capture_path(capture_id)] = history.complete_snapshot(
                    capture_id, records=history.resolved_snapshots[capture_id])
            continue
        previous = None
        for capture_id in history.ordered_capture_ids():
            document = _capture_document(history, capture_id)
            manifest = execution_manifests.compact(document, family=history.family.name,
                implementation=history.implementation, version=history.captures[capture_id]["runtime_version"],
                receipt_documents=receipts, receipt_names=receipt_names) if compact_executions else None
            if manifest is not None:
                existing = store.resources.get(history.capture_path(capture_id).relative_to(store.root).as_posix(), {})
                execution = existing.get("execution") or f"executions/{history.family.name}--{capture_id}.yaml"
                documents[store.root / execution] = manifest
                document = {"execution": execution, **({"changes": document["changes"]} if document["changes"] else {})}
            if manifest is not None and previous is not None:
                document = {"parent": previous, **document}
            documents[history.capture_path(capture_id)] = document
            previous = capture_id
    documents.update({store.root / reference: value for reference, value in receipts.items()})
    return documents


def compact_execution_store(root: Path) -> list[Path]:
    """Factor receipts only after complete decoded-record parity in staging."""
    root = _canonical_store_root(root)
    with _store_mutation_lock(root):
        store = _load_store_unlocked(root)
        documents = _store_documents(store, compact_executions=True)
        return _commit_store_documents(root, documents, expected_store=store)


def archive_versions(root: Path) -> dict:
    """Install engine/version archives only after exact logical-byte parity."""
    root = _canonical_store_root(root)
    with _store_mutation_lock(root):
        before = _store_inventory_unlocked(root)
        with tempfile.TemporaryDirectory(prefix="unified-archive-", dir=root.parent) as temporary:
            staged = Path(temporary) / root.name
            with unified_archives.expanded(root) as expanded:
                shutil.copytree(expanded, staged)
            unified_archives.pack(staged, CAPTURE_DIRECTORY_RE, _capture_release_sort_key)
            if _store_inventory_unlocked(staged) != before:
                raise ValueError("archiving changed logical YAML bytes or membership")
            files = [path for path in staged.rglob("*") if path.is_file()]
            result = {"logical_documents": len(before), "files": len(files),
                      "bytes": sum(path.stat().st_size for path in files),
                      "archives": len(unified_archives.paths(staged))}
            publish_paths_transactionally([(staged, root)], backup_parent=root.parent)
            return result


def _validate_store_parity(store: Store, candidate: Store) -> None:
    if set(candidate.families) != set(store.families) or any(
        not typed_equal(family.document, candidate.families[name].document)
        for name, family in store.families.items()
    ):
        raise ValueError("execution compaction changed authored inputs or GOLDEN")
    if set(candidate.histories) != set(store.histories):
        raise ValueError("execution compaction changed history membership")
    for key, history in store.histories.items():
        other = candidate.histories[key]
        if history.ordered_capture_ids() != other.ordered_capture_ids():
            raise ValueError("execution compaction changed checkpoint membership")
        for expected, actual in zip(history.iter_resolved_snapshots(include_members=True), other.iter_resolved_snapshots(include_members=True)):
            capture_id = expected[0]
            if history.snapshot_suppressed_inheritance.get(capture_id, set()) != other.snapshot_suppressed_inheritance.get(capture_id, set()):
                raise ValueError(f"Unified snapshot inheritance projection differs: {capture_id}")
            if not typed_equal(expected, actual):
                raise ValueError(f"execution compaction changed decoded records: {key}/{expected[0]}")


def _commit_complete_documents(store_root: Path, documents: dict[Path, dict], expected_store: Store | None) -> list[Path]:
    desired = {path.relative_to(store_root).as_posix(): document for path, document in documents.items()}
    retained = set()
    present = set()

    def compare(name, data, _archive):
        present.add(name)
        if name in desired and typed_equal(yaml.load(data, Loader=StrictLoader), desired[name]):
            retained.add(name)

    unified_archives.inventory(store_root, member_callback=compare)
    changes = {
        name: (dump_yaml(desired[name], shared_metadata=True).encode("utf-8") if name in desired else None)
        for name in (present | set(desired)) - retained
    }
    if not changes:
        return []
    with tempfile.TemporaryDirectory(prefix="unified-history-", dir=store_root.parent) as temporary:
        staged = Path(temporary) / store_root.name
        if unified_archives.paths(store_root):
            unified_archives.stage_selected(store_root, staged, changes, CAPTURE_DIRECTORY_RE, _capture_release_sort_key)
        else:
            shutil.copytree(store_root, staged, copy_function=os.link)
            for name, data in changes.items():
                target = staged / name
                target.unlink(missing_ok=True)
                if data is not None:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(data)
        candidate = _load_store_unlocked(staged)
        if expected_store is not None:
            _validate_store_parity(expected_store, candidate)
        publish_paths_transactionally([(staged, store_root)], backup_parent=store_root.parent)
    return [store_root / name for name in sorted(changes)]


def _commit_store_documents(store_root: Path, documents: dict[Path, dict], *, expected_store: Store | None = None) -> list[Path]:
    """Validate and publish one complete store generation."""
    store_root = Path(store_root)
    captures = [document for path, document in documents.items()
                if path.parent.parent.name == "families" and path.name != FAMILY_DOCUMENT_NAME]
    if captures and all(document.get("schema_version") == 2 for document in captures):
        return _commit_complete_documents(store_root, documents, expected_store)
    items = sorted(documents.items())
    rendered = dict(generation_workers.bounded_map(
        _render_document, items, phase="render-store", jobs=max(1, len(items) // 128),
    ))
    current = {store_root / row["path"]: row for row in _store_inventory_unlocked(store_root)}
    current_paths = set(current)
    desired_paths = set(rendered)
    changed = {
        path
        for path, text in rendered.items()
        if path not in current or current[path]["sha256"] != hashlib.sha256(text.encode("utf-8")).hexdigest()
    }
    changed.update(current_paths - desired_paths)
    if not changed:
        if expected_store is not None:
            _validate_store_parity(expected_store, _load_store_unlocked(store_root))
        return []

    with tempfile.TemporaryDirectory(prefix="unified-history-", dir=store_root.parent) as temp:
        staged_root = Path(temp) / store_root.name
        staged_root.mkdir()
        for path, text in sorted(rendered.items()):
            staged_path = staged_root / path.relative_to(store_root)
            staged_path.parent.mkdir(parents=True, exist_ok=True)
            staged_path.write_text(text, encoding="utf-8", newline="\n")
        if (store_root / unified_archives.DIRECTORY).exists():
            unified_archives.pack(staged_root, CAPTURE_DIRECTORY_RE, _capture_release_sort_key)
        candidate = _load_store_unlocked(staged_root)
        if expected_store is not None:
            _validate_store_parity(expected_store, candidate)
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
    if candidate.is_symlink():
        raise ValueError(f"transaction candidate must be a regular file or directory: {candidate}")
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
    *,
    compact_executions: bool | None = None,
    expected_inventory: list[dict] | None = None,
    skip_no_sources: bool = False,
) -> list[Path]:
    store_root = _canonical_store_root(store_root)
    with _store_mutation_lock(store_root):
        if expected_inventory is not None and not typed_equal(_store_inventory_unlocked(store_root), expected_inventory):
            raise ValueError("store generation changed before mutation")
        store = _load_store_unlocked(store_root)
        updated = mutate(store)
        # Loose-source readers report None only when no applicable source was visited.
        # An empty document map can still accompany corrections to existing history.
        if skip_no_sources and updated is None:
            return []
        compact = (store.root / "executions").exists() if compact_executions is None else compact_executions
        return _commit_store_documents(store.root, _store_documents(store, compact_executions=compact),
                                       expected_store=store if compact or any(history.resolved_snapshots for history in store.histories.values()) else None)


def _case_document(metadata: dict, family: str, case_key: str, record: dict) -> dict:
    return {**metadata, "family": family, "cases": {case_key: record}}


def migrate_descriptive_ids(store_root: Path) -> list[Path]:
    """Change active display metadata, retaining capture bytes and published aliases."""
    def migrate(store: Store) -> None:
        for family in store.families.values():
            for case in family.cases.values():
                if case["lifecycle"] != "active":
                    continue
                old_id = case["display_id"]
                new_id = fixture_disposition.canonical_unified_case_key(
                    family.name, old_id, case["scenario"],
                )
                if old_id != new_id:
                    case["historical_ids"] = sorted(set(case["historical_ids"]) | {old_id})
                    case["display_id"] = new_id

    return _mutate_store(store_root, migrate)


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


def _render_document(item: tuple[Path, dict]) -> tuple[Path, str]:
    path, document = item
    return path, dump_yaml(document, shared_metadata=path.parent.name in {"executions", "receipts"})


def _write_materialized_document(item: tuple[Path, dict]) -> None:
    path, text = _render_document(item)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _write_materialized_documents(documents: dict[Path, dict]) -> None:
    items = sorted(documents.items())
    for _ in generation_workers.bounded_map(
        _write_materialized_document, items, phase="materialize-documents", jobs=max(1, len(items) // 128),
    ):
        pass


def _materialized_record(case: dict, change: dict) -> tuple[dict, dict]:
    observation = change["observation"]
    state = next(iter(observation))
    if state == "value":
        record = dict(observation[state])
    else:
        payload = observation[state]
        record = {state: capture_stimulus.capture_failure_detail(payload)}
    stimulus = change["stimulus"]
    if "ref" in stimulus:
        record["capture_input"] = case["request"]
    elif "inline" in stimulus:
        record["capture_input"] = stimulus["inline"]
    else:
        record["capture_stimulus"] = copy.deepcopy(stimulus)
    if state in {"error", "unavailable"}:
        record["capture_observation"] = {state: copy.deepcopy(observation[state])}
    return record, change["document"]


def materialized_record(case: dict, change: dict) -> tuple[dict, dict]:
    return _materialized_record(case, change)


def source_identity(relative: str) -> dict:
    """Classify a logical source filename without decoding historical YAML."""
    path = Path(relative)
    parts = path.parts
    match = CAPTURE_DIRECTORY_RE.fullmatch(path.stem)
    return {
        "family": parts[1] if len(parts) >= 3 and parts[0] == "families" else None,
        "implementation": match["implementation"] if match else None,
        "version": match["runtime_version"] if match else None,
        "kind": "receipt" if parts[0] == "receipts" else "execution" if parts[0] == "executions" else "history" if match else "inputs-and-golden",
    }


def selected_materialization_store(root: Path, captures: list[str]) -> tuple[Store, dict[str, str]]:
    """Inventory each archive once, retaining and decoding only selected members."""
    checkout = root.parent.parent
    if capture_sources.is_current(checkout):
        if any(not capture.startswith("latest:") for capture in captures):
            raise ValueError("frozen-version fixtures require an explicit pinned history source")
        requested = {capture.removeprefix("latest:") for capture in captures}
        if len(requested) != len(captures):
            raise ValueError("request one selected version per implementation")
        current = capture_sources.current(checkout)
        rows = [row for row in current["captures"] if row["surface"] == "unified" and row["implementation"] in requested]
        targets = {}
        for implementation in requested:
            versions = [row["version"] for row in rows if row["implementation"] == implementation]
            if not versions:
                raise ValueError(f"no captures for {implementation}")
            targets[implementation] = max(versions, key=_capture_release_sort_key)
        resources, source_data = [], {}
        for name, data in capture_sources.authored_sources(checkout):
            source_data[name.removeprefix(capture_sources.STORE + "/")] = data
            resources.append({**source_identity(name.removeprefix(capture_sources.STORE + "/")), "path": name})
        for row in rows:
            name = row["source"]
            source_data[name.removeprefix(capture_sources.STORE + "/")] = capture_sources.checked_bytes(checkout, row)
            resources.append({**source_identity(name.removeprefix(capture_sources.STORE + "/")), "path": name})
        store = selected_store(root, resources, targets, source_data=source_data,
                               preselected=True, lineage=current["lineage"])
        if capture_sources.current(checkout) != current:
            raise ValueError("current capture changed during direct fixture read")
        return store, targets
    requested = {}
    for capture in captures:
        if capture.startswith("latest:"):
            implementation, version = capture.removeprefix("latest:"), None
        else:
            match = CAPTURE_DIRECTORY_RE.fullmatch(capture)
            if match is None:
                raise ValueError(f"invalid selected capture: {capture}")
            implementation, version = match["implementation"], match["runtime_version"]
        if implementation in requested:
            raise ValueError("request one selected version per implementation")
        requested[implementation] = version
    if (checkout / capture_sources.DATASETS).is_file():
        store = _routed_history_store(root)
        targets = {}
        for implementation, version in requested.items():
            versions = [capture["runtime_version"] for (family, impl), history in store.histories.items()
                        if impl == implementation for capture in history.captures.values()]
            if not versions:
                raise ValueError(f"no captures for {implementation}")
            targets[implementation] = version or max(versions, key=_capture_release_sort_key)
        return store, targets
    retained = {}
    selected = {}
    latest = {}

    def retain(name, data, archive):
        identity = source_identity(name)
        implementation, version = identity["implementation"], identity["version"]
        if identity["kind"] == "inputs-and-golden":
            retained[name] = data
        elif implementation in requested:
            key = _capture_release_sort_key(version)
            if implementation not in latest or key > _capture_release_sort_key(latest[implementation]):
                latest[implementation] = version
            ceiling = requested[implementation]
            if ceiling is not None and key > _capture_release_sort_key(ceiling):
                return
            family_key = identity["family"], implementation
            previous = selected.get(family_key)
            if previous is None or key > previous[0]:
                if previous is not None:
                    del retained[previous[1]]
                selected[family_key] = key, name
                retained[name] = data

    prefix = "conformance/fixtures-unified-v2/"
    with _store_read_lock(root):
        inventory = unified_archives.inventory(root, member_callback=retain)
        resources = []
        for row in inventory:
            resource = {**row, **source_identity(row["path"]), "path": prefix + row["path"]}
            if "archive" in row:
                resource["archive"] = {**row["archive"], "path": prefix + row["archive"]["path"]}
            resources.append(resource)
        for implementation in requested:
            if implementation not in latest:
                raise ValueError(f"no captures for {implementation}")
        targets = {implementation: version or latest[implementation] for implementation, version in requested.items()}
        store = selected_store(root, resources, targets, source_data=retained)
    return store, targets


def selected_store(root: Path, resources: list[dict], targets: dict[str, str], *, source_data: dict[str, bytes] | None = None,
                   preselected: bool = False, lineage: list[dict] | None = None) -> Store:
    """Read complete selected checkpoints without loading historical dependencies."""
    prefix = "conformance/fixtures-unified-v2/"
    source_index = {resource["path"]: resource for resource in resources}
    source_index.update({row["path"]: row for row in lineage or []})
    lineage_path = root / "lineage.json"
    if lineage_path.is_file():
        source_index.update({row["path"]: row for row in json.loads(lineage_path.read_bytes())})
    chosen = [resource for resource in resources if resource["kind"] == "inputs-and-golden"]
    if preselected:
        chosen.extend(resource for resource in resources if resource["kind"] == "history")
    for implementation, version in (() if preselected else targets.items()):
        families = {}
        for resource in resources:
            if resource.get("implementation") != implementation:
                continue
            if _capture_release_sort_key(resource["version"]) > _capture_release_sort_key(version):
                continue
            family = resource["family"]
            if family not in families or _capture_release_sort_key(resource["version"]) > _capture_release_sort_key(families[family]["version"]):
                families[family] = resource
        if not families:
            raise ValueError(f"no selected captures: {implementation}-{version}")
        chosen.extend(families.values())
    local = []
    for resource in chosen:
        row = copy.deepcopy(resource)
        if not row["path"].startswith(prefix):
            raise ValueError("selected source is outside Unified store")
        row["path"] = row["path"][len(prefix):]
        if "archive" in row:
            if not row["archive"]["path"].startswith(prefix):
                raise ValueError("selected archive is outside Unified store")
            row["archive"]["path"] = row["archive"]["path"][len(prefix):]
        local.append(row)
    documents = {}
    with _store_read_lock(root):
        sources = unified_archives.read_sources(root, local) if source_data is None else ((resource, source_data[resource["path"]]) for resource in local)
        for resource, data in sources:
            documents[resource["path"]] = yaml.load(data, Loader=StrictLoader)
    families = {}
    for resource in local:
        if resource["kind"] == "inputs-and-golden":
            families[resource["family"]] = _validate_family(root / resource["path"], documents[resource["path"]])
    histories = {}
    for resource in local:
        if resource["kind"] != "history":
            continue
        name, implementation, version = resource["family"], resource["implementation"], resource["version"]
        path = root / resource["path"]
        document = documents[resource["path"]]
        _validate_complete_snapshot(path, document, families[name], implementation, version)
        for record in document["records"].values():
            origin = source_index.get(record["member"])
            if (origin is None or origin.get("family") != name or origin.get("implementation") != implementation
                    or _capture_release_sort_key(origin["version"]) > _capture_release_sort_key(version)):
                raise ValueError("selected snapshot producer is unavailable or inconsistent")
        capture_id = path.stem
        capture = {"runtime_version": version, "provenance": document["provenance"], "document": document["document"]}
        records = {case: record["value"] for case, record in document["records"].items()}
        members = {case: record["member"] for case, record in document["records"].items()}
        histories[name, implementation] = History(families[name], implementation, {capture_id: capture}, path.parent,
                                                  {capture_id: path}, {capture_id: records}, {capture_id: members},
                                                  {capture_id: {key for key, row in document["records"].items() if row.get("suppress_inherited_from")}})
    return Store(root, families, histories)


def materialize_store(
    root: Path,
    destination: Path,
    *,
    include_current_inputs: bool = True,
    derived_release_versions: dict[str, str] | None = None,
    store: Store | None = None,
    current_capture_id: str | None = None,
    selected_capture_ids: set[str] | None = None,
) -> None:
    """Materialize inputs and captures, filtering retired cases only from the current view.

    Historical captures retain retired cases so their old columns remain reproducible. The
    selected current capture is different: its record set must match today's active inputs or
    the live-capture guard treats historical aliases as extra current requests. A current
    release may be a derived view when the sparse history has no checkpoint for that version.
    """
    if store is None:
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
            if directory == current_capture_id and (
                case["lifecycle"] != "active"
                or history.implementation == "dynamo_v2" and case["scenario"] in capture_stimulus.V1_SPLIT_SCENARIOS
            ):
                continue
            # Captures retain their original labels as evidence. Current views
            # follow the stable owner when taxonomy numbers are reassigned.
            case_key = case["display_id"] if case["lifecycle"] == "active" else change["case_key"]
            record, document_metadata = _materialized_record(case, change)
            document_metadata = dict(document_metadata)
            # Re-ingestion must distinguish this generated view from original
            # capture metadata, so the importer can discard the marker.
            document_metadata["capture_provenance"] = {"format": MATERIALIZED_FORMAT}
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
        for capture_id, state in history.iter_resolved_snapshots(selected_capture_ids):
            add_capture_state(capture_id, history, state)

    # The YAML store remains sparse: a release with no changed family output has
    # no checkpoint file. Consumers still need a complete directory for the
    # released version, so extraction may request a derived release view.
    for implementation, runtime_version in (derived_release_versions or {}).items():
        if not isinstance(implementation, str) or not isinstance(runtime_version, str):
            raise ValueError("derived release versions must map strings to strings")
        if not UNIFIED_VERSION_RE.fullmatch(runtime_version):
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
            if selected_capture_ids is not None and target_id not in selected_capture_ids:
                continue
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
                state = history.resolve(source_id, view_capture_id=target_id)
                if target_id == current_capture_id:
                    state = {case_id: change for case_id, change in state.items()
                             if history.family.cases[case_id]["lifecycle"] == "active"}
                add_capture_state(target_id, history, state)

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
    explicit_stimulus = record.get("capture_stimulus")
    if explicit_stimulus is not None:
        if not isinstance(explicit_stimulus, dict) or set(explicit_stimulus) not in (
            {"unavailable"},
            {"error"},
            {"partial"},
        ):
            raise ValueError(f"invalid explicit capture stimulus: {relative}")
        state_name = next(iter(explicit_stimulus))
        state = explicit_stimulus[state_name]
        if not isinstance(state, dict) or (
            state_name != "partial" and not isinstance(state.get("code"), str)
        ):
            raise ValueError(f"explicit capture stimulus needs a typed {state_name} code: {relative}")
        original = capture_stimulus.original_capture_input(record, raw, relative, bindings)
        if current is not None and typed_equal(original, capture_stimulus.capture_input(current)):
            stimulus = {
                "ref": "current",
                "semantic_sha256": _request_digest(original),
            }
        else:
            stimulus = copy.deepcopy(explicit_stimulus)
    else:
        original = capture_stimulus.original_capture_input(record, raw, relative, bindings)
        if original is None:
            stimulus = {"unavailable": {"code": "original_request_not_retained"}}
        elif current is not None and typed_equal(original, capture_stimulus.capture_input(current)):
            stimulus = {
                "ref": "current",
                "semantic_sha256": _request_digest(original),
            }
        elif isinstance(original, dict) and original.keys() == REQUEST_KEYS and original.get("tools") is not None:
            stimulus = {"inline": original, "semantic_sha256": _request_digest(original)}
        else:
            stimulus = {"partial": original}
    explicit_observation = record.get("capture_observation")
    if explicit_observation is not None:
        if not isinstance(explicit_observation, dict) or set(explicit_observation) not in ({"error"}, {"unavailable"}):
            raise ValueError(f"invalid explicit capture observation: {relative}")
        observation_state = next(iter(explicit_observation))
        observation_value = explicit_observation[observation_state]
        if not isinstance(observation_value, dict) or not isinstance(observation_value.get("code"), str):
            raise ValueError(f"explicit capture observation needs a typed {observation_state} code: {relative}")

    value = dict(record)
    value.pop("capture_input", None)
    value.pop("capture_stimulus", None)
    value.pop("capture_observation", None)
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
        typed = explicit_observation.get("error") if isinstance(explicit_observation, dict) else None
        if typed is None and isinstance(explicit_stimulus, dict):
            typed = explicit_stimulus.get("error")
        observation = {"error": copy.deepcopy(typed) if isinstance(typed, dict) else {"code": "legacy_unclassified", "detail": detail}}
    elif "unavailable" in value:
        detail = value.pop("unavailable")
        typed = explicit_observation.get("unavailable") if isinstance(explicit_observation, dict) else None
        if typed is None and isinstance(explicit_stimulus, dict):
            typed = explicit_stimulus.get("unavailable")
        observation = {"unavailable": copy.deepcopy(typed) if isinstance(typed, dict) else {"code": "legacy_unclassified", "detail": detail}}
    else:
        observation = {"value": value}
    return stimulus, observation, metadata


def _canonical_json(value: Any) -> str:
    return canonical_typed(value)


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


def _is_correctable_capture_placeholder(
    before: dict, after: dict, case_id: str, request: dict | None
) -> bool:
    """Allow a corrected peer result only when its recorded request is unchanged."""
    old_observation = before["observation"]
    new_observation = after["observation"]
    unavailable = old_observation.get("unavailable")

    def has_verified_request_binding(change: dict) -> bool:
        stimulus = change["stimulus"]
        digest = stimulus.get("semantic_sha256")
        if not isinstance(digest, str) or not digest:
            return False
        if stimulus.get("ref") == "current":
            return request is not None and digest == _request_digest(request)
        if "inline" in stimulus:
            return digest == _request_digest(stimulus["inline"])
        return False

    return (
        before["case_key"] == after["case_key"]
        and _canonical_json(_semantic_stimulus(before["stimulus"]))
        == _canonical_json(_semantic_stimulus(after["stimulus"]))
        and has_verified_request_binding(before)
        and has_verified_request_binding(after)
        and isinstance(unavailable, dict)
        and unavailable.get("code") in CORRECTABLE_CAPTURE_UNAVAILABLE_CODES
        and set(new_observation) == {"value"}
    )


def _stored_change(value: dict) -> dict:
    return {key: copy.deepcopy(value[key]) for key in CHANGE_KEY_ORDER}


def _capture_changes(
    before: dict, after: dict, *, include_absent: bool = True,
    preserve_binding_details: bool = True,
) -> dict:
    changes = {}
    for case_id in sorted(set(before) | set(after)):
        old = before.get(case_id)
        new = after.get(case_id)
        if new is None:
            if include_absent and old is not None:
                changes[case_id] = {"absent": True}
            continue
        if old is None:
            changes[case_id] = _stored_change(new)
            continue

        delta = {}
        stimulus_changed = not typed_equal(old["stimulus"], new["stimulus"])
        if not preserve_binding_details:
            stimulus_changed = not typed_equal(_semantic_stimulus(old["stimulus"]), _semantic_stimulus(new["stimulus"]))
        if stimulus_changed:
            delta["stimulus"] = copy.deepcopy(new["stimulus"])
        if not typed_equal(old["observation"], new["observation"]):
            delta["observation"] = copy.deepcopy(new["observation"])
        if old["case_key"] != new["case_key"] and (preserve_binding_details or delta):
            delta = {"case_key": copy.deepcopy(new["case_key"]), **delta}
        if delta:
            changes[case_id] = delta
    return changes


def _install_complete_snapshot(history: History, capture_id: str, records: dict, members: dict[str, str]) -> None:
    """Install values, lineage, and derived-view provenance as one state change."""
    current = history.resolved_snapshots.get(capture_id, {})
    prior_suppressed = history.snapshot_suppressed_inheritance.get(capture_id, set())
    member_path = f"conformance/fixtures-unified-v2/families/{history.family.name}/{capture_id}.yaml"
    suppressed = set()
    for case_id, record in records.items():
        if "inherited_from" in record["document"]:
            continue
        old = current.get(case_id)
        same_origin = old is not None and typed_equal(_capture_semantic(old, case_id), _capture_semantic(record, case_id))
        same_provenance = old is not None and typed_equal(
            {key: old["document"][key] for key in ("captured_with", "capture_origin") if key in old["document"]},
            {key: record["document"][key] for key in ("captured_with", "capture_origin") if key in record["document"]},
        )
        if members[case_id] != member_path or case_id in prior_suppressed and same_origin and same_provenance:
            suppressed.add(case_id)
    history.resolved_snapshots[capture_id] = copy.deepcopy(records)
    history.snapshot_members[capture_id] = dict(members)
    history.snapshot_suppressed_inheritance[capture_id] = suppressed


def _set_capture_snapshot(history: History, capture_id: str, target: dict, previous: dict, *, cursor: _HistoryCursor | None = None) -> None:
    if history.resolved_snapshots:
        current = history.resolved_snapshots.get(capture_id, {})
        members = history.snapshot_members.get(capture_id, {})
        path = (Path("conformance/fixtures-unified-v2") / "families" / history.family.name
                / f"{capture_id}.yaml").as_posix()
        previous_members = {}
        ordered = history.ordered_capture_ids()
        if capture_id in ordered:
            index = ordered.index(capture_id)
            for previous_id in reversed(ordered[:index]):
                for case_id, member in history.snapshot_members.get(previous_id, {}).items():
                    previous_members.setdefault(case_id, member)
        updated_members = {}
        for case_id, record in target.items():
            current_record = current.get(case_id)
            if current_record is not None and _canonical_json(_capture_semantic(current_record, case_id)) == _canonical_json(_capture_semantic(record, case_id)):
                updated_members[case_id] = members[case_id]
            elif case_id in previous and _canonical_json(_capture_semantic(previous[case_id], case_id)) == _canonical_json(_capture_semantic(record, case_id)):
                updated_members[case_id] = previous_members.get(case_id, path)
            else:
                updated_members[case_id] = path
        _install_complete_snapshot(history, capture_id, target, updated_members)
        return
    changes, metadata_changes, _document_overrides = _snapshot_delta(target, previous)
    capture = history.captures[capture_id]
    capture.update(
        changes=changes, metadata_changes=metadata_changes,
        document_overrides={},
    )
    # Resolve the sparse fields with the capture's real defaults before adding
    # provenance overrides, so already minimal checkpoints remain unchanged.
    if cursor is None:
        cursor = _HistoryCursor(history)
        for previous_id in history.ordered_capture_ids():
            if previous_id == capture_id:
                break
            cursor.advance(previous_id)
    natural = cursor.preview(capture_id)
    _changes, _metadata_changes, document_overrides = _snapshot_delta(target, natural)
    capture["document_overrides"] = document_overrides
    cursor.advance(capture_id)


def _compact_history(history: History) -> None:
    """Remove repeated case fields while preserving each resolved checkpoint."""
    snapshots = list(history.iter_resolved_snapshots())
    previous = {}
    cursor = _HistoryCursor(history)
    for capture_id, snapshot in snapshots:
        _set_capture_snapshot(history, capture_id, snapshot, previous, cursor=cursor)
        previous = snapshot


def _replace_capture_snapshot(history: History, capture_id: str, target: dict) -> None:
    if history.resolved_snapshots:
        ordered = history.ordered_capture_ids()
        index = ordered.index(capture_id)
        previous = history.resolve(ordered[index - 1]) if index else {}
        _set_capture_snapshot(history, capture_id, target, previous)
        return
    ordered = history.ordered_capture_ids()
    index = ordered.index(capture_id)
    previous = history.resolve(ordered[index - 1]) if index else {}
    later = {key: history.resolve(key) for key in ordered[index + 1 :]}
    _set_capture_snapshot(history, capture_id, target, previous)
    previous = target
    for key, snapshot in later.items():
        # Importers and placeholder corrections must not execute a release by inheritance.
        _set_capture_snapshot(history, key, snapshot, previous)
        previous = snapshot


def _insert_historical_capture(
    history: History,
    capture_id: str,
    runtime_version: str,
    records: dict,
) -> bool:
    """Insert a newly executed older peer release without changing newer results."""
    ordered = history.ordered_capture_ids()
    if not ordered or _capture_release_sort_key(runtime_version) >= _capture_release_sort_key(
        history.captures[ordered[0]]["runtime_version"]
    ):
        return False
    if any(
        history.captures[current]["metadata_changes"]
        or history.captures[current]["document_overrides"]
        for current in ordered
    ):
        return False
    history.captures[capture_id] = {
        "runtime_version": runtime_version,
        "provenance": {
            "status": "legacy",
            "captured_with": {history.implementation: runtime_version},
        },
        "document": {"mode": "unified"},
        "changes": {},
        "metadata_changes": {},
        "document_overrides": {},
    }
    history.capture_paths[capture_id] = history.family.path.parent / f"{capture_id}.yaml"
    if history.resolved_snapshots:
        member = (Path("conformance/fixtures-unified-v2") / "families" / history.family.name
                  / f"{capture_id}.yaml").as_posix()
        _install_complete_snapshot(history, capture_id, records, {case_id: member for case_id in records})
        return True
    _replace_capture_snapshot(history, capture_id, records)
    return True


def _add_bound_legacy_cases(
    history: History,
    capture_id: str,
    records: dict,
    additions: list[str],
) -> bool:
    """Add only new, request-bound rows to a legacy checkpoint without rewriting old rows."""
    capture = history.captures[capture_id]
    if capture["provenance"]["status"] != "legacy":
        return False
    if any(history.family.cases[case_id]["lifecycle"] != "active" for case_id in additions):
        return False
    for case_id in additions:
        record = records[case_id]
        stimulus = record["stimulus"]
        origin = record["document"].get("capture_origin")
        if (
            len(stimulus.keys() & {"ref", "inline"}) != 1
            or not isinstance(stimulus.get("semantic_sha256"), str)
            or not stimulus["semantic_sha256"]
            or (origin is not None and origin.get("crate_version") != capture["runtime_version"])
        ):
            return False

    target = copy.deepcopy(history.resolve(capture_id))

    for case_id in additions:
        target[case_id] = copy.deepcopy(records[case_id])

    _replace_capture_snapshot(history, capture_id, target)
    return True


def _snapshot_delta(
    target: dict,
    previous: dict,
    *,
    include_absent: bool = True,
    preserve_binding_details: bool = True,
) -> tuple[dict, dict, dict]:
    changes = _capture_changes(
        previous, target, include_absent=include_absent,
        preserve_binding_details=preserve_binding_details,
    )
    metadata_changes = {}
    document_overrides = {}
    for case_id in sorted(set(target) | set(previous)):
        before = previous.get(case_id)
        after = target.get(case_id)
        if after is None:
            continue
        after_document = after.get("document", {})
        before_document = {} if before is None else before.get("document", {})
        after_metadata = after_document.get("record_metadata", {})
        before_metadata = before_document.get("record_metadata", {})
        if _canonical_json(after_metadata) != _canonical_json(before_metadata):
            metadata_changes[case_id] = copy.deepcopy(after_metadata)
        after_path = after_document.get("parser_path")
        before_path = before_document.get("parser_path")
        if after_path != before_path:
            # A newly inserted older capture must not add its parser path to
            # later snapshots that originally omitted that metadata.
            document_overrides.setdefault(case_id, {})["parser_path"] = after_path
        provenance_keys = ("captured_with", "capture_origin", "inherited_from")
        after_provenance = {
            key: copy.deepcopy(after_document.get(key)) for key in provenance_keys
        }
        before_provenance = {
            key: copy.deepcopy(before_document.get(key)) if before is not None else None
            for key in provenance_keys
        }
        if not typed_equal(after_provenance, before_provenance):
            document_overrides.setdefault(case_id, {})[
                "capture_provenance"
            ] = after_provenance
    return changes, metadata_changes, document_overrides


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
) -> dict[Path, dict] | None:
    loose_root = Path(loose_root)
    documents = {}
    if not loose_root.is_dir():
        return None
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
    missing_excluded_dirs = sorted(excluded_capture_dirs - capture_dirs)
    if missing_excluded_dirs:
        raise ValueError(
            "cannot exclude Unified capture directories absent from the loose corpus: "
            f"{', '.join(missing_excluded_dirs)}"
        )
    required_exclusions = sorted(excluded_capture_dirs & required_capture_dirs)
    if complete_snapshot and required_exclusions:
        raise ValueError(
            "complete snapshot cannot exclude required Unified captures: "
            f"{', '.join(required_exclusions)}"
        )
    for capture_id in sorted(excluded_capture_dirs):
        match = CAPTURE_DIRECTORY_RE.fullmatch(capture_id)
        assert match is not None
        family_dirs = sorted(
            path.name
            for path in (loose_root / capture_id).iterdir()
            if path.is_dir()
        )
        if not family_dirs:
            raise ValueError(
                f"cannot exclude Unified capture {capture_id} without family directories"
            )
        for family_name in family_dirs:
            history = store.histories.get((family_name, match["implementation"]))
            if history is None:
                raise ValueError(
                    f"cannot exclude Unified capture {capture_id}/{family_name}: "
                    "absent from that family's YAML history"
                )
            if capture_id not in history.captures:
                target_release = _capture_release_sort_key(match["runtime_version"])
                prior_captures = [
                    known_id
                    for known_id, capture in history.captures.items()
                    if _capture_release_sort_key(capture["runtime_version"])
                    <= target_release
                ]
                if not prior_captures:
                    raise ValueError(
                        f"cannot exclude Unified capture {capture_id}/{family_name}: "
                        "no YAML checkpoint exists at or before this version"
                    )
    if complete_snapshot:
        missing_required_capture_dirs = sorted(required_capture_dirs - capture_dirs)
        if missing_required_capture_dirs:
            missing = ", ".join(missing_required_capture_dirs)
            raise ValueError(f"complete snapshot is missing required captures: {missing}")
    visited_capture = False
    for capture_dir in sorted(
        path
        for path in loose_root.iterdir()
        if (
            path.is_dir()
            and CAPTURE_DIRECTORY_RE.fullmatch(path.name) is not None
        )
    ):
        visited_capture = True
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
            if family_name not in store.families:
                raise ValueError(f"no canonical family owns {capture_dir.name}/{family_name}")
            history = store.history_for_capture(family_name, implementation)
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
            excluded = capture_dir.name in excluded_capture_dirs
            for path in sorted((capture_dir / family_name).glob("*.yaml")):
                document = _load_loose_document(path, family_name)
                if not excluded:
                    raw = path.read_bytes()
                    metadata = {
                        name: value for name, value in document.items() if name != "cases"
                    }
                    # The history marker belongs only to the extracted compatibility
                    # view. It must not become canonical capture metadata when that
                    # view is ingested again.
                    if metadata.get("capture_provenance") == {"format": MATERIALIZED_FORMAT}:
                        metadata.pop("capture_provenance")
                    capture_origin = metadata.get("capture_origin")
                    source_provenance = metadata.get("capture_provenance")
                    if capture_origin is None and isinstance(source_provenance, dict):
                        origin_fields = {
                            key: source_provenance[key]
                            for key in (
                                "crate_version",
                                "source_sha256",
                                "git_commit",
                                "adapter_sha256",
                                "job_sha256",
                            )
                            if key in source_provenance
                        }
                        if {"crate_version", "source_sha256"} <= origin_fields.keys():
                            capture_origin = origin_fields
                    if capture_origin is not None:
                        capture_origin = _validate_source_origin(
                            capture_origin,
                            f"{path}.capture_origin",
                            runtime_version,
                        )
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
                    if excluded:
                        if case_id in records:
                            previous_key = records[case_id]["case_key"]
                            raise ValueError(
                                f"duplicate Unified records resolve to {family_name}/{case_id}: "
                                f"{previous_key}, {case_key}"
                            )
                        records[case_id] = {"case_key": case_key}
                        continue
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
                            **({"capture_origin": capture_origin} if capture_origin else {}),
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
            if capture_dir.name in captures or excluded:
                if capture_dir.name in captures:
                    resolved = history.resolve(capture_dir.name)
                else:
                    target_release = _capture_release_sort_key(runtime_version)
                    prior_capture = max(
                        (
                            known_id
                            for known_id, capture in captures.items()
                            if _capture_release_sort_key(capture["runtime_version"])
                            <= target_release
                        ),
                        key=lambda known_id: _capture_release_sort_key(
                            captures[known_id]["runtime_version"]
                        ),
                    )
                    resolved = history.resolve(prior_capture)
                if excluded:
                    additions = sorted(set(records) - set(resolved))
                    if additions:
                        raise ValueError(
                            f"cannot exclude Unified capture {capture_dir.name}/{family_name}; "
                            f"unrecorded cases: {', '.join(additions)}"
                        )
                    # Exclusion keeps checked-in YAML authoritative for already-known rows.
                    _compact_history(history)
                    continue
                if complete_snapshot and not required_capture:
                    missing_existing_cases = [
                        case_id
                        for case_id in sorted(resolved)
                        if (
                            history.family.cases[case_id]["lifecycle"] == "active"
                            and case_id not in records
                        )
                    ]
                    if missing_existing_cases:
                        missing = ", ".join(
                            f"{family_name}/{case_id}" for case_id in missing_existing_cases
                        )
                        raise ValueError(
                            f"complete capture {capture_dir.name} is missing previously "
                            f"recorded active cases: {missing}"
                        )
                # Non-required snapshots are sparse deltas; absent rows inherit the prior result.
                conflicts = []
                corrections = []
                provenance_enrichments = []
                for case_id in sorted(set(resolved) & set(records)):
                    before = resolved[case_id]
                    after = records[case_id]
                    if _canonical_json(_capture_semantic(before, case_id)) == _canonical_json(
                        _capture_semantic(after, case_id)
                    ):
                        origin = after["document"].get("capture_origin")
                        captured_with = before["document"].get("captured_with", {})
                        if (
                            captured_with.get(implementation) != runtime_version
                            and isinstance(origin, dict)
                            and origin.get("crate_version") == runtime_version
                        ):
                            provenance_enrichments.append(case_id)
                        continue
                    if _is_correctable_capture_placeholder(
                        before,
                        after,
                        case_id,
                        history.family.cases[case_id]["request"],
                    ):
                        corrections.append(case_id)
                    else:
                        conflicts.append(case_id)
                if conflicts:
                    raise ValueError(
                        f"capture is immutable; use a new identity: {capture_dir.name}/{family_name}"
                    )
                if corrections:
                    target = copy.deepcopy(resolved)
                    for case_id in corrections:
                        target[case_id] = copy.deepcopy(records[case_id])
                    _replace_capture_snapshot(history, capture_dir.name, target)
                    resolved = target
                if provenance_enrichments:
                    target = copy.deepcopy(resolved)
                    for case_id in provenance_enrichments:
                        target[case_id] = copy.deepcopy(records[case_id])
                    _replace_capture_snapshot(history, capture_dir.name, target)
                    resolved = target
                additions = sorted(set(records) - set(resolved))
                if not additions:
                    _compact_history(history)
                    continue
                if _add_bound_legacy_cases(history, capture_dir.name, records, additions):
                    continue
                raise ValueError(
                    f"capture {capture_dir.name} is already recorded; only bound, conflict-free "
                    "active-case additions to legacy captures are allowed"
                )
            capture_version = _capture_release_sort_key(runtime_version)
            ordered_ids = history.ordered_capture_ids()
            prior_ids = [
                capture_id
                for capture_id in ordered_ids
                if _capture_release_sort_key(
                    history.captures[capture_id]["runtime_version"]
                ) < capture_version
            ]
            later_ids = [
                capture_id
                for capture_id in ordered_ids
                if _capture_release_sort_key(
                    history.captures[capture_id]["runtime_version"]
                ) > capture_version
            ]
            prior = history.resolve(prior_ids[-1]) if prior_ids else {}
            later_snapshots = {
                capture_id: history.resolve(capture_id) for capture_id in later_ids
            }
            changes, metadata_changes, document_overrides = _snapshot_delta(
                records,
                prior,
                include_absent=complete_snapshot and required_capture,
                # Loose views recompute digests and expose current display aliases;
                # those fields alone do not establish a newly executed checkpoint.
                preserve_binding_details=False,
            )
            for case_id in list(document_overrides):
                if case_id not in changes:
                    document_overrides[case_id].pop("capture_provenance", None)
                    if not document_overrides[case_id]:
                        del document_overrides[case_id]
            origins = {
                _canonical_json(record["document"]["capture_origin"]): record["document"]["capture_origin"]
                for record in records.values()
                if "capture_origin" in record["document"]
            }
            if origins:
                if len(origins) != 1:
                    raise ValueError(f"new capture has mixed source origins: {capture_dir.name}")
                provenance = {"status": "captured", "origin": copy.deepcopy(next(iter(origins.values())))}
            else:
                provenance = {
                    "status": "legacy",
                    "captured_with": {implementation: runtime_version},
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
            if not ordered_ids:
                _set_capture_snapshot(history, capture_dir.name, history.resolve(capture_dir.name), {})
            capture_path = history.family.path.parent / f"{capture_dir.name}.yaml"
            history.capture_paths[capture_dir.name] = capture_path
            documents[capture_path] = capture
            if history.resolved_snapshots:
                target = {} if complete_snapshot and required_capture else copy.deepcopy(prior)
                for case_id, change in changes.items():
                    if change == {"absent": True}:
                        target.pop(case_id, None)
                target.update(copy.deepcopy(records))
                member_path = (Path("conformance/fixtures-unified-v2") / "families" / family_name
                               / f"{capture_dir.name}.yaml").as_posix()
                member_map = {}
                prior_members = history.snapshot_members[prior_ids[-1]] if prior_ids else {}
                for case_id, record in target.items():
                    before = prior.get(case_id)
                    member_map[case_id] = (
                        prior_members.get(case_id, member_path)
                        if before is not None and _canonical_json(_capture_semantic(before, case_id))
                        == _canonical_json(_capture_semantic(record, case_id))
                        else member_path
                    )
                _install_complete_snapshot(history, capture_dir.name, target, member_map)
            previous_snapshot = history.resolve(capture_dir.name)
            for later_id in later_ids:
                later_snapshot = later_snapshots[later_id]
                _set_capture_snapshot(history, later_id, later_snapshot, previous_snapshot)
                documents[history.capture_paths[later_id]] = history.captures[later_id]
                previous_snapshot = later_snapshot
    return documents if visited_capture else None


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
        skip_no_sources=True,
    )


def _preserve_historical_stimulus(
    store: Store,
    family_name: str,
    case_id: str,
    request: dict,
    documents: dict[Path, dict],
) -> None:
    raise ValueError(
        f"Authored input {family_name}/{case_id} differs from the recorded request. "
        "Run latest parser tests with the updated input; ask a repository administrator "
        "to update the affected latest capture when a new measurement is needed."
    )


def _shared_corpus_roots(loose_root: Path, base: str) -> list[Path]:
    """The canonical authored corpus has one current root per document type."""
    return [loose_root / base]


def _sync_current_corpus(
    store: Store,
    loose_root: Path,
    *,
    complete_snapshot: bool,
) -> dict[Path, dict] | None:
    loose_root = Path(loose_root)
    documents = {}
    inputs_root = loose_root / "inputs"
    golden_root = loose_root / "golden"
    if not inputs_root.is_dir() or not golden_root.is_dir():
        if complete_snapshot:
            raise ValueError("complete current snapshot needs inputs and golden roots")
        return None
    input_roots = _shared_corpus_roots(loose_root, "inputs")
    golden_roots = _shared_corpus_roots(loose_root, "golden")

    registered_families = set(
        (yaml.safe_load((Path(__file__).resolve().parent / "parser_families.yaml").read_text()) or {})
        .get("unified", {})
    )
    known_families = set(store.families)
    for root in (*input_roots, *golden_roots):
        unknown = sorted(
            path.name for path in root.iterdir() if path.is_dir() and path.name not in known_families
        )
        unregistered = sorted(set(unknown) - registered_families)
        if unregistered:
            raise ValueError(f"current Unified corpus has unregistered families under {root}: {unregistered}")
        for family_name in unknown:
            family_path = store.root / "families" / family_name / FAMILY_DOCUMENT_NAME
            store.families[family_name] = Family(
                family_name,
                {
                    "input_document": {"family": family_name, "mode": "unified", "model_label": family_name},
                    "golden_document": {"family": family_name, "mode": "unified", "captured_with": {"golden": "v1"}},
                    "cases": {},
                },
                family_path,
            )

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
        by_external_id = {
            external_id: case_id
            for case_id, case in cases.items()
            for external_id in [case["display_id"], *case["historical_ids"]]
            if isinstance(external_id, str)
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
                case_id = by_scenario.get(scenario) or by_external_id.get(case_key)
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
        skip_no_sources=True,
    )


def update_store_from_loose(
    store_root: Path,
    loose_root: Path,
    *,
    complete_snapshot: bool,
    excluded_capture_dirs: set[str] | frozenset[str] = frozenset(),
    required_capture_dirs: set[str] | frozenset[str] = frozenset(),
) -> list[Path]:
    def documents(store: Store) -> dict[Path, dict] | None:
        updated = _sync_current_corpus(
            store,
            loose_root,
            complete_snapshot=complete_snapshot,
        )
        captures = _update_from_loose(
            store,
            loose_root,
            complete_snapshot=complete_snapshot,
            excluded_capture_dirs=excluded_capture_dirs,
            required_capture_dirs=required_capture_dirs,
        )
        if updated is None and captures is None:
            return None
        return {**(updated or {}), **(captures or {})}

    return _mutate_store(store_root, documents, skip_no_sources=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--materialize", action="store_true")
    action.add_argument("--validate", action="store_true")
    action.add_argument("--update-from-loose", action="store_true")
    action.add_argument("--sync-current-corpus", action="store_true")
    action.add_argument("--migrate-descriptive-ids", action="store_true")
    action.add_argument("--archive-versions", action="store_true", help="group complete snapshots into deterministic engine/version archives")
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--history-only", action="store_true")
    parser.add_argument("--capture", action="append", default=[], help="materialize only this implementation-version, or latest:implementation")
    parser.add_argument("--history-source", type=Path, help="verified external evidence for frozen-version qualification")
    parser.add_argument(
        "--complete-snapshot",
        action="store_true",
        help="retire active cases omitted from a complete loose corpus",
    )
    args = parser.parse_args(argv)
    if args.history_source is not None:
        checkout = args.store.resolve().parent.parent
        with capture_sources.history_view(checkout, args.history_source) as view:
            arguments = list(argv if argv is not None else sys.argv[1:])
            index = arguments.index("--history-source")
            del arguments[index:index + 2]
            index = arguments.index("--store")
            arguments[index + 1] = str(view / capture_sources.STORE)
            return main(arguments)
    if args.archive_versions:
        print(json.dumps(archive_versions(args.store)))
    elif args.migrate_descriptive_ids:
        print(json.dumps([str(path) for path in migrate_descriptive_ids(args.store)], indent=2))
    elif args.materialize:
        if args.output is None:
            parser.error("--materialize requires --output")
        if args.capture:
            store, targets = selected_materialization_store(args.store.resolve(), args.capture)
            materialize_store(args.store, args.output, include_current_inputs=not args.history_only, store=store,
                              derived_release_versions=targets,
                              current_capture_id=f"dynamo_v2-{targets['dynamo_v2']}" if "dynamo_v2" in targets else None,
                              selected_capture_ids={f"{implementation}-{version}" for implementation, version in targets.items()})
        else:
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
