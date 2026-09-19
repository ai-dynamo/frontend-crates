# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load, validate, migrate, and materialize the Unified YAML history store."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
import tarfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

import capture_stimulus
import fixture_disposition


SCHEMA_VERSION = 1
FAMILY_KEYS = {
    "schema_version",
    "family",
    "input_document",
    "golden_document",
    "cases",
}
HISTORY_KEYS = {"schema_version", "family", "implementation", "captures"}
CASE_KEYS = {
    "lifecycle",
    "scenario",
    "display_id",
    "historical_ids",
    "request",
    "golden",
}
CAPTURE_REQUIRED_KEYS = {
    "parent",
    "runtime_version",
    "provenance",
    "completeness",
    "import_lineage",
    "changes",
    "metadata_changes",
}
CAPTURE_KEYS = CAPTURE_REQUIRED_KEYS | {"document_metadata_changes"}
CHANGE_KEYS = {"case_key", "stimulus", "observation", "document"}
OBSERVATION_STATES = {"value", "error", "unavailable"}
STIMULUS_STATES = {"ref", "inline", "partial", "unavailable"}
REQUEST_KEYS = {"input", "init", "finish_reason", "tools", "chunks"}
DOCUMENT_METADATA_EXCLUDED = {
    "family",
    "record_metadata",
}
DOCUMENT_PROVENANCE_KEYS = {
    "capture_provenance",
    "captured_with",
}


class StrictLoader(yaml.SafeLoader):
    pass


def _construct_mapping(loader: StrictLoader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"duplicate key: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


StrictLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


def load_yaml(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    for token in yaml.scan(text):
        if isinstance(token, (yaml.tokens.AnchorToken, yaml.tokens.AliasToken)):
            raise ValueError(f"YAML anchors and aliases are not allowed: {path}")
        if isinstance(token, yaml.tokens.TagToken):
            raise ValueError(f"YAML tags are not allowed: {path}")
    try:
        value = yaml.load(text, Loader=StrictLoader)
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
    path: Path

    def resolve(self, capture_id: str) -> dict:
        resolved: dict[str, dict] = {}
        visiting: set[str] = set()

        def visit(current: str) -> dict:
            if current in resolved:
                return resolved[current]
            if current in visiting:
                raise ValueError(f"capture parent cycle in {self.path}: {current}")
            if current not in self.captures:
                raise ValueError(f"missing parent in {self.path}: {current}")
            visiting.add(current)
            capture = self.captures[current]
            parent = capture["parent"]
            state = dict(visit(parent)) if parent is not None else {}
            if capture["completeness"] == "snapshot":
                state = {}
            for case_id, change in capture["changes"].items():
                if change == {"absent": True}:
                    state.pop(case_id, None)
                else:
                    state[case_id] = change
            for case_id, metadata in capture["metadata_changes"].items():
                if case_id not in state:
                    raise ValueError(f"metadata change has no observation in {self.path}: {case_id}")
                state[case_id] = copy.deepcopy(state[case_id])
                document = state[case_id]["document"]
                if metadata:
                    document["record_metadata"] = metadata
                else:
                    document.pop("record_metadata", None)
            for case_id, metadata in capture.get("document_metadata_changes", {}).items():
                if case_id not in state:
                    raise ValueError(
                        f"document metadata change has no observation in {self.path}: {case_id}"
                    )
                state[case_id] = copy.deepcopy(state[case_id])
                document = state[case_id]["document"]
                record_metadata = document.get("record_metadata")
                state[case_id]["document"] = dict(metadata)
                if record_metadata:
                    state[case_id]["document"]["record_metadata"] = record_metadata
            visiting.remove(current)
            resolved[current] = state
            return state

        return dict(visit(capture_id))


@dataclass
class Store:
    root: Path
    families: dict[str, Family]
    histories: dict[tuple[str, str], History]


def _validate_family(path: Path, document: dict) -> Family:
    _require_keys(
        document,
        {"schema_version", "family", "cases"},
        FAMILY_KEYS,
        str(path),
    )
    if document["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unknown schema version in {path}: {document['schema_version']}")
    family = document["family"]
    if not isinstance(family, str) or path.stem != family:
        raise ValueError(f"family differs from filename: {path}")
    cases = _mapping(document["cases"], f"{path}.cases")
    display_ids: set[str] = set()
    aliases: set[str] = set()
    for case_id, case in cases.items():
        if not isinstance(case_id, str):
            raise ValueError(f"case ID must be a string: {path}")
        case = _mapping(case, f"{path}:{case_id}")
        _require_keys(case, CASE_KEYS, CASE_KEYS, f"{path}:{case_id}")
        if case["lifecycle"] not in {"active", "retired"}:
            raise ValueError(f"invalid lifecycle in {path}:{case_id}")
        display_id = case["display_id"]
        if display_id is not None:
            _safe_component(display_id, f"{path}:{case_id}.display_id")
            if display_id in display_ids or display_id in aliases:
                raise ValueError(f"duplicate display ID in {path}: {display_id}")
            display_ids.add(display_id)
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
        if case["lifecycle"] == "active":
            if display_id is None or not isinstance(case["scenario"], str):
                raise ValueError(f"active case needs display_id and scenario: {path}:{case_id}")
            _validate_request(case["request"], f"{path}:{case_id}.request")
            _mapping(case["golden"], f"{path}:{case_id}.golden")
        elif case["request"] is not None:
            _validate_request(case["request"], f"{path}:{case_id}.request")
    return Family(family, document, path)


def _validate_history(path: Path, document: dict, families: dict[str, Family]) -> History:
    _require_keys(document, HISTORY_KEYS, HISTORY_KEYS, str(path))
    if document["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unknown schema version in {path}: {document['schema_version']}")
    family_name = document["family"]
    implementation = document["implementation"]
    if family_name not in families or path.parent.name != family_name:
        raise ValueError(f"history family is not declared or differs from its path: {path}")
    if path.stem != implementation:
        raise ValueError(f"history implementation differs from filename: {path}")
    captures = _mapping(document["captures"], f"{path}.captures")
    if not captures:
        raise ValueError(f"history has no captures: {path}")
    for capture_id, capture in captures.items():
        _safe_component(capture_id, f"{path}.captures")
        capture = _mapping(capture, f"{path}:{capture_id}")
        _require_keys(capture, CAPTURE_REQUIRED_KEYS, CAPTURE_KEYS, f"{path}:{capture_id}")
        if capture["parent"] is not None and not isinstance(capture["parent"], str):
            raise ValueError(f"capture parent must be a string: {path}:{capture_id}")
        if capture["completeness"] not in {"snapshot", "delta"}:
            raise ValueError(f"invalid completeness: {path}:{capture_id}")
        if capture["parent"] is None and capture["completeness"] != "snapshot":
            raise ValueError(f"root capture must be a snapshot: {path}:{capture_id}")
        if capture["parent"] is not None and capture["parent"] not in captures:
            raise ValueError(f"missing parent in {path}: {capture['parent']}")
        _mapping(capture["provenance"], f"{path}:{capture_id}.provenance")
        if not isinstance(capture["import_lineage"], list):
            raise ValueError(f"import_lineage must be a list: {path}:{capture_id}")
        changes = _mapping(capture["changes"], f"{path}:{capture_id}.changes")
        for case_id, change in changes.items():
            if case_id not in families[family_name].cases:
                raise ValueError(f"unknown case in {path}:{capture_id}: {case_id}")
            if change == {"absent": True}:
                continue
            change = _mapping(change, f"{path}:{capture_id}:{case_id}")
            _require_keys(change, CHANGE_KEYS, CHANGE_KEYS, f"{path}:{capture_id}:{case_id}")
            _safe_component(change["case_key"], f"{path}:{capture_id}:{case_id}.case_key")
            _validate_stimulus(
                change["stimulus"],
                families[family_name].cases[case_id],
                f"{path}:{capture_id}:{case_id}.stimulus",
            )
            _validate_observation(
                change["observation"],
                f"{path}:{capture_id}:{case_id}.observation",
            )
            _mapping(change["document"], f"{path}:{capture_id}:{case_id}.document")
        metadata_changes = _mapping(
            capture["metadata_changes"], f"{path}:{capture_id}.metadata_changes"
        )
        for case_id, metadata in metadata_changes.items():
            if case_id not in families[family_name].cases:
                raise ValueError(f"unknown metadata case in {path}:{capture_id}: {case_id}")
            _mapping(metadata, f"{path}:{capture_id}:{case_id}.metadata_changes")
        document_metadata_changes = _mapping(
            capture.get("document_metadata_changes", {}),
            f"{path}:{capture_id}.document_metadata_changes",
        )
        for case_id, metadata in document_metadata_changes.items():
            if case_id not in families[family_name].cases:
                raise ValueError(f"unknown document metadata case in {path}:{capture_id}: {case_id}")
            _mapping(metadata, f"{path}:{capture_id}:{case_id}.document_metadata_changes")
    history = History(families[family_name], implementation, captures, path)
    for capture_id in captures:
        history.resolve(capture_id)
    return history


def load_store(root: Path) -> Store:
    root = Path(root)
    family_paths = sorted((root / "families").glob("*.yaml"))
    if not family_paths:
        raise ValueError(f"Unified history has no family files: {root}")
    families = {}
    for path in family_paths:
        family = _validate_family(path, load_yaml(path))
        if family.name in families:
            raise ValueError(f"duplicate family: {family.name}")
        families[family.name] = family
    histories = {}
    for path in sorted((root / "history").glob("*/*.yaml")):
        history = _validate_history(path, load_yaml(path), families)
        key = (history.family.name, history.implementation)
        if key in histories:
            raise ValueError(f"duplicate history: {key}")
        histories[key] = history
    if not histories:
        raise ValueError(f"Unified history has no history files: {root}")
    return Store(root, families, histories)


def store_inventory(root: Path) -> list[dict]:
    store = load_store(root)
    inventory = []
    paths = [*store.root.glob("families/*.yaml"), *store.root.glob("history/*/*.yaml")]
    for path in sorted(paths):
        data = path.read_bytes()
        inventory.append(
            {
                "path": str(path.relative_to(store.root)),
                "sha256": hashlib.sha256(data).hexdigest(),
                "size": len(data),
            }
        )
    return inventory


def store_digest(root: Path) -> tuple[str, int]:
    inventory = store_inventory(root)
    encoded = json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), sum(item["size"] for item in inventory)


def rewrite_store(root: Path) -> None:
    store = load_store(root)
    for family in store.families.values():
        family.path.write_text(dump_yaml(family.document), encoding="utf-8", newline="\n")
    for history in store.histories.values():
        document = {
            "schema_version": SCHEMA_VERSION,
            "family": history.family.name,
            "implementation": history.implementation,
            "captures": history.captures,
        }
        history.path.write_text(dump_yaml(document), encoding="utf-8", newline="\n")


def _case_document(metadata: dict, family: str, case_key: str, record: dict) -> dict:
    return {**metadata, "family": family, "cases": {case_key: record}}


def _write_case(destination: Path, directory: str, family: str, case_key: str, document: dict) -> None:
    _safe_component(directory, "materialized directory")
    _safe_component(family, "materialized family")
    _safe_component(case_key, "materialized case key")
    path = destination / directory / family / f"{case_key}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_yaml(document), encoding="utf-8", newline="\n")


def _binding_bytes(records: dict[str, dict]) -> bytes:
    return (json.dumps({"schema_version": 1, "records": records}, indent=2) + "\n").encode()


def _snapshot_bytes(paths: list[str]) -> bytes:
    return (
        json.dumps({"schema_version": 1, "records": sorted(paths)}, sort_keys=True)
        + "\n"
    ).encode()


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
    elif "partial" in stimulus:
        record["capture_input"] = stimulus["partial"]
    return record, change["document"]


def materialize_store(root: Path, destination: Path, *, include_current_inputs: bool = True) -> None:
    store = load_store(root)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    written: set[Path] = set()

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
            input_path = Path("inputs") / family_name / f"{case_key}.yaml"
            golden_path = Path("golden") / family_name / f"{case_key}.yaml"
            _write_case(
                destination,
                "inputs",
                family_name,
                case_key,
                _case_document(
                    input_metadata,
                    family_name,
                    case_key,
                    {"scenario": case["scenario"], **case["request"]},
                ),
            )
            _write_case(
                destination,
                "golden",
                family_name,
                case_key,
                _case_document(golden_metadata, family_name, case_key, case["golden"]),
            )
            written.update({input_path, golden_path})

    capture_metadata: dict[str, dict] = {}
    capture_bindings: dict[str, dict] = defaultdict(dict)
    for (family_name, _implementation), history in sorted(store.histories.items()):
        for capture_id, capture in history.captures.items():
            metadata = {
                "runtime_version": capture["runtime_version"],
                "provenance": capture["provenance"],
            }
            prior = capture_metadata.setdefault(capture_id, metadata)
            if prior != metadata:
                raise ValueError(f"capture metadata differs across families: {capture_id}")
            state = history.resolve(capture_id)
            for case_id, change in sorted(state.items()):
                case = history.family.cases[case_id]
                case_key = change["case_key"]
                relative = Path(capture_id) / family_name / f"{case_key}.yaml"
                if relative in written:
                    raise ValueError(f"duplicate materialized path: {relative}")
                record, document_metadata = _materialized_record(case, change)
                document_metadata = dict(document_metadata)
                record_metadata = document_metadata.pop("record_metadata", {})
                record.update(record_metadata)
                document = _case_document(document_metadata, family_name, case_key, record)
                _write_case(destination, capture_id, family_name, case_key, document)
                written.add(relative)
                if "capture_input" not in record:
                    continue
                raw = (destination / relative).read_bytes()
                capture_bindings[capture_id][str(relative.relative_to(capture_id))] = {
                    "capture_sha256": hashlib.sha256(raw).hexdigest(),
                    "capture_input": record["capture_input"],
                }

    for capture_id, metadata in capture_metadata.items():
        if "+source." in capture_id:
            paths = [
                str(path.relative_to(destination / capture_id))
                for path in (destination / capture_id).glob("*/*.yaml")
            ]
            (destination / capture_id / fixture_disposition.CAPTURE_SNAPSHOT).write_bytes(
                _snapshot_bytes(paths)
            )


def _archive_files(path: Path, relative: str) -> dict[str, bytes]:
    return fixture_disposition.capture_archive_files(path, relative)


def _archive_layer(path: Path, relative: str) -> dict:
    files = _archive_files(path, relative)
    bindings = json.loads(files["capture-inputs.json"])["records"] if "capture-inputs.json" in files else {}
    snapshot = fixture_disposition.capture_snapshot_members(
        files.get(fixture_disposition.CAPTURE_SNAPSHOT), files
    )
    records = {}
    for member, raw in sorted(files.items()):
        if not member.endswith(".yaml"):
            continue
        document = yaml.safe_load(raw)
        if not isinstance(document, dict) or not isinstance(document.get("cases"), dict):
            raise ValueError(f"invalid Unified archive document: {path}:{member}")
        family = document.get("family")
        if family != PurePosixPath(member).parts[0]:
            raise ValueError(f"archive family differs from its path: {path}:{member}")
        metadata = {key: value for key, value in document.items() if key != "cases"}
        for case_key, record in document["cases"].items():
            records[(family, case_key)] = {
                "record": record,
                "raw": raw,
                "relative": member,
                "bindings": bindings,
                "document": metadata,
            }
    return {"records": records, "snapshot": snapshot is not None, "files": files}


def _capture_order(label: str) -> tuple:
    version = label.split("-", 1)[1].split("+", 1)[0]
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z.-]+))?", version)
    if match:
        prerelease = match[4]
        return (
            0,
            int(match[1]),
            int(match[2]),
            int(match[3]),
            prerelease is None,
            prerelease or "",
            "+" in label,
            label,
        )
    return (1, label)


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


def _document_metadata(document: dict) -> dict:
    return {
        key: value
        for key, value in document.items()
        if key not in DOCUMENT_METADATA_EXCLUDED
    }


def _document_provenance(document: dict) -> dict:
    return {
        key: document[key]
        for key in DOCUMENT_PROVENANCE_KEYS
        if key in document
    }


def _semantic_stimulus(value: dict) -> dict:
    return {key: item for key, item in value.items() if key != "semantic_sha256"}


def _semantic_change(value: dict) -> dict:
    return {
        "stimulus": _semantic_stimulus(value["stimulus"]),
        "observation": value["observation"],
    }


def _comparison_change(value: dict) -> dict:
    return {
        **_semantic_change(value),
        "record_metadata": value["document"].get("record_metadata", {}),
        "document_metadata": _document_metadata(value["document"]),
    }


def _semantic_record(item: dict, current: dict | None) -> dict:
    stimulus, observation, metadata = _legacy_state(
        item["record"], item["raw"], item["relative"], item["bindings"], current
    )
    return {
        "stimulus": _semantic_stimulus(stimulus),
        "observation": observation,
        "record_metadata": metadata,
        "document_metadata": _document_metadata(item["document"]),
    }


def compare_import(repo_root: Path, store_root: Path) -> dict:
    repo_root = Path(repo_root)
    store = load_store(store_root)
    fixture_root = repo_root / "conformance" / "fixtures"
    manifest = json.loads((repo_root / "conformance" / "fixtures-manifest.json").read_text())
    shards = [
        shard
        for shard in fixture_disposition.active_shards(manifest)
        if shard["path"].startswith("unified/") and shard["path"].endswith(".tar.gz")
    ]
    layers = {}
    for shard in shards:
        relative = shard["path"]
        name = Path(relative).name.removesuffix(".tar.gz")
        layers[name] = _archive_layer(fixture_root / relative, f"unified/{name}")

    input_records = {}
    for name, layer in layers.items():
        if name.startswith("inputs"):
            input_records.update(
                {(family, case_key): item["record"] for (family, case_key), item in layer["records"].items()}
            )
    current_inputs, input_aliases = fixture_disposition.canonicalize_unified_inputs(input_records)

    archive_states = {}
    groups: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for name, layer in layers.items():
        if name.startswith(("inputs", "golden")):
            continue
        base, patch = fixture_disposition.capture_layer_sort_key(name)
        groups[base].append((patch, layer))
    for capture_id, capture_layers in groups.items():
        effective = {}
        for _patch, layer in sorted(capture_layers):
            if layer["snapshot"]:
                effective = {}
            for (family, case_key), item in layer["records"].items():
                ident = fixture_disposition.canonical_unified_record_key(
                    family, case_key, input_aliases
                )
                effective[ident] = _semantic_record(item, current_inputs.get(ident))
        archive_states[capture_id] = effective

    yaml_states = defaultdict(dict)
    for (family, _implementation), history in store.histories.items():
        for capture_id in history.captures:
            for case_id, change in history.resolve(capture_id).items():
                case = history.family.cases[case_id]
                canonical = case["display_id"] or next(iter(case["historical_ids"]), None)
                if canonical is None:
                    raise ValueError(f"case has no external identity: {family}/{case_id}")
                canonical = fixture_disposition.historical_unified_case_key(family, canonical)
                yaml_states[capture_id][(family, canonical)] = _comparison_change(change)

    mismatches = []
    for capture_id in sorted(set(archive_states) | set(yaml_states)):
        old = {
            f"{family}/{case_id}": value
            for (family, case_id), value in archive_states.get(capture_id, {}).items()
        }
        new = {
            f"{family}/{case_id}": value
            for (family, case_id), value in yaml_states.get(capture_id, {}).items()
        }
        if _canonical_json(old) != _canonical_json(new):
            mismatches.append(capture_id)
    return {
        "archive_captures": len(archive_states),
        "yaml_captures": len(yaml_states),
        "archive_observations": sum(len(state) for state in archive_states.values()),
        "yaml_observations": sum(len(state) for state in yaml_states.values()),
        "mismatches": mismatches,
    }


def _external_case_id(family: Family, case_id: str) -> tuple[str, str]:
    case = family.cases[case_id]
    external = case["display_id"] or next(iter(case["historical_ids"]), None)
    if external is None:
        raise ValueError(f"case has no external identity: {family.name}/{case_id}")
    return family.name, fixture_disposition.historical_unified_case_key(family.name, external)


def _capture_provenance(
    records: dict[str, dict],
    capture_id: str,
    implementation: str,
    runtime_version: str,
    *,
    strict: bool,
) -> dict:
    identities = {}
    missing = []
    invalid = []
    for case_id, change in records.items():
        document_identity = _document_provenance(change["document"])
        if not document_identity:
            missing.append(case_id)
            continue
        if any(value is None for value in document_identity.values()):
            invalid.append(case_id)
            continue
        if len(document_identity) > 1 and not strict:
            invalid.append(case_id)
            continue
        identities[_canonical_json(document_identity)] = document_identity

    if missing or invalid or len(identities) != 1:
        if strict:
            raise ValueError(f"capture must have one complete provenance identity: {capture_id}")
        if not identities and missing and not invalid:
            return {
                "status": "legacy",
                "captured_with": {implementation: runtime_version},
            }
        return {"status": "mixed", "record_count": len(identities)}

    identity = next(iter(identities.values()))
    if "capture_provenance" in identity:
        return {"status": "captured", "record": identity["capture_provenance"]}
    return {"status": "captured", "captured_with": identity["captured_with"]}


def _validate_addition_provenance(capture: dict, records: dict[str, dict], additions: list[str]) -> None:
    provenance = capture["provenance"]
    expected_record = provenance.get("record")
    expected_captured_with = provenance.get("captured_with")
    if (expected_record is None) == (expected_captured_with is None):
        raise ValueError("capture has no unambiguous provenance identity")
    for case_id in additions:
        document = records[case_id]["document"]
        has_record = "capture_provenance" in document
        has_captured_with = "captured_with" in document
        if expected_record is not None:
            valid = (
                has_record
                and not has_captured_with
                and document["capture_provenance"] == expected_record
            )
        else:
            valid = (
                has_captured_with
                and not has_record
                and document["captured_with"] == expected_captured_with
            )
        if not valid:
            raise ValueError(
                f"capture addition provenance differs from existing identity: {case_id}"
            )


def update_from_loose(store_root: Path, loose_root: Path) -> list[Path]:
    store = load_store(store_root)
    loose_root = Path(loose_root)
    changed_paths = []
    if not loose_root.is_dir():
        return changed_paths
    for capture_dir in sorted(
        path
        for path in loose_root.iterdir()
        if path.is_dir() and re.match(r"^[a-z0-9_]+-\d", path.name)
    ):
        implementation = capture_dir.name.split("-", 1)[0]
        runtime_version = capture_dir.name.split("-", 1)[1]
        families = sorted(path.name for path in capture_dir.iterdir() if path.is_dir())
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
                document = yaml.safe_load(raw)
                metadata = {name: value for name, value in document.items() if name != "cases"}
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
            if capture_dir.name in captures:
                resolved = history.resolve(capture_dir.name)
                conflicts = [
                    case_id
                    for case_id in sorted(set(resolved) & set(records))
                    if _canonical_json(_semantic_change(resolved[case_id]))
                    != _canonical_json(_semantic_change(records[case_id]))
                ]
                if conflicts:
                    raise ValueError(f"capture is immutable; use a new identity: {capture_dir.name}")
                provenance_conflicts = [
                    case_id
                    for case_id in sorted(set(resolved) & set(records))
                    if _canonical_json(_document_provenance(resolved[case_id]["document"]))
                    != _canonical_json(_document_provenance(records[case_id]["document"]))
                ]
                if provenance_conflicts:
                    raise ValueError(
                        f"capture provenance is immutable; use a new identity: {capture_dir.name}"
                    )
                additions = sorted(set(records) - set(resolved))
                _validate_addition_provenance(captures[capture_dir.name], records, additions)
                metadata_changes = {
                    case_id: records[case_id]["document"].get("record_metadata", {})
                    for case_id in sorted(set(resolved) & set(records))
                    if _canonical_json(resolved[case_id]["document"].get("record_metadata", {}))
                    != _canonical_json(records[case_id]["document"].get("record_metadata", {}))
                }
                document_metadata_changes = {
                    case_id: _document_metadata(records[case_id]["document"])
                    for case_id in sorted(set(resolved) & set(records))
                    if _canonical_json(_document_metadata(resolved[case_id]["document"]))
                    != _canonical_json(_document_metadata(records[case_id]["document"]))
                }
                if not additions and not metadata_changes and not document_metadata_changes:
                    continue
                captures[capture_dir.name]["changes"].update(
                    {case_id: records[case_id] for case_id in additions}
                )
                captures[capture_dir.name]["metadata_changes"].update(metadata_changes)
                captures[capture_dir.name].setdefault("document_metadata_changes", {}).update(
                    document_metadata_changes
                )
                document = {
                    "schema_version": SCHEMA_VERSION,
                    "family": family_name,
                    "implementation": implementation,
                    "captures": captures,
                }
                history.path.write_text(dump_yaml(document), encoding="utf-8", newline="\n")
                changed_paths.append(history.path)
                continue
            parent = next(reversed(captures))
            prior = history.resolve(parent)
            changes = {}
            metadata_changes = {}
            document_metadata_changes = {}
            for case_id in sorted(set(prior) | set(records)):
                before = prior.get(case_id)
                after = records.get(case_id)
                before_key = None if before is None else _canonical_json(_semantic_change(before))
                after_key = None if after is None else _canonical_json(_semantic_change(after))
                if before_key == after_key:
                    if before is not None and after is not None:
                        before_metadata = before["document"].get("record_metadata", {})
                        after_metadata = after["document"].get("record_metadata", {})
                        if _canonical_json(before_metadata) != _canonical_json(after_metadata):
                            metadata_changes[case_id] = after_metadata
                        before_document_metadata = _document_metadata(before["document"])
                        after_document_metadata = _document_metadata(after["document"])
                        if _canonical_json(before_document_metadata) != _canonical_json(
                            after_document_metadata
                        ):
                            document_metadata_changes[case_id] = after_document_metadata
                    continue
                changes[case_id] = {"absent": True} if after is None else after
            provenance_records = {
                case_id: records[case_id]
                for case_id, change in changes.items()
                if change != {"absent": True}
            }
            if not provenance_records:
                provenance_records = records
            captures[capture_dir.name] = {
                "parent": parent,
                "runtime_version": runtime_version,
                "provenance": _capture_provenance(
                    {
                        case_id: record
                        for case_id, record in provenance_records.items()
                    },
                    capture_dir.name,
                    implementation,
                    runtime_version,
                    strict=True,
                ),
                "completeness": "delta",
                "import_lineage": [],
                "changes": changes,
                "metadata_changes": metadata_changes,
                "document_metadata_changes": document_metadata_changes,
            }
            document = {
                "schema_version": SCHEMA_VERSION,
                "family": family_name,
                "implementation": implementation,
                "captures": captures,
            }
            history.path.write_text(dump_yaml(document), encoding="utf-8", newline="\n")
            changed_paths.append(history.path)
    load_store(store_root)
    return changed_paths


def sync_current_corpus(store_root: Path, loose_root: Path) -> list[Path]:
    store = load_store(store_root)
    loose_root = Path(loose_root)
    changed_paths = []
    if not (loose_root / "inputs").is_dir() or not (loose_root / "golden").is_dir():
        return []
    for family_name, family in sorted(store.families.items()):
        cases = copy.deepcopy(family.cases)
        by_scenario = {
            case["scenario"]: case_id
            for case_id, case in cases.items()
            if case["lifecycle"] == "active"
        }
        current_scenarios = set()
        input_paths = sorted((loose_root / "inputs" / family_name).glob("*.yaml"))
        if not input_paths:
            continue
        golden_paths = {
            path.stem: path
            for path in (loose_root / "golden" / family_name).glob("*.yaml")
        }
        for input_path in input_paths:
            input_document = yaml.safe_load(input_path.read_bytes())
            for case_key, request in input_document["cases"].items():
                scenario = request.get("scenario")
                if not isinstance(scenario, str):
                    raise ValueError(f"current Unified input has no scenario: {input_path}")
                current_scenarios.add(scenario)
                case_id = by_scenario.get(scenario)
                if case_id is None:
                    case_id = _internal_case_id(case_key, scenario, set(cases))
                    cases[case_id] = {
                        "lifecycle": "active",
                        "scenario": scenario,
                        "display_id": case_key,
                        "historical_ids": [],
                        "request": None,
                        "golden": None,
                    }
                golden_path = golden_paths.get(case_key)
                if golden_path is None:
                    raise ValueError(f"current Unified input has no golden: {family_name}/{case_key}")
                golden_document = yaml.safe_load(golden_path.read_bytes())
                case = cases[case_id]
                old_display = case["display_id"]
                aliases = set(case["historical_ids"])
                if old_display and old_display != case_key:
                    aliases.add(old_display)
                aliases.discard(case_key)
                case.update(
                    {
                        "lifecycle": "active",
                        "scenario": scenario,
                        "display_id": case_key,
                        "historical_ids": sorted(aliases),
                        "request": capture_stimulus.capture_input(request),
                        "golden": golden_document["cases"][case_key],
                    }
                )
                case["request"]["tools"] = request.get("tools")
                family.document["input_document"] = {
                    key: value for key, value in input_document.items() if key != "cases"
                }
                family.document["golden_document"] = {
                    key: value for key, value in golden_document.items() if key != "cases"
                }
        for case in cases.values():
            if case["lifecycle"] == "active" and case["scenario"] not in current_scenarios:
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
        family.document["cases"] = cases
        rendered = dump_yaml(family.document)
        if family.path.read_text() != rendered:
            family.path.write_text(rendered, encoding="utf-8", newline="\n")
            changed_paths.append(family.path)
    load_store(store_root)
    return changed_paths


def import_archives(repo_root: Path, destination: Path) -> dict:
    repo_root = Path(repo_root)
    destination = Path(destination)
    fixture_root = repo_root / "conformance" / "fixtures"
    manifest = json.loads((repo_root / "conformance" / "fixtures-manifest.json").read_text())
    shards = [
        shard
        for shard in fixture_disposition.active_shards(manifest)
        if shard["path"].startswith("unified/") and shard["path"].endswith(".tar.gz")
    ]
    if not shards:
        raise ValueError("manifest has no active Unified archives to import")
    by_path = {shard["path"]: shard for shard in shards}
    layers = {}
    for relative, shard in sorted(by_path.items()):
        path = fixture_root / relative
        data = path.read_bytes()
        if len(data) != shard["size"] or hashlib.sha256(data).hexdigest() != shard["sha256"]:
            raise ValueError(f"Unified archive differs from manifest: {relative}")
        name = Path(relative).name.removesuffix(".tar.gz")
        layers[name] = _archive_layer(path, f"unified/{name}")

    input_records = {}
    golden_records = {}
    input_metadata = {}
    golden_metadata = {}
    raw_ids: dict[tuple[str, str], set[str]] = defaultdict(set)
    for name, layer in layers.items():
        target = input_records if name.startswith("inputs") else golden_records if name.startswith("golden") else None
        metadata_target = input_metadata if name.startswith("inputs") else golden_metadata if name.startswith("golden") else None
        if target is None:
            continue
        for (family, case_key), item in layer["records"].items():
            target[(family, case_key)] = item["record"]
            metadata_target[family] = item["document"]
            raw_ids[(family, fixture_disposition.historical_unified_case_key(family, case_key))].add(case_key)

    current_inputs, input_aliases = fixture_disposition.canonicalize_unified_inputs(input_records)
    current_golden = {}
    for (family, case_key), record in golden_records.items():
        ident = fixture_disposition.canonical_unified_record_key(family, case_key, input_aliases)
        current_golden[ident] = record

    capture_groups: dict[str, list[tuple[int, str, dict]]] = defaultdict(list)
    for name, layer in layers.items():
        if name.startswith(("inputs", "golden")):
            continue
        base, patch = fixture_disposition.capture_layer_sort_key(name)
        capture_groups[base].append((patch, name, layer))

    effective_captures = {}
    lineages = {}
    all_aliases: dict[tuple[str, str], set[str]] = defaultdict(set)
    for capture_id, capture_layers in sorted(capture_groups.items()):
        effective = {}
        lineage = []
        for _patch, name, layer in sorted(capture_layers):
            if layer["snapshot"]:
                effective = {}
            for (family, case_key), item in layer["records"].items():
                ident = fixture_disposition.canonical_unified_record_key(
                    family, case_key, input_aliases
                )
                effective[ident] = item | {"case_key": case_key}
                all_aliases[ident].add(case_key)
            relative = f"unified/{name}.tar.gz"
            lineage.append(
                {
                    "archive": Path(relative).name,
                    "sha256": by_path[relative]["sha256"],
                    "mode": "snapshot" if layer["snapshot"] else "overlay",
                }
            )
        effective_captures[capture_id] = effective
        lineages[capture_id] = lineage

    all_idents = set(current_inputs)
    for records in effective_captures.values():
        all_idents.update(records)

    cases_by_family = defaultdict(dict)
    ident_to_internal = {}
    used_by_family: dict[str, set[str]] = defaultdict(set)
    for family, canonical_case_key in sorted(all_idents):
        ident = (family, canonical_case_key)
        current = current_inputs.get(ident)
        scenario = current.get("scenario") if current else None
        internal = _internal_case_id(canonical_case_key, scenario, used_by_family[family])
        ident_to_internal[ident] = internal
        aliases = set(all_aliases.get(ident, set()))
        aliases.update(raw_ids.get((family, fixture_disposition.historical_unified_case_key(family, canonical_case_key)), set()))
        aliases.discard(canonical_case_key)
        if current is not None:
            request = capture_stimulus.capture_input(current)
            request["tools"] = current.get("tools")
            case = {
                "lifecycle": "active",
                "scenario": scenario,
                "display_id": canonical_case_key,
                "historical_ids": sorted(aliases),
                "request": request,
                "golden": current_golden.get(ident),
            }
            if case["golden"] is None:
                raise ValueError(f"active case has no golden: {family}/{canonical_case_key}")
        else:
            retained = []
            for records in effective_captures.values():
                item = records.get(ident)
                if item is None:
                    continue
                original = capture_stimulus.original_capture_input(
                    item["record"], item["raw"], item["relative"], item["bindings"]
                )
                if isinstance(original, dict) and original.keys() == REQUEST_KEYS and original.get("tools") is not None:
                    retained.append(original)
            unique = {_canonical_json(value): value for value in retained}
            request = next(iter(unique.values())) if len(unique) == 1 else None
            case = {
                "lifecycle": "retired",
                "scenario": None,
                "display_id": None,
                "historical_ids": sorted({canonical_case_key, *aliases}),
                "request": request,
                "golden": None,
            }
        cases_by_family[family][internal] = case

    histories = {}
    labels_by_impl = defaultdict(list)
    for capture_id in effective_captures:
        labels_by_impl[capture_id.split("-", 1)[0]].append(capture_id)
    for implementation in labels_by_impl:
        labels_by_impl[implementation].sort(key=_capture_order)

    states = {}
    for capture_id, records in effective_captures.items():
        state = {}
        for ident, item in records.items():
            current = current_inputs.get(ident)
            stimulus, observation, metadata = _legacy_state(
                item["record"], item["raw"], item["relative"], item["bindings"], current
            )
            state[ident] = {
                "case_key": item["case_key"],
                "stimulus": stimulus,
                "observation": observation,
                "document": {
                    **item["document"],
                    **({"record_metadata": metadata} if metadata else {}),
                },
            }
        states[capture_id] = state

    for implementation, labels in sorted(labels_by_impl.items()):
        capture_provenance = {
            capture_id: _capture_provenance(
                states[capture_id],
                capture_id,
                implementation,
                capture_id.split("-", 1)[1],
                strict=False,
            )
            for capture_id in labels
        }
        for family in sorted(cases_by_family):
            family_labels = [
                label for label in labels if any(ident[0] == family for ident in states[label])
            ]
            if not family_labels:
                continue
            captures = {}
            prior_id = None
            prior_state = {}
            for capture_id in family_labels:
                current_state = {
                    ident_to_internal[ident]: value
                    for ident, value in states[capture_id].items()
                    if ident[0] == family
                }
                changes = {}
                metadata_changes = {}
                document_metadata_changes = {}
                for case_id in sorted(set(prior_state) | set(current_state)):
                    before = prior_state.get(case_id)
                    after = current_state.get(case_id)
                    before_key = None if before is None else _canonical_json(_semantic_change(before))
                    after_key = None if after is None else _canonical_json(_semantic_change(after))
                    if before_key == after_key:
                        if before is not None and after is not None:
                            before_metadata = before["document"].get("record_metadata", {})
                            after_metadata = after["document"].get("record_metadata", {})
                            if _canonical_json(before_metadata) != _canonical_json(after_metadata):
                                metadata_changes[case_id] = after_metadata
                            before_document_metadata = _document_metadata(before["document"])
                            after_document_metadata = _document_metadata(after["document"])
                            if _canonical_json(before_document_metadata) != _canonical_json(
                                after_document_metadata
                            ):
                                document_metadata_changes[case_id] = after_document_metadata
                        continue
                    changes[case_id] = {"absent": True} if after is None else after
                captures[capture_id] = {
                    "parent": prior_id,
                    "runtime_version": capture_id.split("-", 1)[1],
                    "provenance": capture_provenance[capture_id],
                    "completeness": "snapshot" if prior_id is None else "delta",
                    "import_lineage": lineages[capture_id],
                    "changes": changes,
                    "metadata_changes": metadata_changes,
                    "document_metadata_changes": document_metadata_changes,
                }
                prior_id = capture_id
                prior_state = current_state
            histories[(family, implementation)] = captures

    if destination.exists():
        shutil.rmtree(destination)
    (destination / "families").mkdir(parents=True)
    (destination / "history").mkdir(parents=True)
    for family, cases in sorted(cases_by_family.items()):
        document = {
            "schema_version": SCHEMA_VERSION,
            "family": family,
            "input_document": input_metadata.get(family),
            "golden_document": golden_metadata.get(family),
            "cases": cases,
        }
        (destination / "families" / f"{family}.yaml").write_text(
            dump_yaml(document), encoding="utf-8", newline="\n"
        )
    for (family, implementation), captures in sorted(histories.items()):
        path = destination / "history" / family / f"{implementation}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            dump_yaml(
                {
                    "schema_version": SCHEMA_VERSION,
                    "family": family,
                    "implementation": implementation,
                    "captures": captures,
                }
            ),
            encoding="utf-8",
            newline="\n",
        )

    store = load_store(destination)
    observed = sum(
        len(history.resolve(capture_id))
        for history in store.histories.values()
        for capture_id in history.captures
    )
    return {
        "active_archives": len(shards),
        "families": len(store.families),
        "histories": len(store.histories),
        "captures": len(effective_captures),
        "nodes": sum(len(history.captures) for history in store.histories.values()),
        "current_cases": len(current_inputs),
        "case_identities": len(all_idents),
        "effective_observations": sum(len(records) for records in effective_captures.values()),
        "resolved_history_observations": observed,
        "changed_entries": sum(
            len(capture["changes"])
            for history in store.histories.values()
            for capture in history.captures.values()
        ),
        "unchanged_nodes": sum(
            not capture["changes"]
            for history in store.histories.values()
            for capture in history.captures.values()
        ),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--import-archives", action="store_true")
    action.add_argument("--materialize", action="store_true")
    action.add_argument("--validate", action="store_true")
    action.add_argument("--compare-import", action="store_true")
    action.add_argument("--update-from-loose", action="store_true")
    action.add_argument("--sync-current-corpus", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--history-only", action="store_true")
    args = parser.parse_args(argv)
    if args.import_archives:
        print(json.dumps(import_archives(args.repo_root, args.store), indent=2))
    elif args.materialize:
        if args.output is None:
            parser.error("--materialize requires --output")
        materialize_store(args.store, args.output, include_current_inputs=not args.history_only)
    elif args.compare_import:
        result = compare_import(args.repo_root, args.store)
        print(json.dumps(result, indent=2))
        if result["mismatches"]:
            raise SystemExit(1)
    elif args.update_from_loose:
        if args.output is None:
            parser.error("--update-from-loose requires --output")
        print(json.dumps([str(path) for path in update_from_loose(args.store, args.output)], indent=2))
    elif args.sync_current_corpus:
        if args.output is None:
            parser.error("--sync-current-corpus requires --output")
        print(
            json.dumps(
                [str(path) for path in sync_current_corpus(args.store, args.output)],
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
