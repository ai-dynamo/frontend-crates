# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import copy
import hashlib
import io
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

import fixture_disposition
import yaml


def case_sha256(case: object) -> str:
    payload = json.dumps(case, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _documents(
    path: Path,
    root: PurePosixPath,
    *,
    capture_version: str | None,
    allow_any_capture: bool = False,
) -> tuple[dict[str, dict], dict[str, str]]:
    found = {}
    preserved = {}
    members = set()
    case_ids = set()
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            member_path = PurePosixPath(member.name)
            if (
                member_path.is_absolute()
                or ".." in member_path.parts
                or member_path in members
                or (member_path != root and root not in member_path.parents)
                or not (member.isfile() or member.isdir())
            ):
                raise ValueError(f"unexpected stream archive member: {member.name}")
            members.add(member_path)
            if member.isdir():
                continue
            relative = member_path.relative_to(root)
            with archive.extractfile(member) as source:
                payload = source.read()
            if member_path.suffix == ".yaml":
                if len(relative.parts) != 2:
                    raise ValueError(f"unexpected stream archive member: {member.name}")
                document = yaml.safe_load(payload)
                expected_capture = {"dynamo_v2": capture_version} if capture_version is not None else None
                if (
                    not isinstance(document, dict)
                    or document.get("family") != relative.parts[0]
                    or document.get("mode") != "streamv1"
                    or document.get("cases") is None
                    or not isinstance(document.get("cases"), dict)
                    or (capture_version is not None and document.get("captured_with") != expected_capture)
                    or (capture_version is None and not allow_any_capture and "captured_with" in document)
                ):
                    raise ValueError(f"invalid stream archive document: {member.name}")
                for case_id in document["cases"]:
                    ident = (document["family"], case_id)
                    if ident in case_ids:
                        raise ValueError(f"duplicate stream archive case: {ident}")
                    case_ids.add(ident)
                found[str(relative)] = document
            else:
                preserved[str(relative)] = hashlib.sha256(payload).hexdigest()
    return found, preserved


def _approved_case_additions(
    old_docs: dict[str, dict],
    new_docs: dict[str, dict],
    old_members: dict[str, str],
    new_members: dict[str, str],
    allowed_replacements: set[tuple[str, str]] | frozenset[tuple[str, str]] = frozenset(),
) -> tuple[set[tuple[str, str]], set[tuple[str, str]]] | None:
    if old_members != new_members or not old_docs.keys() <= new_docs.keys():
        return None
    added = set()
    replaced = set()
    for name, old_doc in old_docs.items():
        new_doc = new_docs[name]
        old_metadata = {key: value for key, value in old_doc.items() if key != "cases"}
        new_metadata = {key: value for key, value in new_doc.items() if key != "cases"}
        if not _same_yaml_value(old_metadata, new_metadata):
            return None
        old_cases = old_doc["cases"]
        new_cases = new_doc["cases"]
        for case_id, old_case in old_cases.items():
            if case_id not in new_cases:
                return None
            if not _same_yaml_value(new_cases[case_id], old_case):
                ident = (new_doc["family"], case_id)
                if ident not in allowed_replacements:
                    return None
                replaced.add(ident)
        added.update((new_doc["family"], case_id) for case_id in new_cases.keys() - old_cases.keys())
    for name, new_doc in new_docs.items():
        if name not in old_docs:
            added.update((new_doc["family"], case_id) for case_id in new_doc["cases"])
    return added, replaced


def _same_yaml_value(left: object, right: object) -> bool:
    """Compare YAML values without Python's bool/int equality coercion."""
    return json.dumps(left, sort_keys=True, separators=(",", ":"), ensure_ascii=False) == json.dumps(
        right, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def stream_input_case_ids(path: Path) -> set[tuple[str, str]]:
    docs, _members = _documents(path, PurePosixPath("toolcalling/fixtures-stream-v1/inputs"), capture_version=None)
    return {
        (document["family"], case_id)
        for document in docs.values()
        for case_id in document["cases"]
    }


def stream_input_changes_allowed(
    existing: Path,
    candidate: Path,
    allowed_case_replacements: set[tuple[str, str]] | frozenset[tuple[str, str]] = frozenset(),
) -> set[tuple[str, str]] | None:
    changes = stream_input_changes(existing, candidate, allowed_case_replacements)
    return None if changes is None else changes[0]


def stream_input_changes(
    existing: Path,
    candidate: Path,
    allowed_case_replacements: set[tuple[str, str]] | frozenset[tuple[str, str]] = frozenset(),
) -> tuple[set[tuple[str, str]], set[tuple[str, str]]] | None:
    root = PurePosixPath("toolcalling/fixtures-stream-v1/inputs")
    old_docs, old_members = _documents(existing, root, capture_version=None)
    new_docs, new_members = _documents(candidate, root, capture_version=None)
    return _approved_case_additions(old_docs, new_docs, old_members, new_members, allowed_case_replacements)


def stream_capture_case_ids(path: Path, capture_root: str) -> set[tuple[str, str]]:
    root = PurePosixPath("toolcalling/fixtures-stream-v1") / capture_root
    if capture_root != root.name or not capture_root.startswith("dynamo_v2-"):
        raise ValueError(f"invalid stream capture root: {capture_root}")
    capture_version = capture_root.removeprefix("dynamo_v2-")
    docs, _members = _documents(path, root, capture_version=capture_version)
    return {
        (document["family"], case_id)
        for document in docs.values()
        for case_id in document["cases"]
    }


def _case_locations(docs):
    locations = {}
    for relative, document in docs.items():
        for case_id, case in document["cases"].items():
            ident = (document["family"], case_id)
            if ident in locations:
                return None
            locations[ident] = (relative, document, case)
    return locations


def stream_recorded_case_ids(path: Path, capture_root: str) -> set[tuple[str, str]]:
    root = PurePosixPath("toolcalling/fixtures-stream-v1") / capture_root
    if capture_root != root.name or capture_root == "inputs":
        raise ValueError(f"invalid stream capture root: {capture_root}")
    docs, _members = _documents(path, root, capture_version=None, allow_any_capture=True)
    return {
        (document["family"], case_id)
        for document in docs.values()
        for case_id in document["cases"]
    }


def stream_capture_patch_allowed(
    existing: Path,
    candidate: Path,
    capture_root: str,
    patch_root: str,
    required_replacements: set[tuple[str, str]] | frozenset[tuple[str, str]],
    existing_patches: dict[str, Path] | None = None,
) -> bool:
    capture_base, capture_patch = fixture_disposition.capture_layer_sort_key(capture_root)
    patch_base, patch_number = fixture_disposition.capture_layer_sort_key(patch_root)
    existing_patches = existing_patches or {}
    previous_patch_numbers = [capture_patch]
    for previous_root in existing_patches:
        previous_base, previous_patch = fixture_disposition.capture_layer_sort_key(previous_root)
        if previous_base != capture_base:
            return False
        previous_patch_numbers.append(previous_patch)
    if patch_base != capture_base or patch_number <= max(previous_patch_numbers) or patch_number == 0:
        return False

    patch_path = PurePosixPath("toolcalling/fixtures-stream-v1") / patch_root
    patch_docs, patch_members = _documents(candidate, patch_path, capture_version=None, allow_any_capture=True)
    if patch_members:
        return False

    patch_cases = _case_locations(patch_docs)
    if patch_cases is None:
        return False
    if set(patch_cases) != required_replacements:
        return False

    old_layers = {capture_root: existing, **existing_patches}
    old_cases = {}
    for old_root, old_path in sorted(
        old_layers.items(), key=lambda item: fixture_disposition.capture_layer_sort_key(item[0])
    ):
        layer_root = PurePosixPath("toolcalling/fixtures-stream-v1") / old_root
        old_docs, _old_members = _documents(
            old_path, layer_root, capture_version=None, allow_any_capture=True
        )
        layer_cases = _case_locations(old_docs)
        if layer_cases is None:
            return False
        old_cases.update(layer_cases)
    if not required_replacements or not required_replacements <= old_cases.keys():
        return False
    for family, case_id in required_replacements:
        old_relative, old_doc, _old_case = old_cases[(family, case_id)]
        patch_relative, patch_doc, _patch_case = patch_cases[(family, case_id)]
        if old_relative != patch_relative:
            return False
        if not _same_yaml_value(
            {key: value for key, value in old_doc.items() if key != "cases"},
            {key: value for key, value in patch_doc.items() if key != "cases"},
        ):
            return False
    return True


def validate_capture_receipt(
    receipt: dict,
    inputs_path: Path,
    capture_path: Path,
    capture_root: str,
    required_cases: set[tuple[str, str]] | frozenset[tuple[str, str]],
    expected_source_sha256: str,
) -> bool:
    if receipt.get("format") != "dynamo-stream-capture-receipt-v2":
        return False
    captures = receipt.get("captures")
    if not isinstance(captures, dict):
        return False
    root = PurePosixPath("toolcalling/fixtures-stream-v1") / capture_root
    if capture_root != root.name or not capture_root.startswith("dynamo_v2-"):
        raise ValueError(f"invalid stream capture root: {capture_root}")
    version = capture_root.removeprefix("dynamo_v2-")
    input_docs, _input_members = _documents(
        inputs_path, PurePosixPath("toolcalling/fixtures-stream-v1/inputs"), capture_version=None
    )
    capture_docs, _capture_members = _documents(capture_path, root, capture_version=version)
    input_cases = {
        (document["family"], case_id): document["cases"][case_id]
        for document in input_docs.values()
        for case_id in document["cases"]
    }
    captured_cases = {
        (document["family"], case_id): document["cases"][case_id]
        for document in capture_docs.values()
        for case_id in document["cases"]
    }
    capture_receipt = captures.get(capture_root)
    if (
        not isinstance(capture_receipt, dict)
        or capture_receipt.get("producer_source_sha256") != expected_source_sha256
        or not isinstance(capture_receipt.get("cases"), dict)
    ):
        return False
    receipt_cases = capture_receipt["cases"]
    for family, case_id in required_cases:
        relative = next(
            (name for name, document in input_docs.items() if document["family"] == family and case_id in document["cases"]),
            None,
        )
        capture_relative = next(
            (name for name, document in capture_docs.items() if document["family"] == family and case_id in document["cases"]),
            None,
        )
        if relative is None or capture_relative is None:
            return False
        if relative != capture_relative:
            return False
        entries = receipt_cases.get(relative)
        if not isinstance(entries, dict):
            return False
        entry = entries.get(case_id)
        if not isinstance(entry, dict):
            return False
        if (
            entry.get("input_sha256") != case_sha256(input_cases[(family, case_id)])
            or entry.get("result_sha256") != case_sha256(captured_cases[(family, case_id)])
        ):
            return False
    return True


def stream_capture_updates_allowed(
    existing: Path,
    candidate: Path,
    capture_root: str,
    allowed_case_additions: set[tuple[str, str]],
    allowed_case_replacements: set[tuple[str, str]] | frozenset[tuple[str, str]] = frozenset(),
    known_input_cases: set[tuple[str, str]] | frozenset[tuple[str, str]] | None = None,
) -> bool:
    """Allow supplied input-approved additions and explicit corrections."""
    root = PurePosixPath("toolcalling/fixtures-stream-v1") / capture_root
    if capture_root != root.name or not capture_root.startswith("dynamo_v2-"):
        raise ValueError(f"invalid stream capture root: {capture_root}")
    capture_version = capture_root.removeprefix("dynamo_v2-")
    old_docs, old_members = _documents(existing, root, capture_version=capture_version)
    new_docs, new_members = _documents(candidate, root, capture_version=capture_version)
    changes = _approved_case_additions(old_docs, new_docs, old_members, new_members, allowed_case_replacements)
    if known_input_cases is not None and not allowed_case_additions <= known_input_cases:
        return False
    if changes is None or not changes[0] <= allowed_case_additions:
        return False
    old_cases = _case_locations(old_docs)
    new_cases = _case_locations(new_docs)
    if old_cases is None or new_cases is None:
        return False
    return True


def merge_stream_capture_additions(
    existing: Path,
    candidate: Path,
    capture_root: str,
    allowed_case_additions: set[tuple[str, str]],
    allowed_case_replacements: set[tuple[str, str]] | frozenset[tuple[str, str]] = frozenset(),
    known_input_cases: set[tuple[str, str]] | frozenset[tuple[str, str]] | None = None,
) -> bool:
    root = PurePosixPath("toolcalling/fixtures-stream-v1") / capture_root
    if capture_root != root.name or not capture_root.startswith("dynamo_v2-"):
        raise ValueError(f"invalid stream capture root: {capture_root}")
    capture_version = capture_root.removeprefix("dynamo_v2-")
    old_docs, old_members = _documents(existing, root, capture_version=capture_version)
    new_docs, new_members = _documents(candidate, root, capture_version=capture_version)
    changes = _approved_case_additions(old_docs, new_docs, old_members, new_members, allowed_case_replacements)
    if changes is None:
        return False
    if known_input_cases is not None and not allowed_case_additions <= known_input_cases:
        return False

    old_case_locations = _case_locations(old_docs)
    new_case_locations = _case_locations(new_docs)
    if old_case_locations is None or new_case_locations is None:
        return False
    additions, replacements = changes
    selected = (additions & allowed_case_additions) | replacements
    if not selected:
        shutil.copyfile(existing, candidate)
        return True
    merged_docs = copy.deepcopy(old_docs)
    changed_docs = set()
    for family, case_id in selected:
        relative, source_doc, case = new_case_locations[(family, case_id)]
        destination_doc = merged_docs.get(relative)
        if destination_doc is None:
            destination_doc = {key: value for key, value in source_doc.items() if key != "cases"}
            destination_doc["cases"] = {}
            merged_docs[relative] = destination_doc
        elif not _same_yaml_value(
            {key: value for key, value in destination_doc.items() if key != "cases"},
            {key: value for key, value in source_doc.items() if key != "cases"},
        ):
            return False
        destination_doc["cases"][case_id] = copy.deepcopy(case)
        changed_docs.add(relative)

    updated_payloads = {
        str(root / relative): yaml.safe_dump(
            merged_docs[relative], sort_keys=False, allow_unicode=True, width=4096
        ).encode("utf-8")
        for relative in changed_docs
    }
    old_names = {str(root / relative) for relative in old_docs}
    new_names = {str(root / relative) for relative in merged_docs} - old_names
    with tarfile.open(candidate, "r:gz") as new_archive:
        candidate_members = {member.name: member for member in new_archive.getmembers()}
        for name in new_names:
            if name not in candidate_members or not candidate_members[name].isfile():
                return False
    temporary_fd, temporary_name = tempfile.mkstemp(prefix=f"{candidate.name}.", dir=candidate.parent)
    os.close(temporary_fd)
    try:
        with tarfile.open(existing, "r:gz") as old_archive, tarfile.open(temporary_name, "w:gz") as output:
            for member in old_archive.getmembers():
                payload = None
                if member.isfile():
                    with old_archive.extractfile(member) as source:
                        payload = source.read()
                info = copy.copy(member)
                if member.name in updated_payloads:
                    payload = updated_payloads[member.name]
                    info.size = len(payload)
                output.addfile(info, io.BytesIO(payload) if info.isfile() else None)
            for name in sorted(new_names):
                info = copy.copy(candidate_members[name])
                payload = updated_payloads[name]
                info.size = len(payload)
                output.addfile(info, io.BytesIO(payload))
        os.replace(temporary_name, candidate)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    return True
