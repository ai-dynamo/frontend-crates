# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate append-only updates to the current Dynamo stream capture archive."""

import hashlib
import tarfile
from pathlib import Path, PurePosixPath

import yaml


def stream_capture_additions_only(existing: Path, candidate: Path, capture_root: str) -> bool:
    """Return whether candidate adds cases without rewriting captured results.

    Older capture versions remain immutable. The caller permits this only for
    the checked-out Dynamo v2 crate version, so a chore can backfill a newly
    authored stream case into its current semantic archive.
    """
    def documents(path: Path) -> tuple[dict[str, dict], dict[str, str]]:
        found = {}
        preserved = {}
        members = set()
        with tarfile.open(path, "r:gz") as archive:
            for member in archive.getmembers():
                if not member.isfile():
                    continue
                prefix = f"toolcalling/fixtures-stream-v1/{capture_root}/"
                relative = member.name.removeprefix(prefix)
                if (
                    not member.name.startswith(prefix)
                    or ".." in PurePosixPath(relative).parts
                    or relative in members
                ):
                    raise ValueError(f"unexpected stream capture member: {member.name}")
                members.add(relative)
                with archive.extractfile(member) as source:
                    payload = source.read()
                if relative.endswith(".yaml"):
                    found[relative] = yaml.safe_load(payload)
                else:
                    preserved[relative] = hashlib.sha256(payload).hexdigest()
        return found, preserved

    old_docs, old_members = documents(existing)
    new_docs, new_members = documents(candidate)
    if old_members != new_members or old_docs.keys() != new_docs.keys():
        return False
    for name, old_doc in old_docs.items():
        new_doc = new_docs.get(name)
        if (
            new_doc is None
            or old_doc.get("family") != new_doc.get("family")
            or old_doc.get("mode") != new_doc.get("mode")
            or old_doc.get("captured_with") != new_doc.get("captured_with")
        ):
            return False
        old_cases = old_doc.get("cases") or {}
        new_cases = new_doc.get("cases") or {}
        if any(new_cases.get(case_id) != old_case for case_id, old_case in old_cases.items()):
            return False
    return True
