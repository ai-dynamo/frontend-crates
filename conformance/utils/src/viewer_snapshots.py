# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate resolved viewer JSON only inside a staged publication."""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
from pathlib import Path

import unified_history
from viewer_codec import encode_typed


FORMAT = "typed-json-v1"


def json_bytes(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode()


def resolved_metadata_digest(records: dict) -> str:
    metadata = {case_id: {"document": record["value"]["document"], "member": record["member"]}
                for case_id, record in records.items()}
    return hashlib.sha256(json_bytes(encode_typed(metadata))).hexdigest()


def publish_snapshots(root: Path, destination: Path, catalog: dict, *, store: unified_history.Store | None = None) -> dict:
    original = catalog["sources"]["unified"]
    if not original:
        return copy.deepcopy(catalog)
    fingerprint = hashlib.sha256(json_bytes({"resources": original})).hexdigest()
    store = store or unified_history.load_store(root / "conformance/fixtures-unified-v2")
    resources = []

    def write(kind: str, family: str, value: dict, *, implementation: str | None = None, version: str | None = None) -> None:
        name = f"{implementation}-{version}" if implementation else "inputs-and-golden"
        relative = f"conformance/viewer-data/{family}/{name}.json.gz"
        envelope = {"schema": 1, "kind": kind, "family": family,
                    "implementation": implementation, "version": version,
                    "source_fingerprint": fingerprint, "value": encode_typed(value)}
        data = gzip.compress(json_bytes(envelope), mtime=0)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        resources.append({"path": relative, "sha256": hashlib.sha256(data).hexdigest(), "size": len(data),
                          "format": FORMAT, "compression": "gzip", "kind": kind, "family": family,
                          "implementation": implementation, "version": version,
                          "record_contract": catalog.get("metadata", {}).get("unified_capture_policy", {}).get(implementation, {}).get("mode") if implementation else None,
                          "capture_identity": f"{implementation}@unified@{version}" if implementation else None,
                          "source_fingerprint": fingerprint})
        if kind == "resolved-capture":
            resources[-1]["resolved_metadata_sha256"] = resolved_metadata_digest(value)

    for family, authored in sorted(store.families.items()):
        write("inputs-and-golden", family, authored.document)
    versions = {}
    for candidate in catalog["sources"]["candidates"]:
        if candidate["mode"] == "unified":
            versions.setdefault(candidate["implementation"], set()).add(candidate["version"])
    for (family, implementation), history in sorted(store.histories.items()):
        origins = {}
        origin_snapshots = {}
        for capture_id in history.ordered_capture_ids():
            for case_id, change in history.captures[capture_id]["changes"].items():
                if change == {"absent": True}:
                    origins.pop(case_id, None)
                elif case_id not in origins or "observation" in change:
                    origins[case_id] = history.capture_path(capture_id).relative_to(root).as_posix()
            origin_snapshots[capture_id] = dict(origins)
        for version in sorted(versions.get(implementation, []), key=unified_history._capture_release_sort_key):
            eligible = [capture_id for capture_id in history.ordered_capture_ids()
                        if unified_history._capture_release_sort_key(history.captures[capture_id]["runtime_version"])
                        <= unified_history._capture_release_sort_key(version)]
            if not eligible:
                continue
            capture_id = eligible[-1]
            state = history.resolve(capture_id)
            if capture_id != f"{implementation}-{version}":
                for record in state.values():
                    record.get("document", {}).get("record_metadata", {}).pop("verified_recapture", None)
            records = {case_id: {"value": value, "member": origin_snapshots[capture_id][case_id]}
                       for case_id, value in state.items()}
            write("resolved-capture", family, records, implementation=implementation, version=version)
    result = copy.deepcopy(catalog)
    result["source_revision"] = catalog["revision"]
    result["sources"]["original_unified"] = original
    result["sources"]["unified"] = resources
    result.pop("revision")
    result["revision"] = hashlib.sha256(json_bytes(result)).hexdigest()
    return result
