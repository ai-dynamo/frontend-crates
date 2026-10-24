# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate request membership and outcomes for all Unified execution receipts."""

import hashlib
import json
from pathlib import Path, PurePosixPath
import re

from capture_stimulus import batch_final_events, capture_input
from viewer_codec import typed_equal


def request_for_case(case):
    if "request" in case:
        return case["request"]
    return capture_input({**case, "chunks": [{"delta_text": text} for text in case["chunks"]]})


def validate_adapter_bundle(root, manifest):
    """Recheck retained producer bytes in their original receipt hash order."""
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("frozen adapter bundle must be a regular directory")
    files = manifest.get("files")
    support = manifest.get("support_files", [])
    if not isinstance(files, list) or not files or not isinstance(support, list):
        raise ValueError("invalid frozen adapter bundle files")
    seen = set()
    hashed = []
    for entries, include in ((files, True), (support, False)):
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
                raise ValueError("invalid frozen adapter file")
            relative = entry["path"]
            if not isinstance(relative, str) or not relative or "\\" in relative or PurePosixPath(relative).is_absolute() or any(part in {"", ".", ".."} for part in relative.split("/")) or relative in seen:
                raise ValueError("unsafe or duplicate frozen adapter path")
            seen.add(relative)
            path = root / relative
            if any((root / Path(*PurePosixPath(relative).parts[:index])).is_symlink() for index in range(1, len(PurePosixPath(relative).parts) + 1)) or not path.is_file():
                raise ValueError("frozen adapter file is missing or symbolic")
            contents = path.read_bytes()
            if not isinstance(entry["sha256"], str) or re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is None or hashlib.sha256(contents).hexdigest() != entry["sha256"]:
                raise ValueError("frozen adapter file hash mismatch")
            if include:
                hashed.append(contents)
    return hashlib.sha256(b"".join(hashed)).hexdigest()


def validate_execution(document, job, job_bytes, implementation, version, *, expected_adapter_sha256=None):
    version_field = {"sglang_python": "sglang_version", "vllm_python": "vllm_version",
                     "vllm_rust": "vllm_rust_version"}.get(implementation, "version")
    receipt = document["provenance"]
    for key in ("source_sha256", "adapter_sha256", "job_sha256", "results_sha256"):
        if not isinstance(receipt.get(key), str) or re.fullmatch(r"[0-9a-f]{64}", receipt[key]) is None:
            raise ValueError(f"coverage receipt has invalid {key}")
    if expected_adapter_sha256 is not None and receipt["adapter_sha256"] != expected_adapter_sha256:
        raise ValueError("coverage adapter differs from frozen producer bundle")
    if document.get(version_field) != version or receipt["implementation"] != implementation or receipt["version"] != version:
        raise ValueError("coverage producer identity mismatch")
    if receipt["job_sha256"] != hashlib.sha256(job_bytes).hexdigest():
        raise ValueError("coverage request receipt mismatch")
    digest = hashlib.sha256(json.dumps(document["results"], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if receipt["results_sha256"] != digest:
        raise ValueError("coverage result receipt mismatch")
    ids = [case["id"] for case in job["cases"]]
    if len(set(ids)) != len(ids) or set(document["results"]) != set(ids):
        raise ValueError("coverage results differ from frozen cases")
    for case in job["cases"]:
        if not typed_equal(document["results"][case["id"]].get("capture_input"), request_for_case(case)):
            raise ValueError("coverage result request mismatch")


def execution_outcome(record, mode):
    failed = [key for key in ("error", "unavailable") if key in record]
    if len(failed) > 1 or ("unavailable" in record and any(key in record for key in ("assembled", "chunks", "batch_final"))):
        raise ValueError("coverage result must have exactly one outcome")
    if failed:
        if mode == "batch" and any(key in record for key in ("assembled", "chunks", "batch_final")):
            raise ValueError("failed batch capture contains successful output")
        for field in ("chunks", "partial_chunks"):
            if field in record:
                validate_chunks(record[field])
        if "assembled" in record:
            validate_events(record["assembled"])
        status = "parser_error" if failed == ["error"] else "unsupported_capability"
        detail = record.get("capture_observation", {}).get("unavailable", {}).get("code")
        detail = detail or record.get("capture_stimulus", {}).get("unavailable", {}).get("code")
        if detail == "source_unavailable":
            status = "source_build_blocker"
        return status, detail
    if mode == "stream":
        if "batch_final" in record:
            raise ValueError("executed stream lacks chunk emissions")
        validate_chunks(record.get("chunks"))
    elif mode == "batch":
        if "assembled" not in record or "chunks" in record or not typed_equal(record["assembled"], batch_final_events(record.get("batch_final", {}))):
            raise ValueError("executed batch differs from final fields")
    else:
        raise ValueError("unknown execution mode")
    if "assembled" in record:
        validate_events(record["assembled"])
    return "executed", None


def validate_chunks(chunks):
    if not isinstance(chunks, list):
        raise ValueError("executed stream lacks chunk emissions")
    for row in chunks:
        if isinstance(row, dict) and set(row) == {"expected"}:
            validate_events(row["expected"])
        elif isinstance(row, list):
            validate_events(row)
        else:
            raise ValueError("executed stream lacks chunk emissions")


def validate_events(events):
    if not isinstance(events, list):
        raise ValueError("capture events must be a list")
    for event in events:
        if not isinstance(event, dict) or event.get("kind") not in {"text", "reasoning", "tool_call"}:
            raise ValueError("capture event has an invalid kind")
        if event["kind"] in {"text", "reasoning"} and not isinstance(event.get("text"), str):
            raise ValueError("capture text event needs string text")
        if event["kind"] == "tool_call" and event.get("name") is not None and not isinstance(event["name"], str):
            raise ValueError("capture tool name must be a string or null")
