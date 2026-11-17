#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preserve a recovery store before applying v1 archive retention."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

import capture_sources
import reconcile_dynamo_captures
import unified_history
import v1_capture_retention


def migrate(root: Path, work: Path, *, publish=False):
    root = root.resolve()
    work.mkdir(parents=True, exist_ok=False)
    live = root / capture_sources.STORE
    expected = unified_history.store_inventory(live)
    archive = work / "original-store.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        output.add(live, arcname=capture_sources.STORE)
    if unified_history.store_inventory(live) != expected:
        raise ValueError("capture store changed while preserving recovery evidence")
    archive.chmod(0o400)
    candidate = work / "candidate" / capture_sources.STORE
    shutil.copytree(live, candidate)
    current = {}
    for path in candidate.glob("families/*/*.yaml"):
        if path.name != unified_history.FAMILY_DOCUMENT_NAME and "-" not in path.stem:
            relative = path.relative_to(candidate).as_posix()
            current[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
            path.unlink()
    inventory_path = root / "conformance/dynamo-capture-inventory.yaml"
    latest = None
    if inventory_path.is_file():
        selected = v1_capture_retention.selected_targets(unified_history.load_yaml(inventory_path)["targets"])
        latest = next((target["version"] for target in selected if target["implementation"] == "dynamo_v1"), None)
    result = v1_capture_retention.retain(candidate, latest=latest)
    proof = {"original_inventory": expected, "current_sha256": current,
             "original_archive": str(archive), "original_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
             "result": result, "candidate_inventory": unified_history.store_inventory(candidate)}
    (work / "proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    if publish:
        reconcile_dynamo_captures.publish_history_store(root, candidate, expected)
        for name, digest in current.items():
            if hashlib.sha256((live / name).read_bytes()).hexdigest() != digest:
                raise ValueError("v1 migration changed current capture bytes")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    print(json.dumps(migrate(args.root, args.work, publish=args.publish), sort_keys=True))


if __name__ == "__main__":
    main()
