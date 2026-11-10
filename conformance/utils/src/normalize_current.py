#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Normalize active YAML atomically without rewriting historical evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import capture_sources as sources
import normalized_yaml as normalized
import unified_history as history
from viewer_codec import typed_equal


def normalize(root: Path, *, check: bool = False) -> dict:
    root = root.resolve()
    store = root / sources.STORE
    groups = {}
    for path in sources.latest_unified_paths(root):
        groups.setdefault(path.parent, []).append(path)
    before_bytes = after_bytes = records = 0
    pairs = []
    fingerprints = {}
    with history._store_mutation_lock(store), tempfile.TemporaryDirectory(prefix=".normalize-", dir=store.parent) as temporary:
        for directory, paths in sorted(groups.items()):
            authored_path = directory / history.FAMILY_DOCUMENT_NAME
            original = history._parse_yaml_raw(authored_path, authored_path.read_bytes())
            authored = normalized.expand_family(original)
            frozen = list(normalized.expand_requests(original).values()) if original.get("schema_version") == 3 else []
            captures = {}
            for path in sorted(paths):
                raw = history._parse_yaml_raw(path, path.read_bytes())
                capture = normalized.expand_snapshot(raw, original)
                captures[path] = capture
                records += len(capture["records"])
                frozen.extend(record["value"]["stimulus"]["inline"] for record in capture["records"].values()
                              if "inline" in record["value"]["stimulus"])
            encoded = normalized.encode_family(authored, frozen)
            if not typed_equal(normalized.expand_family(encoded)["cases"], authored["cases"]):
                raise ValueError("authored normalization changed a case")
            family = history._validate_family(authored_path, normalized.expand_family(encoded))
            encoded_documents = {authored_path: history.dump_yaml(encoded).encode()}
            for path, capture in captures.items():
                compact = normalized.encode_snapshot(capture, encoded)
                expanded = normalized.expand_snapshot(compact, encoded)
                if not typed_equal(capture, expanded):
                    raise ValueError(f"normalization changed a resolved capture: {path}")
                selection = compact.pop(sources.SELECTION)
                history._validate_complete_snapshot(path, {**expanded, sources.SELECTION: selection}, family,
                                                   selection["implementation"], selection["version"])
                body = history.dump_yaml(compact).encode()
                selection["payload_sha256"] = sources.digest(body)
                encoded_documents[path] = b"capture_selection: " + sources.encoded(selection).rstrip(b"\n") + b"\n" + body
            for path, data in encoded_documents.items():
                old = path.read_bytes()
                fingerprints[path] = hashlib.sha256(old).digest()
                before_bytes += len(old)
                after_bytes += len(data)
                if check:
                    if old != data:
                        raise ValueError(f"active YAML is not normalized: {path}")
                    continue
                target = Path(temporary) / str(len(pairs))
                target.write_bytes(data)
                pairs.append((target, path))
        for path, fingerprint in fingerprints.items():
            if hashlib.sha256(path.read_bytes()).digest() != fingerprint:
                raise ValueError(f"YAML changed during normalization: {path}")
        if pairs:
            history.publish_paths_transactionally(pairs, backup_parent=store.parent)
    return {"files": len(fingerprints), "records": records, "before_bytes": before_bytes, "after_bytes": after_bytes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    print(json.dumps(normalize(args.root, check=args.check)))


if __name__ == "__main__":
    main()
