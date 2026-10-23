#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independently compare a frozen publication with its chunked-v2 successor."""

import argparse
import base64
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import time
import zlib


MAX_PACK = 1024 * 1024
MAX_RESOURCE = 1024 * 1024 * 1024


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def identity(data, expected_hash, expected_size):
    require(len(data) == expected_size and digest(data) == expected_hash, "byte identity mismatch")
    return data


def read(root, relative):
    path = Path(relative)
    require(not path.is_absolute() and ".." not in path.parts, "unsafe resource path")
    resolved = (root / path).resolve(strict=True)
    require(resolved.is_relative_to(root), "resource escapes publication")
    return resolved.read_bytes()


def decompress(data, limit):
    require(0 <= limit <= MAX_RESOURCE, "invalid decompression bound")
    decoder = zlib.decompressobj(31)
    result = decoder.decompress(data, limit + 1)
    require(len(result) <= limit and decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail, "invalid or oversized gzip resource")
    return result


def compressed_json(root, path):
    return json.loads(decompress(read(root, path), MAX_RESOURCE))


def read_index(root, descriptor):
    encoded = identity(read(root, descriptor["path"]), descriptor["sha256"], descriptor["size"])
    decoded = decompress(encoded, descriptor["decoded_size"])
    identity(decoded, descriptor["decoded_sha256"], descriptor["decoded_size"])
    value = json.loads(decoded)
    require(value["schema"] == 2, "incorrect index schema")
    return value["resources"]


class Decoder:
    """Bound shared decoded-pack caching to 64 MiB; never trust a cache hit's metadata."""

    def __init__(self, root):
        self.root = root
        self.cache = OrderedDict()
        self.cached_bytes = 0
        self.pack_reads = 0

    def pack(self, descriptor):
        path = descriptor["path"]
        require(path == f"conformance/viewer-data/packs/{descriptor['sha256']}.bin.gz", "invalid pack path")
        size = descriptor["decoded_size"]
        require(type(size) is int and 0 < size <= MAX_PACK, "invalid decoded pack size")
        if path in self.cache:
            original_descriptor, decoded = self.cache.pop(path)
            require(original_descriptor == descriptor, "conflicting cached pack identity")
        else:
            encoded = identity(read(self.root, path), descriptor["sha256"], descriptor["size"])
            decoded = decompress(encoded, size)
            identity(decoded, descriptor["decoded_sha256"], size)
            self.cached_bytes += len(decoded)
            self.pack_reads += 1
        self.cache[path] = (descriptor, decoded)
        while self.cached_bytes > 64 * MAX_PACK:
            _, (_, evicted) = self.cache.popitem(last=False)
            self.cached_bytes -= len(evicted)
        return decoded

    def reconstruct(self, recipe):
        require(recipe["schema"] == 2, "incorrect recipe schema")
        require(type(recipe["size"]) is int and 0 <= recipe["size"] < MAX_RESOURCE, "invalid logical size")
        result = bytearray()
        for chunk in recipe["chunks"]:
            pack = self.pack(chunk["pack"])
            offset, length = chunk["offset"], chunk["size"]
            require(type(offset) is int and type(length) is int and offset >= 0 and 0 < length <= 128 * 1024 and offset + length <= len(pack), "invalid chunk range")
            data = identity(pack[offset:offset + length], chunk["sha256"], length)
            require(len(result) + length <= recipe["size"], "logical resource exceeds declared size")
            result.extend(data)
        return identity(bytes(result), recipe["sha256"], recipe["size"])


def case_count(envelope):
    value = envelope["value"]
    require(value[0] == "object", "authored family must be an object")
    cases = next(value for key, value in value[1] if key == ["str", "cases"])
    require(cases[0] == "object", "authored cases must be an object")
    active = 0
    for _, case in cases[1]:
        lifecycle = next(value for key, value in case[1] if key == ["str", "lifecycle"])
        active += lifecycle != ["str", "retired"]
    return len(cases[1]), active


def compare(baseline, candidate):
    started = time.monotonic()
    before = compressed_json(baseline, "conformance/viewer-resolved-catalog.json.gz")
    after = compressed_json(candidate, "conformance/viewer-resolved-catalog.json.gz")
    require(after["transport"]["schema"] == 2, "candidate is not transport v2")
    before_resources = {r["path"]: r for r in before["sources"]["unified"] if not (r["kind"] == "inputs-and-golden" and isinstance(r.get("families"), list))}
    after_resources = {r["path"]: r for r in after["sources"]["unified"]}
    require(before_resources.keys() == after_resources.keys(), "logical capture/input selections differ")
    decoder = Decoder(candidate)
    baseline_decoder = Decoder(baseline)
    packed_baseline = before.get("transport", {}).get("schema") == 2
    counts = {"capture_selections": 0, "family_inputs": 0, "authored_unified_cases_total": 0, "authored_unified_cases_active": 0, "original_resources": 0, "legacy_raw_aliases": 0, "logical_bytes": 0, "original_bytes": 0}
    shared_inputs = None
    for path, original in before_resources.items():
        current = after_resources[path]
        for field in ("kind", "family", "families", "implementation", "version", "format", "record_contract", "capture_identity", "source_fingerprint", "resolved_metadata_sha256"):
            require(original.get(field) == current.get(field), f"descriptor metadata differs: {field}: {path}")
        if packed_baseline:
            old_bytes = baseline_decoder.reconstruct(read_index(baseline, original["recipe_index"])[path])
            identity(old_bytes, original["sha256"], original["size"])
        else:
            encoded = identity(read(baseline, path), original["sha256"], original["size"])
            old_bytes = decompress(encoded, MAX_RESOURCE)
        if original["kind"] == "inputs-and-golden":
            if shared_inputs is None:
                shared_inputs = read_index(candidate, current["recipe_index"])
            recipe = shared_inputs[path]
            counts["family_inputs"] += 1
            total, active = case_count(json.loads(old_bytes))
            counts["authored_unified_cases_total"] += total
            counts["authored_unified_cases_active"] += active
        else:
            recipe = read_index(candidate, current["recipe_index"])[path]
            counts["capture_selections"] += 1
        new_bytes = decoder.reconstruct(recipe)
        identity(new_bytes, current["sha256"], current["size"])
        require(old_bytes == new_bytes, f"decoded payload differs: {path}")
        counts["logical_bytes"] += len(new_bytes)
    if packed_baseline:
        baseline_raw = {}
        for descriptor in before["transport"]["raw_buckets"].values():
            baseline_raw.update(read_index(baseline, descriptor))
        original_index = {path: entry["resource"] for path, entry in baseline_raw.items() if "resource" in entry}
    else:
        original_index = compressed_json(baseline, "conformance/viewer-original-resources.json.gz")["resources"]
    seen = set()
    for bucket, descriptor in after["transport"]["raw_buckets"].items():
        for path, entry in read_index(candidate, descriptor).items():
            require(digest(path.encode())[:2] == bucket, "raw path in incorrect bucket")
            if "resource" not in entry:
                require(entry["path"] in original_index and isinstance(entry["member"], str), "unresolved legacy alias")
                if packed_baseline:
                    require(entry == baseline_raw.get(path), f"legacy alias differs: {path}")
                counts["legacy_raw_aliases"] += 1
                continue
            require(path in original_index and path not in seen, f"unexpected/duplicate original: {path}")
            seen.add(path)
            old = original_index[path]
            resource = entry["resource"]
            if packed_baseline:
                old_bytes = baseline_decoder.reconstruct(old["recipe"])
                require(all(resource[key] == old[key] for key in ("path", "sha256", "size")), "original metadata differs")
            else:
                # Historical envelopes remain supported only by this independent verifier.
                envelope_path = f"conformance/viewer-data/originals/{old['source_sha256']}.json.gz"
                envelope_bytes = identity(read(baseline, envelope_path), old["sha256"], old["size"])
                envelope = json.loads(decompress(envelope_bytes, MAX_RESOURCE))
                require(envelope["schema"] == 1 and envelope["encoding"] == "base64", "invalid baseline envelope")
                old_bytes = identity(base64.b64decode(envelope["bytes"], validate=True), old["source_sha256"], old["source_size"])
                require(resource["path"] == path and resource["sha256"] == old["source_sha256"] and resource["size"] == old["source_size"], "original metadata differs")
            new_bytes = decoder.reconstruct(resource["recipe"])
            require(old_bytes == new_bytes, f"original bytes differ: {path}")
            counts["original_resources"] += 1
            counts["original_bytes"] += len(new_bytes)
    require(seen == original_index.keys(), "candidate omits originals")
    if packed_baseline:
        require(counts["legacy_raw_aliases"] == len(baseline_raw) - len(original_index), "candidate omits Raw aliases")
        require(before["metadata"] == after["metadata"], "report metadata differs")
        require(before["sources"]["candidates"] == after["sources"]["candidates"], "selection identities differ")
        require(before["sources"]["original_unified"] == after["sources"]["original_unified"], "provenance differs")
    return {"schema": 1, "baseline": str(baseline), "candidate": str(candidate), "equal": True, **counts, "pack_reads": decoder.pack_reads, "pack_cache_limit_bytes": 64 * MAX_PACK, "wall_seconds": time.monotonic() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="JSON evidence outside either publication")
    parser.add_argument("--baseline-counts", type=Path, help="Frozen cross-suite count evidence; preserved by exact equality of every authored resource")
    args = parser.parse_args()
    baseline, candidate, output = args.baseline.resolve(strict=True), args.candidate.resolve(strict=True), args.output.resolve()
    require(not output.is_relative_to(baseline) and not output.is_relative_to(candidate), "evidence must remain outside publication")
    result = compare(baseline, candidate)
    if args.baseline_counts:
        count_bytes = args.baseline_counts.read_bytes()
        count_evidence = json.loads(count_bytes)
        unified_count = next(tab["parser_case_pairs"] for tab in count_evidence["tabs"] if tab["tab"] == "tab-unified")
        require(unified_count == result["authored_unified_cases_active"], "frozen Unified count differs")
        result["preserved_authored_parser_case_pairs"] = count_evidence["parser_case_pairs"]
        result["count_evidence"] = {"path": str(args.baseline_counts.resolve()), "sha256": digest(count_bytes), "size": len(count_bytes), "basis": "Exact equality of all family inputs and all original archives; active Unified cases recounted independently"}
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
