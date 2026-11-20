# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Renderer evidence participates in the independent publication byte verifier."""

import copy
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from verify_publication_transport import compare


def encoded(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode()


def identity(path: str, data: bytes) -> dict[str, Any]:
    return {"path": path, "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}


def publication(root: Path, result: dict[str, Any], *, include_renderer: bool = True) -> None:
    def put(path: str, data: bytes) -> None:
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    catalog: dict[str, Any] = {
        "schema": 1, "metadata": {"renderer": {"schema": 1}} if include_renderer else {},
        "sources": {"unified": [], "candidates": [], "original_unified": []},
        "transport": {"schema": 2, "raw_buckets": {}},
    }
    if include_renderer:
        payload = encoded({"schema": 1, "kind": "renderer-suite", "renderer": result})
        pack_bytes = gzip.compress(payload, mtime=0)
        pack_path = f"conformance/viewer-data/packs/{hashlib.sha256(pack_bytes).hexdigest()}.bin.gz"
        pack = {**identity(pack_path, pack_bytes), "decoded_sha256": hashlib.sha256(payload).hexdigest(), "decoded_size": len(payload)}
        put(pack_path, pack_bytes)
        resource = {**identity("renderer-suite.json", payload), "kind": "renderer-suite", "format": "typed-json-v1", "transport": "chunked-v2"}
        recipe = {"schema": 2, "sha256": resource["sha256"], "size": len(payload),
                  "chunks": [{"offset": 0, "size": len(payload), "sha256": resource["sha256"], "pack": pack}]}
        index = encoded({"schema": 2, "resources": {resource["path"]: recipe}})
        index_bytes = gzip.compress(index, mtime=0)
        index_path = "conformance/viewer-data/indexes/renderer.json.gz"
        put(index_path, index_bytes)
        resource["recipe_index"] = {**identity(index_path, index_bytes), "decoded_sha256": hashlib.sha256(index).hexdigest(), "decoded_size": len(index)}
        catalog["sources"]["renderer"] = [resource]
    put("conformance/viewer-resolved-catalog.json.gz", gzip.compress(encoded(catalog), mtime=0))


PROMPT = {"text": "Hello 🦀\n", "segments": [{"text": "Hello 🦀\n", "allow_special": False}],
          "pending_segments": 0, "media_message_order": [0]}


def test_renderer_payload_is_compared_and_parser_only_catalogs_remain_supported(tmp_path: Path) -> None:
    for include in (False, True):
        baseline, candidate = tmp_path / f"before-{include}", tmp_path / f"after-{include}"
        publication(baseline, PROMPT, include_renderer=include)
        publication(candidate, PROMPT, include_renderer=include)
        result = compare(baseline, candidate)
        assert result["equal"] is True
        assert result["renderer_suites"] == int(include)


@pytest.mark.parametrize("field,value", [
    ("text", "Hello\n"), ("segments", [{"text": "Hello 🦀\n", "allow_special": True}]),
    ("pending_segments", 1), ("media_message_order", [1, 0]),
])
def test_renderer_transport_verifier_rejects_changed_text_or_metadata(tmp_path: Path, field: str, value: Any) -> None:
    baseline, candidate = tmp_path / "before", tmp_path / "after"
    changed = copy.deepcopy(PROMPT)
    changed[field] = value
    publication(baseline, PROMPT)
    publication(candidate, changed)
    with pytest.raises(ValueError, match="decoded payload differs"):
        compare(baseline, candidate)


def test_renderer_transport_verifier_rejects_omitted_suite(tmp_path: Path) -> None:
    baseline, candidate = tmp_path / "before", tmp_path / "after"
    publication(baseline, PROMPT)
    publication(candidate, PROMPT, include_renderer=False)
    with pytest.raises(ValueError, match="logical capture/input selections differ"):
        compare(baseline, candidate)
