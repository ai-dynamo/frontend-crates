# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Latest snapshots coexist with unchanged, explicitly selected historical evidence."""
import json
import shutil

import pytest

import capture_sources as sources
import ci_publication_cache
import publish_current_capture
from test_unified_history import _store


def retained_root(tmp_path):
    root = tmp_path / "checkout"
    shutil.copytree(_store(tmp_path / "sparse"), root / sources.STORE)
    (root / "conformance/fixtures-manifest.json").write_bytes(b'{"shards":[]}\n')
    return root


def test_retained_history_is_staged_separately_and_rechecked(tmp_path):
    root = retained_root(tmp_path)
    latest = root / sources.latest_path("unified", "gemma4", "dynamo_v2")
    latest.write_bytes(b"latest snapshot is not historical evidence\n")
    pin = sources.history_pin(root)
    evidence = sources.verify(root, pin)
    assert sources.default_history(root) == root
    assert pin["schema"] == 2
    assert latest.relative_to(root).as_posix() not in {row["path"] for row in evidence["files"]}
    destination = tmp_path / "prepared"
    destination.mkdir()
    manifest = sources.stage_history(root, destination, root)
    assert manifest == {"shards": []}
    dataset = json.loads((destination / sources.DATASETS).read_bytes())[0]
    assert not (destination / sources.STORE).exists()
    for row in evidence["files"]:
        if row["path"].startswith(sources.STORE + "/"):
            assert (destination / dataset["root"] / row["path"]).read_bytes() == (root / row["path"]).read_bytes()
    with sources.history_view(root, root) as view:
        assert not (view / sources.DATASETS).exists()
        for row in evidence["files"]:
            assert (view / row["path"]).read_bytes() == (root / row["path"]).read_bytes()
    historical = next((root / sources.STORE / "families").glob("*/*-*.yaml"))
    historical.write_bytes(historical.read_bytes() + b"# changed historical evidence\n")
    with pytest.raises(ValueError, match="retained history changed"):
        sources.verify(root, pin)


def test_retained_archival_fails_before_mutation(tmp_path, monkeypatch):
    root = retained_root(tmp_path)
    monkeypatch.setattr(publish_current_capture, "atomic_write", lambda *_args, **_kwargs: pytest.fail("unexpected archival write"))
    with pytest.raises(ValueError, match="read-only.*deferred"):
        publish_current_capture.publish(root, root, tmp_path / "not-read.yaml", "gemma4", "dynamo_v2", "0.5.3")
    assert not (root / "additions").exists()
    assert not sources.pending_updates(root)


def test_latest_publication_cache_excludes_in_place_history():
    prefix = sources.STORE + "/families/gemma4/"
    assert ci_publication_cache.is_publication_input(prefix + "inputs_and_golden.yaml")
    assert ci_publication_cache.is_publication_input(prefix + "dynamo_v2.yaml")
    assert not ci_publication_cache.is_publication_input(prefix + "dynamo_v2-0.5.2.yaml")
    assert not ci_publication_cache.is_publication_input("conformance/fixtures/toolcalling/fixtures-stream-v1/dynamo_v2-0.5.2.tar.gz")
