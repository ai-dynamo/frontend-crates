# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Determinism, failure propagation, and generation ownership across process workers."""
import json
import multiprocessing
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import generation_workers
import unified_history
import viewer_catalog
from fixture_test_support import native_snapshots, snapshot_bytes
from test_unified_history import _store
from viewer_codec import canonical_typed


def _job(value):
    if value == "fail":
        raise ValueError("worker rejected input")
    return value, generation_workers.worker_count()


def _blocking_job(item):
    value, release = item
    if value == 0:
        assert release.wait(5), "completed work was blocked behind the first submitted job"
    return value


@pytest.fixture
def tiny_worker_admission(monkeypatch):
    # Pool semantics need their requested participants even on a constrained host;
    # the real CPU, cgroup, and memory admission policy has its own tests.
    def admitted_count(jobs=None):
        if multiprocessing.current_process().name != "MainProcess":
            return 1
        requested = generation_workers._requested.get()
        count = 2 if requested == "auto" else int(requested)
        return max(1, min(count, jobs or count))

    monkeypatch.setattr(generation_workers, "worker_count", admitted_count)


def test_completion_drain_releases_fast_jobs_without_submission_order_blocking(tiny_worker_admission):
    with multiprocessing.Manager() as manager, generation_workers.worker_policy(2):
        release = manager.Event()
        results = generation_workers.bounded_map(_blocking_job, [(0, release), (1, release)], phase="completion-test", ordered=False)
        try:
            assert next(results) == 1
            release.set()
            assert list(results) == [0]
        finally:
            release.set()
            results.close()


def test_worker_admission_honors_affinity_memory_and_nested_process(monkeypatch):
    with generation_workers.worker_policy("auto"):
        monkeypatch.setattr(generation_workers.os, "sched_getaffinity", lambda pid: {0})
        assert generation_workers.worker_count() == 1
        monkeypatch.setattr(generation_workers.os, "sched_getaffinity", lambda pid: set(range(32)))
        monkeypatch.setattr(generation_workers, "_available_memory", lambda: 1)
        assert generation_workers.worker_count() == 1
        monkeypatch.setattr(generation_workers.multiprocessing, "current_process", lambda: SimpleNamespace(name="worker"))
        assert generation_workers.worker_count() == 1


def test_final_serial_job_cannot_escape_memory_guard(monkeypatch):
    def crosses_limit(value):
        monkeypatch.setattr(generation_workers, "MEMORY_LIMIT", 1)
        return value

    with generation_workers.worker_policy(1), pytest.raises(MemoryError, match="64 GiB"):
        list(generation_workers.bounded_map(crosses_limit, [1], phase="memory-test"))


def test_pool_orders_results_prevents_nested_pools_and_propagates_failure(tiny_worker_admission):
    with generation_workers.worker_policy(2):
        assert list(generation_workers.bounded_map(_job, [3, 1, 2], phase="test")) == [(3, 1), (1, 1), (2, 1)]
        with pytest.raises(ValueError, match="worker rejected input"):
            list(generation_workers.bounded_map(_job, [3, "fail", 2], phase="test"))


def test_serial_parallel_store_and_bundle_bytes(tmp_path, tiny_worker_admission):
    root = tmp_path / "source"
    store_root = _store(root / "conformance/fixtures-unified-v2")
    directory = store_root / "families/gemma4"
    sibling = store_root / "families/another"
    sibling.mkdir()
    for path in directory.iterdir():
        (sibling / path.name).write_text(path.read_text().replace("gemma4", "another"))
    catalog = {"schema": 1, "revision": "a" * 64, "sources": {"archives": [],
               "unified": [{"path": "source.yaml", "size": 1, "sha256": "b" * 64}],
               "candidates": [{"mode": "unified", "implementation": "dynamo_v2", "version": version}
                              for version in ("0.5.0", "0.5.2", "0.5.3")]},
               "metadata": {"unified_capture_policy": {"dynamo_v2": {"mode": "stream"}}}}
    published, stores = [], []
    for workers in (1, 2):
        with generation_workers.worker_policy(workers):
            store = unified_history.load_store(store_root)
            for (family, implementation), history in store.histories.items():
                assert history.family is store.families[family]
                store.families[family].document["input_document"]["ownership_probe"] = workers
                assert history.family.document["input_document"]["ownership_probe"] == workers
                del store.families[family].document["input_document"]["ownership_probe"]
            stores.append({key: [canonical_typed(value) for _, value in history.iter_resolved_snapshots()]
                           for key, history in store.histories.items()})
            published.append(native_snapshots(root, tmp_path / str(workers), catalog, workers=workers))
    assert stores[0] == stores[1]
    assert json.dumps(published[0]) == json.dumps(published[1])
    for descriptor in published[0]["sources"]["unified"]:
        assert snapshot_bytes(tmp_path / "1", descriptor) == snapshot_bytes(tmp_path / "2", descriptor)
    archive = store_root / "versions/dynamo-0.5.2.tar.gz"
    archive.write_bytes(archive.read_bytes()[:-8])
    with generation_workers.worker_policy(2), pytest.raises(ValueError, match="truncated"):
        unified_history.load_store(store_root)


@pytest.mark.parametrize("mutation", ["content", "add", "remove", "rename", "non_yaml", "directory", "symlink"])
def test_validated_generation_rejects_changed_inventory(tmp_path, mutation):
    root = _store(tmp_path / "conformance/fixtures-unified-v2")
    with viewer_catalog.catalog_generation(tmp_path) as generation:
        store = generation.validated_store(root)
        assert generation.validated_store(root) is store
        path = root / "families/gemma4/dynamo_v2-0.5.2.yaml"
        if mutation == "content":
            path.write_bytes(path.read_bytes() + b"\n")
        elif mutation == "add":
            (path.parent / "extra.yaml").write_bytes(b"{}")
        elif mutation == "remove":
            path.unlink()
        elif mutation == "rename":
            path.rename(path.with_name("dynamo_v2-0.5.3.yaml"))
        elif mutation == "non_yaml":
            (path.parent / "unexpected.bin").write_bytes(b"unexpected")
        elif mutation == "directory":
            (root / "unvalidated-entry").mkdir()
        else:
            (path.parent / "linked").symlink_to(path)
        with pytest.raises(ValueError, match="inventory changed"):
            generation.validated_store(root)
    with pytest.raises(ValueError, match="inactive"):
        generation.validated_store(root)


def test_generation_rejects_wrong_thread_and_root_and_independent_read_revalidates(tmp_path):
    root = _store(tmp_path / "conformance/fixtures-unified-v2")
    with viewer_catalog.catalog_generation(tmp_path) as generation:
        generation.validated_store(root)
        with pytest.raises(ValueError, match="another root"):
            generation.validated_store(tmp_path.parent / "outside")
        with ThreadPoolExecutor(1) as pool, pytest.raises(ValueError, match="another thread"):
            pool.submit(generation.validated_store, root).result()
        # Independent callers must still parse and validate, even during a generation.
        path = root / "families/gemma4/dynamo_v2-0.5.2.yaml"
        path.write_text("bad: [")
        with pytest.raises(ValueError):
            unified_history._load_store_unlocked(root)
