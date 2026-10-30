# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One bounded process policy for fixture generations; commits stay in the caller."""
import argparse
import multiprocessing
import os
import resource
import sys
import threading
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path


GIB = 1024 ** 3
MEMORY_LIMIT = 64 * GIB
_requested = ContextVar("generation_workers", default="1")


def worker_argument(value):
    if value != "auto" and (not value.isdecimal() or int(value) < 1):
        raise argparse.ArgumentTypeError("workers must be auto or a positive integer")
    return value


def add_worker_argument(parser):
    parser.add_argument("--workers", type=worker_argument, default="auto", metavar="auto|N",
                        help="bounded process workers (auto caps at eight; default: auto)")


@contextmanager
def worker_policy(value):
    token = _requested.set(worker_argument(str(value)))
    try:
        yield
    finally:
        _requested.reset(token)


def requested_workers():
    """Pass the requested policy to native consumers that own their admission budget."""
    return _requested.get()


def _available_memory():
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for row in meminfo.read_text().splitlines():
            if row.startswith("MemAvailable:"):
                return int(row.split()[1]) * 1024
    return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")


def worker_count(jobs=None):
    if multiprocessing.current_process().name != "MainProcess":
        return 1
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    memory = min(MEMORY_LIMIT, _available_memory())
    # Walk the active cgroup and ancestors: a parent can impose the tighter limit.
    cgroup = Path("/proc/self/cgroup")
    if cgroup.is_file():
        for row in cgroup.read_text().splitlines():
            if row.startswith("0::"):
                group = Path("/sys/fs/cgroup") / row[3:].lstrip("/")
                for directory in (group, *group.parents):
                    if not directory.is_relative_to("/sys/fs/cgroup"):
                        break
                    maximum, current, quota = directory / "memory.max", directory / "memory.current", directory / "cpu.max"
                    if maximum.is_file() and current.is_file() and maximum.read_text().strip() != "max":
                        memory = min(memory, max(0, int(maximum.read_text()) - int(current.read_text())))
                    if quota.is_file():
                        limit, period = quota.read_text().split()
                        if limit != "max":
                            cpus = min(cpus, max(1, int(limit) // int(period)))
    requested = _requested.get()
    count = 8 if requested == "auto" else int(requested)
    return max(1, min(count, cpus, max(1, (memory - 8 * GIB) // (3 * GIB)), jobs or count))


def _run_job(function, item):
    started = time.process_time()
    result = function(item)
    return result, time.process_time() - started


def _tree_rss(pid):
    """Sum simultaneous resident bytes, conservatively counting shared pages."""
    try:
        root = Path(f"/proc/{pid}")
        rss = int((root / "statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        children = set()
        for task in (root / "task").iterdir():
            try:
                children.update((task / "children").read_text().split())
            except FileNotFoundError:
                continue
        return rss + sum(_tree_rss(int(child)) for child in children)
    except (FileNotFoundError, ProcessLookupError):
        return 0


def bounded_map(function, items, *, phase, jobs=None, ordered=True):
    """Bound owned arguments; callers with deterministic merges can drain by completion."""
    count = worker_count(jobs)
    started, completed, cpu, peak = time.monotonic(), 0, 0.0, 0
    stop = threading.Event()
    over_limit = threading.Event()

    def sample():
        nonlocal peak
        while not stop.is_set():
            measured = _tree_rss(os.getpid()) if Path("/proc/self/statm").is_file() else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            peak = max(peak, measured)
            if measured > MEMORY_LIMIT:
                over_limit.set()
            stop.wait(0.25)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    try:
        if count == 1:
            for item in items:
                if over_limit.is_set():
                    raise MemoryError("generation exceeded 64 GiB aggregate RSS")
                result, usage = _run_job(function, item)
                peak = max(peak, _tree_rss(os.getpid()))
                if over_limit.is_set() or peak > MEMORY_LIMIT:
                    raise MemoryError("generation exceeded 64 GiB aggregate RSS")
                completed += 1
                cpu += usage
                yield result
        else:
            # Spawn avoids inheriting flock descriptors or the coordinator's store.
            with ProcessPoolExecutor(count, mp_context=multiprocessing.get_context("spawn")) as pool:
                pending = deque()
                iterator = iter(items)
                try:
                    exhausted = False
                    while pending or not exhausted:
                        while not exhausted and len(pending) < count:
                            try:
                                item = next(iterator)
                            except StopIteration:
                                exhausted = True
                                break
                            pending.append(pool.submit(_run_job, function, item))
                            del item
                        if pending:
                            future = pending.popleft() if ordered else next(as_completed(pending))
                            if not ordered:
                                pending.remove(future)
                            result, usage = future.result()
                            if over_limit.is_set():
                                raise MemoryError("generation exceeded 64 GiB aggregate RSS")
                            completed += 1
                            cpu += usage
                            if completed % count == 0:
                                print(f"{phase}: completed={completed} workers={count} seconds={time.monotonic()-started:.2f} worker_cpu_seconds={cpu:.2f} aggregate_peak_rss_bytes={peak}", file=sys.stderr, flush=True)
                            yield result
                finally:
                    for future in pending:
                        future.cancel()
    finally:
        stop.set()
        sampler.join()
        print(f"{phase}: completed={completed} workers={count} seconds={time.monotonic()-started:.2f} worker_cpu_seconds={cpu:.2f} aggregate_peak_rss_bytes={peak}", file=sys.stderr, flush=True)
