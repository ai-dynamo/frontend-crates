# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Serve one native publication shared by browser workers."""
import fcntl
import gzip
import json
import os
import sys
import threading
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import pytest

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "conformance/utils/src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))
from native_publication import publish

import unified_history  # noqa: E402


@pytest.fixture(scope="session")
def source_catalog():
    """Declare checkout catalog dependencies, including JavaScript subprocesses."""
    return json.loads((REPO / "conformance/viewer-catalog.json").read_bytes())


@pytest.fixture(scope="session")
def published_viewer(tmp_path_factory, source_catalog):
    configured = os.environ.get("CONFORMANCE_PUBLISHED_SITE")
    expected = source_catalog["revision"]
    if configured:
        site = Path(configured).resolve(strict=True)
    else:
        base = tmp_path_factory.getbasetemp()
        if os.environ.get("PYTEST_XDIST_WORKER"):
            base = base.parent
        parent = base / "native-browser-publication"
        parent.mkdir(exist_ok=True)
        site = parent / "published"
        with (parent / ".native-build.lock").open("a+b") as lease:
            fcntl.flock(lease.fileno(), fcntl.LOCK_EX)
            catalog_path = site / "conformance/viewer-resolved-catalog.json.gz"
            existing = json.loads(gzip.decompress(catalog_path.read_bytes())) if catalog_path.is_file() else {}
            if existing.get("source_revision") != expected or existing.get("transport", {}).get("schema") != 2:
                publish(REPO, site)
    catalog = json.loads(gzip.decompress((site / "conformance/viewer-resolved-catalog.json.gz").read_bytes()))
    assert catalog.get("source_revision") == expected
    assert catalog["transport"]["schema"] == 2
    assert catalog["transport"]["raw_buckets"]
    assert not (site / "conformance/viewer-original-resources.json.gz").exists()
    return site


@pytest.fixture(scope="session")
def viewer_url(published_viewer):
    prefix = "/frontend-crates"

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(published_viewer), **kwargs)

        def translate_path(self, path):
            path = urlsplit(path).path
            if not path.startswith(prefix + "/"):
                return str(published_viewer / "__outside_prefix__")
            return super().translate_path(path[len(prefix):])

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}{prefix}/index.html"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def pytest_addoption(parser):
    parser.addoption("--conformance-scope", choices=("fixtures", "publication"))
    parser.addoption("--conformance-group", help="disjoint sorted test assignment I/N, zero-based")


def needs_publication(item):
    # Closure names include indirect dependencies (e.g. a browser driver).
    return (bool({"published_viewer", "source_catalog"}.intersection(item.fixturenames))
            or item.path.name in {"test_viewer_catalog.py", "test_unified_regeneration.py"}
            or item.name in {"test_reference_backfills_require_request_bound_version_receipts",
                             "test_retained_unified_captures_cover_the_current_corpus"})


@pytest.hookimpl(hookwrapper=True)
def pytest_collection_modifyitems(config, items):
    yield
    scope = config.getoption("--conformance-scope")
    group = config.getoption("--conformance-group")
    eligible = [item for item in items if not scope or needs_publication(item) == (scope == "publication")]
    selected = eligible
    if group:
        index, count = map(int, group.split("/"))
        if not 0 <= index < count:
            raise pytest.UsageError("invalid conformance group")
        selected = [item for ordinal, item in enumerate(sorted(eligible, key=lambda item: item.nodeid))
                    if ordinal % count == index]
    chosen = set(selected)
    config.hook.pytest_deselected(items=[item for item in items if item not in chosen])
    items[:] = selected
    directory = os.environ.get("CONFORMANCE_SELECTION_DIR")
    if directory and os.environ.get("PYTEST_XDIST_WORKER", "gw0") == "gw0":
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        document = {"scope": scope, "group": group, "job": os.environ.get("GITHUB_JOB"),
                    "eligible": sorted(item.nodeid for item in eligible),
                    "selected": sorted(item.nodeid for item in selected)}
        (destination / f"{uuid.uuid4().hex}.json").write_text(json.dumps(document, sort_keys=True))
    reporter = config.pluginmanager.get_plugin("terminalreporter")
    if reporter:
        reporter.write_line(f"conformance selection: {len(selected)}/{len(eligible)} eligible tests; scope={scope}; group={group}")
