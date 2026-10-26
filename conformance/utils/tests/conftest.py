# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Serve one native publication shared by browser workers."""
import fcntl
import json
import os
import shutil
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
from fixture_test_support import publish
from verify_publication_transport import materialize_embedded, publication_metadata, runtime_catalog

import unified_history  # noqa: E402
import unified_archives
import capture_sources


@pytest.fixture(scope="session")
def history_source():
    configured = os.environ.get("CONFORMANCE_HISTORY_SOURCE", str(capture_sources.default_history(REPO)))
    source = Path(configured).resolve(strict=True)
    pin = capture_sources.history_pin(REPO, source)
    capture_sources.verify(source, pin)
    return source


@pytest.fixture(scope="session")
def historical_unified_sources(history_source):
    with capture_sources.history_view(REPO, history_source) as view:
        root = view / capture_sources.STORE
        if (view / capture_sources.DATASETS).is_file():
            resources = json.loads((view / "conformance/viewer-catalog.json").read_bytes())["sources"]["unified"]
        else:
            resources = [{**row, **unified_history.source_identity(row["path"])}
                         for row in unified_archives.inventory(root)]
        yield root, resources


@pytest.fixture(scope="session")
def source_catalog():
    """Declare checkout catalog dependencies, including JavaScript subprocesses."""
    return json.loads((REPO / "conformance/viewer-catalog.json").read_bytes())


def _browser_publication(tmp_path_factory, history_source=None):
    include_history = history_source is not None
    mode = "history" if include_history else "latest"
    configured = os.environ.get("CONFORMANCE_HISTORY_SITE" if include_history else "CONFORMANCE_PUBLISHED_SITE")
    if include_history and not configured and os.environ.get("CONFORMANCE_PUBLISHED_SITE"):
        restored = Path(os.environ["CONFORMANCE_PUBLISHED_SITE"]).resolve(strict=True)
        catalog = runtime_catalog(restored)
        if catalog["capture_generation"]["mode"] == "history":
            configured = str(restored)
    if configured:
        return Path(configured).resolve(strict=True)
    base = tmp_path_factory.getbasetemp()
    if os.environ.get("PYTEST_XDIST_WORKER"):
        base = base.parent
    parent = base / "native-browser-publication" / mode
    parent.mkdir(parents=True, exist_ok=True)
    site = parent / "published"
    expected = capture_sources.publication_inputs_digest(REPO, include_history=include_history, history_source=history_source)
    with (parent / ".native-build.lock").open("a+b") as lease:
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX)
        receipt = parent / "publication-inputs.json"
        existing = json.loads(receipt.read_bytes()) if receipt.is_file() else {}
        if existing.get("publisher_inputs_sha256") != expected or not (site / "index.html").is_file():
            extra = ("--include-history", "--history-source", str(history_source)) if include_history else ()
            publish(REPO, site, extra=extra)
            receipt.write_text(json.dumps({"publisher_inputs_sha256": expected}))
    return site


@pytest.fixture(scope="session")
def published_viewer(tmp_path_factory, source_catalog):
    site = _browser_publication(tmp_path_factory)
    catalog = runtime_catalog(site)
    assert catalog.get("source_revision") == source_catalog["revision"]
    assert catalog["transport"]["schema"] == 2
    assert catalog["transport"]["raw_buckets"]
    assert not (site / "conformance/viewer-original-resources.json.gz").exists()
    return site


@pytest.fixture(scope="session")
def transport_published_viewer(published_viewer, tmp_path_factory):
    """Fault injection uses a private transport projection, never the published artifact."""
    if (published_viewer / "conformance/viewer-resolved-catalog.json.gz").is_file():
        return published_viewer
    return materialize_embedded(published_viewer, tmp_path_factory.mktemp("transport-parent") / "site")


@pytest.fixture(scope="session")
def historical_published_viewer(history_source, tmp_path_factory):
    """Frozen release journeys use verified bundled evidence by default."""
    site = _browser_publication(tmp_path_factory, history_source)
    catalog = runtime_catalog(site)
    publication = publication_metadata(site)
    assert publication["publisher_inputs_sha256"] == capture_sources.publication_inputs_digest(REPO, include_history=True, history_source=history_source)
    assert catalog["capture_generation"]["mode"] == "history"
    assert catalog["capture_generation"]["history_pin"] == capture_sources.history_pin(REPO, history_source)
    assert catalog["capture_generation"]["current"] == capture_sources.current_identities(capture_sources.current(REPO))
    return site


@pytest.fixture
def current_source_view(tmp_path):
    source = tmp_path / "current-source"
    capture_sources.stage_current_sources(REPO, source)
    shutil.copyfile(REPO / "index.html", source / "index.html")
    shutil.copytree(REPO / "conformance/viewer", source / "conformance/viewer",
                    ignore=shutil.ignore_patterns("node_modules"))
    shutil.copytree(REPO / "conformance/utils/src/assets", source / "conformance/utils/src/assets")
    shutil.copyfile(source / "conformance/viewer-catalog.json", source / "conformance/viewer-unified-catalog.json")
    return source


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
    parser.addoption("--include-history", action="store_true", default=os.environ.get("CONFORMANCE_INCLUDE_HISTORY") == "true",
                     help="qualify bundled sparse historical data explicitly")
    parser.addoption("--conformance-scope", choices=("fixtures", "publication"))
    parser.addoption("--conformance-group", help="disjoint sorted test assignment I/N, zero-based")


def needs_publication(item):
    # Closure names include indirect dependencies (e.g. a browser driver).
    return (bool({"published_viewer", "historical_published_viewer", "source_catalog", "native_publisher"}.intersection(item.fixturenames))
            or item.path.name in {"test_viewer_catalog.py", "test_unified_regeneration.py", "test_capture_split_browser.py"}
            or item.name in {"test_reference_backfills_require_request_bound_version_receipts",
                             "test_bundled_history_json_snapshots_match_original_yaml",
                             "test_retained_unified_captures_cover_the_current_corpus"})


@pytest.hookimpl(hookwrapper=True)
def pytest_collection_modifyitems(config, items):
    yield
    scope = config.getoption("--conformance-scope")
    group = config.getoption("--conformance-group")
    include_history = config.getoption("--include-history")
    eligible = [item for item in items
                if (include_history or not {"history_source", "historical_unified_sources", "historical_published_viewer"}.intersection(item.fixturenames))
                and (not scope or needs_publication(item) == (scope == "publication"))]
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
