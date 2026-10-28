# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Isolated checkout identity for synthetic fixture-extraction tests."""

from pathlib import Path
import copy
import json
import shutil
import subprocess
from types import SimpleNamespace
import capture_sources
import extract_fixtures
import unified_archives
import unified_history
import viewer_catalog
import hashlib
import os
import tempfile
from functools import lru_cache
import migrate_capture_sources
from verify_publication_transport import Decoder, read_index, runtime_catalog
import tarfile
from capture_versions import capture_release_sort_key
from verify_publication_transport import runtime_catalog
from collections import defaultdict
from unified_tools import unified_tools


REPO = Path(__file__).resolve().parents[3]


def prepare_isolated_publication(root, history, view, monkeypatch, publisher, *, mode, metadata=None,
                                 retained_history_only=False):
    """Exercise both source modes against the same tiny, verified synthetic dataset."""
    tools = Path(capture_sources.__file__).parent
    for name in ("capture_sources.py", "ci_publication_cache.py", "verify_publication_transport.py", "yaml_fast.py", "unified_archives.py", "execution_manifests.py", "impls.py", "viewer_codec.py", "parser_families.yaml"):
        target = root / "conformance/utils/src" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(tools / name, target)
    (root / "index.html").write_bytes(b"<head></head><body></body>")
    (root / "conformance/CONFORMANCE_v2.html").write_bytes(b'<a href="../">Viewer</a>')
    (root / "conformance/fixtures-manifest.json").write_bytes(b'{"shards":[]}\n')
    subprocess.run(["git", "init", str(root)], check=True, capture_output=True)
    original_run = subprocess.run
    view.mkdir()

    def build_catalog(command, **kwargs):
        if any(str(value).endswith("viewer_catalog.py") for value in command):
            generation = json.loads((view / "conformance/capture-generation.json").read_bytes())
            if not unified_archives.paths(view / capture_sources.STORE):
                unified_archives.pack(view / capture_sources.STORE, unified_history.CAPTURE_DIRECTORY_RE,
                                      unified_history._capture_release_sort_key)
            versions = ["0.5.0", "0.5.2"] if mode == "history" else ["0.5.2"]
            manifest_path = view / "conformance/fixtures-manifest.json"
            manifest_bytes = manifest_path.read_bytes()
            manifest = json.loads(manifest_bytes)
            archives = viewer_catalog._archive_records(manifest, view)
            catalog = fixture_catalog(view, {"capture_generation": generation,
                "metadata": copy.deepcopy(metadata or {"unified_capture_policy": {"dynamo_v2": {"mode": "stream"}}}),
                "sources": {"archives": archives, "lineage": generation["lineage"], "candidates": [
                    {"mode": "unified", "implementation": "dynamo_v2", "version": version,
                     "capture_families": ["gemma4"], "native_unified_families": ["gemma4"],
                     "has_captured_results": True} for version in versions]}})
            if (view / capture_sources.DATASETS).is_file():
                _inventory, _documents, indexed = viewer_catalog._native_catalog_document_index(view / capture_sources.STORE)
                catalog["sources"]["history_datasets"] = viewer_catalog._history_dataset_sources(
                    view, catalog["sources"]["unified"], indexed, catalog["metadata"])
                (view / "conformance/viewer-catalog.json").write_bytes(capture_sources.encoded(catalog))
            manifest["shards"] = [row for row in manifest["shards"] if row.get("format") != "unified-history"]
            manifest_path.write_bytes(capture_sources.encoded(manifest))
            return SimpleNamespace(returncode=0)
        return original_run(command, **kwargs)

    with monkeypatch.context() as patch:
        # Supply synthetic catalog metadata; source discovery and admission stay real.
        patch.setattr(subprocess, "run", build_catalog)
        patch.delenv("CONFORMANCE_DOCUMENT_LOCK", raising=False)
        patch.setenv("CONFORMANCE_CONVERSION_CACHE", str(root / "build/conformance-cache"))
        capture_sources.prepare(root, view, include_history=mode == "history",
                                history_source=history if mode == "history" else None, publisher=str(publisher),
                                retained_history_only=retained_history_only)


def configure_extractor_root(root, monkeypatch, *, manifest_path=None):
    # ROOT selects current versus historical storage and supplies release metadata.
    # Partial path patches can accidentally materialize the real checkout instead.
    package = root / "parsers/v2/Cargo.toml"
    package.parent.mkdir(parents=True, exist_ok=True)
    package.write_bytes((REPO / "parsers/v2/Cargo.toml").read_bytes())
    conformance = root / "conformance"
    fixtures = conformance / "fixtures"
    history = conformance / "fixtures-unified-v2"
    fixtures.mkdir(parents=True, exist_ok=True)
    history.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(extract_fixtures, "ROOT", root)
    monkeypatch.setattr(extract_fixtures, "MANIFEST_PATH", manifest_path or conformance / "fixtures-manifest.json")
    monkeypatch.setattr(extract_fixtures, "FIXTURES_DIR", fixtures)
    monkeypatch.setattr(extract_fixtures, "HISTORY_DIR", history)


ROOT = Path(__file__).resolve().parents[3]


@lru_cache(maxsize=1)
def publisher_binary():
    configured = os.environ.get("CONFORMANCE_PUBLISHER")
    if configured:
        binary = Path(configured).resolve(strict=True)
    else:
        subprocess.run(["cargo", "build", "--locked", "--release", "-p", "conformance-publisher"], cwd=ROOT, check=True)
        binary = Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target")) / "release/conformance-publish"
    return binary


def publish(root, destination, *, workers="auto", extra=(), check=True, revision="a" * 40):
    try:
        return subprocess.run([str(publisher_binary()), "--root", str(root), "--output", str(destination), "--revision", revision,
                              "--workers", str(workers), *extra], cwd=ROOT, capture_output=True, text=True, check=check, timeout=4500)
    except subprocess.CalledProcessError as error:
        error.add_note(error.stderr)
        raise


def fixture_catalog(root, catalog):
    if root.resolve() == ROOT:
        raise ValueError("synthetic catalogs cannot replace maintained catalogs")
    store = root / "conformance/fixtures-unified-v2"
    if not unified_archives.paths(store):
        # Historical test builders remain sparse; publication fixtures use the permanent format.
        with tempfile.TemporaryDirectory(
            prefix="native-fixture-", dir=root.parent
        ) as temporary:
            sparse = Path(temporary) / "sparse"
            store.rename(sparse)
            migrate_capture_sources.convert_snapshots(sparse, store, Path(temporary))
    catalog = copy.deepcopy(catalog)
    catalog.setdefault("schema", 1)
    catalog.setdefault("snapshot", "fixture")
    catalog["sources"].setdefault("archives", [])
    catalog["sources"].setdefault("candidates", [])
    catalog["sources"]["unified"] = viewer_catalog._unified_records(root)
    catalog.setdefault("metadata", {}).setdefault("tabs", [])
    catalog["metadata"].setdefault("source_hashes", {})
    catalog.pop("revision", None)
    catalog["revision"] = hashlib.sha256(
        json.dumps(catalog, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    conformance = root / "conformance"
    (root / "index.html").write_bytes(b"<head></head><body></body>")
    (conformance / "CONFORMANCE_v2.html").write_bytes(b'<a href="../">Viewer</a>')
    (conformance / "fixtures-manifest.json").write_text('{"shards":[]}\n')
    (conformance / "viewer-catalog.json").write_text(json.dumps(catalog))
    (conformance / "viewer-unified-catalog.json").write_text(
        json.dumps(viewer_catalog.unified_projection(catalog))
    )
    return catalog


def native_snapshots(root, destination, catalog, *, workers="auto"):
    root, destination = Path(root), Path(destination)
    if root.resolve() != ROOT:
        fixture_catalog(root, catalog)
    publish(root, destination, workers=workers)
    return proof_catalog(root, destination)


def proof_catalog(root, destination, *, source_resources=None):
    catalog = runtime_catalog(destination)
    if source_resources is None:
        source_catalog = json.loads((Path(root) / "conformance/viewer-catalog.json").read_bytes())
        source_resources = source_catalog["sources"]["unified"]
    catalog["proof_original_unified"] = source_resources
    return catalog


def snapshot_bytes(destination, resource, decoder=None):
    decoder = decoder or Decoder(Path(destination).resolve())
    recipe = read_index(Path(destination).resolve(), resource["recipe_index"])[resource["path"]]
    return decoder.reconstruct(recipe)


ROOT = Path(__file__).resolve().parents[3]


def capture_releases(site):
    catalog = runtime_catalog(site)
    versions = {}
    for resource in catalog['sources']['unified']:
        if resource['kind'] == 'resolved-capture':
            versions.setdefault(resource['implementation'], set()).add(resource['version'])
    return {implementation: sorted(releases, key=capture_release_sort_key)
            for implementation, releases in versions.items()}


def historical_source_bytes(root: Path, path: str) -> bytes:
    """Resolve original-member aliases from the verified history source."""
    if (root / path).is_file():
        return capture_sources.read(root, path)
    declaration = root / capture_sources.DATASETS
    if declaration.is_file():
        for dataset in reversed(json.loads(declaration.read_bytes())):
            candidate = capture_sources.relative(dataset["root"]) + "/" + capture_sources.relative(path)
            if (root / candidate).is_file():
                return capture_sources.read(root, candidate)
    # The session fixture verifies all evidence once; each Raw check still hashes
    # the selected bytes, without decompressing unrelated historical snapshots.
    inventory = capture_sources.verify(root)
    member_path = (path.removeprefix(capture_sources.STORE + "/")
                   if path.startswith(capture_sources.STORE + "/") else path.removeprefix("conformance/"))
    for archive in inventory["files"]:
        prefix = archive["path"].removeprefix("conformance/fixtures/").removesuffix(".tar.gz")
        if archive["path"].endswith(".tar.gz") and "members" not in archive and member_path.startswith(prefix + "/"):
            archive["members"] = capture_sources.archive_members(root / archive["path"])
    matches = [(archive, member) for archive in inventory["files"]
               for member in archive.get("members", []) if member["path"] in {path, member_path}]
    if len(matches) != 1:
        raise ValueError(f"original resource must have one retained owner: {path}")
    archive, member = matches[0]
    with tarfile.open(root / archive["path"]) as stream:
        data = stream.extractfile(member["path"]).read()
    assert len(data) == member["size"] and hashlib.sha256(data).hexdigest() == member["sha256"]
    return data


def run_node_contract(filename: str, *arguments: str) -> str:
    environment = {key: os.environ[key] for key in ("PATH", "LANG", "LC_ALL") if key in os.environ}
    result = subprocess.run(
        ["node", str(ROOT / "conformance/viewer" / filename), *arguments],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def colorize_markup(text: str, family: str, markers: dict) -> str:
    environment = {key: os.environ[key] for key in ("PATH", "LANG", "LC_ALL") if key in os.environ}
    driver = "const mc=require(process.argv[1]);const p=JSON.parse(require('fs').readFileSync(0,'utf8'));process.stdout.write(mc.colorizeMarkup(p.text,p.family,p.markers));"
    result = subprocess.run(
        ["node", "-e", driver, str(ROOT / "conformance/utils/src/assets/colorize.js")],
        input=json.dumps({"text": text, "family": family, "markers": markers}),
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def semantic_key(case):
    request = case.get("request", case)
    golden = case["golden"]
    if isinstance(golden, dict):
        golden = golden["assembled"]
    return json.dumps({
        "input": request["input"],
        "init": request["init"],
        "tools": request.get("tools") if request.get("tools") is not None else unified_tools(),
        "golden": golden,
        "chunks": request.get("chunks", case.get("input_chunks")),
        "finish_reason": request.get("finish_reason", "stop"),
    }, sort_keys=True)


def assert_unique_behaviour(family, cases, *, retained_archives=None):
    retained_archives = retained_archives or {}
    entries = list(cases.items()) if isinstance(cases, dict) else list(cases)
    seen_ids = set()
    seen_semantics = defaultdict(list)
    active_ids = set()
    retired_ids = []
    for case_id, case in entries:
        assert case_id not in seen_ids, f"{family}: reused authored case ID {case_id}"
        seen_ids.add(case_id)
        if case.get("lifecycle") == "retired":
            retired_ids.append(case_id)
            continue
        active_ids.add(case_id)
        seen_semantics[semantic_key(case)].append(case_id)
    assert set(retained_archives) <= active_ids, f"{family}: exception has no active authored case"
    for case_id, binding in retained_archives.items():
        assert all(binding.get(field) for field in ("purpose", "input_source", "capture_source", "version")), (
            f"{family}/{case_id}: retained archive exception requires source and version binding")
    duplicates = [names for names in seen_semantics.values() if len(names) > 1]
    for names in duplicates:
        canonical = [name for name in names if name not in retained_archives]
        assert len(canonical) <= 1, f"{family}: scenarios with identical behaviour: {names}"
    return {"family": family, "active_cases": len(active_ids), "retired_cases": retired_ids, "exact_duplicates": duplicates,
            "retained_archive_exceptions": retained_archives}
