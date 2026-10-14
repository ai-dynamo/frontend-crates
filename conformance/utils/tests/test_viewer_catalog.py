from __future__ import annotations

import json
import io
import tarfile
import threading
import hashlib
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "conformance/utils/src"))
import viewer_catalog
import package_fixtures
import unified_history

import pytest

from viewer_catalog import _availability_document, _apply_unavailable_controls
from conformance.utils.tests.viewer_test_support import run_node_contract


def test_shared_availability_history_controls():
    assert "Selection-independent availability controls" in run_node_contract("availability.test.js")


def test_availability_projection_retains_only_capture_controls():
    document = {"family": "probe", "mode": "streamv1", "cases": {"TOOLCALLING.streamv1.1": {
        "model_text": "private stimulus sentinel", "tools": [{"description": "private schema sentinel"}],
        "expected": {"vllm_python": {"unavailable": "private reason sentinel"}, "dynamo_v1": {"calls": [{"name": "private function sentinel", "arguments": {"value": 2 ** 90}}]}},
        "unavailable": {"sglang": "private peer sentinel"},
        "chunks": [{"delta_text": "private delta sentinel", "expected": {"dynamo_v2": [{"name": "private emitted name sentinel"}]}, "normal_text": {"dynamo_v2": "private output sentinel", "vllm_python": ""}}],
    }}}
    before = deepcopy(document)
    control = _availability_document(document)
    assert control == {"family": "probe", "mode": "streamv1", "cases": {"TOOLCALLING.streamv1.1": {
        "expected": {"vllm_python": "unavailable", "dynamo_v1": "present"},
        "unavailable": {"sglang": True},
        "chunk_runs": [[1, {"expected": ["dynamo_v2"], "normal_text": {"dynamo_v2": True, "vllm_python": False}}]],
    }}}
    assert document == before
    assert "private" not in json.dumps(control)


def test_availability_projection_preserves_sparse_stream_state_and_shape():
    assert _availability_document({"family": "probe", "mode": "streamv1", "cases": {"old": {}, "new": {"unavailable": "reason"}, "failed": {"exception": "reason"}, "short": {"chunks": [{"expected": []}, {"normal_text": "content"}]}}})["cases"] == {
        "old": {}, "new": {"unavailable": True}, "failed": {"exception": True},
        "short": {"chunk_runs": [[1, {"expected": True}], [1, {"normal_text": True}]]},
    }


def test_unavailable_failure_controls_use_shared_predicate_without_raw_reasons():
    controls = []
    projection = _availability_document({"family": "probe", "mode": "batch", "cases": {
        "batch": {"expected": {"dynamo_v1": {"unavailable": "parser not captured: private failure sentinel"}, "vllm_python": {"unavailable": "no parser: private absence sentinel", "parserError": True}}},
        "stream": {"unavailable": {"dynamo_v2": "parsing failed: private state sentinel"}},
        "capture": {"unavailable": "parse error: private capture sentinel"},
    }}, controls)
    _apply_unavailable_controls(controls)
    assert projection["cases"]["batch"]["expected"] == {"dynamo_v1": "unavailable-error", "vllm_python": "unavailable"}
    assert projection["cases"]["stream"]["unavailable"] == {"dynamo_v2": "parser-error"}
    assert projection["cases"]["capture"]["unavailable"] == "parser-error"
    assert "private" not in json.dumps(projection)


def test_viewer_catalog_is_current_and_contains_only_resource_metadata():
    result = subprocess.run(
        ["python3", "conformance/utils/src/viewer_catalog.py", "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    catalog = json.loads((ROOT / "conformance/viewer-catalog.json").read_text())
    manifest = json.loads((ROOT / "conformance/fixtures-manifest.json").read_text())
    assert catalog["schema"] == 1
    assert catalog["metadata"]["tabs"]
    assert catalog["metadata"]["unified_taxonomy"]
    parser_generations = catalog["metadata"]["unified_presentation"]["parser_generations"]
    assert parser_generations["deepseek_v4"]["label"] == "v1/v2 split"
    assert parser_generations["muse_glimmer"]["label"] == "v2"
    assert parser_generations["qwen3"]["label"] == "v2"
    assert parser_generations["deepseek_v3"]["label"] == "v1"
    assert parser_generations["harmony"]["label"] == "v1/v2 split"
    assert parser_generations["gpt_oss"]["label"] == "v1/v2 split"
    assert parser_generations["deepseek_v4"]["native_unified_v2"] is True
    assert parser_generations["deepseek_v3"]["native_unified_v2"] is False
    unified_families = {
        source["family"] for source in catalog["sources"]["unified"]
        if source["kind"] == "inputs-and-golden"
    }
    assert unified_families == set(parser_generations)
    assert len(unified_families) == 27
    for family, generation in parser_generations.items():
        assert generation["label"] in {"v1", "v1/v2 split", "v2"}, family
        assert generation["generation"] == {
            "v1": "v1", "v1/v2 split": "v1+v2 split", "v2": "v2",
        }[generation["label"]], family
        assert isinstance(generation["native_unified_v2"], bool), family
    assert catalog["metadata"]["source_hashes"]["documentation"]
    assert all("members" in archive for archive in catalog["sources"]["archives"])
    identities = [(candidate["implementation"], candidate["mode"], candidate["version"]) for candidate in catalog["sources"]["candidates"]]
    assert len(identities) == len(set(identities))
    assert all(candidate["sources"] for candidate in catalog["sources"]["candidates"])
    assert catalog["metadata"]["unified_aliases"]["owners"]
    revision = catalog.pop("revision")
    assert revision == hashlib.sha256(json.dumps(catalog, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    manifest_archives = {row["path"] for row in manifest["shards"] if row["path"].endswith(".tar.gz")}
    assert {row["path"].removeprefix("conformance/fixtures/") for row in catalog["sources"]["archives"]} == manifest_archives
    forbidden = {"cells", "results", "comparison", "computed_status"}

    def walk(value):
        if isinstance(value, dict):
            assert forbidden.isdisjoint(value)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(catalog)


def _fixture_root(tmp_path, monkeypatch):
    conformance = tmp_path / "conformance"
    (conformance / "fixtures").mkdir(parents=True)
    (conformance / "fixtures-unified-v2").mkdir()
    (conformance / "fixtures-manifest.json").write_text(json.dumps({"snapshot": "old", "shards": [], "inactive_shards": [], "crates": {"prior": "1"}}))
    (conformance / "viewer-catalog.json").write_text('{"snapshot":"old"}')
    monkeypatch.setattr(viewer_catalog, "_source_metadata", lambda root: {"implementations": [], "source_root": str(root)})
    monkeypatch.setattr(package_fixtures, "ROOT", tmp_path)
    monkeypatch.setattr(package_fixtures, "FIXTURES_DIR", conformance / "fixtures")
    monkeypatch.setattr(package_fixtures, "UNIFIED_HISTORY_DIR", conformance / "fixtures-unified-v2")
    monkeypatch.setattr(package_fixtures, "stage_fixtures", lambda *_args: None)
    monkeypatch.setattr(package_fixtures, "build_shards", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(package_fixtures, "sync_store", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(package_fixtures, "_validate_candidate_package", lambda *_args: None)
    return conformance


def _package():
    package_fixtures.package_snapshot("new", "new America/Los_Angeles", {}, {}, dry_run=False, prune=False)


def test_catalog_root_and_staged_resource_urls(tmp_path, monkeypatch):
    conformance = _fixture_root(tmp_path, monkeypatch)
    staged = tmp_path / "staged"
    fixtures = staged / "fixtures"
    history = staged / "history"
    fixtures.mkdir(parents=True)
    history.mkdir()
    archive = fixtures / "input.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        data = b"family: sample\n"
        member = tarfile.TarInfo("sample/case.yaml")
        member.size = len(data)
        handle.addfile(member, io.BytesIO(data))
    payload = archive.read_bytes()
    manifest = staged / "manifest.json"
    manifest.write_text(json.dumps({"snapshot": "staged", "shards": [{"path": archive.name, "sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload)}]}))
    output = staged / "catalog.json"
    viewer_catalog.refresh_catalog(tmp_path, destination=output, fixtures=fixtures, unified=history, manifest=manifest)
    result = json.loads(output.read_text())
    assert result["snapshot"] == "staged"
    assert result["metadata"]["source_root"] == str(tmp_path)
    assert result["sources"]["archives"][0]["path"] == "conformance/fixtures/input.tar.gz"
    assert output.stat().st_mode & 0o004, "published catalog must be readable by the static site server"
    assert json.loads((conformance / "viewer-catalog.json").read_text())["snapshot"] == "old"


def test_catalog_build_failure_preserves_entire_package_generation(tmp_path, monkeypatch):
    conformance = _fixture_root(tmp_path, monkeypatch)
    before = {name: (conformance / name).read_bytes() for name in ["fixtures-manifest.json", "viewer-catalog.json"]}
    def fail(*_args, **_kwargs):
        raise ValueError("injected catalog build failure")
    monkeypatch.setattr(viewer_catalog, "build_catalog", fail)
    with pytest.raises(ValueError, match="injected catalog"):
        _package()
    assert before == {name: (conformance / name).read_bytes() for name in before}


def test_package_publication_failure_rolls_back_catalog_too(tmp_path, monkeypatch):
    conformance = _fixture_root(tmp_path, monkeypatch)
    before = {name: (conformance / name).read_bytes() for name in ["fixtures-manifest.json", "viewer-catalog.json"]}
    replace = unified_history.os.replace
    def fail_candidate(source, destination):
        if Path(source).name == "viewer-catalog.json" and Path(destination) == conformance / "viewer-catalog.json":
            raise OSError("injected catalog publication failure")
        return replace(source, destination)
    monkeypatch.setattr(unified_history.os, "replace", fail_candidate)
    with pytest.raises(OSError, match="injected catalog publication"):
        _package()
    assert before == {name: (conformance / name).read_bytes() for name in before}


def test_explicit_refresh_waits_for_package_generation(tmp_path, monkeypatch):
    conformance = _fixture_root(tmp_path, monkeypatch)
    package_building = threading.Event()
    refresh_building = threading.Event()
    release_package = threading.Event()
    def metadata(root):
        if threading.current_thread().name == "packager":
            package_building.set()
            assert release_package.wait(5)
        else:
            refresh_building.set()
        return {"implementations": []}
    monkeypatch.setattr(viewer_catalog, "_source_metadata", metadata)
    errors = []
    def run(action):
        try:
            action()
        except Exception as error:
            errors.append(error)
    package = threading.Thread(target=lambda: run(_package), name="packager")
    refresh = threading.Thread(target=lambda: run(lambda: viewer_catalog.refresh_catalog(tmp_path)), name="refresher")
    package.start()
    assert package_building.wait(5)
    refresh.start()
    assert not refresh_building.wait(0.2)
    release_package.set()
    package.join(5)
    refresh.join(5)
    assert not package.is_alive() and not refresh.is_alive()
    assert errors == []
    assert refresh_building.is_set()
    assert json.loads((conformance / "fixtures-manifest.json").read_text())["snapshot"] == "new"
    assert json.loads((conformance / "viewer-catalog.json").read_text())["snapshot"] == "new"


def test_catalog_lease_rejects_inactive_and_wrong_root(tmp_path, monkeypatch):
    _fixture_root(tmp_path, monkeypatch)
    with viewer_catalog.catalog_generation(tmp_path) as generation:
        with pytest.raises(ValueError, match="another root"):
            viewer_catalog.refresh_catalog(tmp_path / "elsewhere", generation=generation)
    with pytest.raises(ValueError, match="inactive"):
        viewer_catalog.refresh_catalog(tmp_path, generation=generation)
