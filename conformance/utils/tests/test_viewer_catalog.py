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
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "conformance/utils/src"))
import viewer_catalog
import package_fixtures
import unified_history
import unified_archives

import pytest

from viewer_catalog import _availability_document, _apply_unavailable_controls
from conformance.utils.tests.viewer_test_support import run_node_contract


def test_current_native_metadata_includes_gpt_and_active_structural_scenarios():
    metadata = viewer_catalog._source_metadata()["unified_presentation"]
    declarations = metadata["parser_generations"]
    assert declarations["gpt_oss"]["native_unified_v2"] is True
    assert declarations["harmony"]["native_unified_v2"] is False
    assert declarations["harmony_text"]["native_unified_v2"] is False
    structural = {scenario: families for scenario, families in metadata["scenario_families"].items()
                  if scenario.endswith("_pr424")}
    assert len(structural) == 30
    assert all(families == ["gpt_oss"] for families in structural.values())
    assert set(structural) <= set(metadata["authored_scenarios"])


def test_retained_authored_corpus_does_not_require_optional_legacy_backfill():
    metadata = viewer_catalog._source_metadata()["unified_presentation"]
    assert "v1_split_parallel_tool_calls" not in metadata["authored_scenarios"]
    assert metadata["v1_split_scenarios"] == []
    assert "tool_only" in metadata["authored_scenarios"]


def test_catalog_cli_writes_generation_provenance_without_changing_check(tmp_path, monkeypatch):
    output = tmp_path / "catalog output.json"
    output.write_text(json.dumps({"revision": "catalog-revision"}))
    monkeypatch.setattr(viewer_catalog, "refresh_catalog", lambda *args, **kwargs: True)
    monkeypatch.setattr(viewer_catalog.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="a" * 40 + "\n"))
    monkeypatch.setattr(sys, "argv", ["viewer_catalog.py", "--root", str(tmp_path), "--output", str(output)])
    assert viewer_catalog.main() == 0
    provenance_path = tmp_path / "viewer-publication.json"
    provenance = json.loads(provenance_path.read_text())
    assert provenance["catalog_revision"] == "catalog-revision"
    assert provenance["revision"] == "a" * 40
    assert provenance["generated_revision"] == provenance["revision"]
    assert provenance["output"] == str(output)
    assert f"--output '{output}'" in provenance["command"]
    assert provenance["source_checkout"] is True
    before = provenance_path.read_bytes()
    monkeypatch.setattr(sys, "argv", ["viewer_catalog.py", "--root", str(tmp_path), "--check"])
    assert viewer_catalog.main() == 0
    assert provenance_path.read_bytes() == before


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


def test_unified_catalog_projection_keeps_legacy_selection_identities_without_archives():
    catalog = {
        "schema": 1,
        "revision": "revision",
        "metadata": {
            "tabs": [{"id": "tab-toolcalling-streamv1", "suite": "toolcalling"}, {"id": "tab-unified", "suite": "unified"}],
            "selection_policy": {"unified": {"reference": "dynamo_v2"}, "stream": {"reference": "dynamo_v2"}},
        },
        "sources": {
            "archives": [{"path": "legacy.tar.gz"}],
            "unified": [{"path": "family/capture.yaml"}],
            "candidates": [
                {"mode": "stream", "implementation": "dynamo_v2", "version": "0.7.22"},
                {"mode": "stream", "implementation": "vllm_python", "version": "0.31.0"},
                {"mode": "batch", "implementation": "sglang_python", "version": "0.5.21"},
                {"mode": "batch-on-stream", "implementation": "vllm_python", "version": "0.30.0"},
                {"mode": "reasoning", "implementation": "dynamo_v1", "version": "1.0.0"},
            ],
        },
    }

    projected = viewer_catalog.unified_projection(catalog)

    assert projected["metadata"]["legacy_candidate_identities"] == {
        "dynamo_v2-0-7-22": "dynamo_v2@0.7.22",
        "vllm_python-0-31-0": "vllm_python@0.31.0",
        "sglang_python-b-0-5-21": "sglang_python@0.5.21",
        "vllm_python-s-0-30-0": "vllm_python@0.30.0",
    }
    assert projected["sources"]["archives"] == []
    assert [tab["id"] for tab in projected["metadata"]["tabs"]] == ["tab-unified"]


def test_complete_snapshot_catalog_metadata_is_collected_during_archive_inventory(tmp_path, monkeypatch):
    root = tmp_path / "fixtures-unified-v2"
    family = root / "families/probe"
    family.mkdir(parents=True)
    (family / "inputs_and_golden.yaml").write_text(
        "family: probe\ncases:\n  active: {}\n  retired:\n    lifecycle: retired\n"
    )
    snapshot = {
        "schema_version": 2,
        "records": {
            "active": {"value": {"observation": {"value": {"parser_path": "unified"}}}},
            "other": {"value": {"observation": {"unavailable": {"code": "unsupported_family"}}}},
        },
        "execution": {"manifest": "historical"},
    }
    (family / "dynamo_v2-0.5.0.yaml").write_text(json.dumps(snapshot))
    unified_archives.pack(root, unified_history.CAPTURE_DIRECTORY_RE,
                          unified_history._capture_release_sort_key)
    visits = []
    original_inventory = unified_archives.inventory

    def inventory_with_count(path, *, member_callback=None, source_callback=None):
        def counted(name, data, archive):
            visits.append(name)
            if member_callback:
                member_callback(name, data, archive)
        return original_inventory(path, member_callback=counted, source_callback=source_callback)

    monkeypatch.setattr(unified_archives, "inventory", inventory_with_count)
    resources, documents, metadata = viewer_catalog._catalog_document_index(root)
    assert sorted(visits) == sorted(resource["path"] for resource in resources)
    assert len(visits) == len(set(visits))
    assert documents["families/probe/dynamo_v2-0.5.0.yaml"]["execution"] == {"manifest": "historical"}
    candidates = [{"mode": "unified", "implementation": "dynamo_v2", "version": "0.5.0"}]
    viewer_catalog._index_snapshot_families(candidates, metadata)
    assert candidates[0]["has_captured_results"] is True
    assert candidates[0]["capture_families"] == ["probe"]
    assert candidates[0]["native_unified_families"] == ["probe"]


@pytest.mark.parametrize("implementation", ["sglang_python", "vllm_python", "vllm_rust", "dynamo_v2", "sglang_python_batch"])
@pytest.mark.parametrize("metadata,code", [
    ({}, "unsupported_family"),
    ({"verified_recapture": None}, "unsupported_family"),
    ({"verified_recaptures": [{"result": {"chunks": []}}]}, "unsupported_family"),
    ({"verified_recapture": {"result": {"chunks": []}}}, None),
    ({"verified_recapture": {"result": {"error": "failure"}}}, None),
    ({"verified_recapture": {"result": {"unavailable": "unsupported request", "capture_stimulus": {}}}}, None),
    ({"verified_recapture": {"result": {"capture_observation": {"unavailable": {"code": "unsupported_family"}}}}}, "unsupported_family"),
    ({"verified_recapture": {"result": {"capture_observation": {"unavailable": {"code": ""}}}}}, ""),
])
def test_snapshot_availability_uses_selected_recapture_controls(tmp_path, implementation, metadata, code):
    root = tmp_path / "fixtures-unified-v2"
    family = root / "families/probe"
    family.mkdir(parents=True)
    (family / "inputs_and_golden.yaml").write_text("family: probe\ncases: {active: {}}\n")
    record = {"document": {"parser_path": "unified", "record_metadata": metadata},
              "observation": {"unavailable": {"code": "unsupported_family"}}}
    (family / f"{implementation}-0.5.0.yaml").write_text(json.dumps({
        "schema_version": 2, "records": {"active": {"value": record}},
    }))
    _resources, _documents, summary = viewer_catalog._catalog_document_index(root)
    row = summary["snapshots"][("probe", implementation)][0]
    assert row["code_values"] == [code]
    assert row["has_results"] is False
    assert row["native"] is True
    candidates = [{"mode": "unified", "implementation": implementation, "version": version}
                  for version in ["0.5.0", "0.5.1"]]
    viewer_catalog._index_snapshot_families(candidates, summary)
    assert all(candidate["capture_families"] == ([] if code == "unsupported_family" else ["probe"])
               for candidate in candidates)


@pytest.mark.parametrize("active,codes,available", [
    (["a"], {}, True),
    (["a", "b"], {"a": "unsupported_family", "b": "unsupported_family"}, False),
    (["a", "b"], {"a": "unsupported_family", "retired": "other_unavailable"}, True),
    (["a"], {"a": "unsupported_family", "retired": None}, True),
    (["a", "b"], {"a": "unsupported_family", "b": "other_unavailable"}, False),
    (["a"], {"a": ""}, True),
    ([], {"retired": "unsupported_family"}, False),
])
def test_compact_snapshot_codes_preserve_availability(active, codes, available):
    summary = viewer_catalog._snapshot_code_summary(codes, active)
    assert summary["record_count"] == len(codes)
    candidate = {"mode": "unified", "implementation": "dynamo_v2", "version": "0.5.0"}
    metadata = {"families": {"probe": active}, "snapshots": {("probe", "dynamo_v2"): [
        {**summary, "version": unified_history._capture_release_sort_key("0.5.0"), "has_results": False, "native": False},
    ]}}
    viewer_catalog._index_snapshot_families([candidate], metadata)
    assert ("probe" in candidate["capture_families"]) == available


@pytest.mark.parametrize("snapshot_version,candidate_version,declared,route_recorded,results,count,eligible", [
    ("2.0.0", "2.0.0", True, False, True, 1, True),
    ("1.0.0", "2.0.0", True, False, True, 1, False),
    ("1.0.0", "1.0.0", True, False, True, 1, False),
    ("2.0.0", "2.0.0", False, False, True, 1, False),
    ("2.0.0", "2.0.0", True, True, True, 1, False),
    ("2.0.0", "2.0.0", True, False, False, 1, False),
    ("2.0.0", "2.0.0", True, False, True, 0, False),
])
def test_current_native_declaration_requires_exact_unrouted_evidence(
        snapshot_version, candidate_version, declared, route_recorded, results, count, eligible):
    candidate = {"mode": "unified", "implementation": "dynamo_v2", "version": candidate_version}
    metadata = {"families": {"probe": ["active"]}, "snapshots": {("probe", "dynamo_v2"): [{
        "version": unified_history._capture_release_sort_key(snapshot_version),
        "record_count": count, "code_values": [None] if results else ["unsupported_family"],
        "active_cases_complete": True, "has_results": results, "native": False,
        "route_recorded": route_recorded,
    }]}}
    viewer_catalog._index_snapshot_families([candidate], metadata, active_version="2.0.0",
                                          declared_native_families={"probe"} if declared else set())
    assert candidate["native_unified_families"] == (["probe"] if eligible else [])


@pytest.mark.parametrize("snapshot_version,route_recorded,qualified,eligible", [
    ("1.0.0", False, "1.0.0", True),
    ("0.9.0", False, "1.0.0", False),
    ("1.0.0", True, "1.0.0", False),
    ("1.0.0", False, "0.9.0", False),
])
def test_retained_native_capture_qualification_survives_a_newer_release(
        snapshot_version, route_recorded, qualified, eligible):
    candidate = {"mode": "unified", "implementation": "dynamo_v2", "version": "2.0.0"}
    metadata = {"families": {"probe": ["active"]}, "snapshots": {("probe", "dynamo_v2"): [{
        "version": unified_history._capture_release_sort_key(snapshot_version),
        "record_count": 1, "code_values": [None], "active_cases_complete": True,
        "has_results": True, "native": False, "route_recorded": route_recorded,
    }]}}
    viewer_catalog._index_snapshot_families(
        [candidate], metadata, active_version="2.0.0", declared_native_families={"probe"},
        unrouted_native_captures={"probe": qualified})
    assert candidate["native_unified_families"] == (["probe"] if eligible else [])


@pytest.mark.parametrize("scope", ["metadata", "observation", "document"])
@pytest.mark.parametrize("route", [None, "", "unknown", "split", "unified"])
def test_snapshot_route_presence_includes_explicit_empty_values(tmp_path, scope, route):
    root = tmp_path / "fixtures-unified-v2"
    family = root / "families/probe"
    family.mkdir(parents=True)
    (family / "inputs_and_golden.yaml").write_text("family: probe\ncases: {active: {}}\n")
    value = {"document": {"record_metadata": {}}, "observation": {"value": {}}}
    scopes = {"metadata": value["document"]["record_metadata"],
              "observation": value["observation"]["value"], "document": value["document"]}
    scopes[scope]["parser_path"] = route
    (family / "dynamo_v2-2.0.0.yaml").write_text(json.dumps({
        "schema_version": 2, "records": {"active": {"value": value}},
    }))
    _resources, _documents, summary = viewer_catalog._catalog_document_index(root)
    row = summary["snapshots"][("probe", "dynamo_v2")][0]
    assert row["route_recorded"] is True
    assert row["native"] == (route == "unified")


def test_complete_snapshot_generation_rejects_archive_mutation(tmp_path):
    store = tmp_path / "conformance/fixtures-unified-v2"
    family = store / "families/probe"
    family.mkdir(parents=True)
    (family / "inputs_and_golden.yaml").write_text("family: probe\ncases: {}\n")
    (family / "dynamo_v2-0.5.0.yaml").write_text(json.dumps({"schema_version": 2, "records": {}}))
    unified_archives.pack(store, unified_history.CAPTURE_DIRECTORY_RE,
                          unified_history._capture_release_sort_key)
    generation = viewer_catalog.CatalogGeneration(tmp_path, threading.get_ident())

    viewer_catalog._catalog_document_index(store, generation=generation)
    archive = next((store / "versions").iterdir())
    archive.write_bytes(archive.read_bytes() + b"changed-after-inventory")

    with pytest.raises(ValueError, match="Unified source changed during catalog generation"):
        generation.validate_inventories()


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
    assert catalog["schema"] == 1
    assert catalog["metadata"]["tabs"]
    assert catalog["metadata"]["unified_taxonomy"]
    parser_generations = catalog["metadata"]["unified_presentation"]["parser_generations"]
    assert parser_generations["deepseek_v4"]["label"] == "v1/v2 split"
    assert parser_generations["muse_glimmer"]["label"] == "v2"
    assert parser_generations["qwen3"]["label"] == "v2"
    assert parser_generations["deepseek_v3"]["label"] == "v1/v2 split"
    assert parser_generations["harmony"]["label"] == "v1/v2 split"
    assert parser_generations["gpt_oss"]["label"] == "v1/v2 split"
    assert parser_generations["gpt_oss"]["native_unified_v2"] is True
    assert parser_generations["deepseek_v4"]["native_unified_v2"] is True
    assert parser_generations["deepseek_v3"]["native_unified_v2"] is False
    unified_families = {
        source["family"] for source in catalog["sources"]["unified"]
        if source["kind"] == "inputs-and-golden"
    }
    assert unified_families <= set(parser_generations)
    assert unified_families == {path.parent.name for path in
                                (ROOT / "conformance/fixtures-unified-v2/families").glob("*/inputs_and_golden.yaml")}
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
    assert catalog["capture_generation"]["mode"] == "latest"
    assert catalog["sources"]["archives"] == []
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
    monkeypatch.setattr(package_fixtures, "_validate_candidate_package", lambda *_args, **_kwargs: None)
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


def test_catalog_check_shares_readers_and_excludes_writers(tmp_path, monkeypatch):
    _fixture_root(tmp_path, monkeypatch)
    reader_entered = threading.Event()
    writer_entered = threading.Event()
    errors = []

    def read():
        try:
            with viewer_catalog.catalog_generation(tmp_path, read_only=True):
                reader_entered.set()
        except Exception as error:
            errors.append(error)

    def write():
        try:
            with viewer_catalog.catalog_generation(tmp_path):
                writer_entered.set()
        except Exception as error:
            errors.append(error)

    with unified_history._store_read_lock(tmp_path / "conformance/fixtures-unified-v2"):
        reader = threading.Thread(target=read)
        writer = threading.Thread(target=write)
        reader.start()
        writer.start()
        assert reader_entered.wait(5)
        assert not writer_entered.wait(0.2)
    reader.join(5)
    writer.join(5)
    assert not reader.is_alive() and not writer.is_alive()
    assert writer_entered.is_set()
    assert errors == []
    with viewer_catalog.catalog_generation(tmp_path, read_only=True) as generation:
        with pytest.raises(ValueError, match="writer lease"):
            viewer_catalog.refresh_catalog(tmp_path, generation=generation)


def test_unified_family_presence_is_version_specific_and_keeps_failures():
    def history(states):
        return SimpleNamespace(
            captures={version: {"runtime_version": version, "changes": state, "metadata_changes": {}} for version, state in states.items()},
            ordered_capture_ids=lambda: list(states), resolve=lambda version: states[version],
        )
    def observation(**state):
        return {"case": {"observation": state}}
    store = SimpleNamespace(histories={
        ("A", "v1"): history({"1.0.0": observation(value={})}),
        ("B", "v1"): history({"1.0.0": observation(error={"code": "panic"})}),
        ("B", "v2"): history({"1.0.0": observation(value={})}),
        ("C", "v2"): history({"1.0.0": observation(unavailable={"code": "unsupported_family"}),
                                "2.0.0": observation(value={})}),
        ("inherited_registration", "v2"): history({
            "1.0.0": {"mode": {"observation": {"unavailable": {"code": "unsupported_request_mode"}}}},
            "2.0.0": observation(unavailable={"code": "peer_detector_unavailable"}),
        }),
        ("missing", "v2"): history({"1.0.0": {}}),
        ("registration", "v2"): history({"1.0.0": {**observation(unavailable={"code": "peer_detector_unavailable"}),
                "mode": {"observation": {"unavailable": {"code": "unsupported_request_mode"}}}}}),
    })
    store.families = {family: None for family, _implementation in store.histories}
    candidates = [{"mode": "unified", "implementation": impl, "version": version}
                  for impl, version in [("v1", "1.0.0"), ("v2", "1.0.0"), ("v2", "2.0.0")]]
    viewer_catalog._index_unified_families(store, candidates)
    assert [item["capture_families"] for item in candidates] == [["A", "B", "C", "inherited_registration", "missing", "registration"], ["A", "B", "inherited_registration", "missing"], ["A", "B", "C", "missing"]]


def test_empty_unified_releases_are_not_marked_as_captured_results():
    def history(states):
        return SimpleNamespace(
            captures={version: {"runtime_version": version, "changes": state, "metadata_changes": {}}
                      for version, state in states.items()},
            ordered_capture_ids=lambda: list(states), resolve=lambda version: states[version],
        )

    store = SimpleNamespace(histories={
        ("family", "dynamo_v1_jail"): history({
            "0.9.1": {"case": {"observation": {"unavailable": {"code": "unsupported_request_mode"}}}},
            "1.4.2": {},
            "3.1.0": {"case": {"observation": {"value": {"assembled": []}}}},
            "3.1.1": {},
        }),
    }, families={"family": None})
    candidates = [{"mode": "unified", "implementation": "dynamo_v1_jail", "version": version}
                  for version in ("0.9.1", "1.4.2", "3.1.0", "3.1.1")]

    viewer_catalog._index_unified_families(store, candidates)

    assert [item["has_captured_results"] for item in candidates] == [False, False, True, True]


def test_v2_native_families_follow_recorded_routes_at_selected_version():
    def history(captures):
        return SimpleNamespace(captures=captures, ordered_capture_ids=lambda: list(captures))

    def checkpoint(version, route=None, observation=None, *, clear=False, remove=False):
        return {"runtime_version": version,
                "changes": {"case": {"absent": True}} if remove else {"case": {"observation": observation}} if observation is not None else {},
                "metadata_changes": {"case": {}} if clear else {"case": {"parser_path": route}} if route else {}}

    versions = ["1.0.0", "2.0.0", "3.0.0", "4.0.0", "5.0.0"]
    switched = [checkpoint("1.0.0", "split", {"value": {}}),
                checkpoint("2.0.0", "unified", {"error": {"code": "panic"}}),
                checkpoint("3.0.0", observation={"value": {}}),
                checkpoint("4.0.0", clear=True), checkpoint("5.0.0", remove=True)]
    split = [checkpoint(version, "split", {"value": {}}) for version in versions]
    store = SimpleNamespace(families={family: None for family in ["converted", "gpt_oss", "no_history"]}, histories={
        ("converted", "dynamo_v2"): history(dict(zip(versions, switched))),
        ("gpt_oss", "dynamo_v2"): history(dict(zip(versions, split))),
    })
    candidates = [{"mode": "unified", "implementation": "dynamo_v2", "version": version} for version in versions]
    viewer_catalog._index_unified_families(store, candidates)
    assert [item["native_unified_families"] for item in candidates] == [[], ["converted"], ["converted"], [], []]
    assert all("gpt_oss" in item["capture_families"] for item in candidates), "historical observations remain retained"


def test_v2_native_path_precedence_and_removal():
    captures = {
        "1.0.0": {"runtime_version": "1.0.0", "document": {"parser_path": "split"},
                  "changes": {"case": {"observation": {"value": {}}}}, "metadata_changes": {},
                  "document_overrides": {"case": {"parser_path": "unified"}}},
        "2.0.0": {"runtime_version": "2.0.0", "changes": {},
                  "metadata_changes": {"case": {"parser_path": "split"}}},
        "3.0.0": {"runtime_version": "3.0.0", "changes": {}, "metadata_changes": {"case": {}},
                  "document_overrides": {"case": {"parser_path": None}}},
        "4.0.0": {"runtime_version": "4.0.0", "changes": {"case": {"observation": {"value": {"parser_path": "unified"}}}},
                  "metadata_changes": {}},
    }
    store = SimpleNamespace(families={"A": None}, histories={
        ("A", "dynamo_v2"): SimpleNamespace(captures=captures, ordered_capture_ids=lambda: list(captures)),
    })
    candidates = [{"mode": "unified", "implementation": "dynamo_v2", "version": version} for version in captures]
    viewer_catalog._index_unified_families(store, candidates)
    assert [item["native_unified_families"] for item in candidates] == [["A"], [], [], ["A"]]


def test_new_batch_on_stream_evidence_uses_the_legacy_candidate_owner(tmp_path, monkeypatch):
    _fixture_root(tmp_path, monkeypatch)
    paths = [f"conformance/fixtures/additions/checksum/toolcalling/fixtures-{mode}/vllm_python-2.0.0.tar.gz"
             for mode in ("batch-on-stream-v1", "stream-v1")]
    monkeypatch.setattr(viewer_catalog, "_archive_records", lambda *_args: [{"path": path} for path in paths])
    monkeypatch.setattr(viewer_catalog, "_source_metadata", lambda _root: {"implementations": [{"key": "vllm_python"}], "unified_capture_policy": {}})
    catalog = viewer_catalog.build_catalog(tmp_path)
    selected = {(row["mode"], row["version"]) for row in catalog["sources"]["candidates"]}
    assert selected == {("batch-on-stream", "2.0.0"), ("stream", "2.0.0")}
