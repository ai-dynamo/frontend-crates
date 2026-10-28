# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Normal Unified regeneration must not open retained archive payloads."""
import copy
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "conformance/utils/src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import package_fixtures
import gen_unified_golden as golden
import unified_history
import viewer_catalog
from unified_taxonomy import numbered_id
from fixture_disposition import canonical_unified_case_key
from explode_unified_fixtures import _split_only_cases
from conformance.utils.tests.test_unified_history import _change, _write_capture, _write_family


def _catalog_root(tmp_path, monkeypatch):
    conformance = tmp_path / "conformance"
    store = conformance / "fixtures-unified-v2"
    _write_family(store)
    _write_capture(store, "0.7.16", {"text_only": _change()})
    archive = conformance / "fixtures/toolcalling/fixtures-batch-v1/dynamo_v1-10.0.7.tar.gz"
    archive.parent.mkdir(parents=True)
    with tarfile.open(archive, "w:gz") as output:
        data = b"family: gemma4\nmode: batch\ncases: {}\n"
        member = tarfile.TarInfo("gemma4/TOOLCALLING.batch.1.yaml")
        member.size = len(data)
        output.addfile(member, io.BytesIO(data))
    legacy = {"path": archive.relative_to(conformance / "fixtures").as_posix(),
              "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "size": archive.stat().st_size}
    inactive_path = conformance / "fixtures/toolcalling/retained-old.tar.gz"
    inactive_path.write_bytes(b"retained historical bytes")
    inactive = {"path": inactive_path.relative_to(conformance / "fixtures").as_posix(),
                "sha256": hashlib.sha256(inactive_path.read_bytes()).hexdigest(), "size": inactive_path.stat().st_size,
                "disposition": "superseded", "reason": "retained historical evidence"}
    digest, size = unified_history.store_digest(store)
    manifest = {"snapshot": "fixture", "crates": {"dynamo-parsers-v2": "0.7.16"}, "peers": {},
                "shards": [legacy, {"path": "unified-history", "format": "unified-history", "sha256": digest, "size": size}],
                "inactive_shards": [inactive]}
    (conformance / "fixtures-manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(viewer_catalog, "_source_metadata", lambda root: {"implementations": [{"key": "dynamo_v1"}], "unified_capture_policy": {}})
    viewer_catalog.refresh_catalog(tmp_path)
    catalog = json.loads((conformance / "viewer-catalog.json").read_text())
    manifest["historical_catalog_sha256"] = viewer_catalog.historical_catalog_fingerprint(catalog)
    (conformance / "fixtures-manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(package_fixtures, "ROOT", tmp_path)
    monkeypatch.setattr(package_fixtures, "FIXTURES_DIR", conformance / "fixtures")
    monkeypatch.setattr(package_fixtures, "UNIFIED_HISTORY_DIR", store)
    return conformance, manifest


def test_unified_only_package_and_catalog_never_open_archives(tmp_path, monkeypatch):
    conformance, manifest = _catalog_root(tmp_path, monkeypatch)
    originals = {path.relative_to(conformance): path.read_bytes() for path in (conformance / "fixtures").rglob("*.tar.gz")}
    original_open = Path.open
    merge_options = []
    original_merge = package_fixtures.merge_shards

    def checked_merge(*args, **kwargs):
        merge_options.append(kwargs)
        return original_merge(*args, **kwargs)

    monkeypatch.setattr(package_fixtures, 'merge_shards', checked_merge)

    def guarded_open(path, *args, **kwargs):
        assert not path.name.endswith(".tar.gz"), f"Unified regeneration opened {path}"
        return original_open(path, *args, **kwargs)

    def reject_archive(*args, **kwargs):
        raise AssertionError("Unified regeneration parsed a historical archive")

    with monkeypatch.context() as scope:
        scope.setattr(Path, "open", guarded_open)
        scope.setattr(tarfile, "open", reject_archive)
        package_fixtures.package_snapshot("new", "measured PT", manifest["crates"], manifest["peers"], dry_run=False, prune=False, unified_only=True)
        assert viewer_catalog.refresh_catalog(tmp_path, check=True, unified_only=True)
        # The old full-catalog owner reaches the blocking sentinel on this input.
        with pytest.raises(AssertionError, match="opened|parsed"):
            viewer_catalog.build_catalog(tmp_path)
    assert {path.relative_to(conformance): path.read_bytes() for path in (conformance / "fixtures").rglob("*.tar.gz")} == originals
    current = json.loads((conformance / "fixtures-manifest.json").read_text())
    assert current == manifest
    assert merge_options[0]['reject_unbuilt_drift'] is True
    assert viewer_catalog.build_catalog(tmp_path) == viewer_catalog.build_catalog(tmp_path, unified_only=True)


@pytest.mark.parametrize("corruption", ["pin", "duplicate_pin", "revision", "descriptor", "member", "availability"])
def test_unified_only_catalog_rejects_unbound_historical_metadata(tmp_path, monkeypatch, corruption):
    conformance, manifest = _catalog_root(tmp_path, monkeypatch)
    path = conformance / "viewer-catalog.json"
    catalog = json.loads(path.read_text())
    if corruption == "pin":
        manifest["shards"][0]["sha256"] = "0" * 64
    elif corruption == "duplicate_pin":
        manifest["shards"].append(copy.deepcopy(manifest["shards"][0]))
    elif corruption == "revision":
        catalog["revision"] = "0" * 64
    else:
        if corruption == "descriptor":
            catalog["sources"]["archives"] = []
        elif corruption == "member":
            catalog["sources"]["archives"][0]["members"][0]["path"] = "wrong/TOOLCALLING.batch.1.yaml"
        elif corruption == "availability":
            catalog["sources"]["archives"][0]["unavailable"] = {"wrong": True}
        elif corruption == "candidate_version":
            catalog["sources"]["candidates"][0]["version"] = "10.0.8"
        else:
            catalog["sources"]["candidates"][0]["sources"] = ["conformance/fixtures/not-pinned.tar.gz"]
        content = {key: value for key, value in catalog.items() if key != "revision"}
        catalog["revision"] = hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    path.write_text(json.dumps(catalog))
    with pytest.raises(ValueError, match="retained"):
        viewer_catalog.retained_archive_catalog(tmp_path, manifest)


def test_unified_only_catalog_does_not_relabel_retained_releases(tmp_path, monkeypatch):
    conformance, manifest = _catalog_root(tmp_path, monkeypatch)
    before = json.loads((conformance / "viewer-catalog.json").read_text())
    manifest["crates"]["dynamo-parsers"] = "10.0.8"
    (conformance / "fixtures-manifest.json").write_text(json.dumps(manifest))
    after = viewer_catalog.build_catalog(tmp_path, unified_only=True)
    assert [candidate for candidate in before["sources"]["candidates"] if candidate["mode"] != "unified"] == [candidate for candidate in after["sources"]["candidates"] if candidate["mode"] != "unified"]
    assert after["sources"]["archives"] == before["sources"]["archives"]


def test_unified_only_packaging_rejects_archive_changes_and_pruning(tmp_path, monkeypatch):
    conformance, manifest = _catalog_root(tmp_path, monkeypatch)
    built = [manifest["shards"][1]]
    with pytest.raises(ValueError, match="prune"):
        package_fixtures.merge_shards(built, True, retained_manifest=manifest)
    with pytest.raises(ValueError, match="replace"):
        package_fixtures.merge_shards([manifest["shards"][0]], False, retained_manifest=manifest)
    changed = copy.deepcopy(manifest)
    changed["shards"][0]["size"] += 1
    with pytest.raises(ValueError, match="historical archive pins"):
        package_fixtures._validate_candidate_package(changed, conformance / "fixtures", conformance / "fixtures-unified-v2", retained_manifest=manifest)


@pytest.mark.parametrize("explicit_history", [False, True])
def test_regeneration_dispatches_only_unified_fixture_owners(tmp_path, explicit_history):
    tools = tmp_path / "commands"
    tools.mkdir()
    log = tmp_path / "commands.jsonl"
    command = f"#!{sys.executable}\n" + """import json
import os
import sys
from pathlib import Path
with Path(os.environ['REGEN_COMMAND_LOG']).open('a') as output:
    output.write(json.dumps({'tool': Path(sys.argv[0]).name, 'args': sys.argv[1:], 'fixtures': os.environ.get('CONFORMANCE_FIXTURES_ROOT')}) + '\\n')
if any('extract_fixtures.py' in value for value in sys.argv):
    raise SystemExit(99)
"""
    for name in ("python3", "cargo", "node"):
        path = tools / name
        path.write_text(command)
        path.chmod(0o755)
    environment = {key: os.environ[key] for key in ("PATH", "LANG", "LC_ALL") if key in os.environ}
    environment.update(PATH=str(tools) + os.pathsep + environment["PATH"], TMPDIR=str(tmp_path), REGEN_COMMAND_LOG=str(log))
    if explicit_history:
        environment["CONFORMANCE_HISTORY_SOURCE"] = str(tmp_path / "explicit-history")
    result = subprocess.run(["bash", "conformance/utils/regenerate_unified.sh"], cwd=ROOT, env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    commands = [json.loads(line) for line in log.read_text().splitlines()]
    materializations = [row for row in commands if "--materialize" in row["args"]]
    assert len(materializations) == 2
    assert commands.index(materializations[0]) < next(i for i, row in enumerate(commands) if row["tool"] == "cargo")
    for row in commands:
        if row["tool"] == "cargo":
            assert row["fixtures"].startswith(str(tmp_path))
        if any(name in row["args"] for name in ("conformance/utils/src/package_fixtures.py", "conformance/utils/src/viewer_catalog.py")):
            assert "--unified-only" in row["args"]


@pytest.mark.parametrize("retain_split", [False, True])
def test_regeneration_id_gate_includes_retained_independent_regressions(retain_split):
    script = (ROOT / "conformance/utils/regenerate_unified.sh").read_text()
    fragment = "expected = {" + script.split("expected = {", 1)[1].split("current_version =", 1)[0]
    store = unified_history.load_authored_store(ROOT / "conformance/fixtures-unified-v2")
    split_id = numbered_id("v1_split_parallel_tool_calls")
    if retain_split:
        store.families["gemma4"].cases[split_id] = {
            "scenario": "v1_split_parallel_tool_calls", "display_id": split_id, "lifecycle": "active",
        }
    namespace = {"golden": golden, "numbered_id": numbered_id, "_split_only_cases": _split_only_cases,
                 "canonical_unified_case_key": canonical_unified_case_key, "store": store}
    exec(fragment, namespace)
    canonical = {
        family.name: {canonical_unified_case_key(family.name, case["display_id"], case["scenario"])
                      for case in family.cases.values() if case["lifecycle"] == "active"}
        for family in store.families.values()
    }
    assert namespace["expected"] == canonical
    old_gate = {**namespace}
    exec(fragment.split('for family in golden.scenario_families("v1_split_parallel_tool_calls"):', 1)[0], old_gate)
    missing = {family: ids - old_gate["expected"][family] for family, ids in canonical.items() if ids - old_gate["expected"][family]}
    assert missing == ({"gemma4": {split_id}} if retain_split else {})
