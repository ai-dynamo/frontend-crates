# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Normal Unified regeneration must not open retained archive payloads."""
import copy
from contextlib import contextmanager
import json
import os
import subprocess
import sys
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


def test_golden_output_directory_keeps_default_bytes(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FRONTEND_CRATES_ROOT", str(ROOT))
    monkeypatch.setattr(golden, "__file__", str(tmp_path / "conformance/utils/src/gen_unified_golden.py"))
    default = tmp_path / "conformance/unified/golden_spec"
    custom = tmp_path / "prepared-golden"

    golden.main([])
    default_messages = capsys.readouterr().out.splitlines()
    golden.main(["--output", str(custom)])
    custom_messages = capsys.readouterr().out.splitlines()

    expected = {golden.FAM_FILE[family] for family in golden.FAMILIES}
    assert {path.name for path in default.iterdir()} == expected
    assert {path.name for path in custom.iterdir()} == expected
    assert {name: (default / name).read_bytes() for name in expected} == {
        name: (custom / name).read_bytes() for name in expected
    }
    assert len(default_messages) == len(custom_messages) == len(expected)
    for family, default_message, custom_message in zip(golden.FAMILIES, default_messages, custom_messages):
        suffix = f"({len(golden.build_cases(family))} cases)"
        assert default_message.endswith(suffix)
        assert custom_message.endswith(suffix)


def test_golden_output_directory_rejects_a_file(tmp_path):
    output = tmp_path / "output"
    output.write_text("keep existing file")
    with pytest.raises(FileExistsError):
        golden.main(["--output", str(output)])
    assert output.read_text() == "keep existing file"


def _catalog_root(tmp_path, monkeypatch):
    conformance = tmp_path / "conformance"
    store = conformance / "fixtures-unified-v2"
    _write_family(store)
    _write_capture(store, "0.7.16", {"text_only": _change()})
    digest, size = unified_history.store_digest(store)
    manifest = {"snapshot": "fixture", "crates": {"dynamo-parsers-v2": "0.7.16"}, "peers": {},
                "shards": [{"path": "unified-history", "format": "unified-history", "sha256": digest, "size": size}],
                "inactive_shards": []}
    (conformance / "fixtures-manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(viewer_catalog, "_source_metadata", lambda root: {"implementations": [{"key": "dynamo_v1"}], "unified_capture_policy": {}})
    viewer_catalog.refresh_catalog(tmp_path)
    monkeypatch.setattr(package_fixtures, "ROOT", tmp_path)
    monkeypatch.setattr(package_fixtures, "UNIFIED_HISTORY_DIR", store)
    return conformance, manifest


def test_unified_package_and_catalog_need_no_retired_store(tmp_path, monkeypatch):
    conformance, manifest = _catalog_root(tmp_path, monkeypatch)
    original = {path.relative_to(conformance): path.read_bytes() for path in (conformance / "fixtures-unified-v2").rglob("*.yaml")}
    package_fixtures.package_snapshot("new", "measured PT", manifest["crates"], manifest["peers"], dry_run=False)
    assert not (conformance / "fixtures").exists()
    assert json.loads((conformance / "fixtures-manifest.json").read_bytes()) == manifest
    assert {path.relative_to(conformance): path.read_bytes() for path in (conformance / "fixtures-unified-v2").rglob("*.yaml")} == original
    catalog = viewer_catalog.build_catalog(tmp_path)
    assert catalog["sources"]["archives"] == []
    assert all(row["mode"] == "unified" for row in catalog["sources"]["candidates"])
    assert catalog == viewer_catalog.build_catalog(tmp_path, unified_only=True)


def test_unified_packaging_rejects_missing_or_changed_history(tmp_path, monkeypatch):
    conformance, manifest = _catalog_root(tmp_path, monkeypatch)
    store = conformance / "fixtures-unified-v2"
    changed = copy.deepcopy(manifest)
    changed["shards"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="differs from manifest"):
        package_fixtures._validate_candidate_package(changed, store)
    with pytest.raises(ValueError, match="prune"):
        package_fixtures.package_snapshot("new", "PT", {}, {}, dry_run=True, prune=True)


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


def test_packaging_includes_a_capture_published_before_lock_acquisition(tmp_path, monkeypatch):
    conformance, manifest = _catalog_root(tmp_path, monkeypatch)
    store = conformance / "fixtures-unified-v2"
    original_generation = viewer_catalog.catalog_generation
    capture = store / "families/gemma4/dynamo_v2-0.7.17.yaml"

    @contextmanager
    def competing_publication(root, **kwargs):
        _write_capture(store, "0.7.17", {"text_only": _change()})
        digest, size = unified_history.store_digest(store)
        manifest["shards"][0].update(sha256=digest, size=size)
        (conformance / "fixtures-manifest.json").write_text(json.dumps(manifest))
        with original_generation(root, **kwargs) as generation:
            yield generation

    monkeypatch.setattr(viewer_catalog, "catalog_generation", competing_publication)
    package_fixtures.package_snapshot("new", "PT", manifest["crates"], manifest["peers"], dry_run=False)
    assert capture.is_file(), "packaging dropped the competing publisher's completed capture"
    committed = json.loads((conformance / "fixtures-manifest.json").read_bytes())
    digest, size = unified_history.store_digest(store)
    assert committed["shards"][0]["sha256"] == digest
    assert committed["shards"][0]["size"] == size
