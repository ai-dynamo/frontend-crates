# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import sys
from pathlib import Path

import pytest
import yaml

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import extract_fixtures
import capture_stimulus
import fixture_disposition
import generate_conformance_table as table
import package_fixtures


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    conf = tmp_path / "conformance"
    store = conf / "fixtures"
    false_path = "unified/dynamo_v2-0.6.0.tar.gz"
    path = store / false_path
    path.parent.mkdir(parents=True)
    path.write_bytes(b"original mislabeled evidence")
    inactive = {"path": false_path, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size": path.stat().st_size, "disposition": "quarantined", "reason": "wrong producer"}
    manifest = {"snapshot": "test", "shards": [], "inactive_shards": [inactive]}
    manifest_path = conf / "fixtures-manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(package_fixtures, "ROOT", tmp_path)
    monkeypatch.setattr(package_fixtures, "FIXTURES_DIR", store)
    monkeypatch.setattr(extract_fixtures, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(extract_fixtures, "FIXTURES_DIR", store)
    monkeypatch.setattr(extract_fixtures, "get_cache_root", lambda: tmp_path / "cache")
    return conf, store, manifest, manifest_path


def _case(base, directory, key, record):
    path = base / directory / "gemma4" / f"{key}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"family": "gemma4", "mode": "unified", "cases": {key: record}}))


def test_package_preserves_quarantined_bytes_even_with_prune(evidence, tmp_path, monkeypatch):
    conf, store, manifest, manifest_path = evidence
    _case(conf / "unified", "dynamo_v2-0.6.0", "UNIFIED.gemma-1", {"assembled": []})
    _case(conf / "unified", "dynamo_v2-0.6.0.patch3", "UNIFIED.gemma-1", {"assembled": []})
    monkeypatch.setattr(package_fixtures, "read_versions", lambda: ({}, {}))
    monkeypatch.setattr(sys, "argv", ["package_fixtures.py", "--prune"])
    package_fixtures.main()
    after = json.loads(manifest_path.read_text())
    assert after["inactive_shards"] == manifest["inactive_shards"]
    assert [s["path"] for s in after["shards"]] == ["unified/dynamo_v2-0.6.0.patch3.tar.gz"]
    fixture_disposition.verify_inactive_shards(after, store)


def test_extract_excludes_false_release_but_retains_real_patch(evidence, tmp_path, monkeypatch, capsys):
    conf, store, manifest, manifest_path = evidence
    staging = tmp_path / "staging"
    _case(staging / "unified", "dynamo_v2-0.6.0.patch3", "UNIFIED.gemma-1", {"assembled": []})
    rel = "unified/dynamo_v2-0.6.0.patch3"
    sha, size = package_fixtures._tar_dir(staging / rel, rel, store / f"{rel}.tar.gz")
    manifest["shards"] = [{"path": f"{rel}.tar.gz", "sha256": sha, "size": size}]
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(sys, "argv", ["extract_fixtures.py"])
    extract_fixtures.main()
    snapshot = Path(capsys.readouterr().out.strip().splitlines()[-1])
    assert not (snapshot / "unified/dynamo_v2-0.6.0").exists()
    assert (snapshot / rel / "gemma4/UNIFIED.gemma-1.yaml").is_file()
    assert fixture_disposition.inactive_fixture_dirs(snapshot / "unified") == {"dynamo_v2-0.6.0"}
    # A warm cache must still reject loss or replacement of the inactive evidence.
    (store / manifest["inactive_shards"][0]["path"]).write_bytes(b"replacement")
    with pytest.raises(ValueError, match="pinned bytes"):
        extract_fixtures.main()


def test_quarantined_evidence_cannot_be_reactivated_or_rebuilt(evidence, tmp_path):
    _conf, _store, manifest, _path = evidence
    shard = {key: manifest["inactive_shards"][0][key] for key in ("path", "sha256", "size")}
    with pytest.raises(ValueError, match="also active"):
        fixture_disposition.active_shards(manifest | {"shards": [shard]})
    with pytest.raises(ValueError, match="cannot activate"):
        package_fixtures.merge_shards([shard], prune=True)
    with pytest.raises(ValueError, match="cannot overwrite"):
        package_fixtures.sync_store(tmp_path, [shard], dry_run=False, prune=False)


def test_inactive_only_change_publishes_new_immutable_generation(evidence, tmp_path, monkeypatch, capsys):
    _conf, store, manifest, manifest_path = evidence
    staging = tmp_path / "staging"
    _case(staging / "unified", "inputs", "UNIFIED.gemma-1", {"input": "text"})
    sha, size = package_fixtures._tar_dir(staging / "unified/inputs", "unified/inputs", store / "unified/inputs.tar.gz")
    manifest["shards"] = [{"path": "unified/inputs.tar.gz", "sha256": sha, "size": size}]
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(sys, "argv", ["extract_fixtures.py"])
    extract_fixtures.main()
    original = Path(capsys.readouterr().out.strip().splitlines()[-1])
    before = (original / ".fixtures-state.json").read_bytes()
    manifest["inactive_shards"][0]["reason"] = "updated evidence assessment"
    manifest_path.write_text(json.dumps(manifest))
    extract_fixtures.main()
    revised = Path(capsys.readouterr().out.strip().splitlines()[-1])
    assert revised != original
    assert (original / ".fixtures-state.json").read_bytes() == before
    assert json.loads((revised / ".fixtures-state.json").read_text())["inactive_shards"] == manifest["inactive_shards"]
    assert package_fixtures._extracted_snapshot_dir() == revised


@pytest.mark.parametrize("change", [
    {"path": "unified/another.tar.gz"}, {"sha256": "0" * 64}, {"size": 100},
    {"reason": "new assessment"}, {"disposition": "superseded"},
])
def test_every_inactive_identity_field_invalidates_cache(evidence, change):
    _conf, _store, manifest, _path = evidence
    old = manifest["inactive_shards"]
    changed = [old[0] | change]
    assert extract_fixtures.fixtures_identity([], old) != extract_fixtures.fixtures_identity([], changed)
    assert not extract_fixtures._state_matches({"shards": {}, "inactive_shards": old}, {}, changed)


def test_inactive_order_does_not_change_identity(evidence):
    _conf, _store, manifest, _path = evidence
    first = manifest["inactive_shards"][0]
    second = first | {"path": "unified/other.tar.gz"}
    assert extract_fixtures.fixtures_identity([], [first, second]) == extract_fixtures.fixtures_identity([], [second, first])


@pytest.mark.parametrize("change", [
    {"path": "../outside.tar.gz"}, {"path": "/absolute.tar.gz"}, {"sha256": "bad"},
    {"disposition": "active"}, {"reason": ""}, {"size": -1},
])
def test_disposition_rejects_unpinned_or_ambiguous_evidence(evidence, change):
    _conf, _store, manifest, _path = evidence
    with pytest.raises(ValueError):
        fixture_disposition.inactive_shards({"inactive_shards": [manifest["inactive_shards"][0] | change]})


def test_existing_versioned_archive_cannot_be_overwritten(evidence, tmp_path):
    _conf, store, _manifest, _path = evidence
    path = store / "unified/dynamo_v2-0.3.4.patch2.tar.gz"
    path.write_bytes(b"historical bytes")
    shard = {"path": str(path.relative_to(store)), "sha256": "0" * 64, "size": 1}
    with pytest.raises(ValueError, match="immutable"):
        package_fixtures.sync_store(tmp_path, [shard], dry_run=False, prune=False)
    assert path.read_bytes() == b"historical bytes"


@pytest.mark.parametrize("change", ["added_case", "changed_stimulus", "removed_case"])
@pytest.mark.parametrize("prune", [False, True])
@pytest.mark.parametrize("version", ["0.6.0", "0.7.0-rc.1"])
def test_source_capture_corpus_change_appends_overlay(evidence, monkeypatch, capsys, change, prune, version):
    conf, store, _manifest, manifest_path = evidence
    label = version + "+source." + "a" * 64
    directory = f"dynamo_v2-{label}"
    monkeypatch.setattr(package_fixtures, "read_versions", lambda: ({}, {}))
    monkeypatch.setattr(sys, "argv", ["package_fixtures.py"] + (["--prune"] if prune else []))

    def write_case(key, text):
        stimulus = {"input": text, "tools": [], "chunks": [{"delta_text": text}]}
        _case(conf / "unified", "inputs", key, stimulus)
        _case(conf / "unified", directory, key, {
            "capture_input": capture_stimulus.capture_input(stimulus),
            "assembled": [{"kind": "text", "text": text}],
        })

    write_case("UNIFIED.1-1", "old")
    if change == "removed_case":
        write_case("UNIFIED.1-2", "retired")
    package_fixtures.main()
    original_path = store / "unified" / f"{directory}.tar.gz"
    original_bytes = original_path.read_bytes()
    key = "UNIFIED.1-2" if change == "added_case" else "UNIFIED.1-1"
    if change == "removed_case":
        for tree in ("inputs", directory):
                (conf / "unified" / tree / "gemma4/UNIFIED.1-2.yaml").unlink()
    else:
        write_case(key, "new")
    package_fixtures.main()
    assert original_path.read_bytes() == original_bytes
    patch_path = store / "unified" / f"{directory}.patch1.tar.gz"
    assert patch_path.is_file()
    manifest_after = json.loads(manifest_path.read_text())
    capture_paths = {s["path"] for s in manifest_after["shards"] if directory in s["path"]}
    assert capture_paths == {f"unified/{directory}.tar.gz", f"unified/{directory}.patch1.tar.gz"}
    package_fixtures.main()
    assert not (store / "unified" / f"{directory}.patch2.tar.gz").exists()
    assert original_path.read_bytes() == original_bytes
    capsys.readouterr()
    monkeypatch.setattr(sys, "argv", ["extract_fixtures.py"])
    extract_fixtures.main()
    snapshot = Path(capsys.readouterr().out.strip().splitlines()[-1])
    monkeypatch.setattr(table, "_unified_dynamo_label", lambda _captures: label)
    cases, _caps, versions = table._load_unified_fixtures(snapshot / "unified")
    assert versions["dynamo_v2_all"] == [label]
    expected = {"old", "new"} if change == "added_case" else {"old"} if change == "removed_case" else {"new"}
    assert {case["dynamo"][0]["text"] for case in cases} == expected
    assert all(not case["dynamo_failure"] for case in cases)
    if change == "removed_case":
        # Retain the old input as a control: it must not resurrect the base capture.
        _case(snapshot / "unified", "inputs", "UNIFIED.1-2", {
            "input": "retired", "tools": [], "chunks": [{"delta_text": "retired"}],
        })
        cases, _caps, _versions = table._load_unified_fixtures(snapshot / "unified")
        retired = next(case for case in cases if case["input"] == "retired")
        assert not retired["dynamo"]
        assert retired["dynamo_missing"]
        assert label not in retired["dynamo_by_ver"]


@pytest.mark.parametrize("records", [["missing.yaml"], [], ["a.yaml", "a.yaml"], [1]])
def test_complete_snapshot_rejects_corrupt_member_index(records):
    with pytest.raises(ValueError, match="capture snapshot"):
        fixture_disposition.capture_snapshot_members(
            json.dumps({"schema_version": 1, "records": records}).encode(), ["a.yaml"],
        )


def test_source_capture_patch_order_is_numeric_in_archives_and_renderer(evidence, monkeypatch):
    conf, store, _manifest, _manifest_path = evidence
    label = "0.6.0+source." + "b" * 64
    base = f"dynamo_v2-{label}"
    stimulus = {"input": "latest", "tools": [], "chunks": [{"delta_text": "latest"}]}
    _case(conf / "unified", "inputs", "UNIFIED.1-1", stimulus)
    names = [base, base + ".patch1", base + ".patch2", base + ".patch10"]
    for name in names:
        _case(conf / "unified", name, "UNIFIED.1-1", {
            "capture_input": capture_stimulus.capture_input(stimulus),
            "assembled": [{"kind": "text", "text": name}],
        })
        package_fixtures._tar_dir(conf / "unified" / name, f"unified/{name}", store / "unified" / f"{name}.tar.gz")
    assert [path.name.removesuffix(".tar.gz") for path in fixture_disposition.capture_archive_layers(store, f"unified/{base}")] == names
    monkeypatch.setattr(table, "_unified_dynamo_label", lambda _captures: label)
    cases, _caps, _versions = table._load_unified_fixtures(conf / "unified")
    assert cases[0]["dynamo"][0]["text"] == names[-1]


def test_loose_reader_excludes_only_quarantined_shard_and_preserves_pr_history(evidence, monkeypatch):
    conf, _store, _manifest, _path = evidence
    base = conf / "unified"
    key = "UNIFIED.gemma-1"
    _case(base, "inputs", key, {"scenario": "gemma4_guided_json_visible_call_prose_before_reasoning", "chunks": []})
    _case(base, "golden", key, {"assembled": []})
    for directory, text, cid in [
        ("dynamo_v2-0.6.0", "false", key),
        ("dynamo_v2-0.6.0.patch3", "released", key),
        ("dynamo_v2-0.3.4+pr166", "historical", "UNIFIED.31-29"),
    ]:
        _case(base, directory, cid, {"assembled": [{"kind": "text", "text": text}]})
    observed = {}

    def select(captures):
        observed.update(captures)
        return "0.6.0"

    monkeypatch.setattr(table, "_unified_dynamo_label", select)
    cases, _caps, versions = table._load_unified_fixtures(base)
    assert versions["dynamo_v2_all"] == ["0.3.4+pr166", "0.6.0"]
    assert cases[0]["dynamo"][0]["text"] == "released"
    assert cases[0]["dynamo_by_ver"]["0.3.4+pr166"]["assembled"][0]["text"] == "historical"
    assert "0.6.0" not in observed
    assert len(observed["0.6.0.patch3"]["records"]) == 1
    assert list(observed["0.3.4+pr166"]["records"].values()) == [None]
    assert not observed["0.3.4+pr166"]["complete_snapshot"]


@pytest.mark.parametrize(("family", "old", "new"), [
    ("gemma4", "UNIFIED.31-29", "UNIFIED.gemma-1"),
    ("gemma4", "UNIFIED.31-30", "UNIFIED.gemma-2"),
    ("qwen3", "UNIFIED.31.a", "UNIFIED.31-1"),
    ("gemma4", "UNIFIED.31.x", "UNIFIED.34-6"),
    ("qwen3", "UNIFIED.31-29", "UNIFIED.31-29"),
    ("qwen3", "UNIFIED.30.m", "UNIFIED.30-13"),
    ("qwen3", "UNIFIED.1.a", "UNIFIED.1-1"),
    ("muse_glimmer", "UNIFIED.31-26", "UNIFIED.muse-1"),
    ("gemma4", "UNIFIED.31.y", "UNIFIED.31.y"),
])
def test_historical_aliases_do_not_reassign_other_ids(family, old, new):
    assert fixture_disposition.historical_unified_case_key(family, old) == new


def test_conflicting_capture_aliases_fail_without_touching_files(tmp_path, monkeypatch):
    _case(tmp_path, "inputs", "UNIFIED.gemma-1", {"scenario": "alias", "chunks": []})
    _case(tmp_path, "dynamo_v2-0.3.4.patch2", "UNIFIED.31-29", {"assembled": []})
    _case(tmp_path, "dynamo_v2-0.3.4.patch2", "UNIFIED.g4-1", {"assembled": [{"kind": "text", "text": "conflict"}]})
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.yaml")}
    monkeypatch.setattr(table, "_unified_dynamo_label", lambda captures: "0.3.4")
    with pytest.raises(ValueError, match="conflicting historical aliases"):
        table._load_unified_fixtures(tmp_path)
    assert {p: p.read_bytes() for p in tmp_path.rglob("*.yaml")} == before
