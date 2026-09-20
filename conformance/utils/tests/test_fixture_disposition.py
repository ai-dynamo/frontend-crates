# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import extract_fixtures  # noqa: E402
import capture_stimulus  # noqa: E402
import fixture_disposition  # noqa: E402
import generate_conformance_table as table  # noqa: E402
import package_fixtures  # noqa: E402
import unified_history  # noqa: E402


def test_canonicalize_unified_inputs_rejects_duplicate_scenario_owners():
    records = {
        ("gemma4", "UNIFIED.1-1"): {"scenario": "text_only"},
        ("gemma4", "UNIFIED.9-9"): {"scenario": "text_only"},
    }

    with pytest.raises(ValueError, match="duplicate Unified scenario ownership"):
        fixture_disposition.canonicalize_unified_inputs(records)


def test_canonicalize_unified_inputs_allows_one_historical_alias_rename():
    scenario = "gemma4_guided_json_visible_call_prose_before_reasoning"
    records = {
        ("gemma4", "UNIFIED.31-29"): {"scenario": scenario, "input": "old"},
        ("gemma4", "UNIFIED.g4-1"): {"scenario": scenario, "input": "new"},
    }

    canonical, aliases = fixture_disposition.canonicalize_unified_inputs(records)

    ident = (
        "gemma4",
        fixture_disposition.canonical_unified_case_key(
            "gemma4", "UNIFIED.g4-1", scenario
        ),
    )
    assert canonical == {ident: records[("gemma4", "UNIFIED.g4-1")]}
    assert aliases[("gemma4", "UNIFIED.31-29")] == ident
    assert aliases[("gemma4", "UNIFIED.g4-1")] == ident


def test_checked_in_manifest_pins_unified_history_store():
    repo_root = SRC.parents[2]
    manifest = json.loads((repo_root / "conformance/fixtures-manifest.json").read_text())
    pinned = next(shard for shard in manifest["shards"] if shard.get("format") == "unified-history")

    digest, size = unified_history.store_digest(repo_root / "conformance/fixtures-unified-v2")

    assert pinned["sha256"] == digest
    assert pinned["size"] == size


def test_checked_in_manifest_retains_inactive_unified_evidence():
    repo_root = SRC.parents[2]
    manifest = json.loads((repo_root / "conformance/fixtures-manifest.json").read_text())

    inactive = fixture_disposition.verify_inactive_shards(
        manifest,
        repo_root / "conformance/fixtures",
    )

    assert inactive
    assert all(path.startswith("unified/") for path in inactive)


def test_checked_in_manifest_tracks_inactive_unified_evidence():
    repo_root = SRC.parents[2]
    manifest = json.loads((repo_root / "conformance/fixtures-manifest.json").read_text())
    inactive_paths = sorted(fixture_disposition.inactive_shards(manifest))

    result = subprocess.run(
        ["git", "-C", str(repo_root), "ls-files", "--error-unmatch", "--", *(
            f"conformance/fixtures/{path}" for path in inactive_paths
        )],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


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


def test_package_pins_unified_history_without_writing_an_archive(evidence, tmp_path, monkeypatch):
    _conf, _store, _manifest, _manifest_path = evidence
    history_root = tmp_path / "fixtures-unified-v2"
    family = {
        "schema_version": 2,
        "family": "gemma4",
        "input_document": {"family": "gemma4", "mode": "unified"},
        "golden_document": {"family": "gemma4", "mode": "unified"},
        "cases": {
            "text_only": {
                "lifecycle": "active",
                "scenario": "text_only",
                "description": "Visible text",
                "policy": ["test-policy"],
                "display_id": "UNIFIED.1-1",
                "historical_ids": [],
                "request": {
                    "input": "hello",
                    "init": {},
                    "finish_reason": "stop",
                    "tools": [],
                    "chunks": [],
                },
                "golden": {"assembled": []},
            }
        },
    }
    capture = {
        "schema_version": 2,
        "family": "gemma4",
        "implementation": "dynamo_v2",
        "parent": None,
        "runtime_version": "0.1.0",
        "provenance": {"status": "legacy", "captured_with": {"dynamo_v2": "0.1.0"}},
        "completeness": "snapshot",
        "document": {},
        "import_lineage": [],
        "changes": {},
        "metadata_changes": {},
        "document_overrides": {},
    }
    (history_root / "families/gemma4").mkdir(parents=True)
    (history_root / "families/gemma4/inputs_and_golden.yaml").write_text(
        unified_history.dump_yaml(family)
    )
    (history_root / "families/gemma4/dynamo_v2-0.1.0.yaml").write_text(
        unified_history.dump_yaml(capture)
    )
    monkeypatch.setattr(package_fixtures, "UNIFIED_HISTORY_DIR", history_root)
    monkeypatch.setattr(package_fixtures, "PER_SUBDIR_TREES", ("unified",))
    monkeypatch.setattr(package_fixtures.dynamo_version, "dynamo_v2_label", lambda _root: "0.1.0")

    stage = tmp_path / "stage"
    _case(
        stage / "unified",
        "inputs",
        "UNIFIED.1-1",
        {
            "scenario": "text_only",
            "description": "Visible text",
            "policy": ["test-policy"],
            **family["cases"]["text_only"]["request"],
        },
    )
    _case(stage / "unified", "golden", "UNIFIED.1-1", {"assembled": []})
    monkeypatch.setattr(unified_history, "update_store_from_loose", lambda *_args, **_kwargs: [])
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    shards = package_fixtures.build_shards(stage, blobs)

    digest, size = unified_history.store_digest(history_root)
    assert shards == [{"path": "unified-history", "format": "unified-history", "sha256": digest, "size": size}]
    assert not list(blobs.rglob("*.tar.gz"))


def test_package_dry_run_does_not_update_unified_history(evidence, tmp_path, monkeypatch):
    _conf, _store, _manifest, _manifest_path = evidence
    history_root = tmp_path / "fixtures-unified-v2"
    (history_root / "families").mkdir(parents=True)
    (history_root / "families/gemma4").mkdir(parents=True)
    monkeypatch.setattr(package_fixtures, "UNIFIED_HISTORY_DIR", history_root)
    monkeypatch.setattr(package_fixtures, "PER_SUBDIR_TREES", ("unified",))

    def mutate_history(
        store_root,
        _capture_root,
        *,
        complete_snapshot,
        excluded_capture_dirs,
        required_capture_dirs,
    ):
        assert complete_snapshot is False
        assert "dynamo_v2-0.6.0" in excluded_capture_dirs
        assert required_capture_dirs == frozenset()
        path = store_root / "families/gemma4/dynamo_v2-0.1.0.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("dry run must not write here")
        return [path]

    monkeypatch.setattr(unified_history, "update_store_from_loose", mutate_history)
    monkeypatch.setattr(unified_history, "store_digest", lambda _root: ("digest", 1))

    blobs = tmp_path / "blobs"
    blobs.mkdir()
    package_fixtures.build_shards(tmp_path / "stage", blobs, dry_run=True)

    assert not (history_root / "families/gemma4/dynamo_v2-0.1.0.yaml").exists()


@pytest.mark.parametrize(
    ("current_roots", "complete_snapshot"),
    [
        ((), False),
        (("inputs", "golden"), True),
    ],
)
def test_build_shards_passes_exact_inactive_unified_capture_directories(
    tmp_path,
    monkeypatch,
    current_roots,
    complete_snapshot,
):
    stage = tmp_path / "stage"
    (stage / "unified").mkdir(parents=True)
    for root in current_roots:
        (stage / "unified" / root).mkdir()
    history = tmp_path / "history"
    history.mkdir()
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    observed = {}
    inactive = {
        "unified/dynamo_v2-0.6.0.tar.gz": {},
        "unified/dynamo_v2-0.6.0.patch1.tar.gz": {},
        "unified/inputs+pr200.tar.gz": {},
        "toolcalling/fixtures.tar.gz": {},
    }

    def update_store(
        _store,
        _loose,
        *,
        complete_snapshot,
        excluded_capture_dirs,
        required_capture_dirs,
    ):
        observed["complete_snapshot"] = complete_snapshot
        observed["excluded_capture_dirs"] = excluded_capture_dirs
        observed["required_capture_dirs"] = required_capture_dirs
        return []

    monkeypatch.setattr(package_fixtures, "PER_SUBDIR_TREES", ("unified",))
    monkeypatch.setattr(package_fixtures, "preserved_evidence", lambda: inactive)
    monkeypatch.setattr(package_fixtures.dynamo_version, "dynamo_v2_label", lambda _root: "0.7.0")
    monkeypatch.setattr(unified_history, "update_store_from_loose", update_store)
    monkeypatch.setattr(unified_history, "store_digest", lambda _root: ("0" * 64, 0))

    package_fixtures.build_shards(stage, blobs, history_root=history)

    assert observed == {
        "complete_snapshot": complete_snapshot,
        "excluded_capture_dirs": {
            "dynamo_v2-0.6.0",
            "dynamo_v2-0.6.0.patch1",
        },
        "required_capture_dirs": {"dynamo_v2-0.7.0"} if complete_snapshot else frozenset(),
    }


@pytest.mark.parametrize("failure_stage", ["history", "archives", "manifest"])
def test_package_staging_failure_preserves_live_generation(
    tmp_path,
    monkeypatch,
    failure_stage,
):
    conformance = tmp_path / "conformance"
    fixtures = conformance / "fixtures"
    history = conformance / "fixtures-unified-v2"
    manifest_path = conformance / "fixtures-manifest.json"
    fixtures.mkdir(parents=True)
    history.mkdir()
    (fixtures / "generation").write_text("old archives")
    (history / "generation").write_text("old history")
    manifest_path.write_text(
        json.dumps({"snapshot": "old", "shards": [], "inactive_shards": []}) + "\n"
    )
    before = {
        "fixtures": (fixtures / "generation").read_bytes(),
        "history": (history / "generation").read_bytes(),
        "manifest": manifest_path.read_bytes(),
    }

    monkeypatch.setattr(package_fixtures, "ROOT", tmp_path)
    monkeypatch.setattr(package_fixtures, "FIXTURES_DIR", fixtures)
    monkeypatch.setattr(package_fixtures, "UNIFIED_HISTORY_DIR", history)
    monkeypatch.setattr(package_fixtures, "stage_fixtures", lambda _source, _destination: None)

    def build_shards(_loose, _blobs, _prune, *, history_root):
        if failure_stage == "history":
            (history_root / "generation").write_text("partial history")
            raise OSError("injected failure after history mutation")
        return []

    def sync_store(_blobs, _shards, _dry_run, _prune, *, fixtures_dir, manifest_path):
        if failure_stage == "archives":
            generation = fixtures_dir / "generation"
            generation.unlink()
            generation.write_text("partial archives")
            raise OSError("injected failure after archive mutation")

    def validate_candidate(_manifest, _fixtures_dir, _history_dir):
        if failure_stage == "manifest":
            raise OSError("injected failure after manifest write")

    monkeypatch.setattr(package_fixtures, "build_shards", build_shards)
    monkeypatch.setattr(package_fixtures, "sync_store", sync_store)
    monkeypatch.setattr(package_fixtures, "_validate_candidate_package", validate_candidate)

    with pytest.raises(OSError, match="injected failure after"):
        package_fixtures.package_snapshot(
            "new",
            "new America/Los_Angeles",
            {},
            {},
            dry_run=False,
            prune=False,
        )

    assert (fixtures / "generation").read_bytes() == before["fixtures"]
    assert (history / "generation").read_bytes() == before["history"]
    assert manifest_path.read_bytes() == before["manifest"]


def test_package_snapshots_loose_inputs_after_acquiring_writer_lock(tmp_path, monkeypatch):
    conformance = tmp_path / "conformance"
    fixtures = conformance / "fixtures"
    history = conformance / "fixtures-unified-v2"
    fixtures.mkdir(parents=True)
    history.mkdir()
    (conformance / "fixtures-manifest.json").write_text(
        json.dumps({"snapshot": "old", "shards": [], "inactive_shards": []}) + "\n"
    )
    monkeypatch.setattr(package_fixtures, "ROOT", tmp_path)
    monkeypatch.setattr(package_fixtures, "FIXTURES_DIR", fixtures)
    monkeypatch.setattr(package_fixtures, "UNIFIED_HISTORY_DIR", history)
    monkeypatch.setattr(package_fixtures, "build_shards", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(package_fixtures, "sync_store", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(package_fixtures, "_validate_candidate_package", lambda *_args: None)
    monkeypatch.setattr(unified_history, "publish_paths_transactionally", lambda *_args, **_kwargs: None)

    first_staged = threading.Event()
    second_staged = threading.Event()
    release_first = threading.Event()

    def stage(_source, _destination):
        if threading.current_thread().name == "first-packager":
            first_staged.set()
            assert release_first.wait(timeout=5)
        else:
            second_staged.set()

    monkeypatch.setattr(package_fixtures, "stage_fixtures", stage)
    errors = []

    def package():
        try:
            package_fixtures.package_snapshot(
                "new",
                "new America/Los_Angeles",
                {},
                {},
                dry_run=False,
                prune=False,
            )
        except Exception as error:
            errors.append(error)

    first = threading.Thread(target=package, name="first-packager")
    second = threading.Thread(target=package, name="second-packager")
    first.start()
    assert first_staged.wait(timeout=5)
    second.start()
    assert not second_staged.wait(timeout=0.2)
    release_first.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert errors == []
    assert not first.is_alive()
    assert not second.is_alive()
    assert second_staged.is_set()


def test_extract_holds_generation_lock_through_shard_materialization(tmp_path, monkeypatch):
    conformance = tmp_path / "conformance"
    fixtures = conformance / "fixtures"
    history = conformance / "fixtures-unified-v2"
    manifest_path = conformance / "fixtures-manifest.json"
    fixtures.mkdir(parents=True)
    history.mkdir()
    manifest_path.write_text(
        json.dumps(
            {
                "snapshot": "old",
                "shards": [
                    {
                        "path": "toolcalling/a.tar.gz",
                        "sha256": "a" * 64,
                        "size": 1,
                    }
                ],
                "inactive_shards": [],
            }
        )
        + "\n"
    )
    cache = tmp_path / "cache"
    monkeypatch.setattr(extract_fixtures, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(extract_fixtures, "FIXTURES_DIR", fixtures)
    monkeypatch.setattr(extract_fixtures, "HISTORY_DIR", history)
    monkeypatch.setattr(extract_fixtures, "get_cache_root", lambda: cache)
    reader_at_shard = threading.Event()
    release_reader = threading.Event()
    writer_acquired = threading.Event()

    def shard_file(_shard):
        reader_at_shard.set()
        assert release_reader.wait(timeout=5)
        return fixtures / "toolcalling/a.tar.gz"

    def materialize_shard(_shard, _source, destination, verbose=False):
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "materialized").write_text("old")

    monkeypatch.setattr(extract_fixtures, "shard_file", shard_file)
    monkeypatch.setattr(extract_fixtures, "materialize_shard", materialize_shard)
    monkeypatch.setattr(
        extract_fixtures,
        "publish_extracted_snapshot",
        lambda temporary, *_args, **_kwargs: temporary,
    )
    errors = []

    def read():
        try:
            extract_fixtures.extract_snapshot(
                SimpleNamespace(full_refresh=False, dry_run=False, info=False, verbose=False)
            )
        except Exception as error:
            errors.append(error)

    def write():
        try:
            with unified_history._store_mutation_lock(history):
                writer_acquired.set()
        except Exception as error:
            errors.append(error)

    reader = threading.Thread(target=read)
    writer = threading.Thread(target=write)
    reader.start()
    assert reader_at_shard.wait(timeout=5)
    writer.start()
    assert not writer_acquired.wait(timeout=0.2)
    release_reader.set()
    reader.join(timeout=5)
    writer.join(timeout=5)

    assert errors == []
    assert not reader.is_alive()
    assert not writer.is_alive()
    assert writer_acquired.is_set()


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


def test_inactive_archive_does_not_hide_capture_from_active_unified_history(evidence, tmp_path):
    conf, store, manifest, manifest_path = evidence
    manifest["shards"] = [
        {"path": "unified-history", "format": "unified-history", "sha256": "0" * 64, "size": 0}
    ]
    manifest_path.write_text(json.dumps(manifest))

    assert fixture_disposition.inactive_fixture_dirs(store / "unified") == set()

    snapshot = tmp_path / "snapshot"
    (snapshot / "unified").mkdir(parents=True)
    (snapshot / ".fixtures-state.json").write_text(
        json.dumps(
            {
                "shards": {"unified-history": "0" * 64},
                "inactive_shards": manifest["inactive_shards"],
            }
        )
    )
    assert fixture_disposition.inactive_fixture_dirs(snapshot / "unified") == set()


def test_quarantined_evidence_cannot_be_reactivated_or_rebuilt(evidence, tmp_path):
    _conf, _store, manifest, _path = evidence
    shard = {key: manifest["inactive_shards"][0][key] for key in ("path", "sha256", "size")}
    with pytest.raises(ValueError, match="also active"):
        fixture_disposition.active_shards(manifest | {"shards": [shard]})
    with pytest.raises(ValueError, match="cannot activate"):
        package_fixtures.merge_shards([shard], prune=True)
    with pytest.raises(ValueError, match="cannot overwrite"):
        package_fixtures.sync_store(tmp_path, [shard], dry_run=False, prune=False)


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


def test_identical_capture_aliases_are_accepted_with_cached_records(tmp_path, monkeypatch):
    _case(tmp_path, "inputs", "UNIFIED.gemma-1", {"scenario": "alias", "chunks": []})
    record = {"assembled": [{"kind": "text", "text": "same"}]}
    _case(tmp_path, "dynamo_v2-0.3.4.patch2", "UNIFIED.31-29", record)
    _case(tmp_path, "dynamo_v2-0.3.4.patch2", "UNIFIED.g4-1", record)
    monkeypatch.setattr(table, "_unified_dynamo_label", lambda captures: "0.3.4")

    cases, _captures, _versions = table._load_unified_fixtures(tmp_path)

    assert cases[0]["dynamo"][0]["text"] == "same"
