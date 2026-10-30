# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent qualification against bundled or explicitly supplied preserved evidence."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import capture_sources
import migrate_capture_sources
from fixture_corpus import load_corpus, version_key
import fixture_snapshot
import resolve_fixtures
import resolve_reasoning_fixtures
import resolve_stream_fixtures
import unified_history
from viewer_codec import typed_equal
import yaml
import shutil
import pytest
import capture_sources as sources
import ci_publication_cache
import publish_current_capture
from test_unified_history import _store


ROOT = Path(__file__).resolve().parents[3]


def test_preserved_history_matches_committed_baseline(history_source):
    pin = capture_sources.history_pin(ROOT, history_source)
    inventory = capture_sources.verify(history_source, pin)
    if pin["source_revision"] == "retained-paths":
        pin = {**pin, "source_revision": "HEAD"}
    retained = {row["path"] for row in inventory["files"]}
    migration_path = history_source / "evidence/history-migration.json"
    identities = []
    inventory_by_path = {row["path"]: row for row in inventory["files"]}
    if migration_path.is_file():
        migration = json.loads(migration_path.read_bytes())
        old_inventory = json.loads((history_source / "evidence/complete-history-inventory.json").read_bytes())
        replaced = set(migration["replaced_archives"])
        for row in old_inventory["files"]:
            if row["path"] not in replaced:
                identities.append((row, migration["frozen_revision"], row["path"]))
        assert len(replaced) == 401
        for dataset in migration["datasets"]:
            committed = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", dataset["revision"],
                                                 capture_sources.STORE], cwd=ROOT, text=True).splitlines()
            for name in committed:
                physical = f"{dataset['root']}/{name}"
                row = inventory_by_path[physical]
                identities.append((row, dataset["revision"], name))
    else:
        committed = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", pin["source_revision"],
                                             "conformance/fixtures", "conformance/fixtures-unified-v2", "conformance/fixtures-beam"],
                                            cwd=ROOT, text=True).splitlines()
        committed = [name for name in committed if name.endswith(".tar.gz") or name.endswith("inputs_and_golden.yaml") or name.startswith(capture_sources.STORE + "/") and name.endswith(".yaml") and "-" in Path(name).stem]
        assert set(committed) <= retained, sorted(set(committed) - retained)
        identities = [(row, pin["source_revision"], row["path"]) for row in inventory["files"]]
    assert {row["path"] for row, _revision, _source in identities} <= retained
    process = subprocess.Popen(["git", "cat-file", "--batch"], cwd=ROOT,
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    lfs_objects = 0
    try:
        for row, revision, source in identities:
            process.stdin.write(f"{revision}:{source}\n".encode())
            process.stdin.flush()
            header = process.stdout.readline().decode().strip().split()
            assert len(header) == 3 and header[1] == "blob", row["path"]
            remaining = int(header[2])
            if remaining != row["size"]:
                assert remaining < 1024, row["path"]
                pointer = process.stdout.read(remaining).decode().splitlines()
                assert pointer == ["version https://git-lfs.github.com/spec/v1",
                                   "oid sha256:" + row["sha256"], "size " + str(row["size"])], row["path"]
                assert process.stdout.read(1) == b"\n"
                lfs_objects += 1
                continue
            digest = hashlib.sha256()
            while remaining:
                data = process.stdout.read(min(remaining, 1024 * 1024))
                assert data, row["path"]
                digest.update(data)
                remaining -= len(data)
            assert process.stdout.read(1) == b"\n"
            assert digest.hexdigest() == row["sha256"], row["path"]
        process.stdin.close()
        assert process.wait(timeout=30) == 0, process.stderr.read().decode()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    members = sum(len(row.get("members", [])) for row in inventory["files"])
    print(f"preservation passed: {len(inventory['files'])} committed files ({lfs_objects} LFS identities), {members} archive members")


def test_current_unified_preserves_resolved_latest_records(history_source):
    current = [row for row in capture_sources.current(ROOT)["captures"] if row["surface"] == "unified"]
    with capture_sources.history_view(ROOT, history_source) as view:
        if (view / capture_sources.DATASETS).is_file():
            store = unified_history.load_store(view / capture_sources.STORE)
            for row in current:
                history = store.histories[row["family"], row["implementation"]]
                selected = f"{row['implementation']}-{row['version']}"
                snapshot = yaml.safe_load(capture_sources.read(ROOT, row["path"]))
                assert typed_equal({case: record["value"] for case, record in snapshot["records"].items()}, history.resolve(selected))
                header, payload = capture_sources.read(ROOT, row["path"]).split(b"\n", 1)
                assert json.loads(header.removeprefix(b"capture_selection: "))["payload_sha256"] == hashlib.sha256(payload).hexdigest()
            print(f"current Unified preservation passed: {len(current)} independently bound snapshots")
            return
        store = unified_history.load_store(view / capture_sources.STORE)
        for row in current:
            snapshot = yaml.safe_load(capture_sources.read(ROOT, row["path"]))
            header, payload = capture_sources.read(ROOT, row["path"]).split(b"\n", 1)
            assert json.loads(header.removeprefix(b"capture_selection: "))["payload_sha256"] == hashlib.sha256(payload).hexdigest()
            snapshot.pop(capture_sources.SELECTION)
            family = store.families[row["family"]]
            assert set(snapshot["records"]) <= set(family.cases)
            if row["implementation"].startswith("dynamo"):
                # Refreshed local measurements intentionally replace old parser observations.
                for case, wrapped in snapshot["records"].items():
                    stimulus = wrapped["value"]["stimulus"]
                    assert "ref" not in stimulus
                    if "inline" in stimulus:
                        assert typed_equal(stimulus["inline"], family.cases[case]["request"])
                continue
            history = store.histories[row["family"], row["implementation"]]
            original = history.complete_snapshot(f"{row['implementation']}-{row['version']}", source_member=row["source"])
            authored = yaml.safe_load(capture_sources.read(view, f"{capture_sources.STORE}/families/{row['family']}/inputs_and_golden.yaml"))
            assert typed_equal(capture_sources.independent_snapshot(original, authored), snapshot), row["path"]


def test_current_legacy_matches_existing_resolvers(history_source, tmp_path):
    inventory = capture_sources.verify(history_source)
    stage = tmp_path / "originals"
    originals = {}
    for row in inventory["files"]:
        if not row["path"].startswith("conformance/fixtures/") or not row["path"].endswith(".tar.gz"):
            continue
        with tarfile.open(history_source / row["path"], "r:gz") as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith(".yaml"):
                    continue
                name = capture_sources.relative(member.name)
                data = archive.extractfile(member).read()
                target = stage / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                originals[name] = {**capture_sources.identity(name, data), "archive": row["path"]}
    current = [row for row in capture_sources.current(ROOT)["captures"] if row["surface"] != "unified"]
    expected = {}
    resolved = {}
    for mode, corpus_path in capture_sources.LEGACY_CORPORA.items():
        corpus_root = stage / corpus_path
        if mode == "batch-on-stream":
            for path in corpus_root.glob("*/*.yaml"):
                document = yaml.safe_load(path.read_bytes())
                for implementation, version in document.get("captured_with", {}).items():
                    expected.setdefault((mode, path.parent.name, implementation, str(version)), {})[path.name] = document
            continue
        corpus = load_corpus(corpus_root)
        for path in (corpus_root / "inputs").glob("*/*.yaml"):
            unpacked = ROOT / "conformance/inputs" / mode / path.relative_to(corpus_root / "inputs")
            if unpacked.exists():
                assert unpacked.read_bytes() == path.read_bytes()
        for implementation, versions in corpus.version_dirs.items():
            families = {family for _key, _version, directory in versions
                        for family, _name in corpus.by_directory[directory.name]}
            for family in families:
                _key, version, _directory = max((item for item in versions if family in corpus.families_by_directory[item[2].name]), key=lambda item: item[0])
                selection = mode, implementation, version
                if selection not in resolved:
                    if mode == "stream":
                        documents, _ = resolve_stream_fixtures.resolve_docs(corpus_root, [f"{implementation}-{version}"], corpus,
                                                                         implementations={implementation})
                    elif mode == "batch":
                        documents, _ = resolve_fixtures.resolve_docs(corpus_root, [f"{implementation}-{version}"], corpus)
                    else:
                        output = tmp_path / "reasoning" / implementation / version
                        resolve_reasoning_fixtures.resolve(corpus_root, output, [f"{implementation}-{version}"])
                        documents = {(path.parent.name, path.name): yaml.safe_load(path.read_bytes()) for path in output.glob("*/*.yaml")}
                    resolved[selection] = documents
                documents = resolved[selection]
                expected[mode, family, implementation, version] = {name: doc for (owner, name), doc in documents.items() if owner == family}
    if not current:
        extracted = fixture_snapshot.fixture_snapshot_root()
        actual = {path.relative_to(extracted).as_posix(): path for corpus_path in capture_sources.LEGACY_CORPORA.values()
                  for path in (extracted / corpus_path).rglob("*.yaml")}
        assert set(actual) == set(originals)
        for name, path in actual.items():
            assert capture_sources.identity(name, path.read_bytes()) == {key: originals[name][key] for key in ("path", "size", "sha256")}
        assert expected
        return
    assert {(row["surface"], row["family"], row["implementation"], row["version"]) for row in current} == set(expected)
    documents = 0
    cases = 0
    for row in current:
        assert Path(row["path"]).stem == row["implementation"]
        snapshot = yaml.safe_load(capture_sources.read(ROOT, row["path"]))
        identity = row["surface"], row["family"], row["implementation"], row["version"]
        assert typed_equal(snapshot["documents"], expected[identity]), row["path"]
        corpus_path = capture_sources.LEGACY_CORPORA[row["surface"]]
        dependencies = set()
        for name in originals:
            if not name.startswith(corpus_path + "/") or f"/{row['family']}/" not in name:
                continue
            directory = name.split("/")[len(corpus_path.split("/"))]
            if (row["surface"] == "batch-on-stream" or directory == "inputs"
                    or directory.startswith(row["implementation"] + "-")
                    and version_key(directory.split("-", 1)[1]) <= version_key(row["version"])):
                dependencies.add(name)
        assert {source["path"] for source in snapshot["source_records"]} == dependencies, row["path"]
        for source in snapshot["source_records"]:
            assert source == originals[source["path"]], row["path"]
        documents += len(snapshot["documents"])
        cases += sum(len(doc["cases"]) for doc in snapshot["documents"].values())
    print(f"current legacy parity passed: {len(current)} snapshots, {documents} documents, {cases} complete case records")


def test_frozen_interpretation_delta_preserves_union_fields_and_requests():
    request = {"input": "original"}
    first = {"value": {"stimulus": {"inline": request, "semantic_sha256": "digest"},
                       "observation": {"value": {}}, "document": {"producer": "older", "missing": None}},
             "member": "old.yaml", "suppress_inherited_from": True}
    original = {"value": {"stimulus": {"ref": "current", "semantic_sha256": "digest"},
                          "observation": {"value": {}}, "document": {"producer": "different", "null": None}},
                "member": "retained.yaml"}
    records = {"same": first, "deleted": first}
    frozen = {"schema_version": 2, "provenance": {"version": "old"},
              "execution_proof": {"receipt": {"opaque": b"bytes"}},
              "records": {"same": original, "added": original}}
    authored = {"cases": {key: {"request": request} for key in ("same", "added")}}
    delta = migrate_capture_sources.interpretation_delta(records, frozen, authored)
    restored = migrate_capture_sources.restore_interpretation(records, delta)
    assert typed_equal(restored, frozen)
    assert records["same"]["value"]["stimulus"] == {"inline": request, "semantic_sha256": "digest"}
    assert "deleted" not in restored["records"]
    assert "missing" not in restored["records"]["same"]["value"]["document"]
    assert restored["records"]["same"]["value"]["document"]["null"] is None


def retained_root(tmp_path):
    root = tmp_path / "checkout"
    shutil.copytree(_store(tmp_path / "sparse"), root / sources.STORE)
    (root / "conformance/fixtures-manifest.json").write_bytes(b'{"shards":[]}\n')
    return root


def test_retained_history_is_staged_separately_and_rechecked(tmp_path):
    root = retained_root(tmp_path)
    latest = root / sources.latest_path("unified", "gemma4", "dynamo_v2")
    latest.write_bytes(b"latest snapshot is not historical evidence\n")
    pin = sources.history_pin(root)
    evidence = sources.verify(root, pin)
    assert sources.default_history(root) == root
    assert pin["schema"] == 2
    assert latest.relative_to(root).as_posix() not in {row["path"] for row in evidence["files"]}
    destination = tmp_path / "prepared"
    destination.mkdir()
    manifest = sources.stage_history(root, destination, root)
    assert manifest == {"shards": []}
    dataset = json.loads((destination / sources.DATASETS).read_bytes())[0]
    assert not (destination / sources.STORE).exists()
    for row in evidence["files"]:
        if row["path"].startswith(sources.STORE + "/"):
            assert (destination / dataset["root"] / row["path"]).read_bytes() == (root / row["path"]).read_bytes()
    with sources.history_view(root, root) as view:
        assert not (view / sources.DATASETS).exists()
        for row in evidence["files"]:
            assert (view / row["path"]).read_bytes() == (root / row["path"]).read_bytes()
    historical = next((root / sources.STORE / "families").glob("*/*-*.yaml"))
    historical.write_bytes(historical.read_bytes() + b"# changed historical evidence\n")
    with pytest.raises(ValueError, match="retained history changed"):
        sources.verify(root, pin)


def test_retained_archival_fails_before_mutation(tmp_path, monkeypatch):
    root = retained_root(tmp_path)
    monkeypatch.setattr(publish_current_capture, "atomic_write", lambda *_args, **_kwargs: pytest.fail("unexpected archival write"))
    with pytest.raises(ValueError, match="read-only.*deferred"):
        publish_current_capture.publish(root, root, tmp_path / "not-read.yaml", "gemma4", "dynamo_v2", "0.5.3")
    assert not (root / "additions").exists()
    assert not sources.pending_updates(root)


def test_latest_publication_cache_excludes_in_place_history():
    prefix = sources.STORE + "/families/gemma4/"
    assert ci_publication_cache.is_publication_input(prefix + "inputs_and_golden.yaml")
    assert ci_publication_cache.is_publication_input(prefix + "dynamo_v2.yaml")
    assert not ci_publication_cache.is_publication_input(prefix + "dynamo_v2-0.5.2.yaml")
    assert not ci_publication_cache.is_publication_input("conformance/fixtures/toolcalling/fixtures-stream-v1/dynamo_v2-0.5.2.tar.gz")
