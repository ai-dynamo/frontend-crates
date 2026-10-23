# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent qualification against the explicitly supplied preserved dataset."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

import capture_sources
from fixture_corpus import load_corpus, version_key
import resolve_fixtures
import resolve_reasoning_fixtures
import resolve_stream_fixtures
import unified_archives
import unified_history
from viewer_codec import typed_equal
import yaml


ROOT = Path(__file__).resolve().parents[3]


def test_preserved_history_matches_committed_baseline(history_source):
    pin = json.loads(capture_sources.read(ROOT, capture_sources.PIN))
    inventory = capture_sources.verify(history_source, pin)
    retained = {row["path"] for row in inventory["files"]}
    committed = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", pin["source_revision"],
                                         "conformance/fixtures", "conformance/fixtures-unified-v2", "conformance/fixtures-beam"],
                                        cwd=ROOT, text=True).splitlines()
    assert set(committed) <= retained, sorted(set(committed) - retained)
    process = subprocess.Popen(["git", "cat-file", "--batch"], cwd=ROOT,
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    lfs_objects = 0
    try:
        for row in inventory["files"]:
            process.stdin.write(f"{pin['source_revision']}:{row['path']}\n".encode())
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


def test_current_unified_is_byte_exact_per_family_latest(history_source):
    selected = {}

    def retain(name, data, archive):
        identity = unified_history.source_identity(name)
        if identity["kind"] == "inputs-and-golden":
            assert capture_sources.read(ROOT, capture_sources.STORE + "/" + name) == data
            return
        if identity["kind"] != "history":
            return
        key = identity["family"], identity["implementation"]
        order = unified_history._capture_release_sort_key(identity["version"])
        if key not in selected or selected[key][0] < order:
            selected[key] = order, name, data

    unified_archives.inventory(history_source / capture_sources.STORE, member_callback=retain)
    current = [row for row in capture_sources.current(ROOT)["captures"] if row["surface"] == "unified"]
    assert {(row["family"], row["implementation"]) for row in current} == set(selected)
    records = 0
    for row in current:
        assert Path(row["path"]).stem == row["implementation"]
        _order, name, original = selected[row["family"], row["implementation"]]
        assert row["source"] == capture_sources.STORE + "/" + name
        assert capture_sources.read(ROOT, row["path"]) == original
        snapshot = yaml.safe_load(original)
        records += len(snapshot["records"])
    print(f"current Unified preservation passed: {len(current)} byte-exact snapshots, {records} complete records")


def test_current_legacy_matches_existing_resolvers(history_source, tmp_path):
    inventory = json.loads(capture_sources.read(history_source, capture_sources.INVENTORY))
    stage = tmp_path / "originals"
    originals = {}
    for row in inventory["files"]:
        if not row["path"].startswith("conformance/fixtures/") or "members" not in row:
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
            assert (ROOT / "conformance/inputs" / mode / path.relative_to(corpus_root / "inputs")).read_bytes() == path.read_bytes()
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
