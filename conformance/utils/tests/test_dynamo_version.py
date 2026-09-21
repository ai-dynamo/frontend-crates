# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import dynamo_version as identity  # noqa: E402


def git(repo, *args, input=None):
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        input=input,
        check=True,
        capture_output=True,
        env=identity.git_subprocess_env(),
        text=True,
    ).stdout.strip()


@pytest.fixture
def release_repo(tmp_path, monkeypatch):
    monkeypatch.delenv(identity.ENV_OVERRIDE, raising=False)
    git(tmp_path, "init", "-q")
    for name, contents in {
        "parsers/v2/Cargo.toml": '[package]\nname="dynamo-parsers-v2"\nversion="0.6.0"\n',
        "parsers/v2/src/lib.rs": "pub fn parser() {}\n",
        "parsers/v1/src/lib.rs": "pub fn reasoning() {}\n",
        "protocols/src/lib.rs": "pub struct Tool;\n",
        "Cargo.toml": "[workspace]\n",
        "Cargo.lock": "version = 4\n",
    }.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
    git(tmp_path, "add", ".")
    commit = git(
        tmp_path,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit-tree",
        git(tmp_path, "write-tree"),
        input="fixture\n",
    )
    git(tmp_path, "update-ref", "HEAD", commit)
    git(tmp_path, "update-ref", "refs/tags/dynamo-parsers-v2-v0.6.0", commit)
    return tmp_path


def test_release_capture_uses_a_semantic_label_and_compact_origin(release_repo):
    provenance = identity.dynamo_v2_provenance(release_repo)

    assert identity.dynamo_v2_label(release_repo) == "0.6.0"
    assert identity.select_capture_label(release_repo, {}) == "0.6.0"
    assert identity.validate_capture_provenance(release_repo, provenance) == {
        "crate_version": "0.6.0",
        "source_sha256": identity.source_fingerprint(release_repo),
        "git_commit": provenance["git_commit"],
    }


def test_changed_same_version_is_capturable_but_not_a_new_consumer_identity(release_repo):
    (release_repo / "parsers/v2/src/lib.rs").write_text("pub fn changed() {}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="does not match release tag"):
        identity.dynamo_v2_provenance(release_repo, "0.6.0")
    producer = identity.dynamo_v2_provenance(release_repo, "current")

    assert producer["label"].startswith("0.6.0+source.")
    assert identity.dynamo_v2_label(release_repo) == "0.6.0"
    assert identity.select_capture_label(release_repo, {"0.6.0": [producer]}) == "0.6.0"


def test_reader_keeps_legacy_capture_directories_readable(release_repo):
    assert identity.select_capture_label(release_repo, {"0.6.0.patch2": []}) == "0.6.0.patch2"
    assert identity.select_capture_label(
        release_repo,
        {"0.6.0": {"records": {"gemma4/UNIFIED.1-1": {"format": "schema_v3"}}}},
    ) == "0.6.0"

    (release_repo / "parsers/v2/src/lib.rs").write_text("pub fn changed() {}\n", encoding="utf-8")
    source_label = identity.dynamo_v2_provenance(release_repo, "current")["label"]
    assert identity.select_capture_label(release_repo, {source_label: []}) == source_label


def test_reader_rejects_an_unverified_legacy_release_in_a_tagless_checkout(release_repo, monkeypatch):
    recorded = identity.dynamo_v2_provenance(release_repo)
    git(release_repo, "tag", "-d", "dynamo-parsers-v2-v0.6.0")

    current = identity.dynamo_v2_provenance(release_repo)
    assert current["label"].startswith("0.6.0+source.")
    captures = {
        "0.6.0": {"records": {"gemma4/UNIFIED.1-1": recorded}},
    }
    assert identity.select_capture_label(release_repo, captures) == "0.6.0"
    monkeypatch.setenv(identity.ENV_OVERRIDE, "current")
    assert identity.select_capture_label(release_repo, captures) == current["label"]
    monkeypatch.delenv(identity.ENV_OVERRIDE)
    wrong = {**recorded, "source_id": "wrong"}
    assert identity.select_capture_label(
        release_repo,
        {
            "0.6.0": {"records": {"gemma4/UNIFIED.1-1": recorded}},
            "0.6.0.patch1": {"records": {"gemma4/UNIFIED.1-1": wrong}},
        },
    ) == current["label"]


def test_capture_origin_rejects_a_different_producer_source(release_repo):
    recorded = identity.dynamo_v2_provenance(release_repo)
    recorded["source_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="source identity differs"):
        identity.validate_capture_provenance(release_repo, recorded)


def test_cli_select_capture_reports_only_the_semantic_version(release_repo):
    result = subprocess.run(
        [
            sys.executable,
            identity.__file__,
            "--repo-root",
            str(release_repo),
            "--format",
            "label",
            "--select-capture",
        ],
        input="{}",
        text=True,
        check=True,
        capture_output=True,
        env=identity.git_subprocess_env(),
    )

    assert result.stdout.strip() == "0.6.0"
