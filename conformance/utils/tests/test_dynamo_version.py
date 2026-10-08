# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import dynamo_version as identity  # noqa: E402
import refresh_dynamo_captures  # noqa: E402


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
    assert identity.validate_capture_provenance(release_repo, provenance) == {
        "crate_version": "0.6.0",
        "source_sha256": identity.source_fingerprint(release_repo),
        "git_commit": provenance["git_commit"],
    }


def test_changed_released_version_cannot_be_published_as_a_capture(release_repo):
    (release_repo / "parsers/v2/src/lib.rs").write_text("pub fn changed() {}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="does not match release tag"):
        identity.dynamo_v2_provenance(release_repo, "0.6.0")
    producer = identity.dynamo_v2_provenance(release_repo, "current")

    assert producer["label"].startswith("0.6.0+source.")
    assert identity.dynamo_v2_label(release_repo) == "0.6.0"
    with pytest.raises(ValueError, match="already released with different source"):
        identity.validate_capture_provenance(release_repo, producer)


def test_capture_source_fingerprint_resolves_release_and_unpublished_labels(release_repo):
    released_source = identity.source_fingerprint(
        release_repo, "refs/tags/dynamo-parsers-v2-v0.6.0"
    )
    manifest = release_repo / "parsers/v2/Cargo.toml"
    manifest.write_text(manifest.read_text().replace('"0.6.0"', '"0.6.1"'))
    current_source = identity.source_fingerprint(release_repo)
    current_label = identity.dynamo_v2_provenance(release_repo, "current")["label"]

    assert identity.capture_source_fingerprint(release_repo, "0.6.0") == released_source
    assert identity.capture_source_fingerprint(release_repo, "0.6.0.patch1") == released_source
    assert identity.capture_source_fingerprint(release_repo, "0.6.1") == current_source
    assert identity.capture_source_fingerprint(release_repo, current_label) == current_source
    with pytest.raises(ValueError, match="no verifiable parser source"):
        identity.capture_source_fingerprint(release_repo, "0.6.2")


@pytest.mark.parametrize("version, changed, allowed", [
    ("0.6.0", False, True),
    ("0.6.0", True, False),
    ("0.6.1", True, True),
])
def test_stream_receipt_binds_released_or_unpublished_source(release_repo, monkeypatch, version, changed, allowed):
    # Receipts belong to capture publication, outside the parser fixture input/output schema.
    manifest = release_repo / "parsers/v2/Cargo.toml"
    manifest.write_text(manifest.read_text().replace('"0.6.0"', f'"{version}"'))
    if changed:
        (release_repo / "parsers/v2/src/lib.rs").write_text("pub fn changed() {}\n")
    tree = release_repo / "stream"
    input_file = tree / "inputs/glm47/TOOLCALLING.streamv1.7.yaml"
    input_file.parent.mkdir(parents=True)
    case_id = "TOOLCALLING.streamv1.7.g"
    input_file.write_text(yaml.safe_dump({"family": "glm47", "mode": "streamv1",
                                         "cases": {case_id: {"chunks": [{"delta_text": "payload"}]}}}))
    monkeypatch.setattr(refresh_dynamo_captures, "ROOT", release_repo)
    monkeypatch.setattr(refresh_dynamo_captures, "ensure_tree", lambda _name: tree)
    monkeypatch.setattr(refresh_dynamo_captures, "V2_FAMILIES", ["glm47"])
    monkeypatch.setattr(refresh_dynamo_captures, "run_bin",
                        lambda *_args: json.dumps({case_id: [{"deltas": [{"index": 0, "name": "call"}]}]}))
    receipt_path = release_repo / "receipt.json"
    label = identity.dynamo_v2_label(release_repo, "current")
    if not allowed:
        with pytest.raises(ValueError, match="source does not match release"):
            refresh_dynamo_captures.refresh_stream(label, receipt_path)
        assert not receipt_path.exists()
        assert not (tree / f"dynamo_v2-{version}").exists()
        return
    refresh_dynamo_captures.refresh_stream(label, receipt_path)
    receipt = json.loads(receipt_path.read_text())["captures"][f"dynamo_v2-{version}"]
    assert receipt["producer_source_sha256"] == identity.source_fingerprint(release_repo)
    output = yaml.safe_load((tree / f"dynamo_v2-{version}/glm47/TOOLCALLING.streamv1.7.yaml").read_text())
    assert output["captured_with"] == {"dynamo_v2": version}
    assert output["cases"][case_id]["chunks"][0]["expected"] == [{"index": 0, "name": "call"}]


def test_stream_receipt_rejects_another_version_before_touching_output(release_repo, monkeypatch):
    monkeypatch.setattr(refresh_dynamo_captures, "ROOT", release_repo)
    monkeypatch.setattr(refresh_dynamo_captures, "ensure_tree", lambda _name: pytest.fail("must reject before mutation"))
    with pytest.raises(ValueError, match="capture (version|label)"):
        refresh_dynamo_captures.refresh_stream("0.6.1", release_repo / "receipt.json")


def test_reader_keeps_legacy_capture_directories_readable(release_repo):
    assert identity.select_capture_label(release_repo, {"0.6.0.patch2": []}) == "0.6.0.patch2"
    assert identity.select_capture_label(
        release_repo,
        {"0.6.0": {"records": {"gemma4/UNIFIED.1-1": {"format": "schema_v3"}}}},
    ) == "0.6.0"

    manifest = release_repo / "parsers/v2/Cargo.toml"
    manifest.write_text(manifest.read_text().replace('"0.6.0"', '"0.6.1"'))
    (release_repo / "parsers/v2/src/lib.rs").write_text("pub fn changed() {}\n", encoding="utf-8")
    git(release_repo, "add", "parsers/v2/Cargo.toml", "parsers/v2/src/lib.rs")
    git(
        release_repo,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "capture source",
    )
    commit = git(release_repo, "rev-parse", "--verify", "HEAD")
    recorded = identity.dynamo_v2_provenance(release_repo, "current")
    assert recorded["kind"] == "unpublished"
    assert recorded["git_commit"] == commit
    assert identity.validate_capture_provenance(release_repo, recorded)["crate_version"] == "0.6.1"


def test_legacy_capture_selection_does_not_mutate_capture_inventory(release_repo):
    captures = {
        "0.6.0": [{"format": "legacy"}],
        "0.6.0.patch2": [{"format": "patch"}],
    }
    original = {label: list(records) for label, records in captures.items()}

    assert identity.select_capture_label(release_repo, captures) == "0.6.0"
    assert captures == original


def test_capture_origin_rejects_a_different_producer_source(release_repo):
    recorded = identity.dynamo_v2_provenance(release_repo)
    recorded["source_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="source identity differs"):
        identity.validate_capture_provenance(release_repo, recorded)


def test_cli_reports_only_the_semantic_version(release_repo):
    result = subprocess.run(
        [
            sys.executable,
            identity.__file__,
            "--repo-root",
            str(release_repo),
            "--format",
            "label",
        ],
        text=True,
        check=True,
        capture_output=True,
        env=identity.git_subprocess_env(),
    )

    assert result.stdout.strip() == "0.6.0"


@pytest.mark.parametrize("override", [None, "current", "0.6.0"])
def test_normal_version_lookup_ignores_tags_source_and_git_environment(release_repo, monkeypatch, override):
    git(release_repo, "tag", "-d", "dynamo-parsers-v2-v0.6.0")
    (release_repo / "parsers/v2/src/lib.rs").write_text("pub fn changed() {}\n", encoding="utf-8")
    monkeypatch.setattr(identity, "source_fingerprint", lambda *args: pytest.fail("reader must not fingerprint source"))
    assert identity.dynamo_v2_label(release_repo, override) == "0.6.0"
    result = subprocess.run(
        [sys.executable, identity.__file__, "--repo-root", str(release_repo), "--format", "label"],
        text=True, check=True, capture_output=True, env=identity.git_subprocess_env(),
    )
    assert result.stdout.strip() == "0.6.0"


@pytest.mark.parametrize("label", ["", "0.6.0.patch1", "0.6.0+source." + "a" * 64, "0.6.1"])
def test_normal_version_lookup_rejects_noncanonical_labels(release_repo, label):
    with pytest.raises(ValueError, match="capture version must be"):
        identity.dynamo_v2_label(release_repo, label)
