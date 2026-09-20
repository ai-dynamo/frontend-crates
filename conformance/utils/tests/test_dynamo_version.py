# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import dynamo_version as identity
import explode_unified_fixtures as explode
import generate_conformance_table as table
from capture_stimulus import capture_input


def git(repo, *args, input=None):
    return subprocess.run(
        ["git", "-C", str(repo), *args], input=input,
        check=True, capture_output=True, env=identity.git_subprocess_env(), text=True,
    ).stdout.strip()


@pytest.fixture
def release_repo(tmp_path, monkeypatch):
    monkeypatch.delenv(identity.ENV_OVERRIDE, raising=False)
    git(tmp_path, "init", "-q")
    files = {
        "parsers/v2/Cargo.toml": '[package]\nname="dynamo-parsers-v2"\nversion="0.6.0"\n',
        "parsers/v2/src/lib.rs": "pub fn parser() {}\n",
        "parsers/v1/src/lib.rs": "pub fn reasoning() {}\n",
        "protocols/src/lib.rs": "pub struct Tool;\n",
        "Cargo.toml": "[workspace]\n",
        "Cargo.lock": "version = 4\n",
    }
    for name, contents in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
    git(tmp_path, "add", ".")
    tree = git(tmp_path, "write-tree")
    commit = git(tmp_path, "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                 "commit-tree", tree, input="fixture\n")
    git(tmp_path, "update-ref", "HEAD", commit)
    git(tmp_path, "update-ref", "refs/tags/dynamo-parsers-v2-v0.6.0", commit)
    return tmp_path


def test_release_requires_exact_tag_source(release_repo):
    provenance = identity.dynamo_v2_provenance(release_repo)
    assert provenance["kind"] == "release"
    assert provenance["label"] == "0.6.0"
    assert provenance["git_commit"] == provenance["release_commit"]
    assert provenance["source_sha256"] == identity.source_fingerprint(release_repo, "HEAD")


@pytest.mark.parametrize("name", [
    "parsers/v2/src/lib.rs", "parsers/v2/src/new.rs", "parsers/v2/build.rs",
    "parsers/v1/src/lib.rs", "protocols/src/lib.rs", "Cargo.toml", "Cargo.lock",
    ".cargo/config.toml", "rust-toolchain.toml",
])
def test_same_version_changed_sources_are_unpublished(release_repo, name):
    path = release_repo / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("changed source\n")
    with pytest.raises(ValueError, match="does not match release tag"):
        identity.dynamo_v2_label(release_repo, "0.6.0")
    provenance = identity.dynamo_v2_provenance(release_repo, "current")
    assert provenance["kind"] == "unpublished"
    assert provenance["release_tag"] is None
    assert provenance["label"] == "0.6.0+source." + provenance["source_sha256"]
    assert identity.dynamo_v2_label(release_repo) == provenance["label"]
    assert provenance["source_id"] == "sha256:" + provenance["source_sha256"]
    assert provenance["git_head_tree"] == git(release_repo, "rev-parse", "HEAD^{tree}")
    assert identity.dynamo_v2_label(release_repo, provenance["label"]) == provenance["label"]


def test_staged_ignored_and_deleted_source_changes_are_not_releases(release_repo):
    path = release_repo / "parsers/v2/src/lib.rs"
    path.write_text("staged change\n")
    git(release_repo, "add", str(path))
    with pytest.raises(ValueError, match="does not match release tag"):
        identity.dynamo_v2_label(release_repo, "0.6.0")
    path.write_text("pub fn parser() {}\n")
    # The actual working bytes own identity, not a stale index.
    assert identity.dynamo_v2_label(release_repo) == "0.6.0"
    ignored = release_repo / "parsers/v2/src/ignored.rs"
    (release_repo / ".git/info/exclude").write_text("ignored.rs\n")
    ignored.write_text("ignored source\n")
    with pytest.raises(ValueError, match="does not match release tag"):
        identity.dynamo_v2_label(release_repo, "0.6.0")
    ignored.unlink()
    path.unlink()
    with pytest.raises(ValueError, match="does not match release tag"):
        identity.dynamo_v2_label(release_repo, "0.6.0")


def test_generated_capture_does_not_change_identity(release_repo):
    label = identity.dynamo_v2_label(release_repo, "current")
    output = release_repo / "conformance/unified/results.yaml"
    output.parent.mkdir(parents=True)
    output.write_text("generated: true\n")
    assert identity.dynamo_v2_label(release_repo, "current") == label
    assert identity.dynamo_v2_label(release_repo) == "0.6.0"


def test_candidate_tree_and_working_source_have_identical_fingerprints(release_repo):
    path = release_repo / "parsers/v2/src/lib.rs"
    path.write_text("pub fn candidate() {}\n")
    git(release_repo, "add", str(path))
    candidate_tree = git(release_repo, "write-tree")
    assert identity.source_fingerprint(release_repo, candidate_tree) == identity.source_fingerprint(release_repo)
    assert identity.source_fingerprint(release_repo, "HEAD^{tree}") != identity.source_fingerprint(release_repo)
    with pytest.raises(ValueError, match="does not match release tag"):
        identity.dynamo_v2_label(release_repo, "0.6.0")


@pytest.mark.parametrize("label", ["", " ", "0.5.0", "0.6.0+pr232", "0.6.0+source." + "0" * 64])
def test_wrong_identity_rejected(release_repo, label):
    with pytest.raises(ValueError):
        identity.dynamo_v2_label(release_repo, label)


def test_source_qualified_label_expires_after_edit(release_repo):
    label = identity.dynamo_v2_label(release_repo, "current")
    (release_repo / "parsers/v2/src/lib.rs").write_text("pub fn changed() {}\n")
    with pytest.raises(ValueError, match="does not identify this source"):
        identity.dynamo_v2_label(release_repo, label)


def test_missing_tag_is_not_a_release(release_repo):
    git(release_repo, "tag", "-d", "dynamo-parsers-v2-v0.6.0")
    with pytest.raises(ValueError, match="does not match release tag"):
        identity.dynamo_v2_label(release_repo, "0.6.0")
    assert identity.dynamo_v2_provenance(release_repo, "current")["kind"] == "unpublished"


def test_tagless_shallow_clone_selects_only_source_verified_release(release_repo, tmp_path_factory, monkeypatch):
    recorded = identity.dynamo_v2_provenance(release_repo)
    clone = tmp_path_factory.mktemp("tagless") / "repo"
    git(release_repo, "clone", "--depth=1", "--no-tags", release_repo.as_uri(), str(clone))
    assert git(clone, "rev-parse", "--is-shallow-repository") == "true"
    assert git(clone, "tag", "--list") == ""
    qualified = identity.dynamo_v2_label(clone)
    assert qualified == "0.6.0+source." + recorded["source_sha256"]
    assert identity.select_capture_label(clone, {}) == qualified
    assert identity.select_capture_label(clone, {"0.6.0": [None]}) == qualified
    assert identity.select_capture_label(clone, {"0.6.0": [recorded]}) == "0.6.0"
    result = subprocess.run(
        [sys.executable, identity.__file__, "--repo-root", str(clone), "--format", "label", "--select-capture"],
        input=json.dumps({"0.6.0": [recorded]}), text=True, check=True, capture_output=True,
        env=identity.git_subprocess_env(),
    )
    assert result.stdout.strip() == "0.6.0"
    current = identity.dynamo_v2_provenance(clone)
    assert identity.select_capture_label(clone, {"0.6.0": [recorded], qualified: [current]}) == qualified
    with pytest.raises(ValueError, match="source identity"):
        identity.select_capture_label(clone, {"0.6.0": [recorded], qualified: [recorded]})
    for field, wrong in [("source_sha256", "0" * 64), ("source_id", "sha256:wrong"),
                         ("source_paths", []), ("release_commit", "0" * 40),
                         ("release_tag", "other"), ("kind", "unpublished")]:
        assert identity.select_capture_label(clone, {"0.6.0": [recorded, {**recorded, field: wrong}]}) == qualified
    monkeypatch.setenv(identity.ENV_OVERRIDE, "current")
    assert identity.select_capture_label(clone, {"0.6.0": [recorded]}) == qualified
    monkeypatch.delenv(identity.ENV_OVERRIDE)
    with pytest.raises(ValueError, match="does not match release tag"):
        identity.dynamo_v2_label(clone, "0.6.0")
    (clone / "parsers/v2/src/lib.rs").write_text("pub fn changed() {}\n")
    assert identity.select_capture_label(clone, {"0.6.0": [recorded]}) == identity.dynamo_v2_label(clone)


@pytest.mark.parametrize("job", ["rust", "conformance-table"])
def test_ci_checkout_retains_release_source_identity(release_repo, tmp_path_factory, job):
    recorded = identity.dynamo_v2_provenance(release_repo)
    (release_repo / "capture.json").write_text(json.dumps(recorded))
    git(release_repo, "add", ".")
    commit = git(release_repo, "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                 "commit-tree", git(release_repo, "write-tree"), "-p", "HEAD", input="capture\n")
    git(release_repo, "update-ref", "HEAD", commit)
    workflow = yaml.safe_load((Path(__file__).resolve().parents[3] / ".github/workflows/ci.yml").read_text())
    checkout = next(step for step in workflow["jobs"][job]["steps"]
                    if step.get("uses", "").startswith("actions/checkout@"))
    depth = checkout["with"].get("fetch-depth", 1)
    clone = tmp_path_factory.mktemp("ci-capture") / "repo"
    git(release_repo, "clone", *([f"--depth={depth}"] if depth else []),
        "--no-tags", release_repo.as_uri(), str(clone))
    assert identity.source_fingerprint(clone) == recorded["source_sha256"]
    assert identity.select_capture_label(clone, {recorded["label"]: [recorded]}) == recorded["label"]


@pytest.mark.parametrize("distribution", ["squash", "shallow"])
@pytest.mark.parametrize("kind", ["unpublished", "release", "tagless_release"])
def test_consumer_identity_survives_unavailable_pr_origin(release_repo, tmp_path_factory, distribution, kind):
    base = git(release_repo, "rev-parse", "HEAD")
    git(release_repo, "checkout", "-b", "capture-work")
    if kind == "unpublished":
        (release_repo / "parsers/v2/src/lib.rs").write_text("pub fn changed_parser() {}\n")
    (release_repo / "request.json").write_text("{}\n")

    def commit(parent, message):
        git(release_repo, "add", ".")
        return git(release_repo, "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                   "commit-tree", git(release_repo, "write-tree"), "-p", parent, input=message + "\n")

    anchor = commit(base, "pre-capture PR commit")
    git(release_repo, "update-ref", "HEAD", anchor)
    recorded = identity.dynamo_v2_provenance(release_repo)
    assert identity.validate_capture_provenance(release_repo, recorded) == recorded
    (release_repo / "capture.json").write_text(json.dumps(recorded))
    published = commit(base if distribution == "squash" else anchor, "published capture")
    git(release_repo, "checkout", "-b", "merged", published)
    git(release_repo, "update-ref", "-d", "refs/heads/capture-work")
    clone = tmp_path_factory.mktemp("distributed-capture") / "repo"
    git(release_repo, "clone", *(["--depth=1"] if distribution == "shallow" else []),
        "--no-tags", "--branch", "merged", release_repo.as_uri(), str(clone))
    assert git(clone, "rev-parse", "--is-shallow-repository") == str(distribution == "shallow").lower()
    if kind != "unpublished":
        git(clone, "fetch", "origin", "tag", recorded["release_tag"])
    missing = subprocess.run(
        ["git", "-C", str(clone), "cat-file", "-e", anchor],
        capture_output=True,
        env=identity.git_subprocess_env(),
    )
    assert missing.returncode != 0
    assert identity.source_fingerprint(clone) == recorded["source_sha256"]
    with pytest.raises(ValueError, match="Git anchor .* is unavailable"):
        identity.validate_capture_provenance(clone, recorded)
    if kind == "tagless_release":
        git(clone, "tag", "-d", recorded["release_tag"])
    captures = {recorded["label"]: [recorded]}
    assert identity.select_capture_label(clone, captures) == recorded["label"]
    result = subprocess.run(
        [sys.executable, identity.__file__, "--repo-root", str(clone), "--format", "label", "--select-capture"],
        input=json.dumps(captures), text=True, check=True, capture_output=True,
        env=identity.git_subprocess_env(),
    )
    assert result.stdout.strip() == recorded["label"]
    for field, wrong in [("git_commit", "malformed"), ("git_head_tree", None),
                         ("source_sha256", "0" * 64), ("source_id", "sha256:wrong"),
                         ("source_paths", []), ("label", "other"), ("release_tag", "other")]:
        corrupted = {recorded["label"]: [recorded, {**recorded, field: wrong}]}
        if kind == "tagless_release":
            assert identity.select_capture_label(clone, corrupted) == identity.dynamo_v2_label(clone)
        else:
            with pytest.raises(ValueError, match="source identity"):
                identity.select_capture_label(clone, corrupted)


def test_missing_capture_anchor_has_descriptive_chained_error(release_repo):
    recorded = identity.dynamo_v2_provenance(release_repo, "current")
    recorded["git_commit"] = "0" * 40
    with pytest.raises(ValueError, match="Git anchor .* is unavailable") as error:
        identity.validate_capture_provenance(release_repo, recorded)
    assert isinstance(error.value.__cause__, subprocess.CalledProcessError)


def test_origin_tree_is_producer_proof_not_consumer_source_identity(release_repo):
    recorded = identity.dynamo_v2_provenance(release_repo)
    recorded["git_head_tree"] = "0" * 40
    with pytest.raises(ValueError, match="Git commit/tree mismatch"):
        identity.validate_capture_provenance(release_repo, recorded)
    assert identity.select_capture_label(release_repo, {recorded["label"]: [recorded]}) == recorded["label"]


@pytest.mark.parametrize("override", [None, "current"])
@pytest.mark.parametrize("field", [None, "label", "kind", "source_sha256", "source_id",
                                  "source_paths", "release_tag", "release_commit",
                                  "git_commit", "git_head_tree"])
def test_selected_current_rejects_wrong_provenance(release_repo, monkeypatch, override, field):
    if override:
        monkeypatch.setenv(identity.ENV_OVERRIDE, override)
    recorded = identity.dynamo_v2_provenance(release_repo)
    wrong = None if field is None else {**recorded, field: "wrong"}
    with pytest.raises(ValueError, match="capture.*source identity"):
        identity.select_capture_label(release_repo, {recorded["label"]: [recorded, wrong]})


def test_selected_current_deduplicates_provenance_checks(release_repo, monkeypatch):
    recorded = identity.dynamo_v2_provenance(release_repo)
    fingerprint = identity.source_fingerprint
    calls = []

    def measured(repo, revision=None):
        calls.append(revision)
        return fingerprint(repo, revision)

    monkeypatch.setattr(identity, "source_fingerprint", measured)
    assert identity.select_capture_label(release_repo, {"0.6.0": [recorded] * 603}) == "0.6.0"
    assert calls == [None, recorded["release_commit"]]


@pytest.mark.parametrize("override", [None, "current"])
def test_renderer_rejects_wrong_selected_source_after_folding(release_repo, monkeypatch, override):
    if override:
        monkeypatch.setenv(identity.ENV_OVERRIDE, override)
    provenance = identity.dynamo_v2_provenance(release_repo)
    label = provenance["label"]
    base = release_repo / "fixtures"
    key = "UNIFIED.1-1"
    stimulus = {"scenario": "text_only", "input": "hello", "tools": [],
                "chunks": [{"delta_text": "hello"}]}
    captured = {"capture_input": capture_input(stimulus),
                "assembled": [{"kind": "text", "text": "hello"}]}

    def write(directory, record, recorded=None):
        path = base / directory / "gemma4" / (key + ".yaml")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump({"family": "gemma4", "cases": {key: record},
                                        "capture_provenance": recorded}))

    monkeypatch.setattr(table, "_unified_dynamo_label",
                        lambda captures: identity.select_capture_label(release_repo, captures))
    write("inputs", stimulus)
    write("golden", {"assembled": captured["assembled"]})
    directory = "dynamo_v2-" + label
    write(directory, captured, {**provenance, "source_id": "wrong"})
    with pytest.raises(ValueError, match="source identity"):
        table._load_unified_fixtures(base)
    duplicate = base / directory / "gemma4" / "duplicate.yaml"
    duplicate.write_text(yaml.safe_dump({"family": "gemma4", "cases": {key: captured},
                                         "capture_provenance": provenance}))
    with pytest.raises(ValueError, match="conflicting capture provenance"):
        table._load_unified_fixtures(base)
    duplicate.unlink()
    patch = directory + ".patch1"
    write(patch, captured, provenance)
    if override:
        (base / patch / "capture-snapshot.json").write_text(json.dumps({
            "schema_version": 1, "records": [f"gemma4/{key}.yaml"],
        }))
    cases, _, versions = table._load_unified_fixtures(base)
    assert versions["dynamo_v2"] == label
    assert cases[0]["dynamo"] == captured["assembled"]
    write(patch, captured, None)
    with pytest.raises(ValueError, match="source identity"):
        table._load_unified_fixtures(base)


def test_raw_shards_and_effective_case_inventory_select_identically(release_repo):
    recorded = identity.dynamo_v2_provenance(release_repo)
    git(release_repo, "tag", "-d", recorded["release_tag"])
    qualified = identity.dynamo_v2_label(release_repo)
    for patch, expected in [(recorded, "0.6.0"), (None, qualified),
                            ({**recorded, "source_id": "wrong"}, qualified)]:
        rust = {
            "0.6.0": {"complete_snapshot": False, "records": {"gemma4/UNIFIED.31-29": None}},
            "0.6.0.patch1": {"complete_snapshot": False, "records": {"gemma4/UNIFIED.g4-1": patch}},
        }
        renderer = {"0.6.0": [patch]}
        assert identity.select_capture_label(release_repo, rust) == identity.select_capture_label(release_repo, renderer) == expected
        result = subprocess.run(
            [sys.executable, identity.__file__, "--repo-root", str(release_repo), "--format", "label", "--select-capture"],
            input=json.dumps(rust), text=True, check=True, capture_output=True,
            env=identity.git_subprocess_env(),
        )
        assert result.stdout.strip() == expected


@pytest.mark.parametrize("complete", [False, True])
@pytest.mark.parametrize("surviving_invalid", [False, True])
def test_selected_identity_checks_only_effective_records(release_repo, monkeypatch, complete, surviving_invalid):
    if complete:
        monkeypatch.setenv(identity.ENV_OVERRIDE, "current")
    recorded = identity.dynamo_v2_provenance(release_repo)
    label = recorded["label"]
    captures = {
        label: {"complete_snapshot": False, "records": {
            "gemma4/UNIFIED.31-29": None, "gemma4/obsolete": None,
        }},
        label + ".patch10": {"complete_snapshot": complete, "records": {
            "gemma4/UNIFIED.g4-1": recorded,
        }},
    }
    if not surviving_invalid:
        captures[label]["records"].pop("gemma4/obsolete")
    if surviving_invalid and not complete:
        with pytest.raises(ValueError, match="source identity"):
            identity.select_capture_label(release_repo, captures)
    else:
        assert identity.select_capture_label(release_repo, captures) == label


def test_tagless_release_selection_is_source_identity_not_branch_ancestry(release_repo):
    recorded = identity.dynamo_v2_provenance(release_repo)
    tree = git(release_repo, "rev-parse", "HEAD^{tree}")
    unrelated = git(release_repo, "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "commit-tree", tree, input="same parser source, independent history\n")
    git(release_repo, "update-ref", "HEAD", unrelated)
    git(release_repo, "tag", "-d", recorded["release_tag"])
    ancestry = subprocess.run(
        ["git", "-C", str(release_repo), "merge-base", "--is-ancestor", recorded["release_commit"], "HEAD"],
        capture_output=True,
        env=identity.git_subprocess_env(),
    )
    assert ancestry.returncode == 1
    assert identity.source_fingerprint(release_repo, unrelated) == recorded["source_sha256"]
    assert identity.select_capture_label(release_repo, {"0.6.0": [recorded]}) == "0.6.0"
    # Consumer reuse does not authorize publishing this tagless checkout as a release.
    with pytest.raises(ValueError, match="does not match release tag"):
        identity.dynamo_v2_label(release_repo, "0.6.0")


def test_environment_override_and_explicit_precedence(release_repo, monkeypatch):
    monkeypatch.setenv(identity.ENV_OVERRIDE, "current")
    assert "+source." in identity.dynamo_v2_label(release_repo)
    assert identity.dynamo_v2_label(release_repo, "0.6.0") == "0.6.0"
    monkeypatch.setenv(identity.ENV_OVERRIDE, "")
    with pytest.raises(ValueError, match="empty"):
        identity.dynamo_v2_label(release_repo)


def test_cli_and_python_consumers_share_identity(release_repo):
    command = [sys.executable, identity.__file__, "--repo-root", str(release_repo), "--label", "current"]
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        env=identity.git_subprocess_env(),
        text=True,
    )
    assert json.loads(result.stdout) == identity.dynamo_v2_provenance(release_repo, "current")


def test_explicit_repo_ignores_caller_git_index(release_repo, tmp_path, monkeypatch):
    external_index = tmp_path / "external-index"
    monkeypatch.setenv("GIT_INDEX_FILE", str(external_index))

    assert identity.dynamo_v2_label(release_repo) == "0.6.0"
    assert not external_index.exists()


def test_manifest_version_is_package_scoped(release_repo):
    manifest = release_repo / "parsers/v2/Cargo.toml"
    manifest.write_text('[dependencies.other]\nversion="9.9.9"\n[package]\nversion="0.6.0"\n')
    assert identity.crate_version(manifest) == "0.6.0"


def test_source_symlinks_hash_link_text_without_reading_target(release_repo):
    link = release_repo / "parsers/v2/src/linked.rs"
    link.symlink_to("/external/nonexistent/target")
    before = identity.source_fingerprint(release_repo)
    git(release_repo, "add", str(link))
    tree = git(release_repo, "write-tree")
    assert identity.source_fingerprint(release_repo, tree) == before
    link.unlink()
    link.symlink_to("/external/other/target")
    assert identity.source_fingerprint(release_repo) != before


def test_unrelated_commit_preserves_feed_source_identity(release_repo):
    recorded = identity.dynamo_v2_provenance(release_repo, "current")
    (release_repo / "unrelated.yaml").write_text("unrelated: true\n")
    git(release_repo, "add", "unrelated.yaml")
    tree = git(release_repo, "write-tree")
    commit = git(release_repo, "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                 "commit-tree", tree, "-p", recorded["git_commit"], input="unrelated\n")
    git(release_repo, "update-ref", "HEAD", commit)
    assert identity.validate_capture_provenance(release_repo, recorded) == recorded


@pytest.mark.parametrize("field", ["source_id", "source_sha256", "git_commit", "git_head_tree", "kind"])
def test_feed_identity_cannot_be_restamped(release_repo, field):
    recorded = identity.dynamo_v2_provenance(release_repo, "current")
    recorded[field] = "wrong identity"
    with pytest.raises(ValueError, match="source identity differs"):
        identity.validate_capture_provenance(release_repo, recorded)


def test_feed_source_change_requires_recapture(release_repo):
    recorded = identity.dynamo_v2_provenance(release_repo, "current")
    (release_repo / "parsers/v2/src/lib.rs").write_text("new parser\n")
    with pytest.raises(ValueError, match="does not identify this source"):
        identity.validate_capture_provenance(release_repo, recorded)


@pytest.mark.parametrize("stale", [False, True])
def test_explode_uses_producer_provenance_before_writes(release_repo, monkeypatch, stale):
    build = release_repo / "conformance/unified"
    build.mkdir(parents=True)
    provenance = identity.dynamo_v2_provenance(release_repo, "current")
    feed = {"capture_provenance": provenance, "cases": [{
        "id": "UNIFIED.probe.gemma4", "input": "hello", "dynamo": [{"kind": "text", "text": "hello"}]
    }]}
    (build / "unified_results.yaml").write_text(yaml.safe_dump(feed))
    monkeypatch.setattr(explode, "REPO", release_repo)
    monkeypatch.setattr(explode, "BUILD", build)
    monkeypatch.setattr(explode, "_case_key", lambda _cid: ("UNIFIED.probe", "gemma4", "probe"))
    if stale:
        (release_repo / "parsers/v2/src/lib.rs").write_text("changed parser\n")
        with pytest.raises(ValueError, match="does not identify this source"):
            explode.main()
        assert sorted(p.name for p in build.iterdir()) == ["unified_results.yaml"]
    else:
        explode.main()
        doc = yaml.safe_load((build / f"dynamo_v2-{provenance['label']}/gemma4/UNIFIED.probe.yaml").read_text())
        assert doc["capture_provenance"] == provenance
        assert doc["captured_with"] == {"dynamo_v2": provenance["label"]}


def test_missing_feed_provenance_is_rejected(release_repo):
    with pytest.raises(ValueError, match="no producer source identity"):
        identity.validate_capture_provenance(release_repo, None)
