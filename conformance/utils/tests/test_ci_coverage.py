# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import ci_coverage
import ci_publication_cache
import unified_history
import verify_ci_selection


@pytest.mark.parametrize("path", [
    "conformance/fixtures-unified-v2/versions/dynamo-0.7.18.tar.gz",
    "conformance/viewer/model.js", "conformance/viewer/values.js", "conformance/viewer/app.js",
    "conformance/viewer/data.js", "conformance/viewer/yaml_data.js",
    "conformance/publisher/src/yaml.rs", "conformance/utils/src/viewer_catalog.py",
    "conformance/utils/src/ci_coverage.py", "conformance/parser_families.yaml",
    "conformance/fixtures-manifest.json", ".gitattributes", "Cargo.lock", "unknown-file",
])
def test_evidence_readers_and_unclassified_changes_require_exhaustive(path):
    assert ci_coverage.classify(["index.html", path]) == "exhaustive"
    assert ci_coverage.classify([]) == "exhaustive"
    assert ci_coverage.classify(["index.html"]) == "latest-two"


@pytest.mark.parametrize("status", ["A", "D", "T", "U", "R100", "C100"])
def test_structural_changes_cannot_select_the_smaller_profile(status):
    assert ci_coverage.classify_changes(f"{status}\0index.html\0") == "exhaustive"
    assert ci_coverage.classify_changes("R100\0conformance/viewer/model.js\0index.html\0") == "exhaustive"
    assert ci_coverage.classify_changes("M\0index.html\0") == "latest-two"


@pytest.mark.parametrize("profile", ["exhaustive", "latest-two"])
@pytest.mark.parametrize("shard_count", [4, verify_ci_selection.QUALIFICATION_SHARDS])
def test_capture_shards_have_exact_disjoint_union_and_newest_versions(profile, shard_count):
    rows = [{"path": f"{implementation}/{mode}/{family}/{version}", "implementation": implementation,
             "mode": mode, "family": family, "version": version}
            for implementation in ("dynamo_v2", "sglang") for mode in ("unified", "batch")
            for family in ("a", "b") for version in ("0.7.9", "0.7.10", "0.7.18")]
    whole = ci_coverage.select_resources(rows, profile=profile, shard=0, shards=1)
    shards = [ci_coverage.select_resources(rows, profile=profile, shard=index, shards=shard_count) for index in range(shard_count)]
    paths = [row["path"] for shard in shards for row in shard]
    assert len(paths) == len(set(paths))
    assert sorted(paths) == sorted(row["path"] for row in whole)
    assert len(whole) == (24 if profile == "exhaustive" else 16)
    if profile == "latest-two":
        assert {row["version"] for row in whole} == {"0.7.10", "0.7.18"}


def test_authored_reader_does_not_open_historical_archives(tmp_path, monkeypatch):
    family_path = tmp_path / "families/sample/inputs_and_golden.yaml"
    family_path.parent.mkdir(parents=True)
    family_path.write_text(yaml.safe_dump({"input_document": {"family": "sample"},
                                         "golden_document": {"family": "sample"}, "cases": {}}))
    monkeypatch.setattr(unified_history, "_load_store_unlocked", lambda *_: pytest.fail("full store loaded"))
    versions = tmp_path / "versions"
    versions.mkdir()
    (versions / "dynamo-0.7.18.tar.gz").write_bytes(b"must never be opened")
    store = unified_history.load_authored_store(tmp_path)
    assert set(store.families) == {"sample"}
    assert not store.histories


@pytest.mark.parametrize("linked", ["families", "family", "document"])
def test_authored_reader_rejects_links_in_selected_inputs(tmp_path, linked):
    root = tmp_path / "store"
    family = root / "families/sample"
    family.mkdir(parents=True)
    source = family / "inputs_and_golden.yaml"
    source.write_text("input_document: {family: sample}\ngolden_document: {family: sample}\ncases: {}\n")
    target = {"families": root / "families", "family": family, "document": source}[linked]
    outside = tmp_path / "outside"
    target.rename(outside)
    target.symlink_to(outside, target_is_directory=linked != "document")
    with pytest.raises(ValueError):
        unified_history.load_authored_store(root)


def test_cache_miss_hit_corruption_and_identity_invalidation(tmp_path):
    assert not ci_publication_cache.valid(tmp_path, "source-reader-policy")
    for name in ci_publication_cache.FILES:
        (tmp_path / name).write_bytes(name.encode())
    ci_publication_cache.seal(tmp_path, "source-reader-policy")
    assert ci_publication_cache.valid(tmp_path, "source-reader-policy")
    for identity in ("changed-source", "changed-reader", "changed-authored", "changed-policy"):
        assert not ci_publication_cache.valid(tmp_path, identity)
    (tmp_path / ci_publication_cache.FILES[0]).write_bytes(b"corrupt")
    assert not ci_publication_cache.valid(tmp_path, "source-reader-policy")


def test_cache_key_tracks_publication_inputs_but_reuses_unrelated_changes(tmp_path, monkeypatch):
    paths = [
        "conformance/fixtures-unified-v2/families/sample/dynamo_v2.yaml",
        "conformance/viewer/app.js",
        "conformance/utils/src/viewer_catalog.py",
        "conformance/utils/publication-requirements.in",
        ".gitattributes",
        "conformance/README.md",
        ".github/workflows/ci.yml",
        "README.md",
        "conformance/utils/tests/test_ci_coverage.py",
        "conformance/viewer-catalog.json",
        "conformance/CONFORMANCE_v2.html",
        "conformance/fixtures-unified-v2/families/sample/dynamo_v2-0.8.0.yaml",
        "conformance/fixtures/toolcalling/fixtures-stream-v1/dynamo_v2-0.7.18.tar.gz",
    ]
    untracked = ["conformance/fixtures-unified-v2/families/other/dynamo_v2.yaml"]
    for path in paths:
        source = tmp_path / path
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"original")
    for path in untracked:
        source = tmp_path / path
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"original")

    tag_refs = ["refs/tags/dynamo-parsers-v2-v0.7.18 release-tag"]

    def run(arguments, **kwargs):
        if arguments[:2] == ["git", "ls-files"]:
            selected = untracked if "--others" in arguments else paths
            output = "\0".join(selected).encode() + b"\0"
        elif arguments[:2] == ["git", "for-each-ref"]:
            pattern = arguments[-1]
            output = "\n".join(
                ref for ref in tag_refs if ci_publication_cache.fnmatch.fnmatchcase(ref.split(maxsplit=1)[0], pattern)
            ).encode()
        elif arguments == ["node", "--version"]:
            output = b"node identity"
        else:
            assert arguments == ["rustc", "--version"]
            output = b"rustc identity"
        return SimpleNamespace(stdout=output, returncode=0)

    monkeypatch.setattr(ci_publication_cache.subprocess, "run", run)
    original = ci_publication_cache.cache_key(tmp_path, "native-full-v1")
    assert ci_publication_cache.cache_key(tmp_path, "native-full-v2") != original
    tag_refs.append("refs/tags/dynamo-engine-v1 unrelated-tag")
    assert ci_publication_cache.cache_key(tmp_path, "native-full-v1") == original
    tag_refs[0] = "refs/tags/dynamo-parsers-v2-v0.7.18 changed-release-tag"
    assert ci_publication_cache.cache_key(tmp_path, "native-full-v1") != original
    tag_refs[0] = "refs/tags/dynamo-parsers-v2-v0.7.18 release-tag"
    node_version = ci_publication_cache.subprocess.run
    monkeypatch.setattr(ci_publication_cache.subprocess, "run", lambda arguments, **kwargs: (
        SimpleNamespace(stdout=b"other node identity", returncode=0)
        if arguments == ["node", "--version"] else node_version(arguments, **kwargs)
    ))
    assert ci_publication_cache.cache_key(tmp_path, "native-full-v1") != original
    monkeypatch.setattr(ci_publication_cache.subprocess, "run", node_version)
    python_version = platform.python_version
    monkeypatch.setattr(ci_publication_cache.platform, "python_version", lambda: "test-runtime")
    assert ci_publication_cache.cache_key(tmp_path, "native-full-v1") != original
    monkeypatch.setattr(ci_publication_cache.platform, "python_version", python_version)
    for path in paths[:6]:
        source = tmp_path / path
        source.write_bytes(b"changed")
        assert ci_publication_cache.cache_key(tmp_path, "native-full-v1") != original
        source.write_bytes(b"original")
        assert ci_publication_cache.cache_key(tmp_path, "native-full-v1") == original
    source = tmp_path / untracked[0]
    source.write_bytes(b"changed")
    assert ci_publication_cache.cache_key(tmp_path, "native-full-v1") != original
    source.write_bytes(b"original")
    for path in paths[6:]:
        source = tmp_path / path
        source.write_bytes(b"changed")
        assert ci_publication_cache.cache_key(tmp_path, "native-full-v1") == original


@pytest.fixture
def publication_root(tmp_path):
    (tmp_path / "conformance").mkdir()
    return tmp_path


def test_cache_stamp_updates_source_revision_without_rewriting_generation_provenance(publication_root):
    old_revision = "a" * 40
    revision = "b" * 40
    publication = {
        "schema": 1,
        "revision": old_revision,
        "generated_revision": old_revision,
        "catalog_revision": "catalog-revision",
        "generated_on": "old time",
        "command": f"conformance-publish --root /old/workspace --output /old/pages --revision {old_revision}",
        "output": "/old/pages",
    }
    encoded = json.dumps(publication, separators=(",", ":"))
    (publication_root / "conformance/viewer-publication.json").write_text(encoded)
    (publication_root / "index.html").write_text(
        f'<head><script type="application/json" id="conformance-publication">{encoded}</script></head>'
    )
    report_path = publication_root / "conformance/CONFORMANCE_v2.html"
    report_path.write_bytes((publication_root / "index.html").read_bytes())
    payload_path = publication_root / "payload.bin"
    payload_path.write_bytes(b"\x00untouched published bytes\xff")

    ci_publication_cache.stamp(publication_root, revision)
    first_stamp = (publication_root / "conformance/viewer-publication.json").read_bytes()
    first_index = (publication_root / "index.html").read_bytes()
    ci_publication_cache.stamp(publication_root, revision)
    assert (publication_root / "conformance/viewer-publication.json").read_bytes() == first_stamp
    assert (publication_root / "index.html").read_bytes() == first_index
    ci_publication_cache.stamp(publication_root, "c" * 40)
    revision = "c" * 40

    expected = json.loads((publication_root / "conformance/viewer-publication.json").read_bytes())
    assert expected == {**publication, "revision": revision}
    assert report_path.read_bytes() == (publication_root / "index.html").read_bytes()
    assert payload_path.read_bytes() == b"\x00untouched published bytes\xff"
    assert expected["revision"] == revision
    assert expected["generated_revision"] == old_revision
    assert expected["generated_on"] == "old time"
    assert expected["output"] == "/old/pages"
    assert expected["command"] == publication["command"]
    index = (publication_root / "index.html").read_text()
    embedded = index.split('id="conformance-publication">', 1)[1].split("</script>", 1)[0]
    assert json.loads(embedded) == expected


@pytest.mark.parametrize("history", [False, True])
@pytest.mark.parametrize("origin", ["a", "e"])
@pytest.mark.parametrize("standalone", [False, True])
def test_cache_stamp_keeps_bootstrap_independent_of_sidecar_policy(publication_root, history, origin, standalone):
    identity = {"schema": 1, "revision": "a" * 40,
                "catalog_revision": "b" * 64, "resolved_catalog_revision": "b" * 64}
    identity["revision"] = origin * 40
    publication = {**identity, "generated_revision": origin * 40, "command": f"conformance-publish --revision {origin * 40}",
                   "generation_mode": "history" if history else "latest", "output": "site"}
    if history:
        publication["resolved_catalog_revision"] = "c" * 64
    metadata = publication_root / "conformance/viewer-publication.json"
    if not standalone:
        metadata.write_text(json.dumps(publication))
    page = publication_root / "index.html"
    page.write_text('<script type="application/json" id="conformance-publication">' + json.dumps(identity) + '</script>')
    ci_publication_cache.stamp(publication_root, "d" * 40)
    expected = {**identity, "revision": "d" * 40}
    actual = page.read_text().split('id="conformance-publication">', 1)[1].split('</script>', 1)[0]
    assert json.loads(actual) == expected
    if standalone:
        assert [path.relative_to(publication_root).as_posix() for path in publication_root.rglob("*") if path.is_file()] == ["index.html"]
    else:
        assert json.loads(metadata.read_bytes())["resolved_catalog_revision"] == publication["resolved_catalog_revision"]
    ci_publication_cache.stamp(publication_root, "d" * 40)
    assert json.loads(page.read_text().split('id="conformance-publication">', 1)[1].split('</script>', 1)[0]) == expected


def test_cache_stamp_preserves_legacy_metadata_without_generated_revision(publication_root):
    revision = "a" * 40
    publication = {
        "schema": 1,
        "revision": revision,
        "command": f"conformance-publish --revision {revision}",
    }
    encoded = json.dumps(publication)
    publication_path = publication_root / "conformance/viewer-publication.json"
    index_path = publication_root / "index.html"
    publication_path.write_text(encoded)
    index_path.write_text(
        f'<head><script type="application/json" id="conformance-publication">{encoded}</script></head>'
    )

    ci_publication_cache.stamp(publication_root, "b" * 40)

    updated = json.loads(publication_path.read_bytes())
    assert updated["revision"] == "b" * 40
    assert updated["generated_revision"] == revision
    embedded = index_path.read_text().split('id="conformance-publication">', 1)[1].split("</script>", 1)[0]
    assert json.loads(embedded) == updated
    ci_publication_cache.stamp(publication_root, "b" * 40)
    assert json.loads(publication_path.read_bytes()) == updated


@pytest.mark.parametrize("has_generated_revision", [False, True])
def test_cache_stamp_rejects_command_revision_mismatch(publication_root, has_generated_revision):
    revision = "a" * 40
    publication = {
        "schema": 1,
        "revision": revision,
        "generated_on": "old time",
        "command": f"conformance-publish --revision {'b' * 40}",
    }
    if has_generated_revision:
        publication["revision"] = "b" * 40
        publication["generated_revision"] = revision
    encoded = json.dumps(publication)
    (publication_root / "conformance/viewer-publication.json").write_text(encoded)
    (publication_root / "index.html").write_text(
        f'<head><script type="application/json" id="conformance-publication">{encoded}</script></head>'
    )

    with pytest.raises(ValueError, match="does not match its source revision"):
        ci_publication_cache.stamp(publication_root, "c" * 40)


@pytest.mark.parametrize("generated_revision", [None, 42, "not-a-commit", "A" * 40, "a" * 39])
def test_cache_stamp_rejects_malformed_generated_revision_before_writing(publication_root, generated_revision):
    revision = "a" * 40
    publication = {
        "schema": 1,
        "revision": revision,
        "generated_revision": generated_revision,
        "generated_on": "old time",
        "command": f"conformance-publish --revision {revision}",
    }
    encoded = json.dumps(publication)
    publication_path = publication_root / "conformance/viewer-publication.json"
    index_path = publication_root / "index.html"
    publication_path.write_text(encoded)
    index_path.write_text(
        f'<head><script type="application/json" id="conformance-publication">{encoded}</script></head>'
    )

    with pytest.raises(ValueError, match="generated revision"):
        ci_publication_cache.stamp(publication_root, "b" * 40)

    assert publication_path.read_text() == encoded
    assert f'>{encoded}</script>' in index_path.read_text()


@pytest.mark.parametrize("schema", [True, 1.0])
def test_cache_stamp_rejects_malformed_schema_before_writing(publication_root, schema):
    revision = "a" * 40
    publication = {"schema": schema, "revision": revision,
                   "command": f"conformance-publish --revision {revision}"}
    encoded = json.dumps(publication)
    publication_path = publication_root / "conformance/viewer-publication.json"
    index_path = publication_root / "index.html"
    publication_path.write_text(encoded)
    index = f'<script type="application/json" id="conformance-publication">{encoded}</script>'
    index_path.write_text(index)

    with pytest.raises(ValueError, match="unsupported publication metadata"):
        ci_publication_cache.stamp(publication_root, "b" * 40)

    assert publication_path.read_text() == encoded
    assert index_path.read_text() == index


@pytest.mark.parametrize("attached_revision", ["a" * 40, "b" * 40])
def test_cache_stamp_rejects_duplicate_attached_revision_before_writing(publication_root, attached_revision):
    revision = "a" * 40
    publication = {"schema": 1, "revision": revision, "generated_revision": revision,
                   "command": f"conformance-publish --revision {revision} --revision={attached_revision}"}
    encoded = json.dumps(publication)
    publication_path = publication_root / "conformance/viewer-publication.json"
    index_path = publication_root / "index.html"
    publication_path.write_text(encoded)
    index = f'<script type="application/json" id="conformance-publication">{encoded}</script>'
    index_path.write_text(index)

    with pytest.raises(ValueError, match="does not match its source revision"):
        ci_publication_cache.stamp(publication_root, "c" * 40)

    assert publication_path.read_text() == encoded
    assert index_path.read_text() == index


def test_cache_stamp_cli_uses_native_publication_layout(publication_root):
    revision = "a" * 40
    metadata = {"schema": 1, "revision": revision, "generated_revision": revision,
                "command": f"conformance-publish --root /source --output /pages --revision {revision}"}
    publication_path = publication_root / "conformance/viewer-publication.json"
    publication_path.write_text(json.dumps(metadata))
    index_path = publication_root / "index.html"
    index_path.write_text(f'<head><script type="application/json" id="conformance-publication">{json.dumps(metadata)}</script></head>')
    payload_path = publication_root / "conformance/payload.bin"
    payload_path.write_bytes(b"unchanged generated payload")
    script = Path(ci_publication_cache.__file__)

    for target in ["b" * 40, "b" * 40, "c" * 40]:
        subprocess.run([sys.executable, str(script), "stamp", "--directory", str(publication_root),
                        "--revision", target], check=True, capture_output=True, text=True)
        expected = {**metadata, "revision": target}
        assert json.loads(publication_path.read_bytes()) == expected
        embedded = index_path.read_text().split('id="conformance-publication">', 1)[1].split("</script>", 1)[0]
        assert json.loads(embedded) == expected
        assert payload_path.read_bytes() == b"unchanged generated payload"
        assert not (publication_root / "viewer-publication.json").exists()


def test_cache_identity_does_not_follow_tracked_directory_links(tmp_path):
    outside = tmp_path / "external"
    outside.mkdir()
    link = tmp_path / "skills"
    link.symlink_to(outside, target_is_directory=True)
    before = ci_publication_cache.input_identity(link)
    (outside / "unrelated.yaml").write_bytes(b"not a publication input")
    assert ci_publication_cache.input_identity(link) == before
    link.unlink()
    link.symlink_to(tmp_path / "different", target_is_directory=True)
    assert ci_publication_cache.input_identity(link) != before


@pytest.mark.parametrize("job_name", ["conformance-native-browser", "conformance-viewer"])
def test_browser_jobs_that_publish_install_the_locked_bundler(job_name):
    root = Path(__file__).resolve().parents[3]
    workflow = yaml.safe_load((root / ".github/workflows/ci.yml").read_text())
    steps = workflow["jobs"][job_name]["steps"]
    installation = next(index for index, step in enumerate(steps)
                        if step.get("uses") == "./.github/actions/conformance-viewer-build")
    qualification = next(index for index, step in enumerate(steps)
                         if "conformance/utils/check.sh ci" in step.get("run", ""))
    assert installation < qualification


@pytest.mark.parametrize("include_history", [False, True])
@pytest.mark.parametrize("scope", ["fixtures", "publication"])
def test_browser_collection_declares_latest_and_history_dependencies(tmp_path, include_history, scope):
    root = Path(__file__).resolve().parents[3]
    command = [sys.executable, "-m", "pytest", "conformance/utils/tests/test_browser_viewer.py",
               "--collect-only", "-q", "--conformance-scope", scope]
    if include_history:
        command.append("--include-history")
    result = subprocess.run(command, cwd=root, capture_output=True, text=True,
                            env={"PATH": os.environ["PATH"], "CONFORMANCE_INCLUDE_HISTORY": "false",
                                 "CONFORMANCE_SELECTION_DIR": str(tmp_path)})
    assert result.returncode in (0, 5), result.stdout + result.stderr
    receipts = list(tmp_path.glob("*.json"))
    assert len(receipts) == 1
    selected = {node.split("::")[-1] for node in json.loads(receipts[0].read_bytes())["selected"]}
    latest = {"test_original_yaml_view_and_blob_download_preserve_source_bytes",
              "test_muse_spark_popup_preserves_message_phases"}
    history = {"test_original_archive_view_and_blob_download_preserve_source_bytes",
               "test_historical_muse_spark_popup_preserves_message_phases"}
    assert selected.intersection(latest) == (latest if scope == "publication" else set())
    assert selected.intersection(history) == (history if scope == "publication" and include_history else set())


def test_workflow_publishes_before_qualification_and_requires_every_lane():
    root = Path(__file__).resolve().parents[3]
    workflow = yaml.safe_load((root / ".github/workflows/ci.yml").read_text())
    jobs = workflow["jobs"]
    conformance_filter = next(step["with"]["filters"] for step in jobs["changes"]["steps"]
                              if step.get("id") == "filter")
    assert "- '.gitattributes'" in conformance_filter
    assert jobs["pages-build"]["env"]["CONFORMANCE_PUBLICATION_POLICY"] == "native-latest-single-html-v3"
    identity_step = next(step for step in jobs["pages-build"]["steps"] if step.get("id") == "identity")
    assert "--publication-policy \"$CONFORMANCE_PUBLICATION_POLICY\"" in identity_step["run"]
    catalog_step = next(step for step in jobs["pages-build"]["steps"]
                        if step.get("name") == "Generate source catalog inputs")
    assert "-r conformance/utils/publication-requirements.in" in catalog_step["run"]
    restore = next(step for step in jobs["pages-build"]["steps"] if step.get("name") == "Restore compatible family conversion fragments")
    assert jobs["pages-build"]["steps"].index(restore) < jobs["pages-build"]["steps"].index(catalog_step)
    assert "--include-history" in catalog_step["run"]
    assert "--include-history" in identity_step["run"]
    assert "${{ github.ref }}" in restore["with"]["key"]

    assert jobs["conformance-fixtures"]["needs"] == "changes"
    assert jobs["rust"].get("needs") is None
    required = set(jobs["conformance-table"]["needs"])
    for name, job in jobs.items():
        if name.startswith("conformance-") and name != "conformance-table":
            assert name in required
            assert job["timeout-minutes"] == 20
    publication = json.dumps(jobs["pages-build"])
    assert "snapshot_parity" not in publication
    assert "pytest" not in "\n".join(step.get("run", "") for step in jobs["pages-build"]["steps"] if "pip" not in step.get("run", ""))
    qualification = jobs["conformance-qualification"]
    assert qualification["strategy"]["matrix"]["shard"] == "${{ fromJSON(needs.changes.outputs.qualification_matrix) }}"
    assert qualification["env"]["CONFORMANCE_PARITY_SHARDS"] == "${{ strategy.job-total }}"
    assert jobs["changes"]["outputs"]["qualification_matrix"] == "${{ steps.qualification.outputs.matrix }}"
    step = next(step for step in jobs["changes"]["steps"] if step.get("id") == "qualification")
    assert "verify_ci_selection.py --shard-matrix" in step["run"]
    result = subprocess.run([sys.executable, str(root / "conformance/utils/src/verify_ci_selection.py"), "--shard-matrix"],
                            capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == list(range(verify_ci_selection.QUALIFICATION_SHARDS))
    assert verify_ci_selection.QUALIFICATION_SHARDS == 8
    check_script = (root / "conformance/utils/check.sh").read_text()
    qualification = check_script.split("    qualification)", 1)[1].split("    native-browser)", 1)[0]
    configured = set(re.findall(r"::(test_[A-Za-z0-9_]+)", qualification))
    assert configured == verify_ci_selection.QUALIFICATION


@pytest.mark.parametrize("arguments", [[], ["--baseline", "."], ["--shard-matrix", "--selections", "."]])
def test_ci_accounting_cli_rejects_incomplete_or_mixed_modes(arguments):
    result = subprocess.run([sys.executable, verify_ci_selection.__file__, *arguments],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert not result.stdout


@pytest.mark.parametrize("fault", [None, "missing", "duplicate", "missing-shard", "extra"])
def test_ci_accounting_rejects_lost_tests_and_incomplete_qualification(fault):
    qualification = sorted(verify_ci_selection.QUALIFICATION)
    baseline = [{"selected": ["fixture", "browser", *qualification]}]
    assignments = [{"selected": ["fixture"]}, {"selected": ["browser"]}]
    assignments.extend({"selected": qualification} for _ in range(verify_ci_selection.QUALIFICATION_SHARDS))
    if fault == "missing":
        assignments.pop(0)
    elif fault == "duplicate":
        assignments.append({"selected": ["browser"]})
    elif fault == "missing-shard":
        assignments.pop()
    elif fault == "extra":
        assignments.append({"selected": ["uncollected"]})
    if fault:
        with pytest.raises(ValueError, match="CI selection mismatch"):
            verify_ci_selection.verify(baseline, assignments)
    else:
        assert verify_ci_selection.verify(baseline, assignments)["unique_tests"] == 2 + len(qualification)
