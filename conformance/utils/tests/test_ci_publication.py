# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import json
from pathlib import Path
import subprocess
import sys

import pytest

import ci_publication
import ci_publication_cache


@pytest.fixture
def publication(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    temporary = tmp_path / "runtime"
    temporary.mkdir()
    for key in ("CONFORMANCE_PUBLISHED_SITE", "CONFORMANCE_PUBLISHER", "CONFORMANCE_PUBLICATION_CONTEXT", "CONFORMANCE_CONVERSION_CACHE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("GITHUB_ENV", str(tmp_path / "environment"))
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "outputs"))
    monkeypatch.setenv("GITHUB_SHA", "a" * 40)
    monkeypatch.setenv("CONFORMANCE_INCLUDE_HISTORY", "true")
    return ci_publication.Publication(root, temporary)


def test_configuration_preserves_other_step_environment(publication):
    destination = Path(os.environ["GITHUB_ENV"])
    destination.write_text("EXISTING=value\n")
    publication.configure()
    values = dict(line.split("=", 1) for line in destination.read_text().splitlines())
    assert values["EXISTING"] == "value"
    assert Path(values["CONFORMANCE_PUBLISHED_SITE"]) == publication.site
    assert Path(values["CONFORMANCE_PUBLICATION_CONTEXT"]) == publication.root / ci_publication_cache.CONTEXT


def _package(publication):
    # Independent synthetic bytes expose wrong archive membership and restoration;
    # these filenames are the existing publication/cache transport protocol.
    site = {"index.html": b"qualified HTML", "history/data.json.gz": b"qualified history"}
    catalogs = {"conformance/viewer-catalog.json": b"source catalog",
                "conformance/viewer-unified-catalog.json": b"unified catalog",
                "conformance/viewer-latest-catalog.json": b"latest catalog",
                ci_publication_cache.CONTEXT: b"sealed source context"}
    for relative, data in site.items():
        destination = publication.site / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    for relative, data in catalogs.items():
        destination = publication.root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    publication.publisher.parent.mkdir(parents=True, exist_ok=True)
    publication.publisher.write_bytes(b"qualified native publisher")
    publication.package()
    ci_publication_cache.seal(publication.payload, "qualified-identity")
    return site, catalogs


def test_package_and_admitted_cache_restore_exact_qualified_payload(publication, monkeypatch):
    site, catalogs = _package(publication)
    monkeypatch.setenv("CONFORMANCE_PUBLISHED_SITE", str(publication.temporary / "restored"))
    monkeypatch.setenv("CONFORMANCE_PUBLISHER", str(publication.temporary / "bin/publisher"))
    restored = ci_publication.Publication(publication.root, publication.temporary)
    for relative in catalogs:
        (publication.root / relative).write_bytes(b"stale checkout")
    assert restored.restore("qualified-identity")
    assert {path.relative_to(restored.site).as_posix(): path.read_bytes()
            for path in restored.site.rglob("*") if path.is_file()} == site
    assert {relative: (restored.root / relative).read_bytes() for relative in catalogs} == catalogs
    assert restored.publisher.read_bytes() == b"qualified native publisher"
    assert os.access(restored.publisher, os.X_OK)
    assert Path(os.environ["GITHUB_OUTPUT"]).read_text() == "usable=true\n"


@pytest.mark.parametrize("fault", ["absent", "wrong-identity", "corrupt"])
def test_invalid_cache_falls_back_without_touching_candidate(publication, fault):
    if fault != "absent":
        _package(publication)
        if fault == "corrupt":
            (publication.payload / ci_publication.SITE_ARCHIVE).write_bytes(b"unqualified replacement")
    publication.site.mkdir(parents=True, exist_ok=True)
    marker = publication.site / "must-survive"
    marker.write_bytes(b"current work")
    assert not publication.restore("different-identity" if fault == "wrong-identity" else "qualified-identity")
    assert marker.read_bytes() == b"current work"
    assert Path(os.environ["GITHUB_OUTPUT"]).read_text() == "usable=false\n"


def test_admitted_but_unextractable_cache_fails_instead_of_claiming_fallback(publication):
    _package(publication)
    (publication.payload / ci_publication.SITE_ARCHIVE).write_bytes(b"not a tar archive")
    ci_publication_cache.seal(publication.payload, "qualified-identity")
    with pytest.raises(subprocess.CalledProcessError):
        publication.restore("qualified-identity")
    assert not Path(os.environ["GITHUB_OUTPUT"]).exists()


def test_oversized_payload_stops_before_candidate_can_be_sealed(publication, monkeypatch):
    monkeypatch.setattr(ci_publication, "MAX_SITE_ARCHIVE_BYTES", 1)
    with pytest.raises(ValueError, match="archive size limit"):
        _package(publication)
    assert not (publication.payload / "cache-manifest.json").exists()


@pytest.mark.parametrize("phase", ["snapshot", "catalog", "publish", "verify", "identify"])
def test_generation_phases_reject_latest_mode_before_launching_tools(publication, monkeypatch, phase):
    monkeypatch.setenv("CONFORMANCE_INCLUDE_HISTORY", "false")
    calls = []
    monkeypatch.setattr(publication, "run", lambda command, **kwargs: calls.append(command))
    with pytest.raises(ValueError, match="requires history"):
        {"snapshot": publication.snapshot, "catalog": publication.catalog, "publish": publication.publish,
         "verify": publication.verify, "identify": publication.identify}[phase]()
    assert calls == []


def test_publisher_failure_preserves_real_exit_code(publication):
    publication.publisher.parent.mkdir(parents=True, exist_ok=True)
    publication.publisher.write_text(f"#!{sys.executable}\nraise SystemExit(23)\n")
    publication.publisher.chmod(0o700)
    with pytest.raises(subprocess.CalledProcessError) as failure:
        publication.publish()
    assert failure.value.returncode == 23
    assert not (publication.payload / "cache-manifest.json").exists()


@pytest.mark.parametrize("arguments", [["restore"], ["seal"], ["build", "--identity", "unexpected"],
                                        ["delivery"], ["build", "--url", "https://example.invalid/"]])
def test_phase_cli_rejects_missing_or_misplaced_context(arguments, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["ci_publication.py", *arguments])
    with pytest.raises(SystemExit) as failure:
        ci_publication.main()
    assert failure.value.code == 2


@pytest.mark.parametrize("remote,current", [("a" * 40, True), ("b" * 40, False)])
def test_only_current_main_candidate_can_deploy(publication, monkeypatch, remote, current):
    metadata = publication.site / "conformance/viewer-publication.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text('{"revision":"' + "a" * 40 + '"}')
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=f"{remote}\trefs/heads/main\n")
    monkeypatch.setattr(publication, "run", run)
    assert publication.current() is current
    assert calls == [["git", "ls-remote", "--exit-code", "origin", "refs/heads/main"]]
    assert Path(os.environ["GITHUB_OUTPUT"]).read_text() == f"current={str(current).lower()}\n"
    report = (publication.temporary / "conformance-deployment.json").read_text()
    assert ('"outcome": "current"' if current else '"outcome": "superseded"') in report


@pytest.mark.parametrize("fault", ["metadata", "remote", "network"])
def test_deployment_admission_fails_closed(publication, monkeypatch, fault):
    metadata = publication.site / "conformance/viewer-publication.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text('{"revision":"' + ("b" if fault == "metadata" else "a") * 40 + '"}')
    def run(command, **kwargs):
        if fault == "network":
            raise subprocess.CalledProcessError(128, command)
        return subprocess.CompletedProcess(command, 0, stdout="unrecognized response")
    monkeypatch.setattr(publication, "run", run)
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        publication.current()
    assert not Path(os.environ["GITHUB_OUTPUT"]).exists()


def test_partial_rerun_rebuilds_binary_and_restores_frozen_sources(publication, monkeypatch):
    site, catalogs = _package(publication)
    monkeypatch.setenv("GITHUB_SERVER_URL", "https://github.com")
    monkeypatch.setenv("GITHUB_REPOSITORY", "example/repo")
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    retained = publication.temporary / "retained-publication"
    retained.mkdir()
    (retained / "publication-status.json").write_text(json.dumps({"commit": "a" * 40,
        "run_url": "https://github.com/example/repo/actions/runs/123"}))
    metadata = retained / "conformance/viewer-publication.json"
    metadata.parent.mkdir()
    metadata.write_text(json.dumps({"revision": "a" * 40}))
    nested = metadata.parent / "nested/publication-status.json"
    nested.parent.mkdir()
    nested.write_bytes(b"reader-facing nested resource")
    nested_evidence = metadata.parent / "nested/ci-evidence/provenance.json"
    nested_evidence.parent.mkdir()
    nested_evidence.write_bytes(b"reader-facing nested provenance")
    for name, content in site.items():
        path = retained / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    for name, content in catalogs.items():
        path = retained / "ci-evidence/publication-inputs" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    (retained / "ci-evidence/publication-inputs/cache-manifest.json").write_text('{"identity":"frozen"}')
    for name in ("conformance-test-baseline", "conformance-selection-conformance-check-0"):
        path = retained / "ci-evidence" / name / (name + ".json")
        path.parent.mkdir()
        path.write_text('{"revision":"current"}')
    calls = []
    monkeypatch.setattr(publication, "python", lambda *args, **kwargs: calls.append(args))
    monkeypatch.setattr(publication, "build", lambda: publication.publisher.write_bytes(b"rebuilt executable"))
    publication.recover()
    assert calls[0] == ("ci_publication_cache.py", "context-check", "--context", publication.context)
    assert calls[-1][-1] == "frozen"
    assert publication.publisher.read_bytes() == b"rebuilt executable"
    assert (publication.site / "index.html").read_bytes() == site["index.html"]
    assert not (publication.site / "ci-evidence").exists()
    assert not (publication.site / "publication-status.json").exists()
    assert (publication.site / "conformance/nested/publication-status.json").read_bytes() == nested.read_bytes()
    assert (publication.site / "conformance/nested/ci-evidence/provenance.json").read_bytes() == nested_evidence.read_bytes()


@pytest.mark.parametrize("commit,run", [("b" * 40, "123"), ("a" * 40, "999")])
def test_recovery_rejects_foreign_publication_before_building(publication, monkeypatch, commit, run):
    monkeypatch.setenv("GITHUB_SERVER_URL", "https://github.com")
    monkeypatch.setenv("GITHUB_REPOSITORY", "example/repo")
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    retained = publication.temporary / "retained-publication"
    retained.mkdir()
    (retained / "publication-status.json").write_text(json.dumps({"commit": commit,
        "run_url": f"https://github.com/example/repo/actions/runs/{run}"}))
    with pytest.raises(ValueError, match="does not match this run"):
        publication.recover()
    assert not publication.publisher.exists()


@pytest.mark.parametrize("present", [False, True])
def test_deployment_recreates_only_missing_pages_transfer(publication, monkeypatch, present):
    monkeypatch.setenv("GITHUB_REPOSITORY", "example/repo")
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    artifacts = [{"name": "github-pages", "expired": False}] if present else []
    monkeypatch.setattr(publication, "run", lambda *args, **kwargs:
                        subprocess.CompletedProcess(args, 0, stdout=json.dumps([{"artifacts": []}, {"artifacts": artifacts}])))
    publication.pages_needed()
    assert Path(os.environ["GITHUB_OUTPUT"]).read_text() == f"needed={str(not present).lower()}\n"


def test_partial_cleanup_restores_accounting_even_when_native_transfer_survives(publication, monkeypatch):
    monkeypatch.setenv("GITHUB_SERVER_URL", "https://github.com")
    monkeypatch.setenv("GITHUB_REPOSITORY", "example/repo")
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    publication.payload.mkdir()
    (publication.payload / "native-pages.tar").write_bytes(b"surviving native transfer")
    retained = publication.temporary / "retained-publication"
    baseline = retained / "ci-evidence/conformance-test-baseline/complete.json"
    baseline.parent.mkdir(parents=True)
    baseline.write_bytes(b"complete baseline")
    assignment = retained / "ci-evidence/conformance-selection-conformance-check-0/selected.json"
    assignment.parent.mkdir()
    assignment.write_bytes(b"previous assignment")
    (retained / "publication-status.json").write_text(json.dumps({"commit": "a" * 40,
        "run_url": "https://github.com/example/repo/actions/runs/123"}))
    publication.selections()
    assert (publication.temporary / "baseline-selection/complete.json").read_bytes() == b"complete baseline"
    assert (publication.temporary / "assignments/selected.json").read_bytes() == b"previous assignment"
    assert (publication.payload / "native-pages.tar").read_bytes() == b"surviving native transfer"
    fresh = publication.temporary / "fresh-selections/conformance-selection-conformance-check-0/new-uuid.json"
    fresh.parent.mkdir(parents=True)
    fresh.write_bytes(b"fresh assignment")
    untouched = retained / "ci-evidence/conformance-selection-conformance-check-1/other-uuid.json"
    untouched.parent.mkdir()
    untouched.write_bytes(b"untouched lane")
    publication.selections()
    assert sorted(path.name for path in (publication.temporary / "assignments").iterdir()) == ["new-uuid.json", "other-uuid.json"]
    assert (publication.temporary / "assignments/new-uuid.json").read_bytes() == b"fresh assignment"
    fresh_baseline = publication.temporary / "fresh-baseline/new-baseline-uuid.json"
    fresh_baseline.parent.mkdir()
    fresh_baseline.write_bytes(b"fresh complete baseline")
    publication.selections()
    assert [path.name for path in (publication.temporary / "baseline-selection").iterdir()] == [fresh_baseline.name]
    assert (publication.temporary / "baseline-selection" / fresh_baseline.name).read_bytes() == fresh_baseline.read_bytes()


def test_missing_baseline_fails_even_when_download_failure_was_expected(publication):
    with pytest.raises(ValueError, match="baseline is unavailable"):
        publication.selections()
