# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
import subprocess

import pytest

from ci_publication_cleanup import cleanup
from ci_publication_finalize import Finalizer, PUBLICATION
from test_ci_publication_finalize import ArtifactService


class CleanupService(ArtifactService):
    def __init__(self):
        super().__init__()
        self.run = {"id": 123, "head_sha": "a" * 40, "run_attempt": 1, "status": "completed",
                    "path": ".github/workflows/ci.yml", "event": "push", "head_branch": "pull-request/42"}
        self.jobs = [{"name": "finalize-publication", "conclusion": "success",
                      "run_id": 123, "run_attempt": 1, "head_sha": "a" * 40, "status": "completed",
                      "started_at": "1970-01-01T00:00:00+00:00", "completed_at": "2040-01-01T00:00:00+00:00"}]
        self.changed_run = False
        self.run_reads = 0

    def api(self, endpoint, *, method="GET", destination=None):
        if endpoint.endswith("/runs/123"):
            self.run_reads += 1
            return {**self.run, **({"status": "in_progress"} if self.changed_run and self.run_reads > 1 else {})}
        if "/attempts/1/jobs?" in endpoint:
            page = int(endpoint.split("page=")[-1])
            return {"jobs": self.jobs[(page - 1) * 100:page * 100], "total_count": len(self.jobs)}
        return super().api(endpoint, method=method, destination=destination)


@pytest.fixture
def source(tmp_path, monkeypatch):
    service = CleanupService()
    event = {"repository": {"full_name": "example/repo"},
             "workflow_run": {"id": 123, "head_sha": "a" * 40, "run_attempt": 1,
                              "head_repository": {"full_name": "example/repo"}}}
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps(event))
    environment = {"GITHUB_REPOSITORY": "example/repo", "GITHUB_EVENT_NAME": "workflow_run",
                   "GITHUB_EVENT_PATH": str(event_path), "RUNNER_TEMP": str(tmp_path),
                   "GITHUB_SERVER_URL": "https://github.com"}
    monkeypatch.setattr(Finalizer, "api", lambda self, *args, **kwargs: service.api(*args, **kwargs))
    status = {"commit": "a" * 40, "run_url": "https://github.com/example/repo/actions/runs/123", "run_attempt": 1}
    return service, environment, event, status


def publish(service, status, **extra):
    return service.add(PUBLICATION, {"index.html": b"site", "publication-status.json": json.dumps(status), **extra})


@pytest.mark.parametrize("branch", ["main", "pull-request/42"])
def test_completed_run_keeps_one_publication_without_executing_its_files(source, branch):
    service, environment, _, status = source
    service.run["head_branch"] = branch
    target = Finalizer(Path(environment["RUNNER_TEMP"]), {
        **environment, "GITHUB_RUN_ID": "123", "GITHUB_SHA": "a" * 40})
    for name in sorted(target.temporary_names):
        service.add(name, {"evidence": b"data"})
    published = publish(service, status, **{"../../pwned.py": b"raise AssertionError('untrusted artifact executed')",
                                          "ci_publication.py": b"raise AssertionError('untrusted import')"})
    cleanup(environment)
    assert list(service.items) == [published]
    root = Path(environment["RUNNER_TEMP"])
    assert sorted(path.name for path in root.iterdir()) == ["cleanup-publication", "cleanup-publication.zip", "event.json", "publication-artifacts.json"]
    assert sorted(path.name for path in (root / "cleanup-publication").iterdir()) == ["publication-status.json"]


def test_cleanup_can_repeat_without_deleting_the_retained_publication(source):
    service, environment, _, status = source
    service.add("conformance-validation", {"result.json": b"{}"})
    published = publish(service, status)
    cleanup(environment)
    deleted = list(service.deleted)
    cleanup(environment)
    assert list(service.items) == [published]
    assert service.deleted == deleted


@pytest.mark.parametrize("field,value", [("head_sha", "b" * 40), ("run_attempt", 2), ("status", "in_progress"),
                                        ("path", ".github/workflows/release.yml"), ("event", "pull_request"),
                                        ("head_branch", "feature/untrusted")])
def test_mismatched_run_cannot_delete_artifacts(source, field, value):
    service, environment, _, status = source
    publish(service, status)
    service.run[field] = value
    with pytest.raises(ValueError, match="not the completed conformance CI attempt"):
        cleanup(environment)
    assert service.deleted == []


@pytest.mark.parametrize("location", ["repository", "head_repository", "event"])
def test_only_same_repository_completion_events_are_accepted(source, location):
    service, environment, event, status = source
    publish(service, status)
    if location == "event":
        environment["GITHUB_EVENT_NAME"] = "workflow_dispatch"
    elif location == "repository":
        event["repository"]["full_name"] = "other/repo"
    else:
        event["workflow_run"]["head_repository"]["full_name"] = "fork/repo"
    Path(environment["GITHUB_EVENT_PATH"]).write_text(json.dumps(event))
    with pytest.raises(ValueError, match="same-repository workflow_run"):
        cleanup(environment)
    assert service.deleted == []


@pytest.mark.parametrize("field,value", [("commit", "b" * 40), ("run_url", "https://github.com/example/repo/actions/runs/999"),
                                        ("run_attempt", 2)])
def test_publication_metadata_must_match_run_and_attempt(source, field, value):
    service, environment, _, status = source
    status[field] = value
    publish(service, status)
    with pytest.raises(ValueError, match="another attempt|does not match this run"):
        cleanup(environment)
    assert service.deleted == []


@pytest.mark.parametrize("finalizers", [[], [{"name": "finalize-publication", "conclusion": "failure"}]])
def test_missing_or_failed_finalizer_preserves_all_transfers(source, finalizers, capsys):
    service, environment, _, status = source
    service.jobs = finalizers
    published = publish(service, status)
    service.fail_download.add(published)
    cleanup(environment)
    assert service.deleted == []
    assert "Finalization did not succeed" in capsys.readouterr().out


def test_no_site_preserves_failure_transfers(source, capsys):
    service, environment, _, _ = source
    service.add("conformance-validation", {"failure.log": b"no site"})
    cleanup(environment)
    assert service.deleted == []
    assert "No generated publication" in capsys.readouterr().out


@pytest.mark.parametrize("created", ["1969-01-01T00:00:00+00:00", "2041-01-01T00:00:00+00:00"])
def test_old_or_late_publication_cannot_authorize_cleanup(source, created):
    service, environment, _, status = source
    published = publish(service, status)
    service.items[published]["created_at"] = created
    with pytest.raises(ValueError, match="no fresh publication"):
        cleanup(environment)
    assert service.deleted == []


def test_new_attempt_during_collection_preserves_transfers(source):
    service, environment, _, status = source
    publish(service, status)
    service.changed_run = True
    with pytest.raises(ValueError, match="source run changed"):
        cleanup(environment)
    assert service.deleted == []


@pytest.mark.parametrize("field,value", [("run_id", 999), ("run_attempt", 2), ("head_sha", "b" * 40), ("status", "in_progress")])
def test_finalizer_job_identity_is_checked_before_cleanup(source, field, value):
    service, environment, _, status = source
    publish(service, status)
    service.jobs[0][field] = value
    with pytest.raises(ValueError, match="finalizer job does not match"):
        cleanup(environment)
    assert service.deleted == []


def test_metadata_read_is_bounded(source):
    service, environment, _, status = source
    status["padding"] = "x" * 16_384
    publish(service, status)
    with pytest.raises(ValueError, match="metadata limit"):
        cleanup(environment)
    assert service.deleted == []


def test_jobs_are_paginated_before_accepting_finalizer(source):
    service, environment, _, status = source
    service.jobs = [{"name": "other-job"}] * 100 + service.jobs
    published = publish(service, status)
    cleanup(environment)
    assert list(service.items) == [published]


def test_download_failure_prevents_cleanup(source):
    service, environment, _, status = source
    service.fail_download.add(publish(service, status))
    with pytest.raises(subprocess.CalledProcessError):
        cleanup(environment)
    assert service.deleted == []


def test_missing_metadata_prevents_cleanup(source):
    service, environment, _, _ = source
    service.add(PUBLICATION, {"index.html": b"candidate"})
    with pytest.raises(KeyError, match="publication-status"):
        cleanup(environment)
    assert service.deleted == []


def test_foreign_artifact_metadata_prevents_cleanup(source):
    service, environment, _, status = source
    published = publish(service, status)
    service.items[published]["workflow_run"]["id"] = 999
    with pytest.raises(ValueError, match="another run or revision"):
        cleanup(environment)
    assert service.deleted == []
