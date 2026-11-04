# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import io
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import tarfile
import zipfile

import pytest
import yaml

from ci_publication_finalize import CANDIDATE, INVENTORY, PUBLICATION, SITE_CANDIDATE, Finalizer


class ArtifactService:
    def __init__(self):
        self.items = {}
        self.bundles = {}
        self.deleted = []
        self.fail_deletion = set()
        self.fail_download = set()

    def add(self, name, files, *, run=123, revision="a" * 40):
        identifier = max(self.bundles, default=0) + 1
        self.items[identifier] = {"id": identifier, "name": name, "expired": False,
                                  "created_at": datetime.fromtimestamp(identifier, timezone.utc).isoformat(),
                                  "workflow_run": {"id": run, "head_sha": revision}}
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w") as bundle:
            for path, content in files.items():
                bundle.writestr(path, content)
        self.bundles[identifier] = output.getvalue()
        return identifier

    def api(self, endpoint, *, method="GET", destination=None):
        if "/runs/" in endpoint:
            assert "/runs/123/artifacts?" in endpoint
            return {"artifacts": list(self.items.values()), "total_count": len(self.items)}
        identifier = int(endpoint.split("/artifacts/")[1].split("/")[0])
        if method == "DELETE":
            if identifier in self.fail_deletion:
                raise subprocess.CalledProcessError(1, ["gh", "api", endpoint])
            self.deleted.append(identifier)
            del self.items[identifier]
        else:
            if identifier in self.fail_download:
                raise subprocess.CalledProcessError(1, ["gh", "api", endpoint])
            destination.write_bytes(self.bundles[identifier])


def candidate_files(html=b"<html>report</html>"):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as archive:
        for path, content in {"index.html": html, "history/data.json.gz": b"history", ".nojekyll": b""}.items():
            member = tarfile.TarInfo(path)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    site = output.getvalue()
    sources = io.BytesIO()
    with tarfile.open(fileobj=sources, mode="w") as archive:
        member = tarfile.TarInfo("conformance/publication-context.json")
        member.size = 2
        archive.addfile(member, io.BytesIO(b"{}"))
    return {"native-pages.tar": site, "source-catalogs.tar": sources.getvalue(),
            "conformance-publish": b"executable", "cache-manifest.json": b'{"identity":"source-identity"}'}


@pytest.fixture
def finalizer(tmp_path, monkeypatch):
    service = ArtifactService()
    environment = {"GITHUB_RUN_ID": "123", "GITHUB_SHA": "a" * 40, "GITHUB_REPOSITORY": "example/repo",
                   "GITHUB_SERVER_URL": "https://github.com", "GITHUB_RUN_ATTEMPT": "1",
                   "GITHUB_OUTPUT": str(tmp_path / "outputs"), "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
                   "GITHUB_EVENT_NAME": "push", "GITHUB_REF_NAME": "pull-request/42",
                   "NEEDS_JSON": json.dumps({"conformance-table": {"result": "success", "outputs": {"validation": "success"}},
                                              "deploy-pages": {"result": "skipped", "outputs": {}}})}
    result = Finalizer(tmp_path, environment)
    monkeypatch.setattr(result, "api", service.api)
    return result, service


@pytest.mark.parametrize("validation,deployment", [("success", "success"), ("failure", "skipped"),
                                                   ("skipped", "skipped"), ("success", "failure")])
def test_publication_contains_site_and_available_diagnostics(finalizer, validation, deployment):
    target, service = finalizer
    service.add(CANDIDATE, candidate_files())
    for name in ("conformance-test-baseline", "conformance-selection-conformance-fixtures-0",
                 "conformance-selection-conformance-check-0", "capture-verification", "conformance-validation",
                 "conformance-delivery-" + "a" * 40):
        service.add(name, {"evidence.json": b'{"result":"observed"}'})
    target.environment["NEEDS_JSON"] = json.dumps({"conformance-table": {"result": validation, "outputs": {"validation": validation}},
                                                   "deploy-pages": {"result": deployment}})
    assert target.collect()
    status = json.loads((target.directory / "publication-status.json").read_text())
    assert status["commit"] == "a" * 40
    assert status["run_url"] == "https://github.com/example/repo/actions/runs/123"
    assert status["validation"] == validation
    assert status["deployment"] == deployment
    assert status["status"] == ("validated publication" if validation == "success" else "unvalidated candidate")
    assert (target.directory / "history/data.json.gz").read_bytes() == b"history"
    assert (target.directory / ".nojekyll").is_file()
    assert len(list((target.directory / "ci-evidence").rglob("evidence.json"))) == 6
    members = {path.relative_to(target.directory).as_posix() for path in target.directory.rglob("*") if path.is_file()}
    assert "index.html" in members
    assert not members.intersection({"native-pages.tar", "source-catalogs.tar", "conformance-publish"})
    assert service.deleted == []
    assert (target.directory / "ci-evidence/publication-inputs/conformance/publication-context.json").is_file()
    inventory = json.loads((target.directory / INVENTORY).read_text())
    assert inventory == {"version": 1, "artifacts": [{"id": item["id"], "name": item["name"]}
                                                     for item in service.items.values()]}


def test_missing_evidence_does_not_hide_generated_candidate(finalizer):
    target, service = finalizer
    service.add(SITE_CANDIDATE, {"index.html": b"generated before build failure"})
    target.environment["NEEDS_JSON"] = json.dumps({"conformance-table": {"result": "failure", "outputs": {}},
                                                   "deploy-pages": {"result": "skipped"}})
    assert target.collect()
    assert json.loads((target.directory / "publication-status.json").read_text())["status"] == "unvalidated candidate"


def test_no_site_uploads_nothing_and_retains_transfers_in_logs(finalizer, capsys):
    target, service = finalizer
    service.add("conformance-validation", {"failure.log": b"no build"})
    assert not target.collect()
    assert "No generated site" in capsys.readouterr().out
    assert Path(target.environment["GITHUB_OUTPUT"]).read_text() == "available=false\n"
    assert service.deleted == []


def test_cleanup_removes_only_snapshot_transfers_after_confirmed_upload(finalizer):
    target, service = finalizer
    candidate = service.add(CANDIDATE, candidate_files())
    pages = service.add("github-pages", {"artifact.tar": b"pages"})
    target.collect()
    uploaded = service.add(PUBLICATION, {"index.html": b"new"})
    target.cleanup(uploaded)
    assert service.deleted == [candidate, pages]
    assert list(service.items) == [uploaded]


def test_upload_failure_preserves_old_publication_and_every_transfer(finalizer):
    target, service = finalizer
    old = service.add(PUBLICATION, {"index.html": b"old",
        "ci-evidence/previous.json": b"{}",
        "publication-status.json": json.dumps({"commit": target.revision, "run_url": target.run_url})})
    candidate = service.add(CANDIDATE, candidate_files())
    target.collect()
    with pytest.raises(ValueError, match="no artifacts deleted"):
        target.cleanup(999)
    assert set(service.items) == {old, candidate}
    assert service.deleted == []


def test_cleanup_failure_names_remaining_artifacts_and_continues(finalizer):
    target, service = finalizer
    candidate = service.add(CANDIDATE, candidate_files())
    evidence = service.add("capture-verification", {"result.json": b"{}"})
    target.collect()
    uploaded = service.add(PUBLICATION, {"index.html": b"new"})
    service.fail_deletion.add(candidate)
    with pytest.raises(RuntimeError, match="remaining artifacts=.*conformance-native-publication"):
        target.cleanup(uploaded)
    assert service.deleted == [evidence]
    assert set(service.items) == {candidate, uploaded}


@pytest.mark.parametrize("new_transfer", [False, True])
def test_unrecognized_or_newly_arriving_artifacts_are_retained_and_reported(finalizer, new_transfer):
    target, service = finalizer
    service.add(CANDIDATE, candidate_files())
    unknown = service.add("unrelated", {"evidence": b"keep"})
    target.collect()
    assert unknown not in {item["id"] for item in json.loads((target.directory / INVENTORY).read_text())["artifacts"]}
    if new_transfer:
        unknown = service.add("capture-verification", {"evidence": b"arrived after snapshot"})
    uploaded = service.add(PUBLICATION, {"index.html": b"new"})
    with pytest.raises(RuntimeError, match="remaining artifacts"):
        target.cleanup(uploaded)
    assert unknown in service.items


@pytest.mark.parametrize("run,revision", [(999, "a" * 40), (123, "b" * 40)])
def test_foreign_run_or_revision_fails_before_any_mutation(finalizer, run, revision):
    target, service = finalizer
    service.add(CANDIDATE, candidate_files(), run=run, revision=revision)
    with pytest.raises(ValueError, match="another run or revision"):
        target.collect()
    assert not target.inventory.exists()
    assert service.deleted == []


@pytest.mark.parametrize("field,value", [("commit", "b" * 40), ("run_url", "https://github.com/example/repo/actions/runs/999")])
def test_retained_payload_identity_is_checked_even_when_artifact_metadata_matches(finalizer, field, value):
    target, service = finalizer
    status = {"commit": target.revision, "run_url": target.run_url, field: value}
    service.add(PUBLICATION, {"index.html": b"old", "publication-status.json": json.dumps(status)})
    with pytest.raises(ValueError, match="does not match this run or revision"):
        target.collect()
    assert service.deleted == []


def test_partial_rerun_reuses_site_and_evidence_then_replaces_old_publication(finalizer, monkeypatch):
    target, service = finalizer
    service.add(CANDIDATE, candidate_files())
    service.add("capture-verification", {"evidence.json": b"retained"})
    target.collect()
    files = {path.relative_to(target.directory).as_posix(): path.read_bytes() for path in target.directory.rglob("*") if path.is_file()}
    old = service.add(PUBLICATION, files)
    target.cleanup(old)
    rerun_dir = target.inventory.parent / "rerun"
    rerun_dir.mkdir()
    environment = {**target.environment, "GITHUB_RUN_ATTEMPT": "2"}
    rerun = Finalizer(rerun_dir, environment)
    monkeypatch.setattr(rerun, "api", service.api)
    assert rerun.collect()
    assert old in service.items  # The replacement has not uploaded yet.
    assert (rerun.directory / "ci-evidence/capture-verification/evidence.json").read_bytes() == b"retained"
    replacement = service.add(PUBLICATION, {"index.html": b"replacement"})
    rerun.cleanup(replacement)
    assert list(service.items) == [replacement]
    rerun.cleanup(replacement)  # Retrying cleanup is harmless.


def test_rerun_after_partial_cleanup_preserves_already_consolidated_evidence(finalizer, monkeypatch):
    target, service = finalizer
    candidate = service.add(CANDIDATE, candidate_files())
    service.add("capture-verification", {"evidence.json": b"only copy after deletion"})
    target.collect()
    files = {path.relative_to(target.directory).as_posix(): path.read_bytes() for path in target.directory.rglob("*") if path.is_file()}
    old = service.add(PUBLICATION, files)
    service.fail_deletion.add(candidate)
    with pytest.raises(RuntimeError, match="cleanup failed"):
        target.cleanup(old)
    assert set(service.items) == {candidate, old}
    rerun_dir = target.inventory.parent / "retry"
    rerun_dir.mkdir()
    rerun = Finalizer(rerun_dir, {**target.environment, "GITHUB_RUN_ATTEMPT": "2"})
    monkeypatch.setattr(rerun, "api", service.api)
    assert rerun.collect()
    assert (rerun.directory / "ci-evidence/capture-verification/evidence.json").read_bytes() == b"only copy after deletion"
    service.fail_deletion.clear()
    new = service.add(PUBLICATION, {"index.html": b"new"})
    rerun.cleanup(new)
    assert set(service.items) == {new}


def test_summary_and_pr_comment_use_the_exact_publication_link(finalizer, monkeypatch):
    target, service = finalizer
    service.add(CANDIDATE, candidate_files())
    target.collect()
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: calls.append(command))
    target.announce(456)
    summary = Path(target.environment["GITHUB_STEP_SUMMARY"]).read_text()
    assert "[validated publication](https://github.com/example/repo/actions/runs/123/artifacts/456)" in summary
    assert calls[0][:4] == ["gh", "pr", "comment", "42"]
    assert calls[0][-1] == summary.strip()


def test_fresh_uuid_receipts_replace_retained_artifact_owner(finalizer):
    target, service = finalizer
    name = "conformance-selection-conformance-check-0"
    service.add(PUBLICATION, {"index.html": b"old",
        f"ci-evidence/{name}/old-uuid.json": b"old assignment",
        "publication-status.json": json.dumps({"commit": target.revision, "run_url": target.run_url})})
    service.add(CANDIDATE, candidate_files())
    service.add(name, {"new-uuid.json": b"new assignment"})
    assert target.collect()
    owner = target.directory / "ci-evidence" / name
    assert [path.name for path in owner.iterdir()] == ["new-uuid.json"]


@pytest.mark.parametrize("name,phase", [
    (PUBLICATION, "collect"), (CANDIDATE, "collect"), ("capture-verification", "collect"),
    (PUBLICATION, "restore"), (CANDIDATE, "restore"),
    ("conformance-test-baseline", "receipts"), ("conformance-selection-conformance-check-0", "receipts"),
])
def test_latest_artifact_uses_creation_time_when_github_ids_decrease(finalizer, name, phase):
    target, service = finalizer
    identifiers = []
    for content in (b"old", b"new"):
        if name == PUBLICATION:
            files = {"index.html": content, "ci-evidence/result.json": content,
                     "publication-status.json": json.dumps({"commit": target.revision, "run_url": target.run_url})}
        elif name == CANDIDATE:
            files = candidate_files(content)
        else:
            files = {"result.json": content}
        identifiers.append(service.add(name, files))
    if name == "capture-verification":
        service.add(CANDIDATE, candidate_files())
    # Real hosted reruns allocated a lower ID to the later publication.
    for original, replacement in zip(identifiers, (100, 50)):
        artifact = service.items.pop(original)
        artifact["id"] = replacement
        service.items[replacement] = artifact
        service.bundles[replacement] = service.bundles.pop(original)
    if phase == "collect":
        assert target.collect()
        result = (target.directory / "ci-evidence/capture-verification/result.json"
                  if name == "capture-verification" else target.directory / "index.html")
    elif phase == "restore":
        target.restore()
        if name == CANDIDATE:
            with tarfile.open(target.inventory.parent / "native-publication/native-pages.tar") as archive:
                assert archive.extractfile("index.html").read() == b"new"
            assert Path(target.environment["GITHUB_OUTPUT"]).read_text() == "recovered=false\n"
            return
        result = target.inventory.parent / "retained-publication/index.html"
        assert Path(target.environment["GITHUB_OUTPUT"]).read_text() == "recovered=true\n"
    else:
        target.receipts()
        result = (target.inventory.parent / "fresh-baseline/result.json" if name == "conformance-test-baseline"
                  else target.inventory.parent / "fresh-selections" / name / "result.json")
    assert result.read_bytes() == b"new"


def test_restore_download_failure_does_not_fall_back_to_an_older_publication(finalizer):
    target, service = finalizer
    candidate = service.add(CANDIDATE, candidate_files())
    service.add(PUBLICATION, {"index.html": b"older publication"})
    service.fail_download.add(candidate)
    with pytest.raises(subprocess.CalledProcessError):
        target.restore()
    assert not (target.inventory.parent / "retained-publication").exists()
    assert not Path(target.environment["GITHUB_OUTPUT"]).exists()


def test_restore_requires_a_candidate_or_retained_publication(finalizer):
    target, service = finalizer
    with pytest.raises(ValueError, match="no candidate or retained publication"):
        target.restore()


def test_receipts_restore_prior_evidence_after_only_some_transfers_were_deleted(finalizer):
    target, service = finalizer
    service.add(PUBLICATION, {"ci-evidence/old.json": b"retained evidence"})
    service.add("conformance-selection-conformance-check-0", {"new.json": b"fresh receipt"})
    target.receipts()
    assert (target.inventory.parent / "retained-publication/ci-evidence/old.json").read_bytes() == b"retained evidence"
    assert (target.inventory.parent / "fresh-selections/conformance-selection-conformance-check-0/new.json").read_bytes() == b"fresh receipt"


def test_finalization_workflow_keeps_contributor_code_read_only():
    root = Path(__file__).resolve().parents[3]
    workflow = yaml.safe_load((root / ".github/workflows/ci.yml").read_text())
    jobs = workflow["jobs"]
    finalizer = jobs["finalize-publication"]
    assert finalizer["if"] == "${{ always() }}"
    assert set(finalizer["needs"]) == {"conformance-table", "deploy-pages"}
    assert workflow["permissions"] == {"contents": "read"}
    assert not [name for name, job in jobs.items() if job.get("permissions", {}).get("actions") == "write"]
    assert finalizer["permissions"]["actions"] == "read"
    for name in ("conformance-check", "conformance-table", "deploy-pages"):
        assert jobs[name]["permissions"]["actions"] == "read"
    steps = finalizer["steps"]
    upload = next(index for index, step in enumerate(steps) if step.get("id") == "publication")
    assert not any(" cleanup " in step.get("run", "") for step in steps)
    assert steps[upload]["with"]["name"] == PUBLICATION
    assert steps[upload]["name"] == "Retain one downloadable publication"
    assert steps[upload]["with"]["retention-days"] == 7
    assert not steps[upload]["with"].get("overwrite", False)
    assert jobs["conformance-table"]["outputs"]["validation"] == "${{ steps.accounting.outcome }}"
    verifiers = [step for step in jobs["conformance-table"]["steps"] if "verify_ci_selection.py" in step.get("run", "")]
    assert len(verifiers) == 2
    assert all(step["shell"] == "bash" for step in verifiers)
    gate = jobs["conformance-table"]["steps"]
    restore = next(index for index, step in enumerate(gate) if "ci_publication.py selections" in step.get("run", ""))
    receipts = next(index for index, step in enumerate(gate) if "ci_publication_finalize.py receipts" in step.get("run", ""))
    assert receipts < restore
    assert "if" not in gate[restore]
    composite = yaml.safe_load((root / ".github/actions/conformance-publication/action.yml").read_text())
    assert any("ci_publication_finalize.py restore" in step.get("run", "") for step in composite["runs"]["steps"])
    assert "actions/download-artifact" not in json.dumps(workflow) + json.dumps(composite)
    failed_pipeline = subprocess.run(["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c",
                                      'python3 -c "raise SystemExit(23)" | tee /dev/null'], capture_output=True)
    assert failed_pipeline.returncode == 23


def test_cleanup_workflow_uses_default_branch_code_and_serializes_only_same_run():
    root = Path(__file__).resolve().parents[3]
    workflow = yaml.load((root / ".github/workflows/conformance-cleanup.yml").read_text(), Loader=yaml.BaseLoader)
    ci = yaml.safe_load((root / ".github/workflows/ci.yml").read_text())
    assert workflow["on"] == {"workflow_run": {"workflows": ["ci"], "types": ["completed"]}}
    job = workflow["jobs"]["cleanup"]
    assert workflow["permissions"] == {"contents": "read"}
    assert job["permissions"] == {"contents": "read", "actions": "write"}
    assert "head_repository.full_name == github.repository" in job["if"]
    checkout, cleanup = job["steps"]
    assert checkout["with"] == {"ref": "${{ github.workflow_sha }}", "persist-credentials": "false"}
    assert cleanup["run"] == "python3 conformance/utils/src/ci_publication_cleanup.py"
    assert cleanup["env"] == {"GH_TOKEN": "${{ github.token }}"}
    assert ci["concurrency"] == {"group": "conformance-artifacts-${{ github.run_id }}", "cancel-in-progress": False}
    assert job["concurrency"] == {"group": "conformance-artifacts-${{ github.event.workflow_run.id }}", "cancel-in-progress": "false"}
