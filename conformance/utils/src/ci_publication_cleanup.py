# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Delete completed CI transfers using only trusted default-branch code."""
from datetime import datetime
import json
import os
from pathlib import Path
import re
import zipfile

from ci_publication import retained_publication
from ci_publication_finalize import Finalizer, PUBLICATION, latest_artifacts


def cleanup(environment: dict) -> None:
    event = json.loads(Path(environment["GITHUB_EVENT_PATH"]).read_bytes())
    source = event["workflow_run"]
    repository = environment["GITHUB_REPOSITORY"]
    if (environment["GITHUB_EVENT_NAME"] != "workflow_run"
            or event["repository"]["full_name"] != repository
            or source["head_repository"]["full_name"] != repository):
        raise ValueError("cleanup requires a same-repository workflow_run event")
    target_environment = {**environment, "GITHUB_RUN_ID": str(source["id"]),
                          "GITHUB_SHA": source["head_sha"], "GITHUB_RUN_ATTEMPT": str(source["run_attempt"])}
    target = Finalizer(Path(environment["RUNNER_TEMP"]), target_environment)
    endpoint = f"{target.api_root}/runs/{target.run_id}"
    run = target.api(endpoint)
    if (run["id"] != source["id"] or run["head_sha"] != target.revision
            or run["run_attempt"] != source["run_attempt"] or run["status"] != "completed"
            or run["path"] != ".github/workflows/ci.yml" or run["event"] not in {"push", "workflow_dispatch"}
            or re.fullmatch(r"main|pull-request/[0-9]+", run["head_branch"]) is None):
        raise ValueError("cleanup source is not the completed conformance CI attempt")
    jobs = target.items(f"{endpoint}/attempts/{source['run_attempt']}/jobs", "jobs")
    finalizers = [job for job in jobs if job["name"] == "finalize-publication"]
    if len(finalizers) != 1:
        print("No unique finalizer job; transfers retained. See the source CI logs.")
        return
    job = finalizers[0]
    if (job["run_id"] != target.run_id or job["run_attempt"] != source["run_attempt"]
            or job["head_sha"] != target.revision or job["status"] != "completed"):
        raise ValueError("finalizer job does not match this completed run attempt")
    uploads = [step for step in job["steps"] if step["name"] == "Retain one downloadable publication"]
    if len(uploads) != 1 or uploads[0]["conclusion"] != "success":
        print("Publication upload did not succeed; transfers retained. See the source CI logs.")
        return
    artifacts = target.artifacts()
    publication = latest_artifacts(artifacts).get(PUBLICATION)
    if publication is None:
        print("No generated publication; transfers retained. See the source CI logs.")
        return
    created = datetime.fromisoformat(publication["created_at"])
    if (publication["expired"] or not datetime.fromisoformat(job["started_at"]) <= created
            <= datetime.fromisoformat(job["completed_at"])):
        raise ValueError("no fresh publication from this finalizer attempt; no artifacts deleted")
    archive = target.inventory.parent / "cleanup-publication.zip"
    target.api(f"{target.api_root}/artifacts/{publication['id']}/zip", destination=archive)
    # The contributor controls the ZIP. Read bounded metadata only; never extract
    # its scripts, imports, executables, or workflow files into the trusted checkout.
    with zipfile.ZipFile(archive) as bundle:
        metadata = bundle.getinfo("publication-status.json")
        bundle.getinfo("index.html")
        if metadata.file_size > 16_384:
            raise ValueError("publication status exceeds the cleanup metadata limit")
        status = json.loads(bundle.read(metadata))
    directory = target.inventory.parent / "cleanup-publication"
    directory.mkdir(exist_ok=True)
    (directory / "publication-status.json").write_text(json.dumps(status))
    retained_publication(directory, target_environment)
    if status["run_attempt"] != source["run_attempt"]:
        raise ValueError("publication belongs to another attempt; no artifacts deleted")
    target.inventory.write_text(json.dumps(artifacts))
    current = target.api(endpoint)
    if any(current[field] != run[field] for field in ("id", "head_sha", "run_attempt", "status")):
        raise ValueError("source run changed during collection; no artifacts deleted")
    target.cleanup(publication["id"])
    print(f"Retained one {PUBLICATION}: {target.run_url}/artifacts/{publication['id']}")


if __name__ == "__main__":
    cleanup(dict(os.environ))
