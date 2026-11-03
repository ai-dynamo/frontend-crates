# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collect a run's publication, then remove transfers only after a successful upload."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import zipfile

import ci_plan


PUBLICATION = "conformance-publication"
CANDIDATE = "conformance-native-publication"
SITE_CANDIDATE = "conformance-site-candidate"


class Finalizer:
    def __init__(self, temporary: Path, environment: dict):
        self.environment = environment
        self.run_id = int(environment["GITHUB_RUN_ID"])
        self.revision = environment["GITHUB_SHA"]
        self.repository = environment["GITHUB_REPOSITORY"]
        if re.fullmatch(r"[\w.-]+/[\w.-]+", self.repository) is None or re.fullmatch(r"[0-9a-f]{40}", self.revision) is None:
            raise ValueError("invalid publication repository or revision")
        self.api_root = f"repos/{self.repository}/actions"
        self.directory = temporary / PUBLICATION
        self.inventory = temporary / "publication-artifacts.json"
        self.run_url = f"{environment['GITHUB_SERVER_URL']}/{self.repository}/actions/runs/{self.run_id}"
        self.temporary_names = {
            CANDIDATE, SITE_CANDIDATE, "capture-verification", "conformance-test-baseline",
            "conformance-validation", "github-pages", f"conformance-delivery-{self.revision}",
            "conformance-selection-conformance-fixtures-0",
            *[f"conformance-selection-conformance-check-{index}" for index in range(len(ci_plan.plan()))],
        }

    def api(self, endpoint: str, *, method: str = "GET", destination: Path | None = None):
        command = ["gh", "api", "--method", method, endpoint]
        if destination is not None:
            with destination.open("wb") as output:
                subprocess.run(command, stdout=output, check=True)
            return None
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return json.loads(result.stdout) if result.stdout.strip() else None

    def artifacts(self) -> list:
        artifacts = []
        page = 1
        while True:
            result = self.api(f"{self.api_root}/runs/{self.run_id}/artifacts?per_page=100&page={page}")
            for artifact in result["artifacts"]:
                # Never trust an artifact ID without its run and source binding.
                if artifact["workflow_run"]["id"] != self.run_id or artifact["workflow_run"]["head_sha"] != self.revision:
                    raise ValueError(f"artifact belongs to another run or revision: {artifact['name']}")
            artifacts.extend(result["artifacts"])
            if len(artifacts) >= result["total_count"]:
                return artifacts
            if not result["artifacts"]:
                raise ValueError("incomplete run artifact inventory")
            page += 1

    def download(self, artifact: dict, destination: Path, *, replace: bool = False) -> None:
        archive = self.inventory.parent / f"artifact-{artifact['id']}.zip"
        self.api(f"{self.api_root}/artifacts/{artifact['id']}/zip", destination=archive)
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                path = Path(member.filename)
                if path.is_absolute() or ".." in path.parts or (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError(f"unsafe artifact member: {member.filename}")
            if replace and destination.exists():
                shutil.rmtree(destination)
            destination.mkdir(parents=True, exist_ok=True)
            bundle.extractall(destination)

    def output(self, name: str, value: str) -> None:
        with Path(self.environment["GITHUB_OUTPUT"]).open("a") as output:
            output.write(f"{name}={value}\n")

    def collect(self) -> bool:
        artifacts = self.artifacts()
        self.inventory.write_text(json.dumps(artifacts) + "\n")
        latest = {item["name"]: item for item in sorted(artifacts, key=lambda item: item["id"])}
        self.directory.mkdir(parents=True)
        retained = self.inventory.parent / "previous-publication"
        if PUBLICATION in latest:
            self.download(latest[PUBLICATION], retained)
            previous = json.loads((retained / "publication-status.json").read_text())
            if previous["commit"] != self.revision or previous["run_url"] != self.run_url:
                raise ValueError("retained publication belongs to another run or revision")
            # A previous cleanup can have removed only some transfers. Merge its
            # evidence even when a leftover native candidate still exists.
            shutil.copytree(retained / "ci-evidence", self.directory / "ci-evidence")
        if CANDIDATE in latest:
            payload = self.inventory.parent / "finalizer-candidate"
            self.download(latest[CANDIDATE], payload)
            # Extract only the site. The executable and source-catalog transfer
            # archive belong to qualification and must never enter this download.
            with tarfile.open(payload / "native-pages.tar") as archive:
                if any(not (member.isfile() or member.isdir()) for member in archive.getmembers()):
                    raise ValueError("candidate site contains a link or special file")
                archive.extractall(self.directory, filter="data")
            # Failed-job reruns need these frozen source identities after the
            # transfer archives and binary have been removed. Keep plain evidence;
            # consumers rebuild the publisher from this revision when necessary.
            inputs = self.directory / "ci-evidence/publication-inputs"
            inputs.mkdir(parents=True, exist_ok=True)
            with tarfile.open(payload / "source-catalogs.tar") as archive:
                if any(not member.isfile() for member in archive.getmembers()):
                    raise ValueError("source catalogs contain a link or special file")
                archive.extractall(inputs, filter="data")
            (inputs / "cache-manifest.json").write_bytes((payload / "cache-manifest.json").read_bytes())
        elif SITE_CANDIDATE in latest:
            self.download(latest[SITE_CANDIDATE], self.directory)
        elif PUBLICATION in latest:
            # A partial rerun can start after the previous finalizer consumed its
            # transfers. Retain the previous site and evidence until re-upload.
            shutil.copytree(retained, self.directory, dirs_exist_ok=True)
        if not (self.directory / "index.html").is_file():
            print("No generated site is available; publication and cleanup skipped. See upstream job logs.")
            self.output("available", "false")
            return False
        evidence = self.directory / "ci-evidence"
        evidence.mkdir(exist_ok=True)
        for name, artifact in latest.items():
            if name in self.temporary_names - {CANDIDATE, SITE_CANDIDATE, "github-pages"}:
                self.download(artifact, evidence / name, replace=True)
        jobs = json.loads(self.environment["NEEDS_JSON"])
        validation = jobs["conformance-table"]["outputs"].get("validation", "skipped")
        label = "validated publication" if validation == "success" else "unvalidated candidate"
        status = {"commit": self.revision, "run_url": self.run_url,
                  "run_attempt": int(self.environment["GITHUB_RUN_ATTEMPT"]),
                  "validation": validation, "deployment": jobs["deploy-pages"]["result"], "status": label}
        (evidence / "finalization-results.json").write_text(json.dumps(jobs, indent=2) + "\n")
        (self.directory / "publication-status.json").write_text(json.dumps(status, indent=2) + "\n")
        self.output("available", "true")
        self.output("label", label)
        return True

    def cleanup(self, uploaded_id: int) -> None:
        current = self.artifacts()
        uploaded = [item for item in current if item["id"] == uploaded_id and item["name"] == PUBLICATION and not item["expired"]]
        if len(uploaded) != 1:
            raise ValueError("successful publication upload is not present in this run; no artifacts deleted")
        previous = json.loads(self.inventory.read_text())
        permitted = {item["id"] for item in previous if item["name"] in self.temporary_names | {PUBLICATION}}
        failures = []
        for artifact in current:
            if artifact["id"] == uploaded_id or artifact["id"] not in permitted:
                continue
            try:
                self.api(f"{self.api_root}/artifacts/{artifact['id']}", method="DELETE")
            except subprocess.CalledProcessError as error:
                # Cleanup is a supervisor boundary: try every identified transfer
                # and fail with all remaining names rather than hiding an error.
                failures.append(f"{artifact['name']} ({artifact['id']}): {error}")
        remaining = [f"{item['name']} ({item['id']})" for item in self.artifacts() if item["id"] != uploaded_id]
        if failures or remaining:
            raise RuntimeError(f"artifact cleanup failed; errors={failures}; remaining artifacts={remaining}")

    def announce(self, uploaded_id: int) -> None:
        status = json.loads((self.directory / "publication-status.json").read_text())
        link = f"{self.run_url}/artifacts/{uploaded_id}"
        body = f"**Parser conformance viewer**: [{status['status']}]({link}). Download and extract the ZIP; open index.html or serve the directory for historical selections."
        with Path(self.environment["GITHUB_STEP_SUMMARY"]).open("a") as summary:
            summary.write(body + "\n")
        ref = self.environment["GITHUB_REF_NAME"]
        if self.environment["GITHUB_EVENT_NAME"] == "push" and re.fullmatch(r"pull-request/[0-9]+", ref):
            subprocess.run(["gh", "pr", "comment", ref.split("/")[1], "--repo", self.repository,
                            "--edit-last", "--create-if-none", "--body", body], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("collect", "cleanup", "announce"))
    parser.add_argument("--artifact-id", type=int)
    args = parser.parse_args()
    if (args.phase != "collect") != bool(args.artifact_id):
        parser.error("cleanup and announce require --artifact-id; collect does not accept it")
    finalizer = Finalizer(Path(os.environ["RUNNER_TEMP"]), dict(os.environ))
    if args.phase == "collect":
        finalizer.collect()
    elif args.phase == "cleanup":
        finalizer.cleanup(args.artifact_id)
    else:
        finalizer.announce(args.artifact_id)


if __name__ == "__main__":
    main()
