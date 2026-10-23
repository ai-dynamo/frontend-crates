#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Freeze every stable parser release, retaining yanks and missing source tags."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import urllib.request

from packaging.version import Version
import yaml


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "frontend-crates-conformance/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def freeze(output: Path, snapshots: Path, retained: Path):
    snapshots.mkdir(parents=True, exist_ok=True)
    targets = []
    previous = yaml.safe_load(retained.read_text())["targets"]
    identities = {(item["implementation"], item["version"]): item for item in previous}
    for implementation, package in (("dynamo_v1", "dynamo-parsers"), ("dynamo_v2", "dynamo-parsers-v2")):
        document = fetch(f"https://crates.io/api/v1/crates/{package}")
        (snapshots / f"{package}.json").write_text(json.dumps(document, sort_keys=True))
        for release in document["versions"]:
            version = release["num"]
            if not re.fullmatch(r"\d+\.\d+\.\d+", version):
                continue
            target = {"implementation": implementation, "version": version, "source_kind": "crate",
                      "crate": package, "checksum": release["checksum"], "yanked": release["yanked"],
                      "published_at": release["created_at"]}
            target.update(identities.get((implementation, version), {}))
            if implementation == "dynamo_v2" and "reasoning_version" not in target:
                raise ValueError(f"release lacks a reviewed matching reasoning adapter: {version}")
            targets.append(target)
    targets.extend(item for item in previous if item["source_kind"] == "git")
    for package, repository in (("vllm", "vllm-project/vllm"), ("sglang", "sgl-project/sglang")):
        document = fetch(f"https://pypi.org/pypi/{package}/json")
        (snapshots / f"{package}.json").write_text(json.dumps(document, sort_keys=True))
        tags = subprocess.run(["git", "ls-remote", "--tags", f"https://github.com/{repository}.git"], check=True, capture_output=True, text=True).stdout
        (snapshots / f"{package}-tags.json").write_text(json.dumps(dict(line.split()[::-1] for line in tags.splitlines())))
        refs = dict(line.split()[::-1] for line in tags.splitlines())
        for version, files in document["releases"].items():
            parsed = Version(version)
            if parsed.is_prerelease or parsed.is_devrelease:
                continue
            commit = refs.get(f"refs/tags/v{version}^{{}}", refs.get(f"refs/tags/v{version}"))
            target = {"implementation": package, "version": version, "source_kind": "pypi",
                      "repository": f"https://github.com/{repository}", "git_commit": commit,
                      "artifacts": [{"filename": item["filename"], "url": item["url"], "sha256": item["digests"]["sha256"],
                                     "yanked": item["yanked"], "yanked_reason": item["yanked_reason"],
                                     "requires_python": item["requires_python"]} for item in files]}
            if commit is None:
                target["source_blocker"] = "No matching upstream release tag in frozen refs"
            if not files:
                target["artifact_blocker"] = "PyPI release has no retained artifacts"
            targets.append(target)
    targets.sort(key=lambda item: (item["implementation"], Version(item["version"])))
    snapshot_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(snapshots.glob("*.json"))}
    inventory = {"frozen_at": datetime.now(timezone.utc).isoformat(), "corpus": "UNIFIED",
                 "policy": "every published stable release; retained source labels remain distinct",
                 "registry_snapshots": snapshot_hashes, "targets": targets}
    output.write_text(yaml.safe_dump(inventory, sort_keys=False, width=4096))
    return inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--retained", type=Path, default=Path(__file__).resolve().parents[2] / "dynamo-capture-inventory.yaml")
    args = parser.parse_args()
    inventory = freeze(args.output, args.snapshots, args.retained)
    print(json.dumps({name: sum(item["implementation"] == name for item in inventory["targets"]) for name in sorted({item["implementation"] for item in inventory["targets"]})}))


if __name__ == "__main__":
    main()
