# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Combine validated execution accounting and reject missing or duplicate members."""

import argparse
import sys
from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import tomllib
import yaml
from impls import (
    BATCH_SOURCE_BLOCKERS,
    BATCH_SOURCE_IMPL,
    IMPL_SPECS,
    STREAM_IMPL_KEYS,
    UNIFIED_BATCH_IMPLS,
)
from datetime import datetime, timezone
import re
import subprocess
import urllib.request
from packaging.version import Version


def file_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def registered_inventory_modes():
    specs = {spec.key: spec for spec in IMPL_SPECS}
    result = {}
    for implementation in (*STREAM_IMPL_KEYS, *UNIFIED_BATCH_IMPLS):
        source = BATCH_SOURCE_IMPL.get(implementation, implementation)
        spec = specs[source]
        engine = source if spec.engine == "dynamo" else spec.engine
        result.setdefault(engine, []).append(implementation)
    return result


def source_rows(path):
    with path.open() as stream:
        if path.suffix == ".jsonl":
            for line in stream:
                yield json.loads(line)
        else:
            yield from json.load(stream)["combinations"]


def combine_coverage(inventory_path, jobs, sources, destination):
    inventory_bytes = inventory_path.read_bytes()
    inventory_hash = hashlib.sha256(inventory_bytes).hexdigest()
    inventory = yaml.safe_load(inventory_bytes)
    modes = registered_inventory_modes()
    expected = {}
    for target in inventory["targets"]:
        for implementation in modes[target["implementation"]]:
            identity = (implementation, target["version"])
            if identity in expected:
                raise ValueError("Duplicate inventory version and mode")
            expected[identity] = set()
    cases = {}
    job_hashes = {}
    for path in sorted(jobs.glob("*.json")):
        raw = path.read_bytes()
        job = json.loads(raw)
        if job["family"] != path.stem:
            raise ValueError("Coverage job family differs from filename")
        job_hashes[path.name] = hashlib.sha256(raw).hexdigest()
        for case in job["cases"]:
            identity = (job["family"], case["scenario"])
            if identity in cases:
                raise ValueError("Duplicate family case in coverage jobs")
            cases[identity] = (case["id"], case.get("case_id", case["id"]))
    if not cases or not expected:
        raise ValueError("Coverage requires nonempty jobs and inventory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".{os.getpid()}.pending")
    counts = Counter()
    blockers = {}
    input_hashes = []
    statuses = {"executed", "parser_error", "unsupported_capability", "source_build_blocker", "missing_evidence"}
    try:
        with temporary.open("x") as output:
            for source in sources:
                before = file_digest(source)
                for row in source_rows(source):
                    identity = (row["implementation"], row["version"])
                    case = (row["family"], row["scenario"])
                    if identity not in expected or case not in cases or row["case_id"] not in cases[case]:
                        raise ValueError("Coverage member differs from inventory or requests")
                    if case in expected[identity]:
                        raise ValueError("Duplicate coverage member")
                    if row["status"] not in statuses:
                        raise ValueError("Unknown coverage status")
                    expected[identity].add(case)
                    counts[row["status"]] += 1
                    if row["status"] == "source_build_blocker":
                        blockers.setdefault(identity, set()).add(row.get("detail", "Source/build blocker"))
                    output.write(json.dumps({**row, "case_id": cases[case][0]}, ensure_ascii=False) + "\n")
                if file_digest(source) != before:
                    raise ValueError("Coverage source changed during accounting")
                input_hashes.append({"path": str(source.resolve()), "sha256": before})
        if any(len(seen) != len(cases) for seen in expected.values()):
            raise ValueError("Missing coverage members")
        if file_digest(inventory_path) != inventory_hash or {path.name: file_digest(path) for path in jobs.glob("*.json")} != job_hashes:
            raise ValueError("Coverage inventory or jobs changed during accounting")
        summary = {
            "inventory_sha256": inventory_hash, "jobs": job_hashes,
            "sources": input_hashes, "version_modes": len(expected), "family_cases": len(cases),
            "combinations": sum(counts.values()), "counts": dict(counts),
            "source_build_blockers": [{"implementation": key[0], "version": key[1], "details": sorted(details)}
                                      for key, details in sorted(blockers.items())],
            "retained_historical_blockers": [{"implementation": key[0], "version": key[1], "detail": detail}
                                             for key, detail in sorted(BATCH_SOURCE_BLOCKERS.items()) if key not in expected],
            "path": str(destination.resolve()), "sha256": file_digest(temporary),
        }
        os.replace(temporary, destination)
        destination.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = combine_coverage(args.inventory, args.jobs, args.source, args.output)
    print(json.dumps({key: summary[key] for key in ("version_modes", "family_cases", "combinations", "counts", "sha256")}))


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "frontend-crates-conformance/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def published_v2_identity(target):
    """Read producer identities from the checksum-verified published archive."""
    version = target["version"]
    request = urllib.request.Request(
        f"https://static.crates.io/crates/dynamo-parsers-v2/dynamo-parsers-v2-{version}.crate",
        headers={"User-Agent": "frontend-crates-conformance/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        artifact = response.read()
    if hashlib.sha256(artifact).hexdigest() != target["checksum"]:
        raise ValueError(f"published v2 artifact checksum mismatch: {version}")
    prefix = f"dynamo-parsers-v2-{version}/"
    with tarfile.open(fileobj=io.BytesIO(artifact), mode="r:gz") as archive:
        contents = {}
        for name in (".cargo_vcs_info.json", "Cargo.lock"):
            members = [member for member in archive.getmembers() if member.name == prefix + name]
            if len(members) != 1 or not members[0].isfile():
                raise ValueError(f"published v2 artifact lacks one regular {name}: {version}")
            contents[name] = archive.extractfile(members[0]).read()
    commit = json.loads(contents[".cargo_vcs_info.json"])["git"]["sha1"]
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError(f"published v2 artifact has invalid commit: {version}")
    packages = tomllib.loads(contents["Cargo.lock"].decode())["package"]
    reasoning = [package for package in packages if package["name"] == "dynamo-parsers"]
    if not reasoning:
        request = urllib.request.Request(
            f"https://raw.githubusercontent.com/ai-dynamo/frontend-crates/{commit}/Cargo.toml",
            headers={"User-Agent": "frontend-crates-conformance/1.0"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            manifest = response.read()
        dependency = tomllib.loads(manifest.decode())["workspace"]["dependencies"]["dynamo-parsers"]
        reasoning_version = dependency["version"] if isinstance(dependency, dict) else dependency
        if not isinstance(reasoning_version, str) or not re.fullmatch(r"\d+\.\d+\.\d+", reasoning_version):
            raise ValueError(f"published v2 source lacks an exact reasoning dependency: {version}")
        return {"git_commit": commit, "reasoning_version": reasoning_version,
                "reasoning_source_sha256": hashlib.sha256(manifest).hexdigest()}
    if len(reasoning) != 1 or not re.fullmatch(r"\d+\.\d+\.\d+", reasoning[0]["version"]):
        raise ValueError(f"published v2 artifact lacks an exact reasoning dependency: {version}")
    return {"git_commit": commit, "reasoning_version": reasoning[0]["version"]}


def discover_dynamo_releases(previous, snapshots: Path, *, v1_minimum="11.0.0"):
    """Retain existing evidence and discover stable non-yanked producer releases."""
    snapshots.mkdir(parents=True, exist_ok=True)
    identities = {(target["implementation"], target["version"]): dict(target) for target in previous}
    if len(identities) != len(previous):
        raise ValueError("duplicate retained release identity")
    for implementation, package in (("dynamo_v1", "dynamo-parsers"), ("dynamo_v2", "dynamo-parsers-v2")):
        document = fetch(f"https://crates.io/api/v1/crates/{package}")
        (snapshots / f"{package}.json").write_text(json.dumps(document, sort_keys=True) + "\n")
        for release in document["versions"]:
            version = release["num"]
            key = (implementation, version)
            if not re.fullmatch(r"\d+\.\d+\.\d+", version):
                continue
            if key in identities:
                retained = identities[key]
                if retained["source_kind"] != "crate" or retained["checksum"] != release["checksum"]:
                    raise ValueError(f"published artifact conflicts with retained identity: {implementation} {version}")
                if "yanked" in retained or release["yanked"]:
                    retained["yanked"] = release["yanked"]
                continue
            if release["yanked"] or (implementation == "dynamo_v1" and Version(version) < Version(v1_minimum)):
                continue
            target = {"implementation": implementation, "version": version, "source_kind": "crate",
                      "crate": package, "checksum": release["checksum"], "yanked": False,
                      "published_at": release["created_at"]}
            if implementation == "dynamo_v2":
                target.update(published_v2_identity(target))
            identities[key] = target
    return sorted(identities.values(), key=lambda target: (target["implementation"], Version(target["version"])))


def freeze(output: Path, snapshots: Path, retained: Path):
    snapshots.mkdir(parents=True, exist_ok=True)
    previous = yaml.safe_load(retained.read_text())["targets"]
    targets = discover_dynamo_releases(
        [target for target in previous if target["implementation"] in {"dynamo_v1", "dynamo_v2"}], snapshots,
    )
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


def freeze_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument(
        "--retained",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "dynamo-capture-inventory.yaml",
    )
    args = parser.parse_args()
    inventory = freeze(args.output, args.snapshots, args.retained)
    print(
        json.dumps(
            {
                name: sum(
                    item["implementation"] == name for item in inventory["targets"]
                )
                for name in sorted(
                    {item["implementation"] for item in inventory["targets"]}
                )
            }
        )
    )


if __name__ == "__main__":
    if sys.argv[1:2] == ["freeze"]:
        del sys.argv[1]
        raise SystemExit(freeze_main())
    main()
