#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execute a frozen Dynamo release inventory using the shared family jobs and adapters."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import queue
import subprocess
import tomllib
from pathlib import Path

import yaml
import yaml_fast  # noqa: F401

import capture_dynamo_unified
import capture_dynamo_unified_batch
from capture_unified_job import build_job
from capture_stimulus import execution_outcome, validate_execution, validate_adapter_bundle

ROOT = Path(__file__).resolve().parents[3]


def freeze_adapters(work: Path) -> str:
    manifest = {"schema_version": 1, "bundles": {}, "implementations": {}, "captures": {}}
    for implementation in ("dynamo_v1", "dynamo_v1_jail", "dynamo_v2"):
        paths = capture_dynamo_unified.adapter_files() if implementation == "dynamo_v2" else capture_dynamo_unified_batch.adapter_files(implementation.endswith("_jail"))
        contents = [(str(path.relative_to(ROOT)), path.read_bytes()) for path in paths]
        digest = hashlib.sha256(b"".join(value for _path, value in contents)).hexdigest()
        directory = work / "adapter" / digest
        entries = []
        for relative, value in contents:
            path = directory / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
            entries.append({"path": relative, "sha256": hashlib.sha256(value).hexdigest()})
        included = {relative for relative, _value in contents}
        support = []
        for path in (ROOT / "conformance/utils/src/capture_stimulus.py", ROOT / "conformance/utils/src/impls.py"):
            relative = str(path.relative_to(ROOT))
            if relative in included:
                continue
            value = path.read_bytes()
            destination = directory / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(value)
            support.append({"path": relative, "sha256": hashlib.sha256(value).hexdigest()})
        manifest["bundles"][digest] = {"directory": str(directory.relative_to(work)), "files": entries, "support_files": support}
        manifest["implementations"][implementation] = digest
    data = (json.dumps(manifest, sort_keys=True) + "\n").encode()
    (work / "adapter-manifest.json").write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def validate_frozen_adapters(work: Path, result: dict) -> dict:
    path = work / "adapter-manifest.json"
    expected = result.get("adapter_manifest_sha256")
    if not isinstance(expected, str) or len(expected) != 64 or path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("frozen adapter manifest differs from execution run")
    manifest = json.loads(path.read_text())
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1 or not isinstance(manifest.get("bundles"), dict) or not manifest["bundles"]:
        raise ValueError("invalid frozen adapter manifest")
    for digest, bundle in manifest["bundles"].items():
        directory = bundle.get("directory")
        if not isinstance(directory, str) or not directory or "\\" in directory or Path(directory).is_absolute() or any(part in {"", ".", ".."} for part in directory.split("/")):
            raise ValueError("unsafe frozen adapter bundle directory")
        if any((work / Path(*Path(directory).parts[:index])).is_symlink() for index in range(1, len(Path(directory).parts) + 1)):
            raise ValueError("symbolic frozen adapter bundle directory")
        if validate_adapter_bundle(work / directory, bundle) != digest:
            raise ValueError("frozen adapter bundle digest mismatch")
    for key in ("implementations", "captures"):
        if not isinstance(manifest.get(key), dict) or any(value not in manifest["bundles"] for value in manifest[key].values()):
            raise ValueError("invalid frozen adapter binding")
    for relative in manifest["captures"]:
        if not isinstance(relative, str) or not relative.startswith("captures/") or "\\" in relative or any(part in {"", ".", ".."} for part in relative.split("/")):
            raise ValueError("unsafe frozen capture adapter binding")
    return manifest


def validate_adapter_binding(manifest: dict, identity: str, relative: str) -> str:
    digest = manifest["captures"].get(relative, manifest["implementations"].get(identity))
    if digest is None:
        raise ValueError("capture has no frozen adapter binding")
    bundle = manifest["bundles"][digest]
    names = {Path(item["path"]).name for item in bundle["files"]}
    support = {Path(item["path"]).name for item in bundle.get("support_files", [])}
    required = {"legacy_batch_probe.rs", "parser_families.yaml"}
    if identity == "dynamo_v2":
        required = {path.name for path in capture_dynamo_unified.adapter_files()}
    elif identity == "dynamo_v1_jail":
        required.add("legacy_jail_probe.rs")
    if not required <= names or "capture_dynamo_unified_batch.py" not in names | support or not {"capture_stimulus.py", "impls.py"} <= support:
        raise ValueError("frozen adapter omits executed producer closure")
    return digest


def run_inventory(inventory: Path, work: Path, workers: int, versions: list[str] | None = None, target_root: Path | None = None, families: list[str] | None = None) -> dict:
    targets = yaml.safe_load(inventory.read_text())["targets"]
    if versions:
        targets = [target for target in targets if target["version"] in versions]
    if not targets:
        raise ValueError("empty release inventory")
    work.mkdir(parents=True, exist_ok=False)
    adapter_manifest_sha256 = freeze_adapters(work)
    store = ROOT / "conformance/fixtures-unified-v2/families"
    registered = {path.parent.name for path in store.glob("*/inputs_and_golden.yaml")}
    if families and set(families) - registered:
        raise ValueError("requested family is outside the Unified corpus")
    jobs = {}
    for mode in ("stream", "batch"):
        directory = work / "jobs" / mode
        directory.mkdir(parents=True)
        jobs[mode] = []
        for family_path in sorted(store.glob("*/inputs_and_golden.yaml")):
            family = family_path.parent.name
            if families and family not in families:
                continue
            path = directory / (family + ".json")
            path.write_text(json.dumps(build_job(store, family, mode=mode), ensure_ascii=False) + "\n")
            jobs[mode].append(path)
    frozen = {str(path.relative_to(work)): hashlib.sha256(path.read_bytes()).hexdigest() for paths in jobs.values() for path in paths}
    (work / "frozen-jobs.json").write_text(json.dumps(frozen, sort_keys=True))
    (work / "inventory.yaml").write_bytes(inventory.read_bytes())
    slots = queue.Queue()
    for index in range(workers):
        slots.put((target_root or work) / f"cargo-target-{index}")

    def execute(target):
        target_dir = slots.get()
        try:
            implementation = target["implementation"]
            version = target["version"]
            producer = work / "producers" / f"{implementation}-{version}"
            outputs = []
            if implementation == "dynamo_v2":
                try:
                    outputs = capture_dynamo_unified.capture(target, jobs["stream"], work / "captures" / f"{implementation}-{version}", producer, target_dir)
                except (RuntimeError, OSError, ValueError, subprocess.CalledProcessError) as error:
                    return {"implementation": implementation, "version": version, "blocker": str(error), "producer": str(producer)}
            else:
                for jail in (False, True):
                    mode_impl = "dynamo_v1_jail" if jail else implementation
                    try:
                        captured = capture_dynamo_unified_batch.capture_jobs(version, jobs["batch"], work / "captures" / f"{mode_impl}-{version}",
                            producer / mode_impl, target_dir, jail_buffer=jail,
                            lockfile=producer / "dynamo_v1" / jobs["batch"][0].stem / "Cargo.lock" if jail else None)
                        for output in captured:
                            receipt = json.loads(output.read_text())["provenance"]
                            if receipt["crate_checksum"] != target["checksum"]:
                                raise ValueError("v1 producer artifact differs from frozen inventory")
                            outputs.append(output)
                    except (RuntimeError, OSError, ValueError, subprocess.CalledProcessError) as error:
                        for job in jobs["batch"]:
                            outputs.append({"implementation": mode_impl, "family": job.stem, "blocker": str(error), "producer": str(producer / mode_impl)})
            return {"implementation": implementation, "version": version, "outputs": [str(path) if isinstance(path, Path) else path for path in outputs]}
        finally:
            slots.put(target_dir)

    completed = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(execute, target): target for target in targets}
        for future, target in futures.items():
            result = future.result()
            completed.append(result)
            (work / "run-status.json").write_text(json.dumps(completed, indent=2) + "\n")
            print(f"{target['implementation']} {target['version']}: {'BLOCKED' if 'blocker' in result else 'recorded'}", flush=True)
    validate_frozen_jobs(work, frozen)
    result = {"targets": completed, "jobs": frozen, "adapter_manifest_sha256": adapter_manifest_sha256}
    adapters = validate_frozen_adapters(work, result)
    for bundle in adapters["bundles"].values():
        for entry in [*bundle["files"], *bundle.get("support_files", [])]:
            if hashlib.sha256((ROOT / entry["path"]).read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError("executed Dynamo adapter changed during capture")
    return result


def validate_frozen_jobs(work: Path, frozen: dict) -> None:
    manifest = json.loads((work / "frozen-jobs.json").read_text())
    actual = {str(path.relative_to(work)) for path in (work / "jobs").rglob("*.json")}
    if not frozen or manifest != frozen or actual != set(frozen):
        raise ValueError("coverage jobs differ from frozen manifest")
    families = {mode: set() for mode in ("stream", "batch")}
    for relative, expected in frozen.items():
        path = Path(relative)
        if len(path.parts) != 3 or path.parts[0] != "jobs" or path.parts[1] not in families or path.suffix != ".json":
            raise ValueError("invalid frozen job path")
        families[path.parts[1]].add(path.stem)
        if hashlib.sha256((work / path).read_bytes()).hexdigest() != expected:
            raise ValueError("frozen job changed during execution")
    if not families["stream"] or families["stream"] != families["batch"]:
        raise ValueError("frozen stream and batch family sets differ")


def validate_source_identity(work: Path, receipt: dict, frozen: dict, identity: str, first_family: str) -> None:
    if identity == "dynamo_v2":
        source_identity = {"target": frozen, "lock_sha256": receipt["cargo_lock_sha256"]}
    else:
        source_identity = {"crate_checksum": frozen["checksum"], "cargo_lock_sha256": receipt["cargo_lock_sha256"]}
        # Retained receipts predate API adaptation. Their original source hash
        # remains valid; profiled runs also bind the actual compiler flags.
        legacy = hashlib.sha256(json.dumps(source_identity, sort_keys=True).encode()).hexdigest()
        if receipt["source_sha256"] == legacy and "compatibility_flags" not in receipt:
            return
        producer = work / "producers" / f"dynamo_v1-{frozen['version']}" / identity / first_family
        lock = producer / "Cargo.lock"
        if hashlib.sha256(lock.read_bytes()).hexdigest() != receipt["cargo_lock_sha256"] or capture_dynamo_unified_batch._crate_checksum(lock, frozen["version"]) != frozen["checksum"]:
            raise ValueError("coverage producer lock differs from execution")
        # _cargo retains stdout followed by stderr, which may contain warnings.
        metadata, _end = json.JSONDecoder().raw_decode((producer / "source-metadata.log").read_text())
        packages = metadata["packages"]
        sources = [Path(package["manifest_path"]).parent for package in packages
                   if package["name"] == "dynamo-parsers" and package["version"] == frozen["version"]]
        if len(sources) != 1:
            raise ValueError("coverage source package differs from execution")
        has_protocol = identity == "dynamo_v1_jail" and any(package["name"] == "dynamo-protocols" for package in tomllib.loads(lock.read_text())["package"])
        flags = capture_dynamo_unified_batch.compatibility_flags(sources[0], has_protocol)
        if "compatibility_flags" in receipt and receipt["compatibility_flags"] != flags:
            raise ValueError("coverage compatibility flags differ from source")
        source_identity["compatibility_flags"] = flags
    if receipt["source_sha256"] != hashlib.sha256(json.dumps(source_identity, sort_keys=True).encode()).hexdigest():
        raise ValueError("coverage source and lock identity mismatch")


def coverage_inventory(work: Path, result: dict) -> dict:
    """Account for every frozen request without treating absent output as success."""
    validate_frozen_jobs(work, result["jobs"])
    jobs = {mode: {path.stem: json.loads(path.read_text()) for path in (work / "jobs" / mode).glob("*.json")}
            for mode in ("stream", "batch")}
    rows = []
    counts = {}
    validated_sources = set()
    adapters = None
    frozen_targets = yaml.safe_load((work / "inventory.yaml").read_text())["targets"]
    frozen_identities = {(target["implementation"], target["version"]) for target in frozen_targets}
    observed = {(target["implementation"], target["version"]): target for target in result["targets"]}
    if len(observed) != len(result["targets"]) or set(observed) - frozen_identities:
        raise ValueError("coverage targets differ from frozen inventory")
    for frozen in frozen_targets:
        target = observed.get((frozen["implementation"], frozen["version"]), frozen)
        implementation = target["implementation"]
        identities = (implementation,) if implementation == "dynamo_v2" else ("dynamo_v1", "dynamo_v1_jail")
        outputs = target.get("outputs", [])
        for identity in identities:
            mode = "stream" if identity == "dynamo_v2" else "batch"
            for family, job in jobs[mode].items():
                path = work / "captures" / f"{identity}-{target['version']}" / (family + ".json")
                blocker = target.get("blocker") or next((output["blocker"] for output in outputs
                    if isinstance(output, dict) and output["implementation"] == identity and output["family"] == family), None)
                document = json.loads(path.read_text()) if path.is_file() and not blocker else None
                if document:
                    if adapters is None:
                        adapters = validate_frozen_adapters(work, result)
                    adapter = validate_adapter_binding(adapters, identity, str(path.relative_to(work)))
                    receipt = document["provenance"]
                    validate_execution(document, job, (work / "jobs" / mode / (family + ".json")).read_bytes(), identity, target["version"], expected_adapter_sha256=adapter)
                    if receipt.get("crate_checksum") != frozen.get("checksum"):
                        raise ValueError("coverage producer artifact differs from frozen inventory")
                    if implementation == "dynamo_v2" and receipt.get("git_commit") != frozen.get("git_commit"):
                        raise ValueError("coverage producer commit differs from frozen inventory")
                    source_key = (identity, target["version"], receipt["source_sha256"], receipt["cargo_lock_sha256"],
                                  json.dumps(receipt.get("compatibility_flags")))
                    if source_key not in validated_sources:
                        validate_source_identity(work, receipt, frozen, identity, min(jobs[mode]))
                        validated_sources.add(source_key)
                for case in job["cases"]:
                    record = document["results"][case["id"]] if document else None
                    detail = blocker
                    if record is None:
                        status = "source_build_blocker" if blocker else "missing_evidence"
                    else:
                        status, detail = execution_outcome(record, mode)
                    counts[status] = counts.get(status, 0) + 1
                    rows.append({"implementation": identity, "version": target["version"], "family": family,
                                 "case_id": case["case_id"], "scenario": case["scenario"], "status": status,
                                 **({"detail": detail} if detail else {})})
    return {"counts": counts, "combinations": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=ROOT / "conformance/dynamo-capture-inventory.yaml")
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--versions", nargs="+")
    parser.add_argument("--families", nargs="+", help="recapture only selected registered corpus families")
    parser.add_argument("--target-root", type=Path, help="exclusively owned, inactive Cargo build slots from an earlier run")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    result = run_inventory(args.inventory, args.work, args.workers, args.versions, args.target_root, args.families)
    (args.work / "run.json").write_text(json.dumps(result, indent=2) + "\n")
    coverage = coverage_inventory(args.work, result)
    (args.work / "coverage.json").write_text(json.dumps(coverage, ensure_ascii=False) + "\n")
    print(json.dumps(coverage["counts"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
