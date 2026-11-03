#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execute frozen peer releases through the existing family-scoped Unified adapters."""

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile

import yaml
import yaml_fast  # noqa: F401

from capture_unified_job import build_job
from capture_stimulus import capture_input, unavailable_result
from capture_stimulus import execution_outcome, request_for_case, validate_execution
from peer_modules import published_module_paths

ROOT = Path(__file__).resolve().parents[3]
IMAGE_NAMES = {"vllm": "vllm/vllm-openai:v0.31.0", "sglang": "lmsysorg/sglang:v0.5.14"}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False).encode()


def compatible_artifact(target):
    artifacts = target["artifacts"]
    source = next((item for item in artifacts if item["filename"].endswith(".tar.gz")), None)
    wheel = next((item for item in artifacts if item["filename"].endswith("-py3-none-any.whl")), None)
    native = next((item for item in artifacts if "-cp312-cp312-manylinux" in item["filename"]
                   and item["filename"].endswith("_x86_64.whl")), None)
    abi = next((item for item in artifacts if (match := re.search(r"-cp(3\d+)-abi3-manylinux[^/]*_x86_64\.whl$", item["filename"]))
                and int(match[1]) <= 312), None)
    return source or wheel or native or abi


def select_artifact(target):
    compatible = compatible_artifact(target)
    if compatible:
        return compatible
    foreign = [(int(match[1]), item) for item in target["artifacts"]
               if (match := re.search(r"-cp(3\d+)-cp3\d+-manylinux[^/]*_x86_64\.whl$", item["filename"]))]
    if foreign:
        return max(foreign, key=lambda entry: entry[0])[1]
    raise ValueError("No source distribution or Python 3.12 x86_64 wheel in the frozen release")


def target_artifact_record(target):
    inspection_only = compatible_artifact(target) is None
    return {**target, "artifact": select_artifact(target), **({"inspection_only": True,
            "artifact_selection": "foreign_abi_inspection"} if inspection_only else {})}


def validate_artifact_selection(target, missing):
    artifact = target["artifact"]
    inspection_only = compatible_artifact({"artifacts": [artifact]}) is None
    if bool(target.get("inspection_only")) != inspection_only or (
            inspection_only and target.get("artifact_selection") != "foreign_abi_inspection"):
        raise ValueError("artifact ABI selection differs from explicit inspection policy")
    if inspection_only and not missing:
        raise ValueError("foreign ABI artifact cannot execute a recognized parser pipeline")
    return inspection_only


def package_payload(archive, engine):
    if archive.suffix == ".whl":
        with zipfile.ZipFile(archive) as stream:
            members = [(name, stream.read(name)) for name in stream.namelist() if not name.endswith("/")]
    else:
        with tarfile.open(archive) as stream:
            members = [(member.name.split("/", 1)[-1], stream.extractfile(member).read())
                       for member in stream if member.isfile()]
    metadata = None
    payload = {}
    for name, data in members:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe published package member")
        if name == "PKG-INFO" or (".dist-info/" in name and name.endswith("/METADATA")):
            metadata = data
        if not (name.startswith(engine + "/") or (".dist-info/" in name and name.endswith("/METADATA"))):
            continue
        if relative.suffix in {".md", ".txt"}:
            continue
        if name in payload:
            raise ValueError("Duplicate published package member")
        payload[name] = data
    if engine + "/__init__.py" not in payload or metadata is None:
        raise ValueError("Published artifact lacks its Python package or version metadata")
    if not any(".dist-info/" in name and name.endswith("/METADATA") for name in payload):
        payload[engine + "-frozen.dist-info/METADATA"] = metadata
    return payload


def extract_package(archive, destination, engine):
    """Extract the exact published Python package without invoking its GPU build."""
    payload = package_payload(archive, engine)
    destination.mkdir(parents=True, exist_ok=False)
    for relative, data in payload.items():
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def validate_extracted_artifact(directory, target):
    artifact = target["artifact"]
    archive = directory / artifact["filename"]
    if sha256(archive.read_bytes()) != artifact["sha256"]:
        raise ValueError("Execution artifact differs from frozen hash")
    expected = {name: sha256(data) for name, data in package_payload(archive, target["implementation"]).items()}
    package = directory / "package"
    actual = {}
    for path in package.rglob("*"):
        if path.is_symlink():
            raise ValueError("Extracted package contains a symbolic link")
        if path.is_file():
            actual[str(path.relative_to(package))] = sha256(path.read_bytes())
    if actual != expected:
        raise ValueError("Extracted package differs from frozen artifact")
    return sha256(json_bytes(expected))


def download_artifact(artifact, destination):
    request = urllib.request.Request(artifact["url"], headers={"User-Agent": "frontend-crates-conformance/1"})
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)
    if sha256(destination.read_bytes()) != artifact["sha256"]:
        raise ValueError("Downloaded artifact differs from frozen hash")


def worker(target_path, jobs, output, mode):
    target = json.loads(target_path.read_text())
    engine, expected_version = target["implementation"], target["version"]
    package_digest = validate_extracted_artifact(target_path.parent, target)
    expected_root = target_path.parent / "package" / engine
    roles, missing = mode_api_inventory(expected_root, engine, mode)
    inspection_only = validate_artifact_selection(target, missing)
    if importlib.metadata.version(engine) != expected_version:
        raise ValueError("Imported package metadata differs from frozen version")
    adapter = importlib.import_module("capture_unified_batch")
    pipeline = None
    stream_module = None
    if missing:
        before = adapter.source_identity_root(engine, expected_root, expected_version)
    else:
        package = importlib.import_module(engine)
        if not Path(package.__file__).resolve().is_relative_to(expected_root.resolve()):
            raise ValueError("Imported package is outside the isolated frozen artifact")
        if mode == "batch":
            pipeline = adapter.vllm_pipeline() if engine == "vllm" else adapter.sglang_pipeline()
        else:
            stream_module = importlib.import_module("capture_" + engine + "_unified")
        before = adapter.source_identity(engine)
    if before["version"] != expected_version:
        raise ValueError("Imported producer version differs from frozen version")
    output.mkdir(parents=True, exist_ok=True)
    for job_path in sorted(jobs.glob("*.json")):
        job = json.loads(job_path.read_text())
        if missing:
            detail = f"Published {engine} {expected_version} lacks {', '.join(missing)} parser APIs for the {mode} pipeline"
            capture = {"version" if mode == "batch" else engine + "_version": expected_version,
                       "results": {case["id"]: unavailable_result("unsupported_capture_mode", detail,
                           observation=True, capture=request_for_case(case)) for case in job["cases"]}}
        elif mode == "batch":
            capture = {"version": expected_version, "results": adapter.capture_cases(
                job, *pipeline, supports_reasoning_init=engine == "sglang" and adapter.sglang_supports_reasoning_init())}
        elif job["family"] not in stream_module.FAMILY_PARSERS:
            detail = f"The shared {engine} stream adapter has no registered pipeline for {job['family']}"
            capture = {engine + "_version": expected_version, "results": {
                case["id"]: unavailable_result("peer_detector_unavailable", detail, observation=True,
                    capture=capture_input({**case, "chunks": [{"delta_text": text} for text in case["chunks"]]}))
                for case in job["cases"]}}
        else:
            original_in, original_out = sys.stdin, sys.stdout
            try:
                sys.stdin, sys.stdout = io.StringIO(json.dumps(job)), io.StringIO()
                stream_module.main()
                capture = yaml.safe_load(sys.stdout.getvalue())
            finally:
                sys.stdin, sys.stdout = original_in, original_out
            if capture.get(engine + "_version") != expected_version:
                raise ValueError("Stream producer version differs from frozen version")
        if set(capture["results"]) != {case["id"] for case in job["cases"]}:
            raise ValueError("Capture omitted or added frozen cases")
        receipt = {"implementation": engine + "_python" + ("_batch" if mode == "batch" else ""),
                   "version": expected_version, "artifact_sha256": target["artifact"]["sha256"],
                   "extracted_package_sha256": package_digest,
                   "status": "capability_inspection" if missing else "executed",
                   "parser_apis": roles,
                   "source_sha256": before["source_sha256"], "adapter_sha256": before["adapter_sha256"],
                   "job_sha256": sha256(job_path.read_bytes()), "results_sha256": sha256(json_bytes(capture["results"]))}
        if inspection_only:
            receipt.update(inspection_only=True, artifact_selection="foreign_abi_inspection")
        capture["provenance"] = receipt
        validate_execution(capture, job, job_path.read_bytes(), receipt["implementation"], expected_version)
        for record in capture["results"].values():
            execution_outcome(record, mode)
        (output / job_path.name).write_text(json.dumps(capture, ensure_ascii=False))
    after = adapter.source_identity_root(engine, expected_root, expected_version) if missing else adapter.source_identity(engine)
    if before != after:
        raise ValueError("Producer source changed during execution")
    if package_digest != validate_extracted_artifact(target_path.parent, target):
        raise ValueError("Extracted package changed during capture")
    (output / "source.json").write_text(json.dumps(before, sort_keys=True))


def mode_api_inventory(package_root, engine, mode):
    roles = published_module_paths(package_root, engine)
    required = ("protocol",) if engine == "vllm" and mode == "stream" and roles["combined"] else ("protocol", "reasoning", "tools")
    return roles, [role for role in required if roles[role] is None]


def worker_modes(target_path, jobs, output, modes=None):
    target = json.loads(target_path.read_text())
    statuses = {}
    for mode in modes or ("stream", "batch"):
        directory = output / f"{target['implementation']}_python{'_batch' if mode == 'batch' else ''}-{target['version']}"
        job_directory = jobs / ("batch" if mode == "batch" else "unified")
        log = target_path.parent / (mode + ".log")
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--target", str(target_path),
                   "--jobs", str(job_directory), "--output", str(directory), "--mode", mode]
        with log.open("wb") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        statuses[mode] = {"exit_code": result.returncode, "log": str(log), "output": str(directory)}
        if result.returncode:
            statuses[mode]["blocker"] = "Isolated published-package adapter failed; see retained log"
        (target_path.parent / "worker-status.json").write_text(json.dumps(statuses))


def run_target(target, work, bundle, images, artifact_cache=None, modes=None):
    engine, version = target["implementation"], target["version"]
    directory = work / "producers" / f"{engine}-{version}"
    directory.mkdir(parents=True, exist_ok=False)
    modes = tuple(modes or ("stream", "batch"))
    result = {"implementation": engine, "version": version, "modes": {}, "requested_modes": list(modes)}
    try:
        target_record = target_artifact_record(target)
        artifact = target_record["artifact"]
        archive = directory / artifact["filename"]
        cached = artifact_cache / "producers" / directory.name / artifact["filename"] if artifact_cache else None
        if cached is not None and cached.is_file():
            shutil.copyfile(cached, archive)
            if sha256(archive.read_bytes()) != artifact["sha256"]:
                raise ValueError("Cached artifact differs from frozen hash")
        else:
            download_artifact(artifact, archive)
        extract_package(archive, directory / "package", engine)
        target_path = directory / "target.json"
        target_path.write_text(json.dumps(target_record))
        result["artifact_sha256"] = artifact["sha256"]
    except (OSError, ValueError, tarfile.TarError, zipfile.BadZipFile) as error:
        result["blocker"] = f"{type(error).__name__}: {error}"
        (directory / "status.json").write_text(json.dumps(result))
        return result
    runtime = directory / "runtime"
    (runtime / "tmp").mkdir(parents=True)
    name = f"unified-peer-{sha256(str(work.resolve()).encode())[:8]}-{engine}-{version}".replace("+", "-")
    command = ["docker", "run", "--rm", "--name", name, "--user", f"{os.getuid()}:{os.getgid()}",
               "--network", "none", "--entrypoint", "python3", "-v", f"{work}:{work}",
               "-e", f"PYTHONPATH={bundle}:{directory / 'package'}", "-e", "PYTHONDONTWRITEBYTECODE=1",
               "-e", "USER=capture", "-e", "LOGNAME=capture", "-e", f"HOME={runtime}",
               "-e", f"TMPDIR={runtime / 'tmp'}", "-e", f"TORCHINDUCTOR_CACHE_DIR={runtime / 'inductor'}",
               "-e", "VLLM_TARGET_DEVICE=cpu", "-e", "VLLM_LOGGING_LEVEL=ERROR", images[engine],
               str(bundle / Path(__file__).name), "--worker", "--target", str(target_path),
               "--jobs", str(work / "jobs"), "--output", str(work / "captures"), "--mode", "all", "--only-modes", *modes]
    log = directory / "container.log"
    with log.open("wb") as stream:
        try:
            completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=1200)
            result["container_exit_code"] = completed.returncode
            if completed.returncode:
                result["blocker"] = f"Isolated container failed (exit {completed.returncode}); see {log}"
        except subprocess.TimeoutExpired:
            stopped = subprocess.run(["docker", "stop", "--time", "5", name], capture_output=True)
            result.update(blocker="Isolated adapters exceeded 1200 seconds", stop_exit_code=stopped.returncode)
    status_path = directory / "worker-status.json"
    if status_path.is_file():
        result["modes"] = json.loads(status_path.read_text())
    for mode in modes:
        if mode not in result["modes"]:
            result["modes"][mode] = {"blocker": result.get("blocker", "Worker did not report completion"), "log": str(log)}
    (directory / "status.json").write_text(json.dumps(result))
    return result


def resolve_images(overrides=None):
    names = {**IMAGE_NAMES, **(overrides or {})}
    if set(names) != set(IMAGE_NAMES) or any(not isinstance(image, str) or not image.strip() for image in names.values()):
        raise ValueError("image overrides must name nonempty peer images")
    return {engine: subprocess.check_output(["docker", "image", "inspect", image, "--format", "{{.Id}}"], text=True).strip()
            for engine, image in names.items()}


def run_inventory(inventory, work, workers, versions=None, artifact_cache=None, image_overrides=None, modes=None):
    modes = tuple(("stream", "batch") if modes is None else modes)
    if not modes or len(set(modes)) != len(modes) or not set(modes).issubset({"stream", "batch"}):
        raise ValueError("capture modes must be distinct stream/batch selections")
    targets = [target for target in yaml.safe_load(inventory.read_text())["targets"]
               if target["implementation"] in IMAGE_NAMES and (not versions or target["version"] in versions)]
    if not targets or workers < 1:
        raise ValueError("Nonempty peer inventory and positive worker count required")
    work.mkdir(parents=True, exist_ok=False)
    bundle = work / "adapter"
    shutil.copytree(Path(__file__).parent, bundle, ignore=shutil.ignore_patterns("__pycache__", "*.md", "*.txt"))
    adapter_hashes = {str(path.relative_to(bundle)): sha256(path.read_bytes()) for path in bundle.rglob("*") if path.is_file()}
    (work / "adapter-manifest.json").write_text(json.dumps(adapter_hashes, sort_keys=True))
    (work / "inventory.yaml").write_bytes(inventory.read_bytes())
    for mode in ("batch", "unified"):
        jobs = work / "jobs" / mode
        jobs.mkdir(parents=True)
        for path in sorted((ROOT / "conformance/fixtures-unified-v2/families").glob("*/inputs_and_golden.yaml")):
            (jobs / (path.parent.name + ".json")).write_text(json.dumps(build_job(path.parent.parent, path.parent.name, mode=mode), ensure_ascii=False) + "\n")
    frozen = {str(path.relative_to(work)): sha256(path.read_bytes()) for path in (work / "jobs").rglob("*.json")}
    (work / "frozen-jobs.json").write_text(json.dumps(frozen, sort_keys=True))
    images = resolve_images(image_overrides)
    (work / "images.json").write_text(json.dumps(images))
    completed = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_target, target, work, bundle, images, artifact_cache, modes) for target in targets]
        for future in as_completed(futures):
            result = future.result()
            completed.append(result)
            (work / "run-status.json").write_text(json.dumps(completed, indent=2))
            print(f"{result['implementation']} {result['version']}: {json.dumps(result['modes'])}", flush=True)
    if any(sha256((work / path).read_bytes()) != digest for path, digest in frozen.items()):
        raise ValueError("Frozen job changed during execution")
    if adapter_hashes != {str(path.relative_to(bundle)): sha256(path.read_bytes()) for path in bundle.rglob("*") if path.is_file()}:
        raise ValueError("Frozen adapter changed during execution")
    return completed


def retained_capture_output(work, mode):
    original_root = work
    mapping = work / "retained-execution.json"
    if mapping.exists():
        metadata = json.loads(mapping.read_bytes())
        original_root = Path(metadata["original_root"])
        if metadata.get("schema_version") != 1 or not original_root.is_absolute() or ".." in original_root.parts:
            raise ValueError("invalid retained execution root mapping")
        for name, key in (("inventory.yaml", "inventory_sha256"), ("frozen-jobs.json", "frozen_jobs_sha256"),
                          ("adapter-manifest.json", "adapter_manifest_sha256")):
            if metadata.get(key) != sha256((work / name).read_bytes()):
                raise ValueError("retained execution mapping differs from frozen evidence")
    output = Path(mode["output"])
    if output != original_root / "captures" / output.name:
        raise ValueError("capture output is outside the isolated execution")
    return work / "captures" / output.name


def required_adapter_files(bundle, engine, mode, roles, missing):
    """Derive eager shared imports from frozen code, then add actual dynamic entrypoints."""
    pending = ["capture_peer_unified_versions", "capture_unified_batch"]
    if mode == "stream" and not missing:
        pending.append("capture_" + engine + "_unified")
        if engine == "vllm" and not roles["combined"]:
            pending.append("peer_modules")
    required = {"parser_families.yaml"}
    while pending:
        module = pending.pop()
        base = bundle.joinpath(*module.split("."))
        path = base.with_suffix(".py") if base.with_suffix(".py").is_file() else base / "__init__.py"
        if not path.is_file():
            raise ValueError("executed adapter entrypoint is absent from frozen bundle")
        relative = str(path.relative_to(bundle))
        if relative in required:
            continue
        required.add(relative)
        for index in range(1, len(module.split("."))):
            parent = ".".join(module.split(".")[:index])
            if (bundle.joinpath(*parent.split(".")) / "__init__.py").is_file():
                pending.append(parent)
        package = module.split(".") if path.name == "__init__.py" else module.split(".")[:-1]
        nodes = list(ast.iter_child_nodes(ast.parse(path.read_bytes())))
        while nodes:
            node = nodes.pop()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                prefix = package[:len(package) - node.level + 1] if node.level else []
                base_name = ".".join([*prefix, *([node.module] if node.module else [])])
                names = [base_name, *(base_name + "." + alias.name for alias in node.names)]
            for name in names:
                local = bundle.joinpath(*name.split("."))
                if name and (local.with_suffix(".py").is_file() or (local / "__init__.py").is_file()):
                    pending.append(name)
            nodes.extend(ast.iter_child_nodes(node))
    return required


def validate_source_evidence(work, directory, mode, target, capture_mode):
    target_record = json.loads((directory / "target.json").read_text())
    extracted_digest = validate_extracted_artifact(directory, target_record)
    output = retained_capture_output(work, mode)
    source = json.loads((output / "source.json").read_text())
    if source["version"] != target["version"]:
        raise ValueError("source manifest version differs from frozen target")
    frozen_adapter = json.loads((work / "adapter-manifest.json").read_text())
    roles, missing = mode_api_inventory(directory / "package" / target["implementation"], target["implementation"], capture_mode)
    validate_artifact_selection(target_record, missing)
    required = required_adapter_files(work / "adapter", target["implementation"], capture_mode, roles, missing)
    if not required.issubset(source["adapter_files"]):
        raise ValueError("source manifest omits executed adapter closure")
    for relative, digest in source["adapter_files"].items():
        if frozen_adapter.get(relative) != digest or sha256((work / "adapter" / relative).read_bytes()) != digest:
            raise ValueError("executed adapter differs from frozen bundle")
    package = directory / "package" / target["implementation"]
    actual_package = {str(path.relative_to(package)): sha256(path.read_bytes()) for path in package.rglob("*.py")}
    if not actual_package or source["package_files"] != actual_package:
        raise ValueError("executed package differs from complete frozen source manifest")
    for key, collection in (("source_sha256", "package_files"), ("adapter_sha256", "adapter_files")):
        if source[key] != sha256(json.dumps(source[collection], sort_keys=True).encode()):
            raise ValueError("source manifest hash is corrupt")
    return {**source, "extracted_package_sha256": extracted_digest}


def coverage_jobs(work):
    frozen = json.loads((work / "frozen-jobs.json").read_text())
    actual = {str(path.relative_to(work)) for path in (work / "jobs").rglob("*.json")}
    if not frozen or set(frozen) != actual or any(sha256((work / path).read_bytes()) != digest for path, digest in frozen.items()):
        raise ValueError("coverage jobs differ from frozen manifest")
    jobs = {mode: {path.stem: (path, json.loads(path.read_text())) for path in (work / "jobs" / mode).glob("*.json")}
            for mode in ("batch", "unified")}
    if not jobs["batch"] or set(jobs["batch"]) != set(jobs["unified"]):
        raise ValueError("coverage family modes differ")
    return jobs


def peer_targets(inventory):
    targets = [target for target in inventory["targets"] if target["implementation"] in IMAGE_NAMES]
    identities = [(target["implementation"], target["version"]) for target in targets]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate frozen release target")
    return targets


def coverage_rows(work):
    jobs = coverage_jobs(work)
    targets = peer_targets(yaml.safe_load((work / "inventory.yaml").read_text()))
    yield from coverage_target_rows(work, jobs, targets)


def coverage_target_rows(work, jobs, targets, documents=None):
    for target in targets:
        engine, version = target["implementation"], target["version"]
        directory = work / "producers" / f"{engine}-{version}"
        status_path = directory / "status.json"
        status = json.loads(status_path.read_text()) if status_path.is_file() else {}
        if status and (status["implementation"], status["version"]) != (engine, version):
            raise ValueError("coverage target identity differs from frozen inventory")
        for mode in ("stream", "batch"):
            implementation = engine + "_python" + ("_batch" if mode == "batch" else "")
            outcome = status.get("modes", {}).get(mode, {})
            blocker = status.get("blocker") or outcome.get("blocker")
            source = None
            if outcome.get("exit_code") == 0:
                if blocker:
                    raise ValueError("successful mode has conflicting blocker")
                target_evidence = json.loads((directory / "target.json").read_text())
                selected = select_artifact(target)
                if target_evidence != target_artifact_record(target) or sha256((directory / selected["filename"]).read_bytes()) != selected["sha256"]:
                    raise ValueError("execution artifact differs from frozen release")
                source = validate_source_evidence(work, directory, outcome, target, mode)
            for family, (job_path, job) in jobs["batch" if mode == "batch" else "unified"].items():
                document = None
                if source is not None:
                    capture_path = retained_capture_output(work, outcome) / (family + ".json")
                    capture_bytes, job_bytes = capture_path.read_bytes(), job_path.read_bytes()
                    document = json.loads(capture_bytes)
                    validate_execution(document, job, job_bytes, implementation, version)
                    receipt = document["provenance"]
                    if receipt["artifact_sha256"] != selected["sha256"] or any(receipt.get(key) != source[key] for key in ("source_sha256", "adapter_sha256", "extracted_package_sha256")):
                        raise ValueError("execution receipt differs from artifact or source manifest")
                    roles, missing = mode_api_inventory(directory / "package" / engine, engine, mode)
                    inspection_only = validate_artifact_selection(target_evidence, missing)
                    if bool(receipt.get("inspection_only")) != inspection_only or (
                            inspection_only and receipt.get("artifact_selection") != "foreign_abi_inspection"):
                        raise ValueError("execution receipt differs from artifact ABI inspection policy")
                    if receipt.get("parser_apis") != roles or receipt.get("status") != ("capability_inspection" if missing else "executed"):
                        raise ValueError("execution receipt differs from published API capabilities")
                    if missing and any("unavailable" not in record or "unsupported_capture_mode" != record.get("capture_observation", {}).get("unavailable", {}).get("code") for record in document["results"].values()):
                        raise ValueError("capability inspection cannot claim parser execution")
                    if documents is not None:
                        documents({"implementation": implementation, "version": version, "family": family,
                                   "job": str(job_path), "capture": str(capture_path), "provenance": receipt,
                                   "job_sha256": sha256(job_bytes), "capture_sha256": sha256(capture_bytes),
                                   "evidence_root": str(work)})
                for case in job["cases"]:
                    if document is not None:
                        state, detail = execution_outcome(document["results"][case["id"]], mode)
                    else:
                        state, detail = ("source_build_blocker", blocker) if blocker else ("missing_evidence", None)
                    yield {"implementation": implementation, "version": version, "family": family,
                           "case_id": case.get("case_id", case["id"]), "scenario": case["scenario"], "status": state,
                           **({"detail": detail} if detail else {})}


def validate_frozen_adapter(work):
    frozen = json.loads((work / "adapter-manifest.json").read_text())
    bundle = work / "adapter"
    paths = list(bundle.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise ValueError("frozen adapter contains a symbolic link")
    actual = {str(path.relative_to(bundle)): sha256(path.read_bytes()) for path in paths if path.is_file()}
    if not frozen or actual != frozen:
        raise ValueError("frozen adapter differs from its manifest")


def evidence_root_binding(root):
    work = Path(root).resolve()
    return {"root": str(work), **{name.replace("-", "_").replace(".json", "").replace(".yaml", "") + "_sha256":
            sha256((work / name).read_bytes())
            for name in ("inventory.yaml", "frozen-jobs.json", "adapter-manifest.json")}}


def validate_supersessions(manifest, evidence, master):
    if manifest is None:
        return {}
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise ValueError("unsupported evidence supersession schema")
    known = {str(work): frozen for work, _jobs, frozen in evidence}
    replacements = {}
    for entry in manifest["replacements"]:
        identity = entry["implementation"], entry["version"]
        engine = entry["implementation"].removesuffix("_batch").removesuffix("_python")
        target = engine, entry["version"]
        if entry["implementation"] not in (engine + "_python", engine + "_python_batch") or target not in master:
            raise ValueError("supersession mode differs from frozen inventory")
        if identity in replacements or not entry["reason"].strip() or not entry["invalidated"]:
            raise ValueError("duplicate or unexplained evidence supersession")
        bindings = [entry["replacement"], *entry["invalidated"]]
        paths = [binding["root"] for binding in bindings]
        if len(set(paths)) != len(paths):
            raise ValueError("duplicate supersession root")
        for binding in bindings:
            if binding["root"] not in known or target not in known[binding["root"]] or binding != evidence_root_binding(binding["root"]):
                raise ValueError("supersession root differs from bound execution evidence")
        applicable = {root for root, frozen in known.items() if target in frozen}
        if set(paths) != applicable:
            raise ValueError("supersession must account for every prior mode evidence root")
        mode = "batch" if entry["implementation"].endswith("_batch") else "stream"
        status = json.loads((Path(paths[0]) / "producers" / f"{engine}-{entry['version']}" / "status.json").read_bytes())
        if status.get("modes", {}).get(mode, {}).get("exit_code") != 0:
            raise ValueError("supersession replacement mode has no successful capture")
        replacements[identity] = entry["replacement"]["root"]
    return replacements


def master_coverage_rows(inventory, roots, documents=None, supersessions=None):
    """Combine ordered, independently validated execution roots without counting retries twice."""
    inventory = yaml.safe_load(Path(inventory).read_bytes()) if isinstance(inventory, (str, Path)) else inventory
    targets = peer_targets(inventory)
    master = {(target["implementation"], target["version"]): target for target in targets}
    evidence = []
    canonical_jobs = None
    seen = set()
    for root in roots:
        work = Path(root).resolve()
        if work in seen:
            raise ValueError("duplicate coverage evidence root")
        seen.add(work)
        frozen_targets = peer_targets(yaml.safe_load((work / "inventory.yaml").read_bytes()))
        frozen = {(target["implementation"], target["version"]): target for target in frozen_targets}
        if any(master.get(key) != target for key, target in frozen.items()):
            raise ValueError("evidence target differs from master frozen inventory")
        jobs = coverage_jobs(work)
        job_hashes = json.loads((work / "frozen-jobs.json").read_bytes())
        if canonical_jobs is not None and canonical_jobs != job_hashes:
            raise ValueError("evidence roots contain different frozen requests")
        canonical_jobs = job_hashes
        validate_frozen_adapter(work)
        expected = {f"{engine}-{version}" for engine, version in frozen}
        if any(path.name not in expected for path in (work / "producers").iterdir()):
            raise ValueError("evidence producer is absent from its frozen inventory")
        evidence.append((work, jobs, frozen))
    if not evidence:
        raise ValueError("at least one coverage evidence root is required")
    replacements = validate_supersessions(supersessions, evidence, master)
    strength = {"missing_evidence": 0, "source_build_blocker": 1, "unsupported_capability": 2,
                "executed": 3, "parser_error": 3}
    for target in targets:
        identity = target["implementation"], target["version"]
        selected, blockers = {}, {}
        for work, jobs, frozen in evidence:
            if identity not in frozen:
                continue
            pending_documents = []
            for row in coverage_target_rows(work, jobs, [target], pending_documents.append if documents else None):
                replacement = replacements.get((row["implementation"], row["version"]))
                if replacement is not None and str(work) != replacement:
                    continue
                key = row["implementation"], row["family"], row["case_id"], row["scenario"]
                if row["status"] == "source_build_blocker":
                    blockers.setdefault(key, []).append({"evidence_root": str(work), **row})
                candidate = {**row, **({"evidence_root": str(work)} if row["status"] != "missing_evidence" else {})}
                if key not in selected or strength[row["status"]] >= strength[selected[key]["status"]]:
                    selected[key] = candidate
            for entry in pending_documents:
                replacement = replacements.get((entry["implementation"], entry["version"]))
                if replacement is None or str(work) == replacement:
                    documents(entry)
        if not selected:
            # Every master target still needs rows when all evidence inventories
            # are subsets; a missing release must not disappear from accounting.
            work, jobs, _frozen = evidence[0]
            for row in coverage_target_rows(work, jobs, [target]):
                key = row["implementation"], row["family"], row["case_id"], row["scenario"]
                selected[key] = row
        for key, row in selected.items():
            previous = [blocker for blocker in blockers.get(key, [])
                        if blocker["evidence_root"] != row.get("evidence_root")]
            yield {**row, **({"prior_blockers": previous} if previous else {})}


def write_coverage_report(rows, destination, summary_path, metadata=None):
    counts, row_count = {}, 0
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent,
                                         prefix="." + destination.name + ".", suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            for row in rows:
                counts[row["status"]] = counts.get(row["status"], 0) + 1
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                row_count += 1
        with temporary.open("rb") as output:
            digest = hashlib.file_digest(output, "sha256").hexdigest()
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    summary = {**(metadata or {}), "counts": counts, "combinations": row_count, "path": str(destination), "sha256": digest}
    summary_path.write_bytes(json.dumps(summary, sort_keys=True).encode())
    return summary


def write_coverage(work):
    return write_coverage_report(coverage_rows(work), work / "coverage.jsonl", work / "coverage-summary.json")


def write_master_coverage(inventory, roots, destination, imports_path=None, supersessions=None):
    roots = list(roots)
    inventory_bytes = Path(inventory).read_bytes() if isinstance(inventory, (str, Path)) else json_bytes(inventory)
    imports, successful_modes = [], {}

    def retain_document(entry):
        identity = entry["implementation"], entry["version"]
        owner = successful_modes.setdefault(identity, entry["evidence_root"])
        if owner != entry["evidence_root"]:
            raise ValueError("multiple successful execution roots cannot form one peer import")
        imports.append(entry)

    summary_path = destination.with_name(destination.stem + "-summary.json")
    summary = write_coverage_report(master_coverage_rows(inventory, roots, retain_document if imports_path else None, supersessions),
                                    destination, summary_path, {"inventory_sha256": sha256(inventory_bytes),
                                    "evidence_roots": [str(Path(root).resolve()) for root in roots],
                                    **({"supersessions": supersessions, "supersessions_sha256": sha256(json_bytes(supersessions))}
                                       if supersessions is not None else {})})
    if imports_path is not None:
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="wb", dir=imports_path.parent, prefix="." + imports_path.name,
                                             suffix=".tmp", delete=False) as output:
                temporary = Path(output.name)
                output.write(json_bytes(imports))
            os.replace(temporary, imports_path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        summary["import_list"] = {"path": str(imports_path), "sha256": sha256(imports_path.read_bytes()),
                                  "documents": len(imports), "successful_modes": len(successful_modes)}
        summary_path.write_bytes(json.dumps(summary, sort_keys=True).encode())
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=ROOT / "conformance/all-capture-inventory.yaml")
    parser.add_argument("--work", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--versions", nargs="+")
    parser.add_argument("--artifact-cache", type=Path, help="reuse only hash-verified published artifacts from an earlier isolated run")
    parser.add_argument("--vllm-image", help="installed dependency image for vLLM; the resolved immutable ID is retained")
    parser.add_argument("--sglang-image", help="installed dependency image for SGLang; the resolved immutable ID is retained")
    parser.add_argument("--coverage-only", action="store_true", help="validate retained evidence without running producers")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--target", type=Path)
    parser.add_argument("--jobs", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mode", choices=("stream", "batch", "all"))
    parser.add_argument("--only-modes", choices=("stream", "batch"), nargs="+", help="modes selected by the isolated worker supervisor")
    args = parser.parse_args()
    if args.worker:
        if args.mode == "all":
            worker_modes(args.target, args.jobs, args.output, args.only_modes)
        else:
            worker(args.target, args.jobs, args.output, args.mode)
    elif args.work:
        if args.coverage_only:
            print(json.dumps(write_coverage(args.work), sort_keys=True))
        else:
            overrides = {engine: image for engine, image in (("vllm", args.vllm_image), ("sglang", args.sglang_image)) if image}
            run_inventory(args.inventory, args.work, args.workers, args.versions, args.artifact_cache, overrides,
                          [args.mode] if args.mode in {"stream", "batch"} else None)
    else:
        parser.error("--work is required")


if __name__ == "__main__":
    main()
