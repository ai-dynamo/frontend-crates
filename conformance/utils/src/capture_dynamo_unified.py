#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the existing cross-version harness against a pinned crate or retained source."""

import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import tarfile
import tomllib
import urllib.request
from pathlib import Path

from capture_dynamo_unified_batch import _cargo
import capture_dynamo_unified_batch
from dynamo_version import SOURCE_PATHS, source_fingerprint

ROOT = Path(__file__).resolve().parents[3]
HARNESS = ROOT / "conformance/tests/capture_cross_version.rs"
COMMON = HARNESS.parent / "common"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def crate_source(target: dict, directory: Path) -> Path:
    request = urllib.request.Request(
        f"https://static.crates.io/crates/dynamo-parsers-v2/dynamo-parsers-v2-{target['version']}.crate",
        headers={"User-Agent": "frontend-crates-conformance"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        artifact = response.read()
    if digest(artifact) != target["checksum"]:
        raise ValueError("published crate checksum differs from frozen inventory")
    (directory / "producer.crate").write_bytes(artifact)
    with tarfile.open(fileobj=io.BytesIO(artifact), mode="r:gz") as archive:
        archive.extractall(directory / "source", filter="data")
    source = directory / "source" / f"dynamo-parsers-v2-{target['version']}"
    identity = json.loads((source / ".cargo_vcs_info.json").read_text())
    if identity["git"]["sha1"] != target["git_commit"]:
        raise ValueError("published crate commit differs from frozen inventory")
    return source


def retained_source(target: dict, directory: Path) -> Path:
    commit = target["git_commit"]
    if source_fingerprint(ROOT, commit) != target["source_sha256"]:
        raise ValueError("retained source fingerprint differs from frozen inventory")
    archive_bytes = subprocess.run(["git", "archive", commit], cwd=ROOT, check=True, capture_output=True).stdout
    source = directory / "source"
    source.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive_bytes)) as archive:
        archive.extractall(source, filter="data")
    # Git attributes may filter archive members; verify every fingerprinted blob.
    tree = subprocess.run(["git", "ls-tree", "-rz", commit, "--", *SOURCE_PATHS], cwd=ROOT, check=True, capture_output=True).stdout
    for entry in tree.split(b"\0"):
        if not entry:
            continue
        meta, name = entry.split(b"\t", 1)
        mode, kind, oid = meta.split()
        expected = subprocess.run(["git", "cat-file", "blob", oid.decode()], cwd=ROOT, check=True, capture_output=True).stdout
        path = source / os.fsdecode(name)
        actual = os.fsencode(os.readlink(path)) if mode == b"120000" else path.read_bytes()
        if kind != b"blob" or actual != expected:
            raise ValueError(f"archive changed retained source blob {os.fsdecode(name)}")
    return source / "parsers/v2"


def muse_guided_json_unsupported(source: Path) -> bool:
    path = source / "src/tool_calling/muse_glimmer.rs"
    if not path.is_file():
        return False
    body = path.read_text()
    return ("if init.tool_output_mode != UnifiedToolOutputMode::Native" in body
            and "muse_glimmer has no reasoning marker pair" in body)


def prepare(target: dict, work: Path, target_dir: Path) -> dict:
    work.mkdir(parents=True)
    source = crate_source(target, work) if target["source_kind"] == "crate" else retained_source(target, work)
    actual = tomllib.loads((source / "Cargo.toml").read_text())["package"]["version"]
    if actual != target["version"]:
        raise ValueError("producer manifest differs from frozen version")
    lib = (source / "src/lib.rs").read_text()
    unified = source / "src/unified/mod.rs"
    tool_traits = source / "src/tool_calling/traits.rs"
    flags = []
    if muse_guided_json_unsupported(source):
        flags.append("conformance_muse_guided_unsupported")
    if "create_unified_parser_for_family" not in lib:
        flags.append("conformance_split_only")
    if "create_unified_parser_for_family" in lib and "UnifiedParserEvent" not in lib:
        flags.append("conformance_legacy_delta")
    if not unified.is_file() or "fn initialize_request" not in unified.read_text():
        flags.append("conformance_legacy_init")
    if not tool_traits.is_file() or "pub complete:" not in tool_traits.read_text():
        flags.append("conformance_legacy_terminal")
    registry_names = []
    if "REGISTERED_FAMILIES" not in lib:
        flags.append("conformance_legacy_registry")
        factory_source = (source / "src/tool_calling/mod.rs").read_text()
        for arm in re.findall(r'^\s*((?:"[^"\n]+"\s*\|\s*)*"[^"\n]+")\s*=>', factory_source, re.MULTILINE):
            registry_names.extend(re.findall(r'"([^"\n]+)"', arm))
        if not registry_names:
            raise ValueError("cannot resolve old artifact's registered tool families")
    manifest = work / "Cargo.toml"
    dependency = f'"={target["version"]}"' if target["source_kind"] == "crate" else '{ path = ' + json.dumps(str(source)) + ' }'
    manifest.write_text(
        '[package]\nname = "unified-release-probe"\nversion = "0.1.0"\nedition = "2024"\n\n'
        '[dev-dependencies]\nanyhow = "1"\nserde = { version = "1", features = ["derive"] }\n'
        'serde_json = "1"\nserde_yaml = "0.9"\n'
        f'dynamo-parsers-v2 = {dependency}\ndynamo-parsers = "={target["reasoning_version"]}"\n'
    )
    (work / "tests").mkdir()
    shutil.copyfile(HARNESS, work / "tests/capture_cross_version.rs")
    shutil.copytree(COMMON, work / "tests/common")
    (work / "utils/src").mkdir(parents=True)
    shutil.copyfile(ROOT / "conformance/utils/src/unified_tools.json", work / "utils/src/unified_tools.json")
    env = {key: value for key, value in os.environ.items() if key in {"PATH", "HOME", "USER", "CARGO_HOME", "RUSTUP_HOME", "TMPDIR", "TMP", "TEMP", "SSL_CERT_FILE", "SSL_CERT_DIR"}}
    env["XVER_TOOL_FAMILIES"] = json.dumps(registry_names)
    env["CARGO_TARGET_DIR"] = str(target_dir)
    env["RUSTFLAGS"] = " ".join(f"--cfg {flag}" for flag in flags)
    _cargo("cargo", ["generate-lockfile", "--manifest-path", str(manifest)], work_dir=work, log=work / "lock.log", env=env)
    # Some published releases no longer compile with newer semver-compatible
    # dependencies. Freeze reviewed transitive versions in the target inventory;
    # both that inventory and the resulting lock are bound into the receipt.
    for package, version in target.get("dependency_pins", {}).items():
        _cargo("cargo", ["update", "--manifest-path", str(manifest), "--package", package,
                         "--precise", version], work_dir=work,
               log=work / f"pin-{package}.log", env=env)
    lock = tomllib.loads((work / "Cargo.lock").read_text())
    for package_name, version in (("dynamo-parsers-v2", target["version"]), ("dynamo-parsers", target["reasoning_version"])):
        matches = [package for package in lock["package"] if package["name"] == package_name and package["version"] == version]
        if len(matches) != 1:
            raise ValueError(f"lockfile does not pin exact {package_name} {version}")
        if package_name.endswith("-v2") and target["source_kind"] == "crate" and matches[0].get("checksum") != target["checksum"]:
            raise ValueError("Cargo resolved a different published artifact")
    return {"env": env, "manifest": manifest, "source": source, "flags": flags}


def adapter_files() -> list[Path]:
    return [HARNESS, *sorted(COMMON.glob("*.rs")), ROOT / "conformance/utils/src/parser_families.yaml",
                     ROOT / "conformance/utils/src/unified_tools.json", Path(__file__), Path(capture_dynamo_unified_batch.__file__)]


def capture(target: dict, jobs: list[Path], output: Path, work: Path, target_dir: Path) -> list[Path]:
    adapter_bytes = {path: path.read_bytes() for path in adapter_files()}
    prepared = prepare(target, work, target_dir)
    before = {path: path.read_bytes() for path in jobs}
    all_jobs = [json.loads(before[path]) for path in jobs]
    (work / "jobs.json").write_text(json.dumps(all_jobs, ensure_ascii=False))
    env = {**prepared["env"], "XVER_JOBS": str(work / "jobs.json"), "XVER_JSON_OUTPUT": str(work / "results.json"),
           "CONFORMANCE_FAMILIES": str(ROOT / "conformance/utils/src/parser_families.yaml"),
           "XVER_FAMILIES": str(ROOT / "conformance/utils/src/parser_families.yaml")}
    _cargo("cargo", ["test", "--locked", "--manifest-path", str(prepared["manifest"]), "--test", "capture_cross_version", "capture_family_jobs", "--", "--exact", "--nocapture"], work_dir=work, log=work / "capture.log", env=env)
    results = json.loads((work / "results.json").read_text())
    if set(results) != {job["family"] for job in all_jobs}:
        raise ValueError("producer did not cover the exact requested family set")
    if any(path.read_bytes() != contents for path, contents in adapter_bytes.items()):
        raise ValueError("capture adapter changed during execution")
    adapter_hash = digest(b"".join(adapter_bytes.values()))
    source_hash = digest(json.dumps({"target": target, "lock_sha256": digest((work / "Cargo.lock").read_bytes())}, sort_keys=True).encode())
    output.mkdir(parents=True)
    paths = []
    for path, job in zip(jobs, all_jobs):
        if path.read_bytes() != before[path]:
            raise ValueError("capture job changed during execution")
        provenance = {"implementation": "dynamo_v2", "version": target["version"], "git_commit": target["git_commit"],
                      "source_sha256": source_hash, "adapter_sha256": adapter_hash, "job_sha256": digest(before[path]),
                      "crate_checksum": target.get("checksum"), "reasoning_version": target["reasoning_version"],
                      "results_sha256": digest(json.dumps(results[job["family"]], sort_keys=True, ensure_ascii=False).encode()),
                      "cargo_lock_sha256": digest((work / "Cargo.lock").read_bytes())}
        document = {"version": target["version"], "provenance": provenance, "results": results[job["family"]]}
        destination = output / (job["family"] + ".json")
        destination.write_text(json.dumps(document, ensure_ascii=False) + "\n")
        paths.append(destination)
    return paths
