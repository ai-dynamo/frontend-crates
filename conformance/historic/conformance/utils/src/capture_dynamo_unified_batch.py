#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the shared legacy Dynamo batch probe against one exact crate release."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tomllib
from pathlib import Path

import yaml

from capture_stimulus import batch_final_events, unavailable_result
from impls import BATCH_SOURCE_BLOCKERS, DYNAMO_JAIL_UNIFIED_PIPELINES, DYNAMO_UNIFIED_PIPELINES, parser_families_path


PROBE = Path(__file__).resolve().with_name("legacy_batch_probe.rs")
JAIL_PROBE = PROBE.with_name("legacy_jail_probe.rs")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _adapter_sha256(jail_buffer: bool) -> str:
    return _sha256(b"".join(path.read_bytes() for path in adapter_files(jail_buffer)))


def adapter_files(jail_buffer: bool) -> list[Path]:
    return [PROBE, *([JAIL_PROBE] if jail_buffer else []), parser_families_path(), Path(__file__)]


def _probe_job(job: dict, jail_buffer: bool = False) -> dict:
    family = job.get("family")
    pipelines = DYNAMO_JAIL_UNIFIED_PIPELINES if jail_buffer else DYNAMO_UNIFIED_PIPELINES
    if family not in pipelines and family not in yaml.safe_load(parser_families_path().read_text())["unified"]:
        raise ValueError(f"no Dynamo v1 batch pipeline registered for family {family!r}")
    reasoning_parser, tool_parser = pipelines.get(family, ("", ""))
    return {
        "reasoning_parser": reasoning_parser,
        "tool_parser": tool_parser,
        "job": job,
    }


def _crate_checksum(lock_path: Path, version: str) -> str:
    lock = tomllib.loads(lock_path.read_text())
    matches = [
        package for package in lock["package"]
        if package["name"] == "dynamo-parsers" and package["version"] == version
    ]
    if len(matches) != 1 or matches[0].get("source") != "registry+https://github.com/rust-lang/crates.io-index":
        raise ValueError(f"Cargo.lock does not pin dynamo-parsers {version} from crates.io")
    checksum = matches[0].get("checksum")
    if not isinstance(checksum, str) or len(checksum) != 64:
        raise ValueError(f"Cargo.lock has no registry checksum for dynamo-parsers {version}")
    return checksum


def source_unavailable(version: str, job_path: Path, output_path: Path, jail_buffer: bool = False) -> dict:
    implementation = "dynamo_v1_jail" if jail_buffer else "dynamo_v1"
    detail = BATCH_SOURCE_BLOCKERS.get((implementation, version))
    if detail is None:
        raise ValueError(f"no explicit source blocker is registered for {implementation} {version}")
    job_bytes = job_path.read_bytes()
    job = json.loads(job_bytes)
    _probe_job(job, jail_buffer)
    results = {
        case["id"]: unavailable_result("source_unavailable", detail, capture=case["request"], observation=True)
        for case in job["cases"]
    }
    provenance = {
        "implementation": implementation,
        "version": version,
        "status": "source_unavailable",
        "adapter_sha256": _adapter_sha256(jail_buffer),
        "job_sha256": _sha256(job_bytes),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.write_text(json.dumps({"version": version, "results": results, "provenance": provenance}, ensure_ascii=False) + "\n")
    return provenance


def _cargo(cargo: str, args: list[str], *, work_dir: Path, log: Path, env: dict[str, str]) -> subprocess.CompletedProcess:
    result = subprocess.run(
        [cargo, *args], cwd=work_dir, env=env, capture_output=True, check=False,
    )
    log.write_bytes(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f"cargo command failed with exit {result.returncode}; see {log}")
    return result


def compatibility_flags(source: Path, has_protocol: bool) -> list[str]:
    """Bind historical API adaptations to the actual checksum-verified crate."""
    tool_source = (source / "src/tool_calling/parsers.rs").read_text()
    reason_source = (source / "src/reasoning/mod.rs").read_text()
    tool_module = (source / "src/tool_calling/mod.rs").read_text()
    flags = []
    if "detect_and_parse_tool_call_with_recovery" not in tool_source:
        flags.append("conformance_old_tool_api")
    if "async fn detect_and_parse_tool_call(" in tool_source:
        flags.append("conformance_async_tool_api")
    if "tools: Option<&[ToolDefinition]>" in tool_source:
        flags.append("conformance_tool_api_schemas")
    if "ToolDefinition" not in tool_module:
        flags.append("conformance_no_tool_definitions")
    if "pub strict:" not in tool_module:
        flags.append("conformance_no_tool_strict")
    if "fn set_in_reasoning" not in reason_source:
        flags.append("conformance_no_reasoning_init")
    if not has_protocol or not (source / "src/tool_calling/jail.rs").exists() and not (source / "src/tool_calling/jail/mod.rs").exists():
        flags.append("conformance_no_jail")
    return flags


def capture(
    version: str,
    job_path: Path,
    output_path: Path,
    work_dir: Path,
    cargo: str = "cargo",
    target_dir: Path | None = None,
    lockfile: Path | None = None,
    jail_buffer: bool = False,
) -> dict:
    if jail_buffer and lockfile is None:
        raise ValueError("jail-buffer capture requires the reviewed release lockfile")
    protocol_version = None
    if jail_buffer:
        packages = tomllib.loads(lockfile.read_text())["package"]
        protocol_version = next((package["version"] for package in packages if package["name"] == "dynamo-protocols"), None)
    if not version or any(character not in "0123456789." for character in version):
        raise ValueError(f"invalid Dynamo parser release version: {version!r}")
    if not job_path.is_file():
        raise FileNotFoundError(job_path)
    if not PROBE.is_file():
        raise FileNotFoundError(PROBE)
    work_dir.mkdir(parents=True, exist_ok=True)
    if any(work_dir.iterdir()):
        raise ValueError(f"producer work directory is not empty: {work_dir}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(output_path)

    adapter_before = _adapter_sha256(jail_buffer)
    job_bytes = job_path.read_bytes()
    job = json.loads(job_bytes)
    probe_job_path = work_dir / "probe-job.json"
    probe_job_path.write_text(json.dumps(_probe_job(job, jail_buffer), ensure_ascii=False) + "\n")
    (work_dir / "Cargo.toml").write_text(
        "[package]\nname = \"dynamo-unified-batch-probe\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n"
        "[dependencies]\n"
        f"dynamo-parsers = \"={version}\"\n"
        "serde_json = \"1\"\n"
        "tokio = { version = \"1\", features = [\"macros\", \"rt\"] }\n"
        + (f'futures = "0.3"\ndynamo-protocols = "={protocol_version}"\n' if jail_buffer and protocol_version else "")
        + '\n[features]\njail = []\n'
    )
    source_dir = work_dir / "src"
    source_dir.mkdir()
    shutil.copyfile(PROBE, source_dir / "main.rs")
    if jail_buffer:
        shutil.copyfile(JAIL_PROBE, source_dir / "legacy_jail_probe.rs")
    env = {key: value for key, value in os.environ.items() if key in {"PATH", "HOME", "USER", "CARGO_HOME", "RUSTUP_HOME", "TMPDIR", "TMP", "TEMP", "SSL_CERT_FILE", "SSL_CERT_DIR"}}
    env["CARGO_TARGET_DIR"] = str(target_dir or work_dir / "target")
    lock_path = work_dir / "Cargo.lock"
    if lockfile is None:
        _cargo(cargo, ["generate-lockfile", "--manifest-path", str(work_dir / "Cargo.toml")],
               work_dir=work_dir, log=work_dir / "cargo-lock.log", env=env)
    else:
        if not lockfile.is_file():
            raise FileNotFoundError(lockfile)
        lock_path.write_bytes(lockfile.read_bytes())
        if jail_buffer and protocol_version:
            _cargo(cargo, ["metadata", "--offline", "--format-version", "1", "--features", "jail", "--manifest-path", str(work_dir / "Cargo.toml")],
                   work_dir=work_dir, log=work_dir / "cargo-lock.log", env=env)
    crate_checksum = _crate_checksum(lock_path, version)
    metadata = _cargo(cargo, ["metadata", "--locked", "--format-version", "1", "--manifest-path", str(work_dir / "Cargo.toml")],
        work_dir=work_dir, log=work_dir / "source-metadata.log", env=env)
    packages = json.loads(metadata.stdout)["packages"]
    source = Path(next(package["manifest_path"] for package in packages if package["name"] == "dynamo-parsers" and package["version"] == version)).parent
    flags = compatibility_flags(source, bool(protocol_version))
    env["RUSTFLAGS"] = " ".join(f"--cfg {flag}" for flag in flags)
    result = _cargo(
        cargo,
        ["run", "--locked", "--quiet", "--manifest-path", str(work_dir / "Cargo.toml"),
         *(["--features", "jail"] if jail_buffer else []), "--", str(probe_job_path)],
        work_dir=work_dir,
        log=work_dir / "cargo-run.log",
        env=env,
    )
    results = json.loads(result.stdout)
    for record in results.values():
        if "batch_final" in record:
            record["assembled"] = batch_final_events(record["batch_final"])
    adapter_sha256 = _adapter_sha256(jail_buffer)
    if adapter_sha256 != adapter_before or job_path.read_bytes() != job_bytes:
        raise ValueError("batch adapter or request changed during execution")
    lock_sha256 = _sha256(lock_path.read_bytes())
    source_sha256 = _sha256(json.dumps({
        "crate_checksum": crate_checksum,
        "cargo_lock_sha256": lock_sha256,
        "compatibility_flags": flags,
    }, sort_keys=True).encode())
    provenance = {
        "implementation": "dynamo_v1_jail" if jail_buffer else "dynamo_v1",
        "version": version,
        "source_sha256": source_sha256,
        "adapter_sha256": adapter_sha256,
        "job_sha256": _sha256(job_bytes),
        "crate_checksum": crate_checksum,
        "cargo_lock_sha256": lock_sha256,
        "compatibility_flags": flags,
    }
    provenance["results_sha256"] = _sha256(json.dumps(results, sort_keys=True, ensure_ascii=False).encode())
    capture_document = {"version": version, "results": results, "provenance": provenance}
    output_path.write_text(json.dumps(capture_document, ensure_ascii=False) + "\n")
    return provenance


def capture_jobs(version: str, jobs: list[Path], output: Path, work: Path, target_dir: Path, *, jail_buffer=False, lockfile=None) -> list[Path]:
    """Build one exact-version producer, then execute every family with that binary."""
    if not jobs:
        raise ValueError("empty batch capture jobs")
    frozen = {job: job.read_bytes() for job in jobs}
    adapter = _adapter_sha256(jail_buffer)
    first = jobs[0]
    receipt = capture(version, first, output / first.name, work / first.stem, target_dir=target_dir,
                      jail_buffer=jail_buffer, lockfile=lockfile)
    binary = target_dir / "debug/dynamo-unified-batch-probe"
    if not binary.is_file():
        raise FileNotFoundError(binary)
    binary_hash = _sha256(binary.read_bytes())
    paths = [output / first.name]
    env = {key: value for key, value in os.environ.items() if key in {"PATH", "HOME", "USER", "TMPDIR", "TMP", "TEMP"}}
    for job in jobs[1:]:
        family_work = work / job.stem
        family_work.mkdir(parents=True)
        probe_job = family_work / "probe-job.json"
        probe_job.write_text(json.dumps(_probe_job(json.loads(frozen[job]), jail_buffer), ensure_ascii=False))
        result = subprocess.run([str(binary), str(probe_job)], env=env, capture_output=True, check=False)
        (family_work / "execution.log").write_bytes(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f"batch producer failed with exit {result.returncode}; see {family_work / 'execution.log'}")
        results = json.loads(result.stdout)
        for record in results.values():
            if "batch_final" in record:
                record["assembled"] = batch_final_events(record["batch_final"])
        family_receipt = {**receipt, "job_sha256": _sha256(frozen[job]),
            "results_sha256": _sha256(json.dumps(results, sort_keys=True, ensure_ascii=False).encode()), "binary_sha256": binary_hash}
        destination = output / job.name
        if destination.exists():
            raise FileExistsError(destination)
        destination.write_text(json.dumps({"version": version, "results": results, "provenance": family_receipt}, ensure_ascii=False) + "\n")
        paths.append(destination)
    if adapter != _adapter_sha256(jail_buffer) or binary_hash != _sha256(binary.read_bytes()) or any(job.read_bytes() != data for job, data in frozen.items()):
        raise ValueError("batch producer, adapter, or frozen jobs changed during execution")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="exact published dynamo-parsers release")
    parser.add_argument("--job", required=True, type=Path, help="family-scoped batch job JSON")
    parser.add_argument("--output", required=True, type=Path, help="capture JSON output; existing files are never overwritten")
    parser.add_argument("--work-dir", required=True, type=Path, help="empty task-owned producer directory for this version")
    parser.add_argument("--cargo", default="cargo", help="Cargo executable")
    parser.add_argument("--target-dir", type=Path, help="version-isolated Cargo build cache")
    parser.add_argument("--lockfile", type=Path, help="reuse a reviewed lockfile for this exact release")
    parser.add_argument("--source-unavailable", action="store_true", help="record a registered source blocker without claiming execution")
    parser.add_argument("--jail-buffer", action="store_true", help="drive the separate jail-buffer capture identity")
    args = parser.parse_args()
    if args.source_unavailable:
        if args.work_dir.exists():
            raise ValueError("source-unavailable records do not create a producer work directory")
        provenance = source_unavailable(args.version, args.job, args.output, args.jail_buffer)
    else:
        provenance = capture(
            args.version,
            args.job,
            args.output,
            args.work_dir,
            args.cargo,
            args.target_dir,
            args.lockfile,
            args.jail_buffer,
        )
    print(json.dumps({"version": args.version, "provenance": provenance}, sort_keys=True))


if __name__ == "__main__":
    main()
