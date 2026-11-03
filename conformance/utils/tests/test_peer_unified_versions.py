# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import io
import json
from pathlib import Path
import shutil
import tarfile
import zipfile

import pytest
import yaml

import capture_peer_unified_versions as peers
from capture_stimulus import execution_outcome
from capture_stimulus import request_for_case
from capture_stimulus import batch_final_events


def source_archive(path, members):
    with tarfile.open(path, "w:gz") as archive:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


def test_published_metadata_and_source_survive_extraction_without_building(tmp_path):
    archive = tmp_path / "sglang-1.0.0.tar.gz"
    metadata = b"Metadata-Version: 2.1\nName: sglang\nVersion: 1.0.0\n"
    source_archive(archive, {"sglang-1.0.0/PKG-INFO": metadata,
                             "sglang-1.0.0/sglang/__init__.py": b"__version__='1.0.0'\n",
                             "sglang-1.0.0/setup.py": b"raise RuntimeError('GPU build must not run')\n"})
    target = tmp_path / "package"
    peers.extract_package(archive, target, "sglang")
    assert (target / "sglang/__init__.py").read_bytes() == b"__version__='1.0.0'\n"
    assert (target / "sglang-frozen.dist-info/METADATA").read_bytes() == metadata
    assert not (target / "setup.py").exists()


def test_package_traversal_is_rejected_before_writing_outside_target(tmp_path):
    archive = tmp_path / "malformed.tar.gz"
    source_archive(archive, {"package/sglang/../../escaped.py": b"wrong"})
    with pytest.raises(ValueError, match="Unsafe"):
        peers.extract_package(archive, tmp_path / "package", "sglang")
    assert not (tmp_path / "escaped.py").exists()


def test_download_must_match_frozen_artifact(monkeypatch, tmp_path):
    monkeypatch.setattr(peers.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(b"altered"))
    with pytest.raises(ValueError, match="frozen hash"):
        peers.download_artifact({"url": "https://example.invalid/release", "sha256": "0" * 64}, tmp_path / "artifact")


def test_native_wheels_are_selected_for_the_actual_worker_python():
    artifacts = [{"filename": name} for name in ["sglang-1-cp310-cp310-manylinux_2_34_x86_64.whl",
                                                 "sglang-1-cp312-cp312-manylinux_2_34_aarch64.whl",
                                                 "sglang-1-cp312-cp312-manylinux_2_34_x86_64.whl"]]
    assert peers.select_artifact({"artifacts": artifacts}) is artifacts[2]


def test_vllm_stable_abi_wheel_supports_python312_without_relabelling():
    artifacts = [{"filename": name} for name in ["vllm-0.15.0-cp38-abi3-manylinux_2_31_aarch64.whl",
                                                 "vllm-0.15.0-cp313-abi3-manylinux_2_31_x86_64.whl",
                                                 "vllm-0.15.0-cp38-abi3-manylinux_2_31_x86_64.whl"]]
    assert peers.select_artifact({"artifacts": artifacts}) is artifacts[2]


def test_dependency_image_overrides_retain_resolved_immutable_ids(monkeypatch):
    calls = []

    def inspect(command, *, text):
        assert text and command[:3] == ["docker", "image", "inspect"]
        calls.append(command[3])
        return "sha256:" + str(len(calls)) + "\n"

    monkeypatch.setattr(peers.subprocess, "check_output", inspect)
    images = peers.resolve_images({"vllm": "vllm/vllm-openai:v0.23.0"})
    assert calls == ["vllm/vllm-openai:v0.23.0" if engine == "vllm" else image for engine, image in peers.IMAGE_NAMES.items()]
    assert list(images.values()) == ["sha256:1", "sha256:2"]


@pytest.mark.parametrize("present_apis", [False, True])
def test_foreign_abi_is_explicit_import_free_inspection_and_never_executes_present_apis(tmp_path, monkeypatch, present_apis):
    filename = "vllm-0.3.2-cp311-cp311-manylinux1_x86_64.whl"
    archive = tmp_path / filename
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("vllm-0.3.2.dist-info/METADATA", "Name: vllm\nVersion: 0.3.2\n")
        stream.writestr("vllm/__init__.py", "raise AssertionError('foreign engine must not import')\n")
        if present_apis:
            for relative in ("entrypoints/openai/protocol.py", "reasoning/__init__.py", "tool_parsers/__init__.py"):
                stream.writestr("vllm/" + relative, "# published parser API\n")
    artifact = {"filename": filename, "sha256": peers.sha256(archive.read_bytes())}
    target = {"implementation": "vllm", "version": "0.3.2", "artifacts": [artifact]}
    record = peers.target_artifact_record(target)
    assert record["inspection_only"] is True and record["artifact_selection"] == "foreign_abi_inspection"
    peers.extract_package(archive, tmp_path / "package", "vllm")
    monkeypatch.syspath_prepend(str(tmp_path / "package"))
    target_path = tmp_path / "target.json"
    target_path.write_bytes(json.dumps(record).encode())
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    case = {"id": "UNIFIED.text", "family": "gemma4", "scenario": "text", "input": "hi",
            "init": {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None},
            "finish_reason": "stop", "tools": [], "chunks": ["hi"]}
    (jobs / "gemma4.json").write_bytes(json.dumps({"family": "gemma4", "cases": [case]}).encode())
    if present_apis:
        with pytest.raises(ValueError, match="cannot execute"):
            peers.worker(target_path, jobs, tmp_path / "captures", "stream")
    else:
        peers.worker(target_path, jobs, tmp_path / "captures", "stream")
        document = json.loads((tmp_path / "captures/gemma4.json").read_bytes())
        assert document["provenance"]["inspection_only"] is True
        assert document["provenance"]["status"] == "capability_inspection"
        assert execution_outcome(document["results"][case["id"]], "stream") == ("unsupported_capability", "unsupported_capture_mode")
    roles, missing = peers.mode_api_inventory(tmp_path / "package/vllm", "vllm", "stream")
    with pytest.raises(ValueError, match="explicit inspection policy"):
        peers.validate_artifact_selection({**record, "inspection_only": False}, missing)


def test_imported_image_package_cannot_be_relabelled_as_requested_source(monkeypatch, tmp_path):
    archive = tmp_path / "sglang-1.0.0.tar.gz"
    source_archive(archive, {"package/PKG-INFO": b"Name: sglang\nVersion: 1.0.0\n", "package/sglang/__init__.py": b"",
                             "package/sglang/srt/entrypoints/openai/protocol.py": b"",
                             "package/sglang/srt/parser/reasoning_parser.py": b"",
                             "package/sglang/srt/function_call/function_call_parser.py": b""})
    peers.extract_package(archive, tmp_path / "package", "sglang")
    target = tmp_path / "target.json"
    target.write_text(json.dumps({"implementation": "sglang", "version": "1.0.0", "artifact": {
        "filename": archive.name, "sha256": peers.sha256(archive.read_bytes())}}))
    foreign = type("Package", (), {"__file__": "/foreign/sglang/__init__.py"})()
    monkeypatch.setattr(peers.importlib.metadata, "version", lambda engine: "1.0.0")
    monkeypatch.setattr(peers.importlib, "import_module", lambda name: foreign)
    with pytest.raises(ValueError, match="outside the isolated frozen artifact"):
        peers.worker(target, tmp_path / "jobs", tmp_path / "output", "stream")


@pytest.mark.parametrize("mutation", ["modify", "add", "remove", "symlink"])
def test_complete_extracted_source_is_bound_to_the_official_archive(tmp_path, mutation):
    archive = tmp_path / "sglang-1.0.0.tar.gz"
    source_archive(archive, {"package/PKG-INFO": b"Name: sglang\nVersion: 1.0.0\n",
                             "package/sglang/__init__.py": b"",
                             "package/sglang/srt/parser/probe.py": b"OUTPUT='official'\n"})
    target = {"implementation": "sglang", "artifact": {"filename": archive.name, "sha256": peers.sha256(archive.read_bytes())}}
    peers.extract_package(archive, tmp_path / "package", "sglang")
    assert peers.validate_extracted_artifact(tmp_path, target)
    parser = tmp_path / "package/sglang/srt/parser/probe.py"
    if mutation == "modify":
        parser.write_bytes(b"OUTPUT='modified'\n")
    elif mutation == "add":
        parser.with_name("injected.py").write_bytes(b"INJECTED=True\n")
    elif mutation == "remove":
        parser.unlink()
    else:
        parser.with_name("injected.py").symlink_to(parser)
    with pytest.raises(ValueError, match="frozen artifact|symbolic link"):
        peers.validate_extracted_artifact(tmp_path, target)


@pytest.mark.parametrize("record", [{"chunks": [123]}, {"chunks": [[{}]]}, {"chunks": [], "batch_final": {}},
                                    {"unavailable": "missing", "assembled": []}])
def test_invalid_capture_shape_cannot_count_as_an_execution(record):
    with pytest.raises(ValueError):
        execution_outcome(record, "stream")


def test_stream_parser_error_preserves_partial_results_and_source_blockers_are_separate():
    assert execution_outcome({"error": "parser failed", "assembled": [{"kind": "text", "text": "partial"}],
                              "chunks": [[{"kind": "text", "text": "partial"}]]}, "stream") == ("parser_error", None)
    result = {"unavailable": "no source", "capture_observation": {"unavailable": {"code": "source_unavailable"}}}
    assert execution_outcome(result, "batch") == ("source_build_blocker", "source_unavailable")
    assert execution_outcome({"assembled": [], "chunks": [{"expected": []}]}, "stream") == ("executed", None)


def test_absent_published_apis_are_inspected_without_claiming_parser_execution(tmp_path, monkeypatch):
    archive = tmp_path / "sglang-1.0.0.tar.gz"
    source_archive(archive, {"package/PKG-INFO": b"Name: sglang\nVersion: 1.0.0\n",
                             "package/sglang/__init__.py": b"raise AssertionError('must not execute absent API')\n"})
    peers.extract_package(archive, tmp_path / "package", "sglang")
    monkeypatch.syspath_prepend(str(tmp_path / "package"))
    target = tmp_path / "target.json"
    target.write_text(json.dumps({"implementation": "sglang", "version": "1.0.0", "artifact": {
        "filename": archive.name, "sha256": peers.sha256(archive.read_bytes())}}))
    job = {"family": "gemma4", "cases": [{"id": "UNIFIED.text.gemma4", "family": "gemma4", "scenario": "text",
        "input": "hello", "init": {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None},
        "finish_reason": "stop", "tools": [], "chunks": ["hello"]}]}
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    (jobs / "gemma4.json").write_text(json.dumps(job))
    output = tmp_path / "output"
    peers.worker(target, jobs, output, "stream")
    capture = json.loads((output / "gemma4.json").read_text())
    assert capture["provenance"]["status"] == "capability_inspection"
    assert capture["provenance"]["extracted_package_sha256"] == peers.validate_extracted_artifact(tmp_path, json.loads(target.read_text()))
    record = capture["results"]["UNIFIED.text.gemma4"]
    assert execution_outcome(record, "stream") == ("unsupported_capability", "unsupported_capture_mode")


def coverage_root(tmp_path, name, *, state="executed", version="1.0.0", jobs_text="hello"):
    work = tmp_path / name
    directory = work / "producers" / f"sglang-{version}"
    directory.mkdir(parents=True)
    archive = directory / f"sglang-{version}.tar.gz"
    source_archive(archive, {"package/PKG-INFO": f"Name: sglang\nVersion: {version}\n".encode(),
                             "package/sglang/__init__.py": b"",
                             "package/sglang/srt/entrypoints/openai/protocol.py": b"",
                             "package/sglang/srt/parser/reasoning_parser.py": b"",
                             "package/sglang/srt/function_call/function_call_parser.py": b""})
    artifact = {"filename": archive.name, "sha256": peers.sha256(archive.read_bytes())}
    target = {"implementation": "sglang", "version": version, "artifacts": [artifact]}
    (work / "inventory.yaml").write_bytes(yaml.safe_dump({"targets": [target]}).encode())
    (directory / "target.json").write_bytes(json.dumps({**target, "artifact": artifact}).encode())
    peers.extract_package(archive, directory / "package", "sglang")
    bundle = work / "adapter"
    bundle.mkdir()
    adapter_root = Path(peers.__file__).parent
    for path in adapter_root.rglob("*.py"):
        destination = bundle / path.relative_to(adapter_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
    (bundle / "parser_families.yaml").write_bytes((adapter_root / "parser_families.yaml").read_bytes())
    (bundle / "probe.py").write_bytes(b"official_adapter = True\n")
    adapter_files = {str(path.relative_to(bundle)): peers.sha256(path.read_bytes()) for path in bundle.rglob("*") if path.is_file()}
    (work / "adapter-manifest.json").write_bytes(json.dumps(adapter_files).encode())
    package = directory / "package/sglang"
    package_files = {str(path.relative_to(package)): peers.sha256(path.read_bytes()) for path in package.rglob("*.py")}
    source = {"version": version, "package_files": package_files, "adapter_files": adapter_files,
              "source_sha256": peers.sha256(json.dumps(package_files, sort_keys=True).encode()),
              "adapter_sha256": peers.sha256(json.dumps(adapter_files, sort_keys=True).encode())}
    digest = peers.validate_extracted_artifact(directory, {**target, "artifact": artifact})
    case = {"id": "UNIFIED.text", "family": "gemma4", "scenario": "text", "input": jobs_text,
            "init": {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None},
            "finish_reason": "stop", "tools": [], "chunks": [jobs_text]}
    jobs, modes = {}, {}
    for mode, job_mode in (("stream", "unified"), ("batch", "batch")):
        job = {"family": "gemma4", "cases": [case]}
        path = work / "jobs" / job_mode / "gemma4.json"
        path.parent.mkdir(parents=True)
        path.write_bytes(json.dumps(job).encode())
        jobs[str(path.relative_to(work))] = peers.sha256(path.read_bytes())
        if state == "missing_evidence":
            continue
        if state == "source_build_blocker":
            modes[mode] = {"blocker": "official dependency cannot build"}
            continue
        output = work / "captures" / ("sglang_python" + ("_batch" if mode == "batch" else "") + "-" + version)
        output.mkdir(parents=True)
        roles, missing = peers.mode_api_inventory(package, "sglang", mode)
        required = peers.required_adapter_files(bundle, "sglang", mode, roles, missing)
        source["adapter_files"] = {name: adapter_files[name] for name in required}
        source["adapter_sha256"] = peers.sha256(json.dumps(source["adapter_files"], sort_keys=True).encode())
        (output / "source.json").write_bytes(json.dumps(source).encode())
        final = {"reasoning_text": "", "normal_text": jobs_text, "calls": []}
        record = {"capture_input": request_for_case(case)}
        if state == "unsupported_capability":
            record.update(unavailable="not registered", capture_observation={"unavailable": {"code": "peer_detector_unavailable"}})
        elif state == "parser_error":
            record["error"] = "upstream parser failure"
        elif mode == "stream":
            record.update(assembled=batch_final_events(final), chunks=[batch_final_events(final)])
        else:
            record.update(assembled=batch_final_events(final), batch_final=final)
        results = {case["id"]: record}
        roles, _missing = peers.mode_api_inventory(package, "sglang", mode)
        receipt = {"implementation": "sglang_python" + ("_batch" if mode == "batch" else ""),
                   "version": version, "artifact_sha256": artifact["sha256"], "extracted_package_sha256": digest,
                   "source_sha256": source["source_sha256"], "adapter_sha256": source["adapter_sha256"],
                   "job_sha256": jobs[str(path.relative_to(work))], "results_sha256": peers.sha256(peers.json_bytes(results)),
                   "status": "executed", "parser_apis": roles}
        document = {"version" if mode == "batch" else "sglang_version": version, "provenance": receipt, "results": results}
        (output / "gemma4.json").write_bytes(json.dumps(document).encode())
        modes[mode] = {"exit_code": 0, "output": str(output)}
    (work / "frozen-jobs.json").write_bytes(json.dumps(jobs).encode())
    if modes:
        (directory / "status.json").write_bytes(json.dumps({"implementation": "sglang", "version": version, "modes": modes}).encode())
    return work, target


def equalize_frozen_artifact(left, right):
    # Tar gzip headers carry timestamps; freeze the identical official bytes in
    # each retained run rather than relying on archive creation wall clock.
    left_target = yaml.safe_load((left / "inventory.yaml").read_bytes())["targets"][0]
    right_target_path = right / "producers" / ("sglang-" + left_target["version"]) / "target.json"
    old = json.loads(right_target_path.read_bytes())
    artifact = left_target["artifacts"][0]
    old_digest = old["artifact"]["sha256"]
    new_digest = artifact["sha256"]
    source_archive_path = left / "producers" / ("sglang-" + left_target["version"]) / artifact["filename"]
    (right_target_path.parent / artifact["filename"]).write_bytes(source_archive_path.read_bytes())
    right_target_path.write_bytes(json.dumps({**left_target, "artifact": artifact}).encode())
    (right / "inventory.yaml").write_bytes((left / "inventory.yaml").read_bytes())
    for capture in (right / "captures").glob("*/*.json"):
        if capture.name != "source.json":
            capture.write_bytes(capture.read_bytes().replace(old_digest.encode(), new_digest.encode()))


def test_master_coverage_selects_strongest_and_latest_and_retains_blockers(tmp_path):
    executed, target = coverage_root(tmp_path, "executed")
    blocker, _ = coverage_root(tmp_path, "blocker", state="source_build_blocker")
    unsupported, _ = coverage_root(tmp_path, "unsupported", state="unsupported_capability")
    error, _ = coverage_root(tmp_path, "error", state="parser_error")
    for root in (blocker, unsupported, error):
        equalize_frozen_artifact(executed, root)
    rows = list(peers.master_coverage_rows({"targets": [target]}, [executed, blocker, unsupported]))
    assert len(rows) == 2 and {row["status"] for row in rows} == {"executed"}
    assert all(row["evidence_root"] == str(executed) and row["prior_blockers"][0]["evidence_root"] == str(blocker) for row in rows)
    rows = list(peers.master_coverage_rows({"targets": [target]}, [executed, error]))
    assert {row["status"] for row in rows} == {"parser_error"}
    assert {row["evidence_root"] for row in rows} == {str(error)}


@pytest.mark.parametrize("mutation", ["request", "target", "adapter", "weak_receipt", "source", "artifact", "empty_adapter", "empty_package"])
def test_master_coverage_rejects_mismatching_or_weak_evidence_even_when_older_success_exists(tmp_path, mutation):
    first, target = coverage_root(tmp_path, "first")
    later, _ = coverage_root(tmp_path, "later", jobs_text="changed" if mutation == "request" else "hello")
    equalize_frozen_artifact(first, later)
    if mutation == "target":
        inventory = yaml.safe_load((later / "inventory.yaml").read_bytes())
        inventory["targets"][0]["yanked"] = True
        (later / "inventory.yaml").write_bytes(yaml.safe_dump(inventory).encode())
    elif mutation == "adapter":
        (later / "adapter/probe.py").write_bytes(b"altered = True\n")
    elif mutation == "source":
        (later / "producers/sglang-1.0.0/package/sglang/__init__.py").write_bytes(b"altered=True\n")
    elif mutation == "artifact":
        (later / "producers/sglang-1.0.0/sglang-1.0.0.tar.gz").write_bytes(b"corrupt")
    elif mutation == "weak_receipt":
        path = later / "captures/sglang_python-1.0.0/gemma4.json"
        document = json.loads(path.read_bytes())
        del document["provenance"]["extracted_package_sha256"]
        path.write_bytes(json.dumps(document).encode())
    elif mutation in ("empty_adapter", "empty_package"):
        source_path = later / "captures/sglang_python-1.0.0/source.json"
        source = json.loads(source_path.read_bytes())
        files_key, hash_key = ("adapter_files", "adapter_sha256") if mutation == "empty_adapter" else ("package_files", "source_sha256")
        source[files_key] = {}
        source[hash_key] = peers.sha256(json.dumps({}).encode())
        source_path.write_bytes(json.dumps(source).encode())
        path = source_path.with_name("gemma4.json")
        document = json.loads(path.read_bytes())
        document["provenance"][hash_key] = source[hash_key]
        path.write_bytes(json.dumps(document).encode())
    with pytest.raises(ValueError):
        list(peers.master_coverage_rows({"targets": [target]}, [first, later]))


def test_master_coverage_counts_absent_master_targets_and_explicit_blockers(tmp_path):
    root, target = coverage_root(tmp_path, "root", state="source_build_blocker")
    absent = {**target, "version": "2.0.0"}
    rows = list(peers.master_coverage_rows({"targets": [target, absent]}, [root]))
    assert len(rows) == 4
    assert [row["status"] for row in rows] == ["source_build_blocker"] * 2 + ["missing_evidence"] * 2
    with pytest.raises(ValueError, match="duplicate coverage"):
        list(peers.master_coverage_rows({"targets": [target]}, [root, root]))


def test_retained_execution_mapping_preserves_receipts_and_checks_containment(tmp_path):
    original, target = coverage_root(tmp_path, "original")
    retained = tmp_path / "retained"
    shutil.copytree(original, retained)
    before = (retained / "captures/sglang_python-1.0.0/gemma4.json").read_bytes()
    mapping = {"schema_version": 1, "original_root": str(original),
               "inventory_sha256": peers.sha256((retained / "inventory.yaml").read_bytes()),
               "frozen_jobs_sha256": peers.sha256((retained / "frozen-jobs.json").read_bytes()),
               "adapter_manifest_sha256": peers.sha256((retained / "adapter-manifest.json").read_bytes())}
    path = retained / "retained-execution.json"
    path.write_bytes(json.dumps(mapping).encode())
    rows = list(peers.master_coverage_rows({"targets": [target]}, [retained]))
    assert {row["status"] for row in rows} == {"executed"}
    assert (retained / "captures/sglang_python-1.0.0/gemma4.json").read_bytes() == before
    mapping["original_root"] = str(tmp_path / "unrelated")
    path.write_bytes(json.dumps(mapping).encode())
    with pytest.raises(ValueError, match="outside the isolated"):
        list(peers.master_coverage_rows({"targets": [target]}, [retained]))
    mapping["original_root"] = str(original)
    mapping["frozen_jobs_sha256"] = "0" * 64
    path.write_bytes(json.dumps(mapping).encode())
    with pytest.raises(ValueError, match="mapping differs"):
        list(peers.master_coverage_rows({"targets": [target]}, [retained]))


def test_master_coverage_publication_is_atomic_and_summary_matches_written_records(tmp_path):
    root, target = coverage_root(tmp_path, "root")
    inventory = {"targets": [target]}
    destination = tmp_path / "coverage.jsonl"
    summary = peers.write_master_coverage(inventory, [root], destination)
    records = [json.loads(line) for line in destination.read_bytes().splitlines()]
    assert len(records) == summary["combinations"] == 2
    assert summary["counts"] == {"executed": 2}
    assert summary["sha256"] == peers.sha256(destination.read_bytes())
    before = destination.read_bytes()
    (root / "adapter/probe.py").write_bytes(b"altered=True\n")
    with pytest.raises(ValueError):
        peers.write_master_coverage(inventory, [root], destination)
    assert destination.read_bytes() == before
    assert not list(tmp_path.glob(".coverage.jsonl.*.tmp"))


def test_peer_import_list_uses_the_same_validated_successful_roots_and_document_bytes(tmp_path):
    root, target = coverage_root(tmp_path, "root")
    destination, imports_path = tmp_path / "coverage.jsonl", tmp_path / "imports.json"
    summary = peers.write_master_coverage({"targets": [target]}, [root], destination, imports_path)
    entries = json.loads(imports_path.read_bytes())
    assert summary["import_list"]["successful_modes"] == 2
    assert summary["import_list"]["documents"] == len(entries) == 2
    for entry in entries:
        assert entry["job_sha256"] == peers.sha256(Path(entry["job"]).read_bytes())
        assert entry["capture_sha256"] == peers.sha256(Path(entry["capture"]).read_bytes())
        assert entry["provenance"] == json.loads(Path(entry["capture"]).read_bytes())["provenance"]
    second, _target = coverage_root(tmp_path, "duplicate")
    equalize_frozen_artifact(root, second)
    before = destination.read_bytes(), imports_path.read_bytes()
    with pytest.raises(ValueError, match="multiple successful"):
        peers.write_master_coverage({"targets": [target]}, [root, second], destination, imports_path)
    assert (destination.read_bytes(), imports_path.read_bytes()) == before


def supersession_fixture(tmp_path):
    invalid, target = coverage_root(tmp_path, "invalid", state="parser_error")
    replacement, _ = coverage_root(tmp_path, "replacement", state="unsupported_capability")
    equalize_frozen_artifact(invalid, replacement)
    status_path = replacement / "producers/sglang-1.0.0/status.json"
    status = json.loads(status_path.read_bytes())
    del status["modes"]["batch"]
    status_path.write_bytes(json.dumps(status).encode())
    manifest = {"schema_version": 1, "replacements": [{"implementation": "sglang_python", "version": "1.0.0",
                "reason": "Earlier adapter called an unsupported upstream API",
                "replacement": peers.evidence_root_binding(replacement),
                "invalidated": [peers.evidence_root_binding(invalid)]}]}
    return invalid, replacement, target, manifest


def test_whole_mode_supersession_replaces_errors_with_unsupported_and_keeps_batch(tmp_path):
    invalid, replacement, target, manifest = supersession_fixture(tmp_path)
    destination, imports = tmp_path / "coverage.jsonl", tmp_path / "imports.json"
    summary = peers.write_master_coverage({"targets": [target]}, [invalid, replacement], destination, imports, manifest)
    rows = [json.loads(line) for line in destination.read_bytes().splitlines()]
    assert {(row["implementation"], row["status"], row["evidence_root"]) for row in rows} == {
        ("sglang_python", "unsupported_capability", str(replacement)),
        ("sglang_python_batch", "parser_error", str(invalid))}
    entries = json.loads(imports.read_bytes())
    assert {(entry["implementation"], entry["evidence_root"]) for entry in entries} == {
        ("sglang_python", str(replacement)), ("sglang_python_batch", str(invalid))}
    assert summary["supersessions"] == manifest


@pytest.mark.parametrize("mutation", ["digest", "omitted", "duplicate", "version", "schema", "replacement", "old_corruption"])
def test_whole_mode_supersession_rejects_unbound_or_invalid_evidence(tmp_path, mutation):
    invalid, replacement, target, manifest = supersession_fixture(tmp_path)
    entry = manifest["replacements"][0]
    if mutation == "digest":
        entry["replacement"]["adapter_manifest_sha256"] = "0" * 64
    elif mutation == "omitted":
        entry["invalidated"] = []
    elif mutation == "duplicate":
        manifest["replacements"].append(entry)
    elif mutation == "version":
        entry["version"] = "2.0.0"
    elif mutation == "schema":
        manifest["schema_version"] = True
    elif mutation == "replacement":
        path = replacement / "producers/sglang-1.0.0/status.json"
        status = json.loads(path.read_bytes())
        status["modes"]["stream"]["exit_code"] = 1
        path.write_bytes(json.dumps(status).encode())
    else:
        (invalid / "captures/sglang_python-1.0.0/gemma4.json").write_bytes(b"{}")
    with pytest.raises((ValueError, KeyError)):
        list(peers.master_coverage_rows({"targets": [target]}, [invalid, replacement], supersessions=manifest))
