# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
from pathlib import Path
from subprocess import CompletedProcess

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import capture_dynamo_unified as unified_capture  # noqa: E402
import capture_dynamo_unified_batch as capture  # noqa: E402
from impls import batch_capture_label


@pytest.mark.parametrize(
    ("symbols", "legacy"),
    [
        ("", True),
        ("pub use unified::tool_arguments_raw;", True),
        ("pub use unified::assemble_with_tool_indices;", True),
        ("pub use unified::{assemble_with_tool_indices, tool_arguments_raw};", False),
    ],
)
def test_native_assembly_cfg_uses_available_source_capabilities(symbols: str, legacy: bool) -> None:
    assert unified_capture.native_assembly_cfg(symbols) == (
        ["conformance_legacy_indexed_assembly"] if legacy else []
    )


@pytest.mark.parametrize("modern", [False, True])
def test_unified_probe_compiles_only_public_optional_apis(tmp_path, monkeypatch, modern):
    source = tmp_path / "source"
    (source / "src/unified").mkdir(parents=True)
    (source / "src/tool_calling").mkdir()
    (source / "Cargo.toml").write_text('[package]\nversion="0.6.0"\n')
    exports = "create_unified_parser_for_family UnifiedParserEvent REGISTERED_FAMILIES assemble_with_tool_indices tool_arguments_raw"
    (source / "src/lib.rs").write_text(exports + (" structural_tag_builder_for_family" if modern else ""))
    (source / "src/unified/mod.rs").write_text("fn initialize_request" + (" fn preserve_message_phases" if modern else ""))
    (source / "src/tool_calling/traits.rs").write_text("pub complete:")
    monkeypatch.setattr(unified_capture, "crate_source", lambda target, directory: source)

    def cargo(_program, arguments, *, work_dir, log, env):
        assert arguments[0] == "generate-lockfile"
        (work_dir / "Cargo.lock").write_text(
            '[[package]]\nname="dynamo-parsers-v2"\nversion="0.6.0"\nchecksum="' + "a" * 64
            + '"\n[[package]]\nname="dynamo-parsers"\nversion="8.2.2"\n')

    monkeypatch.setattr(unified_capture, "_cargo", cargo)
    monkeypatch.setenv("CONFORMANCE_GRAMMAR_PYTHON", "/verified/python")
    monkeypatch.setenv("CONFORMANCE_STRUCTURAL_TAG_CHECKER", "/untracked/checker.py")
    prepared = unified_capture.prepare({"version": "0.6.0", "source_kind": "crate",
                                       "checksum": "a" * 64, "reasoning_version": "8.2.2"},
                                      tmp_path / "work", tmp_path / "target")
    assert ("conformance_legacy_message_phases" in prepared["flags"]) is not modern
    assert ("conformance_no_structural_tag" in prepared["flags"]) is not modern
    assert prepared["env"]["CONFORMANCE_GRAMMAR_PYTHON"] == "/verified/python"
    checker = tmp_path / "work/utils/src/check_structural_tag.py"
    assert prepared["env"]["CONFORMANCE_STRUCTURAL_TAG_CHECKER"] == str(checker)
    for path in unified_capture.STRUCTURAL_VERIFIER_FILES:
        copied = tmp_path / "work" / path.relative_to(unified_capture.ROOT / "conformance")
        assert copied.read_bytes() == path.read_bytes()
        assert path in unified_capture.adapter_files()


def test_probe_job_uses_registered_gpt_oss_pipeline_without_changing_request() -> None:
    job = {"family": "gpt_oss", "cases": [{"id": "UNIFIED.example.gpt_oss", "request": {"input": "authored"}}]}

    wrapped = capture._probe_job(job)

    assert wrapped["reasoning_parser"] == "gpt_oss"
    assert wrapped["tool_parser"] == "harmony"
    assert wrapped["job"] is job


def test_probe_job_rejects_unregistered_family() -> None:
    with pytest.raises(ValueError, match="no Dynamo v1 batch pipeline"):
        capture._probe_job({"family": "unknown", "cases": []})


def test_crate_checksum_requires_exact_registry_package(tmp_path: Path) -> None:
    lock = tmp_path / "Cargo.lock"
    lock.write_text(
        'version = 3\n\n[[package]]\nname = "dynamo-parsers"\nversion = "10.0.1"\n'
        'source = "registry+https://github.com/rust-lang/crates.io-index"\n'
        f'checksum = "{"a" * 64}"\n'
    )

    assert capture._crate_checksum(lock, "10.0.1") == "a" * 64
    with pytest.raises(ValueError, match="does not pin"):
        capture._crate_checksum(lock, "9.2.13")


def test_registered_source_blocker_binds_every_unattempted_request(tmp_path: Path) -> None:
    job_path = tmp_path / "job.json"
    output_path = tmp_path / "capture.json"
    request = {
        "input": "authored",
        "init": {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None},
        "finish_reason": "stop",
        "tools": [],
        "chunks": [{"delta_text": "authored"}],
    }
    job = {"family": "gpt_oss", "cases": [{"id": "UNIFIED.example.gpt_oss", "request": request}]}
    job_path.write_text(json.dumps(job))

    provenance = capture.source_unavailable("9.2.13", job_path, output_path)
    result = json.loads(output_path.read_text())

    assert provenance["status"] == "source_unavailable"
    assert "source_sha256" not in provenance
    assert result["results"]["UNIFIED.example.gpt_oss"]["capture_input"] == request
    assert result["results"]["UNIFIED.example.gpt_oss"]["capture_observation"]["unavailable"]["code"] == "source_unavailable"


@pytest.mark.parametrize("jail_buffer", [False, True])
def test_capture_uses_the_requested_version_isolated_target_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, jail_buffer: bool) -> None:
    job_path = tmp_path / "job.json"
    output_path = tmp_path / "capture.json"
    work_dir = tmp_path / "producer"
    target_dir = tmp_path / "target-10.0.1"
    request = {
        "input": "authored",
        "init": {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None},
        "finish_reason": "stop",
        "tools": [],
        "chunks": [{"delta_text": "authored"}],
    }
    job = {"family": "gpt_oss", "cases": [{"id": "case", "request": request}]}
    job_path.write_text(json.dumps(job))
    calls = []

    lock_text = (
        'version = 3\n\n[[package]]\nname = "dynamo-parsers"\nversion = "10.0.1"\n'
        'source = "registry+https://github.com/rust-lang/crates.io-index"\n'
        f'checksum = "{"a" * 64}"\n'
        '\n[[package]]\nname = "dynamo-protocols"\nversion = "8.1.0"\n'
    )
    reviewed_lock = tmp_path / "reviewed.lock"
    reviewed_lock.write_text(lock_text)
    published = tmp_path / "published"
    (published / "src/tool_calling").mkdir(parents=True)
    (published / "src/reasoning").mkdir()
    (published / "src/tool_calling/parsers.rs").write_text("pub async fn detect_and_parse_tool_call_with_recovery; tools: Option<&[ToolDefinition]>")
    (published / "src/tool_calling/mod.rs").write_text("ToolDefinition; pub strict:")
    (published / "src/tool_calling/jail.rs").write_text("JailedStream")
    (published / "src/reasoning/mod.rs").write_text("fn set_in_reasoning")

    def fake_run(args, *, cwd, env, capture_output, check):
        calls.append((args, cwd, env["CARGO_TARGET_DIR"]))
        if "generate-lockfile" in args or "metadata" in args:
            (cwd / "Cargo.lock").write_text(lock_text)
            metadata = {"packages": [{"name": "dynamo-parsers", "version": "10.0.1", "manifest_path": str(published / "Cargo.toml")}]}
            return CompletedProcess(args, 0, json.dumps(metadata).encode() if "metadata" in args else b"", b"")
        return CompletedProcess(
            args,
            0,
            json.dumps({
                "case": {
                    "batch_final": {"reasoning_text": "", "normal_text": "authored", "calls": []},
                    "capture_input": request,
                }
            }).encode(),
            b"",
        )

    monkeypatch.setattr(capture.subprocess, "run", fake_run)
    monkeypatch.setattr(capture, "PROBE", tmp_path / "legacy_batch_probe.rs")
    capture.PROBE.write_text("probe")

    receipt = capture.capture("10.0.1", job_path, output_path, work_dir, target_dir=target_dir,
                              lockfile=reviewed_lock if jail_buffer else None, jail_buffer=jail_buffer)

    assert [call[2] for call in calls] == [str(target_dir)] * 3
    assert receipt["job_sha256"] == capture._sha256(job_path.read_bytes())
    result = json.loads(output_path.read_text())["results"]["case"]
    assert result["capture_input"] == request
    assert result["assembled"] == [{"kind": "text", "text": "authored"}]
    assert receipt["implementation"] == ("dynamo_v1_jail" if jail_buffer else "dynamo_v1")
    assert ("--features" in calls[-1][0]) is jail_buffer


def test_jail_requires_the_reviewed_exact_release_lockfile(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires the reviewed release lockfile"):
        capture.capture("3.0.0", tmp_path / "job.json", tmp_path / "output.json", tmp_path / "producer",
                        jail_buffer=True)


@pytest.mark.parametrize("version", ["9.2.13"])
def test_jail_source_blockers_preserve_request_and_separate_identity(tmp_path: Path, version: str) -> None:
    job = tmp_path / "job.json"
    output = tmp_path / "result.json"
    request = {"input": "authored"}
    job.write_text(json.dumps({"family": "muse_glimmer", "cases": [{"id": "case", "request": request}]}))
    receipt = capture.source_unavailable(version, job, output, jail_buffer=True)
    result = json.loads(output.read_text())
    assert receipt["implementation"] == "dynamo_v1_jail"
    assert receipt["status"] == "source_unavailable"
    assert result["results"]["case"]["capture_input"] == request
    assert "source_sha256" not in receipt


def test_published_release_without_jail_api_is_not_a_missing_source(tmp_path: Path) -> None:
    assert "source unavailable" not in batch_capture_label("dynamo_v1_jail", "3.0.0")
    with pytest.raises(ValueError, match="no explicit source blocker"):
        capture.source_unavailable("3.0.0", tmp_path / "job.json", tmp_path / "result.json", jail_buffer=True)


def test_dynamo_adapter_identity_binds_registry_bytes(tmp_path, monkeypatch):
    job = tmp_path / "job.json"
    job.write_text(json.dumps({"family": "gpt_oss", "cases": []}))
    config = tmp_path / "parser_families.yaml"
    config.write_text("first registry")
    monkeypatch.setattr(capture, "parser_families_path", lambda: config)
    before = capture.source_unavailable("9.2.13", job, tmp_path / "first.json")
    config.write_text("changed registry")
    after = capture.source_unavailable("9.2.13", job, tmp_path / "second.json")
    assert before["adapter_sha256"] != after["adapter_sha256"]
    assert before["job_sha256"] == after["job_sha256"]


@pytest.mark.parametrize("jail", [False, True])
def test_ds41_public_generic_config_is_batch_only_and_does_not_rewrite_request(jail):
    request = {"input": "<｜DSML｜ calls><｜DSML｜ invoke name=\"f\"></｜DSML｜ invoke></｜DSML｜ calls>",
               "init": {"starting_state": "Response", "tool_output_mode": "Native", "named_tool": None},
               "tools": [], "finish_reason": "stop", "chunks": []}
    job = {"family": "deepseek_v41", "cases": [{"id": "case", "request": request}]}
    before = json.loads(json.dumps(job))
    wrapped = capture._probe_job(job, jail_buffer=jail)
    assert job == before and wrapped["job"] is job
    assert ("tool_config" in wrapped) is not jail
    if not jail:
        config = wrapped["tool_config"]["parser_config"]
        assert config == {
            "type": "dsml",
            "block_start": "<｜DSML｜ calls>", "block_end": "</｜DSML｜ calls>",
            "invoke_start_prefix": "<｜DSML｜ invoke name=", "invoke_end": "</｜DSML｜ invoke>",
            "parameter_prefix": "<｜DSML｜ parameter name=", "parameter_end": "</｜DSML｜ parameter>",
            "allow_eof_recovery": True,
        }
