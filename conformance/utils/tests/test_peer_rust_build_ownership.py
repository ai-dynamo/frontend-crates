# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import capture_vllm_rust_unified as capture


@pytest.mark.parametrize("target_mode", ["local", "absolute", "relative"])
def test_interleaved_release_builds_execute_their_own_binary(tmp_path, monkeypatch, target_mode):
    sources = {}
    for version in ("0.25.1", "0.26.0"):
        source = tmp_path / version / "rust"
        for name in ("parser", "tokenizer"):
            crate = source / "src" / name
            crate.mkdir(parents=True)
            (crate / "Cargo.toml").write_text(f'[package]\nname = "vllm-{name}"\nversion = "0.0.0"\n')
        unified = source / "src/parser/src/unified"
        unified.mkdir(parents=True)
        (unified / "mod.rs").write_text("trait UnifiedParser { fn parse_into(&mut self, delta: &str); }\n")
        sources[version] = source
    monkeypatch.setattr(capture, "_vllm_rust_version", lambda source, _parser: source.parent.name)
    if target_mode == "absolute":
        monkeypatch.setenv("CARGO_TARGET_DIR", str(tmp_path / "shared-target"))
    elif target_mode == "relative":
        monkeypatch.setenv("CARGO_TARGET_DIR", "../shared-target")
    else:
        monkeypatch.delenv("CARGO_TARGET_DIR", raising=False)

    real_run = subprocess.run
    second_results = []
    built_paths = []

    def run(command, **kwargs):
        if command[0] != "cargo":
            return real_run(command, **kwargs)
        if command[1] == "build":
            crate = Path(kwargs["cwd"])
            manifest = tomllib.loads((crate / "Cargo.toml").read_text())
            source = Path(manifest["dependencies"]["vllm-parser"]["path"]).parents[1]
            version = source.parent.name
            target = Path(os.environ.get("CARGO_TARGET_DIR", str(crate / "target")))
            if not target.is_absolute():
                target = (crate / target).resolve()
            binary = target / "debug" / manifest["package"]["name"]
            binary.parent.mkdir(parents=True, exist_ok=True)
            binary.write_text(
                f"#!{sys.executable}\n"
                "import json\nimport sys\n"
                f"print(json.dumps({{'version': {version!r}, 'job': json.load(sys.stdin)}}))\n"
            )
            binary.chmod(0o700)
            built_paths.append(binary)
            if version == "0.25.1":
                # A second build finishes and runs after the first publication,
                # before its caller spawns the executable. No timing is assumed.
                second_results.append(json.loads(capture.build_and_run(sources["0.26.0"], '{"job": "second"}')))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(capture.subprocess, "run", run)
    first_result = json.loads(capture.build_and_run(sources["0.25.1"], '{"job": "first"}'))
    assert first_result == {"version": "0.25.1", "job": {"job": "first"}}
    assert second_results == [{"version": "0.26.0", "job": {"job": "second"}}]
    assert len(set(built_paths)) == 2
    if target_mode != "local":
        assert built_paths[0].parent == built_paths[1].parent
