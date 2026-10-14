# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dynamo_version import git_subprocess_env

SCRIPT = Path(__file__).resolve().parents[1] / "piece1_coupling_gate.sh"


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        env=git_subprocess_env(),
        text=True,
    )


def test_piece1_gate_rejects_history_only_corpus_change(tmp_path, monkeypatch):
    external_index = tmp_path / "external-index"
    monkeypatch.setenv("GIT_INDEX_FILE", str(external_index))
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "Conformance Test")
    _git(repo, "config", "user.email", "conformance-test@example.invalid")

    script = repo / "conformance/utils/piece1_coupling_gate.sh"
    history = repo / "conformance/fixtures-unified-v2/families/gemma4/inputs_and_golden.yaml"
    manifest = repo / "conformance/fixtures-manifest.json"
    fixtures = repo / "conformance/fixtures/.keep"
    script.parent.mkdir(parents=True)
    history.parent.mkdir(parents=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    fixtures.parent.mkdir(parents=True)
    shutil.copy2(SCRIPT, script)
    history.write_text("baseline\n")
    manifest.write_text("{}\n")
    fixtures.write_text("")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "baseline")

    history.write_text("changed history only\n")
    _git(repo, "add", str(history.relative_to(repo)))
    _git(repo, "commit", "-m", "change history")

    result = subprocess.run(
        [str(script), "HEAD^"],
        cwd=repo,
        check=False,
        capture_output=True,
        env=git_subprocess_env(),
        text=True,
    )

    assert result.returncode == 1
    assert "FAIL conformance fixture stores + manifest" in result.stdout
    assert "MODIFIED" in result.stdout
    assert not external_index.exists()
