# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runs the `parser_roundtrip` Rust example: one JSONL job per case in, one
answer per case out, each a list of ordered Unified parser events."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .report import open_private

REPO = Path(__file__).resolve().parents[4]
PARSER_COMMAND = (
    "cargo",
    "run",
    "--quiet",
    "-p",
    "dynamo-conformance-fixtures-v2",
    "--example",
    "parser_roundtrip",
)


def run_parser(
    jobs: list[dict[str, Any]], command: Sequence[str], stderr_path: Path
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """The parser's header line and its answers by case ID. Its stderr goes to
    `stderr_path` (0600): a panic or error message can quote model text."""
    stdin = "".join(json.dumps(job, ensure_ascii=False) + "\n" for job in jobs)
    with open_private(stderr_path) as stderr:
        completed = subprocess.run(
            list(command),
            input=stdin,
            stdout=subprocess.PIPE,
            stderr=stderr,
            encoding="utf-8",
            cwd=REPO,
        )
    if completed.returncode:
        raise RuntimeError(
            f"the parser exited {completed.returncode}; stderr is in {stderr_path}"
        )
    # Split on LF only: JSON keeps U+2028 literal, which splitlines() would break.
    lines = [json.loads(line) for line in completed.stdout.split("\n") if line]
    if not lines:
        raise RuntimeError("the parser wrote no header")
    answers = {answer["id"]: answer for answer in lines[1:]}
    if len(answers) != len(jobs) or any(job["id"] not in answers for job in jobs):
        raise RuntimeError(f"the parser answered {len(answers)} of {len(jobs)} jobs")
    return lines[0], answers
