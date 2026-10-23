# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def run_node_contract(filename: str, *arguments: str) -> str:
    environment = {key: os.environ[key] for key in ("PATH", "LANG", "LC_ALL") if key in os.environ}
    result = subprocess.run(
        ["node", str(ROOT / "conformance/viewer" / filename), *arguments],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def colorize_markup(text: str, family: str, markers: dict) -> str:
    environment = {key: os.environ[key] for key in ("PATH", "LANG", "LC_ALL") if key in os.environ}
    driver = "const mc=require(process.argv[1]);const p=JSON.parse(require('fs').readFileSync(0,'utf8'));process.stdout.write(mc.colorizeMarkup(p.text,p.family,p.markers));"
    result = subprocess.run(
        ["node", "-e", driver, str(ROOT / "conformance/utils/src/assets/colorize.js")],
        input=json.dumps({"text": text, "family": family, "markers": markers}),
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout
