# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import hashlib
import os
import subprocess
import tarfile
from pathlib import Path

import capture_sources
import unified_archives

ROOT = Path(__file__).resolve().parents[3]


def historical_source_bytes(root: Path, path: str) -> bytes:
    """Use the verified inventory for original-member aliases, never checkout archives."""
    if (root / path).is_file():
        return capture_sources.read(root, path)
    declaration = root / capture_sources.DATASETS
    if declaration.is_file():
        for dataset in reversed(json.loads(declaration.read_bytes())):
            candidate = capture_sources.relative(dataset["root"]) + "/" + capture_sources.relative(path)
            if (root / candidate).is_file():
                return capture_sources.read(root, candidate)
    # The session fixture verifies all evidence once; each Raw check still hashes
    # the selected bytes, without decompressing unrelated historical snapshots.
    inventory = json.loads(capture_sources.read(root, capture_sources.INVENTORY))
    for descriptor in sorted((root / "additions").glob("*/*.json")):
        raw = capture_sources.read(root, descriptor.relative_to(root).as_posix())
        assert descriptor.stem == capture_sources.digest(raw)
        inventory["files"].append(json.loads(raw)["archive"])
    member_path = (path.removeprefix(capture_sources.STORE + "/")
                   if path.startswith(capture_sources.STORE + "/") else path.removeprefix("conformance/"))
    matches = [(archive, member) for archive in inventory["files"]
               for member in archive.get("members", []) if member["path"] in {path, member_path}]
    if len(matches) != 1:
        raise ValueError(f"original resource must have one retained owner: {path}")
    archive, member = matches[0]
    with tarfile.open(root / archive["path"]) as stream:
        data = stream.extractfile(member["path"]).read()
    assert len(data) == member["size"] and hashlib.sha256(data).hexdigest() == member["sha256"]
    return data


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
