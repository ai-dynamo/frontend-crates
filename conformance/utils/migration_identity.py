#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render and seal a report against an exact Git tree, or verify its seal."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_tree(source, tree):
    entries = subprocess.check_output(["git", "ls-tree", "-rz", tree], cwd=source)
    tracked = set()
    for entry in entries.split(b"\0"):
        if not entry:
            continue
        metadata, name = entry.split(b"\t", 1)
        mode, kind, oid = metadata.decode().split()
        path = source / os.fsdecode(name)
        tracked.add(path)
        if kind != "blob":
            raise ValueError(f"unsupported tree entry: {path}")
        if (mode == "120000") != path.is_symlink():
            raise ValueError(f"candidate/source mode mismatch: {path}")
        if mode != "120000" and bool(path.stat().st_mode & 0o111) != (mode == "100755"):
            raise ValueError(f"candidate/source mode mismatch: {path}")
        data = os.readlink(path).encode() if mode == "120000" else path.read_bytes()
        actual = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if actual != oid:
            # Original archive checkouts may contain smudged LFS bytes.
            pointer = subprocess.check_output(["git", "cat-file", "blob", oid], cwd=source)
            expected = f"oid sha256:{hashlib.sha256(data).hexdigest()}\nsize {len(data)}\n".encode()
            if not (pointer.startswith(b"version https://git-lfs.github.com/spec/v1\n") and pointer.endswith(expected)):
                raise ValueError(f"candidate/source identity mismatch: {path}")
    # Report producers can import untracked siblings. Include ignored files in
    # this survey; a .gitignore rule must not make an executable owner invisible.
    untracked = subprocess.check_output(["git", "ls-files", "--others", "-z"], cwd=source)
    for name in untracked.split(b"\0"):
        if not name:
            continue
        relative = Path(os.fsdecode(name))
        path = source / relative
        if path in tracked or any(part in {"target", ".venv", "__pycache__", "node_modules", ".pytest_cache"} for part in relative.parts):
            continue
        if path.suffix in {".py", ".rs", ".sh", ".js", ".j2", ".toml"}:
            raise ValueError(f"untracked source outside candidate tree: {path}")
    return len(entries.split(b"\0")) - 1


def cache_inventory(cache):
    if not cache.is_dir():
        raise ValueError(f"missing report fixture cache: {cache}")
    inventory = {}
    for path in sorted(cache.rglob("*")):
        relative = path.relative_to(cache).as_posix()
        if path.is_symlink():
            inventory[relative] = {"symlink": os.readlink(path)}
        elif path.is_file():
            inventory[relative] = {"sha256": file_hash(path)}
    if not inventory:
        raise ValueError(f"empty report fixture cache: {cache}")
    return inventory


def verify_seal(seal):
    source = Path(seal["source"])
    verify_tree(source, seal["tree"])
    for filename, expected in seal["outputs"].items():
        if file_hash(Path(filename)) != expected:
            raise ValueError(f"candidate/output identity mismatch: {filename}")
    if "fixture_inputs" not in seal:
        raise ValueError("report seal has no fixture input identity; render a new seal")
    if cache_inventory(Path(seal["cache"])) != seal["fixture_inputs"]:
        raise ValueError("candidate/fixture input identity mismatch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--tree")
    parser.add_argument("--cache", type=Path, help="Parent for a new retained isolated fixture cache")
    parser.add_argument("--html", type=Path)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.verify:
        verify_seal(json.loads(args.verify.read_text()))
        print("report identity verified")
        return
    if any(value is None for value in (args.source, args.tree, args.cache, args.html, args.log, args.output)):
        parser.error("render requires source, tree, cache, html, log, and output")
    args.source = args.source.resolve()
    args.html = args.html.resolve()
    files = verify_tree(args.source, args.tree)
    # Extraction trusts existing cache state without checking fixture bytes.
    # A tree seal therefore needs a fresh cache, retained for report fixture links;
    # its inventory also detects later changes to those linked inputs. This binds
    # report inputs and outputs, not the correctness of the renderer itself.
    args.cache.mkdir(parents=True, exist_ok=True)
    cache = Path(tempfile.mkdtemp(prefix="migration-report-", dir=args.cache.resolve()))
    # The HTTP server must traverse the retained fixture targets linked by the report.
    cache.chmod(0o755)
    command = ["bash", "conformance/utils/render_table_v2.sh", "--output", str(args.html)]
    with args.log.open("w") as log:
        subprocess.run(command, cwd=args.source, env=dict(os.environ, XDG_CACHE_HOME=str(cache)),
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    verify_tree(args.source, args.tree)
    seal = {"source": str(args.source), "tree": args.tree, "tracked_files": files,
            "command": command, "exit_code": 0, "cache": str(cache),
            "fixture_inputs": cache_inventory(cache),
            "outputs": {str(path): file_hash(path) for path in (args.html, args.html.with_suffix(".json"))}}
    args.output.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n")
    verify_seal(seal)
    print(json.dumps(seal))


if __name__ == "__main__":
    main()
