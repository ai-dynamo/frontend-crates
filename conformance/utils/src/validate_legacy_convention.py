#!/usr/bin/env python3
"""Validate the active legacy non-Unified streaming naming convention."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
from pathlib import Path

import yaml
import legacy_history

OLD_TO_NEW = {
    "fixtures-batch-on-stream-" + "v2": "fixtures-batch-on-stream-v1",
    "fixtures-stream-" + "v2": "fixtures-stream-v1",
    "TOOLCALLING_STREAMING_" + "V2": "TOOLCALLING_STREAMING_V1",
    "TOOLCALLING.stream" + "v2": "TOOLCALLING.streamv1",
    "stream" + "v2": "streamv1",
}
OLD_TOKENS = tuple(OLD_TO_NEW)
def _stored_legacy_archives(repo: Path) -> set[str]:
    store = repo / "conformance/fixtures"
    return {
        str(path.relative_to(repo / "conformance/fixtures"))
        for path in store.rglob("*.tar.gz")
    }


def _manifest(repo: Path) -> dict:
    return json.loads((repo / "conformance/fixtures-manifest.json").read_text())


def _active_archives(repo: Path, manifest: dict) -> list[tuple[Path, str]]:
    store = repo / "conformance/fixtures"
    return [
        (store / shard["path"], shard["path"])
        for shard in manifest["shards"]
        if shard["path"].endswith(".tar.gz")
        and (
            "fixtures-stream-v1/" in shard["path"]
            or shard["path"] == "toolcalling/fixtures-batch-on-stream-v1.tar.gz"
        )
    ]


def _archive_errors(archive: Path, relative: str) -> list[str]:
    errors = []
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            errors.append(f"{relative}: duplicate archive member path")
        for member in members:
            if any(token in member.name for token in OLD_TOKENS):
                errors.append(f"{relative}:{member.name}: stale legacy-v2 name")
            if not member.isfile():
                continue
            data = tar.extractfile(member).read()
            if any(token.encode() in data for token in OLD_TOKENS):
                errors.append(f"{relative}:{member.name}: stale legacy-v2 content")
            try:
                document = yaml.safe_load(data) or {}
            except yaml.YAMLError as exc:
                raise ValueError(f"{relative}:{member.name}: invalid YAML") from exc
            if document.get("mode") == "stream" + "v2":
                errors.append(f"{relative}:{member.name}: stale stream mode")
            if any(
                isinstance(case_id, str) and "TOOLCALLING.stream" + "v2" in case_id
                for case_id in (document.get("cases") or {})
            ):
                errors.append(f"{relative}:{member.name}: stale stream case key")
    return errors


def _text_errors(repo: Path) -> list[str]:
    errors = []
    ignored = {"validate_legacy_convention.py"}
    for path in repo.glob("**/*"):
        if not path.is_file() or ".git" in path.parts or path.name in ignored:
            continue
        if path.suffix in {".tar", ".gz", ".so", ".o", ".rlib"}:
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        for token in OLD_TOKENS:
            if token in text:
                errors.append(f"{path}: stale {token}")
    return errors


def validate(repo: Path) -> list[str]:
    errors = _text_errors(repo)
    manifest = _manifest(repo)
    seen = set()
    history = repo / "conformance/fixtures-v1"
    for shard in manifest["shards"]:
        path = shard["path"]
        if path in seen:
            errors.append(f"manifest: duplicate shard path {path}")
        seen.add(path)
        if any(token in path for token in OLD_TOKENS):
            errors.append(f"manifest: stale shard path {path}")
        if path.endswith(".tar.gz"):
            errors.append(f"manifest: archived fixture remains active {path}")
        if shard.get("format") == legacy_history.HISTORY_FORMAT:
            if path != legacy_history.HISTORY_PATH:
                errors.append(f"manifest: invalid legacy YAML path {path}")
            elif not history.is_dir():
                errors.append(f"manifest: missing legacy YAML store {history}")
            else:
                try:
                    digest, size = legacy_history.store_digest(history)
                except ValueError as exc:
                    errors.append(f"manifest: {exc}")
                else:
                    if (digest, size) != (shard["sha256"], shard["size"]):
                        errors.append("manifest: legacy YAML store differs from pin")
    if legacy_history.HISTORY_PATH not in seen:
        errors.append("manifest: legacy YAML store has no pin")
    for path in sorted(_stored_legacy_archives(repo)):
        errors.append(f"archive store: obsolete archive {path}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()
    errors = validate(args.root)
    for error in errors:
        print(error, file=sys.stderr)
    print(f"legacy convention validation: {len(errors)} problem(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
