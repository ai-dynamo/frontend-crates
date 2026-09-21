# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture origin validation shared by refresh, explode, and capture harnesses.

Unified capture directories and rendered columns are keyed only by the crate
semantic version. The first capture of that version retains its source digest as
origin metadata; later checkouts with the same version do not create another
capture identity. The digest covers parser crates, the split-path protocol
dependency, and workspace build inputs; conformance outputs are excluded so a
capture cannot change its own origin.
"""

import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

from fixture_disposition import DYNAMO_VERSION_RE

ENV_OVERRIDE = "CONFORMANCE_DYNAMO_V2_LABEL"
SOURCE_PATHS = (
    "parsers/v1/src", "parsers/v1/Cargo.toml", "parsers/v1/build.rs",
    "parsers/v2/src", "parsers/v2/Cargo.toml", "parsers/v2/build.rs",
    "protocols/src", "protocols/Cargo.toml", "protocols/build.rs",
    "Cargo.toml", "Cargo.lock",
    "rust-toolchain", "rust-toolchain.toml", ".cargo",
)
_EXTERNAL_GIT_ENV = (
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_COMMON_DIR",
    "GIT_DIR",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_WORK_TREE",
)


def _capture_records(captures: dict, version: str) -> list:
    layer = captures.get(version)
    if isinstance(layer, dict):
        records = layer.get("records")
        return list(records.values()) if isinstance(records, dict) else []
    return layer if isinstance(layer, list) else []


def _legacy_records(captures: dict, version: str) -> list:
    records = _capture_records(captures, version)
    for label in captures:
        if isinstance(label, str) and re.fullmatch(rf"{re.escape(version)}\.patch\d+", label):
            records.extend(_capture_records(captures, label))
    return records


def _legacy_record_matches_current_source(record: object, current: dict) -> bool:
    return isinstance(record, dict) and all(
        record.get(key) == current[key]
        for key in ("crate_version", "source_sha256", "source_id", "source_paths")
    )


def git_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    for name in _EXTERNAL_GIT_ENV:
        env.pop(name, None)
    return env


def crate_version(cargo_toml: Path) -> str:
    package = tomllib.loads(cargo_toml.read_text()).get("package", {})
    version = package.get("version")
    if not isinstance(version, str) or not DYNAMO_VERSION_RE.fullmatch(version):
        raise ValueError(f"no explicit valid [package] version in {cargo_toml}")
    return version


def _git(repo_root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        capture_output=True,
        env=git_subprocess_env(),
    ).stdout


def source_fingerprint(repo_root: Path, revision: str | None = None) -> str:
    entries = {}
    if revision is not None:
        # Read blobs directly: export-ignore/subst attributes must not alter identity.
        tree = _git(repo_root, "ls-tree", "-rz", revision, "--", *SOURCE_PATHS)
        records = [entry.split(b"\t", 1) for entry in tree.split(b"\0") if entry]
        objects = b"".join(meta.split()[2] + b"\n" for meta, _ in records)
        blobs = subprocess.run(
            ["git", "-C", str(repo_root), "cat-file", "--batch"],
            input=objects,
            check=True,
            capture_output=True,
            env=git_subprocess_env(),
        ).stdout
        stream = io.BytesIO(blobs)
        for meta, name in records:
            mode, kind, _oid = meta.split()
            if kind != b"blob" or mode not in (b"100644", b"100755", b"120000"):
                raise ValueError(f"unsupported source entry: {os.fsdecode(name)}")
            size = int(stream.readline().split()[2])
            entries[os.fsdecode(name)] = (2 if mode == b"120000" else int(mode == b"100755"), stream.read(size))
            assert stream.read(1) == b"\n"
    else:
        # Include ignored/untracked files too: Rust can compile an untracked module.
        paths = _git(repo_root, "ls-files", "-z", "--cached", "--others", "--", *SOURCE_PATHS)
        for raw in set(paths.split(b"\0")) - {b""}:
            name = os.fsdecode(raw)
            path = repo_root / name
            if path.is_symlink():
                # Git owns the link text, never the external target's bytes.
                entries[name] = (2, os.fsencode(os.readlink(path)))
                continue
            if not path.exists():
                continue
            entries[name] = (bool(path.stat().st_mode & 0o111), path.read_bytes())
    digest = hashlib.sha256(b"dynamo-capture-source-v1\0")
    for name, (executable, contents) in sorted(entries.items()):
        digest.update(os.fsencode(name) + b"\0" + bytes([executable]))
        digest.update(len(contents).to_bytes(8, "big"))
        digest.update(contents)
    return digest.hexdigest()


def dynamo_v2_provenance(repo_root: Path, override: str | None = None) -> dict:
    repo_root = repo_root.resolve()
    actual_root = Path(os.fsdecode(_git(repo_root, "rev-parse", "--show-toplevel")).strip())
    if actual_root != repo_root:
        raise ValueError(f"expected repository root {actual_root}, got {repo_root}")
    version = crate_version(repo_root / "parsers/v2/Cargo.toml")
    supplied = override if override is not None else os.environ.get(ENV_OVERRIDE)
    if supplied is not None and not supplied.strip():
        raise ValueError("empty Dynamo v2 capture label")
    requested = None if supplied is None else supplied.strip()
    fingerprint = source_fingerprint(repo_root)
    qualified = f"{version}+source.{fingerprint}"
    tag = f"dynamo-parsers-v2-v{version}"
    ref = f"refs/tags/{tag}"
    tags = _git(repo_root, "tag", "--list", tag).decode().splitlines()
    release_commit = None
    released = False
    if tag in tags:
        release_commit = _git(repo_root, "rev-parse", f"{ref}^{{commit}}").decode().strip()
        released = source_fingerprint(repo_root, release_commit) == fingerprint
    if requested is None:
        requested = version if released else "current"
    if requested == version:
        if not released:
            raise ValueError(
                f"source does not match release tag {tag}; use an explicit 'current' "
                f"capture ({qualified}), not release label {version}"
            )
        label, kind = version, "release"
    elif requested in ("current", qualified):
        label, kind = qualified, "unpublished"
    else:
        raise ValueError(f"capture label {requested!r} does not identify this source; expected {qualified!r}")
    return {
        "label": label,
        "kind": kind,
        "crate_version": version,
        "source_sha256": fingerprint,
        "source_id": f"sha256:{fingerprint}",
        "source_paths": list(SOURCE_PATHS),
        "git_commit": _git(repo_root, "rev-parse", "HEAD").decode().strip(),
        "git_head_tree": _git(repo_root, "rev-parse", "HEAD^{tree}").decode().strip(),
        "release_tag": tag if kind == "release" else None,
        "release_commit": release_commit if kind == "release" else None,
    }


def dynamo_v2_label(repo_root: Path, override: str | None = None) -> str:
    provenance = dynamo_v2_provenance(repo_root, override)
    return provenance["crate_version"]


def select_capture_label(repo_root: Path, captures: dict) -> str:
    """Select the semantic current view, with read-only legacy-directory fallback."""
    current = dynamo_v2_provenance(repo_root)
    version = current["crate_version"]
    semantic_records = _capture_records(captures, version)
    if semantic_records:
        # Schema-v3 materializations are a semantic current view. This generated
        # marker distinguishes them from older fixtures that explicitly stored null.
        if all(
            isinstance(record, dict) and record.get("format") == "schema_v3"
            for record in semantic_records
        ):
            return version
        if current["label"] != version and current["label"] in captures:
            return current["label"]
        records = _legacy_records(captures, version)
        if (
            os.environ.get(ENV_OVERRIDE) is None
            and records
            and all(_legacy_record_matches_current_source(record, current) for record in records)
        ):
            return version
        # A legacy release directory in a tagless checkout is not proof that it
        # represents the current source. Preserve the old reader's fail-closed path.
        return current["label"]
    # New schema-v3 materializations always provide the semantic directory. These
    # fallbacks only keep existing Rust readers able to open older extracted trees.
    if current["label"] in captures:
        return current["label"]
    patch_pattern = re.compile(rf"{re.escape(version)}\.patch(?P<number>\d+)$")
    patches = [
        (int(match["number"]), label)
        for label in captures
        if isinstance(label, str) and (match := patch_pattern.fullmatch(label))
    ]
    return max(patches)[1] if patches else version


def validate_capture_provenance(repo_root: Path, recorded: dict) -> dict:
    if not isinstance(recorded, dict) or not isinstance(recorded.get("label"), str):
        raise ValueError("capture feed has no producer source identity; recapture it")
    current = dynamo_v2_provenance(repo_root, recorded["label"])
    _validate_provenance_identity(recorded, current)
    _validate_provenance_origin(repo_root, recorded)
    supplied = os.environ.get(ENV_OVERRIDE)
    origin = {
        key: recorded[key]
        for key in ("crate_version", "source_sha256", "git_commit")
    }
    if supplied is not None and dynamo_v2_label(repo_root, supplied) != origin["crate_version"]:
        raise ValueError("capture feed version differs from the requested capture version")
    return origin


def _validate_provenance_identity(recorded: dict, current: dict) -> None:
    if not isinstance(recorded, dict):
        raise ValueError("capture feed has no producer source identity; recapture it")
    # Squash merges can discard PR origin objects without changing captured source.
    # Consumers bind source identity; only producers require the origin objects.
    anchors = {"git_commit", "git_head_tree"}
    if {key: value for key, value in recorded.items() if key not in anchors} != {
        key: value for key, value in current.items() if key not in anchors
    }:
        raise ValueError("capture feed source identity differs from the current checkout; recapture it")
    commit = recorded.get("git_commit", "")
    tree = recorded.get("git_head_tree", "")
    if (not isinstance(commit, str) or not isinstance(tree, str)
            or not re.fullmatch(r"[0-9a-f]{40,64}", commit) or not re.fullmatch(r"[0-9a-f]{40,64}", tree)):
        raise ValueError("capture feed source identity differs: invalid Git anchor")


def _validate_provenance_origin(repo_root: Path, recorded: dict) -> None:
    commit = recorded["git_commit"]
    try:
        tree = _git(repo_root, "rev-parse", f"{commit}^{{tree}}").decode().strip()
    except subprocess.CalledProcessError as exc:
        raise ValueError(f"capture feed source identity cannot be verified: Git anchor {commit} is unavailable") from exc
    if tree != recorded["git_head_tree"]:
        raise ValueError("capture feed source identity differs: Git commit/tree mismatch")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--label", help="published version, current, or exact source-qualified label")
    parser.add_argument("--format", choices=("json", "label"), default="json")
    parser.add_argument("--select-capture", action="store_true", help="select a consumer label using capture provenance JSON on stdin")
    args = parser.parse_args()
    if args.select_capture:
        if args.label is not None or args.format != "label":
            parser.error("--select-capture requires --format label and no --label")
        print(select_capture_label(args.repo_root, json.load(sys.stdin)))
        return
    provenance = dynamo_v2_provenance(args.repo_root, args.label)
    print(json.dumps(provenance, sort_keys=True) if args.format == "json" else provenance["label"])


if __name__ == "__main__":
    main()
