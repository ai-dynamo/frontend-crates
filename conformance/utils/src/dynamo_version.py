# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture identity shared by refresh, explode, and the Rust capture harnesses.

Plain versions require source equality with the release tag. Unpublished source
defaults to ``<version>+source.<sha256>``; ``current`` explicitly selects that form.
The digest covers parser crates, the split-path protocol dependency, and workspace
build inputs; conformance outputs are excluded so capture cannot change its own ID.
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

from fixture_disposition import DYNAMO_VERSION_RE, capture_layer_sort_key, historical_unified_case_key

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
    return dynamo_v2_provenance(repo_root, override)["label"]


def effective_capture_provenance(captures: dict) -> dict[str, list]:
    """Fold case ownership before validating identities, like the rendered overlays."""
    by_version = {}
    for shard in sorted(captures, key=capture_layer_sort_key):
        version, patch = capture_layer_sort_key(shard)
        layer = captures[shard]
        if isinstance(layer, list):
            # Already-folded callers have no per-case ownership to resolve.
            if patch:
                raise ValueError("capture source identity inventory needs per-case patch ownership")
            by_version[version] = dict(enumerate(layer))
            continue
        complete = layer["complete_snapshot"]
        if complete and "+source." not in version:
            raise ValueError("complete capture snapshot requires a source-qualified capture")
        if patch and "+source." in version and not complete:
            raise ValueError("source capture patch requires a complete capture snapshot")
        records = {}
        for key, provenance in layer["records"].items():
            family, case = key.split("/", 1)
            canonical = f"{family}/{historical_unified_case_key(family, case)}"
            if canonical in records and records[canonical] != provenance:
                raise ValueError(f"conflicting capture source identity aliases: {shard}/{canonical}")
            records[canonical] = provenance
        if not patch or complete:
            by_version[version] = records
        else:
            by_version.setdefault(version, {}).update(records)
    return {version: list(records.values()) for version, records in by_version.items()}


def select_capture_label(repo_root: Path, captures: dict) -> str:
    captures = effective_capture_provenance(captures)
    current = dynamo_v2_provenance(repo_root)
    label = current["label"]
    if label in captures:
        # One source fingerprint per selection, not per corpus case. Validate each
        # distinct record once, without ignoring a surviving legacy record.
        for recorded in _unique_provenance(captures[label]):
            _validate_provenance_identity(recorded, current)
        return label
    if current["kind"] == "release" or ENV_OVERRIDE in os.environ:
        return label
    version = current["crate_version"]
    tag = f"dynamo-parsers-v2-v{version}"
    if _git(repo_root, "tag", "--list", tag).strip():
        return label
    records = captures.get(version, [])
    if not records:
        return label
    # A tagless clone may consume a previously verified release, but a directory
    # name alone is not evidence. Every record must bind that release to these bytes.
    expected = {
        key: current[key]
        for key in ("crate_version", "source_sha256", "source_id", "source_paths")
    }
    expected.update(label=version, kind="release", release_tag=tag)
    verified_commits = set()
    for recorded in _unique_provenance(records):
        if not isinstance(recorded, dict) or any(recorded.get(k) != v for k, v in expected.items()):
            return label
        commit = recorded.get("release_commit")
        if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40,64}", commit):
            return label
        try:
            _validate_provenance_identity(recorded, {**current, **expected, "release_commit": commit})
        except ValueError:
            return label
        if commit in verified_commits:
            continue
        try:
            fingerprint = source_fingerprint(repo_root, commit)
        except subprocess.CalledProcessError:
            return label
        if fingerprint != current["source_sha256"]:
            return label
        verified_commits.add(commit)
    return version


def _unique_provenance(records: list):
    return {json.dumps(record, sort_keys=True): record for record in records}.values()


def validate_capture_provenance(repo_root: Path, recorded: dict) -> dict:
    if not isinstance(recorded, dict) or not isinstance(recorded.get("label"), str):
        raise ValueError("capture feed has no producer source identity; recapture it")
    current = dynamo_v2_provenance(repo_root, recorded["label"])
    _validate_provenance_identity(recorded, current)
    _validate_provenance_origin(repo_root, recorded)
    supplied = os.environ.get(ENV_OVERRIDE)
    if supplied is not None and dynamo_v2_label(repo_root, supplied) != recorded["label"]:
        raise ValueError("capture feed label differs from the requested capture label")
    return recorded


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
