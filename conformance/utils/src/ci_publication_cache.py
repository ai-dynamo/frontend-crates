# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Publication-input caching with checked payload identities."""
import argparse
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess

from verify_publication_transport import publication_metadata


FILES = ("native-pages.tar", "source-catalogs.tar", "conformance-publish")
PUBLICATION_DOCUMENTS = frozenset({
    "conformance/README.md",
    "conformance/utils/lib/parsers/REASONING_CASES.md",
    "conformance/utils/lib/parsers/TOOLCALLING_CASES.md",
    "conformance/utils/lib/parsers/TOOLCALLING_STREAMING_V1_CASES.md",
    "conformance/utils/lib/parsers/UNIFIED_CASES.md",
})
PARSER_REGISTRATIONS = frozenset({
    "parsers/v1/src/tool_calling/config.rs",
    "parsers/v1/src/tool_calling/parsers.rs",
})
GENERATED_CATALOGS = frozenset({
    "conformance/CONFORMANCE_v2.html",
    "conformance/CONFORMANCE_v2.json",
    "conformance/viewer-catalog.json",
    "conformance/viewer-unified-catalog.json",
    "conformance/viewer-resolved-catalog.json",
    "conformance/viewer-resolved-catalog.json.gz",
    "conformance/viewer-publication.json",
})


def file_hash(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def input_identity(path):
    if path.is_symlink():
        # Agent configuration links are tracked but can target directories
        # outside the checkout. Their link text belongs to this tree.
        return ["link", hashlib.sha256(os.readlink(path).encode()).hexdigest()]
    return ["file", file_hash(path)] if path.is_file() else ["missing", None]


def is_publication_input(name):
    if name.startswith("conformance/fixtures/"):
        return False
    if name.startswith("conformance/fixtures-unified-v2/"):
        # Historical bytes participate only in the explicit history identity.
        path = Path(name)
        return path.suffix == ".yaml" and "-" not in path.stem
    if name in {".gitattributes", "Cargo.toml", "Cargo.lock", "rust-toolchain.toml", "index.html"}:
        return True
    if name in PARSER_REGISTRATIONS:
        return True
    if (not name.startswith("conformance/") or name == "conformance/historic-baseline.json" or name in GENERATED_CATALOGS
            or fnmatch.fnmatch(name, "conformance/.CONFORMANCE_v2-*.json")
            or name.startswith(("conformance/viewer-data/", "conformance/historic/"))):
        return False
    parts = Path(name).parts
    if "tests" in parts or "node_modules" in parts or name.endswith(".test.js"):
        return False
    if name.endswith((".md", ".rst", ".txt")) and name not in PUBLICATION_DOCUMENTS:
        return False
    return True


def cache_key(root, publication_policy, *, include_history=False, history_source=None):
    if not publication_policy:
        raise ValueError("publication policy is required")
    # Hash publication inputs, including actual LFS content, rather than every
    # tracked file or the commit SHA. The generated page metadata is restamped
    # for the current commit after a cache hit.
    paths = subprocess.run(["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True).stdout
    new_sources = subprocess.run(["git", "ls-files", "--others", "--exclude-standard", "--exclude=conformance/historic/", "-z"],
                                 cwd=root, capture_output=True, check=True).stdout
    paths += new_sources
    rust = subprocess.run(["rustc", "--version"], capture_output=True, check=True).stdout.decode().strip()
    node = subprocess.run(["node", "--version"], capture_output=True, check=True).stdout.decode().strip()
    tags = subprocess.run(
        ["git", "for-each-ref", "--format=%(refname) %(objectname)", "refs/tags/dynamo-parsers-v2-v*"],
        cwd=root,
        capture_output=True,
        check=False,
    )
    if tags.returncode not in (0, 1):
        raise RuntimeError("cannot inventory release tags")
    inputs = {path: input_identity(root / path) for path in paths.decode().split("\0")
              if path and is_publication_input(path)}
    if history_source is not None and not include_history:
        raise ValueError("--history-source requires --include-history")
    history = None
    if include_history:
        history_source = history_source or root
        history = json.loads(subprocess.check_output([
            "python3", str(root / "conformance/utils/src/capture_sources.py"), "--root", str(root),
            "--verify-history", str(history_source)], text=True))
    identity = {"schema": 4, "publication_policy": publication_policy, "files": inputs, "rust": rust,
                "generation_mode": "history" if include_history else "latest", "history": history,
                "python": platform.python_version(), "node": node, "tags": tags.stdout.decode(),
                "platform": [platform.system(), platform.machine()]}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def seal(directory, identity):
    manifest = {"identity": identity, "files": {name: file_hash(directory / name) for name in FILES}}
    (directory / "cache-manifest.json").write_text(json.dumps(manifest, sort_keys=True))


def valid(directory, identity):
    try:
        manifest = json.loads((directory / "cache-manifest.json").read_bytes())
        return (manifest["identity"] == identity and set(manifest["files"]) == set(FILES)
                and all((directory / name).is_file() and not (directory / name).is_symlink()
                        and file_hash(directory / name) == manifest["files"][name] for name in FILES))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def stamp(directory, revision):
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("revision must be a full lowercase commit SHA")
    publication_path = directory / "conformance/viewer-publication.json"
    standalone = not publication_path.is_file()
    publication = publication_metadata(directory)
    if (not isinstance(publication, dict) or type(publication.get("schema")) is not int
            or publication["schema"] != 1):
        raise ValueError("unsupported publication metadata")
    if standalone:
        if (set(publication) != {"schema", "revision", "catalog_revision", "resolved_catalog_revision"}
                or not isinstance(publication.get("revision"), str)
                or re.fullmatch(r"[0-9a-f]{40}", publication.get("revision", "")) is None
                or not isinstance(publication.get("catalog_revision"), str)
                or re.fullmatch(r"[0-9a-f]{64}", publication.get("catalog_revision", "")) is None
                or publication["resolved_catalog_revision"] != publication["catalog_revision"]):
            raise ValueError("invalid standalone publication identity")
    else:
        command_text = publication.get("command")
        if not isinstance(command_text, str):
            raise ValueError("publication command is missing")
        command = shlex.split(command_text)
        old_revision = publication.get("revision")
        generated_revision = publication.get("generated_revision")
        if ("generated_revision" in publication
                and (not isinstance(generated_revision, str)
                     or re.fullmatch(r"[0-9a-f]{40}", generated_revision) is None)):
            raise ValueError("publication generated revision is malformed")
        revision_args = [index for index, argument in enumerate(command) if argument == "--revision"]
        if (not command or command[0] != "conformance-publish"
                or not isinstance(old_revision, str)
                or re.fullmatch(r"[0-9a-f]{40}", old_revision) is None
                or any(argument.startswith("--revision=") for argument in command)
                or len(revision_args) != 1 or revision_args[0] + 1 >= len(command)
                or command[revision_args[0] + 1] != (generated_revision or old_revision)):
            raise ValueError("publication command does not match its source revision")
    # Cache hits change source links, not the provenance of the generated bytes.
    # Normalize legacy metadata before its original revision is overwritten.
    original_publication = dict(publication)
    if not standalone:
        publication["generated_revision"] = generated_revision or old_revision
    publication["revision"] = revision
    encoded = json.dumps(publication, separators=(",", ":"), ensure_ascii=False)
    pattern = re.compile(r'(<script type="application/json" id="conformance-publication">)(.*?)(</script>)', re.DOTALL)
    pages = {}
    for path in (directory / "index.html", directory / "conformance/CONFORMANCE_v2.html"):
        if not path.is_file() and path.name != "index.html":
            continue
        page = path.read_text()
        matches = list(pattern.finditer(page))
        if not matches and path.name != "index.html":
            continue  # Compatibility with cached redirect-only reports.
        if len(matches) != 1:
            raise ValueError("index publication metadata does not match the publication file")
        identity = json.loads(matches[0].group(2))
        if identity == original_publication:
            updated = publication
        else:
            keys = {"schema", "revision", "generated_revision", "catalog_revision", "resolved_catalog_revision", "candidate_tree"}
            if (not isinstance(identity, dict) or set(identity) - keys
                    or not isinstance(identity.get("catalog_revision"), str)
                    or re.fullmatch(r"[a-f0-9]{64}", identity["catalog_revision"]) is None
                    or identity.get("resolved_catalog_revision") != identity["catalog_revision"]
                    or any(identity.get(key) != original_publication.get(key) for key in
                           ("schema", "revision", "catalog_revision"))
                    or any(identity[key] != original_publication.get(key) for key in ("generated_revision", "candidate_tree") if key in identity)):
                raise ValueError("index publication metadata does not match the publication file")
            updated = {**identity, "revision": revision}
        embedded_identity = json.dumps(updated, separators=(",", ":"), ensure_ascii=False).replace("<", "\\u003c")
        pages[path] = pattern.sub(lambda match: f"{match.group(1)}{embedded_identity}{match.group(3)}", page, count=1)
    if not standalone:
        publication_path.write_text(encoded)
    for path, page in pages.items():
        path.write_text(page)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("key", "seal", "check", "stamp"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--publication-policy")
    parser.add_argument("--revision")
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--identity")
    parser.add_argument("--include-history", action="store_true")
    parser.add_argument("--history-source", type=Path)
    args = parser.parse_args()
    if args.action == "key":
        if not args.publication_policy:
            parser.error("key requires --publication-policy")
        print(cache_key(args.root, args.publication_policy, include_history=args.include_history, history_source=args.history_source))
        return 0
    if args.action == "stamp":
        if args.directory is None or not args.revision:
            parser.error("stamp requires --directory and --revision")
        stamp(args.directory, args.revision)
        return 0
    if args.directory is None or not args.identity:
        parser.error("--directory and --identity are required")
    if args.action == "seal":
        seal(args.directory, args.identity)
        return 0
    return 0 if valid(args.directory, args.identity) else 1


if __name__ == "__main__":
    raise SystemExit(main())
