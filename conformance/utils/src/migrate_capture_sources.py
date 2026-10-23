#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert verified legacy overlays with their existing semantic readers."""
import argparse
import json
from pathlib import Path
import shutil
import tarfile
import tempfile

import yaml
import yaml_fast  # noqa: F401
import capture_sources as sources
from fixture_corpus import load_corpus, version_key
import resolve_fixtures
import resolve_stream_fixtures
import resolve_reasoning_fixtures
from viewer_codec import typed_equal


CORPORA = sources.LEGACY_CORPORA


def convert(root: Path, history: Path) -> dict:
    pin = json.loads(sources.read(root, sources.PIN))
    inventory = sources.verify(history, pin)
    current = sources.current(root)
    rows = [row for row in current["captures"] if row["surface"] == "unified"]
    with tempfile.TemporaryDirectory(prefix="legacy-capture-conversion-") as temporary:
        stage = Path(temporary)
        originals = {}
        for row in inventory["files"]:
            if not row["path"].startswith("conformance/fixtures/") or "members" not in row:
                continue
            with tarfile.open(history / row["path"], "r:gz") as archive:
                for member in archive:
                    if not member.isfile():
                        continue
                    name = sources.relative(member.name)
                    target = stage / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    data = archive.extractfile(member).read()
                    target.write_bytes(data)
                    originals[name] = {**sources.identity(name, data), "archive": row["path"]}
        for mode, corpus_path in CORPORA.items():
            corpus_root = stage / corpus_path
            if mode == "batch-on-stream":
                families = {}
                for path in corpus_root.glob("*/*.yaml"):
                    document = yaml.safe_load(path.read_bytes())
                    for implementation, version in document.get("captured_with", {}).items():
                        families.setdefault((path.parent.name, implementation, str(version)), {})[path.name] = document
                selections = [(family, impl, version, documents) for (family, impl, version), documents in families.items()]
            else:
                corpus = load_corpus(corpus_root)
                authored = root / "conformance/inputs" / mode
                if authored.exists():
                    raise ValueError(f"authored destination already exists: {authored}")
                shutil.copytree(corpus_root / "inputs", authored)
                selections = []
                for implementation, versions in corpus.version_dirs.items():
                    families = {family for _key, _version, directory in versions
                                for family, _name in corpus.by_directory[directory.name]}
                    for family in families:
                        selected = max((item for item in versions if family in corpus.families_by_directory[item[2].name]), key=lambda item: item[0])
                        _key, version, _directory = selected
                        if mode == "stream":
                            documents, _ = resolve_stream_fixtures.resolve_docs(corpus_root, [f"{implementation}-{version}"], corpus,
                                                                             implementations={implementation})
                        elif mode == "batch":
                            documents, _ = resolve_fixtures.resolve_docs(corpus_root, [f"{implementation}-{version}"], corpus)
                        else:
                            output = stage / "resolved" / family / implementation
                            resolve_reasoning_fixtures.resolve(corpus_root, output, [f"{implementation}-{version}"])
                            documents = {(path.parent.name, path.name): yaml.safe_load(path.read_bytes())
                                         for path in output.glob("*/*.yaml")}
                        selections.append((family, implementation, version,
                                           {name: doc for (owner, name), doc in documents.items() if owner == family}))
            for family, implementation, version, documents in selections:
                dependencies = [row for name, row in originals.items()
                                if name.startswith(corpus_path + "/") and f"/{family}/" in name
                                and (mode == "batch-on-stream" or name.split("/")[2] == "inputs"
                                     or (name.split("/")[2].startswith(implementation + "-")
                                         and version_key(name.split("/")[2].split("-", 1)[1]) <= version_key(version)))]
                snapshot = {"schema_version": 1, "surface": mode, "family": family,
                            "implementation": implementation, "version": version,
                            "source_records": dependencies, "documents": documents}
                data = yaml.safe_dump(snapshot, sort_keys=False, allow_unicode=True, width=4096).encode()
                if not typed_equal(yaml.safe_load(data), snapshot):
                    raise ValueError("legacy conversion changed typed records")
                path = f"conformance/captures/{mode}/{family}/{implementation}.yaml"
                target = root / path
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    raise ValueError(f"duplicate latest legacy selection: {path}")
                target.write_bytes(data)
                rows.append({**sources.identity(path, data), "surface": mode, "mode": mode,
                             "family": family, "implementation": implementation, "version": version})
    current["captures"] = sorted(rows, key=lambda row: row["path"])
    (root / sources.CURRENT).write_bytes(sources.encoded(current))
    return {"captures": len(rows), "bytes": sum(row["size"] for row in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--history-source", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(convert(args.root.resolve(), args.history_source.resolve())))


if __name__ == "__main__":
    main()
