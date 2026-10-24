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
from viewer_codec import decode_typed, encode_typed, typed_equal


CORPORA = sources.LEGACY_CORPORA


def convert(root: Path, history: Path, *, inventory: dict | None = None, all_versions: bool = False) -> dict:
    inventory = inventory or sources.verify(history, sources.history_pin(root, history))
    current = sources.current(root) if sources.is_current(root) else {"schema": 1, "captures": [], "lineage": []}
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
            if not corpus_root.is_dir():
                continue
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
                        available = sorted((item for item in versions if family in corpus.families_by_directory[item[2].name]), key=lambda item: item[0])
                        for selected in (available if all_versions else available[-1:]):
                            _key, version, _directory = selected
                            if mode == "stream":
                                documents, _ = resolve_stream_fixtures.resolve_docs(corpus_root, [f"{implementation}-{version}"], corpus,
                                                                                 implementations={implementation})
                            elif mode == "batch":
                                documents, _ = resolve_fixtures.resolve_docs(corpus_root, [f"{implementation}-{version}"], corpus)
                            else:
                                output = stage / "resolved" / family / implementation / version
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
                filename = f"{implementation}-{version}" if all_versions else implementation
                path = f"conformance/captures/{mode}/{family}/{filename}.yaml"
                target = root / path
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    raise ValueError(f"duplicate latest legacy selection: {path}")
                row = {"surface": mode, "mode": mode, "family": family, "implementation": implementation, "version": version}
                data = sources.selected_bytes(data, row, [])
                target.write_bytes(data)
                rows.append({**sources.identity(path, data), **row})
    current["captures"] = sorted(rows, key=lambda row: row["path"])
    return {"captures": len(rows), "bytes": sum(row["size"] for row in rows), "rows": rows}


def interpretation_delta(records: dict, frozen: dict, authored: dict) -> dict:
    """Retain a frozen interpretation without duplicating unchanged observations."""
    normalized = sources.independent_snapshot(frozen, authored)
    retained = normalized["records"]
    references = [case for case, wrapped in frozen["records"].items()
                  if wrapped["value"]["stimulus"].get("ref") == "current"]
    return {
        "schema": 1,
        "metadata": encode_typed({key: value for key, value in frozen.items() if key != "records"}),
        "set": encode_typed({case: wrapped for case, wrapped in retained.items()
                             if case not in records or not typed_equal(records[case], wrapped)}),
        "remove": sorted(records.keys() - retained.keys()),
        "current_references": references,
    }


def restore_interpretation(records: dict, delta: dict) -> dict:
    if delta["schema"] != 1:
        raise ValueError("unsupported frozen interpretation evidence")
    # Decode into independent values so evidence inspection cannot mutate a cache.
    restored = decode_typed(encode_typed(records))
    for case in delta["remove"]:
        del restored[case]
    restored.update(decode_typed(delta["set"]))
    for case in delta["current_references"]:
        stimulus = restored[case]["value"]["stimulus"]
        del stimulus["inline"]
        stimulus["ref"] = "current"
    document = decode_typed(delta["metadata"])
    document["records"] = restored
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--history-source", type=Path, required=True)
    parser.add_argument("--all-versions", action="store_true")
    args = parser.parse_args()
    print(json.dumps(convert(args.root.resolve(), args.history_source.resolve(), all_versions=args.all_versions)))


if __name__ == "__main__":
    main()
