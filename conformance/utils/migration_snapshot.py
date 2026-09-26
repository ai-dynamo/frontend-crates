#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Snapshot resolved fixtures using the reader revision selected by PYTHONPATH.

Run this in a separate process for each reader revision. No candidate module is
allowed to stand in for the original reader when certifying a migration.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import tempfile

import yaml

import fixture_corpus
import resolve_fixtures
import resolve_reasoning_fixtures
import resolve_stream_fixtures


CORPORA = {
    "batch": "toolcalling/fixtures-batch-v1",
    "stream": "toolcalling/fixtures-stream-v1",
    "reasoning": "reasoning/fixtures-v1",
    "batch_on_stream": "toolcalling/fixtures-batch-on-stream-v1",
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def flat_docs(root):
    return {(p.parent.name, p.name): yaml.safe_load(p.read_text()) for p in sorted(root.glob("*/*.yaml"))}


def record_docs(records, label, docs):
    for (family, filename), doc in sorted(docs.items()):
        prefix = f"{label}/{family}/{filename}"
        records[prefix] = {"metadata": {k: v for k, v in doc.items() if k != "cases"}, "cases": doc.get("cases", {})}


def raw_history(root, corpus, names, records):
    """Retain capture-only cases and fields ignored by consuming readers.

    A compact checkpoint omits unchanged *whole observations*. Fold each
    implementation independently; input expectations seed only reasoning.
    """
    states = {}
    if corpus == "reasoning":
        for (family, filename), doc in flat_docs(root / "inputs").items():
            for case_id, case in doc.get("cases", {}).items():
                for impl, value in case.get("expected", {}).items():
                    states.setdefault(impl, {})[(family, filename, case_id)] = {"expected": {impl: copy.deepcopy(value)}}
    ordered = sorted(names, key=lambda name: (
        fixture_corpus.split_sel(name)[0],
        fixture_corpus.version_key(fixture_corpus.split_sel(name)[1]),
        "+" in name or ".patch" in name, name,
    ))
    for capture in ordered:
        impl, _version = fixture_corpus.split_sel(capture)
        state = states.setdefault(impl, {})
        docs = flat_docs(root / capture)
        for (family, filename), doc in docs.items():
            for case_id, case in doc.get("cases", {}).items():
                state[(family, filename, case_id)] = copy.deepcopy(case)
        for (family, filename, case_id), case in state.items():
            records[f"{corpus}/raw-history/{capture}/{family}/{filename}/{case_id}"] = copy.deepcopy(case)


def snapshot(source, fixtures):
    readers = {}
    for module in (fixture_corpus, resolve_fixtures, resolve_stream_fixtures, resolve_reasoning_fixtures):
        path = Path(module.__file__).resolve()
        path.relative_to(source.resolve())
        readers[str(path.relative_to(source))] = hashlib.sha256(path.read_bytes()).hexdigest()
    records, inventories = {}, {}
    for corpus, relative in CORPORA.items():
        root = fixtures / relative
        if not root.is_dir():
            raise ValueError(f"missing corpus: {root}")
        if corpus == "batch_on_stream":
            record_docs(records, corpus + "/snapshot", flat_docs(root))
            continue
        names = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name != "inputs")
        inventories[corpus] = names
        record_docs(records, corpus + "/inputs", flat_docs(root / "inputs"))
        raw_history(root, corpus, names, records)
        # Capture annotations must survive even when repeated observations compact.
        for name in names:
            for (family, filename), doc in flat_docs(root / name).items():
                metadata = {k: v for k, v in doc.items() if k != "cases"}
                records[f"{corpus}/metadata/{name}/{family}/{filename}"] = metadata
                if corpus in ("batch", "reasoning"):
                    for case, value in doc.get("cases", {}).items():
                        annotations = {k: v for k, v in value.items() if k != "expected"}
                        if annotations:
                            records[f"{corpus}/annotations/{name}/{family}/{filename}/{case}"] = annotations
        loaded = fixture_corpus.load_corpus(root) if corpus != "reasoning" else None
        for selection in [None, *names]:
            selected = [] if selection is None else [selection]
            if corpus == "reasoning":
                with tempfile.TemporaryDirectory(prefix="migration-reasoning-") as temporary:
                    output = Path(temporary)
                    resolve_reasoning_fixtures.resolve(root, output, selected)
                    docs = flat_docs(output)
            else:
                resolver = resolve_fixtures if corpus == "batch" else resolve_stream_fixtures
                docs = resolver.resolve_docs(root, selected, corpus=loaded)[0]
            record_docs(records, f"{corpus}/{selection or 'default'}", docs)
    return {"readers": readers, "inventory": inventories, "records": records,
            "fixture_files": {str(p.relative_to(fixtures)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted(fixtures.rglob("*.yaml"))}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--fixtures", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = snapshot(args.source, args.fixtures)
    args.output.write_text(json.dumps(result, sort_keys=True) + "\n")
    print(json.dumps({"records": len(result["records"]), "readers": result["readers"], "output_sha256": digest(result)}))


if __name__ == "__main__":
    main()
